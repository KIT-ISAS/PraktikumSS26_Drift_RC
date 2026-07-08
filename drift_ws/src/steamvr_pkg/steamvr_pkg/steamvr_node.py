"""
SteamVR tracking source node: turns bridge datagrams into the tracking contract.

Receives the host bridge's UDP samples (see ``lib/protocol.py`` and
``steamvr_bridge/`` at the repo root), converts them from OpenVR's y-up
tracking space (``lib/conversion.py``), and publishes what every tracking
source owes downstream (see car_localization_pkg):

  * TF ``steamvr_world -> tracker_link``   (dynamic, raw tracker pose)
  * TF ``world -> steamvr_world``          (static, from calibration params)
  * ``imu``   sensor_msgs/Imu              (pseudo-IMU, ``frame_id=tracker_link``)
  * ``pose``  geometry_msgs/PoseStamped    (debug, world frame)
  * ``/map/lighthouses``  geometry_msgs/PoseArray (world frame, calibration aid)

OpenVR has no raw accelerometer API, so the IMU is synthesized: angular
velocity comes from OpenVR directly, linear acceleration is finite-differenced
from OpenVR's linear velocity (noisy — a compatibility signal, not RL-grade).
It is stamped with the **tracker frame**, so car_state_node's frame_id-based
TF lookup rotates it into base_link without any libsurvive-style imu_link hop.

Messages are stamped with ROS time at datagram receipt; the UDP hop plus the
bridge's poll loop add ~ms-level latency that is currently uncorrected (the
protocol carries the host monotonic sample time for a smarter scheme later).
"""

import math
import socket
import threading

from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from shared_pkg import Frames, Topics
from vehicle_simulation.transforms import (
    quaternion_from_yaw,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)

from .lib.conversion import pose_from_openvr, vector_from_openvr, vector_to_body
from .lib.protocol import (
    DEFAULT_PORT,
    DEVICE_CLASS_LIGHTHOUSE,
    DEVICE_CLASS_TRACKER,
    ProtocolError,
    TrackerSample,
    decode_sample,
)

GRAVITY = 9.81


class SteamVrNode(Node):
    """Publishes the SteamVR tracker as a standard tracking source."""

    def __init__(self) -> None:
        super().__init__("steamvr_node")
        self.declare_parameter("bind_address", "127.0.0.1")
        self.declare_parameter("udp_port", DEFAULT_PORT)
        # Serial of the car's tracker. Empty = lock onto the first tracker seen
        # (mirrors holodeck's tracker_serial convention).
        self.declare_parameter("tracker_serial", "")
        self.declare_parameter("world_frame", Frames.WORLD)
        self.declare_parameter("tracking_frame", Frames.STEAMVR_WORLD)
        self.declare_parameter("tracker_frame", Frames.TRACKER_LINK)
        # Calibration of world -> steamvr_world (the SteamVR analogue of the
        # constants hardcoded in holodeck's component.cpp). Tune per the
        # procedure in the package README.
        self.declare_parameter("world_x", 0.0)
        self.declare_parameter("world_y", 0.0)
        self.declare_parameter("world_z", 0.0)
        self.declare_parameter("world_yaw_deg", 0.0)
        # Add +g up to the pseudo-IMU so it reads specific force like a real
        # accelerometer (car_state_node consumers expect gravity in the signal).
        self.declare_parameter("include_gravity", True)
        self.declare_parameter("publish_lighthouses", True)
        self.declare_parameter("stale_timeout_s", 1.0)

        self._tracker_serial = str(self.get_parameter("tracker_serial").value)
        self._world = str(self.get_parameter("world_frame").value)
        self._tracking = str(self.get_parameter("tracking_frame").value)
        self._tracker = str(self.get_parameter("tracker_frame").value)
        self._include_gravity = bool(self.get_parameter("include_gravity").value)
        self._publish_lighthouses = bool(self.get_parameter("publish_lighthouses").value)

        # world -> steamvr_world from the calibration parameters.
        self._cal_yaw_q = quaternion_from_yaw(
            math.radians(float(self.get_parameter("world_yaw_deg").value))
        )
        self._cal_t = np.array(
            [
                float(self.get_parameter("world_x").value),
                float(self.get_parameter("world_y").value),
                float(self.get_parameter("world_z").value),
            ]
        )

        self._tf = TransformBroadcaster(self)
        self._tf_static = StaticTransformBroadcaster(self)
        self._tf_static.sendTransform(
            [self._transform(self._world, self._tracking, self._cal_t, self._cal_yaw_q)]
        )

        self._imu_pub = self.create_publisher(Imu, "imu", 10)
        self._pose_pub = self.create_publisher(PoseStamped, "pose", 10)
        self._lighthouse_pub = self.create_publisher(PoseArray, Topics.LIGHTHOUSE_POSES, 10)

        self._lighthouses: dict[str, Pose] = {}
        self._prev_velocity: np.ndarray | None = None
        self._prev_t_mono: float | None = None
        self._last_rx_mono: float | None = None

        stale_timeout = float(self.get_parameter("stale_timeout_s").value)
        self.create_timer(stale_timeout, self._check_stale)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.0)
        bind_address = str(self.get_parameter("bind_address").value)
        self._sock.bind((bind_address, int(self.get_parameter("udp_port").value)))
        self._running = True
        # rclpy publishers are thread-safe, so samples are published straight
        # from the receive thread (no queue hop, lower latency).
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._rx_thread.start()
        self.get_logger().info(
            f"Listening for bridge datagrams on udp://{self._sock.getsockname()[0]}"
            f":{self._sock.getsockname()[1]}"
        )

    def destroy_node(self) -> None:
        """Stop the receive thread before tearing the node down."""
        self._running = False
        self._sock.close()
        self._rx_thread.join(timeout=2.0)
        super().destroy_node()

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed on shutdown
            try:
                sample = decode_sample(data)
            except ProtocolError as e:
                self.get_logger().warning(f"Dropping datagram: {e}", throttle_duration_sec=5.0)
                continue
            if sample.device_class == DEVICE_CLASS_TRACKER:
                self._on_tracker(sample)
            elif sample.device_class == DEVICE_CLASS_LIGHTHOUSE and self._publish_lighthouses:
                self._on_lighthouse(sample)

    def _on_tracker(self, sample: TrackerSample) -> None:
        if not self._tracker_serial:
            self._tracker_serial = sample.serial
            self.get_logger().info(f"Locked onto tracker {sample.serial}")
        if sample.serial != self._tracker_serial or not sample.pose_valid:
            return
        self._last_rx_mono = self.get_clock().now().nanoseconds * 1e-9

        position, quat = pose_from_openvr(sample.matrix)
        stamp = self.get_clock().now().to_msg()
        tf = self._transform(self._tracking, self._tracker, position, quat)
        tf.header.stamp = stamp
        self._tf.sendTransform(tf)

        self._publish_imu(sample, quat, stamp)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self._world
        self._fill_pose(pose.pose, *self._to_world(position, quat))
        self._pose_pub.publish(pose)

    def _publish_imu(self, sample: TrackerSample, quat, stamp) -> None:
        # OpenVR reports velocities in the tracking frame (its angular-velocity
        # frame is under-documented; if bring-up shows it is body-frame, relabel
        # with conversion._VR_BODY_TO_TRACKER instead of converting+rotating).
        angular_world = vector_from_openvr(sample.angular_velocity)
        velocity_world = vector_from_openvr(sample.velocity)

        acceleration_world = None
        if self._prev_t_mono is not None and sample.t_mono > self._prev_t_mono:
            dt = sample.t_mono - self._prev_t_mono
            acceleration_world = (velocity_world - self._prev_velocity) / dt
            if self._include_gravity:
                acceleration_world = acceleration_world + np.array([0.0, 0.0, GRAVITY])
        self._prev_velocity = velocity_world
        self._prev_t_mono = sample.t_mono
        if acceleration_world is None:
            return  # need two samples for the finite difference

        imu = Imu()
        imu.header.stamp = stamp
        # Stamped with the tracker frame: car_state_node rotates the signal
        # into base_link via TF from frame_id, so no imu_link hop is needed.
        imu.header.frame_id = self._tracker
        wb = vector_to_body(angular_world, quat)
        ab = vector_to_body(acceleration_world, quat)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, wb)
        (
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ) = map(float, ab)
        imu.orientation_covariance[0] = -1.0  # orientation not provided
        self._imu_pub.publish(imu)

    def _on_lighthouse(self, sample: TrackerSample) -> None:
        if not sample.pose_valid:
            return
        position, quat = pose_from_openvr(sample.matrix)
        if sample.serial not in self._lighthouses:
            self._tf_static.sendTransform(
                [self._transform(self._tracking, sample.serial, position, quat)]
            )
            self.get_logger().info(f"Lighthouse {sample.serial} at {np.round(position, 3)}")
        pose = Pose()
        self._fill_pose(pose, *self._to_world(position, quat))
        self._lighthouses[sample.serial] = pose

        array = PoseArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.header.frame_id = self._world
        array.poses = [self._lighthouses[serial] for serial in sorted(self._lighthouses)]
        self._lighthouse_pub.publish(array)

    def _to_world(self, position, quat) -> tuple[np.ndarray, np.ndarray]:
        """Compose the calibration transform onto a steamvr_world pose."""
        position_world = rotate_vector_by_quaternion(self._cal_yaw_q, position) + self._cal_t
        quat_world = quaternion_multiply(self._cal_yaw_q, quat)
        return position_world, quat_world

    def _check_stale(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        timeout = float(self.get_parameter("stale_timeout_s").value)
        if self._last_rx_mono is None or now - self._last_rx_mono > timeout:
            self.get_logger().warning(
                "No tracker samples from the SteamVR bridge — is steamvr_bridge.py "
                "running on the host and SteamVR tracking?",
                throttle_duration_sec=10.0,
            )

    @staticmethod
    def _transform(parent: str, child: str, translation, quaternion) -> TransformStamped:
        tf = TransformStamped()
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(translation[0])
        tf.transform.translation.y = float(translation[1])
        tf.transform.translation.z = float(translation[2])
        tf.transform.rotation.x = float(quaternion[0])
        tf.transform.rotation.y = float(quaternion[1])
        tf.transform.rotation.z = float(quaternion[2])
        tf.transform.rotation.w = float(quaternion[3])
        return tf

    @staticmethod
    def _fill_pose(pose: Pose, position, quaternion) -> None:
        pose.position.x, pose.position.y, pose.position.z = map(float, position)
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])


def main(args=None) -> None:
    """Run the SteamVR tracking source node."""
    rclpy.init(args=args)
    node = SteamVrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
