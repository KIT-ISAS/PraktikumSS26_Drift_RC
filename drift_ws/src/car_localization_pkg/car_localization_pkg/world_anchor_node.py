"""
Anchor the shared ``world`` frame to the lighthouse constellation.

Source-agnostic: whichever tracking source is running (libsurvive or
steamvr) only has to latch static TFs ``<tracking_frame> -> <serial>`` for
the base stations it sees. This node polls those frames, fits the planar
``world -> <tracking_frame>`` transform against the reference positions in
``config/lighthouses_world.yaml`` (see ``lighthouse_calibration``), latches
it on ``/tf_static``, and publishes the lighthouse poses in the world frame
on ``/map/lighthouses``.

The lighthouses thereby *define* the playground: move the constellation and
the world frame — and the track defined in it — moves along. A lighthouse
that moved relative to the others shows up as a large fit residual and is
warned about; the fix is re-recording the reference file (see
steamvr_pkg's ``steamvr_calibrate --record-reference``).

Serial matching: SteamVR reports ``LHB-XXXXXXXX``; libsurvive frames carry
the same serial (its numeric ID is the identical value in hex), so both
sources match one reference file. Decimal or bare-hex frame names are
normalized just in case.

``world_calibration: manual`` skips the fit and latches the fixed ``world_*``
parameters instead (frozen frame for strictly comparable recordings).
"""

import math
import os
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener
import yaml

from drift_core.transforms import quaternion_from_yaw
from shared_pkg import Frames, Topics

from .lighthouse_calibration import (
    load_lighthouse_reference,
    normalize_serial,
    solve_world_from_lighthouses,
)


class WorldAnchorNode(Node):
    """Fits and latches ``world -> <tracking_frame>`` from the lighthouses."""

    def __init__(self) -> None:
        super().__init__("world_anchor_node")
        self.declare_parameter("tracking_frame", "")
        self.declare_parameter("world_frame", Frames.WORLD)
        # "" = the lighthouses_world.yaml shipped with this package.
        self.declare_parameter("reference_file", "")
        self.declare_parameter("world_calibration", "auto")
        self.declare_parameter("min_lighthouses", 3)
        self.declare_parameter("warn_threshold_m", 0.15)
        self.declare_parameter("poll_rate", 2.0)
        # Grace period to see the FULL reference constellation before the first
        # fit: lighthouses warm up at different times, and fitting early with a
        # subset means a re-fit (= world-frame jump) when the rest appear.
        self.declare_parameter("settle_timeout_s", 10.0)
        # Manual mode only: fixed transform instead of the fit.
        self.declare_parameter("world_x", 0.0)
        self.declare_parameter("world_y", 0.0)
        self.declare_parameter("world_z", 0.0)
        self.declare_parameter("world_yaw_deg", 0.0)

        self._tracking = str(self.get_parameter("tracking_frame").value)
        if not self._tracking:
            raise ValueError("tracking_frame parameter is required (e.g. 'steamvr_world')")
        self._world = str(self.get_parameter("world_frame").value)
        self._reference = self._load_reference()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_static = StaticTransformBroadcaster(self)
        self._lighthouse_pub = self.create_publisher(
            PoseArray,
            Topics.LIGHTHOUSE_POSES,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self._cal_yaw = 0.0
        self._cal_t = np.zeros(3)
        self._calibrated = False
        self._fitted_serials: set[str] = set()
        self._warned_serials: set[str] = set()
        self._started = time.monotonic()

        if str(self.get_parameter("world_calibration").value) == "manual":
            self._cal_yaw = math.radians(float(self.get_parameter("world_yaw_deg").value))
            self._cal_t = np.array(
                [
                    float(self.get_parameter("world_x").value),
                    float(self.get_parameter("world_y").value),
                    float(self.get_parameter("world_z").value),
                ]
            )
            self._latch_world_transform()
            self._calibrated = True
            self.get_logger().info(f"Manual world -> {self._tracking}: fixed parameters latched.")

        poll_rate = float(self.get_parameter("poll_rate").value)
        self.create_timer(1.0 / poll_rate, self._poll)

    def _load_reference(self) -> dict[str, np.ndarray]:
        path = str(self.get_parameter("reference_file").value)
        if not path:
            path = os.path.join(
                get_package_share_directory("car_localization_pkg"),
                "config",
                "lighthouses_world.yaml",
            )
        return load_lighthouse_reference(path)

    def _collect_lighthouses(self) -> dict[str, np.ndarray]:
        """Read all lighthouse frames under the tracking frame from TF."""
        frames = yaml.safe_load(self._tf_buffer.all_frames_as_yaml()) or {}
        if not isinstance(frames, dict):
            return {}
        measured = {}
        for frame, info in frames.items():
            serial = normalize_serial(frame)
            if serial is None or info.get("parent") != self._tracking:
                continue
            try:
                tf = self._tf_buffer.lookup_transform(self._tracking, frame, rclpy.time.Time())
            except TransformException:
                continue
            t = tf.transform.translation
            measured[serial] = np.array([t.x, t.y, t.z])
        return measured

    def _poll(self) -> None:
        measured = self._collect_lighthouses()
        self._warn_unknown(measured)
        known = {s: p for s, p in measured.items() if s in self._reference}

        auto = str(self.get_parameter("world_calibration").value) == "auto"
        if auto and set(known) != self._fitted_serials:
            # < 2 correspondences would make the solver raise inside this timer.
            min_needed = max(2, int(self.get_parameter("min_lighthouses").value))
            if len(known) < min_needed:
                self.get_logger().warning(
                    f"Waiting for lighthouses to anchor the world frame: "
                    f"{len(known)}/{min_needed} "
                    f"reference serials visible under '{self._tracking}'.",
                    throttle_duration_sec=10.0,
                )
            elif not self._settling(known):
                self._fit(known)
        if self._calibrated:
            self._publish_lighthouses(measured)

    def _settling(self, known: dict[str, np.ndarray]) -> bool:
        """Before the first fit, briefly wait for the full reference constellation."""
        if self._calibrated or set(known) == set(self._reference):
            return False
        timeout = float(self.get_parameter("settle_timeout_s").value)
        if time.monotonic() - self._started >= timeout:
            return False
        self.get_logger().info(
            f"{len(known)}/{len(self._reference)} reference lighthouses visible; "
            f"waiting up to {timeout:.0f} s for the full constellation before "
            "the first fit.",
            throttle_duration_sec=5.0,
        )
        return True

    def _fit(self, known: dict[str, np.ndarray]) -> None:
        result = solve_world_from_lighthouses(known, self._reference)
        new_yaw = math.radians(result.world_yaw_deg)
        new_t = np.array([result.world_x, result.world_y, result.world_z])
        if self._calibrated:
            # A mid-session re-fit (lighthouse set changed) moves the latched
            # frame — every downstream pose jumps. Make that diagnosable.
            delta_yaw = math.degrees(abs(math.remainder(new_yaw - self._cal_yaw, 2.0 * math.pi)))
            delta_t = float(np.linalg.norm(new_t - self._cal_t))
            self.get_logger().warning(
                f"Re-anchoring world -> {self._tracking}: lighthouse set changed to "
                f"{sorted(known)}; frame moves by {delta_t * 100:.1f} cm / "
                f"{delta_yaw:.2f} deg — downstream poses jump accordingly."
            )
        self._cal_yaw = new_yaw
        self._cal_t = new_t
        self._latch_world_transform()
        self._calibrated = True
        self._fitted_serials = set(known)
        residuals = ", ".join(
            f"{serial}: {residual * 100:.1f} cm"
            for serial, residual in sorted(result.residuals_m.items())
        )
        self.get_logger().info(
            f"world -> {self._tracking} anchored to {len(known)} lighthouses "
            f"(yaw {result.world_yaw_deg:.2f} deg; residuals: {residuals})"
        )
        threshold = float(self.get_parameter("warn_threshold_m").value)
        for serial, residual in result.residuals_m.items():
            if residual > threshold:
                self.get_logger().warning(
                    f"Lighthouse {serial} has a {residual:.2f} m fit residual (threshold "
                    f"{threshold:.2f} m): the constellation no longer matches the "
                    "reference file — a lighthouse moved relative to the others. "
                    "Re-record it: ros2 run steamvr_pkg steamvr_calibrate "
                    "--record-reference (tracker at the floor center)."
                )

    def _warn_unknown(self, measured: dict[str, np.ndarray]) -> None:
        for serial in set(measured) - set(self._reference) - self._warned_serials:
            self._warned_serials.add(serial)
            self.get_logger().warning(
                f"Lighthouse {serial} is not in the reference file — new/replaced base "
                "station? Update config/lighthouses_world.yaml."
            )

    def _latch_world_transform(self) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._world
        tf.child_frame_id = self._tracking
        tf.transform.translation.x = float(self._cal_t[0])
        tf.transform.translation.y = float(self._cal_t[1])
        tf.transform.translation.z = float(self._cal_t[2])
        qx, qy, qz, qw = quaternion_from_yaw(self._cal_yaw)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_static.sendTransform([tf])

    def _publish_lighthouses(self, measured: dict[str, np.ndarray]) -> None:
        c, s = math.cos(self._cal_yaw), math.sin(self._cal_yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        array = PoseArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.header.frame_id = self._world
        for serial in sorted(measured):
            p = rot @ measured[serial] + self._cal_t
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, p)
            pose.orientation.w = 1.0
            array.poses.append(pose)
        self._lighthouse_pub.publish(array)


def main(args=None) -> None:
    """Run the world anchor node."""
    rclpy.init(args=args)
    node = WorldAnchorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
