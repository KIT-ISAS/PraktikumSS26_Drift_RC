"""
Central, source-agnostic owner of the canonical car state.

Whatever tracking system is running only has to publish the dynamic transform
``world -> tracker_link``; the static mount (``mount_tf_node``) completes the
chain to ``base_link`` (the CoG, ISO 8855). This node reads ``world ->
base_link`` from tf2, finite-differences the body-frame velocities, and rotates
the raw tracker IMU into ``base_link`` — publishing everything **already
transformed** so no downstream node re-derives coordinate maths:

  * ``/car/pose``  geometry_msgs/PoseStamped   (pose only, world frame)
  * ``/car/odom``  nav_msgs/Odometry           (pose in world + body-frame twist)
  * ``/car/imu``   sensor_msgs/Imu             (raw IMU rotated into base_link)

The live twist is a simple causal finite difference; offline training can still
recompute smoothed velocities from the recorded poses if it wants higher
fidelity. IMU is rotation-only for now (no CoG lever-arm correction).
"""

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformException, TransformListener

from drift_core.transforms import (
    rotate_vector_by_quaternion,
    world_to_body_velocity,
    wrap_angle,
    yaw_from_quaternion,
)
from shared_pkg import Frames, Topics


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class CarStateNode(Node):
    """Publishes the canonical pose / odometry / IMU for one car."""

    def __init__(self) -> None:
        super().__init__("car_state_node")
        self.declare_parameter("world_frame", Frames.WORLD)
        self.declare_parameter("base_frame", Frames.BASE_LINK)
        self.declare_parameter("raw_imu_topic", "imu")
        self.declare_parameter("update_rate", 100.0)

        self._world = self.get_parameter("world_frame").value
        self._base = self.get_parameter("base_frame").value
        rate = float(self.get_parameter("update_rate").value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pose_pub = self.create_publisher(PoseStamped, Topics.CAR_POSE, 10)
        self._odom_pub = self.create_publisher(Odometry, Topics.CAR_ODOM, 10)
        self._imu_pub = self.create_publisher(Imu, Topics.CAR_IMU, 10)
        self.create_subscription(Imu, self.get_parameter("raw_imu_topic").value, self._on_imu, 10)

        self._last_t: float | None = None
        self._last_xyh: tuple[float, float, float] | None = None
        self.create_timer(1.0 / rate, self._on_timer)

    def _on_timer(self) -> None:
        try:
            tf = self._tf_buffer.lookup_transform(self._world, self._base, rclpy.time.Time())
        except TransformException:
            return  # frames not available yet

        t = _stamp_to_sec(tf.header.stamp)
        tr = tf.transform.translation
        q = tf.transform.rotation
        heading = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        x, y = tr.x, tr.y

        pose = PoseStamped()
        pose.header.stamp = tf.header.stamp
        pose.header.frame_id = self._world
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = tr.z
        pose.pose.orientation = q
        self._pose_pub.publish(pose)

        odom = Odometry()
        odom.header = pose.header
        odom.child_frame_id = self._base
        odom.pose.pose = pose.pose
        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            px, py, ph = self._last_xyh
            vx_b, vy_b = world_to_body_velocity((x - px) / dt, (y - py) / dt, heading)
            odom.twist.twist.linear.x = vx_b
            odom.twist.twist.linear.y = vy_b
            odom.twist.twist.angular.z = wrap_angle(heading - ph) / dt
        self._odom_pub.publish(odom)

        self._last_t = t
        self._last_xyh = (x, y, heading)

    def _on_imu(self, msg: Imu) -> None:
        # Rotate the raw IMU (in its own frame) into base_link. Rotation only:
        # the small CoG lever-arm correction is left as a future refinement.
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base, msg.header.frame_id, rclpy.time.Time()
            )
        except TransformException:
            return
        r = tf.transform.rotation
        quat = (r.x, r.y, r.z, r.w)

        out = Imu()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._base
        w = msg.angular_velocity
        a = msg.linear_acceleration
        wb = rotate_vector_by_quaternion(quat, (w.x, w.y, w.z))
        ab = rotate_vector_by_quaternion(quat, (a.x, a.y, a.z))
        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = map(float, wb)
        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = map(
            float, ab
        )
        out.orientation_covariance[0] = -1.0  # orientation not provided
        self._imu_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
