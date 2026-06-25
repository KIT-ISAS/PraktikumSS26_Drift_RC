"""
Publish the static tracker->body and tracker->IMU transforms for one car.

Source-agnostic: it only encodes how the tracker is bolted to the car (from the
measured car parameters), independent of which tracking system reports
``world -> tracker_link``. Run one instance per car; namespace + frame
parameters keep a real and a simulated car from colliding (see the package
README).
"""

import math

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from shared_pkg import Frames
from vehicle_simulation.car_parameters import load_car_parameters

from .mount import tracker_to_base, tracker_to_imu


def _static_tf(node: Node, parent: str, child: str, translation, quaternion) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = translation[0]
    tf.transform.translation.y = translation[1]
    tf.transform.translation.z = translation[2]
    tf.transform.rotation.x = quaternion[0]
    tf.transform.rotation.y = quaternion[1]
    tf.transform.rotation.z = quaternion[2]
    tf.transform.rotation.w = quaternion[3]
    return tf


class MountTfNode(Node):
    """Latches the car's static mount transforms onto /tf_static."""

    def __init__(self) -> None:
        super().__init__("mount_tf_node")
        self.declare_parameter("car", "drift_rc")
        self.declare_parameter("tracker_frame", Frames.TRACKER_LINK)
        self.declare_parameter("base_frame", Frames.BASE_LINK)
        self.declare_parameter("imu_frame", Frames.IMU_LINK)
        # Empirical mount knobs — confirm against a known pose at bring-up.
        self.declare_parameter("tracker_yaw_offset_deg", 90.0)
        self.declare_parameter("tracker_upside_down", True)

        car_name = self.get_parameter("car").value
        tracker = self.get_parameter("tracker_frame").value
        base = self.get_parameter("base_frame").value
        imu = self.get_parameter("imu_frame").value
        yaw = math.radians(float(self.get_parameter("tracker_yaw_offset_deg").value))
        upside_down = bool(self.get_parameter("tracker_upside_down").value)

        car = load_car_parameters(car_name)
        base_t, base_q = tracker_to_base(car, yaw_offset_rad=yaw, upside_down=upside_down)
        imu_t, imu_q = tracker_to_imu()

        self._broadcaster = StaticTransformBroadcaster(self)
        self._broadcaster.sendTransform(
            [
                _static_tf(self, tracker, base, base_t, base_q),
                _static_tf(self, tracker, imu, imu_t, imu_q),
            ]
        )
        self.get_logger().info(
            f"Published static {tracker}->{base} and {tracker}->{imu} for car '{car_name}'"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MountTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
