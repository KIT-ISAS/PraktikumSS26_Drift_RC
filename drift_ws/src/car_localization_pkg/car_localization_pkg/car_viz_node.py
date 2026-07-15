"""
Republish a car pose as a 3D mesh marker for Foxglove.

Extracted from the libsurvive node so visualization is source-agnostic and can be
instantiated once per car: point it at ``/car/pose`` for the real car and at
``/sim/car/pose`` for a simulated one, each with its own colour, and both render
in the shared ``world`` frame.
"""

import math

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker

from shared_pkg import Topics
from vehicle_simulation.transforms import quaternion_from_yaw, quaternion_multiply


class CarVizNode(Node):
    """Subscribes to a pose topic and publishes a coloured car mesh marker."""

    def __init__(self) -> None:
        super().__init__("car_viz_node")
        self.declare_parameter("pose_topic", Topics.CAR_POSE)
        # Absolute canonical topic by default: the marker only depends on the
        # source-agnostic /car/pose, so it must not inherit the (per-source) node
        # namespace. Override only if you intentionally want a per-car namespace.
        self.declare_parameter("marker_topic", Topics.CAR_VISUALIZATION)
        self.declare_parameter("mesh_resource", "package://shared_pkg/meshes/porsche.glb")
        self.declare_parameter("use_embedded_materials", True)
        self.declare_parameter("color_rgba", [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("mesh_scale", 7.0)
        # Fixed yaw correction (deg, +CCW/left about z) aligning the mesh's native
        # forward axis with the car's heading (+x). The porsche.glb points 90 deg
        # to the right of +x, so it needs +90.
        self.declare_parameter("mesh_yaw_offset_deg", 90.0)
        self.declare_parameter("namespace", "car")

        self._mesh = self.get_parameter("mesh_resource").value
        self._embedded = bool(self.get_parameter("use_embedded_materials").value)
        self._color = [float(c) for c in self.get_parameter("color_rgba").value]
        self._scale = float(self.get_parameter("mesh_scale").value)
        self._yaw_offset_q = quaternion_from_yaw(
            math.radians(float(self.get_parameter("mesh_yaw_offset_deg").value))
        )
        self._ns = self.get_parameter("namespace").value

        # Latched so a late-joining Foxglove still gets the marker.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._marker_pub = self.create_publisher(
            Marker, self.get_parameter("marker_topic").value, latched
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("pose_topic").value, self._on_pose, 10
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        marker = Marker()
        marker.header = msg.header
        marker.ns = self._ns
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = self._mesh
        # color_rgba only tints the mesh when embedded materials are OFF; with
        # them ON (default) the glb's own materials win and color is ignored.
        marker.mesh_use_embedded_materials = self._embedded
        marker.pose = msg.pose
        # Apply the fixed mesh-alignment yaw in the car's body frame (post-multiply).
        q = msg.pose.orientation
        rq = quaternion_multiply((q.x, q.y, q.z, q.w), self._yaw_offset_q)
        marker.pose.orientation.x = float(rq[0])
        marker.pose.orientation.y = float(rq[1])
        marker.pose.orientation.z = float(rq[2])
        marker.pose.orientation.w = float(rq[3])
        marker.scale.x = marker.scale.y = marker.scale.z = self._scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = self._color
        self._marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
