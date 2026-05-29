"""Track boundary publisher node."""

from math import atan2

from geometry_msgs.msg import Point, PoseArray
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker

from shared_pkg import Topics


class TrackPublisherNode(Node):
    """Subscribes to lighthouse poses and publishes a rectangular track boundary."""

    def __init__(self):
        """Initialize the node, subscriber, publisher, and parameters."""
        super().__init__("track_publisher_node")

        self._lighthouse_positions = []
        self._frame_id = ""

        # Match the transient_local QoS of the lighthouse pose publisher so we
        # receive the latched poses even when subscribing after they were sent.
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._sub = self.create_subscription(
            PoseArray, Topics.LIGHTHOUSE_POSES, self._lighthouse_poses_callback, qos
        )
        self._pub = self.create_publisher(Marker, Topics.TRACK_BOUNDARY, 10)

        self.get_logger().info("Track publisher node started")

    def _lighthouse_poses_callback(self, msg: PoseArray):
        """Process incoming lighthouse poses and update the track boundary."""
        positions = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        if positions == self._lighthouse_positions:
            return

        self._lighthouse_positions = positions
        self._frame_id = msg.header.frame_id
        for x, y, z in positions:
            self.get_logger().info(f"Lighthouse at ({x:.3f}, {y:.3f}, {z:.3f})")

        if len(positions) >= 2:
            self._publish_boundary()

    def _publish_boundary(self):
        """Publish a LINE_STRIP marker connecting lighthouse positions as boundary corners."""
        positions = self._lighthouse_positions
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)

        # Sort corners by angle from centroid so the polygon is non-self-intersecting.
        corners = sorted(positions, key=lambda p: atan2(p[1] - cy, p[0] - cx))

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._frame_id
        marker.ns = "track_boundary"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        for pos in corners + [corners[0]]:  # close the loop
            p = Point()
            p.x = pos[0]
            p.y = pos[1]
            p.z = 0.0
            marker.points.append(p)

        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.0
        marker.color.a = 1.0

        self._pub.publish(marker)
        corner_str = ", ".join(f"({p[0]:.2f}, {p[1]:.2f})" for p in corners)
        self.get_logger().info(f"Published track boundary with corners: {corner_str}")


def main(args=None):
    """Entry point for the track publisher node."""
    rclpy.init(args=args)
    node = TrackPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
