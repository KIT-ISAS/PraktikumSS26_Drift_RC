"""Track boundary publisher node."""

from math import atan2

from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

TRANSIENT_LOCAL_QOS = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class TrackPublisherNode(Node):
    """Subscribes to /tf_static lighthouse poses and publishes a rectangular track boundary."""

    def __init__(self):
        """Initialize the node, subscriber, publisher, and parameters."""
        super().__init__("track_publisher_node")
        self.declare_parameter("tracking_frame", "libsurvive_world")
        self._tracking_frame = (
            self.get_parameter("tracking_frame").get_parameter_value().string_value
        )

        self._lighthouse_positions = {}

        self._sub = self.create_subscription(
            TFMessage, "/tf_static", self._tf_static_callback, TRANSIENT_LOCAL_QOS
        )
        self._pub = self.create_publisher(Marker, "/track/boundary", TRANSIENT_LOCAL_QOS)

        self.get_logger().info(
            f"Track publisher node started, tracking frame: {self._tracking_frame}"
        )

    def _tf_static_callback(self, msg: TFMessage):
        """Process incoming static transforms and update the track boundary."""
        updated = False
        for transform in msg.transforms:
            if transform.header.frame_id != self._tracking_frame:
                continue
            frame = transform.child_frame_id
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            if self._lighthouse_positions.get(frame) != (x, y, z):
                self._lighthouse_positions[frame] = (x, y, z)
                updated = True
                self.get_logger().info(f"Lighthouse {frame} at ({x:.3f}, {y:.3f}, {z:.3f})")

        if updated and len(self._lighthouse_positions) >= 2:
            self._publish_boundary()

    def _publish_boundary(self):
        """Publish a LINE_STRIP marker connecting lighthouse positions as boundary corners."""
        positions = list(self._lighthouse_positions.values())
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)

        # Sort corners by angle from centroid so the polygon is non-self-intersecting.
        corners = sorted(positions, key=lambda p: atan2(p[1] - cy, p[0] - cx))

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._tracking_frame
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
