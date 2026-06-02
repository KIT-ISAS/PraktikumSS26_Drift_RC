"""Track boundary publisher node."""

from math import atan2, cos, sin
import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose, PoseArray
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from shared_interfaces.msg import Track
from shared_pkg import Topics
from track_pkg.bezier_track import BezierTrack


class TrackPublisherNode(Node):
    """Publishes pre-loaded track centerline and rectangular lighthouse boundary."""

    def __init__(self):
        """Initialize the node, subscriber, publishers, and parameters."""
        super().__init__("track_publisher_node")

        self._lighthouse_positions = []
        self._frame_id = ""

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._sub = self.create_subscription(
            PoseArray, Topics.LIGHTHOUSE_POSES, self._lighthouse_poses_callback, latched_qos
        )
        self._boundary_pub = self.create_publisher(Marker, Topics.TRACK_BOUNDARY, 10)
        self._centerline_pub = self.create_publisher(Track, Topics.TRACK_CENTERLINE, latched_qos)
        self._visualization_pub = self.create_publisher(
            MarkerArray, Topics.TRACK_VISUALIZATION, latched_qos
        )

        default_track = os.path.join(
            get_package_share_directory("track_pkg"), "config", "figure_8.yaml"
        )
        self.declare_parameter("track_file", default_track)
        self.declare_parameter("points_per_segment", 20)

        self._load_and_publish_track()

        self.get_logger().info("Track publisher node started")

    def _load_and_publish_track(self) -> None:
        """Load the bezier track from YAML and publish it once on the latched topics."""
        track_file = self.get_parameter("track_file").get_parameter_value().string_value
        points_per_segment = (
            self.get_parameter("points_per_segment").get_parameter_value().integer_value
        )

        bezier = BezierTrack.from_yaml(track_file)
        waypoints, curvatures, cumul = bezier.discretize(points_per_segment)

        track_msg = Track()
        track_msg.header.stamp = self.get_clock().now().to_msg()
        track_msg.header.frame_id = bezier.frame_id
        track_msg.name = bezier.name

        for x, y, heading in waypoints:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            half = heading / 2.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = sin(half)
            pose.orientation.w = cos(half)
            track_msg.waypoints.append(pose)

        track_msg.curvature = [float(k) for k in curvatures]
        track_msg.cumulative_distance = [float(d) for d in cumul]

        self._centerline_pub.publish(track_msg)
        self._publish_track_visualization(waypoints, bezier.frame_id)

        self.get_logger().info(
            f"Track '{bezier.name}' loaded: {len(waypoints)} waypoints, "
            f"total length {cumul[-1]:.2f} m"
        )

    def _publish_track_visualization(
        self, waypoints: list, frame_id: str
    ) -> None:
        """Publish a cyan LINE_STRIP MarkerArray for the track centerline."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "track_centerline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        for x, y, _ in waypoints:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        if waypoints:
            p = Point()
            p.x = waypoints[0][0]
            p.y = waypoints[0][1]
            p.z = 0.0
            marker.points.append(p)

        marker.scale.x = 0.03
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        array = MarkerArray()
        array.markers.append(marker)
        self._visualization_pub.publish(array)

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

        self._boundary_pub.publish(marker)
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
