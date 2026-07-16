"""Track boundary publisher node."""

from math import cos, sin

from geometry_msgs.msg import Point, Pose, PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from drift_core.track import TrackProjector, load_track
from shared_interfaces.msg import Track, TrackProgress
from shared_pkg import Topics


class TrackPublisherNode(Node):
    """Publishes the pre-loaded track centerline and a config-derived boundary."""

    def __init__(self):
        """Initialize the node, subscriber, publishers, and parameters."""
        super().__init__("track_publisher_node")

        self._projector: TrackProjector | None = None
        self._track_frame_id = ""
        self._track_name = ""

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._boundary_pub = self.create_publisher(Marker, Topics.TRACK_BOUNDARY, latched_qos)
        self._centerline_pub = self.create_publisher(Track, Topics.TRACK_CENTERLINE, latched_qos)
        self._visualization_pub = self.create_publisher(
            MarkerArray, Topics.TRACK_VISUALIZATION, latched_qos
        )
        self._progress_pub = self.create_publisher(TrackProgress, Topics.TRACK_PROGRESS, 10)
        self._progress_viz_pub = self.create_publisher(
            MarkerArray, Topics.TRACK_PROGRESS_VISUALIZATION, 10
        )

        self.declare_parameter("track_file", "figure_8")
        self.declare_parameter("points_per_segment", 20)
        self.declare_parameter("car_pose_topic", Topics.CAR_POSE)
        self.declare_parameter("search_window_m", 0.6)
        self.declare_parameter("lost_distance_m", 0.5)
        self.declare_parameter("checkpoint_spacing_m", 1.0)
        self.declare_parameter("track_width_m", 0.8)
        self.declare_parameter("continuity_threshold_m", 3.0)
        # Boundary corners [x0, y0, x1, y1, ...] in the track frame. Empty (default)
        # derives an axis-aligned box around the centerline with boundary_margin_m.
        self.declare_parameter("arena_corners", [])
        self.declare_parameter("boundary_margin_m", 0.5)

        self._load_and_publish_track()

        car_pose_topic = self.get_parameter("car_pose_topic").get_parameter_value().string_value
        self._car_pose_sub = self.create_subscription(
            PoseStamped, car_pose_topic, self._car_pose_callback, 10
        )

        self.get_logger().info("Track publisher node started")

    def _load_and_publish_track(self) -> None:
        """Load the bezier track (by packaged name or path) and publish it on latched topics."""
        track_file = self.get_parameter("track_file").get_parameter_value().string_value
        points_per_segment = (
            self.get_parameter("points_per_segment").get_parameter_value().integer_value
        )

        bezier = load_track(track_file)
        waypoints, curvatures, cumul = bezier.discretize(points_per_segment)

        self._track_frame_id = bezier.frame_id
        self._track_name = bezier.name
        self._projector = TrackProjector(
            waypoints,
            cumul,
            search_window_m=self.get_parameter("search_window_m").value,
            lost_distance_m=self.get_parameter("lost_distance_m").value,
            checkpoint_spacing_m=self.get_parameter("checkpoint_spacing_m").value,
            track_width_m=self.get_parameter("track_width_m").value,
            continuity_threshold_m=self.get_parameter("continuity_threshold_m").value,
        )

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
        self._publish_boundary(waypoints, bezier.frame_id)

        self.get_logger().info(
            f"Track '{bezier.name}' loaded: {len(waypoints)} waypoints, "
            f"total length {cumul[-1]:.2f} m"
        )

    def _boundary_corners(self, waypoints: list) -> list[tuple[float, float]]:
        """Boundary polygon corners: explicit arena_corners, else a centerline bbox."""
        flat = list(self.get_parameter("arena_corners").value or [])
        if flat:
            return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]

        margin = float(self.get_parameter("boundary_margin_m").value)
        xs = [w[0] for w in waypoints]
        ys = [w[1] for w in waypoints]
        x0, x1 = min(xs) - margin, max(xs) + margin
        y0, y1 = min(ys) - margin, max(ys) + margin
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _publish_track_visualization(self, waypoints: list, frame_id: str) -> None:
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

    def _car_pose_callback(self, msg: PoseStamped) -> None:
        """Project the incoming car pose onto the track and publish progress."""
        if self._projector is None:
            return

        proj = self._projector.project(msg.pose.position.x, msg.pose.position.y)
        if proj is None:
            return

        progress_msg = TrackProgress()
        progress_msg.header.stamp = msg.header.stamp
        progress_msg.header.frame_id = self._track_frame_id
        progress_msg.track_name = self._track_name
        progress_msg.lap = proj.lap
        progress_msg.lap_progress = float(proj.lap_progress)
        progress_msg.distance = float(proj.distance)
        progress_msg.total_length = float(self._projector.total_length)
        progress_msg.lateral_error = float(proj.lateral_error)
        progress_msg.off_track = bool(proj.off_track)

        pose = Pose()
        pose.position.x = proj.x
        pose.position.y = proj.y
        pose.position.z = 0.0
        half = proj.heading / 2.0
        pose.orientation.z = sin(half)
        pose.orientation.w = cos(half)
        progress_msg.projected_pose = pose

        self._progress_pub.publish(progress_msg)
        self._publish_progress_visualization(proj)

    def _publish_progress_visualization(self, proj) -> None:
        """Publish a projection sphere plus a lap/progress text and bar for Foxglove."""
        now = self.get_clock().now().to_msg()
        frame = self._track_frame_id
        array = MarkerArray()

        # Sphere marking the projected point on the centerline.
        sphere = Marker()
        sphere.header.stamp = now
        sphere.header.frame_id = frame
        sphere.ns = "track_progress"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = proj.x
        sphere.pose.position.y = proj.y
        sphere.pose.position.z = 0.05
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.15
        sphere.color.g = 1.0
        sphere.color.a = 1.0
        array.markers.append(sphere)

        # Horizontal progress bar above the track: grey background + green fill.
        bar_width = 2.0
        bar_x0 = -bar_width / 2.0
        bar_y = 1.5

        background = Marker()
        background.header.stamp = now
        background.header.frame_id = frame
        background.ns = "track_progress"
        background.id = 1
        background.type = Marker.CUBE
        background.action = Marker.ADD
        background.pose.position.x = 0.0
        background.pose.position.y = bar_y
        background.pose.position.z = 0.05
        background.pose.orientation.w = 1.0
        background.scale.x = bar_width
        background.scale.y = 0.15
        background.scale.z = 0.02
        background.color.r = background.color.g = background.color.b = 0.3
        background.color.a = 0.4
        array.markers.append(background)

        fill_len = max(proj.lap_progress * bar_width, 1e-3)
        fill = Marker()
        fill.header.stamp = now
        fill.header.frame_id = frame
        fill.ns = "track_progress"
        fill.id = 2
        fill.type = Marker.CUBE
        fill.action = Marker.ADD
        fill.pose.position.x = bar_x0 + fill_len / 2.0
        fill.pose.position.y = bar_y
        fill.pose.position.z = 0.06
        fill.pose.orientation.w = 1.0
        fill.scale.x = fill_len
        fill.scale.y = 0.15
        fill.scale.z = 0.03
        fill.color.g = 1.0
        fill.color.a = 0.9
        array.markers.append(fill)

        # Text label above the bar.
        text = Marker()
        text.header.stamp = now
        text.header.frame_id = frame
        text.ns = "track_progress"
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = 0.0
        text.pose.position.y = bar_y + 0.25
        text.pose.position.z = 0.0
        text.pose.orientation.w = 1.0
        text.scale.z = 0.2
        text.color.r = text.color.g = text.color.b = 1.0
        text.color.a = 1.0
        text.text = f"Lap {proj.lap} — {proj.lap_progress * 100.0:.0f}%"
        array.markers.append(text)

        self._progress_viz_pub.publish(array)

    def _publish_boundary(self, waypoints: list, frame_id: str):
        """Publish a LINE_STRIP marker for the (config-derived) track boundary."""
        corners = self._boundary_corners(waypoints)
        if len(corners) < 2:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
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
