"""
Bring up the source-agnostic localization subgraph for one car.

mount_tf_node (static tracker->CoG/IMU) + car_state_node (canonical
pose/odom/imu) + car_viz_node (mesh marker). Reusable per car: a simulated car
is launched with a different ``namespace`` + ``frame_prefix`` and ``color`` so it
coexists with the real car in the shared ``world`` frame. It does NOT start a
tracking source — that is the job of the source adapter (e.g. holodeck) or the
simulation node, which publishes ``world -> <prefix>tracker_link``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Launch mount_tf, car_state, and car_viz for one (real or sim) car."""
    car = LaunchConfiguration("car")
    namespace = LaunchConfiguration("namespace")
    prefix = LaunchConfiguration("frame_prefix")
    color = LaunchConfiguration("color_rgba")
    raw_imu_topic = LaunchConfiguration("raw_imu_topic")

    # world stays shared/unprefixed; the per-car frames are prefixed.
    base = PythonExpression(["'", prefix, "' + 'base_link'"])
    tracker = PythonExpression(["'", prefix, "' + 'tracker_link'"])
    imu = PythonExpression(["'", prefix, "' + 'imu_link'"])

    frames = {
        "world_frame": "world",
        "base_frame": base,
        "tracker_frame": tracker,
        "imu_frame": imu,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("car", default_value="drift_rc"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument(
                "frame_prefix",
                default_value="",
                description="Frame name prefix, e.g. 'sim/' for a simulated car.",
            ),
            DeclareLaunchArgument("color_rgba", default_value="[1.0, 1.0, 1.0, 1.0]"),
            DeclareLaunchArgument("raw_imu_topic", default_value="imu"),
            Node(
                package="car_localization_pkg",
                executable="mount_tf_node",
                name="mount_tf_node",
                namespace=namespace,
                output="screen",
                parameters=[{"car": car, **frames}],
            ),
            Node(
                package="car_localization_pkg",
                executable="car_state_node",
                name="car_state_node",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "world_frame": frames["world_frame"],
                        "base_frame": base,
                        "raw_imu_topic": raw_imu_topic,
                    }
                ],
            ),
            Node(
                package="car_localization_pkg",
                executable="car_viz_node",
                name="car_viz_node",
                namespace=namespace,
                output="screen",
                parameters=[{"color_rgba": color}],
            ),
        ]
    )
