"""
Bring up the localization subgraph for one car and (optionally) its tracking source.

This launch is the **single place that decides the tracking source**. Selected
with ``tracking_source:=libsurvive|steamvr``, it starts the matching source
adapter (which publishes ``world -> <prefix>tracker_link`` + the lighthouse
frames), then derives the source-specific bits internally — the raw IMU topic
and the tracker mount convention — so the parent launches only have to forward
the ``tracking_source`` argument.

The rest is source-agnostic and reusable per car:
mount_tf_node (static tracker->CoG/IMU) + car_state_node (canonical
pose/odom/imu) + car_viz_node (mesh marker). A simulated car is launched with a
different ``namespace`` + ``frame_prefix`` + ``color`` so it coexists with the
real car in the shared ``world`` frame; such a car provides its own tracker TF,
so it passes ``start_tracking_source:=false`` to skip the source adapter.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from shared_pkg import Topics


def generate_launch_description():
    """Launch the tracking source (optional) plus mount_tf, car_state, car_viz."""
    car = LaunchConfiguration("car")
    namespace = LaunchConfiguration("namespace")
    prefix = LaunchConfiguration("frame_prefix")
    color = LaunchConfiguration("color_rgba")
    tracking_source = LaunchConfiguration("tracking_source")
    start_source = LaunchConfiguration("start_tracking_source")
    foxbridge = LaunchConfiguration("foxbridge")

    libsurvive_launch = os.path.join(
        get_package_share_directory("libsurvive_pkg"), "launch", "libsurvive_tracking.launch.py"
    )
    steamvr_launch = os.path.join(
        get_package_share_directory("steamvr_pkg"), "launch", "steamvr_tracking.launch.py"
    )

    # Everything below is derived from tracking_source, so callers pick the
    # source in one place. is_libsurvive gates both the source adapter and the
    # per-source knobs (a steamvr tracker reports a different body frame + IMU).
    is_libsurvive = PythonExpression(["'", tracking_source, "' == 'libsurvive'"])
    raw_imu_topic = PythonExpression(
        [
            f"'{Topics.LIBSURVIVE_IMU}' if '",
            tracking_source,
            f"' == 'libsurvive' else '{Topics.STEAMVR_IMU}'",
        ]
    )
    # steamvr's tracker_link is y-forward (LED toward the car front) and
    # right-side up (see steamvr_pkg/lib/conversion.py), so it only needs a
    # -90 deg yaw and no flip; libsurvive's is y-forward but mounted upside-down.
    tracker_yaw_offset_deg = PythonExpression(["'90.0' if ", is_libsurvive, " else '-90.0'"])
    tracker_upside_down = PythonExpression(["'true' if ", is_libsurvive, " else 'false'"])

    # The source adapters are included here, gated on start_tracking_source AND a
    # source match, so exactly one — or none (sim car / externally started
    # source) — comes up. The source's own namespace is passed explicitly: this
    # launch's "namespace" arg (the car's, "" for the real car) would otherwise
    # be inherited by the include and clobber the source's default, dropping its
    # topics from /steamvr/* and /libsurvive/* down to bare /imu, /pose.
    should_start = PythonExpression(["'", start_source, "' == 'true'"])
    tracking_source_group = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(libsurvive_launch),
                condition=IfCondition(PythonExpression([should_start, " and ", is_libsurvive])),
                launch_arguments={"foxbridge": foxbridge, "namespace": "libsurvive"}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(steamvr_launch),
                condition=IfCondition(
                    PythonExpression([should_start, " and not ", is_libsurvive])
                ),
                launch_arguments={"foxbridge": foxbridge, "namespace": "steamvr"}.items(),
            ),
        ]
    )

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

    # The canonical /car/* topics are absolute (source-agnostic), so a node
    # namespace alone does not separate them. Remap them under /<namespace> so a
    # second (e.g. sim) instance coexists with the real car instead of colliding.
    # Empty namespace -> the remap target is identical, i.e. a no-op for the real car.
    topic_prefix = PythonExpression(["'/' + '", namespace, "' if '", namespace, "' else ''"])
    state_remaps = [
        (Topics.CAR_POSE, [topic_prefix, Topics.CAR_POSE]),
        (Topics.CAR_ODOM, [topic_prefix, Topics.CAR_ODOM]),
        (Topics.CAR_IMU, [topic_prefix, Topics.CAR_IMU]),
    ]
    viz_remaps = [
        (Topics.CAR_POSE, [topic_prefix, Topics.CAR_POSE]),
        (Topics.CAR_VISUALIZATION, [topic_prefix, Topics.CAR_VISUALIZATION]),
    ]

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
            DeclareLaunchArgument(
                "tracking_source",
                default_value="libsurvive",
                choices=["libsurvive", "steamvr"],
                description="Which tracking source to start and derive the mount/IMU "
                "convention from (steamvr needs steamvr_bridge.py on the host).",
            ),
            DeclareLaunchArgument(
                "start_tracking_source",
                default_value="true",
                description="Start the selected tracking source. Set false for a "
                "simulated car or when the source is launched elsewhere.",
            ),
            DeclareLaunchArgument(
                "foxbridge",
                default_value="false",
                description="Forwarded to the tracking source: start a foxglove bridge.",
            ),
            tracking_source_group,
            Node(
                package="car_localization_pkg",
                executable="mount_tf_node",
                name="mount_tf_node",
                namespace=namespace,
                output="screen",
                parameters=[
                    {
                        "car": car,
                        **frames,
                        "tracker_yaw_offset_deg": ParameterValue(
                            tracker_yaw_offset_deg, value_type=float
                        ),
                        "tracker_upside_down": ParameterValue(
                            tracker_upside_down, value_type=bool
                        ),
                    }
                ],
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
                remappings=state_remaps,
            ),
            Node(
                package="car_localization_pkg",
                executable="car_viz_node",
                name="car_viz_node",
                namespace=namespace,
                output="screen",
                parameters=[{"color_rgba": color}],
                remappings=viz_remaps,
            ),
        ]
    )
