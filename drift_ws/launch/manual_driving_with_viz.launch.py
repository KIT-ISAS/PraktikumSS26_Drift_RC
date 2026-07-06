"""Top-level launch for manual driving with visualization."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from shared_pkg import Topics


def generate_launch_description():
    this_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(this_dir, "..", "src"))

    joystick_launch = os.path.join(src_dir, "joystick_pkg", "launch", "joystick_mapper.launch.py")
    vehicle_launch = os.path.join(
        src_dir, "vehicle_communication_pkg", "launch", "vehicle_communication.launch.py"
    )
    holodeck_launch = os.path.join(
        src_dir, "holodeck_pkg", "launch", "holodeck_tracking.launch.py"
    )
    steamvr_launch = os.path.join(src_dir, "steamvr_pkg", "launch", "steamvr_tracking.launch.py")
    localization_launch = os.path.join(
        src_dir, "car_localization_pkg", "launch", "car_localization.launch.py"
    )

    # Exclusive tracking source: libsurvive (holodeck) or steamvr (needs the
    # host-side bridge running, see steamvr_bridge/). They cannot run at the
    # same time anyway — both claim the tracker's USB dongle.
    tracking_source = LaunchConfiguration("tracking_source")
    use_libsurvive = PythonExpression(["'", tracking_source, "' == 'libsurvive'"])
    use_steamvr = PythonExpression(["'", tracking_source, "' == 'steamvr'"])

    joystick_include = IncludeLaunchDescription(PythonLaunchDescriptionSource(joystick_launch))
    vehicle_include = IncludeLaunchDescription(PythonLaunchDescriptionSource(vehicle_launch))
    holodeck_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(holodeck_launch),
        condition=IfCondition(use_libsurvive),
        launch_arguments={"foxbridge": "true"}.items(),
    )
    steamvr_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(steamvr_launch),
        condition=IfCondition(use_steamvr),
        launch_arguments={"foxbridge": "true"}.items(),
    )
    # Derives the canonical CoG/ISO-8855 state from the tracking source's raw
    # tracker TF and IMU topic.
    localization_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch),
        # namespace="" is passed explicitly: the tracking include (processed first)
        # declares a "namespace" arg defaulting to "libsurvive"/"steamvr", which leaks
        # into the shared launch context, so car_localization's own default_value=""
        # would not take effect. Without this the real car lands under that namespace
        # and its /car/* outputs get remapped e.g. to /libsurvive/car/*.
        launch_arguments={
            "namespace": "",
            "raw_imu_topic": PythonExpression(
                [
                    f"'{Topics.LIBSURVIVE_IMU}' if '",
                    tracking_source,
                    f"' == 'libsurvive' else '{Topics.STEAMVR_IMU}'",
                ]
            ),
            # Empirical mount knobs per source (how the tracker body frame maps
            # to the car). The steamvr values are provisional — verify at
            # bring-up (see steamvr_pkg/README.md) and correct them here.
            "tracker_yaw_offset_deg": PythonExpression(
                ["'90.0' if '", tracking_source, "' == 'libsurvive' else '90.0'"]
            ),
            "tracker_upside_down": PythonExpression(
                ["'true' if '", tracking_source, "' == 'libsurvive' else 'true'"]
            ),
        }.items(),
    )

    track_publisher_node = Node(
        package="track_pkg",
        executable="track_publisher_node",
        name="track_publisher_node",
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tracking_source",
                default_value="libsurvive",
                choices=["libsurvive", "steamvr"],
                description="Which tracking system publishes world -> tracker_link",
            ),
            joystick_include,
            vehicle_include,
            holodeck_include,
            steamvr_include,
            localization_include,
            track_publisher_node,
        ]
    )
