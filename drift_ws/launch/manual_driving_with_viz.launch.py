"""Top-level launch for manual driving with visualization."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    this_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(this_dir, "..", "src"))

    joystick_launch = os.path.join(src_dir, "joystick_pkg", "launch", "joystick_mapper.launch.py")
    vehicle_launch = os.path.join(
        src_dir, "vehicle_communication_pkg", "launch", "vehicle_communication.launch.py"
    )
    localization_launch = os.path.join(
        src_dir, "car_localization_pkg", "launch", "car_localization.launch.py"
    )

    # Each include is wrapped in a scoped GroupAction: launch arguments declared
    # inside one include (e.g. "params_file" in both the joystick and vehicle
    # launches) would otherwise leak into the shared context and pre-empt the
    # next include's own default. Scoping discards them when the group exits.
    joystick_include = GroupAction(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(joystick_launch))]
    )
    vehicle_include = GroupAction(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(vehicle_launch))]
    )
    # car_localization owns the tracking source: it starts the one selected by
    # tracking_source and derives the raw IMU topic + mount convention from it.
    # namespace="" is passed explicitly (belt and suspenders with the scoping
    # above): the real car must stay un-namespaced or its /car/* outputs would be
    # remapped e.g. to /libsurvive/car/*.
    localization_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(localization_launch),
                launch_arguments={
                    "namespace": "",
                    "tracking_source": LaunchConfiguration("tracking_source"),
                    "foxbridge": "true",
                }.items(),
            )
        ]
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
                description="Which tracking system publishes world -> tracker_link "
                "(steamvr needs steamvr_bridge.py running on the host)",
            ),
            joystick_include,
            vehicle_include,
            localization_include,
            track_publisher_node,
        ]
    )
