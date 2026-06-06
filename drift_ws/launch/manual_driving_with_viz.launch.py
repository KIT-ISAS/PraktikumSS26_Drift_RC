"""
Top-level launch for manual driving with visualization.

Brings up joystick mapper, vehicle communication, holodeck tracking,
and the track publisher node.
"""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


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

    joystick_include = IncludeLaunchDescription(PythonLaunchDescriptionSource(joystick_launch))
    vehicle_include = IncludeLaunchDescription(PythonLaunchDescriptionSource(vehicle_launch))
    holodeck_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(holodeck_launch),
        launch_arguments={"foxbridge": "true"}.items(),
    )

    track_publisher_node = Node(
        package="track_pkg",
        executable="track_publisher_node",
        name="track_publisher_node",
        output="screen",
    )

    return LaunchDescription(
        [joystick_include, vehicle_include, holodeck_include, track_publisher_node]
    )
