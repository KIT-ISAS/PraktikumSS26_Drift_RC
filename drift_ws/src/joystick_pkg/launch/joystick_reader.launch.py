"""Launch file for joystick device manager with automatic device detection."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for joystick device manager."""
    default_params_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "joystick_params.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to joystick_reader parameters YAML",
    )

    joystick_reader_node = Node(
        package="joystick_pkg",
        executable="joystick_reader",
        name="joystick_reader",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_file_arg, joystick_reader_node])
