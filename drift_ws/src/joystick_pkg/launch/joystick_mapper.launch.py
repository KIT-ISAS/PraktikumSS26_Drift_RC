"""Launch file for joystick mapper."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for joystick device manager."""
    default_params_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "joystick_mapper_params.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to joystick_mapper parameters YAML",
    )

    joystick_mapper_node = Node(
        package="joystick_pkg",
        executable="joystick_mapper",
        name="joystick_mapper",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_file_arg, joy_node, joystick_mapper_node])
