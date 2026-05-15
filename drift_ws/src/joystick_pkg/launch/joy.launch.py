"""
Launch file for ROS2 joy node without overhead of automatic joystick detection.
Must be launched only after joystick is plugged in.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for joy node."""
    default_params_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "joy_params.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to joy node parameters YAML",
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_file_arg, joy_node])
