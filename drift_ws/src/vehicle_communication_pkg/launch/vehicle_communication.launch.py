"""Launch file for the vehicle communication node."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for the vehicle communication node."""
    default_params_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "config",
        "vehicle_communication_params.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to vehicle_communication parameters YAML",
    )

    vehicle_communication_node = Node(
        package="vehicle_communication_pkg",
        executable="vehicle_communication_node",
        name="vehicle_communication_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_file_arg, vehicle_communication_node])
