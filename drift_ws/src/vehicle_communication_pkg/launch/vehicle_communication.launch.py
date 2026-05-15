"""Launch file for the vehicle communication node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for the vehicle communication node."""
    vehicle_communication_node = Node(
        package="vehicle_communication_pkg",
        executable="vehicle_communication_node",
        name="vehicle_communication_node",
        output="screen",
    )

    return LaunchDescription([vehicle_communication_node])
