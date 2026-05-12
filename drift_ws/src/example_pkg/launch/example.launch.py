"""Launch description for example ROS 2 nodes."""

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description():
    """Launch the example node and the foxglove bridge."""
    return LaunchDescription(
        [
            Node(
                package="example_pkg",
                executable="example_node",
                name="example_node",
                output="screen",
            ),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
            ),
        ]
    )
