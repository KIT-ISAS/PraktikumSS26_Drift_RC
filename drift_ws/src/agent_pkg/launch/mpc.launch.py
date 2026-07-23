"""
Launch the path-following MPC node as the control source.

Deterministic alternative to agent.launch.py — no training run required. Works
against either the real car stack (tracking + track publisher + vehicle
communication) or the simulation — the node only sees the canonical topics.

Usage:
    ros2 launch agent_pkg mpc.launch.py
"""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mpc_node = Node(
        package="agent_pkg",
        executable="mpc_node",
        name="mpc_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("agent_pkg"), "config", "mpc_params.yaml"]),
        ],
    )

    return LaunchDescription([mpc_node])
