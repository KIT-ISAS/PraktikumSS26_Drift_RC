"""
Launch the RL agent node as the control source.

Works against either the real car stack (holodeck + track publisher + vehicle
communication) or the simulation (simulation_pkg + track publisher) — the node
only sees the canonical topics.

Usage:
    ros2 launch agent_pkg agent.launch.py run_dir:=/path/to/runs/<id>
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    run_dir_arg = DeclareLaunchArgument(
        "run_dir",
        description="Training run directory (with best_model.zip / agent.zip) or a SAC .zip",
    )

    agent_node = Node(
        package="agent_pkg",
        executable="agent_node",
        name="agent_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("agent_pkg"), "config", "agent_params.yaml"]),
            {"run_dir": LaunchConfiguration("run_dir")},
        ],
    )

    return LaunchDescription([run_dir_arg, agent_node])
