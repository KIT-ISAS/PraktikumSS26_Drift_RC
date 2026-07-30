"""
Launch the data-collection node: drive the sim policy and record real transitions.

Works against either the real car stack (libsurvive + car_localization + track
publisher + vehicle communication) or the simulation (simulation_pkg + track
publisher) — the node only sees the canonical topics. Transitions are written to
``.npz`` files under the node's ``output_dir`` for off-board fine-tuning with
``drift_core.agents.finetune_agent``.

Usage:
    ros2 launch agent_pkg record.launch.py run_dir:=/path/to/runs/<id>
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    run_dir_arg = DeclareLaunchArgument(
        "run_dir",
        description="Training run directory (with config.yaml + best_model.zip / agent.zip)",
    )

    data_collection_node = Node(
        package="agent_pkg",
        executable="data_collection_node",
        name="data_collection_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("agent_pkg"), "config", "record_params.yaml"]),
            {"run_dir": LaunchConfiguration("run_dir")},
        ],
    )

    return LaunchDescription([run_dir_arg, data_collection_node])
