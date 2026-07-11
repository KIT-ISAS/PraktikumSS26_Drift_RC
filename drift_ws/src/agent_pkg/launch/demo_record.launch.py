"""
Launch the demo-recorder node: passively record human joystick driving.

Works against either the real car stack (holodeck + car_localization + track
publisher + vehicle communication + joystick) or the simulation (simulation_pkg +
track publisher + joystick) — the node only listens to the canonical topics.
Transitions are written to ``.npz`` files under the node's ``output_dir`` for
off-board fine-tuning with ``vehicle_simulation.agents.finetune_agent``.

Usage:
    ros2 launch agent_pkg demo_record.launch.py stage_config:=stage3_drift
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    stage_config_arg = DeclareLaunchArgument(
        "stage_config",
        default_value="stage3_drift",
        description="Stage whose reward/termination config scores the demos (name or YAML path)",
    )

    demo_recorder_node = Node(
        package="agent_pkg",
        executable="demo_recorder_node",
        name="demo_recorder_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("agent_pkg"), "config", "demo_record_params.yaml"]
            ),
            {"stage_config": LaunchConfiguration("stage_config")},
        ],
    )

    return LaunchDescription([stage_config_arg, demo_recorder_node])
