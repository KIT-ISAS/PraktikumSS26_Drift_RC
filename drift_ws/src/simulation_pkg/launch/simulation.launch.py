r"""
Launch the simulation stack: vehicle model + joystick + track publisher.

Replaces libsurvive_tracking.launch.py — publishes /car/pose from a model
instead of real lighthouse tracking. All downstream nodes (track_pkg,
visualization, RL agent) work unchanged.

Usage:
    # Default arcade model, manual joystick driving:
    ros2 launch simulation_pkg simulation.launch.py

    # Neural model:
    ros2 launch simulation_pkg simulation.launch.py \
        model_type:=neural model_weights:=/path/to/model.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_type_arg = DeclareLaunchArgument(
        "model_type",
        default_value="arcade",
        description="Vehicle model: arcade | kinematic | dynamic | neural | hybrid | run-dir path",
    )
    model_config_arg = DeclareLaunchArgument(
        "model_config",
        default_value="",
        description="Path to kinematic model YAML params (empty = use defaults)",
    )
    model_weights_arg = DeclareLaunchArgument(
        "model_weights",
        default_value="",
        description="Path to neural model .pt checkpoint",
    )

    simulation_node = Node(
        package="simulation_pkg",
        executable="simulation_node",
        name="simulation_node",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("simulation_pkg"), "config", "simulation_params.yaml"]
            ),
            {
                "model_type": LaunchConfiguration("model_type"),
                "model_config": LaunchConfiguration("model_config"),
                "model_weights": LaunchConfiguration("model_weights"),
            },
        ],
    )

    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("joystick_pkg"), "launch", "joystick_mapper.launch.py"]
            )
        )
    )

    # Source-agnostic car mesh marker (the same node the real car uses): renders
    # the simulated /car/pose as a mesh on /car/visualization for Foxglove.
    car_viz_node = Node(
        package="car_localization_pkg",
        executable="car_viz_node",
        name="car_viz_node",
        output="screen",
    )

    return LaunchDescription(
        [
            model_type_arg,
            model_config_arg,
            model_weights_arg,
            simulation_node,
            joystick_launch,
            car_viz_node,
        ]
    )
