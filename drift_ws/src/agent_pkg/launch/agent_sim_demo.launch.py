"""
Demo launch: let the trained RL agent drive the simulated car, watch in Foxglove.

Brings up the simulation_node (arcade model by default — the model the agent was
trained on), the track_publisher_node, a foxglove_bridge, the car mesh marker,
and the agent node closing the loop.

Usage:
    ros2 launch agent_pkg agent_sim_demo.launch.py run_dir:=/path/to/runs/<id>

Connect Foxglove to ws://localhost:8765.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    run_dir_arg = DeclareLaunchArgument(
        "run_dir",
        description="Training run directory (with best_model.zip / agent.zip) or a SAC .zip",
    )
    model_type_arg = DeclareLaunchArgument(
        "model_type",
        default_value="arcade",
        description="Simulated vehicle model (should match what the agent was trained on)",
    )
    model_weights_arg = DeclareLaunchArgument(
        "model_weights",
        default_value="",
        description="Checkpoint .pt for trained models (required for 'neural'/'hybrid')",
    )
    stochastic_arg = DeclareLaunchArgument(
        "stochastic",
        default_value="false",
        description="Sample the model's learned SDE each step (realistic wheelspin "
        "instability); false = deterministic mean dynamics",
    )
    track_file_arg = DeclareLaunchArgument(
        "track_file",
        default_value="figure_8",
        description="Track name (in drift_core/tracks) or path to a track YAML",
    )

    simulation_node = Node(
        package="simulation_pkg",
        executable="simulation_node",
        name="simulation_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("simulation_pkg"), "config", "simulation_params.yaml"]
            ),
            {
                "model_type": LaunchConfiguration("model_type"),
                "model_weights": LaunchConfiguration("model_weights"),
                "stochastic": LaunchConfiguration("stochastic"),
            },
        ],
    )

    track_publisher_node = Node(
        package="track_pkg",
        executable="track_publisher_node",
        name="track_publisher_node",
        output="screen",
        parameters=[{"track_file": LaunchConfiguration("track_file")}],
    )

    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="log",
        parameters=[{"port": 8765}],
    )

    car_viz_node = Node(
        package="car_localization_pkg",
        executable="car_viz_node",
        name="car_viz_node",
        output="screen",
    )

    agent_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("agent_pkg"), "launch", "agent.launch.py"])
        ),
        launch_arguments={"run_dir": LaunchConfiguration("run_dir")}.items(),
    )

    return LaunchDescription(
        [
            run_dir_arg,
            model_type_arg,
            model_weights_arg,
            stochastic_arg,
            track_file_arg,
            simulation_node,
            track_publisher_node,
            foxglove_bridge_node,
            car_viz_node,
            agent_launch,
        ]
    )
