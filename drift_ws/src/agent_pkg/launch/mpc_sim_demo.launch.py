"""
Demo launch: let the path-following MPC drive the simulated car, watch in Foxglove.

Brings up the simulation_node, the track_publisher_node, a foxglove_bridge, the
car mesh marker, and the MPC node closing the loop. Start driving with:

    ros2 topic pub --once /agent/start std_msgs/msg/Empty

Usage:
    ros2 launch agent_pkg mpc_sim_demo.launch.py [model_type:=kinematic] [track_file:=figure_8]

Connect Foxglove to ws://localhost:8765.
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
        description="Simulated vehicle model",
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
            {"model_type": LaunchConfiguration("model_type")},
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

    mpc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("agent_pkg"), "launch", "mpc.launch.py"])
        )
    )

    return LaunchDescription(
        [
            model_type_arg,
            track_file_arg,
            simulation_node,
            track_publisher_node,
            foxglove_bridge_node,
            car_viz_node,
            mpc_launch,
        ]
    )
