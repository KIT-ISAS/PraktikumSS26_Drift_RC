r"""
Demo launch: drive a simple arcade car in sim and watch it in Foxglove.

Brings up the simulation_node (default: the game-like ``arcade`` model), the
track_publisher_node (centerline + lap progress), and a foxglove_bridge. Then
feed it control inputs from any one of:

  * keyboard  -> ros2 run joystick_pkg keyboard_control   (separate terminal)
  * a rosbag  -> ros2 bag play <bag> --topics /car/vehicle_action_request
  * a gamepad -> launch with joystick:=true

Usage:
    # Arcade model + track + Foxglove (then drive via keyboard or bag):
    ros2 launch simulation_pkg simulation_demo.launch.py

    # Include the joystick stack as the input source:
    ros2 launch simulation_pkg simulation_demo.launch.py joystick:=true

    # Try the no-slip kinematic model instead:
    ros2 launch simulation_pkg simulation_demo.launch.py model_type:=kinematic

Connect Foxglove to ws://localhost:8765.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_type_arg = DeclareLaunchArgument(
        "model_type",
        default_value="arcade",
        description="Vehicle model: 'arcade' (game-like drift), 'kinematic', or 'neural'",
    )
    track_file_arg = DeclareLaunchArgument(
        "track_file",
        default_value="figure_8",
        description="Track name (in vehicle_simulation/tracks) or path to a track YAML",
    )
    joystick_arg = DeclareLaunchArgument(
        "joystick",
        default_value="false",
        description="Set true to bring up the gamepad stack as the input source",
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

    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("joystick_pkg"), "launch", "joystick_mapper.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("joystick")),
    )

    # Source-agnostic car mesh marker on /car/visualization (same node the real
    # car uses) so the simulated car shows as a mesh in Foxglove.
    car_viz_node = Node(
        package="car_localization_pkg",
        executable="car_viz_node",
        name="car_viz_node",
        output="screen",
    )

    return LaunchDescription(
        [
            model_type_arg,
            track_file_arg,
            joystick_arg,
            simulation_node,
            track_publisher_node,
            foxglove_bridge_node,
            car_viz_node,
            joystick_launch,
        ]
    )
