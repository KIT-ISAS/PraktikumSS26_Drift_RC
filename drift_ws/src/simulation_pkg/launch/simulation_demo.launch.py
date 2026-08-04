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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# simulation_node parameters that a launch argument may override. Each argument
# defaults to "" ("leave it to simulation_params.yaml"); only a non-empty value is
# forwarded as a per-node override. A parameter dict passed after the YAML file wins
# unconditionally, so a non-empty default here (as before) would clobber the YAML even
# when the argument was left untouched — which silently disabled the file's settings.
_NODE_PARAM_ARGS = ("model_type", "model_weights", "stochastic")


def _simulation_node(context, *_args, **_kwargs):
    overrides = {}
    for key in _NODE_PARAM_ARGS:
        value = LaunchConfiguration(key).perform(context)
        if value != "":
            overrides[key] = value
    # `stochastic` is declared as a bool parameter; the launch arg arrives as a string.
    if "stochastic" in overrides:
        overrides["stochastic"] = overrides["stochastic"].lower() in ("1", "true", "yes")
    return [
        Node(
            package="simulation_pkg",
            executable="simulation_node",
            name="simulation_node",
            output="screen",
            parameters=[
                PathJoinSubstitution(
                    [FindPackageShare("simulation_pkg"), "config", "simulation_params.yaml"]
                ),
                overrides,
            ],
        )
    ]


def generate_launch_description():
    model_type_arg = DeclareLaunchArgument(
        "model_type",
        default_value="",
        description="Vehicle model ('arcade' / 'kinematic' / 'neural' / 'hybrid' / run dir); "
        "empty = use simulation_params.yaml",
    )
    model_weights_arg = DeclareLaunchArgument(
        "model_weights",
        default_value="",
        description="Checkpoint .pt or run dir for trained models ('neural'/'hybrid')",
    )
    stochastic_arg = DeclareLaunchArgument(
        "stochastic",
        default_value="",
        description="'true' samples the model's learned SDE each step (realistic wheelspin "
        "instability); empty = use simulation_params.yaml",
    )
    track_file_arg = DeclareLaunchArgument(
        "track_file",
        default_value="figure_8",
        description="Track name (in drift_core/track/tracks) or path to a track YAML",
    )
    joystick_arg = DeclareLaunchArgument(
        "joystick",
        default_value="false",
        description="Set true to bring up the gamepad stack as the input source",
    )

    simulation_node = OpaqueFunction(function=_simulation_node)

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
            model_weights_arg,
            stochastic_arg,
            track_file_arg,
            joystick_arg,
            simulation_node,
            track_publisher_node,
            foxglove_bridge_node,
            car_viz_node,
            joystick_launch,
        ]
    )
