"""
Top-level launch for MPC-driven (real car) driving with visualization.

Same stack as agent_driving_with_viz, but the deterministic path-following MPC
replaces the RL agent as the control source (no run_dir needed):

    tracking (+ foxglove bridge) -> car_localization -> track_publisher
    -> mpc_node -> VehicleActionRequest -> vehicle_communication -> ESP32

The joystick stays up as an emergency override (agent_override_mode): it is
passive while the MPC drives, but significant stick/trigger movement stops the
MPC (/agent/stop + /agent/override) and takes over control instantly.
Publishing /agent/start hands control back to the MPC.

Usage:
    ros2 launch drift_ws/launch/mpc_driving_with_viz.launch.py [tracking_source:=steamvr]

Connect Foxglove to ws://localhost:8765.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    this_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(this_dir, "..", "src"))

    joystick_launch = os.path.join(src_dir, "joystick_pkg", "launch", "joystick_mapper.launch.py")
    vehicle_launch = os.path.join(
        src_dir, "vehicle_communication_pkg", "launch", "vehicle_communication.launch.py"
    )
    localization_launch = os.path.join(
        src_dir, "car_localization_pkg", "launch", "car_localization.launch.py"
    )
    mpc_launch = os.path.join(src_dir, "agent_pkg", "launch", "mpc.launch.py")

    tracking_source_arg = DeclareLaunchArgument(
        "tracking_source",
        default_value="libsurvive",
        choices=["libsurvive", "steamvr"],
        description="Which tracking system publishes world -> tracker_link "
        "(steamvr needs steamvr_bridge.py running on the host)",
    )

    # Each include is wrapped in a scoped GroupAction: launch arguments declared
    # inside one include (e.g. "params_file" in both the joystick and vehicle
    # launches, "namespace" in libsurvive's) would otherwise leak into the shared
    # context and pre-empt the next include's own default. Scoping discards them
    # when the group exits; top-level arguments like tracking_source still flow in.
    #
    # Joystick as emergency override: passive until significant movement, then it
    # stops the MPC and takes over.
    joystick_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(joystick_launch),
                launch_arguments={"agent_override_mode": "true"}.items(),
            )
        ]
    )
    vehicle_include = GroupAction(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(vehicle_launch))]
    )
    # car_localization owns the tracking source: it starts the one selected by
    # tracking_source and derives the raw IMU topic + mount convention from it.
    # namespace="" is passed explicitly (belt and suspenders with the scoping
    # above): the real car must stay un-namespaced or its /car/* outputs would be
    # remapped to /libsurvive/car/*.
    localization_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(localization_launch),
                launch_arguments={
                    "namespace": "",
                    "tracking_source": LaunchConfiguration("tracking_source"),
                    "foxbridge": "true",
                }.items(),
            )
        ]
    )
    mpc_include = GroupAction(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(mpc_launch))]
    )

    track_publisher_node = Node(
        package="track_pkg",
        executable="track_publisher_node",
        name="track_publisher_node",
        output="screen",
    )

    return LaunchDescription(
        [
            tracking_source_arg,
            joystick_include,
            vehicle_include,
            localization_include,
            track_publisher_node,
            mpc_include,
        ]
    )
