"""
Top-level launch for real-car data collection with visualization.

Same stack as agent_driving_with_viz, but the data-collection node replaces the
agent node as the control source: it drives the car with the sim-trained policy
(sampled stochastically, throttle clamped) and records
``(obs, action, reward, next_obs, done)`` transitions to ``.npz`` files for
off-board fine-tuning:

    holodeck tracking (+ foxglove bridge) -> car_localization -> track_publisher
    -> data_collection_node -> VehicleActionRequest -> vehicle_communication -> ESP32

The joystick stays up as an emergency override (agent_override_mode): it is
passive while the policy drives, but significant stick/trigger movement stops
recording (/agent/stop + /agent/override) and takes over control instantly.
Publishing /agent/start arms recording again.

Usage:
    ros2 launch drift_ws/launch/data_collection_with_viz.launch.py run_dir:=/path/to/runs/<id>

Connect Foxglove to ws://localhost:8765.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from shared_pkg import Topics


def generate_launch_description():
    this_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(this_dir, "..", "src"))

    joystick_launch = os.path.join(src_dir, "joystick_pkg", "launch", "joystick_mapper.launch.py")
    vehicle_launch = os.path.join(
        src_dir, "vehicle_communication_pkg", "launch", "vehicle_communication.launch.py"
    )
    holodeck_launch = os.path.join(
        src_dir, "holodeck_pkg", "launch", "holodeck_tracking.launch.py"
    )
    localization_launch = os.path.join(
        src_dir, "car_localization_pkg", "launch", "car_localization.launch.py"
    )
    record_launch = os.path.join(src_dir, "agent_pkg", "launch", "record.launch.py")

    run_dir_arg = DeclareLaunchArgument(
        "run_dir",
        description="Training run directory (with config.yaml + best_model.zip / agent.zip)",
    )

    # Each include is wrapped in a scoped GroupAction: launch arguments declared
    # inside one include (e.g. "params_file" in both the joystick and vehicle
    # launches, "namespace" in holodeck's) would otherwise leak into the shared
    # context and pre-empt the next include's own default. Scoping discards them
    # when the group exits; top-level arguments like run_dir still flow in.
    #
    # Joystick as emergency override: passive until significant movement, then it
    # stops recording and takes over.
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
    holodeck_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(holodeck_launch),
                launch_arguments={"foxbridge": "true"}.items(),
            )
        ]
    )
    # Derives the canonical CoG/ISO-8855 state from holodeck's raw tracker TF.
    # namespace="" is passed explicitly (belt and suspenders with the scoping
    # above): the real car must stay un-namespaced or its /car/* outputs would be
    # remapped to /libsurvive/car/*.
    localization_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(localization_launch),
                launch_arguments={
                    "namespace": "",
                    "raw_imu_topic": Topics.LIBSURVIVE_IMU,
                }.items(),
            )
        ]
    )
    record_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(record_launch),
                launch_arguments={"run_dir": LaunchConfiguration("run_dir")}.items(),
            )
        ]
    )

    track_publisher_node = Node(
        package="track_pkg",
        executable="track_publisher_node",
        name="track_publisher_node",
        output="screen",
    )

    return LaunchDescription(
        [
            run_dir_arg,
            joystick_include,
            vehicle_include,
            holodeck_include,
            localization_include,
            track_publisher_node,
            record_include,
        ]
    )
