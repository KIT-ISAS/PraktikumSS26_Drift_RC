"""
Top-level launch for recording human driving demonstrations.

Same tracking stack as manual_driving_with_viz, plus two passive recorders: the
demo-recorder node reconstructs training-format ``(obs, action, reward, next_obs,
done)`` transitions from the human's joystick commands (``.npz`` files for
``vehicle_simulation.agents.finetune_agent``), and ``ros2 bag record`` keeps the
raw topics so observations can be recomputed offline if the format ever changes:

    holodeck tracking (+ foxglove bridge) -> car_localization -> track_publisher
    joystick -> VehicleActionRequest -> vehicle_communication -> ESP32
                                     -> demo_recorder_node (+ rosbag)

The joystick runs in normal manual mode — the human is the only control source.
Recording starts automatically once the car is on track and moving; episodes are
cut on input gaps and on off-track/stall terminations, mirroring the training env.

Usage:
    ros2 launch drift_ws/launch/human_demo_recording.launch.py [stage_config:=stage3_drift]

Connect Foxglove to ws://localhost:8765.
"""

from datetime import datetime
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
)
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
    demo_record_launch = os.path.join(src_dir, "agent_pkg", "launch", "demo_record.launch.py")
    bag_qos_overrides = os.path.join(src_dir, "agent_pkg", "config", "demo_bag_qos.yaml")

    stage_config_arg = DeclareLaunchArgument(
        "stage_config",
        default_value="stage3_drift",
        description="Stage whose reward/termination config scores the demos (name or YAML path)",
    )

    # Each include is wrapped in a scoped GroupAction so launch arguments declared
    # inside one include cannot leak into the next (see data_collection_with_viz).
    joystick_include = GroupAction(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(joystick_launch))]
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
    demo_record_include = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(demo_record_launch),
                launch_arguments={"stage_config": LaunchConfiguration("stage_config")}.items(),
            )
        ]
    )

    track_publisher_node = Node(
        package="track_pkg",
        executable="track_publisher_node",
        name="track_publisher_node",
        output="screen",
    )

    # Raw-topic backup alongside the .npz transitions. Kept next to the npz output
    # (demo_record_params.yaml's output_dir) under a per-session directory.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.expanduser(f"~/recordings/human/bags/{stamp}")
    rosbag_record = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            bag_path,
            "--qos-profile-overrides-path",
            bag_qos_overrides,
            Topics.JOY_NODE,
            Topics.VEHICLE_ACTION_REQUEST,
            Topics.VEHICLE_STATUS,
            Topics.CAR_POSE,
            Topics.CAR_ODOM,
            Topics.CAR_IMU,
            Topics.TRACK_PROGRESS,
            Topics.TRACK_CENTERLINE,
            Topics.TF,
            Topics.TF_STATIC,
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            stage_config_arg,
            joystick_include,
            vehicle_include,
            holodeck_include,
            localization_include,
            track_publisher_node,
            demo_record_include,
            rosbag_record,
        ]
    )
