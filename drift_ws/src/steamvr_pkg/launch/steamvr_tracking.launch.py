"""
Bring up the SteamVR tracking source.

Container-side half of the SteamVR pipeline — the host must be running
``steamvr_bridge/steamvr_bridge.py`` (see that README). Mirrors
holodeck_tracking.launch.py's options so the two sources are interchangeable
behind the ``tracking_source`` argument of the top-level launch files.
"""

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

BAG_FILE = os.path.join(launch.logging.launch_config.log_dir, "steamvr.bag")

DEFAULT_PARAMS_FILE = os.path.join(
    get_package_share_directory("steamvr_pkg"), "config", "steamvr_params.yaml"
)


def generate_launch_description():
    """Launch steamvr_node plus the optional foxglove bridge / rosbag record."""
    arguments = [
        DeclareLaunchArgument(
            "namespace", default_value="steamvr", description="Namespace for the non-TF topics"
        ),
        # Uniquely named ("params_file" alone would collide with the argument the
        # joystick include leaks into a shared top-level launch context).
        DeclareLaunchArgument(
            "steamvr_params_file",
            default_value=DEFAULT_PARAMS_FILE,
            description="Node parameters incl. the world -> steamvr_world calibration",
        ),
        DeclareLaunchArgument(
            "foxbridge", default_value="false", description="Launch a foxglove bridge"
        ),
        DeclareLaunchArgument(
            "record", default_value="false", description="Record data with rosbag"
        ),
    ]

    steamvr_node = Node(
        package="steamvr_pkg",
        executable="steamvr_node",
        name="steamvr_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[LaunchConfiguration("steamvr_params_file")],
    )

    foxbridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        condition=IfCondition(LaunchConfiguration("foxbridge")),
        parameters=[
            {"port": 8765},
        ],
        output="log",
    )

    bag_record_node = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", BAG_FILE] + ["/tf", "/tf_static"],
        condition=IfCondition(LaunchConfiguration("record")),
        output="log",
    )

    return LaunchDescription(
        arguments
        + [
            steamvr_node,
            foxbridge_node,
            bag_record_node,
        ]
    )
