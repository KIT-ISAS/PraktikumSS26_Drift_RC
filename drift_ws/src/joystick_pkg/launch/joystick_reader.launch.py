"""Launch file for joystick device manager with automatic device detection."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for joystick device manager."""
    # Declare launch arguments
    device_check_interval_arg = DeclareLaunchArgument(
        "device_check_interval",
        default_value="1.0",
        description="Time in seconds between device detection checks",
    )

    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="50",
        description="Publishing rate in Hz for the joy_node",
    )

    deadzone_arg = DeclareLaunchArgument(
        "deadzone",
        default_value="0.05",
        description="Deadzone threshold for joystick inputs",
    )

    # Create the joystick reader node
    joystick_reader_node = Node(
        package="joystick_pkg",
        executable="joystick_reader",
        name="joystick_reader",
        output="screen",
        parameters=[
            {
                "device_check_interval": LaunchConfiguration("device_check_interval"),
                "publish_rate": LaunchConfiguration("publish_rate"),
                "deadzone": LaunchConfiguration("deadzone"),
            }
        ],
    )

    return LaunchDescription(
        [
            device_check_interval_arg,
            publish_rate_arg,
            deadzone_arg,
            joystick_reader_node,
        ]
    )
