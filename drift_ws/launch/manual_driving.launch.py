"""
Top-level launch to run manual driving components.

This brings up the joystick mapper and the vehicle communication nodes
by including their package-specific launch files.
"""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    this_dir = os.path.dirname(__file__)

    joystick_launch = os.path.abspath(
        os.path.join(this_dir, "..", "src", "joystick_pkg", "launch", "joystick_mapper.launch.py")
    )

    vehicle_launch = os.path.abspath(
        os.path.join(
            this_dir,
            "..",
            "src",
            "vehicle_communication_pkg",
            "launch",
            "vehicle_communication.launch.py",
        )
    )

    joystick_include = IncludeLaunchDescription(PythonLaunchDescriptionSource(joystick_launch))
    vehicle_include = IncludeLaunchDescription(PythonLaunchDescriptionSource(vehicle_launch))

    return LaunchDescription([joystick_include, vehicle_include])
