# Copyright 2022 Andrew Symington
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# Bag to save data
BAG_FILE = os.path.join(launch.logging.launch_config.log_dir, "libsurvive.bag")

# Default libsurvive configuration file
CFG_FILE = os.path.join(get_package_share_directory("holodeck_pkg"), "config", "config.json")

# Sow we don't have to repeat for composable and non-composable versions.
PARAMETERS = [
    {
        "driver_args": (
            "-v 5"
            "--disable-calibrate 1 "
            # Honest accelerometer noise at rest, automatic down-weighting while driving (norm penalty), gyro de-trusted against vibration
            "--imu-acc-variance 2e-3 "
            "--imu-acc-norm-penalty 10 "
            "--imu-gyro-variance 1e-3 "
            # Freezes the online IMU self-calibration whose saturated scale/bias causes growing tilt
            "--kalman-initial-acc-scale-variance 0 "
            "--kalman-initial-acc-bias-variance 0 "
            "--process-weight-acc-bias 0 "
            # Stops the filter from being slaved to the noisy 167 Hz MPFit poses (main jitter source)
            "--obs-pos-variance 1e-4 "
            "--obs-rot-variance 1e-5 "
            # Process noise sized for a ground vehicle instead of a VR headset
            "--process-weight-jerk 1e4 "
            "--process-weight-ang-vel 5 "
            # Prevents pose dropouts during driving and keeps partial-visibility light batches usable
            "--report-ignore-threshold 1.0 "
            "--light-ignore-threshold 10 "
            "--kalman-lightcap-minimum-sensors 3 "
            "--kalman-joint-lightcap-minimum-sensors 3 "
            # Stationary detection just above your vibration floor; occlusions under 0.5 s don't zero your velocity
            "--move-threshold-acc 0.05 "
            "--move-threshold-gyro 0.1 "
            "--kalman-zvu-no-light-time 0.5 "
            f"--init-configfile {CFG_FILE}"
        )
    },
    # {"driver_args": f"--force-calibrate 1 --init-configfile {CFG_FILE}"},
    {"tracking_frame": "libsurvive_world"},
    # Raw tracker pose is published as TF libsurvive_world -> tracker_link; the
    # canonical /car/pose, /car/odom and /car/imu are derived downstream by
    # car_localization_pkg.
    {"tracker_frame": "tracker_link"},
    {"imu_frame": "imu_link"},
    # Serial of the car's tracker. Empty = use the first available tracker; set it
    # when a controller / second tracker may also be live so only the car owns
    # tracker_link/imu_link.
    {"tracker_serial": ""},
    {"imu_topic": "imu"},
    {"joy_topic": "joy"},
    {"cfg_topic": "cfg"},
    {"lighthouse_rate": 4.0},
]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "namespace", default_value="libsurvive", description="Namespace for the non-TF topics"
        ),
        DeclareLaunchArgument(
            "composable", default_value="false", description="Launch in a composable container"
        ),
        DeclareLaunchArgument(
            "rosbridge", default_value="false", description="Launch a rosbridge"
        ),
        DeclareLaunchArgument(
            "foxbridge", default_value="false", description="Launch a foxglove bridge"
        ),
        DeclareLaunchArgument(
            "record", default_value="false", description="Record data with rosbag"
        ),
    ]

    # Non-composable launch (regular node)
    libsurvive_node = Node(
        package="holodeck_pkg",
        executable="libsurvive_ros2_node",
        name="libsurvive_ros2_node",
        namespace=LaunchConfiguration("namespace"),
        condition=UnlessCondition(LaunchConfiguration("composable")),
        output="screen",
        parameters=PARAMETERS,
    )

    # Composable launch (zero-copy node example)
    libsurvive_composable_node = ComposableNodeContainer(
        package="rclcpp_components",
        executable="component_container",
        name="libsurvive_ros2_container",
        namespace=LaunchConfiguration("namespace"),
        condition=IfCondition(LaunchConfiguration("composable")),
        composable_node_descriptions=[
            ComposableNode(
                package="holodeck_pkg",
                plugin="libsurvive_ros2::Component",
                name="libsurvive_ros2_component",
                parameters=PARAMETERS,
                extra_arguments=[{"use_intra_process_comms": True}],
            )
        ],
        output="log",
    )

    # For ros webdocker bridge
    rosbridge_node = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_server_node",
        condition=IfCondition(LaunchConfiguration("rosbridge")),
        parameters=[
            {"port": 9090},
        ],
        output="log",
    )
    rosapi_node = Node(
        package="rosapi",
        executable="rosapi_node",
        name="rosapi_node",
        condition=IfCondition(LaunchConfiguration("rosbridge")),
        output="log",
    )

    # For foxglove websocket bridge.
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

    # For recording all data from the experiment
    bag_record_node = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", BAG_FILE] + ["/tf", "/tf_static"],
        condition=IfCondition(LaunchConfiguration("record")),
        output="log",
    )

    return LaunchDescription(
        arguments
        + [
            libsurvive_node,
            libsurvive_composable_node,
            foxbridge_node,
            rosbridge_node,
            rosapi_node,
            bag_record_node,
        ]
    )
