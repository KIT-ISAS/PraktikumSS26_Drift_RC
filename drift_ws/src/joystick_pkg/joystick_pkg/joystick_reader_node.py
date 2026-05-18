#!/usr/bin/env python3
"""
Joystick device monitor and joy_node manager.

This node monitors for connected joystick devices and manages ros2-joy node lifecycle.
Additionnally, it processes joystick input and publishes VehicleActionRequest
messages based on the controller mapping.
It will:
1. Detect when a joystick device is connected
2. Start the ROS2 joy_node if not already running
3. Monitor for device disconnection
4. Gracefully handle reconnections

"""

from pathlib import Path
import subprocess
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from shared_interfaces.msg import VehicleActionRequest


class JoystickReaderNode(Node):
    """Manages joystick device detection and joy_node lifecycle."""

    def __init__(self):
        """Initialize the device manager node."""
        super().__init__("joystick_reader")

        self.declare_parameter("device_check_interval", 1.0)
        self.declare_parameter("publish_rate", 50)
        self.declare_parameter("deadzone", 0.05)

        self.device_check_interval = self.get_parameter("device_check_interval").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.deadzone = self.get_parameter("deadzone").value

        self.last_device_count = 0
        self.joy_node_process: Optional[subprocess.Popen] = None
        self.device_detected = False
        self.lock = threading.Lock()

        # Create publisher for vehicle action requests
        self.vehicle_action_publisher = self.create_publisher(
            VehicleActionRequest, "/vehicle_action_request", 10
        )

        # Subscribe to joy topic
        self.joy_subscriber = self.create_subscription(Joy, "/joy", self._joy_callback, 10)

        self.monitor_timer = self.create_timer(
            self.device_check_interval, self._check_devices_callback
        )
        self.get_logger().info(
            f"Joystick device monitor started (check interval: {self.device_check_interval}s)"
        )

    def _check_devices_callback(self) -> None:
        """Check repeatedly for connected joystick devices and manage joy_node."""
        joystick_count = len(list(Path("/dev/input").glob("js*")))

        with self.lock:
            if joystick_count > 0 and not self.device_detected:
                self.get_logger().info("Joystick connected")
                self.device_detected = True
                self._start_joy_node()

            elif joystick_count == 0 and self.device_detected:
                self.get_logger().warn("Joystick disconnected")
                self.device_detected = False
                self._stop_joy_node()

    def _start_joy_node(self) -> None:
        """Start the ros2-joy node."""
        if self.joy_node_process is not None:
            self.get_logger().info("joy_node already running")
            return

        try:
            self.get_logger().info("Starting joy_node...")
            cmd = [
                "ros2",
                "run",
                "joy",
                "joy_node",
                "--ros-args",
                "-p",
                f"publish_rate:={self.publish_rate}",
                "-p",
                f"deadzone:={self.deadzone}",
            ]

            self.joy_node_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.get_logger().info(f"joy_node started (PID: {self.joy_node_process.pid})")
        except Exception as e:
            self.get_logger().error(f"Failed to start joy_node: {e}")
            self.joy_node_process = None

    def _joy_callback(self, msg: Joy) -> None:
        """
        Process joystick input and publish vehicle action requests.

        Xbox One Controller Mapping:
        - axes[0]: left stick X (steering, range: [-1, 1])
        - axes[2]: left trigger / LB (brake)
        - axes[5]: right trigger / RB (acceleration)
        """
        if len(msg.axes) < 6:
            self.get_logger().warn(f"Joy message has insufficient axes: {len(msg.axes)}")
            return

        try:
            # Extract steering from left stick X-axis
            steering = msg.axes[0]

            # Extract acceleration and brake from shoulder triggers
            # Triggers typically range from 1 (not pressed) to -1 (fully pressed)
            # Normalize to [0, 1] range
            forward_acceleration = -((msg.axes[5] + 1.0) / 2.0)  # Right trigger (RB)
            brake = -((msg.axes[2] + 1.0) / 2.0)  # Left trigger (LB)

            # Calculate acceleration: acceleration minus brake (brake overrules)
            # Result range: [-1, 1]
            acceleration = forward_acceleration - brake

            # Clamp values to [-1, 1] range
            steering = -max(-1.0, min(1.0, steering))
            acceleration = max(-1.0, min(1.0, acceleration))

            # Create and publish VehicleActionRequest
            action_msg = VehicleActionRequest()
            action_msg.steering = steering
            action_msg.acceleration = acceleration
            self.vehicle_action_publisher.publish(action_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing joy message: {e}")

    def _stop_joy_node(self) -> None:
        """Stop the ros2-joy node."""
        if self.joy_node_process is None:
            return

        try:
            self.get_logger().info(f"Stopping joy_node (PID: {self.joy_node_process.pid})...")
            self.joy_node_process.terminate()

            try:
                self.joy_node_process.wait(timeout=5.0)
                self.get_logger().info("joy_node stopped gracefully")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("joy_node did not stop in time, killing...")
                self.joy_node_process.kill()
                self.joy_node_process.wait()
                self.get_logger().info("joy_node killed")

        except Exception as e:
            self.get_logger().error(f"Error stopping joy_node: {e}")
        finally:
            self.joy_node_process = None

    def destroy_node(self) -> None:
        """Clean up resources."""
        self._stop_joy_node()
        super().destroy_node()


def main(args=None):
    """Start joystick reader node."""
    rclpy.init(args=args)

    joystick_reader = JoystickReaderNode()

    try:
        rclpy.spin(joystick_reader)
    except KeyboardInterrupt:
        pass
    finally:
        joystick_reader.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
