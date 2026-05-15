#!/usr/bin/env python3
"""
Joystick device monitor and joy_node manager.

This node monitors for connected joystick devices using pygame and automatically
manages the ros2-joy node lifecycle. It will:
1. Detect when a joystick device is connected
2. Start the ROS2 joy_node if not already running
3. Monitor for device disconnection
4. Gracefully handle reconnections

"""

import subprocess
import threading
from typing import Optional

import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from vehicle_communication_pkg.msg import VehicleActionRequest


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

        # Initialize pygame joystick module
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize pygame: {e}")
            raise

        self.last_device_count = 0
        self.joy_node_process: Optional[subprocess.Popen] = None
        self.device_detected = False
        self.lock = threading.Lock()

        # Create publisher for vehicle action requests
        self.vehicle_action_publisher = self.create_publisher(
            VehicleActionRequest, "/vehicle_action", 10
        )

        # Subscribe to joy topic
        self.joy_subscriber = self.create_subscription(
            Joy, "/joy", self._joy_callback, 10
        )

        self.monitor_timer = self.create_timer(
            self.device_check_interval, self._check_devices_callback
        )
        self.get_logger().info(
            f"Joystick device monitor started (check interval: {self.device_check_interval}s)"
        )

    def _check_devices_callback(self) -> None:
        """Check repeatedly for connected joystick devices and manage joy_node."""
        pygame.event.pump()  # Update pygame event queue without blocking
        joystick_count = pygame.joystick.get_count()

        with self.lock:
            # Device connected
            if joystick_count > self.last_device_count:
                self.get_logger().info(
                    f"Joystick detected! Count: {self.last_device_count} -> {joystick_count}"
                )

                # List detected devices
                for i in range(joystick_count):
                    joystick = pygame.joystick.Joystick(i)
                    self.get_logger().info(
                        f"  Device {i}: {joystick.get_name()} "
                        f"(axes: {joystick.get_numaxes()}, "
                        f"buttons: {joystick.get_numbuttons()}, "
                        f"hats: {joystick.get_numhats()})"
                    )

                if not self.device_detected:
                    self._start_joy_node()
                    self.device_detected = True

                self.last_device_count = joystick_count

            # Device disconnected
            elif joystick_count < self.last_device_count:
                self.get_logger().warn(
                    f"Joystick disconnected! Count: {self.last_device_count} -> {joystick_count}"
                )
                self.last_device_count = joystick_count

                if joystick_count == 0:
                    self.device_detected = False
                    self._stop_joy_node()

            # Check if joy_node process is still alive
            if self.joy_node_process is not None:
                returncode = self.joy_node_process.poll()
                if returncode is not None:
                    self.get_logger().warn(
                        f"joy_node process exited with code {returncode}"
                    )
                    self.joy_node_process = None
                    if self.device_detected:
                        self.get_logger().info("Restarting joy_node...")
                        self._start_joy_node()

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
            self.get_logger().info(
                f"joy_node started (PID: {self.joy_node_process.pid})"
            )
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
            self.get_logger().warn(
                f"Joy message has insufficient axes: {len(msg.axes)}"
            )
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
            steering = max(-1.0, min(1.0, steering))
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
            self.get_logger().info(
                f"Stopping joy_node (PID: {self.joy_node_process.pid})..."
            )
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
