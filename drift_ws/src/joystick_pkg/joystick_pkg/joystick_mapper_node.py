"""Maps the Joystick topic with the correct joystick key mapping for the drift car."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Topics

from .lib import JoystickMapper, XBoxOneMapper


class JoystickMapperNode(Node):
    """ROS2 node that maps joystick input to vehicle action requests."""

    def __init__(self):
        """Initialize the joystick mapper node."""
        super().__init__("joystick_mapper")
        self._load_params()
        self._mapper: JoystickMapper = XBoxOneMapper(self.axis_mapping)

        # Create publisher for vehicle action requests
        self.vehicle_action_publisher = self.create_publisher(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10
        )

        # Subscribe to joy topic
        self.joy_subscriber = self.create_subscription(
            Joy, Topics.JOY_NODE, self._joy_callback, 10
        )

    def _joy_callback(self, msg: Joy) -> None:
        """
        Process joystick input and publish vehicle action requests.

        Maps the parameterized joystick axes to steering and throttle values.
        D-pad up/down adjusts gyro gain; d-pad left/right sets gyro mode.
        """
        self._update_gyro_from_dpad(msg)

        action_msg = VehicleActionRequest()
        action_msg.steering = self._mapper.extract_steering(msg)
        action_msg.throttle = self._mapper.extract_throttle(msg)
        action_msg.gyro_mode = self.gyro_mode
        action_msg.gyro_gain = self.gyro_gain
        self.vehicle_action_publisher.publish(action_msg)

    def _update_gyro_from_dpad(self, msg: Joy) -> None:
        """Update gyro state from d-pad input, triggering once per press."""
        dpad_h = msg.axes[self.axis_mapping["dpad_h"]]
        dpad_v = msg.axes[self.axis_mapping["dpad_v"]]

        # Rising-edge detection: only act on the frame the button becomes pressed.
        if dpad_v > 0.5 and self._dpad_prev_v <= 0.5:
            self.gyro_gain = round(min(1.0, self.gyro_gain + self.gyro_gain_step), 10)
            self.get_logger().info(f"Gyro gain: {self.gyro_gain:.2f}")
        elif dpad_v < -0.5 and self._dpad_prev_v >= -0.5:
            self.gyro_gain = round(max(0.0, self.gyro_gain - self.gyro_gain_step), 10)
            self.get_logger().info(f"Gyro gain: {self.gyro_gain:.2f}")

        if dpad_h > 0.5 and self._dpad_prev_h <= 0.5:
            self.gyro_mode = "soft"
            self.get_logger().info("Gyro mode: soft")
        elif dpad_h < -0.5 and self._dpad_prev_h >= -0.5:
            self.gyro_mode = "hard"
            self.get_logger().info("Gyro mode: hard")

        self._dpad_prev_h = dpad_h
        self._dpad_prev_v = dpad_v

    def _load_params(self) -> None:
        """Load parameters from the ROS2 parameter server."""
        self.declare_parameter("axis_mapping.throttle", 0)
        self.declare_parameter("axis_mapping.brake", 0)
        self.declare_parameter("axis_mapping.steering", 0)
        self.declare_parameter("axis_mapping.dpad_h", 6)
        self.declare_parameter("axis_mapping.dpad_v", 7)
        axis_mapping_params = self.get_parameters_by_prefix("axis_mapping")
        self.axis_mapping = {
            param_key: param_value.value for param_key, param_value in axis_mapping_params.items()
        }

        self.declare_parameter("gyro_mode", "soft")
        self.declare_parameter("gyro_gain", 0.0)
        self.declare_parameter("gyro_gain_step", 0.1)
        self.gyro_mode = self.get_parameter("gyro_mode").value
        self.gyro_gain = float(self.get_parameter("gyro_gain").value)
        self.gyro_gain_step = float(self.get_parameter("gyro_gain_step").value)

        self._dpad_prev_h = 0.0
        self._dpad_prev_v = 0.0


def main(args=None):
    """Start joystick reader node."""
    rclpy.init(args=args)

    joystick_mapper = JoystickMapperNode()

    try:
        rclpy.spin(joystick_mapper)
    except KeyboardInterrupt:
        pass
    finally:
        joystick_mapper.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
