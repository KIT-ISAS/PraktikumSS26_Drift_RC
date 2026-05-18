from .JoystickMapper import JoystickMapper
from sensor_msgs.msg import Joy


class XBoxOneMapper(JoystickMapper):
    """Maps Xbox One controller input to vehicle action requests."""

    def extract_steering(self, joy_msg: Joy) -> float:
        """
        Extract steering value from joystick message based on configured axis mapping.
        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] representing the steering value, where negative values indicate left and positive values indicate right.
        """
        steering = -joy_msg.axes[self.axis_mapping["steering"]]  # invert axis
        return max(-1.0, min(1.0, steering))

    def extract_acceleration(self, joy_msg: Joy) -> float:
        """
        Extract acceleration value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] representing the combined throttle and brake value, where positive values indicate acceleration and negative values indicate braking.
        """
        # Normalize throttle and brake from (not pressed, fully pressed) [1, -1] to [0, 1] by inverting and scaling
        throttle = -((joy_msg.axes[self.axis_mapping["throttle"]] + 1.0) / 2.0)
        brake = -((joy_msg.axes[self.axis_mapping["brake"]] + 1.0) / 2.0)
        acceleration = throttle - brake
        return max(-1.0, min(1.0, acceleration))
