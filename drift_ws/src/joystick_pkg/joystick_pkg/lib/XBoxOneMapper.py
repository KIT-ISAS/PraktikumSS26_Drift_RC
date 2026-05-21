from sensor_msgs.msg import Joy

from .JoystickMapper import JoystickMapper


class XBoxOneMapper(JoystickMapper):
    """Maps Xbox One controller input to vehicle action requests."""

    def __init__(self, axis_mapping: dict):
        """
        Initialize the Xbox One mapper with the given axis mapping.

        :param axis_mapping: A dictionary mapping 'throttle', 'brake', and 'steering'
        to their respective joystick axis indices.
        """
        super().__init__(axis_mapping)

    def extract_steering(self, joy_msg: Joy) -> float:
        """
        Extract steering value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] representing the steering value,
        where negative values indicate left and positive values indicate right.
        """
        steering = -joy_msg.axes[self.axis_mapping["steering"]]  # invert axis
        return max(-1.0, min(1.0, steering))

    def extract_throttle(self, joy_msg: Joy) -> float:
        """
        Extract throttle value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] where positive = forward, negative = brake/reverse.
        """
        # Normalize triggers from [1, -1] (released, fully pressed) to [0, 1]
        throttle = (1.0 - joy_msg.axes[self.axis_mapping["throttle"]]) / 2.0
        brake = (1.0 - joy_msg.axes[self.axis_mapping["brake"]]) / 2.0
        return max(-1.0, min(1.0, throttle - brake))
