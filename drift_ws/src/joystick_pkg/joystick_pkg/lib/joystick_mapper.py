from abc import ABC, abstractmethod

from sensor_msgs.msg import Joy


class JoystickMapper(ABC):
    """Abstract base class for mapping joystick input to vehicle action requests."""

    def __init__(self, axis_mapping: dict):
        super().__init__()
        self.axis_mapping = axis_mapping

    @abstractmethod
    def extract_steering(self, joy_msg: Joy) -> float:
        """
        Extract steering value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] representing the steering value,
        where negative values indicate left and positive values indicate right.
        """
        pass

    @abstractmethod
    def extract_throttle(self, joy_msg: Joy) -> float:
        """
        Extract throttle value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] where positive = forward, negative = brake/reverse.
        """
        pass
