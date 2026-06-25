from sensor_msgs.msg import Joy

from .joystick_mapper import JoystickMapper


class XBoxOneMapper(JoystickMapper):
    """Maps Xbox One controller input to vehicle action requests."""

    def __init__(self, axis_mapping: dict):
        """
        Initialize the Xbox One mapper with the given axis mapping.

        :param axis_mapping: A dictionary mapping 'throttle', 'brake', and 'steering'
        to their respective joystick axis indices.
        """
        super().__init__(axis_mapping)
        # joy_node reports unpressed triggers as 0.0 (not 1.0) until the
        # trigger has been moved at least once. Treat the throttle/brake
        # axes as "released" (= 1.0) until we see a clearly non-zero value.
        self._trigger_calibrated: dict[str, bool] = {"throttle": False, "brake": False}

    def extract_steering(self, joy_msg: Joy) -> float:
        """
        Extract steering value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] in ISO 8855: positive = left, negative = right.
        """
        # ISO 8855 (positive = left). The stick's left deflection reads positive on
        # this axis, so it passes straight through (no inversion). The physical
        # servo direction is applied at the actuation edge (ESP32 firmware).
        steering = joy_msg.axes[self.axis_mapping["steering"]]
        return max(-1.0, min(1.0, steering))

    def extract_throttle(self, joy_msg: Joy) -> float:
        """
        Extract throttle value from joystick message based on configured axis mapping.

        :param joy_msg: The incoming joystick ROS message containing axes data.
        :returns: A float in [-1, 1] where positive = forward, negative = brake/reverse.
        """
        throttle_raw = self._trigger_value(joy_msg, "throttle")
        brake_raw = self._trigger_value(joy_msg, "brake")
        # Normalize triggers from [1, -1] (released, fully pressed) to [0, 1].
        throttle = (1.0 - throttle_raw) / 2.0
        brake = (1.0 - brake_raw) / 2.0
        return max(-1.0, min(1.0, throttle - brake))

    def _trigger_value(self, joy_msg: Joy, key: str) -> float:
        """Return the raw trigger axis value, treating pre-calibration zero as released."""
        raw = joy_msg.axes[self.axis_mapping[key]]
        if not self._trigger_calibrated[key]:
            if raw == 0.0:
                return 1.0  # treat as released until we see a real reading
            self._trigger_calibrated[key] = True
        return raw
