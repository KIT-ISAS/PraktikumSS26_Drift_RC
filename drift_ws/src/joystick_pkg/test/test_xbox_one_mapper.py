"""Unit tests for ``XBoxOneMapper`` steering/throttle extraction and trigger calibration."""

import pytest
from sensor_msgs.msg import Joy

from joystick_pkg.lib.xbox_one_mapper import XBoxOneMapper

AXIS_MAPPING = {"steering": 0, "throttle": 1, "brake": 2}


def _joy(steering: float = 0.0, throttle: float = 1.0, brake: float = 1.0) -> Joy:
    """Build a Joy message; triggers default to released (raw 1.0)."""
    msg = Joy()
    msg.axes = [float(steering), float(throttle), float(brake)]
    return msg


def _calibrated_mapper() -> XBoxOneMapper:
    """Return a mapper whose triggers have already reported a non-zero value."""
    mapper = XBoxOneMapper(AXIS_MAPPING)
    mapper.extract_throttle(_joy(throttle=1.0, brake=1.0))
    return mapper


class TestSteering:
    """Axis inversion and clamping of extract_steering."""

    def test_left_input_is_positive(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        assert mapper.extract_steering(_joy(steering=-0.5)) == pytest.approx(0.5)

    def test_right_input_is_negative(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        assert mapper.extract_steering(_joy(steering=0.5)) == pytest.approx(-0.5)

    def test_center_is_zero(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        assert mapper.extract_steering(_joy(steering=0.0)) == pytest.approx(0.0)

    def test_clamps_to_unit_range(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        assert mapper.extract_steering(_joy(steering=-2.0)) == pytest.approx(1.0)
        assert mapper.extract_steering(_joy(steering=2.0)) == pytest.approx(-1.0)


class TestTriggerCalibration:
    """joy_node reports unpressed triggers as 0.0 until they have been moved once."""

    def test_initial_zero_is_treated_as_released(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        # Before calibration a raw 0.0 must not read as a half-pressed trigger.
        assert mapper.extract_throttle(_joy(throttle=0.0, brake=0.0)) == pytest.approx(0.0)

    def test_nonzero_reading_unlocks_calibration(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        mapper.extract_throttle(_joy(throttle=0.0, brake=0.0))
        # A real reading calibrates the throttle trigger...
        assert mapper.extract_throttle(_joy(throttle=-1.0, brake=0.0)) == pytest.approx(1.0)
        # ...after which raw 0.0 means genuinely half-pressed.
        assert mapper.extract_throttle(_joy(throttle=0.0, brake=0.0)) == pytest.approx(0.5)

    def test_triggers_calibrate_independently(self):
        mapper = XBoxOneMapper(AXIS_MAPPING)
        # Calibrate only the brake; the throttle's 0.0 still reads as released.
        assert mapper.extract_throttle(_joy(throttle=0.0, brake=-1.0)) == pytest.approx(-1.0)
        assert mapper.extract_throttle(_joy(throttle=0.0, brake=1.0)) == pytest.approx(0.0)


class TestThrottle:
    """Combined throttle/brake extraction after calibration."""

    def test_throttle_pressed_is_positive(self):
        mapper = _calibrated_mapper()
        assert mapper.extract_throttle(_joy(throttle=-1.0, brake=1.0)) == pytest.approx(1.0)

    def test_brake_pressed_is_negative(self):
        mapper = _calibrated_mapper()
        assert mapper.extract_throttle(_joy(throttle=1.0, brake=-1.0)) == pytest.approx(-1.0)

    def test_both_pressed_cancel_out(self):
        mapper = _calibrated_mapper()
        assert mapper.extract_throttle(_joy(throttle=-1.0, brake=-1.0)) == pytest.approx(0.0)

    def test_half_throttle(self):
        mapper = _calibrated_mapper()
        assert mapper.extract_throttle(_joy(throttle=0.0, brake=1.0)) == pytest.approx(0.5)

    def test_clamps_to_unit_range(self):
        mapper = _calibrated_mapper()
        assert mapper.extract_throttle(_joy(throttle=-3.0, brake=1.0)) == pytest.approx(1.0)
        assert mapper.extract_throttle(_joy(throttle=1.0, brake=-3.0)) == pytest.approx(-1.0)
