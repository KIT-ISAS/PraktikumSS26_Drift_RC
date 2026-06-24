"""Unit tests for the pure tracker-mount geometry (no ROS required)."""

import math
import os
import sys

import pytest

# Make the package importable when run standalone (colcon provides it otherwise).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from car_localization_pkg.mount import (  # noqa: E402
    body_to_tracker_rotation,
    tracker_to_base,
    tracker_to_imu,
)

from vehicle_simulation.car_parameters import load_car_parameters  # noqa: E402
from vehicle_simulation.transforms import rotate_vector_by_quaternion  # noqa: E402

_SQRT_HALF = math.sqrt(0.5)


def test_tracker_to_base_translation_matches_car_params():
    car = load_car_parameters("drift_rc")
    translation, _ = tracker_to_base(car)
    # CoG in the tracker frame: forward offset lands on tracker -y, height on +z.
    assert translation[0] == pytest.approx(0.0, abs=1e-9)
    assert translation[1] == pytest.approx(-car.tracker_offset_from_com_m)
    assert translation[2] == pytest.approx(car.tracker_height_m)


def test_tracker_to_base_rotation_is_180_about_xy_diagonal():
    car = load_car_parameters("drift_rc")
    _, q = tracker_to_base(car)
    assert q == pytest.approx((-_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0))


def test_mount_rotation_maps_tracker_forward_to_body_forward():
    # The tracker's y-axis is the car's forward (+x) axis.
    q_tracker_to_body = body_to_tracker_rotation(math.pi / 2.0, upside_down=True)
    forward = rotate_vector_by_quaternion(q_tracker_to_body, (0.0, 1.0, 0.0))
    assert forward == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_mount_rotation_flips_tracker_up_when_upside_down():
    # Upside-down mount: the tracker's +z points down in the body frame.
    q_tracker_to_body = body_to_tracker_rotation(math.pi / 2.0, upside_down=True)
    up = rotate_vector_by_quaternion(q_tracker_to_body, (0.0, 0.0, 1.0))
    assert up == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_tracker_to_imu_is_identity():
    translation, q = tracker_to_imu()
    assert translation == (0.0, 0.0, 0.0)
    assert q == (0.0, 0.0, 0.0, 1.0)
