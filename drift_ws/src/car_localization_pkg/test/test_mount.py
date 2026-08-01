"""Unit tests for the pure tracker-mount geometry (no ROS required)."""

import math

import pytest

from car_localization_pkg.mount import (
    body_to_tracker_rotation,
    tracker_to_base,
    tracker_to_imu,
)
from drift_core.transforms import (
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)
from drift_core.vehicle_model.car_parameters import load_car_parameters

_SQRT_HALF = math.sqrt(0.5)


def test_tracker_to_base_translation_matches_car_params():
    car = load_car_parameters("drift_rc")
    rel = car.tracker_position_relative_to_cog_m
    translation, _ = tracker_to_base(car)
    # CoG in the tracker frame: forward offset lands on tracker -y, height on +z.
    assert translation[0] == pytest.approx(0.0, abs=1e-9)
    assert translation[1] == pytest.approx(-rel[0])
    assert translation[2] == pytest.approx(rel[2])


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


def test_tracker_to_imu_rotation_is_90_yaw():
    # The raw IMU chip is x-forward/y-right/z-down, a +90 deg yaw from the
    # y-forward tracker pose frame; the mount corrects it with a +90 deg yaw Z.
    translation, q = tracker_to_imu()
    assert translation == (0.0, 0.0, 0.0)
    assert q == pytest.approx((0.0, 0.0, _SQRT_HALF, _SQRT_HALF))


def test_imu_rotates_raw_chip_to_iso_base_link():
    # Full chain imu_link -> base_link must map raw (fwd, right, down) to
    # ISO 8855 (fwd, left, up), i.e. a pure 180 deg roll about X.
    car = load_car_parameters("drift_rc")
    _, q_tracker_to_base = tracker_to_base(car)  # rotates body vec -> tracker
    _, q_tracker_to_imu = tracker_to_imu()
    # base_link <- imu = (base <- tracker) . (tracker <- imu)
    q_base_from_tracker = quaternion_conjugate(q_tracker_to_base)
    q_base_from_imu = quaternion_multiply(q_base_from_tracker, q_tracker_to_imu)
    raw = (1.0, -0.3, -9.7)  # (fwd, right, down)
    out = rotate_vector_by_quaternion(q_base_from_imu, raw)
    assert out == pytest.approx((1.0, 0.3, 9.7), abs=1e-9)  # (fwd, left, up)
