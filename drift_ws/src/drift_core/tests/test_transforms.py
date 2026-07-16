"""Unit tests for the shared coordinate transforms."""

import math

from drift_core.transforms import (
    quaternion_conjugate,
    quaternion_from_yaw,
    quaternion_multiply,
    rotate_vector_by_quaternion,
    world_to_body_velocity,
    wrap_angle,
    yaw_from_quaternion,
)
import numpy as np
import pytest

_YAWS = [0.0, 0.3, math.pi / 2.0, -math.pi / 2.0, 2.5, -3.0, math.pi - 1e-6]


@pytest.mark.parametrize("yaw", _YAWS)
def test_yaw_quaternion_round_trip(yaw):
    x, y, z, w = quaternion_from_yaw(yaw)
    assert yaw_from_quaternion(x, y, z, w) == pytest.approx(yaw, abs=1e-9)


def test_quaternion_from_yaw_is_unit_and_yaw_only():
    x, y, z, w = quaternion_from_yaw(1.234)
    assert (x, y) == (0.0, 0.0)
    assert x * x + y * y + z * z + w * w == pytest.approx(1.0)


def test_yaw_from_quaternion_ignores_roll_pitch():
    # A quaternion with roll/pitch must still report the correct yaw: build
    # yaw * pitch and check yaw is recovered.
    yaw = 0.7
    q_yaw = quaternion_from_yaw(yaw)
    half = 0.4 / 2.0  # pitch about +y
    q_pitch = (0.0, math.sin(half), 0.0, math.cos(half))
    qx, qy, qz, qw = quaternion_multiply(q_yaw, q_pitch)
    assert yaw_from_quaternion(qx, qy, qz, qw) == pytest.approx(yaw, abs=1e-9)


@pytest.mark.parametrize("angle", [0.0, math.pi, -math.pi, 3.5, -3.5, 10.0, -10.0])
def test_wrap_angle_in_range(angle):
    wrapped = wrap_angle(angle)
    assert -math.pi - 1e-12 <= wrapped < math.pi
    # Wrapping is idempotent and preserves the angle modulo 2*pi.
    assert math.cos(wrapped) == pytest.approx(math.cos(angle))
    assert math.sin(wrapped) == pytest.approx(math.sin(angle))


def test_wrap_angle_vectorized():
    a = np.array([0.0, 4.0, -4.0, 7.0])
    out = wrap_angle(a)
    assert np.all(out >= -math.pi)
    assert np.all(out < math.pi)


def test_world_to_body_velocity_pure_heading():
    # Moving along +x_world while pointing +y_world (heading = +90 deg) means the
    # car is moving to its own right: vx_body = 0, vy_body = -1.
    vx_b, vy_b = world_to_body_velocity(1.0, 0.0, math.pi / 2.0)
    assert vx_b == pytest.approx(0.0, abs=1e-9)
    assert vy_b == pytest.approx(-1.0)


def test_world_to_body_velocity_identity_at_zero_heading():
    vx_b, vy_b = world_to_body_velocity(0.8, -0.2, 0.0)
    assert vx_b == pytest.approx(0.8)
    assert vy_b == pytest.approx(-0.2)


def test_world_to_body_velocity_preserves_speed():
    for heading in _YAWS:
        vx_b, vy_b = world_to_body_velocity(0.6, 0.9, heading)
        assert math.hypot(vx_b, vy_b) == pytest.approx(math.hypot(0.6, 0.9))


def test_world_to_body_velocity_vectorized():
    vx = np.array([1.0, 0.0])
    vy = np.array([0.0, 1.0])
    headings = np.array([0.0, math.pi / 2.0])
    vx_b, vy_b = world_to_body_velocity(vx, vy, headings)
    np.testing.assert_allclose(vx_b, [1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(vy_b, [0.0, 0.0], atol=1e-9)


def test_quaternion_multiply_identity():
    q = np.array(quaternion_from_yaw(0.9))
    identity = (0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(quaternion_multiply(q, identity), q, atol=1e-12)


def test_quaternion_multiply_composes_yaws():
    q = quaternion_multiply(quaternion_from_yaw(0.5), quaternion_from_yaw(0.7))
    assert yaw_from_quaternion(*q) == pytest.approx(1.2)


def test_quaternion_conjugate_undoes_rotation():
    q = quaternion_from_yaw(0.6)
    back = quaternion_multiply(q, quaternion_conjugate(q))
    np.testing.assert_allclose(back, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_rotate_vector_by_yaw_quaternion():
    # Rotating +x by +90 deg yaw gives +y.
    q = quaternion_from_yaw(math.pi / 2.0)
    out = rotate_vector_by_quaternion(q, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_rotate_vector_preserves_norm_and_axis():
    q = quaternion_from_yaw(1.1)
    out = rotate_vector_by_quaternion(q, [0.3, -0.4, 1.7])
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm([0.3, -0.4, 1.7]))
    # A yaw rotation leaves the z component untouched.
    assert out[2] == pytest.approx(1.7)
