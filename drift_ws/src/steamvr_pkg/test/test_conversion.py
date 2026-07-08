"""Unit tests for the OpenVR -> ROS coordinate conversion (no ROS required)."""

import math

import numpy as np
import pytest

from steamvr_pkg.lib.conversion import (
    pose_from_openvr,
    quaternion_from_rotation_matrix,
    vector_from_openvr,
    vector_to_body,
)
from vehicle_simulation.transforms import (
    quaternion_from_yaw,
    rotate_vector_by_quaternion,
    wrap_angle,
    yaw_from_quaternion,
)


def _vr_matrix(rotation: np.ndarray, translation=(0.0, 0.0, 0.0)) -> list[float]:
    """Build a row-major HmdMatrix34 list from a 3x3 rotation and a translation."""
    m = np.column_stack([rotation, np.asarray(translation, dtype=float)])
    return list(m.reshape(-1))


def _vr_rotation_about_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# A physically flat tracker (top face up, LED toward VR world +x), expressed in
# the empirically measured OpenVR body convention: +y toward the LED, -z out of
# the top face. Columns are the body axes in VR world coordinates.
_FLAT_TRACKER_LED_ALONG_VR_X = np.column_stack(
    [
        np.array([0.0, 0.0, -1.0]),  # x_vr(body)
        np.array([1.0, 0.0, 0.0]),  # y_vr(body): LED direction
        np.array([0.0, -1.0, 0.0]),  # z_vr(body): into the floor
    ]
)


def test_flat_tracker_comes_out_z_up_with_y_toward_led():
    _, quat = pose_from_openvr(_vr_matrix(_FLAT_TRACKER_LED_ALONG_VR_X))
    up = rotate_vector_by_quaternion(quat, (0.0, 0.0, 1.0))
    led = rotate_vector_by_quaternion(quat, (0.0, 1.0, 0.0))
    assert up == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    # VR world +x relabels to ROS world -y (see vector_from_openvr).
    assert led == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert vector_from_openvr((1.0, 0.0, 0.0)) == pytest.approx(led)


def test_translation_axes_are_relabelled():
    # x_ros = -z_vr, y_ros = -x_vr, z_ros = y_vr
    position, _ = pose_from_openvr(_vr_matrix(np.eye(3), translation=(1.0, 2.0, 3.0)))
    assert position == pytest.approx((-3.0, -1.0, 2.0))


def test_vr_up_maps_to_ros_up():
    assert vector_from_openvr((0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0))


def test_vr_yaw_maps_to_ros_yaw_with_same_sign_and_magnitude():
    # A rotation about the VR up axis (y) must become the same *relative*
    # rotation about the ROS up axis (z): proper rotations preserve angle and
    # handedness.
    angle = math.radians(30.0)
    base = _FLAT_TRACKER_LED_ALONG_VR_X
    _, quat0 = pose_from_openvr(_vr_matrix(base))
    _, quat1 = pose_from_openvr(_vr_matrix(_vr_rotation_about_y(angle) @ base))
    relative = wrap_angle(yaw_from_quaternion(*quat1) - yaw_from_quaternion(*quat0))
    assert relative == pytest.approx(angle)
    # Yaw-only: no roll/pitch leakage.
    up = rotate_vector_by_quaternion(quat1, (0.0, 0.0, 1.0))
    assert up == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_converted_rotation_stays_a_rotation():
    quat = pose_from_openvr(_vr_matrix(_vr_rotation_about_y(1.234)))[1]
    assert np.linalg.norm(quat) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        ((1.0, -1.0, -1.0), (1.0, 0.0, 0.0, 0.0)),
        ((-1.0, 1.0, -1.0), (0.0, 1.0, 0.0, 0.0)),
        ((-1.0, -1.0, 1.0), (0.0, 0.0, 1.0, 0.0)),
    ],
)
def test_quaternion_from_matrix_handles_180_degree_rotations(diagonal, expected):
    # trace = -1 exercises the non-trace branches of Shepperd's method.
    quat = quaternion_from_rotation_matrix(np.diag(diagonal))
    assert tuple(np.abs(quat)) == pytest.approx(expected)


def test_vector_to_body_undoes_the_body_yaw():
    quat_world_body = quaternion_from_yaw(math.pi / 2.0)
    assert vector_to_body((1.0, 0.0, 0.0), quat_world_body) == pytest.approx(
        (0.0, -1.0, 0.0), abs=1e-12
    )
