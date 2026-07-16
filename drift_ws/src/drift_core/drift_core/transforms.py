"""
Coordinate transforms shared across the project (ROS-free, torch-free).

The single home for the small quaternion / frame helpers that were previously
copy-pasted into the simulation node, the MCAP writer, the rosbag reader, and
the preprocessing pipeline.

This module is deliberately **ROS-free and torch-free**: it works on plain
Python scalars and numpy arrays so it can be imported anywhere (offline tools,
ROS nodes, tests) without pulling in a heavy dependency.

Conventions (project-wide, ISO 8855):
  * World/body frames are x-forward, y-left, z-up; yaw is positive
    counter-clockwise (a left turn).
  * Quaternions are expressed component-wise in **(x, y, z, w)** order, matching
    ``geometry_msgs/Quaternion`` field order. The planar car pose is yaw-only
    (x = y = 0), but the rotation helpers are fully 3-D so they also handle the
    raw tracker orientation (which carries roll/pitch) and the IMU.
"""

import math

import numpy as np

__all__ = [
    "quaternion_from_yaw",
    "yaw_from_quaternion",
    "wrap_angle",
    "world_to_body_velocity",
    "quaternion_multiply",
    "quaternion_conjugate",
    "rotate_vector_by_quaternion",
]


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """
    Build a yaw-only quaternion ``(x, y, z, w)`` from a heading [rad].

    Inverse of :func:`yaw_from_quaternion` for planar (z-up) orientations.
    """
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """
    Extract the yaw (rotation about +z) [rad] from a quaternion ``(x, y, z, w)``.

    Uses the full formula, so it is correct even when the quaternion carries
    roll/pitch (e.g. the raw tracker orientation), not just the yaw-only case.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    """Wrap an angle (scalar or array) [rad] to the half-open range ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_body_velocity(vx_world, vy_world, heading):
    """
    Rotate a world-frame planar velocity into the body frame (by ``-heading``).

    Returns ``(vx_body, vy_body)`` with vx forward, vy left. Accepts scalars or
    numpy arrays (broadcast together), so the data pipeline can convert a whole
    trajectory at once.
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    vx_body = vx_world * cos_h + vy_world * sin_h
    vy_body = -vx_world * sin_h + vy_world * cos_h
    return vx_body, vy_body


def quaternion_multiply(q1, q2) -> np.ndarray:
    """
    Hamilton product of two quaternions ``(x, y, z, w)``; returns ``q1 * q2``.

    Composes rotations: applying the result is applying ``q2`` then ``q1``.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_conjugate(q) -> np.ndarray:
    """Conjugate (= inverse for a unit quaternion) of ``(x, y, z, w)``."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


def rotate_vector_by_quaternion(q, v) -> np.ndarray:
    """
    Rotate a 3-vector ``v`` by quaternion ``q`` ``(x, y, z, w)``.

    Used to bring the raw tracker IMU (angular velocity, linear acceleration)
    into the canonical body frame. ``q`` is assumed unit-norm.
    """
    qvec = np.asarray(q[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = float(q[3])
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)
