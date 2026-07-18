"""
Pure geometry of how the tracker is mounted on the car (no ROS imports).

Produces the static transforms that turn the tracker's reported pose into the
canonical car frames. Kept ROS-free so the maths is unit-testable on its own;
:mod:`car_localization_pkg.mount_tf_node` wraps this in a ROS node.

The car-parameters YAML is the single source of truth for the geometry:
``tracker_position_relative_to_cog_m`` is the tracker's ``(x, y, z)`` position
relative to the centre of gravity in the body frame (x forward / y left / z up).
The mount *orientation* — the tracker is mounted with
its y-axis pointing forward and (per the libsurvive flip) upside down — is the
one thing that must be confirmed against a known pose at bring-up, so it is
exposed as ``yaw_offset_rad`` / ``upside_down`` rather than baked in.
"""

import math

from drift_core.car_parameters import CarParameters
from drift_core.transforms import (
    quaternion_conjugate,
    quaternion_from_yaw,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)

# 180 deg about the X axis, as a quaternion (x, y, z, w): flips an upside-down
# mount right-side up while leaving the forward axis unchanged.
_FLIP_X = (1.0, 0.0, 0.0, 0.0)

_SQRT_HALF = math.sqrt(0.5)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def body_to_tracker_rotation(yaw_offset_rad: float, upside_down: bool) -> Quat:
    """
    Rotation (x, y, z, w) taking *tracker*-frame vectors into the *body* frame.

    Built from the mount description: a yaw offset (the tracker's forward axis vs
    the body's) optionally composed with a 180 deg roll for an inverted mount.
    """
    q = quaternion_from_yaw(yaw_offset_rad)
    if upside_down:
        q = quaternion_multiply(q, _FLIP_X)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def tracker_to_base(
    car: CarParameters,
    *,
    yaw_offset_rad: float = math.pi / 2.0,
    upside_down: bool = True,
) -> tuple[Vec3, Quat]:
    """
    Static transform ``tracker_link -> base_link`` as ``(translation, quaternion)``.

    Published with parent ``tracker_link`` and child ``base_link``, so the
    translation is the CoG expressed in the tracker frame and the rotation takes
    body-frame vectors into the tracker frame. Derived by inverting the intuitive
    body-frame mount, so it stays correct if the mount knobs change.
    """
    q_tracker_to_body = body_to_tracker_rotation(yaw_offset_rad, upside_down)
    q_tf = quaternion_conjugate(q_tracker_to_body)  # rotates body vectors -> tracker

    # CoG seen from the tracker: the tracker sits at this body-frame position
    # relative to the CoG, so the CoG is the negation of it, then expressed in
    # the tracker frame.
    rel = car.tracker_position_relative_to_cog_m
    offset_body = (-rel[0], -rel[1], -rel[2])
    t = rotate_vector_by_quaternion(q_tf, offset_body)
    translation = (float(t[0]), float(t[1]), float(t[2]))
    quaternion = (float(q_tf[0]), float(q_tf[1]), float(q_tf[2]), float(q_tf[3]))
    return translation, quaternion


def tracker_to_imu() -> tuple[Vec3, Quat]:
    """
    Static transform ``tracker_link -> imu_link`` as ``(translation, quaternion)``.

    The raw libsurvive IMU is *not* coincident with the tracker pose frame: the
    pose frame is y-forward (see :func:`tracker_to_base`), but the bare IMU chip
    axes are **x forward, y right, z down** (measured from a recorded run — the
    rest gravity vector reads ``-9.7`` on z, and forward/left pushes land on
    +x/-y). Relative to the tracker pose frame that is a +90 deg yaw, so the IMU
    needs its own rotation instead of the identity; without it ``/car/imu`` comes
    out 90 deg off (accel x/y swapped) while z happens to be correct.

    Translation stays zero: the car-parameters file carries no measured IMU
    lever-arm offset yet (a future refinement, needed for CoG correction).
    """
    # +90 deg yaw about Z: composed with tracker_to_base's rotation this makes the
    # net imu_link -> base_link a pure 180 deg roll about X, i.e. (fwd, right, down)
    # -> (fwd, left, up) = ISO 8855.
    return (0.0, 0.0, 0.0), (0.0, 0.0, _SQRT_HALF, _SQRT_HALF)
