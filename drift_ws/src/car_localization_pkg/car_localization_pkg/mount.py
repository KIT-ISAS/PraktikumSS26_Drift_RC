"""
Pure geometry of how the tracker is mounted on the car (no ROS imports).

Produces the static transforms that turn the tracker's reported pose into the
canonical car frames. Kept ROS-free so the maths is unit-testable on its own;
:mod:`car_localization_pkg.mount_tf_node` wraps this in a ROS node.

The car-parameters YAML is the single source of truth for the geometry: the
tracker sits ``tracker_offset_from_com_m`` ahead of the centre of gravity and
``tracker_height_m`` above the ground (both measured in the body frame, x
forward / y left / z up). The mount *orientation* — the tracker is mounted with
its y-axis pointing forward and (per the libsurvive flip) upside down — is the
one thing that must be confirmed against a known pose at bring-up, so it is
exposed as ``yaw_offset_rad`` / ``upside_down`` rather than baked in.
"""

import math

from vehicle_simulation.car_parameters import CarParameters
from vehicle_simulation.transforms import (
    quaternion_conjugate,
    quaternion_from_yaw,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)

# 180 deg about the X axis, as a quaternion (x, y, z, w): flips an upside-down
# mount right-side up while leaving the forward axis unchanged.
_FLIP_X = (1.0, 0.0, 0.0, 0.0)

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

    # CoG seen from the tracker: the tracker is offset (forward, up) in the body
    # frame, so the CoG is the negation of that, expressed in the tracker frame.
    offset_body = (-car.tracker_offset_from_com_m, 0.0, -car.tracker_height_m)
    t = rotate_vector_by_quaternion(q_tf, offset_body)
    translation = (float(t[0]), float(t[1]), float(t[2]))
    quaternion = (float(q_tf[0]), float(q_tf[1]), float(q_tf[2]), float(q_tf[3]))
    return translation, quaternion


def tracker_to_imu() -> tuple[Vec3, Quat]:
    """
    Static transform ``tracker_link -> imu_link`` as ``(translation, quaternion)``.

    Identity by default: the car-parameters file carries no measured IMU offset
    yet, so the IMU is treated as coincident with the tracker pose frame. Split
    out so a measured offset can be added later without touching the node.
    """
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
