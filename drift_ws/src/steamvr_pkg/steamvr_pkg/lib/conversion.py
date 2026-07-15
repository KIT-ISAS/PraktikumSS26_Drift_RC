"""
Coordinate conversion from OpenVR tracking space to the project's ROS frames.

OpenVR tracking space is right-handed with **y up and -z forward** (x right);
the project world frames are ISO 8855 style **z up** (see ``shared_pkg.Frames``).
The fixed axis relabel used throughout is::

    x_ros = -z_vr,   y_ros = -x_vr,   z_ros = y_vr

The tracker **body** frame needs its own, different relabel: OpenVR's body
convention for the Vive tracker was measured at bring-up (static pose + a
slide along the LED direction, 2026-07-08) to be **+y toward the LED and -z
out of the top face**. ``tracker_link`` is defined to match what the
``mount_tf_node`` convention expects: **z = top-face normal, y = LED
direction** (= car forward, LED mounted facing the front), so poses are
converted as ``R @ M @ B.T`` with the two distinct relabels below. A tracker
sitting flat on the floor thus comes out with its z axis up; the remaining
mount correction is a pure yaw (no flip).

ROS-free (works on plain floats/numpy) so it is unit-testable in isolation.
"""

import numpy as np

from vehicle_simulation.transforms import quaternion_conjugate, rotate_vector_by_quaternion

__all__ = [
    "pose_from_openvr",
    "vector_from_openvr",
    "vector_to_body",
    "quaternion_from_rotation_matrix",
]

# Rows express the ROS axes in OpenVR coordinates (proper rotation, det +1).
_VR_TO_ROS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

# Rows express the tracker_link axes in OpenVR *body* coordinates: a 180 deg
# rotation about the LED axis (y), turning OpenVR's "-z out of the top face"
# into tracker_link's "+z out of the top face" (det +1, symmetric).
_VR_BODY_TO_TRACKER = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


def quaternion_from_rotation_matrix(m: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion ``(x, y, z, w)``.

    Uses the numerically stable branch on the largest diagonal element
    (Shepperd's method), so it is safe for all orientations.
    """
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def pose_from_openvr(matrix12) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a row-major OpenVR ``HmdMatrix34`` (12 floats) to a ROS pose.

    Returns ``(position, quaternion)`` of the tracker in the z-up tracking
    frame (``steamvr_world``): position as ``[x, y, z]`` and orientation as
    ``(x, y, z, w)``, with the tracker body relabelled into the
    ``tracker_link`` convention (z = top-face normal, y = LED direction).
    """
    m = np.asarray(matrix12, dtype=np.float64).reshape(3, 4)
    rotation_vr = m[:, :3]
    position_vr = m[:, 3]
    position = _VR_TO_ROS @ position_vr
    rotation = _VR_TO_ROS @ rotation_vr @ _VR_BODY_TO_TRACKER.T
    return position, quaternion_from_rotation_matrix(rotation)


def vector_from_openvr(v) -> np.ndarray:
    """
    Relabel a tracking-space OpenVR 3-vector into z-up ``steamvr_world`` axes.

    Only for tracking-space vectors (velocities); a vector expressed in the
    device's own body axes would need ``_VR_BODY_TO_TRACKER`` instead.
    """
    return _VR_TO_ROS @ np.asarray(v, dtype=np.float64)


def vector_to_body(vector_world, quat_world_body) -> np.ndarray:
    """
    Rotate a world-frame 3-vector into the body frame of ``quat_world_body``.

    Used to express the tracker's velocity-derived pseudo-IMU signals in the
    tracker body frame, as a real strapped-down IMU would report them.
    """
    return rotate_vector_by_quaternion(quaternion_conjugate(quat_world_body), vector_world)
