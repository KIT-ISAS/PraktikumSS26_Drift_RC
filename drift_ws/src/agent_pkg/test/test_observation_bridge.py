"""The msg->observation bridge must reproduce the shared training-time builder exactly."""

import math

import numpy as np
import numpy.testing as npt
import pytest

# Importing agent_node pulls in shared_interfaces (a colcon-built package); skip the
# whole module rather than error when the workspace has not been built yet. Kept in a
# try/except so all imports stay at the top of the file (flake8 E402/import-order clean).
try:
    from nav_msgs.msg import Odometry

    from agent_pkg.agent_node import observation_from_messages
    from shared_interfaces.msg import Track, TrackProgress
    from vehicle_simulation.environments.observation import (
        OBS_DIM,
        build_observation,
        curvature_lookahead,
    )
    from vehicle_simulation.transforms import quaternion_from_yaw
except ImportError:
    pytest.skip("shared_interfaces not built", allow_module_level=True)


def _quat(yaw):
    from geometry_msgs.msg import Quaternion

    x, y, z, w = quaternion_from_yaw(yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


def test_bridge_matches_shared_builder():
    # A circle-ish track: 40 points, constant curvature 1.0.
    n = 40
    total_length = 2.0 * math.pi
    cumulative = [total_length * i / n for i in range(n)]
    curvature = [1.0] * n

    odom = Odometry()
    odom.pose.pose.orientation = _quat(0.9)
    odom.twist.twist.linear.x = 1.4
    odom.twist.twist.linear.y = -0.3
    odom.twist.twist.angular.z = 1.1

    progress = TrackProgress()
    progress.lateral_error = 0.12
    progress.distance = 2.5
    progress.total_length = total_length
    progress.projected_pose.orientation = _quat(0.6)

    track = Track()
    track.curvature = [float(k) for k in curvature]
    track.cumulative_distance = [float(d) for d in cumulative]

    prev_action = np.array([0.2, -0.4], dtype=np.float32)
    obs = observation_from_messages(odom, progress, track, prev_action)

    expected = build_observation(
        lateral_error=0.12,
        heading_error=0.9 - 0.6,
        vx=1.4,
        vy=-0.3,
        yaw_rate=1.1,
        prev_action=prev_action,
        curvatures_ahead=curvature_lookahead(curvature, cumulative, total_length, 2.5),
    )
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    npt.assert_allclose(obs, expected, atol=1e-6)
