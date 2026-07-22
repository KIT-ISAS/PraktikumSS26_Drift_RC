"""The msg->MPC-input bridge must extract the exact signals PathMpc expects."""

import math

import pytest

# Importing the bridge pulls in shared_interfaces (a colcon-built package); skip the
# whole module rather than error when the workspace has not been built yet. Kept in a
# try/except so all imports stay at the top of the file (flake8 E402/import-order clean).
try:
    from nav_msgs.msg import Odometry

    from agent_pkg.mpc_bridge import mpc_inputs_from_messages
    from drift_core.transforms import quaternion_from_yaw
    from shared_interfaces.msg import TrackProgress
except ImportError:
    pytest.skip("shared_interfaces not built", allow_module_level=True)


def _quat(yaw):
    from geometry_msgs.msg import Quaternion

    x, y, z, w = quaternion_from_yaw(yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


def _messages(car_yaw, track_yaw, lateral_error=0.12, vx=1.4, distance=2.5):
    odom = Odometry()
    odom.pose.pose.orientation = _quat(car_yaw)
    odom.twist.twist.linear.x = vx
    progress = TrackProgress()
    progress.lateral_error = lateral_error
    progress.distance = distance
    progress.projected_pose.orientation = _quat(track_yaw)
    return odom, progress


def test_bridge_extracts_signals():
    e_y, e_psi, vx, s = mpc_inputs_from_messages(*_messages(car_yaw=0.9, track_yaw=0.6))
    assert e_y == pytest.approx(0.12)
    assert e_psi == pytest.approx(0.3)
    assert vx == pytest.approx(1.4)
    assert s == pytest.approx(2.5)


def test_heading_error_wraps():
    _, e_psi, _, _ = mpc_inputs_from_messages(
        *_messages(car_yaw=math.pi - 0.1, track_yaw=-math.pi + 0.1)
    )
    assert e_psi == pytest.approx(-0.2)
