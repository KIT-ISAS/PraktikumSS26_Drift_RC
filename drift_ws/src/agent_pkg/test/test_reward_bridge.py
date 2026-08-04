"""The msg->reward bridge must reproduce the shared training-time reward exactly."""

import math

import numpy as np
import pytest

# Importing the bridge pulls in shared_interfaces (a colcon-built package); skip the
# whole module rather than error when the workspace has not been built yet.
try:
    from nav_msgs.msg import Odometry

    from agent_pkg.observation_bridge import reward_from_messages, score_step
    from drift_core.agent.environments.rewards import (
        RewardConfig,
        TerminationConfig,
        compute_reward,
        progress_delta,
    )
    from drift_core.transforms import quaternion_from_yaw, wrap_angle
    from shared_interfaces.msg import Track, TrackProgress
except ImportError:
    pytest.skip("shared_interfaces not built", allow_module_level=True)


def _quat(yaw):
    from geometry_msgs.msg import Quaternion

    x, y, z, w = quaternion_from_yaw(yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


def _messages(vx, vy, yaw_rate, car_yaw, tangent_yaw, lateral_error, curvature_value):
    # A constant-curvature track so kappa_here is curvature_value at any distance.
    n = 40
    total_length = 2.0 * math.pi
    odom = Odometry()
    odom.pose.pose.orientation = _quat(car_yaw)
    odom.twist.twist.linear.x = vx
    odom.twist.twist.linear.y = vy
    odom.twist.twist.angular.z = yaw_rate

    progress = TrackProgress()
    progress.lateral_error = lateral_error
    progress.distance = 2.5
    progress.total_length = total_length
    progress.projected_pose.orientation = _quat(tangent_yaw)

    track = Track()
    track.curvature = [float(curvature_value)] * n
    track.cumulative_distance = [total_length * i / n for i in range(n)]
    return odom, progress, track


def _expected(
    reward_cfg,
    term_cfg,
    vx,
    vy,
    car_yaw,
    tangent_yaw,
    lateral_error,
    kappa,
    action,
    prev_action,
    ds,
    dt,
    stalled,
):
    return compute_reward(
        reward_cfg,
        term_cfg,
        ds=ds,
        speed=math.hypot(vx, vy),
        lateral_error=lateral_error,
        heading_error=wrap_angle(car_yaw - tangent_yaw),
        sideslip=math.atan2(vy, vx),
        kappa_here=kappa,
        action=action,
        prev_action=prev_action,
        dt=dt,
        stalled=stalled,
    )


def test_reward_bridge_matches_compute_reward():
    reward_cfg = RewardConfig(drift_weight=2.0, straight_beta_weight=0.5)
    term_cfg = TerminationConfig()
    action = np.array([0.3, 0.6], dtype=np.float32)
    prev_action = np.array([0.1, 0.5], dtype=np.float32)
    ds, dt = 0.04, 0.02

    odom, progress, track = _messages(
        vx=1.4,
        vy=-0.35,
        yaw_rate=1.1,
        car_yaw=0.9,
        tangent_yaw=0.6,
        lateral_error=0.12,
        curvature_value=0.8,
    )
    outcome = reward_from_messages(
        reward_cfg, term_cfg, odom, progress, track, action, prev_action, ds, dt, stalled=False
    )
    expected = _expected(
        reward_cfg, term_cfg, 1.4, -0.35, 0.9, 0.6, 0.12, 0.8, action, prev_action, ds, dt, False
    )
    assert outcome.reward == pytest.approx(expected.reward, abs=1e-6)
    assert outcome.terminated == expected.terminated
    assert outcome.reason == expected.reason


def test_reward_bridge_flags_off_track():
    reward_cfg = RewardConfig()
    term_cfg = TerminationConfig(lateral_limit_m=0.4)
    action = np.zeros(2, dtype=np.float32)

    odom, progress, track = _messages(
        vx=2.0,
        vy=0.0,
        yaw_rate=0.0,
        car_yaw=0.0,
        tangent_yaw=0.0,
        lateral_error=0.9,
        curvature_value=0.0,  # |lateral_error| > limit -> off_track
    )
    outcome = reward_from_messages(
        reward_cfg,
        term_cfg,
        odom,
        progress,
        track,
        action,
        action,
        ds=0.04,
        dt=0.02,
        stalled=False,
    )
    assert outcome.terminated
    assert outcome.reason == "off_track"


def test_score_step_matches_reward_from_messages():
    reward_cfg = RewardConfig(drift_weight=2.0)
    term_cfg = TerminationConfig()
    action = np.array([0.3, 0.6], dtype=np.float32)
    prev_action = np.array([0.1, 0.5], dtype=np.float32)
    dt = 0.02

    odom, progress, track = _messages(
        vx=1.4,
        vy=-0.35,
        yaw_rate=1.1,
        car_yaw=0.9,
        tangent_yaw=0.6,
        lateral_error=0.12,
        curvature_value=0.8,
    )
    prev_distance = 2.46  # progress.distance is 2.5, so ds = 0.04
    outcome, stall_time = score_step(
        reward_cfg,
        term_cfg,
        odom,
        progress,
        track,
        action=action,
        prev_action=prev_action,
        prev_distance=prev_distance,
        dt=dt,
        stall_time=0.5,
    )
    ds = progress_delta(progress.distance, prev_distance, progress.total_length)
    expected = reward_from_messages(
        reward_cfg, term_cfg, odom, progress, track, action, prev_action, ds, dt, stalled=False
    )
    assert outcome.reward == pytest.approx(expected.reward, abs=1e-6)
    assert not outcome.terminated
    # Speed is well above stall_speed: the stall timer resets.
    assert stall_time == 0.0


def test_score_step_stall_timer_accumulates_and_terminates():
    reward_cfg = RewardConfig()
    term_cfg = TerminationConfig(stall_speed=0.1, stall_timeout_s=2.0)
    action = np.zeros(2, dtype=np.float32)
    dt = 0.02

    odom, progress, track = _messages(
        vx=0.0,
        vy=0.0,
        yaw_rate=0.0,
        car_yaw=0.0,
        tangent_yaw=0.0,
        lateral_error=0.0,
        curvature_value=0.0,
    )
    stall_time = 0.0
    outcome, stall_time = score_step(
        reward_cfg,
        term_cfg,
        odom,
        progress,
        track,
        action=action,
        prev_action=action,
        prev_distance=progress.distance,
        dt=dt,
        stall_time=stall_time,
    )
    assert stall_time == pytest.approx(dt)
    assert not outcome.terminated

    outcome, stall_time = score_step(
        reward_cfg,
        term_cfg,
        odom,
        progress,
        track,
        action=action,
        prev_action=action,
        prev_distance=progress.distance,
        dt=dt,
        stall_time=term_cfg.stall_timeout_s - dt,
    )
    assert stall_time == pytest.approx(term_cfg.stall_timeout_s)
    assert outcome.terminated
    assert outcome.reason == "stall"


def test_score_step_ds_is_wrap_aware():
    reward_cfg = RewardConfig(
        speed_weight=0.0, lateral_weight=0.0, heading_weight=0.0, action_rate_weight=0.0
    )
    term_cfg = TerminationConfig()
    action = np.zeros(2, dtype=np.float32)

    odom, progress, track = _messages(
        vx=2.0,
        vy=0.0,
        yaw_rate=0.0,
        car_yaw=0.0,
        tangent_yaw=0.0,
        lateral_error=0.0,
        curvature_value=0.0,
    )
    # Crossing the start line: distance jumps from near total_length back to 2.5.
    prev_distance = progress.total_length - 0.02
    outcome, _ = score_step(
        reward_cfg,
        term_cfg,
        odom,
        progress,
        track,
        action=action,
        prev_action=action,
        prev_distance=prev_distance,
        dt=0.02,
        stall_time=0.0,
    )
    # With only the progress term active, the reward is exactly the wrapped ds
    # (clipped to MAX_DS_PER_STEP_M), never the huge negative raw difference.
    assert 0.0 < outcome.reward <= 0.5
