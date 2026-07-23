"""Tests for the stage-parameterized reward function (pure numpy, no RL extras)."""

import numpy as np

from drift_core.environments.rewards import (
    MAX_DS_PER_STEP_M,
    RewardConfig,
    TerminationConfig,
    compute_reward,
    progress_delta,
)

DT = 0.02
ZERO_ACTION = np.zeros(2)


def _step(cfg=None, term=None, **kwargs):
    defaults = dict(
        ds=0.0,
        speed=1.0,
        lateral_error=0.0,
        heading_error=0.0,
        sideslip=0.0,
        kappa_here=0.0,
        action=ZERO_ACTION,
        prev_action=ZERO_ACTION,
        dt=DT,
        stalled=False,
    )
    defaults.update(kwargs)
    return compute_reward(cfg or RewardConfig(), term or TerminationConfig(), **defaults)


def test_progress_delta_wraps_at_lap_boundary():
    total = 13.0
    assert progress_delta(0.1, total - 0.1, total) > 0.19  # forward across the start line
    assert progress_delta(total - 0.1, 0.1, total) < -0.19  # backward across the start line
    assert abs(progress_delta(5.2, 5.0, total) - 0.2) < 1e-12  # plain forward step


def test_progress_delta_clips_projector_jumps():
    total = 13.0
    assert progress_delta(6.4, 0.0, total) == MAX_DS_PER_STEP_M
    assert progress_delta(0.0, 6.4, total) == -MAX_DS_PER_STEP_M


def test_forward_progress_on_centerline_is_positive():
    cfg = RewardConfig()
    ds = cfg.target_speed * DT
    outcome = _step(ds=ds, speed=cfg.target_speed)
    assert outcome.reward > 0.0
    assert not outcome.terminated and outcome.reason is None


def test_backward_progress_is_negative():
    outcome = _step(ds=-0.02, speed=1.0)
    assert outcome.reward < 0.0


def test_off_track_terminates_with_penalty():
    term = TerminationConfig()
    outcome = _step(term=term, lateral_error=term.lateral_limit_m + 0.01)
    assert outcome.terminated and outcome.reason == "off_track"
    assert outcome.reward < -term.off_track_penalty + 1.0


def test_stall_terminates_with_penalty():
    term = TerminationConfig()
    outcome = _step(term=term, speed=0.0, stalled=True)
    assert outcome.terminated and outcome.reason == "stall"
    assert outcome.reward < -term.stall_penalty + 1.0


def test_drift_bonus_gated_on_curvature_speed_and_sideslip():
    cfg = RewardConfig(drift_weight=2.0)
    at_target = dict(sideslip=cfg.drift_beta_target, speed=2.0, kappa_here=1.0)
    baseline = _step(cfg=RewardConfig(drift_weight=0.0), **at_target).reward
    # Bonus present in a curved zone, at speed, at the target sideslip.
    assert _step(cfg=cfg, **at_target).reward > baseline
    # No bonus on straights (|kappa| below the zone threshold).
    assert _step(cfg=cfg, sideslip=cfg.drift_beta_target, speed=2.0, kappa_here=0.0).reward <= (
        _step(cfg=RewardConfig(drift_weight=0.0), sideslip=cfg.drift_beta_target, speed=2.0).reward
    )
    # No bonus below the speed gate.
    slow = dict(sideslip=cfg.drift_beta_target, speed=0.5, kappa_here=1.0)
    no_drift_cfg = RewardConfig(drift_weight=0.0)
    assert _step(cfg=cfg, **slow).reward == _step(cfg=no_drift_cfg, **slow).reward
    # Bonus decays away from the sideslip target.
    peak = _step(cfg=cfg, **at_target).reward
    off_target = _step(
        cfg=cfg,
        sideslip=cfg.drift_beta_target + 3 * cfg.drift_beta_sigma,
        speed=2.0,
        kappa_here=1.0,
    ).reward
    assert peak > off_target


def test_overspeed_penalty_only_above_max_speed():
    cfg = RewardConfig(max_speed=3.0, overspeed_weight=2.0)
    no_cap = RewardConfig(max_speed=3.0, overspeed_weight=0.0)
    # At or below the cap the penalty is identical to having no cap at all.
    at_cap = cfg.max_speed
    assert _step(cfg=cfg, speed=at_cap).reward == _step(cfg=no_cap, speed=at_cap).reward
    assert _step(cfg=cfg, speed=2.0).reward == _step(cfg=no_cap, speed=2.0).reward
    # Above the cap it subtracts a cost that grows with the overshoot.
    assert _step(cfg=cfg, speed=3.5).reward < _step(cfg=no_cap, speed=3.5).reward
    assert _step(cfg=cfg, speed=4.0).reward < _step(cfg=cfg, speed=3.5).reward


def test_overspeed_penalty_disabled_by_default():
    fast = _step(cfg=RewardConfig(), speed=10.0)
    assert fast.reward == _step(cfg=RewardConfig(overspeed_weight=0.0), speed=10.0).reward


def test_straight_sideslip_penalty_only_outside_drift_zone():
    cfg = RewardConfig(straight_beta_weight=0.5)
    clean = _step(cfg=cfg, sideslip=0.0, kappa_here=0.0).reward
    sliding_straight = _step(cfg=cfg, sideslip=0.4, kappa_here=0.0).reward
    sliding_curve = _step(cfg=cfg, sideslip=0.4, kappa_here=1.0).reward
    assert sliding_straight < clean
    assert sliding_curve > sliding_straight  # penalty does not apply in the drift zone


def test_action_rate_penalty():
    same = _step(action=np.array([0.5, 0.5]), prev_action=np.array([0.5, 0.5])).reward
    swing = _step(action=np.array([0.5, 0.5]), prev_action=np.array([-0.5, 0.5])).reward
    assert same > swing
