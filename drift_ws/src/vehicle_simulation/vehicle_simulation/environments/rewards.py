"""
Stage-parameterized reward for the drift environment (pure numpy, no RL extras).

One reward function serves the whole curriculum: the stage YAMLs only change the
weights. Stage 1 (slow path following) and stage 2 (fast laps) run with
``drift_weight = 0``; stage 3 turns the drift bonus on inside curved track zones
while ``straight_beta_weight`` asks the car to straighten out in between.

All shaping terms are scaled by ``dt`` so they are per-second rates and the reward
scale is invariant to the control frequency; progress is paid per meter. The
action-rate penalty is per decision (it already compares consecutive decisions).
"""

from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np


@dataclass
class RewardConfig:
    """Reward weights and drift-zone definition for one curriculum stage."""

    progress_weight: float = 1.0  # per meter of forward progress along the centerline
    target_speed: float = 1.0  # m/s
    speed_weight: float = 0.5  # * (speed - target)^2 * dt
    lateral_weight: float = 4.0  # * lateral_error^2 * dt
    heading_weight: float = 1.0  # * (1 - cos(heading_error)) * dt
    action_rate_weight: float = 0.05  # * ||action - prev_action||^2 per decision
    drift_weight: float = 0.0  # drift bonus peak rate (1/s); 0 disables (stages 1-2)
    drift_beta_target: float = 0.5  # rad — sideslip magnitude the bonus peaks at
    drift_beta_sigma: float = 0.25  # rad — bonus falls off as a Gaussian around the target
    drift_kappa_min: float = 0.5  # 1/m — drift zone: |local curvature| at/above this
    drift_min_speed: float = 1.0  # m/s — no bonus for sliding slowly
    straight_beta_weight: float = 0.0  # * sideslip^2 * dt outside the drift zone


@dataclass
class TerminationConfig:
    """Episode termination limits and penalties for one curriculum stage."""

    lateral_limit_m: float = 0.4  # off-track when |lateral_error| exceeds this
    stall_speed: float = 0.1  # m/s — below this the stall timer runs
    stall_timeout_s: float = 2.0  # continuous slow time before the episode ends
    off_track_penalty: float = 20.0
    stall_penalty: float = 10.0


class StepOutcome(NamedTuple):
    """Reward plus termination decision for one environment step."""

    reward: float
    terminated: bool
    reason: str | None  # "off_track" | "stall" | None


# Guard against projector re-localization jumps (e.g. at the figure-8 crossing): a
# real car cannot advance more than this per 20 ms step.
MAX_DS_PER_STEP_M = 0.5


def progress_delta(s: float, s_prev: float, total_length: float) -> float:
    """Signed arc-length progress from ``s_prev`` to ``s`` on a closed track (wrap-aware)."""
    half = total_length / 2.0
    ds = math.fmod(s - s_prev + half, total_length)
    if ds < 0.0:
        ds += total_length
    ds -= half
    return float(np.clip(ds, -MAX_DS_PER_STEP_M, MAX_DS_PER_STEP_M))


def compute_reward(
    cfg: RewardConfig,
    term: TerminationConfig,
    *,
    ds: float,
    speed: float,
    lateral_error: float,
    heading_error: float,
    sideslip: float,
    kappa_here: float,
    action,
    prev_action,
    dt: float,
    stalled: bool,
) -> StepOutcome:
    """
    Score one step and decide termination.

    ``ds`` is the wrap-corrected progress (use :func:`progress_delta`); ``stalled``
    is the environment's stall-timer verdict; ``kappa_here`` is the signed local
    centerline curvature at the car's projection.
    """
    action = np.asarray(action, dtype=np.float64)
    prev_action = np.asarray(prev_action, dtype=np.float64)

    reward = cfg.progress_weight * ds
    reward -= cfg.speed_weight * (speed - cfg.target_speed) ** 2 * dt
    reward -= cfg.lateral_weight * lateral_error**2 * dt
    reward -= cfg.heading_weight * (1.0 - math.cos(heading_error)) * dt
    reward -= cfg.action_rate_weight * float(np.sum((action - prev_action) ** 2))

    in_drift_zone = abs(kappa_here) >= cfg.drift_kappa_min
    if in_drift_zone:
        if cfg.drift_weight > 0.0 and speed >= cfg.drift_min_speed:
            deviation = (abs(sideslip) - cfg.drift_beta_target) / cfg.drift_beta_sigma
            reward += cfg.drift_weight * math.exp(-(deviation**2)) * dt
    else:
        reward -= cfg.straight_beta_weight * sideslip**2 * dt

    if abs(lateral_error) > term.lateral_limit_m:
        return StepOutcome(reward - term.off_track_penalty, True, "off_track")
    if stalled:
        return StepOutcome(reward - term.stall_penalty, True, "stall")
    return StepOutcome(reward, False, None)
