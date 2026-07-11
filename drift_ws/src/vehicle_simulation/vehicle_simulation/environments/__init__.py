"""
RL environments and the shared observation/action builder (sim-to-real seam).

The observation/reward/config helpers are numpy-only and import eagerly — the
real-car deployment node uses them without RL dependencies. :class:`DriftEnv`
needs ``gymnasium`` (the ``[rl]`` extra) and is imported lazily on first access.
"""

from .config import EnvConfig, SpawnConfig, StageConfig, TrainConfig, list_stages
from .observation import (
    ACTION_DIM,
    LOOKAHEAD_OFFSETS_M,
    N_LOOKAHEAD,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    action_from_array,
    build_observation,
    curvature_lookahead,
    observation_from_state,
)
from .rewards import RewardConfig, StepOutcome, TerminationConfig, compute_reward, progress_delta

__all__ = [
    "ACTION_DIM",
    "DriftEnv",
    "EnvConfig",
    "LOOKAHEAD_OFFSETS_M",
    "N_LOOKAHEAD",
    "OBS_DIM",
    "OBS_HIGH",
    "OBS_LOW",
    "RewardConfig",
    "SpawnConfig",
    "StageConfig",
    "StepOutcome",
    "TerminationConfig",
    "TrainConfig",
    "action_from_array",
    "build_observation",
    "compute_reward",
    "curvature_lookahead",
    "list_stages",
    "observation_from_state",
    "progress_delta",
]


def __getattr__(name: str):
    """Lazily import :class:`DriftEnv` (requires the ``[rl]`` extra) on first access."""
    if name == "DriftEnv":
        from .drift_env import DriftEnv

        return DriftEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
