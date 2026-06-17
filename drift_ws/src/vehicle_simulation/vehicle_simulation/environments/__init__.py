"""
Simulation environments and the shared observation/action builder.

``DriftEnv`` is imported lazily so the lightweight observation/action helpers
(used by the real-car deployment node) can be imported without pulling in
``gymnasium`` / ``stable-baselines3``.
"""

from .observation import (
    ACTION_DIM,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    action_from_array,
    build_observation,
    observation_from_state,
)

__all__ = [
    "DriftEnv",
    "ACTION_DIM",
    "OBS_DIM",
    "OBS_HIGH",
    "OBS_LOW",
    "action_from_array",
    "build_observation",
    "observation_from_state",
]


def __getattr__(name: str):
    """Lazily import the heavy ``DriftEnv`` only when actually requested."""
    if name == "DriftEnv":
        from .drift_env import DriftEnv

        return DriftEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
