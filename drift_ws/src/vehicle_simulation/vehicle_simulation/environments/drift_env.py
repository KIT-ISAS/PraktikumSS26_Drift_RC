"""
DriftEnv — RL training environment (SCAFFOLD; not implemented yet).

This is structure to *prepare* RL-agent training, not an implementation — the
current focus is the vehicle models. Intended design when RL work begins: a
``gymnasium.Env`` wrapping a vehicle model (built via ``build_model``) on a track,
using the shared observation/action builder in :mod:`observation` so the agent
sees identical inputs in simulation and on the real car (the sim-to-real seam).
Reward shaping and termination are deliberately left unimplemented.

Kept import-light on purpose: importing this module does not require
``gymnasium`` / ``stable-baselines3`` (the ``[rl]`` extra) — only actually using
the environment will.
"""

from .observation import ACTION_DIM, OBS_DIM


class DriftEnv:
    """Placeholder RL environment — see the module docstring. Not implemented yet."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DriftEnv is a scaffold; RL-agent training is not implemented yet "
            "(current focus: vehicle models). Intended: a gymnasium.Env wrapping a "
            f"vehicle model + track via the shared observation builder "
            f"(OBS_DIM={OBS_DIM}, ACTION_DIM={ACTION_DIM})."
        )
