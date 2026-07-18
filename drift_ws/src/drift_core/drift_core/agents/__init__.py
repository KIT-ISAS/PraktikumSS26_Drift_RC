"""
RL agent training, evaluation, and inference (requires the ``[rl]`` extra to use).

:class:`DriftAgent` is imported lazily so this package can be imported without
``stable-baselines3``; the training/evaluation CLIs live in
:mod:`.train_agent` and :mod:`.evaluate_agent`.
"""

from .run_dir import create_run_dir, write_meta

__all__ = ["DriftAgent", "create_run_dir", "write_meta"]


def __getattr__(name: str):
    """Lazily import :class:`DriftAgent` (requires stable-baselines3) on first access."""
    if name == "DriftAgent":
        from .policy import DriftAgent

        return DriftAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
