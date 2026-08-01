"""
Default filesystem locations for training artifacts.

Run directories live under ``drift_ws/runs/``, split by what was trained:
``agent/`` (RL policies) and ``vehicle_model/`` (dynamics models). The defaults
are anchored to the repo checkout rather than the working directory, so runs
land in one place no matter where a training command is launched from.

``DRIFT_RUNS_DIR`` overrides the base (e.g. a scratch disk on a GPU server); an
install outside a repo checkout falls back to ``./runs``.
"""

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def runs_root() -> Path:
    """Base directory holding every run directory."""
    override = os.environ.get("DRIFT_RUNS_DIR")
    if override:
        return Path(override).expanduser()
    drift_ws = _PACKAGE_DIR.parents[2]
    if (drift_ws / "src" / "drift_core").is_dir():
        return drift_ws / "runs"
    return Path("runs")


def agent_runs_dir() -> Path:
    """Base directory for RL agent runs (``drift_ws/runs/agent``)."""
    return runs_root() / "agent"


def vehicle_model_runs_dir() -> Path:
    """Base directory for vehicle-model runs (``drift_ws/runs/vehicle_model``)."""
    return runs_root() / "vehicle_model"
