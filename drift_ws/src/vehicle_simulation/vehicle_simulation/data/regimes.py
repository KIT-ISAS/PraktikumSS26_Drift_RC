"""
Label driving regimes (stationary / reverse / straight / corner / drift) per sample.

Regimes let datasets be synthesized per scenario and let evaluation break loss
down by regime ("where does the model work / fail"). Thresholds are starting
guesses — tune them from velocity/sideslip histograms of the real data.
"""

import numpy as np
import pandas as pd

REGIMES = ("stationary", "reverse", "straight", "corner", "drift")


def label_regimes(
    df: pd.DataFrame,
    *,
    speed_min: float = 0.3,
    yaw_thresh: float = 2.0,
    sideslip_thresh: float = 0.17,
) -> pd.DataFrame:
    """
    Add a ``regime`` column ∈ {stationary, reverse, straight, corner, drift}.

    Requires ``vx``, ``vy``, ``yaw_rate`` columns. Classification (first match):
    stationary if ``speed < speed_min``; reverse if the car is not travelling
    forward (``vx <= 0``); otherwise, for forward motion, drift if
    ``|sideslip| >= sideslip_thresh``, corner if ``|yaw_rate| >= yaw_thresh``, else
    straight. ``sideslip_thresh`` is in radians (~0.17 rad ≈ 10°), ``yaw_thresh``
    in rad/s.

    The cornering and drift tests are restricted to forward motion on purpose:
    ``arctan2(vy, vx)`` is ≈ ±π when backing up in a straight line, which would
    otherwise clear any sideslip threshold and label every reverse span ``drift``
    — polluting the drift-only datasets that exist to probe the tire model.
    """
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    yaw_rate = df["yaw_rate"].to_numpy()

    speed = np.hypot(vx, vy)
    sideslip = np.arctan2(vy, vx)
    moving = speed >= speed_min
    forward = moving & (vx > 0.0)

    regime = np.full(len(df), "straight", dtype=object)
    regime[~moving] = "stationary"
    regime[moving & ~forward] = "reverse"
    corner = forward & (np.abs(yaw_rate) >= yaw_thresh)
    regime[corner] = "corner"
    drift = forward & (np.abs(sideslip) >= sideslip_thresh)
    regime[drift] = "drift"

    out = df.copy()
    out["regime"] = regime
    return out


def regime_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return a {regime: n_samples} dict (zero-filled for all known regimes)."""
    counts = {r: 0 for r in REGIMES}
    if "regime" in df.columns:
        for regime, n in df["regime"].value_counts().items():
            counts[str(regime)] = int(n)
    return counts
