"""Filter, transform, and package driving data for model training."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def filter_stationary(df: pd.DataFrame, min_speed: float = 0.05) -> pd.DataFrame:
    """Drop rows where the car is nearly stationary (|v| < min_speed m/s)."""
    speed = np.hypot(df["vx"].values, df["vy"].values)
    return df[speed >= min_speed].reset_index(drop=True)


def split_contiguous(df: pd.DataFrame, keep: np.ndarray) -> list[pd.DataFrame]:
    """
    Return the maximal runs of rows where ``keep`` is True, as separate DataFrames.

    Use this instead of boolean row-dropping whenever the survivors feed a
    consumer that interpolates between adjacent rows. Dropping rows in place
    leaves the excised span as a *hole* the downstream interpolator will happily
    bridge, fabricating data across exactly the interval that was excluded for
    being unrepresentative. Splitting into separate runs makes the hole a real
    boundary that no interpolation can cross.

    Only the time-gap heuristic in :func:`segment_episodes` would otherwise catch
    such a hole, and only if the excised span happened to be longer than its
    ``max_gap_s``.
    """
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(df):
        raise ValueError(f"keep has {len(keep)} entries but df has {len(df)} rows")
    if not keep.any():
        return []
    # Boundaries wherever the mask flips; slice out the True runs.
    edges = np.flatnonzero(np.diff(np.concatenate(([False], keep, [False]))))
    starts, stops = edges[0::2], edges[1::2]
    return [df.iloc[a:b].reset_index(drop=True) for a, b in zip(starts, stops)]


def segment_episodes(df: pd.DataFrame, max_gap_s: float = 1.0) -> list[pd.DataFrame]:
    """
    Split DataFrame into continuous episodes separated by time gaps > max_gap_s.

    Returns only episodes with at least 3 rows.
    """
    if len(df) < 2:
        return []
    gaps = np.diff(df["t"].values)
    bounds = [0, *(np.where(gaps > max_gap_s)[0] + 1).tolist(), len(df)]
    segments = [
        df.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True) for i in range(len(bounds) - 1)
    ]
    return [seg for seg in segments if len(seg) >= 3]


def build_dataset(
    df: pd.DataFrame,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample an episode to uniform dt and produce (state, action, delta) arrays.

    .. deprecated::
        Superseded by :mod:`vehicle_simulation.data.dataset` (``build_windows`` +
        ``DatasetSpec``), which supports variable timesteps and multi-step rollout
        windows. Kept for backward compatibility.

    state:  [vx, vy, yaw_rate]
    action: [steering, throttle]
    delta:  [(vx[t+1]-vx[t])/dt, (vy[t+1]-vy[t])/dt, (yaw_rate[t+1]-yaw_rate[t])/dt]
    """
    import warnings

    warnings.warn(
        "build_dataset is deprecated; use vehicle_simulation.data.dataset.build_windows"
        " + DatasetSpec (variable-dt, multi-step).",
        DeprecationWarning,
        stacklevel=2,
    )
    t = df["t"].values
    t_uniform = np.arange(t[0], t[-1] - dt * 0.5, dt)

    if len(t_uniform) < 2:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    def _interp(col: str) -> np.ndarray:
        return interp1d(t, df[col].values, kind="linear", fill_value="extrapolate")(t_uniform)

    vx = _interp("vx")
    vy = _interp("vy")
    yr = _interp("yaw_rate")
    steering = _interp("steering")
    throttle = _interp("throttle")

    n = len(t_uniform) - 1
    states = np.column_stack([vx[:n], vy[:n], yr[:n]]).astype(np.float32)
    actions = np.column_stack([steering[:n], throttle[:n]]).astype(np.float32)
    deltas = np.column_stack(
        [
            (vx[1:] - vx[:n]) / dt,
            (vy[1:] - vy[:n]) / dt,
            (yr[1:] - yr[:n]) / dt,
        ]
    ).astype(np.float32)

    return states, actions, deltas
