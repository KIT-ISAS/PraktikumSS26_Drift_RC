"""Filter, transform, and package driving data for model training."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def compute_body_frame_velocities(
    df: pd.DataFrame, smooth_window: int = 0, smooth_polyorder: int = 2
) -> pd.DataFrame:
    """
    Add vx, vy, yaw_rate columns estimated from finite differences of pose.

    vx, vy are in the vehicle body frame (vx = forward, vy = left). yaw_rate is
    computed from the unwrapped heading derivative.

    High-rate trackers (e.g. the Vive at ~250 Hz) make raw finite differences very
    noisy. Pass ``smooth_window`` > 1 (samples; forced odd) to Savitzky-Golay
    smooth x / y / unwrapped-heading before differentiating. ``smooth_window=0``
    keeps the raw behaviour.
    """
    t = df["t"].values
    heading_unwrapped = np.unwrap(df["heading"].values)
    x = df["x"].values
    y = df["y"].values

    if smooth_window and smooth_window > 1:
        win = int(smooth_window) | 1  # force odd
        if win <= len(df) and win > smooth_polyorder:
            x = savgol_filter(x, win, smooth_polyorder)
            y = savgol_filter(y, win, smooth_polyorder)
            heading_unwrapped = savgol_filter(heading_unwrapped, win, smooth_polyorder)

    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    cos_h = np.cos(heading_unwrapped)
    sin_h = np.sin(heading_unwrapped)

    df = df.copy()
    df["vx"] = dx * cos_h + dy * sin_h
    df["vy"] = -dx * sin_h + dy * cos_h
    df["yaw_rate"] = np.gradient(heading_unwrapped, t)
    return df


def filter_stationary(df: pd.DataFrame, min_speed: float = 0.05) -> pd.DataFrame:
    """Drop rows where the car is nearly stationary (|v| < min_speed m/s)."""
    speed = np.hypot(df["vx"].values, df["vy"].values)
    return df[speed >= min_speed].reset_index(drop=True)


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
