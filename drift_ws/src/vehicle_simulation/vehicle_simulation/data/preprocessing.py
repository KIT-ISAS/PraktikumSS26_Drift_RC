"""Filter, transform, and package driving data for model training."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def compute_body_frame_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vx, vy, yaw_rate columns estimated from finite differences of pose.

    vx, vy are in the vehicle body frame (vx = forward, vy = left).
    yaw_rate is computed from the unwrapped heading derivative.
    """
    t = df["t"].values
    heading = df["heading"].values

    # World-frame velocity via central differences
    dx = np.gradient(df["x"].values, t)
    dy = np.gradient(df["y"].values, t)

    # Rotate world → body frame
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    df = df.copy()
    df["vx"] = dx * cos_h + dy * sin_h
    df["vy"] = -dx * sin_h + dy * cos_h
    df["yaw_rate"] = np.gradient(np.unwrap(heading), t)
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
    split_at = np.where(gaps > max_gap_s)[0] + 1
    return [seg.reset_index(drop=True) for seg in np.split(df, split_at) if len(seg) >= 3]


def build_dataset(
    df: pd.DataFrame,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample an episode to uniform dt and produce (state, action, delta) arrays.

    state:  [vx, vy, yaw_rate]
    action: [steering, throttle]
    delta:  [(vx[t+1]-vx[t])/dt, (vy[t+1]-vy[t])/dt, (yaw_rate[t+1]-yaw_rate[t])/dt]

    The delta represents the rate of change of body-frame velocities — the quantity
    the NeuralDynamicsModel learns to predict.
    """
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
