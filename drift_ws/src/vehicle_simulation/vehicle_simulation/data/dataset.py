"""
Variable-timestep, multi-step **window** dataset built from driving episodes.

Replaces the fixed-dt single-step ``build_dataset``. Each window is a short
trajectory the trainer rolls a model over: an initial canonical dynamic state, an
action sequence, a per-step dt sequence (optionally resampled to a random dt per
window so models generalize across timesteps), and the ground-truth dynamic +
pose trajectories. Single-step training is just ``horizon=1``.

This module is torch-free; the trainer wraps :func:`stack_windows` output in
tensors. Dynamic state is canonical ``[vx, vy, yaw_rate]`` (open design choice 3).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

DYNAMIC_COLS = ("vx", "vy", "yaw_rate")
# States are continuous signals: linear interpolation between samples is the right
# reconstruction. Actions are not — they are held constant between commands (the
# ESP32 holds the PWM output until the next datagram arrives), so they resample as
# a zero-order hold. Interpolating them linearly would blend across command changes
# and fabricate commands the car never received — in particular a drive/brake
# transition would emit rows with *both* nonzero, which the ESC cannot output
# (see EscStateMachine: brake and drive are mutually exclusive).
_STATE_COLS = ("x", "y", "heading_unwrapped", "vx", "vy", "yaw_rate")
_ACTION_COLS = ("steering", "drive", "brake")
_INTERP_COLS = _STATE_COLS + _ACTION_COLS


@dataclass
class Window:
    """One training/eval window over a contiguous slice of an episode."""

    dyn0: np.ndarray  # (D,) canonical [vx, vy, yaw_rate] at the window start
    actions: np.ndarray  # (H, 3) [steering, drive, brake] ESC-resolved command each step
    dts: np.ndarray  # (H,) per-step timestep [s]
    dyn_traj: np.ndarray  # (H+1, D) ground-truth dynamic states
    pose_traj: np.ndarray  # (H+1, 3) ground-truth [x, y, heading]
    regime: str  # dominant regime over the window
    episode: int  # source-episode index (for leakage-free splitting)


def _episode_interpolators(ep: pd.DataFrame):
    """
    Build per-column interpolators for an episode (heading is unwrapped first).

    States interpolate linearly; actions are held (zero-order hold) — see the
    ``_ACTION_COLS`` note above.
    """
    t = ep["t"].to_numpy()
    data = {c: ep[c].to_numpy() for c in _INTERP_COLS if c != "heading_unwrapped"}
    data["heading_unwrapped"] = np.unwrap(ep["heading"].to_numpy())
    interp = {
        c: interp1d(
            t,
            v,
            kind="previous" if c in _ACTION_COLS else "linear",
            fill_value="extrapolate",
        )
        for c, v in data.items()
    }
    return t, interp


def build_windows(
    episodes: list[pd.DataFrame],
    *,
    horizon: int,
    dt_range: tuple[float, float] = (0.02, 0.05),
    stride: int = 5,
    resample: bool = True,
    seed: int = 0,
    max_dyn_accel: float = 30.0,
    max_yaw_accel: float = 150.0,
) -> list[Window]:
    """
    Slice episodes into fixed-horizon windows with (optionally) variable dt.

    For each episode, windows start every ``stride`` samples. When ``resample``,
    each window draws ``dt ~ U(dt_range)`` and the signals are interpolated onto a
    uniform ``horizon+1`` grid; otherwise the episode's median dt is used. Windows
    that would run past the episode end are skipped. ``regime`` requires a
    ``regime`` column (else labelled ``"unknown"``).

    Windows containing physically impossible transitions are discarded: implied
    linear acceleration above ``max_dyn_accel`` [m/s²] or yaw acceleration above
    ``max_yaw_accel`` [rad/s²] (the car manages roughly 6 m/s² / 40 rad/s²).
    Such samples are tracker-solver glitches that slipped past the row-level
    sanitizer bounds — a handful of them dominates an MSE loss, so the guard
    sits here at the consumer end where the derivative labels are formed.
    """
    rng = np.random.default_rng(seed)
    windows: list[Window] = []

    for ep_idx, ep in enumerate(episodes):
        if len(ep) < 2:
            continue
        t, interp = _episode_interpolators(ep)
        regimes = ep["regime"].to_numpy() if "regime" in ep.columns else None
        median_dt = float(np.median(np.diff(t)))

        for start in range(0, len(ep), stride):
            t0 = t[start]
            dt = float(rng.uniform(*dt_range)) if resample else median_dt
            if dt <= 0:
                continue
            grid = t0 + dt * np.arange(horizon + 1)
            if grid[-1] > t[-1]:
                break  # later starts only run further past the end

            cols = {c: interp[c](grid) for c in _INTERP_COLS}
            dyn_traj = np.stack([cols[c] for c in DYNAMIC_COLS], axis=-1).astype(np.float64)
            implied = np.abs(np.diff(dyn_traj, axis=0)) / dt
            if implied[:, :2].max() > max_dyn_accel or implied[:, 2].max() > max_yaw_accel:
                continue
            pose_traj = np.stack(
                [cols["x"], cols["y"], cols["heading_unwrapped"]], axis=-1
            ).astype(np.float64)
            actions = np.stack(
                [cols["steering"][:-1], cols["drive"][:-1], cols["brake"][:-1]], axis=-1
            ).astype(np.float64)
            dts = np.full(horizon, dt, dtype=np.float64)

            if regimes is not None:
                in_span = (t >= grid[0]) & (t <= grid[-1])
                labels = regimes[in_span]
                regime = Counter(labels).most_common(1)[0][0] if len(labels) else "unknown"
            else:
                regime = "unknown"

            windows.append(
                Window(
                    dyn0=dyn_traj[0].copy(),
                    actions=actions,
                    dts=dts,
                    dyn_traj=dyn_traj,
                    pose_traj=pose_traj,
                    regime=str(regime),
                    episode=ep_idx,
                )
            )
    return windows


def split_windows(
    windows: list[Window],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.0,
    seed: int = 0,
    by_episode: bool = True,
) -> tuple[list[Window], list[Window], list[Window]]:
    """
    Split windows into (train, val, test).

    With ``by_episode`` (default) the split is over source episodes so overlapping
    windows never straddle train/val/test (prevents leakage). When fewer than two
    episodes are present it falls back to a window-level split so the val set is not
    empty. A nonzero ``val_frac``/``test_frac`` yields at least one unit when room
    allows.
    """
    rng = np.random.default_rng(seed)
    unique_episodes = {w.episode for w in windows}

    if by_episode and len(unique_episodes) >= 2:
        episodes = sorted(unique_episodes)
        rng.shuffle(episodes)
        n_test, n_val = _split_counts(len(episodes), val_frac, test_frac)
        test_ep = set(episodes[:n_test])
        val_ep = set(episodes[n_test : n_test + n_val])
        train, val, test = [], [], []
        for w in windows:
            (test if w.episode in test_ep else val if w.episode in val_ep else train).append(w)
        return train, val, test

    idx = np.arange(len(windows))
    rng.shuffle(idx)
    n_test, n_val = _split_counts(len(windows), val_frac, test_frac)
    test = [windows[i] for i in idx[:n_test]]
    val = [windows[i] for i in idx[n_test : n_test + n_val]]
    train = [windows[i] for i in idx[n_test + n_val :]]
    return train, val, test


def _split_counts(n: int, val_frac: float, test_frac: float) -> tuple[int, int]:
    """Counts for (test, val) leaving >=1 for train; >=1 each when frac>0 and room."""
    n_test = max(1, round(test_frac * n)) if test_frac > 0 else 0
    n_test = min(n_test, max(0, n - 1))
    remaining = n - n_test
    n_val = max(1, round(val_frac * n)) if val_frac > 0 else 0
    n_val = min(n_val, max(0, remaining - 1))
    return n_test, n_val


def stack_windows(windows: list[Window]) -> dict:
    """
    Stack a list of (equal-horizon) windows into batched numpy arrays.

    Returns a dict with ``dyn0 (N,D)``, ``actions (N,H,3)``, ``dts (N,H)``,
    ``dyn_traj (N,H+1,D)``, ``pose_traj (N,H+1,3)``, ``regime (N,)``,
    ``episode (N,)``. The trainer converts the numeric arrays to tensors.
    """
    if not windows:
        return {
            "dyn0": np.empty((0, 3)),
            "actions": np.empty((0, 0, 3)),
            "dts": np.empty((0, 0)),
            "dyn_traj": np.empty((0, 0, 3)),
            "pose_traj": np.empty((0, 0, 3)),
            "regime": np.empty((0,), dtype=object),
            "episode": np.empty((0,), dtype=int),
        }
    return {
        "dyn0": np.stack([w.dyn0 for w in windows]),
        "actions": np.stack([w.actions for w in windows]),
        "dts": np.stack([w.dts for w in windows]),
        "dyn_traj": np.stack([w.dyn_traj for w in windows]),
        "pose_traj": np.stack([w.pose_traj for w in windows]),
        "regime": np.array([w.regime for w in windows], dtype=object),
        "episode": np.array([w.episode for w in windows], dtype=int),
    }
