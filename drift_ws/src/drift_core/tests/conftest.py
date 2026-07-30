"""Shared fixtures: synthetic learnable dynamics for trainer/eval tests."""

import numpy as np
import pandas as pd
import pytest

# A known linear dynamics: d(dyn)/dt = A @ dyn + B @ action. The MLP can fit this,
# so training loss should clearly decrease (good for smoke tests). Actions are the
# ESC-resolved 3-channel command [steering, drive, brake].
_A = np.array([[-0.5, 0.0, 0.0], [0.0, -1.0, 0.2], [0.0, -0.3, -0.8]])
_B = np.array([[2.0, 0.0, -1.5], [0.0, 0.5, 0.0], [3.0, 0.0, 0.0]])


def linear_episode(n: int = 300, dt: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Roll the known linear dynamics with smooth random actions into a DataFrame."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=(n, 2))
    kernel = np.ones(15) / 15.0
    steering = np.convolve(raw[:, 0], kernel, mode="same")
    throttle = np.convolve(raw[:, 1], kernel, mode="same")
    # Pedal split the way the ESC replay labels it: forward pedal drives, negative
    # pedal brakes (the synthetic "car" always has its reverse lockout set).
    drive = np.maximum(throttle, 0.0)
    brake = np.maximum(-throttle, 0.0)

    dyn = np.zeros(3)
    x = y = heading = 0.0
    rows = []
    for i in range(n):
        a = np.array([steering[i], drive[i], brake[i]])
        rows.append(
            (i * dt, x, y, heading, dyn[0], dyn[1], dyn[2], steering[i], throttle[i], a[1], a[2])
        )
        deriv = _A @ dyn + _B @ a
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        x += (dyn[0] * cos_h - dyn[1] * sin_h) * dt
        y += (dyn[0] * sin_h + dyn[1] * cos_h) * dt
        heading += dyn[2] * dt
        dyn = dyn + deriv * dt

    df = pd.DataFrame(
        rows,
        columns=[
            "t",
            "x",
            "y",
            "heading",
            "vx",
            "vy",
            "yaw_rate",
            "steering",
            "throttle",
            "drive",
            "brake",
        ],
    )
    df["regime"] = "straight"
    return df


@pytest.fixture
def linear_windows():
    """(train, val) window lists from several synthetic linear episodes."""
    from drift_core.data.dataset import build_windows, split_windows

    episodes = [linear_episode(seed=i) for i in range(5)]
    windows = build_windows(episodes, horizon=5, dt_range=(0.02, 0.04), stride=5, seed=0)
    train, val, _ = split_windows(windows, val_frac=0.3, seed=0)
    return train, val
