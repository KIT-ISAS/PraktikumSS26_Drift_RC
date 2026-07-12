"""Tests for the variable-dt window dataset and the DatasetSpec."""

import numpy as np
import pandas as pd
import pytest

from vehicle_simulation.data.dataset import build_windows, split_windows, stack_windows
from vehicle_simulation.data.preprocessing import build_dataset
from vehicle_simulation.data.spec import DatasetSpec


def _episode(n: int = 200, dt: float = 0.01) -> pd.DataFrame:
    t = np.arange(n) * dt
    vx = 1.0 + 0.1 * np.sin(t)
    vy = 0.05 * np.cos(t)
    yaw_rate = 0.2 * np.sin(t)
    heading = np.cumsum(yaw_rate) * dt
    x = np.cumsum(vx * np.cos(heading) - vy * np.sin(heading)) * dt
    y = np.cumsum(vx * np.sin(heading) + vy * np.cos(heading)) * dt
    return pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": y,
            "heading": heading,
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
            "steering": 0.1 * np.sin(t),
            "throttle": np.full(n, 0.5),
            "drive": np.full(n, 0.5),
            "brake": np.zeros(n),
            "regime": np.where(np.abs(yaw_rate) > 0.15, "corner", "straight"),
        }
    )


def test_window_shapes_and_dt_range():
    horizon = 5
    dt_range = (0.02, 0.05)
    windows = build_windows(
        [_episode()], horizon=horizon, dt_range=dt_range, stride=5, resample=True, seed=1
    )
    assert windows
    for w in windows:
        assert w.actions.shape == (horizon, 3)
        assert w.dts.shape == (horizon,)
        assert w.dyn_traj.shape == (horizon + 1, 3)
        assert w.pose_traj.shape == (horizon + 1, 3)
        assert dt_range[0] - 1e-9 <= w.dts[0] <= dt_range[1] + 1e-9
        assert w.regime in ("straight", "corner")


def test_non_resample_uses_native_dt():
    windows = build_windows([_episode(dt=0.01)], horizon=4, stride=10, resample=False)
    assert windows
    assert all(abs(w.dts[0] - 0.01) < 1e-6 for w in windows)


def test_split_by_episode_has_no_leakage():
    episodes = [_episode(), _episode(), _episode()]
    windows = build_windows(episodes, horizon=5, stride=5, seed=2)
    train, val, test = split_windows(
        windows, val_frac=0.34, test_frac=0.34, seed=3, by_episode=True
    )
    eps = [{w.episode for w in part} for part in (train, val, test)]
    assert eps[0].isdisjoint(eps[1])
    assert eps[0].isdisjoint(eps[2])
    assert eps[1].isdisjoint(eps[2])
    assert len(train) + len(val) + len(test) == len(windows)


def test_stack_windows_shapes():
    windows = build_windows([_episode()], horizon=5, stride=5)
    batch = stack_windows(windows)
    n = len(windows)
    assert batch["dyn0"].shape == (n, 3)
    assert batch["actions"].shape == (n, 5, 3)
    assert batch["dyn_traj"].shape == (n, 6, 3)
    assert batch["regime"].shape == (n,)


def test_dataset_spec_yaml_round_trip(tmp_path):
    spec = DatasetSpec(
        bags=["/data/bag_a", "/data/bag_b"],
        horizon=20,
        dt_range=(0.02, 0.05),
        regimes_included=["drift"],
        notes="drift-only set for tire-model probing",
        filters={"max_gap_s": 1.0, "sideslip_thresh": 0.2},
    )
    path = tmp_path / "dataset_spec.yaml"
    spec.to_yaml(str(path))
    loaded = DatasetSpec.from_yaml(str(path))
    assert loaded.bags == spec.bags
    assert loaded.dt_range == (0.02, 0.05)
    assert isinstance(loaded.dt_range, tuple)
    assert loaded.regimes_included == ["drift"]
    assert loaded.notes == spec.notes
    assert loaded.filters["sideslip_thresh"] == 0.2


def test_build_dataset_is_deprecated():
    df = _episode(n=50)
    with pytest.warns(DeprecationWarning):
        states, actions, deltas = build_dataset(df, dt=0.05)
    assert states.shape[1] == 3
    assert actions.shape[1] == 2
