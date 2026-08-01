"""Tests for driving-regime labeling."""

import pandas as pd

from drift_core.vehicle_model.data.regimes import label_regimes, regime_counts


def test_labels_each_regime():
    df = pd.DataFrame(
        {
            "vx": [0.0, 2.0, 2.0, 2.0],
            "vy": [0.0, 0.0, 0.0, 1.0],  # last row: large sideslip
            "yaw_rate": [0.0, 0.0, 3.0, 0.0],  # row 2: cornering
        }
    )
    out = label_regimes(df, speed_min=0.3, yaw_thresh=2.0, sideslip_thresh=0.17)
    assert list(out["regime"]) == ["stationary", "straight", "corner", "drift"]


def test_drift_takes_priority_over_corner():
    # High yaw AND high sideslip -> drift wins.
    df = pd.DataFrame({"vx": [2.0], "vy": [1.5], "yaw_rate": [5.0]})
    out = label_regimes(df)
    assert out["regime"].iloc[0] == "drift"


def test_straight_reverse_is_not_drift():
    # Backing up in a straight line: arctan2(~0, -vx) is ~pi, which clears any
    # sideslip threshold. It is reverse, not drift.
    df = pd.DataFrame({"vx": [-1.5], "vy": [0.0], "yaw_rate": [0.0]})
    out = label_regimes(df)
    assert out["regime"].iloc[0] == "reverse"


def test_reverse_while_turning_is_still_reverse():
    # Cornering and drift are forward-motion verdicts; reversing is its own regime.
    df = pd.DataFrame({"vx": [-1.5], "vy": [0.3], "yaw_rate": [3.0]})
    out = label_regimes(df)
    assert out["regime"].iloc[0] == "reverse"


def test_slow_reverse_is_stationary():
    # Below speed_min the car is stationary regardless of direction.
    df = pd.DataFrame({"vx": [-0.1], "vy": [0.0], "yaw_rate": [0.0]})
    out = label_regimes(df, speed_min=0.3)
    assert out["regime"].iloc[0] == "stationary"


def test_regime_counts_zero_filled():
    df = pd.DataFrame({"vx": [2.0, 2.0], "vy": [0.0, 0.0], "yaw_rate": [0.0, 0.0]})
    counts = regime_counts(label_regimes(df))
    assert counts["straight"] == 2
    assert counts["drift"] == 0
    assert set(counts) >= {"stationary", "straight", "corner", "drift"}
