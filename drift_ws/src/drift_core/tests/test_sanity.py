"""Tests for localization-jump (Vive solver error) detection and cutting."""

import numpy as np
import pandas as pd

from drift_core.data.sanity import (
    detect_discontinuities,
    sanitize_episodes,
    split_on_discontinuities,
)


def _line_df(n: int = 60, dt: float = 0.02, speed: float = 1.0) -> pd.DataFrame:
    t = np.arange(n) * dt
    return pd.DataFrame({"t": t, "x": speed * t, "y": np.zeros(n), "heading": np.zeros(n)})


def test_clean_data_has_no_jumps():
    mask, reasons = detect_discontinuities(_line_df())
    assert not mask.any()
    assert sum(reasons.values()) == 0


def test_position_jump_flagged_exactly_once():
    df = _line_df()
    df.loc[30:, "x"] += 5.0  # 5 m teleport at the 29->30 transition
    mask, reasons = detect_discontinuities(df)
    assert mask.sum() == 1
    assert mask[29]
    assert reasons["position"] == 1


def test_heading_jump_flagged():
    df = _line_df()
    df.loc[30:, "heading"] += 1.0  # 1 rad jump
    mask, reasons = detect_discontinuities(df)
    assert mask[29]
    assert reasons["heading"] >= 1


def test_accel_spike_flagged_when_velocities_present():
    df = _line_df()
    df["vx"] = 1.0
    df["vy"] = 0.0
    df.loc[30, "vx"] = 11.0  # velocity spike -> up and back down
    # The accel check is opt-in (off by default at high tracker rates).
    _, off = detect_discontinuities(df)
    assert off["accel"] == 0
    _, on = detect_discontinuities(df, max_accel=30.0)
    assert on["accel"] >= 1


def test_split_and_sanitize_keeps_clean_segments():
    df = _line_df(n=60)
    df.loc[30:, "x"] += 5.0
    mask, _ = detect_discontinuities(df)
    segments = split_on_discontinuities(df, mask, min_len=10)
    assert len(segments) == 2
    assert all(len(s) >= 10 for s in segments)

    clean, report = sanitize_episodes([df], min_len=10)
    assert report.n_flagged == 1
    assert report.n_episodes_out == 2
    assert report.n_episodes_in == 1
    d = report.as_dict()
    assert d["reasons"]["position"] == 1
