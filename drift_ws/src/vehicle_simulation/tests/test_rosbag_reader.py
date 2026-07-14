"""Tests for the ESC replay that labels recorded pedal commands (drive / brake)."""

import numpy as np
import pandas as pd
import pytest

from vehicle_simulation.data.rosbag_reader import _label_motor_commands


def _stream(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Build a merged-odom-style frame from (t_s, throttle, vx) rows."""
    t_s, throttle, vx = zip(*rows)
    return pd.DataFrame(
        {
            "t_ns": (np.asarray(t_s) * 1e9).astype(np.int64),
            "throttle": np.asarray(throttle, dtype=float),
            "vx": np.asarray(vx, dtype=float),
        }
    )


def test_negative_pedal_is_brake_while_locked():
    # Forward throttle sets the lockout, so the negative pedal that follows brakes.
    df = _stream([(0.00, 0.5, 1.0), (0.01, -0.4, 1.0)])
    out = _label_motor_commands(df, max_gap_s=1.0)
    assert out["brake"].iloc[1] == pytest.approx(0.4)
    assert out["drive"].iloc[1] == 0.0


def test_neutral_at_standstill_clears_lockout_into_reverse():
    # Forward -> coast to a stop with a neutral pedal -> the next negative pedal
    # is reverse, not brake. This is the transition the replay exists to resolve.
    df = _stream([(0.00, 0.5, 1.0), (0.01, 0.0, 0.0), (0.02, -0.4, 0.0)])
    out = _label_motor_commands(df, max_gap_s=1.0)
    assert out["drive"].iloc[2] == pytest.approx(-0.4)
    assert out["brake"].iloc[2] == 0.0


def test_prefix_before_first_forward_throttle_is_trimmed():
    # The lockout state before the first forward throttle is unknowable.
    df = _stream([(0.00, -0.3, 0.0), (0.01, 0.5, 0.5)])
    out = _label_motor_commands(df, max_gap_s=1.0)
    assert len(out) == 1
    assert out["drive"].iloc[0] == pytest.approx(0.5)


def test_lockout_does_not_carry_across_an_observation_gap():
    # Locked before a 5 s tracking dropout. The car may well have rolled to a stop
    # with a neutral pedal inside the gap (which clears the lockout), so the flag
    # must not survive it: the post-gap span re-synchronizes from its own first
    # forward throttle, and its rows before that are dropped.
    df = _stream(
        [
            (0.00, 0.5, 1.0),  # locks
            (0.01, 0.5, 1.0),
            (5.10, -0.4, 0.0),  # after the gap — unknowable, must be trimmed
            (5.11, 0.5, 0.2),  # re-syncs the span
            (5.12, -0.4, 1.0),  # now legitimately a brake
        ]
    )
    out = _label_motor_commands(df, max_gap_s=1.0)
    assert len(out) == 4  # the post-gap negative pedal is gone
    assert out["brake"].iloc[-1] == pytest.approx(0.4)
    # Continuity of the surviving rows: the pre-gap span is untouched.
    assert out["drive"].iloc[0] == pytest.approx(0.5)


def test_span_that_never_drives_forward_is_dropped():
    df = _stream([(0.00, 0.5, 1.0), (5.10, -0.4, 0.0), (5.11, -0.4, 0.0)])
    out = _label_motor_commands(df, max_gap_s=1.0)
    assert len(out) == 1  # only the pre-gap span survives


def test_bag_with_no_forward_throttle_at_all_raises():
    df = _stream([(0.00, -0.3, 0.0), (0.01, -0.3, 0.0)])
    with pytest.raises(ValueError, match="ever drives forward"):
        _label_motor_commands(df, max_gap_s=1.0)
