"""
Detect and cut out localization discontinuities ("Vive solver jumps").

The Vive tracker occasionally jumps (solver error), producing physically
impossible position/heading steps. These corrupt finite-difference velocities and
any model trained on them, so we detect transitions that violate plausible motion
bounds for their timestep and split the episode at each jump, keeping the clean
sub-segments.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class JumpReport:
    """Summary of discontinuities found and samples dropped during sanitizing."""

    n_episodes_in: int = 0
    n_episodes_out: int = 0
    n_transitions: int = 0
    n_flagged: int = 0
    n_dropped_samples: int = 0
    reasons: dict = field(
        default_factory=lambda: {"position": 0, "heading": 0, "accel": 0, "dt": 0}
    )

    def as_dict(self) -> dict:
        """Return a plain-dict view (for YAML/JSON serialization)."""
        return {
            "n_episodes_in": self.n_episodes_in,
            "n_episodes_out": self.n_episodes_out,
            "n_transitions": self.n_transitions,
            "n_flagged": self.n_flagged,
            "n_dropped_samples": self.n_dropped_samples,
            "reasons": dict(self.reasons),
        }


def detect_discontinuities(
    df: pd.DataFrame,
    *,
    max_speed: float = 8.0,
    max_accel: float | None = None,
    max_yaw_rate: float = 25.0,
    pos_margin: float = 0.05,
    head_margin: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """
    Flag discontinuous transitions in a single (time-ordered) episode.

    Returns a boolean mask over the ``len(df) - 1`` transitions (True = jump) and a
    per-reason count dict. A transition is flagged when the position step or heading
    step exceeds what ``max_speed`` / ``max_yaw_rate`` allow for its timestep, or
    when ``dt <= 0``. The acceleration check (``max_accel``) is **opt-in**: at high
    tracker rates raw finite-difference acceleration is noise-dominated and would
    flag good data, so it is off by default — real solver jumps show up as position
    and heading jumps. Set ``max_accel`` only on suitably smoothed/low-rate data.
    """
    reasons = {"position": 0, "heading": 0, "accel": 0, "dt": 0}
    if len(df) < 2:
        return np.zeros(max(len(df) - 1, 0), dtype=bool), reasons

    t = df["t"].to_numpy()
    dt = np.diff(t)
    dpos = np.hypot(np.diff(df["x"].to_numpy()), np.diff(df["y"].to_numpy()))
    dhead = np.abs(np.diff(np.unwrap(df["heading"].to_numpy())))

    bad_dt = dt <= 0
    safe_dt = np.where(bad_dt, np.inf, dt)  # avoid div/compare issues on bad dt
    pos_bad = dpos > (max_speed * safe_dt + pos_margin)
    head_bad = dhead > (max_yaw_rate * safe_dt + head_margin)

    accel_bad = np.zeros(len(dt), dtype=bool)
    if max_accel is not None and "vx" in df.columns and "vy" in df.columns:
        speed = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy())
        accel_bad = np.abs(np.diff(speed)) > (max_accel * safe_dt)

    reasons["dt"] = int(bad_dt.sum())
    reasons["position"] = int(pos_bad.sum())
    reasons["heading"] = int(head_bad.sum())
    reasons["accel"] = int(accel_bad.sum())

    mask = bad_dt | pos_bad | head_bad | accel_bad
    return mask, reasons


def split_on_discontinuities(
    df: pd.DataFrame, mask: np.ndarray, min_len: int = 10
) -> list[pd.DataFrame]:
    """
    Cut an episode at flagged transitions, dropping the offending step.

    ``mask[i]`` flags the transition between rows ``i`` and ``i+1``; the episode is
    split there so neither corrupt step survives. Sub-segments shorter than
    ``min_len`` rows are discarded.
    """
    if len(df) == 0:
        return []
    bounds = [0, *(np.where(mask)[0] + 1).tolist(), len(df)]
    segments = [
        df.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True) for i in range(len(bounds) - 1)
    ]
    return [seg for seg in segments if len(seg) >= min_len]


def sanitize_episodes(
    episodes: list[pd.DataFrame], *, min_len: int = 10, **detect_kwargs
) -> tuple[list[pd.DataFrame], JumpReport]:
    """
    Run discontinuity detection + splitting over a list of episodes.

    Returns the cleaned sub-episodes and an aggregate :class:`JumpReport`.
    ``detect_kwargs`` are forwarded to :func:`detect_discontinuities`.
    """
    report = JumpReport(n_episodes_in=len(episodes))
    clean: list[pd.DataFrame] = []
    for ep in episodes:
        mask, reasons = detect_discontinuities(ep, **detect_kwargs)
        report.n_transitions += len(mask)
        report.n_flagged += int(mask.sum())
        for k, v in reasons.items():
            report.reasons[k] += v
        segments = split_on_discontinuities(ep, mask, min_len=min_len)
        kept = sum(len(s) for s in segments)
        report.n_dropped_samples += len(ep) - kept
        clean.extend(segments)
    report.n_episodes_out = len(clean)
    return clean, report
