"""
Serializable dataset specification + the end-to-end "synthesize a dataset" builder.

A :class:`DatasetSpec` captures *exactly* how a dataset was made (which bags, which
sanity thresholds, which regimes, horizon, dt range, splits, seed, free-text notes)
so a run is reproducible and self-documenting. :func:`build_dataset_from_spec` runs
the whole pipeline: read bags -> segment -> body-frame velocities -> sanity-cut
localization jumps -> label regimes -> window (variable dt) -> split.
"""

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import yaml

from .dataset import Window, build_windows, split_windows, stack_windows
from .preprocessing import segment_episodes, split_contiguous
from .regimes import label_regimes, regime_counts
from .rosbag_reader import read_driving_data
from .sanity import sanitize_episodes

# gyro_gain at or below this counts as "gyro off" (the commanded gain is a float).
GYRO_OFF_MAX = 0.01


@dataclass
class DatasetSpec:
    """Reproducible recipe for a windowed training dataset (+ computed stats)."""

    bags: list[str]
    car: str = "drift_rc"
    horizon: int = 20
    dt_range: tuple[float, float] = (0.02, 0.05)
    resample: bool = True
    stride: int = 5
    regimes_included: list[str] | None = None  # keep only windows whose regime is in this set
    val_frac: float = 0.15
    test_frac: float = 0.0
    seed: int = 0
    notes: str = ""
    # filters: max_gap_s, min_len, sanity thresholds, regime thresholds (all optional).
    filters: dict = field(default_factory=dict)
    # --- computed at build time (None until built) ---
    n_windows: int | None = None
    n_windows_per_regime: dict | None = None
    jump_report: dict | None = None
    created: str | None = None
    spec_version: int = 1

    def to_dict(self) -> dict:
        """Return a plain-dict view with dt_range as a list (YAML-friendly)."""
        d = asdict(self)
        d["dt_range"] = list(self.dt_range)
        return d

    @staticmethod
    def from_dict(data: dict) -> "DatasetSpec":
        """Build a spec from a parsed mapping (dt_range list -> tuple)."""
        data = dict(data)
        if "dt_range" in data and data["dt_range"] is not None:
            data["dt_range"] = tuple(data["dt_range"])
        return DatasetSpec(**data)

    def to_yaml(self, path: str) -> None:
        """Write the spec to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def from_yaml(path: str) -> "DatasetSpec":
        """Read a spec from a YAML file."""
        with open(path) as f:
            return DatasetSpec.from_dict(yaml.safe_load(f))


def _sanity_kwargs(filters: dict) -> dict:
    keys = ("max_speed", "max_accel", "max_yaw_rate", "pos_margin", "head_margin")
    return {k: filters[k] for k in keys if k in filters}


def default_filters() -> dict:
    """Reasonable filter defaults for the ~100 Hz canonical odom data (tune per dataset)."""
    return {"max_gap_s": 1.0, "min_len": 20}


def _regime_kwargs(filters: dict) -> dict:
    keys = ("speed_min", "yaw_thresh", "sideslip_thresh")
    return {k: filters[k] for k in keys if k in filters}


def build_dataset_from_spec(
    spec: DatasetSpec,
) -> tuple[list[Window], list[Window], list[Window], DatasetSpec]:
    """
    Run the full pipeline for ``spec`` and return ``(train, val, test, spec_out)``.

    ``spec_out`` is a copy of ``spec`` with computed stats filled in
    (``n_windows``, ``n_windows_per_regime``, ``jump_report``, ``created``). Use it
    as the run's saved ``dataset_spec.yaml``.
    """
    max_gap_s = spec.filters.get("max_gap_s", 1.0)
    min_len = spec.filters.get("min_len", 10)

    # 1. read + segment into continuous episodes. Body velocities come recorded
    #    in /car/odom, so nothing is re-derived from the pose here.
    #
    #    With the gyro_off filter, spans driven with the steering-stabilizer gyro
    #    active are excluded (the gyro alters the steering->servo map, so gyro-on
    #    data teaches a different plant). They are *split out*, not dropped in
    #    place: the gain is adjusted in 0.1 steps from the d-pad, so dialling it
    #    down walks through a run of short intermediate-gain spans, each far below
    #    max_gap_s. Row-dropping would leave those as sub-second holes inside a
    #    single "continuous" episode, and build_windows would interpolate straight
    #    across them — fabricating dynamics over exactly the interval that was
    #    excluded for having a different plant. Splitting makes each hole an
    #    episode boundary instead.
    gyro_off = bool(spec.filters.get("gyro_off", False))
    episodes = []
    for bag in spec.bags:
        df = read_driving_data(bag, max_gap_s=max_gap_s)
        spans = (
            split_contiguous(df, df["gyro_gain"].to_numpy() <= GYRO_OFF_MAX) if gyro_off else [df]
        )
        for span in spans:
            episodes.extend(segment_episodes(span, max_gap_s=max_gap_s))

    # 2. cut localization jumps, then label regimes.
    episodes, jump_report = sanitize_episodes(
        episodes, min_len=min_len, **_sanity_kwargs(spec.filters)
    )
    labeled = [label_regimes(ep, **_regime_kwargs(spec.filters)) for ep in episodes]

    # 3. window (variable dt) and optionally keep only chosen regimes.
    windows = build_windows(
        labeled,
        horizon=spec.horizon,
        dt_range=spec.dt_range,
        stride=spec.stride,
        resample=spec.resample,
        seed=spec.seed,
    )
    if spec.regimes_included:
        keep = set(spec.regimes_included)
        windows = [w for w in windows if w.regime in keep]

    # 4. split (leakage-free by episode by default).
    train, val, test = split_windows(
        windows, val_frac=spec.val_frac, test_frac=spec.test_frac, seed=spec.seed
    )

    spec_out = copy.deepcopy(spec)
    spec_out.n_windows = len(windows)
    spec_out.n_windows_per_regime = _count_window_regimes(windows)
    spec_out.jump_report = jump_report.as_dict()
    spec_out.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return train, val, test, spec_out


def _count_window_regimes(windows: list[Window]) -> dict:
    counts: dict[str, int] = {}
    for w in windows:
        counts[w.regime] = counts.get(w.regime, 0) + 1
    return counts


# regime_counts / stack_windows re-exported for convenience.
__all__ = [
    "DatasetSpec",
    "build_dataset_from_spec",
    "regime_counts",
    "stack_windows",
]
