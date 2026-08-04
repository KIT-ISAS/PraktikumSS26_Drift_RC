"""
File-based training-run directory: the single self-contained artifact of a run.

Layout::

    drift_ws/runs/vehicle_model/<timestamp>__<model_type>__<tag>/
      config.yaml          dataset_spec.yaml      notes.md
      model.pt             model_last.pt          metrics.csv
      loss_curve.png       meta.json
      eval/                (filled by the evaluation suite)

No external tracking service; everything needed to understand, reload, and
reproduce a run lives in the directory (GPU-server / no-ROS friendly).
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys


def create_run_dir(base, model_type: str, tag: str = "") -> Path:
    """Create and return ``base/<timestamp>__<model_type>__<tag>/`` (+ ``eval/``)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}__{model_type}" + (f"__{tag}" if tag else "")
    run_dir = Path(base) / name
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_meta(run_dir: Path, *, device: str, seed: int, extra: dict | None = None) -> None:
    """Write meta.json (git sha, host, device, versions, seed, timestamp)."""
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "device": device,
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        pass
    if extra:
        meta.update(extra)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def write_notes(run_dir: Path, notes: str) -> None:
    """Write the free-text notes.md."""
    (run_dir / "notes.md").write_text(notes or "")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


class MetricsLogger:
    """Append per-epoch metrics to ``metrics.csv`` and keep history for plotting."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.csv"
        self.history: list[dict] = []

    def log(self, row: dict) -> None:
        """
        Record one row and rewrite the CSV.

        Rewriting (rather than appending) lets later rows introduce new columns —
        e.g. ``val_loss`` and per-regime metrics first appear on a validation epoch.
        """
        self.history.append(row)
        keys: list[str] = []
        for record in self.history:
            for key in record:
                if key not in keys:
                    keys.append(key)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for record in self.history:
                writer.writerow({k: record.get(k, "") for k in keys})

    def plot(self, out_path: Path) -> bool:
        """Plot train/val loss curves; returns False if matplotlib is unavailable."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        if not self.history:
            return False
        epochs = [r["epoch"] for r in self.history]
        nan = float("nan")
        fig, ax = plt.subplots()
        if any("train_loss" in r for r in self.history):
            ax.plot(epochs, [r.get("train_loss", nan) for r in self.history], label="train")
        val = [
            (r["epoch"], r["val_loss"])
            for r in self.history
            if r.get("val_loss") not in (None, "")
        ]
        if val:
            ax.plot([e for e, _ in val], [v for _, v in val], label="val", marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
