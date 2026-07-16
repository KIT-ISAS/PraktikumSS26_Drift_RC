"""
File-based training-run directory: the single self-contained artifact of a run.

Layout::

    runs/<timestamp>__sac__<tag>/
      config.yaml          resolved StageConfig of the run
      meta.json            git sha, host, device, seed, warm-start source
      progress.csv         SB3 logger output (rollout/eval metrics over time)
      agent.zip            final SAC policy
      best_model.zip       best eval-callback policy (when eval is enabled)
      eval/                evaluation outputs (metrics.json, trajectories.npz)

No external tracking service; everything needed to understand, reload, and
reproduce a run lives in the directory (GPU-server / no-ROS friendly).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys


def create_run_dir(base, kind: str = "sac", tag: str = "") -> Path:
    """Create and return ``base/<timestamp>__<kind>__<tag>/`` (+ ``eval/``)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}__{kind}" + (f"__{tag}" if tag else "")
    run_dir = Path(base) / name
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_meta(run_dir: Path, *, device: str, seed: int, extra: dict | None = None) -> None:
    """Write ``meta.json`` (git sha, host, device, versions, seed, timestamp)."""
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "device": device,
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        meta.update(extra)
    (Path(run_dir) / "meta.json").write_text(json.dumps(meta, indent=2))


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
