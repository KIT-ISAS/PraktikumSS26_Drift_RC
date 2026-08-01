"""
Evaluate and plot every training run under a runs directory.

Usage::

    python -m drift_core.agents.analyze_runs [--runs-root <dir>] [--episodes 5]

For each run directory (recognized by its ``config.yaml`` + ``progress.csv``) this rolls
deterministic evaluation episodes via :mod:`.evaluate_agent` when ``eval/metrics.json``
is missing (``--re-evaluate`` forces a fresh roll), then writes into
``<runs-root>/analysis/``:

- ``learning_curves.png`` — per-run training-episode reward and eval-callback reward
- ``trajectories.png`` — per-run evaluation trajectories over the track centerline
- ``summary.csv`` — one row of evaluation metrics per run (also printed)
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..environments.config import StageConfig
from ..paths import agent_runs_dir
from ..track import load_track
from .evaluate_agent import evaluate_agent

# Reference dataviz palette (categorical slots in fixed, CVD-validated order + chrome).
CATEGORICAL = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")
BLUE, AQUA = CATEGORICAL[0], CATEGORICAL[1]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"

SUMMARY_METRICS = (
    "mean_episode_reward",
    "mean_laps_per_episode",
    "mean_speed",
    "rms_lateral_error",
    "mean_abs_sideslip",
    "drift_fraction",
)


def discover_runs(runs_root) -> list[Path]:
    """Return run directories under ``runs_root`` (those with config.yaml + progress.csv)."""
    return sorted(
        d
        for d in Path(runs_root).iterdir()
        if d.is_dir() and (d / "config.yaml").is_file() and (d / "progress.csv").is_file()
    )


def load_run(run_dir: Path, episodes: int, re_evaluate: bool) -> dict:
    """
    Load one run's artifacts, evaluating the agent first if needed.

    Returns a dict with the run label, stage name, centerline waypoints, the
    ``progress.csv`` frame, the eval-callback arrays, and (when evaluation
    succeeded) the rollout metrics and trajectories.
    """
    config = StageConfig.load(str(run_dir / "config.yaml"))
    waypoints, _, _ = load_track(config.env.track).discretize(config.env.points_per_segment)

    metrics_path = run_dir / "eval" / "metrics.json"
    if re_evaluate or not metrics_path.is_file():
        print(f"evaluating {run_dir.name} ({episodes} episodes) ...")
        try:
            evaluate_agent(run_dir, episodes=episodes)
        except Exception as exc:  # noqa: BLE001 — one broken checkpoint must not kill the sweep
            print(f"  evaluation failed: {exc}")

    run = {
        "name": run_dir.name.split("__")[-1],
        "stage": config.name,
        "waypoints": np.asarray(waypoints),
        "progress": pd.read_csv(run_dir / "progress.csv"),
        "evaluations": None,
        "metrics": None,
        "trajectories": None,
    }
    eval_npz = run_dir / "eval" / "evaluations.npz"
    if eval_npz.is_file():
        with np.load(eval_npz) as data:
            run["evaluations"] = {k: data[k] for k in ("timesteps", "results")}
    if metrics_path.is_file():
        run["metrics"] = json.loads(metrics_path.read_text())
    traj_npz = run_dir / "eval" / "trajectories.npz"
    if traj_npz.is_file():
        with np.load(traj_npz) as data:
            run["trajectories"] = {k: data[k] for k in ("episode", "x", "y")}
    return run


def _apply_style() -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )


def _panel_grid(n: int, panel_size=(5.2, 2.8)):
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def plot_learning_curves(runs: list[dict], out_path: Path) -> None:
    """One panel per run: raw + smoothed training reward and the eval-callback reward band."""
    fig, axes = _panel_grid(len(runs))
    for ax, run in zip(axes, runs):
        progress = run["progress"]
        steps = progress["time/total_timesteps"]
        reward = progress["rollout/ep_rew_mean"]
        window = max(5, len(progress) // 25)
        ax.plot(steps, reward, color=BLUE, lw=1, alpha=0.3)
        ax.plot(
            steps,
            reward.rolling(window, min_periods=1).mean(),
            color=BLUE,
            lw=2,
            label="train reward (smoothed)",
        )
        if run["evaluations"] is not None:
            ev_steps, results = run["evaluations"]["timesteps"], run["evaluations"]["results"]
            ax.fill_between(
                ev_steps, results.min(axis=1), results.max(axis=1), color=AQUA, alpha=0.18, lw=0
            )
            ax.plot(ev_steps, results.mean(axis=1), color=AQUA, lw=2, label="eval reward (mean)")
        ax.set_title(f"{run['name']} — {run['stage']}", loc="left")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("episode reward")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle("Training progress per run", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_trajectories(runs: list[dict], out_path: Path) -> None:
    """One panel per run: deterministic evaluation trajectories over the centerline."""
    fig, axes = _panel_grid(len(runs), panel_size=(4.6, 3.4))
    for ax, run in zip(axes, runs):
        waypoints = run["waypoints"]
        ax.plot(waypoints[:, 0], waypoints[:, 1], ls="--", lw=1, color=MUTED, label="centerline")
        traj = run["trajectories"]
        if traj is None:
            ax.text(0.5, 0.5, "no eval data", transform=ax.transAxes, ha="center", color=INK_2)
        else:
            for i, episode in enumerate(np.unique(traj["episode"])):
                mask = traj["episode"] == episode
                color = CATEGORICAL[i % len(CATEGORICAL)]
                ax.plot(traj["x"][mask], traj["y"][mask], lw=1.2, color=color)
                ax.plot(traj["x"][mask][0], traj["y"][mask][0], "o", ms=4, color=color)
        title = f"{run['name']} — {run['stage']}"
        if run["metrics"] is not None:
            reasons = ", ".join(
                f"{n}× {r}" for r, n in sorted(run["metrics"]["termination_reasons"].items())
            )
            title += f"\n{run['metrics']['episodes']} episodes: {reasons}"
        ax.set_title(title, loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    n_episodes = max(
        (len(np.unique(r["trajectories"]["episode"])) for r in runs if r["trajectories"]),
        default=0,
    )
    handles = [plt.Line2D([], [], color=MUTED, ls="--", lw=1)]
    handles += [
        plt.Line2D([], [], color=CATEGORICAL[i % len(CATEGORICAL)], lw=1.5)
        for i in range(n_episodes)
    ]
    labels = ["centerline", *(f"episode {i}" for i in range(n_episodes))]
    fig.legend(handles, labels, loc="upper right", ncols=4)
    fig.suptitle("Evaluation trajectories (dot = spawn)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def summarize(runs: list[dict], out_path: Path) -> pd.DataFrame:
    """Write and return one summary row per run (training length + evaluation metrics)."""
    rows = []
    for run in runs:
        row = {
            "run": run["name"],
            "stage": run["stage"],
            "timesteps": int(run["progress"]["time/total_timesteps"].iloc[-1]),
        }
        metrics = run["metrics"] or {}
        row.update({key: metrics.get(key) for key in SUMMARY_METRICS})
        row["terminations"] = ", ".join(
            f"{n}× {r}" for r, n in sorted(metrics.get("termination_reasons", {}).items())
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    return summary


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate and plot all training runs.")
    parser.add_argument(
        "--runs-root", help="directory of run dirs to analyze (default: drift_ws/runs/agent)"
    )
    parser.add_argument("--out-dir", help="output directory (default: <runs-root>/analysis)")
    parser.add_argument("--episodes", type=int, default=5, help="evaluation episodes per run")
    parser.add_argument(
        "--re-evaluate", action="store_true", help="re-roll eval episodes even if metrics exist"
    )
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root) if args.runs_root else agent_runs_dir()
    run_dirs = discover_runs(runs_root)
    if not run_dirs:
        raise SystemExit(f"no run directories found under {runs_root}")
    runs = [load_run(d, args.episodes, args.re_evaluate) for d in run_dirs]

    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    plot_learning_curves(runs, out_dir / "learning_curves.png")
    plot_trajectories(runs, out_dir / "trajectories.png")
    summary = summarize(runs, out_dir / "summary.csv")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_dir}/learning_curves.png, trajectories.png, summary.csv")


if __name__ == "__main__":
    main()
