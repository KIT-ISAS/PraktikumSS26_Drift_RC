"""
Evaluate a trained drift agent: roll deterministic episodes, report metrics.

Usage::

    python -m drift_core.agents.evaluate_agent --run-dir drift_ws/runs/agent/<id> [--episodes 10]

Writes ``<run_dir>/eval/metrics.json`` and ``<run_dir>/eval/trajectories.npz``
(per-step arrays concatenated over episodes with an ``episode`` index column) and,
when matplotlib is available, a trajectory-over-centerline plot.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..environments.config import StageConfig
from .policy import DriftAgent

TRAJECTORY_FIELDS = (
    "episode",
    "t",
    "x",
    "y",
    "s",
    "speed",
    "lateral_error",
    "heading_error",
    "sideslip",
    "steering",
    "throttle",
    "reward",
)


def evaluate_agent(
    run_dir,
    config: StageConfig | str | None = None,
    episodes: int = 10,
    deterministic: bool = True,
    out_dir=None,
    seed: int = 0,
) -> dict:
    """
    Roll ``episodes`` evaluation episodes and return a metrics dict.

    ``config`` defaults to the run's own ``config.yaml``; spawn noise is left as
    configured there (use a modified config for fully deterministic spawns).
    """
    from ..environments.drift_env import DriftEnv

    run_dir = Path(run_dir)
    if config is None:
        config = run_dir / "config.yaml"
    if not isinstance(config, StageConfig):
        config = StageConfig.load(str(config))
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = DriftAgent(run_dir)
    env = DriftEnv(config)
    dt = config.env.dt

    rows = {field: [] for field in TRAJECTORY_FIELDS}
    episode_rewards = []
    episode_laps = []
    term_reasons: dict[str, int] = {}
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        step = 0
        while True:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            for field, value in (
                ("episode", episode),
                ("t", step * dt),
                ("x", info["x"]),
                ("y", info["y"]),
                ("s", info["s"]),
                ("speed", info["speed"]),
                ("lateral_error", info["lateral_error"]),
                ("heading_error", info["heading_error"]),
                ("sideslip", info["sideslip"]),
                ("steering", float(action[0])),
                ("throttle", float(action[1])),
                ("reward", reward),
            ):
                rows[field].append(value)
            step += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                term_reasons[reason] = term_reasons.get(reason, 0) + 1
                break
        episode_rewards.append(total_reward)
        episode_laps.append(info["progress_m"] / env.total_length)

    lateral = np.array(rows["lateral_error"])
    sideslip = np.array(rows["sideslip"])
    # Drifting = sideslip within one sigma of the stage's drift bonus target.
    reward_cfg = config.env.reward
    drift_threshold = max(reward_cfg.drift_beta_target - reward_cfg.drift_beta_sigma, 0.0)
    metrics = {
        "episodes": episodes,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_laps_per_episode": float(np.mean(episode_laps)),
        "total_laps": float(np.sum(episode_laps)),
        "mean_speed": float(np.mean(rows["speed"])),
        "mean_abs_lateral_error": float(np.mean(np.abs(lateral))),
        "rms_lateral_error": float(np.sqrt(np.mean(lateral**2))),
        "mean_abs_sideslip": float(np.mean(np.abs(sideslip))),
        "max_abs_sideslip": float(np.max(np.abs(sideslip))),
        "drift_threshold_rad": drift_threshold,
        "drift_fraction": float(np.mean(np.abs(sideslip) > drift_threshold)),
        "termination_reasons": term_reasons,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "trajectories.npz", **{k: np.asarray(v) for k, v in rows.items()}
    )
    _plot(env, rows, out_dir / "trajectories.png")
    return metrics


def _plot(env, rows: dict, out_path: Path) -> bool:
    """Plot driven trajectories over the centerline; False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    centerline = env.waypoints
    ax.plot(centerline[:, 0], centerline[:, 1], "k--", lw=1, alpha=0.5, label="centerline")
    episode = np.asarray(rows["episode"])
    x, y = np.asarray(rows["x"]), np.asarray(rows["y"])
    for ep in np.unique(episode):
        mask = episode == ep
        ax.plot(x[mask], y[mask], lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a trained drift agent.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", help="stage name/YAML override (default: run's config.yaml)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true", help="sample the policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    metrics = evaluate_agent(
        args.run_dir,
        config=args.config,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
