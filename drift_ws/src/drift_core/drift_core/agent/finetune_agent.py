"""
Fine-tune a sim-trained SAC drift agent on recorded real-car transitions (off-board).

Offline continuation of training: warm-start from a simulation run, fill the replay
buffer with real ``(obs, action, reward, next_obs, done)`` transitions recorded on the
car by ``agent_pkg``'s ``data_collection_node`` (``.npz`` files), and run SAC gradient
updates against that buffer. No environment is stepped by default — the real data *is*
the experience — so this runs on any laptop/GPU box with the ``[rl]`` extra.

Usage (needs the ``[rl]`` extra)::

    python -m drift_core.agent.finetune_agent \
        --init-from drift_ws/runs/agent/<sim_run> --data recordings/*.npz --tag real-s3

Each run writes a self-contained ``drift_ws/runs/agent/<timestamp>__sac__<tag>/`` (see
:mod:`.run_dir`), same layout as :mod:`.train_agent`, with ``meta.json`` recording the
warm-start source, the data files, and the transition/gradient-step counts.

Note: pure-offline SAC can over-estimate values (no fresh on-policy data to correct the
critic). Keep the gradient-step budget and learning rate modest, and optionally pass
``--sim-rollout-steps`` to interleave a little simulation experience as a regularizer.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from ..paths import agent_runs_dir
from .environments.config import StageConfig
from .run_dir import create_run_dir, write_meta


def _resolve_data_files(patterns: list[str]) -> list[Path]:
    """Expand file/glob/dir arguments into a sorted list of ``.npz`` files."""
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(path.glob("*.npz")))
        elif any(ch in pattern for ch in "*?["):
            files.extend(sorted(Path(p) for p in glob.glob(pattern)))
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No .npz transition files matched: {patterns}")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"Transition files not found: {missing}")
    return files


def _load_transitions(files: list[Path]) -> dict:
    """Concatenate recorded transitions from one or more ``.npz`` files."""
    keys = ("observations", "next_observations", "actions", "rewards", "terminateds", "truncateds")
    parts: dict[str, list] = {k: [] for k in keys}
    for f in files:
        with np.load(f, allow_pickle=False) as data:
            for k in keys:
                parts[k].append(data[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _fill_replay_buffer(model, transitions: dict) -> int:
    """Add recorded transitions to the model's replay buffer; return the count."""
    obs = transitions["observations"]
    next_obs = transitions["next_observations"]
    actions = transitions["actions"]
    rewards = transitions["rewards"]
    terminateds = transitions["terminateds"]
    truncateds = transitions["truncateds"]
    n = len(rewards)
    for i in range(n):
        # done = episode ended (either terminal or truncated); the TimeLimit.truncated
        # flag tells SAC to still bootstrap on truncation (manual stop) but treat a real
        # terminal (off-track/stall) as absorbing. Shapes carry the single-env axis.
        done = bool(terminateds[i] or truncateds[i])
        model.replay_buffer.add(
            obs[i].reshape(1, -1),
            next_obs[i].reshape(1, -1),
            actions[i].reshape(1, -1),
            np.array([rewards[i]], dtype=np.float32),
            np.array([done]),
            [{"TimeLimit.truncated": bool(truncateds[i])}],
        )
    return n


def finetune(
    stage: StageConfig,
    init_from: str,
    data_files: list[Path],
    *,
    runs_dir=None,
    tag: str = "",
    seed: int = 0,
    gradient_steps: int = 20_000,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str = "auto",
    log_interval: int = 1_000,
    sim_rollout_steps: int = 0,
) -> Path:
    """Fine-tune ``init_from`` on recorded transitions; returns the run directory."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_device

    from .environments.drift_env import DriftEnv

    device = str(get_device(device))
    batch_size = batch_size or stage.train.batch_size

    run_dir = create_run_dir(runs_dir or agent_runs_dir(), "sac", tag or f"finetune_{stage.name}")
    stage.to_yaml(run_dir / "config.yaml")
    write_meta(
        run_dir,
        device=device,
        seed=seed,
        extra={
            "stage": stage.name,
            "init_from": str(init_from),
            "finetune": True,
            "data_files": [str(f) for f in data_files],
            "gradient_steps": gradient_steps,
        },
    )

    # DriftEnv only supplies the observation/action spaces (and, optionally, sim
    # rollouts) — the real data drives the offline updates.
    env = Monitor(DriftEnv(stage))
    checkpoint = Path(init_from)
    if checkpoint.is_dir():
        best = checkpoint / "best_model.zip"
        checkpoint = best if best.is_file() else checkpoint / "agent.zip"
    model = SAC.load(str(checkpoint), env=env, device=device, seed=seed)
    if learning_rate is not None:
        model.learning_rate = learning_rate
        model.lr_schedule = lambda _progress: learning_rate
    model.set_logger(configure(str(run_dir), ["csv", "stdout"]))

    n = _fill_replay_buffer(model, _load_transitions(data_files))
    print(f"Loaded {n} real transitions into the replay buffer")
    if n < batch_size:
        print(f"Warning: fewer transitions ({n}) than batch_size ({batch_size})")

    # Offline gradient updates against the real buffer, logged in chunks.
    done = 0
    while done < gradient_steps:
        chunk = min(log_interval, gradient_steps - done)
        model.train(gradient_steps=chunk, batch_size=batch_size)
        done += chunk
        model.logger.record("finetune/gradient_steps", done)
        model.logger.dump(step=done)

    # Optional: interleave a little simulation experience to anchor the critic.
    if sim_rollout_steps > 0:
        print(f"Interleaving {sim_rollout_steps} simulation steps to regularize")
        model.learn(total_timesteps=sim_rollout_steps, reset_num_timesteps=False)

    model.save(run_dir / "agent.zip")
    return run_dir


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a sim-trained SAC drift agent on recorded real transitions."
    )
    parser.add_argument(
        "--init-from", required=True, help="sim run dir (or SAC .zip) to warm-start from"
    )
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="recorded .npz transition file(s), glob(s), or directory(ies)",
    )
    parser.add_argument(
        "--config",
        help="stage name or YAML for reward/env config (default: the init-from run's config.yaml)",
    )
    parser.add_argument("--gradient-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, help="override the stage's batch size")
    parser.add_argument("--learning-rate", type=float, help="override the learning rate")
    parser.add_argument(
        "--sim-rollout-steps",
        type=int,
        default=0,
        help="optional simulation steps to interleave as a regularizer (experimental)",
    )
    parser.add_argument("--tag", default="", help="run-dir label (default: finetune_<stage>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--runs-dir", help="base directory for run dirs (default: drift_ws/runs/agent)"
    )
    args = parser.parse_args(argv)

    if args.config:
        stage = StageConfig.load(args.config)
    else:
        config_path = Path(args.init_from) / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No config.yaml in {args.init_from}; pass --config with a stage name or YAML"
            )
        stage = StageConfig.from_yaml(config_path)

    data_files = _resolve_data_files(args.data)
    run_dir = finetune(
        stage,
        args.init_from,
        data_files,
        runs_dir=args.runs_dir,
        tag=args.tag,
        seed=args.seed,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        sim_rollout_steps=args.sim_rollout_steps,
    )
    print(f"Fine-tune complete: {run_dir}")


if __name__ == "__main__":
    main()
