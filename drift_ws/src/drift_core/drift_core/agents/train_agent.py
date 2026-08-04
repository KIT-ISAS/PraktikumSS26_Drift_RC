"""
Train the SAC drift agent in simulation, one curriculum stage per run.

Usage (needs the ``[rl]`` extra)::

    python -m drift_core.agents.train_agent --config stage1_path_follow --tag s1
    python -m ...train_agent --config stage2_fast --init-from <stage1_run> --tag s2
    python -m ...train_agent --config stage3_drift --init-from <stage2_run> --tag s3

Each run writes a self-contained ``drift_ws/runs/agent/<timestamp>__sac__<tag>/`` (see
:mod:`.run_dir`). ``--init-from`` warm-starts the policy/critic weights from a
previous stage's run; the replay buffer is deliberately NOT carried over — the
stages reward differently, and stale rewards would poison the new critic.
"""

import argparse
import copy
from pathlib import Path

from ..environments.config import StageConfig
from ..paths import agent_runs_dir
from .run_dir import create_run_dir, write_meta


def _eval_stage(stage: StageConfig) -> StageConfig:
    """A copy of the stage with spawn noise zeroed, for deterministic evaluation."""
    eval_stage = copy.deepcopy(stage)
    eval_stage.env.spawn.speed_noise_std = 0.0
    eval_stage.env.spawn.lateral_noise_std = 0.0
    eval_stage.env.spawn.heading_noise_std = 0.0
    return eval_stage


def train(
    stage: StageConfig,
    *,
    runs_dir: str | Path | None = None,
    tag: str = "",
    seed: int = 0,
    init_from: str | None = None,
    total_steps: int | None = None,
) -> Path:
    """Train one curriculum stage; returns the run directory."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_device

    from ..environments.drift_env import DriftEnv

    cfg = stage.train
    steps = total_steps if total_steps is not None else cfg.total_steps
    device = str(get_device(cfg.device))

    run_dir = create_run_dir(runs_dir or agent_runs_dir(), "sac", tag or stage.name)
    stage.to_yaml(run_dir / "config.yaml")
    write_meta(
        run_dir,
        device=device,
        seed=seed,
        extra={"stage": stage.name, "init_from": str(init_from) if init_from else None},
    )

    env = Monitor(DriftEnv(stage))

    if init_from:
        checkpoint = Path(init_from)
        if checkpoint.is_dir():
            best = checkpoint / "best_model.zip"
            checkpoint = best if best.is_file() else checkpoint / "agent.zip"
        model = SAC.load(str(checkpoint), env=env, device=device, seed=seed)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            tau=cfg.tau,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            learning_starts=cfg.learning_starts,
            ent_coef=cfg.ent_coef,
            policy_kwargs={"net_arch": list(cfg.net_arch)},
            seed=seed,
            device=device,
        )
    model.set_logger(configure(str(run_dir), ["csv", "stdout"]))

    callbacks = []
    if cfg.eval_freq > 0:
        eval_env = Monitor(DriftEnv(_eval_stage(stage)))
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=cfg.eval_freq,
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
            )
        )

    model.learn(total_timesteps=steps, callback=callbacks or None)
    model.save(run_dir / "agent.zip")
    return run_dir


def main(argv=None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the SAC drift agent (one stage).")
    parser.add_argument(
        "--config",
        required=True,
        help="packaged stage name (stage1_path_follow | stage2_fast | stage3_drift) or YAML path",
    )
    parser.add_argument("--init-from", help="previous run dir (or SAC .zip) to warm-start from")
    parser.add_argument("--total-steps", type=int, help="override the stage's total_steps")
    parser.add_argument("--tag", default="", help="run-dir label (default: the stage name)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--runs-dir", help="base directory for run dirs (default: drift_ws/runs/agent)"
    )
    args = parser.parse_args(argv)

    stage = StageConfig.load(args.config)
    run_dir = train(
        stage,
        runs_dir=args.runs_dir,
        tag=args.tag,
        seed=args.seed,
        init_from=args.init_from,
        total_steps=args.total_steps,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
