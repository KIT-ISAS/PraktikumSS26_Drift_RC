r"""
Train an RL agent on the DriftEnv using SB3 SAC.

Usage:
    # Kinematic model (no weights needed):
    python -m vehicle_simulation.training.train_agent \
        --model kinematic \
        --track path/to/figure_8.yaml \
        --out weights/agent.zip

    # Neural model:
    python -m vehicle_simulation.training.train_agent \
        --model neural \
        --model-weights weights/neural_model.pt \
        --track path/to/figure_8.yaml \
        --out weights/agent.zip \
        --n-envs 8
"""

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from ..environments.drift_env import DriftEnv
from ..models import build_model
from ..track import load_track


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL agent on DriftEnv with SB3 SAC")
    p.add_argument("--model", choices=["kinematic", "neural"], default="kinematic")
    p.add_argument("--model-config", default=None, help="YAML params file for kinematic model")
    p.add_argument("--model-weights", default=None, help=".pt checkpoint for neural model")
    p.add_argument("--track", required=True, help="Track name (e.g. figure_8) or YAML path")
    p.add_argument("--out", required=True, help="Output path for trained agent (.zip)")
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--dt", type=float, default=0.02, help="Simulation timestep (s)")
    p.add_argument("--max-laps", type=int, default=3)
    p.add_argument("--stochastic", action="store_true", help="Use stochastic model steps")
    return p.parse_args()


def _make_env_fn(
    model_type: str,
    model_config: str | None,
    model_weights: str | None,
    track_file: str,
    dt: float,
    max_laps: int,
    stochastic: bool,
):
    """Return a no-arg callable that creates a DriftEnv (for VecEnv factories)."""

    def _init() -> DriftEnv:
        track = load_track(track_file)
        extra = {"stochastic": stochastic} if model_type == "neural" else {}
        model = build_model(model_type, config=model_config, weights=model_weights, **extra)
        return DriftEnv(model, track, dt=dt, max_laps=max_laps, stochastic=stochastic)

    return _init


def main() -> None:
    args = _parse_args()

    env_fn = _make_env_fn(
        model_type=args.model,
        model_config=args.model_config,
        model_weights=args.model_weights,
        track_file=args.track,
        dt=args.dt,
        max_laps=args.max_laps,
        stochastic=args.stochastic,
    )

    if args.n_envs > 1:
        env = SubprocVecEnv([env_fn for _ in range(args.n_envs)])
    else:
        env = env_fn()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "tb_logs"

    agent = SAC("MlpPolicy", env, verbose=1, tensorboard_log=str(log_dir))
    agent.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    agent.save(out_path)
    print(f"Agent saved to {out_path}")


if __name__ == "__main__":
    main()
