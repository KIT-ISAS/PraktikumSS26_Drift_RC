"""
End-to-end smoke test: tiny SAC training run, reload, inference, evaluation.

Skipped without the [rl] extra (stable-baselines3).
"""

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from drift_core.agents.evaluate_agent import evaluate_agent  # noqa: E402
from drift_core.agents.policy import DriftAgent  # noqa: E402
from drift_core.agents.train_agent import train  # noqa: E402
from drift_core.environments import OBS_DIM, StageConfig  # noqa: E402
from drift_core.models.base import VehicleAction  # noqa: E402


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """One tiny training run shared by all assertions in this module."""
    stage = StageConfig.load("stage1_path_follow")
    stage.env.max_episode_steps = 60
    stage.train.total_steps = 200
    stage.train.learning_starts = 50
    stage.train.buffer_size = 1000
    stage.train.batch_size = 32
    stage.train.net_arch = [32, 32]
    stage.train.eval_freq = 100
    stage.train.n_eval_episodes = 1
    stage.train.device = "cpu"
    run_dir = train(stage, runs_dir=tmp_path_factory.mktemp("runs"), tag="smoke", seed=0)
    return run_dir


def test_run_dir_artifacts(tiny_run):
    assert (tiny_run / "config.yaml").is_file()
    assert (tiny_run / "agent.zip").is_file()
    assert (tiny_run / "progress.csv").is_file()
    meta = json.loads((tiny_run / "meta.json").read_text())
    assert meta["seed"] == 0
    assert "git_sha" in meta and meta["stage"] == "stage1_path_follow"
    # The run's config reloads into an equal StageConfig.
    reloaded = StageConfig.from_yaml(tiny_run / "config.yaml")
    assert reloaded.name == "stage1_path_follow"
    assert reloaded.train.total_steps == 200


def test_drift_agent_roundtrip(tiny_run):
    agent = DriftAgent(tiny_run)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = agent.act(obs)
    assert isinstance(action, VehicleAction)
    assert -1.0 <= action.steering <= 1.0
    assert -1.0 <= action.throttle <= 1.0
    # Deterministic: the same observation maps to the same action.
    assert np.array_equal(agent.predict(obs), agent.predict(obs))


def test_warm_start_from_previous_run(tiny_run, tmp_path):
    stage = StageConfig.load("stage2_fast")
    stage.env.max_episode_steps = 60
    stage.train.total_steps = 100
    stage.train.eval_freq = 0
    run2 = train(stage, runs_dir=tmp_path, tag="smoke2", seed=1, init_from=tiny_run)
    assert (run2 / "agent.zip").is_file()
    meta = json.loads((run2 / "meta.json").read_text())
    assert meta["init_from"] == str(tiny_run)


def test_evaluate_agent(tiny_run):
    metrics = evaluate_agent(tiny_run, episodes=1, seed=0)
    for key in (
        "mean_episode_reward",
        "mean_laps_per_episode",
        "mean_speed",
        "rms_lateral_error",
        "mean_abs_sideslip",
        "drift_fraction",
        "termination_reasons",
    ):
        assert key in metrics
    assert (tiny_run / "eval" / "metrics.json").is_file()
    npz = np.load(tiny_run / "eval" / "trajectories.npz")
    assert {"episode", "x", "y", "s", "reward"} <= set(npz.files)
    assert len(npz["x"]) > 0
