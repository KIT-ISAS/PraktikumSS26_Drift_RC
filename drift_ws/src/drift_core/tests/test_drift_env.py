"""Tests for DriftEnv (skipped without the [rl] extra's gymnasium)."""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from drift_core.environments import StageConfig  # noqa: E402
from drift_core.environments.drift_env import DriftEnv  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

INFO_KEYS = {
    "s",
    "lap",
    "lateral_error",
    "heading_error",
    "speed",
    "sideslip",
    "x",
    "y",
    "progress_m",
}


@pytest.fixture()
def env():
    return DriftEnv("stage1_path_follow")


def test_env_checker_passes(env):
    check_env(env, skip_render_check=True)


def test_seeded_reset_is_deterministic(env):
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env.reset(seed=7)
    assert np.array_equal(obs_a, obs_b)
    obs_c, _ = env.reset(seed=8)
    assert not np.array_equal(obs_a, obs_c)


def test_seeded_rollout_is_deterministic(env):
    def rollout():
        env.reset(seed=3)
        trace = []
        for i in range(50):
            action = np.array([np.sin(0.1 * i), 0.4], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            trace.append((obs.copy(), reward, terminated, truncated))
            if terminated or truncated:
                break
        return trace

    first, second = rollout(), rollout()
    assert len(first) == len(second)
    for (obs_a, r_a, term_a, trunc_a), (obs_b, r_b, term_b, trunc_b) in zip(first, second):
        assert np.array_equal(obs_a, obs_b)
        assert r_a == r_b and term_a == term_b and trunc_a == trunc_b


def test_fresh_model_per_reset(env):
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.array([0.0, 1.0], dtype=np.float32))  # full throttle builds slip
    model_before = env._model
    assert model_before._slip > 0.0
    env.reset(seed=0)
    assert env._model is not model_before
    assert env._model._slip == 0.0


def test_off_track_terminates(env):
    env.reset(seed=1, options={"spawn_s": 0.0, "deterministic": True})
    terminated = truncated = False
    info = {}
    # Drive straight at full throttle: the figure-8 curves away underneath the car.
    for _ in range(500):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated and info["termination_reason"] == "off_track"


def test_stall_terminates():
    stage = StageConfig.load("stage1_path_follow")
    stage.env.spawn.speed = 0.0
    env = DriftEnv(stage)
    env.reset(seed=0, options={"deterministic": True})
    limit = int(stage.env.termination.stall_timeout_s / stage.env.dt) + 5
    terminated = False
    info = {}
    for _ in range(limit):
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        if terminated:
            break
    assert terminated and info["termination_reason"] == "stall"


def test_truncates_at_max_episode_steps():
    stage = StageConfig.load("stage1_path_follow")
    stage.env.max_episode_steps = 40
    stage.env.termination.lateral_limit_m = 100.0  # never leave "the track"
    stage.env.termination.stall_timeout_s = 1e9
    env = DriftEnv(stage)
    env.reset(seed=0)
    for i in range(40):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.3], dtype=np.float32))
        assert not terminated
    assert truncated and i == 39


def test_observation_and_info_contract(env):
    obs, info = env.reset(seed=5)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert INFO_KEYS <= set(info)
    obs, reward, _, _, info = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    assert INFO_KEYS <= set(info)
