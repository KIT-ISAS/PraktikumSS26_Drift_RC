"""Tests for the curriculum stage config schema and the packaged stage YAMLs."""

import pytest

from drift_core.environments.config import (
    EnvConfig,
    StageConfig,
    TrainConfig,
    list_stages,
)


def test_yaml_roundtrip(tmp_path):
    cfg = StageConfig(name="roundtrip")
    cfg.env.reward.drift_weight = 1.5
    cfg.train.total_steps = 123
    path = tmp_path / "stage.yaml"
    cfg.to_yaml(path)
    loaded = StageConfig.from_yaml(path)
    assert loaded == cfg


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="Unknown"):
        StageConfig.from_dict({"name": "x", "bogus": 1})
    with pytest.raises(ValueError, match="Unknown"):
        EnvConfig.from_dict({"reward": {"not_a_weight": 1.0}})
    with pytest.raises(ValueError, match="Unknown"):
        StageConfig.from_dict({"name": "x", "train": {"nope": 2}})


def test_packaged_stages_load_and_form_a_curriculum():
    names = list_stages()
    assert names == ["stage1_path_follow", "stage2_fast", "stage3_drift"]
    stages = [StageConfig.load(name) for name in names]
    s1, s2, s3 = stages
    for stage, name in zip(stages, names):
        assert stage.name == name
        assert stage.env.dt == 0.02
        assert isinstance(stage.train, TrainConfig)
    # Drift bonus only in stage 3; target speed increases across the curriculum.
    assert s1.env.reward.drift_weight == 0.0 and s2.env.reward.drift_weight == 0.0
    assert s3.env.reward.drift_weight > 0.0
    speeds = [s.env.reward.target_speed for s in stages]
    assert speeds == sorted(speeds) and speeds[0] < speeds[-1]


def test_load_unknown_stage_lists_available():
    with pytest.raises(FileNotFoundError, match="stage1_path_follow"):
        StageConfig.load("no_such_stage")
