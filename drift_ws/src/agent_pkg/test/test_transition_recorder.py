"""The shared transition buffer must persist exactly what finetune_agent expects."""

import numpy as np
import pytest

from agent_pkg.lib.transition_recorder import TransitionRecorder

OBS_DIM = 18


def _recorder(tmp_path, **kwargs):
    defaults = {"rate_hz": 50.0, "stage_name": "stage3_drift", "run_dir": "human"}
    return TransitionRecorder(tmp_path / "session.npz", **defaults | kwargs)


def _obs(fill):
    return np.full(OBS_DIM, fill, dtype=np.float32)


def test_arm_exposes_pending_fields(tmp_path):
    rec = _recorder(tmp_path)
    assert not rec.pending
    rec.arm(_obs(0.0), [0.5, 0.2], [0.1, 0.0], distance=3.25)
    assert rec.pending
    np.testing.assert_array_equal(rec.pending_action, np.array([0.5, 0.2], dtype=np.float32))
    np.testing.assert_array_equal(rec.pending_prev_action, np.array([0.1, 0.0], dtype=np.float32))
    assert rec.pending_distance == pytest.approx(3.25)
    assert len(rec) == 0


def test_complete_appends_and_clears_pending(tmp_path):
    rec = _recorder(tmp_path)
    rec.arm(_obs(1.0), [0.5, 0.2], [0.0, 0.0], distance=0.0)
    rec.complete(reward=0.7, next_obs=_obs(2.0), terminated=False)
    assert not rec.pending
    assert len(rec) == 1


def test_drop_pending_discards_without_appending(tmp_path):
    rec = _recorder(tmp_path)
    rec.arm(_obs(1.0), [0.5, 0.2], [0.0, 0.0], distance=0.0)
    rec.drop_pending()
    assert not rec.pending
    assert len(rec) == 0


def test_truncate_last_skips_terminated(tmp_path):
    rec = _recorder(tmp_path)
    rec.arm(_obs(1.0), [0.5, 0.2], [0.0, 0.0], distance=0.0)
    rec.complete(reward=-20.0, next_obs=_obs(2.0), terminated=True)
    rec.truncate_last()
    rec.flush("figure_8")
    with np.load(rec.out_path, allow_pickle=False) as data:
        assert bool(data["terminateds"][0])
        assert not bool(data["truncateds"][0])


def test_truncate_last_marks_non_terminal(tmp_path):
    rec = _recorder(tmp_path)
    rec.arm(_obs(1.0), [0.5, 0.2], [0.0, 0.0], distance=0.0)
    rec.complete(reward=0.7, next_obs=_obs(2.0), terminated=False)
    rec.truncate_last()
    rec.flush("figure_8")
    with np.load(rec.out_path, allow_pickle=False) as data:
        assert bool(data["truncateds"][0])


def test_truncate_last_on_empty_buffer_is_noop(tmp_path):
    _recorder(tmp_path).truncate_last()


def test_flush_empty_writes_nothing(tmp_path):
    rec = _recorder(tmp_path)
    assert rec.flush("figure_8") == 0
    assert not rec.out_path.exists()


def test_flush_roundtrip_matches_finetune_schema(tmp_path):
    rec = _recorder(tmp_path)
    rec.metadata["gyro_mode"] = "soft"
    rec.metadata["gyro_gain"] = np.float32(0.8)
    for i in range(3):
        rec.arm(_obs(float(i)), [0.1 * i, 0.2], [0.0, 0.0], distance=float(i))
        rec.complete(reward=float(i), next_obs=_obs(float(i + 1)), terminated=i == 2)
    assert rec.flush("figure_8") == 3

    with np.load(rec.out_path, allow_pickle=False) as data:
        # The keys finetune_agent._load_transitions reads, with matching shapes.
        assert data["observations"].shape == (3, OBS_DIM)
        assert data["next_observations"].shape == (3, OBS_DIM)
        assert data["actions"].shape == (3, 2)
        assert data["observations"].dtype == np.float32
        assert data["rewards"].dtype == np.float32
        np.testing.assert_array_equal(data["terminateds"], [False, False, True])
        np.testing.assert_array_equal(data["truncateds"], [False, False, False])
        np.testing.assert_array_equal(data["next_observations"][:-1], data["observations"][1:])
        assert float(data["rate_hz"]) == 50.0
        assert str(data["run_dir"]) == "human"
        assert str(data["stage_name"]) == "stage3_drift"
        assert str(data["track_name"]) == "figure_8"
        assert str(data["gyro_mode"]) == "soft"
        assert float(data["gyro_gain"]) == pytest.approx(0.8)


def test_flush_overwrites_with_growing_buffer(tmp_path):
    rec = _recorder(tmp_path)
    rec.arm(_obs(0.0), [0.0, 0.0], [0.0, 0.0], distance=0.0)
    rec.complete(reward=0.0, next_obs=_obs(1.0), terminated=False)
    assert rec.flush("figure_8") == 1
    rec.arm(_obs(1.0), [0.0, 0.0], [0.0, 0.0], distance=0.1)
    rec.complete(reward=0.0, next_obs=_obs(2.0), terminated=False)
    assert rec.flush("figure_8") == 2
    with np.load(rec.out_path, allow_pickle=False) as data:
        assert len(data["rewards"]) == 2
    assert not rec.out_path.with_name(rec.out_path.name + ".tmp").exists()
