"""Phase 4: recorded-vs-predicted MCAP writing (rosbags only, no torch)."""

import numpy as np

from drift_core.evaluation.mcap_writer import write_eval_mcap


def test_write_and_reopen_mcap(tmp_path):
    n = 20
    t = np.arange(n) * 0.05
    recorded = np.stack([t, np.zeros(n), np.zeros(n)], axis=-1)
    predicted = recorded + 0.1  # constant offset
    actions = np.zeros((n, 2))

    out = write_eval_mcap(str(tmp_path / "eval.mcap"), t, recorded, predicted, actions=actions)

    from rosbags.rosbag2 import Reader

    with Reader(out) as reader:
        topics = {c.topic: c.msgcount for c in reader.connections}

    assert "/eval/pose_recorded" in topics
    assert "/eval/pose_predicted" in topics
    assert "/eval/path_recorded" in topics
    assert "/eval/path_predicted" in topics
    assert topics["/eval/pose_recorded"] == n
    assert topics["/eval/error_position"] == n
    assert topics["/eval/path_recorded"] == 1


def test_write_mcap_with_sampled_fan(tmp_path):
    n = 20
    t = np.arange(n) * 0.05
    recorded = np.stack([t, np.zeros(n), np.zeros(n)], axis=-1)
    predicted = recorded + 0.1
    rng = np.random.default_rng(0)
    samples = [predicted + rng.normal(0, 0.05, size=predicted.shape) for _ in range(4)]

    out = write_eval_mcap(str(tmp_path / "fan.mcap"), t, recorded, predicted, samples=samples)

    from rosbags.rosbag2 import Reader

    with Reader(out) as reader:
        topics = {c.topic: c.msgcount for c in reader.connections}
    for k in range(4):
        assert topics[f"/eval/path_sample_{k:02d}"] == 1
    assert topics["/eval/sample_spread"] == n


def test_sampled_fan_reslice_follows_sanitize_truncation():
    # The fan re-slices df to len(times); this only lines up because sanitize can
    # only *truncate* the window. Inject a localization jump so the mean path is
    # cut short, then check every sampled path matches that shorter window. Uses a
    # torch-free model, so the spread collapses onto the mean — shapes are the point.
    import pandas as pd

    from drift_core.evaluation.mcap_writer import sampled_rollout_fan
    from drift_core.models.kinematic import KinematicBicycleModel

    n = 30
    t = np.arange(n) * 0.02
    x = t.astype(float)
    x[15:] += 5.0  # 5 m step in one 20 ms sample -> flagged as a jump at transition 14
    df = pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": np.zeros(n),
            "heading": np.zeros(n),
            "vx": np.ones(n),
            "vy": np.zeros(n),
            "yaw_rate": np.zeros(n),
            "steering": np.zeros(n),
            "throttle": np.full(n, 0.3),
        }
    )
    model = KinematicBicycleModel()
    times, rec, pred, actions, sampled = sampled_rollout_fan(
        model, df, start_idx=0, horizon=100, samples=3
    )
    assert len(times) < 100  # sanitize truncated well before the requested horizon
    assert len(sampled) == 3
    assert all(s.shape == pred.shape for s in sampled)
    assert all(s.shape[0] == len(times) for s in sampled)


def test_rollout_from_recording_shapes():
    import pandas as pd

    from drift_core.evaluation.mcap_writer import rollout_from_recording
    from drift_core.models import build_model

    n = 50
    t = np.arange(n) * 0.02
    # Constant 1 m/s drive along +x — velocities recorded, as read from /car/odom.
    df = pd.DataFrame(
        {
            "t": t,
            "x": t * 1.0,
            "y": np.zeros(n),
            "heading": np.zeros(n),
            "vx": np.ones(n),
            "vy": np.zeros(n),
            "yaw_rate": np.zeros(n),
            "steering": np.zeros(n),
            "throttle": np.full(n, 0.5),
        }
    )
    model = build_model("kinematic")
    times, rec, pred, actions = rollout_from_recording(model, df, start_idx=0, horizon=20)
    assert rec.shape == pred.shape == (21, 3)
    assert times.shape == (21,)


def _glitchy_df(jump_at: int, n: int = 50):
    import pandas as pd

    t = np.arange(n) * 0.02
    x = t * 1.0
    x[jump_at:] += 5.0  # a 5 m teleport: ~250 m/s over one 20 ms step
    speed = np.ones(n)
    speed[jump_at] = 60.0  # the corrupt finite-differenced twist on the jumped row
    return pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": np.zeros(n),
            "heading": np.zeros(n),
            "vx": speed,
            "vy": np.zeros(n),
            "yaw_rate": np.zeros(n),
            "steering": np.zeros(n),
            "throttle": np.full(n, 0.5),
        }
    )


def test_rollout_truncates_at_a_localization_jump():
    # A tracker jump mid-window must not be integrated across: the clean rollout ends
    # at the jump instead of "predicting" the teleport.
    from drift_core.evaluation.mcap_writer import rollout_from_recording
    from drift_core.models import build_model

    df = _glitchy_df(jump_at=10)
    model = build_model("kinematic")
    times, rec, pred, _ = rollout_from_recording(model, df, start_idx=0, horizon=40)
    assert len(rec) == len(pred) == 10  # stops before the jump at row 10
    assert np.isfinite(pred).all()


def test_rollout_refuses_to_seed_from_a_glitch():
    # Seeding the rollout from the corrupted sample itself is the failure that made a
    # good model look broken; it must raise, not silently roll from a 60 m/s state.
    import pytest

    from drift_core.evaluation.mcap_writer import rollout_from_recording
    from drift_core.models import build_model

    df = _glitchy_df(jump_at=10)
    model = build_model("kinematic")
    with pytest.raises(ValueError, match="localization jump"):
        rollout_from_recording(model, df, start_idx=9, horizon=20)
    # ...but the escape hatch still rolls over it.
    times, rec, pred, _ = rollout_from_recording(
        model, df, start_idx=9, horizon=20, sanitize=False
    )
    assert len(pred) == 21
