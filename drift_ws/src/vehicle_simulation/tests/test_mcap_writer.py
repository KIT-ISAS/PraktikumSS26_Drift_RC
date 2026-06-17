"""Phase 4: recorded-vs-predicted MCAP writing (rosbags only, no torch)."""

import numpy as np

from vehicle_simulation.evaluation.mcap_writer import write_eval_mcap


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


def test_rollout_from_recording_shapes():
    import pandas as pd

    from vehicle_simulation.evaluation.mcap_writer import rollout_from_recording
    from vehicle_simulation.models import build_model

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
