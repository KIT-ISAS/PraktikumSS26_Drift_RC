"""
Dynamic bicycle + hybrid models through the Trainer (torch-guarded).

The hybrid model's own behavioural tests live in test_hybrid.py; here it is exercised
end to end through the training pipeline.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from drift_core.data.spec import DatasetSpec  # noqa: E402
from drift_core.models import build_model  # noqa: E402
from drift_core.models.base import VehicleAction, VehicleState  # noqa: E402
from drift_core.models.loader import load_model_from_run  # noqa: E402
from drift_core.training.config import TrainConfig  # noqa: E402
from drift_core.training.trainer import Trainer  # noqa: E402

_NORM = {
    "input_mean": np.zeros(6),
    "input_std": np.ones(6),
    "output_mean": np.zeros(3),
    "output_std": np.ones(3),
    "dyn_std": np.ones(3),
}


def test_dynamic_builds_and_steps():
    model = build_model("dynamic")
    out = model.step(VehicleState(vx=2.0, vy=0.1, yaw_rate=0.2), VehicleAction(0.3, 0.6), 0.02)
    assert np.isfinite([out.x, out.y, out.heading, out.vx, out.vy, out.yaw_rate]).all()


def test_dynamic_stable_at_low_speed():
    # Slip-angle denominator is floored, so stepping from rest stays finite.
    model = build_model("dynamic")
    out = model.step(VehicleState(), VehicleAction(1.0, 1.0), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()


def test_dynamic_straight_reverse_settles():
    # Straight-line reverse must damp back to vy = r = 0. A sign-preserving vx in
    # the slip-angle denominator puts alpha near ±π, saturating the tire curve at
    # full grip with a sign that flips with vy: a sustained bang-bang limit cycle.
    # Its period is 4 steps, so it aliases away under stroboscopic sampling —
    # every step must be inspected, and the assertion is on the decay envelope.
    import torch

    model = build_model("dynamic")
    state = torch.tensor([[-1.0, 0.05, 0.0]])  # reversing at 1 m/s, lateral perturbation
    action = torch.tensor([[0.0, -0.3, 0.0]])  # straight, light reverse throttle
    vy = []
    with torch.no_grad():
        for _ in range(60):
            state = state + 0.02 * model.derivative(state, action).mean
            vy.append(abs(float(state[0, 1])))

    assert max(vy[-10:]) < 0.1 * max(vy[:10])  # the perturbation decays...
    assert max(vy[-10:]) < 1e-2  # ...to standstill, rather than holding a limit cycle


def _train(tmp_path, windows, model_type, model_kwargs=None, loss="mse"):
    train, val = windows
    cfg = TrainConfig(
        model_type=model_type,
        model=model_kwargs or {},
        loss=loss,
        epochs=3,
        batch_size=64,
        lr=1e-2,
        val_every=1,
        device="cpu",
        tag=model_type,
    )
    spec = DatasetSpec(bags=["synthetic"], horizon=5)
    return Trainer(cfg, train, val, spec, run_base=str(tmp_path)).fit()


def test_dynamic_trains_and_reloads(tmp_path, linear_windows):
    run_dir = _train(tmp_path, linear_windows, "dynamic")
    assert (run_dir / "model.pt").exists()
    model = load_model_from_run(str(run_dir))
    out = model.step(VehicleState(vx=1.0), VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()


def test_hybrid_trains_and_reloads(tmp_path, linear_windows):
    run_dir = _train(tmp_path, linear_windows, "hybrid", {"n_terms": 3})
    assert (run_dir / "model.pt").exists()
    model = build_model(str(run_dir))
    out = model.step(VehicleState(vx=1.0), VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()
