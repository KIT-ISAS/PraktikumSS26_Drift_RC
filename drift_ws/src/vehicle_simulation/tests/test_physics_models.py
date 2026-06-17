"""Phase 3: dynamic bicycle + physics-informed hybrid models (torch-guarded)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from vehicle_simulation.data.spec import DatasetSpec  # noqa: E402
from vehicle_simulation.models import build_model  # noqa: E402
from vehicle_simulation.models.base import VehicleAction, VehicleState  # noqa: E402
from vehicle_simulation.models.loader import load_model_from_run  # noqa: E402
from vehicle_simulation.training.config import TrainConfig  # noqa: E402
from vehicle_simulation.training.trainer import Trainer  # noqa: E402

_NORM = {
    "input_mean": np.zeros(5),
    "input_std": np.ones(5),
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


def test_hybrid_starts_as_pure_physics():
    from vehicle_simulation.models.dynamic import DynamicBicycleModel
    from vehicle_simulation.models.hybrid import PhysicsInformedModel

    hybrid = PhysicsInformedModel.new(_NORM, hidden_dim=16)
    physics = DynamicBicycleModel(hybrid._physics._car, hybrid._physics._p)
    s = VehicleState(vx=2.0, vy=0.2, yaw_rate=0.3)
    a = VehicleAction(0.4, 0.5)
    h = hybrid.step(s, a, 0.03)
    p = physics.step(s, a, 0.03)
    for attr in ("vx", "vy", "yaw_rate"):
        assert getattr(h, attr) == pytest.approx(getattr(p, attr), abs=1e-6)


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
    run_dir = _train(tmp_path, linear_windows, "hybrid", {"hidden_dim": 32})
    assert (run_dir / "model.pt").exists()
    model = build_model(str(run_dir))
    out = model.step(VehicleState(vx=1.0), VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()


def test_hybrid_stochastic_nll_trains(tmp_path, linear_windows):
    run_dir = _train(
        tmp_path, linear_windows, "hybrid", {"hidden_dim": 32, "stochastic": True}, loss="nll"
    )
    assert (run_dir / "model.pt").exists()
