"""Phase 2: a trained run dir reloads and reproduces predictions (torch-guarded)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from drift_core.data.spec import DatasetSpec  # noqa: E402
from drift_core.models import build_model  # noqa: E402
from drift_core.models.base import VehicleAction, VehicleState  # noqa: E402
from drift_core.models.loader import load_model_from_run  # noqa: E402
from drift_core.training.config import TrainConfig  # noqa: E402
from drift_core.training.trainer import Trainer  # noqa: E402


def _train(tmp_path, linear_windows):
    train, val = linear_windows
    cfg = TrainConfig(
        model_type="neural",
        model={"hidden_dim": 32},
        epochs=3,
        batch_size=64,
        val_every=1,
        device="cpu",
        tag="rt",
    )
    spec = DatasetSpec(bags=["synthetic"], horizon=5)
    return Trainer(cfg, train, val, spec, run_base=str(tmp_path)).fit()


def _same_step(a, b):
    s = VehicleState(vx=1.2, vy=-0.1, yaw_rate=0.3)
    act = VehicleAction(steering=0.4, throttle=0.6)
    ra, rb = a.step(s, act, 0.03), b.step(s, act, 0.03)
    for attr in ("x", "y", "heading", "vx", "vy", "yaw_rate"):
        assert getattr(ra, attr) == pytest.approx(getattr(rb, attr), abs=1e-5)


def test_load_model_from_run_reproduces(tmp_path, linear_windows):
    run_dir = _train(tmp_path, linear_windows)
    reloaded = load_model_from_run(str(run_dir))
    # A second reload must be identical to the first.
    _same_step(reloaded, load_model_from_run(str(run_dir)))


def test_build_model_resolves_run_dir(tmp_path, linear_windows):
    run_dir = _train(tmp_path, linear_windows)
    model = build_model(str(run_dir))
    out = model.step(VehicleState(vx=1.0), VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()


def test_build_model_run_dir_kwarg(tmp_path, linear_windows):
    run_dir = _train(tmp_path, linear_windows)
    model = build_model("neural", run_dir=str(run_dir))
    assert model.step(VehicleState(vx=1.0), VehicleAction(0.0, 0.5), 0.02) is not None
