"""
Dynamic bicycle + hybrid models through the Trainer (torch-guarded).

The hybrid model's own behavioural tests live in test_hybrid.py; here it is exercised
end to end through the training pipeline.
"""

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from drift_core.vehicle_model import build_model  # noqa: E402
from drift_core.vehicle_model.data.spec import DatasetSpec  # noqa: E402
from drift_core.vehicle_model.models.base import VehicleAction, VehicleState  # noqa: E402
from drift_core.vehicle_model.models.loader import load_model_from_run  # noqa: E402
from drift_core.vehicle_model.training.config import TrainConfig  # noqa: E402
from drift_core.vehicle_model.training.trainer import Trainer  # noqa: E402

_NORM = {
    "input_mean": np.zeros(6),
    "input_std": np.ones(6),
    "output_mean": np.zeros(3),
    "output_std": np.ones(3),
    "dyn_std": np.ones(3),
}


def test_inv_softplus_saturates_without_overflowing():
    from drift_core.vehicle_model.models.dynamic import _inv_softplus

    # math.expm1 overflows well below this; softplus(x) ~ x in the large-arg regime,
    # so the identity branch must return the argument unchanged and stay finite.
    assert _inv_softplus(1e6) == 1e6
    # ... while staying exact where softplus actually curves: softplus(_inv_softplus(y)) == y.
    y = 0.5
    x = _inv_softplus(y)
    assert abs(math.log1p(math.exp(x)) - y) < 1e-12


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
    run_dir = _train(tmp_path, linear_windows, "hybrid")
    assert (run_dir / "model.pt").exists()
    model = build_model(str(run_dir))
    out = model.step(VehicleState(vx=1.0), VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()


def test_sampled_rollout_propagates_noise_through_the_dynamics():
    # The particle estimator's core: with sample=True the rollout is Euler–Maruyama,
    # so two runs differ and the deterministic rollout stays exactly repeatable.
    import torch

    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel
    from drift_core.vehicle_model.training.trainer import rollout

    model = PhysicsInformedModel.new(_NORM, stochastic=True)
    dyn0 = torch.tensor([[1.0, 0.2, 0.5]])
    actions = torch.zeros((1, 10, 3))
    actions[..., 1] = 0.6
    dts = torch.full((1, 10), 0.02)

    with torch.no_grad():
        det1, _ = rollout(model, dyn0, actions, dts)
        det2, _ = rollout(model, dyn0, actions, dts)
        s1, _ = rollout(model, dyn0, actions, dts, sample=True)
        s2, _ = rollout(model, dyn0, actions, dts, sample=True)
    assert torch.equal(det1, det2), "deterministic rollout must be repeatable"
    assert not torch.allclose(s1, s2), "sampled rollouts must differ"
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all()


def test_sigma_only_calibration_freezes_the_physics(tmp_path, linear_windows):
    # train_scope="diffusion" + init_from: the particle NLL may only move the
    # diffusion head; every physics parameter must come out bit-identical.
    import torch

    run1 = _train(tmp_path, linear_windows, "hybrid", {"stochastic": True}, loss="nll")
    train, val = linear_windows
    cfg = TrainConfig(
        model_type="hybrid",
        loss="nll",
        nll_particles=2,
        init_from=str(run1),
        train_scope="diffusion",
        epochs=2,
        batch_size=64,
        lr=1e-2,
        val_every=1,
        device="cpu",
        tag="sigma-cal",
    )
    run2 = Trainer(cfg, train, val, DatasetSpec(bags=[]), run_base=str(tmp_path)).fit()

    before = load_model_from_run(str(run1))
    after = load_model_from_run(str(run2))
    for name in ("_p", "_front", "_rear", "_nn0"):
        for (k1, v1), (k2, v2) in zip(
            getattr(before, name).state_dict().items(), getattr(after, name).state_dict().items()
        ):
            assert k1 == k2
            assert torch.equal(v1, v2), f"physics parameter {name}.{k1} moved during Sigma-cal"


def test_stochastic_hybrid_trains_with_nll_and_reloads(tmp_path, linear_windows):
    # The paper's SDE path end to end: diffusion head + NLL loss through the Trainer.
    run_dir = _train(tmp_path, linear_windows, "hybrid", {"stochastic": True}, loss="nll")
    assert (run_dir / "model.pt").exists()
    model = load_model_from_run(str(run_dir), stochastic=True)
    out = model.step_stochastic(VehicleState(vx=1.0), VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([out.vx, out.vy, out.yaw_rate]).all()
