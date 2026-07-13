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


def test_hybrid_builds_and_steps():
    from vehicle_simulation.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM)
    out = model.step(VehicleState(vx=2.0, vy=0.2, yaw_rate=0.3), VehicleAction(0.4, 0.5), 0.03)
    assert np.isfinite([out.x, out.y, out.heading, out.vx, out.vy, out.yaw_rate]).all()


def test_hybrid_stable_from_rest():
    from vehicle_simulation.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM)
    s = VehicleState()
    for _ in range(50):
        s = model.step(s, VehicleAction(1.0, 1.0), 0.02)
    assert np.isfinite([s.vx, s.vy, s.yaw_rate]).all()
    assert s.vx > 0.1  # full throttle from rest must move the car forward


def test_hybrid_steering_at_standstill_does_not_turn():
    # A parked car steered at full lock must stay put: non-rolling wheels
    # produce no slip force (regression: the slip-angle vx floor used to
    # fabricate a front force from the steering angle alone).
    from vehicle_simulation.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM)
    s = VehicleState()
    for _ in range(100):
        s = model.step(s, VehicleAction(1.0, 0.0), 0.02)
    assert abs(s.yaw_rate) < 1e-3
    assert abs(s.vy) < 1e-3


def test_hybrid_servo_lag_delays_yaw_response():
    # After a steering step from straight-line driving, the yaw response builds
    # over the servo time constant instead of appearing instantly.
    from vehicle_simulation.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM)
    s = VehicleState(vx=2.0)
    s = model.step(s, VehicleAction(0.0, 0.3), 0.02)  # settle latents on neutral steering
    first = model.step(s, VehicleAction(1.0, 0.3), 0.02)
    later = first
    for _ in range(15):
        later = model.step(later, VehicleAction(1.0, 0.3), 0.02)
    assert abs(later.yaw_rate) > 2.0 * abs(first.yaw_rate)


def test_hybrid_reverse_weaker_than_forward():
    # The ESC caps reverse force at 25% by default: the commanded drive force is
    # asymmetric at init (realized thrust saturates the tire curve, so the
    # asymmetry is asserted on the command force), and reverse must still move
    # the car backward, no faster than forward.
    import torch

    from vehicle_simulation.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM)
    f_fwd = float(model._f_cmd(torch.tensor(1.0)))
    f_rev = float(model._f_cmd(torch.tensor(-1.0)))
    assert 0.0 < abs(f_rev) < 0.5 * f_fwd

    def launch(throttle):
        m = PhysicsInformedModel.new(_NORM)
        s = VehicleState()
        for _ in range(25):
            s = m.step(s, VehicleAction(0.0, throttle), 0.02)
        return s.vx

    fwd, rev = launch(1.0), launch(-1.0)
    assert fwd > 0.0 > rev
    assert abs(rev) <= fwd


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
