"""Model abstraction: pose integration, rollout, and the build_model registry."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from drift_core.models import build_model
from drift_core.models.base import VehicleState
from drift_core.models.integrate import integrate_pose, simulate
from drift_core.models.kinematic import KinematicBicycleModel


def test_vehicle_state_body_vector():
    vs = VehicleState(x=1.0, y=2.0, heading=0.3, vx=1.5, vy=-0.4, yaw_rate=0.2)
    assert list(vs.body()) == pytest.approx([1.5, -0.4, 0.2])


def test_integrate_pose_straight_line():
    pose = integrate_pose(np.array([0.0, 0.0, 0.0]), vx=2.0, vy=0.0, yaw_rate=0.0, dt=0.5)
    assert pose[0] == pytest.approx(1.0)
    assert pose[1] == pytest.approx(0.0)
    assert pose[2] == pytest.approx(0.0)


def test_integrate_pose_midpoint_turn():
    # Quarter-ish turn: midpoint heading = pi/2 sends motion to +y.
    pose = integrate_pose([0.0, 0.0, 0.0], vx=1.0, vy=0.0, yaw_rate=math.pi, dt=1.0)
    assert pose[0] == pytest.approx(0.0, abs=1e-9)
    assert pose[1] == pytest.approx(1.0)
    assert pose[2] == pytest.approx(math.pi)


def test_simulate_runs_kinematic_rollout():
    model = KinematicBicycleModel()
    actions = np.tile([0.0, 1.0], (10, 1))  # full throttle, straight
    states = simulate(model, VehicleState(), actions, 0.05)
    assert len(states) == 11
    assert states[-1].x > 0.0  # moved forward


def test_kinematic_reverses_when_esc_allows_and_brakes_when_locked():
    model = KinematicBicycleModel()
    actions = np.tile([0.0, -1.0], (100, 1))  # fresh ESC: negative pedal = reverse
    states = simulate(model, VehicleState(), actions, 0.02)
    assert states[-1].vx < 0.0
    assert states[-1].vx >= -model.max_reverse_speed - 1e-6

    model = KinematicBicycleModel()
    forward = np.tile([0.0, 1.0], (100, 1))  # sets the reverse lockout...
    state = simulate(model, VehicleState(), forward, 0.02)[-1]
    braking = np.tile([0.0, -1.0], (500, 1))  # ...so the same pedal only brakes
    states = simulate(model, state, braking, 0.02)
    assert states[-1].vx == 0.0


def test_build_model_kinematic_and_unknown():
    assert isinstance(build_model("kinematic"), KinematicBicycleModel)
    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_build_model_neural_requires_weights():
    # The neural model is implemented on this PR but needs a trained checkpoint:
    # without weights/run_dir it must raise a clean ValueError (not ImportError),
    # and the error must mention how to supply the checkpoint.
    with pytest.raises(ValueError, match="weights"):
        build_model("neural")


def test_build_model_dynamic_baseline():
    # The physics bicycle has physically-motivated initial params, so it builds
    # (untrained) as a baseline with no checkpoint.
    pytest.importorskip("torch")
    from drift_core.models.dynamic import DynamicBicycleModel

    assert isinstance(build_model("dynamic"), DynamicBicycleModel)


def test_build_model_hybrid_requires_weights():
    # The hybrid's residual net needs training, so without weights/run_dir it must
    # raise a clean ValueError (not ImportError) pointing at the checkpoint.
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="weights"):
        build_model("hybrid")


def test_kinematic_path_imports_without_torch():
    """Building/stepping the kinematic model must not import torch (light path)."""
    code = (
        "import sys\n"
        "from drift_core.models import build_model\n"
        "from drift_core.models.base import VehicleState, VehicleAction\n"
        "m = build_model('kinematic')\n"
        "m.step(VehicleState(), VehicleAction(0.2, 0.5), 0.02)\n"
        "assert 'torch' not in sys.modules, 'torch was imported on the kinematic path'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def _latent_toy_model():
    """A stochastic model with latents: 3 chassis dims + 2 latents, cov over the chassis."""
    import torch

    from drift_core.models.base import Prediction, TrainableModel

    class LatentToy(TrainableModel):
        latent_dim = 2

        def __init__(self):
            super().__init__()
            self._can_sample = True

        def init_latent(self, dynamic, action):
            return torch.zeros(*dynamic.shape[:-1], 2)

        def derivative(self, dynamic, action):
            cov = torch.full(dynamic.shape[:-1] + (3,), 4.0)  # per-second variance
            return Prediction(torch.zeros_like(dynamic), cov)

        def trainable_parameters(self):
            return []

        def to_checkpoint(self):
            return {}

    return LatentToy()


def test_sampled_step_noises_only_the_chassis_dims():
    # cov covers [vx, vy, yaw_rate]; the latent actuator states are integrated
    # deterministically and must receive no diffusion noise of their own.
    torch = pytest.importorskip("torch")

    from drift_core.models.base import VehicleAction

    model = _latent_toy_model()
    out = model.step_stochastic(VehicleState(vx=1.0), VehicleAction(0.0, 0.0), 0.02)
    assert torch.count_nonzero(model._sim_latent) == 0, "latents must get no noise"
    assert (out.vx, out.vy, out.yaw_rate) != (1.0, 0.0, 0.0), "chassis must actually be noised"


def test_seeded_sampling_is_reproducible_with_latent_states():
    # The seeded generator and the chassis-only noise slice have to work together:
    # same seed reproduces the step exactly, a different seed changes it.
    pytest.importorskip("torch")

    from drift_core.models.base import VehicleAction

    def step_with(seed):
        model = _latent_toy_model()
        model.seed_sampling(seed)
        out = model.step_stochastic(VehicleState(vx=1.0), VehicleAction(0.0, 0.0), 0.02)
        return (out.vx, out.vy, out.yaw_rate)

    assert step_with(1) == step_with(1)
    assert step_with(1) != step_with(2)
