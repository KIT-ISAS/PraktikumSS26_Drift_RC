"""Model abstraction: pose integration, rollout, and the build_model registry."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from vehicle_simulation.models import build_model
from vehicle_simulation.models.base import VehicleState
from vehicle_simulation.models.integrate import integrate_pose, simulate
from vehicle_simulation.models.kinematic import KinematicBicycleModel


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


def test_build_model_unimplemented_models_fail_cleanly():
    # Trainable models dynamic/hybrid are added in later PRs; on this PR they must
    # raise a clean ValueError, never an ImportError.
    for name in ("dynamic", "hybrid"):
        with pytest.raises(ValueError):
            build_model(name)


def test_kinematic_path_imports_without_torch():
    """Building/stepping the kinematic model must not import torch (light path)."""
    code = (
        "import sys\n"
        "from vehicle_simulation.models import build_model\n"
        "from vehicle_simulation.models.base import VehicleState, VehicleAction\n"
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
