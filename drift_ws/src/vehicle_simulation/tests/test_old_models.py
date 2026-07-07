"""Unit tests for the ported predecessor models (``models/old/``)."""

import math

from vehicle_simulation.models import build_model
from vehicle_simulation.models.base import VehicleAction, VehicleState
from vehicle_simulation.models.old import OldSlipAngleVehicleModel, OldStandardVehicleModel

# The predecessor game's fixed tick; at this dt the ports match the originals.
DT = 0.008


def _roll(model, state, action, steps):
    for _ in range(steps):
        state = model.step(state, action, DT)
    return state


def test_build_model_returns_old_models():
    assert isinstance(build_model("old_standard"), OldStandardVehicleModel)
    assert isinstance(build_model("old_slip_angle"), OldSlipAngleVehicleModel)


# --- old_standard -----------------------------------------------------------


def test_standard_throttle_reaches_target_speed():
    model = OldStandardVehicleModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=0.0, throttle=1.0), 3000)
    # Throttle commands a target speed; friction keeps it slightly below max.
    assert state.vx > 0.9 * model.max_speed
    assert state.vx <= model.max_speed + 1e-6
    assert state.x > 0.0
    assert math.isclose(state.y, 0.0, abs_tol=1e-9)
    assert state.vy == 0.0


def test_standard_steers_left_for_positive_steering():
    model = OldStandardVehicleModel()
    state = VehicleState(vx=2.0)
    # Few steps only, so the heading cannot wrap past pi.
    state = _roll(model, state, VehicleAction(steering=1.0, throttle=0.5), 30)
    assert state.heading > 0.0
    assert state.yaw_rate > 0.0


def test_standard_reverses():
    model = OldStandardVehicleModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=0.0, throttle=-1.0), 1000)
    assert state.vx < 0.0
    assert state.vx >= -model.max_reverse_speed - 1e-6
    assert state.x < 0.0


def test_standard_coasts_to_rest():
    model = OldStandardVehicleModel()
    state = VehicleState(vx=2.0)
    state = _roll(model, state, VehicleAction(steering=0.0, throttle=0.0), 5000)
    assert state.vx == 0.0


# --- old_slip_angle ---------------------------------------------------------


def test_slip_angle_drives_straight_without_steering():
    model = OldSlipAngleVehicleModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=0.0, throttle=0.5), 500)
    assert state.vx > 0.0
    assert math.isclose(state.heading, 0.0, abs_tol=1e-9)
    assert math.isclose(state.y, 0.0, abs_tol=1e-6)


def test_slip_angle_gentle_steering_turns_without_drifting():
    model = OldSlipAngleVehicleModel()
    # Steering below the 20° drift threshold: plain bicycle turning, no drift.
    state = VehicleState(vx=1.0)
    state = _roll(model, state, VehicleAction(steering=0.5, throttle=0.4), 100)
    assert not model._drifting
    assert state.heading > 0.0


def test_slip_angle_drifts_under_hard_steering_at_speed():
    model = OldSlipAngleVehicleModel()
    state = VehicleState(vx=3.0)
    action = VehicleAction(steering=1.0, throttle=1.0)
    drifted = False
    for _ in range(50):
        state = model.step(state, action, DT)
        if model._drifting and abs(state.vy) > 1e-3:
            drifted = True
            break
    assert drifted, "hard steering at speed with full throttle should enter a drift"


def test_slip_angle_reverses_from_standstill():
    model = OldSlipAngleVehicleModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=0.0, throttle=-1.0), 200)
    assert model._reverse
    assert state.vx < 0.0


def test_slip_angle_stays_finite_under_extreme_input():
    model = OldSlipAngleVehicleModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=1.0, throttle=1.0), 5000)
    for value in state.to_array():
        assert math.isfinite(value)
    assert -math.pi - 1e-6 <= state.heading <= math.pi + 1e-6
