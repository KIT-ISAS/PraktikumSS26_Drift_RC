"""Unit tests for the game-like ``ArcadeDriftModel``."""

import math

from vehicle_simulation.models import build_model
from vehicle_simulation.models.arcade import ArcadeDriftModel
from vehicle_simulation.models.base import VehicleAction, VehicleState

DT = 0.02


def _roll(model, state, action, steps):
    for _ in range(steps):
        state = model.step(state, action, DT)
    return state


def test_build_model_returns_arcade():
    model = build_model("arcade")
    assert isinstance(model, ArcadeDriftModel)


def test_throttle_accelerates_forward_then_caps_at_max_speed():
    model = ArcadeDriftModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=0.0, throttle=1.0), 1000)
    assert state.vx > 0.0
    assert state.vx <= model.max_speed + 1e-6
    # Driving straight: position advances along +x, no lateral drift.
    assert state.x > 0.0
    assert math.isclose(state.y, 0.0, abs_tol=1e-6)
    assert math.isclose(state.heading, 0.0, abs_tol=1e-9)


def test_cannot_steer_while_parked():
    model = ArcadeDriftModel()
    state = model.step(VehicleState(), VehicleAction(steering=1.0, throttle=0.0), DT)
    assert state.yaw_rate == 0.0
    assert state.heading == 0.0


def test_drag_brings_a_rolling_car_to_rest():
    model = ArcadeDriftModel()
    state = VehicleState(vx=3.0)
    state = _roll(model, state, VehicleAction(steering=0.0, throttle=0.0), 2000)
    assert math.isclose(state.vx, 0.0, abs_tol=1e-3)


def test_low_grip_drifts_more_than_high_grip():
    """Whip the wheel at speed: the looser-grip car should slide sideways more."""
    action = VehicleAction(steering=1.0, throttle=1.0)
    start = VehicleState(vx=4.0)

    loose = _roll(ArcadeDriftModel(grip=0.5), start, action, 25)
    tight = _roll(ArcadeDriftModel(grip=20.0), start, action, 25)

    assert abs(loose.vy) > abs(tight.vy)


def test_front_tires_saturate_into_understeer():
    """At speed, full lock may not pull more lateral accel than the front grip limit."""
    model = ArcadeDriftModel()
    state = model.step(VehicleState(vx=2.5), VehicleAction(steering=1.0, throttle=0.0), DT)
    assert abs(state.yaw_rate * state.vx) <= model.front_grip_limit + 1e-6
    # ...and well below what the unsaturated kinematic bicycle would demand.
    kinematic_yaw = (2.5 / model.wheelbase) * math.tan(model.max_steer_angle)
    assert abs(state.yaw_rate) < 0.5 * kinematic_yaw


def test_wheelspin_makes_flooring_it_slower_than_moderate_throttle():
    """Full throttle spins the tires and launches slower than threshold throttle."""
    steps = 25  # 0.5 s launch
    floored = _roll(
        ArcadeDriftModel(), VehicleState(), VehicleAction(steering=0.0, throttle=1.0), steps
    )
    gentle_throttle = ArcadeDriftModel().slip_lowerbound
    gentle = _roll(
        ArcadeDriftModel(),
        VehicleState(),
        VehicleAction(steering=0.0, throttle=gentle_throttle),
        steps,
    )
    assert 0.0 < floored.vx < gentle.vx


def test_step_is_finite_and_stable_for_extreme_inputs():
    model = ArcadeDriftModel()
    state = _roll(model, VehicleState(), VehicleAction(steering=1.0, throttle=1.0), 5000)
    for value in state.to_array():
        assert math.isfinite(value)
    assert -math.pi - 1e-6 <= state.heading <= math.pi + 1e-6
