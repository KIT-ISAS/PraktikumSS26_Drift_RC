"""Closed-loop acceptance test: MPC + speed controller lap the figure-8 track."""

import numpy as np
import pytest

from drift_core.agent.control import (
    PathMpc,
    PathMpcConfig,
    SpeedController,
    SpeedControllerConfig,
    build_speed_profile,
)
from drift_core.track import TrackProjector, load_track
from drift_core.vehicle_model.car_parameters import load_car_parameters
from drift_core.vehicle_model.models.base import VehicleAction, VehicleState
from drift_core.vehicle_model.models.kinematic import KinematicBicycleModel

DT = 0.02
SPEED_PREVIEW_S = 0.3


def _controllers(v_max: float):
    car = load_car_parameters("drift_rc")
    track = load_track("figure_8")
    waypoints, curvature, cumulative = track.discretize()
    projector = TrackProjector(waypoints, cumulative)
    profile = build_speed_profile(
        curvature,
        cumulative,
        projector.total_length,
        v_max=v_max,
        a_lat_max=3.5,
        a_brake=2.0,
        a_accel=2.0,
    )
    mpc = PathMpc(
        PathMpcConfig(wheelbase_m=car.wheelbase_m, max_steer_angle_rad=car.max_steer_angle_rad)
    )
    mpc.set_track(curvature, cumulative, projector.total_length, profile)
    # throttle_max=1.0: the kinematic sim model needs ~v/2 throttle at steady state,
    # unlike the real car whose 0.4 default is a grip limit, not a speed limit.
    speed = SpeedController(SpeedControllerConfig(throttle_max=1.0))
    model = KinematicBicycleModel(
        wheelbase=car.wheelbase_m, max_steer_angle=car.max_steer_angle_rad
    )
    return waypoints, projector, profile, mpc, speed, model


def _run(v_max: float, offset_m: float = 0.0, duration_s: float = 40.0):
    waypoints, projector, profile, mpc, speed, model = _controllers(v_max)
    x0, y0, heading0 = waypoints[0]
    # Offset perpendicular-left of the track tangent.
    state = VehicleState(
        x=x0 - offset_m * np.sin(heading0), y=y0 + offset_m * np.cos(heading0), heading=heading0
    )
    steering = 0.0
    lateral_errors, off_track_flags, solve_times = [], [], []
    projection = None
    for step in range(int(duration_s / DT)):
        projection = projector.project(state.x, state.y)
        assert projection is not None, "projector lost the car"
        solution = mpc.solve(
            projection.lateral_error,
            state.heading - projection.heading,
            state.vx,
            projection.distance,
            prev_steering=steering,
            dt_ctrl=DT,
        )
        steering = solution.steering
        v_ref = profile.sample(projection.distance + max(state.vx, 0.0) * SPEED_PREVIEW_S)
        throttle = speed.step(float(v_ref), state.vx, DT)
        state = model.step(state, VehicleAction(steering=steering, throttle=throttle), DT)
        solve_times.append(solution.solve_time_s)
        if step * DT >= 2.0:  # skip the spin-up/recovery transient
            lateral_errors.append(abs(projection.lateral_error))
            off_track_flags.append(projection.off_track)
    return projection, lateral_errors, off_track_flags, solve_times


@pytest.mark.parametrize("v_max", [1.5, 2.0])
def test_laps_figure_8_within_tolerance(v_max):
    projection, lateral_errors, off_track_flags, solve_times = _run(v_max)
    assert projection.lap >= 1, "no validated lap completed"
    assert max(lateral_errors) < 0.15
    assert not any(off_track_flags)
    assert np.mean(solve_times) < 2e-3


def test_recovers_from_offset_start():
    projection, lateral_errors, _, _ = _run(1.5, offset_m=0.3)
    assert projection.lap >= 1
    assert max(lateral_errors) < 0.15
