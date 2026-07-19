"""Unit tests for the lateral path-following MPC on synthetic centerlines."""

import math

import numpy as np

from drift_core.control import PathMpc, PathMpcConfig, build_speed_profile

WHEELBASE = 0.256
MAX_STEER = 1.309  # rad (75 deg)
DT_CTRL = 0.02


def _mpc(curvature, cumulative, total_length, **overrides) -> PathMpc:
    cfg = PathMpcConfig(wheelbase_m=WHEELBASE, max_steer_angle_rad=MAX_STEER, **overrides)
    mpc = PathMpc(cfg)
    profile = build_speed_profile(
        curvature, cumulative, total_length, v_max=1.5, a_lat_max=3.5, a_brake=2.0, a_accel=2.0
    )
    mpc.set_track(curvature, cumulative, total_length, profile)
    return mpc


def _straight_mpc(**overrides) -> PathMpc:
    n = 200
    return _mpc(np.zeros(n), np.linspace(0.0, 49.75, n), 50.0, **overrides)


def _circle_mpc(kappa: float, **overrides) -> PathMpc:
    n = 200
    total = 2.0 * math.pi / abs(kappa)
    return _mpc(np.full(n, kappa), np.linspace(0.0, total, n, endpoint=False), total, **overrides)


def test_solve_is_deterministic():
    mpc = _straight_mpc()
    args = dict(
        lateral_error=0.2, heading_error=0.1, v_meas=1.0, s=5.0, prev_steering=0.05, dt_ctrl=0.02
    )
    first = mpc.solve(**args)
    second = mpc.solve(**args)
    assert first.steering == second.steering
    assert np.array_equal(first.curvature_sequence, second.curvature_sequence)


def test_sign_conventions():
    mpc = _straight_mpc()
    # Car left of the path, aligned: must steer right (negative).
    left = mpc.solve(0.3, 0.0, 1.0, 5.0, prev_steering=0.0, dt_ctrl=1.0)
    assert left.curvature_sequence[0] < 0.0
    assert left.steering < 0.0
    # Mirror image steers left.
    right = mpc.solve(-0.3, 0.0, 1.0, 5.0, prev_steering=0.0, dt_ctrl=1.0)
    assert right.steering > 0.0
    assert np.isclose(right.steering, -left.steering)


def test_circle_feedforward_matches_curvature():
    kappa = 1.0
    mpc = _circle_mpc(kappa)
    u_ff = math.atan(WHEELBASE * kappa) / MAX_STEER
    solution = mpc.solve(0.0, 0.0, 1.0, 1.0, prev_steering=u_ff, dt_ctrl=1.0)
    assert np.allclose(solution.curvature_sequence, kappa, atol=1e-6)
    assert np.isclose(solution.steering, u_ff, atol=1e-6)
    assert np.allclose(solution.predicted_errors, 0.0, atol=1e-9)


def test_saturation_pins_at_curvature_limit():
    mpc = _straight_mpc()
    cfg = PathMpcConfig(wheelbase_m=WHEELBASE, max_steer_angle_rad=MAX_STEER)
    solution = mpc.solve(5.0, 0.0, 1.0, 5.0, prev_steering=0.0, dt_ctrl=1.0)
    assert np.isclose(solution.curvature_sequence[0], -cfg.curvature_limit)
    assert abs(solution.steering) <= cfg.steer_cmd_limit + 1e-12
    assert np.all(np.isfinite(solution.curvature_sequence))


def test_steering_rate_limit_applies():
    mpc = _straight_mpc()
    cfg = PathMpcConfig(wheelbase_m=WHEELBASE, max_steer_angle_rad=MAX_STEER)
    solution = mpc.solve(5.0, 0.0, 1.0, 5.0, prev_steering=0.0, dt_ctrl=DT_CTRL)
    assert abs(solution.steering) <= cfg.steer_rate_limit * DT_CTRL + 1e-12


def test_standstill_gives_finite_command():
    mpc = _straight_mpc()
    solution = mpc.solve(0.2, 0.3, 0.0, 0.0, prev_steering=0.0, dt_ctrl=DT_CTRL)
    assert math.isfinite(solution.steering)
    assert solution.curvature_sequence[0] < 0.0  # still steers back toward the path


def test_straight_line_convergence_no_offset():
    """Closed loop on the MPC's own model: 0.3 m offset dies out with no steady error."""
    mpc = _straight_mpc()
    v = 1.5
    e_y, e_psi = 0.3, 0.0
    steering = 0.0
    history = []
    for _ in range(int(4.0 / DT_CTRL)):
        solution = mpc.solve(e_y, e_psi, v, 5.0, prev_steering=steering, dt_ctrl=DT_CTRL)
        steering = solution.steering
        w = math.tan(steering * MAX_STEER) / WHEELBASE
        e_y += v * math.sin(e_psi) * DT_CTRL
        e_psi += v * w * DT_CTRL
        history.append(e_y)
    assert abs(history[-1]) < 0.02
    assert max(abs(e) for e in history[int(2.0 / DT_CTRL) :]) < 0.05  # no limit cycle


def test_solve_requires_track():
    mpc = PathMpc(PathMpcConfig(wheelbase_m=WHEELBASE, max_steer_angle_rad=MAX_STEER))
    try:
        mpc.solve(0.0, 0.0, 1.0, 0.0, prev_steering=0.0, dt_ctrl=DT_CTRL)
    except RuntimeError:
        return
    raise AssertionError("solve() before set_track() should raise RuntimeError")
