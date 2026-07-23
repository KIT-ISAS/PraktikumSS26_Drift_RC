"""Unit tests for the curvature-based speed profile."""

import numpy as np

from drift_core.control import build_speed_profile

V_MAX = 2.0
A_LAT = 3.5
A_BRAKE = 2.0
A_ACCEL = 2.0


def _profile(curvature, cumulative, total_length, **overrides):
    kwargs = dict(v_max=V_MAX, a_lat_max=A_LAT, a_brake=A_BRAKE, a_accel=A_ACCEL, ds=0.05)
    kwargs.update(overrides)
    return build_speed_profile(curvature, cumulative, total_length, **kwargs)


def test_straight_track_runs_at_v_max():
    n = 100
    profile = _profile(np.zeros(n), np.linspace(0.0, 9.9, n), 10.0)
    assert np.allclose(profile.v_grid, V_MAX)


def test_constant_curvature_matches_lateral_limit():
    n = 100
    kappa = 1.5  # tight enough that sqrt(a_lat/kappa) < v_max
    profile = _profile(np.full(n, kappa), np.linspace(0.0, 9.9, n), 10.0)
    assert np.allclose(profile.v_grid, np.sqrt(A_LAT / kappa))


def _corner_track():
    """20 m loop, straight except for a tight corner (|kappa| = 2) around s in [1, 3]."""
    s = np.linspace(0.0, 19.8, 200)
    kappa = np.where((s > 1.0) & (s < 3.0), 2.0, 0.0)
    return kappa, s, 20.0


def test_braking_and_accel_limits_hold_including_wrap():
    kappa, s, total = _corner_track()
    profile = _profile(kappa, s, total)
    v = profile.v_grid[:-1]  # drop the duplicated wrap sample
    ds = np.diff(np.append(profile.s_grid[:-1], total))
    tol = 1e-9
    v_next = np.roll(v, -1)
    assert np.all(v**2 <= v_next**2 + 2.0 * A_BRAKE * ds + tol)
    assert np.all(v_next**2 <= v**2 + 2.0 * A_ACCEL * ds + tol)


def test_corner_forces_early_braking():
    kappa, s, total = _corner_track()
    profile = _profile(kappa, s, total)
    v_corner = np.sqrt(A_LAT / 2.0)
    assert np.isclose(profile.sample(2.0), v_corner, atol=0.05)
    # Just before the corner the profile must already be slowing down.
    assert profile.sample(0.7) < V_MAX - 1e-6


def test_sample_wraps():
    kappa, s, total = _corner_track()
    profile = _profile(kappa, s, total)
    queries = np.array([0.0, 2.0, 7.5, 19.9])
    assert np.allclose(profile.sample(queries + total), profile.sample(queries))


def test_profile_continuous_across_seam():
    """A corner straddling s = 0 must brake the end of the lap, not just the start."""
    s = np.linspace(0.0, 19.8, 200)
    kappa = np.where((s < 1.0) | (s > 19.0), 2.0, 0.0)
    profile = _profile(kappa, s, 20.0)
    assert profile.sample(19.5) <= np.sqrt(A_LAT / 2.0) + 0.05
    assert abs(profile.sample(0.0) - profile.sample(20.0 - 1e-9)) < 0.05
