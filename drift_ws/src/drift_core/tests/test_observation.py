"""Tests for the shared observation/action builder (pure numpy, no RL extras needed)."""

import math

from drift_core.environments.observation import (
    ACTION_DIM,
    CURVATURE_SCALE,
    LATERAL_SCALE,
    LOOKAHEAD_OFFSETS_M,
    N_LOOKAHEAD,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    SPEED_SCALE,
    YAW_RATE_SCALE,
    action_from_array,
    build_observation,
    curvature_lookahead,
    observation_from_state,
)
from drift_core.models.base import VehicleAction, VehicleState
from drift_core.track import TrackProjector, load_track
import numpy as np


def test_dimensions_and_bounds():
    assert OBS_DIM == 8 + N_LOOKAHEAD
    assert ACTION_DIM == 2
    assert OBS_LOW.shape == OBS_HIGH.shape == (OBS_DIM,)
    assert OBS_LOW.dtype == OBS_HIGH.dtype == np.float32
    assert np.all(OBS_LOW < OBS_HIGH)


def test_build_observation_field_order_and_scaling():
    obs = build_observation(
        lateral_error=0.25,
        heading_error=0.5,
        vx=1.5,
        vy=-0.6,
        yaw_rate=2.0,
        prev_action=np.array([0.3, -0.7]),
        curvatures_ahead=np.full(N_LOOKAHEAD, 0.8),
    )
    assert obs.dtype == np.float32
    assert obs.shape == (OBS_DIM,)
    assert obs[0] == np.float32(0.25 / LATERAL_SCALE)
    assert math.isclose(obs[1], math.sin(0.5), abs_tol=1e-6)
    assert math.isclose(obs[2], math.cos(0.5), abs_tol=1e-6)
    assert obs[3] == np.float32(1.5 / SPEED_SCALE)
    assert obs[4] == np.float32(-0.6 / SPEED_SCALE)
    assert obs[5] == np.float32(2.0 / YAW_RATE_SCALE)
    assert obs[6] == np.float32(0.3)
    assert obs[7] == np.float32(-0.7)
    assert np.allclose(obs[8:], 0.8 / CURVATURE_SCALE)
    assert np.all(obs >= OBS_LOW) and np.all(obs <= OBS_HIGH)


def test_build_observation_clips_extremes():
    obs = build_observation(
        lateral_error=10.0,
        heading_error=4.0,  # wrapped to [-pi, pi)
        vx=0.0,
        vy=0.0,
        yaw_rate=0.0,
        prev_action=np.array([2.0, -2.0]),
        curvatures_ahead=np.full(N_LOOKAHEAD, -50.0),
    )
    assert obs[0] == OBS_HIGH[0]
    assert -1.0 <= obs[1] <= 1.0 and -1.0 <= obs[2] <= 1.0
    assert obs[6] == 1.0 and obs[7] == -1.0
    assert np.all(obs[8:] == OBS_LOW[8:])


def test_curvature_lookahead_constant_circle():
    # A circle of radius 2: constant curvature 0.5 everywhere, at any query point.
    n = 100
    radius = 2.0
    total = 2.0 * math.pi * radius
    cumul = np.linspace(0.0, total, n, endpoint=False)
    kappa = np.full(n, 1.0 / radius)
    for s in (0.0, 3.7, total - 0.01):
        samples = curvature_lookahead(kappa, cumul, total, s)
        assert samples.shape == (N_LOOKAHEAD,)
        assert np.allclose(samples, 0.5)


def test_curvature_lookahead_wraps_around():
    # Piecewise profile: straight (kappa 0) for s < 5, arc (kappa 1) for s >= 5, L = 10.
    cumul = np.arange(0.0, 10.0, 0.5)
    kappa = np.where(cumul < 5.0, 0.0, 1.0)
    total = 10.0
    # Querying from near the end must wrap into the straight at the track start.
    samples = curvature_lookahead(kappa, cumul, total, s=9.0, offsets=(0.25, 2.0))
    assert samples[0] > 0.5  # still on the arc (s = 9.25)
    assert samples[1] < 0.5  # wrapped to s = 1.0, on the straight
    # Zero offset reproduces the local curvature.
    assert curvature_lookahead(kappa, cumul, total, s=2.0, offsets=(0.0,))[0] == 0.0
    assert curvature_lookahead(kappa, cumul, total, s=7.25, offsets=(0.0,))[0] == 1.0


def test_lookahead_offsets_sorted_and_local():
    assert list(LOOKAHEAD_OFFSETS_M) == sorted(LOOKAHEAD_OFFSETS_M)
    track = load_track("figure_8")
    waypoints, _, cumul = track.discretize()
    projector = TrackProjector(waypoints, cumul)
    assert max(LOOKAHEAD_OFFSETS_M) < projector.total_length / 2.0


def test_observation_from_state_on_track():
    track = load_track("figure_8")
    waypoints, curvature, cumul = track.discretize()
    projector = TrackProjector(waypoints, cumul)
    x, y, heading = waypoints[3]
    state = VehicleState(x=x, y=y, heading=heading, vx=1.0)
    projection = projector.project(x, y)
    obs = observation_from_state(
        state, projection, np.zeros(2), curvature, cumul, projector.total_length
    )
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))
    # On the centerline with the track's own heading: no lateral error, and only the
    # small analytic-tangent vs segment-chord mismatch in heading.
    assert abs(obs[0]) < 1e-3
    assert abs(obs[1]) < 0.05 and math.isclose(obs[2], 1.0, abs_tol=5e-3)


def test_action_from_array_clips():
    action = action_from_array(np.array([2.0, -3.0]))
    assert isinstance(action, VehicleAction)
    assert action.steering == 1.0
    assert action.throttle == -1.0
