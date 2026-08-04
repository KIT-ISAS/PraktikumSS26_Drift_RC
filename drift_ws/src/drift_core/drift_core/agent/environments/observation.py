"""
Shared observation/action construction for the drift environment (sim-to-real seam).

This module is deliberately pure numpy — no ROS, no gymnasium, no model stepping —
so the exact same code builds the policy's inputs in training (``DriftEnv``) and,
on the real car, in the deployment ROS node. Every input is available on both
sides: body-frame velocities and yaw rate (``/car/odom`` / ``VehicleState``), the
track projection (``TrackProgress`` msg / ``TrackProjector``), the centerline
curvature arrays (``Track`` msg / ``BezierTrack.discretize()``), and the previously
commanded action (the node's own last ``VehicleActionRequest``).

Observation (``OBS_DIM = 8 + N_LOOKAHEAD``), normalized with *fixed* constants
(no learned normalization, so deployment needs no statistics):

====== ============================== =========================================
index  field                          normalization
====== ============================== =========================================
0      lateral error                  / ``LATERAL_SCALE``, clipped to ±3
1, 2   sin/cos of heading error       already in [-1, 1]
3, 4   vx, vy (body frame)            / ``SPEED_SCALE``
5      yaw rate                       / ``YAW_RATE_SCALE``
6, 7   previous steering, throttle    raw, in [-1, 1]
8...   curvature at s + offsets       / ``CURVATURE_SCALE``, clipped to ±2
====== ============================== =========================================

The observation contains only *local* track context (curvature of the upcoming
road, not lap progress or absolute position), so a trained policy is not tied to
one specific track layout. The previous action is included because the reward
penalizes action rate and the car's slip dynamics depend on throttle history.
"""

import math

import numpy as np

from ...transforms import wrap_angle
from ...vehicle_model import VehicleAction, VehicleState

# Arc-length offsets (m) ahead of the car at which upcoming centerline curvature is
# sampled. Spacing grows with distance: dense where control decisions are imminent
# (one car length ahead), sparse for context. The far point is ~1.25 s of travel at
# 3 m/s — enough to see the next loop entry from anywhere on the figure-8 straights.
LOOKAHEAD_OFFSETS_M = (0.15, 0.35, 0.60, 0.90, 1.25, 1.65, 2.10, 2.60, 3.15, 3.75)
N_LOOKAHEAD = len(LOOKAHEAD_OFFSETS_M)

ACTION_DIM = 2
OBS_DIM = 8 + N_LOOKAHEAD

# Fixed normalization scales (SI units per unit of observation).
LATERAL_SCALE = 0.5  # m
SPEED_SCALE = 3.0  # m/s
YAW_RATE_SCALE = 4.0  # rad/s
CURVATURE_SCALE = 1.0  # 1/m

_LATERAL_CLIP = 3.0  # in units of LATERAL_SCALE
_CURVATURE_CLIP = 2.0  # in units of CURVATURE_SCALE

OBS_LOW = np.array(
    [-_LATERAL_CLIP, -1.0, -1.0, -3.0, -3.0, -3.0, -1.0, -1.0] + [-_CURVATURE_CLIP] * N_LOOKAHEAD,
    dtype=np.float32,
)
OBS_HIGH = -OBS_LOW


def curvature_lookahead(
    curvature,
    cumulative_distance,
    total_length: float,
    s: float,
    offsets=LOOKAHEAD_OFFSETS_M,
) -> np.ndarray:
    """
    Interpolate signed centerline curvature at arc lengths ``(s + offsets) % total_length``.

    Consumes exactly what ``BezierTrack.discretize()`` returns and what the ROS
    ``Track`` msg carries: per-waypoint ``curvature`` and ``cumulative_distance``
    arrays plus the closed track's ``total_length`` (which includes the closing
    segment back to the first waypoint). Interpolation is continuous through the
    wrap: a phantom sample at ``total_length`` with the first waypoint's curvature
    closes the profile.
    """
    kappa = np.asarray(curvature, dtype=np.float64)
    dist = np.asarray(cumulative_distance, dtype=np.float64)
    kappa_closed = np.append(kappa, kappa[0])
    dist_closed = np.append(dist, total_length)
    query = np.mod(s + np.asarray(offsets, dtype=np.float64), total_length)
    return np.interp(query, dist_closed, kappa_closed)


def build_observation(
    lateral_error: float,
    heading_error: float,
    vx: float,
    vy: float,
    yaw_rate: float,
    prev_action,
    curvatures_ahead,
) -> np.ndarray:
    """
    Assemble and normalize the float32 observation vector (the single source of truth).

    ``heading_error`` is the car heading minus the track-tangent heading (rad);
    ``curvatures_ahead`` must contain ``N_LOOKAHEAD`` signed curvature samples.
    """
    heading_error = wrap_angle(heading_error)
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0] = np.clip(lateral_error / LATERAL_SCALE, -_LATERAL_CLIP, _LATERAL_CLIP)
    obs[1] = math.sin(heading_error)
    obs[2] = math.cos(heading_error)
    obs[3] = np.clip(vx / SPEED_SCALE, OBS_LOW[3], OBS_HIGH[3])
    obs[4] = np.clip(vy / SPEED_SCALE, OBS_LOW[4], OBS_HIGH[4])
    obs[5] = np.clip(yaw_rate / YAW_RATE_SCALE, OBS_LOW[5], OBS_HIGH[5])
    obs[6:8] = np.clip(np.asarray(prev_action, dtype=np.float32), -1.0, 1.0)
    obs[8:] = np.clip(
        np.asarray(curvatures_ahead, dtype=np.float32) / CURVATURE_SCALE,
        -_CURVATURE_CLIP,
        _CURVATURE_CLIP,
    )
    return obs


def observation_from_state(
    state: VehicleState,
    projection,
    prev_action,
    curvature,
    cumulative_distance,
    total_length: float,
) -> np.ndarray:
    """Build the observation from a ``VehicleState``, a track ``Projection``, and track arrays."""
    curvatures_ahead = curvature_lookahead(
        curvature, cumulative_distance, total_length, projection.distance
    )
    return build_observation(
        lateral_error=projection.lateral_error,
        heading_error=state.heading - projection.heading,
        vx=state.vx,
        vy=state.vy,
        yaw_rate=state.yaw_rate,
        prev_action=prev_action,
        curvatures_ahead=curvatures_ahead,
    )


def action_from_array(action) -> VehicleAction:
    """Clip a 2-element ``[steering, throttle]`` array to [-1, 1] and wrap as VehicleAction."""
    steering, throttle = (float(np.clip(v, -1.0, 1.0)) for v in np.asarray(action)[:2])
    return VehicleAction(steering=steering, throttle=throttle)
