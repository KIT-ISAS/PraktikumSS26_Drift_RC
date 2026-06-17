"""
Shared observation/action construction for the drift environment.

This is deliberately decoupled from ROS and from gymnasium so the exact same
code builds the policy's inputs in training (``DriftEnv``) and, on the real car,
in the deployment node — guaranteeing the agent sees identical observations in
simulation and reality.

Observation (7-dim):
    [lateral_error, vx, vy, yaw_rate, lap_progress, cos(heading_err), sin(heading_err)]
Action (2-dim):
    [steering, throttle]  each in [-1, 1]
"""

import math

import numpy as np

from ..models.base import VehicleAction, VehicleState

OBS_DIM = 7
ACTION_DIM = 2

OBS_LOW = np.array([-2.0, -10.0, -10.0, -20.0, 0.0, -1.0, -1.0], dtype=np.float32)
OBS_HIGH = np.array([2.0, 10.0, 10.0, 20.0, 1.0, 1.0, 1.0], dtype=np.float32)


def build_observation(
    lateral_error: float,
    vx: float,
    vy: float,
    yaw_rate: float,
    lap_progress: float,
    heading_error: float,
) -> np.ndarray:
    """Assemble the 7-d observation vector from raw scalar inputs."""
    return np.array(
        [
            lateral_error,
            vx,
            vy,
            yaw_rate,
            lap_progress,
            math.cos(heading_error),
            math.sin(heading_error),
        ],
        dtype=np.float32,
    )


def observation_from_state(state: VehicleState, projection) -> np.ndarray:
    """Build the observation from a :class:`VehicleState` and a track ``Projection``."""
    heading_error = state.heading - projection.heading
    return build_observation(
        projection.lateral_error,
        state.vx,
        state.vy,
        state.yaw_rate,
        projection.lap_progress,
        heading_error,
    )


def action_from_array(action) -> VehicleAction:
    """Convert a 2-element ``[steering, throttle]`` array into a :class:`VehicleAction`."""
    return VehicleAction(steering=float(action[0]), throttle=float(action[1]))
