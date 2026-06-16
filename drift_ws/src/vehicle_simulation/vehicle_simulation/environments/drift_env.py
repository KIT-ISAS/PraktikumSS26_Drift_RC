"""Gymnasium environment for drift car lap racing."""

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from ..models.base import VehicleModel, VehicleState
from ..track import BezierTrack, TrackProjector
from .observation import (
    ACTION_DIM,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    action_from_array,
    observation_from_state,
)


class DriftEnv(gym.Env):
    """
    Gymnasium environment wrapping a VehicleModel on a BezierTrack.

    Observation (7-dim):
        [lateral_error, vx, vy, yaw_rate, lap_progress, cos(heading_err), sin(heading_err)]

    Action (2-dim):
        [steering, throttle]  each in [-1, 1]

    Reward:
        progress_delta * progress_weight  -  lateral_error² * lateral_weight

    Termination:
        |lateral_error| > max_lateral_error  (off-track)

    Truncation:
        lap >= max_laps
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: VehicleModel,
        track: BezierTrack,
        dt: float = 0.02,
        max_laps: int = 3,
        max_lateral_error: float = 0.5,
        progress_weight: float = 1.0,
        lateral_weight: float = 0.5,
        stochastic: bool = False,
        points_per_segment: int = 20,
    ) -> None:
        super().__init__()
        self._model = model
        self._dt = dt
        self._max_laps = max_laps
        self._max_lateral_error = max_lateral_error
        self._progress_weight = progress_weight
        self._lateral_weight = lateral_weight
        self._stochastic = stochastic

        waypoints, _, cumul = track.discretize(points_per_segment)
        self._projector = TrackProjector(waypoints, cumul)
        self._total_length = self._projector.total_length

        self.observation_space = Box(low=OBS_LOW, high=OBS_HIGH)
        self.action_space = Box(
            low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
            high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
        )

        self._state: VehicleState = VehicleState()
        self._prev_distance: float = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._projector.reset()
        self._state = VehicleState()
        self._prev_distance = 0.0
        return self._make_obs(), {}

    def step(self, action: np.ndarray):
        a = action_from_array(action)
        if self._stochastic:
            self._state = self._model.step_stochastic(self._state, a, self._dt)
        else:
            self._state = self._model.step(self._state, a, self._dt)

        proj = self._projector.project(self._state.x, self._state.y)

        if proj is None:
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}

        # Progress reward: handle lap wrap-around
        progress_delta = proj.distance - self._prev_distance
        if progress_delta < -self._total_length * 0.5:
            progress_delta += self._total_length
        self._prev_distance = proj.distance

        reward = (
            self._progress_weight * progress_delta - self._lateral_weight * proj.lateral_error**2
        )

        terminated = abs(proj.lateral_error) > self._max_lateral_error
        truncated = proj.lap >= self._max_laps

        return self._make_obs(proj), float(reward), terminated, truncated, {}

    def _make_obs(self, proj=None) -> np.ndarray:
        if proj is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        return observation_from_state(self._state, proj)
