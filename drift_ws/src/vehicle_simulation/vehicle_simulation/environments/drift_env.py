"""
DriftEnv — gymnasium environment stepping a vehicle model against a track.

Rolls any :class:`~vehicle_simulation.models.base.VehicleModel` (built via
``build_model``: arcade baseline now, a trained dynamics run dir later) on a
Bézier track at a fixed control rate, with the stage-parameterized reward from
:mod:`.rewards` and observations from the shared builder in :mod:`.observation`
(the sim-to-real seam). Configured by a :class:`~.config.StageConfig` /
:class:`~.config.EnvConfig` or a packaged stage name.

This is the only module in ``environments/`` that imports ``gymnasium`` — import
it lazily (``vehicle_simulation.environments.DriftEnv``) so the light path works
without the ``[rl]`` extra.
"""

import math
from typing import Callable

import gymnasium
from gymnasium import spaces
import numpy as np

from ..models import build_model
from ..models.base import VehicleAction, VehicleModel, VehicleState
from ..track import TrackProjector, load_track
from ..transforms import wrap_angle
from .config import EnvConfig, StageConfig
from .observation import OBS_HIGH, OBS_LOW, observation_from_state
from .rewards import compute_reward, progress_delta


class DriftEnv(gymnasium.Env):
    """Single-agent drift-track environment (continuous steering/throttle in [-1, 1])."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | StageConfig | str = "stage1_path_follow",
        model_factory: Callable[[], VehicleModel] | None = None,
    ) -> None:
        """
        Create the environment for one curriculum stage.

        ``config`` is an :class:`EnvConfig`, a :class:`StageConfig`, a packaged
        stage name, or a stage YAML path. ``model_factory`` overrides the default
        ``build_model(cfg.model_type, ...)`` — a *fresh* model is built on every
        reset so models with internal state (e.g. the arcade slip) start clean.
        """
        if isinstance(config, str):
            config = StageConfig.load(config)
        self._cfg = config.env if isinstance(config, StageConfig) else config
        cfg = self._cfg
        if model_factory is None:
            model_factory = lambda: build_model(  # noqa: E731
                cfg.model_type, config=cfg.model_config, **cfg.model_kwargs
            )
        self._model_factory = model_factory

        track = load_track(cfg.track)
        waypoints, curvature, cumul = track.discretize(cfg.points_per_segment)
        self._waypoints = np.asarray(waypoints, dtype=np.float64)
        self._curvature = np.asarray(curvature, dtype=np.float64)
        self._cumul = np.asarray(cumul, dtype=np.float64)
        self._projector = TrackProjector(waypoints, cumul)
        self._total_length = self._projector.total_length

        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._model: VehicleModel | None = None
        self._state = VehicleState()
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_s = 0.0
        self._step_count = 0
        self._progress_m = 0.0
        self._stall_time = 0.0

    def _centerline_pose(self, s: float) -> tuple[float, float, float]:
        """Interpolate (x, y, heading) on the closed centerline at arc length ``s``."""
        s = float(np.mod(s, self._total_length))
        idx = int(np.searchsorted(self._cumul, s, side="right")) - 1
        x0, y0, _ = self._waypoints[idx]
        nxt = (idx + 1) % len(self._waypoints)
        x1, y1, _ = self._waypoints[nxt]
        seg_start = self._cumul[idx]
        seg_len = (self._cumul[nxt] if nxt else self._total_length) - seg_start
        frac = (s - seg_start) / seg_len if seg_len > 0.0 else 0.0
        heading = math.atan2(y1 - y0, x1 - x0)
        return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0), heading

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Start a new episode at a random point on the centerline.

        ``options`` supports ``spawn_s`` (fixed arc length) and ``deterministic``
        (disable all spawn noise) for reproducible evaluation.
        """
        super().reset(seed=seed)
        options = options or {}
        spawn = self._cfg.spawn

        self._model = self._model_factory()

        s0 = float(options.get("spawn_s", self.np_random.uniform(0.0, self._total_length)))
        x, y, heading = self._centerline_pose(s0)
        speed = spawn.speed
        if not options.get("deterministic", False):
            lateral = float(
                np.clip(
                    self.np_random.normal(0.0, spawn.lateral_noise_std),
                    -2.0 * spawn.lateral_noise_std,
                    2.0 * spawn.lateral_noise_std,
                )
            )
            x += -math.sin(heading) * lateral
            y += math.cos(heading) * lateral
            heading = wrap_angle(heading + self.np_random.normal(0.0, spawn.heading_noise_std))
            # Standing starts stay in-distribution: the real car always launches
            # from parked, so the policy must learn it.
            if self.np_random.random() < spawn.standstill_prob:
                speed = 0.0
            else:
                speed += self.np_random.normal(0.0, spawn.speed_noise_std)
        self._state = VehicleState(x=x, y=y, heading=heading, vx=max(0.0, speed))

        self._projector.reset()
        projection = self._projector.project(self._state.x, self._state.y)
        self._last_s = projection.distance
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._progress_m = 0.0
        self._stall_time = 0.0

        obs = self._observation(projection)
        return obs, self._info(projection, reason=None)

    def step(self, action):
        """Advance one control period (``cfg.dt``)."""
        cfg = self._cfg
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._state = self._model.step(
            self._state, VehicleAction(steering=float(a[0]), throttle=float(a[1])), cfg.dt
        )
        projection = self._projector.project(self._state.x, self._state.y)

        ds = progress_delta(projection.distance, self._last_s, self._total_length)
        self._last_s = projection.distance
        self._progress_m += ds

        speed = self._state.speed
        if speed < cfg.termination.stall_speed:
            self._stall_time += cfg.dt
        else:
            self._stall_time = 0.0

        kappa_here = float(
            np.interp(
                projection.distance,
                np.append(self._cumul, self._total_length),
                np.append(self._curvature, self._curvature[0]),
            )
        )
        heading_error = wrap_angle(self._state.heading - projection.heading)
        outcome = compute_reward(
            cfg.reward,
            cfg.termination,
            ds=ds,
            speed=speed,
            lateral_error=projection.lateral_error,
            heading_error=heading_error,
            sideslip=self._state.sideslip,
            kappa_here=kappa_here,
            action=a,
            prev_action=self._prev_action,
            dt=cfg.dt,
            stalled=self._stall_time >= cfg.termination.stall_timeout_s,
        )
        self._prev_action = a
        self._step_count += 1
        truncated = not outcome.terminated and self._step_count >= cfg.max_episode_steps

        obs = self._observation(projection)
        info = self._info(projection, reason=outcome.reason)
        return obs, outcome.reward, outcome.terminated, truncated, info

    def _observation(self, projection) -> np.ndarray:
        return observation_from_state(
            self._state,
            projection,
            self._prev_action,
            self._curvature,
            self._cumul,
            self._total_length,
        )

    def _info(self, projection, reason: str | None) -> dict:
        info = {
            "s": projection.distance,
            "lap": projection.lap,
            "lateral_error": projection.lateral_error,
            "heading_error": wrap_angle(self._state.heading - projection.heading),
            "speed": self._state.speed,
            "sideslip": self._state.sideslip,
            "x": self._state.x,
            "y": self._state.y,
            "progress_m": self._progress_m,
        }
        if reason is not None:
            info["termination_reason"] = reason
        return info
