"""
Lateral path-following MPC on the kinematic bicycle in Frenet error coordinates.

Deterministic alternative to the RL agent: at every control tick a linear
time-varying MPC steers the car onto the track centerline; the throttle is
handled separately (:mod:`~drift_core.control.longitudinal`).

Formulation
-----------
State per horizon step: ``e = [e_y, e_psi]`` — lateral error (m, + = car left of
the centerline, matching ``TrackProgress.lateral_error``) and heading error
(rad, car heading minus track-tangent heading, wrapped).

The decision variable is the **commanded path curvature** ``w`` [1/m] rather than
the steering command. That keeps the prediction model exactly linear — the
kinematic-bicycle steering nonlinearity ``yaw_rate = v/L * tan(delta)`` is
inverted only once, after the solve: ``delta = atan(L * w)``,
``steering = delta / max_steer_angle``. Sign conventions are ISO 8855 throughout:
positive ``w`` = left turn = positive steering.

Discrete dynamics (forward Euler, per-step speed ``v_k`` scheduled from the
speed profile, curvature ``kappa_k`` interpolated wrap-aware at the predicted
arc lengths, mirroring ``curvature_lookahead``)::

    e_{k+1} = A_k e_k + B_k w_k + g_k
    A_k = [[1, v_k dt], [0, 1]],  B_k = [0, v_k dt],  g_k = [0, -v_k dt kappa_k]

Cost: quadratic tracking (``q_lateral``, ``q_heading``, terminal step scaled by
``q_terminal``) plus deviation from the feedforward curvature
(``r_curvature * (w_k - kappa_k)^2`` — penalizing raw ``w`` would fight steady
cornering) plus smoothness (``r_curvature_rate``). Condensed into one linear
least-squares stack and solved with box bounds ``|w| <= curvature_limit`` by
``scipy.optimize.lsq_linear`` (BVLS: deterministic active-set, ~0.2 ms at
``horizon_steps = 20`` — no extra solver dependency).

The first optimal input is inverted to a steering command and shaped at the
command edge: slew-limited against the previously applied command
(``steer_rate_limit``) and clamped to ``steer_cmd_limit``.
"""

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import lsq_linear

from ..transforms import wrap_angle
from .speed_profile import SpeedProfile

__all__ = ["PathMpc", "PathMpcConfig", "MpcSolution"]


@dataclass(frozen=True)
class PathMpcConfig:
    """Geometry, horizon, weights, and limits for :class:`PathMpc`."""

    wheelbase_m: float  # from CarParameters — do not hardcode
    max_steer_angle_rad: float  # front-wheel angle at full steering command
    horizon_steps: int = 20
    mpc_dt: float = 0.05  # horizon step [s] (coarser than the control rate)
    q_lateral: float = 20.0
    q_heading: float = 3.0
    q_terminal: float = 5.0  # multiplier on the last step's tracking cost
    r_curvature: float = 0.05  # on (w - kappa), not raw w
    r_curvature_rate: float = 0.3
    steer_cmd_limit: float = 0.6  # the real car understeers above ~50 % steer
    steer_rate_limit: float = 6.0  # command units per second
    v_min: float = 0.3  # scheduling floor so B_k never vanishes at standstill
    a_accel: float = 2.0  # caps how fast the scheduled speed may exceed v_meas

    @property
    def curvature_limit(self) -> float:
        """Path curvature at the steering command limit [1/m] (the QP box bound)."""
        return math.tan(self.steer_cmd_limit * self.max_steer_angle_rad) / self.wheelbase_m


@dataclass(frozen=True)
class MpcSolution:
    """One solve's result: the shaped command plus horizon internals for logging/tests."""

    steering: float  # shaped command in [-1, 1], ready to publish
    curvature_sequence: np.ndarray  # (N,) optimal w over the horizon [1/m]
    predicted_errors: np.ndarray  # (N, 2) predicted [e_y, e_psi] at steps 1..N
    solve_time_s: float


class PathMpc:
    """Linear time-varying lateral MPC; call :meth:`set_track` once, then :meth:`solve`."""

    def __init__(self, config: PathMpcConfig) -> None:
        """Store the configuration; the track arrays arrive via :meth:`set_track`."""
        self._cfg = config
        self._kappa_closed: np.ndarray | None = None
        self._dist_closed: np.ndarray | None = None
        self._total_length = 0.0
        self._profile: SpeedProfile | None = None

    def set_track(
        self,
        curvature,
        cumulative_distance,
        total_length: float,
        speed_profile: SpeedProfile,
    ) -> None:
        """
        Cache the centerline arrays (as the ``Track`` msg carries them) and speed profile.

        A phantom sample at ``total_length`` with the first waypoint's curvature makes
        the interpolation continuous through the closing seam.
        """
        kappa = np.asarray(curvature, dtype=np.float64)
        dist = np.asarray(cumulative_distance, dtype=np.float64)
        self._kappa_closed = np.append(kappa, kappa[0])
        self._dist_closed = np.append(dist, total_length)
        self._total_length = float(total_length)
        self._profile = speed_profile

    def solve(
        self,
        lateral_error: float,
        heading_error: float,
        v_meas: float,
        s: float,
        prev_steering: float,
        dt_ctrl: float,
    ) -> MpcSolution:
        """
        Run one MPC solve and return the shaped steering command.

        ``s`` is the car's arc length along the centerline (``TrackProgress.distance``),
        ``prev_steering`` the previously applied command (for the slew limit and the
        smoothness cost), ``dt_ctrl`` the control period the command is applied for.
        """
        if self._profile is None:
            raise RuntimeError("set_track() must be called before solve()")
        t0 = time.perf_counter()
        cfg = self._cfg
        n = cfg.horizon_steps
        dt = cfg.mpc_dt
        e0 = np.array([lateral_error, wrap_angle(heading_error)], dtype=np.float64)

        # Schedule speed and curvature along the predicted arc lengths.
        v_sched = np.empty(n)
        kappa_ref = np.empty(n)
        s_k = math.fmod(s, self._total_length)
        for k in range(n):
            v_prof = float(self._profile.sample(s_k))
            v_sched[k] = max(cfg.v_min, min(v_prof, v_meas + cfg.a_accel * k * dt))
            kappa_ref[k] = np.interp(s_k, self._dist_closed, self._kappa_closed)
            s_k = math.fmod(s_k + v_sched[k] * dt, self._total_length)

        # Condense e_{1..N} = S e0 + Gamma w + c by accumulating the recurrence.
        s_mat = np.empty((2 * n, 2))
        gamma = np.zeros((2 * n, n))
        c_vec = np.empty(2 * n)
        phi = np.eye(2)
        sens = np.zeros((2, n))
        affine = np.zeros(2)
        for k in range(n):
            vdt = v_sched[k] * dt
            a_k = np.array([[1.0, vdt], [0.0, 1.0]])
            phi = a_k @ phi
            sens = a_k @ sens
            sens[1, k] += vdt
            affine = a_k @ affine
            affine[1] -= vdt * kappa_ref[k]
            s_mat[2 * k : 2 * k + 2] = phi
            gamma[2 * k : 2 * k + 2] = sens
            c_vec[2 * k : 2 * k + 2] = affine

        # Stack the weighted least-squares system.
        weights = np.tile([math.sqrt(cfg.q_lateral), math.sqrt(cfg.q_heading)], n)
        weights[-2:] *= math.sqrt(cfg.q_terminal)
        sqrt_r = math.sqrt(cfg.r_curvature)
        sqrt_rd = math.sqrt(cfg.r_curvature_rate)
        w_prev = math.tan(prev_steering * cfg.max_steer_angle_rad) / cfg.wheelbase_m
        diff = np.eye(n)
        diff[np.arange(1, n), np.arange(n - 1)] = -1.0
        b_rate = np.zeros(n)
        b_rate[0] = sqrt_rd * w_prev
        a_ls = np.vstack([weights[:, None] * gamma, sqrt_r * np.eye(n), sqrt_rd * diff])
        b_ls = np.concatenate([-weights * (s_mat @ e0 + c_vec), sqrt_r * kappa_ref, b_rate])

        w_max = cfg.curvature_limit
        result = lsq_linear(a_ls, b_ls, bounds=(-w_max, w_max), method="bvls")
        w_opt = result.x

        # Invert to a steering command and shape it at the command edge.
        steering = math.atan(cfg.wheelbase_m * w_opt[0]) / cfg.max_steer_angle_rad
        slew = cfg.steer_rate_limit * dt_ctrl
        steering = min(max(steering, prev_steering - slew), prev_steering + slew)
        steering = min(max(steering, -cfg.steer_cmd_limit), cfg.steer_cmd_limit)

        return MpcSolution(
            steering=float(steering),
            curvature_sequence=w_opt,
            predicted_errors=(s_mat @ e0 + gamma @ w_opt + c_vec).reshape(n, 2),
            solve_time_s=time.perf_counter() - t0,
        )
