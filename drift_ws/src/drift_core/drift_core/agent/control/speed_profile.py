"""
Curvature-based speed profile for a closed track (ROS-free, pure numpy).

Given the centerline arrays a ``BezierTrack``/``Track`` msg carries (signed
curvature and cumulative arc length) plus acceleration limits, this module
precomputes the fastest speed the car may carry at every arc length:

1. Cap by lateral grip: ``v = sqrt(a_lat_max / |kappa|)``, clipped to ``v_max``.
2. Backward pass: brake early enough that every upcoming cap is reachable
   within ``a_brake``.
3. Forward pass: never demand more acceleration than ``a_accel`` delivers.

The braking/acceleration passes run over the profile twice in sequence so the
limits propagate through the closing seam of the closed track — the profile is
continuous across the wrap.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["SpeedProfile", "build_speed_profile"]


@dataclass(frozen=True)
class SpeedProfile:
    """Precomputed speed reference over arc length, wrap-aware via :meth:`sample`."""

    s_grid: np.ndarray  # (M+1,) arc lengths [m], 0 .. total_length inclusive
    v_grid: np.ndarray  # (M+1,) speed reference [m/s], v_grid[-1] == v_grid[0]
    total_length: float  # closed-track length [m], including the closing chord

    def sample(self, s):
        """Interpolate the speed reference at arc length ``s`` (scalar or array), wrapping."""
        return np.interp(np.mod(s, self.total_length), self.s_grid, self.v_grid)


def _propagate(v: np.ndarray, ds: float, a_limit: float, backward: bool) -> np.ndarray:
    """One accel-limit pass over the doubled profile, returning the converged copy."""
    doubled = np.concatenate([v, v])
    if backward:
        for i in range(len(doubled) - 2, -1, -1):
            doubled[i] = min(doubled[i], np.sqrt(doubled[i + 1] ** 2 + 2.0 * a_limit * ds))
        return doubled[: len(v)]
    for i in range(len(doubled) - 1):
        doubled[i + 1] = min(doubled[i + 1], np.sqrt(doubled[i] ** 2 + 2.0 * a_limit * ds))
    return doubled[len(v) :]


def build_speed_profile(
    curvature,
    cumulative_distance,
    total_length: float,
    *,
    v_max: float,
    a_lat_max: float,
    a_brake: float,
    a_accel: float,
    ds: float = 0.05,
) -> SpeedProfile:
    """
    Build the wrap-aware speed profile from the centerline curvature arrays.

    ``curvature``/``cumulative_distance`` are exactly what ``BezierTrack.discretize()``
    returns and the ``Track`` msg carries; ``total_length`` includes the closing
    chord (a phantom sample with the first waypoint's curvature closes the profile,
    matching ``curvature_lookahead``).
    """
    kappa = np.asarray(curvature, dtype=np.float64)
    dist = np.asarray(cumulative_distance, dtype=np.float64)
    kappa_closed = np.append(kappa, kappa[0])
    dist_closed = np.append(dist, total_length)

    n = max(int(np.ceil(total_length / ds)), 2)
    s_grid = np.linspace(0.0, total_length, n, endpoint=False)
    grid_ds = float(total_length / n)
    kappa_grid = np.interp(s_grid, dist_closed, kappa_closed)

    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(np.abs(kappa_grid), 1e-6)))
    v = _propagate(v, grid_ds, a_brake, backward=True)
    v = _propagate(v, grid_ds, a_accel, backward=False)

    return SpeedProfile(
        s_grid=np.append(s_grid, total_length),
        v_grid=np.append(v, v[0]),
        total_length=float(total_length),
    )
