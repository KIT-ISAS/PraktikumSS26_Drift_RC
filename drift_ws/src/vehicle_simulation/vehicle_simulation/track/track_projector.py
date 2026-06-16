"""
Project a car position onto a discretized closed track and track lap progress.

The projector is intentionally free of ROS dependencies so it can be unit-tested in
isolation. It consumes the same ``waypoints`` and ``cumulative_distance`` that
``BezierTrack.discretize`` produces.
"""

import bisect
from dataclasses import dataclass
from math import atan2, ceil, copysign, hypot


@dataclass
class Projection:
    """Result of projecting a point onto the track."""

    x: float  # projected point on the centerline
    y: float
    heading: float  # track tangent direction at the projection (rad)
    distance: float  # arc length along the centerline (m)
    lateral_error: float  # signed car->centerline distance (m), + = car left of path
    lap: int  # completed laps since start
    lap_progress: float  # current-lap fraction, 0..1


class TrackProjector:
    """
    Projects a 2D point onto a closed centerline with crossing disambiguation.

    The track is treated as closed: a final segment connects the last waypoint back to
    the first. Crossings (e.g. the figure-8 origin) are disambiguated by searching only a
    small arc-length window around the previous projection, since the two branches that
    meet at a crossing are far apart in arc length.
    """

    def __init__(
        self,
        waypoints: list[tuple[float, float, float]],
        cumulative_distance: list[float],
        search_window_m: float = 0.6,
        lost_distance_m: float = 0.5,
    ) -> None:
        """Store the centerline and configure the search window and lost threshold."""
        self._wp = [(float(x), float(y)) for x, y, _ in waypoints]
        self._n = len(self._wp)
        self._cumul = [float(d) for d in cumulative_distance]

        # Closed track: the loop is closed by a segment from the last waypoint to the
        # first, so the total length adds that closing chord.
        if self._n >= 2:
            closing = hypot(self._wp[0][0] - self._wp[-1][0], self._wp[0][1] - self._wp[-1][1])
            self.total_length = self._cumul[-1] + closing
        else:
            self.total_length = 0.0

        self._search_window_m = search_window_m
        self._lost_distance_m = lost_distance_m

        self._last_s: float | None = None
        self._lap = 0

    def reset(self) -> None:
        """Forget the previous projection so the next call re-localizes globally."""
        self._last_s = None
        self._lap = 0

    def _segment(self, i: int) -> tuple[float, float, float, float, float, float]:
        """Return (ax, ay, dx, dy, seg_len, s_start) for segment i -> (i+1) mod n."""
        ax, ay = self._wp[i]
        bx, by = self._wp[(i + 1) % self._n]
        dx, dy = bx - ax, by - ay
        return ax, ay, dx, dy, hypot(dx, dy), self._cumul[i]

    def _project_segment(
        self, i: int, px: float, py: float
    ) -> tuple[float, float, float, float, float, float]:
        """Project (px, py) onto segment i; return (dist, x, y, heading, s, lateral)."""
        ax, ay, dx, dy, seg_len, s_start = self._segment(i)
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
        cx, cy = ax + t * dx, ay + t * dy
        dist = hypot(px - cx, py - cy)
        heading = atan2(dy, dx)
        s = s_start + t * seg_len
        # Sign via cross product of tangent and the car offset: + = car left of travel.
        cross = dx * (py - cy) - dy * (px - cx)
        lateral = copysign(dist, cross) if dist > 1e-9 else 0.0
        return dist, cx, cy, heading, s, lateral

    def _search(
        self, px: float, py: float, indices
    ) -> tuple[float, float, float, float, float, float]:
        """Return the best projection over the given segment indices."""
        best = None
        for i in indices:
            cand = self._project_segment(i % self._n, px, py)
            if best is None or cand[0] < best[0]:
                best = cand
        return best

    def project(self, px: float, py: float) -> Projection | None:
        """
        Project (px, py) onto the track, updating lap count and progress.

        Returns None if the track has too few waypoints to define segments.
        """
        if self._n < 2 or self.total_length <= 0.0:
            return None

        if self._last_s is None:
            best = self._search(px, py, range(self._n))
        else:
            avg_spacing = self.total_length / self._n
            half = max(1, int(ceil(self._search_window_m / max(avg_spacing, 1e-6))))
            center = max(0, min(self._n - 1, bisect.bisect_right(self._cumul, self._last_s) - 1))
            best = self._search(px, py, range(center - half, center + half + 1))
            # Car appears off-track: re-localize with a global search.
            if best[0] > self._lost_distance_m:
                global_best = self._search(px, py, range(self._n))
                if global_best[0] < best[0]:
                    best = global_best

        dist, cx, cy, heading, s, lateral = best
        self._update_laps(s)
        self._last_s = s

        return Projection(
            x=cx,
            y=cy,
            heading=heading,
            distance=s,
            lateral_error=lateral,
            lap=self._lap,
            lap_progress=s / self.total_length,
        )

    def _update_laps(self, s: float) -> None:
        """Increment/decrement the lap count on a forward/backward start-line wrap."""
        if self._last_s is None:
            return
        frac_prev = self._last_s / self.total_length
        frac_now = s / self.total_length
        if frac_prev > 0.9 and frac_now < 0.1:
            self._lap += 1
        elif frac_prev < 0.1 and frac_now > 0.9:
            self._lap = max(0, self._lap - 1)
