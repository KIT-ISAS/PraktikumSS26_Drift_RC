"""Cubic bezier track: loading from YAML and discretization into waypoints."""

import math

import yaml


def _eval(seg: list, t: float) -> tuple[float, float]:
    u = 1.0 - t
    c = (u**3, 3 * u**2 * t, 3 * u * t**2, t**3)
    return (
        sum(c[i] * seg[i][0] for i in range(4)),
        sum(c[i] * seg[i][1] for i in range(4)),
    )


def _deriv(seg: list, t: float) -> tuple[float, float]:
    u = 1.0 - t
    c = (3 * u**2, 6 * u * t, 3 * t**2)
    q = [[seg[i + 1][j] - seg[i][j] for j in range(2)] for i in range(3)]
    return (
        sum(c[i] * q[i][0] for i in range(3)),
        sum(c[i] * q[i][1] for i in range(3)),
    )


def _second_deriv(seg: list, t: float) -> tuple[float, float]:
    c = (6 * (1.0 - t), 6 * t)
    q = [[seg[i + 1][j] - seg[i][j] for j in range(2)] for i in range(3)]
    r = [[q[i + 1][j] - q[i][j] for j in range(2)] for i in range(2)]
    return (
        sum(c[i] * r[i][0] for i in range(2)),
        sum(c[i] * r[i][1] for i in range(2)),
    )


class BezierTrack:
    """Closed track represented as a sequence of cubic bezier segments."""

    def __init__(self, name: str, segments: list, frame_id: str = "world") -> None:
        self.name = name
        self.segments = segments
        self.frame_id = frame_id

    @staticmethod
    def from_yaml(path: str) -> "BezierTrack":
        """Load from a YAML file produced by the track config format."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return BezierTrack(
            name=data["name"],
            segments=data["segments"],
            frame_id=data.get("frame_id", "world"),
        )

    def discretize(
        self, points_per_segment: int = 20
    ) -> tuple[list[tuple[float, float, float]], list[float], list[float]]:
        """
        Return (waypoints, curvature, cumulative_distance) for the full track.

        waypoints: (x, y, heading_rad) per point
        curvature: signed 1/r — positive = CCW (left turn)
        cumulative_distance: arc length from the first waypoint
        """
        xs, ys, headings, curvatures = [], [], [], []

        for seg in self.segments:
            for i in range(points_per_segment):
                t = i / points_per_segment
                x, y = _eval(seg, t)
                dx, dy = _deriv(seg, t)
                d2x, d2y = _second_deriv(seg, t)
                speed_sq = dx * dx + dy * dy
                kappa = (
                    (dx * d2y - dy * d2x) / (speed_sq**1.5)
                    if speed_sq > 1e-9
                    else 0.0
                )
                xs.append(x)
                ys.append(y)
                headings.append(math.atan2(dy, dx))
                curvatures.append(float(kappa))

        cumul = [0.0]
        for i in range(1, len(xs)):
            cumul.append(cumul[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))

        return list(zip(xs, ys, headings)), curvatures, cumul
