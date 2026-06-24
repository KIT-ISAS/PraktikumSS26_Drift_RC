"""Unit tests for bezier evaluation and ``BezierTrack`` discretization / YAML loading."""

import math

import pytest
import yaml

from vehicle_simulation.track import BezierTrack
from vehicle_simulation.track.bezier_track import _deriv, _eval, _second_deriv

# A non-degenerate segment with hand-computable values.
SEGMENT = [[0.0, 0.0], [1.0, 2.0], [3.0, 3.0], [4.0, 0.0]]

# Magic constant for approximating a quarter circle with a cubic bezier.
_K = 0.5522847498307936


def _unit_circle_segments(clockwise: bool = False) -> list:
    """Return four bezier segments approximating the unit circle (CCW by default)."""
    segments = [
        [[1.0, 0.0], [1.0, _K], [_K, 1.0], [0.0, 1.0]],
        [[0.0, 1.0], [-_K, 1.0], [-1.0, _K], [-1.0, 0.0]],
        [[-1.0, 0.0], [-1.0, -_K], [-_K, -1.0], [0.0, -1.0]],
        [[0.0, -1.0], [_K, -1.0], [1.0, -_K], [1.0, 0.0]],
    ]
    if clockwise:
        segments = [list(reversed(seg)) for seg in reversed(segments)]
    return segments


class TestEvaluation:
    """Analytical values of _eval / _deriv / _second_deriv on a known segment."""

    def test_eval_endpoints_and_midpoint(self):
        assert _eval(SEGMENT, 0.0) == pytest.approx((0.0, 0.0))
        assert _eval(SEGMENT, 1.0) == pytest.approx((4.0, 0.0))
        assert _eval(SEGMENT, 0.5) == pytest.approx((2.0, 1.875))

    def test_deriv(self):
        # B'(0) = 3 (p1 - p0), B'(1) = 3 (p3 - p2)
        assert _deriv(SEGMENT, 0.0) == pytest.approx((3.0, 6.0))
        assert _deriv(SEGMENT, 1.0) == pytest.approx((3.0, -9.0))
        assert _deriv(SEGMENT, 0.5) == pytest.approx((4.5, 0.75))

    def test_second_deriv(self):
        # B''(0) = 6 (p2 - 2 p1 + p0), B''(1) = 6 (p3 - 2 p2 + p1)
        assert _second_deriv(SEGMENT, 0.0) == pytest.approx((6.0, -6.0))
        assert _second_deriv(SEGMENT, 1.0) == pytest.approx((-6.0, -24.0))
        assert _second_deriv(SEGMENT, 0.5) == pytest.approx((0.0, -15.0))


class TestDiscretize:
    """Waypoint counts, arc length, and curvature sign of discretize()."""

    def test_waypoint_count(self):
        track = BezierTrack("circle", _unit_circle_segments())
        points_per_segment = 25
        waypoints, curvature, cumul = track.discretize(points_per_segment)
        expected = len(track.segments) * points_per_segment
        assert len(waypoints) == expected
        assert len(curvature) == expected
        assert len(cumul) == expected

    def test_cumulative_distance_monotonic(self):
        track = BezierTrack("circle", _unit_circle_segments())
        _, _, cumul = track.discretize()
        assert cumul[0] == 0.0
        assert all(b >= a for a, b in zip(cumul, cumul[1:]))

    def test_circle_arc_length(self):
        points_per_segment = 50
        track = BezierTrack("circle", _unit_circle_segments())
        _, _, cumul = track.discretize(points_per_segment)
        # The last sample sits one step before closing the loop.
        n = len(track.segments) * points_per_segment
        expected = 2.0 * math.pi * (n - 1) / n
        assert cumul[-1] == pytest.approx(expected, rel=1e-3)

    def test_curvature_sign_ccw_positive(self):
        track = BezierTrack("circle_ccw", _unit_circle_segments())
        _, curvature, _ = track.discretize(50)
        assert all(kappa > 0.0 for kappa in curvature)
        assert all(kappa == pytest.approx(1.0, rel=0.05) for kappa in curvature)

    def test_curvature_sign_cw_negative(self):
        track = BezierTrack("circle_cw", _unit_circle_segments(clockwise=True))
        _, curvature, _ = track.discretize(50)
        assert all(kappa < 0.0 for kappa in curvature)
        assert all(kappa == pytest.approx(-1.0, rel=0.05) for kappa in curvature)

    def test_straight_segment_heading_and_zero_curvature(self):
        straight = [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]
        waypoints, curvature, _ = BezierTrack("line", straight).discretize(10)
        assert all(heading == pytest.approx(0.0) for _, _, heading in waypoints)
        assert all(kappa == pytest.approx(0.0, abs=1e-9) for kappa in curvature)


class TestFromYaml:
    """Round-trip of the YAML track config format."""

    def test_round_trip(self, tmp_path):
        segments = _unit_circle_segments()
        config = {"name": "unit_circle", "frame_id": "test_frame", "segments": segments}
        path = tmp_path / "track.yaml"
        path.write_text(yaml.safe_dump(config))

        track = BezierTrack.from_yaml(str(path))
        assert track.name == "unit_circle"
        assert track.frame_id == "test_frame"
        assert len(track.segments) == len(segments)
        assert track.segments == segments

    def test_frame_id_defaults_to_world(self, tmp_path):
        config = {"name": "minimal", "segments": [SEGMENT]}
        path = tmp_path / "track.yaml"
        path.write_text(yaml.safe_dump(config))

        track = BezierTrack.from_yaml(str(path))
        assert track.frame_id == "world"


class TestLoadTrack:
    """Loading packaged tracks by name and explicit YAML paths."""

    def test_packaged_track_by_name(self):
        from vehicle_simulation.track import list_tracks, load_track

        assert "figure_8" in list_tracks()
        track = load_track("figure_8")
        assert track.name == "figure_8"
        assert len(track.segments) > 0

    def test_explicit_path(self, tmp_path):
        from vehicle_simulation.track import load_track

        config = {"name": "minimal", "segments": [SEGMENT]}
        path = tmp_path / "track.yaml"
        path.write_text(yaml.safe_dump(config))
        track = load_track(str(path))
        assert track.name == "minimal"
