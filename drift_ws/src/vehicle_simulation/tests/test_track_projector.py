"""Unit tests for ``TrackProjector`` using analytically computable centerlines."""

import math

import pytest

from vehicle_simulation.track import TrackProjector

SIDE = 4.0
SPACING = 1.0


def _square_centerline() -> tuple[list[tuple[float, float, float]], list[float]]:
    """
    CCW square with corners (0,0), (4,0), (4,4), (0,4), one waypoint per meter.

    The closing segment runs from (0, 1) back to (0, 0), so the total length is 16 m.
    """
    n = int(SIDE / SPACING)
    waypoints = []
    for i in range(n):
        waypoints.append((i * SPACING, 0.0, 0.0))
    for i in range(n):
        waypoints.append((SIDE, i * SPACING, math.pi / 2))
    for i in range(n):
        waypoints.append((SIDE - i * SPACING, SIDE, math.pi))
    for i in range(n):
        waypoints.append((0.0, SIDE - i * SPACING, -math.pi / 2))
    cumulative = [i * SPACING for i in range(len(waypoints))]
    return waypoints, cumulative


def _thin_loop_centerline(
    length: float = 10.0, gap: float = 0.3, spacing: float = 0.5
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Closed loop whose outbound (y=0) and return (y=gap) branches run close together."""
    n = int(length / spacing)
    waypoints = []
    for i in range(n):
        waypoints.append((i * spacing, 0.0, 0.0))
    for i in range(n):
        waypoints.append((length - i * spacing, gap, math.pi))
    cumulative = [0.0]
    for i in range(1, len(waypoints)):
        step = math.hypot(
            waypoints[i][0] - waypoints[i - 1][0], waypoints[i][1] - waypoints[i - 1][1]
        )
        cumulative.append(cumulative[-1] + step)
    return waypoints, cumulative


def _square_projector() -> TrackProjector:
    waypoints, cumulative = _square_centerline()
    return TrackProjector(waypoints, cumulative)


def _drive(projector: TrackProjector, waypoints, reverse: bool = False):
    """Project every waypoint in (reverse) order, returning the last projection."""
    ordered = list(reversed(waypoints)) if reverse else list(waypoints)
    projection = None
    for x, y, _ in ordered:
        projection = projector.project(x, y)
    return projection


class TestProjection:
    """Geometric correctness of single projections."""

    def test_straight_stretch(self):
        projection = _square_projector().project(1.5, 0.0)
        assert projection.distance == pytest.approx(1.5)
        assert projection.lateral_error == pytest.approx(0.0)
        assert projection.heading == pytest.approx(0.0)
        assert (projection.x, projection.y) == pytest.approx((1.5, 0.0))
        assert projection.lap == 0
        assert projection.lap_progress == pytest.approx(1.5 / 16.0)

    def test_second_side_heading(self):
        projection = _square_projector().project(4.0, 1.5)
        assert projection.distance == pytest.approx(5.5)
        assert projection.heading == pytest.approx(math.pi / 2)

    def test_lateral_error_left_is_positive(self):
        # Travel direction on the bottom side is +x, so +y is left of the path.
        projection = _square_projector().project(1.5, 0.2)
        assert projection.lateral_error == pytest.approx(0.2)
        assert projection.distance == pytest.approx(1.5)

    def test_lateral_error_right_is_negative(self):
        projection = _square_projector().project(1.5, -0.2)
        assert projection.lateral_error == pytest.approx(-0.2)
        assert projection.distance == pytest.approx(1.5)

    def test_too_few_waypoints_returns_none(self):
        projector = TrackProjector([(0.0, 0.0, 0.0)], [0.0])
        assert projector.project(1.0, 1.0) is None
        projector = TrackProjector([], [])
        assert projector.project(0.0, 0.0) is None


class TestLapCounting:
    """Lap increments, decrements, and reset across the start-line wrap."""

    def test_forward_lap_increments(self):
        waypoints, _ = _square_centerline()
        projector = _square_projector()
        _drive(projector, waypoints)
        # Cross the start line into the next lap.
        projector.project(*waypoints[0][:2])
        projection = projector.project(*waypoints[1][:2])
        assert projection.lap == 1
        assert projection.distance == pytest.approx(1.0)

    def test_reverse_traversal_decrements(self):
        waypoints, _ = _square_centerline()
        projector = _square_projector()
        _drive(projector, waypoints)
        projector.project(*waypoints[0][:2])
        assert projector.project(*waypoints[1][:2]).lap == 1
        # Drive backwards across the start line.
        projector.project(*waypoints[0][:2])
        projection = projector.project(*waypoints[-1][:2])
        assert projection.lap == 0

    def test_lap_never_negative(self):
        waypoints, _ = _square_centerline()
        projector = _square_projector()
        projector.project(*waypoints[1][:2])
        # Reverse over the start line without ever completing a lap.
        projector.project(*waypoints[0][:2])
        projection = projector.project(*waypoints[-1][:2])
        assert projection.lap == 0

    def test_reset_clears_lap_and_relocalizes(self):
        waypoints, _ = _square_centerline()
        projector = _square_projector()
        _drive(projector, waypoints)
        projector.project(*waypoints[0][:2])
        assert projector.project(*waypoints[1][:2]).lap == 1

        projector.reset()
        projection = projector.project(10.0, 1.5)  # far off-track, near side two
        assert projection.lap == 0
        assert projection.distance == pytest.approx(5.5)


class TestBranchDisambiguation:
    """The search window keeps the projection on the branch being tracked."""

    def test_fresh_projector_picks_nearest_branch(self):
        waypoints, cumulative = _thin_loop_centerline()
        projector = TrackProjector(waypoints, cumulative)
        projection = projector.project(5.0, 0.1)
        # Globally nearest branch is the outbound one (y = 0).
        assert projection.distance == pytest.approx(5.0)
        assert projection.lateral_error == pytest.approx(0.1)

    def test_tracked_projector_stays_on_far_branch(self):
        waypoints, cumulative = _thin_loop_centerline(length=10.0, gap=0.3)
        projector = TrackProjector(waypoints, cumulative)
        # Localize on the return branch (y = 0.3), which runs in -x direction.
        seeded = projector.project(5.0, 0.25)
        assert seeded.distance > 10.0

        # The car drifts towards the outbound branch but is still within the
        # lost-distance threshold of the return branch: stay on it.
        projection = projector.project(5.0, 0.1)
        assert projection.distance > 10.0
        assert projection.y == pytest.approx(0.3)
        assert abs(projection.lateral_error) == pytest.approx(0.2)
