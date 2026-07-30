"""Unit tests for the world-calibration solver (no ROS required)."""

import math

import numpy as np
import pytest

from car_localization_pkg.lighthouse_calibration import (
    load_lighthouse_reference,
    normalize_serial,
    solve_world_calibration,
    solve_world_from_lighthouses,
)

REFERENCE = {
    "LHB-AAAA0001": np.array([-2.7, -2.3, 2.2]),
    "LHB-AAAA0002": np.array([-2.7, 2.3, 2.2]),
    "LHB-AAAA0003": np.array([2.8, 2.3, 2.2]),
    "LHB-AAAA0004": np.array([2.8, -2.3, 2.3]),
}


def _inverse_map(p_world, yaw, t):
    """steamvr_world position of a world point, for a given world->steamvr calib."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot.T @ (np.asarray(p_world) - t)


def _synthetic_measurements(yaw, t):
    measured = {k: _inverse_map(v, yaw, t) for k, v in REFERENCE.items()}
    tracker = _inverse_map((0.0, 0.0, 0.0), yaw, t)  # tracker at world origin
    return measured, tracker


def test_exact_roundtrip_recovers_transform():
    yaw, t = math.radians(42.4), np.array([-2.63, -2.41, 2.22])
    measured, tracker = _synthetic_measurements(yaw, t)
    result = solve_world_calibration(measured, REFERENCE, tracker)
    assert result.world_yaw_deg == pytest.approx(math.degrees(yaw))
    assert (result.world_x, result.world_y, result.world_z) == pytest.approx(tuple(t))
    assert all(r == pytest.approx(0.0, abs=1e-9) for r in result.residuals_m.values())
    assert result.apply(tracker) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_noisy_measurements_recover_approximately():
    yaw, t = math.radians(-77.0), np.array([1.5, -0.5, 2.0])
    measured, tracker = _synthetic_measurements(yaw, t)
    rng = np.random.default_rng(7)
    measured = {k: v + rng.normal(0.0, 0.01, 3) for k, v in measured.items()}
    result = solve_world_calibration(measured, REFERENCE, tracker)
    assert result.world_yaw_deg == pytest.approx(math.degrees(yaw), abs=0.5)
    assert max(result.residuals_m.values()) < 0.05


def test_extra_serials_outside_reference_are_ignored():
    yaw, t = math.radians(10.0), np.array([0.5, 0.5, 1.0])
    measured, tracker = _synthetic_measurements(yaw, t)
    measured["LHB-UNKNOWN1"] = np.array([99.0, 99.0, 99.0])
    result = solve_world_calibration(measured, REFERENCE, tracker)
    assert "LHB-UNKNOWN1" not in result.residuals_m
    assert result.world_yaw_deg == pytest.approx(math.degrees(yaw))


def test_too_few_correspondences_raises():
    with pytest.raises(ValueError, match="at least 2"):
        solve_world_calibration(
            {"LHB-AAAA0001": REFERENCE["LHB-AAAA0001"]}, REFERENCE, np.zeros(3)
        )


def test_lighthouse_only_solve_recovers_transform():
    # Auto-calibration path: no tracker pin, translation from the centroids
    # and z from the mean height difference.
    yaw, t = math.radians(123.0), np.array([0.7, -1.9, 2.25])
    measured = {k: _inverse_map(v, yaw, t) for k, v in REFERENCE.items()}
    result = solve_world_from_lighthouses(measured, REFERENCE)
    assert result.world_yaw_deg == pytest.approx(math.degrees(yaw))
    assert (result.world_x, result.world_y, result.world_z) == pytest.approx(tuple(t))
    assert all(r == pytest.approx(0.0, abs=1e-9) for r in result.residuals_m.values())


def test_lighthouse_only_solve_flags_moved_lighthouse_in_residuals():
    yaw, t = math.radians(10.0), np.array([0.0, 0.0, 2.0])
    measured = {k: _inverse_map(v, yaw, t) for k, v in REFERENCE.items()}
    measured["LHB-AAAA0004"] = measured["LHB-AAAA0004"] + np.array([0.5, 0.0, 0.0])
    result = solve_world_from_lighthouses(measured, REFERENCE)
    worst = max(result.residuals_m, key=result.residuals_m.get)
    assert worst == "LHB-AAAA0004"
    assert result.residuals_m[worst] > 0.3


def test_lighthouse_only_solve_too_few_raises():
    with pytest.raises(ValueError, match="at least 2"):
        solve_world_from_lighthouses({"LHB-AAAA0001": np.zeros(3)}, REFERENCE)


def test_load_lighthouse_reference(tmp_path):
    path = tmp_path / "ref.yaml"
    path.write_text("lighthouses:\n  LHB-00000001: [1.0, 2.0, 3.0]\n")
    ref = load_lighthouse_reference(str(path))
    assert ref["LHB-00000001"] == pytest.approx((1.0, 2.0, 3.0))


@pytest.mark.parametrize(
    ("frame_name", "expected"),
    [
        ("LHB-132B7D13", "LHB-132B7D13"),  # SteamVR serial: passthrough
        ("321617171", "LHB-132B7D13"),  # libsurvive numeric ID (decimal)
        ("132b7d13", "LHB-132B7D13"),  # bare hex, lowercase
        ("tracker_link", None),
        ("world", None),
        ("LHR-FA4A8BDB", None),  # a tracker serial is not a lighthouse
    ],
)
def test_normalize_serial(frame_name, expected):
    assert normalize_serial(frame_name) == expected
