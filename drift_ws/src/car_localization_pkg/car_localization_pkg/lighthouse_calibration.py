"""
World-calibration solver: fit ``world -> <source>_world`` from lighthouses.

The **definition** of the physical world frame is the lighthouse reference
file (``config/lighthouses_world.yaml``): it states where each base station
sits in world coordinates, keyed by serial (``LHB-XXXXXXXX`` — SteamVR
reports it directly, libsurvive's numeric ID is the same value in hex).
Given where a tracking source sees those same lighthouses in its own
tracking frame, :func:`solve_world_from_lighthouses` fits the planar
yaw + translation mapping ``p_world = R(yaw) @ p_source + t`` — this is what
``world_anchor_node`` runs at startup for whichever source is active, so the
world frame follows the constellation with no manual procedure and both
sources agree on the same playground. The fit residuals double as a health
check: they only grow when the lighthouses moved *relative to each other*
(i.e. the reference file is stale).

:func:`solve_world_calibration` is the tracker-pinned variant (tracker placed
at the world origin) used when *creating* a reference file for a rearranged
arena — see steamvr_pkg's ``steamvr_calibrate`` script.

ROS-free (numpy + pyyaml only) so it is unit-testable in isolation.
"""

from dataclasses import dataclass
import math

import numpy as np
import yaml

__all__ = [
    "CalibrationResult",
    "load_lighthouse_reference",
    "solve_world_calibration",
    "solve_world_from_lighthouses",
]


@dataclass
class CalibrationResult:
    """Solved ``world -> steamvr_world`` parameters plus fit diagnostics."""

    world_x: float
    world_y: float
    world_z: float
    world_yaw_deg: float
    # Per-serial distance [m] between the mapped and the reference position.
    residuals_m: dict[str, float]

    def apply(self, p_steamvr) -> np.ndarray:
        """Map a steamvr_world position into the world frame."""
        yaw = math.radians(self.world_yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([self.world_x, self.world_y, self.world_z])
        return rot @ np.asarray(p_steamvr, dtype=np.float64) + t


def load_lighthouse_reference(path: str) -> dict[str, np.ndarray]:
    """Load the ``serial -> world position`` reference map from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    lighthouses = data["lighthouses"]
    return {
        str(serial): np.asarray(position, dtype=np.float64)
        for serial, position in lighthouses.items()
    }


def solve_world_from_lighthouses(
    measured: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> CalibrationResult:
    """
    Solve the calibration from lighthouse correspondences alone.

    Planar Procrustes fit: yaw from the centered XY correspondences,
    XY translation from the centroids, z translation from the mean height
    difference. Used by steamvr_node's automatic startup calibration.

    :raises ValueError: If fewer than two lighthouses match the reference.
    """
    serials = sorted(set(measured) & set(reference))
    if len(serials) < 2:
        raise ValueError(
            f"need at least 2 lighthouse correspondences, got {len(serials)} "
            f"(measured: {sorted(measured)}, reference: {sorted(reference)})"
        )
    a = np.stack([np.asarray(measured[s], dtype=np.float64) for s in serials])
    b = np.stack([np.asarray(reference[s], dtype=np.float64) for s in serials])
    a_centered = a[:, :2] - a[:, :2].mean(axis=0)
    b_centered = b[:, :2] - b[:, :2].mean(axis=0)
    cross = a_centered[:, 0] * b_centered[:, 1] - a_centered[:, 1] * b_centered[:, 0]
    dot = a_centered * b_centered
    yaw = math.atan2(float(np.sum(cross)), float(np.sum(dot)))

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = b.mean(axis=0) - rot @ a.mean(axis=0)

    result = CalibrationResult(
        world_x=float(t[0]),
        world_y=float(t[1]),
        world_z=float(t[2]),
        world_yaw_deg=math.degrees(yaw),
        residuals_m={},
    )
    result.residuals_m = {
        serial: float(np.linalg.norm(result.apply(measured[serial]) - reference[serial]))
        for serial in serials
    }
    return result


def solve_world_calibration(
    measured: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    tracker: np.ndarray,
) -> CalibrationResult:
    """
    Solve the calibration from lighthouse correspondences and the origin tracker.

    :param measured: Lighthouse positions in ``steamvr_world``, keyed by serial.
    :param reference: Known lighthouse positions in ``world`` (same serials).
    :param tracker: Tracker position in ``steamvr_world``; it marks the world
        origin (tracker physically placed at the floor center).
    :raises ValueError: If fewer than two lighthouses match the reference.
    """
    serials = sorted(set(measured) & set(reference))
    if len(serials) < 2:
        raise ValueError(
            f"need at least 2 lighthouse correspondences, got {len(serials)} "
            f"(measured: {sorted(measured)}, reference: {sorted(reference)})"
        )
    tracker = np.asarray(tracker, dtype=np.float64)

    # Yaw via 2D Procrustes with the tracker as the pinned common origin.
    cross = dot = 0.0
    for serial in serials:
        a = (np.asarray(measured[serial], dtype=np.float64) - tracker)[:2]
        b = np.asarray(reference[serial], dtype=np.float64)[:2]
        cross += a[0] * b[1] - a[1] * b[0]
        dot += a @ b
    yaw = math.atan2(cross, dot)

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = -rot @ tracker  # tracker -> world origin

    result = CalibrationResult(
        world_x=float(t[0]),
        world_y=float(t[1]),
        world_z=float(t[2]),
        world_yaw_deg=math.degrees(yaw),
        residuals_m={},
    )
    result.residuals_m = {
        serial: float(np.linalg.norm(result.apply(measured[serial]) - reference[serial]))
        for serial in serials
    }
    return result
