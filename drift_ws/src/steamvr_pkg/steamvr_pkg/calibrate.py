"""
Calibration utility for the SteamVR tracking source.

The world frame is *defined* by the lighthouse reference file
(``config/lighthouses_world.yaml``); ``steamvr_node`` fits
``world -> steamvr_world`` against it automatically at startup, so there is
no routine manual calibration. This tool covers the two remaining cases:

* **Diagnostic (default, no arguments):** run the same fit the node runs and
  print the solved transform with per-lighthouse residuals. Large residuals
  mean the constellation no longer matches the reference file.

* **``--record-reference PATH``:** create a new reference file after the
  arena/lighthouses were rearranged. Place the tracker at the desired world
  origin (floor center) with its **LED pointing along the desired +x axis**;
  the tool records the lighthouse positions in that tracker-defined frame.
  Write to ``drift_ws/src/car_localization_pkg/config/lighthouses_world.yaml``
  and **rebuild** so the installed copy updates.

* **``--write PARAMS_YAML``:** for ``world_calibration: manual`` setups,
  solve the transform with the tracker pinning the origin (tracker at the
  floor center) and write the ``world_*`` values into a params file for
  ``world_anchor_node``.

Requires the bridge (host) and ``steamvr_node`` (container) to be running;
raw TF is used, so the node's current calibration state does not matter.
"""

import argparse
import os
import re
import sys
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

from car_localization_pkg.lighthouse_calibration import (
    load_lighthouse_reference,
    normalize_serial,
    solve_world_calibration,
    solve_world_from_lighthouses,
)
from shared_pkg import Frames
from vehicle_simulation.transforms import rotate_vector_by_quaternion

_WRITE_KEYS = ("world_x", "world_y", "world_z", "world_yaw_deg")


def _collect_tf(node: Node, buffer: Buffer, tracking_frame: str, tracker_frame: str, wait_s):
    """Collect lighthouse positions and the tracker pose from TF (or die trying)."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    frames = yaml.safe_load(buffer.all_frames_as_yaml()) or {}
    lighthouses = {}
    for frame, info in frames.items():
        # normalize_serial: libsurvive may name the frames by decimal/hex ID
        # instead of LHB-...; key by the canonical serial like the anchor node.
        serial = normalize_serial(frame)
        if serial is None or info.get("parent") != tracking_frame:
            continue
        try:
            tf = buffer.lookup_transform(tracking_frame, frame, rclpy.time.Time())
        except TransformException:
            continue  # listed but raced the buffer — skip, don't crash
        t = tf.transform.translation
        lighthouses[serial] = np.array([t.x, t.y, t.z])
    tracker_position = tracker_quat = None
    try:
        tf = buffer.lookup_transform(tracking_frame, tracker_frame, rclpy.time.Time())
        t, r = tf.transform.translation, tf.transform.rotation
        tracker_position = np.array([t.x, t.y, t.z])
        tracker_quat = (r.x, r.y, r.z, r.w)
    except TransformException:
        pass  # only needed by the tracker-based modes
    return lighthouses, tracker_position, tracker_quat


def _require_tracker(tracker_position, tracking_frame: str, tracker_frame: str):
    if tracker_position is None:
        raise SystemExit(
            f"No {tracking_frame} -> {tracker_frame} TF — this mode needs the tracker "
            "placed at the floor center and visible to the lighthouses."
        )


def _record_reference(path: str, lighthouses: dict, tracker_position, tracker_quat) -> None:
    """Write a new reference file for the world frame defined by the tracker."""
    # LED direction = tracker_link +y (see lib/conversion.py); its horizontal
    # projection defines the world +x axis, the tracker position the origin.
    led = rotate_vector_by_quaternion(tracker_quat, (0.0, 1.0, 0.0))
    if float(np.hypot(led[0], led[1])) < 0.5:
        raise SystemExit("Tracker LED points near-vertical — lay the tracker flat.")
    alpha = float(np.arctan2(led[1], led[0]))
    c, s = np.cos(-alpha), np.sin(-alpha)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    lines = [
        "# Reference positions of the lighthouses in the physical world frame [m].",
        "# Defines the world frame: world_anchor_node (car_localization_pkg) fits",
        "# world -> <source>_world against these at startup, for every source.",
        "#",
        "# Recorded by `steamvr_calibrate --record-reference`: tracker at the floor",
        "# center (world origin), LED along +x. Re-record after rearranging the",
        "# lighthouses, then rebuild (colcon build --packages-select car_localization_pkg).",
        "lighthouses:",
    ]
    for serial in sorted(lighthouses):
        p = rot @ (lighthouses[serial] - tracker_position)
        lines.append(f"  {serial}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        print(f"  {serial}: world ({p[0]:7.3f}, {p[1]:7.3f}, {p[2]:6.3f})")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lighthouses)} lighthouses to {path} — rebuild to install:")
    print("  colcon build --packages-select car_localization_pkg")


def _write_params(path: str, result) -> None:
    text = open(path).read()
    for key in _WRITE_KEYS:
        value = getattr(result, key)
        text, n = re.subn(rf"(?m)^(\s*{key}:\s*)[-+0-9.eE]+", rf"\g<1>{value:.4f}", text)
        if n != 1:
            raise SystemExit(f"expected exactly one '{key}:' entry in {path}, found {n}")
    open(path, "w").write(text)


def _print_result(result) -> None:
    print(f"\nSolved from {len(result.residuals_m)} lighthouses:")
    for serial, residual in sorted(result.residuals_m.items()):
        flag = "  <-- moved? (--record-reference to redefine)" if residual > 0.15 else ""
        print(f"  {serial}: residual {residual * 100:5.1f} cm{flag}")
    print(
        f"\n  world_x: {result.world_x:.4f}\n  world_y: {result.world_y:.4f}"
        f"\n  world_z: {result.world_z:.4f}\n  world_yaw_deg: {result.world_yaw_deg:.3f}"
    )


def main(args=None) -> None:
    """Diagnose, record, or write the SteamVR world calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    default_reference = os.path.join(
        get_package_share_directory("car_localization_pkg"), "config", "lighthouses_world.yaml"
    )
    parser.add_argument("--reference", default=default_reference)
    parser.add_argument("--tracking-frame", default=Frames.STEAMVR_WORLD)
    parser.add_argument("--tracker-frame", default=Frames.TRACKER_LINK)
    parser.add_argument("--wait", type=float, default=4.0, help="TF collection time [s]")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SERIAL",
        help="Ignore a lighthouse (repeatable), e.g. one with a bad solve",
    )
    parser.add_argument(
        "--record-reference",
        metavar="REFERENCE_YAML",
        help="Record a NEW reference file from the tracker-defined frame "
        "(tracker at the floor center, LED along +x); then rebuild",
    )
    parser.add_argument(
        "--write",
        metavar="PARAMS_YAML",
        help="Manual mode: solve with the tracker pinning the origin and "
        "rewrite world_x/y/z/yaw_deg in this params file (then rebuild)",
    )
    opts = parser.parse_args(args=args)

    rclpy.init()
    node = rclpy.create_node("steamvr_calibrate")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa: F841  (must stay referenced)
    print(f"Collecting TF for {opts.wait:.0f} s ...")
    try:
        lighthouses, tracker_position, tracker_quat = _collect_tf(
            node, buffer, opts.tracking_frame, opts.tracker_frame, opts.wait
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    for serial in opts.exclude:
        lighthouses.pop(serial, None)
    if not lighthouses:
        raise SystemExit("No lighthouse TFs found — is steamvr_node receiving bridge data?")

    if opts.record_reference:
        _require_tracker(tracker_position, opts.tracking_frame, opts.tracker_frame)
        _record_reference(opts.record_reference, lighthouses, tracker_position, tracker_quat)
        sys.exit(0)

    reference = load_lighthouse_reference(opts.reference)
    unknown = sorted(set(lighthouses) - set(reference))
    if unknown:
        print(f"Ignoring lighthouses not in the reference: {', '.join(unknown)}")

    if opts.write:
        _require_tracker(tracker_position, opts.tracking_frame, opts.tracker_frame)
        result = solve_world_calibration(lighthouses, reference, tracker_position)
        _print_result(result)
        _write_params(opts.write, result)
        print(f"\nWrote {opts.write} — if that file is installed by a package, rebuild it.")
    else:
        result = solve_world_from_lighthouses(lighthouses, reference)
        _print_result(result)
        print("\nDiagnostic only — steamvr_node solves this automatically at startup.")
    sys.exit(0)


if __name__ == "__main__":
    main()
