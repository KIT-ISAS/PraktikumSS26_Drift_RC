#!/usr/bin/env python3
"""
Host-side SteamVR bridge: polls OpenVR tracker poses and streams them via UDP.

Runs **on the host** (where SteamVR runs), NOT in the dev container: OpenVR
clients talk to vrserver over local IPC that is not available inside the
container. The container side (``steamvr_pkg``'s ``steamvr_node``) receives the
datagrams on localhost — the dev container uses host networking.

The script is deliberately dumb: it forwards the raw OpenVR matrices and
tracking-space velocities untouched; all coordinate conversion happens in the
unit-tested ROS package. The wire format is owned by
``drift_ws/src/steamvr_pkg/steamvr_pkg/lib/protocol.py`` (stdlib-only, imported
from the repo checkout so both ends always agree).

Usage (see steamvr_bridge/README.md):
    python3 steamvr_bridge.py [--host 127.0.0.1] [--port 15250] [--rate 100]
                              [--universe standing|raw]
"""

import argparse
import pathlib
import socket
import sys
import time

import openvr

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent.parent / "drift_ws" / "src" / "steamvr_pkg")
)
from steamvr_pkg.lib.protocol import (  # noqa: E402
    DEFAULT_PORT,
    DEVICE_CLASS_LIGHTHOUSE,
    DEVICE_CLASS_TRACKER,
    TrackerSample,
    encode_sample,
)

# Lighthouses are static; resend them about once a second for late joiners.
LIGHTHOUSE_INTERVAL_S = 1.0
_UNIVERSES = {
    "standing": openvr.TrackingUniverseStanding,
    "raw": openvr.TrackingUniverseRawAndUncalibrated,
}
_FORWARDED_CLASSES = (DEVICE_CLASS_TRACKER, DEVICE_CLASS_LIGHTHOUSE)


def _init_openvr_blocking() -> openvr.IVRSystem:
    """Connect to the SteamVR runtime, retrying until it is available."""
    while True:
        try:
            vr_system = openvr.init(openvr.VRApplication_Background)
            print("Connected to SteamVR.")
            return vr_system
        except openvr.error_code.OpenVRError as e:
            print(f"SteamVR not available ({e}); retrying in 2 s ...")
            time.sleep(2.0)


def _flatten_matrix34(m) -> list[float]:
    return [float(m[row][col]) for row in range(3) for col in range(4)]


class SerialCache:
    """Caches the immutable per-device-index serial number lookups."""

    def __init__(self, vr_system: openvr.IVRSystem):
        self._vr_system = vr_system
        self._serials: dict[int, str] = {}

    def get(self, device_index: int) -> str:
        if device_index not in self._serials:
            serial = self._vr_system.getStringTrackedDeviceProperty(
                device_index, openvr.Prop_SerialNumber_String
            )
            self._serials[device_index] = serial
            print(f"Device {device_index}: serial {serial}")
        return self._serials[device_index]


def run(host: str, port: int, rate: float, universe: int) -> None:
    """Poll OpenVR at the given rate and forward device samples over UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose_array = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
    period = 1.0 / rate

    while True:
        vr_system = _init_openvr_blocking()
        serials = SerialCache(vr_system)
        next_tick = time.monotonic()
        last_lighthouse_send = 0.0
        try:
            while True:
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

                vr_system.getDeviceToAbsoluteTrackingPose(universe, 0.0, pose_array)
                t_mono = time.monotonic()
                t_wall = time.time()
                send_lighthouses = t_mono - last_lighthouse_send >= LIGHTHOUSE_INTERVAL_S
                for index in range(openvr.k_unMaxTrackedDeviceCount):
                    device_class = vr_system.getTrackedDeviceClass(index)
                    if device_class not in _FORWARDED_CLASSES:
                        continue
                    if device_class == DEVICE_CLASS_LIGHTHOUSE and not send_lighthouses:
                        continue
                    pose = pose_array[index]
                    if not pose.bDeviceIsConnected:
                        continue
                    sample = TrackerSample(
                        serial=serials.get(index),
                        device_class=int(device_class),
                        t_mono=t_mono,
                        t_wall=t_wall,
                        pose_valid=bool(pose.bPoseIsValid),
                        device_connected=bool(pose.bDeviceIsConnected),
                        tracking_result=int(pose.eTrackingResult),
                        matrix=_flatten_matrix34(pose.mDeviceToAbsoluteTracking),
                        velocity=[float(v) for v in pose.vVelocity.v],
                        angular_velocity=[float(v) for v in pose.vAngularVelocity.v],
                    )
                    sock.sendto(encode_sample(sample), (host, port))
                if send_lighthouses:
                    last_lighthouse_send = t_mono
        except openvr.error_code.OpenVRError as e:
            print(f"Lost connection to SteamVR ({e}); reconnecting ...")
            openvr.shutdown()


def main() -> None:
    """Parse CLI arguments and run the bridge until Ctrl-C."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="destination address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="destination UDP port")
    parser.add_argument("--rate", type=float, default=100.0, help="poll rate [Hz]")
    parser.add_argument(
        "--universe",
        choices=sorted(_UNIVERSES),
        default="standing",
        help="OpenVR tracking universe (standing = floor-calibrated room setup)",
    )
    args = parser.parse_args()
    try:
        run(args.host, args.port, args.rate, _UNIVERSES[args.universe])
    except KeyboardInterrupt:
        print("Bridge stopped.")
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
