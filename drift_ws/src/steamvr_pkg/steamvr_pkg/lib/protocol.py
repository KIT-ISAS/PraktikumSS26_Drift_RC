"""
Wire format for the host-side SteamVR bridge -> steamvr_node UDP link.

One JSON datagram per tracked-device sample, carrying the **raw OpenVR data**
(row-major HmdMatrix34, tracking-space velocities); all coordinate conversion
happens container-side in :mod:`steamvr_pkg.lib.conversion`.

This module is deliberately **stdlib-only** (no ROS, no numpy): the bridge
script runs on the host with plain Python + ``openvr`` and imports it via
``sys.path``, so it is the single source of truth for the wire format on both
ends.
"""

from dataclasses import asdict, dataclass
import json

PROTOCOL_VERSION = 1
# 12345 is the ESP32 control link, 8765 the Foxglove bridge.
DEFAULT_PORT = 15250

# openvr.TrackedDeviceClass_* enum values (mirrored so this stays stdlib-only).
DEVICE_CLASS_TRACKER = 3  # TrackedDeviceClass_GenericTracker
DEVICE_CLASS_LIGHTHOUSE = 4  # TrackedDeviceClass_TrackingReference


class ProtocolError(ValueError):
    """A datagram could not be decoded (malformed, wrong version, bad shape)."""


@dataclass
class TrackerSample:
    """One pose sample of a tracked device, in raw OpenVR conventions."""

    serial: str
    device_class: int  # DEVICE_CLASS_TRACKER or DEVICE_CLASS_LIGHTHOUSE
    t_mono: float  # host time.monotonic() at poll [s]
    t_wall: float  # host time.time() at poll [s]
    pose_valid: bool
    device_connected: bool
    tracking_result: int  # openvr.ETrackingResult
    matrix: list[float]  # HmdMatrix34, 12 floats row-major (rotation | translation)
    velocity: list[float]  # tracking-space linear velocity [m/s], 3 floats
    angular_velocity: list[float]  # tracking-space angular velocity [rad/s], 3 floats


def encode_sample(sample: TrackerSample) -> bytes:
    """Serialize a sample to a versioned JSON datagram."""
    payload = {"v": PROTOCOL_VERSION, **asdict(sample)}
    return json.dumps(payload).encode("utf-8")


def decode_sample(data: bytes) -> TrackerSample:
    """
    Parse a datagram back into a :class:`TrackerSample`.

    :raises ProtocolError: On malformed JSON, a version mismatch, missing
        fields, or wrong array lengths. The receiver logs and drops such
        datagrams instead of crashing.
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"malformed datagram: {e}") from e
    if not isinstance(payload, dict):
        raise ProtocolError("datagram is not a JSON object")
    version = payload.pop("v", None)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version mismatch: got {version}, want {PROTOCOL_VERSION}")
    try:
        sample = TrackerSample(**payload)
    except TypeError as e:
        raise ProtocolError(f"bad fields: {e}") from e
    for name, values, length in (
        ("matrix", sample.matrix, 12),
        ("velocity", sample.velocity, 3),
        ("angular_velocity", sample.angular_velocity, 3),
    ):
        if not isinstance(values, list) or len(values) != length:
            raise ProtocolError(f"{name} must be a list of {length} floats")
    return sample
