"""Unit tests for the bridge wire format (stdlib-only, no ROS required)."""

import dataclasses
import json

import pytest

from steamvr_pkg.lib.protocol import (
    DEVICE_CLASS_TRACKER,
    PROTOCOL_VERSION,
    ProtocolError,
    TrackerSample,
    decode_sample,
    encode_sample,
)


def _sample() -> TrackerSample:
    return TrackerSample(
        serial="LHR-12345678",
        device_class=DEVICE_CLASS_TRACKER,
        t_mono=12.34,
        t_wall=1750000000.5,
        pose_valid=True,
        device_connected=True,
        tracking_result=200,
        matrix=[1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 1.0, -0.25],
        velocity=[0.1, 0.2, 0.3],
        angular_velocity=[-0.1, 0.0, 2.5],
    )


def test_roundtrip_preserves_all_fields():
    sample = _sample()
    assert decode_sample(encode_sample(sample)) == sample


def test_version_mismatch_is_rejected():
    payload = json.loads(encode_sample(_sample()).decode())
    payload["v"] = PROTOCOL_VERSION + 1
    with pytest.raises(ProtocolError, match="version"):
        decode_sample(json.dumps(payload).encode())


def test_garbage_bytes_are_rejected():
    with pytest.raises(ProtocolError):
        decode_sample(b"\xff\xfe not json at all")


def test_non_object_json_is_rejected():
    with pytest.raises(ProtocolError):
        decode_sample(b"[1, 2, 3]")


def test_missing_field_is_rejected():
    payload = json.loads(encode_sample(_sample()).decode())
    del payload["serial"]
    with pytest.raises(ProtocolError, match="fields"):
        decode_sample(json.dumps(payload).encode())


def test_wrong_matrix_length_is_rejected():
    sample = dataclasses.replace(_sample(), matrix=[0.0] * 9)
    with pytest.raises(ProtocolError, match="matrix"):
        decode_sample(encode_sample(sample))
