"""Unit tests for ``UdpCommunicationClient``; all socket I/O is mocked (offline)."""

import json
import socket
import time
from unittest import mock

import pytest

from vehicle_communication_pkg.lib.udp_communication import UdpCommunicationClient

IP = "10.42.0.99"
PORT = 12345

EXPECTED_KEYS = {"session_id", "steering", "throttle", "gyro_mode", "gyro_gain", "seq"}


@pytest.fixture
def connected_client():
    """Yield (client, mock socket) with connect() already called; closes on teardown."""
    patcher = mock.patch("vehicle_communication_pkg.lib.udp_communication.socket.socket")
    socket_cls = patcher.start()
    sock = socket_cls.return_value

    # Keep the receiver thread cycling without delivering data (and without busy-spinning).
    def _no_data(*args, **kwargs):
        time.sleep(0.01)
        raise socket.timeout

    sock.recvfrom.side_effect = _no_data
    client = UdpCommunicationClient(IP, PORT)
    client.connect()
    yield client, sock
    client.close()
    patcher.stop()


def _sent_payloads(sock) -> list[dict]:
    return [json.loads(call.args[0].decode("utf-8")) for call in sock.sendto.call_args_list]


def _simulate_receive(client: UdpCommunicationClient, status: dict) -> None:
    """Run the receive loop synchronously over a single fake datagram."""
    sock = mock.Mock()
    sock.recvfrom.side_effect = [
        (json.dumps(status).encode("utf-8"), (IP, PORT)),
        OSError(),  # second iteration: pretend the socket was closed externally
    ]
    sock.fileno.return_value = -1
    client._running = True
    client._receive_loop(sock)


class TestSendValue:
    """JSON framing and sequence numbering of send_value."""

    def test_before_connect_raises(self):
        client = UdpCommunicationClient(IP, PORT)
        with pytest.raises(RuntimeError):
            client.send_value(0.0, 0.0, "soft", 0.5)

    def test_sends_valid_json_with_expected_keys(self, connected_client):
        client, sock = connected_client
        client.send_value(0.5, -0.25, "soft", 0.7)

        sock.sendto.assert_called_once()
        data, address = sock.sendto.call_args.args
        assert address == (IP, PORT)
        payload = json.loads(data.decode("utf-8"))
        assert set(payload.keys()) == EXPECTED_KEYS
        assert payload["steering"] == pytest.approx(0.5)
        assert payload["throttle"] == pytest.approx(-0.25)
        assert payload["gyro_mode"] == "soft"
        assert payload["gyro_gain"] == pytest.approx(0.7)

    def test_sequence_numbers_increment_by_one(self, connected_client):
        client, sock = connected_client
        client.send_value(0.0, 0.0, "soft", 0.0)
        client.send_value(0.0, 0.0, "soft", 0.0)

        first, second = _sent_payloads(sock)
        assert second["seq"] - first["seq"] == 1
        assert second["session_id"] == first["session_id"]


class TestGetLastStatus:
    """Status caching and timeout behavior."""

    def test_none_before_any_receive(self):
        client = UdpCommunicationClient(IP, PORT)
        assert client.get_last_status() is None

    def test_returns_payload_after_receive(self):
        client = UdpCommunicationClient(IP, PORT)
        status = {"state": "armed", "battery": 7.4}
        _simulate_receive(client, status)
        assert client.get_last_status() == status

    def test_none_after_timeout(self):
        client = UdpCommunicationClient(IP, PORT)
        _simulate_receive(client, {"state": "armed"})
        assert client.get_last_status() is not None

        stale = time.monotonic() + UdpCommunicationClient.STATUS_TIMEOUT_S + 0.1
        with mock.patch(
            "vehicle_communication_pkg.lib.udp_communication.time.monotonic",
            return_value=stale,
        ):
            assert client.get_last_status() is None


class TestLifecycle:
    """connect()/close() state handling."""

    def test_is_open_reflects_connection(self, connected_client):
        client, _ = connected_client
        assert client.is_open()
        client.close()
        assert not client.is_open()

    def test_close_without_connect_does_not_raise(self):
        client = UdpCommunicationClient(IP, PORT)
        client.close()
        client.close()

    def test_close_is_idempotent_after_connect(self, connected_client):
        client, _ = connected_client
        client.close()
        client.close()
