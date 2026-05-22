"""UDP client implementation for ESP32 communication."""

import json
import socket
import threading
import time

from .communication_interface import CommunicationInterface


class UdpCommunicationClient(CommunicationInterface):
    """
    UDP client for communicating with the ESP32 vehicle controller.

    Sends control commands as JSON datagrams and receives status replies
    from the ESP32 in a background thread.
    """

    STATUS_TIMEOUT_S = 0.5

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self._sock: socket.socket | None = None
        self._seq: int = 0
        self._rx_thread: threading.Thread | None = None
        self._running: bool = False
        self._last_status: dict | None = None
        self._last_status_time: float = 0.0

    def connect(self) -> None:
        """Open the UDP socket and start the status receiver thread."""
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.0)
        self._running = True
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._rx_thread.start()

    def is_connected(self) -> bool:
        """Return True if the socket is open."""
        return self._sock is not None

    def close(self) -> None:
        """Stop the receiver thread and close the socket."""
        self._running = False
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=2.0)
            self._rx_thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send_value(
        self,
        steering: float,
        throttle: float,
        gyro_mode: str,
        gyro_gain: float,
    ) -> None:
        """
        Send a control command as a JSON UDP datagram.

        :param steering: Steering value in [-1, 1].
        :param throttle: Throttle value in [-1, 1].
        :param gyro_mode: "soft" or "hard".
        :param gyro_gain: Gyro strength in [0, 1].

        :raises RuntimeError: If not connected.

        """
        if self._sock is None:
            raise RuntimeError("UDP socket not open. Call connect() first.")
        self._seq += 1
        message = {
            "steering": steering,
            "throttle": throttle,
            "gyro_mode": gyro_mode,
            "gyro_gain": gyro_gain,
            "seq": self._seq,
        }
        data = json.dumps(message).encode("utf-8")
        self._sock.sendto(data, (self.ip, self.port))

    def get_last_status(self) -> dict | None:
        """Return the most recently received status from the ESP32, or None."""
        if time.monotonic() - self._last_status_time > self.STATUS_TIMEOUT_S:
            return None
        return self._last_status

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(512)
                status = json.loads(data.decode("utf-8"))
                self._last_status = status
                self._last_status_time = time.monotonic()
            except socket.timeout:
                continue
            except (json.JSONDecodeError, OSError):
                if not self._running:
                    break
