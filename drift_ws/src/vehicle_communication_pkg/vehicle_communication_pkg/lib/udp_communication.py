"""UDP client implementation for ESP32 communication."""

import json
import random
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
        # Random per-process session id so the firmware can detect a sender
        # restart and reset its seq window immediately.
        self._session_id: int = random.getrandbits(32)
        self._rx_thread: threading.Thread | None = None
        self._running: bool = False
        self._status_lock = threading.Lock()
        self._last_status: dict | None = None
        self._last_status_time: float = 0.0
        self._consecutive_send_errors: int = 0

    def connect(self) -> None:
        """Open the UDP socket and start the status receiver thread."""
        if self._sock is not None:
            self.close()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.0)
        self._running = True
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._rx_thread.start()

    def is_open(self) -> bool:
        """Return True if the socket is open. Does not imply peer reachability."""
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

    _MAX_CONSECUTIVE_SEND_ERRORS = 10

    def _recreate_socket(self) -> None:
        """Close and reopen the socket to recover from a broken socket state."""
        self.connect()
        self._consecutive_send_errors = 0

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
            "session_id": self._session_id,
            "steering": steering,
            "throttle": throttle,
            "gyro_mode": gyro_mode,
            "gyro_gain": gyro_gain,
            "seq": self._seq,
        }
        data = json.dumps(message).encode("utf-8")
        try:
            self._sock.sendto(data, (self.ip, self.port))
            self._consecutive_send_errors = 0
        except OSError:
            self._consecutive_send_errors += 1
            if self._consecutive_send_errors >= self._MAX_CONSECUTIVE_SEND_ERRORS:
                self._recreate_socket()
            raise

    def get_last_status(self) -> dict | None:
        """Return the most recently received status from the ESP32, or None."""
        with self._status_lock:
            if time.monotonic() - self._last_status_time > self.STATUS_TIMEOUT_S:
                return None
            return self._last_status

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(512)
                status = json.loads(data.decode("utf-8"))
                with self._status_lock:
                    self._last_status = status
                    self._last_status_time = time.monotonic()
            except socket.timeout:
                continue
            except (json.JSONDecodeError, OSError):
                if not self._running:
                    break
                time.sleep(0.05)
                continue
