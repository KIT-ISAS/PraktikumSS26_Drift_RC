"""TCP client implementation for Raspberry Pi communication."""

import json
import socket

from .communication_interface import CommunicationInterface


class TcpCommunicationClient(CommunicationInterface):
    """
    TCP client for communicating with a Raspberry Pi vehicle controller.

    Sends acceleration and steering commands as JSON over a TCP socket.
    """

    def __init__(self, ip: str, port: int):
        """
        Initialize TCP client with connection parameters.

        :param ip: IP address of the ESP32.
        :param port: TCP port to connect to.

        """
        self.ip: str = ip
        self.port: int = port
        self._sock: socket.socket | None = None
        self._seq: int = 0

    def connect(self) -> None:
        """Initialize and open a TCP connection."""
        if self._sock is not None:
            return  # Already connected

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5)
        self._sock.connect((self.ip, self.port))

    def is_connected(self) -> bool:
        """Check if the TCP connection is open."""
        return self._sock is not None

    def close(self) -> None:
        """Close the TCP connection."""
        if self._sock:
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
        Send a control command as newline-delimited JSON over TCP.

        :param steering: Steering value in [-1, 1].
        :param throttle: Throttle value in [-1, 1].
        :param gyro_mode: "soft" or "hard".
        :param gyro_gain: Gyro strength in [0, 1].

        :raises RuntimeError: If not connected.

        """
        if self._sock is None:
            raise RuntimeError("TCP connection not initialized. Call connect() first.")

        self._seq += 1
        message = {
            "steering": steering,
            "throttle": throttle,
            "gyro_mode": gyro_mode,
            "gyro_gain": gyro_gain,
            "seq": self._seq,
        }
        data = json.dumps(message) + "\n"
        try:
            self._sock.sendall(data.encode("utf-8"))
        except OSError:
            self.close()
            raise
