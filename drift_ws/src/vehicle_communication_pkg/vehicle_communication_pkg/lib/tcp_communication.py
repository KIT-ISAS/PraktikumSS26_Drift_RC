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

        :param ip: IP address of the Raspberry Pi.
        :param port: TCP port to connect to.

        """
        self.ip: str = ip
        self.port: int = port
        self._sock: socket.socket | None = None

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

    def send_value(self, acceleration: float, steering: float) -> None:
        """
        Send acceleration and steering values as JSON over TCP.

        :param acceleration: acceleration command value.
        :param steering: Steering command value.

        :raises RuntimeError: If not connected.

        """
        if self._sock is None:
            raise RuntimeError("TCP connection not initialized. Call connect() first.")

        message = {"speed": acceleration, "steering": steering}
        data = json.dumps(message) + "\n"
        try:
            self._sock.sendall(data.encode("utf-8"))
        except OSError:
            self.close()
            raise
