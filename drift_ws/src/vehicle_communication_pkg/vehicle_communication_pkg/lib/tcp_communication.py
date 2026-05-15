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

        Args:
            ip: IP address of the Raspberry Pi.
            port: TCP port to connect to.
        """
        self.ip = ip
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        """Initialize and open a TCP connection."""
        if self._sock is not None:
            return  # Already connected

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

        Args:
            acceleration: acceleration command value.
            steering: Steering command value.

        Raises:
            RuntimeError: If not connected.
        """
        if self._sock is None:
            raise RuntimeError("TCP connection not initialized. Call connect() first.")

        message = {"acceleration": acceleration, "steering": steering}
        data = json.dumps(message) + "\n"
        self._sock.sendall(data.encode("utf-8"))
