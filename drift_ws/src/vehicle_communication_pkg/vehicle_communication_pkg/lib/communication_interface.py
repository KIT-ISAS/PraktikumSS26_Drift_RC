"""Abstract base class for vehicle communication implementations."""

from abc import ABC, abstractmethod


class CommunicationInterface(ABC):
    """
    Abstract interface for communication with the vehicle.

    Defines the contract for any communication implementation (TCP, CAN, etc.)
    that sends commands to the vehicle.
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish a connection to the vehicle.

        :raises RuntimeError: If connection fails or is already established.

        """
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """
        Check if the underlying transport (e.g. socket) is open.

        Note: this does not imply the peer is reachable. For connectionless
        transports like UDP, callers should consult status replies (or an
        equivalent liveness signal) to know if the peer is responding.

        :returns: True if the transport is open, False otherwise.

        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the communication connection and clean up resources.

        Should be safe to call multiple times.
        """
        pass

    @abstractmethod
    def send_value(
        self,
        steering: float,
        throttle: float,
        gyro_mode: str,
        gyro_gain: float,
    ) -> None:
        """
        Send a control command to the vehicle.

        :param steering: Steering value in [-1, 1], ISO 8855: positive = left, negative = right.
        :param throttle: Throttle value in [-1, 1]. Negative = brake/reverse, positive = forward.
        :param gyro_mode: Gyro correction mode, either "soft" or "hard".
        :param gyro_gain: Gyro correction strength in [0, 1].

        :raises RuntimeError: If not connected or send fails.

        """
        pass
