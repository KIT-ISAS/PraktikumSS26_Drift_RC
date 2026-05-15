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

        Raises:
            RuntimeError: If connection fails or is already established.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the communication interface is currently connected.

        Returns:
            True if connected, False otherwise.
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
    def send_value(self, speed: float, steering: float) -> None:
        """
        Send a control command to the vehicle.

        Args:
            speed: Speed command value.
            steering: Steering command value.

        Raises:
            RuntimeError: If not connected or send fails.
        """
        pass
