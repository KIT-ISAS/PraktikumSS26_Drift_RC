"""ROS 2 node for vehicle communication."""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from .lib.communication_interface import (
    CommunicationInterface,
)
from shared_interfaces.msg import VehicleActionRequest


class VehicleCommunicationNode(Node):
    """
    ROS 2 node that subscribes to VehicleActionRequest messages and forwards them to a vehicle.

    Uses a pluggable communication interface for flexibility (TCP, CAN, etc.).
    """

    def __init__(self, communication_interface: CommunicationInterface):
        """
        Initialize the vehicle communication node.

        Args:
            communication_interface: An instance implementing CommunicationInterface.
                                   The node takes ownership and will close it on shutdown.
        """
        super().__init__("vehicle_communication_node")
        self.interface = communication_interface

        # Subscribe to vehicle action requests
        self.subscription = self.create_subscription(
            VehicleActionRequest,
            "vehicle_action_request",
            self._vehicle_action_callback,
            10,
        )

        # Provide services for manual connection control
        self.connect_service = self.create_service(
            Trigger, "connect", self._handle_connect
        )
        self.disconnect_service = self.create_service(
            Trigger, "disconnect", self._handle_disconnect
        )

        # Auto-connect on startup
        try:
            self.interface.connect()
            self.get_logger().info("Vehicle communication interface connected.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect on startup: {e}")

    def _vehicle_action_callback(self, msg: VehicleActionRequest) -> None:
        """
        Callback for received VehicleActionRequest messages.

        Forwards the message to the communication interface.

        Args:
            msg: The received VehicleActionRequest message.
        """
        if not self.interface.is_connected():
            self.get_logger().warn(
                "Received vehicle action but interface is not connected. Skipping."
            )
            return

        try:
            self.interface.send_value(msg.acceleration, msg.steering)
        except Exception as e:
            self.get_logger().error(
                f"Failed to send vehicle action (acceleration={msg.acceleration}, "
                f"steering={msg.steering}): {e}"
            )

    def _handle_connect(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """
        Handle request to connect the communication interface.

        Args:
            request: Empty request.
            response: Response with success flag and message.

        Returns:
            Updated response.
        """
        try:
            if self.interface.is_connected():
                response.success = True
                response.message = "Already connected."
            else:
                self.interface.connect()
                response.success = True
                response.message = "Connected successfully."
                self.get_logger().info("Vehicle communication interface connected.")
        except Exception as e:
            response.success = False
            response.message = f"Connection failed: {str(e)}"
            self.get_logger().error(f"Failed to connect: {e}")

        return response

    def _handle_disconnect(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """
        Handle request to disconnect the communication interface.

        Args:
            request: Empty request.
            response: Response with success flag and message.

        Returns:
            Updated response.
        """
        try:
            self.interface.close()
            response.success = True
            response.message = "Disconnected successfully."
            self.get_logger().info("Vehicle communication interface disconnected.")
        except Exception as e:
            response.success = False
            response.message = f"Disconnection failed: {str(e)}"
            self.get_logger().error(f"Failed to disconnect: {e}")

        return response

    def destroy_node(self) -> None:
        """Clean up resources when the node is destroyed."""
        try:
            self.interface.close()
        except Exception as e:
            self.get_logger().error(f"Error closing interface on shutdown: {e}")

        super().destroy_node()


def main(args=None):
    """Entry point for the vehicle communication node."""
    rclpy.init(args=args)

    # Create communication interface instance
    from .lib.tcp_communication import (
        TcpCommunicationClient,
    )

    communication_interface = TcpCommunicationClient(ip="192.168.137.5", port=12345)

    # Create and spin node
    node = VehicleCommunicationNode(communication_interface)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
