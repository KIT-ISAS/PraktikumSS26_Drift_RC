"""ROS 2 node for vehicle communication."""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Topics

from .lib import CommunicationInterface, UdpCommunicationClient


class VehicleCommunicationNode(Node):
    """
    ROS 2 node that subscribes to VehicleActionRequest messages and forwards them to a vehicle.

    Uses a pluggable communication interface for flexibility (UDP, TCP, CAN, etc.).
    """

    def __init__(self):
        """Initialize the vehicle communication node."""
        super().__init__("vehicle_communication_node")
        self._load_params()

        self.interface: CommunicationInterface = UdpCommunicationClient(self.ip, self.port)

        self.subscription = self.create_subscription(
            VehicleActionRequest,
            Topics.VEHICLE_ACTION_REQUEST,
            self._vehicle_action_callback,
            10,
        )

        self.connect_service = self.create_service(Trigger, "connect", self._handle_connect)
        self.disconnect_service = self.create_service(
            Trigger, "disconnect", self._handle_disconnect
        )

        self._last_logged_status: dict | None = None

        # Log ESP32 status at 1 Hz
        self.create_timer(1.0, self._log_status)

        try:
            self.interface.connect()
            self.get_logger().info(f"UDP socket open, sending to {self.ip}:{self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open UDP socket: {e}")

    def _vehicle_action_callback(self, msg: VehicleActionRequest) -> None:
        """Forward a VehicleActionRequest to the vehicle."""
        if not self.interface.is_connected():
            try:
                self.interface.connect()
                self.get_logger().info("Opened UDP socket.")
            except Exception:
                self.get_logger().warn(
                    "Cannot open UDP socket, skipping command.",
                    throttle_duration_sec=3.0,
                )
                return
        try:
            self.interface.send_value(msg.steering, msg.throttle, msg.gyro_mode, msg.gyro_gain)
        except Exception as e:
            self.get_logger().error(
                f"Failed to send vehicle action (steering={msg.steering}, "
                f"throttle={msg.throttle}): {e}"
            )

    def _log_status(self) -> None:
        """Log the latest ESP32 status if it changed or is newly absent."""
        status = self.interface.get_last_status()
        if status != self._last_logged_status:
            if status is None:
                self.get_logger().warn("ESP32 status timeout — no reply received.")
            else:
                self.get_logger().info(
                    f"ESP32: state={status.get('state', '?')} "
                    f"armed={status.get('armed', '?')} "
                    f"last_seq={status.get('last_seq', '?')}"
                )
            self._last_logged_status = status

    def _handle_connect(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle request to open the UDP socket."""
        try:
            if self.interface.is_connected():
                response.success = True
                response.message = "Already connected."
            else:
                self.interface.connect()
                response.success = True
                response.message = "UDP socket opened."
                self.get_logger().info("UDP socket opened.")
        except Exception as e:
            response.success = False
            response.message = f"Failed: {str(e)}"
            self.get_logger().error(f"Failed to open UDP socket: {e}")
        return response

    def _handle_disconnect(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle request to close the UDP socket."""
        try:
            self.interface.close()
            response.success = True
            response.message = "UDP socket closed."
            self.get_logger().info("UDP socket closed.")
        except Exception as e:
            response.success = False
            response.message = f"Failed: {str(e)}"
            self.get_logger().error(f"Failed to close UDP socket: {e}")
        return response

    def _load_params(self) -> None:
        """Load parameters from the ROS parameter server."""
        self.declare_parameter("ip_address", "10.42.0.99")
        self.declare_parameter("port", 12345)
        self.ip = self.get_parameter("ip_address").value
        self.port = self.get_parameter("port").value

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
    node = VehicleCommunicationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
