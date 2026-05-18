"""Maps the Joystick topic with the correct joystick key mapping for the drift car."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg.topics import Topics

from .lib import JoystickMapper


class JoystickMapperNode(Node):
    """ROS2 node that maps joystick input to vehicle action requests."""

    def __init__(self, mapper: JoystickMapper):
        """Initialize the joystick mapper node."""
        super().__init__("joystick_mapper")
        self._load_params()
        self._mapper = mapper

        # Create publisher for vehicle action requests
        self.vehicle_action_publisher = self.create_publisher(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10
        )

        # Subscribe to joy topic
        self.joy_subscriber = self.create_subscription(Joy, Topics.JOY_NODE, self._joy_callback, 10)

    def _joy_callback(self, msg: Joy) -> None:
        """
        Process joystick input and publish vehicle action requests.

        Maps the parameterized joystick axes to steering and acceleration values.
        """
        action_msg = VehicleActionRequest()
        action_msg.steering = self._mapper.extract_steering(msg)
        action_msg.acceleration = self._mapper.extract_acceleration(msg)
        self.vehicle_action_publisher.publish(action_msg)

    def _load_params(self) -> None:
        """Load parameters from the ROS2 parameter server."""
        # Load axis mapping parameters
        self.declare_parameter("axis_mapping.throttle", 0)
        self.declare_parameter("axis_mapping.brake", 0)
        self.declare_parameter("axis_mapping.steering", 0)
        axis_mapping_params = self.get_parameters_by_prefix("axis_mapping")
        self.axis_mapping = {
            param_key: param_value.value for param_key, param_value in axis_mapping_params.items()
        }


def main(args=None):
    """Start joystick reader node."""
    rclpy.init(args=args)

    from .lib import XBoxOneMapper

    joystick_mapper = JoystickMapperNode(XBoxOneMapper())

    try:
        rclpy.spin(joystick_mapper)
    except KeyboardInterrupt:
        pass
    finally:
        joystick_mapper.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
