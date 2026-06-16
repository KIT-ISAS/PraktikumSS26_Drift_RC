"""ROS2 node that runs a vehicle_simulation model and publishes /car/pose."""

import math

from geometry_msgs.msg import PoseStamped, Quaternion
import rclpy
from rclpy.node import Node

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Topics
from vehicle_simulation.models import VehicleAction, VehicleState, build_model


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


class SimulationNode(Node):
    """Drives a vehicle model forward in time and publishes the simulated pose."""

    def __init__(self) -> None:
        super().__init__("simulation_node")
        self._load_params()
        self._model = self._build_model()
        self._state = VehicleState(
            x=self._initial_x,
            y=self._initial_y,
            heading=self._initial_heading,
        )
        self._last_action = VehicleAction()

        self.create_subscription(
            VehicleActionRequest,
            Topics.VEHICLE_ACTION_REQUEST,
            self._action_callback,
            10,
        )
        self._pose_pub = self.create_publisher(PoseStamped, Topics.CAR_POSE, 10)
        self.create_timer(self._dt, self._step)

    def _action_callback(self, msg: VehicleActionRequest) -> None:
        self._last_action = VehicleAction(steering=msg.steering, throttle=msg.throttle)

    def _step(self) -> None:
        self._state = self._model.step(self._state, self._last_action, self._dt)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = self._state.x
        pose_msg.pose.position.y = self._state.y
        pose_msg.pose.orientation = _yaw_to_quaternion(self._state.heading)
        self._pose_pub.publish(pose_msg)

    def _load_params(self) -> None:
        self.declare_parameter("model_type", "kinematic")
        self.declare_parameter("model_config", "")
        self.declare_parameter("model_weights", "")
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_heading", 0.0)

        self._model_type = self.get_parameter("model_type").value
        self._model_config = self.get_parameter("model_config").value
        self._model_weights = self.get_parameter("model_weights").value
        self._dt = float(self.get_parameter("dt").value)
        self._initial_x = float(self.get_parameter("initial_x").value)
        self._initial_y = float(self.get_parameter("initial_y").value)
        self._initial_heading = float(self.get_parameter("initial_heading").value)

    def _build_model(self):
        return build_model(
            self._model_type,
            config=self._model_config or None,
            weights=self._model_weights or None,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
