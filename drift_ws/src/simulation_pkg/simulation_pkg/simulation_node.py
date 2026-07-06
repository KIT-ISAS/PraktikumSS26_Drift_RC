"""
Run a vehicle_simulation model and publish the canonical car state.

The simulation is its own tracking source *and* state owner, so it publishes the
same canonical surface as the real ``car_state_node`` — ``/car/pose`` (world
frame), ``/car/odom`` (pose + body-frame twist) and the ``world -> base_link``
transform — making it a drop-in for the real car. Unlike the real node it does
not finite-difference: it knows the exact body-frame velocities, so the odom
twist is filled straight from the model's ``VehicleState``.
"""

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Frames, Topics
from vehicle_simulation.models import VehicleAction, VehicleState, build_model
from vehicle_simulation.transforms import quaternion_from_yaw


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Yaw-only quaternion (x, y, z, w) via the shared transforms helper."""
    x, y, z, w = quaternion_from_yaw(yaw)
    return Quaternion(x=x, y=y, z=z, w=w)


class SimulationNode(Node):
    """Drives a vehicle model forward in time and publishes the simulated pose."""

    def __init__(self) -> None:
        super().__init__("simulation_node")
        self._load_params()
        self._model = self._build_model()
        self.get_logger().info(
            f"Simulation model: {self._model_type!r} -> {type(self._model).__name__}"
        )
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
        self._odom_pub = self.create_publisher(Odometry, Topics.CAR_ODOM, 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(self._dt, self._step)

    def _action_callback(self, msg: VehicleActionRequest) -> None:
        self._last_action = VehicleAction(steering=msg.steering, throttle=msg.throttle)

    def _step(self) -> None:
        self._state = self._model.step(self._state, self._last_action, self._dt)

        stamp = self.get_clock().now().to_msg()
        orientation = _yaw_to_quaternion(self._state.heading)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._world_frame
        pose_msg.pose.position.x = self._state.x
        pose_msg.pose.position.y = self._state.y
        pose_msg.pose.orientation = orientation
        self._pose_pub.publish(pose_msg)

        # Full canonical state. The sim knows the exact body-frame velocities, so
        # the twist is filled directly (no finite differencing like the real node).
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = self._base_frame
        odom_msg.pose.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = self._state.vx
        odom_msg.twist.twist.linear.y = self._state.vy
        odom_msg.twist.twist.angular.z = self._state.yaw_rate
        self._odom_pub.publish(odom_msg)

        # The sim is its own tracking source: broadcast world -> base_link directly
        # (no tracker_link, so this is the car's single dynamic transform).
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._world_frame
        tf.child_frame_id = self._base_frame
        tf.transform.translation.x = self._state.x
        tf.transform.translation.y = self._state.y
        tf.transform.rotation = orientation
        self._tf_broadcaster.sendTransform(tf)

    def _load_params(self) -> None:
        self.declare_parameter("model_type", "kinematic")
        self.declare_parameter("model_config", "")
        self.declare_parameter("model_weights", "")
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_heading", 0.0)
        # Frame names: a second (sim) instance can prefix base_frame (e.g.
        # "sim/base_link") to coexist with the real car in the shared world frame.
        self.declare_parameter("world_frame", Frames.WORLD)
        self.declare_parameter("base_frame", Frames.BASE_LINK)

        self._world_frame = self.get_parameter("world_frame").value
        self._base_frame = self.get_parameter("base_frame").value
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
