"""
Drive the car with a trained RL agent.

Closes the loop the training environment simulates: subscribes to the canonical
car state (``/car/odom`` — pose + body-frame twist) and the track topics
(``/track/progress``, latched ``/track/centerline``), builds the policy
observation with the **shared** builder from
``vehicle_simulation.environments.observation`` (the sim-to-real seam — the exact
code the agent was trained on), and publishes ``VehicleActionRequest`` at the
control rate. Because both the simulation and the real car publish the same
canonical surface, this node is sim/real agnostic.

Safety: actions are only published while *all* inputs are fresh
(``input_timeout_s``); on stale localization or track progress the node stops
publishing (it never sends zeros, so a joystick or the firmware failsafe can
take over) and warns.
"""

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from shared_interfaces.msg import Track, TrackProgress, VehicleActionRequest
from shared_pkg import Topics
from vehicle_simulation.environments.observation import build_observation, curvature_lookahead
from vehicle_simulation.transforms import yaw_from_quaternion


def observation_from_messages(
    odom: Odometry,
    progress: TrackProgress,
    track: Track,
    prev_action: np.ndarray,
) -> np.ndarray:
    """
    Build the policy observation from the ROS messages (pure, unit-testable).

    Mirrors ``DriftEnv``'s observation exactly: every field maps 1:1 onto what
    the training environment feeds the shared builder.
    """
    car = odom.pose.pose.orientation
    tangent = progress.projected_pose.orientation
    heading = yaw_from_quaternion(car.x, car.y, car.z, car.w)
    track_heading = yaw_from_quaternion(tangent.x, tangent.y, tangent.z, tangent.w)
    curvatures_ahead = curvature_lookahead(
        track.curvature,
        track.cumulative_distance,
        progress.total_length,
        progress.distance,
    )
    return build_observation(
        lateral_error=progress.lateral_error,
        heading_error=heading - track_heading,
        vx=odom.twist.twist.linear.x,
        vy=odom.twist.twist.linear.y,
        yaw_rate=odom.twist.twist.angular.z,
        prev_action=prev_action,
        curvatures_ahead=curvatures_ahead,
    )


class AgentNode(Node):
    """Runs a trained drift agent as the control source at a fixed rate."""

    def __init__(self) -> None:
        super().__init__("agent_node")
        self._load_params()
        self._agent = self._load_agent()

        self._odom: Odometry | None = None
        self._progress: TrackProgress | None = None
        self._track: Track | None = None
        self._odom_stamp = None
        self._progress_stamp = None
        self._prev_action = np.zeros(2, dtype=np.float32)

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, Topics.CAR_ODOM, self._odom_callback, 10)
        self.create_subscription(TrackProgress, Topics.TRACK_PROGRESS, self._progress_callback, 10)
        self.create_subscription(Track, Topics.TRACK_CENTERLINE, self._track_callback, latched_qos)
        self._action_pub = self.create_publisher(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10
        )
        self.create_timer(1.0 / self._rate_hz, self._control_step)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg
        self._odom_stamp = self.get_clock().now()

    def _progress_callback(self, msg: TrackProgress) -> None:
        self._progress = msg
        self._progress_stamp = self.get_clock().now()

    def _track_callback(self, msg: Track) -> None:
        self._track = msg
        self.get_logger().info(
            f"Received track {msg.name!r} ({len(msg.curvature)} centerline points)"
        )

    def _control_step(self) -> None:
        if not self._inputs_ready():
            return
        obs = observation_from_messages(self._odom, self._progress, self._track, self._prev_action)
        action = self._agent.predict(obs)
        self._prev_action = action

        msg = VehicleActionRequest()
        msg.steering = float(action[0])
        msg.throttle = float(action[1])
        msg.gyro_mode = self._gyro_mode
        msg.gyro_gain = self._gyro_gain
        self._action_pub.publish(msg)

    def _inputs_ready(self) -> bool:
        """All inputs present and fresh; warn (throttled) and hold off otherwise."""
        if self._track is None or self._odom is None or self._progress is None:
            self.get_logger().warning(
                "Waiting for inputs (odom / track progress / centerline)",
                throttle_duration_sec=2.0,
            )
            return False
        now = self.get_clock().now()
        for name, stamp in (("odom", self._odom_stamp), ("track progress", self._progress_stamp)):
            age = (now - stamp).nanoseconds * 1e-9
            if age > self._input_timeout_s:
                self.get_logger().warning(
                    f"{name} is stale ({age:.2f} s > {self._input_timeout_s} s) — not publishing",
                    throttle_duration_sec=2.0,
                )
                return False
        return True

    def _load_params(self) -> None:
        self.declare_parameter("run_dir", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("prefer_best", True)
        self.declare_parameter("rate_hz", 50.0)  # matches the training env's dt = 0.02
        self.declare_parameter("input_timeout_s", 0.3)
        self.declare_parameter("gyro_mode", "soft")
        self.declare_parameter("gyro_gain", 0.0)

        self._run_dir = self.get_parameter("run_dir").value
        self._device = self.get_parameter("device").value
        self._prefer_best = bool(self.get_parameter("prefer_best").value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._input_timeout_s = float(self.get_parameter("input_timeout_s").value)
        self._gyro_mode = self.get_parameter("gyro_mode").value
        self._gyro_gain = float(self.get_parameter("gyro_gain").value)

    def _load_agent(self):
        # Lazy import: pulls in stable-baselines3/torch, which only this node needs.
        from vehicle_simulation.agents import DriftAgent

        if not self._run_dir:
            raise ValueError(
                "The 'run_dir' parameter is required (a training run directory or SAC .zip)"
            )
        agent = DriftAgent(self._run_dir, device=self._device, prefer_best=self._prefer_best)
        self.get_logger().info(f"Loaded agent checkpoint: {agent.path}")
        return agent


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
