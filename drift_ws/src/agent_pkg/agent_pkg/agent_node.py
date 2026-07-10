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

Gating: the node is armed by a start signal (``/agent/start``) and disarmed by a
stop signal (``/agent/stop``). Until it has been started it publishes nothing, so
a joystick or the firmware failsafe stays in control. This lets an operator hold
the agent off the throttle until the car is positioned and cut it instantly.

Stopping: a stop signal does not simply go silent — that would leave the ESP32
holding the last (possibly throttled) command until its failsafe times out. The
node instead brakes the car down to a standstill (negative throttle, steering
straightened) using the ``/car/odom`` speed, then sends a neutral command before
releasing control so a joystick or the firmware failsafe can take over. If the
odometry is stale/missing (speed unknown) it skips straight to neutral.

Override (``/agent/override``): sent when another controller (e.g. the joystick
in override mode) has already seized the vehicle action topic. The agent releases
control *immediately* — the braking phase would fight the new controller's
commands, and the sender guarantees the car is being commanded.

Safety: while armed, actions are only published while *all* inputs are fresh
(``input_timeout_s``); on stale localization or track progress the node holds off
publishing (letting a joystick or the firmware failsafe take over) and warns.
"""

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty

from agent_pkg.observation_bridge import observation_from_messages
from shared_interfaces.msg import Track, TrackProgress, VehicleActionRequest
from shared_pkg import Topics

# Re-exported for backward compatibility (tests and callers import it from here).
__all__ = ["AgentNode", "observation_from_messages", "main"]


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
        # Disarmed until a start signal arrives; a stop signal disarms again.
        self._running = False
        # Braking-to-standstill phase entered on a stop signal (see _stop_step).
        self._stopping = False
        self._stop_started = None
        self._neutral_ticks = 0
        # Publish neutral for a short settle window (a few packets, in case of loss)
        # before releasing control, so the car is confirmed at zero throttle.
        self._stop_neutral_ticks = max(1, round(self._rate_hz * 0.3))

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, Topics.CAR_ODOM, self._odom_callback, 10)
        self.create_subscription(TrackProgress, Topics.TRACK_PROGRESS, self._progress_callback, 10)
        self.create_subscription(Track, Topics.TRACK_CENTERLINE, self._track_callback, latched_qos)
        self.create_subscription(Empty, Topics.AGENT_START, self._start_callback, 10)
        self.create_subscription(Empty, Topics.AGENT_STOP, self._stop_callback, 10)
        self.create_subscription(Empty, Topics.AGENT_OVERRIDE, self._override_callback, 10)
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

    def _start_callback(self, _msg: Empty) -> None:
        if self._running:
            return
        self._running = True
        # Cancel any in-progress braking phase so a start re-arms immediately.
        self._stopping = False
        # Start from a clean slate so a restart never feeds a stale prev_action.
        self._prev_action = np.zeros(2, dtype=np.float32)
        self.get_logger().info("Received start signal — agent armed")

    def _stop_callback(self, _msg: Empty) -> None:
        if not self._running and not self._stopping:
            return
        # Disarm and enter the braking-to-standstill phase (idempotent if already
        # stopping — don't restart the brake timer on a repeated stop press).
        self._running = False
        if not self._stopping:
            self._stopping = True
            self._stop_started = self.get_clock().now()
            self._neutral_ticks = 0
            self.get_logger().info("Received stop signal — braking to a standstill")

    def _override_callback(self, _msg: Empty) -> None:
        if not self._running and not self._stopping:
            return
        # Another controller has taken over the vehicle action topic: release
        # immediately, including cancelling an in-progress braking phase.
        self._running = False
        self._stopping = False
        self.get_logger().warning("Override received — releasing control immediately")

    def _control_step(self) -> None:
        if self._stopping:
            self._stop_step()
            return
        if not self._running:
            self.get_logger().info(
                f"Waiting for start signal on {Topics.AGENT_START}",
                throttle_duration_sec=5.0,
            )
            return
        if not self._inputs_ready():
            return
        obs = observation_from_messages(self._odom, self._progress, self._track, self._prev_action)
        action = self._agent.predict(obs)
        self._prev_action = action
        self._publish_action(float(action[0]), float(action[1]))

    def _stop_step(self) -> None:
        """Brake to a standstill, then send neutral and release control."""
        elapsed = (self.get_clock().now() - self._stop_started).nanoseconds * 1e-9
        speed = self._current_speed()
        # Brake (steering straight) while we can see the car still moving and the
        # safety timeout has not elapsed. Unknown speed (stale/missing odom) falls
        # through to neutral, which is the safe command regardless.
        moving = speed is not None and speed > self._stop_speed_threshold
        if moving and elapsed < self._stop_timeout_s:
            self._publish_action(0.0, self._stop_brake_throttle)
            self._prev_action = np.zeros(2, dtype=np.float32)
            return
        # Standing (or can't tell / timed out): hold neutral for a short settle
        # window so the zero-throttle command lands, then release control.
        self._publish_action(0.0, 0.0)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._neutral_ticks += 1
        if self._neutral_ticks >= self._stop_neutral_ticks:
            self._stopping = False
            self.get_logger().info("Car stopped — agent disarmed, releasing control")

    def _current_speed(self) -> float | None:
        """Body-frame speed from fresh odom, or None if unavailable/stale."""
        if self._odom is None or self._odom_stamp is None:
            return None
        age = (self.get_clock().now() - self._odom_stamp).nanoseconds * 1e-9
        if age > self._input_timeout_s:
            return None
        vx = self._odom.twist.twist.linear.x
        vy = self._odom.twist.twist.linear.y
        return float(np.hypot(vx, vy))

    def _publish_action(self, steering: float, throttle: float) -> None:
        msg = VehicleActionRequest()
        msg.steering = steering
        msg.throttle = throttle
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
        # Braking-to-standstill behaviour on a stop signal.
        self.declare_parameter("stop_brake_throttle", -0.4)
        self.declare_parameter("stop_speed_threshold", 0.2)  # m/s below this = standing
        self.declare_parameter("stop_timeout_s", 2.0)  # cap on active braking

        self._run_dir = self.get_parameter("run_dir").value
        self._device = self.get_parameter("device").value
        self._prefer_best = bool(self.get_parameter("prefer_best").value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._input_timeout_s = float(self.get_parameter("input_timeout_s").value)
        self._gyro_mode = self.get_parameter("gyro_mode").value
        self._gyro_gain = float(self.get_parameter("gyro_gain").value)
        self._stop_brake_throttle = float(self.get_parameter("stop_brake_throttle").value)
        self._stop_speed_threshold = float(self.get_parameter("stop_speed_threshold").value)
        self._stop_timeout_s = float(self.get_parameter("stop_timeout_s").value)

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
    node = None
    try:
        node = AgentNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
