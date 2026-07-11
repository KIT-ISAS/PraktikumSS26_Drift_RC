"""
Shared base node for the policy-driven control nodes (`agent_node`, `data_collection_node`).

Owns everything the two nodes have in common: the shared parameters, the canonical
input subscriptions (odom / track progress / centerline) with freshness stamps, the
start/stop/override signal handling, the action publisher, and the control-rate
timer. The safety-critical arm/stop/brake decisions are delegated to the pure
:class:`~agent_pkg.lib.control_gate.ControlGate` state machine; this class is the
thin ROS shell around it (clocks, messages, logging).

Subclasses implement ``_active_step()`` (called at the control rate while armed and
all inputs are fresh) and may override the ``_on_*`` lifecycle hooks to run extra
work on start/stop/override/dropout transitions (e.g. episode-buffer handling).
"""

from nav_msgs.msg import Odometry
import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty

from agent_pkg.lib.control_gate import ControlGate, InputFreshness
from shared_interfaces.msg import Track, TrackProgress, VehicleActionRequest
from shared_pkg import Topics


class AgentControlNode(Node):
    """Base for nodes that drive the car with a trained policy, gated by start/stop."""

    def __init__(self, node_name: str, *, activity: str, stale_verb: str) -> None:
        """
        Set up parameters, gate, subscriptions, publisher, and the control timer.

        ``activity``/``stale_verb`` name the node's job in log messages
        (e.g. "agent"/"publishing" or "recording"/"recording").
        """
        super().__init__(node_name)
        self._activity = activity
        self._stale_verb = stale_verb
        self._load_params()
        self._gate = ControlGate(
            rate_hz=self._rate_hz,
            input_timeout_s=self._input_timeout_s,
            disarm_timeout_s=self._disarm_timeout_s,
            stop_brake_throttle=self._stop_brake_throttle,
            stop_speed_threshold=self._stop_speed_threshold,
            stop_timeout_s=self._stop_timeout_s,
        )

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
        self.create_subscription(Empty, Topics.AGENT_START, self._start_callback, 10)
        self.create_subscription(Empty, Topics.AGENT_STOP, self._stop_callback, 10)
        self.create_subscription(Empty, Topics.AGENT_OVERRIDE, self._override_callback, 10)
        self._action_pub = self.create_publisher(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10
        )
        self.create_timer(1.0 / self._rate_hz, self._control_step)

    # --- hooks for subclasses --------------------------------------------------

    def _active_step(self) -> None:
        """One armed control tick with all inputs fresh; publish the action here."""
        raise NotImplementedError

    def _on_start(self) -> None:
        """Extra work when the node arms (base already resets ``_prev_action``)."""

    def _on_stop(self) -> None:
        """Extra work on a stop signal, before entering the braking phase."""

    def _on_override(self) -> None:
        """Extra work on an override, after control has been released."""

    def _on_dropout_disarm(self) -> None:
        """Extra work when a long input dropout disarms, before entering braking."""

    def _on_inputs_stale(self) -> None:
        """Extra work when an armed tick is held off because an input is stale."""

    # --- ROS callbacks -----------------------------------------------------------

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
        if not self._gate.start():
            return
        # Start from a clean slate so a restart never feeds a stale prev_action.
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._on_start()
        self.get_logger().info(f"Received start signal — {self._activity} armed")

    def _stop_callback(self, _msg: Empty) -> None:
        if not self._gate.running and not self._gate.stopping:
            return
        self._on_stop()
        if self._gate.stop(self._now_s()):
            self.get_logger().info("Received stop signal — braking to a standstill")

    def _override_callback(self, _msg: Empty) -> None:
        # Another controller has taken over the vehicle action topic: release
        # immediately, including cancelling an in-progress braking phase.
        if not self._gate.override():
            return
        self._on_override()
        self.get_logger().warning("Override received — releasing control immediately")

    # --- control loop ------------------------------------------------------------

    def _control_step(self) -> None:
        if self._gate.stopping:
            self._stop_step()
            return
        if not self._gate.running:
            self.get_logger().info(
                f"Waiting for start signal on {Topics.AGENT_START}",
                throttle_duration_sec=5.0,
            )
            return
        if not self._inputs_ready():
            self._on_inputs_stale()
            return
        self._active_step()

    def _stop_step(self) -> None:
        """One braking-phase tick: publish the gate's brake/neutral command."""
        command = self._gate.stop_step(self._now_s(), self._current_speed())
        self._publish_action(command.steering, command.throttle)
        self._prev_action = np.zeros(2, dtype=np.float32)
        if command.released:
            self.get_logger().info(f"Car stopped — {self._activity} disarmed, releasing control")

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
            freshness = self._gate.classify_input_age(age)
            if freshness is InputFreshness.FRESH:
                continue
            if freshness is InputFreshness.DROPOUT:
                self._disarm_after_dropout(name, age)
            else:
                self.get_logger().warning(
                    f"{name} is stale ({age:.2f} s > {self._input_timeout_s} s) "
                    f"— not {self._stale_verb}",
                    throttle_duration_sec=2.0,
                )
            return False
        return True

    def _disarm_after_dropout(self, name: str, age: float) -> None:
        """Disarm on a long input dropout — a silent resume would be surprising."""
        self._on_dropout_disarm()
        self._gate.disarm(self._now_s())
        self.get_logger().warning(
            f"{name} lost for {age:.2f} s (> {self._disarm_timeout_s} s) — disarming, "
            f"a fresh start signal on {Topics.AGENT_START} is required"
        )

    # --- shared helpers ------------------------------------------------------------

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

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

    def _load_params(self) -> None:
        """Declare and read the shared parameters; subclasses extend via ``super()``."""
        self.declare_parameter("run_dir", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("prefer_best", True)
        self.declare_parameter("rate_hz", 50.0)  # matches the training env's dt = 0.02
        self.declare_parameter("input_timeout_s", 0.3)
        self.declare_parameter("disarm_timeout_s", 2.0)  # dropout longer than this disarms
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
        self._disarm_timeout_s = float(self.get_parameter("disarm_timeout_s").value)
        self._gyro_mode = self.get_parameter("gyro_mode").value
        self._gyro_gain = float(self.get_parameter("gyro_gain").value)
        self._stop_brake_throttle = float(self.get_parameter("stop_brake_throttle").value)
        self._stop_speed_threshold = float(self.get_parameter("stop_speed_threshold").value)
        self._stop_timeout_s = float(self.get_parameter("stop_timeout_s").value)

    def _load_agent(self):
        # Lazy import: pulls in stable-baselines3/torch, which only these nodes need.
        from vehicle_simulation.agents import DriftAgent

        if not self._run_dir:
            raise ValueError(
                "The 'run_dir' parameter is required (a training run directory or SAC .zip)"
            )
        agent = DriftAgent(self._run_dir, device=self._device, prefer_best=self._prefer_best)
        self.get_logger().info(f"Loaded agent checkpoint: {agent.path}")
        return agent
