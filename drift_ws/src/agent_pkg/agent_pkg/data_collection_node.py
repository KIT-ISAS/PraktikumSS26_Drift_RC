"""
Record real-car transitions for off-board SAC fine-tuning.

Drives the car with a sim-trained policy (like :mod:`.agent_node`) but samples the
policy stochastically for exploration and, at every control step, reconstructs the
same ``(obs, action, reward, next_obs, done)`` transition the training environment
would have produced — using the shared bridges in :mod:`.observation_bridge` and the
reward/termination config the policy was trained on (read from the run dir's
``config.yaml``). Transitions are appended to an in-memory buffer and written to a
pure-numpy ``.npz`` (no torch), which is then fed to
``vehicle_simulation.agents.finetune_agent`` on a laptop/server.

The policy still runs the car, so all of :mod:`.agent_node`'s safety machinery is
reused: start/stop gating, an input-freshness gate, and brake-to-standstill on stop
or on an off-track/stall termination. Exploration is bounded by clamping throttle to
``[throttle_min, throttle_cap]`` (steering left full-range); the applied (clamped)
action is what gets logged, so the buffer reflects what the car actually did.

Transition timing mirrors ``DriftEnv.step``: the reward for the action taken at tick
``t`` is scored at tick ``t+1`` from the resulting state, so each stored transition
is ``(obs_t, action_t, reward_t, obs_{t+1}, terminated, truncated)``.
"""

from datetime import datetime
from pathlib import Path

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty

from agent_pkg.observation_bridge import observation_from_messages, reward_from_messages
from shared_interfaces.msg import Track, TrackProgress, VehicleActionRequest
from shared_pkg import Topics
from vehicle_simulation.environments.config import StageConfig
from vehicle_simulation.environments.rewards import progress_delta


class DataCollectionNode(Node):
    """Drive the car with a sim policy and record transitions for fine-tuning."""

    def __init__(self) -> None:
        super().__init__("data_collection_node")
        self._load_params()
        self._stage = self._load_stage()
        self._reward_cfg = self._stage.env.reward
        self._term_cfg = self._stage.env.termination
        self._dt = 1.0 / self._rate_hz
        self._agent = self._load_agent()

        self._odom: Odometry | None = None
        self._progress: TrackProgress | None = None
        self._track: Track | None = None
        self._odom_stamp = None
        self._progress_stamp = None
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Gating / braking, identical to agent_node.
        self._running = False
        self._stopping = False
        self._stop_started = None
        self._neutral_ticks = 0
        self._stop_neutral_ticks = max(1, round(self._rate_hz * 0.3))

        # Transition-building state (see _record_step). A "pending" transition is one
        # whose action has been applied but whose reward/next_obs are not yet known
        # (they arrive with the next state).
        self._stall_time = 0.0
        self._pending = False
        self._pending_obs = np.zeros(0, dtype=np.float32)
        self._pending_action = np.zeros(2, dtype=np.float32)
        self._pending_prev_action = np.zeros(2, dtype=np.float32)
        self._pending_distance = 0.0
        self._transitions: list[dict] = []

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._out_path = Path(self._output_dir) / f"{stamp}__recording.npz"
        self._out_path.parent.mkdir(parents=True, exist_ok=True)

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, Topics.CAR_ODOM, self._odom_callback, 10)
        self.create_subscription(TrackProgress, Topics.TRACK_PROGRESS, self._progress_callback, 10)
        self.create_subscription(Track, Topics.TRACK_CENTERLINE, self._track_callback, latched_qos)
        self.create_subscription(Empty, Topics.AGENT_START, self._start_callback, 10)
        self.create_subscription(Empty, Topics.AGENT_STOP, self._stop_callback, 10)
        self._action_pub = self.create_publisher(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10
        )
        self.create_timer(self._dt, self._control_step)
        if self._flush_period_s > 0.0:
            self.create_timer(self._flush_period_s, self._flush)
        self.get_logger().info(f"Recording transitions to {self._out_path}")

    # --- ROS callbacks -------------------------------------------------------

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
        self._stopping = False
        self._begin_episode()
        self.get_logger().info("Received start signal — recording armed")

    def _stop_callback(self, _msg: Empty) -> None:
        if not self._running and not self._stopping:
            return
        self._running = False
        # A manual stop truncates the current episode (bootstrap, not a true
        # terminal): flag the last completed transition so SAC bootstraps from it,
        # and drop the incomplete pending one. Then persist the buffer.
        self._truncate_last()
        self._pending = False
        self._flush()
        if not self._stopping:
            self._stopping = True
            self._stop_started = self.get_clock().now()
            self._neutral_ticks = 0
            self.get_logger().info("Received stop signal — braking to a standstill")

    # --- control / recording loop -------------------------------------------

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
        self._record_step()

    def _record_step(self) -> None:
        """Score the pending transition, then sample and apply the next action."""
        odom, progress, track = self._odom, self._progress, self._track
        obs = observation_from_messages(odom, progress, track, self._prev_action)

        # Close out the transition whose action we applied last tick: its reward and
        # next_obs are known now that the resulting state has arrived.
        if self._pending:
            ds = progress_delta(progress.distance, self._pending_distance, progress.total_length)
            speed = self._current_speed() or 0.0
            self._stall_time = (
                self._stall_time + self._dt if speed < self._term_cfg.stall_speed else 0.0
            )
            stalled = self._stall_time >= self._term_cfg.stall_timeout_s
            outcome = reward_from_messages(
                self._reward_cfg,
                self._term_cfg,
                odom,
                progress,
                track,
                action=self._pending_action,
                prev_action=self._pending_prev_action,
                ds=ds,
                dt=self._dt,
                stalled=stalled,
            )
            self._append_transition(
                self._pending_obs, self._pending_action, outcome.reward, obs, outcome.terminated
            )
            if outcome.terminated:
                self.get_logger().info(
                    f"Episode terminated ({outcome.reason}) — braking; "
                    f"{len(self._transitions)} transitions so far"
                )
                self._pending = False
                self._enter_braking()
                return

        # Sample the next action (stochastic for exploration) and clamp throttle.
        action = self._agent.predict(obs, deterministic=self._deterministic)
        action = np.array(
            [action[0], np.clip(action[1], self._throttle_min, self._throttle_cap)],
            dtype=np.float32,
        )
        self._publish_action(float(action[0]), float(action[1]))

        # Arm the next pending transition (reward/next_obs arrive next tick).
        self._pending = True
        self._pending_obs = obs
        self._pending_prev_action = self._prev_action
        self._pending_action = action
        self._pending_distance = progress.distance
        self._prev_action = action

    # --- transition buffer ---------------------------------------------------

    def _append_transition(self, obs, action, reward, next_obs, terminated) -> None:
        self._transitions.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": np.float32(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "terminated": bool(terminated),
                "truncated": False,
            }
        )

    def _truncate_last(self) -> None:
        """Mark the last completed transition truncated (unless it truly terminated)."""
        if self._transitions and not self._transitions[-1]["terminated"]:
            self._transitions[-1]["truncated"] = True

    def _flush(self) -> None:
        """Write the accumulated buffer to the session npz (overwrites in place)."""
        if not self._transitions:
            return
        obs = np.stack([t["obs"] for t in self._transitions])
        next_obs = np.stack([t["next_obs"] for t in self._transitions])
        actions = np.stack([t["action"] for t in self._transitions])
        rewards = np.array([t["reward"] for t in self._transitions], dtype=np.float32)
        terminateds = np.array([t["terminated"] for t in self._transitions], dtype=bool)
        truncateds = np.array([t["truncated"] for t in self._transitions], dtype=bool)
        track_name = self._track.name if self._track is not None else ""
        np.savez(
            self._out_path,
            observations=obs,
            next_observations=next_obs,
            actions=actions,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            rate_hz=np.float32(self._rate_hz),
            run_dir=str(self._run_dir),
            stage_name=self._stage.name,
            track_name=track_name,
        )
        self.get_logger().info(f"Wrote {len(self._transitions)} transitions to {self._out_path}")

    # --- episode lifecycle ---------------------------------------------------

    def _begin_episode(self) -> None:
        """Reset per-episode state so a (re)start never reuses stale history."""
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._stall_time = 0.0
        self._pending = False

    def _enter_braking(self) -> None:
        """Stop recording this episode and brake to a standstill (off-track/stall)."""
        self._running = False
        # No-op for the terminal transition (it keeps terminated=True).
        self._truncate_last()
        self._flush()
        if not self._stopping:
            self._stopping = True
            self._stop_started = self.get_clock().now()
            self._neutral_ticks = 0

    def _stop_step(self) -> None:
        """Brake to a standstill, then send neutral and release control."""
        elapsed = (self.get_clock().now() - self._stop_started).nanoseconds * 1e-9
        speed = self._current_speed()
        moving = speed is not None and speed > self._stop_speed_threshold
        if moving and elapsed < self._stop_timeout_s:
            self._publish_action(0.0, self._stop_brake_throttle)
            self._prev_action = np.zeros(2, dtype=np.float32)
            return
        self._publish_action(0.0, 0.0)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._neutral_ticks += 1
        if self._neutral_ticks >= self._stop_neutral_ticks:
            self._stopping = False
            self.get_logger().info("Car stopped — recording disarmed, releasing control")

    # --- shared helpers (mirror agent_node) ----------------------------------

    def _current_speed(self) -> float | None:
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
                    f"{name} is stale ({age:.2f} s > {self._input_timeout_s} s) — not recording",
                    throttle_duration_sec=2.0,
                )
                return False
        return True

    def _load_params(self) -> None:
        self.declare_parameter("run_dir", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("prefer_best", True)
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("input_timeout_s", 0.3)
        self.declare_parameter("gyro_mode", "soft")
        self.declare_parameter("gyro_gain", 0.0)
        # Exploration: sample the policy (deterministic=False) and clamp throttle so
        # the car cannot explore into a high-speed loss of control.
        self.declare_parameter("deterministic", False)
        self.declare_parameter("throttle_cap", 0.5)
        self.declare_parameter("throttle_min", -0.4)
        # Recording output.
        self.declare_parameter("output_dir", "recordings")
        self.declare_parameter("flush_period_s", 10.0)
        # Braking-to-standstill behaviour (shared with agent_node).
        self.declare_parameter("stop_brake_throttle", -0.4)
        self.declare_parameter("stop_speed_threshold", 0.2)
        self.declare_parameter("stop_timeout_s", 2.0)

        self._run_dir = self.get_parameter("run_dir").value
        self._device = self.get_parameter("device").value
        self._prefer_best = bool(self.get_parameter("prefer_best").value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._input_timeout_s = float(self.get_parameter("input_timeout_s").value)
        self._gyro_mode = self.get_parameter("gyro_mode").value
        self._gyro_gain = float(self.get_parameter("gyro_gain").value)
        self._deterministic = bool(self.get_parameter("deterministic").value)
        self._throttle_cap = float(self.get_parameter("throttle_cap").value)
        self._throttle_min = float(self.get_parameter("throttle_min").value)
        self._output_dir = self.get_parameter("output_dir").value
        self._flush_period_s = float(self.get_parameter("flush_period_s").value)
        self._stop_brake_throttle = float(self.get_parameter("stop_brake_throttle").value)
        self._stop_speed_threshold = float(self.get_parameter("stop_speed_threshold").value)
        self._stop_timeout_s = float(self.get_parameter("stop_timeout_s").value)

    def _load_stage(self) -> StageConfig:
        """Load the StageConfig the policy was trained on (from the run dir's config.yaml)."""
        if not self._run_dir:
            raise ValueError("The 'run_dir' parameter is required (a training run directory)")
        config_path = Path(self._run_dir) / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No config.yaml in {self._run_dir} — needed for the reward/termination config"
            )
        return StageConfig.from_yaml(config_path)

    def _load_agent(self):
        from vehicle_simulation.agents import DriftAgent

        agent = DriftAgent(self._run_dir, device=self._device, prefer_best=self._prefer_best)
        self.get_logger().info(f"Loaded agent checkpoint: {agent.path}")
        return agent


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
