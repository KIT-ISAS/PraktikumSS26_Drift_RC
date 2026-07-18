"""
Record human joystick demonstrations for off-board SAC fine-tuning.

Passive counterpart to :mod:`.data_collection_node`: the human drives via the
joystick as in normal manual mode and this node publishes nothing — it listens to
the human's ``VehicleActionRequest`` commands alongside the canonical state topics
and reconstructs the same ``(obs, action, reward, next_obs, done)`` transitions the
training environment would have produced (scored with the reward/termination config
of a stage YAML, ``stage_config`` parameter). The resulting ``.npz`` files feed
``drift_core.agents.finetune_agent`` exactly like policy recordings; SAC is
off-policy, so human transitions are valid replay data.

Since no policy holds the car, there is no start/stop gating: recording runs
automatically whenever all inputs (odom, track progress, joystick command) are
fresh. The human's command is latched and sampled at the control rate (50 Hz), so
the stored action is the command that was held during each training-time step.

Episode handling mirrors ``DriftEnv``: a transition that the reward function marks
terminated (off-track / stall) ends the episode, and recording then holds until the
car has *recovered* — back inside the track bounds and moving — so a car parked or
carried around does not flood the buffer with one-step terminal episodes. An input
gap cuts the episode (last completed transition is truncated, the pending one is
dropped), just like the policy recorder.
"""

from datetime import datetime
import math
from pathlib import Path

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from agent_pkg.lib.transition_recorder import TransitionRecorder
from agent_pkg.observation_bridge import observation_from_messages, score_step
from drift_core.environments.config import StageConfig
from drift_core.environments.rewards import TerminationConfig
from shared_interfaces.msg import Track, TrackProgress, VehicleActionRequest
from shared_pkg import Topics

# Resume recording only once |lateral_error| is back below this fraction of the
# termination limit, so the episode boundary doesn't chatter right at the limit.
RECOVERY_LATERAL_FRACTION = 0.9


def is_recovered(lateral_error: float, speed: float, term_cfg: TerminationConfig) -> bool:
    """Whether the car is back on track and moving, so a new episode can start."""
    on_track = abs(lateral_error) <= RECOVERY_LATERAL_FRACTION * term_cfg.lateral_limit_m
    return on_track and speed > term_cfg.stall_speed


class DemoRecorderNode(Node):
    """Passively record human joystick driving as fine-tuning transitions."""

    def __init__(self) -> None:
        super().__init__("demo_recorder_node")
        self._load_params()
        stage = StageConfig.load(self._stage_config)
        self._reward_cfg = stage.env.reward
        self._term_cfg = stage.env.termination
        self._dt = 1.0 / self._rate_hz

        self._odom: Odometry | None = None
        self._progress: TrackProgress | None = None
        self._track: Track | None = None
        self._action_msg: VehicleActionRequest | None = None
        self._odom_stamp = None
        self._progress_stamp = None
        self._action_stamp = None
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._stall_time = 0.0
        # Start in the recovery hold: recording begins once the car is on track
        # and moving, not while it is still being carried to the start.
        self._recovering = True
        # Gyro assist settings, latched when the first transition is armed. The
        # ESP32 applies the correction after the command is logged, so all
        # transitions in a session file must share one setting for the npz
        # metadata to be truthful; a mid-session change cuts the episode and
        # holds recording until the latched settings are restored.
        self._gyro: tuple[str, float] | None = None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recorder = TransitionRecorder(
            Path(self._output_dir) / f"{stamp}__human_demo.npz",
            rate_hz=self._rate_hz,
            stage_name=stage.name,
            run_dir="human",
        )

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, Topics.CAR_ODOM, self._odom_callback, 10)
        self.create_subscription(TrackProgress, Topics.TRACK_PROGRESS, self._progress_callback, 10)
        self.create_subscription(Track, Topics.TRACK_CENTERLINE, self._track_callback, latched_qos)
        self.create_subscription(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, self._action_callback, 10
        )
        self.create_timer(1.0 / self._rate_hz, self._tick)
        if self._flush_period_s > 0.0:
            self.create_timer(self._flush_period_s, self.flush)
        self.get_logger().info(
            f"Recording human demonstrations to {self._recorder.out_path} (stage {stage.name!r})"
        )

    # --- ROS callbacks ---------------------------------------------------------

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

    def _action_callback(self, msg: VehicleActionRequest) -> None:
        self._action_msg = msg
        self._action_stamp = self.get_clock().now()

    # --- recording loop ----------------------------------------------------------

    def _tick(self) -> None:
        """Score the pending transition, then latch the human's current command."""
        if not self._inputs_ready() or not self._gyro_matches():
            self._cut_episode()
            return
        odom, progress, track = self._odom, self._progress, self._track
        obs = observation_from_messages(odom, progress, track, self._prev_action)

        # Close out the transition armed last tick: its reward and next_obs are
        # known now that the resulting state has arrived.
        if self._recorder.pending:
            outcome, self._stall_time = score_step(
                self._reward_cfg,
                self._term_cfg,
                odom,
                progress,
                track,
                action=self._recorder.pending_action,
                prev_action=self._recorder.pending_prev_action,
                prev_distance=self._recorder.pending_distance,
                dt=self._dt,
                stall_time=self._stall_time,
            )
            self._recorder.complete(outcome.reward, obs, outcome.terminated)
            if outcome.terminated:
                self.get_logger().info(
                    f"Episode terminated ({outcome.reason}) — holding until the car "
                    f"recovers; {len(self._recorder)} transitions so far"
                )
                self._begin_episode()
                self.flush()
                return

        if self._recovering:
            speed = math.hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
            if not is_recovered(progress.lateral_error, speed, self._term_cfg):
                return
            self._recovering = False
            self.get_logger().info("Car on track and moving — recording")

        if self._gyro is None:
            self._latch_gyro()
        action = np.array([self._action_msg.steering, self._action_msg.throttle], dtype=np.float32)
        self._recorder.arm(obs, action, self._prev_action, progress.distance)
        self._prev_action = action

    def _latch_gyro(self) -> None:
        """Latch the session's gyro assist settings into the npz metadata."""
        self._gyro = (self._action_msg.gyro_mode, self._action_msg.gyro_gain)
        self._recorder.metadata["gyro_mode"] = self._gyro[0]
        self._recorder.metadata["gyro_gain"] = np.float32(self._gyro[1])
        if self._gyro[1] != 0.0:
            self.get_logger().warning(
                f"Recording with gyro gain {self._gyro[1]:.2f}: the ESP32 corrects the "
                "steering after it is logged and the training sim has no such assist — "
                "set the gain to 0 (d-pad down) for clean demos"
            )

    def _gyro_matches(self) -> bool:
        """The latched gyro settings still apply; hold recording otherwise."""
        if self._action_msg is None or self._gyro is None:
            return True
        current = (self._action_msg.gyro_mode, self._action_msg.gyro_gain)
        if current == self._gyro:
            return True
        self.get_logger().warning(
            f"Gyro settings changed to {current[0]}/{current[1]:.2f} mid-session — recording "
            f"on hold; restore {self._gyro[0]}/{self._gyro[1]:.2f} (the session's npz "
            "metadata) to resume",
            throttle_duration_sec=2.0,
        )
        return False

    def _inputs_ready(self) -> bool:
        """All inputs present and fresh; warn (throttled) otherwise."""
        if (
            self._track is None
            or self._odom is None
            or self._progress is None
            or self._action_msg is None
        ):
            self.get_logger().warning(
                "Waiting for inputs (odom / track progress / centerline / joystick command)",
                throttle_duration_sec=2.0,
            )
            return False
        now = self.get_clock().now()
        stamps = (
            ("odom", self._odom_stamp),
            ("track progress", self._progress_stamp),
            ("joystick command", self._action_stamp),
        )
        for name, stamp in stamps:
            age = (now - stamp).nanoseconds * 1e-9
            if age > self._input_timeout_s:
                self.get_logger().warning(
                    f"{name} is stale ({age:.2f} s > {self._input_timeout_s} s) — not recording",
                    throttle_duration_sec=2.0,
                )
                return False
        return True

    # --- episode lifecycle ---------------------------------------------------

    def _cut_episode(self) -> None:
        """An input gap or gyro change cuts the episode (mirrors the policy recorder)."""
        if not self._recorder.pending:
            return
        self._recorder.truncate_last()
        self._begin_episode()
        self.get_logger().warning("Episode cut — dropping the pending transition")

    def _begin_episode(self) -> None:
        """Reset per-episode state and hold recording until the car has recovered."""
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._stall_time = 0.0
        self._recorder.drop_pending()
        self._recovering = True

    def flush(self) -> None:
        """Write the accumulated buffer to the session npz (atomic overwrite)."""
        track_name = self._track.name if self._track is not None else ""
        count = self._recorder.flush(track_name)
        if count:
            self.get_logger().info(f"Wrote {count} transitions to {self._recorder.out_path}")

    def shutdown(self) -> None:
        """Truncate the interrupted episode (a shutdown is not a true terminal) and persist."""
        self._recorder.truncate_last()
        self._recorder.drop_pending()
        self.flush()

    # --- config ----------------------------------------------------------------

    def _load_params(self) -> None:
        self.declare_parameter("stage_config", "")
        self.declare_parameter("rate_hz", 50.0)  # matches the training env's dt = 0.02
        self.declare_parameter("input_timeout_s", 0.3)
        self.declare_parameter("output_dir", "~/recordings/human")
        self.declare_parameter("flush_period_s", 10.0)

        self._stage_config = self.get_parameter("stage_config").value
        if not self._stage_config:
            raise ValueError(
                "The 'stage_config' parameter is required (a packaged stage name or YAML path)"
            )
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._input_timeout_s = float(self.get_parameter("input_timeout_s").value)
        self._output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
        self._flush_period_s = float(self.get_parameter("flush_period_s").value)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DemoRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
