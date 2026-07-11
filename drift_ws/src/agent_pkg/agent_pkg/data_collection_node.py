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
reused (shared via :mod:`agent_pkg.lib`): start/stop gating, an input-freshness
gate (a dropout longer than ``disarm_timeout_s`` disarms — fresh start signal
required), and brake-to-standstill on stop or on an off-track/stall termination.
Exploration is bounded by clamping throttle to
``[throttle_min, throttle_cap]`` (steering left full-range); the applied (clamped)
action is what gets logged, so the buffer reflects what the car actually did.

Transition timing mirrors ``DriftEnv.step``: the reward for the action taken at tick
``t`` is scored at tick ``t+1`` from the resulting state, so each stored transition
is ``(obs_t, action_t, reward_t, obs_{t+1}, terminated, truncated)``.
"""

from datetime import datetime
import os
from pathlib import Path

import numpy as np
import rclpy

from agent_pkg.lib.agent_control_node import AgentControlNode
from agent_pkg.observation_bridge import observation_from_messages, reward_from_messages
from vehicle_simulation.environments.config import StageConfig
from vehicle_simulation.environments.rewards import progress_delta


class DataCollectionNode(AgentControlNode):
    """Drive the car with a sim policy and record transitions for fine-tuning."""

    def __init__(self) -> None:
        super().__init__("data_collection_node", activity="recording", stale_verb="recording")
        self._stage = self._load_stage()
        self._reward_cfg = self._stage.env.reward
        self._term_cfg = self._stage.env.termination
        self._dt = 1.0 / self._rate_hz
        self._agent = self._load_agent()

        # Transition-building state (see _active_step). A "pending" transition is one
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

        if self._flush_period_s > 0.0:
            self.create_timer(self._flush_period_s, self.flush)
        self.get_logger().info(f"Recording transitions to {self._out_path}")

    # --- lifecycle hooks (see AgentControlNode) --------------------------------

    def _on_start(self) -> None:
        self._begin_episode()

    def _on_stop(self) -> None:
        # A manual stop truncates the current episode (bootstrap, not a true
        # terminal): flag the last completed transition so SAC bootstraps from it,
        # and drop the incomplete pending one. Then persist the buffer.
        self._truncate_last()
        self._pending = False
        self.flush()

    def _on_override(self) -> None:
        # The episode ends here, so truncate the last completed transition,
        # drop the incomplete pending one, and persist the buffer.
        self._truncate_last()
        self._pending = False
        self.flush()

    def _on_dropout_disarm(self) -> None:
        self._enter_braking()

    def _on_inputs_stale(self) -> None:
        # A data gap cuts the episode: the pending transition's reward/next_obs
        # would span the whole gap while being scored with dt = one tick,
        # poisoning the fine-tuning buffer. Truncate the last completed
        # transition and start fresh once inputs are back.
        if not self._pending:
            return
        self._truncate_last()
        self._begin_episode()
        self.get_logger().warning(
            "Inputs went stale — episode cut, dropping the pending transition"
        )

    # --- control / recording loop -------------------------------------------

    def _active_step(self) -> None:
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

    def flush(self) -> None:
        """Write the accumulated buffer to the session npz (atomic overwrite)."""
        if not self._transitions:
            return
        obs = np.stack([t["obs"] for t in self._transitions])
        next_obs = np.stack([t["next_obs"] for t in self._transitions])
        actions = np.stack([t["action"] for t in self._transitions])
        rewards = np.array([t["reward"] for t in self._transitions], dtype=np.float32)
        terminateds = np.array([t["terminated"] for t in self._transitions], dtype=bool)
        truncateds = np.array([t["truncated"] for t in self._transitions], dtype=bool)
        track_name = self._track.name if self._track is not None else ""
        # Write to a temp file and rename, so a crash mid-write cannot corrupt the
        # whole session's recording.
        tmp_path = self._out_path.with_name(self._out_path.name + ".tmp")
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
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
        os.replace(tmp_path, self._out_path)
        self.get_logger().info(f"Wrote {len(self._transitions)} transitions to {self._out_path}")

    # --- episode lifecycle ---------------------------------------------------

    def _begin_episode(self) -> None:
        """Reset per-episode state so a (re)start never reuses stale history."""
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._stall_time = 0.0
        self._pending = False

    def _enter_braking(self) -> None:
        """Stop recording this episode and brake to a standstill (off-track/stall/dropout)."""
        # No-op for the terminal transition (it keeps terminated=True).
        self._truncate_last()
        self.flush()
        self._gate.disarm(self._now_s())

    # --- config / loading -----------------------------------------------------

    def _load_params(self) -> None:
        super()._load_params()
        # Exploration: sample the policy (deterministic=False) and clamp throttle so
        # the car cannot explore into a high-speed loss of control.
        self.declare_parameter("deterministic", False)
        self.declare_parameter("throttle_cap", 0.5)
        self.declare_parameter("throttle_min", -0.4)
        # Recording output.
        self.declare_parameter("output_dir", "recordings")
        self.declare_parameter("flush_period_s", 10.0)

        self._deterministic = bool(self.get_parameter("deterministic").value)
        self._throttle_cap = float(self.get_parameter("throttle_cap").value)
        self._throttle_min = float(self.get_parameter("throttle_min").value)
        # Relative paths land wherever `ros2 launch` was invoked; anchor them to
        # the run dir so recordings live next to the policy that produced them.
        output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(self._run_dir) / output_dir
        self._output_dir = output_dir.resolve()
        self._flush_period_s = float(self.get_parameter("flush_period_s").value)

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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DataCollectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.flush()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
