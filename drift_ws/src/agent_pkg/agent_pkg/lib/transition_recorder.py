"""
In-memory transition buffer with atomic ``.npz`` persistence.

Shared by the recording nodes (:mod:`agent_pkg.data_collection_node`,
:mod:`agent_pkg.demo_recorder_node`): both reconstruct the
``(obs, action, reward, next_obs, done)`` transitions ``DriftEnv.step`` would have
produced and persist them for off-board SAC fine-tuning with
``drift_core.agent.finetune_agent``. Pure numpy (no ROS), so the
pending/buffer mechanics are unit-testable in isolation.

A "pending" transition is one whose action has been applied but whose reward and
``next_obs`` are not yet known — they arrive with the next state, one control tick
later, mirroring ``DriftEnv.step``'s timing: the transition stored for tick ``t`` is
``(obs_t, action_t, reward_t, obs_{t+1}, terminated, truncated)``.
"""

import os
from pathlib import Path

import numpy as np


class TransitionRecorder:
    """Buffer transitions and write them to a session ``.npz`` (atomic overwrite)."""

    def __init__(self, out_path, *, rate_hz: float, stage_name: str, run_dir: str) -> None:
        """
        Create the buffer; ``out_path``'s parent directory is created if needed.

        ``run_dir`` records provenance in the npz (the training run directory for
        policy recordings, ``"human"`` for joystick demonstrations). Extra scalar
        fields for the npz can be added to the public ``metadata`` dict.
        """
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata: dict = {}
        self._rate_hz = float(rate_hz)
        self._stage_name = stage_name
        self._run_dir = run_dir
        self._transitions: list[dict] = []
        self._pending = False
        self._pending_obs = np.zeros(0, dtype=np.float32)
        self._pending_action = np.zeros(2, dtype=np.float32)
        self._pending_prev_action = np.zeros(2, dtype=np.float32)
        self._pending_distance = 0.0

    def __len__(self) -> int:
        return len(self._transitions)

    # --- pending transition ----------------------------------------------------

    @property
    def pending(self) -> bool:
        """Whether an armed transition is waiting for its reward/next_obs."""
        return self._pending

    @property
    def pending_action(self) -> np.ndarray:
        return self._pending_action

    @property
    def pending_prev_action(self) -> np.ndarray:
        return self._pending_prev_action

    @property
    def pending_distance(self) -> float:
        return self._pending_distance

    def arm(self, obs, action, prev_action, distance: float) -> None:
        """Arm the next transition: its reward/next_obs arrive with the next state."""
        self._pending = True
        self._pending_obs = np.asarray(obs, dtype=np.float32)
        self._pending_action = np.asarray(action, dtype=np.float32)
        self._pending_prev_action = np.asarray(prev_action, dtype=np.float32)
        self._pending_distance = float(distance)

    def drop_pending(self) -> None:
        """Discard the armed transition (episode cut before its outcome arrived)."""
        self._pending = False

    def complete(self, reward: float, next_obs, terminated: bool) -> None:
        """Close out the armed transition with its now-known reward and next state."""
        self._transitions.append(
            {
                "obs": self._pending_obs,
                "action": self._pending_action,
                "reward": np.float32(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "terminated": bool(terminated),
                "truncated": False,
            }
        )
        self._pending = False

    # --- buffer ------------------------------------------------------------------

    def truncate_last(self) -> None:
        """Mark the last completed transition truncated (unless it truly terminated)."""
        if self._transitions and not self._transitions[-1]["terminated"]:
            self._transitions[-1]["truncated"] = True

    def flush(self, track_name: str) -> int:
        """
        Write the accumulated buffer to the session npz; returns the count written.

        Writes to a temp file and renames, so a crash mid-write cannot corrupt the
        whole session's recording.
        """
        if not self._transitions:
            return 0
        obs = np.stack([t["obs"] for t in self._transitions])
        next_obs = np.stack([t["next_obs"] for t in self._transitions])
        actions = np.stack([t["action"] for t in self._transitions])
        rewards = np.array([t["reward"] for t in self._transitions], dtype=np.float32)
        terminateds = np.array([t["terminated"] for t in self._transitions], dtype=bool)
        truncateds = np.array([t["truncated"] for t in self._transitions], dtype=bool)
        tmp_path = self.out_path.with_name(self.out_path.name + ".tmp")
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
                run_dir=self._run_dir,
                stage_name=self._stage_name,
                track_name=track_name,
                **self.metadata,
            )
        os.replace(tmp_path, self.out_path)
        return len(self._transitions)
