"""
Arm/stop/brake state machine shared by ``agent_node`` and ``data_collection_node``.

Pure Python on purpose: no rclpy, no clocks, no logging, no publishing. Time is
injected as ``float`` seconds and every method returns a decision for the caller
to act on, so the safety-critical start -> stop -> brake -> neutral -> release and
override paths are unit-testable without a running ROS graph.

States: *idle* (disarmed, silent) -> ``start()`` -> *armed* (caller publishes
actions) -> ``stop()``/``disarm()`` -> *stopping* (caller publishes the
``stop_step`` brake/neutral commands each tick) -> *released* (back to idle).
``override()`` releases immediately from any active state — another controller
has seized the action topic and braking would fight its commands.
"""

import enum
from typing import NamedTuple


class InputFreshness(enum.Enum):
    """How old an input is relative to the hold-off and disarm thresholds."""

    FRESH = enum.auto()
    STALE = enum.auto()  # hold off publishing, expect the input back shortly
    DROPOUT = enum.auto()  # too long gone — disarm, a silent resume would be surprising


class StopCommand(NamedTuple):
    """One braking-phase tick: the action to publish and whether control is released."""

    steering: float
    throttle: float
    released: bool


class ControlGate:
    """Tracks armed/stopping state and decides the braking-to-standstill commands."""

    def __init__(
        self,
        rate_hz: float,
        input_timeout_s: float,
        disarm_timeout_s: float,
        stop_brake_throttle: float,
        stop_speed_threshold: float,
        stop_timeout_s: float,
    ) -> None:
        self.input_timeout_s = input_timeout_s
        self.disarm_timeout_s = disarm_timeout_s
        self.stop_brake_throttle = stop_brake_throttle
        self.stop_speed_threshold = stop_speed_threshold
        self.stop_timeout_s = stop_timeout_s
        # Publish neutral for a short settle window (a few packets, in case of loss)
        # before releasing control, so the car is confirmed at zero throttle.
        self._stop_neutral_ticks = max(1, round(rate_hz * 0.3))

        self.running = False
        self.stopping = False
        self._stop_started = 0.0
        self._neutral_ticks = 0

    def start(self) -> bool:
        """Arm; cancels an in-progress braking phase. True if newly armed."""
        if self.running:
            return False
        self.running = True
        self.stopping = False
        return True

    def stop(self, now: float) -> bool:
        """
        Disarm and enter the braking phase. True if the braking phase is new.

        Idempotent: a repeated stop while already braking does not restart the
        brake timer. A stop while idle is a no-op.
        """
        if not self.running and not self.stopping:
            return False
        self.running = False
        return self._enter_stopping(now)

    def override(self) -> bool:
        """Release control immediately, cancelling any braking phase. True if active."""
        if not self.running and not self.stopping:
            return False
        self.running = False
        self.stopping = False
        return True

    def disarm(self, now: float) -> None:
        """Disarm through the braking phase (long input dropout, terminal state)."""
        self.running = False
        self._enter_stopping(now)

    def _enter_stopping(self, now: float) -> bool:
        if self.stopping:
            return False
        self.stopping = True
        self._stop_started = now
        self._neutral_ticks = 0
        return True

    def classify_input_age(self, age: float) -> InputFreshness:
        if age <= self.input_timeout_s:
            return InputFreshness.FRESH
        if age > self.disarm_timeout_s:
            return InputFreshness.DROPOUT
        return InputFreshness.STALE

    def stop_step(self, now: float, speed: float | None) -> StopCommand:
        """
        One tick of the braking phase; call only while ``stopping``.

        Brakes (steering straight) while the car is visibly still moving and the
        safety timeout has not elapsed. Unknown speed (stale/missing odom) falls
        through to neutral, which is the safe command regardless. Once standing
        (or timed out), holds neutral for the settle window, then releases.
        """
        elapsed = now - self._stop_started
        moving = speed is not None and speed > self.stop_speed_threshold
        if moving and elapsed < self.stop_timeout_s:
            return StopCommand(0.0, self.stop_brake_throttle, released=False)
        self._neutral_ticks += 1
        if self._neutral_ticks >= self._stop_neutral_ticks:
            self.stopping = False
            return StopCommand(0.0, 0.0, released=True)
        return StopCommand(0.0, 0.0, released=False)
