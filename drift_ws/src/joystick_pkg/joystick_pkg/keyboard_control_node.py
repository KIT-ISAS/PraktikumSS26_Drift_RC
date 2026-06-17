"""
Keyboard teleop: drive the simulated car without a controller.

Publishes :class:`VehicleActionRequest` on ``Topics.VEHICLE_ACTION_REQUEST`` so
the simulation_node consumes it exactly like joystick input. Run it in its own
terminal (it needs a real TTY for the key presses)::

    ros2 run joystick_pkg keyboard_control

Controls:
    W / Up      throttle up   (forward)
    S / Down    throttle down (brake / reverse)
    A / Left    steer left    (hold)
    D / Right   steer right   (hold)
    Space       stop: throttle 0 and re-center
    Q / Esc     quit

Throttle is a *latched setpoint* (cruise-control style): tapping W/S nudges it
and it holds — you do not keep a key held. A terminal can only auto-repeat one
key at a time, so a held throttle key would block steering input; latching lets
you set the throttle once and then steer freely (accelerate + steer "at once").
Steering is hold-to-turn and auto-centers shortly after you release A/D.
"""

import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Topics

_STEER_STEP = 0.18  # per tick while a steer key is held
_THROTTLE_STEP = 0.06  # per repeated key event; throttle is a latched setpoint
_STEER_DECAY = 0.80  # per tick toward center once a steer key is released
# Grace window before steering starts re-centering. A terminal's key auto-repeat
# is slower than the tick and starts with an initial delay, so many ticks see no
# byte for a key that is still held; without this grace steering would decay in
# those gaps and never build up. Must exceed the auto-repeat gap (incl. the
# initial delay) so a held key stays pinned; on real release it re-centers after
# this lag. (Throttle is latched, so it needs no grace/decay.)
_INPUT_GRACE_S = 0.3


class KeyboardControlNode(Node):
    """Reads raw key presses and republishes them as vehicle action requests."""

    def __init__(self) -> None:
        super().__init__("keyboard_control_node")
        self.declare_parameter("rate_hz", 50.0)
        rate = float(self.get_parameter("rate_hz").value)

        self._steering = 0.0
        self._throttle = 0.0  # latched setpoint
        self._last_steer_t = 0.0
        self._pub = self.create_publisher(VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            "Keyboard control ready. W/S = throttle, A/D = steer "
            "(arrows work too), Space = stop, Q = quit."
        )

    def _read_keys(self) -> str:
        """Return all key bytes available on stdin right now (non-blocking)."""
        chars = ""
        while select.select([sys.stdin], [], [], 0.0)[0]:
            chars += sys.stdin.read(1)
        return chars

    def _tick(self) -> None:
        buf = self._read_keys()
        now = time.monotonic()

        if buf == "\x1b" or "q" in buf.lower():
            raise KeyboardInterrupt

        # Fold arrow escape sequences into their WASD equivalents, consuming them
        # so the trailing bracket letters aren't also misread as WASD presses.
        buf = (
            buf.replace("\x1b[A", "w")
            .replace("\x1b[B", "s")
            .replace("\x1b[D", "a")
            .replace("\x1b[C", "d")
        )
        low = buf.lower()
        up, down, left, right = ("w" in low, "s" in low, "a" in low, "d" in low)

        # Throttle is a latched setpoint: W/S nudge it and it holds (no decay), so
        # you can set cruise once and then steer without keeping a key held.
        if up:
            self._throttle = min(1.0, self._throttle + _THROTTLE_STEP)
        if down:
            self._throttle = max(-1.0, self._throttle - _THROTTLE_STEP)
        # ISO 8855 convention (matches the joystick mapper and the models):
        # positive steering = left (CCW, +yaw_rate), negative = right.
        if left:
            self._steering = min(1.0, self._steering + _STEER_STEP)
        if right:
            self._steering = max(-1.0, self._steering - _STEER_STEP)
        if left or right:
            self._last_steer_t = now
        if " " in buf:
            self._throttle = 0.0
            self._steering = 0.0
            self._last_steer_t = 0.0

        # Steering re-centers only after the key has been quiet longer than the
        # auto-repeat gap, so a held key keeps it pinned and a real release lets
        # it return to center. Throttle is latched, so it is never decayed here.
        if now - self._last_steer_t > _INPUT_GRACE_S:
            self._steering *= _STEER_DECAY

        self._publish()

    def _publish(self) -> None:
        msg = VehicleActionRequest()
        msg.steering = float(self._steering)
        msg.throttle = float(self._throttle)
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeyboardControlNode()
    old_attr = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
