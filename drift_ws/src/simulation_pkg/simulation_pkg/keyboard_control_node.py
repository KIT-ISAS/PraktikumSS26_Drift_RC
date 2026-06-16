"""
Keyboard teleop: drive the simulated car without a controller.

Publishes :class:`VehicleActionRequest` on ``Topics.VEHICLE_ACTION_REQUEST`` so
the simulation_node consumes it exactly like joystick input. Run it in its own
terminal (it needs a real TTY for the key presses)::

    ros2 run simulation_pkg keyboard_control

Controls:
    W / Up      accelerate forward
    S / Down    brake / reverse
    A / Left    steer left
    D / Right   steer right
    Space       cut throttle and re-center
    Q / Esc     quit

Steering auto-centers and throttle coasts back to zero when no key is held, so
holding a direction drifts; tapping nudges.
"""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Topics

_STEER_STEP = 0.18
_THROTTLE_STEP = 0.12
_STEER_DECAY = 0.80  # per tick toward center when no steer key is pressed
_THROTTLE_DECAY = 0.97  # per tick coast-down when no throttle key is pressed


class KeyboardControlNode(Node):
    """Reads raw key presses and republishes them as vehicle action requests."""

    def __init__(self) -> None:
        super().__init__("keyboard_control_node")
        self.declare_parameter("rate_hz", 50.0)
        rate = float(self.get_parameter("rate_hz").value)

        self._steering = 0.0
        self._throttle = 0.0
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

        if not buf:
            self._steering *= _STEER_DECAY
            self._throttle *= _THROTTLE_DECAY
            self._publish()
            return

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

        if up:
            self._throttle = min(1.0, self._throttle + _THROTTLE_STEP)
        if down:
            self._throttle = max(-1.0, self._throttle - _THROTTLE_STEP)
        if left:
            self._steering = max(-1.0, self._steering - _STEER_STEP)
        if right:
            self._steering = min(1.0, self._steering + _STEER_STEP)
        if " " in buf:
            self._throttle = 0.0
            self._steering = 0.0

        if not (up or down):
            self._throttle *= _THROTTLE_DECAY
        if not (left or right):
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
