"""
Maps the Joystick topic with the correct joystick key mapping for the drift car.

Two modes:

- Default: every joystick frame is mapped to a ``VehicleActionRequest`` (manual
  driving — the joystick is the only control source).
- Agent-override (``agent_override_mode``): the RL agent is the control source
  and the joystick is a passive dead-man's switch. The node publishes nothing
  until a significant input (steering or throttle beyond
  ``override_threshold``); it then signals the agent to release control
  immediately (``/agent/stop`` + ``/agent/override``) and forwards joystick
  commands from that same frame on. A subsequent ``/agent/start`` hands control
  back to the agent and re-arms the passive watch.

In both modes two mapped buttons (``button_mapping``) control the agent: the
start button publishes ``/agent/start`` (arming the agent and, in override
mode, returning the joystick to passive), the stop button publishes
``/agent/stop`` (the agent brakes to a standstill before releasing control).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty

from shared_interfaces.msg import VehicleActionRequest
from shared_pkg import Topics

from .lib import JoystickMapper, XBoxOneMapper


class JoystickMapperNode(Node):
    """ROS2 node that maps joystick input to vehicle action requests."""

    def __init__(self):
        """Initialize the joystick mapper node."""
        super().__init__("joystick_mapper")
        self._load_params()
        self._dpad_prev_h: float | None = None
        self._dpad_prev_v: float | None = None
        self._start_btn_prev = False
        self._stop_btn_prev = False
        self._mapper: JoystickMapper = XBoxOneMapper(self.axis_mapping)

        # In override mode the joystick starts passive (the agent drives) and
        # only takes over on significant movement.
        self._passive = self.agent_override_mode

        # Create publisher for vehicle action requests
        self.vehicle_action_publisher = self.create_publisher(
            VehicleActionRequest, Topics.VEHICLE_ACTION_REQUEST, 10
        )

        self.agent_start_publisher = self.create_publisher(Empty, Topics.AGENT_START, 10)
        self.agent_stop_publisher = self.create_publisher(Empty, Topics.AGENT_STOP, 10)

        if self.agent_override_mode:
            self.agent_override_publisher = self.create_publisher(Empty, Topics.AGENT_OVERRIDE, 10)
            # A start signal hands control back to the agent — go passive again.
            self.agent_start_subscriber = self.create_subscription(
                Empty, Topics.AGENT_START, self._agent_start_callback, 10
            )
            self.get_logger().info(
                "Agent-override mode: joystick passive, watching for significant movement "
                f"(threshold {self.override_threshold:.2f})"
            )

        # Subscribe to joy topic
        self.joy_subscriber = self.create_subscription(
            Joy, Topics.JOY_NODE, self._joy_callback, 10
        )

    def _joy_callback(self, msg: Joy) -> None:
        """
        Process joystick input and publish vehicle action requests.

        Maps the parameterized joystick axes to steering and throttle values.
        D-pad up/down adjusts gyro gain; d-pad left/right sets gyro mode.
        The mapped start/stop buttons signal the agent (rising edge only).
        In agent-override mode, nothing is published while passive; significant
        movement triggers the takeover first.
        """
        self._check_agent_signal(msg)
        self._update_gyro_from_dpad(msg)

        steering = self._mapper.extract_steering(msg)
        throttle = self._mapper.extract_throttle(msg)

        if self._passive:
            if max(abs(steering), abs(throttle)) < self.override_threshold:
                return
            self._passive = False
            self.agent_stop_publisher.publish(Empty())
            self.agent_override_publisher.publish(Empty())
            self.get_logger().warning(
                f"Joystick override (steering {steering:+.2f}, throttle {throttle:+.2f}) "
                "— stopping agent, joystick takes control"
            )

        action_msg = VehicleActionRequest()
        action_msg.steering = steering
        action_msg.throttle = throttle
        action_msg.gyro_mode = self.gyro_mode
        action_msg.gyro_gain = self.gyro_gain
        self.vehicle_action_publisher.publish(action_msg)

    def _agent_start_callback(self, _msg: Empty) -> None:
        """Return to passive watching when the agent is (re)started."""
        if self._passive:
            return
        self._passive = True
        self.get_logger().info("Agent started — joystick passive until significant movement")

    def _check_agent_signal(self, msg: Joy) -> None:
        """Publish agent start/stop signals on the rising edge of the mapped buttons."""
        start_pressed = bool(msg.buttons[self.button_mapping["agent_start"]])
        stop_pressed = bool(msg.buttons[self.button_mapping["agent_stop"]])

        if start_pressed and not self._start_btn_prev:
            self.agent_start_publisher.publish(Empty())
            self.get_logger().info("Start button pressed — starting agent")
            if self.agent_override_mode:
                # Go passive right away instead of waiting for the looped-back
                # /agent/start, so this frame doesn't fight the agent.
                self._agent_start_callback(Empty())
        if stop_pressed and not self._stop_btn_prev:
            self.agent_stop_publisher.publish(Empty())
            self.get_logger().info("Stop button pressed — agent braking to a standstill")

        self._start_btn_prev = start_pressed
        self._stop_btn_prev = stop_pressed

    def _update_gyro_from_dpad(self, msg: Joy) -> None:
        """Update gyro state from d-pad input, triggering once per press."""
        dpad_h = msg.axes[self.axis_mapping["dpad_h"]]
        dpad_v = msg.axes[self.axis_mapping["dpad_v"]]

        if self._dpad_prev_h is not None and self._dpad_prev_v is not None:
            # Rising-edge detection: only act on the frame the button becomes pressed.
            if dpad_v > 0.5 and self._dpad_prev_v <= 0.5:
                self.gyro_gain = round(min(1.0, self.gyro_gain + self.gyro_gain_step), 10)
                self.get_logger().info(f"Gyro gain: {self.gyro_gain:.2f}")
            elif dpad_v < -0.5 and self._dpad_prev_v >= -0.5:
                self.gyro_gain = round(max(0.0, self.gyro_gain - self.gyro_gain_step), 10)
                self.get_logger().info(f"Gyro gain: {self.gyro_gain:.2f}")

            if dpad_h > 0.5 and self._dpad_prev_h <= 0.5:
                self.gyro_mode = "soft"
                self.get_logger().info("Gyro mode: soft")
            elif dpad_h < -0.5 and self._dpad_prev_h >= -0.5:
                self.gyro_mode = "hard"
                self.get_logger().info("Gyro mode: hard")

        self._dpad_prev_h = dpad_h
        self._dpad_prev_v = dpad_v

    def _load_params(self) -> None:
        """Load parameters from the ROS2 parameter server."""
        self.declare_parameter("axis_mapping.throttle", 0)
        self.declare_parameter("axis_mapping.brake", 0)
        self.declare_parameter("axis_mapping.steering", 0)
        self.declare_parameter("axis_mapping.dpad_h", 6)
        self.declare_parameter("axis_mapping.dpad_v", 7)
        axis_mapping_params = self.get_parameters_by_prefix("axis_mapping")
        self.axis_mapping = {
            param_key: param_value.value for param_key, param_value in axis_mapping_params.items()
        }

        self.declare_parameter("button_mapping.agent_start", 0)
        self.declare_parameter("button_mapping.agent_stop", 1)
        button_mapping_params = self.get_parameters_by_prefix("button_mapping")
        self.button_mapping = {
            param_key: param_value.value
            for param_key, param_value in button_mapping_params.items()
        }

        self.declare_parameter("gyro_mode", "soft")
        self.declare_parameter("gyro_gain", 0.0)
        self.declare_parameter("gyro_gain_step", 0.1)
        self.gyro_mode = self.get_parameter("gyro_mode").value
        self.gyro_gain = float(self.get_parameter("gyro_gain").value)
        self.gyro_gain_step = float(self.get_parameter("gyro_gain_step").value)

        self.declare_parameter("agent_override_mode", False)
        self.declare_parameter("override_threshold", 0.3)
        self.agent_override_mode = bool(self.get_parameter("agent_override_mode").value)
        self.override_threshold = float(self.get_parameter("override_threshold").value)


def main(args=None):
    """Start joystick reader node."""
    rclpy.init(args=args)

    joystick_mapper = JoystickMapperNode()

    try:
        rclpy.spin(joystick_mapper)
    except KeyboardInterrupt:
        pass
    finally:
        joystick_mapper.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
