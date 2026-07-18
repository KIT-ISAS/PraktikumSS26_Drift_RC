"""
Drive the car with a trained RL agent.

Closes the loop the training environment simulates: subscribes to the canonical
car state (``/car/odom`` — pose + body-frame twist) and the track topics
(``/track/progress``, latched ``/track/centerline``), builds the policy
observation with the **shared** builder from
``drift_core.environments.observation`` (the sim-to-real seam — the exact
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
publishing (letting a joystick or the firmware failsafe take over) and warns. A
dropout longer than ``disarm_timeout_s`` disarms the node instead of holding off —
after a long gap (occluded tracker, car picked up and repositioned) a silent
resume would be surprising, so a fresh start signal is required.

The gating/stopping state machine and everything else shared with
``data_collection_node`` live in :mod:`agent_pkg.lib`.
"""

import rclpy

from agent_pkg.lib.agent_control_node import AgentControlNode
from agent_pkg.observation_bridge import observation_from_messages

__all__ = ["AgentNode", "main"]


class AgentNode(AgentControlNode):
    """Runs a trained drift agent as the control source at a fixed rate."""

    def __init__(self) -> None:
        super().__init__("agent_node", activity="agent", stale_verb="publishing")
        self._agent = self._load_agent()

    def _active_step(self) -> None:
        obs = observation_from_messages(self._odom, self._progress, self._track, self._prev_action)
        action = self._agent.predict(obs)
        self._prev_action = action
        self._publish_action(float(action[0]), float(action[1]))


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
