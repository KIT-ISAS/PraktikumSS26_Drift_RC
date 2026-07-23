"""
Drive the car deterministically along the track centerline with a lateral MPC.

Drop-in alternative control source to the RL agent (:mod:`.agent_node`): same
canonical inputs (``/car/odom``, ``/track/progress``, latched
``/track/centerline``), same start/stop/override gating and safety braking
(inherited from :class:`~agent_pkg.lib.agent_control_node.AgentControlNode`),
same ``VehicleActionRequest`` output at the control rate — but the action comes
from :class:`drift_core.control.PathMpc` (linear time-varying MPC on the
kinematic bicycle in Frenet error coordinates) plus a curvature-based speed
profile tracked by a PI throttle controller. No training run required.

On each latched centerline the node rebuilds the speed profile and hands the
curvature arrays to the MPC; each armed tick it solves for the steering command
(solve time ~0.5 ms, far inside the 20 ms budget at 50 Hz) and samples the speed
profile slightly ahead of the car for the throttle loop.
"""

import math

import numpy as np
import rclpy

from agent_pkg.lib.agent_control_node import AgentControlNode
from agent_pkg.mpc_bridge import mpc_inputs_from_messages
from drift_core.car_parameters import load_car_parameters
from drift_core.control import (
    PathMpc,
    PathMpcConfig,
    SpeedController,
    SpeedControllerConfig,
    build_speed_profile,
)
from shared_interfaces.msg import Track

__all__ = ["MpcNode", "main"]


class MpcNode(AgentControlNode):
    """Runs the path-following MPC as the control source at a fixed rate."""

    def __init__(self) -> None:
        """Build the MPC and speed controller from the car and node parameters."""
        super().__init__("mpc_node", activity="MPC", stale_verb="publishing")
        car = load_car_parameters(self._car)
        self._mpc = PathMpc(
            PathMpcConfig(
                wheelbase_m=car.wheelbase_m,
                max_steer_angle_rad=car.max_steer_angle_rad,
                horizon_steps=self._horizon_steps,
                mpc_dt=self._mpc_dt,
                q_lateral=self._q_lateral,
                q_heading=self._q_heading,
                q_terminal=self._q_terminal,
                r_curvature=self._r_curvature,
                r_curvature_rate=self._r_curvature_rate,
                steer_cmd_limit=self._steer_cmd_limit,
                steer_rate_limit=self._steer_rate_limit,
                v_min=self._v_min,
                a_accel=self._a_accel,
            )
        )
        self._speed_controller = SpeedController(
            SpeedControllerConfig(
                kp=self._throttle_kp,
                ki=self._throttle_ki,
                kff=self._throttle_kff,
                throttle_min=self._throttle_min,
                throttle_max=self._throttle_max,
            )
        )
        self._profile = None

    def _track_callback(self, msg: Track) -> None:
        """Cache the centerline (base) and rebuild the speed profile + MPC track data."""
        super()._track_callback(msg)
        last = msg.waypoints[-1].position
        first = msg.waypoints[0].position
        total_length = float(msg.cumulative_distance[-1]) + math.hypot(
            first.x - last.x, first.y - last.y
        )
        self._profile = build_speed_profile(
            msg.curvature,
            msg.cumulative_distance,
            total_length,
            v_max=self._v_max,
            a_lat_max=self._a_lat_max,
            a_brake=self._a_brake,
            a_accel=self._a_accel,
        )
        self._mpc.set_track(msg.curvature, msg.cumulative_distance, total_length, self._profile)
        self.get_logger().info(
            f"Speed profile for {msg.name!r}: "
            f"{self._profile.v_grid.min():.2f} - {self._profile.v_grid.max():.2f} m/s"
        )

    def _active_step(self) -> None:
        lateral_error, heading_error, vx, s = mpc_inputs_from_messages(self._odom, self._progress)
        dt_ctrl = 1.0 / self._rate_hz
        solution = self._mpc.solve(
            lateral_error,
            heading_error,
            vx,
            s,
            prev_steering=float(self._prev_action[0]),
            dt_ctrl=dt_ctrl,
        )
        v_ref = float(self._profile.sample(s + max(vx, 0.0) * self._speed_preview_s))
        throttle = self._speed_controller.step(v_ref, vx, dt_ctrl)
        self._prev_action = np.array([solution.steering, throttle], dtype=np.float32)
        self._publish_action(solution.steering, throttle)

    def _on_start(self) -> None:
        self._speed_controller.reset()

    def _on_stop(self) -> None:
        self._speed_controller.reset()

    def _on_override(self) -> None:
        self._speed_controller.reset()

    def _load_params(self) -> None:
        """Declare the MPC/longitudinal parameters on top of the shared base set."""
        super()._load_params()
        self.declare_parameter("car", "drift_rc")
        self.declare_parameter("horizon_steps", 20)
        self.declare_parameter("mpc_dt", 0.05)
        self.declare_parameter("q_lateral", 20.0)
        self.declare_parameter("q_heading", 3.0)
        self.declare_parameter("q_terminal", 5.0)
        self.declare_parameter("r_curvature", 0.05)
        self.declare_parameter("r_curvature_rate", 0.3)
        self.declare_parameter("steer_cmd_limit", 0.6)
        self.declare_parameter("steer_rate_limit", 6.0)
        self.declare_parameter("v_min", 0.3)
        self.declare_parameter("v_max", 1.5)
        self.declare_parameter("a_lat_max", 3.5)
        self.declare_parameter("a_brake", 2.0)
        self.declare_parameter("a_accel", 2.0)
        self.declare_parameter("throttle_kp", 0.4)
        self.declare_parameter("throttle_ki", 0.6)
        self.declare_parameter("throttle_kff", 0.15)
        self.declare_parameter("throttle_min", -0.5)
        self.declare_parameter("throttle_max", 0.4)
        self.declare_parameter("speed_preview_s", 0.3)

        self._car = self.get_parameter("car").value
        self._horizon_steps = int(self.get_parameter("horizon_steps").value)
        self._mpc_dt = float(self.get_parameter("mpc_dt").value)
        self._q_lateral = float(self.get_parameter("q_lateral").value)
        self._q_heading = float(self.get_parameter("q_heading").value)
        self._q_terminal = float(self.get_parameter("q_terminal").value)
        self._r_curvature = float(self.get_parameter("r_curvature").value)
        self._r_curvature_rate = float(self.get_parameter("r_curvature_rate").value)
        self._steer_cmd_limit = float(self.get_parameter("steer_cmd_limit").value)
        self._steer_rate_limit = float(self.get_parameter("steer_rate_limit").value)
        self._v_min = float(self.get_parameter("v_min").value)
        self._v_max = float(self.get_parameter("v_max").value)
        self._a_lat_max = float(self.get_parameter("a_lat_max").value)
        self._a_brake = float(self.get_parameter("a_brake").value)
        self._a_accel = float(self.get_parameter("a_accel").value)
        self._throttle_kp = float(self.get_parameter("throttle_kp").value)
        self._throttle_ki = float(self.get_parameter("throttle_ki").value)
        self._throttle_kff = float(self.get_parameter("throttle_kff").value)
        self._throttle_min = float(self.get_parameter("throttle_min").value)
        self._throttle_max = float(self.get_parameter("throttle_max").value)
        self._speed_preview_s = float(self.get_parameter("speed_preview_s").value)


def main(args=None) -> None:
    """Spin the MPC node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = MpcNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
