"""
Port of the predecessor's ``standard_vehicle.Vehicle`` (WS 25/26).

A kinematic bicycle model, like our ``kinematic.KinematicBicycleModel``, but with
a different longitudinal scheme: throttle commands a *target speed*
(``throttle * max_speed``) that a proportional controller chases with separate
acceleration / braking rates plus constant rolling friction — and it supports
reverse driving (capped at ``reverse_speed_fraction * max_speed``).

Adaptation from the original (see ``old/README.md`` for the full list):

* Interface: implements ``VehicleModel.step(state, action, dt)`` instead of the
  mutable ``Controllable`` object; the heading is an angle, not a forward vector.
* Units: the original worked in pixels (55 px car, 49.5 px wheelbase, 800 px/s
  max speed). Defaults are converted to SI with the scale implied by the real
  car's 0.256 m wheelbase: 1 px = 0.256 / 49.5 m ≈ 5.17 mm. Original pixel
  values are noted next to each field.
* Bug fixes: removed a leftover debug ``print``; the forward-vector
  renormalization that was computed but never applied is moot with an angle
  representation; friction is no longer double-counted while braking.
"""

from dataclasses import dataclass
import math

from ..base import VehicleAction, VehicleModel, VehicleState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sign(value: float) -> float:
    return math.copysign(1.0, value) if value else 0.0


@dataclass
class OldStandardVehicleModel(VehicleModel):
    """
    Predecessor kinematic bicycle: target-speed throttle controller, no slip.

    ``vy`` is always 0 — the car only ever moves along its heading, so it cannot
    drift. Stateless between steps (unlike the slip-angle model).
    """

    max_speed: float = 4.14  # m/s (800 px/s) — top forward speed
    reverse_speed_fraction: float = 0.3  # reverse cap as a fraction of max_speed
    accel_rate: float = 4.14  # m/s² (800 px/s²) — forward acceleration authority
    brake_rate: float = 5.17  # m/s² (1000 px/s²) — braking authority
    friction: float = 0.41  # m/s² (80 px/s²) — constant rolling friction
    wheelbase: float = 0.256  # m (49.5 px) — front-to-rear axle distance
    max_steer_angle: float = 0.5236  # rad (30°) — full-lock road-wheel angle
    coast_speed_eps: float = 0.005  # m/s (1 px/s) — below this, coasting friction is off
    stop_speed_eps: float = 0.0026  # m/s (0.5 px/s) — speeds below this snap to 0

    @property
    def max_reverse_speed(self) -> float:
        """Top reverse speed [m/s]."""
        return self.reverse_speed_fraction * self.max_speed

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        steer_angle = _clamp(action.steering, -1.0, 1.0) * self.max_steer_angle
        throttle = _clamp(action.throttle, -1.0, 1.0)
        speed = state.vx  # signed forward speed; the model is slip-free, vy is ignored

        # Throttle commands a target speed; the gap sets the control effort.
        delta_speed = 0.0
        force_scale = 0.0
        if throttle > 0:
            delta_speed = self.max_speed * throttle - speed
            force_scale = abs(delta_speed) / self.max_speed
        elif throttle < 0:
            delta_speed = self.max_reverse_speed * throttle - speed
            force_scale = abs(delta_speed) / self.max_reverse_speed

        if abs(throttle) < 0.01:
            # Coasting: only rolling friction, and none once (almost) stopped.
            accel = -_sign(speed) * self.friction if abs(speed) > self.coast_speed_eps else 0.0
        elif throttle < 0 < speed:
            # Braking out of forward motion.
            accel = -self.brake_rate * abs(throttle) - self.friction
        else:
            if delta_speed > 0:  # need to speed up (toward target)
                accel = self.accel_rate * force_scale
            elif delta_speed < 0:  # need to slow down
                accel = -self.brake_rate * force_scale
            else:
                accel = 0.0
            accel -= _sign(speed) * self.friction

        speed = _clamp(speed + accel * dt, -self.max_reverse_speed, self.max_speed)
        if abs(speed) < self.stop_speed_eps:
            speed = 0.0

        # Kinematic bicycle yaw; as in the original, the car then moves along the
        # freshly rotated heading.
        yaw_rate = (speed / self.wheelbase) * math.tan(steer_angle)
        heading = state.heading + yaw_rate * dt
        heading = math.atan2(math.sin(heading), math.cos(heading))
        x = state.x + math.cos(heading) * speed * dt
        y = state.y + math.sin(heading) * speed * dt

        return VehicleState(x=x, y=y, heading=heading, vx=speed, vy=0.0, yaw_rate=yaw_rate)
