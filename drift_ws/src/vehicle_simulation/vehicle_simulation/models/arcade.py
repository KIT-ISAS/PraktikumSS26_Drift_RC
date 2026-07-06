"""
Arcade drift model: a game-like car that drives cleanly and slides when floored.

This is a *demo* model, not a physically accurate one -- it is meant for driving
the simulation by hand (keyboard / gamepad / replayed bag) and watching it in
Foxglove. It is a slip-state arcade model in the spirit of a top-down racing
game, expressed in our SI + ISO 8855 body frame (``vx`` forward, ``vy`` left,
yaw +CCW; positive steering = left):

* **Normal driving = kinematic bicycle.** Yaw rate is ``vx / wheelbase *
  tan(steer_angle)``, so the turn *radius* is ``wheelbase / tan(steer_angle)`` --
  independent of speed. The car carves the same line whether you crawl or fly,
  which is what makes it feel predictable (unlike a fixed-max-yaw model, whose
  turns balloon with speed). At ``vx = 0`` it cannot turn; in reverse the yaw
  flips sign, so backing up steers intuitively.
* **Throttle induces slip.** A throttle-driven ``slip`` scalar rises toward
  ``slip_max`` above ``slip_lowerbound`` and decays when you lift off. Slip both
  trims engine force and *loosens lateral grip*, so flooring it out of a corner
  breaks the rear loose into a drift; easing the throttle hooks grip back up and
  the slide recovers. Countersteering works for free: steering rotates the body
  while the velocity lags, and grip bleeds the leftover lateral velocity away.

``grip`` is the headline knob: high grip behaves almost like the no-slip
kinematic model; low grip gives long, loose slides.
"""

from dataclasses import dataclass, field
import math

from .base import VehicleAction, VehicleModel, VehicleState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ArcadeDriftModel(VehicleModel):
    """
    Body-frame arcade car: kinematic-bicycle steering + throttle-driven drift.

    The state's body-frame velocity ``(vx, vy)`` is the source of truth. Each
    step: the throttle updates ``slip`` and the forward speed, the bicycle model
    yaws the car, the velocity vector is rotated into the new body frame (this is
    what creates lateral velocity), and grip -- slackened by ``slip`` -- bleeds
    that lateral velocity away.
    """

    max_speed: float = 5.0  # m/s — top forward speed
    max_reverse_speed: float = 1.5  # m/s — top reverse speed
    max_accel: float = 8.0  # m/s² — engine authority at full throttle
    drag: float = 0.8  # 1/s — speed-proportional rolling drag
    wheelbase: float = 0.25  # m — front-to-rear axle distance (bicycle steering)
    max_steer_angle: float = 1.309  # rad (75°) — full-lock road-wheel angle
    # Throttle -> slip: below `lower` no slip builds; at/above `upper` it targets
    # `slip_max`. Slip rises instantly and decays at `slip_decay_rate` on lift-off.
    slip_lowerbound: float = 0.3
    slip_upperbound: float = 0.9
    slip_max: float = 0.95
    slip_decay_rate: float = 3.0  # 1/s — how fast grip returns when you ease off
    grip: float = 8.0  # 1/s — lateral grip with no slip (high = sticks, low = slides)
    drift_grip: float = 0.6  # 1/s — residual lateral grip at full slip (the slide)

    # Runtime state (not a constructor argument): the throttle-driven slip carries
    # across steps. Only meaningful when one model instance is stepped in sequence
    # (the interactive sim / a rollout), which is exactly how the arcade model is used.
    _slip: float = field(init=False, default=0.0)

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        steer = _clamp(action.steering, -1.0, 1.0)
        throttle = _clamp(action.throttle, -1.0, 1.0)

        vx = state.vx
        vy = state.vy

        # Throttle-driven slip: instant rise, exponential decay back to target.
        slip_factor = _clamp(
            (throttle - self.slip_lowerbound) / (self.slip_upperbound - self.slip_lowerbound),
            0.0,
            1.0,
        )
        target_slip = slip_factor * self.slip_max
        if target_slip > self._slip:
            self._slip = target_slip
        else:
            self._slip += (target_slip - self._slip) * (1.0 - math.exp(-self.slip_decay_rate * dt))
        slip = self._slip

        # Longitudinal: slip trims engine force, drag always pulls toward zero.
        vx += (1.0 - 0.5 * slip) * throttle * self.max_accel * dt
        vx -= self.drag * vx * dt
        vx = _clamp(vx, -self.max_reverse_speed, self.max_speed)

        # Steering = kinematic bicycle: constant turn radius, no turning while
        # parked, sign flips in reverse (vx < 0) so backing up steers naturally.
        steer_angle = steer * self.max_steer_angle
        yaw_rate = (vx / self.wheelbase) * math.tan(steer_angle)
        dtheta = yaw_rate * dt

        # Rotate the velocity into the freshly-yawed body frame. Momentum is
        # conserved here; the mismatch between heading and velocity is the slide.
        cos_d = math.cos(-dtheta)
        sin_d = math.sin(-dtheta)
        vx, vy = vx * cos_d - vy * sin_d, vx * sin_d + vy * cos_d

        # Grip pulls lateral velocity back toward zero; slip slackens it so a
        # floored corner drifts and lifting off recovers.
        grip_eff = self.grip * (1.0 - slip) + self.drift_grip * slip
        vy *= max(0.0, 1.0 - grip_eff * dt)

        # Integrate the pose in the world frame using the new heading.
        heading = math.atan2(math.sin(state.heading + dtheta), math.cos(state.heading + dtheta))
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        x = state.x + (vx * cos_h - vy * sin_h) * dt
        y = state.y + (vx * sin_h + vy * cos_h) * dt

        return VehicleState(x=x, y=y, heading=heading, vx=vx, vy=vy, yaw_rate=yaw_rate)
