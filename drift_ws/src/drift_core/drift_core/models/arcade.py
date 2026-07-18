"""
Arcade drift model: a game-like car that drives cleanly and slides when floored.

This is a *demo* model, not a physically accurate one -- it is meant for driving
the simulation by hand (keyboard / gamepad / replayed bag) and watching it in
Foxglove. It is a slip-state arcade model in the spirit of a top-down racing
game, expressed in our SI + ISO 8855 body frame (``vx`` forward, ``vy`` left,
yaw +CCW; positive steering = left):

* **Normal driving = kinematic bicycle, until the front tires saturate.** Yaw
  rate is ``vx / wheelbase * tan(steer_angle)``, so the turn *radius* is
  ``wheelbase / tan(steer_angle)`` -- independent of speed. But the front axle
  can only hold so much lateral acceleration (``a_lat = yaw_rate * vx``): past
  ``front_grip_limit`` the front washes out and the yaw rate is capped, so
  cranking the wheel at speed *understeers* (the radius grows) instead of
  snapping the car around. At ``vx = 0`` it cannot turn; in reverse the yaw
  flips sign, so backing up steers intuitively.
* **Throttle induces slip.** Above ``slip_lowerbound`` the rear tires start to
  spin: a throttle-driven ``slip`` scalar rises toward ``slip_max`` and decays
  when you lift off. Spinning tires shed most of their drive force
  (``slip_accel_loss``) *and loosen lateral grip*, so flooring it breaks the
  rear loose into a slow, sustained drift rather than launching the car; easing
  the throttle hooks grip back up and the slide recovers. Countersteering works
  for free: steering rotates the body while the velocity lags, and grip bleeds
  the leftover lateral velocity away.
* **Brake vs reverse is decided upstream, by the ESC input stage** (see
  :mod:`drift_core.models.esc` for the measured lockout behaviour).
  This model only supplies the physics of the resolved command: ``drive`` is
  engine force (backwards in reverse, capped at ``max_reverse_speed``);
  ``brake`` decelerates toward standstill and never through it.

``grip`` is the headline knob: high grip behaves almost like the no-slip
kinematic model; low grip gives long, loose slides. The defaults are tuned to
the real car: an RC car on plastic drift tires on a slick industrial floor,
i.e. low friction all around (``front_grip_limit`` / ``rear_grip_limit``
around 0.4 g).
"""

from dataclasses import dataclass, field
import math

from .base import VehicleModel, VehicleState
from .esc import MotorCommand


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

    max_speed: float = 3.0  # m/s — top forward speed
    max_reverse_speed: float = 1.5  # m/s — top reverse speed
    max_accel: float = 4.5  # m/s² — engine authority at full throttle, traction-limited
    drag: float = 0.7  # 1/s — speed-proportional rolling drag
    wheelbase: float = 0.25  # m — front-to-rear axle distance (bicycle steering)
    max_steer_angle: float = 1.396  # rad (80°) — full-lock road-wheel angle
    # Friction limits: max lateral acceleration each axle can hold (≈ µ·g; drift
    # tires on a slick floor → µ ≈ 0.4). The front limit caps the yaw rate
    # (understeer); the rear limit caps how hard grip can pull `vy` back.
    front_grip_limit: float = 4.0  # m/s²
    rear_grip_limit: float = 4.0  # m/s²
    # Throttle -> slip: below `lower` no slip builds (the tires hold); at/above
    # `upper` it targets `slip_max`. Slip rises instantly and decays at
    # `slip_decay_rate` on lift-off. Spinning tires deliver only kinetic
    # friction: engine force is trimmed by `slip_accel_loss * slip`.
    slip_lowerbound: float = 0.4
    slip_upperbound: float = 0.9
    slip_max: float = 0.95
    slip_decay_rate: float = 3.0  # 1/s — how fast grip returns when you ease off
    slip_accel_loss: float = 0.7  # fraction of engine force lost at full slip
    grip: float = 5.0  # 1/s — lateral grip with no slip (high = sticks, low = slides)
    drift_grip: float = 0.6  # 1/s — residual lateral grip at full slip (the slide)

    # Runtime state (not a constructor argument): the throttle-driven slip carries
    # across steps. Only meaningful when one model instance is stepped in sequence
    # (the interactive sim / a rollout), which is exactly how the arcade model is used.
    _slip: float = field(init=False, default=0.0)

    def _step_motor(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        steer = _clamp(steering, -1.0, 1.0)

        vx = state.vx
        vy = state.vy

        # Drive-driven slip: instant rise, exponential decay back to target.
        slip_factor = _clamp(
            (motor.drive - self.slip_lowerbound) / (self.slip_upperbound - self.slip_lowerbound),
            0.0,
            1.0,
        )
        target_slip = slip_factor * self.slip_max
        if target_slip > self._slip:
            self._slip = target_slip
        else:
            self._slip += (target_slip - self._slip) * (1.0 - math.exp(-self.slip_decay_rate * dt))
        slip = self._slip

        # Longitudinal: spinning tires shed drive force, drag pulls toward zero.
        # Drive and brake are mutually exclusive (the ESC emits one or the other):
        # drive is engine force (backwards in reverse), brake decelerates toward
        # standstill and never through it.
        vx += (1.0 - self.slip_accel_loss * slip) * motor.drive * self.max_accel * dt
        if motor.brake > 0.0:
            decel = motor.brake * self.max_accel * dt
            vx -= math.copysign(min(decel, abs(vx)), vx)
        vx -= self.drag * vx * dt
        vx = _clamp(vx, -self.max_reverse_speed, self.max_speed)

        # Steering = kinematic bicycle: constant turn radius, no turning while
        # parked, sign flips in reverse (vx < 0) so backing up steers naturally.
        steer_angle = steer * self.max_steer_angle
        yaw_rate = (vx / self.wheelbase) * math.tan(steer_angle)
        # ...but the front tires can only hold `front_grip_limit` of lateral
        # acceleration (a_lat = yaw_rate * vx). Past that the front washes out
        # and the car understeers: yaw saturates, the turn radius grows.
        if abs(vx) > 1e-6:
            yaw_limit = self.front_grip_limit / abs(vx)
            yaw_rate = _clamp(yaw_rate, -yaw_limit, yaw_limit)
        dtheta = yaw_rate * dt

        # Rotate the velocity into the freshly-yawed body frame. Momentum is
        # conserved here; the mismatch between heading and velocity is the slide.
        cos_d = math.cos(-dtheta)
        sin_d = math.sin(-dtheta)
        vx, vy = vx * cos_d - vy * sin_d, vx * sin_d + vy * cos_d

        # Grip pulls lateral velocity back toward zero; slip slackens it so a
        # floored corner drifts and lifting off recovers. The pull is friction-
        # limited (`rear_grip_limit`), so a big slide bleeds off slowly instead
        # of the tires generating arbitrarily large lateral force.
        grip_eff = self.grip * (1.0 - slip) + self.drift_grip * slip
        lat_decel = min(grip_eff * abs(vy), self.rear_grip_limit)
        vy -= math.copysign(min(lat_decel * dt, abs(vy)), vy)

        # Integrate the pose in the world frame using the new heading.
        heading = math.atan2(math.sin(state.heading + dtheta), math.cos(state.heading + dtheta))
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        x = state.x + (vx * cos_h - vy * sin_h) * dt
        y = state.y + (vx * sin_h + vy * cos_h) * dt

        return VehicleState(x=x, y=y, heading=heading, vx=vx, vy=vy, yaw_rate=yaw_rate)
