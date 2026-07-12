"""
Port of the predecessor's ``slip_angle_vehicle.SlipAngleVehicle`` (WS 25/26).

An arcade drift model built around a scripted state machine: normal driving is a
kinematic bicycle; hard steering at speed with enough throttle-induced ``slip``
enters a *drifting* mode where the heading yaws away from the (momentum-carrying)
velocity vector, steering/throttle modulate the slip angle, and slip angles past
90° trigger a *spin-out* routine. A separate *reverse* mode kicks in when
backing up from (near) standstill.

Adaptation from the original (see ``old/README.md`` for the full list):

* Interface: implements ``VehicleModel.step(state, action, dt)``. Pose and
  velocity live in :class:`VehicleState` (body-frame ``vx``/``vy``); the drift
  state machine (``slip``, drifting, spin-out, reverse flags) stays *inside* the
  model instance and carries across steps — like the arcade model's ``_slip``,
  it is only meaningful when one instance is stepped sequentially.
* Units: the original worked in pixels (55 px car, 41.25 px wheelbase,
  800 px/s max speed) — defaults are converted to SI with the scale implied by
  the real car's 0.256 m wheelbase: 1 px = 0.256 / 41.25 m ≈ 6.21 mm. Original
  pixel values are noted next to each field.
* Timestep: several original updates were per-*frame* (no ``dt``) at the game's
  fixed 8 ms tick — the drift slip-angle nudges and the whole spin-out routine.
  They are converted to per-*second* rates (divided by 0.008 s), so at
  ``dt = 0.008`` this port matches the original step for step.
* The original never enforced ``max_speed``; top speed emerges from the
  engine-force / drag balance (kept as-is, ≈ 8 m/s at full throttle and slip).
"""

from dataclasses import dataclass, field
import math

import numpy as np

from ..base import VehicleAction, VehicleModel, VehicleState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class OldSlipAngleVehicleModel(VehicleModel):
    """
    Predecessor arcade drift model: bicycle steering + scripted drift/spin-out.

    The world-frame velocity vector is the source of truth (converted from/to
    the body frame at the step boundaries). Throttle builds ``slip``; steering
    hard (> ``drift_steer_threshold``) at speed with ``slip`` > 0.4 breaks the
    car into the drifting mode, where the velocity direction lags the heading.
    """

    # --- longitudinal ---------------------------------------------------------
    mass: float = 4.0  # original unitless "mass" divisor
    engine_force: float = 124.0  # (20000 px·mass/s²) — full-throttle drive force
    drag: float = 8.0  # speed-proportional drag force coefficient (scale-free)
    wheelbase: float = 0.256  # m (41.25 px) — front-to-rear axle distance
    max_steer_angle: float = 0.5236  # rad (30°) — full-lock road-wheel angle
    # --- throttle -> slip (as in the original slip_update) ---------------------
    slip_lowerbound: float = 0.3  # throttle below this builds no slip
    slip_upperbound: float = 0.9  # throttle at/above this targets slip_max
    slip_max: float = 0.95
    slip_decay_rate: float = 1.0  # 1/s — exponential slip decay on lift-off
    # --- drift state machine ----------------------------------------------------
    drift_steer_threshold: float = 0.349  # rad (20°) — min steer angle to break out
    drift_entry_speed: float = 1.86  # m/s (300 px/s) — min speed to start drifting
    drift_exit_speed: float = 1.09  # m/s (175 px/s) — below this the drift ends
    drift_exit_slip: float = 0.3  # drift also ends when slip decays below this
    min_drift_angle: float = 0.1  # rad — slip angle is held at least this deep
    countersteer_rate: float = 12.5  # 1/s (0.1 per 8 ms frame) — steer -> slip-angle
    throttle_slip_rate: float = 6.25  # 1/s (0.05 per 8 ms frame) — throttle -> slip-angle
    drift_yaw_gain: float = 0.2  # fraction of slip angle fed to bicycle yaw
    drift_yaw_deadband: float = 0.2  # rad/s — smaller drift yaw rates are ignored
    # --- spin-out (per-frame in the original, converted to rates) --------------
    spinout_yaw_per_speed: float = 8.79  # (rad/s)/(m/s) (speed_px/40 ° per frame)
    spinout_decel: float = 2.33  # m/s² (3 px per frame)
    spinout_stop_speed: float = 0.031  # m/s (5 px/s) — spin ends below this
    # --- reverse ----------------------------------------------------------------
    max_reverse_speed: float = 1.24  # m/s (200 px/s) — reverse speed at full throttle
    reverse_enter_speed: float = 0.062  # m/s (10 px/s) — max speed to engage reverse
    reverse_exit_speed: float = 0.006  # m/s (1 px/s) — max speed to leave reverse

    # Runtime state (not constructor arguments): the drift state machine carries
    # across steps, so one instance must be stepped sequentially (as in the
    # interactive sim / a rollout).
    _slip: float = field(init=False, default=0.0)
    _drifting: bool = field(init=False, default=False)
    _drifting_sign: float = field(init=False, default=0.0)
    _spinning_out: bool = field(init=False, default=False)
    _slip_sign: float = field(init=False, default=1.0)
    _reverse: bool = field(init=False, default=False)

    # This port predates the ESC input stage and keeps the original game's
    # brake/reverse behaviour, so it overrides step() and bypasses the ESC.
    def _step_motor(self, state, steering, motor, dt):
        raise NotImplementedError("old models override step() directly")

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        steer_angle = _clamp(action.steering, -1.0, 1.0) * self.max_steer_angle
        throttle = _clamp(action.throttle, -1.0, 1.0)

        heading = state.heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # Body frame (ISO 8855: vy positive left) -> world-frame velocity vector.
        vel = np.array(
            [
                state.vx * cos_h - state.vy * sin_h,
                state.vx * sin_h + state.vy * cos_h,
            ]
        )
        speed = float(np.linalg.norm(vel))

        # Throttle-driven slip: instant rise, exponential decay (slip_update).
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

        # Slip trims engine force; drag always opposes motion (forces_update).
        accel = (1.0 - 0.5 * self._slip) * throttle * self.engine_force / self.mass
        drag_accel = speed * self.drag / self.mass

        # Hard steering at speed with enough slip breaks the car into a drift.
        if (
            not self._drifting
            and abs(steer_angle) > self.drift_steer_threshold
            and self._slip > 0.4
            and speed > self.drift_entry_speed
        ):
            self._drifting = True
            self._drifting_sign = math.copysign(1.0, steer_angle)

        if self._spinning_out:
            # Spin-out routine: yaw wildly in the slip direction while momentum
            # carries the car on and bleeds off, until it (almost) stops.
            accel = 0.0
            if speed > self.spinout_stop_speed:
                heading -= self._slip_sign * self.spinout_yaw_per_speed * speed * dt
                direction = vel / speed
                speed = max(0.0, speed - self.spinout_decel * dt)
            else:
                self._spinning_out = False
                direction = np.zeros(2)
                speed = 0.0
        elif self._drifting:
            slip_angle = _wrap(math.atan2(vel[1], vel[0]) - heading)
            # Steering further into the drift deepens the slip angle (smaller
            # radius); countersteering shallows it. Throttle past 0.5 digs in too.
            throttle_change = max(0.0, throttle - 0.5)
            slip_change_rate = self.countersteer_rate * steer_angle
            if self._drifting_sign > 0:
                slip_change_rate += self.throttle_slip_rate * throttle_change
            else:
                slip_change_rate -= self.throttle_slip_rate * throttle_change
            slip_change = slip_change_rate * dt
            slip_angle -= slip_change

            # Hold the drift at least min_drift_angle deep (on the drift's side).
            if self._drifting_sign > 0:
                slip_angle = min(slip_angle, self.min_drift_angle)
            else:
                slip_angle = max(slip_angle, -self.min_drift_angle)

            self._slip_sign = math.copysign(1.0, slip_angle)
            if abs(slip_angle) > math.pi / 2.0:
                self._spinning_out = True
            # Countersteering past neutral flips which side the drift is on.
            if self._drifting_sign > 0 and steer_angle < 0 and slip_angle > 0:
                self._drifting_sign *= -1.0
            if self._drifting_sign < 0 and steer_angle > 0 and slip_angle < 0:
                self._drifting_sign *= -1.0
            heading += slip_change

            drift_yaw_rate = (speed / self.wheelbase) * math.tan(self.drift_yaw_gain * slip_angle)
            if abs(drift_yaw_rate) > self.drift_yaw_deadband:
                # Rotate heading and velocity together (slip angle preserved).
                dtheta = -drift_yaw_rate * dt
                cos_d = math.cos(dtheta)
                sin_d = math.sin(dtheta)
                heading += dtheta
                vel = np.array(
                    [
                        vel[0] * cos_d - vel[1] * sin_d,
                        vel[0] * sin_d + vel[1] * cos_d,
                    ]
                )
                direction = vel / speed if speed > 0.0 else np.zeros(2)
            else:
                # As in the original: near standstill the "direction" is the raw
                # (tiny) velocity vector, which effectively stops the car.
                direction = vel / speed if speed > 1e-3 else vel.copy()
        else:
            # Normal driving: kinematic bicycle (handle_slipless_steering).
            if abs(steer_angle) > 1e-4:
                yaw = (speed / self.wheelbase) * math.tan(steer_angle)
                if self._reverse:
                    yaw = -yaw
            else:
                yaw = 0.0
            heading += yaw * dt
            direction = np.array([math.cos(heading), math.sin(heading)])

        if self._slip < self.drift_exit_slip or speed < self.drift_exit_speed:
            self._drifting = False

        if self._reverse:
            self._drifting = False
            direction = -np.array([math.cos(heading), math.sin(heading)])
            vel = direction * max(self.max_reverse_speed * -throttle, 0.0)
        else:
            vel = direction * (speed + accel * dt)
        if speed < self.reverse_enter_speed and throttle < 0:
            self._reverse = True
        if speed < self.reverse_exit_speed and throttle > 0 and self._reverse:
            self._reverse = False

        vel = vel - direction * drag_accel * dt
        x = state.x + float(vel[0]) * dt
        y = state.y + float(vel[1]) * dt

        heading = _wrap(heading)
        # Net yaw over the step (drift + spin-out included); assumes < pi per step.
        yaw_rate = _wrap(heading - state.heading) / dt if dt > 0.0 else 0.0
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        vx = float(vel[0]) * cos_h + float(vel[1]) * sin_h
        vy = -float(vel[0]) * sin_h + float(vel[1]) * cos_h
        return VehicleState(x=x, y=y, heading=heading, vx=vx, vy=vy, yaw_rate=yaw_rate)
