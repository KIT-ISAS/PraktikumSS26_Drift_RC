"""Kinematic bicycle model: simple, game-like, no tire slip."""

from dataclasses import dataclass
import math

import numpy as np

from .base import VehicleModel, VehicleState
from .esc import MotorCommand


@dataclass
class KinematicBicycleModel(VehicleModel):
    """
    Rear-wheel-drive kinematic bicycle model.

    Implements pure kinematic equations (no tire slip). Suitable for initial
    RL prototyping where physical accuracy is less important than simulation speed.

    Stochastic variant: set noise_vx / noise_yaw_rate > 0 and call step_stochastic().
    """

    max_accel: float = 3.0  # m/s²
    max_speed: float = 5.0  # m/s — top forward speed
    max_reverse_speed: float = 1.5  # m/s — top reverse speed
    drag: float = 1.5  # 1/s — linear speed-proportional drag
    wheelbase: float = 0.25  # m — distance between axles
    max_steer_angle: float = 1.309  # rad (75°) — physical steering limit
    noise_vx: float = 0.0  # m/s std — forward-speed noise for step_stochastic
    noise_yaw_rate: float = 0.0  # rad/s std — yaw-rate noise for step_stochastic

    def _step_motor(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        steer_rad = steering * self.max_steer_angle
        speed = state.vx  # signed forward speed; the model is slip-free, vy stays 0

        # Linear acceleration model with drag: drive is signed engine force
        # (negative = reverse), brake decelerates toward standstill, never through it.
        speed += (motor.drive * self.max_accel - speed * self.drag) * dt
        if motor.brake > 0.0:
            decel = motor.brake * self.max_accel * dt
            speed -= math.copysign(min(decel, abs(speed)), speed)
        speed = max(-self.max_reverse_speed, min(self.max_speed, speed))

        # Yaw rate from bicycle geometry; the sign of `speed` flips it in reverse,
        # so backing up steers naturally.
        yaw_rate = (speed / self.wheelbase * math.tan(steer_rad)) if abs(speed) > 1e-4 else 0.0

        # Mid-point integration for smoother heading update
        heading = state.heading + yaw_rate * dt
        mid_heading = state.heading + yaw_rate * dt / 2.0
        x = state.x + speed * math.cos(mid_heading) * dt
        y = state.y + speed * math.sin(mid_heading) * dt

        return VehicleState(x=x, y=y, heading=heading, vx=speed, vy=0.0, yaw_rate=yaw_rate)

    def _step_motor_stochastic(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        next_state = self._step_motor(state, steering, motor, dt)
        if self.noise_vx > 0.0:
            noisy = next_state.vx + float(np.random.normal(0.0, self.noise_vx))
            next_state.vx = max(-self.max_reverse_speed, min(self.max_speed, noisy))
        if self.noise_yaw_rate > 0.0:
            next_state.yaw_rate += float(np.random.normal(0.0, self.noise_yaw_rate))
        return next_state
