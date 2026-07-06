"""Kinematic bicycle model: simple, game-like, no tire slip."""

from dataclasses import dataclass
import math

import numpy as np

from .base import VehicleAction, VehicleModel, VehicleState


@dataclass
class KinematicBicycleModel(VehicleModel):
    """
    Rear-wheel-drive kinematic bicycle model.

    Implements pure kinematic equations (no tire slip). Suitable for initial
    RL prototyping where physical accuracy is less important than simulation speed.

    Stochastic variant: set noise_vx / noise_yaw_rate > 0 and call step_stochastic().
    """

    max_accel: float = 3.0  # m/s²
    max_speed: float = 5.0  # m/s
    drag: float = 1.5  # 1/s — linear speed-proportional drag
    wheelbase: float = 0.25  # m — distance between axles
    max_steer_angle: float = 1.309  # rad (75°) — physical steering limit
    noise_vx: float = 0.0  # m/s std — forward-speed noise for step_stochastic
    noise_yaw_rate: float = 0.0  # rad/s std — yaw-rate noise for step_stochastic

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        steer_rad = action.steering * self.max_steer_angle
        speed = state.speed

        # Linear acceleration model with drag
        accel = action.throttle * self.max_accel - speed * self.drag
        speed = max(0.0, min(self.max_speed, speed + accel * dt))

        # Yaw rate from bicycle geometry
        yaw_rate = (speed / self.wheelbase * math.tan(steer_rad)) if speed > 1e-4 else 0.0

        # Mid-point integration for smoother heading update
        heading = state.heading + yaw_rate * dt
        mid_heading = state.heading + yaw_rate * dt / 2.0
        x = state.x + speed * math.cos(mid_heading) * dt
        y = state.y + speed * math.sin(mid_heading) * dt

        return VehicleState(x=x, y=y, heading=heading, vx=speed, vy=0.0, yaw_rate=yaw_rate)

    def step_stochastic(
        self, state: VehicleState, action: VehicleAction, dt: float
    ) -> VehicleState:
        next_state = self.step(state, action, dt)
        if self.noise_vx > 0.0:
            next_state.vx = max(0.0, next_state.vx + float(np.random.normal(0.0, self.noise_vx)))
        if self.noise_yaw_rate > 0.0:
            next_state.yaw_rate += float(np.random.normal(0.0, self.noise_yaw_rate))
        return next_state
