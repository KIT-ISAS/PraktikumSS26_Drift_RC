"""
Arcade drift model: a simple, game-like car that slides when you turn hard.

This is a *demo* model, not a physically accurate one. It is meant for showing
off the simulation (drive it with a controller / keyboard / replayed bag and
watch it in Foxglove). Drift "feel" comes from a single ``grip`` parameter:
turning rotates the car faster than its velocity vector follows, and low grip
lets that leftover sideways velocity persist for a moment -- the slide.
"""

from dataclasses import dataclass
import math

from .base import VehicleAction, VehicleModel, VehicleState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ArcadeDriftModel(VehicleModel):
    """
    Velocity-space arcade car with controllable slide.

    The state's body-frame velocity ``(vx, vy)`` is the source of truth. Each
    step: throttle/drag update the forward speed, steering yaws the car, the
    velocity vector is rotated into the new body frame (this is what creates
    lateral velocity), and ``grip`` bleeds that lateral velocity away. Lower
    ``grip`` => longer, looser slides; high ``grip`` behaves almost like the
    kinematic model.
    """

    max_speed: float = 5.0  # m/s — top forward speed
    max_reverse_speed: float = 1.5  # m/s — top reverse speed
    max_accel: float = 4.0  # m/s² — throttle authority
    drag: float = 0.8  # 1/s — longitudinal (rolling) drag
    max_yaw_rate: float = 2.8  # rad/s — turn rate at full steer
    steer_speed_full: float = 1.2  # m/s — speed at which steering is fully effective
    grip: float = 2.5  # 1/s — how fast lateral velocity is killed (low = more drift)
    lateral_drag: float = 0.6  # 1/s — extra damping on lateral velocity

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        steer = _clamp(action.steering, -1.0, 1.0)
        throttle = _clamp(action.throttle, -1.0, 1.0)

        vx = state.vx
        vy = state.vy

        # Longitudinal: throttle accelerates, drag always pulls back toward zero.
        vx += throttle * self.max_accel * dt
        vx *= max(0.0, 1.0 - self.drag * dt)
        vx = _clamp(vx, -self.max_reverse_speed, self.max_speed)

        # Steering becomes effective with speed (can't turn while parked) and
        # follows the travel direction so reversing steers intuitively.
        if self.steer_speed_full > 0.0:
            speed_factor = min(1.0, abs(vx) / self.steer_speed_full)
        else:
            speed_factor = 1.0
        direction = 1.0 if vx >= 0.0 else -1.0
        yaw_rate = steer * self.max_yaw_rate * speed_factor * direction

        # Rotate the velocity vector into the freshly-yawed body frame. Momentum
        # is conserved here; the mismatch between heading and velocity is the slide.
        dtheta = yaw_rate * dt
        cos_d = math.cos(-dtheta)
        sin_d = math.sin(-dtheta)
        vx, vy = vx * cos_d - vy * sin_d, vx * sin_d + vy * cos_d

        # Grip + lateral drag pull the sideways velocity back toward zero.
        vy *= max(0.0, 1.0 - (self.grip + self.lateral_drag) * dt)

        # Integrate the pose in the world frame using the new heading.
        heading = math.atan2(math.sin(state.heading + dtheta), math.cos(state.heading + dtheta))
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        x = state.x + (vx * cos_h - vy * sin_h) * dt
        y = state.y + (vx * sin_h + vy * cos_h) * dt

        return VehicleState(x=x, y=y, heading=heading, vx=vx, vy=vy, yaw_rate=yaw_rate)
