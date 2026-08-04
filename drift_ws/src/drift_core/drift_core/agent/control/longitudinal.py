"""
Longitudinal speed controller: feedforward + PI with anti-windup (ROS-free).

Tracks the :class:`~drift_core.agent.control.speed_profile.SpeedProfile` reference with
a throttle command in ``[-1, 1]``. Deliberately simple — the ESC/tire longitudinal
dynamics are poorly identified (wheelspin above ~40 % throttle on the real car),
so the lateral MPC is decoupled from it and this loop only has to keep the speed
roughly on the profile.
"""

from dataclasses import dataclass

__all__ = ["SpeedController", "SpeedControllerConfig"]


@dataclass(frozen=True)
class SpeedControllerConfig:
    """Gains and output limits for :class:`SpeedController`."""

    kp: float = 0.4  # throttle per m/s of speed error
    ki: float = 0.6  # throttle per m of integrated speed error
    kff: float = 0.15  # feedforward throttle per m/s of reference speed
    throttle_min: float = -0.5  # brake floor
    throttle_max: float = 0.4  # forward ceiling (wheelspin above ~40 % on the real car)


class SpeedController:
    """PI + feedforward throttle controller with conditional anti-windup."""

    def __init__(self, config: SpeedControllerConfig) -> None:
        """Store the gains and start with a clean integrator."""
        self._cfg = config
        self._integral = 0.0

    def reset(self) -> None:
        """Clear the integrator (call on start/stop/override transitions)."""
        self._integral = 0.0

    def step(self, v_ref: float, v_meas: float, dt: float) -> float:
        """Return the throttle command in ``[throttle_min, throttle_max]`` for one tick."""
        cfg = self._cfg
        error = v_ref - v_meas
        raw = cfg.kff * v_ref + cfg.kp * error + cfg.ki * (self._integral + error * dt)
        throttle = min(max(raw, cfg.throttle_min), cfg.throttle_max)
        # Conditional anti-windup: only integrate while the output is unsaturated
        # or the error is pushing back out of saturation.
        if (
            (throttle == raw)
            or (raw > cfg.throttle_max and error < 0.0)
            or (raw < cfg.throttle_min and error > 0.0)
        ):
            self._integral += error * dt
        return throttle
