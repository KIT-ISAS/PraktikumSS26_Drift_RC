"""
ESC + rear-wheel model: command→torque, motor droop, brake, and the wheel torque balance.

The car has no wheel-speed sensor, so ``omega_r`` is a latent integrated from this
torque balance (paper measures it directly). The chassis force balance and the
tire/slip-ratio coupling live in :mod:`.model`; this module is the stateless ESC/wheel
dynamics it composes, plus :class:`_WheelParams`, the model's learnable scalars.
"""

import math

import torch
import torch.nn as nn

from ..dynamic import _inv_softplus

_OMEGA_BRAKE_FADE = 5.0  # rad/s over which the brake torque reverses sign around omega = 0
# The ESC's brake actuation is not instantaneous: a 0.19 s recorded brake pump barely
# slows the real car while an instant-torque model bites at ~3 m/s^2. The drive torque
# already goes through the learnable tau_w lag; the brake gets its own *fixed* lag —
# 11.5 s of recorded braking cannot identify a time constant, so it is set, not fitted.
_TAU_BRAKE = 0.05  # s

# Relaxation used to settle the latent wheel speed at a window start (see init_latent).
# The inertia here is deliberately larger than the model's own: it only sets how fast the
# relaxation converges, not where it converges to, and a big one lets 40 cheap steps do it.
_WHEEL_SETTLE_STEPS, _WHEEL_SETTLE_DT, _WHEEL_SETTLE_INERTIA = 40, 0.01, 0.01


class _WheelParams(nn.Module):
    """
    The model's learnable scalars (softplus-positive, or sigmoid-bounded where weak).

    Most describe the ESC/drivetrain and steering actuator (``torque_gain``, ``rev_frac``,
    ``brake_torque``, ``iw``, ``tau_w``, ``steer_scale``, ``tau_s``); ``izz`` (yaw
    inertia), ``drag``, and the tire's ``f0`` ride along here because they are the same
    kind of thing — one checkpointed group of scalars — and keeping them together keeps
    the ``params_state`` checkpoint key stable.
    """

    _FIELDS = (
        "izz",
        "iw",
        "steer_scale",
        "torque_gain",
        "rev_frac",
        "brake_torque",
        "drag",
        "f0",
        "tau_s",
        "tau_w",
    )
    # Bounds carry the physics we know and stop the optimizer from taking the
    # degenerate shortcuts we have already watched it take:
    #   f0    — weights wheel spin (sigma²) against slip angle (tan²) in the combined
    #           slip; free, it collapsed to ~0.02 and switched the friction circle off.
    #   iw    — rear-axle inertia (wheels + reflected rotor), physically ~2e-4 kg m^2. It
    #           no longer has to be inflated for stability: `advance` integrates the wheel
    #           implicitly, so the stiff ~2 ms wheel-slip mode is fine at a 20-50 ms step.
    #   izz    — scaled by the physical plate estimate; a free izz fitted at 5.7x it
    #           in the v8f run, which sluggishes the yaw response into a car that
    #           will not rotate.
    #   rev_frac — the QuicRun ESC caps reverse force at 25% by default.
    #   steer_scale — fraction of the car's measured steering lock actually reached at
    #           full stick. The lock is eyeballed and it is the *servo-commanded* angle
    #           that matters, not what the knuckles allow, so the fit gets a band around
    #           the measurement rather than the number itself. Left free it collapsed to
    #           30 deg against a measured ~75 and the front tire curve absorbed the rest.
    # The motor's speed droop is NOT here: it comes from the car's kV and pack voltage
    # (`wheel_noload_speed_rad_s`), because it is a spec, not a mystery. Fitted, it came
    # out at 170 rad/s against a physical 396. With no droop at all a wheel spun up in a
    # donut freewheeled for ~20 s, holding the friction circle open, and the car drifted
    # until braked.
    #   torque_gain — must be able to overpower the rear tire, or wheelspin is impossible
    #           by construction. The data cannot identify it (2% of samples above 0.8
    #           throttle, nothing above 1.6 m/s), so physics sets the floor instead.
    _BOUNDED = {
        "f0": (0.2, 5.0),
        "iw": (1e-4, 2e-3),
        "steer_scale": (0.5, 1.1),
        "rev_frac": (0.05, 0.6),
        "brake_torque": (0.05, 1.0),
        "torque_gain": (0.08, 0.4),
        "tau_s": (0.02, 0.15),
        "tau_w": (0.03, 0.30),
    }

    def __init__(self, bounded: bool = True, **values: float) -> None:
        super().__init__()
        # Per-instance, so a run can be trained with the identifiability bands off
        # (`bounded=False`) to see where an unconstrained fit actually goes. Recorded in
        # the checkpoint, because the raw parameters mean different things either way.
        self._bounds = dict(self._BOUNDED) if bounded else {}
        for name in self._FIELDS:
            v = float(values[name])
            if name in self._bounds:
                lo, hi = self._bounds[name]
                t = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
                raw = math.log(t / (1.0 - t))
            else:
                raw = _inv_softplus(v)
            setattr(self, f"raw_{name}", nn.Parameter(torch.tensor(raw)))

    @classmethod
    def placeholder(cls, bounded: bool = True) -> "_WheelParams":
        values = {name: 1.0 for name in cls._FIELDS}
        values.update({name: 0.5 * (lo + hi) for name, (lo, hi) in cls._BOUNDED.items()})
        return cls(bounded=bounded, **values)

    def __getattr__(self, name: str):
        if name in self._FIELDS:
            raw = super().__getattr__(f"raw_{name}")
            bounds = self.__dict__.get("_bounds", self._BOUNDED)
            if name in bounds:
                lo, hi = bounds[name]
                return lo + (hi - lo) * torch.sigmoid(raw)
            return torch.nn.functional.softplus(raw)
        return super().__getattr__(name)


def torque_cmd(p: _WheelParams, drive: torch.Tensor) -> torch.Tensor:
    """Commanded wheel torque: the ESC caps reverse (rev_frac of forward)."""
    return p.torque_gain * (drive.clamp(min=0.0) + p.rev_frac * drive.clamp(max=0.0))


def wheel_net_torque(p, omega, tq, tb, fxr, rw, omega_noload):
    """
    Net torque on the rear wheel: drive - motor droop - tire reaction - brake.

    ``omega_dot = wheel_net_torque(...) / I_w``. The droop ``torque_gain/omega_noload·omega``
    is the motor's torque-speed line (back-EMF + drivetrain drag); the brake reverses sign
    smoothly across omega = 0 over ``_OMEGA_BRAKE_FADE``.
    """
    t_droop = (p.torque_gain / omega_noload) * omega
    t_brake = tb * torch.tanh(omega / _OMEGA_BRAKE_FADE)
    return tq - t_droop - rw * fxr - t_brake


def wheel_damping_k(p, omega, tb, dfxr_domega, rw, omega_noload):
    """
    Local damping ``k = -d(net)/d(omega) >= 0`` for the linearly-implicit wheel step.

    ``dfxr_domega`` is the finite-difference slope of the tire reaction (the caller has
    the tire model); the droop contributes a constant slope and the brake its own
    (its torque flips sign over +-_OMEGA_BRAKE_FADE, and tb/iw is ~1000 rad/s^2, so
    leaving it out made a braked wheel yo-yo through zero at 20-50 ms steps).
    """
    k = rw * dfxr_domega + p.torque_gain / omega_noload
    k = k + tb * (1.0 - torch.tanh(omega / _OMEGA_BRAKE_FADE) ** 2) / _OMEGA_BRAKE_FADE
    return k.clamp(min=0.0)


def wheel_implicit_step(omega, net, k, dt, inertia):
    """One backward-Euler step of the linearized wheel equation; stable for any dt."""
    return omega + dt * net / (inertia + dt * k)
