"""
Physics-informed hybrid: a neural-tire bicycle model.

Dynamic bicycle structure with *learned tire curves* and first-order actuator
states. Architecture (after Djeumou et al., "Reference-Free Formula Drift with
Reinforcement Learning", arXiv:2410.20990): the single-track force balance is
fixed physics; the neural parts are small **constrained tire-force curves**
("ExpTanh" bases — odd, saturating, with an optional post-peak falloff), so
every learned function is a physically-shaped force law. Unlike an additive
neural residual, this structure cannot learn to cancel the physics and then
extrapolate arbitrarily in closed loop — the failure mode that made the
residual-based hybrid unstable outside the training distribution.

Two latent first-order actuator states are integrated along with the dynamic
state (both are real mechanisms, not modelling conveniences):

* ``delta`` — actual steering angle; the servo lags the command (``tau_s``).
* ``f_drive`` — drive-train force surrogate; the ESC's punch control ramps the
  start force (Hobbywing QuicRun 10BL120, "Start Mode" default level 7), so
  thrust lags the command (``tau_w``). Forward and reverse have separate gains:
  the ESC caps reverse at 25% force by default. ``f_drive`` also acts as the
  rear longitudinal-slip surrogate in the combined-slip tire curve (upgrade
  path: measured wheel speed from the sensored motor's Hall lines).

The rear tire uses one combined-slip curve ``Ftot = Phi_r(s)`` with
``s² = tan²(alpha_r) + (f0·f_drive)²``, decomposed along the slip direction —
a *smooth* friction circle: as drive grows, lateral force fades along the
learned curve instead of collapsing at a hard clamp (which caused yaw jitter
at full throttle in the analytic model). Deterministic; canonical dynamic
state ``[vx, vy, yaw_rate]`` plus ``latent_dim=2``.
"""

import math

import torch
import torch.nn as nn

from ..car_parameters import load_car_parameters
from .base import Prediction, TrainableModel
from .dynamic import _BRAKE_FADE_SPEED, _VX_MIN, _inv_softplus

_EPS = 1e-6
_ROLL_FADE_SPEED = 0.1  # m/s rolling speed over which slip-angle tire forces fade in
_SCRUB_DAMPING = 8.0  # 1/s decay of lateral/yaw motion from tire scrub when not rolling


class _ExpTanhCurve(nn.Module):
    """
    Constrained tire curve ``Phi(s) = sum_i a_i tanh(b_i s)·(eps + (1-eps) e^(-c_i s²))``.

    All coefficients are positive (softplus), so Phi is odd, zero at zero slip
    and bounded. The exp term expresses the post-peak grip falloff of a sliding
    tire, but only down to the **sliding-friction floor** ``eps`` (sigmoid-bounded
    to [0.3, 1]): kinetic friction is lower than the peak, never zero, and
    letting the force decay to zero lets training kill the force at large slip
    entirely (the car then plows straight at full lock instead of drifting). The
    first basis term is initialized to the analytic
    ``peak·tanh(stiffness/peak·s)`` curve; the remaining terms start near zero.
    """

    def __init__(self, peak: float, stiffness: float, n_terms: int = 4) -> None:
        super().__init__()
        raw_a = [_inv_softplus(peak)] + [_inv_softplus(0.05 * peak)] * (n_terms - 1)
        raw_b = [_inv_softplus(stiffness / peak)] + [
            _inv_softplus(0.5 * (i + 1) * stiffness / peak) for i in range(n_terms - 1)
        ]
        self.raw_a = nn.Parameter(torch.tensor(raw_a))
        self.raw_b = nn.Parameter(torch.tensor(raw_b))
        # exp-decay starts tiny: the curve begins as a plain saturating tanh.
        self.raw_c = nn.Parameter(torch.full((n_terms,), -4.0))
        self.raw_eps = nn.Parameter(torch.tensor(1.5))  # sigmoid(1.5) -> eps ≈ 0.86

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.softplus(self.raw_a)
        b = torch.nn.functional.softplus(self.raw_b)
        c = torch.nn.functional.softplus(self.raw_c)
        eps = 0.3 + 0.7 * torch.sigmoid(self.raw_eps)
        su = s.unsqueeze(-1)
        decay = eps + (1.0 - eps) * torch.exp(-c * su**2)
        return (a * torch.tanh(b * su) * decay).sum(-1)


class _HybridParams(nn.Module):
    """Learnable scalars of the neural-tire bicycle (softplus-positive or bounded)."""

    _FIELDS = (
        "izz",
        "steer_gain",
        "fwd_gain",
        "rev_gain",
        "brake_gain",
        "drag",
        "f0",
        "tau_s",
        "tau_w",
    )
    # Actuator time constants are sigmoid-bounded to physically plausible ranges
    # (RC servo ~0.02-0.15 s; ESC punch/spool-up ~0.03-0.3 s). Left unbounded,
    # training drifts them slow — a low-pass on the command that trades transient
    # fidelity for smoother average fits and makes the sim feel sluggish.
    _BOUNDED = {"tau_s": (0.02, 0.15), "tau_w": (0.03, 0.30)}

    def __init__(self, **values: float) -> None:
        super().__init__()
        for name in self._FIELDS:
            v = float(values[name])
            if name in self._BOUNDED:
                lo, hi = self._BOUNDED[name]
                t = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
                raw = math.log(t / (1.0 - t))
            else:
                raw = _inv_softplus(v)
            setattr(self, f"raw_{name}", nn.Parameter(torch.tensor(raw)))

    @classmethod
    def placeholder(cls) -> "_HybridParams":
        values = {name: 1.0 for name in cls._FIELDS}
        values.update({name: 0.5 * (lo + hi) for name, (lo, hi) in cls._BOUNDED.items()})
        return cls(**values)

    def __getattr__(self, name: str):
        if name in self._FIELDS:
            raw = super().__getattr__(f"raw_{name}")
            if name in self._BOUNDED:
                lo, hi = self._BOUNDED[name]
                return lo + (hi - lo) * torch.sigmoid(raw)
            return torch.nn.functional.softplus(raw)
        return super().__getattr__(name)


class PhysicsInformedModel(TrainableModel):
    """Bicycle force balance + learned ExpTanh tires + actuator lag states."""

    #: latent actuator states integrated alongside [vx, vy, yaw_rate]:
    #: [delta (servo angle, rad), f_drive (drive-force surrogate, N)]
    latent_dim = 2

    def __init__(
        self,
        car,
        params: _HybridParams,
        front: _ExpTanhCurve,
        rear: _ExpTanhCurve,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self._car = car
        self._p = params.to(self._device)
        self._front = front.to(self._device)
        self._rear = rear.to(self._device)
        self._m = float(car.mass_total_kg)
        self._a = float(car.cog_distance_from_front_axle_m)
        self._b = float(car.cog_distance_from_rear_axle_m)

    # --- construction -------------------------------------------------------

    @classmethod
    def new(
        cls,
        normalization: dict | None = None,
        *,
        car="drift_rc",
        n_terms: int = 4,
        device: str = "cpu",
        **kwargs,
    ) -> "PhysicsInformedModel":
        """Physically-motivated init (normalization unused: inputs are slip quantities)."""
        cp = load_car_parameters(car) if isinstance(car, str) else car
        g = 9.81
        izz0 = cp.mass_total_kg * (cp.length_base_plate_m**2 + cp.track_width_mean_m**2) / 12.0
        fzf = cp.front_axle_load_fraction * cp.mass_total_kg * g
        fzr = cp.rear_axle_load_fraction * cp.mass_total_kg * g
        fwd = cp.mass_total_kg * 4.0
        params = _HybridParams(
            izz=max(izz0, 1e-3),
            steer_gain=0.4,
            fwd_gain=fwd,
            # ESC default "Max Reverse Force" is 25% (QuicRun 10BL120 manual).
            rev_gain=0.25 * fwd,
            brake_gain=cp.mass_total_kg * 6.0,
            drag=0.5,
            f0=0.05,
            tau_s=0.08,  # servo response
            tau_w=0.12,  # ESC punch / drive-train spool-up
        )
        front = _ExpTanhCurve(peak=0.5 * fzf, stiffness=fzf / 0.1, n_terms=n_terms)
        rear = _ExpTanhCurve(peak=0.5 * fzr, stiffness=fzr / 0.1, n_terms=n_terms)
        return cls(cp, params, front, rear, device=device)

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "PhysicsInformedModel":
        cp = load_car_parameters(ckpt.get("car", "drift_rc"))
        n_terms = ckpt.get("n_terms", 4)
        params = _HybridParams.placeholder()
        params.load_state_dict(ckpt["params_state"])
        front = _ExpTanhCurve(1.0, 1.0, n_terms=n_terms)
        front.load_state_dict(ckpt["front_state"])
        rear = _ExpTanhCurve(1.0, 1.0, n_terms=n_terms)
        rear.load_state_dict(ckpt["rear_state"])
        return cls(cp, params, front, rear, device=device)

    def to_checkpoint(self) -> dict:
        return {
            "model_type": "hybrid",
            "car": self._car.name,
            "n_terms": self._front.raw_a.shape[0],
            "params_state": self._p.state_dict(),
            "front_state": self._front.state_dict(),
            "rear_state": self._rear.state_dict(),
        }

    # --- latent actuator states ----------------------------------------------

    def _f_cmd(self, drive: torch.Tensor) -> torch.Tensor:
        """Commanded drive force: asymmetric forward/reverse gains."""
        p = self._p
        return p.fwd_gain * drive.clamp(min=0.0) + p.rev_gain * drive.clamp(max=0.0)

    def init_latent(self, dynamic: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Quasi-steady latents at a window start: actuators settled on the command."""
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        return torch.stack([self._p.steer_gain * steering, self._f_cmd(drive)], dim=-1)

    # --- derivative (source of truth) ----------------------------------------

    def derivative(self, dynamic: torch.Tensor, action: torch.Tensor) -> Prediction:
        """Per-second derivative of ``[vx, vy, yaw_rate, delta, f_drive]``."""
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        delta, f_drive = dynamic[..., 3], dynamic[..., 4]
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        brake = action[..., 2].clamp(0.0, 1.0)
        p, m, a, b = self._p, self._m, self._a, self._b

        # First-order actuator dynamics (servo lag; ESC punch / spool-up).
        delta_dot = (p.steer_gain * steering - delta) / p.tau_s
        f_drive_dot = (self._f_cmd(drive) - f_drive) / p.tau_w

        # Sign-preserving magnitude floor: reverse driving (vx < 0) exists in the
        # data, so the denominator must keep the sign, not clamp up to +_VX_MIN.
        # Slip *angles* feed the curves directly (not tan(alpha) as in the paper:
        # tan flips sign and explodes past 90°, which spins in this data reach).
        # The denominator is |vx| (floored): the lateral force must oppose the
        # axle's lateral slip in *both* driving directions — with a signed vx,
        # reverse driving puts alpha near ±π, flipping the force sign and (via
        # the combined-slip norm) starving the longitudinal share entirely.
        vx_mag = vx.abs().clamp(min=_VX_MIN)
        alpha_f = torch.atan2(vy + a * r, vx_mag) - delta
        alpha_r = torch.atan2(vy - b * r, vx_mag)

        # Slip-angle forces exist only while the wheels *roll*: the vx floor above
        # fabricates a slip angle from the steering angle alone at standstill,
        # which would let a parked car spin up when steered (a body-speed gate is
        # not enough — its own small forces build vy, reopen the gate, and feed a
        # self-sustaining spin). Gate on rolling speed |vx|; the drive share
        # stays ungated — thrust must launch the car from rest.
        roll_gate = torch.tanh(vx.abs() / _ROLL_FADE_SPEED)

        fyf = -self._front(alpha_f) * roll_gate

        # Rear: one combined-slip curve, decomposed along the slip direction — a
        # smooth friction circle. Drive consumes grip continuously; full throttle
        # fades the lateral force out along the learned curve (power-over drift)
        # instead of collapsing at a hard clamp.
        sx = p.f0 * f_drive
        sy = alpha_r
        s = torch.sqrt(sx**2 + sy**2 + _EPS)
        f_tot = self._rear(s)
        fxr = f_tot * sx / s
        fyr = -f_tot * sy / s * roll_gate

        # Brake force opposes the current motion and fades out around zero speed
        # (proportional ESC brake); drag covers rolling/aero resistance (the
        # ESC's drag-brake setting is 0% by default, i.e. free coasting).
        fx = fxr - p.brake_gain * brake * torch.tanh(vx / _BRAKE_FADE_SPEED) - p.drag * vx

        # Non-rolling tires *scrub*: lateral/rotational motion of a (near-)parked
        # car is strongly damped by sliding friction rather than driven by
        # cornering forces. Fades out as the wheels start rolling.
        scrub = _SCRUB_DAMPING * (1.0 - roll_gate)

        vx_dot = (fx - fyf * torch.sin(delta)) / m + vy * r
        vy_dot = (fyf * torch.cos(delta) + fyr) / m - vx * r - scrub * vy
        r_dot = (a * fyf * torch.cos(delta) - b * fyr) / p.izz - scrub * r
        return Prediction(
            mean=torch.stack([vx_dot, vy_dot, r_dot, delta_dot, f_drive_dot], dim=-1), cov=None
        )

    def trainable_parameters(self):
        return (
            list(self._p.parameters())
            + list(self._front.parameters())
            + list(self._rear.parameters())
        )

    def set_train_mode(self, mode: bool = True) -> None:
        self._p.train(mode)
        self._front.train(mode)
        self._rear.train(mode)
