"""
Physics-informed hybrid: a neural-tire bicycle with wheel speed as a *latent* state.

Fixed single-track force balance, learned tire curves, and first-order actuator states
(after Djeumou et al., "Reference-Free Formula Drift with Reinforcement Learning",
arXiv:2410.20990). The rear combined-slip input is the paper's **slip ratio**:

    sigma_r = (R·omega_r - vx) / max(|vx|, v_floor)

with the rear wheel speed ``omega_r`` integrated as a latent state from the wheel's
own torque balance::

    I_w · omega_r_dot = tq - R·Fxr - b_w·omega_r - T_brake

The car carries no wheel-speed sensor, so ``omega_r`` is a latent. An earlier hybrid
substituted the ESC drive *force* for the slip ratio, but a force cannot represent
wheel *spin*: it is capped by the ESC gain and does not grow when the tire breaks
loose, so the longitudinal share of the friction circle (``sx/s``) collapsed as the
slip angle grew — the fitted model made no thrust in a slide, kept full rear lateral
grip, and could not sustain a drift. That variant has been removed.

A slip ratio has the missing property: when the rear tire saturates it stops
absorbing motor torque, the wheel spins up, ``sigma_r`` grows, and the friction
circle bleeds lateral force away. Power-over drift falls out of the physics.

``omega_r`` is unmeasured, but it is not free: the tire reaction ``-R·Fxr`` couples
it to the wheel torque balance, it drives ``vx`` through ``Fxr`` (which *is*
observed), and each window starts it at the wheel's *torque equilibrium* given the
recorded state (see :meth:`init_latent` — seeding it at the rolling speed instead
starts every drift window with a rear tire that still has grip).
The absolute scale is only weakly identified (``I_w``, ``torque_gain`` and ``f0``
trade off), which is why those three are bounded rather than free. Once the AS5600
encoder lands, compare this latent against the measured wheel speed — that is the
validation this model cannot do for itself.

Deterministic. Canonical dynamic state ``[vx, vy, yaw_rate]`` plus ``latent_dim=3``
(``delta``, ``tq``, ``omega_r``).
"""

import math

import torch
import torch.nn as nn

from ..car_parameters import load_car_parameters
from .base import Prediction, TrainableModel
from .dynamic import _VX_MIN, _inv_softplus

_EPS = 1e-6
_ROLL_FADE_SPEED = 0.1  # m/s rolling speed over which slip-angle tire forces fade in
_SCRUB_DAMPING = 8.0  # 1/s decay of lateral/yaw motion from tire scrub when not rolling

# Slip ratio divides by forward speed. The recorded data spends most of its time
# below 1 m/s, so an unfloored sigma explodes at every launch; the floor trades
# slip-ratio fidelity at a crawl (where the tire is far from its limit anyway) for
# a bounded, integrable model.
_V_SLIP_FLOOR = 0.4  # m/s
_SIGMA_LIMIT = 5.0  # soft clamp on |sigma_r| (a spun-up wheel saturates the tire long before)
_OMEGA_BRAKE_FADE = 5.0  # rad/s over which the brake torque reverses sign around omega = 0

# Relaxation used to settle the latent wheel speed at a window start (see init_latent).
# The inertia here is deliberately larger than the model's own: it only sets how fast the
# relaxation converges, not where it converges to, and a big one lets 40 cheap steps do it.
_WHEEL_SETTLE_STEPS, _WHEEL_SETTLE_DT, _WHEEL_SETTLE_INERTIA = 40, 0.01, 0.01

# Sliding-friction floor: the force a *fully sliding* tire keeps, as a fraction of its
# peak. These are hard-plastic drift tires, not rubber — their friction is close to
# Coulomb (kinetic ~ static, force ~ mu*N once sliding, little peak-then-falloff), so
# both floors sit high and the curves are near-flat past saturation. What makes this car
# drift is a low mu plus the friction circle turning the rear force longitudinal as the
# wheel spins (sx/s -> 1, leaving nothing lateral) — NOT the force magnitude collapsing
# with slip angle. Two things went wrong when the rear was allowed to collapse: the car
# pivoted about its rear axle instead of its front (the recorded car's donut ICR is
# +0.18 m from the CoG, right at the front axle), and a spun-up rear tire exerted almost
# no force back on its own wheel, so nothing braked it — it freewheeled for ~20 s and the
# car drifted until braked.
_EPS_MIN_FRONT, _RAW_EPS_INIT_FRONT = 0.7, 1.0  # -> eps starts at 0.92
_EPS_MIN_REAR, _RAW_EPS_INIT_REAR = 0.7, 1.0  # -> eps starts at 0.92

# Peak friction, bounded by what the car is *measured* to reach: the p90 of its smoothed
# CoG acceleration is ~3 m/s^2 (mu ~ 0.30) and the p99 ~4.5 m/s^2 (mu ~ 0.45), on hard
# plastic drift tires. Left free the fit chose 0.49 — a tire so grippy the wheel never
# spins at part throttle, which is what kept the model from drifting and made it drive
# straight out of recorded donuts.
_MU_BOUNDS = (0.20, 0.40)
_MU_INIT = 0.30


class _MuTireCurve(nn.Module):
    """
    A tire curve whose *peak* is a bounded friction coefficient: ``F(s) = mu·Fz·phi(s)``.

    ``phi`` is the ExpTanh shape with softmax weights, so it saturates at ~1 and ``mu`` is
    the peak friction coefficient outright — which means it can be bounded by something we
    measured rather than left to the fit. The car's own acceleration envelope says
    ``mu ~ 0.3`` (p90 of its smoothed CoG acceleration is ~3 m/s²; these are hard plastic
    drift tires on a smooth floor). Fitted freely it came out at **0.49**, and that single
    error is why the model would not drift properly: with a rear tire that grippy, 39%
    throttle never spins the wheel, so the friction circle stays shut, the rear keeps its
    lateral grip, and the model drives out of a recorded donut instead of holding it —
    8 m of position error one second in, on a car that moves half a metre in that time.
    """

    def __init__(
        self,
        fz: float,
        stiffness: float,
        n_terms: int = 4,
        *,
        mu_bounds: tuple[float, float] = _MU_BOUNDS,
        mu_init: float = _MU_INIT,
        eps_min: float = 0.7,
        raw_eps_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.fz = float(fz)
        self.mu_lo, self.mu_hi = mu_bounds
        self.eps_min = float(eps_min)
        t = (mu_init - self.mu_lo) / (self.mu_hi - self.mu_lo)
        self.raw_mu = nn.Parameter(torch.tensor(math.log(t / (1.0 - t))))
        # Shape: softmax weights (sum to 1) x saturating tanh bases.
        self.raw_w = nn.Parameter(torch.zeros(n_terms))
        b0 = stiffness / max(fz * mu_init, _EPS)
        self.raw_b = nn.Parameter(
            torch.tensor([_inv_softplus(b0 * (0.5 + 0.5 * i)) for i in range(1, n_terms + 1)])
        )
        self.raw_c = nn.Parameter(torch.full((n_terms,), -1.0))
        self.raw_eps = nn.Parameter(torch.tensor(raw_eps_init))

    @property
    def mu(self) -> torch.Tensor:
        """Peak friction coefficient (bounded to the measured envelope)."""
        return self.mu_lo + (self.mu_hi - self.mu_lo) * torch.sigmoid(self.raw_mu)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.raw_w, dim=0)
        b = torch.nn.functional.softplus(self.raw_b)
        c = torch.nn.functional.softplus(self.raw_c)
        eps = self.eps_min + (1.0 - self.eps_min) * torch.sigmoid(self.raw_eps)
        su = s.unsqueeze(-1)
        decay = eps + (1.0 - eps) * torch.exp(-c * su**2)
        phi = (w * torch.tanh(b * su) * decay).sum(-1)  # saturates at ~1
        return self.mu * self.fz * phi


class _WheelParams(nn.Module):
    """Learnable scalars (softplus-positive, or sigmoid-bounded where identifiability is weak)."""

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
    #   f0    — free, it collapsed to ~0.02 and switched the friction circle off.
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
    def placeholder(cls) -> "_WheelParams":
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
    """Bicycle force balance + learned tires + latent wheel speed (true slip ratio)."""

    #: [delta (servo angle, rad), tq (drive torque at the wheel, N m), omega_r (rad/s)]
    latent_dim = 3

    def __init__(
        self,
        car,
        params: _WheelParams,
        front: _MuTireCurve,
        rear: _MuTireCurve,
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
        self._r_tire = float(car.tire_radius_m)
        # The car's measured steering lock anchors the servo map; the fit may only
        # scale it within `steer_scale`'s band (see _BOUNDED).
        self._steer_max = float(car.max_steer_angle_rad)
        # Motor torque-speed line: T(omega) = torque_gain*duty - torque_gain/omega_noload*omega.
        # From the kV and the pack, not from the data.
        self._omega_noload = float(car.wheel_noload_speed_rad_s)

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
        # Full-throttle wheel torque. Set it from the *tire*: the drivetrain must be able
        # to overpower the rear's grip (mu*Fzr) or the wheel can never spin up. With the
        # tire now bounded to the measured mu, this needs far less torque than the
        # over-grippy fit did — and a weaker drivetrain also stops the model rocketing
        # away at part throttle, which it did while the real car sat in a steady donut.
        torque0 = 2.0 * (_MU_INIT * fzr) * cp.tire_radius_m
        params = _WheelParams(
            izz=max(izz0, 1e-3),
            iw=3e-4,  # wheels + reflected rotor; see _BOUNDED
            steer_scale=0.9,
            torque_gain=torque0,
            rev_frac=0.25,
            brake_torque=2.0 * torque0,
            drag=0.5,
            f0=1.0,  # dimensionless now: sigma and alpha are both O(1) at the limit
            tau_s=0.08,
            tau_w=0.12,
        )
        front = _MuTireCurve(
            fz=fzf,
            stiffness=fzf / 0.1,
            n_terms=n_terms,
            eps_min=_EPS_MIN_FRONT,
            raw_eps_init=_RAW_EPS_INIT_FRONT,
        )
        rear = _MuTireCurve(
            fz=fzr,
            stiffness=fzr / 0.1,
            n_terms=n_terms,
            eps_min=_EPS_MIN_REAR,
            raw_eps_init=_RAW_EPS_INIT_REAR,
        )
        return cls(cp, params, front, rear, device=device)

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "PhysicsInformedModel":
        # Checkpoints from the pre-2026-07 "hybrid" (a drive-force bicycle with a 2-D
        # latent, replaced because it could not represent wheel spin and so could not
        # drift) carry no "mu_bounds" and a different parameter set. They cannot load
        # here; fail clearly instead of raising an opaque state_dict shape error.
        if "mu_bounds" not in ckpt:
            raise ValueError(
                "This is a pre-wheel 'hybrid' checkpoint (drive-force bicycle), which is "
                "no longer supported — retrain with the current hybrid model."
            )
        cp = load_car_parameters(ckpt.get("car", "drift_rc"))
        g = 9.81
        n_terms = ckpt.get("n_terms", 4)
        mu_bounds = tuple(ckpt.get("mu_bounds", _MU_BOUNDS))
        params = _WheelParams.placeholder()
        params.load_state_dict(ckpt["params_state"])
        fzf = cp.front_axle_load_fraction * cp.mass_total_kg * g
        fzr = cp.rear_axle_load_fraction * cp.mass_total_kg * g
        front = _MuTireCurve(
            fz=fzf,
            stiffness=1.0,
            n_terms=n_terms,
            mu_bounds=mu_bounds,
            eps_min=ckpt.get("eps_min_front", _EPS_MIN_FRONT),
        )
        front.load_state_dict(ckpt["front_state"])
        rear = _MuTireCurve(
            fz=fzr,
            stiffness=1.0,
            n_terms=n_terms,
            mu_bounds=mu_bounds,
            eps_min=ckpt.get("eps_min_rear", _EPS_MIN_REAR),
        )
        rear.load_state_dict(ckpt["rear_state"])
        return cls(cp, params, front, rear, device=device)

    def to_checkpoint(self) -> dict:
        return {
            "model_type": "hybrid",
            "car": self._car.name,
            "n_terms": self._front.raw_w.shape[0],
            "mu_bounds": [self._front.mu_lo, self._front.mu_hi],
            "eps_min_front": self._front.eps_min,
            "eps_min_rear": self._rear.eps_min,
            "params_state": self._p.state_dict(),
            "front_state": self._front.state_dict(),
            "rear_state": self._rear.state_dict(),
        }

    # --- latent actuator / wheel states --------------------------------------

    def _torque_cmd(self, drive: torch.Tensor) -> torch.Tensor:
        """Commanded wheel torque: the ESC caps reverse (rev_frac of forward)."""
        p = self._p
        return p.torque_gain * (drive.clamp(min=0.0) + p.rev_frac * drive.clamp(max=0.0))

    def advance(self, dynamic, action, deriv, dt):
        """
        Euler for the chassis; a linearly-implicit step for the wheel speed.

        The wheel-slip mode is *stiff*: the tire's longitudinal force turns over within a
        couple of degrees of slip ratio, giving the wheel a ~2 ms time constant, while the
        data (and the sim) step at 20-50 ms. Explicit Euler on that is unconditionally
        unstable here — ``lambda*dt`` lands around 10-25, far past the limit of 2 — and it
        silently wrecked long rollouts: a recorded donut would "predict" the car reaching
        18 m/s, which its own tires (mu ~ 0.3, so a <= 2.8 m/s^2) cannot physically do.

        The fix is the integrator, not a fatter inertia. One backward-Euler step on the
        linearized wheel equation, ``omega += dt*net / (iw + dt*k)`` with ``k`` the local
        damping ``-d(net)/d(omega)``, is stable for any dt and still first-order accurate,
        so the wheel inertia can stay physical.
        """
        nxt = dynamic + deriv * dt
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        omega = dynamic[..., 5]
        p, rw = self._p, self._r_tire

        # local damping of the wheel equation (finite difference on the tire reaction)
        h = 1e-2
        fxr0, _, _ = self._rear_slip(vx, vy, r, omega)
        fxr1, _, _ = self._rear_slip(vx, vy, r, omega + h)
        k = rw * (fxr1 - fxr0) / h + p.torque_gain / self._omega_noload
        k = k.clamp(min=0.0)

        # dt arrives as a (B, 1) tensor from the training rollout and a float from the sim.
        dt_w = dt[..., 0] if torch.is_tensor(dt) and dt.dim() > 0 else dt
        net = deriv[..., 5] * p.iw  # the explicit net torque, recovered
        omega_next = omega + dt_w * net / (p.iw + dt_w * k)
        return torch.cat([nxt[..., :5], omega_next.unsqueeze(-1)], dim=-1)

    def _rear_slip(self, vx, vy, r, omega):
        """Rear combined slip: returns (fxr, fyr_ungated, sigma)."""
        p, rw, b = self._p, self._r_tire, self._b
        alpha_r = torch.atan2(vy - b * r, vx.abs().clamp(min=_VX_MIN))
        sigma = (rw * omega - vx) / vx.abs().clamp(min=_V_SLIP_FLOOR)
        sigma = _SIGMA_LIMIT * torch.tanh(sigma / _SIGMA_LIMIT)
        sx, sy = p.f0 * sigma, alpha_r
        s = torch.sqrt(sx**2 + sy**2 + _EPS)
        f_tot = self._rear(s)
        return f_tot * sx / s, -f_tot * sy / s, sigma

    def init_latent(self, dynamic: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Window start: servo and ESC settled on the command, wheel at *torque equilibrium*.

        Seeding the wheel at the rolling speed (sigma = 0) is right for a car driving
        normally and badly wrong for one that is drifting: a real drifting rear wheel is
        spun up far past rolling (sigma ~ 0.6-2.0), and that spin is exactly what holds
        the rear grip down. Starting every window with a fully-gripping rear tire made
        the model kill its own donuts — 8 m of position error one second into a recorded
        donut, against 0.25 m in a drift — and taught training that slides are harder to
        sustain than they really are.

        So relax the wheel to where the tire force balances the commanded drive torque,
        holding the chassis state fixed. On a straight this returns the rolling speed;
        sideways under power it returns a spun-up wheel. The relaxation uses an inflated
        inertia purely to converge in few steps — the fixed point (net torque = 0) does
        not depend on it.
        """
        p, rw = self._p, self._r_tire
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        tq = self._torque_cmd(drive)

        omega = vx / rw
        for _ in range(_WHEEL_SETTLE_STEPS):
            fxr, _, _ = self._rear_slip(vx, vy, r, omega)
            t_droop = (p.torque_gain / self._omega_noload) * omega
            net = tq - t_droop - rw * fxr
            omega = omega + _WHEEL_SETTLE_DT * net / _WHEEL_SETTLE_INERTIA

        delta = self._p.steer_scale * self._steer_max * steering
        return torch.stack([delta, tq, omega], dim=-1)

    # --- derivative (source of truth) ----------------------------------------

    def derivative(self, dynamic: torch.Tensor, action: torch.Tensor) -> Prediction:
        """Per-second derivative of ``[vx, vy, yaw_rate, delta, tq, omega_r]``."""
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        delta, tq, omega = dynamic[..., 3], dynamic[..., 4], dynamic[..., 5]
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        brake = action[..., 2].clamp(0.0, 1.0)
        p, m, a, b, rw = self._p, self._m, self._a, self._b, self._r_tire

        delta_dot = (p.steer_scale * self._steer_max * steering - delta) / p.tau_s
        tq_dot = (self._torque_cmd(drive) - tq) / p.tau_w

        roll_gate = torch.tanh(vx.abs() / _ROLL_FADE_SPEED)

        # Front slip angle in the *tire's own frame*: rotate the front axle's contact
        # velocity by the steering angle, then take the slip against the tire's own
        # longitudinal direction. The usual shortcut — atan2(vy + a*r, |vx|) - delta —
        # assumes a small steering angle and forward travel. This car has ~70 deg of
        # lock, and a spun car rolls backwards: in that corner the shortcut flips the
        # sign of the front force, so it pushed the front axle the way it was already
        # sliding. Flicking from full lock to full opposite lock and releasing the
        # controls then left the car spinning about its rear axle forever, with the
        # tires feeding it +1 W out of nothing. Slip taken in the tire frame opposes
        # the contact-patch velocity by construction, at any lock and either direction.
        ct, stt = torch.cos(delta), torch.sin(delta)
        v_fy = vy + a * r
        u_f = vx * ct + v_fy * stt  # along the tire
        w_f = -vx * stt + v_fy * ct  # across the tire
        alpha_f = torch.atan2(w_f, u_f.abs().clamp(min=_VX_MIN))
        f_lat_f = -self._front(alpha_f) * roll_gate  # opposes w_f: dissipative

        # Rear combined slip, with the paper's slip ratio: the wheel spinning faster
        # than the ground (omega·R > vx) is what erodes the lateral force.
        fxr, fyr_ungated, _ = self._rear_slip(vx, vy, r, omega)
        fyr = fyr_ungated * roll_gate

        # The tire reacts back on the wheel: torque it cannot put down spins it up.
        # The motor's torque-speed droop (back-EMF + drivetrain drag) is what caps the
        # wheel's top speed and drags a spun-up wheel back down once the throttle is
        # released — without it the wheel freewheels for tens of seconds and the car
        # never stops drifting.
        t_brake = p.brake_torque * brake * torch.tanh(omega / _OMEGA_BRAKE_FADE)
        t_droop = (p.torque_gain / self._omega_noload) * omega
        omega_dot = (tq - t_droop - rw * fxr - t_brake) / p.iw

        # The front tire's force lives in the tire frame; project it back into the body.
        fxf = -f_lat_f * stt
        fyf = f_lat_f * ct

        fx = fxr + fxf - p.drag * vx
        scrub = _SCRUB_DAMPING * (1.0 - roll_gate)

        vx_dot = fx / m + vy * r
        vy_dot = (fyf + fyr) / m - vx * r - scrub * vy
        r_dot = (a * fyf - b * fyr) / p.izz - scrub * r
        return Prediction(
            mean=torch.stack([vx_dot, vy_dot, r_dot, delta_dot, tq_dot, omega_dot], dim=-1),
            cov=None,
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
