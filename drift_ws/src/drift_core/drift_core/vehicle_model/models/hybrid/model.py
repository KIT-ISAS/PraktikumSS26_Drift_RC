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

Canonical dynamic state ``[vx, vy, yaw_rate]`` plus ``latent_dim=4`` (``delta``, ``tq``,
``omega_r``, ``tb`` — the ESC brake's actuation lag). Deterministic by default; with
``stochastic=True`` a learned diagonal diffusion Sigma(x, u) (the paper's neural SDE)
returns a per-second chassis covariance for ``loss="nll"`` training and for sampled
rollouts during RL training.

The learned components live in sibling modules: :mod:`.tire` (ExpTanh curve +
friction-circle direction), :mod:`.esc` (wheel/ESC torque balance + scalar params),
and :mod:`.diffusion` (the neural-SDE Sigma head).
"""

import torch

from ...car_parameters import load_car_parameters
from ..base import Prediction, TrainableModel
from ..dynamic import _VX_MIN
from . import esc
from ._nn import _mlp_shape
from .diffusion import _DiffusionNet
from .esc import _WheelParams
from .tire import (
    _MU_INIT,
    _TIRE_HIDDEN,
    _TIRE_LAYERS,
    _DirectionNet,
    _ExpTanhCurve,
    _feat_scale_of,
    build_curves,
)

_EPS = 1e-6
# Contact speed over which slip-angle tire forces fade in (and the standstill scrub
# damper fades out). This MUST gate on the planar speed |(vx, vy)|, not |vx|: in a deep
# drift the body-frame vx collapses as sideslip passes ~90 deg while the car is still
# moving fast sideways, and a |vx| gate then switched the tires off mid-slide and let
# the standstill damper brake at 4+ m/s^2 — more than the tires (mu ~ 0.3, ~3 m/s^2)
# could ever do. A counter-steer flick out of a drift died in half a second.
_ROLL_FADE_SPEED = 0.1  # m/s
_SCRUB_DAMPING = 8.0  # 1/s decay of lateral/yaw motion from tire scrub at standstill

# Slip-ratio denominator: the *faster* of ground speed and wheel surface speed,
# max(|vx|, R|omega|), floored only near true standstill. Normalizing by |vx| alone
# needed a fat 0.4 m/s floor against launch blow-ups (R·omega large, vx ~ 0), and that
# floor gutted braking: a locked wheel's sigma faded as -vx/0.4, so the brake force
# died quadratically below 0.4 m/s and the model crept at the end of every stop
# instead of stopping. With the two-sided denominator both extremes are exact:
# spin-up from rest gives sigma -> +1, a locked wheel gives sigma -> -1 (Coulomb
# force all the way down), and the tiny floor only breaks the 0/0 at standstill.
_V_SLIP_FLOOR = 0.1  # m/s, matches _ROLL_FADE_SPEED: below this the scrub damper owns it
_SIGMA_LIMIT = 5.0  # soft clamp on |sigma_r| (mixed-sign wheel/ground can still exceed 1)

# Shapes of the pre-paper nets, assumed for checkpoints that predate the shape keys.
_LEGACY_NN0_HIDDEN, _LEGACY_DIFF_HIDDEN, _LEGACY_LAYERS = 16, 32, 1


class PhysicsInformedModel(TrainableModel):
    """Bicycle force balance + learned tires + latent wheel speed (true slip ratio)."""

    #: [delta (servo angle, rad), tq (drive torque, N m), omega_r (rad/s),
    #:  tb (brake torque magnitude, N m — the ESC's brake actuation lag)]
    latent_dim = 4

    def __init__(
        self,
        car,
        params: _WheelParams,
        front: _ExpTanhCurve,
        rear: _ExpTanhCurve,
        nn0: _DirectionNet,
        diffusion: "_DiffusionNet | None" = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self._car = car
        self._p = params.to(self._device)
        self._front = front.to(self._device)
        self._rear = rear.to(self._device)
        self._nn0 = nn0.to(self._device)
        self._diff = diffusion.to(self._device) if diffusion is not None else None
        self._can_sample = self._diff is not None
        self._m = float(car.mass_total_kg)
        self._a = float(car.cog_distance_from_front_axle_m)
        self._b = float(car.cog_distance_from_rear_axle_m)
        self._r_tire = float(car.tire_radius_m)
        # The car's measured steering lock anchors the servo map; the fit may only
        # scale it within `steer_scale`'s band (see _WheelParams._BOUNDED).
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
        stochastic: bool = False,
        tire_features: bool = True,
        bounded_params: bool = True,
        nn0_residual: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> "PhysicsInformedModel":
        """
        Physically-motivated init (normalization unused: inputs are slip quantities).

        ``tire_features=False`` falls back to fixed per-axle tire curves instead of the
        paper's state-conditioned coefficients — the pre-2026-07-22 shape, for A/B runs.
        ``bounded_params=False`` drops the identifiability bands on the wheel/actuator
        scalars (they stay softplus-positive), letting an unconstrained fit run to see
        where it actually goes — see :class:`.esc._WheelParams` for what each band prevents.
        ``nn0_residual=False`` takes the rear force direction straight from NN0, the
        paper's literal form, instead of as a correction to the geometric slip direction.
        """
        cp = load_car_parameters(car) if isinstance(car, str) else car
        g = 9.81
        izz0 = cp.mass_total_kg * (cp.length_base_plate_m**2 + cp.track_width_mean_m**2) / 12.0
        fzf = cp.front_axle_load_fraction * cp.mass_total_kg * g
        fzr = cp.rear_axle_load_fraction * cp.mass_total_kg * g
        # Full-throttle wheel torque. Set it from the *tire*: the drivetrain must be able
        # to overpower the rear's grip (mu*Fzr) or the wheel can never spin up. With the
        # tire initialized at the measured mu, this needs far less torque than the
        # over-grippy fit did — and a weaker drivetrain also stops the model rocketing
        # away at part throttle, which it did while the real car sat in a steady donut.
        torque0 = 2.0 * (_MU_INIT * fzr) * cp.tire_radius_m
        params = _WheelParams(
            bounded=bounded_params,
            izz=max(izz0, 1e-3),
            iw=3e-4,  # wheels + reflected rotor; see _WheelParams._BOUNDED
            steer_scale=0.9,
            torque_gain=torque0,
            rev_frac=0.25,
            brake_torque=2.0 * torque0,
            drag=0.5,
            f0=1.0,  # dimensionless now: sigma and alpha are both O(1) at the limit
            tau_s=0.08,
            tau_w=0.12,
        )
        front, rear = build_curves(fzf, fzr, tire_features, cp.wheel_noload_speed_rad_s)
        diffusion = _DiffusionNet() if stochastic else None
        nn0 = _DirectionNet(residual=nn0_residual)
        return cls(cp, params, front, rear, nn0, diffusion, device=device)

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "PhysicsInformedModel":
        # Older "hybrid" checkpoints — the pre-wheel drive-force bicycle and the
        # bounded-mu wheel-slip variant — carry different parameter sets and cannot
        # load here; fail clearly instead of raising an opaque state_dict shape error.
        if ckpt.get("tire_form") != "exptanh":
            raise ValueError(
                "This hybrid checkpoint predates the paper-form ExpTanh tires (free "
                "peak + learned friction-circle direction) — retrain with the current "
                "hybrid model."
            )
        cp = load_car_parameters(ckpt.get("car", "drift_rc"))
        params = _WheelParams.placeholder(bounded=ckpt.get("params_bounded", True))
        params.load_state_dict(ckpt["params_state"])
        # No tire_* shape keys means a checkpoint from before the state-conditioned
        # coefficients: fixed scalar curves (feat_scale=None).
        tire_shape = {
            "hidden": ckpt.get("tire_hidden", _TIRE_HIDDEN),
            "layers": ckpt.get("tire_layers", _TIRE_LAYERS),
        }
        front_feat = ckpt.get("front_feat_scale")
        rear_feat = ckpt.get("rear_feat_scale")
        front = _ExpTanhCurve(
            peak=1.0,
            f4=1.0,
            positive_f5=ckpt.get("front_positive_f5", False),
            feat_scale=front_feat,
            **tire_shape,
        )
        front.load_state_dict(ckpt["front_state"])
        rear = _ExpTanhCurve(
            peak=1.0,
            f4=1.0,
            positive_f5=ckpt.get("rear_positive_f5", False),
            feat_scale=rear_feat,
            **tire_shape,
        )
        rear.load_state_dict(ckpt["rear_state"])
        nn0 = _DirectionNet(
            hidden=ckpt.get("nn0_hidden", _LEGACY_NN0_HIDDEN),
            layers=ckpt.get("nn0_layers", _LEGACY_LAYERS),
            residual=ckpt.get("nn0_residual", True),
        )
        nn0.load_state_dict(ckpt["nn0_state"])
        # A checkpoint trained stochastic keeps its diffusion head (so derivative
        # returns cov and NLL fine-tuning works); the `stochastic` argument only
        # gates whether the sim *samples* from it (mirrors NeuralDynamicsModel).
        diffusion = None
        if ckpt.get("stochastic", False):
            diffusion = _DiffusionNet(
                hidden=ckpt.get("diff_hidden", _LEGACY_DIFF_HIDDEN),
                layers=ckpt.get("diff_layers", _LEGACY_LAYERS),
            )
            diffusion.load_state_dict(ckpt["diff_state"])
        model = cls(cp, params, front, rear, nn0, diffusion, device=device)
        model._can_sample = bool(stochastic and diffusion is not None)
        return model

    def to_checkpoint(self) -> dict:
        nn0_hidden, nn0_layers = _mlp_shape(self._nn0.net)
        ckpt = {
            "model_type": "hybrid",
            "car": self._car.name,
            "tire_form": "exptanh",
            "nn0_hidden": nn0_hidden,
            "nn0_layers": nn0_layers,
            "nn0_residual": self._nn0.residual,
            "params_state": self._p.state_dict(),
            "front_state": self._front.state_dict(),
            "rear_state": self._rear.state_dict(),
            "nn0_state": self._nn0.state_dict(),
            "front_feat_scale": _feat_scale_of(self._front),
            "rear_feat_scale": _feat_scale_of(self._rear),
            "front_positive_f5": self._front.positive_f5,
            "rear_positive_f5": self._rear.positive_f5,
            "params_bounded": bool(self._p._bounds),
            "stochastic": self._diff is not None,
        }
        if self._front.net is not None:
            ckpt["tire_hidden"], ckpt["tire_layers"] = _mlp_shape(self._front.net)
        if self._diff is not None:
            diff_hidden, diff_layers = _mlp_shape(self._diff.net)
            ckpt["diff_hidden"] = diff_hidden
            ckpt["diff_layers"] = diff_layers
            ckpt["diff_state"] = self._diff.state_dict()
        return ckpt

    # --- latent actuator / wheel states --------------------------------------

    def advance(self, dynamic, action, deriv, dt):
        """
        Explicit Euler for the chassis; exact steps for the stiff actuator/wheel modes.

        Four of the seven states are linear or stiff and would need a tiny step under plain
        Euler; each gets an integrator matched to its dynamics instead, so the chassis can
        stay explicit at the data's 20-50 ms step.

        *delta, tq and tb* are first-order lags toward a target the action holds constant
        across the step (``x_dot = (target - x)/tau``), so they integrate *exactly*:
        ``x_next = x + x_dot*tau*(1 - exp(-dt/tau))``. This is unconditionally stable at any
        dt. Explicit Euler instead diverges once ``dt > 2*tau``, and ``tau_s`` is bounded down
        to 0.02 s against 20-50 ms steps (``dt/tau_s`` up to 2.5), so the servo angle could
        oscillate divergently in a rollout; the exact step removes that failure mode. As
        ``tau -> inf`` the factor ``(1 - exp(-dt/tau)) -> dt/tau`` and this recovers Euler.

        *omega* (wheel speed) is *stiff*: the tire's longitudinal force turns over within a
        couple of degrees of slip ratio, giving the wheel a ~2 ms time constant. Explicit
        Euler on that is unconditionally unstable here — ``lambda*dt`` lands around 10-25,
        far past the limit of 2 — and it silently wrecked long rollouts: a recorded donut
        would "predict" the car reaching 18 m/s, which its own tires (mu ~ 0.3, so
        a <= 2.8 m/s^2) cannot physically do. The fix is the integrator, not a fatter
        inertia. One backward-Euler step on the linearized wheel equation (see
        :func:`.esc.wheel_implicit_step`) is stable for any dt and still first-order
        accurate, so the inertia can stay physical.
        """
        nxt = dynamic + deriv * dt  # explicit Euler; delta, tq, omega, tb overridden below
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        delta, tq, omega = dynamic[..., 3], dynamic[..., 4], dynamic[..., 5]
        tb = dynamic[..., 6]
        p, rw = self._p, self._r_tire

        # dt arrives as a (B, 1) tensor from the training rollout and a float from the sim.
        dt_w = dt[..., 0] if torch.is_tensor(dt) and dt.dim() > 0 else dt

        # delta, tq, tb: exact linear-lag step (expm1(-y) = exp(-y) - 1, stable for small y).
        delta_next = delta - deriv[..., 3] * p.tau_s * torch.expm1(-dt_w / p.tau_s)
        tq_next = tq - deriv[..., 4] * p.tau_w * torch.expm1(-dt_w / p.tau_w)
        tb_next = tb - deriv[..., 6] * esc._TAU_BRAKE * torch.expm1(
            -torch.as_tensor(dt_w) / esc._TAU_BRAKE
        )

        # omega: linearly-implicit step with the wheel equation's local damping (finite
        # difference of the tire reaction, plus the droop/brake slopes; see esc.py).
        h = 1e-2
        fxr0, _, _ = self._rear_slip(vx, vy, r, omega)
        fxr1, _, _ = self._rear_slip(vx, vy, r, omega + h)
        k = esc.wheel_damping_k(p, omega, tb, (fxr1 - fxr0) / h, rw, self._omega_noload)
        net = deriv[..., 5] * p.iw  # the explicit net torque, recovered
        omega_next = esc.wheel_implicit_step(omega, net, k, dt_w, p.iw)

        return torch.stack(
            [nxt[..., 0], nxt[..., 1], nxt[..., 2], delta_next, tq_next, omega_next, tb_next],
            dim=-1,
        )

    def _tire_feat(self, vx, vy, r, omega=None):
        """
        The paper's tire-curve features: ``[r, V, beta]``, plus ``omega_r`` at the rear.

        ``None`` for an unconditioned curve, so the fixed-scalar tires cost nothing.
        Note ``beta = atan2(vy, vx)`` is the paper's literal feature and it *wraps*: a
        car sliding backwards jumps between +pi and -pi, and the curve reshapes
        discontinuously across it. The full-size car never goes there; this one spins.
        Watch this if a conditioned fit misbehaves in spins but not in drifts.
        """
        curve = self._front if omega is None else self._rear
        if curve.net is None:
            return None
        feat = [r, torch.sqrt(vx**2 + vy**2 + _EPS), torch.atan2(vy, vx)]
        if omega is not None:
            feat.append(omega)
        return torch.stack(feat, dim=-1)

    def _rear_slip(self, vx, vy, r, omega):
        """Rear combined slip (the paper's form): returns (fxr, fyr_ungated, sigma)."""
        p, rw, b = self._p, self._r_tire, self._b
        tan_ar = (vy - b * r) / vx.abs().clamp(min=_VX_MIN)
        denom = torch.maximum(vx.abs(), (rw * omega).abs()).clamp(min=_V_SLIP_FLOOR)
        sigma = (rw * omega - vx) / denom
        sigma = _SIGMA_LIMIT * torch.tanh(sigma / _SIGMA_LIMIT)
        # One curve over the paper's squared combined slip gives the total force; the
        # learned direction NN0 splits it. f0 weights wheel spin against slip angle in
        # both, so at (zero-init) start the direction is the geometric slip vector
        # (sqrt(f0)·sigma, -tan_ar) — a plain friction circle — and z is its square.
        # Deferred: the residual direction is unpinned at zero slip, so a trained NN0 can
        # shift the zero-force point to a nonzero wheel slip (a parked wheel then settles
        # at omega != 0). Scaling the residual by the slip magnitude would pin it; parked
        # with the F(0) boundary-condition question until the next training campaign.
        z = tan_ar**2 + p.f0 * sigma**2
        f_tot = self._rear(z, self._tire_feat(vx, vy, r, omega)).clamp(min=0.0)
        res = self._nn0(torch.atan(tan_ar), sigma)
        if self._nn0.residual:
            dx = p.f0.sqrt() * sigma + res[..., 0]
            dy = -tan_ar + res[..., 1]
        else:  # the paper's literal form: the direction *is* the network output
            dx, dy = res[..., 0], res[..., 1]
        norm = torch.sqrt(dx**2 + dy**2 + _EPS)
        return f_tot * dx / norm, f_tot * dy / norm, sigma

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

        So relax the wheel to where the tire force balances the commanded drive torque
        (minus any commanded brake), holding the chassis state fixed. On a straight this
        returns the rolling speed; sideways under power it returns a spun-up wheel. The
        relaxation uses an inflated inertia purely to converge in few steps — the fixed
        point (net torque = 0) does not depend on it — and each step is the same
        linearly-implicit form as ``advance``, so the stiff low-speed slip mode cannot
        make the settle loop itself oscillate.
        """
        p, rw = self._p, self._r_tire
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        brake = action[..., 2].clamp(0.0, 1.0)
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        tq = esc.torque_cmd(p, drive)
        tb = p.brake_torque * brake  # brake lag settled on the command

        h = 1e-2
        omega = vx / rw
        for _ in range(esc._WHEEL_SETTLE_STEPS):
            fxr, _, _ = self._rear_slip(vx, vy, r, omega)
            fxr1, _, _ = self._rear_slip(vx, vy, r, omega + h)
            k = esc.wheel_damping_k(p, omega, tb, (fxr1 - fxr) / h, rw, self._omega_noload)
            net = esc.wheel_net_torque(p, omega, tq, tb, fxr, rw, self._omega_noload)
            omega = esc.wheel_implicit_step(
                omega, net, k, esc._WHEEL_SETTLE_DT, esc._WHEEL_SETTLE_INERTIA
            )

        delta = p.steer_scale * self._steer_max * steering
        return torch.stack([delta, tq, omega, tb], dim=-1)

    # --- derivative (source of truth) ----------------------------------------

    def derivative(self, dynamic: torch.Tensor, action: torch.Tensor) -> Prediction:
        """Per-second derivative of ``[vx, vy, yaw_rate, delta, tq, omega_r, tb]``."""
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        delta, tq, omega = dynamic[..., 3], dynamic[..., 4], dynamic[..., 5]
        tb = dynamic[..., 6]
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        brake = action[..., 2].clamp(0.0, 1.0)
        p, m, a, b, rw = self._p, self._m, self._a, self._b, self._r_tire

        delta_dot = (p.steer_scale * self._steer_max * steering - delta) / p.tau_s
        tq_dot = (esc.torque_cmd(p, drive) - tq) / p.tau_w
        tb_dot = (p.brake_torque * brake - tb) / esc._TAU_BRAKE

        speed = torch.sqrt(vx**2 + vy**2 + _EPS)
        roll_gate = torch.tanh(speed / _ROLL_FADE_SPEED)

        # Front slip in the *tire's own frame*: rotate the front axle's contact velocity
        # by the steering angle and take the slip against the tire's longitudinal axis.
        # This is NOT a different (or more accurate) trig form of the paper's
        # alpha_f = atan((vy + a*r)/vx) - delta — expanding tan(atan(v_fy/vx) - delta)
        # and clearing cos(delta) gives exactly w_f/u_f, so the two agree to machine
        # precision at any steering angle. Nothing here is a small-angle correction.
        # What differs is the *denominator*: the paper divides by the signed along-tire
        # speed, we divide by |u_f|. The paper's car only ever drives forwards; when
        # this one rolls backwards or slips past 90 deg, u_f goes negative, tan(alpha_f)
        # flips sign, and the front force pushed the axle the way it was already
        # sliding. Flicking to opposite lock and releasing then left the car spinning
        # about its rear axle forever, tires feeding it +1 W out of nothing. With |u_f|
        # the force always opposes w_f, i.e. stays dissipative, in every quadrant.
        # The floor is _VX_MIN, which also bounds tan(alpha_f) as u_f -> 0.
        ct, stt = torch.cos(delta), torch.sin(delta)
        v_fy = vy + a * r
        u_f = vx * ct + v_fy * stt  # along the tire
        w_f = -vx * stt + v_fy * ct  # across the tire
        tan_af = w_f / u_f.abs().clamp(min=_VX_MIN)
        f_lat_f = -self._front(tan_af, self._tire_feat(vx, vy, r)) * roll_gate  # opposes w_f

        # Rear combined slip, with the paper's slip ratio: the wheel spinning faster
        # than the ground (omega·R > vx) is what erodes the lateral force.
        fxr, fyr_ungated, sigma_r = self._rear_slip(vx, vy, r, omega)
        fyr = fyr_ungated * roll_gate

        # The tire reacts back on the wheel: torque it cannot put down spins it up.
        # The motor's torque-speed droop (back-EMF + drivetrain drag) is what caps the
        # wheel's top speed and drags a spun-up wheel back down once the throttle is
        # released — without it the wheel freewheels for tens of seconds and the car
        # never stops drifting. (Droop + brake reaction live in esc.wheel_net_torque.)
        omega_dot = esc.wheel_net_torque(p, omega, tq, tb, fxr, rw, self._omega_noload) / p.iw

        # The front tire's force lives in the tire frame; project it back into the body.
        fxf = -f_lat_f * stt
        fyf = f_lat_f * ct

        fx = fxr + fxf - p.drag * vx
        scrub = _SCRUB_DAMPING * (1.0 - roll_gate)

        vx_dot = fx / m + vy * r
        vy_dot = (fyf + fyr) / m - vx * r - scrub * vy
        r_dot = (a * fyf - b * fyr) / p.izz - scrub * r
        cov = None
        if self._diff is not None:
            # Deferred: the Sigma head is ungated, so it carries tracker measurement
            # noise even at standstill — a parked stochastic sim wanders. Gating Sigma by
            # the combined slip magnitude (process noise originates in the tires) is the
            # fix; parked until the next training campaign, as current calibrations assume
            # no gate.
            cov = self._diff(dynamic[..., :3], torch.stack([steering, drive, brake], dim=-1))
        return Prediction(
            mean=torch.stack(
                [vx_dot, vy_dot, r_dot, delta_dot, tq_dot, omega_dot, tb_dot], dim=-1
            ),
            cov=cov,
        )

    def trainable_parameters(self):
        params = (
            list(self._p.parameters())
            + list(self._front.parameters())
            + list(self._rear.parameters())
            + list(self._nn0.parameters())
        )
        if self._diff is not None:
            params += list(self._diff.parameters())
        return params

    def diffusion_parameters(self):
        """Only the diffusion head's parameters (for Sigma-only calibration)."""
        if self._diff is None:
            raise ValueError(
                "This hybrid has no diffusion head — train (or load) it with "
                "stochastic=True before calibrating Sigma."
            )
        return list(self._diff.parameters())

    def set_train_mode(self, mode: bool = True) -> None:
        self._p.train(mode)
        self._front.train(mode)
        self._rear.train(mode)
        self._nn0.train(mode)
        if self._diff is not None:
            self._diff.train(mode)
