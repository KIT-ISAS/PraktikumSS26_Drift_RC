"""
Single-track dynamic bicycle model (physics) with learnable tire/actuator params.

Known geometry/mass come from :class:`CarParameters`; the unmeasured quantities
(cornering stiffness, grip limits, yaw inertia, and the normalized-command ->
force maps) are learnable ``nn.Parameter`` s fit by the same Trainer.
Deterministic. Canonical dynamic state ``[vx, vy, yaw_rate]``.

Two modelling choices matter for drifting:

* **Saturating lateral tires + rear friction circle** — ``fy = -D tanh(C α / D)``
  per axle, with initial slope ``C`` (cornering stiffness) and friction limit
  ``D`` (≈ µ Fz). A linear tire (``-C α``) can never break traction, which makes
  the model self-stabilizing and structurally unable to drift. On the driven
  rear axle the motor force additionally *consumes* the friction circle: the
  drive/brake force is traction-capped at ``Dr`` and the lateral limit shrinks
  to ``sqrt(Dr² - fx²)`` — full throttle breaks the rear loose (power-over).
* **ESC-resolved longitudinal input** — ``derivative`` takes the resolved motor
  command ``(drive, brake)``, not the raw throttle pedal (see
  :class:`~.esc.EscStateMachine`). ``drive`` is signed thrust; the ``brake``
  force opposes the current motion and fades out around zero speed, so braking
  decelerates toward standstill instead of pushing through into reverse.
"""

import math

import torch
import torch.nn as nn

from ..car_parameters import CarParameters, load_car_parameters
from .base import Prediction, TrainableModel

_VX_MIN = 0.5  # m/s magnitude floor in slip-angle denominators (low-speed stability)
_BRAKE_FADE_SPEED = 0.1  # m/s scale over which the brake force fades out at standstill


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, to initialize a positive parameter at value ``y``."""
    y = max(y, 1e-6)
    if y > 20.0:  # softplus(x) ~ x here; expm1 would overflow long before it matters
        return y
    return math.log(math.expm1(y))


class _BicycleParams(nn.Module):
    """Learnable positive parameters of the bicycle model (raw + softplus)."""

    def __init__(
        self,
        cf: float,
        cr: float,
        df: float,
        dr: float,
        izz: float,
        steer_gain: float,
        throttle_gain: float,
        brake_gain: float,
        drag: float,
    ) -> None:
        super().__init__()
        self.raw_cf = nn.Parameter(torch.tensor(_inv_softplus(cf)))
        self.raw_cr = nn.Parameter(torch.tensor(_inv_softplus(cr)))
        self.raw_df = nn.Parameter(torch.tensor(_inv_softplus(df)))
        self.raw_dr = nn.Parameter(torch.tensor(_inv_softplus(dr)))
        self.raw_izz = nn.Parameter(torch.tensor(_inv_softplus(izz)))
        self.raw_steer = nn.Parameter(torch.tensor(_inv_softplus(steer_gain)))
        self.raw_throttle = nn.Parameter(torch.tensor(_inv_softplus(throttle_gain)))
        self.raw_brake = nn.Parameter(torch.tensor(_inv_softplus(brake_gain)))
        self.raw_drag = nn.Parameter(torch.tensor(_inv_softplus(drag)))

    @classmethod
    def placeholder(cls) -> "_BicycleParams":
        """Unit-valued params to receive a ``load_state_dict`` from a checkpoint."""
        return cls(
            cf=1.0,
            cr=1.0,
            df=1.0,
            dr=1.0,
            izz=1.0,
            steer_gain=0.4,
            throttle_gain=1.0,
            brake_gain=1.0,
            drag=0.5,
        )

    @property
    def cf(self):
        return torch.nn.functional.softplus(self.raw_cf)

    @property
    def cr(self):
        return torch.nn.functional.softplus(self.raw_cr)

    @property
    def df(self):
        return torch.nn.functional.softplus(self.raw_df)

    @property
    def dr(self):
        return torch.nn.functional.softplus(self.raw_dr)

    @property
    def izz(self):
        return torch.nn.functional.softplus(self.raw_izz)

    @property
    def steer_gain(self):
        return torch.nn.functional.softplus(self.raw_steer)

    @property
    def throttle_gain(self):
        return torch.nn.functional.softplus(self.raw_throttle)

    @property
    def brake_gain(self):
        return torch.nn.functional.softplus(self.raw_brake)

    @property
    def drag(self):
        return torch.nn.functional.softplus(self.raw_drag)


class DynamicBicycleModel(TrainableModel):
    """Physics single-track bicycle; learnable tire stiffness/grip, inertia, actuator maps."""

    def __init__(self, car: CarParameters, params: _BicycleParams, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._car = car
        self._p = params.to(self._device)
        self._m = float(car.mass_total_kg)
        self._a = float(car.cog_distance_from_front_axle_m)
        self._b = float(car.cog_distance_from_rear_axle_m)

    # --- construction -------------------------------------------------------

    @classmethod
    def new(
        cls, normalization: dict | None = None, *, car="drift_rc", device: str = "cpu", **kwargs
    ) -> "DynamicBicycleModel":
        """Build a model with physically-motivated initial parameters (normalization unused)."""
        cp = load_car_parameters(car) if isinstance(car, str) else car
        g = 9.81
        # Box-approx yaw inertia; rough cornering stiffness from axle load; grip
        # limits at µ ≈ 0.5 of the static axle load (drift RC on a slick floor).
        izz0 = cp.mass_total_kg * (cp.length_base_plate_m**2 + cp.track_width_mean_m**2) / 12.0
        fzf = cp.front_axle_load_fraction * cp.mass_total_kg * g
        fzr = cp.rear_axle_load_fraction * cp.mass_total_kg * g
        params = _BicycleParams(
            cf=fzf / 0.1,
            cr=fzr / 0.1,
            df=0.5 * fzf,
            dr=0.5 * fzr,
            izz=max(izz0, 1e-3),
            steer_gain=0.4,
            throttle_gain=cp.mass_total_kg * 4.0,
            brake_gain=cp.mass_total_kg * 6.0,
            drag=0.5,
        )
        return cls(cp, params, device=device)

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "DynamicBicycleModel":
        """Rebuild from a checkpoint dict (car name + parameter state)."""
        cp = load_car_parameters(ckpt.get("car", "drift_rc"))
        params = _BicycleParams.placeholder()
        params.load_state_dict(ckpt["params_state"])
        return cls(cp, params, device=device)

    def to_checkpoint(self) -> dict:
        return {
            "model_type": "dynamic",
            "car": self._car.name,
            "params_state": self._p.state_dict(),
        }

    # --- physics ------------------------------------------------------------

    def derivative(self, dynamic: torch.Tensor, action: torch.Tensor) -> Prediction:
        """Per-second derivative of [vx, vy, yaw_rate] from the bicycle dynamics."""
        vx, vy, r = dynamic[..., 0], dynamic[..., 1], dynamic[..., 2]
        steering = action[..., 0].clamp(-1.0, 1.0)
        drive = action[..., 1].clamp(-1.0, 1.0)
        brake = action[..., 2].clamp(0.0, 1.0)
        p, m, a, b = self._p, self._m, self._a, self._b

        delta = p.steer_gain * steering
        # The denominator is |vx| (floored), not a sign-preserving vx: the lateral
        # force must oppose the axle's lateral slip in *both* driving directions.
        # A signed vx puts alpha near ±π in reverse (vx < 0, which the data has),
        # where the tanh curve saturates at full grip and the force sign flips with
        # the sign of vy — a bang-bang limit cycle instead of a settling trajectory.
        vx_mag = vx.abs().clamp(min=_VX_MIN)
        alpha_f = torch.atan2(vy + a * r, vx_mag) - delta
        alpha_r = torch.atan2(vy - b * r, vx_mag)
        # Friction-limited lateral force: linear (-C α) for small slip, saturating
        # smoothly at the grip limit D — past it the axle slides instead of
        # gripping harder, which is what makes drifting possible at all.
        fyf = -p.df * torch.tanh(p.cf * alpha_f / p.df)
        # Rear axle (driven): friction circle. The motor force is traction-capped
        # at the rear grip limit, and whatever fraction of the circle it uses is
        # no longer available laterally — full throttle breaks the rear loose
        # (power-over), which is *the* drift mechanism on an RC car.
        fxr = (
            p.throttle_gain * drive - p.brake_gain * brake * torch.tanh(vx / _BRAKE_FADE_SPEED)
        ).clamp(-p.dr, p.dr)
        dr_lat = p.dr * torch.sqrt((1.0 - (fxr / p.dr) ** 2).clamp(min=1e-3))
        fyr = -dr_lat * torch.tanh(p.cr * alpha_r / dr_lat)
        fx = fxr - p.drag * vx

        vx_dot = (fx - fyf * torch.sin(delta)) / m + vy * r
        vy_dot = (fyf * torch.cos(delta) + fyr) / m - vx * r
        r_dot = (a * fyf * torch.cos(delta) - b * fyr) / p.izz
        return Prediction(mean=torch.stack([vx_dot, vy_dot, r_dot], dim=-1), cov=None)

    def trainable_parameters(self):
        return self._p.parameters()

    def set_train_mode(self, mode: bool = True) -> None:
        self._p.train(mode)
