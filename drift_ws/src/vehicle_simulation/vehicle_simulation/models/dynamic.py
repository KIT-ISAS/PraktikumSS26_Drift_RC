"""
Single-track dynamic bicycle model (physics) with learnable tire/actuator params.

Known geometry/mass come from :class:`CarParameters`; the unmeasured quantities
(cornering stiffness, yaw inertia, and the normalized-steering->angle /
normalized-throttle->force maps) are learnable ``nn.Parameter`` s fit by the same
Trainer. Deterministic. Canonical dynamic state ``[vx, vy, yaw_rate]``.
"""

import math

import torch
import torch.nn as nn

from ..car_parameters import CarParameters, load_car_parameters
from .base import Prediction, TrainableModel

_VX_MIN = 0.5  # m/s floor used in slip-angle denominators to stay stable at low speed


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, to initialize a positive parameter at value ``y``."""
    return math.log(math.expm1(max(y, 1e-6)))


class _BicycleParams(nn.Module):
    """Learnable positive parameters of the bicycle model (raw + softplus)."""

    def __init__(
        self,
        cf: float,
        cr: float,
        izz: float,
        steer_gain: float,
        throttle_gain: float,
        drag: float,
    ) -> None:
        super().__init__()
        self.raw_cf = nn.Parameter(torch.tensor(_inv_softplus(cf)))
        self.raw_cr = nn.Parameter(torch.tensor(_inv_softplus(cr)))
        self.raw_izz = nn.Parameter(torch.tensor(_inv_softplus(izz)))
        self.raw_steer = nn.Parameter(torch.tensor(_inv_softplus(steer_gain)))
        self.raw_throttle = nn.Parameter(torch.tensor(_inv_softplus(throttle_gain)))
        self.raw_drag = nn.Parameter(torch.tensor(_inv_softplus(drag)))

    @classmethod
    def placeholder(cls) -> "_BicycleParams":
        """Unit-valued params to receive a ``load_state_dict`` from a checkpoint."""
        return cls(cf=1.0, cr=1.0, izz=1.0, steer_gain=0.4, throttle_gain=1.0, drag=0.5)

    @property
    def cf(self):
        return torch.nn.functional.softplus(self.raw_cf)

    @property
    def cr(self):
        return torch.nn.functional.softplus(self.raw_cr)

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
    def drag(self):
        return torch.nn.functional.softplus(self.raw_drag)


class DynamicBicycleModel(TrainableModel):
    """Physics single-track bicycle; learnable tire stiffness, inertia, actuator maps."""

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
        # Box-approx yaw inertia; rough cornering stiffness from axle load.
        izz0 = cp.mass_total_kg * (cp.length_base_plate_m**2 + cp.track_width_mean_m**2) / 12.0
        cf0 = cp.front_axle_load_fraction * cp.mass_total_kg * g / 0.1
        cr0 = cp.rear_axle_load_fraction * cp.mass_total_kg * g / 0.1
        params = _BicycleParams(
            cf=cf0,
            cr=cr0,
            izz=max(izz0, 1e-3),
            steer_gain=0.4,
            throttle_gain=cp.mass_total_kg * 4.0,
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
        throttle = action[..., 1].clamp(-1.0, 1.0)
        p, m, a, b = self._p, self._m, self._a, self._b

        delta = p.steer_gain * steering
        vx_safe = vx.clamp(min=_VX_MIN)
        alpha_f = torch.atan2(vy + a * r, vx_safe) - delta
        alpha_r = torch.atan2(vy - b * r, vx_safe)
        fyf = -p.cf * alpha_f
        fyr = -p.cr * alpha_r
        fx = p.throttle_gain * throttle - p.drag * vx

        vx_dot = (fx - fyf * torch.sin(delta)) / m + vy * r
        vy_dot = (fyf * torch.cos(delta) + fyr) / m - vx * r
        r_dot = (a * fyf * torch.cos(delta) - b * fyr) / p.izz
        return Prediction(mean=torch.stack([vx_dot, vy_dot, r_dot], dim=-1), cov=None)

    def trainable_parameters(self):
        return self._p.parameters()

    def set_train_mode(self, mode: bool = True) -> None:
        self._p.train(mode)
