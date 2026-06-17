"""
Physics-informed hybrid model: dynamic bicycle + learned neural residual.

The per-second derivative is the bicycle derivative plus a neural residual that
absorbs effects not captured by the physics (tire nonlinearity, actuator maps,
unmodeled dynamics). The residual net's last layer is zero-initialized, so the
model starts as pure physics and learns the correction. Optionally stochastic
(per-dim log-std head -> per-second covariance). Canonical dynamic state.
"""

import torch

from ..car_parameters import load_car_parameters
from .base import Prediction, TrainableModel
from .dynamic import DynamicBicycleModel, _BicycleParams
from .neural import DynamicsNet

_EPS = 1e-8


class PhysicsInformedModel(TrainableModel):
    """Dynamic-bicycle derivative + neural residual (the likely end-state model)."""

    def __init__(
        self,
        physics: DynamicBicycleModel,
        residual: DynamicsNet,
        normalization: dict,
        *,
        stochastic: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self._physics = physics
        self._residual = residual.to(self._device)
        self._hidden_dim = residual.net[0].out_features
        self._can_sample = bool(stochastic and residual.stochastic)
        self._in_mean = torch.as_tensor(
            normalization["input_mean"], dtype=torch.float32, device=self._device
        )
        self._in_std = torch.as_tensor(
            normalization["input_std"], dtype=torch.float32, device=self._device
        )

    # --- construction -------------------------------------------------------

    @classmethod
    def new(
        cls,
        normalization: dict,
        *,
        car="drift_rc",
        hidden_dim: int = 64,
        stochastic: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> "PhysicsInformedModel":
        """Build physics + a zero-initialized residual net (starts as pure physics)."""
        physics = DynamicBicycleModel.new(car=car, device=device)
        residual = DynamicsNet(hidden_dim=hidden_dim, stochastic=stochastic)
        last = residual.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        return cls(physics, residual, normalization, stochastic=stochastic, device=device)

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "PhysicsInformedModel":
        cp = load_car_parameters(ckpt.get("car", "drift_rc"))
        params = _BicycleParams.placeholder()
        params.load_state_dict(ckpt["params_state"])
        physics = DynamicBicycleModel(cp, params, device=device)
        trained_stochastic = ckpt.get("stochastic", False)
        residual = DynamicsNet(
            hidden_dim=ckpt.get("hidden_dim", 64), stochastic=trained_stochastic
        )
        residual.load_state_dict(ckpt["residual_state"])
        normalization = {"input_mean": ckpt["input_mean"], "input_std": ckpt["input_std"]}
        return cls(
            physics,
            residual,
            normalization,
            stochastic=stochastic and trained_stochastic,
            device=device,
        )

    def to_checkpoint(self) -> dict:
        return {
            "model_type": "hybrid",
            "car": self._physics._car.name,
            "params_state": self._physics._p.state_dict(),
            "residual_state": self._residual.state_dict(),
            "hidden_dim": self._hidden_dim,
            "stochastic": self._residual.stochastic,
            "input_mean": self._in_mean.cpu().numpy().tolist(),
            "input_std": self._in_std.cpu().numpy().tolist(),
        }

    # --- derivative (source of truth) --------------------------------------

    def derivative(self, dynamic: torch.Tensor, action: torch.Tensor) -> Prediction:
        """Bicycle derivative + neural residual; per-second covariance if stochastic."""
        deriv_phys = self._physics.derivative(dynamic, action).mean
        x = torch.cat([dynamic, action], dim=-1)
        x_norm = (x - self._in_mean) / (self._in_std + _EPS)
        out = self._residual(x_norm)

        if self._residual.stochastic:
            mu = out[..., : self._residual.output_dim]
            log_std = out[..., self._residual.output_dim :].clamp(-4.0, 2.0)
        else:
            mu, log_std = out, None

        deriv = deriv_phys + mu
        cov = (torch.exp(log_std) ** 2) if log_std is not None else None
        return Prediction(mean=deriv, cov=cov)

    def trainable_parameters(self):
        return list(self._physics.trainable_parameters()) + list(self._residual.parameters())

    def set_train_mode(self, mode: bool = True) -> None:
        self._physics.set_train_mode(mode)
        self._residual.train(mode)
