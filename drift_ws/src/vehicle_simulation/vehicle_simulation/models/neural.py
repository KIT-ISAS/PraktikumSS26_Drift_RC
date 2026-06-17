"""Neural network dynamics model trained from rosbag driving data."""

import torch
import torch.nn as nn

from .base import Prediction, TrainableModel

_EPS = 1e-8


class DynamicsNet(nn.Module):
    """MLP mapping [vx, vy, yaw_rate, steering, throttle] to body-frame derivatives."""

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 3,
        hidden_dim: int = 256,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        out_dim = output_dim * 2 if stochastic else output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.stochastic = stochastic
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralDynamicsModel(TrainableModel):
    """
    Data-driven dynamics model: an MLP predicting body-frame velocity derivatives.

    Input  [vx, vy, yaw_rate, steering, throttle]; output the per-second derivative
    of [vx, vy, yaw_rate]. :meth:`derivative` is the differentiable, batched source
    of truth used by training; ``step`` (inherited from :class:`TrainableModel`)
    integrates it for numpy inference. The stochastic variant adds a per-dim
    log-std head and returns a per-second covariance.
    """

    def __init__(
        self,
        net: DynamicsNet,
        normalization: dict,
        *,
        stochastic: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self._net = net.to(self._device)
        self._hidden_dim = net.net[0].out_features
        self._can_sample = bool(stochastic and net.stochastic)
        self._norm = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self._device)
            for k, v in normalization.items()
        }

    # --- construction -------------------------------------------------------

    @classmethod
    def new(
        cls,
        normalization: dict,
        *,
        hidden_dim: int = 256,
        stochastic: bool = False,
        device: str = "cpu",
    ) -> "NeuralDynamicsModel":
        """Build a fresh (untrained) model for training."""
        net = DynamicsNet(hidden_dim=hidden_dim, stochastic=stochastic)
        return cls(net, normalization, stochastic=stochastic, device=device)

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "NeuralDynamicsModel":
        """Rebuild a trained model from a checkpoint dict (run dir or legacy .pt)."""
        hidden_dim = ckpt.get("hidden_dim", 256)
        trained_stochastic = ckpt.get("stochastic", False)
        net = DynamicsNet(hidden_dim=hidden_dim, stochastic=trained_stochastic)
        net.load_state_dict(ckpt["model_state_dict"])
        normalization = {
            "input_mean": ckpt["input_mean"],
            "input_std": ckpt["input_std"],
            "output_mean": ckpt["output_mean"],
            "output_std": ckpt["output_std"],
        }
        return cls(net, normalization, stochastic=stochastic and trained_stochastic, device=device)

    @classmethod
    def from_weights(
        cls, weights_path: str, device: str = "cpu", stochastic: bool = False
    ) -> "NeuralDynamicsModel":
        """Load from a bare ``.pt`` checkpoint path (back-compat with build_model)."""
        # weights_only=True: checkpoints are plain dicts of tensors + primitive
        # metadata, so the restricted unpickler suffices and avoids the
        # arbitrary-code-execution risk of full pickle.
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        return cls.from_checkpoint(ckpt, device=device, stochastic=stochastic)

    def to_checkpoint(self) -> dict:
        """Return a self-contained checkpoint dict for saving in a run dir."""
        return {
            "model_type": "neural",
            "model_state_dict": self._net.state_dict(),
            "hidden_dim": self._hidden_dim,
            "stochastic": self._net.stochastic,
            "input_mean": self._norm["input_mean"].cpu().numpy().tolist(),
            "input_std": self._norm["input_std"].cpu().numpy().tolist(),
            "output_mean": self._norm["output_mean"].cpu().numpy().tolist(),
            "output_std": self._norm["output_std"].cpu().numpy().tolist(),
        }

    # --- derivative (source of truth) --------------------------------------

    def derivative(self, dynamic: torch.Tensor, action: torch.Tensor) -> Prediction:
        """Per-second derivative of [vx, vy, yaw_rate] (+ per-second covariance)."""
        x = torch.cat([dynamic, action], dim=-1)
        x_norm = (x - self._norm["input_mean"]) / (self._norm["input_std"] + _EPS)
        out = self._net(x_norm)

        if self._net.stochastic:
            mu_norm = out[..., : self._net.output_dim]
            log_std_norm = out[..., self._net.output_dim :].clamp(-4.0, 2.0)
        else:
            mu_norm, log_std_norm = out, None

        deriv = mu_norm * self._norm["output_std"] + self._norm["output_mean"]
        cov = None
        if log_std_norm is not None:
            sigma = torch.exp(log_std_norm) * self._norm["output_std"]
            cov = sigma**2  # per-second covariance (framework scales by dt)
        return Prediction(mean=deriv, cov=cov)

    def trainable_parameters(self):
        return self._net.parameters()

    def set_train_mode(self, mode: bool = True) -> None:
        self._net.train(mode)
