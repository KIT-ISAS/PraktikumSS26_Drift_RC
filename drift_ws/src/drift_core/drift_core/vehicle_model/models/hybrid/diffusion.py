"""The paper's learned diffusion Sigma(x, u) — the neural-SDE head."""

import math

import torch
import torch.nn as nn

from ._nn import _mlp

# Network shape (paper Sec. IV-A): "Sigma is a diagonal matrix parametrized by a neural
# network with two hidden layers of 16 neurons each and tanh activations."
_DIFF_HIDDEN, _DIFF_LAYERS = 16, 2

# Diffusion log-std clamp: e^-4 ~ 0.02 to e^2 ~ 7.4 (m/s^2 resp. rad/s^2) per-second std.
_LOG_STD_RANGE = (-4.0, 2.0)
_LOG_STD_INIT = math.log(0.5)


class _DiffusionNet(nn.Module):
    """
    The paper's learned diffusion Sigma(x, u).

    A per-second diagonal variance for the chassis derivative ``[vx_dot, vy_dot, r_dot]``.

    This is what makes the hybrid the paper's neural *SDE* rather than an ODE: trained
    with ``loss="nll"`` it learns *where* the physics is trustworthy (the framework
    scales the variance by dt, Euler–Maruyama style), and sampling from it during RL
    training is the paper's stated substitute for domain randomization. Zero-initialized
    weights with a constant bias, so the initial noise is state-independent and NLL
    calibrates it from the residuals.
    """

    def __init__(self, hidden: int = _DIFF_HIDDEN, layers: int = _DIFF_LAYERS) -> None:
        super().__init__()
        self.net = _mlp(6, hidden, layers, 3)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, _LOG_STD_INIT)

    def forward(self, dyn3: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-second *variance* (B, 3) from the chassis state and resolved action."""
        log_std = self.net(torch.cat([dyn3, action], dim=-1)).clamp(*_LOG_STD_RANGE)
        return torch.exp(2.0 * log_std)
