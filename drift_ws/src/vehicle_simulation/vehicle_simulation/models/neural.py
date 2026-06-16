"""Neural network dynamics model trained from rosbag driving data."""

import math

import numpy as np
import torch
import torch.nn as nn

from .base import VehicleAction, VehicleModel, VehicleState


class DynamicsNet(nn.Module):
    """MLP that maps [vx, vy, yaw_rate, steering, throttle] to body-frame deltas."""

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 3,
        hidden_dim: int = 256,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        # Stochastic head outputs (mu, log_std) → 2× output_dim
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


class NeuralDynamicsModel(VehicleModel):
    """
    Data-driven dynamics model: MLP predicting body-frame velocity deltas.

    Input:  [vx, vy, yaw_rate, steering, throttle]
    Output: [d(vx)/dt, d(vy)/dt, d(yaw_rate)/dt]  (normalized at train time)

    Position (x, y) and heading are integrated from the predicted body-frame
    velocities, so the model is rotation-equivariant.

    step() always returns the deterministic (mean) prediction.
    step_stochastic() samples from the learned distribution when the model was
    trained with stochastic=True; otherwise falls back to step().
    """

    def __init__(
        self,
        weights_path: str,
        stochastic: bool = False,
        device: str = "cpu",
    ) -> None:
        self._can_sample = stochastic
        self._device = torch.device(device)

        ckpt = torch.load(weights_path, map_location=self._device, weights_only=True)
        hidden_dim = ckpt.get("hidden_dim", 256)
        trained_stochastic = ckpt.get("stochastic", False)

        self._input_mean = torch.tensor(
            ckpt["input_mean"], device=self._device, dtype=torch.float32
        )
        self._input_std = torch.tensor(ckpt["input_std"], device=self._device, dtype=torch.float32)
        self._output_mean = torch.tensor(
            ckpt["output_mean"], device=self._device, dtype=torch.float32
        )
        self._output_std = torch.tensor(
            ckpt["output_std"], device=self._device, dtype=torch.float32
        )

        self._net = DynamicsNet(
            hidden_dim=hidden_dim,
            stochastic=trained_stochastic,
        )
        self._net.load_state_dict(ckpt["model_state_dict"])
        self._net.to(self._device)
        self._net.eval()

        # Can only sample if the model was trained with a stochastic head
        self._can_sample = stochastic and trained_stochastic

    def _predict_delta(
        self, state: VehicleState, action: VehicleAction, sample: bool
    ) -> np.ndarray:
        """Return [dvx, dvy, dyaw_rate] prediction per second."""
        x = torch.tensor(
            [state.vx, state.vy, state.yaw_rate, action.steering, action.throttle],
            dtype=torch.float32,
            device=self._device,
        )
        x_norm = (x - self._input_mean) / (self._input_std + 1e-8)

        with torch.no_grad():
            out = self._net(x_norm)

        if sample and self._net.stochastic:
            n = self._net.output_dim
            mu_norm = out[:n]
            log_std = out[n:].clamp(-4.0, 2.0)
            std = torch.exp(log_std)
            delta_norm = mu_norm + std * torch.randn_like(std)
        else:
            # Use only the mean (first half if stochastic head, all if deterministic)
            delta_norm = out[: self._net.output_dim]

        delta = delta_norm * self._output_std + self._output_mean
        return delta.cpu().numpy()

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        delta = self._predict_delta(state, action, sample=False)
        return self._integrate(state, delta, dt)

    def step_stochastic(
        self, state: VehicleState, action: VehicleAction, dt: float
    ) -> VehicleState:
        delta = self._predict_delta(state, action, sample=self._can_sample)
        return self._integrate(state, delta, dt)

    @staticmethod
    def _integrate(state: VehicleState, delta: np.ndarray, dt: float) -> VehicleState:
        """Euler-integrate body-frame deltas into a new VehicleState."""
        dvx, dvy, dyaw_rate = float(delta[0]), float(delta[1]), float(delta[2])

        vx = state.vx + dvx * dt
        vy = state.vy + dvy * dt
        yaw_rate = state.yaw_rate + dyaw_rate * dt
        heading = state.heading + yaw_rate * dt

        # Body-frame → world-frame position update
        cos_h = math.cos(state.heading)
        sin_h = math.sin(state.heading)
        x = state.x + (cos_h * vx - sin_h * vy) * dt
        y = state.y + (sin_h * vx + cos_h * vy) * dt

        return VehicleState(x=x, y=y, heading=heading, vx=vx, vy=vy, yaw_rate=yaw_rate)
