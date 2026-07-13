"""Base abstractions for vehicle state, actions, and dynamics models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, ClassVar, NamedTuple

import numpy as np

from .esc import EscStateMachine, MotorCommand


@dataclass
class VehicleState:
    """Full vehicle state in world and body frames (the one canonical state)."""

    x: float = 0.0  # m, world frame
    y: float = 0.0  # m, world frame
    heading: float = 0.0  # rad, world frame (0 = +x direction)
    vx: float = 0.0  # m/s, body frame (forward)
    vy: float = 0.0  # m/s, body frame (lateral, + = left)
    yaw_rate: float = 0.0  # rad/s (+ = CCW)

    def to_array(self) -> np.ndarray:
        """Return state as [x, y, heading, vx, vy, yaw_rate]."""
        return np.array(
            [self.x, self.y, self.heading, self.vx, self.vy, self.yaw_rate], dtype=np.float32
        )

    @staticmethod
    def from_array(arr: np.ndarray) -> "VehicleState":
        """Construct from a 6-element array [x, y, heading, vx, vy, yaw_rate]."""
        return VehicleState(
            x=float(arr[0]),
            y=float(arr[1]),
            heading=float(arr[2]),
            vx=float(arr[3]),
            vy=float(arr[4]),
            yaw_rate=float(arr[5]),
        )

    def body(self) -> np.ndarray:
        """Return the body-frame dynamic vector [vx, vy, yaw_rate] (what models predict)."""
        return np.array([self.vx, self.vy, self.yaw_rate], dtype=np.float32)

    @property
    def speed(self) -> float:
        """Speed V [m/s]: magnitude of the body-frame velocity."""
        return math.hypot(self.vx, self.vy)

    @property
    def sideslip(self) -> float:
        """Sideslip angle β [rad]: angle of the velocity vector relative to heading."""
        return math.atan2(self.vy, self.vx) if (self.vx or self.vy) else 0.0


@dataclass
class VehicleAction:
    """Control inputs to the vehicle."""

    steering: float = 0.0  # [-1, 1]: positive = left (CCW, +yaw_rate), negative = right
    throttle: float = 0.0  # [-1, 1]: negative = brake/reverse, positive = forward

    def to_array(self) -> np.ndarray:
        """Return action as [steering, throttle]."""
        return np.array([self.steering, self.throttle], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "VehicleAction":
        """Construct from a 2-element array [steering, throttle]."""
        return VehicleAction(steering=float(arr[0]), throttle=float(arr[1]))


class Prediction(NamedTuple):
    """
    A trainable model's per-second derivative prediction of ``[vx, vy, yaw_rate]``.

    ``mean`` is the derivative (``(D,)`` or batched ``(B, D)``; numpy or torch).
    ``cov`` is the per-second (diagonal) covariance for stochastic models, or
    ``None`` for deterministic ones. The framework applies the timestep: the next
    state is ``state + mean·dt`` and the increment covariance is ``cov·dt``
    (Euler / Euler–Maruyama), so models stay timestep-agnostic.
    """

    mean: Any
    cov: Any | None = None


class VehicleModel(ABC):
    """
    Abstract base for vehicle dynamics models (numpy inference interface).

    ``step`` / ``step_stochastic`` first run the throttle pedal through the ESC
    input stage (:class:`~.esc.EscStateMachine`) — resolving the single
    brake/reverse axis into an unambiguous :class:`~.esc.MotorCommand`, exactly
    like the real car's ESC — and then advance the dynamics via the model's
    :meth:`_step_motor` hook. Models therefore implement pure dynamics and never
    see an ambiguous negative throttle. Because the ESC (like the model-internal
    slip state) carries across steps, one instance must be stepped in sequence;
    ``DriftEnv`` builds a fresh model per episode.
    """

    #: Whether constructing/using this model requires torch.
    requires_torch: ClassVar[bool] = False

    @property
    def esc(self) -> EscStateMachine:
        """The ESC input stage in front of this model (per instance, lazily created)."""
        esc = getattr(self, "_esc", None)
        if esc is None:
            esc = EscStateMachine()
            self._esc = esc
        return esc

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        """Return deterministic next state after applying action for dt seconds."""
        motor = self.esc.resolve(action.throttle, state.vx)
        return self._step_motor(state, action.steering, motor, dt)

    def step_stochastic(
        self, state: VehicleState, action: VehicleAction, dt: float
    ) -> VehicleState:
        """Return stochastic next state. Default: same dynamics as step()."""
        motor = self.esc.resolve(action.throttle, state.vx)
        return self._step_motor_stochastic(state, action.steering, motor, dt)

    @abstractmethod
    def _step_motor(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        """Advance the dynamics one step from the ESC-resolved motor command."""

    def _step_motor_stochastic(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        """Stochastic variant of :meth:`_step_motor`. Default: deterministic."""
        return self._step_motor(state, steering, motor, dt)

    @classmethod
    def from_config(cls, path: str) -> "VehicleModel":
        """Load a model from a YAML config file (``params:`` mapping)."""
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg.get("params", {}))


class TrainableModel(VehicleModel):
    """
    Torch model that predicts the per-second derivative of ``[vx, vy, yaw_rate]``.

    Subclasses implement :meth:`derivative` (timestep-free) plus the training and
    checkpoint hooks. ``_step_motor`` / ``_step_motor_stochastic`` are provided
    here: they run one ``derivative`` under ``no_grad`` and integrate the body
    velocities (Euler, plus Euler–Maruyama noise when sampling a stochastic model)
    and the pose. The training rollout integrates the same ``derivative`` in the
    same way, so training and inference share one contract.
    """

    requires_torch: ClassVar[bool] = True

    def __init__(self, device: str = "cpu") -> None:
        import torch

        self._device = torch.device(device)
        self._can_sample = False

    @abstractmethod
    def derivative(self, dynamic, action) -> Prediction:
        """
        Per-second derivative of ``[vx, vy, yaw_rate]`` (timestep-free).

        ``dynamic`` ``(B, 3)``, ``action`` ``(B, 3)`` of ``[steering, drive,
        brake]`` — the ESC-*resolved* motor command, not the raw pedal — as torch
        tensors. Training data must be labeled the same way: replay the recorded
        throttle stream through :class:`~.esc.EscStateMachine` (raw negative
        throttle is ambiguous between brake and reverse; the resolved command is
        Markov). Deterministic models return ``Prediction(mean, None)``;
        stochastic models also return a per-second covariance ``cov``.
        """

    @abstractmethod
    def trainable_parameters(self):
        """Return the torch parameters to optimize."""

    def set_train_mode(self, mode: bool = True) -> None:
        """Put the underlying net(s) in train/eval mode (override if needed)."""

    @abstractmethod
    def to_checkpoint(self) -> dict:
        """Return a self-contained checkpoint dict (must include ``model_type``)."""

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: str = "cpu", stochastic: bool = False
    ) -> "TrainableModel":
        """Reconstruct a trained model from a loaded checkpoint dict."""
        raise NotImplementedError(f"{cls.__name__} cannot be loaded from a checkpoint")

    # --- shared numpy inference (integrates derivative + pose) ---------------

    def _step_motor(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        return self._integrate(state, steering, motor, dt, sample=False)

    def _step_motor_stochastic(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float
    ) -> VehicleState:
        return self._integrate(state, steering, motor, dt, sample=self._can_sample)

    def _integrate(
        self, state: VehicleState, steering: float, motor: MotorCommand, dt: float, sample: bool
    ) -> VehicleState:
        import torch

        from .integrate import integrate_pose

        dynamics = torch.tensor(
            [[state.vx, state.vy, state.yaw_rate]], dtype=torch.float32, device=self._device
        )
        action_t = torch.tensor(
            [[steering, motor.drive, motor.brake]], dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            pred = self.derivative(dynamics, action_t)
            new_dynamics = dynamics + pred.mean * dt
            if sample and pred.cov is not None:
                new_dynamics = new_dynamics + torch.sqrt(
                    pred.cov.clamp_min(0.0) * dt
                ) * torch.randn_like(new_dynamics)
        new_dynamics = new_dynamics[0].cpu().numpy()
        # Semi-implicit Euler: advance the pose with the *updated* velocities so the
        # returned pose and velocities are consistent (one integration of the step).
        pose = integrate_pose(
            np.array([state.x, state.y, state.heading], dtype=np.float64),
            float(new_dynamics[0]),
            float(new_dynamics[1]),
            float(new_dynamics[2]),
            float(dt),
        )
        return VehicleState(
            x=float(pose[0]),
            y=float(pose[1]),
            heading=float(pose[2]),
            vx=float(new_dynamics[0]),
            vy=float(new_dynamics[1]),
            yaw_rate=float(new_dynamics[2]),
        )
