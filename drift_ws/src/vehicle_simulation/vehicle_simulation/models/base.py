"""Base abstractions for vehicle state, actions, and dynamics models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    """Full vehicle state in world and body frames."""

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


@dataclass
class VehicleAction:
    """Control inputs to the vehicle."""

    steering: float = 0.0  # [-1, 1]: negative = left, positive = right
    throttle: float = 0.0  # [-1, 1]: negative = brake/reverse, positive = forward

    def to_array(self) -> np.ndarray:
        """Return action as [steering, throttle]."""
        return np.array([self.steering, self.throttle], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "VehicleAction":
        """Construct from a 2-element array [steering, throttle]."""
        return VehicleAction(steering=float(arr[0]), throttle=float(arr[1]))


class VehicleModel(ABC):
    """Abstract base for vehicle dynamics models."""

    @abstractmethod
    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        """Return deterministic next state after applying action for dt seconds."""

    def step_stochastic(
        self, state: VehicleState, action: VehicleAction, dt: float
    ) -> VehicleState:
        """Return stochastic next state. Default: same as step()."""
        return self.step(state, action, dt)

    @classmethod
    def from_config(cls, path: str) -> "VehicleModel":
        """Load a model from a YAML config file."""
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg.get("params", {}))
