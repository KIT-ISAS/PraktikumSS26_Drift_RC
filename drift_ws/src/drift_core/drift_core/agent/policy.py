"""
DriftAgent — ROS-free inference wrapper around a trained SAC run directory.

This is what the future deployment ROS node calls at the control rate: build the
observation with :mod:`drift_core.agent.environments.observation` (the shared
sim-to-real builder) and map it to a :class:`VehicleAction` here. Loading needs
``stable-baselines3`` (the ``[rl]`` extra), imported lazily.
"""

from pathlib import Path

import numpy as np

from ..vehicle_model import VehicleAction
from .environments.observation import action_from_array


class DriftAgent:
    """Deterministic policy loaded from a training run directory (or a SAC ``.zip``)."""

    def __init__(self, run_dir, device: str = "cpu", prefer_best: bool = True) -> None:
        """
        Load ``best_model.zip`` (falling back to ``agent.zip``) from ``run_dir``.

        ``run_dir`` may also point directly at a SAC ``.zip`` file. Set
        ``prefer_best=False`` to load the final ``agent.zip`` even when a best
        eval checkpoint exists.
        """
        from stable_baselines3 import SAC

        path = Path(run_dir)
        if path.is_dir():
            candidates = ["best_model.zip", "agent.zip"] if prefer_best else ["agent.zip"]
            for name in candidates:
                if (path / name).is_file():
                    path = path / name
                    break
            else:
                raise FileNotFoundError(
                    f"No agent checkpoint (best_model.zip/agent.zip) in {run_dir}"
                )
        self.path = path
        self._model = SAC.load(str(path), device=device)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Map one observation to a ``[steering, throttle]`` array in [-1, 1].

        ``deterministic=True`` (the deployment default) uses the SAC policy mean;
        pass ``deterministic=False`` to sample the squashed-Gaussian policy for
        exploration (used when collecting real transitions for fine-tuning).
        """
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def act(self, obs: np.ndarray) -> VehicleAction:
        """Map one observation to a :class:`VehicleAction`."""
        return action_from_array(self.predict(obs))
