"""
Physics-informed hybrid model (Djeumou et al., arXiv:2410.20990), split by component.

- :mod:`.model` — the bicycle force balance, latent-state integration, and checkpoint I/O
  (:class:`PhysicsInformedModel`).
- :mod:`.tire` — the ExpTanh tire curve and the learned friction-circle direction net.
- :mod:`.esc` — the ESC/wheel torque balance and the learnable scalar parameters.
- :mod:`.diffusion` — the neural-SDE diffusion Sigma head.

The public import ``from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel`` is
preserved, and ``_WheelParams`` is re-exported for post-fit diagnostics.
"""

from .esc import _WheelParams
from .model import PhysicsInformedModel

__all__ = ["PhysicsInformedModel", "_WheelParams"]
