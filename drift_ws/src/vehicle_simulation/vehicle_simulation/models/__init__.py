"""
Vehicle dynamics models and a factory to build them by name, path, or run dir.

Concrete models are imported lazily inside :func:`build_model` (and the registry)
so that using a lightweight model (e.g. kinematic) does not pull in heavy optional
dependencies such as ``torch`` (only the learned models need it).
"""

from pathlib import Path

from .base import (
    Prediction,
    TrainableModel,
    VehicleAction,
    VehicleModel,
    VehicleState,
)
from .esc import EscStateMachine, MotorCommand

__all__ = [
    "VehicleAction",
    "VehicleModel",
    "TrainableModel",
    "VehicleState",
    "Prediction",
    "EscStateMachine",
    "MotorCommand",
    "build_model",
]


def _is_run_dir_or_checkpoint(value: str) -> bool:
    """True if ``value`` looks like a run directory (has model.pt) or a .pt file."""
    p = Path(value)
    return (p.is_dir() and (p / "model.pt").exists()) or (p.suffix == ".pt" and p.exists())


def build_model(
    model_type: str,
    config: str | None = None,
    weights: str | None = None,
    run_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
) -> VehicleModel:
    """
    Build a vehicle model by registry name, run directory, or checkpoint path.

    Resolution order (mirrors ``load_track`` / ``load_car_parameters``):

    1. ``run_dir`` given, or ``model_type`` points at a run dir / ``.pt`` file
       -> load the trained model via :func:`models.loader.load_model_from_run`.
    2. ``model_type`` is a registry name. Basic models: ``"kinematic"``,
       ``"arcade"``, and the predecessor ports ``"old_standard"`` /
       ``"old_slip_angle"`` (all accept an optional YAML ``config``). The
       trainable ``"neural"`` model is built from a ``.pt`` checkpoint
       (``weights``); the remaining trainable models (``"dynamic"``,
       ``"hybrid"``) are added in later PRs.

    Args:
        model_type: registry name, run-directory path, or ``.pt`` path.
        config: optional YAML params path (kinematic / arcade).
        weights: ``.pt`` checkpoint path (trained models; back-compat).
        run_dir: trained-model run directory.
        device: torch device for learned models.
        kwargs: forwarded to the concrete model constructor.

    """
    stochastic = kwargs.pop("stochastic", False)

    if run_dir is not None:
        from .loader import load_model_from_run

        return load_model_from_run(run_dir, device=device, stochastic=stochastic)

    if _is_run_dir_or_checkpoint(model_type):
        from .loader import load_model_from_run

        return load_model_from_run(model_type, device=device, stochastic=stochastic)

    if model_type == "kinematic":
        from .kinematic import KinematicBicycleModel

        if config:
            return KinematicBicycleModel.from_config(config)
        return KinematicBicycleModel(**kwargs)

    if model_type == "arcade":
        from .arcade import ArcadeDriftModel

        if config:
            return ArcadeDriftModel.from_config(config)
        return ArcadeDriftModel(**kwargs)

    if model_type == "old_standard":
        from .old.standard_vehicle import OldStandardVehicleModel

        if config:
            return OldStandardVehicleModel.from_config(config)
        return OldStandardVehicleModel(**kwargs)

    if model_type == "old_slip_angle":
        from .old.slip_angle_vehicle import OldSlipAngleVehicleModel

        if config:
            return OldSlipAngleVehicleModel.from_config(config)
        return OldSlipAngleVehicleModel(**kwargs)

    if model_type == "neural":
        # Validate before importing: .neural needs torch, and a missing checkpoint
        # must surface as this ValueError, not as torch's ImportError.
        if not weights:
            raise ValueError(
                "build_model('neural', ...) needs a trained checkpoint: pass "
                "weights='<path.pt>' or a run_dir / run-dir path."
            )
        from .neural import NeuralDynamicsModel

        return NeuralDynamicsModel.from_weights(weights, device=device, stochastic=stochastic)

    # Remaining trainable (torch) models — dynamic / hybrid — register their own
    # build_model branches in the PRs that add them.
    raise ValueError(f"Unknown model_type: {model_type!r}")
