"""
Vehicle dynamics models and a factory to build them by name.

Concrete models are imported lazily inside :func:`build_model` so that using a
lightweight model (e.g. kinematic) does not pull in heavy optional dependencies
such as ``torch`` (only the neural model needs it).
"""

from .base import VehicleAction, VehicleModel, VehicleState

__all__ = ["VehicleAction", "VehicleModel", "VehicleState", "build_model"]


def build_model(
    model_type: str,
    config: str | None = None,
    weights: str | None = None,
    **kwargs,
) -> VehicleModel:
    """
    Build a vehicle model by name.

    Args:
        model_type: ``"kinematic"`` or ``"neural"``.
        config: optional YAML params path for the kinematic model.
        weights: ``.pt`` checkpoint path; required for the neural model.
        kwargs: forwarded to the concrete model constructor.

    Returns:
        A ready-to-use :class:`VehicleModel` instance.

    """
    if model_type == "kinematic":
        from .kinematic import KinematicBicycleModel

        if config:
            return KinematicBicycleModel.from_config(config)
        return KinematicBicycleModel(**kwargs)

    if model_type == "neural":
        if not weights:
            raise ValueError("'weights' is required for model_type='neural'")
        from .neural import NeuralDynamicsModel

        return NeuralDynamicsModel(weights, **kwargs)

    raise ValueError(f"Unknown model_type: '{model_type}'")
