"""
Central registry mapping model-type names to their classes (lazily imported).

Lazy getters keep the torch-free light path intact: resolving ``"kinematic"`` /
``"arcade"`` does not import torch. Trainable (torch) models register themselves
here too — their getter imports torch only when called.
"""

from collections.abc import Callable

from .base import VehicleModel


def _kinematic() -> type[VehicleModel]:
    from .kinematic import KinematicBicycleModel

    return KinematicBicycleModel


def _arcade() -> type[VehicleModel]:
    from .arcade import ArcadeDriftModel

    return ArcadeDriftModel


def _old_standard() -> type[VehicleModel]:
    from .old.standard_vehicle import OldStandardVehicleModel

    return OldStandardVehicleModel


def _old_slip_angle() -> type[VehicleModel]:
    from .old.slip_angle_vehicle import OldSlipAngleVehicleModel

    return OldSlipAngleVehicleModel


def _neural() -> type[VehicleModel]:
    from .neural import NeuralDynamicsModel

    return NeuralDynamicsModel


def _dynamic() -> type[VehicleModel]:
    from .dynamic import DynamicBicycleModel

    return DynamicBicycleModel


def _hybrid() -> type[VehicleModel]:
    from .hybrid import PhysicsInformedModel

    return PhysicsInformedModel


# model_type -> zero-arg callable returning the class. Only implemented models are
# registered, so resolving an unknown name fails cleanly (never an ImportError).
MODEL_REGISTRY: dict[str, Callable[[], type[VehicleModel]]] = {
    "kinematic": _kinematic,
    "arcade": _arcade,
    "old_standard": _old_standard,
    "old_slip_angle": _old_slip_angle,
    "neural": _neural,
    "dynamic": _dynamic,
    "hybrid": _hybrid,
}


def get_model_class(model_type: str) -> type[VehicleModel]:
    """Return the model class registered under ``model_type``."""
    try:
        getter = MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Known: {sorted(MODEL_REGISTRY)}"
        ) from None
    return getter()
