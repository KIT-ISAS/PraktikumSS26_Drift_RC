"""
Central registry mapping model-type names to their classes (lazily imported).

Lazy getters keep the torch-free light path intact: resolving ``"kinematic"`` /
``"arcade"`` does not import torch. Trainable (torch) models register themselves
here too — their getter imports torch only when called. Further trainable models
(dynamic, hybrid) are added in later PRs.
"""

from collections.abc import Callable

from .base import VehicleModel


def _kinematic() -> type[VehicleModel]:
    from .kinematic import KinematicBicycleModel

    return KinematicBicycleModel


def _arcade() -> type[VehicleModel]:
    from .arcade import ArcadeDriftModel

    return ArcadeDriftModel


def _neural() -> type[VehicleModel]:
    from .neural import NeuralDynamicsModel

    return NeuralDynamicsModel


# model_type -> zero-arg callable returning the class. Only implemented models are
# registered, so resolving an unknown name fails cleanly (never an ImportError).
MODEL_REGISTRY: dict[str, Callable[[], type[VehicleModel]]] = {
    "kinematic": _kinematic,
    "arcade": _arcade,
    "neural": _neural,
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
