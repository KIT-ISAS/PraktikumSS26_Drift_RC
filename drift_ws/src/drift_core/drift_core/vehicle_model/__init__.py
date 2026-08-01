"""
Everything about the car itself: dynamics models, data, training, evaluation.

The model interface and the :func:`build_model` factory are re-exported here so
consumers (e.g. ``simulation_pkg``) import from ``drift_core.vehicle_model``
without reaching into :mod:`.models`. The measured car geometry lives in
:mod:`.car_parameters` (packaged YAML under ``cars/``).
"""

from .models import (
    EscStateMachine,
    MotorCommand,
    Prediction,
    TrainableModel,
    VehicleAction,
    VehicleModel,
    VehicleState,
    build_model,
)

__all__ = [
    "EscStateMachine",
    "MotorCommand",
    "Prediction",
    "TrainableModel",
    "VehicleAction",
    "VehicleModel",
    "VehicleState",
    "build_model",
]
