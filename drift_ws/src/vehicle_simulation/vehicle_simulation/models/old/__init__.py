"""Predecessor (WS 25/26) vehicle models, adapted to the ``VehicleModel`` interface."""

from .slip_angle_vehicle import OldSlipAngleVehicleModel
from .standard_vehicle import OldStandardVehicleModel

__all__ = ["OldSlipAngleVehicleModel", "OldStandardVehicleModel"]
