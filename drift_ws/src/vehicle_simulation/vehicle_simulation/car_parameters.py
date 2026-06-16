"""
Measured physical parameters of the RC car, loadable by models and humans.

The measurements live in packaged YAML files under ``cars/`` (e.g.
``cars/drift_rc.yaml``); this module loads one into a typed
:class:`CarParameters` dataclass and exposes the *derived* quantities a vehicle
model typically needs (tire radius, total gear reduction, longitudinal centre
of mass, axle load split) as properties so they cannot drift out of sync with
the raw measurements.

Example::

    from vehicle_simulation.car_parameters import load_car_parameters

    car = load_car_parameters("drift_rc")
    car.wheelbase_m                      # measured
    car.tire_radius_m                    # derived
    car.com_distance_from_front_axle_m   # derived from axle loads
"""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

__all__ = ["CarParameters", "load_car_parameters", "list_cars"]


@dataclass(frozen=True)
class CarParameters:
    """
    Measured car properties (SI units) plus derived dynamics quantities.

    Field values mirror the packaged YAML one-to-one; ``@property`` members are
    computed from them. See ``cars/drift_rc.yaml`` for the documented source
    measurements and the coordinate convention (x forward, y left).
    """

    name: str
    description: str

    # Chassis geometry [m]
    wheelbase_m: float
    track_width_front_m: float
    track_width_rear_m: float

    # Tires [m]
    tire_diameter_m: float
    tire_width_full_m: float
    tire_width_contact_m: float

    # Drivetrain
    primary_reduction_teeth: tuple[int, int]
    secondary_reduction: float

    # Vive tracker mounting [m]
    tracker_offset_from_front_axle_m: float
    tracker_height_m: float

    # Mass [kg]
    mass_total_kg: float
    mass_front_axle_kg: float
    mass_rear_axle_kg: float

    # Bounding dimensions [m]
    length_total_m: float
    length_without_cables_m: float
    length_base_plate_m: float
    height_with_cover_mount_m: float
    height_without_cover_mount_m: float

    # --- Derived quantities -------------------------------------------------

    @property
    def tire_radius_m(self) -> float:
        """Rolling radius of a tire [m]."""
        return self.tire_diameter_m / 2.0

    @property
    def track_width_mean_m(self) -> float:
        """Mean of the front and rear track widths [m]."""
        return (self.track_width_front_m + self.track_width_rear_m) / 2.0

    @property
    def total_gear_reduction(self) -> float:
        """Overall reduction from motor shaft to wheel (output/input)."""
        pinion, spur = self.primary_reduction_teeth
        return (spur / pinion) * self.secondary_reduction

    @property
    def axle_load_total_kg(self) -> float:
        """Sum of the measured axle loads [kg] (may differ slightly from mass_total_kg)."""
        return self.mass_front_axle_kg + self.mass_rear_axle_kg

    @property
    def front_axle_load_fraction(self) -> float:
        """Fraction of weight on the front axle, from the axle-load measurements."""
        return self.mass_front_axle_kg / self.axle_load_total_kg

    @property
    def rear_axle_load_fraction(self) -> float:
        """Fraction of weight on the rear axle, from the axle-load measurements."""
        return self.mass_rear_axle_kg / self.axle_load_total_kg

    @property
    def com_distance_from_front_axle_m(self) -> float:
        """Longitudinal centre-of-mass distance behind the front axle [m]."""
        return self.rear_axle_load_fraction * self.wheelbase_m

    @property
    def com_distance_from_rear_axle_m(self) -> float:
        """Longitudinal centre-of-mass distance ahead of the rear axle [m]."""
        return self.front_axle_load_fraction * self.wheelbase_m

    @property
    def tracker_offset_from_com_m(self) -> float:
        """Tracker offset from the centre of mass [m]; positive = ahead of the CoM."""
        return self.com_distance_from_front_axle_m - self.tracker_offset_from_front_axle_m


def _cars_dir():
    """Return the importlib traversable for the packaged ``cars/`` directory."""
    return files("vehicle_simulation").joinpath("cars")


def _from_dict(data: dict) -> CarParameters:
    """Build :class:`CarParameters` from a parsed YAML mapping."""
    data = dict(data)
    if "primary_reduction_teeth" in data:
        data["primary_reduction_teeth"] = tuple(data["primary_reduction_teeth"])
    return CarParameters(**data)


def load_car_parameters(name_or_path: str = "drift_rc") -> CarParameters:
    """
    Load car parameters by packaged name (e.g. ``"drift_rc"``) or YAML path.

    A value ending in ``.yaml`` or pointing at an existing file is read
    directly; anything else is resolved against the packaged ``cars/`` directory.
    """
    p = Path(name_or_path)
    if p.suffix == ".yaml" or p.exists():
        return _from_dict(yaml.safe_load(p.read_text()))
    resource = _cars_dir().joinpath(f"{name_or_path}.yaml")
    return _from_dict(yaml.safe_load(resource.read_text()))


def list_cars() -> list[str]:
    """Return the names of all cars bundled in the packaged ``cars/`` directory."""
    names = [p.name for p in _cars_dir().iterdir() if p.name.endswith(".yaml")]
    return sorted(n[: -len(".yaml")] for n in names)
