"""
Measured physical parameters of the RC car, loadable by models and humans.

The measurements live in packaged YAML files under ``cars/`` (e.g.
``cars/drift_rc.yaml``); this module loads one into a typed
:class:`CarParameters` dataclass and exposes the *derived* quantities a vehicle
model typically needs (tire radius, total gear reduction, total mass, axle load
split, centre-of-gravity position) as properties so they cannot drift out of
sync with the raw measurements.

Positions are expressed as ``(x, y, z)`` coordinates in the **vehicle frame**:
origin at the **front-axle centre**, ``x`` forward, ``y`` left, ``z`` up (ISO
8855). Points behind the front axle therefore have a **negative** ``x``. Using
coordinates rather than scalar "offset"/"distance" values keeps signs
unambiguous (a recurring source of bugs).

Example::

    from vehicle_simulation.car_parameters import load_car_parameters

    car = load_car_parameters("drift_rc")
    car.wheelbase_m                       # measured
    car.tire_radius_m                     # derived
    car.cog_position_m                    # derived from axle loads, (x, y, z)
    car.tracker_position_relative_to_cog_m  # derived, (x, y, z)
"""

from dataclasses import dataclass
from importlib.resources import files
import math
from pathlib import Path

import yaml

__all__ = ["CarParameters", "load_car_parameters", "list_cars"]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CarParameters:
    """
    Measured car properties (SI units) plus derived dynamics quantities.

    Field values mirror the packaged YAML one-to-one; ``@property`` members are
    computed from them. Positions are ``(x, y, z)`` in the vehicle frame (origin
    at the front-axle centre, x forward, y left, z up). See ``cars/drift_rc.yaml``
    for the documented source measurements.
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

    # Steering [deg]
    max_steer_angle_deg: float

    # Drivetrain
    primary_reduction_teeth: tuple[int, int]
    secondary_reduction: float
    motor_kv_rpm_per_volt: float
    battery_cells: int
    cell_nominal_voltage_v: float

    # Vive tracker mounting: position in the vehicle frame [m] (x, y, z)
    tracker_position_m: Vec3

    # Mass [kg]
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
    def max_steer_angle_rad(self) -> float:
        """Front-wheel angle at full steering command [rad]."""
        return math.radians(self.max_steer_angle_deg)

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
    def battery_nominal_voltage_v(self) -> float:
        """Nominal pack voltage [V]."""
        return self.battery_cells * self.cell_nominal_voltage_v

    @property
    def motor_noload_speed_rad_s(self) -> float:
        """Motor no-load speed at the nominal pack voltage [rad/s]."""
        rpm = self.motor_kv_rpm_per_volt * self.battery_nominal_voltage_v
        return rpm * 2.0 * math.pi / 60.0

    @property
    def wheel_noload_speed_rad_s(self) -> float:
        """Wheel no-load speed [rad/s]: where the drivetrain's torque-speed line hits zero."""
        return self.motor_noload_speed_rad_s / self.total_gear_reduction

    @property
    def mass_total_kg(self) -> float:
        """Total mass [kg], the sum of the measured front and rear axle loads."""
        return self.mass_front_axle_kg + self.mass_rear_axle_kg

    @property
    def front_axle_load_fraction(self) -> float:
        """Fraction of weight on the front axle, from the axle-load measurements."""
        return self.mass_front_axle_kg / self.mass_total_kg

    @property
    def rear_axle_load_fraction(self) -> float:
        """Fraction of weight on the rear axle, from the axle-load measurements."""
        return self.mass_rear_axle_kg / self.mass_total_kg

    @property
    def cog_position_m(self) -> Vec3:
        """
        Centre-of-gravity position in the vehicle frame [m] (x, y, z).

        ``x`` is negative (behind the front axle) and derived from the axle-load
        split; ``y`` is assumed centred; ``z`` (height) is not measured and is
        reported as ``0.0``.
        """
        x = -(self.rear_axle_load_fraction * self.wheelbase_m)
        return (x, 0.0, 0.0)

    @property
    def cog_distance_from_front_axle_m(self) -> float:
        """Longitudinal CoG distance behind the front axle [m] (the model's l_f)."""
        return -self.cog_position_m[0]

    @property
    def cog_distance_from_rear_axle_m(self) -> float:
        """Longitudinal CoG distance ahead of the rear axle [m] (the model's l_r)."""
        return self.wheelbase_m - self.cog_distance_from_front_axle_m

    @property
    def tracker_position_relative_to_cog_m(self) -> Vec3:
        """
        Tracker position relative to the CoG in the vehicle frame [m] (x, y, z).

        Positive ``x`` means the tracker sits ahead of the CoG.
        """
        t, c = self.tracker_position_m, self.cog_position_m
        return (t[0] - c[0], t[1] - c[1], t[2] - c[2])


def _cars_dir():
    """Return the importlib traversable for the packaged ``cars/`` directory."""
    return files("vehicle_simulation").joinpath("cars")


def _from_dict(data: dict) -> CarParameters:
    """Build :class:`CarParameters` from a parsed YAML mapping."""
    data = dict(data)
    if "primary_reduction_teeth" in data:
        data["primary_reduction_teeth"] = tuple(data["primary_reduction_teeth"])
    if "tracker_position_m" in data:
        data["tracker_position_m"] = tuple(data["tracker_position_m"])
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
