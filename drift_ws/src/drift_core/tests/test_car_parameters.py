"""Unit tests for loading and deriving measured car parameters."""

import math

import pytest

from drift_core.car_parameters import (
    CarParameters,
    list_cars,
    load_car_parameters,
)


def test_drift_rc_is_packaged():
    assert "drift_rc" in list_cars()


def test_loads_measured_values():
    car = load_car_parameters("drift_rc")
    assert isinstance(car, CarParameters)
    assert car.name == "drift_rc"
    assert car.wheelbase_m == pytest.approx(0.256)
    assert car.tracker_position_m == pytest.approx((-0.134, 0.0, 0.067))
    assert car.primary_reduction_teeth == (24, 83)


def test_total_mass_is_sum_of_axle_loads():
    car = load_car_parameters("drift_rc")
    assert car.mass_total_kg == pytest.approx(car.mass_front_axle_kg + car.mass_rear_axle_kg)
    assert car.mass_total_kg == pytest.approx(0.445 + 1.011)


def test_derived_tire_radius_and_gear_ratio():
    car = load_car_parameters("drift_rc")
    assert car.tire_radius_m == pytest.approx(0.0603 / 2.0)
    assert car.total_gear_reduction == pytest.approx((83 / 24) * 2.6)


def test_derived_cog_from_axle_loads():
    car = load_car_parameters("drift_rc")
    # Load fractions sum to 1 and the rear carries more (rear-heavy car).
    assert car.front_axle_load_fraction + car.rear_axle_load_fraction == pytest.approx(1.0)
    assert car.rear_axle_load_fraction > car.front_axle_load_fraction
    # CoG sits behind the front axle (negative x in the vehicle frame).
    cog_x, cog_y, cog_z = car.cog_position_m
    assert cog_x < 0.0
    assert (cog_y, cog_z) == pytest.approx((0.0, 0.0))
    # Distances from front + from rear must add up to the wheelbase.
    a = car.cog_distance_from_front_axle_m
    b = car.cog_distance_from_rear_axle_m
    assert a == pytest.approx(-cog_x)
    assert a + b == pytest.approx(car.wheelbase_m)
    # The tracker sits ahead of the CoG for this layout (positive x).
    rel = car.tracker_position_relative_to_cog_m
    assert rel[0] == pytest.approx(car.tracker_position_m[0] - cog_x)
    assert rel[0] > 0.0


def test_frozen_dataclass_is_immutable():
    car = load_car_parameters("drift_rc")
    with pytest.raises(Exception):
        car.wheelbase_m = 0.3  # type: ignore[misc]


def test_all_dimensions_finite_and_positive():
    car = load_car_parameters("drift_rc")
    for value in (
        car.wheelbase_m,
        car.track_width_front_m,
        car.track_width_rear_m,
        car.tire_diameter_m,
        car.mass_total_kg,
        car.length_total_m,
        car.height_with_cover_mount_m,
    ):
        assert math.isfinite(value) and value > 0.0
