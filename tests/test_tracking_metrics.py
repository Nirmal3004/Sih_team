"""Tests for the tracking metrics (SIH26078, Phase 2).

The metrics must survive degenerate input: identical coordinates, missing
timestamps, zero elapsed time and invalid values. None of these may raise, and
none may divide by zero.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from extreme_weather.tracking.metrics import (
    CALM,
    CARDINAL_DIRECTIONS,
    EARTH_RADIUS_KM,
    TREND_STABLE,
    TREND_STRENGTHENING,
    TREND_WEAKENING,
    bearing_degrees,
    build_track_id,
    compass_direction,
    elapsed_hours,
    haversine_distance_km,
    intensity_trend,
    movement_speed_kmh,
    total_distance_km,
)

UTC = timezone.utc
START = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #
def test_distance_matches_known_values() -> None:
    # One degree of latitude with the mean Earth radius.
    assert haversine_distance_km(10.0, 70.0, 11.0, 70.0) == pytest.approx(111.19, abs=0.1)
    # Equator: one degree of longitude equals one degree of latitude.
    assert haversine_distance_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.1)
    # Longitude degrees shrink with the cosine of the latitude.
    equator = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
    high = haversine_distance_km(60.0, 0.0, 60.0, 1.0)
    assert high == pytest.approx(0.5 * equator, rel=0.01)


def test_distance_is_symmetric_and_zero_for_identical_points() -> None:
    forward = haversine_distance_km(13.0, 77.0, 12.0, 76.0)
    backward = haversine_distance_km(12.0, 76.0, 13.0, 77.0)
    assert forward == pytest.approx(backward)

    assert haversine_distance_km(13.0, 77.0, 13.0, 77.0) == 0.0
    assert EARTH_RADIUS_KM == pytest.approx(6371.0, abs=1.0)


def test_distance_handles_invalid_values() -> None:
    assert haversine_distance_km(float("nan"), 77.0, 13.0, 77.0) == 0.0
    assert haversine_distance_km(13.0, float("inf"), 13.0, 77.0) == 0.0


def test_total_distance_sums_a_path() -> None:
    straight = total_distance_km([(10.0, 70.0), (11.0, 70.0), (12.0, 70.0)])
    assert straight == pytest.approx(222.38, abs=0.3)
    assert total_distance_km([(10.0, 70.0)]) == 0.0
    assert total_distance_km([]) == 0.0


# --------------------------------------------------------------------------- #
# Elapsed time
# --------------------------------------------------------------------------- #
def test_elapsed_hours_counts_absolute_difference() -> None:
    assert elapsed_hours(START, START + timedelta(hours=3)) == pytest.approx(3.0)
    assert elapsed_hours(START + timedelta(hours=3), START) == pytest.approx(3.0)
    assert elapsed_hours(START, START + timedelta(minutes=30)) == pytest.approx(0.5)


def test_elapsed_hours_handles_missing_and_equal_timestamps() -> None:
    assert elapsed_hours(None, START) == 0.0
    assert elapsed_hours(START, None) == 0.0
    assert elapsed_hours(None, None) == 0.0
    assert elapsed_hours(START, START) == 0.0


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #
def test_speed_is_distance_over_time() -> None:
    assert movement_speed_kmh(100.0, 2.0) == pytest.approx(50.0)
    assert movement_speed_kmh(0.0, 5.0) == 0.0
    assert movement_speed_kmh(50.0, 0.0) == 0.0


def test_speed_never_divides_by_zero_or_negative_time() -> None:
    assert movement_speed_kmh(50.0, -2.0) == 0.0
    assert movement_speed_kmh(float("nan"), 2.0) == 0.0
    assert movement_speed_kmh(50.0, float("inf")) == 0.0
    assert movement_speed_kmh(None, 2.0) == 0.0


# --------------------------------------------------------------------------- #
# Bearing and compass direction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ((11.0, 70.0), 0.0),  # due north
        ((10.0, 71.0), 90.0),  # due east
        ((9.0, 70.0), 180.0),  # due south
        ((10.0, 69.0), 270.0),  # due west
    ],
)
def test_bearing_cardinal_directions(target, expected) -> None:
    bearing = bearing_degrees(10.0, 70.0, *target)
    assert bearing == pytest.approx(expected, abs=0.5)
    assert 0.0 <= bearing < 360.0


def test_bearing_is_undefined_for_identical_points() -> None:
    assert bearing_degrees(10.0, 70.0, 10.0, 70.0) is None
    assert bearing_degrees(float("nan"), 70.0, 10.0, 70.0) is None
    # A stationary event is reported as CALM.
    assert compass_direction(bearing_degrees(10.0, 70.0, 10.0, 70.0)) == CALM


def test_compass_direction_sectors() -> None:
    assert compass_direction(0.0) == "N"
    assert compass_direction(45.0) == "NE"
    assert compass_direction(90.0) == "E"
    assert compass_direction(135.0) == "SE"
    assert compass_direction(180.0) == "S"
    assert compass_direction(225.0) == "SW"
    assert compass_direction(270.0) == "W"
    assert compass_direction(315.0) == "NW"
    assert compass_direction(360.0) == "N"  # wraps around
    assert compass_direction(-90.0) == "W"
    assert compass_direction(22.4) == "NNE"


def test_compass_direction_reports_calm_for_stationary_or_invalid() -> None:
    assert compass_direction(None) == CALM
    assert compass_direction(float("nan")) == CALM
    assert compass_direction(float("inf")) == CALM
    assert len(CARDINAL_DIRECTIONS) == 16


# --------------------------------------------------------------------------- #
# Intensity trend
# --------------------------------------------------------------------------- #
def test_intensity_strengthening_weakening_and_stable() -> None:
    assert intensity_trend(10.0, 40.0) == TREND_STRENGTHENING
    assert intensity_trend(40.0, 10.0) == TREND_WEAKENING
    assert intensity_trend(10.0, 10.2) == TREND_STABLE
    assert intensity_trend(10.0, 10.0) == TREND_STABLE


def test_intensity_trend_handles_missing_and_zero_values() -> None:
    assert intensity_trend(None, 10.0) == TREND_STABLE
    assert intensity_trend(10.0, None) == TREND_STABLE
    assert intensity_trend(0.0, 0.0) == TREND_STABLE
    assert intensity_trend(float("nan"), 1.0) == TREND_STABLE
    # A small change inside the tolerance band stays stable.
    assert intensity_trend(100.0, 103.0, tolerance=0.05) == TREND_STABLE
    assert intensity_trend(100.0, 120.0, tolerance=0.05) == TREND_STRENGTHENING


# --------------------------------------------------------------------------- #
# Track identifiers
# --------------------------------------------------------------------------- #
def test_track_id_format_is_deterministic() -> None:
    track_id = build_track_id("EXTREME_RAINFALL", START, 2)
    assert track_id == "TRK-EXTREME_RAINFALL-20240601T0000-002"
    assert build_track_id("extreme_rainfall", START, 2) == track_id
    assert build_track_id("EXTREME_WIND", START, 1) == "TRK-EXTREME_WIND-20240601T0000-001"
