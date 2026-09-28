"""Tests for the data quality control layer (SIH26078, Phase 6).

Real data fails in ways the deterministic generator never does, so the checks
here are the ones that decide whether a file is allowed to reach Phase 1.

The two-severity split is the point of this module and is asserted throughout:

* **errors** block processing (all-NaN variable, invalid coordinate, a
  non-monotonic time axis);
* **warnings** are recorded but do not block (some NaN, duplicate timestamps,
  implausible values).

A check that quietly "fixed" its problem instead of reporting it would be worse
than no check, so the tests assert on the *messages* as well as the counts.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.ingestion import check_quality, require_valid
from extreme_weather.ingestion.exceptions import QualityCheckError
from extreme_weather.ingestion.quality import PLAUSIBLE_RANGES

from tests.ingestion_fixtures import SMALL_LAT, SMALL_LON, make_dataset, make_times

SHAPE = (4, 3, 3)


# --------------------------------------------------------------------------- #
# A clean dataset
# --------------------------------------------------------------------------- #
def test_a_clean_contract_dataset_is_valid_and_silent() -> None:
    report = check_quality(make_dataset())
    assert report.valid is True
    assert report.errors == ()
    assert report.warnings == ()
    assert report.missing_fraction == pytest.approx(0.0)


def test_a_clean_dataset_reports_its_shape() -> None:
    report = check_quality(make_dataset())
    assert report.time_steps == 4
    assert report.spatial_points == SMALL_LAT * SMALL_LON
    assert set(report.variables) >= {"temperature", "precipitation", "pressure"}


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def test_an_all_nan_variable_is_an_error_not_a_warning() -> None:
    """A wholly empty field cannot be analysed, so it must block."""
    report = check_quality(
        make_dataset(variables={"precipitation": (np.full(SHAPE, np.nan), "mm h-1")})
    )
    assert report.valid is False
    assert any("entirely NaN" in error for error in report.errors)


def test_a_high_nan_fraction_is_a_warning() -> None:
    """A quarter of a field missing is a warning, not a blocker."""
    values = np.full(SHAPE, 5.0)
    values[0, :, :] = np.nan  # 1 of 4 time steps == 25 %
    report = check_quality(
        make_dataset(variables={"precipitation": (values, "mm h-1")})
    )
    assert report.valid is True
    assert any("NaN" in warning for warning in report.warnings)
    assert report.missing_fraction == pytest.approx(0.25)


def test_the_nan_fraction_is_reported_per_variable() -> None:
    values = np.full(SHAPE, 5.0)
    values[0, 0, :] = np.nan
    report = check_quality(
        make_dataset(variables={"precipitation": (values, "mm h-1")})
    )
    assert report.per_variable_nan["precipitation"] > 0.0
    assert report.per_variable_nan.get("temperature", 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def test_duplicate_timestamps_are_counted_and_warned() -> None:
    stamps = make_times(4)
    duplicated = np.array(
        [stamps[0], stamps[0], stamps[1], stamps[2]], dtype="datetime64[ns]"
    )
    report = check_quality(make_dataset(times=duplicated))
    assert report.duplicate_timestamps == 1
    assert any("duplicate timestamp" in warning for warning in report.warnings)
    # Duplicates alone do not block: the normalizer removes them.
    assert report.valid is True


def test_a_non_monotonic_time_axis_is_an_error() -> None:
    stamps = make_times(4)
    shuffled = np.array(
        [stamps[3], stamps[0], stamps[2], stamps[1]], dtype="datetime64[ns]"
    )
    report = check_quality(make_dataset(times=shuffled))
    assert report.valid is False
    assert any("not strictly increasing" in error for error in report.errors)


def test_an_empty_time_coordinate_is_an_error() -> None:
    report = check_quality(make_dataset(times=np.array([], dtype="datetime64[ns]")))
    assert report.valid is False


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def test_a_latitude_outside_the_valid_range_is_an_error() -> None:
    report = check_quality(make_dataset(latitude=np.array([12.0, 12.5, 95.0])))
    assert report.valid is False
    assert any("latitude" in error and "valid range" in error for error in report.errors)


def test_a_longitude_beyond_360_is_an_error() -> None:
    report = check_quality(make_dataset(longitude=np.array([76.5, 77.0, 400.0])))
    assert report.valid is False
    assert any("longitude" in error for error in report.errors)


def test_a_zero_to_360_longitude_is_still_acceptable() -> None:
    """0-360 is a legal convention; the check must not reject it outright."""
    report = check_quality(make_dataset(longitude=np.array([0.5, 1.5, 359.5])))
    assert report.valid is True


def test_duplicate_coordinates_are_counted_and_warned() -> None:
    report = check_quality(make_dataset(latitude=np.array([12.0, 12.0, 13.0])))
    assert report.duplicate_coordinates == 1
    assert any("duplicate 'latitude'" in warning for warning in report.warnings)


def test_a_non_finite_coordinate_is_an_error() -> None:
    report = check_quality(make_dataset(latitude=np.array([12.0, np.nan, 13.0])))
    assert report.valid is False
    assert any("NaN or infinite" in error for error in report.errors)


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
def test_implausible_values_are_a_warning_by_default() -> None:
    """Out-of-range data is reported, never silently clipped."""
    values = np.full(SHAPE, 5.0)
    values[0, 0, 0] = 9_000.0  # 9000 mm/h is not a real rainfall rate
    report = check_quality(
        make_dataset(variables={"precipitation": (values, "mm h-1")})
    )
    assert report.valid is True
    assert any("plausible range" in warning for warning in report.warnings)
    # Nothing was altered: the check reports, it does not fix.
    assert report.per_variable_nan["precipitation"] == 0.0


def test_strict_ranges_promote_implausible_values_to_an_error() -> None:
    values = np.full(SHAPE, 5.0)
    values[0, 0, 0] = 9_000.0
    report = check_quality(
        make_dataset(variables={"precipitation": (values, "mm h-1")}),
        strict_ranges=True,
    )
    assert report.valid is False
    assert any("plausible range" in error for error in report.errors)


def test_every_plausible_range_is_ordered() -> None:
    for name, (low, high) in PLAUSIBLE_RANGES.items():
        assert low < high, f"{name} has an inverted plausible range"


# --------------------------------------------------------------------------- #
# Structure and metadata
# --------------------------------------------------------------------------- #
def test_a_dataset_with_no_variables_is_an_error() -> None:
    empty = xr.Dataset(
        coords={"time": make_times(), "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]}
    )
    report = check_quality(empty)
    assert report.valid is False
    assert any("no data variables" in error for error in report.errors)


def test_a_missing_required_dimension_is_an_error() -> None:
    report = check_quality(make_dataset().isel(time=0, drop=True))
    assert report.valid is False
    assert any("time" in error for error in report.errors)


def test_unexpected_dimensions_are_a_warning() -> None:
    report = check_quality(make_dataset().expand_dims(level=[1000.0, 850.0]))
    assert report.valid is True
    assert any("Unexpected dimension" in warning for warning in report.warnings)
    assert "level" in report.unexpected_dimensions


def test_a_variable_without_units_is_a_warning() -> None:
    dataset = make_dataset()
    dataset["temperature"].attrs.pop("units")
    report = check_quality(dataset)
    assert any("no 'units' attribute" in warning for warning in report.warnings)


def test_missing_contract_variables_are_reported_not_invented() -> None:
    report = check_quality(
        make_dataset(variables={"precipitation": (np.full(SHAPE, 3.0), "mm h-1")})
    )
    assert set(report.missing_standard_variables) == {
        "temperature",
        "pressure",
        "relative_humidity",
        "u_wind",
        "v_wind",
    }
    assert report.valid is True  # a subset is usable, just narrower


# --------------------------------------------------------------------------- #
# The report object
# --------------------------------------------------------------------------- #
def test_the_report_serialises_every_field() -> None:
    payload = check_quality(make_dataset()).as_dict()
    for key in (
        "valid",
        "errors",
        "warnings",
        "missing_fraction",
        "time_steps",
        "spatial_points",
        "variables",
        "per_variable_nan",
        "duplicate_timestamps",
        "generated_at",
    ):
        assert key in payload


def test_error_and_warning_counts_match_the_lists() -> None:
    report = check_quality(
        make_dataset(
            variables={"precipitation": (np.full(SHAPE, np.nan), "mm h-1")},
            latitude=np.array([12.0, 95.0, 96.0]),
        )
    )
    assert report.error_count == len(report.errors)
    assert report.warning_count == len(report.warnings)


def test_require_valid_returns_the_report_when_valid() -> None:
    report = check_quality(make_dataset())
    assert require_valid(report, "demo") is report


def test_require_valid_raises_and_lists_every_error() -> None:
    report = check_quality(
        make_dataset(
            variables={"precipitation": (np.full(SHAPE, np.nan), "mm h-1")},
            latitude=np.array([12.0, 95.0, 96.0]),
        )
    )
    with pytest.raises(QualityCheckError) as info:
        require_valid(report, "demo")
    message = str(info.value)
    assert "demo" in message
    for error in report.errors:
        assert error in message

