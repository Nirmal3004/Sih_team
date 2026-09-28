"""Tests for Phase 7 climatology coverage diagnostics.

The behaviour under test is a refusal: a short dataset must be reported as
**insufficient**, not quietly filled from another bucket. A smooth, confident,
wrong climatology is more dangerous than an absent one.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.anomaly.climatology import ClimatologyGrouping
from extreme_weather.validation.climatology_validation import (
    INSUFFICIENT_DAY_OF_YEAR,
    INSUFFICIENT_MONTH,
    MIN_SAMPLES_PER_BUCKET,
    assess_climatology_coverage,
    build_climatology_diagnostics,
)

from tests.validation_fixtures import make_times, make_validation_dataset


# --------------------------------------------------------------------------- #
# Coverage assessment
# --------------------------------------------------------------------------- #
def test_a_week_of_data_is_not_a_day_of_year_climatology() -> None:
    coverage = assess_climatology_coverage(make_times(24 * 7), "day_of_year")
    assert coverage.usable is False
    assert coverage.message == INSUFFICIENT_DAY_OF_YEAR
    assert coverage.buckets_expected == 366
    assert coverage.buckets_sufficient < 366


def test_a_week_of_data_is_not_a_monthly_climatology() -> None:
    coverage = assess_climatology_coverage(make_times(24 * 7), "month")
    assert coverage.usable is False
    assert coverage.message == INSUFFICIENT_MONTH


def test_a_full_year_of_monthly_data_is_usable() -> None:
    times = np.array(
        [np.datetime64(f"2024-{month:02d}-15T00:00", "m") for month in range(1, 13)],
        dtype="datetime64[ns]",
    )
    coverage = assess_climatology_coverage(times, "month", min_samples_per_bucket=1)
    assert coverage.distinct_months == 12
    assert coverage.buckets_sufficient == 12
    assert coverage.usable is True
    assert coverage.message == ""


def test_a_partially_covered_climatology_is_not_usable() -> None:
    """Ten months is not twelve. Being 'mostly' complete is not complete."""
    times = np.array(
        [np.datetime64(f"2024-{month:02d}-15T00:00", "m") for month in range(1, 11)],
        dtype="datetime64[ns]",
    )
    coverage = assess_climatology_coverage(times, "month", min_samples_per_bucket=1)
    assert coverage.buckets_covered == 10
    assert coverage.usable is False


def test_the_coverage_counts_the_missing_fraction() -> None:
    times = np.array(
        [np.datetime64(f"2024-{month:02d}-15T00:00", "m") for month in range(1, 7)],
        dtype="datetime64[ns]",
    )
    coverage = assess_climatology_coverage(times, "month", min_samples_per_bucket=1)
    assert coverage.missing_fraction == pytest.approx(0.5)


def test_an_empty_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        assess_climatology_coverage(np.array([], dtype="datetime64[ns]"), "month")


def test_an_unknown_grouping_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown grouping"):
        assess_climatology_coverage(make_times(10), "fortnight")


# --------------------------------------------------------------------------- #
# Full diagnostics
# --------------------------------------------------------------------------- #
def test_diagnostics_describe_the_dataset(validation_dataset: xr.Dataset) -> None:
    diagnostics = build_climatology_diagnostics(
        validation_dataset, split="train", dataset_id="demo"
    )
    assert diagnostics.split == "train"
    assert diagnostics.time_steps == int(validation_dataset.sizes["time"])
    assert diagnostics.spatial_points == int(
        validation_dataset.sizes["latitude"] * validation_dataset.sizes["longitude"]
    )
    assert "precipitation" in diagnostics.variables
    assert diagnostics.period_start < diagnostics.period_end


def test_a_clean_dataset_has_no_missing_values(short_dataset: xr.Dataset) -> None:
    diagnostics = build_climatology_diagnostics(short_dataset)
    assert diagnostics.missing_fraction == pytest.approx(0.0)


def test_missing_values_are_counted_per_variable() -> None:
    dataset = make_validation_dataset(steps=10)
    dataset["precipitation"] = dataset["precipitation"].where(
        dataset["precipitation"] > 50.0
    )
    diagnostics = build_climatology_diagnostics(dataset)
    assert diagnostics.missing_fraction > 0.0
    assert diagnostics.per_variable_missing["precipitation"] > 0.0
    assert diagnostics.per_variable_missing["temperature"] == 0.0


def test_a_short_dataset_is_reported_as_insufficient() -> None:
    diagnostics = build_climatology_diagnostics(make_validation_dataset(steps=48))
    assert diagnostics.is_sufficient is False
    day_of_year = diagnostics.coverage("day_of_year")
    assert day_of_year is not None and day_of_year.usable is False
    assert day_of_year.message == INSUFFICIENT_DAY_OF_YEAR


def test_hour_of_day_is_refused_on_a_short_record() -> None:
    """Two hours of one afternoon is not a diurnal cycle."""
    diagnostics = build_climatology_diagnostics(make_validation_dataset(steps=8))
    assert diagnostics.hour_of_day_supported is False
    assert "hour-of-day" in diagnostics.hour_of_day_message


def test_the_diagnostics_serialise_every_block() -> None:
    payload = build_climatology_diagnostics(make_validation_dataset(steps=48)).as_dict()
    for key in (
        "split",
        "dataset_id",
        "time_steps",
        "spatial_points",
        "variables",
        "missing_fraction",
        "coverages",
        "hour_of_day_supported",
        "hour_of_day_message",
    ):
        assert key in payload
    for coverage in payload["coverages"].values():
        assert "usable" in coverage and "message" in coverage


def test_a_dataset_without_time_is_rejected() -> None:
    dataset = xr.Dataset({"x": ("latitude", [1.0, 2.0])})
    with pytest.raises(ValueError, match="1-D 'time'"):
        build_climatology_diagnostics(dataset)


def test_the_diagnostics_are_deterministic(validation_dataset: xr.Dataset) -> None:
    first = build_climatology_diagnostics(validation_dataset).as_dict()
    second = build_climatology_diagnostics(validation_dataset).as_dict()
    first.pop("generated_at"), second.pop("generated_at")
    assert first == second

