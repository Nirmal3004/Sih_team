"""Tests for the reference distribution (climatology) builder.

The reference distributions used here are built from **synthetic** data, exactly
like the production code path. They are prototype statistics, not real-world
climatology.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.anomaly.climatology import (
    GROUP_COORD_NAME,
    POOLED_GROUP_KEY,
    QUANTILE_COORD_NAME,
    ClimatologyError,
    ClimatologyGrouping,
    build_climatology,
    resolve_group_keys,
)
from extreme_weather.anomaly.scores import align_climatology_statistic

MONTHLY_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def make_dataset(
    *,
    days: int = 40,
    start: str = "2024-01-01",
    shape: tuple[int, int] = (4, 5),
    variable: str = "precipitation",
    value: float | None = None,
    with_nan: bool = False,
) -> xr.Dataset:
    """Build a small, fully deterministic test dataset."""
    time = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(start, "D") + np.timedelta64(days, "D"),
    )
    latitudes = np.linspace(10.0, 10.3, shape[0])
    longitudes = np.linspace(70.0, 70.4, shape[1])

    if value is None:
        # Deterministic, slightly seasonal signal so monthly groups differ.
        season = np.sin(np.arange(days) / 6.0)[:, None, None]
        gradient = np.linspace(0.0, 5.0, shape[0])[None, :, None]
        signal = 10.0 + 2.0 * season + gradient
        data = np.broadcast_to(signal, (days, *shape)).astype("float64").copy()
    else:
        data = np.full((days, *shape), float(value))

    if with_nan:
        data = data.copy()
        data[0, 0, 0] = np.nan
        data[days // 2, shape[0] - 1, shape[1] - 1] = np.nan

    return xr.Dataset(
        {variable: (("time", "latitude", "longitude"), data.astype("float32"))},
        coords={"time": time, "latitude": latitudes, "longitude": longitudes},
    )


# --------------------------------------------------------------------------- #
# Creation / dimensions
# --------------------------------------------------------------------------- #
def test_climatology_has_the_expected_structure() -> None:
    climatology = build_climatology(make_dataset(), "precipitation", "month")

    assert set(climatology.data_vars) == {"mean", "std", "count", "min", "max", "quantiles"}
    assert GROUP_COORD_NAME in climatology.dims
    assert climatology["quantiles"].dims == (
        GROUP_COORD_NAME,
        QUANTILE_COORD_NAME,
        "latitude",
        "longitude",
    )
    assert climatology.sizes["latitude"] == 4
    assert climatology.sizes["longitude"] == 5
    assert climatology["mean"].dims == (GROUP_COORD_NAME, "latitude", "longitude")


def test_climatology_records_provenance_and_the_prototype_disclaimer() -> None:
    climatology = build_climatology(make_dataset(), "precipitation", "month")

    assert climatology.attrs["variable"] == "precipitation"
    assert climatology.attrs["grouping"] == "month"
    assert climatology.attrs["fallback"] == "pooled"
    assert climatology.attrs["min_samples"] == 10
    assert climatology.attrs["sample_count"] == 40
    assert "NOT a real-world climatology" in climatology.attrs["comment"]


def test_climatology_statistics_match_numpy() -> None:
    dataset = make_dataset(days=20)
    climatology = build_climatology(dataset, "precipitation", "all")
    expected = dataset["precipitation"].values

    assert np.allclose(climatology["mean"].sel(group=0).values, expected.mean(axis=0), rtol=1e-5)
    assert np.allclose(climatology["std"].sel(group=0).values, expected.std(axis=0), rtol=1e-5)
    assert np.all(climatology["count"].sel(group=0).values == 20)
    assert np.allclose(climatology["min"].sel(group=0).values, expected.min(axis=0))
    assert np.allclose(climatology["max"].sel(group=0).values, expected.max(axis=0))


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def test_monthly_grouping_splits_by_calendar_month() -> None:
    climatology = build_climatology(make_dataset(days=70), "precipitation", "month")
    groups = climatology[GROUP_COORD_NAME].values.tolist()

    assert 1 in groups  # January
    assert 2 in groups  # February
    assert POOLED_GROUP_KEY in groups
    # Count per group must match the number of days in that month (2024 is a
    # leap year, so February contributes 29 days to this window).
    counts = climatology["count"]
    assert int(counts.sel(group=1).values.min()) == 31
    assert int(counts.sel(group=2).values.min()) == 29


def test_day_of_year_grouping_creates_one_group_per_day() -> None:
    climatology = build_climatology(make_dataset(days=10), "precipitation", "day_of_year")
    groups = climatology[GROUP_COORD_NAME].values.tolist()

    # Groups are sorted ascending, so the pooled reference (key -1) comes first.
    assert groups == [POOLED_GROUP_KEY, *range(1, 11)]  # 2024-01-01 .. 2024-01-10


def test_all_grouping_produces_a_single_group() -> None:
    climatology = build_climatology(make_dataset(), "precipitation", "all")
    groups = climatology[GROUP_COORD_NAME].values.tolist()

    assert groups == [0]
    assert POOLED_GROUP_KEY not in groups


def test_group_keys_are_computed_correctly() -> None:
    times = np.array(
        [
            np.datetime64("2024-01-15T00:00"),
            np.datetime64("2024-02-29T12:00"),
            np.datetime64("2023-12-31T23:00"),
        ]
    )
    assert resolve_group_keys(times, "month").tolist() == [1, 2, 12]
    assert resolve_group_keys(np.array([np.datetime64("2024-03-01")]), "day_of_year").tolist() == [61]
    assert resolve_group_keys(times, "all").tolist() == [0, 0, 0]


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def test_missing_values_are_skipped_in_the_statistics() -> None:
    dataset = make_dataset(days=20, with_nan=True)
    climatology = build_climatology(dataset, "precipitation", "all")
    counts = climatology["count"].values
    means = climatology["mean"].values

    # The two cells with one NaN each report 19 samples, the others 20.
    assert int(counts[0, 0, 0]) == 19
    assert int(counts[0, -1, -1]) == 19
    assert int(counts[0, 0, 1]) == 20
    assert np.all(np.isfinite(means))


def test_all_nan_group_produces_nan_statistics_without_crashing() -> None:
    dataset = make_dataset(days=6, value=np.nan)
    climatology = build_climatology(dataset, "precipitation", "all", min_samples=1)

    assert np.isnan(climatology["mean"].values).all()
    assert int(climatology["count"].values.min()) == 0
    assert climatology["quantiles"].notnull().sum() == 0


# --------------------------------------------------------------------------- #
# Fallback behaviour
# --------------------------------------------------------------------------- #
def test_pooled_fallback_is_used_for_uncovered_groups() -> None:
    reference = build_climatology(
        make_dataset(days=30, start="2024-01-01"), "precipitation", "month", min_samples=10
    )
    pooled_mean = reference["mean"].sel(group=POOLED_GROUP_KEY).values

    other_times = np.arange(
        np.datetime64("2024-07-01"), np.datetime64("2024-07-04"), np.timedelta64(1, "D")
    )
    aligned = align_climatology_statistic(reference, other_times, "mean")

    assert aligned.shape[0] == 3
    assert np.allclose(aligned.values, pooled_mean)


def test_fallback_none_yields_nan_for_uncovered_groups() -> None:
    reference = build_climatology(
        make_dataset(days=30, start="2024-01-01"),
        "precipitation",
        "month",
        min_samples=10,
        fallback="none",
    )
    other_times = np.array([np.datetime64("2024-07-01"), np.datetime64("2024-01-15")])
    aligned = align_climatology_statistic(reference, other_times, "mean")

    assert np.isnan(aligned.values[0]).all()
    assert not np.isnan(aligned.values[1]).any()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_missing_variable_raises_a_clear_error() -> None:
    with pytest.raises(ClimatologyError, match="not part of the dataset"):
        build_climatology(make_dataset(), "wind_speed", "month")


def test_unknown_grouping_raises() -> None:
    with pytest.raises(ValueError, match="Unknown climatology grouping"):
        build_climatology(make_dataset(), "precipitation", "weekly")


def test_invalid_quantile_levels_raise() -> None:
    with pytest.raises(ClimatologyError, match="strictly inside"):
        build_climatology(
            make_dataset(), "precipitation", "month", quantile_levels=(0.0, 0.5, 1.0)
        )

    with pytest.raises(ClimatologyError, match="strictly increasing"):
        build_climatology(
            make_dataset(), "precipitation", "month", quantile_levels=(0.5, 0.25)
        )


def test_grouping_enum_round_trip() -> None:
    assert ClimatologyGrouping.from_value("MONTH") is ClimatologyGrouping.MONTH
    assert ClimatologyGrouping.from_value(" day_of_year ") is ClimatologyGrouping.DAY_OF_YEAR
    with pytest.raises(ValueError, match="Unknown climatology grouping"):
        ClimatologyGrouping.from_value("hourly")


def test_custom_quantile_levels_are_honoured() -> None:
    dataset = make_dataset(days=20)
    climatology = build_climatology(
        dataset, "precipitation", "all", quantile_levels=MONTHLY_LEVELS
    )
    levels = climatology[QUANTILE_COORD_NAME].values

    assert levels.tolist() == list(MONTHLY_LEVELS)
    assert climatology["quantiles"].sizes[QUANTILE_COORD_NAME] == len(MONTHLY_LEVELS)
    median = climatology["quantiles"].sel(quantile=0.5).values
    assert np.allclose(median, np.median(dataset["precipitation"].values, axis=0), rtol=1e-5)
