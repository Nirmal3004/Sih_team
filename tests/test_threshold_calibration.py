"""Tests for Phase 7 threshold calibration.

Calibration is only meaningful if it is (a) derived from the **training** data,
(b) deterministic, and (c) honest about what it is not. The last one matters
most: the thresholds are statistically calibrated, never meteorologically
validated, and the report says so.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.validation.threshold_calibration import (
    CALIBRATION_WARNING,
    DEFAULT_PERCENTILES,
    Method,
    calibrate_thresholds,
    variable_quantiles,
)

from tests.validation_fixtures import make_validation_dataset


# --------------------------------------------------------------------------- #
# Quantiles
# --------------------------------------------------------------------------- #
def test_quantiles_ignore_nan_rather_than_imputing() -> None:
    """A quantile over imputed values would be a quantile of a guess."""
    values = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0])
    count, quantiles = variable_quantiles(values, (50.0, 100.0))
    assert count == 5
    assert quantiles[0] == pytest.approx(3.0)
    assert quantiles[1] == pytest.approx(5.0)


def test_an_all_nan_field_yields_nan_not_zero() -> None:
    count, quantiles = variable_quantiles(np.full(10, np.nan), (50.0,))
    assert count == 0
    assert np.isnan(quantiles[0])


def test_an_out_of_range_percentile_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        variable_quantiles(np.arange(10.0), (150.0,))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_derives_thresholds_from_the_data(validation_dataset: xr.Dataset) -> None:
    calibration = calibrate_thresholds(validation_dataset, dataset_id="demo")
    precipitation = calibration.variable("precipitation")
    assert precipitation is not None
    assert precipitation.sample_count > 0
    assert len(precipitation.selected_thresholds) == len(DEFAULT_PERCENTILES)


def test_the_thresholds_ascend(validation_dataset: xr.Dataset) -> None:
    precipitation = calibrate_thresholds(
        validation_dataset, dataset_id="demo"
    ).variable("precipitation")
    values = [step[0] for step in precipitation.selected_thresholds]
    assert values == sorted(values)
    assert all(b > a for a, b in zip(values, values[1:]))


def test_calibration_is_deterministic(validation_dataset: xr.Dataset) -> None:
    """Same training data, same percentiles, same thresholds - always."""
    first = calibrate_thresholds(validation_dataset, dataset_id="demo")
    second = calibrate_thresholds(validation_dataset, dataset_id="demo")
    assert [c.quantiles for c in first.variables] == [c.quantiles for c in second.variables]
    assert [c.selected_thresholds for c in first.variables] == [
        c.selected_thresholds for c in second.variables
    ]


def test_a_higher_percentile_gives_a_higher_threshold(
    validation_dataset: xr.Dataset,
) -> None:
    low = calibrate_thresholds(
        validation_dataset, dataset_id="d", percentiles=(50.0, 99.0, 99.9, 100.0)
    ).variable("precipitation")
    high = calibrate_thresholds(
        validation_dataset, dataset_id="d", percentiles=(99.0, 99.9, 99.99, 100.0)
    ).variable("precipitation")
    assert low.quantiles[0] < high.quantiles[0]


def test_the_calibration_period_is_recorded(validation_dataset: xr.Dataset) -> None:
    calibration = calibrate_thresholds(validation_dataset, dataset_id="demo")
    assert calibration.calibration_period
    assert calibration.calibration_period[0] < calibration.calibration_period[-1]
    assert calibration.split == "train"


def test_every_calibration_carries_the_not_validated_warning(
    validation_dataset: xr.Dataset,
) -> None:
    payload = calibrate_thresholds(validation_dataset, dataset_id="demo").as_dict()
    assert any(CALIBRATION_WARNING in warning for warning in payload["warnings"])


def test_a_threshold_set_is_produced_for_the_pipeline(
    validation_dataset: xr.Dataset,
) -> None:
    calibration = calibrate_thresholds(validation_dataset, dataset_id="demo")
    assert calibration.thresholds is not None
    assert calibration.thresholds.ladders


def test_a_seasonal_method_falls_back_when_only_one_month_is_present(
    validation_dataset: xr.Dataset,
) -> None:
    """A week of June cannot be split into seasons, and the code must say so."""
    calibration = calibrate_thresholds(
        validation_dataset, dataset_id="demo", method=Method.QUANTILE_SEASONAL
    )
    assert calibration.method == Method.PERCENTILE
    assert any("not separable" in warning for warning in calibration.warnings)


def test_a_seasonal_method_is_kept_when_several_months_are_present() -> None:
    """Six distinct months are separable, so the seasonal method is kept."""
    import datetime as dt

    expanded = xr.concat(
        [
            make_validation_dataset(steps=1, start=dt.datetime(2024, month, 5, 12))
            for month in range(1, 7)
        ],
        dim="time",
    )
    calibration = calibrate_thresholds(
        expanded, dataset_id="demo", method=Method.QUANTILE_SEASONAL
    )
    assert calibration.method == Method.QUANTILE_SEASONAL
    assert calibration.variable("precipitation").seasonal

def test_an_unknown_method_is_rejected(validation_dataset: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrate_thresholds(validation_dataset, dataset_id="d", method="magic")


def test_the_percentile_count_must_match_the_severity_count(
    validation_dataset: xr.Dataset,
) -> None:
    with pytest.raises(ValueError, match="percentiles"):
        calibrate_thresholds(validation_dataset, dataset_id="d", percentiles=(99.0,))


def test_a_dataset_without_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D 'time'"):
        calibrate_thresholds(xr.Dataset({"x": ("a", [1.0])}), dataset_id="d")


def test_a_missing_variable_is_reported_not_silently_skipped(
    validation_dataset: xr.Dataset,
) -> None:
    calibration = calibrate_thresholds(
        validation_dataset, dataset_id="d", variables=("precipitation", "hail")
    )
    assert any("hail" in warning for warning in calibration.warnings)


def test_the_serialised_calibration_reports_every_required_field(
    validation_dataset: xr.Dataset,
) -> None:
    payload = calibrate_thresholds(validation_dataset, dataset_id="d").as_dict()
    for key in ("method", "split", "calibration_period", "variables", "warnings"):
        assert key in payload
    for variable in payload["variables"]:
        for key in ("variable", "unit", "sample_count", "quantiles", "selected_thresholds"):
            assert key in variable

