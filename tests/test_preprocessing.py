"""Tests for missing-value handling and the preprocessing pipeline.

The gap-filling tests follow the documented semantics of
:mod:`extreme_weather.data.missing`:

* interior gaps up to ``max_gap`` are interpolated,
* longer gaps stay missing,
* leading/trailing gaps stay missing (no extrapolation),
* all-missing arrays are handled without crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extreme_weather.data.loader import load_dataset
from extreme_weather.data.missing import (
    count_missing,
    fill_missing,
    fill_missing_dataset,
    find_missing_gaps,
    missing_fraction,
)
from extreme_weather.data.preprocessing import (
    PreprocessingReport,
    default_processed_path,
    preprocess_dataset,
    preprocess_file,
)
from extreme_weather.data.validator import DatasetValidationError

NAN = np.nan


def build_series(values: list[float], *, name: str = "precipitation") -> xr.DataArray:
    """Build a float32 ``time`` series DataArray from a list (NaN allowed)."""
    time = np.datetime64("2024-06-01T00:00:00", "s") + np.arange(len(values)) * np.timedelta64(1, "h")
    return xr.DataArray(
        np.asarray(values, dtype="float32"),
        dims=("time",),
        coords={"time": time},
        name=name,
        attrs={"units": "mm h-1", "long_name": "Synthetic precipitation rate"},
    )


# --------------------------------------------------------------------------- #
# 1. fill_missing gap semantics
# --------------------------------------------------------------------------- #
def test_fill_missing_without_gaps_keeps_values() -> None:
    series = build_series([1.0, 2.0, 3.0, 4.0])
    filled = fill_missing(series, max_gap=3)

    assert not np.isnan(filled.values).any()
    assert np.allclose(filled.values, series.values)
    assert filled.dims == series.dims
    assert filled.name == series.name
    assert filled.attrs["missing_value_handling"].startswith("linear interpolation")


def test_fill_missing_fills_a_single_value_gap() -> None:
    series = build_series([1.0, NAN, 3.0])
    filled = fill_missing(series, max_gap=1)

    assert np.allclose(filled.values, [1.0, 2.0, 3.0], atol=1e-6)


def test_fill_missing_fills_a_gap_of_exactly_max_gap() -> None:
    series = build_series([0.0, NAN, NAN, NAN, 4.0])
    filled = fill_missing(series, max_gap=3)

    assert not np.isnan(filled.values).any()
    assert np.allclose(filled.values, [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-6)


def test_fill_missing_keeps_gaps_larger_than_max_gap() -> None:
    series = build_series([0.0, NAN, NAN, NAN, NAN, 5.0])
    filled = fill_missing(series, max_gap=3)

    assert np.isnan(filled.values[1:5]).all()
    assert filled.values[0] == pytest.approx(0.0)
    assert filled.values[5] == pytest.approx(5.0)


def test_fill_missing_preserves_leading_gap() -> None:
    series = build_series([NAN, NAN, 3.0, 4.0, 5.0])
    filled = fill_missing(series, max_gap=5)

    assert np.isnan(filled.values[:2]).all()
    assert np.allclose(filled.values[2:], [3.0, 4.0, 5.0])


def test_fill_missing_preserves_trailing_gap() -> None:
    series = build_series([1.0, 2.0, 3.0, NAN, NAN])
    filled = fill_missing(series, max_gap=5)

    assert np.allclose(filled.values[:3], [1.0, 2.0, 3.0])
    assert np.isnan(filled.values[3:]).all()


def test_fill_missing_handles_leading_and_interior_gaps_together() -> None:
    series = build_series([NAN, 2.0, NAN, NAN, 8.0, NAN])
    filled = fill_missing(series, max_gap=2)

    assert np.isnan(filled.values[0])  # leading gap untouched
    assert np.isnan(filled.values[5])  # trailing gap untouched
    assert np.allclose(filled.values[2:4], [4.0, 6.0], atol=1e-6)


def test_fill_missing_all_missing_array_does_not_crash() -> None:
    series = build_series([NAN, NAN, NAN, NAN])

    for max_gap in (0, 2, None):
        filled = fill_missing(series, max_gap=max_gap)
        assert np.isnan(filled.values).all()

    assert count_missing(filled) == 4
    assert missing_fraction(filled) == pytest.approx(1.0)


def test_fill_missing_max_gap_zero_fills_nothing() -> None:
    series = build_series([1.0, NAN, 3.0])
    filled = fill_missing(series, max_gap=0)

    assert count_missing(filled) == 1


def test_fill_missing_max_gap_none_fills_all_interior_gaps() -> None:
    series = build_series([0.0, NAN, NAN, NAN, NAN, 5.0])
    filled = fill_missing(series, max_gap=None)

    assert not np.isnan(filled.values).any()
    assert np.allclose(filled.values, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-6)


def test_fill_missing_dtype_and_input_are_preserved() -> None:
    series = build_series([1.0, NAN, 3.0])
    original = series.values.copy()

    filled = fill_missing(series, max_gap=1)

    assert filled.dtype == np.float32
    assert np.array_equal(series.values, original, equal_nan=True)  # input untouched
    assert filled.attrs["units"] == "mm h-1"
    again = fill_missing(series, max_gap=1)
    assert np.array_equal(filled.values, again.values, equal_nan=True)


@pytest.mark.parametrize("invalid_gap", [-1, 1.5, "3", True])
def test_fill_missing_rejects_invalid_max_gap(invalid_gap) -> None:
    series = build_series([1.0, NAN, 3.0])

    with pytest.raises(ValueError, match="max_gap"):
        fill_missing(series, max_gap=invalid_gap)


def test_fill_missing_rejects_non_floating_dtype() -> None:
    integer_series = xr.DataArray(
        np.array([1, 2, 3], dtype="int32"),
        dims=("time",),
        coords={"time": build_series([1.0, 2.0, 3.0])["time"].values},
        name="counts",
    )

    with pytest.raises(ValueError, match="floating point"):
        fill_missing(integer_series)


def test_fill_missing_requires_the_requested_dimension() -> None:
    array = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), name="field")

    with pytest.raises(ValueError, match="Dimension 'time' not found"):
        fill_missing(array)


def test_find_missing_gaps_classifies_runs() -> None:
    values = np.array([NAN, NAN, 1.0, NAN, 3.0, NAN, NAN, NAN], dtype="float64")

    gaps = find_missing_gaps(values)

    assert [(gap.start, gap.end, gap.kind, gap.length) for gap in gaps] == [
        (0, 1, "leading", 2),
        (3, 3, "interior", 1),
        (5, 7, "trailing", 3),
    ]
    assert find_missing_gaps(np.array([1.0, 2.0])) == []
    assert [(gap.kind, gap.length) for gap in find_missing_gaps(np.full(3, NAN))] == [
        ("all_missing", 3)
    ]


def test_fill_missing_dataset_only_touches_matching_variables(small_dataset) -> None:
    with_gaps = small_dataset.copy(deep=True)
    with_gaps["precipitation"][2, 3, 3] = np.nan
    with_gaps["temperature"][2, 3, 3] = np.nan
    derived_like = xr.DataArray(
        np.arange(4, dtype="float32"), dims=("level",), name="level_weight"
    )
    with_gaps["level_weight"] = derived_like

    filled = fill_missing_dataset(with_gaps, max_gap=3, variables=("precipitation",))

    assert count_missing(filled["precipitation"]) == 0
    assert count_missing(filled["temperature"]) == 1  # not selected
    assert count_missing(filled["level_weight"]) == 0  # no time dimension
    assert count_missing(with_gaps["precipitation"]) == 1  # input dataset untouched
    assert count_missing(with_gaps)["precipitation"] == 1


# --------------------------------------------------------------------------- #
# 2. Preprocessing pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_preserves_variables_and_coordinates(small_dataset, processed_result) -> None:
    processed = processed_result.dataset

    for name in small_dataset.data_vars:
        assert name in processed.data_vars
    assert set(processed.data_vars) == set(small_dataset.data_vars) | {
        "wind_speed",
        "wind_direction",
    }
    assert tuple(processed["precipitation"].dims) == ("time", "latitude", "longitude")
    assert processed.sizes == small_dataset.sizes
    assert np.array_equal(processed["latitude"].values, small_dataset["latitude"].values)
    assert np.array_equal(processed["longitude"].values, small_dataset["longitude"].values)
    assert np.array_equal(processed["time"].values, small_dataset["time"].values)
    assert processed["precipitation"].attrs["units"] == small_dataset["precipitation"].attrs["units"]
    assert processed.attrs["processed"] == "true"
    assert processed.attrs["project"] == "SIH26078"


def test_pipeline_does_not_mutate_the_raw_dataset(small_dataset) -> None:
    with_gaps = small_dataset.copy(deep=True)
    with_gaps["temperature"][1, 2, 2] = np.nan
    before = with_gaps["temperature"].values.copy()

    result = preprocess_dataset(with_gaps, max_gap=3)

    assert count_missing(with_gaps["temperature"]) == 1
    assert np.array_equal(with_gaps["temperature"].values, before, equal_nan=True)
    assert "wind_speed" not in with_gaps.data_vars
    assert count_missing(result.dataset["temperature"]) == 0


def test_pipeline_adds_correct_derived_wind_speed(processed_result, small_dataset) -> None:
    expected = np.sqrt(
        small_dataset["u_wind"].values.astype("float64") ** 2
        + small_dataset["v_wind"].values.astype("float64") ** 2
    )

    assert np.allclose(processed_result.dataset["wind_speed"].values, expected, rtol=1e-6)
    assert processed_result.dataset["wind_direction"].values.min() >= 0.0
    assert processed_result.dataset["wind_direction"].values.max() < 360.0
    assert set(processed_result.report.derived_variables) == {"wind_speed", "wind_direction"}


def test_pipeline_respects_the_max_gap_option(small_dataset) -> None:
    with_gaps = small_dataset.copy(deep=True)
    with_gaps["precipitation"][4:9, 4, 4] = np.nan  # 5 consecutive missing steps

    strict = preprocess_dataset(with_gaps, max_gap=2)
    lenient = preprocess_dataset(with_gaps, max_gap=None)

    assert count_missing(strict.dataset["precipitation"]) == 5
    assert count_missing(lenient.dataset["precipitation"]) == 0
    assert strict.report.values_filled == 0
    assert lenient.report.values_filled == 5


def test_pipeline_report_records_missing_statistics(small_dataset) -> None:
    with_gaps = small_dataset.copy(deep=True)
    with_gaps["precipitation"][3, 1, 1] = np.nan
    with_gaps["temperature"][3, 1, 1] = np.nan

    result = preprocess_dataset(with_gaps, max_gap=3)
    report = result.report

    assert isinstance(report, PreprocessingReport)
    assert report.missing_before["precipitation"] == 1
    assert report.missing_before["temperature"] == 1
    assert report.total_missing_before == 2
    assert report.total_missing_after == 0
    assert report.values_filled == 2
    assert report.max_gap == 3
    assert report.fill_dim == "time"
    assert report.variables == tuple(small_dataset.data_vars)
    assert report.as_dict()["derived_variables"] == ["wind_speed", "wind_direction"]


def test_pipeline_writes_a_valid_processed_netcdf(processed_result, raw_generation) -> None:
    output_path = processed_result.output_path
    assert output_path is not None and output_path.is_file()

    reloaded = load_dataset(output_path, required_variables=("precipitation", "wind_speed"))
    raw = load_dataset(raw_generation.dataset_path)

    assert set(reloaded.data_vars) == set(raw.data_vars) | {"wind_speed", "wind_direction"}
    assert tuple(reloaded["precipitation"].dims) == ("time", "latitude", "longitude")
    assert reloaded.sizes["time"] == raw.sizes["time"]
    assert str(reloaded["time"].values[0]) == str(raw["time"].values[0])
    assert reloaded["precipitation"].attrs["units"] == raw["precipitation"].attrs["units"]
    assert reloaded.attrs["processed"] == "true"
    assert reloaded.attrs["project"] == "SIH26078"
    assert "missing_value_policy" in reloaded.attrs
    assert reloaded.attrs["random_seed"] == raw.attrs["random_seed"]  # provenance preserved

    expected_speed = np.sqrt(
        reloaded["u_wind"].values.astype("float64") ** 2
        + reloaded["v_wind"].values.astype("float64") ** 2
    )
    assert np.allclose(reloaded["wind_speed"].values, expected_speed, rtol=1e-5)


def test_pipeline_creates_the_output_directory(tmp_path: Path, raw_generation) -> None:
    destination = tmp_path / "deeply" / "nested" / "processed" / "sample_history_processed.nc"
    assert not destination.parent.exists()

    result = preprocess_file(raw_generation.dataset_path, destination)

    assert destination.parent.is_dir()
    assert destination.is_file()
    assert result.output_path == destination
    assert result.report.input_path == str(raw_generation.dataset_path)


def test_pipeline_is_deterministic(raw_generation, tmp_path: Path) -> None:
    first = preprocess_file(raw_generation.dataset_path, tmp_path / "first.nc", max_gap=3)
    second = preprocess_file(raw_generation.dataset_path, tmp_path / "second.nc", max_gap=3)

    xr.testing.assert_allclose(first.dataset, second.dataset)
    assert first.report.as_dict() | {"output_path": None} == second.report.as_dict() | {
        "output_path": None
    }


def test_pipeline_rejects_an_invalid_dataset(small_dataset) -> None:
    broken = small_dataset.drop_vars("u_wind")

    with pytest.raises(DatasetValidationError, match="u_wind"):
        preprocess_dataset(broken)


def test_pipeline_validates_the_file_before_processing(tmp_path: Path, small_dataset) -> None:
    from extreme_weather.data.loader import save_dataset

    broken_path = tmp_path / "broken.nc"
    save_dataset(small_dataset.drop_vars("temperature"), broken_path)

    with pytest.raises(DatasetValidationError, match="temperature"):
        preprocess_file(broken_path, tmp_path / "out.nc")


def test_default_processed_path_uses_the_processed_directory(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "sample_history.nc"

    derived = default_processed_path(source)
    assert derived.name == "sample_history_processed.nc"

    explicit = default_processed_path(source, output_dir=tmp_path / "custom")
    assert explicit == tmp_path / "custom" / "sample_history_processed.nc"
    assert explicit.parent == tmp_path / "custom"
