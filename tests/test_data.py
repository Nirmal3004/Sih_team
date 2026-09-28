"""Tests for synthetic generation, validation, loading and derived variables.

Every assertion checks *behaviour* (values, shapes, physical plausibility,
provenance content) rather than mere existence of files or variables.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extreme_weather.config import (
    DOWNSAMPLE_TARGET_RESOLUTION_DEG,
    KM_PER_DEGREE_LATITUDE,
    NWP_NATIVE_RESOLUTION_DEG,
    PROJECT_ID,
)
from extreme_weather.data.derived import add_derived_variables, wind_direction, wind_speed
from extreme_weather.data.generator import (
    MANIFEST_SUFFIX,
    SYNTHETIC_SOURCE,
    VARIABLES,
    SyntheticConfig,
    generate_and_save,
    generate_synthetic_dataset,
    manifest_path_for,
)
from extreme_weather.data.loader import (
    DatasetLoadError,
    load_dataset,
    resolve_dataset_path,
    save_dataset,
)
from extreme_weather.data.validator import (
    DatasetValidationError,
    collect_validation_issues,
    summarise_dataset,
    validate_dataset,
)


# --------------------------------------------------------------------------- #
# 1. Synthetic data generation
# --------------------------------------------------------------------------- #
def test_generation_is_deterministic_for_a_fixed_seed(small_config: SyntheticConfig) -> None:
    first = generate_synthetic_dataset(small_config)
    second = generate_synthetic_dataset(small_config)

    xr.testing.assert_identical(first, second)
    assert float(first["precipitation"].max()) == float(second["precipitation"].max())


def test_different_seeds_produce_different_fields(small_config: SyntheticConfig) -> None:
    other = SyntheticConfig(
        seed=small_config.seed + 1,
        time_steps=small_config.time_steps,
        latitude_min=small_config.latitude_min,
        latitude_max=small_config.latitude_max,
        longitude_min=small_config.longitude_min,
        longitude_max=small_config.longitude_max,
        resolution_deg=small_config.resolution_deg,
        event=small_config.event,
    )
    baseline = generate_synthetic_dataset(small_config)
    alternative = generate_synthetic_dataset(other)

    assert not np.allclose(baseline["temperature"].values, alternative["temperature"].values)

    # The seed only perturbs the *background* field: the injected event is
    # seed independent, so both runs peak at the event core and the difference
    # between them stays far below the injected intensity.
    event = small_config.event
    window = slice(event.start_index, event.start_index + event.duration_steps)
    difference = np.abs(
        baseline["precipitation"].isel(time=window).values
        - alternative["precipitation"].isel(time=window).values
    )

    assert float(baseline["precipitation"].max()) > event.peak_precipitation_mm_per_hour * 0.8
    assert float(alternative["precipitation"].max()) > event.peak_precipitation_mm_per_hour * 0.8
    assert float(difference.max()) < 0.25 * event.peak_precipitation_mm_per_hour


def test_default_config_respects_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from extreme_weather.config import Settings
    from extreme_weather.data.generator import default_config

    settings = Settings(
        synthetic_seed=99,
        synthetic_time_steps=40,
        synthetic_grid_resolution_deg=0.2,
        synthetic_latitude_min=5.0,
        synthetic_latitude_max=9.0,
        synthetic_longitude_min=70.0,
        synthetic_longitude_max=74.0,
    )
    config = default_config(settings=settings)

    assert config.seed == 99
    assert config.time_steps == 40
    assert config.resolution_deg == 0.2
    assert config.latitude.size == 21  # (9 - 5) / 0.2 + 1
    assert config.longitude.size == 21
    # The default event centre (13, 77) is outside this domain, so it must fall
    # back to the domain centre instead of raising.
    assert config.event.center_latitude == pytest.approx(7.0)
    assert config.event.center_longitude == pytest.approx(72.0)
    assert config.event.start_index == 13  # round(40 / 3)
    assert config.event.duration_steps == 7  # round(40 / 6)


# --------------------------------------------------------------------------- #
# 2. Dimensions
# --------------------------------------------------------------------------- #
def test_dataset_dimensions_match_the_configuration(small_config: SyntheticConfig, small_dataset) -> None:
    assert small_dataset.sizes["time"] == small_config.time_steps
    assert small_dataset.sizes["latitude"] == small_config.latitude.size
    assert small_dataset.sizes["longitude"] == small_config.longitude.size
    assert tuple(small_dataset.dims) == ("time", "latitude", "longitude")

    latitude = small_dataset["latitude"].values
    longitude = small_dataset["longitude"].values
    assert latitude[0] == pytest.approx(12.0)
    assert latitude[-1] == pytest.approx(14.0)
    assert longitude[0] == pytest.approx(76.0)
    assert longitude[-1] == pytest.approx(78.0)
    assert np.all(np.diff(latitude) > 0.0)
    assert np.all(np.diff(longitude) > 0.0)
    assert np.allclose(np.diff(latitude), 0.25)


def test_time_coordinate_is_hourly_and_increasing(small_dataset) -> None:
    time = small_dataset["time"].values
    assert np.issubdtype(time.dtype, np.datetime64)
    assert np.all(np.diff(time) == np.timedelta64(1, "h"))
    assert str(time[0]) == "2024-06-01T00:00:00"


# --------------------------------------------------------------------------- #
# 3. Required variables and physical plausibility
# --------------------------------------------------------------------------- #
def test_required_variables_exist_with_units_and_plausible_values(small_dataset) -> None:
    for name in VARIABLES:
        assert name in small_dataset.data_vars, f"missing synthetic variable {name}"
        variable = small_dataset[name]
        assert np.issubdtype(variable.dtype, np.floating)
        assert variable.attrs.get("units"), f"{name} has no units attribute"
        assert variable.attrs.get("long_name"), f"{name} has no long_name attribute"
        assert not np.isnan(variable.values).any(), f"{name} contains unexpected NaN values"
        assert variable.dims == ("time", "latitude", "longitude")

    precipitation = small_dataset["precipitation"].values
    temperature = small_dataset["temperature"].values
    pressure = small_dataset["pressure"].values
    humidity = small_dataset["relative_humidity"].values
    u_wind = small_dataset["u_wind"].values
    v_wind = small_dataset["v_wind"].values

    assert precipitation.min() >= 0.0
    assert 250.0 < temperature.min() and temperature.max() < 320.0
    assert 95_000.0 < pressure.min() and pressure.max() < 105_000.0
    assert 0.0 <= humidity.min() and humidity.max() <= 100.0
    assert np.abs(u_wind).max() <= 45.0
    assert np.abs(v_wind).max() <= 45.0

    # Background precipitation must stay well below the injected event peak.
    assert float(np.median(precipitation)) < 2.0


def test_dataset_attributes_label_the_data_as_synthetic(small_dataset, small_config) -> None:
    attrs = small_dataset.attrs
    assert attrs["project"] == PROJECT_ID
    assert attrs["dataset_type"] == "synthetic"
    assert attrs["source"] == SYNTHETIC_SOURCE
    assert "synthetic" in attrs["summary"].lower()
    assert int(attrs["random_seed"]) == small_config.seed
    assert float(attrs["grid_resolution_deg"]) == pytest.approx(0.25)
    assert float(attrs["approximate_grid_resolution_km"]) == pytest.approx(
        0.25 * KM_PER_DEGREE_LATITUDE, rel=1e-4
    )
    assert float(attrs["nwp_native_reference_resolution_deg"]) == pytest.approx(
        NWP_NATIVE_RESOLUTION_DEG
    )
    assert float(attrs["downscaling_target_resolution_deg"]) == pytest.approx(
        DOWNSAMPLE_TARGET_RESOLUTION_DEG
    )


# --------------------------------------------------------------------------- #
# 4. Manifest provenance
# --------------------------------------------------------------------------- #
def test_manifest_documents_provenance(raw_generation, small_config) -> None:
    manifest_path = raw_generation.manifest_path
    assert manifest_path is not None
    assert manifest_path.name == f"sample_history{MANIFEST_SUFFIX}"
    assert manifest_path == manifest_path_for(raw_generation.dataset_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "1.0"
    assert manifest["project"]["id"] == PROJECT_ID
    assert manifest["generator"]["module"] == "extreme_weather.data.generator"
    assert manifest["generator"]["version"] == raw_generation.manifest["generator"]["version"]
    assert manifest["random_seed"] == small_config.seed

    dataset_info = manifest["dataset"]
    assert dataset_info["synthetic"] is True
    assert dataset_info["source"] == SYNTHETIC_SOURCE
    assert "SYNTHETIC DATA" in dataset_info["statement"]
    assert dataset_info["file"] == "sample_history.nc"
    assert dataset_info["file_bytes"] == raw_generation.dataset_path.stat().st_size
    assert dataset_info["file_bytes"] > 0
    assert len(dataset_info["sha256"]) == 64

    assert manifest["dimensions"] == {"time": 16, "latitude": 9, "longitude": 9}
    assert [entry["name"] for entry in manifest["variables"]] == list(VARIABLES)
    assert all(entry["units"] for entry in manifest["variables"])

    assert manifest["coordinates"]["longitude"]["resolution_deg"] == pytest.approx(0.25)
    assert manifest["coordinates"]["time"]["step_hours"] == 1
    assert manifest["grid_resolution"]["degrees"] == pytest.approx(0.25)
    assert manifest["grid_resolution"]["approximate_km"] == pytest.approx(27.83, abs=0.01)
    assert manifest["grid_resolution"]["downscaling_target_reference_deg"] == pytest.approx(
        DOWNSAMPLE_TARGET_RESOLUTION_DEG
    )

    event = manifest["event_injection"]
    assert event["enabled"] is True
    assert event["name"] == small_config.event.name
    assert event["center_latitude"] == pytest.approx(13.0)
    assert event["start_index"] == 4
    assert event["duration_steps"] == 6
    assert event["peak_precipitation_mm_per_hour"] == pytest.approx(55.0)


def test_manifest_is_reproducible_for_a_fixed_timestamp(
    tmp_path: Path, small_config: SyntheticConfig
) -> None:
    timestamp = "2024-06-01T00:00:00Z"
    first = generate_and_save(
        tmp_path / "first.nc",
        config=small_config,
        created_utc=timestamp,
        write_manifest_file=False,
    )
    second = generate_and_save(
        tmp_path / "second.nc",
        config=small_config,
        created_utc=timestamp,
        write_manifest_file=False,
    )

    first_manifest = dict(first.manifest)
    second_manifest = dict(second.manifest)
    # File name/size/checksum legitimately differ; everything else must match.
    first_manifest.pop("dataset")
    second_manifest.pop("dataset")
    assert first_manifest == second_manifest
    assert first_manifest["created_utc"] == timestamp
    xr.testing.assert_identical(first.dataset, second.dataset)


def test_manifest_written_next_to_a_custom_output_path(
    tmp_path: Path, small_config: SyntheticConfig
) -> None:
    config = SyntheticConfig(
        seed=11,
        time_steps=6,
        resolution_deg=small_config.resolution_deg,
        latitude_min=small_config.latitude_min,
        latitude_max=small_config.latitude_max,
        longitude_min=small_config.longitude_min,
        longitude_max=small_config.longitude_max,
        event=replace(small_config.event, start_index=1, duration_steps=3),
    )
    result = generate_and_save(tmp_path / "nested" / "custom.nc", config=config)

    assert result.dataset_path == tmp_path / "nested" / "custom.nc"
    assert result.manifest_path == tmp_path / "nested" / "custom.manifest.json"
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert result.manifest["dimensions"] == {"time": 6, "latitude": 9, "longitude": 9}


def test_invalid_configuration_is_rejected_with_a_clear_message(small_config) -> None:
    with pytest.raises(ValueError, match="Event window exceeds the time axis"):
        SyntheticConfig(
            seed=1,
            time_steps=4,
            event=replace(small_config.event, start_index=2, duration_steps=6),
        )

    with pytest.raises(ValueError, match="is outside the domain"):
        SyntheticConfig(
            seed=1,
            time_steps=16,
            latitude_min=40.0,
            latitude_max=44.0,
            event=small_config.event,
        )

    with pytest.raises(ValueError, match="resolution_deg must be > 0"):
        SyntheticConfig(seed=1, time_steps=4, resolution_deg=0.0)


# --------------------------------------------------------------------------- #
# 5. Injected extreme event behaviour
# --------------------------------------------------------------------------- #
def test_event_is_temporally_localised(small_dataset, quiet_dataset, small_config) -> None:
    event = small_config.event
    window = slice(event.start_index, event.start_index + event.duration_steps)

    difference = (small_dataset["precipitation"] - quiet_dataset["precipitation"]).values
    # Outside the injection window the two datasets must be identical ...
    mask = np.ones(small_dataset.sizes["time"], dtype=bool)
    mask[window] = False
    assert np.allclose(difference[mask], 0.0, atol=1e-6)
    # ... and inside the window the injected rain must be substantial.
    assert difference[window].max() > 40.0


def test_event_is_spatially_localised_around_its_centre(
    small_dataset, quiet_dataset, small_config
) -> None:
    event = small_config.event
    window = slice(event.start_index, event.start_index + event.duration_steps)

    peak = small_dataset["precipitation"].isel(time=window).max(dim="time")
    flat_index = int(np.argmax(np.asarray(peak.values)))
    peak_latitude = float(peak["latitude"].values[flat_index // peak.sizes["longitude"]])
    peak_longitude = float(peak["longitude"].values[flat_index % peak.sizes["longitude"]])

    assert peak_latitude == pytest.approx(event.center_latitude, abs=0.25)
    assert peak_longitude == pytest.approx(event.center_longitude, abs=0.25)
    assert float(peak.max()) > event.peak_precipitation_mm_per_hour * 0.8

    # A far corner receives essentially no extra rain from the event.
    corner_difference = float(
        (
            small_dataset["precipitation"]
            - quiet_dataset["precipitation"]
        )
        .isel(time=window, latitude=0, longitude=0, drop=True)
        .max()
    )
    assert corner_difference < 0.01


def test_event_peak_happens_in_the_middle_of_the_window(small_dataset, small_config) -> None:
    event = small_config.event
    window = slice(event.start_index, event.start_index + event.duration_steps)
    time_series = small_dataset["precipitation"].isel(latitude=4, longitude=4).values

    peak_index = int(np.argmax(time_series))
    middle_start = event.start_index + event.duration_steps // 3
    middle_end = event.start_index + 2 * event.duration_steps // 3

    assert middle_start <= peak_index <= middle_end
    assert time_series[event.start_index] < time_series[peak_index]
    assert time_series[event.start_index + event.duration_steps - 1] < time_series[peak_index]
    # Before the window the centre point only carries background rain.
    assert float(time_series[: event.start_index].max()) < 5.0
    assert window.start < window.stop


def test_event_cools_temperature_and_deepens_pressure(small_dataset, quiet_dataset, small_config) -> None:
    event = small_config.event
    window = slice(event.start_index, event.start_index + event.duration_steps)
    centre = {"latitude": 4, "longitude": 4}

    temperature_difference = (
        small_dataset["temperature"].isel(time=window, **centre).values
        - quiet_dataset["temperature"].isel(time=window, **centre).values
    )
    pressure_difference = (
        small_dataset["pressure"].isel(time=window, **centre).values
        - quiet_dataset["pressure"].isel(time=window, **centre).values
    )

    assert temperature_difference.min() == pytest.approx(event.temperature_anomaly_k, rel=0.05)
    assert pressure_difference.min() == pytest.approx(event.pressure_anomaly_pa, rel=0.05)


def test_disabled_event_leaves_only_background_rain(quiet_dataset, small_dataset) -> None:
    precipitation = quiet_dataset["precipitation"].values
    assert precipitation.min() >= 0.0
    # Background rain stays an order of magnitude below the injected event peak
    # (the realised maximum depends on the seed, hence the generous bound).
    assert float(precipitation.max()) < 10.0
    assert float(precipitation.max()) < 0.25 * float(small_dataset["precipitation"].max())
    assert quiet_dataset.attrs["event_injection"]
    assert '"enabled": false' in quiet_dataset.attrs["event_injection"]


# --------------------------------------------------------------------------- #
# 6. Dataset validation
# --------------------------------------------------------------------------- #
def test_generated_dataset_passes_validation(small_dataset) -> None:
    issues = validate_dataset(small_dataset)
    assert issues.ok
    assert issues.warnings == ()


def test_validation_reports_missing_variable_with_a_useful_message(small_dataset) -> None:
    incomplete = small_dataset.drop_vars("pressure")
    issues = collect_validation_issues(incomplete)

    assert not issues.ok
    assert any("pressure" in message for message in issues.errors)

    with pytest.raises(DatasetValidationError, match="pressure"):
        validate_dataset(incomplete)


def test_validation_detects_missing_dimension(small_dataset) -> None:
    broken = small_dataset.drop_dims("longitude")
    issues = collect_validation_issues(broken)

    assert not issues.ok
    assert any("longitude" in message for message in issues.errors)


def test_validation_detects_non_monotonic_latitude(small_dataset) -> None:
    shuffled = small_dataset.copy(deep=True)
    latitude = shuffled["latitude"].values.copy()
    latitude[2], latitude[3] = latitude[3], latitude[2]
    shuffled = shuffled.assign_coords(latitude=latitude)

    issues = collect_validation_issues(shuffled)
    assert not issues.ok
    assert any("latitude" in message and "monotonic" in message for message in issues.errors)


def test_validation_detects_out_of_range_latitude(small_dataset) -> None:
    broken = small_dataset.assign_coords(latitude=small_dataset["latitude"].values * 100.0)
    issues = collect_validation_issues(broken)

    assert not issues.ok
    assert any("latitude" in message for message in issues.errors)


def test_validation_detects_unsorted_time(small_dataset) -> None:
    broken = small_dataset.assign_coords(time=small_dataset["time"].values[::-1])
    issues = collect_validation_issues(broken)

    assert not issues.ok
    assert any("increasing" in message for message in issues.errors)


def test_validation_warns_about_all_missing_variable(small_dataset) -> None:
    with_gap = small_dataset.copy(deep=True)
    with_gap["temperature"] = xr.full_like(with_gap["temperature"], np.nan)

    issues = validate_dataset(with_gap)  # must not raise
    assert issues.ok
    assert any("temperature" in message and "missing" in message for message in issues.warnings)


def test_summary_reports_grid_and_coordinates(raw_generation, small_config) -> None:
    summary = summarise_dataset(
        raw_generation.dataset, name="sample_history", path=str(raw_generation.dataset_path)
    )

    assert summary.dimensions == {"time": 16, "latitude": 9, "longitude": 9}
    assert summary.data_variables == VARIABLES
    assert summary.grid_resolution_deg == pytest.approx(0.25)
    assert summary.approximate_grid_resolution_km == pytest.approx(27.83, abs=0.01)
    assert summary.time_step_seconds == pytest.approx(3600.0)
    assert summary.time_start == "2024-06-01T00:00:00"
    assert summary.latitude_min == pytest.approx(12.0)
    assert summary.longitude_max == pytest.approx(78.0)
    assert summary.is_synthetic is True
    assert summary.attributes["random_seed"] == small_config.seed
    assert summary.path == str(raw_generation.dataset_path)


# --------------------------------------------------------------------------- #
# 7. Derived wind variables
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("u_value", "v_value", "expected_speed"),
    [
        (3.0, 4.0, 5.0),
        (0.0, 0.0, 0.0),
        (-6.0, -8.0, 10.0),
        (-12.0, 5.0, 13.0),
    ],
)
def test_wind_speed_matches_pythagoras(u_value, v_value, expected_speed) -> None:
    u_wind = xr.DataArray(np.full((2, 3), u_value, dtype="float32"), dims=("y", "x"))
    v_wind = xr.DataArray(np.full((2, 3), v_value, dtype="float32"), dims=("y", "x"))

    speed = wind_speed(u_wind, v_wind)

    assert speed.name == "wind_speed"
    assert speed.dims == ("y", "x")
    assert speed.attrs["units"] == "m s-1"
    assert np.allclose(speed.values, expected_speed, atol=1e-5)


def test_wind_speed_matches_numpy_on_the_synthetic_dataset(small_dataset) -> None:
    derived = add_derived_variables(small_dataset)

    expected = np.sqrt(
        small_dataset["u_wind"].values.astype("float64") ** 2
        + small_dataset["v_wind"].values.astype("float64") ** 2
    )
    assert "wind_speed" in derived.data_vars
    assert "wind_direction" in derived.data_vars
    assert derived["wind_speed"].dims == ("time", "latitude", "longitude")
    assert np.allclose(derived["wind_speed"].values, expected, rtol=1e-6)
    assert derived["wind_speed"].attrs["units"] == "m s-1"
    assert "wind_speed" in derived.attrs["derived_variables"]


@pytest.mark.parametrize(
    ("u_value", "v_value", "expected_direction"),
    [
        (0.0, -1.0, 0.0),  # blowing towards the south -> wind from the north
        (-1.0, 0.0, 90.0),  # blowing towards the west -> wind from the east
        (0.0, 1.0, 180.0),  # blowing towards the north -> wind from the south
        (1.0, 0.0, 270.0),  # blowing towards the east -> wind from the west
    ],
)
def test_wind_direction_uses_meteorological_convention(u_value, v_value, expected_direction) -> None:
    u_wind = xr.DataArray([u_value], dims=("point",))
    v_wind = xr.DataArray([v_value], dims=("point",))

    direction = wind_direction(u_wind, v_wind)

    assert direction.name == "wind_direction"
    assert direction.attrs["units"] == "degree"
    assert "clockwise from true north" in direction.attrs["convention"]
    assert float(direction.values[0]) == pytest.approx(expected_direction, abs=1e-6)


def test_add_derived_variables_is_optional_and_non_destructive(small_dataset) -> None:
    only_speed = add_derived_variables(small_dataset, include_wind_direction=False)

    assert "wind_speed" in only_speed.data_vars
    assert "wind_direction" not in only_speed.data_vars
    # The input dataset must not gain the derived variables.
    assert "wind_speed" not in small_dataset.data_vars
    assert "wind_direction" not in small_dataset.data_vars


def test_add_derived_variables_requires_wind_components(small_dataset) -> None:
    without_wind = small_dataset.drop_vars(["u_wind"])
    with pytest.raises(KeyError, match="u_wind"):
        add_derived_variables(without_wind)


# --------------------------------------------------------------------------- #
# 8. NetCDF loading / saving
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tmp_path: Path, small_dataset, small_config) -> None:
    target = tmp_path / "roundtrip" / "history.nc"
    saved = save_dataset(small_dataset, target)
    assert saved == target

    loaded = load_dataset(saved)

    assert tuple(loaded.dims) == ("time", "latitude", "longitude")
    assert set(loaded.data_vars) == set(VARIABLES)
    assert loaded.attrs["random_seed"] == small_config.seed
    assert loaded.attrs["dataset_type"] == "synthetic"
    # Xarray decodes the time coordinate into its own resolution unit
    # (datetime64[ns] after a round trip), so compare values, not strings.
    assert np.array_equal(loaded["time"].values, small_dataset["time"].values)
    assert str(loaded["time"].values[0])[:16] == "2024-06-01T00:00"
    for name in VARIABLES:
        assert np.allclose(
            loaded[name].values, small_dataset[name].values.astype("float32"), equal_nan=True
        )


def test_loader_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset(tmp_path / "does_not_exist.nc")


def test_loader_wraps_unreadable_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.nc"
    broken.write_text("this is definitely not a netcdf file", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Could not read NetCDF"):
        load_dataset(broken)


def test_loader_validates_structure_on_load(tmp_path: Path, small_dataset) -> None:
    target = tmp_path / "incomplete.nc"
    save_dataset(small_dataset.drop_vars("relative_humidity"), target)

    with pytest.raises(DatasetValidationError, match="relative_humidity"):
        load_dataset(target)

    # Validation can be disabled for structural inspection only.
    lenient = load_dataset(target, validate=False)
    assert "relative_humidity" not in lenient.data_vars


def test_resolve_dataset_path_rejects_escapes_and_wrong_suffix(tmp_path: Path) -> None:
    resolved = resolve_dataset_path(tmp_path, "raw/sample_history.nc")
    assert resolved == (tmp_path / "raw" / "sample_history.nc").resolve()

    with pytest.raises(DatasetLoadError, match="escapes the data directory"):
        resolve_dataset_path(tmp_path, "../outside.nc")

    with pytest.raises(DatasetLoadError, match="Unsupported dataset extension"):
        resolve_dataset_path(tmp_path, "raw/sample_history.txt")
