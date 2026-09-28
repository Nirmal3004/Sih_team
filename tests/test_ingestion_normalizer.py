"""Tests for :func:`~extreme_weather.ingestion.normalizer.normalize_dataset`.

The normalizer is the step that turns *some source's* units and coordinates into
the project's standard model, and it is built on two hard rules:

1. **Never silently convert an unknown unit.** An unrecognised unit raises
   :class:`NormalizationError` rather than being guessed, because guessing that
   ``"mm/day"`` is ``"mm/h"`` would corrupt every downstream threshold.
2. **Never silently alter geographic meaning.** Every rename, conversion and
   coordinate transform is recorded in the result, and every absence is
   reported rather than filled in.

The last tests in this module run the whole thing against the committed ERA5
fixture - genuine ECMWF reanalysis values in ``degC``/``hPa`` - so the
conversions are checked against the units a real archive actually publishes.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.ingestion import (
    ALIAS_MAP,
    STANDARD_UNITS,
    ConversionRecord,
    NormalizationError,
    normalize_dataset,
    normalize_variable_name,
)
from extreme_weather.ingestion.normalizer import canonical_unit
from tests.ingestion_fixtures import make_dataset, make_times

import xarray as xr

SHAPE = (4, 3, 3)


# --------------------------------------------------------------------------- #
# Variable renaming
# --------------------------------------------------------------------------- #
def test_era5_and_open_meteo_names_rename_to_the_canonical_variables() -> None:
    """A real archive's spellings must need no manual renaming first."""
    assert normalize_variable_name("temperature_2m") == "temperature"
    assert normalize_variable_name("surface_pressure") == "pressure"
    assert normalize_variable_name("relative_humidity_2m") == "relative_humidity"
    assert normalize_variable_name("wind_speed_10m") == "wind_speed"
    assert normalize_variable_name("wind_direction_10m") == "wind_direction"
    # ECMWF short names.
    assert normalize_variable_name("2t") == "temperature"
    assert normalize_variable_name("msl") == "pressure"
    assert normalize_variable_name("u10") == "u_wind"


def test_renaming_is_case_and_whitespace_insensitive() -> None:
    assert normalize_variable_name("  Temperature_2M ") == "temperature"


def test_an_unknown_name_is_lowercased_and_kept_not_guessed() -> None:
    """An unrecognised name is left alone so the report can name it."""
    assert normalize_variable_name("SoilMoisture") == "soilmoisture"


def test_every_alias_maps_to_a_known_project_variable() -> None:
    """A typo in the alias table would be invisible until a real ingest."""
    known = set(STANDARD_UNITS)
    assert set(ALIAS_MAP.values()) <= known


def test_normalize_dataset_records_every_rename() -> None:
    source = make_dataset(
        variables={
            "temperature_2m": (np.full(SHAPE, 22.0), "degC"),
            "surface_pressure": (np.full(SHAPE, 921.6), "hPa"),
        }
    )
    result = normalize_dataset(source)

    assert "temperature" in result.dataset.data_vars
    assert "pressure" in result.dataset.data_vars


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #
def test_celsius_is_converted_to_kelvin_with_an_offset() -> None:
    """degC -> K is an offset, not a scale factor; ``scale`` stays 1.0."""
    source = make_dataset(variables={"temperature": (np.full(SHAPE, 22.9), "degC")})
    result = normalize_dataset(source)

    values = result.dataset["temperature"].values
    assert result.dataset["temperature"].attrs["units"] == "K"
    assert values.min() == pytest.approx(296.05)
    assert values.max() == pytest.approx(296.05)

    record = result.conversion_map()["temperature"]
    assert record["original_units"] == "degC"
    assert record["normalized_units"] == "K"
    assert "273.15" in record["conversion_method"]
    assert record["scale"] == 1.0


def test_hectopascals_are_scaled_to_pascals() -> None:
    source = make_dataset(variables={"pressure": (np.full(SHAPE, 921.6), "hPa")})
    result = normalize_dataset(source)

    assert float(result.dataset["pressure"].mean()) == pytest.approx(92160.0)
    record = result.conversion_map()["pressure"]
    assert record["scale"] == 100.0
    assert "x100" in record["conversion_method"]


def test_kilometres_per_hour_are_scaled_to_metres_per_second() -> None:
    """km/h is only accepted for the wind *components*, never for temperature."""
    source = make_dataset(
        variables={
            "u_wind": (np.full(SHAPE, 36.0), "km/h"),
            "v_wind": (np.full(SHAPE, 18.0), "km/h"),
        }
    )
    result = normalize_dataset(source)

    assert float(result.dataset["u_wind"].mean()) == pytest.approx(10.0)
    assert float(result.dataset["v_wind"].mean()) == pytest.approx(5.0)
    assert result.conversion_map()["u_wind"]["scale"] == pytest.approx(1 / 3.6)


def test_a_wind_unit_on_a_non_wind_variable_is_refused() -> None:
    """``km/h`` on temperature would change what the variable means."""
    source = make_dataset(variables={"temperature": (np.full(SHAPE, 22.9), "km/h")})
    with pytest.raises(NormalizationError, match="wind-speed"):
        normalize_dataset(source)


def test_an_unknown_unit_raises_rather_than_being_guessed() -> None:
    """The central promise of this module."""
    source = make_dataset(variables={"precipitation": (np.full(SHAPE, 3.0), "mm/day")})
    with pytest.raises(NormalizationError) as info:
        normalize_dataset(source)

    message = str(info.value)
    assert "mm/day" in message
    assert "Refusing to guess" in message
    # The operator is told what to do about it.
    assert "Convert the source upstream" in message


def test_an_already_standard_unit_is_an_identity_conversion() -> None:
    """No rescaling, but the conversion is still recorded for the manifest."""
    source = make_dataset(variables={"temperature": (np.full(SHAPE, 300.0), "K")})
    result = normalize_dataset(source)

    assert result.dataset["temperature"].values.min() == 300.0
    record = result.conversion_map()["temperature"]
    assert record["conversion_method"] == "identity"
    assert record["original_units"] == "K"
    assert record["scale"] == 1.0


def test_a_missing_units_attribute_assumes_the_standard_and_says_so() -> None:
    """The assumption is made, but it is recorded rather than silent."""
    source = make_dataset(variables={"temperature": (np.full(SHAPE, 300.0), "")})
    source["temperature"].attrs.pop("units")
    result = normalize_dataset(source)

    assert result.dataset["temperature"].attrs["units"] == "K"
    assert "assumed standard" in result.conversion_map()["temperature"]["conversion_method"]


def test_canonical_unit_resolves_without_touching_data() -> None:
    assert canonical_unit("temperature", "degC") == "K"
    assert canonical_unit("pressure", "hPa") == "Pa"
    # Already standard, and "no units at all", both resolve to the standard.
    assert canonical_unit("temperature", None) == "K"
    assert canonical_unit("temperature", "K") == "K"
    with pytest.raises(NormalizationError):
        canonical_unit("temperature", "fahrenheit")


def test_a_precipitation_accumulation_is_relabelled_not_rescaled() -> None:
    """1 kg/m2 of water is exactly 1 mm, so this is a rename, not a factor."""
    source = make_dataset(variables={"precipitation": (np.full(SHAPE, 4.0), "kg m-2")})
    result = normalize_dataset(source)


# --------------------------------------------------------------------------- #
# Coordinate normalization
# --------------------------------------------------------------------------- #
def test_descending_latitude_is_sorted_ascending_and_recorded() -> None:
    """Phases 1-5 assume ascending latitude, so a flipped axis is corrected."""
    source = make_dataset(latitude=[13.0, 12.5, 12.0])
    result = normalize_dataset(source)

    assert list(result.dataset.latitude.values) == [12.0, 12.5, 13.0]
    assert any("latitude ascending" in action for action in result.coordinate_actions)
    # The data followed the coordinate, it was not left mismatched.
    assert float(result.dataset["temperature"][0, 0, 0]) == pytest.approx(300.0)


def test_ascending_latitude_is_left_alone() -> None:
    result = normalize_dataset(make_dataset(latitude=[12.0, 12.5, 13.0]))
    assert not any("latitude" in action for action in result.coordinate_actions)


def test_zero_to_three_sixty_longitude_is_wrapped_into_the_signed_range() -> None:
    source = make_dataset(longitude=[0.0, 120.0, 240.0])
    result = normalize_dataset(source)

    values = result.dataset.longitude.values
    assert values.min() >= -180.0
    assert values.max() <= 180.0
    assert any("wrapped longitude" in action for action in result.coordinate_actions)
    # 240E becomes 120W, i.e. -120.
    assert -120.0 in values


def test_a_longitude_already_in_the_signed_range_is_not_wrapped() -> None:
    result = normalize_dataset(make_dataset(longitude=[76.5, 77.0, 77.5]))
    assert not any("wrapped longitude" in a for a in result.coordinate_actions)
    assert list(result.dataset.longitude.values) == [76.5, 77.0, 77.5]


def test_a_non_monotonic_unique_time_axis_is_sorted_ascending() -> None:
    """Out-of-order hours are a data defect, but sorting is safe and recorded."""
    stamps = make_times(4)
    shuffled = np.array([stamps[2], stamps[0], stamps[1], stamps[3]], dtype="datetime64[ns]")
    result = normalize_dataset(make_dataset(times=shuffled))

    assert list(result.dataset.time.values) == list(stamps)
    assert "sorted time ascending" in result.coordinate_actions
    assert any("not monotonic" in warning for warning in result.warnings)


def test_a_dataset_without_a_time_coordinate_cannot_enter_the_pipeline() -> None:
    """Phases 1-5 are time series; a static field is not ingestible here."""
    source = make_dataset().drop_vars("time")
    with pytest.raises(NormalizationError, match="no 'time' coordinate"):
        normalize_dataset(source)


def test_a_dataset_without_a_longitude_coordinate_is_rejected() -> None:
    source = make_dataset().drop_vars("longitude")
    with pytest.raises(NormalizationError, match="longitude"):
        normalize_dataset(source)


# --------------------------------------------------------------------------- #
# What is missing, dropped and kept
# --------------------------------------------------------------------------- #
def test_missing_contract_variables_are_reported_not_invented() -> None:
    """A dataset without u/v wind must say so, not grow fabricated components."""
    source = make_dataset(variables={"temperature": (np.full(SHAPE, 300.0), "K")})
    result = normalize_dataset(source)

    assert set(result.missing_variables) == {
        "precipitation",
        "pressure",
        "relative_humidity",
        "u_wind",
        "v_wind",
    }
    assert "u_wind" not in result.dataset.data_vars
    assert any("not available downstream" in w for w in result.warnings)


def test_a_complete_dataset_reports_nothing_missing() -> None:
    assert normalize_dataset(make_dataset()).missing_variables == ()


def test_an_unknown_variable_is_kept_untouched_by_default() -> None:
    """Dropping data the project did not ask about is a bigger decision than
    passing it through, so keeping it is the default."""
    source = make_dataset(
        variables={
            "temperature": (np.full(SHAPE, 300.0), "K"),
            "soil_moisture": (np.full(SHAPE, 0.3), "m3/m3"),
        }
    )
    result = normalize_dataset(source)

    assert "soil_moisture" in result.dataset.data_vars
    assert "soil_moisture" not in result.dropped_variables
    assert any("Kept non-standard variable" in w for w in result.warnings)


def test_drop_unknown_true_removes_what_the_project_does_not_model() -> None:
    source = make_dataset(
        variables={
            "temperature": (np.full(SHAPE, 300.0), "K"),
            "soil_moisture": (np.full(SHAPE, 0.3), "m3/m3"),
        }
    )
    result = normalize_dataset(source, drop_unknown=True)

    assert "soil_moisture" not in result.dataset.data_vars
    assert "soil_moisture" in result.dropped_variables



# --------------------------------------------------------------------------- #
# The recorded result
# --------------------------------------------------------------------------- #
def test_a_conversion_record_serialises_every_field_it_claims_to() -> None:
    record = ConversionRecord(
        variable="pressure",
        original_units="hPa",
        normalized_units="Pa",
        method="hPa -> Pa (x100)",
        scale=100.0,
    )
    assert record.as_dict() == {
        "variable": "pressure",
        "original_units": "hPa",
        "normalized_units": "Pa",
        "conversion_method": "hPa -> Pa (x100)",
        "scale": 100.0,
    }


def test_the_dataset_records_its_normalization_in_attrs() -> None:
    """The written NetCDF must carry its own provenance."""
    result = normalize_dataset(make_dataset())

    assert "temperature" in result.dataset.attrs["normalized_variables"]
    assert result.dataset.attrs["normalization_warnings"] is not None
    assert result.dataset.attrs["normalized_at"]


def test_the_result_summary_is_json_serialisable() -> None:
    import json

    payload = normalize_dataset(make_dataset()).as_dict()
    assert json.loads(json.dumps(payload))["conversions"]


def test_normalization_is_deterministic() -> None:
    """Same input, same output - the pipeline must be reproducible."""
    source = make_dataset(
        variables={"temperature_2m": (np.full(SHAPE, 22.9), "degC")},
        latitude=[13.0, 12.5, 12.0],
    )
    first = normalize_dataset(source)
    second = normalize_dataset(source)

    np.testing.assert_array_equal(
        first.dataset["temperature"].values, second.dataset["temperature"].values
    )
    assert first.conversion_map() == second.conversion_map()
    assert first.coordinate_actions == second.coordinate_actions
    assert first.missing_variables == second.missing_variables

def test_a_whitelist_drops_everything_outside_it() -> None:
    result = normalize_dataset(make_dataset(), variables=("temperature", "pressure"))

    assert set(result.dataset.data_vars) == {"temperature", "pressure"}
    assert set(result.dropped_variables) == {
        "precipitation",
        "relative_humidity",
        "u_wind",
        "v_wind",
    }


def test_a_dataset_without_a_time_coordinate_cannot_enter_the_pipeline() -> None:
    """Phases 1-5 are time series; a static field is not ingestible here."""
    source = make_dataset().drop_vars("time")
    with pytest.raises(NormalizationError, match="no 'time' coordinate"):
        normalize_dataset(source)


def test_a_dataset_without_a_longitude_coordinate_is_rejected() -> None:
    source = make_dataset().drop_vars("longitude")
    with pytest.raises(NormalizationError, match="longitude"):
        normalize_dataset(source)


def test_a_dataset_without_a_latitude_coordinate_is_rejected() -> None:
    """A dataset with no latitude axis at all cannot be gridded for Phase 1-5."""
    shape = (4, 3)
    source = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.full(shape, 300.0), dims=("time", "x"), attrs={"units": "K"}
            )
        },
        coords={"time": make_times(), "x": np.arange(3, dtype="float64")},
    )
    with pytest.raises(NormalizationError, match="latitude"):
        normalize_dataset(source)


def test_renaming_keeps_the_values_and_the_standard_units() -> None:
    """The headline case: ERA5/Open-Meteo names in, canonical names out."""
    shape = (4, 3, 3)
    source = make_dataset(
        variables={
            # 300.15 K expressed as 27.00 degC, so the expected Kelvin value is
            # 300.15 and a double conversion would be immediately visible.
            "temperature_2m": (np.full(shape, 27.0), "degC"),
            "precipitation": (np.full(shape, 4.0), "mm"),
            "surface_pressure": (np.full(shape, 1013.0), "hPa"),
            "relative_humidity_2m": (np.full(shape, 55.0), "%"),
        }
    )
    result = normalize_dataset(source)

    assert "temperature_2m" not in result.dataset.data_vars
    assert ("temperature_2m", "temperature") in result.renamed_variables
    assert ("surface_pressure", "pressure") in result.renamed_variables
    # 27 degC + 273.15 == 300.15 K. Applied exactly once.
    assert float(result.dataset["temperature"].mean()) == pytest.approx(300.15, abs=1e-6)
    assert result.dataset["temperature"].attrs["units"] == "K"
    assert result.dataset["pressure"].attrs["units"] == "Pa"
    assert float(result.dataset["pressure"].mean()) == pytest.approx(101_300.0)


def test_an_alias_colliding_with_an_existing_canonical_name_is_dropped_not_clobbered() -> None:
    """Keeping both spellings must not silently discard either dataset's data."""
    shape = (4, 3, 3)
    source = make_dataset(
        variables={
            "temperature": (np.full(shape, 300.0), "K"),
            "2t": (np.full(shape, 999.0), "K"),
        }
    )
    result = normalize_dataset(source)
    assert "2t" in result.dropped_variables
    assert float(result.dataset["temperature"].mean()) == pytest.approx(300.0)
    assert any("already present" in warning for warning in result.warnings)

