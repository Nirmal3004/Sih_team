"""Tests for the baseline downscaling implementation and its registry."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.config import DOWNSAMPLE_TARGET_RESOLUTION_DEG, NWP_NATIVE_RESOLUTION_DEG
from extreme_weather.downscaling import (
    DownscalingError,
    UnknownDownscalingMethodError,
    available_methods,
    build_target_coordinate,
    cubic_downscale,
    downscale,
    get_method,
    nearest_downscale,
    register_method,
    registered_methods,
)

LATITUDE = np.arange(10.0, 14.0 + 1e-9, 0.125)
LONGITUDE = np.arange(74.0, 78.0 + 1e-9, 0.125)


def analytic_field(latitude: np.ndarray, longitude: np.ndarray, time_index: float = 0.0) -> np.ndarray:
    """Smooth, exactly representable reference field used to judge interpolation."""
    lat_2d, lon_2d = np.meshgrid(latitude, longitude, indexing="ij")
    return (
        np.sin(np.deg2rad((lat_2d - 10.0) * 30.0))
        * np.cos(np.deg2rad((lon_2d - 74.0) * 25.0))
        + 0.1 * time_index
    )


@pytest.fixture(scope="module")
def coarse_field() -> xr.DataArray:
    """2-D coarse (12 km style) field on a 0.125 deg grid."""
    values = analytic_field(LATITUDE, LONGITUDE)
    return xr.DataArray(
        values.astype("float32"),
        dims=("latitude", "longitude"),
        coords={"latitude": LATITUDE, "longitude": LONGITUDE},
        name="precipitation",
        attrs={"units": "mm h-1", "long_name": "Synthetic precipitation rate"},
    )


@pytest.fixture(scope="module")
def coarse_time_series() -> xr.DataArray:
    """3-D coarse field with an additional non-spatial dimension."""
    values = np.stack(
        [analytic_field(LATITUDE, LONGITUDE, time_index=index) for index in range(3)]
    )
    return xr.DataArray(
        values.astype("float32"),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2024-06-01T00", "2024-06-01T01", "2024-06-01T02"], dtype="datetime64[h]"),
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
        name="temperature",
        attrs={"units": "K"},
    )


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def test_build_target_coordinate_spans_the_source_domain() -> None:
    target = build_target_coordinate(LATITUDE, 0.05)

    assert target[0] == pytest.approx(LATITUDE.min())
    assert target[-1] == pytest.approx(LATITUDE.max())
    assert np.allclose(np.diff(target), 0.05, atol=1e-9)
    assert target.size == 81  # (14 - 10) / 0.05 + 1


def test_nearest_downscaling_reaches_the_target_resolution(coarse_field) -> None:
    fine = downscale(coarse_field, 0.05)

    assert fine.sizes["latitude"] == 81
    assert fine.sizes["longitude"] == 81
    assert np.allclose(np.diff(fine["latitude"].values), 0.05, atol=1e-9)
    assert np.allclose(np.diff(fine["longitude"].values), 0.05, atol=1e-9)
    assert fine["latitude"].values[0] == pytest.approx(10.0)
    assert fine["latitude"].values[-1] == pytest.approx(14.0)
    assert fine["longitude"].values[0] == pytest.approx(74.0)
    assert fine["longitude"].values[-1] == pytest.approx(78.0)
    # 0.125 -> 0.05 turns a 33x33 grid into an 81x81 grid.
    assert fine.size / coarse_field.size == pytest.approx(
        (81 * 81) / (coarse_field.sizes["latitude"] * coarse_field.sizes["longitude"])
    )


def test_downscaling_matches_the_12km_to_5km_reference_task() -> None:
    latitude = np.arange(10.0, 14.0 + 1e-9, NWP_NATIVE_RESOLUTION_DEG)
    longitude = np.arange(74.0, 78.0 + 1e-9, NWP_NATIVE_RESOLUTION_DEG)
    values = analytic_field(latitude, longitude)
    coarse = xr.DataArray(
        values.astype("float32"),
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
        name="precipitation",
    )

    fine = downscale(coarse, DOWNSAMPLE_TARGET_RESOLUTION_DEG, method="nearest")

    assert coarse.sizes["latitude"] == 33  # ~12 km grid
    assert fine.sizes["latitude"] == 81  # ~5 km grid
    assert np.allclose(np.diff(fine["latitude"].values), 0.05, atol=1e-9)
    assert float(np.mean(np.abs(np.diff(fine["latitude"].values)))) * 111.32 == pytest.approx(
        5.57, abs=0.02
    )


def test_downscaling_preserves_name_attributes_and_extra_dimensions(coarse_time_series) -> None:
    fine = downscale(coarse_time_series, 0.1, method="nearest")

    assert fine.name == "temperature"
    assert fine.attrs["units"] == "K"
    assert fine.sizes["time"] == 3
    assert np.array_equal(fine["time"].values, coarse_time_series["time"].values)
    assert fine.sizes["latitude"] == 41
    assert fine.sizes["longitude"] == 41


# --------------------------------------------------------------------------- #
# Nearest interpolation
# --------------------------------------------------------------------------- #
def test_nearest_is_exact_at_the_source_points(coarse_field) -> None:
    # 0.025 deg divides 0.125 deg exactly, so the target grid contains every
    # source point - ideal for verifying "nearest" without tolerances.
    fine = downscale(coarse_field, 0.025, method="nearest")

    subset = fine.sel(latitude=LATITUDE, longitude=LONGITUDE, method="nearest")
    assert np.array_equal(subset.values, coarse_field.values)

    # ... and everywhere else it is exactly one of the coarse samples.
    source_values = np.unique(coarse_field.values)
    assert np.isin(fine.values, source_values).all()


def test_nearest_and_cubic_are_registered_as_separate_behaviours(coarse_field) -> None:
    nearest = nearest_downscale(coarse_field, 0.1)
    cubic = cubic_downscale(coarse_field, 0.1)

    assert nearest.shape == cubic.shape
    assert not np.allclose(nearest.values, cubic.values, atol=1e-6)


def test_cubic_interpolation_is_more_accurate_than_nearest(coarse_field) -> None:
    target_latitude = np.arange(10.0, 14.0 + 1e-9, 0.05)
    target_longitude = np.arange(74.0, 78.0 + 1e-9, 0.05)
    truth = analytic_field(target_latitude, target_longitude)

    nearest = downscale(coarse_field, 0.05, method="nearest")
    cubic = downscale(coarse_field, 0.05, method="cubic")

    nearest_error = float(np.sqrt(np.mean((nearest.values - truth) ** 2)))
    cubic_error = float(np.sqrt(np.mean((cubic.values - truth) ** 2)))

    assert cubic_error < nearest_error
    assert cubic_error < 0.01


def test_cubic_requires_enough_source_points() -> None:
    small = xr.DataArray(
        np.arange(4.0, dtype="float32").reshape(2, 2),
        dims=("latitude", "longitude"),
        coords={"latitude": [10.0, 10.125], "longitude": [74.0, 74.125]},
        name="precipitation",
    )

    with pytest.raises(DownscalingError, match="at least 4 points"):
        cubic_downscale(small, 0.05)


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
def test_registry_exposes_the_builtin_methods() -> None:
    methods = available_methods()
    assert "nearest" in methods
    assert "cubic" in methods
    assert methods == tuple(sorted(methods))

    registry = registered_methods()
    assert set(registry) == set(methods)
    assert registry["nearest"].function is nearest_downscale
    assert registry["cubic"].function is cubic_downscale
    assert registry["cubic"].minimum_points_per_dim == 4
    assert registry["nearest"].description

    assert get_method("NEAREST ").name == "nearest"  # names are normalised


def test_registry_accepts_custom_methods(coarse_field) -> None:
    calls: list[str] = []

    @register_method("identity_test", description="Test-only nearest alias.")
    def identity_test(data: xr.DataArray, target_resolution: float, **kwargs) -> xr.DataArray:
        calls.append("called")
        return nearest_downscale(data, target_resolution, **kwargs)

    try:
        assert "identity_test" in available_methods()
        result = downscale(coarse_field, 0.1, method="identity_test")
        assert calls == ["called"]
        assert result.sizes["latitude"] == 41
        assert get_method("identity_test").description == "Test-only nearest alias."
    finally:
        # Keep the registry clean for the other tests in this session.
        from extreme_weather.downscaling import registry as registry_module

        registry_module._REGISTRY.pop("identity_test", None)

    assert "identity_test" not in available_methods()


def test_registry_rejects_duplicate_names_without_overwrite() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_method("nearest")
        def duplicate(data, target_resolution, **kwargs):  # pragma: no cover - never registered
            return data


def test_unknown_method_error_lists_the_available_methods(coarse_field) -> None:
    with pytest.raises(UnknownDownscalingMethodError) as error:
        downscale(coarse_field, 0.05, method="diffusion_sr")

    message = str(error.value)
    assert "diffusion_sr" in message
    assert "nearest" in message and "cubic" in message
    assert "register_method" in message
    # The error doubles as a ValueError so callers can catch it either way.
    assert isinstance(error.value, ValueError)


def test_unknown_method_from_get_method(coarse_field) -> None:
    with pytest.raises(UnknownDownscalingMethodError, match="not-a-method"):
        get_method("not-a-method")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("resolution", [0.0, -0.05, None, "0.05", True])
def test_invalid_target_resolution_is_rejected(coarse_field, resolution) -> None:
    with pytest.raises((DownscalingError, TypeError)):
        downscale(coarse_field, resolution)


def test_target_resolution_coarser_than_the_domain_is_rejected(coarse_field) -> None:
    with pytest.raises(DownscalingError, match="coarser than"):
        downscale(coarse_field, 20.0)


def test_dataset_input_is_rejected_with_a_clear_message(coarse_field) -> None:
    dataset = coarse_field.to_dataset(name="precipitation")

    with pytest.raises(TypeError, match="DataArray"):
        downscale(dataset, 0.05)


def test_missing_spatial_dimension_is_rejected() -> None:
    field = xr.DataArray(np.ones((3, 3)), dims=("y", "x"), name="field")

    with pytest.raises(DownscalingError, match="latitude"):
        downscale(field, 0.05)


def test_custom_dimension_names_are_supported() -> None:
    field = xr.DataArray(
        analytic_field(LATITUDE, LONGITUDE).astype("float32"),
        dims=("lat", "lon"),
        coords={"lat": LATITUDE, "lon": LONGITUDE},
        name="precipitation",
    )

    fine = downscale(field, 0.05, method="nearest", latitude_dim="lat", longitude_dim="lon")

    assert fine.sizes["lat"] == 81
    assert fine.sizes["lon"] == 81
    assert np.allclose(np.diff(fine["lat"].values), 0.05, atol=1e-9)
