"""Shared fixtures for the Phase 6 ingestion tests (SIH26078).

Two kinds of fixture live here, and the distinction matters:

* **Synthetic fixtures** - tiny NetCDF files written by the tests through
  ``xarray``, used for every unit test. No network, no external data, no
  large binaries in the repository.
* **A stored ERA5 JSON payload** - a *real* API response captured once, reduced
  to 4 time steps, and committed as a fixture. The conversion from that payload
  to NetCDF is a pure function, so testing it against this fixture exercises
  the real code path without the tests ever going online.

The ERA5 payload is genuine reanalysis data (ECMWF ERA5 via the Open-Meteo
archive). It is included so the unit conversions - degrees Celsius to Kelvin,
hPa to Pa, mm to mm/h - are tested against the units a real archive actually
uses, not against values invented to make a test pass.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from extreme_weather.config import Settings, get_settings
from extreme_weather.ingestion import (
    IngestionService,
    ManifestStore,
    NetCDFSource,
    check_quality,
    normalize_dataset,
)
from extreme_weather.ingestion.registry import DatasetRegistry

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)

#: A 4-hour, 3x3 grid - the smallest thing that still exercises every axis.
SMALL_TIMESTEPS = 4
SMALL_LAT = 3
SMALL_LON = 3


def make_times(count: int = SMALL_TIMESTEPS, start: datetime = T0) -> np.ndarray:
    """Return ``count`` hourly stamps starting at ``start``."""
    return np.array(
        [np.datetime64(start + timedelta(hours=index), "m") for index in range(count)],
        dtype="datetime64[ns]",
    )


def make_dataset(
    *,
    variables: dict[str, tuple[np.ndarray, str]] | None = None,
    times: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Build a small gridded dataset on ``(time, latitude, longitude)``.

    Args:
        variables: ``{name: (values, units)}``. Defaults to a complete six-field
            contract dataset with plausible values.
        times: Time coordinate. Defaults to :func:`make_times`.
        latitude: Latitude coordinate. Defaults to 12.0, 12.5, 13.0.
        longitude: Longitude coordinate. Defaults to 76.5, 77.0, 77.5.
        attrs: Dataset-level attributes.
    """
    stamps = make_times() if times is None else times
    lats = np.array([12.0, 12.5, 13.0]) if latitude is None else latitude
    lons = np.array([76.5, 77.0, 77.5]) if longitude is None else longitude
    shape = (len(stamps), len(lats), len(lons))

    if variables is None:
        variables = {
            "temperature": (np.full(shape, 300.0), "K"),
            "precipitation": (np.full(shape, 5.0), "mm h-1"),
            "pressure": (np.full(shape, 101_000.0), "Pa"),
            "relative_humidity": (np.full(shape, 60.0), "%"),
            "u_wind": (np.full(shape, 2.0), "m s-1"),
            "v_wind": (np.full(shape, 1.0), "m s-1"),
        }

    data_vars = {
        name: (
            # Dimension *names* go here; the coordinate values are attached
            # through ``coords=`` below. Passing the coordinate arrays here
            # makes Xarray raise "unhashable type: numpy.ndarray".
            ("time", "latitude", "longitude"),
            np.broadcast_to(np.asarray(values, dtype="float64"), shape).copy(),
            {"units": units, "long_name": name.replace("_", " ")},
        )
        for name, (values, units) in variables.items()
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": stamps,
            "latitude": ("latitude", lats),
            "longitude": ("longitude", lons),
        },
        attrs=dict(attrs or {"source": "unit_test", "is_synthetic": "true"}),
    )


def write_netcdf(dataset: xr.Dataset, path: Path) -> Path:
    """Write ``dataset`` to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, mode="w")
    return path


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings whose data directory is a temporary folder."""
    resolved = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    resolved.ensure_directories()
    resolved.real_raw_dir.mkdir(parents=True, exist_ok=True)
    resolved.synthetic_raw_dir.mkdir(parents=True, exist_ok=True)
    return resolved


@pytest.fixture()
def contract_dataset() -> xr.Dataset:
    """A complete six-variable dataset in the standard units."""
    return make_dataset()


@pytest.fixture()
def contract_file(contract_dataset: xr.Dataset, settings: Settings) -> Path:
    """A complete six-variable dataset written to the real raw directory.

    Named ``contract.nc`` so a test can ingest it under a *different* dataset id;
    ingesting it under ``contract`` would overwrite this very file, which is a
    useful thing to avoid doing by accident.
    """
    return write_netcdf(contract_dataset, settings.real_raw_dir / "contract.nc")


@pytest.fixture()
def source_file(contract_dataset: xr.Dataset, settings: Settings) -> Path:
    """A source file outside every directory ingestion writes to.

    Use this when a test ingests twice: the output must not be the input, or the
    second run would read its own output.
    """
    return write_netcdf(contract_dataset, settings.data_dir / "incoming" / "source.nc")


@pytest.fixture()
def service(settings: Settings) -> IngestionService:
    """An ingestion service wired to the temporary directories."""
    return IngestionService(
        registry=DatasetRegistry(ManifestStore(settings.manifests_dir)),
        settings=settings,
    )


#: Path of the committed, genuine ERA5 archive payload fixture.
ERA5_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "era5_archive_sample.json"


@pytest.fixture()
def era5_payload() -> list[dict[str, Any]]:
    """A real ERA5 archive response, reduced to 2 points x 4 time steps.

    Genuine ECMWF ERA5 reanalysis (served by the Open-Meteo archive). The tests
    never open a network connection: the payload is a fixture, and the
    conversion from it is a pure function.
    """
    return json.loads(ERA5_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture()
def era5_dataset(era5_payload: list[dict[str, Any]]) -> xr.Dataset:
    """The ERA5 payload converted to an :class:`xarray.Dataset` in source units."""
    from extreme_weather.ingestion import era5_payload_to_dataset

    return era5_payload_to_dataset(era5_payload)


@pytest.fixture()
def service_without_grib(settings: Settings) -> IngestionService:
    """An ingestion service whose registry is empty and isolated."""
    return IngestionService(
        registry=DatasetRegistry(ManifestStore(settings.manifests_dir)),
        settings=settings,
    )


@pytest.fixture()
def data_client(settings: Settings) -> Iterator[TestClient]:
    """A FastAPI test client whose data directory is a temporary folder."""
    from extreme_weather.api.main import create_app

    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()

