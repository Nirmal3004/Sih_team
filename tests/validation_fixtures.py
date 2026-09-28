"""Shared fixtures for the Phase 7 validation tests (SIH26078).

Every fixture here is **synthetic and tiny**: the tests must not need the
network, and a validation suite that downloaded ERA5 on every run would be
unusable in CI. The real-data path is covered separately by one test that runs
the pipeline over a committed ERA5 JSON fixture, using the project's own
production code.

The one property these fixtures are built to exercise is **temporal leakage**:
``max(train) < min(validation) < min(test)``. Almost every test below exists to
assert that.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)

#: Enough steps that a 60/20/20 split gives every block real content.
FULL_STEPS = 100
SMALL_LAT = 4
SMALL_LON = 4


def make_times(count: int = FULL_STEPS, start: datetime = T0) -> np.ndarray:
    """Return ``count`` hourly ``datetime64`` stamps starting at ``start``."""
    return np.array(
        [np.datetime64(start + timedelta(hours=index), "m") for index in range(count)],
        dtype="datetime64[ns]",
    )


def make_validation_dataset(
    *,
    steps: int = FULL_STEPS,
    start: datetime = T0,
    seed: int = 11,
) -> xr.Dataset:
    """Build a small but complete gridded dataset for the validation tests.

    The precipitation field carries a genuine diurnal and a slow trend, so a
    percentile calibration has something real to fit, and the variables are in
    the project's standard units (Phase 6 already normalised them).
    """
    times = make_times(steps, start)
    lat = np.linspace(12.0, 13.0, SMALL_LAT)
    lon = np.linspace(76.0, 77.0, SMALL_LON)
    rng = np.random.default_rng(seed)

    hour = times.astype("datetime64[h]").astype(int) % 24
    trend = np.arange(steps, dtype="float64") * 0.05
    diurnal = np.sin(2.0 * np.pi * hour / 24.0) * 4.0
    precipitation = np.maximum(
        2.0 + diurnal[:, None, None] + trend[:, None, None] + rng.normal(0, 0.4, (steps, SMALL_LAT, SMALL_LON)),
        0.0,
    )

    return xr.Dataset(
        data_vars={
            "precipitation": (("time", "latitude", "longitude"), precipitation, {"units": "mm h-1"}),
            "temperature": (
                ("time", "latitude", "longitude"),
                300.0 + np.sin(2.0 * np.pi * hour / 24.0)[:, None, None]
                + rng.normal(0, 0.3, (steps, SMALL_LAT, SMALL_LON)),
                {"units": "K"},
            ),
            "pressure": (
                ("time", "latitude", "longitude"),
                101_000.0 + rng.normal(0, 40.0, (steps, SMALL_LAT, SMALL_LON)),
                {"units": "Pa"},
            ),
            "relative_humidity": (
                ("time", "latitude", "longitude"),
                np.clip(70.0 + rng.normal(0, 5.0, (steps, SMALL_LAT, SMALL_LON)), 0, 100),
                {"units": "%"},
            ),
            "u_wind": (
                ("time", "latitude", "longitude"),
                2.0 + rng.normal(0, 0.5, (steps, SMALL_LAT, SMALL_LON)),
                {"units": "m s-1"},
            ),
            "v_wind": (
                ("time", "latitude", "longitude"),
                1.0 + rng.normal(0, 0.5, (steps, SMALL_LAT, SMALL_LON)),
                {"units": "m s-1"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("latitude", lat),
            "longitude": ("longitude", lon),
        },
        attrs={"source": "unit_test", "is_synthetic": "true"},
    )


@pytest.fixture()
def validation_dataset() -> xr.Dataset:
    """A complete, leakable-looking dataset for the split tests."""
    return make_validation_dataset()


@pytest.fixture()
def short_dataset() -> xr.Dataset:
    """A dataset deliberately too short for a meaningful climatology."""
    return make_validation_dataset(steps=24)


@pytest.fixture()
def long_dataset() -> xr.Dataset:
    """A dataset spanning every month, so monthly coverage is complete."""
    times = np.array(
        [
            np.datetime64(datetime(2023, month, 1, 12, tzinfo=UTC), "m")
            for month in range(1, 13)
        ],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        data_vars={
            "precipitation": (
                ("time", "latitude", "longitude"),
                np.linspace(1.0, 12.0, 12)[:, None, None]
                * np.ones((1, SMALL_LAT, SMALL_LON)),
                {"units": "mm h-1"},
            )
        },
        coords={
            "time": times,
            "latitude": np.linspace(12.0, 13.0, SMALL_LAT),
            "longitude": np.linspace(76.0, 77.0, SMALL_LON),
        },
    )


@pytest.fixture()
def validation_settings(tmp_path: Path):
    """Settings whose data directory is a temporary folder."""
    from extreme_weather.config import Settings

    resolved = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    resolved.ensure_directories()
    return resolved


@pytest.fixture()
def reports_dir(tmp_path: Path) -> Path:
    """An empty reports directory, standing in for a fresh installation."""
    target = tmp_path / "data" / "validation" / "reports"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture()
def validation_client(validation_settings) -> Iterator[TestClient]:
    """A FastAPI test client backed by a temporary data directory."""
    from extreme_weather.api.main import create_app
    from extreme_weather.config import get_settings

    application = create_app(validation_settings)
    application.dependency_overrides[get_settings] = lambda: validation_settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()
