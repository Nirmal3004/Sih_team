"""Test doubles for the Phase 8 near-real-time tests (SIH26078).

These exist so the pipeline can be exercised **without a network and without an
API key**. That constraint is deliberate: a suite that reaches a live provider
is a suite that fails intermittently and cannot prove anything about failure
handling.

The doubles control the three things a test needs to vary - the data returned,
the error raised, and the number of attempts - so every failure path is
deterministic and asserted rather than waited for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import xarray as xr

from extreme_weather.realtime.exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from extreme_weather.realtime.providers import (
    ProviderFetch,
    WeatherProvider,
    WeatherRequest,
)

__all__ = [
    "FlakyProvider",
    "REFERENCE_NOW",
    "StubProvider",
    "make_dataset",
    "make_dataset_without_time",
    "make_malformed_dataset",
    "make_nan_heavy_dataset",
    "make_non_monotonic_dataset",
    "make_duplicate_time_dataset",
]

UTC = timezone.utc

#: A fixed reference instant, so freshness assertions never depend on the clock.
REFERENCE_NOW = datetime(2024, 6, 30, 12, 0, tzinfo=UTC)

#: The variables a conforming dataset carries.
DEFAULT_VARIABLES: tuple[str, ...] = (
    "precipitation",
    "temperature",
    "u_wind",
    "v_wind",
)


def _stamps(start: datetime, hours: int) -> np.ndarray:
    """Build an hourly datetime64 array from a naive-UTC start."""
    base = start.astimezone(UTC).replace(tzinfo=None)
    return np.array(
        [base + np.timedelta64(h * 60, "m") for h in range(hours)],
        dtype="datetime64[ns]",
    )


def make_dataset(
    start: datetime = REFERENCE_NOW - timedelta(hours=5),
    hours: int = 6,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    latitude: tuple[float, ...] = (12.0, 12.5, 13.0),
    longitude: tuple[float, ...] = (76.0, 76.5, 77.0),
    seed: int = 7,
) -> xr.Dataset:
    """Build a small, valid weather dataset.

    Args:
        start: First timestamp.
        hours: Number of hourly steps.
        variables: Variable names to include.
        latitude: Latitude points.
        longitude: Longitude points.
        seed: Random seed, so the dataset is reproducible.

    Returns:
        A conforming :class:`xarray.Dataset`.
    """
    stamps = _stamps(start, hours)
    rng = np.random.default_rng(seed)
    shape = (hours, len(latitude), len(longitude))

    data: dict[str, Any] = {}
    for name in variables:
        if "temp" in name:
            values = 300.0 + rng.normal(0.0, 0.5, shape)
            units = "K"
        elif "prec" in name or name == "tp":
            values = np.abs(rng.normal(2.0, 0.5, shape))
            units = "mm"
        elif "wind" in name:
            values = rng.normal(1.0, 0.3, shape)
            units = "m s-1"
        else:
            values = np.abs(rng.normal(1.0, 0.1, shape))
            units = "1"
        data[name] = xr.DataArray(
            values.astype("float64"),
            dims=("time", "latitude", "longitude"),
            attrs={"units": units},
        )

    return xr.Dataset(
        data_vars=data,
        coords={
            "time": stamps,
            "latitude": np.asarray(latitude, dtype="float64"),
            "longitude": np.asarray(longitude, dtype="float64"),
        },
    )


def make_dataset_without_time(hours: int = 6) -> xr.Dataset:
    """Return a dataset with no time coordinate at all."""
    dataset = make_dataset(hours=hours)
    return dataset.drop_vars("time")


def make_malformed_dataset() -> xr.Dataset:
    """Return a dataset with no latitude/longitude dimensions."""
    stamps = _stamps(REFERENCE_NOW - timedelta(hours=5), 6)
    values = np.random.default_rng(3).normal(1.0, 0.1, (6, 2, 2))
    return xr.Dataset(
        data_vars={
            "precipitation": xr.DataArray(
                values, dims=("time", "y", "x"), attrs={"units": "mm"}
            )
        },
        coords={"time": stamps, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )


def make_nan_heavy_dataset(fraction: float = 0.95) -> xr.Dataset:
    """Return a dataset that is mostly missing values."""
    dataset = make_dataset()
    rng = np.random.default_rng(11)
    for name in dataset.data_vars:
        values = np.asarray(dataset[name].values, dtype="float64").copy()
        mask = rng.random(values.shape) < fraction
        values[mask] = np.nan
        dataset[name] = xr.DataArray(
            values, dims=dataset[name].dims, attrs=dataset[name].attrs
        )
    return dataset


def make_non_monotonic_dataset() -> xr.Dataset:
    """Return a dataset whose time axis runs backwards part-way through."""
    dataset = make_dataset()
    times = np.asarray(dataset.coords["time"].values).copy()
    times[[1, 2]] = times[[2, 1]]
    dataset = dataset.assign_coords(time=times)
    return dataset


def make_duplicate_time_dataset() -> xr.Dataset:
    """Return a dataset with a repeated timestamp."""
    dataset = make_dataset()
    times = np.asarray(dataset.coords["time"].values).copy()
    times[3] = times[2]
    return dataset.assign_coords(time=times)


class StubProvider(WeatherProvider):
    """A provider whose every behaviour a test dictates.

    Attributes:
        calls: How many times :meth:`fetch` was called.
        requests: Every request received, in order.
    """

    name = "stub"
    description = "Test double."

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        *,
        error: BaseException | None = None,
        errors: list[BaseException] | None = None,
        available: bool = True,
    ) -> None:
        self._dataset = dataset
        self._error = error
        self._errors = list(errors) if errors else []
        self._available = bool(available)
        self.calls = 0
        self.requests: list[WeatherRequest] = []

    def available(self) -> bool:
        """Return the configured availability, without touching anything."""
        return self._available

    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Return the configured dataset, or raise the configured error.

        Raises:
            BaseException: Whatever the test configured, when one is pending.
        """
        self.calls += 1
        self.requests.append(request)
        if self._errors:
            raise self._errors.pop(0)
        if self._error is not None:
            raise self._error
        dataset = (
            self._dataset
            if self._dataset is not None
            else make_dataset(start=request.start, hours=request.hours)
        )
        return ProviderFetch(
            provider=self.name,
            dataset=dataset,
            requested_at=datetime.now(UTC),
            notes=("stub",),
        )


class FlakyProvider(StubProvider):
    """Fails a configured number of times, then succeeds.

    Attributes:
        calls: Total fetch attempts made.
    """

    name = "flaky"
    description = "Fails a configured number of times, then succeeds."

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        *,
        failures: int = 2,
        error: BaseException | None = None,
    ) -> None:
        # Do NOT pass the error to super().__init__: StubProvider.fetch re-raises
        # a stored _error on every call, which would make the post-recovery
        # success impossible to express. FlakyProvider owns its own failure.
        super().__init__(dataset)
        self._remaining = int(failures)
        self._flaky_error = error or ProviderTimeoutError("simulated provider timeout")

    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Fail while attempts remain, then return data.

        Raises:
            ProviderError: While ``failures`` attempts remain.
        """
        if self._remaining > 0:
            self._remaining -= 1
            self.calls += 1
            self.requests.append(request)
            raise self._flaky_error
        return super().fetch(request)
