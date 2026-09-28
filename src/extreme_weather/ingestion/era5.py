"""ERA5 historical reanalysis via the Open-Meteo archive API (SIH26078, Phase 6).

Why this provider
-----------------
ERA5 is the ECMWF reanalysis of the real atmosphere, and it is available
through the Open-Meteo **archive** endpoint without an API key. That makes it
the one real data source a student project can actually reproduce, which is why
it is here rather than a NOAA- or CDS-only adapter that would be untestable.

What it is, precisely
---------------------
* **Real observations, assimilated into a model.** ERA5 is a reanalysis: built
  from real measurements plus a model, not a raw station archive.
* **HISTORICAL.** It is a *past* period. Fetching it now says nothing about the
  present or the future, and nothing here calls it "live".
* **Not a forecast.** There is no initialisation, no ensemble, no skill claim.

Credentials
-----------
None are required. An optional ``OPEN_METEO_API_KEY`` only raises the rate
limit; it is never required and never logged.

Network safety
--------------
* A response advertising more than ``max_download_mb`` is refused, so a
  mistyped global request cannot fill the disk.
* Every call is bounded by a timeout and a failure raises
  :class:`DataSourceUnavailableError` - the provider never fabricates data when
  the network is down.
* Normal unit tests do **not** touch the network: the JSON-to-NetCDF conversion
  is a pure function tested against a stored fixture.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from extreme_weather.ingestion.base import DataOrigin, SourceInfo, utc_now
from extreme_weather.ingestion.exceptions import (
    DataSourceUnavailableError,
    IngestionError,
    SourceNotFoundError,
)

__all__ = [
    "ERA5_API_URL",
    "ERA5_ATTRIBUTION",
    "ERA5_LICENSE",
    "ERA5Source",
    "OpenMeteoRequest",
    "era5_payload_to_dataset",
]


#: The archive endpoint. This is the documented public API address.
ERA5_API_URL: str = "https://archive-api.open-meteo.com/v1/archive"

#: Attribution recorded in every manifest that used this provider.
ERA5_ATTRIBUTION: str = (
    "ERA5 reanalysis served by the Open-Meteo archive API (ECMWF ERA5, "
    "Copernicus Climate Change Service)."
)

#: ECMWF's data licence for ERA5. Recorded in the manifest, not enforced.
ERA5_LICENSE: str = "ECMWF ERA5 licence (Copernicus) - free for non-commercial use"

#: Environment variable that may raise the rate limit. Never required.
API_KEY_ENV: str = "OPEN_METEO_API_KEY"

#: Default request timeout in seconds.
DEFAULT_TIMEOUT_SECONDS: int = 30

#: Open-Meteo hourly variable name -> project canonical variable.
VARIABLE_MAP: dict[str, str] = {
    "precipitation": "precipitation",
    "temperature_2m": "temperature",
    "surface_pressure": "pressure",
    "relative_humidity_2m": "relative_humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}

#: Source unit for each Open-Meteo variable, as the API documents it.
SOURCE_UNITS: dict[str, str] = {
    "precipitation": "mm",
    "temperature_2m": "degC",
    "surface_pressure": "hPa",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "m s-1",
    "wind_direction_10m": "degree",
}


@dataclass(frozen=True)
class OpenMeteoRequest:
    """A bounded archive request.

    The bounds are explicit because an unbounded request is how a 2 GB download
    starts by accident: a regional demo should name a small box.

    Attributes:
        latitudes: Latitude points.
        longitudes: Longitude points.
        start_date: First day, ``YYYY-MM-DD``.
        end_date: Last day, ``YYYY-MM-DD``.
        variables: Open-Meteo hourly variable names.
        timeout_seconds: Network timeout.
    """

    latitudes: tuple[float, ...]
    longitudes: tuple[float, ...]
    start_date: str
    end_date: str
    variables: tuple[str, ...] = ("precipitation", "temperature_2m")
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.latitudes or not self.longitudes:
            raise ValueError("An archive request needs at least one point per axis.")
        for value in self.latitudes:
            if not -90.0 <= value <= 90.0:
                raise ValueError(f"Latitude {value} is outside [-90, 90].")
        for value in self.longitudes:
            if not -180.0 <= value <= 180.0:
                raise ValueError(f"Longitude {value} is outside [-180, 180].")
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date '{self.start_date}' is after end_date '{self.end_date}'."
            )
        if not self.variables:
            raise ValueError("An archive request needs at least one variable.")

    def point_count(self) -> int:
        """Number of grid points in the request."""
        return len(self.latitudes) * len(self.longitudes)

    def build_url(self) -> str:
        """Return the full request URL.

        The optional API key is appended last, so callers must not log the
        result verbatim.
        """
        params = {
            "latitude": ",".join(f"{value:g}" for value in self.latitudes),
            "longitude": ",".join(f"{value:g}" for value in self.longitudes),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": ",".join(self.variables),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }
        key = os.environ.get(API_KEY_ENV)
        if key:
            params["apikey"] = key
        return f"{ERA5_API_URL}?{urllib.parse.urlencode(params)}"



def era5_payload_to_dataset(
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> xr.Dataset:
    """Convert an Open-Meteo archive payload into an :class:`xarray.Dataset`.

    This is a **pure function**: no network, no clock, no environment. The unit
    tests exercise it against a stored fixture, so the conversion is covered
    without ever going online.

    The API returns one object per grid point when several points are requested
    and a single object for one point; both shapes are accepted.

    Args:
        payload: The decoded JSON response.

    Returns:
        A dataset on ``(time, latitude, longitude)`` in the **source's own**
        units, with source variable names and ``is_synthetic=False``. Unit
        conversion is left to :mod:`extreme_weather.ingestion.normalizer`.

    Raises:
        IngestionError: If the payload is not a usable archive response.
    """
    if isinstance(payload, Mapping):
        records: list[Mapping[str, Any]] = [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        records = list(payload)
    else:
        records = []

    if not records or not isinstance(records[0], Mapping):
        raise IngestionError(
            "The archive payload is not a list of per-point records; got "
            f"{type(payload).__name__}."
        )

    # A per-point record must carry its coordinates and an hourly block. Without
    # them the payload cannot become a grid, and the resulting KeyError would
    # tell an operator nothing about what was wrong with the *response*.
    for index, record in enumerate(records):
        missing = [key for key in ("latitude", "longitude", "hourly") if key not in record]
        if missing:
            raise IngestionError(
                f"Archive record {index} is missing required key(s) {missing}. "
                "Each record must carry 'latitude', 'longitude' and an 'hourly' block."
            )
        if not isinstance(record["hourly"], Mapping) or "time" not in record["hourly"]:
            raise IngestionError(
                f"Archive record {index} has no 'hourly.time' series, so the "
                "payload cannot be placed on a time axis."
            )

    latitudes = [float(record["latitude"]) for record in records]
    longitudes = [float(record["longitude"]) for record in records]
    hours = list(records[0]["hourly"]["time"])
    names = [key for key in records[0]["hourly"] if key != "time"]

    if any(len(list(record["hourly"]["time"])) != len(hours) for record in records):
        raise IngestionError(
            "The archive returned a different number of time steps per grid point; "
            "the payload cannot be assembled into a regular grid."
        )

    shape = (len(hours), len(latitudes), len(longitudes))
    stamps = np.array(
        [np.datetime64(value, "m") for value in hours], dtype="datetime64[m]"
    )

    data_vars: dict[str, xr.DataArray] = {}
    for name in names:
        grid = np.empty(shape, dtype="float64")
        for lat_slot, record in enumerate(records):
            series = record["hourly"].get(name)
            if series is None:
                grid[:, lat_slot, :] = np.nan
                continue
            values = np.array(
                [np.nan if value is None else float(value) for value in series],
                dtype="float64",
            )
            grid[:, lat_slot, :] = values[:, None]
        data_vars[name] = xr.DataArray(
            grid,
            dims=("time", "latitude", "longitude"),
            attrs={
                "units": SOURCE_UNITS.get(name, ""),
                "long_name": f"ERA5/Open-Meteo {name}",
            },
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": xr.DataArray(stamps, dims=("time",), attrs={"standard_name": "time"}),
            "latitude": xr.DataArray(
                np.array(latitudes),
                dims=("latitude",),
                attrs={"units": "degrees_north", "standard_name": "latitude"},
            ),
            "longitude": xr.DataArray(
                np.array(longitudes),
                dims=("longitude",),
                attrs={"units": "degrees_east", "standard_name": "longitude"},
            ),
        },
        attrs={
            "source": "open-meteo-archive",
            "dataset_type": "historical_reanalysis",
            "is_synthetic": "false",
            "provider_attribution": ERA5_ATTRIBUTION,
            "license": ERA5_LICENSE,
            "retrieved_at": utc_now().isoformat(),
            "coordinates": "time latitude longitude",
        },
    )



class ERA5Source:
    """Fetch ERA5 reanalysis subsets as NetCDF via the Open-Meteo archive.

    Args:
        cache_dir: Where downloads are written. Defaults to
            ``<DATA_DIR>/cache``.
        max_download_mb: Refuse a response larger than this. Guards against a
            mistyped global request.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        max_download_mb: int | None = None,
    ) -> None:
        from extreme_weather.config import get_settings

        settings = get_settings()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else settings.cache_dir
        self._max_bytes = (
            int(max_download_mb) * 1024 * 1024
            if max_download_mb is not None
            else int(settings.max_download_mb) * 1024 * 1024
        )

    # -- interface ---------------------------------------------------- #
    def name(self) -> str:
        """Return the stable source identifier."""
        return "era5"

    def available(self) -> bool:
        """True when this source exists and could be used.

        Only a genuine module-level capability is reported. Whether the *network*
        currently works is a different question, answered by actually calling,
        which raises :class:`DataSourceUnavailableError`.
        """
        return True

    def default_origin(self) -> str:
        """ERA5 is a reanalysis of the past: HISTORICAL, never SYNTHETIC."""
        return DataOrigin.HISTORICAL

    def metadata(self) -> SourceInfo:
        """Describe the source without making a network call."""
        return SourceInfo(
            name=self.name(),
            kind="remote",
            description=(
                "ERA5 reanalysis subsets via the Open-Meteo archive API. Real past "
                "weather, no API key required."
            ),
            formats=("json", "netcdf"),
            requires_credentials=False,
            required_environment=(),
            default_origin=DataOrigin.HISTORICAL,
            notes=(
                f"{ERA5_ATTRIBUTION} {ERA5_LICENSE} Historical reanalysis - not a "
                f"forecast and not live. Optional {API_KEY_ENV} only raises the rate "
                f"limit."
            ),
        )

    @property
    def attribution(self) -> str:
        """Attribution string to record in a manifest."""
        return ERA5_ATTRIBUTION

    @property
    def license(self) -> str:
        """Licence string to record in a manifest."""
        return ERA5_LICENSE

    def build_url(self, request: OpenMeteoRequest) -> str:
        """Return the request URL (do not log the result verbatim)."""
        return request.build_url()

    def download_json(self, request: OpenMeteoRequest) -> list[Mapping[str, Any]]:
        """Fetch the archive JSON payload for ``request``.

        Raises:
            DataSourceUnavailableError: If the network call fails, times out, or
                returns a non-2xx status. No data is invented on failure.
            IngestionError: If the response exceeds the download limit.
        """
        url = self.build_url(request)
        try:
            with urllib.request.urlopen(url, timeout=request.timeout_seconds) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > self._max_bytes:
                    raise IngestionError(
                        f"The archive response is {int(declared) / 1e6:.1f} MB, over the "
                        f"{self._max_bytes // (1024 * 1024)} MB limit. Request a smaller "
                        "area or time range."
                    )
                raw = response.read(self._max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise DataSourceUnavailableError(
                f"The archive API returned HTTP {exc.code} for the requested box. Check "
                "the date range and the requested variables."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DataSourceUnavailableError(
                f"Could not reach the archive API: {type(exc).__name__}. This is a "
                "network problem, not a data problem."
            ) from exc

        if len(raw) > self._max_bytes:
            raise IngestionError("The archive response exceeded the download limit.")
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DataSourceUnavailableError(
                f"The archive API returned a non-JSON response: {exc}"
            ) from exc


    def fetch(
        self,
        request: OpenMeteoRequest | None = None,
        *,
        dataset_id: str = "era5",
        **kwargs: Any,
    ) -> Path:
        """Download a subset and write it to the cache as NetCDF.

        Raises:
            ValueError: If no request is supplied.
            IngestionError: If the NetCDF file cannot be written.
        """
        if request is None:
            raise ValueError("ERA5Source.fetch needs an OpenMeteoRequest.")
        dataset = era5_payload_to_dataset(self.download_json(request))
        target = self._cache_dir / f"{dataset_id}.nc"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_netcdf(target, mode="w")
        except OSError as exc:
            raise IngestionError(f"Could not cache the ERA5 subset: {exc}") from exc
        return target

    def load(self, path: Path | str, **kwargs: Any) -> xr.Dataset:
        """Open a previously downloaded ERA5 NetCDF subset."""
        from extreme_weather.ingestion.netcdf import NetCDFSource

        return NetCDFSource().load(path, **kwargs)

    def load_request(self, request: OpenMeteoRequest) -> xr.Dataset:
        """Download and convert in one step, without writing to the cache."""
        return era5_payload_to_dataset(self.download_json(request))

