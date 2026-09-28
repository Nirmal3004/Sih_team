"""Weather providers for near-real-time ingestion (SIH26078, Phase 8).

A provider answers one question: *given a window of time and a box on the
Earth, bring back weather data.* Everything downstream - the quality gate, the
freshness tracker, the orchestrator - is written against
:class:`WeatherProvider` and never against a specific service.

.. code-block:: text

    WeatherProvider
        LocalFileProvider      a file already on disk
        OpenMeteoProvider      the Phase 6 ERA5/archive path, unchanged
        RemoteHttpProvider     a configurable HTTP endpoint
        FixtureProvider        deterministic in-memory data (tests, demos)

Why the abstraction matters
---------------------------
The pipeline must be runnable against a provider nobody has configured yet, and
testable without a network. A concrete provider class baked into the
orchestrator would make both impossible.

Every provider
--------------
* returns a dataset in the **source's own units**, because Phase 6's normaliser
  is the single place that converts them;
* records a :class:`ProviderFetch` describing what it asked for and what it got,
  including the response checksum used for idempotency;
* raises a *specific* exception, so a timeout can be retried and a malformed
  response cannot;
* **never invents data**. A provider that cannot deliver raises; it does not
  synthesise a plausible field.

Remote safety
-------------
``RemoteHttpProvider`` validates its URL scheme, enforces a timeout, caps the
response size and never embeds a credential in a URL that could be logged. API
keys are read from the environment and passed as headers, never as query
parameters.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from extreme_weather.ingestion.manifest import content_hash
from extreme_weather.realtime.exceptions import (
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_RESPONSE_BYTES",
    "FixtureProvider",
    "LocalFileProvider",
    "OpenMeteoProvider",
    "ProviderFetch",
    "RemoteHttpProvider",
    "WeatherProvider",
    "WeatherRequest",
    "build_provider",
    "default_providers",
]

#: URL schemes a remote provider may use. Anything else is refused outright, so
#: a misconfigured endpoint cannot become a ``file://`` read.
ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"https"})

#: Default network timeout in seconds. Every remote call is bounded.
DEFAULT_TIMEOUT_SECONDS: int = 30

#: Hard cap on a downloaded response, in bytes. A misconfigured endpoint must
#: not be able to fill the disk or the process memory.
MAX_RESPONSE_BYTES: int = 256 * 1024 * 1024

#: Environment variables a remote provider may read a token from. They are never
#: logged and never placed in a URL.
TOKEN_ENV_VARS: tuple[str, ...] = ("WEATHER_API_TOKEN", "OPEN_METEO_API_KEY")

UTC = timezone.utc


def _as_utc(text: str) -> datetime | None:
    """Parse an ISO timestamp into an aware UTC datetime, or return ``None``.

    A bare timestamp is interpreted as UTC, which is the convention every
    provider in this project stores data under.
    """
    candidate = str(text).strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True)
class WeatherRequest:
    """What to fetch.

    Attributes:
        start: Inclusive window start, timezone-aware.
        end: Inclusive window end, timezone-aware.
        variables: Provider-specific variable names.
        latitude: Latitude points of the bounding box.
        longitude: Longitude points of the bounding box.
        timeout_seconds: Per-request network timeout.
    """

    start: datetime
    end: datetime
    variables: tuple[str, ...]
    latitude: tuple[float, ...]
    longitude: tuple[float, ...]
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ProviderError(
                f"Request window ends ({self.end}) before it starts ({self.start})."
            )
        if not self.variables:
            raise ProviderError("A weather request needs at least one variable.")
        if not self.latitude or not self.longitude:
            raise ProviderError(
                "A weather request needs an explicit bounding box; an unbounded "
                "request would be a multi-gigabyte download."
            )
        for value in self.latitude:
            if not -90.0 <= value <= 90.0:
                raise ProviderError(f"Latitude {value} is outside [-90, 90].")
        for value in self.longitude:
            if not -180.0 <= value <= 180.0:
                raise ProviderError(f"Longitude {value} is outside [-180, 180].")
        if int(self.timeout_seconds) <= 0:
            raise ProviderError("timeout_seconds must be positive.")

    @property
    def hours(self) -> int:
        """Length of the window in whole hours, at least one."""
        return max(1, int((self.end - self.start).total_seconds() // 3600) + 1)

    def latitudes_key(self) -> tuple[float, ...]:
        """Latitude points rounded for a stable identity string."""
        return tuple(round(float(v), 4) for v in self.latitude)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the request."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "hours": self.hours,
            "variables": list(self.variables),
            "latitude": list(self.latitudes_key()),
            "longitude": [round(float(v), 4) for v in self.longitude],
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class ProviderFetch:
    """What a provider actually did.

    Attributes:
        provider: Provider name.
        dataset: The fetched data, in the source's own units.
        requested_at: When the request was issued.
        source_url: The URL used, or ``None`` for a local file.
        checksum: SHA-256 of the array values, used for idempotency.
        source_timestamp: The newest timestamp the provider reported.
        variables: Variable names actually returned.
        notes: Anything worth recording about this fetch.
    """

    provider: str
    dataset: xr.Dataset
    requested_at: datetime
    source_url: str | None = None
    checksum: str = ""
    source_timestamp: str | None = None
    variables: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.checksum:
            object.__setattr__(self, "checksum", content_hash(self.dataset))
        if not self.variables:
            object.__setattr__(
                self,
                "variables",
                tuple(sorted(str(n) for n in self.dataset.data_vars)),
            )
        if self.source_timestamp is None and "time" in self.dataset.coords:
            times = np.asarray(self.dataset.coords["time"].values)
            if times.size:
                object.__setattr__(
                    self,
                    "source_timestamp",
                    str(np.datetime_as_string(times[-1], unit="s")),
                )

    @property
    def latest_time(self) -> datetime | None:
        """The newest timestamp in the fetched data, as an aware UTC datetime.

        Parsed from the ISO string rather than from the numpy scalar: converting
        a numpy datetime64 to a Python datetime depends on the array's dtype
        unit and is not reliable across the datetime64 and cftime paths a
        NetCDF file may use.
        """
        if "time" not in self.dataset.coords:
            return None
        times = np.asarray(self.dataset.coords["time"].values)
        if not times.size:
            return None
        stamp = _as_utc(str(np.datetime_as_string(times[-1], unit="s")))
        if stamp is None and self.source_timestamp:
            stamp = _as_utc(self.source_timestamp)
        return stamp

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, with no array data."""
        return {
            "provider": self.provider,
            "source_url": self.source_url,
            "requested_at": self.requested_at.isoformat(),
            "source_timestamp": self.source_timestamp,
            "checksum": self.checksum,
            "variables": list(self.variables),
            "time_steps": int(self.dataset.sizes.get("time", 0)),
            "spatial_points": int(self.dataset.sizes.get("latitude", 0))
            * int(self.dataset.sizes.get("longitude", 0)),
            "notes": list(self.notes),
        }


class WeatherProvider(ABC):
    """The interface every weather provider implements."""

    #: Stable provider identifier, used in manifests and run identities.
    name: str = "weather"

    #: Human-readable description shown by ``--help`` and the API.
    description: str = ""

    @abstractmethod
    def available(self) -> bool:
        """Return True when this provider can be used right now.

        Must not raise and must not touch the network: it answers "is this
        configured?", which is what a status endpoint needs.
        """

    @abstractmethod
    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Fetch a window of weather data.

        Returns:
            A :class:`ProviderFetch` carrying the data in its source units.

        Raises:
            ProviderUnavailableError: The provider could not be reached.
            ProviderTimeoutError: The provider did not answer in time.
            ProviderResponseError: The response was unusable.
        """

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the provider."""
        return {
            "name": self.name,
            "description": self.description,
            "available": self.available(),
        }



class LocalFileProvider(WeatherProvider):
    """Read a NetCDF/GRIB file that is already on disk.

    This is the provider used for replay, for air-gapped runs and for testing,
    and it is the reason a local file never needs a network or an API key.
    """

    name = "local"
    description = "A NetCDF or GRIB file already present on disk."

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The configured file path."""
        return self._path

    def available(self) -> bool:
        """True when the configured file exists."""
        return self._path.is_file()

    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Open the file and return it, unchanged.

        The time window is *not* applied here: Phase 6's normaliser and the
        incremental processor own time handling, and silently trimming the file
        in the provider would make the checksum depend on the request rather
        than on the data.
        """
        from extreme_weather.ingestion.netcdf import NetCDFSource

        if not self._path.is_file():
            raise ProviderUnavailableError(
                f"The configured local file '{self._path}' does not exist."
            )
        opened = NetCDFSource().load(self._path)
        try:
            dataset = opened.load()
        finally:
            try:
                opened.close()
            except Exception:  # noqa: BLE001 - closing must not mask the body
                pass
        return ProviderFetch(
            provider=self.name,
            dataset=dataset,
            requested_at=datetime.now(UTC),
            source_url=None,
            notes=(f"Read from local file {self._path.name}.",),
        )


class OpenMeteoProvider(WeatherProvider):
    """The Phase 6 ERA5/Open-Meteo archive path, reused unchanged.

    Deliberately a *thin* wrapper: the request construction, the payload
    conversion and the attribution all come from
    :class:`~extreme_weather.ingestion.era5.ERA5Source`, so there is exactly one
    implementation of "how ERA5 is fetched" in this project.
    """

    name = "open-meteo"
    description = (
        "ERA5 reanalysis via the Open-Meteo archive API. Historical, not live."
    )

    def available(self) -> bool:
        """The archive needs no credentials, so it is always configured."""
        return True

    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Fetch an ERA5 subset for the requested window and box.

        Raises:
            ProviderResponseError: The archive answered with something unusable.
            ProviderUnavailableError: The archive could not be reached.
        """
        from extreme_weather.ingestion.era5 import ERA5Source, OpenMeteoRequest
        from extreme_weather.ingestion.exceptions import (
            DataSourceUnavailableError,
            IngestionError,
        )

        archive_request = OpenMeteoRequest(
            latitudes=request.latitude,
            longitudes=request.longitude,
            start_date=request.start.date().isoformat(),
            end_date=request.end.date().isoformat(),
            variables=request.variables,
            timeout_seconds=request.timeout_seconds,
        )
        source = ERA5Source()
        try:
            dataset = source.load_request(archive_request)
        except DataSourceUnavailableError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        except IngestionError as exc:
            raise ProviderResponseError(
                f"The archive response could not be converted to a dataset: {exc}"
            ) from exc

        return ProviderFetch(
            provider=self.name,
            dataset=dataset,
            requested_at=datetime.now(UTC),
            source_url=ERA5Source().metadata().description and None,
            notes=(
                "ERA5 reanalysis: real observations fused with a model, "
                "describing the past. Not a forecast and not live.",
            ),
        )



class RemoteHttpProvider(WeatherProvider):
    """Fetch from a configurable HTTPS endpoint returning Open-Meteo-shaped JSON.

    Written for the case where a deployment has its own NWP gateway. The
    endpoint is *not* hard-coded: it arrives through configuration, and an
    unrecognised scheme is refused rather than attempted.

    Safety properties, all enforced below:

    * only ``https`` is accepted, so a misconfigured endpoint cannot become a
      local file read;
    * every request carries a timeout;
    * the response is capped at :data:`MAX_RESPONSE_BYTES`;
    * a token is read from the environment and sent as a **header**, never as a
      query parameter, so it cannot end up in a log line or a crash report.
    """

    name = "remote-http"
    description = "A configurable HTTPS weather endpoint returning JSON."

    def __init__(
        self,
        url: str,
        *,
        token_env: str | None = None,
        parser: Any | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._url = self._validate_url(url)
        self._token_env = token_env
        self._parser = parser
        self._timeout = int(timeout_seconds)

    @staticmethod
    def _validate_url(url: str) -> str:
        """Return the URL, or refuse it.

        Raises:
            ProviderConfigurationError: The URL is missing, malformed, or uses a
                scheme that is not allowed.
        """
        from extreme_weather.realtime.exceptions import PipelineConfigurationError

        candidate = str(url).strip()
        if not candidate:
            raise PipelineConfigurationError("A remote provider needs a URL.")
        parsed = urllib.parse.urlparse(candidate)
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            raise PipelineConfigurationError(
                f"Remote provider URL scheme '{parsed.scheme}' is not allowed; "
                f"use one of {', '.join(sorted(ALLOWED_URL_SCHEMES))}."
            )
        if not parsed.netloc:
            raise PipelineConfigurationError(
                f"Remote provider URL '{candidate}' has no host."
            )
        return candidate

    @property
    def url(self) -> str:
        """The validated endpoint URL."""
        return self._url

    def available(self) -> bool:
        """A configured URL means the provider is configured.

        Deliberately does not probe the network: an availability check that
        made a request would make a status endpoint slow and would consume the
        provider's rate limit every time it was polled.
        """
        return bool(self._url)

    def _headers(self) -> dict[str, str]:
        """Build request headers, reading any token from the environment."""
        headers = {"Accept": "application/json", "User-Agent": "deepweather-ai/0.1"}
        names = (self._token_env,) if self._token_env else TOKEN_ENV_VARS
        for name in names:
            value = os.environ.get(str(name))
            if value:
                # Header, never a query parameter: a URL can be logged.
                headers["Authorization"] = f"Bearer {value}"
                break
        return headers

    def build_url(self, request: WeatherRequest) -> str:
        """Return the fully parameterised request URL.

        The window, the box and the variables go in as query parameters. The
        token deliberately does **not**.
        """
        params = {
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "latitude": ",".join(f"{v:g}" for v in request.latitudes_key()),
            "longitude": ",".join(f"{v:g}" for v in request.longitude),
            "variables": ",".join(request.variables),
            "timezone": "UTC",
        }
        return f"{self._url}?{urllib.parse.urlencode(params)}"

    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Fetch and parse the endpoint's JSON response.

        Raises:
            ProviderUnavailableError: The endpoint could not be reached.
            ProviderTimeoutError: The endpoint did not answer in time.
            ProviderResponseError: The body was not usable JSON or not a
                recognised archive shape.
        """
        url = self.build_url(request)
        timeout = int(request.timeout_seconds or self._timeout)
        try:
            with urllib.request.urlopen(
                url, timeout=timeout, headers=self._headers()
            ) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_RESPONSE_BYTES:
                    raise ProviderResponseError(
                        f"The endpoint advertises {int(declared)} bytes, over the "
                        f"{MAX_RESPONSE_BYTES // (1024 * 1024)} MB limit."
                    )
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise ProviderUnavailableError(
                f"The endpoint returned HTTP {exc.code}.", attempts=1
            ) from exc
        except (TimeoutError,) as exc:
            raise ProviderTimeoutError(
                f"The endpoint did not answer within {timeout}s.", attempts=1
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise ProviderTimeoutError(
                    f"The endpoint did not answer within {timeout}s.", attempts=1
                ) from exc
            raise ProviderUnavailableError(
                f"The endpoint could not be reached: {reason}.", attempts=1
            ) from exc
        except OSError as exc:
            raise ProviderUnavailableError(
                f"The endpoint could not be reached: {exc}.", attempts=1
            ) from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderResponseError(
                "The endpoint response exceeded the size limit."
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"The endpoint did not return valid JSON: {exc}"
            ) from exc

        parser = self._parser or _parse_archive_payload
        dataset = parser(payload)
        return ProviderFetch(
            provider=self.name,
            dataset=dataset,
            requested_at=datetime.now(UTC),
            source_url=url,
            notes=(f"Fetched from the configured endpoint {self._url}.",),
        )


def _parse_archive_payload(payload: Any) -> xr.Dataset:
    """Convert an Open-Meteo-shaped payload into a dataset.

    Raises:
        ProviderResponseError: The payload does not have the expected shape.
    """
    from extreme_weather.ingestion.era5 import era5_payload_to_dataset
    from extreme_weather.ingestion.exceptions import IngestionError

    try:
        return era5_payload_to_dataset(payload)
    except IngestionError as exc:
        raise ProviderResponseError(
            f"The endpoint response is not a usable archive payload: {exc}"
        ) from exc



class FixtureProvider(WeatherProvider):
    """Deterministic in-memory weather, for tests, demos and dry runs.

    This provider exists so the whole pipeline can be exercised with **no
    network and no API key**, which is what makes the orchestration testable at
    all. The data is a smooth analytic field plus a reproducible diurnal term -
    clearly synthetic, and labelled as such in the fetch notes. It is never
    presented as an observation.

    The same request always produces the same bytes, so idempotency can be
    tested for real rather than mocked away.
    """

    name = "fixture"
    description = "Deterministic synthetic weather, for tests and demos."

    def __init__(
        self,
        *,
        latitude: tuple[float, ...] = (12.0, 12.5, 13.0),
        longitude: tuple[float, ...] = (76.0, 76.5, 77.0),
        variables: tuple[str, ...] = ("precipitation", "temperature"),
        seed: int = 20240601,
    ) -> None:
        self._latitude = latitude
        self._longitude = longitude
        self._variables = variables
        self._seed = int(seed)

    def available(self) -> bool:
        """A fixture provider is always usable; it has no dependency."""
        return True

    def fetch(self, request: WeatherRequest) -> ProviderFetch:
        """Build a deterministic dataset for the requested window.

        Returns:
            A :class:`ProviderFetch` whose checksum is stable for a given
            (window, box, variables, seed) - which is what makes the
            idempotency test meaningful.
        """
        # datetime64 holds no timezone, so the window is converted to naive UTC
        # before storing. NetCDF stores naive times too, and every provider in
        # this project stores UTC, so nothing is lost.
        base = request.start.astimezone(UTC).replace(tzinfo=None)
        stamps = np.array(
            [
                np.datetime64(base + _timedelta(hours=offset), "m")
                for offset in range(request.hours)
            ],
            dtype="datetime64[ns]",
        )
        rng = np.random.default_rng(self._seed)
        shape = (stamps.size, len(self._latitude), len(self._longitude))

        hour = stamps.astype("datetime64[h]").astype(int) % 24
        diurnal = np.sin(2.0 * np.pi * hour / 24.0)

        data: dict[str, Any] = {}
        # Honour the variables the caller actually asked for; falling back to
        # this provider's own defaults would silently ignore the request and
        # make a fetch look successful while returning the wrong fields.
        wanted = request.variables or self._variables
        for name in wanted:
            if name in ("temperature", "temperature_2m"):
                data[name] = 300.0 + 3.0 * diurnal[:, None, None] + rng.normal(
                    0, 0.2, shape
                )
            elif name in ("precipitation", "tp"):
                data[name] = np.maximum(
                    1.0 + 4.0 * diurnal[:, None, None] + rng.normal(0, 0.3, shape), 0.0
                )
            else:
                # Any other requested variable gets a bounded, harmless field so
                # a test asking for it does not crash.
                data[name] = np.abs(rng.normal(1.0, 0.2, shape))

        dataset = xr.Dataset(
            data_vars={
                name: xr.DataArray(
                    values.astype("float64"),
                    dims=("time", "latitude", "longitude"),
                    attrs={"units": "K" if "temp" in name else "mm"},
                )
                for name, values in data.items()
            },
            coords={
                "time": stamps,
                "latitude": np.asarray(self._latitude, dtype="float64"),
                "longitude": np.asarray(self._longitude, dtype="float64"),
            },
            attrs={
                "source": "fixture",
                "is_synthetic": "true",
                "dataset_type": "synthetic_fixture",
            },
        )
        return ProviderFetch(
            provider=self.name,
            dataset=dataset,
            requested_at=datetime.now(UTC),
            source_url=None,
            notes=(
                "Deterministic SYNTHETIC fixture data. Not an observation and not "
                "a forecast.",
            ),
        )


def _timedelta(**kwargs: Any):
    """Return a ``datetime.timedelta``; a tiny indirection keeps imports local."""
    from datetime import timedelta

    return timedelta(**kwargs)


def default_providers() -> dict[str, WeatherProvider]:
    """Return the providers that need no configuration.

    The remote provider is **not** included: it needs a URL, and a provider that
    cannot work is not something to advertise in a status listing.
    """
    return {
        "local": LocalFileProvider(Path(".")),
        "open-meteo": OpenMeteoProvider(),
        "fixture": FixtureProvider(),
    }


def build_provider(
    name: str,
    *,
    local_path: Path | str | None = None,
    remote_url: str | None = None,
    token_env: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> WeatherProvider:
    """Construct a provider by name.

    Args:
        name: ``local``, ``open-meteo``, ``remote-http`` or ``fixture``.
        local_path: Required for ``local``.
        remote_url: Required for ``remote-http``.
        token_env: Optional environment variable holding a bearer token.
        timeout_seconds: Per-request network timeout.

    Returns:
        The constructed provider.

    Raises:
        PipelineConfigurationError: The name is unknown, or a required
            argument was not supplied.
    """
    from extreme_weather.realtime.exceptions import PipelineConfigurationError

    key = str(name).strip().lower()
    if key == "local":
        if local_path is None:
            raise PipelineConfigurationError(
                "The 'local' provider needs a --file path."
            )
        return LocalFileProvider(local_path)
    if key in ("open-meteo", "era5", "openmeteo"):
        return OpenMeteoProvider()
    if key in ("remote", "remote-http", "http"):
        if not remote_url:
            raise PipelineConfigurationError(
                "The 'remote-http' provider needs a --url."
            )
        return RemoteHttpProvider(
            remote_url, token_env=token_env, timeout_seconds=timeout_seconds
        )
    if key in ("fixture", "test", "synthetic"):
        return FixtureProvider()
    known = ", ".join(sorted({"local", "open-meteo", "remote-http", "fixture"}))
    raise PipelineConfigurationError(
        f"Unknown provider '{name}'. Known providers: {known}."
    )

