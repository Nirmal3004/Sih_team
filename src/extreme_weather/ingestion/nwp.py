"""NWP model-output source (SIH26078, Phase 6).

NWP (numerical weather prediction) data is what an operational system would
actually consume: a model run's own output fields, not an analysis. This adapter
covers the two shapes that arrive in practice without claiming to support any
particular centre's catalogue.

Supported
---------
* **GRIB / GRIB2** - the operational standard, via
  :class:`~extreme_weather.ingestion.grib.GRIBSource` (needs an optional
  Xarray backend).
* **NetCDF** - what a centre publishes after conversion.

Credentials
-----------
A true GRIB endpoint (NCEP NOMADS, ECMWF open data) is normally anonymous, but
some mirrors require a key. This adapter therefore supports an **optional**
bearer token read from the environment:

.. code-block:: text

    NWP_API_URL=https://example.invalid/nwp
    NWP_API_TOKEN=...        # optional; never logged, never returned

When a token is required and absent, the adapter raises
:class:`~extreme_weather.ingestion.exceptions.ProviderCredentialsMissingError`
naming the variable to set. It never fabricates data and never echoes a token.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from extreme_weather.ingestion.base import DataOrigin, SourceInfo
from extreme_weather.ingestion.exceptions import (
    DataSourceUnavailableError,
    IngestionError,
    ProviderCredentialsMissingError,
    SourceNotFoundError,
)

__all__ = [
    "NWP_API_TOKEN_ENV",
    "NWP_API_URL_ENV",
    "NWPFormRequest",
    "NWPSource",
]


#: Environment variable naming a remote NWP endpoint (optional).
NWP_API_URL_ENV: str = "NWP_API_URL"

#: Environment variable holding a bearer token (optional, never logged).
NWP_API_TOKEN_ENV: str = "NWP_API_TOKEN"

#: Default request timeout in seconds.
DEFAULT_TIMEOUT_SECONDS: int = 30

#: URL fragments that mark an endpoint as needing a bearer token. Matched
#: against the whole lower-cased URL, so both ``https://api.host/...`` and
#: ``https://host/api/...`` are recognised as authenticated endpoints.
_AUTH_MARKERS: tuple[str, ...] = (
    "/api/",
    "api.",
    "/secure/",
    "token",
    "auth",
    "member",
    "private",
)


def _requires_auth(url: str) -> bool:
    """Return True when a URL advertises itself as protected.

    Only endpoints that look protected are held to the token rule; a plain
    anonymous https mirror is not blocked, because refusing it would break a
    legitimate public source.
    """
    lowered = str(url).lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)


@dataclass(frozen=True)
class NWPFormRequest:
    """A bounded NWP model-output request.

    Attributes:
        url: Endpoint returning a GRIB or NetCDF file.
        lat0: Bottom edge of the latitude range.
        lat1: Top edge of the latitude range.
        lon0: Left edge of the longitude range.
        lon1: Right edge of the longitude range.
        level: Pressure level in hPa, or ``None`` for the surface.
        forecast_hour: Lead time in hours, or ``None`` for an analysis.
        variables: Requested field names.
    """

    url: str
    lat0: float
    lat1: float
    lon0: float
    lon1: float
    level: float | None = None
    forecast_hour: int | None = None
    variables: tuple[str, ...] = ("2t",)

    def __post_init__(self) -> None:
        if not str(self.url).strip():
            raise ValueError("An NWP request needs a url.")
        if self.lat0 > self.lat1:
            raise ValueError(f"lat0 {self.lat0} is north of lat1 {self.lat1}.")
        if self.lon0 > self.lon1:
            raise ValueError(f"lon0 {self.lon0} is east of lon1 {self.lon1}.")
        if not self.variables:
            raise ValueError("An NWP request needs at least one variable.")

    def target(self) -> Path:
        """Return the local cache path this request would write to."""
        parts = [f"{value:g}" for value in (self.lat0, self.lat1, self.lon0, self.lon1)]
        stem = "_".join(parts)
        if self.level is not None:
            stem += f"_lvl{self.level:g}"
        if self.forecast_hour is not None:
            stem += f"_f{self.forecast_hour:03d}"
        suffix = ".nc" if str(self.url).lower().endswith((".nc", ".nc4")) else ".grib"
        return Path(f"nwp_{stem}{suffix}")


class NWPSource:
    """Read NWP model output from a local file or a bounded remote endpoint.

    Args:
        base_dir: Optional directory that local paths are resolved inside.
        cache_dir: Where a remote response is written before being loaded.
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        from extreme_weather.config import get_settings

        settings = get_settings()
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None
        self._cache_dir = Path(cache_dir) if cache_dir is not None else settings.cache_dir

    # -- interface ---------------------------------------------------- #
    def name(self) -> str:
        """Return the stable source identifier."""
        return "nwp"

    def available(self) -> bool:
        """True when NWP output can be read at all.

        A local GRIB/NetCDF file can always be handled; GRIB additionally needs
        its optional backend, which the manifest and the API report separately.
        """
        return True

    def default_origin(self) -> str:
        """A model run is a past forecast, not an observation."""
        return DataOrigin.HISTORICAL

    def metadata(self) -> SourceInfo:
        """Describe the source, including whether an endpoint is configured."""
        configured = bool(os.environ.get(NWP_API_URL_ENV))
        return SourceInfo(
            name=self.name(),
            kind="remote" if configured else "file",
            description=(
                "Numerical weather prediction model output (GRIB/GRIB2 or NetCDF), "
                "local or from a configured endpoint."
            ),
            formats=("grib", "grib2", "netcdf"),
            requires_credentials=not configured,
            required_environment=(NWP_API_URL_ENV,),
            default_origin=DataOrigin.HISTORICAL,
            notes=(
                f"Remote endpoint configured: {configured}. {NWP_API_TOKEN_ENV} is "
                "optional, used only when the endpoint requires it, and is never "
                "logged or returned. Model output is a forecast, not an observation."
            ),
        )


    def resolve(self, path: Path | str) -> Path:
        """Resolve a local path, rejecting anything that escapes ``base_dir``."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and self._base_dir is not None:
            candidate = self._base_dir / candidate
        resolved = candidate.resolve()
        if self._base_dir is not None and resolved != self._base_dir:
            if self._base_dir not in resolved.parents:
                raise SourceNotFoundError(
                    f"Path '{path}' resolves outside the configured data directory "
                    f"'{self._base_dir}'. Dataset paths must stay inside it."
                )
        return resolved

    def fetch(
        self,
        request: NWPFormRequest | None = None,
        *,
        path: Path | str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Return a local path for NWP output, downloading it when remote.

        Exactly one of ``path`` (a local file) or ``request`` (a remote
        endpoint) must be supplied.

        Raises:
            ValueError: If neither or both are supplied.
            SourceNotFoundError: If the local file does not exist.
            ProviderCredentialsMissingError: If the endpoint needs a token and
                ``NWP_API_TOKEN_ENV`` is not set. The message names the variable
                and never its value.
            DataSourceUnavailableError: If the endpoint is unreachable.
        """
        if (path is None) == (request is None):
            raise ValueError("Supply exactly one of 'path' (local) or 'request' (remote).")

        if path is not None:
            resolved = self.resolve(path)
            if not resolved.is_file():
                raise SourceNotFoundError(f"NWP file not found: '{resolved}'.")
            return resolved

        target = self._cache_dir / request.target()  # type: ignore[union-attr]
        if target.is_file():
            return target

        token = os.environ.get(NWP_API_TOKEN_ENV)
        if str(request.url).startswith("https://") and not token:  # type: ignore[union-attr]
            if _requires_auth(request.url):  # type: ignore[union-attr]
                raise ProviderCredentialsMissingError(
                    "This NWP endpoint requires a bearer token. Set "
                    f"{NWP_API_TOKEN_ENV} in the environment. No token is required "
                    "for the public local-file path."
                )

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            request_object = urllib.request.Request(request.url, headers=headers)  # type: ignore[union-attr]
            with urllib.request.urlopen(
                request_object, timeout=DEFAULT_TIMEOUT_SECONDS
            ) as response:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            raise DataSourceUnavailableError(
                f"The NWP endpoint returned HTTP {exc.code} for the requested box."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DataSourceUnavailableError(
                f"Could not reach the NWP endpoint: {type(exc).__name__}. This is a "
                "network problem, not a data problem."
            ) from exc
        return target

    def load(self, path: Path | str, **kwargs: Any) -> xr.Dataset:
        """Open a local NWP file, choosing the reader by extension.

        A GRIB file without its optional backend raises a clear
        :class:`MissingDependencyError` rather than returning an empty dataset.
        """
        from extreme_weather.ingestion.grib import GRIB_SUFFIXES, GRIBSource
        from extreme_weather.ingestion.netcdf import NETCDF_SUFFIXES, NetCDFSource

        resolved = self.resolve(path)
        if not resolved.is_file():
            raise SourceNotFoundError(f"NWP file not found: '{resolved}'.")
        if resolved.suffix.lower() in GRIB_SUFFIXES:
            return GRIBSource().load(resolved, **kwargs)
        if resolved.suffix.lower() in NETCDF_SUFFIXES:
            return NetCDFSource().load(resolved, **kwargs)
        raise IngestionError(
            f"Unsupported NWP file extension '{resolved.suffix}'. Supported: "
            f"{', '.join(NETCDF_SUFFIXES + GRIB_SUFFIXES)}."
        )

