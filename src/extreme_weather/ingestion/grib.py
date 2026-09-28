"""GRIB / GRIB2 file reader (SIH26078, Phase 6).

GRIB is the format operational NWP centres actually publish, so supporting it is
what makes the "real data" claim more than a NetCDF round-trip.

Optional dependency
-------------------
Reading GRIB needs an Xarray backend (``cfgrib``/``eccodes`` or ``pygrib``).
Those are **not** added as hard project dependencies, because:

* they are large native builds that are painful to install on Windows;
* NetCDF - the format this project already uses - needs none of them;
* a user who only ever ingests NetCDF should not pay for them.

So the dependency is optional, its availability is detected, and its absence
raises a clear :class:`MissingDependencyError` that names what to install. It
never degrades into an empty dataset, which would look like "no events found".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from extreme_weather.ingestion.base import DataOrigin, SourceInfo
from extreme_weather.ingestion.exceptions import (
    FormatNotSupportedError,
    IngestionError,
    MissingDependencyError,
    SourceNotFoundError,
)

__all__ = [
    "GRIB_SUFFIXES",
    "GRIBSource",
    "grib_backend_available",
    "grib_backend_name",
]


#: Extensions that identify a GRIB file.
GRIB_SUFFIXES: tuple[str, ...] = (".grib", ".grib2", ".grb", ".grb2")

#: The Xarray engines that can decode GRIB, in preference order.
_CANDIDATE_ENGINES: tuple[str, ...] = ("cfgrib", "eccodes", "pygrib")

#: pip requirement that provides the preferred engine.
_INSTALL_HINT = "cfgrib + eccodes (pip install cfgrib eccodes)"


def _detect_engine() -> str | None:
    """Return the first available GRIB engine name, or ``None``.

    Detection is by trying to locate the module, not by inspecting a file, so a
    missing backend is reported identically on every platform.
    """
    import importlib.util

    for engine in _CANDIDATE_ENGINES:
        try:
            if importlib.util.find_spec(engine) is not None:
                return engine
        except (ImportError, ValueError):  # pragma: no cover - defensive
            continue
    return None


def grib_backend_name() -> str | None:
    """Return the available GRIB engine name, or ``None`` when absent."""
    return _detect_engine()


def grib_backend_available() -> bool:
    """Return True when a GRIB decoding engine is installed.

    Used by :meth:`GRIBSource.available` and by the API's source list, so the
    dashboard can show "GRIB - engine: not installed" honestly instead of
    pretending the format is supported.
    """
    return _detect_engine() is not None


class GRIBSource:
    """Read GRIB/GRIB2 files, when a decoding engine is installed."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None

    # -- interface ---------------------------------------------------- #
    def name(self) -> str:
        """Return the stable source identifier."""
        return "grib"

    def available(self) -> bool:
        """True only when a GRIB engine is actually installed."""
        return grib_backend_available()

    def default_origin(self) -> str:
        """A local model file is a historical artefact by default."""
        return DataOrigin.HISTORICAL

    def metadata(self) -> SourceInfo:
        """Describe the source, including whether the engine is present."""
        engine = grib_backend_name()
        return SourceInfo(
            name=self.name(),
            kind="file",
            description="GRIB/GRIB2 (.grib/.grib2/.grb/.grb2) model or analysis files.",
            formats=GRIB_SUFFIXES,
            requires_credentials=False,
            optional_dependency=engine or f"none installed ({_INSTALL_HINT} to enable)",
            default_origin=DataOrigin.HISTORICAL,
            notes=(
                "Optional dependency: GRIB decoding is not a core requirement. "
                f"Installed engine: {engine or 'none'}."
            ),
        )

    def resolve(self, path: Path | str) -> Path:
        """Resolve a GRIB path, rejecting anything that escapes ``base_dir``."""
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
        if resolved.suffix.lower() not in GRIB_SUFFIXES:
            raise FormatNotSupportedError(
                f"Unsupported extension '{resolved.suffix}' for the grib source; "
                f"supported: {', '.join(GRIB_SUFFIXES)}."
            )
        return resolved

    def fetch(self, path: Path | str, **_: Any) -> Path:
        """Validate a local GRIB path and return it.

        The engine check happens in :meth:`load`, not here, so the API can still
        report a missing file (404) before a missing dependency (422).
        """
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise SourceNotFoundError(f"Dataset not found: '{resolved}'.")
        return resolved

    def load(
        self,
        path: Path | str,
        *,
        chunks: Any | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a GRIB file and return a (possibly lazy) dataset.

        Raises:
            MissingDependencyError: If no GRIB engine is installed. The message
                names the package to install; it never returns an empty dataset.
            SourceNotFoundError: If the file does not exist.
            IngestionError: If the engine is installed but cannot decode the file.
        """
        engine = _detect_engine()
        if engine is None:
            raise MissingDependencyError(
                "Reading GRIB requires an optional Xarray backend that is not "
                f"installed. Install it with `pip install {_INSTALL_HINT}`, or "
                "convert the file to NetCDF first. NetCDF ingestion is unaffected."
            )

        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise SourceNotFoundError(f"Dataset not found: '{resolved}'.")

        options: dict[str, Any] = dict(kwargs)
        options.setdefault("engine", engine)
        if chunks is not None:
            options["chunks"] = chunks

        try:
            return xr.open_dataset(resolved, **options)
        except Exception as exc:  # noqa: BLE001 - normalised into a domain error
            raise IngestionError(
                f"Could not read GRIB dataset '{resolved}' with the '{engine}' "
                f"engine: {type(exc).__name__}: {exc}"
            ) from exc

