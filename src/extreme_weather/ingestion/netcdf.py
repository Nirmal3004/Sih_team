"""NetCDF file reader (SIH26078, Phase 6).

NetCDF is the project's native format, so this reader is the reference
implementation of :class:`~extreme_weather.ingestion.base.WeatherDataSource`.

Large-data safety
-----------------
A real reanalysis file is easily tens of gigabytes, so :meth:`load` opens the
dataset **lazily** by default (``xr.open_dataset`` without ``.load()``). The
caller decides when to materialise it, and :meth:`load` accepts a ``chunks``
argument to pass straight through to Xarray/Dask.

The returned dataset keeps the file handle open, so this class also exposes
:meth:`open_context`, a context manager that loads and closes. Callers that need
a fully in-memory dataset (the normalizer, the tests) should use it - on Windows
an open handle blocks deletion or overwrite of the file.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import xarray as xr

from extreme_weather.ingestion.base import DataOrigin, SourceInfo
from extreme_weather.ingestion.exceptions import (
    FormatNotSupportedError,
    IngestionError,
    SourceNotFoundError,
)

__all__ = [
    "NETCDF_SUFFIXES",
    "NetCDFSource",
    "default_chunks",
    "file_timestamp",
    "format_extensions",
]

#: Extensions this reader accepts (classic and the CF/HDF5 flavours).
NETCDF_SUFFIXES: tuple[str, ...] = (".nc", ".nc4", ".cdf", ".netcdf")


def format_extensions() -> tuple[str, ...]:
    """Return the file extensions this source can read."""
    return NETCDF_SUFFIXES


def default_chunks(time_steps: int) -> dict[str, int]:
    """Return a sensible chunking for a regional gridded dataset.

    ``time`` is chunked so a chunk holds roughly a day of hourly data. The
    spatial dimensions are left whole because regional grids are small enough
    that splitting them rarely pays off.

    Args:
        time_steps: Number of time steps in the dataset.

    Returns:
        A mapping suitable for ``xr.open_dataset(..., chunks=...)``.
    """
    return {"time": max(1, min(int(time_steps), 24))}


def file_timestamp(path: Path | str) -> str:
    """Return the modification time of a file as an ISO 8601 UTC string."""
    stamp = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
    return stamp.replace(microsecond=0).isoformat()


class NetCDFSource:
    """Read NetCDF files from the local filesystem.

    Args:
        base_dir: Optional directory that relative local paths are resolved
            inside. When given, :meth:`fetch` rejects paths that escape it, which
            is what keeps the API's ingest endpoint from reading arbitrary files.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None

    # -- interface ---------------------------------------------------- #
    def name(self) -> str:
        """Return the stable source identifier."""
        return "netcdf"

    def available(self) -> bool:
        """NetCDF support is a core dependency, so this is always True."""
        return True

    def default_origin(self) -> str:
        """A local file carries no intrinsic origin; callers must state it."""
        return DataOrigin.HISTORICAL

    def metadata(self) -> SourceInfo:
        """Describe the source without reading any data."""
        return SourceInfo(
            name=self.name(),
            kind="file",
            description="NetCDF (.nc/.nc4/.cdf) files from the local filesystem.",
            formats=NETCDF_SUFFIXES,
            requires_credentials=False,
            default_origin=DataOrigin.HISTORICAL,
            notes=(
                "The file's own metadata decides whether it is synthetic or real; "
                "this reader never guesses an origin."
            ),
        )

    def resolve(self, path: Path | str) -> Path:
        """Resolve a path, rejecting anything that escapes ``base_dir``.

        Raises:
            SourceNotFoundError: If the path escapes ``base_dir``.
            FormatNotSupportedError: If the extension is not NetCDF.
        """
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
        if resolved.suffix.lower() not in NETCDF_SUFFIXES:
            raise FormatNotSupportedError(
                f"Unsupported extension '{resolved.suffix}' for the netcdf source; "
                f"supported: {', '.join(NETCDF_SUFFIXES)}."
            )
        return resolved

    def fetch(self, path: Path | str, **_: Any) -> Path:
        """Validate a local NetCDF path and return it.

        Raises:
            SourceNotFoundError: If the file does not exist or is a directory.
            FormatNotSupportedError: If the extension is not NetCDF.
        """
        resolved = self.resolve(path)
        if not resolved.exists():
            raise SourceNotFoundError(f"Dataset not found: '{resolved}'.")
        if not resolved.is_file():
            raise SourceNotFoundError(
                f"Expected a NetCDF file but got a directory: '{resolved}'."
            )
        return resolved

    def load(
        self,
        path: Path | str,
        *,
        chunks: Any | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a NetCDF file and return a (possibly lazy) dataset.

        Args:
            path: Local ``.nc`` path.
            chunks: Xarray/Dask chunk specification. ``None`` keeps the dataset
                lazy; pass e.g. ``{"time": 24}`` to enable chunked reads.
            **kwargs: Forwarded to :func:`xarray.open_dataset`.

        Raises:
            SourceNotFoundError: If the file does not exist.
            IngestionError: If the file cannot be decoded.
        """
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise SourceNotFoundError(f"Dataset not found: '{resolved}'.")

        options: dict[str, Any] = dict(kwargs)
        if chunks is not None:
            options["chunks"] = chunks

        try:
            return xr.open_dataset(resolved, **options)
        except Exception as exc:  # noqa: BLE001 - normalised into a domain error
            raise IngestionError(
                f"Could not read NetCDF dataset '{resolved}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @contextmanager
    def open_context(
        self,
        path: Path | str,
        *,
        chunks: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[xr.Dataset]:
        """Yield a **loaded** dataset and always close the file handle."""
        opened = self.load(path, chunks=chunks, **kwargs)
        try:
            yield opened.load()
        finally:
            try:
                opened.close()
            except Exception:  # noqa: BLE001 - closing must never mask the body
                pass

