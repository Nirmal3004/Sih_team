"""NetCDF dataset loading and saving for DeepWeather AI (SIH26078).

Phase 0 only needs robust local NetCDF I/O via Xarray:

* :func:`load_dataset` - read a NetCDF file, close the file handle eagerly and
  validate the structure against the Phase 0 data contract.
* :func:`save_dataset` - write a NetCDF file with compression, creating parent
  directories automatically.
* :func:`resolve_dataset_path` - safely resolve a user supplied relative path
  inside the data directory (used by the API to reject path traversal).

Remote/streamed NWP ingestion (GRIB, OPeNDAP, provider APIs) is FUTURE work and
is intentionally not implemented here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from extreme_weather.data.validator import (
    REQUIRED_DIMENSIONS,
    REQUIRED_VARIABLES,
    DatasetLoadError,
    DatasetValidationError,
    validate_dataset,
)

__all__ = [
    "DatasetLoadError",
    "DatasetValidationError",
    "default_encoding",
    "load_dataset",
    "resolve_dataset_path",
    "save_dataset",
]

NETCDF_SUFFIX: str = ".nc"
DEFAULT_COMPRESSION_LEVEL: int = 4


def resolve_dataset_path(base_dir: Path | str, relative_path: str | Path) -> Path:
    """Resolve ``relative_path`` inside ``base_dir`` and reject escapes.

    Args:
        base_dir: Directory that is allowed to contain the dataset.
        relative_path: Dataset path, normally relative to ``base_dir``.

    Returns:
        The absolute, resolved path of the dataset.

    Raises:
        DatasetLoadError: If the path escapes ``base_dir`` or is not a ``.nc`` path.
    """
    base = Path(base_dir).expanduser().resolve()
    candidate = (base / Path(relative_path)).expanduser().resolve()

    if candidate != base and base not in candidate.parents:
        raise DatasetLoadError(
            f"Path '{relative_path}' escapes the data directory '{base}'. "
            "Dataset paths must stay inside the configured data directory."
        )
    if candidate.suffix.lower() != NETCDF_SUFFIX:
        raise DatasetLoadError(
            f"Unsupported dataset extension '{candidate.suffix}'; "
            f"only '{NETCDF_SUFFIX}' files are supported in Phase 0."
        )
    return candidate


def load_dataset(
    path: Path | str,
    *,
    required_variables: Iterable[str] = REQUIRED_VARIABLES,
    required_dimensions: Iterable[str] = REQUIRED_DIMENSIONS,
    validate: bool = True,
    check_plausible_ranges: bool = True,
    engine: str | None = None,
) -> xr.Dataset:
    """Load a NetCDF dataset from disk and validate its structure.

    Args:
        path: Path to a NetCDF (``.nc``) file.
        required_variables: Variables that must be present.
        required_dimensions: Dimensions that must be present.
        validate: When True (default) run the Phase 0 data contract validation.
        check_plausible_ranges: Emit warnings for implausible values.
        engine: Optional Xarray backend engine (for example ``"netcdf4"``).

    Returns:
        An in-memory :class:`xarray.Dataset` (the file handle is closed).

    Raises:
        DatasetLoadError: If the file is missing or cannot be decoded.
        DatasetValidationError: If the dataset breaks the data contract.
    """
    dataset_path = Path(path).expanduser()

    if not dataset_path.exists():
        raise DatasetLoadError(f"Dataset not found: '{dataset_path}'.")
    if not dataset_path.is_file():
        raise DatasetLoadError(f"Expected a NetCDF file but got a directory: '{dataset_path}'.")

    try:
        with xr.open_dataset(dataset_path, engine=engine) as opened:
            # Eagerly load so the file handle is released (important on Windows,
            # where an open handle blocks deletion/overwrite of the file).
            dataset = opened.load()
    except Exception as exc:  # noqa: BLE001 - normalised into a domain error
        raise DatasetLoadError(
            f"Could not read NetCDF dataset '{dataset_path}': {type(exc).__name__}: {exc}"
        ) from exc

    if validate:
        validate_dataset(
            dataset,
            required_variables=tuple(required_variables),
            required_dimensions=tuple(required_dimensions),
            check_plausible_ranges=check_plausible_ranges,
        )
    return dataset


def default_encoding(
    dataset: xr.Dataset,
    *,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
) -> dict[str, dict[str, object]]:
    """Return a conservative NetCDF encoding for ``dataset``.

    Floating point variables are written as ``float32`` with NaN as the fill
    value and zlib compression, which keeps the synthetic demo files small while
    preserving missing values round-trip.
    """
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in dataset.data_vars.items():
        if np.issubdtype(variable.dtype, np.floating):
            encoding[name] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": int(compression_level),
                "_FillValue": np.float32("nan"),
            }
    return encoding


def save_dataset(
    dataset: xr.Dataset,
    path: Path | str,
    *,
    encoding: dict[str, dict[str, object]] | None = None,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    unlimited_dims: Iterable[str] = ("time",),
) -> Path:
    """Write ``dataset`` to ``path`` as NetCDF, creating directories as needed.

    Args:
        dataset: Dataset to persist (it is not modified).
        path: Destination ``.nc`` path.
        encoding: Explicit NetCDF encoding; defaults to :func:`default_encoding`.
        compression_level: zlib level used by the default encoding.
        unlimited_dims: Dimensions written as unlimited (default: ``time``).

    Returns:
        The resolved output path.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"Expected an xarray.Dataset, got {type(dataset).__name__}.")

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_encoding = (
        default_encoding(dataset, compression_level=compression_level)
        if encoding is None
        else encoding
    )
    unlimited = tuple(dim for dim in unlimited_dims if dim in dataset.dims) or None

    dataset.to_netcdf(output_path, mode="w", encoding=resolved_encoding, unlimited_dims=unlimited)
    return output_path
