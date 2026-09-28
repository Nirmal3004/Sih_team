"""Dataset contract validation for DeepWeather AI (SIH26078).

This module defines the *minimum* structure a weather dataset must have to enter
the processing pipeline, and it reports problems as human readable strings
instead of opaque assertions.

The contract is deliberately small for Phase 0:

* dimensions ``time``, ``latitude``, ``longitude``,
* the six core surface variables of the project,
* monotonic, finite, physically plausible coordinates.

Nothing here knows about a specific NWP provider - FUTURE phases may extend
:data:`REQUIRED_VARIABLES` per source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from extreme_weather.config import KM_PER_DEGREE_LATITUDE

# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
TIME_DIM: str = "time"
LATITUDE_DIM: str = "latitude"
LONGITUDE_DIM: str = "longitude"

REQUIRED_DIMENSIONS: tuple[str, ...] = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM)

REQUIRED_VARIABLES: tuple[str, ...] = (
    "precipitation",
    "temperature",
    "pressure",
    "relative_humidity",
    "u_wind",
    "v_wind",
)

#: Plausible physical ranges used for *warnings* (never for hard failures).
#: ``relative_humidity`` is percent, ``pressure`` is pascal (CF convention),
#: ``temperature`` is kelvin, ``precipitation`` is mm/h, winds are m/s.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "precipitation": (0.0, 500.0),
    "temperature": (180.0, 350.0),
    "pressure": (85_000.0, 110_000.0),
    "relative_humidity": (0.0, 100.0),
    "u_wind": (-150.0, 150.0),
    "v_wind": (-150.0, 150.0),
}

MIN_COORDINATE_POINTS: int = 2


class DatasetError(Exception):
    """Base class for dataset related failures (loading or validation)."""


class DatasetLoadError(DatasetError):
    """Raised when a dataset cannot be found, read or safely located on disk."""


class DatasetValidationError(DatasetError):
    """Raised when a dataset violates the Phase 0 data contract."""


@dataclass(frozen=True)
class VariableInfo:
    """Lightweight description of one data variable (metadata only)."""

    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    units: str | None = None
    long_name: str | None = None


@dataclass(frozen=True)
class DatasetSummary:
    """Metadata-only summary of a dataset (no bulk array access)."""

    name: str
    path: str | None
    dimensions: dict[str, int]
    data_variables: tuple[str, ...]
    variables: tuple[VariableInfo, ...]
    time_steps: int
    time_start: str | None
    time_end: str | None
    time_step_seconds: float | None
    latitude_min: float | None
    latitude_max: float | None
    latitude_points: int
    longitude_min: float | None
    longitude_max: float | None
    longitude_points: int
    grid_resolution_deg: float | None
    approximate_grid_resolution_km: float | None
    is_synthetic: bool
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the summary as a plain (JSON friendly) dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssues:
    """Container for validation findings."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when no hard contract violation was found."""
        return not self.errors


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def _is_monotonic_increasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) > 0))


def _is_monotonic_decreasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) < 0))


def _check_horizontal_coordinate(
    dataset: xr.Dataset,
    name: str,
    errors: list[str],
    warnings: list[str],
    *,
    abs_limit: float,
    axis_label: str,
) -> None:
    """Validate a 1-D latitude/longitude coordinate and record findings."""
    if name not in dataset.coords:
        errors.append(
            f"Coordinate '{name}' is missing; expected a 1-D '{axis_label}' coordinate."
        )
        return

    coord = dataset.coords[name]
    if coord.ndim != 1:
        errors.append(
            f"Coordinate '{name}' must be one-dimensional, found {coord.ndim} dimensions "
            f"{tuple(coord.dims)}."
        )
        return

    values = np.asarray(coord.values, dtype="float64")
    if values.size < MIN_COORDINATE_POINTS:
        errors.append(
            f"Coordinate '{name}' needs at least {MIN_COORDINATE_POINTS} points, "
            f"found {values.size}."
        )
        return
    if not np.all(np.isfinite(values)):
        errors.append(f"Coordinate '{name}' contains non-finite values (NaN/inf).")
        return
    if not (_is_monotonic_increasing(values) or _is_monotonic_decreasing(values)):
        errors.append(
            f"Coordinate '{name}' must be strictly monotonic; got start={values[0]:.4f} and "
            f"end={values[-1]:.4f} with non-monotonic steps in between."
        )
    if np.any(np.abs(values) > abs_limit):
        errors.append(
            f"Coordinate '{name}' has values outside [-{abs_limit}, {abs_limit}]: "
            f"min={values.min():.4f}, max={values.max():.4f}."
        )
    if name == LONGITUDE_DIM and (values.min() < -180.0 or values.max() > 180.0):
        warnings.append(
            f"Coordinate '{name}' exceeds the conventional [-180, 180] range "
            f"(min={values.min():.4f}, max={values.max():.4f}); consider normalising it."
        )


def _check_time_coordinate(dataset: xr.Dataset, errors: list[str], warnings: list[str]) -> None:
    """Validate the ``time`` coordinate (dtype, monotonicity, spacing)."""
    if TIME_DIM not in dataset.coords:
        errors.append(
            f"Coordinate '{TIME_DIM}' is missing; expected a 1-D datetime64 coordinate."
        )
        return

    coord = dataset.coords[TIME_DIM]
    if coord.ndim != 1:
        errors.append(
            f"Coordinate '{TIME_DIM}' must be one-dimensional, found {coord.ndim} dimensions "
            f"{tuple(coord.dims)}."
        )
        return

    values = np.asarray(coord.values)
    if not np.issubdtype(values.dtype, np.datetime64):
        errors.append(
            f"Coordinate '{TIME_DIM}' must have a datetime64 dtype, found '{values.dtype}'. "
            "Open the file with decode_times=True."
        )
        return

    if values.size < MIN_COORDINATE_POINTS:
        errors.append(
            f"Coordinate '{TIME_DIM}' needs at least {MIN_COORDINATE_POINTS} steps, "
            f"found {values.size}."
        )
        return

    if np.isnat(values).any():
        errors.append(f"Coordinate '{TIME_DIM}' contains NaT (missing timestamps).")
        return

    if not _is_monotonic_increasing(values):
        errors.append(
            f"Coordinate '{TIME_DIM}' must be strictly increasing; "
            f"first={values[0]}, last={values[-1]}."
        )
        return

    deltas = np.diff(values).astype("timedelta64[s]").astype("int64")
    if deltas.size and len(set(deltas.tolist())) > 1:
        warnings.append(
            f"Coordinate '{TIME_DIM}' is not uniformly spaced "
            f"(min step={int(deltas.min())}s, max step={int(deltas.max())}s)."
        )



# --------------------------------------------------------------------------- #
# Public validation API
# --------------------------------------------------------------------------- #
def collect_validation_issues(
    dataset: xr.Dataset,
    *,
    required_variables: tuple[str, ...] = REQUIRED_VARIABLES,
    required_dimensions: tuple[str, ...] = REQUIRED_DIMENSIONS,
    check_plausible_ranges: bool = True,
) -> ValidationIssues:
    """Inspect ``dataset`` and return all contract violations and warnings.

    Errors make the dataset unusable for the pipeline; warnings describe
    suspicious but still processable content (for example a variable that is
    entirely NaN, or a value outside its plausible physical range).
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"Expected an xarray.Dataset, got {type(dataset).__name__}.")

    errors: list[str] = []
    warnings: list[str] = []

    missing_variables = [name for name in required_variables if name not in dataset.data_vars]
    if missing_variables:
        errors.append(
            f"Missing required data variable(s): {missing_variables}. "
            f"Found: {sorted(dataset.data_vars)}."
        )

    missing_dimensions = [name for name in required_dimensions if name not in dataset.dims]
    if missing_dimensions:
        errors.append(
            f"Missing required dimension(s): {missing_dimensions}. "
            f"Found: {sorted(dataset.dims)}."
        )

    # Each required dimension should be a coordinate, not just a size.
    for name in required_dimensions:
        if name in dataset.dims and name not in dataset.coords:
            errors.append(
                f"Dimension '{name}' has no coordinate; add coordinate values so that "
                "spatial/temporal indexing is meaningful."
            )

    # Variables must span the required dimensions.
    for name in required_variables:
        if name not in dataset.data_vars:
            continue
        variable = dataset.data_vars[name]
        missing_dims = [dim for dim in required_dimensions if dim not in variable.dims]
        if missing_dims:
            errors.append(
                f"Variable '{name}' is missing required dimension(s) {missing_dims}; "
                f"found dimensions {tuple(variable.dims)}."
            )
        if not np.issubdtype(variable.dtype, np.floating):
            warnings.append(
                f"Variable '{name}' has non-floating dtype '{variable.dtype}'; missing-value "
                "handling expects float data."
            )

    if LATITUDE_DIM in dataset.dims:
        _check_horizontal_coordinate(
            dataset, LATITUDE_DIM, errors, warnings, abs_limit=90.0, axis_label="lat"
        )
    if LONGITUDE_DIM in dataset.dims:
        _check_horizontal_coordinate(
            dataset, LONGITUDE_DIM, errors, warnings, abs_limit=360.0, axis_label="lon"
        )
    if TIME_DIM in dataset.dims:
        _check_time_coordinate(dataset, errors, warnings)

    if check_plausible_ranges:
        for name, (low, high) in PLAUSIBLE_RANGES.items():
            if name not in dataset.data_vars:
                continue
            values = np.asarray(dataset.data_vars[name].values, dtype="float64")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                warnings.append(
                    f"Variable '{name}' contains only missing values (NaN) - "
                    "check the source data or the missing-value strategy."
                )
                continue
            if finite.min() < low or finite.max() > high:
                warnings.append(
                    f"Variable '{name}' has values outside the plausible range "
                    f"[{low}, {high}]: min={finite.min():.4g}, max={finite.max():.4g}."
                )

    return ValidationIssues(errors=tuple(errors), warnings=tuple(warnings))


def validate_dataset(
    dataset: xr.Dataset,
    *,
    required_variables: tuple[str, ...] = REQUIRED_VARIABLES,
    required_dimensions: tuple[str, ...] = REQUIRED_DIMENSIONS,
    check_plausible_ranges: bool = True,
) -> ValidationIssues:
    """Validate ``dataset`` and raise :class:`DatasetValidationError` on errors.

    Returns the :class:`ValidationIssues` (including warnings) when the dataset
    is acceptable.
    """
    issues = collect_validation_issues(
        dataset,
        required_variables=required_variables,
        required_dimensions=required_dimensions,
        check_plausible_ranges=check_plausible_ranges,
    )
    if issues.errors:
        bullet_list = "\n".join(f"  - {message}" for message in issues.errors)
        raise DatasetValidationError(f"Dataset validation failed:\n{bullet_list}")
    return issues



# --------------------------------------------------------------------------- #
# Metadata summary (used by the API/dashboard layer)
# --------------------------------------------------------------------------- #
def summarise_dataset(
    dataset: xr.Dataset, *, name: str | None = None, path: str | None = None
) -> DatasetSummary:
    """Build a metadata-only :class:`DatasetSummary` for the API/dashboard."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"Expected an xarray.Dataset, got {type(dataset).__name__}.")

    variables = tuple(
        VariableInfo(
            name=variable_name,
            dimensions=tuple(variable.dims),
            shape=tuple(int(size) for size in variable.shape),
            dtype=str(variable.dtype),
            units=variable.attrs.get("units"),
            long_name=variable.attrs.get("long_name"),
        )
        for variable_name, variable in dataset.data_vars.items()
    )

    latitude = _coordinate_values(dataset, LATITUDE_DIM)
    longitude = _coordinate_values(dataset, LONGITUDE_DIM)
    time = _coordinate_values(dataset, TIME_DIM)

    spacings: list[float] = []
    for values in (latitude, longitude):
        if values is not None and values.size >= 2:
            spacings.append(float(np.mean(np.abs(np.diff(values)))))
    resolution_deg = round(float(np.mean(spacings)), 6) if spacings else None

    time_step_seconds: float | None = None
    if time is not None and time.size >= 2 and np.issubdtype(time.dtype, np.datetime64):
        deltas = np.diff(time).astype("timedelta64[s]").astype("int64")
        time_step_seconds = float(np.mean(deltas))

    attributes: dict[str, Any] = {
        str(key): (value.item() if hasattr(value, "item") else value)
        for key, value in dataset.attrs.items()
    }
    source = str(attributes.get("source", "")).lower()
    dataset_type = str(attributes.get("dataset_type", "")).lower()

    return DatasetSummary(
        name=name or str(attributes.get("title", "dataset")),
        path=path,
        dimensions={dim: int(size) for dim, size in dataset.sizes.items()},
        data_variables=tuple(dataset.data_vars),
        variables=variables,
        time_steps=int(time.size) if time is not None else 0,
        time_start=_iso_timestamp(time[0]) if time is not None and time.size else None,
        time_end=_iso_timestamp(time[-1]) if time is not None and time.size else None,
        time_step_seconds=time_step_seconds,
        latitude_min=float(np.min(latitude)) if latitude is not None and latitude.size else None,
        latitude_max=float(np.max(latitude)) if latitude is not None and latitude.size else None,
        latitude_points=int(latitude.size) if latitude is not None else 0,
        longitude_min=(
            float(np.min(longitude)) if longitude is not None and longitude.size else None
        ),
        longitude_max=(
            float(np.max(longitude)) if longitude is not None and longitude.size else None
        ),
        longitude_points=int(longitude.size) if longitude is not None else 0,
        grid_resolution_deg=resolution_deg,
        approximate_grid_resolution_km=(
            round(resolution_deg * KM_PER_DEGREE_LATITUDE, 3) if resolution_deg else None
        ),
        is_synthetic="synthetic" in source or "synthetic" in dataset_type,
        attributes=attributes,
    )


def _coordinate_values(dataset: xr.Dataset, name: str) -> np.ndarray | None:
    """Return the values of a 1-D coordinate, or ``None`` when unavailable."""
    if name not in dataset.coords:
        return None
    coord = dataset.coords[name]
    if coord.ndim != 1:
        return None
    return np.asarray(coord.values)


def _iso_timestamp(value: Any) -> str:
    """Format a datetime64-like scalar as a compact ISO 8601 string."""
    return str(np.datetime_as_string(np.asarray(value), unit="s"))
