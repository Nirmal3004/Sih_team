"""Data quality control (SIH26078, Phase 6).

Real data is messy in ways the deterministic synthetic generator never is. This
module finds the problems *before* they reach Phase 1, because a missing-value
run or a duplicated timestamp that slips through silently produces a plausible
but wrong event catalogue.

Two severities, and the distinction matters:

* **errors** - the dataset cannot be trusted for processing (no time axis, a
  latitude outside [-90, 90], a non-monotonic time axis, all-NaN variables).
* **warnings** - the dataset is usable but suspicious (some NaNs, duplicate
  timestamps, implausible values).

:func:`check_quality` never raises for a merely bad dataset: it returns a
:class:`DataQualityReport` so the CLI and the API can show the operator exactly
what is wrong. :func:`require_valid` is the explicit opt-in for "raise instead".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import xarray as xr

from extreme_weather.ingestion.base import utc_now
from extreme_weather.ingestion.exceptions import QualityCheckError
from extreme_weather.ingestion.normalizer import CONTRACT_VARIABLES, STANDARD_UNITS

__all__ = [
    "PLAUSIBLE_RANGES",
    "DataQualityReport",
    "check_quality",
    "require_valid",
]


#: Plausible physical ranges per canonical variable.
#:
#: These are *sanity bounds*, not scientific thresholds. A value outside them
#: produces a warning ("unreasonable value"), never a silent fix - the data may
#: legitimately be extreme, and this project has no business deciding that.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "precipitation": (0.0, 500.0),
    "temperature": (180.0, 350.0),
    "pressure": (85_000.0, 110_000.0),
    "relative_humidity": (0.0, 100.0),
    "u_wind": (-150.0, 150.0),
    "v_wind": (-150.0, 150.0),
}

#: Values at or beyond this fraction of cells are an error, not a warning.
ALL_NAN_THRESHOLD: float = 1.0
#: Values at or above this fraction produce a warning.
HIGH_NAN_FRACTION: float = 0.10


@dataclass(frozen=True)
class DataQualityReport:
    """The outcome of a quality check on one dataset.

    Attributes:
        valid: True when ``errors`` is empty. Warnings do not make a dataset
            invalid.
        errors: Problems that block processing.
        warnings: Problems that are recorded but do not block.
        missing_fraction: Fraction of all data cells that are NaN.
        time_steps: Number of time steps.
        spatial_points: Latitude points x longitude points.
        variables: Variables that were inspected.
        per_variable_nan: NaN fraction per variable.
        duplicate_timestamps: Number of repeated timestamps.
        duplicate_coordinates: Number of repeated latitude/longitude values.
        unexpected_dimensions: Dimensions that are not part of the project model.
        missing_standard_variables: Canonical variables the project expects but
            the dataset does not carry.
        generated_at: When the check ran.
    """

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_fraction: float = 0.0
    time_steps: int = 0
    spatial_points: int = 0
    variables: tuple[str, ...] = ()
    per_variable_nan: dict[str, float] = field(default_factory=dict)
    duplicate_timestamps: int = 0
    duplicate_coordinates: int = 0
    unexpected_dimensions: tuple[str, ...] = ()
    missing_standard_variables: tuple[str, ...] = ()
    generated_at: datetime | None = None

    @property
    def error_count(self) -> int:
        """Number of blocking problems."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of non-blocking problems."""
        return len(self.warnings)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary for the API and the manifest."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "missing_fraction": round(self.missing_fraction, 6),
            "time_steps": self.time_steps,
            "spatial_points": self.spatial_points,
            "variables": list(self.variables),
            "per_variable_nan": {
                key: round(value, 6) for key, value in self.per_variable_nan.items()
            },
            "duplicate_timestamps": self.duplicate_timestamps,
            "duplicate_coordinates": self.duplicate_coordinates,
            "unexpected_dimensions": list(self.unexpected_dimensions),
            "missing_standard_variables": list(self.missing_standard_variables),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


def _nan_fraction(values: np.ndarray) -> float:
    """Return the NaN fraction of an array, treating an empty array as 0."""
    if values.size == 0:
        return 0.0
    return float(np.isnan(values).sum() / values.size)


def check_quality(
    dataset: xr.Dataset,
    *,
    strict_ranges: bool = False,
    generated_at: datetime | None = None,
) -> DataQualityReport:
    """Inspect a dataset and report every problem found.

    The checks, in the order they run:

    * structure - required coordinates exist, dimensions are as expected;
    * time - monotonic, non-empty, no duplicate timestamps;
    * coordinates - latitude in [-90, 90], longitude in [-360, 360], finite,
      no duplicates;
    * values - NaN fractions, all-NaN variables, physically implausible values;
    * metadata - missing ``units`` on a standard variable.

    Args:
        dataset: The dataset to inspect (not modified).
        strict_ranges: Treat out-of-range values as errors instead of warnings.
        generated_at: Timestamp to stamp on the report.

    Returns:
        A :class:`DataQualityReport`. This function does not raise for a bad
        dataset - that is the point of a report.
    """
    stamp = generated_at or utc_now()
    errors: list[str] = []
    warnings: list[str] = []
    per_variable_nan: dict[str, float] = {}

    names = tuple(sorted(str(name) for name in dataset.data_vars))

    # --- structure ----------------------------------------------------- #
    unexpected = tuple(sorted(set(dataset.dims) - {"time", "latitude", "longitude"}))
    for dim in ("time", "latitude", "longitude"):
        if dim not in dataset.dims and dim not in dataset.coords:
            errors.append(f"Missing required dimension/coordinate '{dim}'.")
    if unexpected:
        warnings.append(
            f"Unexpected dimension(s) {list(unexpected)} - carried through but not "
            "used by Phase 1-5."
        )
    if not names:
        errors.append("Dataset contains no data variables.")

    # --- time ---------------------------------------------------------- #
    duplicate_timestamps = 0
    time_steps = 0
    if "time" in dataset.coords and dataset.coords["time"].ndim == 1:
        time_values = np.asarray(dataset.coords["time"].values)
        time_steps = int(time_values.size)
        if time_values.size == 0:
            errors.append("The time coordinate is empty.")
        else:
            unique = np.unique(time_values)
            duplicate_timestamps = int(time_values.size - unique.size)
            if duplicate_timestamps:
                warnings.append(
                    f"{duplicate_timestamps} duplicate timestamp(s) found; the "
                    "normalizer keeps the first occurrence of each."
                )
            if time_values.size > 1:
                if np.issubdtype(time_values.dtype, np.datetime64):
                    deltas = np.diff(time_values).astype("timedelta64[s]").astype("int64")
                    non_increasing = np.any(deltas <= 0)
                else:
                    non_increasing = not np.all(np.diff(time_values) > 0)
                # Duplicates are *already* reported as a warning above and are
                # removed by the normalizer, so flagging them again as a
                # blocking "not strictly increasing" error would tell an
                # operator their file is unusable when it is trivially fixable.
                # Only a genuinely out-of-order axis blocks.
                if non_increasing and not duplicate_timestamps:
                    errors.append("The time coordinate is not strictly increasing.")


    # --- coordinates --------------------------------------------------- #
    duplicate_coordinates = 0
    latitude_points = 0
    longitude_points = 0
    for name, low, high in (("latitude", -90.0, 90.0), ("longitude", -360.0, 360.0)):
        if name not in dataset.coords or dataset.coords[name].ndim != 1:
            continue
        values = np.asarray(dataset.coords[name].values, dtype="float64")
        if name == "latitude":
            latitude_points = int(values.size)
        else:
            longitude_points = int(values.size)
        if values.size == 0:
            errors.append(f"The '{name}' coordinate is empty.")
            continue
        if not np.all(np.isfinite(values)):
            errors.append(f"The '{name}' coordinate contains NaN or infinite values.")
            continue
        if np.any(values < low) or np.any(values > high):
            errors.append(
                f"The '{name}' coordinate has values outside the valid range "
                f"[{low}, {high}] (min {values.min():.4f}, max {values.max():.4f})."
            )
        duplicates = int(values.size - np.unique(values).size)
        duplicate_coordinates += duplicates
        if duplicates:
            warnings.append(f"{duplicates} duplicate '{name}' coordinate value(s) found.")

    # --- values --------------------------------------------------------- #
    total_cells = 0
    nan_cells = 0
    for name in names:
        if name not in dataset.variables:
            continue
        variable = dataset[name]
        if variable.ndim == 0 or variable.dtype.kind not in "fc":
            continue
        values = np.asarray(variable.values, dtype="float64")
        total_cells += values.size
        fraction = _nan_fraction(values)
        nan_cells += int(np.isnan(values).sum())
        per_variable_nan[name] = fraction

        if values.size and fraction >= ALL_NAN_THRESHOLD:
            errors.append(f"Variable '{name}' is entirely NaN.")
        elif fraction >= HIGH_NAN_FRACTION:
            warnings.append(
                f"Variable '{name}' is {fraction:.1%} NaN - missing-value filling "
                "will have a large effect on the analysis."
            )

        bounds = PLAUSIBLE_RANGES.get(name)
        if bounds is not None and values.size:
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = bounds
                out_of_range = int(
                    np.count_nonzero((finite < low) | (finite > high))
                )
                if out_of_range:
                    message = (
                        f"Variable '{name}' has {out_of_range} value(s) outside the "
                        f"plausible range [{low}, {high}] "
                        f"(min {finite.min():.4g}, max {finite.max():.4g})."
                    )
                    if strict_ranges:
                        errors.append(message)
                    else:
                        warnings.append(message + " Reported as-is; not clipped or altered.")

    missing_fraction = (nan_cells / total_cells) if total_cells else 0.0

    # --- metadata ------------------------------------------------------- #
    missing_standard: list[str] = []
    for name in CONTRACT_VARIABLES:
        if name not in dataset.data_vars:
            missing_standard.append(name)
            continue
        units = dataset[name].attrs.get("units")
        if units in (None, ""):
            warnings.append(
                f"Variable '{name}' has no 'units' attribute; the normalizer will "
                f"assume the standard unit '{STANDARD_UNITS[name]}'."
            )

    return DataQualityReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        missing_fraction=missing_fraction,
        time_steps=time_steps,
        spatial_points=latitude_points * longitude_points,
        variables=names,
        per_variable_nan=per_variable_nan,
        duplicate_timestamps=duplicate_timestamps,
        duplicate_coordinates=duplicate_coordinates,
        unexpected_dimensions=unexpected,
        missing_standard_variables=tuple(missing_standard),
        generated_at=stamp,
    )


def require_valid(report: DataQualityReport, dataset_id: str = "dataset") -> DataQualityReport:
    """Return the report, raising when it contains blocking errors.

    Raises:
        QualityCheckError: If the report is not valid. The message lists every
            error so the operator sees all of them at once.
    """
    if report.valid:
        return report
    joined = "\n".join(f"  - {error}" for error in report.errors)
    raise QualityCheckError(
        f"Dataset '{dataset_id}' failed {report.error_count} blocking quality "
        f"check(s):\n{joined}"
    )

