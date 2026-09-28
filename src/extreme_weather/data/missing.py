"""Missing-value handling with explicit ``max_gap`` semantics (SIH26078).

Design decisions (Phase 0)
--------------------------
* Only *interior* gaps are filled. A gap is interior when it is bracketed by a
  valid value on **both** sides. Leading and trailing gaps have no bracketing
  observation, so they are intentionally left missing instead of being
  extrapolated - extrapolation would silently invent data at the series edges.
* A run of ``L`` consecutive missing values is filled **only** when
  ``L <= max_gap``. Longer runs stay missing, so the pipeline never hides a long
  outage behind a smooth interpolation.
* Filling uses linear interpolation in index space (time step space), computed in
  float64 for numerical stability and cast back to the input dtype.
* ``max_gap=0`` fills nothing; ``max_gap=None`` fills all interior gaps.
* All-missing arrays are returned unchanged (no exception), and they raise no
  crash in downstream statistics.

Because the interpolation is explicit NumPy code, no accelerated backend is
required: neither ``bottleneck`` nor ``bottleneck``-style optional dependencies
are used by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import xarray as xr

from extreme_weather.data.validator import TIME_DIM

__all__ = [
    "DEFAULT_MAX_GAP",
    "MissingGap",
    "MissingValueError",
    "count_missing",
    "fill_missing",
    "fill_missing_dataset",
    "find_missing_gaps",
    "missing_fraction",
]

#: Default maximum number of consecutive missing steps that may be filled.
DEFAULT_MAX_GAP: int = 3

INTERIOR: str = "interior"
LEADING: str = "leading"
TRAILING: str = "trailing"
ALL_MISSING: str = "all_missing"


class MissingValueError(ValueError):
    """Raised when missing-value handling is requested with invalid arguments."""


@dataclass(frozen=True)
class MissingGap:
    """A run of consecutive missing values inside a 1-D series."""

    start: int
    end: int
    kind: str

    @property
    def length(self) -> int:
        """Number of consecutive missing values in the run."""
        return self.end - self.start + 1


def find_missing_gaps(values: np.ndarray) -> list[MissingGap]:
    """Return every consecutive run of missing values in a 1-D array.

    Gaps are classified as ``leading`` (before the first valid value),
    ``trailing`` (after the last valid value) or ``interior``. Bracket an
    interior gap and you have the two neighbours used for interpolation.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise MissingValueError(f"find_missing_gaps expects a 1-D array, got {array.ndim}-D.")

    missing = ~np.isfinite(array)
    if not missing.any():
        return []

    valid = ~missing
    if not valid.any():
        return [MissingGap(start=0, end=array.size - 1, kind=ALL_MISSING)]

    first_valid = int(np.argmax(valid))
    last_valid = int(array.size - 1 - np.argmax(valid[::-1]))

    gaps: list[MissingGap] = []
    index = 0
    while index < array.size:
        if missing[index]:
            start = index
            while index < array.size and missing[index]:
                index += 1
            end = index - 1
            if end < first_valid:
                kind = LEADING
            elif start > last_valid:
                kind = TRAILING
            else:
                kind = INTERIOR
            gaps.append(MissingGap(start=start, end=end, kind=kind))
        else:
            index += 1
    return gaps


def _validate_max_gap(max_gap: int | None) -> int | None:
    """Validate the ``max_gap`` argument and return it normalised."""
    if max_gap is None:
        return None
    if isinstance(max_gap, bool) or not isinstance(max_gap, Integral):
        raise MissingValueError(
            f"max_gap must be an integer number of steps or None, got {type(max_gap).__name__}."
        )
    if int(max_gap) < 0:
        raise MissingValueError(f"max_gap must be >= 0 or None, got {max_gap}.")
    return int(max_gap)


def _fill_row(row: np.ndarray, max_gap: int | None) -> np.ndarray:
    """Fill interior gaps of a single 1-D float row (see module docstring)."""
    filled = np.array(row, dtype="float64", copy=True)
    valid = np.isfinite(filled)
    if not valid.any() or valid.all():
        return filled

    first_valid = int(np.argmax(valid))
    last_valid = int(filled.size - 1 - np.argmax(valid[::-1]))
    if last_valid - first_valid < 2:
        return filled

    valid_index = np.flatnonzero(valid)
    for gap in find_missing_gaps(filled):
        if gap.kind != INTERIOR:
            continue
        if max_gap is not None and gap.length > max_gap:
            continue
        targets = np.arange(gap.start, gap.end + 1, dtype="float64")
        filled[gap.start : gap.end + 1] = np.interp(targets, valid_index, filled[valid_index])
    return filled


def fill_missing(
    data: xr.DataArray,
    max_gap: int | None = DEFAULT_MAX_GAP,
    *,
    dim: str = TIME_DIM,
) -> xr.DataArray:
    """Fill short interior gaps of ``data`` along ``dim``, honouring ``max_gap``.

    Args:
        data: Floating point :class:`xarray.DataArray`.
        max_gap: Maximum number of consecutive missing steps that may be filled.
            Use ``None`` to fill every interior gap and ``0`` to fill nothing.
        dim: Dimension along which gaps are interpolated (default ``"time"``).

    Returns:
        A **new** DataArray with the same dims/coords/attrs as ``data``. Interior
        gaps longer than ``max_gap`` and all leading/trailing gaps stay missing.

    Raises:
        MissingValueError: If ``data`` is not a float DataArray, ``dim`` is
            absent, or ``max_gap`` is invalid.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"fill_missing expects an xarray.DataArray, got {type(data).__name__}.")
    if dim not in data.dims:
        raise MissingValueError(
            f"Dimension '{dim}' not found in DataArray '{data.name}'; available dims: {data.dims}."
        )
    if not np.issubdtype(data.dtype, np.floating):
        raise MissingValueError(
            f"fill_missing requires floating point data, got dtype '{data.dtype}' "
            f"for DataArray '{data.name}'."
        )

    validated_gap = _validate_max_gap(max_gap)
    other_dims = [name for name in data.dims if name != dim]
    transposed = data.transpose(*other_dims, dim)

    values = np.asarray(transposed.values, dtype="float64")
    flat = values.reshape(-1, values.shape[-1])
    output = np.empty_like(flat)
    for row_index in range(flat.shape[0]):
        output[row_index] = _fill_row(flat[row_index], validated_gap)

    result = transposed.copy(deep=True)
    result.values = output.reshape(values.shape).astype(data.dtype, copy=False)
    result = result.transpose(*data.dims)
    result.attrs = dict(data.attrs)
    result.attrs["missing_value_handling"] = (
        "linear interpolation of interior gaps with max_gap="
        + ("unlimited" if validated_gap is None else str(validated_gap))
        + f" along '{dim}'; leading/trailing gaps preserved"
    )
    return result


def fill_missing_dataset(
    dataset: xr.Dataset,
    max_gap: int | None = DEFAULT_MAX_GAP,
    *,
    dim: str = TIME_DIM,
    variables: tuple[str, ...] | None = None,
) -> xr.Dataset:
    """Apply :func:`fill_missing` to every suitable variable of ``dataset``.

    Variables that are non-floating, or that do not contain ``dim``, are copied
    unchanged. The input dataset is never mutated.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"fill_missing_dataset expects an xarray.Dataset, got {type(dataset).__name__}."
        )
    validated_gap = _validate_max_gap(max_gap)

    result = dataset.copy(deep=True)
    for name, variable in dataset.data_vars.items():
        if variables is not None and name not in variables:
            continue
        if dim not in variable.dims or not np.issubdtype(variable.dtype, np.floating):
            continue
        result[name] = fill_missing(variable, validated_gap, dim=dim)
    return result


def count_missing(data: xr.DataArray | xr.Dataset) -> int | dict[str, int]:
    """Count missing (NaN) values.

    Returns an ``int`` for a DataArray and a ``{variable: count}`` mapping for a
    Dataset.
    """
    if isinstance(data, xr.DataArray):
        values = np.asarray(data.values)
        if not np.issubdtype(values.dtype, np.floating):
            return 0
        return int(np.count_nonzero(np.isnan(values)))
    if isinstance(data, xr.Dataset):
        return {name: int(count_missing(variable)) for name, variable in data.data_vars.items()}
    raise TypeError(f"count_missing expects a DataArray or Dataset, got {type(data).__name__}.")


def missing_fraction(data: xr.DataArray) -> float:
    """Return the fraction of missing values in ``data`` (0.0 when empty)."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"missing_fraction expects an xarray.DataArray, got {type(data).__name__}.")
    if data.size == 0:
        return 0.0
    return float(count_missing(data)) / float(data.size)
