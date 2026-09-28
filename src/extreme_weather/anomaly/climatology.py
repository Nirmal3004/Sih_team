"""Climatology / reference distribution builder (SIH26078, Phase 1).

**Prototype reference distribution - not real-world climatology.**

The climatology produced here is a *reference distribution* computed from the
dataset handed to :func:`build_climatology`. In the Phase 0/1 demo that dataset
is **synthetic**, so the resulting statistics describe a synthetic sample, not
the Indian monsoon climatology. The module never claims operational validity: it
provides the statistics (mean, standard deviation, sample count and an empirical
quantile function) that the anomaly scores in :mod:`extreme_weather.anomaly.scores`
need in order to judge how unusual a value is.

Grouping
--------
``grouping="month"``
    Statistics per calendar month (the leading month of the year).
``grouping="day_of_year"``
    Statistics per day of year (1-366), a finer but data-hungry grouping.
``grouping="all"``
    Pooled statistics over the whole reference period.

Short reference periods
-----------------------
A real climatology needs decades of data; a short sample can leave a requested
group with too few values (or none at all). ``fallback="pooled"`` (the default)
therefore lets such groups use the pooled ``"all"`` statistics, and
``min_samples`` controls how many values a group needs before the fallback is
applied. ``fallback="none"`` returns NaN for uncovered groups instead, which is
useful to make coverage gaps explicit.

Storage
-------
Per group and grid cell the climatology stores ``mean``, ``std``, ``count``,
``min``, ``max`` and an empirical quantile function sampled on
:data:`DEFAULT_QUANTILE_LEVELS`. Percentiles of new values are estimated by
interpolating that quantile function - a memory-light approximation that keeps
the reference small for large datasets. Use ``quantile_levels`` to tune the
resolution.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from extreme_weather.data.validator import LATITUDE_DIM, LONGITUDE_DIM, TIME_DIM

__all__ = [
    "ClimatologyError",
    "ClimatologyGrouping",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_QUANTILE_LEVELS",
    "GROUP_COORD_NAME",
    "QUANTILE_COORD_NAME",
    "build_climatology",
    "resolve_group_keys",
]

#: Name of the dimension carrying the climatology group key.
GROUP_COORD_NAME: str = "group"
#: Name of the dimension carrying the quantile probability levels.
QUANTILE_COORD_NAME: str = "quantile"

#: Probability levels of the stored empirical quantile function.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = tuple(
    round(float(level), 4) for level in np.arange(0.01, 1.0, 0.01)
)

#: Minimum number of reference values a group needs before pooling kicks in.
DEFAULT_MIN_SAMPLES: int = 10

#: Reference dimensions kept for the spatial grid (everything else is reduced).
_SPATIAL_DIMS: tuple[str, ...] = (LATITUDE_DIM, LONGITUDE_DIM)


class ClimatologyError(ValueError):
    """Raised when a climatology cannot be built or used as requested."""


class ClimatologyGrouping(StrEnum):
    """Supported climatology groupings."""

    MONTH = "month"
    DAY_OF_YEAR = "day_of_year"
    ALL = "all"

    @classmethod
    def from_value(cls, value: str) -> "ClimatologyGrouping":
        """Parse a grouping from a case-insensitive string.

        Raises:
            ValueError: If the grouping is unknown.
        """
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown climatology grouping '{value}'. Supported groupings: {supported}."
            ) from exc


# --------------------------------------------------------------------------- #
# Group helpers
# --------------------------------------------------------------------------- #
def resolve_group_keys(
    times: np.ndarray, grouping: ClimatologyGrouping | str
) -> np.ndarray:
    """Return the integer group key of every timestamp.

    Args:
        times: Array of ``datetime64`` values.
        grouping: One of :class:`ClimatologyGrouping`.

    Returns:
        Integer array of the same length, suitable for ``groupby``.

    Raises:
        ClimatologyError: If ``times`` is not a datetime array.
    """
    resolved = ClimatologyGrouping.from_value(grouping)
    values = np.asarray(times)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ClimatologyError(
            f"Climatology requires a datetime64 time axis, got dtype '{values.dtype}'."
        )

    if resolved is ClimatologyGrouping.MONTH:
        # 1..12 so that month numbers read naturally on the group coordinate.
        return (values.astype("datetime64[M]").astype("int64") % 12) + 1
    if resolved is ClimatologyGrouping.DAY_OF_YEAR:
        day_of_year = (
            values.astype("datetime64[D]") - values.astype("datetime64[Y]").astype("datetime64[D]")
        )
        return (day_of_year / np.timedelta64(1, "D")).astype("int64") + 1
    return np.zeros(values.size, dtype="int64")


def _validate_quantile_levels(levels: Sequence[float]) -> np.ndarray:
    """Validate and return the quantile probability levels."""
    array = np.asarray(levels, dtype="float64")
    if array.ndim != 1 or array.size == 0:
        raise ClimatologyError("quantile_levels must be a non-empty 1-D sequence.")
    if np.any(array <= 0.0) or np.any(array >= 1.0):
        raise ClimatologyError(
            "quantile_levels must lie strictly inside (0, 1); got "
            f"min={array.min()}, max={array.max()}."
        )
    if np.any(np.diff(array) <= 0.0):
        raise ClimatologyError("quantile_levels must be strictly increasing.")
    return array


def _require_variable(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    """Return ``dataset[variable]`` with a clear error when it is missing."""
    if not isinstance(dataset, xr.Dataset):
        raise ClimatologyError(
            f"build_climatology expects an xarray.Dataset, got {type(dataset).__name__}."
        )
    if variable not in dataset.data_vars:
        raise ClimatologyError(
            f"Variable '{variable}' is not part of the dataset. Available: "
            f"{sorted(dataset.data_vars)}."
        )
    field = dataset[variable]
    if TIME_DIM not in field.dims:
        raise ClimatologyError(
            f"Variable '{variable}' has no '{TIME_DIM}' dimension (dims={field.dims}); "
            "a climatology needs a time series."
        )
    return field


#: Group key used for the pooled ("all samples") statistics.
POOLED_GROUP_KEY: int = -1


def _reduce_field(field: xr.DataArray, levels: np.ndarray) -> xr.Dataset:
    """Reduce one time slice of data into a statistics dataset.

    The result contains ``mean``, ``std``, ``count``, ``min``, ``max`` and the
    empirical quantile function ``quantiles(quantile, latitude, longitude)``.
    NaN values are skipped (``skipna=True``), so ``count`` doubles as a data
    availability mask.
    """
    statistics = xr.Dataset(
        {
            "mean": field.mean(dim=TIME_DIM, skipna=True),
            "std": field.std(dim=TIME_DIM, skipna=True),
            "count": field.count(dim=TIME_DIM),
            "min": field.min(dim=TIME_DIM, skipna=True),
            "max": field.max(dim=TIME_DIM, skipna=True),
        }
    )
    quantiles = field.quantile(q=levels.tolist(), dim=TIME_DIM, skipna=True)
    statistics["quantiles"] = quantiles
    return statistics


def build_climatology(
    dataset: xr.Dataset,
    variable: str,
    grouping: ClimatologyGrouping | str = ClimatologyGrouping.MONTH,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    fallback: str = "pooled",
) -> xr.Dataset:
    """Build a reference (climatology) dataset for ``variable``.

    Args:
        dataset: Source dataset with a ``time`` coordinate and the spatial grid.
        variable: Variable name, for example ``"precipitation"`` or
            ``"wind_speed"``.
        grouping: ``"month"``, ``"day_of_year"`` or ``"all"``.
        quantile_levels: Probability levels of the stored quantile function.
        min_samples: Values a group needs before the fallback applies.
        fallback: ``"pooled"`` (default) lets under-sampled or missing groups use
            the pooled statistics; ``"none"`` leaves them NaN.

    Returns:
        An :class:`xarray.Dataset` with variables ``mean``, ``std``, ``count``,
        ``min``, ``max`` and ``quantiles`` and dimensions ``group`` (plus
        ``quantile`` for ``quantiles`` and the spatial dimensions). Global
        attributes record the variable, grouping, fallback policy, quantile
        levels and the synthetic-data disclaimer.

    Raises:
        ClimatologyError: If the variable is missing, the grouping is unknown or
            the quantile levels are invalid.
    """
    resolved_grouping = ClimatologyGrouping.from_value(grouping)
    levels = _validate_quantile_levels(quantile_levels)
    if int(min_samples) < 1:
        raise ClimatologyError(f"min_samples must be >= 1, got {min_samples}.")
    fallback = str(fallback).strip().lower()
    if fallback not in {"pooled", "none"}:
        raise ClimatologyError(
            f"fallback must be 'pooled' or 'none', got '{fallback}'."
        )

    field = _require_variable(dataset, variable)
    times = np.asarray(dataset[TIME_DIM].values)
    keys = resolve_group_keys(times, resolved_grouping)

    parts: list[xr.Dataset] = []
    for key in np.unique(keys):
        selection = field.isel({TIME_DIM: np.flatnonzero(keys == key)})
        statistics = _reduce_field(selection, levels)
        statistics = statistics.expand_dims({GROUP_COORD_NAME: [int(key)]})
        parts.append(statistics)

    if resolved_grouping is not ClimatologyGrouping.ALL and fallback == "pooled":
        pooled = _reduce_field(field, levels).expand_dims(
            {GROUP_COORD_NAME: [POOLED_GROUP_KEY]}
        )
        parts.append(pooled)

    climatology = (
        parts[0] if len(parts) == 1 else xr.concat(parts, dim=GROUP_COORD_NAME)
    )
    climatology = climatology.transpose(GROUP_COORD_NAME, ..., *_SPATIAL_DIMS).sortby(
        GROUP_COORD_NAME
    )

    climatology.attrs.update(
        {
            "variable": variable,
            "grouping": resolved_grouping.value,
            "group_coordinate": GROUP_COORD_NAME,
            "pooled_group_key": POOLED_GROUP_KEY if fallback == "pooled" else "",
            "fallback": fallback,
            "min_samples": int(min_samples),
            "quantile_levels": ",".join(f"{level:g}" for level in levels),
            "sample_count": int(times.size),
            "source": "synthetic (Phase 0/1 prototype reference distribution)",
            "long_name": (
                f"Prototype reference distribution for '{variable}' "
                f"({resolved_grouping.value} grouping)"
            ),
            "comment": (
                "Prototype climatology computed from a synthetic sample. It is NOT a real-world "
                "climatology and must not be used for operational decisions."
            ),
        }
    )
    return climatology
