"""Anomaly scores (SIH26078, Phase 1).

Five scores are provided, all as Xarray-native operations that preserve
coordinates, dimensions and attributes:

A. **Absolute anomaly** (:func:`absolute_anomaly`)
   ``anomaly = value - reference_mean``
   Same unit as the variable. Signed: positive is above the reference mean.

B. **Standardized anomaly** (:func:`standardized_anomaly`)
   ``z = (value - mean) / std``
   Dimensionless. A near-zero or NaN standard deviation yields NaN (configurable
   via ``zero_variance``) because a z-score is mathematically undefined there.

C. **Percentile / empirical extremeness** (:func:`percentile_score`,
   :func:`extremeness_score`)
   The percentile ``p in [levels[0], levels[-1]]`` is estimated by interpolating
   the stored empirical quantile function of the climatology.
   ``extremeness = clip((p - 0.5) / 0.5, 0, 1)`` maps that to a bounded
   ``[0, 1]`` tail score (0 = median, 1 = beyond the upper reference quantile).

D. **EFI-style score** (:func:`efi_score`) - see :data:`EFI_FORMULA` for the
   exact definition and its limits. It is *inspired by* the ECMWF Extreme
   Forecast Index, but this project has no ensemble, no climate-model
   distribution and no calibration: it is a **deterministic prototype score**,
   not a reproduction of an operational EFI product.

E. All of the above are registered in :data:`SCORE_FUNCTIONS` so the service and
   the API can select them by name.

Units and conventions
---------------------
* Absolute anomalies carry the units of the input variable; all other scores are
  dimensionless and set ``units="1"`` (CF "unitless").
* NaN inputs stay NaN, and every score propagates NaN, so a missing cell is
  never reported as extreme.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import xarray as xr
from scipy.special import ndtri

from extreme_weather.anomaly.climatology import (
    GROUP_COORD_NAME,
    POOLED_GROUP_KEY,
    QUANTILE_COORD_NAME,
    ClimatologyError,
    resolve_group_keys,
)
from extreme_weather.data.validator import LATITUDE_DIM, LONGITUDE_DIM, TIME_DIM

__all__ = [
    "DEFAULT_EFI_CLIMATOLOGY_PERCENTILE",
    "DEFAULT_MIN_STD",
    "EFI_FORMULA",
    "EFI_MAX",
    "EFI_MIN",
    "SCORE_FUNCTIONS",
    "ScoreError",
    "ScoreName",
    "absolute_anomaly",
    "align_climatology_statistic",
    "compute_score",
    "efi_score",
    "extremeness_score",
    "percentile_score",
    "quantile_levels_of",
    "standardized_anomaly",
]

#: Default name of a score function.
ScoreName = str

#: Human readable definition of the EFI-style score implemented in this project.
EFI_FORMULA = (
    "EFI = Phi^-1( (p_fc - p_clim) / (1 - p_clim) ), with p_fc the percentile of the "
    "current value inside the reference distribution of the matching climatology "
    "group, p_clim the configured climatological percentile (default 0.9) and "
    "Phi^-1 the standard-normal quantile function. The ratio is clipped to "
    "[0, 1 - 1e-6] and the result to [EFI_MIN, EFI_MAX], so the score is bounded. "
    "Prototype implementation: deterministic (no ensemble), uncalibrated, and NOT "
    "an operational ECMWF EFI."
)

#: Hard bounds of the EFI-style score (standard-normal units).
EFI_MIN: float = -3.0
EFI_MAX: float = 3.0

#: Default climatological percentile of the EFI-style score.
DEFAULT_EFI_CLIMATOLOGY_PERCENTILE: float = 0.9

#: Standard deviations at or below this value count as "no variance".
DEFAULT_MIN_STD: float = 1e-6


class ScoreError(ValueError):
    """Raised when a score cannot be computed with the given inputs."""


# --------------------------------------------------------------------------- #
# Climatology alignment
# --------------------------------------------------------------------------- #
def _climatology_attr(climatology: xr.Dataset, name: str, default: str) -> str:
    return str(climatology.attrs.get(name, default))


def _select_group_index(
    climatology: xr.Dataset, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map every timestamp to a row of the climatology group dimension.

    Returns:
        ``(selector, resolved)`` where ``selector`` indexes the group dimension
        and ``resolved`` marks the steps that have usable reference statistics.
    """
    grouping = _climatology_attr(climatology, "grouping", "all")
    fallback = _climatology_attr(climatology, "fallback", "none")
    min_samples = int(climatology.attrs.get("min_samples", 1) or 1)

    available = np.asarray(climatology[GROUP_COORD_NAME].values, dtype="int64")
    counts = climatology["count"] if "count" in climatology.data_vars else None
    pooled = np.flatnonzero(available == POOLED_GROUP_KEY)

    keys = resolve_group_keys(np.asarray(times), grouping)
    selector = np.zeros(keys.size, dtype="int64")
    resolved = np.zeros(keys.size, dtype=bool)

    for position, key in enumerate(keys):
        matches = np.flatnonzero(available == key)
        if matches.size:
            index = int(matches[0])
            if counts is None or float(counts.isel({GROUP_COORD_NAME: index}).min()) >= min_samples:
                selector[position] = index
                resolved[position] = True
                continue
        if fallback == "pooled" and pooled.size:
            selector[position] = int(pooled[0])
            resolved[position] = True
        else:
            # No usable reference for this step: any row works because the result
            # is masked to NaN below. Use the first *position* (not the group
            # coordinate value) of the climatology.
            selector[position] = 0
    return selector, resolved


def align_climatology_statistic(
    climatology: xr.Dataset,
    times: np.ndarray,
    statistic: str,
) -> xr.DataArray:
    """Return a per-time-step climatology statistic aligned with ``times``.

    Each timestamp is mapped to its climatology group. Groups that are missing or
    that hold fewer than ``min_samples`` values fall back to the pooled group
    when the climatology was built with ``fallback="pooled"``; otherwise the
    result is NaN for those steps, which makes coverage gaps explicit.

    Args:
        climatology: Dataset returned by
            :func:`extreme_weather.anomaly.climatology.build_climatology`.
        times: ``datetime64`` values of the data to be scored.
        statistic: Name of a climatology variable (``mean``, ``std``, ``count``,
            ``min``, ``max`` or ``quantiles``).

    Returns:
        A DataArray with a leading ``time`` dimension in the same order as
        ``times``, plus the spatial dimensions of the climatology.

    Raises:
        ScoreError: If the requested statistic does not exist.
    """
    if statistic not in climatology.data_vars:
        raise ScoreError(
            f"Climatology has no statistic '{statistic}'. Available: "
            f"{sorted(climatology.data_vars)}."
        )

    selector, resolved = _select_group_index(climatology, times)
    aligned = climatology[statistic].isel({GROUP_COORD_NAME: selector})
    # Fancy indexing keeps the (now meaningless) group dimension: drop its index
    # coordinate and rename the dimension to the analysis time axis.
    aligned = aligned.drop_vars(GROUP_COORD_NAME, errors="ignore").rename(
        {GROUP_COORD_NAME: TIME_DIM}
    )
    if not bool(resolved.all()):
        mask = xr.DataArray(resolved, dims=(TIME_DIM,))
        aligned = aligned.where(mask, other=np.nan)
    return aligned.assign_coords({TIME_DIM: np.asarray(times)})


def quantile_levels_of(climatology: xr.Dataset) -> np.ndarray:
    """Return the probability levels stored in ``climatology``."""
    if QUANTILE_COORD_NAME in climatology.coords:
        return np.asarray(climatology[QUANTILE_COORD_NAME].values, dtype="float64")
    levels = str(climatology.attrs.get("quantile_levels", "")).strip()
    if not levels:
        raise ClimatologyError(
            "Climatology carries no quantile levels; rebuild it with build_climatology()."
        )
    return np.asarray([float(level) for level in levels.split(",") if level], dtype="float64")


def _times_of(values: xr.DataArray) -> np.ndarray:
    """Return the time coordinates of a data field."""
    if TIME_DIM not in values.dims:
        raise ScoreError(
            f"Anomaly scoring needs a '{TIME_DIM}' dimension, got dims={values.dims}."
        )
    return np.asarray(values[TIME_DIM].values)


def _score_attrs(values: xr.DataArray, units: str, long_name: str, formula: str) -> dict[str, str]:
    """Build consistent attributes for a score field."""
    attrs = {
        "units": units,
        "long_name": long_name,
        "formula": formula,
        "source": "synthetic prototype (SIH26078 Phase 1)",
    }
    return attrs


def _align_reference(reference: xr.DataArray, values: xr.DataArray) -> xr.DataArray:
    """Give a climatology statistic the coordinates of the field being scored.

    The aligned statistic already carries the analysis time axis in the same
    order as ``values``; only the spatial coordinates are copied so that Xarray
    can combine both arrays positionally (and still complain about mismatched
    grid sizes instead of silently broadcasting).

    Raises:
        ScoreError: If the reference grid does not match the scored field.
    """
    coordinates: dict[str, xr.DataArray] = {}
    for dim in (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM):
        if dim in values.dims:
            if dim not in reference.dims:
                raise ScoreError(
                    f"Climatology statistic is missing the '{dim}' dimension "
                    f"(has {reference.dims})."
                )
            if reference.sizes[dim] != values.sizes[dim]:
                raise ScoreError(
                    f"Climatology and data disagree on the size of '{dim}': "
                    f"{reference.sizes[dim]} vs {values.sizes[dim]}. The reference must be "
                    "built on the same grid as the analysed dataset."
                )
            coordinates[dim] = values[dim]
    return reference.assign_coords(coordinates)


# --------------------------------------------------------------------------- #
# A. Absolute anomaly
# --------------------------------------------------------------------------- #
def absolute_anomaly(
    values: xr.DataArray,
    climatology: xr.Dataset,
    *,
    name: str = "absolute_anomaly",
) -> xr.DataArray:
    """Return ``value - reference_mean`` in the units of ``values``.

    Args:
        values: Data field with a ``time`` dimension.
        climatology: Reference distribution from
            :func:`~extreme_weather.anomaly.climatology.build_climatology`.
        name: Name of the resulting DataArray.

    Returns:
        Signed anomaly; NaN where either side is NaN (typically because the
        climatology does not cover that group).
    """
    reference = align_climatology_statistic(climatology, _times_of(values), "mean")
    anomaly = values - _align_reference(reference, values)
    anomaly.name = name
    anomaly.attrs = _score_attrs(
        values,
        str(values.attrs.get("units", "")),
        f"absolute anomaly of {values.name or 'field'}",
        "value - climatology_mean",
    )
    return anomaly


# --------------------------------------------------------------------------- #
# B. Standardized anomaly
# --------------------------------------------------------------------------- #
def standardized_anomaly(
    values: xr.DataArray,
    climatology: xr.Dataset,
    *,
    min_std: float = DEFAULT_MIN_STD,
    zero_variance: str = "nan",
    name: str = "standardized_anomaly",
) -> xr.DataArray:
    """Return ``z = (value - mean) / std``.

    Args:
        values: Data field with a ``time`` dimension.
        climatology: Reference distribution.
        min_std: Standard deviations at or below this value are treated as
            "no variance".
        zero_variance: ``"nan"`` (default) returns NaN where the reference has no
            spread, ``"zero"`` returns 0.0.
        name: Name of the resulting DataArray.

    Returns:
        Dimensionless z-score; NaN where inputs or the reference are undefined.

    Raises:
        ScoreError: If ``zero_variance`` is not ``"nan"`` or ``"zero"``.
    """
    if str(zero_variance).lower() not in {"nan", "zero"}:
        raise ScoreError(
            f"zero_variance must be 'nan' or 'zero', got '{zero_variance}'."
        )
    if float(min_std) < 0.0:
        raise ScoreError(f"min_std must be >= 0, got {min_std}.")

    times = _times_of(values)
    mean = _align_reference(
        align_climatology_statistic(climatology, times, "mean"), values
    )
    std = _align_reference(
        align_climatology_statistic(climatology, times, "std"), values
    )

    usable = std > float(min_std)
    safe_std = xr.where(usable, std, np.nan)
    zscore = (values - mean) / safe_std
    if str(zero_variance).lower() == "zero":
        zscore = zscore.fillna(0.0).where(values.notnull())

    zscore.name = name
    zscore.attrs = _score_attrs(
        values,
        "1",
        f"standardized anomaly of {values.name or 'field'}",
        "(value - climatology_mean) / climatology_std",
    )
    zscore.attrs["min_std"] = float(min_std)
    return zscore


# --------------------------------------------------------------------------- #
# C. Percentile and extremeness
# --------------------------------------------------------------------------- #
def _percentile_from_quantiles(
    quantile_values: np.ndarray, levels: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Estimate the percentile of ``values`` from a stored quantile function.

    Fully vectorised: for every value the number of stored quantiles below it is
    found with ``searchsorted``-style counting, then the probability is linearly
    interpolated between the two bracketing levels.

    Args:
        quantile_values: ``(..., Q)`` array of quantile values (ascending in Q).
        levels: ``(Q,)`` ascending probability levels.
        values: Array broadcastable to ``quantile_values.shape[:-1]``.

    Returns:
        Percentiles in ``[levels[0], levels[-1]]``.
    """
    curves = np.moveaxis(np.asarray(quantile_values, dtype="float64"), -1, 0)
    broadcast = np.broadcast_to(np.asarray(values, dtype="float64"), curves.shape[1:])
    below = np.sum(curves <= broadcast[None, ...], axis=0)

    lower = np.clip(below - 1, 0, levels.size - 1)
    upper = np.clip(below, 0, levels.size - 1)
    lower_values = np.take_along_axis(curves, lower[None, ...], axis=0)[0]
    upper_values = np.take_along_axis(curves, upper[None, ...], axis=0)[0]

    span = upper_values - lower_values
    fraction = np.divide(
        broadcast - lower_values,
        span,
        out=np.zeros_like(broadcast),
        where=span > 0.0,
    )
    percentile = levels[lower] + fraction * (levels[upper] - levels[lower])
    percentile = np.clip(percentile, float(levels[0]), float(levels[-1]))
    # NaN in, NaN out: a missing value must never look like the coldest one.
    return np.where(np.isfinite(broadcast), percentile, np.nan)


def percentile_score(
    values: xr.DataArray,
    climatology: xr.Dataset,
    *,
    name: str = "percentile",
) -> xr.DataArray:
    """Return the reference percentile of each value in ``[q_min, q_max]``.

    The percentile is estimated by interpolating the climatology's empirical
    quantile function, so it saturates at the extreme stored levels (by default
    0.01 and 0.99). NaN in, NaN out.

    Args:
        values: Data field with a ``time`` dimension.
        climatology: Reference distribution holding a ``quantiles`` variable.
        name: Name of the resulting DataArray.

    Raises:
        ScoreError: If the climatology has no ``quantiles`` variable.
    """
    if "quantiles" not in climatology.data_vars:
        raise ScoreError(
            "Climatology has no 'quantiles' variable; rebuild it with build_climatology()."
        )

    times = _times_of(values)
    curves = align_climatology_statistic(climatology, times, "quantiles")
    levels = quantile_levels_of(climatology)

    # Move the quantile dimension last, matching _percentile_from_quantiles.
    aligned = _align_reference(curves, values).transpose(..., QUANTILE_COORD_NAME)
    percentile_values = _percentile_from_quantiles(aligned.values, levels, values.values)
    percentile = xr.DataArray(
        percentile_values, dims=values.dims, coords=values.coords, name=name
    )
    percentile.attrs = _score_attrs(
        values,
        "1",
        f"reference percentile of {values.name or 'field'}",
        "linear interpolation of the climatology quantile function",
    )
    percentile.attrs["percentile_min"] = float(levels[0])
    percentile.attrs["percentile_max"] = float(levels[-1])
    return percentile


def extremeness_score(
    values: xr.DataArray,
    climatology: xr.Dataset,
    *,
    name: str = "extremeness",
) -> xr.DataArray:
    """Return a bounded tail score in ``[0, 1]`` derived from the percentile.

    ``extremeness = clip((percentile - 0.5) / 0.5, 0, 1)``: 0 for values at or
    below the reference median, 1 for values beyond the highest stored
    reference quantile.
    """
    percentile = percentile_score(values, climatology)
    extremeness = ((percentile - 0.5) / 0.5).clip(0.0, 1.0)
    extremeness.name = name
    extremeness.attrs = _score_attrs(
        values,
        "1",
        f"reference extremeness of {values.name or 'field'}",
        "clip((percentile - 0.5) / 0.5, 0, 1)",
    )
    return extremeness


# --------------------------------------------------------------------------- #
# D. EFI-style score
# --------------------------------------------------------------------------- #
def efi_score(
    values: xr.DataArray,
    climatology: xr.Dataset,
    *,
    climatology_percentile: float = DEFAULT_EFI_CLIMATOLOGY_PERCENTILE,
    bounds: tuple[float, float] = (EFI_MIN, EFI_MAX),
    name: str = "efi_score",
) -> xr.DataArray:
    """Return the bounded EFI-style score described in :data:`EFI_FORMULA`.

    ``EFI = Phi^-1( (p_fc - p_clim) / (1 - p_clim) )`` with ``p_fc`` the
    reference percentile of the value and ``p_clim`` the climatological
    percentile (default 0.9). The ratio is clipped to ``[0, 1 - 1e-6]`` and the
    result to ``bounds`` (default ``[-3, 3]``), so the score is always finite.

    Interpretation of the default scale:

    =====================  ==========================================
    score                  meaning
    =====================  ==========================================
    ``<= -3``              at or below the climatological percentile
                            (the ratio is clipped at 0, so the score
                            saturates at the lower bound there)
    ``0``                  a value just above ``climatology_percentile``
    ``+1``                 about the 84th percentile of the upper tail
    ``+2``                 about the 97.7th percentile of the upper tail
    ``>= +3``              beyond the stored reference quantiles
    =====================  ==========================================

    Args:
        values: Data field with a ``time`` dimension.
        climatology: Reference distribution.
        climatology_percentile: ``p_clim`` in ``(0, 1)``.
        bounds: Output bounds of the score.
        name: Name of the resulting DataArray.

    Raises:
        ScoreError: If ``climatology_percentile`` is outside ``(0, 1)``.
    """
    if not 0.0 < float(climatology_percentile) < 1.0:
        raise ScoreError(
            f"climatology_percentile must lie inside (0, 1), got {climatology_percentile}."
        )
    lower_bound, upper_bound = (float(bounds[0]), float(bounds[1]))
    if lower_bound >= upper_bound:
        raise ScoreError(f"bounds must be increasing, got {bounds}.")

    percentile = percentile_score(values, climatology, name="_efi_percentile")
    ratio = (percentile - float(climatology_percentile)) / (
        1.0 - float(climatology_percentile)
    )
    ratio = ratio.clip(0.0, 1.0 - 1e-6)
    score = xr.DataArray(
        np.clip(ndtri(ratio.values), lower_bound, upper_bound),
        dims=percentile.dims,
        coords=percentile.coords,
        name=name,
    )
    score.attrs = _score_attrs(
        values,
        "1",
        f"EFI-style score of {values.name or 'field'} (prototype, not operational EFI)",
        EFI_FORMULA,
    )
    score.attrs["climatology_percentile"] = float(climatology_percentile)
    score.attrs["score_min"] = lower_bound
    score.attrs["score_max"] = upper_bound
    return score


# --------------------------------------------------------------------------- #
# E. Registry
# --------------------------------------------------------------------------- #
#: Score name -> implementation. Every function takes ``(values, climatology)``.
SCORE_FUNCTIONS: Mapping[str, Callable[..., xr.DataArray]] = {
    "absolute": absolute_anomaly,
    "standardized": standardized_anomaly,
    "percentile": percentile_score,
    "extremeness": extremeness_score,
    "efi": efi_score,
}


def compute_score(
    name: str,
    values: xr.DataArray,
    climatology: xr.Dataset,
    **kwargs: object,
) -> xr.DataArray:
    """Compute a registered score by name.

    Args:
        name: One of the keys of :data:`SCORE_FUNCTIONS`.
        values: Data field with a ``time`` dimension.
        climatology: Reference distribution.
        **kwargs: Forwarded to the score implementation.

    Raises:
        ScoreError: If the score name is unknown.
    """
    key = str(name).strip().lower()
    if key not in SCORE_FUNCTIONS:
        raise ScoreError(
            f"Unknown anomaly score '{name}'. Available scores: "
            f"{', '.join(sorted(SCORE_FUNCTIONS))}."
        )
    return SCORE_FUNCTIONS[key](values, climatology, **kwargs)
