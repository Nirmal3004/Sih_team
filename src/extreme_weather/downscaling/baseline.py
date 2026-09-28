"""Baseline (non-AI) spatial downscaling methods.

Phase 0 implements only classical interpolation, which establishes the interface
and the validation surface that the FUTURE AI downscaler will reuse:

* :func:`nearest_downscale` - nearest-neighbour interpolation (fast, no new
  information, useful as a reference/"do not smooth" baseline).
* :func:`cubic_downscale` - cubic spline interpolation (smoother fields, needs at
  least 4 source points per interpolated dimension and SciPy, which is already a
  project dependency).

Both operate on a single :class:`xarray.DataArray` and rebuild the target grid so
that it spans exactly the source domain at the requested resolution.

FUTURE (Phase 4): a diffusion/CNN super-resolution model for the reference
12 km -> 5 km task. It will register itself through the same registry, e.g.
``register_method("diffusion_sr", ...)``, and be selected with
``downscale(da, 0.05, method="diffusion_sr")``.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import xarray as xr

from extreme_weather.downscaling.registry import register_method

DEFAULT_LATITUDE_DIM: str = "latitude"
DEFAULT_LONGITUDE_DIM: str = "longitude"

#: Minimum number of source points required by cubic interpolation.
CUBIC_MIN_POINTS: int = 4

__all__ = [
    "CUBIC_MIN_POINTS",
    "DEFAULT_LATITUDE_DIM",
    "DEFAULT_LONGITUDE_DIM",
    "DownscalingError",
    "cubic_downscale",
    "nearest_downscale",
    "target_coordinates",
]


class DownscalingError(ValueError):
    """Raised when a downscaling request is invalid or unsupported.

    This is the single error type of the package: the Phase 0 baselines, the
    Phase 4 model and the Phase 4 inference path all raise it, so a caller can
    catch one class regardless of which stage failed.
    """


def build_target_coordinate(source: np.ndarray, target_resolution: float) -> np.ndarray:
    """Return a uniform target axis covering ``source`` at ``target_resolution``.

    The first and last target points coincide exactly with the source bounds so
    that no extrapolation is required during interpolation. When the source span
    is not an exact multiple of ``target_resolution`` the final interval is
    slightly shorter.
    """
    values = np.asarray(source, dtype="float64")
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    intervals = max(int(round(span / target_resolution)), 1)
    target = lower + target_resolution * np.arange(intervals + 1, dtype="float64")
    target[-1] = upper
    if target.size < 2 or not np.all(np.diff(target) > 0.0):
        raise DownscalingError(
            f"Could not build a strictly increasing target axis spanning [{lower}, {upper}] "
            f"at {target_resolution} deg."
        )
    return target


def _validate_resolution(target_resolution: float | Real) -> float:
    """Validate the requested target resolution in degrees."""
    if isinstance(target_resolution, bool) or not isinstance(target_resolution, Real):
        raise DownscalingError(
            "target_resolution must be a number of degrees, got "
            f"{type(target_resolution).__name__}."
        )
    resolution = float(target_resolution)
    if not np.isfinite(resolution):
        raise DownscalingError(f"target_resolution must be finite, got {target_resolution}.")
    if resolution <= 0.0:
        raise DownscalingError(f"target_resolution must be > 0 degrees, got {resolution}.")
    return resolution


def target_coordinates(
    data: xr.DataArray,
    target_resolution: float,
    *,
    latitude_dim: str = DEFAULT_LATITUDE_DIM,
    longitude_dim: str = DEFAULT_LONGITUDE_DIM,
) -> dict[str, np.ndarray]:
    """Return the ``{latitude_dim: ..., longitude_dim: ...}`` target coordinate map.

    Raises:
        DownscalingError: If the input is not a DataArray, lacks the spatial
            dimensions, or the requested resolution is coarser than the domain.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Downscaling expects an xarray.DataArray, got {type(data).__name__}. "
            "Select a single variable first (for example ds['precipitation'])."
        )
    for dim in (latitude_dim, longitude_dim):
        if dim not in data.dims:
            raise DownscalingError(
                f"Dimension '{dim}' is not present in DataArray '{data.name}'; "
                f"available dims: {data.dims}."
            )
        if data.sizes[dim] < 2:
            raise DownscalingError(
                f"Dimension '{dim}' needs at least 2 points for downscaling, "
                f"found {data.sizes[dim]}."
            )

    resolution = _validate_resolution(target_resolution)
    latitude = np.asarray(data[latitude_dim].values, dtype="float64")
    longitude = np.asarray(data[longitude_dim].values, dtype="float64")

    for name, values in ((latitude_dim, latitude), (longitude_dim, longitude)):
        span = float(values.max() - values.min())
        if resolution > span:
            raise DownscalingError(
                f"target_resolution {resolution} deg is coarser than the {name} domain span "
                f"{span:.4f} deg; pick a finer resolution than the source grid."
            )

    return {
        latitude_dim: build_target_coordinate(latitude, resolution),
        longitude_dim: build_target_coordinate(longitude, resolution),
    }


@register_method(
    "nearest",
    description="Nearest-neighbour resampling to the target grid (reference baseline).",
    minimum_points_per_dim=2,
)
def nearest_downscale(
    data: xr.DataArray,
    target_resolution: float,
    *,
    latitude_dim: str = DEFAULT_LATITUDE_DIM,
    longitude_dim: str = DEFAULT_LONGITUDE_DIM,
) -> xr.DataArray:
    """Resample ``data`` to ``target_resolution`` using nearest neighbours.

    The returned DataArray keeps the variable name, attributes and all
    non-spatial dimensions/coordinates of the input.
    """
    targets = target_coordinates(
        data, target_resolution, latitude_dim=latitude_dim, longitude_dim=longitude_dim
    )
    return data.interp(targets, method="nearest")


@register_method(
    "cubic",
    description="Cubic spline interpolation to the target grid (smooth baseline).",
    minimum_points_per_dim=CUBIC_MIN_POINTS,
)
def cubic_downscale(
    data: xr.DataArray,
    target_resolution: float,
    *,
    latitude_dim: str = DEFAULT_LATITUDE_DIM,
    longitude_dim: str = DEFAULT_LONGITUDE_DIM,
) -> xr.DataArray:
    """Resample ``data`` to ``target_resolution`` using cubic spline interpolation.

    Cubic interpolation needs at least :data:`CUBIC_MIN_POINTS` source points per
    interpolated dimension, otherwise a :class:`DownscalingError` is raised
    instead of letting SciPy fail with an obscure message.
    """
    targets = target_coordinates(
        data, target_resolution, latitude_dim=latitude_dim, longitude_dim=longitude_dim
    )
    for dim in (latitude_dim, longitude_dim):
        if data.sizes[dim] < CUBIC_MIN_POINTS:
            raise DownscalingError(
                f"Cubic downscaling requires at least {CUBIC_MIN_POINTS} points along "
                f"'{dim}', found {data.sizes[dim]}. Use method='nearest' or provide a "
                "larger source grid."
            )
    return data.interp(targets, method="cubic")
