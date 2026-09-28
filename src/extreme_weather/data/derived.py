"""Derived meteorological variables (SIH26078, Phase 0).

Only the two wind quantities needed as a foundation are implemented:

``wind_speed``
    ``sqrt(u_wind**2 + v_wind**2)`` in metres per second.

``wind_direction`` (optional, documented convention)
    **Meteorological** convention: the direction the wind blows *from*, measured
    in degrees clockwise from true north. Examples: ``u=0, v=-1`` (westerly flow
    blowing towards the south) gives ``0`` degrees = "wind from the north";
    ``u=-1, v=0`` gives ``90`` degrees = "wind from the east".

    Formula: ``direction = degrees(atan2(-u, -v)) mod 360``.

All functions preserve Xarray coordinates, dimensions and attributes, and never
mutate their inputs.

FUTURE phases add thermo-dynamic derived products (CAPE, precipitable water,
moisture flux convergence, ...) on top of this module.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

WIND_SPEED_NAME: str = "wind_speed"
WIND_DIRECTION_NAME: str = "wind_direction"
U_WIND_NAME: str = "u_wind"
V_WIND_NAME: str = "v_wind"

__all__ = [
    "WIND_DIRECTION_NAME",
    "WIND_SPEED_NAME",
    "add_derived_variables",
    "uv_from_speed_direction",
    "wind_direction",
    "wind_speed",
]


def uv_from_speed_direction(
    speed: xr.DataArray,
    direction: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return ``(u_wind, v_wind)`` from a speed and a meteorological direction.

    Many reanalysis and point-API archives publish wind as a **speed plus a
    direction** rather than as u/v components, but the Phase 0 data contract -
    and the whole Phase 1-5 pipeline - is defined on ``u_wind``/``v_wind``. This
    is the standard, reversible conversion between the two, so such a dataset can
    enter the pipeline without a manual pre-processing step.

    Uses the **meteorological** convention this module already documents:
    ``direction`` is the bearing the wind blows *from*, in degrees clockwise from
    true north. Therefore::

        u = -speed * sin(radians(direction))
        v = -speed * cos(radians(direction))

    Args:
        speed: Wind speed array, in m/s.
        direction: Meteorological direction array, in degrees.

    Returns:
        ``(u_wind, v_wind)`` in m/s, with ``units`` set on both.

    Raises:
        TypeError: If either input is not an :class:`xarray.DataArray`.
    """
    if not isinstance(speed, xr.DataArray) or not isinstance(direction, xr.DataArray):
        raise TypeError("uv_from_speed_direction expects two xarray.DataArray inputs.")

    radians = np.deg2rad(direction)
    u = (-speed * np.sin(radians)).rename(U_WIND_NAME).assign_attrs(units="m s-1")
    v = (-speed * np.cos(radians)).rename(V_WIND_NAME).assign_attrs(units="m s-1")
    return u, v


def _require_pair(u_wind: xr.DataArray, v_wind: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate two wind DataArrays and broadcast them to a common shape."""
    for label, component in (("u_wind", u_wind), ("v_wind", v_wind)):
        if not isinstance(component, xr.DataArray):
            raise TypeError(
                f"Expected an xarray.DataArray for {label}, got {type(component).__name__}."
            )
    return xr.broadcast(u_wind, v_wind)


def wind_speed(u_wind: xr.DataArray, v_wind: xr.DataArray) -> xr.DataArray:
    """Return ``sqrt(u_wind**2 + v_wind**2)`` in m/s.

    Coordinates, dimensions and the input attributes are preserved; a new
    ``units`` attribute is set to ``m s-1``.
    """
    u_component, v_component = _require_pair(u_wind, v_wind)
    speed = np.sqrt(u_component**2 + v_component**2)
    speed = speed.rename(WIND_SPEED_NAME)
    speed.attrs = dict(u_component.attrs)
    speed.attrs.update(
        {
            "units": "m s-1",
            "long_name": "wind speed",
            "standard_name": "wind_speed",
            "derived_from": f"{U_WIND_NAME}, {V_WIND_NAME}",
            "formula": "sqrt(u_wind**2 + v_wind**2)",
        }
    )
    return speed


def wind_direction(u_wind: xr.DataArray, v_wind: xr.DataArray) -> xr.DataArray:
    """Return the *meteorological* wind direction in degrees (direction ``from``).

    The convention is "the direction the wind blows from, clockwise from true
    north": 0 = from the north, 90 = from the east, 180 = from the south,
    270 = from the west. For calm winds (both components zero) the result is 0.
    """
    u_component, v_component = _require_pair(u_wind, v_wind)
    direction = np.degrees(np.arctan2(-u_component, -v_component)) % 360.0
    direction = direction.rename(WIND_DIRECTION_NAME)
    direction.attrs = dict(u_component.attrs)
    direction.attrs.update(
        {
            "units": "degree",
            "long_name": "meteorological wind direction (direction the wind blows FROM)",
            "standard_name": "wind_from_direction",
            "convention": "clockwise from true north; 0=N, 90=E, 180=S, 270=W",
            "formula": "degrees(atan2(-u_wind, -v_wind)) mod 360",
        }
    )
    return direction


def add_derived_variables(
    dataset: xr.Dataset,
    *,
    include_wind_direction: bool = True,
    overwrite: bool = False,
) -> xr.Dataset:
    """Return a copy of ``dataset`` with the derived wind variables added.

    Args:
        dataset: Dataset containing ``u_wind`` and ``v_wind``.
        include_wind_direction: Also add the meteorological direction.
        overwrite: Replace existing ``wind_speed``/``wind_direction`` variables.

    Raises:
        KeyError: If ``u_wind`` or ``v_wind`` is missing.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"add_derived_variables expects an xarray.Dataset, got {type(dataset).__name__}."
        )
    missing = [name for name in (U_WIND_NAME, V_WIND_NAME) if name not in dataset.data_vars]
    if missing:
        raise KeyError(
            f"Cannot derive wind variables: required variable(s) {missing} are missing. "
            f"Available variables: {sorted(dataset.data_vars)}."
        )

    result = dataset.copy(deep=True)
    targets = [WIND_SPEED_NAME] + ([WIND_DIRECTION_NAME] if include_wind_direction else [])

    if WIND_SPEED_NAME not in result or overwrite:
        result[WIND_SPEED_NAME] = wind_speed(result[U_WIND_NAME], result[V_WIND_NAME])
    if include_wind_direction and (WIND_DIRECTION_NAME not in result or overwrite):
        result[WIND_DIRECTION_NAME] = wind_direction(result[U_WIND_NAME], result[V_WIND_NAME])

    result.attrs = dict(dataset.attrs)
    result.attrs["derived_variables"] = ", ".join(targets)
    return result
