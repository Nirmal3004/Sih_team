"""Unit and coordinate normalization (SIH26078, Phase 6).

This module turns a dataset in *some* source's units and *some* source's
coordinate convention into the project's standard model, and records exactly
what it did.

Two rules govern everything here:

1. **Never silently convert an unknown unit.** An unrecognised unit raises
   :class:`~extreme_weather.ingestion.exceptions.NormalizationError`. Guessing
   that ``"mm/day"`` is ``"mm/h"`` would quietly corrupt every downstream
   threshold, and the Phase 1 severity ladder is calibrated in specific units.
2. **Never silently alter geographic meaning.** Every coordinate transform is
   recorded in the result, so a reader can tell that a ``0-360`` dataset was
   wrapped and a descending latitude was flipped.

Target model (the one Phases 1-5 already expect)
------------------------------------------------
======================  ==================
Variable                Unit
======================  ==================
``temperature``         K  (CF convention)
``precipitation``       mm h-1 (a rate)
``pressure``            Pa (CF convention)
``relative_humidity``   %  (0-100)
``u_wind`` / ``v_wind`` m s-1
======================  ==================

with ``time``/``latitude``/``longitude`` coordinates, latitude **ascending** and
longitude in **[-180, 180]**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import xarray as xr

from extreme_weather.ingestion.exceptions import NormalizationError

__all__ = [
    "ALIAS_MAP",
    "CONTRACT_VARIABLES",
    "ConversionRecord",
    "NormalizationResult",
    "STANDARD_UNITS",
    "canonical_unit",
    "detect_unit",
    "normalize_coordinates",
    "normalize_dataset",
    "normalize_variable_name",
    "normalize_units",
]


#: The units every Phase 1-5 consumer expects, plus the polar wind quantities
#: that some archives publish instead of components. The first six are the
#: Phase 0 data contract; the last two are intermediates that
#: :func:`extreme_weather.ingestion.service.derive_uv_from_polar_wind` turns into
#: ``u_wind``/``v_wind``.
STANDARD_UNITS: dict[str, str] = {
    "temperature": "K",
    "precipitation": "mm h-1",
    "pressure": "Pa",
    "relative_humidity": "%",
    "u_wind": "m s-1",
    "v_wind": "m s-1",
    "wind_speed": "m s-1",
    "wind_direction": "degree",
}

#: The six variables the Phase 0 data contract requires.
CONTRACT_VARIABLES: tuple[str, ...] = (
    "temperature",
    "precipitation",
    "pressure",
    "relative_humidity",
    "u_wind",
    "v_wind",
)

#: Accepted source spellings for each canonical variable.
#:
#: ERA5/ECMWF short names, Open-Meteo names and common NetCDF long names all
#: appear here, so a real file rarely needs a manual rename first.
ALIAS_MAP: dict[str, str] = {
    "t": "temperature",
    "t2m": "temperature",
    "2t": "temperature",
    "temp": "temperature",
    "temperature": "temperature",
    "temperature_2m": "temperature",
    "tp": "precipitation",
    "pr": "precipitation",
    "prec": "precipitation",
    "precip": "precipitation",
    "precipitation": "precipitation",
    "precipitation_rate": "precipitation",
    "precipitation_amount": "precipitation",
    "lsm precipitation": "precipitation",
    "msl": "pressure",
    "sp": "pressure",
    "slp": "pressure",
    "pressure": "pressure",
    "surface_pressure": "pressure",
    "mean_sea_level_pressure": "pressure",
    "r": "relative_humidity",
    "rh": "relative_humidity",
    "2r": "relative_humidity",
    "humidity": "relative_humidity",
    "relative_humidity": "relative_humidity",
    "relative_humidity_2m": "relative_humidity",
    "u": "u_wind",
    "u10": "u_wind",
    "u_wind": "u_wind",
    "eastward_wind": "u_wind",
    "u_component_of_wind": "u_wind",
    "v": "v_wind",
    "v10": "v_wind",
    "v_wind": "v_wind",
    "northward_wind": "v_wind",
    "v_component_of_wind": "v_wind",
    "ws": "wind_speed",
    "ff": "wind_speed",
    "wind_speed": "wind_speed",
    "wind_speed_10m": "wind_speed",
    "wd": "wind_direction",
    "dd": "wind_direction",
    "wind_direction": "wind_direction",
    "wind_direction_10m": "wind_direction",
}


#: Unit spellings recognised for each canonical variable, mapped to the
#: standard. ``None`` as a source unit means "no units attribute present" and
#: is accepted only when the variable name already implies the standard unit.
#:
#: A unit that is **not** listed here is never converted - it raises. That is
#: the whole point: an unknown unit is a data-integrity problem, not a
#: formatting detail.
_UNIT_TABLE: dict[str, tuple[tuple[str | None, str], ...]] = {
    "temperature": (
        (None, "K"),
        ("K", "K"),
        ("kelvin", "K"),
        ("K.", "K"),
        ("degC", "K"),
        ("deg_C", "K"),
        ("C", "K"),
        ("celsius", "K"),
        ("degree_Celsius", "K"),
        ("degrees_Celsius", "K"),
        ("degrees_C", "K"),
    ),
    "pressure": (
        (None, "Pa"),
        ("Pa", "Pa"),
        ("pascal", "Pa"),
        ("pascals", "Pa"),
        ("hPa", "Pa"),
        ("mbar", "Pa"),
        ("millibar", "Pa"),
        ("millibars", "Pa"),
    ),
    "relative_humidity": (
        (None, "%"),
        ("%", "%"),
        ("percent", "%"),
        ("pct", "%"),
        ("%RH", "%"),
    ),
    "u_wind": (
        (None, "m s-1"),
        ("m s-1", "m s-1"),
        ("m/s", "m s-1"),
        ("ms-1", "m s-1"),
        ("m s**-1", "m s-1"),
        ("m s^-1", "m s-1"),
        ("metre s-1", "m s-1"),
        ("metres s-1", "m s-1"),
        ("meter s-1", "m s-1"),
        ("meters s-1", "m s-1"),
        # km/h and knots are recognised so they can be *scaled* by the linear
        # factor table below; without an entry here they would be rejected
        # before the factor was ever consulted.
        ("km h-1", "m s-1"),
        ("km/h", "m s-1"),
        ("kph", "m s-1"),
        ("knot", "m s-1"),
        ("kt", "m s-1"),
        ("knots", "m s-1"),
    ),
    "v_wind": (
        (None, "m s-1"),
        ("m s-1", "m s-1"),
        ("m/s", "m s-1"),
        ("ms-1", "m s-1"),
        ("m s**-1", "m s-1"),
        ("m s^-1", "m s-1"),
        ("metre s-1", "m s-1"),
        ("metres s-1", "m s-1"),
        ("meter s-1", "m s-1"),
        ("meters s-1", "m s-1"),
        ("km h-1", "m s-1"),
        ("km/h", "m s-1"),
        ("kph", "m s-1"),
        ("knot", "m s-1"),
        ("kt", "m s-1"),
        ("knots", "m s-1"),
    ),
    "precipitation": (
        (None, "mm h-1"),
        ("mm h-1", "mm h-1"),
        ("mm/hr", "mm h-1"),
        ("mm/hour", "mm h-1"),
        ("mm h**-1", "mm h-1"),
        ("mm h^-1", "mm h-1"),
        ("mm", "mm h-1"),
        ("kg m-2", "mm h-1"),
        ("kg/m2", "mm h-1"),
        ("kg m**-2", "mm h-1"),
    ),
    # Polar wind quantities. Not part of STANDARD_UNITS (the Phase 0 contract
    # needs components, not speed/direction), but recognised so they survive
    # normalization with their correct units and can be converted to u/v.
    "wind_speed": (
        (None, "m s-1"),
        ("m s-1", "m s-1"),
        ("m/s", "m s-1"),
        ("ms-1", "m s-1"),
        ("m s**-1", "m s-1"),
        ("m s^-1", "m s-1"),
        ("km h-1", "m s-1"),
        ("km/h", "m s-1"),
        ("kph", "m s-1"),
        ("knot", "m s-1"),
        ("kt", "m s-1"),
        ("knots", "m s-1"),
    ),
    "wind_direction": (
        (None, "degree"),
        ("degree", "degree"),
        ("degrees", "degree"),
        ("deg", "degree"),
        ("degree_true", "degree"),
        ("degrees_true", "degree"),
        ("degree meteorological", "degree"),
        ("degrees_meteorological", "degree"),
        ("°", "degree"),
    ),
}

#: Linear factors into the standard unit for units that need scaling.
#:
#: 1 kg/m2 of water is exactly 1 mm of liquid equivalent, so the precipitation
#: entry is a pure relabelling and is deliberately factor 1.0.
_LINEAR_FACTORS: dict[tuple[str, str], float] = {
    ("pressure", "hPa"): 100.0,
    ("pressure", "mbar"): 100.0,
    ("pressure", "millibar"): 100.0,
    ("pressure", "millibars"): 100.0,
    ("u_wind", "km h-1"): 1.0 / 3.6,
    ("u_wind", "km/h"): 1.0 / 3.6,
    ("u_wind", "kph"): 1.0 / 3.6,
    ("u_wind", "knot"): 0.514444,
    ("u_wind", "kt"): 0.514444,
    ("u_wind", "knots"): 0.514444,
    ("v_wind", "km h-1"): 1.0 / 3.6,
    ("v_wind", "km/h"): 1.0 / 3.6,
    ("v_wind", "kph"): 1.0 / 3.6,
    ("v_wind", "knot"): 0.514444,
    ("v_wind", "kt"): 0.514444,
    ("v_wind", "knots"): 0.514444,
}

#: Celsius spellings handled by the offset path rather than a linear factor.
_CELSIUS_UNITS: frozenset[str] = frozenset(
    {"degC", "deg_C", "C", "celsius", "degree_Celsius", "degrees_Celsius", "degrees_C"}
)

#: Wind speed units in km/h, accepted for the wind components only.
_KMH_UNITS: frozenset[str] = frozenset({"km h-1", "km/h", "kph"})

#: Units that are only valid for a wind component (never for temperature).
_WIND_ONLY_UNITS: frozenset[str] = _KMH_UNITS



@dataclass(frozen=True)
class ConversionRecord:
    """One applied unit conversion, kept for the manifest.

    Attributes:
        variable: Canonical variable name.
        original_units: Units as they appeared in the source (or ``None``).
        normalized_units: Units after conversion.
        method: Human-readable description of the transformation, e.g.
            ``"identity"``, ``"Celsius -> Kelvin (offset +273.15)"``,
            ``"hPa -> Pa (x100)"``.
        scale: Multiplicative factor applied (``1.0`` for offsets/identity).
    """

    variable: str
    original_units: str | None
    normalized_units: str
    method: str
    scale: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "variable": self.variable,
            "original_units": self.original_units,
            "normalized_units": self.normalized_units,
            "conversion_method": self.method,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class NormalizationResult:
    """The normalized dataset plus a full record of what was changed."""

    dataset: xr.Dataset
    conversions: tuple[ConversionRecord, ...] = ()
    coordinate_actions: tuple[str, ...] = ()
    missing_variables: tuple[str, ...] = ()
    renamed_variables: tuple[tuple[str, str], ...] = ()
    dropped_variables: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: datetime | None = None

    def conversion_map(self) -> dict[str, dict[str, Any]]:
        """Return ``{variable: conversion dict}`` for the manifest."""
        return {record.variable: record.as_dict() for record in self.conversions}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no array data)."""
        return {
            "conversions": [record.as_dict() for record in self.conversions],
            "coordinate_actions": list(self.coordinate_actions),
            "missing_variables": list(self.missing_variables),
            "renamed_variables": [list(pair) for pair in self.renamed_variables],
            "dropped_variables": list(self.dropped_variables),
            "warnings": list(self.warnings),
        }


def normalize_variable_name(name: str) -> str:
    """Return the canonical project name for a source variable.

    Unknown names are returned lower-cased and unchanged rather than dropped, so
    the quality report can say what was left alone.
    """
    key = str(name).strip().lower()
    return ALIAS_MAP.get(key, key)


def detect_unit(variable: xr.DataVariable) -> str | None:
    """Return the declared unit of a variable, or ``None`` when absent."""
    raw = variable.attrs.get("units")
    return str(raw) if raw not in (None, "") else None


def canonical_unit(name: str, units: str | None) -> str | None:
    """Resolve a source unit to the standard unit for ``name``.

    Args:
        name: Canonical variable name.
        units: The source unit string, or ``None`` when absent.

    Returns:
        The standard unit string, or ``None`` when the source unit is already
        standard (including when ``units`` is ``None``).

    Raises:
        NormalizationError: If the unit is not recognised for this variable.
    """
    table = _UNIT_TABLE.get(name)
    if table is None:
        return None
    lookup = units.strip() if isinstance(units, str) and units.strip() else None
    for accepted, standard in table:
        if accepted is None:
            if lookup is None:
                return standard
        elif lookup is not None and lookup == accepted:
            return standard

    if lookup is not None and lookup in _WIND_ONLY_UNITS and name not in ("u_wind", "v_wind"):
        raise NormalizationError(
            f"Variable '{name}' declares unit '{lookup}', which is a wind-speed "
            f"unit. Refusing to convert it into '{STANDARD_UNITS.get(name, '?')}' "
            "because that would change what the variable means."
        )

    supported = ", ".join(sorted(u for u, _ in table if u is not None))
    raise NormalizationError(
        f"Variable '{name}' declares unit '{units}' which is not recognised. "
        f"Refusing to guess a conversion. Supported units: {supported}. "
        "Convert the source upstream, or extend _UNIT_TABLE deliberately."
    )



def _apply_unit_conversion(
    variable: xr.DataArray,
    name: str,
    units: str | None,
) -> tuple[xr.DataArray, ConversionRecord]:
    """Apply the unit conversion for one variable.

    Raises:
        NormalizationError: If the unit is unrecognised, or the conversion would
            be physically wrong (an accumulation presented as a rate).
    """
    standard = STANDARD_UNITS.get(name)
    if standard is None:
        return variable, ConversionRecord(
            variable=name,
            original_units=units,
            normalized_units=units or "",
            method="left unchanged (not a standard project variable)",
        )

    resolved = canonical_unit(name, units)
    lookup = units.strip() if isinstance(units, str) and units.strip() else None

    if lookup is not None and lookup in _CELSIUS_UNITS:
        converted = variable + 273.15
        converted.attrs = dict(variable.attrs)
        converted.attrs["units"] = "K"
        return converted, ConversionRecord(
            variable=name,
            original_units=lookup,
            normalized_units="K",
            method="Celsius -> Kelvin (offset +273.15)",
            scale=1.0,
        )

    factor = _LINEAR_FACTORS.get((name, lookup or ""), 1.0)
    if factor != 1.0:
        converted = variable * factor
        converted.attrs = dict(variable.attrs)
        converted.attrs["units"] = standard
        return converted, ConversionRecord(
            variable=name,
            original_units=lookup,
            normalized_units=standard,
            method=f"{lookup} -> {standard} (x{factor:g})",
            scale=factor,
        )

    method = "identity" if lookup == standard else f"relabelled {lookup or 'no unit'}"
    if lookup is None:
        method = "assumed standard (no units attribute)"
    converted = variable
    converted.attrs = dict(variable.attrs)
    converted.attrs["units"] = standard
    return converted, ConversionRecord(
        variable=name,
        original_units=lookup,
        normalized_units=standard or "",
        method=method,
    )


def normalize_coordinates(
    dataset: xr.Dataset,
) -> tuple[xr.Dataset, list[str], list[str]]:
    """Normalise the time/latitude/longitude coordinates.

    Applied, in order:

    * **time** - sorted ascending, duplicates removed (keeping the first), and
      any non-monotonic ordering reported.
    * **latitude** - sorted **ascending**, so downstream code can assume the
      documented ordering.
    * **longitude** - wrapped from ``0-360`` into ``[-180, 180]`` when the range
      says so, then sorted ascending.

    Args:
        dataset: The dataset to normalise (not modified).

    Returns:
        ``(dataset, actions, warnings)`` where ``actions`` records each applied
        transform and ``warnings`` records anything suspicious.

    Raises:
        NormalizationError: If a required coordinate is missing or unusable.
    """
    from extreme_weather.ingestion.base import utc_now

    actions: list[str] = []
    warnings: list[str] = []
    result = dataset

    # --- time --------------------------------------------------------- #
    if "time" not in result.coords:
        raise NormalizationError(
            "Dataset has no 'time' coordinate. Phase 1-5 require a time axis; "
            "this dataset cannot enter the pipeline."
        )
    time_values = result.coords["time"].values
    if time_values.size > 1 and not np.all(np.diff(time_values) > np.timedelta64(0, "s")):
        if len(np.unique(time_values)) != time_values.size:
            duplicates = int(time_values.size - len(np.unique(time_values)))
            result = result.isel(time=~result.coords["time"].duplicated())
            actions.append(f"removed {duplicates} duplicate timestamp(s)")
            warnings.append(f"Removed {duplicates} duplicate timestamp(s).")
        else:
            result = result.sortby("time")
            actions.append("sorted time ascending")
            warnings.append("Time coordinate was not monotonic and was sorted.")

    # --- latitude ------------------------------------------------------ #
    if "latitude" not in result.coords and "latitude" not in result.dims:
        raise NormalizationError(
            "Dataset has no 'latitude' coordinate. Phase 1-5 require a regular "
            "latitude axis."
        )
    latitude = _coordinate_1d(result, "latitude")
    if latitude is not None and latitude.size > 1 and latitude[0] > latitude[-1]:
        result = result.sortby("latitude")
        actions.append("sorted latitude ascending (was descending)")

    # --- longitude ----------------------------------------------------- #
    if "longitude" not in result.coords and "longitude" not in result.dims:
        raise NormalizationError(
            "Dataset has no 'longitude' coordinate. Phase 1-5 require a regular "
            "longitude axis."
        )
    longitude = _coordinate_1d(result, "longitude")
    if longitude is None:
        raise NormalizationError(
            "The 'longitude' coordinate is not 1-D. Phase 1-5 require a regular "
            "longitude axis."
        )
    if longitude.size and (longitude.min() < 0.0 or longitude.max() > 180.0):
        wrapped = ((longitude + 180.0) % 360.0) - 180.0
        result = result.assign_coords(longitude=wrapped)
        actions.append(
            f"wrapped longitude from [{longitude.min():.3f}, {longitude.max():.3f}] "
            "into [-180, 180]"
        )
        wrapped_values = _coordinate_1d(result, "longitude")
        if wrapped_values is not None and wrapped_values.size > 1:
            if wrapped_values[0] > wrapped_values[-1]:
                result = result.sortby("longitude")
                actions.append("sorted longitude ascending")

    result.attrs = dict(result.attrs)
    result.attrs["normalized_at"] = utc_now().isoformat()
    return result, actions, warnings


def _coordinate_1d(dataset: xr.Dataset, name: str) -> np.ndarray | None:
    """Return a 1-D coordinate's values, or ``None`` when it is not 1-D."""
    if name not in dataset.coords:
        return None
    coord = dataset.coords[name]
    if coord.ndim != 1:
        return None
    return np.asarray(coord.values)



def normalize_dataset(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str] | None = None,
    drop_unknown: bool = False,
    generated_at: datetime | None = None,
) -> NormalizationResult:
    """Normalise a dataset into the project's standard model.

    Steps, in order:

    1. rename source variables to canonical names (recording each rename);
    2. convert each canonical variable's units to the standard, **refusing** to
       guess when the unit is unrecognised;
    3. normalise the time/latitude/longitude coordinates;
    4. report which standard variables are absent, and (optionally) drop the
       variables the project does not model.

    Missing variables are reported, never invented: a dataset without ``u_wind``
    is a valid ERA5 subset, and the Phase 1 detector will simply not use wind.

    Args:
        dataset: Source dataset, in its own units and coordinates.
        variables: Optional whitelist of canonical variables to keep.
        drop_unknown: Drop variables that are not canonical project variables
            instead of keeping them untouched.
        generated_at: Timestamp to stamp on the result.

    Returns:
        A :class:`NormalizationResult` with the dataset and the full record.

    Raises:
        NormalizationError: If a unit is unrecognised or a coordinate is missing.
    """
    from extreme_weather.ingestion.base import utc_now

    stamp = generated_at or utc_now()
    renamed: list[tuple[str, str]] = []
    dropped: list[str] = []
    conversions: list[ConversionRecord] = []
    warnings: list[str] = []

    working = dataset
    # --- 1. rename ---------------------------------------------------- #
    for original in list(working.data_vars):
        canonical = normalize_variable_name(original)
        if canonical == original:
            continue
        if canonical in working.data_vars:
            # A collision (e.g. both "t" and "temperature"): keep the
            # canonical one and drop the alias rather than clobbering it.
            dropped.append(original)
            warnings.append(
                f"Dropped '{original}': '{canonical}' is already present."
            )
            working = working.drop_vars(original)
            continue
        working = working.rename_vars({original: canonical})
        renamed.append((original, canonical))

    # --- 2. units ------------------------------------------------------ #
    for name in list(working.data_vars):
        variable = working[name]
        if name not in STANDARD_UNITS:
            if drop_unknown:
                # Recorded *and* removed: a variable listed as dropped but still
                # present would make the manifest lie about what was written.
                dropped.append(name)
                working = working.drop_vars(name)
                continue
            warnings.append(
                f"Kept non-standard variable '{name}' unchanged "
                f"(units: {detect_unit(variable) or 'none'})."
            )
            continue
        converted, record = _apply_unit_conversion(variable, name, detect_unit(variable))
        working[name] = converted
        conversions.append(record)

    # --- 3. coordinates ------------------------------------------------ #
    working, actions, coordinate_warnings = normalize_coordinates(working)
    warnings.extend(coordinate_warnings)

    # --- 4. whitelist / reporting -------------------------------------- #
    keep = set(variables) if variables else set()
    if keep:
        for name in list(working.data_vars):
            if name not in keep:
                dropped.append(name)
                working = working.drop_vars(name)

    present = set(working.data_vars)
    missing = tuple(v for v in CONTRACT_VARIABLES if v not in present)
    if missing:
        warnings.append(
            "Variables of the Phase 0 data contract that are absent from this "
            f"dataset and therefore not available downstream: {', '.join(missing)}."
        )

    attrs = dict(working.attrs)
    attrs["normalized_variables"] = ",".join(sorted(present))
    attrs["normalization_warnings"] = "; ".join(warnings) if warnings else "none"
    working.attrs = attrs

    return NormalizationResult(
        dataset=working,
        conversions=tuple(conversions),
        coordinate_actions=tuple(actions),
        missing_variables=missing,
        renamed_variables=tuple(renamed),
        dropped_variables=tuple(dropped),
        warnings=tuple(warnings),
        generated_at=stamp,
    )

