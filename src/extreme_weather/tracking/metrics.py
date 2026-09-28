"""Tracking metrics (SIH26078, Phase 2).

Geometric and temporal helpers used to turn a sequence of
:class:`~extreme_weather.schemas.event.ExtremeWeatherEvent` objects into a
trajectory with speed and heading.

Every function is defensive by contract:

* a non-finite or missing input never raises - it produces a documented neutral
  value (``0.0`` for distances/speeds, ``None`` for an undefined bearing);
* identical coordinates give a distance of ``0.0``, a speed of ``0.0`` and an
  **undefined** bearing, which the caller reports as the direction ``"CALM"``;
* a zero (or negative, or missing) time difference never divides by zero - the
  speed is ``0.0``.

Conventions
-----------
* distance: haversine great-circle distance on a sphere of radius
  :data:`EARTH_RADIUS_KM`, in kilometres;
* bearing: degrees clockwise from **true north** in ``[0, 360)``;
* speed: kilometres per hour.
"""

from __future__ import annotations

from datetime import datetime
from math import asin, atan2, cos, degrees, isfinite, radians, sin, sqrt
from typing import Sequence

__all__ = [
    "CARDINAL_DIRECTIONS",
    "CALM",
    "EARTH_RADIUS_KM",
    "STABLE_TOLERANCE",
    "TREND_STABLE",
    "TREND_STRENGTHENING",
    "TREND_WEAKENING",
    "bearing_degrees",
    "build_track_id",
    "compass_direction",
    "elapsed_hours",
    "haversine_distance_km",
    "intensity_trend",
    "movement_speed_kmh",
    "total_distance_km",
]

#: Mean Earth radius (km) - the same value used by the Phase 1 clustering module.
EARTH_RADIUS_KM: float = 6371.0088

#: Direction reported when an event did not move.
CALM: str = "CALM"

#: 16-point compass rose, starting at north and going clockwise.
CARDINAL_DIRECTIONS: tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

TREND_STRENGTHENING: str = "STRENGTHENING"
TREND_WEAKENING: str = "WEAKENING"
TREND_STABLE: str = "STABLE"

#: Relative change below which an intensity is considered stable.
STABLE_TOLERANCE: float = 0.05


def haversine_distance_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Great-circle distance in kilometres between two coordinates.

    Args:
        latitude_a: Latitude of the first point, degrees north.
        longitude_a: Longitude of the first point, degrees east.
        latitude_b: Latitude of the second point, degrees north.
        longitude_b: Longitude of the second point, degrees east.

    Returns:
        Distance in kilometres; ``0.0`` for identical coordinates and for any
        non-finite input.
    """
    values = (latitude_a, longitude_a, latitude_b, longitude_b)
    if not all(isfinite(float(value)) for value in values):
        return 0.0

    lat_a = radians(float(latitude_a))
    lon_a = radians(float(longitude_a))
    lat_b = radians(float(latitude_b))
    lon_b = radians(float(longitude_b))

    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    inner = sin(delta_lat / 2.0) ** 2 + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(min(1.0, max(0.0, inner))))


def elapsed_hours(start: datetime | None, end: datetime | None) -> float:
    """Absolute time difference in hours, tolerant of missing timestamps.

    Returns ``0.0`` when either timestamp is ``None`` or when naive and aware
    datetimes cannot be compared, so callers can divide without guarding.
    """
    if start is None or end is None:
        return 0.0
    try:
        delta = end - start
    except TypeError:
        return 0.0
    return abs(delta.total_seconds()) / 3600.0


def movement_speed_kmh(distance_km: float, hours: float) -> float:
    """Speed in km/h, returning ``0.0`` instead of dividing by zero.

    A missing, non-finite, zero or negative ``hours`` yields ``0.0``; so does a
    missing or non-finite distance.
    """
    try:
        distance = float(distance_km)
        duration = float(hours)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(distance) or not isfinite(duration) or duration <= 0.0:
        return 0.0
    return distance / duration


def bearing_degrees(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float | None:
    """Initial bearing from point A to point B in degrees clockwise from north.

    Returns:
        Bearing in ``[0, 360)``, or ``None`` when the direction is undefined
        (coincident points or non-finite input) - the caller reports that as the
        direction ``"CALM"``.
    """
    values = (latitude_a, longitude_a, latitude_b, longitude_b)
    if not all(isfinite(float(value)) for value in values):
        return None

    lat_a = radians(float(latitude_a))
    lon_a = radians(float(longitude_a))
    lat_b = radians(float(latitude_b))
    lon_b = radians(float(longitude_b))

    delta_lon = lon_b - lon_a
    y = sin(delta_lon) * cos(lat_b)
    x = cos(lat_a) * sin(lat_b) - sin(lat_a) * cos(lat_b) * cos(delta_lon)
    if y == 0.0 and x == 0.0:
        return None  # coincident points have no direction

    return (degrees(atan2(y, x)) + 360.0) % 360.0


def compass_direction(bearing: float | None) -> str:
    """Convert a bearing into a 16-point compass direction.

    Args:
        bearing: Bearing in degrees, or ``None`` for an undefined direction.

    Returns:
        One of :data:`CARDINAL_DIRECTIONS`, or ``"CALM"`` when the event did not
        move.
    """
    if bearing is None:
        return CALM
    try:
        value = float(bearing)
    except (TypeError, ValueError):
        return CALM
    if not isfinite(value):
        return CALM

    sector = int((value % 360.0 + 11.25) // 22.5) % len(CARDINAL_DIRECTIONS)
    return CARDINAL_DIRECTIONS[sector]


def intensity_trend(
    first_intensity: float | None,
    last_intensity: float | None,
    *,
    tolerance: float = STABLE_TOLERANCE,
) -> str:
    """Classify how the intensity of a track evolved.

    Returns ``"STRENGTHENING"``, ``"WEAKENING"`` or ``"STABLE"``. A relative
    change of at most ``tolerance`` (5 % by default) counts as stable, and a
    missing or non-finite intensity is treated as stable.
    """
    if first_intensity is None or last_intensity is None:
        return TREND_STABLE
    try:
        start = float(first_intensity)
        end = float(last_intensity)
    except (TypeError, ValueError):
        return TREND_STABLE
    if not isfinite(start) or not isfinite(end):
        return TREND_STABLE

    scale = max(abs(start), abs(end))
    if scale == 0.0:
        return TREND_STABLE

    relative_change = (end - start) / scale
    if relative_change > tolerance:
        return TREND_STRENGTHENING
    if relative_change < -tolerance:
        return TREND_WEAKENING
    return TREND_STABLE


def total_distance_km(points: Sequence[tuple[float, float]]) -> float:
    """Sum of the great-circle distances along ``(latitude, longitude)`` points."""
    total = 0.0
    for (lat_a, lon_a), (lat_b, lon_b) in zip(points, points[1:]):
        total += haversine_distance_km(lat_a, lon_a, lat_b, lon_b)
    return total


def build_track_id(event_type: str, start_time: datetime, index: int) -> str:
    """Build the deterministic track identifier ``TRK-<TYPE>-<YYYYMMDDTHHMM>-<NNN>``.

    Args:
        event_type: Event type of the track.
        start_time: Timestamp of the first point of the track.
        index: 1-based index among the tracks starting at the same time, assigned
            by the tracker in a deterministic order.
    """
    return f"TRK-{str(event_type).upper()}-{start_time.strftime('%Y%m%dT%H%M')}-{int(index):03d}"
