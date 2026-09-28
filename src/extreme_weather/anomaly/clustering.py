"""Spatial event clustering (SIH26078, Phase 1).

A boolean extreme-cell mask is grouped into individual
:class:`~extreme_weather.schemas.event.ExtremeWeatherEvent` objects, one per
connected spatial cluster per time step.

Method
------
1. Each time slice of the mask is labelled with
   :func:`scipy.ndimage.label` (4- or 8-connectivity, no Python loop over cells).
2. For every label:
   * **cell count** - number of labelled cells;
   * **centroid** - intensity-weighted mean of the cell coordinates, where the
     weight of a cell is ``max(score - detection_level, 0) + weight_floor``. The
     floor keeps a constant-score cluster (where every weight would be zero)
     from collapsing onto a single point;
   * **radius** - the largest geodesic distance (haversine approximation) from
     that centroid to any cell of the cluster, so the value is derived from the
     geometry instead of being hard-coded;
   * **intensity** - the peak physical value of the cluster;
   * **severity** - classification of the peak score with the threshold ladder.
3. Clusters smaller than ``min_cells`` are discarded as noise.

NaN scores never reach this module (the detector masks them out), but a NaN in
the *value* field is ignored when computing intensities and the mean.

Event identifiers are deterministic:
``EVT-<TYPE>-<YYYYMMDDTHHMM>-<NNN>`` where ``NNN`` is the cluster index inside
that time step (ordered by descending intensity).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from datetime import datetime
from scipy.ndimage import label as connected_components
from typing import Any

from extreme_weather.data.validator import LATITUDE_DIM, LONGITUDE_DIM, TIME_DIM
from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent


def timestamp_to_datetime(value: np.datetime64) -> datetime:
    """Convert a ``datetime64`` value into a timezone-naive Python ``datetime``.

    The :class:`ExtremeWeatherEvent` model normalises naive timestamps to UTC.
    """
    return datetime.fromisoformat(str(np.datetime64(value, "s")))

__all__ = [
    "DEFAULT_CONNECTIVITY",
    "DEFAULT_MIN_CELLS",
    "DEFAULT_WEIGHT_FLOOR",
    "EARTH_RADIUS_KM",
    "ClusteringError",
    "build_event_id",
    "cluster_extreme_events",
    "haversine_km",
    "intensity_weighted_centroid",
    "timestamp_to_datetime",
]

#: Spatial connectivity passed to ``scipy.ndimage.label``.
DEFAULT_CONNECTIVITY: int = 8

#: Clusters with fewer cells are treated as noise and dropped.
DEFAULT_MIN_CELLS: int = 1

#: Minimum weight added to every cell so a constant-score cluster still has a
#: well-defined (geometric) centroid.
DEFAULT_WEIGHT_FLOOR: float = 1e-6

#: Earth radius used by the haversine approximation, in kilometres.
EARTH_RADIUS_KM: float = 6371.0088


class ClusteringError(ValueError):
    """Raised when clustering inputs are inconsistent."""


def haversine_km(
    latitude_a: np.ndarray,
    longitude_a: np.ndarray,
    latitude_b: np.ndarray,
    longitude_b: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in kilometres between coordinate pairs.

    Haversine approximation on a spherical Earth - accurate to a few tenths of a
    percent, which is far below the resolution of any prototype grid.
    """
    lat_a = np.radians(np.asarray(latitude_a, dtype="float64"))
    lon_a = np.radians(np.asarray(longitude_a, dtype="float64"))
    lat_b = np.radians(np.asarray(latitude_b, dtype="float64"))
    lon_b = np.radians(np.asarray(longitude_b, dtype="float64"))

    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def intensity_weighted_centroid(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    weights: np.ndarray,
    *,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
) -> tuple[float, float]:
    """Return the intensity-weighted centroid of a cluster.

    Args:
        latitudes: Cell latitudes (degrees).
        longitudes: Cell longitudes (degrees).
        weights: Non-negative cell weights (for example ``score - level``).
        weight_floor: Constant added to every weight so a cluster with zero total
            weight still yields its geometric centroid.

    Returns:
        ``(latitude, longitude)`` of the centroid.
    """
    weight_values = np.nan_to_num(np.asarray(weights, dtype="float64"), nan=0.0, posinf=0.0)
    if np.any(weight_values < 0.0):
        weight_values = np.abs(weight_values)
    total = float(np.sum(weight_values + weight_floor))
    if total <= 0.0:
        return float(np.mean(latitudes)), float(np.mean(longitudes))

    latitude = float(np.sum(latitudes * (weight_values + weight_floor)) / total)
    longitude = float(np.sum(longitudes * (weight_values + weight_floor)) / total)
    return latitude, longitude


def build_event_id(event_type: EventType | str, timestamp: np.datetime64, index: int) -> str:
    """Build the deterministic event identifier.

    Args:
        event_type: Event type of the event.
        timestamp: Valid time of the time step.
        index: Cluster index inside the time step (1-based).

    Returns:
        ``EVT-<TYPE>-<YYYYMMDDTHHMM>-<NNN>``.
    """
    resolved = (
        event_type if isinstance(event_type, EventType) else EventType.from_value(str(event_type))
    )
    stamp = (
        str(np.datetime_as_string(np.asarray(timestamp), unit="m"))
        .replace(":", "")
        .replace("-", "")
    )
    return f"EVT-{resolved.value}-{stamp}-{int(index):03d}"


def _require_fields(
    mask: xr.DataArray,
    scores: xr.DataArray,
    values: xr.DataArray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and return the numpy arrays of the clustering inputs."""
    if not isinstance(mask, xr.DataArray):
        raise ClusteringError(
            f"cluster_extreme_events expects an xarray.DataArray mask, got {type(mask).__name__}."
        )
    if LATITUDE_DIM not in mask.dims or LONGITUDE_DIM not in mask.dims:
        raise ClusteringError(
            f"Mask must have '{LATITUDE_DIM}' and '{LONGITUDE_DIM}' dimensions, got {mask.dims}."
        )
    if scores.dims != mask.dims:
        raise ClusteringError(f"Score dims {scores.dims} do not match the mask dims {mask.dims}.")
    if values is not None and values.dims != mask.dims:
        raise ClusteringError(f"Value dims {values.dims} do not match the mask dims {mask.dims}.")

    mask_values = np.asarray(mask.values, dtype=bool)
    # NaN scores are kept here on purpose: the clustering loop uses them to drop
    # cells that have no score at all.
    score_values = np.asarray(scores.values, dtype="float64")
    value_values = (
        np.asarray(values.values, dtype="float64")
        if values is not None
        else None
    )
    return mask_values, score_values, value_values


def cluster_extreme_events(
    mask: xr.DataArray,
    scores: xr.DataArray,
    *,
    ladder: object,
    event_type: EventType | str = EventType.EXTREME_RAINFALL,
    values: xr.DataArray | None = None,
    detection_level: float | None = None,
    min_cells: int = DEFAULT_MIN_CELLS,
    connectivity: int = DEFAULT_CONNECTIVITY,
    variable: str | None = None,
    units: str | None = None,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
) -> list[ExtremeWeatherEvent]:
    """Group extreme grid cells into :class:`ExtremeWeatherEvent` objects.

    Args:
        mask: Boolean extreme-cell mask with ``(time,) latitude, longitude``.
        scores: Anomaly score field with the same dims as ``mask``.
        ladder: Threshold ladder; provides the detection level (first threshold)
            and classifies the peak score into a severity.
        event_type: Event type assigned to every produced event.
        values: Optional physical field (same dims) used for ``intensity``,
            ``max_value`` and ``mean_value``. Without it the peak score is used
            as the intensity.
        detection_level: Score level of the mask; defaults to the first threshold
            of ``ladder``.
        min_cells: Minimum cluster size; smaller clusters are dropped as noise.
        connectivity: 4 or 8 (default) neighbourhood connectivity.
        variable: Source variable name recorded on the events.
        units: Units of the source variable recorded on the events.
        weight_floor: See :func:`intensity_weighted_centroid`.

    Returns:
        Events ordered by time step, then by descending peak score, with
        deterministic ``event_id`` values.

    Raises:
        ClusteringError: If the inputs are inconsistent or ``connectivity`` is
            neither 4 nor 8.
    """
    if int(connectivity) not in (4, 8):
        raise ClusteringError(f"connectivity must be 4 or 8, got {connectivity}.")
    if int(min_cells) < 1:
        raise ClusteringError(f"min_cells must be >= 1, got {min_cells}.")
    if not hasattr(ladder, "classify") or not hasattr(ladder, "thresholds"):
        raise ClusteringError(
            "ladder must expose 'thresholds' and 'classify'; pass a ThresholdLadder."
        )

    mask_values, score_values, value_values = _require_fields(mask, scores, values)
    level = (
        float(detection_level)
        if detection_level is not None
        else float(np.min(np.asarray(ladder.thresholds, dtype="float64")))
    )

    latitudes = np.asarray(mask[LATITUDE_DIM].values, dtype="float64")
    longitudes = np.asarray(mask[LONGITUDE_DIM].values, dtype="float64")
    if TIME_DIM in mask.dims:
        stamps = [
            np.datetime64(value, "us").astype("datetime64[s]")
            for value in np.asarray(mask[TIME_DIM].values)
        ]
    else:
        stamps = [np.datetime64("1970-01-01T00:00", "s")]

    structure = np.ones((3, 3), dtype=bool) if int(connectivity) == 8 else None
    collected: list[dict[str, Any]] = []

    for step in range(mask_values.shape[0]):
        step_mask = mask_values[step]
        # Defensive: a cell with no score can never be an extreme cell, even if
        # a hand-built mask marks it.
        step_mask = step_mask & np.isfinite(score_values[step])
        if not step_mask.any():
            continue

        labels, cluster_count = connected_components(step_mask, structure=structure)
        for cluster_index in range(1, int(cluster_count) + 1):
            selection = (labels == cluster_index).reshape(-1)
            cell_count = int(np.count_nonzero(selection))
            if cell_count < int(min_cells):
                continue

            cluster_scores = score_values[step].reshape(-1)[selection]
            peak_score = float(np.max(cluster_scores))
            cell_latitudes = np.repeat(latitudes, longitudes.size)[selection]
            cell_longitudes = np.tile(longitudes, latitudes.size)[selection]
            weights = np.clip(cluster_scores - level, 0.0, None)
            centroid_latitude, centroid_longitude = intensity_weighted_centroid(
                cell_latitudes, cell_longitudes, weights, weight_floor=weight_floor
            )
            radius = float(
                np.max(
                    haversine_km(
                        centroid_latitude, centroid_longitude, cell_latitudes, cell_longitudes
                    )
                )
            )

            if value_values is not None:
                cluster_values = value_values[step].reshape(-1)[selection]
                finite_values = cluster_values[np.isfinite(cluster_values)]
                intensity = float(np.max(finite_values)) if finite_values.size else float(peak_score)
                mean_value = float(np.mean(finite_values)) if finite_values.size else None
            else:
                intensity = peak_score
                mean_value = None

            collected.append(
                {
                    "step": step,
                    "timestamp": stamps[step],
                    "peak_score": peak_score,
                    "latitude": centroid_latitude,
                    "longitude": centroid_longitude,
                    "intensity": intensity,
                    "severity": ladder.classify(peak_score),
                    "affected_radius_km": radius,
                    "cell_count": cell_count,
                    "mean_value": mean_value,
                }
            )

    collected.sort(key=lambda item: (item["step"], -item["peak_score"], item["latitude"]))
    counters: dict[int, int] = {}
    events: list[ExtremeWeatherEvent] = []
    for item in collected:
        step = int(item["step"])
        counters[step] = counters.get(step, 0) + 1
        events.append(
            ExtremeWeatherEvent(
                event_id=build_event_id(event_type, item["timestamp"], counters[step]),
                event_type=event_type,
                timestamp=timestamp_to_datetime(item["timestamp"]),
                latitude=item["latitude"],
                longitude=item["longitude"],
                intensity=item["intensity"],
                anomaly_score=item["peak_score"],
                severity=item["severity"],
                affected_radius_km=item["affected_radius_km"],
                cell_count=item["cell_count"],
                variable=variable,
                units=units,
                max_value=item["intensity"],
                mean_value=item["mean_value"],
            )
        )
    return events
