"""Tests for spatial event clustering.

Clusters are built from explicit masks and scores, so every geometric property
(centroid, radius, cell count) is verifiable without any randomness.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.anomaly.clustering import (
    EARTH_RADIUS_KM,
    ClusteringError,
    build_event_id,
    cluster_extreme_events,
    haversine_km,
    intensity_weighted_centroid,
    timestamp_to_datetime,
)
from extreme_weather.anomaly.thresholds import ThresholdLadder
from extreme_weather.schemas.event import EventType, Severity

TIMES = np.arange(
    np.datetime64("2024-06-01T00:00"), np.datetime64("2024-06-01T06:00"), np.timedelta64(1, "h")
)
LATITUDES = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
LONGITUDES = np.array([70.0, 70.5, 71.0, 71.5, 72.0])
LADDER = ThresholdLadder(
    name="test_ladder",
    event_type=EventType.EXTREME_RAINFALL,
    variable="precipitation",
    score_name="absolute",
    steps=((10.0, Severity.WATCH), (40.0, Severity.SEVERE), (60.0, Severity.EXTREME)),
)
SHAPE = (TIMES.size, LATITUDES.size, LONGITUDES.size)


def build_fields(
    mask: np.ndarray, scores: np.ndarray, values: np.ndarray | None = None
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None]:
    """Wrap ``(time, latitude, longitude)`` arrays into the clustering inputs."""
    coords = {"time": TIMES, "latitude": LATITUDES, "longitude": LONGITUDES}
    dims = ("time", "latitude", "longitude")
    score_array = xr.DataArray(scores.astype("float32"), dims=dims, coords=coords, name="score")
    mask_array = xr.DataArray(mask.astype(bool), dims=dims, coords=coords, name="extreme_mask")
    value_array = (
        xr.DataArray(values.astype("float32"), dims=dims, coords=coords, name="precipitation")
        if values is not None
        else None
    )
    return mask_array, score_array, value_array


def empty_fields() -> tuple[np.ndarray, np.ndarray]:
    """Return empty (all-False) mask/score arrays."""
    return np.zeros(SHAPE, dtype=bool), np.zeros(SHAPE)


def cluster(
    mask: np.ndarray, scores: np.ndarray, values: np.ndarray | None = None, **kwargs
) -> list:
    """Convenience wrapper around :func:`cluster_extreme_events`."""
    mask_array, score_array, value_array = build_fields(mask, scores, values)
    return cluster_extreme_events(
        mask_array,
        score_array,
        ladder=LADDER,
        event_type=EventType.EXTREME_RAINFALL,
        values=value_array,
        variable="precipitation" if value_array is not None else None,
        units="mm h-1" if value_array is not None else None,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def test_haversine_matches_known_distances() -> None:
    # One degree of latitude is ~111.19 km with the mean Earth radius.
    north = haversine_km(np.array([10.0]), np.array([70.0]), np.array([11.0]), np.array([70.0]))
    assert float(north[0]) == pytest.approx(111.19, abs=0.1)

    zero = haversine_km(np.array([10.0]), np.array([70.0]), np.array([10.0]), np.array([70.0]))
    assert float(zero[0]) == pytest.approx(0.0, abs=1e-9)

    # Longitude degrees shrink with latitude (cos factor).
    equator = haversine_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    high = haversine_km(np.array([60.0]), np.array([0.0]), np.array([60.0]), np.array([1.0]))
    assert float(high[0]) == pytest.approx(0.5 * float(equator[0]), rel=0.01)
    assert EARTH_RADIUS_KM == pytest.approx(6371.0, abs=1.0)


def test_intensity_weighted_centroid() -> None:
    latitudes = np.array([10.0, 10.0, 11.0])
    longitudes = np.array([70.0, 70.0, 71.0])

    # A strong weight pulls the centroid towards the last cell.
    pulled = intensity_weighted_centroid(latitudes, longitudes, np.array([0.0, 0.0, 100.0]))
    assert pulled[0] == pytest.approx(11.0, abs=1e-6)
    assert pulled[1] == pytest.approx(71.0, abs=1e-6)

    # All-zero weights fall back to the geometric centroid.
    geometric = intensity_weighted_centroid(latitudes, longitudes, np.zeros(3))
    assert geometric[0] == pytest.approx(10.3333333, abs=1e-5)
    assert geometric[1] == pytest.approx(70.3333333, abs=1e-5)


def test_event_id_is_deterministic() -> None:
    first = build_event_id(EventType.EXTREME_RAINFALL, np.datetime64("2024-06-01T05:30"), 2)
    second = build_event_id("EXTREME_RAINFALL", np.datetime64("2024-06-01T05:30"), 2)

    assert first == second
    assert first == "EVT-EXTREME_RAINFALL-20240601T0530-002"


def test_timestamp_conversion() -> None:
    converted = timestamp_to_datetime(np.datetime64("2024-06-01T05:30"))
    assert (converted.year, converted.month, converted.day) == (2024, 6, 1)
    assert (converted.hour, converted.minute) == (5, 30)


# --------------------------------------------------------------------------- #
# Cluster cases
# --------------------------------------------------------------------------- #
def test_no_cells_produces_no_events() -> None:
    mask, scores = empty_fields()
    assert cluster(mask, scores) == []


def test_single_cell_produces_one_event() -> None:
    mask, scores = empty_fields()
    mask[0, 2, 3] = True
    scores[0, 2, 3] = 45.0

    events = cluster(mask, scores, values=scores + 100.0)

    assert len(events) == 1
    event = events[0]
    assert event.cell_count == 1
    assert event.latitude == pytest.approx(11.0)
    assert event.longitude == pytest.approx(71.5)
    assert event.anomaly_score == pytest.approx(45.0)
    assert event.affected_radius_km == pytest.approx(0.0)
    assert event.severity is Severity.SEVERE
    assert event.timestamp.isoformat().startswith("2024-06-01T00:00")
    assert event.intensity == pytest.approx(145.0)


def test_adjacent_cells_form_one_cluster_with_a_geometric_centroid() -> None:
    mask, scores = empty_fields()
    mask[0, 1:3, 1:3] = True
    scores[0, 1:3, 1:3] = 30.0

    events = cluster(mask, scores)

    assert len(events) == 1
    event = events[0]
    assert event.cell_count == 4
    assert event.latitude == pytest.approx(10.75)  # mean of 10.5 and 11.0
    assert event.longitude == pytest.approx(70.75)  # mean of 70.5 and 71.0
    # The farthest cell is half a diagonal cell away (~39 km at this latitude).
    assert event.affected_radius_km == pytest.approx(39.3, abs=1.0)


def test_separated_cells_form_separate_events() -> None:
    mask, scores = empty_fields()
    mask[0, 0, 0] = True
    mask[0, 4, 4] = True
    scores[0, 0, 0] = 20.0
    scores[0, 4, 4] = 70.0

    events = cluster(mask, scores)

    assert len(events) == 2
    # Ordered by descending peak score inside the time step.
    assert events[0].anomaly_score == pytest.approx(70.0)
    assert events[0].event_id.endswith("-001")
    assert events[1].anomaly_score == pytest.approx(20.0)
    assert events[1].event_id.endswith("-002")
    assert events[0].latitude == pytest.approx(12.0)
    assert events[1].latitude == pytest.approx(10.0)


def test_events_at_different_times_get_distinct_ids() -> None:
    mask, scores = empty_fields()
    mask[0, 1, 1] = True
    mask[3, 1, 1] = True
    scores[0, 1, 1] = 25.0
    scores[3, 1, 1] = 25.0

    events = cluster(mask, scores)

    assert len(events) == 2
    assert events[0].event_id == "EVT-EXTREME_RAINFALL-20240601T0000-001"
    assert events[1].event_id == "EVT-EXTREME_RAINFALL-20240601T0300-001"
    assert events[0].timestamp < events[1].timestamp


def test_intensity_weighting_moves_the_centroid_towards_the_peak() -> None:
    mask, scores = empty_fields()
    mask[0, 2, 0] = True
    mask[0, 2, 1] = True
    scores[0, 2, 0] = 15.0
    scores[0, 2, 1] = 95.0

    events = cluster(mask, scores)

    assert len(events) == 1
    # Without weighting the centroid would be 70.25; the strong cell pulls it east.
    assert events[0].longitude > 70.25
    assert events[0].longitude < 70.5
    assert events[0].cell_count == 2


def test_nan_scores_are_never_clustered() -> None:
    mask, scores = empty_fields()
    mask[0, 1, 1] = True
    mask[0, 3, 3] = True
    scores[0, 1, 1] = np.nan
    scores[0, 3, 3] = 30.0

    events = cluster(mask, scores)

    # The cell without a score is dropped, the other one is kept.
    assert len(events) == 1
    assert events[0].cell_count == 1
    assert events[0].latitude == pytest.approx(11.5)


def test_nan_values_do_not_break_intensity_statistics() -> None:
    mask, scores = empty_fields()
    mask[0, 1:3, 1:3] = True
    scores[0, 1:3, 1:3] = 30.0
    values = scores + 100.0
    values[0, 1, 1] = np.nan

    events = cluster(mask, scores, values=values)

    assert len(events) == 1
    assert events[0].cell_count == 4
    assert events[0].intensity == pytest.approx(130.0)
    assert events[0].mean_value == pytest.approx(130.0)
    assert events[0].max_value == pytest.approx(130.0)


def test_very_small_events_are_filtered_by_min_cells() -> None:
    mask, scores = empty_fields()
    mask[0, 0, 0] = True
    mask[0, 4, 4] = True
    scores[0, 0, 0] = 30.0
    scores[0, 4, 4] = 40.0

    assert len(cluster(mask, scores, min_cells=1)) == 2
    assert cluster(mask, scores, min_cells=2) == []


def test_severity_comes_from_the_ladder() -> None:
    mask, scores = empty_fields()
    # Three separated cells, each with a different score band.
    mask[0, 0, 0] = True
    mask[0, 0, 2] = True
    mask[0, 0, 4] = True
    scores[0, 0, 0] = 12.0
    scores[0, 0, 2] = 45.0
    scores[0, 0, 4] = 75.0

    events = cluster(mask, scores)

    assert [event.severity for event in events] == [
        Severity.EXTREME,
        Severity.SEVERE,
        Severity.WATCH,
    ]
    # The event with the highest score is listed first inside the time step.
    assert [event.anomaly_score for event in events] == [75.0, 45.0, 12.0]

def test_connectivity_setting_changes_grouping() -> None:
    # Diagonal-only neighbours: 4-connectivity splits them, 8-connectivity joins.
    mask, scores = empty_fields()
    mask[0, 1, 1] = True
    mask[0, 2, 2] = True
    scores[0, 1, 1] = 20.0
    scores[0, 2, 2] = 25.0

    assert len(cluster(mask, scores, connectivity=8)) == 1
    assert len(cluster(mask, scores, connectivity=4)) == 2


def test_clustering_input_validation() -> None:
    mask, scores = empty_fields()
    mask[0, 0, 0] = True
    scores[0, 0, 0] = 20.0
    mask_array, score_array, _ = build_fields(mask, scores)

    with pytest.raises(ClusteringError, match="connectivity must be 4 or 8"):
        cluster(mask, scores, connectivity=6)
    with pytest.raises(ClusteringError, match="min_cells must be >= 1"):
        cluster(mask, scores, min_cells=0)
    with pytest.raises(ClusteringError, match="expose 'thresholds' and 'classify'"):
        cluster_extreme_events(mask_array, score_array, ladder=object())
    with pytest.raises(ClusteringError, match="expects an xarray.DataArray"):
        cluster_extreme_events(np.asarray(mask_array), score_array, ladder=LADDER)
