"""Test doubles and helpers for the Phase 9 evaluation tests (SIH26078).

Two things live here:

* deterministic **synthetic sample sets** for the GNN baseline comparisons, so
  the evaluator can be tested with no data, no model and no network;
* a real minimal :class:`ExtremeWeatherEvent` / :class:`EventTrack` pair, so
  explainability can be tested against the actual Phase 1 and Phase 2 schemas
  rather than a hand-rolled stand-in.

The synthetic samples are deliberately *structured* (a smooth drift plus noise)
rather than pure noise, because pure noise makes every baseline tie and would
hide bugs in the comparison logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest

from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent, Severity
from extreme_weather.schemas.track import (
    EventTrack,
    TrackPoint,
    TrackStatus,
)

UTC = timezone.utc

#: A fixed reference instant, so freshness-style assertions never depend on the clock.
REFERENCE_TIME = datetime(2024, 6, 30, 12, 0, tzinfo=UTC)

__all__ = [
    "REFERENCE_TIME",
    "make_event",
    "make_samples",
    "make_track",
]


def make_samples(
    count: int = 60,
    seed: int = 11,
    *,
    model_skill: float = 0.5,
    constant_velocity: bool = True,
) -> dict[str, np.ndarray]:
    """Build a structured synthetic next-state sample set.

    The truth is a steady eastward drift. A persistence forecast is therefore
    *worse* than a constant-velocity forecast, which is the relationship a real
    moving-event dataset shows and which the comparison logic must handle.

    Args:
        count: Number of samples.
        seed: Random seed.
        model_skill: Scale of the model's error, in degrees. Smaller is better.
        constant_velocity: Include the previous-step positions needed by the
            constant-velocity baseline.

    Returns:
        The sample dictionary consumed by
        :func:`~extreme_weather.evaluation.evaluator.evaluate_gnn_baselines`.
    """
    generator = np.random.default_rng(seed)
    steps = np.arange(count, dtype="float64")

    previous_lat = 12.0 + 0.10 * steps
    previous_lon = 76.0 + 0.15 * steps
    current_lat = previous_lat + 0.10
    current_lon = previous_lon + 0.15
    true_lat = current_lat + 0.10
    true_lon = current_lon + 0.15

    current_intensity = 30.0 + generator.normal(0.0, 1.0, count)
    true_intensity = current_intensity + generator.normal(0.0, 0.5, count)

    predicted_lat = true_lat + generator.normal(0.0, model_skill * 0.02, count)
    predicted_lon = true_lon + generator.normal(0.0, model_skill * 0.02, count)
    predicted_intensity = true_intensity + generator.normal(0.0, model_skill * 0.5, count)

    samples: dict[str, Any] = {
        "current_latitude": current_lat,
        "current_longitude": current_lon,
        "current_intensity": current_intensity,
        "predicted_latitude": predicted_lat,
        "predicted_longitude": predicted_lon,
        "predicted_intensity": predicted_intensity,
        "true_latitude": true_lat,
        "true_longitude": true_lon,
        "true_intensity": true_intensity,
    }
    if constant_velocity:
        samples["previous_latitude"] = previous_lat
        samples["previous_longitude"] = previous_lon
    return samples


def make_event(
    event_id: str = "EXTREME_RAINFALL_20240630T1200Z_0",
    *,
    intensity: float = 55.0,
    anomaly_score: float = 4.2,
) -> ExtremeWeatherEvent:
    """Return a minimal but valid Phase 1 event."""
    return ExtremeWeatherEvent(
        event_id=event_id,
        event_type=EventType.EXTREME_RAINFALL,
        timestamp=REFERENCE_TIME,
        latitude=12.5,
        longitude=76.5,
        intensity=intensity,
        anomaly_score=anomaly_score,
        severity=Severity.SEVERE,
        affected_radius_km=42.0,
        cell_count=18,
        variable="precipitation",
        units="mm h-1",
        max_value=intensity,
        mean_value=intensity * 0.6,
    )


def make_track(
    event_id: str = "EXTREME_RAINFALL_20240630T1200Z_0",
    *,
    points: int = 4,
) -> EventTrack:
    """Return a valid Phase 2 track containing ``event_id``.

    The point geometry and the track-level summary statistics are derived from
    one another, because :class:`EventTrack` validates its own internal
    consistency - ``max_speed_kmh`` must not exceed the largest point speed.
    Hand-writing the summary numbers would produce an invalid track.
    """
    from extreme_weather.validation.ground_truth import haversine_km

    hour = timedelta(hours=1)
    base = REFERENCE_TIME - timedelta(hours=points - 1)
    step_speed = 12.0  # km/h, constant so the geometry is easy to verify

    distances = [step_speed * 1.0 for _ in range(max(0, points - 1))]
    track_points = tuple(
        TrackPoint(
            event_id=event_id if index == points - 1 else f"{event_id}_{index}",
            timestamp=base + hour * index,
            latitude=12.0 + 0.02 * index,
            longitude=76.0 + 0.03 * index,
            intensity=30.0 + 5.0 * index,
            anomaly_score=3.0 + 0.4 * index,
            severity=Severity.SEVERE,
            speed_kmh=0.0 if index == 0 else step_speed,
            bearing_degrees=None if index == 0 else 56.0,
            direction="CALM" if index == 0 else "NORTHEAST",
            distance_from_previous_km=0.0 if index == 0 else step_speed,
        )
        for index in range(points)
    )

    speeds = [point.speed_kmh for point in track_points if point.speed_kmh > 0.0]
    intensities = [point.intensity for point in track_points]
    return EventTrack(
        track_id="track-test-1",
        event_type=EventType.EXTREME_RAINFALL,
        status=TrackStatus.ACTIVE,
        start_time=track_points[0].timestamp,
        end_time=track_points[-1].timestamp,
        duration_hours=float(points - 1),
        points=track_points,
        peak_intensity=max(intensities),
        mean_intensity=sum(intensities) / len(intensities),
        max_speed_kmh=max(speeds) if speeds else 0.0,
        mean_speed_kmh=(sum(speeds) / len(speeds)) if speeds else 0.0,
        distance_travelled_km=sum(distances),
        intensity_trend="STRENGTHENING",
        direction="NORTHEAST",
        bearing_degrees=56.0,
        variable="precipitation",
        units="mm h-1",
        peak_severity=Severity.SEVERE,
    )


@pytest.fixture()
def samples() -> dict[str, np.ndarray]:
    """Return a structured synthetic sample set for the GNN comparisons."""
    return make_samples()


@pytest.fixture()
def validation_samples() -> dict[str, np.ndarray]:
    """Return a distinct validation sample set, so the splits are not aliased."""
    return make_samples(60, seed=23)


@pytest.fixture()
def event() -> ExtremeWeatherEvent:
    """Return a minimal detected event."""
    return make_event()


@pytest.fixture()
def track() -> EventTrack:
    """Return a valid track containing the fixture event."""
    return make_track()
