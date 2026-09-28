"""Shared deterministic fixtures for the GNN tests (SIH26078, Phase 3).

Every fixture here is tiny and hand-built - no file I/O, no randomness, no
network - so the GNN tests are fast and reproducible. Tracks are constructed
through the real Phase 2 :class:`EventTrack` validator, which means the GNN code
is exercised against exactly the same schema the pipeline produces; if Phase 2
ever changes its contract these fixtures fail loudly rather than silently
diverging.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from extreme_weather.schemas.event import EventType, Severity
from extreme_weather.schemas.track import EventTrack, TrackPoint, TrackStatus
from extreme_weather.tracking.metrics import (
    bearing_degrees,
    compass_direction,
    haversine_distance_km,
    movement_speed_kmh,
)

BASE_TIME = datetime(2024, 6, 2, 0, 0, tzinfo=timezone.utc)


def build_points(
    track_id: str,
    positions: list[tuple[float, float, float]],
    start: datetime = BASE_TIME,
    step_hours: float = 1.0,
) -> list[TrackPoint]:
    """Build the points of a track, filling in the Phase 2 movement metrics.

    ``positions`` holds ``(latitude, longitude, intensity)`` triples. The speed,
    bearing and compass direction of each point are derived exactly as the
    tracker derives them, so the fixtures are internally consistent.
    """
    points: list[TrackPoint] = []
    for index, (latitude, longitude, intensity) in enumerate(positions):
        timestamp = start + timedelta(hours=step_hours * index)
        if index == 0:
            speed, bearing, direction = 0.0, None, "CALM"
        else:
            previous = positions[index - 1]
            distance = haversine_distance_km(previous[0], previous[1], latitude, longitude)
            speed = movement_speed_kmh(distance, step_hours)
            bearing = bearing_degrees(previous[0], previous[1], latitude, longitude)
            direction = compass_direction(bearing)
        points.append(
            TrackPoint(
                event_id=f"{track_id}-E{index:03d}",
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                intensity=intensity,
                anomaly_score=0.75,
                severity=Severity.SEVERE,
                speed_kmh=speed,
                bearing_degrees=bearing,
                direction=direction,
                cell_count=10,
            )
        )
    return points


def build_track(
    track_id: str,
    positions: list[tuple[float, float, float]],
    start: datetime = BASE_TIME,
    step_hours: float = 1.0,
    event_type: EventType = EventType.EXTREME_RAINFALL,
    end_status: TrackStatus = TrackStatus.ENDED,
) -> EventTrack:
    """Assemble a valid :class:`EventTrack` from ``(lat, lon, intensity)`` triples.

    The derived aggregates are computed from the points so the model validator's
    internal-consistency checks pass.
    """
    points = build_points(track_id, positions, start, step_hours)
    distances = [0.0] * len(points)
    for index in range(1, len(points)):
        previous, current = points[index - 1], points[index]
        distance = haversine_distance_km(
            previous.latitude, previous.longitude, current.latitude, current.longitude
        )
        distances[index] = distance
        points[index] = points[index].model_copy(
            update={"distance_from_previous_km": distance}
        )

    speeds = [point.speed_kmh for point in points]
    return EventTrack(
        track_id=track_id,
        event_type=event_type,
        status=end_status if len(points) > 1 else TrackStatus.NEW,
        start_time=points[0].timestamp,
        end_time=points[-1].timestamp,
        duration_hours=step_hours * (len(points) - 1),
        points=points,
        peak_intensity=max(point.intensity for point in points),
        mean_intensity=sum(point.intensity for point in points) / len(points),
        max_speed_kmh=max(speeds),
        mean_speed_kmh=sum(speeds) / len(speeds),
        distance_travelled_km=sum(distances),
        intensity_trend="STABLE",
        direction=points[-1].direction,
        bearing_degrees=points[-1].bearing_degrees,
        variable="precipitation",
        units="mm h-1",
        peak_severity=Severity.SEVERE,
    )


@pytest.fixture
def single_track() -> EventTrack:
    """One 5-point track moving north-east, hourly."""
    return build_track(
        "TRK-TEST-A",
        [
            (13.00, 77.00, 20.0),
            (13.01, 77.01, 25.0),
            (13.02, 77.02, 30.0),
            (13.03, 77.03, 35.0),
            (13.04, 77.04, 40.0),
        ],
    )


@pytest.fixture
def two_nearby_tracks() -> list[EventTrack]:
    """Two concurrent tracks ~7.8 km apart, so a spatial edge is created."""
    return [
        build_track(
            "TRK-TEST-A", [(13.00, 77.00, 20.0), (13.01, 77.01, 25.0), (13.02, 77.02, 30.0)]
        ),
        build_track(
            "TRK-TEST-B", [(13.05, 77.05, 12.0), (13.06, 77.06, 14.0), (13.07, 77.07, 16.0)]
        ),
    ]


@pytest.fixture
def two_far_tracks() -> list[EventTrack]:
    """Two concurrent tracks ~1100 km apart, so no spatial edge is created."""
    return [
        build_track(
            "TRK-TEST-A", [(13.00, 77.00, 20.0), (13.01, 77.01, 25.0), (13.02, 77.02, 30.0)]
        ),
        build_track(
            "TRK-TEST-B", [(23.00, 87.00, 12.0), (23.01, 87.01, 14.0), (23.02, 87.02, 16.0)]
        ),
    ]


@pytest.fixture
def many_tracks() -> list[EventTrack]:
    """Four tracks of 10 points each, spread over 9 hourly steps.

    This is the "tiny synthetic fixture" used for the training test: 4 * 9 = 36
    prediction samples, enough for a chronological 60/20/20 split to give every
    block at least one sample.
    """
    tracks: list[EventTrack] = []
    for track_index, base in enumerate([(13.0, 77.0), (15.0, 79.0), (11.0, 75.0), (14.0, 78.0)]):
        positions = [
            (base[0] + 0.01 * step, base[1] + 0.01 * step, 20.0 + track_index + step)
            for step in range(10)
        ]
        tracks.append(build_track(f"TRK-SYN-{track_index:03d}", positions))
    return tracks
