"""Event tracking endpoints (SIH26078, Phase 2).

These endpoints expose **real tracker output**: they read the file written by
``scripts/track_events.py`` (or by ``TrackingService.run``) through
:class:`~extreme_weather.tracking.store.TrackStore`. No track is ever fabricated
here - when no tracking file exists the response is an empty list with an
explanatory ``message``.

```text
GET /api/v1/tracks                          # list + filters
GET /api/v1/tracks/{track_id}               # one track
GET /api/v1/tracks/{track_id}/trajectory    # positions for the map layer
```

Query filters: ``event_type``, ``status``, ``start_time``, ``end_time``. Invalid
filter values are rejected by FastAPI with 422.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status

from extreme_weather.config import Settings, get_settings
from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.track import (
    EventTrack,
    TrackListResponse,
    TrackStatus,
    TrajectoryResponse,
    TrajectoryWaypoint,
)
from extreme_weather.tracking.store import TrackStore, TrackingStoreError

router = APIRouter(prefix="/api/v1/tracks", tags=["tracks"])

EMPTY_TRACKING_MESSAGE = (
    "No event tracking available yet. Run 'python scripts/detect_anomalies.py' and then "
    "'python scripts/track_events.py'."
)


def get_track_store(settings: Settings = Depends(get_settings)) -> TrackStore:
    """Dependency providing the file-backed track store."""
    return TrackStore(settings.processed_dir / settings.tracks_filename)


StoreDependency = Annotated[TrackStore, Depends(get_track_store)]


def _load_or_503(store: TrackStore):
    """Load the stored result, translating store errors into HTTP responses."""
    try:
        return store.load()
    except TrackingStoreError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get("", response_model=TrackListResponse, summary="List tracked extreme weather events")
def list_tracks(
    store: StoreDependency,
    event_type: Annotated[
        EventType | None, Query(description="Filter by event type.")
    ] = None,
    status: Annotated[
        TrackStatus | None, Query(description="Filter by lifecycle status.")
    ] = None,
    start_time: Annotated[
        datetime | None, Query(description="Only tracks that end at or after this time (UTC).")
    ] = None,
    end_time: Annotated[
        datetime | None, Query(description="Only tracks that start at or before this time (UTC).")
    ] = None,
) -> TrackListResponse:
    """Return the tracks of the latest tracking run, optionally filtered."""
    result = _load_or_503(store)
    if result is None:
        return TrackListResponse(
            count=0,
            tracks=[],
            source=str(store.path),
            message=EMPTY_TRACKING_MESSAGE,
        )

    try:
        tracks = store.list(
            event_type=event_type, status=status, start_time=start_time, end_time=end_time
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return TrackListResponse(
        count=len(tracks),
        tracks=tracks,
        source=str(store.path),
        generated_at=result.generated_at,
        message=None if tracks else "No track matches the given filters.",
    )


def _require_track(store: TrackStore, track_id: str) -> EventTrack:
    """Fetch one track or raise a 404 with a helpful message."""
    result = _load_or_503(store)
    if result is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Track '{track_id}' not found: {EMPTY_TRACKING_MESSAGE}",
        )

    for track in result.tracks:
        if track.track_id == track_id:
            return track

    raise HTTPException(
        status_code=http_status.HTTP_404_NOT_FOUND,
        detail=(
            f"Track '{track_id}' not found. The tracking file contains "
            f"{len(result.tracks)} track(s)."
        ),
    )


@router.get("/{track_id}", response_model=EventTrack, summary="Get one tracked event")
def get_track(track_id: str, store: StoreDependency) -> EventTrack:
    """Return a single track by its deterministic identifier."""
    return _require_track(store, track_id)


@router.get(
    "/{track_id}/trajectory",
    response_model=TrajectoryResponse,
    summary="Trajectory positions of one tracked event",
)
def get_track_trajectory(track_id: str, store: StoreDependency) -> TrajectoryResponse:
    """Return the ordered positions of a track (consumed by the map layer)."""
    track = _require_track(store, track_id)
    return TrajectoryResponse(
        track_id=track.track_id,
        event_type=track.event_type,
        status=track.status,
        point_count=track.point_count,
        distance_travelled_km=track.distance_travelled_km,
        trajectory=[
            TrajectoryWaypoint(
                timestamp=point.timestamp,
                longitude=point.longitude,
                latitude=point.latitude,
                intensity=point.intensity,
                speed_kmh=point.speed_kmh,
                direction=point.direction,
                event_id=point.event_id,
            )
            for point in track.points
        ],
    )
