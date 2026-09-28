"""Extreme weather event endpoints (SIH26078, Phase 1).

These endpoints expose **real detector output**: they read the analysis file
written by ``scripts/detect_anomalies.py`` (or by
``AnomalyDetectionService.write_outputs``) through
:class:`~extreme_weather.anomaly.service.EventStore`. No events are ever
fabricated here: when no analysis exists the response is an empty list with an
explanatory ``message``.

```text
GET /api/v1/events                    # list + filters
GET /api/v1/events/{event_id}         # single event
```

Query filters: ``event_type``, ``severity``, ``start_time``, ``end_time`` and
``min_cell_count``. Invalid filter values are rejected by FastAPI with 422.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from extreme_weather.anomaly.service import AnalysisError, EventStore
from extreme_weather.config import Settings, get_settings
from extreme_weather.schemas.event import (
    EventListResponse,
    EventType,
    ExtremeWeatherEvent,
    Severity,
)

router = APIRouter(prefix="/api/v1/events", tags=["events"])

EMPTY_ANALYSIS_MESSAGE = (
    "No anomaly analysis available yet. Run 'python scripts/detect_anomalies.py' to generate one."
)


def get_event_store(settings: Settings = Depends(get_settings)) -> EventStore:
    """Dependency providing the file-backed event store."""
    return EventStore(settings.processed_dir / settings.events_filename)


StoreDependency = Annotated[EventStore, Depends(get_event_store)]


@router.get("", response_model=EventListResponse, summary="List detected extreme weather events")
def list_events(
    store: StoreDependency,
    event_type: Annotated[EventType | None, Query(description="Filter by event type.")] = None,
    severity: Annotated[Severity | None, Query(description="Filter by severity class.")] = None,
    start_time: Annotated[
        datetime | None, Query(description="Only events at or after this valid time (UTC).")
    ] = None,
    end_time: Annotated[
        datetime | None, Query(description="Only events at or before this valid time (UTC).")
    ] = None,
    min_cell_count: Annotated[
        int | None, Query(ge=1, description="Only events with at least this many grid cells.")
    ] = None,
) -> EventListResponse:
    """Return the events of the latest analysis, optionally filtered."""
    try:
        result = store.load()
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    if result is None:
        return EventListResponse(count=0, events=[], source=str(store.path), message=EMPTY_ANALYSIS_MESSAGE)

    events = list(result.events)
    if event_type is not None:
        events = [event for event in events if event.event_type is event_type]
    if severity is not None:
        events = [event for event in events if event.severity is severity]
    if start_time is not None:
        events = [event for event in events if event.timestamp >= start_time]
    if end_time is not None:
        events = [event for event in events if event.timestamp <= end_time]
    if min_cell_count is not None:
        events = [event for event in events if event.cell_count >= min_cell_count]

    return EventListResponse(
        count=len(events),
        events=events,
        source=str(store.path),
        generated_at=result.generated_at,
        message=None if events else "No event matches the given filters.",
    )


@router.get(
    "/{event_id}",
    response_model=ExtremeWeatherEvent,
    summary="Get one detected extreme weather event",
)
def get_event(event_id: str, store: StoreDependency) -> ExtremeWeatherEvent:
    """Return a single event by its deterministic identifier."""
    try:
        result = store.load()
    except AnalysisError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event '{event_id}' not found: {EMPTY_ANALYSIS_MESSAGE}",
        )

    for event in result.events:
        if event.event_id == event_id:
            return event

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"Event '{event_id}' not found. The analysis contains "
            f"{len(result.events)} event(s)."
        ),
    )
