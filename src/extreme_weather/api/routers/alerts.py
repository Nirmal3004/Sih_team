"""Alerting and emergency-response endpoints (SIH26078, Phase 5).

These endpoints expose **real pipeline output**: alerts are read from the file
written by ``scripts/generate_alerts.py`` (or by ``AlertService.generate``).
Nothing is fabricated in the router - when no alert file exists the list is
empty with an explanatory ``message``, not an invented entry.

```text
GET  /api/v1/alerts                      # list + filters
GET  /api/v1/alerts/{alert_id}           # one alert with its timeline
POST /api/v1/alerts/generate             # run the generation pipeline
POST /api/v1/alerts/{alert_id}/acknowledge
POST /api/v1/alerts/{alert_id}/resolve
POST /api/v1/alerts/simulate             # deterministic demo walk-through
GET  /api/v1/alerts/summary              # dashboard counters
GET  /api/v1/notifications               # local notification log
```

Status codes
------------
=====  ==========================================================
200     Success.
404     The alert id is unknown, or no alert file exists at all.
422     Invalid filter value, invalid body, illegal lifecycle
        transition, or a corrupt store file.
503     ``POST /generate`` while ``ALERT_GENERATION_ENABLED`` is off -
        a genuinely unavailable capability, not a fake success.
=====  ==========================================================

Every response carries the prototype notice, so a client cannot display an
alert severity without also showing that it is not an official warning.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status

from extreme_weather.alerts.notifications import (
    MockNotificationProvider,
    NotificationError,
    NotificationStore,
)
from extreme_weather.alerts.schemas import (
    AcknowledgeRequest,
    AlertActionResponse,
    AlertDetailResponse,
    AlertGenerateRequest,
    AlertGenerateResponse,
    AlertListResponse,
    AlertSummaryResponse,
    NotificationListResponse,
    ResolveRequest,
    SimulationRequest,
    SimulationResponse,
)
from extreme_weather.alerts.severity import AlertSeverity
from extreme_weather.alerts.service import (
    AlertError,
    AlertService,
    AlertTransitionError,
    AlertUnknownError,
)
from extreme_weather.alerts.simulation import run_simulation
from extreme_weather.alerts.store import AlertStore, AlertStoreError
from extreme_weather.config import Settings, get_settings
from extreme_weather.schemas.event import EventType

router = APIRouter(prefix="/api/v1", tags=["alerts"])

EMPTY_ALERTS_MESSAGE = (
    "No alerts available yet. Run 'python scripts/generate_alerts.py' after "
    "'python scripts/detect_anomalies.py' and 'python scripts/track_events.py'."
)

GENERATION_DISABLED_MESSAGE = (
    "Alert generation is disabled (ALERT_GENERATION_ENABLED is false). The stored "
    "alerts are still served; enable generation in .env to create new ones."
)


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def get_alert_service(settings: Settings = Depends(get_settings)) -> AlertService:
    """Dependency providing the alert service with the configured paths.

    Every store is passed explicitly. Constructing them without a path would
    fall back to ``get_settings()`` - the *process-wide* settings - which would
    make the service read and write the real data directory even when the
    application was created with an overridden configuration (as the tests do).
    """
    from extreme_weather.alerts.notifications import NotificationStore
    from extreme_weather.anomaly.service import EventStore
    from extreme_weather.tracking.store import TrackStore

    notification_store = NotificationStore(
        settings.processed_dir / settings.notifications_filename
    )
    return AlertService(
        alert_store=AlertStore(settings.processed_dir / settings.alerts_filename),
        event_store=EventStore(settings.processed_dir / settings.events_filename),
        track_store=TrackStore(settings.processed_dir / settings.tracks_filename),
        notification_providers=[MockNotificationProvider(notification_store)],
    )


def get_notification_store(settings: Settings = Depends(get_settings)) -> NotificationStore:
    """Dependency providing the local notification log."""
    return NotificationStore(settings.processed_dir / settings.notifications_filename)


ServiceDependency = Annotated[AlertService, Depends(get_alert_service)]
NotificationStoreDependency = Annotated[NotificationStore, Depends(get_notification_store)]


def _http_error(exc: Exception) -> HTTPException:
    """Translate a service error into the right HTTP status code."""
    if isinstance(exc, AlertUnknownError):
        return HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, AlertTransitionError):
        return HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    if isinstance(exc, (AlertError, AlertStoreError)):
        return HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return HTTPException(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/alerts
# --------------------------------------------------------------------------- #
@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List extreme weather alerts (Phase 5)",
)
def list_alerts(
    service: ServiceDependency,
    status: Annotated[
        str | None,
        Query(description="Filter by lifecycle status (NEW, ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED)."),
    ] = None,
    severity: Annotated[
        str | None, Query(description="Filter by alert severity.")
    ] = None,
    event_type: Annotated[
        EventType | None, Query(description="Filter by the Phase 1 event type.")
    ] = None,
    start_time: Annotated[
        datetime | None, Query(description="Only alerts created at or after this UTC time.")
    ] = None,
    end_time: Annotated[
        datetime | None, Query(description="Only alerts created at or before this UTC time.")
    ] = None,
    priority: Annotated[
        str | None, Query(description="Filter by operator priority.")
    ] = None,
) -> AlertListResponse:
    """Return the stored alerts, optionally filtered.

    Invalid filter values are rejected by FastAPI / the store with 422 - the
    endpoint never silently ignores a filter it does not understand.
    """
    try:
        result = service.alert_store.load()
        if result is None:
            return AlertListResponse(
                count=0,
                alerts=[],
                source=str(service.alert_store.path),
                message=EMPTY_ALERTS_MESSAGE,
            )
        alerts = service.alert_store.list(
            status=status,
            severity=severity,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            priority=priority,
        )
    except AlertStoreError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return AlertListResponse(
        count=len(alerts),
        alerts=alerts,
        source=str(service.alert_store.path),
        generated_at=result.generated_at,
        message=None if alerts else "No alert matches the given filters.",
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/alerts/summary
# --------------------------------------------------------------------------- #
@router.get(
    "/alerts/summary",
    response_model=AlertSummaryResponse,
    summary="Headline alert counters for the dashboard (Phase 5)",
)
def alert_summary(service: ServiceDependency) -> AlertSummaryResponse:
    """Return the dashboard counters. Never fails on an empty store."""
    return AlertSummaryResponse(summary=service.summary())


# --------------------------------------------------------------------------- #
# GET /api/v1/alerts/{alert_id}
# --------------------------------------------------------------------------- #
@router.get(
    "/alerts/{alert_id}",
    response_model=AlertDetailResponse,
    summary="Get one alert with its lifecycle timeline (Phase 5)",
)
def get_alert(alert_id: str, service: ServiceDependency) -> AlertDetailResponse:
    """Return one alert plus the timeline needed by the details view.

    Declared after ``/summary`` so the literal path is not captured by the
    ``{alert_id}`` parameter.
    """
    from extreme_weather.alerts.simulation import build_timeline

    try:
        alert = service.get_alert(alert_id)
    except AlertStoreError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except AlertUnknownError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"{exc} {EMPTY_ALERTS_MESSAGE}",
        ) from exc
    return AlertDetailResponse(alert=alert, timeline=build_timeline(alert))



# --------------------------------------------------------------------------- #
# POST /api/v1/alerts/generate
# --------------------------------------------------------------------------- #
@router.post(
    "/alerts/generate",
    response_model=AlertGenerateResponse,
    summary="Run the alert generation pipeline (Phase 5)",
)
def generate_alerts(
    request: AlertGenerateRequest,
    service: ServiceDependency,
    settings: Settings = Depends(get_settings),
) -> AlertGenerateResponse:
    """Read the Phase 1-4 outputs, score them, apply the rules and store alerts.

    With ``dry_run: true`` neither the alert store nor the notification log is
    written - the response reports what would have happened and nothing is
    touched on disk.
    """
    if not settings.alert_generation_enabled:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=GENERATION_DISABLED_MESSAGE,
        )
    try:
        if request.expire_stale and not request.dry_run:
            service.expire_alerts()
        result = service.generate(dry_run=request.dry_run, notify=request.notify)
    except AlertError as exc:
        raise _http_error(exc) from exc
    except AlertStoreError as exc:
        raise _http_error(exc) from exc

    return AlertGenerateResponse(
        dry_run=result.dry_run,
        created_count=result.created_count,
        updated_count=result.updated_count,
        suppressed_count=result.suppressed_count,
        alert_count=result.alert_count,
        alerts=result.stored.alerts,
        notification_count=(
            len(result.notifications.records) if result.notifications else 0
        ),
        degraded=list(result.degraded),
        source=str(service.alert_store.path),
    )


# --------------------------------------------------------------------------- #
# POST /api/v1/alerts/{alert_id}/acknowledge
# --------------------------------------------------------------------------- #
@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertActionResponse,
    summary="Acknowledge an alert (operator action, Phase 5)",
)
def acknowledge_alert(
    alert_id: str,
    request: AcknowledgeRequest,
    service: ServiceDependency,
) -> AlertActionResponse:
    """Record that an operator has seen the alert and taken it on.

    Raises 404 for an unknown alert and 422 for an illegal transition (for
    example acknowledging an already-resolved alert).
    """
    try:
        alert = service.acknowledge_alert(alert_id, note=request.note)
    except (AlertUnknownError, AlertTransitionError, AlertError, AlertStoreError) as exc:
        raise _http_error(exc) from exc
    return AlertActionResponse(
        alert=alert,
        message=(
            f"Alert '{alert_id}' acknowledged. This records an operator action on a "
            "prototype alert; it is not a public warning acknowledgement."
        ),
    )


# --------------------------------------------------------------------------- #
# POST /api/v1/alerts/{alert_id}/resolve
# --------------------------------------------------------------------------- #
@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertActionResponse,
    summary="Resolve an alert (operator action, Phase 5)",
)
def resolve_alert(
    alert_id: str,
    request: ResolveRequest,
    service: ServiceDependency,
) -> AlertActionResponse:
    """Close an alert. ``RESOLVED`` is terminal - it cannot be reopened."""
    try:
        alert = service.resolve_alert(alert_id, reason=request.reason)
    except (AlertUnknownError, AlertTransitionError, AlertError, AlertStoreError) as exc:
        raise _http_error(exc) from exc
    return AlertActionResponse(
        alert=alert,
        message=f"Alert '{alert_id}' resolved.",
    )



# --------------------------------------------------------------------------- #
# POST /api/v1/alerts/simulate
# --------------------------------------------------------------------------- #
@router.post(
    "/alerts/simulate",
    response_model=SimulationResponse,
    summary="Deterministic simulation of the alert lifecycle (Phase 5)",
)
def simulate_alerts(
    request: SimulationRequest,
    service: ServiceDependency,
    settings: Settings = Depends(get_settings),
) -> SimulationResponse:
    """Walk a synthetic event through the full alert lifecycle.

    This is a **demonstration on synthetic data**, labelled as such in the
    response. It reuses the real service, rules, store and notification
    adapters, and it is deterministic: the same inputs always produce the same
    steps, ids and scores.
    """
    if not settings.alert_generation_enabled:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=GENERATION_DISABLED_MESSAGE,
        )
    # A simulated downscaled threat field is attached to the stored synthetic
    # events, so the demo exercises the "downscaling contributed" path without
    # inventing a model output.
    threat_lookup: dict[str, tuple[float, float | None]] = {}
    try:
        events = service.event_store.load()
    except Exception:  # a missing/corrupt event file simply means no threats
        events = None
    if events is not None:
        for index, event in enumerate(events.events):
            threat_lookup[event.event_id] = (event.intensity, 5.0)

    try:
        result = run_simulation(
            service,
            steps=request.steps,
            acknowledge_index=request.acknowledge_index,
            resolve_index=request.resolve_index,
            dry_run=request.dry_run,
            threat_lookup=threat_lookup,
        )
    except (AlertError, AlertStoreError, ValueError) as exc:
        raise _http_error(exc) from exc

    return SimulationResponse(
        steps=list(result.steps),
        alerts=list(result.alerts),
        notification_count=result.notification_count,
        dry_run=result.dry_run,
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/notifications
# --------------------------------------------------------------------------- #
@router.get(
    "/notifications",
    response_model=NotificationListResponse,
    summary="Local notification log (Phase 5)",
)
def list_notifications(
    store: NotificationStoreDependency,
    limit: Annotated[
        int | None, Query(ge=1, description="Return only the most recent N records.")
    ] = None,
) -> NotificationListResponse:
    """Return the notifications that were recorded locally.

    Nothing here was delivered to an external service: the only enabled provider
    writes a local JSON log. A missing log is an empty result with a message,
    not an error.
    """
    try:
        result = store.load()
    except NotificationError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    records = result.records
    if limit is not None:
        records = records[-limit:]
    return NotificationListResponse(
        count=len(records),
        records=records,
        source=result.source,
        message=(
            None
            if records
            else "No notifications recorded yet. Run 'python scripts/generate_alerts.py'."
        ),
    )

