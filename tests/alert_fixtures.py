"""Shared fixtures for the Phase 5 alerting tests (SIH26078).

The fixtures build a **real** Phase 1-2 pipeline over synthetic events (exactly
as ``tests/test_tracking_api.py`` does) so every assertion exercises the same
code path the CLI uses. Nothing in the alert tests is hand-written JSON: an
alert is always produced by the real service, and the stores are always
temporary folders.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from extreme_weather.alerts.engine import RiskConfig
from extreme_weather.alerts.models import Alert
from extreme_weather.alerts.notifications import (
    MockNotificationProvider,
    NotificationStore,
)
from extreme_weather.alerts.rules import RuleConfig
from extreme_weather.alerts.service import AlertService
from extreme_weather.alerts.store import AlertStore
from extreme_weather.anomaly.models import AnomalyAnalysisResult
from extreme_weather.anomaly.service import EventStore
from extreme_weather.config import Settings, get_settings
from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent, Severity
from extreme_weather.tracking.service import TrackingService
from extreme_weather.tracking.store import TrackStore

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)

#: Fixed timestamp used wherever a test needs a reproducible "now".
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_event(
    event_id: str,
    *,
    hour: int,
    latitude: float = 13.0,
    longitude: float = 77.0,
    intensity: float = 20.0,
    severity: Severity = Severity.MODERATE,
    event_type: EventType = EventType.EXTREME_RAINFALL,
    affected_radius_km: float = 40.0,
) -> ExtremeWeatherEvent:
    """Build a valid Phase 1 event used as alerting input."""
    return ExtremeWeatherEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=T0 + timedelta(hours=hour),
        latitude=latitude,
        longitude=longitude,
        intensity=intensity,
        anomaly_score=intensity,
        severity=severity,
        affected_radius_km=affected_radius_km,
        cell_count=12,
        variable="precipitation",
        units="mm h-1",
    )


def storm_events() -> tuple[ExtremeWeatherEvent, ...]:
    """A four-hour strengthening storm that the tracker links into one track."""
    return (
        make_event("E0", hour=0, intensity=20.0, severity=Severity.WATCH),
        make_event("E1", hour=1, longitude=77.10, intensity=30.0, severity=Severity.MODERATE),
        make_event("E2", hour=2, longitude=77.20, intensity=40.0, severity=Severity.SEVERE),
        make_event("E3", hour=3, longitude=77.30, intensity=45.0, severity=Severity.SEVERE),
    )


def write_events(path: Path, events: tuple[ExtremeWeatherEvent, ...]) -> None:
    """Write Phase 1 events the way the anomaly service would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    analysis = AnomalyAnalysisResult(events=events, variables=("precipitation",), generated_at=T0)
    path.write_text(json.dumps(analysis.as_dict()), encoding="utf-8")


@pytest.fixture()
def alert_settings(tmp_path: Path) -> Settings:
    """Settings whose data directories are a temporary folder."""
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    settings.ensure_directories()
    return settings


@pytest.fixture()
def seeded_pipeline(alert_settings: Settings) -> Settings:
    """Settings with a real Phase 1 + Phase 2 run already written to disk."""
    events_path = alert_settings.processed_dir / "events.json"
    write_events(events_path, storm_events())
    TrackingService(
        event_store=EventStore(events_path),
        track_store=TrackStore(alert_settings.processed_dir / "tracks.json"),
    ).run(write_output=True)
    return alert_settings


def build_service(
    settings: Settings,
    *,
    gnn_service: Any | None = None,
    persist_notifications: bool = True,
    rule_config: RuleConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> AlertService:
    """Build an :class:`AlertService` wired to the temporary stores."""
    notification_store = NotificationStore(
        settings.processed_dir / settings.notifications_filename
    )
    return AlertService(
        alert_store=AlertStore(settings.processed_dir / settings.alerts_filename),
        event_store=EventStore(settings.processed_dir / "events.json"),
        track_store=TrackStore(settings.processed_dir / "tracks.json"),
        gnn_service=gnn_service,
        notification_providers=[
            MockNotificationProvider(notification_store, persist=persist_notifications)
        ],
        risk_config=risk_config,
        rule_config=rule_config,
    )


@pytest.fixture()
def service(seeded_pipeline: Settings) -> AlertService:
    """A ready-to-use service over a real Phase 1 + Phase 2 pipeline."""
    return build_service(seeded_pipeline)


@pytest.fixture()
def generated_alerts(service: AlertService) -> list[Alert]:
    """Run the pipeline once and return the alerts it produced."""
    return list(service.generate(now=NOW).stored.alerts)


@pytest.fixture()
def alert_client(seeded_pipeline: Settings) -> Iterator[TestClient]:
    """FastAPI test client backed by a real Phase 1 + Phase 2 pipeline."""
    from extreme_weather.api.main import create_app

    application = create_app(seeded_pipeline)
    application.dependency_overrides[get_settings] = lambda: seeded_pipeline
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


@pytest.fixture()
def empty_alert_client(alert_settings: Settings) -> Iterator[TestClient]:
    """FastAPI test client whose processed directory holds no pipeline output."""
    from extreme_weather.api.main import create_app

    application = create_app(alert_settings)
    application.dependency_overrides[get_settings] = lambda: alert_settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()

