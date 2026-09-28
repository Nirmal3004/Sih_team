"""Tests for the Phase 5 deterministic alert simulation (SIH26078).

The simulation exists to demonstrate the lifecycle, so the properties that
matter are: it is deterministic, it is labelled as synthetic everywhere, it
never invents a timestamp, and it reuses the real service (so a passing test
here means the real acknowledge/resolve path works).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from extreme_weather.alerts.models import AlertStatus
from extreme_weather.alerts.service import AlertService
from extreme_weather.alerts.simulation import (
    SIMULATION_LABEL,
    SIMULATION_START,
    build_timeline,
    run_simulation,
)
from extreme_weather.config import Settings

from tests.alert_fixtures import build_service


def threat_lookup() -> dict[str, tuple[float, float | None]]:
    """A simulated ~5 km threat value for the newest stored event."""
    return {"E3": (45.0, 5.0)}


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_simulation_is_deterministic(seeded_pipeline: Settings) -> None:
    first = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    second = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert [(s.stage, s.alert_id) for s in first.steps] == [
        (s.stage, s.alert_id) for s in second.steps
    ]
    assert [a.alert_id for a in first.alerts] == [a.alert_id for a in second.alerts]
    assert [a.risk.score for a in first.alerts] == [a.risk.score for a in second.alerts]


def test_simulation_uses_a_fixed_timeline_anchor(seeded_pipeline: Settings) -> None:
    """It must not depend on the wall clock."""
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert result.steps[0].timestamp == SIMULATION_START


def test_simulation_is_labelled_as_synthetic(seeded_pipeline: Settings) -> None:
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert result.label == SIMULATION_LABEL
    assert "SYNTHETIC" in result.label


# --------------------------------------------------------------------------- #
# Lifecycle walk-through
# --------------------------------------------------------------------------- #
def test_simulation_walks_create_acknowledge_resolve(seeded_pipeline: Settings) -> None:
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert [step.stage for step in result.steps] == [
        "alert_created",
        "acknowledged",
        "resolved",
    ]
    assert result.steps[1].status is AlertStatus.ACKNOWLEDGED
    assert result.steps[2].status is AlertStatus.RESOLVED


def test_simulation_steps_carry_real_timestamps(seeded_pipeline: Settings) -> None:
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert all(step.timestamp is not None for step in result.steps)
    stamps = [step.timestamp for step in result.steps]
    assert stamps == sorted(stamps)


def test_simulation_can_be_replayed_over_an_unchanged_store(
    seeded_pipeline: Settings,
) -> None:
    """A second run must not fail on the already-closed alert of the first."""
    service = build_service(seeded_pipeline)
    run_simulation(service, threat_lookup=threat_lookup())
    again = run_simulation(service, threat_lookup=threat_lookup())
    assert [step.stage for step in again.steps] == [
        "alert_created",
        "acknowledged",
        "resolved",
    ]


def test_simulation_resolves_the_alert_it_acknowledged(seeded_pipeline: Settings) -> None:
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert result.alerts
    alert = result.alerts[0]
    assert alert.status is AlertStatus.RESOLVED
    assert alert.acknowledged_at is not None
    assert alert.resolved_at is not None
    assert "simulation" in (alert.resolved_reason or "")


def test_simulation_uses_the_attached_downscaled_threat(
    seeded_pipeline: Settings,
) -> None:
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    alert = next(a for a in result.alerts if a.event_id == "E3")
    assert "downscaling" in alert.source_components
    assert "no AI-downscaled threat field" not in alert.risk.explanation


def test_simulation_records_notifications_through_the_mock(
    seeded_pipeline: Settings,
) -> None:
    result = run_simulation(build_service(seeded_pipeline), threat_lookup=threat_lookup())
    assert result.notification_count > 0


def test_simulation_rejects_an_inverted_lifecycle(seeded_pipeline: Settings) -> None:
    with pytest.raises(ValueError, match="must be greater than acknowledge_index"):
        run_simulation(
            build_service(seeded_pipeline),
            acknowledge_index=4,
            resolve_index=2,
            threat_lookup=threat_lookup(),
        )


# --------------------------------------------------------------------------- #
# Dry run and empty inputs
# --------------------------------------------------------------------------- #
def test_simulation_dry_run_writes_nothing(seeded_pipeline: Settings) -> None:
    result = run_simulation(
        build_service(seeded_pipeline), dry_run=True, threat_lookup=threat_lookup()
    )
    assert result.dry_run is True
    assert result.notification_count == 0
    assert (
        seeded_pipeline.processed_dir / seeded_pipeline.alerts_filename
    ).exists() is False
    assert any("dry run" in step.description.lower() for step in result.steps[1:])


def test_simulation_without_events_explains_itself(alert_settings: Settings) -> None:
    result = run_simulation(build_service(alert_settings))
    assert len(result.steps) == 1
    assert result.steps[0].stage == "no_input"
    assert "detect_anomalies.py" in result.steps[0].description
    assert result.alerts == ()


# --------------------------------------------------------------------------- #
# Timeline builder
# --------------------------------------------------------------------------- #
def test_build_timeline_covers_every_stage(service: AlertService) -> None:
    alert = service.generate().stored.alerts[0]
    assert [step.stage for step in build_timeline(alert)] == [
        "detected",
        "tracked",
        "predicted",
        "downscaled",
        "alert_created",
    ]


def test_build_timeline_marks_absent_stages_with_no_timestamp(
    service: AlertService,
) -> None:
    alert = service.generate().stored.alerts[0]
    by_stage = {step.stage: step for step in build_timeline(alert)}
    assert "gnn" not in alert.source_components
    assert by_stage["predicted"].timestamp is None
    assert "no gnn prediction" in by_stage["predicted"].description.lower()
    assert by_stage["downscaled"].timestamp is None


def test_build_timeline_shows_a_downscaled_stage_when_present(
    service: AlertService,
) -> None:
    alert = service.generate().stored.alerts[0]
    refreshed = alert.model_copy(update={"source_components": [*alert.source_components, "downscaling"]})
    by_stage = {step.stage: step for step in build_timeline(refreshed)}
    assert by_stage["downscaled"].timestamp is not None


def test_build_timeline_adds_acknowledged_and_resolved_once_recorded(
    service: AlertService,
) -> None:
    alert = service.generate().stored.alerts[0]
    service.acknowledge_alert(alert.alert_id)
    service.resolve_alert(alert.alert_id)
    timeline = build_timeline(service.get_alert(alert.alert_id))
    assert [step.stage for step in timeline][-2:] == ["acknowledged", "resolved"]
    assert timeline[-1].status is AlertStatus.RESOLVED


# --------------------------------------------------------------------------- #
# API endpoint
# --------------------------------------------------------------------------- #
def test_simulate_endpoint_returns_the_labelled_walkthrough(alert_client: TestClient) -> None:
    response = alert_client.post("/api/v1/alerts/simulate", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "SIMULATION - SYNTHETIC DATA"
    assert payload["deterministic"] is True
    assert "not a real weather event" in payload["disclaimer"]
    assert [step["stage"] for step in payload["steps"]] == [
        "alert_created",
        "acknowledged",
        "resolved",
    ]


def test_simulate_endpoint_is_deterministic(alert_client: TestClient) -> None:
    first = alert_client.post("/api/v1/alerts/simulate", json={}).json()
    second = alert_client.post("/api/v1/alerts/simulate", json={}).json()
    assert [s["stage"] for s in first["steps"]] == [s["stage"] for s in second["steps"]]
    assert first["steps"][0]["alert_id"] == second["steps"][0]["alert_id"]


def test_simulate_endpoint_validates_its_body(alert_client: TestClient) -> None:
    assert alert_client.post("/api/v1/alerts/simulate", json={"steps": 0}).status_code == 422
    assert (
        alert_client.post("/api/v1/alerts/simulate", json={"steps": 99}).status_code == 422
    )
    assert (
        alert_client.post("/api/v1/alerts/simulate", json={"bogus": 1}).status_code == 422
    )


def test_simulate_endpoint_rejects_an_inverted_lifecycle(alert_client: TestClient) -> None:
    response = alert_client.post(
        "/api/v1/alerts/simulate", json={"acknowledge_index": 4, "resolve_index": 2}
    )
    assert response.status_code == 422
    assert "acknowledge_index" in response.json()["detail"]


def test_simulate_endpoint_dry_run_writes_nothing(
    alert_client: TestClient, alert_settings: Settings
) -> None:
    response = alert_client.post("/api/v1/alerts/simulate", json={"dry_run": True})
    assert response.status_code == 200
    assert response.json()["dry_run"] is True
    assert (alert_settings.processed_dir / alert_settings.alerts_filename).exists() is False


def test_simulate_endpoint_without_events_reports_no_input(
    empty_alert_client: TestClient,
) -> None:
    response = empty_alert_client.post("/api/v1/alerts/simulate", json={})
    assert response.status_code == 200
    assert response.json()["steps"][0]["stage"] == "no_input"

