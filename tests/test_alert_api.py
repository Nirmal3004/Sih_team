"""Tests for the Phase 5 alerts API (SIH26078).

The client is backed by a real Phase 1 + Phase 2 pipeline, so every response
comes from the same code path the CLI uses. The status-code contract (404, 422,
503) is asserted explicitly: the API must never return a fake success.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from extreme_weather.config import Settings

UTC = timezone.utc


def generate(client: TestClient, **body: object) -> dict:
    """POST /api/v1/alerts/generate and return the JSON payload."""
    response = client.post("/api/v1/alerts/generate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def alerts_ready(alert_client: TestClient) -> TestClient:
    """A client whose alert store has been populated by a real run."""
    generate(alert_client)
    return alert_client


# --------------------------------------------------------------------------- #
# GET /api/v1/alerts
# --------------------------------------------------------------------------- #
def test_empty_store_returns_an_explanatory_empty_list(empty_alert_client: TestClient) -> None:
    response = empty_alert_client.get("/api/v1/alerts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert payload["alerts"] == []
    assert "generate_alerts.py" in payload["message"]


def test_list_returns_the_generated_alerts(alerts_ready: TestClient) -> None:
    response = alerts_ready.get("/api/v1/alerts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] > 0
    assert len(payload["alerts"]) == payload["count"]
    assert payload["generated_at"] is not None


def test_every_alert_response_carries_the_prototype_notice(alerts_ready: TestClient) -> None:
    payload = alerts_ready.get("/api/v1/alerts").json()
    assert "NOT an official warning" in payload["notice"]
    for alert in payload["alerts"]:
        assert "NOT an official warning" in alert["notice"]
        assert alert["risk"]["explanation"]
        assert "not a validated disaster-risk index" in alert["risk"]["notice"]


def test_list_filters_by_status(alerts_ready: TestClient) -> None:
    target = alerts_ready.get("/api/v1/alerts").json()["alerts"][0]
    payload = alerts_ready.get(f"/api/v1/alerts?status={target['status']}").json()
    assert all(alert["status"] == target["status"] for alert in payload["alerts"])


def test_list_filters_by_severity(alerts_ready: TestClient) -> None:
    target = alerts_ready.get("/api/v1/alerts").json()["alerts"][0]
    payload = alerts_ready.get(f"/api/v1/alerts?severity={target['severity']}").json()
    assert all(alert["severity"] == target["severity"] for alert in payload["alerts"])


def test_list_filters_by_priority(alerts_ready: TestClient) -> None:
    target = alerts_ready.get("/api/v1/alerts").json()["alerts"][0]
    payload = alerts_ready.get(f"/api/v1/alerts?priority={target['priority']}").json()
    assert all(alert["priority"] == target["priority"] for alert in payload["alerts"])


def test_list_filters_by_event_type(alerts_ready: TestClient) -> None:
    payload = alerts_ready.get("/api/v1/alerts?event_type=EXTREME_RAINFALL").json()
    assert all(alert["event_type"] == "EXTREME_RAINFALL" for alert in payload["alerts"])


def test_list_filters_by_time_window(alerts_ready: TestClient) -> None:
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    response = alerts_ready.get("/api/v1/alerts", params={"start_time": future})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert "No alert matches" in payload["message"]


def test_list_accepts_a_past_time_window(alerts_ready: TestClient) -> None:
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    payload = alerts_ready.get("/api/v1/alerts", params={"start_time": past}).json()
    assert payload["count"] > 0


def test_list_rejects_an_unparsable_time(alerts_ready: TestClient) -> None:
    response = alerts_ready.get("/api/v1/alerts", params={"start_time": "yesterday"})
    assert response.status_code == 422


def test_list_rejects_an_invalid_filter_value(alerts_ready: TestClient) -> None:
    assert alerts_ready.get("/api/v1/alerts?severity=IMD_RED").status_code == 422
    assert alerts_ready.get("/api/v1/alerts?status=CLOSED").status_code == 422
    assert alerts_ready.get("/api/v1/alerts?event_type=HURRICANE").status_code == 422


def test_corrupt_alert_file_is_reported_as_422(alert_settings: Settings) -> None:
    from extreme_weather.api.main import create_app
    from extreme_weather.config import get_settings

    (alert_settings.processed_dir / alert_settings.alerts_filename).write_text(
        "{broken", encoding="utf-8"
    )
    application = create_app(alert_settings)
    application.dependency_overrides[get_settings] = lambda: alert_settings
    try:
        with TestClient(application) as client:
            response = client.get("/api/v1/alerts")
    finally:
        application.dependency_overrides.clear()
    assert response.status_code == 422
    assert "Could not read alert file" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# GET /api/v1/alerts/{alert_id}
# --------------------------------------------------------------------------- #
def test_get_one_alert_includes_its_timeline(alerts_ready: TestClient) -> None:
    target = alerts_ready.get("/api/v1/alerts").json()["alerts"][0]
    response = alerts_ready.get(f"/api/v1/alerts/{target['alert_id']}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["alert"]["alert_id"] == target["alert_id"]
    stages = [step["stage"] for step in payload["timeline"]]
    assert stages[:5] == [
        "detected",
        "tracked",
        "predicted",
        "downscaled",
        "alert_created",
    ]


def test_timeline_says_when_a_stage_did_not_happen(alerts_ready: TestClient) -> None:
    target = alerts_ready.get("/api/v1/alerts").json()["alerts"][0]
    timeline = alerts_ready.get(f"/api/v1/alerts/{target['alert_id']}").json()["timeline"]
    by_stage = {step["stage"]: step for step in timeline}
    # A missing stage must be stated, with no fabricated timestamp.
    predicted = by_stage["predicted"]["description"].lower()
    assert "no gnn prediction" in predicted
    assert "no ai-downscaled threat field" in by_stage["downscaled"]["description"].lower()
    assert by_stage["downscaled"]["timestamp"] is None
    assert by_stage["alert_created"]["timestamp"] is not None


def test_timeline_only_shows_acknowledged_and_resolved_when_recorded(
    alerts_ready: TestClient,
) -> None:
    target = alerts_ready.get("/api/v1/alerts").json()["alerts"][0]
    stages = [
        step["stage"]
        for step in alerts_ready.get(
            f"/api/v1/alerts/{target['alert_id']}"
        ).json()["timeline"]
    ]
    assert "acknowledged" not in stages
    assert "resolved" not in stages


def test_unknown_alert_returns_404(alerts_ready: TestClient) -> None:
    response = alerts_ready.get("/api/v1/alerts/ALT-does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_alert_lookup_without_a_store_returns_404(empty_alert_client: TestClient) -> None:
    response = empty_alert_client.get("/api/v1/alerts/ALT-anything")
    assert response.status_code == 404
    assert "generate_alerts.py" in response.json()["detail"]


def test_summary_is_not_captured_by_the_alert_id_route(alerts_ready: TestClient) -> None:
    response = alerts_ready.get("/api/v1/alerts/summary")
    assert response.status_code == 200
    assert "summary" in response.json()


# --------------------------------------------------------------------------- #
# GET /api/v1/alerts/summary
# --------------------------------------------------------------------------- #
def test_summary_counts_the_generated_alerts(alerts_ready: TestClient) -> None:
    listed = alerts_ready.get("/api/v1/alerts").json()["count"]
    summary = alerts_ready.get("/api/v1/alerts/summary").json()["summary"]
    assert summary["total_alerts"] == listed
    assert summary["active_alerts"] == listed
    assert summary["highest_severity"] is not None
    assert len(summary["by_status"]) == 5


def test_summary_of_an_empty_store_explains_itself(empty_alert_client: TestClient) -> None:
    summary = empty_alert_client.get("/api/v1/alerts/summary").json()["summary"]
    assert summary["total_alerts"] == 0
    assert summary["highest_severity"] is None
    assert "generate_alerts.py" in summary["message"]



# --------------------------------------------------------------------------- #
# POST /api/v1/alerts/generate
# --------------------------------------------------------------------------- #
def test_generate_creates_and_returns_the_alerts(alert_client: TestClient) -> None:
    payload = generate(alert_client)
    assert payload["dry_run"] is False
    assert payload["created_count"] > 0
    assert payload["alert_count"] == len(payload["alerts"])
    assert payload["notification_count"] == payload["created_count"]
    assert any("GNN" in note for note in payload["degraded"])


def test_generate_dry_run_does_not_touch_the_store(
    alert_client: TestClient, alert_settings: Settings
) -> None:
    payload = generate(alert_client, dry_run=True)
    assert payload["dry_run"] is True
    assert payload["created_count"] > 0
    assert (alert_settings.processed_dir / alert_settings.alerts_filename).exists() is False
    assert (
        alert_settings.processed_dir / alert_settings.notifications_filename
    ).exists() is False


def test_generate_dry_run_over_a_populated_store_changes_nothing(
    alerts_ready: TestClient, alert_settings: Settings
) -> None:
    path = alert_settings.processed_dir / alert_settings.alerts_filename
    before = path.read_bytes()
    generate(alerts_ready, dry_run=True)
    assert path.read_bytes() == before


def test_generate_does_not_duplicate_alerts(alerts_ready: TestClient) -> None:
    """The fixture already ran once; two more runs must only refresh."""
    before = alerts_ready.get("/api/v1/alerts").json()["count"]
    first = generate(alerts_ready)
    second = generate(alerts_ready)
    assert first["created_count"] == 0
    assert second["created_count"] == 0
    assert second["updated_count"] == before
    assert second["alert_count"] == before == first["alert_count"]


def test_generate_without_notify_records_no_notifications(alert_client: TestClient) -> None:
    assert generate(alert_client, notify=False)["notification_count"] == 0


def test_generate_rejects_an_unknown_body_field(alert_client: TestClient) -> None:
    response = alert_client.post("/api/v1/alerts/generate", json={"bogus": True})
    assert response.status_code == 422


def test_generate_returns_503_when_generation_is_disabled(alert_settings: Settings) -> None:
    from extreme_weather.api.main import create_app
    from extreme_weather.config import get_settings

    settings = Settings(
        data_dir=alert_settings.data_dir,
        model_dir=alert_settings.model_dir,
        alert_generation_enabled=False,
    )
    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(application) as client:
            response = client.post("/api/v1/alerts/generate", json={})
    finally:
        application.dependency_overrides.clear()
    assert response.status_code == 503
    assert "ALERT_GENERATION_ENABLED" in response.json()["detail"]



# --------------------------------------------------------------------------- #
# POST /api/v1/alerts/{alert_id}/acknowledge and /resolve
# --------------------------------------------------------------------------- #
def first_alert_id(client: TestClient) -> str:
    """Return the id of the highest-priority stored alert."""
    return client.get("/api/v1/alerts").json()["alerts"][0]["alert_id"]


def test_acknowledge_moves_the_alert_and_records_a_timestamp(
    alerts_ready: TestClient,
) -> None:
    alert_id = first_alert_id(alerts_ready)
    response = alerts_ready.post(
        f"/api/v1/alerts/{alert_id}/acknowledge", json={"note": "team notified"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["alert"]["status"] == "ACKNOWLEDGED"
    assert payload["alert"]["acknowledged_at"] is not None
    assert "team notified" in payload["alert"]["resolved_reason"]
    assert "NOT an official warning" in payload["notice"]


def test_resolve_moves_the_alert_and_records_a_timestamp(alerts_ready: TestClient) -> None:
    alert_id = first_alert_id(alerts_ready)
    response = alerts_ready.post(
        f"/api/v1/alerts/{alert_id}/resolve", json={"reason": "event passed"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["alert"]["status"] == "RESOLVED"
    assert payload["alert"]["resolved_at"] is not None
    assert payload["alert"]["resolved_reason"] == "event passed"


def test_resolved_alert_drops_out_of_the_active_summary(alerts_ready: TestClient) -> None:
    before = alerts_ready.get("/api/v1/alerts/summary").json()["summary"]["active_alerts"]
    alerts_ready.post(f"/api/v1/alerts/{first_alert_id(alerts_ready)}/resolve", json={})
    after = alerts_ready.get("/api/v1/alerts/summary").json()["summary"]
    assert after["active_alerts"] == before - 1
    assert after["resolved_alerts"] == 1


def test_acknowledging_an_unknown_alert_returns_404(alerts_ready: TestClient) -> None:
    response = alerts_ready.post("/api/v1/alerts/ALT-nope/acknowledge", json={})
    assert response.status_code == 404


def test_resolving_an_unknown_alert_returns_404(alerts_ready: TestClient) -> None:
    assert alerts_ready.post("/api/v1/alerts/ALT-nope/resolve", json={}).status_code == 404


def test_acknowledging_a_resolved_alert_returns_422(alerts_ready: TestClient) -> None:
    alert_id = first_alert_id(alerts_ready)
    alerts_ready.post(f"/api/v1/alerts/{alert_id}/resolve", json={})
    response = alerts_ready.post(f"/api/v1/alerts/{alert_id}/acknowledge", json={})
    assert response.status_code == 422
    assert "Cannot move an alert" in response.json()["detail"]


def test_resolving_a_resolved_alert_is_idempotent(alerts_ready: TestClient) -> None:
    alert_id = first_alert_id(alerts_ready)
    first = alerts_ready.post(f"/api/v1/alerts/{alert_id}/resolve", json={})
    second = alerts_ready.post(f"/api/v1/alerts/{alert_id}/resolve", json={})
    assert first.status_code == second.status_code == 200
    assert first.json()["alert"]["resolved_at"] == second.json()["alert"]["resolved_at"]


def test_action_bodies_reject_unknown_fields(alerts_ready: TestClient) -> None:
    alert_id = first_alert_id(alerts_ready)
    assert (
        alerts_ready.post(
            f"/api/v1/alerts/{alert_id}/acknowledge", json={"nope": 1}
        ).status_code
        == 422
    )
    assert (
        alerts_ready.post(
            f"/api/v1/alerts/{alert_id}/resolve", json={"nope": 1}
        ).status_code
        == 422
    )


def test_note_length_is_validated(alerts_ready: TestClient) -> None:
    alert_id = first_alert_id(alerts_ready)
    response = alerts_ready.post(
        f"/api/v1/alerts/{alert_id}/acknowledge", json={"note": "x" * 501}
    )
    assert response.status_code == 422



# --------------------------------------------------------------------------- #
# GET /api/v1/notifications
# --------------------------------------------------------------------------- #
def test_notifications_are_empty_before_a_run(empty_alert_client: TestClient) -> None:
    response = empty_alert_client.get("/api/v1/notifications")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert "generate_alerts.py" in payload["message"]
    assert "external service" in payload["notice"]


def test_notifications_are_recorded_locally_by_a_run(alerts_ready: TestClient) -> None:
    payload = alerts_ready.get("/api/v1/notifications").json()
    assert payload["count"] > 0
    record = payload["records"][0]
    assert record["provider"] == "mock"
    assert record["delivered"] is True
    assert "Why this alert exists" in record["body"]


def test_notifications_support_a_limit(alerts_ready: TestClient) -> None:
    total = alerts_ready.get("/api/v1/notifications").json()["count"]
    assert alerts_ready.get("/api/v1/notifications?limit=1").json()["count"] == 1
    assert total > 1


def test_notification_limit_is_validated(alerts_ready: TestClient) -> None:
    assert alerts_ready.get("/api/v1/notifications?limit=0").status_code == 422
    assert alerts_ready.get("/api/v1/notifications?limit=abc").status_code == 422


# --------------------------------------------------------------------------- #
# Registration and non-regression
# --------------------------------------------------------------------------- #
def test_all_alert_routes_are_registered(alerts_ready: TestClient) -> None:
    paths = alerts_ready.get("/openapi.json").json()["paths"]
    for path in (
        "/api/v1/alerts",
        "/api/v1/alerts/{alert_id}",
        "/api/v1/alerts/generate",
        "/api/v1/alerts/{alert_id}/acknowledge",
        "/api/v1/alerts/{alert_id}/resolve",
        "/api/v1/alerts/simulate",
        "/api/v1/alerts/summary",
        "/api/v1/notifications",
    ):
        assert path in paths, f"{path} is not registered"


def test_registering_the_alert_router_does_not_break_the_earlier_phases(
    alerts_ready: TestClient,
) -> None:
    """Phase 0-4 endpoints must keep working alongside Phase 5."""
    assert alerts_ready.get("/health").status_code == 200
    assert alerts_ready.get("/api/v1/events").json()["count"] == 4
    assert alerts_ready.get("/api/v1/tracks").json()["count"] == 1
    assert alerts_ready.get("/api/v1/gnn/status").status_code == 200
    assert alerts_ready.get("/api/v1/downscaling/status").status_code == 200
    assert alerts_ready.get("/api/v1/datasets").status_code == 200

