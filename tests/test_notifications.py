"""Tests for the Phase 5 notification adapters (SIH26078).

The contract being verified is deliberately narrow: notifications are optional,
the local mock is the only enabled provider, nothing is ever delivered to an
external service, and a broken or unwritable channel degrades a notification
without ever breaking the alert it belongs to.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from extreme_weather.alerts.models import (
    Alert,
    AlertStatus,
    ConfidenceLabel,
    ContributingFactor,
    RiskAssessment,
)
from extreme_weather.alerts.notifications import (
    NOTIFICATION_DISCLAIMER,
    EmailNotificationProvider,
    MockNotificationProvider,
    NotificationError,
    NotificationProvider,
    NotificationRecord,
    NotificationResult,
    NotificationStatus,
    NotificationStore,
    WebhookNotificationProvider,
    build_notification_providers,
)
from extreme_weather.schemas.event import EventType

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_alert(**overrides: object) -> Alert:
    """Build a valid alert for notification tests."""
    risk = RiskAssessment.from_score(
        score=52.0,
        factors=[
            ContributingFactor(
                name="intensity",
                label="Observed intensity",
                weight=14.0,
                normalized_value=0.9,
                points=12.6,
                detail="45.00 against a reference of 50.00",
            )
        ],
        explanation="WARNING priority (score 52.00/100) because:\n- Observed intensity",
        generated_at=NOW,
    )
    payload: dict = {
        "alert_id": "ALT-E0-WARNING-1",
        "event_id": "E0",
        "track_id": "TRK-1",
        "event_type": EventType.EXTREME_RAINFALL,
        "severity": "WARNING",
        "priority": "WARNING",
        "title": "WARNING Extreme Rainfall alert",
        "message": "A synthetic event was detected.",
        "created_at": NOW,
        "updated_at": NOW,
        "valid_from": NOW,
        "valid_until": NOW + timedelta(hours=6),
        "status": AlertStatus.ACTIVE,
        "confidence_label": ConfidenceLabel.LOW,
        "risk": risk,
        "current_lat": 13.0,
        "current_lon": 77.0,
        "predicted_lat": 13.4,
        "predicted_lon": 77.4,
        "predicted_intensity": 52.0,
        "intensity": 45.0,
        "source_components": ["events", "tracking"],
        "deduplication_key": "E0|WARNING|000000001234",
    }
    payload.update(overrides)
    return Alert(**payload)


@pytest.fixture()
def notification_store(tmp_path: Path) -> NotificationStore:
    """A notification log in a temporary folder."""
    return NotificationStore(tmp_path / "notifications.json")


# --------------------------------------------------------------------------- #
# Adapter interface
# --------------------------------------------------------------------------- #
def test_mock_provider_implements_the_protocol(notification_store: NotificationStore) -> None:
    provider = MockNotificationProvider(notification_store)
    assert isinstance(provider, NotificationProvider)
    assert provider.name() == "mock"
    assert provider.supports() is True


def test_only_the_local_mock_is_enabled_by_default(notification_store: NotificationStore) -> None:
    providers = build_notification_providers(notification_store)
    assert [provider.name() for provider in providers] == ["mock"]


def test_external_stubs_are_registered_but_never_enabled() -> None:
    for stub in (EmailNotificationProvider(), WebhookNotificationProvider()):
        assert stub.supports() is False


def test_external_stubs_report_non_delivery_and_never_contact_anything() -> None:
    alert = make_alert()
    for stub in (EmailNotificationProvider(), WebhookNotificationProvider()):
        record = stub.send(alert)
        assert record.delivered is False
        assert "not implemented" in record.detail
        assert "no credential" in record.detail or "no webhook url" in record.detail


# --------------------------------------------------------------------------- #
# Mock provider
# --------------------------------------------------------------------------- #
def test_mock_provider_records_the_alert_locally(
    notification_store: NotificationStore,
) -> None:
    record = MockNotificationProvider(notification_store).send(make_alert())
    assert record.delivered is True
    assert record.provider == "mock"
    assert record.channel == "local"
    assert record.alert_id == "ALT-E0-WARNING-1"
    assert notification_store.path.is_file()


def test_mock_record_body_carries_the_explanation_and_no_accuracy_claim(
    notification_store: NotificationStore,
) -> None:
    record = MockNotificationProvider(notification_store).send(make_alert())
    assert "[WARNING]" in record.subject
    assert "Observed intensity" in record.body
    assert "Why this alert exists" in record.body
    assert "Risk score" in record.body
    lowered = record.body.lower()
    for forbidden in ("guaranteed", "95% accurate", "exact evacuation"):
        assert forbidden not in lowered


def test_mock_body_states_which_components_were_used(
    notification_store: NotificationStore,
) -> None:
    record = MockNotificationProvider(notification_store).send(
        make_alert(source_components=["events"], predicted_lat=None, predicted_lon=None)
    )
    assert "Components  : events" in record.body
    assert "no GNN prediction" in record.body


def test_mock_provider_keeps_records_in_memory(notification_store: NotificationStore) -> None:
    provider = MockNotificationProvider(notification_store)
    provider.send(make_alert())
    provider.send(make_alert(alert_id="ALT-E1-WARNING-1"))
    assert [record.alert_id for record in provider.records] == [
        "ALT-E0-WARNING-1",
        "ALT-E1-WARNING-1",
    ]


def test_mock_provider_can_skip_persistence(notification_store: NotificationStore) -> None:
    provider = MockNotificationProvider(notification_store, persist=False)
    record = provider.send(make_alert())
    assert record.delivered is True
    assert notification_store.path.exists() is False


def test_a_failed_log_write_degrades_the_notification_not_the_alert(
    tmp_path: Path,
) -> None:
    """A broken channel must not raise into the alert pipeline."""
    blocked = tmp_path / "notifications.json"
    blocked.mkdir(parents=True, exist_ok=True)
    record = MockNotificationProvider(NotificationStore(blocked)).send(make_alert())
    assert record.delivered is False
    assert "local log write failed" in record.detail
    assert record.alert_id == "ALT-E0-WARNING-1"


# --------------------------------------------------------------------------- #
# Notification store
# --------------------------------------------------------------------------- #
def test_missing_log_reads_as_an_empty_result(notification_store: NotificationStore) -> None:
    result = notification_store.load()
    assert result.records == []
    assert "No notifications recorded yet" in (result.message or "")


def test_appending_accumulates_records(notification_store: NotificationStore) -> None:
    provider = MockNotificationProvider(notification_store)
    provider.send(make_alert())
    provider.send(make_alert(alert_id="ALT-E1-WARNING-1"))
    result = notification_store.load()
    assert len(result.records) == 2
    assert result.delivered_count == 2


def test_log_file_is_versioned_json(notification_store: NotificationStore) -> None:
    MockNotificationProvider(notification_store).send(make_alert())
    payload = json.loads(notification_store.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["record_count"] == 1
    assert "notice" in payload


def test_empty_log_file_is_not_an_error(notification_store: NotificationStore) -> None:
    notification_store.path.parent.mkdir(parents=True, exist_ok=True)
    notification_store.path.write_text("", encoding="utf-8")
    assert notification_store.load().records == []


def test_malformed_log_raises_a_notification_error(notification_store: NotificationStore) -> None:
    notification_store.path.parent.mkdir(parents=True, exist_ok=True)
    notification_store.path.write_text("{broken", encoding="utf-8")
    with pytest.raises(NotificationError, match="Could not read notification file"):
        notification_store.load()


def test_malformed_records_in_the_log_are_skipped(notification_store: NotificationStore) -> None:
    MockNotificationProvider(notification_store).send(make_alert())
    payload = json.loads(notification_store.path.read_text(encoding="utf-8"))
    payload["records"].append({"notification_id": "incomplete"})
    notification_store.path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(notification_store.load().records) == 1


def test_clear_removes_the_log(notification_store: NotificationStore) -> None:
    MockNotificationProvider(notification_store).send(make_alert())
    assert notification_store.exists() is True
    notification_store.clear()
    assert notification_store.exists() is False


def test_disclaimer_says_nothing_was_delivered() -> None:
    lowered = NOTIFICATION_DISCLAIMER.lower()
    assert "never" in lowered
    assert "external" in lowered


def test_notification_result_from_dict_tolerates_junk() -> None:
    result = NotificationResult.from_dict("not a dict")
    assert result.records == []
    assert result.providers == []


def test_notification_record_requires_the_mandatory_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NotificationRecord.model_validate({"alert_id": "only-this"})
    assert NotificationStatus(provider="mock", delivered=True).detail == ""

