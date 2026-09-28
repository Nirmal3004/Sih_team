"""Tests for the Phase 5 alert service: generation, lifecycle and queries.

Everything here runs the **real** service over a real Phase 1 + Phase 2
pipeline (see :mod:`tests.alert_fixtures`), so the tests assert the behaviour
the CLI and the API actually get - not a hand-written alert object.
"""

from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path

import pytest

from extreme_weather.alerts.models import AlertStatus, AlertStoreResult
from extreme_weather.alerts.notifications import NotificationStore
from extreme_weather.alerts.rules import RuleConfig
from extreme_weather.alerts.service import (
    ALLOWED_TRANSITIONS,
    AlertError,
    AlertService,
    AlertTransitionError,
    AlertUnknownError,
)
from extreme_weather.alerts.store import AlertStore
from extreme_weather.anomaly.service import EventStore
from extreme_weather.config import Settings
from extreme_weather.schemas.event import EventType
from extreme_weather.tracking.store import TrackStore

from tests.alert_fixtures import NOW, build_service, storm_events, write_events

UTC = timezone.utc


def alert_path(settings: Settings) -> Path:
    """Path of the alert store used by the test services."""
    return settings.processed_dir / settings.alerts_filename


def notification_path(settings: Settings) -> Path:
    """Path of the notification log used by the test services."""
    return settings.processed_dir / settings.notifications_filename


# --------------------------------------------------------------------------- #
# Generation over a real pipeline
# --------------------------------------------------------------------------- #
def test_generation_produces_one_alert_per_newest_event_in_a_track(
    service: AlertService,
) -> None:
    """Older points of a track are superseded, so they are not re-alerted."""
    result = service.generate(now=NOW)
    assert result.created_count >= 1
    event_ids = [alert.event_id for alert in result.stored.alerts]
    assert len(event_ids) == len(set(event_ids))


def test_generation_writes_the_alert_store(service: AlertService, seeded_pipeline: Settings) -> None:
    service.generate(now=NOW)
    assert alert_path(seeded_pipeline).is_file()
    reloaded = AlertStore(alert_path(seeded_pipeline)).load()
    assert reloaded is not None
    assert reloaded.alert_count > 0


def test_dry_run_writes_neither_store(service: AlertService, seeded_pipeline: Settings) -> None:
    result = service.generate(dry_run=True, now=NOW)
    assert result.dry_run is True
    assert result.alert_count > 0
    assert result.notifications is None
    assert alert_path(seeded_pipeline).exists() is False
    assert notification_path(seeded_pipeline).exists() is False


def test_dry_run_does_not_modify_an_existing_store(
    service: AlertService, seeded_pipeline: Settings
) -> None:
    """A dry run over a populated store must leave it byte-for-byte identical."""
    service.generate(now=NOW)
    before = alert_path(seeded_pipeline).read_bytes()
    notifications_before = notification_path(seeded_pipeline).read_bytes()

    service.generate(dry_run=True, now=NOW + timedelta(minutes=5))

    assert alert_path(seeded_pipeline).read_bytes() == before
    assert notification_path(seeded_pipeline).read_bytes() == notifications_before


def test_generation_notifies_through_the_mock_provider(
    service: AlertService, seeded_pipeline: Settings
) -> None:
    result = service.generate(now=NOW)
    assert result.notifications is not None
    assert len(result.notifications.records) == result.created_count
    log = NotificationStore(notification_path(seeded_pipeline)).load()
    assert len(log.records) == result.created_count


def test_generation_can_skip_notification(
    service: AlertService, seeded_pipeline: Settings
) -> None:
    result = service.generate(now=NOW, notify=False)
    assert result.notifications is None
    assert notification_path(seeded_pipeline).exists() is False


def test_alert_ids_are_deterministic_across_runs(service: AlertService) -> None:
    """Re-running the generator over the same data must not churn ids."""
    first = {a.alert_id for a in service.generate(dry_run=True, now=NOW).stored.alerts}
    second = {a.alert_id for a in service.generate(dry_run=True, now=NOW).stored.alerts}
    assert first == second


def test_scores_are_reproducible_across_runs(service: AlertService) -> None:
    first = {a.alert_id: a.risk.score for a in service.generate(dry_run=True, now=NOW).stored.alerts}
    second = {a.alert_id: a.risk.score for a in service.generate(dry_run=True, now=NOW).stored.alerts}
    assert first == second


# --------------------------------------------------------------------------- #
# Deduplication - no alert spam
# --------------------------------------------------------------------------- #
def test_rerunning_in_the_same_window_updates_instead_of_duplicating(
    service: AlertService,
) -> None:
    first = service.generate(now=NOW)
    second = service.generate(now=NOW + timedelta(minutes=10))
    assert first.created_count > 0
    assert second.created_count == 0
    assert second.updated_count == first.created_count
    assert second.alert_count == first.alert_count


def test_rerunning_in_a_later_window_creates_a_new_alert(service: AlertService) -> None:
    """A new time bucket means a legitimately new alert, not a duplicate."""
    first = service.generate(now=NOW)
    later = service.generate(now=NOW + timedelta(hours=3))
    assert first.created_count > 0
    assert later.created_count == first.created_count
    # The earlier alerts stay in the store: history is preserved.
    assert later.alert_count == first.alert_count * 2


def test_deduplication_window_is_configurable(seeded_pipeline: Settings) -> None:
    service = build_service(seeded_pipeline, rule_config=RuleConfig(deduplication_window_minutes=1))
    first = service.generate(now=NOW)
    second = service.generate(now=NOW + timedelta(minutes=5))
    assert first.created_count > 0
    assert second.created_count > 0  # a different bucket -> a new alert


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #
def test_missing_gnn_model_still_generates_alerts(
    service: AlertService, seeded_pipeline: Settings
) -> None:
    """No GNN service at all: alerts are produced, degraded and labelled."""
    result = service.generate(now=NOW)
    assert result.alert_count > 0
    for alert in result.stored.alerts:
        assert "gnn" not in alert.source_components
    assert any("GNN" in note for note in result.degraded)


def test_a_failing_gnn_service_is_recorded_not_raised(seeded_pipeline: Settings) -> None:
    class BrokenGNN:
        def predict(self, track_id: str):
            raise RuntimeError("no checkpoint on disk")

    result = build_service(seeded_pipeline, gnn_service=BrokenGNN()).generate(now=NOW)
    assert result.alert_count > 0
    assert any("GNN prediction unavailable" in note for note in result.degraded)


def test_a_working_gnn_service_enriches_the_alert(seeded_pipeline: Settings) -> None:
    class FakeGNN:
        def predict(self, track_id: str):
            from extreme_weather.gnn.models import GNNPrediction

            return GNNPrediction(
                track_id=track_id,
                timestamp=NOW,
                event_type="EXTREME_RAINFALL",
                current_latitude=13.0,
                current_longitude=77.3,
                current_intensity=45.0,
                predicted_latitude=13.5,
                predicted_longitude=77.8,
                predicted_intensity=60.0,
                delta_latitude=0.5,
                delta_longitude=0.5,
                delta_intensity=15.0,
                predicted_speed_kmh=40.0,
                predicted_direction="NE",
                model_version="test",
            )

    result = build_service(seeded_pipeline, gnn_service=FakeGNN()).generate(now=NOW)
    enriched = [a for a in result.stored.alerts if "gnn" in a.source_components]
    assert enriched, "expected at least one alert enriched by the GNN"
    assert enriched[0].has_prediction is True
    assert enriched[0].predicted_intensity == 60.0


def test_missing_tracks_file_degrades_to_detection_only_alerts(
    alert_settings: Settings,
) -> None:
    write_events(alert_settings.processed_dir / "events.json", storm_events())
    service = AlertService(
        alert_store=AlertStore(alert_path(alert_settings)),
        event_store=EventStore(alert_settings.processed_dir / "events.json"),
        track_store=TrackStore(alert_settings.processed_dir / "tracks.json"),
    )
    result = service.generate(now=NOW)
    assert result.alert_count > 0
    assert all("tracking" not in a.source_components for a in result.stored.alerts)
    assert any("tracking file" in note for note in result.degraded)


def test_missing_events_file_produces_no_alerts_and_says_why(
    alert_settings: Settings,
) -> None:
    service = AlertService(
        alert_store=AlertStore(alert_path(alert_settings)),
        event_store=EventStore(alert_settings.processed_dir / "events.json"),
        track_store=TrackStore(alert_settings.processed_dir / "tracks.json"),
    )
    result = service.generate(now=NOW)
    assert result.alert_count == 0
    assert any("event analysis" in note for note in result.degraded)


def test_missing_downscaling_model_is_reported_in_the_explanation(
    service: AlertService,
) -> None:
    result = service.generate(now=NOW)
    for alert in result.stored.alerts:
        assert "downscaling" not in alert.source_components
        assert "no AI-downscaled threat field" in alert.risk.explanation


def test_with_downscaled_threat_adds_the_component(seeded_pipeline: Settings) -> None:
    service = build_service(seeded_pipeline).with_downscaled_threat({"E3": (45.0, 5.0)})
    result = service.generate(now=NOW)
    enriched = [a for a in result.stored.alerts if "downscaling" in a.source_components]
    assert len(enriched) == 1
    assert enriched[0].event_id == "E3"



# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_legal_transitions_are_declared() -> None:
    assert ALLOWED_TRANSITIONS[AlertStatus.RESOLVED] == ()
    assert ALLOWED_TRANSITIONS[AlertStatus.EXPIRED] == ()
    assert AlertStatus.ACKNOWLEDGED in ALLOWED_TRANSITIONS[AlertStatus.ACTIVE]
    assert AlertStatus.EXPIRED in ALLOWED_TRANSITIONS[AlertStatus.ACTIVE]


def test_acknowledge_moves_the_alert_and_records_a_real_timestamp(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    acknowledged = service.acknowledge_alert(alert.alert_id, note="on it", now=NOW)
    assert acknowledged.status is AlertStatus.ACKNOWLEDGED
    assert acknowledged.acknowledged_at == NOW
    assert "on it" in (acknowledged.resolved_reason or "")
    assert service.get_alert(alert.alert_id).status is AlertStatus.ACKNOWLEDGED


def test_acknowledging_twice_keeps_the_first_timestamp(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    first = service.acknowledge_alert(alert.alert_id, now=NOW)
    second = service.acknowledge_alert(
        alert.alert_id, now=NOW + timedelta(hours=1)
    )
    assert second.acknowledged_at == first.acknowledged_at


def test_resolve_moves_the_alert_and_records_a_real_timestamp(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    resolved = service.resolve_alert(alert.alert_id, reason="passed", now=NOW)
    assert resolved.status is AlertStatus.RESOLVED
    assert resolved.resolved_at == NOW
    assert resolved.resolved_reason == "passed"


def test_resolving_twice_is_idempotent(service: AlertService, generated_alerts: list) -> None:
    alert = generated_alerts[0]
    first = service.resolve_alert(alert.alert_id, now=NOW)
    second = service.resolve_alert(alert.alert_id, now=NOW + timedelta(hours=1))
    assert second.resolved_at == first.resolved_at


def test_promote_moves_a_new_alert_to_active(
    service: AlertService, generated_alerts: list
) -> None:
    new_alert = next(a for a in generated_alerts if a.status is AlertStatus.NEW)
    promoted = service.promote_alert(new_alert.alert_id, now=NOW)
    assert promoted.status is AlertStatus.ACTIVE


def test_promoting_an_active_alert_is_a_no_op(
    service: AlertService, generated_alerts: list
) -> None:
    active = next(a for a in generated_alerts if a.status is AlertStatus.ACTIVE)
    assert service.promote_alert(active.alert_id, now=NOW).status is AlertStatus.ACTIVE


def test_acknowledging_a_resolved_alert_is_rejected(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    service.resolve_alert(alert.alert_id, now=NOW)
    with pytest.raises(AlertTransitionError, match="Cannot move an alert"):
        service.acknowledge_alert(alert.alert_id, now=NOW)


def test_acknowledging_an_expired_alert_is_rejected(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    service.expire_alerts(now=NOW + timedelta(days=1), hours=0.0)
    with pytest.raises(AlertTransitionError):
        service.acknowledge_alert(alert.alert_id, now=NOW)


def test_terminal_states_have_no_way_back_to_open(
    service: AlertService, generated_alerts: list
) -> None:
    """RESOLVED and EXPIRED are terminal: no reopen, no re-acknowledge."""
    alert = generated_alerts[0]
    service.resolve_alert(alert.alert_id, now=NOW)
    with pytest.raises(AlertTransitionError):
        service.promote_alert(alert.alert_id, now=NOW)
    with pytest.raises(AlertTransitionError):
        service.acknowledge_alert(alert.alert_id, now=NOW)


def test_unknown_alert_raises_alert_unknown(service: AlertService) -> None:
    with pytest.raises(AlertUnknownError, match="not found"):
        service.get_alert("ALT-nope")


# --------------------------------------------------------------------------- #
# Expiration (R6)
# --------------------------------------------------------------------------- #
def test_expire_alerts_expires_an_alert_past_its_validity(
    service: AlertService, generated_alerts: list
) -> None:
    expired = service.expire_alerts(now=NOW + timedelta(days=1))
    assert len(expired) == len(generated_alerts)
    assert all(alert.status is AlertStatus.EXPIRED for alert in expired)
    assert all(alert.resolved_at is not None for alert in expired)


def test_expire_alerts_keeps_a_fresh_alert(service: AlertService, generated_alerts: list) -> None:
    assert service.expire_alerts(now=NOW + timedelta(minutes=5)) == []


def test_expire_alerts_never_expires_an_acknowledged_alert(
    service: AlertService, generated_alerts: list
) -> None:
    """An acknowledgement is an operator decision, not a stale record."""
    alert = generated_alerts[0]
    service.acknowledge_alert(alert.alert_id, now=NOW)
    expired = service.expire_alerts(now=NOW + timedelta(days=1))
    assert alert.alert_id not in {a.alert_id for a in expired}
    assert service.get_alert(alert.alert_id).status is AlertStatus.ACKNOWLEDGED


def test_expire_alerts_honours_an_explicit_window(
    service: AlertService, generated_alerts: list
) -> None:
    assert service.expire_alerts(now=NOW + timedelta(hours=1), hours=0.5)
    assert service.get_alert(generated_alerts[0].alert_id).status is AlertStatus.EXPIRED



# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def test_create_alert_stores_a_new_alert(service: AlertService, generated_alerts: list) -> None:
    alert = generated_alerts[0]
    service.alert_store.clear()
    created = service.create_alert(alert, now=NOW)
    assert created.alert_id == alert.alert_id
    assert service.get_alert(alert.alert_id).alert_id == alert.alert_id


def test_create_alert_rejects_a_duplicate_id(
    service: AlertService, generated_alerts: list
) -> None:
    with pytest.raises(AlertError, match="already exists"):
        service.create_alert(generated_alerts[0], now=NOW)


def test_update_alert_applies_a_validated_change(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    updated = service.update_alert(alert.alert_id, {"movement_direction": "NE"}, now=NOW)
    assert updated.movement_direction == "NE"
    assert updated.updated_at == NOW


def test_update_alert_rejects_an_invalid_change(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    with pytest.raises(AlertError, match="Invalid change"):
        service.update_alert(alert.alert_id, {"current_lat": 500.0}, now=NOW)


def test_update_alert_cannot_bypass_the_lifecycle_check(
    service: AlertService, generated_alerts: list
) -> None:
    alert = generated_alerts[0]
    for field in ("status", "alert_id", "created_at", "acknowledged_at", "resolved_at"):
        with pytest.raises(AlertError, match="cannot be changed"):
            service.update_alert(alert.alert_id, {field: "RESOLVED"}, now=NOW)


# --------------------------------------------------------------------------- #
# Queries and summary
# --------------------------------------------------------------------------- #
def test_get_active_alerts_excludes_terminal_states(
    service: AlertService, generated_alerts: list
) -> None:
    active_before = service.get_active_alerts()
    assert len(active_before) == len(generated_alerts)
    service.resolve_alert(generated_alerts[0].alert_id, now=NOW)
    active_after = service.get_active_alerts()
    assert len(active_after) == len(generated_alerts) - 1


def test_get_alert_history_can_exclude_open_alerts(
    service: AlertService, generated_alerts: list
) -> None:
    service.resolve_alert(generated_alerts[0].alert_id, now=NOW)
    assert len(service.get_alert_history()) == len(generated_alerts)
    closed = service.get_alert_history(include_open=False)
    assert len(closed) == 1
    assert closed[0].status is AlertStatus.RESOLVED


def test_alert_history_is_newest_first(service: AlertService, generated_alerts: list) -> None:
    history = service.get_alert_history()
    stamps = [alert.created_at for alert in history]
    assert stamps == sorted(stamps, reverse=True)


def test_summary_counts_the_stored_alerts(service: AlertService, generated_alerts: list) -> None:
    summary = service.summary()
    assert summary.total_alerts == len(generated_alerts)
    assert summary.active_alerts == len(generated_alerts)
    assert summary.resolved_alerts == 0
    assert summary.expired_alerts == 0
    assert summary.highest_severity is not None
    assert len(summary.by_status) == len(AlertStatus)


def test_summary_reflects_acknowledged_and_resolved_counts(
    service: AlertService, generated_alerts: list
) -> None:
    service.acknowledge_alert(generated_alerts[0].alert_id, now=NOW)
    service.resolve_alert(generated_alerts[1].alert_id, now=NOW)
    summary = service.summary()
    assert summary.acknowledged_alerts == 1
    assert summary.resolved_alerts == 1
    assert summary.active_alerts == len(generated_alerts) - 1


def test_summary_of_an_empty_store_explains_itself(alert_settings: Settings) -> None:
    service = AlertService(alert_store=AlertStore(alert_path(alert_settings)))
    summary = service.summary()
    assert summary.total_alerts == 0
    assert summary.highest_severity is None
    assert "generate_alerts.py" in (summary.message or "")


def test_summary_counts_predicted_threats_only_when_present(
    service: AlertService, generated_alerts: list
) -> None:
    assert service.summary().predicted_threats == 0


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def test_full_pipeline_from_events_to_notifications(
    service: AlertService, seeded_pipeline: Settings
) -> None:
    """Phase 1 + 2 -> risk -> rules -> dedup -> store -> notification."""
    result = service.generate(now=NOW)
    assert result.created_count > 0
    assert alert_path(seeded_pipeline).is_file()
    assert notification_path(seeded_pipeline).is_file()

    alert = result.stored.alerts[0]
    assert alert.event_id
    assert alert.risk.explanation
    assert 0.0 <= alert.risk.score <= 100.0
    assert alert.notice.lower().count("not an official") == 1

    log = NotificationStore(notification_path(seeded_pipeline)).load()
    notified = {record.alert_id for record in log.records}
    assert {a.alert_id for a in result.stored.alerts} == notified
    assert all(record.delivered for record in log.records)

