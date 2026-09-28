"""Tests for the Phase 5 alert generation *engine* as a whole (SIH26078).

``test_alert_risk.py`` covers the scoring function; this module covers how the
service assembles its inputs, applies the rules, deduplicates and records the
result, over a real Phase 1 + Phase 2 pipeline.
"""

from __future__ import annotations

from datetime import timedelta

from extreme_weather.alerts.models import AlertStatus
from extreme_weather.alerts.service import AlertService
from extreme_weather.config import Settings
from extreme_weather.schemas.event import Severity

from tests.alert_fixtures import NOW, build_service, make_event, write_events


# --------------------------------------------------------------------------- #
# Source-component provenance
# --------------------------------------------------------------------------- #
def test_every_alert_records_the_components_it_actually_used(
    service: AlertService,
) -> None:
    result = service.generate(now=NOW)
    for alert in result.stored.alerts:
        assert alert.source_components
        assert alert.source_components[0] == "events"
        assert "gnn" not in alert.source_components
        assert "downscaling" not in alert.source_components


def test_tracked_alerts_report_the_tracking_component(service: AlertService) -> None:
    result = service.generate(now=NOW)
    tracked = [a for a in result.stored.alerts if "tracking" in a.source_components]
    assert tracked, "expected at least one alert from the Phase 2 track"
    assert tracked[0].track_id is not None
    assert tracked[0].status is AlertStatus.ACTIVE


def test_untracked_alerts_stay_new(alert_settings: Settings) -> None:
    """An untracked event has no trajectory, so its alert cannot be ACTIVE."""
    write_events(alert_settings.processed_dir / "events.json", (make_event("E0", hour=0),))
    result = build_service(alert_settings).generate(now=NOW)
    assert result.stored.alerts
    for alert in result.stored.alerts:
        assert "tracking" not in alert.source_components
        assert alert.status is AlertStatus.NEW
        assert alert.track_id is None


# --------------------------------------------------------------------------- #
# Message and title construction
# --------------------------------------------------------------------------- #
def test_alert_message_states_the_real_facts_and_the_limits(
    service: AlertService,
) -> None:
    result = service.generate(now=NOW)
    for alert in result.stored.alerts:
        lowered = alert.message.lower()
        assert "anomaly score" in lowered
        assert "no gnn prediction is available" in lowered
        assert "no ai-downscaled threat field is available" in lowered
        assert "not an official warning" in lowered


def test_alert_message_mentions_the_track_when_present(service: AlertService) -> None:
    result = service.generate(now=NOW)
    tracked = next(a for a in result.stored.alerts if a.track_id)
    assert "tracked for" in tracked.message
    assert tracked.movement_direction in tracked.message


def test_alert_title_carries_the_severity_and_event_type(service: AlertService) -> None:
    result = service.generate(now=NOW)
    for alert in result.stored.alerts:
        assert alert.severity.value in alert.title
        assert any(
            word in alert.title for word in ("Rainfall", "Wind", "Temperature")
        )


# --------------------------------------------------------------------------- #
# Rule outcome integration
# --------------------------------------------------------------------------- #
def test_a_strengthening_storm_is_elevated_above_a_weak_one(
    alert_settings: Settings,
) -> None:
    """A severe detection must outrank a weak first detection."""
    write_events(
        alert_settings.processed_dir / "events.json",
        (
            make_event("WEAK", hour=0, intensity=5.0, severity=Severity.WATCH),
            make_event("STRONG", hour=1, intensity=55.0, severity=Severity.EXTREME),
        ),
    )
    result = build_service(alert_settings).generate(now=NOW)
    by_event = {alert.event_id: alert for alert in result.stored.alerts}
    assert by_event["STRONG"].risk.score > by_event["WEAK"].risk.score
    assert by_event["STRONG"].priority.rank >= by_event["WEAK"].priority.rank


def test_a_normal_detection_is_suppressed_by_the_trigger_rule(
    alert_settings: Settings,
) -> None:
    """Alerting on a NORMAL detection would be exactly the alert spam we refuse."""
    write_events(
        alert_settings.processed_dir / "events.json",
        (make_event("CALM", hour=0, intensity=1.0, severity=Severity.NORMAL),),
    )
    result = build_service(alert_settings).generate(now=NOW)
    assert result.alert_count == 0
    assert result.suppressed_count == 1


# --------------------------------------------------------------------------- #
# Store metadata and counters
# --------------------------------------------------------------------------- #
def test_stored_metadata_records_the_configuration_used(service: AlertService) -> None:
    result = service.generate(now=NOW)
    metadata = result.stored.metadata
    assert metadata["dry_run"] is False
    assert metadata["event_count"] == 4
    assert metadata["track_count"] == 1
    assert metadata["risk_config"]["factor_weights"]
    assert metadata["rule_config"]["min_trigger_severity"] == "WATCH"
    assert any("GNN" in note for note in metadata["notes"])


def test_counts_are_reported_per_run(service: AlertService) -> None:
    first = service.generate(now=NOW)
    assert first.created_count == first.stored.created_count
    assert first.updated_count == 0
    assert first.suppressed_count == 0
    second = service.generate(now=NOW + timedelta(minutes=5))
    assert second.created_count == 0
    assert second.updated_count == first.created_count


def test_alerts_are_ordered_by_priority(service: AlertService) -> None:
    result = service.generate(now=NOW)
    ranks = [alert.priority.rank for alert in result.stored.alerts]
    assert ranks == sorted(ranks, reverse=True)


def test_validity_window_comes_from_the_configured_minutes(service: AlertService) -> None:
    result = service.generate(now=NOW)
    for alert in result.stored.alerts:
        minutes = (alert.valid_until - alert.valid_from).total_seconds() / 60.0
        assert minutes == service.rule_config.default_validity_minutes

