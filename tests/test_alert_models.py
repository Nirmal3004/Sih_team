"""Tests for the Phase 5 alert domain model and severity ladder (SIH26078).

Covers: model validation, severity ordering, deterministic id/dedup keys, and
the honest-labelling guarantees (no accuracy claims, explicit provenance).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from extreme_weather.alerts.models import (
    ALERT_NOTICE,
    Alert,
    AlertStatus,
    AlertStoreResult,
    ConfidenceLabel,
    ContributingFactor,
    RiskAssessment,
    build_alert_id,
    deduplication_key,
    time_bucket,
)
from extreme_weather.alerts.severity import (
    DEFAULT_SEVERITY_THRESHOLDS,
    SEVERITY_DISCLAIMER,
    AlertSeverity,
    SeverityThresholds,
    max_severity,
    severity_for_score,
)
from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.event import Severity as EventSeverity

from tests.alert_fixtures import NOW

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 6, 0, tzinfo=UTC)


def make_assessment(score: float = 50.0) -> RiskAssessment:
    """Build a valid risk assessment for alert construction."""
    return RiskAssessment.from_score(
        score=score,
        factors=[
            ContributingFactor(
                name="intensity",
                label="Observed intensity",
                weight=14.0,
                normalized_value=1.0,
                raw_value=50.0,
                points=14.0,
                detail="test factor",
            )
        ],
        explanation="WARNING priority (score 50.00/100) because:\n- test factor",
        generated_at=NOW,
    )


def make_alert(**overrides: object) -> Alert:
    """Build a valid alert; every field can be overridden per test."""
    payload: dict = {
        "alert_id": "ALT-E0-WARNING-000000001234",
        "event_id": "E0",
        "track_id": "TRK-TEST-001",
        "event_type": EventType.EXTREME_RAINFALL,
        "severity": AlertSeverity.WARNING,
        "priority": AlertSeverity.WARNING,
        "title": "WARNING Extreme Rainfall alert",
        "message": "A synthetic event was detected.",
        "created_at": NOW,
        "updated_at": NOW,
        "valid_from": NOW,
        "valid_until": NOW + timedelta(hours=6),
        "status": AlertStatus.NEW,
        "confidence_label": ConfidenceLabel.LOW,
        "risk": make_assessment(),
        "current_lat": 13.0,
        "current_lon": 77.0,
        "intensity": 45.0,
        "source_components": ["events", "tracking"],
        "deduplication_key": "E0|WARNING|000000001234",
    }
    payload.update(overrides)
    return Alert(**payload)


# --------------------------------------------------------------------------- #
# Severity ordering
# --------------------------------------------------------------------------- #
def test_severity_ladder_is_ordered() -> None:
    order = [
        AlertSeverity.INFO,
        AlertSeverity.WATCH,
        AlertSeverity.ADVISORY,
        AlertSeverity.WARNING,
        AlertSeverity.SEVERE_WARNING,
        AlertSeverity.EXTREME,
    ]
    ranks = [severity.rank for severity in order]
    assert ranks == sorted(ranks) == [0, 1, 2, 3, 4, 5]


def test_max_severity_picks_the_highest() -> None:
    assert max_severity(AlertSeverity.WATCH, AlertSeverity.EXTREME) is AlertSeverity.EXTREME
    assert max_severity() is AlertSeverity.INFO


def test_severity_from_value_is_case_insensitive_and_validates() -> None:
    assert AlertSeverity.from_value("severe_warning") is AlertSeverity.SEVERE_WARNING
    with pytest.raises(ValueError, match="Unknown alert severity"):
        AlertSeverity.from_value("IMD_RED")


def test_alert_severity_is_not_an_official_warning_category() -> None:
    """The ladder must never be presented as an official IMD/WMO category."""
    assert "NOT official" in SEVERITY_DISCLAIMER
    assert "IMD" in SEVERITY_DISCLAIMER
    assert "WMO" in SEVERITY_DISCLAIMER
    for severity in AlertSeverity:
        assert severity.value.isupper()


def test_event_severity_maps_explicitly_onto_the_alert_ladder() -> None:
    assert (
        AlertSeverity.from_event_severity(EventSeverity.EXTREME)
        is AlertSeverity.SEVERE_WARNING
    )
    assert (
        AlertSeverity.from_event_severity(EventSeverity.NORMAL) is AlertSeverity.INFO
    )
    assert AlertSeverity.from_event_severity("SEVERE") is AlertSeverity.WARNING


def test_severity_for_score_respects_custom_thresholds() -> None:
    assert severity_for_score(80.0) is AlertSeverity.EXTREME
    assert severity_for_score(50.0) is AlertSeverity.SEVERE_WARNING
    assert severity_for_score(30.0) is AlertSeverity.WARNING
    assert severity_for_score(0.0) is AlertSeverity.INFO
    custom = SeverityThresholds(warning=10.0, severe_warning=20.0, extreme=30.0)
    assert severity_for_score(35.0, custom) is AlertSeverity.EXTREME
    assert severity_for_score(15.0, custom) is AlertSeverity.WARNING
    assert severity_for_score(50.0, DEFAULT_SEVERITY_THRESHOLDS) is AlertSeverity.SEVERE_WARNING
    assert severity_for_score(15.0, DEFAULT_SEVERITY_THRESHOLDS) is AlertSeverity.ADVISORY


def test_invalid_severity_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="Severity thresholds"):
        SeverityThresholds(warning=50.0, severe_warning=40.0, extreme=30.0)


# --------------------------------------------------------------------------- #
# Model validation
# --------------------------------------------------------------------------- #
def test_valid_alert_round_trips_through_json() -> None:
    alert = make_alert()
    restored = Alert.model_validate(alert.model_dump(mode="json"))
    assert restored.alert_id == alert.alert_id
    assert restored.status is AlertStatus.NEW


def test_alert_rejects_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        make_alert(accuracy=0.95)


def test_alert_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValidationError):
        make_alert(current_lat=200.0)
    with pytest.raises(ValidationError):
        make_alert(predicted_lon=400.0)


def test_alert_requires_both_predicted_coordinates_together() -> None:
    with pytest.raises(ValidationError, match="provided together"):
        make_alert(predicted_lat=13.5)
    ok = make_alert(predicted_lat=13.5, predicted_lon=77.5)
    assert ok.has_prediction is True


def test_alert_rejects_inverted_validity_window() -> None:
    with pytest.raises(ValidationError, match="valid_until"):
        make_alert(valid_from=NOW, valid_until=NOW - timedelta(hours=1))


def test_acknowledged_alert_requires_a_timestamp() -> None:
    with pytest.raises(ValidationError, match="ACKNOWLEDGED alert must carry"):
        make_alert(status=AlertStatus.ACKNOWLEDGED, acknowledged_at=None)


def test_resolved_alert_requires_a_timestamp() -> None:
    with pytest.raises(ValidationError, match="RESOLVED alert must carry"):
        make_alert(status=AlertStatus.RESOLVED, resolved_at=None)


def test_alert_rejects_duplicate_source_components() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        make_alert(source_components=["events", "events"])


def test_naive_timestamps_are_interpreted_as_utc() -> None:
    naive = datetime(2024, 6, 1, 12, 0)
    alert = make_alert(created_at=naive, updated_at=naive, valid_from=naive)
    assert alert.created_at.tzinfo is UTC


def test_alert_status_helpers() -> None:
    assert AlertStatus.NEW.is_open and not AlertStatus.NEW.is_terminal
    assert AlertStatus.ACKNOWLEDGED.is_open
    assert AlertStatus.RESOLVED.is_terminal and not AlertStatus.RESOLVED.is_open
    assert AlertStatus.EXPIRED.is_terminal


def test_alert_status_from_value_validates() -> None:
    assert AlertStatus.from_value("acknowledged") is AlertStatus.ACKNOWLEDGED
    with pytest.raises(ValueError, match="Unknown alert status"):
        AlertStatus.from_value("CLOSED")



# --------------------------------------------------------------------------- #
# Confidence labelling is about inputs, not accuracy
# --------------------------------------------------------------------------- #
def test_confidence_label_reflects_input_completeness() -> None:
    assert (
        ConfidenceLabel.for_components(["events", "tracking", "gnn", "downscaling"])
        is ConfidenceLabel.HIGH
    )
    assert (
        ConfidenceLabel.for_components(["events", "tracking", "gnn"])
        is ConfidenceLabel.MEDIUM
    )
    assert (
        ConfidenceLabel.for_components(["events", "tracking", "downscaling"])
        is ConfidenceLabel.MEDIUM
    )
    assert ConfidenceLabel.for_components(["events"]) is ConfidenceLabel.LOW
    assert ConfidenceLabel.for_components([]) is ConfidenceLabel.LOW


def test_notices_reject_operational_claims() -> None:
    """Neither the alert nor the risk notice may imply operational accuracy."""
    for notice in (ALERT_NOTICE, make_assessment().notice):
        lowered = notice.lower()
        assert "prototype" in lowered
        assert "not an official" in lowered or "not a validated" in lowered
        for forbidden in ("guaranteed", "95%", "exact evacuation"):
            assert forbidden not in lowered


# --------------------------------------------------------------------------- #
# Deterministic identifiers and deduplication keys
# --------------------------------------------------------------------------- #
def test_time_bucket_is_epoch_aligned_and_stable() -> None:
    first = time_bucket(datetime(2024, 6, 1, 6, 5, tzinfo=UTC))
    second = time_bucket(datetime(2024, 6, 1, 6, 55, tzinfo=UTC))
    third = time_bucket(datetime(2024, 6, 1, 7, 5, tzinfo=UTC))
    assert first == second
    assert first != third


def test_deduplication_key_combines_event_level_and_bucket() -> None:
    key = deduplication_key("E0", AlertSeverity.WARNING, T0)
    assert key == "E0|WARNING|" + time_bucket(T0)


def test_deduplication_key_changes_with_level() -> None:
    warning = deduplication_key("E0", AlertSeverity.WARNING, T0)
    severe = deduplication_key("E0", AlertSeverity.SEVERE_WARNING, T0)
    assert warning != severe


def test_alert_id_is_deterministic_and_contains_no_randomness() -> None:
    first = build_alert_id("E0", AlertSeverity.WARNING, T0)
    second = build_alert_id("E0", AlertSeverity.WARNING, T0)
    assert first == second
    assert first.startswith("ALT-E0-WARNING-")
    assert "|" not in first


def test_time_bucket_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive number of minutes"):
        time_bucket(T0, 0)


# --------------------------------------------------------------------------- #
# Store container
# --------------------------------------------------------------------------- #
def test_store_result_round_trips_and_keeps_duplicate_ids_out() -> None:
    alert = make_alert()
    result = AlertStoreResult(alerts=[alert, alert], created_count=1)
    restored = AlertStoreResult.from_dict(result.as_dict())
    assert restored.alert_count == 1
    assert restored.metadata["skipped_records"] == 1


def test_store_result_skips_malformed_records() -> None:
    payload = AlertStoreResult(alerts=[make_alert()]).as_dict()
    payload["alerts"].append({"alert_id": "broken"})
    restored = AlertStoreResult.from_dict(payload)
    assert restored.alert_count == 1
    assert restored.metadata["skipped_records"] == 1

