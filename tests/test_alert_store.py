"""Tests for the Phase 5 file-backed alert store (SIH26078).

The store has to survive the failure modes a JSON file really hits, and each of
them is asserted here: missing file, empty file, malformed JSON, malformed
records, duplicate ids and unwritable paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from extreme_weather.alerts.models import (
    Alert,
    AlertStatus,
    AlertStoreResult,
    ConfidenceLabel,
    ContributingFactor,
    RiskAssessment,
)
from extreme_weather.alerts.store import AlertStore, AlertStoreError
from extreme_weather.config import Settings
from extreme_weather.schemas.event import EventType

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_alert(alert_id: str = "ALT-E0-WARNING-1", **overrides: object) -> Alert:
    """Build a valid alert for store round-trip tests."""
    risk = RiskAssessment.from_score(
        score=40.0,
        factors=[
            ContributingFactor(
                name="intensity",
                label="Observed intensity",
                weight=14.0,
                normalized_value=0.5,
                points=7.0,
                detail="test",
            )
        ],
        explanation="WARNING priority because:\n- test",
        generated_at=NOW,
    )
    payload: dict = {
        "alert_id": alert_id,
        "event_id": "E0",
        "event_type": EventType.EXTREME_RAINFALL,
        "severity": "WARNING",
        "priority": "WARNING",
        "title": "WARNING Extreme Rainfall alert",
        "message": "synthetic",
        "created_at": NOW,
        "updated_at": NOW,
        "valid_from": NOW,
        "valid_until": NOW + timedelta(hours=6),
        "status": AlertStatus.NEW,
        "confidence_label": ConfidenceLabel.LOW,
        "risk": risk,
        "current_lat": 13.0,
        "current_lon": 77.0,
        "intensity": 30.0,
        "source_components": ["events"],
        "deduplication_key": f"E0|WARNING|{alert_id}",
    }
    payload.update(overrides)
    return Alert(**payload)


@pytest.fixture()
def store(tmp_path: Path) -> AlertStore:
    """An alert store in a temporary folder."""
    return AlertStore(tmp_path / "alerts.json")


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def test_missing_file_is_an_empty_store_not_an_error(store: AlertStore) -> None:
    assert store.exists() is False
    assert store.load() is None
    assert store.list() == []
    assert store.get("nope") is None


def test_empty_file_is_treated_as_an_empty_store(store: AlertStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("", encoding="utf-8")
    result = store.load()
    assert result is not None
    assert result.alert_count == 0
    assert result.metadata["empty_file"] is True


def test_save_and_load_round_trip(store: AlertStore) -> None:
    alert = make_alert()
    store.save(AlertStoreResult(alerts=[alert], created_count=1, generated_at=NOW))
    loaded = store.load()
    assert loaded is not None
    assert loaded.alert_count == 1
    assert loaded.created_count == 1
    assert loaded.alerts[0].alert_id == alert.alert_id
    assert loaded.generated_at == NOW


def test_stored_file_is_a_versioned_json_object(store: AlertStore) -> None:
    store.save(AlertStoreResult(alerts=[make_alert()], generated_at=NOW))
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["project"] == "SIH26078"
    assert payload["phase"].startswith("Phase 5")
    assert "status_counts" in payload
    assert "severity_counts" in payload


def test_get_returns_one_alert(store: AlertStore) -> None:
    store.save(AlertStoreResult(alerts=[make_alert()]))
    assert store.get("ALT-E0-WARNING-1") is not None
    assert store.get("missing") is None


def test_clear_removes_the_file(store: AlertStore) -> None:
    store.save(AlertStoreResult(alerts=[make_alert()]))
    assert store.exists() is True
    store.clear()
    assert store.exists() is False


# --------------------------------------------------------------------------- #
# Malformed input
# --------------------------------------------------------------------------- #
def test_malformed_json_raises_a_store_error(store: AlertStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AlertStoreError, match="Could not read alert file"):
        store.load()


def test_non_object_payload_raises_a_store_error(store: AlertStore) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AlertStoreError, match="must contain a JSON object"):
        store.load()


def test_malformed_records_are_skipped_not_fatal(store: AlertStore) -> None:
    payload = AlertStoreResult(alerts=[make_alert()]).as_dict()
    payload["alerts"].append({"alert_id": "incomplete"})
    payload["alerts"].append("not even an object")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    result = store.load()
    assert result is not None
    assert result.alert_count == 1
    assert result.metadata["skipped_records"] == 2


def test_duplicate_alert_ids_are_collapsed(store: AlertStore) -> None:
    payload = AlertStoreResult(alerts=[make_alert()]).as_dict()
    payload["alerts"].append(json.loads(json.dumps(payload["alerts"][0])))
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    result = store.load()
    assert result is not None
    assert result.alert_count == 1
    assert result.metadata["skipped_records"] == 1


def test_alerts_field_of_the_wrong_type_is_ignored(store: AlertStore) -> None:
    payload = {"alerts": "not a list", "generated_at": None}
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    result = store.load()
    assert result is not None
    assert result.alert_count == 0


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_list_filters_by_status_severity_and_type(store: AlertStore) -> None:
    store.save(
        AlertStoreResult(
            alerts=[
                make_alert("ALT-A", status=AlertStatus.ACTIVE),
                make_alert("ALT-B", severity="EXTREME", priority="EXTREME"),
            ]
        )
    )
    assert len(store.list(status=AlertStatus.ACTIVE)) == 1
    assert len(store.list(status="active")) == 1
    assert len(store.list(severity="EXTREME")) == 1
    assert len(store.list(event_type=EventType.EXTREME_RAINFALL)) == 2
    assert len(store.list(priority="EXTREME")) == 1
    assert len(store.list(status=AlertStatus.RESOLVED)) == 0


def test_list_filters_by_time_window(store: AlertStore) -> None:
    early = make_alert(
        "ALT-EARLY",
        created_at=NOW - timedelta(hours=10),
        updated_at=NOW - timedelta(hours=10),
    )
    late = make_alert("ALT-LATE")
    store.save(AlertStoreResult(alerts=[early, late]))
    assert len(store.list(start_time=NOW - timedelta(hours=1))) == 1
    assert len(store.list(end_time=NOW - timedelta(hours=1))) == 1
    assert len(store.list(start_time=NOW - timedelta(days=1))) == 2


def test_list_rejects_an_unknown_filter_value(store: AlertStore) -> None:
    store.save(AlertStoreResult(alerts=[make_alert()]))
    with pytest.raises(ValueError, match="Unknown alert status"):
        store.list(status="CLOSED")
    with pytest.raises(ValueError, match="Unknown alert severity"):
        store.list(severity="IMD_RED")


def test_open_alerts_excludes_terminal_states(store: AlertStore) -> None:
    store.save(
        AlertStoreResult(
            alerts=[
                make_alert("ALT-OPEN", status=AlertStatus.ACTIVE),
                make_alert(
                    "ALT-DONE",
                    status=AlertStatus.RESOLVED,
                    resolved_at=NOW,
                    updated_at=NOW,
                ),
            ]
        )
    )
    open_alerts = store.open_alerts()
    assert [alert.alert_id for alert in open_alerts] == ["ALT-OPEN"]


# --------------------------------------------------------------------------- #
# Defaults and paths
# --------------------------------------------------------------------------- #
def test_store_defaults_to_the_configured_processed_path() -> None:
    settings = Settings(data_dir=Path("data"), model_dir=Path("models"))
    default_store = AlertStore()
    assert default_store.path == settings.processed_dir / settings.alerts_filename
    assert default_store.path.name == "alerts.json"


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    nested = AlertStore(tmp_path / "a" / "b" / "alerts.json")
    nested.save(AlertStoreResult(alerts=[make_alert()]))
    assert nested.path.is_file()


def test_write_failure_raises_a_store_error(tmp_path: Path) -> None:
    """A directory in place of the file must surface as AlertStoreError."""
    blocked = tmp_path / "alerts.json"
    blocked.mkdir(parents=True, exist_ok=True)
    store = AlertStore(blocked)
    with pytest.raises(AlertStoreError, match="Could not write alerts"):
        store.save(AlertStoreResult(alerts=[]))

