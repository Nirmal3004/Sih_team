"""Tests for the events API endpoints.

The API is fed by a real analysis: a synthetic dataset is analysed with the
production service, the result is written through :class:`EventStore`, and the
endpoints read that file. Nothing is fabricated in the router.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from extreme_weather.anomaly.models import AnomalyAnalysisResult
from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore
from extreme_weather.api.main import create_app
from extreme_weather.config import Settings, get_settings
from extreme_weather.data.generator import (
    ExtremeEventConfig,
    SyntheticConfig,
    generate_synthetic_dataset,
)

EVENT_CENTRE = (13.0, 77.0)


def build_config(*, event_enabled: bool, seed: int) -> SyntheticConfig:
    """Return a small, reproducible synthetic configuration."""
    return SyntheticConfig(
        seed=seed,
        time_steps=16,
        latitude_min=12.0,
        latitude_max=14.0,
        longitude_min=76.0,
        longitude_max=78.0,
        resolution_deg=0.25,
        event=ExtremeEventConfig(
            enabled=event_enabled,
            center_latitude=EVENT_CENTRE[0],
            center_longitude=EVENT_CENTRE[1],
            radius_deg=0.5,
            start_index=4,
            duration_steps=6,
            peak_precipitation_mm_per_hour=55.0,
            temperature_anomaly_k=-4.0,
        ),
    )


@pytest.fixture()
def analysis_settings(tmp_path: Path) -> Settings:
    """Settings whose processed directory is a temporary folder."""
    return Settings(data_dir=tmp_path, model_dir=tmp_path / "models")


@pytest.fixture()
def analysed_client(
    analysis_settings: Settings,
) -> Iterator[tuple[TestClient, AnomalyAnalysisResult]]:
    """Test client backed by a real analysis of synthetic data."""
    target = generate_synthetic_dataset(build_config(event_enabled=True, seed=4242))
    reference = generate_synthetic_dataset(build_config(event_enabled=False, seed=4243))

    store = EventStore(analysis_settings.processed_dir / "events.json")
    service = AnomalyDetectionService(store=store)
    result = service.analyse(
        target,
        reference=reference,
        variables=("precipitation",),
        source_path="raw/sample_history.nc",
        write_outputs=True,
    )

    application = create_app(analysis_settings)
    application.dependency_overrides[get_settings] = lambda: analysis_settings
    try:
        with TestClient(application) as client:
            yield client, result
    finally:
        application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# GET /api/v1/events
# --------------------------------------------------------------------------- #
def test_event_listing_returns_detected_events(analysed_client) -> None:
    client, result = analysed_client

    response = client.get("/api/v1/events")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == len(result.events)
    assert payload["count"] > 0, "the injected synthetic event must be detected"
    assert payload["message"] is None
    assert payload["source"].endswith("events.json")
    assert payload["generated_at"] is not None

    first = payload["events"][0]
    assert set(first) >= {
        "event_id",
        "event_type",
        "timestamp",
        "latitude",
        "longitude",
        "intensity",
        "anomaly_score",
        "severity",
        "affected_radius_km",
        "cell_count",
    }
    assert first["event_type"] == "EXTREME_RAINFALL"
    assert -90.0 <= first["latitude"] <= 90.0
    assert -180.0 <= first["longitude"] <= 180.0
    assert first["cell_count"] >= 1
    assert first["affected_radius_km"] >= 0.0
    # The detected event sits on the injected centre.
    assert abs(first["latitude"] - EVENT_CENTRE[0]) <= 0.5
    assert abs(first["longitude"] - EVENT_CENTRE[1]) <= 0.5


def test_event_listing_filters_by_type_and_severity(analysed_client) -> None:
    client, _ = analysed_client
    everything = client.get("/api/v1/events").json()["events"]

    by_type = client.get("/api/v1/events", params={"event_type": "EXTREME_RAINFALL"}).json()
    assert by_type["count"] == len(everything)
    assert client.get("/api/v1/events", params={"event_type": "EXTREME_WIND"}).json()["count"] == 0

    severities = {event["severity"] for event in everything}
    target = sorted(severities)[0]
    filtered = client.get("/api/v1/events", params={"severity": target}).json()
    assert filtered["count"] >= 1
    assert {event["severity"] for event in filtered["events"]} == {target}


def test_event_listing_filters_by_time_window(analysed_client) -> None:
    client, result = analysed_client
    stamps = sorted(event.timestamp for event in result.events)
    midpoint = stamps[len(stamps) // 2]

    early = client.get(
        "/api/v1/events", params={"end_time": midpoint.isoformat()}
    ).json()
    late = client.get(
        "/api/v1/events", params={"start_time": midpoint.isoformat()}
    ).json()

    assert early["count"] >= 1
    assert late["count"] >= 1
    assert early["count"] + late["count"] >= len(result.events)
    assert all(
        datetime.fromisoformat(event["timestamp"]) <= midpoint
        for event in early["events"]
    )


def test_event_listing_filters_by_min_cell_count(analysed_client) -> None:
    client, result = analysed_client
    largest = max(event.cell_count for event in result.events)

    payload = client.get("/api/v1/events", params={"min_cell_count": largest}).json()

    assert payload["count"] >= 1
    assert all(event["cell_count"] >= largest for event in payload["events"])


def test_event_listing_reports_an_empty_filter_result(analysed_client) -> None:
    client, _ = analysed_client

    payload = client.get(
        "/api/v1/events", params={"start_time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}
    ).json()

    assert payload["count"] == 0
    assert payload["events"] == []
    assert "No event matches" in payload["message"]


def test_event_listing_is_empty_without_an_analysis(analysis_settings: Settings) -> None:
    application = create_app(analysis_settings)
    application.dependency_overrides[get_settings] = lambda: analysis_settings
    try:
        with TestClient(application) as client:
            response = client.get("/api/v1/events")
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert payload["events"] == []
    assert "detect_anomalies.py" in payload["message"]


def test_event_endpoints_reject_invalid_filters(analysed_client) -> None:
    client, _ = analysed_client

    assert client.get("/api/v1/events", params={"event_type": "EXTREME_HAIL"}).status_code == 422
    assert client.get("/api/v1/events", params={"severity": "CATASTROPHIC"}).status_code == 422
    assert client.get("/api/v1/events", params={"min_cell_count": 0}).status_code == 422
    assert client.get("/api/v1/events", params={"start_time": "not-a-date"}).status_code == 422


# --------------------------------------------------------------------------- #
# GET /api/v1/events/{event_id}
# --------------------------------------------------------------------------- #
def test_single_event_lookup(analysed_client) -> None:
    client, result = analysed_client
    expected = result.events[0]

    response = client.get(f"/api/v1/events/{expected.event_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"] == expected.event_id
    assert payload["severity"] == expected.severity.value
    assert payload["cell_count"] == expected.cell_count
    assert payload["timestamp"].startswith(expected.timestamp.isoformat()[:16])


def test_unknown_event_id_returns_404(analysed_client) -> None:
    client, result = analysed_client

    response = client.get("/api/v1/events/EVT-EXTREME_RAINFALL-19700101T0000-999")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not found" in detail
    assert str(len(result.events)) in detail


def test_event_lookup_without_analysis_returns_404(analysis_settings: Settings) -> None:
    application = create_app(analysis_settings)
    application.dependency_overrides[get_settings] = lambda: analysis_settings
    try:
        with TestClient(application) as client:
            response = client.get("/api/v1/events/EVT-EXTREME_RAINFALL-20240601T0000-001")
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 404
    assert "detect_anomalies.py" in response.json()["detail"]


def test_corrupt_analysis_file_is_reported(analysis_settings: Settings) -> None:
    analysis_settings.processed_dir.mkdir(parents=True, exist_ok=True)
    broken = analysis_settings.processed_dir / "events.json"
    broken.write_text("{not valid json", encoding="utf-8")

    application = create_app(analysis_settings)
    application.dependency_overrides[get_settings] = lambda: analysis_settings
    try:
        with TestClient(application) as client:
            response = client.get("/api/v1/events")
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422
    assert "Could not read analysis file" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Stored analysis file (stable schema)
# --------------------------------------------------------------------------- #
def test_stored_analysis_file_has_a_stable_schema(analysis_settings: Settings) -> None:
    target = generate_synthetic_dataset(build_config(event_enabled=True, seed=4242))
    reference = generate_synthetic_dataset(build_config(event_enabled=False, seed=4243))
    store = EventStore(analysis_settings.processed_dir / "events.json")

    service = AnomalyDetectionService(store=store)
    result = service.analyse(
        target, reference=reference, variables=("precipitation",), write_outputs=True
    )
    payload = json.loads(store.path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["project"] == "SIH26078"
    assert payload["score_name"] == "absolute"
    assert payload["grouping"] == "month"
    assert payload["variables"] == ["precipitation"]
    assert payload["event_count"] == result.event_count
    assert payload["metadata"]["threshold_disclaimer"].startswith("Prototype severity thresholds")
    assert "operational forecast" in payload["metadata"]["data_disclaimer"]
    assert "Phi^-1" in payload["metadata"]["score_formula"]
    assert payload["metadata"]["thresholds"]["precipitation_absolute"]["steps"][0] == [10.0, "WATCH"]
    assert payload["events"][0]["event_id"] == result.events[0].event_id


def test_event_store_handles_missing_and_cleared_files(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "nested" / "events.json")

    assert store.exists() is False
    assert store.load() is None

    store.clear()  # must not raise on a missing file
