"""Tests for the tracking API endpoints (SIH26078, Phase 2).

The API is fed by a real tracking run over Phase 1 events, so every response is
produced by the same code path the CLI uses - nothing is fabricated in the router.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from extreme_weather.anomaly.models import AnomalyAnalysisResult
from extreme_weather.api.main import create_app
from extreme_weather.config import Settings, get_settings
from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent, Severity
from extreme_weather.tracking.service import TrackingService
from extreme_weather.tracking.store import TrackStore

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)


def make_event(
    event_id: str,
    *,
    hour: int,
    latitude: float = 13.0,
    longitude: float = 77.0,
    intensity: float = 20.0,
    event_type: EventType = EventType.EXTREME_RAINFALL,
) -> ExtremeWeatherEvent:
    """Build a valid Phase 1 event used as tracking input."""
    return ExtremeWeatherEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=T0 + timedelta(hours=hour),
        latitude=latitude,
        longitude=longitude,
        intensity=intensity,
        anomaly_score=intensity,
        severity=Severity.MODERATE,
        affected_radius_km=40.0,
        cell_count=12,
        variable="precipitation",
        units="mm h-1",
    )


def storm_events() -> list[ExtremeWeatherEvent]:
    """A three-hour eastward storm plus an unrelated one-hour wind event."""
    return [
        make_event("E0", hour=0, longitude=77.00, intensity=20.0),
        make_event("E1", hour=1, longitude=77.10, intensity=30.0),
        make_event("E2", hour=2, longitude=77.20, intensity=45.0),
        make_event("W0", hour=2, latitude=22.0, longitude=80.0, event_type=EventType.EXTREME_WIND),
    ]


@pytest.fixture()
def track_settings(tmp_path: Path) -> Settings:
    """Settings whose processed directory is a temporary folder."""
    return Settings(data_dir=tmp_path, model_dir=tmp_path / "models")


def build_tracking(settings: Settings) -> None:
    """Run the real tracking pipeline into the temporary processed directory."""
    from extreme_weather.anomaly.service import EventStore

    events_path = settings.processed_dir / "events.json"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    analysis = AnomalyAnalysisResult(events=tuple(storm_events()), variables=("precipitation",))
    events_path.write_text(json.dumps(analysis.as_dict()), encoding="utf-8")

    TrackingService(
        event_store=EventStore(events_path),
        track_store=TrackStore(settings.processed_dir / "tracks.json"),
    ).run(write_output=True)


@pytest.fixture()
def tracked_client(track_settings: Settings) -> Iterator[TestClient]:
    """Test client backed by a real tracking run."""
    build_tracking(track_settings)
    application = create_app(track_settings)
    application.dependency_overrides[get_settings] = lambda: track_settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


@pytest.fixture()
def empty_client(track_settings: Settings) -> Iterator[TestClient]:
    """Test client without any tracking file."""
    application = create_app(track_settings)
    application.dependency_overrides[get_settings] = lambda: track_settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# GET /api/v1/tracks
# --------------------------------------------------------------------------- #
def test_track_listing_returns_real_tracks(tracked_client: TestClient) -> None:
    response = tracked_client.get("/api/v1/tracks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2  # the rainfall track plus the wind event
    assert payload["message"] is None
    assert payload["source"].endswith("tracks.json")
    assert payload["generated_at"] is not None

    rainfall = next(
        track for track in payload["tracks"] if track["event_type"] == "EXTREME_RAINFALL"
    )
    assert rainfall["track_id"] == "TRK-EXTREME_RAINFALL-20240601T0000-001"
    assert rainfall["status"] == "ACTIVE"
    assert len(rainfall["points"]) == 3  # point_count is derived, not serialised
    assert rainfall["duration_hours"] == pytest.approx(2.0)
    assert rainfall["peak_intensity"] == pytest.approx(45.0)
    assert rainfall["max_speed_kmh"] > 0.0
    assert len(rainfall["points"]) == 3
    assert rainfall["points"][0]["speed_kmh"] == 0.0
    assert rainfall["points"][0]["direction"] == "CALM"


def test_track_listing_filters_by_event_type(tracked_client: TestClient) -> None:
    rainfall = tracked_client.get("/api/v1/tracks", params={"event_type": "EXTREME_RAINFALL"})
    wind = tracked_client.get("/api/v1/tracks", params={"event_type": "EXTREME_WIND"})
    temperature = tracked_client.get("/api/v1/tracks", params={"event_type": "EXTREME_TEMPERATURE"})

    assert rainfall.json()["count"] == 1
    assert wind.json()["count"] == 1
    assert temperature.json()["count"] == 0


def test_track_listing_filters_by_status(tracked_client: TestClient) -> None:
    active = tracked_client.get("/api/v1/tracks", params={"status": "ACTIVE"})
    ended = tracked_client.get("/api/v1/tracks", params={"status": "ENDED"})

    assert active.json()["count"] == 1
    assert ended.json()["count"] == 0


def test_track_listing_filters_by_time_window(tracked_client: TestClient) -> None:
    everything = tracked_client.get("/api/v1/tracks").json()["tracks"]

    # `start_time` keeps tracks that END at or after it: nothing is that late.
    after_everything = tracked_client.get(
        "/api/v1/tracks", params={"start_time": (T0 + timedelta(days=5)).isoformat()}
    ).json()
    # `end_time` keeps tracks that START at or before it: only the storm, which
    # starts at t0; the wind event starts at t+2.
    before_the_second_step = tracked_client.get(
        "/api/v1/tracks", params={"end_time": (T0 + timedelta(hours=1)).isoformat()}
    ).json()

    assert after_everything["count"] == 0
    assert before_the_second_step["count"] == 1
    assert before_the_second_step["tracks"][0]["event_type"] == "EXTREME_RAINFALL"
    assert len(everything) == 2


def test_track_listing_reports_an_empty_filter_result(tracked_client: TestClient) -> None:
    payload = tracked_client.get(
        "/api/v1/tracks", params={"start_time": (T0 + timedelta(days=5)).isoformat()}
    ).json()

    assert payload["count"] == 0
    assert payload["tracks"] == []
    assert "No track matches" in payload["message"]


def test_track_listing_is_empty_without_tracking(empty_client: TestClient) -> None:
    response = empty_client.get("/api/v1/tracks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 0
    assert payload["tracks"] == []
    assert "track_events.py" in payload["message"]


def test_track_endpoints_reject_invalid_filters(tracked_client: TestClient) -> None:
    assert tracked_client.get("/api/v1/tracks", params={"status": "EXPLODED"}).status_code == 422
    assert tracked_client.get("/api/v1/tracks", params={"event_type": "HAIL"}).status_code == 422
    assert (
        tracked_client.get("/api/v1/tracks", params={"start_time": "yesterday"}).status_code == 422
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/tracks/{track_id}
# --------------------------------------------------------------------------- #
def test_single_track_lookup(tracked_client: TestClient) -> None:
    response = tracked_client.get("/api/v1/tracks/TRK-EXTREME_RAINFALL-20240601T0000-001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["event_type"] == "EXTREME_RAINFALL"
    assert payload["status"] == "ACTIVE"
    assert len(payload["points"]) == 3
    assert payload["intensity_trend"] == "STRENGTHENING"
    assert payload["points"][-1]["event_id"] == "E2"


def test_unknown_track_returns_404(tracked_client: TestClient) -> None:
    response = tracked_client.get("/api/v1/tracks/TRK-EXTREME_RAINFALL-19700101T0000-999")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not found" in detail
    assert "2 track(s)" in detail


def test_track_lookup_without_tracking_returns_404(empty_client: TestClient) -> None:
    response = empty_client.get("/api/v1/tracks/TRK-EXTREME_RAINFALL-20240601T0000-001")

    assert response.status_code == 404
    assert "track_events.py" in response.json()["detail"]


def test_corrupt_tracking_file_is_reported(track_settings: Settings) -> None:
    track_settings.processed_dir.mkdir(parents=True, exist_ok=True)
    (track_settings.processed_dir / "tracks.json").write_text("{not json", encoding="utf-8")

    application = create_app(track_settings)
    application.dependency_overrides[get_settings] = lambda: track_settings
    try:
        with TestClient(application) as client:
            response = client.get("/api/v1/tracks")
    finally:
        application.dependency_overrides.clear()

    assert response.status_code == 422
    assert "Could not read tracking file" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# GET /api/v1/tracks/{track_id}/trajectory
# --------------------------------------------------------------------------- #
def test_track_trajectory(tracked_client: TestClient) -> None:
    response = tracked_client.get("/api/v1/tracks/TRK-EXTREME_RAINFALL-20240601T0000-001/trajectory")

    assert response.status_code == 200
    payload = response.json()
    assert payload["track_id"] == "TRK-EXTREME_RAINFALL-20240601T0000-001"
    assert payload["event_type"] == "EXTREME_RAINFALL"
    assert payload["point_count"] == 3
    assert payload["distance_travelled_km"] > 0.0
    assert len(payload["trajectory"]) == 3

    first, last = payload["trajectory"][0], payload["trajectory"][-1]
    assert first["event_id"] == "E0"
    assert first["direction"] == "CALM"
    assert first["speed_kmh"] == 0.0
    assert last["event_id"] == "E2"
    assert last["longitude"] == pytest.approx(77.20)
    assert last["latitude"] == pytest.approx(13.0)
    assert -90.0 <= last["latitude"] <= 90.0
    assert -180.0 <= last["longitude"] <= 180.0


def test_track_trajectory_of_unknown_track_returns_404(tracked_client: TestClient) -> None:
    response = tracked_client.get("/api/v1/tracks/nope/trajectory")

    assert response.status_code == 404


def test_events_endpoint_still_works_alongside_tracks(track_settings: Settings) -> None:
    """Registering the tracking router must not break the Phase 1 events API."""
    from extreme_weather.anomaly.service import EventStore

    build_tracking(track_settings)
    application = create_app(track_settings)
    application.dependency_overrides[get_settings] = lambda: track_settings
    try:
        with TestClient(application) as client:
            events = client.get("/api/v1/events")
            tracks = client.get("/api/v1/tracks")
            schema = client.get("/openapi.json").json()
    finally:
        application.dependency_overrides.clear()

    assert events.status_code == 200
    assert events.json()["count"] == 4
    assert tracks.status_code == 200
    assert "/api/v1/tracks/{track_id}/trajectory" in schema["paths"]
    assert "/api/v1/events" in schema["paths"]
    assert EventStore is not None