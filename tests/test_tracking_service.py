"""Tests for the tracking service, tracker and track store (SIH26078, Phase 2).

The service must be independent of the input order, deterministic in its ids and
metrics, and it must never touch ``data/raw``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from extreme_weather.anomaly.models import AnomalyAnalysisResult
from extreme_weather.anomaly.service import EventStore
from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent, Severity
from extreme_weather.schemas.track import EventTrack, TrackStatus
from extreme_weather.tracking.matching import MatchingConfig
from extreme_weather.tracking.models import TrackResult
from extreme_weather.tracking.service import TrackingError, TrackingService
from extreme_weather.tracking.store import TrackStore, TrackingStoreError
from extreme_weather.tracking.tracker import EventTracker, group_events_by_time, sort_events

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
    """Build a valid Phase 1 event at ``T0 + hour`` hours."""
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


def moving_storm() -> list[ExtremeWeatherEvent]:
    """Three hourly detections drifting east and intensifying."""
    return [
        make_event("E0", hour=0, longitude=77.00, intensity=20.0),
        make_event("E1", hour=1, longitude=77.10, intensity=30.0),
        make_event("E2", hour=2, longitude=77.20, intensity=45.0),
    ]


# --------------------------------------------------------------------------- #
# Sorting and grouping
# --------------------------------------------------------------------------- #
def test_events_are_sorted_and_grouped_by_time() -> None:
    events = [make_event("E2", hour=2), make_event("E0", hour=0), make_event("E1", hour=1)]
    events.append(make_event("A0", hour=0, latitude=20.0))  # same step, second event

    ordered = sort_events(events)
    grouped = group_events_by_time(events)

    assert [event.event_id for event in ordered] == ["A0", "E0", "E1", "E2"]
    assert list(grouped) == [T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)]
    assert [event.event_id for event in grouped[T0]] == ["A0", "E0"]


# --------------------------------------------------------------------------- #
# Track building
# --------------------------------------------------------------------------- #
def test_single_event_creates_a_new_track() -> None:
    tracks = EventTracker().track([make_event("E0", hour=0)])

    assert len(tracks) == 1
    track = tracks[0]
    assert track.status is TrackStatus.NEW
    assert track.point_count == 1
    assert track.duration_hours == 0.0
    assert track.track_id == "TRK-EXTREME_RAINFALL-20240601T0000-001"


def test_moving_storm_becomes_an_active_track() -> None:
    tracks = EventTracker().track(moving_storm())

    assert len(tracks) == 1
    track = tracks[0]
    assert track.status is TrackStatus.ACTIVE
    assert track.point_count == 3
    assert track.duration_hours == pytest.approx(2.0)
    assert track.peak_intensity == pytest.approx(45.0)
    assert track.intensity_trend == "STRENGTHENING"
    assert track.peak_severity is Severity.MODERATE
    assert [point.event_id for point in track.points] == ["E0", "E1", "E2"]


def test_disappearing_event_ends_the_track() -> None:
    # The storm is last seen at t+1 and is gone in the final step t+2.
    events = [
        make_event("E0", hour=0),
        make_event("E1", hour=1, longitude=77.1),
        make_event("F0", hour=2, latitude=30.0),
    ]

    tracks = EventTracker().track(events)
    ended = [track for track in tracks if track.status is TrackStatus.ENDED]

    assert len(ended) == 1
    assert ended[0].point_count == 2
    assert ended[0].duration_hours == pytest.approx(1.0)
    # The unrelated far-away event is a separate NEW track.
    assert len([track for track in tracks if track.status is TrackStatus.NEW]) == 1


def test_distant_event_is_not_connected() -> None:
    events = [make_event("E0", hour=0, longitude=77.0), make_event("E1", hour=1, longitude=85.0)]

    tracks = EventTracker(MatchingConfig(max_distance_km=250.0)).track(events)

    assert len(tracks) == 2
    assert all(track.point_count == 1 for track in tracks)
    assert {track.status for track in tracks} == {TrackStatus.NEW}


def test_multiple_simultaneous_events_are_tracked_separately() -> None:
    events = [
        make_event("A0", hour=0, latitude=13.0, longitude=77.0),
        make_event("B0", hour=0, latitude=20.0, longitude=80.0),
        make_event("A1", hour=1, latitude=13.01, longitude=77.0),
        make_event("B1", hour=1, latitude=20.01, longitude=80.0),
    ]

    tracks = EventTracker().track(events)

    assert len(tracks) == 2
    assert all(track.point_count == 2 for track in tracks)
    assert all(track.status is TrackStatus.ACTIVE for track in tracks)
    assert {track.track_id for track in tracks} == {
        "TRK-EXTREME_RAINFALL-20240601T0000-001",
        "TRK-EXTREME_RAINFALL-20240601T0000-002",
    }


def test_event_types_are_never_mixed_in_one_track() -> None:
    events = [
        make_event("R0", hour=0, event_type=EventType.EXTREME_RAINFALL),
        make_event("W0", hour=1, event_type=EventType.EXTREME_WIND),
    ]

    tracks = EventTracker().track(events)

    assert len(tracks) == 2
    assert {track.event_type for track in tracks} == {
        EventType.EXTREME_RAINFALL,
        EventType.EXTREME_WIND,
    }


def test_tracking_is_independent_of_input_order() -> None:
    events = moving_storm()
    forward = EventTracker().track(events)
    shuffled = EventTracker().track([events[2], events[0], events[1]])

    assert [track.track_id for track in forward] == [track.track_id for track in shuffled]
    assert forward[0].model_dump() == shuffled[0].model_dump()
    assert forward[0].points == shuffled[0].points


def test_tracking_is_deterministic_across_runs() -> None:
    first = EventTracker().track(moving_storm())
    second = EventTracker().track(moving_storm())

    assert first[0].track_id == second[0].track_id
    assert first[0].max_speed_kmh == second[0].max_speed_kmh
    assert first[0].duration_hours == second[0].duration_hours


def test_stationary_event_reports_calm_and_zero_speed() -> None:
    events = [
        make_event("E0", hour=0, longitude=77.0, intensity=20.0),
        make_event("E1", hour=1, longitude=77.0, intensity=20.0),
    ]

    track = EventTracker().track(events)[0]

    assert track.max_speed_kmh == 0.0
    assert track.mean_speed_kmh == 0.0
    assert track.distance_travelled_km == 0.0
    assert track.direction == "CALM"
    assert track.bearing_degrees is None
    assert track.points[0].speed_kmh == 0.0
    assert track.points[0].direction == "CALM"
    assert track.intensity_trend == "STABLE"


def test_weakening_event_is_reported() -> None:
    events = [
        make_event("E0", hour=0, intensity=50.0),
        make_event("E1", hour=1, intensity=10.0),
    ]

    assert EventTracker().track(events)[0].intensity_trend == "WEAKENING"


def test_moving_track_reports_speed_bearing_and_distance() -> None:
    track = EventTracker().track(moving_storm())[0]

    # ~11 km east in one hour.
    assert track.max_speed_kmh == pytest.approx(11.1, abs=0.5)
    assert track.distance_travelled_km == pytest.approx(22.2, abs=1.0)
    assert track.bearing_degrees == pytest.approx(90.0, abs=1.0)
    assert track.direction == "E"
    assert track.points[1].distance_from_previous_km == pytest.approx(11.1, abs=0.5)


def test_matching_can_be_disabled_for_diagnostics() -> None:
    tracks = EventTracker(match=False).track(moving_storm())

    assert len(tracks) == 3
    assert all(track.point_count == 1 for track in tracks)


def test_empty_input_produces_no_tracks() -> None:
    assert EventTracker().track([]) == []


# --------------------------------------------------------------------------- #
# Service and store
# --------------------------------------------------------------------------- #
def write_events(tmp_path: Path, events: list[ExtremeWeatherEvent]) -> Path:
    """Write a Phase 1 analysis file with the given events."""
    analysis = AnomalyAnalysisResult(events=tuple(events), variables=("precipitation",))
    path = tmp_path / "events.json"
    path.write_text(json.dumps(analysis.as_dict()), encoding="utf-8")
    return path


def test_service_tracks_events_and_writes_the_result(tmp_path: Path) -> None:
    events_path = write_events(tmp_path, moving_storm())
    tracks_path = tmp_path / "tracks.json"
    service = TrackingService(track_store=TrackStore(tracks_path))

    result = service.run(events_path=events_path, write_output=True)

    assert result.event_count == 3
    assert result.track_count == 1
    assert result.matched_event_count == 3
    assert result.duration_hours == pytest.approx(2.0)
    assert tracks_path.is_file()
    assert result.metadata["matching"]["max_distance_km"] == 250.0
    assert "operational forecast" in result.metadata["data_disclaimer"]

    reloaded = TrackStore(tracks_path).load()
    assert reloaded is not None
    assert reloaded.tracks[0].track_id == result.tracks[0].track_id
    assert reloaded.event_count == 3


def test_service_does_not_modify_raw_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    marker = raw_dir / "keepme.nc"
    marker.write_bytes(b"synthetic-raw-placeholder")

    events_path = write_events(tmp_path, moving_storm())
    service = TrackingService(track_store=TrackStore(tmp_path / "tracks.json"))
    service.run(events_path=events_path, write_output=True)

    assert marker.read_bytes() == b"synthetic-raw-placeholder"
    assert sorted(item.name for item in raw_dir.iterdir()) == ["keepme.nc"]


def test_service_reports_a_missing_analysis(tmp_path: Path) -> None:
    service = TrackingService(
        event_store=EventStore(tmp_path / "missing.json"),
        track_store=TrackStore(tmp_path / "tracks.json"),
    )

    with pytest.raises(TrackingError, match="No event analysis found"):
        service.run()


def test_service_accepts_events_directly(tmp_path: Path) -> None:
    service = TrackingService(track_store=TrackStore(tmp_path / "tracks.json"))

    result = service.run(moving_storm())

    assert result.track_count == 1
    assert not (tmp_path / "tracks.json").exists()  # not written unless requested


def test_store_supports_save_load_get_list_and_clear(tmp_path: Path) -> None:
    store = TrackStore(tmp_path / "tracks.json")
    result = TrackingService(track_store=store).run(moving_storm())

    assert store.exists() is False
    assert store.load() is None
    assert store.get("TRK-EXTREME_RAINFALL-20240601T0000-001") is None

    store.save(result)
    assert store.exists() is True
    assert store.get("TRK-EXTREME_RAINFALL-20240601T0000-001").point_count == 3
    assert store.get("nope") is None
    assert len(store.list()) == 1
    assert store.list(event_type="EXTREME_WIND") == []
    assert store.list(status="ACTIVE")[0].point_count == 3
    assert store.list(status="ENDED") == []
    assert store.list(start_time=T0 + timedelta(hours=1))[0].point_count == 3
    assert store.list(start_time=T0 + timedelta(days=2)) == []
    assert store.list(end_time=T0 - timedelta(hours=1)) == []
    assert store.list(end_time=T0 + timedelta(days=2))[0].point_count == 3

    store.clear()
    assert store.exists() is False
    store.clear()  # clearing twice is harmless


def test_store_reports_a_corrupt_file(tmp_path: Path) -> None:
    broken = tmp_path / "tracks.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(TrackingStoreError, match="Could not read tracking file"):
        TrackStore(broken).load()


def test_store_rejects_non_object_payloads(tmp_path: Path) -> None:
    listy = tmp_path / "tracks.json"
    listy.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TrackingStoreError, match="must contain a JSON object"):
        TrackStore(listy).load()


def test_result_helpers(tmp_path: Path) -> None:
    # The storm stops at t+1 and the last step (t+3) holds an unrelated event,
    # so the run contains one ENDED track and one NEW track.
    events = [
        make_event("E0", hour=0),
        make_event("E1", hour=1, longitude=77.1),
        make_event("F0", hour=3, latitude=30.0),
    ]
    result = TrackingService(track_store=TrackStore(tmp_path / "t.json")).run(events)

    assert result.longest() is not None
    assert result.longest().track_id == "TRK-EXTREME_RAINFALL-20240601T0000-001"
    assert result.fastest() is not None
    assert result.fastest().max_speed_kmh > 0.0
    assert len(result.by_status("ENDED")) == 1
    assert len(result.by_status("NEW")) == 1
    assert len(result.by_status("ACTIVE")) == 0
    assert TrackResult().longest() is None
    assert TrackResult().fastest() is None
    assert TrackResult().duration_hours == 0.0
