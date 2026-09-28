"""Tests for deterministic event matching (SIH26078, Phase 2).

The matcher must link the same storm across steps, refuse implausible links and
assign each detection at most once - deterministically, with no randomness.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent, Severity
from extreme_weather.tracking.matching import (
    MatchingConfig,
    TrackingError,
    intensity_mismatch,
    match_cost,
    match_events,
)

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(hours=1)


def make_event(
    event_id: str,
    *,
    timestamp: datetime = T0,
    latitude: float = 13.0,
    longitude: float = 77.0,
    intensity: float = 20.0,
    event_type: EventType = EventType.EXTREME_RAINFALL,
    severity: Severity = Severity.MODERATE,
) -> ExtremeWeatherEvent:
    """Build a minimal but valid event for the matching tests."""
    return ExtremeWeatherEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        intensity=intensity,
        anomaly_score=intensity,
        severity=severity,
        affected_radius_km=40.0,
        cell_count=10,
        variable="precipitation",
        units="mm h-1",
    )


# --------------------------------------------------------------------------- #
# Cost function
# --------------------------------------------------------------------------- #
def test_cost_grows_with_distance() -> None:
    config = MatchingConfig()
    previous = make_event("A")
    near = make_event("B", timestamp=T1, latitude=13.05)
    far = make_event("C", timestamp=T1, latitude=13.5)

    assert match_cost(previous, near, config) < match_cost(previous, far, config)
    assert match_cost(previous, near, config) >= 0.0


def test_cost_grows_with_intensity_difference() -> None:
    config = MatchingConfig()
    previous = make_event("A", intensity=20.0)
    similar = make_event("B", timestamp=T1, intensity=21.0)
    different = make_event("C", timestamp=T1, intensity=80.0)

    assert match_cost(previous, similar, config) < match_cost(previous, different, config)


def test_intensity_mismatch_bounds() -> None:
    assert intensity_mismatch(10.0, 10.0) == pytest.approx(0.0)
    assert intensity_mismatch(10.0, 20.0) == pytest.approx(0.5)
    assert 0.0 <= intensity_mismatch(10.0, 0.0) <= 1.0
    assert intensity_mismatch(0.0, 0.0) == 0.0
    assert intensity_mismatch(float("nan"), 5.0) == 0.0


# --------------------------------------------------------------------------- #
# Rejection rules
# --------------------------------------------------------------------------- #
def test_event_type_mismatch_is_rejected() -> None:
    previous = make_event("A")
    other = make_event("B", timestamp=T1, event_type=EventType.EXTREME_WIND)

    result = match_events([previous], [other])

    assert result.pairs == {}
    assert result.rejected_pairs == 1
    assert result.unmatched_previous == ("A",)
    assert result.unmatched_next == ("B",)


def test_distance_rejection() -> None:
    previous = make_event("A", latitude=10.0)
    distant = make_event("B", timestamp=T1, latitude=20.0)  # ~1100 km

    assert match_events([previous], [distant], MatchingConfig(max_distance_km=250.0)).pairs == {}
    # The same pair matches when both the distance and the speed gate are opened.
    assert match_events(
        [previous], [distant], MatchingConfig(max_distance_km=5000.0, max_speed_kmh=5000.0)
    ).pairs == {"A": "B"}


def test_speed_rejection() -> None:
    previous = make_event("A", latitude=10.0)
    jump = make_event("B", timestamp=T1, latitude=10.2)  # ~22 km in 1 h

    strict = MatchingConfig(max_distance_km=500.0, max_speed_kmh=10.0)
    result = match_events([previous], [jump], strict)

    assert result.pairs == {}
    assert result.rejected_pairs == 1


def test_time_gap_rejection() -> None:
    previous = make_event("A")
    late = make_event("B", timestamp=T0 + timedelta(hours=10))

    assert match_events([previous], [late], MatchingConfig(max_time_gap_hours=6.0)).pairs == {}
    # Same time (zero gap) is also rejected: a track must move forward in time.
    assert match_events([previous], [make_event("C", timestamp=T0)]).pairs == {}
    assert match_events([previous], [make_event("D", timestamp=T0 - timedelta(hours=1))]).pairs == {}


def test_same_location_is_matched() -> None:
    previous = make_event("A")
    same = make_event("B", timestamp=T1, latitude=13.0, longitude=77.0)

    result = match_events([previous], [same])

    assert result.pairs == {"A": "B"}
    assert result.match_count == 1
    assert result.unmatched_previous == ()
    assert result.unmatched_next == ()
    assert result.continuation_of("A") == "B"


# --------------------------------------------------------------------------- #
# One-to-one assignment and determinism
# --------------------------------------------------------------------------- #
def test_assignment_is_one_to_one() -> None:
    previous = [make_event("A", latitude=13.0), make_event("B", latitude=13.001)]
    following = [
        make_event("C", timestamp=T1, latitude=13.0005),
        make_event("D", timestamp=T1, latitude=13.002),
    ]

    result = match_events(previous, following)

    assert sorted(result.pairs.values()) == ["C", "D"]
    assert len(set(result.pairs.values())) == 2  # no detection used twice
    assert len(result.pairs) == 2


def test_one_detection_cannot_continue_two_tracks() -> None:
    previous = [make_event("A", latitude=13.0), make_event("B", latitude=13.0002)]
    following = [make_event("C", timestamp=T1, latitude=13.0001)]

    result = match_events(previous, following)

    assert len(result.pairs) == 1
    # A is closer to C than B is, so the cheapest pair wins.
    assert result.pairs == {"A": "C"}
    assert result.unmatched_previous == ("B",)
    assert result.unmatched_next == ()


def test_tie_breaking_is_deterministic_and_order_independent() -> None:
    # A and B are (almost) equidistant from C. Whether floating-point noise
    # favours A or B is irrelevant: the requirement is that the *same* choice is
    # made for every input order.
    previous = [make_event("B", latitude=13.0), make_event("A", latitude=13.002)]
    following = [make_event("C", timestamp=T1, latitude=13.001)]

    first = match_events(previous, following)
    second = match_events(list(reversed(previous)), following)
    third = match_events([make_event("A", latitude=13.002), make_event("B", latitude=13.0)], following)

    assert first.pairs == second.pairs == third.pairs
    assert len(first.pairs) == 1  # one-to-one: C can only continue one track
    assert set(first.pairs.values()) == {"C"}
    assert len(first.unmatched_previous) == 1


def test_matching_prefers_the_cheapest_pair() -> None:
    # D sits on top of A/B, C is far away. With a 10 km gate only the two
    # zero-distance pairs survive, and D can only be used once.
    previous = [make_event("A", latitude=13.00), make_event("B", latitude=13.00)]
    following = [
        make_event("C", timestamp=T1, latitude=13.30),
        make_event("D", timestamp=T1, latitude=13.00),
    ]

    result = match_events(previous, following, MatchingConfig(max_distance_km=10.0))

    assert result.pairs == {"A": "D"}
    assert result.unmatched_previous == ("B",)
    assert result.unmatched_next == ("C",)
    assert result.rejected_pairs == 2  # both pairs involving C are rejected


def test_duplicate_ids_in_a_step_are_rejected() -> None:
    duplicated = [make_event("A"), make_event("A", timestamp=T1)]

    with pytest.raises(TrackingError, match="Duplicate event ids"):
        match_events(duplicated, [make_event("B", timestamp=T1)])


def test_empty_inputs_produce_no_pairs() -> None:
    assert match_events([], []).pairs == {}
    assert match_events([make_event("A")], []).pairs == {}
    assert match_events([], [make_event("B", timestamp=T1)]).pairs == {}


def test_matching_config_validation() -> None:
    with pytest.raises(ValueError):
        MatchingConfig(max_distance_km=-1.0)
    with pytest.raises(ValueError):
        MatchingConfig(distance_weight=-0.5)
    with pytest.raises(ValueError):
        MatchingConfig(max_time_gap_hours=0.0)
