"""Tests for Phase 7 tracking validation.

Tracking diagnostics describe what the tracker produced. They are explicitly
**not** accuracy, because this project has no reference trajectories. The tests
pin that wording down: a future change that quietly drops the caveat is a
regression, even though every number would still be right.
"""

from __future__ import annotations

import pytest

from extreme_weather.validation.ground_truth import GROUND_TRUTH_UNAVAILABLE
from extreme_weather.validation.metrics import compute_tracking_diagnostics


class Point:
    def __init__(self, intensity: float) -> None:
        self.intensity = intensity


class Track:
    def __init__(
        self,
        points: list[Point],
        *,
        duration: float = 3.0,
        displacement: float = 45.0,
        speed: float = 30.0,
        direction: str = "NE",
        status: str = "ENDED",
    ) -> None:
        self.points = points
        self.duration_hours = duration
        self.displacement_km = displacement
        self.max_speed_kmh = speed
        self.direction = direction
        self.status = status


def test_tracks_are_summarised() -> None:
    payload = compute_tracking_diagnostics(
        [Track([Point(4.0), Point(6.0), Point(7.0)])],
        split="validation",
        dataset_id="demo",
    ).as_dict()
    assert payload["track_count"] == 1
    assert payload["duration_hours"]["mean"] == pytest.approx(3.0)
    assert payload["speed_kmh"]["mean"] == pytest.approx(30.0)
    assert payload["direction_distribution"] == {"NE": 1}
    assert payload["status_distribution"] == {"ENDED": 1}


def test_the_diagnostics_never_call_themselves_accuracy() -> None:
    payload = compute_tracking_diagnostics(
        [Track([Point(1.0), Point(2.0)])], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["note"] == GROUND_TRUTH_UNAVAILABLE
    assert payload["ground_truth_available"] is False
    assert "NOT tracking accuracy" in payload["caveat"]
    for key in payload:
        assert "accuracy" not in key


def test_intensity_evolution_is_reported() -> None:
    payload = compute_tracking_diagnostics(
        [
            Track([Point(2.0), Point(5.0)]),  # strengthening
            Track([Point(5.0), Point(2.0)]),  # weakening
        ],
        split="validation",
        dataset_id="demo",
    ).as_dict()
    evolution = payload["intensity_evolution"]
    assert evolution["strengthening"] == 1
    assert evolution["weakening"] == 1
    assert evolution["count"] == 2


def test_unmatched_events_are_counted() -> None:
    payload = compute_tracking_diagnostics(
        [Track([Point(1.0), Point(2.0)])],
        split="validation",
        dataset_id="demo",
        event_count=5,
    ).as_dict()
    assert payload["event_count"] == 5
    assert payload["matched_events"] == 2
    assert payload["unmatched_events"] == 3


def test_single_point_tracks_are_flagged() -> None:
    payload = compute_tracking_diagnostics(
        [Track([Point(1.0)])], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["continuity"]["single_point_tracks"] == 1
    assert payload["continuity"]["multi_point_tracks"] == 0
    assert any("single point" in warning for warning in payload["warnings"])


def test_no_tracks_produces_an_explicit_warning() -> None:
    payload = compute_tracking_diagnostics(
        [], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["track_count"] == 0
    assert payload["duration_hours"] == {"count": 0}
    assert any("No tracks" in warning for warning in payload["warnings"])


def test_ground_truth_presence_does_not_invent_a_skill_score() -> None:
    """Even with a catalogue, this project reports no tracking skill."""
    from extreme_weather.validation.ground_truth import (
        GroundTruth,
        GroundTruthRecord,
    )
    import numpy as np

    truth = GroundTruth(
        records=(
            GroundTruthRecord(
                "G1", np.datetime64("2024-06-01T06:00"), 12.0, 77.0, "EXTREME_RAINFALL"
            ),
        ),
        provenance="IMD verified trajectories",
    )
    payload = compute_tracking_diagnostics(
        [Track([Point(1.0), Point(2.0)])],
        split="validation",
        dataset_id="demo",
        truth=truth,
    ).as_dict()
    assert payload["ground_truth_available"] is True
    # The flag flips, but the refusal stands: there are no reference
    # trajectories to score against.
    assert payload["note"] == GROUND_TRUTH_UNAVAILABLE
    assert "NOT tracking accuracy" in payload["caveat"]


def test_the_report_is_serialisable() -> None:
    payload = compute_tracking_diagnostics(
        [Track([Point(1.0), Point(2.0)])], split="validation", dataset_id="demo"
    ).as_dict()
    for key in (
        "track_count",
        "duration_hours",
        "displacement_km",
        "speed_kmh",
        "direction_distribution",
        "intensity_evolution",
        "continuity",
        "caveat",
    ):
        assert key in payload

