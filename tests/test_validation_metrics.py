"""Tests for Phase 7 evaluation metrics, baselines and honest refusals.

The central behaviour is the refusal: without a reference catalogue, POD / FAR /
CSI must be **absent** and explicitly explained, never zero. A zero would read
as "the method has no skill", which is a completely different claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.validation.ground_truth import (
    GROUND_TRUTH_UNAVAILABLE,
    GroundTruth,
    GroundTruthRecord,
    GroundTruthStatus,
    haversine_km,
    match_events,
)
from extreme_weather.validation.metrics import (
    NO_GROUND_TRUTH,
    compare_metrics,
    compute_alert_diagnostics,
    compute_anomaly_diagnostics,
    compute_downscaling_comparison,
    compute_gnn_comparison,
    compute_tracking_diagnostics,
    evaluate_detection_skill,
    persistence_baseline,
    summarise,
)


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def test_an_empty_catalogue_is_unavailable() -> None:
    truth = GroundTruth()
    assert truth.available is False
    assert truth.status == GroundTruthStatus.UNAVAILABLE
    assert truth.as_dict()["note"] == GROUND_TRUTH_UNAVAILABLE


def test_a_catalogue_without_provenance_is_rejected() -> None:
    """A file of timestamps with no source cannot be treated as truth."""
    record = GroundTruthRecord("G1", np.datetime64("2024-06-01"), 12.0, 77.0, "EXTREME_RAINFALL")
    truth = GroundTruth(records=(record,), provenance="")
    assert truth.available is False
    assert truth.status == GroundTruthStatus.UNKNOWN_PROVENANCE


def test_a_catalogue_with_provenance_is_usable() -> None:
    record = GroundTruthRecord("G1", np.datetime64("2024-06-01"), 12.0, 77.0, "EXTREME_RAINFALL")
    truth = GroundTruth(records=(record,), provenance="IMD verified event list")
    assert truth.available is True
    assert truth.status == GroundTruthStatus.AVAILABLE
    assert truth.matching_criteria["distance_formula"] == "haversine"


def test_haversine_is_zero_for_the_same_point() -> None:
    assert haversine_km(12.0, 77.0, 12.0, 77.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_measures_a_known_distance() -> None:
    # One degree of latitude is ~111 km everywhere.
    assert haversine_km(12.0, 77.0, 13.0, 77.0) == pytest.approx(111.32, abs=0.1)


# --------------------------------------------------------------------------- #
# Matching and skill
# --------------------------------------------------------------------------- #
def _truth() -> GroundTruth:
    return GroundTruth(
        records=(
            GroundTruthRecord("G1", np.datetime64("2024-06-01T06:00"), 12.0, 77.0, "EXTREME_RAINFALL"),
            GroundTruthRecord("G2", np.datetime64("2024-06-01T12:00"), 12.5, 77.5, "EXTREME_RAINFALL"),
        ),
        provenance="IMD verified event list",
    )


def test_matching_counts_the_contingency_table() -> None:
    detected = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0, "event_type": "EXTREME_RAINFALL"},
        {"event_id": "D2", "time": "2024-06-01T12:00", "latitude": 12.5, "longitude": 77.5, "event_type": "EXTREME_RAINFALL"},
        {"event_id": "D3", "time": "2024-06-01T18:00", "latitude": 12.1, "longitude": 77.1, "event_type": "EXTREME_RAINFALL"},
    ]
    counts = match_events(detected, _truth())
    assert counts["true_positives"] == 2
    assert counts["false_positives"] == 1
    assert counts["false_negatives"] == 0


def test_a_missed_event_counts_as_a_false_negative() -> None:
    detected = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0, "event_type": "EXTREME_RAINFALL"}
    ]
    counts = match_events(detected, _truth())
    assert counts["false_negatives"] == 1


def test_an_event_far_outside_the_tolerance_does_not_match() -> None:
    detected = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0, "event_type": "EXTREME_RAINFALL"},
        {"event_id": "D2", "time": "2024-06-01T12:00", "latitude": 20.0, "longitude": 85.0, "event_type": "EXTREME_RAINFALL"},
    ]
    counts = match_events(detected, _truth())
    assert counts["true_positives"] == 1
    assert counts["false_negatives"] == 1


def test_a_different_event_type_does_not_match() -> None:
    detected = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0, "event_type": "EXTREME_WIND"}
    ]
    counts = match_events(detected, _truth())
    assert counts["true_positives"] == 0
    assert counts["false_positives"] == 1


# --------------------------------------------------------------------------- #
# The refusal - the most important tests in this file
# --------------------------------------------------------------------------- #
def test_skill_is_not_computed_without_ground_truth() -> None:
    detected = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0, "event_type": "EXTREME_RAINFALL"}
    ]
    skill = evaluate_detection_skill(detected, GroundTruth())
    assert skill.available is False
    assert skill.pod is None
    assert skill.far is None
    assert skill.csi is None
    assert skill.note == NO_GROUND_TRUTH


def test_an_empty_reference_does_not_produce_a_zero_skill() -> None:
    """A zero would read as "no skill"; the truth is "cannot be evaluated"."""
    truth = GroundTruth(records=(), provenance="catalogue with no rows")
    skill = evaluate_detection_skill([], truth)
    assert skill.available is False
    assert skill.pod is None


def test_skill_is_computed_against_real_ground_truth() -> None:
    detected = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0, "event_type": "EXTREME_RAINFALL"},
        {"event_id": "D2", "time": "2024-06-01T12:00", "latitude": 12.5, "longitude": 77.5, "event_type": "EXTREME_RAINFALL"},
    ]
    skill = evaluate_detection_skill(detected, _truth())
    assert skill.available is True
    assert skill.pod == pytest.approx(1.0)
    assert skill.far == pytest.approx(0.0)
    assert skill.csi == pytest.approx(1.0)
    assert skill.bias == pytest.approx(1.0)


def test_skill_serialises_its_criteria() -> None:
    payload = evaluate_detection_skill([], _truth()).as_dict()
    assert payload["matching_criteria"]["distance_formula"] == "haversine"
    assert payload["matching_criteria"]["time_tolerance_hours"] == 6.0


# --------------------------------------------------------------------------- #
# Baseline comparison
# --------------------------------------------------------------------------- #
def test_a_lower_error_is_an_improvement() -> None:
    result = compare_metrics(1.0, 2.0)
    assert result["verdict"] == "improves"
    assert result["improves_on_baseline"] is True
    assert result["relative_change"] == pytest.approx(-0.5)


def test_a_higher_error_is_a_loss_and_is_reported_as_such() -> None:
    result = compare_metrics(3.0, 2.0)
    assert result["verdict"] == "worse"
    assert result["improves_on_baseline"] is False


def test_equal_errors_are_a_tie() -> None:
    assert compare_metrics(2.0, 2.0)["verdict"] == "tie"


def test_a_missing_value_is_not_compared() -> None:
    result = compare_metrics(None, 2.0)
    assert result["verdict"] == "not_computed"
    assert result["reason"] == NO_GROUND_TRUTH


def test_higher_is_better_can_be_requested() -> None:
    assert compare_metrics(0.9, 0.8, lower_is_better=False)["verdict"] == "improves"


# --------------------------------------------------------------------------- #
# Persistence baseline
# --------------------------------------------------------------------------- #
def test_the_persistence_baseline_scores_a_perfect_forecast_as_zero_error() -> None:
    current = np.array([12.0, 12.5])
    future = np.array([12.0, 12.5])
    result = persistence_baseline(current, current, current * 0, future, future, future * 0)
    assert result["latitude_mae_deg"] == pytest.approx(0.0)
    assert result["displacement_error_km_mean"] == pytest.approx(0.0, abs=1e-9)


def test_the_persistence_baseline_measures_real_movement() -> None:
    """Persistence predicts the current position, so the error is the movement."""
    current_lat = np.array([12.0])
    current_lon = np.array([77.0])
    zero = np.array([0.0])
    future_lat = np.array([13.0])
    future_lon = np.array([77.0])
    result = persistence_baseline(
        current_lat, current_lon, zero, future_lat, future_lon, zero
    )
    assert result["latitude_mae_deg"] == pytest.approx(1.0)
    assert result["displacement_error_km_mean"] == pytest.approx(111.32, abs=0.1)


def test_the_persistence_baseline_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        persistence_baseline(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))


def test_the_persistence_baseline_rejects_an_empty_input() -> None:
    empty = np.zeros(0)
    with pytest.raises(ValueError, match="at least one"):
        persistence_baseline(empty, empty, empty, empty, empty, empty)


def test_the_gnn_comparison_reports_a_loss_against_persistence() -> None:
    """The honest case: a model worse than 'nothing moves'."""
    zeros = np.zeros(3)
    result = compute_gnn_comparison(
        predicted_lat=np.array([0.5, 0.5, 0.5]),
        predicted_lon=zeros,
        predicted_intensity=zeros,
        current_lat=zeros,
        current_lon=zeros,
        current_intensity=zeros,
        future_lat=zeros,
        future_lon=zeros,
        future_intensity=zeros,
        dataset="demo",
        split="test",
        model_checkpoint="ckpt",
    )
    assert result.comparison["latitude_mae_deg"]["verdict"] == "worse"
    assert result.baseline_metrics["latitude_mae_deg"] == pytest.approx(0.0)
    assert result.comparison["_summary"]["metrics_worsened"] >= 1


def test_the_gnn_comparison_reports_a_win_against_persistence() -> None:
    zeros = np.zeros(3)
    truth = np.array([0.5, 0.5, 0.5])
    result = compute_gnn_comparison(
        predicted_lat=truth,
        predicted_lon=zeros,
        predicted_intensity=zeros,
        current_lat=zeros,
        current_lon=zeros,
        current_intensity=zeros,
        future_lat=truth,
        future_lon=zeros,
        future_intensity=zeros,
        dataset="demo",
        split="test",
        model_checkpoint="ckpt",
    )
    assert result.comparison["latitude_mae_deg"]["verdict"] == "improves"
    assert result.comparison["_summary"]["verdict"] == "beats persistence"


def test_the_gnn_comparison_rejects_an_empty_test_split() -> None:
    zeros = np.zeros(0)
    with pytest.raises(ValueError, match="at least one test sample"):
        compute_gnn_comparison(
            zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
            dataset="d", split="test", model_checkpoint=None,
        )


def test_the_gnn_comparison_records_its_limitations() -> None:
    zeros = np.zeros(2)
    result = compute_gnn_comparison(
        zeros, zeros, zeros, zeros, zeros, zeros,
        np.array([0.1, 0.2]), zeros, zeros,
        dataset="d", split="test", model_checkpoint="c",
        limitations=["a stated limitation"],
    )
    assert result.limitations == ("a stated limitation",)
    assert "a stated limitation" in result.as_dict()["limitations"]


# --------------------------------------------------------------------------- #
# Downscaling comparison
# --------------------------------------------------------------------------- #
def _frames() -> tuple[np.ndarray, np.ndarray]:
    reference = np.linspace(1.0, 10.0, 100).reshape(1, 10, 10)
    return reference, reference


def test_downscaling_compares_every_method_on_identical_frames() -> None:
    reference, _ = _frames()
    perfect = reference.copy()
    result = compute_downscaling_comparison(
        {"ai": perfect, "nearest": perfect * 0.9, "cubic": perfect * 0.8},
        reference,
        dataset="demo",
        split="test",
    )
    # Every method is scored on the *same* frames, so only the deliberately
    # wrong fields have non-zero error.
    assert result.methods["ai"]["mae"] == pytest.approx(0.0)
    assert result.methods["nearest"]["mae"] > 0.0
    assert result.methods["cubic"]["mae"] > result.methods["nearest"]["mae"]
    assert result.frame_count == reference.size
    for name in ("ai", "nearest", "cubic"):
        assert result.methods[name]["shape"] == list(reference.shape)


def test_downscaling_reports_a_loss_against_cubic() -> None:
    """A worse-than-cubic result must be stated, not averaged away."""
    reference, _ = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference + 5.0, "cubic": reference.copy()},
        reference,
        dataset="demo",
        split="test",
    )
    assert result.methods["cubic"]["mae"] == pytest.approx(0.0)
    assert result.methods["ai"]["mae"] > 0
    assert result.comparison["ai_vs_cubic"]["mae"]["verdict"] == "worse"
    assert "does not beat" in result.comparison["_summary"]["verdict"]


def test_downscaling_pod_is_not_computed_without_a_threshold() -> None:
    reference, _ = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference, "nearest": reference}, reference, dataset="d", split="test"
    )
    assert result.classification["available"] is False
    assert "not computed" in str(result.classification["note"])
    assert "extremes" not in result.methods["ai"]


def test_downscaling_computes_pod_when_a_threshold_is_given() -> None:
    reference, _ = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference, "nearest": reference},
        reference,
        dataset="d",
        split="test",
        event_threshold=5.0,
    )
    assert result.classification["available"] is True
    assert result.methods["ai"]["extremes"]["POD"] == pytest.approx(1.0)


def test_downscaling_rejects_a_shape_mismatch() -> None:
    reference, _ = _frames()
    with pytest.raises(ValueError, match="identical frames"):
        compute_downscaling_comparison(
            {"ai": np.zeros((1, 5, 5))}, reference, dataset="d", split="test"
        )


def test_downscaling_rejects_no_predictions() -> None:
    reference, _ = _frames()
    with pytest.raises(ValueError, match="at least one prediction"):
        compute_downscaling_comparison({}, reference, dataset="d", split="test")


def test_downscaling_names_the_winner_by_mae() -> None:
    """A biased AI field must not be named the winner over an exact baseline."""
    reference, _ = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference + 1.0, "cubic": reference.copy()}, reference, dataset="d", split="test"
    )
    assert result.winner == "cubic"
    assert result.methods["ai"]["mae"] > result.methods["cubic"]["mae"]


def test_downscaling_reports_peak_preservation_and_structure() -> None:
    reference, _ = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference}, reference, dataset="d", split="test"
    )
    assert result.peak_preservation["per_method"]["ai"]["peak_retention"] == pytest.approx(1.0)
    assert result.structural["per_method"]["ai"]["structural_correlation"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Anomaly, tracking and alert diagnostics
# --------------------------------------------------------------------------- #
def test_anomaly_diagnostics_describe_without_claiming_skill() -> None:
    events = [
        {"event_id": "D1", "time": "2024-06-01T06:00", "latitude": 12.0, "longitude": 77.0,
         "event_type": "EXTREME_RAINFALL", "intensity": 12.0, "variable": "precipitation"},
    ]
    payload = compute_anomaly_diagnostics(
        events, split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["event_count"] == 1
    assert payload["detection_skill"]["available"] is False
    assert payload["detection_skill"]["POD"] is None
    assert "not accuracy" in payload["caveat"]


def test_anomaly_diagnostics_accept_model_objects() -> None:
    class FakeEvent:
        event_id = "D1"
        timestamp = np.datetime64("2024-06-01T06:00")
        latitude = 12.0
        longitude = 77.0
        event_type = "EXTREME_RAINFALL"
        intensity = 12.0
        variable = "precipitation"

    payload = compute_anomaly_diagnostics(
        [FakeEvent()], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["event_count"] == 1
    assert payload["peak_intensity"]["EXTREME_RAINFALL"] == pytest.approx(12.0)


def test_tracking_diagnostics_do_not_claim_accuracy() -> None:
    class Point:
        def __init__(self, intensity):
            self.intensity = intensity

    class Track:
        duration_hours = 2.0
        displacement_km = 30.0
        max_speed_kmh = 25.0
        direction = "NE"
        status = "ENDED"
        points = [Point(5.0), Point(7.0)]

    payload = compute_tracking_diagnostics(
        [Track()], split="validation", dataset_id="demo", event_count=2
    ).as_dict()
    assert payload["track_count"] == 1
    assert payload["matched_events"] == 2
    assert payload["unmatched_events"] == 0
    assert payload["note"] == GROUND_TRUTH_UNAVAILABLE
    assert "NOT tracking accuracy" in payload["caveat"]
    assert payload["intensity_evolution"]["strengthening"] == 1


def test_tracking_diagnostics_warn_about_single_point_tracks() -> None:
    class Track:
        duration_hours = 0.0
        displacement_km = 0.0
        max_speed_kmh = 0.0
        direction = "CALM"
        status = "NEW"
        points = []

    payload = compute_tracking_diagnostics(
        [Track()], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["continuity"]["single_point_tracks"] == 1
    assert any("single point" in warning for warning in payload["warnings"])


def test_tracking_diagnostics_handle_no_tracks() -> None:
    payload = compute_tracking_diagnostics(
        [], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["track_count"] == 0
    assert payload["duration_hours"]["count"] == 0
    assert any("No tracks" in warning for warning in payload["warnings"])


def test_alert_diagnostics_report_the_severity_mix() -> None:
    class Risk:
        score = 42.0
        factors = []

    class Alert:
        severity = "WATCH"
        status = "NEW"
        risk = Risk()
        issued_at = np.datetime64("2024-06-01T06:00")
        valid_until = np.datetime64("2024-06-01T12:00")
        source_components = ["events"]
        alert_id = "A1"
        latitude = 12.0
        longitude = 77.0
        event_type = "EXTREME_RAINFALL"

    payload = compute_alert_diagnostics(
        [Alert()], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["alert_count"] == 1
    assert payload["severity_distribution"] == {"WATCH": 1}
    assert payload["trigger_components"] == {"events": 1}
    assert payload["duration_hours"]["mean"] == pytest.approx(6.0)
    assert payload["detection_skill"]["available"] is False
    assert "NOT official IMD or WMO" in payload["caveat"]


def test_alert_diagnostics_warn_when_nothing_was_produced() -> None:
    payload = compute_alert_diagnostics(
        [], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["alert_count"] == 0
    assert any("No alerts" in warning for warning in payload["warnings"])


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def test_a_summary_of_nothing_is_none() -> None:
    assert summarise([]) is None


def test_a_summary_reports_std_even_when_zero() -> None:
    """'Every value was the same' is a result, not a missing field."""
    stats = summarise([2.0, 2.0, 2.0])
    assert stats["count"] == 3
    assert stats["std"] == pytest.approx(0.0)

