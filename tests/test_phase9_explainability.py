"""Explainability tests (SIH26078, sections 8-10).

Required coverage: event evidence, missing evidence, no fabricated evidence, and
the sensitivity calculation.

The prohibition on causal language is asserted mechanically rather than by
eyeball: every string the module produces is searched for the words that would
turn a correlation into a cause.
"""

from __future__ import annotations

import json

import pytest

from extreme_weather.evaluation.explainability import (
    CAUSALITY_DISCLAIMER,
    CONTEXT,
    EVIDENCE_UNAVAILABLE,
    SUPPORTING,
    build_alert_explanation,
    build_event_explanation,
    sensitivity_report,
)

from tests.phase9_support import make_event, make_samples, make_track

#: Words that would turn a correlation into a causal claim. None of them may
#: appear in generated output.
FORBIDDEN_CAUSAL_WORDS = (
    "caused by",
    "this caused",
    "because of the",
    "responsible for",
    "led to the",
    "triggered the event",
    "root cause",
    "due to the intensity",
)


def _all_strings(payload) -> list[str]:
    """Return every string in a nested JSON-like structure."""
    found: list[str] = []
    if isinstance(payload, str):
        found.append(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            found.extend(_all_strings(key))
            found.extend(_all_strings(value))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found.extend(_all_strings(item))
    return found


# --------------------------------------------------------------------------- #
# Event evidence
# --------------------------------------------------------------------------- #
def test_an_event_yields_its_recorded_evidence(event, track) -> None:
    """Intensity, cell count, radius and speed are all recorded by the pipeline."""
    explanation = build_event_explanation(event, track=track)
    features = {item.feature for item in explanation.available_evidence}

    assert "intensity" in features
    assert "anomaly_score" in features
    assert "affected_cell_count" in features
    assert "affected_radius_km" in features
    assert "track_speed_kmh" in features
    assert "persistence_hours" in features
    assert "track_direction" in features


def test_evidence_carries_units_and_context(event) -> None:
    """A number without a unit or a sentence is not usable evidence."""
    explanation = build_event_explanation(event)
    intensity = next(
        item for item in explanation.evidence if item.feature == "intensity"
    )

    assert intensity.unit == "mm h-1"
    assert "Peak physical value" in intensity.context


def test_track_features_are_labelled_as_context_not_causal_support(event, track) -> None:
    """Direction and trend are recorded facts, not model inputs."""
    explanation = build_event_explanation(event, track=track)
    direction = next(
        item for item in explanation.evidence if item.feature == "track_direction"
    )

    assert direction.contribution_type == CONTEXT
    assert direction.value == "NORTHEAST"


def test_every_explanation_carries_the_causality_disclaimer(event) -> None:
    """The caveat is not optional and cannot be configured away."""
    payload = build_event_explanation(event).as_dict()

    assert payload["disclaimer"] == CAUSALITY_DISCLAIMER
    assert CAUSALITY_DISCLAIMER in payload["limitations"]


def test_no_generated_text_uses_causal_language(event, track) -> None:
    """The single most important guarantee of this module."""
    payload = build_event_explanation(event, track=track).as_dict()
    blob = " ".join(_all_strings(payload)).lower()

    for phrase in FORBIDDEN_CAUSAL_WORDS:
        assert phrase not in blob, f"causal language '{phrase}' leaked into the output"


def test_the_disclaimer_states_the_negative_claim_explicitly() -> None:
    """A reader must be told what the evidence is not."""
    assert "not causal" in CAUSALITY_DISCLAIMER
    assert "does not establish" in CAUSALITY_DISCLAIMER


# --------------------------------------------------------------------------- #
# Missing evidence
# --------------------------------------------------------------------------- #
def test_an_untracked_event_reports_its_missing_evidence(event) -> None:
    """No track means no movement evidence - reported, not zeroed."""
    explanation = build_event_explanation(event, track=None)
    missing = {item.feature for item in explanation.unavailable_evidence}

    assert "track_speed_kmh" in missing
    assert "persistence_hours" in missing
    speed = next(
        item for item in explanation.evidence if item.feature == "track_speed_kmh"
    )
    assert speed.value is None
    assert speed.contribution_type == EVIDENCE_UNAVAILABLE


def test_a_missing_gnn_prediction_is_never_invented(event) -> None:
    """No model means no prediction - not an interpolated one."""
    explanation = build_event_explanation(event, gnn_prediction=None)
    gnn = next(item for item in explanation.evidence if item.feature == "gnn_prediction")

    assert gnn.available is False
    assert gnn.value is None
    assert "No prediction was invented" in gnn.context


def test_an_available_downscaled_field_is_reported(event) -> None:
    """Availability is stated either way."""
    present = build_event_explanation(event, downscaling_available=True)
    absent = build_event_explanation(event, downscaling_available=False)

    assert next(
        item for item in present.evidence if item.feature == "downscaled_impact_field"
    ).available
    assert not next(
        item for item in absent.evidence if item.feature == "downscaled_impact_field"
    ).available


def test_percentile_evidence_is_declared_unavailable_on_purpose(event) -> None:
    """With no climatological reference, a percentile would be fabricated."""
    explanation = build_event_explanation(event)

    assert any("percentile" in text.lower() for text in explanation.limitations)


def test_available_and_unavailable_evidence_are_counted_separately(event, track) -> None:
    """The UI needs both lists, and the counts must agree with them."""
    explanation = build_event_explanation(event, track=track)
    payload = explanation.as_dict()

    assert payload["evidence_counts"]["available"] == len(payload["available_evidence"])
    assert payload["evidence_counts"]["unavailable"] == len(payload["unavailable_evidence"])
    assert (
        payload["evidence_counts"]["available"]
        + payload["evidence_counts"]["unavailable"]
        == len(payload["evidence"])
    )


def test_the_explanation_is_json_serialisable(event, track) -> None:
    """It is returned over HTTP."""
    json.dumps(build_event_explanation(event, track=track).as_dict())


def test_freshness_is_reported_as_context_when_known(event) -> None:
    """Stale data is a caveat on every other item."""
    explanation = build_event_explanation(event, data_freshness="STALE")
    item = next(i for i in explanation.evidence if i.feature == "data_freshness")

    assert item.value == "STALE"
    assert item.contribution_type == CONTEXT
    assert "weakens" in item.context or "caveat" in item.context


# --------------------------------------------------------------------------- #
# Alert explanation
# --------------------------------------------------------------------------- #
def test_an_alert_without_risk_factors_reports_them_as_missing(event) -> None:
    """An unexplained score would be the worst case, so it is flagged."""

    class _BareAlert:
        alert_id = "alert-1"
        event_id = "event-1"
        source_components: list[str] = []

    explanation = build_alert_explanation(_BareAlert(), event=event)
    risk = next(item for item in explanation.evidence if item.feature == "risk_score")

    assert risk.available is False
    assert "invented" in risk.context


def test_alert_explanation_lists_declared_source_components(event) -> None:
    """A degraded run must show exactly which stages contributed."""

    class _Alert:
        alert_id = "alert-2"
        event_id = "event-2"
        source_components = ["events", "tracking"]

    explanation = build_alert_explanation(_Alert(), event=event)
    item = next(i for i in explanation.evidence if i.feature == "source_components")

    assert item.value == "events, tracking"


def test_alert_explanation_disclaims_skill(event) -> None:
    """No verified ground truth exists, so no skill claim may appear."""
    explanation = build_alert_explanation(event, event=event)

    assert any("no verified ground-truth" in text.lower() for text in explanation.limitations)


# --------------------------------------------------------------------------- #
# Feature sensitivity
# --------------------------------------------------------------------------- #
def test_sensitivity_reports_the_prediction_delta() -> None:
    """The measurement itself must be correct."""
    report = sensitivity_report(
        feature_names=["latitude", "intensity"],
        baseline_features=[12.0, 30.0],
        perturbed_features=[{"latitude": 13.0}, {"intensity": 35.0}],
        baseline_prediction=[12.0, 30.0],
        perturbed_predictions=[[12.5, 30.0], [12.0, 30.2]],
    )
    by_feature = {item["feature"]: item for item in report["experiments"]}

    assert by_feature["latitude"]["prediction_delta"] == pytest.approx([0.5, 0.0])
    assert by_feature["intensity"]["prediction_delta"] == pytest.approx([0.0, 0.2])


def test_sensitivity_is_reported_as_sensitivity_not_causality() -> None:
    """A perturbation measures the function, not the world."""
    report = sensitivity_report(
        ["a"], [1.0], [{"a": 2.0}], [1.0], [[2.0]]
    )

    assert report["reported_as"] == "input_sensitivity"
    assert "not evidence that any feature" in report["interpretation"]
    assert report["disclaimer"] == CAUSALITY_DISCLAIMER


def test_sensitivity_contains_no_causal_language() -> None:
    """The prohibition applies to the sensitivity output too."""
    report = sensitivity_report(
        ["a", "b"], [1.0, 2.0], [{"a": 5.0}], [1.0, 2.0], [[9.0, 2.0]]
    )
    blob = " ".join(_all_strings(report)).lower()

    for phrase in FORBIDDEN_CAUSAL_WORDS:
        assert phrase not in blob


def test_sensitivity_ranks_features_by_response() -> None:
    """A reader needs the ordering, not just a bag of deltas."""
    report = sensitivity_report(
        ["small", "large"],
        [0.0, 0.0],
        [{"small": 1.0}, {"large": 1.0}],
        [0.0, 0.0],
        [[0.1, 0.0], [0.0, 5.0]],
    )

    assert report["ranked_by_response"][0] == "large"


def test_perturbing_a_non_input_says_so_instead_of_reporting_zero() -> None:
    """A delta of 0 would read as "this feature does not matter"."""
    report = sensitivity_report(
        ["latitude"], [12.0], [{"not_a_feature": 1.0}], [12.0], [[12.0]]
    )
    entry = report["experiments"][0]

    assert entry["perturbed"] is False
    assert "not a model input" in entry["reason"]
    assert "delta_magnitude" not in entry


def test_sensitivity_rejects_inconsistent_inputs() -> None:
    """Mismatched lists would silently misalign feature to prediction."""
    with pytest.raises(ValueError, match="perturbed inputs"):
        sensitivity_report(["a"], [1.0], [{"a": 2.0}], [1.0], [])

    with pytest.raises(ValueError, match="baseline feature values"):
        sensitivity_report(["a", "b"], [1.0], [{"a": 2.0}], [1.0], [[1.0]])
