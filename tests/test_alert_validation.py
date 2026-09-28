"""Tests for Phase 7 alert validation.

The severities are this project's own categories, and the report must never
imply otherwise. These tests pin that down, and cover the descriptive statistics
that *can* be reported without ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.validation.ground_truth import (
    GROUND_TRUTH_UNAVAILABLE,
    GroundTruth,
    GroundTruthRecord,
)
from extreme_weather.validation.metrics import compute_alert_diagnostics


class Factor:
    def __init__(self, name: str, points: float) -> None:
        self.name = name
        self.points = points


class Risk:
    def __init__(self, score: float = 55.0, factors: list[Factor] | None = None) -> None:
        self.score = score
        self.factors = factors or [Factor("intensity", 30.0), Factor("proximity", 25.0)]


class Alert:
    def __init__(
        self,
        alert_id: str,
        severity: str,
        components: list[str],
        score: float = 55.0,
    ) -> None:
        self.alert_id = alert_id
        self.severity = severity
        self.status = "NEW"
        self.risk = Risk(score)
        self.issued_at = np.datetime64("2024-06-01T06:00")
        self.valid_until = np.datetime64("2024-06-01T12:00")
        self.source_components = components
        self.latitude = 12.0
        self.longitude = 77.0
        self.event_type = "EXTREME_RAINFALL"


def test_alert_counts_and_severities_are_reported() -> None:
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"]), Alert("A2", "WARNING", ["events", "gnn"])],
        split="validation",
        dataset_id="demo",
    ).as_dict()
    assert payload["alert_count"] == 2
    assert payload["severity_distribution"] == {"WATCH": 1, "WARNING": 1}


def test_trigger_components_are_counted() -> None:
    payload = compute_alert_diagnostics(
        [
            Alert("A1", "WATCH", ["events"]),
            Alert("A2", "WARNING", ["events", "gnn"]),
            Alert("A3", "WATCH", ["events", "tracking"]),
        ],
        split="validation",
        dataset_id="demo",
    ).as_dict()
    assert payload["trigger_components"] == {"events": 3, "gnn": 1, "tracking": 1}


def test_risk_scores_are_summarised() -> None:
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"], score=40.0), Alert("A2", "WARNING", ["events"], score=80.0)],
        split="validation",
        dataset_id="demo",
    ).as_dict()
    assert payload["risk_score"]["count"] == 2
    assert payload["risk_score"]["mean"] == pytest.approx(60.0)
    assert payload["risk_score"]["min"] == pytest.approx(40.0)


def test_alert_durations_are_reported() -> None:
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"])], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["duration_hours"]["mean"] == pytest.approx(6.0)


def test_the_severity_system_is_not_called_validated() -> None:
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"])], split="validation", dataset_id="demo"
    ).as_dict()
    assert "NOT official IMD or WMO" in payload["caveat"]
    assert "not a validated disaster-risk index" in payload["caveat"]
    assert "not a probability of occurrence" in payload["caveat"]


def test_skill_is_refused_without_ground_truth() -> None:
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"])], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["ground_truth_available"] is False
    assert payload["detection_skill"]["available"] is False
    assert payload["detection_skill"]["POD"] is None
    assert payload["detection_skill"]["note"] == GROUND_TRUTH_UNAVAILABLE


def test_skill_is_computed_when_a_catalogue_exists() -> None:
    truth = GroundTruth(
        records=(
            GroundTruthRecord(
                "G1", np.datetime64("2024-06-01T06:00"), 12.0, 77.0, "EXTREME_RAINFALL"
            ),
        ),
        provenance="IMD verified event list",
    )
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"])],
        split="validation",
        dataset_id="demo",
        truth=truth,
    ).as_dict()
    assert payload["detection_skill"]["available"] is True
    assert payload["detection_skill"]["true_positives"] == 1
    assert any("does not mean" in warning for warning in payload["warnings"])


def test_no_alerts_produces_a_warning_and_no_skill() -> None:
    payload = compute_alert_diagnostics(
        [], split="validation", dataset_id="demo"
    ).as_dict()
    assert payload["alert_count"] == 0
    assert payload["risk_score"] == {"count": 0}
    assert payload["detection_skill"]["available"] is False
    assert any("No alerts" in warning for warning in payload["warnings"])


def test_risk_factors_contribute_to_the_summary() -> None:
    payload = compute_alert_diagnostics(
        [Alert("A1", "WATCH", ["events"])], split="validation", dataset_id="demo"
    ).as_dict()
    contribution = payload["component_contribution"]
    assert "intensity" in contribution
    assert contribution["intensity"]["count"] == 1
    assert contribution["intensity"]["points_total"] == pytest.approx(30.0)

