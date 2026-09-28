"""Tests for the Phase 5 deterministic risk / priority engine (SIH26078).

The properties under test are the ones the design promises: the score is a
transparent weighted sum, it is reproducible, missing optional stages degrade
to a documented zero instead of crashing, and the explanation never claims more
than the data supports.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extreme_weather.alerts.engine import (
    FACTOR_WEIGHTS,
    RISK_DISCLAIMER,
    RiskConfig,
    RiskContext,
    assess_risk,
    clamp01,
)
from extreme_weather.alerts.severity import AlertSeverity, SeverityThresholds
from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.event import Severity as EventSeverity
from extreme_weather.tracking.metrics import (
    TREND_STABLE,
    TREND_STRENGTHENING,
    TREND_WEAKENING,
)

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_context(**overrides: object) -> RiskContext:
    """Build a baseline context; every field can be overridden per test."""
    payload: dict = {
        "event_id": "E0",
        "event_type": EventType.EXTREME_RAINFALL,
        "intensity": 45.0,
        "event_severity": EventSeverity.SEVERE,
        "affected_radius_km": 90.0,
    }
    payload.update(overrides)
    return RiskContext(**payload)


# --------------------------------------------------------------------------- #
# Determinism and transparency
# --------------------------------------------------------------------------- #
def test_factor_weights_sum_to_one_hundred() -> None:
    """The weighted sum must land directly on the 0-100 severity scale."""
    assert sum(FACTOR_WEIGHTS.values()) == pytest.approx(100.0)


def test_scoring_is_deterministic() -> None:
    context = make_context(
        track_id="TRK-1", duration_hours=4.0, intensity_trend=TREND_STRENGTHENING
    )
    first = assess_risk(context, generated_at=NOW)
    second = assess_risk(context, generated_at=NOW)
    assert first.score == second.score
    assert first.level is second.level
    assert first.explanation == second.explanation
    assert first.model_dump() == second.model_dump()


def test_score_equals_the_sum_of_its_factors() -> None:
    """The score is fully attributable: no hidden term, no rounding gap."""
    assessment = assess_risk(
        make_context(track_id="TRK-1", duration_hours=5.0), generated_at=NOW
    )
    assert assessment.total_points == pytest.approx(assessment.score, abs=0.01)


def test_every_factor_is_reported_with_its_weight_and_detail() -> None:
    assessment = assess_risk(
        make_context(track_id="TRK-1", duration_hours=5.0), generated_at=NOW
    )
    names = [factor.name for factor in assessment.contributing_factors]
    assert set(names) == set(FACTOR_WEIGHTS)
    for factor in assessment.contributing_factors:
        assert factor.weight == FACTOR_WEIGHTS[factor.name]
        assert 0.0 <= factor.normalized_value <= 1.0
        assert factor.detail  # every factor explains itself


def test_explanation_names_the_level_and_lists_reasons() -> None:
    assessment = assess_risk(make_context(), generated_at=NOW)
    assert assessment.level.value in assessment.explanation
    assert "because" in assessment.explanation
    lines = assessment.reason_lines()
    assert len(lines) >= 2
    assert any(line.startswith("-") for line in lines)


def test_risk_disclaimer_rejects_disaster_risk_claims() -> None:
    lowered = RISK_DISCLAIMER.lower()
    assert "not a validated disaster-risk index" in lowered
    assert "not a probability" in lowered


def test_clamp01_bounds_and_rejects_non_finite() -> None:
    assert clamp01(-1.0) == 0.0
    assert clamp01(2.0) == 1.0
    assert clamp01(0.5) == 0.5
    assert clamp01(float("nan")) == 0.0
    assert clamp01(float("inf")) == 0.0
    assert clamp01("not a number") == 0.0


# --------------------------------------------------------------------------- #
# Factor behaviour
# --------------------------------------------------------------------------- #
def test_higher_severity_scores_higher() -> None:
    low = assess_risk(make_context(event_severity=EventSeverity.WATCH), generated_at=NOW)
    high = assess_risk(make_context(event_severity=EventSeverity.EXTREME), generated_at=NOW)
    assert high.score > low.score


def test_strengthening_scores_above_weakening() -> None:
    common = dict(track_id="TRK-1", duration_hours=4.0, peak_intensity=40.0)
    strengthening = assess_risk(
        make_context(intensity_trend=TREND_STRENGTHENING, **common), generated_at=NOW
    )
    weakening = assess_risk(
        make_context(intensity_trend=TREND_WEAKENING, **common), generated_at=NOW
    )
    assert strengthening.score > weakening.score


def test_a_larger_affected_radius_scores_higher() -> None:
    small = assess_risk(make_context(affected_radius_km=20.0), generated_at=NOW)
    large = assess_risk(make_context(affected_radius_km=140.0), generated_at=NOW)
    assert large.score > small.score


def test_persistence_only_counts_when_tracked() -> None:
    untracked = assess_risk(make_context(), generated_at=NOW)
    tracked = assess_risk(
        make_context(track_id="TRK-1", duration_hours=10.0), generated_at=NOW
    )
    assert tracked.score > untracked.score
    factor = next(f for f in untracked.contributing_factors if f.name == "persistence")
    assert factor.points == 0.0
    assert "not tracked" in factor.detail


def test_prediction_factors_add_points_when_a_prediction_exists() -> None:
    without = assess_risk(make_context(track_id="TRK-1"), generated_at=NOW)
    with_prediction = assess_risk(
        make_context(track_id="TRK-1", predicted_intensity=70.0, predicted_speed_kmh=30.0),
        generated_at=NOW,
    )
    assert with_prediction.score > without.score
    names = {factor.name for factor in without.contributing_factors}
    assert "predicted_intensity_change" in names
    assert "predicted_movement" in names


def test_downscaled_threat_only_counts_when_supplied() -> None:
    without = assess_risk(make_context(), generated_at=NOW)
    with_threat = assess_risk(
        make_context(downscaled_threat=45.0, downscaled_resolution_km=5.0), generated_at=NOW
    )
    assert with_threat.score > without.score


# --------------------------------------------------------------------------- #
# Graceful degradation
# --------------------------------------------------------------------------- #
def test_missing_prediction_and_threat_score_zero_without_raising() -> None:
    """A missing GNN / downscaling model must not raise or inflate the score."""
    assessment = assess_risk(make_context(), generated_at=NOW)
    by_name = {factor.name: factor for factor in assessment.contributing_factors}
    for name in (
        "predicted_intensity_change",
        "predicted_movement",
        "downscaled_threat",
    ):
        assert by_name[name].points == 0.0
        assert by_name[name].detail


def test_degraded_run_says_so_in_the_explanation() -> None:
    assessment = assess_risk(make_context(), generated_at=NOW)
    text = assessment.explanation.lower()
    assert "no gnn prediction available" in text
    assert "no ai-downscaled threat field" in text


def test_untracked_event_does_not_invent_a_trend() -> None:
    """Without track history the trend must be neutral, not inferred from zero."""
    assessment = assess_risk(make_context(), generated_at=NOW)
    factor = next(
        f for f in assessment.contributing_factors if f.name == "intensity_trend"
    )
    assert "no track history" in factor.detail
    assert factor.normalized_value == pytest.approx(0.4)


def test_trend_is_inferred_when_history_exists() -> None:
    assessment = assess_risk(
        make_context(
            track_id="TRK-1", duration_hours=3.0, peak_intensity=20.0, intensity=60.0
        ),
        generated_at=NOW,
    )
    factor = next(
        f for f in assessment.contributing_factors if f.name == "intensity_trend"
    )
    assert "inferred strengthening" in factor.detail


def test_unknown_trend_label_is_treated_as_stable() -> None:
    assessment = assess_risk(
        make_context(track_id="TRK-1", intensity_trend="EXPLODING"), generated_at=NOW
    )
    factor = next(
        f for f in assessment.contributing_factors if f.name == "intensity_trend"
    )
    assert "unrecognised trend" in factor.detail
    assert factor.normalized_value == pytest.approx(0.4)


def test_stable_track_trend_is_neutral_not_extreme() -> None:
    assessment = assess_risk(
        make_context(track_id="TRK-1", duration_hours=3.0, intensity_trend=TREND_STABLE),
        generated_at=NOW,
    )
    factor = next(
        f for f in assessment.contributing_factors if f.name == "intensity_trend"
    )
    assert factor.normalized_value == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_custom_config_changes_the_score() -> None:
    context = make_context(intensity=45.0)
    default = assess_risk(context, RiskConfig(), generated_at=NOW)
    # Lower the rainfall reference from 50 to 10 so the intensity factor saturates.
    tightened = RiskConfig(
        intensity_reference={EventType.EXTREME_RAINFALL: 10.0},
    )
    raised = assess_risk(context, tightened, generated_at=NOW)
    assert raised.score > default.score


def test_intensity_reference_fallback_applies_to_unmapped_types() -> None:
    """An event type absent from the reference map falls back to the fallback."""
    context = make_context(intensity=45.0, event_type=EventType.EXTREME_WIND)
    empty_map = {EventType.EXTREME_RAINFALL: 10.0}
    low_reference = assess_risk(
        context,
        RiskConfig(intensity_reference=empty_map, intensity_reference_fallback=5.0),
        generated_at=NOW,
    )
    high_reference = assess_risk(
        context,
        RiskConfig(intensity_reference=empty_map, intensity_reference_fallback=500.0),
        generated_at=NOW,
    )
    assert low_reference.score > high_reference.score


def test_custom_severity_thresholds_change_the_level() -> None:
    config = RiskConfig(
        severity_thresholds=SeverityThresholds(
            warning=1.0, severe_warning=2.0, extreme=3.0
        )
    )
    assessment = assess_risk(make_context(), config, generated_at=NOW)
    assert assessment.level is AlertSeverity.EXTREME
    assert assessment.severity_thresholds["extreme"] == 3.0


def test_risk_config_rejects_non_positive_references() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RiskConfig(radius_reference_km=0.0)


def test_risk_config_is_json_serialisable() -> None:
    payload = RiskConfig().as_dict()
    assert payload["factor_weights"] == FACTOR_WEIGHTS
    assert payload["severity_thresholds"]["warning"] == 25.0


def test_intensity_reference_is_per_event_type() -> None:
    config = RiskConfig()
    assert config.intensity_reference_for(EventType.EXTREME_WIND) == 15.0
    assert config.intensity_reference_for("EXTREME_TEMPERATURE") == 8.0
    assert config.intensity_reference_for(None) == config.intensity_reference_fallback


def test_missing_observed_values_are_rejected() -> None:
    broken = RiskContext(
        event_id="E0",
        event_type=EventType.EXTREME_RAINFALL,
        intensity=45.0,
        event_severity=None,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="observed intensity and event severity"):
        assess_risk(broken, generated_at=NOW)

