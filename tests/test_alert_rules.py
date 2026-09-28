"""Tests for the Phase 5 alert rule engine (SIH26078).

Each rule is tested in isolation against a synthetic :class:`RuleContext`, so a
failure names the rule that broke rather than "generation is wrong".
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extreme_weather.alerts.models import AlertStatus
from extreme_weather.alerts.rules import (
    ALERT_RULES,
    RuleConfig,
    RuleContext,
    RuleID,
    active_rules,
    apply_rules,
    rule_duplicate_suppressed,
    rule_extreme_anomaly_creates_alert,
    rule_missing_event_expires,
    rule_persistent_track_maintains,
    rule_predicted_increase_raises_level,
    rule_severe_strengthening_elevates,
    rule_weakening_downgrades_or_resolves,
)
from extreme_weather.alerts.severity import AlertSeverity
from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.event import Severity as EventSeverity
from extreme_weather.tracking.metrics import (
    TREND_STABLE,
    TREND_STRENGTHENING,
    TREND_WEAKENING,
)

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 6, 0, tzinfo=UTC)


def make_context(**overrides: object) -> RuleContext:
    """Build a baseline rule context; every field can be overridden."""
    payload: dict = {
        "event_id": "E0",
        "event_type": EventType.EXTREME_RAINFALL,
        "event_severity": EventSeverity.SEVERE,
        "intensity": 45.0,
        "observed_at": T0,
        "risk_level": AlertSeverity.WARNING,
        "risk_score": 40.0,
    }
    payload.update(overrides)
    return RuleContext(**payload)


def outcome_for(rule: object, context: RuleContext, config: RuleConfig | None = None):
    """Evaluate one rule and return its outcome."""
    return rule(context, config or RuleConfig())  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_all_seven_rules_are_registered_in_order() -> None:
    assert [rule.rule_id for rule in ALERT_RULES] == [
        RuleID.TRIGGER,
        RuleID.ELEVATE_PRIORITY,
        RuleID.MAINTAIN_ACTIVE,
        RuleID.RAISE_LEVEL,
        RuleID.RESOLVE,
        RuleID.EXPIRE,
        RuleID.NO_DUPLICATE,
    ]


def test_every_rule_is_documented() -> None:
    for rule in ALERT_RULES:
        assert rule.name and rule.description
        assert callable(rule.apply)


def test_apply_rules_returns_one_outcome_per_enabled_rule() -> None:
    outcomes = apply_rules(make_context())
    assert len(outcomes) == len(ALERT_RULES)
    assert [outcome.rule_id for outcome in outcomes] == [r.rule_id for r in ALERT_RULES]


def test_disabled_rules_are_skipped() -> None:
    from dataclasses import replace

    subset = [replace(ALERT_RULES[0], enabled=False), ALERT_RULES[0]]
    assert len(active_rules(subset)) == 1
    assert len(apply_rules(make_context(), rules=subset)) == 1


# --------------------------------------------------------------------------- #
# R1 - extreme anomaly creates an alert
# --------------------------------------------------------------------------- #
def test_r1_creates_an_alert_at_or_above_the_trigger_severity() -> None:
    outcome = outcome_for(rule_extreme_anomaly_creates_alert, make_context())
    assert outcome.rule_id == RuleID.TRIGGER
    assert outcome.triggered is True
    assert outcome.suppress is False


def test_r1_suppresses_below_the_trigger_severity() -> None:
    context = make_context(event_severity=EventSeverity.NORMAL)
    outcome = outcome_for(rule_extreme_anomaly_creates_alert, context)
    assert outcome.triggered is False
    assert outcome.suppress is True
    assert "below the trigger level" in outcome.reason


def test_r1_trigger_level_is_configurable() -> None:
    context = make_context(event_severity=EventSeverity.WATCH)
    strict = outcome_for(
        rule_extreme_anomaly_creates_alert,
        context,
        RuleConfig(min_trigger_severity=EventSeverity.SEVERE),
    )
    assert strict.suppress is True


# --------------------------------------------------------------------------- #
# R2 - severe + strengthening elevates the priority
# --------------------------------------------------------------------------- #
def test_r2_elevates_priority_for_a_strengthening_severe_event() -> None:
    context = make_context(intensity_trend=TREND_STRENGTHENING)
    outcome = outcome_for(rule_severe_strengthening_elevates, context)
    assert outcome.triggered is True
    assert outcome.priority is not None
    assert outcome.priority.rank > context.risk_level.rank


def test_r2_does_not_fire_without_a_strengthening_trend() -> None:
    outcome = outcome_for(
        rule_severe_strengthening_elevates, make_context(intensity_trend=TREND_STABLE)
    )
    assert outcome.triggered is False


def test_r2_does_not_fire_for_a_mild_severity() -> None:
    context = make_context(
        event_severity=EventSeverity.WATCH, intensity_trend=TREND_STRENGTHENING
    )
    outcome = outcome_for(rule_severe_strengthening_elevates, context)
    assert outcome.triggered is False


def test_r2_never_lowers_the_priority() -> None:
    context = make_context(
        risk_level=AlertSeverity.EXTREME, intensity_trend=TREND_STRENGTHENING
    )
    outcome = outcome_for(rule_severe_strengthening_elevates, context)
    assert outcome.triggered is True
    assert outcome.priority is AlertSeverity.EXTREME


# --------------------------------------------------------------------------- #
# R3 - a persistent tracked event stays ACTIVE
# --------------------------------------------------------------------------- #
def test_r3_keeps_a_long_tracked_event_active() -> None:
    context = make_context(track_id="TRK-1", duration_hours=5.0)
    outcome = outcome_for(rule_persistent_track_maintains, context)
    assert outcome.triggered is True
    assert outcome.status is AlertStatus.ACTIVE


def test_r3_preserves_an_acknowledged_status() -> None:
    context = make_context(
        track_id="TRK-1", duration_hours=5.0, existing_status=AlertStatus.ACKNOWLEDGED
    )
    outcome = outcome_for(rule_persistent_track_maintains, context)
    assert outcome.status is AlertStatus.ACKNOWLEDGED


def test_r3_does_not_fire_for_a_short_track() -> None:
    outcome = outcome_for(
        rule_persistent_track_maintains, make_context(track_id="TRK-1", duration_hours=1.0)
    )
    assert outcome.triggered is False


def test_r3_does_not_fire_for_an_untracked_event() -> None:
    outcome = outcome_for(
        rule_persistent_track_maintains, make_context(duration_hours=9.0)
    )
    assert outcome.triggered is False


# --------------------------------------------------------------------------- #
# R4 - a rising predicted intensity raises the level
# --------------------------------------------------------------------------- #
def test_r4_raises_the_level_when_the_prediction_grows() -> None:
    context = make_context(predicted_intensity=60.0)
    outcome = outcome_for(rule_predicted_increase_raises_level, context)
    assert outcome.triggered is True
    assert outcome.severity.rank > context.risk_level.rank


def test_r4_is_inert_without_a_prediction() -> None:
    """A missing GNN model must never be read as 'the event will intensify'."""
    outcome = outcome_for(rule_predicted_increase_raises_level, make_context())
    assert outcome.triggered is False
    assert outcome.severity is None
    assert "no GNN prediction" in outcome.reason


def test_r4_does_not_fire_when_the_prediction_is_flat() -> None:
    outcome = outcome_for(
        rule_predicted_increase_raises_level, make_context(predicted_intensity=45.0)
    )
    assert outcome.triggered is False



# --------------------------------------------------------------------------- #
# R5 - weakening downgrades or resolves
# --------------------------------------------------------------------------- #
def test_r5_resolves_an_event_that_fell_below_the_threshold() -> None:
    context = make_context(
        intensity=1.0, intensity_trend=TREND_WEAKENING, risk_level=AlertSeverity.WARNING
    )
    outcome = outcome_for(
        rule_weakening_downgrades_or_resolves,
        context,
        RuleConfig(resolve_intensity_threshold=5.0),
    )
    assert outcome.triggered is True
    assert outcome.status is AlertStatus.RESOLVED
    assert outcome.severity.rank < AlertSeverity.WARNING.rank


def test_r5_with_the_default_zero_threshold_only_resolves_at_zero() -> None:
    """The default threshold of 0.0 must not resolve a merely weak event."""
    context = make_context(
        intensity=0.5, intensity_trend=TREND_WEAKENING, risk_level=AlertSeverity.WARNING
    )
    outcome = outcome_for(rule_weakening_downgrades_or_resolves, context)
    assert outcome.status is None  # downgraded, not resolved


def test_r5_downgrades_a_weakening_event_that_is_still_significant() -> None:
    context = make_context(
        intensity=30.0, intensity_trend=TREND_WEAKENING, risk_level=AlertSeverity.WARNING
    )
    outcome = outcome_for(rule_weakening_downgrades_or_resolves, context)
    assert outcome.triggered is True
    assert outcome.status is None
    assert outcome.severity.rank < AlertSeverity.WARNING.rank


def test_r5_resolve_threshold_is_configurable() -> None:
    context = make_context(intensity=8.0, intensity_trend=TREND_WEAKENING)
    outcome = outcome_for(
        rule_weakening_downgrades_or_resolves,
        context,
        RuleConfig(resolve_intensity_threshold=5.0),
    )
    assert outcome.status is None  # 8.0 is above the configured threshold of 5.0


def test_r5_does_not_fire_for_a_non_weakening_event() -> None:
    outcome = outcome_for(
        rule_weakening_downgrades_or_resolves, make_context(intensity_trend=TREND_STABLE)
    )
    assert outcome.triggered is False


def test_r5_downgrade_can_be_disabled() -> None:
    context = make_context(intensity=30.0, intensity_trend=TREND_WEAKENING)
    outcome = outcome_for(
        rule_weakening_downgrades_or_resolves,
        context,
        RuleConfig(downgrade_on_weakening=False),
    )
    assert outcome.severity is None


# --------------------------------------------------------------------------- #
# R6 - a missing event expires
# --------------------------------------------------------------------------- #
def test_r6_expires_an_existing_alert_that_went_stale() -> None:
    context = make_context(existing_status=AlertStatus.ACTIVE, hours_since_observed=9.0)
    outcome = outcome_for(rule_missing_event_expires, context)
    assert outcome.triggered is True
    assert outcome.status is AlertStatus.EXPIRED


def test_r6_keeps_a_recent_alert_open() -> None:
    context = make_context(existing_status=AlertStatus.ACTIVE, hours_since_observed=1.0)
    outcome = outcome_for(rule_missing_event_expires, context)
    assert outcome.triggered is False


def test_r6_does_not_expire_a_newly_created_alert() -> None:
    """A first evaluation of a historical dataset is not staleness."""
    context = make_context(existing_status=None, hours_since_observed=100.0)
    outcome = outcome_for(rule_missing_event_expires, context)
    assert outcome.triggered is False
    assert "not subject to the expiry window" in outcome.reason


def test_r6_never_overrides_an_operator_decision() -> None:
    for status in (AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED):
        context = make_context(existing_status=status, hours_since_observed=99.0)
        outcome = outcome_for(rule_missing_event_expires, context)
        assert outcome.triggered is False


def test_r6_expiry_window_is_configurable() -> None:
    context = make_context(existing_status=AlertStatus.ACTIVE, hours_since_observed=4.0)
    outcome = outcome_for(
        rule_missing_event_expires, context, RuleConfig(expire_after_hours=2.0)
    )
    assert outcome.triggered is True


# --------------------------------------------------------------------------- #
# R7 - duplicates are suppressed
# --------------------------------------------------------------------------- #
def test_r7_detects_an_existing_open_alert() -> None:
    context = make_context(existing_status=AlertStatus.ACTIVE)
    outcome = outcome_for(rule_duplicate_suppressed, context)
    assert outcome.triggered is True
    assert "updated instead of creating a new one" in outcome.reason


def test_r7_fires_for_no_existing_alert() -> None:
    outcome = outcome_for(rule_duplicate_suppressed, make_context())
    assert outcome.triggered is False
    assert "no open alert matches" in outcome.reason


def test_r7_does_not_count_a_terminal_alert_as_a_duplicate() -> None:
    context = make_context(existing_status=AlertStatus.RESOLVED)
    outcome = outcome_for(rule_duplicate_suppressed, context)
    assert outcome.triggered is False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_rule_config_is_json_serialisable() -> None:
    payload = RuleConfig().as_dict()
    assert payload["min_trigger_severity"] == "WATCH"
    assert payload["deduplication_window_minutes"] == 60
    assert "SEVERE" in payload["elevate_for_severities"]

