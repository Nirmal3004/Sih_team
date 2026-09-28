"""Configurable alert rule engine (SIH26078, Phase 5).

The rules are deliberately simple, named and individually toggleable. Each one
is a pure function of a :class:`RuleContext` that returns an
:class:`RuleOutcome`, so a rule can be unit-tested on its own and the full
decision is auditable: the applied rules are recorded on the alert.

Rules
-----
===== ================================ ==========================================
ID    Name                              Effect
===== ================================ ==========================================
R1    extreme_anomaly_creates_alert     an EXTREME/SEVERE detection is alertable
R2    severe_strengthening_elevates     SEVERE + strengthening raises priority
R3    persistent_track_maintains        a persisted, tracked event stays ACTIVE
R4    predicted_increase_raises_level   a rising predicted intensity raises it
R5    weakening_downgrades_or_resolves  a weakening event below the bar resolves
R6    missing_event_expires             an event gone for N hours expires
R7    duplicate_suppressed              same event+level+bucket never duplicates
===== ================================ ==========================================

None of these thresholds is an official warning criterion; they are prototype
tuning knobs on :class:`RuleConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from extreme_weather.alerts.models import AlertStatus, ConfidenceLabel
from extreme_weather.alerts.severity import AlertSeverity, max_severity
from extreme_weather.schemas.event import EventType, Severity as EventSeverity
from extreme_weather.tracking.metrics import TREND_STRENGTHENING, TREND_WEAKENING

__all__ = [
    "ALERT_RULES",
    "AlertRule",
    "RULE_ELEVATE_PRIORITY",
    "RULE_EXPIRE",
    "RULE_MAINTAIN_ACTIVE",
    "RULE_NO_DUPLICATE",
    "RULE_RAISE_LEVEL",
    "RULE_RESOLVE",
    "RULE_TRIGGER",
    "RuleConfig",
    "RuleContext",
    "RuleOutcome",
    "apply_rules",
    "active_rules",
]


class RuleID:
    """Stable identifiers of the rules, used in tests and in alert metadata."""

    TRIGGER = "R1"
    ELEVATE_PRIORITY = "R2"
    MAINTAIN_ACTIVE = "R3"
    RAISE_LEVEL = "R4"
    RESOLVE = "R5"
    EXPIRE = "R6"
    NO_DUPLICATE = "R7"


RULE_TRIGGER = RuleID.TRIGGER
RULE_ELEVATE_PRIORITY = RuleID.ELEVATE_PRIORITY
RULE_MAINTAIN_ACTIVE = RuleID.MAINTAIN_ACTIVE
RULE_RAISE_LEVEL = RuleID.RAISE_LEVEL
RULE_RESOLVE = RuleID.RESOLVE
RULE_EXPIRE = RuleID.EXPIRE
RULE_NO_DUPLICATE = RuleID.NO_DUPLICATE


@dataclass(frozen=True)
class RuleConfig:
    """Tunable thresholds of the rule engine.

    Every field here is a prototype tuning knob. None of them is derived from,
    or claimed to match, an official warning criterion.
    """

    #: Minimum Phase 1 severity that may create an alert (R1).
    min_trigger_severity: EventSeverity = EventSeverity.WATCH
    #: Phase 1 severities whose alerts get an elevated priority (R2).
    elevate_for_severities: tuple[EventSeverity, ...] = (
        EventSeverity.SEVERE,
        EventSeverity.EXTREME,
    )
    #: Intensity that must grow to be called "strengthening" (R2/R4).
    strengthening_delta: float = 0.0
    #: Minimum track duration in hours for an event to count as persistent (R3).
    persistence_hours: float = 3.0
    #: Minimum predicted intensity gain to raise the level (R4).
    predicted_increase_delta: float = 0.0
    #: Intensity below which a weakening event is resolved (R5).
    resolve_intensity_threshold: float = 0.0
    #: Hours without a detection after which an alert expires (R6).
    expire_after_hours: float = 6.0
    #: Deduplication window in minutes (R7).
    deduplication_window_minutes: int = 60
    #: Validity of a freshly created alert, in minutes.
    default_validity_minutes: int = 360
    #: Whether a weakening-but-not-gone event is downgraded instead of resolved.
    downgrade_on_weakening: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        return {
            "min_trigger_severity": self.min_trigger_severity.value,
            "elevate_for_severities": [s.value for s in self.elevate_for_severities],
            "strengthening_delta": self.strengthening_delta,
            "persistence_hours": self.persistence_hours,
            "predicted_increase_delta": self.predicted_increase_delta,
            "resolve_intensity_threshold": self.resolve_intensity_threshold,
            "expire_after_hours": self.expire_after_hours,
            "deduplication_window_minutes": self.deduplication_window_minutes,
            "default_validity_minutes": self.default_validity_minutes,
            "downgrade_on_weakening": self.downgrade_on_weakening,
        }


@dataclass(frozen=True)
class RuleContext:
    """The facts a rule may inspect, assembled by the service.

    Every optional field corresponds to a stage that may be missing, and every
    rule treats "missing" as a documented neutral value rather than a crash.
    """

    event_id: str
    event_type: EventType
    event_severity: EventSeverity
    intensity: float
    observed_at: datetime
    risk_level: AlertSeverity = AlertSeverity.INFO
    risk_score: float = 0.0
    # --- optional pipeline stages ---
    track_id: str | None = None
    duration_hours: float = 0.0
    intensity_trend: str | None = None
    predicted_intensity: float | None = None
    has_downscaled_threat: bool = False
    # --- existing alert, when re-evaluating one ---
    existing_status: AlertStatus | None = None
    existing_level: AlertSeverity | None = None
    hours_since_observed: float = 0.0


@dataclass(frozen=True)
class RuleOutcome:
    """What one rule decided.

    Attributes:
        rule_id: Stable rule identifier (``R1`` ... ``R7``).
        name: Human-readable rule name.
        triggered: Whether the rule fired.
        reason: Explanation recorded on the alert.
        severity: Resulting severity, when the rule changed it.
        priority: Resulting priority, when the rule changed it.
        status: Resulting status, when the rule changed it.
        suppress: True when the rule says "do not create an alert".
    """

    rule_id: str
    name: str
    triggered: bool
    reason: str = ""
    severity: AlertSeverity | None = None
    priority: AlertSeverity | None = None
    status: AlertStatus | None = None
    suppress: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the outcome as a JSON-serialisable dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "triggered": self.triggered,
            "reason": self.reason,
            "severity": self.severity.value if self.severity else None,
            "priority": self.priority.value if self.priority else None,
            "status": self.status.value if self.status else None,
            "suppress": self.suppress,
        }


@dataclass(frozen=True)
class AlertRule:
    """One named, documented, individually testable rule."""

    rule_id: str
    name: str
    description: str
    apply: Any  # Callable[[RuleContext, RuleConfig], RuleOutcome]
    enabled: bool = True


def _next_level(level: AlertSeverity) -> AlertSeverity:
    """Return the next severity up the ladder (capped at ``EXTREME``)."""
    order = list(AlertSeverity)
    return order[min(level.rank + 1, len(order) - 1)]


def _previous_level(level: AlertSeverity) -> AlertSeverity:
    """Return the next severity down the ladder (capped at ``INFO``)."""
    order = list(AlertSeverity)
    return order[max(level.rank - 1, 0)]



# --------------------------------------------------------------------------- #
# R1 - an extreme anomaly creates an alert
# --------------------------------------------------------------------------- #
def rule_extreme_anomaly_creates_alert(
    context: RuleContext, config: RuleConfig
) -> RuleOutcome:
    """R1: create an alert when the detection is at or above the trigger level.

    A detection whose Phase 1 severity is ``NORMAL`` never produces an alert:
    alerting on normal conditions is exactly the alert spam this project
    refuses to generate.
    """
    allowed = context.event_severity.rank >= config.min_trigger_severity.rank
    return RuleOutcome(
        rule_id=RuleID.TRIGGER,
        name="extreme_anomaly_creates_alert",
        triggered=allowed,
        reason=(
            f"Phase 1 severity {context.event_severity.value} is at or above the "
            f"trigger level {config.min_trigger_severity.value}"
            if allowed
            else (
                f"Phase 1 severity {context.event_severity.value} is below the "
                f"trigger level {config.min_trigger_severity.value}; no alert"
            )
        ),
        suppress=not allowed,
    )


# --------------------------------------------------------------------------- #
# R2 - a severe, strengthening event is elevated
# --------------------------------------------------------------------------- #
def rule_severe_strengthening_elevates(
    context: RuleContext, config: RuleConfig
) -> RuleOutcome:
    """R2: elevate the priority of a severe event that is strengthening.

    Elevation never lowers a severity: it only ever raises the *priority*, so an
    already-EXTREME event is not affected.
    """
    qualifying = context.event_severity in config.elevate_for_severities
    strengthening = context.intensity_trend == TREND_STRENGTHENING
    if qualifying and strengthening:
        return RuleOutcome(
            rule_id=RuleID.ELEVATE_PRIORITY,
            name="severe_strengthening_elevates",
            triggered=True,
            reason=(
                f"severity {context.event_severity.value} and the event is "
                "strengthening; operator priority raised"
            ),
            priority=_next_level(max(context.risk_level, AlertSeverity.ADVISORY)),
        )
    return RuleOutcome(
        rule_id=RuleID.ELEVATE_PRIORITY,
        name="severe_strengthening_elevates",
        triggered=False,
        reason=(
            f"needs severity in "
            f"{[s.value for s in config.elevate_for_severities]} and a "
            f"strengthening trend; got {context.event_severity.value} / "
            f"{context.intensity_trend or 'unknown'}"
        ),
    )


# --------------------------------------------------------------------------- #
# R3 - a persistent tracked event stays ACTIVE
# --------------------------------------------------------------------------- #
def rule_persistent_track_maintains(
    context: RuleContext, config: RuleConfig
) -> RuleOutcome:
    """R3: keep a persistently tracked event ACTIVE.

    An alert on a long-lived track must not quietly drop back to ``NEW``: the
    event is still being tracked, so the alert stays open.
    """
    if context.track_id is None or context.duration_hours < config.persistence_hours:
        return RuleOutcome(
            rule_id=RuleID.MAINTAIN_ACTIVE,
            name="persistent_track_maintains",
            triggered=False,
            reason=(
                f"needs a track lasting at least {config.persistence_hours} h; got "
                f"{context.duration_hours:.2f} h"
            ),
        )
    status = AlertStatus.ACKNOWLEDGED if context.existing_status is AlertStatus.ACKNOWLEDGED else AlertStatus.ACTIVE
    return RuleOutcome(
        rule_id=RuleID.MAINTAIN_ACTIVE,
        name="persistent_track_maintains",
        triggered=True,
        reason=(
            f"event has been tracked for {context.duration_hours:.2f} h "
            f"(>= {config.persistence_hours} h); alert kept {status.value}"
        ),
        status=status,
    )


# --------------------------------------------------------------------------- #
# R4 - a rising predicted intensity raises the level
# --------------------------------------------------------------------------- #
def rule_predicted_increase_raises_level(
    context: RuleContext, config: RuleConfig
) -> RuleOutcome:
    """R4: raise the level when the Phase 3 prediction shows growth.

    The rule is inert without a prediction: a missing GNN model must never be
    read as "the event will intensify".
    """
    if context.predicted_intensity is None:
        return RuleOutcome(
            rule_id=RuleID.RAISE_LEVEL,
            name="predicted_increase_raises_level",
            triggered=False,
            reason="no GNN prediction available; the level is unchanged",
        )
    delta = context.predicted_intensity - context.intensity
    if delta > config.predicted_increase_delta:
        return RuleOutcome(
            rule_id=RuleID.RAISE_LEVEL,
            name="predicted_increase_raises_level",
            triggered=True,
            reason=(
                f"predicted intensity {context.predicted_intensity:.2f} exceeds the "
                f"current {context.intensity:.2f} by {delta:+.2f}; level raised"
            ),
            severity=_next_level(context.risk_level),
            priority=_next_level(max(context.risk_level, AlertSeverity.WATCH)),
        )
    return RuleOutcome(
        rule_id=RuleID.RAISE_LEVEL,
        name="predicted_increase_raises_level",
        triggered=False,
        reason=(
            f"predicted intensity change {delta:+.2f} does not exceed "
            f"{config.predicted_increase_delta:+.2f}; the level is unchanged"
        ),
    )



# --------------------------------------------------------------------------- #
# R5 - a weakening event is downgraded or resolved
# --------------------------------------------------------------------------- #
def rule_weakening_downgrades_or_resolves(
    context: RuleContext, config: RuleConfig
) -> RuleOutcome:
    """R5: downgrade a weakening event, or resolve it once it dies out.

    Two distinct situations share one rule because they are the same decision
    at different points on the same curve:

    * the event is still detected but fading -> downgrade (or, once it is below
      the resolve threshold, resolve it);
    * the event is fading and already below the threshold -> resolve.
    """
    if context.intensity_trend != TREND_WEAKENING:
        return RuleOutcome(
            rule_id=RuleID.RESOLVE,
            name="weakening_downgrades_or_resolves",
            triggered=False,
            reason=(
                f"event is not weakening (trend {context.intensity_trend or 'unknown'})"
            ),
        )

    if context.intensity < config.resolve_intensity_threshold:
        return RuleOutcome(
            rule_id=RuleID.RESOLVE,
            name="weakening_downgrades_or_resolves",
            triggered=True,
            reason=(
                f"intensity {context.intensity:.2f} fell below the resolve threshold "
                f"{config.resolve_intensity_threshold:.2f}; alert resolved"
            ),
            status=AlertStatus.RESOLVED,
            severity=_previous_level(context.risk_level),
        )

    if not config.downgrade_on_weakening:
        return RuleOutcome(
            rule_id=RuleID.RESOLVE,
            name="weakening_downgrades_or_resolves",
            triggered=True,
            reason=(
                f"intensity is weakening to {context.intensity:.2f}; downgrade is "
                "disabled by configuration, so the alert is left unchanged"
            ),
        )

    return RuleOutcome(
        rule_id=RuleID.RESOLVE,
        name="weakening_downgrades_or_resolves",
        triggered=True,
        reason=(
            f"intensity is weakening to {context.intensity:.2f}, still at or above "
            f"the resolve threshold {config.resolve_intensity_threshold:.2f}; level lowered"
        ),
        severity=_previous_level(context.risk_level),
    )


# --------------------------------------------------------------------------- #
# R6 - an event that disappeared expires
# --------------------------------------------------------------------------- #
def rule_missing_event_expires(context: RuleContext, config: RuleConfig) -> RuleOutcome:
    """R6: expire an alert whose event has not been seen for long enough.

    Expiry is a safety net against alerts that would otherwise stay open
    forever, so it only applies to an alert that **already exists**: a freshly
    created alert starts open regardless of how old the underlying event is,
    because the first evaluation of a historical dataset is not "staleness".

    It also never applies to an alert a human already resolved or acknowledged,
    because those states record an operator decision.
    """
    if context.existing_status in (AlertStatus.RESOLVED, AlertStatus.ACKNOWLEDGED):
        return RuleOutcome(
            rule_id=RuleID.EXPIRE,
            name="missing_event_expires",
            triggered=False,
            reason=(
                f"alert is already {context.existing_status.value}; expiry does not "
                "override an operator decision"
            ),
        )
    if context.existing_status is None:
        return RuleOutcome(
            rule_id=RuleID.EXPIRE,
            name="missing_event_expires",
            triggered=False,
            reason="this is a new alert, so it is not subject to the expiry window",
        )
    if context.hours_since_observed >= config.expire_after_hours:
        return RuleOutcome(
            rule_id=RuleID.EXPIRE,
            name="missing_event_expires",
            triggered=True,
            reason=(
                f"the event has not been detected for {context.hours_since_observed:.2f} h "
                f"(>= {config.expire_after_hours} h); alert expired"
            ),
            status=AlertStatus.EXPIRED,
        )
    return RuleOutcome(
        rule_id=RuleID.EXPIRE,
        name="missing_event_expires",
        triggered=False,
        reason=(
            f"the event was seen {context.hours_since_observed:.2f} h ago, within the "
            f"{config.expire_after_hours} h expiry window"
        ),
    )


# --------------------------------------------------------------------------- #
# R7 - duplicates are suppressed
# --------------------------------------------------------------------------- #
def rule_duplicate_suppressed(
    context: RuleContext, config: RuleConfig
) -> RuleOutcome:
    """R7: suppress a candidate that duplicates an existing open alert.

    Duplication is decided by the *deduplication key*
    (``event_id + alert_level + time_bucket``) computed by the service, which
    this rule receives through :attr:`RuleContext.risk_level` and the existing
    alert. Keeping the decision here means the "no alert spam" policy is stated
    in one place alongside the other rules.
    """
    duplicate = bool(context.existing_status) and context.existing_status.is_open
    return RuleOutcome(
        rule_id=RuleID.NO_DUPLICATE,
        name="duplicate_suppressed",
        triggered=duplicate,
        reason=(
            f"an open alert already covers this event at {context.risk_level.value} "
            f"within the {config.deduplication_window_minutes} min window; "
            "the existing alert is updated instead of creating a new one"
            if duplicate
            else "no open alert matches the deduplication key; a new alert is created"
        ),
        suppress=False,
    )



#: The rule registry, in evaluation order. ``R1`` gates everything, ``R7`` runs
#: last because it is the duplicate check the service acts on.
ALERT_RULES: tuple[AlertRule, ...] = (
    AlertRule(
        rule_id=RuleID.TRIGGER,
        name="extreme_anomaly_creates_alert",
        description=(
            "A detection at or above the configured trigger severity creates an "
            "alert; anything weaker is suppressed."
        ),
        apply=rule_extreme_anomaly_creates_alert,
    ),
    AlertRule(
        rule_id=RuleID.ELEVATE_PRIORITY,
        name="severe_strengthening_elevates",
        description=(
            "A severe detection whose tracked intensity is strengthening gets an "
            "elevated operator priority."
        ),
        apply=rule_severe_strengthening_elevates,
    ),
    AlertRule(
        rule_id=RuleID.MAINTAIN_ACTIVE,
        name="persistent_track_maintains",
        description=(
            "An event tracked for at least the configured duration keeps its alert "
            "open (ACTIVE), and keeps ACKNOWLEDGED if a human already acknowledged it."
        ),
        apply=rule_persistent_track_maintains,
    ),
    AlertRule(
        rule_id=RuleID.RAISE_LEVEL,
        name="predicted_increase_raises_level",
        description=(
            "A Phase 3 prediction showing growth above the configured delta raises "
            "the alert level; with no prediction the rule is inert."
        ),
        apply=rule_predicted_increase_raises_level,
    ),
    AlertRule(
        rule_id=RuleID.RESOLVE,
        name="weakening_downgrades_or_resolves",
        description=(
            "A weakening event is downgraded, or resolved once its intensity falls "
            "below the configured resolve threshold."
        ),
        apply=rule_weakening_downgrades_or_resolves,
    ),
    AlertRule(
        rule_id=RuleID.EXPIRE,
        name="missing_event_expires",
        description=(
            "An alert whose event has not been detected for the configured window "
            "expires, unless a human already acted on it."
        ),
        apply=rule_missing_event_expires,
    ),
    AlertRule(
        rule_id=RuleID.NO_DUPLICATE,
        name="duplicate_suppressed",
        description=(
            "An open alert with the same deduplication key is updated rather than "
            "duplicated, so a re-run inside the window cannot spam alerts."
        ),
        apply=rule_duplicate_suppressed,
    ),
)


def active_rules(rules: Sequence[AlertRule] = ALERT_RULES) -> list[AlertRule]:
    """Return only the enabled rules, preserving registry order."""
    return [rule for rule in rules if rule.enabled]


def apply_rules(
    context: RuleContext,
    config: RuleConfig | None = None,
    rules: Sequence[AlertRule] = ALERT_RULES,
) -> list[RuleOutcome]:
    """Evaluate every enabled rule against ``context``.

    The rules are evaluated in registry order and each sees the *original*
    context. Combining their effects (a raised level, a resolved status) is the
    service's job, because the precedence between "raise the level" and "resolve
    the alert" is a policy decision that belongs in one place.

    Returns:
        One :class:`RuleOutcome` per enabled rule, in evaluation order.
    """
    effective = config or RuleConfig()
    return [rule.apply(context, effective) for rule in active_rules(rules)]

