"""Alerting and emergency response (SIH26078, Phase 5).

```text
Phase 1 events ─┐
Phase 2 tracks ─┼─> engine.assess_risk  -> rules.apply_rules -> store.AlertStore
Phase 3 GNN  ───┤                                              -> notifications
Phase 4 down. ──┘
```

Quick start::

    from extreme_weather.alerts import AlertService, AlertStore

    service = AlertService()
    result = service.generate()
    print(result.alert_count, "alerts")

Everything in this package is a **prototype**: the scores and severities are
rule-based, the data is synthetic, and none of it is an official warning. See
``docs/ALERTING.md``.
"""

from __future__ import annotations

from extreme_weather.alerts.engine import (
    FACTOR_WEIGHTS,
    RISK_DISCLAIMER,
    RiskConfig,
    RiskContext,
    assess_risk,
    clamp01,
)
from extreme_weather.alerts.models import (
    ALERT_NOTICE,
    CONFIDENCE_LABELS,
    Alert,
    AlertListResponse,
    AlertStatus,
    AlertStoreResult,
    AlertSummary,
    ConfidenceLabel,
    ContributingFactor,
    RiskAssessment,
    SeverityCount,
    StatusCount,
    build_alert_id,
    deduplication_key,
    time_bucket,
)
from extreme_weather.alerts.notifications import (
    MockNotificationProvider,
    NotificationError,
    NotificationProvider,
    NotificationRecord,
    NotificationResult,
    NotificationStore,
    build_notification_providers,
)
from extreme_weather.alerts.rules import (
    ALERT_RULES,
    AlertRule,
    RuleConfig,
    RuleOutcome,
    apply_rules,
)
from extreme_weather.alerts.service import (
    AlertError,
    AlertGenerationResult,
    AlertService,
    AlertStoreError,
    AlertTransitionError,
    AlertUnknownError,
)
from extreme_weather.alerts.severity import (
    DEFAULT_SEVERITY_THRESHOLDS,
    SEVERITY_DISCLAIMER,
    AlertSeverity,
    SeverityThresholds,
    max_severity,
    severity_for_score,
)
from extreme_weather.alerts.store import AlertStore
from extreme_weather.alerts.simulation import SimulationResult, run_simulation

__all__ = [
    "ALERT_NOTICE",
    "ALERT_RULES",
    "Alert",
    "AlertError",
    "AlertGenerationResult",
    "AlertListResponse",
    "AlertRule",
    "AlertService",
    "AlertSeverity",
    "AlertStatus",
    "AlertStore",
    "AlertStoreError",
    "AlertStoreResult",
    "AlertSummary",
    "AlertTransitionError",
    "AlertUnknownError",
    "CONFIDENCE_LABELS",
    "ConfidenceLabel",
    "ContributingFactor",
    "DEFAULT_SEVERITY_THRESHOLDS",
    "FACTOR_WEIGHTS",
    "MockNotificationProvider",
    "NotificationError",
    "NotificationProvider",
    "NotificationRecord",
    "NotificationResult",
    "NotificationStore",
    "RISK_DISCLAIMER",
    "RiskAssessment",
    "RiskConfig",
    "RiskContext",
    "RuleConfig",
    "RuleOutcome",
    "SEVERITY_DISCLAIMER",
    "SeverityCount",
    "SeverityThresholds",
    "SimulationResult",
    "StatusCount",
    "apply_rules",
    "assess_risk",
    "build_alert_id",
    "build_notification_providers",
    "clamp01",
    "deduplication_key",
    "max_severity",
    "run_simulation",
    "severity_for_score",
    "time_bucket",
]
