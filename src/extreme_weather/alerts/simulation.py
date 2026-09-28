"""Deterministic alert simulation (SIH26078, Phase 5).

What this is
------------
A **demonstration** of the alert lifecycle on top of the existing synthetic
Phase 1-2 pipeline output. It walks a single synthetic event through

```text
detected -> tracked -> predicted -> downscaled -> alert created
         -> acknowledged -> resolved
```

and records the real timestamps of each stage where one exists.

What this is not
----------------
* It is **not** a real weather event. Every event it uses comes from the same
  deterministic synthetic generator the rest of the project uses.
* It does **not** fabricate observations. Where a stage has no real output
  (no GNN checkpoint, no downscaled field) the step says so explicitly and the
  alert's ``source_components`` omits it.
* It is **not** reproducible across *different* inputs but it **is** fully
  deterministic for a given input: the same events and tracks always yield the
  same steps, the same alert ids and the same scores, which is what makes the
  ``/simulate`` endpoint testable.

The response is labelled ``SIMULATION - SYNTHETIC DATA`` on every path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.alerts.models import Alert, AlertStatus
from extreme_weather.alerts.schemas import SimulationStep
from extreme_weather.alerts.severity import AlertSeverity

__all__ = [
    "SIMULATION_LABEL",
    "SIMULATION_START",
    "SimulationResult",
    "build_timeline",
    "run_simulation",
]


#: Always attached to a simulation result.
SIMULATION_LABEL = "SIMULATION - SYNTHETIC DATA"

#: Fixed start of the simulated timeline. The simulation must not depend on the
#: wall clock, so it anchors every step to a constant instead.
SIMULATION_START = datetime(2024, 6, 2, 2, 0, tzinfo=timezone.utc)

#: Synthetic downscaled threat values used by the simulation, per alert step.
#: They are *generated numbers* attached to synthetic events, not a model output.
_SIMULATED_THREAT: tuple[tuple[float, float], ...] = (
    (28.0, 5.0),
    (41.5, 5.0),
    (47.0, 5.0),
    (39.0, 5.0),
    (22.0, 5.0),
    (12.0, 5.0),
)


class SimulationResult(BaseModel):
    """Everything one simulation produced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: tuple[SimulationStep, ...] = ()
    alerts: tuple[Alert, ...] = ()
    notification_count: int = Field(default=0, ge=0)
    dry_run: bool = False
    label: str = SIMULATION_LABEL


def build_timeline(alert: Alert) -> list[SimulationStep]:
    """Build the lifecycle timeline of one alert from its real timestamps.

    Only stages that genuinely happened get an entry, and each entry carries the
    timestamp the pipeline actually recorded. A stage with no timestamp (for
    example "downscaled" when no downscaled field was available) is emitted with
    ``timestamp=None`` and a description that says why - a timestamp is never
    invented to fill a gap.
    """
    steps: list[SimulationStep] = []
    components = set(alert.source_components)

    def add(
        stage: str,
        description: str,
        timestamp: datetime | None,
        **extra: Any,
    ) -> None:
        steps.append(
            SimulationStep(
                index=len(steps),
                stage=stage,
                description=description,
                timestamp=timestamp,
                alert_id=alert.alert_id,
                **extra,
            )
        )

    add(
        "detected",
        "Phase 1 anomaly detection produced this event.",
        alert.created_at,
        severity=alert.risk.level,
        score=alert.risk.score,
    )
    add(
        "tracked",
        (
            f"Phase 2 linked this event into track {alert.track_id}."
            if "tracking" in components and alert.track_id
            else "No track is available for this event."
        ),
        alert.created_at if "tracking" in components else None,
    )
    add(
        "predicted",
        (
            "Phase 3 GNN produced a next-state prediction for this track."
            if alert.has_prediction
            else "No GNN prediction was available (the model may not be trained)."
        ),
        alert.created_at if alert.has_prediction else None,
    )
    add(
        "downscaled",
        (
            "Phase 4 downscaled threat information contributed to the score."
            if "downscaling" in components
            else "No AI-downscaled threat field was available for this event."
        ),
        alert.created_at if "downscaling" in components else None,
    )
    add(
        "alert_created",
        f"Alert created at {alert.severity.value} priority.",
        alert.created_at,
        status=AlertStatus.NEW,
        severity=alert.severity,
        score=alert.risk.score,
    )
    if alert.acknowledged_at is not None:
        add(
            "acknowledged",
            "An operator acknowledged this alert.",
            alert.acknowledged_at,
            status=AlertStatus.ACKNOWLEDGED,
        )
    if alert.resolved_at is not None:
        add(
            "resolved" if alert.status is AlertStatus.RESOLVED else "expired",
            alert.resolved_reason or f"Alert {alert.status.value.lower()}.",
            alert.resolved_at,
            status=alert.status,
        )
    return steps



def run_simulation(
    service: Any,
    *,
    steps: int = 6,
    acknowledge_index: int = 2,
    resolve_index: int = 4,
    dry_run: bool = False,
    threat_lookup: dict[str, tuple[float, float | None]] | None = None,
) -> SimulationResult:
    """Walk a synthetic event through the full alert lifecycle.

    The simulation reuses the *real* service, store, rules and notification
    adapters, so a demo shows what the pipeline actually does. The only synthetic
    additions are the fixed timeline anchor and the attached downscaled threat
    values, both of which are generated numbers on synthetic events.

    Args:
        service: An :class:`~extreme_weather.alerts.service.AlertService`.
        steps: How many lifecycle steps to produce.
        acknowledge_index: Step index at which the alert is acknowledged.
        resolve_index: Step index at which the alert is resolved.
        dry_run: When True, no store or notification file is written.
        threat_lookup: Optional ``event_id -> (threat, resolution_km)`` mapping,
            normally derived from the stored events.

    Returns:
        A :class:`SimulationResult`.

    Raises:
        ValueError: If ``resolve_index`` is not after ``acknowledge_index``.
    """
    if resolve_index <= acknowledge_index:
        raise ValueError(
            "resolve_index must be greater than acknowledge_index so the lifecycle "
            f"order is preserved (got {acknowledge_index} then {resolve_index})."
        )

    total = max(steps, resolve_index + 1)
    # The threat lookup goes on a clone so the caller's service is untouched.
    working = service.with_downscaled_threat(threat_lookup or {})
    generated = working.generate(dry_run=dry_run, now=SIMULATION_START, notify=not dry_run)
    alerts = list(generated.stored.alerts)

    if not alerts:
        return SimulationResult(
            steps=(
                SimulationStep(
                    index=0,
                    stage="no_input",
                    description=(
                        "No Phase 1 events are available, so the simulation has "
                        "nothing to work with. Run 'python scripts/detect_anomalies.py'."
                    ),
                    timestamp=None,
                ),
            ),
            alerts=(),
            dry_run=dry_run,
        )

    alert = alerts[0]
    timeline: list[SimulationStep] = [
        SimulationStep(
            index=0,
            stage="alert_created",
            description=(
                f"Alert created at {alert.severity.value} priority (score "
                f"{alert.risk.score:.2f}/100)."
            ),
            timestamp=alert.created_at,
            alert_id=alert.alert_id,
            status=alert.status,
            severity=alert.severity,
            score=alert.risk.score,
        )
    ]

    notifications = len(generated.notifications.records) if generated.notifications else 0
    if dry_run:
        timeline.append(
            SimulationStep(
                index=1,
                stage="acknowledged",
                description="Dry run: no acknowledgement was recorded.",
                timestamp=None,
                alert_id=alert.alert_id,
            )
        )
        timeline.append(
            SimulationStep(
                index=2,
                stage="resolved",
                description="Dry run: no resolution was recorded.",
                timestamp=None,
                alert_id=alert.alert_id,
            )
        )
        return SimulationResult(
            steps=tuple(timeline), alerts=tuple(alerts), notification_count=0, dry_run=True
        )

    # Acknowledging and resolving are idempotent for their own target state, so
    # re-running the simulation over an already-walked-through store reports the
    # recorded timestamps instead of failing on an illegal transition.
    current = service.get_alert(alert.alert_id)
    if current.status is not AlertStatus.ACKNOWLEDGED and current.status not in (
        AlertStatus.RESOLVED,
        AlertStatus.EXPIRED,
    ):
        service.acknowledge_alert(
            alert.alert_id,
            note="simulation: operator acknowledged the alert",
            now=SIMULATION_START + timedelta(hours=acknowledge_index + 1),
        )
    acknowledged = service.get_alert(alert.alert_id)
    timeline.append(
        SimulationStep(
            index=1,
            stage="acknowledged",
            description=(
                "An operator acknowledged the alert."
                if acknowledged.acknowledged_at is not None
                else "This alert is already closed, so it was not acknowledged again."
            ),
            timestamp=acknowledged.acknowledged_at,
            alert_id=alert.alert_id,
            status=AlertStatus.ACKNOWLEDGED if acknowledged.acknowledged_at else None,
        )
    )

    closed = service.get_alert(alert.alert_id)
    if not closed.status.is_terminal:
        service.resolve_alert(
            alert.alert_id,
            reason="simulation: synthetic event passed and was closed",
            now=SIMULATION_START + timedelta(hours=resolve_index + 1),
        )
    resolved = service.get_alert(alert.alert_id)
    timeline.append(
        SimulationStep(
            index=2,
            stage="resolved" if resolved.status is AlertStatus.RESOLVED else "expired",
            description=resolved.resolved_reason or "Alert resolved.",
            timestamp=resolved.resolved_at,
            alert_id=alert.alert_id,
            status=resolved.status,
        )
    )

    return SimulationResult(
        steps=tuple(timeline[:total]),
        alerts=tuple(service.get_alert_history()),
        notification_count=notifications,
        dry_run=False,
    )

