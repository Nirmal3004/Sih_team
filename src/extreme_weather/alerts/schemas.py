"""Request and response schemas for the alerts API (SIH26078, Phase 5).

These mirror the domain models in :mod:`extreme_weather.alerts.models` in the
same way the other phases separate "the object" from "what the endpoint takes
and returns". Nothing here contains business logic: an endpoint validates its
input with these models and then calls
:class:`~extreme_weather.alerts.service.AlertService`.

Every request model uses ``extra="forbid"`` so a typo in a request body is a
422 rather than a silently ignored field, and every response model carries the
prototype notice so a client cannot display a severity without also seeing that
it is not an official warning.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.alerts.models import (
    ALERT_NOTICE,
    Alert,
    AlertListResponse,
    AlertStatus,
    AlertSummary,
    AlertSeverity,
)
from extreme_weather.alerts.notifications import NotificationRecord

__all__ = [
    "AcknowledgeRequest",
    "AlertActionResponse",
    "AlertDetailResponse",
    "AlertGenerateRequest",
    "AlertGenerateResponse",
    "AlertListResponse",
    "AlertSummaryResponse",
    "NotificationListResponse",
    "ResolveRequest",
    "SimulationRequest",
    "SimulationResponse",
    "SimulationStep",
]


class AlertGenerateRequest(BaseModel):
    """Body of ``POST /api/v1/alerts/generate``."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool = Field(
        default=False,
        description="When true, neither the alert store nor the notification log is written.",
    )
    notify: bool = Field(
        default=True, description="When false, the notification adapters are skipped."
    )
    expire_stale: bool = Field(
        default=True,
        description="Also expire alerts that are past their validity window (R6).",
    )


class AcknowledgeRequest(BaseModel):
    """Body of ``POST /api/v1/alerts/{alert_id}/acknowledge``."""

    model_config = ConfigDict(extra="forbid")

    note: str | None = Field(
        default=None,
        max_length=500,
        description="Free-text note recorded on the alert."
    )


class ResolveRequest(BaseModel):
    """Body of ``POST /api/v1/alerts/{alert_id}/resolve``."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(
        default=None,
        max_length=500,
        description="Why the alert was resolved."
    )


class AlertActionResponse(BaseModel):
    """Result of an acknowledge/resolve call."""

    model_config = ConfigDict(extra="forbid")

    alert: Alert
    message: str = Field(description="Human-readable confirmation.")
    notice: str = Field(default=ALERT_NOTICE)


class AlertGenerateResponse(BaseModel):
    """Result of ``POST /api/v1/alerts/generate``."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    created_count: int = Field(ge=0)
    updated_count: int = Field(ge=0)
    suppressed_count: int = Field(
        ge=0, description="Candidates dropped by deduplication or by the trigger rule."
    )
    alert_count: int = Field(ge=0)
    alerts: list[Alert] = Field(default_factory=list)
    notification_count: int = Field(
        default=0, ge=0, description="Notifications recorded during this run."
    )
    degraded: list[str] = Field(
        default_factory=list,
        description="Notes about which optional pipeline inputs were missing."
    )
    source: str | None = Field(default=None)
    notice: str = Field(default=ALERT_NOTICE)


class AlertDetailResponse(BaseModel):
    """A single alert plus the context needed to render its details view."""

    model_config = ConfigDict(extra="forbid")

    alert: Alert
    timeline: list["SimulationStep"] = Field(
        default_factory=list,
        description="Ordered lifecycle milestones with their real timestamps."
    )
    notice: str = Field(default=ALERT_NOTICE)


class AlertSummaryResponse(BaseModel):
    """Result of ``GET /api/v1/alerts/summary``."""

    model_config = ConfigDict(extra="forbid")

    summary: AlertSummary
    notice: str = Field(default=ALERT_NOTICE)


class NotificationListResponse(BaseModel):
    """Result of ``GET /api/v1/notifications``."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    records: list[NotificationRecord] = Field(default_factory=list)
    source: str | None = Field(default=None)
    message: str | None = Field(default=None)
    notice: str = Field(
        default=(
            "Prototype notification log. Nothing here was delivered to an external service."
        ),
    )


class SimulationRequest(BaseModel):
    """Body of ``POST /api/v1/alerts/simulate``."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(
        default=6,
        ge=1,
        le=12,
        description="How many lifecycle steps to walk through (max 12)."
    )
    acknowledge_index: int = Field(
        default=2, ge=0, le=11, description="Step at which the alert is acknowledged."
    )
    resolve_index: int = Field(
        default=4, ge=0, le=11, description="Step at which the alert is resolved."
    )
    dry_run: bool = Field(
        default=False, description="When true, nothing is written to the stores."
    )


class SimulationStep(BaseModel):
    """One step of a simulation, with only real timestamps."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    stage: str = Field(
        min_length=1,
        description="Pipeline stage this step represents (detected, tracked, ...)."
    )
    description: str = Field(min_length=1)
    timestamp: datetime | None = Field(
        default=None, description="Real timestamp, or null when not applicable."
    )
    alert_id: str | None = None
    status: AlertStatus | None = None
    severity: AlertSeverity | None = None
    score: float | None = None
    simulated: Literal[True] = True


class SimulationResponse(BaseModel):
    """Result of ``POST /api/v1/alerts/simulate``."""

    model_config = ConfigDict(extra="forbid")

    steps: list[SimulationStep] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    notification_count: int = Field(default=0, ge=0)
    dry_run: bool = False
    deterministic: Literal[True] = True
    label: str = Field(
        default="SIMULATION - SYNTHETIC DATA",
        description="Always shown so a demo can never be mistaken for a real event.",
    )
    disclaimer: str = Field(
        default=(
            "This is a deterministic simulation built on synthetic event data. It "
            "demonstrates the alert lifecycle; it is not a real weather event, not a "
            "forecast, and not an official warning."
        )
    )
    notice: str = Field(default=ALERT_NOTICE)


# Resolve the forward reference now that SimulationStep exists.
AlertDetailResponse.model_rebuild()
