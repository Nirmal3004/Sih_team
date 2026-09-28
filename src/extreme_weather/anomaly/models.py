"""Analysis result models (SIH26078, Phase 1).

The strongly typed event object lives in
:mod:`extreme_weather.schemas.event` because it is part of the public API
contract; it is re-exported here so callers only need one import. This module
adds the container types that describe a *whole* analysis run: which dataset,
climatology and thresholds produced it, and the detected events.

``AnomalyAnalysisResult`` deliberately holds only JSON-serialisable metadata and
the event objects. The heavy score/mask arrays stay in Xarray objects and are
passed separately (see :mod:`extreme_weather.anomaly.service`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping

from extreme_weather.schemas.event import (
    EventListResponse,
    EventType,
    ExtremeWeatherEvent,
    Severity,
)

__all__ = [
    "AnomalyAnalysisResult",
    "EventListResponse",
    "EventType",
    "ExtremeWeatherEvent",
    "Severity",
]


@dataclass(frozen=True)
class AnomalyAnalysisResult:
    """Everything one anomaly-detection run produced.

    Attributes:
        events: Detected :class:`ExtremeWeatherEvent` objects, in analysis order.
        variables: Variables that were analysed, for example ``["precipitation"]``.
        source_dataset: Path of the analysed dataset.
        reference_dataset: Path of the reference dataset (``None`` when the
            analysis used its own dataset as reference).
        score_name: Score used for the detection (``efi`` by default).
        grouping: Climatology grouping used.
        generated_at: UTC timestamp of the analysis.
        metadata: Additional provenance (thresholds, counts, versions).
    """

    events: tuple[ExtremeWeatherEvent, ...] = ()
    variables: tuple[str, ...] = ()
    source_dataset: str | None = None
    reference_dataset: str | None = None
    score_name: str = "absolute"
    grouping: str = "month"
    generated_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def event_count(self) -> int:
        """Number of detected events."""
        return len(self.events)

    def by_severity(self, severity: Severity | str) -> list[ExtremeWeatherEvent]:
        """Return the events of one severity class."""
        wanted = severity if isinstance(severity, Severity) else Severity.from_value(str(severity))
        return [event for event in self.events if event.severity is wanted]

    def by_event_type(self, event_type: EventType | str) -> list[ExtremeWeatherEvent]:
        """Return the events of one event type."""
        wanted = (
            event_type if isinstance(event_type, EventType) else EventType.from_value(str(event_type))
        )
        return [event for event in self.events if event.event_type is wanted]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary (stable key order)."""
        return {
            "schema_version": "1.0",
            "project": "SIH26078",
            "service": "DeepWeather AI",
            "phase": "Phase 1 - anomaly detection (prototype)",
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "source_dataset": self.source_dataset,
            "reference_dataset": self.reference_dataset,
            "score_name": self.score_name,
            "grouping": self.grouping,
            "variables": list(self.variables),
            "event_count": self.event_count,
            "metadata": dict(self.metadata),
            "events": [event.model_dump(mode="json") for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnomalyAnalysisResult":
        """Rebuild a result from :meth:`as_dict` output."""
        return cls(
            events=tuple(ExtremeWeatherEvent.model_validate(item) for item in payload.get("events", [])),
            variables=tuple(payload.get("variables", ())),
            source_dataset=payload.get("source_dataset"),
            reference_dataset=payload.get("reference_dataset"),
            score_name=str(payload.get("score_name", "efi")),
            grouping=str(payload.get("grouping", "month")),
            generated_at=(
                datetime.fromisoformat(payload["generated_at"]) if payload.get("generated_at") else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_asdict(self) -> dict[str, Any]:
        """Alias of :meth:`as_dict` kept for symmetry with the Phase 0 reports."""
        return asdict(self)
