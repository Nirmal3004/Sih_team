"""Data model for Phase 9 evaluation (SIH26078).

Three records carry the whole story of an evaluation:

``EvaluationRun``
    One end-to-end evaluation: what was evaluated, what could be measured, and
    what could not.
``MetricResult``
    A single number, or the honest reason there is not one.
``Provenance``
    Everything needed to reproduce the run.

The central design decision is in :class:`MetricResult`. Every number this
project reports is one of three things:

``MEASURED``
    Computed from a valid comparison on a held-out split.
``DERIVED``
    Computed, but from a quantity that is itself an estimate (for example a
    bootstrap interval width).
``UNAVAILABLE``
    **Not** a zero. Carries a reason explaining what was missing.

A ``MetricResult`` can never hold a value without a status, and a value of
``0.0`` is only legitimate when zero was actually measured. This is enforced
structurally: the alternative - returning ``None`` from a plain dict - is
exactly the mechanism by which a previous phase had to be careful, and a
dedicated type makes the mistake hard to write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

__all__ = [
    "DERIVED",
    "EvaluationRun",
    "FAILED",
    "INSUFFICIENT_DATA",
    "MEASURED",
    "NOT_AVAILABLE",
    "NOT_EVALUATED",
    "PARTIAL",
    "Provenance",
    "SUCCESS",
    "MetricResult",
    "utc_now",
]

#: The value was computed from a valid comparison on a held-out split.
MEASURED = "MEASURED"

#: The value was computed from an estimated quantity, not measured directly.
DERIVED = "DERIVED"

#: No value: the evaluation could not be performed. **Never** zero.
NOT_AVAILABLE = "NOT_AVAILABLE"

#: Enough data existed to attempt the evaluation, but the sample was too small
#: for the result to mean anything. Distinct from UNAVAILABLE: this one was
#: tried.
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

#: Deliberately out of scope, with the reason recorded.
NOT_EVALUATED = "NOT_EVALUATED"

#: The run completed and every component it attempted succeeded.
SUCCESS = "SUCCESS"

#: The run completed, but some components could not be evaluated.
PARTIAL = "PARTIAL"

#: Not enough data to evaluate anything meaningful.
RUN_INSUFFICIENT = "INSUFFICIENT_DATA"

#: The run could not complete.
FAILED = "FAILED"

UTC = timezone.utc


def utc_now() -> datetime:
    """Return the current UTC time as an aware datetime."""
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    """Return an ISO-8601 string, or ``None``."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class MetricResult:
    """One metric, with an explicit status.

    Construct a measured value with :meth:`measured`; construct an absent one
    with :meth:`unavailable` or :meth:`insufficient`. Both factories require the
    reason to be non-empty, so "no value, no explanation" is not
    representable.

    Attributes:
        name: Metric identifier, e.g. ``position_error_km_mean``.
        value: The measured value, or ``None``.
        status: MEASURED, DERIVED, NOT_AVAILABLE, INSUFFICIENT_DATA or
            NOT_EVALUATED.
        unit: Physical unit of ``value`` (e.g. ``km``), or ``None``.
        reason: Why the value is absent, or a note about how it was obtained.
        sample_count: Number of samples behind the value, when applicable.
        method: How it was computed, for reproducibility.
    """

    name: str
    value: float | None
    status: str
    unit: str | None = None
    reason: str = ""
    sample_count: int | None = None
    method: str = ""

    @classmethod
    def measured(
        cls,
        name: str,
        value: float,
        *,
        unit: str | None = None,
        sample_count: int | None = None,
        method: str = "",
    ) -> "MetricResult":
        """Return a value that was actually computed on a held-out split.

        Raises:
            ValueError: If the value is not finite. A NaN reaching a report
                would read as a real number to anything that parses it.
        """
        import math

        if not math.isfinite(float(value)):
            raise ValueError(
                f"Metric '{name}' was given a non-finite value ({value!r}). "
                "Use unavailable() or insufficient() instead."
            )
        return cls(
            name=name,
            value=float(value),
            status=MEASURED,
            unit=unit,
            sample_count=sample_count,
            method=method,
        )

    @classmethod
    def derived(
        cls,
        name: str,
        value: float,
        *,
        unit: str | None = None,
        reason: str = "",
        sample_count: int | None = None,
        method: str = "",
    ) -> "MetricResult":
        """Return a value computed from an estimated quantity.

        Args:
            name: Metric identifier.
            value: The derived value.
            unit: Physical unit.
            reason: What makes this derived rather than measured.
            sample_count: Samples behind it.
            method: How it was computed.
        """
        import math

        if not math.isfinite(float(value)):
            raise ValueError(f"Metric '{name}' has a non-finite value ({value!r}).")
        return cls(
            name=name,
            value=float(value),
            status=DERIVED,
            unit=unit,
            reason=reason,
            sample_count=sample_count,
            method=method,
        )

    @classmethod
    def unavailable(cls, name: str, reason: str, *, unit: str | None = None) -> "MetricResult":
        """Return "not computed", with the reason.

        Raises:
            ValueError: If ``reason`` is empty. An unexplained absence is
                indistinguishable from a bug, and defeats the point.
        """
        if not str(reason).strip():
            raise ValueError(
                f"Metric '{name}' was marked unavailable without a reason. "
                "Every absent metric must explain what was missing."
            )
        return cls(name=name, value=None, status=NOT_AVAILABLE, unit=unit, reason=reason)

    @classmethod
    def insufficient(
        cls, name: str, reason: str, *, unit: str | None = None, sample_count: int | None = None
    ) -> "MetricResult":
        """Return "attempted, but the sample was too small to mean anything"."""
        if not str(reason).strip():
            raise ValueError(f"Metric '{name}' was marked insufficient without a reason.")
        return cls(
            name=name,
            value=None,
            status=INSUFFICIENT_DATA,
            unit=unit,
            reason=reason,
            sample_count=sample_count,
        )

    @classmethod
    def not_evaluated(cls, name: str, reason: str, *, unit: str | None = None) -> "MetricResult":
        """Return "deliberately not attempted", with the reason."""
        if not str(reason).strip():
            raise ValueError(f"Metric '{name}' was marked not-evaluated without a reason.")
        return cls(name=name, value=None, status=NOT_EVALUATED, unit=unit, reason=reason)

    @property
    def available(self) -> bool:
        """True when a usable value is present."""
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "unit": self.unit,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "method": self.method,
        }

    def __float__(self) -> float:
        """Return the value, or raise.

        Raises:
            TypeError: If the metric has no value. Converting an absent metric
                to a float must fail loudly rather than yield 0.0.
        """
        if self.value is None:
            raise TypeError(
                f"Metric '{self.name}' has no value (status={self.status}: {self.reason}). "
                "Refusing to coerce an unavailable metric to a number."
            )
        return float(self.value)


@dataclass
class Provenance:
    """Everything needed to reproduce an evaluation.

    Attributes:
        seed: Random seed, so a bootstrap or ensemble is reproducible.
        dataset_id: Identifier of the evaluation dataset.
        dataset_checksum: Content hash of the dataset.
        split_boundaries: Train/validation/test boundaries.
        model_checkpoints: Checkpoint path and hash per model.
        configuration: The full evaluation configuration.
        package_version: The package version string.
        code_fingerprint: A hash of the evaluation source files, so a result
            can be traced to the code that produced it.
    """

    seed: int = 0
    dataset_id: str = ""
    dataset_checksum: str | None = None
    split_boundaries: dict[str, Any] = field(default_factory=dict)
    model_checkpoints: dict[str, str | None] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    package_version: str = ""
    code_fingerprint: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            "split_boundaries": dict(self.split_boundaries),
            "model_checkpoints": dict(self.model_checkpoints),
            "configuration": dict(self.configuration),
            "package_version": self.package_version,
            "code_fingerprint": self.code_fingerprint,
            "environment": dict(self.environment),
        }


@dataclass
class EvaluationRun:
    """One end-to-end evaluation.

    Attributes:
        run_id: Deterministic identifier derived from the inputs, so re-running
            the same evaluation produces the same id.
        dataset_id: The dataset evaluated.
        started_at: When the evaluation began.
        completed_at: When it finished.
        status: SUCCESS, PARTIAL, INSUFFICIENT_DATA or FAILED.
        models: Which models were evaluated, and their availability.
        metrics: Measured metrics, by group (``gnn``, ``downscaling``...).
        unavailable_metrics: Metrics that could not be produced, with reasons.
        warnings: Non-fatal problems.
        provenance: Reproducibility record.
    """

    run_id: str
    dataset_id: str
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = SUCCESS
    models: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    unavailable_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    comparisons: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    DISCLAIMER = (
        "Research prototype (SIH26078). Not an operational emergency-warning "
        "system. These results are measurements on a historical record, not "
        "demonstrated forecast skill, and no disaster-reduction or lives-saved "
        "claim is implied."
    )

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration, or ``None`` while the run is unfinished."""
        if self.completed_at is None:
            return None
        if self.started_at.tzinfo is None:
            start = self.started_at.replace(tzinfo=UTC)
        else:
            start = self.started_at
        end = self.completed_at
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        return max(0.0, (end - start).total_seconds())

    def add_metric(self, group: str, metric: MetricResult) -> MetricResult:
        """Record a metric, routing absent values to ``unavailable_metrics``.

        Returns:
            The metric, for chaining.
        """
        if metric.available:
            self.metrics.setdefault(group, {})[metric.name] = metric.as_dict()
        else:
            bucket = self.unavailable_metrics.setdefault(group, {})
            bucket[metric.name] = metric.as_dict()
        return metric

    def unavailable_reason(self, group: str, name: str) -> str:
        """Return the reason a metric is absent, or a generic message."""
        entry = self.unavailable_metrics.get(group, {}).get(name)
        if entry:
            return str(entry.get("reason") or entry.get("status") or "not computed")
        return "not computed"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the whole evaluation."""
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "models": dict(self.models),
            "metrics": {k: dict(v) for k, v in self.metrics.items()},
            "unavailable_metrics": {k: dict(v) for k, v in self.unavailable_metrics.items()},
            "comparisons": dict(self.comparisons),
            "uncertainty": dict(self.uncertainty),
            "explainability": dict(self.explainability),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "provenance": self.provenance.as_dict(),
            "disclaimer": self.DISCLAIMER,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationRun":
        """Rebuild a run from :meth:`as_dict` output.

        Raises:
            ValueError: If the payload is missing a required field.
        """
        for key in ("run_id", "dataset_id"):
            if not payload.get(key):
                raise ValueError(f"Evaluation run payload is missing '{key}'.")
        started = payload.get("started_at")
        completed = payload.get("completed_at")
        provenance = payload.get("provenance", {}) or {}
        return cls(
            run_id=str(payload["run_id"]),
            dataset_id=str(payload["dataset_id"]),
            started_at=datetime.fromisoformat(str(started)) if started else utc_now(),
            completed_at=(
                datetime.fromisoformat(str(completed)) if completed else None
            ),
            status=str(payload.get("status", SUCCESS)),
            models=dict(payload.get("models", {}) or {}),
            metrics=dict(payload.get("metrics", {}) or {}),
            unavailable_metrics=dict(payload.get("unavailable_metrics", {}) or {}),
            warnings=list(payload.get("warnings", []) or []),
            provenance=Provenance(
                seed=int(provenance.get("seed", 0) or 0),
                dataset_id=str(provenance.get("dataset_id", "") or ""),
                dataset_checksum=provenance.get("dataset_checksum"),
                split_boundaries=dict(provenance.get("split_boundaries", {}) or {}),
                model_checkpoints=dict(provenance.get("model_checkpoints", {}) or {}),
                configuration=dict(provenance.get("configuration", {}) or {}),
                package_version=str(provenance.get("package_version", "") or ""),
                code_fingerprint=provenance.get("code_fingerprint"),
                environment=dict(provenance.get("environment", {}) or {}),
            ),
            comparisons=dict(payload.get("comparisons", {}) or {}),
            uncertainty=dict(payload.get("uncertainty", {}) or {}),
            explainability=dict(payload.get("explainability", {}) or {}),
            errors=list(payload.get("errors", []) or []),
        )
