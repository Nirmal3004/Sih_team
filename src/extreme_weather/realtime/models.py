"""Run and stage records for the near-real-time pipeline (SIH26078, Phase 8).

Two dataclasses carry the whole story of a run: :class:`StageResult` for one
step, and :class:`PipelineRun` for the run. Both serialise straight to JSON for
the API, and both are written to disk so a run can be inspected after the fact.

The status vocabulary is deliberately narrow, and the distinction that matters
most is ``SKIPPED`` versus ``FAILED``:

* ``SKIPPED``  - the stage was not applicable (no trained model, no events).
  The run as a whole can still be a success.
* ``FAILED``   - the stage tried and did not work. The run is not a success.

Collapsing the two would force a choice between "the pipeline is broken" and
"a prediction was silently omitted", and the honest answer is neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "EXPIRED",
    "FAILED",
    "FRESH",
    "IngestionRecord",
    "PENDING",
    "RUNNING",
    "SKIPPED",
    "STALE",
    "SUCCESS",
    "StageResult",
    "StageStatus",
    "UNKNOWN",
    "WARNING",
    "utc_now",
]

#: A stage is queued but not started.
PENDING = "PENDING"
#: A stage is currently executing.
RUNNING = "RUNNING"
#: The stage completed and produced its artifact.
SUCCESS = "SUCCESS"
#: The stage completed, but something in the result is worth flagging.
WARNING = "WARNING"
#: The stage attempted its work and it did not succeed.
FAILED = "FAILED"
#: The stage was not applicable; the run may still be a success.
SKIPPED = "SKIPPED"

#: The data is within its freshness window.
FRESH = "FRESH"
#: The data is older than the fresh window but newer than the expiry window.
STALE = "STALE"
#: The data is older than the expiry window and must not be used as current.
EXPIRED = "EXPIRED"
#: The source timestamp could not be determined.
UNKNOWN = "UNKNOWN"

#: Allowed values of :attr:`StageResult.status`.
StageStatus = str

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


def _age(start: datetime | None, end: datetime | None) -> float | None:
    """Return the elapsed seconds between two timestamps, or ``None``."""
    if start is None or end is None:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return max(0.0, (end - start).total_seconds())


@dataclass
class StageResult:
    """The outcome of one pipeline stage.

    Attributes:
        name: Stage identifier, e.g. ``anomaly``.
        status: One of the module-level status constants.
        started_at: When the stage began.
        completed_at: When the stage finished.
        duration: Elapsed seconds, derived when not supplied.
        message: Human-readable explanation, always populated on
            ``WARNING``/``FAILED``/``SKIPPED`` so the reason is never implicit.
        output: Artifact path or summary the stage produced.
    """

    name: str
    status: str = PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: float | None = None
    message: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def complete(
        self,
        status: str,
        *,
        message: str = "",
        output: dict[str, Any] | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
    ) -> "StageResult":
        """Stamp the stage as finished and return it.

        A ``FAILED`` or ``SKIPPED`` stage with no message is filled in with its
        status, so a bare ``{"status": "FAILED"}`` can never be the whole record
        a user sees.
        """
        self.status = status
        self.completed_at = completed_at or utc_now()
        self.duration = _age(self.started_at, self.completed_at)
        if output:
            self.output.update(output)
        if error:
            self.error = error
        if not self.message:
            self.message = message or f"{self.name} {status.lower()}"
        return self

    @property
    def succeeded(self) -> bool:
        """True when the stage completed usefully (``SUCCESS`` or ``WARNING``)."""
        return self.status in (SUCCESS, WARNING)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the stage."""
        return {
            "name": self.name,
            "status": self.status,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "duration": self.duration,
            "message": self.message,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class IngestionRecord:
    """Provenance for one ingestion attempt (SIH26078, Phase 8, section 5).

    Attributes:
        run_id: The run this ingestion belongs to.
        provider: Provider name.
        source_url: Endpoint or file used.
        source_timestamp: The newest timestamp the source reported.
        ingested_at: When this system received the data.
        freshness: FRESH, STALE, EXPIRED or UNKNOWN.
        status: SUCCESS or FAILED.
        checksum: SHA-256 of the fetched data, for idempotency.
        attempts: How many attempts were made, including retries.
    """

    run_id: str
    provider: str
    status: str = SUCCESS
    source_url: str | None = None
    source_timestamp: str | None = None
    ingested_at: datetime = field(default_factory=utc_now)
    freshness: str = UNKNOWN
    age_minutes: float | None = None
    variables: tuple[str, ...] = ()
    time_steps: int = 0
    spatial_extent: dict[str, Any] = field(default_factory=dict)
    temporal_resolution: str | None = None
    checksum: str | None = None
    artifact_path: str | None = None
    attempts: int = 1
    error: str | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record.

        The source URL is scrubbed of any embedded credentials first: a URL is
        the one field most likely to be pasted into a bug report.
        """
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "status": self.status,
            "source_url": redact_url(self.source_url),
            "source_timestamp": self.source_timestamp,
            "ingested_at": _iso(self.ingested_at),
            "freshness": self.freshness,
            "age_minutes": self.age_minutes,
            "variables": list(self.variables),
            "time_steps": self.time_steps,
            "spatial_extent": self.spatial_extent,
            "temporal_resolution": self.temporal_resolution,
            "checksum": self.checksum,
            "artifact_path": self.artifact_path,
            "attempts": self.attempts,
            "error": self.error,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IngestionRecord":
        """Rebuild a record from :meth:`as_dict` output."""
        ingested = payload.get("ingested_at")
        parsed = datetime.fromisoformat(str(ingested)) if ingested else utc_now()
        return cls(
            run_id=str(payload.get("run_id", "")),
            provider=str(payload.get("provider", "")),
            status=str(payload.get("status", SUCCESS)),
            source_url=payload.get("source_url"),
            source_timestamp=payload.get("source_timestamp"),
            ingested_at=parsed,
            freshness=str(payload.get("freshness", UNKNOWN)),
            age_minutes=payload.get("age_minutes"),
            variables=tuple(payload.get("variables", ()) or ()),
            time_steps=int(payload.get("time_steps", 0) or 0),
            spatial_extent=dict(payload.get("spatial_extent", {}) or {}),
            temporal_resolution=payload.get("temporal_resolution"),
            checksum=payload.get("checksum"),
            artifact_path=payload.get("artifact_path"),
            attempts=int(payload.get("attempts", 1) or 1),
            error=payload.get("error"),
            dry_run=bool(payload.get("dry_run", False)),
        )

#: Query parameters whose values are treated as credentials when redacting a URL.
_CREDENTIAL_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "apikey",
        "api_key",
        "key",
        "token",
        "access_token",
        "password",
        "secret",
        "sig",
    }
)


def redact_url(url: str | None) -> str | None:
    """Return a URL safe to log or return over the API.

    Strips userinfo (``https://user:pass@host``) and masks the value of any
    credential-shaped query parameter, keeping the parameter *name* so a reader
    can still see that a key was meant to be there.

    Args:
        url: The URL to redact, or ``None``.

    Returns:
        The redacted URL, or ``None``.
    """
    if not url:
        return None
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    parsed = urlparse(str(url))
    if parsed.netloc and "@" in parsed.netloc:
        parsed = parsed._replace(netloc=parsed.netloc.rsplit("@", 1)[-1])
    if parsed.query:
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        masked = [
            (key, "REDACTED" if key.lower() in _CREDENTIAL_QUERY_KEYS else value)
            for key, value in pairs
        ]
        parsed = parsed._replace(query=urlencode(masked))
    return urlunparse(parsed)


@dataclass
class PipelineRun:
    """One end-to-end execution of the near-real-time pipeline.

    Attributes:
        run_id: Deterministic identifier; see
            :mod:`extreme_weather.realtime.idempotency`.
        status: Overall run status.
        started_at: When the run began.
        completed_at: When the run finished.
        provider: Provider used for ingestion.
        freshness: Freshness of the ingested data.
        stages: Per-stage outcomes, in execution order.
        errors: Error messages collected across stages.
        warnings: Non-fatal messages collected across stages.
    """

    run_id: str
    status: str = PENDING
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    provider: str = ""
    freshness: str = UNKNOWN
    source_timestamp: str | None = None
    age_minutes: float | None = None
    stages: list[StageResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ingestion: IngestionRecord | None = None
    incremental: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def total_duration(self) -> float | None:
        """Total wall-clock seconds, or ``None`` while the run is unfinished."""
        return _age(self.started_at, self.completed_at)

    def stage(self, name: str) -> StageResult | None:
        """Return the named stage, or ``None`` if it never ran."""
        for item in self.stages:
            if item.name == name:
                return item
        return None

    def add_stage(self, stage: StageResult) -> StageResult:
        """Append a stage to the run and return it."""
        self.stages.append(stage)
        return stage

    @property
    def is_terminal(self) -> bool:
        """True once the run has reached a final status."""
        return self.status in (SUCCESS, WARNING, FAILED, SKIPPED)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the entire run."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "total_duration": self.total_duration,
            "provider": self.provider,
            "source": self.provider,
            "freshness": self.freshness,
            "source_timestamp": self.source_timestamp,
            "age_minutes": self.age_minutes,
            "stages": [stage.as_dict() for stage in self.stages],
            "current_stage": self.current_stage,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "ingestion": self.ingestion.as_dict() if self.ingestion else None,
            "incremental": self.incremental,
            "dry_run": self.dry_run,
        }

    @property
    def current_stage(self) -> str | None:
        """Name of the stage currently running, or ``None`` when idle.

        Derived from the stage list rather than stored, so it can never drift out
        of sync with the stages it describes.
        """
        for stage in self.stages:
            if stage.status == RUNNING:
                return stage.name
        return None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineRun":
        """Rebuild a run from :meth:`as_dict` output."""
        started = payload.get("started_at")
        completed = payload.get("completed_at")
        ingestion = payload.get("ingestion")
        return cls(
            run_id=str(payload.get("run_id", "")),
            status=str(payload.get("status", PENDING)),
            started_at=(
                datetime.fromisoformat(str(started)) if started else utc_now()
            ),
            completed_at=(
                datetime.fromisoformat(str(completed)) if completed else None
            ),
            provider=str(payload.get("provider", "") or ""),
            freshness=str(payload.get("freshness", UNKNOWN) or UNKNOWN),
            source_timestamp=payload.get("source_timestamp"),
            age_minutes=payload.get("age_minutes"),
            stages=[
                StageResult(
                    name=str(item.get("name", "")),
                    status=str(item.get("status", PENDING)),
                    started_at=(
                        datetime.fromisoformat(str(item["started_at"]))
                        if item.get("started_at")
                        else None
                    ),
                    completed_at=(
                        datetime.fromisoformat(str(item["completed_at"]))
                        if item.get("completed_at")
                        else None
                    ),
                    duration=item.get("duration"),
                    message=str(item.get("message", "") or ""),
                    output=dict(item.get("output", {}) or {}),
                    error=item.get("error"),
                )
                for item in payload.get("stages", []) or []
            ],
            errors=list(payload.get("errors", []) or []),
            warnings=list(payload.get("warnings", []) or []),
            ingestion=(
                IngestionRecord.from_dict(ingestion) if isinstance(ingestion, dict) else None
            ),
            incremental=dict(payload.get("incremental", {}) or {}),
            dry_run=bool(payload.get("dry_run", False)),
        )

