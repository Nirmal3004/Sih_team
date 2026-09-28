"""Near-real-time pipeline endpoints (SIH26078, Phase 8).

```text
GET  /api/v1/pipeline/status        # current state, freshness and stage summary
GET  /api/v1/pipeline/runs          # run history, newest first
GET  /api/v1/pipeline/runs/{id}     # one run, with every stage
POST /api/v1/pipeline/run           # execute the pipeline now
POST /api/v1/pipeline/run?dry_run=true
GET  /api/v1/ingestion/status       # latest ingestion and its provenance
```

Status codes are chosen so a client can react without parsing prose:

* **404** - the run id is unknown. Not a 200 with an empty object.
* **422** - the request is invalid, or the data was rejected by the quality
  gate. A ``422`` here means retrying the same request will not help.
* **503** - a provider or optional dependency is unavailable. A ``503`` here
  means retrying *might* help.

The distinction between 422 and 503 is the useful part: a caller must not retry
a rejected dataset, and should retry an unreachable provider.

Run execution is synchronous. For a prototype with a handful of stages and
second-scale datasets that keeps the API honest - the response describes a run
that has actually finished. Long windows should use the CLI's ``--interval``
mode rather than holding an HTTP request open.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from extreme_weather.config import Settings, get_settings
from extreme_weather.realtime.exceptions import (
    PipelineConfigurationError,
    ProviderUnavailableError,
    QualityGateRejection,
)
from extreme_weather.realtime.orchestrator import PipelineOrchestrator
from extreme_weather.realtime.providers import (
    WeatherRequest,
    build_provider,
)

router = APIRouter(prefix="/api/v1", tags=["pipeline"])

UTC = timezone.utc

#: Returned alongside every pipeline payload so a client cannot display these
#: numbers as operational guidance without the caveat attached.
PROTOTYPE_NOTICE = (
    "Research prototype (SIH26078). Not an operational emergency-warning "
    "system. No demonstrated forecast skill, and no verified accuracy claims."
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PipelineRunRequest(BaseModel):
    """Body of ``POST /api/v1/pipeline/run``."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default="fixture",
        description="Provider name: local, open-meteo, remote-http or fixture.",
    )
    hours: int = Field(
        default=6,
        ge=1,
        le=720,
        description="Length of the window ending at `end`, in hours.",
    )
    end: datetime | None = Field(
        default=None,
        description="Window end. Defaults to now.",
    )
    latitude_min: float = Field(default=12.0, ge=-90.0, le=90.0)
    latitude_max: float = Field(default=13.0, ge=-90.0, le=90.0)
    longitude_min: float = Field(default=76.0, ge=-180.0, le=180.0)
    longitude_max: float = Field(default=77.0, ge=-180.0, le=180.0)
    variables: list[str] = Field(
        default_factory=lambda: ["precipitation", "temperature"],
        description="Variables to request from the provider.",
    )
    file_path: str | None = Field(
        default=None, description="Local file, required by the 'local' provider."
    )
    url: str | None = Field(
        default=None, description="HTTPS endpoint, required by 'remote-http'."
    )
    dry_run: bool = Field(default=False, description="Plan the run without writing.")
    force: bool = Field(
        default=False, description="Reprocess even if this ingestion was seen before."
    )
    allow_stale: bool = Field(
        default=True,
        description="Continue when data is STALE or EXPIRED. Freshness is always "
        "reported either way.",
    )

    @field_validator("variables")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one variable is required")
        return [str(v).strip() for v in value if str(v).strip()]

    @field_validator("latitude_max")
    @classmethod
    def _latitude_order(cls, value: float, info: Any) -> float:
        low = info.data.get("latitude_min")
        if low is not None and value < low:
            raise ValueError("latitude_max must be >= latitude_min")
        return value

    @field_validator("longitude_max")
    @classmethod
    def _longitude_order(cls, value: float, info: Any) -> float:
        low = info.data.get("longitude_min")
        if low is not None and value < low:
            raise ValueError("longitude_max must be >= longitude_min")
        return value


class StageResponse(BaseModel):
    """One stage of a run."""

    model_config = ConfigDict(extra="allow")

    name: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    message: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineRunResponse(BaseModel):
    """A completed (or failed) pipeline run."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    total_duration: float | None = None
    provider: str = ""
    freshness: str = "UNKNOWN"
    source_timestamp: str | None = None
    age_minutes: float | None = None
    current_stage: str | None = None
    stages: list[StageResponse] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    incremental: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    disclaimer: str = PROTOTYPE_NOTICE


class PipelineStatusResponse(BaseModel):
    """Aggregate state of the near-real-time pipeline."""

    model_config = ConfigDict(extra="allow")

    provider: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    source_timestamp: str | None = None
    latest_timestamp: str | None = None
    freshness: dict[str, Any] = Field(default_factory=dict)
    last_successful_run: str | None = None
    last_run: str | None = None
    current_stage: str | None = None
    current_state: str = "IDLE"
    stages: list[StageResponse] = Field(default_factory=list)
    stage_order: list[str] = Field(default_factory=list)
    incremental: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    disclaimer: str = PROTOTYPE_NOTICE


class IngestionStatusResponse(BaseModel):
    """Latest ingestion and its provenance."""

    model_config = ConfigDict(extra="allow")

    latest_ingestion: dict[str, Any] | None = None
    source: str = ""
    freshness: dict[str, Any] = Field(default_factory=dict)
    latest_timestamp: str | None = None
    last_successful_run: str | None = None
    current_state: str = "IDLE"
    current_stage: str | None = None
    disclaimer: str = PROTOTYPE_NOTICE


class PipelineRunListResponse(BaseModel):
    """A page of run history."""

    model_config = ConfigDict(extra="allow")

    total: int
    runs: list[PipelineRunResponse] = Field(default_factory=list)
    disclaimer: str = PROTOTYPE_NOTICE

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _grid(start: float, stop: float, count: int = 3) -> tuple[float, ...]:
    """Build a small evenly spaced grid of points between ``start`` and ``stop``."""
    if count <= 1:
        return (float(start),)
    step = (float(stop) - float(start)) / (count - 1)
    return tuple(float(start) + step * i for i in range(count))


def _orchestrator(
    provider_name: str,
    *,
    file_path: str | None = None,
    url: str | None = None,
    settings: Settings | None = None,
) -> PipelineOrchestrator:
    """Build an orchestrator, translating configuration errors into 400s.

    Args:
        provider_name: The provider to construct.
        file_path: Local file, for the ``local`` provider.
        url: Endpoint, for the ``remote-http`` provider.
        settings: Application settings. Taken from the app when omitted, so a
            deployment with a custom ``DATA_DIR`` gets its pipeline state there
            rather than in the process-wide default.

    Raises:
        HTTPException: 400 when the provider name or its arguments are invalid.
    """
    try:
        provider = build_provider(provider_name, local_path=file_path, remote_url=url)
    except PipelineConfigurationError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return PipelineOrchestrator(provider, settings=settings or get_settings())


def _request_from(payload: PipelineRunRequest) -> WeatherRequest:
    """Translate the request body into a provider request."""
    end = payload.end or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(hours=int(payload.hours))
    return WeatherRequest(
        start=start,
        end=end,
        variables=tuple(payload.variables),
        latitude=_grid(payload.latitude_min, payload.latitude_max),
        longitude=_grid(payload.longitude_min, payload.longitude_max),
    )


def _run_response(run: Any) -> PipelineRunResponse:
    """Convert a PipelineRun (or an OrchestratorResult) into the API model."""
    record = getattr(run, "run", run)
    payload = record.as_dict()
    payload["disclaimer"] = PROTOTYPE_NOTICE
    return PipelineRunResponse.model_validate(payload)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get(
    "/pipeline/status",
    response_model=PipelineStatusResponse,
    summary="Current near-real-time pipeline state (Phase 8)",
)
def pipeline_status(
    provider: str = Query("fixture"),
    settings: Settings = Depends(get_settings),
) -> PipelineStatusResponse:
    """Return the pipeline's current state, freshness and latest stage summary.

    Never raises when no run has happened: it reports ``IDLE`` with a null
    timestamp, which is the honest answer for a pipeline that has not run.
    """
    orchestrator = _orchestrator(provider, settings=settings)
    orchestrator.load_state()
    return PipelineStatusResponse.model_validate(orchestrator.status())


@router.get(
    "/ingestion/status",
    response_model=IngestionStatusResponse,
    summary="Latest ingestion, freshness and provenance (Phase 8)",
)
def ingestion_status(
    provider: str = Query("fixture"),
    settings: Settings = Depends(get_settings),
) -> IngestionStatusResponse:
    """Return the most recent ingestion and the current pipeline state.

    Shared with the Phase 6 ``/api/v1/ingestion`` router: Phase 8 extends it
    with freshness rather than replacing it.
    """
    orchestrator = _orchestrator(provider, settings=settings)
    orchestrator.load_state()
    status = orchestrator.status()
    runs = orchestrator.load_runs(limit=1)
    latest = runs[0] if runs else None
    return IngestionStatusResponse(
        latest_ingestion=latest.ingestion.as_dict() if latest and latest.ingestion else None,
        source=status.get("source", ""),
        freshness=status.get("freshness", {}),
        latest_timestamp=status.get("latest_timestamp"),
        last_successful_run=status.get("last_successful_run"),
        current_state=status.get("current_state", "IDLE"),
        current_stage=status.get("current_stage"),
    )


@router.get(
    "/pipeline/runs",
    response_model=PipelineRunListResponse,
    summary="Pipeline run history, newest first (Phase 8)",
)
def pipeline_runs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    settings: Settings = Depends(get_settings),
) -> PipelineRunListResponse:
    """Return persisted runs, newest first.

    Returns an empty list rather than a 404 when nothing has run yet: "no runs
    yet" is a valid state, not a missing resource.
    """
    orchestrator = _orchestrator("fixture", settings=settings)
    orchestrator.load_state()
    runs = orchestrator.load_runs()
    window = runs[offset : offset + limit]
    return PipelineRunListResponse(
        total=len(runs),
        runs=[_run_response(run) for run in window],
    )


@router.get(
    "/pipeline/runs/{run_id}",
    response_model=PipelineRunResponse,
    summary="One pipeline run with every stage (Phase 8)",
    responses={404: {"description": "Unknown run id."}},
)
def pipeline_run_detail(
    run_id: str, settings: Settings = Depends(get_settings)
) -> PipelineRunResponse:
    """Return one run in full.

    Raises:
        HTTPException: 404 when the run id is not known.
    """
    orchestrator = _orchestrator("fixture", settings=settings)
    orchestrator.load_state()
    for run in orchestrator.load_runs():
        if run.run_id == run_id:
            return _run_response(run)
    raise HTTPException(
        status_code=http_status.HTTP_404_NOT_FOUND,
        detail=f"No pipeline run with id '{run_id}'.",
    )


@router.post(
    "/pipeline/run",
    response_model=PipelineRunResponse,
    summary="Execute the near-real-time pipeline now (Phase 8)",
    responses={
        400: {"description": "Invalid provider configuration."},
        404: {"description": "Unknown run id, when `run_id` is supplied."},
        422: {"description": "Invalid request, or the data was rejected."},
        503: {"description": "A provider or dependency is unavailable."},
    },
)
def pipeline_run(
    payload: PipelineRunRequest,
    dry_run: bool | None = Query(
        None, description="Plan the run without writing any artifact."
    ),
    run_id: str | None = Query(
        None, description="Reserved; a run's id is derived from its data."
    ),
    settings: Settings = Depends(get_settings),
) -> PipelineRunResponse:
    """Run the pipeline and return the finished run.

    The ``dry_run`` query parameter overrides the body field, so both
    ``POST /pipeline/run?dry_run=true`` and a body-level ``dry_run`` work.

    The request body is **required**. A state-changing endpoint should not fire
    on an empty request, so the provider and the window must always be stated
    explicitly.

    Raises:
        HTTPException: 400 for a bad provider configuration, 503 when the
            provider is unavailable, 422 when the quality gate rejected the data.
    """
    body = payload
    effective_dry_run = body.dry_run if dry_run is None else bool(dry_run)

    orchestrator = _orchestrator(
        body.provider, file_path=body.file_path, url=body.url, settings=settings
    )
    orchestrator.load_state()
    request = _request_from(body)

    if not orchestrator.provider.available() and body.provider == "local":
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The configured local file is not available.",
        )

    try:
        result = orchestrator.run(
            request, dry_run=effective_dry_run, force=body.force, allow_stale=body.allow_stale
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except PipelineConfigurationError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except QualityGateRejection as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return _run_response(result)

