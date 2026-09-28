"""Model evaluation endpoints (SIH26078, Phase 9).

```text
GET  /api/v1/evaluation/status           # what has been evaluated, and how
GET  /api/v1/evaluation/summary          # the unified Phase 9 report
GET  /api/v1/evaluation/gnn              # GNN vs persistence vs constant velocity
GET  /api/v1/evaluation/downscaling      # AI vs nearest vs cubic
GET  /api/v1/evaluation/uncertainty      # intervals, method and measured coverage
GET  /api/v1/evaluation/explain/{id}     # evidence behind one event
POST /api/v1/evaluation/run              # execute an evaluation now
GET  /api/v1/evaluation/runs/{run_id}    # one stored run
```

Status codes
------------
* **404** - no evaluation has been run, or the id/event is unknown.
* **422** - the request is invalid, or the evaluation was refused because it
  would leak (a non-chronological split).
* **503** - a dependency needed for the run is unavailable.

A **404 with no evaluation** is the honest response for the read endpoints. A
200 with null metrics would be indistinguishable from a measurement of zero, and
that is precisely the confusion this project has to avoid: a missing evaluation
is not a result.

Nothing here reports an accuracy percentage. Every payload carries the
provenance of the measurement and the prototype disclaimer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.config import Settings, get_settings
from extreme_weather.evaluation import EvaluationRun
from extreme_weather.evaluation.evaluator import run_evaluation
from extreme_weather.evaluation.explainability import (
    CAUSALITY_DISCLAIMER,
    build_event_explanation,
)

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])

#: Attached to every response so no consumer can present a measurement as an
#: operational claim.
EVALUATION_DISCLAIMER = (
    "Research prototype (SIH26078). Measurements on a historical record, not "
    "demonstrated forecast skill. No verified ground-truth warning catalogue "
    "exists for this project, so no accuracy or skill claim is made."
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class EvaluationStatusResponse(BaseModel):
    """Whether a Phase 9 evaluation exists, and what it covered."""

    model_config = ConfigDict(extra="allow")

    has_run: bool = False
    runs_present: list[str] = Field(default_factory=list)
    run_count: int = 0
    latest_run_id: str | None = None
    latest_status: str | None = None
    dataset_id: str | None = None
    measured_groups: list[str] = Field(default_factory=list)
    unavailable_groups: list[str] = Field(default_factory=list)
    message: str = ""
    disclaimer: str = EVALUATION_DISCLAIMER


class EvaluationSummaryResponse(BaseModel):
    """The unified Phase 9 report."""

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    status: str | None = None
    dataset_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    unavailable_metrics: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    comparisons: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = EVALUATION_DISCLAIMER


class GnnEvaluationResponse(BaseModel):
    """GNN results beside both baselines."""

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    sample_count: int = 0
    model_available: bool = False
    methods: dict[str, Any] = Field(default_factory=dict)
    comparisons: dict[str, Any] = Field(default_factory=dict)
    baselines: dict[str, Any] = Field(default_factory=dict)
    unavailable: dict[str, Any] = Field(default_factory=dict)
    interpretation: str = ""
    disclaimer: str = EVALUATION_DISCLAIMER


class DownscalingEvaluationResponse(BaseModel):
    """AI results beside the interpolation baselines."""

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    methods: dict[str, Any] = Field(default_factory=dict)
    comparisons: dict[str, Any] = Field(default_factory=dict)
    unavailable: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = EVALUATION_DISCLAIMER


class UncertaintyResponse(BaseModel):
    """Intervals, the method used, and any measured coverage."""

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    method: str = ""
    intervals: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    residual_distribution: dict[str, Any] = Field(default_factory=dict)
    caveat: str = ""
    disclaimer: str = EVALUATION_DISCLAIMER


class ExplanationResponse(BaseModel):
    """Evidence behind one event, with its gaps and caveats."""

    model_config = ConfigDict(extra="allow")

    event_id: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    available_evidence: list[dict[str, Any]] = Field(default_factory=list)
    unavailable_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_counts: dict[str, int] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str = CAUSALITY_DISCLAIMER


class EvaluationRunRequest(BaseModel):
    """Body of ``POST /api/v1/evaluation/run``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str | None = Field(
        default=None, description="Identifier of an existing evaluation dataset."
    )
    model: str = Field(
        default="fixture", description="Model whose availability is recorded."
    )
    seed: int = Field(default=20240601, description="Seed for the bootstrap.")
    run_id: str | None = Field(default=None, description="Explicit run id.")
    dry_run: bool = Field(default=False, description="Plan without persisting.")
    sample_count: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="Synthetic sample count for a self-test run. 0 means no run.",
    )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def _runs_dir(settings: Settings):
    """Return the directory evaluation runs are written to."""
    return settings.data_dir / "evaluation" / "runs"


def _read_runs(settings: Settings) -> list[dict[str, Any]]:
    """Return persisted evaluation runs, newest first.

    Unreadable files are skipped rather than raising: one corrupt record must
    not make the history endpoint fail.
    """
    import json

    directory = _runs_dir(settings)
    if not directory.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("eval-*.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as handle:
                runs.append(json.load(handle))
        except (OSError, ValueError):
            continue
    return runs


def _latest(settings: Settings) -> dict[str, Any] | None:
    """Return the most recent evaluation run, or ``None``."""
    runs = _read_runs(settings)
    return runs[0] if runs else None


def _write_run(settings: Settings, run: EvaluationRun) -> str | None:
    """Persist an evaluation run and return the path it was written to.

    Dry runs are filtered out by the caller, which is the only layer that
    knows whether the current request was a dry run.

    Returns:
        The written path, or ``None`` if the record could not be written.
        A failed write never fails the evaluation itself: losing the log of a
        run is bad, losing the run is worse.
    """
    import json

    try:
        directory = _runs_dir(settings)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run.run_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(run.as_dict(), handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        return str(path)
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/status", response_model=EvaluationStatusResponse, summary="Phase 9 evaluation status")
def evaluation_status(settings: Settings = Depends(get_settings)) -> EvaluationStatusResponse:
    """Return whether an evaluation exists and what it covered.

    Reports ``IDLE`` with no metrics when nothing has been run. It never
    invents a zero-valued result to fill the gap.
    """
    runs = _read_runs(settings)
    if not runs:
        return EvaluationStatusResponse(
            has_run=False,
            message=(
                "No Phase 9 evaluation has been run yet. Run "
                "POST /api/v1/evaluation/run, or scripts/run_phase9_evaluation.py."
            ),
        )
    latest = runs[0]
    return EvaluationStatusResponse(
        has_run=True,
        runs_present=[str(r.get("run_id")) for r in runs],
        run_count=len(runs),
        latest_run_id=str(latest.get("run_id")),
        latest_status=str(latest.get("status", "")),
        dataset_id=str(latest.get("dataset_id", "")),
        measured_groups=sorted(latest.get("metrics", {}).keys()),
        unavailable_groups=sorted(latest.get("unavailable_metrics", {}).keys()),
    )


@router.get("/summary", response_model=EvaluationSummaryResponse, summary="Unified Phase 9 report")
def evaluation_summary(settings: Settings = Depends(get_settings)) -> EvaluationSummaryResponse:
    """Return the most recent evaluation in full.

    Raises:
        HTTPException: 404 when no evaluation has been run.
    """
    latest = _latest(settings)
    if latest is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                "No Phase 9 evaluation has been run, so there is no summary to "
                "report. A 404 is returned rather than an empty 200, because a "
                "missing evaluation is not a result."
            ),
        )
    return EvaluationSummaryResponse(
        run_id=str(latest.get("run_id")),
        status=str(latest.get("status", "")),
        dataset_id=str(latest.get("dataset_id", "")),
        metrics=dict(latest.get("metrics", {}) or {}),
        unavailable_metrics=dict(latest.get("unavailable_metrics", {}) or {}),
        models=dict(latest.get("models", {}) or {}),
        comparisons=dict(latest.get("comparisons", {}) or {}),
        warnings=list(latest.get("warnings", []) or []),
        provenance=dict(latest.get("provenance", {}) or {}),
    )


@router.get("/gnn", response_model=GnnEvaluationResponse, summary="GNN vs baselines")
def evaluation_gnn(settings: Settings = Depends(get_settings)) -> GnnEvaluationResponse:
    """Return the GNN comparison against persistence and constant velocity.

    Raises:
        HTTPException: 404 when no evaluation has been run.
    """
    latest = _latest(settings)
    if latest is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="No Phase 9 evaluation has been run.",
        )
    comparison = (latest.get("comparisons", {}) or {}).get("gnn", {})
    return GnnEvaluationResponse(
        run_id=str(latest.get("run_id")),
        sample_count=int(comparison.get("sample_count", 0) or 0),
        model_available=bool(comparison.get("model_available", False)),
        methods=dict(comparison.get("methods", {}) or {}),
        comparisons=dict(comparison.get("comparisons", {}) or {}),
        baselines=dict(comparison.get("baselines", {}) or {}),
        unavailable=dict(comparison.get("unavailable", {}) or {}),
        interpretation=str(comparison.get("interpretation", "")),
    )


@router.get("/downscaling", response_model=DownscalingEvaluationResponse, summary="AI vs interpolation baselines")
def evaluation_downscaling(settings: Settings = Depends(get_settings)) -> DownscalingEvaluationResponse:
    """Return the downscaling comparison.

    Raises:
        HTTPException: 404 when no evaluation has been run.
    """
    latest = _latest(settings)
    if latest is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="No Phase 9 evaluation has been run.",
        )
    comparison = (latest.get("comparisons", {}) or {}).get("downscaling", {})
    if not comparison:
        return DownscalingEvaluationResponse(
            run_id=str(latest.get("run_id")),
            unavailable=latest.get("unavailable_metrics", {}).get("downscaling", {}),
        )
    return DownscalingEvaluationResponse(
        run_id=str(latest.get("run_id")),
        methods=dict(comparison.get("methods", {}) or {}),
        comparisons=dict(comparison.get("comparisons", {}) or {}),
        unavailable=dict(comparison.get("unavailable", {}) or {}),
    )


@router.get("/uncertainty", response_model=UncertaintyResponse, summary="Intervals and measured coverage")
def evaluation_uncertainty(settings: Settings = Depends(get_settings)) -> UncertaintyResponse:
    """Return the uncertainty intervals, the method, and any measured coverage.

    Raises:
        HTTPException: 404 when no evaluation has been run.
    """
    latest = _latest(settings)
    if latest is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="No Phase 9 evaluation has been run.",
        )
    uncertainty = latest.get("uncertainty", {}) or {}
    return UncertaintyResponse(
        run_id=str(latest.get("run_id")),
        method=str(uncertainty.get("method", "")),
        intervals=dict(uncertainty.get("intervals", {}) or {}),
        calibration=dict(uncertainty.get("calibration", {}) or {}),
        residual_distribution=dict(uncertainty.get("residual_distribution", {}) or {}),
        caveat=str(uncertainty.get("caveat", "")),
    )


@router.get(
    "/explain/{event_id}",
    response_model=ExplanationResponse,
    summary="Evidence behind one event",
    responses={404: {"description": "Unknown event id."}},
)
def explain_event(
    event_id: str, settings: Settings = Depends(get_settings)
) -> ExplanationResponse:
    """Return the supporting evidence recorded for one detected event.

    Every item is evidence the pipeline recorded or the model consumed. None of
    it is causal, and the response says so. Evidence that does not exist is
    returned as an explicit ``unavailable`` entry, never omitted.

    Raises:
        HTTPException: 404 when the event is unknown.
    """
    event = _load_event(event_id, settings)
    if event is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"No detected event with id '{event_id}'. Explanation requires a "
                "real detection; none is generated for an unknown id."
            ),
        )
    explanation = build_event_explanation(event, track=_load_track_for(event_id, settings))
    return ExplanationResponse.model_validate(explanation.as_dict())


@router.post(
    "/run",
    response_model=EvaluationSummaryResponse,
    summary="Execute a Phase 9 evaluation",
    responses={422: {"description": "Invalid request."}},
)
def run_phase9_evaluation(
    payload: EvaluationRunRequest,
    settings: Settings = Depends(get_settings),
) -> EvaluationSummaryResponse:
    """Run an evaluation and return the result.

    The API deliberately exposes no way to point the evaluation at the test
    split for training: the runner fits everything on validation and scores
    once on test. A request cannot change that, because there is no parameter
    to change it with.

    Raises:
        HTTPException: 422 when the request is invalid or the run is refused.
    """
    if payload.seed < 0:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="seed must be a non-negative integer.",
        )
    if payload.sample_count and payload.dry_run:
        # A dry run must not write, so it is a plan only.
        return EvaluationSummaryResponse(
            run_id=None,
            status="DRY_RUN",
            dataset_id=payload.dataset_id or "",
            message=(
                "Dry run: the plan was validated and no evaluation was "
                "performed, because a dry run writes no artifacts."
            ),
        )

    try:
        outcome = run_evaluation(
            None,
            model_available=payload.model != "fixture",
            model_checkpoint=None if payload.model == "fixture" else payload.model,
            seed=payload.seed,
            run_id=payload.run_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    outcome.run.models = {"gnn": {"name": payload.model, "scored": bool(outcome.run.metrics.get("gnn"))}}
    _write_run(settings, outcome.run)
    payload_dict = outcome.run.as_dict()
    return EvaluationSummaryResponse(
        run_id=str(payload_dict["run_id"]),
        status=str(payload_dict["status"]),
        dataset_id=str(payload_dict["dataset_id"]),
        metrics=dict(payload_dict["metrics"]),
        unavailable_metrics=dict(payload_dict["unavailable_metrics"]),
        models=dict(payload_dict["models"]),
        comparisons=dict(payload_dict["comparisons"]),
        warnings=list(payload_dict["warnings"]),
        provenance=dict(payload_dict["provenance"]),
    )


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationSummaryResponse,
    summary="One stored evaluation run",
    responses={404: {"description": "Unknown run id."}},
)
def evaluation_run_detail(
    run_id: str, settings: Settings = Depends(get_settings)
) -> EvaluationSummaryResponse:
    """Return one stored evaluation run.

    Raises:
        HTTPException: 404 when the run id is unknown.
    """
    for payload in _read_runs(settings):
        if str(payload.get("run_id")) == run_id:
            return EvaluationSummaryResponse(
                run_id=run_id,
                status=str(payload.get("status", "")),
                dataset_id=str(payload.get("dataset_id", "")),
                metrics=dict(payload.get("metrics", {}) or {}),
                unavailable_metrics=dict(payload.get("unavailable_metrics", {}) or {}),
                models=dict(payload.get("models", {}) or {}),
                comparisons=dict(payload.get("comparisons", {}) or {}),
                warnings=list(payload.get("warnings", []) or []),
                provenance=dict(payload.get("provenance", {}) or {}),
            )
    raise HTTPException(
        status_code=http_status.HTTP_404_NOT_FOUND,
        detail=f"No evaluation run with id '{run_id}'.",
    )


# --------------------------------------------------------------------------- #
# Event lookup
# --------------------------------------------------------------------------- #
def _load_event(event_id: str, settings: Settings) -> Any | None:
    """Load a detected event by id, or return ``None``."""
    from extreme_weather.anomaly.service import EventStore

    try:
        analysis = EventStore(settings.processed_dir / settings.events_filename).load()
    except Exception:
        return None
    if analysis is None:
        return None
    for event in analysis.events:
        if str(event.event_id) == event_id:
            return event
    return None


def _load_track_for(event_id: str, settings: Settings) -> Any | None:
    """Return the track containing an event, or ``None``."""
    from extreme_weather.tracking.store import TrackStore

    try:
        result = TrackStore(settings.processed_dir / settings.tracks_filename).load()
    except Exception:
        return None
    if result is None:
        return None
    # load() returns a TrackResult wrapper, not a bare list of tracks.
    for track in getattr(result, "tracks", ()) or ():
        for point in getattr(track, "points", ()) or ():
            if str(getattr(point, "event_id", "")) == event_id:
                return track
    return None
