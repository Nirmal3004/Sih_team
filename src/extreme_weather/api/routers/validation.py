"""Validation and evaluation endpoints (SIH26078, Phase 7).

```text
GET /api/v1/validation/status       # what has been run, and whether it completed
GET /api/v1/validation/summary      # the unified Phase 7 report
GET /api/v1/validation/metrics      # the per-stage numeric reports
GET /api/v1/validation/provenance   # what produced the numbers
```

What this router will not do
----------------------------
It returns **404** when a report has not been produced, rather than a 200 with
empty fields. A client that showed "POD: -" would be indistinguishable from one
showing "POD: 0.00", and the second is a claim the project cannot support.

Every payload carries ``ground_truth_available`` and the limitations list, so
no consumer can display a descriptive count without the caveat that says it is
not accuracy.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status as http_status
from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.config import Settings, get_settings
from extreme_weather.validation.dataset_builder import ValidationDatasetBuilder
from extreme_weather.validation.ground_truth import GROUND_TRUTH_UNAVAILABLE
from extreme_weather.validation.reports import (
    REPORT_FILENAMES,
    read_report,
)

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])

#: The per-stage reports the metrics endpoint can return.
_METRIC_REPORTS: tuple[str, ...] = (
    "threshold_calibration",
    "climatology",
    "anomaly",
    "tracking",
    "gnn",
    "downscaling",
    "alerts",
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ValidationStatusResponse(BaseModel):
    """Whether a Phase 7 run exists, and which stages it completed."""

    model_config = ConfigDict(extra="allow")

    has_run: bool
    reports_present: list[str] = Field(default_factory=list)
    reports_missing: list[str] = Field(default_factory=list)
    complete: bool = False
    ground_truth_available: bool = False
    ground_truth_note: str = GROUND_TRUTH_UNAVAILABLE
    message: str = ""


class ValidationSummaryResponse(BaseModel):
    """The unified Phase 7 report."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(alias="schema")
    project: str
    service: str
    package_version: str
    generated_at: str
    ground_truth_availability: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


class ValidationMetricsResponse(BaseModel):
    """Per-stage metrics, each marked available or explicitly not produced."""

    model_config = ConfigDict(extra="allow")

    ground_truth_available: bool = False
    ground_truth_note: str = GROUND_TRUTH_UNAVAILABLE
    reports: dict[str, Any] = Field(default_factory=dict)
    reports_missing: list[str] = Field(default_factory=list)


class ValidationProvenanceResponse(BaseModel):
    """The provenance record of the run that produced the reports."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    package_version: str
    started_at: str
    finished_at: str | None = None
    complete: bool = False
    dataset: dict[str, Any] = Field(default_factory=dict)
    split: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    stages: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def get_reports_dir(settings: Settings = Depends(get_settings)) -> Any:
    """Dependency returning the reports directory."""
    return ValidationDatasetBuilder(settings).reports_dir


def get_provenance_dir(settings: Settings = Depends(get_settings)) -> Any:
    """Dependency returning the provenance directory."""
    return ValidationDatasetBuilder(settings).provenance_dir


ReportsDependency = Any
ProvenanceDependency = Any


# --------------------------------------------------------------------------- #
# GET /api/v1/validation/status
# --------------------------------------------------------------------------- #
@router.get(
    "/status",
    response_model=ValidationStatusResponse,
    summary="Whether a Phase 7 validation run exists",
)
def validation_status(reports_dir: Any = Depends(get_reports_dir)) -> ValidationStatusResponse:
    """Report which stage reports exist, without pretending any are valid.

    This endpoint always answers 200: the absence of a run is information, not
    an error. It is the per-report endpoints that return 404.
    """
    present = [key for key in _METRIC_REPORTS if read_report(reports_dir, key) is not None]
    missing = [key for key in _METRIC_REPORTS if key not in present]
    summary = read_report(reports_dir, "summary")
    truth = (
        dict(summary.get("ground_truth_availability", {}))
        if summary
        else {}
    )
    return ValidationStatusResponse(
        has_run=summary is not None,
        reports_present=present,
        reports_missing=missing,
        complete=bool(summary and not summary.get("reports_missing")),
        ground_truth_available=bool(truth.get("available", False)),
        message=(
            "Phase 7 validation reports exist."
            if summary
            else (
                "No Phase 7 validation run has been produced yet. Run "
                "scripts/run_phase7_validation.py."
            )
        ),
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/validation/summary
# --------------------------------------------------------------------------- #
@router.get(
    "/summary",
    response_model=ValidationSummaryResponse,
    summary="The unified Phase 7 validation report",
)
def validation_summary(reports_dir: Any = Depends(get_reports_dir)) -> ValidationSummaryResponse:
    """Return the unified Phase 7 report.

    Raises:
        HTTPException: 404 when no run has been produced.
    """
    summary = read_report(reports_dir, "summary")
    if summary is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                "No Phase 7 validation report exists. Run "
                "'python scripts/run_phase7_validation.py' first."
            ),
        )
    return ValidationSummaryResponse(**summary)


# --------------------------------------------------------------------------- #
# GET /api/v1/validation/metrics
# --------------------------------------------------------------------------- #
@router.get(
    "/metrics",
    response_model=ValidationMetricsResponse,
    summary="Per-stage Phase 7 metrics",
)
def validation_metrics(reports_dir: Any = Depends(get_reports_dir)) -> ValidationMetricsResponse:
    """Return every per-stage report that exists.

    Each entry is marked ``available`` so a client can tell "the GNN was worse"
    from "the GNN was never run".

    Raises:
        HTTPException: 404 when no per-stage report exists at all.
    """
    reports: dict[str, Any] = {}
    missing: list[str] = []
    for key in _METRIC_REPORTS:
        payload = read_report(reports_dir, key)
        if payload is None:
            missing.append(key)
            reports[key] = {
                "available": False,
                "reason": f"'{REPORT_FILENAMES[key]}' has not been produced yet.",
            }
        else:
            reports[key] = {"available": True, **payload}

    if not reports or all(not entry["available"] for entry in reports.values()):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                "No Phase 7 metric reports exist. Run "
                "'python scripts/run_phase7_validation.py' first."
            ),
        )

    truth_available = bool(
        (reports.get("anomaly", {}) or {})
        .get("detection_skill", {})
        .get("available", False)
    )
    return ValidationMetricsResponse(
        ground_truth_available=truth_available,
        reports=reports,
        reports_missing=missing,
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/validation/provenance
# --------------------------------------------------------------------------- #
@router.get(
    "/provenance",
    response_model=ValidationProvenanceResponse,
    summary="Provenance of the run that produced the reports",
)
def validation_provenance(
    provenance_dir: Any = Depends(get_provenance_dir),
) -> ValidationProvenanceResponse:
    """Return the most recent run record.

    Raises:
        HTTPException: 404 when no run record exists.
    """
    from extreme_weather.validation.provenance import read_run_record

    if not provenance_dir.is_dir():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"No Phase 7 provenance directory at '{provenance_dir}'. Run "
                "'python scripts/run_phase7_validation.py' first."
            ),
        )
    records = sorted(provenance_dir.glob("*.json"))
    if not records:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"No Phase 7 run records in '{provenance_dir}'. Run "
                "'python scripts/run_phase7_validation.py' first."
            ),
        )
    record = read_run_record(records[-1])
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"The most recent run record '{records[-1].name}' could not be read.",
        )
    return ValidationProvenanceResponse(**record.as_dict())

