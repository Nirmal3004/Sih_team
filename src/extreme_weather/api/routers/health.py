"""Health / readiness endpoints (SIH26078)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID, PROJECT_NAME, Settings, get_settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Payload returned by ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    project: str
    service: str
    version: str
    environment: str


class ServiceInfoResponse(BaseModel):
    """Payload returned by ``GET /``."""

    model_config = ConfigDict(extra="forbid")

    project: str
    service: str
    version: str
    phase: str
    docs_url: str
    health_url: str
    datasets_url: str
    summary: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness/readiness probe",
    description=(
        "Returns the service identity and status. Used by the React dashboard, by "
        "docker-compose health checks and by the acceptance tests."
    ),
)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return the service health payload."""
    return HealthResponse(
        status="ok",
        project=PROJECT_ID,
        service=PROJECT_NAME,
        version=PACKAGE_VERSION,
        environment=settings.app_env,
    )


@router.get(
    "/",
    response_model=ServiceInfoResponse,
    summary="Service information",
    description="Entry point describing the service and its Phase 0 endpoints.",
)
def service_info() -> ServiceInfoResponse:
    """Return a short description of the service and its current endpoints."""
    return ServiceInfoResponse(
        project=PROJECT_ID,
        service=PROJECT_NAME,
        version=PACKAGE_VERSION,
        phase="Phase 0 - synthetic data pipeline, preprocessing, baseline downscaling",
        docs_url="/docs",
        health_url="/health",
        datasets_url="/api/v1/datasets",
        summary=(
            "AI-based extreme weather intelligence: anomaly detection, event tracking and "
            "12 km -> 5 km downscaling. Phase 0 exposes the data foundation only."
        ),
    )


@router.get(
    "/ready",
    summary="Readiness probe (Phase 10)",
    description=(
        "Reports whether this process should receive traffic. Distinct from "
        "`/health`, which answers only 'is the process running'. Returns "
        "READY, DEGRADED or NOT_READY. A missing trained model yields "
        "DEGRADED, never READY."
    ),
)
def ready(settings: Settings = Depends(get_settings)) -> Any:
    """Return the readiness verdict.

    Delegates to the operations router so both endpoints cannot drift apart.
    """
    from extreme_weather.api.routers.operations import _build_readiness

    return _build_readiness(settings)
