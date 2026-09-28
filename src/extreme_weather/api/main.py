"""FastAPI application for DeepWeather AI (SIH26078).

Run locally with::

    uvicorn extreme_weather.api.main:app --reload

Phase 0 endpoints
-----------------
``GET  /``                          - service information
``GET  /health``                    - health probe (used by the dashboard)
``GET  /api/v1/datasets``           - list NetCDF datasets in the data directory
``GET  /api/v1/datasets/{p}/summary`` - metadata summary of one dataset
``POST /api/v1/datasets/validate``  - dry-run the data contract on one dataset
``GET  /docs``                      - OpenAPI documentation

Phase 1 endpoints
-----------------
``GET  /``                          - service information
``GET  /health``                    - health probe (used by the dashboard)
``GET  /api/v1/datasets``           - list NetCDF datasets in the data directory
``GET  /api/v1/datasets/{p}/summary`` - metadata summary of one dataset
``POST /api/v1/datasets/validate``  - dry-run the data contract on one dataset
``GET  /api/v1/events``             - detected extreme weather events (Phase 1)
``GET  /api/v1/events/{event_id}``  - one detected event
``GET  /api/v1/tracks``             - tracked event trajectories (Phase 2)
``GET  /api/v1/tracks/{track_id}``  - one tracked event
``GET  /api/v1/tracks/{track_id}/trajectory`` - positions of one track
``GET  /api/v1/gnn/status``            - GNN model availability (Phase 3)
``GET  /api/v1/gnn/model``             - GNN architecture and feature contract
``POST /api/v1/gnn/predict``           - prototype next-state prediction
``GET  /docs``                      - OpenAPI documentation

All imports are absolute (``extreme_weather...``) so the app runs both from an
installed package and from the repository using ``pythonpath=src``. The module
therefore never relies on fragile relative imports.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from extreme_weather.api.routers import (
    alerts,
    datasets,
    downscaling,
    evaluation,
    events,
    gnn,
    health,
    ingestion,
    operations,
    pipeline,
    system,
    tracks,
    validation,
)
from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID, PROJECT_NAME, Settings, get_settings

API_DESCRIPTION = (
    "DeepWeather AI (SIH26078) - AI-based extreme weather intelligence.\n\n"
    "**Phase 0 scope:** synthetic NWP-style data generation, Xarray validation, "
    "missing-value handling, derived wind variables, preprocessing pipeline, "
    "baseline downscaling and dataset metadata endpoints.\n\n"
    "**Phase 1:** anomaly detection, thresholds and event clustering.\n\n"
    "**Phase 2:** event tracking - trajectories, movement metrics and lifecycle.\n\n"
    "**Phase 3 (prototype):** a GNN that predicts the next state of a tracked\n"
    "event. Trained on synthetic data; NOT an operational forecast.\n\n"
    "**Phase 4 (prototype):** a residual CNN that refines a ~12 km field onto a\n"
    "~5 km grid. Higher resolution is not higher accuracy.\n\n"
    "**Phase 5 (prototype):** deterministic alerting and an emergency-response\n"
    "dashboard - risk scoring, alert rules, lifecycle, acknowledge/resolve and\n"
    "optional notification adapters. The alert severities are project prototype\n"
    "categories, NOT official IMD or WMO warnings, and must not be used for\n"
    "emergency decisions.\n\n"
    "**Phase 7:** leakage-free real-data validation - chronological splits,\n"
    "train-only climatology and threshold calibration, baseline comparisons, and\n"
    "every metric that could not be computed reported as unavailable rather than\n"
    "as zero.\n\n"
    "**Phase 8:** near-real-time ingestion and orchestration - a weather-provider\n"
    "abstraction, freshness and data-quality gates, a retrying orchestrator that\n"
    "drives Phases 0-5 with optional models reported as SKIPPED rather than faked,\n"
    "deterministic idempotency, a scheduler, and /api/v1/pipeline endpoints.\n"
    "Adding a near-real-time loop does NOT make this operational: there is no SLA,\n"
    "no failover, and no demonstrated forecast skill.\n\n"
    "**Phase 9:** real-data model evaluation, explainability and uncertainty.\n"
    "Every quantity is reported as MEASURED, DERIVED, UNAVAILABLE or\n"
    "INSUFFICIENT_DATA with the evidence that produced it. Unavailable metrics\n"
    "are never returned as zero, feature evidence is never described as causal,\n"
    "and no model is called better than a baseline unless the measurement says\n"
    "so on that specific metric."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override (used by tests).

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    effective_settings = settings or get_settings()

    application = FastAPI(
        title=f"{PROJECT_NAME} API ({PROJECT_ID})",
        description=API_DESCRIPTION,
        version=PACKAGE_VERSION,
        contact={"name": "DeepWeather AI Team", "url": "https://sih.gov.in/"},
        license_info={"name": "MIT"},
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=effective_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router)
    application.include_router(datasets.router)
    application.include_router(events.router)
    application.include_router(tracks.router)
    application.include_router(gnn.router)
    application.include_router(downscaling.router)
    application.include_router(evaluation.router)
    application.include_router(alerts.router)
    application.include_router(ingestion.router)
    application.include_router(pipeline.router)
    application.include_router(validation.router)
    application.include_router(operations.router)
    # Phase 11: the single "what is this system" page. Registered last so it
    # composes the readiness verdict the operations router owns.
    application.include_router(system.router)

    # Phase 10: request correlation, API metrics and the standard error
    # envelope. Installed last so it wraps every other middleware.
    from extreme_weather.api.middleware import install_middleware
    from extreme_weather.observability.logging import configure_logging

    configure_logging(
        effective_settings.log_level, fmt=effective_settings.log_format, force=True
    )
    install_middleware(application)

    return application


app = create_app()
