"""Operational endpoints: readiness, metrics and integrity (SIH26078, Phase 10).

```text
GET /ready                        # readiness with an explicit state
GET /api/v1/operations/status     # the full operational picture
GET /api/v1/operations/metrics    # system counters and histograms
GET /api/v1/operations/integrity  # model and data integrity checks
```

Liveness versus readiness
-------------------------
``GET /health`` stays exactly as it was: it answers "is this process running",
which is what a container restart policy needs. It deliberately does **not**
consult the filesystem or the models, because a liveness probe that fails when a
model is missing will restart a perfectly healthy process in a loop.

``GET /ready`` answers the different question "should this process receive
traffic". It reports one of three states:

``READY``
    Configuration is valid, directories are writable, and every component's
    availability is known.
``DEGRADED``
    Serving, but something optional is missing - most often an untrained model.
    Honest wording matters here: this project has no trained model by default,
    so ``DEGRADED`` is the normal state, not a fault.
``NOT_READY``
    Cannot serve: configuration is invalid, or a required directory is missing
    and unwritable. A container orchestrator should hold traffic back.

A missing model is **never** reported as ``READY``.

System status (Phase 11)
------------------------
``GET /api/v1/system/status`` is the single page a judge or an operator opens
first. It answers "what is this system, what has it actually done, and can it
be trusted right now?" in one request, by composing the services that already
exist rather than adding a new source of truth.

It deliberately does **not** report a number the system cannot measure. Where a
quantity is unavailable - no trained checkpoint, no evaluation on record - the
field is ``null`` with a reason, never ``0``. A dashboard that shows zeros for
"models trained" or "alerts generated" is indistinguishable from a system that
tried and failed, and that is exactly the confusion this project must not
create.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.config import (
    PACKAGE_VERSION,
    PROJECT_ID,
    PROJECT_NAME,
    Settings,
    get_settings,
)
from extreme_weather.observability.metrics import get_registry
from extreme_weather.security import directory_free_bytes, writable_directory

router = APIRouter(prefix="/api/v1/operations", tags=["operations"])

DEGRADED = "DEGRADED"
NOT_READY = "NOT_READY"
READY = "READY"

#: Below this much free space, readiness reports the filesystem as degraded.
#: 256 MiB is enough headroom for the small artifacts this project writes and
#: low enough to catch a full volume before a write fails mid-run.
_MIN_FREE_BYTES = 256 * 1024 * 1024


class ComponentStatus(BaseModel):
    """One component's availability, as observed not as assumed."""

    model_config = ConfigDict(extra="allow")

    name: str
    state: str
    detail: str = ""
    required: bool = False

    @property
    def available(self) -> bool:
        """True when the component is usable.

        ``DEGRADED`` counts as unavailable: a component that cannot serve its
        purpose is not available, whatever else is true of it.
        """
        return self.state == "AVAILABLE"


class ReadinessResponse(BaseModel):
    """Payload returned by ``GET /ready``."""

    model_config = ConfigDict(extra="allow")

    state: str
    ready: bool
    project: str
    service: str
    version: str
    environment: str
    components: list[ComponentStatus] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Research prototype (SIH26078). Not an operational disaster-warning "
        "system. READY means this process can serve requests; it makes no "
        "claim about forecast skill."
    )


class MetricsResponse(BaseModel):
    """System metrics. Contains no model-quality metric, by design."""

    model_config = ConfigDict(extra="allow")

    uptime_seconds: float
    counters: dict[str, Any] = Field(default_factory=dict)
    histograms: dict[str, Any] = Field(default_factory=dict)
    gauges: dict[str, Any] = Field(default_factory=dict)
    note: str = (
        "System counters only: request volume, error counts and durations. "
        "These are not model-quality metrics. No accuracy, precision, recall "
        "or forecast-skill figure is produced or implied here."
    )


def _check_gnn(settings: Settings) -> ComponentStatus:
    """Report GNN availability from its own status, not from a file guess."""
    try:
        from extreme_weather.gnn.service import GNNService

        status = GNNService().status()
        if getattr(status, "available", False):
            return ComponentStatus(
                name="gnn",
                state="AVAILABLE",
                detail=(
                    f"Checkpoint {getattr(status, 'model_version', 'unknown')} loaded."
                ),
                required=False,
            )
        return ComponentStatus(
            name="gnn",
            state="UNAVAILABLE",
            detail=str(getattr(status, "message", "no trained checkpoint")),
            required=False,
        )
    except Exception as exc:  # noqa: BLE001 - availability is a status, not a crash
        return ComponentStatus(
            name="gnn",
            state="UNAVAILABLE",
            detail=f"{type(exc).__name__}: {exc}",
            required=False,
        )


def _check_downscaling(settings: Settings) -> ComponentStatus:
    """Report AI downscaling availability."""
    try:
        from extreme_weather.downscaling.service import DownscalingService

        status = DownscalingService().status()
        if status.get("available"):
            return ComponentStatus(
                name="downscaling",
                state="AVAILABLE",
                detail=f"Checkpoint {status.get('model_version')} loaded.",
                required=False,
            )
        return ComponentStatus(
            name="downscaling",
            state="UNAVAILABLE",
            detail=str(status.get("message", "no trained checkpoint")),
            required=False,
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(
            name="downscaling",
            state="UNAVAILABLE",
            detail=f"{type(exc).__name__}: {exc}",
            required=False,
        )


def _check_directories(settings: Settings) -> tuple[ComponentStatus, dict[str, Any]]:
    """Check the data and model directories are present and writable."""
    details: dict[str, Any] = {}
    problems: list[str] = []
    for label, directory in (
        ("data_dir", settings.data_dir),
        ("model_dir", settings.model_dir),
    ):
        writable = writable_directory(directory)
        free = directory_free_bytes(directory)
        details[label] = {
            "path": str(directory),
            "writable": writable,
            "free_bytes": free,
            "low_on_space": free is not None and free < _MIN_FREE_BYTES,
        }
        if not writable:
            problems.append(f"{label} '{directory}' is missing or not writable.")
    component = ComponentStatus(
        name="directories",
        state="AVAILABLE" if not problems else "UNAVAILABLE",
        detail="; ".join(problems) if problems else "Data and model directories are writable.",
        required=True,
    )
    return component, details


def _build_readiness(settings: Settings) -> ReadinessResponse:
    """Assemble the readiness verdict.

    The state is the *worst* component state, with one deliberate exception:
    an unavailable optional model yields ``DEGRADED``, not ``NOT_READY``,
    because this project ships without trained checkpoints by design and a
    missing model must not make the process unable to serve.
    """
    directory_status, directory_details = _check_directories(settings)
    gnn = _check_gnn(settings)
    downscaling = _check_downscaling(settings)

    components = [directory_status, gnn, downscaling]
    checks = {
        "configuration_valid": True,
        "directories": directory_details,
        "environment": settings.app_env,
        "cors_origin_count": len(settings.cors_origin_list),
    }
    notes: list[str] = []

    if not directory_status.available:
        state = NOT_READY
        notes.append(
            "A required directory is missing or not writable, so this process "
            "cannot serve requests."
        )
    else:
        low = [
            name
            for name, entry in directory_details.items()
            if entry.get("low_on_space")
        ]
        if low:
            state = DEGRADED
            notes.append(
                f"Low free space on {', '.join(low)}. Writes may fail before the "
                "volume is full."
            )
        else:
            state = READY

    missing = [c for c in components if c.state == "UNAVAILABLE" and c.name != "directories"]
    if missing:
        notes.append(
            "No trained checkpoint for: "
            f"{', '.join(c.name for c in missing)}. Those stages report SKIPPED. "
            "This is a prototype limitation, not a fault, and no prediction is "
            "substituted for the missing model."
        )
        if state == READY:
            state = DEGRADED

    return ReadinessResponse(
        state=state,
        ready=state in (READY, DEGRADED),
        project=PROJECT_ID,
        service=PROJECT_NAME,
        version=PACKAGE_VERSION,
        environment=settings.app_env,
        components=components,
        checks=checks,
        notes=notes,
    )


@router.get(
    "/status",
    response_model=ReadinessResponse,
    summary="Full operational status (Phase 10)",
)
def operations_status(settings: Settings = Depends(get_settings)) -> ReadinessResponse:
    """Return the same verdict as ``GET /ready``, under the versioned prefix.

    Provided so an operations dashboard can read one endpoint rather than
    scraping several.
    """
    return _build_readiness(settings)


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics (Phase 10)",
)
def operations_metrics() -> MetricsResponse:
    """Return process-local system metrics.

    These are **system** counters - request volume, errors, durations. They say
    nothing about model quality, and no accuracy or skill figure is derived
    from them.
    """
    snapshot = get_registry().snapshot()
    return MetricsResponse(
        uptime_seconds=float(snapshot.get("uptime_seconds", 0.0)),
        counters=dict(snapshot.get("counters", {})),
        histograms=dict(snapshot.get("histograms", {})),
        gauges=dict(snapshot.get("gauges", {})),
    )


@router.get(
    "/integrity",
    summary="Model and data integrity (Phase 10)",
)
def operations_integrity(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Check every model and data artifact this deployment depends on.

    Each artifact reports ``VALID``, ``INCOMPATIBLE``, ``CORRUPT`` or
    ``MISSING``. A corrupt artifact is reported, never silently used.
    """
    from extreme_weather.integrity import check_checkpoint

    checks: dict[str, Any] = {}
    for label, path in (
        ("gnn_checkpoint", settings.model_dir / "gnn" / "gnn_prototype.pt"),
        ("downscaling_checkpoint", settings.model_dir / "downscaling" / "downscaling_cnn.pt"),
    ):
        result = check_checkpoint(path, model_dir=settings.model_dir)
        checks[label] = result.as_dict()

    return {
        "checks": checks,
        "all_valid": all(entry["usable"] for entry in checks.values()),
        "note": (
            "A MISSING checkpoint is the normal state for this prototype: no "
            "model is shipped trained. It is reported rather than treated as an "
            "error, and the affected stage reports SKIPPED."
        ),
    }
