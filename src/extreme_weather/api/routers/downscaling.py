"""AI downscaling endpoints (SIH26078, Phase 4).

```text
GET  /api/v1/downscaling/status      # model availability + provenance
GET  /api/v1/downscaling/model       # architecture, features, normalisation
POST /api/v1/downscaling/run         # downscale one frame with one method
POST /api/v1/downscaling/evaluate    # nearest vs cubic vs ai on the SAME test set
```

No endpoint returns a raw array: a downscaled field is tens of thousands of
values, so responses carry metadata, statistics and file references.

Status codes for ``/run``: 200 completed, 404 no processed dataset, 422 invalid
variable/resolution/method or malformed data, 503 ``method="ai"`` without a
trained checkpoint. ``/status`` and ``/model`` never fail - a missing model is
reported as ``available: false`` so a dashboard renders "not trained yet".
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status as http_status

from extreme_weather.downscaling.api_models import (
    DownscalingEvaluationResponse,
    DownscalingMetricsResponse,
    DownscalingModelInfo,
    DownscalingModelStatus,
    DownscalingRequest,
    DownscalingResponse,
    EvaluationRequest,
    ExtremeEventMetricsResponse,
)
from extreme_weather.downscaling.dataset import (
    SplitConfig,
    build_downscaling_dataset,
)
from extreme_weather.downscaling.metrics import DEFAULT_WET_THRESHOLD
from extreme_weather.downscaling.reference import REFERENCE_DISCLAIMER
from extreme_weather.downscaling.service import (
    DownscalingDataError,
    DownscalingModelUnavailableError,
    DownscalingResolutionError,
    DownscalingService,
    DownscalingVariableError,
)
from extreme_weather.tracking.metrics import EARTH_RADIUS_KM

router = APIRouter(prefix="/api/v1/downscaling", tags=["downscaling"])

VALID_METHODS = ("nearest", "cubic", "ai")


def get_downscaling_service() -> DownscalingService:
    """Dependency providing the downscaling service with default paths."""
    return DownscalingService()


ServiceDependency = Annotated[DownscalingService, Depends(get_downscaling_service)]


def _degrees_to_km(degrees: float) -> float:
    """Approximate a latitude spacing in km, for human-readable reporting only.

    The degrees value stays authoritative - a longitude degree shrinks with
    latitude - so this is a label, not a measurement.
    """
    return degrees * math.pi / 180.0 * EARTH_RADIUS_KM


def _http_error(exc: Exception) -> HTTPException:
    """Map a service exception to the right HTTP status."""
    if isinstance(exc, DownscalingModelUnavailableError):
        return HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    if isinstance(exc, DownscalingDataError):
        code = (
            http_status.HTTP_404_NOT_FOUND
            if "No dataset" in str(exc)
            else http_status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        return HTTPException(status_code=code, detail=str(exc))
    return HTTPException(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
    )


@router.get(
    "/status",
    response_model=DownscalingModelStatus,
    summary="AI downscaling model availability and provenance",
)
def get_status(service: ServiceDependency) -> DownscalingModelStatus:
    """Report whether a trained checkpoint is available, and its provenance."""
    payload: dict[str, Any] = service.status()
    payload.setdefault("methods", list(VALID_METHODS))
    return DownscalingModelStatus(**payload)


@router.get(
    "/model",
    response_model=DownscalingModelInfo,
    summary="AI downscaling architecture and feature contract",
)
def get_model(service: ServiceDependency) -> DownscalingModelInfo:
    """Describe the architecture, the input features and the normalisation."""
    info = service.model_info()
    info["reference_disclaimer"] = REFERENCE_DISCLAIMER
    return DownscalingModelInfo(**info)


@router.post(
    "/run",
    response_model=DownscalingResponse,
    summary="Downscale one frame with nearest, cubic or ai",
)
def run(request: DownscalingRequest, service: ServiceDependency) -> DownscalingResponse:
    """Downscale a single time frame and return its metadata and statistics."""
    from extreme_weather.config import get_settings

    settings = get_settings()
    resolution = request.target_resolution or settings.downsample_target_resolution_deg
    if request.method not in VALID_METHODS:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown method '{request.method}'. Use one of {list(VALID_METHODS)}.",
        )
    try:
        coarse, latitude, longitude, _timestamps = service.load_coarse()
        if request.time_index >= coarse.shape[0]:
            raise DownscalingDataError(
                f"time_index {request.time_index} is out of range; the dataset has "
                f"{coarse.shape[0]} frames."
            )
        _, result = service.run_field(
            request.method,
            coarse[request.time_index],
            latitude,
            longitude,
            resolution,
            variable=request.variable,
        )
    except (
        DownscalingDataError,
        DownscalingResolutionError,
        DownscalingVariableError,
        DownscalingModelUnavailableError,
    ) as exc:
        raise _http_error(exc) from exc

    payload = result.as_dict()
    return DownscalingResponse(
        method=payload["method"],
        variable=payload["variable"],
        input_resolution_deg=payload["input_resolution_deg"],
        output_resolution_deg=payload["output_resolution_deg"],
        input_resolution_km=_degrees_to_km(payload["input_resolution_deg"]),
        output_resolution_km=_degrees_to_km(payload["output_resolution_deg"]),
        scale_factor=payload["input_resolution_deg"] / payload["output_resolution_deg"],
        input_shape=payload["input_shape"],
        output_shape=payload["output_shape"],
        statistics=payload["statistics"],
        model_version=payload["model_version"],
        notice=payload["notice"],
    )


@router.post(
    "/evaluate",
    response_model=DownscalingEvaluationResponse,
    summary="Compare nearest, cubic and ai on the same test samples",
)
def evaluate_downscales(
    request: EvaluationRequest, service: ServiceDependency
) -> DownscalingEvaluationResponse:
    """Score every requested method on identical test frames.

    The metrics are agreement with a **synthetic** reference, not forecast
    verification. No ranking claim is made beyond the measured MAE order.
    """
    from extreme_weather.config import get_settings

    settings = get_settings()
    threshold = (
        DEFAULT_WET_THRESHOLD if request.wet_threshold is None else request.wet_threshold
    )
    unknown = [m for m in request.methods if m not in VALID_METHODS]
    if unknown:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown method(s) {unknown}. Use one of {list(VALID_METHODS)}.",
        )

    try:
        coarse, latitude, longitude, timestamps = service.load_coarse()
        dataset = build_downscaling_dataset(
            coarse,
            latitude,
            longitude,
            timestamps,
            settings.downsample_target_resolution_deg,
            variable="precipitation",
            split_config=SplitConfig(),
            time_subset=slice(0, request.max_frames) if request.max_frames else None,
        )
        report = service.compare(dataset, methods=tuple(request.methods), threshold=threshold)
    except (
        DownscalingDataError,
        DownscalingResolutionError,
        DownscalingVariableError,
        DownscalingModelUnavailableError,
    ) as exc:
        raise _http_error(exc) from exc

    rows = [DownscalingMetricsResponse(**row.as_dict()) for row in report.metrics]
    event_rows = [
        ExtremeEventMetricsResponse(**row.as_dict()) for row in report.extreme_event
    ]
    # Reported exactly as measured, ordered by MAE; not an endorsement.
    ranking = [row.method for row in sorted(report.metrics, key=lambda r: r.error.mae)]

    return DownscalingEvaluationResponse(
        variable=report.variable,
        test_sample_count=report.test_sample_count,
        input_resolution_deg=dataset.input_resolution_deg,
        output_resolution_deg=dataset.output_resolution_deg,
        methods=rows,
        extreme_event=event_rows,
        ranking=ranking,
        notice=report.notice,
    )
