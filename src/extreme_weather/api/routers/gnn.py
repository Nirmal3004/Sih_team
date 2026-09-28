"""GNN event-evolution endpoints (SIH26078, Phase 3).

These endpoints expose **real model output**: predictions come from the
checkpoint written by ``scripts/train_gnn.py`` and applied to the tracks written
by ``scripts/track_events.py``. Nothing is fabricated.

```text
GET  /api/v1/gnn/status      # availability, version, losses, training metadata
GET  /api/v1/gnn/model       # architecture, features, targets, hyper-parameters
POST /api/v1/gnn/predict     # next-state prediction for one track
```

Status codes for ``/predict``:

======  ==========================================================
200     Prediction produced (always a *prototype* prediction).
404     The track id is not in ``tracks.json``.
422     The request body is invalid, or no usable tracking file exists.
503     No valid checkpoint - the model has not been trained yet.
======  ==========================================================

``/status`` never fails: a missing model is a *status* (``available: false``)
rather than an error, so a dashboard can render "not trained yet" cleanly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status as http_status

from extreme_weather.gnn.models import (
    GNNModelInfo,
    GNNModelStatus,
    GNNPrediction,
    GNNPredictionRequest,
)
from extreme_weather.gnn.service import (
    GNNInferenceError,
    GNNInvalidTrackError,
    GNNModelUnavailableError,
    GNNService,
    GNNUnknownTrackError,
)

router = APIRouter(prefix="/api/v1/gnn", tags=["gnn"])

MISSING_MODEL_MESSAGE = (
    "No GNN model available. Run 'python scripts/build_graph_dataset.py' and then "
    "'python scripts/train_gnn.py'."
)


def get_gnn_service() -> GNNService:
    """Dependency providing the GNN service with default paths."""
    return GNNService()


ServiceDependency = Annotated[GNNService, Depends(get_gnn_service)]


@router.get(
    "/status",
    response_model=GNNModelStatus,
    summary="GNN model availability and training provenance",
)
def get_status(service: ServiceDependency) -> GNNModelStatus:
    """Report whether a trained checkpoint is available, and its provenance."""
    return service.status()


@router.get(
    "/model",
    response_model=GNNModelInfo,
    summary="GNN architecture, feature contract and hyper-parameters",
)
def get_model(service: ServiceDependency) -> GNNModelInfo:
    """Describe the architecture and the exact feature/target contract."""
    return service.model_info()


@router.post(
    "/predict",
    response_model=GNNPrediction,
    summary="Predict the next state of a tracked event (prototype)",
)
def predict(
    request: GNNPredictionRequest,
    service: ServiceDependency,
) -> GNNPrediction:
    """Return the prototype next-state prediction for one track."""
    try:
        return service.predict(request.track_id)
    except GNNUnknownTrackError as exc:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GNNModelUnavailableError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except GNNInvalidTrackError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except GNNInferenceError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
