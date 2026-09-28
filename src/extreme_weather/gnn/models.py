"""Pydantic schemas for the GNN service (SIH26078, Phase 3).

These are the *external* contract of the Phase 3 API. They are defined once here
and reused by the router, the service and the tests, mirroring how Phase 1 and
Phase 2 define their responses in :mod:`extreme_weather.schemas.event` and
:mod:`extreme_weather.schemas.track`.

Every response model carries the prototype notice: an endpoint that returns a
prediction must make its own non-operational status visible in the payload, not
only in the surrounding documentation.

Semantics
---------
* ``current_*`` describe the last observed state of the track; they are copied
  from the Phase 2 :class:`~extreme_weather.schemas.track.TrackPoint`, not
  predicted.
* ``delta_*`` are the model output, in degrees and in the physical unit of the
  variable (mm h-1, m s-1, K).
* ``predicted_*`` are ``current_* + delta_*`` decoded back into physical units
  and clamped to valid geographic ranges.
* ``model_version`` identifies the checkpoint that produced the prediction.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GNNModelInfo",
    "GNNModelStatus",
    "GNNPrediction",
    "GNNPredictionRequest",
    "PROTOTYPE_NOTICE",
]

#: Shown by the API and the dashboard wherever a prediction is displayed.
PROTOTYPE_NOTICE = (
    "Research Prototype - Not an Operational Forecast. "
    "Predictions come from a model trained on synthetic data and are not "
    "validated against observed weather."
)


class GNNModelStatus(BaseModel):
    """Availability and provenance of the trained model (``GET /gnn/status``)."""

    model_config = ConfigDict(extra="forbid")

    available: bool = Field(
        description="True when a valid checkpoint could be loaded and a model built."
    )
    backend: str = Field(description="Tensor backend used for inference.")
    model_version: str | None = Field(
        default=None, description="Architecture version of the checkpoint, when available."
    )
    checkpoint_path: str | None = Field(
        default=None, description="Location of the checkpoint, when available."
    )
    training: dict = Field(
        default_factory=dict,
        description="Training metadata from the checkpoint (seed, epochs, counts).",
    )
    train_loss: float | None = Field(default=None, description="Final training loss.")
    validation_loss: float | None = Field(
        default=None, description="Lowest validation loss reached during training."
    )
    test_loss: float | None = Field(
        default=None, description="Held-out test loss, evaluated once after training."
    )
    best_epoch: int | None = Field(
        default=None, description="Epoch of the best validation loss."
    )
    message: str | None = Field(
        default=None, description="Why the model is unavailable, or extra context."
    )
    notice: str = Field(default=PROTOTYPE_NOTICE, description="Prototype disclaimer.")


class GNNModelInfo(BaseModel):
    """Architecture and feature contract (``GET /gnn/model``)."""

    model_config = ConfigDict(extra="forbid")

    architecture: str = Field(description="Human-readable architecture name.")
    backend: str = Field(description="Tensor backend used for inference.")
    model_version: str = Field(description="Architecture version.")
    message_layers: int = Field(ge=1, description="Number of message-passing layers.")
    hidden_dim: int = Field(ge=1, description="Width of the internal representation.")
    aggregation: str = Field(description="Neighbour aggregation used in message passing.")
    node_feature_names: list[str] = Field(description="Ordered node feature names.")
    edge_feature_names: list[str] = Field(description="Ordered edge feature names.")
    target_names: list[str] = Field(description="Ordered regression target names.")
    normalization: dict = Field(
        description="Scaling divisors applied to the raw physical inputs."
    )
    hyperparameters: dict = Field(
        description="Model configuration and optimisation settings."
    )
    notice: str = Field(default=PROTOTYPE_NOTICE, description="Prototype disclaimer.")


class GNNPredictionRequest(BaseModel):
    """Request body of ``POST /gnn/predict``."""

    model_config = ConfigDict(extra="forbid")

    track_id: str = Field(
        min_length=1,
        max_length=200,
        description="Deterministic Phase 2 track id, e.g. TRK-EXTREME_RAINFALL-20240602T0200-001.",
    )


class GNNPrediction(BaseModel):
    """One next-state prediction (``POST /gnn/predict``).

    The model predicts *changes*; ``predicted_*`` is the decoded next state
    (``current_* + delta_*``), clamped to valid geographic ranges.
    """

    model_config = ConfigDict(extra="forbid")

    track_id: str = Field(min_length=1)
    timestamp: datetime = Field(description="Valid time of the last observed point.")
    event_type: str | None = Field(default=None, description="Event type of the track.")
    current_latitude: float = Field(ge=-90.0, le=90.0)
    current_longitude: float = Field(ge=-180.0, le=180.0)
    current_intensity: float
    predicted_latitude: float = Field(ge=-90.0, le=90.0)
    predicted_longitude: float = Field(ge=-180.0, le=180.0)
    predicted_intensity: float
    delta_latitude: float = Field(description="Predicted change in latitude, in degrees.")
    delta_longitude: float = Field(description="Predicted change in longitude, in degrees.")
    delta_intensity: float = Field(
        description="Predicted change in intensity, in the variable's physical unit."
    )
    predicted_speed_kmh: float | None = Field(
        default=None, ge=0.0, description="Haversine speed implied by the predicted move."
    )
    predicted_direction: str | None = Field(
        default=None, description="16-point compass direction of the predicted move."
    )
    model_version: str = Field(
        description="Checkpoint version that produced this prediction."
    )
    node_index: int | None = Field(
        default=None, ge=0, description="Index of the node used for the prediction."
    )
    notice: str = Field(default=PROTOTYPE_NOTICE, description="Prototype disclaimer.")
