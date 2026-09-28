"""GNN inference and the service that ties it together (SIH26078, Phase 3).

Inference answers one question: *given the last observed state of a Phase 2
track, what does the trained model predict for the next one?*

Decoding
--------
The network regresses deltas, so the next state is reconstructed as

```text
next_latitude  = current_latitude  + delta_latitude
next_longitude = current_longitude + delta_longitude
next_intensity = current_intensity + delta_intensity
```

The decoded position is clamped to ``[-90, 90] x [-180, 180]``. Clamping is a
guard, not a correction: a prototype model can emit an out-of-range delta, and a
lat/lon of 213 deg is not a forecast, it is a bug. The raw delta is still
reported unchanged so the caller can see what the model actually produced.

Error handling
--------------
Each failure mode raises a distinct, catchable exception so the API can map it
to the right status code:

* :class:`GNNModelUnavailableError` - no checkpoint (or an unreadable one) -> 503
* :class:`GNNUnknownTrackError` - the track id is not in ``tracks.json`` -> 404
* :class:`GNNInvalidTrackError` - the track cannot yield a prediction -> 422
* :class:`GNNInferenceError` - the model ran but produced nothing usable -> 500

Nothing here fabricates a prediction: without a checkpoint or a track, the answer
is an error, never a guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from extreme_weather.gnn.checkpoint import (
    CheckpointError,
    GNNCheckpoint,
    load_checkpoint,
)
from extreme_weather.gnn.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES
from extreme_weather.gnn.graph import GraphConfig, build_graph
from extreme_weather.gnn.model import MODEL_VERSION
from extreme_weather.gnn.models import (
    GNNModelInfo,
    GNNModelStatus,
    GNNPrediction,
    PROTOTYPE_NOTICE,
)
from extreme_weather.gnn.training import predict_all
from extreme_weather.schemas.track import EventTrack
from extreme_weather.tracking.metrics import (
    bearing_degrees,
    compass_direction,
    haversine_distance_km,
)
from extreme_weather.tracking.store import TrackStore, TrackingStoreError

__all__ = [
    "GNNInferenceError",
    "GNNInvalidTrackError",
    "GNNModelUnavailableError",
    "GNNService",
    "GNNUnknownTrackError",
    "decode_prediction",
]


class GNNModelUnavailableError(RuntimeError):
    """No usable checkpoint exists (maps to HTTP 503)."""


class GNNUnknownTrackError(LookupError):
    """The requested track id is not present in the tracking file (HTTP 404)."""


class GNNInvalidTrackError(ValueError):
    """The track exists but cannot produce a prediction (HTTP 422)."""


class GNNInferenceError(RuntimeError):
    """The model ran but produced an unusable result (HTTP 500)."""


def _backend_name() -> str:
    """Name of the tensor backend, reported in the status/model payloads."""
    import torch

    return f"pytorch {torch.__version__} (cpu)"


def decode_prediction(
    current_latitude: float,
    current_longitude: float,
    current_intensity: float,
    deltas: list[float] | tuple[float, float, float],
    step_hours: float = 1.0,
) -> dict[str, float | str | None]:
    """Turn predicted deltas into a decoded next state.

    Args:
        current_latitude: Observed latitude of the last point.
        current_longitude: Observed longitude of the last point.
        current_intensity: Observed intensity of the last point.
        deltas: ``[Δlat, Δlon, Δintensity]`` as predicted.
        step_hours: Time step the prediction covers, used for the implied speed.

    Returns:
        A dict with the clamped ``predicted_*`` values, the unclamped ``delta_*``
        values, the implied ``predicted_speed_kmh`` and ``predicted_direction``.

    Raises:
        GNNInvalidTrackError: If fewer than three deltas are supplied.
    """
    if len(deltas) < 3:
        raise GNNInvalidTrackError(
            f"Expected 3 deltas (latitude, longitude, intensity); got {len(deltas)}."
        )
    delta_lat, delta_lon, delta_intensity = (
        float(deltas[0]),
        float(deltas[1]),
        float(deltas[2]),
    )

    predicted_latitude = max(-90.0, min(90.0, current_latitude + delta_lat))
    predicted_longitude = max(-180.0, min(180.0, current_longitude + delta_lon))

    distance = haversine_distance_km(
        current_latitude, current_longitude, predicted_latitude, predicted_longitude
    )
    hours = step_hours if step_hours > 0.0 else 1.0
    bearing = bearing_degrees(
        current_latitude, current_longitude, predicted_latitude, predicted_longitude
    )

    return {
        "predicted_latitude": predicted_latitude,
        "predicted_longitude": predicted_longitude,
        "predicted_intensity": current_intensity + delta_intensity,
        "delta_latitude": delta_lat,
        "delta_longitude": delta_lon,
        "delta_intensity": delta_intensity,
        "predicted_speed_kmh": distance / hours,
        "predicted_direction": compass_direction(bearing),
    }


class GNNService:
    """Loads the checkpoint and the tracks, then produces predictions.

    Args:
        checkpoint_path: Checkpoint location; defaults to ``models/gnn/best.json``.
        track_store: Source of the Phase 2 tracks; defaults to the configured
            :class:`~extreme_weather.tracking.store.TrackStore`.
        graph_config: Graph settings used when building the inference graph. It
            must be compatible with the training graph; the node/edge feature
            contract is verified by the checkpoint loader regardless.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        track_store: TrackStore | None = None,
        graph_config: GraphConfig | None = None,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._store = track_store
        self.graph_config = graph_config or GraphConfig()
        self._checkpoint: GNNCheckpoint | None = None

    @property
    def store(self) -> TrackStore:
        """The track store, created on first use."""
        if self._store is None:
            self._store = TrackStore()
        return self._store

    def checkpoint(self) -> GNNCheckpoint:
        """Load and cache the checkpoint.

        Raises:
            GNNModelUnavailableError: If no valid checkpoint exists.
        """
        if self._checkpoint is None:
            try:
                self._checkpoint = load_checkpoint(self._checkpoint_path)
            except CheckpointError as exc:
                raise GNNModelUnavailableError(str(exc)) from exc
        return self._checkpoint

    def has_model(self) -> bool:
        """True when a checkpoint can be loaded and a model built from it."""
        try:
            self.checkpoint().build_model()
        except (GNNModelUnavailableError, CheckpointError):
            return False
        return True

    def status(self) -> GNNModelStatus:
        """Availability, version and training provenance of the model.

        Never raises: a missing model is a *status*, not an error, so the
        dashboard can render "not trained yet" instead of a failure.
        """
        try:
            checkpoint = self.checkpoint()
        except GNNModelUnavailableError as exc:
            return GNNModelStatus(
                available=False,
                backend=_backend_name(),
                model_version=None,
                checkpoint_path=str(self._checkpoint_path) if self._checkpoint_path else None,
                training={},
                message=str(exc),
            )

        training = dict(checkpoint.training)
        return GNNModelStatus(
            available=True,
            backend=_backend_name(),
            model_version=checkpoint.model_version,
            checkpoint_path=str(checkpoint.path) if checkpoint.path else None,
            training=training,
            train_loss=checkpoint.train_loss,
            validation_loss=checkpoint.best_validation_loss,
            test_loss=training.get("test_loss"),
            best_epoch=training.get("best_epoch"),
        )

    def model_info(self) -> GNNModelInfo:
        """Architecture, feature contract and hyper-parameters.

        The model configuration is read from the checkpoint when one exists, so
        the description always matches the weights that would be used. Without a
        checkpoint the default configuration is described and the caller is told
        the model is untrained via :meth:`status`.
        """
        try:
            checkpoint = self.checkpoint()
            config = checkpoint.model_config
            normalization = {
                "intensity_scale": checkpoint.feature_config.intensity_scale,
                "speed_scale": checkpoint.feature_config.speed_scale,
                "cell_count_scale": checkpoint.feature_config.cell_count_scale,
                "edge_distance_scale": checkpoint.feature_config.edge_distance_scale,
                "edge_time_scale": checkpoint.feature_config.edge_time_scale,
                "method": "fixed divisors (not fitted statistics)",
            }
            hyperparameters: dict[str, Any] = dict(config.as_dict())
            hyperparameters["available"] = True
            model_version = checkpoint.model_version
        except GNNModelUnavailableError:
            from extreme_weather.gnn.model import ModelConfig

            config = ModelConfig()
            normalization = {
                "intensity_scale": self.graph_config.feature_config.intensity_scale,
                "speed_scale": self.graph_config.feature_config.speed_scale,
                "cell_count_scale": self.graph_config.feature_config.cell_count_scale,
                "edge_distance_scale": self.graph_config.feature_config.edge_distance_scale,
                "edge_time_scale": self.graph_config.feature_config.edge_time_scale,
                "method": "fixed divisors (not fitted statistics)",
            }
            hyperparameters = dict(config.as_dict())
            hyperparameters["available"] = False
            model_version = MODEL_VERSION

        return GNNModelInfo(
            architecture=(
                "input projection -> "
                f"{config.message_layers} x (mean-aggregation message passing) -> MLP head"
            ),
            backend=_backend_name(),
            model_version=model_version,
            message_layers=config.message_layers,
            hidden_dim=config.hidden_dim,
            aggregation="mean over incoming edges (isolated nodes keep their own state)",
            node_feature_names=list(NODE_FEATURE_NAMES),
            edge_feature_names=list(EDGE_FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            normalization=normalization,
            hyperparameters=hyperparameters,
        )

    def _load_tracks(self) -> list[EventTrack]:
        """Read the Phase 2 tracks, mapping store failures to a service error.

        Raises:
            GNNInvalidTrackError: If the tracking file is missing or corrupt.
        """
        try:
            result = self.store.load()
        except TrackingStoreError as exc:
            raise GNNInvalidTrackError(f"The tracking file could not be read: {exc}") from exc
        if result is None:
            raise GNNInvalidTrackError(
                f"No tracking file at '{self.store.path}'. Run 'python scripts/track_events.py' "
                "before predicting."
            )
        return list(result.tracks)

    def _find_track(self, tracks: list[EventTrack], track_id: str) -> EventTrack:
        """Locate one track by id.

        Raises:
            GNNUnknownTrackError: If the id is not in the tracking file.
        """
        for track in tracks:
            if track.track_id == track_id:
                return track
        raise GNNUnknownTrackError(
            f"Track '{track_id}' not found. The tracking file contains {len(tracks)} track(s)."
        )

    def _last_node_of(self, batch: Any, track_id: str) -> int | None:
        """Node index of the last point of ``track_id`` (``None`` when absent)."""
        found: int | None = None
        for index, node_track_id in enumerate(batch.node_track_ids):
            if node_track_id == track_id:
                found = index
        return found

    @staticmethod
    def _typical_step_hours(track: EventTrack) -> float:
        """Median spacing between the track's points, in hours (``1.0`` default)."""
        if len(track.points) < 2:
            return 1.0
        gaps = sorted(
            (later.timestamp - earlier.timestamp).total_seconds() / 3600.0
            for earlier, later in zip(track.points, track.points[1:])
            if (later.timestamp - earlier.timestamp).total_seconds() > 0.0
        )
        return gaps[len(gaps) // 2] if gaps else 1.0

    def predict(self, track_id: str) -> GNNPrediction:
        """Predict the next state of ``track_id``.

        The graph is rebuilt from the *stored* tracks and the model runs over the
        whole graph, so the node inherits its real neighbours: the previous
        trajectory step and any concurrent nearby event. Only the **last** point
        of the track is predicted, because that is the only point whose future is
        genuinely unknown - earlier points already have observations, so
        predicting them would score the model against data it was shown.

        Args:
            track_id: Deterministic Phase 2 track id.

        Returns:
            A validated :class:`~extreme_weather.gnn.models.GNNPrediction`.

        Raises:
            GNNModelUnavailableError: If no valid checkpoint exists.
            GNNUnknownTrackError: If the track id is unknown.
            GNNInvalidTrackError: If there is no usable tracking file, or the
                track cannot yield features.
            GNNInferenceError: If the model ran but produced nothing usable.
        """
        if not isinstance(track_id, str) or not track_id.strip():
            raise GNNInvalidTrackError("track_id must be a non-empty string.")

        checkpoint = self.checkpoint()  # raises GNNModelUnavailableError
        tracks = self._load_tracks()
        track = self._find_track(tracks, track_id)

        if not track.points:
            raise GNNInvalidTrackError(f"Track '{track_id}' contains no points.")

        batch = build_graph(tracks, self.graph_config)
        if batch.node_count == 0:
            raise GNNInferenceError("The inference graph is empty.")

        node_index = self._last_node_of(batch, track_id)
        if node_index is None:
            raise GNNInferenceError(f"Track '{track_id}' is not present in the inference graph.")

        model = checkpoint.build_model()
        try:
            predictions = predict_all(model, batch)
        except Exception as exc:  # any torch failure means "the model could not run"
            raise GNNInferenceError(f"The model could not be evaluated: {exc}") from exc

        last_point = track.points[-1]
        decoded = decode_prediction(
            current_latitude=float(last_point.latitude),
            current_longitude=float(last_point.longitude),
            current_intensity=float(last_point.intensity),
            deltas=predictions[node_index].tolist(),
            step_hours=self._typical_step_hours(track),
        )

        return GNNPrediction(
            track_id=track.track_id,
            timestamp=last_point.timestamp,
            event_type=track.event_type.value,
            current_latitude=float(last_point.latitude),
            current_longitude=float(last_point.longitude),
            current_intensity=float(last_point.intensity),
            predicted_latitude=float(decoded["predicted_latitude"]),
            predicted_longitude=float(decoded["predicted_longitude"]),
            predicted_intensity=float(decoded["predicted_intensity"]),
            delta_latitude=float(decoded["delta_latitude"]),
            delta_longitude=float(decoded["delta_longitude"]),
            delta_intensity=float(decoded["delta_intensity"]),
            predicted_speed_kmh=float(decoded["predicted_speed_kmh"]),
            predicted_direction=str(decoded["predicted_direction"]),
            model_version=checkpoint.model_version,
            node_index=node_index,
        )
