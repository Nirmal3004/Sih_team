"""Deriving evaluation samples from real Phase 2 tracks (SIH26078).

Given a ``tracks.json`` produced by Phase 2, this builds the next-state sample
sets the GNN comparison needs. It reuses the Phase 3 graph pipeline -
:func:`~extreme_weather.gnn.dataset.build_dataset` and
:func:`~extreme_weather.gnn.training.predict_all` - rather than
reimplementing graph construction, so a Phase 9 evaluation and a Phase 3
training run see exactly the same features and targets.

Each usable graph node yields one sample:

* ``current_*``  - the node's own state
* ``previous_*`` - the previous node of the same track, for constant velocity
* ``true_*``     - the node's target, the *observed* next state
* ``predicted_*`` - the model's prediction, when a checkpoint ran

Nodes without a target (the last point of every track) are excluded, because
predicting them would score the model against a state it was shown. Nodes whose
model prediction is missing or non-finite are reported, not silently dropped,
so the sample count in the report matches what the evaluator actually saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["TrackSamples", "samples_from_tracks"]

#: Every array a sample set may contain.
SAMPLE_FIELDS: tuple[str, ...] = (
    "current_latitude",
    "current_longitude",
    "current_intensity",
    "previous_latitude",
    "previous_longitude",
    "predicted_latitude",
    "predicted_longitude",
    "predicted_intensity",
    "true_latitude",
    "true_longitude",
    "true_intensity",
)


@dataclass
class TrackSamples:
    """Sample sets drawn from one split of a track graph.

    Attributes:
        validation: Samples for fitting the uncertainty residuals.
        test: Samples for scoring.
        model_available: Whether a checkpoint produced predictions.
        test_node_count: Nodes the split contains.
        usable_test_nodes: Nodes that produced a usable sample.
        excluded_nodes: Why nodes were dropped, by reason.
    """

    validation: dict[str, np.ndarray] = field(default_factory=dict)
    test: dict[str, np.ndarray] = field(default_factory=dict)
    model_available: bool = False
    test_node_count: int = 0
    usable_test_nodes: int = 0
    excluded_nodes: dict[str, int] = field(default_factory=dict)

    @property
    def has_test_samples(self) -> bool:
        """True when at least one test sample carries a known next state."""
        return bool(self.test) and int(self.test.get("true_latitude", np.array([])).size) > 0

    @property
    def has_validation_samples(self) -> bool:
        """True when the uncertainty layer has residuals to fit."""
        return (
            bool(self.validation)
            and int(self.validation.get("true_latitude", np.array([])).size) > 0
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of what was extracted."""
        return {
            "model_available": self.model_available,
            "test_node_count": self.test_node_count,
            "usable_test_nodes": self.usable_test_nodes,
            "excluded_nodes": dict(self.excluded_nodes),
            "validation_samples": (
                int(self.validation["true_latitude"].size)
                if self.has_validation_samples
                else 0
            ),
            "test_samples": (
                int(self.test["true_latitude"].size) if self.has_test_samples else 0
            ),
        }


def _collect(
    batch: Any,
    indices: Sequence[int],
    predictions: np.ndarray | None,
    require_prediction: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Gather samples for one set of node indices.

    Args:
        batch: The Phase 3 :class:`GraphBatch`.
        indices: Node indices belonging to this split.
        predictions: ``(num_nodes, 3)`` model output, or ``None``.
        require_prediction: Whether a node lacking a usable prediction is
            excluded.

    Returns:
        The sample arrays and a per-reason exclusion count.
    """
    columns: dict[str, list[float]] = {name: [] for name in SAMPLE_FIELDS}
    excluded: dict[str, int] = {}

    for index in indices:
        if index < 0 or index >= len(batch.targets):
            excluded["out_of_range"] = excluded.get("out_of_range", 0) + 1
            continue
        target = batch.targets[index]
        if target is None:
            # The last point of a track has no future state to predict.
            excluded["no_target"] = excluded.get("no_target", 0) + 1
            continue

        features = np.asarray(batch.node_features[index], dtype="float64")
        position = int(getattr(batch, "node_indices", [0] * len(batch.targets))[index])

        # Node features carry scaled latitude/longitude; the graph's own
        # convention is lat/90 and lon/180.
        current_lat = float(features[0] * 90.0)
        current_lon = float(features[1] * 180.0)
        current_intensity = float(features[4] * 100.0)

        previous_lat = float("nan")
        previous_lon = float("nan")
        if position > 0 and (index - 1) >= 0:
            previous = np.asarray(batch.node_features[index - 1], dtype="float64")
            previous_lat = float(previous[0] * 90.0)
            previous_lon = float(previous[1] * 180.0)

        true_lat = current_lat + float(target[0])
        true_lon = current_lon + float(target[1])
        true_intensity = current_intensity + float(target[2])

        predicted_lat = float("nan")
        predicted_lon = float("nan")
        predicted_intensity = float("nan")
        if predictions is not None:
            predicted = np.asarray(predictions[index], dtype="float64")
            if np.all(np.isfinite(predicted)):
                predicted_lat = current_lat + float(predicted[0])
                predicted_lon = current_lon + float(predicted[1])
                predicted_intensity = current_intensity + float(predicted[2])

        record = {
            "current_latitude": current_lat,
            "current_longitude": current_lon,
            "current_intensity": current_intensity,
            "previous_latitude": previous_lat,
            "previous_longitude": previous_lon,
            "true_latitude": true_lat,
            "true_longitude": true_lon,
            "true_intensity": true_intensity,
            "predicted_latitude": predicted_lat,
            "predicted_longitude": predicted_lon,
            "predicted_intensity": predicted_intensity,
        }

        if require_prediction and not all(
            np.isfinite(record[key])
            for key in ("predicted_latitude", "predicted_longitude", "predicted_intensity")
        ):
            excluded["no_prediction"] = excluded.get("no_prediction", 0) + 1
            continue

        for name, value in record.items():
            columns[name].append(value)

    samples = {
        name: np.asarray(values, dtype="float64")
        for name, values in columns.items()
        if values
    }
    return samples, excluded


def samples_from_tracks(
    tracks_path: Path | str,
    *,
    checkpoint: str | None = None,
) -> TrackSamples:
    """Build validation and test sample sets from a Phase 2 tracks file.

    Args:
        tracks_path: Location of ``tracks.json``.
        checkpoint: A trained Phase 3 checkpoint. When omitted, the baselines are
            still built but no model prediction is produced, and the evaluator
            reports the model as unavailable rather than inventing one.

    Returns:
        A :class:`TrackSamples`. An empty result is a valid outcome: a record
        with too few tracked events yields no evaluable samples, and that is
        reported rather than padded.
    """
    from extreme_weather.gnn.dataset import SplitConfig, build_dataset_from_file

    dataset = build_dataset_from_file(tracks_path, split_config=SplitConfig())
    batch = dataset.batch

    predictions: np.ndarray | None = None
    model_available = False
    if checkpoint:
        try:
            from extreme_weather.gnn.checkpoint import load_checkpoint
            from extreme_weather.gnn.model import WeatherGNN
            from extreme_weather.gnn.training import predict_all

            loaded = load_checkpoint(checkpoint)
            model = WeatherGNN()
            model.load_state_dict(loaded.state_dict)
            predictions = predict_all(model, batch)
            model_available = True
        except Exception:
            # An unusable checkpoint leaves the model unavailable. The
            # evaluator then reports NOT_AVAILABLE, which is the honest state.
            predictions = None
            model_available = False

    validation, _ = _collect(batch, dataset.split.validation, predictions, False)
    test, excluded = _collect(batch, dataset.split.test, predictions, True)
    return TrackSamples(
        validation=validation,
        test=test,
        model_available=model_available,
        test_node_count=len(dataset.split.test),
        usable_test_nodes=int(test.get("true_latitude", np.array([])).size),
        excluded_nodes=excluded,
    )
