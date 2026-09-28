"""Checkpoint persistence for the GNN prototype (SIH26078, Phase 3).

Format
------
A checkpoint is a single JSON document (``models/gnn/best.json`` by default) with
these top-level keys:

``schema_version``   Checkpoint layout version.
``model_version``    Architecture version (see
                     :data:`~extreme_weather.gnn.model.MODEL_VERSION`).
``model_config``     Every hyper-parameter, including the seed.
``node_feature_names`` / ``edge_feature_names`` / ``target_names``
                     The feature contract, so a consumer can verify it matches
                     the code that is about to run.
``feature_config``   The scaling constants, so inference transforms inputs
                     exactly as training did.
``training``         Optimisation settings, sample counts, final/best losses,
                     best epoch, seed and the creation timestamp.
``parameters``       Flat ``{name: [floats]}`` mapping of the model state.
``disclaimer``       Prototype notice, stored with the artefact.

Why JSON and not ``torch.save``: the checkpoint stays human-inspectable, diffable
and loadable without torch deserialisation (which is not a safe format to read
from untrusted files). The weights are a few thousand floats, so the size cost is
irrelevant.

Integrity
---------
:class:`CheckpointError` is raised for a missing file, malformed JSON, a missing
key, a feature-contract mismatch or a parameter set that does not match the
model. Every load validates before returning, so a corrupt checkpoint fails
loudly instead of producing plausible-looking garbage predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from extreme_weather.gnn.features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    TARGET_NAMES,
    FeatureConfig,
)
from extreme_weather.gnn.model import MODEL_VERSION, ModelConfig, WeatherGNN

__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROTOTYPE_DISCLAIMER",
    "CheckpointError",
    "GNNCheckpoint",
    "checkpoint_path",
    "build_training_metadata",
    "load_checkpoint",
    "save_checkpoint",
]

CHECKPOINT_SCHEMA_VERSION = "1.0"
CHECKPOINT_FILENAME = "best.json"

#: Repeated in every checkpoint so the artefact is never mistaken for a product.
PROTOTYPE_DISCLAIMER = (
    "Research prototype trained on synthetic data. Not an operational forecast, "
    "not validated against observed weather, and not suitable for any emergency "
    "decision."
)

_REQUIRED_KEYS = (
    "schema_version",
    "model_version",
    "model_config",
    "node_feature_names",
    "edge_feature_names",
    "target_names",
    "feature_config",
    "training",
    "parameters",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, malformed or inconsistent."""


def _shape_parameters(
    model: WeatherGNN, parameters: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Reshape flat checkpoint tensors to the shapes the model expects.

    :meth:`GNNCheckpoint.as_dict` stores each parameter as a flat list (JSON has
    no tensor type), so the original shape is lost on the way out. The reference
    model's ``state_dict`` carries the authoritative shapes, so they are restored
    from there - and a length mismatch is an error rather than a silent reshape.
    """
    reference = model.state_dict()
    restored: dict[str, torch.Tensor] = {}
    for name, value in reference.items():
        flat = parameters.get(name)
        if flat is None:
            raise CheckpointError(
                f"Checkpoint is missing the parameter '{name}' required by the architecture."
            )
        if flat.numel() != value.numel():
            raise CheckpointError(
                f"Checkpoint parameter '{name}' has {flat.numel()} value(s) but the model "
                f"expects {value.numel()}. The checkpoint does not match the architecture."
            )
        restored[name] = flat.reshape(value.shape)
    return restored


@dataclass
class GNNCheckpoint:
    """A validated checkpoint: configuration, provenance and weights.

    Attributes:
        model_config: Architecture and loss weights used for training.
        feature_config: Scaling constants used for the features.
        training: Training metadata (losses, counts, seed, timestamps).
        parameters: The model state as ``{name: tensor}``.
        model_version: Architecture version string.
        path: Where the checkpoint was loaded from, when known.
    """

    model_config: ModelConfig
    feature_config: FeatureConfig
    training: dict[str, Any]
    parameters: dict[str, torch.Tensor]
    model_version: str = MODEL_VERSION
    path: Path | None = None

    @property
    def seed(self) -> int:
        """Seed recorded with the training run."""
        return int(self.training.get("seed", self.model_config.seed))

    @property
    def best_validation_loss(self) -> float | None:
        """Lowest validation loss reached during training, if recorded."""
        value = self.training.get("best_validation_loss")
        return float(value) if value is not None else None

    @property
    def train_loss(self) -> float | None:
        """Final training loss, if recorded."""
        value = self.training.get("train_loss")
        return float(value) if value is not None else None

    def build_model(self) -> WeatherGNN:
        """Instantiate the model and load the stored parameters into it.

        Raises:
            CheckpointError: If the parameters do not match the architecture.
        """
        model = WeatherGNN(self.model_config)
        try:
            model.load_state_dict(_shape_parameters(model, self.parameters))
        except (RuntimeError, KeyError) as exc:
            raise CheckpointError(
                f"Checkpoint parameters do not match the model architecture: {exc}"
            ) from exc
        model.eval()
        return model

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation of the whole checkpoint."""
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "project": "SIH26078",
            "service": "DeepWeather AI",
            "phase": "Phase 3 - GNN event evolution (prototype)",
            "model_version": self.model_version,
            "model_config": self.model_config.as_dict(),
            "node_feature_names": list(NODE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "feature_config": {
                "intensity_scale": self.feature_config.intensity_scale,
                "speed_scale": self.feature_config.speed_scale,
                "cell_count_scale": self.feature_config.cell_count_scale,
                "edge_distance_scale": self.feature_config.edge_distance_scale,
                "edge_time_scale": self.feature_config.edge_time_scale,
            },
            "training": self.training,
            "parameters": {
                name: [float(value) for value in tensor.detach().flatten().tolist()]
                for name, tensor in self.parameters.items()
            },
            "disclaimer": PROTOTYPE_DISCLAIMER,
        }


def checkpoint_path(base_dir: Path | str | None = None) -> Path:
    """Return the conventional checkpoint location.

    Defaults to ``<settings.model_dir>/gnn/<CHECKPOINT_FILENAME>``, i.e.
    ``models/gnn/best.json``.
    """
    if base_dir is None:
        from extreme_weather.config import get_settings

        base_dir = get_settings().model_dir / "gnn"
    return Path(base_dir) / CHECKPOINT_FILENAME


def save_checkpoint(checkpoint: GNNCheckpoint, path: Path | str | None = None) -> Path:
    """Write a checkpoint to ``path`` (or the default location).

    Parent directories are created. The JSON is written with sorted keys and a
    trailing newline so the artefact is stable and diff-friendly.

    Returns:
        The path written to.

    Raises:
        CheckpointError: If the file cannot be written.
    """
    target = Path(path) if path is not None else checkpoint_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(checkpoint.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise CheckpointError(f"Could not write the checkpoint to '{target}': {exc}") from exc
    return target


def _validate_contract(payload: dict[str, Any], path: Path) -> None:
    """Check required keys and that the feature contract matches this code.

    Raises:
        CheckpointError: On a missing key or a contract mismatch.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(
            f"Checkpoint '{path}' is missing required key(s): {', '.join(missing)}."
        )

    for key, expected in (
        ("node_feature_names", NODE_FEATURE_NAMES),
        ("edge_feature_names", EDGE_FEATURE_NAMES),
        ("target_names", TARGET_NAMES),
    ):
        stored = list(payload[key])
        if stored != list(expected):
            raise CheckpointError(
                f"Checkpoint '{path}' has an incompatible {key}: stored {stored}, "
                f"code expects {list(expected)}. Retrain the model."
            )

    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"Checkpoint '{path}' has schema_version {payload['schema_version']!r}; "
            f"this code reads {CHECKPOINT_SCHEMA_VERSION!r}."
        )


def load_checkpoint(path: Path | str | None = None) -> GNNCheckpoint:
    """Load and fully validate a checkpoint.

    Args:
        path: Checkpoint location; defaults to :func:`checkpoint_path`.

    Returns:
        A :class:`GNNCheckpoint` with tensors already shaped.

    Raises:
        CheckpointError: If the file is missing, is not valid JSON, is missing a
            required key, has an incompatible feature contract, or holds
            parameters that do not match the architecture.
    """
    target = Path(path) if path is not None else checkpoint_path()
    if not target.is_file():
        raise CheckpointError(
            f"No GNN checkpoint at '{target}'. Run 'python scripts/train_gnn.py' to create one."
        )

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Checkpoint '{target}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint '{target}' must contain a JSON object.")

    _validate_contract(payload, target)

    try:
        model_config = ModelConfig(**payload["model_config"])
        feature_config = FeatureConfig(**payload["feature_config"])
    except Exception as exc:  # pydantic ValidationError and friends
        raise CheckpointError(f"Checkpoint '{target}' has an invalid configuration: {exc}") from exc

    parameters = {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in payload["parameters"].items()
    }
    if not parameters:
        raise CheckpointError(f"Checkpoint '{target}' contains no model parameters.")

    return GNNCheckpoint(
        model_config=model_config,
        feature_config=feature_config,
        training=dict(payload["training"]),
        parameters=parameters,
        model_version=str(payload["model_version"]),
        path=target,
    )


def build_training_metadata(
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    train_loss: float | None,
    validation_loss: float | None,
    best_validation_loss: float | None,
    best_epoch: int,
    test_loss: float | None = None,
    node_count: int | None = None,
    edge_count: int | None = None,
    track_count: int | None = None,
) -> dict[str, Any]:
    """Assemble the ``training`` block of a checkpoint.

    Every value is a measured quantity from the run - nothing here is a
    placeholder or an estimate.
    """
    return {
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "test_samples": int(test_samples),
        "train_loss": None if train_loss is None else float(train_loss),
        "validation_loss": None if validation_loss is None else float(validation_loss),
        "best_validation_loss": None if best_validation_loss is None else float(best_validation_loss),
        "best_epoch": int(best_epoch),
        "test_loss": None if test_loss is None else float(test_loss),
        "track_count": track_count,
        "node_count": node_count,
        "edge_count": edge_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_note": "Trained on synthetic prototype data; not a real-weather model.",
    }
