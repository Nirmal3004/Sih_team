"""Checkpoint persistence for the AI downscaler (SIH26078, Phase 4).

Format
------
A single **PyTorch** file, ``models/downscaling/best.pt`` by default. Unlike the
Phase 3 GNN (a few thousand small floats), this checkpoint holds convolutional
weight tensors, so the native format is used rather than flattening them to JSON.

Payload
-------
``model_state``          The ``state_dict`` of :class:`DownscalingModel`.
``model_config``         Architecture hyper-parameters.
``model_version``        Architecture version string.
``feature_names``        The input feature contract.
``normalization``        Per-channel mean/std **fitted on training data only**.
``input_resolution_deg`` / ``output_resolution_deg``  Detected from the data.
``input_shape`` / ``output_shape``  Grid sizes as ``(lat, lon)``.
``variable``             Downscaled variable name.
``training``             Seed, epochs, learning rate, losses, best epoch, counts.
``disclaimer``           The prototype notice, stored with the artefact.

Integrity
---------
:class:`DownscalingCheckpointError` is raised for a missing file, a payload that
is not a mapping, a missing key, a feature-contract mismatch, or parameters that
do not match the architecture. Every load validates before returning, so a
corrupt checkpoint fails loudly instead of producing plausible-looking garbage.

``torch.load`` is called with ``weights_only=True``, restricting the payload to
tensors and plain containers. This artefact is produced locally; a checkpoint
from an untrusted source should still not be loaded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from extreme_weather.downscaling.dataset import NormalizationStats
from extreme_weather.downscaling.features import FEATURE_NAMES
from extreme_weather.downscaling.model import MODEL_VERSION, DownscalingModel, ModelConfig

__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROTOTYPE_DISCLAIMER",
    "DownscalingCheckpoint",
    "DownscalingCheckpointError",
    "build_training_metadata",
    "checkpoint_path",
    "load_checkpoint",
    "save_checkpoint",
]

CHECKPOINT_SCHEMA_VERSION = "1.0"
CHECKPOINT_FILENAME = "best.pt"

PROTOTYPE_DISCLAIMER = (
    "Research prototype trained on synthetic data against a SYNTHETIC high-resolution "
    "reference. Not an operational forecast, not validated against observations, and "
    "not suitable for any emergency decision. Higher resolution does not imply higher "
    "accuracy."
)

_REQUIRED_KEYS = (
    "schema_version",
    "model_version",
    "model_config",
    "feature_names",
    "normalization",
    "input_resolution_deg",
    "output_resolution_deg",
    "input_shape",
    "output_shape",
    "variable",
    "training",
    "model_state",
)


class DownscalingCheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, malformed or inconsistent."""


class DownscalingCheckpoint:
    """A validated checkpoint: configuration, provenance and weights.

    Attributes:
        model_config: Architecture hyper-parameters.
        normalization: Statistics fitted on the training block only.
        training: Training metadata (seed, epochs, losses, counts).
        model_state: The model weights as a ``state_dict``.
        input_resolution_deg: Detected coarse resolution, degrees.
        output_resolution_deg: Detected fine resolution, degrees.
        input_shape: ``(lat, lon)`` coarse grid size.
        output_shape: ``(lat, lon)`` fine grid size.
        variable: Downscaled variable name.
        model_version: Architecture version string.
        path: Where the checkpoint was loaded from, when known.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        normalization: NormalizationStats,
        training: dict[str, Any],
        model_state: dict[str, torch.Tensor],
        input_resolution_deg: float,
        output_resolution_deg: float,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        variable: str,
        model_version: str = MODEL_VERSION,
        path: Path | None = None,
    ) -> None:
        self.model_config = model_config
        self.normalization = normalization
        self.training = training
        self.model_state = model_state
        self.input_resolution_deg = input_resolution_deg
        self.output_resolution_deg = output_resolution_deg
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.variable = variable
        self.model_version = model_version
        self.path = path

    @property
    def scale_factor(self) -> float:
        """Ratio of input to output resolution."""
        return self.input_resolution_deg / self.output_resolution_deg

    @property
    def best_validation_loss(self) -> float | None:
        """Lowest validation loss reached during training, if recorded."""
        value = self.training.get("best_validation_loss")
        return float(value) if value is not None else None

    @property
    def seed(self) -> int:
        """Seed recorded with the training run."""
        return int(self.training.get("seed", 0))

    def build_model(self) -> DownscalingModel:
        """Instantiate the model and load the stored weights.

        Raises:
            DownscalingCheckpointError: If the weights do not match the
                architecture.
        """
        model = DownscalingModel(self.model_config)
        try:
            model.load_state_dict(self.model_state)
        except (RuntimeError, KeyError) as exc:
            raise DownscalingCheckpointError(
                f"Checkpoint weights do not match the architecture: {exc}"
            ) from exc
        model.eval()
        return model

    def as_dict(self) -> dict[str, Any]:
        """Return the payload that :func:`save_checkpoint` writes."""
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "project": "SIH26078",
            "service": "DeepWeather AI",
            "phase": "Phase 4 - AI downscaling (research prototype)",
            "model_version": self.model_version,
            "model_config": self.model_config.as_dict(),
            "feature_names": list(FEATURE_NAMES),
            "normalization": self.normalization.as_dict(),
            "input_resolution_deg": self.input_resolution_deg,
            "output_resolution_deg": self.output_resolution_deg,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "variable": self.variable,
            "training": self.training,
            "model_state": self.model_state,
            "disclaimer": PROTOTYPE_DISCLAIMER,
        }


def checkpoint_path(base_dir: Path | str | None = None) -> Path:
    """Return the conventional checkpoint location.

    Defaults to ``<settings.model_dir>/downscaling/best.pt``.
    """
    if base_dir is None:
        from extreme_weather.config import get_settings

        base_dir = get_settings().model_dir / "downscaling"
    return Path(base_dir) / CHECKPOINT_FILENAME


def save_checkpoint(checkpoint: DownscalingCheckpoint, path: Path | str | None = None) -> Path:
    """Write a checkpoint, creating parent directories.

    Returns:
        The path written to.

    Raises:
        DownscalingCheckpointError: If the file cannot be written.
    """
    target = Path(path) if path is not None else checkpoint_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint.as_dict(), target)
    except OSError as exc:
        raise DownscalingCheckpointError(
            f"Could not write the checkpoint to '{target}': {exc}"
        ) from exc
    return target


def load_checkpoint(path: Path | str | None = None) -> DownscalingCheckpoint:
    """Load and fully validate a checkpoint.

    Args:
        path: Checkpoint location; defaults to :func:`checkpoint_path`.

    Returns:
        A validated :class:`DownscalingCheckpoint`.

    Raises:
        DownscalingCheckpointError: If the file is missing, unreadable, missing a
            key, or inconsistent with this code.
    """
    target = Path(path) if path is not None else checkpoint_path()
    if not target.is_file():
        raise DownscalingCheckpointError(
            f"No downscaling checkpoint at '{target}'. Run "
            "'python scripts/train_downscaling.py' to create one."
        )
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises many types for a bad file
        raise DownscalingCheckpointError(
            f"Could not read the checkpoint '{target}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DownscalingCheckpointError(
            f"Checkpoint '{target}' must contain a mapping, got {type(payload).__name__}."
        )

    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise DownscalingCheckpointError(
            f"Checkpoint '{target}' is missing required key(s): {', '.join(missing)}."
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise DownscalingCheckpointError(
            f"Checkpoint '{target}' has schema_version {payload['schema_version']!r}; "
            f"this code reads {CHECKPOINT_SCHEMA_VERSION!r}."
        )
    if list(payload["feature_names"]) != list(FEATURE_NAMES):
        raise DownscalingCheckpointError(
            f"Checkpoint '{target}' has an incompatible feature contract: stored "
            f"{list(payload['feature_names'])}, code expects {list(FEATURE_NAMES)}. Retrain."
        )

    try:
        model_config = ModelConfig(**payload["model_config"])
        normalization = NormalizationStats.from_dict(payload["normalization"])
    except Exception as exc:
        raise DownscalingCheckpointError(
            f"Checkpoint '{target}' has an invalid configuration: {exc}"
        ) from exc

    return DownscalingCheckpoint(
        model_config=model_config,
        normalization=normalization,
        training=dict(payload["training"]),
        model_state=payload["model_state"],
        input_resolution_deg=float(payload["input_resolution_deg"]),
        output_resolution_deg=float(payload["output_resolution_deg"]),
        input_shape=(int(payload["input_shape"][0]), int(payload["input_shape"][1])),
        output_shape=(int(payload["output_shape"][0]), int(payload["output_shape"][1])),
        variable=str(payload["variable"]),
        model_version=str(payload["model_version"]),
        path=target,
    )


def build_training_metadata(
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    train_loss: float | None,
    validation_loss: float | None,
    best_validation_loss: float | None,
    best_epoch: int,
    test_loss: float | None = None,
    stopped_early: bool = False,
) -> dict[str, Any]:
    """Assemble the ``training`` block from measured values of the run."""
    return {
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "test_samples": int(test_samples),
        "train_loss": None if train_loss is None else float(train_loss),
        "validation_loss": None if validation_loss is None else float(validation_loss),
        "best_validation_loss": (
            None if best_validation_loss is None else float(best_validation_loss)
        ),
        "best_epoch": int(best_epoch),
        "test_loss": None if test_loss is None else float(test_loss),
        "stopped_early": bool(stopped_early),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_note": (
            "Trained against a SYNTHETIC high-resolution reference generated from the "
            "coarse field. Not observational truth; metrics are not real-world skill."
        ),
    }

