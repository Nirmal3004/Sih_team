"""Training loop for the AI downscaler (SIH26078, Phase 4).

Protocol
--------
* Only the **training** block updates the weights.
* The **validation** block is evaluated after every epoch and the parameters with
  the lowest validation loss are kept; that snapshot is the checkpoint.
* The **test** block is never used to update or select anything. It is evaluated
  once at the end, purely as a diagnostic.

Because the split is chronological, validation is a period the model was never
fitted on, so its loss is a meaningful (if small-sample) generalisation signal.

Loss
----
```text
loss = MSE(predicted_residual, target_residual)
```

MSE is used because the target is a *residual*: small, zero-centred and unbounded,
so a squared error is well behaved and needs no weighting. MAE is **reported**
alongside it so the two are visible, but the optimised loss stays the single,
explainable MSE - no gradient-L1, no adversarial term and no spatial-gradient
term, because none of them is justified by an experiment here and inventing one
to move a metric would be dishonest.

Determinism
-----------
The seed controls the shuffle of the training sample order, drawn from a local
generator, so two runs with the same seed produce the same loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from extreme_weather.downscaling.dataset import DownscalingDataset
from extreme_weather.downscaling.model import (
    DownscalingError,
    DownscalingModel,
    ModelConfig,
    init_generator,
)

__all__ = [
    "EpochRecord",
    "TrainingConfig",
    "TrainingHistory",
    "TrainingResult",
    "evaluate_block",
    "train",
]


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings.

    Attributes:
        epochs: Maximum number of passes over the training block.
        learning_rate: Adam step size.
        batch_size: Samples per gradient step.
        seed: Seed of the sample-order generator.
        weight_decay: Adam L2 penalty (0 disables it).
        patience: Early-stopping patience in epochs; ``None`` disables it.
        verbose: Print one line per epoch.
    """

    epochs: int = 60
    learning_rate: float = 2e-3
    batch_size: int = 8
    seed: int = 20260601
    weight_decay: float = 0.0
    # Early stopping is OFF by default. On this dataset validation loss falls
    # slowly and monotonically, so a patience window cut training off long before
    # it converged (measured: val MSE 0.0387 stopping at epoch 1 vs 0.0254 running
    # to epoch 56). Best-on-validation checkpoint selection is the primary
    # mechanism, exactly as the phase design allows; pass patience=N to re-enable.
    patience: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}.")
        if not self.learning_rate > 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if self.patience is not None and self.patience <= 0:
            raise ValueError(f"patience must be > 0 or None, got {self.patience}.")


@dataclass(frozen=True)
class EpochRecord:
    """Losses of one epoch."""

    epoch: int
    train_loss: float
    validation_loss: float | None
    validation_mae: float | None = None


@dataclass
class TrainingHistory:
    """Full loss curve, the best epoch and the best-weights snapshot."""

    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float | None = None
    best_state: dict[str, Tensor] | None = None
    stopped_early: bool = False

    @property
    def final_train_loss(self) -> float | None:
        """Training loss of the last recorded epoch."""
        return self.records[-1].train_loss if self.records else None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view (no tensors)."""
        return {
            "epochs": [
                {
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "validation_loss": record.validation_loss,
                    "validation_mae": record.validation_mae,
                }
                for record in self.records
            ],
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
        }


@dataclass
class TrainingResult:
    """Outcome of a training run.

    Attributes:
        model: The model with the best-validation weights restored.
        history: The full loss curve.
        config: The optimisation settings used.
        test_loss: MSE on the held-out test block, evaluated once.
    """

    model: DownscalingModel
    history: TrainingHistory
    config: TrainingConfig
    test_loss: float | None = None


def _tensors(dataset: DownscalingDataset, indices: list[int]) -> tuple[Tensor, Tensor]:
    """Return the normalised input and residual target tensors of a block."""
    features = torch.tensor(
        dataset.normalization.apply(dataset.features[indices]), dtype=torch.float32
    )
    targets = torch.tensor(dataset.targets[indices], dtype=torch.float32)
    return features, targets


def evaluate_block(
    model: DownscalingModel,
    dataset: DownscalingDataset,
    indices: list[int],
) -> tuple[float | None, float | None]:
    """Return ``(mse, mae)`` of a block, or ``(None, None)`` when it is empty."""
    if not indices:
        return None, None
    features, targets = _tensors(dataset, indices)
    model.eval()
    with torch.no_grad():
        predicted = model(features, target_shape=dataset.output_shape)
        error = predicted - targets
        return (
            float(torch.mean(error**2).item()),
            float(torch.mean(torch.abs(error)).item()),
        )


def train(
    dataset: DownscalingDataset,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train the downscaler and restore the best-validation snapshot.

    Args:
        dataset: Features, residual targets, split and normalisation statistics.
        model_config: Architecture hyper-parameters.
        training_config: Optimisation settings.

    Returns:
        A :class:`TrainingResult` whose model holds the best weights.

    Raises:
        DownscalingError: If the training block is empty.
    """
    cfg = training_config or TrainingConfig()
    model = DownscalingModel(model_config)
    history = TrainingHistory()

    train_indices = dataset.split.train
    if not train_indices:
        raise DownscalingError("The training split is empty; nothing to learn from.")

    features_all = torch.tensor(dataset.normalized_features(), dtype=torch.float32)
    targets_all = torch.tensor(dataset.targets, dtype=torch.float32)
    train_tensor = torch.tensor(train_indices, dtype=torch.long)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = init_generator(cfg.seed)
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(train_tensor.shape[0], generator=generator)

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, order.shape[0], cfg.batch_size):
            batch_indices = train_tensor[order[start : start + cfg.batch_size]]
            optimiser.zero_grad()
            predicted = model(
                features_all[batch_indices], target_shape=dataset.output_shape
            )
            loss = torch.mean((predicted - targets_all[batch_indices]) ** 2)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        validation_loss, validation_mae = evaluate_block(
            model, dataset, dataset.split.validation
        )
        history.records.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_mae=validation_mae,
            )
        )
        if cfg.verbose:
            text = "n/a" if validation_loss is None else f"{validation_loss:.6f}"
            print(f"  epoch {epoch:4d}  train={train_loss:.6f}  val={text}")

        if validation_loss is not None:
            improved = (
                history.best_validation_loss is None
                or validation_loss < history.best_validation_loss
            )
            if improved:
                history.best_validation_loss = validation_loss
                history.best_epoch = epoch
                history.best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if cfg.patience is not None and epochs_without_improvement >= cfg.patience:
                    history.stopped_early = True
                    break

    if history.best_state is not None:
        model.load_state_dict(history.best_state)

    result = TrainingResult(model=model, history=history, config=cfg)
    test_loss, _ = evaluate_block(model, dataset, dataset.split.test)
    result.test_loss = test_loss
    return result


