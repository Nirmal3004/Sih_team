"""Training loop for the GNN prototype (SIH26078, Phase 3).

Protocol
--------
* Only the **training** block updates the weights.
* The **validation** block is evaluated after every epoch and the parameters with
  the lowest validation loss are kept - that snapshot becomes the checkpoint.
* The **test** block is never used to update or select anything. It is evaluated
  once, at the end, purely as a diagnostic.

Because the split is chronological (:mod:`extreme_weather.gnn.dataset`), the
validation block is a period the model was never fitted on, so its loss is a
meaningful - if small-sample - generalisation signal.

Determinism
-----------
The seed controls the shuffling of the training sample order (initialisation uses
:attr:`ModelConfig.seed`). Both come from a local generator, so two runs with the
same seed produce the same loss curve without touching global RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from extreme_weather.gnn.graph import GraphBatch
from extreme_weather.gnn.model import (
    GNNError,
    ModelConfig,
    WeatherGNN,
    compute_loss,
    init_generator,
)

__all__ = [
    "EpochRecord",
    "TrainingConfig",
    "TrainingHistory",
    "TrainingResult",
    "evaluate",
    "predict_all",
    "train",
]


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings.

    Attributes:
        epochs: Number of passes over the training block.
        learning_rate: Adam step size.
        seed: Seed of the sample-order generator.
        verbose: Print one line per epoch.
    """

    epochs: int = 200
    learning_rate: float = 0.01
    seed: int = 20260601
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0 (got {self.epochs}).")
        if not self.learning_rate > 0.0:
            raise ValueError(f"learning_rate must be > 0 (got {self.learning_rate}).")


@dataclass(frozen=True)
class EpochRecord:
    """Losses of one epoch."""

    epoch: int
    train_loss: float
    validation_loss: float | None


@dataclass
class TrainingHistory:
    """Full loss curve plus the best epoch."""

    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float | None = None
    best_state: dict[str, Tensor] | None = None

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
                }
                for record in self.records
            ],
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
        }


@dataclass
class TrainingResult:
    """Outcome of a training run.

    Attributes:
        model: The model with the **best** (lowest validation loss) weights
            restored, ready for inference.
        history: The full loss curve.
        config: The optimisation settings used.
        test_loss: Loss on the held-out test block, evaluated once at the end
            (``None`` when that block is empty).
    """

    model: WeatherGNN
    history: TrainingHistory
    config: TrainingConfig
    test_loss: float | None = None


def graph_tensors(batch: GraphBatch) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a :class:`GraphBatch` into model-ready tensors (float32).

    An edgeless graph yields a correctly shaped ``(0, 0)`` edge tensor rather
    than an empty list, so the model can still run on isolated nodes.
    """
    if batch.node_count == 0:
        raise GNNError("Cannot build tensors for an empty graph.")
    nodes = torch.tensor(batch.node_features, dtype=torch.float32)
    if batch.edge_features:
        edges = torch.tensor(batch.edge_features, dtype=torch.float32)
    else:
        edges = torch.zeros((0, batch.edge_dim or 1), dtype=torch.float32)
    sources, targets = batch.edge_index
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return nodes, edges, edge_index


def predict_all(model: WeatherGNN, batch: GraphBatch) -> Tensor:
    """Run the model over the whole graph, returning ``(num_nodes, 3)``."""
    if batch.node_count == 0:
        raise GNNError("Cannot predict on an empty graph.")
    nodes, edges, edge_index = graph_tensors(batch)
    model.eval()
    with torch.no_grad():
        return model(nodes, edges, edge_index)


def evaluate(
    model: WeatherGNN,
    batch: GraphBatch,
    node_indices: list[int],
) -> float | None:
    """Loss over ``node_indices``; ``None`` when the block is empty."""
    if not node_indices:
        return None
    predictions = predict_all(model, batch)
    targets = torch.tensor([batch.targets[index] for index in node_indices], dtype=torch.float32)
    return float(compute_loss(predictions[node_indices], targets, model.config).item())


def train(
    batch: GraphBatch,
    train_indices: list[int],
    validation_indices: list[int],
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    test_indices: list[int] | None = None,
) -> TrainingResult:
    """Train the model and restore the best-validation snapshot.

    Each epoch is a **full-batch** gradient step over the training nodes: the
    graph is small (a prototype dataset has tens of nodes), so mini-batching would
    add complexity without a benefit. The training nodes are permuted
    deterministically each epoch, which keeps the shuffling reproducible and
    slightly decorrelates the samples seen by the optimiser.

    Args:
        batch: The graph holding features, edges and targets.
        train_indices: Node indices used for weight updates.
        validation_indices: Node indices used for model selection.
        model_config: Architecture and loss weights.
        training_config: Optimisation settings.
        test_indices: Held-out nodes, evaluated **once** after training.

    Returns:
        A :class:`TrainingResult` whose ``model`` carries the best weights.

    Raises:
        GNNError: If the graph is empty or no training sample is available.
    """
    cfg = training_config or TrainingConfig()
    model = WeatherGNN(model_config)

    if batch.node_count == 0:
        raise GNNError("Cannot train on an empty graph.")
    if not train_indices:
        raise GNNError("No training samples: the chronological split produced an empty block.")

    nodes, edges, edge_index = graph_tensors(batch)
    targets = torch.tensor([batch.targets[index] for index in train_indices], dtype=torch.float32)
    train_tensor = torch.tensor(train_indices, dtype=torch.long)

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = init_generator(cfg.seed)
    history = TrainingHistory()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimiser.zero_grad()

        order = torch.randperm(train_tensor.shape[0], generator=generator)
        predictions = model(nodes, edges, edge_index)[train_tensor[order]]
        loss = compute_loss(predictions, targets[order], model.config)
        loss.backward()
        optimiser.step()

        train_loss = float(loss.item())
        validation_loss = evaluate(model, batch, validation_indices)
        history.records.append(
            EpochRecord(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss)
        )

        if cfg.verbose:
            validation_text = "n/a" if validation_loss is None else f"{validation_loss:.6f}"
            print(f"  epoch {epoch:4d}  train={train_loss:.6f}  val={validation_text}")

        # Model selection on validation loss. A run without validation data has
        # nothing to select on and simply keeps the final weights.
        if validation_loss is not None and (
            history.best_validation_loss is None or validation_loss < history.best_validation_loss
        ):
            history.best_validation_loss = validation_loss
            history.best_epoch = epoch
            history.best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

    if history.best_state is not None:
        model.load_state_dict(history.best_state)

    result = TrainingResult(model=model, history=history, config=cfg)
    if test_indices:
        result.test_loss = evaluate(model, batch, test_indices)
    return result
