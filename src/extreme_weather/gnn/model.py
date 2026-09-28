"""The message-passing GNN prototype (SIH26078, Phase 3).

Architecture
------------
```text
node features (16)  ──▶ input projection (Linear → ReLU)  ──▶ h
                                                              │
   ┌──────────────────────────────────────────────────────────┘
   ▼
 layer 1  h' = ReLU( W_self · h + mean_{j in N(i)} ( W_msg · [h_j ; e_ij] ) )
   ▼
 layer 2  h'' = ReLU( W_self2 · h' + mean_{j in N'(i)} ( W_msg2 · [h'_j ; e_ij] ) )
   ▼
 prediction head  Linear → ReLU → Linear(→ 3)   =   [Δlat, Δlon, Δintensity]
```

Message passing, precisely
--------------------------
For a node ``i`` with neighbours ``j``:

* the **message** from ``j`` is ``m_ij = W_msg · concat(h_j, e_ij)``, i.e. it mixes
  the neighbour's state with the *edge* describing the relation to ``i``. This is
  the message-passing scheme of Gilmer et al. (2017) in its simplest form;
* the **aggregation** is the **mean** over incoming edges. Mean is used rather
  than sum because a node's neighbourhood size varies (a track step has one
  incoming edge, a node next to two other storms has two more), and a sum would
  let the magnitude of the prediction depend on how crowded the region is. A
  node with **no** incoming edges keeps its own representation - the message is
  skipped rather than averaged to zero;
* the **update** adds a self-transformed copy, so a node retains its own state:
  ``h_i' = ReLU(W_self · h_i + mean_j m_ij)``;
* the aggregation is computed for all edges at once, so layer ``L`` is a single
  vectorised scatter-add. There is no attention, no edge weighting and no
  recurrence, which keeps the model inspectable and CPU-cheap.

Targets
-------
The head regresses **deltas** (``Δlatitude``, ``Δlongitude``, ``Δintensity``), not
absolute coordinates. Predicting an absolute latitude would make the loss
dominated by the large constant offset of the coordinates (13 deg) rather than by
the movement that matters (millidegrees). Decoding is therefore
``next = current + predicted delta``.

Determinism
-----------
Weights are drawn from a local ``torch.Generator`` seeded with
:attr:`ModelConfig.seed`, so initialisation does not depend on global RNG state
and two models built with the same seed are bit-identical. Inference runs under
``eval()`` with no dropout, so the same input always yields the same output.

Dependencies
------------
PyTorch is already a Phase 0 project dependency (used from Phase 2 onwards), and
it runs on CPU. **PyTorch Geometric is deliberately not used**: a four-layer MLP
message passing is a few dozen lines of plain ``index_add_``, and PyG would add
a heavy, platform-sensitive dependency for no benefit at this scale.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from extreme_weather.gnn.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES

__all__ = [
    "GNNError",
    "MODEL_VERSION",
    "MessagePassingLayer",
    "ModelConfig",
    "PredictionHead",
    "WeatherGNN",
    "compute_loss",
    "init_generator",
    "loss_components",
]

#: Bumped whenever the parameter layout or the architecture changes, so an old
#: checkpoint cannot be silently loaded into a different model.
MODEL_VERSION = "gnn-prototype-1.0"


class GNNError(RuntimeError):
    """Raised when the model is misconfigured or fed inconsistent tensors."""


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of the prototype network.

    Attributes:
        node_dim: Width of the node feature vector (16, see
            :data:`~extreme_weather.gnn.features.NODE_FEATURE_NAMES`).
        edge_dim: Width of the edge feature vector (6).
        hidden_dim: Width of the internal representation.
        message_layers: Number of message-passing layers (>= 1).
        target_dim: Width of the regression output (3).
        position_weight: Weight of the position term in the loss (see
            :func:`compute_loss`).
        intensity_weight: Weight of the intensity term in the loss.
        seed: Seed of the *initialisation* generator.
    """

    node_dim: int = len(NODE_FEATURE_NAMES)
    edge_dim: int = len(EDGE_FEATURE_NAMES)
    hidden_dim: int = 32
    message_layers: int = 2
    target_dim: int = len(TARGET_NAMES)
    position_weight: float = 1.0
    intensity_weight: float = 0.5
    seed: int = 20260601

    def __post_init__(self) -> None:
        if self.node_dim <= 0 or self.edge_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("node_dim, edge_dim and hidden_dim must all be > 0.")
        if self.message_layers < 1:
            raise ValueError(f"message_layers must be >= 1 (got {self.message_layers}).")
        if self.target_dim <= 0:
            raise ValueError("target_dim must be > 0.")
        if self.position_weight < 0.0 or self.intensity_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if self.position_weight + self.intensity_weight <= 0.0:
            raise ValueError("At least one loss weight must be > 0.")

    def as_dict(self) -> dict[str, int | float]:
        """Plain-dict view, used when writing the checkpoint."""
        return dict(asdict(self))


def init_generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded with ``seed``.

    Using an explicit generator (instead of the global RNG) is what makes model
    initialisation reproducible regardless of what else ran before it.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


class MessagePassingLayer(nn.Module):
    """One mean-aggregation message-passing layer.

    Shapes: ``node_states`` ``(num_nodes, hidden_dim)``, ``edge_states``
    ``(num_edges, edge_dim)``, ``edge_index`` ``(2, num_edges)`` of
    ``[sources; targets]``; returns ``(num_nodes, hidden_dim)``.
    """

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        # m_ij = W_msg . [h_j ; e_ij]
        self.message = nn.Linear(hidden_dim + edge_dim, hidden_dim)
        # h_i <- W_self . h_i
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_states: Tensor, edge_states: Tensor, edge_index: Tensor) -> Tensor:
        """Aggregate neighbour messages and update every node.

        Nodes without incoming edges keep ``update(h_i)``; nodes with incoming
        edges get the **mean** of their messages added to it.
        """
        if node_states.ndim != 2 or node_states.shape[1] != self.hidden_dim:
            raise GNNError(
                f"node_states must be (num_nodes, {self.hidden_dim}); "
                f"got {tuple(node_states.shape)}."
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise GNNError(f"edge_index must be (2, num_edges); got {tuple(edge_index.shape)}.")
        if edge_states.shape[0] != edge_index.shape[1]:
            raise GNNError(
                "edge_states and edge_index disagree: "
                f"{edge_states.shape[0]} edge features vs {edge_index.shape[1]} edges."
            )

        self_update = self.update(node_states)
        if edge_index.shape[1] == 0:
            return self.activation(self_update)

        sources, targets = edge_index[0], edge_index[1]
        messages = self.message(torch.cat([node_states[sources], edge_states], dim=1))

        num_nodes = node_states.shape[0]
        aggregated = torch.zeros(num_nodes, self.hidden_dim, dtype=messages.dtype)
        aggregated.index_add_(0, targets, messages)

        # Mean aggregation: divide by the in-degree of each node.
        in_degree = torch.zeros(num_nodes, dtype=messages.dtype)
        in_degree.index_add_(0, targets, torch.ones(targets.shape[0], dtype=messages.dtype))
        has_neighbour = in_degree > 0
        safe_degree = torch.where(has_neighbour, in_degree, torch.ones_like(in_degree))
        aggregated = aggregated / safe_degree.unsqueeze(1)
        # Isolated nodes must not receive the (zero) aggregate of "no edges".
        aggregated = aggregated * has_neighbour.unsqueeze(1).to(aggregated.dtype)

        return self.activation(self_update + aggregated)


class PredictionHead(nn.Module):
    """Two-layer MLP mapping the final representation to the three deltas."""

    def __init__(self, hidden_dim: int, target_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, target_dim)
        self.activation = nn.ReLU()

    def forward(self, node_states: Tensor) -> Tensor:
        """Return ``(num_nodes, target_dim)``."""
        return self.output(self.activation(self.hidden(node_states)))


class WeatherGNN(nn.Module):
    """Deterministic message-passing network for next-state prediction.

    Args:
        config: Hyper-parameters. With a fixed ``seed`` the initialisation is
            bit-for-bit reproducible.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.input_projection = nn.Linear(cfg.node_dim, cfg.hidden_dim)
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList(
            MessagePassingLayer(cfg.hidden_dim, cfg.edge_dim)
            for _ in range(cfg.message_layers)
        )
        self.head = PredictionHead(cfg.hidden_dim, cfg.target_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise every weight from the seeded generator."""
        generator = init_generator(self.config.seed)
        for parameter in self.parameters():
            if parameter.dim() >= 2:
                bound = 1.0 / math.sqrt(parameter.shape[1])
                with torch.no_grad():
                    parameter.uniform_(-bound, bound, generator=generator)
            else:
                with torch.no_grad():
                    parameter.zero_()

    @property
    def layer_count(self) -> int:
        """Number of message-passing layers."""
        return len(self.layers)

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Predict the deltas of every node.

        Args:
            node_features: ``(num_nodes, node_dim)``
            edge_features: ``(num_edges, edge_dim)``
            edge_index: ``(2, num_edges)`` of ``[sources; targets]``

        Returns:
            ``(num_nodes, target_dim)`` of ``[Δlat, Δlon, Δintensity]``.

        Raises:
            GNNError: If the input widths or the edge index are inconsistent.
        """
        if node_features.ndim != 2 or node_features.shape[1] != self.config.node_dim:
            raise GNNError(
                f"node_features must be (num_nodes, {self.config.node_dim}); "
                f"got {tuple(node_features.shape)}."
            )
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.edge_dim:
            raise GNNError(
                f"edge_features must be (num_edges, {self.config.edge_dim}); "
                f"got {tuple(edge_features.shape)}."
            )
        if node_features.shape[0] == 0:
            raise GNNError("Cannot run the model on an empty graph.")

        states = self.activation(self.input_projection(node_features))
        for layer in self.layers:
            states = layer(states, edge_features, edge_index)
        return self.head(states)


def compute_loss(
    predictions: Tensor,
    targets: Tensor,
    config: ModelConfig | None = None,
) -> Tensor:
    """Multi-objective regression loss over the three deltas.

    ```text
    position_error  = MSE(Δlat_pred, Δlat_true) + MSE(Δlon_pred, Δlon_true)
    intensity_error = MSE(Δintensity_pred, Δintensity_true)

    loss = position_weight * position_error + intensity_weight * intensity_error
    ```

    The two position components are **summed** (not averaged) so the term is the
    squared Euclidean displacement error in degrees, which is the quantity of
    physical interest; the two are kept separate from intensity because a
    millidegree of latitude error and a unit of mm h-1 are not comparable
    numbers, and mixing them naively would let whichever term is numerically
    larger dictate the optimisation.

    The default weights (position 1.0, intensity 0.5) express that *where* the
    storm goes matters more than how strong it is: for trajectory extrapolation
    the position is the primary quantity, while intensity is secondary. Both are
    configurable through :class:`ModelConfig`.

    Args:
        predictions: ``(num_samples, 3)`` predicted deltas.
        targets: ``(num_samples, 3)`` true deltas.
        config: Supplies the loss weights; defaults to :class:`ModelConfig`.

    Returns:
        A scalar tensor.

    Raises:
        GNNError: If the shapes disagree or no sample is given.
    """
    cfg = config or ModelConfig()
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise GNNError(f"predictions must be (num_samples, 3); got {tuple(predictions.shape)}.")
    if tuple(targets.shape) != tuple(predictions.shape):
        raise GNNError(
            f"targets must match predictions; got {tuple(targets.shape)} vs "
            f"{tuple(predictions.shape)}."
        )
    if predictions.shape[0] == 0:
        raise GNNError("compute_loss requires at least one sample.")

    position_error = torch.mean((predictions[:, 0] - targets[:, 0]) ** 2) + torch.mean(
        (predictions[:, 1] - targets[:, 1]) ** 2
    )
    intensity_error = torch.mean((predictions[:, 2] - targets[:, 2]) ** 2)
    return cfg.position_weight * position_error + cfg.intensity_weight * intensity_error


def loss_components(
    predictions: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    """Return the individual loss terms for reporting and diagnostics."""
    with torch.no_grad():
        position_lat = torch.mean((predictions[:, 0] - targets[:, 0]) ** 2).item()
        position_lon = torch.mean((predictions[:, 1] - targets[:, 1]) ** 2).item()
        intensity = torch.mean((predictions[:, 2] - targets[:, 2]) ** 2).item()
    return {
        "position_latitude": float(position_lat),
        "position_longitude": float(position_lon),
        "intensity": float(intensity),
        "position_total": float(position_lat + position_lon),
    }
