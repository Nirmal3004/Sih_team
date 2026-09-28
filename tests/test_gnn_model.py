"""Model, message-passing and loss tests (SIH26078, Phase 3)."""

from __future__ import annotations

import pytest
import torch

from extreme_weather.gnn.graph import GraphConfig, build_graph
from extreme_weather.gnn.model import (
    MODEL_VERSION,
    GNNError,
    MessagePassingLayer,
    ModelConfig,
    WeatherGNN,
    compute_loss,
    init_generator,
    loss_components,
)
from extreme_weather.gnn.training import graph_tensors


@pytest.fixture
def small_model() -> WeatherGNN:
    """A tiny model so the tests stay fast on CPU."""
    return WeatherGNN(ModelConfig(hidden_dim=8, message_layers=2, seed=7))


# --- configuration ----------------------------------------------------------


def test_default_config_matches_the_feature_contract():
    config = ModelConfig()
    assert config.node_dim == 16
    assert config.edge_dim == 6
    assert config.target_dim == 3


def test_config_rejects_invalid_architecture():
    with pytest.raises(ValueError, match="hidden_dim"):
        ModelConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="message_layers"):
        ModelConfig(message_layers=0)


def test_config_rejects_all_zero_loss_weights():
    with pytest.raises(ValueError, match="loss weight"):
        ModelConfig(position_weight=0.0, intensity_weight=0.0)


def test_model_version_is_exposed():
    assert MODEL_VERSION.startswith("gnn-prototype")


# --- shape ------------------------------------------------------------------


def test_output_shape_is_nodes_by_three(small_model, single_track):
    nodes, edges, edge_index = graph_tensors(build_graph([single_track]))
    assert small_model(nodes, edges, edge_index).shape == (nodes.shape[0], 3)


def test_model_runs_on_an_edgeless_graph(small_model, many_tracks):
    """Isolated nodes must not crash the message passing."""
    batch = build_graph(many_tracks, GraphConfig(include_spatial_edges=False))
    nodes, edges, edge_index = graph_tensors(batch)
    output = small_model(nodes, edges, edge_index)
    assert output.shape == (batch.node_count, 3)
    assert torch.isfinite(output).all()


def test_model_output_is_finite(small_model, many_tracks):
    nodes, edges, edge_index = graph_tensors(build_graph(many_tracks))
    assert torch.isfinite(small_model(nodes, edges, edge_index)).all()


def test_wrong_node_width_is_rejected(small_model, single_track):
    nodes, edges, edge_index = graph_tensors(build_graph([single_track]))
    with pytest.raises(GNNError, match="node_features"):
        small_model(nodes[:, :5], edges, edge_index)


def test_wrong_edge_width_is_rejected(small_model, single_track):
    nodes, edges, edge_index = graph_tensors(build_graph([single_track]))
    with pytest.raises(GNNError, match="edge_features"):
        small_model(nodes, edges[:, :2], edge_index)


def test_empty_graph_is_rejected(small_model):
    with pytest.raises(GNNError, match="empty graph"):
        small_model(
            torch.zeros((0, 16)), torch.zeros((0, 6)), torch.zeros((2, 0), dtype=torch.long)
        )


def test_layer_rejects_mismatched_edge_counts():
    layer = MessagePassingLayer(4, 6)
    with pytest.raises(GNNError, match="disagree"):
        layer(torch.zeros((2, 4)), torch.zeros((3, 6)), torch.zeros((2, 2), dtype=torch.long))


def test_layer_rejects_a_malformed_edge_index():
    layer = MessagePassingLayer(4, 6)
    with pytest.raises(GNNError, match="edge_index"):
        layer(torch.zeros((2, 4)), torch.zeros((2, 6)), torch.zeros((3, 2), dtype=torch.long))


# --- determinism ------------------------------------------------------------


def test_initialisation_is_deterministic_for_a_seed():
    first = WeatherGNN(ModelConfig(hidden_dim=8, seed=42))
    second = WeatherGNN(ModelConfig(hidden_dim=8, seed=42))
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_different_seeds_give_different_weights():
    first = WeatherGNN(ModelConfig(hidden_dim=8, seed=1))
    second = WeatherGNN(ModelConfig(hidden_dim=8, seed=2))
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters())
    )


def test_initialisation_does_not_depend_on_global_rng_state():
    torch.manual_seed(1)
    first = WeatherGNN(ModelConfig(hidden_dim=8, seed=99))
    torch.manual_seed(12345)
    _ = torch.randn(10)
    second = WeatherGNN(ModelConfig(hidden_dim=8, seed=99))
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_inference_is_deterministic(small_model, many_tracks):
    nodes, edges, edge_index = graph_tensors(build_graph(many_tracks))
    small_model.eval()
    with torch.no_grad():
        first = small_model(nodes, edges, edge_index)
        second = small_model(nodes, edges, edge_index)
    assert torch.equal(first, second)


def test_generator_is_reproducible():
    assert torch.equal(
        torch.rand(5, generator=init_generator(3)), torch.rand(5, generator=init_generator(3))
    )


# --- message passing --------------------------------------------------------


def test_layer_count_follows_the_configuration():
    assert WeatherGNN(ModelConfig(message_layers=3)).layer_count == 3


def test_messages_change_the_target_representation_only():
    """A node's output must depend on its neighbours; a source node must not."""
    torch.manual_seed(0)
    layer = MessagePassingLayer(4, 6)
    nodes = torch.randn(3, 4)
    empty = torch.zeros(0, 6)
    without = layer(nodes, empty, torch.zeros((2, 0), dtype=torch.long))
    linked = layer(nodes, torch.zeros(1, 6), torch.tensor([[0], [2]], dtype=torch.long))
    assert not torch.equal(without[2], linked[2])
    # Messages flow source -> target only, so the source is untouched.
    assert torch.equal(without[0], linked[0])


def test_isolated_nodes_keep_their_own_state():
    layer = MessagePassingLayer(4, 6)
    output = layer(torch.randn(2, 4), torch.zeros(0, 6), torch.zeros((2, 0), dtype=torch.long))
    assert output.shape == (2, 4)


def test_aggregation_is_a_mean_not_a_sum():
    """A duplicated message must not change the mean."""
    torch.manual_seed(0)
    layer = MessagePassingLayer(4, 6)
    nodes = torch.randn(2, 4)
    edges = torch.zeros(2, 6)
    one = layer(nodes, edges[:1], torch.tensor([[0], [1]], dtype=torch.long))
    two = layer(nodes, edges, torch.tensor([[0, 0], [1, 1]], dtype=torch.long))
    assert torch.allclose(one[1], two[1], atol=1e-6)


# --- loss -------------------------------------------------------------------


def test_loss_is_zero_for_a_perfect_prediction():
    targets = torch.tensor([[0.1, 0.2, -1.0], [0.0, 0.0, 0.0]])
    assert float(compute_loss(targets.clone(), targets)) == pytest.approx(0.0, abs=1e-9)


def test_loss_matches_the_documented_formula():
    config = ModelConfig(position_weight=1.0, intensity_weight=0.5)
    loss = compute_loss(torch.tensor([[0.2, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]), config)
    # float32 tolerance: 0.2**2 = 0.04 is not exact in single precision.
    assert float(loss) == pytest.approx(1.0 * 0.2**2 + 0.5 * 0.0, rel=1e-6)


def test_loss_weights_are_configurable():
    predictions, targets = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]])
    single = compute_loss(
        predictions, targets, ModelConfig(position_weight=1.0, intensity_weight=0.0)
    )
    double = compute_loss(
        predictions, targets, ModelConfig(position_weight=2.0, intensity_weight=0.0)
    )
    assert float(double) == pytest.approx(2.0 * float(single), rel=1e-6)


def test_loss_is_positive_for_a_wrong_prediction():
    assert float(compute_loss(torch.tensor([[1.0, 1.0, 1.0]]), torch.zeros(1, 3))) > 0.0


def test_loss_is_differentiable():
    predictions = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    compute_loss(predictions, torch.zeros(1, 3)).backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_loss_rejects_mismatched_shapes():
    with pytest.raises(GNNError, match="targets must match"):
        compute_loss(torch.zeros(2, 3), torch.zeros(3, 3))


def test_loss_rejects_an_empty_sample_set():
    with pytest.raises(GNNError, match="at least one sample"):
        compute_loss(torch.zeros(0, 3), torch.zeros(0, 3))


def test_loss_components_sum_to_the_weighted_total():
    predictions, targets = torch.tensor([[0.3, -0.2, 1.0]]), torch.tensor([[0.0, 0.0, 0.0]])
    components = loss_components(predictions, targets)
    config = ModelConfig()
    expected = config.position_weight * components["position_total"] + (
        config.intensity_weight * components["intensity"]
    )
    assert float(compute_loss(predictions, targets, config)) == pytest.approx(expected, rel=1e-6)


def test_model_is_small_enough_for_cpu_prototyping():
    """A prototype should stay well under 100k parameters."""
    assert sum(p.numel() for p in WeatherGNN().parameters()) < 100_000
