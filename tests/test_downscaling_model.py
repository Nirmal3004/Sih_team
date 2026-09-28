"""Model, residual-learning and output-constraint tests (SIH26078, Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from extreme_weather.downscaling.model import (
    MODEL_VERSION,
    DownscalingError,
    DownscalingModel,
    ModelConfig,
    ResidualBlock,
    init_generator,
)


@pytest.fixture
def tiny_model() -> DownscalingModel:
    """A small model so the tests stay fast on CPU."""
    return DownscalingModel(ModelConfig(hidden_channels=8, num_blocks=2))


# --- configuration ----------------------------------------------------------


def test_default_config_matches_the_feature_contract():
    from extreme_weather.downscaling.features import FEATURE_COUNT

    config = ModelConfig()
    assert config.in_channels == FEATURE_COUNT
    assert config.output_activation == "relu"


def test_config_rejects_invalid_architecture():
    with pytest.raises(ValueError, match="hidden_channels"):
        ModelConfig(hidden_channels=0)
    with pytest.raises(ValueError, match="num_blocks"):
        ModelConfig(num_blocks=0)


def test_only_relu_output_activation_is_supported():
    """The non-negativity constraint is a documented, enforced design choice."""
    with pytest.raises(ValueError, match="relu"):
        ModelConfig(output_activation="sigmoid")


def test_model_version_is_exposed():
    assert MODEL_VERSION.startswith("downscaling-cnn")


# --- shapes -----------------------------------------------------------------


def test_output_shape_is_batch_one_at_the_target_size(tiny_model):
    features = torch.randn(2, 10, 9, 9)
    output = tiny_model(features, target_shape=(21, 21))
    assert output.shape == (2, 1, 21, 21)


def test_output_defaults_to_the_input_size(tiny_model):
    features = torch.randn(1, 10, 9, 9)
    assert tiny_model(features).shape == (1, 1, 9, 9)


@pytest.mark.parametrize("shape", [(21, 21), (41, 33), (13, 9)])
def test_any_target_size_is_supported(tiny_model, shape):
    """The scale factor is read from the data, not assumed by the architecture."""
    output = tiny_model(torch.randn(1, 10, 9, 9), target_shape=shape)
    assert output.shape == (1, 1, *shape)


def test_model_handles_a_single_node_grid(tiny_model):
    assert tiny_model(torch.randn(1, 10, 1, 1), target_shape=(4, 4)).shape == (1, 1, 4, 4)


def test_model_rejects_bad_input():
    model = DownscalingModel(ModelConfig(hidden_channels=4, num_blocks=1))
    with pytest.raises(DownscalingError, match=r"\[B, C, H, W\]"):
        model(torch.randn(10, 9, 9))
    with pytest.raises(DownscalingError, match="channels"):
        model(torch.randn(1, 3, 9, 9))
    with pytest.raises(DownscalingError, match="Empty input"):
        model(torch.randn(0, 10, 9, 9))
    with pytest.raises(DownscalingError, match="Invalid target shape"):
        model(torch.randn(1, 10, 9, 9), target_shape=(0, 5))


# --- non-negativity ---------------------------------------------------------


def test_residual_is_non_negative(tiny_model):
    """ReLU on the output is the documented precipitation constraint."""
    output = tiny_model(torch.randn(4, 10, 9, 9), target_shape=(21, 21))
    assert (output >= 0.0).all()


def test_output_never_falls_below_the_baseline(tiny_model):
    """Because the residual is non-negative, output = baseline + residual >= baseline."""
    from extreme_weather.downscaling.dataset import bilinear_baseline

    baseline = bilinear_baseline(np.full((9, 9), 5.0), (21, 21))
    with torch.no_grad():
        residual = tiny_model(torch.randn(1, 10, 9, 9), target_shape=(21, 21))
    total = np.maximum(baseline + residual.numpy()[0, 0], 0.0)
    assert (total >= baseline - 1e-6).all()


# --- determinism ------------------------------------------------------------


def test_initialisation_is_deterministic():
    """Same config -> bit-identical weights, independent of global RNG state."""
    torch.manual_seed(1)
    first = DownscalingModel(ModelConfig(hidden_channels=8, num_blocks=2))
    torch.manual_seed(999)
    _ = torch.randn(50)
    second = DownscalingModel(ModelConfig(hidden_channels=8, num_blocks=2))
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_inference_is_deterministic(tiny_model):
    features = torch.randn(2, 10, 9, 9)
    tiny_model.eval()
    with torch.no_grad():
        first = tiny_model(features, target_shape=(21, 21))
        second = tiny_model(features, target_shape=(21, 21))
    assert torch.equal(first, second)


def test_generator_is_reproducible():
    assert torch.equal(
        torch.rand(4, generator=init_generator(5)), torch.rand(4, generator=init_generator(5))
    )


# --- residual block ---------------------------------------------------------


def test_residual_block_keeps_its_input_path():
    """A zeroed convolution pair must return the input unchanged."""
    block = ResidualBlock(4)
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv1.bias.zero_()
        block.conv2.weight.zero_()
        block.conv2.bias.zero_()
    x = torch.randn(2, 4, 5, 5)
    assert torch.allclose(block(x), x)


def test_model_stays_small_enough_for_cpu():
    assert DownscalingModel().parameter_count < 100_000
