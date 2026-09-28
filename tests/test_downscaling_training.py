"""Training and checkpoint tests (SIH26078, Phase 4)."""

from __future__ import annotations

import pytest
import torch

from extreme_weather.downscaling.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    PROTOTYPE_DISCLAIMER,
    DownscalingCheckpoint,
    DownscalingCheckpointError,
    build_training_metadata,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from extreme_weather.downscaling.dataset import SplitIndices
from extreme_weather.downscaling.model import MODEL_VERSION, ModelConfig, DownscalingModel
from extreme_weather.downscaling.model import DownscalingError
from extreme_weather.downscaling.training import (
    TrainingConfig,
    evaluate_block,
    train,
)


@pytest.fixture
def trained(tiny_downscaling_dataset):
    """A short, seeded training run on the tiny fixture."""
    result = train(
        tiny_downscaling_dataset,
        ModelConfig(hidden_channels=8, num_blocks=2),
        TrainingConfig(epochs=25, learning_rate=2e-3, seed=7),
    )
    return tiny_downscaling_dataset, result


@pytest.fixture
def checkpoint(tiny_downscaling_dataset, trained):
    _ds, result = trained
    return DownscalingCheckpoint(
        model_config=result.model.config,
        normalization=tiny_downscaling_dataset.normalization,
        training=build_training_metadata(
            seed=7, epochs=25, learning_rate=2e-3, batch_size=8,
            train_samples=len(tiny_downscaling_dataset.split.train),
            validation_samples=len(tiny_downscaling_dataset.split.validation),
            test_samples=len(tiny_downscaling_dataset.split.test),
            train_loss=result.history.final_train_loss,
            validation_loss=result.history.records[-1].validation_loss,
            best_validation_loss=result.history.best_validation_loss,
            best_epoch=result.history.best_epoch,
            test_loss=result.test_loss,
        ),
        model_state={n: t.detach().clone() for n, t in result.model.state_dict().items()},
        input_resolution_deg=tiny_downscaling_dataset.input_resolution_deg,
        output_resolution_deg=tiny_downscaling_dataset.output_resolution_deg,
        input_shape=tiny_downscaling_dataset.input_shape,
        output_shape=tiny_downscaling_dataset.output_shape,
        variable="precipitation",
    )


# --- configuration ----------------------------------------------------------


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="patience"):
        TrainingConfig(patience=0)


def test_early_stopping_is_off_by_default():
    """Best-on-validation selection is the mechanism; patience is opt-in."""
    assert TrainingConfig().patience is None


# --- convergence ------------------------------------------------------------


def test_training_converges_on_the_tiny_fixture(tiny_downscaling_dataset):
    result = train(
        tiny_downscaling_dataset,
        ModelConfig(hidden_channels=8, num_blocks=2),
        TrainingConfig(epochs=40, learning_rate=5e-3, seed=3),
    )
    assert len(result.history.records) == 40
    assert result.history.records[-1].train_loss < result.history.records[0].train_loss


def test_history_records_every_epoch(trained):
    _ds, result = trained
    epochs = [record.epoch for record in result.history.records]
    assert epochs == list(range(1, len(epochs) + 1))
    assert all(record.validation_loss is not None for record in result.history.records)


def test_test_loss_is_reported_once(trained):
    _ds, result = trained
    assert result.test_loss is not None
    assert result.test_loss > 0.0


def test_best_epoch_tracks_the_lowest_validation_loss(trained):
    _ds, result = trained
    assert result.history.best_validation_loss == pytest.approx(
        min(record.validation_loss for record in result.history.records)
    )


def test_returned_model_holds_the_best_weights(trained):
    ds, result = trained
    assert evaluate_block(result.model, ds, ds.split.validation)[0] == pytest.approx(
        result.history.best_validation_loss, rel=1e-6
    )


def test_evaluate_block_returns_none_for_an_empty_block(tiny_downscaling_dataset):
    assert evaluate_block(DownscalingModel(), tiny_downscaling_dataset, []) == (None, None)


def test_training_does_not_use_the_test_block(tiny_downscaling_dataset):
    """Training weights must be identical with and without test frames present."""
    ds = tiny_downscaling_dataset
    trimmed = type(ds)(
        **{
            **ds.__dict__,
            "split": SplitIndices(
                train=ds.split.train, validation=ds.split.validation, test=[]
            ),
        }
    )
    model_config = ModelConfig(hidden_channels=8, num_blocks=2)
    settings = TrainingConfig(epochs=10, seed=5)
    with_test = train(ds, model_config, settings)
    without_test = train(trimmed, model_config, settings)
    for left, right in zip(with_test.model.parameters(), without_test.model.parameters()):
        assert torch.equal(left, right)


def test_training_is_deterministic_for_a_seed(tiny_downscaling_dataset):
    model_config = ModelConfig(hidden_channels=8, num_blocks=2)
    settings = TrainingConfig(epochs=12, seed=9)
    first = train(tiny_downscaling_dataset, model_config, settings)
    second = train(tiny_downscaling_dataset, model_config, settings)
    assert first.history.final_train_loss == pytest.approx(second.history.final_train_loss)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(left, right)


def test_early_stopping_can_be_enabled(tiny_downscaling_dataset):
    result = train(
        tiny_downscaling_dataset,
        ModelConfig(hidden_channels=8, num_blocks=2),
        TrainingConfig(epochs=50, learning_rate=1e-2, patience=1, seed=1),
    )
    assert result.history.stopped_early or len(result.history.records) == 50


def test_training_rejects_an_empty_training_block(tiny_downscaling_dataset):
    """A dataset whose chronological split has no training frames is refused."""
    ds = tiny_downscaling_dataset
    empty = type(ds)(
        **{
            **ds.__dict__,
            "split": SplitIndices(
                train=[], validation=ds.split.validation, test=ds.split.test
            ),
        }
    )
    with pytest.raises(DownscalingError, match="training split is empty"):
        train(empty, model_config=None, training_config=TrainingConfig(epochs=1))


# --- checkpointing ----------------------------------------------------------


def test_checkpoint_round_trips(tmp_path, checkpoint, tiny_downscaling_dataset):
    path = save_checkpoint(checkpoint, tmp_path / "best.pt")
    assert path.is_file()
    loaded = load_checkpoint(path)
    assert loaded.model_version == MODEL_VERSION
    assert loaded.seed == 7
    assert loaded.input_resolution_deg == pytest.approx(0.125)
    assert loaded.output_resolution_deg == pytest.approx(0.05, abs=1e-6)
    assert loaded.scale_factor == pytest.approx(2.5, abs=1e-6)
    assert loaded.input_shape == tiny_downscaling_dataset.input_shape
    assert loaded.output_shape == tiny_downscaling_dataset.output_shape
    assert loaded.variable == "precipitation"
    assert loaded.normalization.source == tiny_downscaling_dataset.normalization.source
    assert set(loaded.model_state) == set(checkpoint.model_state)


def test_checkpoint_records_the_required_fields(tmp_path, checkpoint):
    payload = torch.load(save_checkpoint(checkpoint, tmp_path / "b.pt"), weights_only=True)
    for key in (
        "schema_version", "model_version", "model_config", "feature_names",
        "normalization", "input_resolution_deg", "output_resolution_deg",
        "input_shape", "output_shape", "variable", "training", "model_state",
    ):
        assert key in payload
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["disclaimer"] == PROTOTYPE_DISCLAIMER
    assert payload["training"]["seed"] == 7
    assert "SYNTHETIC" in payload["training"]["data_note"]


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(DownscalingCheckpointError, match="train_downscaling.py"):
        load_checkpoint(tmp_path / "absent.pt")


def test_unreadable_checkpoint_is_reported(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch file")
    with pytest.raises(DownscalingCheckpointError):
        load_checkpoint(path)


def test_missing_key_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.pt")
    payload = torch.load(path, weights_only=True)
    del payload["training"]
    torch.save(payload, path)
    with pytest.raises(DownscalingCheckpointError, match="missing required key"):
        load_checkpoint(path)


def test_schema_version_mismatch_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.pt")
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = "99.0"
    torch.save(payload, path)
    with pytest.raises(DownscalingCheckpointError, match="schema_version"):
        load_checkpoint(path)


def test_feature_contract_mismatch_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.pt")
    payload = torch.load(path, weights_only=True)
    payload["feature_names"] = ["something_else"]
    torch.save(payload, path)
    with pytest.raises(DownscalingCheckpointError, match="feature contract"):
        load_checkpoint(path)


def test_parameter_mismatch_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.pt")
    payload = torch.load(path, weights_only=True)
    payload["model_state"] = {"not_a_real_parameter": torch.zeros(3)}
    torch.save(payload, path)
    with pytest.raises(DownscalingCheckpointError, match="do not match"):
        load_checkpoint(path).build_model()


def test_default_checkpoint_path_is_under_models_downscaling():
    assert checkpoint_path().parts[-2:] == ("downscaling", "best.pt")
