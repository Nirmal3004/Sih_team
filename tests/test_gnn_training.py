"""Training and checkpoint tests (SIH26078, Phase 3)."""

from __future__ import annotations

import json

import pytest

from extreme_weather.gnn.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    PROTOTYPE_DISCLAIMER,
    CheckpointError,
    GNNCheckpoint,
    build_training_metadata,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from extreme_weather.gnn.dataset import build_dataset
from extreme_weather.gnn.graph import GraphBatch
from extreme_weather.gnn.model import MODEL_VERSION, GNNError, ModelConfig, WeatherGNN
from extreme_weather.gnn.training import TrainingConfig, evaluate, predict_all, train


@pytest.fixture
def trained(many_tracks):
    """A short, seeded training run on the tiny synthetic fixture."""
    dataset = build_dataset(many_tracks)
    return dataset, train(
        batch=dataset.batch,
        train_indices=dataset.split.train,
        validation_indices=dataset.split.validation,
        model_config=ModelConfig(hidden_dim=8, seed=11),
        training_config=TrainingConfig(epochs=60, learning_rate=0.05, seed=11),
    )


# --- configuration ----------------------------------------------------------


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0.0)


# --- convergence ------------------------------------------------------------


def test_training_converges_on_the_tiny_fixture(many_tracks):
    """40 epochs on a 21-sample fixture must cut the loss substantially."""
    dataset = build_dataset(many_tracks)
    result = train(
        dataset.batch,
        dataset.split.train,
        validation_indices=dataset.split.validation,
        model_config=ModelConfig(hidden_dim=8, seed=5),
        training_config=TrainingConfig(epochs=40, learning_rate=0.05, seed=5),
    )
    history = result.history
    assert len(history.records) == 40
    assert history.records[-1].train_loss < 0.5 * history.records[0].train_loss


def test_history_records_every_epoch(trained):
    _dataset, result = trained
    epochs = [record.epoch for record in result.history.records]
    assert epochs == list(range(1, len(epochs) + 1))
    assert all(record.validation_loss is not None for record in result.history.records)


def test_test_loss_is_reported_only_once(many_tracks):
    dataset = build_dataset(many_tracks)
    result = train(
        dataset.batch,
        dataset.split.train,
        dataset.split.validation,
        model_config=ModelConfig(hidden_dim=8, seed=3),
        training_config=TrainingConfig(epochs=10, seed=3),
        test_indices=dataset.split.test,
    )
    assert result.test_loss is not None
    assert result.test_loss > 0.0


def test_test_loss_is_none_when_the_block_is_empty(many_tracks):
    dataset = build_dataset(many_tracks)
    result = train(
        dataset.batch,
        dataset.split.train,
        dataset.split.validation,
        model_config=ModelConfig(hidden_dim=8, seed=3),
        training_config=TrainingConfig(epochs=5, seed=3),
    )
    assert result.test_loss is None


# --- best-model selection ---------------------------------------------------


def test_best_epoch_tracks_the_lowest_validation_loss(trained):
    _dataset, result = trained
    assert result.history.best_validation_loss == pytest.approx(
        min(record.validation_loss for record in result.history.records)
    )


def test_returned_model_holds_the_best_weights(trained):
    dataset, result = trained
    assert evaluate(result.model, dataset.batch, dataset.split.validation) == pytest.approx(
        result.history.best_validation_loss, rel=1e-6
    )


def test_training_does_not_select_on_the_test_block(many_tracks):
    """Passing the test block must not change the selected weights."""
    dataset = build_dataset(many_tracks)
    model_config = ModelConfig(hidden_dim=8, seed=9)
    settings = TrainingConfig(epochs=20, seed=9)
    with_test = train(
        dataset.batch, dataset.split.train, dataset.split.validation,
        model_config, settings, test_indices=dataset.split.test,
    )
    without_test = train(
        dataset.batch, dataset.split.train, dataset.split.validation, model_config, settings
    )
    for left, right in zip(with_test.model.parameters(), without_test.model.parameters()):
        assert left.detach().numpy().tolist() == right.detach().numpy().tolist()


# --- determinism ------------------------------------------------------------


def test_training_is_deterministic_for_a_seed(many_tracks):
    dataset = build_dataset(many_tracks)
    config = ModelConfig(hidden_dim=8, seed=4)
    settings = TrainingConfig(epochs=15, seed=4)
    first = train(dataset.batch, dataset.split.train, dataset.split.validation, config, settings)
    second = train(dataset.batch, dataset.split.train, dataset.split.validation, config, settings)
    assert first.history.final_train_loss == pytest.approx(second.history.final_train_loss)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        assert left.detach().numpy().tolist() == right.detach().numpy().tolist()


# --- guards -----------------------------------------------------------------


def test_training_rejects_an_empty_graph():
    with pytest.raises(GNNError, match="empty graph"):
        train(GraphBatch(), [], [], model_config=None, training_config=TrainingConfig(epochs=1))


def test_training_rejects_an_empty_training_block(many_tracks):
    dataset = build_dataset(many_tracks)
    with pytest.raises(GNNError, match="No training samples"):
        train(
            dataset.batch,
            [],
            dataset.split.validation,
            model_config=None,
            training_config=TrainingConfig(epochs=1),
        )


def test_predict_all_rejects_an_empty_graph():
    with pytest.raises(GNNError, match="empty graph"):
        predict_all(WeatherGNN(), GraphBatch())


def test_evaluate_returns_none_for_an_empty_block(many_tracks):
    dataset = build_dataset(many_tracks)
    assert evaluate(WeatherGNN(ModelConfig(hidden_dim=8)), dataset.batch, []) is None


# --- checkpointing ----------------------------------------------------------


@pytest.fixture
def checkpoint(trained) -> GNNCheckpoint:
    _dataset, result = trained
    return GNNCheckpoint(
        model_config=result.model.config,
        feature_config=_dataset.feature_config,
        training=build_training_metadata(
            seed=11,
            epochs=60,
            learning_rate=0.05,
            train_samples=len(_dataset.split.train),
            validation_samples=len(_dataset.split.validation),
            test_samples=len(_dataset.split.test),
            train_loss=result.history.final_train_loss,
            validation_loss=result.history.records[-1].validation_loss,
            best_validation_loss=result.history.best_validation_loss,
            best_epoch=result.history.best_epoch,
        ),
        parameters={n: t.detach().clone() for n, t in result.model.state_dict().items()},
    )


def test_checkpoint_round_trips(tmp_path, checkpoint, trained):
    dataset, result = trained
    path = save_checkpoint(checkpoint, tmp_path / "best.json")
    assert path.is_file()
    loaded = load_checkpoint(path)

    assert loaded.model_version == MODEL_VERSION
    assert loaded.seed == 11
    assert loaded.best_validation_loss == pytest.approx(checkpoint.best_validation_loss)
    assert set(loaded.parameters) == set(checkpoint.parameters)

    # The rebuilt model must reproduce the original predictions.
    original = predict_all(result.model, dataset.batch)
    restored = predict_all(loaded.build_model(), dataset.batch)
    for left, right in zip(original.numpy().tolist(), restored.numpy().tolist()):
        assert left == pytest.approx(right, abs=1e-6)


def test_checkpoint_records_the_feature_contract(tmp_path, checkpoint):
    from extreme_weather.gnn.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES

    payload = json.loads(save_checkpoint(checkpoint, tmp_path / "b.json").read_text())
    assert payload["node_feature_names"] == list(NODE_FEATURE_NAMES)
    assert payload["edge_feature_names"] == list(EDGE_FEATURE_NAMES)
    assert payload["target_names"] == list(TARGET_NAMES)
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["disclaimer"] == PROTOTYPE_DISCLAIMER


def test_checkpoint_records_the_training_metadata(tmp_path, checkpoint):
    payload = json.loads(save_checkpoint(checkpoint, tmp_path / "b.json").read_text())
    training = payload["training"]
    assert training["seed"] == 11
    assert training["epochs"] == 60
    assert training["train_samples"] > 0
    assert training["best_epoch"] > 0
    assert "synthetic" in training["data_note"]


def test_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="train_gnn.py"):
        load_checkpoint(tmp_path / "absent.json")


def test_corrupt_json_is_reported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(CheckpointError, match="not valid JSON"):
        load_checkpoint(path)


def test_non_object_checkpoint_is_reported(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(CheckpointError, match="must contain a JSON object"):
        load_checkpoint(path)


def test_missing_key_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.json")
    payload = json.loads(path.read_text())
    del payload["training"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointError, match="missing required key"):
        load_checkpoint(path)


def test_feature_contract_mismatch_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.json")
    payload = json.loads(path.read_text())
    payload["node_feature_names"] = ["something_else"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointError, match="incompatible node_feature_names"):
        load_checkpoint(path)


def test_schema_version_mismatch_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.json")
    payload = json.loads(path.read_text())
    payload["schema_version"] = "99.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointError, match="schema_version"):
        load_checkpoint(path)


def test_parameter_mismatch_is_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.json")
    payload = json.loads(path.read_text())
    payload["parameters"] = {"not_a_real_parameter": [1.0, 2.0]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(path).build_model()


def test_empty_parameters_are_reported(tmp_path, checkpoint):
    path = save_checkpoint(checkpoint, tmp_path / "b.json")
    payload = json.loads(path.read_text())
    payload["parameters"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointError, match="no model parameters"):
        load_checkpoint(path)


def test_default_checkpoint_path_is_under_models_gnn():
    assert checkpoint_path().parts[-2:] == ("gnn", "best.json")
