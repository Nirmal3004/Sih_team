"""Inference, decoding and service tests (SIH26078, Phase 3)."""

from __future__ import annotations

import json

import pytest

from extreme_weather.gnn.checkpoint import GNNCheckpoint, load_checkpoint, save_checkpoint
from extreme_weather.gnn.dataset import build_dataset
from extreme_weather.gnn.model import ModelConfig, WeatherGNN
from extreme_weather.gnn.service import (
    GNNInvalidTrackError,
    GNNModelUnavailableError,
    GNNService,
    GNNUnknownTrackError,
    decode_prediction,
)
from extreme_weather.gnn.training import TrainingConfig, train
from extreme_weather.tracking.models import TrackResult
from extreme_weather.tracking.store import TrackStore

# --- decoding ---------------------------------------------------------------


def test_decode_adds_the_deltas_to_the_current_state():
    decoded = decode_prediction(13.0, 77.0, 50.0, [0.1, 0.2, -5.0])
    assert decoded["predicted_latitude"] == pytest.approx(13.1)
    assert decoded["predicted_longitude"] == pytest.approx(77.2)
    assert decoded["predicted_intensity"] == pytest.approx(45.0)


def test_decode_returns_the_raw_deltas_unchanged():
    decoded = decode_prediction(13.0, 77.0, 50.0, [0.1, 0.2, -5.0])
    assert decoded["delta_latitude"] == pytest.approx(0.1)
    assert decoded["delta_longitude"] == pytest.approx(0.2)
    assert decoded["delta_intensity"] == pytest.approx(-5.0)


def test_decode_clamps_an_out_of_range_latitude():
    decoded = decode_prediction(89.0, 0.0, 10.0, [50.0, 0.0, 0.0])
    assert decoded["predicted_latitude"] == 90.0
    # The raw delta is still reported, so the caller can see the model output.
    assert decoded["delta_latitude"] == pytest.approx(50.0)


def test_decode_clamps_an_out_of_range_longitude():
    assert decode_prediction(0.0, 179.0, 10.0, [0.0, 40.0, 0.0])["predicted_longitude"] == 180.0


def test_decode_reports_an_implied_speed_and_direction():
    decoded = decode_prediction(13.0, 77.0, 50.0, [0.1, 0.0, 0.0], step_hours=1.0)
    assert decoded["predicted_speed_kmh"] > 0.0
    assert decoded["predicted_direction"] in ("N", "NNE", "NE", "ENE", "E")


def test_decode_of_a_stationary_prediction_is_calm():
    decoded = decode_prediction(13.0, 77.0, 50.0, [0.0, 0.0, 0.0])
    assert decoded["predicted_speed_kmh"] == pytest.approx(0.0)
    assert decoded["predicted_direction"] == "CALM"


def test_decode_rejects_too_few_deltas():
    with pytest.raises(GNNInvalidTrackError, match="Expected 3 deltas"):
        decode_prediction(13.0, 77.0, 50.0, [0.1, 0.2])


# --- service fixtures -------------------------------------------------------


@pytest.fixture
def tracks_file(tmp_path, many_tracks):
    """A Phase 2 tracking file on disk plus a store pointing at it."""
    result = TrackResult(
        tracks=tuple(many_tracks), event_count=40, matched_event_count=36, generated_at=None
    )
    path = tmp_path / "tracks.json"
    TrackStore(path).save(result)
    return path


@pytest.fixture
def model_file(tmp_path, many_tracks):
    """A trained, saved checkpoint on disk."""
    dataset = build_dataset(many_tracks)
    result = train(
        batch=dataset.batch,
        train_indices=dataset.split.train,
        validation_indices=dataset.split.validation,
        model_config=ModelConfig(hidden_dim=8, seed=21),
        training_config=TrainingConfig(epochs=10, seed=21),
    )
    checkpoint = GNNCheckpoint(
        model_config=result.model.config,
        feature_config=dataset.feature_config,
        training={"seed": 21, "epochs": 10, "best_validation_loss": 0.5, "best_epoch": 3},
        parameters={n: t.detach().clone() for n, t in result.model.state_dict().items()},
    )
    return save_checkpoint(checkpoint, tmp_path / "best.json")


@pytest.fixture
def service(model_file, tracks_file):
    """A fully wired service: a real checkpoint and a real tracking file."""
    return GNNService(checkpoint_path=model_file, track_store=TrackStore(tracks_file))


# --- prediction -------------------------------------------------------------


def test_predict_returns_the_documented_fields(service, many_tracks):
    prediction = service.predict(many_tracks[0].track_id)
    assert prediction.track_id == many_tracks[0].track_id
    assert prediction.current_latitude == many_tracks[0].points[-1].latitude
    assert prediction.current_intensity == many_tracks[0].points[-1].intensity
    assert prediction.model_version
    assert prediction.notice


def test_predict_uses_the_last_observed_state(service, many_tracks):
    prediction = service.predict(many_tracks[0].track_id)
    last = many_tracks[0].points[-1]
    assert prediction.timestamp == last.timestamp
    assert prediction.current_latitude == pytest.approx(last.latitude)
    assert prediction.current_longitude == pytest.approx(last.longitude)


def test_predicted_state_is_current_plus_delta(service, many_tracks):
    prediction = service.predict(many_tracks[0].track_id)
    assert prediction.predicted_latitude == pytest.approx(
        prediction.current_latitude + prediction.delta_latitude
    )
    assert prediction.predicted_longitude == pytest.approx(
        prediction.current_longitude + prediction.delta_longitude
    )
    assert prediction.predicted_intensity == pytest.approx(
        prediction.current_intensity + prediction.delta_intensity
    )


def test_predicted_position_stays_in_valid_ranges(service, many_tracks):
    prediction = service.predict(many_tracks[0].track_id)
    assert -90.0 <= prediction.predicted_latitude <= 90.0
    assert -180.0 <= prediction.predicted_longitude <= 180.0


def test_prediction_is_deterministic(service, many_tracks):
    first = service.predict(many_tracks[0].track_id)
    second = service.predict(many_tracks[0].track_id)
    assert first.model_dump() == second.model_dump()


def test_every_track_can_be_predicted(service, many_tracks):
    for track in many_tracks:
        assert service.predict(track.track_id).track_id == track.track_id


def test_prediction_reports_a_node_index(service, many_tracks):
    prediction = service.predict(many_tracks[0].track_id)
    assert prediction.node_index is not None
    assert prediction.node_index >= 0


# --- error handling ---------------------------------------------------------


def test_unknown_track_is_rejected(service):
    with pytest.raises(GNNUnknownTrackError, match="not found"):
        service.predict("TRK-DOES-NOT-EXIST-001")


def test_missing_checkpoint_is_reported(tmp_path, tracks_file):
    service = GNNService(
        checkpoint_path=tmp_path / "absent.json", track_store=TrackStore(tracks_file)
    )
    with pytest.raises(GNNModelUnavailableError, match="train_gnn.py"):
        service.predict("TRK-ANY")


def test_empty_track_id_is_rejected(service):
    with pytest.raises(GNNInvalidTrackError, match="non-empty string"):
        service.predict("   ")


def test_missing_tracking_file_is_reported(model_file, tmp_path):
    service = GNNService(
        checkpoint_path=model_file, track_store=TrackStore(tmp_path / "absent.json")
    )
    with pytest.raises(GNNInvalidTrackError, match="track_events.py"):
        service.predict("TRK-ANY")


def test_corrupt_tracking_file_is_reported(model_file, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    service = GNNService(checkpoint_path=model_file, track_store=TrackStore(path))
    with pytest.raises(GNNInvalidTrackError):
        service.predict("TRK-ANY")


# --- status and model info --------------------------------------------------


def test_status_reports_a_missing_model_as_unavailable(tmp_path, tracks_file):
    service = GNNService(
        checkpoint_path=tmp_path / "absent.json", track_store=TrackStore(tracks_file)
    )
    status = service.status()
    assert status.available is False
    assert status.model_version is None
    assert "train_gnn.py" in status.message


def test_status_reports_a_trained_model(service):
    status = service.status()
    assert status.available is True
    assert status.model_version
    assert status.validation_loss == pytest.approx(0.5)
    assert status.training["seed"] == 21
    assert "pytorch" in status.backend


def test_has_model_matches_the_status(service, tmp_path, tracks_file):
    assert service.has_model() is True
    missing = GNNService(
        checkpoint_path=tmp_path / "absent.json", track_store=TrackStore(tracks_file)
    )
    assert missing.has_model() is False


def test_model_info_describes_the_architecture(service):
    info = service.model_info()
    assert info.message_layers == 2
    assert info.hidden_dim == 8
    assert len(info.node_feature_names) == 16
    assert len(info.edge_feature_names) == 6
    assert info.target_names == ["delta_latitude", "delta_longitude", "delta_intensity"]
    assert "mean" in info.aggregation
    assert info.normalization["method"]


def test_model_info_works_without_a_checkpoint(tmp_path, tracks_file):
    service = GNNService(
        checkpoint_path=tmp_path / "absent.json", track_store=TrackStore(tracks_file)
    )
    info = service.model_info()
    assert info.hyperparameters["available"] is False
    assert info.node_feature_names


def test_every_response_carries_the_prototype_notice(service):
    assert "Not an Operational Forecast" in service.status().notice
    assert "Not an Operational Forecast" in service.model_info().notice


def test_model_can_be_rebuilt_from_the_checkpoint(model_file):
    rebuilt = load_checkpoint(model_file).build_model()
    assert isinstance(rebuilt, WeatherGNN)
    assert rebuilt.config.hidden_dim == 8


def test_tracks_file_is_not_modified_by_prediction(service, tracks_file):
    before = tracks_file.read_text(encoding="utf-8")
    service.predict("TRK-SYN-000")
    assert tracks_file.read_text(encoding="utf-8") == before
    json.loads(before)  # still valid JSON
