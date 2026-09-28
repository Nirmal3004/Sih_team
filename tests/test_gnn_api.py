"""GNN API tests (SIH26078, Phase 3)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from extreme_weather.gnn.checkpoint import GNNCheckpoint, save_checkpoint
from extreme_weather.gnn.dataset import build_dataset
from extreme_weather.gnn.model import ModelConfig
from extreme_weather.gnn.service import GNNService
from extreme_weather.gnn.training import TrainingConfig, train
from extreme_weather.tracking.models import TrackResult
from extreme_weather.tracking.store import TrackStore

API = "/api/v1/gnn"


@pytest.fixture
def tracks_file(tmp_path, many_tracks):
    """A Phase 2 tracking file on disk."""
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
        model_config=ModelConfig(hidden_dim=8, seed=13),
        training_config=TrainingConfig(epochs=10, seed=13),
    )
    checkpoint = GNNCheckpoint(
        model_config=result.model.config,
        feature_config=dataset.feature_config,
        training={
            "seed": 13,
            "epochs": 10,
            "train_samples": len(dataset.split.train),
            "best_validation_loss": 0.25,
            "best_epoch": 4,
        },
        parameters={n: t.detach().clone() for n, t in result.model.state_dict().items()},
    )
    return save_checkpoint(checkpoint, tmp_path / "best.json")


@pytest.fixture
def client_factory():
    """Build a client whose GNN dependency points at the given files."""
    from extreme_weather.api.main import create_app

    created: list[TestClient] = []

    def build(model_file, tracks_file) -> TestClient:
        from extreme_weather.api.routers.gnn import get_gnn_service

        application = create_app()
        application.dependency_overrides[get_gnn_service] = lambda: GNNService(
            checkpoint_path=model_file, track_store=TrackStore(tracks_file)
        )
        test_client = TestClient(application)
        created.append(test_client)
        return test_client

    yield build
    for test_client in created:
        test_client.close()


# --- status -----------------------------------------------------------------


def test_status_reports_a_missing_model(tmp_path, tracks_file, client_factory):
    response = client_factory(tmp_path / "absent.json", tracks_file).get(f"{API}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["model_version"] is None
    assert "train_gnn.py" in payload["message"]


def test_status_reports_a_trained_model(model_file, tracks_file, client_factory):
    response = client_factory(model_file, tracks_file).get(f"{API}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["model_version"]
    assert payload["validation_loss"] == pytest.approx(0.25)
    assert payload["best_epoch"] == 4
    assert payload["training"]["seed"] == 13
    assert "Not an Operational Forecast" in payload["notice"]


def test_status_and_model_never_fail_without_a_model(tmp_path, tracks_file, client_factory):
    client = client_factory(tmp_path / "absent.json", tracks_file)
    assert client.get(f"{API}/status").status_code == 200
    assert client.get(f"{API}/model").status_code == 200


# --- model info -------------------------------------------------------------


def test_model_endpoint_describes_the_architecture(model_file, tracks_file, client_factory):
    payload = client_factory(model_file, tracks_file).get(f"{API}/model").json()
    assert payload["message_layers"] == 2
    assert payload["hidden_dim"] == 8
    assert len(payload["node_feature_names"]) == 16
    assert len(payload["edge_feature_names"]) == 6
    assert payload["target_names"] == ["delta_latitude", "delta_longitude", "delta_intensity"]
    assert "normalization" in payload
    assert "hidden_dim" in payload["hyperparameters"]
    assert "Not an Operational Forecast" in payload["notice"]


def test_model_endpoint_works_untrained(tmp_path, tracks_file, client_factory):
    payload = client_factory(tmp_path / "absent.json", tracks_file).get(f"{API}/model").json()
    assert payload["hyperparameters"]["available"] is False


# --- prediction -------------------------------------------------------------


def test_predict_returns_a_prediction(model_file, tracks_file, many_tracks, client_factory):
    response = client_factory(model_file, tracks_file).post(
        f"{API}/predict", json={"track_id": many_tracks[0].track_id}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["track_id"] == many_tracks[0].track_id
    assert payload["predicted_latitude"] == pytest.approx(
        payload["current_latitude"] + payload["delta_latitude"]
    )
    assert payload["model_version"]
    assert "Not an Operational Forecast" in payload["notice"]


def test_predict_is_deterministic_over_http(model_file, tracks_file, many_tracks, client_factory):
    client = client_factory(model_file, tracks_file)
    body = {"track_id": many_tracks[0].track_id}
    first = client.post(f"{API}/predict", json=body).json()
    second = client.post(f"{API}/predict", json=body).json()
    assert first == second


def test_unknown_track_returns_404(model_file, tracks_file, client_factory):
    response = client_factory(model_file, tracks_file).post(
        f"{API}/predict", json={"track_id": "TRK-NOPE-001"}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_missing_checkpoint_returns_503(tmp_path, tracks_file, many_tracks, client_factory):
    response = client_factory(tmp_path / "absent.json", tracks_file).post(
        f"{API}/predict", json={"track_id": many_tracks[0].track_id}
    )
    assert response.status_code == 503
    assert "train_gnn.py" in response.json()["detail"]


def test_missing_tracking_file_returns_422(model_file, tmp_path, many_tracks, client_factory):
    response = client_factory(model_file, tmp_path / "absent.json").post(
        f"{API}/predict", json={"track_id": many_tracks[0].track_id}
    )
    assert response.status_code == 422
    assert "track_events.py" in response.json()["detail"]


def test_empty_track_id_returns_422(model_file, tracks_file, client_factory):
    response = client_factory(model_file, tracks_file).post(
        f"{API}/predict", json={"track_id": ""}
    )
    assert response.status_code == 422


def test_missing_body_field_returns_422(model_file, tracks_file, client_factory):
    response = client_factory(model_file, tracks_file).post(f"{API}/predict", json={})
    assert response.status_code == 422


def test_unknown_body_field_returns_422(model_file, tracks_file, client_factory):
    response = client_factory(model_file, tracks_file).post(
        f"{API}/predict", json={"track_id": "TRK-A", "unexpected": 1}
    )
    assert response.status_code == 422


# --- coexistence with the earlier phases ------------------------------------


def test_gnn_routes_do_not_disturb_earlier_endpoints(
    model_file, tracks_file, many_tracks, client_factory
):
    """The Phase 3 router must not shadow the Phase 1/2 endpoints."""
    client = client_factory(model_file, tracks_file)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/tracks").status_code == 200
    assert client.get(f"{API}/status").status_code == 200
    # The GNN router must not shadow the tracks routes: an unknown id is a 404
    # from the tracks router (a routing clash would produce a different error).
    assert client.get("/api/v1/tracks/TRK-SYN-000").status_code == 404


def test_openapi_documents_the_gnn_endpoints(model_file, tracks_file, client_factory):
    paths = client_factory(model_file, tracks_file).get("/openapi.json").json()["paths"]
    assert f"{API}/status" in paths
    assert f"{API}/model" in paths
    assert f"{API}/predict" in paths
