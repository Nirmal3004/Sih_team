"""AI downscaling API tests (SIH26078, Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from extreme_weather.downscaling.checkpoint import DownscalingCheckpoint, save_checkpoint
from extreme_weather.downscaling.model import ModelConfig
from extreme_weather.downscaling.service import DownscalingService
from extreme_weather.downscaling.training import TrainingConfig, train

from .downscaling_fixtures import FINE_RESOLUTION, make_coarse_field

API = "/api/v1/downscaling"


@pytest.fixture
def coarse_file(tmp_path):
    """A tiny processed NetCDF with a precipitation field."""
    field, latitude, longitude, timestamps = make_coarse_field(time_steps=8, size=9)
    path = tmp_path / "coarse.nc"
    xr.Dataset(
        {"precipitation": (("time", "latitude", "longitude"), field)},
        coords={
            "time": np.array(timestamps, dtype="datetime64[ns]"),
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_netcdf(path)
    return path


@pytest.fixture
def model_file(tmp_path, tiny_downscaling_dataset):
    """A trained checkpoint on disk."""
    result = train(
        tiny_downscaling_dataset,
        ModelConfig(hidden_channels=8, num_blocks=2),
        TrainingConfig(epochs=8, seed=3),
    )
    checkpoint = DownscalingCheckpoint(
        model_config=result.model.config,
        normalization=tiny_downscaling_dataset.normalization,
        training={"seed": 3, "epochs": 8, "best_validation_loss": 0.2, "best_epoch": 2},
        model_state={n: t.detach().clone() for n, t in result.model.state_dict().items()},
        input_resolution_deg=tiny_downscaling_dataset.input_resolution_deg,
        output_resolution_deg=tiny_downscaling_dataset.output_resolution_deg,
        input_shape=tiny_downscaling_dataset.input_shape,
        output_shape=tiny_downscaling_dataset.output_shape,
        variable="precipitation",
    )
    return save_checkpoint(checkpoint, tmp_path / "best.pt")


@pytest.fixture
def client_factory(coarse_file):
    """Build clients whose downscaling service points at the given files."""
    from extreme_weather.api.main import create_app
    from extreme_weather.api.routers.downscaling import get_downscaling_service

    created: list[TestClient] = []

    def build(model_path, dataset_path=coarse_file) -> TestClient:
        application = create_app()
        application.dependency_overrides[get_downscaling_service] = lambda: DownscalingService(
            checkpoint_path=model_path, dataset_path=dataset_path
        )
        test_client = TestClient(application)
        created.append(test_client)
        return test_client

    yield build
    for test_client in created:
        test_client.close()


# --- status and model -------------------------------------------------------


def test_status_reports_a_missing_model(tmp_path, client_factory):
    payload = client_factory(tmp_path / "absent.pt").get(f"{API}/status").json()
    assert payload["available"] is False
    assert payload["model_version"] is None
    assert "train_downscaling.py" in payload["message"]
    assert "Not an Operational Forecast" in payload["notice"]


def test_status_reports_a_trained_model(model_file, client_factory):
    response = client_factory(model_file).get(f"{API}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["model_version"]
    assert payload["input_resolution_deg"] == pytest.approx(0.125)
    assert payload["output_resolution_deg"] == pytest.approx(FINE_RESOLUTION, abs=1e-6)
    assert payload["scale_factor"] == pytest.approx(2.5, abs=1e-6)
    assert payload["variable"] == "precipitation"
    assert set(payload["methods"]) == {"nearest", "cubic", "ai"}


def test_status_and_model_never_fail(tmp_path, client_factory):
    client = client_factory(tmp_path / "absent.pt")
    assert client.get(f"{API}/status").status_code == 200
    assert client.get(f"{API}/model").status_code == 200


def test_model_endpoint_describes_the_architecture(model_file, client_factory):
    response = client_factory(model_file).get(f"{API}/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["residual_learning"] is True
    assert "ReLU" in payload["output_constraint"]
    assert len(payload["input_features"]) == 10
    assert payload["output_variable"] == "precipitation"
    assert "SYNTHETIC" in payload["reference_disclaimer"]
    assert "Not an Operational Forecast" in payload["notice"]


def test_model_endpoint_works_untrained(tmp_path, client_factory):
    payload = client_factory(tmp_path / "absent.pt").get(f"{API}/model").json()
    assert payload["available"] is False
    assert payload["input_features"]


# --- run --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nearest", "cubic"])
def test_run_with_a_baseline_method(model_file, client_factory, method):
    response = client_factory(model_file).post(
        f"{API}/run", json={"method": method, "time_index": 0}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == method
    assert payload["input_resolution_deg"] == pytest.approx(0.125)
    assert payload["output_resolution_deg"] == pytest.approx(FINE_RESOLUTION, abs=1e-6)
    assert payload["input_shape"] == [9, 9]
    assert payload["output_shape"] == [21, 21]
    assert payload["scale_factor"] == pytest.approx(2.5, abs=1e-6)
    assert payload["statistics"]["max"] >= payload["statistics"]["min"]
    assert payload["model_version"] is None


def test_run_with_the_ai_method(model_file, client_factory):
    response = client_factory(model_file).post(f"{API}/run", json={"method": "ai"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "ai"
    assert payload["model_version"]
    assert payload["output_shape"] == [21, 21]


def test_run_reports_the_target_resolution(model_file, client_factory):
    payload = client_factory(model_file).post(
        f"{API}/run", json={"method": "nearest", "target_resolution": 0.1}
    ).json()
    assert payload["output_shape"] == [11, 11]  # 1 deg / 0.1 deg
    assert payload["output_resolution_deg"] == pytest.approx(0.1, abs=1e-6)


def test_ai_run_returns_503_without_a_checkpoint(tmp_path, client_factory):
    response = client_factory(tmp_path / "absent.pt").post(
        f"{API}/run", json={"method": "ai"}
    )
    assert response.status_code == 503
    assert "train_downscaling.py" in response.json()["detail"]


def test_run_rejects_an_unknown_method(model_file, client_factory):
    response = client_factory(model_file).post(f"{API}/run", json={"method": "magic"})
    assert response.status_code == 422
    assert "nearest" in response.json()["detail"]


def test_run_rejects_an_unsupported_variable(model_file, client_factory):
    response = client_factory(model_file).post(
        f"{API}/run", json={"method": "nearest", "variable": "pressure"}
    )
    assert response.status_code == 422


def test_run_rejects_an_invalid_resolution(model_file, client_factory):
    assert (
        client_factory(model_file)
        .post(f"{API}/run", json={"method": "nearest", "target_resolution": -1.0})
        .status_code
        == 422
    )


def test_run_rejects_an_out_of_range_time_index(model_file, client_factory):
    response = client_factory(model_file).post(
        f"{API}/run", json={"method": "nearest", "time_index": 9999}
    )
    assert response.status_code == 422
    assert "out of range" in response.json()["detail"]


def test_run_rejects_a_negative_time_index(model_file, client_factory):
    assert (
        client_factory(model_file)
        .post(f"{API}/run", json={"method": "nearest", "time_index": -1})
        .status_code
        == 422
    )


def test_run_rejects_an_unknown_body_field(model_file, client_factory):
    response = client_factory(model_file).post(
        f"{API}/run", json={"method": "nearest", "unexpected": 1}
    )
    assert response.status_code == 422


def test_run_reports_404_without_a_dataset(model_file, tmp_path, client_factory):
    response = client_factory(model_file, dataset_path=tmp_path / "absent.nc").post(
        f"{API}/run", json={"method": "nearest"}
    )
    assert response.status_code == 404


def test_run_does_not_return_the_whole_array(model_file, client_factory):
    """A downscaled field is thousands of values; only statistics travel."""
    payload = client_factory(model_file).post(f"{API}/run", json={"method": "nearest"}).json()
    assert "field" not in payload
    assert "values" not in payload
    assert set(payload["statistics"]) == {"min", "max", "mean", "std", "min_positive"}


# --- evaluate ---------------------------------------------------------------


def test_evaluate_compares_every_method_on_the_same_samples(model_file, client_factory):
    response = client_factory(model_file).post(f"{API}/evaluate", json={"max_frames": 8})
    assert response.status_code == 200
    payload = response.json()
    assert [row["method"] for row in payload["methods"]] == ["nearest", "cubic", "ai"]
    # Every method is scored on exactly the same number of grid points.
    assert len({row["n_points"] for row in payload["methods"]}) == 1
    assert payload["test_sample_count"] > 0
    assert len(payload["extreme_event"]) == 3


def test_evaluate_reports_every_metric(model_file, client_factory):
    row = client_factory(model_file).post(
        f"{API}/evaluate", json={"methods": ["nearest"], "max_frames": 8}
    ).json()["methods"][0]
    for key in ("mae", "rmse", "bias", "correlation", "pod", "far", "csi", "frequency_bias"):
        assert key in row
    assert row["mae"] >= 0.0
    assert row["n_points"] > 0


def test_evaluate_reports_the_extreme_event(model_file, client_factory):
    event = client_factory(model_file).post(
        f"{API}/evaluate", json={"methods": ["nearest"], "max_frames": 8}
    ).json()["extreme_event"][0]
    for key in (
        "peak_value", "reference_peak", "peak_ratio", "centroid_distance_km",
        "wet_area_km2", "area_ratio", "structure_correlation",
    ):
        assert key in event


def test_evaluate_ranking_is_ordered_by_measured_mae(model_file, client_factory):
    """The ranking is reported as measured - it is not an endorsement."""
    payload = client_factory(model_file).post(
        f"{API}/evaluate", json={"max_frames": 8}
    ).json()
    by_method = {row["method"]: row["mae"] for row in payload["methods"]}
    assert payload["ranking"] == sorted(by_method, key=by_method.get)


def test_evaluate_never_claims_accuracy(model_file, client_factory):
    payload = client_factory(model_file).post(
        f"{API}/evaluate", json={"methods": ["nearest"], "max_frames": 8}
    ).json()
    assert "SYNTHETIC" in payload["notice"]
    assert "does not imply higher accuracy" in payload["notice"]
    assert "Not an Operational Forecast" in payload["notice"]


def test_evaluate_rejects_an_unknown_method(model_file, client_factory):
    response = client_factory(model_file).post(
        f"{API}/evaluate", json={"methods": ["nearest", "magic"], "max_frames": 8}
    )
    assert response.status_code == 422
    assert "magic" in response.json()["detail"]


def test_evaluate_rejects_an_empty_method_list(model_file, client_factory):
    assert (
        client_factory(model_file)
        .post(f"{API}/evaluate", json={"methods": [], "max_frames": 8})
        .status_code
        == 422
    )


def test_evaluate_rejects_a_negative_threshold(model_file, client_factory):
    assert (
        client_factory(model_file)
        .post(f"{API}/evaluate", json={"wet_threshold": -1.0, "max_frames": 8})
        .status_code
        == 422
    )


def test_evaluate_honours_a_custom_threshold(model_file, client_factory):
    low = client_factory(model_file).post(
        f"{API}/evaluate",
        json={"methods": ["nearest"], "wet_threshold": 0.01, "max_frames": 8},
    ).json()
    high = client_factory(model_file).post(
        f"{API}/evaluate",
        json={"methods": ["nearest"], "wet_threshold": 100.0, "max_frames": 8},
    ).json()
    # A threshold above every value means nothing is ever wet, so POD is undefined.
    assert high["methods"][0]["pod"] is None
    assert low["methods"][0]["pod"] is not None


# --- coexistence with earlier phases ----------------------------------------


def test_downscaling_routes_do_not_disturb_earlier_endpoints(model_file, client_factory):
    client = client_factory(model_file)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/events").status_code == 200
    assert client.get("/api/v1/tracks").status_code == 200
    assert client.get("/api/v1/gnn/status").status_code == 200
    assert client.get(f"{API}/status").status_code == 200


def test_openapi_documents_the_downscaling_endpoints(model_file, client_factory):
    paths = client_factory(model_file).get("/openapi.json").json()["paths"]
    for path in ("status", "model", "run", "evaluate"):
        assert f"{API}/{path}" in paths

