"""Tests for the FastAPI surface: health, service info and dataset metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from extreme_weather.api.main import create_app
from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID, PROJECT_NAME, Settings, get_settings
from extreme_weather.data.loader import load_dataset, save_dataset
from extreme_weather.data.generator import generate_and_save


# --------------------------------------------------------------------------- #
# Health / service information
# --------------------------------------------------------------------------- #
def test_health_endpoint_contract(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["project"] == "SIH26078"
    assert payload["service"] == "DeepWeather AI"
    assert payload["version"] == PACKAGE_VERSION
    assert payload["environment"] == "development"


def test_health_endpoint_needs_no_data_directory(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "absent", model_dir=tmp_path / "models")
    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        listing = client.get("/api/v1/datasets")
        assert listing.status_code == 200
        assert listing.json()["count"] == 0

    # Importing/using the API must not silently create directories on disk.
    assert not (tmp_path / "absent").exists()


def test_root_endpoint_describes_the_service(api_client: TestClient) -> None:
    response = api_client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["project"] == PROJECT_ID
    assert payload["service"] == PROJECT_NAME
    assert payload["health_url"] == "/health"
    assert payload["datasets_url"] == "/api/v1/datasets"
    assert "Phase 0" in payload["phase"]


def test_openapi_schema_exposes_the_endpoints(api_client: TestClient) -> None:
    schema = api_client.get("/openapi.json").json()

    assert schema["info"]["title"].startswith(PROJECT_NAME)
    assert "/health" in schema["paths"]
    assert "/api/v1/datasets" in schema["paths"]
    assert "/api/v1/datasets/validate" in schema["paths"]
    assert "/api/v1/datasets/{dataset_path}/summary" in schema["paths"]


def test_cors_allows_the_frontend_dev_origin(api_client: TestClient) -> None:
    response = api_client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


# --------------------------------------------------------------------------- #
# Dataset endpoints
# --------------------------------------------------------------------------- #
def test_dataset_listing_finds_the_synthetic_file(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/datasets")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["datasets"][0]["name"] == "raw/sample_history.nc"
    assert payload["datasets"][0]["subdirectory"] == "raw"
    assert payload["datasets"][0]["size_bytes"] > 0
    # mtime must be a parseable ISO timestamp.
    datetime.fromisoformat(payload["datasets"][0]["modified_utc"])


def test_dataset_summary_returns_real_metadata(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/datasets/raw/sample_history.nc/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "sample_history"
    assert payload["dimensions"] == {"time": 16, "latitude": 9, "longitude": 9}
    assert payload["data_variables"] == [
        "precipitation",
        "temperature",
        "pressure",
        "relative_humidity",
        "u_wind",
        "v_wind",
    ]
    assert payload["time_steps"] == 16
    assert payload["time_start"] == "2024-06-01T00:00:00"
    assert payload["time_step_seconds"] == 3600.0
    assert payload["latitude_points"] == 9
    assert payload["grid_resolution_deg"] == pytest.approx(0.25)
    assert payload["approximate_grid_resolution_km"] == pytest.approx(27.83, abs=0.01)
    assert payload["is_synthetic"] is True
    assert payload["attributes"]["project"] == PROJECT_ID
    assert payload["attributes"]["dataset_type"] == "synthetic"

    precipitation = next(entry for entry in payload["variables"] if entry["name"] == "precipitation")
    assert precipitation["units"] == "mm h-1"
    assert precipitation["dimensions"] == ["time", "latitude", "longitude"]
    assert precipitation["shape"] == [16, 9, 9]


def test_dataset_summary_missing_file_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/datasets/raw/unknown.nc/summary")

    assert response.status_code == 404
    assert "unknown.nc" in response.json()["detail"]


def test_dataset_endpoints_reject_path_traversal(api_client: TestClient) -> None:
    # Percent-encoded traversal: Starlette decodes it into the route parameter,
    # so the guard inside resolve_dataset_path must reject it.
    encoded = api_client.get("/api/v1/datasets/..%2F..%2Foutside.nc/summary")
    assert encoded.status_code == 400
    assert "escapes the data directory" in encoded.json()["detail"]

    body = api_client.post("/api/v1/datasets/validate", json={"path": "../outside.nc"})
    assert body.status_code == 400
    assert "escapes the data directory" in body.json()["detail"]

    wrong_suffix = api_client.post("/api/v1/datasets/validate", json={"path": "raw/sample_history.txt"})
    assert wrong_suffix.status_code == 400
    assert "Unsupported dataset extension" in wrong_suffix.json()["detail"]


def test_validate_endpoint_accepts_the_generated_dataset(api_client: TestClient) -> None:
    response = api_client.post("/api/v1/datasets/validate", json={"path": "raw/sample_history.nc"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is True
    assert payload["errors"] == []
    assert payload["warnings"] == []
    assert payload["path"] == "raw/sample_history.nc"


def test_validate_endpoint_reports_contract_violations(
    api_client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "raw" / "sample_history.nc"
    assert source.is_file()

    incomplete_path = tmp_path / "raw" / "incomplete.nc"
    save_dataset(load_dataset(source, validate=False).drop_vars("u_wind"), incomplete_path)

    response = api_client.post("/api/v1/datasets/validate", json={"path": "raw/incomplete.nc"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert any("u_wind" in message for message in payload["errors"])


def test_validate_endpoint_missing_file_returns_404(api_client: TestClient) -> None:
    response = api_client.post("/api/v1/datasets/validate", json={"path": "processed/nope.nc"})

    assert response.status_code == 404
    assert "nope.nc" in response.json()["detail"]


def test_dataset_listing_includes_processed_output(api_client: TestClient, tmp_path: Path) -> None:
    from extreme_weather.data.preprocessing import preprocess_file

    raw_path = tmp_path / "raw" / "sample_history.nc"
    processed = tmp_path / "processed" / "sample_history_processed.nc"
    preprocess_file(raw_path, processed)

    payload = api_client.get("/api/v1/datasets").json()
    names = {entry["name"] for entry in payload["datasets"]}

    assert payload["count"] == 2
    assert names == {"raw/sample_history.nc", "processed/sample_history_processed.nc"}

    summary = api_client.get(
        "/api/v1/datasets/processed/sample_history_processed.nc/summary"
    ).json()
    assert "wind_speed" in summary["data_variables"]
    assert "wind_direction" in summary["data_variables"]
    assert summary["attributes"]["processed"] == "true"


def test_api_client_fixture_is_isolated_from_the_repository_data_dir(api_client: TestClient) -> None:
    payload = api_client.get("/api/v1/datasets").json()
    data_dir = Path(payload["data_dir"])

    assert data_dir.is_absolute()
    assert data_dir.resolve() != (Path.cwd() / "data").resolve()
    assert (data_dir / "raw" / "sample_history.nc").is_file()


@pytest.fixture()
def regenerated_dataset_path(tmp_path: Path) -> Path:
    """Helper fixture used by :func:`test_preprocessing_via_api_paths`."""
    target = tmp_path / "samples" / "regenerated.nc"
    generate_and_save(target, write_manifest_file=False)
    return target


def test_preprocessing_via_api_paths(api_client: TestClient, regenerated_dataset_path: Path) -> None:
    """The API must be able to summarise a freshly generated default dataset."""
    response = api_client.get("/api/v1/datasets/samples/regenerated.nc/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dimensions"]["time"] == 72
    assert payload["grid_resolution_deg"] == pytest.approx(0.125)
    assert payload["approximate_grid_resolution_km"] == pytest.approx(13.92, abs=0.05)
    assert payload["is_synthetic"] is True
