"""API tests for the Phase 8 pipeline router (SIH26078, Phase 8, section 11).

Required coverage: status, run, run history, run detail, 404, 422 and 503.

Runs are executed against the deterministic ``fixture`` provider so the suite
needs no network and no credentials. Status-code assertions are made explicitly
because they carry meaning here: 503 means "retry may help", 422 means "retry
will not help".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from extreme_weather.api.main import create_app
from extreme_weather.config import Settings, get_settings

UTC = timezone.utc


@pytest.fixture()
def pipeline_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Return a client whose pipeline writes only under ``tmp_path``."""
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    application = create_app(settings)
    # Same override the rest of the suite uses, so the pipeline router reads
    # this tmp_path rather than the repository data/ directory.
    application.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


def _body(hours: int = 3) -> dict:
    """Return a valid run request body."""
    return {
        "provider": "fixture",
        "hours": hours,
        "variables": ["precipitation", "temperature", "u_wind", "v_wind"],
    }


# --------------------------------------------------------------------------- #
# GET /pipeline/status
# --------------------------------------------------------------------------- #
def test_status_reports_idle_before_any_run(pipeline_client: TestClient) -> None:
    """A pipeline that has not run is IDLE, with no invented values."""
    response = pipeline_client.get("/api/v1/pipeline/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_state"] == "IDLE"
    assert payload["latest_timestamp"] is None
    assert payload["last_successful_run"] is None


def test_status_lists_the_pipeline_stages(pipeline_client: TestClient) -> None:
    """The dashboard renders stage order from this field."""
    payload = pipeline_client.get("/api/v1/pipeline/status").json()

    assert payload["stage_order"] == [
        "ingest",
        "validate",
        "preprocess",
        "anomaly",
        "track",
        "gnn",
        "downscale",
        "alert",
    ]


def test_status_reports_provider_availability(pipeline_client: TestClient) -> None:
    """Provider configuration is part of the status a client needs."""
    payload = pipeline_client.get("/api/v1/pipeline/status").json()

    assert payload["provider"]["name"] == "fixture"
    assert payload["provider"]["available"] is True


def test_status_includes_freshness_thresholds(pipeline_client: TestClient) -> None:
    """The client can show why data counts as stale."""
    payload = pipeline_client.get("/api/v1/pipeline/status").json()

    assert payload["freshness"]["thresholds"]["fresh_within_hours"] == 3.0


def test_status_carries_the_prototype_disclaimer(pipeline_client: TestClient) -> None:
    """The disclaimer travels with the data, not only in the docs."""
    payload = pipeline_client.get("/api/v1/pipeline/status").json()

    assert "Not an operational emergency-warning system" in payload["disclaimer"]


# --------------------------------------------------------------------------- #
# POST /pipeline/run
# --------------------------------------------------------------------------- #
def test_run_executes_the_pipeline(pipeline_client: TestClient) -> None:
    """A successful run returns every stage."""
    response = pipeline_client.post("/api/v1/pipeline/run", json=_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"].startswith("run-")
    assert [s["name"] for s in payload["stages"]] == [
        "ingest",
        "validate",
        "preprocess",
        "anomaly",
        "track",
        "gnn",
        "downscale",
        "alert",
    ]


def test_run_reports_freshness(pipeline_client: TestClient) -> None:
    """Freshness is part of every run response."""
    payload = pipeline_client.post("/api/v1/pipeline/run", json=_body()).json()

    assert payload["freshness"] in ("FRESH", "STALE", "EXPIRED", "UNKNOWN")
    assert payload["source_timestamp"] is not None


def test_run_records_an_ingestion(pipeline_client: TestClient) -> None:
    """Provenance travels with the run."""
    payload = pipeline_client.post("/api/v1/pipeline/run", json=_body()).json()

    assert payload["ingestion"]["provider"] == "fixture"
    assert payload["ingestion"]["checksum"]


def test_dry_run_query_parameter_is_honoured(pipeline_client: TestClient) -> None:
    """POST /pipeline/run?dry_run=true must plan without writing."""
    response = pipeline_client.post("/api/v1/pipeline/run?dry_run=true", json=_body())

    assert response.status_code == 200
    assert response.json()["dry_run"] is True


def test_dry_run_body_field_is_honoured(pipeline_client: TestClient) -> None:
    """The body field works too, so a scripted client need not build a query."""
    body = {**_body(), "dry_run": True}
    response = pipeline_client.post("/api/v1/pipeline/run", json=body)

    assert response.json()["dry_run"] is True


def test_dry_run_writes_no_artifacts(pipeline_client: TestClient, tmp_path: Path) -> None:
    """A dry run must leave the runs directory empty."""
    pipeline_client.post("/api/v1/pipeline/run?dry_run=true", json=_body())

    runs_dir = tmp_path / "data" / "realtime" / "runs"
    assert not runs_dir.exists() or not list(runs_dir.glob("run-*.json"))


def test_query_parameter_overrides_the_body(pipeline_client: TestClient) -> None:
    """An explicit query flag wins, so a caller can force safety."""
    body = {**_body(), "dry_run": False}
    response = pipeline_client.post("/api/v1/pipeline/run?dry_run=true", json=body)

    assert response.json()["dry_run"] is True


# --------------------------------------------------------------------------- #
# GET /pipeline/runs and /pipeline/runs/{id}
# --------------------------------------------------------------------------- #
def test_run_history_is_empty_before_any_run(pipeline_client: TestClient) -> None:
    """"No runs yet" is a valid state, not a 404."""
    response = pipeline_client.get("/api/v1/pipeline/runs")

    assert response.status_code == 200
    assert response.json() == {**response.json(), "total": 0, "runs": []}


def test_run_history_lists_a_completed_run(pipeline_client: TestClient) -> None:
    """A finished run appears in the history."""
    created = pipeline_client.post("/api/v1/pipeline/run", json=_body()).json()

    payload = pipeline_client.get("/api/v1/pipeline/runs").json()

    assert payload["total"] == 1
    assert payload["runs"][0]["run_id"] == created["run_id"]


def test_run_history_respects_the_limit(pipeline_client: TestClient) -> None:
    """A client polling a long history must be able to bound the response."""
    for _ in range(2):
        pipeline_client.post("/api/v1/pipeline/run", json=_body())

    payload = pipeline_client.get("/api/v1/pipeline/runs?limit=1").json()

    assert len(payload["runs"]) == 1
    assert payload["total"] == 2


def test_run_detail_returns_every_stage(pipeline_client: TestClient) -> None:
    """One run is returned in full, with timings and messages."""
    created = pipeline_client.post("/api/v1/pipeline/run", json=_body()).json()

    payload = pipeline_client.get(f"/api/v1/pipeline/runs/{created['run_id']}").json()

    assert payload["run_id"] == created["run_id"]
    assert len(payload["stages"]) == 8
    assert payload["total_duration"] is not None


def test_unknown_run_returns_404(pipeline_client: TestClient) -> None:
    """An unknown run id is a 404, not a 200 with empty fields."""
    response = pipeline_client.get("/api/v1/pipeline/runs/run-does-not-exist")

    assert response.status_code == 404
    assert "run-does-not-exist" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 422 - invalid request
# --------------------------------------------------------------------------- #
def test_invalid_hours_returns_422(pipeline_client: TestClient) -> None:
    """A non-positive window length is a client error."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run", json={**_body(), "hours": 0}
    )

    assert response.status_code == 422


def test_empty_variable_list_returns_422(pipeline_client: TestClient) -> None:
    """Requesting no variables would ingest nothing."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run", json={**_body(), "variables": []}
    )

    assert response.status_code == 422


def test_inverted_latitude_box_returns_422(pipeline_client: TestClient) -> None:
    """A min above its max is not a bounding box."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run",
        json={**_body(), "latitude_min": 20.0, "latitude_max": 10.0},
    )

    assert response.status_code == 422


def test_inverted_longitude_box_returns_422(pipeline_client: TestClient) -> None:
    """The same rule applies to longitude."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run",
        json={**_body(), "longitude_min": 90.0, "longitude_max": 10.0},
    )

    assert response.status_code == 422


def test_unknown_body_field_returns_422(pipeline_client: TestClient) -> None:
    """An unrecognised field is rejected rather than silently ignored."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run", json={**_body(), "unexpected": 1}
    )

    assert response.status_code == 422


def test_missing_body_is_rejected(pipeline_client: TestClient) -> None:
    """Omitting the body entirely is a client error, not a default run."""
    response = pipeline_client.post("/api/v1/pipeline/run")

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 503 - unavailable provider
# --------------------------------------------------------------------------- #
def test_unavailable_local_file_returns_503(pipeline_client: TestClient) -> None:
    """A configured-but-absent local file is a dependency failure."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run",
        json={**_body(), "provider": "local", "file_path": "does/not/exist.nc"},
    )

    assert response.status_code == 503


def test_local_provider_without_a_path_returns_400(pipeline_client: TestClient) -> None:
    """A missing required argument is a configuration error, not 503."""
    response = pipeline_client.post("/api/v1/pipeline/run", json={**_body(), "provider": "local"})

    assert response.status_code == 400


def test_remote_provider_without_a_url_returns_400(pipeline_client: TestClient) -> None:
    """The same distinction applies to the remote provider."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run", json={**_body(), "provider": "remote-http"}
    )

    assert response.status_code == 400


def test_insecure_remote_url_returns_400(pipeline_client: TestClient) -> None:
    """A non-https endpoint is refused before any request is made."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run",
        json={**_body(), "provider": "remote-http", "url": "http://weather.invalid/api"},
    )

    assert response.status_code == 400


def test_unknown_provider_returns_400(pipeline_client: TestClient) -> None:
    """An unknown provider name is a configuration error."""
    response = pipeline_client.post(
        "/api/v1/pipeline/run", json={**_body(), "provider": "nonsense"}
    )

    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# GET /ingestion/status
# --------------------------------------------------------------------------- #
def test_ingestion_status_is_available(pipeline_client: TestClient) -> None:
    """The Phase 6 ingestion router is extended, not replaced."""
    response = pipeline_client.get("/api/v1/ingestion/status")

    assert response.status_code == 200
    payload = response.json()
    assert "freshness" in payload
    assert "latest_ingestion" in payload


def test_ingestion_status_is_empty_before_a_run(pipeline_client: TestClient) -> None:
    """No ingestion yet is reported as null, not as a zero timestamp."""
    payload = pipeline_client.get("/api/v1/ingestion/status").json()

    assert payload["latest_ingestion"] is None
    assert payload["latest_timestamp"] is None


def test_ingestion_status_reflects_the_latest_run(pipeline_client: TestClient) -> None:
    """After a run, the latest ingestion and freshness are reported."""
    pipeline_client.post("/api/v1/pipeline/run", json=_body())

    payload = pipeline_client.get("/api/v1/ingestion/status").json()

    assert payload["latest_ingestion"] is not None
    assert payload["latest_ingestion"]["checksum"]
    assert payload["last_successful_run"] is not None


def test_ingestion_status_carries_the_disclaimer(pipeline_client: TestClient) -> None:
    """The caveat travels with the provenance."""
    payload = pipeline_client.get("/api/v1/ingestion/status").json()

    assert "Not an operational emergency-warning system" in payload["disclaimer"]


# --------------------------------------------------------------------------- #
# OpenAPI and non-regression
# --------------------------------------------------------------------------- #
def test_pipeline_routes_are_documented(pipeline_client: TestClient) -> None:
    """The new endpoints appear in the OpenAPI schema."""
    paths = pipeline_client.get("/openapi.json").json()["paths"]

    assert "/api/v1/pipeline/status" in paths
    assert "/api/v1/pipeline/runs" in paths
    assert "/api/v1/pipeline/runs/{run_id}" in paths
    assert "/api/v1/pipeline/run" in paths
    assert "/api/v1/ingestion/status" in paths


def test_phase7_validation_routes_still_work(pipeline_client: TestClient) -> None:
    """Registering Phase 8 must not disturb Phase 7."""
    response = pipeline_client.get("/api/v1/validation/status")

    assert response.status_code == 200
    assert "has_run" in response.json()


def test_health_still_works(pipeline_client: TestClient) -> None:
    """The base health probe is unaffected."""
    assert pipeline_client.get("/health").status_code == 200
