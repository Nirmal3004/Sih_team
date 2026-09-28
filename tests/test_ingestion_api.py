"""Tests for the Phase 6 data API (SIH26078).

The client is backed by a temporary data directory, so every response comes
from the real router and the real registry rather than a stub.

What is asserted here is the status-code contract and the two rules a data API
must never break:

* **no invented provenance** - a local file is reported as ``source=local_file``
  with no URL, and ``is_live`` is always False;
* **no silent mixing** - synthetic and real datasets are listed separately and
  can be filtered apart.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from extreme_weather.config import Settings
from extreme_weather.ingestion import (
    DataOrigin,
    DatasetManifest,
    ManifestStore,
    SOURCE_LOCAL_FILE,
)

from tests.ingestion_fixtures import make_dataset, write_netcdf

UTC = timezone.utc

ROUTES = (
    "/api/v1/data/sources",
    "/api/v1/data/datasets",
    "/api/v1/data/datasets/{dataset_id}",
    "/api/v1/data/datasets/{dataset_id}/quality",
    "/api/v1/data/validate",
    "/api/v1/data/ingest",
    "/api/v1/data/datasets/{dataset_id}/process",
)


def register(
    settings: Settings,
    dataset_id: str = "alpha",
    origin: str = DataOrigin.HISTORICAL,
) -> str:
    """Write a data file and a manifest; return the path relative to data_dir."""
    base = (
        settings.synthetic_raw_dir
        if origin == DataOrigin.SYNTHETIC
        else settings.real_raw_dir
    )
    base.mkdir(parents=True, exist_ok=True)
    write_netcdf(make_dataset(), base / f"{dataset_id}.nc")

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        source=SOURCE_LOCAL_FILE,
        retrieved_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        file_name=f"{dataset_id}.nc",
        file_hash="0" * 64,
        format="netcdf",
        origin=origin,
        variables=("precipitation", "temperature"),
        time_start="2024-06-01T00:00:00",
        time_end="2024-06-01T03:00:00",
    )
    ManifestStore(settings.manifests_dir).save(manifest)
    return (base / f"{dataset_id}.nc").relative_to(settings.data_dir).as_posix()


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_every_data_route_is_registered(data_client: TestClient) -> None:
    paths = data_client.get("/openapi.json").json()["paths"]
    for route in ROUTES:
        assert route in paths, f"{route} is not registered"


# --------------------------------------------------------------------------- #
# GET /api/v1/data/sources
# --------------------------------------------------------------------------- #
def test_sources_lists_every_registered_source(data_client: TestClient) -> None:
    payload = data_client.get("/api/v1/data/sources").json()
    assert payload["count"] == 4
    assert {source["name"] for source in payload["sources"]} == {
        "netcdf",
        "grib",
        "era5",
        "nwp",
    }


def test_sources_report_availability_without_hiding_anything(
    data_client: TestClient,
) -> None:
    """An unusable source is listed with available=false, not omitted."""
    from extreme_weather.ingestion import grib_backend_available

    sources = {
        s["name"]: s for s in data_client.get("/api/v1/data/sources").json()["sources"]
    }
    assert f"available={grib_backend_available()}" in sources["grib"]["notes"]


def test_sources_state_that_era5_is_historical_not_live(data_client: TestClient) -> None:
    sources = {
        s["name"]: s for s in data_client.get("/api/v1/data/sources").json()["sources"]
    }
    assert sources["era5"]["default_origin"] == DataOrigin.HISTORICAL
    assert "not live" in sources["era5"]["notes"].lower()


def test_sources_never_expose_a_credential(data_client: TestClient) -> None:
    body = data_client.get("/api/v1/data/sources").text
    for secret_marker in ("apikey=", "Bearer", "token="):
        assert secret_marker not in body


# --------------------------------------------------------------------------- #
# GET /api/v1/data/datasets
# --------------------------------------------------------------------------- #
def test_an_empty_registry_returns_an_explanatory_empty_list(
    data_client: TestClient,
) -> None:
    payload = data_client.get("/api/v1/data/datasets").json()
    assert payload["count"] == 0
    assert "No datasets have been ingested" in payload["message"]


def test_datasets_are_listed_with_their_provenance(
    data_client: TestClient, settings: Settings
) -> None:
    register(settings, "alpha")
    payload = data_client.get("/api/v1/data/datasets").json()
    assert payload["count"] == 1
    manifest = payload["datasets"][0]
    assert manifest["dataset_id"] == "alpha"
    assert manifest["origin"] == DataOrigin.HISTORICAL
    assert manifest["is_synthetic"] is False
    assert manifest["is_live"] is False
    assert manifest["source"] == "local_file"
    assert manifest["source_url"] is None
    assert manifest["status_label"] == "HISTORICAL"


def test_synthetic_and_real_datasets_are_filterable(
    data_client: TestClient, settings: Settings
) -> None:
    register(settings, "fake", origin=DataOrigin.SYNTHETIC)
    register(settings, "real", origin=DataOrigin.HISTORICAL)
    assert data_client.get("/api/v1/data/datasets").json()["count"] == 2

    synthetic = data_client.get("/api/v1/data/datasets?origin=SYNTHETIC").json()
    assert [d["dataset_id"] for d in synthetic["datasets"]] == ["fake"]
    assert synthetic["datasets"][0]["is_synthetic"] is True
    assert synthetic["datasets"][0]["status_label"] == "SYNTHETIC"

    historical = data_client.get("/api/v1/data/datasets?origin=HISTORICAL").json()
    assert [d["dataset_id"] for d in historical["datasets"]] == ["real"]


def test_an_unknown_origin_filter_is_rejected(data_client: TestClient) -> None:
    assert data_client.get("/api/v1/data/datasets?origin=LIVE").status_code == 422


def test_datasets_can_be_filtered_by_time(
    data_client: TestClient, settings: Settings
) -> None:
    """The registered dataset covers 2024-06-01T00:00 .. 03:00."""
    register(settings, "june")
    hit = data_client.get(
        "/api/v1/data/datasets",
        params={"start_time": "2024-06-01T02:00:00", "end_time": "2024-06-01T05:00:00"},
    ).json()
    assert [d["dataset_id"] for d in hit["datasets"]] == ["june"]

    # A window that ends before the dataset starts overlaps nothing.
    miss = data_client.get(
        "/api/v1/data/datasets",
        params={"start_time": "2024-01-01T00:00:00", "end_time": "2024-01-05T00:00:00"},
    ).json()
    assert miss["count"] == 0


# --------------------------------------------------------------------------- #
# GET /api/v1/data/datasets/{id} and /quality
# --------------------------------------------------------------------------- #
def test_one_dataset_is_returned_with_its_provenance_chain(
    data_client: TestClient, settings: Settings
) -> None:
    register(settings, "alpha")
    response = data_client.get("/api/v1/data/datasets/alpha")
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["dataset_id"] == "alpha"
    assert "processing_steps" in manifest
    assert manifest["file_hash_algorithm"] == "sha256"


def test_an_unknown_dataset_returns_404(data_client: TestClient) -> None:
    response = data_client.get("/api/v1/data/datasets/nope")
    assert response.status_code == 404
    assert "not registered" in response.json()["detail"]


def test_the_quality_endpoint_rechecks_the_file(
    data_client: TestClient, settings: Settings
) -> None:
    register(settings, "alpha")
    payload = data_client.get("/api/v1/data/datasets/alpha/quality").json()
    assert payload["valid"] is True
    assert payload["time_steps"] == 4
    assert payload["spatial_points"] == 9
    assert "per_variable_nan" in payload


def test_quality_for_an_unknown_dataset_returns_404(data_client: TestClient) -> None:
    assert data_client.get("/api/v1/data/datasets/nope/quality").status_code == 404


# --------------------------------------------------------------------------- #
# POST /api/v1/data/validate
# --------------------------------------------------------------------------- #
def test_validate_reports_a_good_file(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    response = data_client.post("/api/v1/data/validate", json={"path": relative})
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_validate_writes_nothing(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    before = sorted(path.name for path in settings.manifests_dir.glob("*.json"))
    data_client.post("/api/v1/data/validate", json={"path": relative})
    assert sorted(path.name for path in settings.manifests_dir.glob("*.json")) == before


def test_validate_on_a_missing_file_returns_404(data_client: TestClient) -> None:
    response = data_client.post(
        "/api/v1/data/validate", json={"path": "raw/real/ghost.nc"}
    )
    assert response.status_code == 404


def test_validate_rejects_a_traversal_path(data_client: TestClient) -> None:
    """The API must not read outside the configured data directory."""
    response = data_client.post(
        "/api/v1/data/validate", json={"path": "../../etc/passwd"}
    )
    assert response.status_code != 200
    assert response.status_code in (404, 422)


def test_validate_rejects_an_unsupported_extension(data_client: TestClient) -> None:
    response = data_client.post("/api/v1/data/validate", json={"path": "raw/real/x.txt"})
    assert response.status_code == 422


def test_validate_rejects_an_unknown_body_field(data_client: TestClient) -> None:
    response = data_client.post(
        "/api/v1/data/validate", json={"path": "a.nc", "bogus": 1}
    )
    assert response.status_code == 422



# --------------------------------------------------------------------------- #
# POST /api/v1/data/ingest
# --------------------------------------------------------------------------- #
def test_ingest_registers_a_dataset(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    response = data_client.post(
        "/api/v1/data/ingest", json={"dataset_id": "alpha_copy", "path": relative}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "alpha_copy"
    assert payload["quality"]["valid"] is True
    assert payload["manifest"]["origin"] == DataOrigin.HISTORICAL
    assert payload["manifest"]["is_synthetic"] is False
    assert payload["manifest"]["source_url"] is None
    # And it is immediately queryable.
    assert data_client.get("/api/v1/data/datasets/alpha_copy").status_code == 200


def test_ingest_dry_run_registers_nothing(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    response = data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "preview", "path": relative, "dry_run": True},
    )
    assert response.status_code == 200
    assert response.json()["dry_run"] is True
    assert data_client.get("/api/v1/data/datasets/preview").status_code == 404
    assert not (settings.real_raw_dir / "preview.nc").exists()


def test_ingest_writes_real_data_under_raw_real(
    data_client: TestClient, settings: Settings
) -> None:
    relative = register(settings, "alpha")
    data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "observed", "path": relative, "origin": DataOrigin.HISTORICAL},
    )
    assert (settings.real_raw_dir / "observed.nc").is_file()
    assert not (settings.synthetic_raw_dir / "observed.nc").exists()


def test_ingest_writes_synthetic_data_under_raw_synthetic(
    data_client: TestClient, settings: Settings
) -> None:
    relative = register(settings, "alpha")
    data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "made_up", "path": relative, "origin": DataOrigin.SYNTHETIC},
    )
    assert (settings.synthetic_raw_dir / "made_up.nc").is_file()
    assert not (settings.real_raw_dir / "made_up.nc").exists()
    manifest = data_client.get("/api/v1/data/datasets/made_up").json()
    assert manifest["is_synthetic"] is True


def test_ingest_rejects_an_unknown_origin(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    response = data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "x", "path": relative, "origin": "LIVE"},
    )
    assert response.status_code == 422


def test_ingest_rejects_a_missing_file(data_client: TestClient) -> None:
    response = data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "ghost", "path": "raw/real/ghost.nc"},
    )
    assert response.status_code == 404


def test_ingest_rejects_a_traversal_path(data_client: TestClient) -> None:
    response = data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "escape", "path": "../../secret.nc"},
    )
    assert response.status_code != 200


def test_ingest_rejects_an_unknown_body_field(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    response = data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "x", "path": relative, "bogus": True},
    )
    assert response.status_code == 422


def test_ingest_rejects_an_unknown_source(data_client: TestClient, settings: Settings) -> None:
    relative = register(settings, "alpha")
    response = data_client.post(
        "/api/v1/data/ingest",
        json={"dataset_id": "x", "path": relative, "source": "nope"},
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# POST /api/v1/data/datasets/{id}/process
# --------------------------------------------------------------------------- #
def test_process_runs_the_phase0_pipeline(data_client: TestClient, settings: Settings) -> None:
    register(settings, "alpha")
    response = data_client.post("/api/v1/data/datasets/alpha/process", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"] == "alpha"
    assert payload["time_steps"] == 4

    processed = Path(payload["processed_path"])
    assert processed.is_file()
    # The processed file is the ingested one plus the configured suffix, and it
    # lands in the processed directory, not back in raw/.
    assert processed.name == "alpha" + settings.processed_real_suffix
    assert processed.parent == settings.processed_dir


def test_process_for_an_unknown_dataset_returns_404(data_client: TestClient) -> None:
    assert (
        data_client.post("/api/v1/data/datasets/nope/process", json={}).status_code == 404
    )


def test_process_rejects_an_invalid_max_gap(data_client: TestClient, settings: Settings) -> None:
    register(settings, "alpha")
    response = data_client.post(
        "/api/v1/data/datasets/alpha/process", json={"max_gap": -1}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Non-regression: the earlier phases still answer
# --------------------------------------------------------------------------- #
def test_registering_the_data_router_keeps_phases_0_to_5_alive(
    data_client: TestClient,
) -> None:
    assert data_client.get("/health").status_code == 200
    assert data_client.get("/api/v1/datasets").status_code == 200
    assert data_client.get("/api/v1/events").status_code == 200
    assert data_client.get("/api/v1/tracks").status_code == 200
    assert data_client.get("/api/v1/gnn/status").status_code == 200
    assert data_client.get("/api/v1/downscaling/status").status_code == 200
    assert data_client.get("/api/v1/alerts").status_code == 200

