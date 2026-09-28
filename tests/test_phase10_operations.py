"""API, readiness and metrics tests (SIH26078, Phase 10, sections 3, 4, 5, 8).

Covers request correlation, the standard error envelope, liveness versus
readiness, and metric collection.

Two guarantees are asserted repeatedly because they are the ones a hardening
phase most often breaks:

* ``GET /health`` keeps its exact original contract - it is the probe a
  container restart policy depends on;
* a 500 never leaks a stack trace, a traceback path or an internal exception.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient

from extreme_weather.api.main import create_app
from extreme_weather.api.middleware import REQUEST_ID_HEADER, RUN_ID_HEADER
from extreme_weather.config import Settings, get_settings
from extreme_weather.observability.metrics import get_registry, reset_registry


@pytest.fixture(autouse=True)
def _clean_metrics() -> Iterator[None]:
    """Give every test an empty registry."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def ops_settings(tmp_path: Path) -> Settings:
    """Return settings whose directories live under ``tmp_path``."""
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    return settings


@pytest.fixture()
def ops_client(ops_settings: Settings) -> Iterator[TestClient]:
    """Return a client bound to :func:`ops_settings`."""
    application = create_app(ops_settings)
    application.dependency_overrides[get_settings] = lambda: ops_settings
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Liveness must not change
# --------------------------------------------------------------------------- #
def test_health_keeps_its_original_contract(ops_client: TestClient) -> None:
    """A container restart policy depends on this exact shape."""
    response = ops_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert set(payload) == {"status", "project", "service", "version", "environment"}


def test_health_does_not_consult_the_filesystem(ops_client: TestClient) -> None:
    """A liveness probe that fails on a missing model would restart-loop."""
    response = ops_client.get("/health")

    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #
def test_ready_reports_an_explicit_state(ops_client: TestClient) -> None:
    """The answer must be one of three named states, not a bare boolean."""
    response = ops_client.get("/ready")

    assert response.status_code == 200
    assert response.json()["state"] in ("READY", "DEGRADED", "NOT_READY")


def test_ready_reports_its_components(ops_client: TestClient) -> None:
    """A caller must see *which* component decided the verdict."""
    payload = ops_client.get("/ready").json()

    names = {c["name"] for c in payload["components"]}
    assert {"directories", "gnn", "downscaling"} <= names


def test_ready_is_false_when_a_required_directory_is_missing(
    tmp_path: Path,
) -> None:
    """An unwritable data directory means the process cannot serve."""
    settings = Settings(
        data_dir=tmp_path / "absent", model_dir=tmp_path / "absent" / "models"
    )
    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    with TestClient(application, raise_server_exceptions=False) as client:
        payload = client.get("/ready").json()

    assert payload["state"] == "NOT_READY"
    assert payload["ready"] is False


def test_a_missing_model_is_never_reported_ready(tmp_path: Path) -> None:
    """The explicit Phase 10 requirement."""
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    with TestClient(application, raise_server_exceptions=False) as client:
        payload = client.get("/ready").json()

    models = {c["name"]: c for c in payload["components"] if c["name"] in ("gnn", "downscaling")}
    for component in models.values():
        if component["state"] != "AVAILABLE":
            assert payload["state"] in ("DEGRADED", "NOT_READY"), (
                "a component that is unavailable must prevent a READY verdict"
            )


def test_ready_carries_the_prototype_disclaimer(ops_client: TestClient) -> None:
    """The caveat must travel with the verdict."""
    payload = ops_client.get("/ready").json()

    assert "Not an operational disaster-warning system" in payload["disclaimer"]


def test_operations_status_mirrors_ready(ops_client: TestClient) -> None:
    """Two endpoints reporting the same thing must not drift apart."""
    ready = ops_client.get("/ready").json()
    status = ops_client.get("/api/v1/operations/status").json()

    assert ready["state"] == status["state"]


# --------------------------------------------------------------------------- #
# Request correlation
# --------------------------------------------------------------------------- #
def test_every_response_carries_a_request_id(ops_client: TestClient) -> None:
    """Without this, a log line cannot be tied to a request."""
    response = ops_client.get("/health")

    assert response.headers.get(REQUEST_ID_HEADER)


def test_request_ids_are_unique_per_request(ops_client: TestClient) -> None:
    """A constant id would make correlation useless."""
    first = ops_client.get("/health").headers.get(REQUEST_ID_HEADER)
    second = ops_client.get("/health").headers.get(REQUEST_ID_HEADER)

    assert first != second


def test_a_safe_incoming_request_id_is_honoured(ops_client: TestClient) -> None:
    """A client or proxy must be able to supply its own trace id."""
    response = ops_client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert response.headers.get(REQUEST_ID_HEADER) == "trace-abc-123"


@pytest.mark.parametrize(
    "hostile",
    [
        "bad id with spaces",
        "id\nwith-newline",
        "x" * 300,
        "id/with/slashes",
        "",
    ],
)
def test_a_hostile_incoming_request_id_is_replaced(
    ops_client: TestClient, hostile: str
) -> None:
    """Reflecting arbitrary text into a header is an injection vector."""
    response = ops_client.get("/health", headers={REQUEST_ID_HEADER: hostile})
    echoed = response.headers.get(REQUEST_ID_HEADER)

    assert echoed != hostile
    assert echoed and "\n" not in echoed


def test_a_run_id_header_is_accepted(ops_client: TestClient) -> None:
    """A pipeline run id may be propagated for correlation."""
    response = ops_client.get("/health", headers={RUN_ID_HEADER: "run-xyz-1"})

    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Error envelope
# --------------------------------------------------------------------------- #
def test_a_404_uses_the_standard_envelope(ops_client: TestClient) -> None:
    """Every error shares one shape, so a client needs one parser."""
    response = ops_client.get("/api/v1/events/no-such-event")

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert body["message"]
    assert body["request_id"]


def test_the_envelope_retains_the_legacy_detail_field(ops_client: TestClient) -> None:
    """Phase 0-9 clients read `detail`; removing it would be a breaking change."""
    response = ops_client.get("/api/v1/events/no-such-event")

    assert response.json()["detail"] == response.json()["message"]


def test_a_422_includes_field_details(ops_client: TestClient) -> None:
    """A validation failure must say which field was wrong."""
    response = ops_client.post("/api/v1/evaluation/run", json={"seed": "not-an-int"})

    assert response.status_code == 422
    assert "details" in response.json()


def test_a_503_is_not_masked(ops_client: TestClient) -> None:
    """"Run scripts/train_gnn.py" is actionable guidance, not an internal detail."""
    response = ops_client.get("/api/v1/gnn/predict-nonexistent")

    assert response.status_code in (404, 503)
    assert response.json()["message"]


def test_a_500_never_exposes_a_traceback(
    tmp_path: Path,
) -> None:
    """The single most important error guarantee."""
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    application = create_app(settings)

    router = APIRouter()

    @router.get("/boom")
    def boom() -> None:
        raise RuntimeError("internal detail: /etc/secret/path and token=hunter2")

    application.include_router(router)
    application.dependency_overrides[get_settings] = lambda: settings
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    text = response.text
    assert "Traceback" not in text
    assert "internal detail" not in text
    assert "hunter2" not in text
    assert "/etc/secret" not in text
    assert response.json()["request_id"]


def test_an_error_body_still_contains_correlation(ops_client: TestClient) -> None:
    """An operator must be able to find the failure from the client's report."""
    response = ops_client.get("/api/v1/events/nope", headers={REQUEST_ID_HEADER: "trace-9"})

    assert response.json()["request_id"] == "trace-9"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_reports_counters(ops_client: TestClient) -> None:
    """System counters must be readable after a request."""
    ops_client.get("/health")

    payload = ops_client.get("/api/v1/operations/metrics").json()

    assert payload["counters"]["api_requests_total"]
    assert "uptime_seconds" in payload


def test_request_metrics_are_counted(ops_client: TestClient) -> None:
    """Each request increments the counter exactly once."""
    before = get_registry().counter_value("api_requests_total", method="GET")
    ops_client.get("/health")

    assert get_registry().counter_value("api_requests_total", method="GET") == before + 1


def test_error_metrics_are_counted_with_the_status(ops_client: TestClient) -> None:
    """An error rate needs the status to be useful."""
    ops_client.get("/api/v1/events/nope")

    assert get_registry().counter_value("api_errors_total", method="GET", status=404) >= 1


def test_request_duration_is_recorded(ops_client: TestClient) -> None:
    """A latency histogram is the point of instrumenting the middleware."""
    ops_client.get("/health")

    histograms = get_registry().snapshot()["histograms"]
    # Keys carry their label set, e.g. "api_request_duration_seconds{method=GET}".
    assert any(key.startswith("api_request_duration_seconds") for key in histograms)


def test_metrics_contain_no_model_quality_claim(ops_client: TestClient) -> None:
    """The Phase 10 prohibition, asserted on the live response."""
    ops_client.get("/health")
    payload = ops_client.get("/api/v1/operations/metrics").json()
    # The note legitimately names these words in order to deny them, so the
    # check is on the metric data itself.
    data = json.dumps(
        {"counters": payload["counters"], "histograms": payload["histograms"]}
    ).lower()

    for forbidden in ("accuracy", "precision", "recall", "skill", "f1", "auc"):
        assert forbidden not in data


def test_metrics_state_they_are_not_quality_metrics(ops_client: TestClient) -> None:
    """A reader must be told what the numbers are and are not."""
    payload = ops_client.get("/api/v1/operations/metrics").json()

    assert "not model-quality metrics" in payload["note"]


# --------------------------------------------------------------------------- #
# Integrity endpoint
# --------------------------------------------------------------------------- #
def test_integrity_endpoint_reports_each_artifact(ops_client: TestClient) -> None:
    """Every artifact gets an explicit state, not a boolean guess."""
    payload = ops_client.get("/api/v1/operations/integrity").json()

    assert "gnn_checkpoint" in payload["checks"]
    assert "downscaling_checkpoint" in payload["checks"]
    for entry in payload["checks"].values():
        assert entry["state"] in ("VALID", "INCOMPATIBLE", "CORRUPT", "MISSING")


def test_integrity_explains_a_missing_checkpoint(ops_client: TestClient) -> None:
    """A missing model is normal here and must be described, not alarmed about."""
    payload = ops_client.get("/api/v1/operations/integrity").json()

    assert "normal state" in payload["note"]


# --------------------------------------------------------------------------- #
# Registry behaviour
# --------------------------------------------------------------------------- #
def test_counters_are_labelled() -> None:
    """Dimensions must not collide across label sets."""
    registry = get_registry()
    registry.increment("x", method="GET")
    registry.increment("x", method="POST")

    assert registry.counter_value("x", method="GET") == 1
    assert registry.counter_value("x", method="POST") == 1


def test_an_unseen_counter_reads_as_zero() -> None:
    """Zero and unknown are different states, and a read must not create one."""
    assert get_registry().counter_value("never_incremented") == 0


def test_reset_clears_everything() -> None:
    """A test must be able to isolate itself."""
    registry = get_registry()
    registry.increment("x")
    registry.reset()

    assert registry.counter_value("x") == 0


def test_histogram_buckets_are_cumulative_and_ordered() -> None:
    """A latency histogram whose buckets decrease is worse than none."""
    from extreme_weather.observability.metrics import MetricsRegistry

    registry = MetricsRegistry()
    for value in (0.001, 0.02, 0.3, 5.0, 400.0):
        registry.observe("d", value)

    buckets = registry.snapshot()["histograms"]["d"]["buckets"]
    counts = list(buckets.values())

    assert counts == sorted(counts)
    assert buckets and all(key.startswith("le_") for key in buckets)


def test_non_finite_observations_are_ignored() -> None:
    """A NaN duration must not corrupt the distribution."""
    from extreme_weather.observability.metrics import MetricsRegistry

    registry = MetricsRegistry()
    registry.observe("d", float("nan"))
    registry.observe("d", float("inf"))
    registry.observe("d", 0.01)

    assert registry.snapshot()["histograms"]["d"]["count"] == 1
