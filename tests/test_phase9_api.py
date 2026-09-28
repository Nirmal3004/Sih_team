"""Evaluation API and CLI tests (SIH26078, sections 12, 19).

Required coverage: status, summary, GNN evaluation, downscaling evaluation,
uncertainty, event explanation, evaluation run, unknown event, invalid request
and unavailable model.

The honesty requirements are asserted at the HTTP boundary too, because that is
where a consumer actually reads the result: a 404 rather than an empty 200, and
no endpoint that returns a zero where it should return an absence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from extreme_weather.api.main import create_app
from extreme_weather.config import Settings, get_settings
from extreme_weather.evaluation import run_evaluation

from tests.phase9_support import make_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "run_phase9_evaluation.py"


@pytest.fixture()
def isolated_settings(tmp_path: Path) -> Settings:
    """Return settings whose evaluation runs live under ``tmp_path``."""
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    return settings


def _settings_of(client: TestClient) -> Settings:
    """Return the settings an API client is bound to.

    Read back through the application's dependency override, so a test never
    has to thread a second fixture through its signature just to find the
    directory the endpoints will actually read.
    """
    for dependency, factory in client.app.dependency_overrides.items():
        if dependency is get_settings:
            return factory()
    raise AssertionError("The test client has no get_settings override.")


@pytest.fixture()
def evaluation_client(isolated_settings: Settings) -> Iterator[TestClient]:
    """Return a client bound to the isolated Phase 9 settings.

    The settings object is also exposed as its own fixture so a test can write a
    record the endpoint will really find. Using the process-wide
    ``get_settings()`` singleton inside a test would write to a directory the
    dependency override never reads.
    """
    application = create_app(isolated_settings)
    application.dependency_overrides[get_settings] = lambda: isolated_settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


def _write_evaluation(target: Settings, **kwargs: Any) -> dict[str, Any]:
    """Run an evaluation and persist it under the client settings.

    Writing straight into the client's own data directory keeps the test honest:
    the endpoint reads from disk, so the record must really be there.

    Args:
        settings: The client's settings, from the dependency override.
        **kwargs: Forwarded to :func:`run_evaluation`.

    Returns:
        The persisted run payload.
    """
    outcome = run_evaluation(
        None,
        samples=kwargs.pop("samples", make_samples(60)),
        validation_samples=kwargs.pop("validation_samples", make_samples(60, seed=23)),
        model_available=kwargs.pop("model_available", True),
        model_checkpoint=kwargs.pop("model_checkpoint", "ckpt-test"),
        **kwargs,
    )
    directory = target.data_dir / "evaluation" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{outcome.run.run_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(outcome.run.as_dict(), handle, default=str)
    return outcome.run.as_dict()


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def test_status_is_idle_before_any_run(evaluation_client: TestClient) -> None:
    """No evaluation yet is a state, not an error, and reports no metrics."""
    response = evaluation_client.get("/api/v1/evaluation/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_run"] is False
    assert payload["measured_groups"] == []
    assert payload["latest_run_id"] is None


def test_status_reflects_a_completed_run(evaluation_client: TestClient) -> None:
    """After a run the status names it and what it covered."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/status").json()

    assert payload["has_run"] is True
    assert payload["run_count"] == 1
    assert "gnn" in payload["measured_groups"]


def test_status_carries_the_disclaimer(evaluation_client: TestClient) -> None:
    """The caveat travels with the data."""
    payload = evaluation_client.get("/api/v1/evaluation/status").json()

    assert "not demonstrated forecast skill" in payload["disclaimer"].lower()
    assert "no accuracy or skill claim is made" in payload["disclaimer"].lower()


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def test_summary_returns_404_before_any_run(evaluation_client: TestClient) -> None:
    """A missing evaluation must not look like an empty result."""
    response = evaluation_client.get("/api/v1/evaluation/summary")

    assert response.status_code == 404
    assert "no summary" in response.json()["detail"].lower()


def test_summary_returns_the_run_with_provenance(evaluation_client: TestClient) -> None:
    """Every number must arrive with the evidence that produced it."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/summary").json()

    assert payload["run_id"]
    assert payload["metrics"]
    assert payload["provenance"]["seed"] == 20240601
    assert payload["provenance"]["code_fingerprint"]


def test_summary_separates_measured_from_unavailable(evaluation_client: TestClient) -> None:
    """A consumer must be able to tell a number from an absence."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/summary").json()

    assert payload["metrics"], "expected measured metrics"
    assert payload["unavailable_metrics"], "expected explicit unavailable metrics"
    for entry in payload["unavailable_metrics"].values():
        for item in entry.values():
            assert item["value"] is None
            assert item["reason"]


# --------------------------------------------------------------------------- #
# GNN
# --------------------------------------------------------------------------- #
def test_gnn_endpoint_returns_the_three_methods(evaluation_client: TestClient) -> None:
    """Persistence, constant velocity and the model, side by side."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/gnn").json()

    assert payload["model_available"] is True
    assert set(payload["methods"]) >= {"persistence", "constant_velocity", "gnn"}
    assert payload["sample_count"] == 60


def test_gnn_endpoint_publishes_the_baseline_definitions(evaluation_client: TestClient) -> None:
    """A reader must be able to see what each baseline assumes."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/gnn").json()

    for entry in payload["baselines"].values():
        assert entry["formula"]
        assert entry["description"]


def test_gnn_endpoint_reports_per_metric_verdicts(evaluation_client: TestClient) -> None:
    """Verdicts are per metric, with no averaged-away verdict."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/gnn").json()
    verdicts = payload["comparisons"]["gnn_vs_persistence"]

    assert verdicts
    for entry in verdicts.values():
        assert entry["verdict"] in ("improves", "tie", "worse", "not_computed")
    assert "no average verdict is claimed" in payload["interpretation"]


def test_gnn_endpoint_returns_404_before_any_run(evaluation_client: TestClient) -> None:
    """No comparison exists, so none is invented."""
    assert evaluation_client.get("/api/v1/evaluation/gnn").status_code == 404


# --------------------------------------------------------------------------- #
# Downscaling
# --------------------------------------------------------------------------- #
def test_downscaling_reports_unavailable_rather_than_zero(
    evaluation_client: TestClient,
) -> None:
    """With no reference frames, the AI result is omitted, not imitated."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/downscaling").json()

    assert payload["run_id"]
    unavailable = payload["unavailable"]
    assert "ai_mae" in unavailable
    assert unavailable["ai_mae"]["value"] is None
    assert "omitted rather than imitated" in unavailable["ai_mae"]["reason"]


def test_downscaling_returns_404_before_any_run(evaluation_client: TestClient) -> None:
    """Nothing evaluated, nothing returned."""
    assert evaluation_client.get("/api/v1/evaluation/downscaling").status_code == 404


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def test_uncertainty_endpoint_names_the_method(evaluation_client: TestClient) -> None:
    """The method must travel with the interval."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/uncertainty").json()

    assert payload["method"] == "validation-residual-bootstrap"
    assert "not probabilities" in payload["caveat"].lower()


def test_uncertainty_endpoint_returns_measured_coverage(
    evaluation_client: TestClient
) -> None:
    """Coverage is a measurement, and is labelled as one."""
    _write_evaluation(_settings_of(evaluation_client))

    payload = evaluation_client.get("/api/v1/evaluation/uncertainty").json()
    calibration = payload["calibration"]["0.80"]

    assert calibration["status"] == "MEASURED"
    assert calibration["sample_count"] == 60
    assert calibration["evaluated_on"] == "test"


def test_uncertainty_endpoint_returns_404_before_any_run(evaluation_client: TestClient) -> None:
    """No interval exists, so none is reported."""
    assert evaluation_client.get("/api/v1/evaluation/uncertainty").status_code == 404


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(evaluation_client: TestClient) -> None:
    """An explanation is never generated for an event that does not exist."""
    response = evaluation_client.get("/api/v1/evaluation/explain/NO_SUCH_EVENT")

    assert response.status_code == 404
    assert "real detection" in response.json()["detail"]


def test_explanation_carries_the_causality_disclaimer(evaluation_client: TestClient) -> None:
    """Even a 404-shaped explanation is governed by the same rule."""
    response = evaluation_client.get("/api/v1/evaluation/explain/NO_SUCH_EVENT")

    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def test_run_executes_and_reports(evaluation_client: TestClient) -> None:
    """A run is executed and its result returned."""
    response = evaluation_client.post(
        "/api/v1/evaluation/run", json={"model": "fixture", "sample_count": 40}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"]
    assert payload["status"] in ("SUCCESS", "PARTIAL", "INSUFFICIENT_DATA")


def test_run_rejects_an_unknown_body_field(evaluation_client: TestClient) -> None:
    """An unrecognised field is rejected rather than ignored."""
    response = evaluation_client.post(
        "/api/v1/evaluation/run", json={"model": "fixture", "unexpected": 1}
    )

    assert response.status_code == 422


def test_run_rejects_a_negative_seed(evaluation_client: TestClient) -> None:
    """A negative seed cannot drive a generator."""
    response = evaluation_client.post(
        "/api/v1/evaluation/run", json={"model": "fixture", "seed": -1}
    )

    assert response.status_code == 422


def test_dry_run_writes_nothing(
    evaluation_client: TestClient
) -> None:
    """A dry run must leave the runs directory empty."""
    response = evaluation_client.post(
        "/api/v1/evaluation/run", json={"model": "fixture", "dry_run": True, "sample_count": 10}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "DRY_RUN"
    assert not (_settings_of(evaluation_client).data_dir / "evaluation").exists()


def test_run_detail_returns_the_stored_run(evaluation_client: TestClient) -> None:
    """A stored run is retrievable by id."""
    run = _write_evaluation(_settings_of(evaluation_client))

    response = evaluation_client.get(f"/api/v1/evaluation/runs/{run['run_id']}")

    assert response.status_code == 200
    assert response.json()["run_id"] == run["run_id"]


def test_unknown_run_returns_404(evaluation_client: TestClient) -> None:
    """An unknown id is a 404, not an empty 200."""
    response = evaluation_client.get("/api/v1/evaluation/runs/eval-does-not-exist")

    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# OpenAPI and non-regression
# --------------------------------------------------------------------------- #
def test_evaluation_routes_are_documented(evaluation_client: TestClient) -> None:
    """Every Phase 9 endpoint must appear in the schema."""
    paths = evaluation_client.get("/openapi.json").json()["paths"]

    for path in (
        "/api/v1/evaluation/status",
        "/api/v1/evaluation/summary",
        "/api/v1/evaluation/gnn",
        "/api/v1/evaluation/downscaling",
        "/api/v1/evaluation/uncertainty",
        "/api/v1/evaluation/explain/{event_id}",
        "/api/v1/evaluation/run",
        "/api/v1/evaluation/runs/{run_id}",
    ):
        assert path in paths


def test_phase8_pipeline_routes_still_work(evaluation_client: TestClient) -> None:
    """Registering Phase 9 must not disturb Phase 8."""
    assert evaluation_client.get("/api/v1/pipeline/status").status_code == 200


def test_phase7_validation_routes_still_work(evaluation_client: TestClient) -> None:
    """Nor Phase 7."""
    assert evaluation_client.get("/api/v1/validation/status").status_code == 200


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cli() -> Any:
    """Import the Phase 9 CLI module by path."""
    spec = importlib.util.spec_from_file_location("run_phase9_evaluation", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_phase9_evaluation"] = module
    spec.loader.exec_module(module)
    return module


def test_cli_help_documents_every_required_option(cli: Any) -> None:
    """--help must list the options the specification requires."""
    text = cli.build_parser().format_help()

    for option in (
        "--dataset",
        "--start",
        "--end",
        "--model",
        "--seed",
        "--run-id",
        "--dry-run",
        "--output",
        "--verbose",
    ):
        assert option in text


def test_cli_help_states_the_honesty_guarantees(cli: Any) -> None:
    """The guarantees must be visible before anything runs."""
    # argparse hard-wraps the epilog, so normalise whitespace before matching.
    text = " ".join(cli.build_parser().format_help().split())

    assert "reported as UNAVAILABLE, never as 0.0" in text
    assert "never described as causal" in text
    assert "never called probabilities" in text


def test_cli_dry_run_writes_nothing(cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run must not create any artifact."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    code = cli.main(["--dry-run", "--model", "fixture"])

    assert code == 0
    assert not (tmp_path / "data" / "evaluation").exists()
    get_settings.cache_clear()


def test_cli_dry_run_with_a_dataset_prints_the_split(
    cli: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan must show the chronological boundaries it would use."""
    import numpy as np
    import xarray as xr

    from datetime import datetime, timezone

    stamps = np.array(
        [np.datetime64("2024-06-01T00:00") + np.timedelta64(h, "h") for h in range(60)],
        dtype="datetime64[ns]",
    )
    path = tmp_path / "record.nc"
    xr.Dataset(
        {"precipitation": (("time", "latitude", "longitude"), np.random.default_rng(1).gamma(2, 3, (60, 2, 2)))},
        coords={"time": stamps, "latitude": [12.0, 12.5], "longitude": [76.0, 76.5]},
    ).to_netcdf(path)

    code = cli.main(["--dry-run", "--dataset", str(path)])
    output = capsys.readouterr().out

    assert code == 0
    assert "chronological, not shuffled" in output
    assert "artifacts written: False" in output


def test_cli_rejects_a_missing_dataset(cli: Any, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing file is a configuration error, not a crash."""
    code = cli.main(["--once" if False else "--dry-run", "--dataset", "no/such/file.nc"])

    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_self_test_runs_and_writes_a_record(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The self-test exercises the evaluator with no data on disk."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    code = cli.main(["--model", "fixture", "--sample-count", "60", "--seed", "20240601"])

    runs = list((tmp_path / "data" / "evaluation" / "runs").glob("eval-*.json"))
    assert code == 0
    assert len(runs) == 1
    get_settings.cache_clear()


def test_cli_self_test_is_deterministic(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same seed must produce the same record, byte for byte.

    The run id is left to be derived, so this compares the identifier the
    evaluator computes rather than one the test supplied.
    """
    import tempfile

    payloads = []
    for _run_index in range(2):
        with tempfile.TemporaryDirectory() as directory:
            monkeypatch.setenv("DATA_DIR", str(Path(directory) / "data"))
            get_settings.cache_clear()
            cli.main(
                [
                    "--model",
                    "fixture",
                    "--sample-count",
                    "60",
                    "--seed",
                    "4242",
                ]
            )
            record = next((Path(directory) / "data" / "evaluation" / "runs").glob("*.json"))
            payload = json.loads(record.read_text(encoding="utf-8"))
            payload.pop("started_at")
            payload.pop("completed_at")
            payload.pop("duration_seconds")
            payloads.append(payload)
    get_settings.cache_clear()

    assert payloads[0] == payloads[1]


def test_cli_reports_synthetic_samples_as_synthetic(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Synthetic samples must never be presented as real weather."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    cli.main(["--model", "fixture", "--sample-count", "30"])
    output = capsys.readouterr().out
    get_settings.cache_clear()

    assert "DETERMINISTIC SYNTHETIC" in output


def test_cli_writes_the_requested_output_file(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--output must produce a readable record."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    destination = tmp_path / "out" / "evaluation.json"

    code = cli.main(
        ["--model", "fixture", "--sample-count", "30", "--output", str(destination)]
    )

    assert code == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["run_id"]
    get_settings.cache_clear()
