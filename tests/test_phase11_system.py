"""Phase 11 system status and demo tests (SIH26078).

Covers the three things Phase 11 adds:

* ``GET /api/v1/system/status`` - the single "what is this system" page;
* ``scripts/run_sih_demo.py`` - the end-to-end demonstration command;
* ``scripts/verify_reproducibility.py`` - the determinism check.

The status tests exist mainly to enforce two rules that are easy to break and
expensive to get wrong: an unavailable quantity must be ``null`` rather than
``0``, and the endpoint must never leak a secret or a stack trace.
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

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> Any:
    """Import a script from ``scripts/``, which is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo() -> Any:
    """Return the loaded demo script."""
    return _load("run_sih_demo", "scripts/run_sih_demo.py")


@pytest.fixture(scope="module")
def reproducibility() -> Any:
    """Return the loaded reproducibility script."""
    return _load("verify_reproducibility", "scripts/verify_reproducibility.py")


@pytest.fixture()
def status_settings(tmp_path: Path) -> Settings:
    """Return isolated settings for the status endpoint."""
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    return settings


@pytest.fixture()
def status_client(status_settings: Settings) -> Iterator[TestClient]:
    """Return a client bound to :func:`status_settings`."""
    application = create_app(status_settings)
    application.dependency_overrides[get_settings] = lambda: status_settings
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Normal response
# --------------------------------------------------------------------------- #


def test_system_status_answers_200(status_client: TestClient) -> None:
    """The page a judge opens first must always answer."""
    response = status_client.get("/api/v1/system/status")

    assert response.status_code == 200


def test_system_status_reports_its_identity(status_client: TestClient) -> None:
    """Version, phase and environment must be present and truthful."""
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["project"] == "SIH26078"
    assert payload["service"] == "DeepWeather AI"
    assert payload["version"]
    assert payload["phase"] == "11"
    assert payload["environment"]


def test_system_status_carries_a_readiness_verdict(status_client: TestClient) -> None:
    """It must not invent its own readiness; it reuses the operations one."""
    payload = status_client.get("/api/v1/system/status").json()
    ready = status_client.get("/ready").json()

    assert payload["readiness"] in ("READY", "DEGRADED", "NOT_READY")
    assert payload["readiness"] == ready["state"]
    assert payload["ready"] == ready["ready"]


def test_system_status_always_carries_the_disclaimer(status_client: TestClient) -> None:
    """A client must not be able to render this page as operational guidance."""
    payload = status_client.get("/api/v1/system/status").json()

    assert "Research prototype" in payload["disclaimer"]
    assert "not an operational" in payload["disclaimer"].lower()
    assert "lives saved" in payload["disclaimer"]


def test_system_status_makes_no_accuracy_claim(status_client: TestClient) -> None:
    """No *value* on the page may present itself as a skill or accuracy figure.

    The disclaimer legitimately contains the word "accuracy" in order to deny
    such a claim, so the prose is excluded and only the structured data is
    checked. A key named ``accuracy`` would be a real violation; a sentence
    saying "no accuracy has been measured" is the opposite.
    """
    payload = status_client.get("/api/v1/system/status").json()

    forbidden = {"accuracy", "precision", "recall", "f1", "f1_score", "skill_score"}
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                assert str(key).lower() not in forbidden, f"status exposes {key!r}"
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)

    # And the disclaimer must be the thing that carries the caveat.
    assert "no forecast skill" in payload["disclaimer"].lower()


def test_system_status_reports_a_data_source(status_client: TestClient) -> None:
    """The data kind must be explicit, and must be a known label."""
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["data_source"]["kind"] in ("SYNTHETIC", "HISTORICAL", "REALTIME_FIXTURE")
    assert payload["data_source"]["description"]


def test_system_status_reports_ingestion_and_counts(status_client: TestClient) -> None:
    """Ingestion state and the produced counts must both be present."""
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["ingestion"]["name"] == "ingestion"
    assert "available" in payload["ingestion"]
    for key in ("events", "tracks", "alerts", "datasets", "pipeline_runs"):
        assert key in payload["counts"]


def test_system_status_reports_both_optional_models(status_client: TestClient) -> None:
    """GNN and downscaling are optional; both must be reported either way."""
    payload = status_client.get("/api/v1/system/status").json()

    assert "gnn" in payload["models"]
    assert "downscaling" in payload["models"]
    for component in payload["models"].values():
        assert "available" in component
        assert "state" in component


def test_system_status_reports_evaluation_state(status_client: TestClient) -> None:
    """Evaluation must be reported, including when nothing was measured."""
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["evaluation"]["name"] == "evaluation"
    assert payload["evaluation"]["available"] in (True, False)


# --------------------------------------------------------------------------- #
# Unavailable is null, never zero
# --------------------------------------------------------------------------- #


def test_missing_counts_are_null_not_zero(status_client: TestClient) -> None:
    """Nothing on record must read as ``null``, never as ``0``.

    "No events file" and "zero events were found" are different facts. A
    dashboard that cannot tell them apart is wrong by accident, and ``0`` is
    the specific value that misleads.
    """
    payload = status_client.get("/api/v1/system/status").json()

    for key in ("events", "tracks", "alerts", "pipeline_runs", "evaluation_runs"):
        assert payload["counts"][key] is None, (
            f"{key} should be null when nothing is stored, "
            f"not {payload['counts'][key]!r}"
        )


def test_no_data_is_reported_as_such_with_a_reason(
    status_client: TestClient,
) -> None:
    """A fresh install must say "no data", and say what to do about it."""
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["ingestion"]["available"] is False
    assert payload["ingestion"]["state"] == "NO_DATA"
    assert payload["ingestion"]["reason"]


def test_no_evaluation_is_reported_as_not_evaluated(
    status_client: TestClient,
) -> None:
    """A missing evaluation must not look like a perfect or a zero score."""
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["evaluation"]["available"] is False
    assert payload["evaluation"]["state"] == "NOT_EVALUATED"
    assert "accuracy" in payload["evaluation"]["reason"].lower()


def test_stored_counts_are_reported_when_present(
    status_client: TestClient, status_settings: Settings
) -> None:
    """A stored file must be counted, not reported as unknown.

    Written through the real store so the fixture satisfies the same schema the
    pipeline produces; a hand-rolled payload would test the test, not the code.
    """
    from extreme_weather.alerts.models import AlertStoreResult
    from extreme_weather.alerts.store import AlertStore

    result = AlertStoreResult()
    result.alerts = []  # a valid, empty, well-formed store
    AlertStore(status_settings.processed_dir / status_settings.alerts_filename).save(
        result
    )
    payload = status_client.get("/api/v1/system/status").json()

    # An empty but valid store is a *measured* zero, not "unknown".
    assert payload["counts"]["alerts"] == 0
    assert payload["ingestion"]["available"] is True


def test_a_corrupt_store_is_unknown_not_zero(
    status_client: TestClient, status_settings: Settings
) -> None:
    """A file that cannot be parsed must not be reported as an empty result."""
    (status_settings.processed_dir / status_settings.tracks_filename).write_text(
        "{not json", encoding="utf-8"
    )
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["counts"]["tracks"] is None


# --------------------------------------------------------------------------- #
# Degraded state
# --------------------------------------------------------------------------- #


def test_readiness_is_never_ready_without_every_component(
    status_client: TestClient,
) -> None:
    """A missing model must never be reported as READY.

    The repository ships trained checkpoints, so this checkout legitimately
    reports READY. The rule being asserted is the conditional one: whenever a
    model *is* unavailable, the verdict must not be READY.
    """
    payload = status_client.get("/api/v1/system/status").json()
    models_available = [c["available"] for c in payload["models"].values()]

    assert models_available, "the status page reported no models at all"
    if not all(models_available):
        assert payload["readiness"] != "READY"
        assert payload["ready"] is False
    else:
        # Every component available: READY is then a legitimate verdict.
        assert payload["readiness"] in ("READY", "DEGRADED")


def test_missing_models_are_reported_unavailable_not_failed(
    status_client: TestClient,
) -> None:
    """An absent checkpoint is UNAVAILABLE with a reason, never an error state."""
    payload = status_client.get("/api/v1/system/status").json()

    for name, component in payload["models"].items():
        if not component["available"]:
            assert component["state"] == "UNAVAILABLE", name
            assert component["reason"], f"{name} is unavailable with no reason given"


def test_alerting_reports_its_configuration(
    status_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alerting availability must reflect the real configuration switch."""
    from extreme_weather.config import get_settings as real_get_settings

    settings = real_get_settings()
    monkeypatch.setattr(settings, "alert_generation_enabled", False, raising=False)
    payload = status_client.get("/api/v1/system/status").json()

    assert payload["alerting"]["state"] == "DISABLED"
    assert payload["alerting"]["available"] is False
    assert "503" in payload["alerting"]["reason"]


# --------------------------------------------------------------------------- #
# Safe serialization
# --------------------------------------------------------------------------- #


def test_no_environment_secret_is_exposed(status_client: TestClient) -> None:
    """No environment value may reach the response.

    A status endpoint composes many services; if any of them echoed an
    environment variable, that value would become a public API response.
    """
    body = json.dumps(status_client.get("/api/v1/system/status").json())

    for marker in ("os.environ", "environ[", "API_KEY", "SECRET", "PASSWORD", "TOKEN"):
        assert marker not in body


# --------------------------------------------------------------------------- #
# Configuration audit
# --------------------------------------------------------------------------- #


def _documented_env_vars() -> set[str]:
    """Every ``NAME=`` assignment in ``.env.example`` or ``.env``."""
    names: set[str] = set()
    for filename in (".env.example", ".env"):
        path = ROOT / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name = stripped.split("=", 1)[0].strip()
            if name and name.replace("_", "").isalnum():
                names.add(name)
    return names


def test_every_documented_env_var_actually_exists() -> None:
    """A documented variable that is never read is a configuration lie.

    The brief is explicit: no fake or dead configuration keys. This is checked
    automatically rather than by reading, because a key can stop being read
    without anybody noticing in review.
    """
    documented = _documented_env_vars()
    known = {name.upper() for name in Settings.model_fields}

    missing = documented - known
    assert not missing, (
        "documented in .env.example but not a Settings field: "
        f"{', '.join(sorted(missing))}"
    )


def test_the_env_example_commits_no_secret() -> None:
    """A committed secret is a real incident, not a style issue."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8").lower()

    # A commented-out key with a plausible value would still be a leak waiting
    # to be uncommented, so the values are checked too.
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        lowered = name.strip().lower()
        if any(
            marker in lowered
            for marker in ("key", "secret", "password", "token", "credential")
        ):
            assert not value.strip(), f"{name.strip()} has a committed value"


def test_no_absolute_machine_path_is_committed() -> None:
    """Configuration must be portable, not tied to one developer's machine."""
    offenders: list[str] = []
    for relative in (".env.example", "docker-compose.yml", "Dockerfile"):
        path = ROOT / relative
        if not path.is_file():
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for marker in ("C:\\", "/Users/", "/home/"):
                if marker in line and "appuser" not in line:
                    offenders.append(f"{relative}:{number}")
    assert not offenders, f"machine-specific absolute paths at {offenders}"


# --------------------------------------------------------------------------- #
# Security audit (Phase 10 mechanisms, exercised against the new surface)
# --------------------------------------------------------------------------- #


def test_the_status_endpoint_survives_a_hostile_path(
    status_client: TestClient,
) -> None:
    """A traversal attempt must not read outside the data directory.

    The status page composes stores that take paths, so it is a plausible place
    for a traversal to reappear even though `security.safe_path` already
    guards everything else.
    """
    response = status_client.get(
        "/api/v1/system/status?path=../../../../etc/passwd"
    )

    assert response.status_code == 200
    body = response.text
    assert "root:" not in body
    assert "Traceback" not in body


def test_the_status_endpoint_rejects_an_overlong_path(
    status_client: TestClient,
) -> None:
    """An absurdly long query string must not be echoed or crash the page."""
    response = status_client.get("/api/v1/system/status?path=" + "A" * 10_000)

    assert response.status_code in (200, 422)
    assert "Traceback" not in response.text


def test_credential_shaped_strings_are_redacted_everywhere(
    status_client: TestClient,
) -> None:
    """The redaction backstop must cover the new endpoint too.

    A secret can reach a response through a service that echoes its own
    configuration, so this asserts on the live response rather than on the
    redactor in isolation.
    """
    body = status_client.get("/api/v1/system/status").text

    assert "Bearer " not in body
    assert "api_key=" not in body
    assert "password=" not in body


def test_the_error_envelope_hides_internals_on_this_route(
    status_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash inside the status page must not leak a traceback to the client.

    This is the failure mode the envelope exists for: the page is what an
    operator opens when something is already wrong.
    """
    from extreme_weather.api.routers import system as system_router

    def explode(settings: object) -> None:
        raise RuntimeError("internal detail that must not escape")

    monkeypatch.setattr(system_router, "build_system_status", explode)
    response = status_client.get("/api/v1/system/status")

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"] == "internal_error"
    assert "internal detail" not in response.text
    assert "Traceback" not in response.text
    assert payload["request_id"]


def test_no_wildcard_cors_by_default(status_client: TestClient) -> None:
    """A wildcard origin lets any website read this API from a browser."""
    response = status_client.get(
        "/api/v1/system/status", headers={"Origin": "https://attacker.example"}
    )

    assert response.headers.get("access-control-allow-origin") is None



def test_no_stack_trace_or_filesystem_path_leaks(
    status_client: TestClient, status_settings: Settings
) -> None:
    """The response must not carry a traceback or an absolute data path."""
    body = status_client.get("/api/v1/system/status").text

    assert "Traceback" not in body
    assert 'File "' not in body
    resolved = str(status_settings.data_dir.resolve()).replace("\\", "/")
    assert resolved not in body.replace("\\", "/")


def test_the_status_page_survives_a_corrupt_run_record(
    status_client: TestClient, status_settings: Settings
) -> None:
    """A status page that 500s is useless exactly when something is broken."""
    runs = status_settings.data_dir / "realtime" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "broken.json").write_text("{not json", encoding="utf-8")

    response = status_client.get("/api/v1/system/status")

    assert response.status_code == 200
    # The bad record is ignored rather than reported as data.
    assert response.json()["data_source"]["latest_timestamp"] is None


def test_status_is_covered_by_the_request_id_middleware(
    status_client: TestClient,
) -> None:
    """A status page a judge loads must be traceable in the log."""
    response = status_client.get("/api/v1/system/status")

    assert response.headers.get("X-Request-ID")


# --------------------------------------------------------------------------- #
# The orchestrator settings bug this phase found
# --------------------------------------------------------------------------- #


def test_the_orchestrator_persists_tracks(
    status_settings: Settings, tmp_path: Path
) -> None:
    """A successful run must leave ``tracks.json`` on disk.

    Regression test. ``TrackingService.run()`` defaults to
    ``write_output=False``, and the orchestrator called it without that flag:
    tracks were computed, returned in memory, and then thrown away. The run
    reported ``track_count: 1`` while ``GET /api/v1/tracks`` stayed empty - a
    green pipeline producing no track data.
    """
    from datetime import datetime, timedelta, timezone

    from extreme_weather.realtime.orchestrator import PipelineOrchestrator
    from extreme_weather.realtime.providers import WeatherRequest, build_provider

    sample = ROOT / "data" / "raw" / "sample_history.nc"
    assert sample.is_file(), (
        "the bundled sample dataset is missing, so this run would detect no "
        "events and the assertion below could not distinguish a fixed bug from "
        "an absent fixture"
    )

    end = datetime(2024, 6, 3, tzinfo=timezone.utc)
    result = PipelineOrchestrator(
        build_provider("local", local_path=sample), settings=status_settings
    ).run(
        WeatherRequest(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=end,
            variables=("precipitation", "temperature"),
            latitude=(12.0, 12.5, 13.0),
            longitude=(76.0, 76.5, 77.0),
        ),
        dry_run=False,
        force=True,
    )

    assert result.succeeded, result.run.errors
    track_stage = result.run.stage("track")
    assert track_stage is not None
    count = track_stage.output.get("track_count", 0)
    assert count > 0, "the run reported no tracks"
    # The file must exist, or the computed tracks were discarded.
    tracks_file = status_settings.processed_dir / status_settings.tracks_filename
    assert tracks_file.is_file(), "tracks.json was not written by a successful run"

    from extreme_weather.tracking.store import TrackStore

    assert len(TrackStore(tracks_file).load().tracks) == count


def test_orchestrator_stores_follow_its_own_settings(tmp_path: Path) -> None:
    """A run must write into the settings it was given.

    Regression test. The orchestrator accepted a ``settings`` argument but
    built its event, track and alert stores through services that defaulted to
    the process-wide ``get_settings()``. A run pointed at a scratch directory
    therefore wrote its artifacts into the *repository's* data directory,
    silently ignoring the configuration it was handed.

    The observable symptom: after an isolated run, the scratch directory was
    empty and the repository's ``events.json`` had grown.
    """
    from datetime import datetime, timedelta, timezone

    from extreme_weather.config import Settings
    from extreme_weather.realtime.orchestrator import PipelineOrchestrator
    from extreme_weather.realtime.providers import WeatherRequest, build_provider

    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()

    end = datetime(2024, 6, 3, tzinfo=timezone.utc)
    orchestrator = PipelineOrchestrator(
        build_provider("fixture"), settings=settings
    )
    result = orchestrator.run(
        WeatherRequest(
            start=end - timedelta(hours=6),
            end=end,
            variables=("precipitation", "temperature"),
            latitude=(12.0, 12.5, 13.0),
            longitude=(76.0, 76.5, 77.0),
        ),
        dry_run=False,
        force=True,
    )

    assert result.succeeded, result.run.errors
    # The events file must exist under the settings that were passed in.
    assert (settings.processed_dir / settings.events_filename).is_file()


def test_an_isolated_orchestrator_run_writes_nothing_to_the_repository(
    tmp_path: Path,
) -> None:
    """The isolation must be real, not merely apparent from the run record.

    This is the same defect seen from the other side: the repository's own
    artifacts must be byte-identical before and after an isolated run.
    """
    from datetime import datetime, timedelta, timezone

    from extreme_weather.config import Settings
    from extreme_weather.realtime.orchestrator import PipelineOrchestrator
    from extreme_weather.realtime.providers import WeatherRequest, build_provider

    def _fingerprint(root: Path) -> dict[str, int]:
        """Map every file under ``root`` to its size, for a before/after diff."""
        if not root.is_dir():
            return {}
        return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}

    repo_data = Path("data").resolve()
    before = _fingerprint(repo_data)

    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()

    end = datetime(2024, 6, 3, tzinfo=timezone.utc)
    PipelineOrchestrator(build_provider("fixture"), settings=settings).run(
        WeatherRequest(
            start=end - timedelta(hours=6),
            end=end,
            variables=("precipitation", "temperature"),
            latitude=(12.0, 12.5, 13.0),
            longitude=(76.0, 76.5, 77.0),
        ),
        dry_run=False,
        force=True,
    )

    after = _fingerprint(repo_data)
    assert after == before, (
        "an isolated orchestrator run modified the repository's data directory"
    )
    # And the artifacts really did land in the scratch directory.
    assert (settings.processed_dir / settings.events_filename).is_file()


# --------------------------------------------------------------------------- #
# The demo command
# --------------------------------------------------------------------------- #


def test_the_demo_dry_run_writes_nothing(demo: Any, tmp_path: Path) -> None:
    """A dry run must leave no trace, in a scratch directory.

    This is what makes ``--dry-run`` the safe first command on a fresh clone.
    """
    runner = demo.DemoRunner(dry_run=True, provider="fixture")
    runner.settings = _redirect(runner, tmp_path)
    runner.run_all()

    assert runner.failed == []
    assert not list((tmp_path / "data" / "realtime" / "runs").glob("*.json"))


def _redirect(runner: Any, tmp_path: Path) -> Any:
    """Point a runner's settings at a scratch directory."""
    from extreme_weather.config import Settings

    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    settings.ensure_directories()
    runner.settings = settings
    runner._raw_before = runner._snapshot_raw()
    return settings


def test_the_demo_reports_every_required_field(demo: Any, tmp_path: Path) -> None:
    """The summary must carry every field the brief requires.

    A demo that silently omits, say, the run id leaves a judge guessing, and an
    omitted field is indistinguishable from a zero.
    """
    runner = demo.DemoRunner(dry_run=True, provider="fixture")
    _redirect(runner, tmp_path)
    runner.run_all()
    report = runner.summary()

    assert report["project"] == "SIH26078"
    assert report["run_id"]
    assert "SYNTHETIC" in report["data_source_label"]
    assert "Research prototype" in report["disclaimer"]

    names = {stage["name"] for stage in report["stages"]}
    for required in ("pipeline", "tracks", "graph", "models", "alerts", "evaluation"):
        assert required in names, f"the demo omitted the {required} stage"


def test_the_demo_prints_unavailable_never_zero(demo: Any) -> None:
    """A quantity that was not measured must print as unavailable.

    This is the single most important honesty property of the demo output.
    """
    assert demo._format_value(None) == "unavailable"
    assert demo._format_value(0) == "0"
    assert demo._format_value(False) == "no"


def test_the_demo_summary_carries_the_disclaimer(demo: Any) -> None:
    """The rendered summary must state the prototype limitation."""
    rendered = demo.render_summary(demo.DemoRunner(dry_run=True, provider="fixture"))

    assert "PROTOTYPE DISCLAIMER" in rendered
    assert "not an operational" in rendered.lower()
    assert "no forecast skill" in rendered.lower()


def test_the_demo_refuses_contradictory_flags(demo: Any) -> None:
    """--dry-run and --clean contradict each other and must be rejected."""
    assert demo.main(["--dry-run", "--clean"]) == 2


def test_the_demo_exposes_help(demo: Any) -> None:
    """The CLI must document itself, like every other entry point."""
    import argparse

    parser = demo.build_parser()
    text = parser.format_help()

    assert isinstance(parser, argparse.ArgumentParser)
    for flag in ("--dry-run", "--clean", "--reuse-existing", "--verbose"):
        assert flag in text


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_volatile_fields_are_stripped_before_comparison(
    reproducibility: Any,
) -> None:
    """Timestamps and durations must not be treated as reproducibility bugs."""
    payload = {
        "generated_at": "2024-06-01T00:00:00Z",
        "duration": 1.25,
        "node_count": 8,
    }
    stripped = reproducibility.strip_volatile(payload)

    assert "generated_at" not in stripped
    assert "duration" not in stripped
    assert stripped["node_count"] == 8


def test_identical_records_compare_equal(reproducibility: Any) -> None:
    """Two equal records must produce a PASS, not a FAIL."""
    record = {"a": 1, "generated_at": "x", "b": [1, 2]}

    check = reproducibility._compare_payload(
        "x", record, {"a": 1, "generated_at": "y", "b": [1, 2]}
    )

    assert check.status == reproducibility.PASS


def test_differing_records_fail_with_the_field_named(
    reproducibility: Any,
) -> None:
    """A difference must be reported with the field that caused it."""
    check = reproducibility._compare_payload(
        "x", {"node_count": 8}, {"node_count": 9}
    )

    assert check.status == reproducibility.FAIL
    assert "node_count" in check.detail


def test_missing_input_is_not_applicable_not_pass(
    reproducibility: Any,
) -> None:
    """NOT_APPLICABLE must never be reported as a pass.

    A check that silently degrades to "fine" when it had nothing to compare is
    worse than no check at all.
    """
    check = reproducibility._compare_ids("x", None, ["A"])

    assert check.status == reproducibility.NA
    assert check.passed is False


def test_the_reproducibility_script_runs_and_reports(
    reproducibility: Any,
) -> None:
    """The whole script must complete and produce no FAIL.

    The pipeline runs are slow, so this exercises the cheap, pure checks and
    the comparison logic rather than re-running the pipeline twice.
    """
    first = reproducibility.check_evaluation_run_id()
    second = reproducibility.check_code_fingerprint()

    for check in (first, second):
        assert check.status != reproducibility.FAIL, check.detail


# --------------------------------------------------------------------------- #
# The validation-record generator
# --------------------------------------------------------------------------- #


def test_the_reported_verdicts_are_machine_readable(
    reproducibility: Any,
) -> None:
    """A verdict line must be parseable, or the record silently reads zero.

    Regression test. The report right-aligned its status inside brackets
    ("[           PASS]"), so a consumer matching the literal "[PASS]" found
    nothing and reported "0 passed" - which looks exactly like a pass. A
    validation record that silently reports nothing is worse than none.
    """
    import re

    checks = [
        reproducibility.Check(name="a", status=reproducibility.PASS, detail="ok"),
        reproducibility.Check(name="b", status=reproducibility.FAIL, detail="bad"),
        reproducibility.Check(name="c", status=reproducibility.NA, detail="n/a"),
    ]
    rendered = reproducibility.render(checks)

    for status in (reproducibility.PASS, reproducibility.FAIL, reproducibility.NA):
        assert re.search(rf"\[\s*{re.escape(status)}\s*\]", rendered), (
            f"{status} is not machine-readable in the rendered report"
        )
    # And the counts must be on a single, greppable line.
    assert re.search(r"RESULT:\s*\d+ passed", rendered)
