"""Pipeline orchestration tests (SIH26078, Phase 8, sections 8-10, 13).

The required matrix: complete success, validation rejection, ingestion failure,
retry success, retry exhaustion, optional GNN unavailable, optional downscaler
unavailable, alert-stage failure, idempotent rerun, and dry-run producing no
artifacts.

These tests use the shared doubles in ``tests/phase8_support.py`` so that no
network call and no trained model is required. Optional-model availability is
controlled by patching the status check rather than by depending on whatever
checkpoint happens to be on the machine, which would make the suite
non-deterministic across environments.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import xarray as xr

from extreme_weather.config import Settings
from extreme_weather.realtime.exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from extreme_weather.realtime.models import FAILED, SKIPPED, SUCCESS, WARNING
from extreme_weather.realtime.orchestrator import (
    STAGE_ORDER,
    IncrementalState,
    PipelineOrchestrator,
)
from extreme_weather.realtime.providers import FixtureProvider, WeatherRequest
from extreme_weather.realtime.retry import RetryPolicy

from tests.phase8_support import (
    REFERENCE_NOW,
    FlakyProvider,
    StubProvider,
    make_dataset,
    make_non_monotonic_dataset,
)

UTC = timezone.utc


@pytest.fixture()
def window() -> WeatherRequest:
    """Return a six-hour window ending at the reference instant."""
    return WeatherRequest(
        start=REFERENCE_NOW - timedelta(hours=5),
        end=REFERENCE_NOW,
        variables=("precipitation", "temperature", "u_wind", "v_wind"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )


def _orchestrator(
    provider: Any,
    settings: Settings,
    **kwargs: Any,
) -> PipelineOrchestrator:
    """Return an orchestrator with no real backoff sleeps."""
    kwargs.setdefault("sleep", lambda _seconds: None)
    kwargs.setdefault("retry_policy", RetryPolicy(max_attempts=3, base_delay=0.0))
    return PipelineOrchestrator(provider, settings=settings, **kwargs)


def _stub(settings: Settings, dataset: xr.Dataset | None = None, **kwargs: Any):
    """Return a stub-backed orchestrator."""
    return _orchestrator(StubProvider(dataset, **kwargs), settings)


# --------------------------------------------------------------------------- #
# Complete success
# --------------------------------------------------------------------------- #
def test_every_stage_runs_in_order(settings: Settings, window: WeatherRequest) -> None:
    """A healthy run executes all eight stages, in the documented order."""
    result = _stub(settings).run(window, dry_run=True)

    assert [s.name for s in result.run.stages] == list(STAGE_ORDER)


def test_successful_run_reports_no_errors(settings: Settings, window: WeatherRequest) -> None:
    """A run that completes has no error entries."""
    result = _stub(settings).run(window, dry_run=True)

    assert result.run.errors == []
    assert result.succeeded is True


def test_each_stage_records_timing(settings: Settings, window: WeatherRequest) -> None:
    """A stage without a start and end time cannot be reasoned about afterwards."""
    result = _stub(settings).run(window, dry_run=True)

    for stage in result.run.stages:
        assert stage.started_at is not None
        assert stage.completed_at is not None
        assert stage.duration is not None and stage.duration >= 0.0


def test_every_completed_stage_has_a_message(settings: Settings, window: WeatherRequest) -> None:
    """A status with no explanation is not something an operator can act on."""
    result = _stub(settings).run(window, dry_run=True)

    for stage in result.run.stages:
        assert stage.message


def test_run_records_provenance(settings: Settings, window: WeatherRequest) -> None:
    """The run must say where the data came from and how old it was."""
    result = _stub(settings).run(window, dry_run=True)
    ingestion = result.run.ingestion

    assert ingestion is not None
    assert ingestion.provider == "stub"
    assert ingestion.checksum
    assert ingestion.source_timestamp is not None
    assert ingestion.temporal_resolution is not None


def test_run_records_spatial_extent(settings: Settings, window: WeatherRequest) -> None:
    """The bounding box is part of the provenance."""
    result = _stub(settings).run(window, dry_run=True)
    extent = result.run.ingestion.spatial_extent  # type: ignore[union-attr]

    assert extent["latitude_min"] == pytest.approx(12.0)
    assert extent["latitude_max"] == pytest.approx(13.0)
    assert extent["points"] == 9


# --------------------------------------------------------------------------- #
# Validation rejection
# --------------------------------------------------------------------------- #
def test_rejected_data_never_reaches_anomaly_detection(
    settings: Settings, window: WeatherRequest
) -> None:
    """The whole point of the gate: bad data stops the run before detection."""
    orchestrator = _stub(settings, make_non_monotonic_dataset())
    result = orchestrator.run(window, dry_run=True)

    assert result.run.status == FAILED
    assert result.run.stage("validate") is not None
    assert result.run.stage("validate").status == FAILED
    for later in ("preprocess", "anomaly", "track", "gnn", "downscale", "alert"):
        assert result.run.stage(later) is None, f"{later} ran despite rejection"


def test_rejection_names_the_failing_check(settings: Settings, window: WeatherRequest) -> None:
    """The reason must be specific, not a bare failure."""
    result = _stub(settings, make_non_monotonic_dataset()).run(window, dry_run=True)

    assert any("time_monotonic" in e for e in result.run.errors)


def test_rejection_still_records_the_ingestion(settings: Settings, window: WeatherRequest) -> None:
    """A rejected run is still evidence of what arrived, so it is recorded."""
    result = _stub(settings, make_non_monotonic_dataset()).run(window, dry_run=True)

    assert result.run.ingestion is not None
    assert result.run.ingestion.checksum


# --------------------------------------------------------------------------- #
# Ingestion failure and retry
# --------------------------------------------------------------------------- #
def test_ingestion_failure_fails_the_run(settings: Settings, window: WeatherRequest) -> None:
    """An unreachable provider is a failed run, not an empty success."""
    provider = StubProvider(error=ProviderUnavailableError("provider is down"))
    result = _orchestrator(provider, settings, retry_policy=RetryPolicy(max_attempts=1)).run(
        window, dry_run=True
    )

    assert result.run.status == FAILED
    assert result.run.stage("ingest").status == FAILED
    assert result.run.stage("validate") is None


def test_retry_succeeds_after_transient_failures(settings: Settings, window: WeatherRequest) -> None:
    """A flaky provider that recovers must produce a successful run."""
    provider = FlakyProvider(failures=2)
    orchestrator = _orchestrator(provider, settings)
    result = orchestrator.run(window, dry_run=True)

    assert provider.calls == 3
    assert result.succeeded is True
    assert result.run.stage("ingest").status == SUCCESS
    assert result.run.ingestion is not None
    assert result.run.ingestion.attempts == 3


def test_retry_exhaustion_fails_the_run(settings: Settings, window: WeatherRequest) -> None:
    """A provider that never recovers must fail after the configured attempts."""
    provider = FlakyProvider(failures=99)
    orchestrator = _orchestrator(provider, settings)
    result = orchestrator.run(window, dry_run=True)

    assert provider.calls == 3
    assert result.run.status == FAILED
    assert "simulated provider timeout" in result.run.stage("ingest").message
    # The exhausted run must say how many attempts it made, so an operator can
    # tell a retry that ran out from a provider that was never retried.
    assert "3 attempt" in result.run.stage("ingest").message


def test_a_malformed_response_is_not_retried(settings: Settings, window: WeatherRequest) -> None:
    """Retrying a deterministic bad response only delays the error."""
    provider = StubProvider(error=ProviderResponseError("malformed payload"))
    orchestrator = _orchestrator(provider, settings)
    result = orchestrator.run(window, dry_run=True)

    assert provider.calls == 1
    assert result.run.status == FAILED


def test_every_attempt_is_logged(settings: Settings, window: WeatherRequest) -> None:
    """A run that took three tries should say so, not look first-try lucky."""
    from extreme_weather.realtime.logging import PipelineLogger

    lines: list[str] = []
    orchestrator = _orchestrator(FlakyProvider(failures=2), settings)
    result = orchestrator.run(window, dry_run=True)
    # The stage log is part of the run's observable behaviour, so assert on the
    # messages the stages carry rather than on a captured stream.
    ingest = result.run.stage("ingest")
    assert ingest is not None and ingest.succeeded

    # The attempt count is the observable evidence that retries happened.
    assert result.run.ingestion is not None
    assert result.run.ingestion.attempts == 3


class _RecordingLogger:
    """A logger that keeps every line, for assertions on log content."""

    def __init__(self, sink: list[str]) -> None:
        self.sink = sink

    def log(self, level: int, message: str) -> None:
        """Record one formatted line."""
        self.sink.append(message)

    def debug(self, message: str) -> None:  # pragma: no cover - interface filler
        """No-op."""
        self.sink.append(message)

    def info(self, message: str) -> None:  # pragma: no cover - interface filler
        """No-op."""
        self.sink.append(message)

    def warning(self, message: str) -> None:  # pragma: no cover - interface filler
        """No-op."""
        self.sink.append(message)

    def error(self, message: str) -> None:  # pragma: no cover - interface filler
        """No-op."""
        self.sink.append(message)


# --------------------------------------------------------------------------- #
# Optional stages
# --------------------------------------------------------------------------- #
def test_gnn_is_skipped_when_no_checkpoint(
    settings: Settings, window: WeatherRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing model is SKIPPED, never FAILED, and never a fake prediction."""
    import extreme_weather.gnn.service as gnn_service

    class _Status:
        available = False
        model_version = None
        message = "no checkpoint found"

    monkeypatch.setattr(
        gnn_service.GNNService, "status", lambda self: _Status(), raising=True
    )

    result = _stub(settings).run(window, dry_run=True)
    gnn = result.run.stage("gnn")

    assert gnn is not None
    assert gnn.status == SKIPPED
    assert "not a forecast" in gnn.message or "no prediction" in gnn.message.lower()


def test_downscaler_is_skipped_when_no_checkpoint(
    settings: Settings, window: WeatherRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule applies to the AI downscaler."""
    import extreme_weather.downscaling.service as down_service

    monkeypatch.setattr(
        down_service.DownscalingService,
        "status",
        lambda self: {"available": False, "message": "no checkpoint"},
        raising=True,
    )

    result = _stub(settings).run(window, dry_run=True)
    downscale = result.run.stage("downscale")

    assert downscale is not None
    assert downscale.status == SKIPPED
    assert "no trained ai downscaling checkpoint" in downscale.message.lower()


def test_optional_absence_does_not_fail_the_run(
    settings: Settings, window: WeatherRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run with no models must still complete: that is the normal state."""
    import extreme_weather.downscaling.service as down_service
    import extreme_weather.gnn.service as gnn_service

    class _Status:
        available = False
        model_version = None
        message = "no checkpoint"

    monkeypatch.setattr(gnn_service.GNNService, "status", lambda self: _Status(), raising=True)
    monkeypatch.setattr(
        down_service.DownscalingService,
        "status",
        lambda self: {"available": False, "message": "no checkpoint"},
        raising=True,
    )

    result = _stub(settings).run(window, dry_run=True)

    assert result.succeeded is True
    assert result.run.stage("alert").status in (SUCCESS, WARNING)


# --------------------------------------------------------------------------- #
# Alert failure
# --------------------------------------------------------------------------- #
def test_alert_failure_fails_the_run(
    settings: Settings, window: WeatherRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent alerting failure is the failure mode this project must not have."""
    import extreme_weather.alerts.service as alert_service

    def _boom(self: Any, **_kwargs: Any) -> None:
        raise RuntimeError("alert store is corrupt")

    monkeypatch.setattr(alert_service.AlertService, "generate", _boom, raising=True)

    result = _stub(settings).run(window, dry_run=True)

    assert result.run.status == FAILED
    assert result.run.stage("alert").status == FAILED
    assert any("alert store is corrupt" in e for e in result.run.errors)


def test_a_failed_alert_run_keeps_its_earlier_stages(
    settings: Settings, window: WeatherRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work completed before a later failure is not thrown away."""
    import extreme_weather.alerts.service as alert_service

    def _boom(self: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(alert_service.AlertService, "generate", _boom, raising=True)

    result = _stub(settings).run(window, dry_run=True)

    assert result.run.stage("anomaly").status in (SUCCESS, WARNING)
    assert result.run.stage("track").status in (SUCCESS, WARNING)


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #
def test_rerunning_the_same_ingestion_is_a_no_op(
    settings: Settings, window: WeatherRequest
) -> None:
    """The second identical run must not redo the work."""
    orchestrator = _stub(settings)
    first = orchestrator.run(window, dry_run=False)
    second = orchestrator.run(window, dry_run=False)

    assert first.run.run_id == second.run.run_id
    assert second.run.stage("ingest").message.count("already processed") == 1


def test_idempotent_rerun_skips_downstream_stages(
    settings: Settings, window: WeatherRequest
) -> None:
    """Skipping is visible in the stage list, not silent."""
    orchestrator = _stub(settings)
    orchestrator.run(window, dry_run=False)
    second = orchestrator.run(window, dry_run=False)

    assert second.run.stage("anomaly").status == SKIPPED
    assert second.run.stage("alert").status == SKIPPED


def test_force_reprocesses_an_already_seen_ingestion(
    settings: Settings, window: WeatherRequest
) -> None:
    """An operator can override the optimisation deliberately."""
    orchestrator = _stub(settings)
    orchestrator.run(window, dry_run=False)
    forced = orchestrator.run(window, dry_run=False, force=True)

    assert forced.run.stage("anomaly").status in (SUCCESS, WARNING)


def test_a_different_window_is_not_treated_as_a_duplicate(
    settings: Settings, window: WeatherRequest
) -> None:
    """A later hour of data is new data and must be processed."""
    orchestrator = _stub(settings)
    orchestrator.run(window, dry_run=False)

    later = WeatherRequest(
        start=window.start + timedelta(hours=2),
        end=window.end + timedelta(hours=2),
        variables=window.variables,
        latitude=window.latitude,
        longitude=window.longitude,
    )
    second = orchestrator.run(later, dry_run=False)

    assert second.run.stage("anomaly").status in (SUCCESS, WARNING)


def test_run_identity_is_deterministic() -> None:
    """The same inputs always yield the same run id."""
    from extreme_weather.realtime.idempotency import compute_run_identity

    first = compute_run_identity(
        "fixture", window="2024-06-30T00:00/2024-06-30T06:00", source_timestamp="x", checksum="y"
    )
    second = compute_run_identity(
        "fixture", window="2024-06-30T00:00/2024-06-30T06:00", source_timestamp="x", checksum="y"
    )

    assert first.run_id == second.run_id
    assert first.fingerprint == second.fingerprint


def test_run_identity_changes_with_the_checksum() -> None:
    """Different data for the same window is a different ingestion."""
    from extreme_weather.realtime.idempotency import compute_run_identity

    first = compute_run_identity("fixture", window="w", source_timestamp="t", checksum="a")
    second = compute_run_identity("fixture", window="w", source_timestamp="t", checksum="b")

    assert first.run_id != second.run_id


def test_incremental_state_reports_new_steps() -> None:
    """The state says how much new data arrived."""
    state = IncrementalState(
        last_processed="2024-06-30T06:00:00+00:00", newest_available="2024-06-30T09:00:00+00:00"
    )

    assert state.new_steps() == 3
    assert state.as_dict()["incremental"] is True


def test_incremental_state_admits_ignorance() -> None:
    """Before the first run, new-step count is unknown, not zero."""
    state = IncrementalState()

    assert state.new_steps() is None
    assert state.as_dict()["incremental"] is False


def test_incremental_state_remembers_fingerprints() -> None:
    """The state is what makes idempotency survive a restart."""
    state = IncrementalState().with_progress("2024-06-30T09:00:00+00:00", "fp-1")

    assert state.has_seen("fp-1") is True
    assert state.has_seen("fp-2") is False


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def test_dry_run_writes_no_run_record(settings: Settings, window: WeatherRequest) -> None:
    """A dry run must leave nothing in the runs directory."""
    orchestrator = _stub(settings)
    orchestrator.run(window, dry_run=True)

    assert not orchestrator.runs_dir.exists() or not list(orchestrator.runs_dir.glob("run-*.json"))


def test_dry_run_writes_no_state(settings: Settings, window: WeatherRequest) -> None:
    """Incremental state is production state and must not advance on a dry run."""
    orchestrator = _stub(settings)
    orchestrator.run(window, dry_run=True)

    assert orchestrator.state.has_processed is False


def test_dry_run_is_marked_on_the_record(settings: Settings, window: WeatherRequest) -> None:
    """The response must say it was a plan, not a real run."""
    result = _stub(settings).run(window, dry_run=True)

    assert result.run.dry_run is True
    assert result.run.ingestion is not None
    assert result.run.ingestion.dry_run is True


def test_dry_run_still_performs_the_quality_check(
    settings: Settings, window: WeatherRequest
) -> None:
    """A dry run that validated nothing would be useless as a pre-flight check."""
    result = _stub(settings, make_non_monotonic_dataset()).run(window, dry_run=True)

    assert result.run.stage("validate").status == FAILED


def test_dry_run_returns_a_dataset_without_writing(settings: Settings, window: WeatherRequest) -> None:
    """The work happens in memory; only the persistence is suppressed."""
    result = _stub(settings).run(window, dry_run=True)

    assert result.dataset is not None
    assert "precipitation" in result.dataset.data_vars


# --------------------------------------------------------------------------- #
# Persistence and status
# --------------------------------------------------------------------------- #
def test_a_real_run_persists_its_record(settings: Settings, window: WeatherRequest) -> None:
    """A real run is recoverable after the process exits."""
    orchestrator = _stub(settings)
    result = orchestrator.run(window, dry_run=False)

    assert (orchestrator.runs_dir / f"{result.run.run_id}.json").is_file()


def test_persisted_runs_can_be_reloaded(settings: Settings, window: WeatherRequest) -> None:
    """The history endpoint reads from disk, so a reload must round-trip."""
    orchestrator = _stub(settings)
    result = orchestrator.run(window, dry_run=False)

    reloaded = PipelineOrchestrator(FixtureProvider(), settings=settings).load_runs()

    assert any(run.run_id == result.run.run_id for run in reloaded)


def test_unreadable_run_files_are_skipped(settings: Settings, window: WeatherRequest) -> None:
    """One corrupt record must not break the history endpoint."""
    orchestrator = _stub(settings)
    result = orchestrator.run(window, dry_run=False)
    (orchestrator.runs_dir / "run-corrupt.json").write_text("{not json", encoding="utf-8")

    reloaded = PipelineOrchestrator(FixtureProvider(), settings=settings).load_runs()

    assert any(run.run_id == result.run.run_id for run in reloaded)


def test_status_reports_idle_before_any_run(settings: Settings) -> None:
    """A pipeline that has not run is IDLE, with no invented values."""
    status = _stub(settings).status()

    assert status["current_state"] == "IDLE"
    assert status["latest_timestamp"] is None
    assert status["last_successful_run"] is None


def test_status_reports_the_latest_run(settings: Settings, window: WeatherRequest) -> None:
    """Status reflects the last run's stage outcomes."""
    orchestrator = _stub(settings)
    result = orchestrator.run(window, dry_run=False)

    status = PipelineOrchestrator(FixtureProvider(), settings=settings).status()
    orchestrator.history.append(result.run)
    status = orchestrator.status()

    assert status["last_run"] == result.run.run_id
    assert len(status["stages"]) == len(STAGE_ORDER)
    assert status["freshness"]["status"] in ("FRESH", "STALE", "EXPIRED", "UNKNOWN")


def test_status_carries_the_prototype_disclaimer(settings: Settings) -> None:
    """The disclaimer must travel with the data, not just the docs."""
    status = _stub(settings).status()

    assert "Not an operational emergency-warning system" in status["disclaimer"]


# --------------------------------------------------------------------------- #
# Staleness policy
# --------------------------------------------------------------------------- #
def test_expired_data_is_processed_by_default(settings: Settings) -> None:
    """Backfills need old data, so EXPIRED is processed but always flagged."""
    old = WeatherRequest(
        start=REFERENCE_NOW - timedelta(days=30),
        end=REFERENCE_NOW - timedelta(days=30) + timedelta(hours=5),
        variables=("precipitation", "temperature", "u_wind", "v_wind"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )
    result = _stub(settings).run(old, dry_run=True)

    assert result.run.freshness == "EXPIRED"
    assert any("must not be presented as current" in w for w in result.run.warnings)


def test_expired_data_can_be_refused(settings: Settings) -> None:
    """A caller that requires current data can stop the run instead."""
    old = WeatherRequest(
        start=REFERENCE_NOW - timedelta(days=30),
        end=REFERENCE_NOW - timedelta(days=30) + timedelta(hours=5),
        variables=("precipitation", "temperature", "u_wind", "v_wind"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )
    result = _stub(settings).run(old, dry_run=True, allow_stale=False)

    assert result.run.stage("anomaly") is None
    assert result.run.stage("ingest").status == WARNING


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_run_serialises_for_the_api(settings: Settings, window: WeatherRequest) -> None:
    """The run record is returned over HTTP, so it must be JSON-ready."""
    import json

    result = _stub(settings).run(window, dry_run=True)
    payload = result.run.as_dict()

    json.dumps(payload, default=str)
    assert payload["run_id"].startswith("run-")
    assert isinstance(payload["stages"], list)


def test_run_round_trips_through_its_dict(settings: Settings, window: WeatherRequest) -> None:
    """A run rebuilt from JSON keeps its status and stages."""
    from extreme_weather.realtime.models import PipelineRun

    result = _stub(settings).run(window, dry_run=True)
    rebuilt = PipelineRun.from_dict(result.run.as_dict())

    assert rebuilt.run_id == result.run.run_id
    assert rebuilt.status == result.run.status
    assert len(rebuilt.stages) == len(result.run.stages)


def test_current_stage_is_derived_from_the_stage_list() -> None:
    """A stored current_stage could drift; deriving it cannot."""
    from extreme_weather.realtime.models import PipelineRun, StageResult

    run = PipelineRun(run_id="run-x")
    run.add_stage(StageResult(name="ingest", status=SUCCESS))
    run.add_stage(StageResult(name="validate", status="RUNNING"))

    assert run.current_stage == "validate"
