"""Scheduler, retry, logging and CLI tests (SIH26078, Phase 8, sections 3, 9, 12, 14).

These cover the parts of Phase 8 that sit around the pipeline rather than inside
it: the retry policy's rules, the credential redaction guarantee, the scheduler's
loop behaviour, and the CLI's argument handling and dry-run guarantee.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from extreme_weather.config import Settings
from extreme_weather.realtime.exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from extreme_weather.realtime.logging import (
    STAGE_LABELS,
    PipelineLogger,
    get_logger,
    log_stage,
    redact,
)
from extreme_weather.realtime.models import redact_url
from extreme_weather.realtime.orchestrator import PipelineOrchestrator
from extreme_weather.realtime.providers import FixtureProvider, WeatherRequest
from extreme_weather.realtime.retry import RetryPolicy, retry_call
from extreme_weather.realtime.scheduler import (
    RealtimeScheduler,
    SchedulerConfig,
)

from tests.phase8_support import REFERENCE_NOW

UTC = timezone.utc

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "run_realtime_pipeline.py"


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


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
def test_retry_returns_immediately_on_first_success() -> None:
    """A working call is not retried."""
    calls = {"n": 0}

    def _ok() -> str:
        calls["n"] += 1
        return "ok"

    result, attempts = retry_call(_ok, RetryPolicy(max_attempts=3, base_delay=0.0))

    assert result == "ok"
    assert calls["n"] == 1
    assert len(attempts) == 1
    assert attempts[0].succeeded is True


def test_retry_recovers_after_two_timeouts() -> None:
    """A transient failure is retried and can succeed."""
    state = {"n": 0}

    def _flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise ProviderTimeoutError("timed out")
        return "ok"

    result, attempts = retry_call(
        _flaky, RetryPolicy(max_attempts=3, base_delay=0.0), sleep=lambda _s: None
    )

    assert result == "ok"
    assert len(attempts) == 3
    assert [a.succeeded for a in attempts] == [False, False, True]


def test_retry_gives_up_after_the_configured_attempts() -> None:
    """A permanently failing call raises once attempts are exhausted."""
    def _always() -> None:
        raise ProviderUnavailableError("down")

    with pytest.raises(ProviderUnavailableError):
        retry_call(
            _always, RetryPolicy(max_attempts=3, base_delay=0.0), sleep=lambda _s: None
        )


def test_retry_does_not_repeat_a_malformed_response() -> None:
    """A deterministic bad response will be bad again, so retrying only delays."""
    calls = {"n": 0}

    def _bad() -> None:
        calls["n"] += 1
        raise ProviderResponseError("malformed")

    with pytest.raises(ProviderResponseError):
        retry_call(_bad, RetryPolicy(max_attempts=5, base_delay=0.0), sleep=lambda _s: None)

    assert calls["n"] == 1


def test_backoff_grows_exponentially() -> None:
    """Each delay is the previous one multiplied, up to the cap."""
    policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0, max_delay=100.0)

    delays = [policy.delay_for(n) for n in range(1, 5)]

    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped() -> None:
    """A long-running run must not stall on an unbounded delay."""
    policy = RetryPolicy(base_delay=1.0, backoff_factor=10.0, max_delay=5.0)

    assert policy.delay_for(5) == 5.0


def test_zero_attempts_is_refused() -> None:
    """A policy that never runs is a configuration error."""
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        RetryPolicy(max_attempts=0)


def test_shrinking_backoff_is_refused() -> None:
    """A factor below 1 would make retries fire faster than the first attempt."""
    with pytest.raises(ValueError, match="backoff_factor must be >= 1.0"):
        RetryPolicy(backoff_factor=0.5)


def test_every_attempt_is_reported_to_the_callback() -> None:
    """A caller can log the retry history as it happens."""
    seen: list[Any] = []

    retry_call(
        lambda: "ok",
        RetryPolicy(max_attempts=2),
        on_attempt=seen.append,
    )

    assert len(seen) == 1
    assert seen[0].number == 1


def test_a_failed_run_is_never_reported_as_complete() -> None:
    """Either the call returns a real result or the error propagates."""
    with pytest.raises(ProviderUnavailableError):
        retry_call(
            lambda: (_ for _ in ()).throw(ProviderUnavailableError("x")),
            RetryPolicy(max_attempts=1),
        )


# --------------------------------------------------------------------------- #
# Logging and credential safety
# --------------------------------------------------------------------------- #
def test_stage_labels_cover_every_pipeline_stage() -> None:
    """Each stage in the pipeline diagram has a label."""
    assert set(STAGE_LABELS) == {
        "ingest",
        "validate",
        "preprocess",
        "anomaly",
        "track",
        "gnn",
        "downscale",
        "alert",
    }


def test_stage_line_is_tagged_and_run_scoped() -> None:
    """A log line must be attributable to one run and one stage."""
    import logging

    line = PipelineLogger(logging.getLogger("t"), run_id="run-abc").stage("ingest", "ok")

    assert line.startswith("[INGEST")
    assert "ok" in line


def test_unknown_stage_is_still_logged() -> None:
    """A new stage can be logged before its label is registered."""
    import logging

    line = PipelineLogger(logging.getLogger("t")).stage("brand_new", "hello")

    assert "BRAND_NEW" in line


def test_log_stage_helper_formats_a_line() -> None:
    """The convenience helper produces the same shape as the method."""
    import logging

    line = log_stage("alert", "done", logger=logging.getLogger("t"), run_id="r1")

    assert line.startswith("[ALERT")


@pytest.mark.parametrize(
    "text",
    [
        "api_key=SUPERSECRET123",
        "API-KEY: SUPERSECRET123",
        "token=SUPERSECRET123",
        "password=hunter2secret",
        "Authorization: Bearer abcdef123456",
    ],
)
def test_credentials_are_masked_in_log_output(text: str) -> None:
    """A secret in a message must not survive into the log."""
    masked = redact(text)

    assert "SUPERSECRET123" not in masked
    assert "hunter2secret" not in masked
    assert "abcdef123456" not in masked
    assert "***" in masked


def test_key_name_is_kept_when_masking() -> None:
    """Masking the value but keeping the name keeps the log informative."""
    masked = redact("api_key=SUPERSECRET123")

    assert "api_key" in masked


def test_ordinary_text_is_untouched() -> None:
    """Redaction must not mangle normal operational messages."""
    message = "[VALIDATE] status=VALID checks=9 blocking_failures=0"

    assert redact(message) == message


def test_log_stage_redacts_its_message() -> None:
    """The redaction is applied on the logging path, not just in a helper."""
    import logging

    line = log_stage("ingest", "token=LEAKEDVALUE", logger=logging.getLogger("t"))

    assert "LEAKEDVALUE" not in line


def test_url_credentials_are_redacted() -> None:
    """A URL is the field most likely to be pasted into a bug report."""
    redacted = redact_url("https://api.example.com/v1/data?apikey=SECRETVALUE&x=1")

    assert "SECRETVALUE" not in str(redacted)
    assert "apikey" in str(redacted)


def test_url_userinfo_is_redacted() -> None:
    """Credentials in the authority section are stripped."""
    redacted = redact_url("https://user:SECRETPASS@api.example.com/v1")

    assert "SECRETPASS" not in str(redacted)
    assert "api.example.com" in str(redacted)


def test_clean_url_is_unchanged() -> None:
    """A URL with nothing sensitive is returned as-is."""
    url = "https://api.example.com/v1/data?latitude=12"

    assert redact_url(url) == url


def test_none_url_stays_none() -> None:
    """A missing URL is not an error."""
    assert redact_url(None) is None


def test_ingestion_record_never_exposes_a_credential() -> None:
    """The manifest is persisted and returned over HTTP."""
    from extreme_weather.realtime.models import IngestionRecord

    record = IngestionRecord(
        run_id="run-1",
        provider="remote-http",
        source_url="https://api.example.com/x?apikey=SECRETVALUE",
    )

    payload = json.dumps(record.as_dict(), default=str)
    assert "SECRETVALUE" not in payload


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def _scheduler(settings: Settings, window: WeatherRequest, config: SchedulerConfig):
    """Return a scheduler with no real sleeps and a fixture provider."""
    orchestrator = PipelineOrchestrator(
        FixtureProvider(),
        settings=settings,
        sleep=lambda _s: None,
        retry_policy=RetryPolicy(max_attempts=1, base_delay=0.0),
    )
    return RealtimeScheduler(orchestrator, lambda: window, config, sleep=lambda _s: None)


def test_one_shot_configuration_runs_once(settings: Settings, window: WeatherRequest) -> None:
    """A zero interval means a single run."""
    scheduler = _scheduler(settings, window, SchedulerConfig(interval_minutes=0))

    ticks = scheduler.run(dry_run=True)

    assert len(ticks) == 1
    assert ticks[0].succeeded is True


def test_interval_runs_stop_at_max_runs(settings: Settings, window: WeatherRequest) -> None:
    """An interval loop is bounded by max_runs, so it terminates in tests."""
    scheduler = _scheduler(
        settings, window, SchedulerConfig(interval_minutes=1, max_runs=3)
    )

    ticks = scheduler.run(dry_run=True)

    assert len(ticks) == 3


def test_stop_flag_ends_the_loop(settings: Settings, window: WeatherRequest) -> None:
    """The loop is interruptible from outside."""
    scheduler = _scheduler(
        settings, window, SchedulerConfig(interval_minutes=1, max_runs=10)
    )
    original = scheduler.run_once

    def _run_and_stop(**kwargs: Any):
        tick = original(**kwargs)
        scheduler.stop()
        return tick

    scheduler.run_once = _run_and_stop  # type: ignore[method-assign]
    ticks = scheduler.run(dry_run=True)

    assert len(ticks) == 1


def test_dry_run_plan_lists_stages(settings: Settings, window: WeatherRequest) -> None:
    """A pre-flight plan must name the stages it would run."""
    scheduler = _scheduler(settings, window, SchedulerConfig())

    plan = scheduler.dry_run_plan()

    assert plan["planned_stages"] == [
        "ingest",
        "validate",
        "preprocess",
        "anomaly",
        "track",
        "gnn",
        "downscale",
        "alert",
    ]


def test_dry_run_plan_writes_nothing(settings: Settings, window: WeatherRequest) -> None:
    """The plan must state that it produced no artifacts."""
    scheduler = _scheduler(settings, window, SchedulerConfig())

    plan = scheduler.dry_run_plan()

    assert plan["would_write"] is False
    assert not (settings.data_dir / "realtime" / "runs").exists()


def test_dry_run_plan_includes_the_window(settings: Settings, window: WeatherRequest) -> None:
    """The plan shows what would be requested."""
    scheduler = _scheduler(settings, window, SchedulerConfig())

    plan = scheduler.dry_run_plan()

    assert plan["request"]["variables"] == list(window.variables)
    assert plan["provider"]["name"] == "fixture"


def test_scheduler_config_rejects_a_negative_interval() -> None:
    """A negative interval is a configuration error."""
    with pytest.raises(ValueError, match="cannot be negative"):
        SchedulerConfig(interval_minutes=-1)


def test_scheduler_tick_reports_success(settings: Settings, window: WeatherRequest) -> None:
    """A tick serialises for the CLI and the API."""
    scheduler = _scheduler(settings, window, SchedulerConfig())

    tick = scheduler.run_once(dry_run=True)
    payload = tick.as_dict()

    assert payload["succeeded"] is True
    assert payload["run_id"].startswith("run-")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cli() -> Any:
    """Import the CLI module by path, since ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("run_realtime_pipeline", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_realtime_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def test_cli_help_lists_every_documented_option(cli: Any) -> None:
    """--help must document the options the specification requires."""
    text = cli.build_parser().format_help()

    for option in (
        "--once",
        "--dry-run",
        "--interval-minutes",
        "--provider",
        "--start",
        "--end",
        "--force",
        "--verbose",
    ):
        assert option in text


def test_cli_help_lists_every_provider(cli: Any) -> None:
    """A configurable remote provider must be advertised."""
    text = cli.build_parser().format_help()

    for provider in ("local", "open-meteo", "remote-http", "fixture"):
        assert provider in text


def test_cli_help_carries_the_prototype_disclaimer(cli: Any) -> None:
    """The caveat is visible before anything is run."""
    assert "not an operational emergency-warning system" in cli.build_parser().format_help()


def test_cli_rejects_an_inverted_window(cli: Any) -> None:
    """A window that ends before it starts is a configuration error."""
    code = cli.main(
        [
            "--dry-run",
            "--start",
            "2024-06-30T12:00:00",
            "--end",
            "2024-06-30T06:00:00",
        ]
    )

    assert code == 2


def test_cli_rejects_an_unknown_provider(cli: Any) -> None:
    """An unknown provider name fails before any work is attempted."""
    assert cli.main(["--dry-run", "--provider", "nonsense"]) == 2


def test_cli_dry_run_succeeds(cli: Any, capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run validates the configuration and prints a plan."""
    code = cli.main(["--dry-run", "--provider", "fixture"])

    assert code == 0
    output = capsys.readouterr().out
    assert "dry run" in output.lower()
    assert "artifacts written: False" in output


def test_cli_dry_run_writes_no_artifacts(
    cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run must not create the runs directory."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    cli.main(["--dry-run", "--provider", "fixture"])

    assert not (tmp_path / "data" / "realtime" / "runs").exists()


def test_cli_rejects_an_unavailable_local_file(cli: Any) -> None:
    """A missing local file is a configuration failure, not a crash."""
    code = cli.main(["--once", "--provider", "local", "--file", "no/such/file.nc"])

    assert code == 2


def test_cli_parses_a_grid(cli: Any) -> None:
    """A comma-separated grid becomes a tuple of floats."""
    values = cli._parse_floats("12.0, 12.5 ,13.0", "lat")

    assert values == (12.0, 12.5, 13.0)


def test_cli_rejects_an_empty_grid(cli: Any) -> None:
    """An empty grid would be an unbounded request."""
    from extreme_weather.realtime.exceptions import PipelineConfigurationError

    with pytest.raises(PipelineConfigurationError, match="at least one value"):
        cli._parse_floats(" , ", "lat")


def test_cli_parses_a_zulu_timestamp(cli: Any) -> None:
    """A trailing Z is a common ISO form."""
    parsed = cli._parse_moment("2024-06-30T12:00:00Z")

    assert parsed is not None
    assert parsed.tzinfo is not None


def test_cli_returns_none_for_an_unparseable_timestamp(cli: Any) -> None:
    """Nonsense yields None so the caller can raise a clear error."""
    assert cli._parse_moment("not a date") is None


def test_cli_window_defaults_to_a_recent_range(cli: Any) -> None:
    """With no --start/--end the window ends now."""
    start, end = cli._parse_window(
        cli.build_parser().parse_args(["--hours", "3"])
    )

    assert end - start == timedelta(hours=3)
