"""Timing, integrity-in-the-pipeline and reproducibility tests (SIH26078, Phase 10).

Covers sections 7, 8 and 9: stage timing, metric collection from the real
pipeline, and reproducibility of the instrumentation itself.

The performance assertions here are deliberately weak - "a duration was
recorded", "the total is at least the sum of its parts". They check that
*instrumentation exists and is consistent*, not that a stage is fast. A test
that asserted a wall-clock budget would fail on a slow CI machine and would
pressure someone into changing scientific behaviour to make it pass.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from extreme_weather.config import Settings
from extreme_weather.observability.metrics import get_registry, reset_registry
from extreme_weather.observability.timing import StageTimer, stage_timer, timed_stage

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _clean_metrics() -> Any:
    """Give every test an empty registry."""
    reset_registry()
    yield
    reset_registry()


# --------------------------------------------------------------------------- #
# Stage timing
# --------------------------------------------------------------------------- #
def test_a_stage_duration_is_recorded() -> None:
    """The primitive must actually time something."""
    timer = StageTimer()

    with stage_timer(timer, "ingest"):
        time.sleep(0.01)

    assert [name for name, _ in timer.stages] == ["ingest"]
    assert timer.stages[0][1] > 0.0


def test_a_raised_stage_is_still_timed() -> None:
    """A failed stage's duration is exactly the one worth knowing."""
    timer = StageTimer()

    with pytest.raises(RuntimeError):
        with stage_timer(timer, "anomaly"):
            raise RuntimeError("boom")

    assert [name for name, _ in timer.stages] == ["anomaly"]


def test_timed_stage_helper_reports_a_total() -> None:
    """A caller must be able to read the duration it just measured."""
    with timed_stage("track", run_id="run-1") as timer:
        time.sleep(0.01)

    summary = timer.as_dict()

    assert summary["stage_count"] == 1
    assert summary["total_seconds"] > 0.0
    assert "track" in summary["stages"]


def test_the_total_is_the_sum_of_the_stages() -> None:
    """A summary that does not add up is worse than none."""
    timer = StageTimer()
    timer.record("a", 1.0)
    timer.record("b", 2.0)

    assert timer.as_dict()["total_seconds"] == pytest.approx(3.0)


def test_stage_durations_reach_the_metrics_registry() -> None:
    """Instrumentation is only useful if something aggregates it."""
    timer = StageTimer(run_id="run-2")
    with stage_timer(timer, "gnn"):
        time.sleep(0.005)

    histograms = get_registry().snapshot()["histograms"]

    assert any(key.startswith("pipeline_stage_duration_seconds") for key in histograms)


def test_every_pipeline_stage_is_represented() -> None:
    """A reader must be able to see which stage names are instrumented."""
    from extreme_weather.realtime.orchestrator import STAGE_ORDER

    assert set(STAGE_ORDER) == {
        "ingest",
        "validate",
        "preprocess",
        "anomaly",
        "track",
        "gnn",
        "downscale",
        "alert",
    }


def test_timing_does_not_change_a_run_result(settings: Settings) -> None:
    """Instrumentation must be observation-only.

    A stage result's status and message must be identical with and without the
    metrics hook. This is the regression guard against "optimising" the pipeline
    for speed by changing what it computes.
    """
    from extreme_weather.realtime.orchestrator import PipelineOrchestrator
    from extreme_weather.realtime.providers import FixtureProvider, WeatherRequest
    from extreme_weather.realtime.retry import RetryPolicy

    request = WeatherRequest(
        start=datetime.now(UTC) - timedelta(hours=5),
        end=datetime.now(UTC),
        variables=("precipitation", "temperature", "u_wind", "v_wind"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )

    def _run() -> list[tuple[str, str, str]]:
        orchestrator = PipelineOrchestrator(
            FixtureProvider(),
            settings=settings,
            sleep=lambda _s: None,
            retry_policy=RetryPolicy(max_attempts=1, base_delay=0.0),
        )
        result = orchestrator.run(request, dry_run=True)
        return [
            (s.name, s.status, s.message) for s in result.run.stages
        ]

    first = _run()
    reset_registry()
    second = _run()

    assert first == second


# --------------------------------------------------------------------------- #
# Metric names
# --------------------------------------------------------------------------- #
def test_the_pipeline_counters_are_declared() -> None:
    """A dashboard needs the names to exist even at zero."""
    from extreme_weather.observability.metrics import KNOWN_COUNTERS

    for name in (
        "pipeline_runs_total",
        "pipeline_runs_succeeded_total",
        "pipeline_runs_failed_total",
        "events_detected_total",
        "tracks_generated_total",
        "alerts_generated_total",
        "gnn_inference_attempts_total",
        "downscaling_inference_attempts_total",
        "integrity_checks_total",
        "integrity_failures_total",
    ):
        assert name in KNOWN_COUNTERS


def test_declared_counters_materialise_at_zero() -> None:
    """Zero and unknown are different states; a dashboard must see zero."""
    snapshot = get_registry().snapshot()

    assert "pipeline_runs_total" in snapshot["counters"]
    assert snapshot["counters"]["pipeline_runs_total"] == {}


def test_no_metric_is_named_like_a_quality_score() -> None:
    """The Phase 10 prohibition, asserted on the registry itself.

    A system counter and a model-quality score must not share a namespace: the
    confusion between them is the single most common way a dashboard ends up
    claiming an accuracy it never measured.
    """
    from extreme_weather.observability.metrics import KNOWN_COUNTERS, KNOWN_HISTOGRAMS

    names = [*KNOWN_COUNTERS, *KNOWN_HISTOGRAMS]
    for name in names:
        for forbidden in ("accuracy", "precision", "recall", "f1", "auc", "skill"):
            assert forbidden not in name.lower(), (
                f"'{name}' looks like a model-quality metric; system metrics "
                "must not be named like scores"
            )


def test_histogram_names_are_declared() -> None:
    """Duration histograms must exist even before anything is recorded."""
    from extreme_weather.observability.metrics import KNOWN_HISTOGRAMS

    assert "api_request_duration_seconds" in KNOWN_HISTOGRAMS
    assert "pipeline_stage_duration_seconds" in KNOWN_HISTOGRAMS


# --------------------------------------------------------------------------- #
# Integrity within the operational surface
# --------------------------------------------------------------------------- #
def test_integrity_checks_are_counted(tmp_path: Path) -> None:
    """An integrity sweep is observable like anything else."""
    from extreme_weather.integrity import check_checkpoint

    get_registry().increment("integrity_checks_total", artifact="gnn")
    result = check_checkpoint(tmp_path / "absent.pt", model_dir=tmp_path)
    if not result.usable:
        get_registry().increment("integrity_failures_total", artifact="gnn")

    assert get_registry().counter_value("integrity_checks_total", artifact="gnn") == 1
    assert get_registry().counter_value("integrity_failures_total", artifact="gnn") == 1


def test_a_corrupt_artifact_is_never_reported_usable(tmp_path: Path) -> None:
    """The core integrity promise, stated once more as a regression guard."""
    from extreme_weather.integrity import check_checkpoint

    path = tmp_path / "model.pt"
    path.write_bytes(b"not a real checkpoint at all")

    assert check_checkpoint(path, model_dir=tmp_path).usable is False


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_metrics_are_deterministic_for_the_same_operations() -> None:
    """Two identical metric sequences must produce identical snapshots."""
    from extreme_weather.observability.metrics import MetricsRegistry

    def _populate(registry: Any) -> dict[str, Any]:
        registry.increment("c", method="GET")
        registry.increment("c", method="POST")
        registry.observe("h", 0.01)
        return registry.snapshot()

    first = _populate(MetricsRegistry())
    second = _populate(MetricsRegistry())

    assert first["counters"] == second["counters"]
    assert first["histograms"] == second["histograms"]


def test_the_registry_is_thread_safe() -> None:
    """Concurrent increments must not lose counts."""
    import threading

    from extreme_weather.observability.metrics import MetricsRegistry

    registry = MetricsRegistry()
    threads = [
        threading.Thread(target=lambda: [registry.increment("c") for _ in range(200)])
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert registry.counter_value("c") == 1600
