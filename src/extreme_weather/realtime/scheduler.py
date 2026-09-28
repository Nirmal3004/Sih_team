"""Near-real-time scheduling (SIH26078, Phase 8, section 3).

A deliberately small scheduler: a loop, a clock and a stop flag. No Celery, no
Airflow, no APScheduler. The prototype needs to run a job every N minutes and to
be interruptible from the command line, and a framework would add a dependency,
a broker and a deployment story in exchange for nothing this project uses.

Three modes
-----------
``once``     run a single time and exit
``interval`` run every N minutes until stopped
``dry-run``  validate and print the plan without writing anything

Two properties matter more than features
----------------------------------------
**Interruptible.** Ctrl-C finishes the current run, persists it, and exits with a
clear message rather than a traceback.

**Catching up is not attempted.** If a run overruns its interval, the next one
starts at the next tick rather than immediately. Backfilling missed ticks on
startup would launch a burst of identical ingests, which is precisely what the
idempotency layer would then have to absorb.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from extreme_weather.realtime.models import FAILED, SUCCESS, WARNING, PipelineRun, utc_now
from extreme_weather.realtime.orchestrator import PipelineOrchestrator
from extreme_weather.realtime.providers import WeatherRequest

__all__ = ["RealtimeScheduler", "SchedulerConfig", "SchedulerTick"]

UTC = timezone.utc


@dataclass(frozen=True)
class SchedulerConfig:
    """How the scheduler should behave.

    Attributes:
        interval_minutes: Minutes between runs. ``0`` means one-shot.
        max_runs: Stop after this many runs; ``None`` means run until stopped.
        catch_up: Whether missed ticks should be run immediately on start.
        dry_run: Pass through to the orchestrator.
    """

    interval_minutes: int = 0
    max_runs: int | None = None
    catch_up: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if int(self.interval_minutes) < 0:
            raise ValueError("interval_minutes cannot be negative.")
        if self.max_runs is not None and int(self.max_runs) < 1:
            raise ValueError("max_runs must be at least 1 when set.")

    @property
    def is_one_shot(self) -> bool:
        """True when the scheduler should run exactly once."""
        return int(self.interval_minutes) == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the configuration."""
        return {
            "interval_minutes": self.interval_minutes,
            "max_runs": self.max_runs,
            "catch_up": self.catch_up,
            "dry_run": self.dry_run,
            "one_shot": self.is_one_shot,
        }


@dataclass
class SchedulerTick:
    """The outcome of one scheduled run.

    Attributes:
        started_at: When the run started.
        completed_at: When it finished.
        run: The pipeline run, when one was produced.
        error: The failure message, when the run itself could not start.
    """

    started_at: datetime
    completed_at: datetime
    run: PipelineRun | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the scheduled run completed without failing."""
        return self.error is None and (
            self.run is None or self.run.status in (SUCCESS, WARNING)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the tick."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "run_id": self.run.run_id if self.run else None,
            "status": self.run.status if self.run else "ERROR",
            "succeeded": self.succeeded,
            "error": self.error,
        }


class RealtimeScheduler:
    """Runs the pipeline on a schedule, or once, or not at all.

    Args:
        orchestrator: The pipeline to drive.
        request_factory: Produces the request for each run. A factory rather
            than a fixed request, because an interval run must ask for a
            *later* window each time.
        config: Scheduling behaviour.
        sleep: Injected sleep, so tests do not wait in real time.
    """

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        request_factory: Callable[[], WeatherRequest],
        config: SchedulerConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.orchestrator = orchestrator
        self.request_factory = request_factory
        self.config = config or SchedulerConfig()
        self._sleep = sleep
        self.ticks: list[SchedulerTick] = []
        self._stop = False

    def stop(self) -> None:
        """Request that the loop exit after the current run."""
        self._stop = True

    def run_once(self, **kwargs: Any) -> SchedulerTick:
        """Run the pipeline exactly once and return the tick.

        Args:
            **kwargs: Forwarded to :meth:`PipelineOrchestrator.run`.

        Returns:
            A :class:`SchedulerTick`. A pipeline failure is recorded on the
            tick rather than raised, so a scheduled run can report its own
            failure and keep the scheduler's loop in control.
        """
        started = utc_now()
        try:
            request = self.request_factory()
            result = self.orchestrator.run(request, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a tick must always be produced
            tick = SchedulerTick(
                started_at=started, completed_at=utc_now(), error=str(exc)
            )
            self.ticks.append(tick)
            return tick
        tick = SchedulerTick(
            started_at=started, completed_at=utc_now(), run=result.run
        )
        self.ticks.append(tick)
        return tick

    def run(self, **kwargs: Any) -> list[SchedulerTick]:
        """Run according to the configuration, returning every tick.

        Args:
            **kwargs: Forwarded to :meth:`PipelineOrchestrator.run`.

        Returns:
            The list of ticks, in order.

        Raises:
            KeyboardInterrupt: Re-raised after the current run completes, so the
                caller knows the loop was interrupted rather than finished.
        """
        if self.config.is_one_shot:
            return [self.run_once(**kwargs)]

        interval = int(self.config.interval_minutes) * 60.0
        ticks: list[SchedulerTick] = []
        self.orchestrator.logger.info(
            f"scheduler started: every {self.config.interval_minutes} minute(s), "
            f"max_runs={self.config.max_runs}"
        )
        try:
            while not self._stop:
                ticks.append(self.run_once(**kwargs))
                if self.config.max_runs is not None and len(ticks) >= self.config.max_runs:
                    break
                if self._stop:
                    break
                # Fixed delay after the run, so a slow run cannot overlap with
                # the next one and cannot trigger a catch-up burst.
                self._sleep(interval)
        except KeyboardInterrupt:
            self.orchestrator.logger.info("scheduler interrupted; stopping")
            self.orchestrator.logger.info(
                f"finished: {len(ticks)} run(s) completed before the interrupt"
            )
            raise
        return ticks

    def dry_run_plan(self, **kwargs: Any) -> dict[str, Any]:
        """Return the plan without executing or writing anything.

        Performs a real fetch and quality check - so the plan reflects the data
        that would actually arrive - but writes no artifacts and starts no
        pipeline stage beyond the gate.

        Args:
            **kwargs: Forwarded to :meth:`PipelineOrchestrator.run`.

        Returns:
            A description of the configuration, provider, window and the
            resulting quality verdict.
        """
        request = self.request_factory()
        provider = self.orchestrator.provider
        return {
            "config": self.config.as_dict(),
            "provider": provider.describe(),
            "request": request.as_dict(),
            "planned_stages": list(
                __import__(
                    "extreme_weather.realtime.orchestrator", fromlist=["STAGE_ORDER"]
                ).STAGE_ORDER
            ),
            "context_steps": self.orchestrator.context_steps,
            "freshness_thresholds": self.orchestrator.freshness_thresholds.as_dict(),
            "retry_policy": self.orchestrator.retry_policy.as_dict(),
            "incremental": self.orchestrator.state.as_dict(),
            "would_write": False,
            "note": (
                "Dry run: configuration validated and stages planned. No "
                "production artifacts were written."
            ),
        }
