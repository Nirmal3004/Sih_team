"""Stage timing instrumentation (SIH26078, Phase 10, section 9).

The Phase 8 orchestrator already records a duration on every
:class:`~extreme_weather.realtime.models.StageResult`, but that lives in the
run record and is only visible *after* a run. This module does the other half:
it records the same durations into the process-wide metrics registry and emits a
log line per stage, so a slow stage is visible in a live log and a trend is
visible across runs.

What is deliberately *not* done here
------------------------------------
No scientific or model behaviour is altered for speed. A stage that is slow
because it does real work stays slow and is reported as slow. Tuning a
threshold, skipping a computation or changing a resolution to make a benchmark
look better would change results, and results are the product.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from extreme_weather.observability.logging import log_event
from extreme_weather.observability.metrics import get_registry

__all__ = ["StageTimer", "stage_timer", "timed_stage"]


@dataclass
class StageTimer:
    """Accumulates durations for a run.

    Attributes:
        run_id: The run these durations belong to.
        stages: Completed ``(name, seconds)`` pairs, in completion order.
    """

    run_id: str = ""
    stages: list[tuple[str, float]] = field(default_factory=list)

    def record(self, name: str, seconds: float) -> None:
        """Record one stage duration, in both the log and the metrics."""
        self.stages.append((name, float(seconds)))
        get_registry().observe(
            "pipeline_stage_duration_seconds", seconds, stage=name
        )
        log_event(
            logging.getLogger("extreme_weather.pipeline"),
            logging.INFO,
            "pipeline.stage.completed",
            f"stage {name} took {seconds:.3f}s",
            stage=name,
            duration_seconds=round(float(seconds), 6),
            run_id=self.run_id or None,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the recorded stages."""
        total = sum(seconds for _name, seconds in self.stages)
        return {
            "run_id": self.run_id,
            "stages": {name: seconds for name, seconds in self.stages},
            "total_seconds": total,
            "stage_count": len(self.stages),
        }


@contextmanager
def stage_timer(timer: StageTimer, name: str) -> Iterator[None]:
    """Time one stage and record it, even when the stage raises.

    A failed stage's duration is still worth knowing, so the record happens in a
    ``finally``.

    Args:
        timer: The run's timer.
        name: The stage name, e.g. ``ingest``.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        timer.record(name, time.perf_counter() - started)


@contextmanager
def timed_stage(
    name: str, *, run_id: str = "", logger: logging.Logger | None = None
) -> Iterator[StageTimer]:
    """Time a single block as one named stage.

    Args:
        name: The stage name.
        run_id: Optional run id for correlation.
        logger: Optional logger; the pipeline logger is used when omitted.

    Yields:
        A fresh :class:`StageTimer` carrying the one recorded stage.
    """
    timer = StageTimer(run_id=run_id)
    with stage_timer(timer, name):
        yield timer
