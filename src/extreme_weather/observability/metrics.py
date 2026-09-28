"""Runtime metrics for Phase 10 (SIH26078).

What these metrics are
----------------------
**System** metrics: how often something ran, how long it took, how often it
failed. That is the whole scope.

What they are not
-----------------
Not model-quality metrics. There is no accuracy, precision, recall, forecast
skill or disaster-reduction figure anywhere in this module, and none can be
derived from what it records - it never sees a prediction or a truth, only a
call.

Why that matters
----------------
A dashboard showing "accuracy 94%" is the single most misleading thing such a
system can do, and it is usually produced by quietly mixing a system counter
with a model-quality claim. Keeping the two in separate modules, with separate
names, makes that mistake harder to write by accident.

Design
------
Counters and histograms live in a process-local registry. There is no database
and no metrics client: this is a single-process research prototype, and a
Prometheus/OTLP dependency would be unjustified for it. The registry is
thread-safe, and reset is explicit so a test cannot leak counters into another.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

__all__ = [
    "MetricsRegistry",
    "get_registry",
    "observe_duration",
    "reset_registry",
]

#: Buckets in seconds, chosen to resolve a few-ms API call from a multi-minute
#: batch job. The first bucket is a catch-all below the first edge.
DURATION_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    300.0,
)


@dataclass
class _Histogram:
    """Cumulative bucket counts plus a running sum and count.

    Attributes:
        buckets: Bucket upper edges in seconds.
        counts: Observation count per bucket, cumulative at read time.
        total: Sum of all observed durations.
        count: Number of observations.
    """

    buckets: tuple[float, ...] = DURATION_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        """Record one observation.

        Args:
            value: Duration in seconds. Non-finite values are ignored rather
                than corrupting the histogram.
        """
        if value != value or value in (float("inf"), float("-inf")):
            return
        self.total += float(value)
        self.count += 1
        # Find the first bucket whose upper edge the value does not exceed.
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1
                return
        self.counts[-1] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view with cumulative counts."""
        cumulative = 0
        buckets: dict[str, int] = {}
        for index, edge in enumerate(self.buckets):
            cumulative += self.counts[index]
            buckets[f"le_{edge}"] = cumulative
        return {
            "count": self.count,
            "sum_seconds": self.total,
            "avg_seconds": (self.total / self.count) if self.count else None,
            "buckets": buckets,
        }


class MetricsRegistry:
    """A small, thread-safe counter/histogram registry.

    Attributes:
        started_at: Monotonic origin for uptime reporting.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, int]] = defaultdict(dict)
        self._histograms: dict[str, _Histogram] = {}
        self._gauges: dict[str, float] = {}
        self.started_at = time.time()

    # --- writes ---------------------------------------------------------
    def increment(self, name: str, value: int = 1, **labels: Any) -> int:
        """Increment a counter and return its new value.

        Args:
            name: Counter name, e.g. ``api_requests_total``.
            value: Increment; use ``0`` to initialise a counter at zero.
            **labels: Dimensions, e.g. ``method="GET"``.

        Returns:
            The counter's value after the increment.
        """
        key = _label_key(labels)
        with self._lock:
            series = self._counters[name]
            series[key] = series.get(key, 0) + int(value)
            return series[key]

    def observe(self, name: str, value: float, **labels: Any) -> None:
        """Record an observation into a histogram.

        Args:
            name: Histogram name, e.g. ``api_request_duration_seconds``.
            value: Observed value.
            **labels: Dimensions.
        """
        key = _label_key(labels)
        with self._lock:
            bucket_key = f"{name}{key}"
            histogram = self._histograms.get(bucket_key)
            if histogram is None:
                histogram = _Histogram()
                self._histograms[bucket_key] = histogram
            histogram.observe(value)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to an absolute value."""
        with self._lock:
            self._gauges[name] = float(value)

    # --- reads ----------------------------------------------------------
    def counter_value(self, name: str, **labels: Any) -> int:
        """Return one counter's value, or ``0`` when never incremented."""
        with self._lock:
            return self._counters.get(name, {}).get(_label_key(labels), 0)

    def snapshot(self) -> dict[str, Any]:
        """Return every metric as a JSON-serialisable mapping.

        Counters are materialised at zero for the names declared in
        :data:`KNOWN_COUNTERS` so a dashboard can show "0 errors" rather than
        "no data", which are different states.
        """
        with self._lock:
            counters = {
                name: dict(series) for name, series in self._counters.items()
            }
            histograms = {
                name: histogram.snapshot()
                for name, histogram in self._histograms.items()
            }
            gauges = dict(self._gauges)

        for name in KNOWN_COUNTERS:
            counters.setdefault(name, {})
        return {
            "counters": counters,
            "histograms": histograms,
            "gauges": gauges,
            "uptime_seconds": max(0.0, time.time() - self.started_at),
        }

    def reset(self) -> None:
        """Clear every recorded metric.

        Explicit and separate from construction so a test can isolate itself
        without reaching into private state.
        """
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self.started_at = time.time()


#: Counters materialised at zero so a dashboard distinguishes "0" from "unknown".
KNOWN_COUNTERS: tuple[str, ...] = (
    "api_requests_total",
    "api_errors_total",
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
)

#: Histograms materialised at zero so a dashboard can always show a distribution.
KNOWN_HISTOGRAMS: tuple[str, ...] = (
    "api_request_duration_seconds",
    "pipeline_stage_duration_seconds",
    "pipeline_run_duration_seconds",
)


def _label_key(labels: Mapping[str, Any]) -> str:
    """Render label dimensions as a stable, sorted key."""
    if not labels:
        return ""
    parts = ",".join(f"{k}={labels[k]}" for k in sorted(labels))
    return "{" + parts + "}"


#: The process-wide registry.
_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry."""
    return _REGISTRY


def reset_registry() -> None:
    """Reset the process-wide registry. Intended for tests."""
    _REGISTRY.reset()


@contextmanager
def observe_duration(
    name: str, **labels: Any
) -> Iterator[dict[str, Any]]:
    """Time a block and record the duration in a histogram.

    Args:
        name: Histogram name.
        **labels: Dimensions, e.g. ``stage="anomaly"``.

    Yields:
        A dict the caller may add to; it is returned unchanged and is used to
        let a caller report the outcome alongside the duration.
    """
    started = time.perf_counter()
    context: dict[str, Any] = {}
    try:
        yield context
    finally:
        _REGISTRY.observe(name, time.perf_counter() - started, **labels)
