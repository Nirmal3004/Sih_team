"""Observability primitives (SIH26078, Phase 10).

Three small pieces, deliberately without a metrics or logging vendor
dependency:

- :mod:`~extreme_weather.observability.logging` - structured, redacting logs
  with request/run correlation
- :mod:`~extreme_weather.observability.metrics` - system counters and duration
  histograms, which are **not** model-quality metrics
- :mod:`~extreme_weather.observability.timing` - per-stage timing

The boundary between "the system worked" and "the model was right" is kept
explicit: nothing in this package can produce an accuracy, precision, recall or
forecast-skill figure, because it never sees a prediction or a truth.
"""

from __future__ import annotations

from extreme_weather.observability.logging import (
    COMPONENT_ROOT,
    JsonFormatter,
    TextFormatter,
    bind_context,
    configure_logging,
    get_correlation_ids,
    get_logger,
    log_event,
    new_request_id,
    redact,
    request_context,
    run_context,
)
from extreme_weather.observability.metrics import (
    KNOWN_COUNTERS,
    KNOWN_HISTOGRAMS,
    MetricsRegistry,
    get_registry,
    observe_duration,
    reset_registry,
)
from extreme_weather.observability.timing import StageTimer, stage_timer, timed_stage

__all__ = [
    "COMPONENT_ROOT",
    "KNOWN_COUNTERS",
    "KNOWN_HISTOGRAMS",
    "JsonFormatter",
    "MetricsRegistry",
    "StageTimer",
    "TextFormatter",
    "bind_context",
    "configure_logging",
    "get_correlation_ids",
    "get_logger",
    "get_registry",
    "log_event",
    "new_request_id",
    "observe_duration",
    "redact",
    "request_context",
    "reset_registry",
    "run_context",
    "stage_timer",
    "timed_stage",
]
