"""Near-real-time ingestion and pipeline orchestration (SIH26078, Phase 8).

Phase 8 wraps the existing pipeline in an orchestrator that can fetch new data
periodically, gate it on quality and freshness, and drive Phases 0-5 without
recomputing history it has already seen.

```text
WeatherProvider            local | open-meteo | remote-http | fixture
      |
      v
IngestionService (Phase 6)  validate, normalise, manifest, register
      |
      v
QualityGate                VALID | VALID_WITH_WARNINGS | REJECTED
      |
      v
FreshnessTracker           FRESH | STALE | EXPIRED | UNKNOWN
      |
      v
PipelineOrchestrator
      PREPROCESS -> ANOMALY -> TRACK -> GNN? -> DOWNSCALE? -> ALERT
```

Three rules the design is built around:

1. **Stale data is never silently treated as current.** Freshness is a first
   class field on every run, with configurable thresholds and its own state.
2. **An optional model that is absent is a ``SKIPPED`` stage, not a failure and
   not a fabricated prediction.** The pipeline still completes, and says why.
3. **One ingestion, one run.** A deterministic identity over the provider, the
   time window and the data checksum makes a repeat ingestion idempotent, so
   scheduling a job twice does not double the event catalogue.

This is a **research prototype**. It is not an operational emergency-warning
system, it has no SLA, and nothing here demonstrates forecast skill.
"""

from __future__ import annotations

from extreme_weather.realtime.exceptions import (
    PipelineConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QualityGateRejection,
)
from extreme_weather.realtime.freshness import (
    DEFAULT_FRESHNESS_THRESHOLDS,
    FRESH,
    STALE,
    UNKNOWN,
    EXPIRED,
    FreshnessAssessment,
    FreshnessTracker,
    assess_freshness,
)
from extreme_weather.realtime.idempotency import (
    RunIdentity,
    compute_run_identity,
)
from extreme_weather.realtime.logging import (
    STAGE_LABELS,
    get_logger,
    log_stage,
)
from extreme_weather.realtime.models import (
    FAILED,
    PENDING,
    RUNNING,
    SKIPPED,
    SUCCESS,
    WARNING,
    IngestionRecord,
    PipelineRun,
    StageResult,
    StageStatus,
)
from extreme_weather.realtime.orchestrator import (
    PipelineOrchestrator,
    OrchestratorResult,
)
from extreme_weather.realtime.providers import (
    FixtureProvider,
    LocalFileProvider,
    OpenMeteoProvider,
    RemoteHttpProvider,
    WeatherProvider,
    WeatherRequest,
    build_provider,
    default_providers,
)
from extreme_weather.realtime.quality_gate import (
    QualityGate,
    QualityVerdict,
)
from extreme_weather.realtime.retry import RetryPolicy, retry_call
from extreme_weather.realtime.scheduler import (
    RealtimeScheduler,
    SchedulerConfig,
)

__all__ = [
    "DEFAULT_FRESHNESS_THRESHOLDS",
    "EXPIRED",
    "FAILED",
    "FRESH",
    "FixtureProvider",
    "FreshnessAssessment",
    "FreshnessTracker",
    "IngestionRecord",
    "LocalFileProvider",
    "OpenMeteoProvider",
    "PENDING",
    "PipelineConfigurationError",
    "PipelineOrchestrator",
    "PipelineRun",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QualityGate",
    "QualityGateRejection",
    "QualityVerdict",
    "RUNNING",
    "RemoteHttpProvider",
    "RealtimeScheduler",
    "RetryPolicy",
    "RunIdentity",
    "SKIPPED",
    "STAGE_LABELS",
    "STALE",
    "STAGE_LABELS",
    "SUCCESS",
    "SchedulerConfig",
    "StageResult",
    "StageStatus",
    "UNKNOWN",
    "WARNING",
    "WeatherProvider",
    "OrchestratorResult",
    "assess_freshness",
    "WeatherProvider",
    "WeatherRequest",
    "assess_freshness",
    "build_provider",
    "compute_run_identity",
    "default_providers",
    "get_logger",
    "log_stage",
    "retry_call",
]
__all__ = [
    # exceptions
    "PipelineConfigurationError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QualityGateRejection",
    # freshness
    "DEFAULT_FRESHNESS_THRESHOLDS",
    "EXPIRED",
    "FRESH",
    "FreshnessAssessment",
    "FreshnessTracker",
    "STALE",
    "UNKNOWN",
    "assess_freshness",
    # identity and logging
    "RunIdentity",
    "STAGE_LABELS",
    "compute_run_identity",
    "get_logger",
    "log_stage",
    # models
    "FAILED",
    "IngestionRecord",
    "PENDING",
    "RUNNING",
    "SKIPPED",
    "SUCCESS",
    "WARNING",
    "PipelineRun",
    "StageResult",
    "StageStatus",
    # orchestration
    "OrchestratorResult",
    "PipelineOrchestrator",
    "RealtimeScheduler",
    "SchedulerConfig",
    # providers
    "FixtureProvider",
    "LocalFileProvider",
    "OpenMeteoProvider",
    "RemoteHttpProvider",
    "WeatherProvider",
    "WeatherRequest",
    "build_provider",
    "default_providers",
    # quality, retry
    "QualityGate",
    "QualityVerdict",
    "RetryPolicy",
    "retry_call",
]

