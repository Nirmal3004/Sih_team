"""Extreme weather anomaly detection (SIH26078, Phase 1).

```text
historical/reference dataset
        |
        v
climatology.build_climatology()      # reference distribution per group/cell
        |
        v
scores.compute_score()               # absolute | standardized | percentile |
        |                             # extremeness | EFI-style
        v
detector.detect_extreme_cells()      # score + thresholds -> boolean mask
        |
        v
clustering.cluster_extreme_events()  # connected cells -> ExtremeWeatherEvent
        |
        v
service.AnomalyDetectionService      # orchestration + file-backed store
```

Everything is a **prototype** built on the synthetic Phase 0 dataset: the
reference distribution is synthetic, the thresholds are prototype values and the
EFI-style score is not an operational meteorological product. See
``docs/ANOMALY_DETECTION.md``.

Quick start::

    from extreme_weather.anomaly import AnomalyDetectionService

    result = AnomalyDetectionService().run("data/raw/sample_history.nc")
    print(result.event_count, "events")
"""

from __future__ import annotations

from extreme_weather.anomaly.climatology import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_QUANTILE_LEVELS,
    GROUP_COORD_NAME,
    QUANTILE_COORD_NAME,
    ClimatologyError,
    ClimatologyGrouping,
    build_climatology,
    resolve_group_keys,
)
from extreme_weather.anomaly.clustering import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_MIN_CELLS,
    ClusteringError,
    build_event_id,
    cluster_extreme_events,
    haversine_km,
    intensity_weighted_centroid,
)
from extreme_weather.anomaly.detector import (
    DetectionError,
    detect_extreme_cells,
    detection_summary,
    mask_from_ladder,
)
from extreme_weather.anomaly.models import AnomalyAnalysisResult
from extreme_weather.anomaly.scores import (
    EFI_FORMULA,
    EFI_MAX,
    EFI_MIN,
    SCORE_FUNCTIONS,
    ScoreError,
    absolute_anomaly,
    compute_score,
    efi_score,
    extremeness_score,
    percentile_score,
    standardized_anomaly,
)
from extreme_weather.anomaly.service import (
    DEFAULT_ANALYSIS_VARIABLES,
    AnalysisError,
    AnomalyDetectionService,
    EventStore,
    analyse_dataset,
)
from extreme_weather.anomaly.thresholds import (
    DEFAULT_THRESHOLD_LADDERS,
    EFI_THRESHOLD_LADDERS,
    PROTOTYPE_THRESHOLD_WARNING,
    STANDARDIZED_THRESHOLD_LADDERS,
    ThresholdsError,
    ThresholdLadder,
    ThresholdSet,
    classify_score,
    classify_scores,
    default_thresholds,
)
from extreme_weather.schemas.event import (
    EventListResponse,
    EventType,
    ExtremeWeatherEvent,
    Severity,
)

__all__ = [
    "DEFAULT_ANALYSIS_VARIABLES",
    "DEFAULT_CONNECTIVITY",
    "DEFAULT_MIN_CELLS",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_QUANTILE_LEVELS",
    "DEFAULT_THRESHOLD_LADDERS",
    "EFI_FORMULA",
    "EFI_MAX",
    "EFI_MIN",
    "EFI_THRESHOLD_LADDERS",
    "GROUP_COORD_NAME",
    "PROTOTYPE_THRESHOLD_WARNING",
    "QUANTILE_COORD_NAME",
    "SCORE_FUNCTIONS",
    "STANDARDIZED_THRESHOLD_LADDERS",
    "AnalysisError",
    "AnomalyAnalysisResult",
    "AnomalyDetectionService",
    "ClimatologyError",
    "ClimatologyGrouping",
    "ClusteringError",
    "DetectionError",
    "EventListResponse",
    "EventStore",
    "EventType",
    "ExtremeWeatherEvent",
    "ScoreError",
    "Severity",
    "ThresholdLadder",
    "ThresholdSet",
    "ThresholdsError",
    "absolute_anomaly",
    "analyse_dataset",
    "build_climatology",
    "build_event_id",
    "classify_score",
    "classify_scores",
    "cluster_extreme_events",
    "compute_score",
    "default_thresholds",
    "detect_extreme_cells",
    "detection_summary",
    "efi_score",
    "extremeness_score",
    "haversine_km",
    "intensity_weighted_centroid",
    "mask_from_ladder",
    "percentile_score",
    "resolve_group_keys",
    "standardized_anomaly",
]
