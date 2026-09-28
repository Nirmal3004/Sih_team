"""Real-data validation, calibration and evaluation (SIH26078, Phase 7).

Phase 7 turns a working prototype into a **measured** one. Its purpose is
explicitly *not* to show that the system is accurate. It is to measure what the
system does, on real historical data, under a method that cannot leak the test
period into training - and to report the negative results as prominently as the
positive ones.

```text
real historical dataset (ERA5 reanalysis)
        |
        v
   temporal split          train | validation | test, strictly chronological
        |
        +-- climatology ......... built on TRAIN ONLY, never on test
        +-- threshold calibration  percentiles from TRAIN ONLY
        +-- anomaly detection .... evaluated on VALIDATION and TEST
        +-- tracking ............. descriptive diagnostics
        +-- GNN .................. vs a persistence baseline
        +-- downscaling .......... nearest vs cubic vs AI, same test frames
        +-- alerts ............... descriptive diagnostics
        |
        v
   unified report + REST API + dashboard panel
```

The governing rules, which the tests exist to enforce:

1. **No leakage.** ``max(train_time) < min(validation_time) < min(test_time)``,
   and no statistic used for detection is computed from the test period.
2. **No fabricated metrics.** Without a reference dataset, skill metrics
   (POD/FAR/CSI) are *not* computed; the report says
   ``"Ground truth unavailable - metric not computed."`` rather than emitting a
   zero, which would read as "zero skill".
3. **Baselines must be real.** Persistence for the GNN, nearest and cubic for
   downscaling, prototype thresholds for calibration. If the AI loses, the
   report says so.
4. **Honest coverage.** A climatology bucket with too few samples is reported as
   insufficient, not quietly filled from another bucket.

Quick start::

    python scripts/run_phase7_validation.py --end-date 2024-06-30
"""

from __future__ import annotations

from extreme_weather.validation.climatology_validation import (
    ClimatologyCoverage,
    Diagnostics,
    assess_climatology_coverage,
    build_climatology_diagnostics,
)
from extreme_weather.validation.dataset_builder import (
    ValidationDatasetConfig,
    ValidationDatasetBuilder,
)
from extreme_weather.validation.ground_truth import (
    GroundTruth,
    GroundTruthStatus,
    load_ground_truth,
    match_events,
)
from extreme_weather.validation.metrics import (
    NO_GROUND_TRUTH,
    AlertDiagnostics,
    AnomalyDiagnostics,
    DetectionSkill,
    DownscalingComparison,
    GnnComparison,
    TrackingDiagnostics,
    compute_alert_diagnostics,
    compute_anomaly_diagnostics,
    compute_downscaling_comparison,
    compute_gnn_comparison,
    compute_tracking_diagnostics,
    evaluate_detection_skill,
)
from extreme_weather.validation.provenance import (
    ProvenanceRecord,
    ValidationRunRecord,
    read_run_record,
    write_run_record,
)
from extreme_weather.validation.reports import (
    REPORT_FILENAMES,
    build_phase7_summary,
    read_report,
    render_markdown,
    write_reports,
)
from extreme_weather.validation.splits import (
    SplitSummary,
    TemporalSplit,
    split_timestamps,
    split_dataset,
)
from extreme_weather.validation.threshold_calibration import (
    ThresholdCalibration,
    calibrate_thresholds,
)

__all__ = [
    "NO_GROUND_TRUTH",
    "REPORT_FILENAMES",
    "AlertDiagnostics",
    "AnomalyDiagnostics",
    "ClimatologyCoverage",
    "DetectionSkill",
    "Diagnostics",
    "DownscalingComparison",
    "GnnComparison",
    "GroundTruth",
    "GroundTruthStatus",
    "ProvenanceRecord",
    "SplitSummary",
    "TemporalSplit",
    "ThresholdCalibration",
    "TrackingDiagnostics",
    "ValidationDatasetBuilder",
    "ValidationDatasetConfig",
    "ValidationRunRecord",
    "assess_climatology_coverage",
    "build_climatology_diagnostics",
    "calibrate_thresholds",
    "compute_alert_diagnostics",
    "compute_anomaly_diagnostics",
    "compute_downscaling_comparison",
    "compute_gnn_comparison",
    "compute_tracking_diagnostics",
    "evaluate_detection_skill",
    "load_ground_truth",
    "match_events",
    "read_report",
    "read_run_record",
    "render_markdown",
    "split_dataset",
    "split_timestamps",
    "write_reports",
    "write_run_record",
]
