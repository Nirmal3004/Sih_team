"""Real-data model evaluation, explainability and uncertainty (SIH26078, Phase 9).

Phase 9 answers five questions about the models built in Phases 3 and 4:

1. How does the GNN compare against persistence and constant velocity?
2. How does the downscaler compare against nearest and cubic interpolation?
3. What evidence contributed to a detected event, track or alert?
4. How uncertain is a prediction, and is that uncertainty calibrated?
5. Can all of the above be reproduced from a documented split?

```text
      WeatherProvider / ERA5
              |
        Phase 6 ingestion
              |
        Phase 7 chronological split
              |
        Phase 1 anomaly -> Phase 2 tracking
              |
        Phase 3 GNN -> Phase 4 downscaling
              |
         +----+----+-----------+
         |    |    |           |
      Baseline  Expl.   Uncertainty
         +----+----+-----------+
              |
       EvaluationRun
              |
        Dashboard / API
```

The package is built around one commitment: **a measurement is either reported
with the evidence that produced it, or reported as unavailable with the reason
why.** There is no code path that turns a missing number into ``0.0``, and no
path that turns a correlation into a cause.

- :mod:`~extreme_weather.evaluation.dataset` - the leak-checked evaluation set
- :mod:`~extreme_weather.evaluation.baselines` - persistence and constant velocity
- :mod:`~extreme_weather.evaluation.evaluator` - measurement and orchestration
- :mod:`~extreme_weather.evaluation.uncertainty` - residual bootstrap and coverage
- :mod:`~extreme_weather.evaluation.explainability` - evidence, never causation
- :mod:`~extreme_weather.evaluation.models` - `MetricResult`, `EvaluationRun`

This is a **research prototype**. Nothing here establishes forecast skill,
disaster reduction, or any accuracy claim: no verified ground-truth warning
catalogue exists for this project.
"""

from __future__ import annotations

from extreme_weather.evaluation.baselines import (
    BASELINE_DEFINITIONS,
    CONSTANT_VELOCITY,
    PERSISTENCE,
    constant_velocity_baseline,
    describe_baselines,
    evaluate_baseline,
)
from extreme_weather.evaluation.dataset import (
    EvaluationDataset,
    EvaluationSplit,
    LeakageError,
    assert_chronological,
    build_evaluation_dataset,
    fit_train_only_scaler,
)
from extreme_weather.evaluation.evaluator import (
    EVALUATION_CODE_FILES,
    GNN_METRIC_UNITS,
    EvaluationOutcome,
    code_fingerprint,
    evaluate_gnn_baselines,
    evaluate_multistep_rollout,
    run_evaluation,
    summarise_gnn_comparison,
)
from extreme_weather.evaluation.explainability import (
    CAUSALITY_DISCLAIMER,
    EVIDENCE_UNAVAILABLE,
    SUPPORTING,
    AlertExplanation,
    EventExplanation,
    Evidence,
    build_alert_explanation,
    build_event_explanation,
    sensitivity_report,
)
from extreme_weather.evaluation.models import (
    DERIVED,
    FAILED,
    INSUFFICIENT_DATA,
    MEASURED,
    NOT_AVAILABLE,
    NOT_EVALUATED,
    PARTIAL,
    SUCCESS,
    EvaluationRun,
    MetricResult,
    Provenance,
)
from extreme_weather.evaluation.uncertainty import (
    BOOTSTRAP_METHOD,
    UNCERTAINTY_CAVEAT,
    CalibrationResult,
    PredictionUncertainty,
    ResidualBootstrap,
    evaluate_coverage,
    residual_intervals,
)

__all__ = [
    "BASELINE_DEFINITIONS",
    "BOOTSTRAP_METHOD",
    "CAUSALITY_DISCLAIMER",
    "CONSTANT_VELOCITY",
    "DERIVED",
    "EVALUATION_CODE_FILES",
    "EVIDENCE_UNAVAILABLE",
    "FAILED",
    "GNN_METRIC_UNITS",
    "INSUFFICIENT_DATA",
    "MEASURED",
    "NOT_AVAILABLE",
    "NOT_EVALUATED",
    "PARTIAL",
    "PERSISTENCE",
    "SUPPORTING",
    "SUCCESS",
    "UNCERTAINTY_CAVEAT",
    "AlertExplanation",
    "CalibrationResult",
    "EvaluationDataset",
    "EvaluationOutcome",
    "EvaluationRun",
    "EvaluationSplit",
    "EventExplanation",
    "Evidence",
    "LeakageError",
    "MetricResult",
    "PredictionUncertainty",
    "Provenance",
    "ResidualBootstrap",
    "assert_chronological",
    "build_alert_explanation",
    "build_evaluation_dataset",
    "build_event_explanation",
    "code_fingerprint",
    "constant_velocity_baseline",
    "describe_baselines",
    "evaluate_baseline",
    "evaluate_coverage",
    "evaluate_gnn_baselines",
    "evaluate_multistep_rollout",
    "fit_train_only_scaler",
    "residual_intervals",
    "run_evaluation",
    "sensitivity_report",
    "summarise_gnn_comparison",
]
