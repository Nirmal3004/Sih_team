"""Downscaling evaluation for Phase 9 (SIH26078, section 5).

This module is a thin adapter, and deliberately so. The field metrics, the
identical-frame comparison, the POD/FAR contingency tables and the extreme-event
analysis all already exist in
:mod:`extreme_weather.validation.metrics`. Reimplementing them here would
create two implementations that could drift, and the Phase 7 numbers would
stop matching the Phase 9 ones for no good reason.

What this module adds is **availability policy**:

* every method is scored on the *same* test frames;
* the AI method is included only when a checkpoint actually produced output -
  an absent checkpoint means the method is omitted, never imitated;
* a missing high-resolution reference means POD/FAR/CSI are reported as not
  computed, because a contingency table against an invented threshold is worse
  than no table;
* nothing is tuned on the test split. The comparison is a measurement.

Every limitation the underlying comparison reports is carried through verbatim,
so a Phase 9 report cannot quietly lose a caveat Phase 7 attached.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from extreme_weather.evaluation.models import (
    INSUFFICIENT_DATA,
    NOT_AVAILABLE,
    MetricResult,
)

__all__ = [
    "AI_METHOD",
    "CUBIC_METHOD",
    "NEAREST_METHOD",
    "evaluate_downscaling",
    "record_downscaling_metrics",
]

#: Method identifiers, matching the Phase 4 downscaler.
NEAREST_METHOD = "nearest"
CUBIC_METHOD = "cubic"
AI_METHOD = "ai"

#: Unit for each reported field metric.
_FIELD_UNITS: dict[str, str] = {
    "mae": "source-unit",
    "rmse": "source-unit",
    "bias": "source-unit",
    "correlation": "1",
}

_NO_REFERENCE_REASON = (
    "No high-resolution reference frames exist for this record, so no "
    "downscaling method can be scored. Interpolation and AI fields are both "
    "omitted rather than compared against each other, because a comparison "
    "with no truth has nothing to measure."
)

_NO_CHECKPOINT_REASON = (
    "No trained AI downscaling checkpoint is available, so the AI method is "
    "omitted from the comparison. Its output is not approximated by a "
    "baseline and no AI result is claimed."
)

_NO_THRESHOLD_REASON = (
    "No defensible event threshold and reference truth exist, so POD, FAR and "
    "CSI are reported as not computed. A contingency table against an "
    "invented threshold would be a number with no meaning."
)


def evaluate_downscaling(
    *,
    predictions: Mapping[str, np.ndarray],
    reference: np.ndarray | None,
    dataset: str,
    split: str = "test",
    variable: str = "precipitation",
    event_threshold: float | None = None,
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    """Score every available downscaling method on identical test frames.

    Args:
        predictions: ``{method_name: predicted_field}``. All must have the same
            shape as ``reference``; the underlying comparison enforces this, so
            a method scored on different frames cannot slip through.
        reference: The high-resolution truth, or ``None`` when none exists.
        dataset: Dataset identifier, recorded in the result.
        split: Which split was scored. Always the test split in practice.
        variable: The variable being downscaled.
        event_threshold: Value defining an "event". Required for POD/FAR; when
            omitted, those are reported as not computed.
        limitations: Extra limitations to record verbatim.

    Returns:
        A record with per-method metrics, per-metric verdicts against the
        nearest baseline, and explicit unavailable entries.
    """
    from extreme_weather.validation.metrics import compute_downscaling_comparison

    available = {name: field for name, field in predictions.items() if field is not None}
    record: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "variable": variable,
        "methods": {},
        "comparisons": {},
        "unavailable": {},
        "frame_count": 0,
        "event_threshold": event_threshold,
    }

    if reference is None or not available:
        record["unavailable"]["all_methods"] = _NO_REFERENCE_REASON
        record["reason"] = _NO_REFERENCE_REASON
        return record

    if AI_METHOD not in available:
        record["unavailable"][AI_METHOD] = _NO_CHECKPOINT_REASON

    try:
        comparison = compute_downscaling_comparison(
            available,
            reference,
            dataset=dataset,
            split=split,
            variable=variable,
            event_threshold=event_threshold,
            ai_method=AI_METHOD,
            limitations=limitations,
        )
    except ValueError as exc:
        record["unavailable"]["all_methods"] = str(exc)
        record["reason"] = str(exc)
        return record

    payload = comparison.as_dict()
    # Phase 7's `frame_count` is the number of *values* compared, not the
    # number of time frames. It is carried through under its original name for
    # consistency, and the two unambiguous counts are added alongside so a
    # reader is not misled by the label.
    record["frame_count"] = int(payload.get("frame_count", 0) or 0)
    reference_array = np.asarray(reference)
    record["frames"] = int(reference_array.shape[0]) if reference_array.ndim else 0
    record["sample_count"] = int(reference_array.size)
    record["methods"] = dict(payload.get("methods", {}) or {})
    record["comparisons"] = dict(payload.get("comparison", {}) or {})
    record["classification"] = dict(payload.get("classification", {}) or {})
    record["peak_preservation"] = dict(payload.get("peak_preservation", {}) or {})
    record["structural"] = dict(payload.get("structural", {}) or {})
    record["limitations"] = list(payload.get("limitations", []) or [])
    record["interpretation"] = payload.get("interpretation", "")
    # Phase 7 calls this `winner_by_mae`; keep that name so the two phases agree.
    record["winner_by_mae"] = payload.get("winner_by_mae")
    record["comparison_summary"] = payload.get("comparison", {}).get("_summary")

    # POD/FAR/CSI exist only behind an explicit threshold and a real reference.
    if event_threshold is None:
        record["unavailable"]["pod_far_csi"] = _NO_THRESHOLD_REASON
    elif AI_METHOD not in available:
        record["unavailable"]["ai_pod_far_csi"] = _NO_CHECKPOINT_REASON

    return record


def record_downscaling_metrics(
    result: Mapping[str, Any], run: Any
) -> list[MetricResult]:
    """Fold a downscaling comparison into a run's metrics.

    Measured values go to ``metrics``; absent ones go to ``unavailable_metrics``
    with their reason, so the run can never present a missing downscaling
    result as a zero.

    Args:
        result: The record returned by :func:`evaluate_downscaling`.
        run: The :class:`~extreme_weather.evaluation.models.EvaluationRun`.

    Returns:
        The metrics that were recorded.
    """
    recorded: list[MetricResult] = []
    frames = int(result.get("frame_count", 0) or 0)

    for name, entry in (result.get("methods", {}) or {}).items():
        # The Phase 7 payload stores each method's metrics flat under its own
        # name, not nested under a "metrics" key.
        stats = entry if isinstance(entry, dict) else {}
        for key, unit in _FIELD_UNITS.items():
            value = stats.get(key) if isinstance(stats, dict) else None
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                recorded.append(
                    MetricResult.measured(
                        f"{name}_{key}",
                        float(value),
                        unit=unit,
                        sample_count=frames,
                        method=f"{name} downscaling on the test split",
                    )
                )
            else:
                recorded.append(
                    MetricResult.unavailable(
                        f"{name}_{key}",
                        f"The {name} method produced no finite {key} for these frames.",
                        unit=unit,
                    )
                )

    for name, reason in (result.get("unavailable", {}) or {}).items():
        if name == "all_methods":
            for method in (NEAREST_METHOD, CUBIC_METHOD, AI_METHOD):
                recorded.append(
                    MetricResult.unavailable(f"{method}_mae", reason, unit="source-unit")
                )
        else:
            recorded.append(MetricResult.unavailable(f"{name}_mae", reason, unit="source-unit"))

    for metric in recorded:
        run.add_metric("downscaling", metric)
    return recorded
