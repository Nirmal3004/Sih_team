"""Evaluation metrics for downscaling (SIH26078, Phase 4).

> **These are agreement metrics against a SYNTHETIC reference.** They describe how
> closely a method reproduces the generated high-resolution target. They are not
> forecast verification, not skill scores against observations, and not evidence
> of real-world accuracy. No meteorological skill score is invented here.

Continuous error metrics
------------------------
With ``p`` the prediction, ``o`` the reference and ``n`` the number of points:

```text
MAE  = mean(|p - o|)
RMSE = sqrt(mean((p - o)^2))
Bias = mean(p - o)          # > 0 means the method over-predicts
```

``Bias`` is signed on purpose: a large |Bias| with a small MAE means a systematic
offset, a different failure from random error and one that should be read as such.

Correlation
-----------
``r`` is the Pearson correlation over the flattened arrays, reported as ``None``
when either array is constant. The correlation of a constant field is
mathematically undefined (0/0), not zero - and "not computable" and "uncorrelated"
are different facts, so the distinction is preserved.

Wet/dry classification metrics
------------------------------
With a configurable threshold ``t`` (default 0.1 mm h-1), a point is *wet* when
the value is ``>= t``. With ``a`` hits, ``b`` false alarms, ``c`` misses:

```text
POD = a / (a + c)              probability of detection
FAR = b / (a + b)              false alarm ratio
CSI = a / (a + b + c)          critical success index
frequency_bias = (a + b) - (a + c)   in grid points, not a percentage
```

* High POD with high FAR means the method detects everything by predicting rain
  everywhere, so POD must never be read alone.
* ``CSI`` penalises misses *and* false alarms in one number, making it a fairer
  single summary.
* ``frequency_bias`` counts **grid points** of excess (positive) or deficit
  (negative) wet area, so its magnitude scales with the domain; it is
  deliberately not a percentage.

Undefined cases return ``None`` rather than a fabricated 0 or 1 (for example POD
with no observed events). The comparison table renders those cells as ``n/a``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_WET_THRESHOLD",
    "ClassificationMetrics",
    "DownscalingMetrics",
    "ErrorMetrics",
    "classification_metrics",
    "error_metrics",
    "evaluate",
    "pearson_correlation",
]

#: Default wet/dry threshold in mm h-1. Configurable, never hard-coded downstream.
DEFAULT_WET_THRESHOLD: float = 0.1


def _as_arrays(prediction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype="float64")
    ref = np.asarray(reference, dtype="float64")
    if pred.shape != ref.shape:
        raise ValueError(f"prediction and reference shapes differ: {pred.shape} vs {ref.shape}.")
    if pred.size == 0:
        raise ValueError("Cannot compute metrics on an empty array.")
    return pred, ref


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when it is undefined.

    Returns ``None`` if either array is constant (zero denominator) or contains a
    non-finite value.
    """
    left, right = _as_arrays(a, b)
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return None
    left_centred = left - left.mean()
    right_centred = right - right.mean()
    denominator = float(np.sqrt(np.sum(left_centred**2) * np.sum(right_centred**2)))
    if denominator <= 0.0:
        return None
    return float(np.sum(left_centred * right_centred) / denominator)


@dataclass(frozen=True)
class ErrorMetrics:
    """Continuous error metrics of one downscaling method.

    Attributes:
        mae: Mean absolute error, in the variable's units (mm h-1).
        rmse: Root mean squared error, same units.
        bias: Mean signed error; positive means over-prediction.
        correlation: Pearson correlation, or ``None`` when undefined.
        n_points: Number of grid points compared.
    """

    mae: float
    rmse: float
    bias: float
    correlation: float | None
    n_points: int

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return asdict(self)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Wet/dry contingency metrics of one method.

    Attributes:
        pod: Probability of detection, or ``None`` when there are no events.
        far: False alarm ratio, or ``None`` when nothing was predicted.
        csi: Critical success index, or ``None`` when undefined.
        frequency_bias: ``(a + b) - (a + c)`` in grid points.
        hits: Number of correctly predicted wet points.
        false_alarms: Wet points predicted where the reference is dry.
        misses: Wet points missed.
        correct_negatives: Dry points correctly predicted dry.
        threshold: Wet/dry threshold used, in the variable's units.
    """

    pod: float | None
    far: float | None
    csi: float | None
    frequency_bias: int
    hits: int
    false_alarms: int
    misses: int
    correct_negatives: int
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return asdict(self)


@dataclass(frozen=True)
class DownscalingMetrics:
    """Every metric of one method, forming one comparison row.

    Attributes:
        method: Method name (``nearest``, ``cubic``, ``ai``, ...).
        error: Continuous error metrics.
        classification: Wet/dry metrics, or ``None`` when not requested.
    """

    method: str
    error: ErrorMetrics
    classification: ClassificationMetrics | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view used by the API and the CLI."""
        payload: dict[str, Any] = {"method": self.method, **self.error.as_dict()}
        classification = self.classification
        payload["pod"] = classification.pod if classification else None
        payload["far"] = classification.far if classification else None
        payload["csi"] = classification.csi if classification else None
        payload["frequency_bias"] = (
            classification.frequency_bias if classification else None
        )
        return payload


def error_metrics(method: str, prediction: np.ndarray, reference: np.ndarray) -> ErrorMetrics:
    """Compute MAE, RMSE, Bias and correlation for one method."""
    pred, ref = _as_arrays(prediction, reference)
    error = pred - ref
    return ErrorMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(np.mean(error)),
        correlation=pearson_correlation(pred, ref),
        n_points=int(pred.size),
    )


def classification_metrics(
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    threshold: float = DEFAULT_WET_THRESHOLD,
) -> ClassificationMetrics:
    """Compute the wet/dry contingency metrics for one method.

    Args:
        method: Method name (kept for symmetry with the other helpers).
        prediction: Predicted field.
        reference: Reference field.
        threshold: Wet/dry threshold in the variable's units.

    Returns:
        A :class:`ClassificationMetrics`; undefined ratios are ``None``.
    """
    pred, ref = _as_arrays(prediction, reference)
    predicted_wet = pred >= threshold
    observed_wet = ref >= threshold

    hits = int(np.sum(predicted_wet & observed_wet))
    false_alarms = int(np.sum(predicted_wet & ~observed_wet))
    misses = int(np.sum(~predicted_wet & observed_wet))
    correct_negatives = int(np.sum(~predicted_wet & ~observed_wet))

    pod = hits / (hits + misses) if (hits + misses) > 0 else None
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else None
    csi = (
        hits / (hits + false_alarms + misses)
        if (hits + false_alarms + misses) > 0
        else None
    )

    return ClassificationMetrics(
        pod=None if pod is None else float(pod),
        far=None if far is None else float(far),
        csi=None if csi is None else float(csi),
        frequency_bias=(hits + false_alarms) - (hits + misses),
        hits=hits,
        false_alarms=false_alarms,
        misses=misses,
        correct_negatives=correct_negatives,
        threshold=float(threshold),
    )


def evaluate(
    method: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    threshold: float | None = DEFAULT_WET_THRESHOLD,
) -> DownscalingMetrics:
    """Compute every metric of one method.

    Args:
        method: Method name, carried into the result.
        prediction: Predicted field.
        reference: Reference field.
        threshold: Wet/dry threshold; ``None`` skips the classification metrics.

    Returns:
        A :class:`DownscalingMetrics`.
    """
    classification = (
        None
        if threshold is None
        else classification_metrics(method, prediction, reference, threshold)
    )
    return DownscalingMetrics(
        method=method,
        error=error_metrics(method, prediction, reference),
        classification=classification,
    )


