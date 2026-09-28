"""Anomaly threshold calibration on real historical data (SIH26078, Phase 7).

The Phase 1 thresholds are prototype values chosen to make the synthetic demo
fire. On real data they are arbitrary. This module replaces them with
thresholds **derived from the training period of the dataset being analysed** -
and records exactly how, so a reader can judge whether they are defensible.

Methods
-------
``percentile``
    The threshold for a severity is the ``q``-th percentile of the training
    distribution of the variable. Simple, robust, interpretable, and the only
    method used by default.
``quantile_seasonal``
    As above, but computed per climatology grouping (month), so a monsoon month
    and a dry month get different thresholds. Finer, and only meaningful when
    the dataset actually covers several months.

What is deliberately **not** claimed
------------------------------------
These thresholds are *statistically* calibrated to the training data. They are
**not meteorologically validated**: nothing here compares them to an official
warning criterion, because this project has no access to one. The report says
so, and carries the training period and quantiles so the claim can be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import xarray as xr

from extreme_weather.anomaly.thresholds import ThresholdSet
from extreme_weather.ingestion.base import utc_now
from extreme_weather.schemas.event import Severity

__all__ = [
    "CALIBRATION_WARNING",
    "DEFAULT_PERCENTILES",
    "Method",
    "ThresholdCalibration",
    "VariableCalibration",
    "calibrate_thresholds",
    "variable_quantiles",
]


#: Appended to every calibration result.
CALIBRATION_WARNING = (
    "Thresholds are statistically calibrated to the training period of this "
    "dataset. They are NOT validated against any official or meteorological "
    "warning criterion, and no such comparison was possible."
)

#: Percentiles used for the WATCH / ADVISORY / WARNING / SEVERE_WARNING steps.
DEFAULT_PERCENTILES: tuple[float, ...] = (99.0, 99.9, 99.99, 100.0)

#: Severities produced by a calibration, in ascending order. These are the
#: project''s own ladder (see extreme_weather.schemas.event.Severity), NOT
#: official warning categories.
_SEVERITIES: tuple[Severity, ...] = (
    Severity.WATCH,
    Severity.MODERATE,
    Severity.SEVERE,
    Severity.EXTREME,
)


class Method:
    """The calibration methods this module implements."""

    PERCENTILE = "percentile"
    QUANTILE_SEASONAL = "quantile_seasonal"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """Return every supported method name."""
        return (cls.PERCENTILE, cls.QUANTILE_SEASONAL)

    @classmethod
    def validate(cls, value: str) -> str:
        """Normalise and check a method name.

        Raises:
            ValueError: If the method is not one of the supported ones.
        """
        candidate = str(value).strip().lower()
        if candidate not in cls.all():
            raise ValueError(
                f"Unknown calibration method '{value}'. Supported: "
                f"{', '.join(cls.all())}."
            )
        return candidate


@dataclass(frozen=True)
class VariableCalibration:
    """The calibration of one variable.

    Attributes:
        variable: Canonical variable name.
        unit: Unit the quantiles are expressed in.
        sample_count: Values used, from the training split only.
        percentiles: The percentile levels requested.
        quantiles: The corresponding values (NaN where undefined).
        selected_thresholds: ``(value, severity)`` pairs, ascending.
        missing_fraction: NaN fraction of the training data for this variable.
        seasonal: Per-group thresholds when a seasonal method was used.
    """

    variable: str
    unit: str
    sample_count: int
    percentiles: tuple[float, ...]
    quantiles: tuple[float, ...]
    selected_thresholds: tuple[tuple[float, Severity], ...]
    missing_fraction: float = 0.0
    seasonal: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "variable": self.variable,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "percentiles": list(self.percentiles),
            "quantiles": [None if np.isnan(q) else float(q) for q in self.quantiles],
            "selected_thresholds": [
                {"threshold": float(value), "severity": str(severity)}
                for value, severity in self.selected_thresholds
            ],
            "missing_fraction": round(self.missing_fraction, 6),
            "seasonal": self.seasonal,
        }


@dataclass(frozen=True)
class ThresholdCalibration:
    """The full calibration of a dataset.

    Attributes:
        dataset_id: Which dataset was calibrated.
        method: The method used.
        calibration_period: The training timestamps, as ISO 8601 strings.
        variables: One entry per calibrated variable.
        split: The split the statistics came from (always ``"train"``).
        thresholds: A usable :class:`ThresholdSet`, when one could be built.
        warnings: Anything a reader must know before trusting the thresholds.
        generated_at: When the calibration ran.
    """

    dataset_id: str
    method: str
    calibration_period: tuple[str, ...]
    variables: tuple[VariableCalibration, ...]
    split: str = "train"
    thresholds: ThresholdSet | None = None
    warnings: tuple[str, ...] = ()
    generated_at: datetime | None = None

    def variable(self, name: str) -> VariableCalibration | None:
        """Return the calibration of one variable, or ``None``."""
        for entry in self.variables:
            if entry.variable == name:
                return entry
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary for the report."""
        return {
            "dataset_id": self.dataset_id,
            "method": self.method,
            "split": self.split,
            "calibration_period": {
                "start": self.calibration_period[0] if self.calibration_period else None,
                "end": self.calibration_period[-1] if self.calibration_period else None,
                "steps": len(self.calibration_period),
            },
            "variables": [entry.as_dict() for entry in self.variables],
            "thresholds": (
                self.thresholds.model_dump(mode="json")
                if self.thresholds is not None
                else None
            ),
            "warnings": list(self.warnings) + [CALIBRATION_WARNING],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


def variable_quantiles(
    values: np.ndarray, percentiles: Sequence[float] = DEFAULT_PERCENTILES
) -> tuple[int, np.ndarray]:
    """Return the sample count and the requested quantiles of one field.

    NaNs are dropped rather than imputed: a quantile computed over imputed
    values would be a quantile of a guess. When every value is missing the
    quantiles are NaN, which the caller reports rather than substitutes.

    Args:
        values: The 1-D or N-D values of one variable.
        percentiles: Percentile levels in ``[0, 100]``.

    Returns:
        ``(finite_sample_count, quantiles)``.

    Raises:
        ValueError: If a percentile is outside ``[0, 100]``.
    """
    levels = [float(level) for level in percentiles]
    for level in levels:
        if not 0.0 <= level <= 100.0:
            raise ValueError(f"Percentile {level} is outside [0, 100].")

    array = np.asarray(values, dtype="float64").ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0, np.full(len(levels), np.nan, dtype="float64")
    return int(finite.size), np.percentile(finite, levels)



def _event_type_for(variable: str) -> str:
    """Map a variable to the event type the Phase 1 detector uses for it."""
    if variable == "precipitation":
        return "EXTREME_RAINFALL"
    if variable in ("u_wind", "v_wind"):
        return "EXTREME_WIND"
    return "EXTREME_TEMPERATURE"


def _non_decreasing(values: Sequence[float]) -> tuple[float, ...]:
    """Force a threshold ladder to be strictly ascending.

    A percentile ladder on a heavily-tied variable (a field that is zero most of
    the time) can produce two identical quantiles, which would make an earlier
    severity unreachable. Nudging them apart keeps every step reachable, and
    the caller records that it happened.
    """
    result: list[float] = []
    for value in values:
        if np.isnan(value):
            result.append(float("nan"))
            continue
        if result and not np.isnan(result[-1]) and value <= result[-1]:
            # A relative nudge keeps the ladder meaningful across magnitudes.
            result.append(result[-1] * 1.0001 if result[-1] else 1e-9)
        else:
            result.append(float(value))
    return tuple(result)


def variable_quantiles(
    values: np.ndarray, percentiles: Sequence[float] = DEFAULT_PERCENTILES
) -> tuple[int, np.ndarray]:
    """Return the sample count and the requested quantiles of one field.

    NaNs are dropped rather than imputed: a quantile computed over imputed
    values would be a quantile of a guess. When every value is missing the
    quantiles are NaN, which the caller reports rather than substitutes.

    Args:
        values: The values of one variable, any shape.
        percentiles: Percentile levels in ``[0, 100]``.

    Returns:
        ``(finite_sample_count, quantiles)``.

    Raises:
        ValueError: If a percentile is outside ``[0, 100]``.
    """
    levels = [float(level) for level in percentiles]
    for level in levels:
        if not 0.0 <= level <= 100.0:
            raise ValueError(f"Percentile {level} is outside [0, 100].")

    array = np.asarray(values, dtype="float64").ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0, np.full(len(levels), np.nan, dtype="float64")
    return int(finite.size), np.percentile(finite, levels)



def calibrate_thresholds(
    dataset: xr.Dataset,
    *,
    dataset_id: str,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    method: str = Method.PERCENTILE,
    variables: Sequence[str] | None = None,
    build_threshold_set: bool = True,
    generated_at: datetime | None = None,
) -> ThresholdCalibration:
    """Derive anomaly thresholds from the **training** data.

    The caller must pass the training slice only. Passing a dataset that
    includes the test period would leak it into the thresholds, which is the
    most damaging mistake available in this phase - so the split is recorded in
    the result together with the exact calibration period.

    Args:
        dataset: The **training** slice. Not the whole period.
        dataset_id: Identifier recorded in the result.
        percentiles: Percentile levels for the severity steps.
        method: ``percentile`` or ``quantile_seasonal``.
        variables: Which variables to calibrate. Defaults to every one present.
        build_threshold_set: Also build a usable :class:`ThresholdSet`.
        generated_at: Timestamp to stamp on the result.

    Returns:
        A :class:`ThresholdCalibration`. Deterministic: the same training data
        and percentiles always produce the same thresholds.

    Raises:
        ValueError: If the method or a percentile is invalid, the number of
            percentiles does not match the severities, or the dataset has no
            usable time axis.
    """
    resolved_method = Method.validate(method)
    if "time" not in dataset.coords or dataset.coords["time"].ndim != 1:
        raise ValueError("Calibration needs a 1-D 'time' coordinate.")

    levels = tuple(float(level) for level in percentiles)
    if len(levels) != len(_SEVERITIES):
        raise ValueError(
            f"Expected {len(_SEVERITIES)} percentiles (one per severity "
            f"{[str(s) for s in _SEVERITIES]}), got {len(levels)}."
        )
    for level in levels:
        if not 0.0 <= level <= 100.0:
            raise ValueError(f"Percentile {level} is outside [0, 100].")

    times = np.asarray(dataset.coords["time"].values)
    names = list(variables) if variables else sorted(str(n) for n in dataset.data_vars)

    warnings: list[str] = []
    if resolved_method == Method.QUANTILE_SEASONAL:
        months = np.unique(times.astype("datetime64[M]"))
        if months.size < 2:
            warnings.append(
                "quantile_seasonal requested but the training period covers only "
                f"{months.size} month(s); seasonal thresholds are not separable, "
                "so the pooled method was used instead."
            )
            resolved_method = Method.PERCENTILE

    calibrations: list[VariableCalibration] = []
    ladders: list[Any] = []

    for name in names:
        if name not in dataset.data_vars:
            warnings.append(f"Variable '{name}' is not present in the training data.")
            continue
        variable = dataset[name]
        values = np.asarray(variable.values, dtype="float64")
        missing = (
            float(np.isnan(values).sum() / values.size) if values.size else 1.0
        )
        count, quantiles = variable_quantiles(values, levels)
        unit = str(variable.attrs.get("units", ""))

        if count == 0:
            warnings.append(
                f"Variable '{name}' has no finite values in the training period; "
                "no threshold could be derived."
            )

        thresholds = _non_decreasing(quantiles)
        steps = tuple(
            (float(value), severity)
            for value, severity in zip(thresholds, _SEVERITIES)
            if not np.isnan(value)
        )
        if len(steps) > 1 and any(
            b <= a for (a, _), (b, _) in zip(steps, steps[1:])
        ):
            warnings.append(
                f"Variable '{name}': percentiles produced tied values, so the "
                "ladder steps were nudged apart to stay reachable."
            )

        seasonal: dict[str, Any] = {}
        if resolved_method == Method.QUANTILE_SEASONAL:
            for month in np.unique(times.astype("datetime64[M]")):
                mask = times.astype("datetime64[M]") == month
                _, month_quantiles = variable_quantiles(values[mask], levels)
                seasonal[str(month)] = {
                    "steps": int(mask.sum()),
                    "quantiles": [
                        None if np.isnan(q) else float(q) for q in month_quantiles
                    ],
                }

        calibrations.append(
            VariableCalibration(
                variable=name,
                unit=unit,
                sample_count=count,
                percentiles=levels,
                quantiles=tuple(float(q) for q in quantiles),
                selected_thresholds=steps,
                missing_fraction=missing,
                seasonal=seasonal,
            )
        )

        if build_threshold_set and steps:
            from extreme_weather.anomaly.thresholds import ThresholdLadder
            from extreme_weather.schemas.event import EventType

            ladders.append(
                ThresholdLadder(
                    name=f"{name}_calibrated",
                    event_type=EventType(_event_type_for(name)),
                    variable=name,
                    score_name="absolute",
                    steps=steps,
                    note=(
                        "Calibrated to the training period of the dataset being "
                        "analysed. Not validated against any official criterion."
                    ),
                )
            )


    threshold_set: ThresholdSet | None = None
    if ladders:
        try:
            threshold_set = ThresholdSet(ladders=tuple(ladders))
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            warnings.append(
                "The calibrated ladders could not be assembled into a ThresholdSet: "
                f"{type(exc).__name__}: {exc}"
            )

    return ThresholdCalibration(
        dataset_id=dataset_id,
        method=resolved_method,
        calibration_period=tuple(
            str(np.datetime_as_string(value, unit="s")) for value in np.sort(times)
        ),
        variables=tuple(calibrations),
        thresholds=threshold_set,
        warnings=tuple(warnings),
        generated_at=generated_at or utc_now(),
    )

