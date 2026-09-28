"""Uncertainty estimation for Phase 9 (SIH26078, sections 6-7).

Method: **validation-residual bootstrap**
---------------------------------------

Two options were on the table. An ensemble of independently seeded models is
the textbook choice, but it needs several training runs per evaluation, which
is expensive and would make the suite slow and machine-dependent. A residual
bootstrap needs one trained model and the validation predictions it already
produces.

So this project uses the residual bootstrap, and is explicit about what it does
and does not give:

**What it is.** The validation residuals - ``truth - prediction`` on the
validation split - are resampled with replacement to form empirical quantiles of
the prediction error. The reported interval is ``point + quantile(residuals)``.

**What it is not.** It is *not* a calibrated probability, and it is not a
Bayesian posterior. It is an empirical error distribution, valid only if
validation and test residuals are exchangeable - an assumption this project
**cannot verify**, because verifying it would require using the test split.

The word "confidence" is deliberately avoided throughout. A residual quantile
is not a probability that the prediction is correct, and labelling it as one
would be the single most misleading thing this module could do.

Calibration
-----------
:meth:`evaluate_coverage` measures whether the nominal interval actually covers
the test residuals, using the test split **once**, after the interval width has
been fixed from validation. When the sample is too small to say anything
meaningful, it returns ``INSUFFICIENT_DATA`` rather than a number.

Reproducibility
---------------
Every draw goes through an explicitly seeded
:class:`numpy.random.Generator`, so the same seed gives the same intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from extreme_weather.evaluation.models import (
    INSUFFICIENT_DATA,
    MEASURED,
    MetricResult,
    NOT_AVAILABLE,
)

__all__ = [
    "BOOTSTRAP_METHOD",
    "CalibrationResult",
    "PredictionUncertainty",
    "ResidualBootstrap",
    "evaluate_coverage",
    "residual_intervals",
]

#: The method identifier reported alongside every interval.
BOOTSTRAP_METHOD = "validation-residual-bootstrap"

#: Below this many validation residuals, quantiles are not estimable. Chosen so
#: a 90% interval has at least ~10 samples per tail.
MIN_RESIDUALS_FOR_INTERVALS = 20

#: Below this many test residuals, coverage cannot be measured meaningfully.
MIN_RESIDUALS_FOR_COVERAGE = 30

#: The one-sentence caveat that must accompany every interval.
UNCERTAINTY_CAVEAT = (
    "Empirical validation-residual quantiles. Not a probability and not a "
    "calibrated posterior: coverage is only verified where "
    "evaluate_coverage() reports it, and the exchangeability of validation and "
    "test residuals is an unverified assumption."
)


def _validate_1d(name: str, values: Any) -> np.ndarray:
    """Return a finite-only 1-D float array.

    Non-finite entries are dropped rather than propagated: a NaN residual would
    silently poison every quantile downstream.
    """
    array = np.asarray(values, dtype="float64").ravel()
    return array[np.isfinite(array)]


@dataclass(frozen=True)
class PredictionUncertainty:
    """A point prediction with its uncertainty interval.

    Attributes:
        latitude: Predicted latitude, degrees.
        longitude: Predicted longitude, degrees.
        intensity: Predicted intensity.
        latitude_interval: ``(low, high)`` for latitude, or ``None``.
        longitude_interval: ``(low, high)`` for longitude, or ``None``.
        intensity_interval: ``(low, high)`` for intensity, or ``None``.
        method: Identifier of the uncertainty method.
        nominal: The nominal coverage the interval was built for.
        horizon: Forecast horizon in steps.
        calibration_status: Whether coverage was actually verified.
        sample_count: Validation residuals behind the interval.
    """

    latitude: float
    longitude: float
    intensity: float
    latitude_interval: tuple[float, float] | None
    longitude_interval: tuple[float, float] | None
    intensity_interval: tuple[float, float] | None
    method: str
    nominal: float
    horizon: int = 1
    calibration_status: str = NOT_AVAILABLE
    sample_count: int = 0
    caveat: str = UNCERTAINTY_CAVEAT

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "prediction": {
                "latitude": float(self.latitude),
                "longitude": float(self.longitude),
                "intensity": float(self.intensity),
            },
            "uncertainty": {
                "latitude_interval": list(self.latitude_interval)
                if self.latitude_interval
                else None,
                "longitude_interval": list(self.longitude_interval)
                if self.longitude_interval
                else None,
                "intensity_interval": list(self.intensity_interval)
                if self.intensity_interval
                else None,
                "method": self.method,
                "nominal_coverage": self.nominal,
                "horizon": self.horizon,
                "calibration_status": self.calibration_status,
                "calibration_dataset": "validation",
                "sample_count": self.sample_count,
                "caveat": self.caveat,
            },
        }


@dataclass
class ResidualBootstrap:
    """Empirical residual quantiles fitted on the validation split.

    Attributes:
        residuals: Per-output residual samples, keyed ``latitude``,
            ``longitude`` and ``intensity``.
        sample_count: Residuals used per output.
        method: The method identifier.
        seed: Seed used to draw the bootstrap samples.
        draws: How many bootstrap resamples were taken.
    """

    residuals: dict[str, np.ndarray] = field(default_factory=dict)
    sample_count: int = 0
    method: str = BOOTSTRAP_METHOD
    seed: int = 0
    draws: int = 0

    def quantiles(self, nominal: float) -> dict[str, tuple[float, float] | None]:
        """Return the two-sided residual quantiles for a nominal coverage.

        Args:
            nominal: Nominal coverage, e.g. ``0.8`` for an 80% interval.

        Returns:
            ``{output: (low, high)}``, with ``None`` for any output that has too
            few residuals to estimate.
        """
        if not 0.0 < nominal < 1.0:
            raise ValueError(f"nominal must be in (0, 1), got {nominal}.")
        alpha = (1.0 - float(nominal)) / 2.0
        result: dict[str, tuple[float, float] | None] = {}
        for name in ("latitude", "longitude", "intensity"):
            values = self.residuals.get(name, np.array([], dtype="float64"))
            if values.size < MIN_RESIDUALS_FOR_INTERVALS:
                result[name] = None
                continue
            generator = np.random.default_rng(self.seed)
            drawn = generator.choice(values, size=(self.draws, values.size), replace=True)
            quantiles = np.percentile(drawn, [100.0 * alpha, 100.0 * (1.0 - alpha)], axis=1)
            result[name] = (float(np.median(quantiles[0])), float(np.median(quantiles[1])))
        return result

    def interval_for(
        self,
        latitude: float,
        longitude: float,
        intensity: float,
        *,
        nominal: float = 0.8,
        horizon: int = 1,
        calibration_status: str = NOT_AVAILABLE,
    ) -> PredictionUncertainty:
        """Return a point prediction with its interval attached.

        The residual quantiles are *added to* the point prediction, so the
        reported interval lives in the same space as the prediction it brackets.
        Publishing the raw residuals next to a latitude would invite a reader
        to compare degrees against a near-zero error.
        """
        bounds = self.quantiles(nominal)
        centre = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "intensity": float(intensity),
        }

        def _shifted(name: str) -> tuple[float, float] | None:
            interval = bounds.get(name)
            if interval is None:
                return None
            return (
                float(centre[name] + interval[0]),
                float(centre[name] + interval[1]),
            )

        return PredictionUncertainty(
            latitude=float(latitude),
            longitude=float(longitude),
            intensity=float(intensity),
            latitude_interval=_shifted("latitude"),
            longitude_interval=_shifted("longitude"),
            intensity_interval=_shifted("intensity"),
            method=self.method,
            nominal=float(nominal),
            horizon=int(horizon),
            calibration_status=calibration_status,
            sample_count=self.sample_count,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the fitted object."""
        return {
            "method": self.method,
            "seed": self.seed,
            "draws": self.draws,
            "sample_count": self.sample_count,
            "residual_counts": {
                name: int(values.size) for name, values in self.residuals.items()
            },
            "residual_summary": {
                name: {
                    "mean": float(np.mean(values)) if values.size else None,
                    "std": float(np.std(values)) if values.size else None,
                    "min": float(np.min(values)) if values.size else None,
                    "max": float(np.max(values)) if values.size else None,
                }
                for name, values in self.residuals.items()
            },
            "caveat": UNCERTAINTY_CAVEAT,
        }


def residual_intervals(
    *,
    validation_true_latitude: Sequence[float],
    validation_predicted_latitude: Sequence[float],
    validation_true_longitude: Sequence[float],
    validation_predicted_longitude: Sequence[float],
    validation_true_intensity: Sequence[float],
    validation_predicted_intensity: Sequence[float],
    seed: int = 20240601,
    draws: int = 512,
) -> ResidualBootstrap:
    """Fit residual quantiles from validation predictions.

    Args:
        validation_true_latitude: Actual validation latitudes, degrees.
        validation_predicted_latitude: Predicted validation latitudes, degrees.
        validation_true_longitude: Actual validation longitudes, degrees.
        validation_predicted_longitude: Predicted validation longitudes, degrees.
        validation_true_intensity: Actual validation intensities.
        validation_predicted_intensity: Predicted validation intensities.
        seed: Random seed for the bootstrap. Fixed, so the result is
            reproducible.
        draws: Number of bootstrap resamples.

    Returns:
        A fitted :class:`ResidualBootstrap`. Non-finite residuals are dropped.
    """
    residuals = {
        "latitude": _validate_1d(
            "latitude",
            np.asarray(validation_true_latitude, dtype="float64")
            - np.asarray(validation_predicted_latitude, dtype="float64"),
        ),
        "longitude": _validate_1d(
            "longitude",
            np.asarray(validation_true_longitude, dtype="float64")
            - np.asarray(validation_predicted_longitude, dtype="float64"),
        ),
        "intensity": _validate_1d(
            "intensity",
            np.asarray(validation_true_intensity, dtype="float64")
            - np.asarray(validation_predicted_intensity, dtype="float64"),
        ),
    }
    sizes = [values.size for values in residuals.values()]
    return ResidualBootstrap(
        residuals=residuals,
        sample_count=int(min(sizes)) if sizes else 0,
        seed=int(seed),
        draws=int(draws),
    )


@dataclass
class CalibrationResult:
    """Measured coverage of a nominal interval on the test split.

    Attributes:
        nominal: The nominal coverage that was claimed.
        empirical_coverage: Fraction of test residuals inside the interval.
        interval_width: Mean width of the interval, per output.
        sample_count: Test residuals evaluated.
        status: MEASURED or INSUFFICIENT_DATA.
        reason: Why coverage is unavailable, when it is.
    """

    nominal: float
    empirical_coverage: dict[str, float | None] = field(default_factory=dict)
    interval_width: dict[str, float | None] = field(default_factory=dict)
    sample_count: int = 0
    status: str = NOT_AVAILABLE
    reason: str = ""

    @property
    def is_calibrated(self) -> bool:
        """True when every output's coverage is close to nominal.

        Deliberately a *measurement*, not an assertion: this returns whether
        the numbers say the interval is honest. It does not license any
        stronger claim.
        """
        if self.status != MEASURED:
            return False
        for value in self.empirical_coverage.values():
            if value is None:
                return False
        return all(
            abs(value - self.nominal) <= 0.15
            for value in self.empirical_coverage.values()  # type: ignore[operator]
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "nominal_coverage": self.nominal,
            "empirical_coverage": dict(self.empirical_coverage),
            "interval_width": dict(self.interval_width),
            "sample_count": self.sample_count,
            "status": self.status,
            "reason": self.reason,
            "is_calibrated": self.is_calibrated,
            "evaluated_on": "test",
        }

    def metrics(self) -> list[MetricResult]:
        """Return the calibration as reportable metrics."""
        if self.status != MEASURED:
            return [
                MetricResult.insufficient(
                    f"coverage_{name}", self.reason, sample_count=self.sample_count
                )
                for name in ("latitude", "longitude", "intensity")
            ]
        results: list[MetricResult] = []
        for name, value in self.empirical_coverage.items():
            if value is None:
                results.append(
                    MetricResult.insufficient(
                        f"coverage_{name}",
                        "No finite test residuals for this output.",
                        sample_count=self.sample_count,
                    )
                )
                continue
            results.append(
                MetricResult.measured(
                    f"coverage_{name}",
                    value,
                    unit="fraction",
                    sample_count=self.sample_count,
                    method="empirical coverage of the validation-residual interval",
                )
            )
        return results


def evaluate_coverage(
    bootstrap: ResidualBootstrap,
    *,
    test_true_latitude: Sequence[float],
    test_predicted_latitude: Sequence[float],
    test_true_longitude: Sequence[float],
    test_predicted_longitude: Sequence[float],
    test_true_intensity: Sequence[float],
    test_predicted_intensity: Sequence[float],
    nominal: float = 0.8,
) -> CalibrationResult:
    """Measure whether a nominal interval actually covers the test residuals.

    The test split is used **once**, and only after the interval width has been
    fixed from validation. Nothing here feeds back into the interval, so this
    is a measurement and not a tuning step.

    Args:
        bootstrap: The fitted residual distribution.
        test_true_latitude: Actual test latitudes, degrees.
        test_predicted_latitude: Predicted test latitudes, degrees.
        test_true_longitude: Actual test longitudes, degrees.
        test_predicted_longitude: Predicted test longitudes, degrees.
        test_true_intensity: Actual test intensities.
        test_predicted_intensity: Predicted test intensities.
        nominal: The nominal coverage to check, e.g. ``0.8``.

    Returns:
        A :class:`CalibrationResult`. Returns ``INSUFFICIENT_DATA`` rather than
        a number when the sample is too small for the fraction to mean anything.
    """
    truth = {
        "latitude": _validate_1d("latitude", test_true_latitude),
        "longitude": _validate_1d("longitude", test_true_longitude),
        "intensity": _validate_1d("intensity", test_true_intensity),
    }
    predicted = {
        "latitude": _validate_1d("latitude", test_predicted_latitude),
        "longitude": _validate_1d("longitude", test_predicted_longitude),
        "intensity": _validate_1d("intensity", test_predicted_intensity),
    }

    sizes = [values.size for values in truth.values()]
    count = int(min(sizes)) if sizes else 0
    if count < MIN_RESIDUALS_FOR_COVERAGE:
        return CalibrationResult(
            nominal=float(nominal),
            status=INSUFFICIENT_DATA,
            reason=(
                f"Coverage needs at least {MIN_RESIDUALS_FOR_COVERAGE} test "
                f"residuals to be informative; only {count} were available. "
                "No coverage figure is reported, because any fraction over a "
                "sample this small would be dominated by sampling noise."
            ),
            sample_count=count,
        )

    bounds = bootstrap.quantiles(float(nominal))
    coverage: dict[str, float | None] = {}
    width: dict[str, float | None] = {}
    for name in ("latitude", "longitude", "intensity"):
        interval = bounds.get(name)
        if interval is None or truth[name].size == 0:
            coverage[name] = None
            width[name] = None
            continue
        low, high = interval
        errors = np.abs(truth[name] - predicted[name])
        inside = np.count_nonzero((errors >= low) & (errors <= high))
        coverage[name] = float(inside / errors.size)
        width[name] = float(high - low)

    return CalibrationResult(
        nominal=float(nominal),
        empirical_coverage=coverage,
        interval_width=width,
        sample_count=count,
        status=MEASURED,
    )
