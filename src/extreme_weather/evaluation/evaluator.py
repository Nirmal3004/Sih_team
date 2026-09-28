"""The Phase 9 evaluation runner (SIH26078, sections 3-5).

This module performs the measurements and assembles an
:class:`~extreme_weather.evaluation.models.EvaluationRun`. Three rules govern
everything it does:

**No test-set tuning.** The test split is read exactly once, to score models
whose configuration was already fixed. Uncertainty widths are fitted on
validation; coverage is measured on test afterwards and never feeds back.

**No fabricated metrics.** Every absent quantity becomes a
:class:`MetricResult` with a reason. There is no code path that turns a
missing measurement into ``0.0``.

**No superiority claims without measurement.** A model is described as beating a
baseline only when :func:`~extreme_weather.validation.metrics.compare_metrics`
says ``improves`` on that specific metric. A model that ties or loses is
reported as tying or losing, and a model that cannot be scored at all is
reported as unavailable.

GNN horizons
------------
One-step prediction and multi-step rollout are evaluated separately and never
pooled. A recursive rollout compounds its own error, so mixing the two would
hide exactly the effect worth seeing. A horizon with no valid targets is
reported ``INSUFFICIENT_DATA`` rather than skipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from extreme_weather.evaluation.baselines import (
    CONSTANT_VELOCITY,
    PERSISTENCE,
    describe_baselines,
    evaluate_baseline,
)
from extreme_weather.evaluation.dataset import EvaluationDataset, LeakageError
from extreme_weather.evaluation.models import (
    FAILED,
    INSUFFICIENT_DATA,
    MEASURED,
    MetricResult,
    NOT_AVAILABLE,
    PARTIAL,
    Provenance,
    RUN_INSUFFICIENT,
    SUCCESS,
    EvaluationRun,
    utc_now,
)
from extreme_weather.evaluation.uncertainty import (
    BOOTSTRAP_METHOD,
    ResidualBootstrap,
    evaluate_coverage,
    residual_intervals,
)
from extreme_weather.validation.metrics import compare_metrics

__all__ = [
    "EVALUATION_CODE_FILES",
    "GNN_METRIC_UNITS",
    "EvaluationOutcome",
    "code_fingerprint",
    "evaluate_gnn_baselines",
    "evaluate_multistep_rollout",
    "run_evaluation",
    "summarise_gnn_comparison",
]

UTC = timezone.utc

#: Source files whose content determines a result, hashed into the provenance
#: so a number can be traced back to the code that produced it.
EVALUATION_CODE_FILES: tuple[str, ...] = (
    "evaluation/baselines.py",
    "evaluation/dataset.py",
    "evaluation/evaluator.py",
    "evaluation/explainability.py",
    "evaluation/metrics.py",
    "evaluation/uncertainty.py",
    "validation/metrics.py",
    "validation/splits.py",
)

#: Unit for each GNN metric, so a report never leaves a number unqualified.
GNN_METRIC_UNITS: dict[str, str] = {
    "latitude_mae_deg": "deg",
    "longitude_mae_deg": "deg",
    "latitude_rmse_deg": "deg",
    "longitude_rmse_deg": "deg",
    "intensity_mae": "source-unit",
    "intensity_rmse": "source-unit",
    "displacement_error_km_mean": "km",
    "displacement_error_km_median": "km",
    "displacement_error_km_std": "km",
    "displacement_error_km_p90": "km",
    "combined_loss": "deg+source-unit",
}

#: How many samples a horizon needs before its numbers are reported at all.
MIN_HORIZON_SAMPLES = 1


@dataclass
class EvaluationOutcome:
    """What one evaluation produced.

    Attributes:
        run: The run record.
        dataset: The dataset it was measured on, when one was built.
    """

    run: EvaluationRun
    dataset: EvaluationDataset | None = None

    @property
    def succeeded(self) -> bool:
        """True when the run finished and produced at least one measurement."""
        return self.run.status in (SUCCESS, PARTIAL) and bool(self.run.metrics)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        payload = self.run.as_dict()
        if self.dataset is not None:
            payload["dataset"] = self.dataset.as_dict()
        return payload


def code_fingerprint() -> str | None:
    """Hash the evaluation source files, so a result records its own code.

    Returns:
        A hex digest, or ``None`` when the source tree cannot be located.
    """
    try:
        import extreme_weather.evaluation as package

        root = Path(package.__file__).resolve().parent.parent
    except Exception:  # pragma: no cover - defensive
        return None

    digest = hashlib.sha256()
    found = 0
    for relative in EVALUATION_CODE_FILES:
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
        found += 1
    if not found:
        return None
    return digest.hexdigest()[:32]


def _environment_facts() -> dict[str, str]:
    """Return platform facts, reusing the Phase 7 helper."""
    try:
        from extreme_weather.validation.provenance import environment_facts

        return environment_facts()
    except Exception:  # pragma: no cover - defensive
        return {}


def _empty_arrays() -> dict[str, np.ndarray]:
    """Return an empty array for every expected sample field."""
    return {
        name: np.array([], dtype="float64")
        for name in (
            "current_latitude",
            "current_longitude",
            "current_intensity",
            "previous_latitude",
            "previous_longitude",
            "predicted_latitude",
            "predicted_longitude",
            "predicted_intensity",
            "true_latitude",
            "true_longitude",
            "true_intensity",
        )
    }


def _finite_lengths(samples: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return the finite count per field, for the smallest usable sample set."""
    return {name: int(np.isfinite(np.asarray(v, dtype="float64")).sum()) for name, v in samples.items()}


def _require_aligned(
    samples: Mapping[str, np.ndarray], names: Sequence[str]
) -> int:
    """Return the number of samples usable across all named fields.

    Raises:
        ValueError: If a required field is absent.
    """
    missing = [name for name in names if name not in samples]
    if missing:
        raise ValueError(f"Sample set is missing {missing}.")
    sizes = [int(np.asarray(samples[name]).size) for name in names]
    return min(sizes)


def evaluate_gnn_baselines(
    samples: Mapping[str, np.ndarray],
    *,
    model_available: bool,
    model_checkpoint: str | None = None,
    horizon: int = 1,
) -> dict[str, Any]:
    """Score a GNN and both baselines on identical samples.

    The model and the baselines are scored on the *same* current and future
    states, so any difference in the numbers is the method and not the sample.

    Args:
        samples: ``current_*``, ``previous_*``, ``predicted_*`` and ``true_*``
            arrays. ``previous_*`` is required only for constant velocity.
        model_available: Whether a trained checkpoint exists.
        model_checkpoint: Checkpoint path or version, for provenance.
        horizon: Forecast horizon in steps, recorded in the result.

    Returns:
        A record with per-method metrics, the per-metric verdicts against both
        baselines, and explicit unavailable entries when the model cannot be
        scored.

    Raises:
        ValueError: If no usable samples were supplied.
    """
    required = (
        "current_latitude",
        "current_longitude",
        "current_intensity",
        "true_latitude",
        "true_longitude",
        "true_intensity",
    )
    count = _require_aligned(samples, required)
    if count == 0:
        raise ValueError(
            "GNN evaluation needs at least one sample with a known next state."
        )

    record: dict[str, Any] = {
        "horizon": int(horizon),
        "horizon_kind": "one_step" if int(horizon) == 1 else f"{horizon}_step_rollout",
        "sample_count": int(count),
        "model_available": bool(model_available),
        "model_checkpoint": model_checkpoint,
        "baselines": describe_baselines(),
        "methods": {},
        "comparisons": {},
        "unavailable": {},
    }

    # --- baselines: always computable from the samples ------------------
    persistence = evaluate_baseline(
        PERSISTENCE,
        current_latitude=samples["current_latitude"],
        current_longitude=samples["current_longitude"],
        current_intensity=samples["current_intensity"],
        future_latitude=samples["true_latitude"],
        future_longitude=samples["true_longitude"],
        future_intensity=samples["true_intensity"],
    )
    record["methods"][PERSISTENCE] = persistence

    has_previous = (
        "previous_latitude" in samples
        and "previous_longitude" in samples
        and np.asarray(samples["previous_latitude"]).size >= count
        and np.asarray(samples["previous_longitude"]).size >= count
        and np.isfinite(np.asarray(samples["previous_latitude"], dtype="float64")[:count]).all()
        and np.isfinite(np.asarray(samples["previous_longitude"], dtype="float64")[:count]).all()
    )
    if has_previous:
        record["methods"][CONSTANT_VELOCITY] = evaluate_baseline(
            CONSTANT_VELOCITY,
            current_latitude=samples["current_latitude"],
            current_longitude=samples["current_longitude"],
            current_intensity=samples["current_intensity"],
            previous_latitude=samples["previous_latitude"],
            previous_longitude=samples["previous_longitude"],
            future_latitude=samples["true_latitude"],
            future_longitude=samples["true_longitude"],
            future_intensity=samples["true_intensity"],
        )
    else:
        record["methods"][CONSTANT_VELOCITY] = {
            "method": CONSTANT_VELOCITY,
            "status": INSUFFICIENT_DATA,
            "reason": (
                "The evaluation record has no previous-step position for some "
                "samples, so a movement velocity cannot be formed. The "
                "constant-velocity baseline is therefore not computed rather "
                "than being reported as zero error."
            ),
        }
        record["unavailable"][CONSTANT_VELOCITY] = record["methods"][CONSTANT_VELOCITY]["reason"]

    # --- the model ------------------------------------------------------
    model_metrics: dict[str, Any] | None = None
    if not model_available:
        reason = (
            "No trained GNN checkpoint was available, so the model could not be "
            "scored. Its absence is reported rather than simulated, and no model "
            "result is inferred from the baselines."
        )
        record["methods"]["gnn"] = {
            "method": "gnn",
            "status": NOT_AVAILABLE,
            "reason": reason,
        }
        record["unavailable"]["gnn"] = reason
    else:
        for name in ("predicted_latitude", "predicted_longitude", "predicted_intensity"):
            if name not in samples or np.asarray(samples[name]).size < count:
                record["methods"]["gnn"] = {
                    "method": "gnn",
                    "status": NOT_AVAILABLE,
                    "reason": (
                        f"The model produced no '{name}' for the evaluation "
                        "samples, so it cannot be scored. No result is inferred."
                    ),
                }
                record["unavailable"]["gnn"] = record["methods"]["gnn"]["reason"]
                model_metrics = None
                break
            if not np.isfinite(np.asarray(samples[name], dtype="float64")[:count]).all():
                record["methods"]["gnn"] = {
                    "method": "gnn",
                    "status": NOT_AVAILABLE,
                    "reason": (
                        f"The model's '{name}' contains non-finite values, so its "
                        "error cannot be computed. Those samples are excluded from "
                        "the comparison entirely rather than being scored as zero error."
                    ),
                }
                record["unavailable"]["gnn"] = record["methods"]["gnn"]["reason"]
                model_metrics = None
                break
        if model_metrics is None and "gnn" not in record["methods"]:
            from extreme_weather.validation.metrics import haversine_km

            pred_lat = np.asarray(samples["predicted_latitude"], dtype="float64")[:count]
            pred_lon = np.asarray(samples["predicted_longitude"], dtype="float64")[:count]
            pred_int = np.asarray(samples["predicted_intensity"], dtype="float64")[:count]
            true_lat = np.asarray(samples["true_latitude"], dtype="float64")[:count]
            true_lon = np.asarray(samples["true_longitude"], dtype="float64")[:count]
            true_int = np.asarray(samples["true_intensity"], dtype="float64")[:count]

            lat_error = true_lat - pred_lat
            lon_error = true_lon - pred_lon
            intensity_error = true_int - pred_int
            displacement = np.array(
                [haversine_km(a, b, c, d) for a, b, c, d in zip(pred_lat, pred_lon, true_lat, true_lon)],
                dtype="float64",
            )
            model_metrics = {
                "method": "gnn",
                "sample_count": int(count),
                "latitude_mae_deg": float(np.mean(np.abs(lat_error))),
                "longitude_mae_deg": float(np.mean(np.abs(lon_error))),
                "intensity_mae": float(np.mean(np.abs(intensity_error))),
                "latitude_rmse_deg": float(np.sqrt(np.mean(lat_error**2))),
                "longitude_rmse_deg": float(np.sqrt(np.mean(lon_error**2))),
                "intensity_rmse": float(np.sqrt(np.mean(intensity_error**2))),
                "displacement_error_km_mean": float(np.mean(displacement)),
                "displacement_error_km_median": float(np.median(displacement)),
                "displacement_error_km_std": float(np.std(displacement)),
                "displacement_error_km_p90": float(np.percentile(displacement, 90)),
                "combined_loss": float(np.mean(np.abs(lat_error) + np.abs(lon_error) + np.abs(intensity_error))),
            }
            record["methods"]["gnn"] = model_metrics

    # --- verdicts -------------------------------------------------------
    if model_metrics is not None:
        for baseline_name, baseline in record["methods"].items():
            if baseline_name == "gnn" or "latitude_mae_deg" not in baseline:
                continue
            record["comparisons"][f"gnn_vs_{baseline_name}"] = {
                key: compare_metrics(model_metrics.get(key), baseline.get(key))
                for key in GNN_METRIC_UNITS
                if key in model_metrics and key in baseline
            }
    record["interpretation"] = (
        "Verdicts are stated per metric. A model that ties or loses to a "
        "baseline is reported as such; no average verdict is claimed, because "
        "beating one metric and losing another is not an overall win."
    )
    return record


def summarise_gnn_comparison(comparison: dict[str, Any]) -> str:
    """Return a one-line, non-overclaiming summary of a GNN comparison.

    Args:
        comparison: The record returned by :func:`evaluate_gnn_baselines`.

    Returns:
        A sentence stating what was and was not measured.
    """
    gnn = comparison.get("methods", {}).get("gnn", {})
    if "latitude_mae_deg" not in gnn:
        return (
            "The GNN was not scored: "
            f"{gnn.get('reason', 'no reason recorded')}"
        )
    verdicts = comparison.get("comparisons", {}).get("gnn_vs_persistence", {})
    if not verdicts:
        return "The GNN was scored; no verdict could be formed against any baseline."
    wins = [k for k, v in verdicts.items() if v.get("verdict") == "improves"]
    losses = [k for k, v in verdicts.items() if v.get("verdict") == "worse"]
    ties = [k for k, v in verdicts.items() if v.get("verdict") == "tie"]
    parts = []
    if wins:
        parts.append(f"{len(wins)} metric(s) better than persistence")
    if losses:
        parts.append(f"{len(losses)} worse")
    if ties:
        parts.append(f"{len(ties)} tied")
    return (
        f"Against persistence on {len(verdicts)} metric(s): "
        + (", ".join(parts) if parts else "no verdict")
        + ". This is a per-metric statement, not an overall claim of superiority."
    )


def evaluate_multistep_rollout(
    samples_by_horizon: Mapping[int, Mapping[str, np.ndarray]],
    *,
    model_available: bool,
) -> dict[str, Any]:
    """Evaluate several forecast horizons, keeping them strictly separate.

    One-step prediction and recursive rollout are **never pooled**: a rollout
    compounds its own error, so averaging the two would hide the very effect
    the comparison exists to show.

    Args:
        samples_by_horizon: ``{horizon: samples}`` for each horizon to score.
        model_available: Whether a trained checkpoint exists.

    Returns:
        A record with one entry per requested horizon, including horizons that
        could not be evaluated.
    """
    horizons: dict[str, Any] = {}
    for horizon, samples in sorted(samples_by_horizon.items()):
        try:
            horizons[str(horizon)] = evaluate_gnn_baselines(
                samples, model_available=model_available, horizon=int(horizon)
            )
        except ValueError as exc:
            horizons[str(horizon)] = {
                "horizon": int(horizon),
                "horizon_kind": "one_step" if int(horizon) == 1 else f"{horizon}_step_rollout",
                "status": INSUFFICIENT_DATA,
                "reason": str(exc),
                "sample_count": 0,
                "methods": {},
                "comparisons": {},
            }
    return {
        "horizons": horizons,
        "available_horizons": [
            h for h, entry in horizons.items() if "latitude_mae_deg" in entry.get("methods", {}).get("gnn", {})
        ],
        "note": (
            "One-step prediction and recursive multi-step rollout are reported "
            "separately and are never averaged together. A recursive rollout "
            "feeds its own output back in, so its error grows with horizon for "
            "reasons that are structural, not necessarily model error."
        ),
    }


def _deterministic_run_id(
    dataset_id: str, seed: int, model_checkpoint: str | None, parts: Sequence[str]
) -> str:
    """Return a stable run id derived from the evaluation's inputs."""
    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "seed": int(seed),
            "model_checkpoint": model_checkpoint,
            "parts": sorted(parts),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"eval-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def run_evaluation(
    dataset: EvaluationDataset | None = None,
    *,
    samples: Mapping[str, np.ndarray] | None = None,
    validation_samples: Mapping[str, np.ndarray] | None = None,
    samples_by_horizon: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    model_available: bool = False,
    model_checkpoint: str | None = None,
    seed: int = 20240601,
    run_id: str | None = None,
    uncertainty_nominal: Sequence[float] = (0.8, 0.9),
    downscaling_result: Mapping[str, Any] | None = None,
) -> EvaluationOutcome:
    """Run every Phase 9 measurement and assemble the run record.

    Args:
        dataset: The evaluation dataset, when one was built.
        samples: Test-split GNN samples, for the one-step comparison.
        validation_samples: Validation-split GNN samples, for fitting the
            uncertainty residuals. **Not** used for any test-set quantity.
        samples_by_horizon: Per-horizon samples, for the rollout evaluation.
        model_available: Whether a trained GNN checkpoint exists.
        model_checkpoint: Checkpoint identifier, recorded in the provenance.
        seed: Random seed, recorded and used for the bootstrap.
        run_id: Explicit run id; derived from the inputs when omitted.
        uncertainty_nominal: Nominal coverages to build and check.
        downscaling_result: A pre-computed Phase 7 downscaling comparison.
        explainability_result: A pre-computed explainability summary.

    Returns:
        An :class:`EvaluationOutcome`. Never raises for a data shortage: an
        absent measurement becomes a reason on the run.
    """
    dataset_id = dataset.dataset_id if dataset is not None else "unknown"
    identifier = run_id or _deterministic_run_id(
        dataset_id, seed, model_checkpoint, ["gnn", "uncertainty"]
    )
    provenance = (
        dataset.provenance()
        if dataset is not None
        else Provenance(dataset_id=dataset_id)
    )
    provenance.seed = int(seed)
    provenance.model_checkpoints = {"gnn": model_checkpoint}
    provenance.code_fingerprint = code_fingerprint()
    provenance.environment = _environment_facts()
    provenance.configuration = {
        **provenance.configuration,
        "seed": int(seed),
        "uncertainty_method": BOOTSTRAP_METHOD,
        "uncertainty_nominal": [float(n) for n in uncertainty_nominal],
    }

    run = EvaluationRun(
        run_id=identifier,
        dataset_id=dataset_id,
        status=SUCCESS,
        provenance=provenance,
    )

    try:
        if dataset is not None:
            dataset.assert_no_leakage()
    except LeakageError as exc:
        run.status = FAILED
        run.errors.append(f"leakage: {exc}")
        run.completed_at = utc_now()
        return EvaluationOutcome(run=run, dataset=dataset)

    measured = 0
    unavailable = 0

    # --- GNN baselines ---------------------------------------------------
    if samples is None or not samples:
        reason = (
            "No test-split samples with a known next state were available, so "
            "neither the GNN nor its baselines could be scored. No metric is "
            "reported; a zero would misread as perfect prediction."
        )
        run.add_metric(
            "gnn", MetricResult.unavailable("gnn_position_error_km", reason, unit="km")
        )
        run.warnings.append(reason)
        unavailable += 1
    else:
        try:
            comparison = evaluate_gnn_baselines(
                samples, model_available=model_available, model_checkpoint=model_checkpoint
            )
            run.comparisons["gnn"] = comparison
            for method, metrics in comparison["methods"].items():
                if "latitude_mae_deg" not in metrics:
                    run.add_metric(
                        "gnn",
                        MetricResult.unavailable(
                            f"{method}_position_error_km",
                            str(metrics.get("reason", "not computed")),
                            unit="km",
                        ),
                    )
                    unavailable += 1
                    continue
                for key, unit in GNN_METRIC_UNITS.items():
                    if key in metrics and isinstance(metrics[key], (int, float)):
                        run.add_metric(
                            "gnn",
                            MetricResult.measured(
                                f"{method}_{key}",
                                float(metrics[key]),
                                unit=unit,
                                sample_count=int(metrics.get("sample_count", 0) or 0),
                                method=f"{method} baseline" if method != "gnn" else "gnn inference",
                            ),
                        )
                        measured += 1
            run.warnings.append(summarise_gnn_comparison(comparison))
        except ValueError as exc:
            run.add_metric("gnn", MetricResult.unavailable("gnn_position_error_km", str(exc), unit="km"))
            run.warnings.append(f"GNN evaluation unavailable: {exc}")
            unavailable += 1

    # --- multi-step -----------------------------------------------------
    if samples_by_horizon:
        rollout = evaluate_multistep_rollout(
            samples_by_horizon, model_available=model_available
        )
        run.comparisons["gnn_rollout"] = rollout
        for horizon, entry in rollout["horizons"].items():
            gnn = entry.get("methods", {}).get("gnn", {})
            if "latitude_mae_deg" in gnn:
                run.add_metric(
                    "gnn_rollout",
                    MetricResult.measured(
                        f"horizon_{horizon}_position_error_km_mean",
                        float(gnn["displacement_error_km_mean"]),
                        unit="km",
                        sample_count=int(gnn.get("sample_count", 0) or 0),
                        method=f"{entry['horizon_kind']} rollout",
                    ),
                )
                measured += 1
            else:
                run.add_metric(
                    "gnn_rollout",
                    MetricResult.insufficient(
                        f"horizon_{horizon}_position_error_km_mean",
                        str(entry.get("reason", gnn.get("reason", "not computed"))),
                        unit="km",
                        sample_count=int(entry.get("sample_count", 0) or 0),
                    ),
                )
                unavailable += 1

    # --- uncertainty ----------------------------------------------------
    run.uncertainty = _run_uncertainty(
        run, validation_samples, samples, uncertainty_nominal, seed
    )
    measured += len(run.uncertainty.get("available_metrics", []))

    # --- downscaling ----------------------------------------------------
    if downscaling_result:
        run.comparisons["downscaling"] = dict(downscaling_result)
        for method, metrics in (downscaling_result.get("methods", {}) or {}).items():
            stats = metrics.get("metrics", metrics)
            for key, unit in (("mae", "source-unit"), ("rmse", "source-unit"), ("bias", "source-unit"), ("correlation", "1")):
                value = stats.get(key)
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    run.add_metric(
                        "downscaling",
                        MetricResult.measured(
                            f"{method}_{key}",
                            float(value),
                            unit=unit,
                            sample_count=int(downscaling_result.get("sample_count", 0) or 0),
                            method=f"{method} downscaling",
                        ),
                    )
                    measured += 1
    else:
        reason = (
            "No downscaling evaluation was supplied. Either no high-resolution "
            "reference frames exist for this record or no checkpoint was "
            "trained. The AI result is omitted rather than imitated."
        )
        run.add_metric("downscaling", MetricResult.unavailable("ai_mae", reason))
        unavailable += 1

    # --- status ---------------------------------------------------------
    if measured == 0:
        run.status = RUN_INSUFFICIENT
        run.warnings.append(
            "No metric could be measured on this record. Every quantity is "
            "reported as unavailable; none is reported as zero."
        )
    elif unavailable or run.warnings:
        run.status = PARTIAL
    run.completed_at = utc_now()
    return EvaluationOutcome(run=run, dataset=dataset)


def _run_uncertainty(
    run: EvaluationRun,
    validation_samples: Mapping[str, np.ndarray] | None,
    test_samples: Mapping[str, np.ndarray] | None,
    nominal_levels: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    """Fit, report and calibrate the uncertainty layer."""
    from extreme_weather.evaluation.uncertainty import MIN_RESIDUALS_FOR_INTERVALS

    result: dict[str, Any] = {
        "method": BOOTSTRAP_METHOD,
        "nominal_levels": [float(n) for n in nominal_levels],
        "intervals": {},
        "calibration": {},
        "available_metrics": [],
        "caveat": (
            "These are empirical validation-residual quantiles, not "
            "probabilities. No calibrated-posterior claim is made."
        ),
    }

    needed = (
        "true_latitude",
        "true_longitude",
        "true_intensity",
        "predicted_latitude",
        "predicted_longitude",
        "predicted_intensity",
    )
    if not validation_samples or any(name not in validation_samples for name in needed):
        reason = (
            "No validation-split predictions were available, so residual "
            "quantiles could not be fitted and no interval is reported. An "
            "interval from a different split would be a different - and "
            "unvalidated - quantity."
        )
        result["intervals"] = {"status": NOT_AVAILABLE, "reason": reason}
        run.add_metric("uncertainty", MetricResult.unavailable("interval_width", reason))
        return result

    counts = {
        name: int(np.isfinite(np.asarray(validation_samples[name], dtype="float64")).sum())
        for name in needed
    }
    bootstrap: ResidualBootstrap = residual_intervals(
        validation_true_latitude=validation_samples["true_latitude"],
        validation_predicted_latitude=validation_samples["predicted_latitude"],
        validation_true_longitude=validation_samples["true_longitude"],
        validation_predicted_longitude=validation_samples["predicted_longitude"],
        validation_true_intensity=validation_samples["true_intensity"],
        validation_predicted_intensity=validation_samples["predicted_intensity"],
        seed=int(seed),
    )
    result["residual_distribution"] = bootstrap.as_dict()

    smallest = min(counts.values())
    if smallest < MIN_RESIDUALS_FOR_INTERVALS:
        reason = (
            f"Interval quantiles need at least {MIN_RESIDUALS_FOR_INTERVALS} "
            f"validation residuals; only {smallest} were available. The "
            "quantiles of so few residuals are dominated by individual samples, "
            "so no interval is reported."
        )
        result["intervals"] = {"status": INSUFFICIENT_DATA, "reason": reason}
        run.add_metric(
            "uncertainty",
            MetricResult.insufficient("interval_width", reason, sample_count=smallest),
        )
        return result

    # Report a representative interval around the median test prediction, when
    # test predictions exist, so a concrete example is available.
    if test_samples and all(
        name in test_samples and np.asarray(test_samples[name]).size
        for name in needed
    ):
        centre = {
            "latitude": float(np.median(np.asarray(test_samples["predicted_latitude"], dtype="float64"))),
            "longitude": float(np.median(np.asarray(test_samples["predicted_longitude"], dtype="float64"))),
            "intensity": float(np.median(np.asarray(test_samples["predicted_intensity"], dtype="float64"))),
        }
    else:
        centre = {"latitude": 0.0, "longitude": 0.0, "intensity": 0.0}

    for nominal in nominal_levels:
        uncertainty = bootstrap.interval_for(
            centre["latitude"],
            centre["longitude"],
            centre["intensity"],
            nominal=float(nominal),
        )
        result["intervals"][f"{float(nominal):.2f}"] = uncertainty.as_dict()
        for name, interval in (
            ("latitude", uncertainty.latitude_interval),
            ("longitude", uncertainty.longitude_interval),
            ("intensity", uncertainty.intensity_interval),
        ):
            if interval is not None:
                metric = MetricResult.derived(
                    f"interval_width_{name}_{float(nominal):.2f}",
                    float(interval[1] - interval[0]),
                    unit=GNN_METRIC_UNITS.get(f"{name}_mae_deg", "source-unit"),
                    reason=(
                        "Width of an empirical validation-residual quantile "
                        "interval; it is not a confidence interval until coverage "
                        "is measured below."
                    ),
                    sample_count=smallest,
                    method=BOOTSTRAP_METHOD,
                )
                run.add_metric("uncertainty", metric)
                result["available_metrics"].append(metric.name)

    # --- calibration, on the test split, once ---------------------------
    if test_samples and all(name in test_samples for name in needed):
        for nominal in nominal_levels:
            calibration = evaluate_coverage(
                bootstrap,
                test_true_latitude=test_samples["true_latitude"],
                test_predicted_latitude=test_samples["predicted_latitude"],
                test_true_longitude=test_samples["true_longitude"],
                test_predicted_longitude=test_samples["predicted_longitude"],
                test_true_intensity=test_samples["true_intensity"],
                test_predicted_intensity=test_samples["predicted_intensity"],
                nominal=float(nominal),
            )
            result["calibration"][f"{float(nominal):.2f}"] = calibration.as_dict()
            for metric in calibration.metrics():
                run.add_metric("uncertainty", metric)
                if metric.available:
                    result["available_metrics"].append(metric.name)
    else:
        result["calibration"] = {
            "status": NOT_AVAILABLE,
            "reason": (
                "No test-split predictions were available, so interval coverage "
                "could not be measured. The intervals below are therefore "
                "uncalibrated: their nominal level is a construction parameter, "
                "not a verified probability."
            ),
        }

    return result
