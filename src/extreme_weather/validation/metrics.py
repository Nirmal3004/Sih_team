"""Evaluation metrics and honest baseline comparison (SIH26078, Phase 7).

Every function here is written under one constraint: **a metric is either
computed from real data or explicitly not computed.** There is no code path that
returns a plausible number it cannot justify.

What that means concretely:

* **Descriptive diagnostics** - event counts, durations, intensities, track
  speeds, severity distributions - are always computable and are always
  reported, because they describe what the system *did*.
* **Skill metrics** - POD, FAR, CSI, precision, recall - are only computed
  against a reference catalogue with stated provenance. Without one, they are
  reported as the string ``"Ground truth unavailable - metric not computed."``
  and never as ``0.0``, which would read as "zero skill" - a different and
  wrong claim.
* **Baseline comparisons** are mandatory. The GNN is measured against
  persistence, the downscaler against nearest and cubic, the calibration
  against the prototype thresholds. A method that loses is recorded as losing.

The comparison convention
-------------------------
``improves_on_baseline`` is ``True`` only when the candidate error is
*strictly* below the baseline error. ``"tie"`` and ``"worse"`` are equally
valid outcomes and appear verbatim in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from extreme_weather.ingestion.base import utc_now
from extreme_weather.validation.ground_truth import (
    GROUND_TRUTH_UNAVAILABLE,
    GroundTruth,
    NOT_COMPUTED,
    haversine_km,
    match_events,
)

__all__ = [
    "NO_GROUND_TRUTH",
    "AlertDiagnostics",
    "AnomalyDiagnostics",
    "DetectionSkill",
    "DownscalingComparison",
    "GnnComparison",
    "TrackingDiagnostics",
    "compare_metrics",
    "compute_alert_diagnostics",
    "compute_anomaly_diagnostics",
    "compute_downscaling_comparison",
    "compute_gnn_comparison",
    "compute_tracking_diagnostics",
    "evaluate_detection_skill",
    "persistence_baseline",
    "summarise",
]


#: Re-exported so callers need not reach into the ground-truth module.
NO_GROUND_TRUTH = GROUND_TRUTH_UNAVAILABLE


def summarise(values: Sequence[float]) -> dict[str, float] | None:
    """Return count/min/mean/median/max/std for a series, or ``None`` if empty.

    ``std`` is reported explicitly even when it is 0.0, because "every track had
    the same speed" is a result and omitting the field would hide it.
    """
    array = np.asarray(list(values), dtype="float64")
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def compare_metrics(
    candidate: float | None,
    baseline: float | None,
    *,
    lower_is_better: bool = True,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare one error metric against a baseline and state the verdict.

    Args:
        candidate: The method's error. Lower is better by default.
        baseline: The baseline's error.
        lower_is_better: Set False when a higher value is better (correlation).
        tolerance: Absolute difference below which the two are called a tie.

    Returns:
        ``{"candidate", "baseline", "verdict", "improves_on_baseline",
        "relative_change"}`` where ``verdict`` is ``"improves"``, ``"tie"`` or
        ``"worse"``. When either value is missing, the verdict is
        ``"not_computed"`` and the reason is given.
    """
    result: dict[str, Any] = {
        "candidate": candidate,
        "baseline": baseline,
        "lower_is_better": bool(lower_is_better),
        "improves_on_baseline": None,
        "relative_change": None,
    }
    if candidate is None or baseline is None:
        result["verdict"] = "not_computed"
        result["reason"] = NOT_COMPUTED
        return result

    difference = float(candidate) - float(baseline)
    absolute = abs(difference)
    if absolute <= tolerance:
        result["verdict"] = "tie"
    elif (difference < 0) == lower_is_better:
        result["verdict"] = "improves"
    else:
        result["verdict"] = "worse"
    result["improves_on_baseline"] = result["verdict"] == "improves"
    if baseline != 0.0:
        result["relative_change"] = float(difference / abs(float(baseline)))
    return result


@dataclass(frozen=True)
class DetectionSkill:
    """Contingency-table skill metrics, or an explicit refusal.

    Every field is ``None`` and ``note`` carries the reason when no reference
    data exists. Zeros are never substituted: ``POD = 0.0`` says "the method
    found nothing", which is a completely different claim from "the method
    cannot be evaluated".
    """

    available: bool
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    pod: float | None = None
    far: float | None = None
    csi: float | None = None
    bias: float | None = None
    note: str = NOT_COMPUTED
    ground_truth_status: str = ""
    matching_criteria: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "available": self.available,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "POD": self.pod,
            "FAR": self.far,
            "CSI": self.csi,
            "bias": self.bias,
            "note": self.note,
            "ground_truth_status": self.ground_truth_status,
            "matching_criteria": self.matching_criteria,
        }


def evaluate_detection_skill(
    detected: Sequence[Mapping[str, Any]],
    truth: GroundTruth,
) -> DetectionSkill:
    """Compute POD / FAR / CSI when - and only when - ground truth exists.

    * ``POD = TP / (TP + FN)`` - the share of reference events that were found.
    * ``FAR = FP / (TP + FP)`` - the share of detections that were spurious.
    * ``CSI = TP / (TP + FP + FN)`` - the critical success index.
    * ``bias = (TP + FP) / (TP + FN)`` - 1.0 is unbiased, >1 over-detects.

    A division whose denominator is zero returns ``None`` rather than 0.0: "no
    reference events were found" is not "the method has zero skill".

    Args:
        detected: The detected events.
        truth: The reference catalogue.

    Returns:
        A :class:`DetectionSkill`, available or explicitly not.
    """
    counts = match_events(detected, truth)
    if not truth.available:
        return DetectionSkill(
            available=False,
            note=NOT_COMPUTED,
            ground_truth_status=truth.status,
        )

    true_positives = int(counts["true_positives"] or 0)
    false_positives = int(counts["false_positives"] or 0)
    false_negatives = int(counts["false_negatives"] or 0)

    observed = true_positives + false_negatives
    predicted = true_positives + false_positives
    # CSI = TP / (TP + FP + FN). Summing 'observed + predicted' would count the
    # true positives twice and halve the index for a perfect forecast, so the
    # denominator is built explicitly from the three counts.
    total = true_positives + false_positives + false_negatives

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    return DetectionSkill(
        available=True,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        pod=ratio(true_positives, observed),
        far=ratio(false_positives, predicted),
        csi=ratio(true_positives, total),
        bias=ratio(predicted, observed),
        note=None,  # type: ignore[arg-type]
        ground_truth_status=truth.status,
        matching_criteria=truth.matching_criteria,
    )



@dataclass(frozen=True)
class AnomalyDiagnostics:
    """Descriptive statistics of the events the detector produced.

    These describe **what the system did**, not whether it was right. They are
    always computable, and are always paired with the skill block - which says
    plainly whether correctness could be measured at all.
    """

    split: str
    dataset_id: str
    event_count: int
    event_types: dict[str, int]
    peak_intensity: dict[str, float]
    mean_intensity: dict[str, float]
    duration_hours: dict[str, Any]
    spatial_extent_km: dict[str, Any]
    monthly_distribution: dict[str, int]
    variable_distribution: dict[str, int]
    skill: DetectionSkill
    thresholds: dict[str, Any] = field(default_factory=dict)
    method: str = "prototype thresholds (Phase 1)"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "split": self.split,
            "dataset_id": self.dataset_id,
            "method": self.method,
            "event_count": self.event_count,
            "event_types": dict(self.event_types),
            "peak_intensity": dict(self.peak_intensity),
            "mean_intensity": dict(self.mean_intensity),
            "duration_hours": dict(self.duration_hours),
            "spatial_extent_km": dict(self.spatial_extent_km),
            "monthly_distribution": dict(self.monthly_distribution),
            "variable_distribution": dict(self.variable_distribution),
            "thresholds": dict(self.thresholds),
            "detection_skill": self.skill.as_dict(),
            "notes": list(self.notes),
            "caveat": (
                "Counts and intensities describe what the detector produced. They "
                "are not accuracy, and without a reference catalogue they cannot "
                "be converted into skill."
            ),
        }


def _iso(value: Any) -> str:
    """Format a timestamp as an ISO 8601 string (minutes are enough here)."""
    return str(np.datetime_as_string(np.asarray(value), unit="m"))


def compute_anomaly_diagnostics(
    events: Sequence[Mapping[str, Any]],
    *,
    split: str,
    dataset_id: str,
    truth: GroundTruth | None = None,
    thresholds: Mapping[str, Any] | None = None,
    method: str = "prototype thresholds (Phase 1)",
) -> AnomalyDiagnostics:
    """Summarise detected events without claiming they were correct.

    Args:
        events: The detected events (Phase 1 model objects or mappings).
        split: Which split the events came from, recorded in the result.
        dataset_id: Identifier recorded in the result.
        truth: Optional reference catalogue. When absent, the skill block says
            so instead of reporting a number.
        thresholds: The threshold configuration that produced the events.
        method: Human-readable description of the threshold method.

    Returns:
        An :class:`AnomalyDiagnostics`.
    """
    records: list[Mapping[str, Any]] = []
    for event in events:
        # Phase 1 returns model objects; accept plain mappings too so the
        # function is usable from a stored events.json.
        if isinstance(event, Mapping):
            records.append(event)
            continue
        records.append(
            {
                "event_id": getattr(event, "event_id", ""),
                "time": getattr(event, "timestamp", None),
                "latitude": getattr(event, "latitude", None),
                "longitude": getattr(event, "longitude", None),
                "event_type": str(getattr(event, "event_type", "unknown")),
                "intensity": getattr(event, "intensity", None),
                "variable": getattr(event, "variable", None),
            }
        )
    reference = truth or GroundTruth()
    skill = evaluate_detection_skill(records, reference)

    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for event in records:
        by_type.setdefault(str(getattr(event, "event_type", event.get("event_type", "unknown"))), []).append(event)

    def field_of(event: Mapping[str, Any], name: str, default: Any = None) -> Any:
        return getattr(event, name, event.get(name, default))

    # A Phase 1 event is an *instant* (one grid slice), so it has no duration of
    # its own. Durations are reported by the tracker instead; here the gap
    # between consecutive events is recorded, which is a real, if different,
    # quantity and is labelled as such.
    stamps = sorted(
        np.datetime64(field_of(event, "timestamp"))
        for event in records
        if field_of(event, "timestamp") is not None
    )
    durations = [
        float((later - earlier) / np.timedelta64(1, "h"))
        for earlier, later in zip(stamps, stamps[1:])
    ]

    # Spatial extent: the largest great-circle distance between any two events,
    # which says how widely the detections were scattered.
    extent = 0.0
    points = [
        (float(field_of(e, "latitude", 0.0)), float(field_of(e, "longitude", 0.0)))
        for e in records
    ]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            extent = max(extent, haversine_km(*points[i], *points[j]))

    months: dict[str, int] = {}
    variables: dict[str, int] = {}
    for event in records:
        start = field_of(event, "timestamp")
        if start is not None:
            key = str(np.datetime_as_string(np.datetime64(start).astype("datetime64[M]")))
            months[key] = months.get(key, 0) + 1
        variable = str(field_of(event, "variable", "") or "unknown")
        variables[variable] = variables.get(variable, 0) + 1

    notes = (
        f"{skill.note}" if not skill.available else "Skill computed against a reference catalogue."
    )
    return AnomalyDiagnostics(
        split=split,
        dataset_id=dataset_id,
        event_count=len(records),
        event_types={name: len(items) for name, items in by_type.items()},
        peak_intensity={
            name: float(
                max(float(field_of(e, "intensity", 0.0) or 0.0) for e in items)
            )
            for name, items in by_type.items()
        },
        mean_intensity={
            name: float(
                np.mean([float(field_of(e, "intensity", 0.0) or 0.0) for e in items])
            )
            for name, items in by_type.items()
        },
        duration_hours=summarise(durations) or {"count": 0},
        spatial_extent_km={
            "maximum_pairwise_km": round(extent, 3),
            "note": "Largest great-circle distance between any two detected events.",
        },
        monthly_distribution=months,
        variable_distribution=variables,
        skill=skill,
        thresholds=dict(thresholds or {}),
        method=method,
        notes=(notes,),
    )



@dataclass(frozen=True)
class TrackingDiagnostics:
    """Descriptive statistics of the tracks the tracker produced.

    Explicitly **not** tracking accuracy: there are no reference trajectories
    in this project, so "the tracker produced 2 tracks with mean speed X" is
    what can honestly be said, and no more.
    """

    split: str
    dataset_id: str
    track_count: int
    event_count: int
    matched_events: int
    unmatched_events: int
    duration_hours: dict[str, Any]
    displacement_km: dict[str, Any]
    speed_kmh: dict[str, Any]
    direction_distribution: dict[str, int]
    intensity_evolution: dict[str, Any]
    continuity: dict[str, Any]
    status_distribution: dict[str, int]
    ground_truth_available: bool
    note: str = NOT_COMPUTED
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "split": self.split,
            "dataset_id": self.dataset_id,
            "track_count": self.track_count,
            "event_count": self.event_count,
            "matched_events": self.matched_events,
            "unmatched_events": self.unmatched_events,
            "duration_hours": dict(self.duration_hours),
            "displacement_km": dict(self.displacement_km),
            "speed_kmh": dict(self.speed_kmh),
            "direction_distribution": dict(self.direction_distribution),
            "intensity_evolution": dict(self.intensity_evolution),
            "continuity": dict(self.continuity),
            "status_distribution": dict(self.status_distribution),
            "ground_truth_available": self.ground_truth_available,
            "note": self.note,
            "caveat": (
                "These are descriptive statistics of the produced tracks, NOT "
                "tracking accuracy. No reference trajectories exist for this "
                "dataset, so correctness could not be measured."
            ),
            "warnings": list(self.warnings),
        }



def compute_tracking_diagnostics(
    tracks: Sequence[Any],
    *,
    split: str,
    dataset_id: str,
    event_count: int | None = None,
    truth: GroundTruth | None = None,
) -> TrackingDiagnostics:
    """Summarise tracks descriptively, without claiming they are correct.

    Args:
        tracks: The tracked events (Phase 2 model objects).
        split: Which split the tracks came from.
        dataset_id: Identifier recorded in the result.
        event_count: Total events detected, for the matched/unmatched split.
        truth: Optional reference catalogue. Its presence only toggles the
            ground-truth flag: this project computes no tracking skill even when
            one exists, because the Phase 2 tracker has no reference
            trajectories to score against.

    Returns:
        A :class:`TrackingDiagnostics`.
    """
    records = list(tracks)
    reference = truth or GroundTruth()

    durations: list[float] = []
    displacements: list[float] = []
    speeds: list[float] = []
    directions: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    point_counts: list[int] = []
    first_intensities: list[float] = []
    last_intensities: list[float] = []

    for track in records:
        duration = getattr(track, "duration_hours", None)
        if duration is not None:
            durations.append(float(duration))
        displacement = getattr(track, "displacement_km", None)
        if displacement is not None:
            displacements.append(float(displacement))
        speed = getattr(track, "max_speed_kmh", None)
        if speed is not None:
            speeds.append(float(speed))

        direction = str(getattr(track, "direction", "unknown"))
        directions[direction] = directions.get(direction, 0) + 1
        status = str(getattr(track, "status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

        points = list(getattr(track, "points", []) or [])
        point_counts.append(len(points))
        if points:
            first_intensities.append(float(getattr(points[0], "intensity", 0.0) or 0.0))
            last_intensities.append(float(getattr(points[-1], "intensity", 0.0) or 0.0))

    matched = sum(point_counts)
    total_events = int(event_count) if event_count is not None else matched
    singleton = sum(1 for count in point_counts if count <= 1)

    evolution: dict[str, Any] = {"count": len(records)}
    if first_intensities and last_intensities:
        deltas = [b - a for a, b in zip(first_intensities, last_intensities)]
        evolution = dict(summarise(deltas) or evolution)
        evolution["description"] = "last intensity minus first intensity, per track"
        evolution["strengthening"] = sum(1 for delta in deltas if delta > 0)
        evolution["weakening"] = sum(1 for delta in deltas if delta < 0)

    warnings: list[str] = []
    if singleton:
        warnings.append(
            f"{singleton} of {len(records)} track(s) have a single point, so no "
            "movement could be measured for them."
        )
    if not records:
        warnings.append("No tracks were produced, so every distribution is empty.")

    return TrackingDiagnostics(
        split=split,
        dataset_id=dataset_id,
        track_count=len(records),
        event_count=total_events,
        matched_events=matched,
        unmatched_events=max(0, total_events - matched),
        duration_hours=summarise(durations) or {"count": 0},
        displacement_km=summarise(displacements) or {"count": 0},
        speed_kmh=summarise(speeds) or {"count": 0},
        direction_distribution=directions,
        intensity_evolution=evolution,
        continuity={
            "points_per_track": summarise(point_counts) or {"count": 0},
            "single_point_tracks": singleton,
            "multi_point_tracks": len(records) - singleton,
        },
        status_distribution=status_counts,
        ground_truth_available=reference.available,
        note=NOT_COMPUTED,
        warnings=tuple(warnings),
    )



def persistence_baseline(
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    current_intensity: np.ndarray,
    future_lat: np.ndarray,
    future_lon: np.ndarray,
    future_intensity: np.ndarray,
) -> dict[str, float | int]:
    """Score the "nothing moves" forecast: the persistence baseline.

    The baseline for any event-forecasting model is ``predicted_next =
    current_state``. A GNN that cannot beat this on a test set is adding noise,
    not signal, and the report must be able to say so.

    Args:
        current_lat: Current latitudes, degrees.
        current_lon: Current longitudes, degrees.
        current_intensity: Current intensities.
        future_lat: Actual next-step latitudes.
        future_lon: Actual next-step longitudes.
        future_intensity: Actual next-step intensities.

    Returns:
        The same error keys the GNN is scored on, so the two are directly
        comparable.

    Raises:
        ValueError: If the arrays are not the same length or are empty.
    """
    lat0 = np.asarray(current_lat, dtype="float64")
    lon0 = np.asarray(current_lon, dtype="float64")
    int0 = np.asarray(current_intensity, dtype="float64")
    lat1 = np.asarray(future_lat, dtype="float64")
    lon1 = np.asarray(future_lon, dtype="float64")
    int1 = np.asarray(future_intensity, dtype="float64")

    if not (
        lat0.shape == lon0.shape == int0.shape == lat1.shape == lon1.shape == int1.shape
    ):
        raise ValueError(
            "Persistence baseline inputs must all have the same shape; got "
            f"{lat0.shape}, {lon0.shape}, {int0.shape}, {lat1.shape}, {lon1.shape}, "
            f"{int1.shape}."
        )
    if lat0.size == 0:
        raise ValueError("Persistence baseline needs at least one sample.")

    displacement = np.array(
        [haversine_km(a0, b0, a1, b1) for a0, b0, a1, b1 in zip(lat0, lon0, lat1, lon1)],
        dtype="float64",
    )
    return {
        "sample_count": int(lat0.size),
        "latitude_mae_deg": float(np.mean(np.abs(lat1 - lat0))),
        "longitude_mae_deg": float(np.mean(np.abs(lon1 - lon0))),
        "intensity_mae": float(np.mean(np.abs(int1 - int0))),
        "latitude_rmse_deg": float(np.sqrt(np.mean((lat1 - lat0) ** 2))),
        "longitude_rmse_deg": float(np.sqrt(np.mean((lon1 - lon0) ** 2))),
        "intensity_rmse": float(np.sqrt(np.mean((int1 - int0) ** 2))),
        "displacement_error_km_mean": float(np.mean(displacement)),
        "displacement_error_km_median": float(np.median(displacement)),
    }



@dataclass(frozen=True)
class GnnComparison:
    """A GNN's test-set error, beside the persistence baseline's.

    The ``comparison`` block states the verdict per metric. A model that loses
    is reported as losing - there is no averaging-away of a bad result.
    """

    dataset: str
    split: str
    sample_counts: dict[str, int]
    model_checkpoint: str | None
    metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    comparison: dict[str, Any]
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "sample_counts": dict(self.sample_counts),
            "model_checkpoint": self.model_checkpoint,
            "metrics": dict(self.metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "comparison": dict(self.comparison),
            "limitations": list(self.limitations),
        }


_GNN_ERROR_KEYS = (
    "latitude_mae_deg",
    "longitude_mae_deg",
    "intensity_mae",
    "latitude_rmse_deg",
    "longitude_rmse_deg",
    "intensity_rmse",
    "displacement_error_km_mean",
)


def compute_gnn_comparison(
    predicted_lat: np.ndarray,
    predicted_lon: np.ndarray,
    predicted_intensity: np.ndarray,
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    current_intensity: np.ndarray,
    future_lat: np.ndarray,
    future_lon: np.ndarray,
    future_intensity: np.ndarray,
    *,
    dataset: str,
    split: str,
    model_checkpoint: str | None,
    sample_counts: dict[str, int] | None = None,
    limitations: Sequence[str] = (),
) -> GnnComparison:
    """Score a GNN against the persistence baseline on the test split.

    Both methods are scored on **identical samples** - the same current and
    future state arrays - so any difference in the numbers is the model, not
    the sample.

    Args:
        predicted_lat: Model-predicted next latitudes.
        predicted_lon: Model-predicted next longitudes.
        predicted_intensity: Model-predicted next intensities.
        current_lat: Current-state latitudes (the persistence forecast).
        current_lon: Current-state longitudes.
        current_intensity: Current-state intensities.
        future_lat: Actual next-step latitudes.
        future_lon: Actual next-step longitudes.
        future_intensity: Actual next-step intensities.
        dataset: Dataset identifier, recorded in the result.
        split: Which split was scored, recorded in the result.
        model_checkpoint: Path or identifier of the checkpoint used.
        sample_counts: Extra counts to record (train/validation sizes).
        limitations: Limitations to record verbatim.

    Returns:
        A :class:`GnnComparison` with per-metric verdicts.

    Raises:
        ValueError: If no test samples were supplied.
    """
    pred_lat = np.asarray(predicted_lat, dtype="float64")
    pred_lon = np.asarray(predicted_lon, dtype="float64")
    pred_int = np.asarray(predicted_intensity, dtype="float64")
    true_lat = np.asarray(future_lat, dtype="float64")
    true_lon = np.asarray(future_lon, dtype="float64")
    true_int = np.asarray(future_intensity, dtype="float64")

    if pred_lat.size == 0:
        raise ValueError("GNN evaluation needs at least one test sample.")

    baseline = persistence_baseline(
        current_lat, current_lon, current_intensity, true_lat, true_lon, true_int
    )
    displacement = np.array(
        [haversine_km(p, q, a, b) for p, q, a, b in zip(pred_lat, pred_lon, true_lat, true_lon)],
        dtype="float64",
    )
    metrics = {
        "sample_count": int(pred_lat.size),
        "latitude_mae_deg": float(np.mean(np.abs(true_lat - pred_lat))),
        "longitude_mae_deg": float(np.mean(np.abs(true_lon - pred_lon))),
        "intensity_mae": float(np.mean(np.abs(true_int - pred_int))),
        "latitude_rmse_deg": float(np.sqrt(np.mean((true_lat - pred_lat) ** 2))),
        "longitude_rmse_deg": float(np.sqrt(np.mean((true_lon - pred_lon) ** 2))),
        "intensity_rmse": float(np.sqrt(np.mean((true_int - pred_int) ** 2))),
        "displacement_error_km_mean": float(np.mean(displacement)),
        "displacement_error_km_median": float(np.median(displacement)),
    }

    comparison = {
        key: compare_metrics(metrics.get(key), baseline.get(key))
        for key in _GNN_ERROR_KEYS
    }
    verdicts = [entry["verdict"] for entry in comparison.values()]
    improved = sum(1 for verdict in verdicts if verdict == "improves")
    worsened = sum(1 for verdict in verdicts if verdict == "worse")
    comparison["_summary"] = {
        "metrics_improved": improved,
        "metrics_worsened": worsened,
        "metrics_tied": sum(1 for verdict in verdicts if verdict == "tie"),
        "verdict": (
            "beats persistence"
            if improved > worsened
            else ("matches persistence" if improved == worsened else "worse than persistence")
        ),
    }

    return GnnComparison(
        dataset=dataset,
        split=split,
        sample_counts=dict(sample_counts or {}),
        model_checkpoint=model_checkpoint,
        metrics=metrics,
        baseline_metrics=baseline,
        comparison=comparison,
        limitations=tuple(limitations),
    )



@dataclass(frozen=True)
class DownscalingComparison:
    """Nearest vs cubic vs AI, scored on identical test frames.

    POD/FAR are present only when an event definition is supplied, because a
    POD computed against an arbitrary threshold would be a number with no
    meaning. When none is supplied they carry the explicit "not computed"
    sentence rather than a fabricated contingency table.
    """

    dataset: str
    split: str
    frame_count: int
    variable: str
    methods: dict[str, dict[str, Any]]
    comparison: dict[str, Any]
    classification: dict[str, Any]
    peak_preservation: dict[str, Any]
    structural: dict[str, Any]
    limitations: tuple[str, ...] = ()

    @property
    def winner(self) -> str | None:
        """The method with the lowest MAE, or ``None`` when undetermined."""
        scored = {
            name: stats.get("mae")
            for name, stats in self.methods.items()
            if stats.get("mae") is not None
        }
        if not scored:
            return None
        return min(scored, key=lambda name: scored[name])  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "frame_count": self.frame_count,
            "variable": self.variable,
            "methods": {name: dict(stats) for name, stats in self.methods.items()},
            "comparison": dict(self.comparison),
            "classification": dict(self.classification),
            "peak_preservation": dict(self.peak_preservation),
            "structural": dict(self.structural),
            "winner_by_mae": self.winner,
            "limitations": list(self.limitations),
        }


def _field_metrics(
    prediction: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Return MAE / RMSE / bias / correlation for one prediction."""
    pred = np.asarray(prediction, dtype="float64").ravel()
    ref = np.asarray(reference, dtype="float64").ravel()
    finite = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[finite], ref[finite]
    if pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "correlation": float("nan")}

    error = pred - ref
    correlation: float = float("nan")
    if pred.size > 1 and np.std(pred) > 0 and np.std(ref) > 0:
        correlation = float(np.corrcoef(pred, ref)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "correlation": correlation,
    }


def _extreme_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    """POD / FAR / CSI against an explicitly supplied event threshold."""
    pred = (np.asarray(prediction, dtype="float64") >= threshold).ravel()
    ref = (np.asarray(reference, dtype="float64") >= threshold).ravel()
    hits = int(np.count_nonzero(pred & ref))
    misses = int(np.count_nonzero(~pred & ref))
    false_alarms = int(np.count_nonzero(pred & ~ref))
    observed = hits + misses
    predicted = hits + false_alarms
    return {
        "threshold": float(threshold),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "POD": float(hits / observed) if observed else None,
        "FAR": float(false_alarms / predicted) if predicted else None,
        "CSI": float(hits / (observed + predicted)) if (observed + predicted) else None,
    }



def compute_downscaling_comparison(
    predictions: Mapping[str, np.ndarray],
    reference: np.ndarray,
    *,
    dataset: str,
    split: str,
    variable: str = "precipitation",
    event_threshold: float | None = None,
    ai_method: str = "ai",
    limitations: Sequence[str] = (),
) -> DownscalingComparison:
    """Compare every downscaling method on identical test frames.

    The AI is compared against **every** baseline present, and cubic is always
    included: the Phase 4 prototype often loses to it, and a loss is reported
    as a loss.

    Args:
        predictions: ``{method_name: predicted_field}`` over the same frames.
        reference: The high-resolution truth for those frames.
        dataset: Dataset identifier, recorded in the result.
        split: Which split was scored.
        variable: The variable being downscaled.
        event_threshold: Value at which an "event" begins. **Required** for
            POD/FAR; omitted means those are reported as not computed.
        ai_method: Key of the AI method inside ``predictions``.
        limitations: Limitations to record verbatim.

    Returns:
        A :class:`DownscalingComparison`.

    Raises:
        ValueError: If no methods, no reference frames, or a shape mismatch.
    """
    if not predictions:
        raise ValueError("Downscaling comparison needs at least one prediction.")
    truth = np.asarray(reference, dtype="float64")
    if truth.size == 0:
        raise ValueError("Downscaling comparison needs reference frames.")

    methods: dict[str, dict[str, Any]] = {}
    for name, field in predictions.items():
        array = np.asarray(field, dtype="float64")
        if array.shape != truth.shape:
            raise ValueError(
                f"Method '{name}' produced shape {array.shape} but the reference is "
                f"{truth.shape}. All methods must be scored on identical frames."
            )
        stats = _field_metrics(array, truth)
        stats["shape"] = list(array.shape)
        if event_threshold is not None:
            stats["extremes"] = _extreme_metrics(array, truth, event_threshold)
        methods[name] = stats

    comparison: dict[str, Any] = {}
    if ai_method in methods:
        for name, stats in methods.items():
            if name == ai_method:
                continue
            comparison[f"ai_vs_{name}"] = {
                metric: compare_metrics(
                    methods[ai_method].get(metric), stats.get(metric)
                )
                for metric in ("mae", "rmse", "bias", "correlation")
            }
        verdicts = [
            entry["verdict"]
            for block in comparison.values()
            for key, entry in block.items()
            if key in ("mae", "rmse")
        ]
        comparison["_summary"] = {
            "ai_method": ai_method,
            "baselines": [name for name in methods if name != ai_method],
            "metrics_improved": sum(1 for v in verdicts if v == "improves"),
            "metrics_worsened": sum(1 for v in verdicts if v == "worse"),
            "verdict": (
                "AI improves on every baseline"
                if verdicts and all(v == "improves" for v in verdicts)
                else (
                    "AI does not beat every baseline"
                    if any(v != "improves" for v in verdicts)
                    else NOT_COMPUTED
                )
            ),
        }


    classification: dict[str, Any] = {
        "available": event_threshold is not None,
        "note": (
            None
            if event_threshold is not None
            else "POD/FAR not computed - no defensible event threshold was supplied."
        ),
        "event_threshold": event_threshold,
        "per_method": {
            name: stats.get("extremes")
            for name, stats in methods.items()
            if stats.get("extremes") is not None
        },
    }

    flat_truth = truth.ravel()
    peak_preservation: dict[str, Any] = {
        "reference_peak": float(np.nanmax(flat_truth)) if flat_truth.size else None,
        "per_method": {},
    }
    for name, field in predictions.items():
        array = np.asarray(field, dtype="float64")
        if array.size and np.isfinite(array).any():
            reference_peak = float(np.nanmax(flat_truth))
            peak_preservation["per_method"][name] = {
                "peak_value": float(np.nanmax(array)),
                "peak_retention": (
                    float(np.nanmax(array) / reference_peak)
                    if reference_peak != 0
                    else None
                ),
            }

    structural: dict[str, Any] = {"per_method": {}}
    for name, field in predictions.items():
        array = np.asarray(field, dtype="float64").ravel()
        finite = np.isfinite(array) & np.isfinite(flat_truth)
        values, truth_values = array[finite], flat_truth[finite]
        correlation = (
            float(np.corrcoef(values, truth_values)[0, 1])
            if values.size > 1
            and np.std(values) > 0
            and np.std(truth_values) > 0
            else None
        )
        structural["per_method"][name] = {
            "structural_correlation": correlation,
            "wet_area_ratio": float(np.mean(values > 0.0)) if values.size else None,
            "reference_wet_area_ratio": (
                float(np.mean(truth_values > 0.0)) if truth_values.size else None
            ),
        }
    structural["note"] = (
        "Structural correlation is the Pearson correlation of the predicted and "
        "reference fields - a shape measure independent of scale."
    )

    return DownscalingComparison(
        dataset=dataset,
        split=split,
        frame_count=int(truth.size),
        variable=variable,
        methods=methods,
        comparison=comparison,
        classification=classification,
        peak_preservation=peak_preservation,
        structural=structural,
        limitations=tuple(limitations),
    )



@dataclass(frozen=True)
class AlertDiagnostics:
    """Descriptive statistics of the alerts the Phase 5 rules produced.

    The severity system is **not** meteorologically validated and this object
    does not pretend otherwise. It records how many alerts fired, at what
    severities, and which components contributed - a statement about the rule
    system, not about the weather.
    """

    split: str
    dataset_id: str
    alert_count: int
    severity_distribution: dict[str, int]
    status_distribution: dict[str, int]
    risk_score: dict[str, Any]
    duration_hours: dict[str, Any]
    trigger_components: dict[str, int]
    component_contribution: dict[str, dict[str, Any]]
    ground_truth_available: bool
    skill: DetectionSkill
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "split": self.split,
            "dataset_id": self.dataset_id,
            "alert_count": self.alert_count,
            "severity_distribution": dict(self.severity_distribution),
            "status_distribution": dict(self.status_distribution),
            "risk_score": dict(self.risk_score),
            "duration_hours": dict(self.duration_hours),
            "trigger_components": dict(self.trigger_components),
            "component_contribution": dict(self.component_contribution),
            "ground_truth_available": self.ground_truth_available,
            "detection_skill": self.skill.as_dict(),
            "caveat": (
                "Alert severities are this project's own categories, NOT official "
                "IMD or WMO warnings. The risk score is a transparent rule-based "
                "triage aid, not a validated disaster-risk index and not a "
                "probability of occurrence."
            ),
            "warnings": list(self.warnings),
        }


def _alert_view(alert: Any, index: int) -> dict[str, Any]:
    """Project an alert onto the fields the matcher needs."""
    issued = getattr(alert, "issued_at", None)
    return {
        "event_id": str(getattr(alert, "alert_id", f"ALERT-{index:04d}")),
        "time": str(
            np.datetime_as_string(
                np.datetime64(issued if issued is not None else "2024-01-01T00:00"),
                unit="m",
            )
        ),
        "latitude": float(getattr(alert, "latitude", 0.0) or 0.0),
        "longitude": float(getattr(alert, "longitude", 0.0) or 0.0),
        "event_type": str(getattr(alert, "event_type", "unknown")),
    }



def compute_alert_diagnostics(
    alerts: Sequence[Any],
    *,
    split: str,
    dataset_id: str,
    truth: GroundTruth | None = None,
) -> AlertDiagnostics:
    """Summarise alerts without calling the severity system validated.

    Args:
        alerts: The stored alerts (Phase 5 model objects).
        split: Which split they came from.
        dataset_id: Identifier recorded in the result.
        truth: Optional reference catalogue. When present, alert-to-event skill
            is computed; when absent, the explicit refusal is carried through.

    Returns:
        An :class:`AlertDiagnostics`.
    """
    records = list(alerts)
    reference = truth or GroundTruth()

    severities: dict[str, int] = {}
    statuses: dict[str, int] = {}
    scores: list[float] = []
    durations: list[float] = []
    triggers: dict[str, int] = {}
    contribution: dict[str, dict[str, Any]] = {}

    for alert in records:
        severity = str(getattr(alert, "severity", "unknown"))
        severities[severity] = severities.get(severity, 0) + 1
        status = str(getattr(alert, "status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

        risk = getattr(alert, "risk", None)
        score = getattr(risk, "score", None) if risk is not None else None
        if score is not None:
            scores.append(float(score))

        validity = getattr(alert, "valid_until", None)
        issued = getattr(alert, "issued_at", None)
        if validity is not None and issued is not None:
            durations.append(
                float(
                    (np.datetime64(validity) - np.datetime64(issued))
                    / np.timedelta64(1, "h")
                )
            )

        for name in list(getattr(alert, "source_components", []) or []):
            triggers[str(name)] = triggers.get(str(name), 0) + 1

        for factor in list(getattr(risk, "factors", []) or []) if risk is not None else []:
            name = str(getattr(factor, "name", "unknown"))
            bucket = contribution.setdefault(name, {"count": 0, "points_total": 0.0})
            bucket["count"] += 1
            bucket["points_total"] += float(getattr(factor, "points", 0.0) or 0.0)

    skill = evaluate_detection_skill(
        [_alert_view(alert, index) for index, alert in enumerate(records)], reference
    )

    warnings: list[str] = []
    if not records:
        warnings.append("No alerts were produced on this split.")
    if reference.available:
        warnings.append(
            "Alerts are matched to reference events by issue time and location; a "
            "match does not mean the alert was operationally correct."
        )

    return AlertDiagnostics(
        split=split,
        dataset_id=dataset_id,
        alert_count=len(records),
        severity_distribution=severities,
        status_distribution=statuses,
        risk_score=summarise(scores) or {"count": 0},
        duration_hours=summarise(durations) or {"count": 0},
        trigger_components=triggers,
        component_contribution=contribution,
        ground_truth_available=reference.available,
        skill=skill,
        warnings=tuple(warnings),
    )

