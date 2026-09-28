"""Anomaly detection service (SIH26078, Phase 1).

This module is the business layer: it wires climatology, scores, thresholds,
detection and clustering together, and it owns the file-backed event store. The
FastAPI router and the CLI both call it, so neither contains analysis logic.

```text
analyse_dataset()
    load / accept dataset
      -> build_climatology()
      -> compute_score()
      -> detect_extreme_cells()
      -> cluster_extreme_events()
      -> AnomalyAnalysisResult (events + provenance)
```

File layout (created automatically, never overwriting raw data):

```text
data/processed/events.json      # stable JSON schema, read by the API
data/processed/anomaly.nc       # scores + detection mask (optional)
```

No database is used in Phase 1: the analysis is a file the CLI writes and the
API reads. That keeps the demo reproducible and the infrastructure minimal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from extreme_weather.anomaly.climatology import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_QUANTILE_LEVELS,
    ClimatologyError,
    ClimatologyGrouping,
    build_climatology,
)
from extreme_weather.anomaly.clustering import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_MIN_CELLS,
    ClusteringError,
    cluster_extreme_events,
)
from extreme_weather.anomaly.detector import DetectionError, detect_extreme_cells
from extreme_weather.anomaly.models import AnomalyAnalysisResult
from extreme_weather.anomaly.scores import (
    EFI_FORMULA,
    SCORE_FUNCTIONS,
    ScoreError,
    compute_score,
)
from extreme_weather.anomaly.thresholds import (
    PROTOTYPE_THRESHOLD_WARNING,
    ThresholdsError,
    ThresholdLadder,
    ThresholdSet,
    default_thresholds,
)
from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID, get_settings
from extreme_weather.data.loader import load_dataset, save_dataset
from extreme_weather.schemas.event import EventType, ExtremeWeatherEvent

__all__ = [
    "DEFAULT_ANALYSIS_VARIABLES",
    "EventStore",
    "AnomalyDetectionService",
    "AnalysisError",
    "analyse_dataset",
]

#: Variables analysed by default; ``wind_speed`` is derived by preprocessing.
DEFAULT_ANALYSIS_VARIABLES: tuple[str, ...] = ("precipitation",)

#: Default output file names (relative to ``DATA_DIR/processed``).
#: Default name of the anomaly NetCDF written next to the events file.
ANOMALY_FILENAME: str = "anomaly.nc"


class AnalysisError(RuntimeError):
    """Raised when an anomaly analysis cannot be completed."""


def _utc_now() -> datetime:
    """Return the current UTC time (second resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class EventStore:
    """File-backed store for the latest analysis result.

    Args:
        path: JSON file holding the analysis. Defaults to
            ``<DATA_DIR>/processed/<settings.events_filename>``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        settings = get_settings()
        self._path = (
            Path(path) if path is not None else settings.processed_dir / settings.events_filename
        )

    @property
    def path(self) -> Path:
        """Location of the analysis file."""
        return self._path

    def exists(self) -> bool:
        """True when an analysis file is present."""
        return self._path.is_file()

    def save(self, result: AnomalyAnalysisResult) -> Path:
        """Write ``result`` as JSON, creating parent directories."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.as_dict()
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        return self._path

    def load(self) -> AnomalyAnalysisResult | None:
        """Read the stored analysis, or ``None`` when nothing is stored yet.

        Raises:
            AnalysisError: If the file exists but cannot be parsed.
        """
        if not self._path.is_file():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"Could not read analysis file '{self._path}': {exc}") from exc
        if not isinstance(payload, Mapping):
            raise AnalysisError(
                f"Analysis file '{self._path}' must contain a JSON object, got "
                f"{type(payload).__name__}."
            )
        return AnomalyAnalysisResult.from_dict(payload)

    def clear(self) -> None:
        """Delete the stored analysis if it exists."""
        self._path.unlink(missing_ok=True)


class AnomalyDetectionService:
    """Runs anomaly detection end to end and persists the result.

    The service holds no global state: thresholds, grouping and output paths are
    explicit instance attributes, so two services with different configurations
    can coexist (this is what the tests rely on).

    Example::

        service = AnomalyDetectionService()
        result = service.run("data/raw/sample_history.nc")
        print(result.event_count, "events")
    """

    def __init__(
        self,
        *,
        store: EventStore | None = None,
        thresholds: ThresholdSet | None = None,
        default_grouping: ClimatologyGrouping | str = ClimatologyGrouping.MONTH,
        default_score: str = "absolute",
        min_cells: int = DEFAULT_MIN_CELLS,
        connectivity: int = DEFAULT_CONNECTIVITY,
    ) -> None:
        self.store = store or EventStore()
        self.thresholds = thresholds or default_thresholds()
        self.default_grouping = ClimatologyGrouping.from_value(default_grouping)
        self.default_score = str(default_score)
        self.min_cells = int(min_cells)
        self.connectivity = int(connectivity)

    # -- configuration helpers -------------------------------------------- #
    def _thresholds_for_variable(self, variable: str) -> ThresholdLadder:
        """Return the ladder registered for a variable.

        Raises:
            ThresholdsError: If the variable has no ladder.
        """
        ladder = self.thresholds.for_variable(variable)
        if ladder is None:
            available = ", ".join(sorted(item.variable for item in self.thresholds.ladders))
            raise ThresholdsError(
                f"No threshold ladder configured for variable '{variable}'. Available: {available}."
            )
        return ladder

    @staticmethod
    def _require_variable(dataset: xr.Dataset, variable: str) -> None:
        """Raise a helpful error when the dataset cannot be analysed."""
        if variable not in dataset.data_vars:
            raise AnalysisError(
                f"Variable '{variable}' is not in the dataset. Available: "
                f"{sorted(dataset.data_vars)}. Run the preprocessing pipeline first "
                "so derived variables such as 'wind_speed' exist."
            )

    # -- analysis --------------------------------------------------------- #
    def analyse(
        self,
        dataset: xr.Dataset,
        *,
        variables: Iterable[str] = DEFAULT_ANALYSIS_VARIABLES,
        reference: xr.Dataset | None = None,
        source_path: str | None = None,
        reference_path: str | None = None,
        grouping: ClimatologyGrouping | str | None = None,
        score_name: str | None = None,
        min_cells: int | None = None,
        write_outputs: bool = False,
        keep_fields: bool = False,
    ) -> AnomalyAnalysisResult:
        """Detect extreme events in ``dataset``.

        Args:
            dataset: Dataset to analyse (already preprocessed).
            variables: Variables to analyse. Each one needs a threshold ladder.
            reference: Reference dataset for the climatology. ``None`` means the
                analysis uses its own dataset as reference (a demo shortcut, not
                a real workflow).
            source_path: Provenance path of ``dataset``.
            reference_path: Provenance path of ``reference``.
            grouping: Climatology grouping; defaults to the service setting.
            score_name: Anomaly score; defaults to the service setting (``efi``).
            min_cells: Minimum cluster size; defaults to the service setting.
            write_outputs: Write ``events.json`` (and ``anomaly.nc`` when
                ``keep_fields`` is set) into the processed directory.
            keep_fields: Include the score and mask fields in ``anomaly.nc``.

        Returns:
            An :class:`AnomalyAnalysisResult` with the detected events.

        Raises:
            AnalysisError: If the analysis cannot be completed.
        """
        resolved_grouping = ClimatologyGrouping.from_value(
            grouping if grouping is not None else self.default_grouping
        )
        resolved_score = str(score_name if score_name is not None else self.default_score)
        if resolved_score not in SCORE_FUNCTIONS:
            raise AnalysisError(
                f"Unknown anomaly score '{resolved_score}'. Available: "
                f"{', '.join(sorted(SCORE_FUNCTIONS))}."
            )
        resolved_min_cells = int(self.min_cells if min_cells is None else min_cells)

        reference_dataset = reference if reference is not None else dataset
        events: list[ExtremeWeatherEvent] = []
        analysis_fields: dict[str, xr.DataArray] = {}
        variable_list: list[str] = []

        for variable in variables:
            self._require_variable(dataset, variable)
            self._require_variable(reference_dataset, variable)
            ladder = self._thresholds_for_variable(variable)
            variable_list.append(variable)

            try:
                climatology = build_climatology(
                    reference_dataset, variable, resolved_grouping
                )
                values = dataset[variable]
                score = compute_score(resolved_score, values, climatology)
                mask = detect_extreme_cells(
                    score, ladder, event_type=ladder.event_type, name=f"{variable}_extreme_mask"
                )
                variable_events = cluster_extreme_events(
                    mask,
                    score,
                    ladder=ladder,
                    event_type=ladder.event_type,
                    values=values,
                    min_cells=resolved_min_cells,
                    connectivity=self.connectivity,
                    variable=variable,
                    units=str(values.attrs.get("units", "")) or None,
                )
            except (
                ClimatologyError,
                ScoreError,
                DetectionError,
                ClusteringError,
                ThresholdsError,
            ) as exc:
                raise AnalysisError(
                    f"Anomaly analysis failed for variable '{variable}': {type(exc).__name__}: {exc}"
                ) from exc

            events.extend(variable_events)
            analysis_fields[f"{variable}_score"] = score
            analysis_fields[f"{variable}_mask"] = mask.astype("int8")
            analysis_fields[f"{variable}_climatology_mean"] = climatology["mean"].rename(
                "climatology_mean"
            )

        events.sort(key=lambda event: (event.timestamp, -event.anomaly_score, event.event_id))
        result = AnomalyAnalysisResult(
            events=tuple(events),
            variables=tuple(variable_list),
            source_dataset=source_path,
            reference_dataset=reference_path,
            score_name=resolved_score,
            grouping=resolved_grouping.value,
            generated_at=_utc_now(),
            metadata=self._metadata(resolved_score, resolved_grouping, resolved_min_cells, events),
        )

        if write_outputs:
            self.write_outputs(result, extra_fields=analysis_fields if keep_fields else None)
        return result

    def _metadata(
        self,
        score_name: str,
        grouping: ClimatologyGrouping,
        min_cells: int,
        events: Sequence[ExtremeWeatherEvent],
    ) -> dict[str, Any]:
        """Build the provenance block stored with every analysis."""
        severities = {event.severity for event in events}
        return {
            "analyzer_version": PACKAGE_VERSION,
            "project": PROJECT_ID,
            "score_formula": EFI_FORMULA,
            "score_names_available": sorted(SCORE_FUNCTIONS),
            "climatology_grouping": grouping.value,
            "climatology_min_samples": DEFAULT_MIN_SAMPLES,
            "climatology_quantile_levels": len(DEFAULT_QUANTILE_LEVELS),
            "thresholds": {
                ladder.name: {
                    "variable": ladder.variable,
                    "event_type": ladder.event_type.value,
                    "score_name": ladder.score_name,
                    "steps": [[threshold, severity.value] for threshold, severity in ladder.steps],
                    "note": ladder.note,
                }
                for ladder in self.thresholds.ladders
            },
            "threshold_disclaimer": PROTOTYPE_THRESHOLD_WARNING,
            "min_cells": int(min_cells),
            "connectivity": self.connectivity,
            "severity_counts": {
                severity.value: sum(1 for event in events if event.severity is severity)
                for severity in sorted(severities, key=lambda item: item.rank)
            },
            "data_disclaimer": (
                "Prototype analysis on synthetic data. Not an operational forecast and not "
                "suitable for real warning decisions."
            ),
        }

    def run(self, dataset_path: Path | str, *, reference_path: Path | str | None = None, **kwargs: Any) -> AnomalyAnalysisResult:
        """Load a dataset from disk and analyse it."""
        source = Path(dataset_path)
        dataset = load_dataset(source, validate=False)
        reference = load_dataset(Path(reference_path), validate=False) if reference_path else None
        kwargs.setdefault("source_path", str(source))
        kwargs.setdefault("reference_path", str(reference_path) if reference_path else None)
        return self.analyse(dataset, reference=reference, **kwargs)

    def write_outputs(
        self,
        result: AnomalyAnalysisResult,
        *,
        extra_fields: Mapping[str, xr.DataArray] | None = None,
    ) -> tuple[Path, Path | None]:
        """Persist the result as ``events.json`` and optionally ``anomaly.nc``.

        Returns:
            ``(events_path, anomaly_path)``; ``anomaly_path`` is ``None`` when no
            extra fields were supplied.
        """
        events_path = self.store.save(result)

        anomaly_path: Path | None = None
        if extra_fields:
            anomaly_path = self.store.path.parent / ANOMALY_FILENAME
            payload = xr.Dataset(dict(extra_fields))
            payload.attrs.update(
                {
                    "project": PROJECT_ID,
                    "analyzer_version": PACKAGE_VERSION,
                    "generated_at": result.generated_at.isoformat() if result.generated_at else "",
                    "score_name": result.score_name,
                    "climatology_grouping": result.grouping,
                    "event_count": str(result.event_count),
                    "source_dataset": str(result.source_dataset or ""),
                    "threshold_disclaimer": PROTOTYPE_THRESHOLD_WARNING,
                }
            )
            save_dataset(payload, anomaly_path, encoding={})
        return events_path, anomaly_path


def analyse_dataset(
    dataset_path: Path | str,
    *,
    reference_path: Path | str | None = None,
    store_path: Path | str | None = None,
    write_outputs: bool = True,
    keep_fields: bool = True,
    **kwargs: Any,
) -> AnomalyAnalysisResult:
    """Convenience wrapper: analyse a dataset and persist the result.

    This is the entry point used by the CLI and by tests; it creates a fresh
    :class:`AnomalyDetectionService` (no global state).

    Args:
        dataset_path: Dataset to analyse.
        reference_path: Optional reference dataset for the climatology.
        store_path: Where to write ``events.json`` (default: processed dir).
        write_outputs: Persist the analysis.
        keep_fields: Also write ``anomaly.nc`` with the score/mask fields.
        **kwargs: Forwarded to :meth:`AnomalyDetectionService.analyse`.

    Returns:
        The :class:`AnomalyAnalysisResult`.
    """
    service = AnomalyDetectionService(store=EventStore(store_path))
    return service.run(
        dataset_path,
        reference_path=reference_path,
        write_outputs=write_outputs,
        keep_fields=keep_fields,
        **kwargs,
    )
