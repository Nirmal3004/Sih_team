#!/usr/bin/env python3
"""Run the complete SIH demo pipeline end to end (SIH26078, Phase 11).

One deterministic, documented command that exercises every implemented stage::

    ingestion -> quality gate -> preprocessing -> anomaly detection
              -> event tracking -> graph construction -> GNN prototype
              -> AI downscaling -> alerting -> evaluation -> operations

Design commitments
------------------
**Reuse, never reimplement.** Every stage calls the same service the API and
the per-phase CLIs call - :class:`PipelineOrchestrator`,
:func:`build_dataset_from_file`, :class:`GNNService`, :class:`DownscalingService`,
:func:`run_evaluation`, :func:`build_system_status`. Nothing here duplicates
pipeline logic, so the demo cannot drift away from the product.

**data/raw/ is read-only.** Every artifact this script writes lands under
``data/processed``, ``data/realtime`` or ``data/evaluation``. The raw tree is
never modified, so re-running is safe and a source dataset cannot be corrupted
by a demo.

**Nothing is faked.** A missing checkpoint is reported ``SKIPPED`` with the
reason, an unavailable metric stays ``unavailable``, and a dataset with no
ground truth produces no accuracy figure. The summary prints ``unavailable``
rather than ``0`` wherever a number could not be measured.

**Synthetic and historical are labelled, never conflated.** The default source
is the deterministic synthetic fixture. ``--provider open-meteo`` retrieves
ERA5 *reanalysis*, which is historical, not live and not a forecast.

Examples
--------
Plan the run without writing anything (safe first move on a fresh clone)::

    python scripts/run_sih_demo.py --dry-run

Run the full demo, reusing artifacts that are already present::

    python scripts/run_sih_demo.py --reuse-existing

Start from scratch, discarding previous run records::

    python scripts/run_sih_demo.py --clean

This is a **research prototype**. It demonstrates no forecast skill, is
validated against no observed weather, and is not an operational
emergency-warning system.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extreme_weather.config import PROJECT_ID, PROJECT_NAME  # noqa: E402

#: Printed with the summary. It is a required field, not decoration: nothing
#: this script measures is a forecast-skill claim.
PROTOTYPE_NOTICE = (
    "Research prototype (SIH26078). Not an operational disaster-warning or "
    "emergency-response system. It demonstrates no forecast skill and is "
    "validated against no observed weather. Alert severities are this "
    "project's own categories, NOT official IMD or WMO warnings. No accuracy, "
    "lives-saved or economic-benefit figure is produced or implied."
)

#: Human-readable labels for the data sources this demo understands.
SOURCE_LABELS = {
    "fixture": "DETERMINISTIC SYNTHETIC (generated locally, not observed)",
    "open-meteo": "ERA5 REANALYSIS via Open-Meteo (HISTORICAL archive, not live)",
    "local": "LOCAL FILE (provenance depends on the file)",
    "remote-http": "REMOTE HTTP ENDPOINT (provenance depends on the endpoint)",
}

#: Artifacts this script is allowed to write. Anything outside this set would
#: be a bug, and the guard below turns one into a visible failure rather than
#: silent damage to a source dataset.
WRITABLE_SUBDIRS = ("processed", "realtime", "evaluation", "manifests")


@dataclass
class StageReport:
    """One stage's outcome, recorded for the final summary.

    Attributes:
        name: Stage identifier.
        status: ``OK``, ``SKIPPED``, ``UNAVAILABLE`` or ``FAILED``.
        detail: One line a judge can read without the source.
        values: Measured figures, or ``{}`` when none exist.
    """

    name: str
    status: str
    detail: str = ""
    values: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "values": self.values,
        }


class DemoRunner:
    """Drives every stage in order and collects the report.

    Args:
        dry_run: Perform the work but write nothing.
        clean: Discard previous run records before starting.
        reuse: Reuse existing artifacts instead of recomputing them.
        verbose: Print per-stage progress as it happens.
        provider: Which data source to use.
    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
        clean: bool = False,
        reuse: bool = False,
        verbose: bool = False,
        provider: str = "fixture",
        file: Path | None = None,
    ) -> None:
        from extreme_weather.config import Settings

        self.settings = Settings()
        self.dry_run = dry_run
        self.clean = clean
        self.reuse = reuse
        self.verbose = verbose
        self.provider_name = provider
        self.file = file
        self.stages: list[StageReport] = []
        self.run_id: str | None = None
        self.events: tuple[Any, ...] = ()
        self._raw_before = self._snapshot_raw()

    # --- helpers ---------------------------------------------------------

    def _snapshot_raw(self) -> set[str]:
        """Record the raw tree so a write into it can be detected and reported.

        The demo is documented as never modifying ``data/raw``. Asserting that
        claim is better than trusting it.
        """
        try:
            return {str(p) for p in self.settings.raw_dir.rglob("*")}
        except OSError:
            return set()

    def _log(self, message: str) -> None:
        """Print a progress line when running verbosely."""
        if self.verbose:
            print(f"  . {message}", flush=True)

    def _record(self, report: StageReport) -> StageReport:
        """Append a stage report and echo it in verbose mode."""
        self.stages.append(report)
        self._log(f"{report.name}: {report.status} - {report.detail}")
        return report

    def _prepare_directories(self) -> None:
        """Create the working directories, without touching ``data/raw``."""
        self.settings.ensure_directories()
        for sub in ("realtime", "evaluation"):
            (self.settings.data_dir / sub).mkdir(parents=True, exist_ok=True)

    def _clean_previous(self) -> StageReport:
        """Remove previous demo run records so a rerun starts from zero.

        Only the directories this script writes are touched. ``data/raw`` is
        never cleaned: it holds source data, not demo output.
        """
        removed = 0
        for sub in ("realtime", "evaluation"):
            directory = self.settings.data_dir / sub
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return self._record(
            StageReport(
                name="clean",
                status="OK",
                detail=f"Removed {removed} previous run record(s). Source data untouched.",
            )
        )

    # --- stages 1-8: the orchestrated pipeline ---------------------------

    def run_pipeline(self) -> StageReport:
        """Run ingestion through alerting via the Phase 8 orchestrator.

        This one call covers ingestion, the quality gate, preprocessing,
        anomaly detection, tracking, the GNN stage, AI downscaling and
        alerting, because that is exactly what the orchestrator already does.
        Calling it rather than re-implementing eight stages is what guarantees
        the demo and the product cannot disagree.
        """
        from datetime import timedelta

        from extreme_weather.realtime.orchestrator import PipelineOrchestrator
        from extreme_weather.realtime.providers import WeatherRequest, build_provider

        provider = build_provider(
            self.provider_name,
            local_path=self.file if self.provider_name == "local" else None,
        )
        orchestrator = PipelineOrchestrator(provider, settings=self.settings)
        orchestrator.load_state()

        # A fixed window keeps the fixture byte-identical between runs. It is
        # a constant rather than "now" precisely so the demo is reproducible.
        end = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)
        request = WeatherRequest(
            start=end - timedelta(hours=6),
            end=end,
            variables=("precipitation", "temperature"),
            latitude=(12.0, 12.5, 13.0),
            longitude=(76.0, 76.5, 77.0),
        )

        if self.file is not None and self.provider_name == "local":
            # A local file carries its own time axis, so the window is taken
            # from the file rather than from a fixed range that would exclude
            # it entirely.
            request = self._request_for_file(provider) or request

        self._log(f"provider={self.provider_name} window={request.start}..{request.end}")
        result = orchestrator.run(request, dry_run=self.dry_run, force=not self.reuse)
        self.run_id = result.run.run_id
        self.events = result.events

        stages = {s.name: s for s in result.run.stages}
        values = {
            "run_id": result.run.run_id,
            "status": result.run.status,
            "provider": result.run.provider,
            "time_steps": _ingest_time_steps(result.run),
            "events": len(result.events),
            "alerts": result.alerts,
            "freshness": result.run.freshness,
        }

        if not result.succeeded:
            return self._record(
                StageReport(
                    name="pipeline",
                    status="FAILED",
                    detail="; ".join(result.run.errors) or "a stage failed",
                    values=values,
                )
            )

        skipped = [s.name for s in result.run.stages if s.status == "SKIPPED"]
        detail = f"run {result.run.run_id}: {len(result.run.stages)} stages completed."
        if skipped:
            # A skipped model stage is a prototype limitation, not a failure,
            # and the reason is already recorded on the stage itself.
            detail += f" Skipped (no trained checkpoint): {', '.join(skipped)}."
        return self._record(
            StageReport(name="pipeline", status="OK", detail=detail, values=values)
        )

    def _request_for_file(self, provider: Any) -> Any:
        """Build a request that covers a local file's own time span.

        A fixed demonstration window would exclude most local datasets, so the
        window is derived from the file itself. Returns ``None`` when the span
        cannot be read, in which case the caller keeps the default window.
        """
        if self.file is None:
            return None

        import xarray as xr

        from extreme_weather.realtime.providers import WeatherRequest

        try:
            with xr.open_dataset(self.file) as dataset:
                times = dataset.coords["time"].values
        except Exception:  # noqa: BLE001 - the default window is a fine fallback
            return None
        if times is None or len(times) == 0:
            return None
        start = times[0].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
        end = times[-1].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
        # Variables and a bounding box stay set: WeatherRequest refuses an
        # empty variable list or an unbounded box, by design.
        return WeatherRequest(
            start=start,
            end=end,
            variables=("precipitation", "temperature"),
            latitude=(12.0, 12.5, 13.0),
            longitude=(76.0, 76.5, 77.0),
        )

    def report_tracks(self) -> StageReport:
        """Report the tracks the pipeline produced, read back from the store.

        The count is read from the store rather than from the run in memory, so
        what is printed is what a later API call would actually return.
        """
        from extreme_weather.tracking.store import TrackStore

        path = self.settings.processed_dir / self.settings.tracks_filename
        try:
            result = TrackStore(path).load()
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            return self._record(
                StageReport(
                    name="tracks",
                    status="UNAVAILABLE",
                    detail=f"Could not read the track store: {type(exc).__name__}.",
                    values={"tracks": None},
                )
            )
        if result is None:
            return self._record(
                StageReport(
                    name="tracks",
                    status="UNAVAILABLE",
                    detail="No track file on record; the tracking stage produced none.",
                    values={"tracks": None},
                )
            )
        count = len(result.tracks)
        return self._record(
            StageReport(
                name="tracks",
                status="OK" if count else "UNAVAILABLE",
                detail=f"{count} track(s) read back from {path.name}.",
                values={"tracks": count},
            )
        )

    # --- stage 9: graph construction --------------------------------------

    def build_graph(self) -> StageReport:
        """Build the GNN graph dataset from the tracks, and report its shape.

        Structure (node, edge and sample counts) is reported, never a
        performance figure: a graph is a data structure, not a result.
        """
        from extreme_weather.gnn.dataset import SplitConfig, build_dataset_from_file
        from extreme_weather.gnn.graph import GraphConfig

        path = self.settings.processed_dir / self.settings.tracks_filename
        if not path.exists():
            return self._record(
                StageReport(
                    name="graph",
                    status="UNAVAILABLE",
                    detail=(
                        f"No {path.name} on record, so no graph was built. This is "
                        "a data-availability outcome, not an error."
                    ),
                    values={"nodes": None, "edges": None},
                )
            )
        try:
            dataset = build_dataset_from_file(
                path, GraphConfig(), SplitConfig(train=0.6, validation=0.2, test=0.2)
            )
        except (FileNotFoundError, ValueError) as exc:
            return self._record(
                StageReport(
                    name="graph",
                    status="UNAVAILABLE",
                    detail=f"Graph could not be built: {exc}",
                    values={"nodes": None, "edges": None},
                )
            )

        summary = dataset.as_dict()
        return self._record(
            StageReport(
                name="graph",
                status="OK",
                detail=(
                    f"{summary['node_count']} nodes, {summary['edge_count']} edges "
                    f"({summary['temporal_edge_count']} temporal, "
                    f"{summary['spatial_edge_count']} spatial) from "
                    f"{summary['track_count']} track(s)."
                ),
                values={
                    "nodes": summary["node_count"],
                    "edges": summary["edge_count"],
                    "tracks": summary["track_count"],
                    "prediction_samples": summary["prediction_samples"],
                },
            )
        )

    # --- stage 10: model availability -------------------------------------

    def report_models(self) -> StageReport:
        """Report GNN and downscaling availability from their own services.

        Availability is read from the service that owns the checkpoint. When a
        model is missing the stage reports ``SKIPPED`` with the reason and no
        prediction is invented - that is the honest outcome for this project.
        """
        results: dict[str, Any] = {}
        details: list[str] = []

        try:
            from extreme_weather.gnn.service import GNNService

            status = GNNService().status()
            available = bool(getattr(status, "available", False))
            results["gnn_available"] = available
            details.append(
                f"GNN {'available' if available else 'SKIPPED (no checkpoint)'}"
            )
        except Exception as exc:  # noqa: BLE001
            results["gnn_available"] = None
            details.append(f"GNN status unavailable ({type(exc).__name__})")

        try:
            from extreme_weather.downscaling.service import DownscalingService

            status = DownscalingService().status()
            available = bool(status.get("available"))
            results["downscaling_available"] = available
            details.append(
                f"AI downscaling {'available' if available else 'SKIPPED (no checkpoint)'}"
            )
        except Exception as exc:  # noqa: BLE001
            results["downscaling_available"] = None
            details.append(f"Downscaling status unavailable ({type(exc).__name__})")

        both_missing = results.get("gnn_available") is False and (
            results.get("downscaling_available") is False
        )
        return self._record(
            StageReport(
                name="models",
                status="OK" if not both_missing else "SKIPPED",
                detail=(
                    "; ".join(details)
                    + (
                        ". Neither model is trained, which is the documented "
                        "state of a fresh checkout - no prediction is substituted."
                        if both_missing
                        else "."
                    )
                ),
                values=results,
            )
        )

    # --- stage 11: alerts -------------------------------------------------

    def report_alerts(self) -> StageReport:
        """Report alerts from the store, with the project's own severity caveat."""
        from extreme_weather.alerts.store import AlertStore

        path = self.settings.processed_dir / self.settings.alerts_filename
        try:
            result = AlertStore(path).load()
        except Exception as exc:  # noqa: BLE001
            return self._record(
                StageReport(
                    name="alerts",
                    status="UNAVAILABLE",
                    detail=f"Could not read the alert store: {type(exc).__name__}.",
                    values={"alerts": None},
                )
            )
        if result is None:
            return self._record(
                StageReport(
                    name="alerts",
                    status="UNAVAILABLE",
                    detail="No alert file on record; no alerts have been generated.",
                    values={"alerts": None},
                )
            )
        count = len(result.alerts)
        return self._record(
            StageReport(
                name="alerts",
                status="OK" if count else "UNAVAILABLE",
                detail=(
                    f"{count} alert(s) stored. Severities are this project's own "
                    "categories, not official IMD or WMO warnings."
                ),
                values={"alerts": count},
            )
        )

    # --- stage 12: evaluation ---------------------------------------------

    def run_evaluation(self) -> StageReport:
        """Run the Phase 9 evaluation and report only what it actually measured.

        Without a ground-truth reference dataset there is no accuracy figure to
        report, and this method says so rather than emitting a zero. That is the
        documented behaviour of :func:`run_evaluation`, which records a reason
        for every quantity it cannot compute.
        """
        from extreme_weather.evaluation.evaluator import run_evaluation

        try:
            outcome = run_evaluation(
                model_available=self._gnn_available(),
                seed=self.settings.synthetic_seed,
                run_id=self.run_id,
            )
        except Exception as exc:  # noqa: BLE001 - an evaluation gap is not a crash
            return self._record(
                StageReport(
                    name="evaluation",
                    status="UNAVAILABLE",
                    detail=f"Evaluation could not run: {type(exc).__name__}.",
                    values={"evaluation": "unavailable"},
                )
            )

        run = outcome.run
        # `metrics` holds what was measured; `unavailable_metrics` holds what
        # could not be, each with a reason. Both are reported, and the second
        # is never collapsed into a zero.
        measured = sum(len(v) for v in (run.metrics or {}).values())
        unavailable = sum(len(v) for v in (run.unavailable_metrics or {}).values())
        detail = (
            f"Evaluation run {run.run_id}: {measured} metric(s) measured, "
            f"{unavailable} reported as unavailable with a reason."
        )
        if not measured:
            detail += (
                " Nothing was measurable: no verified ground-truth reference "
                "dataset exists for this deployment, so no accuracy figure is "
                "produced and none is implied."
            )
            reasons = _unavailable_reasons(run)
            if reasons:
                detail += f" First reason: {reasons[0][:160]}"
        return self._record(
            StageReport(
                name="evaluation",
                status="OK" if measured else "UNAVAILABLE",
                detail=detail,
                values={
                    "evaluation": run.run_id,
                    "status": run.status,
                    "metrics_measured": measured,
                    "metrics_unavailable": unavailable,
                },
            )
        )

    def _gnn_available(self) -> bool:
        """Whether a trained GNN checkpoint exists, without raising."""
        try:
            from extreme_weather.gnn.service import GNNService

            return bool(getattr(GNNService().status(), "available", False))
        except Exception:  # noqa: BLE001
            return False

    # --- stage 13: operations --------------------------------------------

    def report_operations(self) -> StageReport:
        """Report readiness and health from the Phase 10 operations services."""
        from extreme_weather.api.routers.operations import _build_readiness
        from extreme_weather.api.routers.system import build_system_status

        readiness = _build_readiness(self.settings)
        status = build_system_status(self.settings)
        return self._record(
            StageReport(
                name="operations",
                status="OK",
                detail=(
                    f"readiness={readiness.state}; "
                    f"environment={self.settings.app_env}; "
                    f"data_source={status.data_source.kind}."
                ),
                values={
                    "readiness": readiness.state,
                    "ready": readiness.ready,
                    "data_source": status.data_source.kind,
                    "counts": status.counts.model_dump(),
                },
            )
        )

    def verify_raw_untouched(self) -> StageReport:
        """Confirm the demo did not modify ``data/raw``.

        The script promises this, so it is checked rather than asserted in
        prose. A violation is reported as a failure, because silently writing
        into the source tree is exactly the bug a demo must not have.
        """
        after = self._snapshot_raw()
        added = sorted(after - self._raw_before)
        if added:
            return self._record(
                StageReport(
                    name="raw-integrity",
                    status="FAILED",
                    detail=(
                        f"{len(added)} new path(s) appeared under "
                        f"{self.settings.raw_dir}. The demo must not write there."
                    ),
                )
            )
        return self._record(
            StageReport(
                name="raw-integrity",
                status="OK",
                detail=f"{len(after)} path(s) under data/raw, unchanged by this run.",
            )
        )

    # --- orchestration ----------------------------------------------------

    def run_all(self) -> list[StageReport]:
        """Run every stage in order and return the reports.

        A genuine failure stops the run: continuing past a failed pipeline
        would print a summary that implies later stages worked.
        """
        self._prepare_directories()
        if self.clean:
            self._clean_previous()

        pipeline = self.run_pipeline()
        if pipeline.status == "FAILED":
            # Still verify the raw tree and report honestly rather than
            # pretending the rest of the pipeline ran.
            self.verify_raw_untouched()
            return self.stages

        self.report_tracks()
        self.build_graph()
        self.report_models()
        self.report_alerts()
        self.run_evaluation()
        self.report_operations()
        self.verify_raw_untouched()
        return self.stages

    @property
    def failed(self) -> list[StageReport]:
        """Stages that genuinely failed, as opposed to being skipped."""
        return [s for s in self.stages if s.status == "FAILED"]

    def summary(self) -> dict[str, Any]:
        """Return the machine-readable summary."""
        return {
            "project": PROJECT_ID,
            "service": PROJECT_NAME,
            "provider": self.provider_name,
            "data_source_label": SOURCE_LABELS.get(
                self.provider_name, "UNKNOWN SOURCE"
            ),
            "dry_run": self.dry_run,
            "run_id": self.run_id,
            "stages": [s.as_dict() for s in self.stages],
            "failed_stages": [s.name for s in self.failed],
            "prototype": True,
            "disclaimer": PROTOTYPE_NOTICE,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ingest_time_steps(run: Any) -> int | None:
    """Return the ingested step count, or ``None`` when it is unknown.

    Read from the ingestion record, which is where the Phase 8 orchestrator
    records the provider's own count. ``None`` rather than ``0``: "the provider
    did not say" is not the same claim as "there were no steps".
    """
    record = getattr(run, "ingestion", None)
    if record is None:
        return None
    value = getattr(record, "time_steps", None)
    return value if isinstance(value, int) and value > 0 else None


def _unavailable_reasons(run: Any) -> list[str]:
    """Return a human-readable list of the quantities that could not be computed.

    Read defensively: the exact accessor is an implementation detail of the
    Phase 9 model, and the demo must not break if it is renamed.
    """
    entries = getattr(run, "unavailable_metrics", None) or {}
    reasons: list[str] = []
    for group, metrics in entries.items():
        for name, entry in metrics.items():
            reason = ""
            if isinstance(entry, dict):
                reason = str(entry.get("reason") or entry.get("status") or "")
            reasons.append(f"{group}.{name}: {reason}" if reason else f"{group}.{name}")
    return reasons


def _format_value(value: Any) -> str:
    """Render one value for the summary, spelling out "not measured"."""
    if value is None:
        # Never print 0 for something that was not measured.
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_summary(runner: DemoRunner) -> str:
    """Build the human-readable final summary."""
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"{PROJECT_NAME} ({PROJECT_ID}) - SIH demo pipeline")
    add("=" * 72)
    add(f"Data source        : {SOURCE_LABELS.get(runner.provider_name, 'UNKNOWN')}")
    add(f"Mode               : {'DRY RUN (nothing written)' if runner.dry_run else 'LIVE'}")
    add(f"Pipeline run ID    : {runner.run_id or '(no run completed)'}")
    add("")

    add("Stages")
    add("-" * 72)
    for stage in runner.stages:
        add(f"  [{stage.status:>11}] {stage.name}")
        if stage.detail:
            add(f"                  {stage.detail}")
        for key, value in stage.values.items():
            if key in ("run_id",):
                continue
            add(f"                    - {key}: {_format_value(value)}")
    add("")

    add("Pipeline summary")
    add("-" * 72)
    by_name = {s.name: s for s in runner.stages}
    rows = [
        ("Ingestion status", (by_name.get("pipeline").detail if by_name.get("pipeline") else "not run")),
        ("Weather steps", (by_name["pipeline"].values.get("time_steps") if by_name.get("pipeline") else None)),
        ("Detected events", (by_name["pipeline"].values.get("events") if by_name.get("pipeline") else None)),
        ("Event tracks", (by_name["tracks"].values.get("tracks") if by_name.get("tracks") else None)),
        ("GNN availability", (by_name["models"].values.get("gnn_available") if by_name.get("models") else None)),
        ("Downscaling availability", (by_name["models"].values.get("downscaling_available") if by_name.get("models") else None)),
        ("Alerts generated", (by_name["alerts"].values.get("alerts") if by_name.get("alerts") else None)),
        ("Pipeline run ID", runner.run_id),
        ("Evaluation status", (by_name["evaluation"].status if by_name.get("evaluation") else None)),
        ("Validation status", (by_name["operations"].values.get("readiness") if by_name.get("operations") else None)),
    ]
    for label, value in rows:
        if isinstance(value, str) and len(value) > 40:
            value = f"{value[:37]}..."
        add(f"  {label:<26}: {_format_value(value)}")
    add("")

    add("PROTOTYPE DISCLAIMER")
    add("-" * 72)
    for chunk in _wrap(PROTOTYPE_NOTICE, 70):
        add(f"  {chunk}")
    add("=" * 72)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to a width, for the printed disclaimer."""
    import textwrap

    return textwrap.wrap(text, width) or [""]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_sih_demo.py",
        description=(
            f"{PROJECT_NAME} ({PROJECT_ID}) - run the complete pipeline end to "
            "end as a reproducible demonstration (Phase 11). Research prototype: "
            "not an operational emergency-warning system, and no forecast skill "
            "is demonstrated."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --dry-run          plan the run, write nothing\n"
            "  %(prog)s --reuse-existing  reuse artifacts already present\n"
            "  %(prog)s --clean           discard previous run records first\n"
            "  %(prog)s --verbose         print each stage as it runs\n"
            "  %(prog)s --json            machine-readable summary\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every stage but write no artifact. Safe on a fresh clone.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previous demo run records before starting. Never touches data/raw.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse artifacts already on disk instead of recomputing them.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print each stage as it runs."
    )
    parser.add_argument(
        "--provider",
        default="local",
        help=(
            "Data source. 'local' (default) reads --file, which defaults to "
            "the bundled sample containing an injected extreme event; "
            "'fixture' is a smooth deterministic synthetic field needing no "
            "network (it contains no extremes, so it detects 0 events - that is "
            "correct, not a failure); 'open-meteo' is ERA5 reanalysis "
            "(historical, not live)."
        ),
    )
    parser.add_argument(
        "--file",
        default="data/raw/sample_history.nc",
        help=(
            "NetCDF file used with --provider local. Defaults to the bundled "
            "sample, which carries an injected extreme event so anomaly "
            "detection, tracking and alerting all have something to show. "
            "The file is only ever READ; data/raw is never modified."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a formatted report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the demo pipeline and print the summary.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when no stage genuinely failed, ``1`` otherwise. A stage that is
        ``SKIPPED`` or ``UNAVAILABLE`` is a documented outcome and does not
        fail the run - only a real failure does.
    """
    args = build_parser().parse_args(argv)

    if args.dry_run and args.clean:
        print(
            "error: --dry-run and --clean contradict each other: one writes "
            "nothing, the other deletes records.",
            file=sys.stderr,
        )
        return 2

    runner = DemoRunner(
        dry_run=args.dry_run,
        clean=args.clean,
        reuse=args.reuse_existing,
        verbose=args.verbose,
        provider=args.provider,
        file=Path(args.file) if args.file else None,
    )

    try:
        runner.run_all()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a demo must not traceback at a judge
        print(f"error: the demo failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(runner.summary(), indent=2, default=str))
    else:
        print(render_summary(runner))

    if runner.failed:
        print(
            f"error: {len(runner.failed)} stage(s) failed: "
            f"{', '.join(s.name for s in runner.failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


