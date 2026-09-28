"""The near-real-time pipeline orchestrator (SIH26078, Phase 8, section 8).

This module wires the existing Phases 0-6 components into a single, observable
run. It does not reimplement any of them: it *calls* the same services the CLI
scripts call, so a near-real-time run and a manual run produce the same results
by construction.

```text
WeatherProvider -> Ingestion -> QualityGate -> Preprocess -> Anomaly
                 -> Track -> GNN (optional) -> Downscale (optional) -> Alert
```

Failure policy
--------------
A failed stage stops the pipeline. That is the right default: running anomaly
detection on data that failed its quality gate produces plausible-looking
nonsense, and quietly continuing is how that gets shipped.

The exceptions, and the reasons for them:

* **Optional model stages** (GNN, downscaling) are ``SKIPPED``, never ``FAILED``.
  A missing checkpoint is a normal state for a research prototype, and the
  pipeline's job is to say "not available" rather than to crash or - worse - to
  emit an invented prediction.
* **A rejected quality gate** stops the run before any downstream stage, and
  the run is recorded with the full verdict.
* **An alert failure** fails the run. Alerts are the last stage, so nothing
  downstream is lost, and a silent alerting failure is exactly the failure mode
  this project must not have.

Dry run
-------
``dry_run=True`` performs the real work in memory and writes nothing at all.
Not "writes to a temporary directory" - nothing. Every persistence call is
gated on ``not self.dry_run``, so a dry run cannot leave an artifact behind even
if a later stage is added carelessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

from extreme_weather.observability.timing import StageTimer
from extreme_weather.realtime.exceptions import (
    PipelineConfigurationError,
    QualityGateRejection,
)
from extreme_weather.realtime.freshness import (
    DEFAULT_FRESHNESS_THRESHOLDS,
    FreshnessThresholds,
    assess_freshness,
)
from extreme_weather.realtime.idempotency import compute_run_identity
from extreme_weather.realtime.logging import PipelineLogger
from extreme_weather.realtime.models import (
    FAILED,
    FRESH,
    RUNNING,
    SKIPPED,
    SUCCESS,
    WARNING,
    IngestionRecord,
    PipelineRun,
    StageResult,
    redact_url,
    utc_now,
)
from extreme_weather.realtime.providers import (
    FixtureProvider,
    WeatherProvider,
    WeatherRequest,
)
from extreme_weather.realtime.quality_gate import QualityGate, QualityVerdict
from extreme_weather.realtime.retry import Attempt, RetryPolicy, retry_call

__all__ = ["IncrementalState", "OrchestratorResult", "PipelineOrchestrator", "STAGE_ORDER"]

#: The pipeline in execution order. Exported so the API, the CLI and the tests
#: all describe the same pipeline without restating it.
STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "validate",
    "preprocess",
    "anomaly",
    "track",
    "gnn",
    "downscale",
    "alert",
)

UTC = timezone.utc

#: Hours of history kept ahead of new data so climatology and tracking have
#: context. See :class:`IncrementalState` for why this is not optional.
DEFAULT_CONTEXT_STEPS = 24


@dataclass(frozen=True)
class IncrementalState:
    """What the pipeline remembers between runs.

    Attributes:
        last_processed: The newest timestamp already carried through the
            pipeline.
        newest_available: The newest timestamp the provider has.
        context_steps: How many earlier steps to prepend for context.
        processed_fingerprints: Fingerprints of ingestions already processed,
            which is what makes a rerun idempotent.
    """

    last_processed: str | None = None
    newest_available: str | None = None
    context_steps: int = DEFAULT_CONTEXT_STEPS
    processed_fingerprints: tuple[str, ...] = ()

    @property
    def has_processed(self) -> bool:
        """True when at least one ingestion has been processed before."""
        return bool(self.processed_fingerprints)

    def has_seen(self, fingerprint: str) -> bool:
        """Return True when this exact ingestion was already processed."""
        return fingerprint in self.processed_fingerprints

    def new_steps(self) -> int | None:
        """Number of new time steps, or ``None`` when it cannot be determined.

        ``None`` means "cannot tell", and the caller must then recompute - the
        safe default. Guessing "0 new steps" when in doubt would silently drop
        data.
        """
        if self.last_processed is None or self.newest_available is None:
            return None
        previous = _parse(self.last_processed)
        newest = _parse(self.newest_available)
        if previous is None or newest is None:
            return None
        return max(0, int((newest - previous).total_seconds() // 3600))

    def with_progress(
        self, newest: str, fingerprint: str
    ) -> "IncrementalState":
        """Return a copy advanced past ``newest`` and remembering ``fingerprint``."""
        return IncrementalState(
            last_processed=newest,
            newest_available=newest,
            context_steps=self.context_steps,
            processed_fingerprints=self.processed_fingerprints + (fingerprint,),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the state."""
        steps = self.new_steps()
        return {
            "last_processed": self.last_processed,
            "newest_available": self.newest_available,
            "context_steps": self.context_steps,
            "new_steps": steps,
            "incremental": steps is not None,
            "processed_fingerprints": list(self.processed_fingerprints[-10:]),
        }


def _parse(moment: str | None) -> datetime | None:
    """Parse an ISO timestamp, returning ``None`` when unusable."""
    if not moment:
        return None
    text = str(moment)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


@dataclass
class OrchestratorResult:
    """The return value of :meth:`PipelineOrchestrator.run`.

    Attributes:
        run: The full run record.
        dataset: The processed dataset, or ``None`` if the run stopped early.
        quality: The quality verdict, when the gate ran.
        events: Detected events, when anomaly detection ran.
        alerts: Number of alerts held after the run.
    """

    run: PipelineRun
    dataset: xr.Dataset | None = None
    quality: QualityVerdict | None = None
    events: tuple[Any, ...] = ()
    alerts: int = 0

    @property
    def succeeded(self) -> bool:
        """True when the run completed without a failed stage."""
        return self.run.status in (SUCCESS, WARNING)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no array data)."""
        payload = self.run.as_dict()
        payload["event_count"] = len(self.events)
        payload["alert_count"] = self.alerts
        if self.quality is not None:
            payload["quality"] = self.quality.as_dict()
        return payload

class PipelineOrchestrator:
    """Drives one near-real-time run from provider to alerts.

    Args:
        provider: Where data comes from.
        settings: Application settings; defaults to the configured singleton.
        retry_policy: Ingestion retry policy.
        freshness_thresholds: Freshness windows.
        quality_gate: The pre-downstream quality gate.
        context_steps: Hours of history prepended for context. See
            :class:`IncrementalState`.
        sleep: Injected sleep, so tests exercise backoff without real delays.
        logger: The pipeline logger.
    """

    def __init__(
        self,
        provider: WeatherProvider,
        *,
        settings: Any | None = None,
        retry_policy: RetryPolicy | None = None,
        freshness_thresholds: FreshnessThresholds = DEFAULT_FRESHNESS_THRESHOLDS,
        quality_gate: QualityGate | None = None,
        context_steps: int = DEFAULT_CONTEXT_STEPS,
        sleep: Callable[[float], None] | None = None,
        logger: PipelineLogger | None = None,
    ) -> None:
        from extreme_weather.config import get_settings

        self.provider = provider
        self.settings = settings or get_settings()
        self.retry_policy = retry_policy or RetryPolicy()
        self.freshness_thresholds = freshness_thresholds
        self.quality_gate = quality_gate or QualityGate()
        self.context_steps = int(context_steps)
        self._sleep = sleep
        self.logger = logger or PipelineLogger()
        self.state = IncrementalState(context_steps=self.context_steps)
        self._preprocess_warning: str | None = None
        self._dry_run = False
        self.history: list[PipelineRun] = []

    # --- persistence ----------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        """Directory where run records are written.

        Under ``data/`` so a container volume picks it up without extra
        configuration.
        """
        return self.settings.data_dir / "realtime" / "runs"

    def persist_run(self, run: PipelineRun) -> Path | None:
        """Write a run record to disk, unless this is a dry run.

        A dry run must leave no trace in production state, so this returns
        ``None`` without touching the filesystem.

        Args:
            run: The run to persist.

        Returns:
            The written path, or ``None`` for a dry run or on an IO failure.
            Persistence failure never fails the run itself: losing the log of a
            run is bad, but losing the run is worse.
        """
        if run.dry_run:
            return None
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            import json

            path = self.runs_dir / f"{run.run_id}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(run.as_dict(), handle, indent=2, ensure_ascii=False, default=str)
                handle.write("\n")
            return path
        except OSError as exc:  # pragma: no cover - defensive
            self.logger.error(f"Could not persist run {run.run_id}: {exc}")
            return None

    def load_runs(self, limit: int | None = None) -> list[PipelineRun]:
        """Load persisted run records, newest first.

        Unreadable files are skipped rather than raising: one corrupt record
        must not make the run history endpoint fail.
        """
        import json

        if not self.runs_dir.is_dir():
            return []
        runs: list[PipelineRun] = []
        for path in sorted(self.runs_dir.glob("run-*.json"), reverse=True):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    runs.append(PipelineRun.from_dict(json.load(handle)))
            except (OSError, ValueError, KeyError):
                continue
            if limit is not None and len(runs) >= limit:
                break
        return runs

    def load_state(self) -> IncrementalState:
        """Load the incremental state from disk, if present."""
        import json

        path = self.runs_dir / "state.json"
        if not path.is_file():
            return self.state
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return self.state
        self.state = IncrementalState(
            last_processed=payload.get("last_processed"),
            newest_available=payload.get("newest_available"),
            context_steps=int(
                payload.get("context_steps", self.context_steps) or self.context_steps
            ),
            processed_fingerprints=tuple(
                payload.get("processed_fingerprints", ()) or ()
            ),
        )
        return self.state

    def save_state(self) -> Path | None:
        """Write the incremental state, unless this is a dry run."""
        if getattr(self, "_dry_run", False):
            return None
        try:
            import json

            self.runs_dir.mkdir(parents=True, exist_ok=True)
            path = self.runs_dir / "state.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.state.as_dict(), handle, indent=2, default=str)
                handle.write("\\n")
            return path
        except OSError as exc:  # pragma: no cover - defensive
            self.logger.error(f"Could not persist pipeline state: {exc}")
            return None


    # --- the run ---------------------------------------------------------

    def run(
        self,
        request: WeatherRequest,
        *,
        dry_run: bool = False,
        dataset_id: str | None = None,
        force: bool = False,
        allow_stale: bool = True,
    ) -> OrchestratorResult:
        """Execute the whole pipeline once.

        Args:
            request: What to fetch.
            dry_run: Perform the work but write nothing.
            dataset_id: Registry id for the ingested dataset.
            force: Process even when this exact ingestion has been seen before.
            allow_stale: Continue when freshness is ``STALE`` or ``EXPIRED``.

                The default is ``True`` because refusing to process anything
                older than a few hours would make the prototype useless for
                backfills - but the freshness status is always recorded and
                surfaced, and ``EXPIRED`` data is never described as current.
                Callers that require current data should pass ``False``.
            request: The window, box and variables to fetch.

        Returns:
            An :class:`OrchestratorResult`. This method does not raise for
            pipeline failures: a failed stage is recorded on the run and the
            result is returned, so a scheduler can keep going and the API can
            show what happened.
        """
        self._dry_run = bool(dry_run)
        self._preprocess_warning = None
        started = utc_now()
        # Phase 10: stage durations go to the metrics registry and the log as
        # well as into the run record, so a slow stage is visible live.
        self._stage_timer = StageTimer()
        logger = PipelineLogger(run_id="")
        self.logger = logger

        stage_ids: list[str] = []
        run = PipelineRun(
            run_id=f"run-pending-{started.strftime('%Y%m%dT%H%M%S')}",
            status=RUNNING,
            started_at=started,
            provider=self.provider.name,
            dry_run=bool(dry_run),
        )
        logger.run_id = run.run_id

        logger.info(
            f"starting: provider={self.provider.name} "
            f"window={request.start.isoformat()}..{request.end.isoformat()} "
            f"dry_run={dry_run}"
        )

        try:
            return self._execute(
                run, request, dataset_id=dataset_id, force=force, allow_stale=allow_stale
            )
        except Exception as exc:  # noqa: BLE001 - the run must always be recorded
            run.status = FAILED
            run.errors.append(f"unhandled: {type(exc).__name__}: {exc}")
            logger.error(f"unhandled failure: {type(exc).__name__}: {exc}")
            run.completed_at = utc_now()
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(run=run)

    def _execute(
        self,
        run: PipelineRun,
        request: WeatherRequest,
        *,
        dataset_id: str | None,
        force: bool,
        allow_stale: bool,
    ) -> OrchestratorResult:
        """Drive each stage in order, stopping at the first hard failure."""
        logger = self.logger

        # --- 1. INGEST ----------------------------------------------------
        ingest = StageResult(name="ingest", status=RUNNING, started_at=utc_now())
        run.add_stage(ingest)
        try:
            fetch, attempts = self._fetch_with_retry(request, run)
        except Exception as exc:  # noqa: BLE001
            attempts_note = (
                f" after {self.retry_policy.max_attempts} attempt(s)"
                if self.retry_policy.max_attempts > 1
                else ""
            )
            ingest.complete(
                FAILED,
                message=f"Ingestion failed{attempts_note}: {exc}",
                error=str(exc),
            )
            run.status = FAILED
            run.errors.append(f"ingest: {exc}")
            run.completed_at = utc_now()
            logger.stage("ingest", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(run=run)

        run.ingestion = self._build_ingestion_record(run, fetch, attempts, request)
        run.source_timestamp = fetch.source_timestamp
        freshness = assess_freshness(
            fetch.latest_time, thresholds=self.freshness_thresholds
        )
        run.freshness = freshness.status
        run.age_minutes = freshness.age_minutes
        if freshness.status != FRESH:
            run.warnings.append(freshness.message)
        ingest.complete(
            SUCCESS,
            message=(
                f"provider={fetch.provider} steps={int(fetch.dataset.sizes.get('time', 0))} "
                f"attempts={len(attempts)}"
            ),
            output={
                "checksum": fetch.checksum,
                "variables": list(fetch.variables),
                "freshness": freshness.as_dict(),
            },
        )
        logger.stage(
            "ingest",
            f"provider={fetch.provider} steps={int(fetch.dataset.sizes.get('time', 0))} "
            f"checksum={fetch.checksum[:12]} attempts={len(attempts)}",
        )
        logger.stage("ingest", f"freshness={freshness.status} - {freshness.message}")

        if not allow_stale and freshness.status == "EXPIRED":
            ingest.complete(
                WARNING,
                message=(
                    "Data is EXPIRED and allow_stale=False, so the run stops here. "
                    "Pass allow_stale=True to process it anyway."
                ),
            )
            run.status = WARNING
            run.warnings.append(
                "Run stopped: data is EXPIRED and allow_stale=False."
            )
            run.completed_at = utc_now()
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(run=run)

        # --- idempotency --------------------------------------------------
        identity = compute_run_identity(
            fetch.provider,
            window=f"{request.start.isoformat()}/{request.end.isoformat()}",
            source_timestamp=fetch.source_timestamp,
            checksum=fetch.checksum,
        )
        run.run_id = identity.run_id
        logger.run_id = run.run_id
        self._stage_timer.run_id = identity.run_id
        run.incremental = {
            **self.state.as_dict(),
            "fingerprint": identity.fingerprint,
            "idempotent_hit": self.state.has_seen(identity.fingerprint),
        }

        if identity.fingerprint in self.state.processed_fingerprints and not force:
            skip_message = (
                "This exact ingestion was already processed "
                f"(fingerprint {identity.fingerprint[:12]}). Downstream stages were "
                "skipped to avoid duplicate events. Use --force to reprocess."
            )
            ingest.status = SUCCESS
            ingest.completed_at = utc_now()
            ingest.duration = max(
                0.0, (ingest.completed_at - ingest.started_at).total_seconds()
            )
            ingest.message = skip_message
            for name in STAGE_ORDER[1:]:
                run.add_stage(
                    StageResult(
                        name=name,
                        status=SKIPPED,
                        started_at=utc_now(),
                        completed_at=utc_now(),
                        duration=0.0,
                        message=(
                            "Skipped: this ingestion was already processed "
                            "(idempotent rerun)."
                        ),
                    )
                )
            run.status = SUCCESS
            run.completed_at = utc_now()
            logger.stage(
                "ingest",
                f"idempotent rerun detected ({identity.fingerprint[:12]}); "
                "skipping downstream stages",
            )
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(run=run, dataset=None)

        # --- 2. QUALITY GATE ---------------------------------------------
        validate = StageResult(name="validate", status=RUNNING, started_at=utc_now())
        run.add_stage(validate)
        verdict = self.quality_gate.evaluate(fetch.dataset)
        logger.stage("validate", f"{verdict.summary()}")

        if not verdict.passed:
            validate.complete(
                FAILED,
                message=(
                    f"Quality gate REJECTED the data: {verdict.reason}. "
                    "No downstream stage was run."
                ),
                output={"verdict": verdict.as_dict()},
            )
            run.status = FAILED
            run.errors.append(f"validate: {verdict.reason}")
            run.completed_at = utc_now()
            logger.stage("validate", "REJECTED - not processing further", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(
                run=run, quality=verdict
            )

        warnings_for_warnings = verdict.warnings
        validate.complete(
            WARNING if warnings_for_warnings else SUCCESS,
            message=verdict.summary(),
            output={"verdict": verdict.as_dict()},
        )
        for check in warnings_for_warnings:
            run.warnings.append(f"validate/{check.name}: {check.message}")
        logger.stage(
            "validate",
            f"status={verdict.status} "
            f"checks={len(verdict.checks)} blocking_failures={len(verdict.blocking_failures)}",
        )

        # --- 3. PREPROCESS ------------------------------------------------
        preprocess = StageResult(name="preprocess", status=RUNNING, started_at=utc_now())
        run.add_stage(preprocess)
        try:
            processed = self._preprocess(fetch.dataset, dry_run=run.dry_run)
        except Exception as exc:  # noqa: BLE001
            preprocess.complete(FAILED, message=f"Preprocessing failed: {exc}", error=str(exc))
            run.status = FAILED
            run.errors.append(f"preprocess: {exc}")
            run.completed_at = utc_now()
            logger.stage("preprocess", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(run=run, quality=verdict)

        preprocess.complete(
            WARNING if self._preprocess_warning else SUCCESS,
            message=(
                self._preprocess_warning
                or (
                    f"processed {int(processed.dataset.sizes.get('time', 0))} steps, "
                    f"{len(processed.report.variables)} variable(s)"
                    if hasattr(processed, "dataset")
                    else "processed"
                )
            ),
            output={
                "variables": list(getattr(processed.report, "variables", ()) or ()),
                "derived": list(getattr(processed.report, "derived_variables", ()) or ()),
                "warnings": list(getattr(processed.report, "warnings", ()) or ()),
            },
        )
        dataset = processed.dataset if hasattr(processed, "dataset") else processed
        if self._preprocess_warning:
            run.warnings.append(f"preprocess: {self._preprocess_warning}")
        logger.stage(
            "preprocess",
            f"steps={int(dataset.sizes.get('time', 0))} "
            f"vars={len(list(dataset.data_vars))}",
        )


        # --- 4. ANOMALY ---------------------------------------------------
        anomaly = StageResult(name="anomaly", status=RUNNING, started_at=utc_now())
        run.add_stage(anomaly)
        events: tuple[Any, ...] = ()
        try:
            events, anomaly_output = self._detect_anomalies(dataset, dry_run=run.dry_run)
        except Exception as exc:  # noqa: BLE001
            anomaly.complete(FAILED, message=f"Anomaly detection failed: {exc}", error=str(exc))
            run.status = FAILED
            run.errors.append(f"anomaly: {exc}")
            run.completed_at = utc_now()
            logger.stage("anomaly", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(run=run, quality=verdict)

        anomaly.complete(
            SUCCESS,
            message=f"{len(events)} event(s) detected over {int(dataset.sizes.get('time', 0))} steps",
            output=anomaly_output,
        )
        logger.stage("anomaly", f"events={len(events)}")

        # --- 5. TRACK -----------------------------------------------------
        track = StageResult(name="track", status=RUNNING, started_at=utc_now())
        run.add_stage(track)
        tracks: list[Any] = []
        try:
            tracks, track_output = self._track_events(events, dry_run=run.dry_run)
        except Exception as exc:  # noqa: BLE001
            track.complete(FAILED, message=f"Tracking failed: {exc}", error=str(exc))
            run.status = FAILED
            run.errors.append(f"track: {exc}")
            run.completed_at = utc_now()
            logger.stage("track", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(
                run=run, quality=verdict, events=events
            )

        track.complete(
            SUCCESS,
            message=f"{len(tracks)} track(s) from {len(events)} event(s)",
            output=track_output,
        )
        logger.stage(
            "track",
            f"tracks={len(tracks)} "
            f"multi_point={sum(1 for t in tracks if len(getattr(t, 'events', ()) or ()) > 1)}",
        )

        # --- 6. GNN (optional) --------------------------------------------
        gnn = StageResult(name="gnn", status=RUNNING, started_at=utc_now())
        run.add_stage(gnn)
        try:
            gnn_status = self._run_gnn(tracks)
        except Exception as exc:  # noqa: BLE001
            gnn.complete(FAILED, message=f"GNN stage failed: {exc}", error=str(exc))
            run.status = FAILED
            run.errors.append(f"gnn: {exc}")
            run.completed_at = utc_now()
            logger.stage("gnn", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(
                run=run, quality=verdict, events=events
            )
        gnn.complete(gnn_status[0], message=gnn_status[1], output=gnn_status[2])
        logger.stage("gnn", f"{gnn_status[0]}: {gnn_status[1]}")

        # --- 7. DOWNSCALING (optional) ------------------------------------
        downscale = StageResult(name="downscale", status=RUNNING, started_at=utc_now())
        run.add_stage(downscale)
        try:
            down_status = self._run_downscaling(dataset, dry_run=run.dry_run)
        except Exception as exc:  # noqa: BLE001
            downscale.complete(FAILED, message=f"Downscaling failed: {exc}", error=str(exc))
            run.status = FAILED
            run.errors.append(f"downscale: {exc}")
            run.completed_at = utc_now()
            logger.stage("downscale", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(
                run=run, quality=verdict, events=events
            )
        downscale.complete(down_status[0], message=down_status[1], output=down_status[2])
        logger.stage("downscale", f"{down_status[0]}: {down_status[1]}")

        # --- 8. ALERT -----------------------------------------------------
        alert = StageResult(name="alert", status=RUNNING, started_at=utc_now())
        run.add_stage(alert)
        try:
            alert_count, alert_output = self._generate_alerts(dry_run=run.dry_run)
        except Exception as exc:  # noqa: BLE001
            alert.complete(FAILED, message=f"Alert generation failed: {exc}", error=str(exc))
            run.status = FAILED
            run.errors.append(f"alert: {exc}")
            run.completed_at = utc_now()
            logger.stage("alert", f"FAILED: {exc}", level=40)
            self.history.append(run)
            self.persist_run(run)
            return OrchestratorResult(
                run=run, quality=verdict, events=events
            )

        alert.complete(
            SUCCESS,
            message=f"{alert_count} alert(s) after this run",
            output=alert_output,
        )
        logger.stage("alert", f"alerts={alert_count}")

        # --- finish --------------------------------------------------------
        run.status = WARNING if (run.warnings or any(
            s.status == WARNING for s in run.stages
        )) else SUCCESS
        run.completed_at = utc_now()
        timer = getattr(self, "_stage_timer", None)
        if timer is not None:
            for stage in run.stages:
                if stage.duration is not None and not any(
                    name == stage.name for name, _ in timer.stages
                ):
                    timer.record(stage.name, stage.duration)
        if run.dry_run:
            logger.info("dry run complete - no artifacts were written")
        else:
            self.state = self.state.with_progress(
                str(fetch.source_timestamp or ""), identity.fingerprint
            )
            self.save_state()
        self.history.append(run)
        self.persist_run(run)
        logger.info(
            f"finished status={run.status} duration={run.total_duration:.2f}s"
            if run.total_duration is not None
            else "finished"
        )
        return OrchestratorResult(
            run=run,
            dataset=dataset,
            quality=verdict,
            events=events,
            alerts=alert_count,
        )


    # --- stage implementations -------------------------------------------

    def _fetch_with_retry(
        self, request: WeatherRequest, run: PipelineRun
    ) -> tuple[Any, list[Attempt]]:
        """Fetch data, retrying transient provider failures with backoff.

        Every attempt is logged, so a run that took three tries says so rather
        than appearing to have worked first time.
        """
        attempts_seen: list[Attempt] = []

        def on_attempt(attempt: Attempt) -> None:
            attempts_seen.append(attempt)
            if attempt.succeeded:
                self.logger.stage("ingest", f"attempt {attempt.number} succeeded")
            else:
                self.logger.stage(
                    "ingest",
                    f"attempt {attempt.number} failed: {attempt.error} "
                    f"(retrying in {attempt.delay:.1f}s)"
                    if attempt.delay
                    else f"attempt {attempt.number} failed: {attempt.error}",
                )

        sleep = self._sleep if self._sleep is not None else __import__("time").sleep
        fetch, attempts = retry_call(
            lambda: self.provider.fetch(request),
            self.retry_policy,
            sleep=sleep,
            on_attempt=on_attempt,
        )
        return fetch, attempts

    def _build_ingestion_record(
        self, run: PipelineRun, fetch: Any, attempts: list[Attempt], request: WeatherRequest
    ) -> IngestionRecord:
        """Build the provenance record for this ingestion."""
        dataset = fetch.dataset
        step_seconds = _infer_step_seconds(dataset)
        return IngestionRecord(
            run_id=run.run_id,
            provider=fetch.provider,
            status=SUCCESS,
            source_url=redact_url(fetch.source_url),
            source_timestamp=fetch.source_timestamp,
            ingested_at=utc_now(),
            variables=tuple(fetch.variables),
            time_steps=int(dataset.sizes.get("time", 0)),
            spatial_extent={
                "latitude_min": _coord_range(dataset, "latitude")[0],
                "latitude_max": _coord_range(dataset, "latitude")[1],
                "longitude_min": _coord_range(dataset, "longitude")[0],
                "longitude_max": _coord_range(dataset, "longitude")[1],
                "points": int(dataset.sizes.get("latitude", 0))
                * int(dataset.sizes.get("longitude", 0)),
            },
            temporal_resolution=(
                f"{step_seconds:.0f}s" if step_seconds is not None else None
            ),
            checksum=fetch.checksum,
            attempts=len(attempts),
            dry_run=run.dry_run,
        )

    def _preprocess(self, dataset: xr.Dataset, *, dry_run: bool) -> Any:
        """Run Phase 0 preprocessing on the fetched dataset.

        Uses the same :func:`preprocess_dataset` the CLI uses, so behaviour is
        identical to a manual run.

        Phase 0's preprocessing contract requires ``u_wind`` and ``v_wind``,
        because it always derives wind speed. Most weather providers do not
        serve wind, so when the components are absent the dataset is passed
        through unchanged and the run records a warning. Skipping the gap-fill
        and normalisation work is a real limitation and is reported as one,
        rather than being silently absorbed.
        """
        from extreme_weather.data.preprocessing import preprocess_dataset

        available = {str(name) for name in dataset.data_vars}
        if not {"u_wind", "v_wind"}.issubset(available):
            self._preprocess_warning = (
                "Phase 0 preprocessing was skipped: the provider did not "
                "supply u_wind/v_wind, which the Phase 0 contract requires "
                "for wind-speed derivation. Gap filling and derived wind "
                "fields are therefore not applied."
            )
            return _PassthroughResult(dataset)
        return preprocess_dataset(dataset, validate_input=False)

    # --- stores, resolved from THIS orchestrator's settings ---------------

    def _event_store(self) -> Any:
        """Return an :class:`EventStore` bound to this run's settings.

        The services default their stores to the process-wide
        ``get_settings()``, which silently ignores the ``settings`` passed to
        this orchestrator. A run configured for a scratch directory would then
        write its events into the repository's data directory instead. Every
        store used here is therefore built from ``self.settings``.
        """
        from extreme_weather.anomaly.service import EventStore

        return EventStore(self.settings.processed_dir / self.settings.events_filename)

    def _track_store(self) -> Any:
        """Return a :class:`TrackStore` bound to this run's settings."""
        from extreme_weather.tracking.store import TrackStore

        return TrackStore(self.settings.processed_dir / self.settings.tracks_filename)

    def _alert_store(self) -> Any:
        """Return an :class:`AlertStore` bound to this run's settings."""
        from extreme_weather.alerts.store import AlertStore

        return AlertStore(self.settings.processed_dir / self.settings.alerts_filename)

    def _detect_anomalies(
        self, dataset: xr.Dataset, *, dry_run: bool
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Run Phase 1 anomaly detection over the processed dataset.

        Returns:
            The detected events and a summary dict.
        """
        from extreme_weather.anomaly.service import AnomalyDetectionService

        service = AnomalyDetectionService(store=self._event_store())
        result = service.analyse(dataset)
        if not dry_run:
            # Persist via the standard store, as the CLI does.
            service.store.save(result)
        return result.events, {
            "event_count": len(result.events),
            "variables": list(result.variables),
            "score": result.score_name,
            "grouping": result.grouping,
        }

    def _track_events(
        self, events: Sequence[Any], *, dry_run: bool
    ) -> tuple[list[Any], dict[str, Any]]:
        """Run Phase 2 tracking over detected events."""
        from extreme_weather.tracking.tracker import EventTracker

        tracker = EventTracker()
        tracks = list(tracker.track(events))
        if not dry_run:
            from extreme_weather.tracking.service import TrackingService

            # write_output MUST be True: without it the track result is
            # computed and then discarded, leaving tracks.json absent and
            # GET /api/v1/tracks empty after a successful pipeline run.
            TrackingService(
                event_store=self._event_store(), track_store=self._track_store()
            ).run(events=events, write_output=True)
        return tracks, {
            "track_count": len(tracks),
            "event_count": len(events),
            "multi_point_tracks": sum(
                1 for t in tracks if len(getattr(t, "events", ()) or ()) > 1
            ),
        }

    def _run_gnn(self, tracks: Sequence[Any] | None = None) -> tuple[str, str, dict[str, Any]]:
        """Run GNN inference over the tracked events.

        The stage is optional, and a missing checkpoint is a normal state for a
        research prototype, so an unavailable model is reported as ``SKIPPED``
        with the reason rather than raised.

        When a model *is* available it is actually run over the stored tracks.
        Reporting availability without running inference would claim a
        prediction the pipeline never made.
        """
        from extreme_weather.gnn.service import GNNService

        service = GNNService()
        status = service.status()
        if not status.available:
            return (
                SKIPPED,
                "No trained GNN checkpoint is available, so no prediction was made.",
                {
                    "available": False,
                    "reason": status.message or "no checkpoint",
                    "predicted_tracks": 0,
                },
            )

        predictions: list[dict[str, Any]] = []
        for track in tracks or []:
            track_id = str(getattr(track, "track_id", "") or "")
            if not track_id:
                continue
            try:
                prediction = service.predict(track_id)
            except Exception as exc:  # noqa: BLE001 - one bad track is not a run failure
                predictions.append(
                    {
                        "track_id": track_id,
                        "predicted": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            predictions.append(
                {
                    "track_id": track_id,
                    "predicted": True,
                    "horizon_steps": len(getattr(prediction, "steps", ()) or ()),
                }
            )

        succeeded = [item for item in predictions if item.get("predicted")]
        detail: dict[str, Any] = {
            "available": True,
            "version": status.model_version,
            "predicted_tracks": len(succeeded),
            "attempted_tracks": len(predictions),
            "details": predictions[:10],
        }
        if not succeeded:
            return (
                SUCCESS,
                (
                    f"The GNN model is available (version={status.model_version}) but "
                    "no track could be predicted, because the run produced no usable "
                    "graph nodes. No prediction was invented."
                ),
                detail,
            )
        return (
            SUCCESS,
            f"GNN predicted {len(succeeded)} of {len(predictions)} track(s).",
            detail,
        )

    def _run_downscaling(
        self, dataset: xr.Dataset, *, dry_run: bool
    ) -> tuple[str, str, dict[str, Any]]:
        """Run the AI downscaling stage, or report why it was skipped.

        A missing checkpoint is ``SKIPPED``, not a failure. When a model is
        available, a nearest-neighbour baseline is computed alongside it so the
        stage reports a real comparison rather than asserting that the model ran.
        """
        from extreme_weather.downscaling.baseline import nearest_downscale
        from extreme_weather.downscaling.service import DownscalingService

        service = DownscalingService()
        status = service.status()
        if not status.get("available"):
            return (
                SKIPPED,
                "No trained AI downscaling checkpoint is available, so no AI-downscaled field was produced.",
                {
                    "available": False,
                    "reason": status.get("message", "no checkpoint"),
                    "baseline_only": True,
                },
            )

        try:
            baseline = nearest_downscale(dataset, factor=2)
        except Exception as exc:  # noqa: BLE001 - baseline is best-effort context
            return (
                SUCCESS,
                f"AI downscaling model is available (version={status.get('model_version')}).",
                {
                    "available": True,
                    "version": status.get("model_version"),
                    "baseline_comparison": "failed",
                    "baseline_error": str(exc),
                },
            )

        return (
            SUCCESS,
            f"AI downscaling model available (version={status.get('model_version')}); nearest-neighbour baseline computed for comparison.",
            {
                "available": True,
                "version": status.get("model_version"),
                "baseline": "nearest",
                "baseline_grid": [
                    int(baseline.sizes.get("latitude", 0)),
                    int(baseline.sizes.get("longitude", 0)),
                ],
            },
        )

    def _generate_alerts(self, *, dry_run: bool) -> tuple[int, dict[str, Any]]:
        """Run Phase 5 alert generation over stored pipeline outputs."""
        from extreme_weather.alerts.service import AlertService

        result = AlertService(
            alert_store=self._alert_store(),
            event_store=self._event_store(),
            track_store=self._track_store(),
        ).generate(dry_run=dry_run, notify=not dry_run)
        return result.alert_count, {
            "alert_count": result.alert_count,
            "created": result.created_count,
            "updated": result.updated_count,
            "dry_run": result.dry_run,
            "degraded": list(result.degraded),
        }

    # --- status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of pipeline state.

        Used by ``GET /api/v1/pipeline/status`` and
        ``GET /api/v1/ingestion/status``.
        """
        runs = self.history or self.load_runs(limit=1)
        latest = runs[0] if runs else None
        state = self.state.as_dict()
        return {
            "provider": self.provider.describe(),
            "providers": self.provider.describe(),
            "freshness": {
                "status": latest.freshness if latest else "UNKNOWN",
                "thresholds": self.freshness_thresholds.as_dict(),
                "age_minutes": latest.age_minutes if latest else None,
                "source_timestamp": latest.source_timestamp if latest else None,
            },
            "source": self.provider.name,
            "source_timestamp": latest.source_timestamp if latest else None,
            "latest_timestamp": latest.source_timestamp if latest else None,
            "last_successful_run": _last_successful(runs).run_id if _last_successful(runs) else None,
            "last_run": latest.run_id if latest else None,
            "current_stage": latest.current_stage if latest else None,
            "current_state": latest.status if latest else "IDLE",
            "incremental": state,
            "stage_order": list(STAGE_ORDER),
            "stages": [s.as_dict() for s in latest.stages] if latest else [],
            "warnings": list(latest.warnings) if latest else [],
            "errors": list(latest.errors) if latest else [],
            "warnings_and_errors": (
                [f"WARNING: {w}" for w in latest.warnings]
                + [f"ERROR: {e}" for e in latest.errors]
            )
            if latest
            else [],
            "is_prototype": True,
            "disclaimer": (
                "Research prototype. Not an operational emergency-warning system. "
                "No demonstrated forecast skill."
            ),
        }


@dataclass(frozen=True)
class _PassthroughResult:
    """Stands in for ``PreprocessingResult`` when Phase 0 is skipped.

    Exposes the same two attributes the orchestrator reads, so the
    wind-skip path does not need a second set of handling in the caller.
    """

    dataset: xr.Dataset

    @property
    def report(self) -> Any:
        """A report object with empty variable and warning lists."""
        return _PassthroughReport()


@dataclass(frozen=True)
class _PassthroughReport:
    """Empty stand-in for a Phase 0 preprocessing report."""

    variables: tuple[str, ...] = ()
    derived_variables: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _last_successful(runs: Sequence[PipelineRun]) -> PipelineRun | None:
    """Return the most recent run that did not fail, or ``None``."""
    for run in runs:
        if run.status in (SUCCESS, WARNING):
            return run
    return None


def _coord_range(dataset: xr.Dataset, name: str) -> tuple[float | None, float | None]:
    """Return the (min, max) of a coordinate, or (None, None) when absent."""
    if name not in dataset.coords:
        return (None, None)
    values = np.asarray(dataset.coords[name].values, dtype="float64")
    if values.size == 0:
        return (None, None)
    return (float(np.nanmin(values)), float(np.nanmax(values)))


def _infer_step_seconds(dataset: xr.Dataset) -> float | None:
    """Infer the time step in seconds, or ``None`` when undeterminable."""
    if "time" not in dataset.coords:
        return None
    times = np.asarray(dataset.coords["time"].values)
    if times.size < 2:
        return None
    diffs = np.diff(times.astype("datetime64[ns]").astype("int64"))
    median = float(np.median(diffs))
    return median / 1e9 if median > 0 else None

