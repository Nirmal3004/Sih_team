"""Project health endpoint (SIH26078, Phase 11).

```text
GET /api/v1/system/status
```

This is the one request a judge, a reviewer or an operator makes first: *what
is this system, what has it actually done, and can its output be trusted right
now?*

Design rules, all of which are enforced by tests:

**Compose, do not duplicate.** Every value is read from the service that owns
it - the readiness builder for the verdict, the GNN and downscaling services
for model availability, the track and alert stores for what has been produced.
There is no second source of truth to drift out of sync.

**Unavailable is ``null``, never ``0``.** "No trained checkpoint" and "a
trained checkpoint that scored zero" are different facts, and a dashboard that
cannot tell them apart is lying by accident. Where nothing has been measured,
the field is ``null`` and a reason is given.

**This says nothing about forecast skill.** It reports what the system *is*
and what it has *done*. It reports no accuracy, no precision, no lives saved
and no economic benefit, because none has been measured. The disclaimer is a
required field, not decoration.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from extreme_weather.config import (
    PACKAGE_VERSION,
    PROJECT_ID,
    PROJECT_NAME,
    Settings,
    get_settings,
)

router = APIRouter(prefix="/api/v1/system", tags=["system"])

#: The project phase this deployment represents. Reported verbatim so a stale
#: build cannot quietly present itself as a later phase.
PROJECT_PHASE = "11"

#: Attached to every response. A client cannot display this page as
#: operational guidance without the caveat travelling with it.
DISCLAIMER = (
    "Research prototype (SIH26078). Not an operational disaster-warning or "
    "emergency-response system, and not validated against observed weather or "
    "an official warning catalogue. It reports system state only: no forecast "
    "skill, accuracy, lives saved or economic benefit has been measured, and "
    "none is implied by anything on this page."
)

#: Where the numbers on this page come from, so a reader is never left
#: wondering whether a figure was measured or estimated.
MEASUREMENT_NOTE = (
    "Counts below are file-based facts: how many events, tracks, alerts and "
    "runs are actually stored. They are NOT accuracy figures. A model whose "
    "availability is false is reported as unavailable, and the dependent stage "
    "reports SKIPPED rather than substituting a default."
)


class ComponentState(BaseModel):
    """One subsystem's availability, with the reason it is in that state."""

    model_config = ConfigDict(extra="allow")

    name: str
    available: bool
    state: str
    detail: str = ""
    #: Why the component is unavailable, or ``None`` when it is available.
    reason: str | None = None


class DataSourceInfo(BaseModel):
    """What kind of data this deployment is looking at.

    The distinction is load-bearing: a synthetic field and a reanalysis field
    are not interchangeable, and calling one the other is a correctness bug,
    not a presentation detail.
    """

    model_config = ConfigDict(extra="allow")

    #: ``SYNTHETIC``, ``HISTORICAL`` or ``REALTIME_FIXTURE``.
    kind: str
    provider: str | None = None
    description: str
    #: Newest timestamp in the stored data, or ``None`` when nothing is stored.
    latest_timestamp: str | None = None
    #: Freshness verdict, or ``None`` when it cannot be determined.
    freshness: str | None = None


class Counts(BaseModel):
    """How much the system has actually produced.

    Every field is ``int | None``: ``None`` means "not measured / nothing
    stored", which is deliberately distinct from ``0``.
    """

    model_config = ConfigDict(extra="allow")

    events: int | None = None
    tracks: int | None = None
    alerts: int | None = None
    datasets: int | None = None
    pipeline_runs: int | None = None
    evaluation_runs: int | None = None


class SystemStatusResponse(BaseModel):
    """The payload returned by ``GET /api/v1/system/status``."""

    model_config = ConfigDict(extra="allow")

    project: str
    service: str
    version: str
    phase: str
    environment: str
    #: The readiness verdict, reused verbatim from the operations router so the
    #: two endpoints cannot disagree.
    readiness: str
    ready: bool
    data_source: DataSourceInfo
    ingestion: ComponentState
    counts: Counts
    models: dict[str, ComponentState]
    alerting: ComponentState
    evaluation: ComponentState
    disclaimer: str = DISCLAIMER
    note: str = MEASUREMENT_NOTE


def _collection_count(path, loader, attribute: str) -> int | None:
    """Count one collection on a stored result, or ``None`` when unavailable.

    ``None`` rather than ``0`` matters: a missing file means "this has not been
    run", which is not the same claim as "this ran and found nothing". A
    corrupt file is likewise "unknown", never a confident zero.

    The three stores differ in shape - ``load()`` returns a result object whose
    ``events``/``tracks``/``alerts`` attribute holds the collection - so the
    attribute is named rather than guessed.
    """
    try:
        if not path.exists():
            return None
        result = loader()
        if result is None:
            return None
        return len(getattr(result, attribute, None) or [])
    except Exception:  # noqa: BLE001 - a corrupt store is "unknown", not zero
        return None


def _read_alerts(path) -> list[Any]:
    """Read the stored alerts through the Phase 5 store, not by re-parsing JSON.

    The store already handles a truncated write and a non-object payload, so
    this reuses that behaviour rather than duplicating a weaker copy of it.
    """
    from extreme_weather.alerts.store import AlertStore

    result = AlertStore(path).load()
    return list(getattr(result, "alerts", None) or []) if result is not None else []


def _data_source(settings: Settings) -> DataSourceInfo:
    """Describe the data this deployment is looking at.

    Reads the newest pipeline run's provenance when one exists, and otherwise
    reports honestly that nothing has been ingested yet.
    """
    from extreme_weather.realtime.models import EXPIRED, FRESH, STALE

    try:
        runs_dir = settings.data_dir / "realtime" / "runs"
        if runs_dir.is_dir():
            run_files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if run_files:
                import json

                payload = json.loads(run_files[-1].read_text(encoding="utf-8"))
                provider = str(payload.get("provider") or "") or None
                # The fixture provider is deterministic synthetic data, and
                # labelling it "realtime" would be a false claim.
                kind = "SYNTHETIC" if provider == "fixture" else "HISTORICAL"
                description = (
                    "Deterministic synthetic fixture generated locally. This is "
                    "not observed weather and not a forecast."
                    if kind == "SYNTHETIC"
                    else (
                        f"Retrieved from the '{provider}' provider: reanalysis "
                        "or archived data, not a live feed."
                    )
                )
                freshness = payload.get("freshness")
                if freshness not in (FRESH, STALE, EXPIRED):
                    freshness = None
                return DataSourceInfo(
                    kind=kind,
                    provider=provider,
                    description=description,
                    latest_timestamp=payload.get("source_timestamp"),
                    freshness=freshness,
                )
    except Exception:  # noqa: BLE001 - status must not fail on a bad record
        pass

    return DataSourceInfo(
        kind="SYNTHETIC",
        provider=None,
        description=(
            "No pipeline run on record. This project generates its own "
            "deterministic synthetic data by default; it does not ingest a "
            "live feed."
        ),
        latest_timestamp=None,
        freshness=None,
    )


def _ingestion_state(settings: Settings) -> ComponentState:
    """Report the state of ingestion: artifacts on record, or nothing yet.

    Any stored pipeline artifact counts as evidence that ingestion ran. The
    Phase 1/2/5 stores are included because they are written by the ingestion
    run itself; looking only at manifests would report "no data" for a
    deployment that has events on disk.
    """
    try:
        manifests = settings.manifests_dir
        registered = len(list(manifests.glob("*.json"))) if manifests.is_dir() else 0
    except OSError:
        registered = 0

    runs_dir = settings.data_dir / "realtime" / "runs"
    try:
        has_run = runs_dir.is_dir() and any(runs_dir.glob("*.json"))
    except OSError:
        has_run = False

    stored_outputs = [
        path
        for path in (
            settings.processed_dir / settings.events_filename,
            settings.processed_dir / settings.tracks_filename,
            settings.processed_dir / settings.alerts_filename,
        )
        if path.is_file()
    ]

    if has_run or registered or stored_outputs:
        sources = []
        if registered:
            sources.append(f"{registered} dataset manifest(s)")
        if stored_outputs:
            sources.append(f"{len(stored_outputs)} pipeline artifact(s)")
        if has_run:
            sources.append("a pipeline run record")
        return ComponentState(
            name="ingestion",
            available=True,
            state="READY",
            detail="Data on record: " + ", ".join(sources) + ".",
        )
    return ComponentState(
        name="ingestion",
        available=False,
        state="NO_DATA",
        detail="No ingested dataset and no pipeline run on record yet.",
        reason=(
            "Nothing has been ingested. Run scripts/generate_sample_data.py, or "
            "scripts/run_sih_demo.py, to produce data."
        ),
    )


def _alerting_state(settings: Settings) -> ComponentState:
    """Report alerting availability, and whether generation is switched on."""
    from extreme_weather.config import get_settings

    configured = get_settings()
    enabled = bool(getattr(configured, "alert_generation_enabled", True))
    store = settings.processed_dir / configured.alerts_filename
    count = _collection_count(store, lambda: _read_alerts(store), "alerts")

    if not enabled:
        return ComponentState(
            name="alerting",
            available=False,
            state="DISABLED",
            detail="Alert generation is switched off by configuration.",
            reason=(
                "ALERT_GENERATION_ENABLED is false, so the API returns 503 "
                "rather than silently generating nothing."
            ),
        )
    return ComponentState(
        name="alerting",
        available=True,
        state="READY",
        detail=(
            f"Alert engine available; {count} alert(s) stored."
            if count is not None
            else "Alert engine available; no alert file on record yet."
        ),
    )


def _evaluation_state(settings: Settings) -> ComponentState:
    """Report whether any evaluation run is on record."""
    runs_dir = settings.data_dir / "evaluation"
    try:
        has_run = runs_dir.is_dir() and any(runs_dir.glob("*.json"))
    except OSError:
        has_run = False

    if has_run:
        return ComponentState(
            name="evaluation",
            available=True,
            state="MEASURED",
            detail=(
                "An evaluation run is on record. Its metrics carry provenance, "
                "and anything it could not compute is reported as unavailable "
                "rather than as zero."
            ),
        )
    return ComponentState(
        name="evaluation",
        available=False,
        state="NOT_EVALUATED",
        detail="No evaluation run on record.",
        reason=(
            "No evaluation has been run, so no accuracy, skill or uncertainty "
            "figure exists for this deployment. Nothing is reported in its place."
        ),
    )


def build_system_status(settings: Settings) -> SystemStatusResponse:
    """Assemble the system status payload.

    Args:
        settings: The active settings.

    Returns:
        A :class:`SystemStatusResponse`. Every optional service is probed
        defensively: an unavailable service is reported as unavailable rather
        than allowed to fail the endpoint, because a status page that 500s is
        useless exactly when something is wrong.
    """
    from extreme_weather.api.routers.operations import _build_readiness
    from extreme_weather.anomaly.service import EventStore
    from extreme_weather.evaluation.evaluator import code_fingerprint
    from extreme_weather.tracking.store import TrackStore

    readiness = _build_readiness(settings)
    components = {c.name: c for c in readiness.components}

    def _component(name: str) -> ComponentState:
        source = components.get(name)
        if source is None:
            return ComponentState(
                name=name,
                available=False,
                state="UNKNOWN",
                detail="Not reported by the readiness check.",
                reason="No readiness component with this name.",
            )
        return ComponentState(
            name=name,
            available=source.available,
            state=source.state,
            detail=source.detail,
            reason=None if source.available else source.detail,
        )

    try:
        files = list(settings.processed_dir.glob("*.nc")) + list(
            settings.samples_dir.glob("*.nc")
        )
        datasets: int | None = len(files)
    except OSError:
        datasets = None

    runs_dir = settings.data_dir / "realtime" / "runs"
    try:
        pipeline_runs = len(list(runs_dir.glob("*.json"))) if runs_dir.is_dir() else None
    except OSError:
        pipeline_runs = None

    evaluation_dir = settings.data_dir / "evaluation"
    try:
        evaluation_runs = (
            len(list(evaluation_dir.glob("*.json"))) if evaluation_dir.is_dir() else None
        )
    except OSError:
        evaluation_runs = None

    # The alert store's `load()` already tolerates a truncated write, and
    # `_collection_count` maps an unreadable file to None rather than 0.
    alerts_file = settings.processed_dir / settings.alerts_filename
    alert_count = _collection_count(
        alerts_file, lambda: _read_alerts(alerts_file), "alerts"
    )

    payload = SystemStatusResponse(
        project=PROJECT_ID,
        service=PROJECT_NAME,
        version=PACKAGE_VERSION,
        phase=PROJECT_PHASE,
        environment=settings.app_env,
        readiness=readiness.state,
        ready=readiness.ready,
        data_source=_data_source(settings),
        ingestion=_ingestion_state(settings),
        counts=Counts(
            events=_collection_count(
                settings.processed_dir / settings.events_filename,
                lambda: EventStore(
                    settings.processed_dir / settings.events_filename
                ).load(),
                "events",
            ),
            tracks=_collection_count(
                settings.processed_dir / settings.tracks_filename,
                lambda: TrackStore(
                    settings.processed_dir / settings.tracks_filename
                ).load(),
                "tracks",
            ),
            alerts=alert_count,
            datasets=datasets,
            pipeline_runs=pipeline_runs,
            evaluation_runs=evaluation_runs,
        ),
        models={
            "gnn": _component("gnn"),
            "downscaling": _component("downscaling"),
        },
        alerting=_alerting_state(settings),
        evaluation=_evaluation_state(settings),
    )
    # Provenance travels with the payload so a stored record can be traced to
    # the code that produced it. Failing to fingerprint is not a failure of the
    # status page, so it degrades to None rather than raising.
    try:
        payload.code_fingerprint = code_fingerprint()
    except Exception:  # noqa: BLE001
        payload.code_fingerprint = None
    return payload


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Project health and demo status (Phase 11)",
    description=(
        "Identity, environment, readiness, data provenance, what the system "
        "has actually produced, and whether each optional model is available. "
        "Unavailable quantities are reported as null with a reason and are "
        "never reported as zero. Carries no accuracy or forecast-skill claim."
    ),
)
def system_status(settings: Settings = Depends(get_settings)) -> SystemStatusResponse:
    """Return the overall system status.

    Raises:
        Nothing: an unavailable optional service is reported in the payload
        rather than raised, so this endpoint answers even on a partly broken
        deployment.
    """
    return build_system_status(settings)
