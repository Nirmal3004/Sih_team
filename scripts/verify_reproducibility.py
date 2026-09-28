#!/usr/bin/env python3
"""Verify that the deterministic parts of this project are reproducible.

SIH26078, Phase 11. Runs the deterministic components twice into two isolated
scratch directories and compares the artifacts that are *promised* to be
identical.

What is compared, and why
-------------------------
**Compared**: event ids, track ids, the graph dataset's structure, the
evaluation run id, alert ids, and the provenance metadata. These are derived
from data and configuration through seeded, order-stable code paths, so a
difference between two runs of identical inputs is a real defect.

**Deliberately not compared**: wall-clock timestamps, durations, and
``generated_at``-style fields. Those are *intended* to differ between runs;
requiring them to match would be asserting a falsehood and would make this
check fail for the wrong reason.

Each check reports ``PASS``, ``FAIL`` or ``NOT_APPLICABLE``. ``NOT_APPLICABLE``
means the input the check needs was not produced - for example, a graph cannot
be built when no track was detected. It is not a pass and not a failure; it is
an honest "this could not be checked", and the summary says so.

Exit codes: ``0`` every applicable check passed, ``1`` at least one failed.

This project remains a **research prototype**. Reproducibility of an artifact
is a statement about the code, not about forecast skill, and none is claimed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PASS = "PASS"
FAIL = "FAIL"
NA = "NOT_APPLICABLE"

#: Fields that are *expected* to vary between runs. Stripped before comparing.
#: Anything listed here must have a reason, or the check becomes theatre.
VOLATILE_FIELDS: frozenset[str] = frozenset(
    {
        "generated_at",
        "started_at",
        "completed_at",
        "ingested_at",
        "retrieved_at",
        "timestamp",
        "duration",
        "duration_seconds",
        "total_duration",
        "elapsed",
        "wall_clock",
        "host",
        "hostname",
        "python_version",
        "platform",
        "created_at",
        "updated_at",
        "modified_at",
        "resolved_at",
        "acknowledged_at",
        # Phase 5: an alert's validity window is `created_at` plus a
        # configured duration, so it moves with the wall clock exactly as
        # `created_at` does. The alert *identity* is still compared.
        "valid_from",
        "valid_until",
        # Phase 2: the tracker records the absolute path of the events file it
        # read. Two runs use two different scratch directories, so this is the
        # run location, not a property of the result.
        "event_source",
    }
)


@dataclass
class Check:
    """One reproducibility verdict."""

    name: str
    status: str
    detail: str = ""
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only for a real pass."""
        return self.status == PASS

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "values": self.values,
        }


def strip_volatile(value: Any) -> Any:
    """Recursively drop fields that are expected to differ between runs.

    Keys are matched case-insensitively, so ``generatedAt`` is caught too.
    """
    if isinstance(value, dict):
        return {
            k: strip_volatile(v)
            for k, v in value.items()
            if str(k).strip().lower() not in VOLATILE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [strip_volatile(v) for v in value]
    return value


def _load_json(path: Path) -> Any:
    """Read a JSON artifact, returning ``None`` when it is absent."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ids(payload: Any, key: str) -> list[str] | None:
    """Extract an ordered list of ids from a stored record, or ``None``."""
    if not isinstance(payload, dict):
        return None
    items = payload.get(key)
    if not isinstance(items, list):
        return None
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("event_id") or item.get("track_id") or item.get("alert_id")
            if isinstance(value, str):
                out.append(value)
    return out


def _run_once(workdir: Path) -> tuple[_RunArtifacts, Path]:
    """Run the deterministic pipeline once into an isolated directory.

    Uses the same code path the demo uses, with a scratch ``DATA_DIR`` so the
    repository's own data is neither read nor written.

    The bundled sample dataset is preferred over the smooth fixture because it
    contains an injected extreme event. Without one, anomaly detection finds
    nothing and every interesting check degrades to NOT_APPLICABLE - which
    would make this script pass while verifying almost nothing.

    Returns:
        The collected artifacts and the path of the stored tracks file.
    """
    from extreme_weather.config import Settings

    data_dir = workdir / "data"
    model_dir = workdir / "models"
    settings = Settings(data_dir=data_dir, model_dir=model_dir)
    settings.ensure_directories()

    from datetime import datetime, timedelta, timezone

    from extreme_weather.realtime.orchestrator import PipelineOrchestrator
    from extreme_weather.realtime.providers import WeatherRequest, build_provider

    sample = ROOT / "data" / "raw" / "sample_history.nc"
    orchestrator = PipelineOrchestrator(
        build_provider("local", local_path=sample) if sample.is_file()
        else build_provider("fixture"),
        settings=settings,
    )
    # A fixed window and a fixed dataset make the input identical between the
    # two runs, which is the precondition for the whole check. Variables and a
    # bounding box are mandatory: a provider refuses an unbounded request
    # rather than downloading an arbitrary region.
    end = datetime(2024, 6, 3, tzinfo=timezone.utc)
    request = WeatherRequest(
        start=end - timedelta(hours=6),
        end=end,
        variables=("precipitation", "temperature"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )
    if sample.is_file():
        request = _request_spanning(sample) or request

    result = orchestrator.run(request, dry_run=False, force=True)

    events_path = settings.processed_dir / settings.events_filename
    tracks_path = settings.processed_dir / settings.tracks_filename
    alerts_path = settings.processed_dir / settings.alerts_filename
    events_payload = _load_json(events_path)
    tracks_payload = _load_json(tracks_path)
    alerts_payload = _load_json(alerts_path)

    artifacts = _RunArtifacts(
        events=_ids(events_payload, "events"),
        tracks=_ids(tracks_payload, "tracks"),
        alerts=_ids(alerts_payload, "alerts"),
        run_id=result.run.run_id,
        events_payload=events_payload or {},
        tracks_payload=tracks_payload,
        alerts_payload=alerts_payload,
    )
    return artifacts, tracks_path


def _request_spanning(path: Path) -> Any:
    """Build a request covering a local dataset's own time span, or ``None``.

    A fixed window would exclude most of a real dataset, so the window is read
    from the file. Returns ``None`` when the span cannot be determined.
    """
    from datetime import datetime, timezone

    import xarray as xr

    from extreme_weather.realtime.providers import WeatherRequest

    try:
        with xr.open_dataset(path) as dataset:
            times = dataset.coords["time"].values
    except Exception:  # noqa: BLE001 - the caller keeps its default window
        return None
    if times is None or len(times) == 0:
        return None
    start = times[0].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
    end = times[-1].astype("datetime64[s]").astype(datetime).replace(tzinfo=timezone.utc)
    # Variables and a bounding box stay set: WeatherRequest refuses an empty
    # variable list or an unbounded box, by design.
    return WeatherRequest(
        start=start,
        end=end,
        variables=("precipitation", "temperature"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )


@dataclass
class _RunArtifacts:
    """What one pipeline run produced.

    Attributes:
        events: Ordered event ids, or ``None`` when none were stored.
        tracks: Ordered track ids, or ``None`` when none were stored.
        alerts: Ordered alert ids, or ``None`` when none were stored.
        run_id: The pipeline's run id. Derived from the data by design, so it
            *is* expected to be identical between identical runs.
        events_payload: The full stored events record, for field comparison.
        tracks_payload: The full stored tracks record.
        alerts_payload: The full stored alerts record.
    """

    events: list[str] | None
    tracks: list[str] | None
    alerts: list[str] | None
    run_id: str | None
    events_payload: dict[str, Any]
    tracks_payload: dict[str, Any] | None
    alerts_payload: dict[str, Any] | None


def _compare_ids(
    name: str, first: list[str] | None, second: list[str] | None
) -> Check:
    """Compare two ordered id lists from independent runs."""
    if first is None or second is None:
        return Check(
            name=name,
            status=NA,
            detail="Not produced by these runs, so there was nothing to compare.",
        )
    if first == second:
        return Check(
            name=name,
            status=PASS,
            detail=f"{len(first)} id(s) identical across both runs.",
            values={"count": len(first)},
        )
    only_first = [i for i in first if i not in second]
    only_second = [i for i in second if i not in first]
    return Check(
        name=name,
        status=FAIL,
        detail=(
            f"Ids differ between identical runs. Only in run 1: "
            f"{only_first[:3] or 'none'}. Only in run 2: "
            f"{only_second[:3] or 'none'}."
        ),
        values={"run1": len(first), "run2": len(second)},
    )


def _deep_diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Return the dotted paths at which two structures differ.

    A top-level "metadata differs" is useless to the person fixing it; the
    exact leaf is what tells them whether it is a real defect or an expected
    volatile field that escaped the filter.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            child = f"{path}.{key}" if path else str(key)
            out.extend(_deep_diff(a.get(key), b.get(key), child))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}[] (length {len(a)} vs {len(b)})"]
        out = []
        for index, (x, y) in enumerate(zip(a, b)):
            out.extend(_deep_diff(x, y, f"{path}[{index}]"))
        return out
    if a != b:
        return [path or "<root>"]
    return []


def _compare_payload(
    name: str, first: Any, second: Any
) -> Check:
    """Compare two stored records after removing intentionally volatile fields."""
    if first is None or second is None:
        return Check(
            name=name,
            status=NA,
            detail="Not produced by these runs, so there was nothing to compare.",
        )
    a = strip_volatile(first)
    b = strip_volatile(second)
    if a == b:
        return Check(
            name=name,
            status=PASS,
            detail="Every non-volatile field is identical across both runs.",
        )
    differences = _deep_diff(a, b)
    return Check(
        name=name,
        status=FAIL,
        detail=(
            f"{len(differences)} field(s) differ between identical runs: "
            f"{', '.join(differences[:6])}."
        ),
        values={"differing_fields": differences[:40]},
    )


def check_graph_structure(tracks_path_a: Path, tracks_path_b: Path) -> Check:
    """Build the GNN graph from both runs' tracks and compare its structure."""
    if not tracks_path_a.is_file() or not tracks_path_b.is_file():
        return Check(
            name="graph dataset structure",
            status=NA,
            detail="No track file in one of the runs, so no graph could be built.",
        )
    from extreme_weather.gnn.dataset import SplitConfig, build_dataset_from_file
    from extreme_weather.gnn.graph import GraphConfig

    split = SplitConfig(train=0.6, validation=0.2, test=0.2)
    try:
        a = build_dataset_from_file(tracks_path_a, GraphConfig(), split).as_dict()
        b = build_dataset_from_file(tracks_path_b, GraphConfig(), split).as_dict()
    except (FileNotFoundError, ValueError) as exc:
        return Check(
            name="graph dataset structure",
            status=NA,
            detail=f"A graph could not be built from these tracks: {exc}",
        )
    # Only the structural and split fields: a graph has no timestamp field, so
    # the whole record is comparable.
    keys = (
        "node_count",
        "edge_count",
        "track_count",
        "prediction_samples",
        "temporal_edge_count",
        "spatial_edge_count",
        "split",
    )
    differing = [k for k in keys if a.get(k) != b.get(k)]
    if differing:
        return Check(
            name="graph dataset structure",
            status=FAIL,
            detail=f"Graph structure differs between runs: {', '.join(differing)}.",
            values={"run1": {k: a.get(k) for k in keys}},
        )
    return Check(
        name="graph dataset structure",
        status=PASS,
        detail=(
            f"{a['node_count']} nodes, {a['edge_count']} edges, "
            f"split {a['split']} - identical across both runs."
        ),
        values={"nodes": a["node_count"], "edges": a["edge_count"]},
    )


def check_provenance(first: _RunArtifacts, second: _RunArtifacts) -> Check:
    """Compare the provenance metadata each stored record carries."""
    pairs = [
        ("events", first.events_payload, second.events_payload),
        ("tracks", first.tracks_payload, second.tracks_payload),
        ("alerts", first.alerts_payload, second.alerts_payload),
    ]
    problems: list[str] = []
    compared = 0
    for name, a, b in pairs:
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        for key in ("metadata", "schema_version", "project", "service", "phase"):
            if key not in a and key not in b:
                continue
            compared += 1
            diffs = _deep_diff(strip_volatile(a.get(key)), strip_volatile(b.get(key)))
            problems.extend(f"{name}.{d}" for d in diffs)
    if not compared:
        return Check(
            name="provenance metadata",
            status=NA,
            detail="No provenance metadata was present in the stored records.",
        )
    if problems:
        return Check(
            name="provenance metadata",
            status=FAIL,
            detail=f"Provenance differs between runs: {', '.join(problems)}.",
            values={"differing": problems},
        )
    return Check(
        name="provenance metadata",
        status=PASS,
        detail=f"{compared} provenance field(s) identical across both runs.",
    )


def check_evaluation_run_id() -> Check:
    """Check that the evaluation run id is derived, not random.

    Two calls with the same inputs must produce the same id, because the id is
    what makes a stored evaluation traceable. A time-based id would defeat
    that, so it is checked rather than assumed.
    """
    from extreme_weather.evaluation.evaluator import run_evaluation

    try:
        first = run_evaluation(model_available=False, seed=1234).run.run_id
        second = run_evaluation(model_available=False, seed=1234).run.run_id
        different = run_evaluation(model_available=False, seed=9999).run.run_id
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="evaluation run id",
            status=NA,
            detail=f"The evaluation could not be run: {type(exc).__name__}.",
        )
    if first != second:
        return Check(
            name="evaluation run id",
            status=FAIL,
            detail=f"Two identical evaluations produced different ids: {first} vs {second}.",
        )
    if different == first:
        return Check(
            name="evaluation run id",
            status=FAIL,
            detail=(
                "Two evaluations with different seeds produced the same id, so "
                "the id does not identify the run's inputs."
            ),
        )
    return Check(
        name="evaluation run id",
        status=PASS,
        detail=(
            "Identical inputs give an identical id, and a different seed gives "
            "a different one."
        ),
        values={"run_id": first},
    )


def check_code_fingerprint() -> Check:
    """Check that the code fingerprint is stable within one process.

    It is derived from the source files, so two calls must agree. A fingerprint
    that changed on every call would make every stored record's provenance a lie.
    """
    from extreme_weather.evaluation.evaluator import code_fingerprint

    try:
        first = code_fingerprint()
        second = code_fingerprint()
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="code fingerprint",
            status=NA,
            detail=f"The fingerprint could not be computed: {type(exc).__name__}.",
        )
    if first is None:
        return Check(
            name="code fingerprint",
            status=NA,
            detail=(
                "No fingerprint is available in this environment (the source "
                "tree is not readable), so nothing could be compared."
            ),
        )
    if first != second:
        return Check(
            name="code fingerprint",
            status=FAIL,
            detail=f"Fingerprint changed between calls: {first} vs {second}.",
        )
    return Check(
        name="code fingerprint",
        status=PASS,
        detail=f"Stable across calls: {first}.",
        values={"fingerprint": first},
    )


def run_all() -> list[Check]:
    """Run every reproducibility check and return the verdicts."""
    checks: list[Check] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        first, tracks_a = _run_once(base / "run1")
        second, tracks_b = _run_once(base / "run2")

        checks.append(_compare_ids("event ids", first.events, second.events))
        checks.append(_compare_ids("track ids", first.tracks, second.tracks))
        checks.append(_compare_ids("alert ids", first.alerts, second.alerts))
        checks.append(
            _compare_payload("events record (non-volatile)", first.events_payload, second.events_payload)
        )
        checks.append(
            _compare_payload("tracks record (non-volatile)", first.tracks_payload, second.tracks_payload)
        )
        checks.append(
            _compare_payload("alerts record (non-volatile)", first.alerts_payload, second.alerts_payload)
        )
        checks.append(check_graph_structure(tracks_a, tracks_b))
        checks.append(check_provenance(first, second))

        if first.run_id and second.run_id and first.run_id != second.run_id:
            checks.append(
                Check(
                    name="pipeline run id",
                    status=FAIL,
                    detail=(
                        "The run id differed between identical runs, so it is not "
                        f"derived from the data: {first.run_id} vs {second.run_id}."
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    name="pipeline run id",
                    status=PASS if first.run_id else NA,
                    detail=(
                        f"Identical across runs: {first.run_id}."
                        if first.run_id
                        else "No run completed, so there was no id to compare."
                    ),
                    values={"run_id": first.run_id},
                )
            )

    checks.append(check_evaluation_run_id())
    checks.append(check_code_fingerprint())
    return checks


def render(checks: list[Check]) -> str:
    """Build the human-readable report."""
    lines: list[str] = []
    add = lines.append
    add("=" * 72)
    add("Reproducibility verification (SIH26078, Phase 11)")
    add("=" * 72)
    add("Two independent pipeline runs with identical, seeded inputs are")
    add("compared. Intentionally volatile fields (timestamps, durations, host")
    add("names) are excluded; everything else must match.")
    add("")
    for check in checks:
        add(f"  [{check.status}]".ljust(18) + f" {check.name}")
        if check.detail:
            add(f"{'':<21}{check.detail}")
    add("")
    passed = sum(1 for c in checks if c.status == PASS)
    failed = sum(1 for c in checks if c.status == FAIL)
    na = sum(1 for c in checks if c.status == NA)
    # Machine-readable summary line, so a tool does not have to parse the
    # table above. Both forms are kept: the table for a human, this for a script.
    add(
        f"RESULT: {passed} passed, {failed} failed, {na} not applicable"
    )
    if na:
        add(
            "NOT_APPLICABLE means the input needed for that check was not "
            "produced. It is not a pass."
        )
    add("")
    add(
        "This verifies that identical inputs produce identical artifacts. It "
        "is a statement"
    )
    add(
        "about the code, not about forecast skill, and no skill is claimed."
    )
    add("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the verification and print the report.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when no check failed, ``1`` otherwise. A ``NOT_APPLICABLE``
        check is reported but does not fail the run.
    """
    parser = argparse.ArgumentParser(
        prog="verify_reproducibility.py",
        description=(
            "Verify that the deterministic components of SIH26078 produce "
            "equivalent artifacts across repeated runs. Reports PASS / FAIL / "
            "NOT_APPLICABLE per check."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON."
    )
    args = parser.parse_args(argv)

    try:
        checks = run_all()
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print(f"error: verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "checks": [c.as_dict() for c in checks],
                    "passed": sum(1 for c in checks if c.status == PASS),
                    "failed": sum(1 for c in checks if c.status == FAIL),
                    "not_applicable": sum(1 for c in checks if c.status == NA),
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(render(checks))

    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

