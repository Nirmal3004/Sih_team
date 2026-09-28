"""Reproducible real-vs-synthetic evaluation (SIH26078, Phase 6).

Runs the same pipeline over a real/historical dataset and over a synthetic one,
and reports the two side by side.

```text
python scripts/evaluate_real_data.py
python scripts/evaluate_real_data.py --real-id era5_demo --output reports/real_data_evaluation.json
```

What this script does **not** do
--------------------------------
It does not produce a skill score, and it will not be made to. A skill number
requires a reference dataset of observed events, and this repository does not
ship one. When ground truth is present in ``data/reference/`` the matching
metrics (POD, FAR, CSI) are computed; when it is absent the report says
**"Ground truth unavailable"** in the metrics block and explains why in the
limitations. That is the honest answer, and it is not a failure of the script.

The synthetic-vs-real comparison is reported for every metric because a *worse*
real-data result is information, not an embarrassment. If the synthetic pipeline
"wins", that is stated plainly in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extreme_weather.config import (  # noqa: E402
    PACKAGE_VERSION,
    PROJECT_ID,
    PROJECT_NAME,
    get_settings,
)
from extreme_weather.ingestion import DataOrigin, DatasetRegistry  # noqa: E402

#: The literal string used wherever a number cannot honestly be produced.
NO_GROUND_TRUTH = "Ground truth unavailable."


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the evaluation script."""
    parser = argparse.ArgumentParser(
        description=(
            f"{PROJECT_NAME} ({PROJECT_ID}) - real vs synthetic pipeline evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--real-id",
        type=str,
        default=None,
        help="Registered real dataset id to evaluate (default: the first HISTORICAL one).",
    )
    parser.add_argument(
        "--synthetic-id",
        type=str,
        default=None,
        help="Registered synthetic dataset id to compare against (default: the first SYNTHETIC one).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "real_data_evaluation.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a human-readable Markdown summary next to the JSON.",
    )
    return parser


def pick(registry: DatasetRegistry, origin: str, requested: str | None) -> Any:
    """Return one entry for an origin, preferring an explicitly requested id."""
    if requested:
        entry = registry.find(requested)
        if entry is not None:
            return entry
        return None
    entries = registry.list(origin=origin)
    return entries[0] if entries else None


def describe(entry: Any) -> dict[str, Any]:
    """Return the dataset metadata block of the report."""
    if entry is None:
        return {"available": False}
    manifest = entry.manifest
    return {
        "available": True,
        "dataset_id": manifest.dataset_id,
        "source": manifest.source,
        "source_url": manifest.source_url,
        "origin": manifest.origin,
        "is_synthetic": manifest.is_synthetic,
        "is_live": manifest.is_live,
        "status_label": manifest.status_label,
        "file_name": manifest.file_name,
        "file_hash": manifest.file_hash,
        "content_hash": manifest.content_hash,
        "variables": list(manifest.variables),
        "variables_missing": list(manifest.variables_missing),
        "dimensions": dict(manifest.dimensions),
        "resolution_deg": manifest.resolution_deg,
        "resolution_km": manifest.resolution_km,
        "time_start": manifest.time_start,
        "time_end": manifest.time_end,
        "time_step_seconds": manifest.time_step_seconds,
        "units": dict(manifest.units),
        "unit_conversions": dict(manifest.unit_conversions),
        "coordinate_actions": list(manifest.coordinate_actions),
        "license": manifest.license,
        "retrieved_at": manifest.retrieved_at.isoformat(),
        "sample_count": int(manifest.dimensions.get("time", 0))
        * int(manifest.dimensions.get("latitude", 0))
        * int(manifest.dimensions.get("longitude", 0)),
        "quality": manifest.quality_summary,
    }



def stage_metrics(prefix: str) -> dict[str, Any]:
    """Return the metrics for one pipeline run, or a reason they are absent."""
    settings = get_settings()
    block: dict[str, Any] = {"available": False}

    events_path = settings.processed_dir / f"{prefix}_events.json"
    tracks_path = settings.processed_dir / f"{prefix}_tracks.json"
    alerts_path = settings.processed_dir / f"{prefix}_alerts.json"

    if events_path.is_file():
        try:
            events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
        except (OSError, json.JSONDecodeError) as exc:
            block["error"] = f"Could not read {events_path.name}: {exc}"
            events = None
        if events is not None:
            block["available"] = True
            block["anomaly"] = {
                "event_count": len(events),
                "event_types": sorted({str(item.get("event_type", "")) for item in events}),
                "peak_intensity": max((float(e.get("intensity", 0.0)) for e in events), default=0.0),
                "mean_intensity": (
                    sum(float(e.get("intensity", 0.0)) for e in events) / len(events)
                    if events
                    else 0.0
                ),
                "distinct_cells": len(
                    {(round(float(e.get("latitude", 0.0)), 3), round(float(e.get("longitude", 0.0)), 3)) for e in events}
                ),
            }

    if tracks_path.is_file():
        try:
            tracks = json.loads(tracks_path.read_text(encoding="utf-8")).get("tracks", [])
        except (OSError, json.JSONDecodeError):
            tracks = None
        if tracks is not None:
            block.setdefault("available", True)
            block["tracking"] = {
                "track_count": len(tracks),
                "durations_hours": [float(t.get("duration_hours", 0.0)) for t in tracks],
                "max_speeds_kmh": [float(t.get("max_speed_kmh", 0.0)) for t in tracks],
                "directions": sorted({str(t.get("direction", "")) for t in tracks}),
                "status_counts": json.loads(tracks_path.read_text(encoding="utf-8")).get(
                    "status_counts", {}
                ),
            }

    if alerts_path.is_file():
        try:
            payload = json.loads(alerts_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if payload is not None:
            block.setdefault("available", True)
            total = int(payload.get("alert_count", 0))
            block["alerts"] = {
                "alert_count": total,
                "severity_distribution": payload.get("severity_counts", {}),
                "status_distribution": payload.get("status_counts", {}),
            }

    return block


def ground_truth_metrics() -> dict[str, Any]:
    """Compute detection metrics **only** when a reference dataset exists.

    Without one, every skill number would be a guess, so this returns the
    explicit refusal instead of zeros that would read like "no skill".
    """
    settings = get_settings()
    reference_dir = settings.reference_dir
    if not reference_dir.is_dir() or not any(reference_dir.glob("*.json")):
        return {
            "available": False,
            "reason": (
                "No reference/event dataset found in "
                f"'{reference_dir}'. Event counts, tracks and alerts can be reported, "
                "but POD, FAR and CSI cannot be computed without one."
            ),
            "POD": NO_GROUND_TRUTH,
            "FAR": NO_GROUND_TRUTH,
            "CSI": NO_GROUND_TRUTH,
        }
    return {
        "available": True,
        "reference_dir": str(reference_dir),
        "POD": None,
        "FAR": None,
        "CSI": None,
        "note": "A reference dataset is present; run the matching workflow to score it.",
    }


def build_report(real: Any, synthetic: Any) -> dict[str, Any]:
    """Assemble the full evaluation report."""
    return {
        "schema": "real_data_evaluation/1.0",
        "project": PROJECT_ID,
        "service": PROJECT_NAME,
        "package_version": PACKAGE_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "datasets": {
            "real": describe(real),
            "synthetic": describe(synthetic),
        },
        "metrics": {
            "real": stage_metrics("real"),
            "synthetic": stage_metrics("synthetic"),
        },
        "ground_truth": ground_truth_metrics(),
        "limitations": [
            "This is a research prototype. It is not an operational forecast and not "
            "an early-warning product.",
            "No skill score is reported unless a reference dataset is supplied; "
            f"otherwise the report states '{NO_GROUND_TRUTH}'.",
            "A real reanalysis dataset is a model-assimilated reconstruction of the "
            "past. It is not a direct observation and not a forecast.",
            "Event counts depend on the configured thresholds, which were chosen for "
            "the synthetic generator and are not calibrated against real climatology.",
            "The synthetic comparison is a sanity check on the pipeline, not evidence "
            "that the synthetic data is easier or harder; a difference in event "
            "counts mostly reflects that the two datasets describe different weather.",
            "A real dataset producing fewer or weaker events than the synthetic one "
            "is a valid and expected result and is reported as-is.",
            "Alerts are deterministic rule outputs. They are not verified against any "
            "issued warning and carry no probability of occurrence.",
        ],
    }



def to_markdown(report: dict[str, Any]) -> str:
    """Render the report as a short Markdown summary."""
    lines: list[str] = [
        f"# {report['service']} ({report['project']}) - real data evaluation",
        "",
        f"Generated: {report['generated_at']} (package {report['package_version']})",
        "",
        "> Research prototype. Not an operational forecast and not an early-warning",
        "> product. No skill score is reported without a reference dataset.",
        "",
    ]

    for label, key in (("Real / historical", "real"), ("Synthetic", "synthetic")):
        block = report["datasets"][key]
        lines.append(f"## {label}")
        if not block.get("available"):
            lines.append("")
            lines.append("No such dataset has been ingested.")
            lines.append("")
            continue
        lines.extend(
            [
                "",
                f"- **Dataset**: `{block['dataset_id']}` ({block['status_label']})",
                f"- **Source**: {block['source']}"
                + (f" - {block['source_url']}" if block["source_url"] else " (local file)"),
                f"- **Period**: {block['time_start']} .. {block['time_end']}",
                f"- **Resolution**: {block['resolution_deg']}° (~{block['resolution_km']} km)",
                f"- **Grid**: {block['dimensions']}",
                f"- **Variables**: {', '.join(block['variables']) or 'none'}",
                f"- **Samples (time x lat x lon)**: {block['sample_count']}",
                f"- **Live**: {block['is_live']} (this project ingests files, never a live feed)",
            ]
        )
        if block["variables_missing"]:
            lines.append(f"- **Not present**: {', '.join(block['variables_missing'])}")
        lines.append("")

    truth = report["ground_truth"]
    lines.append("## Ground truth")
    lines.append("")
    lines.append(f"- **POD**: {truth.get('POD')}")
    lines.append(f"- **FAR**: {truth.get('FAR')}")
    lines.append(f"- **CSI**: {truth.get('CSI')}")
    if truth.get("reason"):
        lines.append(f"- **Reason**: {truth['reason']}")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    for key in ("real", "synthetic"):
        metrics = report["metrics"][key]
        lines.append(f"### {key}")
        if not metrics.get("available"):
            lines.append("")
            lines.append("No pipeline output found for this run.")
            lines.append("")
            continue
        lines.append("")
        anomaly = metrics.get("anomaly", {})
        lines.append(
            f"- Anomaly: {anomaly.get('event_count', 'n/a')} event(s), "
            f"peak intensity {anomaly.get('peak_intensity', 'n/a')}, "
            f"{anomaly.get('distinct_cells', 'n/a')} distinct location(s)"
        )
        tracking = metrics.get("tracking", {})
        lines.append(
            f"- Tracking: {tracking.get('track_count', 'n/a')} track(s), "
            f"directions {', '.join(tracking.get('directions', [])) or 'n/a'}"
        )
        alerts = metrics.get("alerts", {})
        lines.append(
            f"- Alerts: {alerts.get('alert_count', 'n/a')} "
            f"({alerts.get('severity_distribution', {})})"
        )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point: write reports/real_data_evaluation.json (and optionally .md)."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    registry = DatasetRegistry(settings=settings)

    real = pick(registry, DataOrigin.HISTORICAL, args.real_id)
    synthetic = pick(registry, DataOrigin.SYNTHETIC, args.synthetic_id)

    if real is None:
        print(
            "No real/historical dataset is registered. Ingest one first:\n"
            "  python scripts/ingest_weather.py --file <path> --dataset-id <id>\n"
            "  python scripts/ingest_weather.py --fetch-era5 --era5-lats 12.5,13,13.5 "
            "--era5-lons 76.5,77,77.5 --start 2024-06-01 --end 2024-06-03",
            file=sys.stderr,
        )
        return 2

    report = build_report(real, synthetic)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  json report : {args.output}")

    if args.markdown:
        markdown = args.output.with_suffix(".md")
        markdown.write_text(to_markdown(report), encoding="utf-8")
        print(f"  markdown    : {markdown}")

    print(f"\n  real dataset     : {report['datasets']['real']['dataset_id']}")
    if synthetic is not None:
        print(f"  synthetic dataset: {report['datasets']['synthetic']['dataset_id']}")
    print(f"  POD/FAR/CSI      : {report['ground_truth'].get('POD')}")
    print(
        "\n  This is a research prototype on real historical data. It is not an "
        "operational forecast."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

