"""Report assembly for the Phase 7 validation (SIH26078).

Collects the individual stage reports into one summary, renders it as JSON and
Markdown, and reads it back for the API and the dashboard.

The summary always carries the same blocks, and a block whose data does not
exist says so rather than being omitted. A reader must be able to tell the
difference between "the GNN was worse" and "the GNN was never run", and a
missing key is not enough to make that clear.

The canonical limitations are stated here once, so no report can accidentally
ship without them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID, PROJECT_NAME
from extreme_weather.ingestion.base import utc_now
from extreme_weather.validation.ground_truth import GROUND_TRUTH_UNAVAILABLE

__all__ = [
    "CANONICAL_LIMITATIONS",
    "REPORT_FILENAMES",
    "build_phase7_summary",
    "read_report",
    "render_markdown",
    "write_reports",
]


#: Report files, keyed by the stage that produces them.
REPORT_FILENAMES: dict[str, str] = {
    "threshold_calibration": "threshold_calibration.json",
    "climatology": "climatology_diagnostics.json",
    "anomaly": "anomaly_validation.json",
    "tracking": "tracking_validation.json",
    "gnn": "gnn_validation.json",
    "downscaling": "downscaling_validation.json",
    "alerts": "alert_validation.json",
    "summary": "phase7_summary.json",
    "summary_markdown": "phase7_summary.md",
}

#: Printed verbatim in every report. These are not decoration: each one is a
#: claim this project deliberately does **not** make.
CANONICAL_LIMITATIONS: tuple[str, ...] = (
    "Research prototype. Not production-ready, not operational, not a guaranteed "
    "early warning, not 100% accurate, and not an exact prediction engine.",
    "No ground-truth event catalogue ships with this project, so detection skill "
    "(POD/FAR/CSI/precision/recall) is NOT computed. Descriptive counts are "
    "reported instead and are not accuracy.",
    "ERA5 is a reanalysis of the past: real observations fused with a model. It "
    "is not a direct observation, not a forecast, and not live data.",
    "Phase 1 thresholds are statistically calibrated to the training period only. "
    "They are NOT validated against any official or meteorological warning "
    "criterion.",
    "Alert severities are this project's own categories, not official IMD or WMO "
    "warnings. The risk score is a rule-based triage aid, not a probability of "
    "occurrence.",
    "A research-length record is not a climatology. Day-of-year and hour-of-day "
    "climatologies are reported as insufficient where the data cannot support "
    "them, rather than being filled from another bucket.",
    "Baselines are the comparison standard, not a formality. Where an AI method "
    "loses to nearest-neighbour interpolation, to cubic interpolation, or to "
    "persistence, the report says so.",
    "No result here has been compared against an issued warning, and no claim is "
    "made about disaster reduction, lives saved, or economic benefit.",
)


def _section(payload: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """Return one report block, or an explicit "not produced" marker.

    A missing block is *not* an empty dict: the two mean different things, and
    conflating them would let "the GNN never ran" read as "the GNN was perfect".
    """
    if payload is None:
        return {"available": False, "reason": "This report has not been produced yet."}
    return {"available": True, **dict(payload)}


def build_phase7_summary(
    *,
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    climatology: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    anomaly: Mapping[str, Any] | None = None,
    tracking: Mapping[str, Any] | None = None,
    gnn: Mapping[str, Any] | None = None,
    downscaling: Mapping[str, Any] | None = None,
    alerts: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra_limitations: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Assemble the unified Phase 7 summary.

    Every section is always present, either populated or explicitly marked as
    not produced. ``ground_truth_availability`` and ``limitations`` are never
    omitted.
    """
    produced = {
        "climatology": climatology is not None,
        "threshold_calibration": calibration is not None,
        "anomaly": anomaly is not None,
        "tracking": tracking is not None,
        "gnn": gnn is not None,
        "downscaling": downscaling is not None,
        "alerts": alerts is not None,
    }
    return {
        "schema": "phase7_summary/1.0",
        "project": PROJECT_ID,
        "service": PROJECT_NAME,
        "package_version": PACKAGE_VERSION,
        "generated_at": utc_now().isoformat(),
        "dataset_provenance": dict(dataset),
        "temporal_coverage": dict(split),
        "ground_truth_availability": dict(ground_truth),
        "climatology_diagnostics": _section(climatology, "climatology"),
        "threshold_calibration": _section(calibration, "calibration"),
        "anomaly_diagnostics": _section(anomaly, "anomaly"),
        "tracking_diagnostics": _section(tracking, "tracking"),
        "gnn_metrics": _section(gnn, "gnn"),
        "downscaling_metrics": _section(downscaling, "downscaling"),
        "alert_diagnostics": _section(alerts, "alerts"),
        "reports_produced": produced,
        "reports_missing": [name for name, ok in produced.items() if not ok],
        "reproducibility": dict(provenance or {}),
        "limitations": list(CANONICAL_LIMITATIONS) + list(extra_limitations),
    }


def write_reports(
    reports_dir: Path | str,
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the summary as JSON and Markdown and return the paths written."""
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / REPORT_FILENAMES["summary"]
    markdown_path = directory / REPORT_FILENAMES["summary_markdown"]
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def read_report(reports_dir: Path | str, key: str) -> dict[str, Any] | None:
    """Read one report by its :data:`REPORT_FILENAMES` key.

    Returns ``None`` when the report has not been produced, which the API
    turns into a 404 rather than an empty 200.

    Raises:
        KeyError: If the key is not a known report.
    """
    if key not in REPORT_FILENAMES:
        raise KeyError(
            f"Unknown report '{key}'. Known: {', '.join(sorted(REPORT_FILENAMES))}."
        )
    target = Path(reports_dir) / REPORT_FILENAMES[key]
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    """Format a value for the Markdown report."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _verdict_line(comparison: Mapping[str, Any] | None) -> str:
    """Render the headline comparison verdict of a report block."""
    if not comparison:
        return "Not available."
    summary = comparison.get("_summary")
    if isinstance(summary, Mapping) and "verdict" in summary:
        return str(summary["verdict"])
    return "See per-metric verdicts."



def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the summary as a readable Markdown document.

    The document leads with what was *not* measured, because a reader who
    skims will otherwise take a descriptive count for a skill score.
    """
    lines: list[str] = [
        f"# {summary.get('service', PROJECT_NAME)} "
        f"({summary.get('project', PROJECT_ID)}) - Phase 7 validation",
        "",
        f"Generated: {summary.get('generated_at')} "
        f"(package {summary.get('package_version')})",
        "",
        "> **Research prototype.** Not production-ready, not operational, not a",
        "> guaranteed early warning, not 100% accurate, not an exact prediction",
        "> engine, and not verified against any issued warning.",
        "",
        "## Ground truth",
        "",
        f"- **Status**: {_fmt(summary.get('ground_truth_availability', {}).get('status'))}",
        f"- **Detection skill**: {GROUND_TRUTH_UNAVAILABLE}",
        "",
        "Because no reference event catalogue ships with this project, POD, FAR,",
        "CSI, precision and recall are **not computed**. The counts below describe",
        "what the system produced; they are not accuracy.",
        "",
        "## Dataset provenance",
        "",
    ]
    dataset = summary.get("dataset_provenance", {})
    for key in (
        "dataset_id",
        "source",
        "source_url",
        "file_hash",
        "time_start",
        "time_end",
        "variables",
        "resolution_deg",
        "origin",
        "is_synthetic",
        "is_live",
    ):
        if key in dataset:
            lines.append(f"- **{key.replace('_', ' ').title()}**: {_fmt(dataset[key])}")

    lines.extend(["", "## Temporal coverage", ""])
    for split in summary.get("temporal_coverage", {}).get("splits", []):
        lines.append(
            f"- **{split.get('name')}**: {split.get('size')} steps "
            f"({_fmt(split.get('start'))} .. {_fmt(split.get('end'))})"
        )
    lines.extend(
        [
            "",
            "The split is strictly chronological: `max(train) < min(validation) < "
            "min(test)`, with no overlap.",
        ]
    )

    for title, key in (
        ("Climatology diagnostics", "climatology_diagnostics"),
        ("Threshold calibration", "threshold_calibration"),
        ("Anomaly diagnostics", "anomaly_diagnostics"),
        ("Tracking diagnostics", "tracking_diagnostics"),
        ("GNN vs persistence", "gnn_metrics"),
        ("Downscaling comparison", "downscaling_metrics"),
        ("Alert diagnostics", "alert_diagnostics"),
    ):
        block = summary.get(key, {})
        lines.extend(["", f"## {title}", ""])
        if not block.get("available", False):
            lines.append(f"Not produced. {block.get('reason', '')}")
            continue
        lines.extend(_render_section(key, block))

    lines.extend(["", "## Reports not produced", ""])
    missing = summary.get("reports_missing", [])
    lines.extend(
        (f"- {name}" for name in missing) if missing else ["None - every stage produced a report."]
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary.get("limitations", []))
    lines.extend(["", "## Reproducibility", ""])
    for key, value in summary.get("reproducibility", {}).items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {_fmt(value)}")
    lines.append("")
    return "\n".join(lines)



def _render_section(key: str, block: Mapping[str, Any]) -> list[str]:
    """Render the body of one report section."""
    if key == "gnn_metrics":
        metrics = block.get("metrics", {})
        baseline = block.get("baseline_metrics", {})
        counts = block.get("sample_counts", {})
        lines = [
            f"- **Verdict**: {_verdict_line(block.get('comparison'))}",
            f"- **Test samples**: "
            f"{_fmt(counts.get('test', metrics.get('sample_count')))}",
        ]
        for name in (
            "latitude_mae_deg",
            "longitude_mae_deg",
            "intensity_mae",
            "displacement_error_km_mean",
        ):
            lines.append(
                f"- **{name.replace('_', ' ')}**: GNN {_fmt(metrics.get(name))} "
                f"vs persistence {_fmt(baseline.get(name))}"
            )
        return lines

    if key == "downscaling_metrics":
        lines = [
            f"- **Verdict**: {_verdict_line(block.get('comparison'))}",
            f"- **Best by MAE**: {_fmt(block.get('winner_by_mae'))}",
        ]
        for name, stats in (block.get("methods") or {}).items():
            lines.append(
                f"- **{name}**: MAE {_fmt(stats.get('mae'))}, "
                f"RMSE {_fmt(stats.get('rmse'))}, "
                f"corr {_fmt(stats.get('correlation'))}"
            )
        return lines

    if key == "anomaly_diagnostics":
        skill = block.get("detection_skill", {})
        return [
            f"- **Events detected**: {_fmt(block.get('event_count'))} "
            f"(split: {_fmt(block.get('split'))})",
            f"- **Method**: {_fmt(block.get('method'))}",
            f"- **Skill**: {skill.get('note') or 'computed'}",
        ]

    if key == "tracking_diagnostics":
        return [
            f"- **Tracks**: {_fmt(block.get('track_count'))}",
            f"- **Events**: {_fmt(block.get('event_count'))}",
            f"- **Unmatched events**: {_fmt(block.get('unmatched_events'))}",
        ]

    if key == "alert_diagnostics":
        return [
            f"- **Alerts**: {_fmt(block.get('alert_count'))}",
            f"- **Severities**: {block.get('severity_distribution', {})}",
        ]

    if key == "climatology_diagnostics":
        return [
            f"- **{name}**: usable={coverage.get('usable')} "
            f"({coverage.get('buckets_sufficient')}/"
            f"{coverage.get('buckets_expected')} buckets) "
            f"{coverage.get('message', '')}"
            for name, coverage in (block.get("coverages") or {}).items()
        ]

    if key == "threshold_calibration":
        period = block.get("calibration_period", {})
        lines = [
            f"- **Method**: {_fmt(block.get('method'))}",
            f"- **Calibration period**: {_fmt(period.get('start'))} .. "
            f"{_fmt(period.get('end'))} ({_fmt(period.get('steps'))} steps)",
        ]
        for variable in block.get("variables", []):
            steps = variable.get("selected_thresholds", [])
            lines.append(
                f"- **{variable.get('variable')}** "
                f"({variable.get('sample_count')} samples): "
                + ", ".join(
                    f"{_fmt(step.get('threshold'))} {step.get('severity')}"
                    for step in steps
                )
            )
        return lines

    return ["- No detail available."]

