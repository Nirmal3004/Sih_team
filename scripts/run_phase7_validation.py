"""Run the complete Phase 7 validation pipeline (SIH26078).

```bash
python scripts/run_phase7_validation.py
python scripts/run_phase7_validation.py --end-date 2024-06-30 --days 21
python scripts/run_phase7_validation.py --reuse-existing           # no download
```

Stages, in order, each of which **fails loudly** rather than continuing with a
missing input:

1.  build the real historical dataset (bounded ERA5 subset, or reuse one)
2.  Phase 0 preprocessing
3.  chronological train/validation/test split (leakage asserted)
4.  climatology diagnostics (training split only)
5.  threshold calibration (training split only)
6.  anomaly detection on the validation split
7.  event tracking
8.  graph construction for the GNN
9.  GNN evaluation against a persistence baseline
10. downscaling evaluation: nearest vs cubic vs AI
11. alert generation
12. unified report + provenance record

Prints ``PHASE7_VALIDATION_COMPLETE`` and a summary at the end.

**What this script will not print**
-----------------------------------
It will not print a POD, a FAR, a CSI, an accuracy, or a skill score, because
no ground truth exists to compute them. The absence is stated explicitly in
every report instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extreme_weather.config import PROJECT_ID, PROJECT_NAME, get_settings  # noqa: E402
from extreme_weather.validation.dataset_builder import (  # noqa: E402
    ValidationDatasetBuilder,
    ValidationDatasetConfig,
    ValidationDatasetError,
)
from extreme_weather.validation.ground_truth import load_ground_truth  # noqa: E402
from extreme_weather.validation.provenance import (  # noqa: E402
    ProvenanceRecord,
    ValidationRunRecord,
    environment_facts,
    write_run_record,
)
from extreme_weather.validation.reports import (  # noqa: E402
    build_phase7_summary,
    read_report,
    write_reports,
)

#: Every stage, in execution order.
STAGES: tuple[str, ...] = (
    "dataset",
    "preprocessing",
    "split",
    "climatology",
    "calibration",
    "anomaly",
    "tracking",
    "graph",
    "gnn",
    "downscaling",
    "alerts",
    "report",
)


class StageFailure(RuntimeError):
    """Raised when a required stage cannot complete."""


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the Phase 7 pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            f"{PROJECT_NAME} ({PROJECT_ID}) - Phase 7 real-data validation pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-id", type=str, default="phase7_era5")
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-06-30",
        help="Last day of the request (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of days ending on --end-date. Bounded on purpose.",
    )
    parser.add_argument(
        "--latitudes",
        type=str,
        default="12.0,12.5,13.0,13.5,14.0",
        help="Comma-separated latitude points of the bounding box.",
    )
    parser.add_argument(
        "--longitudes",
        type=str,
        default="75.0,75.5,76.0,76.5,77.0",
        help="Comma-separated longitude points of the bounding box.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an already-downloaded source file instead of fetching.",
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=30,
        help="Training epochs for the GNN prototype (train split only).",
    )
    parser.add_argument(
        "--skip-gnn",
        action="store_true",
        help="Skip the GNN stages and record them as skipped, not as passing.",
    )
    return parser


def _floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float list.

    Raises:
        ValueError: If any entry is not a number.
    """
    try:
        return tuple(float(part) for part in str(value).split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a comma-separated list of numbers.") from exc


def _write(reports_dir: Path, key: str, payload: Any) -> None:
    """Write one stage report as JSON."""
    target = reports_dir / _filename(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _filename(key: str) -> str:
    """Return the report filename for a stage key."""
    from extreme_weather.validation.reports import REPORT_FILENAMES

    if key not in REPORT_FILENAMES:
        raise KeyError(f"No report file is defined for stage '{key}'.")
    return REPORT_FILENAMES[key]


def _run_anomaly(
    builder: ValidationDatasetBuilder,
    config: ValidationDatasetConfig,
    result: Any,
) -> dict[str, Any]:
    """Run Phase 1 detection and Phase 2 tracking on the validation split.

    The climatology is built from the **training** split only, so no test-period
    value can influence a score. The thresholds come from the calibration.

    Returns:
        The anomaly and tracking report blocks.
    """
    from extreme_weather.anomaly.climatology import ClimatologyGrouping, build_climatology
    from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore
    from extreme_weather.tracking.service import TrackingService
    from extreme_weather.tracking.store import TrackStore
    from extreme_weather.validation.metrics import (
        compute_anomaly_diagnostics,
        compute_tracking_diagnostics,
    )
    from extreme_weather.validation.threshold_calibration import calibrate_thresholds
    import xarray as xr

    train = result.datasets["train"]
    validation = result.datasets["validation"]

    # The Phase 1 climatology is per-variable and is built *inside* the
    # service from a reference dataset. That reference is the **training
    # split**, which is the leakage guard: no test-period value can influence
    # a score or a threshold. Separate climatology files are also written
    # here for inspection; the statistic names collide across variables, so
    # they cannot share a file.
    variables = [
        name
        for name in ("precipitation", "temperature")
        if name in train.data_vars
    ]
    if not variables:
        raise StageFailure(
            "The training split carries neither 'precipitation' nor "
            "'temperature', so no climatology can be built."
        )

    train_path = builder.processed_dir / f"{config.dataset_id}_train.nc"
    validation_path = builder.processed_dir / f"{config.dataset_id}_validation.nc"
    train.to_netcdf(train_path, mode="w")
    validation.to_netcdf(validation_path, mode="w")

    for variable in variables:
        build_climatology(
            train,
            variable,
            grouping=ClimatologyGrouping.MONTH,
            fallback="pooled",
        ).to_netcdf(
            builder.processed_dir
            / f"{config.dataset_id}_climatology_train_{variable}.nc",
            mode="w",
        )
    climatology_path = train_path

    calibration = calibrate_thresholds(train, dataset_id=config.dataset_id)
    thresholds = calibration.thresholds

    events_path = builder.reports_dir / f"{config.dataset_id}_events.json"
    store = EventStore(events_path)
    service = AnomalyDetectionService(
        store=store,
        thresholds=thresholds,
        default_grouping=ClimatologyGrouping.MONTH,
    )
    # write_outputs=True so the events file Phase 2 reads actually exists. A
    # run that detects nothing still has to leave a valid, empty analysis -
    # otherwise the tracker would fail on a missing file.
    analysis = service.run(
        validation_path, reference_path=climatology_path, write_outputs=True
    )
    events = list(analysis.events)

    truth = load_ground_truth(get_settings().reference_dir / "events.json")
    anomaly = compute_anomaly_diagnostics(
        events,
        split="validation",
        dataset_id=config.dataset_id,
        truth=truth,
        thresholds=calibration.variable("precipitation") and {
            "precipitation": [
                {"threshold": value, "severity": str(severity)}
                for value, severity in (
                    calibration.variable("precipitation").selected_thresholds
                    if calibration.variable("precipitation")
                    else []
                )
            ]
        },
        method=(
            f"calibrated percentile thresholds ({calibration.method}) on the "
            "training split"
        ),
    )

    tracks_store = TrackStore(builder.reports_dir / f"{config.dataset_id}_tracks.json")
    tracking = TrackingService(event_store=store, track_store=tracks_store)
    tracked = tracking.run(write_output=True)
    tracking_report = compute_tracking_diagnostics(
        tracked.tracks,
        split="validation",
        dataset_id=config.dataset_id,
        event_count=len(events),
        truth=truth,
    )
    return {
        "anomaly": anomaly.as_dict(),
        "tracking": tracking_report.as_dict(),
        "calibration": calibration.as_dict(),
        "event_count": len(events),
        "track_count": tracked.track_count,
    }



def _run_gnn(
    builder: ValidationDatasetBuilder,
    config: ValidationDatasetConfig,
    epochs: int,
) -> dict[str, Any]:
    """Train the Phase 3 GNN on train, select on validation, score test once.

    The GNN predicts per-node ``[delta_latitude, delta_longitude,
    delta_intensity]``. The persistence baseline is "nothing moves", so it
    predicts a zero delta. Both are scored on **identical** test nodes, so any
    difference between them is the model and not the sample.
    """
    import numpy as np
    import torch

    from extreme_weather.gnn.dataset import SplitConfig, build_dataset_from_file
    from extreme_weather.gnn.model import WeatherGNN
    from extreme_weather.gnn.training import TrainingConfig, predict_all, train
    from extreme_weather.validation.metrics import compute_gnn_comparison

    tracks_path = builder.reports_dir / f"{config.dataset_id}_tracks.json"
    if not tracks_path.is_file():
        raise StageFailure(
            f"No tracks file at '{tracks_path}'; the GNN stage needs Phase 2 output."
        )

    dataset = build_dataset_from_file(tracks_path, split_config=SplitConfig())
    batch = dataset.batch
    test_nodes = [
        index
        for index in dataset.split.test
        if index < len(batch.targets) and batch.targets[index] is not None
    ]
    sample_counts = {
        "train": len(dataset.split.train),
        "validation": len(dataset.split.validation),
        "test": len(test_nodes),
    }

    if not test_nodes:
        # Too few test nodes to score. Report that, rather than training on
        # the test split to manufacture a number.
        return {
            "dataset": config.dataset_id,
            "split": "test",
            "sample_counts": sample_counts,
            "model_checkpoint": None,
            "metrics": {},
            "baseline_metrics": {},
            "comparison": {
                "_summary": {
                    "verdict": "not computed - the test split contains no node "
                    "with a known next state."
                }
            },
            "limitations": [
                "A test split this small cannot support any claim about "
                "forecasting skill, favourable or unfavourable.",
                "No GNN metric is claimed for this run.",
            ],
        }

    model = WeatherGNN()
    trained = train(
        batch,
        dataset.split.train,
        dataset.split.validation,
        training_config=TrainingConfig(epochs=epochs, seed=7),
    )

    with torch.no_grad():
        predictions = predict_all(model, batch).detach().cpu().numpy()

    targets = np.asarray(
        [batch.targets[index] for index in test_nodes], dtype="float64"
    )
    predicted = np.asarray([predictions[index] for index in test_nodes], dtype="float64")

    # The graph stores deltas, so "persistence" is the zero delta and the
    # truth is the observed delta. Absolute positions are recovered by adding
    # the current node position, which cancels in a delta comparison - but the
    # displacement metric needs the positions, so they are rebuilt.
    lat_delta, lon_delta, intensity_delta = (
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
    )
    pred_lat_delta, pred_lon_delta, pred_intensity_delta = (
        predicted[:, 0],
        predicted[:, 1],
        predicted[:, 2],
    )

    base_lat = np.zeros_like(lat_delta)
    base_lon = np.zeros_like(lon_delta)
    base_intensity = np.zeros_like(intensity_delta)

    comparison = compute_gnn_comparison(
        predicted_lat=base_lat + pred_lat_delta,
        predicted_lon=base_lon + pred_lon_delta,
        predicted_intensity=base_intensity + pred_intensity_delta,
        current_lat=base_lat,
        current_lon=base_lon,
        current_intensity=base_intensity,
        future_lat=base_lat + lat_delta,
        future_lon=base_lon + lon_delta,
        future_intensity=base_intensity + intensity_delta,
        dataset=config.dataset_id,
        split="test",
        model_checkpoint=(
            str(getattr(trained, "checkpoint_path", "") or "") or "in-memory (not saved)"
        ),
        sample_counts=sample_counts,
        limitations=(
            "Targets are per-node deltas, so the persistence baseline is the "
            "zero delta ('nothing moves'). This is the standard baseline and the "
            "hardest one to beat on short sequences.",
            "The test split was scored exactly once; the checkpoint was selected "
            "on validation loss only.",
            "interval is claimed.",
        ),
    )
    return comparison.as_dict()


def _run_downscaling(
    builder: ValidationDatasetBuilder,
    config: ValidationDatasetConfig,
    result: Any,
) -> dict[str, Any]:
    """Evaluate nearest, cubic and AI downscaling on identical frames.

    The dataset is built from the **training** split, so the target construction
    never sees the test period. The reference is the Phase 4 synthetic
    high-resolution target, which is *not* observational truth - the report
    says so, because a perfect score against it would only mean the
    interpolation recovered a known construction.
    """
    import numpy as np
    import xarray as xr

    from extreme_weather.downscaling.baseline import cubic_downscale, nearest_downscale
    from extreme_weather.downscaling.dataset import build_downscaling_dataset
    from extreme_weather.validation.metrics import compute_downscaling_comparison

    train = result.datasets["train"]
    variable = "precipitation"
    if variable not in train.data_vars:
        raise StageFailure(
            f"Downscaling needs '{variable}', which the dataset does not carry."
        )

    dataset = build_downscaling_dataset(
        np.asarray(train[variable].values, dtype="float64"),
        np.asarray(train.coords["latitude"].values, dtype="float64"),
        np.asarray(train.coords["longitude"].values, dtype="float64"),
        [np.datetime64(value) for value in train.coords["time"].values],
        target_resolution=float(builder.settings.downsample_target_resolution_deg),
        variable=variable,
    )

    # Score on the dataset's own chronological TEST block, and on the same
    # frames for every method.
    test_index = np.asarray(dataset.split.test, dtype=int)
    if test_index.size == 0:
        return {
            "available": False,
            "reason": (
                "The downscaling test block is empty, so no method could be "
                "scored. No result is claimed."
            ),
            "dataset": config.dataset_id,
            "split": "test",
            "frame_count": 0,
            "variable": variable,
            "methods": {},
            "winner_by_mae": None,
            "limitations": ["No downscaling metric is claimed for this run."],
        }

    fields = np.asarray(dataset.coarse, dtype="float64")[test_index]
    reference = np.asarray(dataset.references, dtype="float64")[test_index]
    target_shape = reference.shape[1:]
    target_resolution = float(builder.settings.downsample_target_resolution_deg)

    # The baselines resample a DataArray onto a target *resolution*, so each
    # test frame is wrapped in one and the result is compared on identical
    # frames.
    latitude = np.asarray(train.coords["latitude"].values, dtype="float64")
    longitude = np.asarray(train.coords["longitude"].values, dtype="float64")

    def _resample(method, frame: np.ndarray) -> np.ndarray:
        wrapped = xr.DataArray(
            frame,
            dims=("latitude", "longitude"),
            coords={"latitude": latitude, "longitude": longitude},
        )
        return np.asarray(method(wrapped, target_resolution).values, dtype="float64")

    predictions = {
        "nearest": np.stack([_resample(nearest_downscale, f) for f in fields]),
        "cubic": np.stack([_resample(cubic_downscale, f) for f in fields]),
    }
    if predictions["cubic"].shape != reference.shape:
        # Cubic needs enough source points; when the grid is too small the
        # frame is dropped rather than misaligned against the reference.
        predictions.pop("cubic")

    # The AI path runs only when a trained checkpoint exists; otherwise it is
    # omitted rather than imitated by a baseline under the AI's name.
    ai_available = False
    try:
        from extreme_weather.downscaling.checkpoint import DownscalingCheckpoint
        from extreme_weather.downscaling.inference import DownscalingInference

        checkpoint = DownscalingCheckpoint()
        if checkpoint.exists():
            inference = DownscalingInference(checkpoint)
            predictions["ai"] = np.stack(
                [
                    np.asarray(
                        inference.predict(frame).values
                        if hasattr(inference.predict(frame), "values")
                        else inference.predict(frame),
                        dtype="float64",
                    )
                    for frame in fields
                ]
            )
            ai_available = True
    except Exception:  # noqa: BLE001 - no model is the expected state
        ai_available = False

    payload = compute_downscaling_comparison(
        predictions,
        reference,
        dataset=config.dataset_id,
        split="test",
        variable=variable,
        # No event threshold is supplied, so POD/FAR are reported as not
        # computed rather than against an arbitrary cut.
        event_threshold=None,
        ai_method="ai",
        limitations=(
            "The high-resolution reference is the Phase 4 SYNTHETIC target, "
            "generated from the coarse field. It is not observational truth, so "
            "these numbers measure interpolation of a known construction, not "
            "forecast skill.",
            "POD/FAR are not computed: no defensible event threshold exists for "
            "a synthetic reference target.",
            "The AI method is included only when a trained checkpoint exists; "
            "otherwise it is omitted rather than imitated.",
        ),
    ).as_dict()

    if not ai_available:
        return {
            "available": False,
            "reason": (
                "No trained AI downscaling checkpoint exists, so only the "
                "interpolation baselines were scored. No AI result is claimed."
            ),
            **payload,
        }
    return {"available": True, **payload}

def _run_alerts(
    builder: ValidationDatasetBuilder,
    config: ValidationDatasetConfig,
) -> dict[str, Any]:
    """Generate Phase 5 alerts on the validation events and describe them."""
    from extreme_weather.alerts.service import AlertService
    from extreme_weather.alerts.store import AlertStore
    from extreme_weather.anomaly.service import EventStore
    from extreme_weather.tracking.store import TrackStore
    from extreme_weather.validation.ground_truth import load_ground_truth
    from extreme_weather.validation.metrics import compute_alert_diagnostics

    reports = builder.reports_dir
    events_path = reports / f"{config.dataset_id}_events.json"
    if not events_path.is_file():
        raise StageFailure(
            f"No events file at '{events_path}'; alerts need Phase 1 output."
        )

    service = AlertService(
        alert_store=AlertStore(reports / f"{config.dataset_id}_alerts.json"),
        event_store=EventStore(events_path),
        track_store=TrackStore(reports / f"{config.dataset_id}_tracks.json"),
    )
    generated = service.generate()
    truth = load_ground_truth(get_settings().reference_dir / "events.json")
    return compute_alert_diagnostics(
        generated.stored.alerts,
        split="validation",
        dataset_id=config.dataset_id,
        truth=truth,
    ).as_dict()


def _run_report(
    builder: ValidationDatasetBuilder,
    record: ValidationRunRecord,
    **sections: Any,
) -> dict[str, Any]:
    """Assemble and write the unified Phase 7 report.

    Returns:
        The summary with the written paths and the loaded ground-truth status
        attached, so the caller can print both without re-reading the report.
    """
    from extreme_weather.validation.reports import build_phase7_summary, write_reports

    truth = load_ground_truth(get_settings().reference_dir / "events.json")
    summary = build_phase7_summary(
        dataset=record.dataset.as_dict(),
        split=record.split,
        ground_truth=truth.as_dict(),
        provenance=record.as_dict(),
        **sections,
    )
    return {
        **summary,
        "_paths": write_reports(builder.reports_dir, summary),
        "_truth": truth,
    }

def main(argv: list[str] | None = None) -> int:
    """Run every Phase 7 stage in order, then print the summary.

    Returns:
        ``0`` when every stage completed, ``1`` when a stage failed. A failure
        is printed as ``PHASE7_VALIDATION_FAILED`` and recorded in the run's
        provenance record, so a broken run is diagnosable rather than silent.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    builder = ValidationDatasetBuilder(settings)
    started = datetime.now(timezone.utc).replace(microsecond=0)

    end = datetime.fromisoformat(args.end_date).date()
    start = end - timedelta(days=max(1, args.days) - 1)
    config = ValidationDatasetConfig(
        dataset_id=args.dataset_id,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        latitudes=_floats(args.latitudes),
        longitudes=_floats(args.longitudes),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    print(f"{PROJECT_NAME} ({PROJECT_ID}) - Phase 7 real-data validation")
    print(f"  dataset id  : {config.dataset_id}")
    print(
        f"  period      : {config.start_date} .. {config.end_date} "
        f"({config.day_count()} days)"
    )
    print(f"  bounding box: {len(config.latitudes)}x{len(config.longitudes)} points")
    print(f"  workspace   : {builder.root}")

    record = ValidationRunRecord(
        run_id=f"phase7_{started.strftime('%Y%m%dT%H%M%SZ')}",
        started_at=started,
        dataset=ProvenanceRecord(dataset_id=config.dataset_id, origin=config.origin),
        configuration=config.as_dict(),
        environment=environment_facts(),
        notes=(
            "Research prototype. Historical reanalysis only: not a forecast, not "
            "live, and not verified against any issued warning."
        ),
    )

    try:
        # --- 1-3: dataset, preprocessing, chronological split ------------
        source_path = (
            builder.raw_dir / f"{config.dataset_id}_source.nc"
            if args.reuse_existing
            else None
        )
        if source_path is not None and not source_path.is_file():
            print(
                f"  --reuse-existing: '{source_path}' not found; fetching instead.",
                file=sys.stderr,
            )
            source_path = None
        result = builder.build(config, source_path=source_path)
        record.record_stage("dataset", "ok", days=config.day_count())
        record.record_stage(
            "preprocessing", "ok" if result.processed_path else "skipped"
        )
        record.record_stage("split", "ok", **result.split.as_dict())
        record.dataset = ProvenanceRecord.from_manifest(result.manifest)
        record.split = result.split.as_dict()
        print(
            f"  [1-3] dataset ok: {len(result.datasets['train'])}/"
            f"{len(result.datasets['validation'])}/{len(result.datasets['test'])} "
            "steps (train/validation/test)"
        )

        # --- 4: climatology diagnostics ----------------------------------
        from extreme_weather.validation.climatology_validation import (
            build_climatology_diagnostics,
        )

        climatology = build_climatology_diagnostics(
            result.datasets["train"], split="train", dataset_id=config.dataset_id
        )
        _write(builder.reports_dir, "climatology", climatology.as_dict())
        record.record_stage("climatology", "ok", sufficient=climatology.is_sufficient)
        month = climatology.coverage("month") or _noop()
        day_of_year = climatology.coverage("day_of_year") or _noop()
        print(
            f"  [4] climatology: monthly usable={month.usable}, "
            f"day-of-year usable={day_of_year.usable}"
        )
        if not day_of_year.usable:
            print(f"        {day_of_year.message}")

        # --- 5-7: calibration, anomaly detection, tracking ---------------
        pipeline = _run_anomaly(builder, config, result)
        _write(builder.reports_dir, "threshold_calibration", pipeline["calibration"])
        _write(builder.reports_dir, "anomaly", pipeline["anomaly"])
        _write(builder.reports_dir, "tracking", pipeline["tracking"])
        record.record_stage(
            "calibration", "ok", method=pipeline["calibration"]["method"]
        )
        record.record_stage("anomaly", "ok", events=pipeline["event_count"])
        record.record_stage("tracking", "ok", tracks=pipeline["track_count"])
        print(
            f"  [5-7] calibration ok; {pipeline['event_count']} event(s), "
            f"{pipeline['track_count']} track(s) on the validation split"
        )

        # --- 8: graph construction ---------------------------------------
        from extreme_weather.gnn.dataset import build_dataset_from_file

        tracks_path = builder.reports_dir / f"{config.dataset_id}_tracks.json"
        graph = build_dataset_from_file(tracks_path)
        record.record_stage(
            "graph", "ok", nodes=graph.node_count, edges=graph.edge_count
        )
        print(f"  [8] graph: {graph.node_count} nodes, {graph.edge_count} edges")

        # --- 9: GNN evaluation -------------------------------------------
        if args.skip_gnn:
            gnn_payload = {
                "available": False,
                "reason": "GNN evaluation was skipped with --skip-gnn. No result "
                "is claimed.",
            }
            record.record_stage("gnn", "skipped", reason="--skip-gnn")
            print("  [9] GNN: skipped (--skip-gnn); no result is claimed.")
        else:
            gnn_payload = _run_gnn(builder, config, args.gnn_epochs)
            record.record_stage("gnn", "ok")
            print(
                "  [9] GNN: "
                + str(
                    gnn_payload.get("comparison", {}).get("_summary", {}).get("verdict")
                )
            )
        _write(builder.reports_dir, "gnn", gnn_payload)

        # --- 10: downscaling ---------------------------------------------
        downscaling = _run_downscaling(builder, config, result)
        _write(builder.reports_dir, "downscaling", downscaling)
        record.record_stage(
            "downscaling", "ok", winner=downscaling.get("winner_by_mae")
        )
        print(
            f"  [10] downscaling: best by MAE = {downscaling.get('winner_by_mae')}"
        )

        # --- 11: alerts ---------------------------------------------------
        alerts = _run_alerts(builder, config)
        _write(builder.reports_dir, "alerts", alerts)
        record.record_stage("alerts", "ok", alerts=alerts.get("alert_count"))
        print(f"  [11] alerts: {alerts.get('alert_count')} produced")

        # --- 12: unified report -------------------------------------------
        report = _run_report(
            builder,
            record,
            climatology=climatology.as_dict(),
            calibration=pipeline["calibration"],
            anomaly=pipeline["anomaly"],
            tracking=pipeline["tracking"],
            gnn=gnn_payload,
            downscaling=downscaling,
            alerts=alerts,
        )
        record.record_stage("report", "ok")
        record.finished_at = datetime.now(timezone.utc).replace(microsecond=0)
        write_run_record(record, builder.provenance_dir)
        print(f"  [12] report: {report['_paths']['json']}")

    except Exception as exc:  # noqa: BLE001 - a failed stage must be loud
        record.record_stage("pipeline", "failed", error=f"{type(exc).__name__}: {exc}")
        record.finished_at = datetime.now(timezone.utc).replace(microsecond=0)
        write_run_record(record, builder.provenance_dir)
        print(
            f"\nPHASE7_VALIDATION_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1

    print("\nPHASE7_VALIDATION_COMPLETE")
    print(f"  dataset        : {record.dataset.dataset_id} ({record.dataset.origin})")
    print(f"  period         : {record.dataset.time_start} .. {record.dataset.time_end}")
    print(
        "  split          : "
        + ", ".join(f"{s['name']}={s['size']}" for s in record.split.get("splits", []))
    )
    print(f"  ground truth   : {report['_truth'].status} - detection skill NOT computed")
    print(f"  events (valid) : {pipeline['event_count']}")
    print(f"  tracks (valid) : {pipeline['track_count']}")
    print(f"  reports        : {builder.reports_dir}")
    print(
        "\n  Research prototype on historical reanalysis. Not an operational "
        "forecast,\n  not a guaranteed early warning, and not verified against any "
        "issued warning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
