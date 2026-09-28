#!/usr/bin/env python3
"""Run the Phase 9 model evaluation (SIH26078).

Scores the GNN against persistence and constant velocity, evaluates the
downscaler against interpolation baselines where a high-resolution reference
exists, fits and calibrates an uncertainty interval, and writes a reproducible
evaluation record.

Examples
--------
See what would be evaluated, without running or writing anything::

    python scripts/run_phase9_evaluation.py --dry-run

Evaluate a stored Phase 7 dataset::

    python scripts/run_phase9_evaluation.py --dataset data/validation/reports/era5_2024_06.nc

Self-test the evaluator on deterministic synthetic samples::

    python scripts/run_phase9_evaluation.py --model fixture --sample-count 60

This is a **research prototype**. Nothing it reports is demonstrated forecast
skill: no verified ground-truth warning catalogue exists for this project, and
no accuracy percentage is claimed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extreme_weather.config import PROJECT_ID, PROJECT_NAME, get_settings  # noqa: E402
from extreme_weather.evaluation import (  # noqa: E402
    BOOTSTRAP_METHOD,
    build_evaluation_dataset,
    run_evaluation,
    summarise_gnn_comparison,
)

UTC = timezone.utc


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="run_phase9_evaluation.py",
        description=(
            f"{PROJECT_NAME} ({PROJECT_ID}) - real-data model evaluation, "
            "explainability and uncertainty (Phase 9). Research prototype: "
            "measurements on a historical record, not demonstrated forecast skill."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --dry-run\n"
            "  %(prog)s --model fixture --sample-count 60\n"
            "  %(prog)s --dataset data/validation/reports/era5_2024_06.nc\n"
            "\n"
            "Honesty guarantees:\n"
            "  * unavailable metrics are reported as UNAVAILABLE, never as 0.0\n"
            "  * no model is called better than a baseline without a measurement\n"
            "  * feature evidence is never described as causal\n"
            "  * uncertainty intervals are never called probabilities\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a NetCDF file to evaluate on. Omit to run a self-test on "
        "deterministic synthetic samples.",
    )
    parser.add_argument(
        "--start", default=None, help="Earliest ISO 8601 timestamp to include."
    )
    parser.add_argument(
        "--end", default=None, help="Latest ISO 8601 timestamp to include."
    )
    parser.add_argument(
        "--model",
        default="fixture",
        help="Model whose availability is recorded (default: fixture).",
    )
    parser.add_argument(
        "--tracks",
        default=None,
        help="A Phase 2 tracks.json. When given, the GNN baselines are scored "
        "on the real tracked events it contains rather than on synthetic samples.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="A trained Phase 3 GNN checkpoint. Without it the baselines are "
        "still scored but the model is reported as unavailable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240601,
        help="Seed for the residual bootstrap (default: 20240601).",
    )
    parser.add_argument(
        "--run-id", default=None, help="Explicit run id. Derived when omitted."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the inputs and show the plan without evaluating or writing.",
    )
    parser.add_argument(
        "--output", default=None, help="Write the evaluation JSON to this path."
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=60,
        help="Synthetic samples for a self-test run when no --dataset is given.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.6, help="Training fraction."
    )
    parser.add_argument(
        "--validation-ratio", type=float, default=0.2, help="Validation fraction."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print the full result JSON."
    )
    return parser


def _load_dataset(path: Path) -> Any:
    """Load a NetCDF dataset, or raise a clear error."""
    from extreme_weather.data.loader import load_dataset

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset '{path}' does not exist. Pass --dataset with a real file, "
            "or omit it to run the deterministic self-test."
        )
    return load_dataset(path, validate=False)


def _synthetic_samples(count: int, seed: int) -> dict[str, Any]:
    """Build deterministic, physically plausible evaluation samples.

    These exist so the evaluator itself can be exercised with no data on disk.
    They are **synthetic** and labelled as such; no result derived from them is
    a statement about real weather.
    """
    import numpy as np

    generator = np.random.default_rng(seed)
    # A smooth drift, so the samples are structured rather than pure noise.
    steps = np.arange(count, dtype="float64")
    current_lat = 12.0 + 0.05 * steps + generator.normal(0, 0.01, count)
    current_lon = 76.0 + 0.07 * steps + generator.normal(0, 0.01, count)
    previous_lat = current_lat - 0.05
    previous_lon = current_lon - 0.07
    true_lat = current_lat + 0.05
    true_lon = current_lon + 0.07
    current_intensity = 20.0 + generator.normal(0, 1.0, count)
    true_intensity = current_intensity + generator.normal(0, 0.5, count)
    # A deliberately imperfect "model": persistence plus a small error.
    predicted_lat = true_lat + generator.normal(0, 0.02, count)
    predicted_lon = true_lon + generator.normal(0, 0.02, count)
    predicted_intensity = true_intensity + generator.normal(0, 0.4, count)
    return {
        "current_latitude": current_lat,
        "current_longitude": current_lon,
        "current_intensity": current_intensity,
        "previous_latitude": previous_lat,
        "previous_longitude": previous_lon,
        "predicted_latitude": predicted_lat,
        "predicted_longitude": predicted_lon,
        "predicted_intensity": predicted_intensity,
        "true_latitude": true_lat,
        "true_longitude": true_lon,
        "true_intensity": true_intensity,
        "_synthetic": True,
    }


def _write(settings: Any, run: Any) -> Path | None:
    """Persist an evaluation run under the data directory."""
    try:
        directory = settings.data_dir / "evaluation" / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run.run_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(run.as_dict(), handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        return path
    except OSError as exc:  # pragma: no cover - defensive
        print(f"warning: could not write the run record: {exc}", file=sys.stderr)
        return None


def _print_result(run: Any) -> None:
    """Print an evaluation run in a readable form."""
    print()
    print(f"run {run.run_id}  status={run.status}  dataset={run.dataset_id}")
    provenance = run.provenance
    print(f"  seed {provenance.seed}  checksum {str(provenance.dataset_checksum)[:16]}")
    if provenance.code_fingerprint:
        print(f"  code fingerprint {provenance.code_fingerprint}")

    for group, metrics in sorted(run.metrics.items()):
        print(f"\n  [{group}]")
        for name, entry in sorted(metrics.items()):
            value = entry.get("value")
            unit = entry.get("unit") or ""
            print(
                f"    {name:<52} {value:>12.4f} {unit:<16} "
                f"[{entry.get('status')}]"
            )

    for group, metrics in sorted(run.unavailable_metrics.items()):
        print(f"\n  [{group}] NOT AVAILABLE")
        for name, entry in sorted(metrics.items()):
            print(f"    {name:<52} {entry.get('status')}: {entry.get('reason')}")

    if "gnn" in run.comparisons:
        print("\n  " + summarise_gnn_comparison(run.comparisons["gnn"]))

    uncertainty = run.uncertainty or {}
    if uncertainty:
        print(f"\n  uncertainty method: {uncertainty.get('method')}")
        for level, calibration in sorted((uncertainty.get("calibration") or {}).items()):
            if not isinstance(calibration, dict):
                continue
            coverage = calibration.get("empirical_coverage") or {}
            rendered = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in coverage.items()
            )
            print(
                f"    nominal {level}: coverage [{rendered}] "
                f"status={calibration.get('status')} n={calibration.get('sample_count')}"
            )

    for warning in run.warnings:
        print(f"  note: {warning}")
    print(f"\n  {run.DISCLAIMER}")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when the evaluation could not run, ``2`` for a
        configuration error.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()

    dataset = None
    if args.dataset:
        try:
            source = _load_dataset(Path(args.dataset))
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        try:
            dataset = build_evaluation_dataset(
                source,
                source=str(args.dataset),
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
            )
        except (ValueError, TypeError) as exc:
            print(f"error: the dataset could not be prepared: {exc}", file=sys.stderr)
            return 2

    samples = None
    validation_samples = None
    model_available = args.model != "fixture"
    model_checkpoint = None if args.model == "fixture" else args.model
    track_summary: dict[str, Any] | None = None

    if args.tracks:
        # Real tracked events: score the baselines on them, and the model too
        # when a checkpoint is supplied.
        from extreme_weather.evaluation.tracks import samples_from_tracks

        tracks_path = Path(args.tracks)
        if not tracks_path.is_file():
            print(f"error: tracks file '{tracks_path}' does not exist.", file=sys.stderr)
            return 2
        extracted = samples_from_tracks(tracks_path, checkpoint=args.checkpoint)
        track_summary = extracted.summary()
        model_available = extracted.model_available
        model_checkpoint = args.checkpoint
        if extracted.has_test_samples:
            test = dict(extracted.test)
            samples = test
        if extracted.has_validation_samples:
            validation_samples = dict(extracted.validation)

    if dataset is None and samples is None:
        if args.sample_count < 1:
            print(
                "error: --sample-count must be at least 1 for a self-test run, "
                "or pass --dataset to evaluate a real file.",
                file=sys.stderr,
            )
            return 2
        samples = _synthetic_samples(args.sample_count, args.seed)
        validation_samples = dict(samples)

    if args.dry_run:
        print(f"{PROJECT_NAME} ({PROJECT_ID}) - Phase 9 dry run")
        print("=" * 68)
        if dataset is not None:
            print(f"dataset        : {args.dataset}")
            print(f"dataset id     : {dataset.dataset_id}")
            print(f"period         : {dataset.source_period['start']} -> {dataset.source_period['end']}")
            print(f"variables      : {', '.join(dataset.variables)}")
            print(f"grid           : {dataset.grid}")
            sizes = dataset.split.sizes()
            print(
                f"split          : {sizes['train']}/{sizes['validation']}/{sizes['test']} "
                "(train/validation/test), chronological, not shuffled"
            )
            print(f"boundaries     : train ends {dataset.split.train_end}, test starts {dataset.split.test_start}")
            print(f"checksum       : {dataset.checksum[:16]}")
        else:
            print("dataset        : none supplied; a deterministic self-test would run")
            print(f"samples        : {args.sample_count} synthetic (SYNTHETIC, not real weather)")
        print(f"model          : {args.model}"
              + (f" (checkpoint: {args.checkpoint})" if args.checkpoint else ""))
        if args.tracks:
            print(f"tracks         : {args.tracks}")
        print(f"seed           : {args.seed}")
        print(f"uncertainty    : {BOOTSTRAP_METHOD}")
        print("measures       : GNN vs persistence, GNN vs constant velocity,")
        print("                 uncertainty intervals and measured coverage")
        print("\nartifacts written: False")
        print("Research prototype. Not an operational emergency-warning system.")
        return 0

    try:
        outcome = run_evaluation(
            dataset,
            samples=samples,
            validation_samples=validation_samples,
            model_available=model_available,
            model_checkpoint=model_checkpoint,
            seed=args.seed,
            run_id=args.run_id,
        )
    except (ValueError, TypeError) as exc:
        print(f"error: the evaluation could not run: {exc}", file=sys.stderr)
        return 2

    outcome.run.models = {
        "gnn": {"name": args.model, "scored": bool(outcome.run.metrics.get("gnn"))},
        "samples": "synthetic" if samples is not None and args.tracks is None else "real",
    }
    if track_summary is not None:
        outcome.run.models["tracks"] = track_summary
    if dataset is None and args.tracks is None:
        outcome.run.warnings.append(
            "This run used DETERMINISTIC SYNTHETIC samples, not real weather. "
            "The numbers characterise the evaluator, not any model skill."
        )

    path = _write(settings, outcome.run)

    if args.output:
        try:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(outcome.run.as_dict(), handle, indent=2, default=str)
            print(f"\nwrote {destination}")
        except OSError as exc:
            print(f"error: could not write --output: {exc}", file=sys.stderr)
            return 1

    if args.verbose:
        print(json.dumps(outcome.run.as_dict(), indent=2, default=str))
    else:
        _print_result(outcome.run)
        if path is not None:
            print(f"\n  record: {path}")

    if outcome.run.status == "FAILED":
        return 1
    # INSUFFICIENT_DATA is a valid, honest outcome, not a crash: no metric was
    # measurable, and that is reported rather than papered over.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
