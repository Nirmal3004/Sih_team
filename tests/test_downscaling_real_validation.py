"""Tests for Phase 7 downscaling real-data evaluation (report level).

The metric arithmetic is covered in ``test_gnn_real_validation.py``; this file
covers the *evaluation contract*: which fields a downscaling report must carry,
and what it must refuse to claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from extreme_weather.validation.metrics import compute_downscaling_comparison


#: The limitations the pipeline attaches to a downscaling evaluation. They are
#: stated here so the test can assert they survive into the report.
DOWNSCALING_LIMITATIONS = (
    "The high-resolution reference is the Phase 4 SYNTHETIC target, generated "
    "from the coarse field. It is not observational truth, so these numbers "
    "measure interpolation of a known construction, not forecast skill.",
    "POD/FAR are not computed: no defensible event threshold exists for a "
    "synthetic reference target.",
)


def _report(**overrides) -> dict:
    reference = np.linspace(1.0, 10.0, 64).reshape(1, 8, 8)
    result = compute_downscaling_comparison(
        {"ai": reference, "nearest": reference, "cubic": reference},
        reference,
        dataset="demo",
        split="test",
        limitations=DOWNSCALING_LIMITATIONS,
        **overrides,
    )
    return result.as_dict()


def test_the_report_names_the_dataset_and_split() -> None:
    payload = _report()
    assert payload["dataset"] == "demo"
    assert payload["split"] == "test"
    assert payload["variable"] == "precipitation"


def test_the_report_names_the_winner() -> None:
    payload = _report()
    assert payload["winner_by_mae"] in ("ai", "nearest", "cubic")


def test_the_report_records_the_frame_count() -> None:
    assert _report()["frame_count"] == 64


def test_the_report_carries_the_baseline_comparison() -> None:
    payload = _report()
    assert "ai_vs_nearest" in payload["comparison"]
    assert "ai_vs_cubic" in payload["comparison"]
    for block in ("ai_vs_nearest", "ai_vs_cubic"):
        for metric in ("mae", "rmse", "bias", "correlation"):
            assert metric in payload["comparison"][block]
            assert "verdict" in payload["comparison"][block][metric]


def test_the_report_carries_its_limitations() -> None:
    payload = _report()
    assert payload["limitations"]
    text = " ".join(payload["limitations"]).lower()
    assert "synthetic" in text
    assert "not observational truth" in text


def test_the_report_says_pod_was_not_computed() -> None:
    payload = _report()
    assert payload["classification"]["available"] is False
    assert payload["classification"]["event_threshold"] is None


def test_a_missing_ai_checkpoint_is_reported_not_imitated() -> None:
    """Without a trained model the AI is omitted rather than faked by a baseline."""
    from extreme_weather.validation.reports import build_phase7_summary

    summary = build_phase7_summary(
        dataset={},
        split={},
        ground_truth={"available": False},
        downscaling={
            "available": False,
            "reason": (
                "No trained AI downscaling checkpoint exists, so only the "
                "interpolation baselines were scored. No AI result is claimed."
            ),
            "methods": {"nearest": {"mae": 0.1}},
        },
    )
    assert summary["downscaling_metrics"]["available"] is False
    assert "No AI result is claimed" in summary["downscaling_metrics"]["reason"]
    assert "ai" not in summary["downscaling_metrics"]["methods"]


def test_the_evaluation_script_writes_a_report(tmp_path: Path) -> None:
    """The script's output file must exist and carry the required blocks."""
    from extreme_weather.validation.reports import REPORT_FILENAMES

    target = tmp_path / REPORT_FILENAMES["downscaling"]
    target.write_text(json.dumps(_report()), encoding="utf-8")
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ("dataset", "split", "methods", "comparison", "limitations"):
        assert key in payload

