"""Tests for the Phase 7 validation API.

Two behaviours matter most here:

* **404 when a report does not exist.** A 200 with empty fields would be
  indistinguishable from a report saying "zero", and the second is a claim this
  project cannot support.
* **The limitations travel with the payload.** A client must not be able to read
  a metric without the caveat that says what it is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from extreme_weather.validation.reports import (
    CANONICAL_LIMITATIONS,
    REPORT_FILENAMES,
    build_phase7_summary,
    read_report,
    render_markdown,
    write_reports,
)


# --------------------------------------------------------------------------- #
# Without any report
# --------------------------------------------------------------------------- #
def test_status_answers_200_even_with_no_run(validation_client: TestClient) -> None:
    """Absence is information, not an error - the status endpoint reports it."""
    response = validation_client.get("/api/v1/validation/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["has_run"] is False
    assert payload["reports_present"] == []
    assert payload["ground_truth_available"] is False
    assert "metric not computed" in payload["ground_truth_note"]


def test_summary_returns_404_without_a_run(validation_client: TestClient) -> None:
    response = validation_client.get("/api/v1/validation/summary")
    assert response.status_code == 404
    assert "run_phase7_validation" in response.json()["detail"]


def test_metrics_returns_404_without_a_run(validation_client: TestClient) -> None:
    response = validation_client.get("/api/v1/validation/metrics")
    assert response.status_code == 404


def test_provenance_returns_404_without_a_run(validation_client: TestClient) -> None:
    response = validation_client.get("/api/v1/validation/provenance")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# With reports present
# --------------------------------------------------------------------------- #
@pytest.fixture()
def with_reports(validation_client: TestClient, validation_settings) -> TestClient:
    """Write a full set of reports, then hand back the client.

    The reports go into the *test's* data directory, which is what the client's
    settings override points at - not the repository's real `data/`.
    """
    from extreme_weather.validation.dataset_builder import ValidationDatasetBuilder

    builder = ValidationDatasetBuilder(validation_settings)
    builder.ensure_directories()
    for name in REPORT_FILENAMES:
        if name.startswith("summary"):
            continue
        (builder.reports_dir / REPORT_FILENAMES[name]).write_text(
            json.dumps({"marker": name}), encoding="utf-8"
        )
    # The summary has to satisfy its own Pydantic response model, so a real one
    # is built rather than a stub.
    write_reports(
        builder.reports_dir,
        build_phase7_summary(
            dataset={"dataset_id": "demo"},
            split={"splits": []},
            ground_truth={"status": "UNAVAILABLE", "available": False},
        ),
    )
    return validation_client


def test_status_lists_the_reports_that_exist(with_reports: TestClient) -> None:
    payload = with_reports.get("/api/v1/validation/status").json()
    assert payload["has_run"] is True
    assert "gnn" in payload["reports_present"]


def test_summary_is_served_once_written(with_reports: TestClient) -> None:
    response = with_reports.get("/api/v1/validation/summary")
    assert response.status_code == 200
    assert response.json()["schema"] == "phase7_summary/1.0"


def test_metrics_marks_each_report_available_or_not(with_reports: TestClient) -> None:
    payload = with_reports.get("/api/v1/validation/metrics").json()
    assert payload["reports"]["gnn"]["available"] is True
    assert "ground_truth_note" in payload
    assert "metric not computed" in payload["ground_truth_note"]


def test_a_partially_produced_set_is_marked_missing(
    validation_client: TestClient, validation_settings
) -> None:
    from extreme_weather.validation.dataset_builder import ValidationDatasetBuilder

    builder = ValidationDatasetBuilder(validation_settings)
    builder.ensure_directories()
    (builder.reports_dir / REPORT_FILENAMES["gnn"]).write_text(
        json.dumps({"dataset": "demo"}), encoding="utf-8"
    )
    payload = validation_client.get("/api/v1/validation/metrics").json()
    assert payload["reports"]["gnn"]["available"] is True
    assert payload["reports"]["alerts"]["available"] is False
    assert "reason" in payload["reports"]["alerts"]
    assert "alerts" in payload["reports_missing"]


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def test_the_summary_always_carries_every_section() -> None:
    summary = build_phase7_summary(
        dataset={"dataset_id": "demo"},
        split={"splits": []},
        ground_truth={"status": "UNAVAILABLE", "available": False},
    )
    for key in (
        "dataset_provenance",
        "temporal_coverage",
        "ground_truth_availability",
        "climatology_diagnostics",
        "threshold_calibration",
        "anomaly_diagnostics",
        "tracking_diagnostics",
        "gnn_metrics",
        "downscaling_metrics",
        "alert_diagnostics",
        "limitations",
        "reproducibility",
    ):
        assert key in summary, key


def test_a_missing_section_is_marked_not_produced() -> None:
    """'Not produced' and 'empty' must be distinguishable."""
    summary = build_phase7_summary(
        dataset={}, split={}, ground_truth={"available": False}
    )
    assert summary["gnn_metrics"]["available"] is False
    assert "reason" in summary["gnn_metrics"]
    assert "gnn" in summary["reports_missing"]


def test_the_canonical_limitations_are_always_present() -> None:
    summary = build_phase7_summary(dataset={}, split={}, ground_truth={})
    assert summary["limitations"][: len(CANONICAL_LIMITATIONS)] == list(
        CANONICAL_LIMITATIONS
    )


def test_the_limitations_refuse_every_operational_claim() -> None:
    text = " ".join(CANONICAL_LIMITATIONS).lower()
    for claim in ("production-ready", "operational", "100% accurate", "exact prediction"):
        assert claim in text


def test_extra_limitations_are_appended() -> None:
    summary = build_phase7_summary(
        dataset={}, split={}, ground_truth={}, extra_limitations=("A specific one.",)
    )
    assert summary["limitations"][-1] == "A specific one."


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_the_markdown_leads_with_the_refusal() -> None:
    summary = build_phase7_summary(
        dataset={"dataset_id": "demo"},
        split={"splits": [{"name": "train", "size": 60, "start": "a", "end": "b"}]},
        ground_truth={"status": "UNAVAILABLE"},
    )
    text = render_markdown(summary)
    assert "Ground truth unavailable - metric not computed." in text
    assert "Research prototype" in text
    # The ground-truth section must come before the counts.
    assert text.index("## Ground truth") < text.index("## Dataset provenance")


def test_the_markdown_states_the_split_is_chronological() -> None:
    summary = build_phase7_summary(dataset={}, split={}, ground_truth={})
    assert "max(train) < min(validation)" in render_markdown(summary)


def test_writing_reports_produces_both_files(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path, build_phase7_summary(dataset={}, split={}, ground_truth={})
    )
    assert paths["json"].is_file() and paths["markdown"].is_file()
    assert read_report(tmp_path, "summary") is not None


def test_reading_a_missing_report_returns_none(tmp_path: Path) -> None:
    assert read_report(tmp_path, "gnn") is None


def test_reading_an_unknown_report_key_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown report"):
        read_report(tmp_path, "not_a_report")


# --------------------------------------------------------------------------- #
# The routes are registered
# --------------------------------------------------------------------------- #
def test_every_validation_route_is_registered(validation_client: TestClient) -> None:
    paths = validation_client.get("/openapi.json").json()["paths"]
    for route in (
        "/api/v1/validation/status",
        "/api/v1/validation/summary",
        "/api/v1/validation/metrics",
        "/api/v1/validation/provenance",
    ):
        assert route in paths, route


def test_earlier_phases_keep_working(validation_client: TestClient) -> None:
    """Registering the validation router must not disturb Phases 0-6."""
    assert validation_client.get("/health").status_code == 200
    assert validation_client.get("/api/v1/datasets").status_code == 200
    assert validation_client.get("/api/v1/data/sources").status_code == 200
    assert validation_client.get("/api/v1/alerts").status_code == 200

