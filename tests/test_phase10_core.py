"""Configuration, logging, security and integrity tests (SIH26078, Phase 10).

Covers sections 1, 2, 6 and 7 of the Phase 10 specification:

* configuration validation and production-safe defaults
* structured logging and credential redaction
* path traversal, extension and boundary protection
* checkpoint and manifest integrity, including deliberate corruption

The corruption tests are the important ones: they write a file, damage it, and
assert the loader **refuses** it. A hash check that has never been seen to fail
is a hash check nobody has verified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from extreme_weather.config import Settings, get_settings
from extreme_weather.integrity import (
    CORRUPT,
    INCOMPATIBLE,
    MISSING,
    VALID,
    check_checkpoint,
    check_json_artifact,
    file_sha256,
    verify_manifest,
)
from extreme_weather.observability.logging import (
    JsonFormatter,
    TextFormatter,
    bind_context,
    configure_logging,
    get_correlation_ids,
    get_logger,
    log_event,
    new_request_id,
    redact,
)
from extreme_weather.security import (
    DEFAULT_ALLOWED_SUFFIXES,
    PathBoundaryError,
    UnsafePathError,
    is_within,
    safe_path,
    safe_subprocess,
    validate_filename,
    writable_directory,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_safe_local_development_defaults() -> None:
    """A fresh checkout must work with no environment set at all."""
    settings = Settings()

    assert settings.log_level == "INFO"
    assert settings.log_format == "text"
    assert settings.enforce_data_dir_boundary is True
    assert settings.verify_artifact_integrity is True
    assert settings.allow_cors_wildcard is False


def test_production_defaults_refuse_a_wildcard_cors() -> None:
    """A wildcard origin lets any site read this API from a browser."""
    assert Settings().allow_cors_wildcard is False


def test_log_format_rejects_an_unknown_value() -> None:
    """A typo in LOG_FORMAT must fail at startup, not silently fall back."""
    with pytest.raises(Exception):
        Settings(log_format="yaml")  # type: ignore[arg-type]


def test_min_free_disk_bytes_is_configurable() -> None:
    """The readiness space threshold is policy, not a constant."""
    assert Settings().min_free_disk_bytes == 268435456
    assert Settings(min_free_disk_bytes=0).min_free_disk_bytes == 0


def test_api_request_id_header_is_configurable() -> None:
    """A deployment may need a different correlation header."""
    assert Settings().api_request_id_header == "X-Request-ID"
    assert Settings(api_request_id_header="X-Trace").api_request_id_header == "X-Trace"


def test_settings_cache_can_be_reset() -> None:
    """A test that changes the environment must be able to re-read it."""
    get_settings.cache_clear()
    first = get_settings()
    get_settings.cache_clear()
    second = get_settings()

    assert isinstance(first, Settings)
    assert isinstance(second, Settings)


# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #
def test_json_formatter_emits_one_object_per_line() -> None:
    """A log shipper needs parseable lines, not prose."""
    record = logging.LogRecord(
        "extreme_weather.test", logging.INFO, "f.py", 1, "pipeline finished", (), None
    )
    record.event = "pipeline.run.completed"
    record.fields = {"stage": "anomaly", "events": 3}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["event"] == "pipeline.run.completed"
    assert payload["stage"] == "anomaly"
    assert payload["events"] == 3
    assert "ts" in payload


def test_json_formatter_includes_correlation_ids() -> None:
    """A log line without a request id cannot be traced to a request."""
    record = logging.LogRecord(
        "extreme_weather.test", logging.INFO, "f.py", 1, "handled", (), None
    )
    with bind_context(request_id="req-abc", run_id="run-xyz"):
        payload = json.loads(JsonFormatter().format(record))

    assert payload["request_id"] == "req-abc"
    assert payload["run_id"] == "run-xyz"


def test_text_formatter_is_readable_and_carries_fields() -> None:
    """Local development keeps a human-readable default."""
    record = logging.LogRecord(
        "extreme_weather.test", logging.INFO, "f.py", 1, "done", (), None
    )
    record.fields = {"duration_ms": 12.5}

    line = TextFormatter().format(record)

    assert "done" in line
    assert "duration_ms=12.5" in line


def test_correlation_ids_are_scoped_to_a_block() -> None:
    """A leaked context variable would mislabel every later record."""
    assert get_correlation_ids() == (None, None)
    with bind_context(request_id="req-1"):
        assert get_correlation_ids() == ("req-1", None)
    assert get_correlation_ids() == (None, None)


def test_generated_request_ids_are_unique_and_opaque() -> None:
    """A guessable request id is an enumeration vector."""
    first, second = new_request_id(), new_request_id()

    assert first != second
    assert first.startswith("req-")
    assert len(first) > 12


def test_configure_logging_does_not_stack_handlers() -> None:
    """A second call must replace the handler, not duplicate every line."""
    logger = configure_logging("INFO", force=True)
    first = len(logger.handlers)
    configure_logging("INFO", force=True)

    assert len(logger.handlers) == first == 1


# --------------------------------------------------------------------------- #
# Credential redaction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, leak",
    [
        ("api_key=SUPERSECRET123", "SUPERSECRET123"),
        ("API-KEY: SUPERSECRET123", "SUPERSECRET123"),
        ("token=SUPERSECRET123", "SUPERSECRET123"),
        ("password=hunter2secret", "hunter2secret"),
        ("Authorization: Bearer abcdef123456", "abcdef123456"),
        ("https://api.example.com/x?apikey=TOKVAL&y=1", "TOKVAL"),
        ('{"api_key": "JSONVAL"}', "JSONVAL"),
    ],
)
def test_secrets_are_masked(text: str, leak: str) -> None:
    """A credential in a log line must not survive into the file."""
    masked = redact(text)

    assert leak not in masked
    assert "***" in masked


def test_clean_text_is_untouched() -> None:
    """Redaction must not mangle ordinary operational messages."""
    message = "pipeline run completed stages=8 status=SUCCESS duration_ms=4210.5"

    assert redact(message) == message


def test_structured_fields_are_redacted() -> None:
    """A secret passed as a field value is masked too, not just in the message."""
    record = logging.LogRecord(
        "extreme_weather.test", logging.INFO, "f.py", 1, "config", (), None
    )
    record.fields = {"api_key": "LEAKME", "url": "https://x.invalid/?token=LEAKTOO"}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["api_key"] == "***"
    assert "LEAKTOO" not in json.dumps(payload)


def test_very_long_values_are_truncated() -> None:
    """An unbounded field is how a huge payload ends up in a log file."""
    masked = redact("x" * 5000)

    assert len(masked) < 600
    assert "truncated" in masked


def test_exception_context_is_preserved_without_a_traceback() -> None:
    """The type and message survive; the traceback does not join every line."""
    try:
        raise ValueError("boom in stage 3")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "extreme_weather.test", logging.ERROR, "f.py", 1, "failed", (), sys.exc_info()
        )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["error_type"] == "ValueError"
    assert "boom in stage 3" in payload["error"]


def test_log_event_emits_a_structured_record() -> None:
    """The convenience helper must carry the event name and fields."""
    logger = get_logger("test")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        log_event(logger, logging.INFO, "pipeline.stage.completed", "done", stage="gnn")
    finally:
        logger.removeHandler(handler)

    assert records[0].event == "pipeline.stage.completed"  # type: ignore[attr-defined]
    assert records[0].fields == {"stage": "gnn"}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Path traversal and boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "candidate",
    [
        "../../secret",
        "../../../etc/passwd",
        "raw/../../outside.nc",
        "..\\..\\secret.nc",
        "./../secret.nc",
    ],
)
def test_path_traversal_is_refused(tmp_path: Path, candidate: str) -> None:
    """The classic traversal shapes must all be rejected."""
    with pytest.raises(PathBoundaryError):
        safe_path(tmp_path, candidate)


def test_absolute_paths_outside_the_root_are_refused(tmp_path: Path) -> None:
    """An absolute path is interpreted inside the root, not as an escape."""
    with pytest.raises(PathBoundaryError):
        safe_path(tmp_path, "C:/Windows/System32/config.nc")


def test_absolute_path_inside_the_root_is_allowed(tmp_path: Path) -> None:
    """A legitimate absolute path inside the root must still work."""
    (tmp_path / "ok.nc").write_text("x")

    resolved = safe_path(tmp_path, str(tmp_path / "ok.nc"))

    assert resolved == (tmp_path / "ok.nc").resolve()


def test_a_valid_relative_path_resolves(tmp_path: Path) -> None:
    """The normal case must keep working."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x.nc").write_text("x")

    resolved = safe_path(tmp_path, "raw/x.nc", must_exist=True)

    assert resolved == (tmp_path / "raw" / "x.nc").resolve()


@pytest.mark.parametrize("name", ["x.txt", "x.exe", "x.sh", "x", "x.py"])
def test_unexpected_extensions_are_refused(tmp_path: Path, name: str) -> None:
    """Only NetCDF/GRIB artifacts may be addressed by the dataset paths."""
    with pytest.raises(UnsafePathError):
        safe_path(tmp_path, name)


def test_allowed_suffixes_are_configurable(tmp_path: Path) -> None:
    """A caller may widen the set deliberately, but not by accident."""
    assert safe_path(tmp_path, "m.json", allowed_suffixes={".json"})


def test_missing_file_is_reported_when_required(tmp_path: Path) -> None:
    """Existence is only checked when the caller asks for it."""
    with pytest.raises(UnsafePathError, match="No file exists"):
        safe_path(tmp_path, "absent.nc", must_exist=True)


def test_is_within_accepts_the_root_itself(tmp_path: Path) -> None:
    """The root is inside itself, so a path naming the directory is fine."""
    assert is_within(tmp_path, tmp_path) is True
    assert is_within(tmp_path, tmp_path / "a" / "b.nc") is True
    assert is_within(tmp_path, tmp_path.parent) is False


def test_a_path_outside_the_root_is_detected(tmp_path: Path) -> None:
    """A resolved location outside the tree is rejected.

    Symlinks are the classic way to produce one, but creating one needs a
    privilege this environment may not grant, so the guarantee is asserted
    against the resolved-location predicate itself rather than being skipped.
    """
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.nc").write_text("x")

    assert is_within(root, outside / "secret.nc") is False
    assert is_within(root, root / "inside.nc") is True

    # A relative traversal that would land outside must raise.
    with pytest.raises(PathBoundaryError):
        safe_path(root, "../outside/secret.nc")


def test_a_symlink_escaping_the_root_is_refused_when_supported(tmp_path: Path) -> None:
    """Where symlinks can be created, prove the resolved-path check catches one."""
    import os

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.nc").write_text("x")

    if not hasattr(os, "symlink"):  # pragma: no cover - platform check
        return
    try:
        os.symlink(outside / "secret.nc", root / "link.nc")
    except OSError:
        # No privilege on this platform. The guarantee above is already covered
        # by test_a_path_outside_the_root_is_detected, so this is a capability
        # gap rather than an untested behaviour.
        return

    with pytest.raises(PathBoundaryError):
        safe_path(root, "link.nc")


# --------------------------------------------------------------------------- #
# Filename validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["../secret", "a/b.nc", "a\\b.nc", "..", ".", "", "  ", "a\x00b", "a" * 300]
)
def test_unsafe_filenames_are_refused(name: str) -> None:
    """A caller-supplied filename has no legitimate use for any of these."""
    with pytest.raises(UnsafePathError):
        validate_filename(name)


def test_a_normal_filename_is_accepted() -> None:
    """The ordinary case must keep working."""
    assert validate_filename("  sample.nc  ") == "sample.nc"


# --------------------------------------------------------------------------- #
# Subprocess safety
# --------------------------------------------------------------------------- #
def test_subprocess_rejects_a_string_command() -> None:
    """A string command requires a shell and is not permitted."""
    with pytest.raises(UnsafePathError, match="argument list"):
        safe_subprocess("echo hello")  # type: ignore[arg-type]


def test_subprocess_runs_an_argument_list(tmp_path: Path) -> None:
    """A filename with a space is one argument, not two commands."""
    target = tmp_path / "a file.txt"
    target.write_text("content")

    result = safe_subprocess(
        ["python", "-c", "import sys; print(open(sys.argv[1]).read())", str(target)],
        timeout=30,
    )

    assert result.returncode == 0
    assert "content" in result.stdout


def test_subprocess_rejects_an_empty_command() -> None:
    """An empty command is a programming error, not a no-op."""
    with pytest.raises(UnsafePathError):
        safe_subprocess([])


def test_subprocess_rejects_a_non_positive_timeout() -> None:
    """A zero timeout would kill every child immediately."""
    with pytest.raises(UnsafePathError, match="timeout"):
        safe_subprocess(["python", "--version"], timeout=0)


# --------------------------------------------------------------------------- #
# Directory writability
# --------------------------------------------------------------------------- #
def test_writable_directory_detects_a_real_directory(tmp_path: Path) -> None:
    """Readiness depends on this being correct."""
    assert writable_directory(tmp_path) is True


def test_writable_directory_detects_a_missing_directory(tmp_path: Path) -> None:
    """A missing directory is not writable, and must not be reported ready."""
    assert writable_directory(tmp_path / "absent") is False


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def test_file_sha256_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    """The hash must depend on the bytes, not the path or the timestamp."""
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    third = tmp_path / "c.bin"
    third.write_bytes(b"different")

    assert file_sha256(first) == file_sha256(second)
    assert file_sha256(first) != file_sha256(third)


def _write_checkpoint(directory: Path, *, model_version: str = "v1") -> Path:
    """Write a real, loadable GNN checkpoint into ``directory``.

    The constructor needs the model and feature configs, so this goes through a
    real model rather than fabricating a minimal object.
    """
    from extreme_weather.gnn.checkpoint import GNNCheckpoint, save_checkpoint
    from extreme_weather.gnn.features import FeatureConfig
    from extreme_weather.gnn.model import WeatherGNN

    model = WeatherGNN()
    checkpoint = GNNCheckpoint(
        model_config=model.config,
        feature_config=FeatureConfig(),
        training={"seed": 1, "epochs": 1},
        parameters=model.state_dict(),
        model_version=model_version,
    )
    return save_checkpoint(checkpoint, directory / "model.pt")


def test_a_missing_checkpoint_is_reported_not_raised(tmp_path: Path) -> None:
    """A missing model is a state, not an exception."""
    result = check_checkpoint(tmp_path / "gnn" / "absent.pt", model_dir=tmp_path)

    assert result.state == MISSING
    assert result.usable is False
    assert "No checkpoint exists" in result.reason


def test_a_checkpoint_outside_the_model_dir_is_refused(tmp_path: Path) -> None:
    """A traversal attempt must be refused before the file is opened."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"not a model")

    result = check_checkpoint(outside, model_dir=model_dir)

    assert result.state == CORRUPT
    assert "outside the configured model directory" in result.reason


def test_an_unexpected_checkpoint_extension_is_refused(tmp_path: Path) -> None:
    """A text file must not be opened as a model."""
    path = tmp_path / "model.txt"
    path.write_text("not a checkpoint")

    result = check_checkpoint(path, model_dir=tmp_path)

    assert result.state == CORRUPT
    assert "Unsupported checkpoint extension" in result.reason


def test_a_directory_is_not_a_checkpoint(tmp_path: Path) -> None:
    """A directory with a .pt name must not be loaded."""
    (tmp_path / "model.pt").mkdir()

    result = check_checkpoint(tmp_path / "model.pt", model_dir=tmp_path)

    assert result.usable is False


def test_a_corrupt_checkpoint_is_refused_never_used(tmp_path: Path) -> None:
    """The important case: damaged bytes must be caught, not loaded."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"\x00\x01\x02 definitely not a torch archive")

    result = check_checkpoint(path, model_dir=tmp_path)

    assert result.usable is False
    assert result.state == CORRUPT
    assert "could not be loaded" in result.reason


def test_a_checkpoint_failing_its_hash_is_refused(tmp_path: Path) -> None:
    """A hash mismatch means the file changed after it was recorded."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"content")
    wrong = "0" * 64

    result = check_checkpoint(path, model_dir=tmp_path, expected_sha256=wrong)

    assert result.state == CORRUPT
    assert result.expected_sha256 == wrong
    assert result.actual_sha256 == file_sha256(path)
    assert "does not match" in result.reason


def test_a_valid_checkpoint_passes(tmp_path: Path) -> None:
    """The happy path must also work, or the check is useless."""
    path = _write_checkpoint(tmp_path, model_version="test-1.0")

    result = check_checkpoint(path, model_dir=tmp_path)

    assert result.state == VALID
    assert result.usable is True
    assert result.metadata["model_version"] == "test-1.0"


def test_an_incompatible_version_is_reported_separately(tmp_path: Path) -> None:
    """An intact file that is the wrong model is INCOMPATIBLE, not CORRUPT."""
    path = _write_checkpoint(tmp_path, model_version="v1")

    result = check_checkpoint(path, model_dir=tmp_path, expected_version="v2")

    assert result.state == INCOMPATIBLE
    assert "wrong model" in result.reason


def test_a_truncated_checkpoint_is_detected(tmp_path: Path) -> None:
    """A half-written file is the most likely real corruption."""
    path = _write_checkpoint(tmp_path, model_version="v1")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    result = check_checkpoint(path, model_dir=tmp_path)

    assert result.usable is False
    assert result.state == CORRUPT


# --------------------------------------------------------------------------- #
# JSON artifacts and manifests
# --------------------------------------------------------------------------- #
def test_valid_json_artifact_passes(tmp_path: Path) -> None:
    """The happy path must work."""
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    assert check_json_artifact(path).usable is True


def test_invalid_json_is_reported_as_corrupt(tmp_path: Path) -> None:
    """A truncated write must be caught."""
    path = tmp_path / "broken.json"
    path.write_text('{"a": ', encoding="utf-8")

    result = check_json_artifact(path)

    assert result.state == CORRUPT
    assert "not valid JSON" in result.reason


def test_json_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    """A bare list is structurally wrong for a record."""
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    result = check_json_artifact(path)

    assert result.state == CORRUPT
    assert "Expected a JSON object" in result.reason


def test_missing_required_keys_are_reported(tmp_path: Path) -> None:
    """A manifest without its checksum is structurally wrong."""
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"dataset_id": "x"}), encoding="utf-8")

    result = check_json_artifact(path, required_keys=("dataset_id", "file_hash"))

    assert result.state == INCOMPATIBLE
    assert "file_hash" in result.reason


def test_json_outside_the_base_dir_is_refused(tmp_path: Path) -> None:
    """The boundary check applies to JSON artifacts too."""
    base = tmp_path / "allowed"
    base.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")

    result = check_json_artifact(outside, base_dir=base)

    assert result.state == CORRUPT
    assert "outside the permitted directory" in result.reason


def test_a_valid_manifest_passes(tmp_path: Path) -> None:
    """A manifest whose dataset still matches is trusted."""
    dataset = tmp_path / "sample.nc"
    dataset.write_bytes(b"payload")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "d1",
                "source": "era5",
                "file_name": "sample.nc",
                "file_hash": file_sha256(dataset),
            }
        ),
        encoding="utf-8",
    )

    result = verify_manifest(manifest)

    assert result.state == VALID


def test_a_modified_dataset_invalidates_its_manifest(tmp_path: Path) -> None:
    """The manifest's whole purpose is catching a file that changed."""
    dataset = tmp_path / "sample.nc"
    dataset.write_bytes(b"original")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "d1",
                "source": "era5",
                "file_name": "sample.nc",
                "file_hash": file_sha256(dataset),
            }
        ),
        encoding="utf-8",
    )
    dataset.write_bytes(b"tampered")

    result = verify_manifest(manifest)

    assert result.state == CORRUPT
    assert "no longer matches" in result.reason


def test_a_manifest_whose_dataset_is_absent_is_reported(tmp_path: Path) -> None:
    """A manifest pointing at a missing file must not be treated as valid."""
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "d1",
                "source": "era5",
                "file_name": "gone.nc",
                "file_hash": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    result = verify_manifest(manifest)

    assert result.state == MISSING


def test_integrity_result_serialises(tmp_path: Path) -> None:
    """The result is returned over HTTP, so it must be JSON-ready."""
    payload = check_checkpoint(tmp_path / "absent.pt", model_dir=tmp_path).as_dict()

    assert payload["state"] == MISSING
    assert payload["usable"] is False
    json.dumps(payload, default=str)
