"""Model and data integrity (SIH26078, Phase 10, section 7).

A corrupted model file is one of the few failure modes that produces
*confident wrong answers* rather than an exception. A truncated checkpoint, a
checkpoint whose architecture no longer matches the code, or a manifest whose
content hash no longer matches the file it describes - each of these would
otherwise be loaded and used.

So every artifact this project depends on is hashed, and the hash is checked
before use. Three states exist, and they are kept distinct on purpose:

``VALID``
    The file exists, is inside its expected directory, and hashes as expected.
``INCOMPATIBLE``
    The file is intact but its recorded metadata does not match the code
    asking for it - a different architecture, version or variable contract.
``CORRUPT``
    The file is missing, unreadable, or its hash does not match.

**A corrupt artifact is never silently used.** The loaders return a state
rather than raising deep inside a model constructor, because a caller that
receives ``CORRUPT`` can report it, and a caller that receives an exception from
a tensor constructor cannot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "COMPATIBLE",
    "CORRUPT",
    "INCOMPATIBLE",
    "MISSING",
    "VALID",
    "IntegrityResult",
    "check_checkpoint",
    "check_json_artifact",
    "file_sha256",
    "verify_manifest",
]

UTC = timezone.utc

#: The artifact passed every check.
VALID = "VALID"

#: The artifact is readable but its metadata does not match this code.
INCOMPATIBLE = "INCOMPATIBLE"

#: The artifact is missing, unreadable, or fails its hash check.
CORRUPT = "CORRUPT"

#: The artifact does not exist at all.
MISSING = "MISSING"

#: Read size when hashing. Hashing in chunks keeps memory flat on a large file.
_CHUNK = 1024 * 1024

#: Extensions accepted for a model checkpoint. Anything else is refused before
#: the file is opened, so a directory or a text file cannot be loaded as a model.
CHECKPOINT_SUFFIXES: frozenset[str] = frozenset({".pt", ".pth", ".ckpt", ".ptc"})


def file_sha256(path: Path | str, *, algorithm: str = "sha256") -> str:
    """Return the hash of a file's bytes.

    Args:
        path: The file to hash.
        algorithm: Any algorithm :mod:`hashlib` provides.

    Returns:
        The hex digest.

    Raises:
        OSError: If the file cannot be read. Callers that need a state rather
            than an exception should use :func:`check_checkpoint`.
    """
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IntegrityResult:
    """The outcome of one integrity check.

    Attributes:
        artifact: What was checked.
        state: VALID, INCOMPATIBLE, CORRUPT or MISSING.
        path: Where the artifact was looked for.
        expected_sha256: The hash the caller expected, when one was given.
        actual_sha256: The hash actually computed.
        reason: Why the state is not VALID.
        metadata: Whatever the check learned, e.g. a checkpoint's version.
    """

    artifact: str
    state: str
    path: str | None = None
    expected_sha256: str | None = None
    actual_sha256: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True only when the artifact passed every check."""
        return self.state == VALID

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "artifact": self.artifact,
            "state": self.state,
            "usable": self.usable,
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


def _resolve_within(base: Path, candidate: Path | str) -> Path | None:
    """Resolve ``candidate`` and confirm it stays inside ``base``.

    Returns:
        The resolved path, or ``None`` when it escapes ``base``.

    A checkpoint path arrives from configuration and from API requests, so
    treating it as untrusted is the correct default: a path outside the model
    directory must be refused, not merely logged.
    """
    root = Path(base).expanduser().resolve()
    target = (root / Path(candidate)).expanduser().resolve()
    if target != root and root not in target.parents:
        return None
    return target


def check_checkpoint(
    path: Path | str,
    *,
    model_dir: Path | str | None = None,
    expected_sha256: str | None = None,
    expected_version: str | None = None,
) -> IntegrityResult:
    """Validate a model checkpoint before it is used.

    The check order matters. A path that escapes the model directory is refused
    *before* the file is opened, so a traversal attempt never reaches the
    filesystem read.

    Args:
        path: The checkpoint path.
        model_dir: Directory the checkpoint must live inside. Skipped when
            ``None``, which is only appropriate for a local script run.
        expected_sha256: Hash the file must match.
        expected_version: Model version the code expects, compared against the
            checkpoint's own metadata.

    Returns:
        An :class:`IntegrityResult`. Never raises for a bad artifact; the
        state carries the reason instead.
    """
    raw = Path(path)
    resolved = _resolve_within(model_dir, raw) if model_dir is not None else raw.expanduser().resolve()

    if resolved is None:
        return IntegrityResult(
            artifact="checkpoint",
            state=CORRUPT,
            path=str(raw),
            reason=(
                f"The checkpoint path '{raw}' resolves outside the configured "
                f"model directory '{model_dir}'. Refused without opening the file."
            ),
        )

    if resolved.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return IntegrityResult(
            artifact="checkpoint",
            state=CORRUPT,
            path=str(resolved),
            reason=(
                f"Unsupported checkpoint extension '{resolved.suffix}'; expected one "
                f"of {', '.join(sorted(CHECKPOINT_SUFFIXES))}."
            ),
        )

    if not resolved.is_file():
        return IntegrityResult(
            artifact="checkpoint",
            state=MISSING,
            path=str(resolved),
            reason=f"No checkpoint exists at '{resolved}'.",
        )

    try:
        actual = file_sha256(resolved)
    except OSError as exc:
        return IntegrityResult(
            artifact="checkpoint",
            state=CORRUPT,
            path=str(resolved),
            reason=f"The checkpoint could not be read: {exc}",
        )

    if expected_sha256 is not None and actual != expected_sha256:
        return IntegrityResult(
            artifact="checkpoint",
            state=CORRUPT,
            path=str(resolved),
            expected_sha256=expected_sha256,
            actual_sha256=actual,
            reason=(
                "The checkpoint's SHA-256 does not match the expected value. The "
                "file is modified or corrupt and will not be used."
            ),
        )

    # A hash match proves the bytes are intact; it does not prove the bytes are
    # the right *model*. That needs the metadata comparison below.
    version: str | None = None
    try:
        from extreme_weather.gnn.checkpoint import load_checkpoint

        loaded = load_checkpoint(resolved)
        version = getattr(loaded, "model_version", None)
    except Exception as exc:  # noqa: BLE001 - any load failure is a state
        return IntegrityResult(
            artifact="checkpoint",
            state=CORRUPT,
            path=str(resolved),
            expected_sha256=expected_sha256,
            actual_sha256=actual,
            reason=(
                f"The checkpoint is intact but could not be loaded, so its "
                f"structure is unusable: {type(exc).__name__}: {exc}"
            ),
        )

    metadata = {"model_version": version}
    if expected_version is not None and version is not None and version != expected_version:
        return IntegrityResult(
            artifact="checkpoint",
            state=INCOMPATIBLE,
            path=str(resolved),
            expected_sha256=expected_sha256,
            actual_sha256=actual,
            reason=(
                f"The checkpoint records model version '{version}' but "
                f"'{expected_version}' was expected. The file is intact but the "
                "wrong model; it will not be used."
            ),
            metadata=metadata,
        )

    return IntegrityResult(
        artifact="checkpoint",
        state=VALID,
        path=str(resolved),
        expected_sha256=expected_sha256,
        actual_sha256=actual,
        metadata=metadata,
    )


def check_json_artifact(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    base_dir: Path | str | None = None,
    required_keys: tuple[str, ...] = (),
) -> IntegrityResult:
    """Validate a JSON artifact such as a manifest or a run record.

    Args:
        path: The artifact path.
        expected_sha256: Hash the file must match.
        base_dir: Directory the artifact must live inside.
        required_keys: Top-level keys the file must contain. A manifest missing
            its checksum is structurally wrong even if it parses.

    Returns:
        An :class:`IntegrityResult`.
    """
    raw = Path(path)
    resolved = (
        _resolve_within(base_dir, raw) if base_dir is not None else raw.expanduser().resolve()
    )
    if resolved is None:
        return IntegrityResult(
            artifact="json",
            state=CORRUPT,
            path=str(raw),
            reason=(
                f"The path '{raw}' resolves outside the permitted directory "
                f"'{base_dir}'. Refused without opening the file."
            ),
        )

    if not resolved.is_file():
        return IntegrityResult(
            artifact="json",
            state=MISSING,
            path=str(resolved),
            reason=f"No file exists at '{resolved}'.",
        )

    try:
        actual = file_sha256(resolved)
    except OSError as exc:
        return IntegrityResult(
            artifact="json",
            state=CORRUPT,
            path=str(resolved),
            reason=f"The file could not be read: {exc}",
        )

    if expected_sha256 is not None and actual != expected_sha256:
        return IntegrityResult(
            artifact="json",
            state=CORRUPT,
            path=str(resolved),
            expected_sha256=expected_sha256,
            actual_sha256=actual,
            reason="The file's SHA-256 does not match the expected value.",
        )

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return IntegrityResult(
            artifact="json",
            state=CORRUPT,
            path=str(resolved),
            actual_sha256=actual,
            reason=f"The file is not valid JSON: {type(exc).__name__}: {exc}",
        )

    if not isinstance(payload, Mapping):
        return IntegrityResult(
            artifact="json",
            state=CORRUPT,
            path=str(resolved),
            actual_sha256=actual,
            reason=(
                f"Expected a JSON object, found {type(payload).__name__}. A "
                "truncated write often produces this."
            ),
        )

    missing = [key for key in required_keys if key not in payload]
    if missing:
        return IntegrityResult(
            artifact="json",
            state=INCOMPATIBLE,
            path=str(resolved),
            actual_sha256=actual,
            reason=f"The artifact is missing required key(s): {', '.join(missing)}.",
        )

    return IntegrityResult(
        artifact="json",
        state=VALID,
        path=str(resolved),
        expected_sha256=expected_sha256,
        actual_sha256=actual,
        metadata={"keys": sorted(str(k) for k in payload)[:20]},
    )


def verify_manifest(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
    verify_file_hash: bool = True,
) -> IntegrityResult:
    """Validate an ingestion manifest and, optionally, the file it describes.

    A manifest records the hash of the dataset it describes. This checks the
    manifest parses, carries the expected keys, and - when asked - that the
    dataset on disk still hashes to the recorded value. The second check is
    what catches a file that was modified after ingestion.

    Args:
        path: The manifest path.
        base_dir: Directory the manifest must live inside.
        verify_file_hash: Whether to re-hash the described dataset. Skipped when
            the dataset is not reachable from the manifest.

    Returns:
        An :class:`IntegrityResult`.
    """
    required = ("dataset_id", "source", "file_name", "file_hash")
    result = check_json_artifact(
        path, base_dir=base_dir, required_keys=required
    )
    if not result.usable:
        return result

    if not verify_file_hash:
        return result

    try:
        with Path(result.path or "").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - already parsed
        return IntegrityResult(
            artifact="manifest",
            state=CORRUPT,
            path=result.path,
            reason=f"The manifest became unreadable during verification: {exc}",
        )

    recorded = str(manifest.get("file_hash") or "")
    file_name = str(manifest.get("file_name") or "")
    if not recorded or not file_name:
        return IntegrityResult(
            artifact="manifest",
            state=INCOMPATIBLE,
            path=result.path,
            actual_sha256=result.actual_sha256,
            reason="The manifest records no file hash to verify against.",
        )

    dataset = Path(result.path or "").parent / file_name
    if not dataset.is_file():
        return IntegrityResult(
            artifact="manifest",
            state=MISSING,
            path=str(dataset),
            actual_sha256=result.actual_sha256,
            reason=(
                f"The manifest describes '{file_name}' but that file is not "
                "present beside it."
            ),
            metadata={"file_hash": recorded},
        )

    try:
        actual = file_sha256(dataset)
    except OSError as exc:
        return IntegrityResult(
            artifact="manifest",
            state=CORRUPT,
            path=str(dataset),
            actual_sha256=result.actual_sha256,
            reason=f"The dataset could not be read: {exc}",
        )

    if actual != recorded:
        return IntegrityResult(
            artifact="manifest",
            state=CORRUPT,
            path=str(dataset),
            expected_sha256=recorded,
            actual_sha256=actual,
            reason=(
                "The dataset's SHA-256 no longer matches the manifest. The file "
                "changed after ingestion and will not be trusted."
            ),
        )

    return IntegrityResult(
        artifact="manifest",
        state=VALID,
        path=result.path,
        actual_sha256=result.actual_sha256,
        metadata={"file_hash": recorded, "dataset_id": manifest.get("dataset_id")},
    )
