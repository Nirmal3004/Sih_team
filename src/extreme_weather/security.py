"""Filesystem boundary and input-safety helpers (SIH26078, Phase 10).

What this module does
---------------------
Every path that arrives from outside the process - a query parameter, a JSON
body, a configured directory - is untrusted until proven otherwise. This module
is where that proof happens.

Three protections, and why each exists:

**Path traversal.** A caller-supplied ``../../etc/passwd`` must not resolve
outside the configured directory. The check resolves both sides and compares
parents, which defeats ``..`` sequences, absolute paths, symlink escapes and
Windows drive-relative paths alike.

**Extension allow-listing.** A path ending in ``.txt`` should not be opened as
a model checkpoint, and a ``.nc`` should not be executed. The allowed set is a
parameter with a safe default, so a caller cannot widen it by accident.

**Subprocess safety.** Where this project shells out, it does so with an
argument *list* and ``shell=False``, so a filename containing a space or a
semicolon is one argument rather than a shell command.

What this module does not do
----------------------------
It does not implement authentication. There is none in this project, and
pretending otherwise would be the most misleading thing a hardening phase could
do. See ``docs/SECURITY.md`` for what that means in practice.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "DEFAULT_ALLOWED_SUFFIXES",
    "PathBoundaryError",
    "UnsafePathError",
    "is_within",
    "safe_path",
    "safe_subprocess",
    "validate_filename",
]

#: Extensions a dataset-like artifact may have. NetCDF is what this project
#: reads; GRIB is accepted because the Phase 6 ingestion layer supports it.
DEFAULT_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".nc", ".nc4", ".grb", ".grib", ".grb2"})

#: Characters forbidden in a caller-supplied filename. Rejected outright rather
#: than escaped, because a filename containing them has no legitimate use here.
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


class PathBoundaryError(ValueError):
    """Raised when a path resolves outside its permitted directory."""


class UnsafePathError(ValueError):
    """Raised when a path or filename fails validation."""


def is_within(base: Path | str, candidate: Path | str) -> bool:
    """Return True when ``candidate`` resolves inside ``base``.

    Both sides are fully resolved first, so a symlink that points out of the
    directory is caught rather than followed.

    Args:
        base: The permitted directory.
        candidate: The path to test.

    Returns:
        True when the resolved path is the base itself or below it.
    """
    try:
        root = Path(base).expanduser().resolve()
        target = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError):
        # An unresolvable path (a broken symlink loop, for example) is not
        # inside anything.
        return False
    return target == root or root in target.parents


def safe_path(
    base: Path | str,
    candidate: Path | str,
    *,
    allowed_suffixes: Iterable[str] | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a caller-supplied path inside a permitted directory.

    Args:
        base: The permitted directory. The resolved result is guaranteed to be
            at or below it.
        candidate: The untrusted path, absolute or relative to ``base``.
        allowed_suffixes: Extensions the path must carry. ``None`` applies
            :data:`DEFAULT_ALLOWED_SUFFIXES`.
        must_exist: Require the resolved path to exist.

    Returns:
        The resolved absolute path.

    Raises:
        PathBoundaryError: If the path escapes ``base``.
        UnsafePathError: If the extension is not allowed, or the file is
            missing when ``must_exist`` is set.
    """
    root = Path(base).expanduser().resolve()
    raw = Path(candidate)

    # An absolute path is checked against the root and refused if it is not
    # already inside it. Reinterpreting "/etc/passwd" as "<root>/etc/passwd"
    # would silently answer a different question than the caller asked.
    if raw.is_absolute():
        try:
            candidate_resolved = raw.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise PathBoundaryError(
                f"The absolute path '{raw}' could not be resolved safely: {exc}"
            ) from exc
        root_resolved = root
        if candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents:
            raise PathBoundaryError(
                f"The absolute path '{raw}' is outside the permitted directory "
                f"'{root_resolved}'. Paths supplied from outside the process must "
                "stay inside the configured directory."
            )
    joined = raw
    try:
        resolved = (root / joined).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise PathBoundaryError(
            f"The path '{candidate}' could not be resolved safely: {exc}"
        ) from exc

    if resolved != root and root not in resolved.parents:
        raise PathBoundaryError(
            f"The path '{candidate}' resolves outside the permitted directory "
            f"'{root}'. Paths supplied from outside the process must stay "
            "inside the configured directory."
        )

    suffixes = (
        DEFAULT_ALLOWED_SUFFIXES
        if allowed_suffixes is None
        else frozenset(s.lower() for s in allowed_suffixes)
    )
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise UnsafePathError(
            f"Unsupported file extension '{resolved.suffix}'; expected one of "
            f"{', '.join(sorted(suffixes))}."
        )

    if must_exist and not resolved.is_file():
        raise UnsafePathError(f"No file exists at '{resolved}'.")

    return resolved


def validate_filename(name: str, *, max_length: int = 255) -> str:
    """Validate a caller-supplied filename.

    Args:
        name: The candidate filename.
        max_length: Longest accepted name.

    Returns:
        The validated name, stripped.

    Raises:
        UnsafePathError: If the name is empty, too long, contains a path
            separator, a parent reference, or a control character.
    """
    candidate = str(name).strip()
    if not candidate:
        raise UnsafePathError("A filename is required.")
    if len(candidate) > max_length:
        raise UnsafePathError(
            f"The filename exceeds {max_length} characters."
        )
    if _UNSAFE_NAME_CHARS.search(candidate):
        raise UnsafePathError(
            "The filename contains a character that is not permitted."
        )
    if "/" in candidate or "\\" in candidate:
        raise UnsafePathError(
            "A filename must not contain a path separator. Use the path field "
            "for a path within the permitted directory."
        )
    if candidate in (".", "..") or candidate.startswith(".."):
        raise UnsafePathError(
            "A filename must not reference a parent directory."
        )
    return candidate


def safe_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout: float = 60.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess without a shell and with a timeout.

    Two properties this guarantees, and why they matter:

    * ``shell=False`` with an argument list. A filename containing ``;`` or a
      space becomes one argument rather than a second command.
    * an explicit ``timeout``. A hung child would otherwise pin a worker
      thread indefinitely.

    Args:
        command: The argument list. Never a single string.
        cwd: Working directory.
        timeout: Seconds before the child is killed.
        env: Environment for the child. Inherits the parent when omitted.

    Returns:
        The completed process with captured text output.

    Raises:
        UnsafePathError: If ``command`` is empty or ``timeout`` is not
            positive.
        subprocess.TimeoutExpired: If the child exceeds ``timeout``.
    """
    if not command:
        raise UnsafePathError("A subprocess command must not be empty.")
    if timeout <= 0:
        raise UnsafePathError("A subprocess timeout must be positive.")
    if isinstance(command, (str, bytes)):
        raise UnsafePathError(
            "A subprocess command must be an argument list, not a string. "
            "Passing a string would require a shell and is not permitted."
        )
    return subprocess.run(  # noqa: S603 - argument list, shell=False
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
        shell=False,
        check=False,
    )


def writable_directory(path: Path | str) -> bool:
    """Return True when a directory exists and the process may write to it.

    Used by readiness: a data directory that cannot be written to is a genuine
    operational problem, and reporting ``READY`` would be a lie.
    """
    target = Path(path).expanduser()
    if not target.is_dir():
        return False
    try:
        return os.access(target, os.W_OK | os.X_OK)
    except OSError:  # pragma: no cover - defensive
        return False


def directory_free_bytes(path: Path | str) -> int | None:
    """Return free bytes on the filesystem holding ``path``, if knowable.

    Returns ``None`` rather than raising: disk space is a diagnostic, and its
    absence must not turn into a readiness failure.
    """
    try:
        return shutil.disk_usage(str(Path(path).expanduser())).free
    except (OSError, ValueError):
        return None
