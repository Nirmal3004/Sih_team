"""Structured application logging (SIH26078, Phase 10).

A log line is only useful if it can be *correlated*: which request, which run,
how long, and what happened. This module makes those fields first-class instead
of leaving them to whoever remembers to include them.

Format
------
``text`` (default) - readable for a developer::

    2026-01-01 12:00:00 INFO  [api] request completed method=GET path=/api/v1/events status=200 duration_ms=12.4 request_id=req-8f3a...

``json`` - one object per line, for a log shipper::

    {"ts": "...", "level": "INFO", "component": "api", "event": "request.completed",
     "method": "GET", "status": 200, "duration_ms": 12.4, "request_id": "req-8f3a..."}

Both carry the same fields. JSON is opt-in via ``LOG_FORMAT=json`` so local
development keeps its readable default.

Secrets
-------
Every message and every structured field passes through :func:`redact` before
emission. This is a **backstop**, not the primary defence - the providers and the
URL builders already avoid credentials - but a third-party exception string can
contain anything, and a log file is the least protected artefact in a system.

Correlation
-----------
:func:`bind_context` attaches ``request_id`` / ``run_id`` to every record emitted
inside a ``with`` block, using a :class:`contextvars.ContextVar` so concurrent
requests cannot see each other's identifiers.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

__all__ = [
    "COMPONENT_ROOT",
    "JsonFormatter",
    "bind_context",
    "configure_logging",
    "get_correlation_ids",
    "get_logger",
    "log_event",
    "redact",
    "request_context",
    "run_context",
]

UTC = timezone.utc

#: Root component name for this application's records.
COMPONENT_ROOT = "extreme_weather"

#: Ambient correlation identifiers, propagated by the API middleware.
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_component: ContextVar[str | None] = ContextVar("log_component", default=None)

#: Anything matching these is masked before a record is written. Patterns are
#: deliberately broad: over-masking a log line is a nuisance, under-masking one
#: leaks a credential into a file that outlives the process.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|access[_-]?key|secret[_-]?key|token|"
        r"access[_-]?token|refresh[_-]?token|password|passwd|secret)\b"
        r"(\s*[:=]\s*|\"\s*:\s*\")([^\s,;\"'}\)]+)"
    ),
    re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._\-]{8,})"),
    re.compile(r"(?i)([?&](?:api[_-]?key|apikey|key|token|access_token|signature|sig)=)([^&\s]+)"),
    re.compile(r"(?i)\b(authorization)(\s*[:=]\s*)([^\s,;\"'}\)]+)"),
)

#: Keys whose values are always masked, regardless of how they are written.
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "access_key",
        "secret_key",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "passwd",
        "secret",
        "authorization",
        "credential",
        "credentials",
    }
)

#: Longest value retained in a log line before truncation. An unbounded value in
#: a log field is how a multi-megabyte dataset name ends up in a log file.
_MAX_VALUE_CHARS = 512

REDACTED = "***"


def _mask_kv(match: re.Match[str]) -> str:
    """Mask the value of a ``key = value`` or ``key: value`` match."""
    head = match.group(0)[: match.start(3) - match.start(0)]
    return f"{head}{REDACTED}"


def _mask_bearer(match: re.Match[str]) -> str:
    """Mask a ``Bearer <token>`` value."""
    return f"{match.group(1)} {REDACTED}"


def _mask_query(match: re.Match[str]) -> str:
    """Mask a credential-shaped query parameter, keeping its name."""
    return f"{match.group(1)}{REDACTED}"


#: Applied in order; each entry is (compiled pattern, replacement).
_REDACTIONS: tuple[tuple[re.Pattern[str], Any], ...] = (
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|access[_-]?key|secret[_-]?key|token|"
            r"access[_-]?token|refresh[_-]?token|password|passwd|secret|"
            r"authorization|credential|proxy-authorization)\b"
            r"(\s*[:=]\s*|\"\s*:\s*\")([^\n,;\"}\)]{1,4096})"
        ),
        _mask_kv,
    ),
    # Bearer first: otherwise the kv rule below matches "Authorization: Bearer"
    # and stops at the space, leaving the token itself in the clear.
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}"), _mask_bearer),
    (
        re.compile(
            r"(?i)([?&](?:api[_-]?key|apikey|key|token|access_token|"
            r"signature|sig)=)[^&\s]+"
        ),
        _mask_query,
    ),
)


def redact(text: Any) -> str:
    """Mask credential-shaped content in a string.

    Args:
        text: The value to scrub. Non-strings are coerced rather than
            raising, because this runs on the logging path where a type
            error would turn a log line into a crash.

    Returns:
        The scrubbed string, truncated to a sane length.
    """
    if not isinstance(text, str):
        text = str(text)
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    if len(text) > _MAX_VALUE_CHARS:
        text = text[:_MAX_VALUE_CHARS] + "...[truncated]"
    return text

def _redact_value(value: Any) -> Any:
    """Recursively redact a structured value.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        The value with sensitive keys masked and long strings truncated.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED
                if str(key).strip().lower().replace("-", "_") in _SENSITIVE_KEYS
                else _redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(value)


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the JSON representation of ``record``."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "component": getattr(record, "component", None) or record.name,
            "event": getattr(record, "event", None) or record.getMessage(),
            "message": redact(record.getMessage()),
        }
        # Redact the whole mapping at once so the key-aware branch runs: a field
        # named "api_key" must be masked whatever its value happens to look like.
        fields = getattr(record, "fields", {}) or {}
        payload.update(_redact_value(dict(fields)))

        request_id, run_id = get_correlation_ids()
        if request_id and "request_id" not in payload:
            payload["request_id"] = request_id
        if run_id and "run_id" not in payload:
            payload["run_id"] = run_id
        if record.exc_info:
            # The exception *type and message* are kept; the traceback is not
            # written to every log line, because the handler that owns the
            # traceback will emit it once at ERROR.
            exc_type, exc_value, _ = record.exc_info
            payload["error_type"] = getattr(exc_type, "__name__", str(exc_type))
            payload["error"] = redact(str(exc_value))
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Render a record as a readable single line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the human-readable representation of ``record``."""
        component = getattr(record, "component", None) or record.name
        request_id, run_id = get_correlation_ids()
        parts = [
            f"{datetime.fromtimestamp(record.created, UTC).strftime('%Y-%m-%d %H:%M:%S')}",
            f"{record.levelname:<7}",
            f"[{component}]",
            redact(record.getMessage()),
        ]
        extras = []
        for key, value in getattr(record, "fields", {}).items():
            extras.append(f"{key}={_redact_value(value)}")
        if request_id:
            extras.append(f"request_id={request_id}")
        if run_id:
            extras.append(f"run_id={run_id}")
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            extras.append(f"error={getattr(exc_type, '__name__', exc_type)}: {redact(str(exc_value))}")
        if extras:
            parts.append(" ".join(extras))
        return " ".join(parts)


def configure_logging(
    level: str = "INFO",
    *,
    fmt: str | None = None,
    stream: Any | None = None,
    force: bool = False,
) -> logging.Logger:
    """Install the application log handler.

    Idempotent: calling it again replaces the handler rather than stacking a
    second one, which would double every line.

    Args:
        level: Minimum level to emit, e.g. ``INFO`` or ``DEBUG``.
        fmt: ``text`` or ``json``. Defaults to the ``LOG_FORMAT`` environment
            variable, then ``text``.
        stream: Destination stream; defaults to ``stdout``.
        force: Replace existing handlers even if the logger is already
            configured.

    Returns:
        The configured application logger.
    """
    resolved = (fmt or _env_format()).strip().lower()
    formatter: logging.Formatter = (
        JsonFormatter() if resolved == "json" else TextFormatter()
    )
    logger = logging.getLogger(COMPONENT_ROOT)

    if logger.handlers and not force:
        logger.setLevel(_level_number(level))
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(_level_number(level))
    # Records are handled here; propagating would double them in a host app.
    logger.propagate = False
    return logger


def _env_format() -> str:
    """Return the configured log format, defaulting to text."""
    import os

    return os.environ.get("LOG_FORMAT", "text")


def _level_number(level: str | int) -> int:
    """Convert a level name to its numeric value, tolerating bad input."""
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).strip().upper(), logging.INFO)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger for a component.

    Args:
        name: Component name, e.g. ``api`` or ``pipeline``. Omit for the
            application root.

    Returns:
        A child of the application logger.
    """
    if not name:
        return logging.getLogger(COMPONENT_ROOT)
    return logging.getLogger(f"{COMPONENT_ROOT}.{name}")


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str = "",
    **fields: Any,
) -> None:
    """Emit a structured event.

    Args:
        logger: The logger to emit on.
        level: A :mod:`logging` level.
        event: A stable, machine-greppable event name, e.g.
            ``pipeline.run.completed``. Prefer dots over prose.
        message: Human-readable summary. Redacted before emission.
        **fields: Structured key/value pairs. All values are redacted.
    """
    logger.log(
        level,
        message or event,
        extra={"event": event, "fields": dict(fields)},
    )


def get_correlation_ids() -> tuple[str | None, str | None]:
    """Return the ambient ``(request_id, run_id)``."""
    return _request_id.get(), _run_id.get()


@contextmanager
def bind_context(
    *, request_id: str | None = None, run_id: str | None = None
) -> Iterator[tuple[str | None, str | None]]:
    """Bind correlation identifiers for the duration of a block.

    Uses context variables, so two concurrent requests cannot observe each
    other's identifiers.

    Args:
        request_id: Request identifier to bind.
        run_id: Pipeline run identifier to bind.

    Yields:
        The bound ``(request_id, run_id)`` pair.
    """
    tokens = []
    if request_id is not None:
        tokens.append((_request_id, _request_id.set(request_id)))
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    try:
        yield _request_id.get(), _run_id.get()
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request identifier, generating one when absent.

    Args:
        request_id: An existing identifier, or ``None`` to generate a fresh one.

    Yields:
        The effective request id.
    """
    effective = request_id or new_request_id()
    with bind_context(request_id=effective):
        yield effective


@contextmanager
def run_context(run_id: str) -> Iterator[str]:
    """Bind a pipeline run identifier.

    Args:
        run_id: The run identifier.

    Yields:
        The bound run id.
    """
    with bind_context(run_id=run_id):
        yield run_id


def new_request_id() -> str:
    """Return a fresh, opaque request identifier.

    Uses ``uuid4`` rather than a counter: a counter is guessable, and a
    request id that appears in a header should not be enumerable.
    """
    return f"req-{uuid.uuid4().hex[:16]}"
