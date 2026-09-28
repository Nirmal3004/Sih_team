"""Request correlation and error standardisation (SIH26078, Phase 10).

Two cross-cutting concerns, both implemented as middleware so no individual
router has to remember them.

Request correlation
-------------------
Every response carries ``X-Request-ID``. A client that supplies one has it
honoured, so a request can be traced across a proxy; otherwise a fresh opaque id
is generated. The id is bound to a context variable, so every log line emitted
while handling the request carries it, and a ``run_id`` may be bound alongside
it for pipeline operations.

Incoming ids are **not** trusted blindly. A caller-supplied id is accepted only
if it looks like an id - a bounded, printable, non-control-character string.
Echoing an arbitrary client string back into a response header is a
header-injection vector, so the charset check is a security control, not a
formality.

Error standardisation
---------------------
Errors are returned as a consistent envelope::

    {"error": "not_found", "message": "...", "request_id": "...", "details": {...}}

Existing routers raise ``HTTPException(status_code=..., detail=...)``. This
middleware translates that into the envelope, so **no existing router has to
change** and no existing error-response assertion breaks: the status code is
preserved exactly, and the human-readable ``detail`` string stays in
``message``.

A 500 never exposes a stack trace, an exception repr, or a filesystem path
beyond what the caller already supplied. The traceback goes to the log, where an
operator can see it and a client cannot.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from extreme_weather.observability.logging import (
    bind_context,
    new_request_id,
    redact,
)
from extreme_weather.observability.metrics import get_registry

__all__ = [
    "REQUEST_ID_HEADER",
    "RUN_ID_HEADER",
    "build_error_envelope",
    "error_response",
    "install_middleware",
    "normalise_request_id",
]

#: Header carrying the request correlation id, in and out.
REQUEST_ID_HEADER = "X-Request-ID"

#: Header carrying a pipeline run id, echoed for correlation.
RUN_ID_HEADER = "X-Run-ID"

#: A caller-supplied id must look like this to be echoed. Bounded length plus a
#: printable-ASCII allow-list: a newline in a reflected header value would
#: otherwise allow header injection.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")

#: Longest request path recorded. A pathological path should not bloat a log.
_MAX_PATH_CHARS = 256

#: Why a status code was returned. Mapping is by class, not by message, so a
#: handler's wording can change without changing the reported category.
_STATUS_CATEGORY: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "unavailable",
}


def normalise_request_id(candidate: str | None) -> str | None:
    """Return a safe request id, or ``None`` to generate a fresh one.

    Args:
        candidate: The value from the incoming header, if any.

    Returns:
        The candidate when it is safe to echo, otherwise ``None``.
    """
    if not candidate:
        return None
    trimmed = candidate.strip()
    if not _SAFE_REQUEST_ID.match(trimmed):
        return None
    return trimmed


def _category(status_code: int) -> str:
    """Return a stable machine-readable category for a status code."""
    if status_code in _STATUS_CATEGORY:
        return _STATUS_CATEGORY[status_code]
    if 500 <= status_code < 600:
        return "internal_error"
    if 400 <= status_code < 500:
        return "bad_request"
    return "unknown"


def build_error_envelope(
    status_code: int,
    message: str,
    *,
    request_id: str | None = None,
    details: Any | None = None,
) -> dict[str, Any]:
    """Return the standard error body for a status code.

    Separated from :func:`error_response` so a caller that composes its own
    response can still produce an identically-shaped body.
    """
    body: dict[str, Any] = {
        "error": _category(status_code),
        "message": message,
        # `detail` is retained as an alias of `message`. FastAPI's default
        # handler used it, and Phase 0-9 clients and tests read it, so removing
        # it would be a breaking change to a published contract. The envelope
        # is additive; the status code is unchanged.
        "detail": message,
        "request_id": request_id,
    }
    if details is not None:
        body["details"] = details
    return body


def error_response(
    status_code: int,
    message: str,
    *,
    request_id: str | None = None,
    details: Any | None = None,
) -> JSONResponse:
    """Build a standardised error response.

    Args:
        status_code: The HTTP status to return, preserved exactly.
        message: A human-readable explanation, already safe to expose.
        request_id: Correlation id, echoed in the body.
        details: Optional structured context.

    Returns:
        A :class:`JSONResponse` with the error envelope and the request header.
    """
    body = build_error_envelope(
        status_code, message, request_id=request_id, details=details
    )
    headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


#: Status codes whose message a handler wrote *for a caller* and which is
#: therefore safe and useful to return. 503 is here deliberately: "no trained
#: checkpoint; run scripts/train_gnn.py" is actionable guidance, not an internal
#: detail, and hiding it would leave the caller with nothing to act on.
_CLIENT_FACING_SERVER_ERRORS: frozenset[int] = frozenset({503})


def _safe_message(exc: BaseException, status_code: int) -> str:
    """Return a message safe to show a client.

    The rule is *authorship*, not status class. A handler that raises
    ``HTTPException`` wrote its message to be read by a caller, so it is
    returned (redacted). An exception nobody handled did not, and its text may
    contain a filesystem path, a connection string or a stack frame - so a
    generic message is returned and the detail goes to the log.

    A 500 is always masked, whatever produced it: an internal failure's own
    text is exactly the case that must not reach a client.
    """
    text = getattr(exc, "detail", None)
    authored = isinstance(exc, (HTTPException, StarletteHTTPException))
    if status_code == 500 or not authored:
        return (
            "The service could not complete the request. The failure has been "
            "logged with the request id below."
        )
    if status_code in _CLIENT_FACING_SERVER_ERRORS and not isinstance(text, str):
        return "The service is unavailable for this operation."
    if isinstance(text, str) and text.strip():
        return redact(text)
    if text is not None:
        return "The request could not be processed."
    return "The request could not be processed."


def install_middleware(app: FastAPI) -> None:
    """Install correlation, metrics and error-handling middleware.

    Args:
        app: The application to instrument.

    Middleware order matters. The correlation middleware is added last so it
    wraps the others: every request, including one that fails inside another
    middleware, gets an id.
    """
    logger = logging.getLogger("extreme_weather.api")
    registry = get_registry()

    @app.middleware("http")
    async def correlate(request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        """Bind a request id, time the request, and count it."""
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = normalise_request_id(incoming) or new_request_id()
        run_id = normalise_request_id(request.headers.get(RUN_ID_HEADER))
        request.state.request_id = request_id

        started = time.perf_counter()
        with bind_context(request_id=request_id, run_id=run_id):
            try:
                response = await call_next(request)
            except Exception:
                # An exception escaping every handler is still a request, and
                # still needs a correlation id and a counted error.
                duration = time.perf_counter() - started
                registry.increment("api_requests_total", method=request.method)
                registry.increment("api_errors_total", method=request.method, status=500)
                registry.observe(
                    "api_request_duration_seconds",
                    duration,
                    method=request.method,
                )
                logger.exception(
                    "request failed before a response was produced method=%s path=%s",
                    request.method,
                    request.url.path[:_MAX_PATH_CHARS],
                )
                return error_response(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    _safe_message(RuntimeError(""), 500),
                    request_id=request_id,
                )

        duration = time.perf_counter() - started
        registry.increment("api_requests_total", method=request.method)
        registry.observe(
            "api_request_duration_seconds", duration, method=request.method
        )
        if response.status_code >= 400:
            registry.increment(
                "api_errors_total", method=request.method, status=response.status_code
            )

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path[:_MAX_PATH_CHARS],
            response.status_code,
            duration * 1000.0,
        )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Render any raised HTTPException in the standard envelope."""
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        status_code = int(exc.status_code)
        logger.warning(
            "request rejected method=%s path=%s status=%s",
            request.method,
            request.url.path[:_MAX_PATH_CHARS],
            status_code,
        )
        return error_response(
            status_code, _safe_message(exc, status_code), request_id=request_id
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Render a request-validation failure as a 422 envelope.

        The field-level detail is preserved: it is the caller's own input being
        described back to them, and it contains no server state.
        """
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        return error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The request body or parameters failed validation.",
            request_id=request_id,
            details=redact(_jsonable(exc.errors())),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Render an unexpected failure without leaking internals."""
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        logger.exception(
            "unhandled error method=%s path=%s",
            request.method,
            request.url.path[:_MAX_PATH_CHARS],
        )
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            _safe_message(exc, 500),
            request_id=request_id,
        )


def _jsonable(value: Any) -> Any:
    """Convert validation errors into JSON-serialisable structures.

    Pydantic error payloads can carry exception objects in ``ctx``, which
    ``json.dumps`` cannot handle. The offending type's name is kept instead,
    which is enough for a caller to understand the failure.
    """
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__
