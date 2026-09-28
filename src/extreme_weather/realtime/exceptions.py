"""Exceptions for the near-real-time layer (SIH26078, Phase 8).

Every failure gets a distinct type so the API can return the right status code
and the CLI can say something actionable:

* ``ProviderUnavailableError``  -> 503, the dependency or network is down
* ``ProviderTimeoutError``      -> 504, the provider did not answer in time
* ``ProviderResponseError``     -> 502, the provider answered with something unusable
* ``QualityGateRejection``      -> 422, the data is structurally unusable

The split matters operationally: a 503 means "retry later", while a 422 means
"retrying will not help, the data itself is wrong".
"""

from __future__ import annotations

__all__ = [
    "PipelineConfigurationError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QualityGateRejection",
    "RealtimeError",
]


class RealtimeError(RuntimeError):
    """Base class for every near-real-time failure (maps to HTTP 422)."""


class PipelineConfigurationError(RealtimeError):
    """Raised when a pipeline configuration is internally inconsistent."""


class ProviderError(RealtimeError):
    """Base class for weather-provider failures (maps to HTTP 503)."""


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached, or a dependency it needs is absent.

    Transient by nature: a retry may succeed, which is why the retry policy
    treats this differently from a response error.
    """

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable description for a run record."""
        return {
            "error": "provider_unavailable",
            "message": str(self),
            "attempts": self.attempts,
        }


class ProviderTimeoutError(ProviderUnavailableError):
    """The provider did not answer within the configured timeout."""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable description for a run record."""
        return {
            "error": "provider_timeout",
            "message": str(self),
            "attempts": self.attempts,
        }


class ProviderResponseError(ProviderError):
    """The provider answered, but the response is unusable.

    Not transient: retrying an identical request will produce an identical
    bad response, so the retry policy does not repeat it.
    """

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable description for a run record."""
        return {"error": "provider_response", "message": str(self)}


class QualityGateRejection(RealtimeError):
    """The dataset failed a blocking quality check and must not be processed.

    Carries the full verdict so a caller can show *why* the data was refused
    rather than a bare failure.
    """

    def __init__(self, message: str, verdict: object | None = None) -> None:
        super().__init__(message)
        self.verdict = verdict

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable description for a run record."""
        payload: dict[str, object] = {
            "error": "quality_gate_rejected",
            "message": str(self),
        }
        if self.verdict is not None and hasattr(self.verdict, "as_dict"):
            payload["verdict"] = self.verdict.as_dict()  # type: ignore[attr-defined]
        return payload
