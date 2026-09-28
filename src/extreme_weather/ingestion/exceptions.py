"""Ingestion exceptions (SIH26078, Phase 6).

Every failure mode in the ingestion layer gets its own exception type so the API
can map it to the right status code and the CLI can print an actionable
message. Nothing here is a bare ``RuntimeError``: a caller must be able to tell
"the file is corrupt" from "this provider needs credentials" from "this format
needs an optional dependency you have not installed".
"""

from __future__ import annotations

__all__ = [
    "DataSourceUnavailableError",
    "FormatNotSupportedError",
    "IngestionError",
    "MissingDependencyError",
    "NormalizationError",
    "ProviderCredentialsMissingError",
    "QualityCheckError",
    "SourceNotFoundError",
    "UnsupportedFormatError",
]


class IngestionError(RuntimeError):
    """Base class for every ingestion failure (maps to HTTP 422)."""


class SourceNotFoundError(IngestionError):
    """The requested file, dataset id or provider does not exist (HTTP 404)."""


class FormatNotSupportedError(IngestionError):
    """The file extension is not one this project can read (HTTP 422)."""


class UnsupportedFormatError(FormatNotSupportedError):
    """Alias kept for readability at call sites that read "unsupported"."""


class NormalizationError(IngestionError):
    """A dataset could not be normalized into the standard model.

    Raised rather than silently coercing, because a wrong unit or a lost
    coordinate is a data-integrity failure, not a formatting nuisance.
    """


class QualityCheckError(IngestionError):
    """A dataset failed a blocking quality check and must not be processed."""


class MissingDependencyError(IngestionError):
    """An optional dependency is required for this format but is not installed.

    Raised instead of silently degrading, so a GRIB file never produces an empty
    dataset that looks like "no events found".
    """


class DataSourceUnavailableError(IngestionError):
    """A remote data source could not be reached (HTTP 503).

    Network timeouts and DNS failures land here; they are transient conditions,
    not invalid requests.
    """


class ProviderCredentialsMissingError(IngestionError):
    """A provider needs credentials that are not configured (HTTP 503).

    The message must name the *environment variable* to set, never the value,
    and no credential is ever echoed, logged or returned through the API.
    """
