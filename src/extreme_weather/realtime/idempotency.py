"""Deterministic run identity (SIH26078, Phase 8, section 13).

A scheduled job will, sooner or later, run twice over the same data. Without a
stable identity that produces duplicate anomalies, duplicate events and
duplicate alerts, and the dashboard slowly fills with copies of one storm.

The identity is a hash over the things that define "the same piece of weather
arriving from the same place":

    provider | time window | newest data timestamp | content checksum

Two design decisions are worth stating explicitly:

* **The checksum is of the data, not the request.** Asking for the same window
  twice must produce the same identity even if the provider happened to
  re-serve it differently, and asking for a window that now contains a new hour
  must *not* collide with the previous one.
* **The identity is the run id.** A rerun with the same inputs gets the same
  ``run_id``, so the second run is recognised and short-circuited rather than
  appended. Nothing is silently discarded: the rerun is recorded, and it records
  *why* it was skipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

__all__ = ["RunIdentity", "compute_run_identity"]


@dataclass(frozen=True)
class RunIdentity:
    """The deterministic fingerprint of one ingestion.

    Attributes:
        run_id: Short, stable identifier derived from the fingerprint.
        fingerprint: The full SHA-256 hex digest.
        provider: Provider name.
        window: The requested time window as a string.
        source_timestamp: The newest timestamp in the data.
        checksum: Checksum of the fetched values.
    """

    run_id: str
    fingerprint: str
    provider: str
    window: str
    source_timestamp: str | None
    checksum: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the identity."""
        return {
            "run_id": self.run_id,
            "fingerprint": self.fingerprint,
            "provider": self.provider,
            "window": self.window,
            "source_timestamp": self.source_timestamp,
            "checksum": self.checksum,
        }

    def matches(self, other: "RunIdentity | dict[str, Any] | None") -> bool:
        """Return True when ``other`` describes the same ingestion.

        Args:
            other: Another identity, or its dictionary form, or ``None``.
        """
        if other is None:
            return False
        candidate = other.fingerprint if isinstance(other, RunIdentity) else other.get("fingerprint")
        return bool(candidate) and candidate == self.fingerprint


def compute_run_identity(
    provider: str,
    *,
    window: str,
    source_timestamp: str | None = None,
    checksum: str = "",
) -> RunIdentity:
    """Compute the deterministic identity of an ingestion.

    Args:
        provider: Provider name.
        window: A string describing the requested time window. Callers should
            use an exact representation (``start``/``end`` ISO strings) so that
            a shifted window does not collide.
        source_timestamp: The newest timestamp in the fetched data.
        checksum: Checksum of the fetched values, when available.

    Returns:
        A :class:`RunIdentity`. The same inputs always produce the same
        ``run_id``, which is what makes reruns recognisable.
    """
    payload = {
        "provider": str(provider),
        "window": str(window),
        "source_timestamp": str(source_timestamp) if source_timestamp else None,
        "checksum": str(checksum) if checksum else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return RunIdentity(
        run_id=f"run-{fingerprint[:16]}",
        fingerprint=fingerprint,
        provider=payload["provider"],
        window=payload["window"],
        source_timestamp=payload["source_timestamp"],
        checksum=payload["checksum"] or "",
    )
