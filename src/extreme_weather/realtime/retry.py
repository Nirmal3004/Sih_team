"""Retry with exponential backoff (SIH26078, Phase 8, section 9).

Ingestion is the only stage that talks to the network, so it is the only stage
that retries. The important decision here is **which failures are worth
retrying**:

* ``ProviderTimeoutError`` and ``ProviderUnavailableError`` are transient. The
  server may be briefly overloaded, the network may blip. Retry.
* ``ProviderResponseError`` is not. A malformed payload will be malformed again.
  Retrying just delays the error and hammers a service that is already
  misbehaving. Do not retry.

A ``sleep`` callable is injected so tests can exercise backoff without spending
real seconds, and every attempt is recorded so a run can report "it took three
tries", which is the difference between a flaky network and a broken provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from extreme_weather.realtime.exceptions import (
    ProviderError,
    ProviderResponseError,
    ProviderUnavailableError,
)

__all__ = ["Attempt", "RetryPolicy", "retry_call"]

T = TypeVar("T")


@dataclass(frozen=True)
class Attempt:
    """One recorded attempt at an operation.

    Attributes:
        number: 1-based attempt index.
        succeeded: Whether it worked.
        error: The failure message, when it did not.
        delay: Seconds waited *after* this attempt, before the next one.
    """

    number: int
    succeeded: bool
    error: str | None = None
    delay: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the attempt."""
        return {
            "attempt": self.number,
            "succeeded": self.succeeded,
            "error": self.error,
            "delay_seconds": self.delay,
        }


@dataclass
class RetryPolicy:
    """How many times to retry, and how long to wait between attempts.

    Args:
        max_attempts: Total attempts including the first. ``3`` means up to two
            retries.
        base_delay: The first backoff delay in seconds.
        backoff_factor: Multiplier applied to the delay after each failure.
        max_delay: Upper bound on any single delay, so a long-running run cannot
            stall indefinitely.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0

    def __post_init__(self) -> None:
        if int(self.max_attempts) < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.base_delay < 0:
            raise ValueError("base_delay cannot be negative.")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0.")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay.")

    def delay_for(self, attempt: int) -> float:
        """Return the delay before attempt ``attempt + 1``.

        Args:
            attempt: The 1-based number of the attempt that just failed.

        Returns:
            The number of seconds to wait, capped at ``max_delay``.
        """
        if attempt < 1:
            return 0.0
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        """Return True when this error is worth another attempt.

        Args:
            error: The raised exception.
            attempt: The 1-based number of the attempt that just failed.

        Returns:
            True when attempts remain and the failure is transient.
        """
        if attempt >= self.max_attempts:
            return False
        return not isinstance(error, ProviderResponseError)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the policy."""
        return {
            "max_attempts": self.max_attempts,
            "base_delay": self.base_delay,
            "backoff_factor": self.backoff_factor,
            "max_delay": self.max_delay,
        }


def retry_call(
    operation: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> tuple[T, list[Attempt]]:
    """Call ``operation``, retrying transient failures with backoff.

    Args:
        operation: A zero-argument callable to attempt.
        policy: The retry policy; defaults to three attempts with exponential
            backoff.
        sleep: Injected sleep function, so tests need not wait in real time.
        on_attempt: Optional callback invoked with each :class:`Attempt`,
            including the successful one, so a caller can log the history.

    Returns:
        A tuple of the operation's result and the full list of attempts.

    Raises:
        The last exception raised by ``operation`` once retries are exhausted or
        the failure is not retryable. A partially-successful run is therefore
        never reported as complete: either the call returns a real result or
        the exception propagates.
    """
    active = policy or RetryPolicy()
    attempts: list[Attempt] = []
    last_error: BaseException | None = None

    for number in range(1, int(active.max_attempts) + 1):
        try:
            result = operation()
        except ProviderError as exc:
            last_error = exc
            delay = active.delay_for(number) if active.should_retry(exc, number) else 0.0
            attempt = Attempt(
                number=number, succeeded=False, error=str(exc), delay=delay
            )
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)
            if not active.should_retry(exc, number):
                raise
            if delay:
                sleep(delay)
            continue
        attempt = Attempt(number=number, succeeded=True)
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)
        return result, attempts

    # Unreachable: the loop either returns or raises on the final attempt.
    raise last_error if last_error else ProviderUnavailableError("Retry failed.")
