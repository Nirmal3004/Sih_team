"""Alert severity classes (SIH26078, Phase 5).

.. warning::

   **These are project prototype categories only.** ``INFO``, ``WATCH``,
   ``ADVISORY``, ``WARNING``, ``SEVERE_WARNING`` and ``EXTREME`` are names
   invented for this prototype. They are **not** IMD categories, **not** WMO
   categories, and must not be presented to the public as official warnings.
   The full statement lives in :data:`SEVERITY_DISCLAIMER` and is echoed in
   every API payload that carries a severity.

Design
------
* Severity is an ordered :class:`enum.StrEnum` (like the Phase 1
  :class:`~extreme_weather.schemas.event.Severity`), so comparisons and sorting
  are type-safe and no arbitrary strings travel through the system.
* The mapping from a *Phase 1* ``Severity`` onto an *alert* severity is
  explicit in :meth:`AlertSeverity.from_event_severity` rather than implicit,
  so the two ladders can diverge without breaking anything.
* The cut points that turn a risk score into a severity live in
  :class:`SeverityThresholds` and are therefore **configurable**, not hardcoded
  in the scoring loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from extreme_weather.schemas.event import Severity as EventSeverity

__all__ = [
    "DEFAULT_SEVERITY_THRESHOLDS",
    "SEVERITY_DISCLAIMER",
    "AlertSeverity",
    "SeverityThresholds",
    "max_severity",
    "severity_for_score",
]


#: Attached to every payload that reports an alert severity.
SEVERITY_DISCLAIMER = (
    "Prototype severity categories defined by this project. They are NOT official "
    "IMD or WMO warning categories, they are not calibrated against real weather, "
    "and they must not be used as a public warning product."
)


class AlertSeverity(StrEnum):
    """Ordered prototype alert severity ladder.

    Declaration order *is* the ranking: ``INFO`` is the lowest class and
    ``EXTREME`` the highest. :attr:`rank` exposes that as an integer.
    """

    INFO = "INFO"
    WATCH = "WATCH"
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    SEVERE_WARNING = "SEVERE_WARNING"
    EXTREME = "EXTREME"

    @property
    def rank(self) -> int:
        """Numeric rank, 0 (``INFO``) to 5 (``EXTREME``)."""
        return _SEVERITY_ORDER.index(self)

    @property
    def is_actionable(self) -> bool:
        """True for severities that require an operator decision."""
        return self.rank >= AlertSeverity.WARNING.rank

    @classmethod
    def from_value(cls, value: str) -> "AlertSeverity":
        """Parse a severity from a case-insensitive string.

        Raises:
            ValueError: If the value does not name a supported severity.
        """
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown alert severity '{value}'. Supported severities: {supported}."
            ) from exc

    @classmethod
    def from_event_severity(cls, severity: EventSeverity | str) -> "AlertSeverity":
        """Map a Phase 1 detection severity onto the alert ladder.

        The two ladders are intentionally different: Phase 1 ranks a *field*,
        Phase 5 ranks a *response priority*. The mapping is explicit and
        documented so the translation can be audited.
        """
        value = (
            severity
            if isinstance(severity, EventSeverity)
            else EventSeverity.from_value(str(severity))
        )
        return _EVENT_SEVERITY_MAP[value]


#: Declaration order defines the alert severity ranking.
_SEVERITY_ORDER: tuple[AlertSeverity, ...] = (
    AlertSeverity.INFO,
    AlertSeverity.WATCH,
    AlertSeverity.ADVISORY,
    AlertSeverity.WARNING,
    AlertSeverity.SEVERE_WARNING,
    AlertSeverity.EXTREME,
)

#: Explicit Phase 1 severity -> Phase 5 alert severity mapping.
_EVENT_SEVERITY_MAP: dict[EventSeverity, AlertSeverity] = {
    EventSeverity.NORMAL: AlertSeverity.INFO,
    EventSeverity.WATCH: AlertSeverity.WATCH,
    EventSeverity.MODERATE: AlertSeverity.ADVISORY,
    EventSeverity.SEVERE: AlertSeverity.WARNING,
    EventSeverity.EXTREME: AlertSeverity.SEVERE_WARNING,
}


def max_severity(*severities: AlertSeverity) -> AlertSeverity:
    """Return the highest severity of the arguments (``INFO`` when empty)."""
    if not severities:
        return AlertSeverity.INFO
    return max(severities, key=lambda item: item.rank)


@dataclass(frozen=True)
class SeverityThresholds:
    """Configurable cut points that turn a risk score into a severity.

    A score is compared against the **lower bound** of each class from the top
    down, so the bounds must increase with severity. They default to the
    prototype ladder but nothing forces that: pass a different instance to
    :func:`severity_for_score` to retune without touching the scoring code.
    """

    warning: float = 25.0
    severe_warning: float = 45.0
    extreme: float = 70.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.warning < self.severe_warning < self.extreme <= 100.0):
            raise ValueError(
                "Severity thresholds must satisfy "
                "0 <= warning < severe_warning < extreme <= 100, got "
                f"{(self.warning, self.severe_warning, self.extreme)}."
            )

    def as_dict(self) -> dict[str, float]:
        """Return the thresholds as a JSON-serialisable dictionary."""
        return {
            "warning": self.warning,
            "severe_warning": self.severe_warning,
            "extreme": self.extreme,
        }


#: The prototype default ladder, used when a caller does not supply one.
DEFAULT_SEVERITY_THRESHOLDS = SeverityThresholds()


def severity_for_score(
    score: float,
    thresholds: SeverityThresholds = DEFAULT_SEVERITY_THRESHOLDS,
) -> AlertSeverity:
    """Map a 0-100 risk score onto the alert severity ladder.

    Args:
        score: The risk score produced by the scoring engine.
        thresholds: The cut points to apply.

    Returns:
        The matching :class:`AlertSeverity`.
    """
    value = float(score)
    if value >= thresholds.extreme:
        return AlertSeverity.EXTREME
    if value >= thresholds.severe_warning:
        return AlertSeverity.SEVERE_WARNING
    if value >= thresholds.warning:
        return AlertSeverity.WARNING
    if value >= thresholds.warning * 0.5:
        return AlertSeverity.ADVISORY
    if value > 0.0:
        return AlertSeverity.WATCH
    return AlertSeverity.INFO

