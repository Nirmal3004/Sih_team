"""Configurable severity thresholds (SIH26078, Phase 1).

The detector must not carry hard-coded numbers, so every threshold lives in a
:class:`ThresholdLadder` object that can be replaced, tuned per variable and
serialised. A ladder maps a score to a severity class:

```text
score <  ladder[0].threshold           -> NORMAL
ladder[i].threshold <= score < ladder[i+1].threshold -> ladder[i].severity
score >= ladder[-1].threshold          -> ladder[-1].severity
```

Boundary rule: a score **equal** to a threshold already belongs to that
threshold's severity class (documented, deterministic, and covered by tests).

**Prototype thresholds, not official warning thresholds.** The defaults shipped in
:data:`DEFAULT_THRESHOLD_LADDERS` were chosen to demonstrate the pipeline on
synthetic data. They are *not* India Meteorological Department (IMD) or any other
official warning criteria, they are not calibrated, and they must not be used for
real warnings. Users are expected to supply their own ladders through
configuration, environment variables or code.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from extreme_weather.schemas.event import EventType, Severity

__all__ = [
    "DEFAULT_THRESHOLD_LADDERS",
    "EFI_THRESHOLD_LADDERS",
    "PROTOTYPE_THRESHOLD_WARNING",
    "STANDARDIZED_THRESHOLD_LADDERS",
    "ThresholdLadder",
    "ThresholdSet",
    "ThresholdsError",
    "classify_score",
    "classify_scores",
    "default_thresholds",
]

#: Explicit disclaimer attached to every ladder built by this module.
PROTOTYPE_THRESHOLD_WARNING = (
    "Prototype severity thresholds for the SIH26078 demo. They are NOT official warning "
    "thresholds (for example IMD criteria) and are not calibrated for real weather data."
)


class ThresholdsError(ValueError):
    """Raised when a threshold ladder is invalid."""


class ThresholdLadder(BaseModel):
    """Score -> severity mapping for one variable/event-type combination.

    Attributes:
        name: Identifier of the ladder, for example ``"precipitation_efi"``.
        event_type: Event type the ladder classifies.
        variable: Source variable the ladder is tuned for.
        score_name: Score the thresholds are expressed on (``efi``, ``standardized``...).
        steps: Ascending ``(threshold, severity)`` pairs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    event_type: EventType
    variable: str = Field(min_length=1)
    score_name: str = Field(default="efi", min_length=1)
    steps: tuple[tuple[float, Severity], ...]
    note: str = PROTOTYPE_THRESHOLD_WARNING

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, value: object) -> list[tuple[float, Severity]]:
        """Accept ``[(0.5, "WATCH"), ...]`` or ``{0.5: "WATCH"}`` inputs."""
        if isinstance(value, Mapping):
            entries: list[object] = [(key, severity) for key, severity in value.items()]
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            entries = list(value)
        else:
            raise ThresholdsError(
                "steps must be a sequence or mapping of (threshold, severity)."
            )

        steps: list[tuple[float, Severity]] = []
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ThresholdsError(
                    f"Invalid threshold step {entry!r}; expected (value, severity)."
                )
            threshold, severity = entry
            steps.append(
                (
                    float(threshold),
                    severity if isinstance(severity, Severity) else Severity.from_value(str(severity)),
                )
            )
        return steps

    @model_validator(mode="after")
    def _validate_ladder(self) -> "ThresholdLadder":
        """Ensure the ladder is non-empty, strictly ascending and never NORMAL."""
        if not self.steps:
            raise ThresholdsError(f"Threshold ladder '{self.name}' has no steps.")
        thresholds = [step[0] for step in self.steps]
        if any(threshold > upper for threshold, upper in zip(thresholds, thresholds[1:])):
            raise ThresholdsError(
                f"Thresholds of ladder '{self.name}' must be strictly increasing; got {thresholds}."
            )
        for threshold, severity in self.steps:
            if severity is Severity.NORMAL:
                raise ThresholdsError(
                    f"Ladder '{self.name}' must not contain NORMAL; scores below the first "
                    "threshold are NORMAL by definition."
                )
        if self.steps[-1][1] is not Severity.EXTREME:
            raise ThresholdsError(
                f"The highest threshold of ladder '{self.name}' must be EXTREME; got "
                f"{self.steps[-1][1].value}."
            )
        return self

    @property
    def thresholds(self) -> tuple[float, ...]:
        """The ascending threshold values of the ladder."""
        return tuple(step[0] for step in self.steps)

    def classify(self, score: float) -> Severity:
        """Classify one score.

        Steps are ascending, so the last matching step wins: a score equal to a
        threshold belongs to that threshold's severity.
        """
        if np.isnan(score):
            return Severity.NORMAL
        severity = Severity.NORMAL
        for threshold, candidate in self.steps:
            if score >= threshold:
                severity = candidate
        return severity


# --------------------------------------------------------------------------- #
# Prototype defaults
# --------------------------------------------------------------------------- #
#: Prototype ladders used when the caller does not supply its own.
#:
#: The defaults are expressed on the **absolute anomaly** scale
#: (``value - reference_mean``, in the units of the variable), because that is
#: the only score in this prototype that keeps useful headroom on a short
#: synthetic reference: the EFI-style score saturates at +1.28 once a value
#: passes the highest stored reference quantile, and a z-score is unstable for
#: precipitation whose reference distribution has a large mass at zero. Both
#: alternative scales ship as :data:`EFI_THRESHOLD_LADDERS` and
#: :data:`STANDARDIZED_THRESHOLD_LADDERS`.
DEFAULT_THRESHOLD_LADDERS: tuple[ThresholdLadder, ...] = (
    ThresholdLadder(
        name="precipitation_absolute",
        event_type=EventType.EXTREME_RAINFALL,
        variable="precipitation",
        score_name="absolute",
        steps=(
            (10.0, Severity.WATCH),
            (25.0, Severity.MODERATE),
            (40.0, Severity.SEVERE),
            (50.0, Severity.EXTREME),
        ),
    ),
    ThresholdLadder(
        name="wind_speed_absolute",
        event_type=EventType.EXTREME_WIND,
        variable="wind_speed",
        score_name="absolute",
        steps=(
            (5.0, Severity.WATCH),
            (10.0, Severity.MODERATE),
            (15.0, Severity.SEVERE),
            (20.0, Severity.EXTREME),
        ),
    ),
    ThresholdLadder(
        name="temperature_absolute",
        event_type=EventType.EXTREME_TEMPERATURE,
        variable="temperature",
        score_name="absolute",
        steps=(
            (2.0, Severity.WATCH),
            (4.0, Severity.MODERATE),
            (6.0, Severity.SEVERE),
            (8.0, Severity.EXTREME),
        ),
    ),
)

#: Prototype ladders on the EFI-style scale (see
#: :data:`extreme_weather.anomaly.scores.EFI_FORMULA`). Because that score
#: cannot exceed ``Phi^-1(1) ~= 1.28`` with a bounded percentile, the whole
#: ladder lives inside ``[0, 1.28]``.
EFI_THRESHOLD_LADDERS: tuple[ThresholdLadder, ...] = (
    ThresholdLadder(
        name="precipitation_efi",
        event_type=EventType.EXTREME_RAINFALL,
        variable="precipitation",
        score_name="efi",
        steps=((0.26, Severity.WATCH), (0.52, Severity.MODERATE), (1.02, Severity.SEVERE), (1.28, Severity.EXTREME)),
    ),
    ThresholdLadder(
        name="wind_speed_efi",
        event_type=EventType.EXTREME_WIND,
        variable="wind_speed",
        score_name="efi",
        steps=((0.26, Severity.WATCH), (0.52, Severity.MODERATE), (1.02, Severity.SEVERE), (1.28, Severity.EXTREME)),
    ),
    ThresholdLadder(
        name="temperature_efi",
        event_type=EventType.EXTREME_TEMPERATURE,
        variable="temperature",
        score_name="efi",
        steps=((0.26, Severity.WATCH), (0.52, Severity.MODERATE), (1.02, Severity.SEVERE), (1.28, Severity.EXTREME)),
    ),
)

#: Prototype ladders on the standardised (z) scale.
STANDARDIZED_THRESHOLD_LADDERS: tuple[ThresholdLadder, ...] = tuple(
    ThresholdLadder(
        name=ladder.name.replace("_absolute", "_standardized"),
        event_type=ladder.event_type,
        variable=ladder.variable,
        score_name="standardized",
        steps=((2.0, Severity.WATCH), (3.0, Severity.MODERATE), (4.0, Severity.SEVERE), (5.0, Severity.EXTREME)),
    )
    for ladder in DEFAULT_THRESHOLD_LADDERS
)


def classify_score(score: float, ladder: ThresholdLadder) -> Severity:
    """Classify a single score with ``ladder`` (NaN becomes ``NORMAL``)."""
    return ladder.classify(float(score))


def classify_scores(
    scores: xr.DataArray, ladder: ThresholdLadder
) -> xr.DataArray:
    """Classify a whole score field, preserving coordinates and dimensions.

    Uses ``np.searchsorted`` on the ascending ladder thresholds plus fancy
    indexing, so the cost is independent of the number of grid cells and no
    Python-level loop over cells is needed. NaN scores map to ``NORMAL``.

    Args:
        scores: Score field (any dimensions, typically time/latitude/longitude).
        ladder: Threshold ladder to apply.

    Returns:
        An object-dtype DataArray of :class:`~extreme_weather.schemas.event.Severity`
        values with the same dims/coords.
    """
    thresholds = np.asarray(ladder.thresholds, dtype="float64")
    lookup = np.asarray([Severity.NORMAL] + [severity for _, severity in ladder.steps], dtype=object)
    values = np.asarray(scores.values, dtype="float64")

    # index k = number of thresholds <= score; lookup[0] is NORMAL.
    index = np.searchsorted(thresholds, values, side="right")
    index = np.clip(index, 0, lookup.size - 1)
    classified = np.where(np.isnan(values), Severity.NORMAL, lookup[index])

    result = xr.DataArray(
        classified, dims=scores.dims, coords=scores.coords, name="severity"
    )
    result.attrs = {
        "long_name": "detected severity class",
        "ladder": ladder.name,
        "score_name": ladder.score_name,
        "comment": ladder.note,
    }
    return result


class ThresholdSet(BaseModel):
    """A named collection of ladders, one per variable/event type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ladders: tuple[ThresholdLadder, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique(self) -> "ThresholdSet":
        names = [ladder.name for ladder in self.ladders]
        if len(names) != len(set(names)):
            raise ThresholdsError(f"Duplicate threshold ladder names in {names}.")
        return self

    def for_event_type(self, event_type: EventType | str) -> ThresholdLadder:
        """Return the ladder registered for ``event_type``.

        Raises:
            ThresholdsError: If no ladder matches.
        """
        wanted = (
            event_type if isinstance(event_type, EventType) else EventType.from_value(str(event_type))
        )
        for ladder in self.ladders:
            if ladder.event_type is wanted:
                return ladder
        available = ", ".join(sorted({ladder.event_type.value for ladder in self.ladders}))
        raise ThresholdsError(f"No threshold ladder for event type {wanted.value}. Have: {available}.")

    def for_variable(self, variable: str) -> ThresholdLadder | None:
        """Return the ladder tuned for ``variable``, or ``None``."""
        for ladder in self.ladders:
            if ladder.variable == variable:
                return ladder
        return None

    @classmethod
    def for_event_types(cls, event_types: Iterable[EventType | str]) -> "ThresholdSet":
        """Build a set from the default ladders of the given event types."""
        wanted = {
            event_type if isinstance(event_type, EventType) else EventType.from_value(str(event_type))
            for event_type in event_types
        }
        ladders = [ladder for ladder in DEFAULT_THRESHOLD_LADDERS if ladder.event_type in wanted]
        if not ladders:
            raise ThresholdsError(
                f"No prototype ladder available for {sorted(item.value for item in wanted)}."
            )
        return cls(ladders=tuple(ladders))


def default_thresholds(
    event_types: Iterable[EventType | str] | None = None,
) -> ThresholdSet:
    """Return the prototype threshold ladders.

    Args:
        event_types: Restrict the result to these event types. ``None`` returns
            all prototype ladders.

    Returns:
        A :class:`ThresholdSet`; callers may replace or extend it freely.
    """
    if event_types is None:
        return ThresholdSet(ladders=DEFAULT_THRESHOLD_LADDERS)
    return ThresholdSet.for_event_types(event_types)
