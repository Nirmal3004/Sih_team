"""Event-to-event matching (SIH26078, Phase 2).

Consecutive detections of the same storm must be joined into one track. This
module answers a single question: *given the events at step ``t`` and the events
at step ``t+1``, which detections belong to the same physical event?*

Matching cost
-------------
For a previous event ``a`` (time ``t``) and a candidate event ``b`` (time
``t' > t``) of the **same event type**:

```text
distance_km   = haversine(centroid_a, centroid_b)
gap_hours     = t' - t
speed_kmh     = distance_km / gap_hours

cost = distance_weight * (distance_km / max_distance_km)
     + intensity_weight * intensity_mismatch(a, b)

intensity_mismatch(a, b) = 1 - min(I_a, I_b) / max(I_a, I_b)      [0, 1]
                          = 0 when either intensity is <= 0
```

A pair is **rejected** (cost = infinity, never matched) when any of these holds:

1. the event types differ;
2. ``gap_hours`` is <= 0 or larger than ``max_time_gap_hours``;
3. ``distance_km`` is larger than ``max_distance_km``;
4. ``speed_kmh`` is larger than ``max_speed_kmh`` (a physically impossible jump
   for that time step).

Assignment
----------
The accepted pairs are assigned **greedily in order of increasing cost**; ties are
broken deterministically by ``(cost, previous_event_id, next_event_id)``, so the
result never depends on input order, hashing or randomness. The assignment is
strictly **one-to-one**: a detection at ``t+1`` can extend at most one track, and
a track can be extended by at most one detection per step. Greedy matching is not
a global optimum in general, but it is reproducible, O(n log n) and adequate for
a prototype - stated here so the trade-off is explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.schemas.event import ExtremeWeatherEvent
from extreme_weather.tracking.metrics import haversine_distance_km, movement_speed_kmh

__all__ = [
    "MatchResult",
    "MatchingConfig",
    "TrackingError",
    "intensity_mismatch",
    "match_cost",
    "match_events",
]

INFINITE_COST: float = math.inf


class TrackingError(ValueError):
    """Raised when tracking inputs are inconsistent."""


class MatchingConfig(BaseModel):
    """Tunable gates and weights of the matching step.

    Attributes:
        max_distance_km: Reject pairs further apart than this.
        max_speed_kmh: Reject pairs whose implied movement is faster than this.
        max_time_gap_hours: Reject pairs separated by more time than this.
        distance_weight: Weight of the normalised distance term (default 0.7).
        intensity_weight: Weight of the intensity-similarity term (default 0.3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_distance_km: float = Field(default=250.0, gt=0.0)
    max_speed_kmh: float = Field(default=120.0, gt=0.0)
    max_time_gap_hours: float = Field(default=6.0, gt=0.0)
    distance_weight: float = Field(default=0.7, ge=0.0)
    intensity_weight: float = Field(default=0.3, ge=0.0)

    @property
    def weight_sum(self) -> float:
        """Sum of both weights (the cost is divided by it when > 0)."""
        return self.distance_weight + self.intensity_weight


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching two consecutive detection steps."""

    #: previous event id -> next event id (accepted, one-to-one pairs only).
    pairs: dict[str, str] = field(default_factory=dict)
    #: previous events that were not continued.
    unmatched_previous: tuple[str, ...] = ()
    #: next-step events that did not continue any track.
    unmatched_next: tuple[str, ...] = ()
    #: number of candidate pairs rejected by the gates (diagnostic only).
    rejected_pairs: int = 0

    @property
    def match_count(self) -> int:
        """Number of accepted pairs."""
        return len(self.pairs)

    def continuation_of(self, previous_event_id: str) -> str | None:
        """Return the next-step event continuing ``previous_event_id``."""
        return self.pairs.get(previous_event_id)


def intensity_mismatch(first: float, second: float) -> float:
    """Return ``1 - min/max`` of two intensities, clamped to ``[0, 1]``.

    A non-positive or non-finite intensity gives ``0.0`` (no evidence against
    the match), which keeps the term well behaved for variables that can be zero.
    """
    try:
        a = float(first)
        b = float(second)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - min(a, b) / max(a, b))))


def match_cost(
    previous: ExtremeWeatherEvent,
    candidate: ExtremeWeatherEvent,
    config: MatchingConfig,
) -> float:
    """Return the matching cost, or infinity when the pair is rejected.

    See the module docstring for the exact formula and the rejection rules.
    """
    if previous.event_type is not candidate.event_type:
        return INFINITE_COST

    # The gap must be strictly forward: a track never links to a detection from
    # the same or an earlier step (elapsed_hours is absolute, so it is only used
    # for the speed once the sign has been checked here).
    try:
        gap_hours = (candidate.timestamp - previous.timestamp).total_seconds() / 3600.0
    except TypeError:  # naive vs aware timestamps cannot be compared
        return INFINITE_COST
    if gap_hours <= 0.0 or gap_hours > config.max_time_gap_hours:
        return INFINITE_COST

    distance = haversine_distance_km(
        previous.latitude, previous.longitude, candidate.latitude, candidate.longitude
    )
    if distance > config.max_distance_km:
        return INFINITE_COST

    if movement_speed_kmh(distance, gap_hours) > config.max_speed_kmh:
        return INFINITE_COST

    cost = config.distance_weight * (distance / config.max_distance_km)
    cost += config.intensity_weight * intensity_mismatch(previous.intensity, candidate.intensity)
    total = config.weight_sum
    return cost / total if total > 0.0 else cost


def match_events(
    previous_events: Sequence[ExtremeWeatherEvent],
    next_events: Sequence[ExtremeWeatherEvent],
    config: MatchingConfig | None = None,
) -> MatchResult:
    """Match the events of two consecutive steps deterministically.

    Args:
        previous_events: Events of the earlier step.
        next_events: Events of the later step.
        config: Matching configuration (defaults to :class:`MatchingConfig`).

    Returns:
        A :class:`MatchResult` whose ``pairs`` is a one-to-one mapping from
        previous event id to next event id.

    Raises:
        TrackingError: If a single step contains duplicate event ids.
    """
    effective_config = config or MatchingConfig()
    previous_ids = [event.event_id for event in previous_events]
    next_ids = [event.event_id for event in next_events]
    if len(set(previous_ids)) != len(previous_ids) or len(set(next_ids)) != len(next_ids):
        raise TrackingError("Duplicate event ids inside a single detection step.")

    scored: list[tuple[float, str, str]] = []
    rejected = 0
    for previous in previous_events:
        for candidate in next_events:
            cost = match_cost(previous, candidate, effective_config)
            if cost == INFINITE_COST:
                rejected += 1
                continue
            scored.append((cost, previous.event_id, candidate.event_id))

    # Deterministic order: cheapest first, then lexicographic on the ids.
    scored.sort(key=lambda item: (item[0], item[1], item[2]))

    pairs: dict[str, str] = {}
    used_next: set[str] = set()
    for _cost, previous_id, next_id in scored:
        if previous_id in pairs or next_id in used_next:
            continue
        pairs[previous_id] = next_id
        used_next.add(next_id)

    return MatchResult(
        pairs=pairs,
        unmatched_previous=tuple(
            event.event_id for event in previous_events if event.event_id not in pairs
        ),
        unmatched_next=tuple(
            event.event_id for event in next_events if event.event_id not in used_next
        ),
        rejected_pairs=rejected,
    )
