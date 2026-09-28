"""Node and edge features for the GNN prototype (SIH26078, Phase 3).

Every feature is derived **only from information available at the source node's
own timestamp**: the state of the event, how it moved to get there, and the
calendar hour. No feature reads the future target (the next position or
intensity), so the model cannot leak it; the chronological split in
:mod:`extreme_weather.gnn.dataset` relies on that guarantee.

Node features (:data:`NODE_FEATURE_NAMES`, 16 values)

=====================  =====  ===================================================
Name                   Unit   Meaning
=====================  =====  ===================================================
``latitude``           deg    Centroid latitude of the event.
``longitude``          deg    Centroid longitude of the event.
``latitude_scaled``    1      ``latitude / 90`` - geography in ``[-1, 1]`` so raw
                             degrees cannot dominate the model.
``longitude_scaled``   1      ``longitude / 180`` - same, in ``[-1, 1]``.
``intensity_scaled``   1      ``intensity / intensity_scale`` (default 100, e.g.
                             mm h-1), so a 55 mm h-1 burst is ~0.55.
``anomaly_score``      1      Phase 1 anomaly score of the detection.
``speed_scaled``       1      ``speed_kmh / speed_scale`` (default 120 km/h).
``bearing_sin``        1      ``sin(bearing)``; undefined bearing -> 0.
``bearing_cos``        1      ``cos(bearing)``; undefined bearing -> 1.
``is_stationary``      1      1.0 when the event did not move (``CALM``), else 0.
``cell_count_scaled``  1      ``log1p(cell_count) / log1p(cell_count_scale)``,
                             capped to ``[0, 1]``; missing count -> 0.
``hour_sin``           1      ``sin(2*pi*hour/24)`` - diurnal cycle, continuous
                             across the 24 h wrap-around.
``hour_cos``           1      ``cos(2*pi*hour/24)`` - see ``hour_sin``.
``is_extreme_rainfall`` 1     Event type one-hot block, ordered as
``is_extreme_wind``    1      :data:`EVENT_TYPE_ORDER`. A type outside that tuple
``is_extreme_temp``    1      yields all-zero columns (never mapped onto a real
                             type).
=====================  =====  ===================================================

Using ``sin``/``cos`` pairs rather than raw ``bearing_degrees`` or ``hour`` keeps
the encoding continuous: 359 deg and 1 deg are neighbours instead of opposite
ends of the range.

Edge features (:data:`EDGE_FEATURE_NAMES`, 6 values)

=========================  =====  ===============================================
Name                       Unit   Meaning
=========================  =====  ===============================================
``distance_scaled``        1      ``distance_km / edge_distance_scale`` (250 km).
``bearing_sin``            1      ``sin`` of the source->target bearing.
``bearing_cos``            1      ``cos`` of the source->target bearing.
``time_delta_scaled``      1      ``hours / edge_time_scale`` (6 h).
``is_temporal``            1      1.0 within a track, 0.0 otherwise.
``is_spatial``             1      0.0 within a track, 1.0 otherwise.
=========================  =====  ===============================================

The two indicator columns let the model treat the edge kinds differently, which
matters because a temporal edge is a real trajectory step while a spatial edge
only means "two events were near each other at the same time".

A note on :class:`TrackPoint`: it carries ``severity`` but **not** the event
type, so the type must be supplied from the owning
:class:`~extreme_weather.schemas.track.EventTrack`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.track import TrackPoint
from extreme_weather.tracking.metrics import CALM, bearing_degrees, haversine_distance_km

__all__ = [
    "EDGE_FEATURE_NAMES",
    "EDGE_TEMPORAL",
    "EDGE_SPATIAL",
    "EVENT_TYPE_ORDER",
    "NODE_FEATURE_NAMES",
    "TARGET_NAMES",
    "FeatureConfig",
    "edge_feature_vector",
    "event_type_columns",
    "node_feature_vector",
    "target_vector",
]

#: Deterministic event-type order backing the one-hot columns.
EVENT_TYPE_ORDER: tuple[EventType, ...] = (
    EventType.EXTREME_RAINFALL,
    EventType.EXTREME_WIND,
    EventType.EXTREME_TEMPERATURE,
)

#: Edge labels, stored numerically as 1.0/0.0 in ``is_temporal``/``is_spatial``.
EDGE_TEMPORAL: str = "temporal"
EDGE_SPATIAL: str = "spatial"

NODE_FEATURE_NAMES: tuple[str, ...] = (
    "latitude",
    "longitude",
    "latitude_scaled",
    "longitude_scaled",
    "intensity_scaled",
    "anomaly_score",
    "speed_scaled",
    "bearing_sin",
    "bearing_cos",
    "is_stationary",
    "cell_count_scaled",
    "hour_sin",
    "hour_cos",
    "is_extreme_rainfall",
    "is_extreme_wind",
    "is_extreme_temp",
)



@dataclass(frozen=True)
class FeatureConfig:
    """Divisors mapping raw physical units into the model's input range.

    These are **fixed constants, not statistics fitted on the training data**. A
    checkpoint is therefore readable without the training set, and inference on a
    new track applies exactly the same transform as training. This is a
    deliberate trade-off: less flexible than z-scoring, but reproducible and
    fully inspectable.

    Attributes:
        intensity_scale: Divisor of the physical intensity (mm h-1, m s-1, K).
        speed_scale: Divisor of the Phase 2 speed (km/h).
        cell_count_scale: Divisor inside ``log1p`` for the cell count.
        edge_distance_scale: Divisor of the edge distance (km).
        edge_time_scale: Divisor of the edge time gap (hours).
    """

    intensity_scale: float = 100.0
    speed_scale: float = 120.0
    cell_count_scale: float = 100.0
    edge_distance_scale: float = 250.0
    edge_time_scale: float = 6.0

    def __post_init__(self) -> None:
        for name in (
            "intensity_scale",
            "speed_scale",
            "cell_count_scale",
            "edge_distance_scale",
            "edge_time_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"FeatureConfig.{name} must be finite and > 0 (got {value!r})."
                )

    @property
    def cell_count_log_scale(self) -> float:
        """``log1p(cell_count_scale)``, precomputed for the feature builder."""
        return math.log1p(self.cell_count_scale)


def event_type_columns(event_type: EventType | str | None) -> list[float]:
    """Return the one-hot columns of ``event_type`` in :data:`EVENT_TYPE_ORDER`.

    An unknown or missing type yields all zeros instead of being mapped onto an
    arbitrary real type.
    """
    try:
        resolved = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
    except ValueError:
        resolved = None
    return [1.0 if member == resolved else 0.0 for member in EVENT_TYPE_ORDER]


def _ratio(value: float, divisor: float) -> float:
    """Divide and replace a non-finite result with ``0.0``."""
    try:
        result = float(value) / float(divisor)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _bearing_angles(bearing: float | None) -> tuple[float, float]:
    """``(sin, cos)`` of a bearing; the undefined bearing maps to ``(0, 1)``."""
    if bearing is None:
        return 0.0, 1.0
    try:
        value = float(bearing)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not math.isfinite(value):
        return 0.0, 1.0
    radians = math.radians(value)
    return math.sin(radians), math.cos(radians)


def _hour_angles(timestamp: datetime) -> tuple[float, float]:
    """``(sin, cos)`` of the diurnal cycle, continuous across the 24 h wrap."""
    hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    angle = 2.0 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)


def node_feature_vector(
    point: TrackPoint,
    event_type: EventType | str | None,
    config: FeatureConfig | None = None,
) -> list[float]:
    """Build the feature vector of one track point.

    Args:
        point: The current track point. Its stored ``speed_kmh``,
            ``bearing_degrees`` and ``direction`` come from Phase 2, so the first
            point of a track (speed 0, bearing ``None``, ``"CALM"``) needs no
            special case: it is simply the "did not move" state.
        event_type: The event type of the owning track (``TrackPoint`` carries
            only ``severity``).
        config: Scaling constants; defaults to :class:`FeatureConfig`.

    Returns:
        Exactly ``len(NODE_FEATURE_NAMES)`` finite floats ordered like
        :data:`NODE_FEATURE_NAMES`. Nothing here reads the next point.
    """
    cfg = config or FeatureConfig()
    bearing_sin, bearing_cos = _bearing_angles(point.bearing_degrees)
    hour_sin, hour_cos = _hour_angles(point.timestamp)

    stationary = 1.0 if (point.direction or CALM) == CALM else 0.0
    if point.cell_count is None:
        cell_count_scaled = 0.0
    else:
        cell_count_scaled = min(
            1.0, max(0.0, math.log1p(float(point.cell_count)) / cfg.cell_count_log_scale)
        )

    return [
        float(point.latitude),
        float(point.longitude),
        _ratio(point.latitude, 90.0),
        _ratio(point.longitude, 180.0),
        _ratio(point.intensity, cfg.intensity_scale),
        float(point.anomaly_score),
        _ratio(max(0.0, float(point.speed_kmh)), cfg.speed_scale),
        bearing_sin,
        bearing_cos,
        stationary,
        cell_count_scaled,
        hour_sin,
        hour_cos,
        *event_type_columns(event_type),
    ]


def edge_feature_vector(
    source: TrackPoint,
    target: TrackPoint,
    edge_kind: str = EDGE_TEMPORAL,
    config: FeatureConfig | None = None,
) -> list[float]:
    """Build the feature vector of the directed edge ``source -> target``.

    Only the *source* state and the pair geometry are used; the target's own
    features arrive through message passing. That keeps the edge encoding free of
    information about the destination.

    Args:
        source: Origin node.
        target: Destination node.
        edge_kind: :data:`EDGE_TEMPORAL` or :data:`EDGE_SPATIAL`.
        config: Scaling constants; defaults to :class:`FeatureConfig`.

    Returns:
        Exactly ``len(EDGE_FEATURE_NAMES)`` finite floats.

    Raises:
        ValueError: If ``edge_kind`` is not a known edge kind.
    """
    if edge_kind not in (EDGE_TEMPORAL, EDGE_SPATIAL):
        raise ValueError(
            f"Unknown edge kind {edge_kind!r}; expected {EDGE_TEMPORAL!r} or {EDGE_SPATIAL!r}."
        )
    cfg = config or FeatureConfig()

    distance = haversine_distance_km(
        source.latitude, source.longitude, target.latitude, target.longitude
    )
    try:
        gap_hours = (target.timestamp - source.timestamp).total_seconds() / 3600.0
    except TypeError:  # naive vs aware timestamps cannot be subtracted
        gap_hours = 0.0
    if not math.isfinite(gap_hours) or gap_hours < 0.0:
        gap_hours = 0.0

    bearing_sin, bearing_cos = _bearing_angles(
        bearing_degrees(source.latitude, source.longitude, target.latitude, target.longitude)
    )

    return [
        _ratio(distance, cfg.edge_distance_scale),
        bearing_sin,
        bearing_cos,
        _ratio(gap_hours, cfg.edge_time_scale),
        1.0 if edge_kind == EDGE_TEMPORAL else 0.0,
        1.0 if edge_kind == EDGE_SPATIAL else 0.0,
    ]


def target_vector(current: TrackPoint, following: TrackPoint) -> list[float]:
    """Return ``[delta_latitude, delta_longitude, delta_intensity]``.

    The model predicts **changes**, not absolute coordinates, so the learned
    quantity stays small, zero-centred and free of the Earth's large offset. The
    caller decodes the next state by adding these deltas to ``current``. The cost
    of that choice is documented in ``docs/GNN.md``.
    """
    return [
        float(following.latitude) - float(current.latitude),
        float(following.longitude) - float(current.longitude),
        float(following.intensity) - float(current.intensity),
    ]

EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "distance_scaled",
    "bearing_sin",
    "bearing_cos",
    "time_delta_scaled",
    "is_temporal",
    "is_spatial",
)

#: Regression targets: *changes*, never absolute values (see ``docs/GNN.md``).
TARGET_NAMES: tuple[str, ...] = ("delta_latitude", "delta_longitude", "delta_intensity")
