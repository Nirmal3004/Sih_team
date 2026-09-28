"""Baseline models for Phase 9 evaluation (SIH26078).

A model is only interesting relative to something simpler. This module
implements the two baselines any event-prediction claim must clear:

**Persistence** - the future state equals the current state::

    next_position = current_position
    next_intensity = current_intensity

This is re-used from :mod:`extreme_weather.validation.metrics` rather than
reimplemented, so Phase 7 and Phase 9 cannot drift apart.

**Constant velocity** - the most recent movement continues::

    next_position = current_position + v_last * dt
    next_intensity = current_intensity

Intensity is deliberately *not* extrapolated. A trend extrapolation on
intensity would be a third modelling choice the comparison should not smuggle
in, and the documentation says so explicitly.

Why constant velocity matters
-----------------------------
Persistence is a weak baseline for anything that moves. A GNN that beats
persistence on a fast-moving event may simply have learned "events drift
eastward", which constant velocity already captures for free. Beating *this*
baseline is the comparison that carries information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from extreme_weather.validation.ground_truth import haversine_km
from extreme_weather.validation.metrics import summarise

__all__ = [
    "CONSTANT_VELOCITY",
    "PERSISTENCE",
    "BASELINE_DEFINITIONS",
    "constant_velocity_baseline",
    "describe_baselines",
    "evaluate_baseline",
]

#: Identifier of the persistence baseline.
PERSISTENCE = "persistence"

#: Identifier of the constant-velocity baseline.
CONSTANT_VELOCITY = "constant_velocity"

#: Human-readable definitions, surfaced in the report so a reader does not have
#: to infer what a baseline number means.
BASELINE_DEFINITIONS: dict[str, dict[str, str]] = {
    PERSISTENCE: {
        "name": "Persistence",
        "formula": "next_position = current_position; next_intensity = current_intensity",
        "description": (
            "Assumes the event does not change at all over the forecast step. "
            "The minimum bar any movement model must clear."
        ),
        "why": (
            "If a model cannot beat this, it is adding noise rather than signal."
        ),
    },
    CONSTANT_VELOCITY: {
        "name": "Constant velocity",
        "formula": "next_position = current_position + last_velocity * dt; next_intensity = current_intensity",
        "description": (
            "Extrapolates the most recent observed movement vector. Intensity is "
            "held constant, not extrapolated: a trend model for intensity would "
            "add an unstated modelling choice to the comparison."
        ),
        "why": (
            "Persistence is weak for anything that moves. Beating constant "
            "velocity shows the model learned more than 'events drift'."
        ),
    },
}


def constant_velocity_baseline(
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    current_intensity: np.ndarray,
    previous_lat: np.ndarray,
    previous_lon: np.ndarray,
    future_lat: np.ndarray,
    future_lon: np.ndarray,
    future_intensity: np.ndarray,
) -> dict[str, float | int]:
    """Score the "keep going the way you were going" forecast.

    The velocity is taken from the last two observed points. The step length is
    fixed at one hour, matching the graph's target definition (a target is the
    next hourly state), so the extrapolation is comparable with persistence on
    exactly the same samples.

    Args:
        current_lat: Current latitudes, degrees.
        current_lon: Current longitudes, degrees.
        current_intensity: Current intensities.
        previous_lat: Latitudes one step before ``current``, degrees.
        previous_lon: Longitudes one step before ``current``, degrees.
        future_lat: Actual next-step latitudes, degrees.
        future_lon: Actual next-step longitudes, degrees.
        future_intensity: Actual next-step intensities.

    Returns:
        The same keys :func:`persistence_baseline` returns, so the methods are
        directly comparable.

    Raises:
        ValueError: If the arrays differ in length or are empty.
    """
    cur_lat = np.asarray(current_lat, dtype="float64")
    cur_lon = np.asarray(current_lon, dtype="float64")
    cur_int = np.asarray(current_intensity, dtype="float64")
    prev_lat = np.asarray(previous_lat, dtype="float64")
    prev_lon = np.asarray(previous_lon, dtype="float64")
    fut_lat = np.asarray(future_lat, dtype="float64")
    fut_lon = np.asarray(future_lon, dtype="float64")
    fut_int = np.asarray(future_intensity, dtype="float64")

    shapes = {
        cur_lat.shape,
        cur_lon.shape,
        cur_int.shape,
        prev_lat.shape,
        prev_lon.shape,
        fut_lat.shape,
        fut_lon.shape,
        fut_int.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            "Constant-velocity inputs must all have the same shape; got "
            f"{sorted(shapes)}."
        )
    if cur_lat.size == 0:
        raise ValueError("Constant-velocity baseline needs at least one sample.")

    # Displacement of the last observed step, in degrees, extrapolated one step.
    velocity_lat = cur_lat - prev_lat
    velocity_lon = cur_lon - prev_lon
    predicted_lat = cur_lat + velocity_lat
    predicted_lon = cur_lon + velocity_lon

    lat_error = fut_lat - predicted_lat
    lon_error = fut_lon - predicted_lon
    intensity_error = fut_int - cur_int
    displacement = np.array(
        [
            haversine_km(p0, p1, t0, t1)
            for p0, p1, t0, t1 in zip(predicted_lat, predicted_lon, fut_lat, fut_lon)
        ],
        dtype="float64",
    )
    return {
        "sample_count": int(cur_lat.size),
        "latitude_mae_deg": float(np.mean(np.abs(lat_error))),
        "longitude_mae_deg": float(np.mean(np.abs(lon_error))),
        "intensity_mae": float(np.mean(np.abs(intensity_error))),
        "latitude_rmse_deg": float(np.sqrt(np.mean(lat_error**2))),
        "longitude_rmse_deg": float(np.sqrt(np.mean(lon_error**2))),
        "intensity_rmse": float(np.sqrt(np.mean(intensity_error**2))),
        "displacement_error_km_mean": float(np.mean(displacement)),
        "displacement_error_km_median": float(np.median(displacement)),
        "displacement_error_km_std": float(np.std(displacement)),
        "displacement_error_km_p90": float(np.percentile(displacement, 90)),
        "combined_loss": float(
            np.mean(
                np.abs(lat_error)
                + np.abs(lon_error)
                + np.abs(intensity_error)
            )
        ),
    }


def evaluate_baseline(
    name: str,
    *,
    current_latitude: np.ndarray,
    current_longitude: np.ndarray,
    current_intensity: np.ndarray,
    future_latitude: np.ndarray,
    future_longitude: np.ndarray,
    future_intensity: np.ndarray,
    previous_latitude: np.ndarray | None = None,
    previous_longitude: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score a named baseline on supplied samples.

    Args:
        name: ``persistence`` or ``constant_velocity``.
        current_latitude: Current latitudes, degrees.
        current_longitude: Current longitudes, degrees.
        current_intensity: Current intensities.
        future_latitude: Actual next-step latitudes, degrees.
        future_longitude: Actual next-step longitudes, degrees.
        future_intensity: Actual next-step intensities.
        previous_latitude: Required for ``constant_velocity``.
        previous_longitude: Required for ``constant_velocity``.

    Returns:
        The metric dictionary, plus a ``method`` field naming the baseline.

    Raises:
        ValueError: If the baseline name is unknown, or a required input is
            missing. Failing here prevents a silently-wrong baseline being
            scored.
    """
    from extreme_weather.validation.metrics import persistence_baseline

    if name == PERSISTENCE:
        metrics = persistence_baseline(
            current_latitude,
            current_longitude,
            current_intensity,
            future_latitude,
            future_longitude,
            future_intensity,
        )
    elif name == CONSTANT_VELOCITY:
        if previous_latitude is None or previous_longitude is None:
            raise ValueError(
                "The constant-velocity baseline needs the previous step's "
                "latitudes and longitudes."
            )
        metrics = constant_velocity_baseline(
            current_latitude,
            current_longitude,
            current_intensity,
            previous_latitude,
            previous_longitude,
            future_latitude,
            future_longitude,
            future_intensity,
        )
    else:
        known = ", ".join(sorted(BASELINE_DEFINITIONS))
        raise ValueError(f"Unknown baseline '{name}'. Known baselines: {known}.")

    # The persistence baseline has no std/percentile; add them so every method
    # exposes the same summary fields and the frontend can render them uniformly.
    if "displacement_error_km_std" not in metrics:
        from extreme_weather.validation.ground_truth import haversine_km as _hav

        displacement = np.array(
            [
                _hav(a0, b0, a1, b1)
                for a0, b0, a1, b1 in zip(
                    np.asarray(current_latitude, dtype="float64"),
                    np.asarray(current_longitude, dtype="float64"),
                    np.asarray(future_latitude, dtype="float64"),
                    np.asarray(future_longitude, dtype="float64"),
                )
            ],
            dtype="float64",
        )
        metrics = dict(metrics)
        metrics["displacement_error_km_std"] = float(np.std(displacement))
        metrics["displacement_error_km_p90"] = float(np.percentile(displacement, 90))
        metrics["combined_loss"] = float(
            np.mean(
                np.abs(
                    np.asarray(future_latitude, dtype="float64")
                    - np.asarray(current_latitude, dtype="float64")
                )
                + np.abs(
                    np.asarray(future_longitude, dtype="float64")
                    - np.asarray(current_longitude, dtype="float64")
                )
                + np.abs(
                    np.asarray(future_intensity, dtype="float64")
                    - np.asarray(current_intensity, dtype="float64")
                )
            )
        )

    result: dict[str, Any] = dict(metrics)
    result["method"] = name
    result["definition"] = BASELINE_DEFINITIONS[name]
    summary = summarise(
        [
            float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "sample_count"
        ]
    )
    if summary is not None:
        result["summary"] = summary
    return result


def describe_baselines() -> dict[str, dict[str, str]]:
    """Return the baseline definitions, for the report and the API."""
    return {key: dict(value) for key, value in BASELINE_DEFINITIONS.items()}
