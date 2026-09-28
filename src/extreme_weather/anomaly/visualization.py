"""Lightweight visualisation of an anomaly analysis (SIH26078, Phase 1).

A single PNG with four panels:

1. the analysed physical field (for example precipitation),
2. the anomaly score,
3. the detected extreme cells,
4. the detected extreme cells with the event centroids marked.

Matplotlib is used with the non-interactive ``Agg`` backend, so the utility works
headless (CI, servers, Docker) and the import is done lazily - visualisation is
never required to run the pipeline or the tests of other modules.

Visualisation is a *diagnostic aid*, never a required pipeline step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from extreme_weather.schemas.event import ExtremeWeatherEvent

__all__ = ["VisualizationError", "plot_analysis_overview"]


class VisualizationError(RuntimeError):
    """Raised when a figure cannot be produced."""


def _require_matplotlib() -> tuple[object, object]:
    """Import pyplot with the Agg backend, raising a clear error if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib ships with metpy
        raise VisualizationError(
            "matplotlib is required for visualisation. Install it with 'pip install matplotlib'."
        ) from exc
    return plt, matplotlib


def plot_analysis_overview(
    values: xr.DataArray,
    scores: xr.DataArray,
    mask: xr.DataArray,
    events: Sequence[ExtremeWeatherEvent] = (),
    *,
    output_path: Path | str | None = None,
    title: str = "DeepWeather AI - anomaly detection (prototype)",
    time_index: int | None = None,
) -> Path | None:
    """Render the four-panel analysis overview.

    Args:
        values: Physical field (time, latitude, longitude).
        scores: Anomaly score field with the same dims.
        mask: Boolean extreme-cell mask with the same dims.
        events: Detected events, used to mark the centroids.
        output_path: PNG destination. ``None`` returns the figure without saving
            (useful for tests and notebooks).
        title: Figure title.
        time_index: Time step to plot; defaults to the step with the most
            detected cells.

    Returns:
        The written PNG path, or ``None`` when nothing was saved.
    """
    if values.dims != scores.dims or values.dims != mask.dims:
        raise VisualizationError(
            f"Field dims must match: values={values.dims}, scores={scores.dims}, mask={mask.dims}."
        )
    for dim in values.dims:
        if values.sizes[dim] != mask.sizes[dim]:
            raise VisualizationError(
                f"Field sizes must match along '{dim}': "
                f"values={values.sizes[dim]}, mask={mask.sizes[dim]}."
            )

    plt, _ = _require_matplotlib()

    if time_index is None:
        counts = np.asarray(mask.values, dtype=bool).reshape(mask.values.shape[0], -1).sum(axis=1)
        time_index = int(np.argmax(counts)) if counts.size else 0

    stamp = str(np.asarray(values["time"].values)[time_index])
    slice_events = [
        event
        for event in events
        if str(np.datetime64(event.timestamp.replace(tzinfo=None), "s")) == stamp
    ]

    figure, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    panels = (
        (values, f"{values.name or 'field'} ({stamp})", "YlGnBu"),
        (scores, f"anomaly score ({scores.attrs.get('long_name', 'score')})", "magma"),
        (mask.astype("int8"), "detected extreme cells", "Reds"),
    )

    for axis, (field, label, colormap) in zip(axes[:3], panels):
        data = np.asarray(field.values, dtype="float64")
        if data.ndim == 3:
            data = data[time_index]
        mesh = axis.pcolormesh(
            field["longitude"].values,
            field["latitude"].values,
            data,
            cmap=colormap,
            shading="auto",
        )
        figure.colorbar(mesh, ax=axis, shrink=0.85)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")

    axis = axes[3]
    mask_data = np.asarray(mask.values, dtype="int8")
    if mask_data.ndim == 3:
        mask_data = mask_data[time_index]
    axis.pcolormesh(
        values["longitude"].values,
        values["latitude"].values,
        mask_data,
        cmap="Reds",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    for event in slice_events:
        axis.plot(event.longitude, event.latitude, marker="+", color="black", markersize=12)
        axis.annotate(
            event.event_id,
            (event.longitude, event.latitude),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set_title(f"detected events ({len(slice_events)})", fontsize=9)
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")

    figure.suptitle(f"{title} - prototype, synthetic data", fontsize=11)

    if output_path is None:
        plt.close(figure)
        return None

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    return destination
