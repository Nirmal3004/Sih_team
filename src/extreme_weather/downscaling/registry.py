"""Downscaling method registry / factory (SIH26078).

Methods are registered by name so that the public API, the CLI and the AI models
all select an implementation the same way::

    from extreme_weather.downscaling import downscale, available_methods

    available_methods()                     # ('ai', 'cubic', 'nearest')
    fine = downscale(coarse, 0.05)          # nearest (default baseline)
    fine = downscale(coarse, 0.05, method="cubic")
    fine = downscale(coarse, 0.05, method="ai")   # Phase 4 learned residual model

The Phase 0 baselines are preserved unchanged. The Phase 4 ``ai`` method joins
them in the same registry; it loads ``models/downscaling/best.pt`` itself and
raises a clear error when no checkpoint exists.

Custom methods join the registry with the decorator::

    from extreme_weather.downscaling import register_method

    @register_method("diffusion_sr", description="FUTURE diffusion super-resolution")
    def diffusion_sr(data, target_resolution, **kwargs): ...

The registry is a plain module-level mapping - the only shared state in this
package - because a factory needs to be discoverable from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import xarray as xr

__all__ = [
    "DownscalingMethod",
    "MethodFunction",
    "UnknownDownscalingMethodError",
    "available_methods",
    "get_method",
    "register_method",
    "registered_methods",
    "downscale",
]

#: Signature every downscaling implementation must follow.
MethodFunction = Callable[..., xr.DataArray]


class UnknownDownscalingMethodError(ValueError):
    """Raised when a downscaling method name is not present in the registry."""


@dataclass(frozen=True)
class DownscalingMethod:
    """A registered downscaling implementation."""

    name: str
    function: MethodFunction
    description: str = ""
    minimum_points_per_dim: int = 2


_REGISTRY: dict[str, DownscalingMethod] = {}


def _register(method: DownscalingMethod) -> DownscalingMethod:
    _REGISTRY[method.name] = method
    return method


def register_method(
    name: str,
    *,
    description: str = "",
    minimum_points_per_dim: int = 2,
    overwrite: bool = False,
) -> Callable[[MethodFunction], MethodFunction]:
    """Decorator factory that registers a downscaling implementation under ``name``.

    Args:
        name: Registry key (lower-case, no spaces).
        description: Human readable description exposed through the API.
        minimum_points_per_dim: Source points required per interpolated dimension.
        overwrite: Allow replacing an already registered method.

    Raises:
        ValueError: If ``name`` is empty or already registered without ``overwrite``.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("Downscaling method name must not be empty.")
    if not overwrite and key in _REGISTRY:
        raise ValueError(
            f"Downscaling method '{key}' is already registered. "
            "Pass overwrite=True to replace it."
        )

    def decorator(function: MethodFunction) -> MethodFunction:
        if not callable(function):
            raise TypeError(f"Downscaling method '{key}' must be callable.")
        docstring = (function.__doc__ or "").strip()
        resolved_description = description or (docstring.splitlines()[0] if docstring else "")
        _register(
            DownscalingMethod(
                name=key,
                function=function,
                description=resolved_description,
                minimum_points_per_dim=int(minimum_points_per_dim),
            )
        )
        return function

    return decorator


def _ensure_builtin_methods() -> None:
    """Import the built-in methods so the registry is populated (idempotent)."""
    from extreme_weather.downscaling import baseline  # noqa: F401  (registers methods)
    from extreme_weather.downscaling.inference import register_ai_method

    # Phase 4: the learned 'ai' method joins the Phase 0 baselines in the same
    # registry, so `downscale(da, res, method="ai")` works like the others.
    register_ai_method()


def available_methods() -> tuple[str, ...]:
    """Return the sorted names of all registered downscaling methods."""
    _ensure_builtin_methods()
    return tuple(sorted(_REGISTRY))


def registered_methods() -> dict[str, DownscalingMethod]:
    """Return a copy of the registry as ``{name: DownscalingMethod}``."""
    _ensure_builtin_methods()
    return dict(_REGISTRY)


def get_method(name: str) -> DownscalingMethod:
    """Return the registered method called ``name``.

    Raises:
        UnknownDownscalingMethodError: If the method is not registered; the
            message lists the available method names.
    """
    _ensure_builtin_methods()
    if not isinstance(name, str):
        raise UnknownDownscalingMethodError(
            f"Downscaling method name must be a string, got {type(name).__name__}."
        )
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise UnknownDownscalingMethodError(
            f"Unknown downscaling method '{name}'. Available methods: "
            f"{', '.join(sorted(_REGISTRY))}. Use register_method() to add a new one."
        )
    return _REGISTRY[key]


def downscale(
    data: xr.DataArray,
    target_resolution: float,
    method: str = "nearest",
    **kwargs: Any,
) -> xr.DataArray:
    """Downscale ``data`` to ``target_resolution`` (degrees) using ``method``.

    Args:
        data: Coarse input field, for example ``ds["precipitation"].isel(time=0)``.
        target_resolution: Target grid spacing in degrees. The reference task of
            this project is 0.125 deg (~12 km) -> 0.05 deg (~5 km).
        method: Registered method name (``"nearest"`` or ``"cubic"`` in Phase 0).
        **kwargs: Forwarded to the implementation (for example ``latitude_dim``).

    Returns:
        The downscaled :class:`xarray.DataArray` with the same name/attrs as the
        input and a finer ``latitude``/``longitude`` grid.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"downscale expects an xarray.DataArray, got {type(data).__name__}. "
            "Select a single variable first (for example ds['precipitation'])."
        )
    registered = get_method(method)
    return registered.function(data, target_resolution, **kwargs)
