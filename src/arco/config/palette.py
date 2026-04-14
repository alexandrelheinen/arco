"""Color palette utilities for the ARCO visualization system.

Provides programmatic color derivation from the base colors defined in
``colors.yml``.  All per-algorithm visual layers are derived from the base
method color using HSL manipulation, ensuring visual consistency without
hardcoded hex values for derived shades.

The five rendering layers are (in stacking order, bottom to top):
  a. tree        - exploration tree edges/nodes (very transparent, grayish)
  b. path        - raw planned path (slightly darker, alpha ~0.6)
  c. pruned      - pruned/smoothed path nodes (brighter, alpha ~0.85)
  d. trajectory  - predicted/optimized trajectory (darker, alpha ~0.9)
  e. vehicle     - executed trajectory and vehicle body (base color, alpha 1.0)

Usage::

    from arco.config.palette import (
        annotation_hex, obstacle_hex, method_base_hex,
        layer_hex, layer_rgb, hex_to_rgb, ui_rgb,
        LAYER_ALPHA,
    )
"""

from __future__ import annotations

import colorsys

from arco.config import load_config

_COLORS = load_config("colors")

# Alpha values for each rendering layer.
LAYER_ALPHA: dict[str, float] = {
    "tree": 0.12,  # Very transparent — must not dominate dense trees
    "path": 0.60,  # Raw path — dimmed but visible
    "pruned": 0.85,  # Pruned/smoothed — bright and clear
    "trajectory": 0.90,  # Predicted trajectory — prominent
    "vehicle": 1.00,  # Executed trajectory and vehicle body — fully opaque
}


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def annotation_hex(dark_bg: bool = False) -> str:
    """Return the annotation color as a hex string.

    Args:
        dark_bg: When ``True``, return the light near-white variant
            suitable for dark backgrounds.  Defaults to the black
            variant for light backgrounds.

    Returns:
        Hex color string (e.g. ``"#000000"``).
    """
    if dark_bg:
        return str(_COLORS["annotation"]["dark_bg"])
    return str(_COLORS["annotation"]["default"])


def annotation_rgb(dark_bg: bool = False) -> tuple[int, int, int]:
    """Return the annotation color as an ``(R, G, B)`` tuple in ``[0, 255]``.

    Args:
        dark_bg: When ``True``, return the light near-white variant.

    Returns:
        Integer RGB tuple.
    """
    return hex_to_rgb(annotation_hex(dark_bg))


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------


def obstacle_hex() -> str:
    """Return the obstacle fill color as a hex string.

    Returns:
        Hex color string for the pastel-red obstacle fill.
    """
    return str(_COLORS["obstacle"]["fill"])


def obstacle_rgb() -> tuple[int, int, int]:
    """Return the obstacle fill color as an ``(R, G, B)`` tuple in ``[0, 255]``.

    Returns:
        Integer RGB tuple.
    """
    return hex_to_rgb(obstacle_hex())


def obstacle_float() -> tuple[float, float, float]:
    """Return the obstacle fill color as an ``(r, g, b)`` tuple in ``[0.0, 1.0]``.

    Returns:
        Float RGB tuple.
    """
    return hex_to_float(obstacle_hex())


# ---------------------------------------------------------------------------
# Method / algorithm colors
# ---------------------------------------------------------------------------


def method_base_hex(method: str) -> str:
    """Return the base color for an algorithm as a hex string.

    Args:
        method: Algorithm key — ``"rrt"``, ``"sst"``, ``"astar"``, or
            ``"dstar"``.

    Returns:
        Hex color string.

    Raises:
        KeyError: If *method* is not defined in ``colors.yml``.
    """
    return str(_COLORS["methods"][method]["base"])


def method_base_rgb(method: str) -> tuple[int, int, int]:
    """Return the base color for an algorithm as an ``(R, G, B)`` tuple.

    Args:
        method: Algorithm key.

    Returns:
        Integer RGB tuple in ``[0, 255]``.
    """
    return hex_to_rgb(method_base_hex(method))


def method_base_float(method: str) -> tuple[float, float, float]:
    """Return the base color for an algorithm as an ``(r, g, b)`` tuple.

    Args:
        method: Algorithm key.

    Returns:
        Float RGB tuple in ``[0.0, 1.0]``.
    """
    return hex_to_float(method_base_hex(method))


# ---------------------------------------------------------------------------
# Layer color derivation
# ---------------------------------------------------------------------------


def layer_hex(method: str, layer: str) -> str:
    """Return the derived hex color for a rendering layer of a method.

    Layers are derived programmatically from the base color via HSL
    manipulation so that the full palette remains consistent when a base
    color is changed.

    Layer rules (referenced from the issue specification):
      - ``"tree"``:       grayish tint — desaturate 30%, darken 5%.
      - ``"path"``:       slightly darker — darken 12%.
      - ``"pruned"``:     brighter/more saturated — lighten 5%, saturate 10%.
      - ``"trajectory"``: darker — darken 20%.
      - ``"vehicle"``:    same as base (identity).

    Args:
        method: Algorithm key (e.g. ``"rrt"``).
        layer: Rendering layer — ``"tree"``, ``"path"``, ``"pruned"``,
            ``"trajectory"``, or ``"vehicle"``.

    Returns:
        Hex color string for the derived shade.

    Raises:
        ValueError: If *layer* is not recognized.
    """
    base = method_base_hex(method)
    return _derive(base, layer)


def layer_rgb(method: str, layer: str) -> tuple[int, int, int]:
    """Return the derived ``(R, G, B)`` color for a rendering layer.

    Args:
        method: Algorithm key.
        layer: Rendering layer.

    Returns:
        Integer RGB tuple in ``[0, 255]``.
    """
    return hex_to_rgb(layer_hex(method, layer))


def layer_float(method: str, layer: str) -> tuple[float, float, float]:
    """Return the derived ``(r, g, b)`` color for a rendering layer.

    Args:
        method: Algorithm key.
        layer: Rendering layer.

    Returns:
        Float RGB tuple in ``[0.0, 1.0]``.
    """
    return hex_to_float(layer_hex(method, layer))


# ---------------------------------------------------------------------------
# UI / pygame chrome colors
# ---------------------------------------------------------------------------


def ui_rgb(key: str) -> tuple[int, int, int]:
    """Return a pygame UI chrome color as an ``(R, G, B)`` tuple.

    Args:
        key: Color key within the ``ui`` section of ``colors.yml``, e.g.
            ``"background"``, ``"hud_text"``, ``"road_dot"``.

    Returns:
        Integer RGB tuple in ``[0, 255]``.

    Raises:
        KeyError: If *key* is not defined under ``ui`` in ``colors.yml``.
    """
    v = _COLORS["ui"][key]
    return (int(v[0]), int(v[1]), int(v[2]))


# ---------------------------------------------------------------------------
# Low-level converters
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert a hex color string to an integer ``(R, G, B)`` tuple.

    Args:
        hex_str: Hex color string, optionally prefixed with ``#``.

    Returns:
        Integer RGB tuple in ``[0, 255]``.
    """
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_float(hex_str: str) -> tuple[float, float, float]:
    """Convert a hex color string to a float ``(r, g, b)`` tuple.

    Args:
        hex_str: Hex color string, optionally prefixed with ``#``.

    Returns:
        Float RGB tuple in ``[0.0, 1.0]``.
    """
    r, g, b = hex_to_rgb(hex_str)
    return r / 255.0, g / 255.0, b / 255.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive(base_hex: str, layer: str) -> str:
    """Derive a layer hex color from a base hex color via HSL adjustment.

    Args:
        base_hex: Base hex color string.
        layer: Rendering layer identifier.

    Returns:
        Derived hex color string.

    Raises:
        ValueError: If *layer* is not recognized.
    """
    if layer == "tree":
        # Grayish tint: desaturate 30%, darken 5% — keeps hue visible.
        return _adjust_hsl(
            base_hex, lightness_delta=-0.05, saturation_delta=-0.30
        )
    if layer == "path":
        # Slightly darker shade.
        return _adjust_hsl(base_hex, lightness_delta=-0.12)
    if layer == "pruned":
        # Brighter and slightly more saturated.
        return _adjust_hsl(
            base_hex, lightness_delta=+0.05, saturation_delta=+0.10
        )
    if layer == "trajectory":
        # Noticeably darker but still saturated.
        return _adjust_hsl(base_hex, lightness_delta=-0.20)
    if layer == "vehicle":
        # Identity — same as base.
        return base_hex
    raise ValueError(
        f"Unknown rendering layer {layer!r}. "
        "Valid layers: 'tree', 'path', 'pruned', 'trajectory', 'vehicle'."
    )


def _adjust_hsl(
    hex_str: str,
    lightness_delta: float = 0.0,
    saturation_delta: float = 0.0,
) -> str:
    """Adjust HSL channels of a hex color and return the modified hex string.

    Uses Python's :mod:`colorsys` module (HLS ordering: H, L, S).

    Args:
        hex_str: Source hex color string.
        lightness_delta: Additive lightness adjustment (clamped to ``[0, 1]``).
        saturation_delta: Additive saturation adjustment (clamped to ``[0, 1]``).

    Returns:
        Modified hex color string with the ``#`` prefix.
    """
    r, g, b = hex_to_float(hex_str)
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
    l = max(0.0, min(1.0, l + lightness_delta))  # noqa: E741
    s = max(0.0, min(1.0, s + saturation_delta))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    ri = int(round(r2 * 255))
    gi = int(round(g2 * 255))
    bi = int(round(b2 * 255))
    return f"#{ri:02x}{gi:02x}{bi:02x}"
