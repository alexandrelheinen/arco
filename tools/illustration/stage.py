"""Stage helpers: the world panel every plate is built on.

Keeping the obstacle bodies, the blueprint grid and the framing in one
place is what makes seven plates look like one series rather than seven
scripts.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.axes import Axes

from .canvas import Bounds, Canvas
from .world import World


def world_bounds(world: World, pad: float = 0.0) -> Bounds:
    """Return the full scene window, optionally padded.

    Args:
        world: Scene to frame.
        pad: Growth per side as a fraction of the scene size.

    Returns:
        The framing window.
    """
    return Bounds(0.0, 0.0, world.width, world.height).padded(pad)


def world_stage(
    canvas: Canvas,
    world: World,
    rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    bounds: Bounds = None,
    grid_spacing: float = 10.0,
    grid_alpha: float = 1.0,
    zorder: int = 2,
) -> Axes:
    """Add a world panel with the blueprint grid and obstacle bodies drawn.

    Args:
        canvas: Canvas to draw on.
        world: Scene supplying the obstacle outlines.
        rect: Figure-fraction rectangle of the panel.
        bounds: Window to show; defaults to the whole scene.
        grid_spacing: Blueprint grid pitch in world units, 0 to disable.
        grid_alpha: Multiplier on the blueprint grid opacity.
        zorder: Stacking order of the panel.

    Returns:
        The prepared axes, ready for planner output.
    """
    view = world_bounds(world) if bounds is None else bounds
    ax = canvas.stage(view, rect=rect, zorder=zorder)
    if grid_spacing > 0.0:
        canvas.blueprint(ax, grid_spacing, alpha=grid_alpha)
    canvas.blobs(ax, world.outlines)
    return ax


def corridor_bounds(points: np.ndarray, pad: float = 0.12) -> Bounds:
    """Return a window framing a polyline with a margin.

    Args:
        points: ``(N, 2)`` polyline in world units.
        pad: Margin per side as a fraction of the polyline extent.

    Returns:
        The framing window.
    """
    low = points.min(axis=0)
    high = points.max(axis=0)
    return Bounds(low[0], low[1], high[0], high[1]).padded(pad)
