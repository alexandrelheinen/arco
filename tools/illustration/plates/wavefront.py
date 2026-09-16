"""Plate 2 — The Wavefront.

A* does not draw a line across a map; it floods it.  This plate shows the
flood: every cell the search popped from the frontier, tinted by when it
was popped, with equal-effort contours through the cloud and the returned
route cutting across it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .. import ramps
from ..canvas import Canvas
from ..solution import Solution
from ..stage import world_stage
from ..theme import Theme
from ..world import World

TITLE = "The wavefront"
SUBTITLE = (
    "A* floods the grid before it commits to anything. Tint is expansion "
    "order, contours are equal-effort frontiers, the thread is the route."
)


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
) -> None:
    """Render the A* expansion plate.

    Args:
        theme: Visual configuration.
        world: Scene being searched.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
    """
    run = solution.astar()
    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    ax = world_stage(canvas, world, grid_spacing=10.0, grid_alpha=0.5)

    order, alpha = _expansion_field(world, run.expanded)
    cmap = ramps.wavefront(theme)
    extent = (0.0, world.width, 0.0, world.height)
    rgba = cmap(order)
    rgba[..., 3] = alpha * 0.90
    ax.imshow(
        np.transpose(rgba, (1, 0, 2)),
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        zorder=2.2,
    )
    contoured = np.where(alpha > 0.45, gaussian_filter(order, 1.4), np.nan)
    ax.contour(
        np.linspace(0.0, world.width, order.shape[0]),
        np.linspace(0.0, world.height, order.shape[1]),
        contoured.T,
        levels=np.linspace(0.06, 0.94, 14),
        cmap=cmap,
        linewidths=0.8 * canvas.scale,
        alpha=0.55,
        zorder=2.4,
    )

    canvas.glow(ax, run.path, theme.astar, width=2.6, zorder=14)
    canvas.terminal(ax, world.start, theme.ice, "start", radius=1.9)
    canvas.terminal(ax, world.goal, theme.accent, "goal", radius=1.9)

    share = 100.0 * len(run.expanded) / max(run.cell_count, 1)
    canvas.chrome(
        TITLE,
        SUBTITLE,
        metrics=(
            f"{run.cell_count:>6d}  free cells",
            f"{len(run.expanded):>6d}  expanded  ({share:.0f} %)",
            f"{run.length:>6.1f}  m  route",
            f"{run.seconds * 1000.0:>6.0f}  ms  search",
        ),
        legend=(
            ("expansion order · early", theme.astar),
            ("A* route", theme.astar),
        ),
    )
    canvas.save(output)


def _expansion_field(world: World, expanded: np.ndarray):
    """Return the normalised expansion-order field and its soft mask.

    The raw field is defined only on expanded cells, which gives a
    staircase silhouette.  Filling it by nearest neighbour and feathering
    the coverage mask turns the same data into a continuous tint without
    inventing any expansion that did not happen.

    Args:
        world: Scene supplying the grid geometry.
        expanded: ``(K, 2)`` expanded cell centres in expansion order.

    Returns:
        ``(order, alpha)``, both grid-shaped float arrays in ``[0, 1]``.
    """
    field = np.full(world.grid.shape, np.nan)
    size = float(world.grid.cell_size)
    indices = np.rint(expanded / size).astype(int)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 1] >= 0)
        & (indices[:, 0] < world.grid.shape[0])
        & (indices[:, 1] < world.grid.shape[1])
    )
    for order, (i, j) in enumerate(indices[valid]):
        field[i, j] = float(order)

    mask = ~np.isnan(field)
    nearest = distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    filled = field[tuple(nearest)]
    span = float(np.nanmax(filled) - np.nanmin(filled)) or 1.0
    normalised = (filled - np.nanmin(filled)) / span
    alpha = np.clip(gaussian_filter(mask.astype(float), 1.7), 0.0, 1.0)
    return normalised, alpha**1.15
