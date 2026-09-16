"""Plate 4 — Three searches, one world.

The same start, the same goal, the same obstacle field, handed to a
discrete search and to two samplers.  A* floods and returns the shortest
grid route; RRT* spreads and rewires; SST keeps one representative per
witness cell and stays sparse.  Their answers do not agree, and that
disagreement is the argument for keeping all three in one library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..canvas import Canvas
from ..solution import Solution
from ..stage import world_stage
from ..theme import Theme
from ..world import World

TITLE = "Three searches, one world"
SUBTITLE = (
    "A* on the grid, RRT* and SST on the continuum — same endpoints, same "
    "obstacles, three different commitments."
)
SAMPLES = 8000


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
) -> None:
    """Render the comparison plate.

    Args:
        theme: Visual configuration.
        world: Scene being planned in.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
    """
    rrt = solution.rrt(SAMPLES)
    sst = solution.sst(SAMPLES)
    astar = solution.astar()

    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    ax = world_stage(canvas, world, grid_spacing=10.0, grid_alpha=0.6)

    ax.scatter(
        astar.expanded[:, 0],
        astar.expanded[:, 1],
        s=6.0 * canvas.scale**2,
        c=theme.astar,
        alpha=0.18,
        linewidths=0.0,
        zorder=4,
    )
    canvas.tree(
        ax,
        rrt.nodes,
        rrt.parents,
        color=theme.rrt,
        width=0.65,
        zorder=5,
        alpha_scale=0.62,
    )
    canvas.tree(
        ax,
        sst.nodes,
        sst.parents,
        color=theme.sst,
        width=0.8,
        zorder=6,
        alpha_scale=0.80,
    )

    for path, color, width in (
        (astar.path, theme.astar, 2.4),
        (sst.path, theme.sst, 2.4),
        (rrt.path, theme.rrt, 2.8),
    ):
        if path is not None:
            canvas.glow(ax, np.asarray(path), color, width=width, zorder=14)

    canvas.terminal(ax, world.start, theme.ice, "start", radius=1.9)
    canvas.terminal(ax, world.goal, theme.accent, "goal", radius=1.9)

    canvas.chrome(
        TITLE,
        SUBTITLE,
        metrics=(
            f"A*    {astar.length:>6.1f} m   "
            f"{len(astar.expanded):>5d} cells   "
            f"{astar.seconds:>5.2f} s",
            f"RRT*  {rrt.length:>6.1f} m   "
            f"{len(rrt.nodes):>5d} nodes   "
            f"{rrt.seconds:>5.1f} s",
            f"SST   {sst.length:>6.1f} m   "
            f"{len(sst.nodes):>5d} nodes   "
            f"{sst.seconds:>5.1f} s",
        ),
        legend=(
            ("A* · grid search", theme.astar),
            ("RRT* · asymptotically optimal", theme.rrt),
            ("SST · sparse witnesses", theme.sst),
        ),
    )
    canvas.save(output)
