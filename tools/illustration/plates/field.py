"""Plate 1 — Field of Possibility.

The gallery cover.  One dense RRT* tree fills the frame as filigree,
coloured by the cost-to-come it is rewired to minimise; the solution it
returns blooms through it; the optimised trajectory rides on top as a
speed ribbon.  Planning and guidance in one image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import ramps
from ..canvas import Canvas
from ..quiet import soften
from ..solution import Solution, cost_to_come
from ..stage import world_stage
from ..theme import Theme
from ..world import World

TITLE = "Field of possibility"
SUBTITLE = (
    "RRT* grows 8 000 samples across the basin, keeps the cheapest route "
    "to every node, and hands one curve to the optimiser."
)
SAMPLES = 8000


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
    quiet: bool = False,
) -> None:
    """Render the cover plate.

    Args:
        theme: Visual configuration.
        world: Scene being planned in.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
        quiet: Draw the solver picture only. Skips the poster chrome
            and the lamp under the speed ribbon.
    """
    if quiet:
        theme = soften(theme)
    run = solution.rrt(SAMPLES)
    refine = solution.refinement(SAMPLES)

    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    ax = world_stage(canvas, world, grid_spacing=10.0, grid_alpha=0.75)

    canvas.tree(
        ax,
        run.nodes,
        run.parents,
        color=theme.rrt,
        depth=cost_to_come(run.nodes, run.parents),
        cmap=ramps.cost(theme),
        width=1.0,
        zorder=5,
    )
    if run.path is not None:
        canvas.glow(
            ax,
            run.path,
            theme.rrt,
            width=1.7,
            zorder=11,
            core_alpha=0.55,
        )
    canvas.ribbon(
        ax,
        refine.dense,
        refine.dense_speed,
        ramps.speed(theme),
        width=4.6,
        zorder=14,
        halo=None if quiet else theme.accent,
    )
    canvas.terminal(
        ax, world.start, theme.ice, "" if quiet else "start", radius=1.9
    )
    canvas.terminal(
        ax, world.goal, theme.accent, "" if quiet else "goal", radius=1.9
    )

    if quiet:
        canvas.save(output)
        return

    fastest = float(refine.dense_speed.max())
    slowest = float(refine.dense_speed.min())
    canvas.chrome(
        TITLE,
        SUBTITLE,
        metrics=(
            f"{len(run.nodes):>6d}  tree nodes",
            f"{run.length:>6.1f}  m  raw RRT* path",
            f"{_length(refine.dense):>6.1f}  m  optimised trajectory",
            f"{slowest:>6.1f}–{fastest:.1f}  m/s  speed envelope",
        ),
        legend=(
            ("RRT* tree · cost-to-come", theme.rrt),
            ("RRT* solution", theme.rrt),
            ("optimised trajectory · speed", theme.accent),
        ),
    )
    canvas.save(output)


def _length(points: np.ndarray) -> float:
    """Return the arc length of a polyline.

    Args:
        points: ``(N, 2)`` polyline.

    Returns:
        Arc length in world units.
    """
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
