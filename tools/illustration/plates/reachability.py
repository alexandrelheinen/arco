"""Plate 7 — What the model allows.

Every curve here is a forward integration of the shipped Dubins vehicle
model under a turn-rate ramp, so the fan carries the model's real
acceleration and turn-rate-dot limits.  The bright lobes stay clear of
the occupancy map; the dim ones are what the map removes from the menu.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

from ..canvas import Bounds, Canvas
from ..solution import Solution
from ..stage import world_stage
from ..theme import Theme
from ..world import World

TITLE = "What the model allows"
SUBTITLE = (
    "{count} turn-rate ramps integrated through the Dubins model from one "
    "pose. Bright lobes stay clear of the map; dim ones do not."
)


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
) -> None:
    """Render the reachability plate.

    Args:
        theme: Visual configuration.
        world: Scene the fan is clipped by.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
    """
    fan = solution.reachability()
    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    points = np.vstack(fan.curves)
    bounds = Bounds(
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    ).padded(0.10)
    span = max(bounds.width, bounds.height)
    ax = world_stage(
        canvas, world, bounds=bounds, grid_spacing=5.0, grid_alpha=0.7
    )

    blocked = [c for c, free in zip(fan.curves, fan.free) if not free]
    ax.add_collection(
        LineCollection(
            blocked,
            colors=[theme.obstacle_edge],
            linewidths=0.7 * canvas.scale,
            alpha=0.22,
            zorder=8,
        )
    )
    free = [c for c, free in zip(fan.curves, fan.free) if free]
    for curve in free:
        canvas.glow(
            ax,
            curve,
            theme.ice,
            width=0.9,
            zorder=12,
            ramp=theme.glow_soft,
            core_alpha=0.55,
        )
    if free:
        tips = np.asarray([c[-1] for c in free])
        ax.scatter(
            tips[:, 0],
            tips[:, 1],
            s=8.0 * canvas.scale**2,
            facecolor=theme.accent,
            alpha=0.85,
            linewidths=0.0,
            zorder=14,
        )

    heading = np.array(
        [np.cos(fan.origin[2]), np.sin(fan.origin[2])], dtype=float
    )
    canvas.glow(
        ax,
        np.vstack([fan.origin[:2], fan.origin[:2] + heading * span * 0.16]),
        theme.accent,
        width=2.0,
        zorder=18,
    )
    canvas.terminal(ax, fan.origin[:2], theme.accent, "pose", radius=0.9)

    canvas.chrome(
        TITLE,
        SUBTITLE.format(count=len(fan.curves)),
        metrics=(
            f"rollouts        {len(fan.curves):>5d}",
            f"collision-free  {int(fan.free.sum()):>5d}",
            f"horizon         {fan.horizon:>5.1f} s",
            f"turn rate      ±{float(np.abs(fan.turn_rate).max()):>5.2f} rad/s",
        ),
        legend=(
            ("admissible rollout", theme.ice),
            ("removed by the map", theme.obstacle_edge),
        ),
    )
    canvas.save(output)
