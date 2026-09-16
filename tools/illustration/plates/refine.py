"""Plate 5 — From path to trajectory.

A planner returns a sequence of free waypoints, not something a vehicle
can drive.  Three passes stand between the two: the pruner throws away
the hops a straight line can replace, densification seeds the optimiser,
and the optimiser trades deviation against time under a clearance
barrier.  The speed profile below the frame is the optimiser's answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import ramps
from ..canvas import Canvas
from ..solution import Solution
from ..stage import corridor_bounds, world_stage
from ..theme import Theme
from ..world import World

TITLE = "From path to trajectory"
SUBTITLE = (
    "A 1 200-sample RRT* path, the shortcut chain the pruner leaves, and "
    "the optimised curve coloured by the speed its segment times imply."
)
SAMPLES = 1200


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
) -> None:
    """Render the refinement plate.

    Args:
        theme: Visual configuration.
        world: Scene being planned in.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
    """
    refine = solution.refinement(SAMPLES)
    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    bounds = corridor_bounds(_focus(refine.dense), pad=0.20)
    ax = world_stage(
        canvas,
        world,
        rect=(0.0, 0.215, 1.0, 0.785),
        bounds=bounds,
        grid_spacing=10.0,
        grid_alpha=0.6,
    )

    canvas.glow(
        ax,
        refine.raw,
        theme.rrt,
        width=1.5,
        zorder=10,
        ramp=theme.glow_soft,
        core_alpha=0.45,
    )
    ax.scatter(
        refine.raw[:, 0],
        refine.raw[:, 1],
        s=9.0 * canvas.scale**2,
        facecolor=theme.rrt,
        alpha=0.55,
        linewidths=0.0,
        zorder=11,
    )
    canvas.glow(
        ax,
        refine.pruned,
        theme.sst,
        width=2.0,
        zorder=12,
        ramp=theme.glow_soft,
        core_alpha=0.85,
    )
    ax.scatter(
        refine.pruned[:, 0],
        refine.pruned[:, 1],
        s=26.0 * canvas.scale**2,
        facecolor="none",
        edgecolor=theme.sst,
        linewidths=1.2 * canvas.scale,
        zorder=13,
    )
    canvas.ribbon(
        ax,
        refine.dense,
        refine.dense_speed,
        ramps.speed(theme),
        width=5.2,
        zorder=16,
        halo=theme.accent,
    )
    canvas.terminal(ax, world.start, theme.ice, "start", radius=1.6)
    canvas.terminal(ax, world.goal, theme.accent, "goal", radius=1.6)

    _speed_strip(canvas, refine)

    canvas.chrome(
        TITLE,
        SUBTITLE,
        metrics=(
            f"raw        {len(refine.raw):>3d} pts  "
            f"{_length(refine.raw):>6.1f} m",
            f"pruned     {len(refine.pruned):>3d} pts  "
            f"{_length(refine.pruned):>6.1f} m",
            f"optimised  {len(refine.states):>3d} pts  "
            f"{_length(refine.dense):>6.1f} m",
            f"travel time     {float(refine.durations.sum()):>7.1f} s",
        ),
        legend=(
            ("raw RRT* path", theme.rrt),
            ("pruned shortcuts", theme.sst),
            ("optimised · speed", theme.accent),
        ),
    )
    canvas.save(output)


def _speed_strip(canvas: Canvas, refine) -> None:
    """Draw the speed-versus-distance band under the corridor.

    Args:
        canvas: Canvas being drawn on.
        refine: Refinement run supplying the dense speed profile.
    """
    theme = canvas.theme
    ax = canvas.strip(
        (0.043, 0.105, 0.600, 0.100), label="speed profile · m/s"
    )
    arc = np.concatenate(
        [
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(refine.dense, axis=0), axis=1)),
        ]
    )
    speed = refine.dense_speed
    low = float(speed.min())
    high = float(speed.max())
    margin = max((high - low) * 0.45, 0.4)
    ax.set_xlim(float(arc[0]), float(arc[-1]))
    ax.set_ylim(low - margin, high + margin)
    ax.fill_between(
        arc,
        low - margin,
        speed,
        color=theme.accent,
        alpha=0.10,
        linewidth=0.0,
    )
    canvas.ribbon(
        ax,
        np.column_stack([arc, speed]),
        speed,
        ramps.speed(theme),
        width=2.4,
        zorder=6,
    )
    for value in (low, high):
        ax.axhline(
            value,
            color=theme.rule,
            linewidth=0.8 * canvas.scale,
            alpha=0.7,
            zorder=2,
        )
        ax.text(
            float(arc[-1]),
            value,
            f" {value:.1f}",
            color=theme.text_dim,
            fontsize=canvas.pt(theme.caption_size * 0.9),
            va="center",
            ha="left",
            zorder=7,
        )


def _focus(dense: np.ndarray, window: float = 46.0) -> np.ndarray:
    """Return the stretch of trajectory around its sharpest bend.

    Shown whole, the three refinement stages sit on top of each other and
    the plate says nothing.  Framing the tightest corner is where prune
    and optimise visibly disagree with the raw path.

    Args:
        dense: ``(N, 2)`` optimised trajectory.
        window: Half-width of the framed arc in world units.

    Returns:
        The framed slice of *dense*.
    """
    step = np.diff(dense, axis=0)
    heading = np.arctan2(step[:, 1], step[:, 0])
    turn = np.abs(np.diff(np.unwrap(heading)))
    smooth = np.convolve(turn, np.ones(24) / 24.0, mode="same")
    centre = int(np.argmax(smooth)) + 1
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(step, axis=1))])
    inside = np.abs(arc - arc[centre]) <= window
    return dense[inside]


def _length(points: np.ndarray) -> float:
    """Return the arc length of a polyline.

    Args:
        points: ``(N, 2)`` polyline.

    Returns:
        Arc length in world units.
    """
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
