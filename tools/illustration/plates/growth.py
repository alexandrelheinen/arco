"""Plate 3 — Anatomy of a tree.

Asymptotic optimality is a claim about the limit, and a single frame
cannot show a limit.  Three sample budgets of the same seeded RRT* run
can: the tree thickens, and the route it returns gets shorter.  The band
underneath plots that shortening from the same three runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import ramps
from ..canvas import Canvas
from ..solution import Solution, cost_to_come
from ..stage import world_bounds
from ..theme import Theme, text_font
from ..world import World

TITLE = "Anatomy of a tree"
SUBTITLE = (
    "One seed, three sample budgets. RRT* keeps rewiring what it already "
    "found, so the route it returns keeps getting shorter."
)
BUDGETS = (1200, 3000, 8000)

PANEL_Y = 0.455
PANEL_SIZE = 0.295
PANEL_X = (0.043, 0.3525, 0.662)


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
) -> None:
    """Render the growth plate.

    Args:
        theme: Visual configuration.
        world: Scene being planned in.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
    """
    runs = [solution.rrt(budget) for budget in BUDGETS]
    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    bounds = world_bounds(world)

    for run, left in zip(runs, PANEL_X):
        rect = (left, PANEL_Y, PANEL_SIZE, PANEL_SIZE)
        ax = canvas.stage(bounds, rect=rect, zorder=2)
        canvas.blobs(ax, world.outlines)
        canvas.tree(
            ax,
            run.nodes,
            run.parents,
            color=theme.rrt,
            depth=cost_to_come(run.nodes, run.parents),
            cmap=ramps.cost(theme),
            width=0.8,
            zorder=5,
        )
        if run.path is not None:
            canvas.glow(ax, run.path, theme.accent, width=1.7, zorder=12)
        canvas.terminal(ax, world.start, theme.ice, radius=1.6)
        canvas.terminal(ax, world.goal, theme.accent, radius=1.6)
        summary = (
            "no route yet" if run.path is None else f"route {run.length:.0f} m"
        )
        _caption(
            canvas,
            rect,
            f"{run.sample_count:,} samples".replace(",", " "),
            f"{len(run.nodes)} nodes · {summary}",
        )

    _convergence(canvas, runs)

    solved = [run for run in runs if run.path is not None]
    best = min(run.length for run in solved)
    worst = max(run.length for run in solved)
    canvas.chrome(
        TITLE,
        SUBTITLE,
        metrics=(
            f"step size       {3.4:>5.1f}  m",
            f"goal tolerance  {3.0:>5.1f}  m",
            f"seed            {7:>5d}",
            f"route  {worst:>5.0f} m → {best:.0f} m "
            f"({100 * (1 - best / worst):.0f} % shorter)",
        ),
        legend=(
            ("RRT* tree · cost-to-come", theme.rrt),
            ("returned route", theme.accent),
        ),
    )
    canvas.save(output)


def _convergence(canvas: Canvas, runs) -> None:
    """Plot returned route length against sample budget.

    Args:
        canvas: Canvas being drawn on.
        runs: The three :class:`~illustration.solution.TreeRun` results.
    """
    theme = canvas.theme
    solved = [run for run in runs if run.path is not None]
    ax = canvas.strip(
        (0.043, 0.185, 0.600, 0.150), label="route length · metres"
    )
    x = np.asarray([run.sample_count for run in solved], dtype=float)
    y = np.asarray([run.length for run in solved], dtype=float)
    ax.set_xlim(-float(x.max()) * 0.04, float(x.max()) * 1.10)
    ax.set_ylim(float(y.min()) * 0.975, float(y.max()) * 1.025)
    ax.fill_between(
        x,
        float(y.min()) * 0.985,
        y,
        color=theme.accent,
        alpha=0.08,
        linewidth=0.0,
    )
    canvas.glow(ax, np.column_stack([x, y]), theme.accent, width=2.0, zorder=6)
    for sample, length in zip(x, y):
        ax.scatter(
            [sample],
            [length],
            s=30.0 * canvas.scale**2,
            facecolor=theme.background,
            edgecolor=theme.accent,
            linewidths=1.6 * canvas.scale,
            zorder=8,
        )
        ax.text(
            sample,
            length,
            f"{length:.0f} m",
            color=theme.text,
            fontproperties=text_font(canvas.pt(theme.caption_size * 0.95)),
            va="bottom",
            ha="center",
            zorder=9,
        )
        ax.annotate(
            f"{int(sample):,} samples".replace(",", " "),
            xy=(sample, length),
            xytext=(0.0, -14.0 * canvas.scale),
            textcoords="offset points",
            color=theme.text_dim,
            fontproperties=text_font(canvas.pt(theme.caption_size * 0.85)),
            va="top",
            ha="center",
            zorder=9,
        )


def _caption(
    canvas: Canvas,
    rect: tuple,
    heading: str,
    detail: str,
) -> None:
    """Write a two-line caption under one panel.

    Args:
        canvas: Canvas being drawn on.
        rect: Panel rectangle in figure fractions.
        heading: Bright first line.
        detail: Dim second line.
    """
    theme = canvas.theme
    canvas.fig.text(
        rect[0],
        rect[1] - 0.034,
        heading,
        color=theme.text,
        fontproperties=text_font(canvas.pt(theme.caption_size * 1.2)),
        va="center",
        ha="left",
        zorder=40,
    )
    canvas.fig.text(
        rect[0] + rect[2],
        rect[1] - 0.034,
        detail,
        color=theme.text_dim,
        fontproperties=text_font(canvas.pt(theme.caption_size)),
        va="center",
        ha="right",
        zorder=40,
    )
