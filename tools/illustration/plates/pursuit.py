"""Plate 6 — Closing the loop.

Planning ends where the vehicle starts disagreeing with the plan.  A
Dubins model is driven along the optimised trajectory by pure pursuit
with an artificial-potential-field bias; the executed line is coloured by
its signed lateral error, and the rays are the lookahead points the
controller actually aimed at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

from .. import ramps
from ..canvas import Canvas
from ..solution import Solution
from ..stage import corridor_bounds, world_stage
from ..theme import Theme
from ..world import World

TITLE = "Closing the loop"
SUBTITLE = (
    "Pure pursuit plus an APF bias drives a Dubins model along the plan. "
    "Colour is signed lateral error; the rays are its lookahead points."
)
SAMPLES = 8000
RAY_STRIDE = 14


def render(
    theme: Theme,
    world: World,
    solution: Solution,
    output: Path,
    width_in: float = 16.0,
    dpi: int = 240,
) -> None:
    """Render the tracking plate.

    Args:
        theme: Visual configuration.
        world: Scene being driven through.
        solution: Solver-output accessor.
        output: Destination PNG path.
        width_in: Figure width in inches.
        dpi: Output resolution.
    """
    run = solution.tracking(SAMPLES)
    canvas = Canvas(theme, width_in=width_in, dpi=dpi)
    bounds = corridor_bounds(run.reference, pad=0.16)
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
        run.reference,
        theme.text_dim,
        width=1.6,
        zorder=9,
        ramp=theme.glow_soft,
        core_alpha=0.60,
    )

    poses = run.poses
    rays = [
        [(poses[i, 0], poses[i, 1]), (run.carrots[i, 0], run.carrots[i, 1])]
        for i in range(0, len(poses), RAY_STRIDE)
    ]
    ax.add_collection(
        LineCollection(
            rays,
            colors=[theme.ice],
            linewidths=0.7 * canvas.scale,
            alpha=0.30,
            zorder=11,
        )
    )
    ax.scatter(
        run.carrots[::RAY_STRIDE, 0],
        run.carrots[::RAY_STRIDE, 1],
        s=6.0 * canvas.scale**2,
        facecolor=theme.ice,
        alpha=0.55,
        linewidths=0.0,
        zorder=12,
    )

    limit = float(np.abs(run.cross_track).max())
    signed = np.clip(run.cross_track / (limit if limit > 1e-9 else 1.0), -1, 1)
    canvas.ribbon(
        ax,
        poses[:, :2],
        signed,
        ramps.error(theme),
        width=4.4,
        zorder=16,
        halo=theme.accent,
    )
    _heading_ticks(canvas, ax, poses)
    canvas.terminal(ax, world.start, theme.ice, "start", radius=1.6)
    canvas.terminal(ax, world.goal, theme.accent, "goal", radius=1.6)

    _error_strip(canvas, run)

    canvas.chrome(
        TITLE,
        SUBTITLE,
        metrics=(
            f"lookahead        {12.0:>6.1f} m",
            f"cruise            {9.0:>6.1f} m/s",
            f"peak |lateral|   {limit:>6.2f} m",
            f"rms  |lateral|   "
            f"{float(np.sqrt((run.cross_track ** 2).mean())):>6.2f} m",
        ),
        legend=(
            ("reference trajectory", theme.text_dim),
            ("executed · lateral error", theme.accent),
            ("lookahead rays", theme.ice),
        ),
    )
    canvas.save(output)


def _heading_ticks(canvas: Canvas, ax, poses: np.ndarray) -> None:
    """Draw short heading ticks along the executed line.

    Args:
        canvas: Canvas being drawn on.
        ax: Stage axes.
        poses: ``(T, 3)`` executed poses.
    """
    stride = RAY_STRIDE * 2
    length = 1.6
    sample = poses[::stride]
    normal = np.column_stack([-np.sin(sample[:, 2]), np.cos(sample[:, 2])])
    segments = [
        [tuple(p[:2] - normal[i] * length), tuple(p[:2] + normal[i] * length)]
        for i, p in enumerate(sample)
    ]
    ax.add_collection(
        LineCollection(
            segments,
            colors=[canvas.theme.text],
            linewidths=0.8 * canvas.scale,
            alpha=0.35,
            zorder=17,
        )
    )


def _error_strip(canvas: Canvas, run) -> None:
    """Draw the lateral-error band under the corridor.

    Args:
        canvas: Canvas being drawn on.
        run: Tracking run supplying the error history.
    """
    theme = canvas.theme
    ax = canvas.strip(
        (0.043, 0.105, 0.600, 0.100), label="lateral error · metres"
    )
    time = np.arange(len(run.cross_track)) * run.dt
    limit = float(np.abs(run.cross_track).max()) * 1.25
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(-limit, limit)
    canvas.baseline(ax, 0.0)
    ax.fill_between(
        time,
        0.0,
        run.cross_track,
        color=theme.accent,
        alpha=0.16,
        linewidth=0.0,
    )
    canvas.ribbon(
        ax,
        np.column_stack([time, run.cross_track]),
        np.clip(run.cross_track / (limit or 1.0), -1.0, 1.0),
        ramps.error(theme),
        width=2.0,
        zorder=6,
    )
    ax.text(
        float(time[-1]),
        limit * 0.72,
        f"{float(time[-1]):.0f} s ",
        color=theme.text_dim,
        fontsize=canvas.pt(theme.caption_size * 0.9),
        va="center",
        ha="right",
        zorder=7,
    )
