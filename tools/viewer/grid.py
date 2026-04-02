"""Matplotlib viewer for :class:`~arco.mapping.grid.base.Grid`."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from arco.mapping.grid.base import Grid


def draw_grid(
    grid: Grid,
    path: Optional[Sequence[Tuple[int, ...]]] = None,
    *,
    cell_colors: Optional[Dict[Tuple[int, ...], str]] = None,
    free_color: str = "white",
    obstacle_color: str = "dimgray",
    path_color: str = "tomato",
    start_color: str = "limegreen",
    goal_color: str = "royalblue",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Figure, Axes]:
    """Draw a 2-D :class:`~arco.mapping.grid.base.Grid` with an optional A* path.

    The planner can supply per-cell colour overrides through *cell_colors* to
    highlight explored or frontier cells in any colour it chooses.

    Only 2-D grids are supported; an error is raised for other dimensionalities.

    Args:
        grid: The 2-D grid to visualise.
        path: Optional sequence of ``(row, col)`` tuples representing the
            planned path.
        cell_colors: Optional per-cell colour override mapping
            ``{(row, col): colour}``.
        free_color: Colour for free (unoccupied) cells.
        obstacle_color: Colour for occupied cells.
        path_color: Colour applied to intermediate path cells.
        start_color: Colour for the first cell in *path* (start).
        goal_color: Colour for the last cell in *path* (goal).
        ax: Existing :class:`matplotlib.axes.Axes` to draw on.  A new figure
            is created when *None*.
        title: Optional plot title.
        figsize: Figure size passed to :func:`matplotlib.pyplot.subplots` when
            *ax* is *None*.

    Returns:
        ``(fig, ax)`` tuple.

    Raises:
        ValueError: If *grid* is not 2-D.
    """
    if len(grid.shape) != 2:
        raise ValueError(f"draw_grid only supports 2-D grids; got shape {grid.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    rows, cols = grid.shape

    # Build an RGBA image: white=free, dark=occupied.
    img = np.ones((rows, cols, 4), dtype=float)
    free_rgba = np.array(plt.matplotlib.colors.to_rgba(free_color))
    obs_rgba = np.array(plt.matplotlib.colors.to_rgba(obstacle_color))
    img[grid.data == 0] = free_rgba
    img[grid.data == 1] = obs_rgba

    # Apply optional per-cell color overrides.
    if cell_colors:
        for (r, c), color in cell_colors.items():
            img[r, c] = np.array(plt.matplotlib.colors.to_rgba(color))

    # Apply path colors.
    if path:
        path_rgba = np.array(plt.matplotlib.colors.to_rgba(path_color))
        start_rgba = np.array(plt.matplotlib.colors.to_rgba(start_color))
        goal_rgba = np.array(plt.matplotlib.colors.to_rgba(goal_color))
        for i, (r, c) in enumerate(path):
            if i == 0:
                img[r, c] = start_rgba
            elif i == len(path) - 1:
                img[r, c] = goal_rgba
            else:
                img[r, c] = path_rgba

    # Draw with origin='lower' so that (0,0) is at the bottom-left.
    ax.imshow(img, origin="lower", interpolation="nearest", extent=[0, cols, 0, rows])

    if title:
        ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=free_color, label="Free"),
        plt.Rectangle((0, 0), 1, 1, color=obstacle_color, label="Obstacle"),
    ]
    if path:
        legend_handles += [
            plt.Rectangle((0, 0), 1, 1, color=start_color, label="Start"),
            plt.Rectangle((0, 0), 1, 1, color=goal_color, label="Goal"),
            plt.Rectangle((0, 0), 1, 1, color=path_color, label="Path"),
        ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    return fig, ax
