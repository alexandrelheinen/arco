"""Matplotlib 3D viewer for :class:`~arco.mapping.grid.base.Grid`.

Provides visualization utilities for 3-D grids with obstacle rendering
using voxels and path rendering using line plots.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from arco.mapping.grid.base import Grid


def draw_grid_3d(
    grid: Grid,
    path: Optional[Sequence[Tuple[int, ...]]] = None,
    *,
    obstacle_color: str = "dimgray",
    obstacle_alpha: float = 0.3,
    path_color: str = "tomato",
    start_color: str = "limegreen",
    goal_color: str = "royalblue",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[Figure, Axes]:
    """Draw a 3-D :class:`~arco.mapping.grid.base.Grid` with an optional path.

    Renders obstacles as semi-transparent voxels and the planned path as a
    line with start and goal markers. Only 3-D grids are supported.

    Args:
        grid: The 3-D grid to visualise.
        path: Optional sequence of ``(x, y, z)`` tuples representing the
            planned path.
        obstacle_color: Colour for occupied voxels.
        obstacle_alpha: Transparency of obstacle voxels (0.0 to 1.0).
        path_color: Colour for the path line.
        start_color: Colour for the start marker.
        goal_color: Colour for the goal marker.
        ax: Existing 3D :class:`matplotlib.axes.Axes` to draw on. A new
            figure is created when *None*.
        title: Optional plot title.
        figsize: Figure size passed to :func:`matplotlib.pyplot.subplots`
            when *ax* is *None*.

    Returns:
        ``(fig, ax)`` tuple.

    Raises:
        ValueError: If *grid* is not 3-D.
    """
    if len(grid.shape) != 3:
        raise ValueError(
            f"draw_grid_3d only supports 3-D grids; got shape {grid.shape}"
        )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    nx, ny, nz = grid.shape

    # Render obstacles as voxels
    # grid.data is (nx, ny, nz) with 1=occupied, 0=free
    occupied = grid.data == 1
    if np.any(occupied):
        ax.voxels(
            occupied,
            facecolors=obstacle_color,
            edgecolors="k",
            alpha=obstacle_alpha,
            linewidth=0.3,
        )

    # Render path as a 3D line
    if path and len(path) > 0:
        # Convert cell indices to physical positions
        positions = np.array([grid.position(idx) for idx in path])
        xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

        # Draw path line
        ax.plot(xs, ys, zs, color=path_color, linewidth=2, label="Path")

        # Mark start and goal
        ax.scatter(
            [xs[0]],
            [ys[0]],
            [zs[0]],
            color=start_color,
            s=100,
            marker="o",
            label="Start",
            edgecolors="black",
            linewidths=1.5,
        )
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            [zs[-1]],
            color=goal_color,
            s=100,
            marker="o",
            label="Goal",
            edgecolors="black",
            linewidths=1.5,
        )

    # Set axis labels and limits
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Set limits based on grid physical size
    ax.set_xlim(0, grid.size_m[0])
    ax.set_ylim(0, grid.size_m[1])
    ax.set_zlim(0, grid.size_m[2])

    if title:
        ax.set_title(title)

    if path:
        ax.legend(loc="upper left", fontsize=8)

    return fig, ax
