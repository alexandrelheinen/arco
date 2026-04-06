"""Matplotlib viewer for continuous occupancy maps and sampling planners."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arco.mapping.kdtree import KDTreeOccupancy


def draw_occupancy(
    occupancy: KDTreeOccupancy,
    bounds: Sequence[Tuple[float, float]],
    path: Optional[Sequence[np.ndarray]] = None,
    tree_nodes: Optional[Sequence[np.ndarray]] = None,
    tree_parent: Optional[Dict[int, Optional[int]]] = None,
    *,
    start: Optional[np.ndarray] = None,
    goal: Optional[np.ndarray] = None,
    resolution: int = 200,
    draw_tree: bool = False,
    obstacle_color: str = "dimgray",
    free_color: str = "white",
    path_color: str = "tomato",
    path_alpha: float = 1.0,
    tree_color: str = "lightblue",
    start_color: str = "limegreen",
    goal_color: str = "royalblue",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Figure, Axes]:
    """Draw a :class:`~arco.mapping.KDTreeOccupancy` map with an optional path.

    Renders a heatmap of the occupancy distance field (darker = closer to an
    obstacle) over the given *bounds* rectangle.  Optionally overlays the
    exploration tree and solution path.

    Args:
        occupancy: The KDTree occupancy map to visualize.
        bounds: Sampling bounds as ``[(x_min, x_max), (y_min, y_max)]``.
            Only 2-D bounds are supported.
        path: Optional ordered sequence of 2-D waypoints for the solution
            path.
        tree_nodes: Optional list of 2-D tree node positions (for tree
            rendering).
        tree_parent: Optional parent index map ``{node_idx: parent_idx}``
            (for tree rendering).
        start: Optional start position highlight.
        goal: Optional goal position highlight.
        resolution: Number of grid cells along each axis for the distance
            heatmap background.
        draw_tree: When ``True``, render tree edges between *tree_nodes*
            and their parents.  Off by default.
        obstacle_color: Color used for obstacle points.
        free_color: Background color for free (far) regions.
        path_color: Color for the solution path.
        path_alpha: Opacity of the solution path line (``0.0``–1.0``).
            Values below ``1.0`` are useful when an optimized trajectory
            is drawn on top of the raw reference path.
        tree_color: Color for tree edges.
        start_color: Color for the start marker.
        goal_color: Color for the goal marker.
        ax: Existing axes to draw on.
        title: Optional figure title.
        figsize: Figure size when creating a new figure.

    Returns:
        ``(fig, ax)`` tuple.

    Raises:
        ValueError: If *bounds* is not 2-D.
    """
    if len(bounds) != 2:
        raise ValueError(
            f"draw_occupancy only supports 2-D bounds; got {len(bounds)}-D."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # --- Distance heatmap background ------------------------------------
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    distances = occupancy.query_distances(grid_pts)
    dist_img = distances.reshape(resolution, resolution)

    ax.imshow(
        dist_img,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="Greys_r",
        vmin=0,
        vmax=float(np.percentile(distances, 80)),
        aspect="auto",
        alpha=0.5,
    )

    # --- Obstacle points -------------------------------------------------
    pts = occupancy.points
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=obstacle_color,
        s=15,
        zorder=3,
        label="Obstacles",
    )

    # --- Tree edges ------------------------------------------------------
    if draw_tree and tree_nodes is not None and tree_parent is not None:
        for idx, p_idx in tree_parent.items():
            if p_idx is None:
                continue
            n = tree_nodes[idx]
            p = tree_nodes[p_idx]
            ax.plot(
                [p[0], n[0]],
                [p[1], n[1]],
                color=tree_color,
                linewidth=0.5,
                zorder=2,
            )

    # --- Solution path ---------------------------------------------------
    if path is not None and len(path) >= 2:
        path_arr = np.array(path)
        ax.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            color=path_color,
            linewidth=2.0,
            alpha=path_alpha,
            zorder=4,
            label="Path",
        )

    # --- Start / Goal markers -------------------------------------------
    if start is not None:
        ax.plot(
            start[0],
            start[1],
            "o",
            color=start_color,
            markersize=10,
            zorder=5,
            label="Start",
        )
    if goal is not None:
        ax.plot(
            goal[0],
            goal[1],
            "*",
            color=goal_color,
            markersize=14,
            zorder=5,
            label="Goal",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 1, 0.6, 1))

    return fig, ax
