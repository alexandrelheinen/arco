"""Matplotlib viewer for :class:`~arco.mapping.graph.WeightedGraph`."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arco.mapping.graph import WeightedGraph


def draw_graph(
    graph: WeightedGraph,
    path: Optional[Sequence[int]] = None,
    *,
    node_colors: Optional[Dict[int, str]] = None,
    edge_colors: Optional[Dict[Tuple[int, int], str]] = None,
    default_node_color: str = "lightsteelblue",
    default_edge_color: str = "silver",
    path_node_color: str = "tomato",
    path_edge_color: str = "tomato",
    start_color: str = "limegreen",
    goal_color: str = "royalblue",
    node_size: int = 120,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Figure, Axes]:
    """Draw a :class:`~arco.mapping.graph.WeightedGraph` with an optional A* path.

    The planner can supply per-node and per-edge colour overrides through the
    *node_colors* and *edge_colors* mappings so that it can highlight explored
    nodes, frontier nodes, etc. in any colour it chooses.

    Args:
        graph: The graph to visualise.
        path: Optional sequence of node IDs representing the planned path.
            Path nodes and edges are coloured with *path_node_color* and
            *path_edge_color* unless overridden by *node_colors* /
            *edge_colors*.
        node_colors: Optional per-node colour override mapping
            ``{node_id: colour}``.  Colours use any format accepted by
            matplotlib.
        edge_colors: Optional per-edge colour override mapping
            ``{(node_a, node_b): colour}``.  Keys are order-independent.
        default_node_color: Colour for nodes not in the path or *node_colors*.
        default_edge_color: Colour for edges not in the path or *edge_colors*.
        path_node_color: Colour applied to intermediate path nodes.
        path_edge_color: Colour applied to path edges.
        start_color: Colour for the first node in *path* (start).
        goal_color: Colour for the last node in *path* (goal).
        node_size: Scatter point size used for nodes.
        ax: Existing :class:`matplotlib.axes.Axes` to draw on.  A new figure
            is created when *None*.
        title: Optional plot title.
        figsize: Figure size passed to :func:`matplotlib.pyplot.subplots` when
            *ax* is *None*.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    node_colors = dict(node_colors or {})
    edge_colors = dict(edge_colors or {})

    # Build path edge and node sets for easy lookup.
    path_node_set: set = set()
    path_edge_set: set = set()
    if path:
        path_node_set = set(path[1:-1])  # intermediate nodes (exclude start/goal)
        path_edge_set = {
            (min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:])
        }

    # --- Draw edges ---
    for node_a, node_b, _ in graph.edges:
        xa, ya = graph.position(node_a)
        xb, yb = graph.position(node_b)
        key = (min(node_a, node_b), max(node_a, node_b))
        if key in edge_colors:
            color = edge_colors[key]
            lw = 2.0
            zorder = 3
        elif key in path_edge_set:
            color = path_edge_color
            lw = 2.5
            zorder = 3
        else:
            color = default_edge_color
            lw = 1.0
            zorder = 2
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, zorder=zorder)

    # --- Draw nodes ---
    for node_id in graph.nodes:
        x, y = graph.position(node_id)
        if node_id in node_colors:
            color = node_colors[node_id]
        elif path and node_id == path[0]:
            color = start_color
        elif path and node_id == path[-1]:
            color = goal_color
        elif node_id in path_node_set:
            color = path_node_color
        else:
            color = default_node_color
        ax.scatter(x, y, color=color, s=node_size, zorder=4)
        ax.annotate(str(node_id), (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, color="dimgray", zorder=5)

    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=default_node_color,
                   markersize=8, label="Node"),
        plt.Line2D([0], [0], color=default_edge_color, linewidth=1.5, label="Edge"),
    ]
    if path:
        legend_handles += [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=start_color,
                       markersize=8, label="Start"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=goal_color,
                       markersize=8, label="Goal"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=path_node_color,
                       markersize=8, label="Path node"),
            plt.Line2D([0], [0], color=path_edge_color, linewidth=2.5, label="Path"),
        ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    return fig, ax
