"""Matplotlib viewer for road networks, routes, and vehicle trajectories."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arco.mapping.graph.road import RoadGraph


def draw_road_network(
    graph: RoadGraph,
    route: Optional[Sequence[int]] = None,
    smooth_path: Optional[Sequence[Tuple[float, float]]] = None,
    trajectory: Optional[Sequence[Tuple[float, float, float]]] = None,
    tracking_target: Optional[Tuple[float, float]] = None,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> Tuple[Figure, Axes]:
    """Draw a RoadGraph with optional route, smooth path, and trajectory overlays.

    Renders each road segment as a polyline through its geometry waypoints so
    that curved roads appear as curves rather than straight lines.  Optional
    overlays show the planned discrete route, the interpolated smooth path,
    and the recorded vehicle trajectory from a tracking-loop simulation.

    Args:
        graph: Road graph to visualize.
        route: Optional sequence of node IDs representing the planned discrete
            route.  Route edges are highlighted in a distinct color.
        smooth_path: Optional sequence of ``(x, y)`` points for the
            interpolated smooth path extracted from edge geometry.
        trajectory: Optional sequence of ``(x, y, heading)`` poses recording
            the vehicle trajectory from a tracking simulation.
        tracking_target: Optional ``(x, y)`` coordinates of the current
            lookahead / tracking target to highlight on the plot.
        ax: Existing :class:`~matplotlib.axes.Axes` to draw on.  A new figure
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
        if fig is None:
            raise ValueError("The provided Axes must be attached to a Figure.")

    # Build set of route edges for fast lookup
    route_edge_set: set[tuple[int, int]] = set()
    if route:
        route_edge_set = {
            (min(a, b), max(a, b)) for a, b in zip(route[:-1], route[1:])
        }

    # --- Draw road edges as polylines through their waypoints ---
    for node_a, node_b, _ in graph.edges:
        pts = graph.full_edge_geometry(node_a, node_b)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        key = (min(node_a, node_b), max(node_a, node_b))
        if key in route_edge_set:
            ax.plot(xs, ys, color="tomato", linewidth=2.5, zorder=3)
        else:
            ax.plot(xs, ys, color="silver", linewidth=1.2, zorder=2)

    # --- Draw intersection nodes ---
    for node_id in graph.nodes:
        x, y = graph.position(node_id)
        if route and node_id == route[0]:
            color = "limegreen"
            size = 120
            zorder = 6
        elif route and node_id == route[-1]:
            color = "royalblue"
            size = 120
            zorder = 6
        elif route and node_id in set(route):
            color = "tomato"
            size = 80
            zorder = 5
        else:
            color = "lightsteelblue"
            size = 60
            zorder = 4
        ax.scatter(x, y, color=color, s=size, zorder=zorder)

    # --- Draw smooth path ---
    if smooth_path and len(smooth_path) >= 2:
        spx = [p[0] for p in smooth_path]
        spy = [p[1] for p in smooth_path]
        ax.plot(
            spx,
            spy,
            color="darkorange",
            linewidth=1.8,
            linestyle="--",
            zorder=7,
            label="Smooth path",
        )

    # --- Draw vehicle trajectory ---
    if trajectory and len(trajectory) >= 2:
        tx = [p[0] for p in trajectory]
        ty = [p[1] for p in trajectory]
        ax.plot(
            tx,
            ty,
            color="dodgerblue",
            linewidth=1.5,
            zorder=8,
            label="Trajectory",
        )
        # Arrow at the final position showing heading
        last = trajectory[-1]
        arrow_len = max(
            3.0,
            0.02
            * math.hypot(
                max(tx) - min(tx) if len(tx) > 1 else 1.0,
                max(ty) - min(ty) if len(ty) > 1 else 1.0,
            ),
        )
        ax.annotate(
            "",
            xy=(
                last[0] + arrow_len * math.cos(last[2]),
                last[1] + arrow_len * math.sin(last[2]),
            ),
            xytext=(last[0], last[1]),
            arrowprops=dict(arrowstyle="-|>", color="dodgerblue", lw=2.0),
            zorder=9,
        )

    # --- Draw tracking target ---
    if tracking_target is not None:
        ax.scatter(
            tracking_target[0],
            tracking_target[1],
            color="gold",
            s=150,
            marker="*",
            zorder=10,
            label="Tracking target",
        )

    # --- Legend ---
    legend_handles = [
        plt.Line2D([0], [0], color="silver", linewidth=1.5, label="Road"),
        plt.Line2D([0], [0], color="tomato", linewidth=2.5, label="Route"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="limegreen",
            markersize=8,
            label="Start",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="royalblue",
            markersize=8,
            label="Goal",
        ),
    ]
    if smooth_path:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="darkorange",
                linewidth=1.8,
                linestyle="--",
                label="Smooth path",
            )
        )
    if trajectory:
        legend_handles.append(
            plt.Line2D(
                [0], [0], color="dodgerblue", linewidth=1.5, label="Trajectory"
            )
        )
    if tracking_target is not None:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gold",
                markersize=10,
                label="Tracking target",
            )
        )

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    return fig, ax
