"""Shared rendering engine for workspace and C-space frames.

:class:`FrameRenderer` consumes a :class:`~arco.tools.viewer.scene_snapshot.
SceneSnapshot` and renders every visual layer (obstacles, exploration tree,
found path, pruned path, adjusted trajectory, executed trajectory) onto any
``matplotlib`` axes — whether 2-D or 3-D, workspace or C-space.

This **decouples** rendering from algorithm logic: the same
:meth:`FrameRenderer.render` call produces both the workspace and C-space
panels, which automatically stay in sync because they consume the same
snapshot.

Example
-------
>>> from arco.tools.viewer.scene_snapshot import SceneSnapshot
>>> from arco.tools.viewer.frame_renderer import FrameRenderer, LayerStyle
>>> snap = SceneSnapshot(...)
>>> renderer = FrameRenderer()
>>> renderer.render(ax_ws, snap)   # workspace panel
>>> renderer.render(ax_cs, snap)   # C-space panel — same data, same style
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from arco.config.palette import annotation_hex, layer_hex, obstacle_hex
from arco.tools.viewer.scene_snapshot import SceneSnapshot


@dataclasses.dataclass
class LayerStyle:
    """Visual style overrides for a single rendering layer.

    All fields are optional; ``None`` means *use the palette default*.

    Attributes:
        color: Override line/marker colour.
        linewidth: Override line width.
        alpha: Override opacity in ``[0, 1]``.
        markersize: Override scatter marker size.
        visible: When ``False`` this layer is skipped entirely.
    """

    color: str | None = None
    linewidth: float | None = None
    alpha: float | None = None
    markersize: float | None = None
    visible: bool = True


class FrameRenderer:
    """Renders :class:`~arco.tools.viewer.scene_snapshot.SceneSnapshot` layers.

    The same :class:`FrameRenderer` instance can be used to populate both
    the workspace axes and the C-space axes, ensuring that both panels
    always reflect exactly the same data.

    Args:
        draw_obstacles: When ``True`` (default) render the obstacle layer.
        draw_tree: When ``True`` (default) render the exploration tree.
        draw_found_path: When ``True`` (default) render the raw found path.
        draw_pruned_path: When ``True`` (default) render pruned waypoints.
        draw_trajectory: When ``True`` (default) render the adjusted
            (optimised) trajectory.
        draw_executed: When ``True`` (default) render the executed trace.
        draw_start_goal: When ``True`` (default) render start/goal markers.
        styles: Optional dict of ``layer_name → LayerStyle`` to override
            per-layer defaults.  Valid keys: ``"obstacle"``, ``"tree"``,
            ``"found_path"``, ``"pruned_path"``, ``"trajectory"``,
            ``"executed"``, ``"annotation"``.
        is_3d: When ``True`` expect 3-D state arrays and call 3-D plot
            methods.  The caller must pass 3-D axes when ``is_3d=True``.
    """

    _LAYER_NAMES = frozenset(
        {
            "obstacle",
            "tree",
            "found_path",
            "pruned_path",
            "trajectory",
            "executed",
            "annotation",
        }
    )

    def __init__(
        self,
        *,
        draw_obstacles: bool = True,
        draw_tree: bool = True,
        draw_found_path: bool = True,
        draw_pruned_path: bool = True,
        draw_trajectory: bool = True,
        draw_executed: bool = True,
        draw_start_goal: bool = True,
        styles: dict[str, LayerStyle] | None = None,
        is_3d: bool = False,
    ) -> None:
        self.draw_obstacles = draw_obstacles
        self.draw_tree = draw_tree
        self.draw_found_path = draw_found_path
        self.draw_pruned_path = draw_pruned_path
        self.draw_trajectory = draw_trajectory
        self.draw_executed = draw_executed
        self.draw_start_goal = draw_start_goal
        self.styles: dict[str, LayerStyle] = styles or {}
        self.is_3d = is_3d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, ax: Axes, snapshot: SceneSnapshot) -> None:
        """Render all enabled layers from *snapshot* onto *ax*.

        The rendering order is: obstacles → tree → found path → pruned path
        → adjusted trajectory → executed trace → start/goal markers.

        Args:
            ax: Matplotlib axes to draw on.  Must be a 3-D axes when
                :attr:`is_3d` is ``True``.
            snapshot: The :class:`~arco.tools.viewer.scene_snapshot.
                SceneSnapshot` providing all layer data.
        """
        planner = snapshot.planner or "rrt"
        if self.draw_obstacles:
            self._render_obstacles(ax, snapshot.obstacles)
        if self.draw_tree:
            self._render_tree(ax, snapshot.tree_nodes, snapshot.tree_parent)
        if self.draw_found_path and snapshot.found_path:
            self._render_polyline(
                ax,
                snapshot.found_path,
                color=layer_hex(planner, "path"),
                linewidth=1.5,
                alpha=0.45,
                label="Found path",
                layer_key="found_path",
            )
        if self.draw_pruned_path and snapshot.pruned_path:
            self._render_pruned_landmarks(
                ax,
                snapshot.pruned_path,
                planner=planner,
            )
        if self.draw_trajectory and snapshot.adjusted_trajectory:
            self._render_polyline(
                ax,
                snapshot.adjusted_trajectory,
                color=layer_hex(planner, "trajectory"),
                linewidth=2.2,
                alpha=0.9,
                marker="o",
                markersize=3,
                label="Trajectory",
                layer_key="trajectory",
            )
        if self.draw_executed and snapshot.executed_trajectory:
            self._render_polyline(
                ax,
                snapshot.executed_trajectory,
                color=layer_hex(planner, "vehicle"),
                linewidth=1.8,
                alpha=0.85,
                linestyle="--",
                label="Executed",
                layer_key="executed",
            )
        if self.draw_start_goal:
            self._render_start_goal(ax, snapshot.start, snapshot.goal)

    # ------------------------------------------------------------------
    # Layer renderers
    # ------------------------------------------------------------------

    def _render_obstacles(
        self,
        ax: Axes,
        obstacles: list[list[float]],
    ) -> None:
        """Render obstacle scatter points.

        Args:
            ax: Target axes.
            obstacles: Flat list of obstacle points, each a state-dim list.
        """
        style = self.styles.get("obstacle", LayerStyle())
        if not style.visible or not obstacles:
            return
        pts = np.array(obstacles, dtype=float)
        self._scatter(
            ax,
            pts,
            color=style.color or obstacle_hex(),
            s=style.markersize or 4,
            alpha=style.alpha or 0.4,
            zorder=2,
            label="Obstacle",
        )

    def _render_tree(
        self,
        ax: Axes,
        nodes: list[list[float]],
        parent: list[int],
    ) -> None:
        """Render the exploration tree as thin lines from child to parent.

        Args:
            ax: Target axes.
            nodes: Tree node states.
            parent: Parallel parent index list (``-1`` for root).
        """
        style = self.styles.get("tree", LayerStyle())
        if not style.visible or not nodes or not parent:
            return
        color = style.color or "#aaaaaa"
        alpha = style.alpha or 0.35
        lw = style.linewidth or 0.6
        pts = np.array(nodes, dtype=float)
        dim = pts.shape[1] if pts.ndim == 2 else 2
        for i, pid in enumerate(parent):
            if pid < 0:
                continue
            if self.is_3d and dim >= 3:
                ax.plot(  # type: ignore[attr-defined]
                    [pts[i, 0], pts[pid, 0]],
                    [pts[i, 1], pts[pid, 1]],
                    [pts[i, 2], pts[pid, 2]],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )
            else:
                ax.plot(
                    [pts[i, 0], pts[pid, 0]],
                    [pts[i, 1], pts[pid, 1]],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )

    def _render_polyline(
        self,
        ax: Axes,
        path: list[list[float]],
        *,
        color: str,
        linewidth: float,
        alpha: float,
        marker: str | None = None,
        markersize: float | None = None,
        linestyle: str = "-",
        label: str,
        layer_key: str,
    ) -> None:
        """Render a sequence of states as a connected polyline.

        Args:
            ax: Target axes.
            path: Sequence of state lists.
            color: Line colour.
            linewidth: Line width.
            alpha: Opacity.
            marker: Optional marker style (e.g. ``"o"``).
            markersize: Marker size when *marker* is set.
            linestyle: Line style string.
            label: Legend label.
            layer_key: Key into :attr:`styles` for per-layer overrides.
        """
        style = self.styles.get(layer_key, LayerStyle())
        if not style.visible:
            return
        pts = np.array(path, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return
        kw: dict[str, Any] = {
            "color": style.color or color,
            "linewidth": style.linewidth or linewidth,
            "alpha": style.alpha or alpha,
            "label": label,
        }
        # Build the format string.  When a marker is requested we embed the
        # linestyle into the fmt so matplotlib doesn't see two separate
        # specifications (which triggers a UserWarning).
        if marker is not None:
            # e.g. marker="o" + linestyle="-" → fmt="o-"
            ls_char = linestyle if linestyle in ("-", "--", "-.", ":") else "-"
            fmt = f"{marker}{ls_char}"
            kw["markersize"] = style.markersize or markersize or 3
        else:
            # Encode linestyle in the fmt string to avoid matplotlib's
            # redundant-definition UserWarning that fires when both an
            # empty fmt (implying linestyle="-") and a linestyle kwarg
            # are passed simultaneously.
            fmt = linestyle if linestyle in ("-", "--", "-.", ":") else "-"
        dim = pts.shape[1]
        if self.is_3d and dim >= 3:
            ax.plot(  # type: ignore[attr-defined]
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                fmt,
                **kw,
            )
        else:
            ax.plot(pts[:, 0], pts[:, 1], fmt, **kw)

    def _render_start_goal(
        self,
        ax: Axes,
        start: list[float],
        goal: list[float],
    ) -> None:
        """Render start (square) and goal (×) markers.

        Args:
            ax: Target axes.
            start: Start state list.
            goal: Goal state list.
        """
        style = self.styles.get("annotation", LayerStyle())
        if not style.visible:
            return
        color = style.color or annotation_hex()
        ms = style.markersize or 8
        if not start or not goal:
            return
        if self.is_3d and len(start) >= 3 and len(goal) >= 3:
            ax.scatter(  # type: ignore[attr-defined]
                [start[0]],
                [start[1]],
                [start[2]],
                color=color,
                s=ms * 8,
                marker="s",
                zorder=6,
                label="Start",
            )
            ax.scatter(  # type: ignore[attr-defined]
                [goal[0]],
                [goal[1]],
                [goal[2]],
                color=color,
                s=ms * 8,
                marker="x",
                linewidths=2,
                zorder=6,
                label="Goal",
            )
        else:
            ax.plot(
                start[0],
                start[1],
                "s",
                color=color,
                ms=ms,
                zorder=6,
                label="Start",
            )
            ax.plot(
                goal[0],
                goal[1],
                "x",
                color=color,
                ms=ms,
                mew=2,
                zorder=6,
                label="Goal",
            )

    def _render_pruned_landmarks(
        self,
        ax: Axes,
        pruned_path: list[list[float]],
        *,
        planner: str,
    ) -> None:
        """Render pruned-path nodes as glowing squares.

        Only the **nodes** (not edges) of the pruned path are shown, as they
        are the anchors that define the final trajectory segments.  A three-
        layer halo (outer glow → mid ring → bright core) is drawn so the
        landmarks stand out against the exploration tree.

        Args:
            ax: Target axes.
            pruned_path: Pruned waypoint states (nodes only).
            planner: Planner key for palette lookup.
        """
        style = self.styles.get("pruned_path", LayerStyle())
        if not style.visible:
            return
        pts = np.array(pruned_path, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return
        color = style.color or layer_hex(planner, "path")
        base_s = style.markersize or 60
        # Three concentric square scatter calls create the glow effect:
        #   1. large halo   — very transparent outer bloom
        #   2. medium ring  — semi-transparent mid layer
        #   3. bright core  — opaque inner square
        glow_layers = [
            (base_s * 8, 0.12),
            (base_s * 3, 0.30),
            (base_s,     0.90),
        ]
        dim = pts.shape[1]
        for s_val, alpha_val in glow_layers:
            kw: dict[str, Any] = {
                "c": color,
                "s": s_val,
                "alpha": alpha_val,
                "marker": "s",
                "linewidths": 0,
                "zorder": 4,
            }
            if self.is_3d and dim >= 3:
                ax.scatter(  # type: ignore[attr-defined]
                    pts[:, 0], pts[:, 1], pts[:, 2], **kw
                )
            else:
                ax.scatter(pts[:, 0], pts[:, 1], **kw)
        # Add a single artist for the legend entry (labelled on the core).
        label_kw: dict[str, Any] = {
            "c": color,
            "s": base_s,
            "alpha": 0.90,
            "marker": "s",
            "linewidths": 0,
            "zorder": 4,
            "label": "Pruned waypoints",
        }
        if self.is_3d and dim >= 3:
            ax.scatter(  # type: ignore[attr-defined]
                pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], **label_kw
            )
        else:
            ax.scatter(pts[0:1, 0], pts[0:1, 1], **label_kw)

    def _scatter(
        self,
        ax: Axes,
        pts: np.ndarray,
        *,
        color: str,
        s: float,
        alpha: float,
        zorder: int = 3,
        label: str = "",
    ) -> None:
        """Scatter-plot a set of points in 2-D or 3-D.

        Args:
            ax: Target axes.
            pts: Shape ``(N, D)`` array; only dims 0–2 are used.
            color: Marker colour.
            s: Marker area.
            alpha: Opacity.
            zorder: Z-order layer.
            label: Legend label.
        """
        if pts.ndim != 2 or pts.shape[0] == 0:
            return
        dim = pts.shape[1]
        if self.is_3d and dim >= 3:
            ax.scatter(  # type: ignore[attr-defined]
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=color,
                s=s,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )
        else:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=color,
                s=s,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )
