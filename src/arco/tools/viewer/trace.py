"""Trace drawing for the dynamic actor.

The "trace" visualisation renders:

1. **Tail** — the portion of the executed trajectory already visited,
   rendered as a fading polyline from the oldest to the current position.
2. **Actor** — the moving object at its current position, rendered as a
   prominent marker with an optional heading arrow (for Dubins / SE(2)
   robots).

Both ``arcosim --image`` (static image) and ``arcosim`` (pygame) should use
:func:`draw_trace` so the trailing-trace appearance is standardised across
tools.

Example
-------
>>> from arco.tools.viewer.trace import draw_trace
>>> draw_trace(ax, executed=[(0,0),(0.5,0.5),(1,1)], step=2)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


@dataclasses.dataclass
class TraceStyle:
    """Visual style for the trace rendering.

    Attributes:
        tail_color: Colour of the tail polyline.
        tail_linewidth: Width of the tail polyline.
        tail_alpha_start: Opacity at the oldest (furthest) tail point
            in [0, 1].
        tail_alpha_end: Opacity at the current (newest) tail point
            in [0, 1].
        tail_length: Maximum number of past steps to include in the
            tail.  ``None`` renders the full executed trajectory.
        actor_color: Colour of the actor marker.
        actor_markersize: Size of the actor marker.
        actor_marker: Matplotlib marker spec (default ``"o"``).
        arrow_color: Colour of the heading arrow; ``None`` disables the
            arrow.
        arrow_length: Arrow length in data units.
    """

    tail_color: str = "#2196f3"
    tail_linewidth: float = 2.0
    tail_alpha_start: float = 0.10
    tail_alpha_end: float = 0.85
    tail_length: int | None = None
    actor_color: str = "#f44336"
    actor_markersize: float = 10.0
    actor_marker: str = "o"
    arrow_color: str | None = "#f44336"
    arrow_length: float = 0.05


_DEFAULT_STYLE = TraceStyle()


def draw_trace(
    ax: Axes,
    executed: Sequence[Sequence[float]],
    step: int | None = None,
    *,
    style: TraceStyle | None = None,
    heading_index: int | None = 2,
    is_3d: bool = False,
    label: str = "Actor",
) -> None:
    """Render the dynamic actor and its trailing trace.

    Draws up to ``style.tail_length`` past poses as a fading polyline,
    then renders the actor at ``executed[step]`` with an optional heading
    arrow.

    Args:
        ax: Matplotlib axes to draw on.
        executed: Sequence of state lists — at minimum ``[x, y, …]``; a
            third element is treated as the heading angle (radians) when
            *heading_index* is ``2``.
        step: Current time step index into *executed*.  ``None`` defaults
            to the last element (full static trace).
        style: Optional :class:`TraceStyle` to override defaults.
        heading_index: Index of the heading component within each state
            tuple.  ``None`` suppresses the heading arrow.
        is_3d: When ``True`` render on 3-D axes (z component is
            ``state[2]``).
        label: Legend label for the actor marker.

    Raises:
        ValueError: If *executed* is empty.
    """
    if not executed:
        raise ValueError("executed must be non-empty")
    st = style or _DEFAULT_STYLE

    pts = np.array(executed, dtype=float)
    n = len(pts)
    cur = n - 1 if step is None else max(0, min(step, n - 1))

    # ---- Determine tail window ----------------------------------------
    tail_end = cur + 1
    if st.tail_length is not None:
        tail_start = max(0, tail_end - st.tail_length)
    else:
        tail_start = 0
    tail_pts = pts[tail_start:tail_end]

    # ---- Draw tail with alpha gradient --------------------------------
    n_seg = len(tail_pts) - 1
    if n_seg >= 1:
        for i in range(n_seg):
            # Linear interpolation of alpha from start to end.
            t = i / max(n_seg - 1, 1)
            seg_alpha = st.tail_alpha_start + t * (
                st.tail_alpha_end - st.tail_alpha_start
            )
            if is_3d and pts.shape[1] >= 3:
                ax.plot(  # type: ignore[attr-defined]
                    [tail_pts[i, 0], tail_pts[i + 1, 0]],
                    [tail_pts[i, 1], tail_pts[i + 1, 1]],
                    [tail_pts[i, 2], tail_pts[i + 1, 2]],
                    color=st.tail_color,
                    linewidth=st.tail_linewidth,
                    alpha=seg_alpha,
                )
            else:
                ax.plot(
                    [tail_pts[i, 0], tail_pts[i + 1, 0]],
                    [tail_pts[i, 1], tail_pts[i + 1, 1]],
                    color=st.tail_color,
                    linewidth=st.tail_linewidth,
                    alpha=seg_alpha,
                )

    # ---- Draw actor at current position --------------------------------
    cx, cy = float(pts[cur, 0]), float(pts[cur, 1])
    if is_3d and pts.shape[1] >= 3:
        cz = float(pts[cur, 2])
        ax.scatter(  # type: ignore[attr-defined]
            [cx],
            [cy],
            [cz],
            c=st.actor_color,
            s=st.actor_markersize**2,
            marker=st.actor_marker,
            zorder=8,
            label=label,
        )
    else:
        ax.plot(
            cx,
            cy,
            marker=st.actor_marker,
            color=st.actor_color,
            markersize=st.actor_markersize,
            zorder=8,
            linestyle="none",
            label=label,
        )

    # ---- Draw heading arrow (2-D only) ---------------------------------
    if (
        not is_3d
        and st.arrow_color is not None
        and heading_index is not None
        and pts.shape[1] > heading_index
    ):
        theta = float(pts[cur, heading_index])
        dx = st.arrow_length * np.cos(theta)
        dy = st.arrow_length * np.sin(theta)
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops={
                "arrowstyle": "->",
                "color": st.arrow_color,
                "lw": 1.5,
            },
            zorder=9,
        )
