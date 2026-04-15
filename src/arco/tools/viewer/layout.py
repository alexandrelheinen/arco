"""Standard two-frame figure layout for ARCO visualizations.

Every ARCO example and simulation uses the same canonical layout:

* **Top-left** — Workspace axes: the physical environment (Cartesian space).
* **Top-right** — C-space axes: configuration space representation.
* **Bottom** — Full-width text region: legends, metrics, and status text.

Example
-------
>>> fig, ax_ws, ax_cs, ax_bottom = StandardLayout.create("My scenario")
>>> ax_ws.plot(...)
>>> ax_cs.scatter(...)
>>> StandardLayout.write_metrics(ax_bottom, ["RRT*: 120 nodes", "Time: 3.2 s"])
"""

from __future__ import annotations

from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class StandardLayout:
    """Factory for the canonical ARCO two-frame figure layout.

    Creates a :class:`~matplotlib.figure.Figure` pre-configured with three
    regions:

    * **Top-left** (``ax_ws``) — Workspace subplot.
    * **Top-right** (``ax_cs``) — C-space subplot.
    * **Bottom** (``ax_bottom``) — Text-only metrics strip.

    Both top frames may optionally use a 3-D projection when the scenario
    requires 3-D axes (e.g. PPP, RRP robots).

    Class Attributes:
        FIGSIZE: Default figure width × height in inches ``(14, 9)``.
        HEIGHT_RATIOS: Row height ratios ``[top, bottom]`` = ``[3, 1]``.
        HSPACE: Vertical spacing between rows.
        WSPACE: Horizontal spacing between columns.
    """

    FIGSIZE: tuple[int, int] = (14, 9)
    HEIGHT_RATIOS: list[float] = [3.0, 1.0]
    HSPACE: float = 0.35
    WSPACE: float = 0.30

    @classmethod
    def create(
        cls,
        title: str = "",
        ws_3d: bool = False,
        cs_3d: bool = False,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[Figure, Axes, Axes, Axes]:
        """Create the standard layout figure.

        Args:
            title: Optional ``suptitle`` string displayed above both frames.
            ws_3d: When ``True``, create the workspace axes with
                ``projection="3d"``.
            cs_3d: When ``True``, create the C-space axes with
                ``projection="3d"``.
            figsize: Override the default figure size ``(width, height)`` in
                inches.  Uses :attr:`FIGSIZE` when ``None``.

        Returns:
            A four-tuple ``(fig, ax_ws, ax_cs, ax_bottom)`` where

            * *fig* is the :class:`~matplotlib.figure.Figure`.
            * *ax_ws* is the top-left workspace
              :class:`~matplotlib.axes.Axes`.
            * *ax_cs* is the top-right C-space
              :class:`~matplotlib.axes.Axes`.
            * *ax_bottom* is the bottom text
              :class:`~matplotlib.axes.Axes` (axis decorations disabled).
        """
        size = figsize or cls.FIGSIZE
        fig = plt.figure(figsize=size)
        fig.patch.set_facecolor("white")

        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=cls.HEIGHT_RATIOS,
            hspace=cls.HSPACE,
            wspace=cls.WSPACE,
        )

        ws_kw: dict[str, Any] = {"projection": "3d"} if ws_3d else {}
        cs_kw: dict[str, Any] = {"projection": "3d"} if cs_3d else {}

        ax_ws: Axes = fig.add_subplot(gs[0, 0], **ws_kw)
        ax_cs: Axes = fig.add_subplot(gs[0, 1], **cs_kw)
        ax_bottom: Axes = fig.add_subplot(gs[1, :])

        ax_bottom.set_axis_off()

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold")

        return fig, ax_ws, ax_cs, ax_bottom

    @staticmethod
    def write_metrics(
        ax_bottom: Axes,
        lines: list[str],
        *,
        fontsize: int = 9,
        columns: int = 1,
    ) -> None:
        """Render metric lines in the bottom text strip.

        When *columns* > 1 the lines are distributed evenly across *columns*
        horizontal columns so that wide figures remain readable.

        Args:
            ax_bottom: The bottom text axes returned by :meth:`create`.
            lines: Lines of text to display.  Each element becomes one row.
            fontsize: Font size used for all metric text.
            columns: Number of horizontal columns to split lines across.
                Must be ≥ 1.
        """
        if not lines:
            return

        if columns <= 1:
            ax_bottom.text(
                0.5,
                0.5,
                "\n".join(lines),
                transform=ax_bottom.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                family="monospace",
            )
            return

        # Distribute lines across columns.
        n = len(lines)
        per_col = (n + columns - 1) // columns
        x_positions = [(i + 0.5) / columns for i in range(columns)]
        for col_idx, x in enumerate(x_positions):
            start = col_idx * per_col
            chunk = lines[start : start + per_col]
            if chunk:
                ax_bottom.text(
                    x,
                    0.5,
                    "\n".join(chunk),
                    transform=ax_bottom.transAxes,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    family="monospace",
                )
