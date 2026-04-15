"""Standard two-frame figure layout for ARCO visualizations.

Every ARCO example and simulation uses the same canonical layout:

* **Left** (1/4) — Data panel: metrics, legend text, and status.
* **Top-right** (3/4) — Workspace axes: the physical environment (Cartesian space).
* **Bottom-right** (3/4) — C-space axes: configuration space representation.

Example
-------
>>> fig, ax_ws, ax_cs, ax_data = StandardLayout.create("My scenario")
>>> ax_ws.plot(...)
>>> ax_cs.scatter(...)
>>> StandardLayout.write_metrics(ax_data, ["RRT*: 120 nodes", "Time: 3.2 s"])
"""

from __future__ import annotations

from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class StandardLayout:
    """Factory for the canonical ARCO two-frame figure layout.

    Creates a :class:`~matplotlib.figure.Figure` with:

    * **Left** (``ax_bottom``, 1/4 width) — Data panel: metrics and text.
    * **Top-right** (``ax_ws``, 3/4 width) — Workspace subplot.
    * **Bottom-right** (``ax_cs``, 3/4 width) — C-space subplot.

    Both plot frames may optionally use a 3-D projection when the scenario
    requires 3-D axes (e.g. PPP, RRP robots).

    Class Attributes:
        FIGSIZE: Default figure width × height in inches ``(14, 9)``.
        WIDTH_RATIOS: Column width ratios ``[left, right]`` = ``[1, 3]``.
        HEIGHT_RATIOS_RIGHT: Row height ratios for right column ``[top, bottom]`` = ``[3, 1]``.
        HSPACE: Vertical spacing between rows in right column.
        WSPACE: Horizontal spacing between left and right columns.
    """

    FIGSIZE: tuple[int, int] = (14, 9)
    WIDTH_RATIOS: list[float] = [1.0, 3.0]
    HEIGHT_RATIOS_RIGHT: list[float] = [3.0, 1.0]
    HSPACE: float = 0.35
    WSPACE: float = 0.15

    @classmethod
    def create(
        cls,
        title: str = "",
        ws_3d: bool = False,
        cs_3d: bool = False,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[Figure, Axes, Axes, Axes]:
        """Create the standard layout figure.

        Layout: left column (1/4) for data panel, right (3/4) for plots stacked.

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
            * *ax_ws* is the workspace
              :class:`~matplotlib.axes.Axes` (top-right).
            * *ax_cs* is the C-space
              :class:`~matplotlib.axes.Axes` (bottom-right).
            * *ax_bottom* is the left data panel
              :class:`~matplotlib.axes.Axes` (axis decorations disabled).
        """
        size = figsize or cls.FIGSIZE
        fig = plt.figure(figsize=size)
        fig.patch.set_facecolor("white")

        # Outer layout: left data panel (1/4), right plot area (3/4)
        gs_outer = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=cls.WIDTH_RATIOS,
            wspace=cls.WSPACE,
        )

        # Left column: data panel (ax_bottom)
        ax_bottom: Axes = fig.add_subplot(gs_outer[0, 0])
        ax_bottom.set_axis_off()
        # Optional: add a subtle background to the data panel
        ax_bottom.patch.set_facecolor("#f9f9f9")
        ax_bottom.patch.set_visible(True)

        # Right column: nested grid for workspace (top) and C-space (bottom)
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_outer[0, 1],
            height_ratios=cls.HEIGHT_RATIOS_RIGHT,
            hspace=cls.HSPACE,
        )

        ws_kw: dict[str, Any] = {"projection": "3d"} if ws_3d else {}
        cs_kw: dict[str, Any] = {"projection": "3d"} if cs_3d else {}

        ax_ws: Axes = fig.add_subplot(gs_right[0], **ws_kw)
        ax_cs: Axes = fig.add_subplot(gs_right[1], **cs_kw)

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
