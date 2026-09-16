"""Canvas: the 16:9 stage and the drawing primitives every plate shares.

The gallery has one rule that makes it look like one body of work: no
plate draws a raw matplotlib line.  Strokes go through :meth:`Canvas.glow`
(a bloom ramp), :meth:`Canvas.ribbon` (per-segment colour) or
:meth:`Canvas.tree` (depth-faded edge collection), and every plate wears
the same chrome from :meth:`Canvas.chrome`.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    to_rgb,
    to_rgba,
)
from matplotlib.patches import Polygon  # noqa: E402

from .theme import Theme, display_font, mono_font, text_font

ASPECT = 16.0 / 9.0


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned world window shown by a stage.

    Attributes:
        x0: Lower x limit in world units.
        y0: Lower y limit in world units.
        x1: Upper x limit in world units.
        y1: Upper y limit in world units.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Return the window width in world units."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Return the window height in world units."""
        return self.y1 - self.y0

    def to_aspect(self, aspect: float = ASPECT) -> "Bounds":
        """Return a concentric window widened or heightened to *aspect*.

        Args:
            aspect: Target width / height ratio.

        Returns:
            A new :class:`Bounds` with the same centre and the requested
            aspect ratio, never smaller than the original window.
        """
        cx = 0.5 * (self.x0 + self.x1)
        cy = 0.5 * (self.y0 + self.y1)
        w, h = self.width, self.height
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        return Bounds(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def padded(self, fraction: float) -> "Bounds":
        """Return the window grown by *fraction* of its size on each side.

        Args:
            fraction: Growth per side as a fraction of width / height.

        Returns:
            The enlarged window.
        """
        dx, dy = self.width * fraction, self.height * fraction
        return Bounds(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)


def letterspace(text: str, spaces: int = 1) -> str:
    """Return *text* with blanks inserted between characters.

    Matplotlib has no letter-spacing property, so wide display type is
    faked by padding.  Existing blanks are widened as well so word gaps
    stay readable.

    Args:
        text: Source string.
        spaces: Number of blanks inserted between characters.

    Returns:
        The letter-spaced string.
    """
    pad = " " * spaces
    return pad.join(text)


class Canvas:
    """A 16:9 figure with a painted stage, glow primitives and chrome.

    Args:
        theme: Visual configuration to draw with.
        width_in: Figure width in inches.  Height follows the 16:9 ratio.
        dpi: Output resolution; ``16 x 240`` gives a 3840 x 2160 PNG.
    """

    def __init__(
        self,
        theme: Theme,
        width_in: float = 16.0,
        dpi: int = 240,
    ) -> None:
        """Create the figure and its painted background layer."""
        self.theme = theme
        self.dpi = dpi
        self.scale = width_in / 16.0
        self.fig = plt.figure(
            figsize=(width_in, width_in / ASPECT),
            dpi=dpi,
            facecolor=theme.background,
        )
        self._paint_background()

    # ------------------------------------------------------------------
    # Stage construction
    # ------------------------------------------------------------------

    def _paint_background(self) -> None:
        """Paint the gradient, vignette and grain behind every plate."""
        theme = self.theme
        ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0), zorder=0)
        ax.set_axis_off()
        ny, nx = 360, 640
        gx, gy = np.meshgrid(
            np.linspace(-ASPECT, ASPECT, nx), np.linspace(-1.0, 1.0, ny)
        )
        radius = np.sqrt((gx * 0.62) ** 2 + (gy * 1.02) ** 2)
        field = np.clip(1.0 - radius / 1.28, 0.0, 1.0) ** 1.65
        cmap = LinearSegmentedColormap.from_list(
            "stage", [theme.background, theme.background_core]
        )
        ax.imshow(
            field,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )
        if theme.grain > 0.0:
            rng = np.random.default_rng(4)
            noise = rng.normal(0.5, 0.16, (ny // 2, nx // 2))
            ax.imshow(
                np.clip(noise, 0.0, 1.0),
                cmap="gray",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="auto",
                interpolation="bilinear",
                alpha=theme.grain,
                zorder=1,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        self.background_axes = ax

    def pt(self, size: float) -> float:
        """Return a point size scaled to the figure width.

        Args:
            size: Point size authored against a 16-inch-wide figure.

        Returns:
            The size to pass to matplotlib for this canvas.
        """
        return size * self.scale

    def scrim(
        self,
        rect: Tuple[float, float, float, float],
        strength: float = 0.62,
        direction: str = "left",
    ) -> None:
        """Darken (or lighten) a band so chrome stays legible over artwork.

        Args:
            rect: Figure-fraction rectangle ``(left, bottom, w, h)``.
            strength: Peak opacity of the scrim.
            direction: Edge the scrim is anchored to — ``"left"``,
                ``"right"``, ``"top"`` or ``"bottom"``.
        """
        ax = self.fig.add_axes(rect, zorder=30)
        ax.set_axis_off()
        gradient = np.linspace(1.0, 0.0, 256) ** 0.85
        if direction in ("left", "right"):
            alpha = np.tile(gradient, (2, 1))
            if direction == "right":
                alpha = alpha[:, ::-1]
        else:
            alpha = np.tile(gradient.reshape(-1, 1), (1, 2))
            if direction == "bottom":
                alpha = alpha[::-1, :]
        image = np.zeros(alpha.shape + (4,))
        image[..., :3] = to_rgb(self.theme.background)
        image[..., 3] = np.clip(alpha * strength, 0.0, 1.0)
        ax.imshow(
            image,
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="auto",
            interpolation="bilinear",
        )

    def stage(
        self,
        bounds: Bounds,
        rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        zorder: int = 2,
    ) -> Axes:
        """Add a transparent, equal-aspect drawing area over the stage.

        Args:
            bounds: World window to show.  It is widened to the panel
                aspect so nothing is squashed.
            rect: Figure-fraction rectangle ``(left, bottom, w, h)``.
            zorder: Stacking order of the panel.

        Returns:
            The configured matplotlib axes.
        """
        ax = self.fig.add_axes(rect, zorder=zorder)
        ax.set_facecolor("none")
        ax.set_axis_off()
        panel_aspect = (rect[2] * ASPECT) / rect[3]
        view = bounds.to_aspect(panel_aspect)
        ax.set_xlim(view.x0, view.x1)
        ax.set_ylim(view.y0, view.y1)
        ax.set_aspect("equal", adjustable="box")
        return ax

    def blueprint(self, ax: Axes, spacing: float, alpha: float = 0.5) -> None:
        """Draw the faint measurement grid under a stage.

        Args:
            ax: Stage axes to draw on.
            spacing: Grid pitch in world units.
            alpha: Multiplier on the theme hairline opacity.
        """
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        color = to_rgba(self.theme.rule, 0.42 * alpha)
        segments = []
        for x in np.arange(
            np.ceil(x0 / spacing) * spacing, x1 + 1e-9, spacing
        ):
            segments.append([(x, y0), (x, y1)])
        for y in np.arange(
            np.ceil(y0 / spacing) * spacing, y1 + 1e-9, spacing
        ):
            segments.append([(x0, y), (x1, y)])
        ax.add_collection(
            LineCollection(segments, colors=[color], linewidths=0.55, zorder=1)
        )

    # ------------------------------------------------------------------
    # Stroke primitives
    # ------------------------------------------------------------------

    def glow(
        self,
        ax: Axes,
        points: np.ndarray,
        color: str,
        width: float = 2.2,
        zorder: float = 10.0,
        ramp: Optional[Sequence[Tuple[float, float]]] = None,
        core_alpha: float = 1.0,
    ) -> None:
        """Stroke a polyline as a bloom: wide faint passes under a core.

        Args:
            ax: Stage axes to draw on.
            points: ``(N, 2)`` polyline in world units.
            color: Stroke colour.
            width: Core line width in points.
            zorder: Stacking order of the core pass.
            ramp: Override for the theme bloom ramp.
            core_alpha: Alpha of the topmost (core) pass.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            return
        passes = tuple(ramp if ramp is not None else self.theme.glow)
        width *= self.scale
        for index, (scale, alpha) in enumerate(passes):
            is_core = index == len(passes) - 1
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linewidth=width * scale,
                alpha=(core_alpha if is_core else alpha),
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=zorder - (len(passes) - index) * 0.01,
            )

    def ribbon(
        self,
        ax: Axes,
        points: np.ndarray,
        values: np.ndarray,
        cmap,
        width: float = 3.0,
        zorder: float = 12.0,
        halo: Optional[str] = None,
    ) -> LineCollection:
        """Stroke a polyline whose colour varies segment by segment.

        Args:
            ax: Stage axes to draw on.
            points: ``(N, 2)`` polyline in world units.
            values: ``(N,)`` scalar field, normalised internally to [0, 1].
            cmap: Matplotlib colormap mapping the normalised field.
            width: Core line width in points.
            zorder: Stacking order of the ribbon.
            halo: Optional colour for a soft bloom drawn under the ribbon.

        Returns:
            The added :class:`~matplotlib.collections.LineCollection`.
        """
        pts = np.asarray(points, dtype=float)
        vals = np.asarray(values, dtype=float)
        span = float(vals.max() - vals.min())
        norm = (vals - vals.min()) / (span if span > 1e-12 else 1.0)
        mid = 0.5 * (norm[:-1] + norm[1:])
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        if halo is not None:
            self.glow(
                ax,
                pts,
                halo,
                width=width,
                zorder=zorder - 0.5,
                ramp=self.theme.glow[:-1],
                core_alpha=0.0,
            )
        collection = LineCollection(
            segments,
            colors=cmap(mid),
            linewidths=width * self.scale,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
        ax.add_collection(collection)
        return collection

    def tree(
        self,
        ax: Axes,
        nodes: Sequence[np.ndarray],
        parents,
        color: str,
        depth: Optional[np.ndarray] = None,
        width: float = 0.9,
        zorder: float = 5.0,
        alpha_scale: float = 1.0,
        cmap=None,
    ) -> None:
        """Draw a planner tree as a depth-faded filigree of edges.

        Edges near the root are dim and thick, leaves are bright and thin,
        so a dense tree reads as structure rather than as a hairball.

        Args:
            ax: Stage axes to draw on.
            nodes: Tree node positions, index-aligned with *parents*.
            parents: Mapping of node index to parent index (or ``None``).
            color: Base hue when *cmap* is not given.
            depth: Optional per-node scalar (cost-to-come, generation)
                driving the fade.  Defaults to topological depth.
            width: Line width of leaf edges in points.
            zorder: Stacking order of the edge collection.
            alpha_scale: Multiplier on the theme tree alpha ramp.
            cmap: Optional colormap replacing the single *color*.
        """
        positions = np.asarray(nodes, dtype=float)
        if positions.size == 0:
            return
        if depth is None:
            depth = _topological_depth(parents, len(positions))
        low, high = np.nanpercentile(depth, (2.0, 98.0))
        span = float(high - low)
        norm = (
            np.clip((depth - low) / (span if span > 1e-12 else 1.0), 0.0, 1.0)
            ** 0.72
        )

        segments, shade = [], []
        for child, parent in parents.items():
            if parent is None or parent >= len(positions):
                continue
            segments.append([positions[parent], positions[child]])
            shade.append(norm[child])
        if not segments:
            return
        shade_array = np.asarray(shade)
        base = np.asarray(
            cmap(shade_array)
            if cmap is not None
            else [to_rgb(color)] * len(shade_array)
        )
        alphas = self.theme.tree_alpha + shade_array * (
            self.theme.tree_alpha_tip - self.theme.tree_alpha
        )
        rgba = np.empty((len(shade_array), 4))
        rgba[:, :3] = base[:, :3]
        rgba[:, 3] = np.clip(alphas * alpha_scale, 0.0, 1.0)
        widths = width * self.scale * (1.55 - 0.75 * shade_array)
        ax.add_collection(
            LineCollection(
                segments,
                colors=rgba,
                linewidths=widths,
                capstyle="round",
                zorder=zorder,
            )
        )

    def blobs(
        self,
        ax: Axes,
        polygons: Sequence[np.ndarray],
        zorder: float = 3.0,
    ) -> None:
        """Fill obstacle bodies with a soft rim and a recessed interior.

        Args:
            ax: Stage axes to draw on.
            polygons: Closed obstacle outlines as ``(N, 2)`` arrays.
            zorder: Stacking order of the bodies.
        """
        theme = self.theme
        for poly in polygons:
            for factor, alpha in (
                (1.16, 0.030),
                (1.10, 0.045),
                (1.05, 0.070),
            ):
                ax.add_patch(
                    Polygon(
                        _scaled(poly, factor),
                        closed=True,
                        facecolor=theme.obstacle_edge,
                        edgecolor="none",
                        alpha=alpha,
                        zorder=zorder - 0.3,
                    )
                )
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor=theme.obstacle_fill,
                    edgecolor="none",
                    alpha=theme.obstacle_alpha,
                    zorder=zorder,
                )
            )
            ax.add_patch(
                Polygon(
                    _scaled(poly, 0.90),
                    closed=True,
                    facecolor=theme.background,
                    edgecolor="none",
                    alpha=0.45,
                    zorder=zorder + 0.05,
                )
            )
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor="none",
                    edgecolor=theme.obstacle_edge,
                    linewidth=0.9 * self.scale,
                    alpha=0.50,
                    zorder=zorder + 0.1,
                )
            )

    def terminal(
        self,
        ax: Axes,
        point: np.ndarray,
        color: str,
        label: str = "",
        radius: float = 1.6,
        zorder: float = 20.0,
    ) -> None:
        """Mark a start or goal pose with concentric rings and a label.

        Args:
            ax: Stage axes to draw on.
            point: ``(2,)`` world position.
            color: Ring colour.
            label: Optional small-caps label drawn above the rings.
            radius: Radius of the inner ring in world units.
            zorder: Stacking order of the marker.

        Note:
            The marker is skipped when *point* falls outside the axes
            window, so cropped plates do not show half a ring.
        """
        theme = self.theme
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        margin = radius * 4.0
        if not (
            x0 - margin <= point[0] <= x1 + margin
            and y0 - margin <= point[1] <= y1 + margin
        ):
            return
        for scale, alpha, width in (
            (3.3, 0.16, 1.0),
            (2.1, 0.34, 1.3),
            (1.0, 0.95, 1.8),
        ):
            ax.add_patch(
                plt.Circle(
                    (point[0], point[1]),
                    radius * scale,
                    fill=False,
                    edgecolor=color,
                    linewidth=width,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
        ax.add_patch(
            plt.Circle(
                (point[0], point[1]),
                radius * 0.34,
                facecolor=color,
                edgecolor="none",
                zorder=zorder + 0.1,
            )
        )
        if label:
            below = point[1] > 0.72 * ax.get_ylim()[1]
            offset = -radius * 4.9 if below else radius * 4.4
            ax.text(
                point[0],
                point[1] + offset,
                letterspace(label.upper()),
                color=theme.text,
                fontproperties=text_font(self.pt(theme.caption_size * 0.92)),
                ha="center",
                va="top" if below else "bottom",
                zorder=zorder,
            )

    def strip(
        self,
        rect: Tuple[float, float, float, float],
        label: str = "",
        zorder: int = 35,
    ) -> Axes:
        """Add a bare inset band for a small supporting plot.

        The band carries a baseline and a label and nothing else: gallery
        plates show a trend, not a chart to read values off.

        Args:
            rect: Figure-fraction rectangle ``(left, bottom, w, h)``.
            label: Small-caps label drawn above the band's left edge.
            zorder: Stacking order of the band.

        Returns:
            The prepared axes.
        """
        theme = self.theme
        ax = self.fig.add_axes(rect, zorder=zorder)
        ax.set_facecolor("none")
        ax.set_axis_off()
        if label:
            self.fig.text(
                rect[0],
                rect[1] + rect[3] + 0.012,
                letterspace(label.upper()),
                color=theme.text,
                fontproperties=text_font(self.pt(theme.caption_size * 0.9)),
                va="bottom",
                ha="left",
                zorder=zorder + 1,
            )
        return ax

    def baseline(self, ax: Axes, y: float = 0.0) -> None:
        """Draw the hairline zero reference inside a strip.

        Args:
            ax: Strip axes.
            y: Data value of the reference line.
        """
        ax.axhline(
            y,
            color=self.theme.rule,
            linewidth=0.9 * self.scale,
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Chrome
    # ------------------------------------------------------------------

    def chrome(
        self,
        title: str,
        subtitle: str,
        metrics: Sequence[str] = (),
        legend: Sequence[Tuple[str, str]] = (),
        wrap: int = 74,
    ) -> None:
        """Draw the shared frame every plate in the series wears.

        Two soft scrims letterbox the artwork so type stays legible over
        whatever the solver happened to draw there: title and subtitle
        top-left, legend top-right, wordmark bottom-left, run metrics
        bottom-right.

        Args:
            title: Plate title, rendered letter-spaced in caps.
            subtitle: One-line description, wrapped to *wrap* columns.
            metrics: Monospace lines of real numbers from the run shown.
            legend: ``(label, colour)`` pairs for the top-right key.
            wrap: Wrap width of the subtitle in characters.
        """
        theme = self.theme
        fig = self.fig
        self.scrim((0.0, 0.660, 1.0, 0.340), strength=1.0, direction="top")
        self.scrim((0.0, 0.0, 1.0, 0.200), strength=1.0, direction="bottom")

        fig.text(
            0.042,
            0.930,
            letterspace(title.upper(), 1),
            color=theme.text,
            fontproperties=display_font(self.pt(theme.title_size)),
            va="center",
            ha="left",
            zorder=40,
        )
        fig.lines.append(
            plt.Line2D(
                (0.0425, 0.118),
                (0.902, 0.902),
                transform=fig.transFigure,
                color=theme.rule,
                linewidth=1.0 * self.scale,
                zorder=40,
            )
        )
        for index, line in enumerate(textwrap.wrap(subtitle, wrap)):
            fig.text(
                0.0425,
                0.878 - index * 0.030,
                line,
                color=theme.text_dim,
                fontproperties=display_font(self.pt(theme.subtitle_size)),
                va="center",
                ha="left",
                zorder=40,
            )
        for index, (label, color) in enumerate(legend):
            y = 0.930 - index * 0.032
            fig.text(
                0.944,
                y,
                label,
                color=theme.text_dim,
                fontproperties=text_font(self.pt(theme.caption_size)),
                va="center",
                ha="right",
                zorder=40,
            )
            fig.lines.append(
                plt.Line2D(
                    (0.950, 0.962),
                    (y, y),
                    transform=fig.transFigure,
                    color=color,
                    linewidth=3.0 * self.scale,
                    solid_capstyle="round",
                    zorder=40,
                )
            )
        self._wordmark()
        for index, line in enumerate(reversed(list(metrics))):
            fig.text(
                0.958,
                0.048 + index * 0.029,
                line,
                color=theme.text_dim,
                fontproperties=mono_font(self.pt(theme.caption_size)),
                va="center",
                ha="right",
                zorder=40,
            )

    def _wordmark(self) -> None:
        """Draw the ARCO wordmark and its three-method accent rail."""
        theme = self.theme
        fig = self.fig
        fig.text(
            0.042,
            0.075,
            letterspace("ARCO", 2),
            color=theme.text,
            fontproperties=display_font(self.pt(theme.wordmark_size)),
            va="center",
            ha="left",
            zorder=40,
        )
        for index, color in enumerate((theme.rrt, theme.sst, theme.astar)):
            x0 = 0.042 + index * 0.0175
            fig.lines.append(
                plt.Line2D(
                    (x0, x0 + 0.0135),
                    (0.046, 0.046),
                    transform=fig.transFigure,
                    color=color,
                    linewidth=2.6,
                    solid_capstyle="round",
                    zorder=40,
                )
            )

    def save(self, path) -> None:
        """Write the plate to *path* and release the figure.

        Args:
            path: Destination PNG path; parent directories must exist.
        """
        self.fig.savefig(
            path,
            dpi=self.dpi,
            facecolor=self.theme.background,
            pad_inches=0,
            pil_kwargs={"optimize": True},
        )
        plt.close(self.fig)


def _scaled(polygon: np.ndarray, factor: float) -> np.ndarray:
    """Return *polygon* scaled about its own centroid.

    Args:
        polygon: ``(N, 2)`` outline.
        factor: Uniform scale factor.

    Returns:
        The scaled outline.
    """
    centre = polygon.mean(axis=0)
    return centre + (polygon - centre) * factor


def _topological_depth(parents, count: int) -> np.ndarray:
    """Return hop distance from the root for every node index.

    Args:
        parents: Mapping of node index to parent index (or ``None``).
        count: Number of nodes.

    Returns:
        ``(count,)`` array of hop depths.
    """
    depth = np.zeros(count, dtype=float)
    for index in range(count):
        hops, cursor, guard = 0, parents.get(index), 0
        while cursor is not None and guard < count:
            hops += 1
            cursor = parents.get(cursor)
            guard += 1
        depth[index] = hops
    return depth
