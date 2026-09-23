"""Drawing the web plates. Flat fills, no chrome, no type.

Stroke widths are chosen for a 1600×900 master that is later shown near
320 px: a hairline on the master disappears on the card.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from .budget import cap_edges  # noqa: E402
from .scene import Basin, Flood  # noqa: E402
from .solve import BasinRuns, FloodRun  # noqa: E402
from .theme import Ground  # noqa: E402

MASTER_WIDTH = 1600
MASTER_HEIGHT = 900
MARK_SIZE = 512
OG_WIDTH = 1200
OG_HEIGHT = 630
DPI = 100

CURVE_PT = 7.2
TREE_PT = 3.4
MARK_PT = 14.0
BANDS = 5


def render_mark(ground: Ground, path: Path, dpi: int = DPI) -> None:
    """Draw the three-arc mark and write it to *path*.

    Args:
        ground: Color ground.
        path: Destination PNG.
        dpi: Raster resolution. 512 px at the default.
    """
    fig, ax = _canvas(MARK_SIZE, MARK_SIZE, ground, dpi)
    span = np.linspace(np.deg2rad(206), np.deg2rad(334), 160)
    center = np.array([0.02, -0.18])
    radii = (0.36, 0.54, 0.72)
    colors = (ground.rrt, ground.sst, ground.astar)
    for radius, color in zip(radii, colors):
        pts = np.column_stack(
            [
                center[0] + radius * np.cos(span),
                center[1] + radius * np.sin(span),
            ]
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=MARK_PT,
            solid_capstyle="round",
        )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    _write(fig, path, dpi)


def render_arc(
    ground: Ground, basin: Basin, runs: BasinRuns, path: Path, dpi: int = DPI
) -> None:
    """Draw three discs and the smoothed curve.

    Args:
        ground: Color ground.
        basin: Scene outlines and frame.
        runs: Solver output. Uses ``curve``.
        path: Destination PNG.
        dpi: Raster resolution.
    """
    fig, ax = _basin_canvas(ground, basin, dpi)
    _bodies(ax, basin, ground)
    _polyline(ax, runs.curve, ground.rrt, CURVE_PT)
    _write(fig, path, dpi)


def render_branch(
    ground: Ground, basin: Basin, runs: BasinRuns, path: Path, dpi: int = DPI
) -> None:
    """Draw the capped RRT* tree under the smoothed curve.

    Args:
        ground: Color ground.
        basin: Scene outlines and frame.
        runs: Solver output. Uses the RRT* tree and ``curve``.
        path: Destination PNG.
        dpi: Raster resolution.
    """
    fig, ax = _basin_canvas(ground, basin, dpi)
    _bodies(ax, basin, ground)
    color = ground.ink if ground.tree_uses_ink else ground.rrt
    _tree(
        ax, runs.rrt.nodes, runs.rrt.parents, _rgba(color, ground.tree_alpha)
    )
    _polyline(ax, runs.curve, ground.rrt, CURVE_PT)
    _write(fig, path, dpi)


def render_pair(
    ground: Ground, basin: Basin, runs: BasinRuns, path: Path, dpi: int = DPI
) -> None:
    """Draw the RRT* and SST paths and nothing of either tree.

    Args:
        ground: Color ground.
        basin: Scene outlines and frame.
        runs: Solver output. Uses both raw paths.
        path: Destination PNG.
        dpi: Raster resolution.
    """
    fig, ax = _basin_canvas(ground, basin, dpi)
    _bodies(ax, basin, ground)
    _polyline(ax, runs.sst.path, ground.sst, CURVE_PT)
    _polyline(ax, runs.rrt.path, ground.rrt, CURVE_PT)
    _write(fig, path, dpi)


def render_track(
    ground: Ground, basin: Basin, runs: BasinRuns, path: Path, dpi: int = DPI
) -> None:
    """Draw the reference faintly and the executed motion in front.

    No obstacle bodies: the plate is the gap between the two curves.

    Args:
        ground: Color ground.
        basin: Frame only. The discs are not drawn.
        runs: Solver output. Uses ``reference`` and ``executed``.
        path: Destination PNG.
        dpi: Raster resolution.
    """
    fig, ax = _basin_canvas(ground, basin, dpi)
    _polyline(ax, runs.reference, _rgba(ground.ink, 0.40), 4.6)
    _polyline(ax, runs.executed, ground.rrt, CURVE_PT)
    _write(fig, path, dpi)


def render_ribbon(
    ground: Ground, basin: Basin, runs: BasinRuns, path: Path, dpi: int = DPI
) -> None:
    """Draw the smoothed curve with width taken from its speed.

    Args:
        ground: Color ground.
        basin: Scene outlines and frame.
        runs: Solver output. Uses ``curve`` and ``speed``.
        path: Destination PNG.
        dpi: Raster resolution.
    """
    fig, ax = _basin_canvas(ground, basin, dpi)
    _bodies(ax, basin, ground)
    _ribbon(ax, runs.curve, runs.speed, ground.rrt)
    _write(fig, path, dpi)


def render_flood(
    ground: Ground, flood: Flood, run: FloodRun, path: Path, dpi: int = DPI
) -> None:
    """Draw five expansion bands, the blocks, and the A* path.

    Args:
        ground: Color ground.
        flood: Grid geometry.
        run: Expansion order and path, as cell indices.
        path: Destination PNG.
        dpi: Raster resolution.
    """
    fig, ax = _canvas(MASTER_WIDTH, MASTER_HEIGHT, ground, dpi)
    cell = flood.cell_size
    shape = flood.grid.data.shape
    bands = _band_index(len(run.expanded))
    for cell_ij, band in zip(run.expanded, bands):
        _cell(ax, cell_ij, cell, _band_color(ground, int(band)))
    for i0, i1, j0, j1 in flood.blocks:
        for i in range(i0, i1):
            for j in range(j0, j1):
                _cell(ax, (i, j), cell, ground.obstacle)
    centers = (run.path.astype(float) + 0.5) * cell
    _polyline(ax, centers, ground.ink, CURVE_PT)
    ax.set_xlim(0.0, shape[0] * cell)
    ax.set_ylim(0.0, shape[1] * cell)
    _write(fig, path, dpi)


def og_crop(image: np.ndarray) -> np.ndarray:
    """Center-crop a 16:9 master to 1200×630.

    The crop keeps the full width and trims the top and bottom, which is
    the window the plate compositions leave clear of important geometry.

    Args:
        image: ``(H, W, C)`` array, H/W matching the master.

    Returns:
        ``(630, 1200, C)`` array.
    """
    height, width = image.shape[:2]
    crop_h = int(round(width * OG_HEIGHT / OG_WIDTH))
    crop_h = min(max(crop_h, 1), height)
    top = (height - crop_h) // 2
    cropped = image[top : top + crop_h]
    ys = np.rint(np.linspace(0, cropped.shape[0] - 1, OG_HEIGHT)).astype(int)
    xs = np.rint(np.linspace(0, cropped.shape[1] - 1, OG_WIDTH)).astype(int)
    return cropped[ys][:, xs]


def _basin_canvas(ground: Ground, basin: Basin, dpi: int):
    """Return a 16:9 axes framed on the basin.

    Args:
        ground: Color ground.
        basin: Scene, for its width and height.
        dpi: Raster resolution.

    Returns:
        ``(figure, axes)``.
    """
    fig, ax = _canvas(MASTER_WIDTH, MASTER_HEIGHT, ground, dpi)
    ax.set_xlim(0.0, basin.width)
    ax.set_ylim(0.0, basin.height)
    return fig, ax


def _canvas(width_px: int, height_px: int, ground: Ground, dpi: int):
    """Return a frameless figure of an exact pixel size.

    Args:
        width_px: Width in pixels.
        height_px: Height in pixels.
        ground: Color ground, used as the facecolor.
        dpi: Raster resolution.

    Returns:
        ``(figure, axes)`` with the axes covering the figure.
    """
    fig = plt.figure(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
        facecolor=ground.background,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor=ground.background)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _bodies(ax, basin: Basin, ground: Ground) -> None:
    """Fill each disc.

    Args:
        ax: Target axes.
        basin: Scene outlines.
        ground: Color ground.
    """
    for outline in basin.outlines:
        ax.add_patch(
            Polygon(
                outline,
                closed=True,
                facecolor=ground.obstacle,
                edgecolor="none",
                zorder=1,
            )
        )


def _polyline(ax, points: np.ndarray, color, linewidth: float) -> None:
    """Stroke one polyline.

    Args:
        ax: Target axes.
        points: ``(N, 2)`` polyline.
        color: Any matplotlib color, including an RGBA tuple.
        linewidth: Width in points.
    """
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )


def _tree(ax, nodes: np.ndarray, parents, color) -> None:
    """Stroke the capped tree.

    Args:
        ax: Target axes.
        nodes: ``(N, 2)`` node positions.
        parents: Parent index per node.
        color: RGBA color applied to every kept edge.
    """
    edges = cap_edges(nodes, parents)
    if not edges:
        return
    segments = [(nodes[child], nodes[parent]) for child, parent in edges]
    ax.add_collection(
        LineCollection(
            segments,
            colors=[color],
            linewidths=TREE_PT,
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
    )


def _ribbon(ax, curve: np.ndarray, speed: np.ndarray, color: str) -> None:
    """Stroke *curve* with width taken from a fixed speed band.

    2.5 world units per second is the thin end and 8 is the thick end,
    the same floor and cruise :func:`illustration.web.solve._ribbon_speed`
    uses. Mapping across the sample's own min and max would turn a flat
    cruise into a fake taper.

    Args:
        ax: Target axes.
        curve: ``(N, 2)`` polyline.
        speed: Speed per point, length ``N``.
        color: Stroke color.
    """
    weights = np.clip((speed[:-1] - 2.5) / (8.0 - 2.5), 0.0, 1.0)
    widths = 3.2 + weights * 9.0
    segments = np.stack([curve[:-1], curve[1:]], axis=1)
    ax.add_collection(
        LineCollection(
            segments,
            colors=[color],
            linewidths=widths,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
    )


def _cell(ax, index: Sequence[int], cell: float, color: str) -> None:
    """Fill one grid cell.

    Args:
        ax: Target axes.
        index: ``(i, j)`` cell index.
        cell: Cell edge in world units.
        color: Fill color.
    """
    i, j = int(index[0]), int(index[1])
    ax.add_patch(
        Rectangle(
            (i * cell, j * cell),
            cell,
            cell,
            facecolor=color,
            edgecolor="none",
            zorder=1,
        )
    )


def _band_index(count: int) -> np.ndarray:
    """Return a band id in ``0 .. BANDS-1`` for each expansion step.

    Args:
        count: Number of expanded cells.

    Returns:
        Integer array of length *count*.
    """
    if count == 0:
        return np.zeros(0, dtype=int)
    order = np.arange(count)
    return np.minimum(order * BANDS // count, BANDS - 1)


def _band_color(ground: Ground, band: int) -> str:
    """Blend the A* hue into the ground, early bands quieter.

    Args:
        ground: Color ground.
        band: Band id, 0 earliest.

    Returns:
        Hex color.
    """
    start = 0.22 if ground.name == "dark" else 0.16
    mix = start + (1.0 - start) * (band / (BANDS - 1))
    return _mix(ground.background, ground.astar, mix)


def _mix(origin: str, target: str, amount: float) -> str:
    """Blend two hex colors.

    Args:
        origin: Color at amount 0.
        target: Color at amount 1.
        amount: Blend fraction.

    Returns:
        Hex color.
    """
    a = np.array(_rgb(origin), dtype=float)
    b = np.array(_rgb(target), dtype=float)
    mixed = np.rint((1.0 - amount) * a + amount * b).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _rgb(color: str) -> Tuple[int, int, int]:
    """Parse a ``#rrggbb`` string.

    Args:
        color: Hex color.

    Returns:
        Integer RGB in ``0 .. 255``.
    """
    text = color.lstrip("#")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _rgba(color: str, alpha: float) -> Tuple[float, float, float, float]:
    """Return an RGBA tuple in ``0 .. 1``.

    Args:
        color: Hex color.
        alpha: Opacity.

    Returns:
        ``(r, g, b, a)``.
    """
    red, green, blue = _rgb(color)
    return (red / 255.0, green / 255.0, blue / 255.0, alpha)


def _write(fig, path: Path, dpi: int) -> None:
    """Save *fig* and reject any figure that grew a text artist.

    Args:
        fig: Matplotlib figure.
        path: Destination PNG. Parent directories are created.
        dpi: Raster resolution.

    Raises:
        RuntimeError: If the figure carries a non-empty text artist.
    """
    _reject_type(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _reject_type(fig) -> None:
    """Raise if *fig* has any text to draw.

    Args:
        fig: Matplotlib figure.

    Raises:
        RuntimeError: If a text artist has a non-empty string.
    """
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
        artists.append(ax.title)
        artists.append(ax.xaxis.label)
        artists.append(ax.yaxis.label)
    for artist in artists:
        if artist.get_text().strip():
            raise RuntimeError("web image carries text")
