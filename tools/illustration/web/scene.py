"""The two small scenes the web plates are planned in.

The gallery basin is a field of bodies, which reads as texture once the
image is a card. Here the basin has three discs and the flood grid has
two blocks. Both are ordinary occupancy maps: a
:class:`~arco.mapping.KDTreeOccupancy` for the sampling planners and a
:class:`~arco.mapping.EuclideanGrid` for A*.

Body centers sit inside the frame that survives a 1200×630 center crop
of a 1600×900 master, so the Open Graph crop does not cut a disc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from arco.mapping import EuclideanGrid, KDTreeOccupancy

WIDTH = 160.0
HEIGHT = 90.0
CLEARANCE = 2.2

# (centre x, centre y, radius). Staggered so a left-to-right line cannot
# stay straight, and inset so a 1.91:1 center crop keeps every disc.
BASIN_BODIES: Tuple[Tuple[float, float, float], ...] = (
    (50.0, 54.0, 14.0),
    (92.0, 34.0, 13.0),
    (126.0, 58.0, 11.0),
)
BASIN_START = np.array([16.0, 44.0])
BASIN_GOAL = np.array([146.0, 42.0])

FLOOD_SHAPE = (32, 18)
FLOOD_CELL = 1.0
# (i0, i1, j0, j1), half-open. Two bars with opposite gaps, so A* has
# to flood around both and the bands stay visible.
FLOOD_BLOCKS: Tuple[Tuple[int, int, int, int], ...] = (
    (10, 14, 0, 11),
    (20, 24, 8, 18),
)
FLOOD_START = (1, 2)
FLOOD_GOAL = (30, 15)


@dataclass(frozen=True)
class Basin:
    """Continuous scene shared by the path plates.

    Attributes:
        width: Scene width in world units.
        height: Scene height in world units.
        outlines: Disc outlines, one ``(N, 2)`` array per body.
        occupancy: Map handed to RRT*, SST, the pruner and the optimizer.
        start: Start position in world units.
        goal: Goal position in world units.
        clearance: Inflation radius of *occupancy*.
    """

    width: float
    height: float
    outlines: List[np.ndarray]
    occupancy: KDTreeOccupancy
    start: np.ndarray
    goal: np.ndarray
    clearance: float

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Return sampling bounds in the form the planners expect."""
        return [(0.0, self.width), (0.0, self.height)]


@dataclass(frozen=True)
class Flood:
    """Discrete scene for the A* plate.

    Attributes:
        grid: Euclidean grid A* searches.
        blocks: Half-open rectangles ``(i0, i1, j0, j1)`` that are blocked.
        start: Start cell ``(i, j)``.
        goal: Goal cell ``(i, j)``.
        cell_size: Edge length of one cell in world units.
    """

    grid: EuclideanGrid
    blocks: Tuple[Tuple[int, int, int, int], ...]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    cell_size: float


def circle_outline(
    centre_x: float,
    centre_y: float,
    radius: float,
    vertex_count: int = 96,
) -> np.ndarray:
    """Return a closed disc outline.

    Args:
        centre_x: Centre x in world units.
        centre_y: Centre y in world units.
        radius: Radius in world units.
        vertex_count: Number of outline vertices.

    Returns:
        ``(vertex_count, 2)`` array. The first point is not repeated.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
    return np.column_stack(
        [
            centre_x + radius * np.cos(angle),
            centre_y + radius * np.sin(angle),
        ]
    )


def build_basin(
    bodies: Sequence[Tuple[float, float, float]] = BASIN_BODIES,
    clearance: float = CLEARANCE,
    fill_step: float = 1.6,
) -> Basin:
    """Build the three-disc basin.

    Args:
        bodies: ``(centre x, centre y, radius)`` per disc.
        clearance: Inflation radius for the occupancy map.
        fill_step: Spacing of the interior points that make each disc
            solid to the KD-tree.

    Returns:
        The assembled :class:`Basin`.
    """
    outlines = [circle_outline(*body) for body in bodies]
    cloud = np.vstack(
        [_solid_points(outline, fill_step) for outline in outlines]
    )
    occupancy = KDTreeOccupancy(cloud, clearance=clearance)
    return Basin(
        width=WIDTH,
        height=HEIGHT,
        outlines=outlines,
        occupancy=occupancy,
        start=np.array(BASIN_START, dtype=float),
        goal=np.array(BASIN_GOAL, dtype=float),
        clearance=clearance,
    )


def build_flood(
    shape: Tuple[int, int] = FLOOD_SHAPE,
    cell_size: float = FLOOD_CELL,
    blocks: Sequence[Tuple[int, int, int, int]] = FLOOD_BLOCKS,
) -> Flood:
    """Build the two-block grid A* floods.

    Args:
        shape: Grid shape ``(nx, ny)``.
        cell_size: Cell edge in world units.
        blocks: Half-open blocked rectangles.

    Returns:
        The assembled :class:`Flood`.
    """
    grid = EuclideanGrid(shape=shape, cell_size=cell_size)
    blocked = np.zeros(shape, dtype=np.uint8)
    for i0, i1, j0, j1 in blocks:
        blocked[i0:i1, j0:j1] = 1
    grid.data = blocked
    return Flood(
        grid=grid,
        blocks=tuple(blocks),
        start=FLOOD_START,
        goal=FLOOD_GOAL,
        cell_size=cell_size,
    )


def _solid_points(outline: np.ndarray, step: float) -> np.ndarray:
    """Return outline vertices plus an interior lattice.

    Args:
        outline: Closed ``(N, 2)`` outline.
        step: Interior lattice spacing in world units.

    Returns:
        ``(M, 2)`` point cloud covering the disc.
    """
    from matplotlib.path import Path as MplPath

    path = MplPath(outline)
    low, high = outline.min(axis=0), outline.max(axis=0)
    gx, gy = np.meshgrid(
        np.arange(low[0], high[0], step),
        np.arange(low[1], high[1], step),
    )
    candidates = np.column_stack([gx.ravel(), gy.ravel()])
    inside = candidates[path.contains_points(candidates)]
    if len(inside) == 0:
        return outline
    return np.vstack([outline, inside])
