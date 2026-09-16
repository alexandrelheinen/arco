"""World: the bespoke scene the gallery plates are planned in.

The demo scenarios under ``map/`` are built for the ``arcosim`` renderer:
city blocks, gantry bays, SCARA pillars.  They read as engineering, not
as an illustration.  This module defines one purpose-built 16:9 basin of
organic obstacle bodies instead, chosen so that planner output curves
around large smooth shapes and reads well at poster size — while still
being an ordinary :class:`~arco.mapping.KDTreeOccupancy` and an ordinary
:class:`~arco.mapping.EuclideanGrid`, planned by the shipped planners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

from arco.mapping import EuclideanGrid, KDTreeOccupancy

# (centre x, centre y, mean radius, shape seed) per obstacle body.
BODY_SPECS: Tuple[Tuple[float, float, float, int], ...] = (
    (22.0, 46.0, 14.0, 11),
    (56.0, 72.0, 13.0, 23),
    (96.0, 76.0, 12.0, 31),
    (128.0, 82.0, 10.0, 47),
    (60.0, 38.0, 15.0, 53),
    (99.0, 44.0, 13.0, 61),
    (134.0, 40.0, 12.0, 71),
    (86.0, 11.0, 12.0, 83),
    (128.0, 13.0, 11.0, 97),
    (18.0, 79.0, 10.0, 101),
    (153.0, 11.0, 9.0, 107),
)

LOBE_AMPLITUDES: Tuple[float, ...] = (0.20, 0.13, 0.08)


def body_outline(
    centre_x: float,
    centre_y: float,
    radius: float,
    seed: int,
    vertex_count: int = 240,
) -> np.ndarray:
    """Return one organic obstacle outline as a closed polygon.

    The radius is modulated by a short sum of cosines, which gives a
    rounded body with no straight edges — planner curves hugging it stay
    smooth instead of picking up the staircase of a box world.

    Args:
        centre_x: Body centre x in world units.
        centre_y: Body centre y in world units.
        radius: Mean radius in world units.
        seed: Seed selecting the lobe phases, so a body is reproducible.
        vertex_count: Number of outline vertices.

    Returns:
        ``(vertex_count, 2)`` array of outline points.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(LOBE_AMPLITUDES))
    angle = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
    modulation = np.ones_like(angle)
    for index, amplitude in enumerate(LOBE_AMPLITUDES):
        modulation += amplitude * np.cos((index + 2) * angle + phases[index])
    r = radius * modulation
    return np.column_stack(
        [centre_x + r * np.cos(angle), centre_y + r * np.sin(angle)]
    )


@dataclass(frozen=True)
class World:
    """A planning scene plus the geometry needed to draw it.

    Attributes:
        width: Scene width in world units.
        height: Scene height in world units.
        outlines: Obstacle body outlines, used for rendering only.
        occupancy: Continuous map handed to RRT* / SST / pruner / optimizer.
        grid: Discrete map handed to A*.
        start: Start position in world units.
        goal: Goal position in world units.
        clearance: Inflation radius applied to the obstacle point cloud.
    """

    width: float
    height: float
    outlines: List[np.ndarray]
    occupancy: KDTreeOccupancy
    grid: EuclideanGrid
    start: np.ndarray
    goal: np.ndarray
    clearance: float

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Return sampling bounds in the form the planners expect."""
        return [(0.0, self.width), (0.0, self.height)]

    @property
    def start_cell(self) -> Tuple[int, int]:
        """Return the start position as grid indices."""
        return self._cell(self.start)

    @property
    def goal_cell(self) -> Tuple[int, int]:
        """Return the goal position as grid indices."""
        return self._cell(self.goal)

    def _cell(self, position: np.ndarray) -> Tuple[int, int]:
        """Return the grid indices containing *position*.

        Args:
            position: World-unit position.

        Returns:
            ``(i, j)`` grid indices.
        """
        size = float(self.grid.cell_size)
        return (int(round(position[0] / size)), int(round(position[1] / size)))


def build_world(
    width: float = 160.0,
    height: float = 90.0,
    clearance: float = 2.4,
    cell_size: float = 1.0,
    fill_step: float = 1.5,
    specs: Sequence[Tuple[float, float, float, int]] = BODY_SPECS,
) -> World:
    """Build the gallery scene: organic bodies, occupancy and grid.

    Args:
        width: Scene width in world units (16:9 against *height*).
        height: Scene height in world units.
        clearance: Obstacle inflation radius for continuous planning.
        cell_size: Grid resolution for A*.
        fill_step: Spacing of the interior points that make each body
            solid to the KD-tree, rather than a hollow shell.
        specs: Obstacle body specifications.

    Returns:
        The assembled :class:`World`.
    """
    outlines = [body_outline(*spec) for spec in specs]
    cloud = np.vstack(
        [_solid_points(outline, fill_step) for outline in outlines]
    )
    occupancy = KDTreeOccupancy(cloud, clearance=clearance)

    shape = (int(round(width / cell_size)), int(round(height / cell_size)))
    grid = EuclideanGrid(shape=shape, cell_size=cell_size)
    grid.data = _rasterize(outlines, occupancy, shape, cell_size, clearance)

    return World(
        width=width,
        height=height,
        outlines=outlines,
        occupancy=occupancy,
        grid=grid,
        start=np.array([26.0, 13.0]),
        goal=np.array([147.0, 70.0]),
        clearance=clearance,
    )


def _solid_points(outline: np.ndarray, step: float) -> np.ndarray:
    """Return outline vertices plus an interior lattice of points.

    Args:
        outline: Closed ``(N, 2)`` body outline.
        step: Interior lattice spacing in world units.

    Returns:
        ``(M, 2)`` point cloud covering the body.
    """
    path = MplPath(outline)
    low, high = outline.min(axis=0), outline.max(axis=0)
    gx, gy = np.meshgrid(
        np.arange(low[0], high[0], step), np.arange(low[1], high[1], step)
    )
    candidates = np.column_stack([gx.ravel(), gy.ravel()])
    return np.vstack([outline, candidates[path.contains_points(candidates)]])


def _rasterize(
    outlines: Sequence[np.ndarray],
    occupancy: KDTreeOccupancy,
    shape: Tuple[int, int],
    cell_size: float,
    clearance: float,
) -> np.ndarray:
    """Return the occupancy raster A* searches over.

    A cell is blocked when its centre lies inside a body or within the
    same clearance radius the continuous planners respect, so the two
    map families describe one world rather than two.

    Args:
        outlines: Obstacle body outlines.
        occupancy: Continuous map used for the clearance query.
        shape: Grid shape ``(nx, ny)``.
        cell_size: Grid resolution in world units.
        clearance: Inflation radius in world units.

    Returns:
        ``shape``-shaped ``uint8`` array, 1 where blocked.
    """
    gx, gy = np.meshgrid(
        np.arange(shape[0]) * cell_size,
        np.arange(shape[1]) * cell_size,
        indexing="ij",
    )
    query = np.column_stack([gx.ravel(), gy.ravel()])
    blocked = occupancy.query_distances(query) < clearance
    for outline in outlines:
        blocked |= MplPath(outline).contains_points(query)
    return blocked.reshape(shape).astype(np.uint8)
