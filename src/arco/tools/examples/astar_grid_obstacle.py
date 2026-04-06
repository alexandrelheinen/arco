"""
A* on a 2D grid with a large square obstacle in the center.

The grid uses diagonal (Euclidean) connectivity so that A* can cut diagonally
around the obstacle.  A large square obstacle is placed at the center of the
grid and A* finds the shortest diagonal path from the top-left corner to the
bottom-right corner.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/astar_grid_obstacle_example.py

Save the output image without opening a window::

    python tools/astar_grid_obstacle_example.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from arco.mapping import EuclideanGrid
from arco.planning.discrete.astar import AStarPlanner
from arco.tools.config import load_config
from arco.tools.logging_config import configure_logging
from arco.tools.viewer.grid import draw_grid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (loaded from tools/config/grid.yml)
# ---------------------------------------------------------------------------
_cfg = load_config("grid")


def build_grid_with_obstacle(
    physical_size: list[float] = [float(x) for x in _cfg["physical_size"]],
    cell_size: float = float(_cfg["cell_size"]),
    obstacle_fraction: float = float(_cfg["obstacle_fraction"]),
) -> EuclideanGrid:
    """Build a square Euclidean grid with a centered square obstacle.

    Uses :class:`~arco.mapping.grid.euclidean.EuclideanGrid` (diagonal moves
    allowed) so that A* can navigate diagonally around the obstacle.

    Args:
        physical_size: Physical dimensions of the grid in meters ``[rows, cols]``.
        cell_size: Physical size of one cell in meters.  The grid is
            extended to the nearest multiple of *cell_size* when needed.
        obstacle_fraction: Side length of the obstacle expressed as a
            fraction of the grid size (in cells).

    Returns:
        A :class:`~arco.mapping.grid.euclidean.EuclideanGrid` with the central
        obstacle cells marked as occupied.
    """
    grid = EuclideanGrid(physical_size=physical_size, cell_size=cell_size)

    grid_size = grid.shape[0]
    obs_size = int(grid_size * obstacle_fraction)
    margin = (grid_size - obs_size) // 2
    for r in range(margin, margin + obs_size):
        for c in range(margin, margin + obs_size):
            grid.set_occupied((r, c))

    return grid


def main(save_path: str | None = None) -> None:
    if save_path is not None:
        matplotlib.use("Agg")

    grid = build_grid_with_obstacle()
    n = grid.shape[0]
    start = (0, 0)
    goal = (n - 1, n - 1)

    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    obs_size = int(n * float(_cfg["obstacle_fraction"]))
    title = (
        f"A* on {n}×{n} Euclidean grid — central {obs_size}×{obs_size} obstacle\n"
        + (f"Path length: {len(path)} steps" if path else "No path found")
    )

    fig, ax = draw_grid(grid, path, title=title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved grid obstacle example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
