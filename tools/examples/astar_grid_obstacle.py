"""
A* on a 2D grid with a large square obstacle in the centre.

The grid uses diagonal (Euclidean) connectivity so that A* can cut diagonally
around the obstacle.  A large square obstacle is placed at the centre of the
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
import sys

# Make the package importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..")
)  # expose tools/viewer and tools/config

import matplotlib
import matplotlib.pyplot as plt
from logging_config import configure_logging
from viewer.grid import draw_grid

from arco.mapping import EuclideanGrid
from arco.planning.discrete.astar import AStarPlanner
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (loaded from tools/config/grid.yml)
# ---------------------------------------------------------------------------
_cfg = load_config("grid")
SIZE_M: list[float] = [float(x) for x in _cfg["size_m"]]
CELL_SIZE: float = float(_cfg["cell_size"])
OBSTACLE_FRACTION: float = float(_cfg["obstacle_fraction"])


def build_grid_with_obstacle(
    size_m: list[float] = SIZE_M,
    cell_size: float = CELL_SIZE,
    obstacle_fraction: float = OBSTACLE_FRACTION,
) -> EuclideanGrid:
    """Build a square Euclidean grid with a centred square obstacle.

    Uses :class:`~arco.mapping.grid.euclidean.EuclideanGrid` (diagonal moves
    allowed) so that A* can navigate diagonally around the obstacle.

    Args:
        size_m: Physical dimensions of the grid in metres ``[rows, cols]``.
        cell_size: Physical size of one cell in metres.  The grid is
            extended to the nearest multiple of *cell_size* when needed.
        obstacle_fraction: Side length of the obstacle expressed as a
            fraction of the grid size (in cells).

    Returns:
        A :class:`~arco.mapping.grid.euclidean.EuclideanGrid` with the central
        obstacle cells marked as occupied.
    """
    grid = EuclideanGrid(size_m=size_m, cell_size=cell_size)

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

    obs_size = int(n * OBSTACLE_FRACTION)
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
