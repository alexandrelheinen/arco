"""
A* on a 2D Manhattan grid with a large central square obstacle.

Uses 4-connectivity (axis-aligned moves only) and the Euclidean heuristic,
which is admissible on a Manhattan grid (Euclidean <= Manhattan) and guides
A* toward the diagonal rather than producing the L-shaped path that naive
tie-breaking causes.

The result is a staircase path: the planner moves diagonally toward the
goal until the obstacle forces a detour, navigates around the obstacle's
corner that is closest to the start-to-goal straight line, then continues
in staircase fashion to the goal.

Usage
-----
Run interactively (opens a matplotlib window)::

    python tools/examples/astar.py

Save the output image without opening a window::

    python tools/examples/astar.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from arco.mapping import ManhattanGrid
from arco.planning.discrete.astar import AStarPlanner
from arco.tools.logging_config import configure_logging
from arco.tools.viewer.grid import draw_grid

logger = logging.getLogger(__name__)


def build_grid_with_obstacle(
    physical_size: list[float],
    cell_size: float,
    obstacle_fraction: float,
) -> ManhattanGrid:
    """Build a square Manhattan grid with a centered square obstacle.

    Args:
        physical_size: Physical dimensions of the grid in meters ``[rows, cols]``.
        cell_size: Physical size of one cell in meters.  The grid is
            extended to the nearest multiple of *cell_size* when needed.
        obstacle_fraction: Obstacle side length as a fraction of the grid
            size (in cells).

    Returns:
        A :class:`~arco.mapping.grid.manhattan.ManhattanGrid` with the central
        obstacle cells marked as occupied.
    """
    grid = ManhattanGrid(physical_size=physical_size, cell_size=cell_size)
    grid_size = grid.shape[0]
    obs_size = int(grid_size * obstacle_fraction)
    margin = (grid_size - obs_size) // 2
    for r in range(margin, margin + obs_size):
        for c in range(margin, margin + obs_size):
            grid.set_occupied((r, c))
    return grid


def main(cfg: dict, save_path: str | None = None) -> None:
    if save_path is not None:
        matplotlib.use("Agg")

    grid_cfg = cfg.get("grid", {})
    physical_size = [float(x) for x in grid_cfg["physical_size"]]
    cell_size = float(grid_cfg["cell_size"])
    obstacle_fraction = float(grid_cfg["obstacle_fraction"])

    grid = build_grid_with_obstacle(
        physical_size=physical_size,
        cell_size=cell_size,
        obstacle_fraction=obstacle_fraction,
    )
    n = grid.shape[0]
    start = (0, 0)
    goal = (n - 1, n - 1)

    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    obs_size = int(n * obstacle_fraction)
    title = (
        f"A* on {n}×{n} Manhattan grid — central {obs_size}×{obs_size} obstacle\n"
        + (f"Path length: {len(path)} steps" if path else "No path found")
    )

    fig, ax = draw_grid(grid, path, title=title)
    plt.tight_layout()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved Manhattan grid example to %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    import yaml as _yaml

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario", metavar="FILE", help="Path to scenario YAML file."
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window.",
    )
    args = parser.parse_args()
    with open(args.scenario) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    main(_cfg, save_path=args.save)
