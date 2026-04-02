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

    python tools/astar_manhattan_example.py

Save the output image without opening a window::

    python tools/astar_manhattan_example.py --save path/to/output.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # expose tools/viewer

import matplotlib
import matplotlib.pyplot as plt
from viewer.grid import draw_grid

from arco.mapping import ManhattanGrid
from arco.planning.discrete.astar import AStarPlanner

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
GRID_SIZE = 51
OBSTACLE_FRACTION = 0.4


def build_grid_with_obstacle(
    grid_size: int = GRID_SIZE,
    obstacle_fraction: float = OBSTACLE_FRACTION,
) -> ManhattanGrid:
    """Build a square Manhattan grid with a centred square obstacle.

    Args:
        grid_size: Side length of the grid in cells.
        obstacle_fraction: Obstacle side length as a fraction of *grid_size*.

    Returns:
        A :class:`~arco.mapping.grid.manhattan.ManhattanGrid` with the central
        obstacle cells marked as occupied.
    """
    grid = ManhattanGrid((grid_size, grid_size))
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
    n = GRID_SIZE
    start = (0, 0)
    goal = (n - 1, n - 1)

    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    obs_size = int(n * OBSTACLE_FRACTION)
    title = (
        f"A* on {n}×{n} Manhattan grid — central {obs_size}×{obs_size} obstacle\n"
        + (f"Path length: {len(path)} steps" if path else "No path found")
    )

    fig, ax = draw_grid(grid, path, title=title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved Manhattan grid example to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
