#!/usr/bin/env python
"""PPP robot 3D planning pipeline: obstacle placement → A* → visualization.

Demonstrates a complete 3D planning pipeline for a PPP (Prismatic-Prismatic-
Prismatic) Cartesian robot:

1. **3D workspace** — :class:`~arco.mapping.grid.euclidean.EuclideanGrid`
   creates a 3-D grid representing the robot's box-shaped work volume.
2. **Obstacle placement** — Box-shaped obstacles of varying heights are
   placed on the ground plane (z=0). Obstacles are pre-inflated to account
   for end-effector volume.
3. **Path planning** — :class:`~arco.planning.discrete.astar.AStarPlanner`
   finds a collision-free path from one corner to the opposite corner.
4. **3D visualization** — Obstacles are rendered as semi-transparent voxels,
   and the planned path is shown as a line in 3D space.

The PPP robot has three independent acceleration-controlled axes (x, y, z)
and operates within a configurable box-shaped workspace. Its mounting
geometry prevents passing beneath obstacles, so ground-level boxes represent
the full blocked volume.

Usage
-----
Run interactively (opens a Matplotlib window)::

    python tools/examples/ppp_pipeline.py

Save to file without opening a window (headless / CI mode)::

    python tools/examples/ppp_pipeline.py --save path/to/output.png

Reference
---------
https://github.com/alexandrelheinen/arco
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the package importable when running the script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logging_config import configure_logging
from viewer.grid3d import draw_grid_3d

from arco.guidance.ppp import PPPRobot
from arco.mapping import EuclideanGrid
from arco.planning.discrete.astar import AStarPlanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Workspace dimensions (metres)
WORK_VOLUME = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
CELL_SIZE = 0.5  # metres per cell

# Obstacles: each is [x_min, x_max, y_min, y_max, height_m]
# All obstacles sit on ground plane (z=0) and extend upward
# Pre-inflated to account for end-effector volume
OBSTACLES = [
    [2.0, 4.0, 2.0, 4.0, 3.0],  # Central pillar, 3m tall
    [6.0, 7.0, 1.0, 3.0, 5.0],  # Tall narrow obstacle
    [1.0, 3.0, 6.0, 8.0, 2.0],  # Low wide obstacle
    [7.0, 9.0, 6.0, 9.0, 4.0],  # Corner obstacle
]

# Start and goal positions (cell indices)
START = (0, 0, 0)  # Bottom-left-front corner, ground level
GOAL = (19, 19, 0)  # Top-right-front corner, ground level


# ---------------------------------------------------------------------------
# Helper: place box-shaped obstacles on grid
# ---------------------------------------------------------------------------


def place_obstacles(grid: EuclideanGrid, obstacles: list[list[float]]) -> None:
    """Mark grid cells as occupied for each box-shaped obstacle.

    Obstacles are specified in physical coordinates and converted to
    cell indices. Each obstacle extends from z=0 up to its specified height.

    Args:
        grid: 3-D Euclidean grid to modify.
        obstacles: List of obstacles, each defined as
            ``[x_min, x_max, y_min, y_max, height]`` in metres.
    """
    for x_min, x_max, y_min, y_max, height in obstacles:
        # Convert physical bounds to cell indices
        ix_min = int(np.floor(x_min / grid.cell_size))
        ix_max = int(np.ceil(x_max / grid.cell_size))
        iy_min = int(np.floor(y_min / grid.cell_size))
        iy_max = int(np.ceil(y_max / grid.cell_size))
        iz_max = int(np.ceil(height / grid.cell_size))

        # Mark all cells in the box as occupied
        for ix in range(ix_min, ix_max):
            for iy in range(iy_min, iy_max):
                for iz in range(0, iz_max):
                    if (
                        0 <= ix < grid.shape[0]
                        and 0 <= iy < grid.shape[1]
                        and 0 <= iz < grid.shape[2]
                    ):
                        grid.set_occupied((ix, iy, iz))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(save_path: str | None = None) -> None:
    """Run the PPP 3D planning pipeline and render results.

    Args:
        save_path: File path for saving the figure. When *None* the figure
            is shown interactively.
    """
    if save_path is not None:
        matplotlib.use("Agg")

    logger.info("=" * 60)
    logger.info("PPP Robot 3D Planning Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build 3D workspace grid
    # ------------------------------------------------------------------
    x_size = WORK_VOLUME[0][1] - WORK_VOLUME[0][0]
    y_size = WORK_VOLUME[1][1] - WORK_VOLUME[1][0]
    z_size = WORK_VOLUME[2][1] - WORK_VOLUME[2][0]

    grid = EuclideanGrid(size_m=[x_size, y_size, z_size], cell_size=CELL_SIZE)
    logger.info(
        "[1] Created 3D grid: %d×%d×%d cells (%.1f m × %.1f m × %.1f m)",
        grid.shape[0],
        grid.shape[1],
        grid.shape[2],
        grid.size_m[0],
        grid.size_m[1],
        grid.size_m[2],
    )

    # ------------------------------------------------------------------
    # 2. Place obstacles
    # ------------------------------------------------------------------
    place_obstacles(grid, OBSTACLES)
    occupied_count = np.sum(grid.data == 1)
    logger.info(
        "[2] Placed %d obstacles (%d occupied cells)",
        len(OBSTACLES),
        occupied_count,
    )

    # ------------------------------------------------------------------
    # 3. Plan path with A*
    # ------------------------------------------------------------------
    planner = AStarPlanner(grid)
    path = planner.plan(START, GOAL)

    if path is None:
        logger.error("No path found from %s to %s", START, GOAL)
        logger.info(
            "    Try adjusting obstacle placement or start/goal positions."
        )
        return

    logger.info("[3] A* path: %d steps (%s → %s)", len(path), START, GOAL)

    # ------------------------------------------------------------------
    # 4. Visualize in 3D
    # ------------------------------------------------------------------
    fig, ax = draw_grid_3d(
        grid,
        path,
        title=f"PPP Robot — 3D Planning\n"
        f"{grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]} grid, "
        f"{len(OBSTACLES)} obstacles, path: {len(path)} steps",
        figsize=(12, 10),
    )

    # Adjust viewing angle for better perspective
    ax.view_init(elev=20, azim=45)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved → %s", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to PATH instead of opening an interactive window. "
        "Accepts any .png or .pdf path.",
    )
    args = parser.parse_args()
    main(save_path=args.save)
