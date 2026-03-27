"""
Visualize a 2D Manhattan grid and the shortest A* path from one corner to the opposite.

Parameters (edit these at the top):
    length: float = 100.0  # meters (grid side length)
    resolution: float = 0.5  # meters (voxel size)

This script creates an empty grid, runs A* from (0,0) to (N-1,N-1), and plots the result.

Future: allow user to draw obstacles interactively.
"""

import matplotlib.pyplot as plt
import numpy as np

from arco.mapping import ManhattanGrid
from arco.planning.discrete.astar import AStarPlanner

# --- Parameters ---
length = 100.0  # meters
resolution = 0.5  # meters

# Derived grid size
n = int(length / resolution)
grid = ManhattanGrid((n, n))

start = (0, 0)
goal = (n - 1, n - 1)

# Run A*
planner = AStarPlanner(grid)
path = planner.plan(start, goal)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))

# Show grid (all free)
cmap = plt.get_cmap("Greys")
ax.imshow(
    grid.data.T, cmap=cmap, origin="lower", extent=[0, length, 0, length], alpha=0.3
)

# Plot path
if path:
    path = np.array(path)
    ax.plot(
        path[:, 0] * resolution,
        path[:, 1] * resolution,
        color="red",
        linewidth=2,
        label="A* Path",
    )
    ax.scatter(
        [start[0] * resolution],
        [start[1] * resolution],
        color="green",
        s=80,
        label="Start",
    )
    ax.scatter(
        [goal[0] * resolution], [goal[1] * resolution], color="blue", s=80, label="Goal"
    )
else:
    ax.text(
        length / 2, length / 2, "No path found", color="red", ha="center", va="center"
    )

ax.set_title(f"A* on {n}x{n} Manhattan Grid (empty)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_xlim(0, length)
ax.set_ylim(0, length)
ax.legend()
plt.tight_layout()
plt.show()
