# ARCO Tool Scripts

This folder contains utility scripts for visualising and experimenting with ARCO's
planning and mapping modules.

---

## Examples

### `astar_graph_example.py`

Runs A\* on a randomly generated weighted graph with **30 nodes** arbitrarily
placed in a 100 × 100 unit plane.  Nodes closer than a configurable radius are
connected by edges (weighted by Euclidean distance).  The start and goal are
chosen as the two farthest-apart nodes.

```bash
# interactive window
python tools/astar_graph_example.py

# save to file (headless)
python tools/astar_graph_example.py --save docs/examples/astar_graph.png
```

### `astar_grid_obstacle_example.py`

Runs A\* on a **51 × 51 Manhattan grid** with a large square obstacle placed
at the center (~40 % of the grid size).  The planner finds a path from the
top-left corner to the bottom-right corner around the obstacle.

```bash
# interactive window
python tools/astar_grid_obstacle_example.py

# save to file (headless)
python tools/astar_grid_obstacle_example.py --save docs/examples/astar_grid_obstacle.png
```

---

## Visualization modules (`visualization/`)

| Module | Purpose |
|---|---|
| `visualization/graph_viewer.py` | `draw_graph()` – renders a `WeightedGraph` with optional path highlighting |
| `visualization/grid_viewer.py` | `draw_grid()` – renders a 2-D `Grid` with optional path highlighting |

Both viewers accept per-node / per-cell color overrides so that a planner can
color explored nodes, frontier cells, etc. in any color it chooses.

---

## Setup

Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

