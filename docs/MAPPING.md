# Mapping Layer Overview

The mapping layer in ARCO provides spatial data structures and obstacle-query interfaces for representing environments. Different planners operate on different map types depending on their requirements (discrete vs. continuous, graph vs. grid).

## Architecture

The mapping layer is organized into three main categories:

```
src/arco/mapping/
├── __init__.py
├── graph/            ← Graph topology hierarchy
├── grid/             ← Discrete grid structures
├── occupancy.py      ← Continuous-space obstacle interface (abstract)
└── kdtree.py         ← KDTree-based occupancy implementation
```

## Components

### Graph Hierarchy (`arco.mapping.graph`)

A layered hierarchy of graph abstractions for routing and network planning.

#### Class Hierarchy

```
Graph (base.py)
  ↓
WeightedGraph (weighted.py)  ← adds edge weights
  ↓
CartesianGraph (cartesian.py)  ← adds N-D spatial positions
  ↓
RoadGraph (road.py)  ← adds waypoint geometry for road networks
```

Each layer adds specific functionality while maintaining backward compatibility.

#### Implementations

- **Graph** (`graph/base.py`): Basic node-edge topology
  - Adjacency lists and neighbor queries
  - No spatial information

- **WeightedGraph** (`graph/weighted.py`): Graph with edge costs
  - Edge weight storage and queries
  - Used by shortest-path algorithms (A*, Dijkstra)

- **CartesianGraph** (`graph/cartesian.py`): Graph embedded in N-D space
  - Each node has a position (np.ndarray)
  - Euclidean distance heuristics
  - Spatial indexing support

- **RoadGraph** (`graph/road.py`): Road network with waypoint geometry
  - Edges have intermediate waypoints for curved roads
  - Full geometry reconstruction with `full_edge_geometry()`
  - Used for route planning and path smoothing

#### Graph Loading

- **load_road_graph()** (`graph/loader.py`): Load road networks from JSON
  - Parses node positions and edge waypoints
  - See [City Road Network format](#city-road-network) below for the JSON specification

### Grid Structures (`arco.mapping.grid`)

Discrete grid representations for graph-based planners like A*.

#### Implementations

- **ManhattanGrid** (`grid/manhattan.py`): 4-connected grid
  - Neighbors in cardinal directions only (N, S, E, W)
  - Manhattan distance metric ($L_1$)
  - Fast for axis-aligned environments

- **EuclideanGrid** (`grid/euclidean.py`): 8-connected grid
  - Includes diagonal neighbors (NE, NW, SE, SW)
  - Euclidean distance metric ($L_2$)
  - Better path quality with diagonal movement

Both grids inherit from the `Grid` abstract base class (`grid/base.py`).

#### Grid Interface

```python
from arco.mapping import ManhattanGrid, EuclideanGrid
import numpy as np

# Create grid (0 = free, 1 = occupied)
grid_data = np.zeros((100, 100), dtype=np.uint8)
grid_data[40:60, 40:60] = 1  # Add obstacle

# Manhattan grid (4-connected)
manhattan = ManhattanGrid(shape=(100, 100))
manhattan.data = grid_data

# Euclidean grid (8-connected)
euclidean = EuclideanGrid(shape=(100, 100))
euclidean.data = grid_data

# Query neighbors
neighbors = manhattan.neighbors(index=0)
# Use with A* planner
```

### Occupancy Structures (`arco.mapping.occupancy`)

Continuous-space obstacle representations for sampling-based planners like RRT* and SST.

#### Abstract Interface

The `Occupancy` abstract base class (`occupancy.py`) defines two required methods:

```python
class Occupancy(ABC):
    @abstractmethod
    def is_occupied(self, point: np.ndarray) -> bool:
        """Check if a point is in collision."""
        pass

    @abstractmethod
    def nearest_obstacle(self, point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Find distance and position of nearest obstacle."""
        pass
```

#### Implementations

- **KDTreeOccupancy** (`kdtree.py`): Point cloud obstacle representation
  - Uses scipy KDTree for fast nearest-neighbor queries
  - Configurable clearance radius around obstacles
  - Efficient for static point-based obstacles

```python
from arco.mapping import KDTreeOccupancy
import numpy as np

# Define obstacle points
obstacles = np.array([
    [5.0, 5.0],
    [5.0, 6.0],
    [6.0, 5.0],
    [6.0, 6.0]
])

# Create occupancy map with 0.5 unit clearance
occupancy = KDTreeOccupancy(obstacles, clearance=0.5)

# Query collision
point = np.array([5.5, 5.5])
if occupancy.is_occupied(point):
    dist, nearest = occupancy.nearest_obstacle(point)
    print(f"In collision! Nearest obstacle at {nearest}, distance {dist}")
```

## Planner-Map Compatibility

Different planners require different map types:

| Planner | Map Type | Notes |
|---------|----------|-------|
| **A\*** | Grid, WeightedGraph | Requires discrete connectivity |
| **Route Planning** | CartesianGraph, RoadGraph | Requires spatial positions |
| **RRT\*** | Occupancy | Continuous collision checking |
| **SST** | Occupancy | Continuous collision checking |

## Usage Patterns

### Grid-based Planning (A*)

```python
from arco.mapping import EuclideanGrid
from arco.planning import AStar

# Create grid
grid = EuclideanGrid(shape=(100, 100))
grid.data[40:60, 40:60] = 1  # Obstacle

# Plan path
planner = AStar(grid)
path = planner.plan(start=0, goal=9999)
```

### Road Network Planning

```python
from arco.mapping.graph import load_road_graph
from arco.planning.discrete import RouteRouter

# Load road network
graph = load_road_graph("src/arco/tools/map/city.json")

# Plan route
router = RouteRouter(graph)
result = router.plan(
    start_position=np.array([100.0, 200.0]),
    goal_position=np.array([800.0, 900.0])
)

# Get full path with waypoints
path = graph.full_edge_geometry(result.path[0], result.path[1])
```

### Continuous Planning (RRT*, SST)

```python
from arco.mapping import KDTreeOccupancy
from arco.planning.continuous import RRTPlanner

# Create obstacle map
obstacles = np.array([[5, 5], [6, 6]])
occupancy = KDTreeOccupancy(obstacles, clearance=0.5)

# Plan path
planner = RRTPlanner(
    occupancy=occupancy,
    bounds=[(0, 10), (0, 10)]
)
path = planner.plan(start=np.array([1, 1]), goal=np.array([9, 9]))
```

## References

- Cormen, T. H., et al. (2009). Introduction to Algorithms (graph theory).
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths (grid-based planning).

---

## City Road Network

The file `src/arco/tools/map/city.json` is the hand-crafted road network used
as the default environment for the city scenario in both `arcoex` and `arcosim`.

### JSON format

```json
{
  "nodes": [
    {"id": 0, "x": 365, "y": 110},
    ...
  ],
  "edges": [
    {"from": 0, "to": 1, "waypoints": [[395, 118], [425, 102]]},
    {"from": 0, "to": 5},
    ...
  ]
}
```

#### Node object

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | integer | **yes** | Unique node identifier |
| `x` | number | **yes** | X-coordinate in meters |
| `y` | number | **yes** | Y-coordinate in meters |

#### Edge object

| Field | Type | Required | Description |
|---|---|---|---|
| `from` | integer | **yes** | First endpoint node `id` |
| `to` | integer | **yes** | Second endpoint node `id` |
| `waypoints` | array of `[x, y]` | no | Ordered intermediate control points |

Edges are **undirected**: a single entry covers both traversal directions.

### Waypoint convention

Waypoints are stored in the **canonical direction** — from the lower node `id`
to the higher node `id`. The loader reverses them automatically if `from > to`.

`RoadGraph.full_edge_geometry(a, b)` returns waypoints in the correct order for
both `a < b` (forward) and `a > b` (reversed) traversal.

All waypoints in `city.json` are generated with:

```
wp1 = A + (1/3)*(B-A) + d * perp(B-A)
wp2 = A + (2/3)*(B-A) - d * perp(B-A)
```

where `d = 0.09 * |AB|` and `perp(v)` is the left-perpendicular unit vector.

### Network topology

The network has **61 nodes** and **110 edges** arranged as a triangular mesh
with four symmetric holes and four terminal nodes.

| IDs | Description |
|---|---|
| 0–56 | Mesh nodes — hexagonal triangular lattice |
| 57 | Terminal N — (365, 1070) |
| 58 | Terminal E — (1085, 422) |
| 59 | Terminal S — (365, -70) |
| 60 | Terminal W — (−85, 422) |

### Loading the network

```python
from arco.mapping.graph.loader import load_road_graph

graph = load_road_graph("src/arco/tools/map/city.json")
print(len(graph.nodes), "nodes")   # 61
print(len(graph.edges), "edges")   # 110
```

---

*This document reflects the current state of the mapping layer.*
