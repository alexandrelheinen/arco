# A* Path Planning

A* is a best-first, graph-based search algorithm that finds the shortest path from a start to a goal node using a heuristic to guide the search.

## Key Reference
- Hart, Nilsson, Raphael (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths.

## Algorithm Overview
- Maintains open and closed sets of nodes
- Uses a cost function: $f(n) = g(n) + h(n)$
    - $g(n)$: cost from start to node $n$
    - $h(n)$: heuristic estimate from $n$ to goal
- Returns optimal path if heuristic is admissible

## Implementation Features
- Grid-based, configurable heuristic (default: Manhattan)
- Returns path as a list of coordinates
- Avoids obstacles (grid cells with value 1)

## Example Usage
```python
import numpy as np
from arco.planning import AStar

grid = np.zeros((5, 5), dtype=int)
grid[2, 1:4] = 1  # Add a wall
astar = AStar(grid)
path = astar.search((0, 0), (4, 4))
print(path)
```

## Resources
- [Planning Algorithms, Ch. 2.3](http://planning.cs.uiuc.edu/node36.html)

---

*This document will be updated as the implementation progresses.*
