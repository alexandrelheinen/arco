# ARCO: Autonomous Routing, Control, and Observation

<img src="docs/images/arco.svg" alt="FRET Logo" width="200" align="left">

My career has been built on autonomy systems, but most of that work lives inside proprietary codebases. The algorithms I used daily are not things I can share from those jobs.

ARCO is my open repo: a place to implement these methods from scratch, document the reasoning behind them, and build a reference I can actually point to.

This is also a Python project by choice. My production work has been C++ and embedded systems. ARCO is where I practice writing clean, well-tested Python for scientific computing: numpy, scipy, and eventually ROS 2 integration.

## Modules

- The mapping layer handles spatial representation.
- The planning layer searches or samples for feasible paths through that representation
- The guidance layer tracks the resulting path with feedback controllers (pure pursuit, PID, MPC).

## Architecture

A planning algorithm (*planner*) operates within a *graph*, which serves as the fundamental mapping structure for the environment. Algorithms such as A* and RRT* treat this graph as a primary argument, focusing on how the structure is handled and explored.

The architecture distinguishes between two primary graph types:
* **Manhattan Grid**: A simple side-only structure restricted to axial movements. Implement the $L_1$ (Manhattan) distance between nodes.
* **Euclidean Grid**: A diagonal structure that allows for diagonal connections. Implement the $L_2$ (Euclidean) distance between nodes.

The guidance layer is applied subsequent to the initial search. This layer handles the interpolation of output data and the generation of movement primitives used for exploration, particularly within the RRT family of algorithms.

The mapping layer provides spatial representations for planning algorithms. Two base classes are provided:

- **Grid**: The already described N-dimensional grid for discrete planners (A*, D*, etc), both Manhattan and Euclidean
- **Occupancy**: An abstract base for continuous occupancy maps (for RRT, SST, etc). Subclasses may use point clouds, kd-trees, or other spatial data structures. Provides a unified interface for obstacle queries in continuous space.

Planning algorithms are designed to accept a map object as an argument, allowing them to work with any compatible map type. Grid-based planners (A*, D*) require a Grid; sampling-based planners (RRT, SST) require an Occupancy or compatible subclass.

The guidance layer acts as a bridge between abstract graph nodes and physical constraints:

1.  **Exploration Primitives**: In RRT-based architectures, the graph is grown dynamically. The steering function defines primitives (e.g., Dubins paths) to ensure the graph only contains kinematically reachable states.
2.  **Interpolation**: Raw output from a graph search is typically a discrete sequence of nodes. The guidance layer applies B-splines or shortcutting algorithms to transform these steps into a continuous trajectory.

## Arquitetura (Diagrama de Pacotes)

![Package diagram](docs/pyreverse/packages.png)

## File structure

```
.
├── docs
│   └── images
├── src
│   └── arco
│       ├── guidance
│       ├── mapping
│       └── planning
└── tests
    └── planning
```

## Coding Guidelines

See [docs/guidelines.md](docs/guidelines.md) for the complete, authoritative coding conventions (file structure, naming, docstrings, formatting, typing, and testing).

## Mapping

The mapping layer provides spatial representations for planning algorithms. It includes:

- **ManhattanGrid**: An axis-aligned grid for discrete planners, using $L_1$ (Manhattan) distance and only side neighbors.
- **EuclideanGrid**: A diagonal grid for discrete planners, using $L_2$ (Euclidean) distance and including diagonal neighbors.
- **Occupancy**: An abstract base for continuous occupancy maps (for RRT, SST, etc). Subclasses may use point clouds, kd-trees, or other spatial data structures. Provides a unified interface for obstacle queries in continuous space.

Each map type is a passive data structure. Maps are designed to be used by planners, not to perform planning themselves. All map classes are nouns, following the naming conventions.

### Planning

Graph search and sampling-based methods for finding feasible paths through a represented environment.

| Algorithm | Status | Notes |
|-----------|--------|-------|
| A\* | ✅ Done | Grid-based, configurable heuristics |
| D\* | 🔜 Next | Dynamic replanning for changing environments |
| RRT | ⬜ Planned | Sampling-based, for continuous state spaces |
| RRT\* | ⬜ Planned | Asymptotically optimal variant |
| SST | ⬜ Planned | Stable Sparse RRT for kinodynamic planning |


### Guidance

The guidance layer bridges the gap between discrete plans and physical execution. It includes:

1. **Exploration Primitives**: Classes that define motion primitives or steering functions (e.g., Dubins, Reeds-Shepp) for kinodynamic feasibility, especially in sampling-based planners like RRT.
2. **Interpolation**: Classes that convert discrete node sequences into continuous trajectories, using methods such as B-splines or shortcutting algorithms.
3. **Controllers**: Feedback controllers (e.g., pure pursuit, PID, MPC) that track the planned path and generate control commands for the system.

All guidance classes are nouns, and are designed to be modular and composable with planners and maps.

## Installation

```bash
git clone https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

Python 3.10 or later.

## Development

```bash
pytest tests/
black --target-version py312 src/ tests/
isort src/ tests/
```

## References

Theory notes for each algorithm are in `docs/`. Key references:

- Hart, Nilsson, Raphael (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths.*
- Stentz (1994). *Optimal and Efficient Path Planning for Partially-Known Environments.*
- LaValle (1998). *Rapidly-Exploring Random Trees: A New Tool for Path Planning.*
- LaValle (2006). *Planning Algorithms.* Cambridge University Press. [planning.cs.uiuc.edu](http://planning.cs.uiuc.edu).
- Karaman, Frazzoli (2011). *Sampling-based Algorithms for Optimal Motion Planning.*
- Li et al. (2016). *Asymptotically Optimal Sampling-based Kinodynamic Planning.*

## License

MIT License. See [LICENSE](LICENSE).
