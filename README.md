# ARCO: Autonomous Routing, Control, and Observation

<img src="docs/images/arco.svg" alt="ARCO Logo" width="200" align="left">

ARCO is a Python library for autonomous navigation building blocks:

- Mapping representations for discrete and continuous spaces
- Planning algorithms for graph search and sampling-based exploration
- Guidance components for interpolation, motion primitives, and control

The project emphasizes clear architecture, testability, and documented algorithmic decisions.

## Documentation Index

- [Main docs index](docs/README.md)
- [Coding guidelines (authoritative)](docs/guidelines.md)
- [Contributing guide](CONTRIBUTING.md)
- [Mapping layer overview](docs/MAPPING.md)
- [Planning layer overview](docs/PLANNING.md)
  - [A* algorithm notes](docs/planning_astar.md)
  - [RRT* algorithm notes](docs/planning_rrt.md)
  - [SST algorithm notes](docs/planning_sst.md)
  - [D* Lite notes](docs/planning_dstar.md)
- [Guidance layer overview](docs/GUIDANCE.md)
- [Route planning benchmarks](docs/route_planning_benchmarks.md)
- [Horse auto-follow system design](docs/horse_auto_follow.md)
- [City network descriptor](docs/city_network.md)
- [Roadmap](docs/ROADMAP.md)

## Architecture

A planner operates on a map object:

- Discrete planners (A*, route planning) operate on Grid or Graph structures
- Continuous planners (RRT*, SST) operate on Occupancy structures

Core map families:

- Manhattan Grid: axis-aligned neighbors with Manhattan metric ($L_1$)
- Euclidean Grid: diagonal-capable neighbors with Euclidean metric ($L_2$)
- Graph hierarchy: Graph → WeightedGraph → CartesianGraph → RoadGraph
- Occupancy: abstract continuous-space obstacle-query interface (KDTreeOccupancy)

Guidance is applied after planning:

- Exploration primitives: kinematic steering constraints for graph growth
- Interpolation: conversion of discrete plans into smooth trajectories
- Controllers: path tracking and control law generation

## Pipeline

The full ARCO processing pipeline runs as a sequence of independent steps
(see [docs/PIPELINE.md](docs/PIPELINE.md)):

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   MAPPING    │     │   PLANNING   │     │ OPTIMIZATION │     │  SIMULATION  │
│              │     │              │     │              │     │              │
│ config.yml   │────▶│occupancy.json│────▶│  path.json   │────▶│trajectory    │
│ obstacles    │     │ RRT* / SST   │     │ pruner +     │     │ renderer +   │
│              │     │ C-space      │     │ optimizer    │     │ controller   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
 occupancy.json         path.json        trajectory.json     video / metrics

              ◀──── telemetry side-channel (stop criteria, iter count) ────▶
                     loading screen polls arco_planner_telemetry.json
```

> **IPC note**: telemetry currently uses a JSON temp-file
> (`arco.planning.continuous.telemetry`).  A proper pub/sub middleware is
> planned — see [docs/ROADMAP.md](docs/ROADMAP.md) §"IPC & Telemetry Middleware".

## Modules

- **Mapping**: Spatial data structures (grids, graphs, occupancy) and obstacle-query interfaces
- **Planning**: Path search (A*, route planning) and sampling methods (RRT*, SST)
- **Guidance**: Trajectory shaping (interpolation, primitives) and feedback control (PID, Pure Pursuit, MPC)

## Current Algorithm Status

| Algorithm | Status | Notes |
|-----------|--------|-------|
| A* | ✅ Done | Grid and graph-based, configurable heuristics |
| Route Planning | ✅ Done | A* integration for road networks with waypoint smoothing |
| RRT* | ✅ Done | Asymptotically optimal sampling-based planner |
| SST | ✅ Done | Stable Sparse Trees for kinodynamic planning |
| D* Lite | ⏸️ Stub | Dynamic replanning (API exists, not implemented) |

## Repository Layout

```text
.
├── docs                   ← algorithm notes and design docs
├── src/arco
│   ├── guidance
│   │   ├── control        ← feedback controllers (PID, Pure Pursuit, MPC)
│   │   ├── interpolation  ← path smoothing (B-spline) and trajectory generation
│   │   ├── primitive      ← kinematic exploration primitives (Dubins)
│   │   └── vehicle.py     ← vehicle kinematic models
│   ├── mapping
│   │   ├── graph          ← graph topology hierarchy (weighted, cartesian, road)
│   │   ├── grid           ← discrete grid structures (Manhattan, Euclidean)
│   │   ├── occupancy.py   ← continuous-space obstacle interface
│   │   └── kdtree.py      ← KDTree-based occupancy implementation
│   └── planning
│       ├── discrete       ← graph-search planners (A*, route planning)
│       └── continuous     ← sampling-based planners (RRT*, SST)
├── tests                  ← mirrored test layout
└── tools                  ← examples and visualization utilities
```

## Installation

```bash
git clone https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

The ARCO _config dir_ (directory containing the system configurations/YAML files) can be customized by setting the `ARCO_CONFIG_DIR` envionment variable.

```sh
export ARCO_CONFIG_DIR="<path_to_custom_root_dir>"
```

> ARCO Requires Python 3.10+

## Development

### Run tests

```bash
pytest tests/ -v
```

### Format code

```bash
python -m black --target-version py312 --line-length 79 src/ tools/
python -m isort --line-length 79 src/ tools/
```

### Local examples

```bash
python tools/examples/astar_graph.py
python tools/examples/astar_grid_obstacle.py
python tools/examples/astar_manhattan.py
python tools/examples/route_planning.py
python tools/examples/rrt_planning.py
python tools/examples/sst_planning.py
```

## CI and Merge Policy

GitHub Actions workflows run for pull requests and can be configured as required checks for merge protection on main.

Recommended required checks:

- Tests / Run unit tests
- Generate Images / generate-images

## Contributing

Before contributing, follow [CONTRIBUTING.md](CONTRIBUTING.md) and the conventions in [docs/guidelines.md](docs/guidelines.md).

## References

Theory notes are under [docs](docs/). Core references:

- Hart, Nilsson, Raphael (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
- Stentz (1994). Optimal and Efficient Path Planning for Partially-Known Environments.
- LaValle (1998). Rapidly-Exploring Random Trees: A New Tool for Path Planning.
- LaValle (2006). Planning Algorithms. Cambridge University Press.
- Karaman, Frazzoli (2011). Sampling-based Algorithms for Optimal Motion Planning.
- Li et al. (2016). Asymptotically Optimal Sampling-based Kinodynamic Planning.

## License

MIT License. See [LICENSE](LICENSE).
