# ARCO

<img src="docs/images/arco.svg" alt="ARCO Logo" width="120" align="left">

ARCO (Algorithms for Robotic Control and Optimization) is a Python library for
autonomous navigation building blocks: mapping, planning, and guidance/control.

## Documentation

- [Docs index](docs/README.md)
- [Public API reference](docs/API.md)
- [Coding guidelines](docs/guidelines.md) (authoritative)
- [Tech stack](docs/STACK.md)
- [Mapping](docs/MAPPING.md) · [Planning](docs/PLANNING.md) · [Guidance](docs/GUIDANCE.md)
- [Visualization (`arcosim`)](docs/VISUALIZATION.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Architecture

Planners operate on map objects:

- Discrete planners (A*, route planning) use Grid or Graph structures
- Continuous planners (RRT*, SST) use Occupancy structures

Map families: `ManhattanGrid` (L₁), `EuclideanGrid` (L₂), graph hierarchy
(`Graph` → `WeightedGraph` → `CartesianGraph` → `RoadGraph`), and
`KDTreeOccupancy`.

Guidance and control run after planning: motion primitives, interpolation, and
feedback controllers (PID, Pure Pursuit, path-following MPC).

## Algorithm status

| Algorithm | Status | Notes |
|-----------|--------|-------|
| A* | Done | Grid and graph search |
| Route planning | Done | A* on road networks |
| RRT* | Done | Asymptotically optimal sampling |
| SST | Done | Sparse-tree geometric planning |
| D* Lite | Stub | Not planned — see [ROADMAP](docs/ROADMAP.md) |

## Repository layout

```text
.
├── docs/                 algorithm notes and design docs
├── map/                  arcosim scenario YAML files
├── scripts/              local CI helpers
├── src/arco/
│   ├── config/           shared YAML + palette helpers
│   ├── control/          PID, Pure Pursuit, MPC, tracking
│   ├── guidance/         interpolation, primitives, vehicles
│   ├── kinematics/       RR / RRP arm models
│   ├── mapping/          grids, graphs, occupancy
│   ├── middleware/       in-process typed bus
│   ├── pipeline/         pipeline node runner
│   ├── planning/         discrete + continuous planners
│   └── simulator/        arcosim CLI, scenes, rendering
├── tests/                mirrored unit tests
└── tools/                demos and recorded media
```

## Installation

```bash
git clone https://github.com/alexandrelheinen/arco.git
cd arco
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[mpc]"     # CasADi path-following / joint-space MPC
pip install -e ".[tools]"   # pygame + OpenGL for arcosim
```

Python 3.10+. See [docs/STACK.md](docs/STACK.md) for details.

## Development

```bash
pytest tests/ -v
bash scripts/check_formatting.sh
bash scripts/run_tests.sh
```

Local examples:

```bash
arcosim map/city.yml
arcosim map/ppp.yml --image --record output/ppp.png
```

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/guidelines.md](docs/guidelines.md).

## References

- Hart, Nilsson, Raphael (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
- LaValle (1998). Rapidly-Exploring Random Trees: A New Tool for Path Planning.
- LaValle (2006). Planning Algorithms. Cambridge University Press.
- Karaman, Frazzoli (2011). Sampling-based Algorithms for Optimal Motion Planning.
- Li et al. (2016). Asymptotically Optimal Sampling-based Kinodynamic Planning.

## License

MIT License. See [LICENSE](LICENSE).
