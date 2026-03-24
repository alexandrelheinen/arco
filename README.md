# ARCO: Autonomous Routing, Control, and Observation

<img src="docs/images/arco.svg" alt="FRET Logo" width="200" align="left">

My career has been built on autonomy systems, but most of that work lives inside proprietary codebases. The algorithms I used daily are not things I can share from those jobs.

ARCO is my open repo: a place to implement these methods from scratch, document the reasoning behind them, and build a reference I can actually point to.

This is also a Python project by choice. My production work has been C++ and embedded systems. ARCO is where I practice writing clean, well-tested Python for scientific computing: numpy, scipy, and eventually ROS 2 integration.

## Structure

```
arco/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── arco/
│       ├── __init__.py
│       └── planning/
└── tests/
    └── planning/
```

## Modules

- The mapping layer handles spatial representation.
- The planning layer searches or samples for feasible paths through that representation
- The guidance layer tracks the resulting path with feedback controllers (pure pursuit, PID, MPC).

### Mapping

TBD

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

TBD

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
black src/ tests/
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
