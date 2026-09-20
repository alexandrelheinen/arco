# ARCO Documentation

## Layer docs
- [Mapping](MAPPING.md) — grids, graphs, occupancy
- [Planning](PLANNING.md) — discrete and sampling planners
- [Guidance](GUIDANCE.md) — interpolation, primitives, control usage

## Algorithm notes
- [Core algorithm blocks](ALGORITHMS.md)
- [A*](planning_astar.md)
- [RRT*](planning_rrt.md)
- [SST](planning_sst.md)
- [Trajectory optimizer](planning_optimizer.md)
- [Contouring MPCC path following](control_mpcc.md)
- [Post–path-planning tracking parameters](control_tracking_params.md)

## Tools and design
- [Visualization (`arcosim`)](VISUALIZATION.md)
- [Illustration gallery](GALLERY.md) — 16:9 plates for decks, pages and papers
- [Pipeline](PIPELINE.md)
- [Entity model](ENTITY_MODEL.md)

## Rust port
- [Spec](rust/SPEC.md) — scope, acceptance criteria, traceability ids
- [Rust conventions](rust/STYLE.md) — naming, docs, tooling, PyO3 rules
- [Port deviations](rust/DEVIATIONS.md) — where the port does not mirror Python

## Project
- [README](../README.md)
- [Public API reference](API.md) — user-facing classes and non-interface inventory
- [Failure modes](FAILURE_MODES.md) — invalid inputs and unsatisfiable plans
- [Tech stack](STACK.md)
- [Coding guidelines](guidelines.md) (authoritative)
- [Decision log](decisions.md) — why constraints and approaches were chosen
- [Contributing](../CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)
