# ARCO public API inventory

This file is the authoritative list of every public name the `arco` package exports, the module path each one resolves from, its exact signature with default values, and the exceptions it raises. `FR-API-01` and `FR-API-02` in [rust/SPEC.md](rust/SPEC.md) are claims about this list: an import keeps its module path, and a call keeps its argument names, its positional order and its defaults. Nothing listed here may change shape without an entry in [rust/DEVIATIONS.md](rust/DEVIATIONS.md).

A name absent from this file is an implementation detail. Layer narrative lives in [MAPPING.md](MAPPING.md), [PLANNING.md](PLANNING.md) and [GUIDANCE.md](GUIDANCE.md); constructor and planning failure contracts live in [FAILURE_MODES.md](FAILURE_MODES.md).

## Conventions the surface follows

| Rule | Detail |
|---|---|
| `__all__` is the contract | A layer package declares one, and this file is derived from it |
| The layer package is the import path | `from arco.planning import RRTPlanner`, rather than the module that declares the class |
| Finding nothing is a value | `AStar.search`, `AStarPlanner.plan`, `RRTPlanner.plan`, `SSTPlanner.plan` and `RouteRouter.plan` return `None` when no path exists |
| Constructors validate | An invalid parameter raises `ValueError` at construction rather than at the first call |
| Deprecation warns at construction | `MPCController.__init__` emits `DeprecationWarning` and names `DubinsPathFollowingMPC` |
| Arrays are `numpy.ndarray` | A position, a pose or a state crosses the API as an array, and a path is a list of them |

## How this inventory is derived

Every row comes from introspecting the installed package rather than from reading the sources by hand, so the file can be regenerated and diffed against the package it describes:

- The public surface is the union of the `__all__` lists declared by
  the non-simulator modules. A module without `__all__` holds
  implementation detail and contributes nothing on its own.
- Signatures are `inspect.signature` output, verbatim, including
  annotations and default values.
- Class members come from the method resolution order, restricted to
  classes the package itself defines, which is why an inherited member
  names the class that declares it.
- The exception column merges two sources: a `raise` statement in the
  function body, found by walking the abstract syntax tree, and the
  `Raises:` section of the docstring. A cell reading `none recorded`
  means neither source names an exception, not that the call cannot
  fail: an exception raised by a helper the function calls does not
  appear.
- The capture runs in its own interpreter. Importing every `arco`
  package populates the module-level configuration globals, which
  leaks into anything running afterwards in the same process, for the
  reason ADR-015 in [decisions.md](decisions.md) records.

`benches/capture_baseline.py` performs the same capture for the signature snapshot in `benches/baseline/signatures.json`, which `tests/rust/test_signature_parity.py` compares against the installed package on every run.

## The export contract

The non-simulator packages declare 155 `__all__` entries across 19 modules. They resolve to 108 distinct objects, because a layer package and its subpackage both export the same class, and `arco.guidance` re-exports five controllers from `arco.control`. Both numbers are part of the contract: `FR-API-01` is a claim about the entries, since each one is an import path a caller may have written.

| Module | Entries |
|---|---|
| `arco` | 8 |
| `arco.config` | 16 |
| `arco.control` | 20 |
| `arco.control.mpc` | 12 |
| `arco.control.rigid_body` | 3 |
| `arco.guidance` | 11 |
| `arco.guidance.interpolation` | 3 |
| `arco.guidance.primitive` | 2 |
| `arco.kinematics` | 2 |
| `arco.mapping` | 10 |
| `arco.mapping.graph` | 5 |
| `arco.mapping.grid` | 3 |
| `arco.middleware` | 7 |
| `arco.middleware.types` | 3 |
| `arco.pipeline` | 2 |
| `arco.planning` | 16 |
| `arco.planning.continuous` | 12 |
| `arco.planning.discrete` | 7 |
| `arco.protocols` | 13 |
| **Total** | **155** |

The compiled extension `arco._arco` declares its own `__all__` and sits outside that count, since it is the binding target rather than a path a caller imports.

Where a layer package and its subpackage both export a class, the layer package is the path to write: `from arco.planning import RRTPlanner`, which is what the tests and the examples use. Both paths resolve, and both stay under `FR-API-01`.

## Rust coverage

The port replaces the Python implementation crate by crate, so each name below carries the crate that holds its Rust counterpart. The crate graph and the reason for each structural difference are in [rust/SPEC.md](rust/SPEC.md) and [rust/DEVIATIONS.md](rust/DEVIATIONS.md).

| State | Meaning | Distinct names |
|---|---|---|
| ported | A crate holds the equivalent type, trait or function | 68 |
| partial | A crate holds part of the behavior and the rest stays Python | 2 |
| in progress | The crate exists and the port is being written | 0 |
| none | No crate holds it, and the port does not plan one | 30 |
| namespace | A subpackage name re-exported by `arco/__init__.py` | 8 |

The state describes the crate side alone. Whether a given name already reaches that crate through the extension is the binding layer's question, and `tests/rust/test_signature_parity.py` and `tests/rust/test_extension.py` are where the answer is checked rather than asserted here.

A Rust name differs from its Python spelling wherever the acronym rule applies, so `RRTPlanner` is `RrtPlanner` in the crate and is registered back under its original spelling at the binding. The full mapping is in [rust/STYLE.md](rust/STYLE.md).

The names marked `none` are what phase 10 has to account for before `src/arco/` can hold the binding layer alone.

| Group | Count | Why no crate holds it |
|---|---|---|
| Palette helpers and `LAYER_ALPHA` | 15 | Serve the renderer, and follow `arco.simulator` out of the port's scope |
| Planner telemetry | 6 | The publishing side is the `TelemetryPublisher` trait; the JSON file channel stays Python |
| `PlanningPipeline`, `PipelineResult` | 2 | Composition over three ported stages, holding no algorithm of its own |
| Middleware frames | 3 | Plain data, and the Rust bus carries the frame type as a generic parameter |
| `DStarLite`, `DStarPlanner` | 2 | Stubs raising `NotImplementedError`, kept as placeholders by [ROADMAP.md](ROADMAP.md) |
| `load_road_graph` | 1 | The JSON network reader has no crate-side counterpart, though `RoadGraph` itself is ported |
| `MPCTracker` | 1 | Duck-typed abstract base; `DubinsPathFollowingMPC` is registered onto it as a virtual subclass, but the interface itself holds no crate-side counterpart |
| **Total** | **30** | |

## Reading the tables

| Column | Content |
|---|---|
| Name | The exported name, as written in `__all__` |
| Kind | Class, abstract class, data class, named tuple, protocol, function or constant |
| Defined in | The module that declares the object |
| Also exported by | Every other module whose `__all__` carries the same name |
| Rust state | Whether a crate holds the counterpart: ported, partial, in progress or none |
| Rust counterpart | The crate and the item that holds it |
| Member | A public attribute of a class, including `__init__` and `__call__` |
| Signature | `inspect.signature` output, verbatim |
| Raises | Exception types named in the body or the docstring |
| Declared by | The class in the method resolution order that declares the member |

## `arco`

The top-level package re-exports its eight subpackages and nothing else, so `import arco` reaches the layers without binding a single class name. `arco.config` is importable but absent from `__all__`, so a caller who wants it names it.

| Name | Kind |
|---|---|
| `control` | subpackage |
| `guidance` | subpackage |
| `kinematics` | subpackage |
| `mapping` | subpackage |
| `middleware` | subpackage |
| `pipeline` | subpackage |
| `planning` | subpackage |
| `protocols` | subpackage |

## `arco.config`

Utility for loading package configuration files.

`load_config` resolves its directory from the `ARCO_CONFIG_DIR` environment variable and caches what it reads in a module-level global, so the first import of any module that calls it fixes the configuration for the life of the interpreter. ADR-015 in [decisions.md](decisions.md) records why the ported crates take configuration as an argument instead. The palette helpers serve the renderer and follow `arco.simulator` out of the port's scope.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `annotation_hex` | function | `arco.config.palette` | none | none | none |
| `annotation_rgb` | function | `arco.config.palette` | none | none | none |
| `hex_to_float` | function | `arco.config.palette` | none | none | none |
| `hex_to_rgb` | function | `arco.config.palette` | none | none | none |
| `LAYER_ALPHA` | constant | `arco.config.palette` | none | none | none |
| `layer_float` | function | `arco.config.palette` | none | none | none |
| `layer_hex` | function | `arco.config.palette` | none | none | none |
| `layer_rgb` | function | `arco.config.palette` | none | none | none |
| `load_config` | function | `arco.config` | none | partial | arco-core `config::read_yaml_file`, `config::parse_yaml` |
| `method_base_float` | function | `arco.config.palette` | none | none | none |
| `method_base_hex` | function | `arco.config.palette` | none | none | none |
| `method_base_rgb` | function | `arco.config.palette` | none | none | none |
| `obstacle_float` | function | `arco.config.palette` | none | none | none |
| `obstacle_hex` | function | `arco.config.palette` | none | none | none |
| `obstacle_rgb` | function | `arco.config.palette` | none | none | none |
| `ui_rgb` | function | `arco.config.palette` | none | none | none |

### Functions of `arco.config`

| Function | Signature | Raises |
|---|---|---|
| `arco.config.load_config` | `(name: 'str') -> 'dict[str, Any]'` | `FileNotFoundError` |
| `arco.config.palette.annotation_hex` | `(dark_bg: 'bool' = False) -> 'str'` | none recorded |
| `arco.config.palette.annotation_rgb` | `(dark_bg: 'bool' = False) -> 'tuple[int, int, int]'` | none recorded |
| `arco.config.palette.hex_to_float` | `(hex_str: 'str') -> 'tuple[float, float, float]'` | none recorded |
| `arco.config.palette.hex_to_rgb` | `(hex_str: 'str') -> 'tuple[int, int, int]'` | none recorded |
| `arco.config.palette.layer_float` | `(method: 'str', layer: 'str') -> 'tuple[float, float, float]'` | none recorded |
| `arco.config.palette.layer_hex` | `(method: 'str', layer: 'str') -> 'str'` | `ValueError` |
| `arco.config.palette.layer_rgb` | `(method: 'str', layer: 'str') -> 'tuple[int, int, int]'` | none recorded |
| `arco.config.palette.method_base_float` | `(method: 'str') -> 'tuple[float, float, float]'` | none recorded |
| `arco.config.palette.method_base_hex` | `(method: 'str') -> 'str'` | `KeyError` |
| `arco.config.palette.method_base_rgb` | `(method: 'str') -> 'tuple[int, int, int]'` | none recorded |
| `arco.config.palette.obstacle_float` | `() -> 'tuple[float, float, float]'` | none recorded |
| `arco.config.palette.obstacle_hex` | `() -> 'str'` | none recorded |
| `arco.config.palette.obstacle_rgb` | `() -> 'tuple[int, int, int]'` | none recorded |
| `arco.config.palette.ui_rgb` | `(key: 'str') -> 'tuple[int, int, int]'` | `KeyError` |

### Constants of `arco.config`

| Constant | Value |
|---|---|
| `arco.config.palette.LAYER_ALPHA` | `{'tree': 0.12, 'path': 0.4, 'pruned': 0.7, 'trajectory': 0.7, 'vehicle': 1.0}` |

## `arco.protocols`

Structural protocols for ARCO extension points.

Each name here is a `typing.Protocol` marked `runtime_checkable`, so an implementation matches by having the methods and declares no inheritance. The check that `isinstance` performs covers method names and stops there, which is how a sampler with the wrong signature reaches the inside of a planner loop before failing. The Rust traits these map to move that failure to compile time, which is the structural reason [rust/SPEC.md](rust/SPEC.md) gives for the port.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `AvoidanceStrategy` | protocol | `arco.protocols.avoidance` | none | ported | arco-core `AvoidanceStrategy` (trait) |
| `CostTerm` | protocol | `arco.protocols.cost_term` | none | ported | arco-core `CostTerm` (trait) |
| `DiscreteMap` | protocol | `arco.protocols.discrete_map` | none | ported | arco-core `DiscreteMap` (trait) |
| `OccupancyLike` | protocol | `arco.protocols.occupancy` | none | ported | arco-core `Occupancy` (trait) |
| `OptimizerLike` | protocol | `arco.protocols.optimizer` | none | ported | arco-core `Optimizer` (trait) |
| `PathTracker` | protocol | `arco.protocols.path_tracker` | none | ported | arco-core `PathTracker` (trait) |
| `PlannerLike` | protocol | `arco.protocols.planner` | none | ported | arco-core `Planner` (trait) |
| `PrunerLike` | protocol | `arco.protocols.pruner` | none | ported | arco-core `Pruner` (trait) |
| `Sampler` | protocol | `arco.protocols.sampler` | none | ported | arco-core `Sampler` (trait) |
| `SegmentChecker` | protocol | `arco.protocols.segment_checker` | none | ported | arco-core `SegmentChecker` (trait) |
| `Steerer` | protocol | `arco.protocols.steerer` | none | ported | arco-core `Steerer` (trait) |
| `TelemetryPublisher` | protocol | `arco.protocols.telemetry` | none | ported | arco-core `TelemetryPublisher` (trait) |
| `VehicleModel` | protocol | `arco.protocols.vehicle` | none | ported | arco-core `VehicleModel` (trait) |

### Classes of `arco.protocols`

#### `arco.protocols.avoidance.AvoidanceStrategy`

Reactive avoidance correction used by path-tracking loops.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, x: 'float', y: 'float', theta: 'float') -> 'float'` | none recorded | this class |

#### `arco.protocols.cost_term.CostTerm`

One term of a composite trajectory cost.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, context: 'Dict[str, Any]') -> 'float'` | none recorded | this class |

#### `arco.protocols.discrete_map.DiscreteMap`

Map/graph surface expected by discrete planners (A*, route).

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `distance` | `(self, node_a: 'Any', node_b: 'Any') -> 'float'` | none recorded | this class |
| `neighbors` | `(self, node: 'Any') -> 'Iterator[Any]'` | none recorded | this class |

#### `arco.protocols.occupancy.OccupancyLike`

Continuous occupancy surface used by RRT*/SST/pruner/optimizer.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `is_occupied` | `(self, point: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `nearest_obstacle` | `(self, point: 'np.ndarray') -> 'Tuple[float, np.ndarray]'` | none recorded | this class |

#### `arco.protocols.optimizer.OptimizerLike`

Optimizer stage: `optimize(path) -> result`.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `optimize` | `(self, path: 'List[np.ndarray]') -> 'Any'` | none recorded | this class |

#### `arco.protocols.path_tracker.PathTracker`

Geometric path tracker used by `TrackingLoop`.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `track` | `(self, pose: 'tuple[float, float, float]', path: 'Sequence[tuple[float, float]]', speed: 'float' = 1.0) -> 'tuple[float, float]'` | none recorded | this class |

#### `arco.protocols.planner.PlannerLike`

Continuous planner stage: `plan(start, goal) -> path | None`.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `plan` | `(self, start: 'np.ndarray', goal: 'np.ndarray') -> 'Optional[List[np.ndarray]]'` | none recorded | this class |

#### `arco.protocols.pruner.PrunerLike`

Pruner stage: `prune(path) -> shortened path`.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `prune` | `(self, path: 'List[np.ndarray]') -> 'List[np.ndarray]'` | none recorded | this class |

#### `arco.protocols.sampler.Sampler`

Random-state sampler used by RRT*/SST.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, rng: 'np.random.Generator') -> 'np.ndarray'` | none recorded | this class |

#### `arco.protocols.segment_checker.SegmentChecker`

Collision checker for a straight segment between two states.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, a: 'np.ndarray', b: 'np.ndarray') -> 'bool'` | none recorded | this class |

#### `arco.protocols.steerer.Steerer`

Steering law used by RRT*/SST (and optional pruner feasibility).

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, from_pt: 'np.ndarray', to_pt: 'np.ndarray') -> 'np.ndarray'` | none recorded | this class |

#### `arco.protocols.telemetry.TelemetryPublisher`

Sink for planner telemetry snapshots.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, telemetry: 'Any') -> 'None'` | none recorded | this class |

#### `arco.protocols.vehicle.VehicleModel`

SE(2) vehicle surface used by tracking loops.

Bases: `typing.Protocol`

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `pose` (property) | `(self) -> 'tuple[float, float, float]'` | none recorded | this class |
| `speed` (property) | `(self) -> 'float'` | none recorded | this class |
| `step` | `(self, speed_cmd: 'float', turn_rate_cmd: 'float', dt: 'float') -> 'None'` | none recorded | this class |
| `turn_rate` (property) | `(self) -> 'float'` | none recorded | this class |

## `arco.mapping`

Mapping module for spatial representations.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `CartesianGraph` | class | `arco.mapping.graph.cartesian` | `arco.mapping.graph` | ported | arco-mapping `CartesianGraph` |
| `EuclideanGrid` | class | `arco.mapping.grid.euclidean` | `arco.mapping.grid` | ported | arco-mapping `EuclideanGrid` |
| `Graph` | class | `arco.mapping.graph.base` | `arco.mapping.graph` | ported | arco-core `DiscreteMap` (trait), per A-03 |
| `Grid` | class | `arco.mapping.grid.base` | `arco.mapping.grid` | ported | arco-mapping `GridCells`, per A-03 |
| `KDTreeOccupancy` | class | `arco.mapping.kdtree` | none | ported | arco-mapping `KdTreeOccupancy` |
| `ManhattanGrid` | class | `arco.mapping.grid.manhattan` | `arco.mapping.grid` | ported | arco-mapping `ManhattanGrid` |
| `Occupancy` | abstract class | `arco.mapping.occupancy` | none | ported | arco-core `Occupancy` (trait) |
| `RoadGraph` | class | `arco.mapping.graph.road` | `arco.mapping.graph` | ported | arco-mapping `RoadGraph` |
| `WeightedGraph` | class | `arco.mapping.graph.weighted` | `arco.mapping.graph` | ported | arco-mapping `WeightedGraph` |
| `load_road_graph` | function | `arco.mapping.graph.loader` | `arco.mapping.graph` | none | none |

### Functions of `arco.mapping`

| Function | Signature | Raises |
|---|---|---|
| `arco.mapping.graph.loader.load_road_graph` | `(path: 'Union[str, os.PathLike]') -> 'RoadGraph'` | `FileNotFoundError`, `KeyError`, `ValueError` |

### Classes of `arco.mapping`

#### `arco.mapping.graph.base.Graph`

Representation of a graph G = (V, E).

Construct with `Graph()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | this class |
| `Node` (nested class) | `() -> 'None'` | none recorded | this class |

#### `arco.mapping.graph.cartesian.CartesianGraph`

Weighted graph whose nodes carry N-dimensional Cartesian positions.

Bases: `arco.mapping.graph.weighted.WeightedGraph`

Construct with `CartesianGraph(ndim: 'Optional[int]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self, ndim: 'Optional[int]' = None) -> 'None'` | none recorded | this class |
| `add_edge` | `(self, node_a: 'int', node_b: 'int', weight: 'Optional[float]' = None) -> 'None'` | none recorded | this class |
| `add_node` | `(self, node_id: 'int', *coords: 'float') -> 'None'` | `ValueError` | this class |
| `distance` | `(self, node_a: 'int', node_b: 'int') -> 'float'` | none recorded | this class |
| `edges` (property) | `(self) -> 'List[Tuple[int, int, float]]'` | none recorded | `arco.mapping.graph.weighted.WeightedGraph` |
| `find_nearest_node` | `(self, position: 'np.ndarray', max_radius: 'Optional[float]' = None) -> 'Optional[int]'` | none recorded | this class |
| `heuristic` | `(self, node_a: 'int', node_b: 'int') -> 'float'` | none recorded | this class |
| `ndim` (property) | `(self) -> 'Optional[int]'` | none recorded | this class |
| `neighbors` | `(self, node_id: 'int') -> 'Iterator[int]'` | none recorded | `arco.mapping.graph.weighted.WeightedGraph` |
| `nodes` (property) | `(self) -> 'List[int]'` | none recorded | `arco.mapping.graph.weighted.WeightedGraph` |
| `position` | `(self, node_id: 'int') -> 'np.ndarray'` | none recorded | this class |
| `project_to_nearest_edge` | `(self, position: 'np.ndarray', max_radius: 'Optional[float]' = None) -> 'Optional[Tuple[np.ndarray, int, int, float]]'` | none recorded | this class |

#### `arco.mapping.graph.road.RoadGraph`

Cartesian graph extended with per-edge geometry metadata.

Bases: `arco.mapping.graph.cartesian.CartesianGraph`

Construct with `RoadGraph() -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self) -> 'None'` | none recorded | this class |
| `add_edge` | `(self, node_a: 'int', node_b: 'int', weight: 'Optional[float]' = None, waypoints: 'Optional[List[Tuple[float, float]]]' = None) -> 'None'` | none recorded | this class |
| `add_node` | `(self, node_id: 'int', *coords: 'float') -> 'None'` | `ValueError` | `arco.mapping.graph.cartesian.CartesianGraph` |
| `distance` | `(self, node_a: 'int', node_b: 'int') -> 'float'` | none recorded | `arco.mapping.graph.cartesian.CartesianGraph` |
| `edge_geometry` | `(self, node_a: 'int', node_b: 'int') -> 'List[Tuple[float, float]]'` | none recorded | this class |
| `edges` (property) | `(self) -> 'List[Tuple[int, int, float]]'` | none recorded | `arco.mapping.graph.weighted.WeightedGraph` |
| `find_nearest_node` | `(self, position: 'np.ndarray', max_radius: 'Optional[float]' = None) -> 'Optional[int]'` | none recorded | `arco.mapping.graph.cartesian.CartesianGraph` |
| `full_edge_geometry` | `(self, node_a: 'int', node_b: 'int') -> 'List[np.ndarray]'` | none recorded | this class |
| `heuristic` | `(self, node_a: 'int', node_b: 'int') -> 'float'` | none recorded | `arco.mapping.graph.cartesian.CartesianGraph` |
| `ndim` (property) | `(self) -> 'Optional[int]'` | none recorded | `arco.mapping.graph.cartesian.CartesianGraph` |
| `neighbors` | `(self, node_id: 'int') -> 'Iterator[int]'` | none recorded | `arco.mapping.graph.weighted.WeightedGraph` |
| `nodes` (property) | `(self) -> 'List[int]'` | none recorded | `arco.mapping.graph.weighted.WeightedGraph` |
| `position` | `(self, node_id: 'int') -> 'np.ndarray'` | none recorded | `arco.mapping.graph.cartesian.CartesianGraph` |
| `project_to_nearest_edge` | `(self, position: 'np.ndarray', max_radius: 'Optional[float]' = None) -> 'Optional[Tuple[np.ndarray, int, int, float]]'` | none recorded | `arco.mapping.graph.cartesian.CartesianGraph` |

#### `arco.mapping.graph.weighted.WeightedGraph`

Generic weighted undirected graph.

Bases: `arco.mapping.graph.base.Graph`

Construct with `WeightedGraph() -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self) -> 'None'` | none recorded | this class |
| `add_edge` | `(self, node_a: 'int', node_b: 'int', weight: 'float') -> 'None'` | none recorded | this class |
| `add_node` | `(self, node_id: 'int') -> 'None'` | none recorded | this class |
| `distance` | `(self, node_a: 'int', node_b: 'int') -> 'float'` | `KeyError` | this class |
| `edges` (property) | `(self) -> 'List[Tuple[int, int, float]]'` | none recorded | this class |
| `neighbors` | `(self, node_id: 'int') -> 'Iterator[int]'` | none recorded | this class |
| `nodes` (property) | `(self) -> 'List[int]'` | none recorded | this class |

#### `arco.mapping.grid.base.Grid`

N-dimensional grid for discrete planners (A*, D*, etc).

Bases: `arco.mapping.graph.base.Graph`

Construct with `Grid(shape: 'Sequence[int] \| None' = None, *, physical_size: 'Sequence[float] \| None' = None, cell_size: 'float' = 1.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self, shape: 'Sequence[int] \| None' = None, *, physical_size: 'Sequence[float] \| None' = None, cell_size: 'float' = 1.0) -> 'None'` | `ValueError` | this class |
| `heuristic` | `(self, a: 'Tuple[int, ...]', b: 'Tuple[int, ...]') -> 'float'` | none recorded | this class |
| `is_occupied` | `(self, idx: 'Tuple[int, ...]') -> 'bool'` | none recorded | this class |
| `neighbors` (abstract) | `(self, idx: 'Tuple[int, ...]') -> 'Iterator[Tuple[int, ...]]'` | none recorded | this class |
| `position` | `(self, idx: 'Tuple[int, ...]') -> 'np.ndarray'` | none recorded | this class |
| `set_free` | `(self, idx: 'Tuple[int, ...]') -> 'None'` | none recorded | this class |
| `set_occupied` | `(self, idx: 'Tuple[int, ...]') -> 'None'` | none recorded | this class |

#### `arco.mapping.grid.euclidean.EuclideanGrid`

Grid with diagonal (L2) connectivity and distance.

Bases: `arco.mapping.grid.base.Grid`

Construct with `EuclideanGrid(shape: 'Sequence[int] \| None' = None, *, physical_size: 'Sequence[float] \| None' = None, cell_size: 'float' = 1.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self, shape: 'Sequence[int] \| None' = None, *, physical_size: 'Sequence[float] \| None' = None, cell_size: 'float' = 1.0) -> 'None'` | none recorded | this class |
| `distance` | `(self, a: 'Tuple[int, ...]', b: 'Tuple[int, ...]') -> 'float'` | none recorded | this class |
| `heuristic` | `(self, a: 'Tuple[int, ...]', b: 'Tuple[int, ...]') -> 'float'` | none recorded | `arco.mapping.grid.base.Grid` |
| `is_occupied` | `(self, idx: 'Tuple[int, ...]') -> 'bool'` | none recorded | `arco.mapping.grid.base.Grid` |
| `neighbors` | `(self, idx: 'Tuple[int, ...]') -> 'Iterator[Tuple[int, ...]]'` | none recorded | this class |
| `position` | `(self, idx: 'Tuple[int, ...]') -> 'np.ndarray'` | none recorded | `arco.mapping.grid.base.Grid` |
| `set_free` | `(self, idx: 'Tuple[int, ...]') -> 'None'` | none recorded | `arco.mapping.grid.base.Grid` |
| `set_occupied` | `(self, idx: 'Tuple[int, ...]') -> 'None'` | none recorded | `arco.mapping.grid.base.Grid` |

#### `arco.mapping.grid.manhattan.ManhattanGrid`

Grid with Manhattan (L1) connectivity and distance.

Bases: `arco.mapping.grid.base.Grid`

Construct with `ManhattanGrid(shape: 'Sequence[int] \| None' = None, *, physical_size: 'Sequence[float] \| None' = None, cell_size: 'float' = 1.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self, shape: 'Sequence[int] \| None' = None, *, physical_size: 'Sequence[float] \| None' = None, cell_size: 'float' = 1.0) -> 'None'` | none recorded | this class |
| `distance` | `(self, a: 'Tuple[int, ...]', b: 'Tuple[int, ...]') -> 'int'` | none recorded | this class |
| `heuristic` | `(self, a: 'Tuple[int, ...]', b: 'Tuple[int, ...]') -> 'float'` | none recorded | `arco.mapping.grid.base.Grid` |
| `is_occupied` | `(self, idx: 'Tuple[int, ...]') -> 'bool'` | none recorded | `arco.mapping.grid.base.Grid` |
| `neighbors` | `(self, idx: 'Tuple[int, ...]') -> 'Iterator[Tuple[int, ...]]'` | none recorded | this class |
| `position` | `(self, idx: 'Tuple[int, ...]') -> 'np.ndarray'` | none recorded | `arco.mapping.grid.base.Grid` |
| `set_free` | `(self, idx: 'Tuple[int, ...]') -> 'None'` | none recorded | `arco.mapping.grid.base.Grid` |
| `set_occupied` | `(self, idx: 'Tuple[int, ...]') -> 'None'` | none recorded | `arco.mapping.grid.base.Grid` |

#### `arco.mapping.kdtree.KDTreeOccupancy`

Continuous, sparse occupancy map backed by a KD-tree.

Bases: `arco.mapping.occupancy.Occupancy`

Construct with `KDTreeOccupancy(points: 'Union[np.ndarray, List[Sequence[float]]]', clearance: 'float' = 0.5) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `__init__` | `(self, points: 'Union[np.ndarray, List[Sequence[float]]]', clearance: 'float' = 0.5) -> 'None'` | `ValueError` | this class |
| `dimension` (property) | `(self) -> 'int'` | none recorded | this class |
| `is_occupied` | `(self, point: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `nearest_obstacle` | `(self, point: 'np.ndarray') -> 'Tuple[float, np.ndarray]'` | none recorded | this class |
| `points` (property) | `(self) -> 'np.ndarray'` | none recorded | this class |
| `query_distances` | `(self, points: 'np.ndarray') -> 'np.ndarray'` | none recorded | this class |
| `segment_free` | `(self, a: 'np.ndarray', b: 'np.ndarray', *, sample_count: 'int' = 12) -> 'bool'` | none recorded | `arco.mapping.occupancy.Occupancy` |

#### `arco.mapping.occupancy.Occupancy`

Abstract base for continuous occupancy maps (for RRT, SST, etc).

Bases: `arco.mapping.graph.base.Graph`, `abc.ABC`  
Abstract methods: `is_occupied`, `nearest_obstacle`

Construct with `Occupancy()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `Edge` (nested class) | `(node_0: "'Graph.Node'", node_1: "'Graph.Node'") -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `Node` (nested class) | `() -> 'None'` | none recorded | `arco.mapping.graph.base.Graph` |
| `is_occupied` (abstract) | `(self, point: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `nearest_obstacle` (abstract) | `(self, point: 'np.ndarray') -> 'Tuple[float, np.ndarray]'` | none recorded | this class |
| `query_distances` | `(self, points: 'np.ndarray') -> 'np.ndarray'` | none recorded | this class |
| `segment_free` | `(self, a: 'np.ndarray', b: 'np.ndarray', *, sample_count: 'int' = 12) -> 'bool'` | none recorded | this class |

## `arco.planning`

Planning module for path planning problems.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `AStar` | class | `arco.planning.discrete.api` | `arco.planning.discrete` | ported | arco-planning `discrete::astar::search` |
| `AStarPlanner` | class | `arco.planning.discrete.astar` | `arco.planning.discrete` | ported | arco-planning `discrete::astar::search` |
| `ContinuousPlanner` | abstract class | `arco.planning.continuous.base` | `arco.planning.continuous` | ported | arco-core `Planner` (trait), arco-planning `PlanOutcome` |
| `DiscretePlanner` | class | `arco.planning.discrete.base` | `arco.planning.discrete` | ported | arco-planning `discrete::astar::SearchOptions` |
| `DStarLite` | class | `arco.planning.discrete.api` | `arco.planning.discrete` | none | none |
| `DStarPlanner` | class | `arco.planning.discrete.dstar` | `arco.planning.discrete` | none | none |
| `PipelineResult` | data class | `arco.planning.pipeline` | none | none | none |
| `PlannerCost` | class | `arco.planning.cost` | none | ported | arco-core `PlannerCost` (trait) |
| `PlannerTelemetry` | data class | `arco.planning.continuous.telemetry` | `arco.planning.continuous` | none | none |
| `PlanningPipeline` | class | `arco.planning.pipeline` | none | none | none |
| `RouteResult` | named tuple | `arco.planning.discrete.route` | `arco.planning.discrete` | ported | arco-planning `RouteResult`, `RouteOutcome` |
| `RouteRouter` | class | `arco.planning.discrete.route` | `arco.planning.discrete` | ported | arco-planning `RouteRouter` |
| `RRTPlanner` | class | `arco.planning.continuous.rrt` | `arco.planning.continuous` | ported | arco-planning `RrtPlanner`, `RrtSettings` |
| `SSTPlanner` | class | `arco.planning.continuous.sst` | `arco.planning.continuous` | ported | arco-planning `SstPlanner`, `SstSettings` |
| `StopCriterion` | data class | `arco.planning.continuous.telemetry` | `arco.planning.continuous` | none | none |
| `TrajectoryOptimizer` | class | `arco.planning.continuous.optimizer` | `arco.planning.continuous` | ported | arco-planning `TrajectoryOptimizer`, `OptimizerSettings` |
| `TrajectoryPruner` | class | `arco.planning.continuous.pruner` | `arco.planning.continuous` | ported | arco-planning `TrajectoryPruner` |
| `TrajectoryResult` | data class | `arco.planning.continuous.optimizer` | `arco.planning.continuous` | ported | arco-planning `TrajectoryResult` |
| `DEFAULT_TELEMETRY_PATH` | constant | `arco.planning.continuous.telemetry` | `arco.planning.continuous` | none | none |
| `noop_publisher` | function | `arco.planning.continuous.telemetry` | `arco.planning.continuous` | none | none |
| `read_telemetry` | function | `arco.planning.continuous.telemetry` | `arco.planning.continuous` | none | none |
| `write_telemetry` | function | `arco.planning.continuous.telemetry` | `arco.planning.continuous` | none | none |

### Functions of `arco.planning`

| Function | Signature | Raises |
|---|---|---|
| `arco.planning.continuous.telemetry.noop_publisher` | `(telemetry: 'PlannerTelemetry') -> 'None'` | none recorded |
| `arco.planning.continuous.telemetry.read_telemetry` | `(path: 'Path' = PosixPath('/tmp/arco_planner_telemetry.json')) -> 'Optional[PlannerTelemetry]'` | none recorded |
| `arco.planning.continuous.telemetry.write_telemetry` | `(telemetry: 'PlannerTelemetry', path: 'Path' = PosixPath('/tmp/arco_planner_telemetry.json')) -> 'None'` | none recorded |

### Constants of `arco.planning`

| Constant | Value |
|---|---|
| `arco.planning.continuous.telemetry.DEFAULT_TELEMETRY_PATH` | `PosixPath('/tmp/arco_planner_telemetry.json')` |

### Classes of `arco.planning`

#### `arco.planning.continuous.base.ContinuousPlanner`

Base class for planners operating in continuous state spaces.

Bases: `arco.planning.cost.PlannerCost`, `abc.ABC`  
Abstract methods: `plan`

Construct with `ContinuousPlanner(occupancy: 'Occupancy', cost: 'Optional[PlannerCost]' = None, publisher: 'Optional[TelemetryFn]' = None, seed: 'Optional[int]' = None, rng: 'Optional[np.random.Generator]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, occupancy: 'Occupancy', cost: 'Optional[PlannerCost]' = None, publisher: 'Optional[TelemetryFn]' = None, seed: 'Optional[int]' = None, rng: 'Optional[np.random.Generator]' = None) -> 'None'` | none recorded | this class |
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |
| `make_rng` | `(self) -> 'np.random.Generator'` | none recorded | this class |
| `plan` (abstract) | `(self, start: 'np.ndarray', goal: 'np.ndarray') -> 'Optional[List[np.ndarray]]'` | none recorded | this class |
| `publish_telemetry` | `(self, telemetry: 'PlannerTelemetry') -> 'None'` | none recorded | this class |

#### `arco.planning.continuous.optimizer.TrajectoryOptimizer`

Model-agnostic two-stage trajectory optimizer.

Construct with `TrajectoryOptimizer(occupancy: 'Occupancy', cruise_speed: 'float' = 1.0, weight_time: 'float' = 10.0, weight_deviation: 'float' = 1.0, weight_velocity: 'float' = 1.0, weight_collision: 'float' = 5.0, collision_barrier_scale: 'float' = 50.0, collision_barrier_power: 'float' = 4.0, weight_dynamics: 'float' = 100.0, max_speed: 'Optional[float]' = None, min_speed: 'Optional[float]' = None, time_relaxation: 'float' = 1.5, method: 'str' = 'L-BFGS-B', sample_count: 'int' = 3, max_iter: 'int' = 500, ftol: 'float' = 1e-09, cost_terms: 'Optional[Sequence[CostTerm]]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, occupancy: 'Occupancy', cruise_speed: 'float' = 1.0, weight_time: 'float' = 10.0, weight_deviation: 'float' = 1.0, weight_velocity: 'float' = 1.0, weight_collision: 'float' = 5.0, collision_barrier_scale: 'float' = 50.0, collision_barrier_power: 'float' = 4.0, weight_dynamics: 'float' = 100.0, max_speed: 'Optional[float]' = None, min_speed: 'Optional[float]' = None, time_relaxation: 'float' = 1.5, method: 'str' = 'L-BFGS-B', sample_count: 'int' = 3, max_iter: 'int' = 500, ftol: 'float' = 1e-09, cost_terms: 'Optional[Sequence[CostTerm]]' = None) -> 'None'` | `ValueError` | this class |
| `create_from_config` (static) | `(occupancy: 'Occupancy', cruise_speed: 'float', max_speed: 'Optional[float]' = None, min_speed: 'Optional[float]' = None) -> 'TrajectoryOptimizer'` | `ValueError` | this class |
| `optimize` | `(self, reference_path: 'List[np.ndarray]', inverse_kinematics: 'Optional[Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]]' = None, feasibility: 'Optional[Callable[[np.ndarray], bool]]' = None) -> 'TrajectoryResult'` | `ValueError` | this class |

#### `arco.planning.continuous.optimizer.TrajectoryResult`

Result of a trajectory optimization run.

Construct with `TrajectoryResult(states: 'List[np.ndarray]' = <factory>, commands: 'List[np.ndarray]' = <factory>, durations: 'List[float]' = <factory>, cost: 'float' = 0.0, is_feasible: 'bool' = True, optimizer_success: 'bool' = True, optimizer_status_code: 'int' = 0, optimizer_status_text: 'str' = '', optimizer_iteration_count: 'int' = 0) -> None`.

| Field | Type | Default |
|---|---|---|
| `states` | `List[np.ndarray]` | factory `list` |
| `commands` | `List[np.ndarray]` | factory `list` |
| `durations` | `List[float]` | factory `list` |
| `cost` | `float` | 0.0 |
| `is_feasible` | `bool` | True |
| `optimizer_success` | `bool` | True |
| `optimizer_status_code` | `int` | 0 |
| `optimizer_status_text` | `str` | '' |
| `optimizer_iteration_count` | `int` | 0 |

#### `arco.planning.continuous.pruner.TrajectoryPruner`

Reduce the node count of a raw path before trajectory optimization.

Construct with `TrajectoryPruner(occupancy: 'Occupancy', step_size: 'np.ndarray', collision_check_count: 'int' = 10) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, occupancy: 'Occupancy', step_size: 'np.ndarray', collision_check_count: 'int' = 10) -> 'None'` | `ValueError` | this class |
| `prune` | `(self, path: 'List[np.ndarray]', steer: 'Optional[Callable[[np.ndarray, np.ndarray], bool]]' = None) -> 'List[np.ndarray]'` | none recorded | this class |

#### `arco.planning.continuous.rrt.RRTPlanner`

Asymptotically-optimal RRT* planner for continuous geometric spaces.

Bases: `arco.planning.continuous.base.ContinuousPlanner`

Construct with `RRTPlanner(occupancy: 'Occupancy', bounds: 'Sequence[Tuple[float, float]]', max_sample_count: 'int' = 2000, step_size: 'float \| np.ndarray' = 1.0, goal_tolerance: 'float' = 1.0, rewire_radius: 'Optional[float]' = None, collision_check_count: 'int' = 10, goal_bias: 'float' = 0.05, early_stop: 'bool' = True, sampler: 'Optional[SamplerFn]' = None, steerer: 'Optional[SteererFn]' = None, segment_free: 'Optional[SegmentFreeFn]' = None, cost: 'Optional[PlannerCost]' = None, publisher: 'Optional[Callable[[PlannerTelemetry], None]]' = None, seed: 'Optional[int]' = None, rng: 'Optional[np.random.Generator]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, occupancy: 'Occupancy', bounds: 'Sequence[Tuple[float, float]]', max_sample_count: 'int' = 2000, step_size: 'float \| np.ndarray' = 1.0, goal_tolerance: 'float' = 1.0, rewire_radius: 'Optional[float]' = None, collision_check_count: 'int' = 10, goal_bias: 'float' = 0.05, early_stop: 'bool' = True, sampler: 'Optional[SamplerFn]' = None, steerer: 'Optional[SteererFn]' = None, segment_free: 'Optional[SegmentFreeFn]' = None, cost: 'Optional[PlannerCost]' = None, publisher: 'Optional[Callable[[PlannerTelemetry], None]]' = None, seed: 'Optional[int]' = None, rng: 'Optional[np.random.Generator]' = None) -> 'None'` | `ValueError` | this class |
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `get_tree` | `(self, start: 'np.ndarray', goal: 'np.ndarray') -> 'Tuple[List[np.ndarray], Dict[int, Optional[int]], Optional[List[np.ndarray]]]'` | none recorded | this class |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `is_segment_free` | `(self, a: 'np.ndarray', b: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `make_rng` | `(self) -> 'np.random.Generator'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `plan` | `(self, start: 'np.ndarray', goal: 'np.ndarray') -> 'Optional[List[np.ndarray]]'` | none recorded | this class |
| `publish_telemetry` | `(self, telemetry: 'PlannerTelemetry') -> 'None'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `sample` | `(self, rng: 'np.random.Generator') -> 'np.ndarray'` | none recorded | this class |
| `steer` | `(self, from_pt: 'np.ndarray', to_pt: 'np.ndarray') -> 'np.ndarray'` | none recorded | this class |

#### `arco.planning.continuous.sst.SSTPlanner`

Asymptotically near-optimal SST planner for continuous geometric spaces.

Bases: `arco.planning.continuous.base.ContinuousPlanner`

Construct with `SSTPlanner(occupancy: 'Occupancy', bounds: 'Sequence[Tuple[float, float]]', max_sample_count: 'int' = 3000, step_size: 'float \| np.ndarray' = 1.0, goal_tolerance: 'float' = 1.0, witness_radius: 'float' = 0.5, collision_check_count: 'int' = 10, goal_bias: 'float' = 0.05, early_stop: 'bool' = True, sampler: 'Optional[SamplerFn]' = None, steerer: 'Optional[SteererFn]' = None, segment_free: 'Optional[SegmentFreeFn]' = None, cost: 'Optional[PlannerCost]' = None, publisher: 'Optional[Callable[[PlannerTelemetry], None]]' = None, seed: 'Optional[int]' = None, rng: 'Optional[np.random.Generator]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, occupancy: 'Occupancy', bounds: 'Sequence[Tuple[float, float]]', max_sample_count: 'int' = 3000, step_size: 'float \| np.ndarray' = 1.0, goal_tolerance: 'float' = 1.0, witness_radius: 'float' = 0.5, collision_check_count: 'int' = 10, goal_bias: 'float' = 0.05, early_stop: 'bool' = True, sampler: 'Optional[SamplerFn]' = None, steerer: 'Optional[SteererFn]' = None, segment_free: 'Optional[SegmentFreeFn]' = None, cost: 'Optional[PlannerCost]' = None, publisher: 'Optional[Callable[[PlannerTelemetry], None]]' = None, seed: 'Optional[int]' = None, rng: 'Optional[np.random.Generator]' = None) -> 'None'` | `ValueError` | this class |
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `get_tree` | `(self, start: 'np.ndarray', goal: 'np.ndarray') -> 'Tuple[List[np.ndarray], Dict[int, Optional[int]], Optional[List[np.ndarray]]]'` | none recorded | this class |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `is_segment_free` | `(self, a: 'np.ndarray', b: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `make_rng` | `(self) -> 'np.random.Generator'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `plan` | `(self, start: 'np.ndarray', goal: 'np.ndarray') -> 'Optional[List[np.ndarray]]'` | none recorded | this class |
| `publish_telemetry` | `(self, telemetry: 'PlannerTelemetry') -> 'None'` | none recorded | `arco.planning.continuous.base.ContinuousPlanner` |
| `sample` | `(self, rng: 'np.random.Generator') -> 'np.ndarray'` | none recorded | this class |
| `steer` | `(self, from_pt: 'np.ndarray', to_pt: 'np.ndarray') -> 'np.ndarray'` | none recorded | this class |

#### `arco.planning.continuous.telemetry.PlannerTelemetry`

A telemetry snapshot written by a planner at regular intervals.

Construct with `PlannerTelemetry(algorithm: 'str', step_name: 'str', iteration: 'int', max_iterations: 'int', best_dist_to_goal: 'float', criteria: 'list[StopCriterion]' = <factory>) -> None`.

| Field | Type | Default |
|---|---|---|
| `algorithm` | `str` | required |
| `step_name` | `str` | required |
| `iteration` | `int` | required |
| `max_iterations` | `int` | required |
| `best_dist_to_goal` | `float` | required |
| `criteria` | `list[StopCriterion]` | factory `list` |

#### `arco.planning.continuous.telemetry.StopCriterion`

A single named stop criterion with its current and threshold values.

Construct with `StopCriterion(name: 'str', current: 'float', threshold: 'float', condition: 'str') -> None`.

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `current` | `float` | required |
| `threshold` | `float` | required |
| `condition` | `str` | required |

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `satisfied` | `(self) -> 'bool'` | none recorded | this class |

#### `arco.planning.cost.PlannerCost`

Default distance and heuristic cost functions for planners.

Construct with `PlannerCost()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |

#### `arco.planning.discrete.api.AStar`

Public API wrapper for the A* planner.

Construct with `AStar(grid: 'np.ndarray', grid_type: 'str' = 'manhattan') -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, grid: 'np.ndarray', grid_type: 'str' = 'manhattan') -> 'None'` | `ValueError` | this class |
| `search` | `(self, start: 'Any', goal: 'Any') -> 'Optional[List[Any]]'` | none recorded | this class |

#### `arco.planning.discrete.api.DStarLite`

Public API wrapper for D* planner (stub , not yet implemented).

Construct with `DStarLite(grid: 'np.ndarray') -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, grid: 'np.ndarray') -> 'None'` | none recorded | this class |
| `search` | `(self, start: 'Any', goal: 'Any') -> 'Optional[List[Any]]'` | `NotImplementedError` | this class |

#### `arco.planning.discrete.astar.AStarPlanner`

A* path planner for general graphs (including grids).

Bases: `arco.planning.discrete.base.DiscretePlanner`

Construct with `AStarPlanner(graph: 'Any', heuristic: 'Optional[Callable[[Any, Any], float]]' = None, simplify_path: 'bool' = True, prefer_straight: 'bool' = True) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, graph: 'Any', heuristic: 'Optional[Callable[[Any, Any], float]]' = None, simplify_path: 'bool' = True, prefer_straight: 'bool' = True) -> 'None'` | none recorded | this class |
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.discrete.base.DiscretePlanner` |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |
| `plan` | `(self, start: 'Any', goal: 'Any') -> 'Optional[List[Any]]'` | none recorded | this class |
| `plan_with_diagnostics` | `(self, start: 'Any', goal: 'Any') -> 'tuple[Optional[List[Any]], List[Any], Dict[Any, Any]]'` | none recorded | this class |

#### `arco.planning.discrete.base.DiscretePlanner`

Base class for discrete planners operating on graphs (including grids).

Bases: `arco.planning.cost.PlannerCost`

Construct with `DiscretePlanner(graph: 'Any') -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, graph: 'Any') -> 'None'` | none recorded | this class |
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | this class |

#### `arco.planning.discrete.dstar.DStarPlanner`

D* path planner for dynamic replanning (stub).

Bases: `arco.planning.discrete.base.DiscretePlanner`

Construct with `DStarPlanner(graph: 'Any') -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, graph: 'Any') -> 'None'` | none recorded | `arco.planning.discrete.base.DiscretePlanner` |
| `distance` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.discrete.base.DiscretePlanner` |
| `heuristic` | `(self, state_a: 'Any', state_b: 'Any') -> 'float'` | none recorded | `arco.planning.discrete.base.DiscretePlanner` |
| `plan` | `(self, start: 'Any', goal: 'Any') -> 'Optional[List[Any]]'` | `NotImplementedError` | this class |

#### `arco.planning.discrete.route.RouteResult`

Result of a route planning query.

Bases: `builtins.tuple`

Construct with `RouteResult(path: ForwardRef('List[int]'), start_node: ForwardRef('int'), goal_node: ForwardRef('int'), start_projection: ForwardRef('np.ndarray'), goal_projection: ForwardRef('np.ndarray'), start_distance: ForwardRef('float'), goal_distance: ForwardRef('float'))`.

| Field | Type | Default |
|---|---|---|
| `path` | `ForwardRef('List[int]')` | required |
| `start_node` | `ForwardRef('int')` | required |
| `goal_node` | `ForwardRef('int')` | required |
| `start_projection` | `ForwardRef('np.ndarray')` | required |
| `goal_projection` | `ForwardRef('np.ndarray')` | required |
| `start_distance` | `ForwardRef('float')` | required |
| `goal_distance` | `ForwardRef('float')` | required |

#### `arco.planning.discrete.route.RouteRouter`

Route planner for continuous coordinates on Cartesian road graphs.

Construct with `RouteRouter(graph: 'CartesianGraph', activation_radius: 'Optional[float]' = None, planner: 'Optional[Any]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, graph: 'CartesianGraph', activation_radius: 'Optional[float]' = None, planner: 'Optional[Any]' = None) -> 'None'` | none recorded | this class |
| `plan` | `(self, start_position: 'np.ndarray', goal_position: 'np.ndarray') -> 'Optional[RouteResult]'` | none recorded | this class |

#### `arco.planning.pipeline.PipelineResult`

Snapshot of every stage output from one `run` call.

Construct with `PipelineResult(raw_path: 'Optional[list[np.ndarray]]' = None, pruned_path: 'Optional[list[np.ndarray]]' = None, trajectory: 'Optional[list[np.ndarray]]' = None, durations: 'Optional[list[float]]' = None, total_duration: 'float' = 0.0, planner_time: 'float' = 0.0, pruner_time: 'float' = 0.0, optimizer_time: 'float' = 0.0, planner_status: 'str' = 'not_run', optimizer_status: 'str' = 'not_run', optimizer_success: 'bool' = False, extra: 'dict[str, Any]' = <factory>) -> None`.

| Field | Type | Default |
|---|---|---|
| `raw_path` | `Optional[list[np.ndarray]]` | None |
| `pruned_path` | `Optional[list[np.ndarray]]` | None |
| `trajectory` | `Optional[list[np.ndarray]]` | None |
| `durations` | `Optional[list[float]]` | None |
| `total_duration` | `float` | 0.0 |
| `planner_time` | `float` | 0.0 |
| `pruner_time` | `float` | 0.0 |
| `optimizer_time` | `float` | 0.0 |
| `planner_status` | `str` | 'not_run' |
| `optimizer_status` | `str` | 'not_run' |
| `optimizer_success` | `bool` | False |
| `extra` | `dict[str, Any]` | factory `dict` |

#### `arco.planning.pipeline.PlanningPipeline`

Algorithm-agnostic orchestrator for the full planning pipeline.

Construct with `PlanningPipeline(planner: "Optional['ContinuousPlanner']" = None, pruner: "Optional['TrajectoryPruner']" = None, optimizer: "Optional['TrajectoryOptimizer']" = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, planner: "Optional['ContinuousPlanner']" = None, pruner: "Optional['TrajectoryPruner']" = None, optimizer: "Optional['TrajectoryOptimizer']" = None) -> 'None'` | none recorded | this class |
| `load_result` (static) | `(path: 'str \| Path') -> "'PipelineResult'"` | `FileNotFoundError`, `ValueError` | this class |
| `run` | `(self, start: 'np.ndarray', goal: 'np.ndarray', progress: 'Optional[Callable[[str, int, int], None]]' = None) -> "'PipelineResult'"` | `RuntimeError` | this class |
| `run_from_path` | `(self, raw_path: 'list[np.ndarray]', progress: 'Optional[Callable[[str, int, int], None]]' = None) -> "'PipelineResult'"` | none recorded | this class |
| `save_result` (static) | `(result: "'PipelineResult'", path: 'str \| Path') -> 'None'` | none recorded | this class |

## `arco.control`

Control subpackage: feedback controllers, tracking, and object-centric control.

The MPC names carry deviations a caller has to read before relying on an exact output. A-01 removes `casadi` from the dependency list, and A-02 replaces the nonlinear program with a sequence of linearized programs solved by Clarabel, which returns a different and also valid solution for the same input. A-30 turns the obstacle keep-out into a half-space through the nominal position with a penalized slack, so `obstacle_barrier_power` is accepted but no longer shapes the barrier. A-31 reports the surrogate convex objective in `cost`, comparable across steps of one controller but not against a value the earlier nonlinear solver printed. A-32 replaces the earlier solver's status strings in `solver_status` with `solved`, `solved_inexact`, `invalid_state`, `infeasible`, `unbounded`, `budget_exhausted` and `numerical`. A-09 adds saturation, rate limiting and anti-windup to every command leaving the layer, and A-17 and A-18 add validation of the elapsed interval.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `ActuatorArray` | class | `arco.control.actuator` | none | ported | arco-control `ActuatorArray`, `ActuatorSettings`, `GraspMatrix` |
| `ArtificialPotentialField` | class | `arco.control.avoidance` | none | ported | arco-control `ArtificialPotentialField` |
| `CircleBody` | class | `arco.control.rigid_body.circle` | `arco.control.rigid_body` | ported | arco-control `CircleBody` |
| `Controller` | abstract class | `arco.control.base` | `arco.guidance` | ported | arco-py `Controller` |
| `DubinsPathFollowingMPC` | class | `arco.control.mpc.path_following` | `arco.control.mpc` | ported | arco-control `mpc`, `QpProblem`, `QpSolution` |
| `DubinsVehicleLimits` | data class | `arco.control.mpc.path_following` | `arco.control.mpc` | ported | arco-control `CommandLimits` |
| `JointSpaceMPC` | class | `arco.control.mpc.joint_space` | `arco.control.mpc` | ported | arco-control `mpc` |
| `JointSpaceMPCConfig` | data class | `arco.control.mpc.joint_space` | `arco.control.mpc` | ported | arco-control `mpc` |
| `JointSpaceTracker` | class | `arco.control.joint_tracker` | none | ported | arco-control `JointSpaceTracker`, `JointTrackerSettings` |
| `MPCController` | class | `arco.control.mpc.controller` | `arco.control.mpc`, `arco.guidance` | ported | arco-py `MpcController`, per C-01 |
| `MPCStepResult` | data class | `arco.control.mpc.result` | `arco.control.mpc` | ported | arco-control `mpc` |
| `MPCTracker` | abstract class | `arco.control.mpc.base` | `arco.control.mpc` | none | none |
| `MPCTrackingLoop` | class | `arco.control.mpc.tracking_loop` | `arco.control.mpc` | ported | arco-control `mpc` |
| `PathFollowingMPCConfig` | data class | `arco.control.mpc.path_following` | `arco.control.mpc` | ported | arco-control `mpc` |
| `PIDController` | class | `arco.control.pid` | `arco.guidance` | ported | arco-control `PidController`, `PidGains`, `AntiWindup` |
| `PurePursuitController` | class | `arco.control.pure_pursuit` | `arco.guidance` | ported | arco-control `PurePursuitTracker` |
| `ReferencePath` | class | `arco.control.mpc.reference_path` | `arco.control.mpc` | ported | arco-control `ReferencePath` |
| `RigidBody` | abstract class | `arco.control.rigid_body.base` | `arco.control.rigid_body` | ported | arco-control `RigidBody` (trait), `BodyState` |
| `SquareBody` | class | `arco.control.rigid_body.square` | `arco.control.rigid_body` | ported | arco-control `SquareBody` |
| `TrackingLoop` | class | `arco.control.tracking` | `arco.guidance` | ported | arco-control `TrackingLoop`, `TrackingSettings` |
| `forward_cone_factor` | function | `arco.control.mpc.costs` | `arco.control.mpc` | ported | arco-control `mpc` |
| `obstacle_barrier` | function | `arco.control.mpc.costs` | `arco.control.mpc` | ported | arco-control `mpc` |

### Functions of `arco.control`

| Function | Signature | Raises |
|---|---|---|
| `arco.control.mpc.costs.forward_cone_factor` | `(pose_x: 'Any', pose_y: 'Any', heading: 'Any', obstacle_x: 'Any', obstacle_y: 'Any') -> 'Any'` | none recorded |
| `arco.control.mpc.costs.obstacle_barrier` | `(distance: 'Any', clearance: 'Any', weight: 'float', power: 'float', cone_factor: 'Any') -> 'Any'` | none recorded |

### Classes of `arco.control`

#### `arco.control.actuator.ActuatorArray`

Array of N contact actuators around a 2-D rigid body.

Construct with `ActuatorArray(actuator_count: 'int' = 4, standoff: 'float' = 0.05, omega: 'float' = 10.0, zeta: 'float' = 0.7, spring_stiffness: 'float' = 100.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, actuator_count: 'int' = 4, standoff: 'float' = 0.05, omega: 'float' = 10.0, zeta: 'float' = 0.7, spring_stiffness: 'float' = 100.0) -> 'None'` | `ValueError` | this class |
| `actuator_count` (property) | `(self) -> 'int'` | none recorded | this class |
| `actuator_positions` | `(self, body: 'RigidBody') -> 'np.ndarray'` | none recorded | this class |
| `allocate_forces` | `(self, desired_wrench: 'np.ndarray', body: 'RigidBody') -> 'np.ndarray'` | none recorded | this class |
| `allocate_radial_forces` | `(self, desired_wrench: 'np.ndarray', body: 'RigidBody') -> 'np.ndarray'` | none recorded | this class |
| `angle_velocities` (property) | `(self) -> 'np.ndarray'` | none recorded | this class |
| `angles` (property) | `(self) -> 'np.ndarray'` | none recorded | this class |
| `apply_spring_forces_to_body` | `(self, body: 'RigidBody') -> 'None'` | none recorded | this class |
| `apply_to_body` | `(self, forces: 'np.ndarray', body: 'RigidBody') -> 'None'` | none recorded | this class |
| `compute_ref_radii` | `(self, body: 'RigidBody', desired_forces: 'np.ndarray') -> 'None'` | none recorded | this class |
| `grasp_matrix` | `(self, body: 'RigidBody') -> 'np.ndarray'` | none recorded | this class |
| `init_radii` | `(self, body: 'RigidBody') -> 'None'` | none recorded | this class |
| `omega` (property) | `(self) -> 'float'` | none recorded | this class |
| `radii` (property) | `(self) -> 'np.ndarray \| None'` | none recorded | this class |
| `radii_velocities` (property) | `(self) -> 'np.ndarray \| None'` | none recorded | this class |
| `ref_angles` (property) | `(self) -> 'np.ndarray'` | none recorded | this class |
| `ref_radii` (property) | `(self) -> 'np.ndarray \| None'` | none recorded | this class |
| `repulsive_wrench` | `(self, body: 'RigidBody', nearest_obstacle_fn: 'Callable[[np.ndarray], tuple[float, np.ndarray]]', other_positions: 'np.ndarray', k_rep: 'float', d0: 'float') -> 'np.ndarray'` | none recorded | this class |
| `set_angles` | `(self, angles: 'np.ndarray') -> 'None'` | `ValueError` | this class |
| `spring_forces` | `(self, body: 'RigidBody') -> 'np.ndarray'` | none recorded | this class |
| `spring_stiffness` (property) | `(self) -> 'float'` | none recorded | this class |
| `step_actuators` | `(self, dt: 'float') -> 'None'` | none recorded | this class |
| `update_angles_for_target` | `(self, body: 'RigidBody', target_wrench: 'np.ndarray') -> 'None'` | none recorded | this class |
| `zeta` (property) | `(self) -> 'float'` | none recorded | this class |

#### `arco.control.avoidance.ArtificialPotentialField`

Artificial Potential Field turn-rate bias for obstacle avoidance.

Construct with `ArtificialPotentialField(occupancy: "Optional['Occupancy']" = None, repulsion_gain: 'float' = 0.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__call__` | `(self, x: 'float', y: 'float', theta: 'float') -> 'float'` | none recorded | this class |
| `__init__` | `(self, occupancy: "Optional['Occupancy']" = None, repulsion_gain: 'float' = 0.0) -> 'None'` | none recorded | this class |

#### `arco.control.base.Controller`

Abstract base for feedback controllers.

Bases: `abc.ABC`  
Abstract methods: `control`

Construct with `Controller()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `control` (abstract) | `(self, state: float, reference: float) -> float` | none recorded | this class |

#### `arco.control.joint_tracker.JointSpaceTracker`

N-DOF proportional tracker with velocity/acceleration saturation and APF repulsion.

Construct with `JointSpaceTracker(max_vel: 'float \| np.ndarray', max_acc: 'float \| np.ndarray', proportional_gain: 'float' = 2.0, occupancy: 'Optional[Occupancy]' = None, repulsion_gain: 'float' = 0.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, max_vel: 'float \| np.ndarray', max_acc: 'float \| np.ndarray', proportional_gain: 'float' = 2.0, occupancy: 'Optional[Occupancy]' = None, repulsion_gain: 'float' = 0.0) -> 'None'` | `ValueError` | this class |
| `reset` | `(self, q0: 'np.ndarray') -> 'None'` | none recorded | this class |
| `step` | `(self, target_q: 'np.ndarray', dt: 'float') -> 'np.ndarray'` | none recorded | this class |

#### `arco.control.mpc.base.MPCTracker`

Abstract base for multi-state model-predictive path trackers.

Bases: `abc.ABC`  
Abstract methods: `set_reference`, `step`

Construct with `MPCTracker()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `set_reference` (abstract) | `(self, waypoints: 'Sequence[tuple[float, float]]') -> 'None'` | none recorded | this class |
| `step` (abstract) | `(self, pose: 'tuple[float, float, float]', *, speed: 'float', turn_rate: 'float', dt: 'float') -> 'MPCStepResult'` | none recorded | this class |

#### `arco.control.mpc.controller.MPCController`

Deprecated scalar Model Predictive Controller stub.

Bases: `arco.control.base.Controller`

Construct with `MPCController(horizon: 'int' = 10, dt: 'float' = 0.1) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, horizon: 'int' = 10, dt: 'float' = 0.1) -> 'None'` | none recorded | this class |
| `control` | `(self, state: 'float', reference: 'float') -> 'float'` | none recorded | this class |

#### `arco.control.mpc.joint_space.JointSpaceMPC`

N-DOF receding-horizon tracker for C-space carrots.

Construct with `JointSpaceMPC(max_vel: 'float \| np.ndarray', max_acc: 'float \| np.ndarray', proportional_gain: 'float' = 2.0, occupancy: 'Occupancy \| None' = None, repulsion_gain: 'float' = 0.0, config: 'Optional[JointSpaceMPCConfig]' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, max_vel: 'float \| np.ndarray', max_acc: 'float \| np.ndarray', proportional_gain: 'float' = 2.0, occupancy: 'Occupancy \| None' = None, repulsion_gain: 'float' = 0.0, config: 'Optional[JointSpaceMPCConfig]' = None) -> 'None'` | `ImportError`, `ValueError` | this class |
| `reset` | `(self, q0: 'np.ndarray') -> 'None'` | none recorded | this class |
| `step` | `(self, target_q: 'np.ndarray', dt: 'float') -> 'np.ndarray'` | `ValueError` | this class |

#### `arco.control.mpc.joint_space.JointSpaceMPCConfig`

Horizon and weights for joint-space carrot-tracking MPC.

Construct with `JointSpaceMPCConfig(horizon_step_count: 'int' = 12, dt: 'float' = 0.05, weight_tracking: 'float' = 20.0, weight_velocity: 'float' = 0.5, weight_control: 'float' = 0.05, weight_obstacle: 'float' = 60.0, obstacle_barrier_power: 'float' = 4.0, max_solver_iter_count: 'int' = 40) -> None`.

| Field | Type | Default |
|---|---|---|
| `horizon_step_count` | `int` | 12 |
| `dt` | `float` | 0.05 |
| `weight_tracking` | `float` | 20.0 |
| `weight_velocity` | `float` | 0.5 |
| `weight_control` | `float` | 0.05 |
| `weight_obstacle` | `float` | 60.0 |
| `obstacle_barrier_power` | `float` | 4.0 |
| `max_solver_iter_count` | `int` | 40 |

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `create_from_config` (static) | `() -> 'JointSpaceMPCConfig'` | none recorded | this class |
| `with_horizon_overrides` | `(self, *, step_count: 'int \| None' = None, dt: 'float \| None' = None) -> 'JointSpaceMPCConfig'` | none recorded | this class |

#### `arco.control.mpc.path_following.DubinsPathFollowingMPC`

Receding-horizon contouring NMPC (MPCC) for Dubins vehicles.

Bases: `arco.control.mpc.base.MPCTracker`

Construct with `DubinsPathFollowingMPC(*, vehicle_limits: 'DubinsVehicleLimits', config: 'PathFollowingMPCConfig', occupancy: 'Occupancy \| None' = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, *, vehicle_limits: 'DubinsVehicleLimits', config: 'PathFollowingMPCConfig', occupancy: 'Occupancy \| None' = None) -> 'None'` | `ImportError`, `ValueError` | this class |
| `set_reference` | `(self, waypoints: 'Sequence[tuple[float, float]]') -> 'None'` | none recorded | this class |
| `step` | `(self, pose: 'tuple[float, float, float]', *, speed: 'float', turn_rate: 'float', dt: 'float') -> 'MPCStepResult'` | `RuntimeError` | this class |

#### `arco.control.mpc.path_following.DubinsVehicleLimits`

Dynamic limits mirrored from `DubinsVehicle`.

Construct with `DubinsVehicleLimits(max_speed: 'float', min_speed: 'float', max_turn_rate: 'float', max_acceleration: 'float', max_turn_rate_dot: 'float') -> None`.

| Field | Type | Default |
|---|---|---|
| `max_speed` | `float` | required |
| `min_speed` | `float` | required |
| `max_turn_rate` | `float` | required |
| `max_acceleration` | `float` | required |
| `max_turn_rate_dot` | `float` | required |

#### `arco.control.mpc.path_following.PathFollowingMPCConfig`

Tunable weights and horizon for Dubins path-following MPCC.

Construct with `PathFollowingMPCConfig(horizon_step_count: 'int' = 20, dt: 'float' = 0.05, cruise_speed: 'float' = 0.36, weight_contour: 'float' = 10.0, weight_heading: 'float' = 2.0, weight_progress: 'float' = 1.0, weight_lag: 'float' = 4.0, weight_control: 'float' = 0.1, weight_obstacle: 'float' = 50.0, obstacle_barrier_power: 'float' = 4.0, weight_terminal: 'float' = 20.0, contour_deadzone: 'float' = 0.0, max_solver_iter_count: 'int' = 80) -> None`.

| Field | Type | Default |
|---|---|---|
| `horizon_step_count` | `int` | 20 |
| `dt` | `float` | 0.05 |
| `cruise_speed` | `float` | 0.36 |
| `weight_contour` | `float` | 10.0 |
| `weight_heading` | `float` | 2.0 |
| `weight_progress` | `float` | 1.0 |
| `weight_lag` | `float` | 4.0 |
| `weight_control` | `float` | 0.1 |
| `weight_obstacle` | `float` | 50.0 |
| `obstacle_barrier_power` | `float` | 4.0 |
| `weight_terminal` | `float` | 20.0 |
| `contour_deadzone` | `float` | 0.0 |
| `max_solver_iter_count` | `int` | 80 |

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `create_from_config` (static) | `(cruise_speed: 'Optional[float]' = None) -> 'PathFollowingMPCConfig'` | none recorded | this class |
| `with_horizon_overrides` | `(self, *, step_count: 'int \| None' = None, dt: 'float \| None' = None) -> 'PathFollowingMPCConfig'` | none recorded | this class |
| `with_weight_overrides` | `(self, *, contour: 'float \| None' = None, heading: 'float \| None' = None, progress: 'float \| None' = None, lag: 'float \| None' = None, control: 'float \| None' = None, obstacle: 'float \| None' = None, terminal: 'float \| None' = None, contour_deadzone: 'float \| None' = None) -> 'PathFollowingMPCConfig'` | none recorded | this class |

#### `arco.control.mpc.reference_path.ReferencePath`

Arc-length parameterized reference path for contouring MPC.

Construct with `ReferencePath(waypoints: 'Sequence[tuple[float, float]]') -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, waypoints: 'Sequence[tuple[float, float]]') -> 'None'` | `ValueError` | this class |
| `curvature` | `(self, s: 'float') -> 'float'` | none recorded | this class |
| `heading` | `(self, s: 'float') -> 'float'` | none recorded | this class |
| `position` | `(self, s: 'float') -> 'tuple[float, float]'` | none recorded | this class |
| `project` | `(self, pose: 'tuple[float, float, float]', *, s_hint: 'float \| None' = None, window: 'float \| None' = None) -> 'tuple[float, float, float]'` | none recorded | this class |
| `sample` | `(self, sample_count: 'int') -> 'tuple[np.ndarray, ...]'` | none recorded | this class |
| `tangent` | `(self, s: 'float') -> 'tuple[float, float]'` | none recorded | this class |
| `total_length` (property) | `(self) -> 'float'` | none recorded | this class |
| `waypoint_count` (property) | `(self) -> 'int'` | none recorded | this class |

#### `arco.control.mpc.result.MPCStepResult`

Result of a single `MPCTracker` step.

Construct with `MPCStepResult(speed_cmd: 'float', turn_rate_cmd: 'float', cross_track_error: 'float', heading_error: 'float', progress: 'float', predicted_clearance_min: 'float', solver_success: 'bool', solver_status: 'str', solve_time_s: 'float', cost: 'float', predicted_xy: 'list[tuple[float, float]]' = <factory>) -> None`.

| Field | Type | Default |
|---|---|---|
| `speed_cmd` | `float` | required |
| `turn_rate_cmd` | `float` | required |
| `cross_track_error` | `float` | required |
| `heading_error` | `float` | required |
| `progress` | `float` | required |
| `predicted_clearance_min` | `float` | required |
| `solver_success` | `bool` | required |
| `solver_status` | `str` | required |
| `solve_time_s` | `float` | required |
| `cost` | `float` | required |
| `predicted_xy` | `list[tuple[float, float]]` | factory `list` |

#### `arco.control.mpc.tracking_loop.MPCTrackingLoop`

Local tracking loop driven by a `MPCTracker`.

Construct with `MPCTrackingLoop(vehicle: 'DubinsVehicle', tracker: 'MPCTracker', cruise_speed: 'float' = 1.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, vehicle: 'DubinsVehicle', tracker: 'MPCTracker', cruise_speed: 'float' = 1.0) -> 'None'` | none recorded | this class |
| `history` (property) | `(self) -> 'list[dict[str, Any]]'` | none recorded | this class |
| `metrics` (property) | `(self) -> 'dict[str, Any] \| None'` | none recorded | this class |
| `run` | `(self, path: 'list[tuple[float, float]]', steps: 'int', dt: 'float' = 0.1) -> 'list[dict[str, Any]]'` | none recorded | this class |
| `step` | `(self, path: 'list[tuple[float, float]]', dt: 'float' = 0.1) -> 'dict[str, Any]'` | none recorded | this class |

#### `arco.control.pid.PIDController`

PID controller for path tracking.

Bases: `arco.control.base.Controller`

Construct with `PIDController(kp: 'float' = 1.0, ki: 'float' = 0.0, kd: 'float' = 0.1) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, kp: 'float' = 1.0, ki: 'float' = 0.0, kd: 'float' = 0.1) -> 'None'` | none recorded | this class |
| `control` | `(self, state: 'float', reference: 'float') -> 'float'` | none recorded | this class |

#### `arco.control.pure_pursuit.PurePursuitController`

Pure pursuit controller for 2-D path tracking.

Bases: `arco.control.base.Controller`

Construct with `PurePursuitController(lookahead_distance: 'float' = 1.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, lookahead_distance: 'float' = 1.0) -> 'None'` | none recorded | this class |
| `control` | `(self, state: 'float', reference: 'float') -> 'float'` | none recorded | this class |
| `track` | `(self, pose: 'tuple[float, float, float]', path: 'Sequence[tuple[float, float]]', speed: 'float' = 1.0) -> 'tuple[float, float]'` | none recorded | this class |

#### `arco.control.rigid_body.base.RigidBody`

Abstract 2-D rigid body.

Bases: `abc.ABC`  
Abstract methods: `bounding_radius`, `inertia`

Construct with `RigidBody(mass: 'float', x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, mass: 'float', x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'` | `ValueError` | this class |
| `apply_wrench` | `(self, fx: 'float', fy: 'float', torque: 'float') -> 'None'` | none recorded | this class |
| `bounding_radius` (property) | `(self) -> 'float'` | none recorded | this class |
| `inertia` (property) | `(self) -> 'float'` | none recorded | this class |
| `mass` (property) | `(self) -> 'float'` | none recorded | this class |
| `pose` (property) | `(self) -> 'np.ndarray'` | none recorded | this class |
| `reset` | `(self, x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'` | none recorded | this class |
| `step` | `(self, dt: 'float') -> 'None'` | none recorded | this class |
| `velocity` (property) | `(self) -> 'np.ndarray'` | none recorded | this class |

#### `arco.control.rigid_body.circle.CircleBody`

Uniform-density circular rigid body.

Bases: `arco.control.rigid_body.base.RigidBody`

Construct with `CircleBody(mass: 'float', radius: 'float', x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, mass: 'float', radius: 'float', x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'` | `ValueError` | this class |
| `apply_wrench` | `(self, fx: 'float', fy: 'float', torque: 'float') -> 'None'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `bounding_radius` (property) | `(self) -> 'float'` | none recorded | this class |
| `inertia` (property) | `(self) -> 'float'` | none recorded | this class |
| `mass` (property) | `(self) -> 'float'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `pose` (property) | `(self) -> 'np.ndarray'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `radius` (property) | `(self) -> 'float'` | none recorded | this class |
| `reset` | `(self, x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `step` | `(self, dt: 'float') -> 'None'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `velocity` (property) | `(self) -> 'np.ndarray'` | none recorded | `arco.control.rigid_body.base.RigidBody` |

#### `arco.control.rigid_body.square.SquareBody`

Uniform-density square rigid body.

Bases: `arco.control.rigid_body.base.RigidBody`

Construct with `SquareBody(mass: 'float', side_length: 'float', x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, mass: 'float', side_length: 'float', x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'` | `ValueError` | this class |
| `apply_wrench` | `(self, fx: 'float', fy: 'float', torque: 'float') -> 'None'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `bounding_radius` (property) | `(self) -> 'float'` | none recorded | this class |
| `corners` | `(self) -> 'np.ndarray'` | none recorded | this class |
| `inertia` (property) | `(self) -> 'float'` | none recorded | this class |
| `mass` (property) | `(self) -> 'float'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `pose` (property) | `(self) -> 'np.ndarray'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `reset` | `(self, x: 'float' = 0.0, y: 'float' = 0.0, psi: 'float' = 0.0) -> 'None'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `side_length` (property) | `(self) -> 'float'` | none recorded | this class |
| `step` | `(self, dt: 'float') -> 'None'` | none recorded | `arco.control.rigid_body.base.RigidBody` |
| `velocity` (property) | `(self) -> 'np.ndarray'` | none recorded | `arco.control.rigid_body.base.RigidBody` |

#### `arco.control.tracking.TrackingLoop`

Local tracking loop combining a vehicle model and a path controller.

Construct with `TrackingLoop(vehicle: "'VehicleModel'", controller: "'PathTracker'", cruise_speed: 'float' = 1.0, curvature_gain: 'float' = 0.0, occupancy: "Optional['Occupancy']" = None, repulsion_gain: 'float' = 0.0, avoidance: "Optional['AvoidanceStrategy']" = None) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, vehicle: "'VehicleModel'", controller: "'PathTracker'", cruise_speed: 'float' = 1.0, curvature_gain: 'float' = 0.0, occupancy: "Optional['Occupancy']" = None, repulsion_gain: 'float' = 0.0, avoidance: "Optional['AvoidanceStrategy']" = None) -> 'None'` | none recorded | this class |
| `history` (property) | `(self) -> 'list[dict[str, Any]]'` | none recorded | this class |
| `metrics` (property) | `(self) -> 'dict[str, Any] \| None'` | none recorded | this class |
| `run` | `(self, path: 'list[tuple[float, float]]', steps: 'int', dt: 'float' = 0.1) -> 'list[dict[str, Any]]'` | none recorded | this class |
| `step` | `(self, path: 'list[tuple[float, float]]', dt: 'float' = 0.1) -> 'dict[str, Any]'` | none recorded | this class |

## `arco.guidance`

Guidance module for path tracking and trajectory generation.

Five of the eleven entries are controllers `arco.control` defines: `Controller`, `PIDController`, `PurePursuitController`, `TrackingLoop` and `MPCController`. They resolve from both paths, a caller importing a controller should name `arco.control`, and their detail sits under that package rather than being repeated here.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `BSplineInterpolator` | class | `arco.guidance.interpolation.bspline` | `arco.guidance.interpolation` | ported | arco-guidance `BSplineInterpolator` |
| `Controller` | abstract class | `arco.control.base` | `arco.control` | ported | arco-py `Controller` |
| `DubinsPrimitive` | class | `arco.guidance.primitive.dubins` | `arco.guidance.primitive` | ported | arco-guidance `DubinsPrimitive` |
| `DubinsVehicle` | class | `arco.guidance.vehicle` | none | ported | arco-guidance `DubinsVehicle`, arco-control `CommandLimits` |
| `ExplorationPrimitive` | abstract class | `arco.guidance.primitive.base` | `arco.guidance.primitive` | ported | arco-guidance `ExplorationPrimitive` (trait) |
| `Interpolator` | abstract class | `arco.guidance.interpolation.base` | `arco.guidance.interpolation` | ported | arco-guidance `Interpolator` (trait) |
| `MovingAverageInterpolator` | class | `arco.guidance.interpolation.moving_average` | `arco.guidance.interpolation` | ported | arco-guidance `MovingAverageInterpolator` |
| `MPCController` | class | `arco.control.mpc.controller` | `arco.control`, `arco.control.mpc` | ported | arco-py `MpcController`, per C-01 |
| `PIDController` | class | `arco.control.pid` | `arco.control` | ported | arco-control `PidController`, `PidGains`, `AntiWindup` |
| `PurePursuitController` | class | `arco.control.pure_pursuit` | `arco.control` | ported | arco-control `PurePursuitTracker` |
| `TrackingLoop` | class | `arco.control.tracking` | `arco.control` | ported | arco-control `TrackingLoop`, `TrackingSettings` |

### Classes of `arco.guidance`

#### `arco.guidance.interpolation.base.Interpolator`

Abstract base for interpolation (e.g., B-splines, shortcutting).

Bases: `abc.ABC`  
Abstract methods: `interpolate`

Construct with `Interpolator()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `interpolate` (abstract) | `(self, path: 'List[Any]') -> 'List[Any]'` | none recorded | this class |

#### `arco.guidance.interpolation.bspline.BSplineInterpolator`

B-spline interpolator for smoothing discrete paths.

Bases: `arco.guidance.interpolation.base.Interpolator`

Construct with `BSplineInterpolator(degree: 'int' = 3) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, degree: 'int' = 3) -> 'None'` | none recorded | this class |
| `interpolate` | `(self, path: 'List[Any]') -> 'List[Any]'` | none recorded | this class |

#### `arco.guidance.interpolation.moving_average.MovingAverageInterpolator`

Sliding-window moving-average smoothing of a waypoint polyline.

Bases: `arco.guidance.interpolation.base.Interpolator`

Construct with `MovingAverageInterpolator(*, iterations: 'int' = 1, window: 'int' = 3) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, *, iterations: 'int' = 1, window: 'int' = 3) -> 'None'` | `ValueError` | this class |
| `interpolate` | `(self, path: 'List[Any]') -> 'List[Any]'` | none recorded | this class |

#### `arco.guidance.primitive.base.ExplorationPrimitive`

Abstract base for exploration primitives (e.g., Dubins, Reeds-Shepp).

Bases: `abc.ABC`  
Abstract methods: `steer`

Construct with `ExplorationPrimitive()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `steer` (abstract) | `(self, from_state: 'Any', to_state: 'Any') -> 'List[Any]'` | none recorded | this class |

#### `arco.guidance.primitive.dubins.DubinsPrimitive`

Dubins path primitive for car-like robots.

Bases: `arco.guidance.primitive.base.ExplorationPrimitive`

Construct with `DubinsPrimitive(turning_radius: 'float' = 1.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, turning_radius: 'float' = 1.0) -> 'None'` | none recorded | this class |
| `is_feasible` | `(self, state: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `steer` | `(self, from_state: 'Any', to_state: 'Any') -> 'List[Any]'` | none recorded | this class |

#### `arco.guidance.vehicle.DubinsVehicle`

Dubins-like kinematic vehicle model with bounded dynamics.

Construct with `DubinsVehicle(x: 'float' = 0.0, y: 'float' = 0.0, heading: 'float' = 0.0, max_speed: 'float' = 5.0, min_speed: 'float' = 0.0, max_turn_rate: 'float' = 1.0, max_acceleration: 'float' = 2.0, max_turn_rate_dot: 'float' = 2.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, x: 'float' = 0.0, y: 'float' = 0.0, heading: 'float' = 0.0, max_speed: 'float' = 5.0, min_speed: 'float' = 0.0, max_turn_rate: 'float' = 1.0, max_acceleration: 'float' = 2.0, max_turn_rate_dot: 'float' = 2.0) -> 'None'` | none recorded | this class |
| `inverse_kinematics` | `(self, start: 'np.ndarray', goal: 'np.ndarray', speed: 'float', duration: 'float') -> 'np.ndarray'` | none recorded | this class |
| `is_feasible` | `(self, state: 'np.ndarray') -> 'bool'` | none recorded | this class |
| `pose` (property) | `(self) -> 'tuple[float, float, float]'` | none recorded | this class |
| `reset` | `(self, x: 'float' = 0.0, y: 'float' = 0.0, heading: 'float' = 0.0) -> 'None'` | none recorded | this class |
| `speed` (property) | `(self) -> 'float'` | none recorded | this class |
| `step` | `(self, speed_cmd: 'float', turn_rate_cmd: 'float', dt: 'float') -> 'tuple[float, float, float]'` | none recorded | this class |
| `turn_rate` (property) | `(self) -> 'float'` | none recorded | this class |

## `arco.kinematics`

Kinematics module for robot arm models.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `RRPRobot` | class | `arco.kinematics.rrp` | none | ported | arco-kinematics `RrpRobot` |
| `RRRobot` | class | `arco.kinematics.rr` | none | ported | arco-kinematics `RrRobot` |

### Classes of `arco.kinematics`

#### `arco.kinematics.rr.RRRobot`

Two-link planar revolute-revolute robot arm.

Construct with `RRRobot(l1: 'float' = 1.0, l2: 'float' = 0.8) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, l1: 'float' = 1.0, l2: 'float' = 0.8) -> 'None'` | `ValueError` | this class |
| `forward_kinematics` | `(self, q1: 'float', q2: 'float') -> 'tuple[float, float]'` | none recorded | this class |
| `inverse_kinematics` | `(self, x: 'float', y: 'float', eps: 'float' = 1e-09) -> 'list[tuple[float, float]]'` | none recorded | this class |
| `l1` (property) | `(self) -> 'float'` | none recorded | this class |
| `l2` (property) | `(self) -> 'float'` | none recorded | this class |
| `link_segments` | `(self, q1: 'float', q2: 'float') -> 'tuple[tuple[float, float], tuple[float, float], tuple[float, float]]'` | none recorded | this class |
| `workspace_annulus` | `(self) -> 'tuple[float, float]'` | none recorded | this class |
| `workspace_radius` | `(self) -> 'float'` | none recorded | this class |

#### `arco.kinematics.rrp.RRPRobot`

Two-link planar RR arm with a vertical prismatic joint (SCARA-like).

Construct with `RRPRobot(l1: 'float' = 1.0, l2: 'float' = 0.8, z_min: 'float' = 0.0, z_max: 'float' = 4.0) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, l1: 'float' = 1.0, l2: 'float' = 0.8, z_min: 'float' = 0.0, z_max: 'float' = 4.0) -> 'None'` | `ValueError` | this class |
| `forward_kinematics` | `(self, q1: 'float', q2: 'float', z: 'float') -> 'tuple[float, float, float]'` | none recorded | this class |
| `inverse_kinematics_xy` | `(self, x: 'float', y: 'float', eps: 'float' = 1e-09) -> 'list[tuple[float, float]]'` | none recorded | this class |
| `l1` (property) | `(self) -> 'float'` | none recorded | this class |
| `l2` (property) | `(self) -> 'float'` | none recorded | this class |
| `link_segments` | `(self, q1: 'float', q2: 'float', z: 'float') -> 'tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]'` | none recorded | this class |
| `workspace_annulus` | `(self) -> 'tuple[float, float]'` | none recorded | this class |
| `workspace_radius` | `(self) -> 'float'` | none recorded | this class |
| `z_max` (property) | `(self) -> 'float'` | none recorded | this class |
| `z_min` (property) | `(self) -> 'float'` | none recorded | this class |

## `arco.middleware`

Shared in-memory middleware for the ARCO async pipeline.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `Bus` | abstract class | `arco.middleware.bus` | none | ported | arco-runtime `Bus` |
| `BusPublisher` | class | `arco.middleware.publisher` | none | ported | arco-runtime `Bus::publish`, `PublishReport` |
| `BusSubscriber` | class | `arco.middleware.subscriber` | none | ported | arco-runtime `Subscription` |
| `GuidanceFrame` | data class | `arco.middleware.types.guidance_frame` | `arco.middleware.types` | none | none |
| `InMemoryBus` | class | `arco.middleware.bus` | none | ported | arco-runtime `Bus` |
| `MappingFrame` | data class | `arco.middleware.types.mapping_frame` | `arco.middleware.types` | none | none |
| `PlanFrame` | data class | `arco.middleware.types.plan_frame` | `arco.middleware.types` | none | none |

### Classes of `arco.middleware`

#### `arco.middleware.bus.Bus`

Abstract base for the shared in-memory message bus.

Bases: `abc.ABC`  
Abstract methods: `publish`, `subscribe`, `subscriber_count`

Construct with `Bus()`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `publish` (abstract) | `(self, frame: 'object') -> 'None'` | none recorded | this class |
| `subscribe` (abstract) | `(self, frame_type: 'Type[T]') -> "'queue.Queue[T]'"` | none recorded | this class |
| `subscriber_count` (abstract) | `(self, frame_type: 'Type[T]') -> 'int'` | none recorded | this class |

#### `arco.middleware.bus.InMemoryBus`

Thread-safe in-process bus backed by bounded `Queue` instances.

Bases: `arco.middleware.bus.Bus`

Construct with `InMemoryBus(maxsize: 'int' = 64) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, maxsize: 'int' = 64) -> 'None'` | none recorded | this class |
| `publish` | `(self, frame: 'object') -> 'None'` | none recorded | this class |
| `subscribe` | `(self, frame_type: 'Type[T]') -> "'queue.Queue[T]'"` | none recorded | this class |
| `subscriber_count` | `(self, frame_type: 'Type[T]') -> 'int'` | none recorded | this class |
| `unsubscribe` | `(self, frame_type: 'Type[T]', q: "'queue.Queue[T]'") -> 'None'` | none recorded | this class |

#### `arco.middleware.publisher.BusPublisher`

Mixin that adds bus-publish capability to a pipeline node.

Construct with `BusPublisher() -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self) -> 'None'` | none recorded | this class |
| `attach_bus` | `(self, bus: 'Bus') -> 'None'` | none recorded | this class |
| `publish` | `(self, frame: 'object') -> 'None'` | none recorded | this class |

#### `arco.middleware.subscriber.BusSubscriber`

Mixin that adds bus-subscription capability to a frontend node.

Construct with `BusSubscriber() -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self) -> 'None'` | none recorded | this class |
| `drain_latest` | `(self, frame_type: 'Type[T]') -> 'Optional[T]'` | none recorded | this class |
| `next_frame` | `(self, frame_type: 'Type[T]', block: 'bool' = False, timeout: 'Optional[float]' = None) -> 'Optional[T]'` | none recorded | this class |
| `subscribe` | `(self, bus: 'Bus', frame_type: 'Type[T]') -> "'queue.Queue[T]'"` | none recorded | this class |

#### `arco.middleware.types.guidance_frame.GuidanceFrame`

A single snapshot produced by the guidance layer.

Construct with `GuidanceFrame(timestamp: 'float', trajectory: 'List[List[float]]' = <factory>, durations: 'List[float]' = <factory>) -> None`.

| Field | Type | Default |
|---|---|---|
| `timestamp` | `float` | required |
| `trajectory` | `List[List[float]]` | factory `list` |
| `durations` | `List[float]` | factory `list` |

#### `arco.middleware.types.mapping_frame.MappingFrame`

A single snapshot produced by the mapping layer.

Construct with `MappingFrame(timestamp: 'float', obstacle_points: 'List[List[float]]' = <factory>, bounds: 'List[float]' = <factory>, clearance: 'float' = 0.0) -> None`.

| Field | Type | Default |
|---|---|---|
| `timestamp` | `float` | required |
| `obstacle_points` | `List[List[float]]` | factory `list` |
| `bounds` | `List[float]` | factory `<lambda>` |
| `clearance` | `float` | 0.0 |

#### `arco.middleware.types.plan_frame.PlanFrame`

A single snapshot produced by the planning layer.

Construct with `PlanFrame(timestamp: 'float', waypoints: 'List[List[float]]' = <factory>, planner: 'str' = '') -> None`.

| Field | Type | Default |
|---|---|---|
| `timestamp` | `float` | required |
| `waypoints` | `List[List[float]]` | factory `list` |
| `planner` | `str` | '' |

## `arco.pipeline`

Async pipeline orchestration for the ARCO processing chain.

| Name | Kind | Defined in | Also exported by | Rust state | Rust counterpart |
|---|---|---|---|---|---|
| `PipelineNode` | abstract class | `arco.pipeline.node` | none | ported | arco-runtime `Node` (trait), `Control`, `Handle` |
| `PipelineRunner` | class | `arco.pipeline.runner` | none | partial | arco-runtime `Runner`, per A-06 |

### Classes of `arco.pipeline`

#### `arco.pipeline.node.PipelineNode`

Abstract base class for a single stage in the async pipeline.

Bases: `arco.middleware.publisher.BusPublisher`, `abc.ABC`  
Abstract methods: `run`

Construct with `PipelineNode(name: 'str') -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, name: 'str') -> 'None'` | none recorded | this class |
| `attach_bus` | `(self, bus: 'Bus') -> 'None'` | none recorded | `arco.middleware.publisher.BusPublisher` |
| `is_running` (property) | `(self) -> 'bool'` | none recorded | this class |
| `name` (property) | `(self) -> 'str'` | none recorded | this class |
| `publish` | `(self, frame: 'object') -> 'None'` | none recorded | `arco.middleware.publisher.BusPublisher` |
| `run` (abstract) | `(self) -> 'None'` | none recorded | this class |
| `start` | `(self) -> 'None'` | none recorded | this class |
| `stop` | `(self, timeout: 'float \| None' = None) -> 'None'` | none recorded | this class |
| `stop_requested` (property) | `(self) -> 'bool'` | none recorded | this class |

#### `arco.pipeline.runner.PipelineRunner`

Orchestrates the async pipeline from a YAML configuration file.

Construct with `PipelineRunner(config_path: 'str \| Path', bus_maxsize: 'int' = 64) -> 'None'`.

| Member | Signature | Raises | Declared by |
|---|---|---|---|
| `__init__` | `(self, config_path: 'str \| Path', bus_maxsize: 'int' = 64) -> 'None'` | `FileNotFoundError`, `RuntimeError` | this class |
| `attach_subscriber` | `(self, subscriber: 'BusSubscriber', frame_type: 'Type[T]') -> 'None'` | none recorded | this class |
| `bus` (property) | `(self) -> 'InMemoryBus'` | none recorded | this class |
| `config` (property) | `(self) -> 'Dict[str, Any]'` | none recorded | this class |
| `register_node` | `(self, node: 'PipelineNode') -> 'None'` | none recorded | this class |
| `start` | `(self) -> 'None'` | none recorded | this class |
| `stop` | `(self, timeout: 'Optional[float]' = None) -> 'None'` | none recorded | this class |

## Policy hooks and the interpreter lock

Nine arguments take a caller-supplied callable: `sampler`, `steerer`, `segment_free`, `cost`, `cost_terms`, `heuristic`, `feasibility`, `publisher` and `nearest_obstacle_fn`. Left at their default, the loop dispatches the built-in policy natively and never acquires the interpreter lock, which is the condition `FR-PERF-02` states and the condition the speedup of `FR-PERF-01` is measured under.

Pass a Python callable to any of them and the loop reacquires the lock on every call, so the speedup does not apply. The result is unchanged and only the timing differs. `FR-PERF-03` requires that this be written down where a caller reading the API finds it, and A-07 in [rust/DEVIATIONS.md](rust/DEVIATIONS.md) records it as a deviation.

| Argument | Accepted by |
|---|---|
| `sampler` | `RRTPlanner`, `SSTPlanner` |
| `steerer` | `RRTPlanner`, `SSTPlanner` |
| `segment_free` | `RRTPlanner`, `SSTPlanner` |
| `cost` | `RRTPlanner`, `SSTPlanner`, `ContinuousPlanner` |
| `cost_terms` | `TrajectoryOptimizer` |
| `heuristic` | `AStarPlanner` |
| `feasibility` | `TrajectoryOptimizer.optimize` |
| `publisher` | `RRTPlanner`, `SSTPlanner`, `ContinuousPlanner` |
| `nearest_obstacle_fn` | `ActuatorArray.repulsive_wrench` |

`TrajectoryPruner.prune` takes a `steer` callable and `TrajectoryOptimizer.optimize` takes an `inverse_kinematics` callable on the same terms.

## Exceptions across the surface

`FR-API-04` requires a ported function to raise the type the Python implementation raised, or a subclass of it, so this is the set the binding layer has to reproduce.

| Exception | Raised by |
|---|---|
| `FileNotFoundError` | `PipelineRunner.__init__`, `PlanningPipeline.load_result`, `load_config`, `load_road_graph` |
| `ImportError` | `DubinsPathFollowingMPC.__init__`, `JointSpaceMPC.__init__` |
| `KeyError` | `WeightedGraph.distance`, `load_road_graph`, `method_base_hex`, `ui_rgb` |
| `NotImplementedError` | `DStarLite.search`, `DStarPlanner.plan` |
| `RuntimeError` | `DubinsPathFollowingMPC.step`, `PipelineRunner.__init__`, `PlanningPipeline.run` |
| `ValueError` | `AStar.__init__`, `ActuatorArray.__init__`, `ActuatorArray.set_angles`, `CartesianGraph.add_node`, `CircleBody.__init__`, `DubinsPathFollowingMPC.__init__`, `Grid.__init__`, `JointSpaceMPC.__init__`, `JointSpaceMPC.step`, `JointSpaceTracker.__init__`, `KDTreeOccupancy.__init__`, `MovingAverageInterpolator.__init__`, `PlanningPipeline.load_result`, `RRPRobot.__init__`, `RRRobot.__init__`, `RRTPlanner.__init__`, `ReferencePath.__init__`, `RigidBody.__init__`, `RoadGraph.add_node`, `SSTPlanner.__init__`, `SquareBody.__init__`, `TrajectoryOptimizer.__init__`, `TrajectoryOptimizer.create_from_config`, `TrajectoryOptimizer.optimize`, `TrajectoryPruner.__init__`, `layer_hex`, `load_road_graph` |

Two non-exception failure signals belong next to that table, because a caller branches on them the same way. A planner that finds no path returns `None` rather than raising, and `TrajectoryResult` carries `is_feasible` and `optimizer_success` flags rather than raising when the solve fails.

## Stubs and deprecated names

| Name | State |
|---|---|
| `DStarLite`, `DStarPlanner` | Stubs. `search` and `plan` raise `NotImplementedError`. [ROADMAP.md](ROADMAP.md) records D* Lite as not planned |
| `BSplineInterpolator` | The symbol and its constructor are real; `interpolate` returns the path unchanged |
| `MPCController` | Deprecated scalar stub. `DubinsPathFollowingMPC` replaces it |
| `Graph.Node`, `Graph.Edge` | Nested placeholder types carrying no fields |

## `arco.simulator`, outside the ported surface

The simulator spans 42 modules, 5 of which declare an `__all__`, holding 34 entries between them. [rust/SPEC.md](rust/SPEC.md) places it outside the port, because its rendering stack stays Python and because it is an application surface that moves with the `arcosim` command instead of standing under a compatibility promise. The entries below are listed for completeness and carry no signature detail.

| Module | Entries |
|---|---|
| `arco.simulator.entity` | `BoxGeometry`, `SphereGeometry`, `Geometry`, `geometry_from_dict`, `Entity`, `DubinsAgent`, `CartesianAgent`, `RevoluteJoint`, `PrismaticJoint`, `Link`, `EndEffector`, `KinematicChain`, `Object` |
| `arco.simulator.main` | `city`, `occ`, `ppp`, `rrp` |
| `arco.simulator.scenes` | `ArcosimScene`, `RaceScene`, `SimScene` |
| `arco.simulator.sim` | `run_sim`, `ScreenLayout` |
| `arco.simulator.viewer` | `draw_graph`, `draw_grid`, `draw_road_network`, `draw_trace`, `format_clock`, `FrameRenderer`, `LayerStyle`, `parent_dict_to_list`, `polyline_length`, `SceneSnapshot`, `StandardLayout`, `TraceStyle` |

The `arcosim` console script resolves to `arco.simulator.__main__:main`.

## Related documents

- [rust/SPEC.md](rust/SPEC.md), the requirements this inventory is evidence for
- [rust/DEVIATIONS.md](rust/DEVIATIONS.md), every accepted difference between the two implementations
- [decisions.md](decisions.md), the decision log, including ADR-015 on configuration globals
- [FAILURE_MODES.md](FAILURE_MODES.md), constructor and planning failure contracts
- [ALGORITHMS.md](ALGORITHMS.md), the computational core behind these names
