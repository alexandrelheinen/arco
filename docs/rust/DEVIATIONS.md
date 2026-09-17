# Port deviations

Every place the Rust port does not mirror the Python library exactly.
[SPEC.md](SPEC.md) requires this file under `FR-API-06` and `FR-DOC-03`:
an adaptation that is not recorded here is a bug, not a decision.

Two kinds of entry live here. A **convention** deviation changes how the
code is written and is invisible to callers. An **API** deviation changes
something a caller can observe, and needs review before it lands.

Entries are append-only. When a deviation is later reverted, mark it
resolved rather than deleting it.

## Conventions

### C-01: acronym casing in type names

**Status:** accepted, phase 0.

Python uses `RRTPlanner`, `SSTPlanner`, `KDTreeOccupancy`, `RRPArm`. Rust
uses `RrtPlanner`, `SstPlanner`, `KdTreeOccupancy`, `RrpArm`, because the
Rust API guidelines capitalize only the leading letter of an acronym and
`clippy` flags the alternative.

Invisible to callers: `#[pyclass(name = "RRTPlanner")]` registers the
original name, so every Python import and `repr` is unchanged.

### C-02: line length

**Status:** accepted, phase 0.

ARCO's Python is formatted at 79 columns, an accepted override of the
shared 88-column default. Rust code uses `rustfmt` defaults at 100
columns instead. Forcing 79 wraps generic bounds and `where` clauses
badly, and every Rust tool and reader expects the default.

### C-03: no underscore prefix on private members

**Status:** accepted, phase 0.

Python marks private members `_count`. Rust fields are private unless
declared `pub`, so the prefix carries no information. A leading
underscore in Rust means an intentionally unused binding only.

### C-04: unit tests are not mirrored

**Status:** accepted, phase 0.

ARCO mirrors `tests/` against `src/`. Rust unit tests for private items
must live in an inline `#[cfg(test)] mod tests`, because nothing outside
the file can reach a private item. Integration tests in
`crates/<crate>/tests/` stay mirrored, so the rule holds wherever it can.

### C-05: test formatting is enforced

**Status:** accepted, phase 0.

ARCO exempts `tests/` from Python formatting. `cargo fmt` covers the
whole workspace and excluding tests would need deliberate configuration
for no benefit, so Rust tests are formatted like everything else.

### C-06: file names mirror the Python modules

**Status:** accepted, phase 0.

The shared naming guide asks a file to be named after its primary type.
During the port a file instead keeps the name of the Python module it
replaces (`astar.py` becomes `astar.rs`), so the two trees can be read
side by side. Revisit once the Python sources are gone.

### C-07: criticality assigned per crate

**Status:** accepted, phase 0.

ARCO's Python guidelines apply one standard to `src/` and exempt `tests/`.
The Rust workspace assigns a criticality level per crate instead, so the
lint tier and the required defensive rules differ between `arco-control`
and `arco-runtime`. See [STYLE.md](STYLE.md#1-criticality-per-crate) and
ADR-008.

### C-08: the hardened lint tier lives in crate-root attributes

**Status:** accepted, phase 0. **Upstream fix pending.**

[.guidelines/languages/rs.md](../../.guidelines/languages/rs.md#hardened-for-real-time-unsafe-and-ffi-crates)
says to add the hardened tier as a `[lints.clippy]` table in the crate
that needs it. Cargo rejects that whenever the same crate also opts into
the workspace table, which every ARCO crate does:

```
cannot override `workspace.lints` in `lints`, either remove the overrides
or `lints.workspace = true` and manually specify the lints
```

The alternatives are restating the whole baseline tier in each hardened
crate, or expressing the hardened tier as crate-root attributes. ARCO
takes the second, which also puts the fact that a crate is hardened in
front of anyone reading its first screen rather than in a manifest.

The shared guideline is wrong here rather than ARCO being unusual, so this
entry closes when an upstream fix lands.

### C-09: protocol traits drop the `Like` suffix

**Status:** accepted, phase 1.

Four of the thirteen Python protocols carry a `Like` suffix:
`PlannerLike`, `OccupancyLike`, `OptimizerLike`, `PrunerLike`. That suffix
is a Python typing idiom marking a structural protocol as distinct from
the concrete class of the same name. Rust has no such collision, since the
trait and the implementing type are separate declarations, so the traits
are `Planner`, `Occupancy`, `Optimizer`, and `Pruner`.

The remaining nine keep their names exactly. No Python import changes,
because `arco.protocols` continues to export every original name.

### C-10: a ninth crate holds the test instrumentation

**Status:** accepted, phase 1.

[SPEC.md](SPEC.md) lays out eight crates. A ninth, `arco-testing`, holds
the allocation counter that `FR-SAFE-04` needs.

It exists because counting allocations means implementing
`core::alloc::GlobalAlloc`, which is unsafe to implement, and every
algorithm crate carries `#![forbid(unsafe_code)]`. `forbid` cannot be
relaxed for one module the way `deny` can, and weakening a C2 crate's
strongest stated guarantee to accommodate test-only code is the wrong
trade. One crate that permits `unsafe`, is never published, and is never a
runtime dependency is the cheaper answer.

The crate is C0 under
[workflow/criticality.md](../../.guidelines/workflow/criticality.md),
since nothing it does reaches a machine.

### C-11: the k-d tree is written here rather than taken from a crate

**Status:** accepted, phase 2.

[SPEC.md](SPEC.md) names `kiddo` as the replacement for
`scipy.spatial.KDTree`. `kiddo` fixes the dimension as a const generic
parameter, and ARCO promises N-dimensional maps chosen at run time, which
[docs/guidelines.md](../guidelines.md) section 6 states and which the
existing tests exercise.

The options were to support a fixed set of dimensions and reject the rest,
which breaks that promise, or to carry roughly 150 lines of tree in the
crate. The tree is the cheaper answer, and it buys two things a dependency
would not: the split order is decided by a total comparison, so an
obstacle set with repeated coordinates builds the same tree on every run,
and the content hash of `FR-INV-12` is computed over the same points the
tree was built from.

The tree is verified against a brute-force scan rather than against
itself: four dimensions, two hundred points, two hundred queries each,
agreeing to 1e-9.

## API

### A-09: control output gains saturation, rate limiting, and anti-windup

**Status:** planned, phase 6. Blocking review before merge.

Every command leaving the control layer passes one saturation function and
one rate limiter, and every integrator carries an anti-windup path. Where
the Python implementation already does this, behavior is unchanged. Where
it does not, the output changes, and phase 6 records which controllers
were affected and by how much.

This is the one place the port deliberately improves an algorithm rather
than translating it, which otherwise contradicts the rule in
[PLAN.md](PLAN.md#what-this-plan-does-not-do). The exception is taken
because shipping a controller that winds up is a defect, not a feature to
preserve, and because a differential test against a winding-up reference
would lock the defect in.

### A-10: tolerances become named domain constants

**Status:** planned, phase 1.

Comparisons that used an implicit tolerance now use a named constant in
meters, radians, or seconds. Where the new constant differs from what the
Python code used implicitly, a test asserting the old behavior may change
its result. Phase 1 lists the constants and the call sites they replace.

### A-01: `casadi` leaves the dependency list

**Status:** planned, phase 7. Blocking review before merge.

The `mpc` optional extra and the `casadi>=3.6` dependency are removed.
Anything importing `casadi` through ARCO breaks. Nothing in the public
API surface exposed CasADi types, so the break is limited to callers who
installed the extra for their own use.

### A-02: MPC solutions change

**Status:** planned, phase 7. Blocking review before merge.

The path-following and joint-space controllers solve a linearized problem
with Clarabel instead of a nonlinear program with IPOPT. Given the same
input they return a different, also valid, solution. `FR-MPC-02` bounds
the difference at 20 percent of root-mean-square lateral error, and
phase 7 commits the measured table here.

Any caller asserting exact MPC output values has to move to a tolerance.

### A-03: graph and grid inheritance becomes composition

**Status:** accepted, phase 2.

`Graph` is the base of three separate chains in Python, not one: the
weighted graph hierarchy, the grid family, and `Occupancy`. Rust has no
inheritance, so each chain becomes a struct that owns the layer beneath it
plus a trait for the search surface they share.

| Python class | Rust form |
|---|---|
| `Graph` | No struct. The shared surface is `arco_core::protocols::DiscreteMap`, which the search layer takes as a bound |
| `WeightedGraph` | `WeightedGraph`, owning the adjacency and the edge weights |
| `CartesianGraph` | `CartesianGraph`, owning a `WeightedGraph` plus node positions |
| `RoadGraph` | `RoadGraph`, owning a `CartesianGraph` plus per-edge geometry |
| `Grid` | `GridCells`, owning the extent, the cell size, and the cell states. Not public by itself |
| `ManhattanGrid` | `ManhattanGrid`, owning a `GridCells`, four-connected, L1 distance |
| `EuclideanGrid` | `EuclideanGrid`, owning a `GridCells`, eight-connected, L2 distance |
| `Occupancy` | `arco_core::protocols::Occupancy`, already a trait |
| `KDTreeOccupancy` | `KdTreeOccupancy`, per the acronym rule in C-01 |

Each owning struct exposes the inner layer's methods by delegation rather
than by inheritance, so `CartesianGraph::distance` still resolves and a
Python caller sees no difference. The binding layer preserves the
`isinstance` relationships the Python classes had.

Two consequences a reader should expect. A Rust caller holding a
`RoadGraph` reaches the weighted-graph methods through it rather than
through a base class, which is the same set of calls in a different
shape. And the grid metric is a property of the type rather than an
overridden method, so a four-connected grid with an L2 metric is not
constructible, which the Python version allowed by subclassing
incorrectly.

### A-11: a grid cell has three states, defaulting to free

**Status:** accepted, phase 2.

Python grids are binary: `set_occupied`, `set_free`, `is_occupied`.
`FR-INV-11` requires that an unknown cell never be treated as free unless
the caller asked for it, which needs a third state.

The Rust `Cell` carries `Free`, `Occupied` and `Unknown`. A newly built
grid is `Free` everywhere, matching the Python default so that
`FR-CORE-01` holds and no existing test changes. A grid built from sensor
data starts `Unknown` through a separate constructor, and the query that
treats unknown as blocked is the default while the permissive one is
named.

`is_occupied` keeps returning a boolean and keeps its Python meaning,
reporting true only for `Occupied`. Code that needs the distinction asks
for the cell state instead.

### A-04: Python 3.10 floor kept

**Status:** accepted, phase 0.

[.guidelines/languages/py.md](../../.guidelines/languages/py.md) targets
Python 3.12. ARCO ships `requires-python = ">=3.10"` today and keeps it,
and the wheels target `abi3-py310`. Raising the floor would drop users
for no gain the port needs.

### A-05: `arco.middleware` and `arco.pipeline` share one crate

**Status:** planned, phase 4.

Both Python packages are backed by `arco-runtime`. The Python import
paths stay separate and unchanged; only the crate layout merges them.

### A-06: the pipeline runner keeps a Python display layer

**Status:** planned, phase 4.

`src/arco/pipeline/runner.py` imports pygame at module level. The Rust
runtime crate carries no display dependency, so the pygame-dependent part
of the runner stays Python and calls into the crate. The split is
invisible to callers as long as the current import behavior is preserved,
which the existing tests check.

### A-07: policy hooks taking Python callables are slower

**Status:** planned, phase 5.

A caller-supplied Python callable in `sampler=`, `steerer=`,
`segment_free=`, `cost=`, `cost_terms=`, `heuristic=`, `feasibility=`,
`publisher=`, or `nearest_obstacle_fn=` forces the loop to reacquire the
GIL on every call, and the speedup of `FR-PERF-01` does not apply.
Behavior is unchanged; only timing is. Documented in
[docs/API.md](../API.md) under `FR-PERF-03`.

### A-08: trajectory optimizer results change

**Status:** planned, phase 5.

`scipy.optimize.minimize` becomes `argmin`. Two optimizers reach
different local minima on the same nonconvex problem. The differential
test compares achieved cost rather than the solution vector, and any
caller asserting exact optimizer output has to move to a cost-based
assertion.

### A-12: an exact segment check joins the sampled one

**Status:** accepted, phase 5.

Python checks a segment by sampling it a fixed number of times, and the
sampled policy is kept unchanged as the default so the planners behave as
they were tuned to. Alongside it, `SegmentPolicy::Exact` asks the
occupancy for the exact answer, computed from the distance between each
obstacle and the segment rather than from samples along it.

The sampled check cannot support `FR-INV-01` as written. A segment that
grazes an obstacle occludes a span that shrinks toward zero as the contact
gets shallower, so for any sample count there is a contact it steps over,
and re-checking the returned path more finely then finds a collision the
planner did not. The exact policy has no resolution and therefore no such
gap, which is why the invariant tests use it and why a result under the
sampled policy carries the resolution it was validated at.

### A-13: RRT* carries a rewiring saving down the subtree

**Status:** accepted, phase 5. Blocking review before merge.

When rewiring gives a node a cheaper parent, every node below it becomes
cheaper by the same amount. Python updates only the rewired node and
leaves its descendants holding their old costs, which is the common
shortcut. The port updates the subtree.

Costs change, and so do the paths that follow from them. Two things break
without the update. A later rewiring compares a candidate against a stale
cost that is too high, so it accepts a change that does not improve the
path and rejects one that does. And the cost the planner reports can rise
as the budget grows, since the node it selects carries a number that no
longer describes the path it traces, which is exactly what `FR-INV-07`
forbids. Measured on a fifty metre field, the reported cost under the
Python rule rises between two thousand and four thousand samples and falls
again after; under this rule it is monotone across the whole ladder and
converges to within half a percent of the straight line.

### A-14: SST rejects a witness radius of a whole step or more

**Status:** accepted, phase 5.

Python documents that the witness radius has to be under one normalized
step and does not enforce it. At a radius of one step or more every
candidate lands inside the region its own parent already holds and never
beats it, so the planner spends its whole budget rejecting candidates and
reports an exhausted budget on an open map. The port returns
`Error::OutOfRange` naming the radius instead.

### A-15: a discrete query naming a node outside the map says so

**Status:** accepted, phase 5.

A start or goal the map does not contain is reported as
`PlanFailure::StartOutsideMap` or `PlanFailure::GoalOutsideMap` before any
expansion, where Python either searched from a node that has no neighbors
and reported no path, or raised out of the map's own indexing. Off the
edge of the map and unreachable within it are different diagnoses: no
budget makes the first answerable and no obstacle caused it. `FR-INV-08`
requires the enumeration be closed and every reason in it reachable, and
these two had no producer before.

### A-16: `nearest_obstacle` measures from the obstacle surface

**Status:** accepted, phase 2, recorded in phase 6.

Python returns the distance from the query to the obstacle's center, and
every caller subtracts the clearance itself. The port returns the distance
to its surface, which is the number a planner asking about clearance
actually wants and the one that is directly comparable to zero.

Callers inside the port are converted. `ArtificialPotentialField` adds the
clearance back, because the potential is defined against the center
distance and its influence radius is expressed in those terms. Anything
outside the port calling `nearest_obstacle` gets a number smaller by the
clearance than it used to.

Recorded late. It was decided while porting `arco-mapping` and only
noticed to be unwritten when the control layer had to convert back.

### A-17: every control step validates the elapsed interval

**Status:** accepted, phase 6.

A step reads no clock, so the interval arrives as an argument. Python
documented that it had to be positive and checked nothing, which means a
negative interval after a clock adjustment, a zero one from two reads
inside a tick, and an enormous one after a stall all produced a command
that was arithmetically valid and physically wrong.

Every step in `arco-control` now takes the interval against a configured
[`IntervalBand`], one microsecond to one second by default, and returns
`Error::NotFinite` or `Error::OutOfRange` rather than computing with it.
`FR-INV-10` is the requirement. A caller passing an interval the Python
accepted silently now gets an error, which is the point.

### A-18: the PID controller takes the elapsed interval

**Status:** accepted, phase 6.

`arco.control.pid.PIDController` summed raw errors and differenced raw
errors, with no interval anywhere. That is this controller at an interval
of exactly one second, which is what the binding passes, so nothing
visible from Python changes and the existing tests hold. The Rust
signature takes the interval because a controller whose gains mean
different things at different sample rates is a controller nobody can
tune, and because `FR-INV-10` needs something to check.

One behavior does change at any interval: the first step after
construction or reset takes no derivative. Python differenced against an
initial previous error of zero, which makes the derivative term a spike
proportional to the initial error on the first call of every run.

### A-19: the tracking loop can bound its history

**Status:** accepted, phase 6.

The Python loop appended a metrics dictionary per step and never dropped
one, which is fine for a simulation and wrong for anything running longer.
`TrackingSettings::history_capacity` defaults to `None`, keeping every
sample as before; a caller with a real-time budget sets a capacity and
gets a ring buffer, and `Some(0)` keeps none at all since the step returns
its sample anyway. `FR-SAFE-04` holds only under a bounded history, which
the allocation test states directly.

### A-20: the vehicle takes one limit set rather than five attributes

**Status:** accepted, phase 8.

`DubinsVehicle` carried `max_speed`, `min_speed`, `max_turn_rate`,
`max_acceleration` and `max_turn_rate_dot` as five separate attributes.
The port holds one `arco_control::limits::CommandLimits`, which is the
same type every other command in the control layer passes through, so a
vehicle and a controller cannot be configured with limits that disagree.

Two names change with it: `max_acceleration` becomes `max_speed_rate` and
`max_turn_rate_dot` becomes `max_turn_rate_change`, because the shared
type names the quantity rather than the axis it happens to bound. The
binding maps both spellings, so `FR-API-02` holds and Python callers see
no change. `DubinsVehicle::default_limits()` carries the Python defaults
unchanged.

### A-21: the guidance layer validates what Python assumed

**Status:** accepted, phase 8.

Python checked none of these and produced a plausible-looking answer for
each. The port returns a typed error, which the binding raises as
`ValueError`:

| Input | Python | Port |
|---|---|---|
| A turning radius that is not finite and positive | Divided by it | `Error::OutOfRange` |
| An iteration count of zero | Clamped up to one | `Error::TooFew` |
| A B-spline degree of zero | Produced a degenerate curve | `Error::OutOfRange` |
| A segment duration at or below zero | Floored at 1e-9, returning a saturated turn rate | `Error::OutOfRange` |
| A state shorter than two components | Read past the end | `Error::TooFew` |
| A non-finite waypoint or state | Propagated the NaN | `Error::NotFinite` |

The last one is the reason the rest are here. A NaN compares false against
every bound, so the Python `is_feasible` reported a state carrying one as
feasible, which is the single answer that cannot be right. `is_feasible`
is now fallible for that reason rather than returning a bare boolean.

### A-22: the heading wraps into a half-open turn

**Status:** accepted, phase 8.

`arco_core::geometry::Pose` wraps its heading into `[-pi, pi)`, where
Python's `atan2(sin, cos)` produced `(-pi, pi]`. The two disagree at
exactly `pi` and nowhere else: a heading of `pi` reads back as `-pi`. Both
name the same direction, and `FR-INV-15` asks only that a rotation
representation be normalized on every return, which both satisfy.

### C-12: two guidance placeholders are ported as placeholders

**Status:** accepted, phase 8.

`BSplineInterpolator.interpolate` returns its input unchanged and
`DubinsPrimitive.steer` returns only the two endpoints. Neither is
implemented in Python either, and `tests/guidance/test_guidance.py` pins
both behaviors, so the port reproduces them rather than raising.

That contradicts the rule in
[.guidelines/workflow/tdd.md](../../.guidelines/workflow/tdd.md#marking-intentionally-unimplemented-work)
against a stub that silently returns success. The rule is about new work,
and marking these `todo!()` would fail a suite that phase 8 has to leave
passing unmodified. Both carry the warning in their own documentation so
a caller reaching for either is told before it depends on one.

### A-23: `Grid` stops being instantiable

**Status:** accepted, phase 9.

`Grid.neighbors` carries `@abstractmethod`, but `Grid` never inherits
`ABCMeta`, so the decorator is inert and `Grid(shape=(3, 3))` succeeds.
Calling `neighbors` on the result returns `None`, and the caller meets it
as `TypeError: 'NoneType' object is not iterable` somewhere further along.
It is the only class in the surface with that shape.

The port cannot reproduce "abstract in intent, instantiable in fact"
without deciding to. `Grid` becomes the `DiscreteMap` trait of A-03, so
constructing the base is a compile error rather than a deferred one, and
`ManhattanGrid` or `EuclideanGrid` is what a caller builds. Anything that
was constructing `Grid` directly was already broken.

### A-24: subclassing a base class crosses the interpreter lock

**Status:** accepted, phase 9.

A-07 names the nine keyword hooks and says a Python callable in one of
them gives up the speedup. That is narrower than what callers actually do:
`ContinuousPlanner`, `DiscretePlanner`, `PlannerCost`, `Grid`,
`Occupancy`, `RigidBody`, `Interpolator`, `ExplorationPrimitive`,
`Controller`, `MPCTracker`, `PipelineNode` and `Bus` all expose
overridable methods, the existing tests override several of them, and an
override reached from inside a loop costs exactly what a keyword hook
costs.

Subclassing keeps working. What changes is that the cost is now stated:
overriding `distance`, `heuristic`, `neighbors`, `is_occupied`,
`nearest_obstacle`, `steer`, `sample` or `is_segment_free` puts a Python
call on the inner loop, and `FR-PERF-01` does not apply to a planner
carrying one. The native path is the one where nothing is overridden.

### A-25: configuration is not read from the environment at import

**Status:** accepted, phase 9.

`load_config` resolves its directory from `ARCO_CONFIG_DIR` and caches the
result in a module-level global, and three `create_from_config` static
methods read it. Setting the variable after any `arco` module is imported
therefore has no effect, and the ordering is invisible at the call site.

ADR-015 records why the crates do not reproduce this: configuration is
parsed at construction and held by the object that uses it. The
API-visible consequence, which `FR-API-06` asks be written here rather
than only in the decision log, is that a caller relying on the import-time
environment read has to pass the directory explicitly instead.

### A-26: the `tools` extra lists the package as its own dependency

**Status:** open, phase 11.

`[project.optional-dependencies] tools` contains `"arco"`. Pip resolves it
to the package being installed and moves on, so nothing breaks today, and
it is repaired when phase 11 rewrites the packaging metadata rather than
in the middle of a phase that is not about packaging.

### A-27: bound arguments are positional only

**Status:** accepted, phase 9.

`PyO3` generates a positional-only signature by default, so a method that
read `(self, q1, q2, z)` in Python reads `(self, /, q1, q2, z)` once it is
compiled. Argument names, their order and their defaults are unchanged;
what is lost is the ability to pass them by keyword, so
`robot.forward_kinematics(q1=1.0, q2=0.5, z=2.0)` raises where the
positional form still works.

Restoring keyword calls means an explicit `#[pyo3(signature = (...))]` on
every bound method across seven binding modules, a few hundred of them.
The cost was weighed against adapting the handful of call sites that use
keywords, and adapting the callers won.

`FR-API-02` is narrowed to match: it covers the names, the order and the
defaults, and no longer covers keyword-callability. The parity test in
`tests/rust/test_signature_parity.py` normalizes the marker away and
compares the rest, so a genuine change of an argument name, its position
or its default still fails.

Two things the same test used to report and no longer does, because the
measurement was wrong rather than the port:

A compiled class does not hold inherited members in its own dictionary.
`ManhattanGrid.neighbors` comes from `Grid`, and reading `vars()` called
it a name that had vanished while `ManhattanGrid().neighbors` worked
exactly as before. The capture resolves members through the class instead.

A class built by `PyO3` carries its constructor signature on the class,
through `__text_signature__`, rather than on an `__init__` it does not
define. The capture reads it from there, so `__init__` and `__new__` are
treated as the one construction contract they are.

### A-28: configuration values are coerced, not extracted

**Status:** accepted, phase 9.

`PyYAML` follows the 1.1 spec, where a float's exponent needs a sign, so
`1.0e2` in `src/arco/config/optimizer.yml` parses as the string `"1.0e2"`
and four of the five optimizer weights arrive as text. The Python original
wrapped every read in `float(...)` and never noticed. The binding extracted
instead, and `TrajectoryOptimizer.create_from_config` died with
`TypeError: must be real number, not str` against the repository's own
configuration file, which four shipped simulator scenes call.

The readers coerce, as the Python did. The alternative, quoting the
exponents in the YAML, fixes one file and leaves every other caller's
configuration to fail the same way.

### A-29: a hook that raises surfaces on the call that raised

**Status:** accepted, phase 9.

Two trait methods a Python hook can be reached through, `neighbors` and
the feasibility check, cannot return an error, so an exception raised
inside one is parked and an empty answer handed back. The search would
then finish, report no path, and leave the exception held.

Two things went wrong with that. The call that raised reported
`last_failure` as "no path exists", which contradicts `FR-API-04`. And the
parked exception outlived its call: the next genuine failure surfaced the
stale one instead, so a `ValueError` raised by a second plan came back as
a `KeyError` from the first.

Every call now drains the slot whether the search succeeded or not, and
raises what it finds.
