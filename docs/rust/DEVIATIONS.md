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

### A-03: graph inheritance becomes composition

**Status:** planned, phase 2.

`Graph` to `WeightedGraph` to `CartesianGraph` to `RoadGraph` is four
levels of Python inheritance. Rust has no inheritance, so the hierarchy
becomes traits plus composition. The Python-facing classes keep their
names, their methods, and their `isinstance` relationships through the
binding layer.

The exact mapping from each Python class to its Rust form is written into
this entry before the phase 2 code is written, per the plan.

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
