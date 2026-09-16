# ARCO Rust conventions

ARCO-specific Rust conventions only. The shared Rust guideline in
[.guidelines/languages/rs.md](../../.guidelines/languages/rs.md) is
authoritative for everything not listed here: toolchain, workspace layout,
the `[workspace.lints]` tiers, unsafe policy, error enums, rustdoc
sections, test layout and naming, and the rules for a crate compiled as a
Python extension module. Cross-language defensive rules live in
[.guidelines/style/defensive.md](../../.guidelines/style/defensive.md) and
the rigor levels in
[.guidelines/workflow/criticality.md](../../.guidelines/workflow/criticality.md).

This file stands to Rust as [docs/guidelines.md](../guidelines.md) stands
to Python. Every adaptation it records also has an entry in
[DEVIATIONS.md](DEVIATIONS.md).

## 1. Criticality per crate

Assigned per
[.guidelines/workflow/criticality.md](../../.guidelines/workflow/criticality.md).
The level decides which lint tier a crate carries and which rules from
`defensive.md` are required rather than recommended.

| Crate | Level | Reason |
|---|---|---|
| `arco-core` | C2 | Geometry and RNG reached from every control path |
| `arco-mapping` | C2 | Collision checking decides whether a path is safe |
| `arco-control` | C2 | Produces commands for a machine |
| `arco-guidance` | C2 | Produces commands for a machine |
| `arco-kinematics` | C2 | Joint commands for an arm |
| `arco-planning` | C1 | Output is consumed by a C2 layer that validates it |
| `arco-runtime` | C1 | Transport, not computation |
| `arco-py` | C2 | A panic here is undefined behavior, see section 6 |

A C2 crate carries the hardened lint tier from the shared guideline, and
says so in its crate-level documentation.

## 2. Domain naming

The rules in [docs/guidelines.md](../guidelines.md) section 1 describe
robotics rather than Python, so they carry over unchanged:

- **Maps are nouns.** `Grid`, `Occupancy`, `WeightedGraph`. Passive data.
- **Planners are agent nouns taking `-er`.** `AStarPlanner` becomes
  `RrtPlanner`, and acts on a map.
- **Planners take a map as the first argument**, and the map family is a
  trait bound rather than a runtime check.
- **The who_what rule holds.** `max_speed`, never `speed_ms`. Units live
  in the doc comment or in a newtype, never in the identifier.

## 3. Acronym mapping

The shared guideline capitalizes only the leading letter of an acronym.
ARCO's Python names are fully capitalized, so each one is registered back
under its original spelling at the binding:

| Python | Rust | Registered as |
|---|---|---|
| `RRTPlanner` | `RrtPlanner` | `#[pyclass(name = "RRTPlanner")]` |
| `SSTPlanner` | `SstPlanner` | `#[pyclass(name = "SSTPlanner")]` |
| `KDTreeOccupancy` | `KdTreeOccupancy` | `#[pyclass(name = "KDTreeOccupancy")]` |
| `AStarPlanner` | `AStarPlanner` | unchanged, `A` is a single letter |
| `RRArm`, `RRPArm` | `RrArm`, `RrpArm` | `#[pyclass(name = "RRArm")]` and so on |
| `MPCController` | `MpcController` | `#[pyclass(name = "MPCController")]` |
| `PIDController` | `PidController` | `#[pyclass(name = "PIDController")]` |

No Python import changes. See `FR-API-01`.

## 4. File names during the port

The shared guideline names a file after its primary type and permits a
port to mirror the source module names instead, provided the project
records it. ARCO mirrors: `astar.py` becomes `astar.rs`, `kdtree.py`
becomes `kdtree.rs`, `pure_pursuit.py` becomes `pure_pursuit.rs`. Revisit
once `src/arco/` holds only the binding layer. Recorded as C-06.

## 5. Formatting and headers

`rustfmt` defaults. ARCO's 79-column Python width does not carry over, for
the reason the shared guideline gives. Recorded as C-02.

Every `.rs` file carries the same Apache 2.0 header the `.py` files carry,
as a `//` block above the crate or module doc comment.

## 6. The Python boundary

The shared guideline covers signatures, releasing the interpreter lock,
hook enums, borrowing arrays, `abi3`, and type stubs. Two ARCO additions:

**A panic must not escape `arco-py`.** Unwinding across the boundary is
undefined behavior, and PyO3 converts what it catches into an exception
derived from `BaseException`, which `except Exception` passes over and
which tends to end the interpreter. Every exported function returns a
`PyResult`, and the crate carries `panic = "unwind"` in its test profile
so `#[no_panic]` can run against the entry points. See `FR-SAFE-01`.

**The hook enums are named in the spec, not invented per planner.** The
policy hooks ARCO exposes (`sampler=`, `steerer=`, `segment_free=`,
`cost=`, `cost_terms=`, `heuristic=`, `feasibility=`, `publisher=`,
`nearest_obstacle_fn=`) each get one enum in the crate that owns the hook,
with a native variant per built-in policy and one `Python` variant. See
`FR-PERF-02` and ADR-004.

## 7. Tolerances

No bare epsilon anywhere in the workspace. Tolerances are named constants
in physical units, defined once in `arco-core` and referenced by name:

```rust
/// Positional agreement tolerance, meters.
pub const POSITION_TOLERANCE: f64 = 1e-6;
/// Angular agreement tolerance, radians.
pub const ANGLE_TOLERANCE: f64 = 1e-9;
```

Never use `f64::EPSILON` as a domain tolerance. Comparison follows
[.guidelines/style/defensive.md](../../.guidelines/style/defensive.md#numerical-rules):
an absolute tolerance near zero, a relative one away from it, and
`f64::total_cmp` for anything that sorts. The A\* open set orders on a
cost that can go non-finite, so its comparator is a total order by
construction rather than a partial comparison with the failure case
discarded.

## 8. Budgets

Every search, sampling, or solving entry point takes an explicit budget
and returns a result that distinguishes convergence from exhaustion:

| Entry point | Budget |
|---|---|
| `AStarPlanner::plan` | `max_expansions` |
| `RrtPlanner::plan`, `SstPlanner::plan` | `max_samples` |
| `TrajectoryOptimizer::solve` | `max_iterations` |
| MPC step | `max_sqp_iterations` |

No unlimited variant of the same function a control loop calls. See
`FR-SAFE-02`.

## 9. Commands

```bash
bash scripts/validate.sh
```

That script runs the shared guideline's Rust tooling table plus the
existing Python gates, in the order CI runs them.
