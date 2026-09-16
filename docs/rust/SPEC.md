# Spec for the ARCO Rust port

Status: reviewed

See [.guidelines/workflow/sdd.md](../../.guidelines/workflow/sdd.md) for how
this fits the overall process, and
[.guidelines/workflow/integration.md](../../.guidelines/workflow/integration.md)
for how it feeds the V-cycle. The ordered implementation plan lives in
[PLAN.md](PLAN.md); every adaptation this spec permits is recorded in
[DEVIATIONS.md](DEVIATIONS.md).

## Intent

ARCO ships 26,779 lines of Python across ten packages. The sampling-based
planners, the tracking loops, and the trajectory optimizer all run
sample-collide-rewire loops in interpreted code, and the GIL blocks the
obvious parallelism in multi-start planning and batch simulation. This port
moves the algorithm core to Rust and keeps the Python package as a thin
binding layer, so existing callers keep their imports, their call syntax,
and their test suite while the work underneath runs native and in parallel.

The port also settles a structural problem the current code cannot solve:
`arco.protocols` declares thirteen `runtime_checkable` protocols, and
`runtime_checkable` verifies method names only, never signatures. A wrong
signature on an injected sampler or cost term fails at call time, deep in a
planner loop. Rust traits move that check to compile time.

## Scope

**In scope:**

- Rust reimplementation of `arco.protocols`, `arco.config`,
  `arco.mapping`, `arco.planning`, `arco.control`, `arco.guidance`,
  `arco.kinematics`, `arco.middleware`, and `arco.pipeline`.
- A PyO3 binding crate that re-exposes all 155 non-simulator public names
  under their current import paths, with unchanged call syntax.
- Reformulation of the path-following and joint-space MPC from a CasADi
  nonlinear program to a linear-time-varying MPC solved by sequential
  quadratic programming.
- Native dispatch for the built-in policy hooks (`sampler=`, `steerer=`,
  `segment_free=`, `cost_terms=`, telemetry `publisher=`) so the common
  path never crosses the Python boundary inside a loop.
- Build, packaging, lint, and documentation tooling for a mixed
  Rust/Python repository.
- Rust conventions documented in [STYLE.md](STYLE.md) and referenced from
  [docs/guidelines.md](../guidelines.md).

**Out of scope:**

- `arco.simulator` (15,120 lines). Its rendering stack stays Python for
  the duration of this work. The replacement design, a Rust simulation
  core that writes logs plus a replay viewer, is a separate effort tracked
  in [docs/ROADMAP.md](../ROADMAP.md) and depends on this port landing
  first.
- Any change to the public Python API surface beyond what
  [DEVIATIONS.md](DEVIATIONS.md) records and this spec's acceptance
  criteria permit.
- New algorithms. D* Lite stays a stub, as
  [docs/ROADMAP.md](../ROADMAP.md) states.
- Removal of the Python sources. The Python implementation stays in the
  tree as a differential-test oracle until the crate that replaces it
  passes, and is deleted module by module, never in one commit.

## Acceptance criteria

Traceability scheme: `FR-<DOMAIN>-<NN>`, append-only, never renumbered or
reused. Domains: `API`, `CORE`, `MPC`, `PERF`, `RNG`, `BUILD`, `TEST`,
`DOC`.

### Public API compatibility

- `FR-API-01`: When a caller imports any of the 155 public names listed in
  [API.md](../API.md) outside `arco.simulator`, the system shall resolve
  the import from the same module path as the Python implementation did.
- `FR-API-02`: When a caller invokes a ported public function or
  constructor with the argument names, positional order, and default
  values accepted by the Python implementation, the system shall accept
  the call without modification at the call site.
- `FR-API-03`: When a ported function returns an array, the system shall
  return a `numpy.ndarray` of the same shape and dtype the Python
  implementation returned.
- `FR-API-04`: When a ported function raises, the system shall raise an
  exception whose type is the Python exception type the Python
  implementation raised, or a subclass of it.
- `FR-API-05`: When a caller passes a Python callable to a documented
  policy hook (`sampler=`, `steerer=`, `segment_free=`, `cost=`,
  `cost_terms=`, `publisher=`, `heuristic=`, `feasibility=`,
  `nearest_obstacle_fn=`), the system shall call it and honor its result.
- `FR-API-06`: When the port changes any observable element of the public
  API, the system shall carry an entry in [DEVIATIONS.md](DEVIATIONS.md)
  naming the symbol, the old behavior, the new behavior, and the reason.

### Core behavior

- `FR-CORE-01`: When the existing `tests/` suite runs unmodified against
  the Rust-backed package, the suite shall pass, except for tests that
  [DEVIATIONS.md](DEVIATIONS.md) records as intentionally changed.
- `FR-CORE-02`: When a planner receives a map object of the wrong family
  (a `Grid` where an `Occupancy` is required, or the reverse), the system
  shall raise `TypeError` before starting the search, matching
  [docs/guidelines.md](../guidelines.md) section 5.
- `FR-CORE-03`: When a `CartesianGraph` or `Grid` method receives a
  position array whose dimension the planner does not support, the system
  shall raise `ValueError`.
- `FR-CORE-04`: When a Rust crate in the workspace declares a dependency
  on another ARCO crate, the dependency shall follow the layering in
  [Design notes](#design-notes), which the Cargo workspace enforces at
  compile time.

### MPC reformulation

- `FR-MPC-01`: When the path-following controller runs a control step, the
  system shall solve a convex program built by linearizing the unicycle
  model and the contouring, lag, heading, progress, and control cost terms
  about the previous solution, rather than calling CasADi.
- `FR-MPC-02`: When the reformulated controller tracks any reference path
  in the existing MPC test set, the root-mean-square lateral error shall
  stay within 20 percent of the error the CasADi implementation produced
  on the same input, measured by the differential harness of
  `FR-TEST-02`.
- `FR-MPC-03`: When the reformulated controller returns a solution, every
  state and input constraint that the CasADi formulation enforced shall
  hold on that solution, within the solver's reported primal feasibility
  tolerance.
- `FR-MPC-04`: When the solver fails to find a feasible solution, the
  system shall return the same failure signal the Python implementation
  returned for an infeasible CasADi solve, and shall not panic.
- `FR-MPC-05`: When the controller runs one control step on the reference
  hardware profile, the median solve time shall not exceed the CasADi
  implementation's median solve time on the same input.
- `FR-MPC-06`: When a user installs ARCO, `casadi` shall not be a required
  or optional dependency of the package.

### Performance

- `FR-PERF-01`: When `RRTPlanner.plan` runs on the benchmark scenario
  defined in `benches/`, with all policy hooks left at their defaults, the
  Rust implementation shall complete at least ten times faster than the
  Python implementation on the same machine.
- `FR-PERF-02`: When a caller leaves a policy hook at its default, the
  system shall dispatch that hook natively and shall not acquire the GIL
  inside the planner loop.
- `FR-PERF-03`: When a caller injects a Python callable into a policy
  hook, the system shall document in [API.md](../API.md) that the loop
  reacquires the GIL per call and that the speedup of `FR-PERF-01` does
  not apply.
- `FR-PERF-04`: When a caller runs independent planning problems through
  the documented batch entry point, the system shall run them in parallel
  on all available cores without a GIL-bound section.

### Random number generation

- `FR-RNG-01`: When a sampling planner runs twice with the same integer
  seed, the same map, and the same hooks, the system shall produce the
  identical result.
- `FR-RNG-02`: When a sampling planner runs with a given seed, the system
  shall draw from a PCG64 stream initialized by the same seeding procedure
  `numpy.random.default_rng` uses, so that results match the Python
  implementation value for value.
- `FR-RNG-03`: When a caller passes a `numpy.random.Generator` instance to
  a `sampler=` hook signature, the system shall keep accepting that
  signature.

### Build and packaging

- `FR-BUILD-01`: When a user runs `pip install arco` on Linux, macOS, or
  Windows for CPython 3.10 or later, the system shall install a prebuilt
  wheel with no Rust toolchain present.
- `FR-BUILD-02`: When the wheel is built, it shall target the `abi3-py310`
  stable ABI, so one wheel per platform covers every supported CPython
  version.
- `FR-BUILD-03`: When a contributor runs `bash scripts/validate.sh`, the
  system shall run the same checks CI runs, in the same order, and exit
  non-zero on the first blocking failure.
- `FR-BUILD-04`: When CI runs on a pull request, it shall run
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, the Python test suite against the built extension, and the
  existing formatting gate on the remaining Python sources.
- `FR-BUILD-05`: When the Rust crates build, `cargo build` shall emit no
  warnings.

### Testing

- `FR-TEST-01`: When a Rust module replaces a Python module, the Python
  tests that covered that module shall run unchanged against the
  replacement, and shall not be rewritten to match the new implementation.
- `FR-TEST-02`: When a Rust module replaces a Python module, a
  differential test shall run both implementations on the same inputs and
  compare the outputs within a tolerance the test states explicitly.
- `FR-TEST-03`: When the Rust test suite runs, line, region, and function
  coverage on the ported crates shall each stay at or above 80 percent,
  enforced in CI, matching the gate in
  [.guidelines/workflow/tdd.md](../../.guidelines/workflow/tdd.md). Branch
  coverage is measured where available and is not gated, for the reason
  ADR-007 records.
- `FR-TEST-04`: When a Rust function is left unimplemented on purpose, it
  shall call `todo!()` with a reason string, and its test shall carry
  `#[should_panic(expected = ...)]` naming the requirement that blocks it.

### Soundness and defensive rules

These implement
[.guidelines/style/defensive.md](../../.guidelines/style/defensive.md) at
the criticality levels assigned in [STYLE.md](STYLE.md#1-criticality-per-crate).

- `FR-SAFE-01`: When a panic occurs inside `arco-py`, the system shall
  convert it into a Python exception at the boundary and shall not unwind
  across it.
- `FR-SAFE-02`: When a caller invokes a search, sampling, or solving
  entry point, the system shall accept an explicit budget and shall
  return a result that distinguishes convergence from budget exhaustion.
- `FR-SAFE-03`: When a ported crate is built, it shall contain no direct
  or indirect recursion, and the tree and graph searches shall use an
  explicit stack of stated capacity.
- `FR-SAFE-04`: When a controller executes one step, it shall perform no
  heap allocation, and a test shall prove this with an instrumented
  allocator.
- `FR-SAFE-05`: When the system orders floating-point values, whether to
  sort, to key a map, or to select a minimum, it shall use a total order
  and shall not discard the failure case of a partial comparison.
- `FR-SAFE-06`: When the system compares floating-point values for
  agreement, it shall use a named tolerance constant expressed in a
  physical unit, and shall not use the machine epsilon as a domain
  tolerance.
- `FR-SAFE-07`: When a non-finite value crosses a public API boundary in
  either direction, the system shall reject it with a typed error rather
  than propagate it.
- `FR-SAFE-08`: When CI builds the workspace, the lint policy shall come
  from the `[workspace.lints]` table, and every C2 crate shall carry the
  hardened tier from
  [.guidelines/languages/rs.md](../../.guidelines/languages/rs.md#lints).

### Algorithm invariants

Each is asserted in the implementation and proved by a test. Most are
metamorphic relations, chosen because every algorithm here has the oracle
problem: the optimal path is unknown, while how the output must change
under a known input change is not.

- `FR-INV-01`: When a planner returns a path, that path shall be
  collision-free when re-checked against the same map object it was given,
  at a validity resolution at least as fine as the planning resolution,
  and the result shall carry that resolution.
- `FR-INV-02`: When a planner returns a path, consecutive states shall be
  connected by a motion the segment checker accepts.
- `FR-INV-03`: When path simplification or smoothing runs, it shall not
  turn a valid path invalid and shall not increase path cost.
- `FR-INV-04`: When an obstacle is removed from a map, the optimal cost
  shall not increase; when one is added, it shall not decrease.
- `FR-INV-05`: When the map, the start and the goal are translated and
  rotated together, the returned path shall transform by the same amount,
  for a fixed seed.
- `FR-INV-06`: When `AStarPlanner` runs with an admissible heuristic, the
  returned cost shall equal the cost an uninformed search returns on the
  same graph.
- `FR-INV-07`: When `RrtPlanner` runs, the incumbent solution cost shall
  be monotonically non-increasing across iterations. When `SstPlanner`
  runs, the incumbent cost shall stay inside a stated suboptimality band,
  since SST is asymptotically near-optimal rather than optimal.
- `FR-INV-08`: When a planner or controller declines to produce a result,
  it shall return a reason drawn from a closed enumeration, and every
  reason in that enumeration shall be reachable from the test suite.
- `FR-INV-09`: When a controller returns a command, that command shall lie
  inside the configured limit box and shall be reachable from the previous
  command under the configured rate limit, given the interval passed in.
- `FR-INV-10`: When a controller receives an elapsed interval that is not
  finite, not strictly positive, or outside its configured band, it shall
  return a typed error rather than compute with it.
- `FR-INV-11`: When a map is queried, index and world coordinates shall
  round-trip for every in-bounds cell, an out-of-bounds index shall be
  rejected by the safe accessor, and an unknown cell shall never be
  treated as free unless the caller asked for that explicitly.
- `FR-INV-12`: When a map object exists, its resolution, extent and origin
  shall be immutable, and it shall expose a content hash, so that
  `FR-INV-01` is a checkable claim rather than an assertion about an
  unnamed map.
- `FR-INV-13`: When collision checking and inflation both use a footprint,
  they shall use the same footprint object.
- `FR-INV-14`: When inverse kinematics returns a solution, it shall lie
  inside joint limits, forward kinematics shall reproduce the requested
  pose within a stated tolerance, and a configuration whose commanded
  joint rate would exceed a stated bound shall be rejected rather than
  returned.
- `FR-INV-15`: When any angular difference feeds a control law, it shall
  be the wrapped difference, and a rotation representation shall be
  normalized on every return.

### Documentation

- `FR-DOC-01`: When a Rust item is public, it shall carry a doc comment
  with the sections [STYLE.md](STYLE.md) requires, and CI shall fail on a
  missing one.
- `FR-DOC-02`: When a public Python name is backed by Rust, its docstring
  and type stub shall stay available to `help()`, `mypy`, and IDE
  completion.
- `FR-DOC-03`: When the port adapts any ARCO convention from Python to
  Rust, [STYLE.md](STYLE.md) shall state the Rust form and the reason it
  differs.

## Traceability

| ID | Test(s) |
|---|---|
| `FR-API-01` | `tests/rust/test_import_surface.py` |
| `FR-API-02` | `tests/rust/test_signature_parity.py` |
| `FR-API-03` | existing `tests/` array assertions; `tests/rust/test_array_roundtrip.py` |
| `FR-API-04` | `tests/rust/test_exception_parity.py` |
| `FR-API-05` | existing hook tests in `tests/planning/`, `tests/control/` |
| `FR-API-06` | review gate, not a test |
| `FR-CORE-01` | the whole existing `tests/` suite |
| `FR-CORE-02` | `tests/planning/` map-family tests |
| `FR-CORE-03` | `tests/mapping/test_weighted_graph.py` and siblings |
| `FR-CORE-04` | `cargo build` on the workspace |
| `FR-MPC-01` | `crates/arco-control/tests/` linearization tests |
| `FR-MPC-02` | `tests/rust/test_mpc_differential.py` |
| `FR-MPC-03` | `crates/arco-control/tests/` constraint tests |
| `FR-MPC-04` | `tests/control/` infeasibility tests |
| `FR-MPC-05` | `benches/mpc.rs` |
| `FR-MPC-06` | `tests/rust/test_no_casadi.py` |
| `FR-PERF-01` | `benches/planning.rs` |
| `FR-PERF-02` | `benches/planning.rs` with a GIL-counting harness |
| `FR-PERF-03` | documentation gate, not a test |
| `FR-PERF-04` | `tests/rust/test_batch_parallel.py` |
| `FR-RNG-01` | `tests/planning/continuous/test_telemetry_rng.py` |
| `FR-RNG-02` | `crates/arco-core/tests/pcg64.rs` against captured numpy vectors |
| `FR-RNG-03` | existing `sampler=` hook tests |
| `FR-BUILD-01` | release workflow wheel smoke job |
| `FR-BUILD-02` | release workflow wheel tag assertion |
| `FR-BUILD-03` | `scripts/validate.sh` self-check |
| `FR-BUILD-04` | `.github/workflows/tests.yml` |
| `FR-BUILD-05` | `.github/workflows/tests.yml` |
| `FR-TEST-01` | the whole existing `tests/` suite |
| `FR-TEST-02` | `tests/rust/differential/` |
| `FR-TEST-03` | `cargo llvm-cov` gate in CI |
| `FR-TEST-04` | `cargo test` |
| `FR-SAFE-01` | `tests/rust/test_no_panic_escapes.py`, `#[no_panic]` link job |
| `FR-SAFE-02` | `tests/rust/test_budgets.py`, per-crate budget tests |
| `FR-SAFE-03` | `cargo clippy` plus a review gate; no lint covers this |
| `FR-SAFE-04` | `crates/arco-control/tests/no_alloc.rs` with a counting allocator |
| `FR-SAFE-05` | `crates/arco-core/tests/ordering.rs`, non-finite cost cases |
| `FR-SAFE-06` | Review gate plus a grep for bare float literals in comparisons |
| `FR-SAFE-07` | `tests/rust/test_non_finite_rejected.py` |
| `FR-SAFE-08` | `.github/workflows/tests.yml` |
| `FR-INV-01` | `tests/rust/invariants/test_path_collision_free.py` |
| `FR-INV-02` | `tests/rust/invariants/test_path_connected.py` |
| `FR-INV-03` | `tests/rust/invariants/test_smoothing_metamorphic.py` |
| `FR-INV-04` | `tests/rust/invariants/test_obstacle_monotonicity.py` |
| `FR-INV-05` | `tests/rust/invariants/test_rigid_transform_equivariance.py` |
| `FR-INV-06` | `crates/arco-planning/tests/astar_vs_dijkstra.rs` |
| `FR-INV-07` | `crates/arco-planning/tests/anytime_cost.rs` |
| `FR-INV-08` | `tests/rust/invariants/test_failure_taxonomy.py` |
| `FR-INV-09` | `crates/arco-control/tests/limits.rs` |
| `FR-INV-10` | `crates/arco-control/tests/interval.rs` |
| `FR-INV-11` | existing `tests/mapping/`, extended |
| `FR-INV-12` | `crates/arco-mapping/tests/identity.rs` |
| `FR-INV-13` | review gate plus a construction-time assertion |
| `FR-INV-14` | existing `tests/kinematics/`, extended |
| `FR-INV-15` | `crates/arco-core/tests/angles.rs` |
| `FR-DOC-01` | `#![deny(missing_docs)]` plus `cargo doc` in CI |
| `FR-DOC-02` | `tests/rust/test_docstrings.py`, `mypy --strict` |
| `FR-DOC-03` | review gate, not a test |

## Constraints

- **Python floor stays 3.10.** `pyproject.toml` declares
  `requires-python = ">=3.10"` today. The `abi3-py310` target keeps that
  floor. The shared guideline targets 3.12; ARCO keeps 3.10 and
  [DEVIATIONS.md](DEVIATIONS.md) records why.
- **Rust edition 2021**, per
  [.guidelines/languages/rs.md](../../.guidelines/languages/rs.md). The
  minimum supported Rust version is pinned in `rust-toolchain.toml` and
  raised only in a commit that states the feature requiring it.
- **Platforms**: Linux, macOS (x86_64 and aarch64), and Windows x86_64.
  The simulator's display requirements do not apply, since the ported
  crates draw nothing.
- **No C dependencies in the default build.** This rules out IPOPT and
  OSQP's C core, and is the reason
  [DECISIONS](../decisions.md) selects Clarabel.
- **`numpy` stays a runtime dependency**, since `FR-API-03` requires
  returning `numpy.ndarray`. `scipy` and `casadi` leave the dependency
  list once their users are ported.
- **The existing test suite is the contract.** A change that makes a
  current test fail is a breaking change and needs a
  [DEVIATIONS.md](DEVIATIONS.md) entry plus review, never a quiet edit to
  the test.
- **Criticality is assigned per crate** in
  [STYLE.md](STYLE.md#1-criticality-per-crate), and decides which rules in
  [.guidelines/style/defensive.md](../../.guidelines/style/defensive.md)
  are required rather than recommended. Breaking one needs a deviation
  record in [docs/decisions.md](../decisions.md), per
  [.guidelines/workflow/criticality.md](../../.guidelines/workflow/criticality.md#deviations).
- **Performance budget for the binding layer**: array conversion at the
  Python boundary is allowed to copy on the way out, and must not copy on
  the way in. A planner call that crosses the boundary once per `plan()`
  is fine; one that crosses per loop iteration is not.

## Design notes

### Crate layout and layering

The Cargo workspace mirrors the Python package tree, one crate per
package, so the existing import paths and the crate graph stay legible
against each other. The Python dependency graph, measured from the
current sources, is acyclic once the `TYPE_CHECKING`-only import in
`src/arco/control/mpc/tracking_loop.py:8` is discounted, so the crate
graph can enforce it:

```text
arco-core        protocols (traits), config, geometry, errors, RNG
  ├─ arco-mapping      grids, graphs, KD-tree occupancy
  │    └─ arco-planning    discrete + continuous planners, optimizer
  ├─ arco-kinematics   RR / RRP arm models
  ├─ arco-control      PID, Pure Pursuit, MPC, tracking, rigid bodies
  │    └─ arco-guidance    interpolation, primitives, vehicles
  └─ arco-runtime      middleware bus, pipeline runner
arco-py          PyO3 bindings over all of the above
```

`arco-guidance` depends on `arco-control` because
`src/arco/guidance/__init__.py:17` re-exports five controller names. The
reverse edge exists only under `TYPE_CHECKING` and disappears in Rust,
where `DubinsVehicle` is reached through a trait the control crate
defines.

`arco-runtime` merges `arco.middleware` and `arco.pipeline`, which are
471 and 347 lines and already form a closed pair. The Python import paths
`arco.middleware` and `arco.pipeline` stay separate regardless of which
crate backs them.

### Traits replace protocols

Each of the thirteen `Protocol` classes in `arco/protocols/` becomes a
trait of the same name in `arco_core::protocols`. The binding layer keeps
`arco.protocols.Planner` importable and `runtime_checkable`, so Python
code that checks `isinstance(x, Planner)` keeps working, while Rust code
gets a compile-time check instead.

### Policy hooks: native dispatch with a Python fallback

`rrt.py`, `sst.py`, `optimizer.py`, `astar.py`, `pruner.py`, and
`actuator.py` all accept injected callables inside their inner loops. A
naive binding calls back into Python per sample and reacquires the GIL
millions of times per `plan()`, which erases the speedup entirely. Every
hook is therefore an enum:

```rust
enum Steerer {
    Dubins(DubinsSteerer),
    Straight(LineSteerer),
    Python(Py<PyAny>),
}
```

The binding recognizes the built-in defaults and the built-in policy
objects and maps them to native variants. Only a caller-supplied Python
callable takes the `Python` variant, and that path is documented as slow
per `FR-PERF-03`.

### MPC: what the reformulation keeps and what it changes

The current `path_following.py` builds a nonlinear MPCC in `ca.Opti()`:
B-spline interpolants of the reference path over arclength, contouring
and lag errors, a heading term, progress tracking, a control-effort term,
a deadzone on contour error, exponential obstacle barriers, a turn-rate
constraint, and a terminal cost, all handed to IPOPT.

The Rust version keeps the same state and input vectors, the same cost
terms, and the same tuning parameters, and changes how the problem is
solved:

| Element | CasADi form | Rust form |
|---|---|---|
| Path lookup | `ca.interpolant` B-spline | precomputed cubic spline, evaluated natively |
| Model | nonlinear unicycle, symbolic | per-step linearization about the previous solution |
| Contour and lag cost | exact nonlinear expression | first-order expansion about the reference arclength, giving a quadratic |
| Heading cost | `sin^2 + (1 - cos)^2` | quadratic in the small-angle expansion about the reference heading |
| Contour deadzone | `fmax(abs(e) - dz, 0)` | slack variable with two linear constraints |
| Obstacle barrier | exponential penalty on distance | second-order cone constraint per obstacle per step |
| Turn-rate limit | `v_s * sqrt(k^2 + eps) <= max` | second-order cone constraint |
| Solver | IPOPT through CasADi | Clarabel, called once per SQP iteration |
| Outer loop | single nonlinear solve | one to three SQP iterations, warm-started from the previous control step |

Clarabel handles second-order cones directly, so the obstacle and
turn-rate constraints stay closer to their original geometry than a
pure-QP formulation would allow, and the crate is pure Rust with no C
dependency. The SQP iteration count is a tuning parameter with a default
this spec does not fix; `FR-MPC-02` and `FR-MPC-05` together set the
bound it has to satisfy.

Bit-identical agreement with CasADi is not achievable and is not
required. `FR-MPC-02` states the tolerance instead, measured on tracking
error rather than on the raw solution vector, because two solvers can
return different but equally valid solutions to the same nonconvex
problem.

### RNG: matching numpy value for value

`FR-RNG-02` requires reproducing `numpy.random.default_rng`. That is
achievable: numpy uses PCG64 with a SeedSequence-derived state, and both
are specified and implementable in Rust. Reimplementing them costs one
module in `arco-core` plus a test against captured numpy output. The
alternative, accepting a different stream, would change the exact paths
that seeded tests assert. Only nine seeded call sites exist in `tests/`
today and `test_telemetry_rng.py` asserts behavior rather than exact
streams, so the alternative is survivable. The stricter option is chosen
because it removes a whole class of "the port changed my results"
reports at a fixed one-time cost.

### ARCO is an element out of context

ISO 21448 clause 4.4.3 names the position a reusable library occupies: it
is developed against documented assumptions about its use and ships those
assumptions together with integration requirements the integrating system
must discharge. ARCO makes no safety claim and cannot: a performance level
or an integrity level attaches to a safety function realized in a
subsystem, never to a source package.

What this port owes as a result is one boundary document per algorithm,
stating assumptions of use, the inputs and their required validity, the
limits the algorithm enforces, the failure modes it returns, and the
numbered integration requirements a caller has to satisfy. That document
is also the artifact `FR-INV-08` tests against, so it replaces no existing
practice.

Two things ARCO explicitly does not own, and must never appear to: an
emergency stop, which requires removal of power and cannot be a library
method, and personnel detection, which a path planner's collision check is
not, even when both read the same sensor.

On applicable standards, the mobile and manipulator halves of ARCO differ.
ISO 10218-2:2025 excludes mobile platforms and driverless industrial
trucks in its scope clause, so the mapping, planning, guidance, and
control layers look to ISO 3691-4:2023 instead, while ISO 10218 governs
`arco-kinematics`. See the shared robotics guideline for the full mapping.

### Frames

A pose in a globally referenced frame is discontinuous by specification
and may jump at any time, while an odometry-frame pose is continuous. A
controller that differentiates the former to obtain a velocity is wrong by
construction. Ported control code names the frame each pose argument is
expected in, and the type carries it where the trait bounds allow.

### Migration shape

Every module is ported bottom up, and the Python source it replaces stays
in the tree behind a feature switch until its differential test passes.
Deletion of a Python module happens in a separate commit from the landing
of its Rust replacement, so a revert is a one-commit operation.
[PLAN.md](PLAN.md) gives the order.

## Open questions

- **Batch entry point shape (`FR-PERF-04`).** Parallel planning needs a
  public API that does not exist in the Python library today. Adding one
  is new surface, which this spec otherwise forbids. Resolve before
  phase 6: either add `arco.planning.plan_batch` as a documented addition
  or drop `FR-PERF-04` from this spec.
- ~~**Coverage tooling for the binding crate.**~~ Resolved. `cargo
  llvm-cov show-env` instruments the extension before `maturin develop`
  builds it, so a `pytest` run reports into the same profile:

  ```bash
  source <(cargo llvm-cov show-env --sh)
  cargo llvm-cov clean --workspace
  maturin develop
  pytest
  cargo llvm-cov report --lcov --output-path lcov.info
  ```

  No combined report is needed. The empty-report symptom is the missing
  `show-env` step, not a tool limitation.
- **Cancellation, speed limit, and reset.** The nav2 plugin interfaces
  carry three features ARCO lacks: a cooperative cancellation token passed
  into a long-running plan call, an externally settable speed limit that
  is not a planner parameter, and a reset distinct from teardown. All
  three are new public surface, which this spec otherwise forbids, and all
  three are needed by any caller embedding ARCO in a real robot. Decide
  before phase 9 whether they are added as documented additions or left to
  a follow-up.
- **`arco.config` palette ownership.** `palette.py` and `colors.yml`
  serve the simulator, which stays Python. Porting them may be wasted
  work. Confirm during phase 1 whether `arco-core` carries config at all
  or whether `arco.config` stays Python for now.
