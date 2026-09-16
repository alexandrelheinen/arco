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
- `FR-TEST-03`: When the Rust test suite runs, line coverage on the ported
  crates shall stay at or above 80 percent, enforced in CI, matching the
  gate in [.guidelines/workflow/tdd.md](../../.guidelines/workflow/tdd.md).
- `FR-TEST-04`: When a Rust function is left unimplemented on purpose, it
  shall call `todo!()` with a reason string, and its test shall carry
  `#[should_panic(expected = ...)]` naming the requirement that blocks it.

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
- **Coverage tooling for the binding crate.** `cargo llvm-cov` does not
  see code reached only through PyO3. Either the crates carry enough
  native tests to hit the 80 percent gate without the Python path, or CI
  needs a combined report. Decide during phase 0.
- **`arco.config` palette ownership.** `palette.py` and `colors.yml`
  serve the simulator, which stays Python. Porting them may be wasted
  work. Confirm during phase 1 whether `arco-core` carries config at all
  or whether `arco.config` stays Python for now.
