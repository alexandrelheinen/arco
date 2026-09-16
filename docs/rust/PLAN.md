# Implementation plan for the ARCO Rust port

Status: reviewed

This plan implements [SPEC.md](SPEC.md). Read the spec first: this file
says in what order and against what exit criteria, never what or why.
Adaptations discovered while executing a phase go in
[DEVIATIONS.md](DEVIATIONS.md) before the phase closes, per `FR-API-06`
and `FR-DOC-03`.

## How to read a phase

Each phase is one pass of the V-cycle described in
[.guidelines/workflow/integration.md](../../.guidelines/workflow/integration.md),
and each runs all six steps at the scale the phase warrants:

```text
1. Document     spec section and requirement ids this phase satisfies
2. Architecture crate skeleton, trait signatures, test plan
3. Implement    red/green/refactor per module
4. Verify       bash scripts/validate.sh clean
5. Review       diff against the requirement ids below
6. Merge        human only
```

A phase does not start until the previous phase's exit criteria hold, with
the two exceptions marked below. Phases land as separate commits on
`feat/rust-port`, never as one squash, so `git bisect` stays useful across
a port this size.

## Ordering principle

The port runs bottom up through the dependency graph measured from the
current sources. A crate is ported only after every ARCO crate it depends
on is ported and green, so no phase ever stubs a dependency it does not
control:

```text
arco-core -> arco-mapping -> arco-planning
          -> arco-kinematics
          -> arco-control -> arco-guidance
          -> arco-runtime
          -> arco-py (last)
```

The public facade is the last thing built, not the first, because
`FR-API-01` and `FR-API-02` are only meaningful once there is something
behind them to re-export.

---

## Phase 0: build, tooling, and documentation

**Satisfies:** `FR-BUILD-01` through `FR-BUILD-05`, `FR-DOC-01`,
`FR-DOC-03`, `FR-TEST-03`.

No algorithm code lands in this phase. The phase exists so that every
later phase has a green pipeline to push into.

### Work

1. **Workspace skeleton.** Root `Cargo.toml` with
   `[workspace] members = ["crates/*"]`, resolver 2, and a
   `[workspace.package]` block holding the shared version, edition 2021,
   license `Apache-2.0`, and authorship. One empty crate per entry in the
   layering diagram of [SPEC.md](SPEC.md), each with its lint block and
   an Apache header, so the graph is enforced from the first commit.
2. **Toolchain pin.** `rust-toolchain.toml` pinning the channel and
   listing `rustfmt` and `clippy` components, so a contributor and CI
   resolve the same compiler.
3. **Lint policy.** Crate-root attributes per
   [STYLE.md](STYLE.md): `#![deny(missing_docs)]`,
   `#![deny(unsafe_op_in_unsafe_fn)]`, `#![warn(clippy::pedantic)]` with
   the documented allow list. No `rustfmt.toml`: rustfmt defaults apply,
   and [DEVIATIONS.md](DEVIATIONS.md) records why the Python 79-column
   rule does not carry over.
4. **PyO3 and maturin.** `crates/arco-py` configured as a `cdylib` named
   `arco._arco`, `pyo3` with `abi3-py310` and the `extension-module`
   feature, `numpy` for array conversion. `pyproject.toml` switches
   `build-backend` from setuptools to `maturin`, keeping
   `requires-python = ">=3.10"`, the `[project.scripts]` entry, and the
   package data rules. The Python sources under `src/arco/` stay the
   installed package; the extension lands beside them.
5. **One validation script.** `scripts/validate.sh`, which is the single
   gate [.guidelines/workflow/integration.md](../../.guidelines/workflow/integration.md)
   requires, running in this order and stopping at the first blocking
   failure: `cargo fmt --check`, `cargo clippy --all-targets -- -D
   warnings`, `cargo test --workspace`, `cargo llvm-cov` against the 80
   percent gate, `maturin develop`, `bash scripts/check_formatting.sh`,
   `bash scripts/run_tests.sh`. The existing scripts keep working and
   keep their current jobs; `validate.sh` calls them rather than
   duplicating them.
6. **CI.** Extend `.github/workflows/tests.yml` with a Rust job matching
   `validate.sh` step for step. Extend `.github/workflows/release.yml`
   to build `abi3` wheels for the three platforms of the spec's
   constraints and to smoke-install each one. The smoke workflow and the
   formatting workflow keep their current Python scope.
7. **Documentation artifacts.** [STYLE.md](STYLE.md) written and
   referenced from a new section in [docs/guidelines.md](../guidelines.md);
   [DEVIATIONS.md](DEVIATIONS.md) created with its first entries;
   [docs/decisions.md](../decisions.md) created with the decisions this
   port already made. `cargo doc --no-deps -D warnings` wired into CI.
8. **Benchmark harness.** `benches/` with Criterion, plus a recorded
   Python baseline for the RRT* scenario of `FR-PERF-01` and the MPC step
   of `FR-MPC-05`, captured before any Rust code exists. Capturing the
   baseline later would let the target drift to whatever was achieved.

### Exit criteria

- `bash scripts/validate.sh` exits 0 on an empty workspace.
- CI green on a pull request touching only this phase.
- `pip install -e .` produces a working `import arco` with the Python
  implementation untouched and the empty extension loaded.
- The Python baseline numbers for `FR-PERF-01` and `FR-MPC-05` are
  committed under `benches/baseline/`.

---

## Phase 1: `arco-core`

**Satisfies:** `FR-CORE-04`, `FR-RNG-01`, `FR-RNG-02`, `FR-API-04`
(partially), `FR-TEST-04`.

**Depends on:** phase 0.

Nothing in `arco/protocols/` (296 lines), and nothing in the geometry
helpers the other packages share, imports another ARCO package. This is
the floor of the graph.

### Work

1. **Error hierarchy.** One `ArcoError` enum rooted per
   [.guidelines/style/errors.md](../../.guidelines/style/errors.md), with
   the variants the Python code raises today, and a mapping table from
   variant to the Python exception type `FR-API-04` requires. Written
   before anything can fail, because every later crate returns it.
2. **Protocol traits.** The thirteen `Protocol` classes become traits of
   the same name. Signature only, no implementations: the implementations
   belong to the crates that own them.
3. **Geometry and linear algebra.** The shared vector, pose, and angle
   helpers currently scattered across `mapping/`, `control/`, and
   `guidance/`. Ported here so the duplication the Python tree carries
   does not get copied three times into Rust.
4. **PCG64 and SeedSequence.** The `FR-RNG-02` work. Implement numpy's
   seeding procedure and PCG64 output function, then test against vectors
   captured from `numpy.random.default_rng` and committed as fixtures.
   This is the highest-risk item of the phase and goes first inside it,
   because `FR-RNG-02` is the one requirement that fails loudly and late
   if deferred.
5. **Config loading.** `serde_yaml` over the `.yml` files in
   `src/arco/config/`, subject to the open question in [SPEC.md](SPEC.md)
   about whether the palette belongs here at all. Resolve that question
   in this phase rather than porting `palette.py` speculatively.

### Exit criteria

- `cargo test -p arco-core` green, coverage at or above 80 percent.
- PCG64 fixtures match numpy for at least three seeds and 1000 draws
  each.
- No crate outside `arco-core` compiles yet, and that is expected.

---

## Phase 2: `arco-mapping`

**Satisfies:** `FR-CORE-02`, `FR-CORE-03`, `FR-TEST-01`, `FR-TEST-02`.

**Depends on:** phase 1.

1,188 lines: `Grid` and its `ManhattanGrid` / `EuclideanGrid` subclasses,
the `Graph` to `WeightedGraph` to `CartesianGraph` to `RoadGraph`
hierarchy, and `KDTreeOccupancy`.

### Work

1. **Grid family.** `Grid` as a trait with the `position(cell_idx)`
   contract from [docs/guidelines.md](../guidelines.md) section 6, and
   the two metrics as implementing types.
2. **Graph hierarchy.** The four-level hierarchy is inheritance in
   Python and has to become composition plus traits in Rust. This is the
   first place the port cannot mirror the Python structure literally, so
   it gets a [DEVIATIONS.md](DEVIATIONS.md) entry describing the mapping
   from each Python class to its Rust form, written before the code.
3. **KD-tree occupancy.** `kiddo` replaces `scipy.spatial.KDTree`.
   Dimension checks raise `ValueError` per `FR-CORE-03`.
4. **Binding slice.** The first real work in `arco-py`: expose the
   mapping types, wire `tests/mapping/` at them, run the suite unchanged.
5. **Differential harness.** The `FR-TEST-02` machinery, built here
   because this is the first phase that has two implementations to
   compare. Reused by every later phase.

### Exit criteria

- `pytest tests/mapping/` passes unmodified against the Rust backing.
- Differential tests green on random maps and random queries.
- `scipy.spatial.KDTree` no longer imported by `src/arco/mapping/`.

---

## Phase 3: `arco-kinematics`

**Satisfies:** `FR-TEST-01`, `FR-TEST-02`.

**Depends on:** phase 1 only. **May run in parallel with phase 2.**

384 lines, the RR and RRP arm models, no ARCO dependencies beyond core
geometry. Small, self-contained, and a useful second proof of the binding
pattern before the large crates arrive.

### Exit criteria

- `pytest tests/kinematics/` passes unmodified.
- Forward and inverse kinematics agree with the Python implementation to
  1e-10 on random joint configurations.

---

## Phase 4: `arco-runtime`

**Satisfies:** `FR-TEST-01`, `FR-API-05` (telemetry hook).

**Depends on:** phase 1. **May run in parallel with phases 2 and 3.**

The typed bus (471 lines) and the pipeline runner (347 lines). Ported
before planning because the planners publish telemetry through this path,
and `FR-API-05` requires the `publisher=` hook to keep working.

### Work

1. **Typed bus.** Python's runtime type dispatch becomes generic
   publish/subscribe over a message trait.
2. **Pipeline runner.** `runner.py` imports pygame at module level today.
   The Rust runtime carries no display dependency; the pygame-dependent
   part stays in the Python layer, and the split is recorded in
   [DEVIATIONS.md](DEVIATIONS.md).
3. **Telemetry sink.** The native side of the `publisher=` enum, so the
   default JSON publisher never crosses the GIL from inside a planner
   loop.

### Exit criteria

- `pytest tests/middleware/ tests/pipeline/` passes unmodified.
- `import arco.pipeline` still works without pygame installed, or fails
  exactly as it does today, whichever the current tests assert.

---

## Phase 5: `arco-planning`

**Satisfies:** `FR-CORE-02`, `FR-PERF-01`, `FR-PERF-02`, `FR-RNG-01`,
`FR-RNG-03`, `FR-API-05`.

**Depends on:** phases 1, 2, 4.

4,053 lines and the phase that pays for the whole port. Ported in this
internal order, each step green before the next:

1. **Cost model.** `PlannerCost`, `distance`, `heuristic`, and the
   injectable cost terms. The `cost=` and `heuristic=` hook enums land
   here.
2. **A\*.** `astar.py`, discrete, grid and graph. The
   `graph.heuristic`-not-`graph.distance` default from
   [docs/guidelines.md](../guidelines.md) section 5 is a behavior the
   differential test has to pin, since it is easy to get wrong and the
   symptom is an L-shaped path rather than a crash.
3. **Route planning.** A\* over `RoadGraph`, plus `RouteResult`
   projections.
4. **Hook enums.** `Sampler`, `Steerer`, `SegmentFree`, `Pruner`,
   `Feasibility`, each with its native variants and its `Python`
   fallback, per the design note in [SPEC.md](SPEC.md). Built before
   RRT\* and SST so neither planner grows a Python-callable fast path by
   accident.
5. **RRT\*.** The `FR-PERF-01` benchmark target. Bench against the
   phase 0 baseline in the same commit that lands it.
6. **SST.**
7. **Trajectory optimizer.** `scipy.optimize.minimize` becomes `argmin`.
   This is the second-riskiest item in the plan after the MPC, because
   two optimizers converge to different local minima on the same problem.
   The differential test compares cost achieved, never the solution
   vector.

### Exit criteria

- `pytest tests/planning/` passes unmodified.
- `FR-PERF-01` met: RRT\* at least ten times faster than the phase 0
  baseline, number recorded in the commit body.
- `FR-PERF-02` verified: no GIL acquisition inside the planner loop when
  hooks are at their defaults.
- `scipy` no longer imported by `src/arco/planning/`.

---

## Phase 6: `arco-control` without MPC

**Satisfies:** `FR-TEST-01`, `FR-TEST-02`, `FR-API-05`.

**Depends on:** phases 1, 2, 5.

PID, Pure Pursuit, the tracking loop, the actuator, rigid bodies, and the
APF avoidance wired in recently. Roughly 2,400 of the package's 3,926
lines. MPC is deliberately held back to phase 7 so that a phase-7 failure
cannot block the rest of the control layer from landing.

### Work

1. **Controller trait** and the PID and Pure Pursuit implementations.
2. **Rigid bodies** (`circle`, `square`, base).
3. **Tracking loop** and `actuator.py`, including the
   `nearest_obstacle_fn` hook enum.
4. **Avoidance.** The APF term, injected rather than hardcoded, matching
   the current Python structure.

### Exit criteria

- `pytest tests/control/` passes unmodified except the MPC test modules,
  which still exercise the Python CasADi path.
- `pytest tests/guidance/test_controller.py` and the tracking-loop tests
  pass.

---

## Phase 7: MPC reformulation

**Satisfies:** `FR-MPC-01` through `FR-MPC-06`.

**Depends on:** phase 6.

The only phase that changes an algorithm rather than a language. Highest
risk in the plan, and isolated at the top of the control layer so that a
schedule slip here does not hold anything else back.

### Work

1. **Reference path.** `reference_path.py`: the CasADi B-spline
   interpolant becomes a precomputed cubic spline with analytic
   derivatives for heading and curvature. Tested against the CasADi
   interpolant on the same control points before any solver work starts.
2. **Clarabel integration.** A thin problem builder over the solver, plus
   the failure-mode mapping `FR-MPC-04` requires. Proven on a textbook
   problem with a known answer before the MPCC is pointed at it.
3. **Linearization.** The per-step Jacobians of the unicycle model and
   the first-order expansions of the contouring, lag, and heading costs,
   each with a unit test comparing the analytic Jacobian against a finite
   difference of the nonlinear expression. This is where hand-derived
   derivatives go wrong, so every one gets its own test.
4. **Constraint set.** The deadzone slack, the obstacle cones, and the
   turn-rate cone, each tested for `FR-MPC-03` against a solution known
   to sit on the boundary.
5. **SQP loop.** Warm start, iteration cap, convergence test.
6. **`joint_space.py`.** The same treatment, simpler because the joint
   model is linear in the states already.
7. **`MPCTrackingLoop`** rewired onto the new controller.
8. **Differential evaluation.** `FR-MPC-02` measured across the full
   existing MPC test set, with the tracking-error table committed to
   [DEVIATIONS.md](DEVIATIONS.md) as the record of what changed.

### Exit criteria

- `pytest tests/control/` fully green, MPC modules included.
- `FR-MPC-02` met on every scenario, table committed.
- `FR-MPC-05` met, median solve time recorded against the phase 0
  baseline.
- `casadi` removed from `pyproject.toml`, and the `mpc` optional extra
  removed or emptied.
- `docs/control_mpcc.md` updated to describe the LTV formulation. The
  document currently describes the nonlinear one and would otherwise
  become wrong.

---

## Phase 8: `arco-guidance`

**Satisfies:** `FR-TEST-01`, `FR-API-01`.

**Depends on:** phases 6 and 7.

540 lines: interpolation (B-spline, moving average), the Dubins
primitive, `ExplorationPrimitive`, and `DubinsVehicle`. Last algorithm
phase because `guidance/__init__.py` re-exports five names from
`arco.control` and cannot be finished before control is.

### Work

1. **Interpolators** behind a trait.
2. **Dubins primitive and vehicle.** `DubinsVehicle` is what the
   `TYPE_CHECKING` edge in `control/mpc/tracking_loop.py:8` pointed at;
   in Rust it satisfies a trait the control crate declares, and the
   import cycle stops existing.
3. **Facade preservation.** `arco.guidance` keeps re-exporting the five
   control names, so `FR-API-01` holds for callers who import them from
   the guidance path.

### Exit criteria

- `pytest tests/guidance/` passes unmodified, `test_public_api.py`
  included.

---

## Phase 9: the public API layer

**Satisfies:** `FR-API-01`, `FR-API-02`, `FR-API-03`, `FR-API-04`,
`FR-API-06`, `FR-DOC-02`.

**Depends on:** phases 1 through 8.

Every crate is ported. This phase makes the package look exactly like it
did, which is the acceptance test for the entire effort.

### Work

1. **Import surface audit.** Enumerate the 155 public names outside
   `arco.simulator` and assert every one resolves. `FR-API-01`.
2. **Signature parity.** A generated test that compares
   `inspect.signature` of each public callable against a snapshot taken
   from `main` before the port started. Capture that snapshot in phase 0
   and commit it, for the same reason the performance baseline is
   captured early. `FR-API-02`.
3. **Exception parity.** The `ArcoError` to Python exception mapping
   wired through `create_exception!`, with the hierarchy preserved so
   `except ValueError` still catches what it used to. `FR-API-04`.
4. **Type stubs.** A `.pyi` per ported module so `mypy --strict` and IDE
   completion keep working against the compiled extension. `FR-DOC-02`.
5. **Docstring forwarding.** Rust doc comments surfaced as `__doc__` on
   the bound objects, so `help()` keeps answering. `FR-DOC-02`.
6. **Deviation reconciliation.** Every entry accumulated in
   [DEVIATIONS.md](DEVIATIONS.md) reviewed as one set, since a deviation
   that looked local in phase 2 may read differently next to eight
   others. `FR-API-06`.

### Exit criteria

- The full `pytest tests/` suite passes, simulator tests included.
- `mypy --strict` clean on the stubs.
- Signature-parity test green against the phase 0 snapshot.

---

## Phase 10: Python source removal and test reconciliation

**Satisfies:** `FR-CORE-01`, `FR-TEST-01`, `FR-TEST-02`, `FR-PERF-04`.

**Depends on:** phase 9.

The existing tests stay. That is the point of `FR-TEST-01`, and it is the
strongest evidence the port is faithful. This phase removes the Python
implementations those tests used to exercise, one module per commit.

### Work

1. **Per-module deletion.** For each ported module, delete the Python
   source and the feature switch in a commit that touches nothing else.
   The differential test for that module goes with it, because it has
   nothing left to compare against.
2. **New tests, and only these.** `tests/rust/` holds what the spec's
   traceability table names and the existing suite cannot cover: import
   surface, signature parity, exception parity, array round-trip, batch
   parallelism, and the absence of CasADi. Nothing else is added.
3. **Coverage gate.** Confirm `FR-TEST-03` holds on the final workspace.
4. **Batch entry point.** `FR-PERF-04`, subject to the open question in
   [SPEC.md](SPEC.md). If the question resolves against adding public
   surface, drop the requirement here and record that in
   [docs/decisions.md](../decisions.md).
5. **Documentation sweep.** [docs/API.md](../API.md),
   [docs/STACK.md](../STACK.md), [CONTRIBUTING.md](../../CONTRIBUTING.md),
   and [README.md](../../README.md) updated to describe a Rust-backed
   package: build prerequisites, the policy-hook performance note of
   `FR-PERF-03`, and the new validation command.

### Exit criteria

- No Python implementation remains for any ported module.
- `src/arco/` contains the binding layer, the stubs, and
  `arco/simulator/` only.
- `bash scripts/validate.sh` green from a clean checkout.

---

## Phase 11: packaging and release

**Satisfies:** `FR-BUILD-01`, `FR-BUILD-02`.

**Depends on:** phase 10.

1. Wheel build for Linux, macOS x86_64, macOS aarch64, and Windows
   x86_64, `abi3-py310`.
2. Install smoke test per wheel on a machine with no Rust toolchain,
   which is the actual claim of `FR-BUILD-01`.
3. Source distribution that builds from a Rust toolchain, for platforms
   without a wheel.
4. Version bump and changelog entry describing the port for users:
   unchanged imports, faster planners, no CasADi, the policy-hook
   caveat, and the MPC behavior change of `FR-MPC-02`.

### Exit criteria

- `pip install arco` from the built artifacts works on all four targets
  with no toolchain present.
- Release workflow green.

---

## Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| Hand-derived MPC Jacobians are wrong | 7 | Finite-difference test per Jacobian, before the solver is wired in |
| LTV-MPC tracks worse than the nonlinear form | 7 | `FR-MPC-02` bound measured on the existing test set; phase 6 lands independently so control is not blocked |
| `argmin` converges to a different local minimum than `scipy` | 5 | Differential test compares achieved cost, never the solution vector |
| PCG64 reimplementation drifts from numpy | 1 | Fixture vectors captured from numpy, tested first thing in the phase |
| Python callbacks erase the speedup | 5 | Hook enums built before the planners that use them; `FR-PERF-02` measured |
| Graph inheritance does not map to traits cleanly | 2 | Deviation entry written before the code, so the shape is reviewed before it is built |
| Port drags on and `main` diverges | all | Rebase on `origin/main` per phase; every phase leaves the suite green, so the branch is always mergeable |
| Coverage gate unreachable through PyO3 | 0 | Open question in the spec, resolved in phase 0 before it can block phase 10 |

## What this plan does not do

It does not touch `arco.simulator`, it does not add algorithms, and it
does not improve any ported algorithm beyond what the language change and
the MPC reformulation require. A better heuristic, a smarter pruner, or a
faster collision check found along the way goes in
[docs/ROADMAP.md](../ROADMAP.md) and waits. Mixing a rewrite with
improvements makes the differential tests of `FR-TEST-02` meaningless,
and those tests are the only thing proving the port is faithful.
