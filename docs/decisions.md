# Decision log

Append-only record of decisions that shape ARCO and would otherwise be
unanswerable from the diff alone: why a constraint exists, why a gate was
loosened, why an approach was chosen over the obvious alternative.
Required by [.guidelines/workflow/sdd.md](../.guidelines/workflow/sdd.md).

Newest last. Never edit a decision in place: supersede it with a new one
and mark the old entry superseded.

---

## ADR-001: port the algorithm core to Rust, keep the Python package

**Date:** 2026-09-16. **Status:** accepted.

ARCO's planners and controllers run interpreted loops, and the GIL blocks
the parallelism the algorithms naturally have. The core moves to Rust.

The package stays `pip install arco`, with PyO3 bindings that keep every
import path and call signature. The binding layer is permanent, not a
migration aid: dropping it would strand every current user for a gain the
port does not need.

Alternatives considered. A pure Rust library with no Python surface was
rejected because it breaks every caller. A partial port of only the
hottest loops was rejected because the boundary crossings it needs would
eat most of the gain, and because it leaves two implementations of the
same algorithms alive indefinitely.

Spec: [docs/rust/SPEC.md](rust/SPEC.md).

## ADR-002: replace the CasADi nonlinear MPC with LTV-MPC solved by Clarabel

**Date:** 2026-09-16. **Status:** accepted.

CasADi has no Rust equivalent, and the path-following controller depends
on its symbolic modeling, its automatic derivatives, and IPOPT.

Three options were on the table.

Keeping CasADi through generated C was the cheapest: `opti.to_function()
.generate()` emits C that Rust can link. It keeps the nonlinear
formulation exactly and needs no derivative work. It was rejected because
it puts a C toolchain and a generated-code build step between a
contributor and a working checkout, and because the symbolic model would
stay in Python, which contradicts ADR-001.

Keeping MPC in Python behind the binding was rejected for the same
reason, plus it leaves `casadi` as a dependency forever.

Reformulating as linear time-varying MPC solved by sequential quadratic
programming was chosen. Clarabel is pure Rust, needs no C dependency, and
supports second-order cones, so the obstacle and turn-rate constraints
keep their geometry rather than being flattened into half-spaces. The
cost is hand-derived Jacobians, which is real work and is why every one
of them gets a finite-difference test in phase 7.

The consequence is that MPC output changes. `FR-MPC-02` bounds the change
at 20 percent of root-mean-square lateral error rather than requiring
agreement, because two solvers can return different valid solutions to
the same nonconvex problem.

## ADR-003: reimplement numpy's PCG64 rather than accept a new random stream

**Date:** 2026-09-16. **Status:** accepted.

Sampling planners seeded with the same value should keep producing the
same paths after the port. A native Rust RNG would produce a different
stream, changing every seeded result.

Only nine seeded call sites exist in `tests/` today, and
`test_telemetry_rng.py` asserts behavior rather than exact draws, so
accepting a new stream would have been survivable. It was rejected
anyway: numpy's SeedSequence and PCG64 are both specified and cost one
module plus a fixture test to reproduce, and paying that once removes a
whole class of "the port changed my results" reports from users whose
tests ARCO cannot see.

## ADR-004: policy hooks become enums with a Python fallback

**Date:** 2026-09-16. **Status:** accepted.

`sampler=`, `steerer=`, `segment_free=`, `cost=`, `cost_terms=`,
`heuristic=`, `feasibility=`, `publisher=`, and `nearest_obstacle_fn=`
are all called from inside planner and controller loops. Binding them as
plain Python callables would reacquire the GIL millions of times per
`plan()` and erase the entire speedup.

Each hook becomes a Rust enum whose variants cover the built-in policies
natively, with one `Python` variant for caller-supplied callables. The
built-in defaults never cross the boundary; an injected callable still
works and is documented as slow.

The alternative, refusing Python callables outright, would have been
faster to build and would break the extension points
[docs/guidelines.md](guidelines.md) section 5 promises.

## ADR-005: the existing pytest suite is the port's acceptance test

**Date:** 2026-09-16. **Status:** accepted.

The Rust implementation is accepted when the current `tests/` suite
passes against it unmodified. Tests are not rewritten to match the new
implementation, which is exactly the failure mode
[.guidelines/workflow/tdd.md](../.guidelines/workflow/tdd.md) warns
about: a test written after the code describes what the code does rather
than what it should do.

New tests are added only for properties the existing suite cannot express
(import surface, signature parity, exception parity, array round-trip),
and each is named in the spec's traceability table.

## ADR-006: Rust code uses rustfmt defaults, not ARCO's 79 columns

**Date:** 2026-09-16. **Status:** accepted.

ARCO formats Python at 79 columns. Rust uses `rustfmt` defaults at 100.
Forcing 79 on Rust wraps generic bounds and `where` clauses into noise,
and every Rust tool and reader expects the default. Recorded as C-02 in
[docs/rust/DEVIATIONS.md](rust/DEVIATIONS.md).

## ADR-007: gate coverage on lines, regions and functions, not branches

**Date:** 2026-09-16. **Status:** accepted. **Supersedes part of ADR-005.**

ISO 26262-6 table 9 downgrades statement coverage at the higher assurance
levels and makes branch coverage highly recommended from ASIL B, which
argues for gating on branches rather than lines.

Rust cannot do it. `-Zcoverage-options=branch` needs nightly, its tracking
issue is open, and `cargo-llvm-cov` has no `--fail-under-branches`: its own
CI carries a `# TODO: --fail-under-branches?` comment. The available gates
are lines, regions, functions, and per-file lines. Region coverage is the
closest proxy to branch coverage that can actually fail a build, so
`FR-TEST-03` gates on lines, regions and functions together.

Modified condition or decision coverage is not an option at all. It was
removed from the compiler in 2025 over maintenance cost, and
`cargo-llvm-cov --mcdc` now fails on any toolchain from 1.91 onward.
Reintroduction is an accepted 2026 Rust project goal with AdaCore
committing maintenance; track it, do not plan around it.

The Python side keeps branch coverage, since `coverage.py` supports it.
The gate is therefore asymmetric by language, deliberately.

## ADR-008: assign criticality per crate, not per project

**Date:** 2026-09-16. **Status:** accepted.

Holding a plotting helper and a control loop to one standard means either
the helper drowns in process or the control loop is under-protected.
Criticality is assigned per crate, in
[docs/rust/STYLE.md](rust/STYLE.md#1-criticality-per-crate), following
[.guidelines/workflow/criticality.md](../.guidelines/workflow/criticality.md).

A crate inherits the highest level of anything depending on it, which is
why `arco-core` is C2: its geometry helpers are reached from the
controllers. `arco-planning` is C1 rather than C2 because a plan is
consumed by a control layer that validates it before acting, and that
layer is where an unsafe path has to be caught. If a planner output ever
reaches an actuator without that validation, the level rises and this
entry is superseded.

## ADR-009: a panic must never cross the Python boundary

**Date:** 2026-09-16. **Status:** accepted.

Unwinding out of a foreign function is undefined behavior. PyO3 catches
what it can and raises an exception derived from `BaseException`, which an
ordinary `except Exception` handler passes over and which tends to end the
interpreter.

So a panic in an ARCO planner does not reach a Python caller as a
catchable error; it takes the process down. That is a stronger argument
against panicking constructs than a pure Rust library has, and it is why
`arco-py` is C2 despite containing no algorithm: every exported function
returns a `PyResult`, and a link job runs `#[no_panic]` against the entry
points in an unwind profile.

The alternative, relying on PyO3's own catch, was rejected because it
converts a bug into a process kill rather than into an error the caller
can handle.

## ADR-010: defer to the shared guidelines rather than restating them

**Date:** 2026-09-16. **Status:** accepted.

`docs/rust/STYLE.md` originally carried a full Rust style guide, written
when `.guidelines/languages/rs.md` was a stub that deferred most decisions
upstream. The shared file now covers the toolchain, workspace layout, lint
tiers, unsafe policy, error enums, rustdoc sections, test layout, and
extension-module rules, and two new shared files cover defensive
programming and criticality.

`STYLE.md` is now ARCO deltas only: criticality per crate, the domain
naming that describes robotics rather than a language, the acronym mapping
table, the file-name mirroring during the port, the license header, the
tolerance constants, and the budget parameters. Anything true of Rust in
general belongs upstream, where every project in the family gets it.

The submodule pin moved from v1.0.0 to the guidelines `main` tip in the
same change, because the old pin made every reference in these documents
point at the stub rather than the guideline.

## ADR-011: ISO 3691-4 governs the mobile half, ISO 10218 the manipulator half

**Date:** 2026-09-16. **Status:** accepted.

ISO 10218 is the standard people reach for when a project says "robot",
and for ARCO it is mostly the wrong one. ISO 10218-2:2025 excludes mobile
platforms, driverless industrial trucks, and tele-operated manipulators in
its own scope clause, so `arco-mapping`, `arco-planning`,
`arco-guidance` and `arco-control` look to ISO 3691-4:2023 in Europe and
ANSI/A3 R15.08 in the US. ISO 10218 governs `arco-kinematics`.

A related correction: ISO/TS 15066 is no longer where
collaborative-operation requirements live. The 2025 edition of ISO 10218
absorbed them as normative content and retired the terms "collaborative
robot" and "collaborative operation".

Nothing here obliges ARCO to comply with any of them. The point of
recording it is that an integrator asking "which standard did you build
against" gets a straight answer, and that nobody spends a week reading the
wrong document.

## ADR-012: ARCO makes no safety claim, and ships assumptions instead

**Date:** 2026-09-16. **Status:** accepted.

ISO 21448 clause 4.4.3 names the position a reusable library occupies: an
element out of context, developed against documented assumptions and
shipped with integration requirements the integrating system discharges.

A performance level or a safety integrity level attaches to a safety
function realized in a subsystem, including its hardware architecture,
diagnostic coverage, and common-cause analysis. No amount of test coverage
on a crate produces one. ARCO therefore claims none, and phase 9 produces
one boundary document per algorithm instead.

Two capabilities ARCO must never appear to offer. An **emergency stop**
requires removal of power and cannot be a library method; anything named
`stop` in this API is a normal stop, and naming it otherwise would be a
naming error with safety consequences. **Personnel detection** is not what
a planner's collision check does, even when both read the same sensor.

## ADR-013: invariants are the specification, the Python behavior is not

**Date:** 2026-09-16. **Status:** accepted. **Qualifies ADR-005.**

ADR-005 makes the existing pytest suite the port's acceptance test, which
is right for behavior and wrong for correctness properties the Python code
never asserted.

`FR-INV-01` through `FR-INV-15` are stated as properties of the algorithms
rather than as descriptions of the current implementation. Where one turns
out to be false of the Python behavior, the resolution is a bug report
against the Python implementation and a DEVIATIONS entry, never a
weakened invariant. An invariant relaxed to make a port pass has stopped
being a specification.

Most are metamorphic relations, chosen because every algorithm here has
the oracle problem: the optimal path is unknown, while how the output must
change under a known input transformation is not. `FR-INV-06` is the
exception and the most valuable single test in the plan, since an
admissible heuristic means A* must return exactly the cost an uninformed
search returns on the same graph, giving a differential oracle that needs
no reference implementation.

## ADR-014: pin the toolchain exactly, and let the dependency gate block a merge

**Date:** 2026-09-16. **Status:** accepted.

`rust-toolchain.toml` pins `1.98.1` rather than the `1.98` the shared
guideline shows, because the point of the pin is that a contributor and CI
resolve the same compiler and a two-component channel still floats across
patch releases.

`cargo deny check` is a blocking gate from the first commit, not something
added once the workspace has code in it. It earned that position
immediately: the skeleton pulled `pyo3` 0.27.2, which carries an advisory
for a thread-safety defect closed in 0.29.0, and the gate caught it before
a single algorithm existed. Finding it later would have meant auditing
whatever binding code had been written against the affected API.

The wider point is the one worth keeping. Phase 0 exists so that every
later phase pushes into a pipeline that already works, and a gate that
only starts running once there is something to check is a gate that
arrives after the decisions it was meant to inform.

## ADR-015: configuration is passed explicitly, never read at import time

**Date:** 2026-09-16. **Status:** accepted.

`load_config` resolves its directory from the `ARCO_CONFIG_DIR`
environment variable, and several modules call it at import time into a
module-level global, `src/arco/config/palette.py` being the clearest
case. Once any of those modules is imported, changing the environment
variable has no further effect, because the value is already bound.

Phase 0 hit this from the other direction. A snapshot test that imports
every package to read its signatures left the config globals populated,
and a simulator test running afterwards in the same interpreter read a
configuration it had not selected, failing with a `KeyError` on a key that
belongs to a different file. The snapshot test now captures in its own
interpreter, which is correct for a snapshot test regardless, but the
underlying fragility is a property of the Python design rather than of the
test.

The ported crates do not reproduce it. Configuration is parsed once at
construction, validated there, and held by the object that uses it, so a
caller that wants different configuration builds a different object. No
crate reads an environment variable at load time, and nothing holds
configuration in a global.

This also removes a class of ordering dependency that
[docs/guidelines.md](guidelines.md) section 8 currently handles with a
manual consumer audit whenever a shared configuration file is
restructured. When configuration is an argument, the compiler finds the
consumers.
