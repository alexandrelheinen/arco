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
