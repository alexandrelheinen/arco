# ARCO Rust conventions

This document covers ARCO-specific Rust conventions. Generic Rust style
(edition, safety, error handling, formatting commands) lives in
[.guidelines/languages/rs.md](../../.guidelines/languages/rs.md); the
cross-language naming rules live in
[.guidelines/style/naming.md](../../.guidelines/style/naming.md); comment
and docstring section formats live in
[.guidelines/style/comments.md](../../.guidelines/style/comments.md). This
file only records where ARCO adds to or adapts those defaults, and it is
the Rust counterpart of [docs/guidelines.md](../guidelines.md).

Every adaptation below that changes an observable convention also has an
entry in [DEVIATIONS.md](DEVIATIONS.md). This file states the rule; that
file states what it replaced and why.

## 1. Naming

The ARCO rules from [docs/guidelines.md](../guidelines.md) section 1 carry
over unchanged, because they describe the domain rather than the language:

- **Maps are nouns.** `Grid`, `Occupancy`, `WeightedGraph`. Passive data.
- **Planners are agent nouns with an `-er` suffix.** `AStarPlanner`,
  `RrtPlanner`. They act on maps.
- **The who_what rule holds.** Name the quantity, not the unit:
  `max_speed`, never `speed_ms`. Qualifiers lead: `max_temperature`.
  Counts take `_count`. Collections are plural, never `_list`.
- **US English everywhere**, including doc comments and error strings.

Language-level casing follows standard Rust and needs no ARCO override:
`snake_case` for functions, methods, variables, and modules;
`PascalCase` for types and traits; `SCREAMING_SNAKE_CASE` for constants.

### Acronyms in type names

Python writes `RRTPlanner`, `SSTPlanner`, `KDTreeOccupancy`, and
`RRPArm`, keeping acronyms fully capitalized. Rust's API guidelines
capitalize only the first letter of an acronym in a `PascalCase` name,
and `clippy` flags the alternative. ARCO follows Rust here: `RrtPlanner`,
`SstPlanner`, `KdTreeOccupancy`, `RrpArm`.

The Python-facing name never changes. The binding layer registers the
class under its original name, so `from arco.planning import RRTPlanner`
keeps working:

```rust
#[pyclass(name = "RRTPlanner")]
pub struct RrtPlanner { /* ... */ }
```

### Private members

Python marks private members with a leading underscore (`_count`), since
it has no other mechanism. Rust fields are private by default, so the
underscore carries no information and is not used. A leading underscore
in Rust means one thing only, the same as it does in the shared naming
rules: an intentionally unused binding or parameter (`_event`).

### File names

A file keeps the name of the Python module it replaces, so the two trees
read against each other during the port: `astar.py` becomes `astar.rs`,
`kdtree.py` becomes `kdtree.rs`, `pure_pursuit.py` becomes
`pure_pursuit.rs`. Where that conflicts with the "file matches its
primary type" rule from the shared naming guide, the mirroring wins for
the duration of the port.

### Traits

A trait that replaces a Python `Protocol` keeps the protocol's name:
`Planner`, `Sampler`, `Steerer`, `CostTerm`. No `T` prefix, no forced
`-able` suffix. The one-to-one mapping between `arco.protocols` and
`arco_core::protocols` is worth more than a marginally more idiomatic
trait name.

## 2. Documentation

Doc comments are required on every public item, enforced by
`#![deny(missing_docs)]` at each crate root and by
`cargo doc --no-deps -D warnings` in CI.

The Google-style docstring sections ARCO uses in Python map onto rustdoc
headings:

| Python (Google style) | Rust |
|---|---|
| summary line | first line of the `///` block |
| `Args:` | `# Arguments` |
| `Returns:` | `# Returns` |
| `Raises:` | `# Errors` |
| `Raises:` for programmer error | `# Panics` |
| `Yields:` | `# Returns`, describing the iterator |
| usage example | `# Examples`, as a doctest |

A one-line summary is enough for a function whose signature already says
everything, matching the Python rule. Anything that takes a physical
quantity states its unit in the doc comment, since the who_what rule
keeps units out of the identifier:

```rust
/// Advances the vehicle state by one control step.
///
/// # Arguments
///
/// * `state` - Current pose as `[x, y, heading]`, meters and radians.
/// * `max_speed` - Upper speed bound, meters per second.
/// * `dt` - Integration step, seconds.
///
/// # Returns
///
/// The propagated pose, in the same layout as `state`.
///
/// # Errors
///
/// Returns [`ArcoError::InvalidDimension`] when `state` is not length 3.
pub fn step(
    state: &[f64],
    max_speed: f64,
    dt: f64,
) -> Result<[f64; 3], ArcoError> {
```

Doctests in `# Examples` run under `cargo test` and count as tests. Use
them for the public entry points of each crate, not for every helper.

### Doc comments that reach Python

A doc comment on a `#[pyclass]` or `#[pymethod]` becomes that object's
`__doc__`, which is what `help()` and IDE completion show. Write those
comments for a Python reader: refer to `numpy.ndarray` rather than
`PyReadonlyArray2`, and name arguments as the Python caller passes them.

### License header

Every `.rs` file carries the same Apache 2.0 header the `.py` files
carry, as a `//` block above the crate or module doc comment.

## 3. Formatting

`rustfmt` defaults, no `rustfmt.toml`. Run before every commit:

```bash
cargo fmt --all
```

ARCO's Python code uses 79 columns. That override does not carry to Rust:
rustfmt's 100-column default is what every Rust reader and every Rust
tool expects, and forcing 79 wraps generic bounds and `where` clauses
into noise. [DEVIATIONS.md](DEVIATIONS.md) records the split.

Formatting applies to all Rust code, tests included. The Python rule that
exempts `tests/` from formatting does not carry over, because `cargo fmt`
covers the whole workspace by default and excluding tests would take
deliberate configuration for no gain.

## 4. Lints

Every crate root carries:

```rust
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::pedantic)]
```

CI runs `cargo clippy --all-targets -- -D warnings`, so a `pedantic`
warning blocks a merge. An `#[allow]` needs a comment naming the reason,
the same rule `# type: ignore` follows in Python.

`unsafe` is not expected anywhere in the ported crates. If a block turns
out to be necessary, it needs a `// SAFETY:` comment stating the
invariant and a test exercising it, per the shared Rust guideline.

## 5. Errors

One error enum per crate, all convertible into `arco_core::ArcoError`,
which is the type the binding layer translates into Python exceptions.
Return `Result` for anything a caller could handle; reserve `panic!` for
a broken precondition that means the calling code has a bug.

The Python exception a given variant maps to is part of the public API
under `FR-API-04`, so the mapping table lives next to the enum and
changes to it are breaking changes.

## 6. Tests

`cargo test` covers the Rust side; `pytest` covers the Python-facing
behavior and stays the contract, per `FR-TEST-01`.

- **Unit tests** for private behavior go in an inline
  `#[cfg(test)] mod tests` at the bottom of the file they test. This is
  Rust's idiom and the only way to reach private items.
- **Integration tests** for public behavior go in `crates/<crate>/tests/`,
  mirroring the module they exercise, which matches the mirrored layout
  ARCO already uses in `tests/`.

The Python rule that tests mirror the source tree therefore holds at the
integration level and is relaxed at the unit level, where Rust's
visibility rules leave no alternative.

### Test naming

The spec-sentence rule from
[.guidelines/style/naming.md](../../.guidelines/style/naming.md) holds,
without the `test_` prefix: `#[test]` already marks the function, and
repeating it in the name adds nothing.

```rust
#[test]
fn plans_around_a_blocking_obstacle() { /* ... */ }

#[test]
fn rejects_a_goal_outside_the_grid() { /* ... */ }
```

Never name a test after its inputs.

### Unimplemented work

Per [.guidelines/workflow/tdd.md](../../.guidelines/workflow/tdd.md), a
stub must fail loudly. Rust has no `NotImplementedError`, so the Python
pairing of `raise NotImplementedError` with
`@pytest.mark.xfail(strict=True)` becomes:

```rust
pub fn plan(&self) -> Result<Path, ArcoError> {
    todo!("D* Lite: see docs/ROADMAP.md")
}
```

paired with:

```rust
#[test]
#[should_panic(expected = "D* Lite")]
fn plan_is_not_implemented_yet() { /* ... */ }
```

The `expected` string keeps the marker honest the way `strict=True` does:
it names what is blocking the work, and it has to be removed in the same
commit that lands the implementation.

## 7. PyO3 conventions

- **Signatures are explicit.** Every `#[pyfunction]` and `#[new]` that
  takes optional or keyword arguments carries a `#[pyo3(signature = ...)]`
  reproducing the Python defaults exactly. `FR-API-02` is checked by a
  generated test, but the signature attribute is what makes it pass.
- **Arrays in do not copy.** Accept `PyReadonlyArray1`/`PyReadonlyArray2`
  and read through them. Arrays out may copy.
- **The GIL is released around any loop.** Wrap planner and controller
  bodies in `py.allow_threads`, which is what `FR-PERF-02` and
  `FR-PERF-04` require. A loop that calls back into Python cannot do
  this, which is the whole reason the policy hooks are enums.
- **Policy hooks are enums, never bare `Py<PyAny>`.** The built-in
  variants dispatch natively; the `Python` variant exists for
  caller-supplied callables and is documented as slow.
- **Python-facing names are set explicitly.** `#[pyclass(name = "...")]`
  and `#[pyo3(name = "...")]` wherever the Rust name differs, which is
  every acronym type.

## 8. Commands

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo doc --no-deps
maturin develop
```

One script runs all of it, and CI runs the same script:

```bash
bash scripts/validate.sh
```
