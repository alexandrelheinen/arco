// Copyright 2026 alexandre
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Hardened tier: this crate is C2 in docs/rust/STYLE.md, so a panic or a
// silent wrap here reaches a machine. The baseline tier arrives through
// the workspace `[lints]` table; Cargo rejects a crate-level
// `[lints.clippy]` alongside `workspace = true`, so the hardened tier is
// expressed here instead.
#![deny(clippy::arithmetic_side_effects)]
#![deny(clippy::as_conversions)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_division)]
#![deny(clippy::modulo_arithmetic)]
#![deny(clippy::cast_possible_truncation)]
#![deny(clippy::cast_sign_loss)]
#![deny(clippy::cast_precision_loss)]
#![deny(clippy::large_stack_arrays)]
#![deny(clippy::string_slice)]
#![deny(clippy::exit)]

//! `PyO3` bindings for ARCO.
//!
//! This crate exists so that `import arco` keeps working unchanged. Every
//! public name listed in `docs/API.md` resolves from the same module path
//! it always did, with the same argument names, positional order and
//! default values, per `FR-API-01` and `FR-API-02`.
//!
//! Three rules govern everything added here.
//!
//! A panic must never escape into Python. Unwinding out of a foreign
//! function is undefined behavior, and `PyO3` converts what it catches into
//! an exception derived from `BaseException`, which an ordinary
//! `except Exception` handler passes over. Every exported function
//! returns a [`pyo3::PyResult`], per `FR-SAFE-01` and ADR-009.
//!
//! The interpreter lock is released around anything that computes, and an
//! injected Python callable becomes one variant of an enum rather than the
//! whole hook, so the built-in policies never cross the boundary inside a
//! loop. See ADR-004 and `FR-PERF-02`.
//!
//! A Rust type whose name differs from its Python spelling registers the
//! original, so `RrtPlanner` appears to Python as `RRTPlanner`. The mapping
//! is in `docs/rust/STYLE.md`.
//!
//! Criticality: C2. See `docs/rust/STYLE.md`.

use pyo3::prelude::*;

mod config;
mod control;
mod core;
mod errors;
mod guidance;
mod hooks;
mod kinematics;
mod mapping;
mod mpc;
mod pipeline;
mod planning;
mod runtime;
mod telemetry;

/// Version of the compiled extension, for diagnostics and parity tests.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Entry point registering the compiled module as `arco._arco`.
///
/// # Errors
///
/// Returns an error when a registration fails, which the interpreter
/// surfaces as an `ImportError`.
#[pymodule]
fn _arco(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    core::register(module)?;
    control::register(module)?;
    mpc::register(module)?;
    mapping::register(module)?;
    kinematics::register(module)?;
    guidance::register(module)?;
    planning::register(module)?;
    pipeline::register(module)?;
    telemetry::register(module)?;
    runtime::register(module)?;
    Ok(())
}
