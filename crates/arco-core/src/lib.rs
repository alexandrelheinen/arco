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

#![forbid(unsafe_code)]
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

//! Shared foundations for every other ARCO crate.
//!
//! Holds the traits that replace `arco.protocols`, the error enum every
//! crate converts into, the geometry and angle helpers the rest of the
//! workspace shares, the named tolerance constants of `FR-SAFE-06`, the
//! total-order comparison helpers of `FR-SAFE-05`, and the numpy-compatible
//! random number generator of `FR-RNG-02`.
//!
//! Nothing here depends on another ARCO crate, which is what makes it the
//! floor of the dependency graph.
//!
//! Criticality: C2. See `docs/rust/STYLE.md`.

mod error;
pub mod numeric;
pub mod rng;

pub use error::{Error, PythonException};
