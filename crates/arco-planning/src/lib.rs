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

//! Discrete and sampling-based motion planning.
//!
//! Replaces `arco.planning`. Carries A*, route planning, RRT*, SST and the
//! trajectory optimizer, together with the policy-hook enums of ADR-004
//! and the planner invariants `FR-INV-01` through `FR-INV-08`.
//!
//! Criticality: C1. See `docs/rust/STYLE.md`.
