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

#![deny(unsafe_op_in_unsafe_fn)]

//! Test-only instrumentation shared across the ARCO workspace.
//!
//! This crate exists so that the algorithm crates can keep
//! `#![forbid(unsafe_code)]`, which cannot be relaxed per module. Counting
//! allocations requires implementing [`core::alloc::GlobalAlloc`], and
//! that trait is unsafe to implement, so the one place in the workspace
//! that needs `unsafe` is here rather than inside a crate whose freedom
//! from it is a property worth stating.
//!
//! It is never published and never a runtime dependency.
//!
//! # Examples
//!
//! A test binary installs the allocator once, then asserts against it:
//!
//! ```ignore
//! use arco_testing::CountingAllocator;
//!
//! #[global_allocator]
//! static ALLOCATOR: CountingAllocator = CountingAllocator::new();
//!
//! #[test]
//! fn a_control_step_allocates_nothing() {
//!     let mut controller = build_controller();
//!     arco_testing::assert_no_allocations(|| controller.step(pose, dt));
//! }
//! ```

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

/// Total allocations since the process started.
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Total bytes requested since the process started.
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// A global allocator that counts what passes through it.
///
/// Wraps the system allocator and adds a counter. Install it in a test
/// binary with `#[global_allocator]`; it has no place in a shipped one.
#[derive(Debug, Default)]
pub struct CountingAllocator;

impl CountingAllocator {
    /// Builds the allocator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

// SAFETY: every method forwards to the system allocator with the layout it
// was given, unchanged, and returns exactly what the system allocator
// returned. The counters are atomic and touch no allocation state, so the
// wrapper preserves every guarantee the inner allocator makes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller upholds GlobalAlloc's contract for `layout`,
        // and this forwards it unchanged.
        unsafe { std::alloc::System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller guarantees `pointer` came from this allocator
        // with this `layout`, and this forwards both unchanged.
        unsafe { std::alloc::System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        // SAFETY: the caller upholds GlobalAlloc's contract for `pointer`,
        // `layout` and `new_size`, and this forwards them unchanged.
        unsafe { std::alloc::System.realloc(pointer, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: as `alloc`.
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
}

/// Allocations counted so far.
#[must_use]
pub fn allocation_count() -> usize {
    ALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Bytes requested so far.
#[must_use]
pub fn allocated_bytes() -> usize {
    ALLOCATED_BYTES.load(Ordering::Relaxed)
}

/// What a measured region allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationReport {
    /// Number of allocations, including reallocations.
    pub count: usize,
    /// Bytes requested.
    pub bytes: usize,
}

/// Runs `body`, reporting what it allocated.
///
/// The count is process-wide, so a test using this must not run
/// concurrently with another that allocates. `cargo nextest` gives each
/// test its own process, which is why the workspace uses it.
pub fn measure_allocations<T>(body: impl FnOnce() -> T) -> (T, AllocationReport) {
    let count_before = allocation_count();
    let bytes_before = allocated_bytes();
    let value = body();
    (
        value,
        AllocationReport {
            count: allocation_count().saturating_sub(count_before),
            bytes: allocated_bytes().saturating_sub(bytes_before),
        },
    )
}

/// Runs `body` and panics if it allocated.
///
/// `FR-SAFE-04`. A control step that allocates has an unbounded latency
/// tail, because an allocator may search, split, or ask the operating
/// system for more memory.
///
/// # Panics
///
/// Panics when `body` performs any allocation, naming the count and the
/// bytes so the offending call is findable.
pub fn assert_no_allocations<T>(body: impl FnOnce() -> T) -> T {
    let (value, report) = measure_allocations(body);
    assert!(
        report.count == 0,
        "expected no allocation, got {} allocation(s) totalling {} bytes",
        report.count,
        report.bytes
    );
    value
}
