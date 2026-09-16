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
use core::cell::Cell;

// Per thread rather than per process. A test harness runs several tests
// on several threads in one process, and a process-wide counter charges
// one test's allocations to whichever other test happened to be measuring
// at the time, which makes the assertion fail for reasons that have
// nothing to do with the code under test. Thread-local counters make the
// measurement correct under `cargo test` as well as under `cargo nextest`,
// rather than correct only under the runner that isolates processes.
//
// Both are `const`-initialized, so reading one never allocates and the
// allocator cannot recurse into itself.
thread_local! {
    /// Allocations on this thread.
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };

    /// Bytes requested on this thread.
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

/// Adds one allocation of `bytes` to this thread's counters.
///
/// Uses `try_with` because a thread-local is destroyed before the thread
/// itself is, and an allocation during that teardown would otherwise
/// panic inside the allocator.
fn record(bytes: usize) {
    let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get().saturating_add(1)));
    let _ = ALLOCATED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
}

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
// returned. The counters are `const`-initialized thread-locals holding a
// `Cell<usize>`, so recording touches no allocation state and cannot
// recurse, and the wrapper preserves every guarantee the inner allocator
// makes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
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
        record(new_size);
        // SAFETY: the caller upholds GlobalAlloc's contract for `pointer`,
        // `layout` and `new_size`, and this forwards them unchanged.
        unsafe { std::alloc::System.realloc(pointer, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: as `alloc`.
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
}

/// Allocations counted on this thread so far.
#[must_use]
pub fn allocation_count() -> usize {
    ALLOCATION_COUNT.try_with(Cell::get).unwrap_or_default()
}

/// Bytes requested on this thread so far.
#[must_use]
pub fn allocated_bytes() -> usize {
    ALLOCATED_BYTES.try_with(Cell::get).unwrap_or_default()
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
/// The counters are per thread, so a test using this is unaffected by
/// what other tests allocate alongside it. What it does not see is an
/// allocation `body` caused on another thread, which for a control step
/// is the right tradeoff: a step that spawns a thread has a latency
/// problem this measurement is not the right tool for anyway.
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
