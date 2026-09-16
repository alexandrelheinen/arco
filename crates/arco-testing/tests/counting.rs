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

//! The allocation counter reports what it claims to report.
//!
//! FR-SAFE-04 rests on this, so the instrument is checked before anything
//! is measured with it: a harness that under-reports would let an
//! allocating control step pass unnoticed.

use arco_testing::{CountingAllocator, assert_no_allocations, measure_allocations};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator::new();

#[test]
fn arithmetic_on_the_stack_allocates_nothing() {
    let sum = assert_no_allocations(|| (0..1000_u64).sum::<u64>());
    assert_eq!(sum, 499_500);
}

#[test]
fn writing_into_a_preallocated_buffer_allocates_nothing() {
    // The shape every ported control step has to take: reserve at
    // construction, write during the step.
    let mut buffer = Vec::with_capacity(256);
    assert_no_allocations(|| {
        for value in 0..256_u32 {
            buffer.push(value);
        }
    });
    assert_eq!(buffer.len(), 256);
}

#[test]
fn a_growing_vector_is_caught() {
    let (_, report) = measure_allocations(|| {
        let mut growing = Vec::new();
        for value in 0..1024_u32 {
            growing.push(value);
        }
        growing
    });
    assert!(report.count > 0, "the counter missed a growing vector");
    assert!(report.bytes > 0, "the counter missed the bytes");
}

#[test]
#[should_panic(expected = "expected no allocation")]
fn the_assertion_fails_when_the_body_allocates() {
    assert_no_allocations(|| vec![0_u8; 64]);
}

#[test]
fn measuring_nests() {
    let ((), outer) = measure_allocations(|| {
        let (_, inner) = measure_allocations(|| vec![0_u8; 32]);
        assert!(inner.count > 0);
    });
    assert!(outer.count >= 1, "{outer:?}");
}
