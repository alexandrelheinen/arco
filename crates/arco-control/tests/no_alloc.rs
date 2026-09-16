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

//! FR-SAFE-04: a control step allocates nothing.
//!
//! An allocation on a control path has an unbounded latency tail: the
//! allocator may search a free list, split a block, or ask the operating
//! system for more memory, and none of those has a bound the caller can
//! state. The lint tiers cannot see this, so an instrumented allocator
//! proves it instead.
//!
//! The counter is process-wide, which is why the workspace runs
//! `cargo nextest`: it gives each test its own process, so one test's
//! allocations cannot be charged to another.

mod common;

use arco_control::body::{CircleBody, RigidBody, SquareBody};
use arco_control::limits::{CommandConditioner, CommandLimits, IntervalBand};
use arco_control::pid::{PidController, PidSettings};
use arco_control::pursuit::PurePursuitTracker;
use arco_control::tracking::{TrackingLoop, TrackingSettings};
use arco_core::protocols::{Command, PathTracker, VehicleModel};
use arco_testing::{CountingAllocator, assert_no_allocations};

use common::{NoAvoidance, Unicycle, straight_path};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator::new();

fn bounded_limits() -> CommandLimits {
    CommandLimits {
        max_speed: 2.0,
        min_speed: 0.0,
        max_turn_rate: 1.5,
        max_speed_rate: 5.0,
        max_turn_rate_change: 5.0,
        interval: IntervalBand::default(),
    }
}

#[test]
fn a_pid_step_allocates_nothing() {
    let mut controller = PidController::new(PidSettings::default()).expect("defaults are valid");
    // Warm up outside the measurement: the first step takes the
    // uninitialized-derivative branch, and measuring a branch the loop
    // takes once tells you nothing about the loop.
    controller.step(0.0, 1.0, 0.01).expect("valid step");

    assert_no_allocations(|| {
        for _ in 0..1000 {
            controller.step(0.0, 1.0, 0.01).expect("valid step");
        }
    });
}

#[test]
fn a_command_conditioning_step_allocates_nothing() {
    let mut conditioner = CommandConditioner::new(bounded_limits()).expect("valid limits");
    conditioner
        .apply(
            Command {
                speed: 1.0,
                turn_rate: 0.0,
            },
            0.01,
        )
        .expect("valid step");

    assert_no_allocations(|| {
        for step in 0..1000_i32 {
            let requested = Command {
                speed: if step % 2 == 0 { 10.0 } else { -10.0 },
                turn_rate: if step % 2 == 0 { -5.0 } else { 5.0 },
            };
            conditioner.apply(requested, 0.01).expect("valid step");
        }
    });
}

#[test]
fn a_body_integration_step_allocates_nothing() {
    let mut circle = CircleBody::new(2.0, 0.4, 0.0, 0.0, 0.0).expect("a valid disk");
    let mut square = SquareBody::new(2.0, 0.6, 0.0, 0.0, 0.0).expect("a valid square");

    assert_no_allocations(|| {
        for _ in 0..1000 {
            circle
                .apply_wrench(1.0, -0.5, 0.2)
                .expect("a finite wrench");
            circle.step(0.001).expect("valid step");
            square
                .apply_wrench(-1.0, 0.5, -0.2)
                .expect("a finite wrench");
            square.step(0.001).expect("valid step");
            // The corners are a fixed-size array rather than a vector,
            // which is the only reason this stays inside the assertion.
            let corners = square.corners();
            assert_eq!(corners.len(), 4);
        }
    });
}

#[test]
fn a_pure_pursuit_step_allocates_nothing() {
    let path = straight_path(64);
    let mut tracker = PurePursuitTracker::new(1.5).expect("a valid lookahead");
    let pose = Unicycle::at(0.3, 0.2, 0.1).pose();
    tracker.track(pose, &path, 1.0).expect("valid step");

    assert_no_allocations(|| {
        for _ in 0..500 {
            tracker.track(pose, &path, 1.0).expect("valid step");
        }
    });
}

#[test]
fn a_tracking_loop_step_allocates_nothing_once_its_history_is_bounded() {
    // The history is the one allocating part of a step, and the default
    // keeps every sample because the Python loop did. A caller with a
    // real-time budget sets a capacity, and this is what that buys.
    let path = straight_path(32);
    let mut driving = TrackingLoop::new(
        Unicycle::at(0.0, 0.1, 0.0),
        PurePursuitTracker::new(1.0).expect("a valid lookahead"),
        NoAvoidance,
        TrackingSettings {
            cruise_speed: 1.0,
            curvature_gain: 0.5,
            limits: bounded_limits(),
            history_capacity: Some(16),
        },
    )
    .expect("consistent settings");

    // Fill the ring first: growing it is an allocation, reusing it is not.
    driving.run(&path, 32, 0.01).expect("valid steps");

    assert_no_allocations(|| {
        for _ in 0..500 {
            driving.step(&path, 0.01).expect("valid step");
        }
    });
}

#[test]
fn keeping_no_history_at_all_also_allocates_nothing() {
    let path = straight_path(32);
    let mut driving = TrackingLoop::new(
        Unicycle::at(0.0, 0.1, 0.0),
        PurePursuitTracker::new(1.0).expect("a valid lookahead"),
        NoAvoidance,
        TrackingSettings {
            history_capacity: Some(0),
            limits: bounded_limits(),
            ..TrackingSettings::default()
        },
    )
    .expect("consistent settings");
    driving.step(&path, 0.01).expect("valid step");

    assert_no_allocations(|| {
        for _ in 0..500 {
            driving.step(&path, 0.01).expect("valid step");
        }
    });
    assert_eq!(driving.history().count(), 0);
}
