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

//! Which states a car-like robot could actually be in.
//!
//! The constraint is a ratio, so the tests are mostly about invariance:
//! scaling a speed and a turn rate together cannot change the verdict, and
//! neither can turning the other way.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_core::Error;
use arco_guidance::primitive::{DubinsPrimitive, ExplorationPrimitive};

/// A primitive with the given minimum turning radius, in meters.
fn primitive(turning_radius: f64) -> DubinsPrimitive {
    DubinsPrimitive::new(turning_radius).expect("a valid turning radius")
}

/// A five-element state `(x, y, heading, speed, turn rate)`.
const fn moving(speed: f64, turn_rate: f64) -> [f64; 5] {
    [0.0, 0.0, 0.0, speed, turn_rate]
}

// --------------------------------------------------------- steering ----

#[test]
fn a_segment_starts_where_it_was_asked_and_ends_where_it_was_sent() {
    // The segment is the two endpoints, matching the Python placeholder.
    // What the contract promises either way is that a caller stitching
    // segments together does not have to guess about the ends.
    let segment = primitive(2.0)
        .steer(&[0.0, 0.0, 0.0], &[1.0, 1.0, 0.0])
        .expect("a valid query");
    assert_eq!(
        segment.first().map(Vec::as_slice),
        Some([0.0, 0.0, 0.0].as_slice())
    );
    assert_eq!(
        segment.last().map(Vec::as_slice),
        Some([1.0, 1.0, 0.0].as_slice())
    );
}

#[test]
fn states_that_do_not_describe_the_same_space_are_refused() {
    let primitive = primitive(1.0);
    assert!(matches!(
        primitive.steer(&[0.0, 0.0, 0.0], &[1.0, 1.0]),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        primitive.steer(&[0.0], &[1.0, 1.0]),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        primitive.steer(&[0.0, 0.0], &[f64::NAN, 1.0]),
        Err(Error::NotFinite { .. })
    ));
}

// ------------------------------------------------------- feasibility ----

#[test]
fn a_state_carrying_no_curvature_is_feasible() {
    // Position, heading and speed say nothing about how tightly the
    // vehicle is turning, so there is nothing to compare against the
    // radius and nothing to refuse.
    let primitive = primitive(1.0);
    for state in [
        vec![3.0, 4.0],
        vec![0.0, 0.0, 1.57],
        vec![0.0, 0.0, 0.0, 5.0],
    ] {
        assert!(
            primitive.is_feasible(&state).expect("a valid state"),
            "{state:?}"
        );
    }
}

#[test]
fn straight_line_motion_is_always_feasible() {
    // A turn rate of zero is a circle of infinite radius, which clears any
    // floor a caller could state.
    let primitive = primitive(100.0);
    assert!(
        primitive
            .is_feasible(&moving(3.0, 0.0))
            .expect("a valid state")
    );
}

#[test]
fn a_radius_at_the_floor_is_feasible_and_one_below_it_is_not() {
    // Four meters per second through two radians per second is a two meter
    // circle, which is exactly the floor and therefore allowed: a bound
    // the vehicle cannot reach is not the bound it was given.
    let primitive = primitive(2.0);
    assert!(
        primitive
            .is_feasible(&moving(4.0, 2.0))
            .expect("a valid state")
    );
    assert!(
        primitive
            .is_feasible(&moving(4.1, 2.0))
            .expect("a valid state")
    );
    assert!(
        !primitive
            .is_feasible(&moving(1.0, 2.0))
            .expect("a valid state")
    );
}

#[test]
fn turning_the_other_way_does_not_change_the_verdict() {
    let primitive = primitive(2.0);
    for (speed, turn_rate) in [(4.0, 1.0), (1.0, 4.0), (0.5, 0.2)] {
        assert_eq!(
            primitive
                .is_feasible(&moving(speed, turn_rate))
                .expect("a valid state"),
            primitive
                .is_feasible(&moving(speed, -turn_rate))
                .expect("a valid state"),
            "{speed} at {turn_rate}"
        );
    }
}

#[test]
fn scaling_the_speed_and_the_turn_rate_together_leaves_the_verdict_alone() {
    // The constraint is on the ratio, so driving the same arc faster is
    // the same arc. This is the property that makes the check a statement
    // about geometry rather than about speed.
    let primitive = primitive(1.5);
    for (speed, turn_rate) in [(3.0, 1.0), (1.0, 1.0), (6.0, 2.0)] {
        let verdict = primitive
            .is_feasible(&moving(speed, turn_rate))
            .expect("a valid state");
        for scale in [0.25, 2.0, 40.0] {
            assert_eq!(
                primitive
                    .is_feasible(&moving(speed * scale, turn_rate * scale))
                    .expect("a valid state"),
                verdict,
                "{speed} at {turn_rate} scaled by {scale}"
            );
        }
    }
}

#[test]
fn a_spin_on_the_spot_is_infeasible_for_a_car() {
    // Zero speed through a nonzero turn rate is a circle of zero radius,
    // which is a tank turning in place. A Dubins car cannot do it, and
    // reporting otherwise would let a planner hand one to a vehicle that
    // then drives through whatever it was trying to avoid.
    assert!(
        !primitive(1.0)
            .is_feasible(&moving(0.0, 1.0))
            .expect("a valid state")
    );
}

#[test]
fn a_turning_radius_that_is_not_positive_is_refused() {
    // A radius of zero accepts every state, which reads as a check that
    // passed rather than as one that was never configured.
    for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(DubinsPrimitive::new(radius), Err(Error::OutOfRange { .. })),
            "a radius of {radius} was accepted"
        );
    }
    assert!((primitive(2.5).turning_radius() - 2.5).abs() < 1e-15);
}

#[test]
fn a_state_that_is_not_a_finite_position_is_refused_rather_than_accepted() {
    let primitive = primitive(1.0);
    assert!(matches!(
        primitive.is_feasible(&[0.0]),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        primitive.is_feasible(&[0.0, 0.0, 0.0, 1.0, f64::NAN]),
        Err(Error::NotFinite { .. })
    ));
}
