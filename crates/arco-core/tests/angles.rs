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

//! FR-INV-15: an angular difference feeding a control law is the wrapped one.
//!
//! The failure this guards against is specific and it is not a rounding
//! problem. A heading of 179 degrees and one of minus 179 are two degrees
//! apart, and a controller that subtracts them gets 358, which is a
//! command at the limit in the wrong direction. Every crossing of the
//! branch cut produces it, and a vehicle driving north crosses it
//! constantly.

use core::f64::consts::{PI, TAU};

use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::numeric::{angle_difference, wrap_angle};

#[test]
fn wrapping_lands_every_angle_in_one_turn() {
    let mut angle = -40.0_f64;
    while angle <= 40.0 {
        let wrapped = wrap_angle(angle).expect("a finite angle wraps");
        assert!(
            (-PI - 1e-12..PI + 1e-12).contains(&wrapped),
            "{angle} wrapped to {wrapped}"
        );
        // Wrapping changes the angle by a whole number of turns and
        // nothing else.
        let turns = (angle - wrapped) / TAU;
        assert!(
            (turns - turns.round()).abs() < 1e-9,
            "{angle} to {wrapped} is {turns} turns"
        );
        angle += 0.37;
    }
}

#[test]
fn wrapping_is_idempotent() {
    // A representation that is normalized on every return has to survive
    // being normalized again, or a value's meaning depends on how many
    // times it has been through the library.
    for angle in [-TAU, -PI, -0.1, 0.0, 0.1, PI, TAU, 100.0] {
        let once = wrap_angle(angle).expect("finite");
        let twice = wrap_angle(once).expect("finite");
        assert!((once - twice).abs() < 1e-15, "{angle}: {once} then {twice}");
    }
}

#[test]
fn a_difference_across_the_branch_cut_is_small() {
    // The case the requirement exists for. Naive subtraction gives 358
    // degrees; the answer is 2.
    let left = wrap_angle(179.0_f64.to_radians()).expect("finite");
    let right = wrap_angle((-179.0_f64).to_radians()).expect("finite");
    let difference = angle_difference(left, right).expect("finite");
    assert!(
        difference.abs() < 3.0_f64.to_radians(),
        "the difference came out as {} degrees",
        difference.to_degrees()
    );
}

#[test]
fn a_difference_is_antisymmetric_and_bounded() {
    let mut left = -10.0_f64;
    while left <= 10.0 {
        let mut right = -10.0_f64;
        while right <= 10.0 {
            let forward = angle_difference(left, right).expect("finite");
            let backward = angle_difference(right, left).expect("finite");
            assert!(
                forward.abs() <= PI + 1e-12,
                "{left} minus {right} gave {forward}"
            );
            // Exactly pi is its own negation under wrapping, so the
            // antisymmetry holds up to that one value.
            assert!(
                (forward + backward).abs() < 1e-9 || (forward.abs() - PI).abs() < 1e-9,
                "{forward} and {backward} are not opposites"
            );
            right += 1.13;
        }
        left += 1.13;
    }
}

#[test]
fn adding_whole_turns_to_either_side_changes_nothing() {
    // The metamorphic statement of the same property: a difference is
    // about the rotation between two directions, and a direction is
    // unchanged by a whole turn.
    let base = angle_difference(0.4, -0.9).expect("finite");
    for turns in [-3.0_f64, -1.0, 1.0, 2.0] {
        let shifted = angle_difference(turns.mul_add(TAU, 0.4), -0.9).expect("finite");
        assert!(
            (shifted - base).abs() < 1e-9,
            "{turns} turns gave {shifted}"
        );
        let other = angle_difference(0.4, turns.mul_add(TAU, -0.9)).expect("finite");
        assert!((other - base).abs() < 1e-9, "{turns} turns gave {other}");
    }
}

#[test]
fn a_pose_normalizes_its_heading_on_construction() {
    // The rotation representation the control layer passes around, so
    // nothing downstream has to remember to wrap it.
    for heading in [-10.0, -PI, 0.0, PI, 7.5, 100.0] {
        let pose = Pose::new(1.0, 2.0, heading).expect("finite");
        assert!(
            (-PI - 1e-12..PI + 1e-12).contains(&pose.heading()),
            "{heading} became {}",
            pose.heading()
        );
    }
}

#[test]
fn a_pose_difference_crosses_the_branch_cut_correctly() {
    let north = Pose::new(0.0, 0.0, PI - 0.01).expect("finite");
    let almost = Pose::new(0.0, 0.0, -PI + 0.01).expect("finite");
    let difference = north.heading_difference(almost).expect("finite");
    assert!(
        difference.abs() < 0.03,
        "two poses a fiftieth of a radian apart read as {difference}"
    );
}

#[test]
fn a_non_finite_angle_is_rejected_rather_than_wrapped() {
    // `atan2` of a NaN is a NaN, and a NaN heading poisons every later
    // comparison silently. FR-SAFE-07.
    for angle in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(wrap_angle(angle), Err(Error::NotFinite { .. })),
            "wrapped {angle}"
        );
        assert!(matches!(
            angle_difference(angle, 0.0),
            Err(Error::NotFinite { .. })
        ));
        assert!(matches!(
            angle_difference(0.0, angle),
            Err(Error::NotFinite { .. })
        ));
        assert!(matches!(
            Pose::new(0.0, 0.0, angle),
            Err(Error::NotFinite { .. })
        ));
    }
}
