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

//! FR-INV-14: kinematics round-trip, stay in range, and refuse the rest.

use arco_core::Error;
use arco_core::rng::Pcg64;
use arco_kinematics::rr::{JointAngles, RrRobot};
use arco_kinematics::rrp::{Configuration, RrpRobot};
use core::f64::consts::PI;

const TOLERANCE: f64 = 1e-9;

#[test]
fn forward_and_inverse_kinematics_round_trip() {
    // FR-INV-14, over the whole reachable annulus rather than a few
    // hand-picked poses.
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    let mut generator = Pcg64::seed_from_u64(31);

    for _ in 0..2000 {
        let shoulder = generator.next_f64().mul_add(2.0 * PI, -PI);
        let elbow = generator.next_f64().mul_add(2.0 * PI, -PI);
        let angles = JointAngles { shoulder, elbow };
        let (x, y) = arm.forward_kinematics(angles).unwrap();

        let solutions = arm.inverse_kinematics(x, y, TOLERANCE).unwrap();
        assert!(!solutions.is_empty(), "a reachable pose returned nothing");

        for solution in solutions {
            let (back_x, back_y) = arm.forward_kinematics(solution).unwrap();
            assert!(
                (back_x - x).abs() < 1e-9 && (back_y - y).abs() < 1e-9,
                "({x}, {y}) came back as ({back_x}, {back_y})"
            );
        }
    }
}

#[test]
fn both_elbow_branches_are_returned_and_differ() {
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    let solutions = arm.inverse_kinematics(1.2, 0.4, TOLERANCE).unwrap();
    assert_eq!(solutions.len(), 2);
    assert!(
        solutions[0].elbow * solutions[1].elbow <= 0.0,
        "the two branches should bend opposite ways: {solutions:?}"
    );
}

#[test]
fn a_target_outside_the_annulus_is_refused_rather_than_approximated() {
    // FR-INV-14. Returning the closest reachable pose would be a silently
    // wrong answer, which is worse than no answer.
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    assert!(
        arm.inverse_kinematics(5.0, 0.0, TOLERANCE)
            .unwrap()
            .is_empty()
    );
    assert!(
        arm.inverse_kinematics(0.0, 0.0, TOLERANCE)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_target_on_the_boundary_is_reachable() {
    // The law of cosines lands just outside the domain of acos here
    // through ordinary rounding, and acos of 1.0000001 is NaN. The clamp
    // is what keeps the boundary reachable.
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    for radius in [1.8_f64, 0.2] {
        let solutions = arm.inverse_kinematics(radius, 0.0, TOLERANCE).unwrap();
        assert!(!solutions.is_empty(), "radius {radius} returned nothing");
        for solution in solutions {
            let (x, y) = arm.forward_kinematics(solution).unwrap();
            assert!(
                x.is_finite() && y.is_finite(),
                "{solution:?} gave ({x}, {y})"
            );
        }
    }
}

#[test]
fn every_returned_angle_is_wrapped() {
    // FR-INV-15.
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    let mut generator = Pcg64::seed_from_u64(5);
    for _ in 0..500 {
        let x = generator.next_f64().mul_add(3.6, -1.8);
        let y = generator.next_f64().mul_add(3.6, -1.8);
        for solution in arm.inverse_kinematics(x, y, TOLERANCE).unwrap() {
            assert!((-PI..PI).contains(&solution.shoulder), "{solution:?}");
            assert!((-PI..PI).contains(&solution.elbow), "{solution:?}");
        }
    }
}

#[test]
fn a_non_finite_target_is_rejected() {
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    assert!(matches!(
        arm.inverse_kinematics(f64::NAN, 0.0, TOLERANCE),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        arm.inverse_kinematics(0.0, 0.0, -1.0),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn a_degenerate_arm_is_rejected() {
    assert!(matches!(
        RrRobot::new(0.0, 1.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        RrRobot::new(1.0, -1.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        RrRobot::new(f64::NAN, 1.0),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn the_link_segments_join_end_to_end() {
    let arm = RrRobot::new(1.0, 0.8).unwrap();
    let angles = JointAngles {
        shoulder: 0.4,
        elbow: -0.7,
    };
    let [base, elbow, tip] = arm.link_segments(angles).unwrap();

    assert!(base.0.abs() < 1e-12 && base.1.abs() < 1e-12);
    let first = (elbow.0 - base.0).hypot(elbow.1 - base.1);
    let second = (tip.0 - elbow.0).hypot(tip.1 - elbow.1);
    assert!((first - 1.0).abs() < 1e-12, "{first}");
    assert!((second - 0.8).abs() < 1e-12, "{second}");
    assert_eq!(tip, arm.forward_kinematics(angles).unwrap());
}

#[test]
fn an_equal_link_arm_can_reach_its_own_base() {
    let arm = RrRobot::new(1.0, 1.0).unwrap();
    let (inner, outer) = arm.workspace_annulus();
    assert!(inner.abs() < 1e-12, "{inner}");
    assert!((outer - 2.0).abs() < 1e-12, "{outer}");
    assert!(
        !arm.inverse_kinematics(0.0, 0.0, TOLERANCE)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn the_lift_is_independent_of_the_planar_solution() {
    let robot = RrpRobot::new(1.0, 0.8, 0.0, 4.0).unwrap();
    let planar = robot.arm().inverse_kinematics(1.2, 0.4, TOLERANCE).unwrap();
    let spatial = robot.inverse_kinematics(1.2, 0.4, 2.5, TOLERANCE).unwrap();

    assert_eq!(planar.len(), spatial.len());
    for (flat, lifted) in planar.iter().zip(&spatial) {
        assert_eq!(*flat, lifted.angles);
        assert!((lifted.height - 2.5).abs() < 1e-12);
    }
}

#[test]
fn a_height_outside_the_lift_range_is_rejected() {
    // FR-INV-14: a joint command outside its limits is refused rather
    // than clamped, since clamping hides the caller's mistake.
    let robot = RrpRobot::new(1.0, 0.8, 0.0, 4.0).unwrap();
    for height in [-0.1, 4.1, f64::NAN, f64::INFINITY] {
        assert!(
            robot
                .inverse_kinematics(1.2, 0.4, height, TOLERANCE)
                .is_err(),
            "accepted a height of {height}"
        );
    }
}

#[test]
fn spatial_kinematics_round_trip() {
    let robot = RrpRobot::new(1.0, 0.8, 0.0, 4.0).unwrap();
    let mut generator = Pcg64::seed_from_u64(77);

    for _ in 0..500 {
        let configuration = Configuration {
            angles: JointAngles {
                shoulder: generator.next_f64().mul_add(2.0 * PI, -PI),
                elbow: generator.next_f64().mul_add(2.0 * PI, -PI),
            },
            height: generator.next_f64() * 4.0,
        };
        let (x, y, z) = robot.forward_kinematics(configuration).unwrap();

        for solution in robot.inverse_kinematics(x, y, z, TOLERANCE).unwrap() {
            let (back_x, back_y, back_z) = robot.forward_kinematics(solution).unwrap();
            assert!((back_x - x).abs() < 1e-9, "{back_x} against {x}");
            assert!((back_y - y).abs() < 1e-9, "{back_y} against {y}");
            assert!((back_z - z).abs() < 1e-12, "{back_z} against {z}");
        }
    }
}

#[test]
fn a_degenerate_lift_range_is_rejected() {
    assert!(RrpRobot::new(1.0, 0.8, 4.0, 4.0).is_err());
    assert!(RrpRobot::new(1.0, 0.8, 4.0, 0.0).is_err());
    assert!(RrpRobot::new(1.0, 0.8, f64::NAN, 4.0).is_err());
}

#[test]
fn the_spatial_link_segments_share_the_lift_height() {
    let robot = RrpRobot::new(1.0, 0.8, 0.5, 4.0).unwrap();
    let configuration = Configuration {
        angles: JointAngles {
            shoulder: 0.3,
            elbow: 0.6,
        },
        height: 2.0,
    };
    let segments = robot.link_segments(configuration).unwrap();
    assert!(
        (segments[0].2 - 0.5).abs() < 1e-12,
        "the base sits on the rail"
    );
    for segment in &segments[1..] {
        assert!((segment.2 - 2.0).abs() < 1e-12, "{segment:?}");
    }
}
