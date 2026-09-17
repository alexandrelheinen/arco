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

//! The actuator array and the joint-space tracker.
//!
//! The allocation has a real oracle behind it and most of these use it:
//! a force vector that claims to produce a wrench can be multiplied back
//! through the grasp matrix, and the answer has to be the wrench that was
//! asked for. Nothing needs to know what the right forces are.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_control::actuator::{ActuatorArray, ActuatorSettings, HazardPolicy};
use arco_control::body::{CircleBody, RigidBody};
use arco_control::joint::{JointLimits, JointSpaceTracker, JointTrackerSettings};
use arco_core::Error;
use arco_core::protocols::NearestObstacle;
use arco_mapping::occupancy::KdTreeOccupancy;

fn body() -> CircleBody {
    CircleBody::new(1.0, 1.0, 0.0, 0.0, 0.0).expect("a valid disk")
}

fn array(count: usize) -> ActuatorArray {
    ActuatorArray::new(count, ActuatorSettings::default()).expect("a valid array")
}

/// How far a wrench is from the one that was asked for.
fn wrench_error(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()))
}

// --------------------------------------------------------- allocation ----

#[test]
fn an_allocated_force_vector_produces_the_wrench_it_was_asked_for() {
    // The oracle. Allocation inverts the grasp matrix, so multiplying the
    // answer back through the matrix has to return the input, and that is
    // checkable without knowing what the forces should have been.
    let body = body();
    for count in [3, 4, 6, 12] {
        let array = array(count);
        let matrix = array.grasp_matrix(&body).expect("a valid matrix");
        for wrench in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.5, -1.5, 0.75],
            [-3.0, 4.0, -2.0],
        ] {
            let forces = array
                .allocate_forces(wrench, &body)
                .expect("a valid allocation");
            let produced = matrix.wrench(&forces).expect("the width matches");
            assert!(
                wrench_error(produced, wrench) < 1e-9,
                "{count} actuators: asked for {wrench:?}, got {produced:?}"
            );
        }
    }
}

#[test]
fn a_radial_allocation_leaves_every_tangential_axis_at_zero() {
    // The spring contact model produces no tangential force, so an
    // allocation that assigned one would be describing forces the array
    // cannot deliver.
    let body = body();
    let array = array(6);
    let forces = array
        .allocate_radial_forces([1.0, 2.0, 0.5], &body)
        .expect("a valid allocation");
    assert_eq!(forces.len(), 12);
    for (index, &force) in forces.iter().enumerate() {
        if index % 2 == 1 {
            assert!(force.abs() < 1e-15, "tangential axis {index} got {force}");
        }
    }
}

#[test]
fn a_radial_only_array_cannot_produce_a_pure_torque() {
    // Radial forces point at the body center, so their lever arm is zero
    // and no combination of them makes a torque. The allocation returns
    // the closest it can rather than pretending, which is what a
    // pseudo-inverse is for.
    let body = body();
    let array = array(6);
    let matrix = array.grasp_matrix(&body).expect("a valid matrix");
    let forces = array
        .allocate_radial_forces([0.0, 0.0, 1.0], &body)
        .expect("a valid allocation");
    let produced = matrix.wrench(&forces).expect("the width matches");
    assert!(
        produced.get(2).copied().unwrap_or_default().abs() < 1e-9,
        "radial forces produced a torque of {:?}",
        produced.get(2)
    );
    // And it did not invent a force to compensate either.
    assert!(produced.first().copied().unwrap_or_default().abs() < 1e-9);
}

#[test]
fn a_rotated_body_rotates_the_wrench_the_same_way() {
    // Equivariance: the actuators are placed in the body frame, so turning
    // the body turns the force it can produce by exactly as much.
    let mut turned = body();
    turned
        .reset(0.0, 0.0, core::f64::consts::FRAC_PI_2)
        .expect("finite");
    let array = array(4);

    let upright = array
        .grasp_matrix(&body())
        .expect("a valid matrix")
        .wrench(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("the width matches");
    let rotated = array
        .grasp_matrix(&turned)
        .expect("a valid matrix")
        .wrench(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("the width matches");

    // A quarter turn maps (x, y) to (-y, x).
    let expected = [
        -upright.get(1).copied().unwrap_or_default(),
        upright.first().copied().unwrap_or_default(),
        upright.get(2).copied().unwrap_or_default(),
    ];
    assert!(
        wrench_error(rotated, expected) < 1e-12,
        "{rotated:?} against {expected:?}"
    );
}

#[test]
fn an_array_too_small_to_span_a_wrench_is_rejected() {
    for count in [0, 1, 2] {
        assert!(
            matches!(
                ActuatorArray::new(count, ActuatorSettings::default()),
                Err(Error::TooFew { .. })
            ),
            "accepted {count} actuators"
        );
    }
    assert!(ActuatorArray::new(3, ActuatorSettings::default()).is_ok());
}

#[test]
fn a_degenerate_setting_is_rejected_at_construction() {
    for frequency in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            ActuatorArray::new(
                4,
                ActuatorSettings {
                    natural_frequency: frequency,
                    ..ActuatorSettings::default()
                }
            ),
            Err(Error::OutOfRange { .. })
        ));
    }
    assert!(matches!(
        ActuatorArray::new(
            4,
            ActuatorSettings {
                damping_ratio: -0.1,
                ..ActuatorSettings::default()
            }
        ),
        Err(Error::OutOfRange { .. })
    ));
}

// ---------------------------------------------------------- dynamics ----

#[test]
fn the_actuators_settle_on_their_setpoints() {
    // Both axes are second-order closed loops, so the test of the
    // integration is whether they converge rather than what they do on any
    // one step.
    let body = body();
    let mut array = array(4);
    // Along the second axis rather than the first: aiming along the first
    // happens to land on the placement the array already had, so the run
    // would converge without anything having moved.
    array.aim_at([0.0, 1.0, 0.0], &body).expect("finite");

    let initial: Vec<f64> = array.angles().to_vec();
    let references: Vec<f64> = array.reference_angles().to_vec();
    for _ in 0..4000 {
        array.step(0.001).expect("valid step");
    }

    for (index, (&angle, &reference)) in array.angles().iter().zip(&references).enumerate() {
        assert!(
            (angle - reference).abs() < 1e-3,
            "actuator {index} settled at {angle} rather than {reference}"
        );
    }
    assert!(
        initial
            .iter()
            .zip(&references)
            .any(|(start, target)| (start - target).abs() > 1e-6),
        "the setpoints were where the actuators already sat, so nothing was tested"
    );
}

#[test]
fn the_integration_stays_bounded_at_the_coarse_step_it_is_used_with() {
    // The ordering inside the integrator is not incidental. The
    // velocity-first arrangement has a spectral radius above one at these
    // step sizes and the actuators diverge instead of settling.
    let body = body();
    let mut array = array(4);
    array.aim_at([1.0, 1.0, 0.0], &body).expect("finite");
    for _ in 0..500 {
        array.step(0.01).expect("valid step");
    }
    for &angle in array.angles() {
        assert!(angle.is_finite() && angle.abs() < 100.0, "{angle}");
    }
    for &rate in array.angle_velocities() {
        assert!(rate.is_finite() && rate.abs() < 100.0, "{rate}");
    }
}

#[test]
fn the_radial_axis_appears_only_once_it_is_asked_for() {
    let body = body();
    let mut array = array(4);
    assert!(array.radii().is_none());
    assert!(array.reference_radii().is_none());

    assert!(array.radii_velocities().is_none());

    array.init_radii(&body).expect("a valid body");
    let nominal = body.bounding_radius() + array.settings().standoff;
    for &radius in array.radii().expect("the axis now exists") {
        assert!((radius - nominal).abs() < 1e-12);
    }
    // The axis starts at rest, so a caller reading the rates straight
    // after it appears sees zeros rather than whatever the angular axis
    // happened to be doing.
    for &rate in array.radii_velocities().expect("the axis now exists") {
        assert!(rate.abs() < 1e-12, "{rate}");
    }
}

#[test]
fn a_spring_setpoint_produces_the_force_it_was_computed_for() {
    // The inversion law: a spring compressed by F over k pushes with F, so
    // driving the radial axis to its setpoint has to produce the desired
    // radial force, up to the precompression bias that cancels in the net
    // wrench.
    let body = body();
    let mut array = array(6);
    let desired = array
        .allocate_radial_forces([2.0, 1.0, 0.0], &body)
        .expect("a valid allocation");
    array
        .compute_reference_radii(&desired, &body)
        .expect("a valid inversion");

    // Drive the radial axis onto its setpoint.
    for _ in 0..8000 {
        array.step(0.001).expect("valid step");
    }
    let produced = array.spring_forces(&body).expect("a valid body");

    // The bias is common to every actuator, so the differences between
    // them are what has to match.
    let offsets: Vec<f64> = produced
        .iter()
        .step_by(2)
        .zip(desired.iter().step_by(2))
        .map(|(made, wanted)| made - wanted)
        .collect();
    let first = offsets.first().copied().unwrap_or_default();
    for (index, &offset) in offsets.iter().enumerate() {
        assert!(
            (offset - first).abs() < 1e-2,
            "actuator {index} is off by {offset} against {first} for the first"
        );
    }
}

#[test]
fn a_spring_that_is_stretched_rather_than_compressed_pushes_nothing() {
    // A spring can only push. One that would pull the body back toward the
    // actuator is a contact that has been lost, not a tension.
    let body = body();
    let mut array = array(4);
    array.init_radii(&body).expect("a valid body");
    // Set the setpoint far outside and let the actuators retract.
    array
        .compute_reference_radii(&[-100.0; 8], &body)
        .expect("a valid inversion");
    for _ in 0..2000 {
        array.step(0.001).expect("valid step");
    }
    for force in array.spring_forces(&body).expect("a valid body") {
        assert!(force >= 0.0, "a spring pulled with {force}");
    }
}

#[test]
fn applying_the_spring_forces_moves_the_body() {
    let mut body = body();
    let mut array = array(6);
    let desired = array
        .allocate_radial_forces([5.0, 0.0, 0.0], &body)
        .expect("a valid allocation");
    array
        .compute_reference_radii(&desired, &body)
        .expect("a valid inversion");
    for _ in 0..2000 {
        array.step(0.001).expect("valid step");
    }
    array
        .apply_spring_forces(&mut body)
        .expect("a valid application");
    body.step(0.01).expect("valid step");
    assert!(
        body.state().velocity().first().copied().unwrap_or_default() > 0.0,
        "the body did not move along the requested force"
    );
}

#[test]
fn a_mismatched_force_vector_is_rejected() {
    let body = body();
    let mut array = array(4);
    assert!(matches!(
        array.compute_reference_radii(&[1.0, 2.0], &body),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        array.compute_reference_radii(&[f64::NAN; 8], &body),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        array.set_angles(&[0.0, 1.0]),
        Err(Error::DimensionMismatch { .. })
    ));
}

// --------------------------------------------------------- repulsion ----

#[test]
fn a_hazard_inside_the_influence_radius_pushes_the_body_away_from_it() {
    let body = body();
    let array = array(6);
    let field = KdTreeOccupancy::new(&[vec![3.0, 0.0]], 0.5).expect("a valid field");
    let wrench = array
        .repulsive_wrench(&body, &HazardPolicy::Occupancy(field), &[], 1.0, 5.0)
        .expect("a valid wrench");
    assert!(
        wrench.first().copied().unwrap_or_default() < 0.0,
        "a hazard to the right should push left, got {wrench:?}"
    );
}

#[test]
fn a_hazard_outside_the_influence_radius_does_nothing() {
    let body = body();
    let array = array(6);
    let field = KdTreeOccupancy::new(&[vec![100.0, 0.0]], 0.5).expect("a valid field");
    let wrench = array
        .repulsive_wrench(&body, &HazardPolicy::Occupancy(field), &[], 1.0, 2.0)
        .expect("a valid wrench");
    for value in wrench {
        assert!(value.abs() < 1e-12, "{wrench:?}");
    }
}

#[test]
fn a_peer_is_treated_as_a_hazard_alongside_the_map() {
    let body = body();
    let array = array(6);
    let empty: HazardPolicy<KdTreeOccupancy> = HazardPolicy::None;
    let wrench = array
        .repulsive_wrench(&body, &empty, &[(2.5, 0.0)], 1.0, 5.0)
        .expect("a valid wrench");
    assert!(
        wrench.first().copied().unwrap_or_default() < 0.0,
        "a peer to the right should push left, got {wrench:?}"
    );
}

#[test]
fn a_custom_hazard_query_is_the_one_that_is_asked() {
    let body = body();
    let array = array(6);
    let policy: HazardPolicy<KdTreeOccupancy> = HazardPolicy::Custom(Box::new(|_point| {
        Ok(NearestObstacle {
            distance: 0.5,
            point: vec![0.0, 4.0],
        })
    }));
    let wrench = array
        .repulsive_wrench(&body, &policy, &[], 1.0, 5.0)
        .expect("a valid wrench");
    assert!(
        wrench.get(1).copied().unwrap_or_default() < 0.0,
        "a hazard above should push down, got {wrench:?}"
    );
    assert!(format!("{policy:?}").contains("Custom"));
}

#[test]
fn a_degenerate_repulsion_setting_is_rejected() {
    let body = body();
    let array = array(4);
    let empty: HazardPolicy<KdTreeOccupancy> = HazardPolicy::None;
    for value in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            array.repulsive_wrench(&body, &empty, &[], value, 1.0),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            array.repulsive_wrench(&body, &empty, &[], 1.0, value),
            Err(Error::OutOfRange { .. })
        ));
    }
}

// ----------------------------------------------- joint-space tracker ----

fn tracker() -> JointSpaceTracker<KdTreeOccupancy> {
    JointSpaceTracker::new(
        JointTrackerSettings::new(JointLimits::uniform(3, 1.0, 2.0).expect("valid limits")),
        None,
    )
    .expect("valid settings")
}

#[test]
fn a_tracker_converges_on_its_target() {
    let mut tracker = tracker();
    tracker.reset(&[0.0, 0.0, 0.0]).expect("valid reset");
    let target = [1.0, -0.5, 0.25];
    for _ in 0..2000 {
        tracker.step(&target, 0.01).expect("valid step");
    }
    for (index, (&reached, &wanted)) in tracker.configuration().iter().zip(&target).enumerate() {
        assert!(
            (reached - wanted).abs() < 1e-6,
            "axis {index} stopped at {reached} rather than {wanted}"
        );
    }
}

#[test]
fn no_axis_ever_exceeds_its_velocity_or_acceleration_limit() {
    // The two saturations are per axis and independent, which is what lets
    // a three-axis machine with one slow axis still move the other two at
    // full speed.
    let limits = JointLimits::new(vec![1.0, 0.25, 4.0], vec![2.0, 0.5, 8.0]).expect("valid limits");
    let mut tracker = JointSpaceTracker::<KdTreeOccupancy>::new(
        JointTrackerSettings {
            proportional_gain: 50.0,
            ..JointTrackerSettings::new(limits.clone())
        },
        None,
    )
    .expect("valid settings");
    tracker.reset(&[0.0, 0.0, 0.0]).expect("valid reset");

    let dt = 0.01;
    let mut previous = vec![0.0; 3];
    for step in 0..400_i32 {
        let target = if step % 100 < 50 {
            [10.0, 10.0, 10.0]
        } else {
            [-10.0, -10.0, -10.0]
        };
        let reached = tracker.step(&target, dt).expect("valid step");
        for index in 0..3 {
            let velocity = reached.velocity.get(index).copied().unwrap_or_default();
            let limit = limits.max_velocity.get(index).copied().unwrap_or_default();
            let acceleration = limits
                .max_acceleration
                .get(index)
                .copied()
                .unwrap_or_default();
            assert!(
                velocity.abs() <= limit + 1e-12,
                "axis {index} reached {velocity} against a limit of {limit}"
            );
            let change = (velocity - previous.get(index).copied().unwrap_or_default()).abs() / dt;
            assert!(
                change <= acceleration + 1e-9,
                "axis {index} changed at {change} against a limit of {acceleration}"
            );
        }
        previous = reached.velocity;
    }
}

#[test]
fn repulsion_pushes_the_configuration_off_an_obstacle() {
    let field = KdTreeOccupancy::new(&[vec![1.0, 0.0, 0.0]], 0.5).expect("a valid field");
    let limits = JointLimits::uniform(3, 1.0, 10.0).expect("valid limits");
    let mut tracker = JointSpaceTracker::new(
        JointTrackerSettings {
            repulsion_gain: 1.0,
            ..JointTrackerSettings::new(limits)
        },
        Some(field),
    )
    .expect("valid settings");
    // Start just outside the obstacle, with the target on the far side of
    // it, so the proportional term and the repulsion disagree.
    tracker.reset(&[0.4, 0.0, 0.0]).expect("valid reset");
    let step = tracker.step(&[2.0, 0.0, 0.0], 0.01).expect("valid step");
    assert!(
        step.velocity.first().copied().unwrap_or_default() < 1.0,
        "repulsion did not slow the approach: {:?}",
        step.velocity
    );
}

#[test]
fn a_tracker_with_no_map_never_repels() {
    let mut with_map = JointSpaceTracker::new(
        JointTrackerSettings {
            repulsion_gain: 5.0,
            ..JointTrackerSettings::new(JointLimits::uniform(2, 1.0, 5.0).expect("valid limits"))
        },
        None::<KdTreeOccupancy>,
    )
    .expect("valid settings");
    with_map.reset(&[0.0, 0.0]).expect("valid reset");
    let step = with_map.step(&[1.0, 0.0], 0.01).expect("valid step");
    assert!(step.velocity.get(1).copied().unwrap_or_default().abs() < 1e-15);
}

#[test]
fn a_degenerate_tracker_is_rejected() {
    assert!(matches!(
        JointLimits::uniform(0, 1.0, 1.0),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        JointLimits::uniform(2, 0.0, 1.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        JointLimits::new(vec![1.0, 1.0], vec![1.0]),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        JointSpaceTracker::<KdTreeOccupancy>::new(
            JointTrackerSettings {
                repulsion_gain: -1.0,
                ..JointTrackerSettings::new(JointLimits::uniform(2, 1.0, 1.0).expect("valid"))
            },
            None,
        ),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn a_mismatched_configuration_is_rejected() {
    let mut tracker = tracker();
    assert!(matches!(
        tracker.reset(&[0.0, 0.0]),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tracker.step(&[0.0, 0.0], 0.01),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        tracker.step(&[f64::NAN, 0.0, 0.0], 0.01),
        Err(Error::NotFinite { .. })
    ));
}
