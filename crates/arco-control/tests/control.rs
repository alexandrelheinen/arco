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

//! What the control layer does, as opposed to what it refuses.
//!
//! The limits and the interval band have their own files because they are
//! requirements. This is the behavior underneath them.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

mod common;

use arco_control::avoidance::ArtificialPotentialField;
use arco_control::body::{CircleBody, RigidBody, SquareBody};
use arco_control::limits::{CommandLimits, IntervalBand};
use arco_control::pid::{AntiWindup, PidController, PidGains, PidSettings};
use arco_control::pursuit::PurePursuitTracker;
use arco_control::tracking::{TrackingLoop, TrackingSettings};
use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::protocols::{AvoidanceStrategy, PathTracker};
use arco_mapping::occupancy::KdTreeOccupancy;

use common::{FixedBias, NoAvoidance, Unicycle, straight_path};

/// Whether two triples agree to within rounding.
fn close(left: [f64; 3], right: [f64; 3]) -> bool {
    left.iter().zip(right).all(|(a, b)| (a - b).abs() < 1e-12)
}

// --------------------------------------------------------------- PID ----

#[test]
fn a_unit_interval_reproduces_the_python_controller() {
    // `arco.control.pid.PIDController` summed raw errors and differenced
    // raw errors, which is this controller at an interval of exactly one
    // second. The binding passes that, so nothing visible from Python
    // changes; this is the test that says so.
    let mut controller = PidController::new(PidSettings {
        gains: PidGains {
            proportional: 1.0,
            integral: 0.5,
            derivative: 0.1,
        },
        anti_windup: AntiWindup::None,
        ..PidSettings::default()
    })
    .expect("valid settings");

    // Python: error = 1, integral = 1, derivative = 0 on the first call
    // because there is no previous error to difference against.
    let first = controller.step(0.0, 1.0, 1.0).expect("valid step");
    assert!((first - (1.0 + 0.5)).abs() < 1e-12, "{first}");

    // Python: error = 0.5, integral = 1.5, derivative = 0.5 - 1.0.
    let second = controller.step(0.5, 1.0, 1.0).expect("valid step");
    let expected = 0.5_f64.mul_add(1.5, 0.1_f64.mul_add(-0.5, 0.5));
    assert!((second - expected).abs() < 1e-12, "{second}");
}

#[test]
fn the_first_step_takes_no_derivative_of_nothing() {
    // Treating the absent previous error as zero makes the derivative term
    // a spike proportional to the initial error, which is the derivative
    // kick. A controller that does that on every reset is a controller
    // nobody can tune.
    let mut controller = PidController::new(PidSettings {
        gains: PidGains {
            proportional: 0.0,
            integral: 0.0,
            derivative: 100.0,
        },
        ..PidSettings::default()
    })
    .expect("valid settings");
    let first = controller.step(0.0, 50.0, 0.01).expect("valid step");
    assert!(first.abs() < 1e-12, "the first step kicked by {first}");

    // The second one does differentiate, so the term is not simply dead.
    let second = controller.step(0.0, 51.0, 0.01).expect("valid step");
    assert!(second.abs() > 1.0, "{second}");
}

#[test]
fn back_calculation_stops_the_integrator_winding() {
    // Astrom's point: while the output is saturated the loop is open,
    // because the actuator stays at its limit whatever the process does.
    // An integrator that keeps accumulating through that holds the command
    // pinned long after the error changed sign.
    let settings = |anti_windup| PidSettings {
        gains: PidGains {
            proportional: 1.0,
            integral: 1.0,
            derivative: 0.0,
        },
        output_limits: (-1.0, 1.0),
        anti_windup,
        interval: IntervalBand::default(),
    };

    let mut protected =
        PidController::new(settings(AntiWindup::BackCalculation { tracking_gain: 1.0 }))
            .expect("valid settings");
    let mut winding = PidController::new(settings(AntiWindup::None)).expect("valid settings");

    // Drive both hard into the upper limit for a while.
    for _ in 0..50 {
        protected.step(0.0, 10.0, 0.1).expect("valid step");
        winding.step(0.0, 10.0, 0.1).expect("valid step");
    }
    assert!(
        protected.integral() < winding.integral(),
        "back-calculation wound as far as no anti-windup: {} against {}",
        protected.integral(),
        winding.integral()
    );

    // Now ask for the opposite. The protected controller should leave the
    // limit sooner than the winding one, which has to unwind first.
    let mut protected_steps = 0_usize;
    let mut winding_steps = 0_usize;
    for step in 1..=200_usize {
        if protected.step(0.0, -10.0, 0.1).expect("valid step") <= 0.0 && protected_steps == 0 {
            protected_steps = step;
        }
        if winding.step(0.0, -10.0, 0.1).expect("valid step") <= 0.0 && winding_steps == 0 {
            winding_steps = step;
        }
    }
    assert!(
        protected_steps > 0,
        "the protected controller never reversed"
    );
    assert!(
        protected_steps < winding_steps || winding_steps == 0,
        "protected took {protected_steps} steps to reverse, winding took {winding_steps}"
    );
}

#[test]
fn an_unlimited_controller_behaves_as_though_anti_windup_were_absent() {
    // The back-calculation correction is exactly zero while unsaturated,
    // so it costs nothing in normal operation. Asserted rather than
    // assumed, because a correction that leaked would change the tuning of
    // every controller in the library.
    let gains = PidGains {
        proportional: 2.0,
        integral: 3.0,
        derivative: 0.5,
    };
    let mut protected = PidController::new(PidSettings {
        gains,
        anti_windup: AntiWindup::BackCalculation {
            tracking_gain: 10.0,
        },
        ..PidSettings::default()
    })
    .expect("valid settings");
    let mut plain = PidController::new(PidSettings {
        gains,
        anti_windup: AntiWindup::None,
        ..PidSettings::default()
    })
    .expect("valid settings");

    for step in 0..100_i32 {
        let reference = f64::from(step % 7) - 3.0;
        let left = protected.step(0.25, reference, 0.02).expect("valid step");
        let right = plain.step(0.25, reference, 0.02).expect("valid step");
        assert!((left - right).abs() < 1e-12, "{left} against {right}");
    }
    assert_eq!(protected.saturated_steps(), 0);
}

#[test]
fn conditional_integration_is_available_and_holds_the_integrator() {
    // Offered because it is asked for. Switching the integrator off is an
    // unanalyzed nonlinearity and the loop no longer has the margins its
    // linear design claims, which is why it is not the default.
    let mut controller = PidController::new(PidSettings {
        gains: PidGains {
            proportional: 1.0,
            integral: 1.0,
            derivative: 0.0,
        },
        output_limits: (-1.0, 1.0),
        anti_windup: AntiWindup::ConditionalIntegration,
        ..PidSettings::default()
    })
    .expect("valid settings");

    for _ in 0..5 {
        controller.step(0.0, 10.0, 0.1).expect("valid step");
    }
    let held = controller.integral();
    for _ in 0..50 {
        controller.step(0.0, 10.0, 0.1).expect("valid step");
    }
    assert!(
        (controller.integral() - held).abs() < 1e-9,
        "the integrator moved from {held} to {} while saturated",
        controller.integral()
    );
}

#[test]
fn back_calculation_parks_the_integrator_where_the_output_meets_the_limit() {
    // The steady state is not "stopped": the correction pulls the
    // integrator to exactly the value that puts the unsaturated output on
    // the limit, so the moment the error relents the controller comes off
    // the limit immediately rather than unwinding first. With a
    // proportional term of 1 against a constant error of 5 and a ceiling
    // of 0.5, that value is 0.5 - 5, reached after the first step.
    let mut controller = PidController::new(PidSettings {
        gains: PidGains {
            proportional: 1.0,
            integral: 1.0,
            derivative: 0.0,
        },
        output_limits: (-0.5, 0.5),
        anti_windup: AntiWindup::BackCalculation { tracking_gain: 1.0 },
        ..PidSettings::default()
    })
    .expect("valid settings");

    for _ in 0..40 {
        controller.step(0.0, 5.0, 0.1).expect("valid step");
    }
    assert!(
        controller.integral().abs() < 1e-9,
        "the integrator parked at {} rather than at zero",
        controller.integral()
    );
    assert!(controller.saturated_steps() >= 40);
}

#[test]
fn resetting_a_controller_clears_everything_it_accumulated() {
    let mut controller = PidController::new(PidSettings {
        gains: PidGains {
            proportional: 1.0,
            integral: 1.0,
            derivative: 1.0,
        },
        output_limits: (-0.5, 0.5),
        anti_windup: AntiWindup::None,
        ..PidSettings::default()
    })
    .expect("valid settings");
    for _ in 0..20 {
        controller.step(0.0, 5.0, 0.1).expect("valid step");
    }
    assert!(controller.saturated_steps() > 0);
    assert!(controller.integral().abs() > 0.0);

    controller.reset();
    assert_eq!(controller.saturated_steps(), 0);
    assert!(controller.integral().abs() < 1e-15);

    // And the derivative starts fresh: the first step after a reset takes
    // no derivative, the same as the first step after construction.
    let first = controller.step(0.0, 5.0, 0.1).expect("valid step");
    assert!((first - 0.5).abs() < 1e-12, "{first}");
}

#[test]
fn a_degenerate_controller_is_rejected_at_construction() {
    assert!(matches!(
        PidController::new(PidSettings {
            gains: PidGains {
                proportional: f64::NAN,
                integral: 0.0,
                derivative: 0.0
            },
            ..PidSettings::default()
        }),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        PidController::new(PidSettings {
            output_limits: (1.0, -1.0),
            ..PidSettings::default()
        }),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        PidController::new(PidSettings {
            anti_windup: AntiWindup::BackCalculation {
                tracking_gain: -1.0
            },
            ..PidSettings::default()
        }),
        Err(Error::OutOfRange { .. })
    ));
}

// ------------------------------------------------------ pure pursuit ----

#[test]
fn a_vehicle_on_the_path_and_pointing_along_it_goes_straight() {
    let mut tracker = PurePursuitTracker::new(2.0).expect("a valid lookahead");
    let path = straight_path(10);
    let command = tracker
        .track(Pose::new(0.0, 0.0, 0.0).expect("finite"), &path, 1.5)
        .expect("valid step");

    assert!((command.speed - 1.5).abs() < 1e-12, "speed was altered");
    assert!(command.turn_rate.abs() < 1e-9, "{}", command.turn_rate);
    let errors = tracker.errors();
    assert!(errors.cross_track.abs() < 1e-12);
    assert!(errors.heading.abs() < 1e-12);
}

#[test]
fn a_vehicle_left_of_the_path_is_steered_back_to_the_right() {
    // Sign conventions are where tracking controllers go wrong silently,
    // so both the reported error and the commanded turn are pinned.
    let mut tracker = PurePursuitTracker::new(2.0).expect("a valid lookahead");
    let path = straight_path(10);
    let command = tracker
        .track(Pose::new(3.0, 0.5, 0.0).expect("finite"), &path, 1.0)
        .expect("valid step");

    assert!(
        tracker.errors().cross_track > 0.0,
        "left of the path should read positive, got {}",
        tracker.errors().cross_track
    );
    assert!(
        command.turn_rate < 0.0,
        "a vehicle left of the path should turn right, got {}",
        command.turn_rate
    );
}

#[test]
fn a_vehicle_right_of_the_path_is_steered_back_to_the_left() {
    let mut tracker = PurePursuitTracker::new(2.0).expect("a valid lookahead");
    let path = straight_path(10);
    let command = tracker
        .track(Pose::new(3.0, -0.5, 0.0).expect("finite"), &path, 1.0)
        .expect("valid step");
    assert!(tracker.errors().cross_track < 0.0);
    assert!(command.turn_rate > 0.0);
}

#[test]
fn the_turn_rate_scales_with_speed_at_fixed_geometry() {
    // Pure pursuit commands a curvature; the turn rate is that curvature
    // times the speed. Doubling the speed at the same pose has to double
    // the turn rate, or the vehicle tracks a different arc at every speed.
    let mut tracker = PurePursuitTracker::new(2.0).expect("a valid lookahead");
    let path = straight_path(10);
    let pose = Pose::new(1.0, 0.7, 0.2).expect("finite");

    let slow = tracker.track(pose, &path, 1.0).expect("valid step");
    let curvature = tracker.errors().curvature;
    let fast = tracker.track(pose, &path, 2.0).expect("valid step");

    assert!((tracker.errors().curvature - curvature).abs() < 1e-12);
    assert!((fast.turn_rate - 2.0 * slow.turn_rate).abs() < 1e-12);
}

#[test]
fn a_path_too_short_to_track_is_rejected() {
    let mut tracker = PurePursuitTracker::new(1.0).expect("a valid lookahead");
    let pose = Pose::new(0.0, 0.0, 0.0).expect("finite");
    assert!(matches!(
        tracker.track(pose, &[], 1.0),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        tracker.track(pose, &[(0.0, 0.0)], 1.0),
        Err(Error::TooFew { .. })
    ));
}

#[test]
fn a_non_finite_input_is_rejected_rather_than_tracked() {
    let mut tracker = PurePursuitTracker::new(1.0).expect("a valid lookahead");
    let pose = Pose::new(0.0, 0.0, 0.0).expect("finite");
    assert!(matches!(
        tracker.track(pose, &straight_path(4), f64::NAN),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        tracker.track(pose, &[(0.0, 0.0), (f64::NAN, 1.0)], 1.0),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn a_lookahead_that_would_divide_by_zero_is_rejected() {
    for distance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                PurePursuitTracker::new(distance),
                Err(Error::OutOfRange { .. })
            ),
            "accepted a lookahead of {distance}"
        );
    }
}

#[test]
fn a_repeated_waypoint_has_no_heading_and_reports_none() {
    // Two identical waypoints have no tangent, and deriving one from the
    // rounding noise between them would put an arbitrary heading error in
    // the log for a caller to chase.
    let mut tracker = PurePursuitTracker::new(1.0).expect("a valid lookahead");
    let path = [(0.0, 0.0), (0.0, 0.0), (5.0, 0.0)];
    tracker
        .track(Pose::new(0.0, 0.0, 1.0).expect("finite"), &path, 1.0)
        .expect("valid step");
    assert!(tracker.errors().heading.abs() < 1e-12);
    assert!(tracker.errors().cross_track.abs() < 1e-12);
}

// ------------------------------------------------------- rigid body ----

#[test]
fn a_disk_and_a_square_carry_the_inertia_their_shape_implies() {
    let disk = CircleBody::new(2.0, 3.0, 0.0, 0.0, 0.0).expect("a valid disk");
    assert!((disk.inertia() - 2.0 * 9.0 / 2.0).abs() < 1e-12);
    assert!((disk.bounding_radius() - 3.0).abs() < 1e-12);

    let square = SquareBody::new(2.0, 3.0, 0.0, 0.0, 0.0).expect("a valid square");
    assert!((square.inertia() - 2.0 * 9.0 / 6.0).abs() < 1e-12);
    assert!((square.bounding_radius() - 3.0 * core::f64::consts::SQRT_2 / 2.0).abs() < 1e-12);
}

#[test]
fn a_wrench_accumulates_until_the_step_that_applies_it() {
    // Several contacts in one frame add up. The alternative, where the
    // last wrench wins, silently drops every contact but one.
    let mut body = CircleBody::new(1.0, 1.0, 0.0, 0.0, 0.0).expect("a valid disk");
    body.apply_wrench(1.0, 0.0, 0.0).expect("finite");
    body.apply_wrench(1.0, 0.0, 0.0).expect("finite");
    body.step(1.0).expect("valid step");
    assert!((body.state().velocity()[0] - 2.0).abs() < 1e-12);

    // And the accumulator is cleared, so the next step coasts.
    body.step(1.0).expect("valid step");
    assert!((body.state().velocity()[0] - 2.0).abs() < 1e-12);
    assert!((body.state().pose()[0] - 4.0).abs() < 1e-12);
}

#[test]
fn a_heading_is_not_wrapped_so_a_spin_stays_countable() {
    // A body that has turned three times is a different thing from one
    // that has turned once, and a wrapped angle loses that.
    let mut body = CircleBody::new(1.0, 1.0, 0.0, 0.0, 0.0).expect("a valid disk");
    body.apply_wrench(0.0, 0.0, 1.0).expect("finite");
    body.step(1.0).expect("valid step");
    for _ in 0..50 {
        body.step(1.0).expect("valid step");
    }
    assert!(
        body.state().pose()[2] > core::f64::consts::TAU,
        "heading came back wrapped: {}",
        body.state().pose()[2]
    );
}

#[test]
fn a_square_reports_its_corners_in_the_world_frame() {
    let mut body = SquareBody::new(1.0, 2.0, 5.0, -3.0, 0.0).expect("a valid square");
    let corners = body.corners();
    assert!((corners[0].0 - 4.0).abs() < 1e-12 && (corners[0].1 - -4.0).abs() < 1e-12);
    assert!((corners[2].0 - 6.0).abs() < 1e-12 && (corners[2].1 - -2.0).abs() < 1e-12);

    // A quarter turn maps the near-left corner onto the near-right one.
    body.reset(0.0, 0.0, core::f64::consts::FRAC_PI_2)
        .expect("finite");
    let turned = body.corners();
    assert!((turned[0].0 - 1.0).abs() < 1e-12, "{:?}", turned[0]);
    assert!((turned[0].1 - -1.0).abs() < 1e-12, "{:?}", turned[0]);
}

#[test]
fn resetting_a_body_stops_it_where_it_was_put() {
    let mut body = CircleBody::new(1.0, 1.0, 0.0, 0.0, 0.0).expect("a valid disk");
    body.apply_wrench(5.0, 5.0, 5.0).expect("finite");
    body.step(0.1).expect("valid step");
    body.reset(1.0, 2.0, 3.0).expect("finite");
    assert!(close(body.state().pose(), [1.0, 2.0, 3.0]));
    assert!(close(body.state().velocity(), [0.0; 3]));

    // The pending wrench is cleared too, so the next step does not apply
    // a force the caller thought it had thrown away.
    body.step(0.1).expect("valid step");
    assert!(close(body.state().velocity(), [0.0; 3]));
}

#[test]
fn a_degenerate_body_is_rejected_at_construction() {
    for mass in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            CircleBody::new(mass, 1.0, 0.0, 0.0, 0.0),
            Err(Error::OutOfRange { .. })
        ));
    }
    assert!(matches!(
        CircleBody::new(1.0, 0.0, 0.0, 0.0, 0.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        SquareBody::new(1.0, -2.0, 0.0, 0.0, 0.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        CircleBody::new(1.0, 1.0, f64::NAN, 0.0, 0.0),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn a_non_finite_wrench_is_refused_before_it_reaches_the_pose() {
    let mut body = CircleBody::new(1.0, 1.0, 0.0, 0.0, 0.0).expect("a valid disk");
    assert!(matches!(
        body.apply_wrench(f64::NAN, 0.0, 0.0),
        Err(Error::NotFinite { .. })
    ));
    body.step(0.1).expect("valid step");
    assert!(close(body.state().pose(), [0.0; 3]));
}

// -------------------------------------------------------- avoidance ----

fn obstacle_at(x: f64, y: f64) -> KdTreeOccupancy {
    KdTreeOccupancy::new(&[vec![x, y]], 1.0).expect("a valid field")
}

#[test]
fn an_obstacle_to_the_left_turns_the_vehicle_right() {
    let field = ArtificialPotentialField::new(obstacle_at(0.0, 1.2), 1.0).expect("a valid gain");
    let bias = field
        .turn_rate_bias(Pose::new(0.0, 0.0, 0.0).expect("finite"))
        .expect("a finite bias");
    assert!(bias < 0.0, "expected a right turn, got {bias}");
}

#[test]
fn an_obstacle_to_the_right_turns_the_vehicle_left() {
    let field = ArtificialPotentialField::new(obstacle_at(0.0, -1.2), 1.0).expect("a valid gain");
    let bias = field
        .turn_rate_bias(Pose::new(0.0, 0.0, 0.0).expect("finite"))
        .expect("a finite bias");
    assert!(bias > 0.0, "expected a left turn, got {bias}");
}

#[test]
fn the_bias_falls_to_nothing_at_the_edge_of_the_influence_radius() {
    // Twice the clearance, and the magnitude reaches zero there rather
    // than stepping off, so the correction switches on smoothly.
    let field = ArtificialPotentialField::new(obstacle_at(0.0, 2.0), 1.0).expect("a valid gain");
    let at_edge = field
        .turn_rate_bias(Pose::new(0.0, 0.0, 0.0).expect("finite"))
        .expect("a finite bias");
    assert!(at_edge.abs() < 1e-12, "{at_edge}");

    let inside = ArtificialPotentialField::new(obstacle_at(0.0, 1.5), 1.0)
        .expect("a valid gain")
        .turn_rate_bias(Pose::new(0.0, 0.0, 0.0).expect("finite"))
        .expect("a finite bias");
    assert!(inside.abs() > 0.0);
    assert!(inside.abs() < 10.0, "the bias ran away: {inside}");
}

#[test]
fn the_bias_grows_as_the_obstacle_gets_closer() {
    let mut previous = 0.0;
    for offset in [1.9, 1.6, 1.3, 1.05] {
        let bias = ArtificialPotentialField::new(obstacle_at(0.0, offset), 1.0)
            .expect("a valid gain")
            .turn_rate_bias(Pose::new(0.0, 0.0, 0.0).expect("finite"))
            .expect("a finite bias")
            .abs();
        assert!(
            bias > previous,
            "at {offset} the bias {bias} did not exceed {previous}"
        );
        previous = bias;
    }
}

#[test]
fn a_disabled_field_never_biases_anything() {
    let pose = Pose::new(0.0, 0.0, 0.0).expect("finite");
    let disabled: ArtificialPotentialField<KdTreeOccupancy> = ArtificialPotentialField::disabled();
    assert!(disabled.turn_rate_bias(pose).expect("a finite bias").abs() < 1e-15);

    let zero_gain =
        ArtificialPotentialField::new(obstacle_at(0.0, 1.05), 0.0).expect("a valid gain");
    assert!(zero_gain.turn_rate_bias(pose).expect("finite").abs() < 1e-15);
}

// ----------------------------------------------------- tracking loop ----

fn driving(settings: TrackingSettings) -> TrackingLoop<Unicycle, PurePursuitTracker, NoAvoidance> {
    TrackingLoop::new(
        Unicycle::at(0.0, 0.0, 0.0),
        PurePursuitTracker::new(1.5).expect("a valid lookahead"),
        NoAvoidance,
        settings,
    )
    .expect("consistent settings")
}

#[test]
fn a_vehicle_started_off_the_path_converges_onto_it() {
    let path = straight_path(40);
    let mut loop_ = TrackingLoop::new(
        Unicycle::at(0.0, 1.0, 0.0),
        PurePursuitTracker::new(1.5).expect("a valid lookahead"),
        NoAvoidance,
        TrackingSettings {
            cruise_speed: 1.0,
            ..TrackingSettings::default()
        },
    )
    .expect("consistent settings");

    let first = loop_.step(&path, 0.05).expect("valid step");
    let last = loop_
        .run(&path, 200, 0.05)
        .expect("valid steps")
        .expect("two hundred steps produce a sample");
    assert!(
        last.cross_track_error.abs() < first.cross_track_error.abs() / 4.0,
        "the error went from {} to {}",
        first.cross_track_error,
        last.cross_track_error
    );
}

#[test]
fn the_curvature_gain_slows_the_vehicle_on_a_curve() {
    // Speed is chosen from the curvature the tracker reported last step,
    // because the current one is not known until after the speed is fixed.
    let path: Vec<(f64, f64)> = (0..40)
        .map(|index| {
            let angle = f64::from(index) * 0.15;
            (10.0 * angle.sin(), 10.0 - 10.0 * angle.cos())
        })
        .collect();

    let mut fast = driving(TrackingSettings {
        cruise_speed: 2.0,
        curvature_gain: 0.0,
        ..TrackingSettings::default()
    });
    let mut slowed = driving(TrackingSettings {
        cruise_speed: 2.0,
        curvature_gain: 5.0,
        ..TrackingSettings::default()
    });

    let fast_sample = fast
        .run(&path, 30, 0.05)
        .expect("valid steps")
        .expect("a sample");
    let slow_sample = slowed
        .run(&path, 30, 0.05)
        .expect("valid steps")
        .expect("a sample");
    assert!(
        slow_sample.speed < fast_sample.speed,
        "the curvature gain did not slow anything: {} against {}",
        slow_sample.speed,
        fast_sample.speed
    );
}

#[test]
fn an_avoidance_bias_reaches_the_command_and_is_reported() {
    let path = straight_path(20);
    let mut loop_ = TrackingLoop::new(
        Unicycle::at(0.0, 0.0, 0.0),
        PurePursuitTracker::new(1.5).expect("a valid lookahead"),
        FixedBias(0.3),
        TrackingSettings::default(),
    )
    .expect("consistent settings");

    let sample = loop_.step(&path, 0.05).expect("valid step");
    assert!((sample.avoidance_bias - 0.3).abs() < 1e-12);
    assert!(
        (sample.requested.turn_rate - sample.curvature * sample.requested.speed - 0.3).abs()
            < 1e-12,
        "the bias did not reach the command"
    );
}

#[test]
fn the_limits_apply_after_the_avoidance_bias_not_before() {
    // Limiting first would let the bias push the command back outside the
    // box, which is how an avoidance term ends up asking for a turn rate
    // no actuator can produce.
    let path = straight_path(20);
    let mut loop_ = TrackingLoop::new(
        Unicycle::at(0.0, 0.0, 0.0),
        PurePursuitTracker::new(1.5).expect("a valid lookahead"),
        FixedBias(50.0),
        TrackingSettings {
            limits: CommandLimits {
                max_turn_rate: 0.4,
                ..CommandLimits::default()
            },
            ..TrackingSettings::default()
        },
    )
    .expect("consistent settings");

    let sample = loop_.step(&path, 0.05).expect("valid step");
    assert!(sample.requested.turn_rate > 0.4);
    assert!(sample.applied.turn_rate <= 0.4 + 1e-12);
    assert_eq!(loop_.saturation().magnitude_steps, 1);
}

#[test]
fn a_bounded_history_keeps_the_most_recent_samples() {
    let path = straight_path(20);
    let mut loop_ = driving(TrackingSettings {
        history_capacity: Some(5),
        ..TrackingSettings::default()
    });
    loop_.run(&path, 50, 0.05).expect("valid steps");
    assert_eq!(loop_.history().count(), 5);

    let last = loop_.last().copied().expect("a sample");
    let newest = loop_.history().last().copied().expect("a sample");
    assert!((last.pose.x() - newest.pose.x()).abs() < 1e-15);
}

#[test]
fn an_unbounded_history_keeps_everything_as_the_python_loop_did() {
    let path = straight_path(20);
    let mut loop_ = driving(TrackingSettings::default());
    loop_.run(&path, 37, 0.05).expect("valid steps");
    assert_eq!(loop_.history().count(), 37);

    loop_.reset();
    assert_eq!(loop_.history().count(), 0);
    assert!(!loop_.saturation().saturated());
}

#[test]
fn running_zero_steps_reports_nothing_rather_than_failing() {
    let path = straight_path(20);
    let mut loop_ = driving(TrackingSettings::default());
    assert!(loop_.run(&path, 0, 0.05).expect("valid steps").is_none());
}

#[test]
fn a_degenerate_loop_is_rejected_at_construction() {
    assert!(matches!(
        TrackingLoop::new(
            Unicycle::at(0.0, 0.0, 0.0),
            PurePursuitTracker::new(1.0).expect("a valid lookahead"),
            NoAvoidance,
            TrackingSettings {
                curvature_gain: -1.0,
                ..TrackingSettings::default()
            },
        ),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        TrackingLoop::new(
            Unicycle::at(0.0, 0.0, 0.0),
            PurePursuitTracker::new(1.0).expect("a valid lookahead"),
            NoAvoidance,
            TrackingSettings {
                cruise_speed: f64::NAN,
                ..TrackingSettings::default()
            },
        ),
        Err(Error::NotFinite { .. })
    ));
}
