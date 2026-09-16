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

//! What the vehicle does with a command, and what it refuses.
//!
//! The kinematics are checked against closed forms rather than against
//! recorded numbers: a straight run covers speed times time, and a whole
//! turn of heading closes the circle exactly, because the displacement
//! vectors of a full turn sum to zero whatever the step size was.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use core::f64::consts::{PI, TAU};

use arco_control::limits::{CommandLimits, IntervalBand};
use arco_control::pursuit::PurePursuitTracker;
use arco_control::tracking::{TrackingLoop, TrackingSettings};
use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::numeric::angles_close;
use arco_core::protocols::{AvoidanceStrategy, Command, VehicleModel};
use arco_guidance::vehicle::DubinsVehicle;

/// Limits that never bite, so a test can pin the kinematics alone.
fn unhindered() -> CommandLimits {
    CommandLimits {
        max_speed: f64::INFINITY,
        min_speed: f64::NEG_INFINITY,
        max_turn_rate: f64::INFINITY,
        max_speed_rate: f64::INFINITY,
        max_turn_rate_change: f64::INFINITY,
        interval: IntervalBand::default(),
    }
}

/// A vehicle at the origin facing along the first axis.
fn free_vehicle() -> DubinsVehicle {
    DubinsVehicle::new(0.0, 0.0, 0.0, unhindered()).expect("unhindered limits are valid")
}

/// A command, spelled out where a test needs one inline.
const fn drive(speed: f64, turn_rate: f64) -> Command {
    Command { speed, turn_rate }
}

/// Avoidance that never biases anything.
#[derive(Debug, Clone, Copy)]
struct NoAvoidance;

impl AvoidanceStrategy for NoAvoidance {
    fn turn_rate_bias(&self, _pose: Pose) -> Result<f64, Error> {
        Ok(0.0)
    }
}

/// A straight path along the first axis, one meter apart.
fn straight_path(count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|index| (f64::from(u32::try_from(index).unwrap_or(u32::MAX)), 0.0))
        .collect()
}

// -------------------------------------------------------- kinematics ----

#[test]
fn a_new_vehicle_is_at_rest_where_it_was_placed() {
    let vehicle = DubinsVehicle::new(1.0, 2.0, 0.5, DubinsVehicle::default_limits())
        .expect("the default limits are valid");
    assert!((vehicle.pose().x() - 1.0).abs() < 1e-15);
    assert!((vehicle.pose().y() - 2.0).abs() < 1e-15);
    assert!((vehicle.pose().heading() - 0.5).abs() < 1e-15);
    assert!(vehicle.speed().abs() < 1e-15);
    assert!(vehicle.turn_rate().abs() < 1e-15);
}

#[test]
fn a_straight_run_covers_speed_times_time() {
    let mut vehicle = free_vehicle();
    vehicle.step(drive(2.0, 0.0), 0.5).expect("valid step");
    assert!(
        (vehicle.pose().x() - 1.0).abs() < 1e-12,
        "{:?}",
        vehicle.pose()
    );
    assert!(vehicle.pose().y().abs() < 1e-12, "{:?}", vehicle.pose());
    assert!(vehicle.pose().heading().abs() < 1e-12);
}

#[test]
fn a_run_along_a_diagonal_heading_splits_the_distance_by_the_cosine() {
    let mut vehicle =
        DubinsVehicle::new(0.0, 0.0, PI / 3.0, unhindered()).expect("valid construction");
    vehicle.step(drive(4.0, 0.0), 0.25).expect("valid step");
    // One meter travelled at sixty degrees: half a meter east, and the
    // root of three quarters north.
    assert!(
        (vehicle.pose().x() - 0.5).abs() < 1e-12,
        "{:?}",
        vehicle.pose()
    );
    assert!(
        (vehicle.pose().y() - 0.75_f64.sqrt()).abs() < 1e-12,
        "{:?}",
        vehicle.pose()
    );
}

#[test]
fn a_whole_turn_of_heading_closes_the_circle() {
    // Euler integration of a constant turn walks a regular polygon, and
    // the displacement vectors of a whole turn are equally spaced unit
    // vectors, which sum to zero. So the closure is exact whatever the
    // step size, and any drift here is the integrator being wrong rather
    // than the step being coarse.
    let steps = 1000_u32;
    let dt = TAU / f64::from(steps);
    let mut vehicle = free_vehicle();
    for _ in 0..steps {
        vehicle.step(drive(1.0, 1.0), dt).expect("valid step");
    }
    let closure = vehicle.pose().x().hypot(vehicle.pose().y());
    assert!(closure < 1e-9, "the circle missed itself by {closure}");
    assert!(
        angles_close(vehicle.pose().heading(), 0.0),
        "{}",
        vehicle.pose().heading()
    );
}

#[test]
fn turning_one_way_and_back_again_restores_the_heading() {
    let mut vehicle = DubinsVehicle::new(0.0, 0.0, 0.3, unhindered()).expect("valid construction");
    for _ in 0..10 {
        vehicle.step(drive(1.0, 1.0), 0.1).expect("valid step");
    }
    assert!(vehicle.pose().heading() > 1.0);
    for _ in 0..10 {
        vehicle.step(drive(1.0, -1.0), 0.1).expect("valid step");
    }
    assert!(
        angles_close(vehicle.pose().heading(), 0.3),
        "{}",
        vehicle.pose().heading()
    );
}

#[test]
fn the_heading_stays_wrapped_across_many_turns() {
    // `FR-INV-15`: the pose carries a normalized heading, so a vehicle
    // that has spun eighty times reads the same as one that has spun
    // none, and a controller differencing against it cannot be surprised.
    let mut vehicle = DubinsVehicle::new(0.0, 0.0, 3.0, unhindered()).expect("valid construction");
    for _ in 0..500 {
        vehicle.step(drive(1.0, 5.0), 0.1).expect("valid step");
        let heading = vehicle.pose().heading();
        assert!(
            (-PI..PI).contains(&heading),
            "heading left the interval: {heading}"
        );
    }
}

// ------------------------------------------------------------ limits ----

#[test]
fn acceleration_bounds_how_far_the_speed_moves_in_one_step() {
    let mut vehicle = DubinsVehicle::new(
        0.0,
        0.0,
        0.0,
        CommandLimits {
            max_speed: 100.0,
            min_speed: 0.0,
            max_turn_rate: 100.0,
            max_speed_rate: 1.0,
            max_turn_rate_change: 1.0,
            ..DubinsVehicle::default_limits()
        },
    )
    .expect("valid construction");

    vehicle.step(drive(100.0, 100.0), 0.1).expect("valid step");
    assert!((vehicle.speed() - 0.1).abs() < 1e-12, "{}", vehicle.speed());
    assert!(
        (vehicle.turn_rate() - 0.1).abs() < 1e-12,
        "{}",
        vehicle.turn_rate()
    );

    // And the next step moves it by the same allowance again, so the
    // limiter is a rate rather than a one-off clamp at the start.
    vehicle.step(drive(100.0, 100.0), 0.1).expect("valid step");
    assert!((vehicle.speed() - 0.2).abs() < 1e-12, "{}", vehicle.speed());
}

#[test]
fn the_command_never_leaves_its_box_however_hard_it_is_pushed() {
    let limits = CommandLimits {
        max_speed: 3.0,
        min_speed: 0.0,
        max_turn_rate: 1.5,
        max_speed_rate: f64::INFINITY,
        max_turn_rate_change: f64::INFINITY,
        interval: IntervalBand::default(),
    };
    let mut vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, limits).expect("valid construction");
    for step in 0..200_i32 {
        // Alternate the sign so the lower bound is exercised as hard as
        // the upper one: a vehicle told to reverse must simply stop.
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        vehicle
            .step(drive(sign * 10.0, sign * 10.0), 0.05)
            .expect("valid step");
        assert!(
            (0.0..=3.0).contains(&vehicle.speed()),
            "{}",
            vehicle.speed()
        );
        assert!(vehicle.turn_rate().abs() <= 1.5, "{}", vehicle.turn_rate());
    }
}

#[test]
fn a_vehicle_told_to_reverse_stops_instead() {
    let mut vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, DubinsVehicle::default_limits())
        .expect("valid construction");
    for _ in 0..50 {
        vehicle.step(drive(-10.0, 0.0), 0.1).expect("valid step");
    }
    assert!(vehicle.speed().abs() < 1e-12, "{}", vehicle.speed());
    assert!(
        vehicle.pose().x().abs() < 1e-12,
        "the vehicle went backwards"
    );
}

#[test]
fn an_interval_outside_the_band_is_refused() {
    // Deviation A-17 and `FR-INV-10`. A step reads no clock, so the
    // interval is whatever the caller computed, and a negative one after a
    // clock adjustment integrates the vehicle backwards without saying so.
    let mut vehicle = free_vehicle();
    for dt in [0.0, -0.1, 2.0] {
        assert!(
            matches!(
                vehicle.step(drive(1.0, 0.0), dt),
                Err(Error::OutOfRange { .. })
            ),
            "an interval of {dt} was accepted"
        );
    }
    for dt in [f64::NAN, f64::INFINITY] {
        assert!(matches!(
            vehicle.step(drive(1.0, 0.0), dt),
            Err(Error::NotFinite { .. })
        ));
    }
    // And the refusal left the state alone rather than half-integrating it.
    assert!(vehicle.pose().x().abs() < 1e-15);
    assert!(vehicle.speed().abs() < 1e-15);
}

#[test]
fn a_widened_band_accepts_the_interval_the_default_one_refused() {
    let limits = CommandLimits {
        interval: IntervalBand::new(1e-9, 10.0).expect("a valid band"),
        ..unhindered()
    };
    let mut vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, limits).expect("valid construction");
    vehicle.step(drive(1.0, 0.0), 2.0).expect("valid step");
    assert!((vehicle.pose().x() - 2.0).abs() < 1e-12);
}

#[test]
fn a_command_that_is_not_a_real_number_is_refused() {
    let mut vehicle = free_vehicle();
    for command in [
        drive(f64::NAN, 0.0),
        drive(0.0, f64::NAN),
        drive(f64::INFINITY, 0.0),
    ] {
        assert!(matches!(
            vehicle.step(command, 0.1),
            Err(Error::NotFinite { .. })
        ));
    }
}

#[test]
fn limits_no_command_could_satisfy_are_refused_at_construction() {
    let inverted = CommandLimits {
        max_speed: 1.0,
        min_speed: 2.0,
        ..DubinsVehicle::default_limits()
    };
    assert!(matches!(
        DubinsVehicle::new(0.0, 0.0, 0.0, inverted),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        DubinsVehicle::new(f64::NAN, 0.0, 0.0, DubinsVehicle::default_limits()),
        Err(Error::NotFinite { .. })
    ));

    let mut vehicle = free_vehicle();
    assert!(matches!(
        vehicle.set_limits(inverted),
        Err(Error::OutOfRange { .. })
    ));
    assert!(
        vehicle.limits().max_speed.is_infinite(),
        "the limits changed anyway"
    );
}

#[test]
fn resetting_returns_the_vehicle_to_rest() {
    let mut vehicle = free_vehicle();
    vehicle.step(drive(2.0, 1.0), 0.5).expect("valid step");
    vehicle.reset(3.0, -4.0, 1.0).expect("a finite pose");

    assert!((vehicle.pose().x() - 3.0).abs() < 1e-15);
    assert!((vehicle.pose().y() + 4.0).abs() < 1e-15);
    assert!((vehicle.pose().heading() - 1.0).abs() < 1e-15);
    assert!(vehicle.speed().abs() < 1e-15);
    assert!(vehicle.turn_rate().abs() < 1e-15);
}

// ------------------------------------------------ inverse kinematics ----

#[test]
fn the_turn_rate_closes_the_heading_gap_in_the_time_allowed() {
    let vehicle = free_vehicle();
    let command = vehicle
        .inverse_kinematics(&[0.0, 0.0, 0.0], &[0.0, 1.0], 1.0, 2.0)
        .expect("a valid query");
    // A quarter turn to the left, spread over two seconds.
    assert!((command.turn_rate - PI / 4.0).abs() < 1e-12, "{command:?}");
    assert!((command.speed - 1.0).abs() < 1e-12, "{command:?}");
}

#[test]
fn a_heading_gap_across_the_branch_cut_takes_the_short_way() {
    // `FR-INV-15`. Facing just short of due west and asked to aim just
    // past it, the vehicle turns a fifth of a turn, not four fifths back
    // the other way, which is what plain subtraction would have asked for.
    let vehicle = DubinsVehicle::new(0.0, 0.0, 0.9 * PI, DubinsVehicle::default_limits())
        .expect("valid construction");
    let bearing = -0.9 * PI;
    let goal = [bearing.cos(), bearing.sin()];
    let command = vehicle
        .inverse_kinematics(&[0.0, 0.0, 0.9 * PI], &goal, 1.0, 1.0)
        .expect("a valid query");

    assert!(command.turn_rate > 0.0, "turned the long way: {command:?}");
    assert!((command.turn_rate - 0.2 * PI).abs() < 1e-9, "{command:?}");
}

#[test]
fn a_start_without_a_heading_is_taken_to_face_along_the_first_axis() {
    let vehicle = free_vehicle();
    let ahead = vehicle
        .inverse_kinematics(&[0.0, 0.0], &[1.0, 0.0], 1.0, 1.0)
        .expect("a valid query");
    assert!(ahead.turn_rate.abs() < 1e-12, "{ahead:?}");

    let left = vehicle
        .inverse_kinematics(&[0.0, 0.0], &[0.0, 1.0], 1.0, 1.0)
        .expect("a valid query");
    assert!((left.turn_rate - PI / 2.0).abs() < 1e-12, "{left:?}");
}

#[test]
fn the_commanded_speed_and_turn_rate_land_inside_the_limits() {
    let vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, DubinsVehicle::default_limits())
        .expect("valid construction");
    let limits = vehicle.limits();

    let fast = vehicle
        .inverse_kinematics(&[0.0, 0.0, 0.0], &[0.0, 1.0], 100.0, 0.001)
        .expect("a valid query");
    assert!((fast.speed - limits.max_speed).abs() < 1e-12, "{fast:?}");
    assert!(
        (fast.turn_rate - limits.max_turn_rate).abs() < 1e-12,
        "{fast:?}"
    );

    let backwards = vehicle
        .inverse_kinematics(&[0.0, 0.0, 0.0], &[0.0, -1.0], -100.0, 0.001)
        .expect("a valid query");
    assert!(
        (backwards.speed - limits.min_speed).abs() < 1e-12,
        "{backwards:?}"
    );
    assert!(
        (backwards.turn_rate + limits.max_turn_rate).abs() < 1e-12,
        "{backwards:?}"
    );
}

#[test]
fn an_inverse_kinematics_query_that_cannot_be_answered_is_refused() {
    let vehicle = free_vehicle();
    assert!(matches!(
        vehicle.inverse_kinematics(&[0.0], &[1.0, 1.0], 1.0, 1.0),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        vehicle.inverse_kinematics(&[0.0, f64::NAN], &[1.0, 1.0], 1.0, 1.0),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        vehicle.inverse_kinematics(&[0.0, 0.0], &[1.0, 1.0], f64::NAN, 1.0),
        Err(Error::NotFinite { .. })
    ));
    for duration in [0.0, -1.0, f64::INFINITY] {
        assert!(
            matches!(
                vehicle.inverse_kinematics(&[0.0, 0.0], &[1.0, 1.0], 1.0, duration),
                Err(Error::OutOfRange { .. })
            ),
            "a duration of {duration} was accepted"
        );
    }
}

// ----------------------------------------------------- feasible sets ----

#[test]
fn a_state_carrying_no_dynamics_is_feasible() {
    let vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, DubinsVehicle::default_limits())
        .expect("valid construction");
    for state in [vec![3.0, 4.0], vec![0.0, 0.0, 1.57]] {
        assert!(
            vehicle.is_feasible(&state).expect("a valid state"),
            "{state:?}"
        );
    }
}

#[test]
fn a_speed_outside_the_band_is_infeasible_whatever_else_the_state_says() {
    let vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, DubinsVehicle::default_limits())
        .expect("valid construction");
    assert!(vehicle.is_feasible(&[0.0, 0.0, 0.0, 4.0]).expect("valid"));
    assert!(!vehicle.is_feasible(&[0.0, 0.0, 0.0, 6.0]).expect("valid"));
    assert!(!vehicle.is_feasible(&[0.0, 0.0, 0.0, -1.0]).expect("valid"));
    // The bound is inclusive: a vehicle at exactly its ceiling is in a
    // state it can hold, and rejecting it would make the ceiling
    // unreachable rather than maximal.
    assert!(vehicle.is_feasible(&[0.0, 0.0, 0.0, 5.0]).expect("valid"));
}

#[test]
fn a_turn_rate_past_the_bound_is_infeasible() {
    let vehicle = DubinsVehicle::new(0.0, 0.0, 0.0, DubinsVehicle::default_limits())
        .expect("valid construction");
    assert!(
        vehicle
            .is_feasible(&[0.0, 0.0, 0.0, 2.0, 0.9])
            .expect("valid")
    );
    assert!(
        vehicle
            .is_feasible(&[0.0, 0.0, 0.0, 2.0, -1.0])
            .expect("valid")
    );
    assert!(
        !vehicle
            .is_feasible(&[0.0, 0.0, 0.0, 2.0, 1.1])
            .expect("valid")
    );
    assert!(
        !vehicle
            .is_feasible(&[0.0, 0.0, 0.0, 2.0, -1.1])
            .expect("valid")
    );
}

#[test]
fn a_state_that_is_not_a_finite_position_is_refused_rather_than_accepted() {
    // A NaN compares false against both bounds, so an unchecked state
    // would be reported feasible: the one answer that cannot be right.
    let vehicle = free_vehicle();
    assert!(matches!(
        vehicle.is_feasible(&[0.0]),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        vehicle.is_feasible(&[0.0, 0.0, 0.0, f64::NAN, 0.0]),
        Err(Error::NotFinite { .. })
    ));
}

// ------------------------------------------------------ control loop ----

#[test]
fn the_vehicle_drives_a_tracking_loop_onto_the_path() {
    // The point of the phase: `DubinsVehicle` satisfies the trait
    // `arco-control` declares, so the loop drives it without knowing what
    // it is, and the import cycle the Python carried under `TYPE_CHECKING`
    // has nowhere left to form.
    let vehicle = DubinsVehicle::new(0.0, 1.0, 0.0, DubinsVehicle::default_limits())
        .expect("valid construction");
    let tracker = PurePursuitTracker::new(2.0).expect("a valid lookahead");
    let mut driver = TrackingLoop::new(
        vehicle,
        tracker,
        NoAvoidance,
        TrackingSettings {
            cruise_speed: 1.0,
            limits: DubinsVehicle::default_limits(),
            ..TrackingSettings::default()
        },
    )
    .expect("valid settings");

    let path = straight_path(40);
    let last = driver
        .run(&path, 300, 0.05)
        .expect("valid steps")
        .expect("three hundred steps is not zero steps");

    assert!(
        last.cross_track_error.abs() < 0.05,
        "a meter off the path became {}",
        last.cross_track_error
    );
    assert!(
        last.pose.x() > 5.0,
        "the vehicle barely moved: {:?}",
        last.pose
    );
}
