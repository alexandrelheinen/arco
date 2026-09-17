//! Integration tests for the shared MPC soft-barrier cost helpers.
//!
//! Ports the numeric expectations from `tests/control/mpc/test_costs.py`,
//! against the crate's public API only.

use arco_control::mpc::costs::{forward_cone_factor, obstacle_barrier};

/// A cone factor of one leaves the directional weight at exactly one.
#[test]
fn cone_factor_one_is_identity_directional() {
    let distance = 0.25;
    let clearance = 1.0;
    let weight = 8.0;
    let power = 4.0;

    let got = obstacle_barrier(distance, clearance, weight, power, 1.0);

    let denom = clearance.max(1e-6);
    let penetration = ((clearance - distance) / denom).max(0.0);
    let expected = weight * penetration.powf(power);
    assert!((got - expected).abs() <= 1e-12);
}

/// A distance at or beyond the clearance scores zero.
#[test]
fn barrier_scores_zero_outside_clearance() {
    let got = obstacle_barrier(2.0, 1.0, 10.0, 4.0, 1.0);
    assert!(got.abs() <= 1e-12);

    let at_clearance = obstacle_barrier(1.0, 1.0, 10.0, 4.0, 1.0);
    assert!(at_clearance.abs() <= 1e-12);
}

/// A clearance below the epsilon floor does not divide by zero.
#[test]
fn barrier_does_not_divide_by_zero_below_the_floor() {
    let got = obstacle_barrier(0.0, 0.0, 5.0, 2.0, 0.5);
    assert!(got.is_finite());

    // clearance = 0 clamps the denominator to 1e-6, so penetration is
    // (0 - 0) / 1e-6 = 0, and the barrier itself is zero.
    assert!(got.abs() <= 1e-12);
}

/// The barrier matches the float reference across a spread of inputs.
#[test]
fn barrier_matches_the_float_reference() {
    let cases: [(f64, f64, f64, f64, f64); 5] = [
        (0.5, 1.0, 10.0, 4.0, 1.0),
        (0.2, 1.0, 10.0, 4.0, 0.0),
        (0.0, 0.5, 5.0, 2.0, 0.5),
        (2.0, 1.0, 10.0, 4.0, 1.0),
        (0.8, 1.0, 1.0, 4.0, 1.0),
    ];

    for (distance, clearance, weight, power, cone_factor) in cases {
        let denom = clearance.max(1e-6);
        let penetration = ((clearance - distance) / denom).max(0.0);
        let directional = 0.2 + 0.8 * cone_factor;
        let expected = weight * penetration.powf(power) * directional;

        let got = obstacle_barrier(distance, clearance, weight, power, cone_factor);
        assert!((got - expected).abs() <= 1e-12);
    }
}

/// An obstacle straight ahead scores a cone factor of one.
#[test]
fn cone_factor_is_one_straight_ahead() {
    let got = forward_cone_factor(0.0, 0.0, 0.0, 1.0, 0.0);
    assert!((got - 1.0).abs() <= 1e-9);
}

/// An obstacle behind the vehicle scores zero.
#[test]
fn cone_factor_is_zero_behind_the_vehicle() {
    let got = forward_cone_factor(0.0, 0.0, 0.0, -1.0, 0.0);
    assert!(got.abs() <= 1e-9);
}

/// An obstacle at ninety degrees to the heading scores zero.
#[test]
fn cone_factor_is_zero_perpendicular_to_the_heading() {
    let got = forward_cone_factor(0.0, 0.0, 0.0, 0.0, 1.0);
    assert!(got.abs() <= 1e-9);
}

/// The cone factor equals `max(0, cos(heading - bearing))` away from the
/// pose, for a nonzero separation between the pose and the obstacle.
#[test]
fn cone_factor_equals_the_clamped_cosine_of_the_bearing_error() {
    let cases = [
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 2.0, std::f64::consts::FRAC_PI_4, 2.0, 3.0),
    ];

    for (pose_x, pose_y, heading, obstacle_x, obstacle_y) in cases {
        let got = forward_cone_factor(pose_x, pose_y, heading, obstacle_x, obstacle_y);

        let bearing = (obstacle_y - pose_y).atan2(obstacle_x - pose_x);
        let expected = (heading - bearing).cos().max(0.0);
        assert!((got - expected).abs() <= 1e-9);
    }
}
