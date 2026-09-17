//! Soft-barrier cost helpers shared by the two predictive controllers.
//!
//! Ports the float path of the Python `arco.control.mpc.costs` module.
//! The Python source also carried a symbolic `CasADi` branch, used while
//! path-following and joint-space MPC built a nonlinear program for
//! IPOPT to solve; ADR-002 replaced that program with a sequence of
//! convex programs, so nothing in this crate ever holds a symbolic
//! value and that branch disappears with it.
//!
//! [`forward_cone_factor`] is a different function from the private
//! `forward_cone` in [`crate::mpc::path_following`]. That one feeds the
//! convexified barrier deviation A-30 records: it uses `hypot`, floors
//! the separation at `SEPARATION_FLOOR` instead of an additive epsilon
//! under the square root, and clamps to `[0, 1]` instead of only flooring
//! at zero. The two agree away from the floor, and a reader comparing
//! them side by side is not looking at a bug.

/// Soft clearance barrier with forward-cone weighting.
///
/// Penalizes penetration of the clearance margin with a power barrier,
/// then weights the penalty by directional relevance along the velocity
/// cone. Joint-space MPC has no heading and passes `cone_factor = 1.0` so
/// the directional term is exactly `1`.
///
/// This mirrors the float path of the Python `obstacle_barrier`: it takes
/// `f64` in and returns `f64`, never raises, and propagates a NaN input
/// through to the result rather than rejecting it.
///
/// # Arguments
///
/// * `distance` - Distance to the nearest obstacle, meters.
/// * `clearance` - Required clearance radius, meters. Must be positive at
///   the call site; a value at or below the `1e-6` floor used for the
///   division still produces a finite result rather than a divide by
///   zero.
/// * `weight` - Barrier weight.
/// * `power` - Barrier exponent.
/// * `cone_factor` - Forward-cone factor in `[0, 1]` (1 ahead, 0 behind).
///   Pass `1.0` where no cone weighting applies.
///
/// # Returns
///
/// The scalar barrier cost.
#[must_use]
pub fn obstacle_barrier(
    distance: f64,
    clearance: f64,
    weight: f64,
    power: f64,
    cone_factor: f64,
) -> f64 {
    let denom = clearance.max(1e-6);
    let penetration = ((clearance - distance) / denom).max(0.0);
    let directional = 0.2 + 0.8 * cone_factor;
    weight * penetration.powf(power) * directional
}

/// Smooth forward-cone factor via heading-bearing projection.
///
/// Projects the unit obstacle offset onto the vehicle heading, floored at
/// zero. Numerically equal to `max(0, cos(heading - bearing))` for a
/// nonzero separation between the pose and the obstacle.
///
/// This mirrors the float path of the Python `forward_cone_factor`: it
/// takes `f64` in and returns `f64`, never raises, and propagates a NaN
/// input through to the result rather than rejecting it.
///
/// # Arguments
///
/// * `pose_x` - Vehicle x position, meters.
/// * `pose_y` - Vehicle y position, meters.
/// * `heading` - Vehicle heading, radians.
/// * `obstacle_x` - Obstacle x position, meters.
/// * `obstacle_y` - Obstacle y position, meters.
///
/// # Returns
///
/// The cone factor in `[0, 1]`.
#[must_use]
pub fn forward_cone_factor(
    pose_x: f64,
    pose_y: f64,
    heading: f64,
    obstacle_x: f64,
    obstacle_y: f64,
) -> f64 {
    let dx = obstacle_x - pose_x;
    let dy = obstacle_y - pose_y;
    let distance = (dx * dx + dy * dy + 1e-9).sqrt();
    let forward = (heading.cos() * dx + heading.sin() * dy) / distance;
    forward.max(0.0)
}
