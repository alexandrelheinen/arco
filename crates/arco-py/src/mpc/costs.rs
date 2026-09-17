//! `arco.control.mpc.costs` as Python still sees it.
//!
//! Binds the two shared soft-barrier cost helpers straight through to
//! [`arco_control::mpc::costs`]. Both take and return plain floats and
//! never raise, matching the pure-Python functions they replace, which
//! never raised either and let a NaN propagate rather than rejecting it.

use arco_control::mpc::costs;
use pyo3::prelude::*;

/// Soft clearance barrier with forward-cone weighting.
///
/// Penalizes penetration of the clearance margin with a power barrier,
/// then weights the penalty by directional relevance along the velocity
/// cone. Joint-space MPC has no heading and passes ``cone_factor=1.0`` so
/// the directional term is exactly ``1``.
///
/// Args:
///     distance: Distance to the nearest obstacle, meters.
///     clearance: Required clearance radius, meters. Must be positive at
///         the call site.
///     weight: Barrier weight.
///     power: Barrier exponent.
///     `cone_factor`: Forward-cone factor in ``[0, 1]`` (1 ahead, 0
///         behind). Use ``1.0`` where no cone weighting applies.
///
/// Returns:
///     The scalar barrier cost.
#[pyfunction]
#[pyo3(signature = (distance, clearance, weight, power, cone_factor))]
#[pyo3(text_signature = "(distance, clearance, weight, power, cone_factor)")]
fn obstacle_barrier(
    distance: f64,
    clearance: f64,
    weight: f64,
    power: f64,
    cone_factor: f64,
) -> f64 {
    costs::obstacle_barrier(distance, clearance, weight, power, cone_factor)
}

/// Smooth forward-cone factor via heading-bearing projection.
///
/// Projects the unit obstacle offset onto the vehicle heading, floored at
/// zero. Numerically equal to ``max(0, cos(heading - bearing))`` for a
/// nonzero separation between the pose and the obstacle.
///
/// Args:
///     `pose_x`: Vehicle x position, meters.
///     `pose_y`: Vehicle y position, meters.
///     heading: Vehicle heading, radians.
///     `obstacle_x`: Obstacle x position, meters.
///     `obstacle_y`: Obstacle y position, meters.
///
/// Returns:
///     The cone factor in ``[0, 1]``.
#[pyfunction]
#[pyo3(signature = (pose_x, pose_y, heading, obstacle_x, obstacle_y))]
#[pyo3(text_signature = "(pose_x, pose_y, heading, obstacle_x, obstacle_y)")]
fn forward_cone_factor(
    pose_x: f64,
    pose_y: f64,
    heading: f64,
    obstacle_x: f64,
    obstacle_y: f64,
) -> f64 {
    costs::forward_cone_factor(pose_x, pose_y, heading, obstacle_x, obstacle_y)
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(obstacle_barrier, module)?)?;
    module.add_function(wrap_pyfunction!(forward_cone_factor, module)?)?;
    Ok(())
}
