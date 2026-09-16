//! The Dubins primitive: forward motion with a floor under the turn radius.

use arco_core::Error;
use arco_core::geometry::require_dimension;

use super::ExplorationPrimitive;
use crate::state::require_state;

/// Turn rate below which a maneuver counts as straight, radians per second.
///
/// The instantaneous turning radius is the speed divided by the turn rate,
/// so a turn rate at or under this leaves the radius larger than any
/// minimum a caller could state and there is nothing left to check. The
/// threshold is the one `arco.guidance.primitive.dubins` used; it is far
/// below any turn rate a machine produces, which is what makes it a guard
/// against dividing by a rounding artifact rather than a domain tolerance.
const STRAIGHT_TURN_RATE: f64 = 1e-12;

/// A Dubins path primitive for a car-like robot.
///
/// Carries the minimum turning radius and checks a state against it. A
/// robot moving at speed `v` with turn rate `w` traces a circle of radius
/// `v / |w|`, and the maneuver is executable only when that radius is at
/// least the one configured here.
///
/// **[`DubinsPrimitive::steer`] returns the two endpoints rather than a
/// Dubins arc.** `arco.guidance.primitive.dubins` is a placeholder that
/// returns `[from_state, to_state]`, with a comment saying a real
/// implementation would call a Dubins library, and the port carries that
/// behavior across unchanged under `FR-CORE-01` rather than changing every
/// path that passes through here. The turning-radius constraint is real
/// and lives in [`DubinsPrimitive::is_feasible`], which is where the
/// trajectory optimizer reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DubinsPrimitive {
    turning_radius: f64,
}

impl DubinsPrimitive {
    /// Builds a primitive with a minimum turning radius, in meters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the radius is not finite and
    /// strictly positive. A radius of zero accepts every state, which
    /// makes the check look like it passed rather than like it was never
    /// configured.
    pub fn new(turning_radius: f64) -> Result<Self, Error> {
        if !(turning_radius.is_finite() && turning_radius > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "turning radius",
                value: turning_radius,
                bound: "(0, inf)",
            });
        }
        Ok(Self { turning_radius })
    }

    /// The minimum turning radius, meters.
    #[must_use]
    pub const fn turning_radius(&self) -> f64 {
        self.turning_radius
    }

    /// Whether `state` satisfies the minimum turning radius.
    ///
    /// A state has to carry both a speed and a turn rate to say anything
    /// about curvature, so `(x, y)`, `(x, y, heading)` and
    /// `(x, y, heading, speed)` are all accepted: the constraint is on the
    /// ratio of the last two components, and a state that has only one of
    /// them cannot violate it. A state at the exact minimum radius is
    /// feasible, since the radius is a floor rather than a strict bound.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when the state is not a planar position,
    /// or [`Error::NotFinite`] when it carries a value that is not real.
    pub fn is_feasible(&self, state: &[f64]) -> Result<bool, Error> {
        require_state("state", state)?;
        let (Some(&speed), Some(&turn_rate)) = (state.get(3), state.get(4)) else {
            return Ok(true);
        };
        if turn_rate.abs() <= STRAIGHT_TURN_RATE {
            return Ok(true);
        }
        Ok(speed.abs() / turn_rate.abs() >= self.turning_radius)
    }
}

impl ExplorationPrimitive for DubinsPrimitive {
    /// Returns the segment as its two endpoints, per the note on the type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when a state is not a planar position,
    /// [`Error::DimensionMismatch`] when the states disagree in length, or
    /// [`Error::NotFinite`] when one carries a value that is not real.
    fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<Vec<f64>>, Error> {
        require_state("start state", from)?;
        require_state("target state", to)?;
        require_dimension("target state", to, from.len())?;
        Ok(vec![from.to_vec(), to.to_vec()])
    }
}
