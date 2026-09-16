//! Motion primitives: the segments a kinematically constrained tree grows.

mod dubins;

pub use dubins::DubinsPrimitive;

use arco_core::Error;

/// A segment generator a sampling planner can extend its tree with.
///
/// Distinct from [`arco_core::protocols::Steerer`], which takes one
/// bounded step and returns the single state it reached. A primitive
/// returns the whole segment, because the states in between are what a
/// collision check reads and what a Reeds-Shepp or Dubins maneuver is made
/// of. A planner that only needs the endpoint uses the steerer.
pub trait ExplorationPrimitive {
    /// Returns a feasible segment from `from` to `to`.
    ///
    /// The first state of the segment is `from` and the last is `to`, so a
    /// caller stitching segments together does not have to guess whether
    /// the endpoints are included.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when a state is not a planar position,
    /// [`Error::DimensionMismatch`] when the two states disagree in
    /// length, or [`Error::NotFinite`] when one carries a value that is
    /// not real.
    fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<Vec<f64>>, Error>;
}
