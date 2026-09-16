//! What every guidance entry point requires of a state vector.
//!
//! A guidance state is a position with optional dynamics appended:
//! `(x, y)`, then a heading, then a speed, then a turn rate. The
//! trajectory optimizer builds the five-element form and a planner passes
//! the two-element one, so an entry point reads as far along as it needs
//! and accepts a shorter state rather than demanding a dimension.
//!
//! The position is the part nothing can do without, which is why it is the
//! only part required here.

use arco_core::Error;
use arco_core::geometry::require_finite;

/// Returns the planar position of `state`, rejecting what is not one.
///
/// The check hands back what it validated so that a caller reads the two
/// components it proved were there, rather than looking them up again
/// behind a default that would quietly stand in for a missing one.
///
/// # Errors
///
/// Returns [`Error::TooFew`] when the state carries fewer than two
/// components, and [`Error::NotFinite`] when one of them is NaN or
/// infinite. A NaN compares false against every bound, so an unchecked
/// state reaches a feasibility test and is reported feasible.
pub(crate) fn require_state(quantity: &'static str, state: &[f64]) -> Result<(f64, f64), Error> {
    require_finite(quantity, state)?;
    match *state {
        [x, y, ..] => Ok((x, y)),
        _ => Err(Error::TooFew {
            quantity,
            minimum: 2,
            actual: state.len(),
        }),
    }
}
