//! Rewriting a waypoint list into one a vehicle can follow more smoothly.

mod bspline;
mod moving_average;

pub use bspline::BSplineInterpolator;
pub use moving_average::MovingAverageInterpolator;

use arco_core::Error;

/// A stage that turns a discrete path into a smoother one.
///
/// Waypoints are `(x, y)` pairs rather than the arbitrary objects the
/// Python protocol accepted, which is the shape
/// [`arco_core::protocols::PathTracker`] reads, so an interpolated path
/// goes straight to a tracker without a conversion in between. The Python
/// implementations coerced whatever they were given to a pair of floats
/// anyway.
///
/// An interpolator is free to return a different number of waypoints than
/// it was given. Nothing in the contract says the result is the same
/// length, only that it describes the same route.
pub trait Interpolator {
    /// Returns a continuous trajectory through `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a waypoint carries a value that
    /// is not a real number, since an interpolator averages its input and
    /// one NaN would spread across the whole neighborhood it touches.
    fn interpolate(&self, path: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, Error>;
}

/// Rejects a path carrying a waypoint that is not a real number.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] naming the first offending component.
pub(crate) fn require_finite_path(path: &[(f64, f64)]) -> Result<(), Error> {
    for &(x, y) in path {
        for value in [x, y] {
            if !value.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "path waypoint",
                    value,
                });
            }
        }
    }
    Ok(())
}
