//! The B-spline interpolator, which records a degree and smooths nothing.

use arco_core::Error;

use super::{Interpolator, require_finite_path};

/// A B-spline interpolator over a waypoint polyline.
///
/// **This interpolator returns the path it was given.**
/// `arco.guidance.interpolation.bspline` is a placeholder that stores the
/// degree and returns its argument, with a comment saying a real
/// implementation would call `scipy.interpolate`, and the port carries
/// that behavior across unchanged rather than inventing a curve the Python
/// callers never saw. `FR-CORE-01` is the reason: the existing test suite
/// asserts the identity, and a port that quietly started smoothing would
/// change every path that passes through here.
///
/// [`super::MovingAverageInterpolator`] is the interpolator that actually
/// smooths, and it is what the simulator scenes use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BSplineInterpolator {
    degree: usize,
}

impl BSplineInterpolator {
    /// Builds an interpolator of polynomial `degree`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `degree` is zero, which
    /// describes a step function rather than a curve through the
    /// waypoints. Nothing reads the degree yet, so the check is about what
    /// the caller asked for rather than about what happens next.
    pub fn new(degree: usize) -> Result<Self, Error> {
        if degree == 0 {
            return Err(Error::OutOfRange {
                quantity: "spline degree",
                value: 0.0,
                bound: "at least 1",
            });
        }
        Ok(Self { degree })
    }

    /// The polynomial degree this interpolator was built with.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }
}

impl Interpolator for BSplineInterpolator {
    /// Returns `path` unchanged, per the note on the type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a waypoint is not a real number.
    /// The check is here rather than skipped with the smoothing, so that a
    /// caller swapping interpolators sees the same rejections from both.
    fn interpolate(&self, path: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, Error> {
        require_finite_path(path)?;
        Ok(path.to_vec())
    }
}
