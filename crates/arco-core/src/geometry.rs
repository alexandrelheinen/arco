//! Poses, points, and the distance functions everything shares.
//!
//! ARCO is dimension-generic: a graph node, a grid cell, and a sampled
//! state are all points of whatever dimension the caller chose, which is
//! why the point functions here take slices and check their lengths rather
//! than fixing a dimension in the type. Planners or methods that only
//! support a specific dimension reject the rest, per `FR-CORE-03`.
//!
//! [`Pose`] is the exception. A plane pose is always three numbers, and
//! the control layer reads its heading often enough that keeping it
//! wrapped by construction removes a whole class of bug.

use crate::Error;
use crate::numeric::{POSITION_TOLERANCE, RELATIVE_TOLERANCE, angle_difference, wrap_angle};

/// A pose in the plane: a position and a heading.
///
/// The heading is wrapped into `[-pi, pi)` at construction, so nothing
/// downstream has to remember to wrap it, and an angular difference taken
/// through [`Pose::heading_difference`] is correct across the branch cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    x: f64,
    y: f64,
    heading: f64,
}

impl Pose {
    /// Builds a pose, wrapping the heading.
    ///
    /// # Arguments
    ///
    /// * `x` - Position along the first axis, meters.
    /// * `y` - Position along the second axis, meters.
    /// * `heading` - Orientation, radians, wrapped into `[-pi, pi)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any component is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use arco_core::geometry::Pose;
    /// use core::f64::consts::PI;
    ///
    /// let pose = Pose::new(1.0, 2.0, 3.0 * PI).unwrap();
    /// assert!((pose.heading() - -PI).abs() < 1e-12 || (pose.heading() - PI).abs() < 1e-12);
    /// ```
    pub fn new(x: f64, y: f64, heading: f64) -> Result<Self, Error> {
        for (quantity, value) in [("x", x), ("y", y)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        Ok(Self {
            x,
            y,
            heading: wrap_angle(heading)?,
        })
    }

    /// Position along the first axis, meters.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Position along the second axis, meters.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Orientation, radians, always in `[-pi, pi)`.
    #[must_use]
    pub const fn heading(self) -> f64 {
        self.heading
    }

    /// Straight-line distance to another pose, meters, ignoring heading.
    #[must_use]
    pub fn distance_to(self, other: Self) -> f64 {
        (other.x - self.x).hypot(other.y - self.y)
    }

    /// Signed heading difference `self - other`, wrapped into `[-pi, pi)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] only if an invariant has been broken,
    /// since both headings were wrapped at construction.
    pub fn heading_difference(self, other: Self) -> Result<f64, Error> {
        angle_difference(self.heading, other.heading)
    }
}

/// Rejects a point whose dimension is not the one required.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] when `point` is the wrong length.
pub fn require_dimension(
    quantity: &'static str,
    point: &[f64],
    expected: usize,
) -> Result<(), Error> {
    if point.len() == expected {
        Ok(())
    } else {
        Err(Error::DimensionMismatch {
            quantity,
            expected,
            actual: point.len(),
        })
    }
}

/// Rejects a point carrying a value that is not finite.
///
/// `FR-SAFE-07`. A NaN coordinate propagates through every later
/// operation and makes each comparison against it false, so a planner
/// silently selects a garbage candidate instead of failing.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] naming the first offending value.
pub fn require_finite(quantity: &'static str, point: &[f64]) -> Result<(), Error> {
    match point.iter().copied().find(|value| !value.is_finite()) {
        Some(value) => Err(Error::NotFinite { quantity, value }),
        None => Ok(()),
    }
}

/// Euclidean distance between two points of equal dimension.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] when the points differ in length,
/// or [`Error::NotFinite`] when either carries a non-finite value.
///
/// # Examples
///
/// ```
/// use arco_core::geometry::euclidean_distance;
///
/// let distance = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap();
/// assert!((distance - 5.0).abs() < 1e-12);
/// ```
pub fn euclidean_distance(left: &[f64], right: &[f64]) -> Result<f64, Error> {
    require_dimension("point", right, left.len())?;
    require_finite("point", left)?;
    require_finite("point", right)?;

    let sum_of_squares: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| {
            let difference = a - b;
            difference * difference
        })
        .sum();
    Ok(sum_of_squares.sqrt())
}

/// Manhattan distance between two points of equal dimension.
///
/// # Errors
///
/// As [`euclidean_distance`].
pub fn manhattan_distance(left: &[f64], right: &[f64]) -> Result<f64, Error> {
    require_dimension("point", right, left.len())?;
    require_finite("point", left)?;
    require_finite("point", right)?;

    Ok(left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum())
}

/// Reports whether two points agree within the position tolerance.
///
/// # Errors
///
/// As [`euclidean_distance`].
pub fn points_close(left: &[f64], right: &[f64]) -> Result<bool, Error> {
    let distance = euclidean_distance(left, right)?;
    let scale = left
        .iter()
        .chain(right)
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    Ok(distance <= POSITION_TOLERANCE + scale * RELATIVE_TOLERANCE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn a_pose_wraps_its_heading_at_construction() {
        let pose = Pose::new(0.0, 0.0, 5.0 * PI).unwrap();
        assert!((-PI..PI).contains(&pose.heading()), "{}", pose.heading());
    }

    #[test]
    fn a_pose_rejects_a_non_finite_component() {
        assert!(Pose::new(f64::NAN, 0.0, 0.0).is_err());
        assert!(Pose::new(0.0, f64::INFINITY, 0.0).is_err());
        assert!(Pose::new(0.0, 0.0, f64::NAN).is_err());
    }

    #[test]
    fn a_heading_difference_crosses_the_branch_cut() {
        let left = Pose::new(0.0, 0.0, -0.99 * PI).unwrap();
        let right = Pose::new(0.0, 0.0, 0.99 * PI).unwrap();
        let difference = left.heading_difference(right).unwrap();
        assert!(difference.abs() < 0.1, "{difference}");
    }

    #[test]
    fn distances_agree_with_the_three_four_five_triangle() {
        assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 5.0).abs() < 1e-12);
        assert!((manhattan_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn distance_works_in_any_dimension() {
        // FR-CORE-03: the library is dimension-generic where it can be.
        for dimension in 1_u32..8 {
            let count = usize::try_from(dimension).unwrap_or_default();
            let origin = vec![0.0; count];
            let unit = vec![1.0; count];
            let expected = f64::from(dimension).sqrt();
            let produced = euclidean_distance(&origin, &unit).unwrap();
            assert!((produced - expected).abs() < 1e-12, "dimension {dimension}");
        }
    }

    #[test]
    fn mismatched_dimensions_are_rejected() {
        let error = euclidean_distance(&[0.0, 0.0], &[1.0]).unwrap_err();
        assert!(
            matches!(error, Error::DimensionMismatch { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn a_non_finite_coordinate_is_rejected() {
        let error = euclidean_distance(&[0.0, f64::NAN], &[1.0, 1.0]).unwrap_err();
        assert!(matches!(error, Error::NotFinite { .. }), "{error:?}");
    }

    #[test]
    fn distance_is_symmetric() {
        let left = [1.5, -2.0, 0.25];
        let right = [-3.0, 4.0, 7.5];
        let forward = euclidean_distance(&left, &right).unwrap();
        let backward = euclidean_distance(&right, &left).unwrap();
        assert!((forward - backward).abs() < 1e-12);
    }

    #[test]
    fn a_point_is_close_to_itself_and_not_to_a_distant_one() {
        let point = [1.0, 2.0, 3.0];
        assert!(points_close(&point, &point).unwrap());
        assert!(!points_close(&point, &[1.0, 2.0, 4.0]).unwrap());
    }
}
