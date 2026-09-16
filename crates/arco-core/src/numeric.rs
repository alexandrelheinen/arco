//! Tolerances, total ordering, and angle handling.
//!
//! These exist because floating point defaults are wrong for this domain
//! in three specific ways, each of which has bitten a navigation stack
//! before.
//!
//! A single project-wide epsilon cannot be right, because a fixed
//! relative error expressed in units of the last place wobbles by a factor
//! of the radix across the exponent range. Tolerances here are named
//! constants in meters, radians, and seconds, per `FR-SAFE-06`.
//!
//! Floating point values are only partially ordered, since a NaN compares
//! unordered with everything including itself. Anything that sorts, keys a
//! map, or selects a minimum needs a total order, which is `FR-SAFE-05`.
//!
//! An angular difference taken with plain subtraction is wrong by a full
//! turn across the branch cut, which is `FR-INV-15`.

use crate::Error;

/// Agreement tolerance for a position, in meters.
pub const POSITION_TOLERANCE: f64 = 1e-6;

/// Agreement tolerance for an angle, in radians.
pub const ANGLE_TOLERANCE: f64 = 1e-9;

/// Agreement tolerance for a duration, in seconds.
pub const TIME_TOLERANCE: f64 = 1e-9;

/// Relative tolerance for comparing values far from zero.
///
/// Roughly half the significand, which is the accuracy a well behaved
/// numerical algorithm is expected to retain.
pub const RELATIVE_TOLERANCE: f64 = 1e-9;

/// Reports whether two values agree within the given tolerances.
///
/// The absolute tolerance governs near zero, where no relative tolerance
/// is meaningful, and the relative tolerance governs away from it. Both
/// are required arguments because picking one for the caller is how a
/// single project-wide epsilon gets established by accident.
///
/// This relation is **not transitive**: `a` close to `b` and `b` close to
/// `c` does not imply `a` close to `c`. Never use it as a sort key, a map
/// key, or a deduplication criterion.
///
/// # Arguments
///
/// * `left` - First value.
/// * `right` - Second value.
/// * `absolute` - Absolute tolerance, in the unit of the quantity.
/// * `relative` - Relative tolerance, dimensionless.
///
/// # Examples
///
/// ```
/// use arco_core::numeric::{is_close, POSITION_TOLERANCE, RELATIVE_TOLERANCE};
///
/// assert!(is_close(1.0, 1.0 + 1e-12, POSITION_TOLERANCE, RELATIVE_TOLERANCE));
/// assert!(!is_close(1.0, 1.5, POSITION_TOLERANCE, RELATIVE_TOLERANCE));
/// ```
#[must_use]
pub fn is_close(left: f64, right: f64, absolute: f64, relative: f64) -> bool {
    #[expect(
        clippy::float_cmp,
        reason = "exact equality is the only way to accept two equal infinities, and this compares the caller's values rather than two computed results"
    )]
    if left == right {
        return true;
    }
    if !left.is_finite() || !right.is_finite() {
        return false;
    }

    let difference = (left - right).abs();
    if difference <= absolute {
        return true;
    }

    let largest = left.abs().max(right.abs());
    difference <= largest * relative
}

/// Reports whether two positions agree, in meters.
#[must_use]
pub fn positions_close(left: f64, right: f64) -> bool {
    is_close(left, right, POSITION_TOLERANCE, RELATIVE_TOLERANCE)
}

/// Reports whether two angles agree, in radians, across the branch cut.
#[must_use]
pub fn angles_close(left: f64, right: f64) -> bool {
    match angle_difference(left, right) {
        Ok(difference) => difference.abs() <= ANGLE_TOLERANCE,
        Err(_) => false,
    }
}

/// Wraps an angle into `[-pi, pi)`.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when `angle` is NaN or infinite, since
/// neither has a representative in the interval.
pub fn wrap_angle(angle: f64) -> Result<f64, Error> {
    if !angle.is_finite() {
        return Err(Error::NotFinite {
            quantity: "angle",
            value: angle,
        });
    }

    let turn = core::f64::consts::TAU;
    #[expect(
        clippy::modulo_arithmetic,
        reason = "the truncated remainder keeps the sign of the angle, which is what the two corrections below assume"
    )]
    let mut wrapped = angle % turn;
    if wrapped >= core::f64::consts::PI {
        wrapped -= turn;
    } else if wrapped < -core::f64::consts::PI {
        wrapped += turn;
    }
    Ok(wrapped)
}

/// The signed difference `left - right`, wrapped into `[-pi, pi)`.
///
/// Plain subtraction is wrong by a full turn across the branch cut, which
/// is why no control law in ARCO subtracts two angles directly.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when either angle is NaN or infinite.
///
/// # Examples
///
/// ```
/// use arco_core::numeric::angle_difference;
/// use core::f64::consts::PI;
///
/// // Naive subtraction gives -1.8 pi; the wrapped difference is 0.2 pi.
/// let difference = angle_difference(-0.9 * PI, 0.9 * PI).unwrap();
/// assert!((difference - 0.2 * PI).abs() < 1e-12);
/// ```
pub fn angle_difference(left: f64, right: f64) -> Result<f64, Error> {
    wrap_angle(wrap_angle(left)? - wrap_angle(right)?)
}

/// A floating point value known to be finite, and therefore totally ordered.
///
/// Parsing a value into this type once removes the partial-order problem
/// everywhere downstream: a priority queue keyed on one of these cannot
/// silently misbehave because a cost went to NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Finite(f64);

impl Finite {
    /// Wraps `value`, rejecting anything not finite.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when `value` is NaN or infinite.
    pub fn new(quantity: &'static str, value: f64) -> Result<Self, Error> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(Error::NotFinite { quantity, value })
        }
    }

    /// The wrapped value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl Eq for Finite {}

impl PartialOrd for Finite {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Finite {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // total_cmp is defined for every value including NaN, and the
        // constructor has already excluded the cases where it would
        // disagree with the numeric ordering.
        self.0.total_cmp(&other.0)
    }
}

impl core::fmt::Display for Finite {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn a_value_is_close_to_itself() {
        for value in [0.0, -0.0, 1.0, -1e30, 1e-30] {
            assert!(positions_close(value, value), "{value}");
        }
    }

    #[test]
    fn closeness_is_symmetric() {
        assert_eq!(
            positions_close(1.0, 1.0 + 1e-12),
            positions_close(1.0 + 1e-12, 1.0)
        );
    }

    #[test]
    fn closeness_is_not_transitive() {
        // Stated as a test because the property is load-bearing: this is
        // why a tolerance test can never be a sort or dedup criterion.
        let step = POSITION_TOLERANCE * 0.6;
        let first = 1.0;
        let second = first + step;
        let third = second + step;
        assert!(positions_close(first, second));
        assert!(positions_close(second, third));
        assert!(!positions_close(first, third));
    }

    #[test]
    fn equal_infinities_are_close_and_opposite_ones_are_not() {
        assert!(positions_close(f64::INFINITY, f64::INFINITY));
        assert!(!positions_close(f64::INFINITY, f64::NEG_INFINITY));
    }

    #[test]
    fn nothing_is_close_to_a_nan() {
        assert!(!positions_close(f64::NAN, f64::NAN));
        assert!(!positions_close(0.0, f64::NAN));
    }

    #[test]
    fn wrapping_lands_in_the_half_open_interval() {
        for step in -50..50 {
            let angle = f64::from(step) * 0.37;
            let wrapped = wrap_angle(angle).unwrap();
            assert!((-PI..PI).contains(&wrapped), "{angle} gave {wrapped}");
        }
    }

    #[test]
    fn wrapping_is_idempotent() {
        for step in -50..50 {
            let angle = f64::from(step) * 0.37;
            let once = wrap_angle(angle).unwrap();
            let twice = wrap_angle(once).unwrap();
            assert!(angles_close(once, twice), "{once} against {twice}");
        }
    }

    #[test]
    fn a_difference_across_the_branch_cut_stays_small() {
        let difference = angle_difference(-0.99 * PI, 0.99 * PI).unwrap();
        assert!(difference.abs() < 0.1, "{difference}");
    }

    #[test]
    fn a_non_finite_angle_is_rejected() {
        assert!(wrap_angle(f64::NAN).is_err());
        assert!(wrap_angle(f64::INFINITY).is_err());
        assert!(angle_difference(0.0, f64::NAN).is_err());
    }

    #[test]
    fn finite_rejects_what_it_says_it_rejects() {
        assert!(Finite::new("cost", 1.0).is_ok());
        assert!(Finite::new("cost", f64::NAN).is_err());
        assert!(Finite::new("cost", f64::INFINITY).is_err());
    }

    #[test]
    fn finite_values_sort_numerically() {
        let mut values: Vec<Finite> = [3.0, -1.0, 2.5, 0.0]
            .into_iter()
            .map(|value| Finite::new("cost", value).unwrap())
            .collect();
        values.sort();
        let sorted: Vec<f64> = values.into_iter().map(Finite::get).collect();
        assert_eq!(sorted, vec![-1.0, 0.0, 2.5, 3.0]);
    }
}
