//! Sliding-window smoothing of a waypoint polyline.

use arco_core::Error;

use super::{Interpolator, require_finite_path};

/// The shortest path a window can smooth, in waypoints.
///
/// Two waypoints are a segment with no interior to average, and a single
/// waypoint is not a path at all. Both pass through untouched.
const SMOOTHABLE_LENGTH: usize = 3;

/// A moving-average smoother for a waypoint polyline.
///
/// Filters the lateral wiggle that a sampling planner leaves behind and
/// the staircase a grid search produces. Each interior waypoint is
/// replaced by the mean of the window centered on it, the two endpoints
/// are kept exactly, and repeated passes smooth harder.
///
/// The filter only ever pulls a waypoint toward the local chord, so a
/// lateral excursion shrinks, and so does a corner that was there on
/// purpose. A caller tracking through an obstacle field checks clearance
/// after smoothing rather than assuming it survived; the sparse simulator
/// scene is the worked example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MovingAverageInterpolator {
    iterations: usize,
    window: usize,
}

impl MovingAverageInterpolator {
    /// Builds a smoother running `iterations` passes of `window` waypoints.
    ///
    /// # Arguments
    ///
    /// * `iterations` - How many passes to run, at least one.
    /// * `window` - Waypoints averaged per interior point, an odd count of
    ///   at least three so that the window is centered on the point it
    ///   replaces.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `iterations` is zero, or when
    /// `window` is even or below three. An even window has no center, so
    /// it would shift the path along its own length rather than smooth it.
    pub fn new(iterations: usize, window: usize) -> Result<Self, Error> {
        if iterations == 0 {
            return Err(Error::OutOfRange {
                quantity: "smoothing passes",
                value: 0.0,
                bound: "at least 1",
            });
        }
        if window < SMOOTHABLE_LENGTH || window.is_multiple_of(2) {
            return Err(Error::OutOfRange {
                quantity: "smoothing window",
                value: as_real(window),
                bound: "an odd count of at least 3",
            });
        }
        Ok(Self { iterations, window })
    }

    /// How many passes this smoother runs.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// How many waypoints each interior mean covers.
    #[must_use]
    pub const fn window(&self) -> usize {
        self.window
    }
}

impl Interpolator for MovingAverageInterpolator {
    /// Runs the configured number of smoothing passes over `path`.
    ///
    /// A path shorter than three waypoints comes back as it went in: there
    /// is no interior point to replace, and the endpoints are preserved by
    /// definition.
    ///
    /// Near an endpoint the window is truncated rather than extended past
    /// the end, so the second waypoint of a five-wide window is the mean
    /// of four rather than of five. Mirroring the path instead would
    /// invent waypoints that are not on the route, and extending the
    /// window inward would pull the second point further than the third.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a waypoint is not a real number.
    fn interpolate(&self, path: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, Error> {
        require_finite_path(path)?;
        if path.len() < SMOOTHABLE_LENGTH {
            return Ok(path.to_vec());
        }

        // The window is odd and at least three, so the half width is at
        // least one. The divisor is a literal rather than a value a caller
        // chose, which is what makes the division safe to write at all.
        let half = self.window.div_euclid(2);
        let last = path.len().saturating_sub(1);
        let mut points = path.to_vec();
        let mut smoothed = points.clone();

        for _ in 0..self.iterations {
            for index in 1..last {
                let low = index.saturating_sub(half);
                let high = index.saturating_add(half).saturating_add(1).min(path.len());
                let (Some(window), Some(slot)) = (points.get(low..high), smoothed.get_mut(index))
                else {
                    continue;
                };
                if let Some(mean) = mean(window) {
                    *slot = mean;
                }
            }
            // The two buffers agree at every index this pass did not
            // write, which is the two endpoints, so swapping is the same
            // as copying and keeps the loop free of allocation.
            core::mem::swap(&mut points, &mut smoothed);
        }
        Ok(points)
    }
}

/// The componentwise mean of a window, or `None` when it is empty.
///
/// The count accumulates alongside the sum rather than being converted
/// from the window length, which keeps a lossy integer conversion out of
/// the loop and leaves the empty case answerable without dividing by zero.
fn mean(window: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mut sum = (0.0, 0.0);
    let mut count = 0.0_f64;
    for &(x, y) in window {
        sum.0 += x;
        sum.1 += y;
        count += 1.0;
    }
    if count > 0.0 {
        Some((sum.0 / count, sum.1 / count))
    } else {
        None
    }
}

/// A count as a real number, for an error message.
fn as_real(count: usize) -> f64 {
    u32::try_from(count).map_or(f64::INFINITY, f64::from)
}
