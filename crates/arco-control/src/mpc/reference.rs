//! The path a contouring controller measures itself against.
//!
//! A polyline parameterized by arc length, with a heading and a curvature
//! at every point. The curvature is the delicate part and the comments on
//! [`ReferencePath::curvature`] say why.

use arco_core::Error;
use arco_core::numeric::angle_difference;

/// The longest arc a single polyline turn is spread over, meters.
///
/// A sharp turn between two long collinear segments would otherwise be
/// divided by the length of those segments and come out near zero, which
/// switches off the curve speed limit exactly where it is needed.
const CURVATURE_SPREAD_CEILING: f64 = 20.0;

/// The shortest arc a non-trivial turn is spread over, meters.
///
/// The opposite failure. A grid corner or an optimizer stub can be a
/// metre long, and dividing a right angle by that gives a curvature near
/// one per meter, which asks for a speed no vehicle would accept and
/// leaves the solver with nothing feasible.
const CURVATURE_SPREAD_FLOOR: f64 = 8.0;

/// The largest curvature the path will report, per meter.
const CURVATURE_CEILING: f64 = 0.35;

/// How far ahead a corner's curvature is visible, meters.
///
/// Short on purpose. The receding horizon already previews the corner and
/// plans the braking; a long preview counts the same conservatism twice
/// and holds the speed down on the straights between corners.
const CURVATURE_PREVIEW: f64 = 12.0;

/// The shortest segment that counts as having a direction, meters.
const DEGENERATE_SEGMENT: f64 = 1e-12;

/// Where a pose sits relative to the path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathProjection {
    /// Arc length of the closest point, meters.
    pub arc_length: f64,
    /// Signed distance from the path, meters, positive to its left.
    pub lateral_error: f64,
    /// Heading minus the path heading there, radians, wrapped.
    pub heading_error: f64,
}

/// What the path looks like at one arc length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathSample {
    /// Position along the first axis, meters.
    pub x: f64,
    /// Position along the second axis, meters.
    pub y: f64,
    /// Path heading, radians.
    pub heading: f64,
    /// Path curvature, per meter.
    pub curvature: f64,
}

/// An ordered polyline, parameterized by arc length.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferencePath {
    points: Vec<(f64, f64)>,
    lengths: Vec<f64>,
    cumulative: Vec<f64>,
    headings: Vec<f64>,
    curvatures: Vec<f64>,
}

impl ReferencePath {
    /// Builds a path from ordered waypoints.
    ///
    /// Repeated waypoints are dropped rather than rejected: a planner that
    /// emits one has produced a path that is still followable, and a
    /// zero-length segment has no direction to report.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when fewer than two waypoints survive
    /// that filtering, [`Error::NotFinite`] when one carries a NaN, and
    /// [`Error::OutOfRange`] when the total length is zero.
    pub fn new(waypoints: &[(f64, f64)]) -> Result<Self, Error> {
        if waypoints.len() < 2 {
            return Err(Error::TooFew {
                quantity: "reference waypoints",
                minimum: 2,
                actual: waypoints.len(),
            });
        }
        for &(x, y) in waypoints {
            for (quantity, value) in [("waypoint x", x), ("waypoint y", y)] {
                if !value.is_finite() {
                    return Err(Error::NotFinite { quantity, value });
                }
            }
        }

        let mut points: Vec<(f64, f64)> = Vec::with_capacity(waypoints.len());
        for &point in waypoints {
            let keep = points.last().is_none_or(|&(x, y): &(f64, f64)| {
                (point.0 - x).hypot(point.1 - y) > DEGENERATE_SEGMENT
            });
            if keep {
                points.push(point);
            }
        }
        if points.len() < 2 {
            return Err(Error::OutOfRange {
                quantity: "reference path length",
                value: 0.0,
                bound: "(0, inf)",
            });
        }

        let mut lengths = Vec::with_capacity(points.len().saturating_sub(1));
        let mut headings = Vec::with_capacity(points.len());
        let mut cumulative = vec![0.0];
        for pair in points.windows(2) {
            let [from, to] = pair else { continue };
            let (dx, dy) = (to.0 - from.0, to.1 - from.1);
            let length = dx.hypot(dy);
            lengths.push(length);
            headings.push(dy.atan2(dx));
            cumulative.push(cumulative.last().copied().unwrap_or_default() + length);
        }
        // The final vertex has no outgoing segment, so it keeps the
        // heading of the one arriving at it.
        headings.push(headings.last().copied().unwrap_or_default());

        let curvatures = curvature_profile(&cumulative, &headings)?;
        Ok(Self {
            points,
            lengths,
            cumulative,
            headings,
            curvatures,
        })
    }

    /// Total arc length, meters.
    #[must_use]
    pub fn total_length(&self) -> f64 {
        self.cumulative.last().copied().unwrap_or_default()
    }

    /// How many waypoints survived the zero-length filtering.
    #[must_use]
    pub fn waypoint_count(&self) -> usize {
        self.points.len()
    }

    /// Everything the controller needs at arc length `s`.
    #[must_use]
    pub fn sample_at(&self, arc_length: f64) -> PathSample {
        let clamped = self.clamp(arc_length);
        let index = self.segment_index(clamped);
        let start = self.cumulative.get(index).copied().unwrap_or_default();
        let length = self.lengths.get(index).copied().unwrap_or_default();
        let ratio = if length < DEGENERATE_SEGMENT {
            0.0
        } else {
            (clamped - start) / length
        };
        let from = self.points.get(index).copied().unwrap_or_default();
        let to = self
            .points
            .get(index.saturating_add(1))
            .copied()
            .unwrap_or(from);

        PathSample {
            x: (to.0 - from.0).mul_add(ratio, from.0),
            y: (to.1 - from.1).mul_add(ratio, from.1),
            heading: self.headings.get(index).copied().unwrap_or_default(),
            curvature: self.curvature(clamped),
        }
    }

    /// Curvature at arc length `s`, per meter.
    ///
    /// Interpolated between vertices, since the profile is defined there.
    #[must_use]
    pub fn curvature(&self, arc_length: f64) -> f64 {
        let clamped = self.clamp(arc_length);
        let index = self.segment_index(clamped);
        let start = self.cumulative.get(index).copied().unwrap_or_default();
        let length = self.lengths.get(index).copied().unwrap_or_default();
        let ratio = if length < DEGENERATE_SEGMENT {
            0.0
        } else {
            (clamped - start) / length
        };
        let left = self.curvatures.get(index).copied().unwrap_or_default();
        let right = self
            .curvatures
            .get(index.saturating_add(1))
            .copied()
            .unwrap_or(left);
        (right - left).mul_add(ratio, left)
    }

    /// Projects a pose onto the path.
    ///
    /// # Arguments
    ///
    /// * `pose` - The pose as `(x, y, heading)`.
    /// * `window` - An arc-length hint and half-width to search within.
    ///   Without it the search is global, which after a cut corner can
    ///   snap progress onto a different part of the same route.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when the pose carries a NaN.
    pub fn project(
        &self,
        pose: (f64, f64, f64),
        window: Option<(f64, f64)>,
    ) -> Result<PathProjection, Error> {
        for (quantity, value) in [("pose x", pose.0), ("pose y", pose.1), ("heading", pose.2)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }

        let (low, high) = match window {
            Some((hint, half_width)) if half_width > 0.0 => (
                (hint - half_width).max(0.0),
                (hint + half_width).min(self.total_length()),
            ),
            _ => (0.0, self.total_length()),
        };

        let mut best: Option<(f64, PathProjection)> = None;
        for (index, &length) in self.lengths.iter().enumerate() {
            let start = self.cumulative.get(index).copied().unwrap_or_default();
            let end = self
                .cumulative
                .get(index.saturating_add(1))
                .copied()
                .unwrap_or(start);
            if end < low || start > high || length < DEGENERATE_SEGMENT {
                continue;
            }
            let from = self.points.get(index).copied().unwrap_or_default();
            let to = self
                .points
                .get(index.saturating_add(1))
                .copied()
                .unwrap_or(from);
            let tangent = ((to.0 - from.0) / length, (to.1 - from.1) / length);

            let along = (pose.0 - from.0).mul_add(tangent.0, (pose.1 - from.1) * tangent.1);
            let candidate = (start + along.clamp(0.0, length)).clamp(low, high);
            let offset = candidate - start;
            let closest = (
                offset.mul_add(tangent.0, from.0),
                offset.mul_add(tangent.1, from.1),
            );

            let distance = (pose.0 - closest.0).hypot(pose.1 - closest.1);
            if best
                .as_ref()
                .is_none_or(|&(previous, _)| distance < previous)
            {
                let heading = tangent.1.atan2(tangent.0);
                best = Some((
                    distance,
                    PathProjection {
                        arc_length: candidate,
                        // Left of the tangent is positive.
                        lateral_error: (-(pose.0 - closest.0))
                            .mul_add(tangent.1, (pose.1 - closest.1) * tangent.0),
                        heading_error: angle_difference(pose.2, heading)?,
                    },
                ));
            }
        }

        Ok(best.map_or(
            PathProjection {
                arc_length: 0.0,
                lateral_error: 0.0,
                heading_error: 0.0,
            },
            |(_, projection)| projection,
        ))
    }

    /// Holds `s` inside the path.
    fn clamp(&self, arc_length: f64) -> f64 {
        if arc_length.is_nan() {
            return 0.0;
        }
        arc_length.clamp(0.0, self.total_length())
    }

    /// Which segment contains arc length `s`.
    fn segment_index(&self, arc_length: f64) -> usize {
        let last = self.lengths.len().saturating_sub(1);
        // Linear rather than binary: a reference path is a few hundred
        // vertices and this runs once per stage, so the branch-free scan
        // beats the search on every path this controller sees.
        let mut index = 0;
        for (candidate, &start) in self.cumulative.iter().enumerate() {
            if start > arc_length {
                break;
            }
            index = candidate;
        }
        index.min(last)
    }
}

/// The curvature at every vertex, spread, previewed and capped.
///
/// Each interior vertex turns by the wrapped difference between the
/// heading arriving and the heading leaving, and that turn is divided by
/// an arc length chosen between a floor and a ceiling. A skip-one
/// difference across the vertex would be the obvious alternative and it
/// reports nothing at a right-angle kink, because the two headings
/// either side cancel.
fn curvature_profile(cumulative: &[f64], headings: &[f64]) -> Result<Vec<f64>, Error> {
    let count = headings.len();
    let mut curvature = vec![0.0; count];

    for index in 1..count.saturating_sub(1) {
        let previous = cumulative
            .get(index.saturating_sub(1))
            .copied()
            .unwrap_or_default();
        let here = cumulative.get(index).copied().unwrap_or_default();
        let next = cumulative
            .get(index.saturating_add(1))
            .copied()
            .unwrap_or(here);
        let mut span = (here - previous)
            .min(next - here)
            .min(CURVATURE_SPREAD_CEILING);
        if span < DEGENERATE_SEGMENT {
            continue;
        }

        let turn = angle_difference(
            headings.get(index).copied().unwrap_or_default(),
            headings
                .get(index.saturating_sub(1))
                .copied()
                .unwrap_or_default(),
        )?;
        if turn.abs() > 1e-6 {
            span = span.max(CURVATURE_SPREAD_FLOOR);
        }
        if let Some(slot) = curvature.get_mut(index) {
            *slot = turn / span;
        }
    }

    // The endpoints have no turn of their own, so they take the one next
    // to them rather than reporting a straight where the path bends.
    if count >= 2 {
        let first = curvature.get(1).copied().unwrap_or_default();
        let last = curvature
            .get(count.saturating_sub(2))
            .copied()
            .unwrap_or_default();
        if let Some(slot) = curvature.first_mut() {
            *slot = first;
        }
        if let Some(slot) = curvature.last_mut() {
            *slot = last;
        }
    }

    let previewed = preview(cumulative, &curvature);
    Ok(previewed
        .into_iter()
        .map(|value| value.clamp(-CURVATURE_CEILING, CURVATURE_CEILING))
        .collect())
}

/// Carries the sharpest curvature ahead backward along the path.
///
/// Without it a vehicle meets a corner's curvature at the corner, which
/// is where braking would have to be instantaneous.
fn preview(cumulative: &[f64], curvature: &[f64]) -> Vec<f64> {
    let mut previewed = curvature.to_vec();
    for index in 0..curvature.len() {
        let here = cumulative.get(index).copied().unwrap_or_default();
        let mut strongest = curvature.get(index).copied().unwrap_or_default();
        for ahead in index.saturating_add(1)..curvature.len() {
            if cumulative.get(ahead).copied().unwrap_or_default() - here > CURVATURE_PREVIEW {
                break;
            }
            let candidate = curvature.get(ahead).copied().unwrap_or_default();
            if candidate.abs() > strongest.abs() {
                strongest = candidate;
            }
        }
        if let Some(slot) = previewed.get_mut(index) {
            *slot = strongest;
        }
    }
    previewed
}
