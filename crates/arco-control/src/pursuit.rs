//! Pure pursuit: steer at a point a fixed distance ahead on the path.

use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::numeric::angle_difference;
use arco_core::protocols::{Command, PathTracker, TrackerErrors};

/// A pure pursuit path tracker.
///
/// The turn rate is `2 v sin(alpha) / L`, where `alpha` is the bearing
/// from the vehicle's heading to a point `L` ahead on the path. Cross
/// track and heading error are computed on the way and kept, because a
/// tracking loop logs them and computing them twice is the only
/// alternative.
#[derive(Debug, Clone)]
pub struct PurePursuitTracker {
    lookahead_distance: f64,
    cross_track_error: f64,
    heading_error: f64,
    curvature: f64,
}

impl PurePursuitTracker {
    /// Builds a tracker looking `lookahead_distance` meters ahead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the distance is not finite and
    /// strictly positive, since the turn-rate law divides by it.
    pub fn new(lookahead_distance: f64) -> Result<Self, Error> {
        if !(lookahead_distance.is_finite() && lookahead_distance > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "lookahead distance",
                value: lookahead_distance,
                bound: "(0, inf)",
            });
        }
        Ok(Self {
            lookahead_distance,
            cross_track_error: 0.0,
            heading_error: 0.0,
            curvature: 0.0,
        })
    }

    /// The lookahead distance, meters.
    #[must_use]
    pub const fn lookahead_distance(&self) -> f64 {
        self.lookahead_distance
    }

    /// Signed distance from the path, meters, positive to its left.
    #[must_use]
    pub const fn cross_track_error(&self) -> f64 {
        self.cross_track_error
    }

    /// Heading minus path tangent at the closest waypoint, radians.
    #[must_use]
    pub const fn heading_error(&self) -> f64 {
        self.heading_error
    }

    /// Signed curvature the last command implied, per meter.
    #[must_use]
    pub const fn curvature(&self) -> f64 {
        self.curvature
    }

    /// The index of the path waypoint closest to `pose`.
    fn closest_index(pose: Pose, path: &[(f64, f64)]) -> usize {
        let mut best = (0_usize, f64::INFINITY);
        for (index, &(x, y)) in path.iter().enumerate() {
            let distance = (x - pose.x()).hypot(y - pose.y());
            if distance < best.1 {
                best = (index, distance);
            }
        }
        best.0
    }

    /// Updates the two errors against the segment at `closest`.
    fn update_errors(
        &mut self,
        pose: Pose,
        path: &[(f64, f64)],
        closest: usize,
    ) -> Result<(), Error> {
        let segment = if closest.saturating_add(1) < path.len() {
            pair(path, closest, closest.saturating_add(1))
        } else {
            pair(path, closest.saturating_sub(1), closest)
        };
        let Some(((ax, ay), (bx, by))) = segment else {
            self.cross_track_error = 0.0;
            self.heading_error = 0.0;
            return Ok(());
        };

        let (dx, dy) = (bx - ax, by - ay);
        let length = dx.hypot(dy);
        // A repeated waypoint has no tangent, and inventing one from the
        // rounding noise between two equal points would put an arbitrary
        // heading error into the log.
        if length <= f64::EPSILON {
            self.cross_track_error = 0.0;
            self.heading_error = 0.0;
            return Ok(());
        }

        let Some(&(cx, cy)) = path.get(closest) else {
            return Ok(());
        };
        // Left-pointing unit normal, so a positive error means the vehicle
        // sits to the left of the path.
        self.cross_track_error = (-dy / length) * (pose.x() - cx) + (dx / length) * (pose.y() - cy);
        self.heading_error = angle_difference(pose.heading(), dy.atan2(dx))?;
        Ok(())
    }

    /// The point on `path` about `lookahead_distance` ahead of the vehicle.
    ///
    /// Searches from the segment ending at the closest waypoint, so a
    /// vehicle sitting between two waypoints still finds the segment it is
    /// on. When no segment meets the lookahead circle, which happens once
    /// the vehicle has drifted further off the path than the circle
    /// reaches, the answer is the next waypoint forward rather than the
    /// goal: steering at the goal from off-track cuts every corner between
    /// here and there.
    #[must_use]
    pub fn lookahead_point(&self, pose: Pose, path: &[(f64, f64)], closest: usize) -> (f64, f64) {
        let first = closest.saturating_sub(1);
        for index in first..path.len().saturating_sub(1) {
            let Some(((ax, ay), (bx, by))) = pair(path, index, index.saturating_add(1)) else {
                continue;
            };
            if (bx - pose.x()).hypot(by - pose.y()) < self.lookahead_distance {
                continue;
            }
            if let Some(point) = circle_segment_intersection(
                (pose.x(), pose.y()),
                self.lookahead_distance,
                (ax, ay),
                (bx, by),
            ) {
                return point;
            }
        }
        let next = closest.saturating_add(1).min(path.len().saturating_sub(1));
        path.get(next).copied().unwrap_or((pose.x(), pose.y()))
    }
}

impl PathTracker for PurePursuitTracker {
    /// The speed and turn rate steering toward the lookahead point.
    ///
    /// The speed passes through untouched; pure pursuit is a steering law
    /// and has no opinion about how fast to go.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when the path holds fewer than two
    /// waypoints, and [`Error::NotFinite`] when the speed is not a real
    /// number.
    fn track(&mut self, pose: Pose, path: &[(f64, f64)], speed: f64) -> Result<Command, Error> {
        if !speed.is_finite() {
            return Err(Error::NotFinite {
                quantity: "speed",
                value: speed,
            });
        }
        if path.len() < 2 {
            return Err(Error::TooFew {
                quantity: "path waypoints",
                minimum: 2,
                actual: path.len(),
            });
        }
        for &(x, y) in path {
            if !(x.is_finite() && y.is_finite()) {
                return Err(Error::NotFinite {
                    quantity: "path waypoint",
                    value: if x.is_finite() { y } else { x },
                });
            }
        }

        let closest = Self::closest_index(pose, path);
        self.update_errors(pose, path, closest)?;

        let (lx, ly) = self.lookahead_point(pose, path, closest);
        let (dx, dy) = (lx - pose.x(), ly - pose.y());
        let (sine, cosine) = pose.heading().sin_cos();
        // The lookahead vector in the vehicle frame: ahead is positive x,
        // left is positive y.
        let ahead = cosine.mul_add(dx, sine * dy);
        let left = (-sine).mul_add(dx, cosine * dy);
        let alpha = left.atan2(ahead);

        self.curvature = 2.0 * alpha.sin() / self.lookahead_distance;
        Ok(Command {
            speed,
            turn_rate: speed * self.curvature,
        })
    }

    fn errors(&self) -> TrackerErrors {
        TrackerErrors {
            cross_track: self.cross_track_error,
            heading: self.heading_error,
            curvature: self.curvature,
        }
    }
}

/// Two waypoints by index, when both exist.
fn pair(path: &[(f64, f64)], first: usize, second: usize) -> Option<((f64, f64), (f64, f64))> {
    Some((*path.get(first)?, *path.get(second)?))
}

/// Where a circle meets a segment, farthest along the segment.
///
/// Farthest rather than nearest, because the lookahead point is meant to
/// be ahead: taking the near intersection steers at a point the vehicle
/// has already passed.
#[must_use]
pub fn circle_segment_intersection(
    center: (f64, f64),
    radius: f64,
    start: (f64, f64),
    end: (f64, f64),
) -> Option<(f64, f64)> {
    let (dx, dy) = (end.0 - start.0, end.1 - start.1);
    let (fx, fy) = (start.0 - center.0, start.1 - center.1);

    let a = dx.mul_add(dx, dy * dy);
    if a <= f64::EPSILON {
        return None;
    }
    let b = 2.0 * fx.mul_add(dx, fy * dy);
    let c = fx.mul_add(fx, fy.mul_add(fy, -(radius * radius)));
    let discriminant = b.mul_add(b, -(4.0 * a * c));
    if discriminant < 0.0 {
        return None;
    }

    let root = discriminant.sqrt();
    let far = (-b + root) / (2.0 * a);
    let near = (-b - root) / (2.0 * a);
    for ratio in [far, near] {
        if (0.0..=1.0).contains(&ratio) {
            return Some((dx.mul_add(ratio, start.0), dy.mul_add(ratio, start.1)));
        }
    }
    None
}
