//! A reactive turn-rate bias that steers away from the nearest obstacle.

use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::protocols::{AvoidanceStrategy, Occupancy};

/// An artificial potential field acting on the turn rate alone.
///
/// Inside an influence radius of twice the clearance, the bias is
/// `gain (1/d - 1/d_max)` turned away from the obstacle, where `d` is the
/// distance to it. The magnitude goes to zero at the edge of the influence
/// radius, so the correction switches on smoothly rather than stepping.
///
/// Reactive and local by construction: it sees one obstacle and it cannot
/// reason about whether turning away leads anywhere. It is a last defense
/// layered on a planner that already produced a free path, not a
/// substitute for one.
#[derive(Debug, Clone)]
pub struct ArtificialPotentialField<O> {
    occupancy: Option<O>,
    repulsion_gain: f64,
}

impl<O: Occupancy> ArtificialPotentialField<O> {
    /// Builds a field repelling from `occupancy` at `repulsion_gain`.
    ///
    /// A gain at or below zero disables the field, which is how the
    /// Python default behaved and what keeps a tracking loop that never
    /// asked for avoidance unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when the gain is not a real number.
    pub fn new(occupancy: O, repulsion_gain: f64) -> Result<Self, Error> {
        if !repulsion_gain.is_finite() {
            return Err(Error::NotFinite {
                quantity: "repulsion gain",
                value: repulsion_gain,
            });
        }
        Ok(Self {
            occupancy: Some(occupancy),
            repulsion_gain,
        })
    }

    /// Builds a field that never biases anything.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            occupancy: None,
            repulsion_gain: 0.0,
        }
    }

    /// The gain, radians per second per meter.
    #[must_use]
    pub const fn repulsion_gain(&self) -> f64 {
        self.repulsion_gain
    }
}

impl<O: Occupancy> AvoidanceStrategy for ArtificialPotentialField<O> {
    /// The turn-rate bias at `pose`, radians per second.
    ///
    /// Positive turns left. An obstacle to the left produces a negative
    /// bias and an obstacle to the right a positive one, so the vehicle
    /// turns away from it either way.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy returns.
    fn turn_rate_bias(&self, pose: Pose) -> Result<f64, Error> {
        let Some(occupancy) = self.occupancy.as_ref() else {
            return Ok(0.0);
        };
        if self.repulsion_gain <= 0.0 {
            return Ok(0.0);
        }
        let clearance = occupancy.clearance();
        if clearance <= 0.0 {
            return Ok(0.0);
        }

        let influence_radius = 2.0 * clearance;
        let nearest = occupancy.nearest_obstacle(&[pose.x(), pose.y()])?;
        // `nearest_obstacle` measures from the obstacle surface, and the
        // field is defined against the distance to its center, which is
        // what the influence radius is expressed in.
        let distance = nearest.distance + clearance;
        if distance >= influence_radius || distance < 1e-6 {
            return Ok(0.0);
        }

        let (sine, cosine) = pose.heading().sin_cos();
        let dx = nearest.point.first().copied().unwrap_or_default() - pose.x();
        let dy = nearest.point.get(1).copied().unwrap_or_default() - pose.y();
        // Positive where the obstacle lies to the vehicle's left.
        let lateral = (-sine).mul_add(dx, cosine * dy);

        // The distance is floored well inside the clearance, because the
        // reciprocal runs away as the obstacle is approached and a bias of
        // ten thousand radians per second is not a useful answer.
        let floor = 0.1 * clearance;
        let magnitude = self.repulsion_gain * (1.0 / distance.max(floor) - 1.0 / influence_radius);
        // A lateral of exactly zero means the obstacle is dead ahead or
        // behind, where the sign is arbitrary; `copysign` picks left
        // consistently so the answer stays reproducible.
        Ok(-magnitude * 1.0_f64.copysign(lateral))
    }
}
