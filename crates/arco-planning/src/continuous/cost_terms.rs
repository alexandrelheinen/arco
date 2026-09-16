//! The composite cost a trajectory optimizer minimizes, term by term.
//!
//! Five terms, the same five the Python optimizer summed, as one enum
//! rather than five objects. The reasoning is ADR-004's: every term is
//! evaluated once per cost call and a cost call happens tens of times per
//! finite-difference gradient, so a caller-supplied object on that path
//! costs more than the term itself.

use arco_core::Error;
use arco_core::geometry::euclidean_distance;
use arco_core::protocols::{CostTerm, Occupancy};

/// What every cost term reads.
///
/// Built once per cost evaluation and shared by every term, because the
/// segment lengths and implied speeds are what most of them are about and
/// recomputing them per term would cost more than the terms do.
pub struct TrajectoryContext<'a> {
    /// Segment durations, seconds, one per segment.
    pub durations: &'a [f64],
    /// The waypoints under evaluation, one more than there are segments.
    pub waypoints: &'a [Vec<f64>],
    /// The reference path the waypoints started from, same length.
    pub reference: &'a [Vec<f64>],
    /// Segment lengths, meters, one per segment.
    pub lengths: &'a [f64],
    /// Implied segment speeds, meters per second, one per segment.
    pub speeds: &'a [f64],
    /// The map the collision term queries.
    pub occupancy: &'a dyn Occupancy,
}

impl core::fmt::Debug for TrajectoryContext<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TrajectoryContext")
            .field("segments", &self.durations.len())
            .field("dimension", &self.occupancy.dimension())
            .finish()
    }
}

impl TrajectoryContext<'_> {
    /// The total traversal time, seconds.
    #[must_use]
    pub fn total_duration(&self) -> f64 {
        self.durations.iter().sum()
    }
}

/// One term of the trajectory cost.
///
/// The weights are the ones the Python optimizer took as constructor
/// arguments, kept at the same names and the same meanings so a
/// configuration file written for one works for the other.
pub enum TrajectoryTerm {
    /// Penalizes total traversal time squared.
    ///
    /// The term the optimizer exists to reduce. Alone it drives the total
    /// time to zero, which [`TrajectoryTerm::Velocity`] is what stops.
    Time {
        /// Multiplier on the squared total duration.
        weight: f64,
    },
    /// Penalizes squared distance from the reference path.
    ///
    /// Keeps the optimized path near the one the planner proved free,
    /// which matters because the collision term below is soft.
    Deviation {
        /// Multiplier on the summed squared offsets.
        weight: f64,
    },
    /// Penalizes squared departure from the cruise speed.
    Velocity {
        /// Multiplier on the summed squared speed errors.
        weight: f64,
        /// The speed the trajectory aims to hold, meters per second.
        cruise_speed: f64,
    },
    /// Penalizes penetration into the clearance around an obstacle.
    ///
    /// Two pieces: a quadratic penalty that grows from first contact, and
    /// a barrier that grows much faster so that deep penetration is
    /// effectively refused while the objective stays smooth. A hard
    /// constraint would be more honest and would also stop the solver
    /// dead at the first infeasible step, which is why it is not one.
    Collision {
        /// Multiplier on both pieces.
        weight: f64,
        /// Extra multiplier on the barrier piece.
        barrier_scale: f64,
        /// Exponent on normalized penetration, above two to bite late.
        barrier_power: f64,
        /// Points sampled inside each segment, beyond its endpoints.
        sample_count: usize,
    },
    /// Penalizes implied speeds outside the vehicle's band.
    ///
    /// A soft stand-in for a hard bound. The trajectory is checked
    /// against the same bounds afterward, and the result says whether it
    /// passed, because a penalty that can be paid is not a limit.
    Dynamics {
        /// Multiplier on the summed squared violations.
        weight: f64,
        /// Upper speed bound, meters per second, when there is one.
        max_speed: Option<f64>,
        /// Lower speed bound, meters per second, when there is one.
        min_speed: Option<f64>,
    },
    /// Anything the caller supplied.
    ///
    /// Replaces the `cost_terms=` argument. Slower than the built-ins by
    /// whatever the crossing costs, per deviation A-07.
    Custom(Box<dyn for<'a> CostTerm<TrajectoryContext<'a>> + Send + Sync>),
}

impl core::fmt::Debug for TrajectoryTerm {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match *self {
            Self::Time { .. } => "Time",
            Self::Deviation { .. } => "Deviation",
            Self::Velocity { .. } => "Velocity",
            Self::Collision { .. } => "Collision",
            Self::Dynamics { .. } => "Dynamics",
            Self::Custom(_) => "Custom",
        };
        formatter.write_str(name)
    }
}

impl TrajectoryTerm {
    /// The five terms the Python optimizer built when given no others.
    ///
    /// # Arguments
    ///
    /// * `weights` - The five multipliers, in the order they are summed.
    /// * `cruise_speed` - Target speed, meters per second.
    /// * `barrier` - Scale and exponent of the collision barrier.
    /// * `speed_band` - Upper and lower speed bounds, when they exist.
    /// * `sample_count` - Points sampled inside each segment.
    #[must_use]
    pub fn defaults(
        weights: TermWeights,
        cruise_speed: f64,
        barrier: (f64, f64),
        speed_band: (Option<f64>, Option<f64>),
        sample_count: usize,
    ) -> Vec<Self> {
        vec![
            Self::Time {
                weight: weights.time,
            },
            Self::Deviation {
                weight: weights.deviation,
            },
            Self::Velocity {
                weight: weights.velocity,
                cruise_speed,
            },
            Self::Collision {
                weight: weights.collision,
                barrier_scale: barrier.0,
                barrier_power: barrier.1,
                sample_count,
            },
            Self::Dynamics {
                weight: weights.dynamics,
                max_speed: speed_band.0,
                min_speed: speed_band.1,
            },
        ]
    }
}

/// The five multipliers of the default composite cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermWeights {
    /// On total time squared.
    pub time: f64,
    /// On departure from the reference path.
    pub deviation: f64,
    /// On departure from the cruise speed.
    pub velocity: f64,
    /// On clearance penetration.
    pub collision: f64,
    /// On speed-bound violation.
    pub dynamics: f64,
}

impl Default for TermWeights {
    fn default() -> Self {
        Self {
            time: 10.0,
            deviation: 1.0,
            velocity: 1.0,
            collision: 5.0,
            dynamics: 100.0,
        }
    }
}

impl CostTerm<TrajectoryContext<'_>> for TrajectoryTerm {
    fn evaluate(&self, context: &TrajectoryContext<'_>) -> Result<f64, Error> {
        let value = match self {
            Self::Time { weight } => {
                let total = context.total_duration();
                weight * total * total
            }
            Self::Deviation { weight } => weight * squared_offset(context)?,
            Self::Velocity {
                weight,
                cruise_speed,
            } => {
                weight
                    * context
                        .speeds
                        .iter()
                        .map(|speed| (speed - cruise_speed).powi(2))
                        .sum::<f64>()
            }
            Self::Collision {
                weight,
                barrier_scale,
                barrier_power,
                sample_count,
            } => collision(
                context,
                *weight,
                (*barrier_scale, *barrier_power),
                *sample_count,
            )?,
            Self::Dynamics {
                weight,
                max_speed,
                min_speed,
            } => weight * band_violation(context.speeds, *max_speed, *min_speed),
            Self::Custom(term) => term.evaluate(context)?,
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NotFinite {
                quantity: "trajectory cost term",
                value,
            })
        }
    }
}

/// The summed squared offset of the interior waypoints from the reference.
///
/// Interior only: the endpoints are fixed to the reference and contribute
/// nothing, so including them would add a constant zero and a bounds
/// check per call.
fn squared_offset(context: &TrajectoryContext<'_>) -> Result<f64, Error> {
    let interior = context.waypoints.len().saturating_sub(1);
    let mut total = 0.0;
    for index in 1..interior {
        let (Some(moved), Some(original)) =
            (context.waypoints.get(index), context.reference.get(index))
        else {
            continue;
        };
        let offset = euclidean_distance(moved, original)?;
        total += offset * offset;
    }
    Ok(total)
}

/// The clearance penalty plus its barrier.
fn collision(
    context: &TrajectoryContext<'_>,
    weight: f64,
    barrier: (f64, f64),
    sample_count: usize,
) -> Result<f64, Error> {
    let clearance = context.occupancy.clearance();
    let mut quadratic = 0.0;
    let mut barrier_total = 0.0;
    let scale = clearance.max(f64::EPSILON);

    let mut accumulate = |point: &[f64]| -> Result<(), Error> {
        let free = context.occupancy.nearest_obstacle(point)?.distance;
        // `distance` is already measured from the obstacle surface, so a
        // negative value is the depth of the penetration.
        let penetration = (-free).max(0.0);
        if penetration > 0.0 {
            quadratic += penetration * penetration;
            barrier_total += (penetration / scale).powf(barrier.1);
        }
        Ok(())
    };

    let interior = context.waypoints.len().saturating_sub(1);
    for index in 1..interior {
        if let Some(point) = context.waypoints.get(index) {
            accumulate(point)?;
        }
    }

    if sample_count > 0 {
        let divisor = f64::from(u32::try_from(sample_count.saturating_add(1)).unwrap_or(u32::MAX));
        for pair in context.waypoints.windows(2) {
            let [from, to] = pair else { continue };
            for step in 1..=sample_count {
                let ratio = f64::from(u32::try_from(step).unwrap_or(1)) / divisor;
                let sample: Vec<f64> = from
                    .iter()
                    .zip(to)
                    .map(|(start, end)| start + (end - start) * ratio)
                    .collect();
                accumulate(&sample)?;
            }
        }
    }

    Ok(weight.mul_add(quadratic, weight * barrier.0 * barrier_total))
}

/// The summed squared amount by which speeds leave their band.
fn band_violation(speeds: &[f64], max_speed: Option<f64>, min_speed: Option<f64>) -> f64 {
    if max_speed.is_none() && min_speed.is_none() {
        return 0.0;
    }
    speeds
        .iter()
        .map(|&speed| {
            let over = max_speed.map_or(0.0, |limit| (speed - limit).max(0.0));
            let under = min_speed.map_or(0.0, |limit| (limit - speed).max(0.0));
            over.mul_add(over, under * under)
        })
        .sum()
}
