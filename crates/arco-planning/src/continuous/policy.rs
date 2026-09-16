//! The policy hooks a sampling planner calls, as enums rather than objects.
//!
//! ADR-004. Every one of these is invoked inside the planner loop, some of
//! them millions of times per plan. Binding them as a caller-supplied
//! object means a virtual call at best and, from the Python side, an
//! interpreter lock acquisition per iteration, which gives back the entire
//! speedup the port exists for.
//!
//! So each hook is an enum with a native variant per built-in policy and
//! one variant holding something supplied from outside. The built-ins
//! never leave Rust; the escape hatch stays available and is documented as
//! slow.

use arco_core::Error;
use arco_core::geometry::require_dimension;
use arco_core::protocols::{Occupancy, PlannerCost, Sampler, SegmentChecker, Steerer};
use arco_core::rng::Pcg64;

/// How a planner measures distance between two states.
///
/// Every continuous planner in ARCO measures in units of its own step, so
/// that a tolerance, a rewiring radius and a witness cell all read in the
/// same currency whatever the axes happen to be scaled in.
pub enum CostPolicy {
    /// Euclidean distance after dividing each axis by its own scale.
    ///
    /// An axis with no entry in `step_size` is left at a scale of one, so
    /// an empty vector is plain Euclidean distance.
    Scaled {
        /// Per-axis scale, meters. One unit of distance is one step.
        step_size: Vec<f64>,
    },
    /// Anything the caller supplied.
    ///
    /// Replaces the `cost=` argument the Python planners accepted. Slower
    /// than the built-in by whatever the crossing costs.
    Custom(Box<dyn PlannerCost + Send + Sync>),
}

impl core::fmt::Debug for CostPolicy {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Scaled { step_size } => formatter
                .debug_struct("Scaled")
                .field("axes", &step_size.len())
                .finish(),
            Self::Custom(_) => formatter.write_str("Custom"),
        }
    }
}

impl Default for CostPolicy {
    fn default() -> Self {
        Self::Scaled {
            step_size: Vec::new(),
        }
    }
}

impl CostPolicy {
    /// The scale applied to `axis`, which is one where none was given.
    fn scale(step_size: &[f64], axis: usize) -> f64 {
        step_size.get(axis).copied().unwrap_or(1.0)
    }

    /// The scale this policy divides `axis` by, when it has one.
    ///
    /// A planner measuring a volume needs the scale to express it in the
    /// same units its distances are in. A custom cost keeps its metric to
    /// itself, so it answers `None` and the caller falls back.
    #[must_use]
    pub fn axis_scale(&self, axis: usize) -> Option<f64> {
        match self {
            Self::Scaled { step_size } => Some(Self::scale(step_size, axis)),
            Self::Custom(_) => None,
        }
    }
}

impl PlannerCost for CostPolicy {
    /// Distance from `from` to `to`, in steps.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree,
    /// [`Error::OutOfRange`] when a scale is not strictly positive, or
    /// whatever a custom cost returns.
    fn distance(&self, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        match self {
            Self::Scaled { step_size } => {
                require_dimension("target state", to, from.len())?;
                let mut total = 0.0_f64;
                for (axis, (start, end)) in from.iter().zip(to).enumerate() {
                    let scale = Self::scale(step_size, axis);
                    if !(scale.is_finite() && scale > 0.0) {
                        return Err(Error::OutOfRange {
                            quantity: "step size",
                            value: scale,
                            bound: "(0, inf)",
                        });
                    }
                    let normalized = (end - start) / scale;
                    total += normalized * normalized;
                }
                Ok(total.sqrt())
            }
            Self::Custom(cost) => cost.distance(from, to),
        }
    }

    /// A lower bound on the remaining cost, which is the distance itself.
    ///
    /// # Errors
    ///
    /// As [`CostPolicy::distance`].
    fn heuristic(&self, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        match self {
            Self::Scaled { .. } => self.distance(from, to),
            Self::Custom(cost) => cost.heuristic(from, to),
        }
    }
}

/// Where new states come from.
pub enum SamplerPolicy {
    /// Uniform over an axis-aligned box.
    UniformBox {
        /// Inclusive lower and exclusive upper bound per axis.
        bounds: Vec<(f64, f64)>,
    },
    /// Anything the caller supplied.
    ///
    /// Slower than the built-ins by whatever the crossing costs, which
    /// from Python is an interpreter lock acquisition per call.
    Custom(Box<dyn Sampler + Send + Sync>),
}

impl core::fmt::Debug for SamplerPolicy {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UniformBox { bounds } => formatter
                .debug_struct("UniformBox")
                .field("axes", &bounds.len())
                .finish(),
            Self::Custom(_) => formatter.write_str("Custom"),
        }
    }
}

impl SamplerPolicy {
    /// Draws one state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when the bounds are empty, or whatever a
    /// custom sampler returns.
    pub fn sample(&self, generator: &mut Pcg64) -> Result<Vec<f64>, Error> {
        match self {
            Self::UniformBox { bounds } => {
                if bounds.is_empty() {
                    return Err(Error::TooFew {
                        quantity: "sampling bounds",
                        minimum: 1,
                        actual: 0,
                    });
                }
                Ok(bounds
                    .iter()
                    .map(|&(low, high)| low + generator.next_f64() * (high - low))
                    .collect())
            }
            Self::Custom(sampler) => sampler.sample(generator),
        }
    }

    /// The dimension this policy samples in, when it knows.
    #[must_use]
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::UniformBox { bounds } => Some(bounds.len()),
            Self::Custom(_) => None,
        }
    }
}

/// How the tree grows toward a sample.
pub enum SteererPolicy {
    /// Straight toward the target, capped at one step.
    Straight {
        /// Per-axis scale of one step, meters.
        ///
        /// The same vector the planner measures distance with, so that a
        /// step of one and a distance of one mean the same thing.
        step_size: Vec<f64>,
    },
    /// Anything the caller supplied.
    Custom(Box<dyn Steerer + Send + Sync>),
}

impl core::fmt::Debug for SteererPolicy {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Straight { step_size } => formatter
                .debug_struct("Straight")
                .field("axes", &step_size.len())
                .finish(),
            Self::Custom(_) => formatter.write_str("Custom"),
        }
    }
}

impl SteererPolicy {
    /// Steers from `from` toward `to` by at most one step.
    ///
    /// The step is one unit in the space scaled by `step_size`, so a
    /// diagonal move is bounded by the same step every axis is, rather
    /// than by each axis independently. Capping the axes one at a time
    /// would let a diagonal edge be longer than the step by a factor of
    /// the square root of the dimension, which is the whole reason the
    /// distance is normalized in the first place.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree, or
    /// whatever a custom steerer returns.
    pub fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<f64>, Error> {
        match self {
            Self::Straight { step_size } => {
                require_dimension("target", to, from.len())?;
                let metric = CostPolicy::Scaled {
                    step_size: step_size.clone(),
                };
                let distance = metric.distance(from, to)?;
                if distance <= 1.0 {
                    return Ok(to.to_vec());
                }
                Ok(from
                    .iter()
                    .zip(to)
                    .map(|(start, end)| start + (end - start) / distance)
                    .collect())
            }
            Self::Custom(steerer) => steerer.steer(from, to),
        }
    }
}

/// How an edge is checked for collision.
pub enum SegmentPolicy<O> {
    /// Ask the occupancy for the exact answer.
    ///
    /// The only policy under which `FR-INV-01` holds against a re-check
    /// finer than the planning resolution, because it has no resolution.
    Exact {
        /// The occupancy to query.
        occupancy: O,
    },
    /// Sample the segment a fixed number of times.
    ///
    /// What the Python planners did, and what they were tuned against.
    /// Not exact: an obstacle the segment only grazes occludes a span
    /// that shrinks to nothing as the contact gets shallower, so no fixed
    /// sample count catches every one of them. Cheaper than [`Exact`] on
    /// a dense field, and a returned path carries the resolution it was
    /// checked at rather than a promise it cannot make.
    ///
    /// [`Exact`]: SegmentPolicy::Exact
    Sampled {
        /// The occupancy to query.
        occupancy: O,
        /// Samples per segment, including both endpoints.
        count: usize,
    },
    /// Anything the caller supplied.
    Custom(Box<dyn SegmentChecker + Send + Sync>),
}

impl<O> core::fmt::Debug for SegmentPolicy<O> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Exact { .. } => formatter.write_str("Exact"),
            Self::Sampled { count, .. } => formatter
                .debug_struct("Sampled")
                .field("count", count)
                .finish(),
            Self::Custom(_) => formatter.write_str("Custom"),
        }
    }
}

impl<O: Occupancy> SegmentPolicy<O> {
    /// Whether the segment between two states is free.
    ///
    /// # Errors
    ///
    /// Returns whatever the occupancy or the custom checker returns.
    pub fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        match self {
            Self::Exact { occupancy } => occupancy.is_segment_free(from, to),
            Self::Sampled { occupancy, count } => {
                let steps = (*count).max(2);
                let divisor = f64::from(u32::try_from(steps.saturating_sub(1)).unwrap_or(1));
                let mut sample = vec![0.0; from.len()];
                for step in 0..steps {
                    let ratio = f64::from(u32::try_from(step).unwrap_or(0)) / divisor;
                    for ((slot, start), end) in sample.iter_mut().zip(from).zip(to) {
                        *slot = start + (end - start) * ratio;
                    }
                    if occupancy.is_occupied(&sample)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Custom(checker) => checker.is_segment_free(from, to),
        }
    }

    /// The samples per segment this policy validates at, if it has a limit.
    ///
    /// `FR-INV-01` asks that a result carry the resolution it was checked
    /// at, so that a caller re-checking a path knows what it is allowed to
    /// conclude. `None` means exact, and nothing finer exists.
    #[must_use]
    pub const fn validity_samples(&self) -> Option<usize> {
        match *self {
            Self::Exact { .. } => None,
            Self::Sampled { count, .. } => Some(count),
            // A custom checker knows its own resolution and this type does
            // not, so the conservative answer is the coarsest one.
            Self::Custom(_) => Some(2),
        }
    }
}
