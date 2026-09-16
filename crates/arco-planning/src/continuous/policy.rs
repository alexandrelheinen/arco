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
use arco_core::geometry::euclidean_distance;
use arco_core::protocols::{Occupancy, Sampler, SegmentChecker, Steerer};
use arco_core::rng::Pcg64;

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
    /// Straight toward the target, capped at a per-axis step.
    Straight {
        /// Maximum step along each axis, meters.
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
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree, or
    /// whatever a custom steerer returns.
    pub fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<f64>, Error> {
        match self {
            Self::Straight { step_size } => {
                arco_core::geometry::require_dimension("target", to, from.len())?;
                let distance = euclidean_distance(from, to)?;
                if distance == 0.0 {
                    return Ok(from.to_vec());
                }
                // The cap is the tightest per-axis limit expressed as a
                // fraction of the straight-line distance, so an anisotropic
                // step never exceeds any of its own axis limits.
                let mut scale = 1.0_f64;
                for (axis, (start, end)) in from.iter().zip(to).enumerate() {
                    let delta = (end - start).abs();
                    let limit = step_size.get(axis).copied().unwrap_or(f64::INFINITY);
                    if delta > limit && delta > 0.0 {
                        scale = scale.min(limit / delta);
                    }
                }
                Ok(from
                    .iter()
                    .zip(to)
                    .map(|(start, end)| start + (end - start) * scale)
                    .collect())
            }
            Self::Custom(steerer) => steerer.steer(from, to),
        }
    }
}

/// How an edge is checked for collision.
pub enum SegmentPolicy<O> {
    /// Sample the segment a fixed number of times.
    ///
    /// Not exact: a thin obstacle between two samples is missed, which is
    /// why `FR-INV-01` re-checks a returned path at a stated resolution
    /// rather than trusting this.
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
}
