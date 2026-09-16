//! Tracking a configuration through a space with per-axis limits.
//!
//! The counterpart to the planar tracking loop for anything whose state is
//! a vector of joint values rather than a pose: arms, gantries, and any
//! body controlled in its own configuration space. Proportional position
//! to velocity, saturated per axis in both velocity and acceleration, with
//! an optional repulsion term that pushes the configuration away from the
//! nearest obstacle in that same space.

use arco_core::Error;
use arco_core::protocols::Occupancy;

use crate::limits::IntervalBand;

/// Per-axis limits on how fast a configuration may move and change.
#[derive(Debug, Clone, PartialEq)]
pub struct JointLimits {
    /// Largest speed along each axis, configuration units per second.
    pub max_velocity: Vec<f64>,
    /// Largest acceleration along each axis, units per second squared.
    pub max_acceleration: Vec<f64>,
}

impl JointLimits {
    /// Builds limits applying the same bounds to every axis.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when there are no axes, and
    /// [`Error::OutOfRange`] when a bound is not finite and strictly
    /// positive.
    pub fn uniform(axes: usize, max_velocity: f64, max_acceleration: f64) -> Result<Self, Error> {
        Self::new(vec![max_velocity; axes], vec![max_acceleration; axes])
    }

    /// Builds per-axis limits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when there are no axes,
    /// [`Error::DimensionMismatch`] when the two vectors disagree, and
    /// [`Error::OutOfRange`] when a bound is not finite and strictly
    /// positive.
    pub fn new(max_velocity: Vec<f64>, max_acceleration: Vec<f64>) -> Result<Self, Error> {
        if max_velocity.is_empty() {
            return Err(Error::TooFew {
                quantity: "axes",
                minimum: 1,
                actual: 0,
            });
        }
        if max_velocity.len() != max_acceleration.len() {
            return Err(Error::DimensionMismatch {
                quantity: "acceleration limits",
                expected: max_velocity.len(),
                actual: max_acceleration.len(),
            });
        }
        for (quantity, bounds) in [
            ("maximum velocity", &max_velocity),
            ("maximum acceleration", &max_acceleration),
        ] {
            for &value in bounds {
                if !(value.is_finite() && value > 0.0) {
                    return Err(Error::OutOfRange {
                        quantity,
                        value,
                        bound: "(0, inf)",
                    });
                }
            }
        }
        Ok(Self {
            max_velocity,
            max_acceleration,
        })
    }

    /// How many axes these limits describe.
    #[must_use]
    pub fn axes(&self) -> usize {
        self.max_velocity.len()
    }
}

/// How a joint-space tracker should behave.
#[derive(Debug, Clone, PartialEq)]
pub struct JointTrackerSettings {
    /// The limits every axis obeys.
    pub limits: JointLimits,
    /// Position error to commanded velocity, per second.
    pub proportional_gain: f64,
    /// How hard to push away from an obstacle, units squared per second.
    ///
    /// Zero disables repulsion, which is the default and what a tracker
    /// given no map does regardless.
    pub repulsion_gain: f64,
    /// What elapsed interval a step will accept.
    pub interval: IntervalBand,
}

impl JointTrackerSettings {
    /// Builds settings around `limits`, with repulsion off.
    #[must_use]
    pub fn new(limits: JointLimits) -> Self {
        Self {
            limits,
            proportional_gain: 2.0,
            repulsion_gain: 0.0,
            interval: IntervalBand::default(),
        }
    }
}

/// What one tracker step did.
#[derive(Debug, Clone, PartialEq)]
pub struct JointStep {
    /// The configuration after the step.
    pub configuration: Vec<f64>,
    /// The velocity after the step, units per second.
    pub velocity: Vec<f64>,
    /// Axes whose velocity limit bit this step.
    pub velocity_saturated: usize,
    /// Axes whose acceleration limit bit this step.
    pub acceleration_saturated: usize,
}

/// A proportional tracker over a configuration space.
#[derive(Debug, Clone)]
pub struct JointSpaceTracker<O> {
    settings: JointTrackerSettings,
    occupancy: Option<O>,
    configuration: Vec<f64>,
    velocity: Vec<f64>,
}

impl<O: Occupancy> JointSpaceTracker<O> {
    /// Builds a tracker resting at the origin of its configuration space.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a gain is not a real number, and
    /// [`Error::OutOfRange`] when the repulsion gain is negative.
    pub fn new(settings: JointTrackerSettings, occupancy: Option<O>) -> Result<Self, Error> {
        for (quantity, value) in [
            ("proportional gain", settings.proportional_gain),
            ("repulsion gain", settings.repulsion_gain),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        if settings.repulsion_gain < 0.0 {
            return Err(Error::OutOfRange {
                quantity: "repulsion gain",
                value: settings.repulsion_gain,
                bound: "[0, inf)",
            });
        }
        let axes = settings.limits.axes();
        Ok(Self {
            settings,
            occupancy,
            configuration: vec![0.0; axes],
            velocity: vec![0.0; axes],
        })
    }

    /// How many axes the tracker moves.
    #[must_use]
    pub fn axes(&self) -> usize {
        self.settings.limits.axes()
    }

    /// The current configuration.
    #[must_use]
    pub fn configuration(&self) -> &[f64] {
        &self.configuration
    }

    /// The current velocity, configuration units per second.
    #[must_use]
    pub fn velocity(&self) -> &[f64] {
        &self.velocity
    }

    /// Places the tracker at `configuration` and stops it.
    ///
    /// Called before the first step of a trajectory and after any
    /// replanning, since the velocity carried over from the old trajectory
    /// is about a path that no longer exists.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the axis count is wrong,
    /// and [`Error::NotFinite`] when a value is not a real number.
    pub fn reset(&mut self, configuration: &[f64]) -> Result<(), Error> {
        self.require_axes("initial configuration", configuration)?;
        self.configuration.clear();
        self.configuration.extend_from_slice(configuration);
        self.velocity.iter_mut().for_each(|value| *value = 0.0);
        Ok(())
    }

    /// Moves one step toward `target`.
    ///
    /// The order is proportional command, then repulsion, then the
    /// velocity clamp again, then the acceleration clamp, then integrate.
    /// Clamping before the repulsion would let the repulsion push the
    /// command back outside the envelope, which is how an avoidance term
    /// commands a velocity no axis can reach.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `target` has the wrong
    /// axis count, [`Error::NotFinite`] when a value is not a real number,
    /// [`Error::OutOfRange`] when `dt` leaves the configured band per
    /// `FR-INV-10`, and otherwise whatever the occupancy returns.
    pub fn step(&mut self, target: &[f64], dt: f64) -> Result<JointStep, Error> {
        self.settings.interval.check(dt)?;
        self.require_axes("target configuration", target)?;

        let repulsion = self.repulsion_velocity()?;
        let mut velocity_saturated = 0_usize;
        let mut acceleration_saturated = 0_usize;

        for index in 0..self.axes() {
            let limit = self.limit(index);
            let position = self.configuration.get(index).copied().unwrap_or_default();
            let goal = target.get(index).copied().unwrap_or_default();
            let bias = repulsion.get(index).copied().unwrap_or_default();

            let commanded = (self.settings.proportional_gain * (goal - position))
                .clamp(-limit.velocity, limit.velocity);
            let desired = (commanded + bias).clamp(-limit.velocity, limit.velocity);
            if (desired.abs() - limit.velocity).abs() <= f64::EPSILON * limit.velocity {
                velocity_saturated = velocity_saturated.saturating_add(1);
            }

            let current = self.velocity.get(index).copied().unwrap_or_default();
            let allowance = limit.acceleration * dt;
            let change = (desired - current).clamp(-allowance, allowance);
            if (desired - current).abs() > allowance {
                acceleration_saturated = acceleration_saturated.saturating_add(1);
            }

            let updated = (current + change).clamp(-limit.velocity, limit.velocity);
            if let Some(slot) = self.velocity.get_mut(index) {
                *slot = updated;
            }
            if let Some(slot) = self.configuration.get_mut(index) {
                *slot = updated.mul_add(dt, position);
            }
        }

        Ok(JointStep {
            configuration: self.configuration.clone(),
            velocity: self.velocity.clone(),
            velocity_saturated,
            acceleration_saturated,
        })
    }

    /// The repulsion velocity at the current configuration.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy returns.
    fn repulsion_velocity(&self) -> Result<Vec<f64>, Error> {
        let zero = vec![0.0; self.axes()];
        let Some(occupancy) = self.occupancy.as_ref() else {
            return Ok(zero);
        };
        if self.settings.repulsion_gain <= 0.0 {
            return Ok(zero);
        }
        let clearance = occupancy.clearance();
        if clearance <= 0.0 {
            return Ok(zero);
        }

        let influence = 2.0 * clearance;
        let nearest = occupancy.nearest_obstacle(&self.configuration)?;
        // Measured from the surface here and from the center in the
        // Python, per deviation A-16, so the clearance goes back on.
        let distance = nearest.distance + clearance;
        if distance >= influence {
            return Ok(zero);
        }

        // Floored well inside the clearance: the reciprocal runs away as
        // the obstacle is approached, and an infinite velocity command is
        // not a useful answer.
        let safe = distance.max(0.1 * clearance);
        let magnitude = self.settings.repulsion_gain * (1.0 / safe - 1.0 / influence);
        Ok(self
            .configuration
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let obstacle = nearest.point.get(index).copied().unwrap_or_default();
                magnitude * (value - obstacle) / safe
            })
            .collect())
    }

    /// The limits on one axis.
    fn limit(&self, index: usize) -> AxisLimit {
        AxisLimit {
            velocity: self
                .settings
                .limits
                .max_velocity
                .get(index)
                .copied()
                .unwrap_or(f64::INFINITY),
            acceleration: self
                .settings
                .limits
                .max_acceleration
                .get(index)
                .copied()
                .unwrap_or(f64::INFINITY),
        }
    }

    /// Rejects a configuration of the wrong width or carrying a NaN.
    fn require_axes(&self, quantity: &'static str, values: &[f64]) -> Result<(), Error> {
        if values.len() != self.axes() {
            return Err(Error::DimensionMismatch {
                quantity,
                expected: self.axes(),
                actual: values.len(),
            });
        }
        for &value in values {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        Ok(())
    }
}

/// The two bounds on one axis.
#[derive(Debug, Clone, Copy)]
struct AxisLimit {
    velocity: f64,
    acceleration: f64,
}
