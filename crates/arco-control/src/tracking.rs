//! Closing the loop: a tracker, a vehicle, and the limits between them.

use std::collections::VecDeque;

use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::protocols::{AvoidanceStrategy, Command, PathTracker, VehicleModel};

use crate::limits::{CommandConditioner, CommandLimits, SaturationReport};

/// One step's worth of what happened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingSample {
    /// Signed distance from the path, meters, positive to its left.
    pub cross_track_error: f64,
    /// Heading minus path tangent, radians.
    pub heading_error: f64,
    /// The vehicle pose after the step.
    pub pose: Pose,
    /// The vehicle speed after the step, meters per second.
    pub speed: f64,
    /// The vehicle turn rate after the step, radians per second.
    pub turn_rate: f64,
    /// The curvature the tracker asked for, per meter.
    pub curvature: f64,
    /// The turn-rate bias avoidance added, radians per second.
    pub avoidance_bias: f64,
    /// What the tracker asked for before the limits.
    pub requested: Command,
    /// What the vehicle was given.
    pub applied: Command,
}

/// How a tracking loop should behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingSettings {
    /// The speed to hold on a straight path, meters per second.
    pub cruise_speed: f64,
    /// How much to slow for curvature, meters.
    ///
    /// The commanded speed is `cruise / (1 + gain * |curvature|)`, using
    /// the curvature the tracker reported on the previous step, because
    /// the current one is not known until after the speed is chosen. Zero
    /// holds the cruise speed regardless.
    pub curvature_gain: f64,
    /// The limits every command passes, per deviation A-09.
    pub limits: CommandLimits,
    /// How many samples to keep, or `None` for all of them.
    ///
    /// `None` matches the Python loop, whose history grew without bound.
    /// That is the wrong default for anything running longer than a
    /// simulation, so a caller with a real-time budget sets a capacity and
    /// gets a ring buffer; `Some(0)` keeps none at all, since
    /// [`TrackingLoop::step`] returns the sample anyway.
    pub history_capacity: Option<usize>,
}

impl Default for TrackingSettings {
    fn default() -> Self {
        Self {
            cruise_speed: 1.0,
            curvature_gain: 0.0,
            limits: CommandLimits::default(),
            history_capacity: None,
        }
    }
}

/// A vehicle driven along a path by a tracker.
///
/// The order within a step is fixed and the reason is worth stating:
/// the tracker asks for a command, avoidance biases the turn rate, then
/// the limits apply, then the vehicle integrates. Limiting before the
/// avoidance bias would let the bias push the command back outside the
/// box, which is how an avoidance term ends up commanding a turn rate no
/// actuator can produce.
#[derive(Debug)]
pub struct TrackingLoop<V, T, A> {
    vehicle: V,
    tracker: T,
    avoidance: A,
    settings: TrackingSettings,
    conditioner: CommandConditioner,
    history: VecDeque<TrackingSample>,
}

impl<V: VehicleModel, T: PathTracker, A: AvoidanceStrategy> TrackingLoop<V, T, A> {
    /// Builds a loop from its parts.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when the cruise speed or the curvature
    /// gain is not a real number, [`Error::OutOfRange`] when the curvature
    /// gain is negative, and otherwise as [`CommandConditioner::new`].
    pub fn new(
        vehicle: V,
        tracker: T,
        avoidance: A,
        settings: TrackingSettings,
    ) -> Result<Self, Error> {
        for (quantity, value) in [
            ("cruise speed", settings.cruise_speed),
            ("curvature gain", settings.curvature_gain),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        if settings.curvature_gain < 0.0 {
            return Err(Error::OutOfRange {
                quantity: "curvature gain",
                value: settings.curvature_gain,
                bound: "[0, inf)",
            });
        }
        Ok(Self {
            vehicle,
            tracker,
            avoidance,
            settings,
            conditioner: CommandConditioner::new(settings.limits)?,
            history: VecDeque::new(),
        })
    }

    /// The vehicle being driven.
    pub const fn vehicle(&self) -> &V {
        &self.vehicle
    }

    /// The vehicle, mutably.
    ///
    /// A vehicle wrapping something outside this crate may need to be
    /// re-read before a step, which it cannot do through `&self`.
    pub const fn vehicle_mut(&mut self) -> &mut V {
        &mut self.vehicle
    }

    /// The tracker, mutably, for the reason [`TrackingLoop::vehicle_mut`] is.
    pub const fn tracker_mut(&mut self) -> &mut T {
        &mut self.tracker
    }

    /// The tracker producing commands.
    pub const fn tracker(&self) -> &T {
        &self.tracker
    }

    /// What the limits have clipped so far.
    #[must_use]
    pub const fn saturation(&self) -> SaturationReport {
        self.conditioner.report()
    }

    /// The samples kept, oldest first.
    pub fn history(&self) -> impl Iterator<Item = &TrackingSample> {
        self.history.iter()
    }

    /// The most recent sample, if any step has run.
    #[must_use]
    pub fn last(&self) -> Option<&TrackingSample> {
        self.history.back()
    }

    /// Clears the history, the saturation counters and the rate limiter.
    pub fn reset(&mut self) {
        self.history.clear();
        self.conditioner.reset();
    }

    /// Runs one tracking step along `path`.
    ///
    /// # Errors
    ///
    /// Returns whatever the tracker, the avoidance strategy, the limits or
    /// the vehicle return, including [`Error::OutOfRange`] when `dt`
    /// leaves the configured interval band.
    pub fn step(&mut self, path: &[(f64, f64)], dt: f64) -> Result<TrackingSample, Error> {
        let pose = self.vehicle.pose();
        let previous_curvature = self.tracker.errors().curvature;
        let speed_reference = self.settings.cruise_speed
            / self
                .settings
                .curvature_gain
                .mul_add(previous_curvature.abs(), 1.0);

        let commanded = self.tracker.track(pose, path, speed_reference)?;
        let bias = self.avoidance.turn_rate_bias(pose)?;
        let requested = Command {
            speed: commanded.speed,
            turn_rate: commanded.turn_rate + bias,
        };

        let applied = self.conditioner.apply(requested, dt)?;
        self.vehicle.step(applied, dt)?;

        let errors = self.tracker.errors();
        let sample = TrackingSample {
            cross_track_error: errors.cross_track,
            heading_error: errors.heading,
            pose: self.vehicle.pose(),
            speed: self.vehicle.speed(),
            turn_rate: self.vehicle.turn_rate(),
            curvature: errors.curvature,
            avoidance_bias: bias,
            requested,
            applied,
        };
        self.record(sample);
        Ok(sample)
    }

    /// Runs `steps` tracking steps and returns the last sample.
    ///
    /// Returns `None` only when `steps` is zero, which is the one case
    /// where nothing happened rather than something failing.
    ///
    /// # Errors
    ///
    /// As [`TrackingLoop::step`], on the first step that fails.
    pub fn run(
        &mut self,
        path: &[(f64, f64)],
        steps: usize,
        dt: f64,
    ) -> Result<Option<TrackingSample>, Error> {
        let mut last = None;
        for _ in 0..steps {
            last = Some(self.step(path, dt)?);
        }
        Ok(last)
    }

    /// Stores a sample, dropping the oldest when the ring is full.
    fn record(&mut self, sample: TrackingSample) {
        match self.settings.history_capacity {
            Some(0) => {}
            Some(capacity) => {
                if self.history.len() >= capacity {
                    self.history.pop_front();
                }
                self.history.push_back(sample);
            }
            None => self.history.push_back(sample),
        }
    }
}
