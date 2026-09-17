//! A unicycle that only drives forward and only turns so hard.

use arco_control::limits::{CommandLimits, IntervalBand};
use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::numeric::angle_difference;
use arco_core::protocols::{Command, VehicleModel};

use crate::state::require_state;

/// A Dubins-like unicycle: a pose, a speed, and a turn rate.
///
/// The vehicle is the plant rather than the controller, so the limits it
/// carries are the ones the machine has: a speed band that keeps it from
/// reversing, a turn-rate bound that gives it a minimum turning radius,
/// and a bound on how fast either may change. A command outside any of
/// them is clipped rather than refused, because an actuator asked for more
/// than it can give saturates at its limit.
///
/// The limits live in [`CommandLimits`] rather than in fields of their
/// own. That type carries exactly this vehicle's five bounds plus the
/// interval band of `FR-INV-10`, and the tracking loop already states its
/// limits that way, so a caller configuring a loop and a vehicle writes
/// one kind of thing twice rather than two kinds once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DubinsVehicle {
    pose: Pose,
    speed: f64,
    turn_rate: f64,
    limits: CommandLimits,
}

impl DubinsVehicle {
    /// The limits `arco.guidance.vehicle.DubinsVehicle` starts with.
    ///
    /// Five meters per second, no reversing, one radian per second of
    /// turn, and two per second of change in either, meaning two meters
    /// per second squared and two radians per second squared. They live
    /// here rather than in a constructor's defaults so that the binding
    /// layer and a Rust caller agree on what a default vehicle is.
    #[must_use]
    pub fn default_limits() -> CommandLimits {
        CommandLimits {
            max_speed: 5.0,
            min_speed: 0.0,
            max_turn_rate: 1.0,
            max_speed_rate: 2.0,
            max_turn_rate_change: 2.0,
            interval: IntervalBand::default(),
        }
    }

    /// Builds a vehicle at rest at `(x, y)` facing `heading`.
    ///
    /// # Arguments
    ///
    /// * `x` - Position along the first axis, meters.
    /// * `y` - Position along the second axis, meters.
    /// * `heading` - Orientation, radians, wrapped into `[-pi, pi)`.
    /// * `limits` - What the vehicle can do, per [`Self::default_limits`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a pose component is not a real
    /// number, and [`Error::OutOfRange`] when the limits describe a box no
    /// command could sit in.
    pub fn new(x: f64, y: f64, heading: f64, limits: CommandLimits) -> Result<Self, Error> {
        limits.validate()?;
        Ok(Self {
            pose: Pose::new(x, y, heading)?,
            speed: 0.0,
            turn_rate: 0.0,
            limits,
        })
    }

    /// The limits in force.
    #[must_use]
    pub const fn limits(&self) -> CommandLimits {
        self.limits
    }

    /// Replaces the limits, rejecting a set no command could satisfy.
    ///
    /// The speed and turn rate keep whatever values they had, so a vehicle
    /// moving faster than a newly imposed ceiling is brought inside it by
    /// the next step rather than teleported there.
    ///
    /// # Errors
    ///
    /// As [`CommandLimits::validate`].
    pub fn set_limits(&mut self, limits: CommandLimits) -> Result<(), Error> {
        limits.validate()?;
        self.limits = limits;
        Ok(())
    }

    /// Places the vehicle at a speed and turn rate directly.
    ///
    /// The kinematic state is normally an output: a command goes in and
    /// the state follows. Starting a run part way through a manoeuvre
    /// needs it as an input, and Python let a caller assign the private
    /// attributes to do that. This is the same thing with the limits
    /// checked, which assigning an attribute never was.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a value is not a real number, or
    /// [`Error::OutOfRange`] when it falls outside the configured limits.
    pub fn set_motion(&mut self, speed: f64, turn_rate: f64) -> Result<(), Error> {
        for (quantity, value) in [("speed", speed), ("turn rate", turn_rate)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        let limits = self.limits();
        if speed < limits.min_speed || speed > limits.max_speed {
            return Err(Error::OutOfRange {
                quantity: "speed",
                value: speed,
                bound: "the configured speed band",
            });
        }
        if turn_rate.abs() > limits.max_turn_rate {
            return Err(Error::OutOfRange {
                quantity: "turn rate",
                value: turn_rate,
                bound: "the configured turn-rate limit",
            });
        }
        self.speed = speed;
        self.turn_rate = turn_rate;
        Ok(())
    }

    /// Returns the vehicle to `(x, y, heading)` at rest.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a component is not a real number.
    pub fn reset(&mut self, x: f64, y: f64, heading: f64) -> Result<(), Error> {
        self.pose = Pose::new(x, y, heading)?;
        self.speed = 0.0;
        self.turn_rate = 0.0;
        Ok(())
    }

    /// The command that steers from `start` toward `goal` in `duration`.
    ///
    /// A naive inversion of the unicycle: hold the requested speed and
    /// turn at whatever rate closes the heading gap within the time
    /// allowed, saturated to what the vehicle can do. It is an admissible
    /// first guess for the trajectory optimizer rather than a steering
    /// law, since it points at the goal once and never looks again.
    ///
    /// The gap is a wrapped angular difference, per `FR-INV-15`: a vehicle
    /// facing just west of north and a goal just east of it are a fifth of
    /// a turn apart, and plain subtraction would send it the long way
    /// around.
    ///
    /// # Arguments
    ///
    /// * `start` - Start state, `(x, y)` or longer, meters and radians. A
    ///   state without a heading is taken to face along the first axis.
    /// * `goal` - Goal state, `(x, y)` or longer. Only the position is
    ///   read, since a goal heading is not reachable by a single arc.
    /// * `speed` - Speed to hold, meters per second.
    /// * `duration` - Time allowed for the segment, seconds.
    ///
    /// # Returns
    ///
    /// The saturated speed and turn rate, as the command the vehicle would
    /// be given.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when a state is not a planar position,
    /// [`Error::NotFinite`] when a state or the speed carries a value that
    /// is not real, and [`Error::OutOfRange`] when `duration` is not
    /// finite and strictly positive.
    pub fn inverse_kinematics(
        &self,
        start: &[f64],
        goal: &[f64],
        speed: f64,
        duration: f64,
    ) -> Result<Command, Error> {
        let (start_x, start_y) = require_state("start state", start)?;
        let (goal_x, goal_y) = require_state("goal state", goal)?;
        if !speed.is_finite() {
            return Err(Error::NotFinite {
                quantity: "speed",
                value: speed,
            });
        }
        if !(duration.is_finite() && duration > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "segment duration",
                value: duration,
                bound: "(0, inf)",
            });
        }

        let bearing = (goal_y - start_y).atan2(goal_x - start_x);
        // A two-element state is a bare position, and the Python it
        // replaces read a missing heading as zero rather than refusing the
        // query, which is what lets a planner hand its waypoints straight
        // to the optimizer.
        let heading = start.get(2).copied().unwrap_or(0.0);
        let gap = angle_difference(bearing, heading)?;

        Ok(Command {
            speed: speed.clamp(self.limits.min_speed, self.limits.max_speed),
            turn_rate: (gap / duration)
                .clamp(-self.limits.max_turn_rate, self.limits.max_turn_rate),
        })
    }

    /// Whether `state` is one this vehicle could be in.
    ///
    /// A state carries as much as it knows: a position and a heading say
    /// nothing about dynamics and are always acceptable, a fourth
    /// component is a speed and is checked against the speed band, and a
    /// fifth is a turn rate and is checked against the turn-rate bound.
    /// This is the predicate a trajectory optimizer wraps in its
    /// feasibility policy to reject a solution the vehicle cannot execute.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when the state is not a planar position,
    /// or [`Error::NotFinite`] when it carries a value that is not real.
    /// A NaN speed compares false against both bounds, so an unchecked
    /// state would be reported feasible rather than rejected.
    pub fn is_feasible(&self, state: &[f64]) -> Result<bool, Error> {
        require_state("state", state)?;
        let Some(&speed) = state.get(3) else {
            return Ok(true);
        };
        if speed < self.limits.min_speed || speed > self.limits.max_speed {
            return Ok(false);
        }
        let Some(&turn_rate) = state.get(4) else {
            return Ok(true);
        };
        Ok(turn_rate.abs() <= self.limits.max_turn_rate)
    }
}

impl VehicleModel for DubinsVehicle {
    fn pose(&self) -> Pose {
        self.pose
    }

    fn speed(&self) -> f64 {
        self.speed
    }

    fn turn_rate(&self) -> f64 {
        self.turn_rate
    }

    /// Advances the pose by `dt` seconds under `command`.
    ///
    /// Explicit Euler about the heading held at the start of the step,
    /// which is what the Python did. The heading itself integrates
    /// exactly, since the turn rate is constant across the step, and it
    /// comes back wrapped because [`Pose`] wraps it, per `FR-INV-15`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a command component is not a real
    /// number, [`Error::OutOfRange`] when `dt` falls outside the configured
    /// interval band, per `FR-INV-10` and deviation A-17, and
    /// [`Error::NotFinite`] again when the integrated position overflows.
    fn step(&mut self, command: Command, dt: f64) -> Result<(), Error> {
        self.limits.interval.check(dt)?;
        for (quantity, value) in [
            ("commanded speed", command.speed),
            ("commanded turn rate", command.turn_rate),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }

        self.speed = advance(
            self.speed,
            command.speed,
            self.limits.max_speed_rate * dt,
            self.limits.min_speed,
            self.limits.max_speed,
        );
        self.turn_rate = advance(
            self.turn_rate,
            command.turn_rate,
            self.limits.max_turn_rate_change * dt,
            -self.limits.max_turn_rate,
            self.limits.max_turn_rate,
        );

        let (sine, cosine) = self.pose.heading().sin_cos();
        self.pose = Pose::new(
            (self.speed * cosine).mul_add(dt, self.pose.x()),
            (self.speed * sine).mul_add(dt, self.pose.y()),
            self.turn_rate.mul_add(dt, self.pose.heading()),
        )?;
        Ok(())
    }
}

/// Moves `current` toward `target` by at most `allowance`, then saturates.
///
/// Rate first and magnitude second, which is the order the Python used. It
/// reads as the wrong way round next to
/// [`arco_control::limits::CommandConditioner`], which saturates first so
/// that a request far outside the box cannot spend the rate allowance
/// moving toward a value that is then clamped away. The two orders agree
/// exactly whenever `current` already lies between `lowest` and `highest`,
/// which a vehicle does from its first step onward, so the order is a
/// statement about where the state starts rather than about the limiter.
///
/// An infinite allowance leaves the target untouched rather than producing
/// a NaN, because the difference it clamps is finite.
fn advance(current: f64, target: f64, allowance: f64, lowest: f64, highest: f64) -> f64 {
    // Both clamps are well defined because `CommandLimits::validate` has
    // already rejected a NaN bound, an empty speed band, and a negative
    // rate, which are the three ways `f64::clamp` panics.
    let change = (target - current).clamp(-allowance, allowance);
    (current + change).clamp(lowest, highest)
}
