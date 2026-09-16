//! What a command may be, and how fast it may change.
//!
//! Deviation A-09. Every command leaving this crate passes one saturation
//! function and one rate limiter, and both report separately, because
//! their failure modes are different. A magnitude clamp exhausts control
//! authority: the loop is open while it holds, since the actuator stays at
//! its limit whatever the plant does. A rate limit instead injects phase
//! lag, and NASA TN D-7900, analyzing pilot-induced oscillation on the
//! YF-12, found the rate limits more damaging than the position limits. A
//! caller told only that "the command was limited" cannot tell those
//! apart.
//!
//! This module reports and never decides. It counts saturated steps and
//! accumulates how much command was asked for and not delivered; what a
//! sustained saturation means belongs to the system doing the driving.

use arco_core::Error;
use arco_core::protocols::Command;

/// The elapsed interval a control step will accept, seconds.
///
/// `FR-INV-10`. A step reads no clock, so the interval arrives as an
/// argument and can be anything the caller computed, including a negative
/// one after a clock adjustment or a huge one after a stall. Both produce
/// a command that is arithmetically valid and physically wrong, so the
/// band is checked instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalBand {
    /// Shortest acceptable interval, seconds.
    pub minimum: f64,
    /// Longest acceptable interval, seconds.
    pub maximum: f64,
}

impl Default for IntervalBand {
    /// One microsecond to one second.
    ///
    /// The floor is where a rate limit stops being representable in double
    /// precision for the command magnitudes this library deals in. The
    /// ceiling is an order of magnitude above the 0.1 second step the
    /// tracking loop defaults to, which makes a missed deadline visible
    /// rather than silently integrated over.
    fn default() -> Self {
        Self {
            minimum: 1e-6,
            maximum: 1.0,
        }
    }
}

impl IntervalBand {
    /// Builds a band, rejecting one that is empty or not finite.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when either bound is not finite or
    /// not strictly positive, or when the minimum is above the maximum.
    pub fn new(minimum: f64, maximum: f64) -> Result<Self, Error> {
        for (quantity, value) in [("interval minimum", minimum), ("interval maximum", maximum)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "(0, inf)",
                });
            }
        }
        if minimum > maximum {
            return Err(Error::OutOfRange {
                quantity: "interval minimum",
                value: minimum,
                bound: "at or below the interval maximum",
            });
        }
        Ok(Self { minimum, maximum })
    }

    /// Rejects an interval a control step should not compute with.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when `dt` is NaN or infinite, and
    /// [`Error::OutOfRange`] when it falls outside the band.
    pub fn check(&self, dt: f64) -> Result<(), Error> {
        if !dt.is_finite() {
            return Err(Error::NotFinite {
                quantity: "elapsed interval",
                value: dt,
            });
        }
        if dt < self.minimum || dt > self.maximum {
            return Err(Error::OutOfRange {
                quantity: "elapsed interval",
                value: dt,
                bound: "the configured interval band",
            });
        }
        Ok(())
    }
}

/// The box a command has to lie in, and how fast it may move inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommandLimits {
    /// Highest commandable speed, meters per second.
    pub max_speed: f64,
    /// Lowest commandable speed, meters per second.
    ///
    /// Negative where the vehicle reverses, zero where it does not.
    pub min_speed: f64,
    /// Largest commandable turn rate either way, radians per second.
    pub max_turn_rate: f64,
    /// Largest change in commanded speed, meters per second squared.
    pub max_speed_rate: f64,
    /// Largest change in commanded turn rate, radians per second squared.
    pub max_turn_rate_change: f64,
    /// What interval a step will accept.
    pub interval: IntervalBand,
}

impl Default for CommandLimits {
    /// Wide enough not to bite, so that adding limits is a decision.
    ///
    /// A default that clamps would change the behavior of a caller that
    /// never asked for a limit, which is exactly the silent improvement
    /// deviation A-09 exists to avoid. A caller wanting limits states
    /// them.
    fn default() -> Self {
        Self {
            max_speed: f64::INFINITY,
            min_speed: f64::NEG_INFINITY,
            max_turn_rate: f64::INFINITY,
            max_speed_rate: f64::INFINITY,
            max_turn_rate_change: f64::INFINITY,
            interval: IntervalBand::default(),
        }
    }
}

impl CommandLimits {
    /// Rejects a limit set that no command could satisfy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when a bound is NaN, when the speed
    /// band is empty, or when a rate limit is negative.
    pub fn validate(&self) -> Result<(), Error> {
        for (quantity, value) in [
            ("maximum speed", self.max_speed),
            ("minimum speed", self.min_speed),
        ] {
            if value.is_nan() {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "a real number",
                });
            }
        }
        if self.min_speed > self.max_speed {
            return Err(Error::OutOfRange {
                quantity: "minimum speed",
                value: self.min_speed,
                bound: "at or below the maximum speed",
            });
        }
        for (quantity, value) in [
            ("maximum turn rate", self.max_turn_rate),
            ("maximum speed rate", self.max_speed_rate),
            ("maximum turn rate change", self.max_turn_rate_change),
        ] {
            if value.is_nan() || value < 0.0 {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "[0, inf]",
                });
            }
        }
        Ok(())
    }
}

/// How much command was asked for and not delivered.
///
/// Counts and integrals rather than a verdict: the two saturations are
/// reported apart because they fail differently, and no primary source
/// defines how long a saturation has to persist before it means something,
/// so this type does not guess.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SaturationReport {
    /// Steps where the magnitude clamp bit.
    pub magnitude_steps: usize,
    /// Steps where the rate limiter bit.
    pub rate_steps: usize,
    /// Accumulated speed asked for and not delivered, meters per second.
    pub speed_excess: f64,
    /// Accumulated turn rate asked for and not delivered, radians per second.
    pub turn_rate_excess: f64,
}

impl SaturationReport {
    /// Whether either limiter has bitten since the last reset.
    #[must_use]
    pub const fn saturated(&self) -> bool {
        self.magnitude_steps > 0 || self.rate_steps > 0
    }

    /// Clears the counters, which a caller does at the start of a run.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Applies the magnitude and rate limits to every command that passes.
///
/// Holds the previous applied command, because a rate limit is defined
/// against what was actually applied rather than against what was last
/// asked for. Limiting against the request instead lets a caller walk the
/// command past the limit one rejected step at a time.
#[derive(Debug, Clone)]
pub struct CommandConditioner {
    limits: CommandLimits,
    previous: Option<Command>,
    report: SaturationReport,
}

impl CommandConditioner {
    /// Builds a conditioner enforcing `limits`.
    ///
    /// # Errors
    ///
    /// As [`CommandLimits::validate`].
    pub fn new(limits: CommandLimits) -> Result<Self, Error> {
        limits.validate()?;
        Ok(Self {
            limits,
            previous: None,
            report: SaturationReport::default(),
        })
    }

    /// The limits in force.
    #[must_use]
    pub const fn limits(&self) -> CommandLimits {
        self.limits
    }

    /// What has been clipped so far.
    #[must_use]
    pub const fn report(&self) -> SaturationReport {
        self.report
    }

    /// The last command that was applied, if any.
    #[must_use]
    pub const fn previous(&self) -> Option<Command> {
        self.previous
    }

    /// Forgets the previous command and the counters.
    ///
    /// The next command is then limited in magnitude but not in rate,
    /// which is what starting a run means: there is nothing to have
    /// changed from.
    pub fn reset(&mut self) {
        self.previous = None;
        self.report.reset();
    }

    /// Clamps `requested` into the limits, given the elapsed interval.
    ///
    /// `FR-INV-09`: the returned command lies inside the limit box and is
    /// reachable from the previous one under the rate limit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when the command or the interval
    /// carries a NaN, or [`Error::OutOfRange`] when the interval falls
    /// outside the configured band, per `FR-INV-10`.
    pub fn apply(&mut self, requested: Command, dt: f64) -> Result<Command, Error> {
        self.limits.interval.check(dt)?;
        for (quantity, value) in [
            ("commanded speed", requested.speed),
            ("commanded turn rate", requested.turn_rate),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }

        // Magnitude first, then rate. The other order lets a request far
        // outside the box consume the whole rate allowance moving toward a
        // value that is then clamped away, so the applied command creeps
        // while the reported rate saturation says it did not.
        let clamped = Command {
            speed: requested
                .speed
                .clamp(self.limits.min_speed, self.limits.max_speed),
            turn_rate: requested
                .turn_rate
                .clamp(-self.limits.max_turn_rate, self.limits.max_turn_rate),
        };
        let magnitude_bit = differs(clamped.speed, requested.speed)
            || differs(clamped.turn_rate, requested.turn_rate);

        let applied = match self.previous {
            None => clamped,
            Some(previous) => Command {
                speed: step_toward(
                    previous.speed,
                    clamped.speed,
                    self.limits.max_speed_rate * dt,
                ),
                turn_rate: step_toward(
                    previous.turn_rate,
                    clamped.turn_rate,
                    self.limits.max_turn_rate_change * dt,
                ),
            },
        };
        let rate_bit =
            differs(applied.speed, clamped.speed) || differs(applied.turn_rate, clamped.turn_rate);

        if magnitude_bit {
            self.report.magnitude_steps = self.report.magnitude_steps.saturating_add(1);
        }
        if rate_bit {
            self.report.rate_steps = self.report.rate_steps.saturating_add(1);
        }
        self.report.speed_excess += (requested.speed - applied.speed).abs();
        self.report.turn_rate_excess += (requested.turn_rate - applied.turn_rate).abs();

        self.previous = Some(applied);
        Ok(applied)
    }
}

/// Moves `from` toward `to` by at most `allowance`.
///
/// An infinite allowance passes the target through unchanged rather than
/// producing a NaN, which is what `INFINITY * 0.0` would give if the
/// arithmetic were done the obvious way.
fn step_toward(from: f64, to: f64, allowance: f64) -> f64 {
    if !allowance.is_finite() {
        return to;
    }
    let change = to - from;
    if change.abs() <= allowance {
        to
    } else {
        from + allowance.copysign(change)
    }
}

/// Whether two commands differ by more than rounding.
///
/// A bare inequality reports saturation on the step where the request sits
/// exactly on the limit and the clamp returns a value one unit in the last
/// place away.
fn differs(left: f64, right: f64) -> bool {
    (left - right).abs() > f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}
