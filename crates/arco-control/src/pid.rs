//! Proportional, integral and derivative feedback, with a way out of windup.

use arco_core::Error;

use crate::limits::IntervalBand;

/// The three gains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidGains {
    /// Proportional gain.
    pub proportional: f64,
    /// Integral gain, per second.
    pub integral: f64,
    /// Derivative gain, seconds.
    pub derivative: f64,
}

impl Default for PidGains {
    /// The gains `arco.control.pid.PIDController` defaults to.
    fn default() -> Self {
        Self {
            proportional: 1.0,
            integral: 0.0,
            derivative: 0.1,
        }
    }
}

/// What to do with the integrator while the output is saturated.
///
/// A saturated loop is not a tuning annoyance. Astrom puts it plainly:
/// when the output saturates the feedback loop is broken and the system
/// runs open loop, because the actuator stays at its limit whatever the
/// process does. The integrator meanwhile keeps accumulating error it
/// cannot act on, and the command stays pinned long after the error
/// changed sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AntiWindup {
    /// Feed the clipped amount back into the integrator through a gain.
    ///
    /// The default remedy. The correction is exactly zero while the
    /// output is unsaturated, so it costs nothing in normal operation.
    /// Astrom brackets the tracking time constant between the derivative
    /// and integral time constants; the gain here is its reciprocal.
    BackCalculation {
        /// Reciprocal of the tracking time constant, per second.
        tracking_gain: f64,
    },
    /// Stop integrating while the output is saturated.
    ///
    /// Offered because it is asked for, not because it is good. Switching
    /// the integrator off is an unanalyzed nonlinearity, and a loop
    /// carrying one does not have the stability margins its linear design
    /// says it has.
    ConditionalIntegration,
    /// Let the integrator wind.
    ///
    /// Correct only where the output cannot saturate, which for the
    /// default limits is the case.
    None,
}

impl Default for AntiWindup {
    fn default() -> Self {
        Self::BackCalculation { tracking_gain: 1.0 }
    }
}

/// How a PID controller should behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidSettings {
    /// The gains.
    pub gains: PidGains,
    /// Lower and upper bound on the output.
    ///
    /// Infinite by default, so the controller reproduces the Python one
    /// exactly and a caller wanting a limit asks for it. Deviation A-09
    /// adds the mechanism, not a limit nobody chose.
    pub output_limits: (f64, f64),
    /// What to do with the integrator while saturated.
    pub anti_windup: AntiWindup,
    /// What elapsed interval a step will accept.
    pub interval: IntervalBand,
}

impl Default for PidSettings {
    fn default() -> Self {
        Self {
            gains: PidGains::default(),
            output_limits: (f64::NEG_INFINITY, f64::INFINITY),
            anti_windup: AntiWindup::default(),
            interval: IntervalBand::default(),
        }
    }
}

/// A PID controller carrying its own integrator and previous error.
///
/// The step takes the elapsed interval and reads no clock, which is what
/// makes a run reproducible and a test able to state the sample rate.
/// `arco.control.pid.PIDController` had no interval at all: it summed raw
/// errors and differenced raw errors, which is this controller at an
/// interval of exactly one second. The binding passes that, so nothing
/// visible from Python changes.
#[derive(Debug, Clone)]
pub struct PidController {
    settings: PidSettings,
    integral: f64,
    previous_error: f64,
    started: bool,
    saturated_steps: usize,
}

impl PidController {
    /// Builds a controller from its settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a gain is not a real number, and
    /// [`Error::OutOfRange`] when the output limits are inverted or the
    /// tracking gain is negative.
    pub fn new(settings: PidSettings) -> Result<Self, Error> {
        for (quantity, value) in [
            ("proportional gain", settings.gains.proportional),
            ("integral gain", settings.gains.integral),
            ("derivative gain", settings.gains.derivative),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        let (lower, upper) = settings.output_limits;
        if lower.is_nan() || upper.is_nan() || lower > upper {
            return Err(Error::OutOfRange {
                quantity: "output lower limit",
                value: lower,
                bound: "at or below the upper limit",
            });
        }
        if let AntiWindup::BackCalculation { tracking_gain } = settings.anti_windup
            && !(tracking_gain.is_finite() && tracking_gain >= 0.0)
        {
            return Err(Error::OutOfRange {
                quantity: "tracking gain",
                value: tracking_gain,
                bound: "[0, inf)",
            });
        }
        Ok(Self {
            settings,
            integral: 0.0,
            previous_error: 0.0,
            started: false,
            saturated_steps: 0,
        })
    }

    /// The settings in force.
    #[must_use]
    pub const fn settings(&self) -> PidSettings {
        self.settings
    }

    /// The accumulated integral term.
    #[must_use]
    pub const fn integral(&self) -> f64 {
        self.integral
    }

    /// How many steps returned a saturated output.
    #[must_use]
    pub const fn saturated_steps(&self) -> usize {
        self.saturated_steps
    }

    /// Clears the integrator, the previous error and the counter.
    pub const fn reset(&mut self) {
        self.integral = 0.0;
        self.previous_error = 0.0;
        self.started = false;
        self.saturated_steps = 0;
    }

    /// Computes the command driving `state` toward `reference`.
    ///
    /// # Arguments
    ///
    /// * `state` - The measured value.
    /// * `reference` - The value being tracked.
    /// * `dt` - Elapsed interval since the previous step, seconds.
    ///
    /// # Returns
    ///
    /// The output after the limits, which is what was applied.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when an input is not a real number,
    /// and [`Error::OutOfRange`] when `dt` leaves the configured band, per
    /// `FR-INV-10`.
    pub fn step(&mut self, state: f64, reference: f64, dt: f64) -> Result<f64, Error> {
        self.settings.interval.check(dt)?;
        for (quantity, value) in [("state", state), ("reference", reference)] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }

        let error = reference - state;
        // The first step has no previous error, and pretending it was zero
        // makes the derivative term a spike proportional to the initial
        // error, which is the derivative kick every textbook warns about.
        let derivative = if self.started {
            (error - self.previous_error) / dt
        } else {
            0.0
        };

        let integrating = !matches!(
            self.settings.anti_windup,
            AntiWindup::ConditionalIntegration
        ) || !self.was_saturating(error);
        if integrating {
            self.integral += error * dt;
        }

        let gains = self.settings.gains;
        let unsaturated = gains.proportional.mul_add(
            error,
            gains
                .integral
                .mul_add(self.integral, gains.derivative * derivative),
        );
        let (lower, upper) = self.settings.output_limits;
        let applied = unsaturated.clamp(lower, upper);

        if let AntiWindup::BackCalculation { tracking_gain } = self.settings.anti_windup
            && tracking_gain > 0.0
            && gains.integral.abs() > 0.0
        {
            // Exactly zero while unsaturated, so an unlimited controller
            // behaves as though this branch were not here.
            self.integral += tracking_gain * (applied - unsaturated) * dt / gains.integral;
        }

        // Compared against rounding rather than exactly: a request landing
        // on the limit clamps to a value one unit in the last place away,
        // and counting that as saturation makes the report noise.
        if (applied - unsaturated).abs()
            > f64::EPSILON * applied.abs().max(unsaturated.abs()).max(1.0)
        {
            self.saturated_steps = self.saturated_steps.saturating_add(1);
        }
        self.previous_error = error;
        self.started = true;
        Ok(applied)
    }

    /// Whether integrating further would push deeper into a limit.
    ///
    /// Only consulted by [`AntiWindup::ConditionalIntegration`]. The test
    /// is on the sign of the error against the limit currently held, since
    /// an error driving the output back inside the box has to be
    /// integrated or the controller never leaves the limit.
    fn was_saturating(&self, error: f64) -> bool {
        let gains = self.settings.gains;
        let (lower, upper) = self.settings.output_limits;
        let output = gains
            .proportional
            .mul_add(self.previous_error, gains.integral * self.integral);
        (output >= upper && error * gains.integral > 0.0)
            || (output <= lower && error * gains.integral < 0.0)
    }
}
