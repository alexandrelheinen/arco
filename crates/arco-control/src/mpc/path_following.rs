//! Following a path by contouring, as a short sequence of convex programs.
//!
//! The SE(2) half of ADR-002. [`PathFollowingMpc`] replaces the `CasADi`
//! program of `arco.control.mpc.path_following` with the sequential
//! quadratic loop `FR-MPC-01` asks for: linearize the unicycle model and
//! the three contouring errors about a nominal trajectory, solve the
//! convex program that falls out, take the answer as the next nominal,
//! and stop when the two agree. The joint-space sibling needs no such
//! loop because its model is already linear; this one does, and almost
//! everything below follows from that difference.
//!
//! # What the convex form keeps, and what it cannot
//!
//! Every constraint carries over: the initial pin, the five-row
//! recurrence, the virtual progress law, both speed bounds, the turn
//! rate, the acceleration, the turn-rate derivative, the progress speed
//! against the cruise cap and against the curve limit, and the path
//! parameter inside the path. `FR-MPC-03` is read back off the returned
//! solution in [`HorizonPlan`] rather than assumed of the solver, and a
//! solution breaking a bound is refused instead of commanded.
//!
//! Four cost terms change shape:
//!
//! - The contouring, lag and heading costs become Gauss-Newton
//!   surrogates, meaning the square of the affine expansion of each
//!   error. Dropping the second-order term is what makes each block
//!   positive semidefinite by construction.
//! - The heading term swaps `2 (1 - cos e)` for the square of the
//!   wrapped error at unchanged weight, which agrees to second order and
//!   parts company past a few tenths of a radian. See
//!   [`quadratic_heading_cost`](crate::mpc::model::quadratic_heading_cost).
//! - The contour deadzone becomes a slack variable and two rows. That
//!   one is exact rather than approximate: the epigraph of
//!   `max(|e| - d, 0)^2` is a convex set and the rows describe it.
//! - The obstacle barrier becomes a half-space through the nominal
//!   position, penalized through a slack. The keep-out set is the
//!   complement of a disc and no convex program describes it, so the
//!   half-space is an inner approximation tangent at the nominal, which
//!   is conservative. Its quartic becomes a square, and the forward cone
//!   factor freezes at the nominal heading and multiplies the slack's
//!   weight rather than entering the row.
//!
//! # Two curvatures, for two different jobs
//!
//! [`ReferencePath`] reports a curvature that has been spread over a
//! minimum arc, previewed backward from the sharpest corner ahead, and
//! capped. That is the right number for a speed limit, since braking has
//! to start before the corner, and the wrong number for a Jacobian: on
//! the straight approaching a corner the reference does not actually
//! turn, so the arc-length cross terms of
//! [`linearize_path_errors`] would be full size where the truth is zero.
//!
//! The curve-limited progress cap therefore reads the previewed
//! curvature, and the error expansions read a curvature measured from
//! the reference heading itself, over a span matching the one the path
//! spreads a turn across. The two agree at a corner and differ on the
//! approach to one, which is the whole point.
//!
//! # Why a trust region exists here and not in the joint-space program
//!
//! The affine model is written in absolute variables, which is what
//! keeps a bound on the speed a bound on a variable rather than on an
//! offset. The price is that nothing stops the solver placing the answer
//! where the tangent plane no longer describes the model, and the error
//! of that plane grows with the square of the distance. The heading and
//! the arc length carry a radius for that reason: the heading because
//! every nonlinearity in the position rows is a product with it, and
//! because the quadratic heading surrogate loses its periodicity once it
//! is written on an unwrapped variable; the arc length because it is
//! what moves the frame the errors are measured in.

use arco_core::Error;
use arco_core::numeric::{
    RELATIVE_TOLERANCE, TIME_TOLERANCE, angle_difference, is_close, wrap_angle,
};
use arco_core::protocols::{Command, Occupancy};

use crate::limits::{CommandConditioner, CommandLimits, SaturationReport};
use crate::mpc::joint_space::CONSTRAINT_TOLERANCE;
use crate::mpc::model::{
    ControlInput, ErrorExpansion, INPUT_DIMENSION, PathErrorExpansion, STATE_DIMENSION,
    StageDynamics, VehicleState, linearize_path_errors, linearize_unicycle, path_errors,
    unicycle_step,
};
use crate::mpc::qp::{QpProblem, SolveFailure, Triplets};
use crate::mpc::reference::{PathSample, ReferencePath};

/// The longest horizon a controller will assemble, steps.
const MAX_HORIZON_STEPS: usize = 512;

/// The most sequential iterations a control step may take.
///
/// `FR-SAFE-02` in its outer form. The inner budget bounds one solve and
/// says nothing about how many solves a step runs, so the product is
/// what a real-time caller actually pays.
const MAX_SQP_ITERATIONS: usize = 16;

/// How many points along the reference the obstacle probe samples.
const OBSTACLE_SAMPLE_COUNT: usize = 5;

/// How many points the cruise preview samples along the reference.
const CRUISE_PREVIEW_COUNT: usize = 12;

/// What the curve-limited progress cap adds under its square root.
///
/// The cap is `v_s sqrt(K^2 + eps) <= max_turn_rate`, which stands in for
/// `v_s <= max_turn_rate / |K|` without the pole at a straight. Keeping
/// the smoothing inside the root rather than outside it matters twice:
/// the constraint stays differentiable, and the coefficient the row
/// carries is never exactly zero. A row whose only coefficient evaluates
/// to zero has no entries at all once the sparse builder drops it, and an
/// empty row bounds nothing while looking exactly like a row that does.
const CURVE_CAP_SMOOTHING: f64 = 1e-6;

/// The arc the linearization measures the reference heading turn across.
///
/// Matched to the shortest arc [`ReferencePath`] spreads a polyline turn
/// over, so that a central difference of the heading across this span
/// reproduces the same curvature the path's own profile reports at a
/// vertex, while reporting zero on a straight where the previewed
/// profile does not.
const GRADIENT_CURVATURE_SPAN: f64 = 8.0;

/// The largest curvature the linearization will accept, per meter.
const GRADIENT_CURVATURE_CEILING: f64 = 0.35;

/// How far the arc-length frame may be pushed toward its own singularity.
///
/// The lag error falls at `-1 + K e_contour` per meter of progress, so at
/// a contouring error of one radius of curvature that derivative reaches
/// zero and past it the model claims advancing the path parameter makes
/// the lag error worse. That is a correct linearization of a
/// parameterization which is genuinely invalid at the center of
/// curvature, and a controller three meters wide of a tight corner is an
/// ordinary situation rather than an exotic one. Bounding the product
/// keeps the derivative inside `[-1 - this, -1 + this]`, so progress
/// always pays.
const FRAME_SINGULARITY_MARGIN: f64 = 0.5;

/// The smallest clearance a barrier will normalize a penetration by, meters.
const CLEARANCE_FLOOR: f64 = 1e-3;

/// How near an obstacle a linearization point may sit, meters.
const SEPARATION_FLOOR: f64 = 1e-9;

/// What the forward cone factor is worth when it points backward.
const CONE_FLOOR: f64 = 0.2;

/// How much of the barrier weight the forward cone factor governs.
const CONE_SPAN: f64 = 0.8;

/// The shortest projection window a control step will search, meters.
const PROJECTION_WINDOW_FLOOR: f64 = 30.0;

/// The shortest runway appended past the final waypoint, meters.
const RUNWAY_FLOOR: f64 = 2.0;

/// How far past one horizon the obstacle and cruise probes look.
const LOOK_AHEAD_SCALE: f64 = 1.5;

/// The shortest look-ahead those probes will use, meters.
const LOOK_AHEAD_FLOOR: f64 = 2.0;

/// The most the cruise preview will taper the cruise speed by.
const CRUISE_TAPER_FLOOR: f64 = 0.15;

/// The shortest horizon advance a warm start must show to be reused, meters.
const STALL_ADVANCE_FLOOR: f64 = 1.0;

/// What fraction of an unobstructed horizon counts as still moving.
const STALL_ADVANCE_FRACTION: f64 = 0.1;

/// The smallest curvature the rollout will divide a turn rate by.
const ROLLOUT_CURVATURE_FLOOR: f64 = 1e-3;

/// How far the reported cost may sit from the solver's own value.
///
/// Both numbers describe the same quadratic at the same point, so they
/// agree to rounding or this module handed the solver a program it did
/// not mean to write. A halved cross term or a dropped constant changes
/// the optimum while every constraint still holds exactly, which is the
/// one class of assembly error a feasibility check cannot see.
const COST_AGREEMENT_TOLERANCE: f64 = 1e-6;

/// How a path-following controller should behave.
///
/// The weights and the horizon are the Python dataclass defaults, so a
/// controller built from [`PathFollowingSettings::new`] poses the same
/// problem the `CasADi` version posed, up to the reformulation the module
/// documentation describes. The three fields with no Python counterpart
/// are the two trust radii and the sequential iteration budget, which the
/// nonlinear solver had no need of.
#[derive(Debug, Clone, PartialEq)]
pub struct PathFollowingSettings {
    /// What the vehicle may be commanded, and how fast that may change.
    ///
    /// Deviation A-20 in its `CasADi` form: the five loose attributes of
    /// `DubinsVehicleLimits` become one limit set, with `max_acceleration`
    /// spelled [`CommandLimits::max_speed_rate`] and `max_turn_rate_dot`
    /// spelled [`CommandLimits::max_turn_rate_change`]. The same set
    /// bounds the program and conditions the command that leaves it, so
    /// the two cannot disagree.
    pub limits: CommandLimits,
    /// How many steps the program looks ahead.
    pub horizon_step_count: usize,
    /// The interval one horizon step covers, seconds.
    pub step_interval: f64,
    /// Nominal progress speed on a straight, meters per second.
    pub cruise_speed: f64,
    /// Weight on the lateral error, outside the deadzone.
    pub weight_contour: f64,
    /// Weight on the heading error.
    pub weight_heading: f64,
    /// Reward per meter of arc length advanced.
    ///
    /// Linear rather than a quadratic match against a reference speed,
    /// and the Python comment explaining why is worth repeating: at a
    /// sharp corner the curve-limited speed is small, so a quadratic
    /// makes parking at the corner nearly free while accelerating away
    /// reads as expensive across the whole horizon, and the solver settles
    /// into a permanent stop. A linear reward pays for advancement
    /// everywhere and cannot manufacture a stationary point of its own.
    pub weight_progress: f64,
    /// Weight on the longitudinal error.
    ///
    /// Structural, and refused at zero. The lag error is the only term
    /// coupling the path parameter to the vehicle position, so a zero
    /// weight decouples them and leaves the contouring errors measured
    /// against a point that is free to sit anywhere on the path.
    pub weight_lag: f64,
    /// Weight on the commanded acceleration and turn-rate change.
    pub weight_control: f64,
    /// Weight on penetrating an obstacle's clearance.
    pub weight_obstacle: f64,
    /// Weight on the terminal contouring and heading errors.
    pub weight_terminal: f64,
    /// Lateral band the contouring cost ignores, meters.
    pub contour_deadzone: f64,
    /// How far a predicted heading may sit from the nominal, radians.
    pub trust_heading: f64,
    /// How far a predicted arc length may sit from the nominal, meters.
    pub trust_arc_length: f64,
    /// The most sequential solves one control step may run.
    pub max_sqp_iterations: usize,
    /// How close two successive iterates count as converged.
    ///
    /// Compared against the largest disagreement between them in any of
    /// position, heading or arc length, so it carries no single unit. A
    /// mixed norm is defensible here because the test is whether the
    /// linearization point stopped moving, not how far it moved.
    pub sqp_tolerance: f64,
    /// The interior-point budget for one solve, per `FR-SAFE-02`.
    pub max_solver_iterations: u32,
}

impl PathFollowingSettings {
    /// Builds settings around `limits`, carrying the Python defaults.
    #[must_use]
    pub const fn new(limits: CommandLimits) -> Self {
        Self {
            limits,
            horizon_step_count: 20,
            step_interval: 0.05,
            cruise_speed: 0.36,
            weight_contour: 10.0,
            weight_heading: 2.0,
            weight_progress: 1.0,
            weight_lag: 4.0,
            weight_control: 0.1,
            weight_obstacle: 50.0,
            weight_terminal: 20.0,
            contour_deadzone: 0.0,
            trust_heading: 0.5,
            trust_arc_length: 5.0,
            max_sqp_iterations: 3,
            sqp_tolerance: 1e-3,
            max_solver_iterations: 80,
        }
    }

    /// Rejects settings no usable program could be built from.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when a budget is empty, and
    /// [`Error::OutOfRange`] when the horizon or a budget is past its
    /// ceiling, when a weight is negative or not a real number, when the
    /// lag weight is zero, when a limit is not finite, or when the model
    /// step falls outside the interval band the same limits declare.
    ///
    /// Every one of those was accepted by the Python constructor, which
    /// validated only the lag weight. A program built from a limit set
    /// carrying an infinity has a row whose bound is an infinity, which
    /// the solver wrapper refuses one layer further down and with less to
    /// say about why.
    pub fn validate(&self) -> Result<(), Error> {
        self.limits.validate()?;
        self.validate_budgets()?;
        self.validate_limits()?;
        self.validate_weights()?;

        for (quantity, value) in [
            ("cruise speed", self.cruise_speed),
            ("heading trust radius", self.trust_heading),
            ("arc length trust radius", self.trust_arc_length),
            ("sequential tolerance", self.sqp_tolerance),
            ("model step interval", self.step_interval),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "(0, inf)",
                });
            }
        }
        if self.step_interval < self.limits.interval.minimum
            || self.step_interval > self.limits.interval.maximum
        {
            return Err(Error::OutOfRange {
                quantity: "model step interval",
                value: self.step_interval,
                bound: "the configured interval band",
            });
        }
        Ok(())
    }

    /// Rejects an empty or unbounded budget.
    fn validate_budgets(&self) -> Result<(), Error> {
        for (quantity, count) in [
            ("horizon steps", self.horizon_step_count),
            ("sequential iterations", self.max_sqp_iterations),
        ] {
            if count == 0 {
                return Err(Error::TooFew {
                    quantity,
                    minimum: 1,
                    actual: 0,
                });
            }
        }
        if self.max_solver_iterations == 0 {
            return Err(Error::TooFew {
                quantity: "solver iterations",
                minimum: 1,
                actual: 0,
            });
        }
        for (quantity, count, ceiling) in [
            ("horizon steps", self.horizon_step_count, MAX_HORIZON_STEPS),
            (
                "sequential iterations",
                self.max_sqp_iterations,
                MAX_SQP_ITERATIONS,
            ),
        ] {
            if count > ceiling {
                return Err(Error::OutOfRange {
                    quantity,
                    value: count_as_value(count),
                    bound: "at or below the budget ceiling",
                });
            }
        }
        Ok(())
    }

    /// Rejects a limit the program cannot write a finite row against.
    fn validate_limits(&self) -> Result<(), Error> {
        for (quantity, value) in [
            ("maximum speed", self.limits.max_speed),
            ("minimum speed", self.limits.min_speed),
            ("maximum turn rate", self.limits.max_turn_rate),
            ("maximum speed rate", self.limits.max_speed_rate),
            ("maximum turn rate change", self.limits.max_turn_rate_change),
        ] {
            if !value.is_finite() {
                return Err(Error::NotFinite { quantity, value });
            }
        }
        Ok(())
    }

    /// Rejects a weight that would make the objective concave or empty.
    fn validate_weights(&self) -> Result<(), Error> {
        for (quantity, value) in [
            ("contour weight", self.weight_contour),
            ("heading weight", self.weight_heading),
            ("progress weight", self.weight_progress),
            ("control weight", self.weight_control),
            ("obstacle weight", self.weight_obstacle),
            ("terminal weight", self.weight_terminal),
            ("contour deadzone", self.contour_deadzone),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "[0, inf)",
                });
            }
        }
        if !(self.weight_lag.is_finite() && self.weight_lag > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "lag weight",
                value: self.weight_lag,
                bound: "(0, inf)",
            });
        }
        Ok(())
    }

    /// How far one horizon reaches at the cruise speed, meters.
    fn horizon_distance(&self) -> f64 {
        self.cruise_speed * self.step_interval * count_as_value(self.horizon_step_count)
    }

    /// How far ahead the obstacle and cruise probes look, meters.
    fn look_ahead(&self) -> f64 {
        (self.horizon_distance() * LOOK_AHEAD_SCALE).max(LOOK_AHEAD_FLOOR)
    }
}

/// The trajectory a solved step predicts.
///
/// Also the shape a linearization point takes, so that the answer to one
/// sequential iteration is directly the nominal of the next one and
/// nothing has to be reshaped between them. The state series carry one
/// more entry than the input series, since the horizon ends in a state
/// nothing is commanded from.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonPlan {
    states: Vec<VehicleState>,
    inputs: Vec<ControlInput>,
    arc_lengths: Vec<f64>,
    progress_speeds: Vec<f64>,
}

impl HorizonPlan {
    /// How many steps the plan spans, meaning how many inputs it holds.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.inputs.len()
    }

    /// The predicted state at `step`, up to and including the horizon.
    #[must_use]
    pub fn state(&self, step: usize) -> Option<VehicleState> {
        self.states.get(step).copied()
    }

    /// The input leaving `step`, absent at the final state.
    #[must_use]
    pub fn input(&self, step: usize) -> Option<ControlInput> {
        self.inputs.get(step).copied()
    }

    /// The path parameter at `step`, meters of arc length.
    #[must_use]
    pub fn arc_length(&self, step: usize) -> Option<f64> {
        self.arc_lengths.get(step).copied()
    }

    /// The virtual progress speed leaving `step`, meters per second.
    #[must_use]
    pub fn progress_speed(&self, step: usize) -> Option<f64> {
        self.progress_speeds.get(step).copied()
    }

    /// The predicted positions, which is what a caller draws.
    pub fn positions(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.states.iter().map(|state| (state.x, state.y))
    }

    /// Whether every number in the plan is a real one.
    fn finite(&self) -> bool {
        self.states.iter().all(|state| state.check().is_ok())
            && self.inputs.iter().all(|input| input.check().is_ok())
            && self
                .arc_lengths
                .iter()
                .chain(&self.progress_speeds)
                .all(|value| value.is_finite())
    }

    /// The largest disagreement with another plan, in any coordinate.
    ///
    /// Mixed units on purpose: see [`PathFollowingSettings::sqp_tolerance`].
    fn deviation(&self, other: &Self) -> f64 {
        let mut largest = 0.0_f64;
        for (here, there) in self.states.iter().zip(&other.states) {
            let heading = angle_difference(here.heading, there.heading).unwrap_or_default();
            largest = largest
                .max((here.x - there.x).hypot(here.y - there.y))
                .max(heading.abs());
        }
        for (here, there) in self.arc_lengths.iter().zip(&other.arc_lengths) {
            largest = largest.max((here - there).abs());
        }
        largest
    }

    /// Reads a solution vector back into a plan.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the solver returned
    /// fewer variables than the layout declared, which would mean this
    /// module and the solver disagree about the problem rather than that
    /// the problem was hard.
    fn from_solution(variables: &[f64], layout: &Layout) -> Result<Self, Error> {
        if variables.len() < layout.variable_count {
            return Err(Error::DimensionMismatch {
                quantity: "solution variables",
                expected: layout.variable_count,
                actual: variables.len(),
            });
        }
        let read = |index: usize| variables.get(index).copied().unwrap_or_default();
        let horizon = layout.horizon;

        let mut states = Vec::with_capacity(horizon.saturating_add(1));
        let mut arc_lengths = Vec::with_capacity(horizon.saturating_add(1));
        for step in 0..=horizon {
            let mut values = [0.0_f64; STATE_DIMENSION];
            for (component, slot) in values.iter_mut().enumerate() {
                *slot = read(layout.state(step, component));
            }
            states.push(VehicleState::from_array(values));
            arc_lengths.push(read(layout.arc(step)));
        }

        let mut inputs = Vec::with_capacity(horizon);
        let mut progress_speeds = Vec::with_capacity(horizon);
        for step in 0..horizon {
            let mut values = [0.0_f64; INPUT_DIMENSION];
            for (component, slot) in values.iter_mut().enumerate() {
                *slot = read(layout.input(step, component));
            }
            inputs.push(ControlInput::from_array(values));
            progress_speeds.push(read(layout.progress_speed(step)));
        }

        Ok(Self {
            states,
            inputs,
            arc_lengths,
            progress_speeds,
        })
    }
}

/// Why a control step produced a braking command rather than a plan.
///
/// `FR-MPC-04`. The Python returned one of three status strings and a
/// false success flag; this keeps the same distinction and adds the
/// solver's own taxonomy behind it, because a caller that cannot tell an
/// empty feasible set from an exhausted budget cannot tell whether
/// retrying is worth anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepFailure {
    /// The measured state carried a value that is not a real number.
    ///
    /// The Python `invalid_state`. A sensor dropout is a runtime
    /// condition the controller has to survive with a safe command, not a
    /// programming error, which is why it comes back as an outcome rather
    /// than as an [`Error`].
    InvalidState,
    /// No usable answer came back from a solve.
    Solve(SolveFailure),
}

impl core::fmt::Display for StepFailure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::InvalidState => formatter.write_str("the measured state is not finite"),
            Self::Solve(failure) => failure.fmt(formatter),
        }
    }
}

/// Where a step's command came from.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum StepOutcome {
    /// The program was solved and the command is its first predicted state.
    Solved {
        /// The objective value at the solution.
        ///
        /// The program as this module wrote it, evaluated at the answer,
        /// including the constant the solver cannot carry and the stage
        /// the initial pin makes constant. It is the surrogate objective
        /// rather than the nonlinear one the Python reported, so the two
        /// agree only where the expansions do.
        cost: f64,
        /// How many sequential iterations the step took.
        sqp_iterations: usize,
        /// How many interior-point iterations the last solve took.
        solver_iterations: u32,
        /// Whether the last solve reached the tolerance it was asked for.
        exact: bool,
        /// How far the last iterate moved from the one before it.
        ///
        /// At or below [`PathFollowingSettings::sqp_tolerance`] means the
        /// loop converged; above it means the iteration budget ran out
        /// first, which is a usable command and a less trustworthy one.
        convergence: f64,
    },
    /// No plan was solved, and the command brakes instead.
    SafeStop(StepFailure),
}

impl StepOutcome {
    /// Whether the command came from a solved program.
    #[must_use]
    pub const fn solved(&self) -> bool {
        matches!(*self, Self::Solved { .. })
    }

    /// Why no program was solved, if none was.
    #[must_use]
    pub const fn failure(&self) -> Option<StepFailure> {
        match *self {
            Self::Solved { .. } => None,
            Self::SafeStop(failure) => Some(failure),
        }
    }
}

/// What one control step did.
#[derive(Debug, Clone, PartialEq)]
pub struct PathFollowingStep {
    /// The command after conditioning, which is what the vehicle gets.
    pub command: Command,
    /// The command before conditioning, which is what the program asked for.
    ///
    /// Deviation A-09 wants the two reported apart. On a solved step they
    /// agree, because the program already bounds the speed and the turn
    /// rate of the state it commands; a difference means either the brake
    /// path ran or the conditioner is enforcing something the program was
    /// not told about.
    pub requested: Command,
    /// Signed lateral error against the reference, meters, left positive.
    pub contour_error: f64,
    /// Heading minus the reference heading there, radians, wrapped.
    pub heading_error: f64,
    /// Arc length reached along the reference, meters.
    pub progress: f64,
    /// The smallest predicted distance to an obstacle, meters.
    ///
    /// Infinite when the controller carries no map, and measured from the
    /// obstacle surface per deviation A-16.
    pub predicted_clearance: f64,
    /// What the two limiters have clipped since the last reset.
    pub saturation: SaturationReport,
    /// Whether this came from the program or from the brake.
    pub outcome: StepOutcome,
    /// The predicted trajectory, absent when nothing was solved.
    pub plan: Option<HorizonPlan>,
    /// The nominal the last solve was linearized about.
    ///
    /// Reported because half of what this controller does is choose that
    /// point, and because a trust region is only checkable against it.
    pub linearization: Option<HorizonPlan>,
}

/// A receding-horizon contouring controller over `SE(2)`.
///
/// Holds three things between control steps: the reference, how far along
/// it the vehicle has come, and the previous plan. The first two are the
/// Python controller's own state; the third is the linearization point,
/// which the nonlinear solver took as a warm start and which this one
/// takes as the plane it expands about. Clarabel starts each solve from
/// scratch, so nothing crosses a step at the solver level.
#[derive(Debug, Clone)]
pub struct PathFollowingMpc<O> {
    settings: PathFollowingSettings,
    occupancy: Option<O>,
    conditioner: CommandConditioner,
    reference: Option<ReferencePath>,
    progress: f64,
    warm: Option<HorizonPlan>,
}

impl<O: Occupancy> PathFollowingMpc<O> {
    /// Builds a controller with no reference set yet.
    ///
    /// # Errors
    ///
    /// Returns whatever [`PathFollowingSettings::validate`] returns, and
    /// [`Error::DimensionMismatch`] when the occupancy describes a space
    /// that is not the plane.
    pub fn new(settings: PathFollowingSettings, occupancy: Option<O>) -> Result<Self, Error> {
        settings.validate()?;
        if let Some(occupancy) = occupancy.as_ref()
            && occupancy.dimension() != 2
        {
            return Err(Error::DimensionMismatch {
                quantity: "occupancy dimension",
                expected: 2,
                actual: occupancy.dimension(),
            });
        }
        let conditioner = CommandConditioner::new(settings.limits)?;
        Ok(Self {
            settings,
            occupancy,
            conditioner,
            reference: None,
            progress: 0.0,
            warm: None,
        })
    }

    /// The settings in force.
    #[must_use]
    pub const fn settings(&self) -> &PathFollowingSettings {
        &self.settings
    }

    /// The reference path, once one has been set.
    #[must_use]
    pub const fn reference(&self) -> Option<&ReferencePath> {
        self.reference.as_ref()
    }

    /// How far along the reference the vehicle has come, meters.
    #[must_use]
    pub const fn progress(&self) -> f64 {
        self.progress
    }

    /// What the two limiters have clipped since the last reset.
    #[must_use]
    pub const fn saturation(&self) -> SaturationReport {
        self.conditioner.report()
    }

    /// Sets or replaces the reference path.
    ///
    /// The polyline is extended by one horizon of straight runway along
    /// the final tangent before the path is built. Without it the bound
    /// holding the path parameter inside the path pinches against its own
    /// ceiling over the last meters, and the program reports an empty
    /// feasible set exactly where the run was about to succeed. The
    /// Python did the same thing for the same reason.
    ///
    /// # Errors
    ///
    /// As [`ReferencePath::new`].
    pub fn set_reference(&mut self, waypoints: &[(f64, f64)]) -> Result<(), Error> {
        let mut points: Vec<(f64, f64)> = waypoints.to_vec();
        if let (Some(&(last_x, last_y)), Some(&(previous_x, previous_y))) =
            (points.last(), points.get(points.len().saturating_sub(2)))
        {
            let (dx, dy) = (last_x - previous_x, last_y - previous_y);
            let length = dx.hypot(dy);
            if length > SEPARATION_FLOOR {
                let runway = self.settings.horizon_distance().max(RUNWAY_FLOOR);
                points.push((
                    (runway * dx / length) + last_x,
                    (runway * dy / length) + last_y,
                ));
            }
        }
        self.reference = Some(ReferencePath::new(&points)?);
        self.reset();
        Ok(())
    }

    /// Forgets the progress, the previous plan, and the limiter counters.
    pub fn reset(&mut self) {
        self.progress = 0.0;
        self.warm = None;
        self.conditioner.reset();
    }

    /// Solves one control step and conditions the command it produces.
    ///
    /// The elapsed interval arrives as an argument and is read against
    /// the configured band and against the model step, per `FR-INV-10`
    /// and deviation A-17. Nothing here reads a clock. The Python
    /// discarded this argument outright and used the configured step for
    /// the horizon, which silently tolerated a loop running at one rate
    /// and a model discretized at another.
    ///
    /// A step that cannot be solved is not an error: it returns a braking
    /// command and says why, per `FR-MPC-04`. So does a step whose
    /// measured state carries a value that is not a real number. An
    /// [`Error`] means the call itself was wrong.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when no reference has been set,
    /// [`Error::NotFinite`] or [`Error::OutOfRange`] when `dt` is outside
    /// the band or disagrees with the model step, and otherwise whatever
    /// the occupancy or the solver wrapper returns.
    pub fn step(&mut self, state: VehicleState, dt: f64) -> Result<PathFollowingStep, Error> {
        self.settings.limits.interval.check(dt)?;
        self.require_model_interval(dt)?;
        let total_length = self.require_reference()?.total_length();

        if state.check().is_err() {
            return self.safe_stop(state, dt, StepFailure::InvalidState, 0.0, 0.0);
        }

        // The hint is clamped into the path first. A window built around
        // a hint past the end excludes every segment, and the projection
        // then reports the origin of the path with no error anywhere,
        // which reads as the vehicle having teleported to the start.
        let window = self
            .settings
            .horizon_distance()
            .max(PROJECTION_WINDOW_FLOOR);
        let hint = self.progress.clamp(0.0, total_length);
        let projection = self
            .require_reference()?
            .project((state.x, state.y, state.heading), Some((hint, window)))?;
        // Progress never rewinds: a recovery arc catches up to the path
        // parameter through the lag cost rather than resetting it.
        self.progress = projection
            .arc_length
            .max(self.progress)
            .clamp(0.0, total_length);

        let cruise = self.preview_cruise()?;
        let obstacles = self.probe_obstacles(state)?;
        let context = StepContext {
            state,
            progress: self.progress,
            cruise,
            total_length,
        };

        match self.run(&obstacles, &context)? {
            Err(failure) => self.safe_stop(
                state,
                dt,
                failure,
                projection.lateral_error,
                projection.heading_error,
            ),
            Ok(solved) => self.accept(
                solved,
                &context,
                dt,
                projection.lateral_error,
                projection.heading_error,
            ),
        }
    }

    /// Takes a solved plan, commands its first predicted state, and keeps it.
    ///
    /// The command is the speed and turn rate of the state after the
    /// current one, not the input that produces it: the input is an
    /// acceleration and a turn-rate change, and handing those to a vehicle
    /// expecting a velocity command is a units error no type here would
    /// catch. The Python commanded the same two numbers.
    fn accept(
        &mut self,
        solved: Solved,
        context: &StepContext,
        dt: f64,
        contour_error: f64,
        heading_error: f64,
    ) -> Result<PathFollowingStep, Error> {
        let commanded = solved.plan.state(1).unwrap_or(context.state);
        let requested = Command {
            speed: commanded.speed,
            turn_rate: commanded.turn_rate,
        };
        let command = self.conditioner.apply(requested, dt)?;
        self.progress = solved
            .plan
            .arc_length(1)
            .unwrap_or(self.progress)
            .max(self.progress)
            .clamp(0.0, context.total_length);
        let predicted_clearance = self.predicted_clearance(&solved.plan)?;
        self.warm = Some(solved.plan.clone());

        Ok(PathFollowingStep {
            command,
            requested,
            contour_error,
            heading_error,
            progress: self.progress,
            predicted_clearance,
            saturation: self.conditioner.report(),
            outcome: StepOutcome::Solved {
                cost: solved.cost,
                sqp_iterations: solved.sqp_iterations,
                solver_iterations: solved.solver_iterations,
                exact: solved.exact,
                convergence: solved.convergence,
            },
            plan: Some(solved.plan),
            linearization: Some(solved.linearization),
        })
    }

    /// Sheds one model step of speed and holds the turn rate.
    ///
    /// The Python held the previous command instead whenever a warm start
    /// existed, and only decelerated without one. Holding a stale turn
    /// rate through a failure is a different closed-loop behavior from
    /// braking, and only one of the two is safe when the reason for the
    /// failure is that the situation changed, so this always brakes.
    ///
    /// Nothing is carried over from the plan: the progress is left where
    /// it was, the previous plan is dropped so the next step linearizes
    /// about a fresh rollout, and the reported clearance is infinite
    /// because no trajectory was predicted to measure one along.
    fn safe_stop(
        &mut self,
        state: VehicleState,
        dt: f64,
        failure: StepFailure,
        contour_error: f64,
        heading_error: f64,
    ) -> Result<PathFollowingStep, Error> {
        let limits = self.settings.limits;
        // Shed one model step of speed and hold the turn rate, then let
        // the conditioner do the clamping. Clamping here as well would
        // reproduce the Python's floor at the minimum speed and leave the
        // saturation report saying nothing happened, which is exactly the
        // separation deviation A-09 asks for.
        let decelerated = state.speed - limits.max_speed_rate * self.settings.step_interval;
        let speed = if decelerated.is_finite() {
            decelerated
        } else {
            limits.min_speed
        };
        let turn_rate = if state.turn_rate.is_finite() {
            state.turn_rate
        } else {
            0.0
        };
        let requested = Command { speed, turn_rate };
        let command = self.conditioner.apply(requested, dt)?;
        self.warm = None;

        Ok(PathFollowingStep {
            command,
            requested,
            contour_error,
            heading_error,
            progress: self.progress,
            predicted_clearance: f64::INFINITY,
            saturation: self.conditioner.report(),
            outcome: StepOutcome::SafeStop(failure),
            plan: None,
            linearization: None,
        })
    }
}

/// What one control step is being solved against.
///
/// Everything a stage needs that does not come from the nominal, bundled
/// so that the assembly and the read-back cannot be handed different
/// copies of it.
#[derive(Debug, Clone, Copy)]
struct StepContext {
    /// The measured state, which stage zero is pinned to.
    state: VehicleState,
    /// The arc length stage zero is pinned to, meters.
    progress: f64,
    /// The progress speed cap for this step, meters per second.
    cruise: f64,
    /// How long the reference is, meters.
    total_length: f64,
}

/// What a solved control step produced.
#[derive(Debug, Clone)]
struct Solved {
    plan: HorizonPlan,
    linearization: HorizonPlan,
    cost: f64,
    solver_iterations: u32,
    exact: bool,
    sqp_iterations: usize,
    convergence: f64,
}

/// What one sequential iteration produced.
#[derive(Debug, Clone)]
struct Iterate {
    plan: HorizonPlan,
    cost: f64,
    solver_iterations: u32,
    exact: bool,
}

/// The outcome of one iteration: an answer, a failure, or a broken call.
///
/// Named because the nesting is the point. The inner [`Result`] separates
/// a step that could not be solved, which is a command, from one that was
/// never posed correctly, which is a bug in the caller or here.
type IterationResult = Result<Result<Iterate, StepFailure>, Error>;

impl<O: Occupancy> PathFollowingMpc<O> {
    /// Runs the sequential loop until it converges or the budget ends.
    ///
    /// The nonconvexity does not disappear when the inner solve becomes
    /// convex; it moves into the choice of linearization point, which is
    /// what this loop is. One restart is held back for the case the
    /// shifted warm start describes a program with no feasible point: the
    /// rollout nominal satisfies every row of its own program by
    /// construction, so retrying from it distinguishes a geometry that is
    /// genuinely infeasible from a linearization point that was merely a
    /// poor guess.
    fn run(
        &self,
        obstacles: &[(f64, f64)],
        context: &StepContext,
    ) -> Result<Result<Solved, StepFailure>, Error> {
        let warm = self.usable_warm_start(context);
        let started_warm = warm.is_some();
        let mut nominal = match warm {
            Some(shifted) => shifted,
            None => self.rollout(context)?,
        };
        let mut restarts = usize::from(started_warm);
        let mut solved: Option<Solved> = None;
        let mut iteration = 0_usize;

        while iteration < self.settings.max_sqp_iterations {
            let linearization = self.linearize(nominal, context)?;
            match self.iterate(&linearization, obstacles, context)? {
                // The iteration counter deliberately does not advance
                // here, so a restart costs one extra solve rather than
                // one of the iterations the caller asked for.
                Err(StepFailure::Solve(SolveFailure::Infeasible)) if restarts > 0 => {
                    restarts = restarts.saturating_sub(1);
                    nominal = self.rollout(context)?;
                }
                Err(failure) => return Ok(Err(failure)),
                Ok(iterate) => {
                    iteration = iteration.saturating_add(1);
                    let Linearization {
                        nominal: previous, ..
                    } = linearization;
                    let convergence = iterate.plan.deviation(&previous);
                    nominal = iterate.plan.clone();
                    solved = Some(Solved {
                        plan: iterate.plan,
                        linearization: previous,
                        cost: iterate.cost,
                        solver_iterations: iterate.solver_iterations,
                        exact: iterate.exact,
                        sqp_iterations: iteration,
                        convergence,
                    });
                    if convergence <= self.settings.sqp_tolerance {
                        break;
                    }
                }
            }
        }

        // The loop body runs at least once, since the iteration budget is
        // refused at zero, so the fallback stands for a case the type
        // system cannot rule out rather than one that happens.
        Ok(solved.ok_or(StepFailure::Solve(SolveFailure::Numerical)))
    }

    /// Assembles one program, solves it, and reads the answer back.
    ///
    /// Three things have to agree before the answer counts. The solver
    /// has to report one, the plan has to hold every bound the program
    /// stated, and the objective this module wrote has to take the value
    /// at that point that the solver says it took. The last is the only
    /// check that can see a wrong coefficient in the cost, since a
    /// feasible point stays feasible whatever the objective says.
    fn iterate(
        &self,
        linearization: &Linearization,
        obstacles: &[(f64, f64)],
        context: &StepContext,
    ) -> IterationResult {
        let Program {
            layout,
            objective,
            constraints,
            bounds,
        } = self.assemble(linearization, obstacles, context)?;

        let problem = QpProblem {
            objective: objective.quadratic(),
            gradient: objective.gradient.clone(),
            constraints,
            bounds,
            equality_rows: layout.equality_rows,
            max_iterations: self.settings.max_solver_iterations,
        };
        let solution = match problem.solve()? {
            Ok(solution) => solution,
            Err(failure) => return Ok(Err(StepFailure::Solve(failure))),
        };

        let plan = HorizonPlan::from_solution(&solution.variables, &layout)?;
        if !self.holds_every_bound(&plan, linearization, context) {
            return Ok(Err(StepFailure::Solve(SolveFailure::Numerical)));
        }
        let cost = objective.value(&solution.variables);
        // Scaled against the largest term that went into the comparison
        // rather than against the result of it. The constant absorbs the
        // squared offsets of the linearization point, so a path a
        // kilometre from the origin carries a constant near 1e12 while
        // the cost it cancels down to is of order one. Judging that
        // difference against the result asks for a relative accuracy no
        // double can carry, and the guard then brakes a healthy
        // controller for being far from the origin.
        let scale = cost
            .abs()
            .max(solution.cost.abs())
            .max(objective.constant.abs())
            .max(1.0);
        if !is_close(
            cost,
            solution.cost + objective.constant,
            COST_AGREEMENT_TOLERANCE * scale,
            COST_AGREEMENT_TOLERANCE,
        ) {
            return Ok(Err(StepFailure::Solve(SolveFailure::Numerical)));
        }

        Ok(Ok(Iterate {
            plan,
            cost,
            solver_iterations: solution.iterations,
            exact: solution.exact,
        }))
    }

    /// The previous plan, shifted by one step, when it is worth reusing.
    ///
    /// A warm start that barely advanced over its whole horizon describes
    /// a vehicle that has stopped, and seeding the next step inside that
    /// basin keeps it stopped. The Python detected the same condition and
    /// fell back to a fresh rollout, and the three escapes are its own:
    /// the horizon advanced, the cruise speed is already at its floor so
    /// there was nothing to advance, or the goal is inside one horizon so
    /// a short horizon advance is the correct answer.
    fn usable_warm_start(&self, context: &StepContext) -> Option<HorizonPlan> {
        let previous = self.warm.as_ref()?;
        if previous.step_count() != self.settings.horizon_step_count || !previous.finite() {
            return None;
        }
        let horizon = self.settings.horizon_step_count;
        let advance = previous.arc_length(horizon).unwrap_or_default()
            - previous.arc_length(0).unwrap_or_default();
        let reach = context.cruise * self.settings.step_interval * count_as_value(horizon);
        let moving = advance > STALL_ADVANCE_FLOOR.max(STALL_ADVANCE_FRACTION * reach);
        let floored = context.cruise <= self.settings.limits.min_speed;
        let near_goal = context.total_length - context.progress < reach;
        (moving || floored || near_goal).then(|| self.shift(previous, context))
    }

    /// Shifts a plan by one step and re-pins it to the current state.
    ///
    /// The final column has no successor, so it repeats, which leaves the
    /// last stage linearized about a state the one before it does not
    /// reach. The arc lengths are raised to the current progress as well
    /// as clipped into the path: a nominal sitting behind the vehicle
    /// makes the progress reward pay a second time for ground already
    /// covered.
    fn shift(&self, previous: &HorizonPlan, context: &StepContext) -> HorizonPlan {
        let horizon = self.settings.horizon_step_count;
        let last = horizon.saturating_sub(1);
        let mut states = Vec::with_capacity(horizon.saturating_add(1));
        let mut arc_lengths = Vec::with_capacity(horizon.saturating_add(1));
        for step in 0..=horizon {
            let source = step.saturating_add(1).min(horizon);
            states.push(previous.state(source).unwrap_or(context.state));
            let arc = previous.arc_length(source).unwrap_or(context.progress);
            arc_lengths.push(arc.max(context.progress).clamp(0.0, context.total_length));
        }
        if let Some(slot) = states.first_mut() {
            *slot = context.state;
        }
        if let Some(slot) = arc_lengths.first_mut() {
            *slot = context.progress;
        }

        let mut inputs = Vec::with_capacity(horizon);
        let mut progress_speeds = Vec::with_capacity(horizon);
        for step in 0..horizon {
            let source = step.saturating_add(1).min(last);
            inputs.push(previous.input(source).unwrap_or_default());
            progress_speeds.push(
                previous
                    .progress_speed(source)
                    .unwrap_or_default()
                    .clamp(0.0, self.settings.limits.max_speed),
            );
        }

        HorizonPlan {
            states,
            inputs,
            arc_lengths,
            progress_speeds,
        }
    }
}

impl<O: Occupancy> PathFollowingMpc<O> {
    /// Builds a nominal that the program it anchors is feasible for.
    ///
    /// Two properties matter and they pull in different directions. The
    /// Python placed every predicted pose on the reference, which is what
    /// let the nonlinear solver escape the stopped equilibrium at a sharp
    /// corner, and it left the guess dynamically inconsistent. Here the
    /// rollout integrates the model instead, choosing a feed-forward
    /// acceleration toward the curve-limited speed and a turn rate toward
    /// the reference heading, both inside their own limits.
    ///
    /// Integrating is what makes the trust region safe. A nominal the
    /// vehicle could not reach can sit further from every reachable
    /// trajectory than the trust radius allows, and the program is then
    /// empty for a reason the geometry does not contain. A nominal
    /// produced by the model satisfies the recurrence exactly, sits
    /// inside every box it was clamped into, and lies at the center of
    /// its own trust region, so it is a feasible point of the program
    /// built around it whenever the measured state is one.
    ///
    /// What survives from the Python is the part that mattered: the
    /// rollout accelerates rather than standing still, so the stages
    /// after the first linearize about a moving vehicle, and a moving
    /// vehicle is one whose heading column reaches its position rows.
    fn rollout(&self, context: &StepContext) -> Result<HorizonPlan, Error> {
        let reference = self.require_reference()?;
        let settings = &self.settings;
        let limits = settings.limits;
        let dt = settings.step_interval;
        let horizon = settings.horizon_step_count;

        let mut states = Vec::with_capacity(horizon.saturating_add(1));
        let mut inputs = Vec::with_capacity(horizon);
        let mut arc_lengths = Vec::with_capacity(horizon.saturating_add(1));
        let mut progress_speeds = Vec::with_capacity(horizon);
        states.push(context.state);
        arc_lengths.push(context.progress);

        let mut state = context.state;
        let mut arc = context.progress;
        for _ in 0..horizon {
            let sample = reference.sample_at(arc);
            let curve_speed =
                limits.max_turn_rate / sample.curvature.abs().max(ROLLOUT_CURVATURE_FLOOR);
            let target_speed = settings
                .cruise_speed
                .min(curve_speed)
                .clamp(limits.min_speed, limits.max_speed);
            let target_turn_rate =
                (angle_difference(sample.heading, state.heading).unwrap_or_default() / dt)
                    .clamp(-limits.max_turn_rate, limits.max_turn_rate);
            let input = ControlInput {
                acceleration: ((target_speed - state.speed) / dt)
                    .clamp(-limits.max_speed_rate, limits.max_speed_rate),
                turn_rate_change: ((target_turn_rate - state.turn_rate) / dt)
                    .clamp(-limits.max_turn_rate_change, limits.max_turn_rate_change),
            };
            let next = unicycle_step(state, input, dt)?;

            // The progress speed obeys every row that will bound it,
            // including the one that stops the path parameter running off
            // the end of the path, which is what keeps the progress
            // equality and the path bound satisfiable together.
            let cap = sample
                .curvature
                .mul_add(sample.curvature, CURVE_CAP_SMOOTHING)
                .sqrt();
            let progress_speed = next
                .speed
                .min(context.cruise)
                .min(limits.max_speed)
                .min(limits.max_turn_rate / cap)
                .min((context.total_length - arc) / dt)
                .max(0.0);
            arc = progress_speed.mul_add(dt, arc);

            states.push(next);
            inputs.push(input);
            arc_lengths.push(arc);
            progress_speeds.push(progress_speed);
            state = next;
        }

        Ok(HorizonPlan {
            states,
            inputs,
            arc_lengths,
            progress_speeds,
        })
    }

    /// Takes the model and the three errors about one nominal.
    fn linearize(
        &self,
        nominal: HorizonPlan,
        context: &StepContext,
    ) -> Result<Linearization, Error> {
        let reference = self.require_reference()?;
        let horizon = self.settings.horizon_step_count;
        let dt = self.settings.step_interval;
        let mut dynamics = Vec::with_capacity(horizon);
        let mut errors = Vec::with_capacity(horizon.saturating_add(1));
        let mut caps = Vec::with_capacity(horizon);

        for step in 0..=horizon {
            let state = nominal.state(step).unwrap_or(context.state);
            let arc = nominal
                .arc_length(step)
                .unwrap_or(context.progress)
                .clamp(0.0, context.total_length);
            let sample = reference.sample_at(arc);
            errors.push(expand_errors(reference, state, arc, sample)?);
            if step < horizon {
                let input = nominal.input(step).unwrap_or_default();
                dynamics.push(linearize_unicycle(state, input, dt)?);
                caps.push(
                    sample
                        .curvature
                        .mul_add(sample.curvature, CURVE_CAP_SMOOTHING)
                        .sqrt(),
                );
            }
        }

        Ok(Linearization {
            nominal,
            dynamics,
            errors,
            caps,
        })
    }

    /// The obstacle points this step's barriers are built against.
    ///
    /// One probe at the vehicle and several along the reference ahead of
    /// it, deduplicated, exactly as the Python collected them. The
    /// Python then padded the list to a fixed five slots with points a
    /// thousand kilometers away, because its symbolic graph had to keep
    /// one shape across steps. This program is rebuilt every step, so the
    /// padding buys nothing and the dummies are dropped.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy returns.
    fn probe_obstacles(&self, state: VehicleState) -> Result<Vec<(f64, f64)>, Error> {
        let Some(occupancy) = self.occupancy.as_ref() else {
            return Ok(Vec::new());
        };
        if self.settings.weight_obstacle <= 0.0 {
            return Ok(Vec::new());
        }
        let reference = self.require_reference()?;
        let look_ahead = self.settings.look_ahead();

        let mut probes: Vec<(f64, f64)> = Vec::with_capacity(OBSTACLE_SAMPLE_COUNT + 1);
        probes.push((state.x, state.y));
        for index in 0..OBSTACLE_SAMPLE_COUNT {
            let sample =
                reference.sample_at(self.probe_arc(index, OBSTACLE_SAMPLE_COUNT, look_ahead));
            probes.push((sample.x, sample.y));
        }

        let mut points: Vec<(f64, f64)> = Vec::with_capacity(probes.len());
        for probe in probes {
            let nearest = occupancy.nearest_obstacle(&[probe.0, probe.1])?;
            let (Some(&x), Some(&y)) = (nearest.point.first(), nearest.point.get(1)) else {
                continue;
            };
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            if !points
                .iter()
                .any(|&(held_x, held_y)| (held_x - x).hypot(held_y - y) <= SEPARATION_FLOOR)
            {
                points.push((x, y));
            }
        }
        Ok(points)
    }

    /// The cruise cap for this step, tapered by the clearance ahead.
    ///
    /// Sampled along the reference rather than around the vehicle, so the
    /// program slows before a pinch point enters the horizon instead of
    /// discovering it inside one.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy returns.
    fn preview_cruise(&self) -> Result<f64, Error> {
        let cruise = self.settings.cruise_speed;
        let Some(occupancy) = self.occupancy.as_ref() else {
            return Ok(cruise);
        };
        let clearance = occupancy.clearance();
        if !(clearance.is_finite() && clearance > 0.0) {
            return Ok(cruise);
        }
        let reference = self.require_reference()?;
        let look_ahead = self.settings.look_ahead();

        let mut closest = f64::INFINITY;
        for index in 0..CRUISE_PREVIEW_COUNT {
            let sample =
                reference.sample_at(self.probe_arc(index, CRUISE_PREVIEW_COUNT, look_ahead));
            let nearest = occupancy.nearest_obstacle(&[sample.x, sample.y])?;
            closest = closest.min(nearest.distance);
        }
        if closest >= clearance {
            return Ok(cruise);
        }
        let scale = (closest / clearance).max(CRUISE_TAPER_FLOOR);
        Ok((cruise * scale).max(self.settings.limits.min_speed))
    }

    /// Where the `index`-th of `count` probes along the reference sits.
    fn probe_arc(&self, index: usize, count: usize, look_ahead: f64) -> f64 {
        let span = count_as_value(count.saturating_sub(1)).max(1.0);
        let fraction = count_as_value(index) / span;
        self.progress + fraction * look_ahead
    }

    /// The smallest distance to an obstacle along a predicted trajectory.
    ///
    /// Measured from the obstacle surface, per deviation A-16, which is
    /// the same convention the barrier rows were written in.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy returns.
    fn predicted_clearance(&self, plan: &HorizonPlan) -> Result<f64, Error> {
        let Some(occupancy) = self.occupancy.as_ref() else {
            return Ok(f64::INFINITY);
        };
        let mut closest = f64::INFINITY;
        for (x, y) in plan.positions() {
            closest = closest.min(occupancy.nearest_obstacle(&[x, y])?.distance);
        }
        Ok(closest)
    }

    /// The reference, or the reason there is not one.
    fn require_reference(&self) -> Result<&ReferencePath, Error> {
        self.reference.as_ref().ok_or(Error::TooFew {
            quantity: "reference waypoints",
            minimum: 2,
            actual: 0,
        })
    }

    /// Rejects an elapsed interval the model was not discretized at.
    ///
    /// The Python discarded the argument and used the configured step,
    /// which is the mismatch that makes a loop running at one rate chase
    /// a model predicting at another. The joint-space controller refuses
    /// the same disagreement and this one now matches it.
    fn require_model_interval(&self, dt: f64) -> Result<(), Error> {
        if is_close(
            dt,
            self.settings.step_interval,
            TIME_TOLERANCE,
            RELATIVE_TOLERANCE,
        ) {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                quantity: "elapsed interval",
                value: dt,
                bound: "the model step the settings declare",
            })
        }
    }
}

/// The model and the three errors, taken about one nominal trajectory.
#[derive(Debug, Clone)]
struct Linearization {
    /// The point everything below was taken about.
    nominal: HorizonPlan,
    /// The affine model of each stage, one per input.
    dynamics: Vec<StageDynamics>,
    /// The three error expansions at each node.
    errors: Vec<PathErrorExpansion>,
    /// The coefficient the curve-limited progress cap carries per stage.
    caps: Vec<f64>,
}

/// Expands the three errors, with the curvature the expansion needs.
///
/// The sample's own curvature is previewed and spread, which is what a
/// speed limit wants and what a Jacobian does not. The expansion reads a
/// curvature measured from the reference heading across
/// [`GRADIENT_CURVATURE_SPAN`], bounded so the arc-length frame stays
/// away from its own singularity.
fn expand_errors(
    reference: &ReferencePath,
    state: VehicleState,
    arc_length: f64,
    sample: PathSample,
) -> Result<PathErrorExpansion, Error> {
    let errors = path_errors(state, sample)?;
    let curvature = frame_curvature(reference, arc_length, errors.contour);
    linearize_path_errors(
        state,
        arc_length,
        PathSample {
            curvature,
            ..sample
        },
    )
}

/// How fast the reference heading turns near `arc_length`, per meter.
fn frame_curvature(reference: &ReferencePath, arc_length: f64, contour: f64) -> f64 {
    let half = GRADIENT_CURVATURE_SPAN * 0.5;
    let ahead = (arc_length + half).min(reference.total_length());
    let behind = (arc_length - half).max(0.0);
    let span = ahead - behind;
    if span <= 0.0 {
        return 0.0;
    }
    let turn = angle_difference(
        reference.sample_at(ahead).heading,
        reference.sample_at(behind).heading,
    )
    .unwrap_or_default();
    // A contouring error of one radius of curvature is where the frame
    // stops being usable, so the product is what gets bounded rather than
    // the curvature alone.
    let ceiling = (FRAME_SINGULARITY_MARGIN / contour.abs()).min(GRADIENT_CURVATURE_CEILING);
    (turn / span).clamp(-ceiling, ceiling)
}

/// A program, and what a reader needs to interpret its answer.
#[derive(Debug)]
struct Program {
    layout: Layout,
    objective: Objective,
    constraints: Triplets,
    bounds: Vec<f64>,
}

impl<O: Occupancy> PathFollowingMpc<O> {
    /// Builds the convex program for one sequential iteration.
    ///
    /// The equality blocks are written first because [`QpProblem`] splits
    /// its cone list at a row index rather than by asking what each row
    /// means: a row written out of order lands in the wrong cone, which
    /// turns a bound into a pin or a pin into a bound, and the solver
    /// reports neither.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the program would be larger
    /// than an index can address, and [`Error::DimensionMismatch`] when
    /// the rows written disagree with the rows declared.
    fn assemble(
        &self,
        linearization: &Linearization,
        obstacles: &[(f64, f64)],
        context: &StepContext,
    ) -> Result<Program, Error> {
        let layout = Layout::new(
            self.settings.horizon_step_count,
            obstacles.len(),
            self.settings.contour_deadzone > 0.0,
        )?;
        let mut objective = Objective::new(layout.variable_count);
        self.push_error_costs(&layout, linearization, &mut objective);
        self.push_input_costs(&layout, &mut objective);
        self.push_barrier_costs(&layout, linearization, obstacles, &mut objective);

        let mut rows = Rows::new(layout.row_count);
        push_initial(&layout, context, &mut rows);
        self.push_dynamics(&layout, linearization, &mut rows);
        rows.seal_equalities();
        self.push_stage_bounds(&layout, linearization, context, &mut rows);
        self.push_state_bounds(&layout, linearization, context, &mut rows);
        self.push_deadzone(&layout, linearization, &mut rows);
        self.push_barriers(&layout, linearization, obstacles, &mut rows);

        let (constraints, bounds) = rows.finish(&layout)?;
        Ok(Program {
            layout,
            objective,
            constraints,
            bounds,
        })
    }

    /// Weighs the three contouring errors at every node.
    ///
    /// Each becomes the square of its affine expansion, which is a rank
    /// one positive semidefinite block over the position, the heading and
    /// the arc length. Three asymmetries of the original survive: the
    /// terminal node weighs its contouring and heading errors with the
    /// terminal weight, it weighs its lag error with the ordinary lag
    /// weight, and stage zero contributes a constant because the initial
    /// pin leaves it nothing to vary.
    fn push_error_costs(
        &self,
        layout: &Layout,
        linearization: &Linearization,
        objective: &mut Objective,
    ) {
        let settings = &self.settings;
        if let Some(expansion) = linearization.errors.first() {
            objective.add_constant(self.pinned_stage_cost(expansion));
        }
        for step in 1..=layout.horizon {
            let Some(expansion) = linearization.errors.get(step) else {
                continue;
            };
            let terminal = step == layout.horizon;
            let contour_weight = if terminal {
                settings.weight_terminal
            } else {
                settings.weight_contour
            };
            let heading_weight = if terminal {
                settings.weight_terminal
            } else {
                settings.weight_heading
            };

            if layout.deadzone {
                objective.add_diagonal(layout.deadzone_slack(step), contour_weight);
            } else {
                let (terms, constant) = error_terms(layout, step, &expansion.contour);
                objective.add_square(&terms, constant, contour_weight);
            }
            let (terms, constant) = error_terms(layout, step, &expansion.lag);
            objective.add_square(&terms, constant, settings.weight_lag);
            let (terms, constant) = error_terms(layout, step, &expansion.heading);
            objective.add_square(&terms, constant, heading_weight);
        }
    }

    /// What the errors at the pinned stage are worth.
    ///
    /// They cannot change the answer, so emitting them as a constant is
    /// the only thing they can honestly do. Dropping them instead would
    /// make the reported cost of a step depend on where the vehicle
    /// happened to be pinned rather than on the plan.
    fn pinned_stage_cost(&self, expansion: &PathErrorExpansion) -> f64 {
        let settings = &self.settings;
        let contour = deadzone_excess(expansion.contour.value, settings.contour_deadzone);
        let lag = expansion.lag.value;
        let heading = wrap_angle(expansion.heading.value).unwrap_or_default();
        settings.weight_contour.mul_add(
            contour * contour,
            settings
                .weight_lag
                .mul_add(lag * lag, settings.weight_heading * heading * heading),
        )
    }

    /// Weighs the control effort and pays for progress.
    ///
    /// The progress reward is linear and negative, which makes it the one
    /// term in the objective that is unbounded below on its own. The two
    /// caps on the progress speed are what keep the program bounded, so a
    /// missing cap row shows up as the solver reporting an unbounded
    /// objective rather than as a slightly wrong answer.
    fn push_input_costs(&self, layout: &Layout, objective: &mut Objective) {
        let settings = &self.settings;
        let reward = -settings.weight_progress * settings.step_interval;
        for step in 0..layout.horizon {
            objective.add_diagonal(layout.input(step, 0), settings.weight_control);
            objective.add_diagonal(layout.input(step, 1), settings.weight_control);
            objective.add_linear(layout.progress_speed(step), reward);
        }
    }

    /// Weighs each obstacle's penetration slack.
    ///
    /// The forward cone factor of the original multiplied the barrier by
    /// how directly the vehicle was pointed at the obstacle. Linearizing
    /// it would put a product of two variables back into the cost and
    /// undo the convexity, so it freezes at the nominal heading and
    /// scales the weight instead. The sequential loop recovers the
    /// directionality, because the nominal heading moves between
    /// iterations.
    fn push_barrier_costs(
        &self,
        layout: &Layout,
        linearization: &Linearization,
        obstacles: &[(f64, f64)],
        objective: &mut Objective,
    ) {
        for (slot, &obstacle) in obstacles.iter().enumerate() {
            for step in 1..=layout.horizon {
                let Some(state) = linearization.nominal.state(step) else {
                    continue;
                };
                let cone = forward_cone(state, obstacle);
                let weight = self.settings.weight_obstacle * CONE_SPAN.mul_add(cone, CONE_FLOOR);
                objective.add_diagonal(layout.obstacle_slack(slot, step), weight);
            }
        }
    }

    /// Writes the linearized recurrence and the virtual progress law.
    ///
    /// The recurrence is in absolute variables, so the row reads
    /// `x_next - A x - B u = r` and the residual is what the affine model
    /// carries that the two Jacobians do not. The progress law needs no
    /// linearization at all: it was already linear.
    fn push_dynamics(&self, layout: &Layout, linearization: &Linearization, rows: &mut Rows) {
        let dt = self.settings.step_interval;
        for step in 0..layout.horizon {
            let Some(model) = linearization.dynamics.get(step) else {
                continue;
            };
            let next = step.saturating_add(1);
            for (component, (state_row, input_row)) in model
                .state_jacobian
                .iter()
                .zip(&model.input_jacobian)
                .enumerate()
            {
                let residual = model.residual.get(component).copied().unwrap_or_default();
                let row = rows.open(residual);
                rows.push(row, layout.state(next, component), 1.0);
                for (column, &coefficient) in state_row.iter().enumerate() {
                    rows.push(row, layout.state(step, column), -coefficient);
                }
                for (column, &coefficient) in input_row.iter().enumerate() {
                    rows.push(row, layout.input(step, column), -coefficient);
                }
            }

            let row = rows.open(0.0);
            rows.push(row, layout.arc(next), 1.0);
            rows.push(row, layout.arc(step), -1.0);
            rows.push(row, layout.progress_speed(step), -dt);
        }
    }

    /// Bounds everything a stage commands.
    ///
    /// The progress speed carries four rows rather than one. Three are
    /// the box and the cruise cap; the fourth is the curve limit, which
    /// the original wrote as `v_s sqrt(K^2 + eps) <= max_turn_rate` and
    /// which is bilinear in the progress speed and the arc length before
    /// the curvature is frozen. Freezing it at the nominal arc length
    /// makes the row linear, and the sequential loop recovers the
    /// coupling because the nominal moves.
    fn push_stage_bounds(
        &self,
        layout: &Layout,
        linearization: &Linearization,
        context: &StepContext,
        rows: &mut Rows,
    ) {
        let limits = self.settings.limits;
        for step in 0..layout.horizon {
            let progress_speed = layout.progress_speed(step);
            let row = rows.open(limits.max_speed);
            rows.push(row, progress_speed, 1.0);
            let row = rows.open(0.0);
            rows.push(row, progress_speed, -1.0);
            let row = rows.open(context.cruise);
            rows.push(row, progress_speed, 1.0);
            // Never zero, because of the smoothing under the root. A
            // coefficient that evaluates to zero is dropped by the sparse
            // builder and leaves a row with no entries, which reads as
            // `0 <= max_turn_rate` and bounds nothing at all.
            let cap = linearization.caps.get(step).copied().unwrap_or(1.0);
            let row = rows.open(limits.max_turn_rate);
            rows.push(row, progress_speed, cap);

            let acceleration = layout.input(step, 0);
            let row = rows.open(limits.max_speed_rate);
            rows.push(row, acceleration, 1.0);
            let row = rows.open(limits.max_speed_rate);
            rows.push(row, acceleration, -1.0);

            let turn_rate_change = layout.input(step, 1);
            let row = rows.open(limits.max_turn_rate_change);
            rows.push(row, turn_rate_change, 1.0);
            let row = rows.open(limits.max_turn_rate_change);
            rows.push(row, turn_rate_change, -1.0);
        }
    }

    /// Bounds every predicted state, and holds it near the nominal.
    ///
    /// Node zero carries none of these. It is pinned to the measurement,
    /// and a bound on a pinned variable either says nothing or empties
    /// the feasible set for a reason the caller cannot act on. Arriving
    /// outside a limit shows up on node one instead, which is bounded and
    /// which no input inside its own limit can always bring back.
    fn push_state_bounds(
        &self,
        layout: &Layout,
        linearization: &Linearization,
        context: &StepContext,
        rows: &mut Rows,
    ) {
        let settings = &self.settings;
        let limits = settings.limits;
        for step in 1..=layout.horizon {
            let speed = layout.state(step, 3);
            let row = rows.open(limits.max_speed);
            rows.push(row, speed, 1.0);
            let row = rows.open(-limits.min_speed);
            rows.push(row, speed, -1.0);

            let turn_rate = layout.state(step, 4);
            let row = rows.open(limits.max_turn_rate);
            rows.push(row, turn_rate, 1.0);
            let row = rows.open(limits.max_turn_rate);
            rows.push(row, turn_rate, -1.0);

            let arc = layout.arc(step);
            let row = rows.open(context.total_length);
            rows.push(row, arc, 1.0);
            let row = rows.open(0.0);
            rows.push(row, arc, -1.0);

            let nominal = linearization.nominal.state(step).unwrap_or(context.state);
            let heading = layout.state(step, 2);
            let row = rows.open(nominal.heading + settings.trust_heading);
            rows.push(row, heading, 1.0);
            let row = rows.open(settings.trust_heading - nominal.heading);
            rows.push(row, heading, -1.0);

            let nominal_arc = linearization
                .nominal
                .arc_length(step)
                .unwrap_or(context.progress);
            let row = rows.open(nominal_arc + settings.trust_arc_length);
            rows.push(row, arc, 1.0);
            let row = rows.open(settings.trust_arc_length - nominal_arc);
            rows.push(row, arc, -1.0);
        }
    }

    /// Writes the epigraph of the deadzone, when one is configured.
    ///
    /// `max(|e| - d, 0)^2` is convex, so the two rows and the slack
    /// describe the same function the original penalized rather than an
    /// approximation of it. The slack needs no lower bound: its quadratic
    /// cost drives it to zero wherever both rows are already satisfied.
    fn push_deadzone(&self, layout: &Layout, linearization: &Linearization, rows: &mut Rows) {
        if !layout.deadzone {
            return;
        }
        let band = self.settings.contour_deadzone;
        for step in 1..=layout.horizon {
            let Some(expansion) = linearization.errors.get(step) else {
                continue;
            };
            let (terms, constant) = error_terms(layout, step, &expansion.contour);
            let slack = layout.deadzone_slack(step);

            let row = rows.open(band - constant);
            for &(index, coefficient) in &terms {
                rows.push(row, index, coefficient);
            }
            rows.push(row, slack, -1.0);

            let row = rows.open(band + constant);
            for &(index, coefficient) in &terms {
                rows.push(row, index, -coefficient);
            }
            rows.push(row, slack, -1.0);
        }
    }

    /// Writes one soft half-space per obstacle per predicted node.
    ///
    /// The set a barrier is trying to describe is the complement of a
    /// disc, which is not convex, so no program of this kind holds it.
    /// The half-space through the nominal position with its normal
    /// pointing away from the obstacle is the tangent inner
    /// approximation: every point it admits is genuinely outside the
    /// clearance, and some points that are outside the clearance it
    /// refuses. Erring that way is the right direction for an obstacle.
    ///
    /// The slack keeps an infeasible geometry from emptying the feasible
    /// set, and normalizing by the clearance keeps it reading as the same
    /// dimensionless penetration the Python raised to the fourth power.
    fn push_barriers(
        &self,
        layout: &Layout,
        linearization: &Linearization,
        obstacles: &[(f64, f64)],
        rows: &mut Rows,
    ) {
        let clearance = self.barrier_clearance();
        for (slot, &(obstacle_x, obstacle_y)) in obstacles.iter().enumerate() {
            for step in 1..=layout.horizon {
                let row = rows.open(0.0);
                // Written first, so that a step whose nominal sits on top
                // of the obstacle still leaves a row with an entry. A row
                // with no entries at all is indistinguishable from one
                // whose coefficient was forgotten.
                rows.push(row, layout.obstacle_slack(slot, step), -1.0);

                let Some(state) = linearization.nominal.state(step) else {
                    continue;
                };
                let separation = (state.x - obstacle_x).hypot(state.y - obstacle_y);
                if !(separation.is_finite() && separation > SEPARATION_FLOOR) {
                    continue;
                }
                let normal_x = (state.x - obstacle_x) / separation;
                let normal_y = (state.y - obstacle_y) / separation;
                rows.push(row, layout.state(step, 0), -normal_x / clearance);
                rows.push(row, layout.state(step, 1), -normal_y / clearance);
                rows.set_bound(
                    row,
                    -1.0 - normal_x.mul_add(obstacle_x, normal_y * obstacle_y) / clearance,
                );
            }
        }
    }

    /// The clearance a barrier normalizes a penetration by, floored.
    fn barrier_clearance(&self) -> f64 {
        self.occupancy
            .as_ref()
            .map_or(CLEARANCE_FLOOR, |occupancy| {
                occupancy.clearance().max(CLEARANCE_FLOOR)
            })
    }
}

impl<O: Occupancy> PathFollowingMpc<O> {
    /// Whether a returned plan is one the vehicle may be commanded from.
    ///
    /// `FR-MPC-03`, read off the answer rather than trusted of the
    /// solver. Every row the program states is checked here, in both
    /// directions for the equalities: an equality that leaked into the
    /// inequality cone still holds one way round, and only reading it
    /// back as an equality tells the two apart.
    fn holds_every_bound(
        &self,
        plan: &HorizonPlan,
        linearization: &Linearization,
        context: &StepContext,
    ) -> bool {
        plan.finite()
            && plan.step_count() == self.settings.horizon_step_count
            && holds_pins(plan, context)
            && self.holds_dynamics(plan, linearization)
            && self.holds_stage_bounds(plan, linearization, context)
            && self.holds_state_bounds(plan, linearization, context)
    }

    /// Whether the plan obeys the affine model it was built against.
    fn holds_dynamics(&self, plan: &HorizonPlan, linearization: &Linearization) -> bool {
        let dt = self.settings.step_interval;
        for step in 0..plan.step_count() {
            let (Some(state), Some(input), Some(model)) = (
                plan.state(step),
                plan.input(step),
                linearization.dynamics.get(step),
            ) else {
                return false;
            };
            let next = step.saturating_add(1);
            let (Some(reached), Some(arc), Some(next_arc), Some(progress_speed)) = (
                plan.state(next),
                plan.arc_length(step),
                plan.arc_length(next),
                plan.progress_speed(step),
            ) else {
                return false;
            };
            let predicted = model.propagate(state, input);
            if !reached
                .to_array()
                .into_iter()
                .zip(predicted.to_array())
                .all(|(here, there)| near(here, there))
            {
                return false;
            }
            if !near(next_arc, progress_speed.mul_add(dt, arc)) {
                return false;
            }
        }
        true
    }

    /// Whether every commanded quantity sits inside its own limit.
    fn holds_stage_bounds(
        &self,
        plan: &HorizonPlan,
        linearization: &Linearization,
        context: &StepContext,
    ) -> bool {
        let limits = self.settings.limits;
        for step in 0..plan.step_count() {
            let (Some(input), Some(progress_speed)) = (plan.input(step), plan.progress_speed(step))
            else {
                return false;
            };
            let cap = linearization.caps.get(step).copied().unwrap_or(1.0);
            let within = at_most(progress_speed, limits.max_speed)
                && at_least(progress_speed, 0.0)
                && at_most(progress_speed, context.cruise)
                && at_most(cap * progress_speed, limits.max_turn_rate)
                && at_most(input.acceleration.abs(), limits.max_speed_rate)
                && at_most(input.turn_rate_change.abs(), limits.max_turn_rate_change);
            if !within {
                return false;
            }
        }
        true
    }

    /// Whether every predicted state sits inside its box and its region.
    fn holds_state_bounds(
        &self,
        plan: &HorizonPlan,
        linearization: &Linearization,
        context: &StepContext,
    ) -> bool {
        let settings = &self.settings;
        let limits = settings.limits;
        for step in 1..=plan.step_count() {
            let (Some(state), Some(arc)) = (plan.state(step), plan.arc_length(step)) else {
                return false;
            };
            let nominal = linearization.nominal.state(step).unwrap_or(context.state);
            let nominal_arc = linearization
                .nominal
                .arc_length(step)
                .unwrap_or(context.progress);
            let within = at_most(state.speed, limits.max_speed)
                && at_least(state.speed, limits.min_speed)
                && at_most(state.turn_rate.abs(), limits.max_turn_rate)
                && at_most(arc, context.total_length)
                && at_least(arc, 0.0)
                && at_most(
                    (state.heading - nominal.heading).abs(),
                    settings.trust_heading,
                )
                && at_most((arc - nominal_arc).abs(), settings.trust_arc_length);
            if !within {
                return false;
            }
        }
        true
    }
}

/// The objective under construction, as this module states it.
///
/// Held here rather than pushed straight into the solver's builder for
/// one reason: the value of the assembled objective at the returned point
/// is the only check that can see a wrong coefficient in a cost. Every
/// constraint still holds when a cross term is halved or a constant is
/// dropped, and the answer is still wrong.
///
/// Entries are the upper triangle of the quadratic term. Clarabel copies
/// the entries on and above the diagonal and discards the rest without
/// complaining, so an entry written below the diagonal disappears and the
/// coupling it described is silently absent; and because the solver
/// mirrors what it keeps, a single entry above the diagonal has to carry
/// the whole of the coefficient rather than half of it.
#[derive(Debug, Clone)]
struct Objective {
    entries: Vec<(usize, usize, f64)>,
    gradient: Vec<f64>,
    constant: f64,
}

impl Objective {
    /// An empty objective over `variables` variables.
    fn new(variables: usize) -> Self {
        Self {
            entries: Vec::new(),
            gradient: vec![0.0; variables],
            constant: 0.0,
        }
    }

    /// Adds a term no variable appears in.
    fn add_constant(&mut self, amount: f64) {
        self.constant += amount;
    }

    /// Adds `coefficient` times one variable.
    fn add_linear(&mut self, index: usize, coefficient: f64) {
        if let Some(slot) = self.gradient.get_mut(index) {
            *slot += coefficient;
        }
    }

    /// Adds `weight` times the square of one variable.
    fn add_diagonal(&mut self, index: usize, weight: f64) {
        if weight > 0.0 {
            self.entries.push((index, index, 2.0 * weight));
        }
    }

    /// Adds `weight` times the square of an affine form.
    ///
    /// The form is the sum of each term's coefficient times its variable,
    /// plus `constant`. The indices in `terms` have to be distinct: two
    /// terms naming one variable would need merging before squaring, and
    /// nothing here merges them.
    fn add_square(&mut self, terms: &[(usize, f64)], constant: f64, weight: f64) {
        if weight <= 0.0 {
            return;
        }
        self.constant += weight * constant * constant;
        for (position, &(index, coefficient)) in terms.iter().enumerate() {
            if coefficient == 0.0 {
                continue;
            }
            self.add_linear(index, 2.0 * weight * constant * coefficient);
            for &(other, other_coefficient) in terms.iter().skip(position) {
                if other_coefficient == 0.0 {
                    continue;
                }
                let value = 2.0 * weight * coefficient * other_coefficient;
                self.entries
                    .push((index.min(other), index.max(other), value));
            }
        }
    }

    /// The quadratic term, in the shape the solver's builder takes.
    fn quadratic(&self) -> Triplets {
        let mut triplets = Triplets::new();
        for &(row, column, value) in &self.entries {
            triplets.push(row, column, value);
        }
        triplets
    }

    /// The objective evaluated at a point.
    ///
    /// One half of the symmetric quadratic form, plus the linear term,
    /// plus the constant, which is the same convention the solver reports
    /// its own objective value in once the constant is added back.
    fn value(&self, variables: &[f64]) -> f64 {
        let read = |index: usize| variables.get(index).copied().unwrap_or_default();
        let mut total = self.constant;
        for &(row, column, value) in &self.entries {
            // An off-diagonal entry stands for itself and its mirror, so
            // the half in front of the form cancels against the two of
            // them and only the diagonal keeps it.
            let share = if row == column { 0.5 } else { 1.0 };
            total += share * value * read(row) * read(column);
        }
        for (index, coefficient) in self.gradient.iter().enumerate() {
            total += coefficient * read(index);
        }
        total
    }
}

/// The constraint block under construction.
///
/// Rows are opened in order and filled afterward, which is what keeps the
/// equality rows ahead of the inequality rows without anyone counting.
/// Three things are checked when the block closes, and each of them stands
/// for a failure that is otherwise completely silent: a row count that
/// disagrees with the layout, an equality boundary in the wrong place, and
/// a row nobody wrote a coefficient into. The last is the quiet one. An
/// empty row reads as `0 <= b`, every point satisfies it, and the bound it
/// was supposed to state is simply absent from the program.
#[derive(Debug)]
struct Rows {
    matrix: Triplets,
    bounds: Vec<f64>,
    filled: Vec<bool>,
    next: usize,
    equalities: Option<usize>,
}

impl Rows {
    /// A block sized for `row_count` rows.
    fn new(row_count: usize) -> Self {
        Self {
            matrix: Triplets::new(),
            bounds: vec![0.0; row_count],
            filled: vec![false; row_count],
            next: 0,
            equalities: None,
        }
    }

    /// Opens the next row at `bound`, returning its index.
    fn open(&mut self, bound: f64) -> usize {
        let row = self.next;
        self.set_bound(row, bound);
        self.next = self.next.saturating_add(1);
        row
    }

    /// Replaces the bound on an open row.
    fn set_bound(&mut self, row: usize, bound: f64) {
        if let Some(slot) = self.bounds.get_mut(row) {
            *slot = bound;
        }
    }

    /// Adds a coefficient to an open row.
    ///
    /// A coefficient of exactly zero does not count as having filled the
    /// row, because the sparse builder drops it and the row is then as
    /// empty as one nobody touched.
    fn push(&mut self, row: usize, column: usize, value: f64) {
        self.matrix.push(row, column, value);
        if value != 0.0
            && let Some(slot) = self.filled.get_mut(row)
        {
            *slot = true;
        }
    }

    /// Records that every row opened so far is an equality.
    fn seal_equalities(&mut self) {
        self.equalities = Some(self.next);
    }

    /// Closes the block, checking it against the layout it was sized by.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the rows written, the
    /// equality boundary, or the rows carrying a coefficient disagree
    /// with what the layout declared.
    fn finish(self, layout: &Layout) -> Result<(Triplets, Vec<f64>), Error> {
        if self.next != layout.row_count {
            return Err(Error::DimensionMismatch {
                quantity: "constraint rows written",
                expected: layout.row_count,
                actual: self.next,
            });
        }
        if self.equalities != Some(layout.equality_rows) {
            return Err(Error::DimensionMismatch {
                quantity: "equality rows written",
                expected: layout.equality_rows,
                actual: self.equalities.unwrap_or_default(),
            });
        }
        let filled = self.filled.iter().filter(|&&seen| seen).count();
        if filled != layout.row_count {
            return Err(Error::DimensionMismatch {
                quantity: "constraint rows carrying a coefficient",
                expected: layout.row_count,
                actual: filled,
            });
        }
        Ok((self.matrix, self.bounds))
    }
}

/// Where each variable and each row of the program sits.
///
/// Every accessor clamps its step and its component into the block it
/// belongs to. The arithmetic is checked once in [`Layout::new`], so the
/// clamps are not about overflow: they are about an index one past the
/// end of one block landing on a live variable of the next one, which
/// every check downstream accepts because the index is inside the
/// program.
#[derive(Debug, Clone, Copy)]
struct Layout {
    horizon: usize,
    deadzone: bool,
    input_block: usize,
    arc_block: usize,
    progress_block: usize,
    deadzone_block: usize,
    obstacle_block: usize,
    variable_count: usize,
    equality_rows: usize,
    row_count: usize,
}

impl Layout {
    /// Lays out a program of the given shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the program would be larger
    /// than an index can address.
    fn new(horizon: usize, obstacles: usize, deadzone: bool) -> Result<Self, Error> {
        let too_large = || Error::OutOfRange {
            quantity: "program size",
            value: count_as_value(horizon),
            bound: "addressable by an index",
        };
        let nodes = horizon.checked_add(1).ok_or_else(too_large)?;
        let states = nodes.checked_mul(STATE_DIMENSION).ok_or_else(too_large)?;
        let inputs = horizon.checked_mul(INPUT_DIMENSION).ok_or_else(too_large)?;
        let deadzones = if deadzone { horizon } else { 0 };
        let barriers = horizon.checked_mul(obstacles).ok_or_else(too_large)?;

        let input_block = states;
        let arc_block = input_block.checked_add(inputs).ok_or_else(too_large)?;
        let progress_block = arc_block.checked_add(nodes).ok_or_else(too_large)?;
        let deadzone_block = progress_block.checked_add(horizon).ok_or_else(too_large)?;
        let obstacle_block = deadzone_block
            .checked_add(deadzones)
            .ok_or_else(too_large)?;
        let variable_count = obstacle_block.checked_add(barriers).ok_or_else(too_large)?;

        // Five rows pinning the state and one pinning the arc length,
        // then the five-row recurrence and the progress law per step.
        let equality_rows = horizon
            .checked_mul(6)
            .and_then(|dynamics| dynamics.checked_add(6))
            .ok_or_else(too_large)?;
        // Eight bounds per stage, ten per predicted node, two rows per
        // deadzone slack and one half-space per barrier slack.
        let row_count = horizon
            .checked_mul(18)
            .and_then(|bounds| bounds.checked_add(equality_rows))
            .and_then(|rows| {
                deadzones
                    .checked_mul(2)
                    .and_then(|band| rows.checked_add(band))
            })
            .and_then(|rows| rows.checked_add(barriers))
            .ok_or_else(too_large)?;

        Ok(Self {
            horizon,
            deadzone,
            input_block,
            arc_block,
            progress_block,
            deadzone_block,
            obstacle_block,
            variable_count,
            equality_rows,
            row_count,
        })
    }

    /// The column holding one component of one predicted state.
    fn state(&self, step: usize, component: usize) -> usize {
        step.min(self.horizon)
            .saturating_mul(STATE_DIMENSION)
            .saturating_add(component.min(STATE_DIMENSION.saturating_sub(1)))
    }

    /// The column holding one component of one commanded input.
    fn input(&self, step: usize, component: usize) -> usize {
        self.input_block
            .saturating_add(
                step.min(self.horizon.saturating_sub(1))
                    .saturating_mul(INPUT_DIMENSION),
            )
            .saturating_add(component.min(INPUT_DIMENSION.saturating_sub(1)))
    }

    /// The column holding the path parameter at one node.
    fn arc(&self, step: usize) -> usize {
        self.arc_block.saturating_add(step.min(self.horizon))
    }

    /// The column holding the virtual progress speed leaving one stage.
    fn progress_speed(&self, step: usize) -> usize {
        self.progress_block
            .saturating_add(step.min(self.horizon.saturating_sub(1)))
    }

    /// The column holding the deadzone excess at one node.
    ///
    /// Nodes run from one, since the contouring error at the pinned node
    /// is a constant and a slack carrying it would be one too.
    fn deadzone_slack(&self, step: usize) -> usize {
        self.deadzone_block
            .saturating_add(step.saturating_sub(1).min(self.horizon.saturating_sub(1)))
    }

    /// The column holding one obstacle's penetration at one node.
    fn obstacle_slack(&self, slot: usize, step: usize) -> usize {
        self.obstacle_block
            .saturating_add(slot.saturating_mul(self.horizon))
            .saturating_add(step.saturating_sub(1).min(self.horizon.saturating_sub(1)))
    }
}

/// The affine form of one error, as terms and a constant.
///
/// Six terms, one per state component and one for the arc length, with
/// the zeros left in: [`Objective::add_square`] and the deadzone rows
/// both skip a zero coefficient, and keeping the shape fixed means the
/// caller never has to decide which components an error depends on.
fn error_terms(
    layout: &Layout,
    step: usize,
    expansion: &ErrorExpansion,
) -> ([(usize, f64); 6], f64) {
    let [x, y, heading, speed, turn_rate] = expansion.state_gradient;
    (
        [
            (layout.state(step, 0), x),
            (layout.state(step, 1), y),
            (layout.state(step, 2), heading),
            (layout.state(step, 3), speed),
            (layout.state(step, 4), turn_rate),
            (layout.arc(step), expansion.arc_length_gradient),
        ],
        expansion.constant,
    )
}

/// Pins the horizon to the state and the progress the step arrived with.
fn push_initial(layout: &Layout, context: &StepContext, rows: &mut Rows) {
    for (component, value) in context.state.to_array().into_iter().enumerate() {
        let row = rows.open(value);
        rows.push(row, layout.state(0, component), 1.0);
    }
    let row = rows.open(context.progress);
    rows.push(row, layout.arc(0), 1.0);
}

/// Whether the first node is where the vehicle said it was.
fn holds_pins(plan: &HorizonPlan, context: &StepContext) -> bool {
    let (Some(first), Some(arc)) = (plan.state(0), plan.arc_length(0)) else {
        return false;
    };
    first
        .to_array()
        .into_iter()
        .zip(context.state.to_array())
        .all(|(here, there)| near(here, there))
        && near(arc, context.progress)
}

/// How far a contouring error sticks out of its free band, meters.
fn deadzone_excess(contour: f64, band: f64) -> f64 {
    (contour.abs() - band).max(0.0)
}

/// How directly a pose points at an obstacle, as a factor in `[0, 1]`.
///
/// The cosine of the angle between the heading and the bearing to the
/// obstacle, floored at zero, which is the Python `forward_cone_factor`
/// written without its symbolic scaffolding. An obstacle behind the
/// vehicle scores nothing, since driving away from something is not a
/// reason to weigh it.
fn forward_cone(state: VehicleState, obstacle: (f64, f64)) -> f64 {
    let (dx, dy) = (obstacle.0 - state.x, obstacle.1 - state.y);
    let distance = dx.hypot(dy);
    if distance <= SEPARATION_FLOOR {
        return 1.0;
    }
    let (sin_heading, cos_heading) = state.heading.sin_cos();
    (cos_heading.mul_add(dx, sin_heading * dy) / distance).clamp(0.0, 1.0)
}

/// Whether two numbers agree to the solver's feasibility tolerance.
fn near(left: f64, right: f64) -> bool {
    (left - right).abs() <= CONSTRAINT_TOLERANCE
}

/// Whether a value sits at or below a bound, to that same tolerance.
fn at_most(value: f64, bound: f64) -> bool {
    value <= bound + CONSTRAINT_TOLERANCE
}

/// Whether a value sits at or above a bound, to that same tolerance.
fn at_least(value: f64, bound: f64) -> bool {
    value >= bound - CONSTRAINT_TOLERANCE
}

/// A count, as the `f64` an [`Error`] carries.
fn count_as_value(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}
