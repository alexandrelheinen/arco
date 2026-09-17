//! Tracking a configuration with one convex program per control step.
//!
//! The joint-space half of ADR-002. [`JointSpaceMpc`] replaces the
//! `CasADi` program of `arco.control.mpc.joint_space` with a single convex
//! program per step, and it can do that because the joint model is
//! already linear in its states: a configuration integrates its velocity
//! and a velocity integrates its acceleration. There is no model to
//! linearize, so there is no sequential loop around the solve and the
//! answer is the exact optimum of the problem posed rather than the last
//! iterate of an approximation to it. The contouring controller needs
//! that loop; this one does not.
//!
//! One part of the Python formulation could not carry over unchanged. Its
//! soft obstacle barrier raised a normalized clearance penetration to the
//! fourth power, and that function is neither quadratic nor convex, so no
//! quadratic program expresses it. Here each obstacle becomes a half-space
//! through a nominal trajectory, the penetration is measured along that
//! half-space's normal, and the penalty is its square. A quadratic grows
//! more slowly than a quartic once the clearance is breached, so the port
//! reacts earlier and less violently than the Python did for the same
//! weight.
//!
//! What does carry over unchanged is every constraint: the initial state,
//! both integrator recurrences, the per-axis velocity bound on each
//! predicted step, and the per-axis acceleration bound on each input.
//! `FR-MPC-03` is checked against the returned solution rather than
//! assumed of the solver, and a solution that breaks a bound by more than
//! [`CONSTRAINT_TOLERANCE`] is refused as a numerical failure instead of
//! being commanded.
//!
//! Three things follow from the objective being a sum of squares. It is
//! bounded below by zero, so [`SolveFailure::Unbounded`] is not reachable
//! from here and a program that reports it means this module built
//! something it did not mean to. The barrier is soft, so an obstacle never
//! makes the program infeasible. And the only constraint that can empty
//! the feasible set is the velocity bound, which no acceleration inside
//! its own bound can satisfy when the machine arrives moving faster than
//! the axis is allowed to move: a real condition, reported as
//! [`SolveFailure::Infeasible`] and answered with a brake.

use arco_core::Error;
use arco_core::numeric::{POSITION_TOLERANCE, RELATIVE_TOLERANCE, is_close};
use arco_core::protocols::Occupancy;

use crate::joint::JointLimits;
use crate::limits::IntervalBand;
use crate::mpc::qp::{QpProblem, QpSolution, SolveFailure, Triplets};

/// The longest horizon a controller will assemble, steps.
///
/// Bounded because the program's size is quadratic in nothing but is
/// linear in this, and a caller passing a horizon of a million would
/// otherwise discover the limit as an allocation failure inside the
/// solver rather than as an error at construction.
pub const MAX_HORIZON_STEPS: usize = 512;

/// How far ahead the second obstacle probe looks, in model steps.
///
/// The Python probed `q + v * dt * 4`, and the number is unchanged: far
/// enough that the barrier sees an obstacle before the horizon reaches
/// it, near enough that the probe still describes where the machine is
/// going.
const OBSTACLE_PROBE_STEPS: f64 = 4.0;

/// The slowest motion worth probing ahead along, units per second.
const PROBE_SPEED_FLOOR: f64 = 1e-6;

/// The smallest clearance the barrier will normalize by, configuration units.
///
/// The penetration is a fraction of the clearance, so a clearance at or
/// below zero would divide the whole barrier by nothing.
const CLEARANCE_FLOOR: f64 = 1e-3;

/// How near an obstacle the linearization point may sit, configuration units.
///
/// The half-space normal is the direction from the obstacle to the point
/// the barrier is linearized about. At zero separation that direction does
/// not exist, and no unit vector is a better answer than leaving the
/// barrier off for that step.
const SEPARATION_FLOOR: f64 = 1e-9;

/// How collinear a barrier normal may be with the direction of travel.
///
/// Above this, the half-space of [`JointSpaceMpc::push_barriers`] points
/// back along the route and the only motion it permits is stopping. The
/// program is then asked to go around by turning the normal sideways
/// instead, per deviation A-33.
const COLLINEAR_NORMAL: f64 = 0.9;

/// How far outside a bound a returned solution may sit.
///
/// Clarabel reports a primal residual rather than an exact answer, so
/// every bound holds to a tolerance or not at all. This is the loosest
/// residual it will still call an answer, which is the reduced feasibility
/// tolerance it falls back to when it cannot reach the tight one, and it
/// is stated here rather than read from the solver so that tuning the
/// solver cannot quietly loosen what `FR-MPC-03` is checked against.
pub const CONSTRAINT_TOLERANCE: f64 = 1e-4;

/// How a joint-space model-predictive controller should behave.
///
/// The defaults are the Python dataclass defaults, so a controller built
/// from [`JointMpcSettings::new`] poses the same problem the `CasADi`
/// version posed, up to the barrier shape the module documentation
/// describes.
#[derive(Debug, Clone, PartialEq)]
pub struct JointMpcSettings {
    /// The limits every axis obeys.
    pub limits: JointLimits,
    /// How many steps the program looks ahead.
    pub horizon_step_count: usize,
    /// The interval one horizon step covers, seconds.
    ///
    /// A step is refused unless the elapsed interval it is given agrees
    /// with this, because a model that advances by one number while the
    /// loop advances by another predicts a trajectory the machine never
    /// follows and chases the difference every step.
    pub step_interval: f64,
    /// Weight on the distance from the target configuration.
    pub weight_tracking: f64,
    /// Weight on moving at all, which is what damps the approach.
    pub weight_velocity: f64,
    /// Weight on the commanded acceleration.
    pub weight_control: f64,
    /// Weight on penetrating an obstacle's clearance.
    ///
    /// Zero switches the barrier off, which is also what a controller
    /// given no occupancy does.
    pub weight_obstacle: f64,
    /// The interior-point iteration budget, per `FR-SAFE-02`.
    ///
    /// There is no sequential loop here, so this is the whole budget for
    /// the step rather than the inner half of one.
    pub max_solver_iterations: u32,
    /// What elapsed interval a step will accept, per `FR-INV-10`.
    pub interval: IntervalBand,
}

impl JointMpcSettings {
    /// Builds settings around `limits`, carrying the Python defaults.
    #[must_use]
    pub fn new(limits: JointLimits) -> Self {
        Self {
            limits,
            horizon_step_count: 12,
            step_interval: 0.05,
            weight_tracking: 20.0,
            weight_velocity: 0.5,
            weight_control: 0.05,
            weight_obstacle: 60.0,
            max_solver_iterations: 40,
            interval: IntervalBand::default(),
        }
    }

    /// Rejects settings no usable program could be built from.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when the horizon or the iteration budget
    /// is empty, and [`Error::OutOfRange`] when the horizon is longer than
    /// [`MAX_HORIZON_STEPS`], when a weight is negative or not a real
    /// number, or when the model step falls outside the interval band the
    /// same settings declare. A negative weight is refused rather than
    /// clamped because it makes the objective concave in that variable,
    /// which is a different problem than the one this module solves.
    pub fn validate(&self) -> Result<(), Error> {
        JointLimits::new(
            self.limits.max_velocity.clone(),
            self.limits.max_acceleration.clone(),
        )?;
        if self.horizon_step_count == 0 {
            return Err(Error::TooFew {
                quantity: "horizon steps",
                minimum: 1,
                actual: 0,
            });
        }
        if self.horizon_step_count > MAX_HORIZON_STEPS {
            return Err(Error::OutOfRange {
                quantity: "horizon steps",
                value: count_as_value(self.horizon_step_count),
                bound: "at or below the horizon ceiling",
            });
        }
        if self.max_solver_iterations == 0 {
            return Err(Error::TooFew {
                quantity: "solver iterations",
                minimum: 1,
                actual: 0,
            });
        }
        for (quantity, value) in [
            ("tracking weight", self.weight_tracking),
            ("velocity weight", self.weight_velocity),
            ("control weight", self.weight_control),
            ("obstacle weight", self.weight_obstacle),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "[0, inf)",
                });
            }
        }
        if !(self.step_interval.is_finite() && self.step_interval > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "model step interval",
                value: self.step_interval,
                bound: "(0, inf)",
            });
        }
        if self.step_interval < self.interval.minimum || self.step_interval > self.interval.maximum
        {
            return Err(Error::OutOfRange {
                quantity: "model step interval",
                value: self.step_interval,
                bound: "the configured interval band",
            });
        }
        Ok(())
    }

    /// How many axes these settings describe.
    #[must_use]
    pub fn axes(&self) -> usize {
        self.limits.axes()
    }
}

/// The trajectory a solved step predicts.
///
/// Held flat, one contiguous run of axis values per step, so that reading
/// a step back costs a slice rather than a chase through a vector of
/// vectors. The state series carry one more entry than the input series,
/// since the horizon ends in a state nothing is commanded from.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonPlan {
    axes: usize,
    horizon: usize,
    configurations: Vec<f64>,
    velocities: Vec<f64>,
    accelerations: Vec<f64>,
}

impl HorizonPlan {
    /// How many axes each entry carries.
    #[must_use]
    pub const fn axes(&self) -> usize {
        self.axes
    }

    /// How many steps the plan spans, meaning how many inputs it holds.
    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.horizon
    }

    /// The predicted configuration at `step`, for `step` up to and
    /// including [`HorizonPlan::step_count`].
    #[must_use]
    pub fn configuration(&self, step: usize) -> Option<&[f64]> {
        slice_at(&self.configurations, step, self.axes)
    }

    /// The predicted velocity at `step`, configuration units per second.
    #[must_use]
    pub fn velocity(&self, step: usize) -> Option<&[f64]> {
        slice_at(&self.velocities, step, self.axes)
    }

    /// The acceleration leaving `step`, units per second squared.
    ///
    /// `None` at the final state, which no input leaves.
    #[must_use]
    pub fn acceleration(&self, step: usize) -> Option<&[f64]> {
        slice_at(&self.accelerations, step, self.axes)
    }

    /// Reads a solution vector back into a plan.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the solver returned fewer
    /// variables than the layout declared, which would mean this module
    /// and the solver disagree about the problem rather than that the
    /// problem was hard.
    fn from_solution(variables: &[f64], layout: &Layout) -> Result<Self, Error> {
        if variables.len() < layout.variable_count {
            return Err(Error::DimensionMismatch {
                quantity: "solution variables",
                expected: layout.variable_count,
                actual: variables.len(),
            });
        }
        let states = layout.state_block;
        let inputs = layout.input_block;
        Ok(Self {
            axes: layout.axes,
            horizon: layout.horizon,
            configurations: block(variables, 0, states)?,
            velocities: block(variables, layout.velocity_block, states)?,
            accelerations: block(variables, layout.acceleration_block, inputs)?,
        })
    }
}

/// Where a step's command came from.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum StepOutcome {
    /// The program was solved and the command is its first input.
    Solved {
        /// The objective value at the solution.
        ///
        /// Including the constant term the solver drops, so that two
        /// solutions to the same step are directly comparable and a
        /// perfectly tracked target reads as zero.
        cost: f64,
        /// How many interior-point iterations the step took.
        iterations: u32,
        /// Whether the solver reached the tolerance it was asked for.
        ///
        /// A looser answer is still a command and still a worse one, which
        /// is why it is reported rather than folded into success.
        exact: bool,
    },
    /// No usable answer came back, and the command brakes instead.
    ///
    /// `FR-MPC-04`. The reason is carried rather than reduced to a flag,
    /// because a caller that cannot tell an empty feasible set from an
    /// exhausted budget cannot tell whether retrying is worth anything,
    /// which is what `FR-SAFE-02` asks of every solving entry point.
    SafeStop(SolveFailure),
}

impl StepOutcome {
    /// Whether the command came from a solved program.
    #[must_use]
    pub const fn solved(&self) -> bool {
        matches!(*self, Self::Solved { .. })
    }

    /// Why no program was solved, if none was.
    #[must_use]
    pub const fn failure(&self) -> Option<SolveFailure> {
        match *self {
            Self::Solved { .. } => None,
            Self::SafeStop(failure) => Some(failure),
        }
    }
}

/// What one control step did.
#[derive(Debug, Clone, PartialEq)]
pub struct JointMpcStep {
    /// The configuration after the step.
    pub configuration: Vec<f64>,
    /// The velocity after the step, configuration units per second.
    pub velocity: Vec<f64>,
    /// The acceleration that took it there, units per second squared.
    pub acceleration: Vec<f64>,
    /// Axes whose velocity limit bit on the way out.
    pub velocity_saturated: usize,
    /// Axes whose acceleration limit bit on the way out.
    pub acceleration_saturated: usize,
    /// Whether this came from the program or from the brake.
    pub outcome: StepOutcome,
    /// The predicted trajectory, absent when nothing was solved.
    pub plan: Option<HorizonPlan>,
}

/// A receding-horizon tracker over a configuration space.
///
/// Built around one convex program per step, assembled from the current
/// state and solved from scratch. Nothing is carried between steps except
/// the state itself: an interior-point method does not warm start from a
/// previous solution the way the Python's interior-point solver was asked
/// to, so the warm-start bookkeeping has no counterpart here.
#[derive(Debug, Clone)]
pub struct JointSpaceMpc<O> {
    settings: JointMpcSettings,
    occupancy: Option<O>,
    configuration: Vec<f64>,
    velocity: Vec<f64>,
}

impl<O: Occupancy> JointSpaceMpc<O> {
    /// Builds a controller resting at the origin of its configuration space.
    ///
    /// # Errors
    ///
    /// Returns whatever [`JointMpcSettings::validate`] returns, and
    /// [`Error::DimensionMismatch`] when the occupancy describes a space
    /// of a different dimension than the limits do.
    pub fn new(settings: JointMpcSettings, occupancy: Option<O>) -> Result<Self, Error> {
        settings.validate()?;
        let axes = settings.axes();
        if let Some(occupancy) = occupancy.as_ref()
            && occupancy.dimension() != axes
        {
            return Err(Error::DimensionMismatch {
                quantity: "occupancy dimension",
                expected: axes,
                actual: occupancy.dimension(),
            });
        }
        Ok(Self {
            settings,
            occupancy,
            configuration: vec![0.0; axes],
            velocity: vec![0.0; axes],
        })
    }

    /// The settings in force.
    #[must_use]
    pub const fn settings(&self) -> &JointMpcSettings {
        &self.settings
    }

    /// How many axes the controller moves.
    #[must_use]
    pub fn axes(&self) -> usize {
        self.settings.axes()
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

    /// Places the controller at `configuration` and stops it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the axis count is wrong
    /// and [`Error::NotFinite`] when a value is not a real number.
    pub fn reset(&mut self, configuration: &[f64]) -> Result<(), Error> {
        let axes = self.axes();
        self.reset_with_velocity(configuration, &vec![0.0; axes])
    }

    /// Places the controller at `configuration`, already moving.
    ///
    /// A velocity outside the limits is accepted, because it describes
    /// where the machine is rather than what was asked of it, and a
    /// controller that refused to be told the truth about its own state
    /// would have to be lied to instead. The first step from there cannot
    /// satisfy the velocity bound within one acceleration limit, so it
    /// reports an infeasible program and brakes, which is the honest
    /// answer to arriving too fast.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when either argument has the
    /// wrong axis count and [`Error::NotFinite`] when a value is not a
    /// real number.
    pub fn reset_with_velocity(
        &mut self,
        configuration: &[f64],
        velocity: &[f64],
    ) -> Result<(), Error> {
        self.require_axes("initial configuration", configuration)?;
        self.require_axes("initial velocity", velocity)?;
        self.configuration.clear();
        self.configuration.extend_from_slice(configuration);
        self.velocity.clear();
        self.velocity.extend_from_slice(velocity);
        Ok(())
    }

    /// Solves one step toward `target` and applies its first command.
    ///
    /// The elapsed interval arrives as an argument and is read against the
    /// configured band and against the model step, per `FR-INV-10` and
    /// deviation A-17. Nothing here reads a clock.
    ///
    /// A step that cannot be solved is not an error: it returns a braking
    /// command and says why, per `FR-MPC-04`. An error means the call
    /// itself was wrong.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] or [`Error::OutOfRange`] when `dt` is
    /// outside the band or disagrees with the model step,
    /// [`Error::DimensionMismatch`] when `target` has the wrong axis
    /// count, [`Error::NotFinite`] when it carries a value that is not a
    /// real number, and otherwise whatever the occupancy or the solver
    /// wrapper returns.
    pub fn step(&mut self, target: &[f64], dt: f64) -> Result<JointMpcStep, Error> {
        self.settings.interval.check(dt)?;
        self.require_model_interval(dt)?;
        self.require_axes("target configuration", target)?;

        let obstacles = self.probe_obstacles(target)?;
        let program = self.assemble(target, &obstacles)?;
        let Program {
            problem,
            layout,
            constant_cost,
        } = program;

        match problem.solve()? {
            Err(failure) => Ok(self.brake(failure, dt)),
            Ok(solution) => {
                let plan = HorizonPlan::from_solution(&solution.variables, &layout)?;
                if self.plan_holds_every_bound(&plan) {
                    Ok(self.apply(plan, &solution, constant_cost, dt))
                } else {
                    // The solver called it an answer and it breaks a bound
                    // the machine has to hold, which is a numerical
                    // failure however it is labelled upstream.
                    Ok(self.brake(SolveFailure::Numerical, dt))
                }
            }
        }
    }

    /// Applies the first command of a solved plan.
    ///
    /// Deviation A-09 in its joint-space form: the acceleration passes the
    /// per-axis rate limit and the velocity it produces passes the
    /// per-axis saturation, both drawn from the same [`JointLimits`] the
    /// program was built against. Neither can bite by more than the solver
    /// tolerance on a plan that already holds its bounds, which is the
    /// point of applying them anyway: the guard costs one clamp and
    /// removes the possibility of commanding a number nobody checked.
    fn apply(
        &mut self,
        plan: HorizonPlan,
        solution: &QpSolution,
        constant_cost: f64,
        dt: f64,
    ) -> JointMpcStep {
        let commanded = plan.acceleration(0).unwrap_or_default();
        let mut acceleration = vec![0.0; self.axes()];
        let mut velocity_saturated = 0_usize;
        let mut acceleration_saturated = 0_usize;

        for axis in 0..self.axes() {
            let velocity_limit = self.max_velocity(axis);
            let acceleration_limit = self.max_acceleration(axis);
            let requested = commanded.get(axis).copied().unwrap_or_default();
            let applied = requested.clamp(-acceleration_limit, acceleration_limit);
            if differs(applied, requested) {
                acceleration_saturated = acceleration_saturated.saturating_add(1);
            }

            let current = self.velocity.get(axis).copied().unwrap_or_default();
            let reached = applied.mul_add(dt, current);
            let held = reached.clamp(-velocity_limit, velocity_limit);
            if differs(held, reached) {
                velocity_saturated = velocity_saturated.saturating_add(1);
            }

            // The configuration advances on the velocity it arrived with,
            // which is the recurrence the program solved. Integrating on
            // the new velocity instead would put the machine somewhere the
            // plan does not pass through, and the plan is what the next
            // step's bounds were checked against.
            let position = self.configuration.get(axis).copied().unwrap_or_default();
            if let Some(slot) = self.configuration.get_mut(axis) {
                *slot = current.mul_add(dt, position);
            }
            if let Some(slot) = self.velocity.get_mut(axis) {
                *slot = held;
            }
            if let Some(slot) = acceleration.get_mut(axis) {
                *slot = applied;
            }
        }

        JointMpcStep {
            configuration: self.configuration.clone(),
            velocity: self.velocity.clone(),
            acceleration,
            velocity_saturated,
            acceleration_saturated,
            outcome: StepOutcome::Solved {
                cost: solution.cost + constant_cost,
                iterations: solution.iterations,
                exact: solution.exact,
            },
            plan: Some(plan),
        }
    }

    /// Brakes every axis at its acceleration limit.
    ///
    /// The Python counterpart applied the full limit whatever the velocity
    /// was, so an axis already nearly stopped was driven through zero and
    /// left reversing, and the next brake reversed it again. Here the
    /// change is the smaller of the allowance and what remains, so an axis
    /// stops at zero and stays there.
    ///
    /// The result is not clamped into the velocity box. The velocity being
    /// braked is the machine's state rather than a request, so there is
    /// nothing there for a saturation to refuse, and clamping it would
    /// report a deceleration no actuator could have delivered. What is
    /// limited is the acceleration, which is the command.
    fn brake(&mut self, failure: SolveFailure, dt: f64) -> JointMpcStep {
        let mut acceleration = vec![0.0; self.axes()];
        let mut acceleration_saturated = 0_usize;

        for axis in 0..self.axes() {
            let allowance = self.max_acceleration(axis) * dt;
            let current = self.velocity.get(axis).copied().unwrap_or_default();
            let shed = current.abs().min(allowance);
            if current.abs() > allowance {
                acceleration_saturated = acceleration_saturated.saturating_add(1);
            }
            let change = -shed.copysign(current);
            let reached = current + change;

            let position = self.configuration.get(axis).copied().unwrap_or_default();
            if let Some(slot) = self.configuration.get_mut(axis) {
                // On the braked path the new velocity is what advances the
                // configuration, which moves it less far than the old one
                // would. There is no plan to stay consistent with here,
                // so the more conservative of the two rules wins.
                *slot = reached.mul_add(dt, position);
            }
            if let Some(slot) = self.velocity.get_mut(axis) {
                *slot = reached;
            }
            if let Some(slot) = acceleration.get_mut(axis) {
                *slot = change / dt;
            }
        }

        JointMpcStep {
            configuration: self.configuration.clone(),
            velocity: self.velocity.clone(),
            acceleration,
            velocity_saturated: 0,
            acceleration_saturated,
            outcome: StepOutcome::SafeStop(failure),
            plan: None,
        }
    }

    /// Whether a returned plan is one the machine may be commanded from.
    ///
    /// `FR-MPC-03`, read off the answer rather than trusted of the solver.
    /// Every constraint the `CasADi` formulation carried is checked here:
    /// the initial state, both recurrences, the velocity bound on each
    /// predicted step, and the acceleration bound on each input.
    fn plan_holds_every_bound(&self, plan: &HorizonPlan) -> bool {
        let dt = self.settings.step_interval;
        for step in 0..=plan.step_count() {
            let (Some(configuration), Some(velocity)) =
                (plan.configuration(step), plan.velocity(step))
            else {
                return false;
            };
            if !finite(configuration) || !finite(velocity) {
                return false;
            }
            if step == 0 {
                if !agrees(configuration, &self.configuration) || !agrees(velocity, &self.velocity)
                {
                    return false;
                }
            } else {
                for (axis, &value) in velocity.iter().enumerate() {
                    if value.abs() > self.max_velocity(axis) + CONSTRAINT_TOLERANCE {
                        return false;
                    }
                }
            }

            let Some(acceleration) = plan.acceleration(step) else {
                continue;
            };
            if !finite(acceleration) {
                return false;
            }
            for (axis, &value) in acceleration.iter().enumerate() {
                if value.abs() > self.max_acceleration(axis) + CONSTRAINT_TOLERANCE {
                    return false;
                }
            }

            // The final state has no successor to check the recurrence
            // against, and an acceleration only exists for steps that have
            // one, so reaching here without a next state means the horizon
            // ended rather than the plan being wrong.
            let (Some(next_configuration), Some(next_velocity)) = (
                plan.configuration(step.saturating_add(1)),
                plan.velocity(step.saturating_add(1)),
            ) else {
                continue;
            };
            for axis in 0..plan.axes() {
                let position = configuration.get(axis).copied().unwrap_or_default();
                let speed = velocity.get(axis).copied().unwrap_or_default();
                let rate = acceleration.get(axis).copied().unwrap_or_default();
                let predicted_position = next_configuration.get(axis).copied().unwrap_or_default();
                let predicted_speed = next_velocity.get(axis).copied().unwrap_or_default();
                if (predicted_position - speed.mul_add(dt, position)).abs() > CONSTRAINT_TOLERANCE
                    || (predicted_speed - rate.mul_add(dt, speed)).abs() > CONSTRAINT_TOLERANCE
                {
                    return false;
                }
            }
        }
        true
    }

    /// Builds the program for one step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the problem would be larger than
    /// an index can address.
    fn assemble(&self, target: &[f64], obstacles: &ObstacleProbes) -> Result<Program, Error> {
        let layout = Layout::new(
            self.axes(),
            self.settings.horizon_step_count,
            obstacles.slot_count(),
        )?;
        let mut objective = Triplets::new();
        let mut gradient = vec![0.0; layout.variable_count];
        let constant_cost = self.push_objective(&layout, target, &mut objective, &mut gradient);

        let mut rows = Rows::new(layout.row_count);
        self.push_initial_state(&layout, &mut rows);
        self.push_dynamics(&layout, &mut rows);
        self.push_input_limits(&layout, &mut rows);
        self.push_barriers(&layout, target, obstacles, &mut rows);

        Ok(Program {
            problem: QpProblem {
                objective,
                gradient,
                constraints: rows.matrix,
                bounds: rows.bounds,
                equality_rows: layout.equality_rows,
                max_iterations: self.settings.max_solver_iterations,
            },
            layout,
            constant_cost,
        })
    }

    /// Writes the objective, returning the constant term it drops.
    ///
    /// Clarabel minimizes one half `x' P x + q' x`, so a term weighing
    /// `w (x - t)^2` puts `2 w` on the diagonal and `-2 w t` in the
    /// gradient, and leaves `w t^2` with nowhere to go. That constant is
    /// returned rather than discarded so the reported cost is the value of
    /// the objective as written.
    fn push_objective(
        &self,
        layout: &Layout,
        target: &[f64],
        objective: &mut Triplets,
        gradient: &mut [f64],
    ) -> f64 {
        let settings = &self.settings;
        let mut constant_cost = 0.0;

        for step in 0..=layout.horizon {
            // The Python weighed the final configuration twice as heavily
            // as the ones before it and left the final velocity out of the
            // objective entirely. Both are kept.
            let terminal = step == layout.horizon;
            let tracking = if terminal {
                2.0 * settings.weight_tracking
            } else {
                settings.weight_tracking
            };
            for axis in 0..layout.axes {
                let goal = target.get(axis).copied().unwrap_or_default();
                let index = layout.configuration(step, axis);
                objective.push(index, index, 2.0 * tracking);
                add_to(gradient, index, -2.0 * tracking * goal);
                constant_cost += tracking * goal * goal;

                if !terminal {
                    let index = layout.velocity(step, axis);
                    objective.push(index, index, 2.0 * settings.weight_velocity);
                    let index = layout.acceleration(step, axis);
                    objective.push(index, index, 2.0 * settings.weight_control);
                }
            }
        }

        for slot in 0..layout.obstacles {
            for step in 1..=layout.horizon {
                let index = layout.slack(slot, step);
                objective.push(index, index, 2.0 * settings.weight_obstacle);
                // The quadratic alone is not an exact penalty: its slope
                // at a penetration already paid for is small, so a large
                // tracking weight buys its way through the barrier for a
                // bounded price. The linear term keeps the marginal cost
                // of one more meter of penetration at the barrier weight
                // however deep the machine already is, which is what an
                // exact penalty means and what the quartic of the
                // nonlinear original achieved by curvature instead.
                add_to(gradient, index, settings.weight_obstacle);
            }
        }
        constant_cost
    }

    /// Pins the horizon to the state the machine is in.
    fn push_initial_state(&self, layout: &Layout, rows: &mut Rows) {
        for axis in 0..layout.axes {
            let row = rows.open(self.configuration.get(axis).copied().unwrap_or_default());
            rows.push(row, layout.configuration(0, axis), 1.0);
        }
        for axis in 0..layout.axes {
            let row = rows.open(self.velocity.get(axis).copied().unwrap_or_default());
            rows.push(row, layout.velocity(0, axis), 1.0);
        }
    }

    /// Writes the two integrator recurrences, one row per axis per step.
    fn push_dynamics(&self, layout: &Layout, rows: &mut Rows) {
        let dt = self.settings.step_interval;
        for step in 0..layout.horizon {
            let next = step.saturating_add(1);
            for axis in 0..layout.axes {
                let row = rows.open(0.0);
                rows.push(row, layout.configuration(next, axis), 1.0);
                rows.push(row, layout.configuration(step, axis), -1.0);
                rows.push(row, layout.velocity(step, axis), -dt);
            }
            for axis in 0..layout.axes {
                let row = rows.open(0.0);
                rows.push(row, layout.velocity(next, axis), 1.0);
                rows.push(row, layout.velocity(step, axis), -1.0);
                rows.push(row, layout.acceleration(step, axis), -dt);
            }
        }
    }

    /// Bounds every predicted velocity and every commanded acceleration.
    ///
    /// The velocity at step zero carries no bound, exactly as the Python
    /// left it: it is pinned to the state the machine arrived in, and a
    /// bound on a pinned variable either says nothing or empties the
    /// feasible set for a reason the caller cannot act on. Where arriving
    /// too fast does show up is on the next step, which is bounded and
    /// which no acceleration inside its own limit can bring back.
    fn push_input_limits(&self, layout: &Layout, rows: &mut Rows) {
        for step in 0..layout.horizon {
            let next = step.saturating_add(1);
            for axis in 0..layout.axes {
                let limit = self.max_velocity(axis);
                let index = layout.velocity(next, axis);
                // Both signs, written as two one-sided rows: `v <= limit`
                // and `-v <= limit`. Opening the rows without writing the
                // coefficient leaves two empty inequalities, which every
                // solver satisfies and which bound nothing, so the plan
                // comes back exceeding a limit the program never stated.
                let row = rows.open(limit);
                rows.push(row, index, 1.0);
                let row = rows.open(limit);
                rows.push(row, index, -1.0);
            }
            for axis in 0..layout.axes {
                let limit = self.max_acceleration(axis);
                let index = layout.acceleration(step, axis);
                let row = rows.open(limit);
                rows.push(row, index, 1.0);
                let row = rows.open(limit);
                rows.push(row, index, -1.0);
            }
        }
    }

    /// Writes one soft half-space per obstacle per predicted step.
    ///
    /// Each half-space passes through a nominal trajectory and its normal
    /// points from the obstacle toward that point, which is what makes the
    /// row linear and keeps the program convex. The nominal is the
    /// straight line from here to the target, walked no faster than the
    /// slowest axis allows and stopping there: the Python built its
    /// initial guess the same way, and the choice matters because a
    /// half-space anchored somewhere the machine will not pass either
    /// constrains nothing or constrains the wrong direction.
    ///
    /// A slack variable carries the penetration, so the barrier bends the
    /// answer without ever being able to empty the feasible set. The slack
    /// needs no lower bound of its own: its quadratic cost drives it to
    /// zero wherever the half-space is already satisfied.
    fn push_barriers(
        &self,
        layout: &Layout,
        target: &[f64],
        obstacles: &ObstacleProbes,
        rows: &mut Rows,
    ) {
        if obstacles.slot_count() == 0 {
            return;
        }
        let clearance = self.barrier_clearance();
        let remaining = distance_between(target, &self.configuration);
        let heading = self.travel_direction(target, remaining);

        for step in 1..=layout.horizon {
            let nominal = self.nominal_at(step, target, remaining);

            for slot in 0..obstacles.slot_count() {
                let Some(probe) = obstacles.point(slot, step) else {
                    // The occupancy could not place an obstacle for this
                    // slot at this step, so the row stands empty for the
                    // reason the next comment gives.
                    rows.open(0.0);
                    continue;
                };
                let obstacle = &probe.point;
                let separation = distance_between(&nominal, obstacle);
                // A row of nothing reads as `0 <= 0`, which every point
                // satisfies. Keeping it costs one empty row and keeps the
                // row count matching the layout, where skipping it would
                // shift every row after it.
                let row = rows.open(0.0);
                if !(separation.is_finite() && separation > SEPARATION_FLOOR) {
                    continue;
                }

                let normal = self.barrier_normal(&nominal, probe, separation, &heading);
                let mut offset = 0.0;
                for axis in 0..layout.axes {
                    let there = obstacle.get(axis).copied().unwrap_or_default();
                    let direction = normal.get(axis).copied().unwrap_or_default();
                    rows.push(
                        row,
                        layout.configuration(step, axis),
                        -direction / clearance,
                    );
                    offset += direction * there;
                }
                rows.push(row, layout.slack(slot, step), -1.0);
                rows.set_bound(row, -1.0 - offset / clearance);
            }
        }
    }

    /// The unit vector from here to the target, when there is one.
    ///
    /// Empty when the machine is already there, which is also when no
    /// barrier needs turning: a controller that is not going anywhere is
    /// not going to drive into anything.
    fn travel_direction(&self, target: &[f64], remaining: f64) -> Vec<f64> {
        if !(remaining.is_finite() && remaining > SEPARATION_FLOOR) {
            return Vec::new();
        }
        (0..self.axes())
            .map(|axis| {
                let from = self.configuration.get(axis).copied().unwrap_or_default();
                let to = target.get(axis).copied().unwrap_or_default();
                (to - from) / remaining
            })
            .collect()
    }

    /// Which way the half-space of one barrier faces.
    ///
    /// Normally the direction from the obstacle to the nominal, which is
    /// the supporting hyperplane of deviation A-30, tangent to the
    /// keep-out ball at the closest point of approach.
    ///
    /// Deviation A-33 covers the case that normal cannot express. An
    /// obstacle sitting on the route has a normal pointing back along it,
    /// so the only way to satisfy the row is to stop short: the program
    /// can brake and cannot steer, because a straight run at a symmetric
    /// obstacle carries no lateral gradient at all. Any unit normal gives
    /// a half-space that excludes a slab around the obstacle, and it is
    /// the choice of normal that decides whether going around is
    /// expressible. Turning it across the route is what asks the program
    /// to go around rather than to give up.
    fn barrier_normal(
        &self,
        nominal: &[f64],
        probe: &Probe,
        separation: f64,
        heading: &[f64],
    ) -> Vec<f64> {
        let obstacle = &probe.point;
        let radial: Vec<f64> = (0..self.axes())
            .map(|axis| {
                let here = nominal.get(axis).copied().unwrap_or_default();
                let there = obstacle.get(axis).copied().unwrap_or_default();
                (here - there) / separation
            })
            .collect();
        // Outside the clearance the row costs nothing whichever way it
        // faces, so the radial normal stands and the machine keeps its
        // line until an obstacle is close enough to matter.
        if heading.is_empty() || separation > clearance_of(self) {
            return radial;
        }
        let alignment: f64 = radial
            .iter()
            .zip(heading)
            .map(|(&component, &along)| component * along)
            .sum();
        if alignment.abs() < COLLINEAR_NORMAL {
            return radial;
        }
        // Tilted rather than replaced. A purely sideways normal forbids
        // the collinear path and nothing else, which leaves a machine
        // facing a wall no way to satisfy the row at all when the nearest
        // point slides along that wall beside it. Half of each keeps both
        // answers open: stop short, or step around, whichever the rest of
        // the program finds cheaper.
        let Some(across) = sideways(heading, &self.lateral_offset(obstacle, heading)) else {
            return radial;
        };
        let tilted: Vec<f64> = radial
            .iter()
            .zip(&across)
            .map(|(&back, &side)| back + side)
            .collect();
        let length = norm_of(&tilted);
        if length <= SEPARATION_FLOOR {
            return radial;
        }
        tilted.iter().map(|&value| value / length).collect()
    }

    /// Where the machine sits relative to the line through the obstacle.
    ///
    /// The component of the offset perpendicular to the route, which is
    /// the side the machine is already leaning toward and therefore the
    /// cheaper way around.
    fn lateral_offset(&self, obstacle: &[f64], heading: &[f64]) -> Vec<f64> {
        let offset: Vec<f64> = (0..self.axes())
            .map(|axis| {
                let here = self.configuration.get(axis).copied().unwrap_or_default();
                let there = obstacle.get(axis).copied().unwrap_or_default();
                here - there
            })
            .collect();
        let along: f64 = offset
            .iter()
            .zip(heading)
            .map(|(&component, &direction)| component * direction)
            .sum();
        offset
            .iter()
            .zip(heading)
            .map(|(&component, &direction)| along.mul_add(-direction, component))
            .collect()
    }

    /// Where on the straight run to the target step `step` would land.
    ///
    /// The barriers are written about this line rather than about a
    /// solved trajectory, because nothing is carried between steps: the
    /// machine walks as far as its slowest axis allows and stops at the
    /// target. The Python built its initial guess the same way, and the
    /// choice matters because a half-space anchored somewhere the machine
    /// will not pass either constrains nothing or constrains the wrong
    /// direction.
    fn nominal_at(&self, step: usize, target: &[f64], remaining: f64) -> Vec<f64> {
        if !(remaining.is_finite() && remaining > SEPARATION_FLOOR) {
            return self.configuration.clone();
        }
        let reach = self.settings.step_interval * self.slowest_axis();
        let travelled = (reach * count_as_value(step)).min(remaining);
        (0..self.axes())
            .map(|axis| {
                let from = self.configuration.get(axis).copied().unwrap_or_default();
                let to = target.get(axis).copied().unwrap_or_default();
                (travelled * (to - from) / remaining) + from
            })
            .collect()
    }

    /// The obstacle points this step's barriers are built against.
    ///
    /// Two families of them. The fixed probes are the Python's: where the
    /// machine is, and where it would be in a few steps at its current
    /// velocity. They anchor a barrier that does not move while the
    /// program is solved.
    ///
    /// The route probe is one query per predicted step, taken at the
    /// nominal position that step lands on. Deviation A-34 records why it
    /// is needed: against a flat face, the nearest point of a fixed probe
    /// slides sideways with the machine and sits between it and the
    /// target, so once the machine is past that point every step forward
    /// increases the distance the barrier measures and the barrier reads
    /// as satisfied while the machine is inside the obstacle. A point
    /// probed at the predicted position cannot be passed that way.
    ///
    /// A probe whose answer repeats one already held is dropped, and one
    /// the occupancy cannot locate is skipped rather than raised, since
    /// no barrier is the right answer to not knowing where the obstacle
    /// is.
    ///
    /// # Errors
    ///
    /// Propagates whatever the occupancy returns.
    fn probe_obstacles(&self, target: &[f64]) -> Result<ObstacleProbes, Error> {
        let Some(occupancy) = self.occupancy.as_ref() else {
            return Ok(ObstacleProbes::default());
        };
        if self.settings.weight_obstacle <= 0.0 {
            return Ok(ObstacleProbes::default());
        }

        let mut probes: Vec<Vec<f64>> = Vec::with_capacity(2);
        probes.push(self.configuration.clone());
        let speed = distance_between(&self.velocity, &vec![0.0; self.axes()]);
        if speed > PROBE_SPEED_FLOOR {
            let reach = self.settings.step_interval * OBSTACLE_PROBE_STEPS;
            probes.push(
                self.configuration
                    .iter()
                    .zip(&self.velocity)
                    .map(|(position, speed)| speed.mul_add(reach, *position))
                    .collect(),
            );
        }

        let mut fixed: Vec<Probe> = Vec::with_capacity(probes.len());
        for probe in &probes {
            let nearest = occupancy.nearest_obstacle(probe)?;
            if nearest.point.len() != self.axes() || !finite(&nearest.point) {
                continue;
            }
            if !fixed.iter().any(|held| agrees(&held.point, &nearest.point)) {
                fixed.push(Probe {
                    point: nearest.point,
                });
            }
        }

        let remaining = distance_between(target, &self.configuration);
        let mut route: Vec<Option<Probe>> = Vec::with_capacity(self.settings.horizon_step_count);
        for step in 1..=self.settings.horizon_step_count {
            let nominal = self.nominal_at(step, target, remaining);
            let nearest = occupancy.nearest_obstacle(&nominal)?;
            if nearest.point.len() != self.axes() || !finite(&nearest.point) {
                route.push(None);
                continue;
            }
            route.push(Some(Probe {
                point: nearest.point,
            }));
        }

        Ok(ObstacleProbes { fixed, route })
    }

    /// The clearance the barrier normalizes a penetration by, floored.
    ///
    /// The obstacle points come from [`Occupancy::nearest_obstacle`],
    /// which reports the distance to a surface per deviation A-16 while
    /// the point it returns is the obstacle's center. The barrier is
    /// written against that center and this radius, so nothing here has
    /// to convert between the two conventions.
    fn barrier_clearance(&self) -> f64 {
        self.occupancy
            .as_ref()
            .map_or(CLEARANCE_FLOOR, |occupancy| {
                occupancy.clearance().max(CLEARANCE_FLOOR)
            })
    }

    /// The velocity limit on one axis.
    fn max_velocity(&self, axis: usize) -> f64 {
        self.settings
            .limits
            .max_velocity
            .get(axis)
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// The lowest velocity limit across the axes, units per second.
    ///
    /// How fast the machine can be assumed to walk an arbitrary straight
    /// line through its configuration space, since a line that is not
    /// along one axis spends part of every axis's allowance.
    fn slowest_axis(&self) -> f64 {
        self.settings
            .limits
            .max_velocity
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// The acceleration limit on one axis.
    fn max_acceleration(&self, axis: usize) -> f64 {
        self.settings
            .limits
            .max_acceleration
            .get(axis)
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    /// Rejects an elapsed interval the model was not discretized at.
    ///
    /// The Python refused the same mismatch, and for the same reason: a
    /// model stepping by one interval inside a loop running at another
    /// predicts a trajectory the machine never follows, and the controller
    /// spends every step correcting the difference.
    fn require_model_interval(&self, dt: f64) -> Result<(), Error> {
        if !is_close(
            dt,
            self.settings.step_interval,
            arco_core::numeric::TIME_TOLERANCE,
            RELATIVE_TOLERANCE,
        ) {
            return Err(Error::OutOfRange {
                quantity: "elapsed interval",
                value: dt,
                bound: "the model step the settings declare",
            });
        }
        Ok(())
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

/// A program and what a reader needs to interpret its answer.
#[derive(Debug)]
struct Program {
    problem: QpProblem,
    layout: Layout,
    constant_cost: f64,
}

/// The constraint block under construction.
///
/// Rows are opened in order and filled afterward, which is what keeps the
/// equality rows ahead of the inequality rows without anyone counting:
/// [`QpProblem`] splits its cone list at a row index, so a row written out
/// of order would land in the wrong cone.
#[derive(Debug)]
struct Rows {
    matrix: Triplets,
    bounds: Vec<f64>,
    next: usize,
}

impl Rows {
    /// A block sized for `row_count` rows.
    fn new(row_count: usize) -> Self {
        Self {
            matrix: Triplets::new(),
            bounds: vec![0.0; row_count],
            next: 0,
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
    fn push(&mut self, row: usize, column: usize, value: f64) {
        self.matrix.push(row, column, value);
    }
}

/// Where each variable and each row of the program sits.
///
/// Computed once with checked arithmetic, so every index derived from it
/// afterward is known to fit and the saturating arithmetic the hardened
/// lint tier asks for cannot actually saturate.
#[derive(Debug, Clone, Copy)]
struct Layout {
    axes: usize,
    horizon: usize,
    obstacles: usize,
    state_block: usize,
    input_block: usize,
    velocity_block: usize,
    acceleration_block: usize,
    slack_block: usize,
    variable_count: usize,
    equality_rows: usize,
    row_count: usize,
}

impl Layout {
    /// Lays out a program of the given shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the program would be larger than
    /// an index can address.
    fn new(axes: usize, horizon: usize, obstacles: usize) -> Result<Self, Error> {
        let too_large = || Error::OutOfRange {
            quantity: "program size",
            value: count_as_value(horizon),
            bound: "addressable by an index",
        };
        let states = horizon
            .checked_add(1)
            .and_then(|count| count.checked_mul(axes))
            .ok_or_else(too_large)?;
        let inputs = horizon.checked_mul(axes).ok_or_else(too_large)?;
        let slacks = horizon.checked_mul(obstacles).ok_or_else(too_large)?;

        let velocity_block = states;
        let acceleration_block = states.checked_mul(2).ok_or_else(too_large)?;
        let slack_block = acceleration_block
            .checked_add(inputs)
            .ok_or_else(too_large)?;
        let variable_count = slack_block.checked_add(slacks).ok_or_else(too_large)?;

        // Two rows pinning the state, two recurrences per step and axis.
        let equality_rows = axes
            .checked_mul(2)
            .and_then(|pinned| {
                inputs
                    .checked_mul(2)
                    .and_then(|dynamics| pinned.checked_add(dynamics))
            })
            .ok_or_else(too_large)?;
        // Two-sided velocity and acceleration bounds, plus one half-space
        // per obstacle per predicted step.
        let row_count = inputs
            .checked_mul(4)
            .and_then(|bounds| bounds.checked_add(slacks))
            .and_then(|inequalities| inequalities.checked_add(equality_rows))
            .ok_or_else(too_large)?;

        Ok(Self {
            axes,
            horizon,
            obstacles,
            state_block: states,
            input_block: inputs,
            velocity_block,
            acceleration_block,
            slack_block,
            variable_count,
            equality_rows,
            row_count,
        })
    }

    /// The column holding a configuration value.
    const fn configuration(&self, step: usize, axis: usize) -> usize {
        step.saturating_mul(self.axes).saturating_add(axis)
    }

    /// The column holding a velocity value.
    const fn velocity(&self, step: usize, axis: usize) -> usize {
        self.velocity_block
            .saturating_add(step.saturating_mul(self.axes))
            .saturating_add(axis)
    }

    /// The column holding an acceleration value.
    const fn acceleration(&self, step: usize, axis: usize) -> usize {
        self.acceleration_block
            .saturating_add(step.saturating_mul(self.axes))
            .saturating_add(axis)
    }

    /// The column holding one obstacle's penetration at one step.
    ///
    /// Steps run from one, since the configuration at step zero is pinned
    /// and a penalty on it would be a constant.
    const fn slack(&self, slot: usize, step: usize) -> usize {
        self.slack_block
            .saturating_add(slot.saturating_mul(self.horizon))
            .saturating_add(step.saturating_sub(1))
    }
}

/// The clearance a controller's barrier normalizes by.
///
/// A free function so [`JointSpaceMpc::barrier_normal`] can ask for it
/// without borrowing the controller twice.
fn clearance_of<O: Occupancy>(controller: &JointSpaceMpc<O>) -> f64 {
    controller.barrier_clearance()
}

/// A unit vector across `heading`, leaning toward `preference`.
///
/// Returns nothing when no such direction can be built, which happens
/// when the heading is degenerate. The caller then keeps the radial
/// normal, so a barrier is never dropped for want of a direction.
fn sideways(heading: &[f64], preference: &[f64]) -> Option<Vec<f64>> {
    let leaning = norm_of(preference);
    if leaning > SEPARATION_FLOOR {
        return Some(preference.iter().map(|&value| value / leaning).collect());
    }
    // Perfectly head on, so either side is as good: take the axis the
    // route leans on least, and remove whatever of the route it carries.
    let (thinnest, _smallest) = heading.iter().enumerate().fold(
        (0_usize, f64::INFINITY),
        |(index, smallest), (axis, &component)| {
            if component.abs() < smallest {
                (axis, component.abs())
            } else {
                (index, smallest)
            }
        },
    );
    let along = heading.get(thinnest).copied().unwrap_or_default();
    let candidate: Vec<f64> = heading
        .iter()
        .enumerate()
        .map(|(axis, &component)| {
            let basis = if axis == thinnest { 1.0 } else { 0.0 };
            along.mul_add(-component, basis)
        })
        .collect();
    let length = norm_of(&candidate);
    (length > SEPARATION_FLOOR).then(|| candidate.iter().map(|&value| value / length).collect())
}

/// The Euclidean length of a vector.
fn norm_of(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|&value| value * value)
        .sum::<f64>()
        .sqrt()
}

/// The obstacle points one step's barriers are written against.
///
/// Two kinds, kept apart because they are indexed differently: a fixed
/// point holds for the whole horizon, while a route point belongs to one
/// predicted step. Both occupy one slack column per step, so a caller
/// reading [`Self::slot_count`] gets what the layout needs to allocate.
#[derive(Debug, Clone, Default, PartialEq)]
struct ObstacleProbes {
    /// Points probed once, at the machine and ahead of its velocity.
    fixed: Vec<Probe>,
    /// One point per predicted step, probed at that step's nominal.
    route: Vec<Option<Probe>>,
}

/// One answer from the occupancy, held per slot and per step.
#[derive(Debug, Clone, PartialEq)]
struct Probe {
    /// Where the occupancy put the nearest obstacle.
    point: Vec<f64>,
}

impl ObstacleProbes {
    /// How many barrier families the program carries.
    ///
    /// The route family counts once however many steps it covers, and
    /// counts not at all when no step found an obstacle.
    fn slot_count(&self) -> usize {
        let route = usize::from(self.route.iter().any(Option::is_some));
        self.fixed.len().saturating_add(route)
    }

    /// The probe slot `slot` is written against at `step`, if there is one.
    fn point(&self, slot: usize, step: usize) -> Option<&Probe> {
        self.fixed.get(slot).or_else(|| {
            (slot == self.fixed.len())
                .then(|| self.route.get(step.saturating_sub(1)))
                .flatten()
                .and_then(Option::as_ref)
        })
    }
}

/// A count, as the `f64` an [`Error`] carries.
fn count_as_value(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Adds `amount` at `index`, doing nothing when the index is past the end.
fn add_to(values: &mut [f64], index: usize, amount: f64) {
    if let Some(slot) = values.get_mut(index) {
        *slot += amount;
    }
}

/// Copies `length` values starting at `start`.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] when the run does not fit.
fn block(values: &[f64], start: usize, length: usize) -> Result<Vec<f64>, Error> {
    let end = start.checked_add(length);
    end.and_then(|end| values.get(start..end))
        .map(<[f64]>::to_vec)
        .ok_or(Error::DimensionMismatch {
            quantity: "solution block",
            expected: length,
            actual: values.len(),
        })
}

/// The run of `axes` values belonging to `step`.
fn slice_at(values: &[f64], step: usize, axes: usize) -> Option<&[f64]> {
    let start = step.checked_mul(axes)?;
    let end = start.checked_add(axes)?;
    values.get(start..end)
}

/// Whether every value is a real number.
fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Whether two points name the same place.
fn agrees(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&here, &there)| is_close(here, there, POSITION_TOLERANCE, RELATIVE_TOLERANCE))
}

/// The distance between two points of the same dimension.
fn distance_between(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&here, &there)| (here - there) * (here - there))
        .sum::<f64>()
        .sqrt()
}

/// Whether a conditioner changed a value by more than rounding.
///
/// The same test `limits::CommandConditioner` applies, and for the same
/// reason: a bare inequality reports a saturation on the step where the
/// request sits exactly on the limit and the clamp returns a value one
/// unit in the last place away.
fn differs(left: f64, right: f64) -> bool {
    (left - right).abs() > f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}
