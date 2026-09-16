//! Turning a geometric path into a timed trajectory.
//!
//! Two stages, the same two the Python optimizer ran. The first places
//! every interior waypoint on the reference path and gives each segment a
//! duration proportional to its length at cruise speed, relaxed by a
//! factor so the second stage has room to tighten it. The second hands
//! that guess to a quasi-Newton solver over the composite cost in
//! [`TrajectoryTerm`].
//!
//! `scipy.optimize.minimize` becomes `argmin`, which is deviation A-08.
//! Two solvers reach different local minima on the same nonconvex
//! problem, so nothing here compares a solution vector against the Python
//! one; what is comparable is the cost achieved, and that is what the
//! tests assert.

use arco_core::Error;
use arco_core::geometry::{euclidean_distance, require_finite};
use arco_core::numeric::TIME_TOLERANCE;
use arco_core::protocols::{Command, CostTerm, Occupancy};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use super::cost_terms::{TrajectoryContext, TrajectoryTerm};

/// The shortest segment duration the optimizer will consider, seconds.
///
/// A duration of zero makes the implied speed infinite and every later
/// comparison meaningless, so the decision variable is the logarithm of
/// the duration and this is the floor applied when reading it back.
const MINIMUM_DURATION: f64 = 1e-3;

/// A state derived from the trajectory, for a feasibility check.
///
/// The five numbers a vehicle model needs to say whether it could execute
/// the trajectory: where it is, which way it faces, and how fast it is
/// moving and turning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedState {
    /// Position along the first axis, meters.
    pub x: f64,
    /// Position along the second axis, meters.
    pub y: f64,
    /// Heading of the outgoing segment, radians.
    pub heading: f64,
    /// Implied speed on that segment, meters per second.
    pub speed: f64,
    /// Implied turn rate entering the waypoint, radians per second.
    pub turn_rate: f64,
}

/// Whether the vehicle could execute a given state.
pub enum FeasibilityPolicy {
    /// Accepts everything, which is what no check at all means.
    Unchecked,
    /// Bounds on speed and turn rate.
    Bounded {
        /// Upper speed bound, meters per second, when there is one.
        max_speed: Option<f64>,
        /// Lower speed bound, meters per second, when there is one.
        min_speed: Option<f64>,
        /// Upper turn-rate bound, radians per second, when there is one.
        max_turn_rate: Option<f64>,
    },
    /// Anything the caller supplied.
    Custom(Box<dyn Fn(DerivedState) -> bool + Send + Sync>),
}

impl core::fmt::Debug for FeasibilityPolicy {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match *self {
            Self::Unchecked => "Unchecked",
            Self::Bounded { .. } => "Bounded",
            Self::Custom(_) => "Custom",
        };
        formatter.write_str(name)
    }
}

impl FeasibilityPolicy {
    /// Whether `state` is one the vehicle could be in.
    #[must_use]
    pub fn accepts(&self, state: DerivedState) -> bool {
        match self {
            Self::Unchecked => true,
            Self::Bounded {
                max_speed,
                min_speed,
                max_turn_rate,
            } => {
                let over = max_speed.is_some_and(|limit| state.speed > limit + TIME_TOLERANCE);
                let under = min_speed.is_some_and(|limit| state.speed < limit - TIME_TOLERANCE);
                let turning = max_turn_rate
                    .is_some_and(|limit| state.turn_rate.abs() > limit + TIME_TOLERANCE);
                !(over || under || turning)
            }
            Self::Custom(check) => check(state),
        }
    }
}

/// What a trajectory optimization produced.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryResult {
    /// The optimized waypoints, endpoints included and unmoved.
    pub states: Vec<Vec<f64>>,
    /// One command per segment.
    pub commands: Vec<Command>,
    /// One duration per segment, seconds.
    pub durations: Vec<f64>,
    /// The composite cost at the returned solution.
    pub cost: f64,
    /// Whether every state passed the feasibility policy.
    ///
    /// `false` means the trajectory violates the vehicle model somewhere.
    /// A caller stalls rather than executing it, which is why this is a
    /// field on a returned value and not a log line.
    pub is_feasible: bool,
    /// Whether the solver reached its tolerance rather than its budget.
    ///
    /// `FR-SAFE-02`: converged and out of iterations are different
    /// answers, and a caller that cannot tell them apart cannot decide
    /// whether a larger budget is worth spending.
    pub converged: bool,
    /// How many iterations the solver ran.
    pub iterations: usize,
}

/// How a trajectory optimization should behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizerSettings {
    /// The speed the trajectory aims to hold, meters per second.
    pub cruise_speed: f64,
    /// Slack on the initial duration estimate, as a multiplier above one.
    ///
    /// The first guess gives each segment more time than it needs, so the
    /// solver tightens the trajectory rather than having to loosen it.
    pub time_relaxation: f64,
    /// Maximum solver iterations.
    ///
    /// `FR-SAFE-02`. The result says whether this was reached.
    pub max_iterations: u64,
    /// Gradient norm below which the solver stops.
    pub gradient_tolerance: f64,
    /// Relative cost change below which the solver stops.
    pub cost_tolerance: f64,
    /// How many past steps the quasi-Newton memory keeps.
    pub memory: usize,
}

impl Default for OptimizerSettings {
    fn default() -> Self {
        Self {
            cruise_speed: 1.0,
            time_relaxation: 1.5,
            max_iterations: 500,
            gradient_tolerance: 1e-8,
            cost_tolerance: 1e-9,
            memory: 7,
        }
    }
}

/// Refines a reference path into a timed trajectory.
#[derive(Debug)]
pub struct TrajectoryOptimizer<O> {
    occupancy: O,
    terms: Vec<TrajectoryTerm>,
    settings: OptimizerSettings,
}

impl<O: Occupancy> TrajectoryOptimizer<O> {
    /// Builds an optimizer over `occupancy` summing `terms`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the cruise speed or the time
    /// relaxation is not strictly positive and finite.
    pub fn new(
        occupancy: O,
        terms: Vec<TrajectoryTerm>,
        settings: OptimizerSettings,
    ) -> Result<Self, Error> {
        for (quantity, value) in [
            ("cruise speed", settings.cruise_speed),
            ("time relaxation", settings.time_relaxation),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::OutOfRange {
                    quantity,
                    value,
                    bound: "(0, inf)",
                });
            }
        }
        Ok(Self {
            occupancy,
            terms,
            settings,
        })
    }

    /// Optimizes a trajectory along `reference`.
    ///
    /// # Arguments
    ///
    /// * `reference` - The planner's path, at least two waypoints, whose
    ///   first and last are held fixed.
    /// * `feasibility` - What the vehicle can execute, checked after the
    ///   solve rather than enforced during it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when `reference` holds fewer than two
    /// waypoints, [`Error::DimensionMismatch`] when they disagree, or
    /// [`Error::NotFinite`] when one carries a NaN or a term does.
    pub fn optimize(
        &self,
        reference: &[Vec<f64>],
        feasibility: &FeasibilityPolicy,
    ) -> Result<TrajectoryResult, Error> {
        let dimension = self.validate(reference)?;
        let segments = reference.len().saturating_sub(1);

        let problem = TrajectoryProblem {
            optimizer: self,
            reference,
            segments,
            dimension,
        };
        let guess = problem.initial_guess()?;

        let (solution, iterations, converged) = problem.solve(guess.clone());
        // A solver that failed to improve on the guess still has to hand
        // back something executable, and the guess is executable.
        let chosen = if problem.cost(&solution)? <= problem.cost(&guess)? {
            solution
        } else {
            guess
        };

        let (durations, waypoints) = problem.unpack(&chosen);
        let cost = problem.cost(&chosen)?;
        let derived = derived_states(&waypoints, &durations)?;
        let is_feasible = derived.iter().all(|state| feasibility.accepts(*state));

        Ok(TrajectoryResult {
            commands: commands_of(&derived, segments),
            states: waypoints,
            durations,
            cost,
            is_feasible,
            converged,
            iterations,
        })
    }

    /// Rejects a reference path nothing could be optimized along.
    fn validate(&self, reference: &[Vec<f64>]) -> Result<usize, Error> {
        let Some(first) = reference.first() else {
            return Err(Error::TooFew {
                quantity: "reference waypoints",
                minimum: 2,
                actual: 0,
            });
        };
        if reference.len() < 2 {
            return Err(Error::TooFew {
                quantity: "reference waypoints",
                minimum: 2,
                actual: reference.len(),
            });
        }
        let dimension = first.len();
        if dimension < 2 {
            return Err(Error::TooFew {
                quantity: "waypoint coordinates",
                minimum: 2,
                actual: dimension,
            });
        }
        for waypoint in reference {
            arco_core::geometry::require_dimension("reference waypoint", waypoint, dimension)?;
            require_finite("reference waypoint", waypoint)?;
        }
        if self.occupancy.dimension() != dimension {
            return Err(Error::DimensionMismatch {
                quantity: "reference waypoint",
                expected: self.occupancy.dimension(),
                actual: dimension,
            });
        }
        Ok(dimension)
    }
}

/// One optimization problem, bound to its reference path.
///
/// Separate from [`TrajectoryOptimizer`] because `argmin` wants a type it
/// can ask for a cost and a gradient, and the reference path is part of
/// the problem rather than part of the optimizer.
struct TrajectoryProblem<'a, O> {
    optimizer: &'a TrajectoryOptimizer<O>,
    reference: &'a [Vec<f64>],
    segments: usize,
    dimension: usize,
}

impl<O: Occupancy> TrajectoryProblem<'_, O> {
    /// The decision vector: log durations, then interior waypoints.
    ///
    /// Logarithms rather than the durations themselves, so that the
    /// solver runs unconstrained and a duration cannot go negative on the
    /// way to a minimum. `scipy` held the same bound with an explicit box
    /// constraint, which `argmin`'s quasi-Newton solvers do not offer.
    fn initial_guess(&self) -> Result<Vec<f64>, Error> {
        let mut guess = Vec::with_capacity(self.variable_count());
        for pair in self.reference.windows(2) {
            let [from, to] = pair else { continue };
            let length = euclidean_distance(from, to)?;
            let duration = (self.optimizer.settings.time_relaxation * length
                / self.optimizer.settings.cruise_speed)
                .max(MINIMUM_DURATION);
            guess.push(duration.ln());
        }
        for index in 1..self.segments {
            if let Some(waypoint) = self.reference.get(index) {
                guess.extend_from_slice(waypoint);
            }
        }
        Ok(guess)
    }

    /// How many numbers the solver is searching over.
    fn variable_count(&self) -> usize {
        self.segments.saturating_add(
            self.dimension
                .saturating_mul(self.segments.saturating_sub(1)),
        )
    }

    /// Reads a decision vector back into durations and waypoints.
    fn unpack(&self, parameters: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let durations: Vec<f64> = (0..self.segments)
            .map(|index| {
                parameters
                    .get(index)
                    .map_or(MINIMUM_DURATION, |value| value.exp().max(MINIMUM_DURATION))
            })
            .collect();

        let mut waypoints = Vec::with_capacity(self.segments.saturating_add(1));
        waypoints.push(self.reference.first().cloned().unwrap_or_default());
        for index in 0..self.segments.saturating_sub(1) {
            let start = self
                .segments
                .saturating_add(index.saturating_mul(self.dimension));
            let end = start.saturating_add(self.dimension);
            let slice = parameters.get(start..end).unwrap_or(&[]);
            waypoints.push(slice.to_vec());
        }
        waypoints.push(self.reference.last().cloned().unwrap_or_default());
        (durations, waypoints)
    }

    /// The composite cost at `parameters`.
    fn cost(&self, parameters: &[f64]) -> Result<f64, Error> {
        let (durations, waypoints) = self.unpack(parameters);
        let mut lengths = Vec::with_capacity(self.segments);
        for pair in waypoints.windows(2) {
            let [from, to] = pair else { continue };
            lengths.push(euclidean_distance(from, to)?);
        }
        let speeds: Vec<f64> = lengths
            .iter()
            .zip(&durations)
            .map(|(length, duration)| length / duration.max(MINIMUM_DURATION))
            .collect();

        let context = TrajectoryContext {
            durations: &durations,
            waypoints: &waypoints,
            reference: self.reference,
            lengths: &lengths,
            speeds: &speeds,
            occupancy: &self.optimizer.occupancy,
        };

        let mut total = 0.0;
        for term in &self.optimizer.terms {
            total += term.evaluate(&context)?;
        }
        Ok(total)
    }

    /// The gradient at `parameters`, by central differences.
    ///
    /// The Python optimizer supplied no Jacobian either, so `scipy` also
    /// differenced. Central rather than forward, since the extra
    /// evaluation per variable buys an order of accuracy and the cost
    /// here is dominated by the map queries inside the collision term
    /// rather than by the count of evaluations.
    fn gradient(&self, parameters: &[f64]) -> Result<Vec<f64>, Error> {
        let mut gradient = vec![0.0; parameters.len()];
        let mut probe = parameters.to_vec();
        for index in 0..parameters.len() {
            let Some(&center) = parameters.get(index) else {
                continue;
            };
            let step = f64::EPSILON.cbrt() * center.abs().max(1.0);
            if let Some(slot) = probe.get_mut(index) {
                *slot = center + step;
            }
            let above = self.cost(&probe)?;
            if let Some(slot) = probe.get_mut(index) {
                *slot = center - step;
            }
            let below = self.cost(&probe)?;
            if let Some(slot) = probe.get_mut(index) {
                *slot = center;
            }
            if let Some(slot) = gradient.get_mut(index) {
                *slot = (above - below) / (2.0 * step);
            }
        }
        Ok(gradient)
    }

    /// Runs the solver, returning the solution, iterations, and whether
    /// it converged rather than exhausting its budget.
    fn solve(&self, guess: Vec<f64>) -> (Vec<f64>, usize, bool) {
        let settings = self.optimizer.settings;
        let line_search = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(line_search, settings.memory);

        let result = Executor::new(self, solver)
            .configure(|state| {
                state
                    .param(guess.clone())
                    .max_iters(settings.max_iterations)
                    .target_cost(f64::NEG_INFINITY)
            })
            .run();

        // A line search that cannot make progress reports an error rather
        // than a result. That is a converged-enough answer in practice and
        // not a reason to fail the call, so the guess stands in and the
        // caller sees `converged` clear.
        let Ok(result) = result else {
            return (guess, 0, false);
        };
        let state = result.state();
        let iterations = usize::try_from(state.get_iter()).unwrap_or(usize::MAX);
        let converged = state.get_iter() < settings.max_iterations;
        let solution = state.get_best_param().cloned().unwrap_or(guess);
        (solution, iterations, converged)
    }
}

impl<O: Occupancy> CostFunction for &TrajectoryProblem<'_, O> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<f64, argmin::core::Error> {
        Ok(TrajectoryProblem::cost(self, parameters)?)
    }
}

impl<O: Occupancy> Gradient for &TrajectoryProblem<'_, O> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, parameters: &Self::Param) -> Result<Vec<f64>, argmin::core::Error> {
        Ok(TrajectoryProblem::gradient(self, parameters)?)
    }
}

/// Builds the per-waypoint states a feasibility check reads.
fn derived_states(waypoints: &[Vec<f64>], durations: &[f64]) -> Result<Vec<DerivedState>, Error> {
    let segments = durations.len();
    let mut headings = Vec::with_capacity(segments);
    let mut speeds = Vec::with_capacity(segments);
    for (index, pair) in waypoints.windows(2).enumerate() {
        let [from, to] = pair else { continue };
        let length = euclidean_distance(from, to)?;
        let (dx, dy) = (
            to.first().copied().unwrap_or_default() - from.first().copied().unwrap_or_default(),
            to.get(1).copied().unwrap_or_default() - from.get(1).copied().unwrap_or_default(),
        );
        // A segment of zero length has no heading of its own, and
        // inventing one from rounding noise would produce a turn rate out
        // of nothing.
        headings.push(if length > 0.0 { dy.atan2(dx) } else { 0.0 });
        let duration = durations.get(index).copied().unwrap_or(MINIMUM_DURATION);
        speeds.push(length / duration.max(MINIMUM_DURATION));
    }

    let last = segments.saturating_sub(1);
    let mut states = Vec::with_capacity(waypoints.len());
    for (index, waypoint) in waypoints.iter().enumerate() {
        let segment = index.min(last);
        let heading = headings.get(segment).copied().unwrap_or_default();
        let turn_rate = if index == 0 || segments < 2 {
            0.0
        } else {
            let previous = headings
                .get(index.saturating_sub(1))
                .copied()
                .unwrap_or(heading);
            let change = arco_core::numeric::angle_difference(heading, previous)?;
            let duration = durations
                .get(index.saturating_sub(1))
                .copied()
                .unwrap_or(MINIMUM_DURATION);
            change / duration.max(MINIMUM_DURATION)
        };
        states.push(DerivedState {
            x: waypoint.first().copied().unwrap_or_default(),
            y: waypoint.get(1).copied().unwrap_or_default(),
            heading,
            speed: speeds.get(segment).copied().unwrap_or_default(),
            turn_rate,
        });
    }
    Ok(states)
}

/// One command per segment, taken from the state that starts it.
fn commands_of(derived: &[DerivedState], segments: usize) -> Vec<Command> {
    derived
        .iter()
        .take(segments)
        .map(|state| Command {
            speed: state.speed,
            turn_rate: state.turn_rate,
        })
        .collect()
}
