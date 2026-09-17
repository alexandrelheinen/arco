//! `DubinsPathFollowingMPC`, its limits and its configuration.

use arco_control::limits::{CommandLimits, IntervalBand};
use arco_control::mpc::model::VehicleState;
use arco_control::mpc::path_following::{
    PathFollowingMpc, PathFollowingSettings, StepFailure, StepOutcome,
};
use arco_control::mpc::qp::SolveFailure;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::control::{detached, waypoints};
use crate::errors::OrRaise;
use crate::hooks::{BoundOccupancy, FailureSlot, SharedOccupancy};
use crate::mpc::result::PyMpcStepResult;
use crate::mpc::{attribute_count, attribute_number};

/// Builds the crate limit set a Python limits record describes.
///
/// Deviation A-20: `max_acceleration` is the crate's speed rate and
/// `max_turn_rate_dot` is its turn-rate change, so one type carries what
/// five loose attributes carried.
///
/// # Errors
///
/// Returns whatever reading an attribute raised.
fn limits_of(record: &Bound<'_, PyAny>) -> PyResult<CommandLimits> {
    Ok(CommandLimits {
        max_speed: attribute_number(record, "max_speed")?,
        min_speed: attribute_number(record, "min_speed")?,
        max_turn_rate: attribute_number(record, "max_turn_rate")?,
        max_speed_rate: attribute_number(record, "max_acceleration")?,
        max_turn_rate_change: attribute_number(record, "max_turn_rate_dot")?,
        interval: IntervalBand::default(),
    })
}

/// Builds the crate settings a Python configuration record describes.
///
/// The two trust radii, the sequential budget and its tolerance have no
/// Python counterpart, because the nonlinear solver had no need of them,
/// so they keep the crate defaults. `obstacle_barrier_power` is read past
/// rather than read, per deviation A-30.
///
/// # Errors
///
/// Returns whatever reading an attribute raised.
fn settings_of(
    config: &Bound<'_, PyAny>,
    limits: CommandLimits,
) -> PyResult<PathFollowingSettings> {
    let mut settings = PathFollowingSettings::new(limits);
    settings.horizon_step_count = attribute_count(config, "horizon_step_count")?;
    settings.step_interval = attribute_number(config, "dt")?;
    settings.cruise_speed = attribute_number(config, "cruise_speed")?;
    settings.weight_contour = attribute_number(config, "weight_contour")?;
    settings.weight_heading = attribute_number(config, "weight_heading")?;
    settings.weight_progress = attribute_number(config, "weight_progress")?;
    settings.weight_lag = attribute_number(config, "weight_lag")?;
    settings.weight_control = attribute_number(config, "weight_control")?;
    settings.weight_obstacle = attribute_number(config, "weight_obstacle")?;
    settings.weight_terminal = attribute_number(config, "weight_terminal")?;
    settings.contour_deadzone = attribute_number(config, "contour_deadzone")?;
    settings.max_solver_iterations =
        u32::try_from(attribute_count(config, "max_solver_iter_count")?).unwrap_or(u32::MAX);
    Ok(settings)
}

/// What the solver reported, as the status string a caller reads.
///
/// Deviation A-32. IPOPT's return strings are gone with IPOPT, so the
/// vocabulary is this one: `solved`, `solved_inexact`, `invalid_state`,
/// `infeasible`, `unbounded`, `budget_exhausted` and `numerical`. The
/// Python's `invalid_state` survives unchanged; its single `solve_failed`
/// splits into the four the solver can actually distinguish, which is
/// what tells a caller whether retrying is worth anything.
fn status_of(outcome: StepOutcome) -> &'static str {
    match outcome {
        StepOutcome::Solved { exact: true, .. } => "solved",
        StepOutcome::Solved { exact: false, .. } => "solved_inexact",
        StepOutcome::SafeStop(StepFailure::InvalidState) => "invalid_state",
        StepOutcome::SafeStop(StepFailure::Solve(failure)) => match failure {
            SolveFailure::Infeasible => "infeasible",
            SolveFailure::Unbounded => "unbounded",
            SolveFailure::BudgetExhausted => "budget_exhausted",
            SolveFailure::Numerical => "numerical",
            _unknown => "numerical",
        },
        _unknown => "numerical",
    }
}

/// The objective value at the solution, or zero when none was found.
const fn cost_of(outcome: StepOutcome) -> f64 {
    match outcome {
        StepOutcome::Solved { cost, .. } => cost,
        _unsolved => 0.0,
    }
}

/// Receding-horizon contouring controller for a Dubins vehicle.
///
/// ADR-002: the nonlinear program `CasADi` built and IPOPT solved becomes
/// a short sequence of convex programs Clarabel solves, so the commands
/// differ from the ones the Python produced. Deviation A-30 covers the
/// obstacle barrier, A-31 the reported cost, and A-32 the status strings.
///
/// Args:
///     `vehicle_limits`: What the vehicle can do, as
///         :class:`DubinsVehicleLimits`.
///     `config`: Weights and horizon, as :class:`PathFollowingMPCConfig`.
///     `occupancy`: Optional occupancy map for the soft barriers.
///
/// Raises:
///     `ValueError`: If the limits or the weights describe a program that
///         cannot be built, which includes a zero lag weight and a
///         horizon past its ceiling.
#[pyclass(subclass, name = "DubinsPathFollowingMPC", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyDubinsPathFollowingMpc {
    /// The controller this class is a face for.
    inner: PathFollowingMpc<SharedOccupancy>,
    /// The limits object the caller passed, kept for identity.
    vehicle_limits: Py<PyAny>,
    /// The configuration object the caller passed, kept for identity.
    config: Py<PyAny>,
    /// The map the caller passed, kept so identity survives the trip.
    occupancy: Option<Py<PyAny>>,
    /// Where a query into a Python map parks the exception it raised.
    failure: FailureSlot,
}

#[pymethods]
impl PyDubinsPathFollowingMpc {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a controller with no reference set yet.
    #[new]
    #[pyo3(signature = (*, vehicle_limits, config, occupancy = None))]
    #[pyo3(text_signature = "(*, vehicle_limits, config, occupancy=None)")]
    fn new(
        vehicle_limits: &Bound<'_, PyAny>,
        config: &Bound<'_, PyAny>,
        occupancy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let settings = settings_of(config, limits_of(vehicle_limits)?)?;
        let failure = FailureSlot::default();
        let adopted = occupancy.map(|map| {
            SharedOccupancy::new(BoundOccupancy::adopt(map, failure.clone()).assume_dimension(2))
        });
        Ok(Self {
            inner: PathFollowingMpc::new(settings, adopted).or_raise()?,
            vehicle_limits: vehicle_limits.clone().unbind(),
            config: config.clone().unbind(),
            occupancy: occupancy.map(|map| map.clone().unbind()),
            failure,
        })
    }

    /// What the vehicle can do.
    #[getter]
    fn vehicle_limits(&self, py: Python<'_>) -> Py<PyAny> {
        self.vehicle_limits.clone_ref(py)
    }

    /// The weights and horizon in force.
    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PyAny> {
        self.config.clone_ref(py)
    }

    /// The occupancy map the barriers query, or ``None``.
    #[getter]
    fn _occupancy(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.occupancy.as_ref().map(|map| map.clone_ref(py))
    }

    /// Arc length reached along the reference, meters.
    ///
    /// Published under the private name the Python controller carried,
    /// because a caller watching progress between steps reads it.
    #[getter]
    fn _progress(&self) -> f64 {
        self.inner.progress()
    }

    /// Set or replace the reference path.
    ///
    /// The polyline is extended by one horizon of straight runway along
    /// its final tangent, without which the bound holding the path
    /// parameter inside the path pinches against its own ceiling over the
    /// last meters. Setting a reference forgets the progress and the
    /// previous plan.
    ///
    /// Args:
    ///     `waypoints`: Ordered ``(x, y)`` waypoints in world frame.
    ///
    /// Raises:
    ///     `ValueError`: If fewer than two waypoints are given, or a
    ///         coordinate is not a real number.
    #[pyo3(signature = (waypoints))]
    #[pyo3(text_signature = "(waypoints)")]
    fn set_reference(&mut self, waypoints: &Bound<'_, PyAny>) -> PyResult<()> {
        let points = self::waypoints(waypoints)?;
        self.inner.set_reference(&points).or_raise()
    }

    /// Compute one receding-horizon control step.
    ///
    /// Args:
    ///     `pose`: Current vehicle pose ``(x, y, heading)``.
    ///     `speed`: Current forward speed (m/s).
    ///     `turn_rate`: Current turn rate (rad/s).
    ///     `dt`: Control period until the next call (s). It must agree
    ///         with the configured model step, since a model advancing by
    ///         one number while the loop advances by another predicts a
    ///         trajectory the machine never follows.
    ///
    /// Returns:
    ///     A :class:`MPCStepResult` carrying the command and the
    ///     diagnostics. A step that cannot be solved is not an error: it
    ///     brakes, reports ``solver_success`` as ``False``, and says why
    ///     in ``solver_status``.
    ///
    /// Raises:
    ///     `ValueError`: If no reference has been set, or *dt* falls
    ///         outside the interval band or disagrees with the model step.
    ///     Whatever the map raised, unchanged, when a query raises.
    #[pyo3(signature = (pose, *, speed, turn_rate, dt))]
    #[pyo3(text_signature = "(pose, *, speed, turn_rate, dt)")]
    fn step(
        &mut self,
        py: Python<'_>,
        pose: (f64, f64, f64),
        speed: f64,
        turn_rate: f64,
        dt: f64,
    ) -> PyResult<PyMpcStepResult> {
        let state = VehicleState {
            x: pose.0,
            y: pose.1,
            heading: pose.2,
            speed,
            turn_rate,
        };
        // FR-INV-10 keeps a clock out of a control decision, and this
        // is not one: `solve_time_s` is reported to the caller and read
        // by nothing. The Python published it from `time.perf_counter`,
        // and a caller watching for a solve that is drifting toward its
        // deadline needs it.
        #[expect(
            clippy::disallowed_methods,
            reason = "a reported diagnostic, never an input to the command"
        )]
        let started = std::time::Instant::now();
        let stepped = detached(py, &self.failure, || self.inner.step(state, dt))?;
        let solve_time = started.elapsed().as_secs_f64();

        let predicted = stepped.plan.as_ref().map_or_else(Vec::new, |plan| {
            plan.positions().collect::<Vec<(f64, f64)>>()
        });
        Ok(PyMpcStepResult::from_parts(
            stepped.command.speed,
            stepped.command.turn_rate,
            stepped.contour_error,
            stepped.heading_error,
            stepped.progress,
            stepped.predicted_clearance,
            stepped.outcome.solved(),
            status_of(stepped.outcome).to_owned(),
            solve_time,
            cost_of(stepped.outcome),
            predicted,
        ))
    }
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDubinsPathFollowingMpc>()
}
