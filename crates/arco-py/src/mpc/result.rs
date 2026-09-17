//! `MPCStepResult`, the diagnostics one predictive step reports.

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Result of a single :class:`MPCTracker` step.
///
/// Args:
///     `speed_cmd`: Commanded forward speed (m/s).
///     `turn_rate_cmd`: Commanded turn rate (rad/s).
///     `cross_track_error`: Signed lateral error to the reference (m).
///     `heading_error`: Heading error wrapped to ``(-pi, pi]`` (rad).
///     `progress`: Arc-length progress along the reference (m).
///     `predicted_clearance_min`: Minimum predicted obstacle distance over
///         the horizon (m). ``inf`` when no occupancy map is set.
///     `solver_success`: Whether the solver returned a usable solution.
///     `solver_status`: Solver status string. Deviation A-32: the
///         vocabulary is the convex solver's, not IPOPT's.
///     `solve_time_s`: Wall-clock solve time in seconds.
///     `cost`: Optimal, or fallback, cost value. Deviation A-31: this is
///         the surrogate convex objective.
///     `predicted_xy`: Predicted ``(x, y)`` samples over the horizon,
///         including the current pose. Empty on solver failure.
#[pyclass(subclass, name = "MPCStepResult", module = "arco._arco")]
#[derive(Debug, PartialEq)]
pub(crate) struct PyMpcStepResult {
    /// Commanded forward speed, meters per second.
    #[pyo3(get, set)]
    speed_cmd: f64,
    /// Commanded turn rate, radians per second.
    #[pyo3(get, set)]
    turn_rate_cmd: f64,
    /// Signed lateral error to the reference, meters.
    #[pyo3(get, set)]
    cross_track_error: f64,
    /// Heading error, radians, wrapped.
    #[pyo3(get, set)]
    heading_error: f64,
    /// Arc-length progress along the reference, meters.
    #[pyo3(get, set)]
    progress: f64,
    /// Smallest predicted obstacle distance over the horizon, meters.
    #[pyo3(get, set)]
    predicted_clearance_min: f64,
    /// Whether the solver returned a usable solution.
    #[pyo3(get, set)]
    solver_success: bool,
    /// What the solver reported, as text.
    #[pyo3(get, set)]
    solver_status: String,
    /// Wall-clock solve time, seconds.
    #[pyo3(get, set)]
    solve_time_s: f64,
    /// The objective value at the solution.
    #[pyo3(get, set)]
    cost: f64,
    /// Predicted positions over the horizon, meters.
    predicted_xy: Vec<(f64, f64)>,
}

impl PyMpcStepResult {
    /// Builds a result from what a controller's step produced.
    ///
    /// Held separate from the Python constructor so that a controller
    /// assembles one without going through argument parsing.
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python dataclass carries these eleven fields and the binding mirrors it"
    )]
    pub(crate) fn from_parts(
        speed_cmd: f64,
        turn_rate_cmd: f64,
        cross_track_error: f64,
        heading_error: f64,
        progress: f64,
        predicted_clearance_min: f64,
        solver_success: bool,
        solver_status: String,
        solve_time_s: f64,
        cost: f64,
        predicted_xy: Vec<(f64, f64)>,
    ) -> Self {
        Self {
            speed_cmd,
            turn_rate_cmd,
            cross_track_error,
            heading_error,
            progress,
            predicted_clearance_min,
            solver_success,
            solver_status,
            solve_time_s,
            cost,
            predicted_xy,
        }
    }
}

#[pymethods]
impl PyMpcStepResult {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(
        &self,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) {
    }

    /// Build a result, as the dataclass constructor did.
    #[new]
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python dataclass carries these eleven fields and the binding mirrors it"
    )]
    #[pyo3(signature = (
        speed_cmd,
        turn_rate_cmd,
        cross_track_error,
        heading_error,
        progress,
        predicted_clearance_min,
        solver_success,
        solver_status,
        solve_time_s,
        cost,
        predicted_xy = None,
    ))]
    #[pyo3(
        text_signature = "(speed_cmd, turn_rate_cmd, cross_track_error, heading_error, progress, predicted_clearance_min, solver_success, solver_status, solve_time_s, cost, predicted_xy=...)"
    )]
    fn new(
        speed_cmd: f64,
        turn_rate_cmd: f64,
        cross_track_error: f64,
        heading_error: f64,
        progress: f64,
        predicted_clearance_min: f64,
        solver_success: bool,
        solver_status: String,
        solve_time_s: f64,
        cost: f64,
        predicted_xy: Option<Vec<(f64, f64)>>,
    ) -> Self {
        Self::from_parts(
            speed_cmd,
            turn_rate_cmd,
            cross_track_error,
            heading_error,
            progress,
            predicted_clearance_min,
            solver_success,
            solver_status,
            solve_time_s,
            cost,
            predicted_xy.unwrap_or_default(),
        )
    }

    /// Predicted ``(x, y)`` samples over the horizon.
    ///
    /// A list of tuples, as the dataclass field held, so that a caller
    /// indexing it or passing it to ``list()`` sees what it saw before.
    #[getter]
    fn predicted_xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.predicted_xy.iter().copied())
    }

    /// Replaces the predicted samples.
    #[setter]
    fn set_predicted_xy(&mut self, samples: Vec<(f64, f64)>) {
        self.predicted_xy = samples;
    }

    /// Whether two results carry the same numbers.
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    /// The dataclass representation, field by field.
    fn __repr__(&self) -> String {
        let samples = self
            .predicted_xy
            .iter()
            .map(|&(x, y)| format!("({x}, {y})"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "MPCStepResult(speed_cmd={}, turn_rate_cmd={}, cross_track_error={}, \
             heading_error={}, progress={}, predicted_clearance_min={}, \
             solver_success={}, solver_status={:?}, solve_time_s={}, cost={}, \
             predicted_xy=[{samples}])",
            self.speed_cmd,
            self.turn_rate_cmd,
            self.cross_track_error,
            self.heading_error,
            self.progress,
            self.predicted_clearance_min,
            if self.solver_success { "True" } else { "False" },
            self.solver_status,
            self.solve_time_s,
            self.cost,
        )
    }
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMpcStepResult>()
}
