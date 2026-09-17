//! `MPCTrackingLoop`, the closed loop around a predictive tracker.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::control::waypoints;

/// Local tracking loop driven by an :class:`MPCTracker`.
///
/// Mirrors :class:`~arco.control.tracking.TrackingLoop` metrics for
/// drop-in instrumentation while keeping obstacle avoidance inside the
/// optimizer, with no potential-field blend.
///
/// The loop holds the interpreter for the whole of a step rather than
/// releasing it. Its vehicle and its tracker are both Python objects, so
/// every line of the step is a call back into the interpreter and
/// releasing the lock would only buy the cost of reacquiring it twice.
///
/// Args:
///     `vehicle`: Kinematic vehicle model.
///     `tracker`: Configured MPC tracker. The reference may be set here
///         through :meth:`step` or on the tracker directly.
///     `cruise_speed`: Desired forward speed (m/s), stored for metrics
///         compatibility with :class:`TrackingLoop`.
#[pyclass(subclass, name = "MPCTrackingLoop", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyMpcTrackingLoop {
    /// The vehicle the loop drives.
    #[pyo3(get, set)]
    vehicle: Py<PyAny>,
    /// The tracker the loop asks for a command.
    #[pyo3(get, set)]
    tracker: Py<PyAny>,
    /// Nominal forward speed, meters per second.
    #[pyo3(get, set)]
    cruise_speed: f64,
    /// One entry per step taken.
    history: Vec<Py<PyDict>>,
    /// The waypoints the tracker was last given.
    reference: Option<Vec<(f64, f64)>>,
}

impl PyMpcTrackingLoop {
    /// Hands the tracker a reference only when the path changed.
    ///
    /// The Python compared the whole waypoint list on every step, which
    /// is what keeps a loop called with the same path from resetting the
    /// tracker's progress once per step.
    fn ensure_reference(&mut self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let points = waypoints(path)?;
        if self.reference.as_ref() == Some(&points) {
            return Ok(());
        }
        self.tracker
            .bind(py)
            .call_method1("set_reference", (path,))?;
        self.reference = Some(points);
        Ok(())
    }
}

#[pymethods]
impl PyMpcTrackingLoop {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a loop around a vehicle and a tracker.
    #[new]
    #[pyo3(signature = (vehicle, tracker, cruise_speed = 1.0))]
    #[pyo3(text_signature = "(vehicle, tracker, cruise_speed=1.0)")]
    fn new(vehicle: &Bound<'_, PyAny>, tracker: &Bound<'_, PyAny>, cruise_speed: f64) -> Self {
        Self {
            vehicle: vehicle.clone().unbind(),
            tracker: tracker.clone().unbind(),
            cruise_speed,
            history: Vec::new(),
            reference: None,
        }
    }

    /// Most recent step metrics, or ``None`` if no steps have been run.
    #[getter]
    fn metrics(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.history.last().map(|entry| entry.clone_ref(py))
    }

    /// Full per-step metrics history, as a copy of the list.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.history.iter().map(|entry| entry.clone_ref(py)))
    }

    /// Run one MPC tracking iteration.
    ///
    /// Args:
    ///     `path`: Reference path as ordered ``(x, y)`` waypoints.
    ///     `dt`: Integration time step (s).
    ///
    /// Returns:
    ///     Dictionary with the keys
    ///     :meth:`~arco.control.tracking.TrackingLoop.step` returns, plus
    ///     the ``mpc_*`` diagnostics of :class:`MPCStepResult`.
    ///
    /// Raises:
    ///     Whatever the vehicle or the tracker raised, unchanged.
    #[pyo3(signature = (path, dt = 0.1))]
    #[pyo3(text_signature = "(path, dt=0.1)")]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.ensure_reference(py, path)?;
        let vehicle = self.vehicle.bind(py);
        let arguments = PyDict::new(py);
        arguments.set_item("speed", vehicle.getattr("speed")?)?;
        arguments.set_item("turn_rate", vehicle.getattr("turn_rate")?)?;
        arguments.set_item("dt", dt)?;
        let result = self.tracker.bind(py).call_method(
            "step",
            (vehicle.getattr("pose")?,),
            Some(&arguments),
        )?;

        let speed_command = result.getattr("speed_cmd")?;
        let turn_rate_command = result.getattr("turn_rate_cmd")?;
        vehicle.call_method1(
            "step",
            (speed_command.clone(), turn_rate_command.clone(), dt),
        )?;

        let entry = PyDict::new(py);
        entry.set_item("cross_track_error", result.getattr("cross_track_error")?)?;
        entry.set_item("heading_error", result.getattr("heading_error")?)?;
        entry.set_item("pose", vehicle.getattr("pose")?)?;
        entry.set_item("speed", vehicle.getattr("speed")?)?;
        entry.set_item("turn_rate", vehicle.getattr("turn_rate")?)?;
        // Both stay zero for parity with the pure-pursuit loop, which
        // reports a steering curvature and a repulsion term this
        // controller has no counterpart for: avoidance lives inside the
        // optimizer here.
        entry.set_item("curvature", 0.0)?;
        entry.set_item("repulsion_turn_rate", 0.0)?;
        entry.set_item("mpc_progress", result.getattr("progress")?)?;
        entry.set_item(
            "mpc_predicted_clearance_min",
            result.getattr("predicted_clearance_min")?,
        )?;
        entry.set_item(
            "mpc_predicted_xy",
            py.get_type::<PyList>()
                .call1((result.getattr("predicted_xy")?,))?,
        )?;
        entry.set_item("mpc_solver_success", result.getattr("solver_success")?)?;
        entry.set_item("mpc_solver_status", result.getattr("solver_status")?)?;
        entry.set_item("mpc_solve_time_s", result.getattr("solve_time_s")?)?;
        entry.set_item("mpc_cost", result.getattr("cost")?)?;
        entry.set_item("mpc_speed_cmd", speed_command)?;
        entry.set_item("mpc_turn_rate_cmd", turn_rate_command)?;

        self.history.push(entry.clone().unbind());
        Ok(entry)
    }

    /// Run several tracking steps.
    ///
    /// Args:
    ///     `path`: Reference path as ordered ``(x, y)`` waypoints.
    ///     `steps`: Number of steps to simulate.
    ///     `dt`: Integration time step (s).
    ///
    /// Returns:
    ///     List of per-step metric dictionaries, same schema as
    ///     :meth:`step`.
    ///
    /// Raises:
    ///     Whatever the vehicle or the tracker raised, unchanged.
    #[pyo3(signature = (path, steps, dt = 0.1))]
    #[pyo3(text_signature = "(path, steps, dt=0.1)")]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        steps: usize,
        dt: f64,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut taken = Vec::with_capacity(steps);
        for _step in 0..steps {
            taken.push(self.step(py, path, dt)?);
        }
        PyList::new(py, taken)
    }
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMpcTrackingLoop>()
}
