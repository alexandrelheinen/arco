//! `JointSpaceMPC` and its configuration.

use arco_control::joint::JointLimits;
use arco_control::mpc::joint_space::{JointMpcSettings, JointSpaceMpc};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::control::{axis_limits, detached};
use arco_core::Error;

use crate::errors::{OrRaise, to_exception};
use crate::hooks::{BoundOccupancy, FailureSlot, SharedOccupancy, as_array, coordinates};
use crate::mpc::{attribute_count, attribute_number};

/// Builds the crate settings a Python configuration record describes.
///
/// `obstacle_barrier_power` is read past rather than read: deviation A-30
/// replaced the quartic penetration barrier with a supporting hyperplane
/// carrying a quadratic slack, so the key no longer shapes anything.
///
/// # Errors
///
/// Returns whatever reading an attribute raised.
fn settings_of(config: &Bound<'_, PyAny>, limits: JointLimits) -> PyResult<JointMpcSettings> {
    let mut settings = JointMpcSettings::new(limits);
    settings.horizon_step_count = attribute_count(config, "horizon_step_count")?;
    settings.step_interval = attribute_number(config, "dt")?;
    settings.weight_tracking = attribute_number(config, "weight_tracking")?;
    settings.weight_velocity = attribute_number(config, "weight_velocity")?;
    settings.weight_control = attribute_number(config, "weight_control")?;
    settings.weight_obstacle = attribute_number(config, "weight_obstacle")?;
    settings.max_solver_iterations =
        u32::try_from(attribute_count(config, "max_solver_iter_count")?).unwrap_or(u32::MAX);
    Ok(settings)
}

/// N-DOF receding-horizon tracker for C-space carrots.
///
/// API-compatible with :class:`~arco.control.joint_tracker.JointSpaceTracker`:
/// ``reset(q0)`` then ``step(target_q, dt) -> q``. Obstacle avoidance is
/// inside the optimizer, with no potential-field blend. The unused
/// *repulsion_gain* and *proportional_gain* arguments are accepted for
/// drop-in call-site parity.
///
/// ADR-002: the program is a convex one solved by Clarabel rather than the
/// nonlinear program IPOPT solved, so the commands differ from the ones
/// the Python produced. Deviation A-30 covers the obstacle barrier and
/// A-33 covers what braking does when no answer comes back.
///
/// Args:
///     `max_vel`: Per-axis velocity limit, a scalar or a one-dimensional
///         array. A single value applies to every axis.
///     `max_acc`: Per-axis acceleration limit, a scalar or an array.
///     `proportional_gain`: Accepted for API parity; unused.
///     `occupancy`: Optional C-space occupancy for soft barriers.
///     `repulsion_gain`: Accepted for API parity; unused, since the
///         barriers live inside the optimizer.
///     `config`: Optional weights and horizon. Defaults to
///         :class:`JointSpaceMPCConfig`.
///
/// Raises:
///     `ValueError`: If a velocity or acceleration limit is not strictly
///         positive, or the settings describe a program with no horizon.
#[pyclass(subclass, name = "JointSpaceMPC", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyJointSpaceMpc {
    /// The controller this class is a face for.
    inner: JointSpaceMpc<SharedOccupancy>,
    /// The limits the constructor resolved.
    limits: JointLimits,
    /// The configuration object the caller passed, kept for identity.
    config: Py<PyAny>,
    /// The settings that object described, kept so a reset can rebuild.
    weights: JointMpcSettings,
    /// The adopted map, kept so a wider reset can rebuild the controller.
    adopted: Option<SharedOccupancy>,
    /// The map the caller passed, kept so identity survives the trip.
    occupancy: Option<Py<PyAny>>,
    /// Where a query into a Python map parks the exception it raised.
    failure: FailureSlot,
    /// Whether the last step came from a solved program.
    last_solver_success: bool,
    /// How long the last solve took, seconds.
    last_solve_time_s: f64,
}

#[pymethods]
impl PyJointSpaceMpc {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a controller resting at the origin of its configuration space.
    #[new]
    #[pyo3(signature = (
        max_vel,
        max_acc,
        proportional_gain = 2.0,
        occupancy = None,
        repulsion_gain = 0.0,
        config = None,
    ))]
    #[pyo3(
        text_signature = "(max_vel, max_acc, proportional_gain=2.0, occupancy=None, repulsion_gain=0.0, config=None)"
    )]
    fn new(
        py: Python<'_>,
        max_vel: &Bound<'_, PyAny>,
        max_acc: &Bound<'_, PyAny>,
        proportional_gain: f64,
        occupancy: Option<&Bound<'_, PyAny>>,
        repulsion_gain: f64,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Both gains are read and dropped, as the Python did, so that a
        // call site written against the proportional tracker keeps
        // working against this one.
        let _ = (proportional_gain, repulsion_gain);
        let velocities = axis_limits("max_vel", max_vel)?;
        let accelerations = axis_limits("max_acc", max_acc)?;
        let axes = velocities.len().max(accelerations.len());
        let limits =
            JointLimits::new(widen(velocities, axes), widen(accelerations, axes)).or_raise()?;

        // A caller that passed no configuration gets the dataclass
        // defaults, built by importing the record rather than by
        // duplicating its numbers here.
        let held = match config {
            Some(given) => given.clone(),
            None => py
                .import("arco.control.mpc.joint_space")?
                .getattr("JointSpaceMPCConfig")?
                .call0()?,
        };
        let settings = settings_of(&held, limits.clone())?;

        let failure = FailureSlot::default();
        let adopted = occupancy.map(|map| {
            SharedOccupancy::new(
                BoundOccupancy::adopt(map, failure.clone()).assume_dimension(limits.axes()),
            )
        });
        Ok(Self {
            inner: JointSpaceMpc::new(settings.clone(), adopted.clone()).or_raise()?,
            limits,
            config: held.unbind(),
            weights: settings.clone(),
            adopted,
            occupancy: occupancy.map(|map| map.clone().unbind()),
            failure,
            last_solver_success: true,
            last_solve_time_s: 0.0,
        })
    }

    /// Current configuration.
    #[getter]
    fn q<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, self.inner.configuration())
    }

    /// Current velocity, configuration units per second.
    #[getter]
    fn vel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, self.inner.velocity())
    }

    /// The weights and horizon in force.
    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PyAny> {
        self.config.clone_ref(py)
    }

    /// Whether the last step came from a solved program.
    #[getter]
    const fn last_solver_success(&self) -> bool {
        self.last_solver_success
    }

    /// How long the last solve took, seconds.
    #[getter]
    const fn last_solve_time_s(&self) -> f64 {
        self.last_solve_time_s
    }

    /// The occupancy map the barriers query, or ``None``.
    #[getter]
    fn _occ(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.occupancy.as_ref().map(|map| map.clone_ref(py))
    }

    /// Per-axis velocity limit, as the constructor resolved it.
    #[getter]
    fn _max_vel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.limits.max_velocity)
    }

    /// Per-axis acceleration limit, as the constructor resolved it.
    #[getter]
    fn _max_acc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.limits.max_acceleration)
    }

    /// Reset tracker state to the initial configuration *q0*.
    ///
    /// Args:
    ///     `q0`: Initial configuration array.
    ///
    /// Raises:
    ///     `ValueError`: If the axis count is wrong, or a value is not a
    ///         real number.
    #[pyo3(signature = (q0))]
    #[pyo3(text_signature = "(q0)")]
    fn reset(&mut self, q0: &Bound<'_, PyAny>) -> PyResult<()> {
        let configuration = coordinates(q0)?;
        // A controller built from one number per limit takes its axis
        // count from the first configuration it is reset to, which is the
        // broadcasting the numpy arithmetic did in Python.
        if self.limits.axes() == 1 && configuration.len() != 1 {
            let (Some(&velocity), Some(&acceleration)) = (
                self.limits.max_velocity.first(),
                self.limits.max_acceleration.first(),
            ) else {
                return Err(to_exception(&Error::TooFew {
                    quantity: "axes",
                    minimum: 1,
                    actual: 0,
                }));
            };
            let widened =
                JointLimits::uniform(configuration.len(), velocity, acceleration).or_raise()?;
            let mut widened_settings = self.weights.clone();
            widened_settings.limits = widened.clone();
            self.inner =
                JointSpaceMpc::new(widened_settings.clone(), self.adopted.clone()).or_raise()?;
            self.weights = widened_settings;
            self.limits = widened;
        }
        self.inner.reset(&configuration).or_raise()
    }

    /// Run one step toward *target_q* and return the new configuration.
    ///
    /// Args:
    ///     `target_q`: Carrot configuration on the planned path.
    ///     `dt`: Integration time step (seconds). Must equal the
    ///         configured *dt*, since a model advancing by one number
    ///         while the loop advances by another predicts a trajectory
    ///         the machine never follows.
    ///
    /// Returns:
    ///     The configuration after applying the first optimal command. On
    ///     a failed solve the controller brakes instead, per `FR-MPC-04`,
    ///     and :attr:`last_solver_success` reports it.
    ///
    /// Raises:
    ///     `ValueError`: If *target_q* has the wrong axis count, a value
    ///         is not a real number, or *dt* disagrees with the
    ///         configured step.
    ///     Whatever the map raised, unchanged, when a query raises.
    #[pyo3(signature = (target_q, dt))]
    #[pyo3(text_signature = "(target_q, dt)")]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        target_q: &Bound<'py, PyAny>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let target = coordinates(target_q)?;
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
        let outcome = detached(py, &self.failure, || self.inner.step(&target, dt));
        self.last_solve_time_s = started.elapsed().as_secs_f64();
        let moved = outcome?;
        self.last_solver_success = moved.outcome.solved();
        Ok(as_array(py, &moved.configuration))
    }
}

/// Repeats a single limit across `axes`, as numpy broadcasting did.
///
/// A caller passing one number for a three-axis arm meant that number on
/// every axis, and the Python arithmetic gave it that without being asked.
fn widen(limits: Vec<f64>, axes: usize) -> Vec<f64> {
    match limits.as_slice() {
        [only] if axes > 1 => vec![*only; axes],
        _sized => limits,
    }
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyJointSpaceMpc>()
}
