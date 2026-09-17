//! `arco.guidance` as Python still sees it.
//!
//! Every class here keeps the argument names, the positional order, the
//! default values and the exception types the pure-Python implementation
//! had, per `FR-API-01` and `FR-API-02`. The five controller names
//! `arco.guidance` re-exports from `arco.control` are not repeated here:
//! they are one object reachable from two paths, and [`crate::control`]
//! is where that object is built.
//!
//! Three rules shape the code below. Nothing computes while holding the
//! interpreter, so every entry point past the argument reading runs under
//! `py.detach`. The two base classes a caller subclasses,
//! [`PyInterpolator`] and [`PyExplorationPrimitive`], carry `subclass` and
//! leave their one method raising, per deviation A-24. And the two
//! placeholders of deviation C-12, `BSplineInterpolator.interpolate` and
//! `DubinsPrimitive.steer`, come across as placeholders rather than
//! growing an implementation the Python callers never saw.
//!
//! Deviation A-20 is the one visible seam. `DubinsVehicle` holds a single
//! [`CommandLimits`] where Python carried five attributes, and two of them
//! were renamed on the way in. Both spellings are published, so
//! `vehicle.max_acceleration` and `vehicle.max_speed_rate` name the same
//! bound and a caller that wrote either keeps working.

use arco_control::limits::CommandLimits;
use arco_core::Error;
use arco_core::protocols::{Command, VehicleModel};
use arco_guidance::interpolation::{
    BSplineInterpolator, Interpolator as _, MovingAverageInterpolator,
};
use arco_guidance::primitive::{DubinsPrimitive, ExplorationPrimitive as _};
use arco_guidance::vehicle::DubinsVehicle;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::errors::{OrRaise, to_exception};
use crate::hooks::{as_array, coordinates};

// ---------------------------------------------------------------------
// Reading and writing the shapes Python passes around
// ---------------------------------------------------------------------

/// Reads a waypoint polyline into the pairs the interpolators take.
///
/// The fast path is one extraction of the whole list, which is what a
/// caller holding a list of tuples gets. Anything else is read waypoint
/// by waypoint through [`coordinates`], so a list of numpy rows or of
/// longer states is accepted the way `numpy.asarray` accepted it, and
/// only the first two components are read.
///
/// # Errors
///
/// Returns a `ValueError` when a waypoint carries fewer than two
/// components, and a `TypeError` when it is not a sequence of numbers.
fn waypoints(path: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if let Ok(pairs) = path.extract::<Vec<(f64, f64)>>() {
        return Ok(pairs);
    }
    let mut read = Vec::new();
    for waypoint in path.try_iter()? {
        let components = coordinates(&waypoint?)?;
        let (Some(&x), Some(&y)) = (components.first(), components.get(1)) else {
            return Err(to_exception(&Error::TooFew {
                quantity: "waypoint components",
                minimum: 2,
                actual: components.len(),
            }));
        };
        read.push((x, y));
    }
    Ok(read)
}

/// Writes a waypoint polyline back as the list of tuples Python expects.
///
/// # Errors
///
/// Returns whatever building the list raised.
fn as_waypoints<'py>(py: Python<'py>, path: &[(f64, f64)]) -> PyResult<Bound<'py, PyList>> {
    let points = path
        .iter()
        .map(|&(x, y)| PyTuple::new(py, [x, y]))
        .collect::<PyResult<Vec<Bound<'py, PyTuple>>>>()?;
    PyList::new(py, points)
}

/// Writes a segment back as the list of state tuples Python expects.
///
/// # Errors
///
/// Returns whatever building the list raised.
fn as_segment<'py>(py: Python<'py>, states: &[Vec<f64>]) -> PyResult<Bound<'py, PyList>> {
    let points = states
        .iter()
        .map(|state| PyTuple::new(py, state))
        .collect::<PyResult<Vec<Bound<'py, PyTuple>>>>()?;
    PyList::new(py, points)
}

// ---------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------

/// Abstract base for interpolation (e.g., B-splines, shortcutting).
///
/// Used to convert discrete node sequences to continuous trajectories.
/// Subclass it and implement :meth:`interpolate`; the base itself has no
/// trajectory to return.
#[pyclass(subclass, name = "Interpolator", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyInterpolator;

#[pymethods]
impl PyInterpolator {
    /// Build the base interpolator, which carries no state.
    #[new]
    #[pyo3(signature = ())]
    const fn new() -> Self {
        Self
    }

    /// Return a continuous trajectory from a discrete path.
    ///
    /// The base class defines no smoothing, so this raises. Use
    /// :class:`BSplineInterpolator` or
    /// :class:`MovingAverageInterpolator`, or override this method.
    ///
    /// Args:
    ///     `path`: A list of discrete waypoints.
    ///
    /// Returns:
    ///     A list of waypoints representing the interpolated trajectory.
    ///
    /// Raises:
    ///     `NotImplementedError`: On the base class.
    #[pyo3(signature = (path))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn interpolate(&self, path: &Bound<'_, PyAny>) -> PyResult<Py<PyList>> {
        let _ = path;
        Err(PyNotImplementedError::new_err(
            "Interpolator.interpolate is not implemented on the base class",
        ))
    }
}

/// B-spline interpolator for smoothing discrete paths.
///
/// **This interpolator returns the path it was given.** The Python class
/// stores the degree and returns its argument, with a comment saying a
/// real implementation would call ``scipy.interpolate``, and deviation
/// C-12 carries that across rather than changing every path that passes
/// through here. :class:`MovingAverageInterpolator` is the one that
/// actually smooths.
///
/// Args:
///     `degree`: Degree of the B-spline polynomial. At least 1.
///
/// Raises:
///     `ValueError`: If *degree* is zero, which describes a step function
///         rather than a curve through the waypoints.
#[pyclass(extends = PyInterpolator, subclass, name = "BSplineInterpolator", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyBSplineInterpolator {
    /// The interpolator this class is a face for.
    inner: BSplineInterpolator,
}

#[pymethods]
impl PyBSplineInterpolator {
    /// Build an interpolator of polynomial *degree*.
    #[new]
    #[pyo3(signature = (degree = 3))]
    fn new(degree: usize) -> PyResult<PyClassInitializer<Self>> {
        let inner = BSplineInterpolator::new(degree).or_raise()?;
        Ok(PyClassInitializer::from(PyInterpolator).add_subclass(Self { inner }))
    }

    /// Degree of the B-spline polynomial.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Replace the degree, rejecting one no curve could have.
    #[setter]
    fn set_degree(&mut self, degree: usize) -> PyResult<()> {
        self.inner = BSplineInterpolator::new(degree).or_raise()?;
        Ok(())
    }

    /// Smooth a discrete path using B-spline interpolation (stub).
    ///
    /// Returns the waypoints it was given, per the note on the class.
    ///
    /// Args:
    ///     `path`: A list of discrete waypoints.
    ///
    /// Returns:
    ///     A list of ``(x, y)`` tuples representing the trajectory.
    ///
    /// Raises:
    ///     `ValueError`: If a waypoint is not a finite pair of numbers.
    #[pyo3(signature = (path))]
    fn interpolate<'py>(
        &self,
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let read = waypoints(path)?;
        let smoothed = py.detach(|| self.inner.interpolate(&read)).or_raise()?;
        as_waypoints(py, &smoothed)
    }
}

/// Sliding-window moving-average smoothing of a waypoint polyline.
///
/// Filters the high-frequency lateral wiggle that sampling planners and
/// grid staircases leave in a waypoint list. Endpoints are preserved
/// exactly, interior points are replaced by the window mean, and repeated
/// iterations smooth harder. The filter only ever pulls a waypoint toward
/// the local chord, so a real corner is cut along with the wiggle: check
/// clearance after smoothing rather than assuming it survived.
///
/// Args:
///     `iterations`: Number of smoothing passes. At least 1.
///     `window`: Odd window size in waypoints. At least 3.
///
/// Raises:
///     `ValueError`: If *window* is even or smaller than 3, or if
///         *iterations* is zero.
#[pyclass(extends = PyInterpolator, subclass, name = "MovingAverageInterpolator", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyMovingAverageInterpolator {
    /// The smoother this class is a face for.
    inner: MovingAverageInterpolator,
}

#[pymethods]
impl PyMovingAverageInterpolator {
    /// Build a smoother running *iterations* passes of *window* points.
    ///
    /// Both arguments are keyword-only, as they were in Python.
    #[new]
    #[pyo3(signature = (*, iterations = 1, window = 3))]
    fn new(iterations: usize, window: usize) -> PyResult<PyClassInitializer<Self>> {
        let inner = MovingAverageInterpolator::new(iterations, window).or_raise()?;
        Ok(PyClassInitializer::from(PyInterpolator).add_subclass(Self { inner }))
    }

    /// Number of smoothing passes.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    /// Replace the pass count, rejecting zero.
    #[setter]
    fn set_iterations(&mut self, iterations: usize) -> PyResult<()> {
        self.inner = MovingAverageInterpolator::new(iterations, self.inner.window()).or_raise()?;
        Ok(())
    }

    /// Waypoints averaged per interior point.
    #[getter]
    fn window(&self) -> usize {
        self.inner.window()
    }

    /// Replace the window, rejecting an even one or one below three.
    #[setter]
    fn set_window(&mut self, window: usize) -> PyResult<()> {
        self.inner = MovingAverageInterpolator::new(self.inner.iterations(), window).or_raise()?;
        Ok(())
    }

    /// Smooth a discrete path with repeated moving-average passes.
    ///
    /// Args:
    ///     `path`: A list of discrete waypoints, ``(x, y)``-like.
    ///
    /// Returns:
    ///     A list of ``(x, y)`` tuples with identical first and last
    ///     points. Inputs shorter than 3 points come back unchanged.
    ///
    /// Raises:
    ///     `ValueError`: If a waypoint is not a finite pair of numbers.
    #[pyo3(signature = (path))]
    fn interpolate<'py>(
        &self,
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let read = waypoints(path)?;
        let smoothed = py.detach(|| self.inner.interpolate(&read)).or_raise()?;
        as_waypoints(py, &smoothed)
    }
}

// ---------------------------------------------------------------------
// Motion primitives
// ---------------------------------------------------------------------

/// Abstract base for exploration primitives (e.g., Dubins, Reeds-Shepp).
///
/// Used in RRT-based planners to ensure kinematic feasibility. Subclass it
/// and implement :meth:`steer`; the base itself generates no segment.
#[pyclass(subclass, name = "ExplorationPrimitive", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyExplorationPrimitive;

#[pymethods]
impl PyExplorationPrimitive {
    /// Build the base primitive, which carries no state.
    #[new]
    #[pyo3(signature = ())]
    const fn new() -> Self {
        Self
    }

    /// Return a feasible path segment from *from_state* to *to_state*.
    ///
    /// The base class generates no segment, so this raises. Use
    /// :class:`DubinsPrimitive`, or override this method.
    ///
    /// Args:
    ///     `from_state`: The starting state.
    ///     `to_state`: The target state.
    ///
    /// Returns:
    ///     A list of states representing the path segment.
    ///
    /// Raises:
    ///     `NotImplementedError`: On the base class.
    #[pyo3(signature = (from_state, to_state))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn steer(
        &self,
        from_state: &Bound<'_, PyAny>,
        to_state: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyList>> {
        let _ = (from_state, to_state);
        Err(PyNotImplementedError::new_err(
            "ExplorationPrimitive.steer is not implemented on the base class",
        ))
    }
}

/// Dubins path primitive for car-like robots.
///
/// Enforces no-reverse motion and a minimum turning radius. A robot moving
/// at speed *v* with turn rate *w* traces a circle of radius ``v / |w|``,
/// and the maneuver is executable only when that radius is at least
/// :attr:`turning_radius`.
///
/// **:meth:`steer` returns the two endpoints rather than a Dubins arc.**
/// The Python class is a placeholder that returns
/// ``[from_state, to_state]``, and deviation C-12 carries that across
/// unchanged. The turning-radius constraint is real and lives in
/// :meth:`is_feasible`.
///
/// Args:
///     `turning_radius`: Minimum turning radius for the robot (meters).
///         Must be finite and positive.
///
/// Raises:
///     `ValueError`: If *turning_radius* is not finite and positive.
#[pyclass(extends = PyExplorationPrimitive, subclass, name = "DubinsPrimitive", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyDubinsPrimitive {
    /// The primitive this class is a face for.
    inner: DubinsPrimitive,
}

#[pymethods]
impl PyDubinsPrimitive {
    /// Build a primitive with a minimum turning radius, in meters.
    #[new]
    #[pyo3(signature = (turning_radius = 1.0))]
    fn new(turning_radius: f64) -> PyResult<PyClassInitializer<Self>> {
        let inner = DubinsPrimitive::new(turning_radius).or_raise()?;
        Ok(PyClassInitializer::from(PyExplorationPrimitive).add_subclass(Self { inner }))
    }

    /// Minimum turning radius (meters).
    #[getter]
    fn turning_radius(&self) -> f64 {
        self.inner.turning_radius()
    }

    /// Replace the turning radius, rejecting one no arc could have.
    #[setter]
    fn set_turning_radius(&mut self, turning_radius: f64) -> PyResult<()> {
        self.inner = DubinsPrimitive::new(turning_radius).or_raise()?;
        Ok(())
    }

    /// Return a feasible Dubins path segment between two states.
    ///
    /// Returns the two endpoints, per the note on the class.
    ///
    /// Args:
    ///     `from_state`: The starting state, ``(x, y)`` or longer.
    ///     `to_state`: The target state, of the same length.
    ///
    /// Returns:
    ///     A list of state tuples representing the path segment.
    ///
    /// Raises:
    ///     `ValueError`: If a state is shorter than two components, carries
    ///         a value that is not a real number, or the two disagree in
    ///         length.
    #[pyo3(signature = (from_state, to_state))]
    fn steer<'py>(
        &self,
        py: Python<'py>,
        from_state: &Bound<'py, PyAny>,
        to_state: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let from = coordinates(from_state)?;
        let to = coordinates(to_state)?;
        let segment = py.detach(|| self.inner.steer(&from, &to)).or_raise()?;
        as_segment(py, &segment)
    }

    /// Check whether a state satisfies the minimum turning radius.
    ///
    /// A state has to carry both a speed and a turn rate to say anything
    /// about curvature, so ``(x, y)``, ``(x, y, theta)`` and
    /// ``(x, y, theta, v)`` are all accepted. A state at exactly the
    /// minimum radius is feasible, since the radius is a floor.
    ///
    /// Args:
    ///     `state`: Kinematic state, ``(x, y)`` or longer.
    ///
    /// Returns:
    ///     ``True`` if the state satisfies the turning-radius constraint.
    ///
    /// Raises:
    ///     `ValueError`: If the state is shorter than two components or
    ///         carries a value that is not a real number. A NaN compares
    ///         false against every bound, so an unchecked state would be
    ///         reported feasible, per deviation A-21.
    #[pyo3(signature = (state))]
    fn is_feasible(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<bool> {
        let read = coordinates(state)?;
        py.detach(|| self.inner.is_feasible(&read)).or_raise()
    }
}

// ---------------------------------------------------------------------
// The vehicle
// ---------------------------------------------------------------------

/// Dubins-like kinematic vehicle model with bounded dynamics.
///
/// A unicycle with Dubins-like constraints: no-reverse motion through a
/// configurable *min_speed*, a bounded turn rate, and first-order
/// filtering of both the speed and the turn rate. The state is
/// ``(x, y, heading)`` in the world frame and the controls are
/// ``(speed, turn_rate)``, saturated and rate-limited before a
/// forward-Euler integration.
///
/// The five bounds are held as one limit set, per deviation A-20, and two
/// of them answer to two names: ``max_acceleration`` and
/// ``max_speed_rate`` are the same bound, as are ``max_turn_rate_dot`` and
/// ``max_turn_rate_change``.
///
/// Args:
///     `x`: Initial x position in world frame (meters).
///     `y`: Initial y position in world frame (meters).
///     `heading`: Initial heading angle (radians).
///     `max_speed`: Maximum forward speed (m/s).
///     `min_speed`: Minimum forward speed; 0.0 prevents reversing (m/s).
///     `max_turn_rate`: Maximum absolute turn rate (rad/s).
///     `max_acceleration`: Maximum rate of speed change (m/s^2).
///     `max_turn_rate_dot`: Maximum rate of turn-rate change (rad/s^2).
///
/// Raises:
///     `ValueError`: If a pose component is not a real number, or the
///         limits describe a box no command could sit in.
#[pyclass(subclass, name = "DubinsVehicle", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyDubinsVehicle {
    /// The vehicle this class is a face for.
    pub(crate) inner: DubinsVehicle,
}

#[pymethods]
impl PyDubinsVehicle {
    /// Build a vehicle at rest at ``(x, y)`` facing *heading*.
    #[new]
    #[pyo3(signature = (
        x = 0.0,
        y = 0.0,
        heading = 0.0,
        max_speed = 5.0,
        min_speed = 0.0,
        max_turn_rate = 1.0,
        max_acceleration = 2.0,
        max_turn_rate_dot = 2.0,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python signature is the contract, per FR-API-02"
    )]
    fn new(
        x: f64,
        y: f64,
        heading: f64,
        max_speed: f64,
        min_speed: f64,
        max_turn_rate: f64,
        max_acceleration: f64,
        max_turn_rate_dot: f64,
    ) -> PyResult<Self> {
        let limits = CommandLimits {
            max_speed,
            min_speed,
            max_turn_rate,
            max_speed_rate: max_acceleration,
            max_turn_rate_change: max_turn_rate_dot,
            interval: DubinsVehicle::default_limits().interval,
        };
        Ok(Self {
            inner: DubinsVehicle::new(x, y, heading, limits).or_raise()?,
        })
    }

    /// Current x position in world frame (meters).
    #[getter]
    fn x(&self) -> f64 {
        self.inner.pose().x()
    }

    /// Current y position in world frame (meters).
    #[getter]
    fn y(&self) -> f64 {
        self.inner.pose().y()
    }

    /// Current heading angle (radians), wrapped into ``[-pi, pi)``.
    ///
    /// Python wrapped into ``(-pi, pi]``, so a heading of exactly ``pi``
    /// reads back as ``-pi``. The two name the same direction, per
    /// deviation A-22.
    #[getter]
    fn heading(&self) -> f64 {
        self.inner.pose().heading()
    }

    /// Current pose as ``(x, y, heading)``.
    #[getter]
    fn pose(&self) -> (f64, f64, f64) {
        let pose = self.inner.pose();
        (pose.x(), pose.y(), pose.heading())
    }

    /// Current forward speed (m/s).
    #[getter]
    fn speed(&self) -> f64 {
        VehicleModel::speed(&self.inner)
    }

    /// Places the vehicle at `value` metres per second.
    ///
    /// Writable because starting a run part way through a manoeuvre needs
    /// the speed as an input rather than an output, which is what callers
    /// were assigning `_speed` to do. Checked against the limits, which
    /// assigning an attribute never was.
    #[setter]
    fn set_speed(&mut self, value: f64) -> PyResult<()> {
        let turn_rate = VehicleModel::turn_rate(&self.inner);
        self.inner.set_motion(value, turn_rate).or_raise()
    }

    /// Current turn rate (rad/s).
    #[getter]
    fn turn_rate(&self) -> f64 {
        VehicleModel::turn_rate(&self.inner)
    }

    /// Places the vehicle at `value` radians per second.
    ///
    /// Writable for the reason [`DubinsVehicle::set_speed`] is.
    #[setter]
    fn set_turn_rate(&mut self, value: f64) -> PyResult<()> {
        let speed = VehicleModel::speed(&self.inner);
        self.inner.set_motion(speed, value).or_raise()
    }

    /// Maximum forward speed (m/s).
    #[getter]
    fn max_speed(&self) -> f64 {
        self.inner.limits().max_speed
    }

    /// Replace the maximum forward speed.
    #[setter]
    fn set_max_speed(&mut self, max_speed: f64) -> PyResult<()> {
        let mut limits = self.inner.limits();
        limits.max_speed = max_speed;
        self.inner.set_limits(limits).or_raise()
    }

    /// Minimum forward speed; 0.0 prevents reversing (m/s).
    #[getter]
    fn min_speed(&self) -> f64 {
        self.inner.limits().min_speed
    }

    /// Replace the minimum forward speed.
    #[setter]
    fn set_min_speed(&mut self, min_speed: f64) -> PyResult<()> {
        let mut limits = self.inner.limits();
        limits.min_speed = min_speed;
        self.inner.set_limits(limits).or_raise()
    }

    /// Maximum absolute turn rate (rad/s).
    #[getter]
    fn max_turn_rate(&self) -> f64 {
        self.inner.limits().max_turn_rate
    }

    /// Replace the maximum absolute turn rate.
    #[setter]
    fn set_max_turn_rate(&mut self, max_turn_rate: f64) -> PyResult<()> {
        let mut limits = self.inner.limits();
        limits.max_turn_rate = max_turn_rate;
        self.inner.set_limits(limits).or_raise()
    }

    /// Maximum rate of speed change (m/s^2).
    ///
    /// The same bound as :attr:`max_speed_rate`, under the name the
    /// Python class used. See deviation A-20.
    #[getter]
    fn max_acceleration(&self) -> f64 {
        self.inner.limits().max_speed_rate
    }

    /// Replace the maximum rate of speed change.
    #[setter]
    fn set_max_acceleration(&mut self, max_acceleration: f64) -> PyResult<()> {
        let mut limits = self.inner.limits();
        limits.max_speed_rate = max_acceleration;
        self.inner.set_limits(limits).or_raise()
    }

    /// Maximum rate of speed change (m/s^2), under its Rust name.
    #[getter]
    fn max_speed_rate(&self) -> f64 {
        self.inner.limits().max_speed_rate
    }

    /// Replace the maximum rate of speed change, under its Rust name.
    #[setter]
    fn set_max_speed_rate(&mut self, max_speed_rate: f64) -> PyResult<()> {
        self.set_max_acceleration(max_speed_rate)
    }

    /// Maximum rate of turn-rate change (rad/s^2).
    ///
    /// The same bound as :attr:`max_turn_rate_change`, under the name the
    /// Python class used. See deviation A-20.
    #[getter]
    fn max_turn_rate_dot(&self) -> f64 {
        self.inner.limits().max_turn_rate_change
    }

    /// Replace the maximum rate of turn-rate change.
    #[setter]
    fn set_max_turn_rate_dot(&mut self, max_turn_rate_dot: f64) -> PyResult<()> {
        let mut limits = self.inner.limits();
        limits.max_turn_rate_change = max_turn_rate_dot;
        self.inner.set_limits(limits).or_raise()
    }

    /// Maximum rate of turn-rate change (rad/s^2), under its Rust name.
    #[getter]
    fn max_turn_rate_change(&self) -> f64 {
        self.inner.limits().max_turn_rate_change
    }

    /// Replace the maximum rate of turn-rate change, under its Rust name.
    #[setter]
    fn set_max_turn_rate_change(&mut self, max_turn_rate_change: f64) -> PyResult<()> {
        self.set_max_turn_rate_dot(max_turn_rate_change)
    }

    /// Reset vehicle state to a new pose with zero speed and turn rate.
    ///
    /// Args:
    ///     `x`: New x position (meters).
    ///     `y`: New y position (meters).
    ///     `heading`: New heading angle (radians).
    ///
    /// Raises:
    ///     `ValueError`: If a component is not a real number.
    #[pyo3(signature = (x = 0.0, y = 0.0, heading = 0.0))]
    fn reset(&mut self, x: f64, y: f64, heading: f64) -> PyResult<()> {
        self.inner.reset(x, y, heading).or_raise()
    }

    /// Integrate kinematics one step with saturation and filtering.
    ///
    /// Rate-limits the speed and the turn rate, then integrates the
    /// unicycle equations with a forward-Euler step about the heading held
    /// at the start of the step.
    ///
    /// Args:
    ///     `speed_cmd`: Desired speed command (m/s).
    ///     `turn_rate_cmd`: Desired turn rate command (rad/s).
    ///     `dt`: Time step duration (s).
    ///
    /// Returns:
    ///     Updated pose as ``(x, y, heading)``.
    ///
    /// Raises:
    ///     `ValueError`: If a command is not a real number, or *dt* falls
    ///         outside the configured interval band, per deviation A-17.
    #[pyo3(signature = (speed_cmd, turn_rate_cmd, dt))]
    fn step(
        &mut self,
        py: Python<'_>,
        speed_cmd: f64,
        turn_rate_cmd: f64,
        dt: f64,
    ) -> PyResult<(f64, f64, f64)> {
        let command = Command {
            speed: speed_cmd,
            turn_rate: turn_rate_cmd,
        };
        let mut vehicle = self.inner;
        py.detach(|| vehicle.step(command, dt)).or_raise()?;
        self.inner = vehicle;
        Ok(self.pose())
    }

    /// Compute the command steering from *start* toward *goal*.
    ///
    /// A naive inversion of the unicycle: hold the requested speed and
    /// turn at whatever rate closes the heading gap within the time
    /// allowed, saturated to what the vehicle can do. It is an admissible
    /// first guess for the trajectory optimizer rather than a steering
    /// law, since it points at the goal once and never looks again.
    ///
    /// Args:
    ///     `start`: Starting position ``(x, y)`` or state ``(x, y, theta, ...)``.
    ///     `goal`: Target position ``(x, y)`` or state.
    ///     `speed`: Desired traversal speed (m/s).
    ///     `duration`: Time budget for the segment (s). Must be positive.
    ///
    /// Returns:
    ///     Command vector ``(speed_cmd, turn_rate_cmd)`` as a numpy array
    ///     of shape ``(2,)``.
    ///
    /// Raises:
    ///     `ValueError`: If a state is shorter than two components or
    ///         carries a value that is not a real number, or if *duration*
    ///         is not finite and strictly positive, per deviation A-21.
    #[pyo3(signature = (start, goal, speed, duration))]
    fn inverse_kinematics<'py>(
        &self,
        py: Python<'py>,
        start: &Bound<'py, PyAny>,
        goal: &Bound<'py, PyAny>,
        speed: f64,
        duration: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let from = coordinates(start)?;
        let to = coordinates(goal)?;
        let command = py
            .detach(|| self.inner.inverse_kinematics(&from, &to, speed, duration))
            .or_raise()?;
        Ok(as_array(py, &[command.speed, command.turn_rate]))
    }

    /// Check whether a state is within the vehicle's dynamic limits.
    ///
    /// A position and a heading say nothing about dynamics and are always
    /// acceptable; a fourth component is a speed and is checked against
    /// the speed band; a fifth is a turn rate and is checked against the
    /// turn-rate bound.
    ///
    /// Args:
    ///     `state`: Kinematic state ``(x, y, theta)`` or extended state
    ///         ``(x, y, theta, speed, turn_rate)``.
    ///
    /// Returns:
    ///     ``True`` if the state satisfies all dynamic constraints.
    ///
    /// Raises:
    ///     `ValueError`: If the state is shorter than two components or
    ///         carries a value that is not a real number, per deviation
    ///         A-21.
    #[pyo3(signature = (state))]
    fn is_feasible(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<bool> {
        let read = coordinates(state)?;
        py.detach(|| self.inner.is_feasible(&read)).or_raise()
    }
}

/// Registers the guidance classes on the compiled module.
///
/// # Errors
///
/// Returns an error when a registration fails, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyInterpolator>()?;
    module.add_class::<PyBSplineInterpolator>()?;
    module.add_class::<PyMovingAverageInterpolator>()?;
    module.add_class::<PyExplorationPrimitive>()?;
    module.add_class::<PyDubinsPrimitive>()?;
    module.add_class::<PyDubinsVehicle>()?;
    Ok(())
}
