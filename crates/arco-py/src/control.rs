//! `arco.control` as Python still sees it, apart from the MPC classes.
//!
//! Every class here keeps the argument names, the positional order, the
//! default values and the exception types the pure-Python implementation
//! had, per `FR-API-01` and `FR-API-02`. The MPC surface lives in its own
//! module: `DubinsPathFollowingMPC`, `JointSpaceMPC`, `MPCTracker`,
//! `MPCTrackingLoop`, `MPCStepResult`, `MPCController`, `ReferencePath`
//! and the two cost helpers are built there and are not repeated here.
//!
//! Four rules shape the code below.
//!
//! Nothing computes while holding the interpreter, so a method whose work
//! is native runs under `py.detach`. [`PyTrackingLoop`] is the one
//! exception and says why at its own step: its vehicle, its tracker and
//! its avoidance strategy are all Python objects, so releasing the lock
//! only to reacquire it three times costs more than it saves.
//!
//! Every class a caller subclasses carries `subclass` and an `__init__`
//! that absorbs whatever a Python subclass forwarded upward, because a
//! compiled class does its construction in `__new__` and
//! `object.__init__` refuses arguments. Deviation A-24 records what
//! subclassing costs.
//!
//! Deviation A-18 is visible on [`PyPidController`]. `control` keeps its
//! two Python arguments and passes an interval of exactly one second to
//! the Rust step, which is what reproduces the raw error sum and the raw
//! error difference the Python controller computed.
//!
//! Deviation A-09 puts a saturation function and a rate limiter on every
//! command leaving the control layer. Both default to infinity here, so a
//! caller that never asked for a limit is given what it requested and the
//! closed loop matches the Python one.

use arco_control::actuator::{ActuatorArray, ActuatorSettings, HazardPolicy};
use arco_control::avoidance::ArtificialPotentialField;
use arco_control::body::{BodyState, CircleBody, RigidBody, SquareBody};
use arco_control::joint::{JointLimits, JointSpaceTracker, JointTrackerSettings};
use arco_control::limits::CommandLimits;
use arco_control::pid::{PidController, PidGains, PidSettings};
use arco_control::pursuit::PurePursuitTracker;
use arco_control::tracking::{TrackingLoop, TrackingSample, TrackingSettings};
use arco_core::Error;
use arco_core::geometry::Pose;
use arco_core::protocols::{
    AvoidanceStrategy, Command, NearestObstacle, PathTracker, TrackerErrors, VehicleModel,
};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::errors::{OrRaise, to_exception};
use crate::hooks::{
    BoundOccupancy, FailureSlot, SharedOccupancy, as_array, coordinates, point_rows,
};

/// The interval the PID controller runs at, seconds.
///
/// Deviation A-18. `arco.control.pid.PIDController` summed raw errors and
/// differenced raw errors, which is the Rust controller at exactly one
/// second, so passing this keeps every existing gain meaning what it
/// meant.
const PID_INTERVAL: f64 = 1.0;

/// The mass a body reports when it publishes none.
///
/// Only [`read_body`] uses it, and only for a body handed to the actuator
/// array, which reads a pose and a bounding radius and never a mass.
const ASSUMED_MASS: f64 = 1.0;

// ---------------------------------------------------------------------
// Reading and writing the shapes Python passes around
// ---------------------------------------------------------------------

/// Runs `work` with the interpreter released, surfacing a parked exception.
///
/// An adapter reaching a Python object cannot carry a `PyErr` back
/// through a trait that returns [`Error`], so it parks the exception in
/// `slot` and reports a stand-in. Draining the slot here raises the
/// caller's own exception, with its type and traceback intact, on the
/// call that caused it.
pub(crate) fn detached<T, F>(py: Python<'_>, slot: &FailureSlot, work: F) -> PyResult<T>
where
    F: Ungil + Send + FnOnce() -> Result<T, Error>,
    T: Ungil + Send,
{
    match py.detach(work) {
        Err(failure) => Err(slot.take().unwrap_or_else(|| to_exception(&failure))),
        Ok(value) => slot.take().map_or(Ok(value), Err),
    }
}

/// Reads a waypoint polyline into the pairs the trackers take.
///
/// The fast path is one extraction of the whole list, which is what a
/// caller holding a list of tuples gets. Anything else is read row by
/// row, so a two-dimensional array and a list of longer states are both
/// accepted the way `numpy.asarray` accepted them, and only the first two
/// components are read.
///
/// # Errors
///
/// Returns a `ValueError` when a waypoint carries fewer than two
/// components, and a `TypeError` when it is not a sequence of numbers.
pub(crate) fn waypoints(path: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if let Ok(pairs) = path.extract::<Vec<(f64, f64)>>() {
        return Ok(pairs);
    }
    point_rows(path)?
        .iter()
        .map(|row| match (row.first(), row.get(1)) {
            (Some(&x), Some(&y)) => Ok((x, y)),
            _too_short => Err(to_exception(&Error::TooFew {
                quantity: "waypoint components",
                minimum: 2,
                actual: row.len(),
            })),
        })
        .collect()
}

/// Reads a pose triple into the type the control layer passes around.
///
/// # Errors
///
/// Returns a `ValueError` when the pose carries fewer than three
/// components or one of them is not a real number, and a `TypeError` when
/// it is not a sequence of numbers.
fn pose_of(pose: &Bound<'_, PyAny>) -> PyResult<Pose> {
    let read = coordinates(pose)?;
    let (Some(&x), Some(&y), Some(&heading)) = (read.first(), read.get(1), read.get(2)) else {
        return Err(to_exception(&Error::TooFew {
            quantity: "pose components",
            minimum: 3,
            actual: read.len(),
        }));
    };
    Pose::new(x, y, heading).or_raise()
}

/// Reads a wrench triple into the array the control layer passes around.
///
/// # Errors
///
/// Returns a `ValueError` when the wrench carries fewer than three
/// components, and a `TypeError` when it is not a sequence of numbers.
fn wrench_of(wrench: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
    let read = coordinates(wrench)?;
    let (Some(&fx), Some(&fy), Some(&torque)) = (read.first(), read.get(1), read.get(2)) else {
        return Err(to_exception(&Error::TooFew {
            quantity: "wrench components",
            minimum: 3,
            actual: read.len(),
        }));
    };
    Ok([fx, fy, torque])
}

/// Writes a list of planar points back as the `(N, 2)` array Python had.
///
/// # Errors
///
/// Returns whatever building the array raised.
fn as_points<'py>(py: Python<'py>, points: &[(f64, f64)]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = points.iter().map(|&(x, y)| vec![x, y]).collect();
    Ok(PyArray2::from_vec2(py, &rows)?)
}

// ---------------------------------------------------------------------
// The body every rigid-body entry point works on
// ---------------------------------------------------------------------

/// A planar body carrying its two shape constants alongside its state.
///
/// The crate's [`RigidBody`] trait asks an implementation for an inertia
/// and a bounding radius, and those are exactly what a Python subclass
/// supplies rather than the binding. Holding them as values lets one type
/// stand in for the compiled disk, the compiled square and anything a
/// caller subclassed, so the integrator below has a single code path.
#[derive(Clone, Copy, Debug)]
struct Shaped {
    /// Mass, pose, velocity and the accumulated wrench.
    state: BodyState,
    /// Rotational inertia about the center of mass, kilogram meters squared.
    inertia: f64,
    /// Radius of the circle enclosing the body, meters.
    bounding_radius: f64,
}

impl RigidBody for Shaped {
    fn state(&self) -> &BodyState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut BodyState {
        &mut self.state
    }

    fn inertia(&self) -> f64 {
        self.inertia
    }

    fn bounding_radius(&self) -> f64 {
        self.bounding_radius
    }
}

/// Reads whatever a caller passed as a body into [`Shaped`].
///
/// Reached by name rather than by type, so a compiled `CircleBody`, a
/// compiled `SquareBody` and a Python subclass of `RigidBody` are all
/// accepted, which is what the Python actuator array did by reading the
/// same three attributes off its argument.
///
/// The mass and the inertia are read leniently because the actuator array
/// is the only caller and it reads neither: its geometry comes from the
/// pose and the bounding radius. A body whose `inertia` raises, which the
/// abstract base does, is still usable here.
///
/// # Errors
///
/// Returns a `ValueError` when the pose or the bounding radius is missing
/// or is not a real number.
fn read_body(body: &Bound<'_, PyAny>) -> PyResult<Shaped> {
    let pose = coordinates(&body.getattr("pose")?)?;
    let (Some(&x), Some(&y), Some(&heading)) = (pose.first(), pose.get(1), pose.get(2)) else {
        return Err(to_exception(&Error::TooFew {
            quantity: "pose components",
            minimum: 3,
            actual: pose.len(),
        }));
    };
    let mass = body
        .getattr("mass")
        .and_then(|value| value.extract::<f64>())
        .unwrap_or(ASSUMED_MASS);
    let inertia = body
        .getattr("inertia")
        .and_then(|value| value.extract::<f64>())
        .unwrap_or(f64::NAN);
    let bounding_radius = body.getattr("bounding_radius")?.extract::<f64>()?;
    Ok(Shaped {
        state: BodyState::new(mass, x, y, heading).or_raise()?,
        inertia,
        bounding_radius,
    })
}

// ---------------------------------------------------------------------
// The base class every feedback controller subclasses
// ---------------------------------------------------------------------

/// Abstract base for feedback controllers.
///
/// Subclasses interpret ``state`` and ``reference`` according to their
/// own control law: a path position for pure pursuit, an error signal for
/// PID, a predicted trajectory for MPC. Subclass it and implement
/// :meth:`control`; the base itself computes nothing.
#[pyclass(subclass, name = "Controller", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyController;

#[pymethods]
impl PyController {
    /// Absorbs a subclass calling `super().__init__(...)`.
    ///
    /// A compiled class does its construction in `__new__`, so `__init__`
    /// falls through to `object.__init__`, which refuses arguments. A
    /// Python subclass forwarding its own arguments upward then fails on
    /// a line that worked against the pure-Python base. The arguments are
    /// ignored here because `__new__` has already read them.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Builds the base controller, which carries no state.
    ///
    /// Accepts and ignores whatever a subclass was constructed with, the
    /// way `object` does for any class overriding `__init__`.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    #[pyo3(text_signature = "()")]
    const fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    /// Compute the control command tracking *reference* from *state*.
    ///
    /// The base class defines no control law, so this raises. Use
    /// :class:`PIDController` or :class:`PurePursuitController`, or
    /// override this method.
    ///
    /// Args:
    ///     `state`: The current state value.
    ///     `reference`: The reference value to track.
    ///
    /// Returns:
    ///     The control command as a float.
    ///
    /// Raises:
    ///     `NotImplementedError`: On the base class.
    #[pyo3(signature = (state, reference))]
    #[pyo3(text_signature = "(state, reference)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn control(&self, state: f64, reference: f64) -> PyResult<f64> {
        let _ = (state, reference);
        Err(PyNotImplementedError::new_err(
            "Controller.control is abstract.",
        ))
    }
}

// ---------------------------------------------------------------------
// PID
// ---------------------------------------------------------------------

/// PID controller for path tracking.
///
/// Proportional, integral and derivative feedback on the error between a
/// reference and a measurement. The output carries no limit unless one is
/// asked for, so the anti-windup path deviation A-09 adds is exactly zero
/// here and the arithmetic matches the Python controller.
///
/// One behavior changes, per deviation A-18: the first call after
/// construction takes no derivative. Python differenced against an
/// initial previous error of zero, which made the derivative term a spike
/// proportional to the initial error on the first call of every run.
///
/// Args:
///     `kp`: Proportional gain.
///     `ki`: Integral gain.
///     `kd`: Derivative gain.
///
/// Raises:
///     `ValueError`: If a gain is not a real number.
#[pyclass(extends = PyController, subclass, name = "PIDController", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyPidController {
    /// The controller this class is a face for.
    inner: PidController,
}

#[pymethods]
impl PyPidController {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a controller with the three gains.
    #[new]
    #[pyo3(signature = (kp = 1.0, ki = 0.0, kd = 0.1))]
    #[pyo3(text_signature = "(kp=1.0, ki=0.0, kd=0.1)")]
    fn new(kp: f64, ki: f64, kd: f64) -> PyResult<PyClassInitializer<Self>> {
        let settings = PidSettings {
            gains: PidGains {
                proportional: kp,
                integral: ki,
                derivative: kd,
            },
            ..PidSettings::default()
        };
        let inner = PidController::new(settings).or_raise()?;
        Ok(PyClassInitializer::from(PyController).add_subclass(Self { inner }))
    }

    /// Proportional gain.
    #[getter]
    fn kp(&self) -> f64 {
        self.inner.settings().gains.proportional
    }

    /// Integral gain.
    #[getter]
    fn ki(&self) -> f64 {
        self.inner.settings().gains.integral
    }

    /// Derivative gain.
    #[getter]
    fn kd(&self) -> f64 {
        self.inner.settings().gains.derivative
    }

    /// Accumulated integral term.
    #[getter]
    fn integral(&self) -> f64 {
        self.inner.integral()
    }

    /// Clear the integrator and the previous error.
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "()")]
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Compute the PID output driving *state* toward *reference*.
    ///
    /// Args:
    ///     `state`: The current state value.
    ///     `reference`: The reference value to track.
    ///
    /// Returns:
    ///     The control command as a float.
    ///
    /// Raises:
    ///     `ValueError`: If either argument is not a real number.
    #[pyo3(signature = (state, reference))]
    #[pyo3(text_signature = "(state, reference)")]
    fn control(&mut self, py: Python<'_>, state: f64, reference: f64) -> PyResult<f64> {
        py.detach(|| self.inner.step(state, reference, PID_INTERVAL))
            .or_raise()
    }
}

// ---------------------------------------------------------------------
// Pure pursuit
// ---------------------------------------------------------------------

/// Pure pursuit controller for two-dimensional path tracking.
///
/// Computes a turn rate that steers a unicycle toward a lookahead point a
/// fixed arc length ahead on the reference path, through the standard law
/// ``omega = 2 v sin(alpha) / L``, where *alpha* is the bearing from the
/// vehicle heading to that point. Cross-track error, heading error and
/// curvature are updated on every :meth:`track` call and published for
/// logging.
///
/// Args:
///     `lookahead_distance`: Arc length ahead on the path used to locate
///         the lookahead point (meters). Must be finite and positive.
///
/// Raises:
///     `ValueError`: If *lookahead_distance* is not finite and positive,
///         which the turn-rate law divides by.
#[pyclass(extends = PyController, subclass, name = "PurePursuitController", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyPurePursuitController {
    /// The tracker this class is a face for.
    inner: PurePursuitTracker,
}

#[pymethods]
impl PyPurePursuitController {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a tracker looking a fixed arc length ahead, in meters.
    #[new]
    #[pyo3(signature = (lookahead_distance = 1.0))]
    #[pyo3(text_signature = "(lookahead_distance=1.0)")]
    fn new(lookahead_distance: f64) -> PyResult<PyClassInitializer<Self>> {
        let inner = PurePursuitTracker::new(lookahead_distance).or_raise()?;
        Ok(PyClassInitializer::from(PyController).add_subclass(Self { inner }))
    }

    /// Arc length used to locate the lookahead point (meters).
    #[getter]
    fn lookahead_distance(&self) -> f64 {
        self.inner.lookahead_distance()
    }

    /// Signed distance from the path, positive to its left (meters).
    #[getter]
    fn cross_track_error(&self) -> f64 {
        self.inner.cross_track_error()
    }

    /// Heading minus the path tangent at the closest waypoint (radians).
    #[getter]
    fn heading_error(&self) -> f64 {
        self.inner.heading_error()
    }

    /// Signed curvature the last command implied (per meter).
    #[getter]
    fn curvature(&self) -> f64 {
        self.inner.curvature()
    }

    /// Compute the pure pursuit speed and turn-rate commands.
    ///
    /// Args:
    ///     `pose`: Current vehicle pose ``(x, y, heading)`` in world frame.
    ///     `path`: Ordered sequence of ``(x, y)`` waypoints.
    ///     `speed`: Desired forward speed (m/s), passed through unchanged.
    ///
    /// Returns:
    ///     ``(speed_cmd, turn_rate_cmd)``, the first equal to *speed* and
    ///     the second in rad/s.
    ///
    /// Raises:
    ///     `ValueError`: If the pose or a waypoint is not a finite pair of
    ///         numbers, or the speed is not a real number.
    #[pyo3(signature = (pose, path, speed = 1.0))]
    #[pyo3(text_signature = "(pose, path, speed=1.0)")]
    fn track(
        &mut self,
        py: Python<'_>,
        pose: &Bound<'_, PyAny>,
        path: &Bound<'_, PyAny>,
        speed: f64,
    ) -> PyResult<(f64, f64)> {
        let read = waypoints(path)?;
        // A path of one waypoint has no tangent and no segment to meet,
        // and Python answered it by holding the speed and steering
        // straight rather than raising. The crate refuses it instead, so
        // the early answer stays here where the Python contract is.
        if read.len() < 2 {
            return Ok((speed, 0.0));
        }
        let at = pose_of(pose)?;
        let command = py
            .detach(|| self.inner.track(at, &read, speed))
            .or_raise()?;
        Ok((command.speed, command.turn_rate))
    }

    /// Compute a proportional steering command from a scalar error.
    ///
    /// Satisfies the :class:`Controller` interface for scalar inputs. Use
    /// :meth:`track` for path tracking.
    ///
    /// Args:
    ///     `state`: The current state value.
    ///     `reference`: The reference value to track.
    ///
    /// Returns:
    ///     ``reference - state`` as a float.
    #[pyo3(signature = (state, reference))]
    #[pyo3(text_signature = "(state, reference)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the difference reads no state"
    )]
    const fn control(&self, state: f64, reference: f64) -> f64 {
        reference - state
    }
}

// ---------------------------------------------------------------------
// Rigid bodies
// ---------------------------------------------------------------------

/// Abstract two-dimensional rigid body.
///
/// State is a pose ``(x, y, psi)`` and a velocity ``(vx, vy, omega)``.
/// Wrenches accumulate until the next :meth:`step`, which applies them and
/// clears them, so several contacts in one frame add up rather than the
/// last one winning.
///
/// Subclass it and implement :attr:`inertia` and :attr:`bounding_radius`;
/// the base defines neither. :class:`CircleBody` and :class:`SquareBody`
/// are the two the library ships.
///
/// Args:
///     `mass`: Body mass in kilograms. Must be finite and positive.
///     `x`: Initial x position in meters.
///     `y`: Initial y position in meters.
///     `psi`: Initial heading angle in radians.
///
/// Raises:
///     `ValueError`: If *mass* is not finite and positive, or a pose
///         component is not a real number.
#[pyclass(subclass, name = "RigidBody", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyRigidBody {
    /// Pose, velocity, accumulated wrench and the two shape constants.
    body: Shaped,
    /// Whether a subclass supplied an inertia and a bounding radius.
    ///
    /// False on the base class, whose two abstract properties raise. The
    /// values in `body` are meaningless then and nothing reads them,
    /// because [`PyRigidBody::step`] takes the inertia off the instance.
    has_shape: bool,
}

impl PyRigidBody {
    /// Builds the base state one of the concrete bodies sits on.
    fn of<B: RigidBody>(body: &B) -> Self {
        Self {
            body: Shaped {
                state: *body.state(),
                inertia: body.inertia(),
                bounding_radius: body.bounding_radius(),
            },
            has_shape: true,
        }
    }
}

#[pymethods]
impl PyRigidBody {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a body of *mass* at ``(x, y)`` facing *psi*, at rest.
    #[new]
    #[pyo3(signature = (mass, x = 0.0, y = 0.0, psi = 0.0))]
    #[pyo3(text_signature = "(mass, x=0.0, y=0.0, psi=0.0)")]
    fn new(mass: f64, x: f64, y: f64, psi: f64) -> PyResult<Self> {
        Ok(Self {
            body: Shaped {
                state: BodyState::new(mass, x, y, psi).or_raise()?,
                // Neither is defined on the base, and `has_shape` is what
                // keeps them from being read.
                inertia: f64::NAN,
                bounding_radius: f64::NAN,
            },
            has_shape: false,
        })
    }

    /// Body mass in kilograms.
    #[getter]
    fn mass(&self) -> f64 {
        self.body.state.mass()
    }

    /// Current pose as ``[x, y, psi]``.
    ///
    /// The heading is not wrapped, so a body that has turned three times
    /// is distinguishable from one that has turned once.
    #[getter]
    fn pose<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.body.state.pose())
    }

    /// Current velocity as ``[vx, vy, omega]``.
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.body.state.velocity())
    }

    /// Rotational inertia about the center of mass (kg m^2).
    ///
    /// Raises:
    ///     `NotImplementedError`: On the base class, which has no shape.
    #[getter]
    fn inertia(&self) -> PyResult<f64> {
        if self.has_shape {
            Ok(self.body.inertia)
        } else {
            Err(PyNotImplementedError::new_err(
                "RigidBody.inertia is abstract.",
            ))
        }
    }

    /// Radius of the circle enclosing the body (meters).
    ///
    /// Raises:
    ///     `NotImplementedError`: On the base class, which has no shape.
    #[getter]
    fn bounding_radius(&self) -> PyResult<f64> {
        if self.has_shape {
            Ok(self.body.bounding_radius)
        } else {
            Err(PyNotImplementedError::new_err(
                "RigidBody.bounding_radius is abstract.",
            ))
        }
    }

    /// Accumulate a wrench to be applied at the next :meth:`step`.
    ///
    /// Args:
    ///     `fx`: Force along world x (N).
    ///     `fy`: Force along world y (N).
    ///     `torque`: Torque about z (N m).
    ///
    /// Raises:
    ///     `ValueError`: If a component is not a real number. A NaN force
    ///         would reach the pose through the integrator and stay there,
    ///         and every later comparison against that pose would be
    ///         false.
    #[pyo3(signature = (fx, fy, torque))]
    #[pyo3(text_signature = "(fx, fy, torque)")]
    fn apply_wrench(&mut self, fx: f64, fy: f64, torque: f64) -> PyResult<()> {
        self.body.apply_wrench(fx, fy, torque).or_raise()
    }

    /// Integrate the dynamics forward by *dt* seconds.
    ///
    /// Applies the accumulated wrench with an explicit Euler step, then
    /// clears it. The inertia comes off the instance rather than out of
    /// the base, so a subclass overriding :attr:`inertia` is honored the
    /// way it was when the base was written in Python.
    ///
    /// Args:
    ///     `dt`: Time step in seconds.
    ///
    /// Raises:
    ///     `ValueError`: If *dt* falls outside the interval band, per
    ///         deviation A-17, or the inertia is not finite and positive.
    ///     `NotImplementedError`: On the base class, which has no inertia.
    #[pyo3(signature = (dt))]
    #[pyo3(text_signature = "(dt)")]
    fn step(slf: &Bound<'_, Self>, dt: f64) -> PyResult<()> {
        let inertia = slf.getattr("inertia")?.extract::<f64>()?;
        let mut body = slf.borrow().body;
        body.inertia = inertia;
        slf.py().detach(|| body.step(dt)).or_raise()?;
        slf.borrow_mut().body = body;
        Ok(())
    }

    /// Reset the pose to ``(x, y, psi)`` and zero the velocity.
    ///
    /// Args:
    ///     `x`: New x position in meters.
    ///     `y`: New y position in meters.
    ///     `psi`: New heading angle in radians.
    ///
    /// Raises:
    ///     `ValueError`: If a component is not a real number.
    #[pyo3(signature = (x = 0.0, y = 0.0, psi = 0.0))]
    #[pyo3(text_signature = "(x=0.0, y=0.0, psi=0.0)")]
    fn reset(&mut self, x: f64, y: f64, psi: f64) -> PyResult<()> {
        self.body.reset(x, y, psi).or_raise()
    }
}

/// Uniform-density circular rigid body.
///
/// Inertia is ``mass * radius^2 / 2``, a thin uniform disk about its
/// center, and the bounding radius is the radius.
///
/// Args:
///     `mass`: Body mass in kilograms. Must be finite and positive.
///     `radius`: Circle radius in meters. Must be finite and positive.
///     `x`: Initial x position in meters.
///     `y`: Initial y position in meters.
///     `psi`: Initial heading angle in radians.
///
/// Raises:
///     `ValueError`: If *mass* or *radius* is not finite and positive.
#[pyclass(extends = PyRigidBody, subclass, name = "CircleBody", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyCircleBody {
    /// Circle radius in meters.
    radius: f64,
}

#[pymethods]
impl PyCircleBody {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a disk of *mass* and *radius* at ``(x, y)`` facing *psi*.
    #[new]
    #[pyo3(signature = (mass, radius, x = 0.0, y = 0.0, psi = 0.0))]
    #[pyo3(text_signature = "(mass, radius, x=0.0, y=0.0, psi=0.0)")]
    fn new(mass: f64, radius: f64, x: f64, y: f64, psi: f64) -> PyResult<PyClassInitializer<Self>> {
        let disk = CircleBody::new(mass, radius, x, y, psi).or_raise()?;
        Ok(PyClassInitializer::from(PyRigidBody::of(&disk)).add_subclass(Self { radius }))
    }

    /// Circle radius in meters.
    #[getter]
    fn radius(&self) -> f64 {
        self.radius
    }
}

/// Uniform-density square rigid body.
///
/// Inertia is ``mass * side_length^2 / 6``, a thin uniform plate about its
/// center, and the bounding radius is half the diagonal.
///
/// Args:
///     `mass`: Body mass in kilograms. Must be finite and positive.
///     `side_length`: Length of one side in meters. Must be finite and
///         positive.
///     `x`: Initial x position in meters.
///     `y`: Initial y position in meters.
///     `psi`: Initial heading angle in radians.
///
/// Raises:
///     `ValueError`: If *mass* or *side_length* is not finite and
///         positive.
#[pyclass(extends = PyRigidBody, subclass, name = "SquareBody", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PySquareBody {
    /// Length of one side in meters.
    side_length: f64,
}

#[pymethods]
impl PySquareBody {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a square plate of this mass and side, at ``(x, y)``.
    #[new]
    #[pyo3(signature = (mass, side_length, x = 0.0, y = 0.0, psi = 0.0))]
    #[pyo3(text_signature = "(mass, side_length, x=0.0, y=0.0, psi=0.0)")]
    fn new(
        mass: f64,
        side_length: f64,
        x: f64,
        y: f64,
        psi: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        // The crate names the quantity "side length" and the Python class
        // named the argument, which is what a caller reads in the
        // traceback and what the existing tests match on.
        if !(side_length.is_finite() && side_length > 0.0) {
            return Err(PyValueError::new_err(format!(
                "side_length must be positive, got {side_length:?}."
            )));
        }
        let plate = SquareBody::new(mass, side_length, x, y, psi).or_raise()?;
        Ok(PyClassInitializer::from(PyRigidBody::of(&plate)).add_subclass(Self { side_length }))
    }

    /// Length of one side in meters.
    #[getter]
    fn side_length(&self) -> f64 {
        self.side_length
    }

    /// Return the four corners in world frame.
    ///
    /// Returns:
    ///     Array of shape ``(4, 2)``, counterclockwise from the near left.
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "()")]
    fn corners<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let borrowed = slf.borrow();
        let state = borrowed.as_super().body.state;
        let pose = state.pose();
        let plate = SquareBody::new(
            state.mass(),
            borrowed.side_length,
            pose[0],
            pose[1],
            pose[2],
        )
        .or_raise()?;
        drop(borrowed);
        as_points(slf.py(), &plate.corners())
    }
}

// ---------------------------------------------------------------------
// The actuator array
// ---------------------------------------------------------------------

/// Reads a grasp matrix back column by column.
///
/// The crate publishes the map from actuator forces to a body wrench as
/// an operator rather than as storage, so the numbers come back by
/// probing it with one basis vector per column. That keeps the crate the
/// only place the geometry is written down, at the cost of one
/// multiplication per column, which for the three to eight actuators this
/// array is built with is a few dozen operations.
///
/// # Errors
///
/// Returns whatever [`GraspMatrix::wrench`] returns.
fn grasp_rows(matrix: &arco_control::actuator::GraspMatrix) -> Result<Vec<Vec<f64>>, Error> {
    let width = matrix.width();
    let mut rows = vec![vec![0.0; width]; 3];
    let mut basis = vec![0.0; width];
    for column in 0..width {
        if let Some(slot) = basis.get_mut(column) {
            *slot = 1.0;
        }
        let wrench = matrix.wrench(&basis)?;
        if let Some(slot) = basis.get_mut(column) {
            *slot = 0.0;
        }
        for (row, value) in rows.iter_mut().zip(wrench) {
            if let Some(cell) = row.get_mut(column) {
                *cell = value;
            }
        }
    }
    Ok(rows)
}

/// Array of N contact actuators around a two-dimensional rigid body.
///
/// Each actuator sits at an angle around the body, a standoff beyond its
/// bounding radius, and pushes inward. The grasp matrix maps the per
/// actuator forces to a wrench on the body and allocation inverts it.
/// Both axes of every actuator, angular and radial, are second-order
/// closed loops driven toward a setpoint, so the array is a plant as well
/// as a controller and a caller has to integrate it through
/// :meth:`step_actuators`.
///
/// Args:
///     `actuator_count`: Number of actuators. At least 3, since two can
///         only push along the line between them and the grasp matrix
///         then cannot reach a general planar wrench.
///     `standoff`: Distance beyond the bounding radius at which actuators
///         rest (meters).
///     `omega`: Natural frequency of both actuator loops (rad/s).
///     `zeta`: Damping ratio of both actuator loops.
///     `spring_stiffness`: Contact spring stiffness (N/m).
///
/// Raises:
///     `ValueError`: If *actuator_count* is below 3, if *omega* or
///         *spring_stiffness* is not positive, or if *zeta* or *standoff*
///         is negative.
#[pyclass(subclass, name = "ActuatorArray", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyActuatorArray {
    /// The array this class is a face for.
    inner: ActuatorArray,
}

#[pymethods]
impl PyActuatorArray {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build an array of evenly spaced actuators.
    #[new]
    #[pyo3(signature = (
        actuator_count = 4,
        standoff = 0.05,
        omega = 10.0,
        zeta = 0.7,
        spring_stiffness = 100.0,
    ))]
    #[pyo3(
        text_signature = "(actuator_count=4, standoff=0.05, omega=10.0, zeta=0.7, spring_stiffness=100.0)"
    )]
    fn new(
        actuator_count: usize,
        standoff: f64,
        omega: f64,
        zeta: f64,
        spring_stiffness: f64,
    ) -> PyResult<Self> {
        // The crate counts "actuators" and the Python class named the
        // argument, which is what a caller reads in the traceback.
        if actuator_count < 3 {
            return Err(PyValueError::new_err(format!(
                "actuator_count must be >= 3, got {actuator_count}."
            )));
        }
        let settings = ActuatorSettings {
            standoff,
            natural_frequency: omega,
            damping_ratio: zeta,
            spring_stiffness,
            ..ActuatorSettings::default()
        };
        Ok(Self {
            inner: ActuatorArray::new(actuator_count, settings).or_raise()?,
        })
    }

    /// Number of actuators.
    #[getter]
    fn actuator_count(&self) -> usize {
        self.inner.actuator_count()
    }

    /// Natural frequency of the actuator second-order loop (rad/s).
    #[getter]
    fn omega(&self) -> f64 {
        self.inner.settings().natural_frequency
    }

    /// Damping ratio of the actuator second-order loop.
    #[getter]
    fn zeta(&self) -> f64 {
        self.inner.settings().damping_ratio
    }

    /// Spring stiffness used for the contact force model (N/m).
    #[getter]
    fn spring_stiffness(&self) -> f64 {
        self.inner.settings().spring_stiffness
    }

    /// Current actuator placement angles (radians), shape ``(N,)``.
    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, self.inner.angles())
    }

    /// Current angular velocities (rad/s), shape ``(N,)``.
    #[getter]
    fn angle_velocities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, self.inner.angle_velocities())
    }

    /// Reference angles (radians), shape ``(N,)``.
    #[getter]
    fn ref_angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, self.inner.reference_angles())
    }

    /// Current radial positions (meters), or ``None`` before
    /// :meth:`init_radii`.
    #[getter]
    fn radii<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.radii().map(|values| as_array(py, values))
    }

    /// Reference radial positions (meters), or ``None`` before
    /// :meth:`init_radii`.
    #[getter]
    fn ref_radii<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .reference_radii()
            .map(|values| as_array(py, values))
    }

    /// Current radial velocities (m/s), or ``None`` before
    /// :meth:`init_radii`.
    #[getter]
    fn radii_velocities<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .radii_velocities()
            .map(|values| as_array(py, values))
    }

    /// Override the actuator placement angles, leaving the rates alone.
    ///
    /// Args:
    ///     `angles`: Array of N angles in radians.
    ///
    /// Raises:
    ///     `ValueError`: If the count is wrong or an angle is not a real
    ///         number.
    #[pyo3(signature = (angles))]
    #[pyo3(text_signature = "(angles)")]
    fn set_angles(&mut self, angles: &Bound<'_, PyAny>) -> PyResult<()> {
        let read = coordinates(angles)?;
        self.inner.set_angles(&read).or_raise()
    }

    /// Compute the grasp matrix for the current actuator placement.
    ///
    /// Column ``2i`` is actuator *i* pushing inward and column ``2i + 1``
    /// is the same actuator pushing along the tangent, counterclockwise
    /// positive.
    ///
    /// Args:
    ///     `body`: The rigid body being grasped.
    ///
    /// Returns:
    ///     Array of shape ``(3, 2N)``.
    ///
    /// Raises:
    ///     `ValueError`: If the body's bounding radius is not a positive
    ///         real number.
    #[pyo3(signature = (body))]
    #[pyo3(text_signature = "(body)")]
    fn grasp_matrix<'py>(
        &self,
        py: Python<'py>,
        body: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shaped = read_body(body)?;
        let rows = py
            .detach(|| {
                self.inner
                    .grasp_matrix(&shaped)
                    .and_then(|g| grasp_rows(&g))
            })
            .or_raise()?;
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    /// Compute actuator forces achieving the desired wrench.
    ///
    /// The minimum-norm least-squares solution. Where the wrench is
    /// reachable the result produces it exactly; where it is not, the
    /// result is the closest the array can come, and nothing here reports
    /// which of those happened. Multiply the answer back through the
    /// grasp matrix and compare to find out.
    ///
    /// Args:
    ///     `desired_wrench`: ``[Fx, Fy, torque]`` in world frame.
    ///     `body`: The rigid body being grasped.
    ///
    /// Returns:
    ///     Force vector of shape ``(2N,)``.
    ///
    /// Raises:
    ///     `ValueError`: If the wrench or the body carries a value that is
    ///         not a real number.
    #[pyo3(signature = (desired_wrench, body))]
    #[pyo3(text_signature = "(desired_wrench, body)")]
    fn allocate_forces<'py>(
        &self,
        py: Python<'py>,
        desired_wrench: &Bound<'py, PyAny>,
        body: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let wrench = wrench_of(desired_wrench)?;
        let shaped = read_body(body)?;
        let forces = py
            .detach(|| self.inner.allocate_forces(wrench, &shaped))
            .or_raise()?;
        Ok(as_array(py, &forces))
    }

    /// Compute radial-only actuator forces for the desired wrench.
    ///
    /// What the spring contact model can deliver: a spring at the radial
    /// axis pushes inward and nothing produces a tangential force, so the
    /// tangential entries come back zero.
    ///
    /// Args:
    ///     `desired_wrench`: ``[Fx, Fy, torque]`` in world frame.
    ///     `body`: The rigid body being grasped.
    ///
    /// Returns:
    ///     Force vector of shape ``(2N,)``.
    ///
    /// Raises:
    ///     `ValueError`: If the wrench or the body carries a value that is
    ///         not a real number.
    #[pyo3(signature = (desired_wrench, body))]
    #[pyo3(text_signature = "(desired_wrench, body)")]
    fn allocate_radial_forces<'py>(
        &self,
        py: Python<'py>,
        desired_wrench: &Bound<'py, PyAny>,
        body: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let wrench = wrench_of(desired_wrench)?;
        let shaped = read_body(body)?;
        let forces = py
            .detach(|| self.inner.allocate_radial_forces(wrench, &shaped))
            .or_raise()?;
        Ok(as_array(py, &forces))
    }

    /// Return the world-frame position of every actuator.
    ///
    /// Uses the actual per-actuator radius once the radial axis exists,
    /// and the nominal standoff before that.
    ///
    /// Args:
    ///     `body`: The rigid body.
    ///
    /// Returns:
    ///     Array of shape ``(N, 2)``.
    ///
    /// Raises:
    ///     `ValueError`: If the body's bounding radius is not a positive
    ///         real number.
    #[pyo3(signature = (body))]
    #[pyo3(text_signature = "(body)")]
    fn actuator_positions<'py>(
        &self,
        py: Python<'py>,
        body: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shaped = read_body(body)?;
        let positions = py
            .detach(|| self.inner.actuator_positions(&shaped))
            .or_raise()?;
        as_points(py, &positions)
    }

    /// Apply a force vector to the body through the grasp matrix.
    ///
    /// The wrench reaches the body through its own ``apply_wrench``, so a
    /// subclass overriding that method sees the call the way it did when
    /// this was written in Python.
    ///
    /// Args:
    ///     `forces`: Force vector of shape ``(2N,)``.
    ///     `body`: Target rigid body.
    ///
    /// Raises:
    ///     `ValueError`: If the force vector is not as wide as the grasp
    ///         matrix, or carries a value that is not a real number.
    #[pyo3(signature = (forces, body))]
    #[pyo3(text_signature = "(forces, body)")]
    fn apply_to_body(&self, forces: &Bound<'_, PyAny>, body: &Bound<'_, PyAny>) -> PyResult<()> {
        let read = coordinates(forces)?;
        let shaped = read_body(body)?;
        let wrench = body
            .py()
            .detach(|| {
                self.inner
                    .grasp_matrix(&shaped)
                    .and_then(|matrix| matrix.wrench(&read))
            })
            .or_raise()?;
        body.call_method1("apply_wrench", (wrench[0], wrench[1], wrench[2]))?;
        Ok(())
    }

    /// Set the reference angles so the array lines up with a wrench.
    ///
    /// Rotates the whole array rather than each actuator separately,
    /// which keeps the spacing even and so keeps the precompression bias
    /// in :meth:`compute_ref_radii` cancelling in the net wrench. The
    /// angles computed here are setpoints; the actuators reach them
    /// through :meth:`step_actuators`.
    ///
    /// Args:
    ///     `body`: The rigid body.
    ///     `target_wrench`: Desired ``[Fx, Fy, torque]``.
    ///
    /// Raises:
    ///     `ValueError`: If the wrench carries a value that is not a real
    ///         number.
    #[pyo3(signature = (body, target_wrench))]
    #[pyo3(text_signature = "(body, target_wrench)")]
    fn update_angles_for_target(
        &mut self,
        py: Python<'_>,
        body: &Bound<'_, PyAny>,
        target_wrench: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let wrench = wrench_of(target_wrench)?;
        let shaped = read_body(body)?;
        py.detach(|| self.inner.aim_at(wrench, &shaped)).or_raise()
    }

    /// Create the radial axis at the nominal contact distance.
    ///
    /// Sets every radial position to ``bounding_radius + standoff`` with
    /// zero velocity. Call it before :meth:`step_actuators` when the
    /// spring force model is used.
    ///
    /// Args:
    ///     `body`: The rigid body being manipulated.
    ///
    /// Raises:
    ///     `ValueError`: If the body's bounding radius is not a positive
    ///         real number.
    #[pyo3(signature = (body))]
    #[pyo3(text_signature = "(body)")]
    fn init_radii(&mut self, py: Python<'_>, body: &Bound<'_, PyAny>) -> PyResult<()> {
        let shaped = read_body(body)?;
        py.detach(|| self.inner.init_radii(&shaped)).or_raise()
    }

    /// Set radial setpoints so the springs settle at the desired forces.
    ///
    /// Inverts the spring law: a spring compressed by ``F / k`` pushes
    /// with ``F``, so the setpoint is the nominal radius less that
    /// compression. A spring can only push, so a symmetric precompression
    /// bias makes every desired force non-negative; for an evenly spaced
    /// array that bias cancels in the net wrench.
    ///
    /// Args:
    ///     `body`: The rigid body, which supplies the bounding radius.
    ///     `desired_forces`: Force vector of shape ``(2N,)``. The radial
    ///         components are the ones inverted.
    ///
    /// Raises:
    ///     `ValueError`: If the force vector is not twice the actuator
    ///         count, or carries a value that is not a real number.
    #[pyo3(signature = (body, desired_forces))]
    #[pyo3(text_signature = "(body, desired_forces)")]
    fn compute_ref_radii(
        &mut self,
        py: Python<'_>,
        body: &Bound<'_, PyAny>,
        desired_forces: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let forces = coordinates(desired_forces)?;
        let shaped = read_body(body)?;
        py.detach(|| self.inner.compute_reference_radii(&forces, &shaped))
            .or_raise()
    }

    /// Integrate the second-order actuator dynamics by one time step.
    ///
    /// Forward Euler, position first. The velocity-first ordering gives a
    /// discrete state matrix whose spectral radius exceeds one at the step
    /// sizes this array is used with, so the actuators would diverge
    /// rather than settle. Without a radial axis only the angular
    /// dynamics move.
    ///
    /// Args:
    ///     `dt`: Integration time step in seconds.
    ///
    /// Raises:
    ///     `ValueError`: If *dt* falls outside the interval band, per
    ///         deviation A-17.
    #[pyo3(signature = (dt))]
    #[pyo3(text_signature = "(dt)")]
    fn step_actuators(&mut self, py: Python<'_>, dt: f64) -> PyResult<()> {
        py.detach(|| self.inner.step(dt)).or_raise()
    }

    /// Compute the actuator force vector from the spring contact model.
    ///
    /// The radial force is ``k * max(0, r_nominal - r)`` and the
    /// tangential force is zero, since a spring only pushes. The radial
    /// axis is created first if it does not exist yet, in which case
    /// every force comes back zero.
    ///
    /// Args:
    ///     `body`: The rigid body, which supplies the bounding radius.
    ///
    /// Returns:
    ///     Force vector of shape ``(2N,)``.
    ///
    /// Raises:
    ///     `ValueError`: If the body's bounding radius is not a positive
    ///         real number.
    #[pyo3(signature = (body))]
    #[pyo3(text_signature = "(body)")]
    fn spring_forces<'py>(
        &mut self,
        py: Python<'py>,
        body: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shaped = read_body(body)?;
        let forces = py.detach(|| self.inner.spring_forces(&shaped)).or_raise()?;
        Ok(as_array(py, &forces))
    }

    /// Compute the spring forces and apply them to the body.
    ///
    /// Args:
    ///     `body`: Target rigid body.
    ///
    /// Raises:
    ///     `ValueError`: If the body's bounding radius is not a positive
    ///         real number.
    #[pyo3(signature = (body))]
    #[pyo3(text_signature = "(body)")]
    fn apply_spring_forces_to_body(
        &mut self,
        py: Python<'_>,
        body: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let shaped = read_body(body)?;
        let wrench = py
            .detach(|| {
                let forces = self.inner.spring_forces(&shaped)?;
                self.inner
                    .grasp_matrix(&shaped)
                    .and_then(|matrix| matrix.wrench(&forces))
            })
            .or_raise()?;
        body.call_method1("apply_wrench", (wrench[0], wrench[1], wrench[2]))?;
        Ok(())
    }

    /// Compute the net repulsive wrench acting on the body.
    ///
    /// Each actuator is pushed away from whichever hazard is nearest to
    /// it, static or peer, with magnitude ``k_rep * (d0 - d)^2`` inside
    /// the influence radius and nothing outside it. Local by
    /// construction: it knows one hazard per actuator and cannot reason
    /// about whether pushing away leads anywhere.
    ///
    /// The callable is queried once per actuator per call and each query
    /// crosses back into the interpreter, which is the cost deviation
    /// A-07 describes.
    ///
    /// Args:
    ///     `body`: The rigid body being controlled.
    ///     `nearest_obstacle_fn`: Callable mapping a world point to
    ///         ``(distance, nearest_point)``, measured from the hazard
    ///         center.
    ///     `other_positions`: Array of shape ``(M, 2)`` holding peer
    ///         hazards. Pass an empty ``(0, 2)`` array when there are
    ///         none.
    ///     `k_rep`: Repulsion stiffness (N/m^2).
    ///     `d0`: Influence radius (meters). The force is zero at or
    ///         beyond it.
    ///
    /// Returns:
    ///     Repulsive wrench ``[Fx, Fy, torque]`` in world frame.
    ///
    /// Raises:
    ///     `ValueError`: If *k_rep* or *d0* is not a positive real number.
    ///     Whatever the callable raised, unchanged, when it raises.
    #[pyo3(signature = (body, nearest_obstacle_fn, other_positions, k_rep, d0))]
    #[pyo3(text_signature = "(body, nearest_obstacle_fn, other_positions, k_rep, d0)")]
    fn repulsive_wrench<'py>(
        &self,
        py: Python<'py>,
        body: &Bound<'py, PyAny>,
        nearest_obstacle_fn: &Bound<'py, PyAny>,
        other_positions: &Bound<'py, PyAny>,
        k_rep: f64,
        d0: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shaped = read_body(body)?;
        // Raised rather than dropped: a peer array the reader cannot
        // parse is a caller mistake, and treating it as "no peers" turns
        // an avoidance term off without saying so.
        let peers = waypoints(other_positions)?;
        let failure = FailureSlot::default();
        let query = nearest_obstacle_fn.clone().unbind();
        let parked = failure.clone();
        // The caller's callable is one variant of the hazard policy
        // rather than the whole hook, per ADR-004, so the built-in
        // policies never cross the boundary.
        let policy = HazardPolicy::<SharedOccupancy>::Custom(Box::new(move |point: &[f64]| {
            Python::attach(|inner| {
                query
                    .bind(inner)
                    .call1((as_array(inner, point),))
                    .and_then(|answer| answer.extract::<(f64, Vec<f64>)>())
                    .map(|(distance, obstacle)| NearestObstacle {
                        distance,
                        point: obstacle,
                    })
                    .map_err(|raised| parked.park(raised))
            })
        }));
        let wrench = detached(py, &failure, || {
            self.inner
                .repulsive_wrench(&shaped, &policy, &peers, k_rep, d0)
        })?;
        Ok(as_array(py, &wrench))
    }
}

// ---------------------------------------------------------------------
// Reactive avoidance
// ---------------------------------------------------------------------

/// Artificial potential field turn-rate bias for obstacle avoidance.
///
/// Within ``2 * clearance`` of the nearest obstacle, returns an additive
/// turn-rate bias that steers away from it, with magnitude
/// ``repulsion_gain * (1/d - 1/d_max)`` where *d* is the distance to the
/// obstacle. The magnitude goes to zero at the edge of the influence
/// radius, so the correction switches on smoothly rather than stepping.
///
/// Reactive and local by construction: it sees one obstacle and cannot
/// reason about whether turning away leads anywhere. It is a last defense
/// layered on a planner that already produced a free path.
///
/// Instances satisfy :class:`~arco.protocols.avoidance.AvoidanceStrategy`.
/// Construct with ``occupancy=None`` or a non-positive gain for an
/// instance that always answers ``0.0``.
///
/// Args:
///     `occupancy`: Occupancy map used to find the nearest obstacle. A
///         map publishing ``points`` and ``clearance`` is rebuilt natively
///         and queried without the interpreter; anything else is queried
///         through it, per deviation A-07.
///     `repulsion_gain`: Turn-rate gain (rad/m). Non-positive values
///         disable repulsion.
///
/// Raises:
///     `ValueError`: If *repulsion_gain* is not a real number.
#[pyclass(subclass, name = "ArtificialPotentialField", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyArtificialPotentialField {
    /// The field this class is a face for.
    inner: ArtificialPotentialField<SharedOccupancy>,
    /// The map the caller passed, kept so identity survives the trip.
    occupancy: Option<Py<PyAny>>,
    /// Where a query into a Python map parks the exception it raised.
    failure: FailureSlot,
}

#[pymethods]
impl PyArtificialPotentialField {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a field repelling from *occupancy* at the given gain.
    #[new]
    #[pyo3(signature = (occupancy = None, repulsion_gain = 0.0))]
    #[pyo3(text_signature = "(occupancy=None, repulsion_gain=0.0)")]
    fn new(occupancy: Option<&Bound<'_, PyAny>>, repulsion_gain: f64) -> PyResult<Self> {
        let failure = FailureSlot::default();
        let inner = match occupancy {
            // The gain is kept even with no map: a caller reading
            // `repulsion_gain` back gets what it set, and a map supplied
            // later finds the field already configured.
            None => ArtificialPotentialField::without_map(repulsion_gain),
            Some(map) => {
                let adopted = SharedOccupancy::new(BoundOccupancy::adopt(map, failure.clone()));
                ArtificialPotentialField::new(adopted, repulsion_gain).or_raise()?
            }
        };
        Ok(Self {
            inner,
            occupancy: occupancy.map(|map| map.clone().unbind()),
            failure,
        })
    }

    /// Obstacle-repulsion turn-rate gain (rad/m).
    #[getter]
    fn repulsion_gain(&self) -> f64 {
        self.inner.repulsion_gain()
    }

    /// The occupancy map the field queries, or ``None``.
    #[getter]
    fn _occupancy(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.occupancy.as_ref().map(|map| map.clone_ref(py))
    }

    /// Return the turn-rate correction steering away from an obstacle.
    ///
    /// The sign steers away: an obstacle to the left gives a negative
    /// correction and one to the right a positive one.
    ///
    /// Args:
    ///     `x`: Vehicle x position in world frame (m).
    ///     `y`: Vehicle y position in world frame (m).
    ///     `theta`: Vehicle heading in radians.
    ///
    /// Returns:
    ///     Turn-rate correction in rad/s. Zero when the field is
    ///     disabled, when no map is configured, or outside the influence
    ///     radius.
    ///
    /// Raises:
    ///     `ValueError`: If a pose component is not a real number.
    ///     Whatever the map raised, unchanged, when a query raises.
    #[pyo3(signature = (x, y, theta))]
    fn __call__(&self, py: Python<'_>, x: f64, y: f64, theta: f64) -> PyResult<f64> {
        let pose = Pose::new(x, y, theta).or_raise()?;
        detached(py, &self.failure, || self.inner.turn_rate_bias(pose))
    }
}

// ---------------------------------------------------------------------
// Joint-space tracking
// ---------------------------------------------------------------------

/// Reads a per-axis limit, which Python accepted as a scalar or an array.
///
/// # Errors
///
/// Returns a `ValueError` naming *quantity* when a bound is not finite
/// and strictly positive, in the words the Python constructor used.
pub(crate) fn axis_limits(quantity: &str, value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let read = value
        .extract::<f64>()
        .map_or_else(|_not_scalar| coordinates(value), |scalar| Ok(vec![scalar]))?;
    if read
        .iter()
        .any(|bound| !(bound.is_finite() && *bound > 0.0))
    {
        return Err(PyValueError::new_err(format!(
            "{quantity} must be strictly positive; got {}.",
            value.repr()?
        )));
    }
    Ok(read)
}

/// N-DOF proportional tracker with saturation and potential-field repulsion.
///
/// The reactive tracking layer for configuration-space agents: joint-space
/// arms, Cartesian gantries and body-pose controllers. Each :meth:`step`
/// maps the position error to a velocity, adds the repulsion correction,
/// saturates velocity and acceleration per axis, and integrates once.
///
/// The repulsion velocity points away from the nearest configuration-space
/// obstacle with magnitude ``gain * (1/d - 1/d_max)``, where
/// ``d_max = 2 * clearance`` is the influence radius and *d* is floored at
/// a tenth of the clearance so the reciprocal cannot run away.
///
/// Args:
///     `max_vel`: Per-axis velocity limit, a scalar or a one-dimensional
///         array. A single value applies to every axis.
///     `max_acc`: Per-axis acceleration limit, a scalar or an array.
///     `proportional_gain`: Gain mapping position error to velocity.
///     `occupancy`: Configuration-space occupancy map for repulsion. When
///         it is ``None`` or the gain is zero, repulsion is disabled.
///     `repulsion_gain`: Repulsion gain. Zero disables repulsion.
///
/// Raises:
///     `ValueError`: If an element of *max_vel* or *max_acc* is not
///         strictly positive, or a gain is not a real number.
#[pyclass(subclass, name = "JointSpaceTracker", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyJointSpaceTracker {
    /// The tracker this class is a face for.
    inner: JointSpaceTracker<SharedOccupancy>,
    /// The limits the tracker was built with.
    ///
    /// Held here because the crate publishes no settings accessor, and
    /// because rebuilding them is what reproduces the broadcasting a
    /// one-axis limit had under numpy.
    limits: JointLimits,
    /// Position error to commanded velocity, per second.
    proportional_gain: f64,
    /// How hard to push away from an obstacle.
    repulsion_gain: f64,
    /// The map the caller passed, kept so identity survives the trip.
    occupancy: Option<Py<PyAny>>,
    /// The adopted map, kept so a wider reset can rebuild the tracker.
    adopted: Option<SharedOccupancy>,
    /// Where a query into a Python map parks the exception it raised.
    failure: FailureSlot,
}

impl PyJointSpaceTracker {
    /// Builds the tracker for a configuration of `axes` axes.
    ///
    /// # Errors
    ///
    /// As [`JointSpaceTracker::new`].
    fn assemble(&self, limits: JointLimits) -> PyResult<JointSpaceTracker<SharedOccupancy>> {
        let mut settings = JointTrackerSettings::new(limits);
        settings.proportional_gain = self.proportional_gain;
        settings.repulsion_gain = self.repulsion_gain;
        JointSpaceTracker::new(settings, self.adopted.clone()).or_raise()
    }
}

#[pymethods]
impl PyJointSpaceTracker {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a tracker resting at the origin of its configuration space.
    #[new]
    #[pyo3(signature = (
        max_vel,
        max_acc,
        proportional_gain = 2.0,
        occupancy = None,
        repulsion_gain = 0.0,
    ))]
    #[pyo3(
        text_signature = "(max_vel, max_acc, proportional_gain=2.0, occupancy=None, repulsion_gain=0.0)"
    )]
    fn new(
        max_vel: &Bound<'_, PyAny>,
        max_acc: &Bound<'_, PyAny>,
        proportional_gain: f64,
        occupancy: Option<&Bound<'_, PyAny>>,
        repulsion_gain: f64,
    ) -> PyResult<Self> {
        let velocities = axis_limits("max_vel", max_vel)?;
        let accelerations = axis_limits("max_acc", max_acc)?;
        let limits = JointLimits::new(velocities, accelerations).or_raise()?;
        let failure = FailureSlot::default();
        let adopted =
            occupancy.map(|map| SharedOccupancy::new(BoundOccupancy::adopt(map, failure.clone())));
        let mut settings = JointTrackerSettings::new(limits.clone());
        settings.proportional_gain = proportional_gain;
        settings.repulsion_gain = repulsion_gain;
        Ok(Self {
            inner: JointSpaceTracker::new(settings, adopted.clone()).or_raise()?,
            limits,
            proportional_gain,
            repulsion_gain,
            occupancy: occupancy.map(|map| map.clone().unbind()),
            adopted,
            failure,
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

    /// Repulsion gain. Zero disables repulsion.
    #[getter]
    fn repulsion_gain(&self) -> f64 {
        self.repulsion_gain
    }

    /// Per-axis velocity limit, as the constructor resolved it.
    ///
    /// Published under the name the Python class used, because callers
    /// read the limits back off the tracker to size a trajectory.
    #[getter]
    fn _max_vel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.limits.max_velocity)
    }

    /// Per-axis acceleration limit, as the constructor resolved it.
    #[getter]
    fn _max_acc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.limits.max_acceleration)
    }

    /// Gain mapping position error to commanded velocity.
    #[getter]
    const fn _k_p(&self) -> f64 {
        self.proportional_gain
    }

    /// The occupancy map the tracker queries, or ``None``.
    #[getter]
    fn _occ(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.occupancy.as_ref().map(|map| map.clone_ref(py))
    }

    /// Reset the tracker state to the configuration *q0*.
    ///
    /// Call it before the first :meth:`step` of a trajectory and after any
    /// replanning, since the velocity carried over from the old
    /// trajectory is about a path that no longer exists.
    ///
    /// A tracker built with one limit per axis keeps that axis count. One
    /// built from a single value broadcasts to whatever width *q0* has,
    /// which is what the numpy arithmetic did in Python.
    ///
    /// Args:
    ///     `q0`: Initial configuration.
    ///
    /// Raises:
    ///     `ValueError`: If the axis count is wrong, or a value is not a
    ///         real number.
    #[pyo3(signature = (q0))]
    #[pyo3(text_signature = "(q0)")]
    fn reset(&mut self, q0: &Bound<'_, PyAny>) -> PyResult<()> {
        let configuration = coordinates(q0)?;
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
            self.inner = self.assemble(widened.clone())?;
            self.limits = widened;
        }
        self.inner.reset(&configuration).or_raise()
    }

    /// Run one tracker step toward *target_q*.
    ///
    /// The order is proportional command, then repulsion, then the
    /// velocity clamp again, then the acceleration clamp, then integrate.
    /// Clamping before the repulsion would let the repulsion push the
    /// command back outside the envelope, which is how an avoidance term
    /// commands a velocity no axis can reach.
    ///
    /// Args:
    ///     `target_q`: Target configuration on the planned path.
    ///     `dt`: Integration time step in seconds.
    ///
    /// Returns:
    ///     The configuration after integration.
    ///
    /// Raises:
    ///     `ValueError`: If *target_q* has the wrong axis count, a value
    ///         is not a real number, or *dt* falls outside the interval
    ///         band, per deviation A-17.
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
        let moved = detached(py, &self.failure, || self.inner.step(&target, dt))?;
        Ok(as_array(py, &moved.configuration))
    }
}

// ---------------------------------------------------------------------
// The tracking loop, and the three Python objects it drives
// ---------------------------------------------------------------------

/// What a vehicle reported the last time it was read.
///
/// The crate's [`VehicleModel`] publishes the pose, the speed and the turn
/// rate without a way to report a failure, which a read crossing into the
/// interpreter needs. Reading them once inside the two fallible entry
/// points and answering from the copy afterward is what keeps those three
/// methods honest.
#[derive(Clone, Copy, Debug)]
struct VehicleReading {
    /// Pose after the last read.
    pose: Pose,
    /// Speed after the last read, meters per second.
    speed: f64,
    /// Turn rate after the last read, radians per second.
    turn_rate: f64,
}

/// A vehicle the loop drives through the interpreter.
#[derive(Debug)]
struct BoundVehicle {
    /// The vehicle the caller passed.
    object: Py<PyAny>,
    /// What it reported after the last step.
    reading: VehicleReading,
    /// Where a call that raised parks its exception.
    failure: FailureSlot,
}

impl BoundVehicle {
    /// Adopts `object`, reading its state once.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the pose, the speed or the turn rate
    /// raised.
    fn adopt(object: &Bound<'_, PyAny>, failure: FailureSlot) -> PyResult<Self> {
        Ok(Self {
            object: object.clone().unbind(),
            reading: Self::read(object)?,
            failure,
        })
    }

    /// Reads the three quantities a tracking loop logs.
    ///
    /// # Errors
    ///
    /// Returns whatever the attribute lookups raised, and a `ValueError`
    /// when the pose is not a triple of real numbers.
    fn read(object: &Bound<'_, PyAny>) -> PyResult<VehicleReading> {
        Ok(VehicleReading {
            pose: pose_of(&object.getattr("pose")?)?,
            speed: object.getattr("speed")?.extract::<f64>()?,
            turn_rate: object.getattr("turn_rate")?.extract::<f64>()?,
        })
    }
}

impl BoundVehicle {
    /// Re-reads the vehicle's own attributes.
    ///
    /// The cache exists so the three trait accessors, which take `&self`,
    /// can answer without the interpreter. It has to be refilled before a
    /// step rather than only after one: a caller that assigns
    /// `vehicle.speed` between steps is doing what the Python loop saw
    /// immediately, because it read `self.vehicle.pose` on every pass.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the attributes raised.
    fn refresh(&mut self) -> Result<(), Error> {
        let updated = Python::attach(|py| {
            Self::read(self.object.bind(py)).map_err(|raised| self.failure.park(raised))
        })?;
        self.reading = updated;
        Ok(())
    }
}

impl VehicleModel for BoundVehicle {
    fn pose(&self) -> Pose {
        self.reading.pose
    }

    fn speed(&self) -> f64 {
        self.reading.speed
    }

    fn turn_rate(&self) -> f64 {
        self.reading.turn_rate
    }

    fn step(&mut self, command: Command, dt: f64) -> Result<(), Error> {
        let updated = Python::attach(|py| {
            let object = self.object.bind(py);
            object
                .call_method1("step", (command.speed, command.turn_rate, dt))
                .and_then(|_advanced| Self::read(object))
                .map_err(|raised| self.failure.park(raised))
        })?;
        self.reading = updated;
        Ok(())
    }
}

/// A path tracker the loop drives through the interpreter.
#[derive(Debug)]
struct BoundTracker {
    /// The controller the caller passed.
    object: Py<PyAny>,
    /// What it reported after the last call.
    errors: TrackerErrors,
    /// Where a call that raised parks its exception.
    failure: FailureSlot,
}

impl BoundTracker {
    /// Adopts `object`, reading its error triple once.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the three error attributes raised.
    fn adopt(object: &Bound<'_, PyAny>, failure: FailureSlot) -> PyResult<Self> {
        Ok(Self {
            object: object.clone().unbind(),
            errors: Self::read(object)?,
            failure,
        })
    }

    /// Reads what the controller measured on its last call.
    ///
    /// # Errors
    ///
    /// Returns whatever the attribute lookups raised, which is the
    /// `AttributeError` Python raised for a controller missing one.
    fn read(object: &Bound<'_, PyAny>) -> PyResult<TrackerErrors> {
        Ok(TrackerErrors {
            cross_track: object.getattr("cross_track_error")?.extract::<f64>()?,
            heading: object.getattr("heading_error")?.extract::<f64>()?,
            curvature: object.getattr("curvature")?.extract::<f64>()?,
        })
    }
}

impl BoundTracker {
    /// Re-reads the tracker's own error attributes.
    ///
    /// For the reason [`BoundVehicle::refresh`] exists: the cache lets
    /// `errors()` answer without the interpreter, and a caller that reset
    /// the tracker between steps expects the next step to see it.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the three attributes raised.
    fn refresh(&mut self) -> Result<(), Error> {
        let updated = Python::attach(|py| {
            Self::read(self.object.bind(py)).map_err(|raised| self.failure.park(raised))
        })?;
        self.errors = updated;
        Ok(())
    }
}

impl PathTracker for BoundTracker {
    fn track(&mut self, pose: Pose, path: &[(f64, f64)], speed: f64) -> Result<Command, Error> {
        let answered = Python::attach(|py| {
            let object = self.object.bind(py);
            // The waypoints are handed over as the list of pairs the
            // Python controllers take. Rebuilding it per step rather than
            // holding the caller's own object keeps the adapter free of
            // shared mutable state, and a polyline is a few dozen pairs.
            let waypoints = PyList::new(py, path.iter().map(|&(x, y)| (x, y)))?;
            let command = object
                .call_method1(
                    "track",
                    ((pose.x(), pose.y(), pose.heading()), waypoints, speed),
                )
                .and_then(|answer| answer.extract::<(f64, f64)>())?;
            Ok((command, Self::read(object)?))
        })
        .map_err(|raised: PyErr| self.failure.park(raised))?;
        let ((speed_cmd, turn_rate_cmd), errors) = answered;
        self.errors = errors;
        Ok(Command {
            speed: speed_cmd,
            turn_rate: turn_rate_cmd,
        })
    }

    fn errors(&self) -> TrackerErrors {
        self.errors
    }
}

/// An avoidance strategy the loop calls through the interpreter.
#[derive(Debug)]
struct BoundAvoidance {
    /// The strategy the caller passed or the loop built, if any.
    object: Option<Py<PyAny>>,
    /// Where a call that raised parks its exception.
    failure: FailureSlot,
}

impl AvoidanceStrategy for BoundAvoidance {
    fn turn_rate_bias(&self, pose: Pose) -> Result<f64, Error> {
        let Some(strategy) = self.object.as_ref() else {
            return Ok(0.0);
        };
        Python::attach(|py| {
            strategy
                .bind(py)
                .call1((pose.x(), pose.y(), pose.heading()))
                .and_then(|bias| bias.extract::<f64>())
                .map_err(|raised| self.failure.park(raised))
        })
    }
}

/// Writes one step's sample back as the dictionary Python logged.
///
/// # Errors
///
/// Returns whatever building the dictionary raised.
fn sample_entry<'py>(py: Python<'py>, sample: &TrackingSample) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    entry.set_item("cross_track_error", sample.cross_track_error)?;
    entry.set_item("heading_error", sample.heading_error)?;
    entry.set_item(
        "pose",
        (sample.pose.x(), sample.pose.y(), sample.pose.heading()),
    )?;
    entry.set_item("speed", sample.speed)?;
    entry.set_item("turn_rate", sample.turn_rate)?;
    entry.set_item("curvature", sample.curvature)?;
    entry.set_item("repulsion_turn_rate", sample.avoidance_bias)?;
    Ok(entry)
}

/// Local tracking loop combining a vehicle model and a path controller.
///
/// Closes the feedback loop between a
/// :class:`~arco.protocols.vehicle.VehicleModel` and a
/// :class:`~arco.protocols.path_tracker.PathTracker`. Each :meth:`step`
/// asks the controller for a command, blends in the avoidance bias,
/// applies the command to the vehicle, and records the cross-track error,
/// the heading error, the pose, the speed and the turn rate.
///
/// The order is fixed and worth stating: limiting the command before the
/// avoidance bias would let the bias push it back outside the box, which
/// is how an avoidance term ends up commanding a turn rate no actuator
/// can produce. The limits themselves are infinite unless a caller asks
/// for them, per deviation A-09, so the closed loop matches the Python
/// one.
///
/// Args:
///     `vehicle`: Kinematic vehicle model.
///     `controller`: Path tracker.
///     `cruise_speed`: Desired forward speed (m/s).
///     `curvature_gain`: Speed modulation gain (m). Positive values slow
///         the vehicle on curves through
///         ``v = cruise_speed / (1 + gain * |curvature|)``, using the
///         curvature the tracker reported on the previous step. Zero
///         holds the cruise speed.
///     `occupancy`: Occupancy map used to build the default potential
///         field when *avoidance* is omitted. Ignored when *avoidance* is
///         given.
///     `repulsion_gain`: Turn-rate gain for that default field. A
///         positive value with a map is what switches it on.
///     `avoidance`: An avoidance strategy to use instead of the default
///         field.
///
/// Raises:
///     `ValueError`: If the cruise speed or the curvature gain is not a
///         real number, or the curvature gain is negative.
#[pyclass(subclass, name = "TrackingLoop", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyTrackingLoop {
    /// The loop this class is a face for.
    inner: TrackingLoop<BoundVehicle, BoundTracker, BoundAvoidance>,
    /// The settings the loop was built with, which it does not publish.
    settings: TrackingSettings,
    /// The strategy the loop calls, or nothing.
    avoidance: Option<Py<PyAny>>,
    /// The map the caller passed, kept so identity survives the trip.
    occupancy: Option<Py<PyAny>>,
    /// The gain the default field was built with.
    repulsion_gain: f64,
    /// Where a call into one of the three objects parks its exception.
    failure: FailureSlot,
}

#[pymethods]
impl PyTrackingLoop {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a loop around a vehicle and a controller.
    #[new]
    #[pyo3(signature = (
        vehicle,
        controller,
        cruise_speed = 1.0,
        curvature_gain = 0.0,
        occupancy = None,
        repulsion_gain = 0.0,
        avoidance = None,
    ))]
    #[pyo3(
        text_signature = "(vehicle, controller, cruise_speed=1.0, curvature_gain=0.0, occupancy=None, repulsion_gain=0.0, avoidance=None)"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python signature is the contract, per FR-API-02"
    )]
    fn new(
        py: Python<'_>,
        vehicle: &Bound<'_, PyAny>,
        controller: &Bound<'_, PyAny>,
        cruise_speed: f64,
        curvature_gain: f64,
        occupancy: Option<&Bound<'_, PyAny>>,
        repulsion_gain: f64,
        avoidance: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let failure = FailureSlot::default();
        let strategy = match (avoidance, occupancy) {
            (Some(injected), _any_map) => Some(injected.clone().unbind()),
            // The default field is a real `ArtificialPotentialField`, so
            // a caller reading it back off the loop meets the class it
            // met before, and the loop and a standalone field compute the
            // same bias from the same code.
            (None, Some(map)) if repulsion_gain > 0.0 => Some(
                Py::new(
                    py,
                    PyArtificialPotentialField::new(Some(map), repulsion_gain)?,
                )?
                .into_any(),
            ),
            (None, _disabled) => None,
        };
        let settings = TrackingSettings {
            cruise_speed,
            curvature_gain,
            // Deviation A-09 adds the mechanism, not a limit nobody chose.
            limits: CommandLimits::default(),
            // Deviation A-19: the Python history grew without bound, and
            // a caller wanting a ring buffer sets a capacity in Rust.
            history_capacity: None,
        };
        let inner = TrackingLoop::new(
            BoundVehicle::adopt(vehicle, failure.clone())?,
            BoundTracker::adopt(controller, failure.clone())?,
            BoundAvoidance {
                object: strategy.as_ref().map(|object| object.clone_ref(py)),
                failure: failure.clone(),
            },
            settings,
        )
        .or_raise()?;
        Ok(Self {
            inner,
            settings,
            avoidance: strategy,
            occupancy: occupancy.map(|map| map.clone().unbind()),
            repulsion_gain,
            failure,
        })
    }

    /// The vehicle being driven.
    #[getter]
    fn vehicle(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner.vehicle().object.clone_ref(py)
    }

    /// The controller producing commands.
    #[getter]
    fn controller(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner.tracker().object.clone_ref(py)
    }

    /// Desired forward speed passed to the controller (m/s).
    #[getter]
    const fn cruise_speed(&self) -> f64 {
        self.settings.cruise_speed
    }

    /// Curvature-to-speed scaling factor (m).
    #[getter]
    const fn curvature_gain(&self) -> f64 {
        self.settings.curvature_gain
    }

    /// Turn-rate gain the default potential field was built with (rad/m).
    #[getter]
    const fn repulsion_gain(&self) -> f64 {
        self.repulsion_gain
    }

    /// The occupancy map the default field queries, or ``None``.
    #[getter]
    fn _occupancy(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.occupancy.as_ref().map(|map| map.clone_ref(py))
    }

    /// The avoidance strategy in force, or ``None``.
    #[getter]
    fn _avoidance(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.avoidance.as_ref().map(|object| object.clone_ref(py))
    }

    /// Most recent step metrics, or ``None`` before the first step.
    #[getter]
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .last()
            .map(|sample| sample_entry(py, sample))
            .transpose()
    }

    /// Full per-step metrics history, as a copy.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let entries = self
            .inner
            .history()
            .map(|sample| sample_entry(py, sample))
            .collect::<PyResult<Vec<Bound<'py, PyDict>>>>()?;
        PyList::new(py, entries)
    }

    /// Compute the obstacle-repulsion turn-rate correction.
    ///
    /// Delegates to the configured avoidance strategy, the default field
    /// or an injected one, and answers ``0.0`` when avoidance is
    /// disabled.
    ///
    /// Args:
    ///     `x`: Vehicle x position in world frame (m).
    ///     `y`: Vehicle y position in world frame (m).
    ///     `theta`: Vehicle heading in radians.
    ///
    /// Returns:
    ///     Turn-rate correction in rad/s.
    #[pyo3(signature = (x, y, theta))]
    #[pyo3(text_signature = "(x, y, theta)")]
    fn _repulsion_turn_rate(&self, py: Python<'_>, x: f64, y: f64, theta: f64) -> PyResult<f64> {
        let Some(strategy) = self.avoidance.as_ref() else {
            return Ok(0.0);
        };
        strategy.bind(py).call1((x, y, theta))?.extract::<f64>()
    }

    /// Run one tracking iteration along *path*.
    ///
    /// Args:
    ///     `path`: Reference path as an ordered list of ``(x, y)``
    ///         waypoints.
    ///     `dt`: Integration time step (s).
    ///
    /// Returns:
    ///     A dictionary carrying ``cross_track_error``, ``heading_error``,
    ///     ``pose``, ``speed``, ``turn_rate``, ``curvature`` and
    ///     ``repulsion_turn_rate``.
    ///
    /// Raises:
    ///     `ValueError`: If a waypoint is not a finite pair of numbers, or
    ///         *dt* falls outside the interval band, per deviation A-17.
    ///     Whatever the vehicle, the controller or the avoidance strategy
    ///     raised, unchanged, when one of them raises.
    #[pyo3(signature = (path, dt = 0.1))]
    #[pyo3(text_signature = "(path, dt=0.1)")]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let read = waypoints(path)?;
        // Re-read the vehicle and the tracker before the step, so a
        // caller that assigned to either between steps sees the change
        // take effect on this pass as it did in the Python loop.
        let surface = |slot: &FailureSlot, failure: &Error| {
            slot.take().unwrap_or_else(|| to_exception(failure))
        };
        self.inner
            .vehicle_mut()
            .refresh()
            .map_err(|failure| surface(&self.failure, &failure))?;
        self.inner
            .tracker_mut()
            .refresh()
            .map_err(|failure| surface(&self.failure, &failure))?;
        // The lock is held across the step on purpose. The vehicle, the
        // controller and the avoidance strategy are all Python objects,
        // so releasing it here only to reacquire it three times inside
        // costs more than the arithmetic between the calls.
        let sample = self.inner.step(&read, dt).map_err(|failure| {
            self.failure
                .take()
                .unwrap_or_else(|| to_exception(&failure))
        })?;
        if let Some(raised) = self.failure.take() {
            return Err(raised);
        }
        sample_entry(py, &sample)
    }

    /// Run several tracking steps.
    ///
    /// Args:
    ///     `path`: Reference path as an ordered list of ``(x, y)``
    ///         waypoints.
    ///     `steps`: Number of steps to simulate.
    ///     `dt`: Integration time step (s).
    ///
    /// Returns:
    ///     One metrics dictionary per step, in order.
    ///
    /// Raises:
    ///     As :meth:`step`, on the first step that fails.
    #[pyo3(signature = (path, steps, dt = 0.1))]
    #[pyo3(text_signature = "(path, steps, dt=0.1)")]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        path: &Bound<'py, PyAny>,
        // Signed, because Python returned an empty list for a negative
        // count and `usize` makes PyO3 raise `OverflowError` before the
        // loop can decline to run.
        steps: i64,
        dt: f64,
    ) -> PyResult<Bound<'py, PyList>> {
        // Grown as the steps run rather than reserved from the argument.
        // `run(path, 10**12)` would otherwise ask the allocator for the
        // whole thing up front and abort the interpreter, where Python
        // appended one entry at a time and the caller could interrupt it.
        let mut entries = Vec::new();
        for _ in 0..steps.max(0) {
            entries.push(self.step(py, path, dt)?);
        }
        PyList::new(py, entries)
    }
}

/// Registers the control classes on the compiled module.
///
/// # Errors
///
/// Returns an error when a registration fails, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyController>()?;
    module.add_class::<PyPidController>()?;
    module.add_class::<PyPurePursuitController>()?;
    module.add_class::<PyRigidBody>()?;
    module.add_class::<PyCircleBody>()?;
    module.add_class::<PySquareBody>()?;
    module.add_class::<PyActuatorArray>()?;
    module.add_class::<PyArtificialPotentialField>()?;
    module.add_class::<PyJointSpaceTracker>()?;
    module.add_class::<PyTrackingLoop>()?;

    // `RigidBody` names the two properties a subclass has to supply, the
    // way the Python base did through `abc`. A compiled class builds in
    // `__new__`, which does not consult the flag this sets, so the base
    // still constructs and a subclass forwarding to `super().__init__`
    // keeps working; what the caller gets is the list of what is missing
    // rather than a silent `None` from a property nobody implemented.
    // `abc` refuses to construct a class with an unimplemented abstract
    // method, and the interpreter honors `__abstractmethods__` on any
    // type. Both bases carried one in Python, so both carry it here:
    // `Controller.control` was as abstract as `RigidBody.inertia`, and
    // leaving it off makes a base instantiable that never was.
    module.getattr("RigidBody")?.setattr(
        "__abstractmethods__",
        pyo3::types::PyFrozenSet::new(module.py(), ["inertia", "bounding_radius"])?,
    )?;
    module.getattr("Controller")?.setattr(
        "__abstractmethods__",
        pyo3::types::PyFrozenSet::new(module.py(), ["control"])?,
    )?;
    Ok(())
}
