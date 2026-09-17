// The doc comments in this module become Python docstrings, printed
// verbatim by `help()`. A backtick around a Python type or argument name
// would show up as punctuation to the reader they are written for.
#![expect(
    clippy::doc_markdown,
    reason = "these doc comments are Python docstrings, rendered verbatim by help()"
)]

//! `arco.kinematics`: the two serial arms, unchanged from Python.
//!
//! Both classes keep the argument names, positional order and default
//! values the Python implementation accepted, per `FR-API-02`. Link
//! lengths and lift bounds are validated here rather than deeper down, so
//! that a message names `l1` and `z_max` the way the caller spelled them
//! instead of naming the Rust fields underneath.
//!
//! No method here releases the interpreter lock. Every one of them is a
//! handful of trigonometric operations whose cost does not grow with any
//! argument, and releasing the lock around work that short costs a caller
//! more than it returns. The rule in `.guidelines/languages/rs.md` earns
//! its place where the work scales with the input, which in this crate
//! means the occupancy and graph queries in [`crate::mapping`].

use arco_kinematics::rr::{JointAngles, RrRobot};
use arco_kinematics::rrp::{Configuration, RrpRobot};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::errors::OrRaise;

/// The three points of a planar arm: base, elbow, and end effector.
type PlanarLinkPoints = ((f64, f64), (f64, f64), (f64, f64));

/// The same three points for an arm carried on a lift.
type SpatialLinkPoints = ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64));

/// Rejects a link length the Python implementation refused, in its words.
fn require_positive_length(name: &str, value: f64) -> PyResult<()> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive, got {value:?}."
        )))
    }
}

/// A two-link planar revolute-revolute robot arm.
///
/// The arm is mounted at the world origin and works in the XY plane.
/// Joint 1 rotates the first link of length *l1* about the Z axis, and
/// joint 2 rotates the second link of length *l2* relative to the first.
///
/// Args:
///     l1: Length of the first link in metres. Must be positive.
///     l2: Length of the second link in metres. Must be positive. The
///         second link is typically shorter than the first, which models
///         a SCARA-style pick-and-place arm.
///
/// Raises:
///     ValueError: If *l1* or *l2* is not strictly positive.
#[pyclass(name = "RRRobot", module = "arco._arco", subclass, frozen)]
#[derive(Debug)]
pub struct PyRrRobot {
    inner: RrRobot,
}

#[pymethods]
impl PyRrRobot {
    #[new]
    #[pyo3(signature = (l1 = 1.0, l2 = 0.8))]
    fn new(l1: f64, l2: f64) -> PyResult<Self> {
        require_positive_length("l1", l1)?;
        require_positive_length("l2", l2)?;
        Ok(Self {
            inner: RrRobot::new(l1, l2).or_raise()?,
        })
    }

    /// Length of the first link in metres.
    #[getter]
    fn l1(&self) -> f64 {
        self.inner.first_link_length()
    }

    /// Length of the second link in metres.
    #[getter]
    fn l2(&self) -> f64 {
        self.inner.second_link_length()
    }

    /// Return the end-effector position for a pair of joint angles.
    ///
    /// Args:
    ///     q1: First joint angle in radians, the rotation of link 1 about
    ///         the world Z axis.
    ///     q2: Second joint angle in radians, relative to link 1.
    ///
    /// Returns:
    ///     ``(x, y)`` Cartesian position of the end-effector in metres.
    ///
    /// Raises:
    ///     ValueError: If either angle is NaN or infinite.
    #[pyo3(signature = (q1, q2))]
    fn forward_kinematics(&self, q1: f64, q2: f64) -> PyResult<(f64, f64)> {
        self.inner
            .forward_kinematics(JointAngles {
                shoulder: q1,
                elbow: q2,
            })
            .or_raise()
    }

    /// Return the joint angles placing the end-effector at ``(x, y)``.
    ///
    /// Solved by the law of cosines, which gives up to two answers: elbow
    /// down, with a positive *q2*, first, and elbow up second. On the
    /// boundary of the reachable annulus the two coincide. A target
    /// outside ``[|l1 - l2|, l1 + l2]`` gives an empty list rather than an
    /// approximation.
    ///
    /// Args:
    ///     x: Desired end-effector x coordinate in metres.
    ///     y: Desired end-effector y coordinate in metres.
    ///     eps: Numerical tolerance for the workspace boundary check, in
    ///         metres. It absorbs the rounding the caller's own arithmetic
    ///         introduced.
    ///
    /// Returns:
    ///     A list of ``(q1, q2)`` tuples in radians, holding 0 or 2
    ///     elements.
    ///
    /// Raises:
    ///     ValueError: If *x* or *y* is NaN or infinite, or *eps* is
    ///         negative.
    #[pyo3(signature = (x, y, eps = 1e-9))]
    fn inverse_kinematics(&self, x: f64, y: f64, eps: f64) -> PyResult<Vec<(f64, f64)>> {
        Ok(self
            .inner
            .inverse_kinematics(x, y, eps)
            .or_raise()?
            .into_iter()
            .map(|angles| (angles.shoulder, angles.elbow))
            .collect())
    }

    /// Return the three key points of the arm geometry.
    ///
    /// Args:
    ///     q1: First joint angle in radians.
    ///     q2: Second joint angle in radians.
    ///
    /// Returns:
    ///     ``(origin, joint2, end_effector)``, each an ``(x, y)`` pair in
    ///     metres.
    ///
    /// Raises:
    ///     ValueError: If either angle is NaN or infinite.
    #[pyo3(signature = (q1, q2))]
    fn link_segments(&self, q1: f64, q2: f64) -> PyResult<PlanarLinkPoints> {
        let [origin, joint2, effector] = self
            .inner
            .link_segments(JointAngles {
                shoulder: q1,
                elbow: q2,
            })
            .or_raise()?;
        Ok((origin, joint2, effector))
    }

    /// Return the maximum reach of the arm.
    ///
    /// Returns:
    ///     ``l1 + l2`` in metres.
    fn workspace_radius(&self) -> f64 {
        self.inner.workspace_radius()
    }

    /// Return the inner and outer radii of the reachable annulus.
    ///
    /// The inner radius is ``abs(l1 - l2)``, which is zero when the links
    /// are the same length and the arm can fold back onto its own base.
    ///
    /// Returns:
    ///     ``(r_min, r_max)`` in metres.
    fn workspace_annulus(&self) -> (f64, f64) {
        self.inner.workspace_annulus()
    }

    fn __repr__(&self) -> String {
        format!("RRRobot(l1={}, l2={})", self.l1(), self.l2())
    }
}

/// A two-link planar RR arm on a vertical prismatic joint, SCARA-like.
///
/// The XY kinematics are those of :class:`RRRobot`. The prismatic joint
/// *z* lifts the whole arm assembly along the world Z axis.
///
/// Args:
///     l1: Length of the first revolute link in metres. Must be positive.
///     l2: Length of the second revolute link in metres. Must be positive.
///     z_min: Minimum height of the prismatic joint in metres.
///     z_max: Maximum height of the prismatic joint in metres. Must be
///         strictly greater than *z_min*.
///
/// Raises:
///     ValueError: If *l1* or *l2* is not strictly positive, or if *z_max*
///         is not greater than *z_min*.
#[pyclass(name = "RRPRobot", module = "arco._arco", subclass, frozen)]
#[derive(Debug)]
pub struct PyRrpRobot {
    inner: RrpRobot,
}

#[pymethods]
impl PyRrpRobot {
    #[new]
    #[pyo3(signature = (l1 = 1.0, l2 = 0.8, z_min = 0.0, z_max = 4.0))]
    fn new(l1: f64, l2: f64, z_min: f64, z_max: f64) -> PyResult<Self> {
        require_positive_length("l1", l1)?;
        require_positive_length("l2", l2)?;
        if z_max <= z_min {
            return Err(PyValueError::new_err(format!(
                "z_max ({z_max:?}) must be greater than z_min ({z_min:?})."
            )));
        }
        Ok(Self {
            inner: RrpRobot::new(l1, l2, z_min, z_max).or_raise()?,
        })
    }

    /// Length of the first revolute link in metres.
    #[getter]
    fn l1(&self) -> f64 {
        self.inner.arm().first_link_length()
    }

    /// Length of the second revolute link in metres.
    #[getter]
    fn l2(&self) -> f64 {
        self.inner.arm().second_link_length()
    }

    /// Minimum prismatic joint height in metres.
    #[getter]
    fn z_min(&self) -> f64 {
        self.inner.height_range().0
    }

    /// Maximum prismatic joint height in metres.
    #[getter]
    fn z_max(&self) -> f64 {
        self.inner.height_range().1
    }

    /// Return the end-effector position for a set of joint values.
    ///
    /// Args:
    ///     q1: First revolute joint angle in radians.
    ///     q2: Second revolute joint angle in radians, relative to link 1.
    ///     z: Prismatic joint height in metres.
    ///
    /// Returns:
    ///     ``(x, y, z)`` Cartesian position of the end-effector in metres.
    ///
    /// Raises:
    ///     ValueError: If an angle is NaN or infinite, or if *z* falls
    ///         outside ``[z_min, z_max]``.
    #[pyo3(signature = (q1, q2, z))]
    fn forward_kinematics(&self, q1: f64, q2: f64, z: f64) -> PyResult<(f64, f64, f64)> {
        self.inner
            .forward_kinematics(configuration(q1, q2, z))
            .or_raise()
    }

    /// Return the revolute joint angles placing the end-effector at ``(x, y)``.
    ///
    /// The prismatic joint does not enter the XY kinematics and plays no
    /// part here. As on :class:`RRRobot`, the answer holds elbow-down
    /// first and elbow-up second, and a target outside the reachable
    /// annulus gives an empty list.
    ///
    /// Args:
    ///     x: Desired end-effector x coordinate in metres.
    ///     y: Desired end-effector y coordinate in metres.
    ///     eps: Numerical tolerance for the workspace boundary check, in
    ///         metres.
    ///
    /// Returns:
    ///     A list of ``(q1, q2)`` tuples in radians.
    ///
    /// Raises:
    ///     ValueError: If *x* or *y* is NaN or infinite, or *eps* is
    ///         negative.
    #[pyo3(signature = (x, y, eps = 1e-9))]
    fn inverse_kinematics_xy(&self, x: f64, y: f64, eps: f64) -> PyResult<Vec<(f64, f64)>> {
        // Through the arm rather than through the lift, because the Python
        // method takes no height and therefore cannot check one.
        Ok(self
            .inner
            .arm()
            .inverse_kinematics(x, y, eps)
            .or_raise()?
            .into_iter()
            .map(|angles| (angles.shoulder, angles.elbow))
            .collect())
    }

    /// Return the three key 3-D points of the arm geometry.
    ///
    /// Args:
    ///     q1: First revolute joint angle in radians.
    ///     q2: Second revolute joint angle in radians.
    ///     z: Prismatic joint height in metres.
    ///
    /// Returns:
    ///     ``(origin, joint2, end_effector)``, each an ``(x, y, z)`` triple
    ///     in metres, all three at the height *z*.
    ///
    /// Raises:
    ///     ValueError: If an angle is NaN or infinite, or if *z* falls
    ///         outside ``[z_min, z_max]``.
    #[pyo3(signature = (q1, q2, z))]
    fn link_segments(&self, q1: f64, q2: f64, z: f64) -> PyResult<SpatialLinkPoints> {
        // The crate reports the fixed base at z_min as a fourth point
        // ahead of the three that move. Python never had it, so it is
        // dropped here rather than changing what a caller unpacks.
        let [_base, origin, joint2, effector] = self
            .inner
            .link_segments(configuration(q1, q2, z))
            .or_raise()?;
        Ok((origin, joint2, effector))
    }

    /// Return the maximum horizontal reach of the arm.
    ///
    /// Returns:
    ///     ``l1 + l2`` in metres.
    fn workspace_radius(&self) -> f64 {
        self.inner.workspace_annulus().1
    }

    /// Return the inner and outer radii of the reachable horizontal annulus.
    ///
    /// Returns:
    ///     ``(r_min, r_max)`` in metres, where ``r_min`` is
    ///     ``abs(l1 - l2)``.
    fn workspace_annulus(&self) -> (f64, f64) {
        self.inner.workspace_annulus()
    }

    fn __repr__(&self) -> String {
        format!(
            "RRPRobot(l1={}, l2={}, z_min={}, z_max={})",
            self.l1(),
            self.l2(),
            self.z_min(),
            self.z_max()
        )
    }
}

/// Packs the three Python joint values into the crate's configuration.
const fn configuration(q1: f64, q2: f64, z: f64) -> Configuration {
    Configuration {
        angles: JointAngles {
            shoulder: q1,
            elbow: q2,
        },
        height: z,
    }
}

/// Adds `arco.kinematics` to the compiled module.
///
/// # Arguments
///
/// * `module` - The module every ARCO binding registers into.
///
/// # Errors
///
/// Returns an error when a class fails to register, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRrRobot>()?;
    module.add_class::<PyRrpRobot>()?;
    Ok(())
}
