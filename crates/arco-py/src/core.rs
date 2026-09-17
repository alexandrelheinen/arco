// The doc comments in this module become Python docstrings, printed
// verbatim by `help()`. A backtick around a Python type or argument name
// would show up as punctuation to the reader they are written for.
#![expect(
    clippy::doc_markdown,
    reason = "these doc comments are Python docstrings, rendered verbatim by help()"
)]

//! The shared geometry, tolerance and random-number surface.
//!
//! Everything here comes from `arco-core`, the crate every other ARCO
//! crate sits on. The Python library never exposed these directly, so
//! nothing in this module is constrained by `FR-API-01`: the names are
//! chosen to read the way the rest of the package reads, and they are
//! reachable from `arco._arco` for the binding layer and the parity tests
//! that need a tolerance or a reproducible draw without importing a
//! planner.
//!
//! Two conversions happen at this boundary and nowhere else. A sequence of
//! floats arriving from Python becomes a slice through the reader below,
//! which borrows a numpy array rather than walking it element by element,
//! and an [`arco_core::Error`] leaving it becomes an exception through
//! [`crate::errors`].

use arco_core::geometry::{self, Pose};
use arco_core::numeric;
use arco_core::rng::Pcg64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::errors::OrRaise;

/// Reads a sequence of floats, borrowing a numpy array where it can.
///
/// A `float64` array is borrowed and copied out of its buffer in one go;
/// anything else, a list or a tuple or an integer array, goes through the
/// ordinary conversion. Both end in an owned vector, because every crate
/// below takes a slice and because the interpreter lock is released
/// before the work starts.
///
/// # Arguments
///
/// * `value` - A numpy array of shape `(D,)`, or any sequence of floats.
///
/// # Errors
///
/// Returns `TypeError` when `value` is neither.
pub(crate) fn read_vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_array().to_vec());
    }
    value.extract::<Vec<f64>>()
}

/// Reads a sequence of points, borrowing a numpy array where it can.
///
/// A one-dimensional input is read as a single point, matching
/// `np.asarray(points).reshape(1, -1)` in the Python implementation.
///
/// # Arguments
///
/// * `value` - A numpy array of shape `(N, D)` or `(D,)`, or a sequence of
///   sequences of floats.
///
/// # Errors
///
/// Returns `TypeError` when `value` is none of those, and `ValueError`
/// when the rows disagree in length.
pub(crate) fn read_matrix(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(array
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect());
    }
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(vec![array.as_array().to_vec()]);
    }
    value.extract::<Vec<Vec<f64>>>()
}

/// A pose in the plane: a position and a heading.
///
/// The heading is wrapped into ``[-pi, pi)`` when the pose is built, so a
/// difference taken through :meth:`heading_difference` is right across the
/// branch cut instead of being wrong by a full turn.
///
/// Args:
///     x: Position along the first axis, in meters.
///     y: Position along the second axis, in meters.
///     heading: Orientation in radians, wrapped into ``[-pi, pi)``.
///
/// Raises:
///     ValueError: If any argument is NaN or infinite.
#[pyclass(name = "Pose", module = "arco._arco", frozen)]
#[derive(Debug)]
pub struct PyPose {
    inner: Pose,
}

#[pymethods]
impl PyPose {
    #[new]
    #[pyo3(signature = (x, y, heading))]
    fn new(x: f64, y: f64, heading: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Pose::new(x, y, heading).or_raise()?,
        })
    }

    /// Position along the first axis, in meters.
    #[getter]
    fn x(&self) -> f64 {
        self.inner.x()
    }

    /// Position along the second axis, in meters.
    #[getter]
    fn y(&self) -> f64 {
        self.inner.y()
    }

    /// Orientation in radians, always in ``[-pi, pi)``.
    #[getter]
    fn heading(&self) -> f64 {
        self.inner.heading()
    }

    /// Return the straight-line distance to *other* in meters.
    ///
    /// Headings play no part: this is the distance between the two
    /// positions.
    ///
    /// Args:
    ///     other: The pose to measure to.
    ///
    /// Returns:
    ///     Distance in meters.
    #[pyo3(signature = (other))]
    fn distance_to(&self, other: &Self) -> f64 {
        self.inner.distance_to(other.inner)
    }

    /// Return the signed heading difference ``self - other`` in radians.
    ///
    /// The result is wrapped into ``[-pi, pi)``, so two headings either
    /// side of the branch cut differ by a small number rather than by
    /// almost a full turn.
    ///
    /// Args:
    ///     other: The pose to measure against.
    ///
    /// Returns:
    ///     Difference in radians.
    #[pyo3(signature = (other))]
    fn heading_difference(&self, other: &Self) -> PyResult<f64> {
        self.inner.heading_difference(other.inner).or_raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "Pose(x={}, y={}, heading={})",
            self.inner.x(),
            self.inner.y(),
            self.inner.heading()
        )
    }
}

/// Return the Euclidean distance between two points of equal dimension.
///
/// Args:
///     a: First point, as a numpy array of shape ``(D,)`` or any sequence
///         of floats.
///     b: Second point, of the same length as *a*.
///
/// Returns:
///     Distance as a float, in the unit the coordinates were given in.
///
/// Raises:
///     ValueError: If the two points differ in length, or either carries a
///         NaN or an infinity.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn euclidean_distance(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    let left = read_vector(a)?;
    let right = read_vector(b)?;
    py.detach(|| geometry::euclidean_distance(&left, &right))
        .or_raise()
}

/// Return the Manhattan distance between two points of equal dimension.
///
/// Args:
///     a: First point, as a numpy array of shape ``(D,)`` or any sequence
///         of floats.
///     b: Second point, of the same length as *a*.
///
/// Returns:
///     Sum of the per-axis differences, as a float.
///
/// Raises:
///     ValueError: As :func:`euclidean_distance`.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn manhattan_distance(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    let left = read_vector(a)?;
    let right = read_vector(b)?;
    py.detach(|| geometry::manhattan_distance(&left, &right))
        .or_raise()
}

/// Return True when two points agree within the position tolerance.
///
/// The comparison is absolute near the origin and relative away from it,
/// so it stays meaningful at both ends of the coordinate range.
///
/// Args:
///     a: First point, as a numpy array of shape ``(D,)`` or any sequence
///         of floats.
///     b: Second point, of the same length as *a*.
///
/// Returns:
///     True when the two agree.
///
/// Raises:
///     ValueError: As :func:`euclidean_distance`.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn points_close(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let left = read_vector(a)?;
    let right = read_vector(b)?;
    py.detach(|| geometry::points_close(&left, &right))
        .or_raise()
}

/// Return True when two values agree within the tolerances given.
///
/// Both tolerances are required, because a comparison that picks one for
/// the caller is how a single project-wide epsilon gets established by
/// accident. The relation is not transitive, so it is never a sort key, a
/// dictionary key, or a deduplication criterion.
///
/// Args:
///     left: First value.
///     right: Second value.
///     absolute: Absolute tolerance, in the unit of the quantity. Governs
///         near zero, where no relative tolerance means anything.
///     relative: Relative tolerance, dimensionless. Governs away from
///         zero.
///
/// Returns:
///     True when the two agree.
#[pyfunction]
#[pyo3(signature = (left, right, absolute, relative))]
fn is_close(left: f64, right: f64, absolute: f64, relative: f64) -> bool {
    numeric::is_close(left, right, absolute, relative)
}

/// Return True when two positions agree, in meters.
///
/// Args:
///     left: First position, in meters.
///     right: Second position, in meters.
///
/// Returns:
///     True when the two agree within ``POSITION_TOLERANCE``.
#[pyfunction]
#[pyo3(signature = (left, right))]
fn positions_close(left: f64, right: f64) -> bool {
    numeric::positions_close(left, right)
}

/// Return True when two angles agree, in radians, across the branch cut.
///
/// Args:
///     left: First angle, in radians.
///     right: Second angle, in radians.
///
/// Returns:
///     True when the two agree within ``ANGLE_TOLERANCE``. A NaN or an
///     infinity agrees with nothing, including itself.
#[pyfunction]
#[pyo3(signature = (left, right))]
fn angles_close(left: f64, right: f64) -> bool {
    numeric::angles_close(left, right)
}

/// Return *angle* wrapped into ``[-pi, pi)``.
///
/// Args:
///     angle: Angle in radians.
///
/// Returns:
///     The representative of *angle* inside ``[-pi, pi)``.
///
/// Raises:
///     ValueError: If *angle* is NaN or infinite, neither of which has a
///         representative in that interval.
#[pyfunction]
#[pyo3(signature = (angle))]
fn wrap_angle(angle: f64) -> PyResult<f64> {
    numeric::wrap_angle(angle).or_raise()
}

/// Return the signed difference ``left - right`` wrapped into ``[-pi, pi)``.
///
/// Plain subtraction is wrong by a full turn across the branch cut, which
/// is why no control law in ARCO subtracts two angles directly.
///
/// Args:
///     left: First angle, in radians.
///     right: Second angle, in radians.
///
/// Returns:
///     The difference in radians, inside ``[-pi, pi)``.
///
/// Raises:
///     ValueError: If either angle is NaN or infinite.
#[pyfunction]
#[pyo3(signature = (left, right))]
fn angle_difference(left: f64, right: f64) -> PyResult<f64> {
    numeric::angle_difference(left, right).or_raise()
}

/// A PCG64 generator drawing the stream ``numpy.random.default_rng`` draws.
///
/// Seeding follows numpy's own procedure, so a planner seeded with an
/// integer here produces the values the Python implementation produced
/// from the same seed. That is what makes a seeded run reproducible across
/// the port rather than merely reproducible against itself.
///
/// Args:
///     seed: Integer seed, in ``[0, 2**64)``.
#[pyclass(name = "PCG64", module = "arco._arco")]
#[derive(Debug)]
pub struct PyPcg64 {
    inner: Pcg64,
}

#[pymethods]
impl PyPcg64 {
    #[new]
    #[pyo3(signature = (seed))]
    fn new(seed: u64) -> Self {
        Self {
            inner: Pcg64::seed_from_u64(seed),
        }
    }

    /// Return the next raw 64 bit draw.
    ///
    /// Returns:
    ///     An integer in ``[0, 2**64)``.
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Return the next double in ``[0, 1)``.
    ///
    /// The draw takes the top 53 bits of a raw value, which is the only
    /// conversion giving every representable double in the interval an
    /// equal chance.
    ///
    /// Returns:
    ///     A float in ``[0, 1)``.
    fn next_f64(&mut self) -> f64 {
        self.inner.next_f64()
    }

    /// The current 128 bit internal state.
    #[getter]
    fn state(&self) -> u128 {
        self.inner.state()
    }

    /// The 128 bit increment selecting this generator's stream.
    #[getter]
    fn increment(&self) -> u128 {
        self.inner.increment()
    }

    fn __repr__(&self) -> String {
        format!(
            "PCG64(state={}, increment={})",
            self.state(),
            self.increment()
        )
    }
}

/// Adds the `arco-core` surface to the compiled module.
///
/// # Arguments
///
/// * `module` - The module every ARCO binding registers into.
///
/// # Errors
///
/// Returns an error when a class, function or constant fails to register,
/// which the interpreter surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPose>()?;
    module.add_class::<PyPcg64>()?;

    module.add_function(wrap_pyfunction!(euclidean_distance, module)?)?;
    module.add_function(wrap_pyfunction!(manhattan_distance, module)?)?;
    module.add_function(wrap_pyfunction!(points_close, module)?)?;
    module.add_function(wrap_pyfunction!(is_close, module)?)?;
    module.add_function(wrap_pyfunction!(positions_close, module)?)?;
    module.add_function(wrap_pyfunction!(angles_close, module)?)?;
    module.add_function(wrap_pyfunction!(wrap_angle, module)?)?;
    module.add_function(wrap_pyfunction!(angle_difference, module)?)?;

    // Named constants in meters, radians and seconds, per FR-SAFE-06. A
    // caller reaching for a tolerance takes one of these rather than
    // writing an epsilon of its own.
    module.add("POSITION_TOLERANCE", numeric::POSITION_TOLERANCE)?;
    module.add("ANGLE_TOLERANCE", numeric::ANGLE_TOLERANCE)?;
    module.add("TIME_TOLERANCE", numeric::TIME_TOLERANCE)?;
    module.add("RELATIVE_TOLERANCE", numeric::RELATIVE_TOLERANCE)?;
    Ok(())
}
