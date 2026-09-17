//! `ReferencePath`, the polyline both predictive controllers track.

use arco_control::mpc::reference::{CURVATURE_CEILING, ReferencePath};
use arco_core::Error;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::errors::{OrRaise, to_exception};
use crate::hooks::point_rows;

/// Arc-length parameterized polyline through ordered waypoints.
///
/// Zero-length segments are dropped on the way in, so a path built from a
/// list carrying a repeated point reports fewer waypoints than it was
/// given. Every query clamps its arc length into ``[0, total_length]``
/// rather than refusing it, which is what lets a controller ask about a
/// horizon that runs off the end of the route.
///
/// Args:
///     `waypoints`: Ordered ``(x, y)`` points, at least two of them
///         distinct.
///
/// Raises:
///     `ValueError`: If fewer than two waypoints are given, a coordinate
///         is not a real number, or every segment is degenerate.
#[pyclass(subclass, name = "ReferencePath", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyReferencePath {
    /// The path this class is a face for.
    inner: ReferencePath,
}

/// Reads a sequence of planar waypoints, as the Python constructor did.
///
/// # Errors
///
/// Returns a `ValueError` naming the offending row when one is not a
/// pair, which is the exception the numpy shape check raised.
fn waypoint_pairs(waypoints: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    let rows = point_rows(waypoints)?;
    rows.iter()
        .map(|row| match row.as_slice() {
            [x, y] => Ok((*x, *y)),
            other => Err(to_exception(&Error::DimensionMismatch {
                quantity: "waypoint",
                expected: 2,
                actual: other.len(),
            })),
        })
        .collect()
}

#[pymethods]
impl PyReferencePath {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(
        &self,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) {
    }

    /// Build a reference path from ordered waypoints.
    #[new]
    #[pyo3(signature = (waypoints))]
    #[pyo3(text_signature = "(waypoints)")]
    fn new(waypoints: &Bound<'_, PyAny>) -> PyResult<Self> {
        let points = waypoint_pairs(waypoints)?;
        Ok(Self {
            inner: ReferencePath::new(&points).or_raise()?,
        })
    }

    /// Largest curvature magnitude the profile reports, per meter.
    ///
    /// A polyline corner is a discontinuity, so the finite-difference
    /// estimate at one is bounded rather than reported as it falls out.
    #[classattr]
    const _CURVATURE_ABS_MAX: f64 = CURVATURE_CEILING;

    /// Total arc length of the path (m).
    #[getter]
    fn total_length(&self) -> f64 {
        self.inner.total_length()
    }

    /// Number of waypoints after zero-length filtering.
    #[getter]
    fn waypoint_count(&self) -> usize {
        self.inner.waypoint_count()
    }

    /// Return the reference position at arc length *s*.
    ///
    /// Args:
    ///     `s`: Arc length along the path (m).
    ///
    /// Returns:
    ///     ``(x_ref, y_ref)`` in world frame.
    #[pyo3(signature = (s))]
    #[pyo3(text_signature = "(s)")]
    fn position(&self, s: f64) -> (f64, f64) {
        let sample = self.inner.sample_at(s);
        (sample.x, sample.y)
    }

    /// Return the unit tangent at arc length *s*.
    ///
    /// Args:
    ///     `s`: Arc length along the path (m).
    ///
    /// Returns:
    ///     ``(cos heading, sin heading)``.
    #[pyo3(signature = (s))]
    #[pyo3(text_signature = "(s)")]
    fn tangent(&self, s: f64) -> (f64, f64) {
        let (sine, cosine) = self.inner.sample_at(s).heading.sin_cos();
        (cosine, sine)
    }

    /// Return the reference heading at arc length *s*.
    ///
    /// Args:
    ///     `s`: Arc length along the path (m).
    ///
    /// Returns:
    ///     Heading in radians.
    #[pyo3(signature = (s))]
    #[pyo3(text_signature = "(s)")]
    fn heading(&self, s: f64) -> f64 {
        self.inner.sample_at(s).heading
    }

    /// Return an approximate curvature at arc length *s*.
    ///
    /// Args:
    ///     `s`: Arc length along the path (m).
    ///
    /// Returns:
    ///     Curvature in 1/m, a finite-difference estimate interpolated
    ///     between vertices.
    #[pyo3(signature = (s))]
    #[pyo3(text_signature = "(s)")]
    fn curvature(&self, s: f64) -> f64 {
        self.inner.curvature(s)
    }

    /// Sample path quantities on a uniform arc-length grid.
    ///
    /// Args:
    ///     `sample_count`: Number of samples, at least 2. A smaller
    ///         request is raised to 2, since one sample does not describe
    ///         a path.
    ///
    /// Returns:
    ///     Tuple ``(s, x, y, heading, curvature)``, each of shape
    ///     ``(sample_count,)``.
    #[pyo3(signature = (sample_count))]
    #[pyo3(text_signature = "(sample_count)")]
    fn sample<'py>(&self, py: Python<'py>, sample_count: i64) -> PyResult<Bound<'py, PyTuple>> {
        let count = usize::try_from(sample_count).unwrap_or(2).max(2);
        let total = self.inner.total_length();
        // The grid divides the path into `count - 1` intervals, and
        // `count` is at least two, so the divisor is never zero.
        let spacing = total / f64::from(u32::try_from(count.saturating_sub(1)).unwrap_or(u32::MAX));

        let mut arc_lengths = Vec::with_capacity(count);
        let mut xs = Vec::with_capacity(count);
        let mut ys = Vec::with_capacity(count);
        let mut headings = Vec::with_capacity(count);
        let mut curvatures = Vec::with_capacity(count);
        for index in 0..count {
            let arc_length = if index.saturating_add(1) == count {
                // The last sample lands on the end exactly, rather than
                // wherever the accumulated spacing put it.
                total
            } else {
                spacing * f64::from(u32::try_from(index).unwrap_or(u32::MAX))
            };
            let sample = self.inner.sample_at(arc_length);
            arc_lengths.push(arc_length);
            xs.push(sample.x);
            ys.push(sample.y);
            headings.push(sample.heading);
            curvatures.push(sample.curvature);
        }

        PyTuple::new(
            py,
            [arc_lengths, xs, ys, headings, curvatures]
                .map(|values| PyArray1::from_vec(py, values)),
        )
    }

    /// Project a pose onto the nearest path point.
    ///
    /// Args:
    ///     `pose`: Vehicle pose ``(x, y, heading)``.
    ///     `s_hint`: Optional arc-length center for a local search window.
    ///         Used with *window* so contouring progress cannot flip to a
    ///         distant junction-scale nearest segment after a corner cut.
    ///     `window`: Half-width (m) of the local search around *s_hint*.
    ///         Ignored unless *s_hint* is also given. ``None`` or a
    ///         non-positive value keeps the global nearest-point search.
    ///
    /// Returns:
    ///     ``(s, lateral_error, heading_error)`` where *lateral_error* is
    ///     signed, left positive, and *heading_error* is wrapped.
    ///
    /// Raises:
    ///     `ValueError`: If a pose component is not a real number.
    #[pyo3(signature = (pose, *, s_hint = None, window = None))]
    #[pyo3(text_signature = "(pose, *, s_hint=None, window=None)")]
    fn project(
        &self,
        pose: (f64, f64, f64),
        s_hint: Option<f64>,
        window: Option<f64>,
    ) -> PyResult<(f64, f64, f64)> {
        // Both halves of the window have to be present for a local
        // search, and the half-width has to be positive, which is the
        // condition the Python spelled out the same way.
        let bounds = match (s_hint, window) {
            (Some(hint), Some(half_width)) if half_width > 0.0 => Some((hint, half_width)),
            _ => None,
        };
        let projection = self.inner.project(pose, bounds).or_raise()?;
        Ok((
            projection.arc_length,
            projection.lateral_error,
            projection.heading_error,
        ))
    }
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyReferencePath>()
}
