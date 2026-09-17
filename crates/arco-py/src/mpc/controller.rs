//! `MPCController`, the scalar stub the path-following controller replaced.

use pyo3::PyClassInitializer;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::control::PyController;

/// What the deprecation warning says, verbatim as Python said it.
const DEPRECATION: &str = "MPCController is deprecated; use DubinsPathFollowingMPC \
                           from arco.control.mpc for path-following MPC.";

/// Deprecated scalar Model Predictive Controller stub.
///
/// ``.. deprecated:: 0.4.0``
///     Use :class:`~arco.control.mpc.path_following.DubinsPathFollowingMPC`
///     for multi-state path-following MPC instead.
///
/// Args:
///     `horizon`: Prediction horizon (number of steps).
///     `dt`: Time step duration in seconds.
#[pyclass(extends = PyController, subclass, name = "MPCController", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyMpcController {
    /// Prediction horizon, steps. Carried, never read.
    #[pyo3(get, set)]
    horizon: i64,
    /// Step duration, seconds. Carried, never read.
    #[pyo3(get, set)]
    dt: f64,
}

#[pymethods]
impl PyMpcController {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Initialize `MPCController`.
    #[new]
    #[pyo3(signature = (horizon = 10, dt = 0.1))]
    #[pyo3(text_signature = "(horizon=10, dt=0.1)")]
    fn new(py: Python<'_>, horizon: i64, dt: f64) -> PyResult<PyClassInitializer<Self>> {
        // Stack level two, so the warning points at the caller's
        // construction rather than at this constructor, which is what the
        // Python `stacklevel=2` did.
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            std::ffi::CString::new(DEPRECATION)?.as_c_str(),
            2,
        )?;
        Ok(PyClassInitializer::from(PyController).add_subclass(Self { horizon, dt }))
    }

    /// Compute MPC control output (stub).
    ///
    /// Args:
    ///     `state`: The current state value.
    ///     `reference`: The reference/target value.
    ///
    /// Returns:
    ///     Control command as a float, always zero.
    #[pyo3(signature = (state, reference))]
    #[pyo3(text_signature = "(state, reference)")]
    #[expect(
        clippy::unused_self,
        reason = "the stub reports nothing about its own state"
    )]
    const fn control(&self, state: f64, reference: f64) -> f64 {
        let _ = (state, reference);
        0.0
    }
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMpcController>()
}
