//! `arco.control.mpc` as Python sees it.
//!
//! The predictive controllers are the one part of the control layer whose
//! formulation changed rather than its language, so this module carries
//! more deviation than the rest of the binding does. ADR-002 replaced the
//! nonlinear program `CasADi` built and IPOPT solved with a sequence of
//! convex programs Clarabel solves. A-30 records that the obstacle
//! barrier became a supporting hyperplane with a penalized slack, which
//! is why `obstacle_barrier_power` is accepted and no longer shapes the
//! barrier. A-31 records that the reported cost is the surrogate
//! objective. A-32 records the solver-status vocabulary.
//!
//! The configuration classes are compiled like the rest: they validate
//! what they carry, and `create_from_config` reads the packaged YAML
//! through [`crate::config`], which is the one part of `arco.config` that
//! stays Python because it loads files rather than computing anything.

mod controller;
mod costs;
mod joint;
mod path;
mod reference;
mod result;
mod tracking;

use pyo3::prelude::*;

/// Reads a number off a configuration object.
///
/// The two configuration records stay Python dataclasses, because they
/// carry no algorithm and because `dataclasses.replace` is part of the
/// surface a caller uses: `arco.simulator` replaces the cruise speed on
/// one that way. The controllers read them by attribute here, which also
/// accepts whatever object a caller substitutes.
///
/// # Errors
///
/// Returns whatever the attribute lookup raised, and a `TypeError` when
/// the value is not a number.
pub(crate) fn attribute_number(object: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    object.getattr(name)?.extract::<f64>()
}

/// Reads a count off a configuration object.
///
/// # Errors
///
/// As [`attribute_number`].
pub(crate) fn attribute_count(object: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    object.getattr(name)?.extract::<usize>()
}

/// Adds every MPC name to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    costs::register(module)?;
    reference::register(module)?;
    result::register(module)?;
    joint::register(module)?;
    path::register(module)?;
    controller::register(module)?;
    tracking::register(module)?;
    Ok(())
}
