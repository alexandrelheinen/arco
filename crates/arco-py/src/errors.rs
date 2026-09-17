//! Turning an [`arco_core::Error`] into the exception Python already raised.
//!
//! `arco-core` names the exception rather than constructing it, so that
//! nothing below this layer depends on `PyO3`. This module is where the
//! name becomes an actual exception object, which is the whole of
//! `FR-API-04`: a caller that wrote `except ValueError` around a call into
//! ARCO keeps catching what it used to catch.
//!
//! The mapping itself lives in [`arco_core::Error::python_exception`] and
//! is pinned by `crates/arco-core/tests/error_mapping.rs`. Adding a
//! variant there without deciding its exception fails that test rather
//! than silently arriving here as a `RuntimeError`.

use arco_core::{Error, PythonException};
use pyo3::exceptions::{
    PyFileNotFoundError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;

/// Builds the Python exception an ARCO error surfaces as.
///
/// The message is the error's own `Display` text, which names the quantity
/// and the offending value rather than the function that produced it.
///
/// # Arguments
///
/// * `error` - The failure to report.
///
/// # Returns
///
/// An exception of the type the Python implementation raised for this
/// cause.
#[must_use]
pub(crate) fn to_exception(error: &Error) -> PyErr {
    let message = error.to_string();
    match error.python_exception() {
        PythonException::Value => PyValueError::new_err(message),
        PythonException::Type => PyTypeError::new_err(message),
        PythonException::Key => PyKeyError::new_err(message),
        PythonException::FileNotFound => PyFileNotFoundError::new_err(message),
        // Covers `PythonException::Runtime` and, because that enum is
        // `#[non_exhaustive]`, any variant added upstream. A failure
        // nobody has classified is not attributable to one argument,
        // which is what RuntimeError means.
        _ => PyRuntimeError::new_err(message),
    }
}

/// Carrying an ARCO result across the boundary as a Python result.
///
/// Implemented for every fallible ARCO call so that a binding reads
/// `value.or_raise()?` instead of repeating the closure at each call site.
pub(crate) trait OrRaise<T> {
    /// Converts the error side into the exception it names.
    ///
    /// # Errors
    ///
    /// Returns the exception [`to_exception`] builds for the error.
    fn or_raise(self) -> PyResult<T>;
}

impl<T> OrRaise<T> for Result<T, Error> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(|error| to_exception(&error))
    }
}
