//! Reading the packaged YAML configuration from the binding layer.
//!
//! `arco.config` stays Python because it is a loader over files that ship
//! with the package rather than an algorithm. A `create_from_config` in
//! this crate reaches a loaded mapping through the readers below.
//!
//! The readers coerce rather than extract, per deviation A-28. `PyYAML`
//! follows the 1.1 spec, where a float's exponent needs a sign, so
//! `1.0e2` in a shipped file parses as the string `"1.0e2"`. The Python
//! originals wrapped every read in `float(...)` or `int(...)` and never
//! noticed; a binding that extracted instead died with `TypeError` on the
//! repository's own configuration.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};

/// Reads a required section, naming the key when it is absent.
///
/// # Errors
///
/// Returns a `ValueError` naming the missing key, which is what the
/// Python constructors raised.
pub(crate) fn required<'py>(config: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    let section = config.call_method1("get", (key, config.py().None()))?;
    if section.is_none() {
        return Err(PyValueError::new_err(format!(
            "Config file must specify '{key}' key."
        )));
    }
    Ok(section)
}

/// Reads a number out of a section, or its default.
///
/// # Errors
///
/// Returns whatever reading the key raised.
pub(crate) fn number(section: &Bound<'_, PyAny>, key: &str, fallback: f64) -> PyResult<f64> {
    let read = section.call_method1("get", (key, fallback))?;
    read.call_method0("__float__")
        .or_else(|_not_a_number| read.py().get_type::<PyFloat>().call1((read,)))?
        .extract::<f64>()
}

/// Reads a count out of a section, or its default.
///
/// # Errors
///
/// Returns whatever reading the key raised.
pub(crate) fn count(section: &Bound<'_, PyAny>, key: &str, fallback: usize) -> PyResult<usize> {
    let read = section.call_method1("get", (key, fallback))?;
    read.py()
        .get_type::<PyInt>()
        .call1((read,))?
        .extract::<usize>()
}

/// Reads a string out of a section, or its default.
///
/// # Errors
///
/// Returns whatever reading the key raised.
pub(crate) fn text(section: &Bound<'_, PyAny>, key: &str, fallback: &str) -> PyResult<String> {
    let read = section.call_method1("get", (key, fallback))?;
    read.str()?.extract::<String>()
}
