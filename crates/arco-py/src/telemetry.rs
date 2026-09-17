//! `arco.planning.continuous.telemetry`, the planner's progress snapshot.
//!
//! A planner writes one of these every few hundred iterations and a
//! loading screen in another process reads it back. Both sides are
//! compiled here so that a search does not have to reach into the
//! interpreter to describe its own progress: the snapshot a planner hands
//! its publisher is built natively, and writing it to the shared file
//! never runs Python at all.

use std::fs;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// A single named stop criterion with its current and threshold values.
///
/// Args:
///     `name`: Human-readable criterion label, such as ``"iterations"``.
///     `current`: Current measured value.
///     `threshold`: Target threshold value.
///     `condition`: Comparison operator, one of ``"<"``, ``"<="``,
///         ``"\u{2264}"``, ``">"``, ``">="``, ``"\u{2265}"``, ``"="`` or
///         ``"=="``.
#[pyclass(
    subclass,
    from_py_object,
    name = "StopCriterion",
    module = "arco._arco"
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyStopCriterion {
    /// What the criterion is called.
    #[pyo3(get, set)]
    name: String,
    /// What it measures right now.
    #[pyo3(get, set)]
    current: f64,
    /// What it is aiming at.
    #[pyo3(get, set)]
    threshold: f64,
    /// How the two are compared.
    #[pyo3(get, set)]
    condition: String,
}

#[pymethods]
impl PyStopCriterion {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a criterion, as the dataclass constructor did.
    #[new]
    #[pyo3(signature = (name, current, threshold, condition))]
    #[pyo3(text_signature = "(name, current, threshold, condition)")]
    const fn new(name: String, current: f64, threshold: f64, condition: String) -> Self {
        Self {
            name,
            current,
            threshold,
            condition,
        }
    }

    /// Return True if the criterion is currently met.
    ///
    /// An operator this does not recognize reports False rather than
    /// raising, which is what the Python did: a drawing routine asking
    /// whether to color a row green is not the place to fail a run.
    ///
    /// Returns:
    ///     True when ``current <operator> threshold`` holds.
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "()")]
    fn satisfied(&self) -> bool {
        match self.condition.as_str() {
            "<" => self.current < self.threshold,
            "<=" | "\u{2264}" => self.current <= self.threshold,
            ">" => self.current > self.threshold,
            ">=" | "\u{2265}" => self.current >= self.threshold,
            // The Python compared with `==`, and the lint that bans a
            // float equality test is right in general and wrong here:
            // the caller asked for that comparison by name.
            "=" | "==" => {
                #[expect(
                    clippy::float_cmp,
                    reason = "the caller named this comparison, and the Python did the same"
                )]
                let equal = self.current == self.threshold;
                equal
            }
            _unknown => false,
        }
    }

    /// Whether two criteria carry the same values.
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    /// The dataclass representation, field by field.
    fn __repr__(&self) -> String {
        format!(
            "StopCriterion(name={:?}, current={}, threshold={}, condition={:?})",
            self.name, self.current, self.threshold, self.condition
        )
    }
}

/// A telemetry snapshot written by a planner at regular intervals.
///
/// Args:
///     `algorithm`: Name of the planning algorithm, such as ``"RRT*"``.
///     `step_name`: What the planner is doing at the moment.
///     `iteration`: Current iteration index.
///     `max_iterations`: Iteration budget.
///     `best_dist_to_goal`: Smallest distance to the goal seen so far.
///     `criteria`: Stop criteria with their current values.
#[pyclass(
    subclass,
    from_py_object,
    name = "PlannerTelemetry",
    module = "arco._arco"
)]
#[derive(Debug, Clone)]
pub(crate) struct PyPlannerTelemetry {
    /// Which algorithm is running.
    #[pyo3(get, set)]
    algorithm: String,
    /// What it is doing now.
    #[pyo3(get, set)]
    step_name: String,
    /// How far through it is.
    #[pyo3(get, set)]
    iteration: i64,
    /// How far it may go.
    #[pyo3(get, set)]
    max_iterations: i64,
    /// The closest approach to the goal so far.
    #[pyo3(get, set)]
    best_dist_to_goal: f64,
    /// What would make it stop.
    #[pyo3(get, set)]
    criteria: Vec<PyStopCriterion>,
}

impl PyPlannerTelemetry {
    /// The snapshot as the JSON object the file carries.
    fn as_json(&self) -> serde_json::Value {
        let criteria: Vec<serde_json::Value> = self
            .criteria
            .iter()
            .map(|criterion| {
                serde_json::json!({
                    "name": criterion.name,
                    "current": criterion.current,
                    "threshold": criterion.threshold,
                    "condition": criterion.condition,
                })
            })
            .collect();
        serde_json::json!({
            "algorithm": self.algorithm,
            "step_name": self.step_name,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "best_dist_to_goal": finite_or_null(self.best_dist_to_goal),
            "criteria": criteria,
        })
    }

    /// Reads a snapshot back out of the JSON object the file carries.
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        let criteria = value
            .get("criteria")
            .and_then(serde_json::Value::as_array)
            .map(|listed| {
                listed
                    .iter()
                    .filter_map(|entry| {
                        Some(PyStopCriterion {
                            name: entry.get("name")?.as_str()?.to_owned(),
                            current: entry.get("current")?.as_f64()?,
                            threshold: entry.get("threshold")?.as_f64()?,
                            condition: entry.get("condition")?.as_str()?.to_owned(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Some(Self {
            algorithm: value.get("algorithm")?.as_str()?.to_owned(),
            step_name: value.get("step_name")?.as_str()?.to_owned(),
            iteration: value.get("iteration")?.as_i64()?,
            max_iterations: value.get("max_iterations")?.as_i64()?,
            // A planner that has placed no node yet reports an infinite
            // distance, which JSON cannot carry: `serde_json` refuses it
            // and writes null. Reading null back as infinity is what
            // keeps the round trip whole, per deviation A-38.
            best_dist_to_goal: value
                .get("best_dist_to_goal")
                .map_or(f64::INFINITY, |found| {
                    found.as_f64().unwrap_or(f64::INFINITY)
                }),
            criteria,
        })
    }
}

#[pymethods]
impl PyPlannerTelemetry {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a snapshot, as the dataclass constructor did.
    #[new]
    #[pyo3(signature = (
        algorithm,
        step_name,
        iteration,
        max_iterations,
        best_dist_to_goal,
        criteria = None,
    ))]
    #[pyo3(
        text_signature = "(algorithm, step_name, iteration, max_iterations, best_dist_to_goal, criteria=...)"
    )]
    fn new(
        algorithm: String,
        step_name: String,
        iteration: i64,
        max_iterations: i64,
        best_dist_to_goal: f64,
        criteria: Option<Vec<PyStopCriterion>>,
    ) -> Self {
        Self {
            algorithm,
            step_name,
            iteration,
            max_iterations,
            best_dist_to_goal,
            criteria: criteria.unwrap_or_default(),
        }
    }

    /// The dataclass representation, field by field.
    fn __repr__(&self) -> String {
        format!(
            "PlannerTelemetry(algorithm={:?}, step_name={:?}, iteration={}, \
             max_iterations={}, best_dist_to_goal={}, criteria={:?})",
            self.algorithm,
            self.step_name,
            self.iteration,
            self.max_iterations,
            self.best_dist_to_goal,
            self.criteria,
        )
    }
}

/// The value JSON can carry, or null when the number is not finite.
///
/// A planner reports an infinite distance to the goal until it has placed
/// a node that reaches one, and neither JSON nor `serde_json` can write
/// an infinity. Null is what the file carries, and the reader turns it
/// back into an infinity. Deviation A-38.
fn finite_or_null(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::json!(value)
    } else {
        serde_json::Value::Null
    }
}

/// Where a planner and a loading screen agree to meet.
fn default_path() -> PathBuf {
    std::env::temp_dir().join("arco_planner_telemetry.json")
}

/// Resolves the path argument, which defaults to the shared file.
///
/// # Errors
///
/// Returns whatever converting the argument to a string raised.
fn resolve(path: Option<&Bound<'_, PyAny>>) -> PyResult<PathBuf> {
    match path {
        None => Ok(default_path()),
        Some(given) => Ok(PathBuf::from(given.str()?.to_string_lossy().into_owned())),
    }
}

/// Write *telemetry* as JSON to *path*, atomically.
///
/// The write goes to a sibling ``.tmp`` file that is then renamed over the
/// target, so a reader never sees half a snapshot. Every I/O error is
/// swallowed, because a loading screen that has gone away is not a
/// planning failure.
///
/// Args:
///     `telemetry`: The snapshot to persist.
///     `path`: Destination file. Defaults to
///         :data:`DEFAULT_TELEMETRY_PATH`.
#[pyfunction]
#[pyo3(signature = (telemetry, path = None))]
#[pyo3(text_signature = "(telemetry, path=None)")]
fn write_telemetry(
    telemetry: &PyPlannerTelemetry,
    path: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    write_json(telemetry, &resolve(path)?);
    Ok(())
}

/// Writes one snapshot to `target`, atomically, swallowing every failure.
///
/// Both failures are ignored on purpose, per the docstring `write_telemetry`
/// carries: the snapshot is advisory and the planner owns no part of the
/// reader.
fn write_json(telemetry: &PyPlannerTelemetry, target: &std::path::Path) {
    let temporary = target.with_extension("tmp");
    let encoded = serde_json::to_string(&telemetry.as_json()).unwrap_or_default();
    if fs::write(&temporary, encoded).is_ok() {
        drop(fs::rename(&temporary, target));
    }
}

/// Read and parse the telemetry snapshot at *path*.
///
/// Args:
///     `path`: Source file. Defaults to :data:`DEFAULT_TELEMETRY_PATH`.
///
/// Returns:
///     A :class:`PlannerTelemetry`, or ``None`` when the file is absent,
///     unreadable, or not a snapshot this wrote.
#[pyfunction]
#[pyo3(signature = (path = None))]
#[pyo3(text_signature = "(path=None)")]
fn read_telemetry(path: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PyPlannerTelemetry>> {
    let target = resolve(path)?;
    let Ok(text) = fs::read_to_string(target) else {
        return Ok(None);
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Ok(None);
    };
    Ok(PyPlannerTelemetry::from_json(&parsed))
}

/// Discard a telemetry snapshot, which disables the file exchange.
///
/// Args:
///     `telemetry`: Ignored snapshot.
#[pyfunction]
#[pyo3(signature = (telemetry))]
#[pyo3(text_signature = "(telemetry)")]
fn noop_publisher(telemetry: &PyPlannerTelemetry) {
    let _ = telemetry;
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStopCriterion>()?;
    module.add_class::<PyPlannerTelemetry>()?;
    module.add_function(wrap_pyfunction!(write_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(read_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(noop_publisher, module)?)?;
    let pathlib = module.py().import("pathlib")?.getattr("Path")?;
    module.add(
        "DEFAULT_TELEMETRY_PATH",
        pathlib.call1((default_path().to_string_lossy().into_owned(),))?,
    )
}

/// Builds a snapshot without touching the interpreter.
///
/// A planner reports progress from inside its search, with the lock
/// released, so the snapshot has to exist as a Rust value first and
/// become a Python object only if somebody is waiting for one.
pub(crate) fn snapshot(
    algorithm: &str,
    step_name: &str,
    iteration: usize,
    max_iterations: usize,
    best_dist_to_goal: f64,
) -> PyPlannerTelemetry {
    PyPlannerTelemetry {
        algorithm: algorithm.to_owned(),
        step_name: step_name.to_owned(),
        iteration: i64::try_from(iteration).unwrap_or(i64::MAX),
        max_iterations: i64::try_from(max_iterations).unwrap_or(i64::MAX),
        best_dist_to_goal,
        criteria: Vec::new(),
    }
}

/// Writes a snapshot to the shared file, as `write_telemetry` does.
///
/// Every I/O failure is swallowed, for the reason [`write_telemetry`]
/// gives: the snapshot is advisory and the planner owns no part of the
/// reader.
pub(crate) fn publish_to_file(telemetry: &PyPlannerTelemetry) {
    write_json(telemetry, &default_path());
}

/// Writes a snapshot a caller handed in as a Python object.
///
/// The publishing path a planner subclass reaches through
/// `publish_telemetry` carries whatever object that caller built, which
/// may be a `PlannerTelemetry` or may be something shaped like one.
///
/// # Errors
///
/// Returns whatever reading a field off the object raised.
pub(crate) fn publish_bound(telemetry: &Bound<'_, PyAny>) -> PyResult<()> {
    let extracted = match telemetry.extract::<PyPlannerTelemetry>() {
        Ok(native) => native,
        Err(_not_native) => PyPlannerTelemetry {
            algorithm: telemetry.getattr("algorithm")?.str()?.extract::<String>()?,
            step_name: telemetry.getattr("step_name")?.str()?.extract::<String>()?,
            iteration: telemetry.getattr("iteration")?.extract::<i64>()?,
            max_iterations: telemetry.getattr("max_iterations")?.extract::<i64>()?,
            best_dist_to_goal: telemetry.getattr("best_dist_to_goal")?.extract::<f64>()?,
            criteria: telemetry
                .getattr("criteria")?
                .extract::<Vec<PyStopCriterion>>()
                .unwrap_or_default(),
        },
    };
    publish_to_file(&extracted);
    Ok(())
}
