//! `arco.planning.PlanningPipeline`, and the snapshot it returns.
//!
//! The pipeline sequences three objects a caller injected, times each of
//! them, and records what came back. Every stage is a call into the
//! interpreter, so the loop holds the lock throughout rather than
//! releasing it three times for nothing.
//!
//! The two serialization helpers reach numpy rather than writing the
//! archive here. `.npz` is a zip of `.npy` members, and a reimplementation
//! would be a second definition of a format a caller reads back with
//! `numpy.load`. Calling the library that owns the format keeps the two
//! sides from drifting.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};


/// Snapshot of every stage output from one `PlanningPipeline.run` call.
///
/// Stores the outputs from all stages so that a caller can inspect or
/// replay any intermediate result without running the pipeline again.
///
/// Args:
///     `raw_path`: Unprocessed path from the planner, or ``None`` when
///         planning failed.
///     `pruned_path`: Path after pruning. ``None`` when pruning was
///         skipped or planning failed.
///     `trajectory`: Time-parameterized states from the optimizer.
///         ``None`` when optimization was skipped or failed.
///     `durations`: Per-segment durations (seconds), or ``None``.
///     `total_duration`: Sum of the per-segment durations (seconds).
///     `planner_time`: Wall-clock time spent in the planner (seconds).
///     `pruner_time`: Wall-clock time spent in the pruner (seconds).
///     `optimizer_time`: Wall-clock time spent in the optimizer (seconds).
///     `planner_status`: ``'success'``, ``'no_path'``, ``'pre_planned'``
///         or ``'not_run'``.
///     `optimizer_status`: Optimizer status text, ``'skipped'`` or
///         ``'not_run'``.
///     `optimizer_success`: ``True`` when the optimizer converged.
///     `extra`: Arbitrary extra metadata, such as node counts or costs.
#[pyclass(subclass, name = "PipelineResult", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyPipelineResult {
    /// Unprocessed path from the planner.
    #[pyo3(get, set)]
    raw_path: Option<Py<PyAny>>,
    /// Path after pruning.
    #[pyo3(get, set)]
    pruned_path: Option<Py<PyAny>>,
    /// States the optimizer produced.
    #[pyo3(get, set)]
    trajectory: Option<Py<PyAny>>,
    /// Per-segment durations, seconds.
    #[pyo3(get, set)]
    durations: Option<Py<PyAny>>,
    /// Sum of the per-segment durations, seconds.
    #[pyo3(get, set)]
    total_duration: f64,
    /// Wall-clock time spent planning, seconds.
    #[pyo3(get, set)]
    planner_time: f64,
    /// Wall-clock time spent pruning, seconds.
    #[pyo3(get, set)]
    pruner_time: f64,
    /// Wall-clock time spent optimizing, seconds.
    #[pyo3(get, set)]
    optimizer_time: f64,
    /// How planning ended.
    #[pyo3(get, set)]
    planner_status: String,
    /// How optimization ended.
    #[pyo3(get, set)]
    optimizer_status: String,
    /// Whether the optimizer converged.
    #[pyo3(get, set)]
    optimizer_success: bool,
    /// Whatever else a caller attached.
    #[pyo3(get, set)]
    extra: Py<PyDict>,
}

impl PyPipelineResult {
    /// An empty result, as the dataclass defaults describe one.
    fn empty(py: Python<'_>) -> Self {
        Self {
            raw_path: None,
            pruned_path: None,
            trajectory: None,
            durations: None,
            total_duration: 0.0,
            planner_time: 0.0,
            pruner_time: 0.0,
            optimizer_time: 0.0,
            planner_status: "not_run".to_owned(),
            optimizer_status: "not_run".to_owned(),
            optimizer_success: false,
            extra: PyDict::new(py).unbind(),
        }
    }
}

#[pymethods]
impl PyPipelineResult {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a result, as the dataclass constructor did.
    #[new]
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python dataclass carries these twelve fields and the binding mirrors it"
    )]
    #[pyo3(signature = (
        raw_path = None,
        pruned_path = None,
        trajectory = None,
        durations = None,
        total_duration = 0.0,
        planner_time = 0.0,
        pruner_time = 0.0,
        optimizer_time = 0.0,
        planner_status = "not_run".to_owned(),
        optimizer_status = "not_run".to_owned(),
        optimizer_success = false,
        extra = None,
    ))]
    #[pyo3(
        text_signature = "(raw_path=None, pruned_path=None, trajectory=None, durations=None, total_duration=0.0, planner_time=0.0, pruner_time=0.0, optimizer_time=0.0, planner_status='not_run', optimizer_status='not_run', optimizer_success=False, extra=...)"
    )]
    fn new(
        py: Python<'_>,
        raw_path: Option<Py<PyAny>>,
        pruned_path: Option<Py<PyAny>>,
        trajectory: Option<Py<PyAny>>,
        durations: Option<Py<PyAny>>,
        total_duration: f64,
        planner_time: f64,
        pruner_time: f64,
        optimizer_time: f64,
        planner_status: String,
        optimizer_status: String,
        optimizer_success: bool,
        extra: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            raw_path,
            pruned_path,
            trajectory,
            durations,
            total_duration,
            planner_time,
            pruner_time,
            optimizer_time,
            planner_status,
            optimizer_status,
            optimizer_success,
            extra: extra.unwrap_or_else(|| PyDict::new(py).unbind()),
        }
    }
}

/// Algorithm-agnostic orchestrator for the planning pipeline.
///
/// Connects three injected stages, planner then pruner then optimizer, by
/// passing the output of each as the input to the next. It knows nothing
/// about the algorithms inside a stage, only about the contracts between
/// them, so a caller swaps any stage for its own object.
///
/// Args:
///     `planner`: Object with ``plan(start, goal)``. ``None`` leaves
///         :meth:`run` unusable and :meth:`run_from_path` available.
///     `pruner`: Object with ``prune(path)``, or ``None`` to skip that
///         stage.
///     `optimizer`: Object with ``optimize(path)``, or ``None`` to skip
///         that stage.
#[pyclass(subclass, name = "PlanningPipeline", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyPlanningPipeline {
    /// The planner, when one was given.
    #[pyo3(get, set)]
    planner: Option<Py<PyAny>>,
    /// The pruner, when one was given.
    #[pyo3(get, set)]
    pruner: Option<Py<PyAny>>,
    /// The optimizer, when one was given.
    #[pyo3(get, set)]
    optimizer: Option<Py<PyAny>>,
}

/// Reports one stage starting, when a caller asked to be told.
///
/// # Errors
///
/// Returns whatever the callback raised, which stops the run: a progress
/// sink that fails is a caller bug rather than a planning failure.
fn announce(
    progress: Option<&Bound<'_, PyAny>>,
    stage: &str,
    index: usize,
    total: usize,
) -> PyResult<()> {
    if let Some(sink) = progress {
        sink.call1((stage, index, total))?;
    }
    Ok(())
}

/// How long since `started`, in seconds.
///
/// FR-INV-10 keeps a clock out of a control decision, and this is not
/// one: the three stage timings are reported to the caller and read by
/// nothing. The Python measured them with `time.perf_counter`.
#[expect(
    clippy::disallowed_methods,
    reason = "a reported diagnostic, never an input to a decision"
)]
fn elapsed_since(started: std::time::Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

/// The moment a stage starts.
#[expect(
    clippy::disallowed_methods,
    reason = "a reported diagnostic, never an input to a decision"
)]
fn now() -> std::time::Instant {
    std::time::Instant::now()
}

/// Fills in the trajectory a caller gets when no optimizer ran.
///
/// The path itself becomes the trajectory and every segment is given one
/// second, which is what the Python did so that a caller downstream
/// always has a duration to divide by.
fn fall_back_to_path(
    py: Python<'_>,
    result: &mut PyPipelineResult,
    path: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let count = path.len()?;
    let segments = count.saturating_sub(1).max(1);
    result.trajectory = Some(path.clone().unbind());
    result.durations = Some(
        PyList::new(py, core::iter::repeat_n(1.0_f64, segments))?
            .into_any()
            .unbind(),
    );
    result.total_duration = f64::from(u32::try_from(segments).unwrap_or(u32::MAX));
    Ok(())
}

/// Runs the optimizer stage and records what it produced.
///
/// # Errors
///
/// Returns whatever the optimizer raised, or whatever reading one of its
/// result attributes raised.
fn optimize_into(
    py: Python<'_>,
    optimizer: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    result: &mut PyPipelineResult,
) -> PyResult<()> {
    let started = now();
    let answer = optimizer.call_method1("optimize", (path,))?;
    result.optimizer_time = elapsed_since(started);
    result.optimizer_success = answer.getattr("optimizer_success")?.extract::<bool>()?;
    let code = answer.getattr("optimizer_status_code")?.str()?;
    let text = answer.getattr("optimizer_status_text")?.str()?;
    result.optimizer_status = format!("{code}: {text}");

    let states = answer.getattr("states")?;
    if states.is_truthy()? {
        let durations = answer.getattr("durations")?;
        let total: f64 = durations.extract::<Vec<f64>>()?.iter().sum();
        result.trajectory = Some(py.get_type::<PyList>().call1((states,))?.unbind());
        result.durations = Some(py.get_type::<PyList>().call1((&durations,))?.unbind());
        result.total_duration = total;
    } else {
        // The optimizer answered with nothing, so the path it was given
        // stands as the trajectory rather than the run reporting none.
        fall_back_to_path(py, result, path)?;
    }
    Ok(())
}

#[pymethods]
impl PyPlanningPipeline {
    /// Absorbs a subclass calling `super().__init__(...)`.
    #[pyo3(signature = (*_args, **_kwargs))]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance and the body reads nothing"
    )]
    const fn __init__(&self, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) {}

    /// Build a pipeline over the stages a caller wants.
    #[new]
    #[pyo3(signature = (planner = None, pruner = None, optimizer = None))]
    #[pyo3(text_signature = "(planner=None, pruner=None, optimizer=None)")]
    const fn new(
        planner: Option<Py<PyAny>>,
        pruner: Option<Py<PyAny>>,
        optimizer: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            planner,
            pruner,
            optimizer,
        }
    }

    /// Run the full pipeline, from planning through optimization.
    ///
    /// Each configured stage runs in order and its wall-clock time is
    /// recorded. A planner that returns ``None``, or a path shorter than
    /// two points, skips the remaining stages and the result says so.
    ///
    /// Args:
    ///     `start`: Start configuration.
    ///     `goal`: Goal configuration.
    ///     `progress`: Optional callback, called with
    ///         ``(stage_name, stage_index, total_stages)`` at the start of
    ///         each stage.
    ///
    /// Returns:
    ///     A :class:`PipelineResult` with the output and timing of every
    ///     stage.
    ///
    /// Raises:
    ///     `RuntimeError`: If no planner was configured. Use
    ///         :meth:`run_from_path` when a path is already in hand.
    #[pyo3(signature = (start, goal, progress = None))]
    #[pyo3(text_signature = "(start, goal, progress=None)")]
    fn run(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyPipelineResult> {
        let Some(planner) = self.planner.as_ref() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "PlanningPipeline.run() requires a planner. \
                 Use run_from_path() when no planner is configured.",
            ));
        };
        let total = 1_usize
            .saturating_add(usize::from(self.pruner.is_some()))
            .saturating_add(usize::from(self.optimizer.is_some()));
        let mut stage = 0_usize;
        let mut result = PyPipelineResult::empty(py);

        stage = stage.saturating_add(1);
        announce(progress, "planning", stage, total)?;
        let started = now();
        let found = planner.bind(py).call_method1("plan", (start, goal))?;
        result.planner_time = elapsed_since(started);

        if found.is_none() || found.len().unwrap_or(0) < 2 {
            "no_path".clone_into(&mut result.planner_status);
            return Ok(result);
        }
        "success".clone_into(&mut result.planner_status);
        result.raw_path = Some(found.clone().unbind());
        let mut active = found;

        if let Some(pruner) = self.pruner.as_ref() {
            stage = stage.saturating_add(1);
            announce(progress, "pruning", stage, total)?;
            let started = now();
            let pruned = pruner.bind(py).call_method1("prune", (&active,))?;
            result.pruner_time = elapsed_since(started);
            result.pruned_path = Some(pruned.clone().unbind());
            active = pruned;
        }

        if let Some(optimizer) = self.optimizer.as_ref() {
            stage = stage.saturating_add(1);
            announce(progress, "optimization", stage, total)?;
            optimize_into(py, optimizer.bind(py), &active, &mut result)?;
        } else {
            fall_back_to_path(py, &mut result, &active)?;
            "skipped".clone_into(&mut result.optimizer_status);
        }
        Ok(result)
    }

    /// Run the pruner and optimizer on a path that is already planned.
    ///
    /// Useful when a caller has run its planner itself, to collect the
    /// search tree for a drawing, and wants only the later stages.
    ///
    /// Args:
    ///     `raw_path`: Ordered configurations, at least two of them.
    ///     `progress`: Optional callback, as :meth:`run` takes.
    ///
    /// Returns:
    ///     A :class:`PipelineResult` whose ``planner_status`` reads
    ///     ``'pre_planned'`` and whose ``planner_time`` is zero.
    #[pyo3(signature = (raw_path, progress = None))]
    #[pyo3(text_signature = "(raw_path, progress=None)")]
    fn run_from_path(
        &self,
        py: Python<'_>,
        raw_path: &Bound<'_, PyAny>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyPipelineResult> {
        let mut result = PyPipelineResult::empty(py);
        "pre_planned".clone_into(&mut result.planner_status);
        if raw_path.is_none() || raw_path.len().unwrap_or(0) < 2 {
            "no_path".clone_into(&mut result.planner_status);
            return Ok(result);
        }

        let listed = py.get_type::<PyList>().call1((raw_path,))?;
        result.raw_path = Some(listed.clone().unbind());
        let mut active = listed;

        let total =
            usize::from(self.pruner.is_some()).saturating_add(usize::from(self.optimizer.is_some()));
        let mut stage = 0_usize;

        if let Some(pruner) = self.pruner.as_ref() {
            stage = stage.saturating_add(1);
            announce(progress, "pruning", stage, total)?;
            let started = now();
            let pruned = pruner.bind(py).call_method1("prune", (&active,))?;
            result.pruner_time = elapsed_since(started);
            result.pruned_path = Some(pruned.clone().unbind());
            active = py.get_type::<PyList>().call1((pruned,))?;
        }

        if let Some(optimizer) = self.optimizer.as_ref() {
            stage = stage.saturating_add(1);
            announce(progress, "optimization", stage, total)?;
            optimize_into(py, optimizer.bind(py), &active, &mut result)?;
        } else {
            fall_back_to_path(py, &mut result, &active)?;
            "skipped".clone_into(&mut result.optimizer_status);
        }
        Ok(result)
    }

    /// Save a :class:`PipelineResult` to a compressed ``.npz`` file.
    ///
    /// Paths are stored as stacked arrays and the scalar metadata as one
    /// JSON string inside the archive. The ``.npz`` suffix is added when
    /// the path carries none.
    ///
    /// Args:
    ///     `result`: The result to serialize.
    ///     `path`: Destination file path.
    ///
    /// Raises:
    ///     Whatever numpy raised while writing the archive.
    #[staticmethod]
    #[pyo3(signature = (result, path))]
    #[pyo3(text_signature = "(result, path)")]
    fn save_result(py: Python<'_>, result: &Bound<'_, PyAny>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let numpy = py.import("numpy")?;
        let pathlib = py.import("pathlib")?.getattr("Path")?.call1((path,))?;
        pathlib
            .getattr("parent")?
            .call_method("mkdir", (), Some(&kwargs(py, &[("parents", true), ("exist_ok", true)])?))?;

        let arrays = PyDict::new(py);
        for key in ["raw_path", "pruned_path", "trajectory"] {
            let points = result.getattr(key)?;
            if !points.is_none() && points.len().unwrap_or(0) > 0 {
                let stacked = numpy.call_method(
                    "array",
                    (py.get_type::<PyList>().call1((&points,))?,),
                    Some(&float_dtype(py, &numpy)?),
                )?;
                arrays.set_item(key, stacked)?;
            } else {
                let empty = numpy.call_method(
                    "array",
                    (PyList::empty(py),),
                    Some(&float_dtype(py, &numpy)?),
                )?;
                arrays.set_item(format!("{key}__empty"), empty)?;
            }
        }

        let durations = result.getattr("durations")?;
        if !durations.is_none() {
            let values =
                numpy.call_method("array", (&durations,), Some(&float_dtype(py, &numpy)?))?;
            arrays.set_item("durations", values)?;
        }

        let meta = PyDict::new(py);
        for key in [
            "total_duration",
            "planner_time",
            "pruner_time",
            "optimizer_time",
            "planner_status",
            "optimizer_status",
            "optimizer_success",
            "extra",
        ] {
            meta.set_item(key, result.getattr(key)?)?;
        }
        let encoded = py.import("json")?.call_method1("dumps", (meta,))?;
        arrays.set_item("__meta__", numpy.call_method1("array", (encoded,))?)?;

        numpy.call_method(
            "savez_compressed",
            (pathlib.str()?,),
            Some(&arrays),
        )?;
        Ok(())
    }

    /// Load a :class:`PipelineResult` from a saved ``.npz`` file.
    ///
    /// Args:
    ///     `path`: Path to the archive. The ``.npz`` suffix is added when
    ///         the path carries none.
    ///
    /// Returns:
    ///     The reconstructed :class:`PipelineResult`.
    ///
    /// Raises:
    ///     `FileNotFoundError`: If the file does not exist.
    ///     `ValueError`: If the archive is not one this wrote.
    #[staticmethod]
    #[pyo3(signature = (path))]
    #[pyo3(text_signature = "(path)")]
    fn load_result(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<PyPipelineResult> {
        let numpy = py.import("numpy")?;
        let mut target = py.import("pathlib")?.getattr("Path")?.call1((path,))?;
        if target.getattr("suffix")?.str()?.to_string_lossy().is_empty() {
            target = target.call_method1("with_suffix", (".npz",))?;
        }
        let archive = numpy.call_method(
            "load",
            (target.str()?,),
            Some(&kwargs(py, &[("allow_pickle", false)])?),
        )?;

        let mut result = PyPipelineResult::empty(py);
        for key in ["raw_path", "pruned_path", "trajectory"] {
            if archive.contains(key)? {
                let stacked = archive.get_item(key)?;
                let rows = PyList::empty(py);
                for index in 0..stacked.len()? {
                    rows.append(stacked.get_item(index)?)?;
                }
                let held = Some(rows.into_any().unbind());
                match key {
                    "raw_path" => result.raw_path = held,
                    "pruned_path" => result.pruned_path = held,
                    _trajectory => result.trajectory = held,
                }
            }
        }
        if archive.contains("durations")? {
            result.durations = Some(archive.get_item("durations")?.call_method0("tolist")?.unbind());
        }
        if archive.contains("__meta__")? {
            let encoded = archive.get_item("__meta__")?.str()?;
            let meta = py.import("json")?.call_method1("loads", (encoded,))?;
            result.total_duration = number(&meta, "total_duration")?;
            result.planner_time = number(&meta, "planner_time")?;
            result.pruner_time = number(&meta, "pruner_time")?;
            result.optimizer_time = number(&meta, "optimizer_time")?;
            result.planner_status = text(&meta, "planner_status")?;
            result.optimizer_status = text(&meta, "optimizer_status")?;
            result.optimizer_success = meta
                .call_method1("get", ("optimizer_success", false))?
                .is_truthy()?;
            result.extra = py
                .get_type::<PyDict>()
                .call1((meta.call_method1("get", ("extra", PyDict::new(py)))?,))?
                .extract::<Py<PyDict>>()?;
        }
        Ok(result)
    }
}

/// The keyword arguments a numpy call needs, built from pairs.
///
/// # Errors
///
/// Returns whatever building the mapping raised.
fn kwargs<'py, T: IntoPyObject<'py> + Copy>(
    py: Python<'py>,
    pairs: &[(&str, T)],
) -> PyResult<Bound<'py, PyDict>> {
    let built = PyDict::new(py);
    for (key, value) in pairs {
        built.set_item(key, *value)?;
    }
    Ok(built)
}

/// The `dtype=float` every path array is stored with.
///
/// # Errors
///
/// Returns whatever building the mapping raised.
fn float_dtype<'py>(py: Python<'py>, numpy: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyDict>> {
    let built = PyDict::new(py);
    built.set_item("dtype", numpy.getattr("float64")?)?;
    Ok(built)
}

/// Reads a number out of the archive's metadata, or zero.
///
/// # Errors
///
/// Returns whatever reading the key raised.
fn number(meta: &Bound<'_, PyAny>, key: &str) -> PyResult<f64> {
    meta.call_method1("get", (key, 0.0_f64))?.extract::<f64>()
}

/// Reads a string out of the archive's metadata, or `unknown`.
///
/// # Errors
///
/// Returns whatever reading the key raised.
fn text(meta: &Bound<'_, PyAny>, key: &str) -> PyResult<String> {
    meta.call_method1("get", (key, PyString::new(meta.py(), "unknown")))?
        .str()?
        .extract::<String>()
}

/// Adds this module's names to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPipelineResult>()?;
    module.add_class::<PyPlanningPipeline>()
}
