//! `arco.planning` as Python still sees it.
//!
//! Every class here keeps the argument names, the positional order, the
//! default values and the exception types the pure-Python implementation
//! had, per `FR-API-01` and `FR-API-02`. Where a default cannot be
//! rendered from a Rust expression, an explicit `text_signature` carries
//! the Python one so that `inspect.signature` still answers what it used
//! to.
//!
//! Two rules shape the code below. Every planner releases the
//! interpreter around the call that computes, so a caller can run several
//! plans at once. And every policy hook left at its default selects the
//! native variant of its enum, which is what keeps the inner loop out of
//! the interpreter entirely; only a callable the caller passed becomes
//! the `Custom` variant, and that path is the slow one deviation A-07
//! describes. Each hook says which branch is which at the point it is
//! built.
//!
//! A Rust planner reports why it found nothing, through the closed
//! enumeration of `FR-INV-08`. Python returned a bare `None` and that is
//! what it still returns; the reason is published alongside it, as
//! [`PyPlanFailure`] on the planner's `last_failure`, rather than in
//! place of the `None` a caller already branches on.

use std::sync::Mutex;

use arco_core::Error;
use arco_core::geometry::euclidean_distance;
use arco_core::protocols::PlannerCost as _;
use arco_core::rng::Pcg64;
use arco_mapping::graph::NodeId;
use arco_planning::continuous::{
    CostPolicy, FeasibilityPolicy, OptimizerSettings, RrtSettings, SamplerPolicy, SegmentPolicy,
    SstSettings, SteererPolicy, TermWeights, TrajectoryTerm,
};
use arco_planning::discrete::{
    RouteOutcome, SearchDiagnostics, SearchOptions, search, search_with_diagnostics,
};
use arco_planning::failure::{PlanFailure, PlanOutcome};
use numpy::{PyArray1, PyArray2};
use pyo3::PyTypeInfo;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::config::{count, number, required, text};
use crate::errors::{OrRaise, to_exception};
use crate::hooks::{
    BoundMap, BoundOccupancy, FailureSlot, PyPlannerCost, PySampler, PySegmentChecker, PySteerer,
    PyTrajectoryTerm, SharedOccupancy, as_array, as_path, coordinates, point_rows,
};

/// Decimals a direction vector is rounded to before it is compared.
///
/// The A\* path simplifier collapses a run of steps that all point the
/// same way, and "the same way" has to survive the rounding of a
/// normalization. Eight decimals is what the Python implementation used
/// and the number is part of which path comes back.
const DIRECTION_DECIMALS: f64 = 1e8;

/// The shortest direction vector that still has a direction, in cells.
const DIRECTION_FLOOR: f64 = 1e-12;

// ---------------------------------------------------------------------
// Why a planner found nothing
// ---------------------------------------------------------------------

/// Why a planner declined to produce a path.
///
/// `FR-INV-08`. Published on every planner as `last_failure`, which is
/// `None` until a call fails. The `None` a planner returns is unchanged,
/// so nothing that read the old return value has to move.
#[pyclass(
    eq,
    eq_int,
    frozen,
    from_py_object,
    name = "PlanFailure",
    module = "arco._arco"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PyPlanFailure {
    /// The start state is inside an obstacle.
    StartOccupied,
    /// The goal state is inside an obstacle.
    GoalOccupied,
    /// The start state lies outside the map.
    StartOutsideMap,
    /// The goal state lies outside the map.
    GoalOutsideMap,
    /// The search completed and the goal is not reachable.
    Unreachable,
    /// The budget ran out before the search completed.
    BudgetExhausted,
}

#[pymethods]
impl PyPlanFailure {
    /// Whether a larger budget might change this answer.
    #[getter]
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "a pyo3 getter takes its receiver by reference"
    )]
    const fn is_retryable(&self) -> bool {
        matches!(*self, Self::BudgetExhausted)
    }

    /// The reason, written as a sentence.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "a pyo3 method takes its receiver by reference"
    )]
    fn __str__(&self) -> &'static str {
        match *self {
            Self::StartOccupied => "the start state is occupied",
            Self::GoalOccupied => "the goal state is occupied",
            Self::StartOutsideMap => "the start state is outside the map",
            Self::GoalOutsideMap => "the goal state is outside the map",
            Self::Unreachable => "no path exists",
            Self::BudgetExhausted => "the search budget ran out",
        }
    }
}

impl PyPlanFailure {
    /// Translates the crate's reason, when it is one this binding knows.
    ///
    /// `PlanFailure` is `#[non_exhaustive]`, so a variant added later
    /// arrives here as `None` rather than as a wrong answer. Adding it to
    /// the enum above is part of adding it to the crate.
    const fn of(reason: PlanFailure) -> Option<Self> {
        match reason {
            PlanFailure::StartOccupied => Some(Self::StartOccupied),
            PlanFailure::GoalOccupied => Some(Self::GoalOccupied),
            PlanFailure::StartOutsideMap => Some(Self::StartOutsideMap),
            PlanFailure::GoalOutsideMap => Some(Self::GoalOutsideMap),
            PlanFailure::Unreachable => Some(Self::Unreachable),
            PlanFailure::BudgetExhausted => Some(Self::BudgetExhausted),
            _ => None,
        }
    }
}

/// What the last call to a planner produced, for the extra attributes.
#[derive(Clone, Copy, Debug, Default)]
struct LastCall {
    failure: Option<PyPlanFailure>,
    cost: Option<f64>,
    expanded: usize,
}

impl LastCall {
    /// Records an outcome and hands back its path.
    fn record<T>(slot: &Mutex<Self>, outcome: PlanOutcome<T>) -> Option<Vec<T>> {
        let recorded = Self {
            failure: outcome.failure().and_then(PyPlanFailure::of),
            cost: outcome.cost(),
            expanded: outcome.expanded(),
        };
        if let Ok(mut last) = slot.lock() {
            *last = recorded;
        }
        match outcome {
            PlanOutcome::Found { path, .. } => Some(path),
            PlanOutcome::Failed { .. } => None,
        }
    }

    /// Records a refusal the binding decided on before the search ran.
    fn refuse(slot: &Mutex<Self>, failure: PyPlanFailure) {
        if let Ok(mut last) = slot.lock() {
            *last = Self {
                failure: Some(failure),
                cost: None,
                expanded: 0,
            };
        }
    }

    /// Reads the slot back, or its default when it cannot be locked.
    fn read(slot: &Mutex<Self>) -> Self {
        slot.lock()
            .map_or_else(|_poisoned| Self::default(), |last| *last)
    }
}

// ---------------------------------------------------------------------
// Shared conversions
// ---------------------------------------------------------------------

/// What a diagnostic search hands back, in the shapes Python expects.
///
/// The path, or `None`; the nodes in the order they were expanded; and
/// each discovered node mapped to the one it was reached from.
type Diagnostics = (Option<Vec<Py<PyAny>>>, Vec<Py<PyAny>>, Py<PyDict>);

/// The two methods a subclass replaces to change how a planner measures.
const METRIC_METHODS: [&str; 2] = ["distance", "heuristic"];

/// Whether a Python subclass of `base` replaced any of `names`.
///
/// Walks the concrete type's resolution order down to the binding class
/// and asks each class above it whether it declares the name. Deviation
/// A-24: an override is honored by calling back into the interpreter, so
/// the answer decides between the native policy and the custom one, and
/// it is decided once rather than per call.
///
/// # Errors
///
/// Returns whatever reading the type raised.
fn overrides(
    object: &Bound<'_, PyAny>,
    base: &Bound<'_, pyo3::types::PyType>,
    names: &[&str],
) -> PyResult<bool> {
    let concrete = object.get_type();
    if concrete.is(base) {
        return Ok(false);
    }
    for ancestor in concrete.getattr("__mro__")?.try_iter()? {
        let ancestor = ancestor?;
        if ancestor.is(base) {
            // Everything from here down is the binding itself.
            break;
        }
        let declared = ancestor.getattr("__dict__")?;
        for name in names {
            if declared.contains(*name)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Converts a core error into the exception Python used to raise.
///
/// A hook that raised is restored first, so a caller's own exception type
/// and traceback survive the trip through a trait that cannot carry them.
fn raised(failure: &Error, slot: &FailureSlot) -> PyErr {
    slot.take().unwrap_or_else(|| to_exception(failure))
}

/// Runs `plan` with the interpreter released and restores what raised.
///
/// # Errors
///
/// Returns whatever the planner or one of its hooks reported.
fn detached<T, F>(py: Python<'_>, slot: &FailureSlot, plan: F) -> PyResult<T>
where
    F: Ungil + Send + FnOnce() -> Result<T, Error>,
    T: Ungil + Send,
{
    let outcome = py.detach(plan);
    match outcome {
        Err(failure) => Err(raised(&failure, slot)),
        // A hook that raised from a trait method with no way to report it
        // parks the exception and hands the search an empty answer, so a
        // run can finish looking successful with an exception still held.
        // Draining here raises it on the call that caused it, and leaves
        // nothing behind for the next call to surface with the wrong type.
        Ok(value) => slot.take().map_or(Ok(value), Err),
    }
}

/// Reads the sampling bounds a continuous planner takes.
///
/// # Errors
///
/// Returns a `ValueError` when the sequence is empty, which is the
/// message the Python constructors raised.
fn sampling_bounds(bounds: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    // Each axis arrives as whatever pair the caller had to hand. Python
    // indexed it, so a list, a tuple and a two-element array were all
    // acceptable and callers use each of them; extracting only tuples
    // rejects `[[0.0, 10.0], [0.0, 10.0]]`, which the pipeline tests pass.
    let read: Vec<(f64, f64)> = bounds.extract::<Vec<(f64, f64)>>().or_else(|_not_pairs| {
        bounds
            .extract::<Vec<Vec<f64>>>()?
            .into_iter()
            .map(|axis| match axis.as_slice() {
                [low, high] => Ok((*low, *high)),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "each bound needs a low and a high, got {} value(s).",
                    other.len()
                ))),
            })
            .collect()
    })?;
    if read.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bounds must not be empty.",
        ));
    }
    Ok(read)
}

/// Broadcasts a scalar or per-axis step size over `axes` dimensions.
///
/// # Errors
///
/// Returns a `ValueError` when any component is not strictly positive,
/// which is the message the Python constructors raised.
fn step_sizes(step_size: &Bound<'_, PyAny>, axes: usize) -> PyResult<Vec<f64>> {
    let read = step_size.extract::<f64>().map_or_else(
        |_not_a_scalar| coordinates(step_size),
        |scalar| Ok(vec![scalar; axes]),
    )?;
    if read
        .iter()
        .any(|scale| !(scale.is_finite() && *scale > 0.0))
    {
        let shown = step_size.repr()?;
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step_size must be positive, got {shown}."
        )));
    }
    Ok(read)
}

/// Seeds the native generator the way the Python planner seeded numpy's.
///
/// `FR-RNG-02`: a seed reproduces `numpy.random.default_rng(seed)` draw
/// for draw. A caller-supplied generator seeds this one with a single
/// draw off it, so a reproducible generator still gives a reproducible
/// plan. With neither, the seed comes from the interpreter's own entropy.
///
/// # Errors
///
/// Returns whatever drawing the seed raised.
fn seeded(py: Python<'_>, seed: Option<u64>, rng: Option<&Py<PyAny>>) -> PyResult<Pcg64> {
    if let Some(seed) = seed {
        return Ok(Pcg64::seed_from_u64(seed));
    }
    if let Some(generator) = rng {
        let drawn = generator
            .bind(py)
            .call_method1("integers", (0_u64, u64::MAX))?
            .extract::<u64>()?;
        return Ok(Pcg64::seed_from_u64(drawn));
    }
    let drawn = py
        .import("random")?
        .call_method1("getrandbits", (64_u32,))?
        .extract::<u64>()?;
    Ok(Pcg64::seed_from_u64(drawn))
}

// ---------------------------------------------------------------------
// The base classes a caller subclasses
// ---------------------------------------------------------------------

/// Default distance and heuristic cost functions for planners.
///
/// Subclassed rather than only called: a caller overrides `distance` to
/// change the metric without rewriting a search. Deviation A-24 records
/// what that costs, which is that an override is reached by calling back
/// into the interpreter, the same crossing the keyword hooks pay under
/// A-07.
#[pyclass(subclass, name = "PlannerCost", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyCostModel;

#[pymethods]
impl PyCostModel {
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

    /// Builds the default cost model, which carries no state.
    ///
    /// Accepts and ignores whatever a subclass was constructed with, the
    /// way `object` does for any class overriding `__init__`. A generated
    /// constructor with a fixed empty signature refuses them instead, and
    /// every Python subclass carrying its own arguments stops building.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    #[pyo3(text_signature = "()")]
    const fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    /// Euclidean distance between two states.
    #[pyo3(text_signature = "(state_a, state_b)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn distance(&self, state_a: &Bound<'_, PyAny>, state_b: &Bound<'_, PyAny>) -> PyResult<f64> {
        let (from, to) = (coordinates(state_a)?, coordinates(state_b)?);
        euclidean_distance(&from, &to).or_raise()
    }

    /// An admissible estimate of the remaining cost.
    ///
    /// Reaches `distance` through the instance, so a subclass overriding
    /// the metric changes the estimate with it, which is what the Python
    /// base did by writing `self.distance(...)`.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn heuristic(
        slf: &Bound<'_, Self>,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        slf.call_method1("distance", (state_a, state_b))?.extract()
    }
}

/// Base class for discrete planners operating on graphs, grids included.
#[pyclass(extends = PyCostModel, subclass, name = "DiscretePlanner", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyDiscretePlanner {
    graph: Py<PyAny>,
    has_heuristic: bool,
}

#[pymethods]
impl PyDiscretePlanner {
    /// Builds a planner over `graph`.
    #[new]
    #[pyo3(text_signature = "(graph)")]
    fn new(graph: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(PyCostModel).add_subclass(Self::over(graph)?))
    }

    /// The graph this planner searches.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyAny> {
        self.graph.clone_ref(py)
    }

    /// The graph edge cost between two nodes.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn distance(
        &self,
        py: Python<'_>,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        self.graph
            .bind(py)
            .call_method1("distance", (state_a, state_b))?
            .extract()
    }

    /// A remaining-cost estimate, from the graph when it publishes one.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn heuristic(
        slf: &Bound<'_, Self>,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let planner = slf.borrow();
        if planner.has_heuristic {
            return planner
                .graph
                .bind(slf.py())
                .call_method1("heuristic", (state_a, state_b))?
                .extract();
        }
        drop(planner);
        slf.call_method1("distance", (state_a, state_b))?.extract()
    }
}

impl PyDiscretePlanner {
    /// Builds the base over `graph`, for a subclass to stack onto.
    ///
    /// # Errors
    ///
    /// Returns whatever inspecting the graph raised.
    fn over(graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            graph: graph.clone().unbind(),
            has_heuristic: graph.hasattr("heuristic")?,
        })
    }
}

/// Base class for planners operating in continuous state spaces.
///
/// Carries the four arguments every continuous planner takes and the
/// helpers built on them. A subclass implements `plan`.
#[pyclass(extends = PyCostModel, subclass, name = "ContinuousPlanner", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyContinuousPlanner {
    occupancy: Py<PyAny>,
    cost: Option<Py<PyAny>>,
    publisher: Option<Py<PyAny>>,
    seed: Option<u64>,
    rng: Option<Py<PyAny>>,
}

#[pymethods]
impl PyContinuousPlanner {
    /// Builds a planner over `occupancy`.
    #[new]
    #[pyo3(signature = (occupancy, cost = None, publisher = None, seed = None, rng = None))]
    #[pyo3(text_signature = "(occupancy, cost=None, publisher=None, seed=None, rng=None)")]
    fn new(
        occupancy: &Bound<'_, PyAny>,
        cost: Option<Py<PyAny>>,
        publisher: Option<Py<PyAny>>,
        seed: Option<u64>,
        rng: Option<Py<PyAny>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyCostModel)
            .add_subclass(Self::over(occupancy, cost, publisher, seed, rng))
    }

    /// The occupancy map this planner checks against.
    #[getter]
    fn occupancy(&self, py: Python<'_>) -> Py<PyAny> {
        self.occupancy.clone_ref(py)
    }

    /// Step-size-normalized Euclidean distance between two states.
    ///
    /// Defers to a `cost=` model when one was given, which is the branch
    /// that crosses into the interpreter, per A-07.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn distance(
        slf: &Bound<'_, Self>,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let py = slf.py();
        if let Some(model) = slf.borrow().cost.as_ref() {
            return model
                .bind(py)
                .call_method1("distance", (state_a, state_b))?
                .extract();
        }
        let (from, to) = (coordinates(state_a)?, coordinates(state_b)?);
        // The scale lives on the concrete planner rather than here, the
        // same way the Python base read it off `self`.
        let step_size = slf.getattr("step_size").map_or_else(
            |_unscaled| Ok(vec![1.0; from.len()]),
            |scale| scales(&scale, from.len()),
        )?;
        CostPolicy::Scaled { step_size }
            .distance(&from, &to)
            .or_raise()
    }

    /// A remaining-cost estimate, which defaults to the distance.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn heuristic(
        slf: &Bound<'_, Self>,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let py = slf.py();
        if let Some(model) = slf.borrow().cost.as_ref() {
            return model
                .bind(py)
                .call_method1("heuristic", (state_a, state_b))?
                .extract();
        }
        slf.call_method1("distance", (state_a, state_b))?.extract()
    }

    /// The generator this planner samples with.
    #[pyo3(text_signature = "()")]
    fn make_rng(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        generator_of(py, self.seed, self.rng.as_ref())
    }

    /// Publishes a telemetry snapshot through the configured sink.
    #[pyo3(text_signature = "(telemetry)")]
    fn publish_telemetry(&self, py: Python<'_>, telemetry: &Bound<'_, PyAny>) -> PyResult<()> {
        publish_snapshot(py, self.publisher.as_ref(), telemetry)
    }

    /// Plans a path from `start` to `goal`.
    #[pyo3(text_signature = "(start, goal)")]
    #[expect(
        clippy::unused_self,
        reason = "Python calls this on an instance whatever the body reads"
    )]
    fn plan(&self, _start: &Bound<'_, PyAny>, _goal: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "ContinuousPlanner.plan is abstract.",
        ))
    }
}

impl PyContinuousPlanner {
    /// Builds the base over `occupancy`, for a subclass to stack onto.
    fn over(
        occupancy: &Bound<'_, PyAny>,
        cost: Option<Py<PyAny>>,
        publisher: Option<Py<PyAny>>,
        seed: Option<u64>,
        rng: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            occupancy: occupancy.clone().unbind(),
            cost,
            publisher,
            seed,
            rng,
        }
    }
}

/// Broadcasts a scalar or per-axis scale over `axes` dimensions.
///
/// # Errors
///
/// Returns a `TypeError` when the value is neither a number nor a
/// sequence of numbers.
fn scales(step_size: &Bound<'_, PyAny>, axes: usize) -> PyResult<Vec<f64>> {
    step_size.extract::<f64>().map_or_else(
        |_not_a_scalar| coordinates(step_size),
        |scalar| Ok(vec![scalar; axes]),
    )
}

/// The generator a continuous planner hands its Python hooks.
///
/// # Errors
///
/// Returns whatever importing numpy or building the generator raised.
fn generator_of(py: Python<'_>, seed: Option<u64>, rng: Option<&Py<PyAny>>) -> PyResult<Py<PyAny>> {
    if let Some(generator) = rng {
        return Ok(generator.clone_ref(py));
    }
    Ok(py
        .import("numpy")?
        .getattr("random")?
        .call_method1("default_rng", (seed,))?
        .unbind())
}

/// Publishes one telemetry snapshot, through the sink or to the file.
///
/// # Errors
///
/// Returns whatever the sink raised.
fn publish_snapshot(
    py: Python<'_>,
    publisher: Option<&Py<PyAny>>,
    telemetry: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match publisher {
        Some(sink) => {
            sink.bind(py).call1((telemetry,))?;
        }
        None => {
            py.import("arco.planning.continuous.telemetry")?
                .call_method1("write_telemetry", (telemetry,))?;
        }
    }
    Ok(())
}

/// Publishes one progress snapshot, swallowing whatever the sink raised.
///
/// A sink that fails must not abandon a plan that is going fine: Python's
/// `write_telemetry` swallowed every I/O error for the same reason, and a
/// loading screen that has gone away is not a planning failure.
fn report_progress(
    publisher: Option<&Py<PyAny>>,
    algorithm: &str,
    progress: &arco_planning::continuous::PlannerProgress,
) {
    Python::attach(|py| {
        let Ok(module) = py.import("arco.planning.continuous.telemetry") else {
            return;
        };
        let Ok(kind) = module.getattr("PlannerTelemetry") else {
            return;
        };
        // `inf` is what Python carried before any node had been placed,
        // and the loading screen renders it as an unknown distance.
        let snapshot = kind.call1((
            algorithm,
            "exploring",
            progress.iteration,
            progress.max_iterations,
            progress.best_distance_to_goal,
        ));
        if let Ok(snapshot) = snapshot {
            let _ignored = publish_snapshot(py, publisher, &snapshot);
        }
    });
}

// ---------------------------------------------------------------------
// A* over a discrete map
// ---------------------------------------------------------------------

/// A\* over any graph, including a grid.
///
/// `graph` is read once and rebuilt natively when it is a grid or a
/// positioned graph, after which the search runs with the interpreter
/// released. A graph of any other shape keeps its Python methods and is
/// called per edge, which is the slow branch of deviation A-07.
#[pyclass(extends = PyDiscretePlanner, name = "AStarPlanner", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PyAStarPlanner {
    graph: Py<PyAny>,
    map: BoundMap,
    failure: FailureSlot,
    last: Mutex<LastCall>,
    overridden: Mutex<Option<bool>>,
    /// Whether a successful path has its collinear runs collapsed.
    #[pyo3(get, set)]
    simplify_path: bool,
    /// Whether direction continuity breaks a tie in the open set.
    ///
    /// Always false: the compiled open set orders on cost and then on
    /// insertion sequence, and carries no direction tie-break to switch
    /// on. Reported as what the search does rather than as what was
    /// asked for, and an explicit request for the other value is refused
    /// by the constructor rather than accepted and dropped.
    #[pyo3(get)]
    prefer_straight: bool,
    /// Most nodes the search may expand before it reports an exhausted
    /// budget, per `FR-SAFE-02`.
    #[pyo3(get, set)]
    max_expansions: usize,
}

#[pymethods]
impl PyAStarPlanner {
    /// Builds a planner over `graph`.
    #[new]
    #[pyo3(signature = (graph, heuristic = None, simplify_path = true, prefer_straight = true))]
    #[pyo3(text_signature = "(graph, heuristic=None, simplify_path=True, prefer_straight=True)")]
    fn new(
        graph: &Bound<'_, PyAny>,
        heuristic: Option<Py<PyAny>>,
        simplify_path: bool,
        prefer_straight: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let failure = FailureSlot::default();
        // Native when no heuristic was injected, which is the fast branch
        // and the default. A supplied `heuristic=` has to be consulted per
        // expanded node, so the map falls back to its Python methods and
        // the search reacquires the interpreter per edge, per A-07.
        let map = BoundMap::adopt(graph, failure.clone(), heuristic, None)?;
        Self::assemble(graph, map, failure, simplify_path, prefer_straight)
    }

    /// The graph this planner searches.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyAny> {
        self.graph.clone_ref(py)
    }

    /// Why the last call found nothing, or `None`.
    #[getter]
    fn last_failure(&self) -> Option<PyPlanFailure> {
        LastCall::read(&self.last).failure
    }

    /// What the last successful path cost, or `None`.
    #[getter]
    fn last_cost(&self) -> Option<f64> {
        LastCall::read(&self.last).cost
    }

    /// How many nodes the last call expanded.
    #[getter]
    fn last_expanded(&self) -> usize {
        LastCall::read(&self.last).expanded
    }

    /// Whether a search runs entirely in Rust.
    ///
    /// False once the graph is one this binding could not rebuild, or a
    /// subclass replaced `distance` or `heuristic`, which is the
    /// condition `FR-PERF-03` asks callers to be able to check.
    #[getter]
    fn is_native(slf: &Bound<'_, Self>) -> PyResult<bool> {
        slf.borrow().runs_natively(slf)
    }

    /// The graph edge cost between two nodes.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn distance(&self, state_a: &Bound<'_, PyAny>, state_b: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.map.distance(state_a, state_b)
    }

    /// A remaining-cost estimate between two nodes.
    #[pyo3(text_signature = "(state_a, state_b)")]
    fn heuristic(&self, state_a: &Bound<'_, PyAny>, state_b: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.map.heuristic(state_a, state_b)
    }

    /// Plans a path from `start` to `goal`.
    ///
    /// Returns the node sequence, or `None` when no path exists or the
    /// start node is occupied. `last_failure` says which.
    #[pyo3(text_signature = "(start, goal)")]
    fn plan(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyList>>> {
        let py = slf.py();
        let planner = slf.borrow();
        // The Python search refused an occupied start before expanding
        // anything, and a caller reading `None` has been branching on
        // that ever since.
        if planner.map.is_occupied(start)? {
            LastCall::refuse(&planner.last, PyPlanFailure::StartOccupied);
            return Ok(None);
        }
        let options = SearchOptions {
            max_expansions: planner.max_expansions,
            use_heuristic: true,
            prefer_straight: planner.prefer_straight,
        };

        // A subclass that replaced `distance` or `heuristic` has to be
        // the metric the search measures with, and no native map can call
        // it, so the whole map moves onto its Python methods for the
        // duration. Deviation A-24, and the reason `is_native` reports
        // false once it happens.
        let replaced = if planner.is_overridden(slf)? {
            Some(BoundMap::adopt(
                planner.graph.bind(py),
                planner.failure.clone(),
                None,
                Some(slf.clone().into_any().unbind()),
            )?)
        } else {
            None
        };
        let map = replaced.as_ref().unwrap_or(&planner.map);

        let Some(path) = map.search(py, start, goal, options, &planner.failure, &planner.last)?
        else {
            return Ok(None);
        };
        let path = if planner.simplify_path {
            simplified(py, path)
        } else {
            path
        };
        Ok(Some(PyList::new(py, path)?.unbind()))
    }

    /// Plans and reports the exploration tree alongside the path.
    ///
    /// Returns `(path, expanded_order, parent_map)`, which is what a
    /// visualizer needs to redraw the search without running it again.
    #[pyo3(text_signature = "(start, goal)")]
    fn plan_with_diagnostics(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        let planner = slf.borrow();
        if planner.map.is_occupied(start)? {
            LastCall::refuse(&planner.last, PyPlanFailure::StartOccupied);
            return Ok(PyTuple::new(
                py,
                [
                    py.None(),
                    PyList::empty(py).unbind().into_any(),
                    PyDict::new(py).unbind().into_any(),
                ],
            )?
            .unbind());
        }
        let options = SearchOptions {
            max_expansions: planner.max_expansions,
            use_heuristic: true,
            prefer_straight: planner.prefer_straight,
        };
        let replaced = if planner.is_overridden(slf)? {
            Some(BoundMap::adopt(
                planner.graph.bind(py),
                planner.failure.clone(),
                None,
                Some(slf.clone().into_any().unbind()),
            )?)
        } else {
            None
        };
        let map = replaced.as_ref().unwrap_or(&planner.map);

        let (path, expanded, came_from) =
            map.search_diagnostics(py, start, goal, options, &planner.failure, &planner.last)?;
        let path = path.map(|found| {
            if planner.simplify_path {
                simplified(py, found)
            } else {
                found
            }
        });
        let path = match path {
            Some(found) => PyList::new(py, found)?.unbind().into_any(),
            None => py.None(),
        };
        Ok(PyTuple::new(
            py,
            [
                path,
                PyList::new(py, expanded)?.unbind().into_any(),
                came_from.into_any(),
            ],
        )?
        .unbind())
    }
}

impl PyAStarPlanner {
    /// Whether a subclass replaced how this planner measures.
    ///
    /// Decided once and remembered, since the answer is a property of the
    /// type rather than of the query.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the type raised.
    fn is_overridden(&self, bound: &Bound<'_, Self>) -> PyResult<bool> {
        let base = Self::type_object(bound.py());
        let Ok(mut cached) = self.overridden.lock() else {
            return overrides(bound.as_any(), &base, &METRIC_METHODS);
        };
        if let Some(known) = *cached {
            return Ok(known);
        }
        let found = overrides(bound.as_any(), &base, &METRIC_METHODS)?;
        *cached = Some(found);
        Ok(found)
    }

    /// Whether a search runs without touching the interpreter.
    ///
    /// # Errors
    ///
    /// As [`PyAStarPlanner::is_overridden`].
    fn runs_natively(&self, bound: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(!matches!(self.map, BoundMap::Python(_)) && !self.is_overridden(bound)?)
    }

    /// Stacks the planner on the two bases Python declares above it.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the graph raised.
    fn assemble(
        graph: &Bound<'_, PyAny>,
        map: BoundMap,
        failure: FailureSlot,
        simplify_path: bool,
        prefer_straight: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(PyCostModel)
            .add_subclass(PyDiscretePlanner::over(graph)?)
            .add_subclass(Self {
                graph: graph.clone().unbind(),
                map,
                failure,
                last: Mutex::new(LastCall::default()),
                overridden: Mutex::new(None),
                simplify_path,
                prefer_straight,
                max_expansions: SearchOptions::default().max_expansions,
            }))
    }
}

/// Collapses the collinear runs of a node sequence.
///
/// The Python search did this by default, so the sequence a caller gets
/// back depends on it. A node that is not a numeric sequence has no
/// direction, and the path is then returned untouched, which is what the
/// Python implementation did for a graph whose nodes are identifiers.
fn simplified(py: Python<'_>, path: Vec<Py<PyAny>>) -> Vec<Py<PyAny>> {
    if path.len() < 3 {
        return path;
    }
    let mut points = Vec::with_capacity(path.len());
    for node in &path {
        let Some(point) = direction_vector(node.bind(py)) else {
            return path;
        };
        points.push(point);
    }

    let Some(mut previous) = direction_between(points.first(), points.get(1)) else {
        return path;
    };
    let mut kept = Vec::with_capacity(path.len());
    if let Some(first) = path.first() {
        kept.push(first.clone_ref(py));
    }
    for index in 1..path.len().saturating_sub(1) {
        let Some(current) =
            direction_between(points.get(index), points.get(index.saturating_add(1)))
        else {
            return path;
        };
        if !same_direction(&current, &previous) {
            if let Some(node) = path.get(index) {
                kept.push(node.clone_ref(py));
            }
            previous = current;
        }
    }
    if let Some(last) = path.last() {
        kept.push(last.clone_ref(py));
    }
    kept
}

/// Reads a node as coordinates, when it is a numeric sequence of two or
/// more components.
fn direction_vector(node: &Bound<'_, PyAny>) -> Option<Vec<f64>> {
    let read = coordinates(node).ok()?;
    (read.len() >= 2).then_some(read)
}

/// The unit direction from one point to the next, rounded for comparison.
fn direction_between(from: Option<&Vec<f64>>, to: Option<&Vec<f64>>) -> Option<Vec<f64>> {
    let (from, to) = (from?, to?);
    if from.len() != to.len() {
        return None;
    }
    let length = to
        .iter()
        .zip(from)
        .map(|(end, start)| (end - start) * (end - start))
        .sum::<f64>()
        .sqrt();
    if length <= DIRECTION_FLOOR {
        return None;
    }
    Some(
        to.iter()
            .zip(from)
            .map(|(end, start)| {
                ((end - start) / length * DIRECTION_DECIMALS).round() / DIRECTION_DECIMALS
            })
            .collect(),
    )
}

/// Whether two rounded directions are the same one.
///
/// Compared by total order rather than by `==`, which `clippy::float_cmp`
/// forbids for the good reason that a NaN compares false against itself.
fn same_direction(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(first, second)| first.total_cmp(second).is_eq())
}

// ---------------------------------------------------------------------
// The grid wrapper
// ---------------------------------------------------------------------

/// A\* over a numpy grid where zero is free and one is occupied.
#[pyclass(name = "AStar", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PyAStar {
    planner: Py<PyAStarPlanner>,
}

#[pymethods]
impl PyAStar {
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

    /// Builds a planner over `grid`.
    #[new]
    #[pyo3(signature = (grid, grid_type = "manhattan"))]
    #[pyo3(text_signature = "(grid, grid_type='manhattan')")]
    fn new(py: Python<'_>, grid: &Bound<'_, PyAny>, grid_type: &str) -> PyResult<Self> {
        if !matches!(grid_type, "euclidean" | "manhattan") {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grid_type must be 'euclidean' or 'manhattan', got '{grid_type}'."
            )));
        }
        let failure = FailureSlot::default();
        let map = BoundMap::from_cells(grid, grid_type == "euclidean")?;
        let planner = PyAStarPlanner::assemble(grid, map, failure, true, true)?;
        Ok(Self {
            planner: Py::new(py, planner)?,
        })
    }

    /// Searches for a path from `start` to `goal`.
    #[pyo3(text_signature = "(start, goal)")]
    fn search(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyList>>> {
        PyAStarPlanner::plan(self.planner.bind(py), start, goal)
    }

    /// Why the last search found nothing, or `None`.
    #[getter]
    fn last_failure(&self, py: Python<'_>) -> Option<PyPlanFailure> {
        self.planner.bind(py).borrow().last_failure()
    }
}

// ---------------------------------------------------------------------
// Routing over a positioned graph
// ---------------------------------------------------------------------

/// The named tuple `RouteRouter.plan` returns.
///
/// Built with `collections.namedtuple` rather than declared as a class
/// here, because a caller unpacks it, indexes it, compares it against a
/// plain tuple and calls `_asdict` and `_replace` on it. Those are tuple
/// behaviors rather than attribute behaviors, and reproducing them one by
/// one on a declared class would reproduce most of them and miss the
/// rest. Built once and cached, since the type identity has to be stable
/// for equality between two results to mean anything.
static ROUTE_RESULT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The seven fields of a route result, in the order Python declares them.
const ROUTE_FIELDS: [&str; 7] = [
    "path",
    "start_node",
    "goal_node",
    "start_projection",
    "goal_projection",
    "start_distance",
    "goal_distance",
];

/// The `RouteResult` type, building it the first time it is asked for.
///
/// # Errors
///
/// Returns whatever building the named tuple raised.
fn route_result_type(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    ROUTE_RESULT.get_or_try_init(py, || {
        let built = py
            .import("collections")?
            .getattr("namedtuple")?
            .call1(("RouteResult", ROUTE_FIELDS))?;
        built.setattr("__module__", "arco._arco")?;
        built.setattr(
            "__doc__",
            "Result of a route planning query: the node sequence, the two \
             projected nodes, where each projection landed, and how far \
             each query position was from it.",
        )?;
        Ok(built.unbind())
    })
}

/// Routes continuous positions over a positioned graph.
#[pyclass(name = "RouteRouter", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PyRouteRouter {
    graph: Py<PyAny>,
    router: arco_planning::discrete::RouteRouter<arco_mapping::graph::CartesianGraph>,
    planner: Option<Py<PyAny>>,
    failure: FailureSlot,
    last: Mutex<LastCall>,
    /// How far a position may be from the network and still join it.
    #[pyo3(get, set)]
    activation_radius: Option<f64>,
}

#[pymethods]
impl PyRouteRouter {
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

    /// Builds a router over `graph`.
    #[new]
    #[pyo3(signature = (graph, activation_radius = None, planner = None))]
    #[pyo3(text_signature = "(graph, activation_radius=None, planner=None)")]
    fn new(
        graph: &Bound<'_, PyAny>,
        activation_radius: Option<f64>,
        planner: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let failure = FailureSlot::default();
        let network = BoundMap::cartesian_of(graph)?;
        Ok(Self {
            graph: graph.clone().unbind(),
            router: arco_planning::discrete::RouteRouter::new(network, activation_radius),
            planner,
            failure,
            last: Mutex::new(LastCall::default()),
            activation_radius,
        })
    }

    /// The graph this router routes over.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<PyAny> {
        self.graph.clone_ref(py)
    }

    /// Why the last call found nothing, or `None`.
    #[getter]
    fn last_failure(&self) -> Option<PyPlanFailure> {
        LastCall::read(&self.last).failure
    }

    /// Routes from one continuous position to another.
    ///
    /// Returns `None` when either endpoint is outside the activation
    /// radius or no path connects the two projected nodes.
    #[pyo3(text_signature = "(start_position, goal_position)")]
    fn plan(
        &self,
        py: Python<'_>,
        start_position: &Bound<'_, PyAny>,
        goal_position: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let start = coordinates(start_position)?;
        let goal = coordinates(goal_position)?;
        let outcome = detached(py, &self.failure, || self.router.plan(&start, &goal))?;

        let RouteOutcome::Found(route) = outcome else {
            let reason = outcome.failure().and_then(PyPlanFailure::of);
            if let (Ok(mut last), Some(reason)) = (self.last.lock(), reason) {
                *last = LastCall {
                    failure: Some(reason),
                    cost: None,
                    expanded: outcome.expanded(),
                };
            }
            return Ok(None);
        };

        // A caller-supplied planner replaces the search but not the
        // projection, which is what the Python router did with its own
        // `planner=` argument. The native search above already ran, so
        // this branch pays for both; it exists for parity, not for speed.
        let path = match &self.planner {
            Some(custom) => {
                let answered = custom
                    .bind(py)
                    .call_method1("plan", (route.start.node, route.goal.node))?;
                if answered.is_none() {
                    LastCall::refuse(&self.last, PyPlanFailure::Unreachable);
                    return Ok(None);
                }
                PyList::new(py, answered.extract::<Vec<NodeId>>()?)?
            }
            None => PyList::new(py, route.path.clone())?,
        };

        if let Ok(mut last) = self.last.lock() {
            *last = LastCall {
                failure: None,
                cost: Some(route.cost),
                expanded: route.expanded,
            };
        }
        // The route's cost and expansion count stay off the tuple, where
        // they would change its arity, and are published on the router as
        // `last_cost` and `last_expanded` instead.
        let result = route_result_type(py)?.bind(py).call1((
            path,
            route.start.node,
            route.goal.node,
            as_array(py, &route.start.point),
            as_array(py, &route.goal.point),
            route.start.distance,
            route.goal.distance,
        ))?;
        Ok(Some(result.unbind()))
    }
}

// ---------------------------------------------------------------------
// The sampling planners
// ---------------------------------------------------------------------

/// Everything the two sampling planners hold in common.
///
/// Both take the same sixteen arguments in the same order apart from one,
/// and both build the same four policies out of them, so the policy
/// selection lives here once and says once which branch is native.
#[derive(Debug)]
struct SamplingSetup {
    occupancy: Py<PyAny>,
    shared: SharedOccupancy,
    bounds: Vec<(f64, f64)>,
    step_size: Vec<f64>,
    collision_check_count: usize,
    sampler: Option<Py<PyAny>>,
    steerer: Option<Py<PyAny>>,
    segment_free: Option<Py<PyAny>>,
    cost: Option<Py<PyAny>>,
    publisher: Option<Py<PyAny>>,
    seed: Option<u64>,
    rng: Option<Py<PyAny>>,
    failure: FailureSlot,
    overridden: Mutex<Option<bool>>,
}

impl SamplingSetup {
    /// Reads the arguments both planners share.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` when the bounds are empty or a step size is
    /// not strictly positive, which is what the Python constructors
    /// raised and with the same messages.
    #[expect(
        clippy::too_many_arguments,
        reason = "the count is the Python constructor's, which FR-API-02 pins"
    )]
    fn new(
        occupancy: &Bound<'_, PyAny>,
        bounds: &Bound<'_, PyAny>,
        step_size: &Bound<'_, PyAny>,
        collision_check_count: usize,
        sampler: Option<Py<PyAny>>,
        steerer: Option<Py<PyAny>>,
        segment_free: Option<Py<PyAny>>,
        cost: Option<Py<PyAny>>,
        publisher: Option<Py<PyAny>>,
        seed: Option<u64>,
        rng: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let failure = FailureSlot::default();
        let bounds = sampling_bounds(bounds)?;
        let step_size = step_sizes(step_size, bounds.len())?;
        let shared = SharedOccupancy::new(BoundOccupancy::adopt(occupancy, failure.clone()));
        Ok(Self {
            occupancy: occupancy.clone().unbind(),
            shared,
            bounds,
            step_size,
            collision_check_count,
            sampler,
            steerer,
            segment_free,
            cost,
            publisher,
            seed,
            rng,
            failure,
            overridden: Mutex::new(None),
        })
    }

    /// The metric every tolerance and radius is expressed in.
    ///
    /// `Scaled` is the native branch and the default. A `cost=` model
    /// crosses into the interpreter on every distance the planner
    /// measures, which is millions of calls per plan, per A-07.
    fn cost_policy(&self, py: Python<'_>) -> CostPolicy {
        self.cost.as_ref().map_or_else(
            || CostPolicy::Scaled {
                step_size: self.step_size.clone(),
            },
            |model| {
                CostPolicy::Custom(Box::new(PyPlannerCost::new(
                    model.clone_ref(py),
                    self.failure.clone(),
                )))
            },
        )
    }

    /// The metric the planner loop measures with, for one query.
    ///
    /// Three branches, in the order the Python planner resolved them. A
    /// subclass that replaced `distance` or `heuristic` wraps everything
    /// below it, so the planner object itself becomes the metric and the
    /// loop crosses into the interpreter per measurement, per A-24. With
    /// no override, a `cost=` model is the next layer, and that crossing
    /// is A-07. With neither, the native scaled metric runs and the loop
    /// never leaves Rust, which is the case `FR-PERF-02` is about.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the planner's type raised.
    fn cost_policy_for(
        &self,
        planner: &Bound<'_, PyAny>,
        base: &Bound<'_, pyo3::types::PyType>,
    ) -> PyResult<CostPolicy> {
        if self.is_overridden(planner, base)? {
            return Ok(CostPolicy::Custom(Box::new(PyPlannerCost::new(
                planner.clone().unbind(),
                self.failure.clone(),
            ))));
        }
        Ok(self.cost_policy(planner.py()))
    }

    /// Whether a subclass replaced how this planner measures.
    ///
    /// Decided once and remembered, since the answer is a property of the
    /// type rather than of the query.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the type raised.
    fn is_overridden(
        &self,
        planner: &Bound<'_, PyAny>,
        base: &Bound<'_, pyo3::types::PyType>,
    ) -> PyResult<bool> {
        let Ok(mut cached) = self.overridden.lock() else {
            return overrides(planner, base, &METRIC_METHODS);
        };
        if let Some(known) = *cached {
            return Ok(known);
        }
        let found = overrides(planner, base, &METRIC_METHODS)?;
        *cached = Some(found);
        Ok(found)
    }

    /// Where new states come from.
    ///
    /// `UniformBox` is the native branch and the default. A `sampler=`
    /// callable is invoked once per iteration through the interpreter.
    ///
    /// # Errors
    ///
    /// Returns whatever building the generator for a custom sampler
    /// raised.
    fn sampler_policy(&self, py: Python<'_>) -> PyResult<SamplerPolicy> {
        let Some(callable) = self.sampler.as_ref() else {
            return Ok(SamplerPolicy::UniformBox {
                bounds: self.bounds.clone(),
            });
        };
        Ok(SamplerPolicy::Custom(Box::new(PySampler::new(
            callable.clone_ref(py),
            self.make_rng(py)?,
            self.failure.clone(),
        ))))
    }

    /// How the tree grows toward a sample.
    ///
    /// `Straight` is the native branch and the default. A `steerer=`
    /// callable is invoked once per iteration through the interpreter.
    fn steerer_policy(&self, py: Python<'_>) -> SteererPolicy {
        self.steerer.as_ref().map_or_else(
            || SteererPolicy::Straight {
                step_size: self.step_size.clone(),
            },
            |callable| {
                SteererPolicy::Custom(Box::new(PySteerer::new(
                    callable.clone_ref(py),
                    self.failure.clone(),
                )))
            },
        )
    }

    /// How an edge is checked for collision.
    ///
    /// `Sampled` is the native branch and the default, at the density the
    /// Python planners used, which is `collision_check_count + 2` points
    /// including both endpoints. Deviation A-12 keeps it rather than the
    /// exact check so the planners behave as they were tuned. A
    /// `segment_free=` callable is invoked several times per iteration
    /// through the interpreter.
    fn segment_policy(&self, py: Python<'_>) -> SegmentPolicy<SharedOccupancy> {
        self.segment_free.as_ref().map_or_else(
            || SegmentPolicy::Sampled {
                occupancy: self.shared.clone(),
                count: self.collision_check_count.saturating_add(2),
            },
            |callable| {
                SegmentPolicy::Custom(Box::new(PySegmentChecker::new(
                    callable.clone_ref(py),
                    self.failure.clone(),
                )))
            },
        )
    }

    /// The generator the Python planner would have handed its hooks.
    ///
    /// # Errors
    ///
    /// Returns whatever importing numpy or building the generator raised.
    fn make_rng(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        generator_of(py, self.seed, self.rng.as_ref())
    }

    /// Publishes a telemetry snapshot through the configured sink.
    ///
    /// # Errors
    ///
    /// Returns whatever the sink raised.
    fn publish(&self, py: Python<'_>, telemetry: &Bound<'_, PyAny>) -> PyResult<()> {
        publish_snapshot(py, self.publisher.as_ref(), telemetry)
    }

    /// Whether a plan runs without touching the interpreter at all.
    fn is_native(&self) -> bool {
        self.shared.is_native()
            && self.sampler.is_none()
            && self.steerer.is_none()
            && self.segment_free.is_none()
            && self.cost.is_none()
    }
}

/// Declares a sampling planner, its own methods plus the shared ones.
///
/// The whole `#[pymethods]` block is generated here rather than a run of
/// methods inside one, because `#[pymethods]` rejects a macro call where
/// an item belongs and `multiple-pymethods` is not enabled.
macro_rules! sampling_planner {
    ($planner:ident, $core:ty { $($specific:tt)* }) => {
        #[pymethods]
        impl $planner {
        $($specific)*
        /// The occupancy map this planner checks against.
        #[getter]
        fn occupancy(&self, py: Python<'_>) -> Py<PyAny> {
            self.setup.occupancy.clone_ref(py)
        }

        /// The per-axis step, meters.
        #[getter]
        fn step_size<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
            as_array(py, &self.setup.step_size)
        }

        /// The sampling bounds, one pair per axis.
        #[getter]
        fn bounds(&self) -> Vec<(f64, f64)> {
            self.setup.bounds.clone()
        }

        /// Whether a plan runs entirely in Rust.
        ///
        /// False once any hook is a Python callable, the occupancy is one
        /// this binding could not rebuild, or a subclass replaced
        /// `distance` or `heuristic`. That is the condition
        /// `FR-PERF-03` asks callers to be able to check.
        #[getter]
        fn is_native(slf: &Bound<'_, Self>) -> PyResult<bool> {
            let planner = slf.borrow();
            Ok(planner.setup.is_native()
                && !planner
                    .setup
                    .is_overridden(slf.as_any(), &Self::type_object(slf.py()))?)
        }

        /// Why the last call found nothing, or `None`.
        #[getter]
        fn last_failure(&self) -> Option<PyPlanFailure> {
            LastCall::read(&self.last).failure
        }

        /// What the last successful path cost, or `None`.
        #[getter]
        fn last_cost(&self) -> Option<f64> {
            LastCall::read(&self.last).cost
        }

        /// How many samples the last call drew.
        #[getter]
        fn last_expanded(&self) -> usize {
            LastCall::read(&self.last).expanded
        }

        /// The generator this planner samples with.
        #[pyo3(text_signature = "()")]
        fn make_rng(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            self.setup.make_rng(py)
        }

        /// Publishes a telemetry snapshot through the configured sink.
        #[pyo3(text_signature = "(telemetry)")]
        fn publish_telemetry(&self, py: Python<'_>, telemetry: &Bound<'_, PyAny>) -> PyResult<()> {
            self.setup.publish(py, telemetry)
        }

        /// Step-size-normalized distance between two states.
        #[pyo3(text_signature = "(state_a, state_b)")]
        fn distance(
            &self,
            py: Python<'_>,
            state_a: &Bound<'_, PyAny>,
            state_b: &Bound<'_, PyAny>,
        ) -> PyResult<f64> {
            let (from, to) = (coordinates(state_a)?, coordinates(state_b)?);
            self.setup
                .cost_policy(py)
                .distance(&from, &to)
                .map_err(|failure| raised(&failure, &self.setup.failure))
        }

        /// Remaining-cost estimate between two states.
        #[pyo3(text_signature = "(state_a, state_b)")]
        fn heuristic(
            &self,
            py: Python<'_>,
            state_a: &Bound<'_, PyAny>,
            state_b: &Bound<'_, PyAny>,
        ) -> PyResult<f64> {
            let (from, to) = (coordinates(state_a)?, coordinates(state_b)?);
            self.setup
                .cost_policy(py)
                .heuristic(&from, &to)
                .map_err(|failure| raised(&failure, &self.setup.failure))
        }

        /// Draws one state, honoring a `sampler=` override.
        #[pyo3(text_signature = "(rng)")]
        fn sample<'py>(
            &self,
            py: Python<'py>,
            rng: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            // The public helper draws from the generator it was handed,
            // which is what a caller calling it directly expects. The
            // planner loop uses the native generator instead.
            if let Some(callable) = self.setup.sampler.as_ref() {
                return Ok(as_array(
                    py,
                    &coordinates(&callable.bind(py).call1((rng,))?)?,
                ));
            }
            let (low, high): (Vec<f64>, Vec<f64>) = self.setup.bounds.iter().copied().unzip();
            let drawn = rng.call_method1("uniform", (low, high))?;
            Ok(as_array(py, &coordinates(&drawn)?))
        }

        /// Steers one bounded step, honoring a `steerer=` override.
        #[pyo3(text_signature = "(from_pt, to_pt)")]
        fn steer<'py>(
            &self,
            py: Python<'py>,
            from_pt: &Bound<'py, PyAny>,
            to_pt: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let (from, to) = (coordinates(from_pt)?, coordinates(to_pt)?);
            let stepped = self
                .setup
                .steerer_policy(py)
                .steer(&from, &to)
                .map_err(|failure| raised(&failure, &self.setup.failure))?;
            Ok(as_array(py, &stepped))
        }

        /// Whether a segment is free, honoring a `segment_free=` override.
        #[pyo3(text_signature = "(a, b)")]
        fn is_segment_free(
            &self,
            py: Python<'_>,
            a: &Bound<'_, PyAny>,
            b: &Bound<'_, PyAny>,
        ) -> PyResult<bool> {
            let (from, to) = (coordinates(a)?, coordinates(b)?);
            self.setup
                .segment_policy(py)
                .is_segment_free(&from, &to)
                .map_err(|failure| raised(&failure, &self.setup.failure))
        }

        /// Runs the planner and reports the exploration tree with it.
        ///
        /// Returns `(nodes, parent, path)`, where `parent` maps each node
        /// index to the index it grew from and the root maps to `None`.
        /// SST reports only its active nodes, re-indexed, which is what
        /// keeps its tree the sparse one.
        #[pyo3(text_signature = "(start, goal)")]
        fn get_tree(
            slf: &Bound<'_, Self>,
            start: &Bound<'_, PyAny>,
            goal: &Bound<'_, PyAny>,
        ) -> PyResult<Py<PyTuple>> {
            let py = slf.py();
            let (from, to) = (coordinates(start)?, coordinates(goal)?);
            let (planner, mut generator, failure) = {
                let held = slf.borrow();
                let cost = held
                    .setup
                    .cost_policy_for(slf.as_any(), &Self::type_object(py))?;
                (
                    <$core>::new(
                        held.setup.sampler_policy(py)?,
                        held.setup.steerer_policy(py),
                        held.setup.segment_policy(py),
                        cost,
                        held.settings,
                    ),
                    seeded(py, held.setup.seed, held.setup.rng.as_ref())?,
                    held.setup.failure.clone(),
                )
            };
            let (outcome, tree) =
                detached(py, &failure, || planner.plan_tree(&from, &to, &mut generator))?;

            let nodes = as_path(py, tree.states())?;
            let parent = PyDict::new(py);
            for (index, grew_from) in tree.parents().iter().enumerate() {
                parent.set_item(index, *grew_from)?;
            }
            let path = match LastCall::record(&slf.borrow().last, outcome) {
                Some(found) => as_path(py, &found)?.unbind().into_any(),
                None => py.None(),
            };
            Ok(PyTuple::new(
                py,
                [nodes.unbind().into_any(), parent.unbind().into_any(), path],
            )?
            .unbind())
        }
        }
    };
}

/// Asymptotically optimal RRT\* over a continuous space.
#[pyclass(extends = PyContinuousPlanner, name = "RRTPlanner", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PyRrtPlanner {
    setup: SamplingSetup,
    settings: RrtSettings,
    last: Mutex<LastCall>,
}

sampling_planner!(PyRrtPlanner, arco_planning::continuous::RrtPlanner<SharedOccupancy> {
    /// Builds a planner over `occupancy`.
    #[new]
    #[pyo3(signature = (
        occupancy,
        bounds,
        max_sample_count = 2000,
        step_size = None,
        goal_tolerance = 1.0,
        rewire_radius = None,
        collision_check_count = 10,
        goal_bias = 0.05,
        early_stop = true,
        sampler = None,
        steerer = None,
        segment_free = None,
        cost = None,
        publisher = None,
        seed = None,
        rng = None,
    ))]
    #[pyo3(text_signature = "(occupancy, bounds, max_sample_count=2000, step_size=1.0, \
        goal_tolerance=1.0, rewire_radius=None, collision_check_count=10, goal_bias=0.05, \
        early_stop=True, sampler=None, steerer=None, segment_free=None, cost=None, \
        publisher=None, seed=None, rng=None)")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the count is the Python constructor's, which FR-API-02 pins"
    )]
    fn new(
        py: Python<'_>,
        occupancy: &Bound<'_, PyAny>,
        bounds: &Bound<'_, PyAny>,
        max_sample_count: usize,
        step_size: Option<&Bound<'_, PyAny>>,
        goal_tolerance: f64,
        rewire_radius: Option<f64>,
        collision_check_count: usize,
        goal_bias: f64,
        early_stop: bool,
        sampler: Option<Py<PyAny>>,
        steerer: Option<Py<PyAny>>,
        segment_free: Option<Py<PyAny>>,
        cost: Option<Py<PyAny>>,
        publisher: Option<Py<PyAny>>,
        seed: Option<u64>,
        rng: Option<Py<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let unit = 1.0_f64.into_pyobject(py)?;
        let continuous = PyContinuousPlanner::over(
            occupancy,
            cost.as_ref().map(|model| model.clone_ref(py)),
            publisher.as_ref().map(|sink| sink.clone_ref(py)),
            seed,
            rng.as_ref().map(|generator| generator.clone_ref(py)),
        );
        let setup = SamplingSetup::new(
            occupancy,
            bounds,
            step_size.unwrap_or(unit.as_any()),
            collision_check_count,
            sampler,
            steerer,
            segment_free,
            cost,
            publisher,
            seed,
            rng,
        )?;
        let defaults = RrtSettings::default();
        Ok(PyClassInitializer::from(PyCostModel)
            .add_subclass(continuous)
            .add_subclass(Self {
                setup,
                settings: RrtSettings {
                    max_samples: max_sample_count,
                    goal_tolerance,
                    goal_bias,
                    max_rewire_radius: defaults.max_rewire_radius,
                    fixed_rewire_radius: rewire_radius,
                    early_stop,
                },
                last: Mutex::new(LastCall::default()),
            }))
    }

    /// The maximum number of samples a plan may draw.
    #[getter]
    const fn max_sample_count(&self) -> usize {
        self.settings.max_samples
    }

    /// How close to the goal counts as arriving, in steps.
    #[getter]
    const fn goal_tolerance(&self) -> f64 {
        self.settings.goal_tolerance
    }

    /// How often the goal itself is sampled.
    #[getter]
    const fn goal_bias(&self) -> f64 {
        self.settings.goal_bias
    }

    /// Whether the planner stops at the first solution.
    #[getter]
    const fn early_stop(&self) -> bool {
        self.settings.early_stop
    }

    /// Segment resolution for the built-in collision check.
    #[getter]
    const fn collision_check_count(&self) -> usize {
        self.setup.collision_check_count
    }

    /// Plans a collision-free path from `start` to `goal`.
    ///
    /// Returns the waypoints, or `None` when no path was found within
    /// `max_sample_count`. `last_failure` says which reason applied.
    #[pyo3(text_signature = "(start, goal)")]
    fn plan(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyList>>> {
        let py = slf.py();
        let (from, to) = (coordinates(start)?, coordinates(goal)?);
        let (planner, mut generator, failure) = {
            let held = slf.borrow();
            let cost = held
                .setup
                .cost_policy_for(slf.as_any(), &Self::type_object(py))?;
            (
                arco_planning::continuous::RrtPlanner::new(
                    held.setup.sampler_policy(py)?,
                    held.setup.steerer_policy(py),
                    held.setup.segment_policy(py),
                    cost,
                    held.settings,
                ),
                seeded(py, held.setup.seed, held.setup.rng.as_ref())?,
                held.setup.failure.clone(),
            )
        };
        let publisher = slf.borrow().setup.publisher.as_ref().map(|sink| sink.clone_ref(py));
        let outcome = detached(py, &failure, || {
            planner
                .plan_observed(&from, &to, &mut generator, &mut |progress| {
                    report_progress(publisher.as_ref(), "RRT*", &progress);
                })
                .map(|(outcome, _tree)| outcome)
        })?;
        match LastCall::record(&slf.borrow().last, outcome) {
            Some(path) => Ok(Some(as_path(py, &path)?.unbind())),
            None => Ok(None),
        }
    }
});

/// Sparse stable trees over a continuous space.
#[pyclass(extends = PyContinuousPlanner, name = "SSTPlanner", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PySstPlanner {
    setup: SamplingSetup,
    settings: SstSettings,
    last: Mutex<LastCall>,
}

sampling_planner!(PySstPlanner, arco_planning::continuous::SstPlanner<SharedOccupancy> {
    /// Builds a planner over `occupancy`.
    #[new]
    #[pyo3(signature = (
        occupancy,
        bounds,
        max_sample_count = 3000,
        step_size = None,
        goal_tolerance = 1.0,
        witness_radius = 0.5,
        collision_check_count = 10,
        goal_bias = 0.05,
        early_stop = true,
        sampler = None,
        steerer = None,
        segment_free = None,
        cost = None,
        publisher = None,
        seed = None,
        rng = None,
    ))]
    #[pyo3(text_signature = "(occupancy, bounds, max_sample_count=3000, step_size=1.0, \
        goal_tolerance=1.0, witness_radius=0.5, collision_check_count=10, goal_bias=0.05, \
        early_stop=True, sampler=None, steerer=None, segment_free=None, cost=None, \
        publisher=None, seed=None, rng=None)")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the count is the Python constructor's, which FR-API-02 pins"
    )]
    fn new(
        py: Python<'_>,
        occupancy: &Bound<'_, PyAny>,
        bounds: &Bound<'_, PyAny>,
        max_sample_count: usize,
        step_size: Option<&Bound<'_, PyAny>>,
        goal_tolerance: f64,
        witness_radius: f64,
        collision_check_count: usize,
        goal_bias: f64,
        early_stop: bool,
        sampler: Option<Py<PyAny>>,
        steerer: Option<Py<PyAny>>,
        segment_free: Option<Py<PyAny>>,
        cost: Option<Py<PyAny>>,
        publisher: Option<Py<PyAny>>,
        seed: Option<u64>,
        rng: Option<Py<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let unit = 1.0_f64.into_pyobject(py)?;
        let continuous = PyContinuousPlanner::over(
            occupancy,
            cost.as_ref().map(|model| model.clone_ref(py)),
            publisher.as_ref().map(|sink| sink.clone_ref(py)),
            seed,
            rng.as_ref().map(|generator| generator.clone_ref(py)),
        );
        let setup = SamplingSetup::new(
            occupancy,
            bounds,
            step_size.unwrap_or(unit.as_any()),
            collision_check_count,
            sampler,
            steerer,
            segment_free,
            cost,
            publisher,
            seed,
            rng,
        )?;
        Ok(PyClassInitializer::from(PyCostModel)
            .add_subclass(continuous)
            .add_subclass(Self {
                setup,
                settings: SstSettings {
                    max_samples: max_sample_count,
                    goal_tolerance,
                    witness_radius,
                    goal_bias,
                    early_stop,
                },
                last: Mutex::new(LastCall::default()),
            }))
    }

    /// The maximum number of samples a plan may draw.
    #[getter]
    const fn max_sample_count(&self) -> usize {
        self.settings.max_samples
    }

    /// How close to the goal counts as arriving, in steps.
    #[getter]
    const fn goal_tolerance(&self) -> f64 {
        self.settings.goal_tolerance
    }

    /// Half-width of a witness cell, in steps.
    #[getter]
    const fn witness_radius(&self) -> f64 {
        self.settings.witness_radius
    }

    /// How often the goal itself is sampled.
    #[getter]
    const fn goal_bias(&self) -> f64 {
        self.settings.goal_bias
    }

    /// Whether the planner stops at the first solution.
    #[getter]
    const fn early_stop(&self) -> bool {
        self.settings.early_stop
    }

    /// Segment resolution for the built-in collision check.
    #[getter]
    const fn collision_check_count(&self) -> usize {
        self.setup.collision_check_count
    }

    /// Plans a collision-free path from `start` to `goal`.
    ///
    /// Returns the waypoints, or `None` when no path was found within
    /// `max_sample_count`. `last_failure` says which reason applied.
    ///
    /// Raises `ValueError` when `witness_radius` is not inside one step,
    /// which deviation A-14 records: at a whole step the tree cannot
    /// grow and the planner would spend its budget rejecting candidates.
    #[pyo3(text_signature = "(start, goal)")]
    fn plan(
        slf: &Bound<'_, Self>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyList>>> {
        let py = slf.py();
        let (from, to) = (coordinates(start)?, coordinates(goal)?);
        let (planner, mut generator, failure) = {
            let held = slf.borrow();
            let cost = held
                .setup
                .cost_policy_for(slf.as_any(), &Self::type_object(py))?;
            (
                arco_planning::continuous::SstPlanner::new(
                    held.setup.sampler_policy(py)?,
                    held.setup.steerer_policy(py),
                    held.setup.segment_policy(py),
                    cost,
                    held.settings,
                ),
                seeded(py, held.setup.seed, held.setup.rng.as_ref())?,
                held.setup.failure.clone(),
            )
        };
        let publisher = slf.borrow().setup.publisher.as_ref().map(|sink| sink.clone_ref(py));
        let outcome = detached(py, &failure, || {
            planner
                .plan_observed(&from, &to, &mut generator, &mut |progress| {
                    report_progress(publisher.as_ref(), "SST", &progress);
                })
                .map(|(outcome, _tree)| outcome)
        })?;
        match LastCall::record(&slf.borrow().last, outcome) {
            Some(path) => Ok(Some(as_path(py, &path)?.unbind())),
            None => Ok(None),
        }
    }
});

// ---------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------

/// Reduces a raw path to the fewest waypoints that still connect.
#[pyclass(name = "TrajectoryPruner", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PyTrajectoryPruner {
    occupancy: Py<PyAny>,
    shared: SharedOccupancy,
    step_size: Vec<f64>,
    failure: FailureSlot,
    /// Minimum sample count per segment for the built-in check.
    #[pyo3(get)]
    collision_check_count: usize,
}

#[pymethods]
impl PyTrajectoryPruner {
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

    /// Builds a pruner over `occupancy`.
    #[new]
    #[pyo3(signature = (occupancy, step_size, collision_check_count = 10))]
    #[pyo3(text_signature = "(occupancy, step_size, collision_check_count=10)")]
    fn new(
        occupancy: &Bound<'_, PyAny>,
        step_size: &Bound<'_, PyAny>,
        // Signed, because Python raised `ValueError` for a negative count
        // and `usize` makes PyO3 reject it with `OverflowError` before the
        // check below ever runs.
        collision_check_count: i64,
    ) -> PyResult<Self> {
        if collision_check_count < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "collision_check_count must be at least 1; got {collision_check_count}."
            )));
        }
        let read = coordinates(step_size).map_err(|_not_a_vector| {
            pyo3::exceptions::PyValueError::new_err(
                "step_size must be a non-empty 1-D array; got a scalar.",
            )
        })?;
        if read.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "step_size must be a non-empty 1-D array; got shape (0,).",
            ));
        }
        if read
            .iter()
            .any(|scale| !(scale.is_finite() && *scale > 0.0))
        {
            let shown = step_size.repr()?;
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "step_size elements must be strictly positive; got {shown}."
            )));
        }
        let failure = FailureSlot::default();
        Ok(Self {
            occupancy: occupancy.clone().unbind(),
            shared: SharedOccupancy::new(BoundOccupancy::adopt(occupancy, failure.clone())),
            step_size: read,
            failure,
            collision_check_count: usize::try_from(collision_check_count).unwrap_or_default(),
        })
    }

    /// The occupancy map this pruner checks against.
    #[getter]
    fn occupancy(&self, py: Python<'_>) -> Py<PyAny> {
        self.occupancy.clone_ref(py)
    }

    /// The per-axis planner step, meters.
    #[getter]
    fn step_size<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        as_array(py, &self.step_size)
    }

    /// Returns the fewest waypoints of `path` that still connect.
    ///
    /// The default check is the exact one, which is what the Python
    /// pruner's adaptive sample density was reaching for and what
    /// deviation A-12 makes available. A `steer=` callable replaces it
    /// and is invoked once per candidate shortcut, through the
    /// interpreter.
    #[pyo3(signature = (path, steer = None))]
    #[pyo3(text_signature = "(path, steer=None)")]
    fn prune(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        steer: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyList>> {
        let waypoints = path
            .try_iter()?
            .map(|point| coordinates(&point?))
            .collect::<PyResult<Vec<Vec<f64>>>>()?;
        // Exact is the native branch and the default; a supplied `steer`
        // crosses into the interpreter per candidate shortcut, per A-07.
        let segments = steer.map_or_else(
            || SegmentPolicy::Exact {
                occupancy: self.shared.clone(),
            },
            |callable| {
                SegmentPolicy::Custom(Box::new(PySegmentChecker::new(
                    callable,
                    self.failure.clone(),
                )))
            },
        );
        let pruner = arco_planning::continuous::TrajectoryPruner::new(segments);
        let kept = detached(py, &self.failure, || {
            pruner.shortest_subsequence(&waypoints)
        })?;
        Ok(as_path(py, &kept)?.unbind())
    }
}

// ---------------------------------------------------------------------
// Trajectory optimization
// ---------------------------------------------------------------------

/// What a trajectory optimization produced.
///
/// A dataclass on the Python side, so every field reads and writes, and
/// `==` and `repr` compare and print the nine declared fields in order.
/// `turn_rates` is published beside them and left out of both, since
/// adding a field to either would change what a caller already compares.
/// One of the five default cost terms, as a handle carrying its name.
///
/// Python built five term objects in the constructor and a caller reads
/// `[t.name for t in optimizer.cost_terms]`. The terms themselves are an
/// enum inside the crate with no Python representation, so what crosses
/// the boundary is the name and nothing else. Passing one back into
/// `cost_terms=` is refused rather than silently ignored.
#[pyclass(
    frozen,
    skip_from_py_object,
    name = "DefaultCostTerm",
    module = "arco._arco"
)]
#[derive(Debug, Clone)]
pub(crate) struct PyDefaultTerm {
    /// Which of the five this is.
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl PyDefaultTerm {
    fn __repr__(&self) -> String {
        format!("DefaultCostTerm(name={:?})", self.name)
    }
}

// ---------------------------------------------------------------------
// Standalone cost terms (arco.planning.continuous.cost_terms)
// ---------------------------------------------------------------------
//
// These five classes are the ones a caller could already build by hand
// and pass through `cost_terms=`, kept at their historical names,
// constructor arguments and `name` attributes so an existing call site
// still works. `TrajectoryOptimizer` itself never builds one: its own
// five defaults stay the cheaper `TrajectoryTerm` enum inside the crate,
// reported through `PyDefaultTerm` above. An instance of one of the
// classes below reaches the optimizer through the same path any other
// Python callable does, `cost_terms=[...]`, evaluated once per term per
// cost call.

/// Reads `context[key]`, raising the `KeyError` a dict subscript would.
///
/// # Errors
///
/// Returns a `KeyError` naming `key` when it is missing from `context`.
fn context_item<'py>(context: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    context
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))
}

/// The interior of `items`, dropping the first and last element.
///
/// Mirrors Python's `items[1:-1]`. Fewer than two elements have no
/// interior, so the result is empty rather than negative-length.
fn interior<T>(items: &[T]) -> &[T] {
    let len = items.len();
    if len < 2 {
        return &[];
    }
    let Some(end) = len.checked_sub(1) else {
        return &[];
    };
    items.get(1..end).unwrap_or(&[])
}

/// Penalizes total traversal time squared: `weight * total_time ** 2`.
///
/// One of the five terms `build_default_cost_terms` returns and the
/// term the optimizer exists to reduce; left unopposed it drives the
/// total time to zero, which the velocity term stops.
#[pyclass(skip_from_py_object, name = "TimeCostTerm", module = "arco._arco")]
#[derive(Debug, Clone)]
pub(crate) struct PyTimeCostTerm {
    /// Multiplier for the squared total duration.
    #[pyo3(get, set)]
    weight: f64,
}

#[pymethods]
impl PyTimeCostTerm {
    /// Builds the time cost term.
    #[new]
    #[pyo3(text_signature = "(weight)")]
    fn new(weight: f64) -> Self {
        Self { weight }
    }

    /// This term's name in a composite cost's `cost_terms` list.
    #[getter]
    #[expect(
        clippy::unused_self,
        reason = "the name is a class-level constant Python reads as an instance attribute"
    )]
    const fn name(&self) -> &'static str {
        "time"
    }

    /// Evaluates the time cost.
    ///
    /// Reads `durations` off `context` and returns the weighted square
    /// of their sum.
    fn __call__(&self, context: &Bound<'_, PyDict>) -> PyResult<f64> {
        let durations: Vec<f64> = context_item(context, "durs")?.extract()?;
        let total: f64 = durations.iter().sum();
        Ok(self.weight * total * total)
    }

    fn __repr__(&self) -> String {
        format!("TimeCostTerm(weight={:?})", self.weight)
    }
}

/// Penalizes squared deviation of interior waypoints from the reference.
#[pyclass(skip_from_py_object, name = "DeviationCostTerm", module = "arco._arco")]
#[derive(Debug, Clone)]
pub(crate) struct PyDeviationCostTerm {
    /// Multiplier for the summed squared deviation.
    #[pyo3(get, set)]
    weight: f64,
}

#[pymethods]
impl PyDeviationCostTerm {
    /// Builds the deviation cost term.
    #[new]
    #[pyo3(text_signature = "(weight)")]
    fn new(weight: f64) -> Self {
        Self { weight }
    }

    /// This term's name in a composite cost's `cost_terms` list.
    #[getter]
    #[expect(
        clippy::unused_self,
        reason = "the name is a class-level constant Python reads as an instance attribute"
    )]
    const fn name(&self) -> &'static str {
        "deviation"
    }

    /// Evaluates the path-deviation cost.
    ///
    /// Reads `segment_count`, `pts` and `ref` off `context`, and returns
    /// the weighted sum of squared interior deviations, or zero when
    /// there are no interior waypoints.
    fn __call__(&self, context: &Bound<'_, PyDict>) -> PyResult<f64> {
        let segment_count: usize = context_item(context, "segment_count")?.extract()?;
        if segment_count.saturating_sub(1) == 0 {
            return Ok(0.0);
        }
        let waypoints = point_rows(&context_item(context, "pts")?)?;
        let reference = point_rows(&context_item(context, "ref")?)?;
        let mut total = 0.0;
        for (moved, original) in interior(&waypoints).iter().zip(interior(&reference)) {
            for (a, b) in moved.iter().zip(original) {
                let offset = a - b;
                total += offset * offset;
            }
        }
        Ok(self.weight * total)
    }

    fn __repr__(&self) -> String {
        format!("DeviationCostTerm(weight={:?})", self.weight)
    }
}

/// Penalizes squared deviation of segment speeds from cruise speed.
#[pyclass(skip_from_py_object, name = "VelocityCostTerm", module = "arco._arco")]
#[derive(Debug, Clone)]
pub(crate) struct PyVelocityCostTerm {
    /// Multiplier for the summed squared speed error.
    #[pyo3(get, set)]
    weight: f64,
    /// Target traversal speed, world units per second.
    #[pyo3(get, set)]
    cruise_speed: f64,
}

#[pymethods]
impl PyVelocityCostTerm {
    /// Builds the velocity cost term.
    #[new]
    #[pyo3(text_signature = "(weight, cruise_speed)")]
    fn new(weight: f64, cruise_speed: f64) -> Self {
        Self {
            weight,
            cruise_speed,
        }
    }

    /// This term's name in a composite cost's `cost_terms` list.
    #[getter]
    #[expect(
        clippy::unused_self,
        reason = "the name is a class-level constant Python reads as an instance attribute"
    )]
    const fn name(&self) -> &'static str {
        "velocity"
    }

    /// Evaluates the velocity-tracking cost.
    ///
    /// Reads `speeds` off `context` and returns the weighted sum of
    /// squared `speed - cruise_speed` errors.
    fn __call__(&self, context: &Bound<'_, PyDict>) -> PyResult<f64> {
        let speeds: Vec<f64> = context_item(context, "speeds")?.extract()?;
        let total: f64 = speeds
            .iter()
            .map(|speed| {
                let offset = speed - self.cruise_speed;
                offset * offset
            })
            .sum();
        Ok(self.weight * total)
    }

    fn __repr__(&self) -> String {
        format!(
            "VelocityCostTerm(weight={:?}, cruise_speed={:?})",
            self.weight, self.cruise_speed
        )
    }
}

/// Soft clearance penalty plus barrier-style penetration growth.
///
/// Combines the quadratic clearance violation with the scaled barrier
/// term the optimizer historically summed alongside it.
#[pyclass(skip_from_py_object, name = "CollisionCostTerm", module = "arco._arco")]
#[derive(Debug, Clone)]
pub(crate) struct PyCollisionCostTerm {
    /// Multiplier applied to both the soft and the barrier penalty.
    #[pyo3(get, set)]
    weight: f64,
    /// Extra multiplier on the barrier contribution.
    #[pyo3(get, set)]
    barrier_scale: f64,
    /// Exponent on normalized penetration depth.
    #[pyo3(get, set)]
    barrier_power: f64,
}

#[pymethods]
impl PyCollisionCostTerm {
    /// Builds the collision cost term.
    #[new]
    #[pyo3(signature = (weight, barrier_scale = 50.0, barrier_power = 4.0))]
    #[pyo3(text_signature = "(weight, barrier_scale=50.0, barrier_power=4.0)")]
    fn new(weight: f64, barrier_scale: f64, barrier_power: f64) -> Self {
        Self {
            weight,
            barrier_scale,
            barrier_power,
        }
    }

    /// This term's name in a composite cost's `cost_terms` list.
    #[getter]
    #[expect(
        clippy::unused_self,
        reason = "the name is a class-level constant Python reads as an instance attribute"
    )]
    const fn name(&self) -> &'static str {
        "collision"
    }

    /// Evaluates soft collision plus barrier penalties.
    ///
    /// Reads `pts`, `segment_count`, `occupancy` and `sample_count` off
    /// `context`. Queries `occupancy.query_distances` when it publishes
    /// one, and falls back to calling `occupancy.nearest_obstacle` per
    /// point otherwise, matching the historical Python behavior for a
    /// caller-supplied occupancy that offers only the slower method.
    fn __call__(&self, py: Python<'_>, context: &Bound<'_, PyDict>) -> PyResult<f64> {
        let waypoints = point_rows(&context_item(context, "pts")?)?;
        let segment_count: usize = context_item(context, "segment_count")?.extract()?;
        let occupancy = context_item(context, "occupancy")?;
        let sample_count: usize = context_item(context, "sample_count")?.extract()?;

        let clearance = occupancy
            .getattr("clearance")
            .ok()
            .and_then(|value| value.extract::<f64>().ok())
            .unwrap_or(0.5);

        let mut queried: Vec<Vec<f64>> = Vec::new();
        if segment_count.saturating_sub(1) > 0 {
            queried.extend(interior(&waypoints).iter().cloned());
        }
        if sample_count > 0 {
            let divisor =
                f64::from(u32::try_from(sample_count.saturating_add(1)).unwrap_or(u32::MAX));
            for index in 0..segment_count {
                let (Some(from), Some(to)) =
                    (waypoints.get(index), waypoints.get(index.saturating_add(1)))
                else {
                    continue;
                };
                for step in 1..=sample_count {
                    let ratio = f64::from(u32::try_from(step).unwrap_or(1)) / divisor;
                    let sample: Vec<f64> = from
                        .iter()
                        .zip(to)
                        .map(|(start, end)| start + ratio * (end - start))
                        .collect();
                    queried.push(sample);
                }
            }
        }

        if queried.is_empty() {
            return Ok(0.0);
        }

        let distances: Vec<f64> = if occupancy.hasattr("query_distances")? {
            let batch = PyArray2::from_vec2(py, &queried).map_err(|_fault| {
                pyo3::exceptions::PyValueError::new_err(
                    "query points do not form a rectangular array.",
                )
            })?;
            occupancy
                .call_method1("query_distances", (batch,))?
                .extract()?
        } else {
            let mut found = Vec::with_capacity(queried.len());
            for point in &queried {
                let answer = occupancy.call_method1("nearest_obstacle", (as_array(py, point),))?;
                found.push(answer.get_item(0)?.extract::<f64>()?);
            }
            found
        };

        let clearance_safe = clearance.max(1e-9);
        let mut quadratic = 0.0;
        let mut barrier = 0.0;
        for distance in distances {
            let penetration = (clearance - distance).max(0.0);
            quadratic += penetration * penetration;
            barrier += (penetration / clearance_safe).powf(self.barrier_power);
        }

        Ok(self.weight * quadratic + self.weight * self.barrier_scale * barrier)
    }

    fn __repr__(&self) -> String {
        format!(
            "CollisionCostTerm(weight={:?}, barrier_scale={:?}, barrier_power={:?})",
            self.weight, self.barrier_scale, self.barrier_power
        )
    }
}

/// Penalizes implied segment speeds outside `[min_speed, max_speed]`.
#[pyclass(skip_from_py_object, name = "DynamicsCostTerm", module = "arco._arco")]
#[derive(Debug, Clone)]
pub(crate) struct PyDynamicsCostTerm {
    /// Multiplier for the summed squared bound violations.
    #[pyo3(get, set)]
    weight: f64,
    /// Upper speed limit, world units per second, when there is one.
    #[pyo3(get, set)]
    max_speed: Option<f64>,
    /// Lower speed limit, world units per second, when there is one.
    #[pyo3(get, set)]
    min_speed: Option<f64>,
}

#[pymethods]
impl PyDynamicsCostTerm {
    /// Builds the dynamics cost term.
    #[new]
    #[pyo3(signature = (weight, max_speed = None, min_speed = None))]
    #[pyo3(text_signature = "(weight, max_speed=None, min_speed=None)")]
    fn new(weight: f64, max_speed: Option<f64>, min_speed: Option<f64>) -> Self {
        Self {
            weight,
            max_speed,
            min_speed,
        }
    }

    /// This term's name in a composite cost's `cost_terms` list.
    #[getter]
    #[expect(
        clippy::unused_self,
        reason = "the name is a class-level constant Python reads as an instance attribute"
    )]
    const fn name(&self) -> &'static str {
        "dynamics"
    }

    /// Evaluates the dynamics-bound penalty.
    ///
    /// Reads `speeds` off `context` and returns the weighted sum of
    /// squared speed-bound violations, or zero when both bounds are
    /// unset.
    fn __call__(&self, context: &Bound<'_, PyDict>) -> PyResult<f64> {
        if self.max_speed.is_none() && self.min_speed.is_none() {
            return Ok(0.0);
        }
        let speeds: Vec<f64> = context_item(context, "speeds")?.extract()?;
        let mut total = 0.0;
        if let Some(limit) = self.max_speed {
            total += speeds
                .iter()
                .map(|speed| (speed - limit).max(0.0).powi(2))
                .sum::<f64>();
        }
        if let Some(limit) = self.min_speed {
            total += speeds
                .iter()
                .map(|speed| (limit - speed).max(0.0).powi(2))
                .sum::<f64>();
        }
        Ok(total * self.weight)
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicsCostTerm(weight={:?}, max_speed={:?}, min_speed={:?})",
            self.weight, self.max_speed, self.min_speed
        )
    }
}

/// Builds the five historical default optimizer cost terms.
///
/// Returns an ordered list of five cost term instances, `TimeCostTerm`,
/// `DeviationCostTerm`, `VelocityCostTerm`, `CollisionCostTerm` and
/// `DynamicsCostTerm`, matching the composition order the optimizer has
/// always summed them in. Passing the result back through a
/// `TrajectoryOptimizer`'s `cost_terms=` reproduces its own defaults,
/// only evaluated through the slower, caller-supplied path rather than
/// the optimizer's native one.
#[pyfunction]
#[pyo3(signature = (
    *,
    weight_time,
    weight_deviation,
    weight_velocity,
    weight_collision,
    weight_dynamics,
    cruise_speed,
    collision_barrier_scale,
    collision_barrier_power,
    max_speed,
    min_speed,
))]
#[pyo3(
    text_signature = "(*, weight_time, weight_deviation, weight_velocity, \
    weight_collision, weight_dynamics, cruise_speed, collision_barrier_scale, \
    collision_barrier_power, max_speed, min_speed)"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "one per historical keyword-only parameter"
)]
fn build_default_cost_terms(
    py: Python<'_>,
    weight_time: f64,
    weight_deviation: f64,
    weight_velocity: f64,
    weight_collision: f64,
    weight_dynamics: f64,
    cruise_speed: f64,
    collision_barrier_scale: f64,
    collision_barrier_power: f64,
    max_speed: Option<f64>,
    min_speed: Option<f64>,
) -> PyResult<Py<PyList>> {
    let terms = [
        Py::new(
            py,
            PyTimeCostTerm {
                weight: weight_time,
            },
        )?
        .into_any(),
        Py::new(
            py,
            PyDeviationCostTerm {
                weight: weight_deviation,
            },
        )?
        .into_any(),
        Py::new(
            py,
            PyVelocityCostTerm {
                weight: weight_velocity,
                cruise_speed,
            },
        )?
        .into_any(),
        Py::new(
            py,
            PyCollisionCostTerm {
                weight: weight_collision,
                barrier_scale: collision_barrier_scale,
                barrier_power: collision_barrier_power,
            },
        )?
        .into_any(),
        Py::new(
            py,
            PyDynamicsCostTerm {
                weight: weight_dynamics,
                max_speed,
                min_speed,
            },
        )?
        .into_any(),
    ];
    Ok(PyList::new(py, terms)?.unbind())
}

#[pyclass(get_all, set_all, name = "TrajectoryResult", module = "arco._arco")]
#[derive(Debug)]
pub(crate) struct PyTrajectoryResult {
    /// The optimized waypoints, endpoints included and unmoved.
    states: Py<PyList>,
    /// One command per segment.
    commands: Py<PyList>,
    /// One duration per segment, seconds.
    durations: Vec<f64>,
    /// The composite cost at the returned solution.
    cost: f64,
    /// Whether every state passed the feasibility policy.
    is_feasible: bool,
    /// Whether the solver reached its tolerance rather than its budget.
    optimizer_success: bool,
    /// Zero when the solver converged, one when it did not.
    optimizer_status_code: i32,
    /// The solver's exit condition, written as a sentence.
    optimizer_status_text: String,
    /// How many iterations the solver ran.
    optimizer_iteration_count: usize,
    /// The turn rate each segment implies, radians per second.
    ///
    /// Published beside `commands` rather than inside it: the Python
    /// optimizer wrote a zero turn rate into every command it estimated,
    /// and a caller reading `commands[i][1]` is reading that zero.
    turn_rates: Vec<f64>,
}

#[pymethods]
impl PyTrajectoryResult {
    /// Builds a result directly, as the Python dataclass allowed.
    ///
    /// Every field defaults, because `TrajectoryResult()` with no
    /// arguments is what the dataclass gave and what the pipeline tests
    /// build when they stand in for an optimizer.
    #[new]
    #[pyo3(signature = (
        states = None,
        commands = None,
        durations = None,
        cost = 0.0,
        is_feasible = true,
        optimizer_success = true,
        optimizer_status_code = 0,
        optimizer_status_text = String::new(),
        optimizer_iteration_count = 0,
        turn_rates = None,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one per dataclass field")]
    fn new(
        py: Python<'_>,
        states: Option<Py<PyList>>,
        commands: Option<Py<PyList>>,
        durations: Option<Vec<f64>>,
        cost: f64,
        is_feasible: bool,
        optimizer_success: bool,
        optimizer_status_code: i32,
        optimizer_status_text: String,
        optimizer_iteration_count: usize,
        turn_rates: Option<Vec<f64>>,
    ) -> Self {
        Self {
            states: states.unwrap_or_else(|| PyList::empty(py).unbind()),
            commands: commands.unwrap_or_else(|| PyList::empty(py).unbind()),
            durations: durations.unwrap_or_default(),
            cost,
            is_feasible,
            optimizer_success,
            optimizer_status_code,
            optimizer_status_text,
            optimizer_iteration_count,
            turn_rates: turn_rates.unwrap_or_default(),
        }
    }

    /// The result, written the way the dataclass printed it.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let mut written = String::from("TrajectoryResult(");
        for (position, field) in DATACLASS_FIELDS.iter().enumerate() {
            if position > 0 {
                written.push_str(", ");
            }
            let value = slf.as_any().getattr(*field)?.repr()?;
            written.push_str(field);
            written.push('=');
            written.push_str(&value.to_cow()?);
        }
        written.push(')');
        Ok(written)
    }

    /// Field-by-field comparison, the way the dataclass compared.
    ///
    /// Compares the declared fields as one tuple, which is what the
    /// generated `__eq__` did, so a field holding an array raises here
    /// exactly where it raised before.
    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if !other.is_instance(&slf.as_any().get_type())? {
            return Ok(py.NotImplemented());
        }
        let mine = Self::fields(slf.as_any())?;
        let theirs = Self::fields(other)?;
        Ok(mine
            .bind(py)
            .rich_compare(theirs.bind(py), pyo3::basic::CompareOp::Eq)?
            .unbind())
    }
}

/// The dataclass fields, in declaration order.
///
/// `turn_rates` is deliberately absent: it is published beside the
/// dataclass rather than as part of it.
const DATACLASS_FIELDS: [&str; 9] = [
    "states",
    "commands",
    "durations",
    "cost",
    "is_feasible",
    "optimizer_success",
    "optimizer_status_code",
    "optimizer_status_text",
    "optimizer_iteration_count",
];

impl PyTrajectoryResult {
    /// The declared fields of `object`, as one tuple.
    ///
    /// # Errors
    ///
    /// Returns whatever reading a field raised.
    fn fields(object: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        let read = DATACLASS_FIELDS
            .iter()
            .map(|field| object.getattr(*field))
            .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?;
        Ok(PyTuple::new(object.py(), read)?.unbind())
    }
}

/// Refines a reference path into a timed trajectory.
#[pyclass(name = "TrajectoryOptimizer", module = "arco._arco", subclass)]
#[derive(Debug)]
pub(crate) struct PyTrajectoryOptimizer {
    occupancy: Py<PyAny>,
    shared: SharedOccupancy,
    terms: Option<Py<PyList>>,
    weights: TermWeights,
    barrier: (f64, f64),
    speed_band: (Option<f64>, Option<f64>),
    sample_count: usize,
    settings: OptimizerSettings,
    failure: FailureSlot,
    /// The `scipy` method name the Python optimizer forwarded.
    ///
    /// Kept so the attribute still answers. Deviation A-08 replaced the
    /// solver with `argmin`, which offers one quasi-Newton method, so the
    /// value is recorded and not acted on.
    #[pyo3(get, set)]
    method: String,
}

#[pymethods]
impl PyTrajectoryOptimizer {
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

    /// Builds an optimizer over `occupancy`.
    #[new]
    #[pyo3(signature = (
        occupancy,
        cruise_speed = 1.0,
        weight_time = 10.0,
        weight_deviation = 1.0,
        weight_velocity = 1.0,
        weight_collision = 5.0,
        collision_barrier_scale = 50.0,
        collision_barrier_power = 4.0,
        weight_dynamics = 100.0,
        max_speed = None,
        min_speed = None,
        time_relaxation = 1.5,
        method = "L-BFGS-B".to_owned(),
        sample_count = 3,
        max_iter = 500,
        ftol = 1e-9,
        cost_terms = None,
    ))]
    #[pyo3(text_signature = "(occupancy, cruise_speed=1.0, weight_time=10.0, \
        weight_deviation=1.0, weight_velocity=1.0, weight_collision=5.0, \
        collision_barrier_scale=50.0, collision_barrier_power=4.0, weight_dynamics=100.0, \
        max_speed=None, min_speed=None, time_relaxation=1.5, method='L-BFGS-B', \
        sample_count=3, max_iter=500, ftol=1e-09, cost_terms=None)")]
    #[expect(
        clippy::too_many_arguments,
        reason = "the count is the Python constructor's, which FR-API-02 pins"
    )]
    fn new(
        py: Python<'_>,
        occupancy: &Bound<'_, PyAny>,
        cruise_speed: f64,
        weight_time: f64,
        weight_deviation: f64,
        weight_velocity: f64,
        weight_collision: f64,
        collision_barrier_scale: f64,
        collision_barrier_power: f64,
        weight_dynamics: f64,
        max_speed: Option<f64>,
        min_speed: Option<f64>,
        time_relaxation: f64,
        method: String,
        sample_count: usize,
        max_iter: u64,
        ftol: f64,
        cost_terms: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !(cruise_speed.is_finite() && cruise_speed > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cruise_speed must be positive, got {cruise_speed}."
            )));
        }
        let failure = FailureSlot::default();
        let terms = cost_terms
            .map(|listed| -> PyResult<Py<PyList>> {
                Ok(PyList::new(py, listed.try_iter()?.collect::<PyResult<Vec<_>>>()?)?.unbind())
            })
            .transpose()?;
        let defaults = OptimizerSettings::default();
        Ok(Self {
            occupancy: occupancy.clone().unbind(),
            shared: SharedOccupancy::new(BoundOccupancy::adopt(occupancy, failure.clone())),
            terms,
            weights: TermWeights {
                time: weight_time,
                deviation: weight_deviation,
                velocity: weight_velocity,
                collision: weight_collision,
                dynamics: weight_dynamics,
            },
            barrier: (collision_barrier_scale, collision_barrier_power),
            speed_band: (max_speed, min_speed),
            sample_count,
            settings: OptimizerSettings {
                cruise_speed,
                time_relaxation,
                max_iterations: max_iter,
                gradient_tolerance: defaults.gradient_tolerance,
                cost_tolerance: ftol,
                memory: defaults.memory,
            },
            failure,
            method,
        })
    }

    /// Builds an optimizer from `config/optimizer.yml`.
    #[staticmethod]
    #[pyo3(signature = (occupancy, cruise_speed, max_speed = None, min_speed = None))]
    #[pyo3(text_signature = "(occupancy, cruise_speed, max_speed=None, min_speed=None)")]
    fn create_from_config(
        py: Python<'_>,
        occupancy: &Bound<'_, PyAny>,
        cruise_speed: f64,
        max_speed: Option<f64>,
        min_speed: Option<f64>,
    ) -> PyResult<Self> {
        let config = py
            .import("arco.config")?
            .call_method1("load_config", ("optimizer",))?;
        let weight = required(&config, "weight")?;
        let barrier = required(&config, "collision_barrier")?;
        let method = required(&config, "method")?;

        Self::new(
            py,
            occupancy,
            cruise_speed,
            number(&weight, "time", 1e2)?,
            number(&weight, "deviation", 1e-2)?,
            number(&weight, "velocity", 1e0)?,
            number(&weight, "collision", 1e4)?,
            number(&barrier, "scale", 50.0)?,
            number(&barrier, "power", 4.0)?,
            number(&weight, "dynamics", 1e2)?,
            max_speed,
            min_speed,
            number(&method, "time_relaxation", 1.5)?,
            text(&method, "name", "L-BFGS-B")?,
            count(&method, "sample_count", 3)?,
            count(&method, "max_iter", 30)?
                .try_into()
                .unwrap_or(u64::MAX),
            number(&method, "ftol", 1e-9)?,
            None,
        )
    }

    /// The occupancy map the collision term queries.
    #[getter]
    fn occupancy(&self, py: Python<'_>) -> Py<PyAny> {
        self.occupancy.clone_ref(py)
    }

    /// The target traversal speed, meters per second.
    #[getter]
    const fn cruise_speed(&self) -> f64 {
        self.settings.cruise_speed
    }

    /// The maximum number of solver iterations.
    #[getter]
    const fn max_iter(&self) -> u64 {
        self.settings.max_iterations
    }

    /// The relative cost change below which the solver stops.
    #[getter]
    const fn ftol(&self) -> f64 {
        self.settings.cost_tolerance
    }

    /// The ordered cost terms.
    ///
    /// The five defaults are reported by name when the caller supplied
    /// none, because Python built them eagerly in the constructor and a
    /// caller reading `len(optimizer.cost_terms)` is reading that list.
    #[getter]
    fn cost_terms(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        if let Some(listed) = self.terms.as_ref() {
            return Ok(listed.clone_ref(py));
        }
        let named: Vec<PyDefaultTerm> = ["time", "deviation", "velocity", "collision", "dynamics"]
            .into_iter()
            .map(|name| PyDefaultTerm {
                name: name.to_owned(),
            })
            .collect();
        Ok(PyList::new(py, named)?.unbind())
    }

    /// Optimizes a trajectory along `reference_path`.
    #[pyo3(signature = (reference_path, inverse_kinematics = None, feasibility = None))]
    #[pyo3(text_signature = "(reference_path, inverse_kinematics=None, feasibility=None)")]
    fn optimize(
        &self,
        py: Python<'_>,
        reference_path: &Bound<'_, PyAny>,
        inverse_kinematics: Option<&Bound<'_, PyAny>>,
        feasibility: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyTrajectoryResult>> {
        let reference = reference_path
            .try_iter()?
            .map(|point| coordinates(&point?))
            .collect::<PyResult<Vec<Vec<f64>>>>()?;
        if reference.len() < 2 {
            let count = reference.len();
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reference_path must contain at least two waypoints; got {count}."
            )));
        }

        let optimizer = arco_planning::continuous::TrajectoryOptimizer::new(
            self.shared.clone(),
            self.cost_terms_of(py)?,
            self.settings,
        )
        .map_err(|failure| raised(&failure, &self.failure))?;
        let policy = self.feasibility_of(py, feasibility);

        let solved = detached(py, &self.failure, || {
            optimizer.optimize(&reference, &policy)
        })?;

        let commands = Self::commands_of(py, &solved, inverse_kinematics)?;
        let result = PyTrajectoryResult {
            states: as_path(py, &solved.states)?.unbind(),
            commands,
            durations: solved.durations.clone(),
            cost: solved.cost,
            is_feasible: solved.is_feasible,
            optimizer_success: solved.converged,
            optimizer_status_code: i32::from(!solved.converged),
            optimizer_status_text: if solved.converged {
                "converged".to_owned()
            } else {
                "iteration budget exhausted".to_owned()
            },
            optimizer_iteration_count: solved.iterations,
            turn_rates: solved
                .commands
                .iter()
                .map(|command| command.turn_rate)
                .collect(),
        };
        Py::new(py, result)
    }
}

impl PyTrajectoryOptimizer {
    /// The terms the composite cost sums.
    ///
    /// The five built-ins are the native branch and the default: each one
    /// reads the context straight off a slice. A `cost_terms=` entry is
    /// evaluated through the interpreter, once per term per cost call and
    /// tens of cost calls per finite-difference gradient, which is the
    /// case deviation A-07 is about.
    ///
    /// # Errors
    ///
    /// Returns whatever building a term's settings dictionary raised.
    fn cost_terms_of(&self, py: Python<'_>) -> PyResult<Vec<TrajectoryTerm>> {
        let Some(listed) = self.terms.as_ref() else {
            return Ok(TrajectoryTerm::defaults(
                self.weights,
                self.settings.cruise_speed,
                self.barrier,
                self.speed_band,
                self.sample_count,
            ));
        };
        let settings = self.term_settings(py)?;
        listed
            .bind(py)
            .iter()
            .map(|term| -> PyResult<TrajectoryTerm> {
                Ok(TrajectoryTerm::Custom(Box::new(PyTrajectoryTerm::new(
                    term.unbind(),
                    settings.clone_ref(py),
                    self.failure.clone(),
                ))))
            })
            .collect()
    }

    /// The constant half of the context a Python cost term reads.
    ///
    /// # Errors
    ///
    /// Returns whatever building the dictionary raised.
    fn term_settings(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let settings = PyDict::new(py);
        settings.set_item("occupancy", self.occupancy.clone_ref(py))?;
        settings.set_item("sample_count", self.sample_count)?;
        settings.set_item("cruise_speed", self.settings.cruise_speed)?;
        settings.set_item("weight_time", self.weights.time)?;
        settings.set_item("weight_deviation", self.weights.deviation)?;
        settings.set_item("weight_velocity", self.weights.velocity)?;
        settings.set_item("weight_collision", self.weights.collision)?;
        settings.set_item("weight_dynamics", self.weights.dynamics)?;
        settings.set_item("collision_barrier_scale", self.barrier.0)?;
        settings.set_item("collision_barrier_power", self.barrier.1)?;
        settings.set_item("max_speed", self.speed_band.0)?;
        settings.set_item("min_speed", self.speed_band.1)?;
        Ok(settings.unbind())
    }

    /// What the vehicle is taken to be able to execute.
    ///
    /// `Bounded` is the native branch and the default. A `feasibility=`
    /// callable is invoked once per waypoint, after the solve rather than
    /// inside it, so this hook costs a bounded number of crossings rather
    /// than an unbounded one. It still applies the speed bounds first,
    /// because the Python optimizer checked both.
    fn feasibility_of(&self, py: Python<'_>, check: Option<Py<PyAny>>) -> FeasibilityPolicy {
        let (max_speed, min_speed) = self.speed_band;
        let Some(check) = check else {
            return FeasibilityPolicy::Bounded {
                max_speed,
                min_speed,
                max_turn_rate: None,
            };
        };
        let bounds = FeasibilityPolicy::Bounded {
            max_speed,
            min_speed,
            max_turn_rate: None,
        };
        let slot = self.failure.clone();
        let check = check.clone_ref(py);
        FeasibilityPolicy::Custom(Box::new(move |state| {
            if !bounds.accepts(state) {
                return false;
            }
            Python::attach(|py| {
                let derived = [
                    state.x,
                    state.y,
                    state.heading,
                    state.speed,
                    state.turn_rate,
                ];
                match check
                    .call1(py, (as_array(py, &derived),))
                    .and_then(|answer| answer.extract::<bool>(py))
                {
                    Ok(accepted) => accepted,
                    Err(fault) => {
                        drop(slot.park(fault));
                        false
                    }
                }
            })
        }))
    }

    /// One command per segment, from the inverse kinematics or geometry.
    ///
    /// # Errors
    ///
    /// Returns whatever the inverse kinematics raised.
    fn commands_of(
        py: Python<'_>,
        solved: &arco_planning::continuous::TrajectoryResult,
        inverse_kinematics: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyList>> {
        let mut commands = Vec::with_capacity(solved.commands.len());
        for (index, command) in solved.commands.iter().enumerate() {
            let Some(kinematics) = inverse_kinematics else {
                // The Python optimizer wrote a zero turn rate here and the
                // real one is published as `turn_rates`.
                commands.push(as_array(py, &[command.speed, 0.0]).into_any());
                continue;
            };
            let (Some(from), Some(to), Some(duration)) = (
                solved.states.get(index),
                solved.states.get(index.saturating_add(1)),
                solved.durations.get(index),
            ) else {
                continue;
            };
            let answered = kinematics.call1((
                as_array(py, from),
                as_array(py, to),
                command.speed,
                *duration,
            ))?;
            commands.push(as_array(py, &coordinates(&answered)?).into_any());
        }
        Ok(PyList::new(py, commands)?.unbind())
    }
}

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------

/// Adds every planning name to `module`.
///
/// # Errors
///
/// Returns whatever a registration raised, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPlanFailure>()?;
    module.add_class::<PyCostModel>()?;
    module.add_class::<PyDiscretePlanner>()?;
    module.add_class::<PyContinuousPlanner>()?;
    module.add_class::<PyAStarPlanner>()?;
    module.add_class::<PyAStar>()?;
    module.add("RouteResult", route_result_type(module.py())?)?;
    module.add_class::<PyRouteRouter>()?;
    module.add_class::<PyRrtPlanner>()?;
    module.add_class::<PySstPlanner>()?;
    module.add_class::<PyTrajectoryPruner>()?;
    module.add_class::<PyTrajectoryResult>()?;
    module.add_class::<PyTrajectoryOptimizer>()?;
    module.add_class::<PyTimeCostTerm>()?;
    module.add_class::<PyDeviationCostTerm>()?;
    module.add_class::<PyVelocityCostTerm>()?;
    module.add_class::<PyCollisionCostTerm>()?;
    module.add_class::<PyDynamicsCostTerm>()?;
    module.add_function(wrap_pyfunction!(build_default_cost_terms, module)?)?;
    Ok(())
}

/// Names every node of a search result the way Python named it.
///
/// # Errors
///
/// Returns whatever naming a node raised.
fn rendered<N, F>(
    py: Python<'_>,
    diagnostics: &SearchDiagnostics<N>,
    path: Option<Vec<N>>,
    name: F,
) -> PyResult<Diagnostics>
where
    N: Copy,
    F: Fn(N) -> PyResult<Py<PyAny>>,
{
    let expanded = diagnostics
        .expanded_order
        .iter()
        .map(|node| name(*node))
        .collect::<PyResult<Vec<Py<PyAny>>>>()?;
    let came_from = PyDict::new(py);
    for (node, parent) in &diagnostics.came_from {
        came_from.set_item(name(*node)?, name(*parent)?)?;
    }
    let path = path
        .map(|nodes| {
            nodes
                .into_iter()
                .map(&name)
                .collect::<PyResult<Vec<Py<PyAny>>>>()
        })
        .transpose()?;
    Ok((path, expanded, came_from.unbind()))
}

/// Searches a bound map and records what it produced.
///
/// Lives here rather than beside the map adapters because what it does
/// with the outcome, publishing the reason and returning the `None`
/// Python returned, is a boundary decision rather than a map one.
impl BoundMap {
    /// Runs A\* with the interpreter released where the map allows it.
    ///
    /// # Errors
    ///
    /// Returns whatever the map or one of its hooks reported.
    fn search(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
        options: SearchOptions,
        slot: &FailureSlot,
        last: &Mutex<LastCall>,
    ) -> PyResult<Option<Vec<Py<PyAny>>>> {
        match self {
            Self::Manhattan(grid) => {
                let (Some(from), Some(to)) = (self.cell_of(start)?, self.cell_of(goal)?) else {
                    // Deviation A-15: a cell off the edge of the map is a
                    // different answer from an unreachable one.
                    LastCall::refuse(last, PyPlanFailure::StartOutsideMap);
                    return Ok(None);
                };
                let outcome = detached(py, slot, || search(grid, from, to, options))?;
                self.cells_back(py, LastCall::record(last, outcome))
            }
            Self::Euclidean(grid) => {
                let (Some(from), Some(to)) = (self.cell_of(start)?, self.cell_of(goal)?) else {
                    LastCall::refuse(last, PyPlanFailure::StartOutsideMap);
                    return Ok(None);
                };
                let outcome = detached(py, slot, || search(grid, from, to, options))?;
                self.cells_back(py, LastCall::record(last, outcome))
            }
            Self::Cartesian(graph) => {
                let (from, to) = (start.extract::<NodeId>()?, goal.extract::<NodeId>()?);
                let outcome = detached(py, slot, || search(graph, from, to, options))?;
                LastCall::record(last, outcome)
                    .map(|path| {
                        path.into_iter()
                            .map(|node| Ok(node.into_pyobject(py)?.unbind().into_any()))
                            .collect::<PyResult<Vec<Py<PyAny>>>>()
                    })
                    .transpose()
            }
            Self::Python(map) => {
                let (from, to) = (map.number_of(start)?, map.number_of(goal)?);
                let outcome = detached(py, slot, || search(map, from, to, options))?;
                LastCall::record(last, outcome)
                    .map(|path| {
                        path.into_iter()
                            .map(|node| map.object_of(py, node))
                            .collect::<PyResult<Vec<Py<PyAny>>>>()
                    })
                    .transpose()
            }
        }
    }

    /// Runs A\* and reports what it expanded on the way.
    ///
    /// Written against `arco_planning::discrete::search_with_diagnostics`,
    /// which reports the expansion order and the predecessor map the
    /// plain search drops.
    ///
    /// # Errors
    ///
    /// Returns whatever the map or one of its hooks reported.
    fn search_diagnostics(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        goal: &Bound<'_, PyAny>,
        options: SearchOptions,
        slot: &FailureSlot,
        last: &Mutex<LastCall>,
    ) -> PyResult<Diagnostics> {
        match self {
            Self::Manhattan(grid) => {
                let (Some(from), Some(to)) = (self.cell_of(start)?, self.cell_of(goal)?) else {
                    LastCall::refuse(last, PyPlanFailure::StartOutsideMap);
                    return Ok((None, Vec::new(), PyDict::new(py).unbind()));
                };
                let (outcome, diagnostics) = detached(py, slot, || {
                    search_with_diagnostics(grid, from, to, options)
                })?;
                let found = LastCall::record(last, outcome);
                rendered(py, &diagnostics, found, |linear| self.cell_name(py, linear))
            }
            Self::Euclidean(grid) => {
                let (Some(from), Some(to)) = (self.cell_of(start)?, self.cell_of(goal)?) else {
                    LastCall::refuse(last, PyPlanFailure::StartOutsideMap);
                    return Ok((None, Vec::new(), PyDict::new(py).unbind()));
                };
                let (outcome, diagnostics) = detached(py, slot, || {
                    search_with_diagnostics(grid, from, to, options)
                })?;
                let found = LastCall::record(last, outcome);
                rendered(py, &diagnostics, found, |linear| self.cell_name(py, linear))
            }
            Self::Cartesian(graph) => {
                let (from, to) = (start.extract::<NodeId>()?, goal.extract::<NodeId>()?);
                let (outcome, diagnostics) = detached(py, slot, || {
                    search_with_diagnostics(graph, from, to, options)
                })?;
                let found = LastCall::record(last, outcome);
                rendered(py, &diagnostics, found, |node| {
                    Ok(node.into_pyobject(py)?.unbind().into_any())
                })
            }
            Self::Python(map) => {
                let (from, to) = (map.number_of(start)?, map.number_of(goal)?);
                let (outcome, diagnostics) =
                    detached(py, slot, || search_with_diagnostics(map, from, to, options))?;
                let found = LastCall::record(last, outcome);
                rendered(py, &diagnostics, found, |node| map.object_of(py, node))
            }
        }
    }

    /// The index tuple Python names a linear cell by.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` when the index is outside the grid.
    fn cell_name(&self, py: Python<'_>, linear: usize) -> PyResult<Py<PyAny>> {
        let Some(cells) = self.cells() else {
            return Ok(py.None());
        };
        let index = cells
            .cell_index(linear)
            .map_err(|failure| to_exception(&failure))?;
        Ok(PyTuple::new(py, index)?.unbind().into_any())
    }

    /// Turns linear cell indices back into the index tuples Python names.
    ///
    /// # Errors
    ///
    /// Returns whatever building a tuple raised.
    fn cells_back(
        &self,
        py: Python<'_>,
        path: Option<Vec<usize>>,
    ) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let Some(path) = path else { return Ok(None) };
        let Some(cells) = self.cells() else {
            return Ok(None);
        };
        path.into_iter()
            .map(|linear| {
                let index = cells
                    .cell_index(linear)
                    .map_err(|failure| to_exception(&failure))?;
                Ok(PyTuple::new(py, index)?.unbind().into_any())
            })
            .collect::<PyResult<Vec<Py<PyAny>>>>()
            .map(Some)
    }
}
