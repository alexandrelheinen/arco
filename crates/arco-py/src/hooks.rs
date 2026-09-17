//! Adapters that let a Python object stand in for a Rust policy.
//!
//! ADR-004 splits every planner hook into an enum with a native variant
//! per built-in policy and one variant holding something supplied from
//! outside. This module is that last variant: one adapter per trait in
//! `arco_core::protocols` that a caller can inject through, plus the two
//! map adapters the planners need in order to read a Python map at all.
//!
//! Every adapter here reattaches to the interpreter on each call, which
//! is what deviation A-07 warns about and what `FR-PERF-03` documents.
//! A binding reaches for one only when the caller supplied a callable;
//! the default argument always selects the native policy instead.
//!
//! Two conversions run the other way. [`BoundOccupancy`] and [`BoundMap`]
//! read a Python map once, at construction, and rebuild it natively when
//! its shape is one this workspace already has a type for. The planner
//! loop then never leaves Rust, which is the whole point of the port.
//! Anything unrecognized keeps the Python adapter and pays per query.

use std::sync::{Arc, Mutex};

use arco_core::Error;
use arco_core::protocols::{
    CostTerm, DiscreteMap, NearestObstacle, Occupancy, PlannerCost, Sampler, SegmentChecker,
    Steerer,
};
use arco_core::rng::Pcg64;
use arco_mapping::graph::{CartesianGraph, NodeId};
use arco_mapping::grid::{Cell, EuclideanGrid, GridCells, ManhattanGrid};
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::TrajectoryContext;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::errors::to_exception;

/// What an adapter reports when the Python side raised.
///
/// The trait it implements returns an [`Error`], which carries no Python
/// exception, so the real one is parked in a [`FailureSlot`] and this
/// stands in until the boundary swaps it back. A caller never sees this
/// message: [`FailureSlot::restore`] replaces it with the original
/// exception, its type and its traceback intact, which is what
/// `FR-API-04` asks for.
const HOOK_RAISED: &str = "a Python planner hook raised";

/// Where an adapter parks the exception its Python callable raised.
///
/// Shared by every adapter belonging to one planner, so the boundary has
/// a single place to look before it converts an [`Error`] into a
/// `ValueError` that would have thrown away the caller's own exception
/// type.
#[derive(Clone, Debug, Default)]
pub(crate) struct FailureSlot {
    parked: Arc<Mutex<Option<PyErr>>>,
}

impl FailureSlot {
    /// Parks `raised` and returns the [`Error`] the trait has to hand back.
    pub(crate) fn park(&self, raised: PyErr) -> Error {
        if let Ok(mut slot) = self.parked.lock() {
            // The first exception is the one that stopped the run; a
            // later one is a consequence of it and would only bury the
            // cause.
            if slot.is_none() {
                *slot = Some(raised);
            }
        }
        Error::ConflictingArguments {
            message: HOOK_RAISED.to_owned(),
        }
    }

    /// Takes whatever was parked, leaving the slot empty.
    pub(crate) fn take(&self) -> Option<PyErr> {
        self.parked.lock().ok().and_then(|mut slot| slot.take())
    }
}

/// Reads a Python array-like into coordinates.
///
/// Borrows the buffer when the caller passed a C-contiguous array of
/// doubles, which is the case the planners are given, and falls back to
/// element-wise extraction for a list, a tuple, or an array of another
/// dtype. The fallback exists because every Python entry point ran
/// `numpy.asarray(value, dtype=float)` first and accepted all three.
///
/// # Errors
///
/// Returns a `TypeError` when `object` is not a one-dimensional sequence
/// of numbers.
pub(crate) fn coordinates(object: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(borrowed) = object.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(borrowed.as_array().to_vec());
    }
    object.extract::<Vec<f64>>()
}

/// Reads a Python array-like into a list of points.
///
/// # Errors
///
/// Returns a `TypeError` when `object` is not a two-dimensional sequence
/// of numbers.
pub(crate) fn point_rows(object: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(borrowed) = object.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(borrowed
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect());
    }
    object.extract::<Vec<Vec<f64>>>()
}

/// Wraps coordinates as the one-dimensional array Python expects back.
pub(crate) fn as_array<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, values.to_vec())
}

/// Wraps a path as the list of arrays every Python planner returned.
pub(crate) fn as_path<'py>(py: Python<'py>, path: &[Vec<f64>]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, path.iter().map(|point| as_array(py, point)))
}

/// A sampler that calls back into Python.
///
/// Holds the generator alongside the callable because the Python
/// signature is `(rng) -> state` and the generator a caller's sampler
/// expects is a `numpy.random.Generator`, not the native one the planner
/// draws its goal bias from.
#[derive(Debug)]
pub(crate) struct PySampler {
    callable: Py<PyAny>,
    generator: Py<PyAny>,
    failure: FailureSlot,
}

impl PySampler {
    /// Builds a sampler calling `callable` with `generator`.
    pub(crate) const fn new(
        callable: Py<PyAny>,
        generator: Py<PyAny>,
        failure: FailureSlot,
    ) -> Self {
        Self {
            callable,
            generator,
            failure,
        }
    }
}

impl Sampler for PySampler {
    fn sample(&self, _generator: &mut Pcg64) -> Result<Vec<f64>, Error> {
        Python::attach(|py| {
            let drawn = self
                .callable
                .call1(py, (self.generator.bind(py),))
                .and_then(|state| coordinates(state.bind(py)));
            drawn.map_err(|raised| self.failure.park(raised))
        })
    }
}

/// A steering law that calls back into Python.
#[derive(Debug)]
pub(crate) struct PySteerer {
    callable: Py<PyAny>,
    failure: FailureSlot,
}

impl PySteerer {
    /// Builds a steerer calling `callable` as `(from, to) -> state`.
    pub(crate) const fn new(callable: Py<PyAny>, failure: FailureSlot) -> Self {
        Self { callable, failure }
    }
}

impl Steerer for PySteerer {
    fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<f64>, Error> {
        Python::attach(|py| {
            let stepped = self
                .callable
                .call1(py, (as_array(py, from), as_array(py, to)))
                .and_then(|state| coordinates(state.bind(py)));
            stepped.map_err(|raised| self.failure.park(raised))
        })
    }
}

/// A segment check that calls back into Python.
#[derive(Debug)]
pub(crate) struct PySegmentChecker {
    callable: Py<PyAny>,
    failure: FailureSlot,
}

impl PySegmentChecker {
    /// Builds a checker calling `callable` as `(a, b) -> bool`.
    pub(crate) const fn new(callable: Py<PyAny>, failure: FailureSlot) -> Self {
        Self { callable, failure }
    }
}

impl SegmentChecker for PySegmentChecker {
    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        Python::attach(|py| {
            let free = self
                .callable
                .call1(py, (as_array(py, from), as_array(py, to)))
                .and_then(|answer| answer.extract::<bool>(py));
            free.map_err(|raised| self.failure.park(raised))
        })
    }
}

/// A cost model that calls back into Python.
///
/// The Python argument is a `PlannerCost` instance rather than a
/// callable, so both of its methods are looked up by name.
#[derive(Debug)]
pub(crate) struct PyPlannerCost {
    model: Py<PyAny>,
    failure: FailureSlot,
}

impl PyPlannerCost {
    /// Builds a cost model delegating to `model`.
    pub(crate) const fn new(model: Py<PyAny>, failure: FailureSlot) -> Self {
        Self { model, failure }
    }

    /// Calls one of the model's two methods.
    fn measure(&self, method: &str, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        Python::attach(|py| {
            let measured = self
                .model
                .bind(py)
                .call_method1(method, (as_array(py, from), as_array(py, to)))
                .and_then(|value| value.extract::<f64>());
            measured.map_err(|raised| self.failure.park(raised))
        })
    }
}

impl PlannerCost for PyPlannerCost {
    fn distance(&self, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        self.measure("distance", from, to)
    }

    fn heuristic(&self, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        self.measure("heuristic", from, to)
    }
}

/// One caller-supplied entry of the trajectory optimizer's `cost_terms`.
///
/// The Python term reads a dictionary, so the context is rebuilt as one
/// per evaluation. That is the expensive part, and it is why deviation
/// A-07 says a custom term gives back the speedup: the native terms read
/// the same numbers straight off the slice.
#[derive(Debug)]
pub(crate) struct PyTrajectoryTerm {
    callable: Py<PyAny>,
    settings: Py<PyDict>,
    failure: FailureSlot,
}

impl PyTrajectoryTerm {
    /// Builds a term calling `callable` with a context built from
    /// `settings` plus whatever the optimizer is evaluating.
    pub(crate) const fn new(
        callable: Py<PyAny>,
        settings: Py<PyDict>,
        failure: FailureSlot,
    ) -> Self {
        Self {
            callable,
            settings,
            failure,
        }
    }

    /// Rebuilds the context dictionary the Python terms were written for.
    ///
    /// # Errors
    ///
    /// Returns whatever building the dictionary raised.
    fn context<'py>(
        &self,
        py: Python<'py>,
        context: &TrajectoryContext<'_>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let built = self.settings.bind(py).copy()?;
        let waypoints = as_path(py, context.waypoints)?;
        let reference = as_path(py, context.reference)?;
        let segments = context.durations.len();

        built.set_item("durations", as_array(py, context.durations))?;
        built.set_item("durs", as_array(py, context.durations))?;
        built.set_item("waypoints", &waypoints)?;
        built.set_item("pts", PyArray2::from_vec2(py, context.waypoints)?)?;
        built.set_item("lengths", as_array(py, context.lengths))?;
        built.set_item("speeds", as_array(py, context.speeds))?;
        built.set_item("ref", &reference)?;
        built.set_item("segment_count", segments)?;
        built.set_item("dim", context.occupancy.dimension())?;
        Ok(built)
    }
}

impl CostTerm<TrajectoryContext<'_>> for PyTrajectoryTerm {
    fn evaluate(&self, context: &TrajectoryContext<'_>) -> Result<f64, Error> {
        Python::attach(|py| {
            let value = self
                .context(py, context)
                .and_then(|built| self.callable.call1(py, (built,)))
                .and_then(|value| value.extract::<f64>(py));
            value.map_err(|raised| self.failure.park(raised))
        })
    }
}

/// An occupancy map that calls back into Python.
///
/// The methods are the ones `arco.mapping.Occupancy` declares, looked up
/// by name so that a caller's own map works without inheriting anything.
#[derive(Debug)]
pub(crate) struct PyOccupancy {
    map: Py<PyAny>,
    dimension: usize,
    clearance: f64,
    failure: FailureSlot,
}

impl PyOccupancy {
    /// Reads the two constants off `map` and keeps a handle on the rest.
    ///
    /// A map publishing neither is taken at a dimension and a clearance
    /// of zero, which is what a binary occupancy carries, rather than
    /// refused: the methods below are the ones a planner calls.
    fn adopt(map: &Bound<'_, PyAny>, failure: FailureSlot) -> Self {
        let dimension = map
            .getattr("dimension")
            .and_then(|value| value.extract::<usize>())
            .unwrap_or(0);
        let clearance = map
            .getattr("clearance")
            .and_then(|value| value.extract::<f64>())
            .unwrap_or(0.0);
        Self {
            map: map.clone().unbind(),
            dimension,
            clearance,
            failure,
        }
    }
}

impl Occupancy for PyOccupancy {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn clearance(&self) -> f64 {
        self.clearance
    }

    fn nearest_obstacle(&self, point: &[f64]) -> Result<NearestObstacle, Error> {
        Python::attach(|py| {
            let found = self
                .map
                .bind(py)
                .call_method1("nearest_obstacle", (as_array(py, point),))
                .and_then(|answer| answer.extract::<(f64, Vec<f64>)>());
            found
                .map(|(distance, obstacle)| NearestObstacle {
                    // Deviation A-16: the trait reports the distance to
                    // the obstacle's surface and Python reports it to the
                    // center, so the clearance comes off here.
                    distance: distance - self.clearance,
                    point: obstacle,
                })
                .map_err(|raised| self.failure.park(raised))
        })
    }

    fn is_occupied(&self, point: &[f64]) -> Result<bool, Error> {
        Python::attach(|py| {
            let occupied = self
                .map
                .bind(py)
                .call_method1("is_occupied", (as_array(py, point),))
                .and_then(|answer| answer.extract::<bool>());
            occupied.map_err(|raised| self.failure.park(raised))
        })
    }

    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        Python::attach(|py| {
            let free = self
                .map
                .bind(py)
                .call_method1("segment_free", (as_array(py, from), as_array(py, to)))
                .and_then(|answer| answer.extract::<bool>());
            free.map_err(|raised| self.failure.park(raised))
        })
    }
}

/// The occupancy a planner ended up with, native where that was possible.
///
/// `Native` is the fast branch and the one every standard map takes: the
/// obstacle set is read once and rebuilt as a k-d tree, after which the
/// planner loop never touches the interpreter. `Python` is the fallback
/// for a map this workspace has no type for, and it pays one attach per
/// query, per deviation A-07.
#[derive(Debug)]
pub(crate) enum BoundOccupancy {
    /// An obstacle set rebuilt in Rust, queried without the interpreter.
    Native(KdTreeOccupancy),
    /// Anything else, queried through the interpreter.
    Python(PyOccupancy),
}

impl BoundOccupancy {
    /// Adopts `map`, rebuilding it natively when its shape is recognized.
    ///
    /// Recognition is by the two attributes `arco.mapping.KDTreeOccupancy`
    /// publishes, `points` and `clearance`, rather than by type, so that
    /// the compiled map and the Python one are both taken. The obstacle
    /// set is copied, so a map mutated after a planner was built keeps
    /// planning against the set it was built from.
    pub(crate) fn adopt(map: &Bound<'_, PyAny>, failure: FailureSlot) -> Self {
        let (Ok(points), Ok(clearance)) = (map.getattr("points"), map.getattr("clearance")) else {
            return Self::Python(PyOccupancy::adopt(map, failure));
        };
        let (Ok(points), Ok(clearance)) = (point_rows(&points), clearance.extract::<f64>()) else {
            return Self::Python(PyOccupancy::adopt(map, failure));
        };
        KdTreeOccupancy::new(&points, clearance).map_or_else(
            // An empty set or a non-positive clearance is what the Python
            // constructor already rejected, so the map cannot have come
            // from there and the adapter is the honest answer.
            |_unusable| Self::Python(PyOccupancy::adopt(map, failure)),
            Self::Native,
        )
    }
}

impl Occupancy for BoundOccupancy {
    fn dimension(&self) -> usize {
        match self {
            Self::Native(tree) => tree.dimension(),
            Self::Python(map) => map.dimension(),
        }
    }

    fn clearance(&self) -> f64 {
        match self {
            Self::Native(tree) => tree.clearance(),
            Self::Python(map) => map.clearance(),
        }
    }

    fn nearest_obstacle(&self, point: &[f64]) -> Result<NearestObstacle, Error> {
        match self {
            Self::Native(tree) => tree.nearest_obstacle(point),
            Self::Python(map) => map.nearest_obstacle(point),
        }
    }

    fn is_occupied(&self, point: &[f64]) -> Result<bool, Error> {
        match self {
            Self::Native(tree) => tree.is_occupied(point),
            Self::Python(map) => map.is_occupied(point),
        }
    }

    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        match self {
            // Disambiguated because the tree also implements
            // `SegmentChecker`, whose method carries the same name.
            Self::Native(tree) => Occupancy::is_segment_free(tree, from, to),
            Self::Python(map) => map.is_segment_free(from, to),
        }
    }
}

/// One occupancy shared by a planner and its segment policy.
///
/// `SegmentPolicy` owns its occupancy and a planner holds one policy, so
/// without this every stage would carry its own copy of the obstacle set.
#[derive(Clone, Debug)]
pub(crate) struct SharedOccupancy {
    inner: Arc<BoundOccupancy>,
}

impl SharedOccupancy {
    /// Wraps `occupancy` so several stages can hold the same one.
    pub(crate) fn new(occupancy: BoundOccupancy) -> Self {
        Self {
            inner: Arc::new(occupancy),
        }
    }

    /// Whether queries run without touching the interpreter.
    pub(crate) fn is_native(&self) -> bool {
        matches!(*self.inner, BoundOccupancy::Native(_))
    }
}

impl Occupancy for SharedOccupancy {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn clearance(&self) -> f64 {
        self.inner.clearance()
    }

    fn nearest_obstacle(&self, point: &[f64]) -> Result<NearestObstacle, Error> {
        self.inner.nearest_obstacle(point)
    }

    fn is_occupied(&self, point: &[f64]) -> Result<bool, Error> {
        self.inner.is_occupied(point)
    }

    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error> {
        self.inner.is_segment_free(from, to)
    }
}

/// The nodes a Python map named, numbered so a search can index them.
///
/// `DiscreteMap::Node` has to be `Copy`, `Ord` and `Hash`, and a Python
/// object is none of the three. Each one is therefore given a number the
/// first time it is seen, and the dictionary going the other way is a
/// Python dictionary because the nodes are hashed by Python's own rules.
#[derive(Debug)]
struct NodeTable {
    objects: Vec<Py<PyAny>>,
    numbers: Py<PyDict>,
}

impl NodeTable {
    /// Builds an empty table.
    fn new(py: Python<'_>) -> Self {
        Self {
            objects: Vec::new(),
            numbers: PyDict::new(py).unbind(),
        }
    }

    /// The number standing for `node`, assigning one if it is new.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` when the node is not hashable.
    fn number_of(&mut self, py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        let numbers = self.numbers.bind(py);
        if let Some(known) = numbers.get_item(node)? {
            return known.extract::<usize>();
        }
        let assigned = self.objects.len();
        numbers.set_item(node, assigned)?;
        self.objects.push(node.clone().unbind());
        Ok(assigned)
    }

    /// The node a number stands for.
    fn object_of(&self, number: usize) -> Option<&Py<PyAny>> {
        self.objects.get(number)
    }
}

/// A discrete map that calls back into Python.
///
/// Reads `neighbors`, `distance` and, when they exist, `heuristic` and
/// `is_occupied`, which is the set `arco.planning.discrete` required of a
/// graph. Nodes stay Python objects and are numbered by [`NodeTable`].
#[derive(Debug)]
pub(crate) struct PyDiscreteMap {
    map: Py<PyAny>,
    nodes: Mutex<NodeTable>,
    /// An injected `heuristic=`, which overrides the map's own.
    heuristic: Option<Py<PyAny>>,
    /// An object whose `distance` and `heuristic` replace the map's.
    ///
    /// Set when a Python subclass overrode either one, so that the search
    /// measures with the metric the caller actually installed rather than
    /// with the one the map publishes. Deviation A-24.
    metric: Option<Py<PyAny>>,
    has_heuristic: bool,
    has_occupancy: bool,
    failure: FailureSlot,
}

impl PyDiscreteMap {
    /// Adopts `map`, noting which optional methods it carries.
    ///
    /// # Errors
    ///
    /// Returns whatever inspecting the map raised.
    fn adopt(map: &Bound<'_, PyAny>, failure: FailureSlot) -> PyResult<Self> {
        Ok(Self {
            map: map.clone().unbind(),
            nodes: Mutex::new(NodeTable::new(map.py())),
            heuristic: None,
            metric: None,
            has_heuristic: map.hasattr("heuristic")?,
            has_occupancy: map.hasattr("is_occupied")?,
            failure,
        })
    }

    /// The number standing for a Python node.
    ///
    /// # Errors
    ///
    /// Returns a `RuntimeError` when the table cannot be locked, which
    /// means an earlier call panicked while holding it.
    pub(crate) fn number_of(&self, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut table = self.nodes.lock().map_err(|_poisoned| {
            pyo3::exceptions::PyRuntimeError::new_err("the node table is unusable")
        })?;
        table.number_of(node.py(), node)
    }

    /// The Python node a number stands for.
    ///
    /// # Errors
    ///
    /// As [`PyDiscreteMap::number_of`].
    pub(crate) fn object_of(&self, py: Python<'_>, number: usize) -> PyResult<Py<PyAny>> {
        let table = self.nodes.lock().map_err(|_poisoned| {
            pyo3::exceptions::PyRuntimeError::new_err("the node table is unusable")
        })?;
        table.object_of(number).map_or_else(
            || {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "the search returned a node the map never named",
                ))
            },
            |node| Ok(node.clone_ref(py)),
        )
    }

    /// Whether the map calls `node` occupied.
    ///
    /// A map with no `is_occupied` has no occupied nodes, which is what
    /// the Python search assumed of a plain graph.
    ///
    /// # Errors
    ///
    /// Returns whatever the map raised.
    pub(crate) fn is_occupied(&self, node: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.has_occupancy {
            return Ok(false);
        }
        self.map
            .bind(node.py())
            .call_method1("is_occupied", (node,))?
            .extract::<bool>()
    }

    /// Calls a two-node method and reads a number back.
    ///
    /// Asks the metric object when a subclass installed one, and the map
    /// itself otherwise.
    fn measure(&self, method: &str, from: usize, to: usize) -> Result<f64, Error> {
        Python::attach(|py| {
            let measured = self
                .object_of(py, from)
                .and_then(|first| Ok((first, self.object_of(py, to)?)))
                .and_then(|(first, second)| {
                    self.metric
                        .as_ref()
                        .unwrap_or(&self.map)
                        .bind(py)
                        .call_method1(method, (first, second))?
                        .extract::<f64>()
                });
            measured.map_err(|raised| self.failure.park(raised))
        })
    }
}

impl DiscreteMap for PyDiscreteMap {
    type Node = usize;

    fn contains(&self, _node: usize) -> bool {
        // A Python graph publishes no membership test, and the Python
        // search never asked for one: an unknown node simply had no
        // neighbors and the search reported no path. Answering true keeps
        // that behavior rather than inventing a stricter one.
        true
    }

    fn neighbors(&self, node: usize) -> Vec<usize> {
        Python::attach(|py| {
            let found = self.object_of(py, node).and_then(|current| {
                let listed = self.map.bind(py).call_method1("neighbors", (current,))?;
                let mut numbers = Vec::new();
                for neighbor in listed.try_iter()? {
                    numbers.push(self.number_of(&neighbor?)?);
                }
                Ok(numbers)
            });
            match found {
                Ok(numbers) => numbers,
                Err(raised) => {
                    drop(self.failure.park(raised));
                    Vec::new()
                }
            }
        })
    }

    fn distance(&self, from: usize, to: usize) -> Result<f64, Error> {
        self.measure("distance", from, to)
    }

    fn heuristic(&self, node: usize, goal: usize) -> Result<f64, Error> {
        let Some(injected) = self.heuristic.as_ref() else {
            // A metric object answers both questions itself, including
            // the fallback to distance for a graph that publishes no
            // heuristic, so it is always asked for the one it was asked
            // about. Only a bare map needs that fallback applied here.
            return if self.metric.is_some() || self.has_heuristic {
                self.measure("heuristic", node, goal)
            } else {
                self.measure("distance", node, goal)
            };
        };
        Python::attach(|py| {
            let estimated = self
                .object_of(py, node)
                .and_then(|current| Ok((current, self.object_of(py, goal)?)))
                .and_then(|(current, target)| {
                    injected.bind(py).call1((current, target))?.extract::<f64>()
                });
            estimated.map_err(|raised| self.failure.park(raised))
        })
    }
}

/// The map a discrete planner ended up with, native where possible.
///
/// The three native branches are the fast ones: the search runs with the
/// interpreter released and never reattaches. `Python` is the fallback
/// and pays one attach per edge, per deviation A-07.
#[derive(Debug)]
pub(crate) enum BoundMap {
    /// A four-connected grid, nodes named by index tuple.
    Manhattan(ManhattanGrid),
    /// An eight-connected grid, nodes named by index tuple.
    Euclidean(EuclideanGrid),
    /// A positioned graph, nodes named by identifier.
    Cartesian(CartesianGraph),
    /// Anything else, queried through the interpreter.
    Python(PyDiscreteMap),
}

impl BoundMap {
    /// Adopts `map`, rebuilding it natively when its shape is recognized.
    ///
    /// A grid is recognized by `shape` and `data`, a positioned graph by
    /// `nodes`, `edges` and `position`, and the connectivity of a grid by
    /// how many neighbors its own method reports for an interior cell,
    /// which is what separates the two grid families without naming their
    /// classes. Everything else keeps the Python adapter.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the map raised.
    pub(crate) fn adopt(
        map: &Bound<'_, PyAny>,
        failure: FailureSlot,
        heuristic: Option<Py<PyAny>>,
        metric: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        // An injected heuristic, or a metric a subclass replaced, has to
        // be consulted per expanded node and no native map can call
        // either, so both force the Python adapter. With neither, the
        // native map the object turned out to be is kept.
        if heuristic.is_none() && metric.is_none() {
            if let Some(grid) = adopt_grid(map)? {
                return Ok(grid);
            }
            if let Some(graph) = adopt_graph(map)? {
                return Ok(graph);
            }
        }
        let mut adapter = PyDiscreteMap::adopt(map, failure)?;
        adapter.heuristic = heuristic;
        adapter.metric = metric;
        Ok(Self::Python(adapter))
    }

    /// Builds a grid straight out of a numpy cell array.
    ///
    /// What `arco.planning.discrete.api.AStar` did: zero is free, any
    /// other value is occupied, and the connectivity comes from the
    /// `grid_type` the caller named rather than from the array.
    ///
    /// # Errors
    ///
    /// Returns whatever reading the array raised.
    pub(crate) fn from_cells(cells: &Bound<'_, PyAny>, diagonal: bool) -> PyResult<Self> {
        let shape = cells.getattr("shape")?.extract::<Vec<usize>>()?;
        let mut built = build_cells(&shape, 1.0, cells)?;
        if diagonal {
            let mut grid =
                EuclideanGrid::new_free(&shape, 1.0).map_err(|failure| to_exception(&failure))?;
            core::mem::swap(grid.cells_mut(), &mut built);
            return Ok(Self::Euclidean(grid));
        }
        let mut grid =
            ManhattanGrid::new_free(&shape, 1.0).map_err(|failure| to_exception(&failure))?;
        core::mem::swap(grid.cells_mut(), &mut built);
        Ok(Self::Manhattan(grid))
    }

    /// Rebuilds a Python positioned graph, refusing anything else.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` when the object publishes no node positions,
    /// since routing continuous positions over it is not defined.
    pub(crate) fn cartesian_of(map: &Bound<'_, PyAny>) -> PyResult<CartesianGraph> {
        match adopt_graph(map)? {
            Some(Self::Cartesian(graph)) => Ok(graph),
            _unpositioned => Err(pyo3::exceptions::PyTypeError::new_err(
                "graph must expose nodes, edges and position to be routed over.",
            )),
        }
    }

    /// The cells behind a grid, when this map is one.
    pub(crate) const fn cells(&self) -> Option<&GridCells> {
        match self {
            Self::Manhattan(grid) => Some(grid.cells()),
            Self::Euclidean(grid) => Some(grid.cells()),
            Self::Cartesian(_) | Self::Python(_) => None,
        }
    }

    /// The linear index a Python cell tuple names, when it is inside.
    ///
    /// `None` means the tuple is off the edge of the grid, which
    /// deviation A-15 reports as a distinct answer rather than as an
    /// unreachable goal.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` when the node is not an index tuple.
    pub(crate) fn cell_of(&self, node: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let Some(cells) = self.cells() else {
            return Ok(None);
        };
        let index = node.extract::<Vec<usize>>()?;
        Ok(cells.linear_index(&index).ok())
    }

    /// Whether the map calls `node` occupied.
    ///
    /// # Errors
    ///
    /// Returns whatever the map raised.
    pub(crate) fn is_occupied(&self, node: &Bound<'_, PyAny>) -> PyResult<bool> {
        match self {
            Self::Manhattan(_) | Self::Euclidean(_) => {
                let (Some(cells), Some(linear)) = (self.cells(), self.cell_of(node)?) else {
                    return Ok(false);
                };
                cells
                    .blocks(linear)
                    .map_err(|failure| to_exception(&failure))
            }
            // A positioned graph has no occupancy: a node it does not
            // carry is outside the map, which the search reports itself.
            Self::Cartesian(_) => Ok(false),
            Self::Python(map) => map.is_occupied(node),
        }
    }

    /// The edge cost between two nodes named the way Python names them.
    ///
    /// # Errors
    ///
    /// Returns whatever the map raised.
    pub(crate) fn distance(
        &self,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        self.measure(state_a, state_b, false)
    }

    /// A remaining-cost estimate between two nodes.
    ///
    /// # Errors
    ///
    /// Returns whatever the map raised.
    pub(crate) fn heuristic(
        &self,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        self.measure(state_a, state_b, true)
    }

    /// Answers one of the two metrics for a pair of Python nodes.
    fn measure(
        &self,
        state_a: &Bound<'_, PyAny>,
        state_b: &Bound<'_, PyAny>,
        estimate: bool,
    ) -> PyResult<f64> {
        let missing =
            || pyo3::exceptions::PyKeyError::new_err("the node is outside the map".to_owned());
        let measured = match self {
            Self::Manhattan(grid) => {
                let (Some(from), Some(to)) = (self.cell_of(state_a)?, self.cell_of(state_b)?)
                else {
                    return Err(missing());
                };
                if estimate {
                    grid.heuristic(from, to)
                } else {
                    grid.distance(from, to)
                }
            }
            Self::Euclidean(grid) => {
                let (Some(from), Some(to)) = (self.cell_of(state_a)?, self.cell_of(state_b)?)
                else {
                    return Err(missing());
                };
                if estimate {
                    grid.heuristic(from, to)
                } else {
                    grid.distance(from, to)
                }
            }
            Self::Cartesian(graph) => {
                let (from, to) = (state_a.extract::<NodeId>()?, state_b.extract::<NodeId>()?);
                if estimate {
                    graph.heuristic(from, to)
                } else {
                    graph.distance(from, to)
                }
            }
            Self::Python(map) => {
                let (from, to) = (map.number_of(state_a)?, map.number_of(state_b)?);
                if estimate {
                    map.heuristic(from, to)
                } else {
                    map.distance(from, to)
                }
            }
        };
        measured.map_err(|failure| to_exception(&failure))
    }
}

/// Rebuilds a Python grid natively, or reports that it is not one.
///
/// # Errors
///
/// Returns whatever reading the grid raised, or a `ValueError` when its
/// shape and its cell array disagree.
fn adopt_grid(map: &Bound<'_, PyAny>) -> PyResult<Option<BoundMap>> {
    let (Ok(shape), Ok(data)) = (map.getattr("shape"), map.getattr("data")) else {
        return Ok(None);
    };
    let Ok(shape) = shape.extract::<Vec<usize>>() else {
        return Ok(None);
    };
    let cell_size = map
        .getattr("cell_size")
        .and_then(|value| value.extract::<f64>())
        .unwrap_or(1.0);
    let diagonal = is_diagonal(map, &shape)?;

    let mut cells = build_cells(&shape, cell_size, &data)?;
    if diagonal {
        let mut grid =
            EuclideanGrid::new_free(&shape, cell_size).map_err(|failure| to_exception(&failure))?;
        core::mem::swap(grid.cells_mut(), &mut cells);
        return Ok(Some(BoundMap::Euclidean(grid)));
    }
    let mut grid =
        ManhattanGrid::new_free(&shape, cell_size).map_err(|failure| to_exception(&failure))?;
    core::mem::swap(grid.cells_mut(), &mut cells);
    Ok(Some(BoundMap::Manhattan(grid)))
}

/// Reads a Python grid's cell array into native cells.
///
/// # Errors
///
/// Returns a `ValueError` when the shape and the cell array disagree.
fn build_cells(shape: &[usize], cell_size: f64, data: &Bound<'_, PyAny>) -> PyResult<GridCells> {
    let mut cells =
        GridCells::new_free(shape, cell_size).map_err(|failure| to_exception(&failure))?;
    let flattened = data
        .call_method0("ravel")
        .or_else(|_not_an_array| data.call_method0("flatten"))?
        .extract::<Vec<i64>>()?;
    for (linear, state) in flattened.iter().enumerate() {
        if *state != 0 {
            cells
                .set_cell(linear, Cell::Occupied)
                .map_err(|failure| to_exception(&failure))?;
        }
    }
    Ok(cells)
}

/// Whether a grid's own neighbor method admits a diagonal move.
///
/// The Python grid families differ only in their connectivity and their
/// metric, and both are decided by the class rather than by a flag, so
/// asking the instance is what separates them without importing either.
///
/// # Errors
///
/// Returns whatever the grid raised.
fn is_diagonal(map: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<bool> {
    let dimension = shape.len();
    // An interior cell is the only one whose neighbor count says anything:
    // a cell on an edge is missing neighbors under either connectivity.
    if dimension == 0 || shape.iter().any(|extent| *extent < 3) {
        return Ok(false);
    }
    let interior = PyTuple::new(map.py(), core::iter::repeat_n(1_usize, dimension))?;
    let Ok(listed) = map.call_method1("neighbors", (interior,)) else {
        return Ok(false);
    };
    let mut found = 0_usize;
    for neighbor in listed.try_iter()? {
        drop(neighbor?);
        found = found.saturating_add(1);
    }
    // Four-connected in two dimensions gives four neighbors and
    // eight-connected gives eight, and the same doubling holds in any
    // dimension, so anything above twice the dimension is diagonal.
    Ok(found > dimension.saturating_mul(2))
}

/// Rebuilds a Python positioned graph natively, or reports it is not one.
///
/// # Errors
///
/// Returns whatever reading the graph raised, or a `ValueError` when a
/// node it published cannot be added.
fn adopt_graph(map: &Bound<'_, PyAny>) -> PyResult<Option<BoundMap>> {
    let (Ok(nodes), Ok(edges)) = (map.getattr("nodes"), map.getattr("edges")) else {
        return Ok(None);
    };
    if !map.hasattr("position")? {
        return Ok(None);
    }
    let (Ok(nodes), Ok(edges)) = (
        nodes.extract::<Vec<NodeId>>(),
        edges.extract::<Vec<(NodeId, NodeId, f64)>>(),
    ) else {
        return Ok(None);
    };

    let mut graph = CartesianGraph::new();
    for node in nodes {
        let position = coordinates(&map.call_method1("position", (node,))?)?;
        graph
            .add_node(node, &position)
            .map_err(|failure| to_exception(&failure))?;
    }
    for (from, to, weight) in edges {
        graph
            .add_edge(from, to, Some(weight))
            .map_err(|failure| to_exception(&failure))?;
    }
    Ok(Some(BoundMap::Cartesian(graph)))
}
