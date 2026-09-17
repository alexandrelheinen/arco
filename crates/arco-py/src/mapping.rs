// The doc comments in this module become Python docstrings, printed
// verbatim by `help()`. A backtick around a Python type or argument name
// would show up as punctuation to the reader they are written for.
#![expect(
    clippy::doc_markdown,
    reason = "these doc comments are Python docstrings, rendered verbatim by help()"
)]

//! `arco.mapping`: grids, graphs and the k-d tree occupancy structure.
//!
//! Every class here keeps the argument names, positional order and default
//! values the Python implementation accepted, per `FR-API-02`, and the
//! `isinstance` relationships the Python hierarchy had: `ManhattanGrid`
//! and `EuclideanGrid` derive from `Grid`, and `RoadGraph` derives from
//! `CartesianGraph` which derives from `WeightedGraph`.
//!
//! Deviation A-03 replaced that inheritance with ownership on the Rust
//! side, so the state a Python subclass would have inherited lives once,
//! in the base class, as an enum naming which layer was built. A method
//! that Python resolved through a base class resolves here through that
//! enum, and a caller sees the same call set either way.
//!
//! Two things are validated here rather than deeper down. Arguments this
//! layer names, `shape`, `physical_size`, `cell_size`, `points` and
//! `clearance`, are checked with the Python implementation's own wording,
//! because a message naming `cell_size` is useless if the caller wrote
//! something else. And a cell index arrives as the tuple Python spells it
//! with, which becomes the linear index the crate addresses cells by.
//!
//! The interpreter lock is released around any query whose cost grows with
//! the map: every occupancy query, the tree build, the neighbor
//! enumeration, and the graph-wide scans. It is held for the arithmetic
//! that does not, because releasing it there costs a caller more than it
//! returns.

use arco_core::numeric::is_close;
use arco_core::protocols::{DiscreteMap, Occupancy};
use arco_mapping::graph::{CartesianGraph, NodeId, RoadGraph, WeightedGraph};
use arco_mapping::grid::{EuclideanGrid, GridCells, ManhattanGrid};
use arco_mapping::occupancy::KdTreeOccupancy;
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use crate::core::{read_matrix, read_vector};
use crate::errors::OrRaise;

/// Relative agreement between a requested extent and the cells covering it.
///
/// The Python implementation passed this to `math.isclose` as `rel_tol`
/// when deciding whether an axis had been extended to the next whole cell,
/// and the warning it logs is what a test watches for.
const EXTENT_RELATIVE_TOLERANCE: f64 = 1e-6;

/// The largest cell count an axis may be asked for.
///
/// Well past any grid that fits in memory, and small enough that the count
/// converts to a double and back without losing a unit.
const MAX_CELLS_PER_AXIS: f64 = 1_073_741_824.0;

/// Widens a cell coordinate into meters.
///
/// A coordinate past `u32` cannot be addressed on any machine this runs
/// on, so it is reported as unbounded rather than silently truncated.
fn coordinate_as_f64(coordinate: usize) -> f64 {
    u32::try_from(coordinate).map_or(f64::INFINITY, f64::from)
}

/// Widens a signed cell index into meters, by the same rule.
fn index_as_f64(index: i64) -> f64 {
    i32::try_from(index).map_or_else(
        |_| {
            if index.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        },
        f64::from,
    )
}

/// Reads a cell index, which Python spells as a tuple of integers.
///
/// A bare integer is read as a one-axis index, matching what numpy accepts
/// for a one-dimensional grid.
fn read_index(value: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(single) = value.extract::<i64>() {
        return Ok(vec![single]);
    }
    value.extract::<Vec<i64>>()
}

/// Where a query position lands on the nearest edge, as Python sees it.
type EdgeProjection<'py> = (Bound<'py, PyArray1<f64>>, NodeId, NodeId, f64);

/// Rejects two cell indices that name different numbers of axes.
fn require_same_rank(a: &[i64], b: &[i64]) -> PyResult<()> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "cell indices name {} and {} axes, which cannot be compared.",
            a.len(),
            b.len()
        )))
    }
}

// ------------------------------------------------------------ graph root ----

/// A node in the graph.
///
/// Reachable as ``Graph.Node``, and carrying no data: the graph classes
/// identify a node by an integer id rather than by an object. It exists
/// because the Python class defined it and callers can name it.
#[pyclass(name = "Node", module = "arco._arco", subclass)]
#[derive(Debug)]
pub struct PyGraphNode;

#[pymethods]
impl PyGraphNode {
    #[new]
    #[pyo3(signature = ())]
    const fn new() -> Self {
        Self
    }
}

/// An undirected edge in the graph.
///
/// Reachable as ``Graph.Edge``. Like :class:`Graph.Node` it stores
/// nothing, since the graph classes hold their own topology.
///
/// Args:
///     node_0: The first node.
///     node_1: The second node.
#[pyclass(name = "Edge", module = "arco._arco", subclass)]
#[derive(Debug)]
pub struct PyGraphEdge;

#[pymethods]
impl PyGraphEdge {
    #[new]
    #[pyo3(signature = (node_0, node_1))]
    const fn new(node_0: &Bound<'_, PyAny>, node_1: &Bound<'_, PyAny>) -> Self {
        let (_, _) = (node_0, node_1);
        Self
    }
}

/// Representation of a graph G = (V, E).
///
/// A graph is a set of vertices V and a set of edges E. This class carries
/// the two nested names, :class:`Graph.Node` and :class:`Graph.Edge`, and
/// is the base every map in this module derives from, so a planner can
/// take any of them.
#[pyclass(name = "Graph", module = "arco._arco", subclass)]
#[derive(Debug)]
pub struct PyGraph;

#[pymethods]
impl PyGraph {
    /// Builds the base, ignoring whatever a subclass was constructed with.
    ///
    /// Python's `Graph` took no arguments and a subclass calling
    /// `super().__init__()` passed none, but a subclass constructed as
    /// `RectOccupancy(1.0, 12.0, 0.5, 1.5)` still routes those four
    /// through `Graph.__new__`, which is how `object` behaves for any
    /// class that overrides `__init__`. A generated constructor with a
    /// fixed empty signature refuses them instead, so every Python
    /// subclass of `Occupancy` stops constructing.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    const fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }
}

// ---------------------------------------------------------------- grids ----

/// The metric and neighborhood a grid was built with.
///
/// Python put the metric on a subclass of a shared base, which let a grid
/// be built with a neighborhood and a distance that disagree. The crate
/// fixes the pairing at construction, so this enum records which pairing
/// the caller asked for and the base class reads the cell geometry
/// through it. It carries no occupancy: see [`PyGrid::data`].
#[derive(Debug, Clone)]
enum Metric {
    /// Four-connected, distance along the axes.
    Manhattan(ManhattanGrid),
    /// Eight-connected, straight-line distance.
    Euclidean(EuclideanGrid),
}

impl Metric {
    /// The cells underneath, whichever pairing this is.
    const fn cells(&self) -> &GridCells {
        match *self {
            Self::Manhattan(ref grid) => grid.cells(),
            Self::Euclidean(ref grid) => grid.cells(),
        }
    }

    /// The in-bounds neighbors of a cell, as linear indices.
    ///
    /// Every cell the crate holds is free, because the occupancy state
    /// lives in the numpy array a caller writes into rather than here, so
    /// what comes back is every neighbor inside the grid. That is what the
    /// Python implementation yielded: it filtered nothing, and a planner
    /// asks `is_occupied` about each one itself.
    fn neighbors(&self, linear: usize) -> Vec<usize> {
        match *self {
            Self::Manhattan(ref grid) => grid.neighbors(linear),
            Self::Euclidean(ref grid) => grid.neighbors(linear),
        }
    }
}

/// The extent a caller asked for, resolved into cells and meters.
struct Extent {
    /// Cells along each axis.
    shape: Vec<usize>,
    /// Size of one cell on the ground, meters.
    cell_size: f64,
    /// What the cells actually cover along each axis, meters.
    physical_size: Vec<f64>,
}

/// The cell count covering `dimension` meters at `cell_size` meters a cell.
fn cells_covering(axis: usize, dimension: f64, cell_size: f64) -> PyResult<usize> {
    let exact = dimension / cell_size;
    if !exact.is_finite() || exact < 0.0 {
        return Err(PyValueError::new_err(format!(
            "physical_size[{axis}] must be finite and non-negative, got {dimension:?}."
        )));
    }
    let rounded = exact.ceil();
    if rounded > MAX_CELLS_PER_AXIS {
        return Err(PyValueError::new_err(format!(
            "physical_size[{axis}] needs more than {MAX_CELLS_PER_AXIS:.0} cells at \
             cell_size={cell_size:?}."
        )));
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is a non-negative whole number below MAX_CELLS_PER_AXIS, checked above"
    )]
    Ok(rounded as usize)
}

/// Reports an axis extended to the next whole cell, as Python did.
///
/// The Python implementation logged this through the standard `logging`
/// module, and a test reads the record back, so the report goes to the
/// same logger with the same message rather than to a Rust log sink.
fn report_extended_axis(
    py: Python<'_>,
    axis: usize,
    requested: f64,
    cell_size: f64,
    actual: f64,
    cell_count: usize,
) -> PyResult<()> {
    let logger = py
        .import("logging")?
        .call_method1("getLogger", ("arco.mapping.grid.base",))?;
    logger.call_method1(
        "warning",
        (
            "Grid axis %d: requested %.6g m is not a multiple of cell_size=%.6g m; \
             extended to %.6g m (%d cells).",
            axis,
            requested,
            cell_size,
            actual,
            cell_count,
        ),
    )?;
    Ok(())
}

/// Resolves the two ways a grid can be asked for into one extent.
fn resolve_extent(
    py: Python<'_>,
    shape: Option<Vec<i64>>,
    physical_size: Option<Vec<f64>>,
    cell_size: f64,
) -> PyResult<Extent> {
    match (shape.is_some(), physical_size.is_some()) {
        (false, false) => {
            return Err(PyValueError::new_err(
                "Provide either 'shape' (cells) or 'physical_size' (meters).",
            ));
        }
        (true, true) => {
            return Err(PyValueError::new_err(
                "Provide either 'shape' or 'physical_size', not both.",
            ));
        }
        _ => {}
    }
    if !(cell_size.is_finite() && cell_size > 0.0) {
        return Err(PyValueError::new_err(format!(
            "cell_size must be positive, got {cell_size:?}."
        )));
    }

    if let Some(extents) = physical_size {
        let mut cells = Vec::with_capacity(extents.len());
        let mut actual = Vec::with_capacity(extents.len());
        for (axis, requested) in extents.into_iter().enumerate() {
            let count = cells_covering(axis, requested, cell_size)?;
            let covered = coordinate_as_f64(count) * cell_size;
            if !is_close(covered, requested, 0.0, EXTENT_RELATIVE_TOLERANCE) {
                report_extended_axis(py, axis, requested, cell_size, covered, count)?;
            }
            cells.push(count);
            actual.push(covered);
        }
        return Ok(Extent {
            shape: cells,
            cell_size,
            physical_size: actual,
        });
    }

    let mut cells = Vec::new();
    for (axis, extent) in shape.unwrap_or_default().into_iter().enumerate() {
        cells.push(usize::try_from(extent).map_err(|_| {
            PyValueError::new_err(format!(
                "shape[{axis}] must be a non-negative cell count, got {extent}."
            ))
        })?);
    }
    let physical = cells
        .iter()
        .map(|&count| coordinate_as_f64(count) * cell_size)
        .collect();
    Ok(Extent {
        shape: cells,
        cell_size,
        physical_size: physical,
    })
}

/// An N-dimensional grid of free and occupied cells.
///
/// Each cell is free or occupied, nodes are cell indices, and edges are
/// the moves a subclass defines. The class can be built two ways:
///
/// 1. By cell count, passing *shape* as a sequence of integers. Then
///    ``cell_size`` defaults to 1.0 m.
/// 2. By metric extent, passing *physical_size* in meters together with
///    *cell_size*. The cell count along each axis is
///    ``ceil(physical_size[i] / cell_size)``. When the requested extent is
///    not a whole number of cells, the axis is extended to the next whole
///    cell and the extension is logged.
///
/// Args:
///     shape: Grid dimensions in cells. Mutually exclusive with
///         *physical_size*.
///     physical_size: Physical size of the grid in meters per axis.
///         Mutually exclusive with *shape*, and requires *cell_size*.
///     cell_size: Physical size of one cell in meters, 1.0 by default.
///
/// Raises:
///     ValueError: If neither or both of *shape* and *physical_size* are
///         given, or if *cell_size* is not positive.
///
/// Attributes:
///     shape: Grid dimensions in cells, as a tuple.
///     data: Occupancy array, 0 free and 1 occupied, dtype ``uint8``.
///     cell_size: Physical size of one cell in meters.
///     physical_size: Actual physical extent per axis in meters, as a
///         tuple.
#[pyclass(name = "Grid", module = "arco._arco", extends = PyGraph, subclass)]
#[derive(Debug)]
pub struct PyGrid {
    /// Cells along each axis, as the caller asked for them.
    shape: Vec<usize>,
    /// Size of one cell on the ground, meters.
    cell_size: f64,
    /// What the cells cover along each axis, meters.
    physical_size: Vec<f64>,
    /// The cell geometry, or nothing when an axis has zero length.
    ///
    /// The crate refuses to build a grid holding no cells, and Python
    /// allowed one, so an empty grid is represented by the absence of the
    /// storage rather than by an empty one.
    cells: Option<Metric>,
    /// Which cells are occupied, as the array Python hands out.
    ///
    /// This is the storage, not a view of something else. Python exposed
    /// `data` as a plain attribute and callers assign into it,
    /// `grid.data[40:60, 40:60] = 1` among them, so the array a getter
    /// returns has to be the one the grid reads back.
    data: Py<PyArrayDyn<u8>>,
}

impl PyGrid {
    /// Builds the base state for one of the three grid classes.
    fn build(
        py: Python<'_>,
        shape: Option<Vec<i64>>,
        physical_size: Option<Vec<f64>>,
        cell_size: f64,
        metric: fn(&[usize], f64) -> PyResult<Metric>,
    ) -> PyResult<Self> {
        let extent = resolve_extent(py, shape, physical_size, cell_size)?;
        let cells = if extent.shape.is_empty() || extent.shape.contains(&0) {
            None
        } else {
            Some(metric(&extent.shape, extent.cell_size)?)
        };
        let data = PyArrayDyn::<u8>::zeros(py, extent.shape.clone(), false).unbind();
        Ok(Self {
            shape: extent.shape,
            cell_size: extent.cell_size,
            physical_size: extent.physical_size,
            cells,
            data,
        })
    }

    /// A cell index resolved against the grid, bounds checked.
    fn resolved_index(&self, index: &[i64]) -> PyResult<Vec<usize>> {
        if index.len() != self.shape.len() {
            return Err(PyIndexError::new_err(format!(
                "index names {} axes, but the grid has {}.",
                index.len(),
                self.shape.len()
            )));
        }
        let mut resolved = Vec::with_capacity(index.len());
        for (axis, (&coordinate, &extent)) in index.iter().zip(&self.shape).enumerate() {
            let inside = usize::try_from(coordinate)
                .ok()
                .filter(|&value| value < extent);
            resolved.push(inside.ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "index {coordinate} is out of bounds for axis {axis} with size {extent}."
                ))
            })?);
        }
        Ok(resolved)
    }

    /// The linear index of a cell index, bounds checked.
    fn linear_index(&self, index: &[i64]) -> PyResult<usize> {
        let resolved = self.resolved_index(index)?;
        self.cells
            .as_ref()
            .ok_or_else(|| PyIndexError::new_err("the grid holds no cells."))?
            .cells()
            .linear_index(&resolved)
            .or_raise()
    }

    /// Writes one cell state into the occupancy array.
    fn write_cell(&self, py: Python<'_>, idx: &Bound<'_, PyAny>, state: u8) -> PyResult<()> {
        let index = self.resolved_index(&read_index(idx)?)?;
        let array = self.data.bind(py);
        let mut writable = array
            .try_readwrite()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut view = writable.as_array_mut();
        let cell = view
            .get_mut(index.as_slice())
            .ok_or_else(|| PyIndexError::new_err("the grid holds no cells."))?;
        *cell = state;
        Ok(())
    }
}

#[pymethods]
impl PyGrid {
    /// Grid dimensions in cells.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    /// Physical size of one cell in meters.
    #[getter]
    fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Actual physical extent of the grid in meters per axis.
    #[getter]
    fn physical_size<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.physical_size)
    }

    /// Occupancy array, 0 free and 1 occupied, dtype ``uint8``.
    ///
    /// The array is the grid's own storage rather than a copy of it, so
    /// assigning into a slice of it, ``grid.data[40:60, 40:60] = 1``,
    /// marks those cells occupied.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<u8>> {
        self.data.bind(py).clone()
    }

    #[setter]
    fn set_data(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        // Through numpy rather than by walking the object, so that a list,
        // a boolean array and an integer array all arrive the way the
        // Python implementation's own `np.asarray` delivered them. The
        // result is contiguous, which is what lets the grid index it.
        let replacement = py
            .import("numpy")?
            .call_method1("ascontiguousarray", (value, "uint8"))?
            .extract::<Py<PyArrayDyn<u8>>>()?;
        let shape = replacement.bind(py).shape().to_vec();
        if shape != self.shape {
            return Err(PyValueError::new_err(format!(
                "data has shape {shape:?}, but the grid has shape {:?}.",
                self.shape
            )));
        }
        self.data = replacement;
        Ok(())
    }

    /// Mark a cell as occupied.
    ///
    /// Args:
    ///     idx: Index of the cell, as a tuple of integers.
    ///
    /// Raises:
    ///     IndexError: If *idx* names the wrong number of axes or falls
    ///         outside the grid.
    #[pyo3(signature = (idx))]
    fn set_occupied(&self, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<()> {
        self.write_cell(py, idx, 1)
    }

    /// Mark a cell as free.
    ///
    /// Args:
    ///     idx: Index of the cell, as a tuple of integers.
    ///
    /// Raises:
    ///     IndexError: As :meth:`set_occupied`.
    #[pyo3(signature = (idx))]
    fn set_free(&self, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<()> {
        self.write_cell(py, idx, 0)
    }

    /// Return True if the cell is occupied.
    ///
    /// Args:
    ///     idx: Index of the cell, as a tuple of integers.
    ///
    /// Returns:
    ///     True when the cell is known to be occupied.
    ///
    /// Raises:
    ///     IndexError: As :meth:`set_occupied`.
    #[pyo3(signature = (idx))]
    fn is_occupied(&self, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<bool> {
        let index = self.resolved_index(&read_index(idx)?)?;
        let array = self.data.bind(py);
        let readable = array
            .try_readonly()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        // Exactly 1 is occupied, matching `data[idx] == 1` in the Python
        // implementation; every other value counts as free.
        Ok(readable.as_array().get(index.as_slice()) == Some(&1))
    }

    /// Return the Cartesian position of a grid cell.
    ///
    /// Each index component is multiplied by :attr:`cell_size`, so the
    /// result is in meters.
    ///
    /// Args:
    ///     idx: Cell index tuple, such as ``(row, col)`` for a 2-D grid.
    ///
    /// Returns:
    ///     Position as a numpy array of shape ``(N,)``.
    #[pyo3(signature = (idx))]
    fn position<'py>(
        &self,
        py: Python<'py>,
        idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let index = read_index(idx)?;
        let scale = self.cell_size;
        let position: Vec<f64> = index
            .into_iter()
            .map(|coordinate| index_as_f64(coordinate) * scale)
            .collect();
        Ok(PyArray1::from_vec(py, position))
    }

    /// Return the straight-line distance between two cells, in meters.
    ///
    /// This is the A\* heuristic. It measures between cell positions
    /// rather than cell indices, so a grid of non-unit cells is measured
    /// in meters like everything else, and it never exceeds the true path
    /// cost on either of the two grids below, which is what makes it
    /// admissible.
    ///
    /// Args:
    ///     a: First cell index.
    ///     b: Second cell index.
    ///
    /// Returns:
    ///     Straight-line distance in meters.
    ///
    /// Raises:
    ///     ValueError: If the two indices name different numbers of axes.
    #[pyo3(signature = (a, b))]
    fn heuristic(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
        let (from, to) = (read_index(a)?, read_index(b)?);
        require_same_rank(&from, &to)?;
        let scale = self.cell_size;
        let sum_of_squares: f64 = from
            .iter()
            .zip(&to)
            .map(|(&left, &right)| {
                let difference = (index_as_f64(left) - index_as_f64(right)) * scale;
                difference * difference
            })
            .sum();
        Ok(sum_of_squares.sqrt())
    }

    /// Yield the neighbor indices of a cell.
    ///
    /// The base class defines no neighborhood, so this raises. Use
    /// :class:`ManhattanGrid` or :class:`EuclideanGrid`.
    ///
    /// Args:
    ///     idx: Index of the cell, as a tuple of integers.
    ///
    /// Returns:
    ///     The in-bounds, traversable neighbors, each as an index tuple.
    ///
    /// Raises:
    ///     NotImplementedError: On the base class.
    ///     IndexError: If *idx* falls outside the grid.
    #[pyo3(signature = (idx))]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        idx: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let index = read_index(idx)?;
        let linear = self.linear_index(&index)?;
        let Some(metric) = self.cells.as_ref() else {
            return Ok(Vec::new());
        };
        let found = py.detach(|| {
            metric
                .neighbors(linear)
                .into_iter()
                .map(|neighbor| metric.cells().cell_index(neighbor).or_raise())
                .collect::<PyResult<Vec<Vec<usize>>>>()
        })?;
        found
            .into_iter()
            .map(|neighbor| PyTuple::new(py, neighbor))
            .collect()
    }
}

/// A grid with Manhattan (L1) connectivity and distance.
///
/// Only axis-aligned neighbors are considered, and distance is the sum of
/// the per-axis index differences.
///
/// Args:
///     shape: Grid dimensions in cells. Mutually exclusive with
///         *physical_size*.
///     physical_size: Physical size of the grid in meters per axis.
///     cell_size: Physical size of one cell in meters, 1.0 by default.
#[pyclass(name = "ManhattanGrid", module = "arco._arco", extends = PyGrid, subclass)]
#[derive(Debug)]
pub struct PyManhattanGrid;

#[pymethods]
impl PyManhattanGrid {
    #[new]
    #[pyo3(signature = (shape = None, *, physical_size = None, cell_size = 1.0))]
    fn new(
        py: Python<'_>,
        shape: Option<Vec<i64>>,
        physical_size: Option<Vec<f64>>,
        cell_size: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyGrid::build(py, shape, physical_size, cell_size, |shape, size| {
            Ok(Metric::Manhattan(
                ManhattanGrid::new_free(shape, size).or_raise()?,
            ))
        })?;
        Ok(PyClassInitializer::from(PyGraph)
            .add_subclass(base)
            .add_subclass(Self))
    }

    /// Return the L1 distance between two cell indices.
    ///
    /// Measured in cells, not in meters, which is what the Python
    /// implementation returned and what a unit step cost of 1 assumes.
    ///
    /// Args:
    ///     a: First cell index.
    ///     b: Second cell index.
    ///
    /// Returns:
    ///     Sum of the per-axis index differences, as an int.
    ///
    /// Raises:
    ///     ValueError: If the two indices name different numbers of axes.
    #[pyo3(signature = (a, b))]
    #[expect(
        clippy::unused_self,
        reason = "PyO3 needs a receiver for an instance method, and the metric belongs to the class"
    )]
    fn distance(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<u64> {
        let (from, to) = (read_index(a)?, read_index(b)?);
        require_same_rank(&from, &to)?;
        // Saturating because a sum past u64 needs indices no addressable
        // grid can hold, and an answer of u64::MAX is still ordered above
        // every reachable distance.
        Ok(from.iter().zip(&to).fold(0_u64, |total, (&left, &right)| {
            total.saturating_add(left.abs_diff(right))
        }))
    }
}

/// A grid with diagonal (L2) connectivity and distance.
///
/// Diagonal neighbors are included and distance is measured in a straight
/// line.
///
/// Args:
///     shape: Grid dimensions in cells. Mutually exclusive with
///         *physical_size*.
///     physical_size: Physical size of the grid in meters per axis.
///     cell_size: Physical size of one cell in meters, 1.0 by default.
#[pyclass(name = "EuclideanGrid", module = "arco._arco", extends = PyGrid, subclass)]
#[derive(Debug)]
pub struct PyEuclideanGrid;

#[pymethods]
impl PyEuclideanGrid {
    #[new]
    #[pyo3(signature = (shape = None, *, physical_size = None, cell_size = 1.0))]
    fn new(
        py: Python<'_>,
        shape: Option<Vec<i64>>,
        physical_size: Option<Vec<f64>>,
        cell_size: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let base = PyGrid::build(py, shape, physical_size, cell_size, |shape, size| {
            Ok(Metric::Euclidean(
                EuclideanGrid::new_free(shape, size).or_raise()?,
            ))
        })?;
        Ok(PyClassInitializer::from(PyGraph)
            .add_subclass(base)
            .add_subclass(Self))
    }

    /// Return the L2 distance between two cell indices.
    ///
    /// Measured in cells, not in meters, matching the Python
    /// implementation.
    ///
    /// Args:
    ///     a: First cell index.
    ///     b: Second cell index.
    ///
    /// Returns:
    ///     Straight-line distance in cells, as a float.
    ///
    /// Raises:
    ///     ValueError: If the two indices name different numbers of axes.
    #[pyo3(signature = (a, b))]
    #[expect(
        clippy::unused_self,
        reason = "PyO3 needs a receiver for an instance method, and the metric belongs to the class"
    )]
    fn distance(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
        let (from, to) = (read_index(a)?, read_index(b)?);
        require_same_rank(&from, &to)?;
        let sum_of_squares: f64 = from
            .iter()
            .zip(&to)
            .map(|(&left, &right)| {
                let difference = index_as_f64(left) - index_as_f64(right);
                difference * difference
            })
            .sum();
        Ok(sum_of_squares.sqrt())
    }
}

// --------------------------------------------------------------- graphs ----

/// Which layer of the graph hierarchy an instance was built as.
///
/// Python stacked these with inheritance and the crate stacks them with
/// ownership, per deviation A-03. One variant holds the state for one
/// Python class, and the base class reads whichever it was given.
#[derive(Debug)]
enum Topology {
    /// Edge weights and nothing else.
    Weighted(WeightedGraph),
    /// Edge weights plus node positions.
    Cartesian(CartesianGraph),
    /// Node positions plus per-edge road geometry.
    Road(RoadGraph),
}

impl Topology {
    /// The weighted graph underneath, whichever layer this is.
    const fn weighted(&self) -> &WeightedGraph {
        match *self {
            Self::Weighted(ref graph) => graph,
            Self::Cartesian(ref graph) => graph.topology(),
            Self::Road(ref graph) => graph.positions().topology(),
        }
    }

    /// The weighted graph underneath, mutably.
    ///
    /// Only a plain weighted graph can be edited through this: a
    /// positioned graph needs coordinates, which is why the subclasses
    /// take them.
    fn weighted_mut(&mut self) -> PyResult<&mut WeightedGraph> {
        match *self {
            Self::Weighted(ref mut graph) => Ok(graph),
            _ => Err(positioned_graph_error()),
        }
    }

    /// The positioned graph underneath.
    fn positioned(&self) -> PyResult<&CartesianGraph> {
        match *self {
            Self::Cartesian(ref graph) => Ok(graph),
            Self::Road(ref graph) => Ok(graph.positions()),
            Self::Weighted(_) => Err(unpositioned_graph_error()),
        }
    }

    /// The positioned graph underneath, mutably.
    fn positioned_mut(&mut self) -> PyResult<&mut CartesianGraph> {
        match *self {
            Self::Cartesian(ref mut graph) => Ok(graph),
            Self::Road(ref mut graph) => Ok(graph.positions_mut()),
            Self::Weighted(_) => Err(unpositioned_graph_error()),
        }
    }

    /// The road graph underneath.
    fn road(&self) -> PyResult<&RoadGraph> {
        match *self {
            Self::Road(ref graph) => Ok(graph),
            _ => Err(unpositioned_graph_error()),
        }
    }

    /// The road graph underneath, mutably.
    fn road_mut(&mut self) -> PyResult<&mut RoadGraph> {
        match *self {
            Self::Road(ref mut graph) => Ok(graph),
            _ => Err(unpositioned_graph_error()),
        }
    }
}

/// Names the mistake of editing a positioned graph without coordinates.
fn positioned_graph_error() -> PyErr {
    PyValueError::new_err("this graph carries positions; pass the node coordinates.")
}

/// Names the mistake of asking a plain graph for something positional.
fn unpositioned_graph_error() -> PyErr {
    PyValueError::new_err("this graph carries no positions.")
}

/// A generic weighted undirected graph.
///
/// Nodes are integer ids and edges carry explicit numeric weights. Nothing
/// positional lives here: positions belong to :class:`CartesianGraph`.
#[pyclass(name = "WeightedGraph", module = "arco._arco", extends = PyGraph, subclass)]
#[derive(Debug)]
pub struct PyWeightedGraph {
    /// The layer this instance was built as.
    topology: Topology,
}

#[pymethods]
impl PyWeightedGraph {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyGraph).add_subclass(Self {
            topology: Topology::Weighted(WeightedGraph::new()),
        })
    }

    /// Register a node in the graph.
    ///
    /// Args:
    ///     node_id: Unique integer identifier for the node.
    #[pyo3(signature = (node_id))]
    fn add_node(&mut self, node_id: NodeId) -> PyResult<()> {
        self.topology.weighted_mut()?.add_node(node_id);
        Ok(())
    }

    /// Add an undirected weighted edge between two nodes.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///     weight: Edge weight. Must be finite and non-negative: a
    ///         negative cost breaks the optimality an A\* result rests on,
    ///         silently, so it is refused where it enters.
    ///
    /// Raises:
    ///     ValueError: If *weight* is negative, NaN or infinite.
    #[pyo3(signature = (node_a, node_b, weight))]
    fn add_edge(&mut self, node_a: NodeId, node_b: NodeId, weight: f64) -> PyResult<()> {
        self.topology
            .weighted_mut()?
            .add_edge(node_a, node_b, weight)
            .or_raise()
    }

    /// Return the ids of all nodes directly connected to *node_id*.
    ///
    /// Args:
    ///     node_id: Id of the query node.
    ///
    /// Returns:
    ///     The neighbor ids, in a deterministic order.
    #[pyo3(signature = (node_id))]
    fn neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.topology.weighted().neighbors(node_id)
    }

    /// Return the edge weight between two adjacent nodes.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///
    /// Returns:
    ///     Edge weight as a float.
    ///
    /// Raises:
    ///     KeyError: If no edge exists between the two nodes.
    #[pyo3(signature = (node_a, node_b))]
    fn distance(&self, node_a: NodeId, node_b: NodeId) -> PyResult<f64> {
        self.topology.weighted().distance(node_a, node_b).or_raise()
    }

    /// A list of all node ids.
    #[getter]
    fn nodes(&self, py: Python<'_>) -> Vec<NodeId> {
        py.detach(|| self.topology.weighted().nodes())
    }

    /// All edges as ``(node_a, node_b, weight)`` triples.
    ///
    /// Each undirected edge appears once, with ``node_a <= node_b``.
    #[getter]
    fn edges(&self, py: Python<'_>) -> Vec<(NodeId, NodeId, f64)> {
        py.detach(|| self.topology.weighted().edges())
    }
}

/// A weighted graph whose nodes carry N-dimensional Cartesian positions.
///
/// Edge weight defaults to the straight-line distance between the two
/// endpoint positions. Any dimension works: pass two coordinates for a 2-D
/// graph, three for 3-D, and so on.
///
/// Args:
///     ndim: Expected number of spatial dimensions. When set, every later
///         :meth:`add_node` checks that the position has exactly that many
///         coordinates. Pass ``None``, the default, to take the dimension
///         from the first node added.
#[pyclass(name = "CartesianGraph", module = "arco._arco", extends = PyWeightedGraph, subclass)]
#[derive(Debug)]
pub struct PyCartesianGraph;

impl PyCartesianGraph {
    /// The graph state, which lives in the base class.
    fn state<'a>(slf: &'a PyRef<'_, Self>) -> &'a Topology {
        &slf.as_super().topology
    }

    /// The graph state, mutably.
    fn state_mut<'a>(slf: &'a mut PyRefMut<'_, Self>) -> &'a mut Topology {
        &mut slf.as_super().topology
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 fixes the receiver of a subclass method, and *args, as owned values"
)]
#[pymethods]
impl PyCartesianGraph {
    #[new]
    #[pyo3(signature = (ndim = None))]
    fn new(ndim: Option<usize>) -> PyClassInitializer<Self> {
        let graph = ndim.map_or_else(CartesianGraph::new, CartesianGraph::with_dimension);
        let base = PyWeightedGraph {
            topology: Topology::Cartesian(graph),
        };
        PyClassInitializer::from(PyGraph)
            .add_subclass(base)
            .add_subclass(Self)
    }

    /// Add a node with a Cartesian position.
    ///
    /// Args:
    ///     node_id: Unique integer identifier for the node.
    ///     *coords: Coordinate values defining the node position. Pass two
    ///         floats ``(x, y)`` for a 2-D graph, three for 3-D, and so
    ///         on.
    ///
    /// Raises:
    ///     ValueError: If *coords* is empty, or its length disagrees with
    ///         the dimension of the nodes already added.
    #[pyo3(signature = (node_id, *coords))]
    fn add_node(mut slf: PyRefMut<'_, Self>, node_id: NodeId, coords: Vec<f64>) -> PyResult<()> {
        Self::state_mut(&mut slf)
            .positioned_mut()?
            .add_node(node_id, &coords)
            .or_raise()
    }

    /// Add an undirected weighted edge between two nodes.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///     weight: Edge weight. Defaults to the straight-line distance
    ///         between the two node positions when ``None``.
    ///
    /// Raises:
    ///     KeyError: If either node has no position.
    ///     ValueError: If *weight* is negative, NaN or infinite.
    #[pyo3(signature = (node_a, node_b, weight = None))]
    fn add_edge(
        mut slf: PyRefMut<'_, Self>,
        node_a: NodeId,
        node_b: NodeId,
        weight: Option<f64>,
    ) -> PyResult<()> {
        Self::state_mut(&mut slf)
            .positioned_mut()?
            .add_edge(node_a, node_b, weight)
            .or_raise()
    }

    /// Return the Cartesian position of a node.
    ///
    /// Args:
    ///     node_id: Id of the node.
    ///
    /// Returns:
    ///     Position as a numpy array of shape ``(N,)``.
    ///
    /// Raises:
    ///     KeyError: If the node is not in the graph.
    #[pyo3(signature = (node_id))]
    fn position<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        node_id: NodeId,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let position = Self::state(&slf)
            .positioned()?
            .position(node_id)
            .or_raise()?;
        Ok(PyArray1::from_slice(py, position))
    }

    /// Number of spatial dimensions, or ``None`` before the first node.
    #[getter]
    fn ndim(slf: PyRef<'_, Self>) -> PyResult<Option<usize>> {
        Ok(Self::state(&slf).positioned()?.dimension())
    }

    /// Return the edge weight between two nodes.
    ///
    /// Falls back to the straight-line distance between their positions
    /// when no edge joins them, which is what lets a caller ask about any
    /// pair rather than only adjacent ones.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///
    /// Returns:
    ///     Edge weight as a float.
    ///
    /// Raises:
    ///     KeyError: If either node is not in the graph.
    #[pyo3(signature = (node_a, node_b))]
    fn distance(slf: PyRef<'_, Self>, node_a: NodeId, node_b: NodeId) -> PyResult<f64> {
        let graph = Self::state(&slf).positioned()?;
        match graph.distance(node_a, node_b) {
            Ok(weight) => Ok(weight),
            Err(_) => graph.heuristic(node_a, node_b).or_raise(),
        }
    }

    /// Return the straight-line distance between two node positions.
    ///
    /// Admissible and consistent for A\* on any Cartesian graph, whatever
    /// its dimension, as long as no edge is priced below its own length.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///
    /// Returns:
    ///     Straight-line distance as a float.
    ///
    /// Raises:
    ///     KeyError: If either node is not in the graph.
    #[pyo3(signature = (node_a, node_b))]
    fn heuristic(slf: PyRef<'_, Self>, node_a: NodeId, node_b: NodeId) -> PyResult<f64> {
        Self::state(&slf)
            .positioned()?
            .heuristic(node_a, node_b)
            .or_raise()
    }

    /// Return the id of the node closest to an N-D position.
    ///
    /// Ties break toward the lower node id, so repeated runs agree.
    ///
    /// Args:
    ///     position: Query position as a numpy array of shape ``(N,)``.
    ///     max_radius: Maximum search radius. When set, only nodes within
    ///         this distance count.
    ///
    /// Returns:
    ///     Id of the nearest node, or ``None`` when the graph is empty or
    ///     nothing falls within *max_radius*.
    ///
    /// Raises:
    ///     ValueError: If *position* disagrees with the graph's dimension
    ///         or carries a NaN.
    #[pyo3(signature = (position, max_radius = None))]
    fn find_nearest_node(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        position: &Bound<'_, PyAny>,
        max_radius: Option<f64>,
    ) -> PyResult<Option<NodeId>> {
        let query = read_vector(position)?;
        let graph = Self::state(&slf).positioned()?;
        py.detach(|| graph.find_nearest_node(&query, max_radius))
            .or_raise()
    }

    /// Project a point onto the nearest edge of the graph.
    ///
    /// The closest point on any edge is found by perpendicular projection
    /// onto each edge segment, clamped to the segment. Works in any
    /// dimension.
    ///
    /// Args:
    ///     position: Query position as a numpy array of shape ``(N,)``.
    ///     max_radius: Maximum search radius. When set, a projection
    ///         further away than this is not returned.
    ///
    /// Returns:
    ///     ``(proj, node_a, node_b, distance)`` where *proj* is the
    ///     projected point as a numpy array, *node_a* and *node_b* are the
    ///     endpoints of the nearest edge, and *distance* is how far the
    ///     query lies from the projection. ``None`` when the graph has no
    ///     edges or none falls within *max_radius*.
    ///
    /// Raises:
    ///     ValueError: As :meth:`find_nearest_node`.
    #[pyo3(signature = (position, max_radius = None))]
    fn project_to_nearest_edge<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        position: &Bound<'py, PyAny>,
        max_radius: Option<f64>,
    ) -> PyResult<Option<EdgeProjection<'py>>> {
        let query = read_vector(position)?;
        let graph = Self::state(&slf).positioned()?;
        let found = py
            .detach(|| graph.project_to_nearest_edge(&query))
            .or_raise()?;
        Ok(found
            .filter(|projection| max_radius.is_none_or(|radius| projection.distance <= radius))
            .map(|projection| {
                (
                    PyArray1::from_vec(py, projection.point),
                    projection.from,
                    projection.to,
                    projection.distance,
                )
            }))
    }
}

/// A Cartesian graph whose edges carry road geometry.
///
/// Alongside everything :class:`CartesianGraph` does, each edge stores the
/// ordered intermediate waypoints that describe the road between its two
/// endpoints. The waypoints exclude the endpoints themselves, and they
/// feed spline interpolation, path smoothing and trajectory generation.
#[pyclass(name = "RoadGraph", module = "arco._arco", extends = PyCartesianGraph, subclass)]
#[derive(Debug)]
pub struct PyRoadGraph;

impl PyRoadGraph {
    /// The graph state, which lives two classes up.
    fn state<'a>(slf: &'a PyRef<'_, Self>) -> &'a Topology {
        &slf.as_super().as_super().topology
    }

    /// The graph state, mutably.
    fn state_mut<'a>(slf: &'a mut PyRefMut<'_, Self>) -> &'a mut Topology {
        &mut slf.as_super().as_super().topology
    }

    /// The waypoints of an edge, in the order they were stored.
    ///
    /// Stored against the ordered endpoint pair, so asking in either
    /// direction gives the same list, and an edge that does not exist
    /// gives an empty one rather than raising.
    fn stored_geometry(graph: &RoadGraph, node_a: NodeId, node_b: NodeId) -> Vec<Vec<f64>> {
        let (low, high) = if node_a <= node_b {
            (node_a, node_b)
        } else {
            (node_b, node_a)
        };
        graph.edge_geometry(low, high).unwrap_or_default()
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 fixes the receiver of a subclass method, and *args, as owned values"
)]
#[pymethods]
impl PyRoadGraph {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> PyClassInitializer<Self> {
        let base = PyWeightedGraph {
            topology: Topology::Road(RoadGraph::new()),
        };
        PyClassInitializer::from(PyGraph)
            .add_subclass(base)
            .add_subclass(PyCartesianGraph)
            .add_subclass(Self)
    }

    /// Add an undirected weighted edge with optional geometry waypoints.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///     weight: Edge weight. Defaults to the straight-line distance
    ///         between the two node positions when ``None``.
    ///     waypoints: Intermediate ``(x, y)`` points along the edge,
    ///         describing the road geometry between the two nodes. They
    ///         are stored in the order given, running from the lower node
    ///         id to the higher.
    ///
    /// Raises:
    ///     KeyError: If either node has no position.
    ///     ValueError: If a waypoint carries a NaN or disagrees with the
    ///         graph's dimension.
    #[pyo3(signature = (node_a, node_b, weight = None, waypoints = None))]
    fn add_edge(
        mut slf: PyRefMut<'_, Self>,
        node_a: NodeId,
        node_b: NodeId,
        weight: Option<f64>,
        waypoints: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let geometry = match waypoints {
            Some(value) => read_matrix(value)?,
            None => Vec::new(),
        };
        Self::state_mut(&mut slf)
            .road_mut()?
            .add_edge(node_a, node_b, weight, &geometry)
            .or_raise()
    }

    /// Return the waypoints defining the geometry of an edge.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///
    /// Returns:
    ///     The ``(x, y)`` waypoints, empty when the edge carries none or
    ///     does not exist.
    #[pyo3(signature = (node_a, node_b))]
    fn edge_geometry<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        node_a: NodeId,
        node_b: NodeId,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let graph = Self::state(&slf).road()?;
        Self::stored_geometry(graph, node_a, node_b)
            .into_iter()
            .map(|waypoint| PyTuple::new(py, waypoint))
            .collect()
    }

    /// Return the complete edge geometry, endpoints included.
    ///
    /// Every element is a numpy array, so the result reads the same way in
    /// any dimension. The waypoints run in travel order: a caller going
    /// from the higher node id to the lower gets them reversed.
    ///
    /// Args:
    ///     node_a: Id of the first node.
    ///     node_b: Id of the second node.
    ///
    /// Returns:
    ///     Position arrays starting at *node_a*, through every
    ///     intermediate waypoint, and ending at *node_b*.
    ///
    /// Raises:
    ///     KeyError: If either node is not in the graph.
    #[pyo3(signature = (node_a, node_b))]
    fn full_edge_geometry<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        node_a: NodeId,
        node_b: NodeId,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        let graph = Self::state(&slf).road()?;
        let positions = graph.positions();
        let mut path = vec![positions.position(node_a).or_raise()?.to_vec()];
        let stored = Self::stored_geometry(graph, node_a, node_b);
        if node_a <= node_b {
            path.extend(stored);
        } else {
            path.extend(stored.into_iter().rev());
        }
        path.push(positions.position(node_b).or_raise()?.to_vec());
        Ok(path
            .into_iter()
            .map(|point| PyArray1::from_vec(py, point))
            .collect())
    }
}

// ------------------------------------------------------------ occupancy ----

/// A continuous, sparse occupancy map backed by a k-d tree.
///
/// Only the obstacle points are stored, never empty space, and a k-d tree
/// answers the nearest-neighbor queries. A point counts as occupied when
/// it lies within *clearance* of an obstacle.
///
/// Args:
///     points: Obstacle point coordinates, one point per row. Accepts a
///         numpy array of shape ``(N, D)`` or any sequence of sequences,
///         where *D* is the spatial dimension.
///     clearance: Minimum safe distance from any obstacle point, in
///         meters.
///
/// Raises:
///     ValueError: If *points* is empty, its rows disagree in length, or
///         *clearance* is not positive.
#[pyclass(name = "KDTreeOccupancy", module = "arco._arco", extends = PyGraph, subclass, frozen)]
#[derive(Debug)]
pub struct PyKdTreeOccupancy {
    inner: KdTreeOccupancy,
}

#[pymethods]
impl PyKdTreeOccupancy {
    #[new]
    #[pyo3(signature = (points, clearance = 0.5))]
    fn new(
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        clearance: f64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let obstacles = read_matrix(points)?;
        if obstacles.is_empty() {
            return Err(PyValueError::new_err(
                "points must contain at least one obstacle.",
            ));
        }
        if !clearance.is_finite() || clearance <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "clearance must be positive, got {clearance:?}."
            )));
        }
        let inner = py
            .detach(|| KdTreeOccupancy::new(&obstacles, clearance))
            .or_raise()?;
        Ok(PyClassInitializer::from(PyGraph).add_subclass(Self { inner }))
    }

    /// Return the distance and position of the nearest obstacle.
    ///
    /// The distance is measured to the obstacle point itself, which is what
    /// the pure-Python implementation returned and what `FR-API-02` keeps
    /// unchanged.
    ///
    /// Args:
    ///     point: Query position as a numpy array of shape ``(D,)``.
    ///
    /// Returns:
    ///     A ``(distance, nearest_point)`` pair, where *nearest_point* is
    ///     a numpy array holding the obstacle's coordinates.
    ///
    /// Raises:
    ///     ValueError: If *point* has the wrong dimension or carries a
    ///         NaN.
    #[pyo3(signature = (point))]
    fn nearest_obstacle<'py>(
        &self,
        py: Python<'py>,
        point: &Bound<'py, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let query = read_vector(point)?;
        let nearest = py
            .detach(|| self.inner.nearest_obstacle(&query))
            .or_raise()?;
        // The crate measures from the obstacle's surface, per deviation
        // A-16, which is the number a planner asking about clearance
        // wants. Python was given the distance to the obstacle's centre,
        // so the clearance goes back on here. `ArtificialPotentialField`
        // in arco-control converts the same way, for the same reason.
        let centre_distance = nearest.distance + self.inner.clearance();
        Ok((centre_distance, PyArray1::from_vec(py, nearest.point)))
    }

    /// Return True if *point* lies within the clearance of an obstacle.
    ///
    /// Args:
    ///     point: Query position as a numpy array of shape ``(D,)``.
    ///
    /// Returns:
    ///     True when the nearest obstacle is no further away than
    ///     :attr:`clearance`.
    ///
    /// Raises:
    ///     ValueError: As :meth:`nearest_obstacle`.
    #[pyo3(signature = (point))]
    fn is_occupied(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<bool> {
        let query = read_vector(point)?;
        py.detach(|| self.inner.is_occupied(&query)).or_raise()
    }

    /// Return the distance from each query point to its nearest obstacle.
    ///
    /// The batch counterpart of :meth:`nearest_obstacle`, which walks the
    /// tree once per row with the interpreter lock released.
    ///
    /// Args:
    ///     points: Query positions as a numpy array of shape ``(M, D)``.
    ///
    /// Returns:
    ///     Distance array of shape ``(M,)``, where entry *i* is how far
    ///     ``points[i]`` lies from the nearest obstacle point.
    ///
    /// Raises:
    ///     ValueError: If a row has the wrong dimension or carries a NaN.
    #[pyo3(signature = (points))]
    fn query_distances<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let queries = read_matrix(points)?;
        let clearance = self.inner.clearance();
        let distances = py.detach(|| {
            // Centre distances, as in `nearest_obstacle` above.
            self.inner
                .query_distances(&queries)
                .map(|found| found.into_iter().map(|each| each + clearance).collect())
        });
        let distances: Vec<f64> = distances.or_raise()?;
        Ok(PyArray1::from_vec(py, distances))
    }

    /// Return True if the segment from *a* to *b* is collision-free.
    ///
    /// The segment is sampled *sample_count* times, endpoints included,
    /// and each sample is tested. Sampling is what the planners were tuned
    /// against, and it is not exact: an obstacle thinner than the spacing
    /// between two samples is stepped over, which is why a returned path
    /// is re-checked at a stated resolution rather than declared safe.
    ///
    /// Args:
    ///     a: Segment start, as a numpy array of shape ``(D,)``.
    ///     b: Segment end, of the same dimension.
    ///     sample_count: Number of sample points including both endpoints.
    ///         Values below 2 are treated as 2.
    ///
    /// Returns:
    ///     True when every sample is free.
    ///
    /// Raises:
    ///     ValueError: If the endpoints disagree in dimension or carry a
    ///         NaN.
    #[pyo3(signature = (a, b, *, sample_count = 12))]
    fn segment_free(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        sample_count: usize,
    ) -> PyResult<bool> {
        let from = read_vector(a)?;
        let to = read_vector(b)?;
        py.detach(|| self.inner.is_segment_free_with(&from, &to, sample_count))
            .or_raise()
    }

    /// Obstacle points, as an array of shape ``(N, D)``.
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        PyArray2::from_vec2(py, self.inner.points())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Spatial dimension of the obstacle space.
    #[getter]
    fn dimension(&self) -> usize {
        Occupancy::dimension(&self.inner)
    }

    /// Minimum safe distance from any obstacle point, in meters.
    #[getter]
    fn clearance(&self) -> f64 {
        self.inner.clearance()
    }
}

/// Adds `arco.mapping` to the compiled module.
///
/// # Arguments
///
/// * `module` - The module every ARCO binding registers into.
///
/// # Errors
///
/// Returns an error when a class fails to register, which the interpreter
/// surfaces as an `ImportError`.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGraph>()?;
    module.add_class::<PyGrid>()?;
    module.add_class::<PyManhattanGrid>()?;
    module.add_class::<PyEuclideanGrid>()?;
    module.add_class::<PyWeightedGraph>()?;
    module.add_class::<PyCartesianGraph>()?;
    module.add_class::<PyRoadGraph>()?;
    module.add_class::<PyKdTreeOccupancy>()?;

    // Python declared these inside the class body. PyO3 registers a class
    // at module level, so they are attached to the type afterwards, which
    // is what makes `Graph.Node` and `Graph.Edge` resolve and what lets
    // every subclass inherit them.
    let graph = module.getattr("Graph")?;
    graph.setattr("Node", module.py().get_type::<PyGraphNode>())?;
    graph.setattr("Edge", module.py().get_type::<PyGraphEdge>())?;
    Ok(())
}
