//! Cell storage shared by every grid metric.

use arco_core::Error;

/// What is known about one grid cell.
///
/// Python grids are binary. The third state exists because `FR-INV-11`
/// requires that an unknown cell never be silently treated as free, which
/// a boolean cannot express. See deviation A-11.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cell {
    /// Known to be traversable.
    #[default]
    Free,
    /// Known to be blocked.
    Occupied,
    /// Never observed. Blocked unless the caller says otherwise.
    Unknown,
}

impl Cell {
    /// Whether this cell blocks travel, counting unknown as blocked.
    #[must_use]
    pub const fn blocks(self) -> bool {
        !matches!(self, Self::Free)
    }
}

/// The cells of a grid, their extent, and their size on the ground.
///
/// Not a grid on its own: a grid also needs a metric and a neighborhood,
/// which is what [`super::ManhattanGrid`] and [`super::EuclideanGrid`]
/// add. Keeping the storage separate is what stops a four-connected grid
/// from being paired with a straight-line metric, which the Python
/// hierarchy allowed by subclassing incorrectly.
///
/// Cells are addressed by a linear index in row-major order, which is what
/// makes a grid usable as a [`arco_core::protocols::DiscreteMap`] whose
/// node type has to be `Copy`.
#[derive(Debug, Clone)]
pub struct GridCells {
    shape: Vec<usize>,
    cell_size: f64,
    cells: Vec<Cell>,
    content_hash: u64,
}

impl GridCells {
    /// Builds a grid whose cells are all free.
    ///
    /// Free rather than unknown, matching the Python default so that no
    /// existing behavior changes. A grid built from observations wants
    /// [`GridCells::new_unknown`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when `shape` is empty or names a
    /// zero-length axis, and [`Error::OutOfRange`] when `cell_size` is not
    /// strictly positive and finite.
    pub fn new_free(shape: &[usize], cell_size: f64) -> Result<Self, Error> {
        Self::filled(shape, cell_size, Cell::Free)
    }

    /// Builds a grid whose cells are all unknown.
    ///
    /// # Errors
    ///
    /// As [`GridCells::new_free`].
    pub fn new_unknown(shape: &[usize], cell_size: f64) -> Result<Self, Error> {
        Self::filled(shape, cell_size, Cell::Unknown)
    }

    fn filled(shape: &[usize], cell_size: f64, fill: Cell) -> Result<Self, Error> {
        if shape.is_empty() {
            return Err(Error::TooFew {
                quantity: "grid axes",
                minimum: 1,
                actual: 0,
            });
        }
        if shape.contains(&0) {
            return Err(Error::TooFew {
                quantity: "cells along an axis",
                minimum: 1,
                actual: 0,
            });
        }
        if !(cell_size.is_finite() && cell_size > 0.0) {
            return Err(Error::OutOfRange {
                quantity: "cell_size",
                value: cell_size,
                bound: "(0, inf)",
            });
        }

        let count = shape
            .iter()
            .try_fold(1_usize, |product, &axis| product.checked_mul(axis))
            .ok_or(Error::OutOfRange {
                quantity: "cell count",
                value: f64::INFINITY,
                bound: "addressable memory",
            })?;

        let mut grid = Self {
            shape: shape.to_vec(),
            cell_size,
            cells: vec![fill; count],
            content_hash: 0,
        };
        grid.content_hash = grid.recompute_hash();
        Ok(grid)
    }

    /// Cells along each axis.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of axes.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.shape.len()
    }

    /// Total cell count.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Size of one cell on the ground, meters.
    #[must_use]
    pub const fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// A hash of the extent, the cell size, and every cell state.
    ///
    /// `FR-INV-12`. Without this, "the returned path is collision-free
    /// under the same map" names no particular map and cannot be checked.
    /// The hash is maintained as cells change rather than recomputed, so
    /// reading it costs nothing.
    #[must_use]
    pub const fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// The linear index of a multi-dimensional cell index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `index` has the wrong
    /// number of axes, or [`Error::OutOfRange`] when it leaves the grid.
    pub fn linear_index(&self, index: &[usize]) -> Result<usize, Error> {
        if index.len() != self.shape.len() {
            return Err(Error::DimensionMismatch {
                quantity: "cell index",
                expected: self.shape.len(),
                actual: index.len(),
            });
        }

        let mut linear = 0_usize;
        for (axis, (&coordinate, &extent)) in index.iter().zip(&self.shape).enumerate() {
            if coordinate >= extent {
                return Err(Error::OutOfRange {
                    quantity: "cell index",
                    value: coordinate_as_f64(coordinate),
                    bound: axis_bound(axis),
                });
            }
            linear = linear
                .checked_mul(extent)
                .and_then(|scaled| scaled.checked_add(coordinate))
                .ok_or(Error::OutOfRange {
                    quantity: "cell index",
                    value: f64::INFINITY,
                    bound: "addressable memory",
                })?;
        }
        Ok(linear)
    }

    /// The multi-dimensional index of a linear one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `linear` is past the last cell.
    pub fn cell_index(&self, linear: usize) -> Result<Vec<usize>, Error> {
        if linear >= self.cells.len() {
            return Err(Error::OutOfRange {
                quantity: "linear cell index",
                value: coordinate_as_f64(linear),
                bound: "[0, cell_count)",
            });
        }

        let mut remaining = linear;
        let mut index = vec![0_usize; self.shape.len()];
        for (slot, &extent) in index.iter_mut().zip(&self.shape).rev() {
            *slot = remaining.checked_rem(extent).unwrap_or(0);
            remaining = remaining.checked_div(extent).unwrap_or(0);
        }
        Ok(index)
    }

    /// The state of one cell.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `linear` is past the last cell.
    pub fn cell(&self, linear: usize) -> Result<Cell, Error> {
        self.cells.get(linear).copied().ok_or(Error::OutOfRange {
            quantity: "linear cell index",
            value: coordinate_as_f64(linear),
            bound: "[0, cell_count)",
        })
    }

    /// Sets the state of one cell.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `linear` is past the last cell.
    pub fn set_cell(&mut self, linear: usize, state: Cell) -> Result<(), Error> {
        let previous = self.cell(linear)?;
        if previous == state {
            return Ok(());
        }
        // The hash mixes each cell independently and combines by xor, so a
        // change costs one removal and one insertion rather than a sweep.
        self.content_hash ^= cell_term(linear, previous);
        self.content_hash ^= cell_term(linear, state);
        if let Some(slot) = self.cells.get_mut(linear) {
            *slot = state;
        }
        Ok(())
    }

    /// Whether a cell is known to be blocked.
    ///
    /// Keeps the Python meaning: true only for an occupied cell, never for
    /// an unknown one. Travel decisions use [`GridCells::blocks`].
    ///
    /// # Errors
    ///
    /// As [`GridCells::cell`].
    pub fn is_occupied(&self, linear: usize) -> Result<bool, Error> {
        Ok(self.cell(linear)? == Cell::Occupied)
    }

    /// Whether a cell blocks travel, counting unknown as blocked.
    ///
    /// # Errors
    ///
    /// As [`GridCells::cell`].
    pub fn blocks(&self, linear: usize) -> Result<bool, Error> {
        Ok(self.cell(linear)?.blocks())
    }

    /// The center of a cell in world coordinates, meters.
    ///
    /// # Errors
    ///
    /// As [`GridCells::cell_index`].
    pub fn position(&self, linear: usize) -> Result<Vec<f64>, Error> {
        Ok(self
            .cell_index(linear)?
            .into_iter()
            .map(|coordinate| coordinate_as_f64(coordinate) * self.cell_size)
            .collect())
    }

    fn recompute_hash(&self) -> u64 {
        let mut hash = mix(widen(self.shape.len()) ^ 0x9e37_79b9_7f4a_7c15);
        for (axis, &extent) in self.shape.iter().enumerate() {
            hash ^= mix(widen(axis).wrapping_mul(0x517c_c1b7_2722_0a95) ^ widen(extent));
        }
        hash ^= mix(self.cell_size.to_bits());
        for (linear, &state) in self.cells.iter().enumerate() {
            hash ^= cell_term(linear, state);
        }
        hash
    }
}

/// One cell's contribution to the content hash.
fn cell_term(linear: usize, state: Cell) -> u64 {
    let tag = match state {
        Cell::Free => 1_u64,
        Cell::Occupied => 2,
        Cell::Unknown => 3,
    };
    mix(widen(linear).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ tag)
}

/// A 64 bit avalanche, so a one-cell change moves every bit.
const fn mix(value: u64) -> u64 {
    let mut mixed = value;
    mixed ^= mixed >> 33;
    mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
    mixed ^= mixed >> 33;
    mixed = mixed.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    mixed ^ (mixed >> 33)
}

/// Widens an index for hashing. Saturating rather than wrapping, since a
/// count past `u64::MAX` cannot exist on any addressable machine.
fn widen(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Widens a cell coordinate for an error message.
fn coordinate_as_f64(coordinate: usize) -> f64 {
    u32::try_from(coordinate).map_or(f64::INFINITY, f64::from)
}

/// Names the axis a bound belongs to, without allocating.
const fn axis_bound(axis: usize) -> &'static str {
    match axis {
        0 => "[0, shape[0])",
        1 => "[0, shape[1])",
        2 => "[0, shape[2])",
        _ => "[0, shape[axis])",
    }
}
