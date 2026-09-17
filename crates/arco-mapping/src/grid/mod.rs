//! Grids: cell storage plus a metric and a neighborhood.
//!
//! The Python hierarchy put the metric on a subclass of a shared `Grid`
//! base, which let a grid be built with a neighborhood and a distance that
//! disagree. Here the storage is one type and each metric is another that
//! owns it, so the pairing is fixed at construction. See deviation A-03.

mod cells;

pub use cells::{Cell, GridCells};

use arco_core::Error;
use arco_core::protocols::DiscreteMap;

/// A four-connected grid measured along the axes.
///
/// Neighbors differ by one cell on exactly one axis, and distance is the
/// sum of the per-axis differences, which is the metric that neighborhood
/// makes achievable.
#[derive(Debug, Clone)]
pub struct ManhattanGrid {
    cells: GridCells,
}

/// An eight-connected grid measured in a straight line.
///
/// Neighbors differ by at most one cell on every axis, so a diagonal move
/// exists and straight-line distance is achievable.
#[derive(Debug, Clone)]
pub struct EuclideanGrid {
    cells: GridCells,
}

/// Yields the free neighbors of `linear` under the given connectivity.
///
/// `diagonal` decides whether a move may change more than one axis.
fn neighbors_of(cells: &GridCells, linear: usize, diagonal: bool) -> Vec<usize> {
    let Ok(index) = cells.cell_index(linear) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    let mut offsets = vec![0_i64; index.len()];
    let mut candidate = vec![0_usize; index.len()];

    // Every offset in {-1, 0, 1}^dimension, skipping the origin and, for a
    // four-connected grid, anything touching more than one axis.
    let combinations = 3_usize.saturating_pow(u32::try_from(index.len()).unwrap_or(0));
    for encoded in 0..combinations {
        let mut remaining = encoded;
        let mut touched = 0_usize;
        for slot in &mut offsets {
            *slot = i64::try_from(remaining.checked_rem(3).unwrap_or(0))
                .unwrap_or(0)
                .saturating_sub(1);
            if *slot != 0 {
                touched = touched.saturating_add(1);
            }
            remaining = remaining.checked_div(3).unwrap_or(0);
        }
        if touched == 0 || (!diagonal && touched > 1) {
            continue;
        }

        let mut inside = true;
        for ((slot, &coordinate), (&offset, &extent)) in candidate
            .iter_mut()
            .zip(&index)
            .zip(offsets.iter().zip(cells.shape()))
        {
            let moved = i64::try_from(coordinate)
                .unwrap_or(0)
                .saturating_add(offset);
            let extent_wide = u64::try_from(extent).unwrap_or(u64::MAX);
            if moved < 0 || u64::try_from(moved).unwrap_or(u64::MAX) >= extent_wide {
                inside = false;
                break;
            }
            *slot = usize::try_from(moved).unwrap_or(0);
        }
        if !inside {
            continue;
        }

        if let Ok(neighbor) = cells.linear_index(&candidate) {
            // An occupied or unknown cell is not a neighbor, which is
            // FR-INV-11: unknown is never silently traversable.
            if cells.blocks(neighbor) == Ok(false) {
                found.push(neighbor);
            }
        }
    }
    found
}

macro_rules! grid_impl {
    ($grid:ident, $diagonal:expr, $distance:expr, $doc:literal) => {
        impl $grid {
            #[doc = $doc]
            ///
            /// # Errors
            ///
            /// As [`GridCells::new_free`].
            pub fn new_free(shape: &[usize], cell_size: f64) -> Result<Self, Error> {
                Ok(Self {
                    cells: GridCells::new_free(shape, cell_size)?,
                })
            }

            /// Builds a grid whose cells are all unknown.
            ///
            /// # Errors
            ///
            /// As [`GridCells::new_free`].
            pub fn new_unknown(shape: &[usize], cell_size: f64) -> Result<Self, Error> {
                Ok(Self {
                    cells: GridCells::new_unknown(shape, cell_size)?,
                })
            }

            /// The underlying cells.
            #[must_use]
            pub const fn cells(&self) -> &GridCells {
                &self.cells
            }

            /// The underlying cells, mutably.
            pub const fn cells_mut(&mut self) -> &mut GridCells {
                &mut self.cells
            }
        }

        impl DiscreteMap for $grid {
            type Node = usize;

            fn contains(&self, node: usize) -> bool {
                node < self.cells.cell_count()
            }

            fn neighbors(&self, node: usize) -> Vec<usize> {
                neighbors_of(&self.cells, node, $diagonal)
            }

            fn turn_penalty(&self, previous: Option<usize>, current: usize, next: usize) -> u8 {
                turn_penalty_of(&self.cells, previous, current, next)
            }

            fn distance(&self, from: usize, to: usize) -> Result<f64, Error> {
                let start = self.cells.position(from)?;
                let end = self.cells.position(to)?;
                $distance(&start, &end)
            }

            fn heuristic(&self, node: usize, goal: usize) -> Result<f64, Error> {
                // Straight-line distance is admissible under both metrics,
                // since no move is cheaper than the straight line it
                // covers. FR-INV-06 depends on that staying true.
                let start = self.cells.position(node)?;
                let end = self.cells.position(goal)?;
                arco_core::geometry::euclidean_distance(&start, &end)
            }
        }
    };
}

grid_impl!(
    ManhattanGrid,
    false,
    arco_core::geometry::manhattan_distance,
    "Builds a four-connected grid whose cells are all free."
);
grid_impl!(
    EuclideanGrid,
    true,
    arco_core::geometry::euclidean_distance,
    "Builds an eight-connected grid whose cells are all free."
);

/// Whether the step from `current` to `next` turns, as zero or one.
///
/// Direction is the per-axis sign of the index difference, so on a grid it
/// is exactly the eight compass headings and the comparison is integer. A
/// first step has nothing to have turned from and so never turns.
fn turn_penalty_of(cells: &GridCells, previous: Option<usize>, current: usize, next: usize) -> u8 {
    let Some(previous) = previous else { return 0 };
    let (Ok(before), Ok(here), Ok(after)) = (
        cells.cell_index(previous),
        cells.cell_index(current),
        cells.cell_index(next),
    ) else {
        return 0;
    };

    for axis in 0..here.len() {
        let incoming = step_sign(&before, &here, axis);
        let outgoing = step_sign(&here, &after, axis);
        if incoming != outgoing {
            return 1;
        }
    }
    0
}

/// The sign of the move along one axis, as minus one, zero or one.
fn step_sign(from: &[usize], to: &[usize], axis: usize) -> i8 {
    let start = from.get(axis).copied().unwrap_or_default();
    let end = to.get(axis).copied().unwrap_or_default();
    match end.cmp(&start) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    }
}
