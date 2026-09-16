// Copyright 2026 alexandre
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Grid invariants: FR-INV-11 and FR-INV-12, plus the metric contracts.

use arco_core::Error;
use arco_core::protocols::DiscreteMap;
use arco_mapping::grid::{Cell, EuclideanGrid, GridCells, ManhattanGrid};

#[test]
fn index_and_world_coordinates_round_trip() {
    // FR-INV-11.
    let cells = GridCells::new_free(&[4, 5, 3], 0.25).unwrap();
    for linear in 0..cells.cell_count() {
        let index = cells.cell_index(linear).unwrap();
        assert_eq!(cells.linear_index(&index).unwrap(), linear, "{index:?}");
    }
}

#[test]
fn an_out_of_bounds_index_is_rejected_by_the_safe_accessor() {
    // FR-INV-11.
    let cells = GridCells::new_free(&[3, 3], 1.0).unwrap();
    assert!(matches!(
        cells.linear_index(&[3, 0]),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        cells.cell(cells.cell_count()),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        cells.linear_index(&[0, 0, 0]),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn an_unknown_cell_is_not_free() {
    // FR-INV-11, the whole point of deviation A-11.
    let mut cells = GridCells::new_free(&[2, 2], 1.0).unwrap();
    cells.set_cell(0, Cell::Unknown).unwrap();

    assert!(cells.blocks(0).unwrap(), "unknown must block travel");
    assert!(
        !cells.is_occupied(0).unwrap(),
        "unknown is not the same claim as occupied"
    );
    assert!(!cells.blocks(1).unwrap(), "an untouched cell is free");
}

#[test]
fn a_grid_built_from_observations_starts_unknown() {
    let cells = GridCells::new_unknown(&[2, 2], 1.0).unwrap();
    for linear in 0..cells.cell_count() {
        assert_eq!(cells.cell(linear).unwrap(), Cell::Unknown);
        assert!(cells.blocks(linear).unwrap());
    }
}

#[test]
fn the_content_hash_changes_with_any_cell() {
    // FR-INV-12.
    let mut cells = GridCells::new_free(&[4, 4], 1.0).unwrap();
    let original = cells.content_hash();

    for linear in 0..cells.cell_count() {
        let mut probe = cells.clone();
        probe.set_cell(linear, Cell::Occupied).unwrap();
        assert_ne!(
            probe.content_hash(),
            original,
            "cell {linear} did not move the hash"
        );
    }

    cells.set_cell(0, Cell::Occupied).unwrap();
    cells.set_cell(0, Cell::Free).unwrap();
    assert_eq!(
        cells.content_hash(),
        original,
        "the hash did not return after an undone change"
    );
}

#[test]
fn the_content_hash_separates_shape_and_cell_size() {
    let square = GridCells::new_free(&[4, 4], 1.0).unwrap();
    let oblong = GridCells::new_free(&[2, 8], 1.0).unwrap();
    let finer = GridCells::new_free(&[4, 4], 0.5).unwrap();
    assert_ne!(square.content_hash(), oblong.content_hash());
    assert_ne!(square.content_hash(), finer.content_hash());
}

#[test]
fn the_content_hash_does_not_depend_on_the_order_of_edits() {
    let build = |order: &[usize]| {
        let mut cells = GridCells::new_free(&[3, 3], 1.0).unwrap();
        for &linear in order {
            cells.set_cell(linear, Cell::Occupied).unwrap();
        }
        cells.content_hash()
    };
    assert_eq!(build(&[0, 4, 8]), build(&[8, 0, 4]));
}

#[test]
fn a_cell_position_scales_with_the_cell_size() {
    let cells = GridCells::new_free(&[3, 3], 0.5).unwrap();
    let linear = cells.linear_index(&[2, 1]).unwrap();
    let position = cells.position(linear).unwrap();
    assert!((position[0] - 1.0).abs() < 1e-12, "{position:?}");
    assert!((position[1] - 0.5).abs() < 1e-12, "{position:?}");
}

#[test]
fn a_manhattan_grid_is_four_connected() {
    let grid = ManhattanGrid::new_free(&[5, 5], 1.0).unwrap();
    let center = grid.cells().linear_index(&[2, 2]).unwrap();
    assert_eq!(grid.neighbors(center).len(), 4);
}

#[test]
fn a_euclidean_grid_is_eight_connected() {
    let grid = EuclideanGrid::new_free(&[5, 5], 1.0).unwrap();
    let center = grid.cells().linear_index(&[2, 2]).unwrap();
    assert_eq!(grid.neighbors(center).len(), 8);
}

#[test]
fn a_corner_has_fewer_neighbors_than_a_center() {
    let grid = EuclideanGrid::new_free(&[5, 5], 1.0).unwrap();
    let corner = grid.cells().linear_index(&[0, 0]).unwrap();
    assert_eq!(grid.neighbors(corner).len(), 3);
}

#[test]
fn a_blocked_cell_is_not_a_neighbor() {
    let mut grid = ManhattanGrid::new_free(&[3, 3], 1.0).unwrap();
    let center = grid.cells().linear_index(&[1, 1]).unwrap();
    let above = grid.cells().linear_index(&[0, 1]).unwrap();
    assert_eq!(grid.neighbors(center).len(), 4);

    grid.cells_mut().set_cell(above, Cell::Occupied).unwrap();
    assert_eq!(grid.neighbors(center).len(), 3);
    assert!(!grid.neighbors(center).contains(&above));

    grid.cells_mut().set_cell(above, Cell::Unknown).unwrap();
    assert!(
        !grid.neighbors(center).contains(&above),
        "FR-INV-11: unknown is not traversable"
    );
}

#[test]
fn the_heuristic_never_exceeds_the_distance_it_estimates() {
    // FR-INV-06 rests on admissibility. An overestimate silently breaks
    // the optimality the A* differential test asserts.
    let manhattan = ManhattanGrid::new_free(&[6, 6], 0.5).unwrap();
    let euclidean = EuclideanGrid::new_free(&[6, 6], 0.5).unwrap();

    for from in 0..36_usize {
        for to in 0..36_usize {
            let estimate = manhattan.heuristic(from, to).unwrap();
            let actual = manhattan.distance(from, to).unwrap();
            assert!(estimate <= actual + 1e-12, "manhattan {from}->{to}");

            let estimate = euclidean.heuristic(from, to).unwrap();
            let actual = euclidean.distance(from, to).unwrap();
            assert!(estimate <= actual + 1e-12, "euclidean {from}->{to}");
        }
    }
}

#[test]
fn a_manhattan_step_costs_one_cell() {
    let grid = ManhattanGrid::new_free(&[4, 4], 2.0).unwrap();
    let center = grid.cells().linear_index(&[1, 1]).unwrap();
    for neighbor in grid.neighbors(center) {
        let cost = grid.distance(center, neighbor).unwrap();
        assert!((cost - 2.0).abs() < 1e-12, "{cost}");
    }
}

#[test]
fn a_degenerate_grid_is_rejected() {
    assert!(GridCells::new_free(&[], 1.0).is_err());
    assert!(GridCells::new_free(&[3, 0], 1.0).is_err());
    assert!(GridCells::new_free(&[3, 3], 0.0).is_err());
    assert!(GridCells::new_free(&[3, 3], -1.0).is_err());
    assert!(GridCells::new_free(&[3, 3], f64::NAN).is_err());
}

#[test]
fn a_grid_works_in_any_dimension() {
    // FR-CORE-03.
    for dimension in 1..5_usize {
        let shape = vec![3_usize; dimension];
        let grid = ManhattanGrid::new_free(&shape, 1.0).unwrap();
        assert_eq!(grid.cells().dimension(), dimension);

        let center = vec![1_usize; dimension];
        let linear = grid.cells().linear_index(&center).unwrap();
        assert_eq!(
            grid.neighbors(linear).len(),
            2 * dimension,
            "dimension {dimension}"
        );
    }
}
