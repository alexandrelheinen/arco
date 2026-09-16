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

//! FR-INV-06: A* returns the cost an uninformed search returns.
//!
//! This is the strongest cheap oracle in the whole port. With an
//! admissible heuristic the two searches must agree exactly on cost, and
//! Dijkstra needs no reference implementation, no recorded baseline, and
//! no tolerance.

use arco_core::protocols::DiscreteMap;
use arco_core::rng::Pcg64;
use arco_mapping::grid::{Cell, EuclideanGrid, ManhattanGrid};
use arco_planning::discrete::{SearchOptions, search};
use arco_planning::failure::PlanFailure;

fn astar() -> SearchOptions {
    SearchOptions {
        max_expansions: 1_000_000,
        use_heuristic: true,
    }
}

fn dijkstra() -> SearchOptions {
    SearchOptions {
        max_expansions: 1_000_000,
        use_heuristic: false,
    }
}

/// A grid with a seeded scatter of blocked cells.
fn scattered_grid(seed: u64, side: usize, blocked_fraction: f64) -> ManhattanGrid {
    let mut grid = ManhattanGrid::new_free(&[side, side], 1.0).unwrap();
    let mut generator = Pcg64::seed_from_u64(seed);
    let count = grid.cells().cell_count();
    for linear in 0..count {
        if generator.next_f64() < blocked_fraction {
            grid.cells_mut().set_cell(linear, Cell::Occupied).unwrap();
        }
    }
    // Keep the corners usable so most queries are answerable.
    grid.cells_mut().set_cell(0, Cell::Free).unwrap();
    grid.cells_mut().set_cell(count - 1, Cell::Free).unwrap();
    grid
}

#[test]
fn astar_and_dijkstra_agree_on_cost() {
    // FR-INV-06, over many maps rather than one.
    for seed in 0..20_u64 {
        let grid = scattered_grid(seed, 12, 0.2);
        let goal = grid.cells().cell_count() - 1;

        let informed = search(&grid, 0, goal, astar()).unwrap();
        let uninformed = search(&grid, 0, goal, dijkstra()).unwrap();

        match (informed.cost(), uninformed.cost()) {
            (Some(a), Some(b)) => assert!(
                (a - b).abs() < 1e-12,
                "seed {seed}: A* {a} against Dijkstra {b}"
            ),
            (None, None) => {}
            (a, b) => {
                panic!("seed {seed}: one search found a path and the other did not: {a:?} {b:?}")
            }
        }
    }
}

#[test]
fn astar_expands_no_more_than_dijkstra() {
    // The heuristic has to help or it is not worth having. Equality is
    // allowed, since a heuristic of zero is admissible.
    let grid = ManhattanGrid::new_free(&[20, 20], 1.0).unwrap();
    let goal = grid.cells().cell_count() - 1;

    let informed = search(&grid, 0, goal, astar()).unwrap();
    let uninformed = search(&grid, 0, goal, dijkstra()).unwrap();
    assert!(
        informed.expanded() <= uninformed.expanded(),
        "A* expanded {} against Dijkstra's {}",
        informed.expanded(),
        uninformed.expanded()
    );
}

#[test]
fn a_path_starts_at_the_start_and_ends_at_the_goal() {
    let grid = ManhattanGrid::new_free(&[6, 6], 1.0).unwrap();
    let goal = grid.cells().cell_count() - 1;
    let path = search(&grid, 0, goal, astar()).unwrap();
    let states = path.path().expect("an open grid is traversable");

    assert_eq!(states.first(), Some(&0));
    assert_eq!(states.last(), Some(&goal));
}

#[test]
fn consecutive_path_states_are_neighbors() {
    // FR-INV-02. A path whose states are not adjacent is not a path, and
    // the failure mode is a reconstruction bug rather than a search one.
    let grid = scattered_grid(3, 10, 0.15);
    let goal = grid.cells().cell_count() - 1;
    let outcome = search(&grid, 0, goal, astar()).unwrap();

    if let Some(states) = outcome.path() {
        for pair in states.windows(2) {
            let [from, to] = pair else { continue };
            assert!(
                grid.neighbors(*from).contains(to),
                "{from} and {to} are not adjacent"
            );
        }
    }
}

#[test]
fn the_reported_cost_matches_the_path_it_describes() {
    let grid = scattered_grid(11, 10, 0.15);
    let goal = grid.cells().cell_count() - 1;
    let outcome = search(&grid, 0, goal, astar()).unwrap();

    if let (Some(states), Some(cost)) = (outcome.path(), outcome.cost()) {
        let walked: f64 = states
            .windows(2)
            .filter_map(|pair| match pair {
                [from, to] => grid.distance(*from, *to).ok(),
                _ => None,
            })
            .sum();
        assert!((walked - cost).abs() < 1e-9, "{walked} against {cost}");
    }
}

#[test]
fn a_path_never_crosses_a_blocked_cell() {
    // FR-INV-01, at the grid level: re-checked against the same map the
    // search was given.
    for seed in 0..10_u64 {
        let grid = scattered_grid(seed, 12, 0.25);
        let goal = grid.cells().cell_count() - 1;
        let outcome = search(&grid, 0, goal, astar()).unwrap();

        for state in outcome.path().unwrap_or_default() {
            assert!(
                !grid.cells().blocks(*state).unwrap(),
                "seed {seed}: the path crosses blocked cell {state}"
            );
        }
    }
}

#[test]
fn an_unreachable_goal_is_reported_as_such() {
    // FR-INV-08. Distinct from running out of budget, because retrying
    // will not help.
    let mut grid = ManhattanGrid::new_free(&[5, 5], 1.0).unwrap();
    let goal = grid.cells().linear_index(&[4, 4]).unwrap();
    // Wall the goal off completely.
    for index in [[3, 4], [4, 3]] {
        let linear = grid.cells().linear_index(&index).unwrap();
        grid.cells_mut().set_cell(linear, Cell::Occupied).unwrap();
    }

    let outcome = search(&grid, 0, goal, astar()).unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::Unreachable));
    assert!(!PlanFailure::Unreachable.is_retryable());
}

#[test]
fn an_exhausted_budget_is_reported_as_retryable() {
    // FR-SAFE-02. The whole point of the distinction: a caller can decide
    // to try again with more budget rather than concluding there is no
    // path.
    let grid = ManhattanGrid::new_free(&[40, 40], 1.0).unwrap();
    let goal = grid.cells().cell_count() - 1;
    let outcome = search(
        &grid,
        0,
        goal,
        SearchOptions {
            max_expansions: 10,
            use_heuristic: true,
        },
    )
    .unwrap();

    assert_eq!(outcome.failure(), Some(PlanFailure::BudgetExhausted));
    assert!(PlanFailure::BudgetExhausted.is_retryable());
    assert!(outcome.expanded() <= 10);
}

#[test]
fn a_larger_budget_turns_exhaustion_into_a_path() {
    let grid = ManhattanGrid::new_free(&[40, 40], 1.0).unwrap();
    let goal = grid.cells().cell_count() - 1;
    let small = search(
        &grid,
        0,
        goal,
        SearchOptions {
            max_expansions: 10,
            use_heuristic: true,
        },
    )
    .unwrap();
    let large = search(&grid, 0, goal, astar()).unwrap();

    assert!(small.path().is_none());
    assert!(
        large.path().is_some(),
        "the same query succeeds with budget"
    );
}

#[test]
fn the_start_is_its_own_path() {
    let grid = ManhattanGrid::new_free(&[4, 4], 1.0).unwrap();
    let outcome = search(&grid, 0, 0, astar()).unwrap();
    assert_eq!(outcome.path(), Some([0].as_slice()));
    assert_eq!(outcome.cost(), Some(0.0));
}

#[test]
fn removing_an_obstacle_never_increases_the_optimal_cost() {
    // FR-INV-04, a metamorphic relation: it holds whichever path the
    // planner happened to choose.
    let mut grid = scattered_grid(5, 10, 0.2);
    let goal = grid.cells().cell_count() - 1;
    let before = search(&grid, 0, goal, astar()).unwrap().cost();

    for linear in 0..grid.cells().cell_count() {
        if grid.cells().blocks(linear).unwrap() {
            grid.cells_mut().set_cell(linear, Cell::Free).unwrap();
            break;
        }
    }
    let after = search(&grid, 0, goal, astar()).unwrap().cost();

    match (before, after) {
        (Some(before), Some(after)) => assert!(after <= before + 1e-12, "{after} against {before}"),
        (None, _) => {}
        (Some(_), None) => panic!("removing an obstacle made the goal unreachable"),
    }
}

#[test]
fn adding_an_obstacle_never_decreases_the_optimal_cost() {
    // FR-INV-04, the other direction.
    let mut grid = ManhattanGrid::new_free(&[10, 10], 1.0).unwrap();
    let goal = grid.cells().cell_count() - 1;
    let before = search(&grid, 0, goal, astar()).unwrap().cost().unwrap();

    let blocked = grid.cells().linear_index(&[5, 5]).unwrap();
    grid.cells_mut().set_cell(blocked, Cell::Occupied).unwrap();
    let after = search(&grid, 0, goal, astar()).unwrap().cost().unwrap();

    assert!(after >= before - 1e-12, "{after} against {before}");
}

#[test]
fn the_same_query_gives_the_same_path_every_time() {
    // FR-RNG-01 at the search level. Ties in the open set break on
    // insertion order, which is deterministic.
    let grid = scattered_grid(9, 12, 0.2);
    let goal = grid.cells().cell_count() - 1;
    let first = search(&grid, 0, goal, astar()).unwrap();
    for _ in 0..8 {
        let again = search(&grid, 0, goal, astar()).unwrap();
        assert_eq!(first.path(), again.path());
        assert_eq!(first.expanded(), again.expanded());
    }
}

#[test]
fn a_diagonal_grid_is_cheaper_than_an_axis_aligned_one() {
    let manhattan = ManhattanGrid::new_free(&[9, 9], 1.0).unwrap();
    let euclidean = EuclideanGrid::new_free(&[9, 9], 1.0).unwrap();
    let goal = manhattan.cells().cell_count() - 1;

    let four = search(&manhattan, 0, goal, astar())
        .unwrap()
        .cost()
        .unwrap();
    let eight = search(&euclidean, 0, goal, astar())
        .unwrap()
        .cost()
        .unwrap();
    assert!(
        eight < four,
        "eight-connected {eight}, four-connected {four}"
    );
}

#[test]
fn a_euclidean_grid_also_agrees_with_its_uninformed_search() {
    for seed in 0..10_u64 {
        let mut grid = EuclideanGrid::new_free(&[10, 10], 1.0).unwrap();
        let mut generator = Pcg64::seed_from_u64(seed);
        let count = grid.cells().cell_count();
        for linear in 1..count - 1 {
            if generator.next_f64() < 0.2 {
                grid.cells_mut().set_cell(linear, Cell::Occupied).unwrap();
            }
        }

        let goal = count - 1;
        let informed = search(&grid, 0, goal, astar()).unwrap();
        let uninformed = search(&grid, 0, goal, dijkstra()).unwrap();
        assert_eq!(
            informed.cost().map(|cost| (cost * 1e9).round()),
            uninformed.cost().map(|cost| (cost * 1e9).round()),
            "seed {seed}"
        );
    }
}
