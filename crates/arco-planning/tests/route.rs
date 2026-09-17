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

//! Routing continuous positions over a road network.
//!
//! Mirrors `tests/planning/discrete/test_route.py` and
//! `test_route_road_graph.py`, which stay as they are and run against the
//! bindings in phase 9. The cases here are the same ones, expressed
//! against the typed failures rather than against `None`.

#![expect(clippy::unwrap_used, reason = "test fixtures and assertions")]

use arco_core::protocols::DiscreteMap;
use arco_mapping::graph::{CartesianGraph, RoadGraph};
use arco_planning::discrete::{RouteRouter, SearchOptions, search};
use arco_planning::failure::PlanFailure;

/// A three by three grid of nodes, ten meters apart.
///
/// ```text
/// 0 - 1 - 2
/// |   |   |
/// 3 - 4 - 5
/// |   |   |
/// 6 - 7 - 8
/// ```
fn grid_graph() -> CartesianGraph {
    let mut graph = CartesianGraph::new();
    for row in 0..3_i64 {
        for column in 0..3_i64 {
            let node = row * 3 + column;
            #[expect(clippy::cast_precision_loss, reason = "grid indices are 0 to 2")]
            graph
                .add_node(node, &[column as f64 * 10.0, row as f64 * 10.0])
                .unwrap();
        }
    }
    for row in 0..3_i64 {
        for column in 0..2_i64 {
            graph
                .add_edge(row * 3 + column, row * 3 + column + 1, None)
                .unwrap();
        }
    }
    for row in 0..2_i64 {
        for column in 0..3_i64 {
            graph
                .add_edge(row * 3 + column, (row + 1) * 3 + column, None)
                .unwrap();
        }
    }
    graph
}

#[test]
fn a_query_on_the_nodes_routes_between_them() {
    let router = RouteRouter::new(grid_graph(), None);
    let outcome = router.plan(&[0.0, 0.0], &[20.0, 20.0]).unwrap();

    let route = outcome.result().expect("the grid is connected");
    assert_eq!(route.start.node, 0);
    assert_eq!(route.goal.node, 8);
    assert_eq!(route.path.first(), Some(&0));
    assert_eq!(route.path.last(), Some(&8));
    // Exactly zero rather than near it: the query sits on the node, and
    // the projection is a copy of its position rather than a computation.
    assert!(route.start.distance.abs() < f64::EPSILON);
    assert!(route.goal.distance.abs() < f64::EPSILON);
}

#[test]
fn a_query_off_the_nodes_reports_how_far_off_it_was() {
    // The projection distance is the part of the journey the network does
    // not cover, so a caller needs it rather than the router hiding it.
    let router = RouteRouter::new(grid_graph(), None);
    let outcome = router.plan(&[1.0, 2.0], &[19.0, 18.0]).unwrap();

    let route = outcome.result().unwrap();
    let expected = 1.0_f64.hypot(2.0);
    assert!((route.start.distance - expected).abs() < 1e-9);
    assert!((route.goal.distance - expected).abs() < 1e-9);
    assert_eq!(route.start.point, vec![0.0, 0.0]);
    assert_eq!(route.goal.point, vec![20.0, 20.0]);
}

#[test]
fn a_query_that_snaps_to_one_node_routes_to_itself() {
    let router = RouteRouter::new(grid_graph(), None);
    let outcome = router.plan(&[10.0, 10.0], &[10.1, 10.1]).unwrap();

    let route = outcome.result().unwrap();
    assert_eq!(route.start.node, 4);
    assert_eq!(route.goal.node, 4);
    assert_eq!(route.path, vec![4]);
    assert!(
        route.cost.abs() < f64::EPSILON,
        "a route to itself costs {}",
        route.cost
    );
}

#[test]
fn a_straight_run_takes_the_straight_path() {
    let router = RouteRouter::new(grid_graph(), None);
    let route = router
        .plan(&[0.0, 0.0], &[20.0, 0.0])
        .unwrap()
        .result()
        .cloned()
        .unwrap();
    assert_eq!(route.path, vec![0, 1, 2]);
    assert!((route.cost - 20.0).abs() < 1e-9);
}

#[test]
fn a_position_outside_the_activation_radius_is_named_as_such() {
    // FR-INV-08. Too far from any road is the same kind of answer as off
    // the edge of a grid: no budget makes it reachable, and the caller has
    // to move rather than retry.
    let router = RouteRouter::new(grid_graph(), Some(5.0));

    let from_afar = router.plan(&[50.0, 50.0], &[10.0, 10.0]).unwrap();
    assert_eq!(from_afar.failure(), Some(PlanFailure::StartOutsideMap));
    assert_eq!(from_afar.path(), None);

    let to_afar = router.plan(&[10.0, 10.0], &[50.0, 50.0]).unwrap();
    assert_eq!(to_afar.failure(), Some(PlanFailure::GoalOutsideMap));
}

#[test]
fn both_ends_inside_the_radius_route_normally() {
    let router = RouteRouter::new(grid_graph(), Some(5.0));
    let route = router
        .plan(&[2.0, 2.0], &[18.0, 18.0])
        .unwrap()
        .result()
        .cloned()
        .unwrap();
    assert_eq!(route.start.node, 0);
    assert_eq!(route.goal.node, 8);
}

#[test]
fn no_radius_accepts_a_position_anywhere() {
    // Worth pinning because it is a trap rather than a feature: without a
    // radius a query a hundred meters off the network still routes, from
    // whatever node happens to be least far away.
    let router = RouteRouter::new(grid_graph(), None);
    let route = router
        .plan(&[-100.0, -100.0], &[100.0, 100.0])
        .unwrap()
        .result()
        .cloned()
        .unwrap();
    assert_eq!(route.start.node, 0);
    assert_eq!(route.goal.node, 8);
    assert!(route.start.distance > 100.0);
}

#[test]
fn a_disconnected_network_is_reported_as_unreachable() {
    let mut graph = CartesianGraph::new();
    graph.add_node(0, &[0.0, 0.0]).unwrap();
    graph.add_node(1, &[10.0, 0.0]).unwrap();
    graph.add_edge(0, 1, None).unwrap();
    graph.add_node(2, &[100.0, 0.0]).unwrap();
    graph.add_node(3, &[110.0, 0.0]).unwrap();
    graph.add_edge(2, 3, None).unwrap();

    let router = RouteRouter::new(graph, Some(20.0));
    let outcome = router.plan(&[5.0, 0.0], &[105.0, 0.0]).unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::Unreachable));
    assert!(!PlanFailure::Unreachable.is_retryable());
}

#[test]
fn an_empty_network_routes_nowhere() {
    let router = RouteRouter::new(CartesianGraph::new(), None);
    let outcome = router.plan(&[0.0, 0.0], &[10.0, 10.0]).unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::StartOutsideMap));
    assert_eq!(outcome.expanded(), 0);
}

#[test]
fn a_single_node_network_routes_to_that_node() {
    let mut graph = CartesianGraph::new();
    graph.add_node(0, &[5.0, 5.0]).unwrap();

    let router = RouteRouter::new(graph, None);
    let route = router
        .plan(&[4.0, 4.0], &[6.0, 6.0])
        .unwrap()
        .result()
        .cloned()
        .unwrap();
    assert_eq!(route.path, vec![0]);
    assert_eq!(route.start.node, 0);
    assert_eq!(route.goal.node, 0);
}

#[test]
fn a_shortcut_priced_below_its_length_is_still_taken() {
    // The one case where the straight-line heuristic is not admissible:
    // an edge weighted below the distance it spans. A* still finds the
    // shortcut here because it leaves the start directly, and the
    // admissibility precondition is documented on the heuristic rather
    // than enforced, because a road cheaper than its length is a
    // reasonable thing to model.
    let mut graph = CartesianGraph::new();
    graph.add_node(0, &[0.0, 0.0]).unwrap();
    graph.add_node(1, &[1.0, 1.0]).unwrap();
    graph.add_node(2, &[1.0, -1.0]).unwrap();
    graph.add_node(3, &[2.0, 0.0]).unwrap();
    graph.add_edge(0, 1, Some(2.0)).unwrap();
    graph.add_edge(1, 3, Some(2.0)).unwrap();
    graph.add_edge(0, 2, Some(2.0)).unwrap();
    graph.add_edge(2, 3, Some(2.0)).unwrap();
    graph.add_edge(0, 3, Some(1.0)).unwrap();

    let router = RouteRouter::new(graph, None);
    let route = router
        .plan(&[0.0, 0.0], &[2.0, 0.0])
        .unwrap()
        .result()
        .cloned()
        .unwrap();
    assert_eq!(route.path, vec![0, 3]);
    assert!((route.cost - 1.0).abs() < 1e-9);
}

#[test]
fn a_road_graph_routes_without_being_unwrapped() {
    // Python got this from inheritance, the port from `AsRef`, and the
    // call site reads the same either way. Deviation A-03.
    let mut graph = RoadGraph::new();
    graph.positions_mut().add_node(0, &[0.0, 0.0]).unwrap();
    graph.positions_mut().add_node(1, &[10.0, 0.0]).unwrap();
    graph.positions_mut().add_node(2, &[20.0, 0.0]).unwrap();
    graph.add_edge(0, 1, None, &[vec![5.0, 1.0]]).unwrap();
    graph.add_edge(1, 2, None, &[vec![15.0, -1.0]]).unwrap();

    let router = RouteRouter::new(graph, Some(5.0));
    let route = router
        .plan(&[1.0, 1.0], &[19.0, 1.0])
        .unwrap()
        .result()
        .cloned()
        .unwrap();
    assert_eq!(route.path, vec![0, 1, 2]);
    assert_eq!(route.start.node, 0);
    assert_eq!(route.goal.node, 2);
}

#[test]
fn the_same_query_routes_the_same_way_every_time() {
    let router = RouteRouter::new(grid_graph(), Some(20.0));
    let first = router.plan(&[12.0, 13.0], &[18.0, 17.0]).unwrap();
    for _ in 0..4 {
        assert_eq!(router.plan(&[12.0, 13.0], &[18.0, 17.0]).unwrap(), first);
    }
    assert!(first.result().is_some());
}

#[test]
fn an_exhausted_budget_is_reported_as_retryable() {
    // FR-SAFE-02 reaches the router through the search it delegates to.
    let router = RouteRouter::new(grid_graph(), None).with_options(SearchOptions {
        max_expansions: 1,
        use_heuristic: true,
        prefer_straight: true,
    });
    let outcome = router.plan(&[0.0, 0.0], &[20.0, 20.0]).unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::BudgetExhausted));
    assert!(PlanFailure::BudgetExhausted.is_retryable());
}

#[test]
fn a_malformed_query_is_rejected() {
    let router = RouteRouter::new(grid_graph(), None);
    assert!(router.plan(&[f64::NAN, 0.0], &[10.0, 10.0]).is_err());
    assert!(router.plan(&[0.0, 0.0], &[f64::INFINITY, 1.0]).is_err());
    assert!(router.plan(&[0.0], &[10.0, 10.0]).is_err());
}

#[test]
fn routing_agrees_with_an_uninformed_search_over_the_same_graph() {
    // FR-INV-06 on a graph rather than a grid. The default edge weight is
    // the straight-line distance, so the heuristic is admissible and the
    // two searches have to agree on cost exactly.
    let graph = grid_graph();
    for (start, goal) in [(0_i64, 8_i64), (2, 6), (0, 5), (7, 1)] {
        let informed = search(&graph, start, goal, SearchOptions::default()).unwrap();
        let uninformed = search(
            &graph,
            start,
            goal,
            SearchOptions {
                max_expansions: 1_000_000,
                use_heuristic: false,
                prefer_straight: true,
            },
        )
        .unwrap();
        assert_eq!(
            informed.cost(),
            uninformed.cost(),
            "{start} to {goal} disagreed"
        );
    }
}

#[test]
fn the_graph_reports_which_nodes_it_holds() {
    let graph = grid_graph();
    assert!(DiscreteMap::contains(&graph, 0));
    assert!(DiscreteMap::contains(&graph, 8));
    assert!(!DiscreteMap::contains(&graph, 9));

    let outcome = search(&graph, 0, 99, SearchOptions::default()).unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::GoalOutsideMap));
}
