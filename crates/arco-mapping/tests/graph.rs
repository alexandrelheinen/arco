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

//! The graph layers behave as the Python hierarchy did, minus its holes.

use arco_core::Error;
use arco_mapping::graph::{CartesianGraph, RoadGraph, WeightedGraph};

#[test]
fn an_edge_is_undirected_and_carries_its_weight_both_ways() {
    let mut graph = WeightedGraph::new();
    graph.add_edge(1, 2, 3.5).unwrap();
    assert!((graph.distance(1, 2).unwrap() - 3.5).abs() < 1e-12);
    assert!((graph.distance(2, 1).unwrap() - 3.5).abs() < 1e-12);
    assert_eq!(graph.neighbors(1), vec![2]);
    assert_eq!(graph.neighbors(2), vec![1]);
}

#[test]
fn a_negative_or_non_finite_edge_weight_is_rejected() {
    // FR-INV-06 rests on non-negative finite costs. A negative edge breaks
    // shortest-path optimality silently, which is the worst failure mode
    // available, so it is refused where it enters.
    let mut graph = WeightedGraph::new();
    assert!(matches!(
        graph.add_edge(1, 2, -1.0),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        graph.add_edge(1, 2, f64::NAN),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        graph.add_edge(1, 2, f64::INFINITY),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn re_adding_an_edge_replaces_its_weight_rather_than_duplicating_it() {
    let mut graph = WeightedGraph::new();
    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(1, 2, 9.0).unwrap();
    assert_eq!(graph.neighbors(1), vec![2]);
    assert!((graph.distance(1, 2).unwrap() - 9.0).abs() < 1e-12);
    assert_eq!(graph.edges().len(), 1);
}

#[test]
fn an_absent_edge_is_reported_as_an_unknown_identifier() {
    let graph = WeightedGraph::new();
    assert!(matches!(
        graph.distance(1, 2),
        Err(Error::UnknownIdentifier { .. })
    ));
}

#[test]
fn node_order_is_deterministic() {
    // Ties in a nearest-node query break the same way on every run only
    // because the storage is ordered.
    let mut first = WeightedGraph::new();
    let mut second = WeightedGraph::new();
    for node in [5, 1, 9, 3] {
        first.add_node(node);
    }
    for node in [9, 3, 5, 1] {
        second.add_node(node);
    }
    assert_eq!(first.nodes(), second.nodes());
    assert_eq!(first.nodes(), vec![1, 3, 5, 9]);
}

#[test]
fn a_cartesian_graph_takes_its_dimension_from_its_first_node() {
    let mut graph = CartesianGraph::new();
    assert_eq!(graph.dimension(), None);
    graph.add_node(1, &[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(graph.dimension(), Some(3));
    assert!(matches!(
        graph.add_node(2, &[1.0, 1.0]),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn a_node_position_must_be_finite_and_non_empty() {
    let mut graph = CartesianGraph::new();
    assert!(matches!(graph.add_node(1, &[]), Err(Error::TooFew { .. })));
    assert!(matches!(
        graph.add_node(1, &[0.0, f64::NAN]),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn an_edge_weight_defaults_to_the_straight_line_distance() {
    let mut graph = CartesianGraph::new();
    graph.add_node(1, &[0.0, 0.0]).unwrap();
    graph.add_node(2, &[3.0, 4.0]).unwrap();
    graph.add_edge(1, 2, None).unwrap();
    assert!((graph.distance(1, 2).unwrap() - 5.0).abs() < 1e-12);
}

#[test]
fn the_nearest_node_respects_a_radius() {
    let mut graph = CartesianGraph::new();
    graph.add_node(1, &[0.0, 0.0]).unwrap();
    graph.add_node(2, &[10.0, 0.0]).unwrap();

    assert_eq!(graph.find_nearest_node(&[9.0, 0.0], None).unwrap(), Some(2));
    assert_eq!(
        graph.find_nearest_node(&[9.0, 0.0], Some(0.5)).unwrap(),
        None
    );
    assert_eq!(
        graph.find_nearest_node(&[9.0, 0.0], Some(2.0)).unwrap(),
        Some(2)
    );
}

#[test]
fn the_nearest_node_of_an_empty_graph_is_nothing() {
    let graph = CartesianGraph::new();
    assert_eq!(graph.find_nearest_node(&[0.0, 0.0], None).unwrap(), None);
}

#[test]
fn a_projection_lands_on_the_segment() {
    let mut graph = CartesianGraph::new();
    graph.add_node(1, &[0.0, 0.0]).unwrap();
    graph.add_node(2, &[10.0, 0.0]).unwrap();
    graph.add_edge(1, 2, None).unwrap();

    let projection = graph.project_to_nearest_edge(&[4.0, 3.0]).unwrap().unwrap();
    assert!((projection.point[0] - 4.0).abs() < 1e-12, "{projection:?}");
    assert!(projection.point[1].abs() < 1e-12, "{projection:?}");
    assert!((projection.distance - 3.0).abs() < 1e-12, "{projection:?}");
}

#[test]
fn a_projection_past_an_endpoint_clamps_to_it() {
    let mut graph = CartesianGraph::new();
    graph.add_node(1, &[0.0, 0.0]).unwrap();
    graph.add_node(2, &[10.0, 0.0]).unwrap();
    graph.add_edge(1, 2, None).unwrap();

    let projection = graph
        .project_to_nearest_edge(&[-5.0, 0.0])
        .unwrap()
        .unwrap();
    assert!(projection.point[0].abs() < 1e-12, "{projection:?}");
}

#[test]
fn a_nearly_degenerate_edge_does_not_divide_by_its_length() {
    // The Python guard compared the squared length to exactly zero, so an
    // edge of length 1e-30 passed it and divided anyway. The clamp then
    // hid the result. Here the threshold is relative to the endpoints.
    let mut graph = CartesianGraph::new();
    graph.add_node(1, &[1.0, 1.0]).unwrap();
    graph.add_node(2, &[1.0 + 1e-30, 1.0]).unwrap();
    graph.add_edge(1, 2, Some(0.0)).unwrap();

    let projection = graph.project_to_nearest_edge(&[5.0, 5.0]).unwrap().unwrap();
    assert!(projection.point.iter().all(|value| value.is_finite()));
    assert!(projection.distance.is_finite());
}

#[test]
fn road_geometry_follows_the_direction_asked_for() {
    let mut graph = RoadGraph::new();
    graph.positions_mut().add_node(1, &[0.0, 0.0]).unwrap();
    graph.positions_mut().add_node(2, &[10.0, 0.0]).unwrap();
    graph
        .add_edge(1, 2, None, &[vec![3.0, 1.0], vec![7.0, 1.0]])
        .unwrap();

    let forward = graph.edge_geometry(1, 2).unwrap();
    let backward = graph.edge_geometry(2, 1).unwrap();
    assert_eq!(forward, vec![vec![3.0, 1.0], vec![7.0, 1.0]]);
    assert_eq!(backward, vec![vec![7.0, 1.0], vec![3.0, 1.0]]);
}

#[test]
fn the_full_geometry_includes_both_endpoints() {
    let mut graph = RoadGraph::new();
    graph.positions_mut().add_node(1, &[0.0, 0.0]).unwrap();
    graph.positions_mut().add_node(2, &[10.0, 0.0]).unwrap();
    graph.add_edge(1, 2, None, &[vec![5.0, 2.0]]).unwrap();

    let full = graph.full_edge_geometry(1, 2).unwrap();
    assert_eq!(full, vec![vec![0.0, 0.0], vec![5.0, 2.0], vec![10.0, 0.0]]);
}

#[test]
fn a_straight_road_has_no_intermediate_geometry() {
    let mut graph = RoadGraph::new();
    graph.positions_mut().add_node(1, &[0.0, 0.0]).unwrap();
    graph.positions_mut().add_node(2, &[1.0, 0.0]).unwrap();
    graph.add_edge(1, 2, None, &[]).unwrap();
    assert!(graph.edge_geometry(1, 2).unwrap().is_empty());
    assert_eq!(graph.full_edge_geometry(1, 2).unwrap().len(), 2);
}

#[test]
fn a_waypoint_of_the_wrong_dimension_is_rejected() {
    let mut graph = RoadGraph::new();
    graph.positions_mut().add_node(1, &[0.0, 0.0]).unwrap();
    graph.positions_mut().add_node(2, &[1.0, 0.0]).unwrap();
    assert!(matches!(
        graph.add_edge(1, 2, None, &[vec![0.5, 0.0, 0.0]]),
        Err(Error::DimensionMismatch { .. })
    ));
}
