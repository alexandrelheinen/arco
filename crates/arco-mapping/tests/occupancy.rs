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

//! The k-d tree answers what a brute-force scan answers.

use arco_core::Error;
use arco_core::protocols::{Occupancy, SegmentChecker};
use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;

/// The answer a scan gives, which the tree has to match exactly.
fn brute_force_nearest(points: &[Vec<f64>], query: &[f64]) -> f64 {
    points
        .iter()
        .map(|point| {
            point
                .iter()
                .zip(query)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn random_points(seed: u64, count: usize, dimension: usize) -> Vec<Vec<f64>> {
    let mut generator = Pcg64::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| generator.next_f64() * 20.0 - 10.0)
                .collect()
        })
        .collect()
}

#[test]
fn the_tree_agrees_with_a_brute_force_scan() {
    // The property that makes the tree worth having rather than trusting.
    for dimension in 1..5_usize {
        let points = random_points(7, 200, dimension);
        let occupancy = KdTreeOccupancy::new(&points, 0.0).unwrap();

        let mut generator = Pcg64::seed_from_u64(99);
        for _ in 0..200 {
            let query: Vec<f64> = (0..dimension)
                .map(|_| generator.next_f64() * 24.0 - 12.0)
                .collect();
            let produced = occupancy.nearest_obstacle(&query).unwrap().distance;
            let expected = brute_force_nearest(&points, &query);
            assert!(
                (produced - expected).abs() < 1e-9,
                "dimension {dimension}: {produced} against {expected}"
            );
        }
    }
}

#[test]
fn clearance_turns_a_point_into_a_region() {
    let occupancy = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 1.0).unwrap();
    assert!(occupancy.is_occupied(&[0.5, 0.0]).unwrap());
    assert!(occupancy.is_occupied(&[1.0, 0.0]).unwrap());
    assert!(!occupancy.is_occupied(&[1.5, 0.0]).unwrap());
}

#[test]
fn a_query_of_the_wrong_dimension_is_rejected() {
    // FR-CORE-03.
    let occupancy = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 0.5).unwrap();
    assert!(matches!(
        occupancy.is_occupied(&[0.0, 0.0, 0.0]),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn a_non_finite_query_is_rejected() {
    // FR-SAFE-07.
    let occupancy = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 0.5).unwrap();
    assert!(matches!(
        occupancy.is_occupied(&[f64::NAN, 0.0]),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn a_degenerate_obstacle_set_is_rejected() {
    assert!(matches!(
        KdTreeOccupancy::new(&[], 1.0),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        KdTreeOccupancy::new(&[vec![]], 1.0),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        KdTreeOccupancy::new(&[vec![0.0], vec![0.0, 0.0]], 1.0),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        KdTreeOccupancy::new(&[vec![f64::NAN]], 1.0),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        KdTreeOccupancy::new(&[vec![0.0]], -1.0),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn duplicate_points_build_the_same_tree_every_time() {
    // Ties in the split are broken by a total order, so an obstacle set
    // with repeated coordinates does not depend on sort stability.
    let points = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 2.0]];
    let first = KdTreeOccupancy::new(&points, 0.1).unwrap();
    let second = KdTreeOccupancy::new(&points, 0.1).unwrap();
    assert_eq!(first.content_hash(), second.content_hash());
    for query in [[0.0, 0.0], [1.0, 1.5], [5.0, 5.0]] {
        assert!(
            (first.nearest_obstacle(&query).unwrap().distance
                - second.nearest_obstacle(&query).unwrap().distance)
                .abs()
                < 1e-15
        );
    }
}

#[test]
fn the_content_hash_ignores_the_order_of_the_points() {
    // FR-INV-12: the same field is the same map however it was listed.
    let forward = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    let reversed: Vec<Vec<f64>> = forward.iter().rev().cloned().collect();
    let first = KdTreeOccupancy::new(&forward, 0.5).unwrap();
    let second = KdTreeOccupancy::new(&reversed, 0.5).unwrap();
    assert_eq!(first.content_hash(), second.content_hash());
}

#[test]
fn the_content_hash_separates_different_fields() {
    let base = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 0.5).unwrap();
    let moved = KdTreeOccupancy::new(&[vec![0.0, 0.1]], 0.5).unwrap();
    let wider = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 0.6).unwrap();
    assert_ne!(base.content_hash(), moved.content_hash());
    assert_ne!(base.content_hash(), wider.content_hash());
}

#[test]
fn a_segment_through_an_obstacle_is_not_free() {
    let occupancy = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 1.0).unwrap();
    assert!(
        !occupancy
            .is_segment_free(&[-5.0, 0.0], &[5.0, 0.0])
            .unwrap()
    );
    assert!(
        occupancy
            .is_segment_free(&[-5.0, 5.0], &[5.0, 5.0])
            .unwrap()
    );
}

#[test]
fn sampling_can_miss_a_thin_obstacle() {
    // Stated as a test because FR-INV-01 depends on knowing it: the
    // segment check is a sampled approximation, which is why a returned
    // path is re-checked at a stated resolution rather than trusted.
    // Placed off the coarse sample points, which land at -5, 0 and 5.
    let occupancy = KdTreeOccupancy::new(&[vec![1.0, 0.0]], 0.01).unwrap();
    let coarse = occupancy
        .is_segment_free_with(&[-5.0, 0.0], &[5.0, 0.0], 3)
        .unwrap();
    let fine = occupancy
        .is_segment_free_with(&[-5.0, 0.0], &[5.0, 0.0], 4001)
        .unwrap();
    assert!(coarse, "the coarse check misses the obstacle");
    assert!(!fine, "the fine check finds it");
}

#[test]
fn distances_can_be_queried_in_bulk() {
    let occupancy = KdTreeOccupancy::new(&[vec![0.0, 0.0]], 0.0).unwrap();
    let distances = occupancy
        .query_distances(&[vec![3.0, 4.0], vec![0.0, 0.0]])
        .unwrap();
    assert!((distances[0] - 5.0).abs() < 1e-12);
    assert!(distances[1].abs() < 1e-12);
}

#[test]
fn the_points_are_returned_in_the_order_given() {
    let points = vec![vec![2.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]];
    let occupancy = KdTreeOccupancy::new(&points, 0.1).unwrap();
    assert_eq!(occupancy.points(), points.as_slice());
    assert!((occupancy.clearance() - 0.1).abs() < 1e-12);
}
