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

//! FR-INV-03: what shortening a path is allowed to do to it.
//!
//! A metamorphic requirement, because the shortest valid subsequence of a
//! path is not something a test can look up. What it can check is the
//! relation between input and output: never longer, never invalid where
//! the input was valid, same endpoints.

#![expect(clippy::unwrap_used, reason = "test fixtures and assertions")]

use arco_core::protocols::Pruner;
use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    CostPolicy, RrtPlanner, RrtSettings, SamplerPolicy, SegmentPolicy, SteererPolicy,
    TrajectoryPruner,
};

const STEP_SIZE: f64 = 2.0;

fn obstacle_field(seed: u64, count: usize) -> KdTreeOccupancy {
    let mut generator = Pcg64::seed_from_u64(seed);
    let points: Vec<Vec<f64>> = (0..count)
        .map(|_| {
            vec![
                generator.next_f64().mul_add(42.0, 4.0),
                generator.next_f64().mul_add(42.0, 4.0),
            ]
        })
        .collect();
    KdTreeOccupancy::new(&points, 1.2).unwrap()
}

/// A raw RRT* path over `field`, which is what a pruner is handed.
fn raw_path(field: &KdTreeOccupancy, seed: u64) -> Option<Vec<Vec<f64>>> {
    let planner = RrtPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: vec![(0.0, 50.0), (0.0, 50.0)],
        },
        SteererPolicy::Straight {
            step_size: vec![STEP_SIZE, STEP_SIZE],
        },
        SegmentPolicy::Exact {
            occupancy: field.clone(),
        },
        CostPolicy::Scaled {
            step_size: vec![STEP_SIZE, STEP_SIZE],
        },
        RrtSettings {
            max_samples: 3000,
            goal_tolerance: 1.5,
            ..RrtSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(seed);
    planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap()
        .path()
        .map(<[Vec<f64>]>::to_vec)
}

fn length(path: &[Vec<f64>]) -> f64 {
    path.windows(2)
        .filter_map(|pair| match pair {
            [from, to] => Some((to[0] - from[0]).hypot(to[1] - from[1])),
            _ => None,
        })
        .sum()
}

fn pruner(field: KdTreeOccupancy) -> TrajectoryPruner<KdTreeOccupancy> {
    TrajectoryPruner::new(SegmentPolicy::Exact { occupancy: field })
}

#[test]
fn pruning_never_lengthens_a_path_and_never_invalidates_it() {
    // FR-INV-03, over eight fields so the relation is tested rather than
    // one lucky path.
    for seed in 0..8_u64 {
        let field = obstacle_field(seed, 150);
        let Some(raw) = raw_path(&field, seed) else {
            continue;
        };
        let checker = SegmentPolicy::Exact {
            occupancy: field.clone(),
        };
        let pruned = pruner(field).prune(&raw).unwrap();

        assert!(
            pruned.len() <= raw.len(),
            "seed {seed}: pruning grew the path from {} to {}",
            raw.len(),
            pruned.len()
        );
        assert!(
            length(&pruned) <= length(&raw) + 1e-9,
            "seed {seed}: pruning lengthened the path"
        );
        assert_eq!(pruned.first(), raw.first(), "seed {seed}: start moved");
        assert_eq!(pruned.last(), raw.last(), "seed {seed}: goal moved");
        for pair in pruned.windows(2) {
            let [from, to] = pair else { continue };
            assert!(
                checker.is_segment_free(from, to).unwrap(),
                "seed {seed}: pruning produced an edge through an obstacle"
            );
        }
    }
}

#[test]
fn a_straight_corridor_collapses_to_its_endpoints() {
    // The fewest waypoints that connect, not merely fewer: a clear line
    // needs two, and a greedy scan would also find two here, which is why
    // the next test exists.
    let field = KdTreeOccupancy::new(&[vec![200.0, 200.0]], 0.5).unwrap();
    let raw: Vec<Vec<f64>> = (0..=20).map(|step| vec![f64::from(step), 0.0]).collect();

    let pruned = pruner(field).prune(&raw).unwrap();
    assert_eq!(pruned, vec![vec![0.0, 0.0], vec![20.0, 0.0]]);
}

#[test]
fn a_blocked_shortcut_does_not_stop_the_search() {
    // Reachability along a path is not monotone. Here the jump from the
    // start to the third waypoint is blocked while the jump to the fourth
    // is clear, which is the case a greedy forward scan gets wrong: it
    // stops at the last reachable index and keeps a waypoint it did not
    // need.
    let field = KdTreeOccupancy::new(&[vec![3.0, 0.0]], 0.8).unwrap();
    let raw = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![4.0, 0.0],
        vec![4.0, 6.0],
    ];
    let checker = SegmentPolicy::Exact {
        occupancy: field.clone(),
    };
    assert!(!checker.is_segment_free(&raw[0], &raw[2]).unwrap());
    assert!(checker.is_segment_free(&raw[0], &raw[3]).unwrap());

    let pruned = pruner(field).prune(&raw).unwrap();
    assert_eq!(pruned, vec![vec![0.0, 0.0], vec![4.0, 6.0]]);
}

#[test]
fn a_path_too_short_to_shorten_is_returned_unchanged() {
    let field = KdTreeOccupancy::new(&[vec![200.0, 200.0]], 0.5).unwrap();
    let pruner = pruner(field);
    assert!(pruner.prune(&[]).unwrap().is_empty());
    let single = vec![vec![1.0, 1.0]];
    assert_eq!(pruner.prune(&single).unwrap(), single);
    let pair = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
    assert_eq!(pruner.prune(&pair).unwrap(), pair);
}

#[test]
fn a_path_whose_own_edges_are_blocked_is_returned_unchanged() {
    // The input was already invalid, and dropping waypoints from it would
    // hide that rather than fix it.
    let field = KdTreeOccupancy::new(&[vec![5.0, 0.0]], 2.0).unwrap();
    let raw = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![10.0, 0.0]];
    assert_eq!(pruner(field).prune(&raw).unwrap(), raw);
}
