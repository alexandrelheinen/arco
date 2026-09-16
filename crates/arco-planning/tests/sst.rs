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

//! SST invariants: FR-INV-01, FR-INV-02, FR-INV-08, FR-RNG-01.
//!
//! FR-INV-01 is asserted under the exact segment policy, since a sampled
//! one makes no claim finer than its own resolution.
//!
//! FR-INV-07 for SST is a suboptimality band rather than convergence, and
//! it lives in `anytime_cost.rs` next to the RRT* half of the same
//! requirement.

#![expect(clippy::unwrap_used, reason = "test fixtures and assertions")]

use arco_core::Error;
use arco_core::protocols::Occupancy;
use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    CostPolicy, SamplerPolicy, SegmentPolicy, SstPlanner, SstSettings, SteererPolicy,
};
use arco_planning::failure::PlanFailure;

const SEGMENT_SAMPLES: usize = 12;
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

fn planner(occupancy: KdTreeOccupancy, settings: SstSettings) -> SstPlanner<KdTreeOccupancy> {
    planner_with(
        SegmentPolicy::Sampled {
            occupancy,
            count: SEGMENT_SAMPLES,
        },
        settings,
    )
}

/// The same planner under an exact segment check.
fn exact_planner(occupancy: KdTreeOccupancy, settings: SstSettings) -> SstPlanner<KdTreeOccupancy> {
    planner_with(SegmentPolicy::Exact { occupancy }, settings)
}

fn planner_with(
    segments: SegmentPolicy<KdTreeOccupancy>,
    settings: SstSettings,
) -> SstPlanner<KdTreeOccupancy> {
    SstPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: vec![(0.0, 50.0), (0.0, 50.0)],
        },
        SteererPolicy::Straight {
            step_size: vec![STEP_SIZE, STEP_SIZE],
        },
        segments,
        CostPolicy::Scaled {
            step_size: vec![STEP_SIZE, STEP_SIZE],
        },
        settings,
    )
}

/// Re-checks a path at a finer resolution than the planner used.
///
/// FR-INV-01: a claim of collision freedom is only as strong as the
/// resolution it was checked at, so the check here is deliberately finer
/// than the one the planner ran.
fn path_is_collision_free(occupancy: &KdTreeOccupancy, path: &[Vec<f64>]) -> bool {
    const FINE_SAMPLES: u32 = 64;
    for pair in path.windows(2) {
        let [from, to] = pair else { continue };
        for step in 0..=FINE_SAMPLES {
            let ratio = f64::from(step) / f64::from(FINE_SAMPLES);
            let point: Vec<f64> = from
                .iter()
                .zip(to)
                .map(|(start, end)| start + (end - start) * ratio)
                .collect();
            if occupancy.is_occupied(&point).unwrap() {
                return false;
            }
        }
    }
    true
}

#[test]
fn a_returned_path_is_collision_free_under_the_same_map() {
    let occupancy = obstacle_field(21, 120);
    let planner = exact_planner(
        occupancy.clone(),
        SstSettings {
            max_samples: 4000,
            goal_tolerance: 1.5,
            ..SstSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(21);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    let path = outcome.path().expect("this field is crossable");
    assert!(
        path_is_collision_free(&occupancy, path),
        "a returned path crosses an obstacle when re-checked finely"
    );
}

#[test]
fn a_path_starts_at_the_start_and_reaches_the_goal() {
    let occupancy = obstacle_field(22, 100);
    let planner = planner(
        occupancy,
        SstSettings {
            max_samples: 4000,
            goal_tolerance: 1.5,
            ..SstSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(22);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    let path = outcome.path().expect("this field is crossable");
    assert_eq!(path.first().unwrap(), &vec![2.0, 2.0]);
    let last = path.last().unwrap();
    let remaining = (last[0] - 48.0).hypot(last[1] - 48.0) / STEP_SIZE;
    assert!(remaining <= 1.5 + 1e-9, "path ends {remaining} steps short");
}

#[test]
fn consecutive_states_are_within_one_step() {
    // FR-INV-02. SST propagates by one steering step like RRT* does, so
    // the same bound holds on every edge the tree contributed.
    let occupancy = obstacle_field(23, 100);
    let planner = planner(
        occupancy,
        SstSettings {
            max_samples: 4000,
            goal_tolerance: 1.5,
            ..SstSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(23);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    let path = outcome.path().expect("this field is crossable");
    // The last edge may be the appended goal segment, which is bounded by
    // the goal tolerance rather than by the step.
    for pair in path.windows(2).take(path.len().saturating_sub(2)) {
        let [from, to] = pair else { continue };
        let length = (to[0] - from[0]).hypot(to[1] - from[1]) / STEP_SIZE;
        assert!(
            length <= 1.0 + 1e-9,
            "edge {from:?} to {to:?} spans {length} steps"
        );
    }
}

#[test]
fn the_same_seed_replays_the_same_path() {
    // FR-RNG-01.
    let occupancy = obstacle_field(24, 120);
    let planner = planner(occupancy, SstSettings::default());

    let run = || {
        let mut generator = Pcg64::seed_from_u64(4242);
        planner
            .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
            .unwrap()
    };
    let first = run();
    for _ in 0..4 {
        let again = run();
        assert_eq!(first.path(), again.path());
        assert_eq!(first.cost(), again.cost());
    }
}

#[test]
fn a_denser_witness_grid_keeps_more_of_the_tree() {
    // The defining property of SST: the witness radius is what trades
    // tree size against path quality. A radius near one step retires a
    // node almost every time one is added, so the same budget reaches the
    // goal from far fewer surviving nodes and pays for it in cost.
    let occupancy = obstacle_field(25, 80);
    let settings = |radius| SstSettings {
        max_samples: 6000,
        goal_tolerance: 1.5,
        witness_radius: radius,
        early_stop: false,
        ..SstSettings::default()
    };

    let dense = planner(occupancy.clone(), settings(0.1));
    let sparse = planner(occupancy, settings(0.95));

    let mut dense_generator = Pcg64::seed_from_u64(31);
    let mut sparse_generator = Pcg64::seed_from_u64(31);
    let dense_cost = dense
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut dense_generator)
        .unwrap()
        .cost();
    let sparse_cost = sparse
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut sparse_generator)
        .unwrap()
        .cost();

    let dense_cost = dense_cost.expect("the dense run crosses this field");
    if let Some(sparse_cost) = sparse_cost {
        assert!(
            dense_cost <= sparse_cost + 1e-9,
            "a denser witness grid gave a worse path: {dense_cost} against {sparse_cost}"
        );
    }
}

#[test]
fn a_witness_radius_of_a_whole_step_is_rejected() {
    // At one step the tree cannot grow, and the honest report of that is
    // an error naming the radius rather than an exhausted budget on an
    // open map.
    let occupancy = KdTreeOccupancy::new(&[vec![200.0, 200.0]], 0.5).unwrap();
    for radius in [0.0, -0.5, 1.0, 2.0, f64::NAN] {
        let planner = planner(
            occupancy.clone(),
            SstSettings {
                witness_radius: radius,
                ..SstSettings::default()
            },
        );
        let mut generator = Pcg64::seed_from_u64(1);
        assert!(
            matches!(
                planner.plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator),
                Err(Error::OutOfRange {
                    quantity: "witness radius",
                    ..
                })
            ),
            "accepted a witness radius of {radius}"
        );
    }
}

#[test]
fn an_occupied_start_is_named_as_such() {
    let occupancy = KdTreeOccupancy::new(&[vec![2.0, 2.0]], 1.0).unwrap();
    let planner = planner(occupancy, SstSettings::default());
    let mut generator = Pcg64::seed_from_u64(1);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::StartOccupied));
    assert!(!PlanFailure::StartOccupied.is_retryable());
}

#[test]
fn an_occupied_goal_is_named_as_such() {
    let occupancy = KdTreeOccupancy::new(&[vec![48.0, 48.0]], 1.0).unwrap();
    let planner = planner(occupancy, SstSettings::default());
    let mut generator = Pcg64::seed_from_u64(1);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::GoalOccupied));
}

#[test]
fn running_out_of_samples_is_reported_as_retryable() {
    // FR-SAFE-02. Sampling never proves a goal unreachable, so a failure
    // to find one is always the retryable answer.
    let occupancy = obstacle_field(26, 100);
    let planner = planner(
        occupancy,
        SstSettings {
            max_samples: 5,
            ..SstSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(1);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::BudgetExhausted));
    assert!(PlanFailure::BudgetExhausted.is_retryable());
    assert_eq!(outcome.expanded(), 5);
}

#[test]
fn a_non_finite_or_mismatched_query_is_rejected() {
    let occupancy = obstacle_field(27, 20);
    let planner = planner(occupancy, SstSettings::default());
    let mut generator = Pcg64::seed_from_u64(1);
    assert!(
        planner
            .plan(&[f64::NAN, 2.0], &[48.0, 48.0], &mut generator)
            .is_err()
    );
    assert!(planner.plan(&[2.0, 2.0], &[48.0], &mut generator).is_err());
}

#[test]
fn an_empty_space_is_crossed_directly() {
    let occupancy = KdTreeOccupancy::new(&[vec![200.0, 200.0]], 0.5).unwrap();
    let planner = planner(
        occupancy,
        SstSettings {
            max_samples: 4000,
            goal_tolerance: 1.0,
            ..SstSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(11);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    let cost = outcome.cost().expect("an empty space is always crossable");
    // Cost is measured in steps, so the straight line is converted before
    // the comparison rather than after.
    let straight = (48.0_f64 - 2.0).hypot(48.0 - 2.0) / STEP_SIZE;
    assert!(
        cost >= straight - 1e-9,
        "shorter than a straight line: {cost}"
    );
}
