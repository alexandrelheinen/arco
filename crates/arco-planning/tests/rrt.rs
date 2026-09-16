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

//! RRT* invariants: FR-INV-01, FR-INV-02, FR-INV-07, FR-INV-08, FR-RNG-01.

#![expect(clippy::unwrap_used, reason = "test fixtures and assertions")]

use arco_core::protocols::Occupancy;
use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    RrtPlanner, RrtSettings, SamplerPolicy, SegmentPolicy, SteererPolicy,
};
use arco_planning::failure::PlanFailure;

const SEGMENT_SAMPLES: usize = 12;

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

fn planner(occupancy: KdTreeOccupancy, settings: RrtSettings) -> RrtPlanner<KdTreeOccupancy> {
    RrtPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: vec![(0.0, 50.0), (0.0, 50.0)],
        },
        SteererPolicy::Straight {
            step_size: vec![2.0, 2.0],
        },
        SegmentPolicy::Sampled {
            occupancy,
            count: SEGMENT_SAMPLES,
        },
        settings,
    )
}

/// Re-checks a path at a finer resolution than the planner used.
///
/// FR-INV-01: the claim is only as strong as the resolution it was
/// checked at, so the check here is deliberately finer than the planner's.
fn path_is_collision_free(occupancy: &KdTreeOccupancy, path: &[Vec<f64>]) -> bool {
    const VALIDATION_SAMPLES: usize = 200;
    path.windows(2).all(|pair| {
        let [from, to] = pair else { return true };
        (0..=VALIDATION_SAMPLES).all(|step| {
            let ratio = f64::from(u32::try_from(step).unwrap_or(0))
                / f64::from(u32::try_from(VALIDATION_SAMPLES).unwrap_or(1));
            let sample: Vec<f64> = from
                .iter()
                .zip(to)
                .map(|(start, end)| start + (end - start) * ratio)
                .collect();
            occupancy.is_occupied(&sample) == Ok(false)
        })
    })
}

#[test]
fn a_returned_path_is_collision_free_under_the_same_map() {
    // FR-INV-01, re-checked at 200 samples per segment against the
    // planner's 12.
    for seed in 0..8_u64 {
        let occupancy = obstacle_field(seed, 150);
        let planner = planner(
            occupancy.clone(),
            RrtSettings {
                max_samples: 3000,
                ..RrtSettings::default()
            },
        );
        let mut generator = Pcg64::seed_from_u64(seed);
        let outcome = planner
            .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
            .unwrap();

        if let Some(path) = outcome.path() {
            assert!(
                path_is_collision_free(&occupancy, path),
                "seed {seed}: the path crosses an obstacle"
            );
        }
    }
}

#[test]
fn a_path_starts_at_the_start_and_reaches_the_goal() {
    // FR-INV-02 at the endpoints.
    let occupancy = obstacle_field(1, 100);
    let planner = planner(occupancy, RrtSettings::default());
    let mut generator = Pcg64::seed_from_u64(1);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    if let Some(path) = outcome.path() {
        assert_eq!(path.first().map(Vec::as_slice), Some([2.0, 2.0].as_slice()));
        let last = path.last().unwrap();
        let reached = (last[0] - 48.0).hypot(last[1] - 48.0);
        assert!(
            reached <= RrtSettings::default().goal_tolerance + 1e-9,
            "{reached}"
        );
    }
}

#[test]
fn consecutive_states_are_within_one_step() {
    // FR-INV-02: the steerer caps each edge, and the path is made of
    // those edges, so no edge may exceed the cap.
    let occupancy = obstacle_field(2, 100);
    let planner = planner(
        occupancy,
        RrtSettings {
            max_samples: 2000,
            goal_tolerance: 1.5,
            ..RrtSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(2);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    if let Some(path) = outcome.path() {
        // The last edge may be the appended goal segment, which is only
        // bounded by the goal tolerance.
        for pair in path.windows(2).take(path.len().saturating_sub(2)) {
            let [from, to] = pair else { continue };
            for axis in 0..2 {
                assert!(
                    (to[axis] - from[axis]).abs() <= 2.0 + 1e-9,
                    "edge {from:?} to {to:?} exceeds the step on axis {axis}"
                );
            }
        }
    }
}

#[test]
fn the_same_seed_replays_the_same_path() {
    // FR-RNG-01. Without this, nothing else here is reproducible.
    let occupancy = obstacle_field(3, 120);
    let planner = planner(occupancy, RrtSettings::default());

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
fn different_seeds_explore_differently() {
    let occupancy = obstacle_field(5, 120);
    let planner = planner(occupancy, RrtSettings::default());

    let mut first_generator = Pcg64::seed_from_u64(1);
    let mut second_generator = Pcg64::seed_from_u64(2);
    let first = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut first_generator)
        .unwrap();
    let second = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut second_generator)
        .unwrap();
    assert_ne!(first.path(), second.path());
}

#[test]
fn rewiring_never_makes_the_incumbent_worse() {
    // FR-INV-07. With early stop off, a longer run explores more and may
    // improve the solution, and must never degrade it.
    let occupancy = obstacle_field(6, 100);
    let settings = |samples| RrtSettings {
        max_samples: samples,
        early_stop: false,
        goal_tolerance: 1.5,
        ..RrtSettings::default()
    };

    let short = planner(occupancy.clone(), settings(800));
    let long = planner(occupancy, settings(3000));

    let mut short_generator = Pcg64::seed_from_u64(9);
    let mut long_generator = Pcg64::seed_from_u64(9);
    let short_cost = short
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut short_generator)
        .unwrap()
        .cost();
    let long_cost = long
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut long_generator)
        .unwrap()
        .cost();

    if let (Some(short_cost), Some(long_cost)) = (short_cost, long_cost) {
        assert!(
            long_cost <= short_cost + 1e-9,
            "more samples gave a worse path: {long_cost} against {short_cost}"
        );
    }
}

#[test]
fn an_occupied_start_is_named_as_such() {
    // FR-INV-08.
    let occupancy = KdTreeOccupancy::new(&[vec![2.0, 2.0]], 1.0).unwrap();
    let planner = planner(occupancy, RrtSettings::default());
    let mut generator = Pcg64::seed_from_u64(0);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::StartOccupied));
}

#[test]
fn an_occupied_goal_is_named_as_such() {
    let occupancy = KdTreeOccupancy::new(&[vec![48.0, 48.0]], 1.0).unwrap();
    let planner = planner(occupancy, RrtSettings::default());
    let mut generator = Pcg64::seed_from_u64(0);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::GoalOccupied));
}

#[test]
fn running_out_of_samples_is_reported_as_retryable() {
    // Sampling never proves a goal unreachable, it only runs out of
    // samples, so the answer is always the retryable one.
    let occupancy = obstacle_field(7, 300);
    let planner = planner(
        occupancy,
        RrtSettings {
            max_samples: 5,
            ..RrtSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(0);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    assert_eq!(outcome.failure(), Some(PlanFailure::BudgetExhausted));
    assert!(PlanFailure::BudgetExhausted.is_retryable());
}

#[test]
fn a_non_finite_or_mismatched_query_is_rejected() {
    let occupancy = obstacle_field(8, 50);
    let planner = planner(occupancy, RrtSettings::default());
    let mut generator = Pcg64::seed_from_u64(0);

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
        RrtSettings {
            max_samples: 4000,
            goal_tolerance: 1.0,
            ..RrtSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(11);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut generator)
        .unwrap();

    let cost = outcome.cost().expect("an empty space is always crossable");
    let straight = (48.0_f64 - 2.0).hypot(48.0 - 2.0);
    assert!(
        cost >= straight - 1e-9,
        "shorter than a straight line: {cost}"
    );
    assert!(cost < straight * 2.0, "far longer than necessary: {cost}");
}

#[test]
fn the_benchmark_scenario_actually_finds_a_path() {
    // A benchmark that measures a fast failure measures nothing. This
    // mirrors benches/planning.rs exactly and asserts the work is real.
    let mut generator = Pcg64::seed_from_u64(20_260_916);
    let points: Vec<Vec<f64>> = (0..400)
        .map(|_| {
            vec![
                generator.next_f64().mul_add(42.0, 4.0),
                generator.next_f64().mul_add(42.0, 4.0),
            ]
        })
        .collect();
    let occupancy = KdTreeOccupancy::new(&points, 1.2).unwrap();

    let planner = planner(
        occupancy.clone(),
        RrtSettings {
            max_samples: 4000,
            goal_tolerance: 1.5,
            ..RrtSettings::default()
        },
    );
    let mut planner_generator = Pcg64::seed_from_u64(7);
    let outcome = planner
        .plan(&[2.0, 2.0], &[48.0, 48.0], &mut planner_generator)
        .unwrap();

    let path = outcome
        .path()
        .expect("the benchmark scenario must be solvable, or it times nothing");
    assert!(path.len() > 10, "suspiciously short path: {}", path.len());
    assert!(path_is_collision_free(&occupancy, path));
    assert!(
        outcome.expanded() > 50,
        "barely sampled: {}",
        outcome.expanded()
    );
    println!(
        "benchmark scenario: {} path states, {} samples drawn",
        path.len(),
        outcome.expanded()
    );
}
