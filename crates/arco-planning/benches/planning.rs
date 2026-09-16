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

//! FR-PERF-01, measured against the recorded Python baseline.
//!
//! The scenario is the one `benches/capture_baseline.py` timed before any
//! Rust existed: a fifty metre square, four hundred obstacles at a
//! clearance of 1.2, planning corner to corner with every policy at its
//! default. The Python median was 3.125 seconds, so the target here is
//! 312 milliseconds.
//!
//! The segment policy is the sampled one, matching what the Python
//! planner did, so the two sides are doing the same work rather than the
//! Rust side doing more of it.

// A benchmark is neither library code nor a #[test] function, so the
// allowances in clippy.toml do not reach it, and criterion_main generates
// an undocumented function.
#![expect(
    clippy::expect_used,
    missing_docs,
    reason = "benchmark fixtures, and a generated main"
)]

use std::hint::black_box;

use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    CostPolicy, RrtPlanner, RrtSettings, SamplerPolicy, SegmentPolicy, SteererPolicy,
};
use criterion::{Criterion, criterion_group, criterion_main};

/// Matches the `RRT_SCENARIO` dictionary in `benches/capture_baseline.py`.
const MAP_SEED: u64 = 20_260_916;
const PLANNER_SEED: u64 = 7;
const OBSTACLE_COUNT: usize = 400;
const CLEARANCE: f64 = 1.2;
const MAX_SAMPLES: usize = 4000;
const STEP_SIZE: f64 = 2.0;
const GOAL_TOLERANCE: f64 = 1.5;
const START: [f64; 2] = [2.0, 2.0];
const GOAL: [f64; 2] = [48.0, 48.0];

fn scenario_occupancy() -> KdTreeOccupancy {
    let mut generator = Pcg64::seed_from_u64(MAP_SEED);
    let points: Vec<Vec<f64>> = (0..OBSTACLE_COUNT)
        .map(|_| {
            vec![
                generator.next_f64().mul_add(42.0, 4.0),
                generator.next_f64().mul_add(42.0, 4.0),
            ]
        })
        .collect();
    KdTreeOccupancy::new(&points, CLEARANCE).expect("the scenario field is well formed")
}

fn scenario_planner(occupancy: KdTreeOccupancy) -> RrtPlanner<KdTreeOccupancy> {
    RrtPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: vec![(0.0, 50.0), (0.0, 50.0)],
        },
        SteererPolicy::Straight {
            step_size: vec![STEP_SIZE, STEP_SIZE],
        },
        SegmentPolicy::Sampled {
            occupancy,
            count: 12,
        },
        CostPolicy::Scaled {
            step_size: vec![STEP_SIZE, STEP_SIZE],
        },
        RrtSettings {
            max_samples: MAX_SAMPLES,
            goal_tolerance: GOAL_TOLERANCE,
            ..RrtSettings::default()
        },
    )
}

fn rrt_star(criterion: &mut Criterion) {
    let planner = scenario_planner(scenario_occupancy());

    criterion.bench_function("rrt_star/baseline_scenario", |bencher| {
        bencher.iter(|| {
            let mut generator = Pcg64::seed_from_u64(PLANNER_SEED);
            let outcome = planner
                .plan(&START, &GOAL, &mut generator)
                .expect("the scenario query is well formed");
            black_box(outcome)
        });
    });
}

criterion_group!(benches, rrt_star);
criterion_main!(benches);
