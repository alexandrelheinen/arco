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

//! FR-INV-07: what more search is allowed to do to the answer.
//!
//! The two planners promise different things and the tests differ to
//! match. RRT* is asymptotically optimal, so its incumbent may only
//! improve. SST is asymptotically near-optimal, so its incumbent may only
//! improve and has to land inside a stated band rather than converge.
//!
//! Both planners draw from the same generator in the same order whatever
//! their budget, so a run of `n` samples passes through exactly the state
//! a run of `m < n` samples ended in. A ladder of budgets under one seed
//! is therefore a trace of one run's incumbent, which is what the
//! requirement is written about.

#![expect(clippy::unwrap_used, reason = "test fixtures and assertions")]

use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    CostPolicy, RrtPlanner, RrtSettings, SamplerPolicy, SegmentPolicy, SstPlanner, SstSettings,
    SteererPolicy,
};

const STEP_SIZE: f64 = 2.0;
const GOAL_TOLERANCE: f64 = 1.5;
const START: [f64; 2] = [2.0, 2.0];
const GOAL: [f64; 2] = [48.0, 48.0];
const LADDER: [usize; 5] = [500, 1000, 2000, 4000, 8000];

/// How far above RRT* an SST answer on the same field may sit.
///
/// SST keeps one node per witness cell, so its tree is coarser than RRT*'s
/// by roughly the witness radius and its paths are correspondingly longer.
/// Measured across ten fields the ratio sits between 1.15 and 1.32 at a
/// witness radius of half a step; the band is set above that with room,
/// because the claim being tested is that the gap is bounded rather than
/// that it takes one particular value.
const SUBOPTIMALITY_BAND: f64 = 1.5;

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

fn bounds() -> SamplerPolicy {
    SamplerPolicy::UniformBox {
        bounds: vec![(0.0, 50.0), (0.0, 50.0)],
    }
}

fn steerer() -> SteererPolicy {
    SteererPolicy::Straight {
        step_size: vec![STEP_SIZE, STEP_SIZE],
    }
}

fn metric() -> CostPolicy {
    CostPolicy::Scaled {
        step_size: vec![STEP_SIZE, STEP_SIZE],
    }
}

/// The cost RRT* reports after `budget` samples on `field` under `seed`.
fn rrt_cost(field: &KdTreeOccupancy, seed: u64, budget: usize) -> Option<f64> {
    let planner = RrtPlanner::new(
        bounds(),
        steerer(),
        SegmentPolicy::Exact {
            occupancy: field.clone(),
        },
        metric(),
        RrtSettings {
            max_samples: budget,
            goal_tolerance: GOAL_TOLERANCE,
            early_stop: false,
            ..RrtSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(seed);
    planner.plan(&START, &GOAL, &mut generator).unwrap().cost()
}

/// The cost SST reports after `budget` samples on `field` under `seed`.
fn sst_cost(field: &KdTreeOccupancy, seed: u64, budget: usize) -> Option<f64> {
    let planner = SstPlanner::new(
        bounds(),
        steerer(),
        SegmentPolicy::Exact {
            occupancy: field.clone(),
        },
        metric(),
        SstSettings {
            max_samples: budget,
            goal_tolerance: GOAL_TOLERANCE,
            early_stop: false,
            ..SstSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(seed);
    planner.plan(&START, &GOAL, &mut generator).unwrap().cost()
}

#[test]
fn more_search_never_makes_the_rrt_incumbent_worse() {
    // FR-INV-07 for RRT*. Rewiring only ever lowers a node's cost and the
    // set of goal-reaching nodes only grows, so the cheapest of them can
    // only fall. A rise here means a stale cost survived a rewiring.
    let field = obstacle_field(7, 100);
    let mut previous = f64::INFINITY;
    let mut improvements = 0_usize;
    for budget in LADDER {
        let Some(cost) = rrt_cost(&field, 9, budget) else {
            continue;
        };
        assert!(
            cost <= previous + 1e-9,
            "{budget} samples cost {cost} against {previous} at the budget below"
        );
        if cost < previous - 1e-9 {
            improvements = improvements.saturating_add(1);
        }
        previous = cost;
    }
    // A monotone sequence that never moves would pass the assertion above
    // while proving nothing about the rewiring that is under test.
    assert!(
        improvements >= 2,
        "the incumbent never improved, so monotonicity was free"
    );
}

#[test]
fn more_search_never_makes_the_sst_incumbent_worse() {
    // FR-INV-07 for SST. SST does not rewire, so its incumbent falls only
    // when a cheaper goal-reaching node appears. It must still never rise.
    let field = obstacle_field(8, 100);
    let mut previous = f64::INFINITY;
    for budget in LADDER {
        let Some(cost) = sst_cost(&field, 9, budget) else {
            continue;
        };
        assert!(
            cost <= previous + 1e-9,
            "{budget} samples cost {cost} against {previous} at the budget below"
        );
        previous = cost;
    }
}

#[test]
fn sst_stays_inside_its_suboptimality_band() {
    // FR-INV-07, the half that differs by planner. SST is asymptotically
    // near-optimal, so the claim is a bound against the optimal planner on
    // the same field rather than convergence to it.
    for seed in 0..6_u64 {
        let field = obstacle_field(seed, 100);
        let (Some(optimal), Some(sparse)) =
            (rrt_cost(&field, seed, 6000), sst_cost(&field, seed, 6000))
        else {
            continue;
        };
        assert!(
            sparse <= optimal * SUBOPTIMALITY_BAND,
            "seed {seed}: SST cost {sparse} exceeds {SUBOPTIMALITY_BAND} times {optimal}"
        );
        assert!(
            sparse >= optimal - 1e-9,
            "seed {seed}: SST beat the optimal planner, {sparse} against {optimal}"
        );
    }
}

#[test]
fn rrt_converges_toward_the_unobstructed_straight_line() {
    // The other side of asymptotic optimality: on a field it can cross
    // almost directly, more search has to close most of the gap to the
    // straight line rather than merely stop making things worse.
    let field = obstacle_field(7, 60);
    let straight = (GOAL[0] - START[0]).hypot(GOAL[1] - START[1]) / STEP_SIZE;

    let short = rrt_cost(&field, 9, 500).unwrap();
    let long = rrt_cost(&field, 9, 8000).unwrap();
    assert!(short >= straight, "cost {short} is below the straight line");
    assert!(
        long < straight * 1.05,
        "eight thousand samples still cost {long} against a straight line of {straight}"
    );
    assert!(long <= short);
}
