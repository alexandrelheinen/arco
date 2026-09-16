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

//! The escape hatches, exercised.
//!
//! Every policy enum carries a `Custom` variant so a caller can supply a
//! sampler, a steerer, a segment check, a metric or a cost term of its
//! own. ADR-004 documents that path as the slow one and deviation A-07
//! says so again, which is not the same as saying it works. These tests
//! are what say it works.

use arco_core::Error;
use arco_core::protocols::{CostTerm, PlannerCost, Sampler, SegmentChecker, Steerer};
use arco_core::rng::Pcg64;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    CostPolicy, FeasibilityPolicy, OptimizerSettings, RrtPlanner, RrtSettings, SamplerPolicy,
    SegmentPolicy, SteererPolicy, TermWeights, TrajectoryContext, TrajectoryOptimizer,
    TrajectoryTerm,
};
use arco_planning::failure::{PlanFailure, PlanOutcome};

/// Distance along the axes rather than through them.
#[derive(Debug)]
struct ManhattanCost;

impl PlannerCost for ManhattanCost {
    fn distance(&self, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        arco_core::geometry::manhattan_distance(from, to)
    }

    fn heuristic(&self, from: &[f64], to: &[f64]) -> Result<f64, Error> {
        self.distance(from, to)
    }
}

/// Draws the same state every time, which makes a run inspectable.
#[derive(Debug)]
struct FixedSampler {
    state: Vec<f64>,
}

impl Sampler for FixedSampler {
    fn sample(&self, _generator: &mut Pcg64) -> Result<Vec<f64>, Error> {
        Ok(self.state.clone())
    }
}

/// Steps a fixed fraction of the way toward the target.
#[derive(Debug)]
struct FractionSteerer {
    fraction: f64,
}

impl Steerer for FractionSteerer {
    fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<f64>, Error> {
        arco_core::geometry::require_dimension("target", to, from.len())?;
        Ok(from
            .iter()
            .zip(to)
            .map(|(start, end)| (end - start).mul_add(self.fraction, *start))
            .collect())
    }
}

/// Refuses every segment, including a point against itself.
#[derive(Debug)]
struct ClosedSpace;

impl SegmentChecker for ClosedSpace {
    fn is_segment_free(&self, _from: &[f64], _to: &[f64]) -> Result<bool, Error> {
        Ok(false)
    }
}

/// Accepts every segment.
#[derive(Debug)]
struct OpenSpace;

impl SegmentChecker for OpenSpace {
    fn is_segment_free(&self, _from: &[f64], _to: &[f64]) -> Result<bool, Error> {
        Ok(true)
    }
}

/// A cost term that reports a constant, to prove it was summed.
#[derive(Debug)]
struct FlatTerm {
    value: f64,
}

impl CostTerm<TrajectoryContext<'_>> for FlatTerm {
    fn evaluate(&self, _context: &TrajectoryContext<'_>) -> Result<f64, Error> {
        Ok(self.value)
    }
}

/// A term that reads the context, so a broken context would show here.
#[derive(Debug)]
struct TotalTimeTerm;

impl CostTerm<TrajectoryContext<'_>> for TotalTimeTerm {
    fn evaluate(&self, context: &TrajectoryContext<'_>) -> Result<f64, Error> {
        assert_eq!(context.waypoints.len(), context.reference.len());
        assert_eq!(context.durations.len(), context.speeds.len());
        assert_eq!(context.lengths.len(), context.speeds.len());
        assert!(format!("{context:?}").contains("TrajectoryContext"));
        Ok(context.total_duration())
    }
}

#[test]
fn a_custom_metric_changes_what_the_planner_calls_close() {
    // Manhattan distance is never below the straight line, so a tolerance
    // that the straight-line metric clears may not be cleared here. The
    // point is that the planner asked the policy rather than measuring
    // for itself.
    let metric = CostPolicy::Custom(Box::new(ManhattanCost));
    let straight = CostPolicy::Scaled {
        step_size: Vec::new(),
    };
    let from = [0.0, 0.0];
    let to = [3.0, 4.0];
    assert!((metric.distance(&from, &to).unwrap() - 7.0).abs() < 1e-12);
    assert!((straight.distance(&from, &to).unwrap() - 5.0).abs() < 1e-12);
    assert!((metric.heuristic(&from, &to).unwrap() - 7.0).abs() < 1e-12);
    assert_eq!(metric.axis_scale(0), None);
    assert_eq!(straight.axis_scale(0), Some(1.0));
    assert!(format!("{metric:?}").contains("Custom"));
    assert!(format!("{straight:?}").contains("Scaled"));
}

#[test]
fn a_scale_that_is_not_positive_is_rejected() {
    for scale in [0.0, -2.0, f64::NAN, f64::INFINITY] {
        let metric = CostPolicy::Scaled {
            step_size: vec![scale, 1.0],
        };
        assert!(
            matches!(
                metric.distance(&[0.0, 0.0], &[1.0, 1.0]),
                Err(Error::OutOfRange {
                    quantity: "step size",
                    ..
                })
            ),
            "accepted a step size of {scale}"
        );
    }
}

#[test]
fn a_metric_rejects_states_of_different_dimension() {
    let metric = CostPolicy::default();
    assert!(matches!(
        metric.distance(&[0.0, 0.0], &[1.0]),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn a_custom_sampler_is_the_one_the_planner_draws_from() {
    // The sampler always returns the goal, so the tree reaches it on the
    // first iteration and the run is one step long.
    let planner = RrtPlanner::new(
        SamplerPolicy::Custom(Box::new(FixedSampler {
            state: vec![1.0, 0.0],
        })),
        SteererPolicy::Straight {
            step_size: vec![1.0, 1.0],
        },
        SegmentPolicy::<KdTreeOccupancy>::Custom(Box::new(OpenSpace)),
        CostPolicy::default(),
        RrtSettings {
            max_samples: 50,
            goal_tolerance: 0.1,
            goal_bias: 0.0,
            ..RrtSettings::default()
        },
    );
    let mut generator = Pcg64::seed_from_u64(1);
    let outcome = planner
        .plan(&[0.0, 0.0], &[1.0, 0.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.expanded(), 1);
    assert_eq!(
        outcome.path().unwrap(),
        [vec![0.0, 0.0], vec![1.0, 0.0]].as_slice()
    );
}

#[test]
fn a_sampler_with_no_bounds_says_so() {
    let empty = SamplerPolicy::UniformBox { bounds: Vec::new() };
    let mut generator = Pcg64::seed_from_u64(1);
    assert!(matches!(
        empty.sample(&mut generator),
        Err(Error::TooFew {
            quantity: "sampling bounds",
            ..
        })
    ));
    assert_eq!(empty.dimension(), Some(0));

    let custom = SamplerPolicy::Custom(Box::new(FixedSampler {
        state: vec![0.0, 0.0],
    }));
    assert_eq!(custom.dimension(), None);
    assert!(custom.sample(&mut generator).is_ok());
    assert!(format!("{custom:?}").contains("Custom"));
    assert!(format!("{empty:?}").contains("UniformBox"));
}

#[test]
fn a_custom_steerer_is_the_one_the_planner_steps_with() {
    let steerer = SteererPolicy::Custom(Box::new(FractionSteerer { fraction: 0.25 }));
    let stepped = steerer.steer(&[0.0, 0.0], &[4.0, 8.0]).unwrap();
    assert_eq!(stepped, vec![1.0, 2.0]);
    assert!(format!("{steerer:?}").contains("Custom"));
}

#[test]
fn a_steerer_already_within_a_step_lands_on_the_target() {
    let steerer = SteererPolicy::Straight {
        step_size: vec![2.0, 2.0],
    };
    assert_eq!(
        steerer.steer(&[0.0, 0.0], &[1.0, 1.0]).unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        steerer.steer(&[3.0, 3.0], &[3.0, 3.0]).unwrap(),
        vec![3.0, 3.0]
    );
    assert!(steerer.steer(&[0.0, 0.0], &[1.0]).is_err());
}

#[test]
fn a_custom_segment_check_that_refuses_everything_blocks_the_start() {
    // A custom checker has no notion of a single point, so the planner
    // probes an endpoint as a zero-length segment. A checker that refuses
    // everything therefore reports the start as occupied, which is the
    // branch this covers.
    let planner = RrtPlanner::new(
        SamplerPolicy::UniformBox {
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
        },
        SteererPolicy::Straight {
            step_size: vec![1.0, 1.0],
        },
        SegmentPolicy::<KdTreeOccupancy>::Custom(Box::new(ClosedSpace)),
        CostPolicy::default(),
        RrtSettings::default(),
    );
    let mut generator = Pcg64::seed_from_u64(1);
    let outcome = planner
        .plan(&[0.0, 0.0], &[1.0, 1.0], &mut generator)
        .unwrap();
    assert_eq!(outcome.failure(), Some(PlanFailure::StartOccupied));
}

#[test]
fn a_segment_policy_states_the_resolution_it_checked_at() {
    let occupancy = KdTreeOccupancy::new(&[vec![100.0, 100.0]], 0.5).unwrap();
    let sampled = SegmentPolicy::Sampled {
        occupancy: occupancy.clone(),
        count: 7,
    };
    let exact = SegmentPolicy::Exact { occupancy };
    let custom: SegmentPolicy<KdTreeOccupancy> = SegmentPolicy::Custom(Box::new(OpenSpace));

    assert_eq!(sampled.validity_samples(), Some(7));
    assert_eq!(exact.validity_samples(), None);
    assert_eq!(custom.validity_samples(), Some(2));
    assert!(format!("{sampled:?}").contains("Sampled"));
    assert!(format!("{exact:?}").contains("Exact"));
    assert!(format!("{custom:?}").contains("Custom"));
    assert!(custom.is_segment_free(&[0.0, 0.0], &[1.0, 1.0]).unwrap());
}

#[test]
fn every_failure_reason_reads_as_a_sentence() {
    // FR-INV-08 asks for a closed enumeration a caller can branch on, and
    // a reason that reaches a log as `Failed` helps nobody.
    let reasons = [
        (PlanFailure::StartOccupied, "start"),
        (PlanFailure::GoalOccupied, "goal"),
        (PlanFailure::StartOutsideMap, "outside"),
        (PlanFailure::GoalOutsideMap, "outside"),
        (PlanFailure::Unreachable, "no path"),
        (PlanFailure::BudgetExhausted, "budget"),
    ];
    for (reason, fragment) in reasons {
        let rendered = reason.to_string();
        assert!(
            rendered.contains(fragment),
            "{reason:?} renders as {rendered}, which does not mention {fragment}"
        );
        assert_eq!(
            reason.is_retryable(),
            reason == PlanFailure::BudgetExhausted
        );
    }
}

#[test]
fn an_outcome_answers_about_itself_either_way() {
    let found = PlanOutcome::Found {
        path: vec![1_u8, 2],
        cost: 3.0,
        expanded: 4,
    };
    assert_eq!(found.path(), Some([1_u8, 2].as_slice()));
    assert_eq!(found.cost(), Some(3.0));
    assert_eq!(found.failure(), None);
    assert_eq!(found.expanded(), 4);

    let failed: PlanOutcome<u8> = PlanOutcome::Failed {
        reason: PlanFailure::Unreachable,
        expanded: 9,
    };
    assert_eq!(failed.path(), None);
    assert_eq!(failed.cost(), None);
    assert_eq!(failed.failure(), Some(PlanFailure::Unreachable));
    assert_eq!(failed.expanded(), 9);
}

#[test]
fn a_custom_cost_term_is_summed_with_the_rest() {
    let occupancy = KdTreeOccupancy::new(&[vec![500.0, 500.0]], 0.5).unwrap();
    let reference = vec![vec![0.0, 0.0], vec![4.0, 0.0], vec![8.0, 0.0]];
    let settings = OptimizerSettings {
        cruise_speed: 2.0,
        max_iterations: 0,
        ..OptimizerSettings::default()
    };

    let without = TrajectoryOptimizer::new(
        occupancy.clone(),
        vec![TrajectoryTerm::Time { weight: 1.0 }],
        settings,
    )
    .unwrap()
    .optimize(&reference, &FeasibilityPolicy::Unchecked)
    .unwrap();

    let with = TrajectoryOptimizer::new(
        occupancy,
        vec![
            TrajectoryTerm::Time { weight: 1.0 },
            TrajectoryTerm::Custom(Box::new(FlatTerm { value: 11.0 })),
        ],
        settings,
    )
    .unwrap()
    .optimize(&reference, &FeasibilityPolicy::Unchecked)
    .unwrap();

    assert!((with.cost - without.cost - 11.0).abs() < 1e-9);
}

#[test]
fn a_custom_term_sees_a_consistent_context() {
    let occupancy = KdTreeOccupancy::new(&[vec![500.0, 500.0]], 0.5).unwrap();
    let reference = vec![vec![0.0, 0.0], vec![3.0, 0.0], vec![6.0, 0.0]];
    let result = TrajectoryOptimizer::new(
        occupancy,
        vec![TrajectoryTerm::Custom(Box::new(TotalTimeTerm))],
        OptimizerSettings {
            cruise_speed: 1.5,
            max_iterations: 0,
            ..OptimizerSettings::default()
        },
    )
    .unwrap()
    .optimize(&reference, &FeasibilityPolicy::Unchecked)
    .unwrap();

    let total: f64 = result.durations.iter().sum();
    assert!((result.cost - total).abs() < 1e-9);
}

#[test]
fn a_term_that_returns_a_non_finite_cost_is_refused() {
    // FR-SAFE-05. A NaN cost makes every later comparison false, so the
    // optimizer would silently accept whatever it was holding.
    let occupancy = KdTreeOccupancy::new(&[vec![500.0, 500.0]], 0.5).unwrap();
    let reference = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
    let built = TrajectoryOptimizer::new(
        occupancy,
        vec![TrajectoryTerm::Custom(Box::new(FlatTerm {
            value: f64::NAN,
        }))],
        OptimizerSettings::default(),
    )
    .unwrap()
    .optimize(&reference, &FeasibilityPolicy::Unchecked);
    assert!(matches!(built, Err(Error::NotFinite { .. })));
}

#[test]
fn every_default_term_names_itself() {
    let terms = TrajectoryTerm::defaults(
        TermWeights::default(),
        1.0,
        (50.0, 4.0),
        (Some(2.0), Some(0.1)),
        3,
    );
    let names: Vec<String> = terms.iter().map(|term| format!("{term:?}")).collect();
    assert_eq!(
        names,
        ["Time", "Deviation", "Velocity", "Collision", "Dynamics"]
    );
}

#[test]
fn a_feasibility_policy_names_itself_and_answers() {
    let state = arco_planning::continuous::DerivedState {
        x: 0.0,
        y: 0.0,
        heading: 0.0,
        speed: 3.0,
        turn_rate: 0.5,
    };
    let unchecked = FeasibilityPolicy::Unchecked;
    assert!(unchecked.accepts(state));
    assert!(format!("{unchecked:?}").contains("Unchecked"));

    let bounded = FeasibilityPolicy::Bounded {
        max_speed: Some(2.0),
        min_speed: None,
        max_turn_rate: None,
    };
    assert!(!bounded.accepts(state));
    assert!(format!("{bounded:?}").contains("Bounded"));

    let slow = FeasibilityPolicy::Bounded {
        max_speed: None,
        min_speed: Some(4.0),
        max_turn_rate: None,
    };
    assert!(!slow.accepts(state));

    let turning = FeasibilityPolicy::Bounded {
        max_speed: None,
        min_speed: None,
        max_turn_rate: Some(0.1),
    };
    assert!(!turning.accepts(state));

    let custom = FeasibilityPolicy::Custom(Box::new(|check| check.speed > 0.0));
    assert!(custom.accepts(state));
    assert!(format!("{custom:?}").contains("Custom"));
}
