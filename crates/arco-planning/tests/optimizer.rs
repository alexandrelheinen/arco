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

//! What the trajectory optimizer is allowed to return.
//!
//! Deviation A-08 is the reason none of these compare a solution vector.
//! `argmin` and `scipy` reach different local minima on the same
//! nonconvex problem, and a test that pinned the waypoints would be
//! testing which solver ran rather than whether the answer is good. What
//! is comparable is the cost achieved, the endpoints held, and the
//! feasibility reported, so that is what is asserted.

#![expect(clippy::unwrap_used, reason = "test fixtures and assertions")]

use arco_core::Error;
use arco_mapping::occupancy::KdTreeOccupancy;
use arco_planning::continuous::{
    FeasibilityPolicy, OptimizerSettings, TermWeights, TrajectoryOptimizer, TrajectoryTerm,
};

const CRUISE_SPEED: f64 = 2.0;
const SAMPLES_PER_SEGMENT: usize = 3;

fn settings() -> OptimizerSettings {
    OptimizerSettings {
        cruise_speed: CRUISE_SPEED,
        max_iterations: 200,
        ..OptimizerSettings::default()
    }
}

fn terms(speed_band: (Option<f64>, Option<f64>)) -> Vec<TrajectoryTerm> {
    TrajectoryTerm::defaults(
        TermWeights::default(),
        CRUISE_SPEED,
        (50.0, 4.0),
        speed_band,
        SAMPLES_PER_SEGMENT,
    )
}

/// A field whose only obstacle is far from anything under test.
fn empty_field() -> KdTreeOccupancy {
    KdTreeOccupancy::new(&[vec![500.0, 500.0]], 0.5).unwrap()
}

/// A straight reference path along the first axis.
fn straight_reference(count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|index| vec![f64::from(u32::try_from(index).unwrap()) * 2.0, 0.0])
        .collect()
}

fn optimizer(
    occupancy: KdTreeOccupancy,
    speed_band: (Option<f64>, Option<f64>),
) -> TrajectoryOptimizer<KdTreeOccupancy> {
    TrajectoryOptimizer::new(occupancy, terms(speed_band), settings()).unwrap()
}

/// The cost of the stage-one guess, obtained by refusing to refine it.
fn guess_cost(reference: &[Vec<f64>]) -> f64 {
    TrajectoryOptimizer::new(
        empty_field(),
        terms((None, None)),
        OptimizerSettings {
            cruise_speed: CRUISE_SPEED,
            max_iterations: 0,
            ..OptimizerSettings::default()
        },
    )
    .unwrap()
    .optimize(reference, &FeasibilityPolicy::Unchecked)
    .unwrap()
    .cost
}

#[test]
fn optimizing_never_costs_more_than_the_initial_guess() {
    // The property that makes the stage worth running at all, and the one
    // that survives a change of solver. A solver that cannot improve on
    // the guess has to hand the guess back rather than something worse.
    for count in [2, 4, 8, 16] {
        let reference = straight_reference(count);
        let result = optimizer(empty_field(), (None, None))
            .optimize(&reference, &FeasibilityPolicy::Unchecked)
            .unwrap();

        assert!(result.cost.is_finite());
        assert!(
            result.cost <= guess_cost(&reference) + 1e-9,
            "{count} waypoints: refining cost {} against a guess of {}",
            result.cost,
            guess_cost(&reference)
        );
    }
}

#[test]
fn refining_actually_improves_on_the_guess() {
    // Paired with the test above, which a solver that did nothing would
    // also pass.
    let reference = straight_reference(8);
    let refined = optimizer(empty_field(), (None, None))
        .optimize(&reference, &FeasibilityPolicy::Unchecked)
        .unwrap();
    assert!(
        refined.cost < guess_cost(&reference) * 0.9,
        "refining moved the cost from {} only to {}",
        guess_cost(&reference),
        refined.cost
    );
    assert!(refined.converged);
}

#[test]
fn the_endpoints_of_the_reference_are_held_fixed() {
    let reference = straight_reference(6);
    let result = optimizer(empty_field(), (None, None))
        .optimize(&reference, &FeasibilityPolicy::Unchecked)
        .unwrap();

    assert_eq!(result.states.first(), reference.first());
    assert_eq!(result.states.last(), reference.last());
    assert_eq!(result.states.len(), reference.len());
    assert_eq!(result.durations.len(), reference.len() - 1);
    assert_eq!(result.commands.len(), reference.len() - 1);
}

#[test]
fn every_duration_is_positive() {
    // The decision variable is the logarithm of the duration precisely so
    // this cannot fail, and it is asserted because an unconstrained
    // solver reaching a negative duration would make every implied speed
    // meaningless without ever returning an error.
    let reference = straight_reference(10);
    let result = optimizer(empty_field(), (None, None))
        .optimize(&reference, &FeasibilityPolicy::Unchecked)
        .unwrap();

    for duration in &result.durations {
        assert!(
            *duration > 0.0 && duration.is_finite(),
            "a segment lasts {duration} seconds"
        );
    }
}

#[test]
fn a_uniform_reference_is_timed_uniformly() {
    // A metamorphic property with a real oracle behind it: the reference
    // is symmetric under swapping any two segments, the cost is symmetric
    // in the same way, so the minimum has to be too. Unequal segment
    // times on an equal-length straight run would mean the solver stopped
    // somewhere that is not a minimum.
    let reference = straight_reference(8);
    let result = optimizer(empty_field(), (None, None))
        .optimize(&reference, &FeasibilityPolicy::Unchecked)
        .unwrap();

    let first = result.durations.first().copied().unwrap();
    for (index, duration) in result.durations.iter().enumerate() {
        assert!(
            (duration - first).abs() <= first * 1e-3,
            "segment {index} lasts {duration} against {first} for the first"
        );
    }
}

#[test]
fn a_speed_outside_the_band_is_reported_as_infeasible() {
    // The dynamics term is a penalty, and a penalty can be paid. The
    // check after the solve is what turns that into an answer a caller can
    // act on, which is why the result carries a flag rather than a log
    // line.
    let reference = straight_reference(5);
    let result = optimizer(empty_field(), (None, None))
        .optimize(
            &reference,
            &FeasibilityPolicy::Bounded {
                max_speed: Some(0.01),
                min_speed: None,
                max_turn_rate: None,
            },
        )
        .unwrap();
    assert!(!result.is_feasible);

    let generous = optimizer(empty_field(), (None, None))
        .optimize(
            &reference,
            &FeasibilityPolicy::Bounded {
                max_speed: Some(100.0),
                min_speed: Some(0.0),
                max_turn_rate: Some(100.0),
            },
        )
        .unwrap();
    assert!(generous.is_feasible);
}

#[test]
fn a_custom_feasibility_check_is_consulted() {
    let reference = straight_reference(4);
    let result = optimizer(empty_field(), (None, None))
        .optimize(
            &reference,
            &FeasibilityPolicy::Custom(Box::new(|state| state.speed < 0.0)),
        )
        .unwrap();
    assert!(!result.is_feasible);
}

#[test]
fn a_path_running_into_an_obstacle_is_pushed_away_from_it() {
    // The collision term is the only one that can move a waypoint off the
    // reference, so this is what proves it is wired in: the optimized
    // waypoint sits further from the obstacle than the reference did.
    let occupancy = KdTreeOccupancy::new(&[vec![4.0, 0.4]], 1.0).unwrap();
    let reference = straight_reference(5);
    let result = TrajectoryOptimizer::new(occupancy, terms((None, None)), settings())
        .unwrap()
        .optimize(&reference, &FeasibilityPolicy::Unchecked)
        .unwrap();

    let obstacle = (4.0_f64, 0.4_f64);
    let before = (reference[2][0] - obstacle.0).hypot(reference[2][1] - obstacle.1);
    let after = (result.states[2][0] - obstacle.0).hypot(result.states[2][1] - obstacle.1);
    assert!(
        after > before,
        "the waypoint stayed at {after} from the obstacle, against {before} before"
    );
}

#[test]
fn a_reference_too_short_to_optimize_is_rejected() {
    let optimizer = optimizer(empty_field(), (None, None));
    assert!(matches!(
        optimizer.optimize(&[], &FeasibilityPolicy::Unchecked),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        optimizer.optimize(&[vec![0.0, 0.0]], &FeasibilityPolicy::Unchecked),
        Err(Error::TooFew { .. })
    ));
}

#[test]
fn a_malformed_reference_is_rejected() {
    let optimizer = optimizer(empty_field(), (None, None));
    assert!(matches!(
        optimizer.optimize(&[vec![0.0, 0.0], vec![1.0]], &FeasibilityPolicy::Unchecked),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        optimizer.optimize(
            &[vec![0.0, 0.0], vec![f64::NAN, 1.0]],
            &FeasibilityPolicy::Unchecked
        ),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn a_degenerate_setting_is_rejected_at_construction() {
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let built = TrajectoryOptimizer::new(
            empty_field(),
            terms((None, None)),
            OptimizerSettings {
                cruise_speed: speed,
                ..OptimizerSettings::default()
            },
        );
        assert!(
            matches!(built, Err(Error::OutOfRange { .. })),
            "accepted a cruise speed of {speed}"
        );
    }
}

#[test]
fn a_single_segment_lands_on_the_cost_stationary_point() {
    // The one case with a closed-form oracle. With no interior waypoints
    // and nothing to avoid, the composite cost collapses to
    // `w_time (L/v)^2 + w_velocity (v - cruise)^2` in the single unknown
    // `v`, and the test finds that minimum by bisecting the derivative.
    // Nothing about the solver is assumed; the two have to agree.
    //
    // The answer sits well above the cruise speed, which is what a time
    // weight ten times the velocity weight is asking for. That ratio is
    // the Python default and it is kept.
    let length = 10.0_f64;
    let reference = vec![vec![0.0, 0.0], vec![length, 0.0]];
    let result = optimizer(empty_field(), (None, None))
        .optimize(&reference, &FeasibilityPolicy::Unchecked)
        .unwrap();

    assert_eq!(result.states, reference);
    assert_eq!(result.durations.len(), 1);

    let weights = TermWeights::default();
    let derivative = |speed: f64| {
        2.0f64.mul_add(
            weights.velocity * (speed - CRUISE_SPEED),
            -2.0 * weights.time * length * length / speed.powi(3),
        )
    };
    let mut low = 1e-6_f64;
    let mut high = 1e3_f64;
    for _ in 0..200 {
        let middle = f64::midpoint(low, high);
        if derivative(middle) < 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    let expected = f64::midpoint(low, high);

    let speed = length / result.durations[0];
    assert!(
        (speed - expected).abs() < 1e-4,
        "the solver timed the run at {speed}, the stationary point is {expected}"
    );
}

#[test]
fn the_solver_reports_whether_it_converged() {
    // FR-SAFE-02 for the optimizer: out of iterations and converged are
    // different answers.
    let reference = straight_reference(8);
    let starved = TrajectoryOptimizer::new(
        empty_field(),
        terms((None, None)),
        OptimizerSettings {
            cruise_speed: CRUISE_SPEED,
            max_iterations: 1,
            ..OptimizerSettings::default()
        },
    )
    .unwrap()
    .optimize(&reference, &FeasibilityPolicy::Unchecked)
    .unwrap();
    assert!(!starved.converged);
    assert!(starved.iterations <= 1);
}
