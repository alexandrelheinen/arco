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

//! The joint-space controller, against answers that do not come from it.
//!
//! Deviation A-02 says the port and the `CasADi` original reach different
//! solutions to the same problem, so a test that pins a solution vector
//! would be testing which solver ran. Every assertion here is either an
//! optimum that can be worked out by hand because the horizon is short
//! enough, a bound read back off the answer, or a cost compared against a
//! candidate the test built itself.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_control::joint::JointLimits;
use arco_control::limits::IntervalBand;
use arco_control::mpc::joint_space::{
    CONSTRAINT_TOLERANCE, HorizonPlan, JointMpcSettings, JointSpaceMpc, StepOutcome,
};
use arco_control::mpc::qp::SolveFailure;
use arco_core::Error;
use arco_mapping::occupancy::KdTreeOccupancy;

/// A controller with no map, which is the shape most of these want.
type Blind = JointSpaceMpc<KdTreeOccupancy>;

fn limits(axes: usize, velocity: f64, acceleration: f64) -> JointLimits {
    JointLimits::uniform(axes, velocity, acceleration).expect("the fixture limits are consistent")
}

/// Settings with the Python defaults, over `axes` identical axes.
fn settings(axes: usize) -> JointMpcSettings {
    JointMpcSettings::new(limits(axes, 1.0, 2.0))
}

fn controller(settings: JointMpcSettings) -> Blind {
    JointSpaceMpc::new(settings, None).expect("the fixture settings are consistent")
}

/// The cost of holding the velocity the machine arrived with.
///
/// Feasible whenever that velocity is already inside the box, since zero
/// acceleration satisfies every input bound and every predicted velocity
/// equals the one it started from. That makes it an upper bound on the
/// optimum, computed here rather than asked of the solver.
fn coasting_cost(
    settings: &JointMpcSettings,
    configuration: &[f64],
    velocity: &[f64],
    target: &[f64],
) -> f64 {
    let mut position: Vec<f64> = configuration.to_vec();
    let mut cost = 0.0;
    for step in 0..=settings.horizon_step_count {
        let tracking = if step == settings.horizon_step_count {
            2.0 * settings.weight_tracking
        } else {
            settings.weight_tracking
        };
        for (axis, &here) in position.iter().enumerate() {
            let goal = target.get(axis).copied().unwrap_or_default();
            let speed = velocity.get(axis).copied().unwrap_or_default();
            cost += tracking * (here - goal) * (here - goal);
            if step < settings.horizon_step_count {
                cost += settings.weight_velocity * speed * speed;
            }
        }
        for (axis, here) in position.iter_mut().enumerate() {
            let speed = velocity.get(axis).copied().unwrap_or_default();
            *here += speed * settings.step_interval;
        }
    }
    cost
}

/// The distance from `point` to the nearest predicted configuration.
fn closest_approach(plan: &HorizonPlan, point: &[f64]) -> f64 {
    let mut closest = f64::INFINITY;
    for step in 0..=plan.step_count() {
        let configuration = plan.configuration(step).expect("a step inside the plan");
        let distance = configuration
            .iter()
            .zip(point)
            .map(|(here, there)| (here - there) * (here - there))
            .sum::<f64>()
            .sqrt();
        closest = closest.min(distance);
    }
    closest
}

// ------------------------------------------------- closed-form optima ----

#[test]
fn a_one_step_horizon_commands_no_acceleration() {
    // With one step, the program has one free variable. The configuration
    // it ends on is pinned by the recurrence `q1 = q0 + v0 dt`, which the
    // acceleration does not appear in, so nothing the controller does
    // changes the tracking cost and the only term left is the effort. The
    // optimum is therefore exactly zero, and the cost is exactly the
    // tracking cost of standing still: 20 at the first state plus 40 at
    // the terminal one.
    let mut controller = controller(JointMpcSettings {
        horizon_step_count: 1,
        step_interval: 0.1,
        ..settings(1)
    });
    controller.reset(&[0.0]).expect("a valid reset");

    let step = controller.step(&[1.0], 0.1).expect("a valid step");
    assert!(step.outcome.solved(), "{:?}", step.outcome);
    let applied = step.acceleration.first().copied().unwrap_or_default();
    assert!(applied.abs() < 1e-6, "commanded {applied} rather than zero");

    let StepOutcome::Solved { cost, .. } = step.outcome else {
        panic!("a solved step carries a cost")
    };
    assert!(
        (cost - 60.0).abs() < 1e-6,
        "the objective came back as {cost}"
    );
}

#[test]
fn a_two_step_horizon_lands_where_the_calculus_says() {
    // Two steps, one axis, no velocity weight and no bound in the way. The
    // second input only pays effort, so it is zero, and the first is the
    // stationary point of `w_c a^2 + 2 w_t (a dt^2 - t)^2`, which is
    // `2 w_t dt^2 t / (w_c + 2 w_t dt^4)`.
    let interval = 0.1;
    let mut controller = controller(JointMpcSettings {
        horizon_step_count: 2,
        step_interval: interval,
        weight_tracking: 1.0,
        weight_velocity: 0.0,
        weight_control: 1.0,
        weight_obstacle: 0.0,
        ..JointMpcSettings::new(limits(1, 10.0, 10.0))
    });
    controller.reset(&[0.0]).expect("a valid reset");

    let step = controller.step(&[1.0], interval).expect("a valid step");
    assert!(step.outcome.solved(), "{:?}", step.outcome);

    let squared = interval * interval;
    let expected = 2.0 * squared / (1.0 + 2.0 * squared * squared);
    let applied = step.acceleration.first().copied().unwrap_or_default();
    assert!(
        (applied - expected).abs() < 1e-6,
        "commanded {applied} rather than {expected}"
    );

    let plan = step.plan.expect("a solved step carries its plan");
    let second = plan
        .acceleration(1)
        .and_then(|values| values.first().copied())
        .unwrap_or_default();
    assert!(
        second.abs() < 1e-6,
        "the second input pays effort and buys nothing, so it should be zero, not {second}"
    );
}

#[test]
fn an_acceleration_bound_that_binds_holds_on_its_boundary() {
    // The same problem with the bound moved below the unconstrained
    // optimum. The objective is convex in one variable, so the answer sits
    // exactly on the bound rather than anywhere near it.
    let interval = 0.1;
    let bound = 0.005;
    let mut controller = controller(JointMpcSettings {
        horizon_step_count: 2,
        step_interval: interval,
        weight_tracking: 1.0,
        weight_velocity: 0.0,
        weight_control: 1.0,
        weight_obstacle: 0.0,
        ..JointMpcSettings::new(limits(1, 10.0, bound))
    });
    controller.reset(&[0.0]).expect("a valid reset");

    let step = controller.step(&[1.0], interval).expect("a valid step");
    let applied = step.acceleration.first().copied().unwrap_or_default();
    assert!(
        (applied - bound).abs() < 1e-6,
        "commanded {applied} rather than sitting on the bound at {bound}"
    );
}

// ------------------------------------------------------ FR-MPC-03 ----

#[test]
fn every_bound_holds_on_the_plan_that_comes_back() {
    // Read off the answer rather than trusted of the solver, which is what
    // FR-MPC-03 asks: the initial state, both recurrences, the velocity
    // bound on every predicted step and the acceleration bound on every
    // input. The axes carry different limits so that a check reading the
    // wrong one cannot pass by accident.
    let velocities = vec![1.0, 0.25, 4.0];
    let accelerations = vec![2.0, 0.5, 8.0];
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        ..JointMpcSettings::new(
            JointLimits::new(velocities.clone(), accelerations.clone()).expect("valid limits"),
        )
    });
    controller.reset(&[0.0, 0.0, 0.0]).expect("a valid reset");

    for round in 0..40_i32 {
        let target = if round % 20 < 10 {
            [10.0, -10.0, 10.0]
        } else {
            [-10.0, 10.0, -10.0]
        };
        let before = controller.configuration().to_vec();
        let moving = controller.velocity().to_vec();
        let step = controller.step(&target, interval).expect("a valid step");
        assert!(step.outcome.solved(), "round {round}: {:?}", step.outcome);
        let plan = step.plan.expect("a solved step carries its plan");

        for (axis, (&start, &speed)) in before.iter().zip(&moving).enumerate() {
            let planned = plan.configuration(0).expect("the plan starts somewhere");
            let launched = plan.velocity(0).expect("the plan starts somewhere");
            assert!(
                (planned.get(axis).copied().unwrap_or_default() - start).abs()
                    < CONSTRAINT_TOLERANCE,
                "the plan does not start where the machine is on axis {axis}"
            );
            assert!(
                (launched.get(axis).copied().unwrap_or_default() - speed).abs()
                    < CONSTRAINT_TOLERANCE,
                "the plan does not start at the velocity the machine has on axis {axis}"
            );
        }

        for index in 0..plan.step_count() {
            let configuration = plan.configuration(index).expect("inside the plan");
            let velocity = plan.velocity(index).expect("inside the plan");
            let acceleration = plan.acceleration(index).expect("inside the plan");
            let next_configuration = plan.configuration(index + 1).expect("inside the plan");
            let next_velocity = plan.velocity(index + 1).expect("inside the plan");

            for axis in 0..3 {
                let speed = next_velocity[axis];
                assert!(
                    speed.abs() <= velocities[axis] + CONSTRAINT_TOLERANCE,
                    "step {index} axis {axis} predicts {speed}, past its velocity limit"
                );
                let rate = acceleration[axis];
                assert!(
                    rate.abs() <= accelerations[axis] + CONSTRAINT_TOLERANCE,
                    "step {index} axis {axis} commands {rate}, past its acceleration limit"
                );
                let predicted = configuration[axis] + velocity[axis] * interval;
                assert!(
                    (next_configuration[axis] - predicted).abs() < CONSTRAINT_TOLERANCE,
                    "step {index} axis {axis} does not integrate its own velocity"
                );
                let accelerated = velocity[axis] + rate * interval;
                assert!(
                    (speed - accelerated).abs() < CONSTRAINT_TOLERANCE,
                    "step {index} axis {axis} does not integrate its own acceleration"
                );
            }
        }
    }
}

#[test]
fn the_applied_state_is_the_second_point_of_its_own_plan() {
    // The conditioning on the way out is a guard rather than a policy, so
    // on a solved step it must not move the machine off the trajectory the
    // program predicted.
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        ..settings(2)
    });
    controller.reset(&[0.1, -0.2]).expect("a valid reset");

    for _ in 0..25_i32 {
        let step = controller
            .step(&[1.0, 1.0], interval)
            .expect("a valid step");
        let plan = step.plan.clone().expect("a solved step carries its plan");
        let configuration = plan.configuration(1).expect("inside the plan");
        let velocity = plan.velocity(1).expect("inside the plan");
        for axis in 0..2 {
            assert!(
                (step.configuration[axis] - configuration[axis]).abs() < CONSTRAINT_TOLERANCE,
                "axis {axis} was applied off its own plan"
            );
            assert!(
                (step.velocity[axis] - velocity[axis]).abs() < CONSTRAINT_TOLERANCE,
                "axis {axis} left at a velocity its plan did not predict"
            );
        }
        assert_eq!(step.velocity_saturated, 0);
        assert_eq!(step.acceleration_saturated, 0);
    }
}

#[test]
fn no_solution_costs_more_than_holding_the_velocity_it_arrived_with() {
    // Coasting is feasible from any state already inside the box, so its
    // cost is an upper bound on the optimum. The oracle is computed here
    // from the weights rather than read back from the solver.
    let interval = 0.05;
    let settings = JointMpcSettings {
        step_interval: interval,
        weight_obstacle: 0.0,
        ..settings(3)
    };
    let mut controller = controller(settings.clone());
    let start = [0.2, -0.4, 0.1];
    let moving = [0.3, 0.0, -0.2];
    controller
        .reset_with_velocity(&start, &moving)
        .expect("a valid reset");

    let target = [1.0, 1.0, -1.0];
    let candidate = coasting_cost(&settings, &start, &moving, &target);
    let step = controller.step(&target, interval).expect("a valid step");
    let StepOutcome::Solved { cost, .. } = step.outcome else {
        panic!("this state is inside the box, so the program is feasible")
    };
    assert!(
        cost <= candidate + 1e-6,
        "the optimum came back at {cost}, above the {candidate} of coasting"
    );
    assert!(cost > 0.0, "a target this far away cannot be free");
}

// --------------------------------------------- FR-MPC-04, FR-SAFE-02 ----

#[test]
fn arriving_faster_than_an_axis_allows_brakes_and_names_the_reason() {
    // FR-MPC-04. Every predicted velocity is bounded and no acceleration
    // inside its own bound can bring this one back within a step, so the
    // feasible set is empty. The answer is a brake at the acceleration
    // limit, the reason that says relaxing something is the only way
    // forward, and no plan.
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        ..settings(1)
    });
    controller
        .reset_with_velocity(&[0.0], &[50.0])
        .expect("a valid reset");

    let step = controller.step(&[0.0], interval).expect("a valid step");
    assert_eq!(
        step.outcome,
        StepOutcome::SafeStop(SolveFailure::Infeasible)
    );
    assert_eq!(step.outcome.failure(), Some(SolveFailure::Infeasible));
    assert!(!step.outcome.solved());
    assert!(step.plan.is_none(), "a failed solve predicts nothing");

    let braked = 50.0 - 2.0 * interval;
    assert!(
        (step.velocity[0] - braked).abs() < 1e-12,
        "{:?}",
        step.velocity
    );
    assert!(
        (step.acceleration[0] + 2.0).abs() < 1e-12,
        "{:?}",
        step.acceleration
    );
    assert!(
        (step.configuration[0] - braked * interval).abs() < 1e-12,
        "the configuration jumped: {:?}",
        step.configuration
    );
    assert_eq!(step.acceleration_saturated, 1);
}

#[test]
fn a_starved_budget_is_reported_as_exhausted_rather_than_solved() {
    // FR-SAFE-02 reaches this controller through the solver: out of budget
    // and infeasible are different answers and only one of them is worth
    // trying again. The state here is perfectly ordinary, so nothing but
    // the budget is wrong.
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        max_solver_iterations: 1,
        ..settings(3)
    });
    controller.reset(&[0.0, 0.0, 0.0]).expect("a valid reset");

    let step = controller
        .step(&[1.0, 1.0, 1.0], interval)
        .expect("a valid step");
    assert_eq!(
        step.outcome,
        StepOutcome::SafeStop(SolveFailure::BudgetExhausted)
    );
    assert!(step.plan.is_none());
}

#[test]
fn the_same_step_solves_once_the_budget_allows_it() {
    // The other half of the distinction: nothing was wrong with the
    // problem, only with how long it was given.
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        max_solver_iterations: 200,
        ..settings(3)
    });
    controller.reset(&[0.0, 0.0, 0.0]).expect("a valid reset");

    let step = controller
        .step(&[1.0, 1.0, 1.0], interval)
        .expect("a valid step");
    let StepOutcome::Solved {
        iterations, exact, ..
    } = step.outcome
    else {
        panic!(
            "the budget is the only thing that was wrong: {:?}",
            step.outcome
        )
    };
    assert!(iterations > 1, "it converged in {iterations} iterations");
    assert!(exact, "the answer should reach the tolerance it asked for");
}

#[test]
fn a_braked_axis_stops_at_zero_rather_than_reversing() {
    // The Python applied the full acceleration limit whatever the velocity
    // was, so an axis nearly stopped was driven through zero and left
    // reversing, then reversed again on the next step. The budget is
    // starved here so that every step takes the braked path.
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        max_solver_iterations: 1,
        ..settings(1)
    });
    controller
        .reset_with_velocity(&[0.0], &[0.05])
        .expect("a valid reset");

    // The allowance is 2.0 * 0.05 = 0.1, more than the velocity, so the
    // axis stops exactly rather than overshooting by the difference.
    let first = controller.step(&[0.0], interval).expect("a valid step");
    assert!(!first.outcome.solved());
    assert!(
        first.velocity[0].abs() < 1e-15,
        "the axis reversed to {}",
        first.velocity[0]
    );
    assert_eq!(first.acceleration_saturated, 0);

    let second = controller.step(&[0.0], interval).expect("a valid step");
    assert!(
        second.velocity[0].abs() < 1e-15,
        "a stopped axis started moving again at {}",
        second.velocity[0]
    );
    assert!(second.acceleration[0].abs() < 1e-15);
}

// ------------------------------------------------------------ behavior ----

#[test]
fn the_controller_settles_on_its_target() {
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        ..settings(3)
    });
    controller.reset(&[0.0, 0.0, 0.0]).expect("a valid reset");

    let target = [1.0, -0.5, 0.25];
    for round in 0..400_i32 {
        let step = controller.step(&target, interval).expect("a valid step");
        assert!(step.outcome.solved(), "round {round}: {:?}", step.outcome);
    }
    for (axis, (&reached, &wanted)) in controller.configuration().iter().zip(&target).enumerate() {
        assert!(
            (reached - wanted).abs() < 1e-3,
            "axis {axis} stopped at {reached} rather than {wanted}"
        );
    }
    for (axis, &speed) in controller.velocity().iter().enumerate() {
        assert!(speed.abs() < 1e-3, "axis {axis} never stopped, at {speed}");
    }
}

#[test]
fn no_axis_is_ever_commanded_past_its_own_limits() {
    // Deviation A-09 in its joint-space form: one saturation and one rate
    // limiter on the way out, per axis, swept against a target that keeps
    // jumping to the far side of the space.
    let velocities = vec![1.0, 0.25];
    let accelerations = vec![2.0, 0.5];
    let interval = 0.05;
    let mut controller = controller(JointMpcSettings {
        step_interval: interval,
        ..JointMpcSettings::new(
            JointLimits::new(velocities.clone(), accelerations.clone()).expect("valid limits"),
        )
    });
    controller.reset(&[0.0, 0.0]).expect("a valid reset");

    let mut previous = vec![0.0, 0.0];
    for round in 0..120_i32 {
        let target = if round % 40 < 20 {
            [5.0, 5.0]
        } else {
            [-5.0, -5.0]
        };
        let step = controller.step(&target, interval).expect("a valid step");
        for axis in 0..2 {
            assert!(
                step.velocity[axis].abs() <= velocities[axis] + 1e-12,
                "axis {axis} left the box at {}",
                step.velocity[axis]
            );
            assert!(
                step.acceleration[axis].abs() <= accelerations[axis] + 1e-12,
                "axis {axis} was commanded {}",
                step.acceleration[axis]
            );
            let change = (step.velocity[axis] - previous[axis]).abs() / interval;
            assert!(
                change <= accelerations[axis] + 1e-9,
                "axis {axis} changed velocity at {change}"
            );
        }
        previous = step.velocity;
    }
}

#[test]
fn an_obstacle_keeps_the_prediction_further_from_it() {
    // The barrier is soft, so this is a comparison rather than a bound: the
    // same step planned with and without the map, and the map has to buy
    // clearance. Anchored on the obstacle center, which is what the
    // barrier is written against, per deviation A-16.
    let interval = 0.05;
    let obstacle = [0.5, 0.15];
    let field = KdTreeOccupancy::new(&[obstacle.to_vec()], 0.5).expect("a valid field");
    let settings = JointMpcSettings {
        step_interval: interval,
        ..settings(2)
    };

    let mut blind = controller(settings.clone());
    blind.reset(&[0.0, 0.0]).expect("a valid reset");
    let open = blind
        .step(&[1.0, 0.0], interval)
        .expect("a valid step")
        .plan
        .expect("a solved step carries its plan");

    let mut wary = JointSpaceMpc::new(settings, Some(field)).expect("valid settings");
    wary.reset(&[0.0, 0.0]).expect("a valid reset");
    let avoided = wary
        .step(&[1.0, 0.0], interval)
        .expect("a valid step")
        .plan
        .expect("a solved step carries its plan");

    let without = closest_approach(&open, &obstacle);
    let with = closest_approach(&avoided, &obstacle);
    assert!(
        with > without + 1e-6,
        "the barrier bought nothing: {with} against {without}"
    );
}

// ---------------------------------------------------------- rejections ----

#[test]
fn an_interval_outside_the_band_is_refused() {
    // FR-INV-10, deviation A-17. The step reads no clock, so this is the
    // only place a bad interval can be caught.
    let mut controller = controller(settings(2));
    controller.reset(&[0.0, 0.0]).expect("a valid reset");
    for value in [0.0, -0.05, 2.0] {
        assert!(
            matches!(
                controller.step(&[1.0, 1.0], value),
                Err(Error::OutOfRange { .. })
            ),
            "an interval of {value} was accepted"
        );
    }
    assert!(matches!(
        controller.step(&[1.0, 1.0], f64::NAN),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn an_interval_that_disagrees_with_the_model_step_is_refused() {
    // A model advancing by one interval inside a loop running at another
    // predicts a trajectory the machine never follows, which the Python
    // refused for the same reason.
    let mut controller = controller(JointMpcSettings {
        step_interval: 0.05,
        ..settings(1)
    });
    controller.reset(&[0.0]).expect("a valid reset");
    assert!(matches!(
        controller.step(&[1.0], 0.04),
        Err(Error::OutOfRange { .. })
    ));
    assert!(controller.step(&[1.0], 0.05).is_ok());
}

#[test]
fn a_target_of_the_wrong_width_or_carrying_a_nan_is_refused() {
    let mut controller = controller(settings(3));
    controller.reset(&[0.0, 0.0, 0.0]).expect("a valid reset");
    assert!(matches!(
        controller.step(&[1.0, 1.0], 0.05),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        controller.step(&[1.0, f64::NAN, 1.0], 0.05),
        Err(Error::NotFinite { .. })
    ));
    assert!(matches!(
        controller.reset(&[0.0, 0.0]),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        controller.reset_with_velocity(&[0.0, 0.0, 0.0], &[f64::INFINITY, 0.0, 0.0]),
        Err(Error::NotFinite { .. })
    ));
}

#[test]
fn settings_no_program_could_use_are_refused() {
    let base = settings(2);
    assert!(matches!(
        JointMpcSettings {
            horizon_step_count: 0,
            ..base.clone()
        }
        .validate(),
        Err(Error::TooFew { .. })
    ));
    assert!(matches!(
        JointMpcSettings {
            horizon_step_count: 100_000,
            ..base.clone()
        }
        .validate(),
        Err(Error::OutOfRange { .. })
    ));
    assert!(matches!(
        JointMpcSettings {
            max_solver_iterations: 0,
            ..base.clone()
        }
        .validate(),
        Err(Error::TooFew { .. })
    ));
    for weight in [-1.0, f64::NAN] {
        assert!(matches!(
            JointMpcSettings {
                weight_tracking: weight,
                ..base.clone()
            }
            .validate(),
            Err(Error::OutOfRange { .. })
        ));
        assert!(matches!(
            JointMpcSettings {
                weight_obstacle: weight,
                ..base.clone()
            }
            .validate(),
            Err(Error::OutOfRange { .. })
        ));
    }
    for interval in [0.0, -0.05, f64::NAN, 10.0] {
        assert!(
            matches!(
                JointMpcSettings {
                    step_interval: interval,
                    ..base.clone()
                }
                .validate(),
                Err(Error::OutOfRange { .. })
            ),
            "a model step of {interval} was accepted"
        );
    }
    assert!(matches!(
        JointMpcSettings {
            step_interval: 0.05,
            interval: IntervalBand::new(0.1, 0.2).expect("a valid band"),
            ..base
        }
        .validate(),
        Err(Error::OutOfRange { .. })
    ));
}

#[test]
fn an_occupancy_describing_another_space_is_refused() {
    let field = KdTreeOccupancy::new(&[vec![1.0, 0.0]], 0.5).expect("a valid field");
    assert!(matches!(
        JointSpaceMpc::new(settings(3), Some(field)),
        Err(Error::DimensionMismatch { .. })
    ));
}
