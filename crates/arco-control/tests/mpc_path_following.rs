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

//! The contouring controller, against answers that do not come from it.
//!
//! Deviation A-02 says the port and the `CasADi` original reach different
//! solutions to the same problem, and deviation A-08 says as much about
//! two convex solvers on one program, so a test that pins a solution
//! vector is testing which solver ran. Every assertion here is one of
//! four things: an optimum short enough to work out by hand, a bound read
//! back off the returned plan, a property of the closed loop over many
//! steps, or a comparison against a candidate the test built itself.
//!
//! Two of these exist because of bugs found in the joint-space sibling,
//! and both were silent: the solver reported a solution and the answer
//! was wrong. One was a constraint row opened and never written into,
//! which bounds nothing while looking exactly like a row that does; the
//! other was an off-by-one in a loop that checked the bounds. So the
//! bound tests below drive each limit until it binds and then read the
//! returned plan back against it, rather than asserting that a solve
//! succeeded.

// Not `#[test]` functions, so the allowances in clippy.toml do not reach
// them: a fixture that cannot be built, or an outcome a helper was handed
// and cannot read, is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]
#![expect(clippy::panic, reason = "test fixtures")]

use arco_control::limits::{CommandLimits, IntervalBand};
use arco_control::mpc::model::{VehicleState, unicycle_step};
use arco_control::mpc::path_following::{
    HorizonPlan, PathFollowingMpc, PathFollowingSettings, PathFollowingStep, StepFailure,
    StepOutcome,
};
use arco_control::mpc::qp::SolveFailure;
use arco_core::Error;
use arco_mapping::occupancy::KdTreeOccupancy;

/// How far outside a bound a returned plan may sit.
///
/// The solver's own feasibility tolerance, stated once in the joint-space
/// module and reused rather than restated, so that loosening the solver
/// cannot quietly loosen what `FR-MPC-03` is checked against.
use arco_control::mpc::joint_space::CONSTRAINT_TOLERANCE;

/// A controller with no map, which is the shape most of these want.
type Blind = PathFollowingMpc<KdTreeOccupancy>;

/// Limits wide enough that only the one under test can bind.
fn limits() -> CommandLimits {
    CommandLimits {
        max_speed: 2.0,
        min_speed: 0.0,
        max_turn_rate: 1.0,
        max_speed_rate: 1.5,
        max_turn_rate_change: 3.0,
        interval: IntervalBand::default(),
    }
}

/// Settings over a limit set, at a tenth of a second per step.
fn settings_for(limits: CommandLimits) -> PathFollowingSettings {
    PathFollowingSettings {
        horizon_step_count: 10,
        step_interval: 0.1,
        cruise_speed: 1.0,
        ..PathFollowingSettings::new(limits)
    }
}

/// Settings over the fixture limits.
fn settings() -> PathFollowingSettings {
    settings_for(limits())
}

/// A controller following the positive first axis for forty meters.
fn straight(settings: PathFollowingSettings) -> Blind {
    let mut controller =
        PathFollowingMpc::new(settings, None).expect("consistent fixture settings");
    controller
        .set_reference(&[(0.0, 0.0), (40.0, 0.0)])
        .expect("a two point reference");
    controller
}

/// A controller following an arbitrary polyline.
fn on_path(settings: PathFollowingSettings, waypoints: &[(f64, f64)]) -> Blind {
    let mut controller =
        PathFollowingMpc::new(settings, None).expect("consistent fixture settings");
    controller
        .set_reference(waypoints)
        .expect("a usable reference");
    controller
}

/// A pose on the first axis, moving along it.
fn moving(x: f64, speed: f64) -> VehicleState {
    VehicleState {
        x,
        y: 0.0,
        heading: 0.0,
        speed,
        turn_rate: 0.0,
    }
}

/// The plan of a step that has to have solved.
fn plan_of(step: &PathFollowingStep) -> &HorizonPlan {
    assert!(step.outcome.solved(), "{:?}", step.outcome);
    step.plan.as_ref().expect("a solved step carries a plan")
}

/// The linearization point of a step that has to have solved.
fn nominal_of(step: &PathFollowingStep) -> &HorizonPlan {
    assert!(step.outcome.solved(), "{:?}", step.outcome);
    step.linearization
        .as_ref()
        .expect("a solved step carries the point it expanded about")
}

/// The reported cost of a step that has to have solved.
fn cost_of(step: &PathFollowingStep) -> f64 {
    let StepOutcome::Solved { cost, .. } = step.outcome else {
        panic!("the step did not solve: {:?}", step.outcome)
    };
    cost
}

/// The surrogate objective of a plan on a reference along the first axis.
///
/// An independent statement of what the program minimizes, written out
/// rather than read back, and only valid on this one geometry: along the
/// first axis the reference point at arc length `s` is `(s, 0)` with
/// heading zero, so the contouring error is the second coordinate, the
/// lag error is the first coordinate minus the arc length, and the
/// heading error is the heading. All three are exactly affine there, so
/// the expansion the controller squares is the error itself and the two
/// objectives have to agree to the last few places.
fn axis_objective(settings: &PathFollowingSettings, plan: &HorizonPlan) -> f64 {
    let horizon = plan.step_count();
    let mut total = 0.0;
    for step in 0..=horizon {
        let state = plan.state(step).expect("a node inside the plan");
        let arc = plan.arc_length(step).expect("a node inside the plan");
        let terminal = step == horizon;
        let contour_weight = if terminal {
            settings.weight_terminal
        } else {
            settings.weight_contour
        };
        let heading_weight = if terminal {
            settings.weight_terminal
        } else {
            settings.weight_heading
        };
        let contour = (state.y.abs() - settings.contour_deadzone).max(0.0);
        let lag = state.x - arc;
        total += contour_weight * contour * contour;
        total += settings.weight_lag * lag * lag;
        total += heading_weight * state.heading * state.heading;

        if let (Some(input), Some(progress_speed)) = (plan.input(step), plan.progress_speed(step)) {
            total += settings.weight_control
                * input.acceleration.mul_add(
                    input.acceleration,
                    input.turn_rate_change * input.turn_rate_change,
                );
            total -= settings.weight_progress * progress_speed * settings.step_interval;
        }
    }
    total
}

#[test]
fn a_step_on_a_straight_reference_solves_and_drives_forward() {
    let mut controller = straight(settings());
    let step = controller
        .step(moving(0.0, 0.5), 0.1)
        .expect("a valid control step");
    let plan = plan_of(&step);
    assert_eq!(plan.step_count(), 10);
    assert!(
        step.command.speed > 0.5,
        "commanded {} rather than accelerating",
        step.command.speed
    );
    assert!(step.progress > 0.0, "progress stayed at {}", step.progress);
}

// ------------------------------------------------- closed-form optima ----

#[test]
fn a_one_step_horizon_lands_where_the_calculus_says() {
    // One step, a reference along the first axis, and every weight off
    // except the lag, the progress reward and the control effort. The
    // state at node one is fixed by the recurrence and the pin, so the
    // only free variables are the two inputs and the progress speed, and
    // neither input appears anywhere but its own effort term.
    //
    // What is left is one variable. With `C = x0 + v0 dt - s0` the cost
    // is `w_l (C - v_s dt)^2 - w_p v_s dt`, whose derivative vanishes at
    // `C - v_s dt = -w_p / (2 w_l)`. Here `C = 0.1`, `w_p = 1` and
    // `w_l = 4`, so the lag error lands at exactly `-0.125` and the
    // progress speed at `2.25` meters per second.
    let mut controller = straight(PathFollowingSettings {
        horizon_step_count: 1,
        step_interval: 0.1,
        cruise_speed: 3.0,
        weight_contour: 0.0,
        weight_heading: 0.0,
        weight_terminal: 0.0,
        weight_lag: 4.0,
        weight_progress: 1.0,
        weight_control: 0.1,
        weight_obstacle: 0.0,
        ..settings_for(CommandLimits {
            max_speed: 5.0,
            ..limits()
        })
    });

    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    let plan = plan_of(&step);
    let progress_speed = plan.progress_speed(0).expect("one stage");
    let input = plan.input(0).expect("one stage");
    let arc = plan.arc_length(1).expect("one node past the pin");
    let state = plan.state(1).expect("one node past the pin");

    assert!(
        (progress_speed - 2.25).abs() < 1e-5,
        "the progress speed came back as {progress_speed}"
    );
    assert!(
        (state.x - arc + 0.125).abs() < 1e-5,
        "the lag error came back as {}",
        state.x - arc
    );
    assert!(
        input.acceleration.abs() < 1e-5 && input.turn_rate_change.abs() < 1e-5,
        "the inputs came back as {input:?} rather than at rest"
    );
    // `w_l e^2 - w_p v_s dt` at that point, with nothing else weighing.
    let cost = cost_of(&step);
    assert!(
        (cost + 0.1625).abs() < 1e-5,
        "the objective came back as {cost}"
    );
}

#[test]
fn doubling_the_progress_reward_moves_the_optimum_by_its_own_scale() {
    // The same algebra with `w_p = 2` puts the optimum at
    // `(C + w_p / (2 w_l)) / dt = (0.1 + 0.25) / 0.1`. A reward applied
    // once per stage rather than once per horizon, or scaled by anything
    // other than the model step, lands somewhere else.
    let mut controller = straight(PathFollowingSettings {
        horizon_step_count: 1,
        step_interval: 0.1,
        cruise_speed: 6.0,
        weight_contour: 0.0,
        weight_heading: 0.0,
        weight_terminal: 0.0,
        weight_lag: 4.0,
        weight_progress: 2.0,
        weight_control: 0.1,
        weight_obstacle: 0.0,
        ..settings_for(CommandLimits {
            max_speed: 8.0,
            ..limits()
        })
    });

    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    let progress_speed = plan_of(&step).progress_speed(0).expect("one stage");
    assert!(
        (progress_speed - 3.5).abs() < 1e-5,
        "the progress speed came back as {progress_speed}"
    );
}

#[test]
fn the_reported_cost_is_the_objective_this_module_states() {
    // Every weight on, a lateral offset and a heading offset so that no
    // term is trivially zero, and a reference the three errors are
    // exactly affine along. The objective is then computable from the
    // returned plan alone, without knowing which point the program was
    // expanded about, and it has to agree with what the step reported.
    //
    // This is the one check that can see a wrong coefficient in a cost.
    // Halving a cross term or dropping the constant of an expansion moves
    // the optimum while every constraint still holds exactly, so a test
    // reading bounds back would pass on a program minimizing the wrong
    // thing.
    let settings = PathFollowingSettings {
        horizon_step_count: 8,
        step_interval: 0.1,
        cruise_speed: 1.0,
        weight_contour: 10.0,
        weight_heading: 2.0,
        weight_progress: 1.0,
        weight_lag: 4.0,
        weight_control: 0.1,
        weight_terminal: 20.0,
        weight_obstacle: 0.0,
        ..PathFollowingSettings::new(limits())
    };
    let mut controller = straight(settings.clone());

    let step = controller
        .step(
            VehicleState {
                x: 1.0,
                y: 0.3,
                heading: 0.15,
                speed: 0.8,
                turn_rate: 0.05,
            },
            0.1,
        )
        .expect("a valid control step");
    let reported = cost_of(&step);
    let recomputed = axis_objective(&settings, plan_of(&step));
    assert!(
        (reported - recomputed).abs() < 1e-6 * recomputed.abs().max(1.0),
        "the step reported {reported} where the objective is {recomputed}"
    );
}

// ------------------------------------------- constraints, read back ------

/// Checks every row the program states against the plan it returned.
///
/// `FR-MPC-03`, written here a second time and from the settings rather
/// than from the assembly, so that a row the controller forgot to write
/// is a failing assertion rather than a bound nobody notices is gone. The
/// equalities are checked in both directions: a row that landed in the
/// wrong cone still holds one way round.
fn assert_every_bound_holds(controller: &Blind, step: &PathFollowingStep) {
    let settings = controller.settings();
    let limits = settings.limits;
    let dt = settings.step_interval;
    let reference = controller.reference().expect("a reference is set");
    let total_length = reference.total_length();
    let plan = plan_of(step);
    let nominal = nominal_of(step);

    for stage in 0..plan.step_count() {
        let progress_speed = plan.progress_speed(stage).expect("a stage in the plan");
        let input = plan.input(stage).expect("a stage in the plan");
        let nominal_arc = nominal.arc_length(stage).expect("a stage in the nominal");
        let curvature = reference.curvature(nominal_arc);
        let cap = curvature.mul_add(curvature, 1e-6).sqrt();
        assert!(
            progress_speed >= -CONSTRAINT_TOLERANCE,
            "the progress speed ran backward at stage {stage}: {progress_speed}"
        );
        assert!(
            progress_speed <= limits.max_speed + CONSTRAINT_TOLERANCE,
            "the progress speed passed the speed limit at stage {stage}: {progress_speed}"
        );
        assert!(
            progress_speed <= settings.cruise_speed + CONSTRAINT_TOLERANCE,
            "the progress speed passed the cruise cap at stage {stage}: {progress_speed}"
        );
        assert!(
            cap * progress_speed <= limits.max_turn_rate + CONSTRAINT_TOLERANCE,
            "the progress speed passed the curve limit at stage {stage}: {progress_speed}"
        );
        assert!(
            input.acceleration.abs() <= limits.max_speed_rate + CONSTRAINT_TOLERANCE,
            "the acceleration passed its limit at stage {stage}: {input:?}"
        );
        assert!(
            input.turn_rate_change.abs() <= limits.max_turn_rate_change + CONSTRAINT_TOLERANCE,
            "the turn rate change passed its limit at stage {stage}: {input:?}"
        );

        let arc = plan.arc_length(stage).expect("a stage in the plan");
        let next = plan
            .arc_length(stage.saturating_add(1))
            .expect("a node past the stage");
        assert!(
            (next - progress_speed.mul_add(dt, arc)).abs() <= CONSTRAINT_TOLERANCE,
            "the progress law failed at stage {stage}: {arc} then {next}"
        );
    }

    for node in 1..=plan.step_count() {
        let state = plan.state(node).expect("a node in the plan");
        let arc = plan.arc_length(node).expect("a node in the plan");
        let nominal_state = nominal.state(node).expect("a node in the nominal");
        let nominal_arc = nominal.arc_length(node).expect("a node in the nominal");
        assert!(
            state.speed <= limits.max_speed + CONSTRAINT_TOLERANCE
                && state.speed >= limits.min_speed - CONSTRAINT_TOLERANCE,
            "the speed left its band at node {node}: {}",
            state.speed
        );
        assert!(
            state.turn_rate.abs() <= limits.max_turn_rate + CONSTRAINT_TOLERANCE,
            "the turn rate passed its limit at node {node}: {}",
            state.turn_rate
        );
        assert!(
            arc >= -CONSTRAINT_TOLERANCE && arc <= total_length + CONSTRAINT_TOLERANCE,
            "the path parameter left the path at node {node}: {arc}"
        );
        assert!(
            (state.heading - nominal_state.heading).abs()
                <= settings.trust_heading + CONSTRAINT_TOLERANCE,
            "the heading left the trust region at node {node}: {} against {}",
            state.heading,
            nominal_state.heading
        );
        assert!(
            (arc - nominal_arc).abs() <= settings.trust_arc_length + CONSTRAINT_TOLERANCE,
            "the path parameter left the trust region at node {node}: {arc} against {nominal_arc}"
        );
    }
}

/// Checks the first recurrence against the model rather than the program.
///
/// The affine model is taken about the nominal, and the nominal's first
/// node is the measured state, so at that one stage the tangent plane and
/// the model it came from agree exactly. That makes the nonlinear step an
/// independent oracle for the row the command is read out of.
fn assert_first_step_matches_the_model(controller: &Blind, step: &PathFollowingStep) {
    let plan = plan_of(step);
    let dt = controller.settings().step_interval;
    let from = plan.state(0).expect("the pinned node");
    let input = plan.input(0).expect("the first stage");
    let reached = plan.state(1).expect("the node after the pin");
    let predicted = unicycle_step(from, input, dt).expect("a valid model step");
    for (here, there) in reached.to_array().into_iter().zip(predicted.to_array()) {
        assert!(
            (here - there).abs() <= CONSTRAINT_TOLERANCE,
            "the first recurrence gave {reached:?} where the model gives {predicted:?}"
        );
    }
}

#[test]
fn the_acceleration_bound_binds_rather_than_being_written_and_forgotten() {
    // A row that is opened and never written into reads as zero against
    // its bound, which every point satisfies. Asking for more speed than
    // the limit allows and reading the answer back is what tells the two
    // apart: the acceleration has to arrive at the limit, not past it and
    // not comfortably inside it.
    let mut controller = straight(PathFollowingSettings {
        cruise_speed: 2.0,
        ..settings_for(CommandLimits {
            max_speed_rate: 0.2,
            ..limits()
        })
    });
    let step = controller
        .step(moving(0.0, 0.0), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let applied = plan_of(&step).input(0).expect("a first stage").acceleration;
    assert!(
        (applied - 0.2).abs() < 1e-4,
        "the acceleration came back as {applied} rather than at its limit"
    );
}

#[test]
fn the_speed_bound_binds_on_every_predicted_node() {
    let mut controller = straight(PathFollowingSettings {
        cruise_speed: 2.0,
        ..settings_for(CommandLimits {
            max_speed: 0.4,
            ..limits()
        })
    });
    let step = controller
        .step(moving(0.0, 0.4), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let plan = plan_of(&step);
    let fastest = (1..=plan.step_count())
        .filter_map(|node| plan.state(node))
        .fold(f64::NEG_INFINITY, |held, state| held.max(state.speed));
    assert!(
        (fastest - 0.4).abs() < 1e-3,
        "the fastest predicted speed was {fastest} against a limit of 0.4"
    );
}

#[test]
fn the_minimum_speed_bound_binds_when_the_path_parameter_cannot_keep_up() {
    // With the cruise cap near zero the path parameter barely advances,
    // so every meter the vehicle covers costs lag. The cheapest answer is
    // to stop, and the floor on the commanded speed is what stops it
    // stopping.
    let mut controller = straight(PathFollowingSettings {
        cruise_speed: 0.05,
        weight_lag: 10.0,
        weight_progress: 0.1,
        ..settings_for(CommandLimits {
            min_speed: 0.4,
            ..limits()
        })
    });
    let step = controller
        .step(moving(0.0, 1.2), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let plan = plan_of(&step);
    let slowest = (1..=plan.step_count())
        .filter_map(|node| plan.state(node))
        .fold(f64::INFINITY, |held, state| held.min(state.speed));
    assert!(
        (slowest - 0.4).abs() < 1e-3,
        "the slowest predicted speed was {slowest} against a floor of 0.4"
    );
}

#[test]
fn the_turn_rate_bound_binds_when_the_vehicle_is_wide_of_the_path() {
    let mut controller = straight(PathFollowingSettings {
        weight_contour: 40.0,
        ..settings_for(CommandLimits {
            max_turn_rate: 0.2,
            ..limits()
        })
    });
    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 1.0,
                heading: 0.0,
                speed: 1.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let plan = plan_of(&step);
    let sharpest = (1..=plan.step_count())
        .filter_map(|node| plan.state(node))
        .fold(0.0_f64, |held, state| held.max(state.turn_rate.abs()));
    assert!(
        (sharpest - 0.2).abs() < 1e-3,
        "the sharpest predicted turn rate was {sharpest} against a limit of 0.2"
    );
}

#[test]
fn the_turn_rate_derivative_bound_binds_on_the_first_stage() {
    let mut controller = straight(PathFollowingSettings {
        weight_contour: 40.0,
        ..settings_for(CommandLimits {
            max_turn_rate: 2.0,
            max_turn_rate_change: 0.5,
            ..limits()
        })
    });
    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 1.0,
                heading: 0.0,
                speed: 1.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let applied = plan_of(&step)
        .input(0)
        .expect("a first stage")
        .turn_rate_change;
    assert!(
        (applied.abs() - 0.5).abs() < 1e-3,
        "the turn rate change came back as {applied} rather than at its limit"
    );
}

#[test]
fn the_cruise_cap_binds_the_progress_speed() {
    let mut controller = straight(PathFollowingSettings {
        cruise_speed: 0.3,
        ..settings()
    });
    let step = controller
        .step(moving(0.0, 0.3), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let progress_speed = plan_of(&step).progress_speed(0).expect("a first stage");
    assert!(
        (progress_speed - 0.3).abs() < 1e-4,
        "the progress speed came back as {progress_speed} against a cap of 0.3"
    );
}

#[test]
fn the_curve_limit_binds_the_progress_speed_even_on_a_straight() {
    // The curve-limited cap is `v_s sqrt(K^2 + eps) <= max_turn_rate`,
    // and on a straight the reference reports a curvature of exactly
    // zero. Written without the smoothing under the root the coefficient
    // would be zero as well, the sparse builder would drop it, and what
    // is left is a row with no entries that bounds nothing. With the
    // smoothing the coefficient is a thousandth, so a turn-rate limit of
    // a ten-thousandth caps the progress speed at exactly a tenth of a
    // meter per second, well under both the cruise cap and the speed
    // limit.
    let mut controller = straight(PathFollowingSettings {
        cruise_speed: 5.0,
        ..settings_for(CommandLimits {
            max_speed: 5.0,
            max_turn_rate: 1e-4,
            ..limits()
        })
    });
    let step = controller
        .step(moving(0.0, 0.1), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let progress_speed = plan_of(&step).progress_speed(0).expect("a first stage");
    assert!(
        (progress_speed - 0.1).abs() < 1e-5,
        "the progress speed came back as {progress_speed} against a curve cap of 0.1"
    );
}

#[test]
fn the_path_parameter_stops_at_the_end_of_the_path() {
    // The runway makes the path longer than the waypoints asked for, and
    // the bound is against the whole of it. Driving at the end with more
    // horizon than path left is what puts the last nodes against that
    // bound rather than past it.
    let mut controller = on_path(
        PathFollowingSettings {
            cruise_speed: 2.0,
            ..settings_for(CommandLimits {
                max_speed: 2.0,
                ..limits()
            })
        },
        &[(0.0, 0.0), (5.0, 0.0)],
    );
    let total_length = controller
        .reference()
        .expect("a reference is set")
        .total_length();

    let step = controller
        .step(moving(total_length - 0.5, 2.0), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let plan = plan_of(&step);
    let furthest = plan
        .arc_length(plan.step_count())
        .expect("the terminal node");
    assert!(
        (furthest - total_length).abs() < 1e-3,
        "the path parameter ended at {furthest} against a path of {total_length}"
    );
}

#[test]
fn the_heading_trust_region_holds_the_step_it_was_sized_for() {
    // The affine model is written in absolute variables, so nothing but
    // this stops a solve placing the answer where the tangent plane no
    // longer describes the model. Squeezing the radius below every other
    // limit is what makes it the binding one.
    let mut controller = straight(PathFollowingSettings {
        weight_contour: 40.0,
        trust_heading: 0.02,
        ..settings_for(CommandLimits {
            max_turn_rate: 2.0,
            ..limits()
        })
    });
    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 1.0,
                heading: 0.0,
                speed: 1.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let (plan, nominal) = (plan_of(&step), nominal_of(&step));
    let furthest = (1..=plan.step_count())
        .filter_map(|node| Some((plan.state(node)?, nominal.state(node)?)))
        .fold(0.0_f64, |held, (here, there)| {
            held.max((here.heading - there.heading).abs())
        });
    assert!(
        (furthest - 0.02).abs() < 1e-4,
        "the heading moved {furthest} from the nominal against a radius of 0.02"
    );
}

#[test]
fn the_arc_length_trust_region_holds_the_step_it_was_sized_for() {
    let mut controller = straight(PathFollowingSettings {
        cruise_speed: 2.0,
        trust_arc_length: 0.02,
        ..settings_for(CommandLimits {
            max_speed: 2.0,
            ..limits()
        })
    });
    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let (plan, nominal) = (plan_of(&step), nominal_of(&step));
    let furthest = (1..=plan.step_count())
        .filter_map(|node| Some((plan.arc_length(node)?, nominal.arc_length(node)?)))
        .fold(0.0_f64, |held, (here, there)| {
            held.max((here - there).abs())
        });
    assert!(
        (furthest - 0.02).abs() < 1e-4,
        "the path parameter moved {furthest} from the nominal against a radius of 0.02"
    );
}

#[test]
fn the_first_recurrence_reproduces_the_nonlinear_model_exactly() {
    let mut controller = straight(settings());
    let step = controller
        .step(
            VehicleState {
                x: 0.5,
                y: 0.2,
                heading: -0.1,
                speed: 0.9,
                turn_rate: 0.2,
            },
            0.1,
        )
        .expect("a valid control step");
    assert_first_step_matches_the_model(&controller, &step);
}

#[test]
fn a_bound_nothing_is_pushing_against_stays_strictly_inside_it() {
    // The cone list is split at a row index rather than by asking what a
    // row means, so an inequality written among the equalities becomes a
    // pin: the row `a <= limit` would hold as `a = limit` and the solver
    // would report it solved. A vehicle already tracking the path wants
    // neither acceleration nor steering, so every one of these sitting
    // well inside its bound is what says the split landed where it was
    // meant to.
    let mut controller = straight(settings());
    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let plan = plan_of(&step);
    let input = plan.input(0).expect("a first stage");
    let state = plan.state(1).expect("the node after the pin");
    assert!(
        input.acceleration.abs() < 0.5 * limits().max_speed_rate,
        "the acceleration sat at {} with nothing asking for it",
        input.acceleration
    );
    assert!(
        input.turn_rate_change.abs() < 0.5 * limits().max_turn_rate_change,
        "the turn rate change sat at {} with nothing asking for it",
        input.turn_rate_change
    );
    assert!(
        state.speed > limits().min_speed + 0.1 && state.speed < limits().max_speed - 0.1,
        "the speed sat at {} rather than inside its band",
        state.speed
    );
}

#[test]
fn the_reported_convergence_is_the_distance_between_the_last_two_iterates() {
    // The sequential loop stops when the linearization point stops
    // moving, so the number it reports has to be the distance it
    // measured. Recomputing it from the two plans the step carries is
    // what keeps the stopping rule checkable from outside.
    let mut controller = straight(PathFollowingSettings {
        max_sqp_iterations: 4,
        ..settings()
    });
    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 0.4,
                heading: 0.2,
                speed: 0.6,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("a valid control step");
    let StepOutcome::Solved { convergence, .. } = step.outcome else {
        panic!("the step did not solve: {:?}", step.outcome)
    };

    let (plan, nominal) = (plan_of(&step), nominal_of(&step));
    let mut measured = 0.0_f64;
    for node in 0..=plan.step_count() {
        let (here, there) = (
            plan.state(node).expect("a node in the plan"),
            nominal.state(node).expect("a node in the nominal"),
        );
        measured = measured
            .max((here.x - there.x).hypot(here.y - there.y))
            .max((here.heading - there.heading).abs())
            .max(
                (plan.arc_length(node).unwrap_or_default()
                    - nominal.arc_length(node).unwrap_or_default())
                .abs(),
            );
    }
    assert!(
        (convergence - measured).abs() < 1e-9,
        "the step reported {convergence} against a measured {measured}"
    );
}

// ------------------------------------------------ failures and limits ----

#[test]
fn a_measured_state_that_is_not_finite_brakes_rather_than_raising() {
    // `FR-MPC-04`. A dropped sensor reading is a runtime condition the
    // controller has to survive with a command, not an argument the
    // caller chose wrongly, so it comes back as an outcome. The Python
    // answered the same condition with the same deceleration.
    let mut controller = straight(settings());
    let step = controller
        .step(
            VehicleState {
                x: f64::NAN,
                y: 0.0,
                heading: 0.0,
                speed: 0.5,
                turn_rate: 0.2,
            },
            0.1,
        )
        .expect("an invalid state is not an invalid call");

    assert_eq!(
        step.outcome,
        StepOutcome::SafeStop(StepFailure::InvalidState)
    );
    assert!(step.plan.is_none(), "a failed step carried a plan");
    assert!(
        (step.command.speed - 0.35).abs() < 1e-12,
        "the brake commanded {} rather than one step of deceleration",
        step.command.speed
    );
    assert!(
        (step.command.turn_rate - 0.2).abs() < 1e-12,
        "the brake commanded {} rather than holding the turn rate",
        step.command.turn_rate
    );
}

#[test]
fn a_speed_reading_that_is_not_finite_still_leaves_a_finite_command() {
    let mut controller = straight(settings());
    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 0.0,
                heading: 0.0,
                speed: f64::NAN,
                turn_rate: f64::NAN,
            },
            0.1,
        )
        .expect("an invalid state is not an invalid call");

    assert!(!step.outcome.solved(), "{:?}", step.outcome);
    assert!(
        step.command.speed.is_finite() && step.command.turn_rate.is_finite(),
        "the brake commanded {:?}",
        step.command
    );
    assert!(
        (step.command.speed - limits().min_speed).abs() < 1e-12,
        "the brake commanded {} rather than the slowest it may",
        step.command.speed
    );
}

#[test]
fn arriving_faster_than_one_step_of_braking_can_fix_reports_an_empty_feasible_set() {
    // The speed bound holds on every predicted node and not on the pinned
    // one, so arriving above it is reported rather than accepted, and it
    // is reported as what it is: no input inside its own limit brings the
    // next node back inside the box. The same honest failure the
    // joint-space controller returns for an axis arriving too fast.
    let mut controller = straight(settings());
    let step = controller
        .step(moving(0.0, 3.0), 0.1)
        .expect("an infeasible program is not an invalid call");

    assert_eq!(
        step.outcome,
        StepOutcome::SafeStop(StepFailure::Solve(SolveFailure::Infeasible))
    );
    assert!(
        step.command.speed < 3.0,
        "the brake commanded {} rather than slowing down",
        step.command.speed
    );
    assert!(
        step.saturation.magnitude_steps > 0,
        "the command left the limit box without the limiter saying so"
    );
}

#[test]
fn a_step_that_failed_leaves_the_progress_where_it_was() {
    let mut controller = straight(settings());
    let good = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    assert!(good.progress > 0.0, "the good step made no progress");

    let failed = controller
        .step(
            VehicleState {
                x: f64::INFINITY,
                ..moving(0.0, 1.0)
            },
            0.1,
        )
        .expect("an invalid state is not an invalid call");
    assert!(
        (failed.progress - good.progress).abs() < 1e-12,
        "progress moved from {} to {} on a failed step",
        good.progress,
        failed.progress
    );
}

#[test]
fn a_step_before_a_reference_is_set_is_the_caller_being_wrong() {
    let mut controller: Blind =
        PathFollowingMpc::new(settings(), None).expect("consistent fixture settings");
    let outcome = controller.step(moving(0.0, 1.0), 0.1);
    assert!(
        matches!(
            outcome,
            Err(Error::TooFew {
                quantity: "reference waypoints",
                ..
            })
        ),
        "stepping without a reference gave {outcome:?}"
    );
}

#[test]
fn an_interval_outside_the_band_is_refused() {
    // `FR-INV-10` and deviation A-17. The step reads no clock, so the
    // interval is whatever the caller computed, including a negative one
    // after a clock adjustment.
    let mut controller = straight(settings());
    for interval in [-0.1, 0.0, 2.0] {
        let outcome = controller.step(moving(0.0, 1.0), interval);
        assert!(
            outcome.is_err(),
            "an interval of {interval} was accepted: {outcome:?}"
        );
    }
}

#[test]
fn an_interval_that_disagrees_with_the_model_step_is_refused() {
    // The Python discarded this argument outright. A loop running at one
    // rate against a model discretized at another predicts a trajectory
    // the vehicle never follows and spends every step chasing the
    // difference, which is why the joint-space controller refuses it and
    // why this one now does too.
    let mut controller = straight(settings());
    let outcome = controller.step(moving(0.0, 1.0), 0.05);
    assert!(
        matches!(
            outcome,
            Err(Error::OutOfRange {
                quantity: "elapsed interval",
                ..
            })
        ),
        "a mismatched interval gave {outcome:?}"
    );
}

#[test]
fn a_zero_lag_weight_is_refused_at_construction() {
    // The lag error is the only term coupling the path parameter to the
    // vehicle, so without it the contouring errors are measured against a
    // point free to sit anywhere on the path. The Python raised on the
    // same value for the same reason.
    let outcome = PathFollowingSettings {
        weight_lag: 0.0,
        ..settings()
    }
    .validate();
    assert!(
        matches!(
            outcome,
            Err(Error::OutOfRange {
                quantity: "lag weight",
                ..
            })
        ),
        "a zero lag weight gave {outcome:?}"
    );
}

#[test]
fn a_negative_weight_is_refused_rather_than_clamped() {
    let outcome = PathFollowingSettings {
        weight_contour: -1.0,
        ..settings()
    }
    .validate();
    assert!(
        matches!(
            outcome,
            Err(Error::OutOfRange {
                quantity: "contour weight",
                ..
            })
        ),
        "a negative contour weight gave {outcome:?}"
    );
}

#[test]
fn a_limit_that_is_not_finite_is_refused() {
    // Every bound becomes the right-hand side of a row, and the solver
    // wrapper refuses an infinity there with much less to say about where
    // it came from.
    let outcome = settings_for(CommandLimits {
        max_turn_rate: f64::INFINITY,
        ..limits()
    })
    .validate();
    assert!(
        matches!(
            outcome,
            Err(Error::NotFinite {
                quantity: "maximum turn rate",
                ..
            })
        ),
        "an unbounded turn rate gave {outcome:?}"
    );
}

#[test]
fn an_empty_or_oversized_budget_is_refused() {
    for settings in [
        PathFollowingSettings {
            horizon_step_count: 0,
            ..settings()
        },
        PathFollowingSettings {
            max_sqp_iterations: 0,
            ..settings()
        },
        PathFollowingSettings {
            max_solver_iterations: 0,
            ..settings()
        },
        PathFollowingSettings {
            horizon_step_count: 10_000,
            ..settings()
        },
        PathFollowingSettings {
            max_sqp_iterations: 100,
            ..settings()
        },
    ] {
        assert!(
            settings.validate().is_err(),
            "a budget of {} steps and {} iterations was accepted",
            settings.horizon_step_count,
            settings.max_sqp_iterations
        );
    }
}

#[test]
fn a_model_step_outside_the_interval_band_is_refused() {
    let outcome = PathFollowingSettings {
        step_interval: 2.0,
        ..settings()
    }
    .validate();
    assert!(
        matches!(
            outcome,
            Err(Error::OutOfRange {
                quantity: "model step interval",
                ..
            })
        ),
        "a model step outside the band gave {outcome:?}"
    );
}

// ----------------------------------------------- the deadzone epigraph ---

#[test]
fn a_contour_deadzone_is_an_epigraph_rather_than_an_approximation() {
    // `max(|e| - d, 0)^2` is convex, so the slack and its two rows carry
    // the same function the original penalized. Recomputing the objective
    // from the plan with the deadzone written out is what says the slack
    // settled on the excess rather than on something near it.
    let settings = PathFollowingSettings {
        horizon_step_count: 8,
        step_interval: 0.1,
        cruise_speed: 1.0,
        contour_deadzone: 0.5,
        weight_contour: 10.0,
        weight_heading: 2.0,
        weight_terminal: 20.0,
        ..settings()
    };
    let mut controller = straight(settings.clone());

    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 0.9,
                heading: 0.0,
                speed: 0.8,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("a valid control step");
    assert_every_bound_holds(&controller, &step);

    let reported = cost_of(&step);
    let recomputed = axis_objective(&settings, plan_of(&step));
    assert!(
        (reported - recomputed).abs() < 1e-5 * recomputed.abs().max(1.0),
        "the step reported {reported} where the objective is {recomputed}"
    );
}

#[test]
fn a_contouring_error_inside_the_deadzone_costs_nothing() {
    // Two controllers on the same geometry, one with a band wider than
    // the error and one without a band at all. The first has no reason to
    // steer and the second does.
    let offset = VehicleState {
        x: 0.0,
        y: 0.3,
        heading: 0.0,
        speed: 1.0,
        turn_rate: 0.0,
    };
    let mut banded = straight(PathFollowingSettings {
        contour_deadzone: 1.0,
        ..settings()
    });
    let mut plain = straight(settings());

    let banded_step = banded.step(offset, 0.1).expect("a valid control step");
    let plain_step = plain.step(offset, 0.1).expect("a valid control step");
    let banded_turn = banded_step.command.turn_rate.abs();
    let plain_turn = plain_step.command.turn_rate.abs();

    assert!(
        banded_turn < 1e-6,
        "the banded controller steered at {banded_turn} inside its own deadzone"
    );
    assert!(
        plain_turn > 1e-3,
        "the plain controller steered at {plain_turn} with an error to correct"
    );
}

// --------------------------------------------------- the closed loop -----

/// Advances a vehicle exactly the way the prediction model does.
///
/// The position and the heading move on the state the step arrived with
/// and the speed and turn rate become the command, which is the same
/// recurrence the program wrote. A vehicle integrated any other way would
/// make a tracking test measure the difference between two integrators.
fn advance(state: VehicleState, command: arco_core::protocols::Command, dt: f64) -> VehicleState {
    let (sin_heading, cos_heading) = state.heading.sin_cos();
    VehicleState {
        x: (state.speed * cos_heading).mul_add(dt, state.x),
        y: (state.speed * sin_heading).mul_add(dt, state.y),
        heading: state.turn_rate.mul_add(dt, state.heading),
        speed: command.speed,
        turn_rate: command.turn_rate,
    }
}

/// Runs a controller against that vehicle, returning every step it took.
fn run(
    controller: &mut Blind,
    start: VehicleState,
    steps: usize,
) -> Vec<(VehicleState, PathFollowingStep)> {
    let dt = controller.settings().step_interval;
    let mut state = start;
    let mut history = Vec::with_capacity(steps);
    for _ in 0..steps {
        let step = controller.step(state, dt).expect("a valid control step");
        let command = step.command;
        history.push((state, step));
        state = advance(state, command, dt);
    }
    history
}

#[test]
fn the_controller_pulls_a_straight_reference_back_under_the_vehicle() {
    let mut controller = straight(settings());
    let history = run(
        &mut controller,
        VehicleState {
            x: 0.0,
            y: 0.5,
            heading: 0.0,
            speed: 0.5,
            turn_rate: 0.0,
        },
        80,
    );

    assert!(
        history.iter().all(|(_, step)| step.outcome.solved()),
        "a step on a straight reference failed to solve"
    );
    let settled = history
        .iter()
        .skip(40)
        .fold(0.0_f64, |held, (state, _)| held.max(state.y.abs()));
    assert!(
        settled < 0.05,
        "the lateral error settled at {settled} rather than on the path"
    );
}

#[test]
fn the_controller_rounds_a_right_angle_and_keeps_making_progress() {
    let mut controller = on_path(
        PathFollowingSettings {
            horizon_step_count: 15,
            ..settings()
        },
        &[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
    );
    let history = run(&mut controller, moving(0.0, 1.0), 200);

    let solved = history
        .iter()
        .filter(|(_, step)| step.outcome.solved())
        .count();
    assert!(solved >= 195, "only {solved} of 200 steps solved");
    let (last_state, last_step) = history.last().expect("a run of two hundred steps");
    assert!(
        last_step.progress > 12.0,
        "the vehicle reached {} meters of a fifteen meter corner",
        last_step.progress
    );
    assert!(
        last_state.y > 1.0,
        "the vehicle ended at y = {} rather than around the corner",
        last_state.y
    );
}

#[test]
fn progress_never_rewinds_even_pointed_the_wrong_way() {
    // The path parameter advances with its own non-negative speed and the
    // projection is only ever allowed to push it forward, so a recovery
    // arc has to catch up to the reference point rather than dragging it
    // back. Starting the vehicle pointed backward is the cheapest way to
    // ask for the opposite.
    let mut controller = straight(settings());
    let history = run(
        &mut controller,
        VehicleState {
            x: 5.0,
            y: 0.0,
            heading: core::f64::consts::PI,
            speed: 0.3,
            turn_rate: 0.0,
        },
        40,
    );

    let mut furthest = 0.0_f64;
    for (index, (_, step)) in history.iter().enumerate() {
        assert!(
            step.progress >= furthest - 1e-9,
            "progress fell from {furthest} to {} at step {index}",
            step.progress
        );
        furthest = furthest.max(step.progress);
    }
}

#[test]
fn a_run_to_the_end_of_the_path_never_runs_out_of_path() {
    // Without the runway the bound holding the path parameter inside the
    // path pinches against its own ceiling over the last meters and the
    // program reports an empty feasible set exactly where the run was
    // about to succeed.
    let mut controller = on_path(
        PathFollowingSettings {
            cruise_speed: 1.5,
            ..settings_for(CommandLimits {
                max_speed: 1.5,
                ..limits()
            })
        },
        &[(0.0, 0.0), (6.0, 0.0)],
    );
    let history = run(&mut controller, moving(0.0, 1.0), 60);

    for (index, (state, step)) in history.iter().enumerate() {
        assert!(
            step.outcome.solved(),
            "step {index} at x = {} failed: {:?}",
            state.x,
            step.outcome
        );
    }
    let (last_state, _) = history.last().expect("a run of sixty steps");
    assert!(
        last_state.x > 5.5,
        "the vehicle only reached {} of a six meter path",
        last_state.x
    );
}

#[test]
fn a_solved_step_commands_the_state_after_the_current_one() {
    // Not the first input: that is an acceleration and a turn-rate
    // change, and handing those to a vehicle expecting a velocity command
    // is a units error no type here would catch.
    let mut controller = straight(settings());
    let step = controller
        .step(moving(0.0, 0.5), 0.1)
        .expect("a valid control step");
    let commanded = plan_of(&step).state(1).expect("the node after the pin");
    assert!(
        (step.requested.speed - commanded.speed).abs() < 1e-12
            && (step.requested.turn_rate - commanded.turn_rate).abs() < 1e-12,
        "the step asked for {:?} where the plan predicts {commanded:?}",
        step.requested
    );
}

#[test]
fn a_solved_command_passes_the_limiters_untouched() {
    // Deviation A-09 puts one saturation and one rate limiter on the way
    // out. Neither can bite on a solved step, because the program already
    // bounds the speed and the turn rate of the state it commands and
    // bounds how far they moved from the measured ones. That is what
    // makes the limiters a guard rather than a second controller, and it
    // is worth asserting rather than assuming.
    let mut controller = straight(settings());
    let history = run(&mut controller, moving(0.0, 0.2), 60);

    assert!(
        history.iter().all(|(_, step)| step.outcome.solved()),
        "a step failed to solve"
    );
    let (_, last) = history.last().expect("a run of sixty steps");
    assert!(
        !last.saturation.saturated(),
        "the limiters clipped a solved command: {:?}",
        last.saturation
    );
}

// ------------------------------------------------ the sequential loop ----

/// How far a plan drifts from the model it claims to predict.
///
/// The program carries the tangent plane rather than the model, so the
/// two agree exactly at the point the plane was taken about and drift
/// with the square of the distance from it. Measuring that drift along
/// the answer is how a reader tells a converged iterate from a first
/// guess: both satisfy every row, and only one of them describes a
/// trajectory the vehicle could follow.
fn model_defect(controller: &Blind, plan: &HorizonPlan) -> f64 {
    let dt = controller.settings().step_interval;
    let mut worst = 0.0_f64;
    for stage in 0..plan.step_count() {
        let (Some(state), Some(input), Some(reached)) = (
            plan.state(stage),
            plan.input(stage),
            plan.state(stage.saturating_add(1)),
        ) else {
            continue;
        };
        let predicted = unicycle_step(state, input, dt).expect("a valid model step");
        worst = worst
            .max((reached.x - predicted.x).hypot(reached.y - predicted.y))
            .max((reached.heading - predicted.heading).abs());
    }
    worst
}

#[test]
fn another_sequential_iteration_brings_the_plan_closer_to_the_model() {
    // The point of the loop. One iteration returns the exact optimum of a
    // plane taken about a guess; further ones move the plane onto the
    // answer, and the distance between the plan and the model it stands
    // for is what shrinks.
    let corner = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0)];
    let start = VehicleState {
        x: 4.0,
        y: 0.6,
        heading: 0.3,
        speed: 1.2,
        turn_rate: 0.0,
    };

    let mut single = on_path(
        PathFollowingSettings {
            max_sqp_iterations: 1,
            horizon_step_count: 15,
            ..settings_for(CommandLimits {
                max_speed: 2.0,
                max_turn_rate: 1.5,
                ..limits()
            })
        },
        &corner,
    );
    let mut repeated = on_path(
        PathFollowingSettings {
            max_sqp_iterations: 6,
            horizon_step_count: 15,
            ..settings_for(CommandLimits {
                max_speed: 2.0,
                max_turn_rate: 1.5,
                ..limits()
            })
        },
        &corner,
    );

    let one = single.step(start, 0.1).expect("a valid control step");
    let many = repeated.step(start, 0.1).expect("a valid control step");
    let (loose, tight) = (
        model_defect(&single, plan_of(&one)),
        model_defect(&repeated, plan_of(&many)),
    );
    assert!(
        tight < loose,
        "six iterations left a defect of {tight} against one iteration's {loose}"
    );
}

#[test]
fn the_loop_stops_once_the_linearization_point_stops_moving() {
    let mut controller = straight(PathFollowingSettings {
        max_sqp_iterations: 6,
        ..settings()
    });
    // The first step has nothing to warm start from; by the third the
    // shifted plan is already close to the answer.
    let history = run(&mut controller, moving(0.0, 1.0), 4);
    let (_, last) = history.last().expect("a run of four steps");
    let StepOutcome::Solved {
        convergence,
        sqp_iterations,
        ..
    } = last.outcome
    else {
        panic!("the step did not solve: {:?}", last.outcome)
    };

    assert!(
        convergence <= controller.settings().sqp_tolerance,
        "the loop stopped {convergence} away from its own tolerance"
    );
    assert!(
        sqp_iterations < 6,
        "the loop used its whole budget of {sqp_iterations} on a settled straight"
    );
}

#[test]
fn a_vehicle_at_rest_off_the_path_still_steers_on_its_first_step() {
    // Linearized at rest the model says the heading column reaches
    // neither position row, so turning is free and moves nothing: the
    // first solve cannot steer if every stage is expanded about a stopped
    // vehicle. The rollout accelerates instead of standing still, which
    // is what puts a moving vehicle under every stage after the first.
    let mut controller = straight(settings());
    let step = controller
        .step(
            VehicleState {
                x: 0.0,
                y: 0.6,
                heading: 0.0,
                speed: 0.0,
                turn_rate: 0.0,
            },
            0.1,
        )
        .expect("a valid control step");

    assert!(
        step.command.turn_rate.abs() > 1e-3,
        "the first command from rest steered at {}",
        step.command.turn_rate
    );
    assert!(
        step.command.speed > 0.0,
        "the first command from rest stayed stopped at {}",
        step.command.speed
    );
}

#[test]
fn a_vehicle_parked_at_a_corner_gets_moving_again() {
    // The stopped equilibrium the Python's anti-stall reseed existed to
    // escape. Convexifying the inner problem does not remove it, because
    // it lives in the choice of linearization point rather than in the
    // program.
    let mut controller = on_path(
        PathFollowingSettings {
            horizon_step_count: 15,
            ..settings()
        },
        &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)],
    );
    let history = run(&mut controller, moving(3.6, 0.0), 60);
    let (_, last) = history.last().expect("a run of sixty steps");
    assert!(
        last.progress > 5.0,
        "the vehicle advanced {} meters from a standing start at a corner",
        last.progress
    );
}

// --------------------------------------------------- obstacle barriers ---

/// A controller that can see one obstacle.
fn seeing(settings: PathFollowingSettings, obstacle: (f64, f64), clearance: f64) -> Blind {
    let occupancy = KdTreeOccupancy::new(&[vec![obstacle.0, obstacle.1]], clearance)
        .expect("one obstacle point and a positive clearance");
    let mut controller =
        PathFollowingMpc::new(settings, Some(occupancy)).expect("consistent fixture settings");
    controller
        .set_reference(&[(0.0, 0.0), (20.0, 0.0)])
        .expect("a two point reference");
    controller
}

/// Settings with enough horizon to reach an obstacle a few meters out.
fn reaching() -> PathFollowingSettings {
    PathFollowingSettings {
        horizon_step_count: 20,
        cruise_speed: 2.0,
        ..settings_for(CommandLimits {
            max_speed: 2.0,
            ..limits()
        })
    }
}

/// The closest a plan comes to a point, meters.
fn closest_approach(plan: &HorizonPlan, obstacle: (f64, f64)) -> f64 {
    (0..=plan.step_count())
        .filter_map(|node| plan.state(node))
        .fold(f64::INFINITY, |held, state| {
            held.min((state.x - obstacle.0).hypot(state.y - obstacle.1))
        })
}

#[test]
fn an_obstacle_beside_the_path_bends_the_plan_away_from_it() {
    let obstacle = (3.0, 0.5);
    let start = moving(0.0, 2.0);
    let mut blind = straight(reaching());
    let mut sighted = seeing(reaching(), obstacle, 1.2);

    let blind_step = blind.step(start, 0.1).expect("a valid control step");
    let sighted_step = sighted.step(start, 0.1).expect("a valid control step");
    assert!(sighted_step.outcome.solved(), "{:?}", sighted_step.outcome);

    let (open, avoided) = (
        closest_approach(plan_of(&blind_step), obstacle),
        closest_approach(plan_of(&sighted_step), obstacle),
    );
    assert!(
        avoided > open + 0.05,
        "the barrier held the plan {avoided} from the obstacle against {open} without one"
    );
}

#[test]
fn the_cruise_preview_slows_the_vehicle_before_a_pinch_point() {
    // Sampled along the reference rather than around the vehicle, so the
    // progress cap comes down before the obstacle enters the horizon
    // rather than once it is already inside one.
    let start = moving(0.0, 2.0);
    let mut blind = straight(reaching());
    let mut sighted = seeing(reaching(), (8.0, 0.2), 1.5);

    let blind_step = blind.step(start, 0.1).expect("a valid control step");
    let sighted_step = sighted.step(start, 0.1).expect("a valid control step");
    let (open, pinched) = (
        plan_of(&blind_step)
            .progress_speed(0)
            .expect("a first stage"),
        plan_of(&sighted_step)
            .progress_speed(0)
            .expect("a first stage"),
    );
    assert!(
        pinched < open - 0.05,
        "the preview left the progress speed at {pinched} against {open} in the open"
    );
}

#[test]
fn an_obstacle_on_top_of_the_vehicle_bends_the_answer_rather_than_emptying_the_set() {
    // The slack is what makes the barrier soft. Without it a clearance
    // the vehicle is already inside would be a constraint no point
    // satisfies, and a controller that reports an empty feasible set
    // whenever it is too close to something is a controller that stops
    // working exactly when it matters.
    let mut controller = seeing(reaching(), (0.2, 0.0), 3.0);
    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    assert!(
        step.outcome.solved(),
        "a clearance the vehicle is inside emptied the feasible set: {:?}",
        step.outcome
    );
}

#[test]
fn the_predicted_clearance_is_read_along_the_plan_that_came_back() {
    let obstacle = (4.0, 1.0);
    let clearance = 1.5;
    let mut controller = seeing(reaching(), obstacle, clearance);
    let step = controller
        .step(moving(0.0, 2.0), 0.1)
        .expect("a valid control step");

    // Measured from the obstacle surface, per deviation A-16, which is
    // the convention the barrier rows were written in as well.
    let measured = closest_approach(plan_of(&step), obstacle) - clearance;
    assert!(
        (step.predicted_clearance - measured).abs() < 1e-9,
        "the step reported {} against a measured {measured}",
        step.predicted_clearance
    );
}

#[test]
fn a_controller_carrying_no_map_reports_an_unbounded_clearance() {
    let mut controller = straight(settings());
    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    assert!(
        step.predicted_clearance.is_infinite(),
        "a blind controller reported a clearance of {}",
        step.predicted_clearance
    );
}

#[test]
fn an_occupancy_of_the_wrong_dimension_is_refused_at_construction() {
    let occupancy = KdTreeOccupancy::new(&[vec![0.0, 0.0, 0.0]], 1.0)
        .expect("one obstacle point and a positive clearance");
    let outcome = PathFollowingMpc::new(settings(), Some(occupancy));
    assert!(
        matches!(
            outcome,
            Err(Error::DimensionMismatch {
                quantity: "occupancy dimension",
                expected: 2,
                actual: 3,
            })
        ),
        "a three dimensional map was accepted"
    );
}

// ------------------------------------------------ the reference itself ---

#[test]
fn setting_a_reference_appends_a_runway_along_the_final_tangent() {
    // One horizon of straight path past the last waypoint, floored at two
    // meters. The bound holding the path parameter inside the path is
    // what makes this necessary: without the runway it pinches against
    // its own ceiling over the last meters of a run.
    let settings = PathFollowingSettings {
        horizon_step_count: 10,
        step_interval: 0.1,
        cruise_speed: 4.0,
        ..settings()
    };
    let controller = on_path(settings, &[(0.0, 0.0), (10.0, 0.0)]);
    let reference = controller.reference().expect("a reference is set");

    // Four meters of horizon, which clears the two meter floor.
    assert!(
        (reference.total_length() - 14.0).abs() < 1e-9,
        "the reference came out {} meters long",
        reference.total_length()
    );
    let end = reference.sample_at(reference.total_length());
    assert!(
        (end.y).abs() < 1e-9,
        "the runway left the final tangent at y = {}",
        end.y
    );
}

#[test]
fn a_short_horizon_still_gets_the_two_meter_floor_of_runway() {
    let settings = PathFollowingSettings {
        horizon_step_count: 2,
        step_interval: 0.1,
        cruise_speed: 0.36,
        ..settings()
    };
    let controller = on_path(settings, &[(0.0, 0.0), (10.0, 0.0)]);
    let total_length = controller
        .reference()
        .expect("a reference is set")
        .total_length();
    assert!(
        (total_length - 12.0).abs() < 1e-9,
        "the reference came out {total_length} meters long"
    );
}

#[test]
fn replacing_the_reference_forgets_the_progress_and_the_limiter_counters() {
    let mut controller = straight(settings());
    let history = run(&mut controller, moving(0.0, 3.0), 3);
    assert!(
        history.iter().any(|(_, step)| step.saturation.saturated()),
        "the run never saturated, so the reset has nothing to clear"
    );

    controller
        .set_reference(&[(0.0, 0.0), (10.0, 5.0)])
        .expect("a two point reference");
    assert!(
        controller.progress() < 1e-12,
        "progress survived a new reference at {}",
        controller.progress()
    );
    assert!(
        !controller.saturation().saturated(),
        "the limiter counters survived a new reference: {:?}",
        controller.saturation()
    );
}

#[test]
fn the_curve_limited_cap_comes_down_as_a_corner_approaches() {
    // Two curvatures, for two different jobs. The one the progress cap
    // reads is spread over an arc and previewed backward from the corner
    // ahead, so the cap falls on the approach rather than at the corner
    // itself, which is where braking would have to be instantaneous. The
    // one the error expansions read is measured from the reference
    // heading and is zero on a straight, which is what keeps the
    // arc-length cross terms from being full size where the path does
    // not actually turn.
    let settings = PathFollowingSettings {
        cruise_speed: 1.0,
        ..settings_for(CommandLimits {
            max_turn_rate: 0.03,
            ..limits()
        })
    };
    let corner = [(0.0, 0.0), (10.0, 0.0), (40.0, 0.0), (40.0, 30.0)];
    let mut approaching = on_path(settings.clone(), &corner);
    let mut far_off = on_path(settings, &corner);

    let near = approaching
        .step(moving(35.0, 1.0), 0.1)
        .expect("a valid control step");
    let away = far_off
        .step(moving(5.0, 1.0), 0.1)
        .expect("a valid control step");
    assert_every_bound_holds(&approaching, &near);
    assert_every_bound_holds(&far_off, &away);

    let (near_speed, away_speed) = (
        plan_of(&near).progress_speed(0).expect("a first stage"),
        plan_of(&away).progress_speed(0).expect("a first stage"),
    );
    assert!(
        (away_speed - 1.0).abs() < 1e-3,
        "the progress speed on the open straight was {away_speed} against a cruise cap of 1.0"
    );
    assert!(
        near_speed < 0.6,
        "the progress speed near the corner was {near_speed} against a curve cap near 0.46"
    );
}

#[test]
fn a_plan_carries_one_position_per_node_including_the_one_it_starts_at() {
    // What a caller draws, and what the Python returned as its predicted
    // trajectory: the current pose first, then one position per predicted
    // step.
    let mut controller = straight(settings());
    let step = controller
        .step(moving(0.0, 1.0), 0.1)
        .expect("a valid control step");
    let plan = plan_of(&step);
    let positions: Vec<(f64, f64)> = plan.positions().collect();

    assert_eq!(positions.len(), plan.step_count() + 1);
    let first = positions.first().copied().expect("a first position");
    assert!(
        first.0.abs() < 1e-9 && first.1.abs() < 1e-9,
        "the plan starts at {first:?} rather than under the vehicle"
    );
}

#[test]
fn the_half_space_a_barrier_writes_is_the_one_it_meant() {
    // Reading the barrier back the way the bound tests read the box
    // bounds. Each row says the predicted position stays at least a
    // clearance from the obstacle along the direction from the obstacle
    // to the nominal, and the direction is what a sign error lives in: a
    // flipped normal is still a half-space, still convex, and still
    // solved, and it holds the plan against the obstacle rather than away
    // from it. With the obstacle further off the path than the clearance
    // the row is satisfiable, so the slack settles at zero and the plan
    // has to satisfy the row outright.
    let obstacle = (3.0, 2.0);
    let clearance = 1.0;
    let mut controller = seeing(reaching(), obstacle, clearance);
    let step = controller
        .step(moving(0.0, 2.0), 0.1)
        .expect("a valid control step");
    let (plan, nominal) = (plan_of(&step), nominal_of(&step));

    for node in 1..=plan.step_count() {
        let here = plan.state(node).expect("a node in the plan");
        let anchor = nominal.state(node).expect("a node in the nominal");
        let separation = (anchor.x - obstacle.0).hypot(anchor.y - obstacle.1);
        let normal = (
            (anchor.x - obstacle.0) / separation,
            (anchor.y - obstacle.1) / separation,
        );
        let reach = normal
            .0
            .mul_add(here.x - obstacle.0, normal.1 * (here.y - obstacle.1));
        assert!(
            reach >= clearance - CONSTRAINT_TOLERANCE,
            "node {node} sat {reach} along the barrier normal against a clearance of {clearance}"
        );
    }
}

#[test]
fn a_path_far_from_the_origin_is_tracked_like_any_other() {
    // The cost-agreement guard compares the objective this module built
    // against the one the solver reports, and the constant it carries
    // grows as the square of the distance from the origin. Judged against
    // the result of that cancellation rather than against the terms going
    // into it, the guard refuses every solve on a path far enough out and
    // the controller brakes forever for a reason nothing reports.
    for offset in [0.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6] {
        assert!(
            solves_at(offset),
            "a straight reference offset by {offset} metres did not solve"
        );
    }
}

/// Whether a straight reference offset by `offset` metres solves.
///
/// Asserted on the outcome rather than on the commanded speed: a safe
/// stop decelerates from whatever the vehicle was doing, so it still
/// reports a positive speed on the first braking step and a test reading
/// only the command cannot tell the two apart.
fn solves_at(offset: f64) -> bool {
    let waypoints = [(offset, offset), (offset + 40.0, offset)];
    let mut controller = on_path(settings(), &waypoints);
    let state = VehicleState {
        x: offset,
        y: offset,
        heading: 0.0,
        speed: 0.5,
        turn_rate: 0.0,
    };
    controller
        .step(state, 0.1)
        .is_ok_and(|step| matches!(step.outcome, StepOutcome::Solved { .. }))
}
