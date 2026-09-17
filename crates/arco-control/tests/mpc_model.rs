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

//! FR-MPC-01: every hand-derived derivative against a finite difference.
//!
//! Step three of phase 7 replaced a symbolic differentiator with a page
//! of algebra, and algebra on a page is where a sign goes missing. A
//! wrong Jacobian entry does not fail loudly: the convex program stays
//! solvable, the solve reports success, and the controller tracks
//! slightly worse than it should on the corners nobody drove yet. So
//! every derivative the module claims is compared against a central
//! difference of the nonlinear expression it was taken from, and the two
//! curvature cross terms, which couple the arc length to the error frame,
//! are checked on a left turn, on a right turn, and on a straight where
//! they have to vanish.

// Not `#[test]` functions, so the allowance in clippy.toml does not reach
// them: a fixture that cannot be built is the test being wrong.
#![expect(clippy::expect_used, reason = "test fixtures")]

use arco_control::mpc::model::{
    ControlInput, ErrorExpansion, INPUT_DIMENSION, PathErrors, STATE_DIMENSION, StageDynamics,
    VehicleState, heading_cost, linearize_path_errors, linearize_unicycle, path_errors,
    quadratic_heading_cost, unicycle_step,
};
use arco_control::mpc::reference::PathSample;
use arco_core::Error;

/// A reference curve, read at an arc length.
type Reference = Box<dyn Fn(f64) -> PathSample>;

/// The offset a central difference is taken over.
///
/// A central difference carries a truncation error falling as the square
/// of the offset and a cancellation error growing as its inverse. For the
/// magnitudes here both sit near 1e-10 at 1e-5, which is three orders
/// below the tolerance the assertions use.
const DIFFERENCE_OFFSET: f64 = 1e-5;

/// How far an analytic derivative may sit from its finite difference.
const DERIVATIVE_TOLERANCE: f64 = 1e-7;

/// The central difference of a scalar expression.
fn derivative(mut expression: impl FnMut(f64) -> f64, at: f64) -> f64 {
    (expression(at + DIFFERENCE_OFFSET) - expression(at - DIFFERENCE_OFFSET))
        / (2.0 * DIFFERENCE_OFFSET)
}

/// Linearization points spread over the quadrants, including a standstill.
///
/// The heading enters the model through a sine and a cosine, so a single
/// point proves nothing about the sign of either: a Jacobian wrong by a
/// transposition agrees with its finite difference at a heading of zero.
fn states() -> Vec<VehicleState> {
    vec![
        VehicleState {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
            speed: 1.0,
            turn_rate: 0.0,
        },
        VehicleState {
            x: 3.5,
            y: -2.25,
            heading: 0.7,
            speed: 2.4,
            turn_rate: -0.3,
        },
        VehicleState {
            x: -8.0,
            y: 11.5,
            heading: 2.9,
            speed: 0.4,
            turn_rate: 0.9,
        },
        VehicleState {
            x: 1.0,
            y: 1.0,
            heading: -2.2,
            // A standstill is where the position rows of the Jacobian
            // lose their dependence on the heading, and where a
            // controller recovering from a stop starts every time.
            speed: 0.0,
            turn_rate: 0.15,
        },
        VehicleState {
            x: -0.5,
            y: 4.0,
            heading: -0.95,
            speed: 3.3,
            turn_rate: 1.2,
        },
    ]
}

/// Inputs covering both signs of both components.
fn inputs() -> Vec<ControlInput> {
    vec![
        ControlInput {
            acceleration: 0.0,
            turn_rate_change: 0.0,
        },
        ControlInput {
            acceleration: 1.4,
            turn_rate_change: -0.8,
        },
        ControlInput {
            acceleration: -2.1,
            turn_rate_change: 0.55,
        },
    ]
}

/// A circle of signed radius, parameterized by arc length.
///
/// The derivation of the arc-length gradients assumes `X' = cos T`,
/// `Y' = sin T` and `T' = K`, and a circle satisfies all three exactly at
/// every arc length, which a polyline does not. A negative radius turns
/// the other way and puts a negative curvature through the same test,
/// because a cross term with the wrong sign passes a one-sided sweep.
fn circle(radius: f64) -> impl Fn(f64) -> PathSample {
    move |arc_length| {
        let angle = arc_length / radius;
        PathSample {
            x: radius * angle.sin(),
            y: radius * (1.0 - angle.cos()),
            heading: angle,
            curvature: 1.0 / radius,
        }
    }
}

/// A straight reference, where both curvature cross terms have to vanish.
fn straight(heading: f64) -> impl Fn(f64) -> PathSample {
    move |arc_length| PathSample {
        x: arc_length * heading.cos(),
        y: arc_length * heading.sin(),
        heading,
        curvature: 0.0,
    }
}

/// The three references every arc-length derivative is checked against.
fn references() -> Vec<(&'static str, Reference)> {
    vec![
        ("a straight reference", Box::new(straight(0.6))),
        ("a left-hand circle", Box::new(circle(8.0))),
        ("a right-hand circle", Box::new(circle(-3.5))),
    ]
}

/// The arc lengths the sweep linearizes at.
const ARC_LENGTHS: [f64; 3] = [0.0, 2.0, 7.5];

/// The errors of a pose against a reference read at `arc_length`.
fn errors_at(state: VehicleState, reference: &Reference, arc_length: f64) -> PathErrors {
    path_errors(state, reference(arc_length)).expect("a finite pose against a finite reference")
}

/// Reads the contouring error out of a set, as a function pointer.
///
/// Named rather than written inline at the call site: a closure needs a
/// coercion to sit in an array beside another one, and a cast that says
/// so reads as though something were being converted.
fn contour_error() -> fn(PathErrors) -> f64 {
    |errors| errors.contour
}

/// Reads the lag error out of a set, as a function pointer.
fn lag_error() -> fn(PathErrors) -> f64 {
    |errors| errors.lag
}

/// The finite difference of one error in one state component.
fn error_slope(
    state: VehicleState,
    sample: PathSample,
    component: usize,
    pick: fn(PathErrors) -> f64,
) -> f64 {
    let values = state.to_array();
    derivative(
        |value| {
            let mut perturbed = values;
            perturbed[component] = value;
            pick(
                path_errors(VehicleState::from_array(perturbed), sample)
                    .expect("a finite pose against a finite reference"),
            )
        },
        values[component],
    )
}

/// The finite difference of the step in one state component.
fn state_column(
    state: VehicleState,
    input: ControlInput,
    dt: f64,
    component: usize,
) -> [f64; STATE_DIMENSION] {
    let advanced = |offset: f64| {
        let mut values = state.to_array();
        values[component] += offset;
        unicycle_step(VehicleState::from_array(values), input, dt)
            .expect("a finite state advances")
            .to_array()
    };
    difference(advanced(DIFFERENCE_OFFSET), advanced(-DIFFERENCE_OFFSET))
}

/// The finite difference of the step in one input component.
fn input_column(
    state: VehicleState,
    input: ControlInput,
    dt: f64,
    component: usize,
) -> [f64; STATE_DIMENSION] {
    let advanced = |offset: f64| {
        let mut values = input.to_array();
        values[component] += offset;
        unicycle_step(state, ControlInput::from_array(values), dt)
            .expect("a finite state advances")
            .to_array()
    };
    difference(advanced(DIFFERENCE_OFFSET), advanced(-DIFFERENCE_OFFSET))
}

/// The central difference of two propagated states.
fn difference(
    ahead: [f64; STATE_DIMENSION],
    behind: [f64; STATE_DIMENSION],
) -> [f64; STATE_DIMENSION] {
    let mut slope = [0.0; STATE_DIMENSION];
    for (slot, (high, low)) in slope.iter_mut().zip(ahead.iter().zip(behind.iter())) {
        *slot = (high - low) / (2.0 * DIFFERENCE_OFFSET);
    }
    slope
}

/// The largest absolute difference between two states.
fn largest_gap(left: VehicleState, right: VehicleState) -> f64 {
    left.to_array()
        .iter()
        .zip(right.to_array().iter())
        .fold(0.0_f64, |worst, (one, other)| {
            worst.max((one - other).abs())
        })
}

/// A stage linearized about the middle of the sweep.
fn nominal_stage() -> (VehicleState, ControlInput, f64, StageDynamics) {
    let state = VehicleState {
        x: 3.5,
        y: -2.25,
        heading: 0.7,
        speed: 2.4,
        turn_rate: -0.3,
    };
    let input = ControlInput {
        acceleration: 1.4,
        turn_rate_change: -0.8,
    };
    let dt = 0.05;
    let dynamics = linearize_unicycle(state, input, dt).expect("a finite linearization point");
    (state, input, dt, dynamics)
}

#[test]
fn the_state_jacobian_matches_a_finite_difference() {
    // FR-MPC-01, the five by five. Every entry, at fifteen linearization
    // points and two step lengths, against a central difference of the
    // nonlinear step it was derived from.
    let mut worst = 0.0_f64;
    for state in states() {
        for input in inputs() {
            for dt in [0.02, 0.1] {
                let dynamics =
                    linearize_unicycle(state, input, dt).expect("a finite linearization point");
                for component in 0..STATE_DIMENSION {
                    let numerical = state_column(state, input, dt, component);
                    for (row, slope) in numerical.iter().enumerate() {
                        let analytic = dynamics.state_jacobian[row][component];
                        worst = worst.max((analytic - slope).abs());
                    }
                }
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "the state Jacobian is off its finite difference by {worst:e}"
    );
}

#[test]
fn the_input_jacobian_matches_a_finite_difference() {
    // FR-MPC-01, the five by two. The input enters two rows linearly, so
    // what this catches is an entry in the wrong row rather than a wrong
    // derivative: acceleration reaching the turn rate would be invisible
    // to any test that only checked the acceleration row.
    let mut worst = 0.0_f64;
    for state in states() {
        for input in inputs() {
            for dt in [0.02, 0.1] {
                let dynamics =
                    linearize_unicycle(state, input, dt).expect("a finite linearization point");
                for component in 0..INPUT_DIMENSION {
                    let numerical = input_column(state, input, dt, component);
                    for (row, slope) in numerical.iter().enumerate() {
                        let analytic = dynamics.input_jacobian[row][component];
                        worst = worst.max((analytic - slope).abs());
                    }
                }
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "the input Jacobian is off its finite difference by {worst:e}"
    );
}

#[test]
fn the_affine_stage_reproduces_the_nonlinear_step_at_its_own_point() {
    // What the residual is for. The stage constraint the solver carries
    // has to agree with the model exactly where it was taken, or the
    // first SQP iteration starts from a trajectory the vehicle is not on.
    let mut worst = 0.0_f64;
    for state in states() {
        for input in inputs() {
            for dt in [0.02, 0.1] {
                let dynamics =
                    linearize_unicycle(state, input, dt).expect("a finite linearization point");
                let exact = unicycle_step(state, input, dt).expect("a finite state advances");
                worst = worst.max(largest_gap(dynamics.propagate(state, input), exact));
            }
        }
    }
    assert!(
        worst < 1e-12,
        "the affine stage misses its own linearization point by {worst:e}"
    );
}

#[test]
fn the_affine_stage_agrees_with_the_nonlinear_step_to_second_order() {
    // A tangent plane, not merely a plane through the point: halving the
    // perturbation has to quarter the disagreement. A gradient scaled by
    // the wrong constant still passes the exactness test above, and fails
    // this one.
    let (state, input, dt, dynamics) = nominal_stage();
    let disagreement = |scale: f64| {
        let moved = VehicleState {
            heading: state.heading + 0.7 * scale,
            speed: state.speed + 1.3 * scale,
            ..state
        };
        largest_gap(
            dynamics.propagate(moved, input),
            unicycle_step(moved, input, dt).expect("a finite state advances"),
        )
    };
    let coarse = disagreement(0.02);
    let fine = disagreement(0.01);
    assert!(coarse > 1e-9, "the coarse step is lost in rounding");
    let ratio = coarse / fine;
    assert!(
        (3.5..4.5).contains(&ratio),
        "halving the step changed the error by {ratio}, not by four"
    );
}

#[test]
fn the_error_gradients_in_the_state_match_a_finite_difference() {
    // The three position and heading gradients at once, including the
    // speed and turn-rate columns that have to be exactly zero: an error
    // gradient written into the wrong slot reads as a plausible number
    // until something differentiates it.
    let mut worst = 0.0_f64;
    for (name, reference) in references() {
        for state in states() {
            for arc_length in ARC_LENGTHS {
                let sample = reference(arc_length);
                let expansion = linearize_path_errors(state, arc_length, sample)
                    .expect("a finite linearization point");
                for component in 0..STATE_DIMENSION {
                    for (label, expansion, pick) in [
                        ("contour", expansion.contour, contour_error()),
                        ("lag", expansion.lag, lag_error()),
                    ] {
                        let numerical = error_slope(state, sample, component, pick);
                        let analytic = expansion.state_gradient[component];
                        let deviation = (analytic - numerical).abs();
                        assert!(
                            deviation < DERIVATIVE_TOLERANCE,
                            "d {label} / d state[{component}] on {name} is off by {deviation:e}"
                        );
                        worst = worst.max(deviation);
                    }
                }
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "a position gradient is off its finite difference by {worst:e}"
    );
}

#[test]
fn the_heading_gradient_matches_a_finite_difference() {
    // Taken at a pose pointed roughly along the reference. The wrapped
    // error is discontinuous half a turn away from it, where it has no
    // derivative to compare against, and a controller that far out of
    // alignment is not what the linearization is for.
    let mut worst = 0.0_f64;
    for (name, reference) in references() {
        for arc_length in ARC_LENGTHS {
            let sample = reference(arc_length);
            for offset in [-1.3, -0.4, 0.0, 0.1, 0.9] {
                let state = VehicleState {
                    x: 1.0,
                    y: -0.5,
                    heading: sample.heading + offset,
                    speed: 1.5,
                    turn_rate: 0.1,
                };
                let expansion = linearize_path_errors(state, arc_length, sample)
                    .expect("a finite linearization point");
                for component in 0..STATE_DIMENSION {
                    let numerical = error_slope(state, sample, component, |errors| errors.heading);
                    let deviation = (expansion.heading.state_gradient[component] - numerical).abs();
                    assert!(
                        deviation < DERIVATIVE_TOLERANCE,
                        "d heading / d state[{component}] on {name} is off by {deviation:e}"
                    );
                    worst = worst.max(deviation);
                }
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "the heading gradient is off its finite difference by {worst:e}"
    );
}

#[test]
fn the_contouring_arc_length_derivative_matches_a_finite_difference() {
    // The first cross term, `d contour / ds = -K * lag`. Moving along the
    // path rotates the frame the error is measured in, and the whole term
    // is that rotation: it is zero on a straight and grows with how far
    // ahead or behind the reference point the vehicle sits.
    let mut worst = 0.0_f64;
    for (name, reference) in references() {
        for state in states() {
            for arc_length in ARC_LENGTHS {
                let expansion = linearize_path_errors(state, arc_length, reference(arc_length))
                    .expect("a finite linearization point");
                let numerical = derivative(
                    |value| errors_at(state, &reference, value).contour,
                    arc_length,
                );
                let deviation = (expansion.contour.arc_length_gradient - numerical).abs();
                assert!(
                    deviation < DERIVATIVE_TOLERANCE,
                    "d contour / ds on {name} at s = {arc_length} is off by {deviation:e}"
                );
                worst = worst.max(deviation);
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "the contouring arc-length derivative is off by {worst:e}"
    );
}

#[test]
fn the_lag_arc_length_derivative_matches_a_finite_difference() {
    // The second cross term, `d lag / ds = -1 + K * contour`. The minus
    // one is the reference point sliding forward out from under the
    // vehicle at one meter per meter, and the curvature term is the same
    // frame rotation the contouring error sees, with the two errors
    // swapped and the sign flipped. Getting that sign backwards is the
    // single easiest mistake in this module, and it survives any test
    // taken on a straight path.
    let mut worst = 0.0_f64;
    for (name, reference) in references() {
        for state in states() {
            for arc_length in ARC_LENGTHS {
                let expansion = linearize_path_errors(state, arc_length, reference(arc_length))
                    .expect("a finite linearization point");
                let numerical =
                    derivative(|value| errors_at(state, &reference, value).lag, arc_length);
                let deviation = (expansion.lag.arc_length_gradient - numerical).abs();
                assert!(
                    deviation < DERIVATIVE_TOLERANCE,
                    "d lag / ds on {name} at s = {arc_length} is off by {deviation:e}"
                );
                worst = worst.max(deviation);
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "the lag arc-length derivative is off by {worst:e}"
    );
}

#[test]
fn the_heading_arc_length_derivative_matches_a_finite_difference() {
    // `d heading / ds = -K`, taken where the wrapped error is smooth.
    let mut worst = 0.0_f64;
    for (name, reference) in references() {
        for arc_length in ARC_LENGTHS {
            for offset in [-1.3, -0.4, 0.0, 0.1, 0.9] {
                let heading = reference(arc_length).heading + offset;
                let state = VehicleState {
                    x: 1.0,
                    y: -0.5,
                    heading,
                    speed: 1.5,
                    turn_rate: 0.1,
                };
                let expansion = linearize_path_errors(state, arc_length, reference(arc_length))
                    .expect("a finite linearization point");
                let numerical = derivative(
                    |value| errors_at(state, &reference, value).heading,
                    arc_length,
                );
                let deviation = (expansion.heading.arc_length_gradient - numerical).abs();
                assert!(
                    deviation < DERIVATIVE_TOLERANCE,
                    "d heading / ds on {name} at s = {arc_length} is off by {deviation:e}"
                );
                worst = worst.max(deviation);
            }
        }
    }
    assert!(
        worst < DERIVATIVE_TOLERANCE,
        "the heading arc-length derivative is off by {worst:e}"
    );
}

#[test]
fn an_affine_error_form_is_exact_at_its_linearization_point() {
    // What the constant is for. A caller writing the residual in absolute
    // variables, rather than in offsets from a point that moves every SQP
    // iteration, needs the affine form to return the error itself there.
    let mut worst = 0.0_f64;
    for (_, reference) in references() {
        for state in states() {
            for arc_length in ARC_LENGTHS {
                let expansion = linearize_path_errors(state, arc_length, reference(arc_length))
                    .expect("a finite linearization point");
                for part in [expansion.contour, expansion.lag, expansion.heading] {
                    let restated: f64 = part.evaluate(state, arc_length);
                    worst = worst.max((restated - part.value).abs());
                }
            }
        }
    }
    assert!(
        worst < 1e-9,
        "an affine error form misses its own point by {worst:e}"
    );
}

#[test]
fn an_affine_error_form_agrees_with_the_error_to_second_order() {
    // Gradient and constant together: move the pose and the progress at
    // once, halve the move, and the disagreement has to quarter.
    let reference = circle(-3.5);
    let state = VehicleState {
        x: 0.4,
        y: -1.1,
        heading: 0.2,
        speed: 1.5,
        turn_rate: 0.1,
    };
    let arc_length = 2.0;
    let expansion = linearize_path_errors(state, arc_length, reference(arc_length))
        .expect("a finite linearization point");

    let disagreement = |part: ErrorExpansion, pick: fn(PathErrors) -> f64, scale: f64| {
        let moved = VehicleState {
            x: state.x + 0.9 * scale,
            y: state.y - 0.6 * scale,
            heading: state.heading + 0.5 * scale,
            ..state
        };
        let progress = arc_length + 1.1 * scale;
        let exact = path_errors(moved, reference(progress)).expect("a finite pose");
        (part.evaluate(moved, progress) - pick(exact)).abs()
    };

    for (label, part, pick) in [
        ("contour", expansion.contour, contour_error()),
        ("lag", expansion.lag, lag_error()),
    ] {
        let coarse = disagreement(part, pick, 0.04);
        let fine = disagreement(part, pick, 0.02);
        assert!(
            coarse > 1e-9,
            "the {label} form is linear in this direction, so the test proves nothing"
        );
        let ratio = coarse / fine;
        assert!(
            (3.5..4.5).contains(&ratio),
            "halving the step changed the {label} disagreement by {ratio}, not by four"
        );
    }

    // The heading error is the exception, and it is exact rather than
    // second-order: the reference heading of a constant-curvature curve
    // is a linear function of the arc length, so the expansion is the
    // error itself and not a tangent to it. Asserting the quartering
    // above would have been asserting a rounding ratio.
    for scale in [0.04, 0.02] {
        let gap = disagreement(expansion.heading, |errors| errors.heading, scale);
        assert!(gap < 1e-12, "the heading form is off by {gap:e} at {scale}");
    }
}

#[test]
fn the_two_forms_of_the_heading_cost_are_one_function() {
    // The identity the module claims: `sin(e)^2 + (1 - cos e)^2` expands
    // to `sin^2 + cos^2 + 1 - 2 cos e`, and the Pythagorean identity
    // leaves `2 (1 - cos e)`. Not an approximation, so the tolerance is
    // rounding rather than modelling.
    let mut worst = 0.0_f64;
    let mut error = -7.0_f64;
    while error <= 7.0 {
        let python = error.sin().powi(2) + (1.0 - error.cos()).powi(2);
        let compact = heading_cost(error).expect("a finite heading error");
        worst = worst.max((python - compact).abs());
        error += 0.013;
    }
    assert!(
        worst < 1e-15,
        "the two forms of the heading cost differ by {worst:e}, which is more than rounding"
    );
}

#[test]
fn the_quadratic_heading_cost_tracks_the_exact_one_where_it_is_used() {
    // The Gauss-Newton surrogate is an approximation and the doc comment
    // says where it holds. This is that claim, pinned: inside half a
    // radian the two costs agree to a couple of percent, and the leading
    // term of their difference is `e^4 / 12` across the whole wrapped
    // range.
    let mut worst_relative = 0.0_f64;
    let mut error = -0.5_f64;
    while error <= 0.5 {
        let exact = heading_cost(error).expect("a finite heading error");
        let quadratic = quadratic_heading_cost(error).expect("a finite heading error");
        if exact > 1e-6 {
            worst_relative = worst_relative.max(((quadratic - exact) / exact).abs());
        }
        // The surrogate always charges at least what the exact cost does,
        // so the solver never under-penalizes a heading error.
        assert!(quadratic >= exact - 1e-15, "at e = {error}");
        error += 0.007;
    }
    assert!(
        worst_relative < 0.0212,
        "the surrogate is off by {worst_relative} inside half a radian"
    );

    let mut error = -core::f64::consts::PI;
    while error <= core::f64::consts::PI {
        let gap = quadratic_heading_cost(error).expect("a finite heading error")
            - heading_cost(error).expect("a finite heading error");
        assert!(
            gap <= error.powi(4) / 12.0 + 1e-15,
            "the fourth-order bound is broken at e = {error}"
        );
        error += 0.01;
    }
}

#[test]
fn the_quadratic_heading_cost_is_blind_to_a_full_turn() {
    // What the wrapping buys. Without it a pose reported a turn away from
    // the reference costs forty times what the same pose costs when the
    // angle is reported the other way round.
    let turn = core::f64::consts::TAU;
    for error in [-0.8_f64, -0.1, 0.0, 0.35, 1.2] {
        let plain = quadratic_heading_cost(error).expect("a finite heading error");
        let turned = quadratic_heading_cost(error + turn).expect("a finite heading error");
        assert!(
            (plain - turned).abs() < 1e-12,
            "a full turn changed the cost at e = {error}"
        );
    }
}

#[test]
fn the_heading_error_is_wrapped_where_the_original_subtracted_raw() {
    // The one place this module deliberately differs from the Python
    // expression it was ported from, which leaned on an unwrapped spline
    // heading to keep the subtraction meaningful.
    let sample = PathSample {
        x: 0.0,
        y: 0.0,
        heading: 3.0,
        curvature: 0.0,
    };
    let state = VehicleState {
        x: 0.0,
        y: 0.0,
        heading: -3.0,
        speed: 1.0,
        turn_rate: 0.0,
    };
    let errors = path_errors(state, sample).expect("a finite pose");
    let expected = core::f64::consts::TAU - 6.0;
    assert!(
        (errors.heading - expected).abs() < 1e-12,
        "the heading error is {}, not the short way round",
        errors.heading
    );
}

#[test]
fn a_state_carrying_a_nan_is_rejected_before_it_reaches_a_jacobian() {
    // FR-SAFE-07. One non-finite component spreads to every entry of the
    // matrix built from it, and the solver reports that as a numerical
    // failure with nothing pointing back at the cause.
    for value in [f64::NAN, f64::INFINITY] {
        let state = VehicleState {
            x: value,
            ..VehicleState::default()
        };
        let failure = linearize_unicycle(state, ControlInput::default(), 0.05)
            .expect_err("a non-finite state has no Jacobian");
        assert!(matches!(failure, Error::NotFinite { .. }), "{failure}");
    }
}

#[test]
fn a_non_finite_input_is_rejected() {
    let input = ControlInput {
        acceleration: f64::NAN,
        turn_rate_change: 0.0,
    };
    let failure = unicycle_step(VehicleState::default(), input, 0.05)
        .expect_err("a non-finite input has no step");
    assert!(matches!(failure, Error::NotFinite { .. }), "{failure}");
}

#[test]
fn a_step_that_does_not_advance_time_is_rejected() {
    // A zero or negative step is a caller bug rather than a hard case:
    // the Jacobian it produces is the identity, which looks like a
    // perfectly well-conditioned model of a vehicle that cannot move.
    for dt in [0.0, -0.05] {
        let failure = linearize_unicycle(VehicleState::default(), ControlInput::default(), dt)
            .expect_err("a step has to advance time");
        assert!(matches!(failure, Error::OutOfRange { .. }), "{failure}");
    }
    let failure = linearize_unicycle(VehicleState::default(), ControlInput::default(), f64::NAN)
        .expect_err("a step has to be a real number");
    assert!(matches!(failure, Error::NotFinite { .. }), "{failure}");
}

#[test]
fn a_reference_sample_carrying_a_nan_is_rejected() {
    let sample = PathSample {
        x: 0.0,
        y: 0.0,
        heading: 0.0,
        curvature: f64::NAN,
    };
    let failure = linearize_path_errors(VehicleState::default(), 0.0, sample)
        .expect_err("a non-finite curvature has no gradient");
    assert!(matches!(failure, Error::NotFinite { .. }), "{failure}");

    let failure = linearize_path_errors(
        VehicleState::default(),
        f64::INFINITY,
        PathSample {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
            curvature: 0.0,
        },
    )
    .expect_err("a non-finite arc length has no gradient");
    assert!(matches!(failure, Error::NotFinite { .. }), "{failure}");
}
