//! Where the nonlinear model is replaced by the plane tangent to it.
//!
//! ADR-002 turned one nonlinear program into a sequence of convex ones,
//! and this module is the join between them. It takes a point, and it
//! returns the first-order model of the unicycle and of the three
//! contouring errors about that point. Nothing here holds state, walks a
//! horizon, or reaches the solver, so a wrong sign surfaces as a failing
//! derivative test rather than as a tracking error three layers up.
//!
//! The Python original writes the same expressions symbolically in
//! `_build_nlp` and lets `CasADi` differentiate them. `FR-MPC-01` is what
//! replaces that step, and the price of dropping the symbolic engine is
//! that every derivative below is derived by hand. Each one is checked
//! against a finite difference of the expression it came from, in
//! `tests/mpc_model.rs`.
//!
//! The arc-length derivatives assume a reference obeying `X'(s) = cos T`,
//! `Y'(s) = sin T` and `T'(s) = K`, which is what parameterizing a curve
//! by arc length means, and what
//! [`ReferencePath`](crate::mpc::reference::ReferencePath) approximates
//! with a polyline. On a polyline the heading is constant within a
//! segment and jumps at a vertex, so the reported curvature is a spread
//! version of that jump; the linearization is exact for the smooth curve
//! the spreading stands in for, not for the polyline itself.

use arco_core::Error;
use arco_core::numeric::{angle_difference, wrap_angle};

use crate::mpc::reference::PathSample;

/// How many components a vehicle state carries.
pub const STATE_DIMENSION: usize = 5;

/// How many components a control input carries.
pub const INPUT_DIMENSION: usize = 2;

/// The state the unicycle model advances.
///
/// The field order is the row and column order of every Jacobian here,
/// and [`VehicleState::to_array`] is the only place that order is
/// written down.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VehicleState {
    /// Position along the first axis, meters.
    pub x: f64,
    /// Position along the second axis, meters.
    pub y: f64,
    /// Heading, radians, unwrapped.
    ///
    /// Unwrapped on purpose. Wrapping inside the propagation puts a jump
    /// of a full turn in the predicted trajectory at the branch cut, and
    /// the Jacobian there would describe a vehicle that teleports. The
    /// wrapping happens once, in the heading error, where an angle is
    /// finally compared against another angle.
    pub heading: f64,
    /// Forward speed, meters per second.
    pub speed: f64,
    /// Turn rate, radians per second, positive counterclockwise.
    pub turn_rate: f64,
}

impl VehicleState {
    /// The state in the row order the Jacobians use.
    #[must_use]
    pub const fn to_array(self) -> [f64; STATE_DIMENSION] {
        [self.x, self.y, self.heading, self.speed, self.turn_rate]
    }

    /// Reads a state back out of that order.
    #[must_use]
    pub fn from_array(values: [f64; STATE_DIMENSION]) -> Self {
        let [x, y, heading, speed, turn_rate] = values;
        Self {
            x,
            y,
            heading,
            speed,
            turn_rate,
        }
    }

    /// Rejects a state no Jacobian can be taken at.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a component is NaN or infinite.
    /// `FR-SAFE-07`: one NaN in a linearization point spreads to every
    /// entry of the matrix built from it, and the solver reports that as
    /// a numerical failure with nothing pointing back at the cause.
    pub fn check(self) -> Result<(), Error> {
        check_all([
            ("state x", self.x),
            ("state y", self.y),
            ("state heading", self.heading),
            ("state speed", self.speed),
            ("state turn rate", self.turn_rate),
        ])
    }
}

/// What the controller may ask the vehicle for.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ControlInput {
    /// Linear acceleration, meters per second squared.
    pub acceleration: f64,
    /// Rate of change of the turn rate, radians per second squared.
    pub turn_rate_change: f64,
}

impl ControlInput {
    /// The input in the column order the input Jacobian uses.
    #[must_use]
    pub const fn to_array(self) -> [f64; INPUT_DIMENSION] {
        [self.acceleration, self.turn_rate_change]
    }

    /// Reads an input back out of that order.
    #[must_use]
    pub fn from_array(values: [f64; INPUT_DIMENSION]) -> Self {
        let [acceleration, turn_rate_change] = values;
        Self {
            acceleration,
            turn_rate_change,
        }
    }

    /// Rejects an input no Jacobian can be taken at.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a component is NaN or infinite.
    pub fn check(self) -> Result<(), Error> {
        check_all([
            ("input acceleration", self.acceleration),
            ("input turn rate change", self.turn_rate_change),
        ])
    }
}

/// The unicycle model, linearized about one state and input.
///
/// A stage of the convex program reads
/// `x_next - state_jacobian * x - input_jacobian * u = residual`, in
/// absolute variables rather than in deviations from the linearization
/// point. Writing it that way keeps the solver's variables the states
/// themselves, so a bound on the speed is a bound on a variable instead
/// of a bound on an offset whose origin moves every SQP iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageDynamics {
    /// The five by five derivative of the step with respect to the state.
    pub state_jacobian: [[f64; STATE_DIMENSION]; STATE_DIMENSION],
    /// The five by two derivative of the step with respect to the input.
    pub input_jacobian: [[f64; INPUT_DIMENSION]; STATE_DIMENSION],
    /// What the affine model carries that the two Jacobians do not.
    ///
    /// `f(x, u) - state_jacobian * x - input_jacobian * u` at the
    /// linearization point, which is where the nonlinearity of the step
    /// ends up once the tangent plane has been subtracted from it.
    pub residual: [f64; STATE_DIMENSION],
}

impl StageDynamics {
    /// The affine model evaluated at a state and an input.
    ///
    /// Equal to [`unicycle_step`] at the point the model was taken about,
    /// and second-order accurate away from it.
    #[must_use]
    pub fn propagate(&self, state: VehicleState, input: ControlInput) -> VehicleState {
        let state_values = state.to_array();
        let input_values = input.to_array();
        let mut next = [0.0_f64; STATE_DIMENSION];
        for (((slot, state_row), input_row), residual) in next
            .iter_mut()
            .zip(&self.state_jacobian)
            .zip(&self.input_jacobian)
            .zip(&self.residual)
        {
            *slot = dot(state_row, &state_values) + dot(input_row, &input_values) + residual;
        }
        VehicleState::from_array(next)
    }
}

/// How far a pose sits from the reference point it is measured against.
///
/// The position error is split along the reference tangent rather than
/// reported as a distance, because the two halves mean different things
/// to the controller: the contouring error is what tracking is judged on,
/// and the lag error is what couples the virtual progress to the vehicle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PathErrors {
    /// Lateral error, meters, positive to the left of the reference.
    pub contour: f64,
    /// Longitudinal error, meters, positive ahead of the reference point.
    pub lag: f64,
    /// Heading minus the reference heading, radians, wrapped.
    ///
    /// Wrapped where the Python original subtracts an unwrapped spline
    /// heading. The two agree wherever the vehicle is pointed roughly
    /// along the path, and only the wrapped one is bounded when it is
    /// not.
    pub heading: f64,
}

/// One error, and the affine function that stands in for it.
///
/// `value` is what the error is at the linearization point. The affine
/// form is `state_gradient * x + arc_length_gradient * s + constant`, in
/// absolute variables, and `constant` is what makes it agree with `value`
/// at that point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorExpansion {
    /// The error at the linearization point, in the error's own unit.
    pub value: f64,
    /// The derivative with respect to each state component.
    pub state_gradient: [f64; STATE_DIMENSION],
    /// The derivative with respect to the arc length, per meter.
    pub arc_length_gradient: f64,
    /// The offset that makes the affine form exact at the point.
    pub constant: f64,
}

impl ErrorExpansion {
    /// Builds the expansion, solving for the constant.
    fn about(
        value: f64,
        state_gradient: [f64; STATE_DIMENSION],
        arc_length_gradient: f64,
        state: VehicleState,
        arc_length: f64,
    ) -> Self {
        let linear = dot(&state_gradient, &state.to_array()) + arc_length_gradient * arc_length;
        Self {
            value,
            state_gradient,
            arc_length_gradient,
            constant: value - linear,
        }
    }

    /// The affine form evaluated at a state and an arc length.
    ///
    /// Returns [`ErrorExpansion::value`] at the linearization point, and
    /// the tangent approximation of the error anywhere else.
    #[must_use]
    pub fn evaluate(&self, state: VehicleState, arc_length: f64) -> f64 {
        dot(&self.state_gradient, &state.to_array())
            + self.arc_length_gradient.mul_add(arc_length, self.constant)
    }
}

/// The three contouring errors, each linearized about the same point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathErrorExpansion {
    /// The lateral error and its gradient.
    pub contour: ErrorExpansion,
    /// The longitudinal error and its gradient.
    pub lag: ErrorExpansion,
    /// The heading error and its gradient.
    pub heading: ErrorExpansion,
}

/// Advances the unicycle by one step of explicit Euler.
///
/// The nonlinear model every derivative in this module is taken of, and
/// the rollout an SQP iteration linearizes about. Explicit Euler rather
/// than a higher-order integrator because the Python formulation
/// discretized that way, and matching it is what keeps the tracking
/// comparison of `FR-MPC-02` about the solver rather than about the
/// integrator.
///
/// # Arguments
///
/// * `state` - The state to advance, meters, radians, and their rates.
/// * `input` - Acceleration and turn-rate change, per second squared.
/// * `dt` - The prediction step, seconds.
///
/// # Returns
///
/// The propagated state, with the heading left unwrapped.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when the state, the input, or `dt`
/// carries a NaN or an infinity, and [`Error::OutOfRange`] when `dt` is
/// not strictly positive.
pub fn unicycle_step(
    state: VehicleState,
    input: ControlInput,
    dt: f64,
) -> Result<VehicleState, Error> {
    state.check()?;
    input.check()?;
    check_step(dt)?;

    let (sin_heading, cos_heading) = state.heading.sin_cos();
    Ok(VehicleState {
        x: (state.speed * cos_heading).mul_add(dt, state.x),
        y: (state.speed * sin_heading).mul_add(dt, state.y),
        heading: state.turn_rate.mul_add(dt, state.heading),
        speed: input.acceleration.mul_add(dt, state.speed),
        turn_rate: input.turn_rate_change.mul_add(dt, state.turn_rate),
    })
}

/// Linearizes [`unicycle_step`] about a state and an input.
///
/// Only the first two rows carry anything the caller could get wrong: the
/// heading, speed and turn-rate rows of the model are already linear, so
/// their derivatives are the model itself. The two position rows are
/// where the speed multiplies a sine of the heading, and where the
/// residual comes from.
///
/// # Arguments
///
/// * `state` - The state to linearize about.
/// * `input` - The input to linearize about.
/// * `dt` - The prediction step, seconds.
///
/// # Returns
///
/// The two Jacobians and the affine residual, as [`StageDynamics`].
///
/// # Errors
///
/// As [`unicycle_step`].
pub fn linearize_unicycle(
    state: VehicleState,
    input: ControlInput,
    dt: f64,
) -> Result<StageDynamics, Error> {
    let next = unicycle_step(state, input, dt)?.to_array();
    let (sin_heading, cos_heading) = state.heading.sin_cos();

    let state_jacobian = [
        [
            1.0,
            0.0,
            -state.speed * sin_heading * dt,
            cos_heading * dt,
            0.0,
        ],
        [
            0.0,
            1.0,
            state.speed * cos_heading * dt,
            sin_heading * dt,
            0.0,
        ],
        [0.0, 0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let input_jacobian = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [dt, 0.0], [0.0, dt]];

    // Solved for rather than written out: the closed form of the residual
    // is a pair of terms that look like a typo (`v * heading * sin` and
    // its mate), and subtracting the tangent plane from the model is the
    // definition, so the definition is what runs.
    let state_values = state.to_array();
    let input_values = input.to_array();
    let mut residual = [0.0_f64; STATE_DIMENSION];
    for (((slot, next_value), state_row), input_row) in residual
        .iter_mut()
        .zip(next)
        .zip(&state_jacobian)
        .zip(&input_jacobian)
    {
        *slot = next_value - dot(state_row, &state_values) - dot(input_row, &input_values);
    }

    Ok(StageDynamics {
        state_jacobian,
        input_jacobian,
        residual,
    })
}

/// The contouring, lag and heading errors of a pose against a reference.
///
/// The nonlinear expressions [`linearize_path_errors`] differentiates.
///
/// # Arguments
///
/// * `state` - The pose to measure, with its speed and turn rate unused.
/// * `sample` - The reference at the arc length the pose is measured
///   against, as [`crate::mpc::reference::ReferencePath::sample_at`]
///   returns it.
///
/// # Returns
///
/// The three errors, in meters and radians.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when the state or the sample carries a
/// NaN or an infinity.
pub fn path_errors(state: VehicleState, sample: PathSample) -> Result<PathErrors, Error> {
    state.check()?;
    check_sample(sample)?;

    let (sin_reference, cos_reference) = sample.heading.sin_cos();
    let offset_x = state.x - sample.x;
    let offset_y = state.y - sample.y;
    Ok(PathErrors {
        contour: (-offset_x).mul_add(sin_reference, offset_y * cos_reference),
        lag: offset_x.mul_add(cos_reference, offset_y * sin_reference),
        heading: angle_difference(state.heading, sample.heading)?,
    })
}

/// Linearizes the three errors in the state and in the arc length.
///
/// The position gradients are the reference tangent and its left normal,
/// which is what splitting an offset along a frame gives. The arc-length
/// gradients are the part worth reading twice, because advancing `s`
/// moves the reference point *and* rotates the frame the error is
/// measured in, and the rotation is what produces the two curvature
/// cross terms:
///
/// ```text
/// d contour / ds = -K * lag
/// d lag     / ds = -1 + K * contour
/// d heading / ds = -K
/// ```
///
/// The `-1` is the reference point sliding out from under the vehicle at
/// unit rate, and everything multiplied by `K` is the frame turning. On a
/// straight reference both cross terms vanish and the lag error falls at
/// exactly one meter per meter of progress.
///
/// # Arguments
///
/// * `state` - The pose to linearize about.
/// * `arc_length` - The arc length the sample was taken at, meters.
/// * `sample` - The reference there, including its curvature.
///
/// # Returns
///
/// One [`ErrorExpansion`] per error, each carrying the constant that
/// makes its affine form exact at this point.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when the state, the arc length, or the
/// sample carries a NaN or an infinity.
pub fn linearize_path_errors(
    state: VehicleState,
    arc_length: f64,
    sample: PathSample,
) -> Result<PathErrorExpansion, Error> {
    check_finite("arc length", arc_length)?;
    let errors = path_errors(state, sample)?;

    let (sin_reference, cos_reference) = sample.heading.sin_cos();
    let curvature = sample.curvature;

    Ok(PathErrorExpansion {
        contour: ErrorExpansion::about(
            errors.contour,
            [-sin_reference, cos_reference, 0.0, 0.0, 0.0],
            -curvature * errors.lag,
            state,
            arc_length,
        ),
        lag: ErrorExpansion::about(
            errors.lag,
            [cos_reference, sin_reference, 0.0, 0.0, 0.0],
            curvature.mul_add(errors.contour, -1.0),
            state,
            arc_length,
        ),
        heading: ErrorExpansion::about(
            errors.heading,
            [0.0, 0.0, 1.0, 0.0, 0.0],
            -curvature,
            state,
            arc_length,
        ),
    })
}

/// The heading cost the nonlinear formulation minimizes.
///
/// Python writes it `sin(e)^2 + (1 - cos e)^2`. Expanding gives
/// `sin^2 e + 1 - 2 cos e + cos^2 e`, and the Pythagorean identity
/// collapses that to `2 (1 - cos e)`, so the two expressions are the same
/// function and not an approximation of each other. The test
/// `the_two_forms_of_the_heading_cost_are_one_function` holds the claim
/// down.
///
/// Smooth and periodic, which is why the original used it, and not convex
/// anywhere near half a turn, which is why the quadratic program carries
/// [`quadratic_heading_cost`] instead.
///
/// # Arguments
///
/// * `heading_error` - Heading minus reference heading, radians, wrapped
///   or not.
///
/// # Returns
///
/// A cost in `[0, 4]`, zero when the vehicle points along the reference.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when `heading_error` is NaN or infinite.
pub fn heading_cost(heading_error: f64) -> Result<f64, Error> {
    check_finite("heading error", heading_error)?;
    Ok(2.0 * (1.0 - heading_error.cos()))
}

/// The Gauss-Newton surrogate the quadratic program carries.
///
/// The square of the wrapped error, which a quadratic program can hold
/// directly: a square is positive semidefinite by construction, where the
/// second derivative of [`heading_cost`] is `2 cos e` and turns negative
/// past a quarter turn, leaving the solver with a non-convex objective.
///
/// **This is an approximation, and it is one only near zero.** Both
/// functions have the same value, the same slope, and the same curvature
/// at `e = 0`, and they part company at fourth order: `e^2` exceeds
/// `2 (1 - cos e)` by `e^4 / 12` to leading order. At a tenth of a radian
/// that is a relative disagreement under a thousandth; at half a radian
/// it is about two percent; at half a turn the surrogate charges nearly
/// two and a half times what the original does. A controller holding its
/// heading error inside a few tenths of a radian never leaves the range
/// where they agree, and one that does not has a larger problem than the
/// choice of cost.
///
/// The error is wrapped first, the same way [`angle_difference`] wraps
/// the one [`path_errors`] reports, so that pointing a full turn away
/// from the reference costs nothing rather than costing `(2 pi)^2`.
///
/// # Arguments
///
/// * `heading_error` - Heading minus reference heading, radians, wrapped
///   or not.
///
/// # Returns
///
/// A cost in `[0, pi^2]`, zero when the vehicle points along the
/// reference.
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when `heading_error` is NaN or infinite.
pub fn quadratic_heading_cost(heading_error: f64) -> Result<f64, Error> {
    let wrapped = wrap_angle(heading_error)?;
    Ok(wrapped * wrapped)
}

/// The inner product of a gradient with the values it multiplies.
///
/// Fused per term, which keeps the residual solved for in
/// [`linearize_unicycle`] agreeing with [`unicycle_step`] to the last
/// place rather than to a few of them.
fn dot<const N: usize>(gradient: &[f64; N], values: &[f64; N]) -> f64 {
    gradient
        .iter()
        .zip(values)
        .fold(0.0, |total, (factor, value)| factor.mul_add(*value, total))
}

/// Rejects a named value that is not a real number.
fn check_finite(quantity: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::NotFinite { quantity, value })
    }
}

/// Rejects the first of several named values that is not a real number.
fn check_all<const N: usize>(values: [(&'static str, f64); N]) -> Result<(), Error> {
    for (quantity, value) in values {
        check_finite(quantity, value)?;
    }
    Ok(())
}

/// Rejects a prediction step no forward model could use.
fn check_step(dt: f64) -> Result<(), Error> {
    check_finite("prediction step", dt)?;
    if dt <= 0.0 {
        return Err(Error::OutOfRange {
            quantity: "prediction step",
            value: dt,
            bound: "(0, inf)",
        });
    }
    Ok(())
}

/// Rejects a reference sample that would poison a gradient.
fn check_sample(sample: PathSample) -> Result<(), Error> {
    check_all([
        ("reference x", sample.x),
        ("reference y", sample.y),
        ("reference heading", sample.heading),
        ("reference curvature", sample.curvature),
    ])
}
