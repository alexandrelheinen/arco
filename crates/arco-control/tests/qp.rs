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

//! The solver layer, on problems whose answers are known in closed form.
//!
//! Step two of phase 7: prove the Clarabel wrapper on textbook problems
//! before pointing a contouring controller at it. A failure here would
//! otherwise surface as a tracking error and be blamed on the
//! linearization.

use arco_control::mpc::qp::{QpProblem, SolveFailure, Triplets};

/// The unconstrained minimum of `(x - a)^2 + (y - b)^2` is `(a, b)`.
fn recentered(target: (f64, f64)) -> QpProblem {
    let mut objective = Triplets::new();
    objective.push(0, 0, 2.0);
    objective.push(1, 1, 2.0);
    QpProblem {
        objective,
        gradient: vec![-2.0 * target.0, -2.0 * target.1],
        constraints: Triplets::new(),
        bounds: Vec::new(),
        equality_rows: 0,
        max_iterations: 200,
    }
}

#[test]
fn an_unconstrained_quadratic_lands_on_its_minimum() {
    let solution = recentered((3.0, -2.0))
        .solve()
        .expect("a well-formed program")
        .expect("an unconstrained quadratic is always solvable");
    assert!((solution.variables[0] - 3.0).abs() < 1e-6, "{solution:?}");
    assert!((solution.variables[1] + 2.0).abs() < 1e-6, "{solution:?}");
    assert!(solution.exact);
}

#[test]
fn an_equality_constraint_is_met_exactly() {
    // Minimize x^2 + y^2 subject to x + y = 2. The answer is (1, 1) and
    // the constraint has to hold to solver tolerance, not approximately.
    let mut objective = Triplets::new();
    objective.push(0, 0, 2.0);
    objective.push(1, 1, 2.0);
    let mut constraints = Triplets::new();
    constraints.push(0, 0, 1.0);
    constraints.push(0, 1, 1.0);

    let solution = QpProblem {
        objective,
        gradient: vec![0.0, 0.0],
        constraints,
        bounds: vec![2.0],
        equality_rows: 1,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect("the constraint set is not empty");

    assert!((solution.variables[0] - 1.0).abs() < 1e-6, "{solution:?}");
    assert!((solution.variables[1] - 1.0).abs() < 1e-6, "{solution:?}");
    assert!(
        (solution.variables[0] + solution.variables[1] - 2.0).abs() < 1e-8,
        "the equality was only approximately met"
    );
}

#[test]
fn an_inequality_that_binds_holds_on_its_boundary() {
    // Minimize (x - 5)^2 subject to x <= 2, written as `x + s = 2` with a
    // non-negative slack. The minimum sits on the bound.
    let mut objective = Triplets::new();
    objective.push(0, 0, 2.0);
    let mut constraints = Triplets::new();
    constraints.push(0, 0, 1.0);

    let solution = QpProblem {
        objective,
        gradient: vec![-10.0],
        constraints,
        bounds: vec![2.0],
        equality_rows: 0,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect("the constraint set is not empty");
    assert!((solution.variables[0] - 2.0).abs() < 1e-6, "{solution:?}");
}

#[test]
fn an_inequality_that_does_not_bind_leaves_the_minimum_alone() {
    let mut objective = Triplets::new();
    objective.push(0, 0, 2.0);
    let mut constraints = Triplets::new();
    constraints.push(0, 0, 1.0);

    let solution = QpProblem {
        objective,
        gradient: vec![-2.0],
        constraints,
        bounds: vec![10.0],
        equality_rows: 0,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect("the constraint set is not empty");
    assert!((solution.variables[0] - 1.0).abs() < 1e-6, "{solution:?}");
}

#[test]
fn an_empty_constraint_set_is_reported_as_infeasible() {
    // FR-MPC-04. `x <= 1` and `-x <= -3` cannot both hold, and the answer
    // is that relaxing something is the only way forward.
    let mut objective = Triplets::new();
    objective.push(0, 0, 2.0);
    let mut constraints = Triplets::new();
    constraints.push(0, 0, 1.0);
    constraints.push(1, 0, -1.0);

    let failure = QpProblem {
        objective,
        gradient: vec![0.0],
        constraints,
        bounds: vec![1.0, -3.0],
        equality_rows: 0,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect_err("these constraints contradict each other");
    assert_eq!(failure, SolveFailure::Infeasible);
    assert!(failure.to_string().contains("no solution"));
}

#[test]
fn an_objective_with_no_floor_is_reported_as_unbounded() {
    // A linear objective pushing down a direction nothing constrains. Not
    // a runtime condition: it means a weight or a sign in the model is
    // wrong, and the reason says so.
    let failure = QpProblem {
        objective: Triplets::new(),
        gradient: vec![1.0],
        constraints: Triplets::new(),
        bounds: Vec::new(),
        equality_rows: 0,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect_err("nothing stops this objective falling");
    assert_eq!(failure, SolveFailure::Unbounded);
}

#[test]
fn a_starved_iteration_budget_is_reported_as_retryable() {
    // FR-SAFE-02 reaches the solver too: out of budget and infeasible are
    // different answers, and only one of them is worth retrying. The
    // problem has to be one an interior-point method actually iterates on,
    // so it carries constraints; an unconstrained quadratic is solved in
    // closed form before the first iteration.
    const SIZE: usize = 30;
    let mut objective = Triplets::new();
    let mut constraints = Triplets::new();
    for index in 0..SIZE {
        objective.push(index, index, 2.0);
        // `-x_i <= -1`, which pushes every variable away from the
        // unconstrained minimum at the origin.
        constraints.push(index, index, -1.0);
    }

    let starved = QpProblem {
        objective,
        gradient: vec![0.0; SIZE],
        constraints,
        bounds: vec![-1.0; SIZE],
        equality_rows: 0,
        max_iterations: 1,
    }
    .solve()
    .expect("a well-formed program")
    .expect_err("one iteration cannot converge on this");
    assert_eq!(starved, SolveFailure::BudgetExhausted);
    assert!(starved.to_string().contains("budget"));
}

#[test]
fn the_same_problem_converges_once_the_budget_allows_it() {
    // The other half of the distinction above: nothing was wrong with the
    // problem, only with how long it was given.
    const SIZE: usize = 30;
    let mut objective = Triplets::new();
    let mut constraints = Triplets::new();
    for index in 0..SIZE {
        objective.push(index, index, 2.0);
        constraints.push(index, index, -1.0);
    }

    let solution = QpProblem {
        objective,
        gradient: vec![0.0; SIZE],
        constraints,
        bounds: vec![-1.0; SIZE],
        equality_rows: 0,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect("the constraint set is not empty");
    for (index, &value) in solution.variables.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < 1e-5,
            "variable {index} settled at {value} rather than on its bound"
        );
    }
    assert!(solution.iterations > 1);
}

#[test]
fn a_matrix_entry_outside_its_shape_is_refused() {
    let mut constraints = Triplets::new();
    constraints.push(5, 0, 1.0);
    assert!(
        QpProblem {
            objective: Triplets::new(),
            gradient: vec![0.0],
            constraints,
            bounds: vec![1.0],
            equality_rows: 0,
            max_iterations: 10,
        }
        .solve()
        .is_err()
    );
}

#[test]
fn a_non_finite_coefficient_is_refused_before_the_solver_sees_it() {
    // A NaN in the factorization produces a solution that is entirely NaN
    // and a status that says it solved.
    let mut objective = Triplets::new();
    objective.push(0, 0, f64::NAN);
    assert!(
        QpProblem {
            objective,
            gradient: vec![0.0],
            constraints: Triplets::new(),
            bounds: Vec::new(),
            equality_rows: 0,
            max_iterations: 10,
        }
        .solve()
        .is_err()
    );

    assert!(
        QpProblem {
            objective: Triplets::new(),
            gradient: vec![f64::INFINITY],
            constraints: Triplets::new(),
            bounds: Vec::new(),
            equality_rows: 0,
            max_iterations: 10,
        }
        .solve()
        .is_err()
    );
}

#[test]
fn duplicate_entries_are_summed_rather_than_overwritten() {
    // A caller assembling a block-structured problem writes the same entry
    // from two blocks, and the two contributions are both meant.
    let mut objective = Triplets::new();
    objective.push(0, 0, 1.0);
    objective.push(0, 0, 1.0);
    assert_eq!(objective.len(), 2);

    let solution = QpProblem {
        objective,
        gradient: vec![-2.0],
        constraints: Triplets::new(),
        bounds: Vec::new(),
        equality_rows: 0,
        max_iterations: 200,
    }
    .solve()
    .expect("a well-formed program")
    .expect("an unconstrained quadratic is always solvable");
    // Minimizing x^2 - 2x rather than (1/2) x^2 - 2x: the minimum is at 1,
    // not at 2.
    assert!((solution.variables[0] - 1.0).abs() < 1e-6, "{solution:?}");
}

#[test]
fn an_exact_zero_is_not_stored() {
    let mut triplets = Triplets::new();
    assert!(triplets.is_empty());
    triplets.push(0, 0, 0.0);
    assert!(triplets.is_empty());
    triplets.push(0, 0, 1.0);
    assert_eq!(triplets.len(), 1);
}
