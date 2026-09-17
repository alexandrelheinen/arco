//! Building a convex program and handing it to Clarabel.
//!
//! The solver wants compressed sparse columns, and the problems here are
//! assembled block by block as a horizon is walked. This module is the
//! join: entries go in as triplets in whatever order the caller produces
//! them, and come out as the two matrices the solver reads.

use arco_core::Error;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};

/// Why a solve did not produce a usable answer.
///
/// `FR-MPC-04`. A controller that cannot tell an infeasible problem from
/// one the solver ran out of time on cannot decide whether to relax the
/// problem or to try again, and a bare failure flag forces it to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveFailure {
    /// No point satisfies the constraints.
    ///
    /// Relaxing something is the only way forward; another attempt on the
    /// same problem returns the same answer.
    Infeasible,
    /// The objective is unbounded below on the feasible set.
    ///
    /// A modelling error rather than a runtime condition: some cost term
    /// pays without limit, which means a weight or a sign is wrong.
    Unbounded,
    /// The iteration or time budget ran out first.
    ///
    /// `FR-SAFE-02`: retryable, and distinct from the two above.
    BudgetExhausted,
    /// The solver stopped for a numerical reason.
    Numerical,
}

impl core::fmt::Display for SolveFailure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let reason = match *self {
            Self::Infeasible => "the constraints admit no solution",
            Self::Unbounded => "the objective is unbounded below",
            Self::BudgetExhausted => "the solver budget ran out",
            Self::Numerical => "the solver stopped on a numerical condition",
        };
        formatter.write_str(reason)
    }
}

/// What a solve produced.
#[derive(Debug, Clone, PartialEq)]
pub struct QpSolution {
    /// The primal solution.
    pub variables: Vec<f64>,
    /// The objective value there.
    pub cost: f64,
    /// How many interior-point iterations were taken.
    pub iterations: u32,
    /// Whether the answer is optimal rather than merely acceptable.
    ///
    /// Clarabel reports an almost-solved status when it converged to a
    /// looser tolerance than asked for. That is a usable command and a
    /// worse one, so it is reported rather than folded into success.
    pub exact: bool,
}

/// A matrix under construction, as unordered triplets.
///
/// Duplicates are summed, so a caller adding the same entry from two
/// blocks gets what it meant rather than whichever landed last.
#[derive(Debug, Clone, Default)]
pub struct Triplets {
    entries: Vec<(usize, usize, f64)>,
}

impl Triplets {
    /// An empty builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds `value` at `(row, column)`, skipping an exact zero.
    ///
    /// Skipping zeros is not an optimization: a structural zero in the
    /// sparsity pattern is information the solver's factorization uses,
    /// and storing an explicit one throws it away.
    pub fn push(&mut self, row: usize, column: usize, value: f64) {
        if value != 0.0 {
            self.entries.push((row, column, value));
        }
    }

    /// How many entries have been added.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Assembles a `rows` by `columns` matrix.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when an entry falls outside the
    /// stated shape, or [`Error::NotFinite`] when one is not a real
    /// number, either of which would otherwise reach the solver as
    /// corrupt memory or as a NaN in the factorization.
    pub fn build(mut self, rows: usize, columns: usize) -> Result<CscMatrix<f64>, Error> {
        for &(row, column, value) in &self.entries {
            if row >= rows || column >= columns {
                return Err(Error::OutOfRange {
                    quantity: "matrix entry index",
                    value: f64::from(u32::try_from(row.max(column)).unwrap_or(u32::MAX)),
                    bound: "inside the declared shape",
                });
            }
            if !value.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "matrix entry",
                    value,
                });
            }
        }

        self.entries
            .sort_unstable_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));

        let mut column_pointers = vec![0_usize; columns.saturating_add(1)];
        let mut row_indices: Vec<usize> = Vec::with_capacity(self.entries.len());
        let mut values: Vec<f64> = Vec::with_capacity(self.entries.len());

        let mut previous: Option<(usize, usize)> = None;
        for &(row, column, value) in &self.entries {
            if previous == Some((row, column)) {
                if let Some(slot) = values.last_mut() {
                    *slot += value;
                }
                continue;
            }
            row_indices.push(row);
            values.push(value);
            previous = Some((row, column));
            if let Some(slot) = column_pointers.get_mut(column.saturating_add(1)) {
                *slot = slot.saturating_add(1);
            }
        }
        for index in 1..column_pointers.len() {
            let carried = column_pointers
                .get(index.saturating_sub(1))
                .copied()
                .unwrap_or(0);
            if let Some(slot) = column_pointers.get_mut(index) {
                *slot = slot.saturating_add(carried);
            }
        }

        Ok(CscMatrix::new(
            rows,
            columns,
            column_pointers,
            row_indices,
            values,
        ))
    }
}

/// A convex quadratic program, in the shape Clarabel takes.
///
/// Minimize one half `x' P x + q' x` subject to `A x + s = b` with the
/// first `equality_rows` entries of `s` held at zero and the rest
/// non-negative. Equalities come first because the cone list is ordered
/// and splitting them anywhere else would mean two cones of each kind.
#[derive(Debug)]
pub struct QpProblem {
    /// Upper triangle of the objective's quadratic term.
    pub objective: Triplets,
    /// The objective's linear term, one entry per variable.
    pub gradient: Vec<f64>,
    /// The constraint matrix, equality rows first.
    pub constraints: Triplets,
    /// The constraint right-hand side.
    pub bounds: Vec<f64>,
    /// How many leading rows are equalities.
    pub equality_rows: usize,
    /// The iteration budget, per `FR-SAFE-02`.
    pub max_iterations: u32,
}

impl QpProblem {
    /// Solves the program.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the pieces disagree about
    /// how many variables or rows there are, and otherwise whatever
    /// assembling the matrices returns.
    pub fn solve(self) -> Result<Result<QpSolution, SolveFailure>, Error> {
        let variables = self.gradient.len();
        let rows = self.bounds.len();
        if self.equality_rows > rows {
            return Err(Error::DimensionMismatch {
                quantity: "equality rows",
                expected: rows,
                actual: self.equality_rows,
            });
        }
        for &value in self.gradient.iter().chain(&self.bounds) {
            if !value.is_finite() {
                return Err(Error::NotFinite {
                    quantity: "program coefficient",
                    value,
                });
            }
        }

        let quadratic = self.objective.build(variables, variables)?;
        let constraints = self.constraints.build(rows, variables)?;

        let inequality_rows = rows.saturating_sub(self.equality_rows);
        let mut cones = Vec::with_capacity(2);
        if self.equality_rows > 0 {
            cones.push(SupportedConeT::ZeroConeT(self.equality_rows));
        }
        if inequality_rows > 0 {
            cones.push(SupportedConeT::NonnegativeConeT(inequality_rows));
        }

        let settings = DefaultSettings {
            verbose: false,
            max_iter: self.max_iterations,
            ..DefaultSettings::default()
        };
        let Ok(mut solver) = DefaultSolver::new(
            &quadratic,
            &self.gradient,
            &constraints,
            &self.bounds,
            &cones,
            settings,
        ) else {
            // Clarabel rejects a problem it cannot even set up, which for
            // these programs means the shapes disagree.
            return Ok(Err(SolveFailure::Numerical));
        };
        solver.solve();

        let status = solver.solution.status;
        let exact = matches!(status, SolverStatus::Solved);
        if exact || matches!(status, SolverStatus::AlmostSolved) {
            return Ok(Ok(QpSolution {
                variables: solver.solution.x.clone(),
                cost: solver.solution.obj_val,
                iterations: solver.info.iterations,
                exact,
            }));
        }

        Ok(Err(match status {
            SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
                SolveFailure::Infeasible
            }
            SolverStatus::DualInfeasible | SolverStatus::AlmostDualInfeasible => {
                SolveFailure::Unbounded
            }
            SolverStatus::MaxIterations | SolverStatus::MaxTime => SolveFailure::BudgetExhausted,
            _ => SolveFailure::Numerical,
        }))
    }
}
