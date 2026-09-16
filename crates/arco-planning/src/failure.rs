//! Why a planner declined to produce a path.
//!
//! `FR-INV-08`. "Returned no path" is not a diagnosis, and a caller
//! embedding ARCO in a machine has to decide between retrying with a
//! larger budget, moving the goal, and giving up. The enumeration is
//! closed and every variant is reachable from the test suite.

use core::fmt;

/// Why no path was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanFailure {
    /// The start state is inside an obstacle.
    StartOccupied,
    /// The goal state is inside an obstacle.
    GoalOccupied,
    /// The start state lies outside the map.
    StartOutsideMap,
    /// The goal state lies outside the map.
    GoalOutsideMap,
    /// The search completed and the goal is not reachable.
    ///
    /// Distinct from running out of budget: this is an answer, and
    /// retrying with a larger budget will not change it.
    Unreachable,
    /// The budget ran out before the search completed.
    ///
    /// `FR-SAFE-02`. A larger budget may succeed, which is precisely the
    /// distinction a caller needs and the one a bare `None` destroys.
    BudgetExhausted,
}

impl PlanFailure {
    /// Whether a larger budget might change this answer.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::BudgetExhausted)
    }
}

impl fmt::Display for PlanFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match *self {
            Self::StartOccupied => "the start state is occupied",
            Self::GoalOccupied => "the goal state is occupied",
            Self::StartOutsideMap => "the start state is outside the map",
            Self::GoalOutsideMap => "the goal state is outside the map",
            Self::Unreachable => "no path exists",
            Self::BudgetExhausted => "the search budget ran out",
        };
        formatter.write_str(reason)
    }
}

/// What a search produced, or why it did not.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanOutcome<T> {
    /// A path, and what it cost.
    Found {
        /// The path, start first and goal last.
        path: Vec<T>,
        /// The path's total cost under the map's own metric.
        cost: f64,
        /// How many nodes the search expanded.
        expanded: usize,
    },
    /// No path, and why.
    Failed {
        /// The reason.
        reason: PlanFailure,
        /// How many nodes the search expanded before giving up.
        expanded: usize,
    },
}

impl<T> PlanOutcome<T> {
    /// The path, if one was found.
    #[must_use]
    pub fn path(&self) -> Option<&[T]> {
        match self {
            Self::Found { path, .. } => Some(path),
            Self::Failed { .. } => None,
        }
    }

    /// The cost, if a path was found.
    #[must_use]
    pub const fn cost(&self) -> Option<f64> {
        match *self {
            Self::Found { cost, .. } => Some(cost),
            Self::Failed { .. } => None,
        }
    }

    /// The reason, if none was.
    #[must_use]
    pub const fn failure(&self) -> Option<PlanFailure> {
        match *self {
            Self::Found { .. } => None,
            Self::Failed { reason, .. } => Some(reason),
        }
    }

    /// How many nodes the search expanded either way.
    #[must_use]
    pub const fn expanded(&self) -> usize {
        match *self {
            Self::Found { expanded, .. } | Self::Failed { expanded, .. } => expanded,
        }
    }
}
