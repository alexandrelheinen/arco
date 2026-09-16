//! A* and Dijkstra over anything implementing [`DiscreteMap`].

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use arco_core::Error;
use arco_core::numeric::Finite;
use arco_core::protocols::DiscreteMap;

use crate::failure::{PlanFailure, PlanOutcome};

/// One entry in the open set.
///
/// Ordered by `f` ascending, then by insertion order, so a tie between two
/// equally promising nodes breaks the same way on every run. `Finite`
/// rather than `f64` because the open set is a heap and a heap keyed on a
/// partial order is a silent correctness bug the moment a cost goes to
/// NaN, which `FR-SAFE-05` exists to prevent.
#[derive(Debug, PartialEq, Eq)]
struct Candidate<N> {
    estimated_total: Finite,
    sequence: usize,
    node: N,
}

impl<N: Eq> Ord for Candidate<N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Reversed, because BinaryHeap is a max-heap and a search wants
        // the cheapest node.
        other
            .estimated_total
            .cmp(&self.estimated_total)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl<N: Eq> PartialOrd for Candidate<N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// How a search should behave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchOptions {
    /// Maximum nodes to expand before giving up.
    ///
    /// `FR-SAFE-02`. A planner without a bound is an unbounded loop, and
    /// exhausting the bound is reported distinctly from finding no path.
    pub max_expansions: usize,
    /// Whether to consult the map's heuristic.
    ///
    /// Turning it off makes the search Dijkstra, which is the oracle
    /// `FR-INV-06` compares against.
    pub use_heuristic: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_expansions: 1_000_000,
            use_heuristic: true,
        }
    }
}

/// Searches `map` from `start` to `goal`.
///
/// With `use_heuristic` set and an admissible heuristic, this is A*; with
/// it clear, it is Dijkstra. Both return the same cost on the same map,
/// which is what `FR-INV-06` asserts and what makes one a free oracle for
/// the other.
///
/// A start or goal the map does not contain is reported as
/// [`PlanFailure::StartOutsideMap`] or [`PlanFailure::GoalOutsideMap`]
/// rather than searched for.
///
/// # Errors
///
/// Propagates whatever the map's own `distance` or `heuristic` returns,
/// which includes a non-finite edge cost.
pub fn search<M>(
    map: &M,
    start: M::Node,
    goal: M::Node,
    options: SearchOptions,
) -> Result<PlanOutcome<M::Node>, Error>
where
    M: DiscreteMap,
    M::Node: Ord,
{
    // FR-INV-08. A node off the edge of the map is a different answer
    // from a node the search could not reach, and collapsing the two
    // sends a caller looking for a route that was never askable.
    if !map.contains(start) {
        return Ok(PlanOutcome::Failed {
            reason: PlanFailure::StartOutsideMap,
            expanded: 0,
        });
    }
    if !map.contains(goal) {
        return Ok(PlanOutcome::Failed {
            reason: PlanFailure::GoalOutsideMap,
            expanded: 0,
        });
    }

    let mut open = BinaryHeap::new();
    let mut best_cost: BTreeMap<M::Node, f64> = BTreeMap::new();
    let mut came_from: BTreeMap<M::Node, M::Node> = BTreeMap::new();
    let mut closed: BTreeSet<M::Node> = BTreeSet::new();
    let mut sequence = 0_usize;
    let mut expanded = 0_usize;

    let start_estimate = if options.use_heuristic {
        map.heuristic(start, goal)?
    } else {
        0.0
    };
    open.push(Candidate {
        estimated_total: Finite::new("heuristic", start_estimate)?,
        sequence,
        node: start,
    });
    best_cost.insert(start, 0.0);

    while let Some(candidate) = open.pop() {
        if !closed.insert(candidate.node) {
            continue;
        }
        expanded = expanded.saturating_add(1);

        if candidate.node == goal {
            let cost = best_cost.get(&goal).copied().unwrap_or(0.0);
            return Ok(PlanOutcome::Found {
                path: reconstruct(&came_from, goal),
                cost,
                expanded,
            });
        }

        if expanded >= options.max_expansions {
            return Ok(PlanOutcome::Failed {
                reason: PlanFailure::BudgetExhausted,
                expanded,
            });
        }

        let reached = best_cost
            .get(&candidate.node)
            .copied()
            .unwrap_or(f64::INFINITY);
        for neighbor in map.neighbors(candidate.node) {
            if closed.contains(&neighbor) {
                continue;
            }
            let step = map.distance(candidate.node, neighbor)?;
            let tentative = reached + step;
            let known = best_cost.get(&neighbor).copied().unwrap_or(f64::INFINITY);
            if tentative >= known {
                continue;
            }

            best_cost.insert(neighbor, tentative);
            came_from.insert(neighbor, candidate.node);
            let estimate = if options.use_heuristic {
                map.heuristic(neighbor, goal)?
            } else {
                0.0
            };
            sequence = sequence.saturating_add(1);
            open.push(Candidate {
                estimated_total: Finite::new("estimated total cost", tentative + estimate)?,
                sequence,
                node: neighbor,
            });
        }
    }

    Ok(PlanOutcome::Failed {
        reason: PlanFailure::Unreachable,
        expanded,
    })
}

/// Walks the predecessor map back from the goal.
fn reconstruct<N: Ord + Copy>(came_from: &BTreeMap<N, N>, goal: N) -> Vec<N> {
    let mut path = vec![goal];
    let mut node = goal;
    // Bounded by the number of nodes that have a predecessor, so a cycle
    // in the map cannot make this run forever.
    for _ in 0..=came_from.len() {
        let Some(&previous) = came_from.get(&node) else {
            break;
        };
        path.push(previous);
        node = previous;
    }
    path.reverse();
    path
}
