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
    turn_penalty: u8,
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
            .then_with(|| other.turn_penalty.cmp(&self.turn_penalty))
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
    /// Whether to break a tie toward the path that turns less.
    ///
    /// Only ever separates two paths of equal cost, so it cannot change
    /// which cost the search returns. What it changes is which of several
    /// equally cheap paths comes back: on a uniform grid a straight line
    /// and a staircase cost the same, and without this the staircase is as
    /// likely to win.
    pub prefer_straight: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_expansions: 1_000_000,
            use_heuristic: true,
            prefer_straight: true,
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
    let mut expanded_order = Vec::new();
    let mut came_from = BTreeMap::new();
    search_inner(
        map,
        start,
        goal,
        options,
        &mut expanded_order,
        &mut came_from,
    )
}

/// The one search both entry points run.
fn search_inner<M>(
    map: &M,
    start: M::Node,
    goal: M::Node,
    options: SearchOptions,
    expanded_order: &mut Vec<M::Node>,
    came_from: &mut BTreeMap<M::Node, M::Node>,
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
        turn_penalty: 0,
        sequence,
        node: start,
    });
    best_cost.insert(start, 0.0);

    while let Some(candidate) = open.pop() {
        if !closed.insert(candidate.node) {
            continue;
        }
        expanded = expanded.saturating_add(1);
        expanded_order.push(candidate.node);

        if candidate.node == goal {
            let cost = best_cost.get(&goal).copied().unwrap_or(0.0);
            return Ok(PlanOutcome::Found {
                path: reconstruct(came_from, goal),
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
            let turn_penalty = if options.prefer_straight {
                map.turn_penalty(
                    came_from.get(&candidate.node).copied(),
                    candidate.node,
                    neighbor,
                )
            } else {
                0
            };
            sequence = sequence.saturating_add(1);
            open.push(Candidate {
                estimated_total: Finite::new("estimated total cost", tentative + estimate)?,
                turn_penalty,
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

/// What a search did on the way to its answer.
///
/// Two things a visualizer needs and a plain outcome throws away: the
/// order nodes came off the open set, which is what makes a search
/// watchable, and the predecessor map, which is the tree behind the one
/// path that was returned.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchDiagnostics<N> {
    /// Nodes in the order they were expanded.
    pub expanded_order: Vec<N>,
    /// Which node each node was first reached from.
    pub came_from: BTreeMap<N, N>,
}

/// A search result carrying the work that produced it.
pub type DiagnosedSearch<N> = (PlanOutcome<N>, SearchDiagnostics<N>);

/// Searches `map`, keeping the expansion order and predecessor map.
///
/// The same search as [`search`], which calls this and discards the
/// second half. Recording costs one push per expansion, so a caller on a
/// control budget uses [`search`] and one drawing the result uses this.
///
/// # Errors
///
/// As [`search`].
pub fn search_with_diagnostics<M>(
    map: &M,
    start: M::Node,
    goal: M::Node,
    options: SearchOptions,
) -> Result<DiagnosedSearch<M::Node>, Error>
where
    M: DiscreteMap,
    M::Node: Ord,
{
    let mut expanded_order = Vec::new();
    let mut came_from = BTreeMap::new();
    let outcome = search_inner(
        map,
        start,
        goal,
        options,
        &mut expanded_order,
        &mut came_from,
    )?;
    Ok((
        outcome,
        SearchDiagnostics {
            expanded_order,
            came_from,
        },
    ))
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
