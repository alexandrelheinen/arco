//! The search tree both sampling planners grow, and how a path leaves it.
//!
//! RRT* and SST differ in which node they extend and which they keep, not
//! in how they store the tree or how they turn a leaf into a path, so
//! those two things live here rather than twice.

use std::collections::BTreeMap;

use arco_core::Error;
use arco_core::protocols::{Occupancy, PlannerCost};

use crate::failure::PlanOutcome;

use super::policy::{CostPolicy, SegmentPolicy};

/// A tree of states, stored as parallel arrays.
///
/// Parallel arrays rather than nodes holding children: every inner-loop
/// query is a scan over all states, and one contiguous array of positions
/// is what makes that scan cheap.
#[derive(Debug)]
pub(crate) struct Tree {
    states: Vec<Vec<f64>>,
    parents: Vec<Option<usize>>,
    costs: Vec<f64>,
    first_child: Vec<Option<usize>>,
    next_sibling: Vec<Option<usize>>,
    pending: Vec<usize>,
}

impl Tree {
    /// Builds a tree holding only `root`, reserving room for `capacity`.
    ///
    /// The capacity is the planner's sample budget, so the tree allocates
    /// once at its stated bound instead of growing under the loop.
    pub(crate) fn with_root(root: Vec<f64>, capacity: usize) -> Self {
        let mut states = Vec::with_capacity(capacity);
        let mut parents = Vec::with_capacity(capacity);
        let mut costs = Vec::with_capacity(capacity);
        let mut first_child = Vec::with_capacity(capacity);
        let mut next_sibling = Vec::with_capacity(capacity);
        states.push(root);
        parents.push(None);
        costs.push(0.0);
        first_child.push(None);
        next_sibling.push(None);
        Self {
            states,
            parents,
            costs,
            first_child,
            next_sibling,
            pending: Vec::new(),
        }
    }

    /// The state at `index`, or an empty slice when there is none.
    pub(crate) fn state(&self, index: usize) -> &[f64] {
        self.states.get(index).map_or(&[], Vec::as_slice)
    }

    /// The cost from the root to `index`, infinite when there is none.
    ///
    /// Infinity rather than zero, so an absent node loses every
    /// comparison a search makes instead of winning all of them.
    pub(crate) fn cost(&self, index: usize) -> f64 {
        self.costs.get(index).copied().unwrap_or(f64::INFINITY)
    }

    /// How many states the tree holds.
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }

    /// Every state, in insertion order.
    pub(crate) fn states(&self) -> &[Vec<f64>] {
        &self.states
    }

    /// The parent of `index`, if it has one.
    pub(crate) fn parent(&self, index: usize) -> Option<usize> {
        self.parents.get(index).copied().flatten()
    }

    /// Adds a state under `parent` and returns its index.
    pub(crate) fn push(&mut self, state: Vec<f64>, parent: usize, cost: f64) -> usize {
        let index = self.states.len();
        self.states.push(state);
        self.parents.push(Some(parent));
        self.costs.push(cost);
        self.first_child.push(None);
        self.next_sibling
            .push(self.first_child.get(parent).copied().flatten());
        if let Some(slot) = self.first_child.get_mut(parent) {
            *slot = Some(index);
        }
        index
    }

    /// Repoints `index` at `parent`, and carries the saving downward.
    ///
    /// Everything below `index` reaches the root through it, so lowering
    /// its cost lowers theirs by the same amount. Leaving them stale is
    /// the usual shortcut and it costs more than it saves: a later
    /// rewiring compares against a cost that is too high, accepts a
    /// change that does not improve the path, and the cost a longer run
    /// reports can come out above the cost a shorter one did. That is
    /// exactly what `FR-INV-07` forbids.
    ///
    /// The descent uses an explicit stack, per `FR-SAFE-03`, and is
    /// bounded by the node count so a corrupted link ends it.
    pub(crate) fn repoint(&mut self, index: usize, parent: usize, cost: f64) {
        self.detach(index);
        let head = self.first_child.get(parent).copied().flatten();
        if let Some(slot) = self.next_sibling.get_mut(index) {
            *slot = head;
        }
        if let Some(slot) = self.first_child.get_mut(parent) {
            *slot = Some(index);
        }
        if let Some(slot) = self.parents.get_mut(index) {
            *slot = Some(parent);
        }

        let saving = cost - self.cost(index);
        if let Some(slot) = self.costs.get_mut(index) {
            *slot = cost;
        }
        if saving == 0.0 {
            return;
        }

        let mut pending = core::mem::take(&mut self.pending);
        pending.clear();
        pending.extend(self.first_child.get(index).copied().flatten());
        for _ in 0..self.states.len() {
            let Some(node) = pending.pop() else { break };
            if let Some(slot) = self.costs.get_mut(node) {
                *slot += saving;
            }
            pending.extend(self.first_child.get(node).copied().flatten());
            pending.extend(self.next_sibling.get(node).copied().flatten());
        }
        self.pending = pending;
    }

    /// Unlinks `index` from the sibling list of its current parent.
    fn detach(&mut self, index: usize) {
        let Some(parent) = self.parent(index) else {
            return;
        };
        let following = self.next_sibling.get(index).copied().flatten();
        if self.first_child.get(parent).copied().flatten() == Some(index) {
            if let Some(slot) = self.first_child.get_mut(parent) {
                *slot = following;
            }
            return;
        }
        let mut sibling = self.first_child.get(parent).copied().flatten();
        for _ in 0..self.states.len() {
            let Some(current) = sibling else { break };
            if self.next_sibling.get(current).copied().flatten() == Some(index) {
                if let Some(slot) = self.next_sibling.get_mut(current) {
                    *slot = following;
                }
                return;
            }
            sibling = self.next_sibling.get(current).copied().flatten();
        }
    }

    /// Walks back from `leaf` to the root, root first.
    ///
    /// The walk is bounded by the tree size rather than by reaching the
    /// root, so a parent link corrupted into a cycle ends the walk
    /// instead of hanging the planner.
    pub(crate) fn trace(&self, leaf: usize) -> Vec<Vec<f64>> {
        let mut path = Vec::new();
        let mut node = Some(leaf);
        for _ in 0..=self.states.len() {
            let Some(index) = node else { break };
            path.push(self.state(index).to_vec());
            node = self.parent(index);
        }
        path.reverse();
        path
    }
}

/// The tree a sampling planner grew, for a caller that wants to draw it.
///
/// A plain snapshot rather than a handle: the planner has finished by the
/// time this exists, so there is nothing to keep borrowing and a caller
/// that wants to keep it can.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlannerTree {
    states: Vec<Vec<f64>>,
    parents: Vec<Option<usize>>,
}

impl PlannerTree {
    /// The states, in the order the planner added them.
    #[must_use]
    pub fn states(&self) -> &[Vec<f64>] {
        &self.states
    }

    /// The parent of each state by index, `None` for a root.
    #[must_use]
    pub fn parents(&self) -> &[Option<usize>] {
        &self.parents
    }

    /// How many states the tree holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the tree is empty, which only a failed query produces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl Tree {
    /// A snapshot of the whole tree.
    pub(crate) fn snapshot(&self) -> PlannerTree {
        PlannerTree {
            states: self.states.clone(),
            parents: self.parents.clone(),
        }
    }

    /// A snapshot of `kept` only, renumbered from zero.
    ///
    /// SST retires a node when a cheaper one takes its region, and what a
    /// caller draws is the surviving tree rather than everything that was
    /// ever tried. A parent that did not survive becomes `None`, so the
    /// result is a forest of the nodes that are still growable rather than
    /// a tree with dangling indices.
    pub(crate) fn snapshot_of(&self, kept: impl IntoIterator<Item = usize>) -> PlannerTree {
        let ordered: Vec<usize> = {
            let mut indices: Vec<usize> = kept.into_iter().collect();
            indices.sort_unstable();
            indices.dedup();
            indices
        };
        let renumbered: BTreeMap<usize, usize> = ordered
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();

        let mut states = Vec::with_capacity(ordered.len());
        let mut parents = Vec::with_capacity(ordered.len());
        for &old in &ordered {
            states.push(self.state(old).to_vec());
            parents.push(
                self.parent(old)
                    .and_then(|parent| renumbered.get(&parent).copied()),
            );
        }
        PlannerTree { states, parents }
    }
}

/// Appends the exact goal when that last segment is free, and totals cost.
///
/// A sampling planner stops within a tolerance of the goal rather than on
/// it, so the final hop is a segment nobody has checked yet. A large
/// tolerance makes that hop long enough to cross an obstacle, which is
/// why it is checked rather than assumed.
pub(crate) fn close_path<O: Occupancy>(
    mut path: Vec<Vec<f64>>,
    goal: &[f64],
    cost: &CostPolicy,
    segments: &SegmentPolicy<O>,
    expanded: usize,
) -> Result<PlanOutcome<Vec<f64>>, Error> {
    if let Some(last) = path.last()
        && cost.distance(last, goal)? > 0.0
        && segments.is_segment_free(last, goal)?
    {
        path.push(goal.to_vec());
    }

    let mut total = 0.0;
    for pair in path.windows(2) {
        if let [from, to] = pair {
            total += cost.distance(from, to)?;
        }
    }
    Ok(PlanOutcome::Found {
        path,
        cost: total,
        expanded,
    })
}
