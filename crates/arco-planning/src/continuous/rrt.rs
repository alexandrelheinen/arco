//! RRT*, the asymptotically optimal sampling planner.

use arco_core::Error;
use arco_core::geometry::{euclidean_distance, require_dimension, require_finite};
use arco_core::protocols::Occupancy;
use arco_core::rng::Pcg64;

use crate::failure::{PlanFailure, PlanOutcome};

use super::policy::{SamplerPolicy, SegmentPolicy, SteererPolicy};

/// How an RRT* run should behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrtSettings {
    /// Maximum samples to draw before giving up.
    ///
    /// `FR-SAFE-02`. Exhausting this is reported as retryable, distinct
    /// from proving the goal unreachable, which sampling never does.
    pub max_samples: usize,
    /// How close to the goal counts as arriving, meters.
    pub goal_tolerance: f64,
    /// How often to sample the goal itself rather than the space.
    pub goal_bias: f64,
    /// Scale of the rewiring neighborhood.
    ///
    /// The radius shrinks as `radius_scale * (ln(n) / n)^(1/d)`, which is
    /// the schedule that makes RRT* asymptotically optimal rather than
    /// merely convergent.
    pub radius_scale: f64,
    /// Whether to stop at the first solution rather than improving it.
    ///
    /// Stopping early forfeits the optimality the rewiring buys, so
    /// `FR-INV-07` only holds across iterations when this is off.
    pub early_stop: bool,
}

impl Default for RrtSettings {
    fn default() -> Self {
        Self {
            max_samples: 2000,
            goal_tolerance: 1.0,
            goal_bias: 0.05,
            radius_scale: 2.0,
            early_stop: true,
        }
    }
}

/// An RRT* planner over a continuous space.
///
/// Holds its policies by value so the inner loop calls them directly. See
/// ADR-004 for why that matters more than it looks.
#[derive(Debug)]
pub struct RrtPlanner<O> {
    sampler: SamplerPolicy,
    steerer: SteererPolicy,
    segments: SegmentPolicy<O>,
    settings: RrtSettings,
}

/// The tree, stored as parallel arrays.
struct Tree {
    states: Vec<Vec<f64>>,
    parents: Vec<Option<usize>>,
    costs: Vec<f64>,
}

impl Tree {
    fn with_root(root: Vec<f64>, capacity: usize) -> Self {
        let mut states = Vec::with_capacity(capacity);
        let mut parents = Vec::with_capacity(capacity);
        let mut costs = Vec::with_capacity(capacity);
        states.push(root);
        parents.push(None);
        costs.push(0.0);
        Self {
            states,
            parents,
            costs,
        }
    }

    fn state(&self, index: usize) -> &[f64] {
        self.states.get(index).map_or(&[], Vec::as_slice)
    }

    fn cost(&self, index: usize) -> f64 {
        self.costs.get(index).copied().unwrap_or(f64::INFINITY)
    }
}

impl<O: Occupancy> RrtPlanner<O> {
    /// Builds a planner from its policies and settings.
    #[must_use]
    pub const fn new(
        sampler: SamplerPolicy,
        steerer: SteererPolicy,
        segments: SegmentPolicy<O>,
        settings: RrtSettings,
    ) -> Self {
        Self {
            sampler,
            steerer,
            segments,
            settings,
        }
    }

    /// Plans from `start` to `goal`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree with
    /// each other or with the occupancy, or [`Error::NotFinite`] when one
    /// carries a NaN.
    pub fn plan(
        &self,
        start: &[f64],
        goal: &[f64],
        generator: &mut Pcg64,
    ) -> Result<PlanOutcome<Vec<f64>>, Error> {
        require_finite("start", start)?;
        require_finite("goal", goal)?;
        require_dimension("goal", goal, start.len())?;

        if self.occupancy_blocks(start)? {
            return Ok(PlanOutcome::Failed {
                reason: PlanFailure::StartOccupied,
                expanded: 0,
            });
        }
        if self.occupancy_blocks(goal)? {
            return Ok(PlanOutcome::Failed {
                reason: PlanFailure::GoalOccupied,
                expanded: 0,
            });
        }

        let dimension = start.len();
        let mut tree = Tree::with_root(start.to_vec(), self.settings.max_samples);
        let mut best_goal: Option<usize> = None;
        let mut best_goal_cost = f64::INFINITY;
        let mut near = Vec::new();

        for iteration in 0..self.settings.max_samples {
            let target = if generator.next_f64() < self.settings.goal_bias {
                goal.to_vec()
            } else {
                self.sampler.sample(generator)?
            };
            if target.len() != dimension {
                return Err(Error::DimensionMismatch {
                    quantity: "sampled state",
                    expected: dimension,
                    actual: target.len(),
                });
            }

            let Some(nearest) = nearest_index(&tree, &target)? else {
                continue;
            };
            let candidate = self.steerer.steer(tree.state(nearest), &target)?;
            if !self
                .segments
                .is_segment_free(tree.state(nearest), &candidate)?
            {
                continue;
            }

            let radius = self.rewire_radius(tree.states.len(), dimension);
            collect_near(&tree, &candidate, radius, &mut near)?;

            let (added, cost) = self.graft(&mut tree, candidate.clone(), nearest, &near)?;
            self.rewire(&mut tree, added, &candidate, cost, &near)?;

            if euclidean_distance(&candidate, goal)? <= self.settings.goal_tolerance
                && cost < best_goal_cost
            {
                best_goal_cost = cost;
                best_goal = Some(added);
                if self.settings.early_stop {
                    return self.finish(&tree, added, goal, iteration.saturating_add(1));
                }
            }
        }

        match best_goal {
            Some(index) => self.finish(&tree, index, goal, self.settings.max_samples),
            None => Ok(PlanOutcome::Failed {
                // Sampling never proves a goal unreachable, it only runs
                // out of samples, so this is always the retryable answer.
                reason: PlanFailure::BudgetExhausted,
                expanded: self.settings.max_samples,
            }),
        }
    }

    /// Attaches `candidate` to the cheapest reachable parent.
    ///
    /// The nearest node is only the starting guess: RRT* attaches to
    /// whichever node in the neighborhood reaches the candidate most
    /// cheaply, which is half of what makes it asymptotically optimal.
    ///
    /// Returns the new node's index and its cost from the root.
    ///
    /// # Errors
    ///
    /// Propagates whatever the segment policy returns.
    fn graft(
        &self,
        tree: &mut Tree,
        candidate: Vec<f64>,
        nearest: usize,
        near: &[usize],
    ) -> Result<(usize, f64), Error> {
        let mut parent = nearest;
        let mut cost =
            tree.cost(nearest).max(0.0) + euclidean_distance(tree.state(nearest), &candidate)?;

        for &index in near {
            if index == nearest {
                continue;
            }
            let through = tree.cost(index) + euclidean_distance(tree.state(index), &candidate)?;
            if through < cost
                && self
                    .segments
                    .is_segment_free(tree.state(index), &candidate)?
            {
                cost = through;
                parent = index;
            }
        }

        let added = tree.states.len();
        tree.states.push(candidate);
        tree.parents.push(Some(parent));
        tree.costs.push(cost);
        Ok((added, cost))
    }

    /// Repoints any neighbor now cheaper to reach through the new node.
    ///
    /// The other half of asymptotic optimality: without this the tree
    /// keeps whatever parent each node first happened to get, and
    /// FR-INV-07 stops holding.
    ///
    /// # Errors
    ///
    /// Propagates whatever the segment policy returns.
    fn rewire(
        &self,
        tree: &mut Tree,
        added: usize,
        candidate: &[f64],
        cost: f64,
        near: &[usize],
    ) -> Result<(), Error> {
        let parent = tree.parents.get(added).copied().flatten();
        for &index in near {
            if Some(index) == parent || index == added {
                continue;
            }
            let through = cost + euclidean_distance(candidate, tree.state(index))?;
            if through < tree.cost(index)
                && self
                    .segments
                    .is_segment_free(candidate, tree.state(index))?
            {
                if let Some(slot) = tree.parents.get_mut(index) {
                    *slot = Some(added);
                }
                if let Some(slot) = tree.costs.get_mut(index) {
                    *slot = through;
                }
            }
        }
        Ok(())
    }

    /// Walks the tree back and appends the goal when that segment is free.
    fn finish(
        &self,
        tree: &Tree,
        leaf: usize,
        goal: &[f64],
        expanded: usize,
    ) -> Result<PlanOutcome<Vec<f64>>, Error> {
        let mut path = Vec::new();
        let mut node = Some(leaf);
        for _ in 0..=tree.states.len() {
            let Some(index) = node else { break };
            path.push(tree.state(index).to_vec());
            node = tree.parents.get(index).copied().flatten();
        }
        path.reverse();

        // The tree only reached within the goal tolerance, so the exact
        // goal is appended only when the last segment is actually free.
        if let Some(last) = path.last()
            && euclidean_distance(last, goal)? > 0.0
            && self.segments.is_segment_free(last, goal)?
        {
            path.push(goal.to_vec());
        }

        let mut cost = 0.0;
        for pair in path.windows(2) {
            if let [from, to] = pair {
                cost += euclidean_distance(from, to)?;
            }
        }
        Ok(PlanOutcome::Found {
            path,
            cost,
            expanded,
        })
    }

    fn occupancy_blocks(&self, state: &[f64]) -> Result<bool, Error> {
        match &self.segments {
            SegmentPolicy::Sampled { occupancy, .. } => occupancy.is_occupied(state),
            // A custom checker may have no notion of a single point, so
            // the endpoints are checked as a zero-length segment instead.
            SegmentPolicy::Custom(checker) => Ok(!checker.is_segment_free(state, state)?),
        }
    }

    /// The rewiring radius for a tree of `count` nodes in `dimension` axes.
    fn rewire_radius(&self, count: usize, dimension: usize) -> f64 {
        let nodes = f64::from(u32::try_from(count).unwrap_or(u32::MAX)).max(2.0);
        let axes = f64::from(u32::try_from(dimension).unwrap_or(1)).max(1.0);
        self.settings.radius_scale * (nodes.ln() / nodes).powf(1.0 / axes)
    }
}

/// The index of the tree node closest to `target`.
fn nearest_index(tree: &Tree, target: &[f64]) -> Result<Option<usize>, Error> {
    let mut best: Option<(usize, f64)> = None;
    for (index, state) in tree.states.iter().enumerate() {
        let distance = euclidean_distance(state, target)?;
        if best.is_none_or(|(_, previous)| distance < previous) {
            best = Some((index, distance));
        }
    }
    Ok(best.map(|(index, _)| index))
}

/// Every tree node within `radius` of `state`, reusing `found`.
fn collect_near(
    tree: &Tree,
    state: &[f64],
    radius: f64,
    found: &mut Vec<usize>,
) -> Result<(), Error> {
    found.clear();
    for (index, node) in tree.states.iter().enumerate() {
        if euclidean_distance(node, state)? <= radius {
            found.push(index);
        }
    }
    Ok(())
}
