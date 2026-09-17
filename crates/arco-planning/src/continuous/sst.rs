//! SST, the sparse sampling planner that bounds how much tree it keeps.

use std::collections::BTreeSet;

use arco_core::Error;
use arco_core::geometry::{require_dimension, require_finite};
use arco_core::protocols::{Occupancy, PlannerCost};
use arco_core::rng::Pcg64;

use crate::failure::{PlanFailure, PlanOutcome};

use super::policy::{CostPolicy, SamplerPolicy, SegmentPolicy, SteererPolicy};
use super::tree::{PlannerTree, Tree, close_path};

/// How an SST run should behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SstSettings {
    /// Maximum samples to draw before giving up.
    ///
    /// `FR-SAFE-02`. Exhausting this is reported as retryable, distinct
    /// from proving the goal unreachable, which sampling never does.
    pub max_samples: usize,
    /// How close to the goal counts as arriving, in steps.
    pub goal_tolerance: f64,
    /// Half-width of a witness cell, in steps.
    ///
    /// This is the knob the whole algorithm turns on. Every region of
    /// this radius keeps exactly one node, the cheapest one reached so
    /// far, so a smaller radius keeps a denser tree and a better path
    /// while a larger one keeps less of it. It has to be under one step,
    /// or a new node always lands inside its own parent's cell and the
    /// tree cannot grow.
    pub witness_radius: f64,
    /// How often to sample the goal itself rather than the space.
    pub goal_bias: f64,
    /// Whether to stop at the first solution rather than improving it.
    pub early_stop: bool,
}

impl Default for SstSettings {
    fn default() -> Self {
        Self {
            max_samples: 3000,
            goal_tolerance: 1.0,
            witness_radius: 0.5,
            goal_bias: 0.05,
            early_stop: true,
        }
    }
}

/// An SST planner over a continuous space.
///
/// SST is asymptotically near-optimal rather than optimal: it converges
/// to within a factor of the best path rather than to the best path, and
/// it pays for that with a tree whose size is bounded by the number of
/// witness cells instead of by the sample budget. `FR-INV-07` states the
/// difference as a suboptimality band rather than as convergence.
///
/// Holds its policies by value so the inner loop calls them directly. See
/// ADR-004 for why that matters more than it looks.
#[derive(Debug)]
pub struct SstPlanner<O> {
    sampler: SamplerPolicy,
    steerer: SteererPolicy,
    segments: SegmentPolicy<O>,
    cost: CostPolicy,
    settings: SstSettings,
}

/// A node the planner is about to offer to a region.
#[derive(Debug, Clone, Copy)]
struct Candidate<'a> {
    state: &'a [f64],
    parent: usize,
    cost: f64,
}

/// The witness set: one representative node per occupied region.
#[derive(Debug)]
struct Witnesses {
    positions: Vec<Vec<f64>>,
    representatives: Vec<Option<usize>>,
}

impl Witnesses {
    /// Builds a witness set seeded with the root, which represents itself.
    fn with_root(root: Vec<f64>) -> Self {
        Self {
            positions: vec![root],
            representatives: vec![Some(0)],
        }
    }

    /// The nearest witness within `radius` of `point`, if any.
    ///
    /// # Errors
    ///
    /// Propagates whatever the cost policy returns.
    fn nearest(
        &self,
        point: &[f64],
        radius: f64,
        cost: &CostPolicy,
    ) -> Result<Option<usize>, Error> {
        let mut best: Option<(usize, f64)> = None;
        for (index, witness) in self.positions.iter().enumerate() {
            let distance = cost.distance(witness, point)?;
            if distance <= radius && best.is_none_or(|(_, previous)| distance < previous) {
                best = Some((index, distance));
            }
        }
        Ok(best.map(|(index, _)| index))
    }

    /// Opens a new region around `point` and returns its index.
    fn open(&mut self, point: Vec<f64>) -> usize {
        let index = self.positions.len();
        self.positions.push(point);
        self.representatives.push(None);
        index
    }

    /// The node currently representing region `index`, if any.
    fn representative(&self, index: usize) -> Option<usize> {
        self.representatives.get(index).copied().flatten()
    }

    /// Hands region `index` to `node`.
    fn assign(&mut self, index: usize, node: usize) {
        if let Some(slot) = self.representatives.get_mut(index) {
            *slot = Some(node);
        }
    }
}

impl<O: Occupancy> SstPlanner<O> {
    /// Builds a planner from its policies and settings.
    ///
    /// `cost` is the metric the tolerance and the witness radius are
    /// expressed in, and it should use the same per-axis scale the
    /// steerer steps by.
    #[must_use]
    pub const fn new(
        sampler: SamplerPolicy,
        steerer: SteererPolicy,
        segments: SegmentPolicy<O>,
        cost: CostPolicy,
        settings: SstSettings,
    ) -> Self {
        Self {
            sampler,
            steerer,
            segments,
            cost,
            settings,
        }
    }

    /// Plans from `start` to `goal`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree with
    /// each other or with the occupancy, [`Error::NotFinite`] when one
    /// carries a NaN, or [`Error::OutOfRange`] when the witness radius is
    /// not inside `(0, 1)`.
    pub fn plan(
        &self,
        start: &[f64],
        goal: &[f64],
        generator: &mut Pcg64,
    ) -> Result<PlanOutcome<Vec<f64>>, Error> {
        Ok(self.plan_tree(start, goal, generator)?.0)
    }

    /// Plans, and hands back the tree that survived.
    ///
    /// The active nodes only, renumbered from zero, which is the whole
    /// point of SST: a node retired when a cheaper one took its region is
    /// no longer somewhere the tree can grow from, and drawing it would
    /// make the sparse tree look exactly as dense as an RRT.
    ///
    /// # Errors
    ///
    /// As [`SstPlanner::plan`].
    pub fn plan_tree(
        &self,
        start: &[f64],
        goal: &[f64],
        generator: &mut Pcg64,
    ) -> Result<(PlanOutcome<Vec<f64>>, PlannerTree), Error> {
        require_finite("start", start)?;
        require_finite("goal", goal)?;
        require_dimension("goal", goal, start.len())?;
        self.require_growable_witness_radius()?;

        if self.occupancy_blocks(start)? {
            return Ok((
                PlanOutcome::Failed {
                    reason: PlanFailure::StartOccupied,
                    expanded: 0,
                },
                PlannerTree::default(),
            ));
        }
        if self.occupancy_blocks(goal)? {
            return Ok((
                PlanOutcome::Failed {
                    reason: PlanFailure::GoalOccupied,
                    expanded: 0,
                },
                PlannerTree::default(),
            ));
        }

        let dimension = start.len();
        let mut tree = Tree::with_root(start.to_vec(), self.settings.max_samples);
        let mut witnesses = Witnesses::with_root(start.to_vec());
        let mut active: BTreeSet<usize> = BTreeSet::new();
        active.insert(0);
        let mut best_goal: Option<usize> = None;
        let mut best_goal_cost = f64::INFINITY;

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

            let Some(selected) = self.nearest_active(&tree, &active, &target)? else {
                continue;
            };
            let candidate = self.steerer.steer(tree.state(selected), &target)?;
            if !self
                .segments
                .is_segment_free(tree.state(selected), &candidate)?
            {
                continue;
            }
            let reached =
                tree.cost(selected) + self.cost.distance(tree.state(selected), &candidate)?;

            let Some(added) = self.claim_region(
                &mut tree,
                &mut witnesses,
                &mut active,
                Candidate {
                    state: &candidate,
                    parent: selected,
                    cost: reached,
                },
            )?
            else {
                continue;
            };

            if self.cost.distance(&candidate, goal)? <= self.settings.goal_tolerance
                && reached < best_goal_cost
            {
                best_goal_cost = reached;
                best_goal = Some(added);
                if self.settings.early_stop {
                    let outcome = self.finish(&tree, added, goal, iteration.saturating_add(1))?;
                    return Ok((outcome, tree.snapshot_of(active)));
                }
            }
        }

        let outcome = match best_goal {
            Some(index) => self.finish(&tree, index, goal, self.settings.max_samples)?,
            None => PlanOutcome::Failed {
                // Sampling never proves a goal unreachable, it only runs
                // out of samples, so this is always the retryable answer.
                reason: PlanFailure::BudgetExhausted,
                expanded: self.settings.max_samples,
            },
        };
        Ok((outcome, tree.snapshot_of(active)))
    }

    /// Admits `candidate` only if it beats whatever holds its region.
    ///
    /// This is the whole of SST. A candidate landing in a region already
    /// held by a cheaper node is dropped outright rather than added, and
    /// a candidate that beats the incumbent takes the region and retires
    /// it. The retired node stays in the tree so that paths through it
    /// still trace, but it stops being a place the tree can grow from,
    /// which is what keeps the active set bounded by the number of
    /// regions instead of by the sample budget.
    ///
    /// Returns the new node's index, or `None` when the candidate lost.
    ///
    /// # Errors
    ///
    /// Propagates whatever the cost policy returns.
    fn claim_region(
        &self,
        tree: &mut Tree,
        witnesses: &mut Witnesses,
        active: &mut BTreeSet<usize>,
        candidate: Candidate<'_>,
    ) -> Result<Option<usize>, Error> {
        let region =
            match witnesses.nearest(candidate.state, self.settings.witness_radius, &self.cost)? {
                Some(index) => index,
                None => witnesses.open(candidate.state.to_vec()),
            };

        let incumbent = witnesses.representative(region);
        if let Some(held_by) = incumbent
            && candidate.cost >= tree.cost(held_by)
        {
            return Ok(None);
        }

        let added = tree.push(candidate.state.to_vec(), candidate.parent, candidate.cost);
        active.insert(added);
        if let Some(held_by) = incumbent {
            active.remove(&held_by);
        }
        witnesses.assign(region, added);
        Ok(Some(added))
    }

    /// The active node closest to `target`.
    ///
    /// Geometric SST selects by proximity rather than by cost, which is
    /// what points the tree at the sample instead of at the cheapest
    /// frontier.
    ///
    /// # Errors
    ///
    /// Propagates whatever the cost policy returns.
    fn nearest_active(
        &self,
        tree: &Tree,
        active: &BTreeSet<usize>,
        target: &[f64],
    ) -> Result<Option<usize>, Error> {
        let mut best: Option<(usize, f64)> = None;
        for &index in active {
            let distance = self.cost.distance(tree.state(index), target)?;
            if best.is_none_or(|(_, previous)| distance < previous) {
                best = Some((index, distance));
            }
        }
        Ok(best.map(|(index, _)| index))
    }

    /// Walks the tree back from `leaf` and closes the path on the goal.
    ///
    /// # Errors
    ///
    /// Propagates whatever the cost or segment policy returns.
    fn finish(
        &self,
        tree: &Tree,
        leaf: usize,
        goal: &[f64],
        expanded: usize,
    ) -> Result<PlanOutcome<Vec<f64>>, Error> {
        close_path(tree.trace(leaf), goal, &self.cost, &self.segments, expanded)
    }

    /// Rejects a witness radius that would stop the tree growing.
    ///
    /// At a radius of one step or more, every candidate lands inside the
    /// region its own parent holds and never beats it, so the planner
    /// spends its whole budget rejecting candidates and reports an
    /// exhausted budget on a trivially open map. Failing at the boundary
    /// says what is wrong; the empty run does not.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when the radius is outside `(0, 1)`.
    fn require_growable_witness_radius(&self) -> Result<(), Error> {
        let radius = self.settings.witness_radius;
        if radius.is_finite() && radius > 0.0 && radius < 1.0 {
            Ok(())
        } else {
            Err(Error::OutOfRange {
                quantity: "witness radius",
                value: radius,
                bound: "(0, 1) steps",
            })
        }
    }

    fn occupancy_blocks(&self, state: &[f64]) -> Result<bool, Error> {
        match &self.segments {
            SegmentPolicy::Exact { occupancy } | SegmentPolicy::Sampled { occupancy, .. } => {
                occupancy.is_occupied(state)
            }
            // A custom checker may have no notion of a single point, so
            // the endpoints are checked as a zero-length segment instead.
            SegmentPolicy::Custom(checker) => Ok(!checker.is_segment_free(state, state)?),
        }
    }
}
