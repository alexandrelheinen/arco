//! RRT*, the asymptotically optimal sampling planner.

use arco_core::Error;
use arco_core::geometry::{require_dimension, require_finite};
use arco_core::protocols::{Occupancy, PlannerCost};
use arco_core::rng::Pcg64;

use crate::failure::{PlanFailure, PlanOutcome};

use super::policy::{CostPolicy, SamplerPolicy, SegmentPolicy, SteererPolicy};
use super::tree::{Tree, close_path};

/// How an RRT* run should behave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrtSettings {
    /// Maximum samples to draw before giving up.
    ///
    /// `FR-SAFE-02`. Exhausting this is reported as retryable, distinct
    /// from proving the goal unreachable, which sampling never does.
    pub max_samples: usize,
    /// How close to the goal counts as arriving, in steps.
    pub goal_tolerance: f64,
    /// How often to sample the goal itself rather than the space.
    pub goal_bias: f64,
    /// Ceiling on the rewiring neighborhood, in steps.
    ///
    /// The radius shrinks as `gamma * (ln(n) / n)^(1/d)`, the schedule
    /// from Karaman and Frazzoli that makes RRT* asymptotically optimal
    /// rather than merely convergent, where `gamma` comes from the volume
    /// the sampler covers. Early in a run that formula exceeds the step
    /// the tree grows by, and a neighborhood wider than the reachable set
    /// only costs segment checks, so it is capped here.
    pub max_rewire_radius: f64,
    /// A fixed rewiring radius, in steps, replacing the schedule.
    ///
    /// Setting this forfeits asymptotic optimality, since the proof rests
    /// on the radius shrinking with the tree. It exists for a caller who
    /// needs a predictable per-iteration cost more than the guarantee.
    pub fixed_rewire_radius: Option<f64>,
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
            max_rewire_radius: 2.0,
            fixed_rewire_radius: None,
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
    cost: CostPolicy,
    settings: RrtSettings,
}

impl<O: Occupancy> RrtPlanner<O> {
    /// Builds a planner from its policies and settings.
    ///
    /// `cost` is the metric every tolerance and radius in `settings` is
    /// expressed in, and it should use the same per-axis scale the
    /// steerer steps by.
    #[must_use]
    pub const fn new(
        sampler: SamplerPolicy,
        steerer: SteererPolicy,
        segments: SegmentPolicy<O>,
        cost: CostPolicy,
        settings: RrtSettings,
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
        let mut goal_nodes: Vec<usize> = Vec::new();
        let mut near = Vec::new();
        let gamma = self.rewire_gamma(dimension);

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

            let Some(nearest) = self.nearest_index(&tree, &target)? else {
                continue;
            };
            let candidate = self.steerer.steer(tree.state(nearest), &target)?;
            if !self
                .segments
                .is_segment_free(tree.state(nearest), &candidate)?
            {
                continue;
            }

            let radius = self.rewire_radius(tree.len(), dimension, gamma);
            self.collect_near(&tree, &candidate, radius, &mut near)?;

            let (added, cost) = self.graft(&mut tree, candidate.clone(), nearest, &near)?;
            self.rewire(&mut tree, added, &candidate, cost, &near)?;

            if self.cost.distance(&candidate, goal)? <= self.settings.goal_tolerance {
                goal_nodes.push(added);
                if self.settings.early_stop {
                    return self.finish(&tree, added, goal, iteration.saturating_add(1));
                }
            }
        }

        // Cheapest by current cost rather than by the cost it had when it
        // was added: rewiring lowers costs afterward, and picking by the
        // older number throws away the improvement it just bought.
        // FR-INV-07 is the requirement this serves.
        match cheapest(&tree, &goal_nodes) {
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
            tree.cost(nearest).max(0.0) + self.cost.distance(tree.state(nearest), &candidate)?;

        for &index in near {
            if index == nearest {
                continue;
            }
            let through = tree.cost(index) + self.cost.distance(tree.state(index), &candidate)?;
            if through < cost
                && self
                    .segments
                    .is_segment_free(tree.state(index), &candidate)?
            {
                cost = through;
                parent = index;
            }
        }

        let added = tree.push(candidate, parent, cost);
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
        let parent = tree.parent(added);
        for &index in near {
            if Some(index) == parent || index == added {
                continue;
            }
            let through = cost + self.cost.distance(candidate, tree.state(index))?;
            if through < tree.cost(index)
                && self
                    .segments
                    .is_segment_free(candidate, tree.state(index))?
            {
                tree.repoint(index, added, through);
            }
        }
        Ok(())
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

    /// The index of the tree node closest to `target`.
    fn nearest_index(&self, tree: &Tree, target: &[f64]) -> Result<Option<usize>, Error> {
        let mut best: Option<(usize, f64)> = None;
        for (index, state) in tree.states().iter().enumerate() {
            let distance = self.cost.distance(state, target)?;
            if best.is_none_or(|(_, previous)| distance < previous) {
                best = Some((index, distance));
            }
        }
        Ok(best.map(|(index, _)| index))
    }

    /// Every tree node within `radius` of `state`, reusing `found`.
    fn collect_near(
        &self,
        tree: &Tree,
        state: &[f64],
        radius: f64,
        found: &mut Vec<usize>,
    ) -> Result<(), Error> {
        found.clear();
        for (index, node) in tree.states().iter().enumerate() {
            if self.cost.distance(node, state)? <= radius {
                found.push(index);
            }
        }
        Ok(())
    }

    /// The constant of the rewiring schedule, from the sampled volume.
    ///
    /// Karaman and Frazzoli require `gamma` to exceed
    /// `2 (1 + 1/d)^(1/d) (V / zeta_d)^(1/d)`, where `V` is the volume of
    /// the free space and `zeta_d` the volume of the unit ball, for the
    /// shrinking neighborhood to still connect the tree to itself. The
    /// sampled box stands in for the free space, which overestimates `V`
    /// and so keeps the bound on the safe side.
    ///
    /// Returns `None` when the sampler or the metric declines to say what
    /// it covers, in which case the caller falls back to the ceiling.
    fn rewire_gamma(&self, dimension: usize) -> Option<f64> {
        let SamplerPolicy::UniformBox { bounds } = &self.sampler else {
            return None;
        };
        if bounds.len() != dimension || dimension == 0 {
            return None;
        }
        let mut volume = 1.0_f64;
        for (axis, &(low, high)) in bounds.iter().enumerate() {
            let scale = self.cost.axis_scale(axis)?;
            volume *= (high - low) / scale;
        }
        if !(volume.is_finite() && volume > 0.0) {
            return None;
        }
        let axes = f64::from(u32::try_from(dimension).ok()?);
        let inverse = 1.0 / axes;
        Some(
            2.0 * (1.0 + inverse).powf(inverse)
                * (volume / unit_ball_volume(dimension)).powf(inverse),
        )
    }

    /// The rewiring radius for a tree of `count` nodes in `dimension` axes.
    fn rewire_radius(&self, count: usize, dimension: usize, gamma: Option<f64>) -> f64 {
        if let Some(fixed) = self.settings.fixed_rewire_radius {
            return fixed;
        }
        let Some(gamma) = gamma else {
            return self.settings.max_rewire_radius;
        };
        let nodes = f64::from(u32::try_from(count).unwrap_or(u32::MAX)).max(2.0);
        let axes = f64::from(u32::try_from(dimension).unwrap_or(1)).max(1.0);
        let scheduled = gamma * (nodes.ln() / nodes).powf(1.0 / axes);
        scheduled.min(self.settings.max_rewire_radius)
    }
}

/// The node of `candidates` the tree reaches most cheaply.
fn cheapest(tree: &Tree, candidates: &[usize]) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .map(|index| (index, tree.cost(index)))
        .reduce(|best, current| if current.1 < best.1 { current } else { best })
        .map(|(index, _)| index)
}

/// The volume of the unit ball in `dimension` axes.
///
/// Built by the recurrence `V_d = (2 pi / d) V_{d-2}` from `V_0 = 1` and
/// `V_1 = 2`, which avoids needing a gamma function for what is only ever
/// asked about a whole number of axes.
fn unit_ball_volume(dimension: usize) -> f64 {
    let even = dimension.is_multiple_of(2);
    let mut volume = if even { 1.0 } else { 2.0 };
    let mut axis = if even { 2 } else { 3 };
    while axis <= dimension {
        volume *= 2.0 * core::f64::consts::PI / f64::from(u32::try_from(axis).unwrap_or(u32::MAX));
        axis = axis.saturating_add(2);
    }
    volume
}
