//! The traits that replace `arco.protocols`.
//!
//! Each of these was a `runtime_checkable` Python `Protocol`, which checks
//! that a method name exists and nothing about its signature. A wrong
//! signature on an injected sampler or cost term therefore failed at call
//! time, deep inside a planner loop. As traits they are checked when the
//! implementation is written, which is most of the reason for the port.
//!
//! Every trait here is implemented by another crate. `arco-core` declares
//! the shape and owns none of the behavior, which is what keeps it at the
//! floor of the dependency graph.

use crate::Error;
use crate::geometry::Pose;
use crate::rng::Pcg64;

/// A command leaving a controller: a speed and a turn rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    /// Commanded speed, meters per second.
    pub speed: f64,
    /// Commanded turn rate, radians per second.
    pub turn_rate: f64,
}

/// The distance to the nearest obstacle, and where it is.
#[derive(Debug, Clone, PartialEq)]
pub struct NearestObstacle {
    /// Distance to the obstacle, meters.
    pub distance: f64,
    /// The obstacle point, in the same frame and dimension as the query.
    pub point: Vec<f64>,
}

/// A map a discrete planner can search: nodes, neighbors, and edge costs.
pub trait DiscreteMap {
    /// The type identifying a node.
    type Node: Copy + Eq + core::hash::Hash;

    /// Whether `node` belongs to this map.
    ///
    /// A search asks this about its start and its goal before expanding
    /// anything, so that a query naming a cell off the edge of a grid is
    /// reported as such instead of as an unreachable goal. The two are
    /// different diagnoses and `FR-INV-08` keeps them apart.
    fn contains(&self, node: Self::Node) -> bool;

    /// The nodes adjacent to `node`.
    fn neighbors(&self, node: Self::Node) -> Vec<Self::Node>;

    /// The edge cost between two adjacent nodes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownIdentifier`] when either node is absent, or
    /// [`Error::NotFinite`] when the stored cost is not a real number.
    fn distance(&self, from: Self::Node, to: Self::Node) -> Result<f64, Error>;

    /// A lower bound on the cost from `node` to `goal`.
    ///
    /// Returning zero is always admissible and always useless. An
    /// overestimate breaks the optimality `FR-INV-06` asserts, so an
    /// implementation that cannot bound the remaining cost returns zero
    /// rather than guessing.
    ///
    /// # Errors
    ///
    /// As [`DiscreteMap::distance`].
    fn heuristic(&self, node: Self::Node, goal: Self::Node) -> Result<f64, Error>;
}

/// A continuous obstacle field a sampling planner can query.
pub trait Occupancy {
    /// The dimension of the space this occupancy describes.
    fn dimension(&self) -> usize;

    /// The radius around an obstacle that counts as occupied, meters.
    ///
    /// Already folded into [`Occupancy::nearest_obstacle`], which reports
    /// the distance to the obstacle's surface rather than to its center.
    /// It is exposed separately because a soft collision penalty needs to
    /// normalize a penetration depth by the clearance it ate into, and a
    /// depth without that scale says nothing about how bad it is.
    fn clearance(&self) -> f64;

    /// The nearest obstacle to `point`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `point` has the wrong
    /// dimension, or [`Error::NotFinite`] when it carries a NaN.
    fn nearest_obstacle(&self, point: &[f64]) -> Result<NearestObstacle, Error>;

    /// Whether `point` lies inside an obstacle.
    ///
    /// # Errors
    ///
    /// As [`Occupancy::nearest_obstacle`].
    fn is_occupied(&self, point: &[f64]) -> Result<bool, Error>;

    /// Whether the straight segment between two states touches nothing.
    ///
    /// Exact, not sampled. Sampling a segment at any fixed resolution
    /// misses an obstacle the segment only grazes, because the span it
    /// occludes shrinks to nothing as the contact gets shallower, and no
    /// sample count fixes that. `FR-INV-01` asks that a returned path
    /// survive a re-check finer than the planning resolution, which only
    /// an exact answer can promise.
    ///
    /// # Errors
    ///
    /// As [`Occupancy::nearest_obstacle`].
    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error>;
}

/// The metric a planner measures edges and remaining cost with.
///
/// Replaces `arco.planning.cost.PlannerCost`, which planners accepted
/// through a `cost=` argument so that a caller could change the metric
/// without subclassing the search.
pub trait PlannerCost {
    /// The cost of moving from `from` to `to`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree in
    /// dimension, or [`Error::NotFinite`] when one carries a NaN.
    fn distance(&self, from: &[f64], to: &[f64]) -> Result<f64, Error>;

    /// A lower bound on the remaining cost from `from` to `to`.
    ///
    /// Overestimating breaks the optimality `FR-INV-06` asserts, so an
    /// implementation that cannot bound the remainder returns zero.
    ///
    /// # Errors
    ///
    /// As [`PlannerCost::distance`].
    fn heuristic(&self, from: &[f64], to: &[f64]) -> Result<f64, Error>;
}

/// A source of random states for a sampling planner.
pub trait Sampler {
    /// Draws one state.
    ///
    /// # Errors
    ///
    /// Returns an error when the sampler cannot produce a state, which for
    /// a bounded sampler means the bounds were empty.
    fn sample(&self, generator: &mut Pcg64) -> Result<Vec<f64>, Error>;
}

/// A steering law taking one bounded step from one state toward another.
pub trait Steerer {
    /// Steers from `from` toward `to` by at most one step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree in
    /// dimension.
    fn steer(&self, from: &[f64], to: &[f64]) -> Result<Vec<f64>, Error>;
}

/// A collision check for the straight segment between two states.
pub trait SegmentChecker {
    /// Whether the segment from `from` to `to` is free.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the states disagree in
    /// dimension.
    fn is_segment_free(&self, from: &[f64], to: &[f64]) -> Result<bool, Error>;
}

/// One term of a composite cost, reading a context of type `C`.
///
/// The context is a type parameter rather than an associated type so that
/// a term stays object safe when the context borrows, which a trajectory
/// context does: it holds the waypoints and the map the optimizer is
/// working on rather than copies of them.
pub trait CostTerm<C: ?Sized> {
    /// Evaluates this term.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] rather than returning a NaN, since a
    /// non-finite cost makes every later comparison false and lets an
    /// optimizer select a garbage candidate silently.
    fn evaluate(&self, context: &C) -> Result<f64, Error>;
}

/// A planner producing a path between two states.
pub trait Planner {
    /// Plans from `start` to `goal` within `budget` samples or expansions.
    ///
    /// Returning `Ok(None)` means the planner ran out of budget without a
    /// solution, which `FR-SAFE-02` requires be distinguishable from an
    /// error and from success.
    ///
    /// # Errors
    ///
    /// Returns an error when the query itself is rejected, for instance
    /// because a state is occupied or out of bounds.
    fn plan(
        &self,
        start: &[f64],
        goal: &[f64],
        budget: usize,
    ) -> Result<Option<Vec<Vec<f64>>>, Error>;
}

/// A stage shortening a path without making it infeasible.
pub trait Pruner {
    /// Returns a shortened path.
    ///
    /// `FR-INV-03`: the result is never longer in cost than the input, and
    /// never invalid where the input was valid.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is malformed, for instance when its
    /// states disagree in dimension.
    fn prune(&self, path: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error>;
}

/// A stage turning a path into a time-parameterized trajectory.
pub trait Optimizer {
    /// The trajectory representation this optimizer produces.
    type Trajectory;

    /// Refines `path` within `budget` iterations.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is malformed or the problem is
    /// infeasible as stated.
    fn optimize(&self, path: &[Vec<f64>], budget: usize) -> Result<Self::Trajectory, Error>;
}

/// A geometric tracker turning a pose and a path into a command.
pub trait PathTracker {
    /// Computes the command for the current pose.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when the path is too short to track, or
    /// [`Error::NotFinite`] when the pose carries a non-finite value.
    fn track(&mut self, pose: Pose, path: &[(f64, f64)], speed: f64) -> Result<Command, Error>;
}

/// A vehicle whose state a tracking loop advances.
pub trait VehicleModel {
    /// The current pose.
    fn pose(&self) -> Pose;

    /// The current speed, meters per second.
    fn speed(&self) -> f64;

    /// The current turn rate, radians per second.
    fn turn_rate(&self) -> f64;

    /// Advances the state by one control step.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfRange`] when `dt` is not strictly positive,
    /// or [`Error::NotFinite`] when a command is not a real number, per
    /// `FR-INV-10`.
    fn step(&mut self, command: Command, dt: f64) -> Result<(), Error>;
}

/// A reactive correction added to a tracking command.
pub trait AvoidanceStrategy {
    /// The additive turn rate bias for `pose`, radians per second.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] rather than returning a NaN bias.
    fn turn_rate_bias(&self, pose: Pose) -> Result<f64, Error>;
}

/// A sink for planner telemetry.
pub trait TelemetryPublisher {
    /// The snapshot type this sink accepts.
    type Snapshot;

    /// Publishes one snapshot.
    ///
    /// Publishing never fails the planner. A sink that cannot write drops
    /// the snapshot, because losing telemetry is preferable to abandoning
    /// a plan.
    fn publish(&mut self, snapshot: &Self::Snapshot);
}
