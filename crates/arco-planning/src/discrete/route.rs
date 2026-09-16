//! Routing continuous positions over a road network.
//!
//! A vehicle has a position anywhere in the plane; a road network has
//! nodes. Routing is the join between them: project both endpoints onto
//! the network, search the network, and hand back the path together with
//! what the projection cost, so a caller can drive to the first node
//! before following the route.

use arco_core::Error;
use arco_core::geometry::euclidean_distance;
use arco_mapping::graph::{CartesianGraph, NodeId};

use crate::discrete::{SearchOptions, search};
use crate::failure::{PlanFailure, PlanOutcome};

/// Where a continuous position landed on the network.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeProjection {
    /// The node the position snapped to.
    pub node: NodeId,
    /// That node's position, in the graph's dimension.
    pub point: Vec<f64>,
    /// How far the query was from it, meters.
    ///
    /// The distance a caller still has to cover off the network, which is
    /// why it is reported rather than discarded.
    pub distance: f64,
}

/// A route, and what it cost to get onto the network.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteResult {
    /// The node sequence from start to goal, both included.
    pub path: Vec<NodeId>,
    /// The route's cost under the graph's own edge weights.
    pub cost: f64,
    /// How many nodes the search expanded.
    pub expanded: usize,
    /// Where the start position joined the network.
    pub start: NodeProjection,
    /// Where the goal position joined the network.
    pub goal: NodeProjection,
}

/// What a routing query produced, or why it did not.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteOutcome {
    /// A route.
    Found(RouteResult),
    /// No route, and why.
    Failed {
        /// The reason, from the same closed enumeration every planner uses.
        reason: PlanFailure,
        /// How many nodes the search expanded before giving up.
        expanded: usize,
    },
}

impl RouteOutcome {
    /// The route, if one was found.
    #[must_use]
    pub const fn result(&self) -> Option<&RouteResult> {
        match self {
            Self::Found(result) => Some(result),
            Self::Failed { .. } => None,
        }
    }

    /// The reason, if none was.
    #[must_use]
    pub const fn failure(&self) -> Option<PlanFailure> {
        match *self {
            Self::Found(_) => None,
            Self::Failed { reason, .. } => Some(reason),
        }
    }

    /// The node sequence, if a route was found.
    #[must_use]
    pub fn path(&self) -> Option<&[NodeId]> {
        self.result().map(|result| result.path.as_slice())
    }

    /// How many nodes the search expanded either way.
    #[must_use]
    pub const fn expanded(&self) -> usize {
        match *self {
            Self::Found(RouteResult { expanded, .. }) | Self::Failed { expanded, .. } => expanded,
        }
    }
}

/// Plans routes over a positioned graph from continuous positions.
///
/// Takes anything that can present itself as a [`CartesianGraph`], which
/// includes a `RoadGraph`, so a caller holding either passes it directly.
#[derive(Debug)]
pub struct RouteRouter<G> {
    graph: G,
    activation_radius: Option<f64>,
    options: SearchOptions,
}

impl<G: AsRef<CartesianGraph>> RouteRouter<G> {
    /// Builds a router over `graph`.
    ///
    /// # Arguments
    ///
    /// * `graph` - The road network to route over.
    /// * `activation_radius` - How far a position may be from the network
    ///   and still join it, meters. `None` accepts any distance, which
    ///   means a query in the wrong part of the world silently routes from
    ///   whatever node happens to be least far away.
    #[must_use]
    pub fn new(graph: G, activation_radius: Option<f64>) -> Self {
        Self {
            graph,
            activation_radius,
            options: SearchOptions::default(),
        }
    }

    /// Replaces the search options, which is how the budget is set.
    #[must_use]
    pub const fn with_options(mut self, options: SearchOptions) -> Self {
        self.options = options;
        self
    }

    /// The graph being routed over.
    pub fn graph(&self) -> &CartesianGraph {
        self.graph.as_ref()
    }

    /// Routes from `start` to `goal`, both continuous positions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a position carries a NaN,
    /// [`Error::DimensionMismatch`] when one disagrees with the graph, and
    /// otherwise whatever the graph's own weights return.
    pub fn plan(&self, start: &[f64], goal: &[f64]) -> Result<RouteOutcome, Error> {
        let graph = self.graph.as_ref();

        // A position too far from any road is outside the network in the
        // same sense a cell off the edge of a grid is outside the map, and
        // it gets the same reason: no budget makes it answerable.
        let Some(start) = self.project(graph, start)? else {
            return Ok(RouteOutcome::Failed {
                reason: PlanFailure::StartOutsideMap,
                expanded: 0,
            });
        };
        let Some(goal) = self.project(graph, goal)? else {
            return Ok(RouteOutcome::Failed {
                reason: PlanFailure::GoalOutsideMap,
                expanded: 0,
            });
        };

        match search(graph, start.node, goal.node, self.options)? {
            PlanOutcome::Found {
                path,
                cost,
                expanded,
            } => Ok(RouteOutcome::Found(RouteResult {
                path,
                cost,
                expanded,
                start,
                goal,
            })),
            PlanOutcome::Failed { reason, expanded } => {
                Ok(RouteOutcome::Failed { reason, expanded })
            }
        }
    }

    /// Snaps `position` onto the nearest node inside the radius.
    ///
    /// # Errors
    ///
    /// As [`CartesianGraph::find_nearest_node`].
    fn project(
        &self,
        graph: &CartesianGraph,
        position: &[f64],
    ) -> Result<Option<NodeProjection>, Error> {
        let Some(node) = graph.find_nearest_node(position, self.activation_radius)? else {
            return Ok(None);
        };
        let point = graph.position(node)?.to_vec();
        let distance = euclidean_distance(position, &point)?;
        Ok(Some(NodeProjection {
            node,
            point,
            distance,
        }))
    }
}
