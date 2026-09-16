//! Node positions layered onto a weighted graph.

use std::collections::BTreeMap;

use arco_core::Error;
use arco_core::geometry::{euclidean_distance, require_finite};
use arco_core::numeric::RELATIVE_TOLERANCE;
use arco_core::protocols::DiscreteMap;

use super::weighted::{NodeId, WeightedGraph};

/// Where a query position lands on the nearest edge.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeProjection {
    /// The projected point, in the graph's dimension.
    pub point: Vec<f64>,
    /// The edge's first endpoint.
    pub from: NodeId,
    /// The edge's second endpoint.
    pub to: NodeId,
    /// Distance from the query to the projection, meters.
    pub distance: f64,
}

/// A weighted graph whose nodes carry positions.
///
/// Owns a [`WeightedGraph`] rather than inheriting from one, per deviation
/// A-03, and delegates the topology methods to it so the call set is
/// unchanged.
#[derive(Debug, Clone, Default)]
pub struct CartesianGraph {
    topology: WeightedGraph,
    positions: BTreeMap<NodeId, Vec<f64>>,
    dimension: Option<usize>,
}

impl CartesianGraph {
    /// Builds an empty graph whose dimension is set by its first node.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an empty graph of a fixed dimension.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension: Some(dimension),
            ..Self::default()
        }
    }

    /// The dimension of the node positions, once any node exists.
    #[must_use]
    pub const fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// The underlying topology.
    #[must_use]
    pub const fn topology(&self) -> &WeightedGraph {
        &self.topology
    }

    /// Adds a node at `coordinates`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFew`] when `coordinates` is empty,
    /// [`Error::DimensionMismatch`] when it disagrees with the graph's
    /// dimension, and [`Error::NotFinite`] when it carries a NaN.
    pub fn add_node(&mut self, node: NodeId, coordinates: &[f64]) -> Result<(), Error> {
        if coordinates.is_empty() {
            return Err(Error::TooFew {
                quantity: "coordinates",
                minimum: 1,
                actual: 0,
            });
        }
        require_finite("node position", coordinates)?;

        match self.dimension {
            Some(expected) if expected != coordinates.len() => {
                return Err(Error::DimensionMismatch {
                    quantity: "node position",
                    expected,
                    actual: coordinates.len(),
                });
            }
            Some(_) => {}
            None => self.dimension = Some(coordinates.len()),
        }

        self.topology.add_node(node);
        self.positions.insert(node, coordinates.to_vec());
        Ok(())
    }

    /// The position of a node.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownIdentifier`] when the node is absent.
    pub fn position(&self, node: NodeId) -> Result<&[f64], Error> {
        self.positions
            .get(&node)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::UnknownIdentifier {
                kind: "node",
                identifier: node.to_string(),
            })
    }

    /// Adds an edge, defaulting its weight to the straight-line distance.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownIdentifier`] when either endpoint has no
    /// position, and otherwise as [`WeightedGraph::add_edge`].
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: Option<f64>) -> Result<(), Error> {
        let resolved = match weight {
            Some(given) => given,
            None => euclidean_distance(self.position(from)?, self.position(to)?)?,
        };
        self.topology.add_edge(from, to, resolved)
    }

    /// The edge weight between two adjacent nodes.
    ///
    /// # Errors
    ///
    /// As [`WeightedGraph::distance`].
    pub fn distance(&self, from: NodeId, to: NodeId) -> Result<f64, Error> {
        self.topology.distance(from, to)
    }

    /// Straight-line distance between two nodes, adjacent or not.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownIdentifier`] when either node is absent.
    pub fn heuristic(&self, from: NodeId, to: NodeId) -> Result<f64, Error> {
        euclidean_distance(self.position(from)?, self.position(to)?)
    }

    /// The nodes adjacent to `node`.
    #[must_use]
    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.topology.neighbors(node)
    }

    /// Whether the graph holds a position for `node`.
    #[must_use]
    pub fn contains_node(&self, node: NodeId) -> bool {
        self.positions.contains_key(&node)
    }

    /// The node closest to `position`, optionally within a radius.
    ///
    /// Returns `None` for an empty graph or when nothing falls inside the
    /// radius. Ties break toward the lower node id, because the storage is
    /// ordered, so repeated runs agree.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `position` disagrees with
    /// the graph, or [`Error::NotFinite`] when it carries a NaN.
    pub fn find_nearest_node(
        &self,
        position: &[f64],
        max_radius: Option<f64>,
    ) -> Result<Option<NodeId>, Error> {
        require_finite("query position", position)?;

        let mut nearest: Option<(NodeId, f64)> = None;
        for (&node, stored) in &self.positions {
            let distance = euclidean_distance(position, stored)?;
            if max_radius.is_some_and(|radius| distance > radius) {
                continue;
            }
            if nearest.is_none_or(|(_, best)| distance < best) {
                nearest = Some((node, distance));
            }
        }
        Ok(nearest.map(|(node, _)| node))
    }

    /// Projects `position` onto the closest edge.
    ///
    /// Returns `None` when the graph holds no edge.
    ///
    /// # Errors
    ///
    /// As [`CartesianGraph::find_nearest_node`].
    pub fn project_to_nearest_edge(
        &self,
        position: &[f64],
    ) -> Result<Option<EdgeProjection>, Error> {
        require_finite("query position", position)?;

        let mut nearest: Option<EdgeProjection> = None;
        for (from, to, _) in self.topology.edges() {
            let start = self.position(from)?;
            let end = self.position(to)?;

            let projected = project_onto_segment(position, start, end);
            let distance = euclidean_distance(position, &projected)?;
            if nearest.as_ref().is_none_or(|best| distance < best.distance) {
                nearest = Some(EdgeProjection {
                    point: projected,
                    from,
                    to,
                    distance,
                });
            }
        }
        Ok(nearest)
    }
}

impl AsRef<Self> for CartesianGraph {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl DiscreteMap for CartesianGraph {
    type Node = NodeId;

    fn contains(&self, node: NodeId) -> bool {
        self.contains_node(node)
    }

    fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        Self::neighbors(self, node)
    }

    fn distance(&self, from: NodeId, to: NodeId) -> Result<f64, Error> {
        Self::distance(self, from, to)
    }

    /// Straight-line distance between the two node positions.
    ///
    /// Admissible only while every edge weight is at least the
    /// straight-line distance between its endpoints, which is what the
    /// default weight in [`CartesianGraph::add_edge`] gives. A caller that
    /// supplies a weight below that, a shortcut priced under its own
    /// length, makes the estimate an overestimate and forfeits the
    /// optimality `FR-INV-06` asserts. Nothing rejects such a weight,
    /// because a road cheaper than its length is a reasonable thing to
    /// model; the precondition is stated here instead.
    fn heuristic(&self, node: NodeId, goal: NodeId) -> Result<f64, Error> {
        Self::heuristic(self, node, goal)
    }
}

/// The point on segment `start` to `end` closest to `query`.
///
/// A degenerate segment is detected relative to the scale of its
/// endpoints rather than by comparing its squared length to zero. An exact
/// comparison lets a segment of length 1e-30 through, and the division
/// that follows produces a parameter the clamp then hides.
fn project_onto_segment(query: &[f64], start: &[f64], end: &[f64]) -> Vec<f64> {
    let scale = start
        .iter()
        .chain(end)
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let degenerate = (scale * RELATIVE_TOLERANCE).max(f64::MIN_POSITIVE);

    let length_squared: f64 = start
        .iter()
        .zip(end)
        .map(|(a, b)| {
            let difference = b - a;
            difference * difference
        })
        .sum();

    if length_squared <= degenerate * degenerate {
        return start.to_vec();
    }

    let dot: f64 = query
        .iter()
        .zip(start)
        .zip(end.iter().zip(start))
        .map(|((q, s), (e, s2))| (q - s) * (e - s2))
        .sum();
    let ratio = (dot / length_squared).clamp(0.0, 1.0);

    start
        .iter()
        .zip(end)
        .map(|(s, e)| s + (e - s) * ratio)
        .collect()
}
