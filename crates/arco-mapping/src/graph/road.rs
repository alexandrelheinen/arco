//! Per-edge geometry layered onto a Cartesian graph.

use std::collections::BTreeMap;

use arco_core::Error;
use arco_core::geometry::require_finite;

use super::cartesian::CartesianGraph;
use super::weighted::NodeId;

/// A Cartesian graph whose edges carry intermediate waypoints.
///
/// An edge between two nodes is a road, and a road is rarely straight, so
/// the geometry between its endpoints is stored alongside the topology.
#[derive(Debug, Clone, Default)]
pub struct RoadGraph {
    positions: CartesianGraph,
    geometry: BTreeMap<(NodeId, NodeId), Vec<Vec<f64>>>,
}

impl RoadGraph {
    /// Builds an empty road graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The underlying positioned graph.
    #[must_use]
    pub const fn positions(&self) -> &CartesianGraph {
        &self.positions
    }

    /// The underlying positioned graph, mutably.
    pub const fn positions_mut(&mut self) -> &mut CartesianGraph {
        &mut self.positions
    }

    /// Adds an edge carrying intermediate waypoints.
    ///
    /// The waypoints exclude both endpoints, matching the Python
    /// representation, and are stored once under the ordered key so that
    /// asking in either direction finds them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when a waypoint carries a NaN,
    /// [`Error::DimensionMismatch`] when one disagrees with the graph, and
    /// otherwise as [`CartesianGraph::add_edge`].
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        weight: Option<f64>,
        waypoints: &[Vec<f64>],
    ) -> Result<(), Error> {
        let dimension = self.positions.position(from)?.len();
        for waypoint in waypoints {
            require_finite("waypoint", waypoint)?;
            if waypoint.len() != dimension {
                return Err(Error::DimensionMismatch {
                    quantity: "waypoint",
                    expected: dimension,
                    actual: waypoint.len(),
                });
            }
        }

        self.positions.add_edge(from, to, weight)?;
        if !waypoints.is_empty() {
            self.geometry.insert(ordered(from, to), waypoints.to_vec());
        }
        Ok(())
    }

    /// The intermediate waypoints of an edge, in travel order.
    ///
    /// Empty for a straight edge. The order follows the direction asked
    /// for, so a caller travelling from the higher id to the lower gets
    /// the waypoints reversed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownIdentifier`] when the two nodes share no
    /// edge.
    pub fn edge_geometry(&self, from: NodeId, to: NodeId) -> Result<Vec<Vec<f64>>, Error> {
        self.positions.distance(from, to)?;
        let stored = self
            .geometry
            .get(&ordered(from, to))
            .cloned()
            .unwrap_or_default();
        if from <= to {
            Ok(stored)
        } else {
            Ok(stored.into_iter().rev().collect())
        }
    }

    /// The whole edge, endpoints included, in travel order.
    ///
    /// # Errors
    ///
    /// As [`RoadGraph::edge_geometry`].
    pub fn full_edge_geometry(&self, from: NodeId, to: NodeId) -> Result<Vec<Vec<f64>>, Error> {
        let mut path = vec![self.positions.position(from)?.to_vec()];
        path.extend(self.edge_geometry(from, to)?);
        path.push(self.positions.position(to)?.to_vec());
        Ok(path)
    }
}

impl AsRef<CartesianGraph> for RoadGraph {
    /// The positioned graph underneath, so anything taking a Cartesian
    /// graph takes a road graph too.
    ///
    /// Python got this from inheritance. Deviation A-03 replaced the
    /// inheritance with ownership, and this is what keeps the call sites
    /// reading the same.
    fn as_ref(&self) -> &CartesianGraph {
        &self.positions
    }
}

/// The storage key for an undirected edge.
const fn ordered(from: NodeId, to: NodeId) -> (NodeId, NodeId) {
    if from <= to { (from, to) } else { (to, from) }
}
