//! Topology and edge weights, with no notion of position.

use std::collections::BTreeMap;

use arco_core::Error;

/// A node identifier.
///
/// Signed to match the Python side, where a node id is an ordinary
/// integer and negative ids are not rejected.
pub type NodeId = i64;

/// A graph of weighted edges.
///
/// Fully generic: it has no concept of position, of distance between
/// unconnected nodes, or of spatial queries. [`super::CartesianGraph`]
/// adds those by owning one of these.
///
/// Storage is ordered rather than hashed, so iteration is deterministic
/// and a nearest-node tie breaks the same way on every run. That matters
/// for the same reason seeded planning does.
#[derive(Debug, Clone, Default)]
pub struct WeightedGraph {
    adjacency: BTreeMap<NodeId, Vec<(NodeId, f64)>>,
}

impl WeightedGraph {
    /// Builds an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node, or does nothing if it is already present.
    pub fn add_node(&mut self, node: NodeId) {
        self.adjacency.entry(node).or_default();
    }

    /// Whether the graph holds this node.
    #[must_use]
    pub fn contains_node(&self, node: NodeId) -> bool {
        self.adjacency.contains_key(&node)
    }

    /// Adds an undirected edge, creating either endpoint if absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when `weight` is NaN or infinite, and
    /// [`Error::OutOfRange`] when it is negative. A negative edge cost
    /// breaks the optimality guarantee `FR-INV-06` rests on, silently, so
    /// it is rejected where it enters rather than where it misbehaves.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: f64) -> Result<(), Error> {
        if !weight.is_finite() {
            return Err(Error::NotFinite {
                quantity: "edge weight",
                value: weight,
            });
        }
        if weight < 0.0 {
            return Err(Error::OutOfRange {
                quantity: "edge weight",
                value: weight,
                bound: "[0, inf)",
            });
        }

        self.add_node(from);
        self.add_node(to);
        Self::link(&mut self.adjacency, from, to, weight);
        if from != to {
            Self::link(&mut self.adjacency, to, from, weight);
        }
        Ok(())
    }

    fn link(
        adjacency: &mut BTreeMap<NodeId, Vec<(NodeId, f64)>>,
        from: NodeId,
        to: NodeId,
        weight: f64,
    ) {
        let entry = adjacency.entry(from).or_default();
        if let Some(existing) = entry.iter_mut().find(|(other, _)| *other == to) {
            existing.1 = weight;
        } else {
            entry.push((to, weight));
        }
    }

    /// The nodes adjacent to `node`, in a deterministic order.
    #[must_use]
    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.adjacency
            .get(&node)
            .map(|edges| edges.iter().map(|&(other, _)| other).collect())
            .unwrap_or_default()
    }

    /// The weight of the edge between two adjacent nodes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownIdentifier`] when `from` is absent or the
    /// two nodes share no edge.
    pub fn distance(&self, from: NodeId, to: NodeId) -> Result<f64, Error> {
        self.adjacency
            .get(&from)
            .and_then(|edges| {
                edges
                    .iter()
                    .find(|&&(other, _)| other == to)
                    .map(|&(_, weight)| weight)
            })
            .ok_or_else(|| Error::UnknownIdentifier {
                kind: "edge",
                identifier: format!("{from} to {to}"),
            })
    }

    /// Every node, ascending.
    #[must_use]
    pub fn nodes(&self) -> Vec<NodeId> {
        self.adjacency.keys().copied().collect()
    }

    /// How many nodes the graph holds.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Every undirected edge once, as `(from, to, weight)` with `from < to`.
    #[must_use]
    pub fn edges(&self) -> Vec<(NodeId, NodeId, f64)> {
        let mut found = Vec::new();
        for (&from, edges) in &self.adjacency {
            for &(to, weight) in edges {
                if from <= to {
                    found.push((from, to, weight));
                }
            }
        }
        found
    }
}
