//! Graphs: topology, positions, and road geometry, layered by ownership.
//!
//! Python stacks these with inheritance. Rust stacks them with ownership,
//! each layer delegating to the one below, per deviation A-03.

mod cartesian;
mod road;
mod weighted;

pub use cartesian::{CartesianGraph, EdgeProjection};
pub use road::RoadGraph;
pub use weighted::{NodeId, WeightedGraph};
