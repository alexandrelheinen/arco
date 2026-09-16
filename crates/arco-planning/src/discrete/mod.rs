//! Search over discrete maps: grids and graphs.

mod astar;
mod route;

pub use astar::{SearchOptions, search};
pub use route::{NodeProjection, RouteOutcome, RouteResult, RouteRouter};
