//! Search over discrete maps: grids and graphs.

mod astar;
mod route;

pub use astar::{
    DiagnosedSearch, SearchDiagnostics, SearchOptions, search, search_with_diagnostics,
};
pub use route::{NodeProjection, RouteOutcome, RouteResult, RouteRouter};
