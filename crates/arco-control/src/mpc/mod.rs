//! Receding-horizon control, as a sequence of convex programs.
//!
//! ADR-002. The Python controllers built a nonlinear program with `CasADi`
//! and solved it with IPOPT. These linearize the model and the costs about
//! the previous solution and solve the resulting convex program with
//! Clarabel, iterating a few times per control step. Deviation A-02
//! records that the two reach different solutions on the same input, and
//! `FR-MPC-02` states how far apart they are allowed to be.

pub mod costs;
pub mod joint_space;
pub mod model;
pub mod path_following;
pub mod qp;
pub mod reference;
