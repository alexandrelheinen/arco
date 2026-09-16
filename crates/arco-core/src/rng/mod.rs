//! Random number generation matching `numpy.random.default_rng`.
//!
//! `FR-RNG-02` requires that a seeded ARCO planner draw the same values
//! the Python implementation drew. Reproducing that means reproducing two
//! things: the `SeedSequence` that turns a seed into a 128 bit state and
//! increment, and the PCG64 output function that turns that state into
//! draws. Both live here.
//!
//! The reference vectors in `tests/fixtures/pcg64.json` come from numpy
//! itself and are the oracle for this module.

mod pcg64;
mod seed_sequence;

pub use pcg64::Pcg64;
pub use seed_sequence::SeedSequence;
