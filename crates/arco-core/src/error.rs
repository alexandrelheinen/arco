//! The root error every ARCO crate converts into.
//!
//! Each crate defines its own error enum and converts into this one, so a
//! caller matches on a single type without the leaf crates depending on
//! each other. The binding layer maps each variant onto the Python
//! exception the pure-Python implementation raised, which is what
//! `FR-API-04` requires.

use core::fmt;

/// The Python exception a variant surfaces as.
///
/// `arco-core` names the exception rather than constructing it, so that
/// nothing below the binding layer depends on `PyO3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PythonException {
    /// `ValueError`, for a value that is the right type and the wrong value.
    Value,
    /// `TypeError`, for a value of the wrong type or family.
    Type,
    /// `KeyError`, for a name or identifier that does not resolve.
    Key,
    /// `RuntimeError`, for a failure that is not attributable to one argument.
    Runtime,
    /// `FileNotFoundError`, for a missing configuration or data file.
    FileNotFound,
}

impl fmt::Display for PythonException {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match *self {
            Self::Value => "ValueError",
            Self::Type => "TypeError",
            Self::Key => "KeyError",
            Self::Runtime => "RuntimeError",
            Self::FileNotFound => "FileNotFoundError",
        };
        formatter.write_str(name)
    }
}

/// Anything ARCO refuses to do, and why.
///
/// Variants describe the failure rather than the function that produced
/// it, so a caller can branch on the cause.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A vector or array carried the wrong number of elements.
    #[error("{quantity} has dimension {actual}, expected {expected}")]
    DimensionMismatch {
        /// What was being measured, for the message.
        quantity: &'static str,
        /// The dimension the operation requires.
        expected: usize,
        /// The dimension the caller supplied.
        actual: usize,
    },

    /// A collection that must carry elements was empty or too short.
    #[error("{quantity} needs at least {minimum} element(s), got {actual}")]
    TooFew {
        /// What was being counted, for the message.
        quantity: &'static str,
        /// The smallest acceptable count.
        minimum: usize,
        /// The count the caller supplied.
        actual: usize,
    },

    /// A value fell outside the range the operation accepts.
    #[error("{quantity} is {value}, which is outside {bound}")]
    OutOfRange {
        /// What was being bounded, for the message.
        quantity: &'static str,
        /// The offending value.
        value: f64,
        /// The range, written as a reader would say it.
        bound: &'static str,
    },

    /// A value was NaN or infinite where a finite number is required.
    ///
    /// `FR-SAFE-07`. A non-finite value propagates through every later
    /// operation and silently poisons a comparison, so it is rejected at
    /// the boundary rather than carried inward.
    #[error("{quantity} is {value}, which is not finite")]
    NotFinite {
        /// What was being checked, for the message.
        quantity: &'static str,
        /// The offending value.
        value: f64,
    },

    /// A named variant did not match any the implementation knows.
    #[error("unknown {kind}: {name}")]
    UnknownVariant {
        /// The family the name was expected to belong to.
        kind: &'static str,
        /// The name the caller supplied.
        name: String,
    },

    /// An identifier did not resolve to anything.
    #[error("unknown {kind} identifier: {identifier}")]
    UnknownIdentifier {
        /// The family the identifier was expected to belong to.
        kind: &'static str,
        /// The identifier the caller supplied.
        identifier: String,
    },

    /// Two arguments cannot both be given, or neither was.
    #[error("{message}")]
    ConflictingArguments {
        /// What the caller should have supplied instead.
        message: String,
    },

    /// A map of the wrong family reached a planner that cannot use it.
    ///
    /// `FR-CORE-02`. Grid planners take grids and sampling planners take
    /// occupancy structures, and mixing them is a type error rather than a
    /// value error.
    #[error("expected a {expected} map, got a {actual}")]
    WrongMapFamily {
        /// The family the planner requires.
        expected: &'static str,
        /// The family the caller supplied.
        actual: &'static str,
    },

    /// A required configuration key was absent.
    #[error("configuration is missing the {key} key")]
    MissingConfigurationKey {
        /// The key that was expected.
        key: String,
    },

    /// A configuration or data file could not be read.
    #[error("cannot read {path}: {reason}")]
    UnreadableFile {
        /// The path that failed.
        path: String,
        /// Why it failed, in the operating system's words.
        reason: String,
    },
}

impl Error {
    /// The Python exception this variant surfaces as.
    ///
    /// The mapping is part of the public API under `FR-API-04`, so a
    /// change to it is a breaking change.
    #[must_use]
    pub const fn python_exception(&self) -> PythonException {
        match *self {
            // A value of the right type and the wrong value, which is what
            // ValueError means and what the Python implementation raised
            // for every one of these.
            Self::DimensionMismatch { .. }
            | Self::TooFew { .. }
            | Self::OutOfRange { .. }
            | Self::NotFinite { .. }
            | Self::UnknownVariant { .. }
            | Self::ConflictingArguments { .. } => PythonException::Value,

            // A name that does not resolve, which is a lookup failure
            // rather than a bad value.
            Self::UnknownIdentifier { .. } | Self::MissingConfigurationKey { .. } => {
                PythonException::Key
            }

            // A category mistake rather than a bad value: an occupancy
            // structure where a grid belongs is the wrong kind of thing.
            Self::WrongMapFamily { .. } => PythonException::Type,

            Self::UnreadableFile { .. } => PythonException::FileNotFound,
        }
    }
}
