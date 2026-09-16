// Copyright 2026 alexandre
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! FR-API-04: every variant surfaces as the exception Python already raised.

use arco_core::{Error, PythonException};

/// One of every variant, so a new one cannot be added without deciding
/// which exception it surfaces as.
fn every_variant() -> Vec<(Error, PythonException)> {
    vec![
        (
            Error::DimensionMismatch {
                quantity: "position",
                expected: 2,
                actual: 3,
            },
            PythonException::Value,
        ),
        (
            Error::TooFew {
                quantity: "waypoints",
                minimum: 2,
                actual: 1,
            },
            PythonException::Value,
        ),
        (
            Error::OutOfRange {
                quantity: "step_size",
                value: -1.0,
                bound: "(0, inf)",
            },
            PythonException::Value,
        ),
        (
            Error::NotFinite {
                quantity: "cost",
                value: f64::NAN,
            },
            PythonException::Value,
        ),
        (
            Error::UnknownVariant {
                kind: "joint type",
                name: "helical".to_owned(),
            },
            PythonException::Value,
        ),
        (
            Error::UnknownIdentifier {
                kind: "node",
                identifier: "42".to_owned(),
            },
            PythonException::Key,
        ),
        (
            Error::ConflictingArguments {
                message: "give shape or physical_size, not both".to_owned(),
            },
            PythonException::Value,
        ),
        (
            Error::WrongMapFamily {
                expected: "Grid",
                actual: "Occupancy",
            },
            PythonException::Type,
        ),
        (
            Error::MissingConfigurationKey {
                key: "weight".to_owned(),
            },
            PythonException::Key,
        ),
        (
            Error::UnreadableFile {
                path: "/etc/arco/mpc.yml".to_owned(),
                reason: "no such file".to_owned(),
            },
            PythonException::FileNotFound,
        ),
    ]
}

#[test]
fn every_variant_names_the_exception_python_raised() {
    for (error, expected) in every_variant() {
        assert_eq!(
            error.python_exception(),
            expected,
            "wrong exception for {error:?}"
        );
    }
}

#[test]
fn a_wrong_map_family_is_a_type_error_not_a_value_error() {
    // FR-CORE-02. Passing an occupancy structure to a grid planner is a
    // category mistake, and Python signals those with TypeError.
    let error = Error::WrongMapFamily {
        expected: "Grid",
        actual: "Occupancy",
    };
    assert_eq!(error.python_exception(), PythonException::Type);
}

#[test]
fn a_message_names_the_quantity_and_the_offending_value() {
    let error = Error::OutOfRange {
        quantity: "step_size",
        value: -1.0,
        bound: "(0, inf)",
    };
    let message = error.to_string();
    assert!(message.contains("step_size"), "{message}");
    assert!(message.contains("-1"), "{message}");
    assert!(message.contains("(0, inf)"), "{message}");
}

#[test]
fn a_message_never_names_the_function_that_produced_it() {
    // The guideline asks for FrameTooLarge rather than PipelineRunError,
    // so no variant should mention a call site.
    for (error, _) in every_variant() {
        let message = error.to_string();
        for forbidden in ["plan(", "step(", "__init__", "Error in"] {
            assert!(!message.contains(forbidden), "{message} names a call site");
        }
    }
}

#[test]
fn exception_names_match_the_python_builtins() {
    assert_eq!(PythonException::Value.to_string(), "ValueError");
    assert_eq!(PythonException::Type.to_string(), "TypeError");
    assert_eq!(PythonException::Key.to_string(), "KeyError");
    assert_eq!(PythonException::Runtime.to_string(), "RuntimeError");
    assert_eq!(
        PythonException::FileNotFound.to_string(),
        "FileNotFoundError"
    );
}
