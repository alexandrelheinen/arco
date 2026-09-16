//! Parsing tuning parameters out of YAML.
//!
//! Configuration is a value a caller constructs and hands to the object
//! that uses it. Nothing here reads an environment variable, nothing holds
//! a parsed document in a global, and nothing runs at load time, which is
//! ADR-015 and the reason the Python behavior it replaces was fragile:
//! several modules called `load_config` at import into a module global, so
//! `ARCO_CONFIG_DIR` stopped having any effect after the first import.
//!
//! The ported crates embed their own defaults with `include_str!` and
//! parse them once. A caller overriding a value passes a different
//! document or builds the configuration struct directly.

use std::path::Path;

use serde::de::DeserializeOwned;

use crate::Error;

/// Parses a YAML document into `T`.
///
/// # Arguments
///
/// * `source` - What to call this document in an error message, usually a
///   file name or the word `default`.
/// * `text` - The document.
///
/// # Errors
///
/// Returns [`Error::MissingConfigurationKey`] when a required field is
/// absent, and [`Error::ConflictingArguments`] for any other parse
/// failure, carrying the parser's own message.
///
/// # Examples
///
/// ```
/// use arco_core::config::parse_yaml;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Weights {
///     contour: f64,
/// }
///
/// let weights: Weights = parse_yaml("default", "contour: 10.0").unwrap();
/// assert!((weights.contour - 10.0).abs() < 1e-12);
/// ```
pub fn parse_yaml<T: DeserializeOwned>(source: &str, text: &str) -> Result<T, Error> {
    serde_yaml_ng::from_str(text).map_err(|failure| {
        let message = failure.to_string();
        // serde names an absent field rather than describing it, so the
        // caller gets the key it is missing instead of a parser trace.
        if let Some(key) = missing_field(&message) {
            Error::MissingConfigurationKey { key }
        } else {
            Error::ConflictingArguments {
                message: format!("cannot parse {source}: {message}"),
            }
        }
    })
}

/// Extracts the field name from a serde missing-field message.
fn missing_field(message: &str) -> Option<String> {
    let rest = message.strip_prefix("missing field `")?;
    let (key, _) = rest.split_once('`')?;
    Some(key.to_owned())
}

/// Reads and parses a YAML file.
///
/// # Errors
///
/// Returns [`Error::UnreadableFile`] when the file cannot be read, and
/// otherwise as [`parse_yaml`].
pub fn read_yaml_file<T: DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path).map_err(|failure| Error::UnreadableFile {
        path: path.display().to_string(),
        reason: failure.to_string(),
    })?;
    parse_yaml(&path.display().to_string(), &text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Weights {
        contour: f64,
        heading: f64,
    }

    #[test]
    fn a_complete_document_parses() {
        let parsed: Weights = parse_yaml("test", "contour: 10.0\nheading: 2.0").unwrap();
        assert_eq!(
            parsed,
            Weights {
                contour: 10.0,
                heading: 2.0
            }
        );
    }

    #[test]
    fn an_absent_key_is_named_rather_than_described() {
        let error = parse_yaml::<Weights>("test", "contour: 10.0").unwrap_err();
        match error {
            Error::MissingConfigurationKey { key } => assert_eq!(key, "heading"),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn malformed_yaml_names_its_source() {
        let error = parse_yaml::<Weights>("weights.yml", "contour: [").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("weights.yml"), "{message}");
    }

    #[test]
    fn a_missing_file_is_reported_as_such() {
        let error =
            read_yaml_file::<Weights>(Path::new("/nonexistent/arco/weights.yml")).unwrap_err();
        assert!(
            matches!(error, Error::UnreadableFile { .. }),
            "wrong error: {error:?}"
        );
        assert_eq!(
            error.python_exception(),
            crate::PythonException::FileNotFound
        );
    }

    #[test]
    fn parsing_holds_no_global_state() {
        // ADR-015: two documents parsed in sequence do not influence each
        // other, which is the property the Python import-time globals lost.
        let first: Weights = parse_yaml("a", "contour: 1.0\nheading: 2.0").unwrap();
        let second: Weights = parse_yaml("b", "contour: 3.0\nheading: 4.0").unwrap();
        let first_again: Weights = parse_yaml("a", "contour: 1.0\nheading: 2.0").unwrap();
        assert_eq!(first, first_again);
        assert_ne!(first, second);
    }
}
