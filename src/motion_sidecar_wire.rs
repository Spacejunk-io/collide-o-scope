//! Bounded inspection boundary for versioned export motion sidecars.

use std::fmt;

#[allow(
    clippy::duplicate_mod,
    reason = "the standalone fuzz seam and application root intentionally share this parser source"
)]
#[path = "bounded_json.rs"]
mod bounded_json;

pub const MAX_MOTION_SIDECAR_BYTES: usize = 4 * 1024 * 1024;
pub const MOTION_SIDECAR_SCHEMA_VERSION: u16 = 9;
pub const MAX_MOTION_SIDECAR_SOURCES: usize = 256;
pub const MAX_MOTION_SIDECAR_SCOPES: usize = 256;
pub const MAX_MOTION_SIDECAR_DISTINCT_STATES: usize = 512;
pub const MAX_MOTION_SIDECAR_SYMMETRY_NODES: usize = 256;
pub const MAX_MOTION_SIDECAR_RESIDUAL_NODES: usize = 512;
pub const MAX_MOTION_SIDECAR_WARNINGS: usize = 128;

const MOTION_SIDECAR_LIMITS: bounded_json::JsonLimits = bounded_json::JsonLimits {
    max_bytes: MAX_MOTION_SIDECAR_BYTES,
    max_depth: 64,
    max_nodes: 250_000,
    max_key_bytes: 128,
    max_string_bytes: 256 * 1024,
};

#[derive(Debug)]
pub enum MotionSidecarWireError {
    Boundary(bounded_json::BoundedJsonError),
    Root,
    Schema {
        observed: Option<u64>,
        expected: u16,
    },
    Missing(&'static str),
    ListOverCap {
        field: &'static str,
        observed: usize,
        limit: usize,
    },
}

impl fmt::Display for MotionSidecarWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boundary(error) => write!(formatter, "motion sidecar {error}"),
            Self::Root => formatter.write_str("motion sidecar root must be an object"),
            Self::Schema { observed, expected } => write!(
                formatter,
                "motion sidecar schema {observed:?} does not match {expected}"
            ),
            Self::Missing(field) => write!(
                formatter,
                "motion sidecar field '{field}' is missing or has the wrong type"
            ),
            Self::ListOverCap {
                field,
                observed,
                limit,
            } => write!(
                formatter,
                "motion sidecar list '{field}' has {observed} entries; limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for MotionSidecarWireError {}

fn bounded_list(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    limit: usize,
) -> Result<(), MotionSidecarWireError> {
    let values = object
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or(MotionSidecarWireError::Missing(field))?;
    if values.len() > limit {
        return Err(MotionSidecarWireError::ListOverCap {
            field,
            observed: values.len(),
            limit,
        });
    }
    Ok(())
}

pub fn parse_motion_sidecar(
    bytes: &[u8],
    expected_schema: u16,
) -> Result<serde_json::Value, MotionSidecarWireError> {
    let value = bounded_json::parse_bounded_json(bytes, MOTION_SIDECAR_LIMITS)
        .map_err(MotionSidecarWireError::Boundary)?;
    let object = value.as_object().ok_or(MotionSidecarWireError::Root)?;
    let observed = object
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if observed != Some(u64::from(expected_schema)) {
        return Err(MotionSidecarWireError::Schema {
            observed,
            expected: expected_schema,
        });
    }
    if !object
        .get("artifact")
        .is_some_and(serde_json::Value::is_object)
        || !object
            .get("build_identity")
            .is_some_and(serde_json::Value::is_object)
        || !object
            .get("algorithm_version")
            .is_some_and(serde_json::Value::is_number)
    {
        return Err(MotionSidecarWireError::Missing("fixed header"));
    }
    bounded_list(object, "sources", MAX_MOTION_SIDECAR_SOURCES)?;
    bounded_list(object, "authored_scopes", MAX_MOTION_SIDECAR_SCOPES)?;
    bounded_list(
        object,
        "authored_residual_nodes",
        MAX_MOTION_SIDECAR_RESIDUAL_NODES,
    )?;
    bounded_list(
        object,
        "distinct_dynamic_states",
        MAX_MOTION_SIDECAR_DISTINCT_STATES,
    )?;
    bounded_list(object, "symmetry_fields", MAX_MOTION_SIDECAR_SYMMETRY_NODES)?;
    bounded_list(object, "warnings", MAX_MOTION_SIDECAR_WARNINGS)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &[u8] = include_bytes!("../fuzz/corpus/motion_sidecar_json/minimal.json");

    #[test]
    fn exact_schema_and_bounded_lists_are_required() {
        assert!(parse_motion_sidecar(MINIMAL, MOTION_SIDECAR_SCHEMA_VERSION).is_ok());
        assert!(parse_motion_sidecar(MINIMAL, 8).is_err());
        let mut duplicate = MINIMAL.to_vec();
        while duplicate.last().is_some_and(u8::is_ascii_whitespace) {
            duplicate.pop();
        }
        assert_eq!(duplicate.pop(), Some(b'}'));
        duplicate.extend_from_slice(br#", "warnings":[]}"#);
        assert!(parse_motion_sidecar(&duplicate, 9).is_err());
    }
}
