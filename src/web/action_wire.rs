//! Bounded JSON boundary for WebSocket actions.
//!
//! The full `WebAction` vocabulary stays in `state.rs`, but hostile JSON is
//! admitted here first through a dependency-light parser shared by production
//! and cargo-fuzz. Duplicate keys are rejected instead of silently taking the
//! last value.

use std::fmt;

use serde::de::DeserializeOwned;

#[allow(
    clippy::duplicate_mod,
    reason = "the standalone fuzz seam and application root intentionally share this parser source"
)]
#[path = "../bounded_json.rs"]
mod bounded_json;

pub const MAX_WEB_ACTION_BYTES: usize = 16 * 1024;
pub const MAX_WEB_ACTION_DEPTH: usize = 32;
pub const MAX_WEB_ACTION_NODES: usize = 4_096;
pub const MAX_WEB_ACTION_KEY_BYTES: usize = 128;
pub const MAX_WEB_ACTION_STRING_BYTES: usize = 4 * 1024;

const WEB_ACTION_LIMITS: bounded_json::JsonLimits = bounded_json::JsonLimits {
    max_bytes: MAX_WEB_ACTION_BYTES,
    max_depth: MAX_WEB_ACTION_DEPTH,
    max_nodes: MAX_WEB_ACTION_NODES,
    max_key_bytes: MAX_WEB_ACTION_KEY_BYTES,
    max_string_bytes: MAX_WEB_ACTION_STRING_BYTES,
};

#[derive(Debug)]
pub enum WebActionJsonError {
    Boundary(bounded_json::BoundedJsonError),
    Schema(serde_json::Error),
    Root,
    ActionTag,
}

impl fmt::Display for WebActionJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boundary(error) => write!(formatter, "WebAction {error}"),
            Self::Schema(error) => write!(formatter, "WebAction schema: {error}"),
            Self::Root => formatter.write_str("WebAction JSON root must be an object"),
            Self::ActionTag => formatter.write_str(
                "WebAction JSON must carry one printable non-empty action tag of at most 128 bytes",
            ),
        }
    }
}

impl std::error::Error for WebActionJsonError {}

pub fn parse_web_action_value(bytes: &[u8]) -> Result<serde_json::Value, WebActionJsonError> {
    let value = bounded_json::parse_bounded_json(bytes, WEB_ACTION_LIMITS)
        .map_err(WebActionJsonError::Boundary)?;
    let object = value.as_object().ok_or(WebActionJsonError::Root)?;
    let action = object
        .get("action")
        .and_then(serde_json::Value::as_str)
        .ok_or(WebActionJsonError::ActionTag)?;
    if action.is_empty() || action.len() > 128 || action.chars().any(char::is_control) {
        return Err(WebActionJsonError::ActionTag);
    }
    Ok(value)
}

pub fn parse_web_action<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, WebActionJsonError> {
    let value = parse_web_action_value(bytes)?;
    serde_json::from_value(value).map_err(WebActionJsonError::Schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_depth_node_string_and_byte_limits_fail_closed() {
        assert!(
            parse_web_action_value(br#"{"action":"set_morph","action":"cancel_export"}"#).is_err()
        );
        let mut deep = String::from(r#"{"action":"quantized","inner":"#);
        deep.push_str(&"[".repeat(MAX_WEB_ACTION_DEPTH + 2));
        deep.push_str(&"]".repeat(MAX_WEB_ACTION_DEPTH + 2));
        deep.push('}');
        assert!(parse_web_action_value(deep.as_bytes()).is_err());
        let oversized = vec![b' '; MAX_WEB_ACTION_BYTES + 1];
        assert!(matches!(
            parse_web_action_value(&oversized),
            Err(WebActionJsonError::Boundary(
                bounded_json::BoundedJsonError::OverBytes { .. }
            ))
        ));
    }

    #[test]
    fn canonical_action_reparses_through_the_same_boundary() {
        let parsed = parse_web_action_value(br#"{"action":"set_morph","value":0.5}"#).unwrap();
        let canonical = serde_json::to_vec(&parsed).unwrap();
        assert_eq!(parse_web_action_value(&canonical).unwrap(), parsed);
    }
}
