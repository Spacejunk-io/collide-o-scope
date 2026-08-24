//! Dependency-light hostile YAML boundary shared by patch loading and fuzzing.
//!
//! This module deliberately stops at `serde_yaml::Value`: the application then
//! deserializes that already-bounded tree into `PatchState`, while cargo-fuzz
//! can exercise the exact lexical, depth, node, collection, and scalar laws
//! without linking the renderer or FFmpeg.

pub const MAX_PATCH_FILE_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const MAX_PATCH_YAML_DEPTH: usize = 64;
const MAX_PATCH_YAML_NODES: usize = 250_000;
const MAX_PATCH_YAML_COLLECTION_ENTRIES: usize = 250_000;
const MAX_PATCH_YAML_SCALAR_BYTES: usize = 4 * 1024 * 1024;
const MAX_PATCH_YAML_STRUCTURAL_TOKENS: usize = 500_000;

#[derive(Clone, Copy)]
pub(crate) struct YamlLimits {
    pub(crate) max_depth: usize,
    pub(crate) max_nodes: usize,
    pub(crate) max_collection_entries: usize,
    pub(crate) max_scalar_bytes: usize,
    pub(crate) max_structural_tokens: usize,
}

const PATCH_YAML_LIMITS: YamlLimits = YamlLimits {
    max_depth: MAX_PATCH_YAML_DEPTH,
    max_nodes: MAX_PATCH_YAML_NODES,
    max_collection_entries: MAX_PATCH_YAML_COLLECTION_ENTRIES,
    max_scalar_bytes: MAX_PATCH_YAML_SCALAR_BYTES,
    max_structural_tokens: MAX_PATCH_YAML_STRUCTURAL_TOKENS,
};

fn yaml_indicator_at(bytes: &[u8], index: usize) -> bool {
    let previous_is_boundary = index == 0
        || bytes[index - 1].is_ascii_whitespace()
        || matches!(bytes[index - 1], b'[' | b'{' | b',' | b':' | b'?' | b'-');
    let next_is_name = bytes.get(index + 1).is_some_and(|next| {
        !next.is_ascii_whitespace() && !matches!(next, b',' | b'[' | b']' | b'{' | b'}' | b'#')
    });
    previous_is_boundary && next_is_name
}

/// Reject graph indirection before constructing `serde_yaml::Value`.
/// `PatchState` is a tree, so anchors/aliases add no capability; excluding
/// them also prevents a tiny source from expanding into a large object graph.
fn validate_yaml_lexical_boundary(bytes: &[u8], limits: YamlLimits) -> Result<(), String> {
    std::str::from_utf8(bytes).map_err(|error| format!("patch is not UTF-8: {error}"))?;
    let mut in_single = false;
    let mut doubled_single_quote = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut in_comment = false;
    let mut flow_depth = 0_usize;
    let mut structural_tokens = 0_usize;
    let mut leading_spaces = 0_usize;
    let mut at_line_start = true;

    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            in_comment = false;
            at_line_start = true;
            leading_spaces = 0;
            continue;
        }
        if in_comment {
            continue;
        }
        if at_line_start && byte == b' ' {
            leading_spaces += 1;
            if leading_spaces > limits.max_depth.saturating_mul(4) {
                return Err(format!(
                    "YAML indentation exceeds the {}-level boundary",
                    limits.max_depth
                ));
            }
            continue;
        }
        at_line_start = false;

        if in_double {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_double = false;
            }
            continue;
        }
        if in_single {
            if doubled_single_quote {
                doubled_single_quote = false;
                continue;
            }
            if byte == b'\'' {
                if bytes.get(index + 1) == Some(&b'\'') {
                    doubled_single_quote = true;
                    continue;
                }
                in_single = false;
            }
            continue;
        }
        match byte {
            b'#' => in_comment = true,
            b'"' => in_double = true,
            b'\'' => in_single = true,
            b'&' if yaml_indicator_at(bytes, index) => {
                return Err("YAML anchors are not accepted in patch files".to_string());
            }
            b'*' if yaml_indicator_at(bytes, index) => {
                return Err("YAML aliases are not accepted in patch files".to_string());
            }
            b'[' | b'{' => {
                flow_depth += 1;
                if flow_depth > limits.max_depth {
                    return Err(format!(
                        "YAML flow depth exceeds the {}-level boundary",
                        limits.max_depth
                    ));
                }
                structural_tokens += 1;
            }
            b']' | b'}' => {
                flow_depth = flow_depth.saturating_sub(1);
                structural_tokens += 1;
            }
            b':' | b',' | b'-' => structural_tokens += 1,
            _ => {}
        }
        if structural_tokens > limits.max_structural_tokens {
            return Err(format!(
                "YAML structure exceeds the {}-token boundary",
                limits.max_structural_tokens
            ));
        }
    }
    if in_single || in_double {
        return Err("YAML contains an unterminated quoted scalar".to_string());
    }
    Ok(())
}

pub(crate) fn validate_yaml_value_boundary(
    root: &serde_yaml::Value,
    limits: YamlLimits,
) -> Result<(), String> {
    let mut pending = vec![(root, 1_usize)];
    let mut nodes = 0_usize;
    let mut collection_entries = 0_usize;
    while let Some((value, depth)) = pending.pop() {
        nodes += 1;
        if nodes > limits.max_nodes {
            return Err(format!(
                "YAML node count exceeds the {}-node boundary",
                limits.max_nodes
            ));
        }
        if depth > limits.max_depth {
            return Err(format!(
                "YAML value depth exceeds the {}-level boundary",
                limits.max_depth
            ));
        }
        match value {
            serde_yaml::Value::String(text) => {
                if text.len() > limits.max_scalar_bytes {
                    return Err(format!(
                        "YAML scalar exceeds the {}-byte boundary",
                        limits.max_scalar_bytes
                    ));
                }
            }
            serde_yaml::Value::Sequence(values) => {
                collection_entries = collection_entries.saturating_add(values.len());
                pending.extend(values.iter().map(|value| (value, depth + 1)));
            }
            serde_yaml::Value::Mapping(values) => {
                collection_entries = collection_entries.saturating_add(values.len());
                for (key, value) in values {
                    pending.push((key, depth + 1));
                    pending.push((value, depth + 1));
                }
            }
            serde_yaml::Value::Tagged(tagged) => pending.push((&tagged.value, depth + 1)),
            serde_yaml::Value::Null | serde_yaml::Value::Bool(_) | serde_yaml::Value::Number(_) => {
            }
        }
        if collection_entries > limits.max_collection_entries {
            return Err(format!(
                "YAML collections exceed the {}-entry boundary",
                limits.max_collection_entries
            ));
        }
    }
    Ok(())
}

pub(crate) fn parse_patch_yaml_value(bytes: &[u8]) -> Result<serde_yaml::Value, String> {
    if bytes.is_empty() {
        return Err("patch file is empty".to_string());
    }
    if bytes.len() > MAX_PATCH_FILE_BYTES {
        return Err(format!(
            "patch is {} bytes; limit is {MAX_PATCH_FILE_BYTES}",
            bytes.len()
        ));
    }
    validate_yaml_lexical_boundary(bytes, PATCH_YAML_LIMITS)?;
    let value = serde_yaml::from_slice::<serde_yaml::Value>(bytes)
        .map_err(|error| format!("parse YAML tree: {error}"))?;
    validate_yaml_value_boundary(&value, PATCH_YAML_LIMITS)?;
    Ok(value)
}
