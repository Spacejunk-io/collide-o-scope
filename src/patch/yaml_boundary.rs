//! Dependency-light hostile YAML boundary shared by patch loading and fuzzing.
//!
//! This module deliberately stops at `serde_yaml::Value`: the application then
//! deserializes that already-bounded tree into `PatchState`, while cargo-fuzz
//! can exercise the exact lexical, depth, node, collection, and scalar laws
//! without linking the renderer or FFmpeg.

use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

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

struct YamlTokenParser<'input> {
    raw: Box<unsafe_libyaml::yaml_parser_t>,
    _input: PhantomData<&'input [u8]>,
}

impl<'input> YamlTokenParser<'input> {
    fn new(input: &'input [u8]) -> Result<Self, String> {
        let mut raw = Box::new(MaybeUninit::<unsafe_libyaml::yaml_parser_t>::uninit());
        // SAFETY: `raw` points to suitably aligned, writable storage for the
        // parser, exactly as required by `yaml_parser_initialize`.
        let initialized = unsafe { unsafe_libyaml::yaml_parser_initialize(raw.as_mut_ptr()) };
        if !initialized.ok {
            return Err("initialize YAML token scanner".to_string());
        }

        // SAFETY: initialization succeeded, so every field is initialized.
        // Retyping the allocation without moving its contents keeps the parser
        // at one stable address: `set_input_string` stores that address in its
        // read-handler state.
        let raw =
            unsafe { Box::from_raw(Box::into_raw(raw).cast::<unsafe_libyaml::yaml_parser_t>()) };
        let mut parser = Self {
            raw,
            _input: PhantomData,
        };
        // SAFETY: the parser is initialized and heap-stable, and `input`
        // remains alive for the parser's lifetime through `PhantomData`.
        unsafe {
            unsafe_libyaml::yaml_parser_set_encoding(
                parser.raw.as_mut(),
                unsafe_libyaml::YAML_UTF8_ENCODING,
            );
            unsafe_libyaml::yaml_parser_set_input_string(
                parser.raw.as_mut(),
                input.as_ptr(),
                input.len() as u64,
            );
        }
        Ok(parser)
    }

    fn next(&mut self) -> Result<YamlToken, String> {
        let mut raw = MaybeUninit::<unsafe_libyaml::yaml_token_t>::uninit();
        // SAFETY: both pointers are valid and writable. This parser is used
        // only through the token API, never mixed with parse/load calls.
        let scanned =
            unsafe { unsafe_libyaml::yaml_parser_scan(self.raw.as_mut(), raw.as_mut_ptr()) };
        if !scanned.ok {
            return Err(self.error());
        }
        // SAFETY: a successful scan initializes the complete token. Its Drop
        // implementation below releases every allocation owned by the token.
        Ok(YamlToken(unsafe { raw.assume_init() }))
    }

    fn error(&self) -> String {
        fn message(pointer: *const i8) -> Option<String> {
            if pointer.is_null() {
                None
            } else {
                // SAFETY: libyaml exposes these parser diagnostics as
                // NUL-terminated strings that remain valid until parser drop.
                Some(
                    unsafe { CStr::from_ptr(pointer) }
                        .to_string_lossy()
                        .into_owned(),
                )
            }
        }

        let problem =
            message(self.raw.problem).unwrap_or_else(|| "unknown scanner failure".to_string());
        let context = message(self.raw.context);
        let location = &self.raw.problem_mark;
        match context {
            Some(context) => format!(
                "scan YAML syntax at line {}, column {}: {context}: {problem}",
                location.line + 1,
                location.column + 1
            ),
            None => format!(
                "scan YAML syntax at line {}, column {}: {problem}",
                location.line + 1,
                location.column + 1
            ),
        }
    }
}

impl Drop for YamlTokenParser<'_> {
    fn drop(&mut self) {
        // SAFETY: this parser was successfully initialized and is deleted
        // exactly once while its borrowed input is still alive.
        unsafe { unsafe_libyaml::yaml_parser_delete(self.raw.as_mut()) }
    }
}

struct YamlToken(unsafe_libyaml::yaml_token_t);

impl YamlToken {
    fn kind(&self) -> unsafe_libyaml::yaml_token_type_t {
        self.0.type_
    }

    fn scalar_len(&self) -> usize {
        debug_assert_eq!(self.kind(), unsafe_libyaml::YAML_SCALAR_TOKEN);
        // SAFETY: callers check the token discriminant before reading the
        // scalar union arm.
        usize::try_from(unsafe { self.0.data.scalar.length }).unwrap_or(usize::MAX)
    }
}

impl Drop for YamlToken {
    fn drop(&mut self) {
        // SAFETY: a successful scan initialized this token, and this guard is
        // its unique owner and deletes it exactly once.
        unsafe { unsafe_libyaml::yaml_token_delete(&mut self.0) }
    }
}

/// Reject graph indirection before constructing `serde_yaml::Value`.
/// `PatchState` is a tree, so anchors/aliases add no capability; excluding
/// them also prevents a tiny source from expanding into a large object graph.
fn validate_yaml_lexical_boundary(bytes: &[u8], limits: YamlLimits) -> Result<(), String> {
    std::str::from_utf8(bytes).map_err(|error| format!("patch is not UTF-8: {error}"))?;

    // Use the same libyaml grammar that backs serde_yaml so quotes embedded in
    // plain scalars, comments, and block-scalar bodies can never desynchronise
    // the policy gate. Tokens are produced incrementally; aliases are rejected
    // before serde_yaml can resolve them into a value graph.
    let mut parser = YamlTokenParser::new(bytes)?;
    let mut flow_depth = 0_usize;
    let mut syntax_depth = 0_usize;
    let mut structural_tokens = 0_usize;
    loop {
        let token = parser.next()?;
        let kind = token.kind();
        match kind {
            unsafe_libyaml::YAML_NO_TOKEN => {
                return Err("YAML token scanner ended before the stream terminator".to_string());
            }
            unsafe_libyaml::YAML_ANCHOR_TOKEN => {
                return Err("YAML anchors are not accepted in patch files".to_string());
            }
            unsafe_libyaml::YAML_ALIAS_TOKEN => {
                return Err("YAML aliases are not accepted in patch files".to_string());
            }
            unsafe_libyaml::YAML_SCALAR_TOKEN => {
                let scalar_len = token.scalar_len();
                if scalar_len > limits.max_scalar_bytes {
                    return Err(format!(
                        "YAML scalar exceeds the {}-byte boundary",
                        limits.max_scalar_bytes
                    ));
                }
            }
            unsafe_libyaml::YAML_FLOW_SEQUENCE_START_TOKEN
            | unsafe_libyaml::YAML_FLOW_MAPPING_START_TOKEN => {
                flow_depth += 1;
                if flow_depth > limits.max_depth {
                    return Err(format!(
                        "YAML flow depth exceeds the {}-level boundary",
                        limits.max_depth
                    ));
                }
                syntax_depth += 1;
            }
            unsafe_libyaml::YAML_FLOW_SEQUENCE_END_TOKEN
            | unsafe_libyaml::YAML_FLOW_MAPPING_END_TOKEN => {
                flow_depth = flow_depth.saturating_sub(1);
                syntax_depth = syntax_depth.saturating_sub(1);
            }
            unsafe_libyaml::YAML_BLOCK_SEQUENCE_START_TOKEN
            | unsafe_libyaml::YAML_BLOCK_MAPPING_START_TOKEN => {
                syntax_depth += 1;
            }
            unsafe_libyaml::YAML_BLOCK_END_TOKEN => {
                syntax_depth = syntax_depth.saturating_sub(1);
            }
            _ => {}
        }
        if syntax_depth > limits.max_depth {
            return Err(format!(
                "YAML syntax depth exceeds the {}-level boundary",
                limits.max_depth
            ));
        }
        if matches!(
            kind,
            unsafe_libyaml::YAML_FLOW_SEQUENCE_START_TOKEN
                | unsafe_libyaml::YAML_FLOW_SEQUENCE_END_TOKEN
                | unsafe_libyaml::YAML_FLOW_MAPPING_START_TOKEN
                | unsafe_libyaml::YAML_FLOW_MAPPING_END_TOKEN
                | unsafe_libyaml::YAML_BLOCK_ENTRY_TOKEN
                | unsafe_libyaml::YAML_FLOW_ENTRY_TOKEN
                | unsafe_libyaml::YAML_KEY_TOKEN
                | unsafe_libyaml::YAML_VALUE_TOKEN
        ) {
            structural_tokens += 1;
        }
        if structural_tokens > limits.max_structural_tokens {
            return Err(format!(
                "YAML structure exceeds the {}-token boundary",
                limits.max_structural_tokens
            ));
        }
        if kind == unsafe_libyaml::YAML_STREAM_END_TOKEN {
            break;
        }
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

#[cfg(test)]
mod tests {
    #[test]
    fn canonical_plain_scalars_with_embedded_quotes_reparse() {
        for input in [
            b"'mx+'''".as_slice(),
            b"\"a'\"".as_slice(),
            b"'a\"'".as_slice(),
            b"'a''b'".as_slice(),
            b"\"a\\\"b\"".as_slice(),
            b"key: can't\n".as_slice(),
            b"key: a\"b\n".as_slice(),
        ] {
            let value = super::parse_patch_yaml_value(input).expect("quoted source is valid YAML");
            let canonical = serde_yaml::to_string(&value).expect("value serializes");
            let reparsed = super::parse_patch_yaml_value(canonical.as_bytes())
                .expect("canonical plain scalar with an embedded quote reparses");
            assert_eq!(reparsed, value);
        }
    }

    #[test]
    fn true_unterminated_quotes_remain_rejected() {
        for input in [b"'unterminated".as_slice(), b"\"unterminated".as_slice()] {
            let error = super::parse_patch_yaml_value(input).expect_err("bad quote must fail");
            assert!(error.contains("scan YAML syntax"), "{error}");
        }
    }

    #[test]
    fn quote_characters_cannot_mask_graph_indirection() {
        for input in [
            b"value: &anchor []\n".as_slice(),
            b"value: *alias\n".as_slice(),
            b"left': ok\npayload: &a [1]\ncopy: *a\nright': ok\n".as_slice(),
            b"left\": ok\npayload: &a [1]\ncopy: *a\nright\": ok\n".as_slice(),
            b"a: http:'\nb: &x [1]\nc: *x\nd: http:'\n".as_slice(),
            b"a: why?'\nb: &x [1]\nc: *x\nd: why?'\n".as_slice(),
        ] {
            let error =
                super::parse_patch_yaml_value(input).expect_err("graph indirection must fail");
            assert!(
                error.contains("anchors") || error.contains("aliases"),
                "{error}"
            );
        }
    }

    #[test]
    fn indicators_inside_scalar_data_and_comments_are_ordinary_text() {
        for input in [
            b"value: 'literal &anchor *alias'\n".as_slice(),
            b"value: \"literal &anchor *alias\"\n".as_slice(),
            b"value: literal &anchor *alias\n".as_slice(),
            b"value: ordinary # ignored ' \" &anchor *alias [ ]\n".as_slice(),
        ] {
            super::parse_patch_yaml_value(input).expect("indicator text is ordinary scalar data");
        }
    }

    #[test]
    fn block_scalar_bodies_do_not_change_scanner_state() {
        for input in [
            b"value: |-\n  'starts &anchor *alias [ ] { } # text\n".as_slice(),
            b"value: >-\n  \"starts &anchor *alias [ ] { } # text\n".as_slice(),
        ] {
            super::parse_patch_yaml_value(input)
                .expect("block scalar body is ordinary scalar data");
        }
    }

    #[test]
    fn canonical_block_scalar_content_is_not_mistaken_for_structural_depth() {
        let input = format!("\"x\\n{}y\"", " ".repeat(257));
        let value =
            super::parse_patch_yaml_value(input.as_bytes()).expect("quoted source is valid YAML");
        let canonical = serde_yaml::to_string(&value).expect("multiline value serializes");
        let reparsed = super::parse_patch_yaml_value(canonical.as_bytes())
            .expect("literal-block content indentation is scalar data");
        assert_eq!(reparsed, value);
    }

    #[test]
    fn quote_dense_plain_scalar_stays_within_the_linear_scanner_path() {
        let mut input = Vec::with_capacity(1024 * 1024 + 2);
        input.push(b'a');
        input.extend(std::iter::repeat_n(b'\'', 1024 * 1024));
        input.push(b'\n');
        super::parse_patch_yaml_value(&input).expect("quote-dense plain scalar remains bounded");
    }
}
