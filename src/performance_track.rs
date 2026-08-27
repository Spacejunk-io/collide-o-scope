//! Portable performance-take event contract shared by live authoring and
//! export.
//!
//! This module owns CPU state and laws only. It deliberately has no `wgpu`,
//! clock, filesystem, or UI dependency: a caller supplies an already-derived
//! 30 Hz reference-tick address and one authored control edit, and the take
//! answers with a bounded, checksummable event stream that replays identically
//! offline. Wall time never enters the take, the checksum, or anything derived
//! from them — the reference tick is the only address an edit ever has.
//!
//! The substrate is deliberately the gesture-track contract (`gesture.rs`):
//! the same 30 Hz authoring reference, the same quantized-codes-only
//! representation, the same domain-separated SHA-256 field stream with the
//! honesty flags inside the digest, the same bounded serde on both sides, and
//! the same derive-tick-before-adding-the-delta recorder clock. What a take
//! records is different in kind: not strokes of a canvas but `(tick,
//! param_address, value)` — the accepted authored value edits the program
//! actually performed, so a take built slowly can be played back in real time
//! against completely different footage. The law is derived from BENDR (MIT,
//! © 2026 Steve Blythe); the house adaptation records accepted edits at the
//! coalesced drain rather than change-sampling a flat control surface, and the
//! patch — which carries the take whole — is the opening state, so no
//! synthetic keyframe exists.

#![allow(
    dead_code,
    reason = "B9 freezes the portable performance-take contract before every host surface that consumes it lands"
)]

use std::fmt;

use serde::{de, ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::effects::params::TEMPORAL_REFERENCE_FPS;

/// Append-only algorithm version for the portable performance event stream. It
/// appears both in the sidecar document and inside the hashed field stream, so
/// a future encoding can never be mistaken for this one.
pub const PERFORMANCE_ALGORITHM_VERSION: u16 = 1;

/// Performance events are addressed on the same 30 Hz authoring reference as
/// the temporal history ring and the gesture track. This is that constant, not
/// a second literal: a take recorded at 24, 30, or 60 fps must land on the
/// same tick address.
pub const PERFORMANCE_REFERENCE_FPS: f32 = TEMPORAL_REFERENCE_FPS;

/// Hard cap for recorded performance events retained for deterministic offline
/// replay. Reaching it sets `truncated` and refuses further recording; it
/// never panics and never silently drops the newest edit in place of an older
/// one.
pub const MAX_PERFORMANCE_EVENTS: usize = 16_384;

/// Hard cap for distinct control addresses one take may intern. The event
/// address lane is a `u16`, so the cap is also what keeps every index
/// representable.
pub const MAX_PERFORMANCE_ADDRESSES: usize = 256;

/// Byte ceiling for a serialized performance-take document, enforced on both
/// encode and decode.
pub const MAX_PERFORMANCE_SERIALIZED_BYTES: usize = 512 * 1024;

/// Bounds on the authored identity strings a control address may carry. The
/// 64-byte param bound matches the wire gate's `valid_identifier` ceiling.
pub const MAX_PERFORMANCE_PARAM_BYTES: usize = 64;

/// A discrete law's closed vocabulary is small by construction. Sixty-four is
/// large enough for the engine-owned modulation-source table (including its
/// compatibility aliases) while keeping hostile documents tightly bounded.
pub const MAX_PERFORMANCE_VOCAB_TOKENS: usize = 64;

/// Domain separator for the canonical checksum. The version literal appears
/// here as well as in the hashed `version` field, exactly as the gesture track
/// and the recovery journal repeat theirs.
pub const PERFORMANCE_CHECKSUM_DOMAIN: &[u8] = b"collide-o-scope/performance-take/v1\0";

/// First append-only control code added by the v2 address-capability tranche.
/// The take document and checksum domain remain version 1.
pub const PERFORMANCE_V2_FIRST_CONTROL_CODE: u8 = 15;

/// Fixed width of one encoded event: `tick:u32le`, `address:u16le`,
/// `value:u16le`.
pub const PERFORMANCE_EVENT_ENCODED_BYTES: usize = 8;

/// Append-only checksum flag bits. The flags are part of the hashed stream, so
/// a truncated or explicitly incomplete take can never present itself under a
/// complete take's digest.
pub const PERFORMANCE_FLAG_TRUNCATED: u16 = 1 << 0;
pub const PERFORMANCE_FLAG_INCOMPLETE: u16 = 1 << 1;

/// Q16 scale for the continuous value lattice.
const PERFORMANCE_Q16_SCALE: f32 = 65_535.0;

/// One control identity a take can address.
///
/// The vocabulary is closed and append-only: every arm names an existing
/// coalescible absolute-value wire action, and a layer is identified by its
/// saved stack position — the patch-persistent identity morph slots and saved
/// donor positions already use — never by a process-lifetime live layer ID,
/// which a patch load deliberately re-mints. Values only, by law: topology
/// (including adding or removing routes) and safety controls (blackout,
/// freeze, pause) have no address here and can never enter a take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceControl {
    /// Master effect value (`set_param`).
    Master { param: String },
    /// Master spatial transform value (`set_master_transform`).
    MasterTransform { param: String },
    /// Layer transport/keying value (`set_layer_param`).
    LayerParam { layer: u32, param: String },
    /// Layer effect value (`set_layer_effect`).
    LayerEffect { layer: u32, param: String },
    /// Layer spatial transform value (`set_layer_transform`).
    LayerTransform { layer: u32, param: String },
    /// Layer visibility (`set_layer_visibility`).
    LayerVisible { layer: u32 },
    /// Pattern-synth value on a pattern layer (`set_layer_pattern`).
    LayerPattern { layer: u32, param: String },
    /// NTSC value (`set_ntsc_param`).
    Ntsc { param: String },
    /// Temporal value (`set_temporal`).
    Temporal { param: String },
    /// Master Motion value (`set_motion`, master scope).
    MotionMaster { param: String },
    /// Layer Motion value (`set_motion`, layer scope).
    MotionLayer { layer: u32, param: String },
    /// The A/B bus crossfade (`set_composition_bus_crossfade`).
    BusCrossfade,
    /// A bus-mixer value (`set_composition_bus_mix`).
    BusMix { param: String },
    /// The Morph position fader (`set_morph`).
    MorphPosition,
    /// A gesture-canvas authored scalar (`set_gesture_canvas`).
    GestureCanvas { param: String },
    /// A master-rack node parameter, addressed by its patch-persistent node ID.
    RackNodeMaster {
        node: u64,
        node_kind: String,
        param: String,
    },
    /// A layer-rack node parameter. The layer uses its saved stack position;
    /// the node uses its patch-persistent ID.
    RackNodeLayer {
        layer: u32,
        node: u64,
        node_kind: String,
        param: String,
    },
    /// A composition-group rack node parameter, addressed by the group's and
    /// node's patch-persistent IDs.
    RackNodeGroup {
        group: u64,
        node: u64,
        node_kind: String,
        param: String,
    },
    /// A composition-group value.
    GroupParam { group: u64, param: String },
    /// A composition-group matte value.
    GroupMatteParam { group: u64, param: String },
    /// A text-page value on a saved layer position.
    LayerText { layer: u32, param: String },
    /// The closed Morph blend-law token.
    MorphLaw,
    /// A Morph glide whose target is part of the address identity and whose
    /// value lane carries the duration.
    MorphGlide { target_q16: u16 },
    /// The closed Morph capture-slot token.
    MorphCapture,
    /// One value on a routing entry addressed by saved routing position.
    Routing { routing: u32, param: String },
}

impl PerformanceControl {
    /// Permanent append-only canonical-stream code. Never renumber an existing
    /// entry.
    pub const fn code(&self) -> u8 {
        match self {
            Self::Master { .. } => 0,
            Self::MasterTransform { .. } => 1,
            Self::LayerParam { .. } => 2,
            Self::LayerEffect { .. } => 3,
            Self::LayerTransform { .. } => 4,
            Self::LayerVisible { .. } => 5,
            Self::LayerPattern { .. } => 6,
            Self::Ntsc { .. } => 7,
            Self::Temporal { .. } => 8,
            Self::MotionMaster { .. } => 9,
            Self::MotionLayer { .. } => 10,
            Self::BusCrossfade => 11,
            Self::BusMix { .. } => 12,
            Self::MorphPosition => 13,
            Self::GestureCanvas { .. } => 14,
            Self::RackNodeMaster { .. } => 15,
            Self::RackNodeLayer { .. } => 16,
            Self::RackNodeGroup { .. } => 17,
            Self::GroupParam { .. } => 18,
            Self::GroupMatteParam { .. } => 19,
            Self::LayerText { .. } => 20,
            Self::MorphLaw => 21,
            Self::MorphGlide { .. } => 22,
            Self::MorphCapture => 23,
            Self::Routing { .. } => 24,
        }
    }

    /// Wire token for the document encoding. Tokens are permanent; a future
    /// arm appends a new token and never reuses one.
    pub const fn kind_token(&self) -> &'static str {
        match self {
            Self::Master { .. } => "master",
            Self::MasterTransform { .. } => "master_transform",
            Self::LayerParam { .. } => "layer_param",
            Self::LayerEffect { .. } => "layer_effect",
            Self::LayerTransform { .. } => "layer_transform",
            Self::LayerVisible { .. } => "layer_visible",
            Self::LayerPattern { .. } => "layer_pattern",
            Self::Ntsc { .. } => "ntsc",
            Self::Temporal { .. } => "temporal",
            Self::MotionMaster { .. } => "motion_master",
            Self::MotionLayer { .. } => "motion_layer",
            Self::BusCrossfade => "bus_crossfade",
            Self::BusMix { .. } => "bus_mix",
            Self::MorphPosition => "morph_position",
            Self::GestureCanvas { .. } => "gesture_canvas",
            Self::RackNodeMaster { .. } => "rack_node_master",
            Self::RackNodeLayer { .. } => "rack_node_layer",
            Self::RackNodeGroup { .. } => "rack_node_group",
            Self::GroupParam { .. } => "group_param",
            Self::GroupMatteParam { .. } => "group_matte_param",
            Self::LayerText { .. } => "layer_text",
            Self::MorphLaw => "morph_law",
            Self::MorphGlide { .. } => "morph_glide",
            Self::MorphCapture => "morph_capture",
            Self::Routing { .. } => "routing",
        }
    }

    /// The saved layer position the control addresses, if any.
    pub const fn layer(&self) -> Option<u32> {
        match self {
            Self::LayerParam { layer, .. }
            | Self::LayerEffect { layer, .. }
            | Self::LayerTransform { layer, .. }
            | Self::LayerVisible { layer }
            | Self::LayerPattern { layer, .. }
            | Self::MotionLayer { layer, .. }
            | Self::RackNodeLayer { layer, .. }
            | Self::LayerText { layer, .. } => Some(*layer),
            _ => None,
        }
    }

    /// The patch-persistent node ID the control addresses, if any.
    pub const fn node(&self) -> Option<u64> {
        match self {
            Self::RackNodeMaster { node, .. }
            | Self::RackNodeLayer { node, .. }
            | Self::RackNodeGroup { node, .. } => Some(*node),
            _ => None,
        }
    }

    /// The patch-persistent composition-group ID the control addresses, if
    /// any.
    pub const fn group(&self) -> Option<u64> {
        match self {
            Self::RackNodeGroup { group, .. }
            | Self::GroupParam { group, .. }
            | Self::GroupMatteParam { group, .. } => Some(*group),
            _ => None,
        }
    }

    /// The rack-node kind captured with a node address, if any.
    pub fn node_kind(&self) -> Option<&str> {
        match self {
            Self::RackNodeMaster { node_kind, .. }
            | Self::RackNodeLayer { node_kind, .. }
            | Self::RackNodeGroup { node_kind, .. } => Some(node_kind),
            _ => None,
        }
    }

    /// The Q16 Morph target captured in a glide address, if any.
    pub const fn target_q16(&self) -> Option<u16> {
        match self {
            Self::MorphGlide { target_q16 } => Some(*target_q16),
            _ => None,
        }
    }

    /// The saved routing position the control addresses, if any.
    pub const fn routing(&self) -> Option<u32> {
        match self {
            Self::Routing { routing, .. } => Some(*routing),
            _ => None,
        }
    }

    /// The parameter name the control addresses, if any.
    pub fn param(&self) -> Option<&str> {
        match self {
            Self::Master { param }
            | Self::MasterTransform { param }
            | Self::LayerParam { param, .. }
            | Self::LayerEffect { param, .. }
            | Self::LayerTransform { param, .. }
            | Self::LayerPattern { param, .. }
            | Self::Ntsc { param }
            | Self::Temporal { param }
            | Self::MotionMaster { param }
            | Self::MotionLayer { param, .. }
            | Self::BusMix { param }
            | Self::GestureCanvas { param }
            | Self::RackNodeMaster { param, .. }
            | Self::RackNodeLayer { param, .. }
            | Self::RackNodeGroup { param, .. }
            | Self::GroupParam { param, .. }
            | Self::GroupMatteParam { param, .. }
            | Self::LayerText { param, .. }
            | Self::Routing { param, .. } => Some(param),
            Self::LayerVisible { .. }
            | Self::BusCrossfade
            | Self::MorphPosition
            | Self::MorphLaw
            | Self::MorphGlide { .. }
            | Self::MorphCapture => None,
        }
    }

    fn needs_param(kind: &str) -> bool {
        !matches!(
            kind,
            "layer_visible"
                | "bus_crossfade"
                | "morph_position"
                | "morph_law"
                | "morph_glide"
                | "morph_capture"
        )
    }

    fn needs_layer(kind: &str) -> bool {
        matches!(
            kind,
            "layer_param"
                | "layer_effect"
                | "layer_transform"
                | "layer_visible"
                | "layer_pattern"
                | "motion_layer"
                | "rack_node_layer"
                | "layer_text"
        )
    }

    fn needs_node(kind: &str) -> bool {
        matches!(
            kind,
            "rack_node_master" | "rack_node_layer" | "rack_node_group"
        )
    }

    fn needs_group(kind: &str) -> bool {
        matches!(
            kind,
            "rack_node_group" | "group_param" | "group_matte_param"
        )
    }

    fn needs_node_kind(kind: &str) -> bool {
        Self::needs_node(kind)
    }

    fn needs_target_q16(kind: &str) -> bool {
        kind == "morph_glide"
    }

    fn needs_routing(kind: &str) -> bool {
        kind == "routing"
    }

    fn parse_nonzero_decimal_id(kind: &str, value: String) -> Result<u64, PerformanceError> {
        if value.is_empty()
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || value.starts_with('0')
        {
            return Err(PerformanceError::MalformedAddress(kind.to_string()));
        }
        let parsed = value
            .parse::<u64>()
            .map_err(|_| PerformanceError::MalformedAddress(kind.to_string()))?;
        if parsed == 0 || parsed.to_string() != value {
            return Err(PerformanceError::MalformedAddress(kind.to_string()));
        }
        Ok(parsed)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        kind: &str,
        layer: Option<u32>,
        param: Option<String>,
        node_id: Option<String>,
        group_id: Option<String>,
        node_kind: Option<String>,
        target_q16: Option<u16>,
        routing: Option<u32>,
    ) -> Result<Self, PerformanceError> {
        let wants_param = Self::needs_param(kind);
        let wants_layer = Self::needs_layer(kind);
        let wants_node = Self::needs_node(kind);
        let wants_group = Self::needs_group(kind);
        let wants_node_kind = Self::needs_node_kind(kind);
        let wants_target_q16 = Self::needs_target_q16(kind);
        let wants_routing = Self::needs_routing(kind);
        if wants_param != param.is_some()
            || wants_layer != layer.is_some()
            || wants_node != node_id.is_some()
            || wants_group != group_id.is_some()
            || wants_node_kind != node_kind.is_some()
            || wants_target_q16 != target_q16.is_some()
            || wants_routing != routing.is_some()
        {
            return Err(PerformanceError::MalformedAddress(kind.to_string()));
        }
        let param = param.unwrap_or_default();
        let layer = layer.unwrap_or_default();
        let node = node_id
            .map(|value| Self::parse_nonzero_decimal_id(kind, value))
            .transpose()?
            .unwrap_or_default();
        let group = group_id
            .map(|value| Self::parse_nonzero_decimal_id(kind, value))
            .transpose()?
            .unwrap_or_default();
        let node_kind = node_kind.unwrap_or_default();
        let target_q16 = target_q16.unwrap_or_default();
        let routing = routing.unwrap_or_default();
        let control = match kind {
            "master" => Self::Master { param },
            "master_transform" => Self::MasterTransform { param },
            "layer_param" => Self::LayerParam { layer, param },
            "layer_effect" => Self::LayerEffect { layer, param },
            "layer_transform" => Self::LayerTransform { layer, param },
            "layer_visible" => Self::LayerVisible { layer },
            "layer_pattern" => Self::LayerPattern { layer, param },
            "ntsc" => Self::Ntsc { param },
            "temporal" => Self::Temporal { param },
            "motion_master" => Self::MotionMaster { param },
            "motion_layer" => Self::MotionLayer { layer, param },
            "bus_crossfade" => Self::BusCrossfade,
            "bus_mix" => Self::BusMix { param },
            "morph_position" => Self::MorphPosition,
            "gesture_canvas" => Self::GestureCanvas { param },
            "rack_node_master" => Self::RackNodeMaster {
                node,
                node_kind,
                param,
            },
            "rack_node_layer" => Self::RackNodeLayer {
                layer,
                node,
                node_kind,
                param,
            },
            "rack_node_group" => Self::RackNodeGroup {
                group,
                node,
                node_kind,
                param,
            },
            "group_param" => Self::GroupParam { group, param },
            "group_matte_param" => Self::GroupMatteParam { group, param },
            "layer_text" => Self::LayerText { layer, param },
            "morph_law" => Self::MorphLaw,
            "morph_glide" => Self::MorphGlide { target_q16 },
            "morph_capture" => Self::MorphCapture,
            "routing" => Self::Routing { routing, param },
            other => return Err(PerformanceError::MalformedAddress(other.to_string())),
        };
        control.validate()?;
        Ok(control)
    }

    /// Structural well-formedness: bounded identity strings only. Value
    /// semantics belong to the address's law, not the control.
    fn validate(&self) -> Result<(), PerformanceError> {
        if let Some(param) = self.param() {
            if param.is_empty() || param.len() > MAX_PERFORMANCE_PARAM_BYTES {
                return Err(PerformanceError::ParamBytes(param.len()));
            }
        }
        if self.node().is_some_and(|node| node == 0) || self.group().is_some_and(|group| group == 0)
        {
            return Err(PerformanceError::MalformedAddress(
                self.kind_token().to_string(),
            ));
        }
        if let Some(node_kind) = self.node_kind() {
            if node_kind.is_empty() || node_kind.len() > MAX_PERFORMANCE_PARAM_BYTES {
                return Err(PerformanceError::NodeKindBytes(node_kind.len()));
            }
        }
        Ok(())
    }

    /// Operator-facing name for diagnostics. Stable, path-free, and never
    /// parsed back — the typed fields are the identity.
    pub fn describe(&self) -> String {
        let mut parts = vec![self.kind_token().to_string()];
        if let Some(layer) = self.layer() {
            parts.push(layer.to_string());
        }
        if let Some(group) = self.group() {
            parts.push(group.to_string());
        }
        if let Some(node) = self.node() {
            parts.push(node.to_string());
        }
        if let Some(node_kind) = self.node_kind() {
            parts.push(node_kind.to_string());
        }
        if let Some(target_q16) = self.target_q16() {
            parts.push(target_q16.to_string());
        }
        if let Some(routing) = self.routing() {
            parts.push(routing.to_string());
        }
        if let Some(param) = self.param() {
            parts.push(param.to_string());
        }
        parts.join(":")
    }
}

/// The value law one address encodes its Q16 lane against.
///
/// The law is captured once, when the address is first interned, and is part
/// of the hashed document: the lattice a take was quantized on is the lattice
/// it replays on. Codes are permanent and append-only.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceValueLaw {
    /// A continuous value on the Q16 lattice over a declared finite range.
    Unit { min: f32, max: f32 },
    /// A closed token vocabulary; the value lane carries the token's index.
    Discrete { vocab: Vec<String> },
    /// A boolean; the value lane carries 0 or 1.
    Toggle,
    /// An integer choice; the value lane carries `value - min`. Kept separate
    /// from `Unit` because integer-valued appliers reject a non-integer wire
    /// number, so the round trip must land on exact integers.
    Stepped { min: i64, max: i64 },
}

impl PerformanceValueLaw {
    /// Permanent append-only canonical-stream code. Never renumber an existing
    /// entry.
    pub const fn code(&self) -> u8 {
        match self {
            Self::Unit { .. } => 0,
            Self::Discrete { .. } => 1,
            Self::Toggle => 2,
            Self::Stepped { .. } => 3,
        }
    }

    /// Wire token for the document encoding.
    pub const fn law_token(&self) -> &'static str {
        match self {
            Self::Unit { .. } => "unit",
            Self::Discrete { .. } => "discrete",
            Self::Toggle => "toggle",
            Self::Stepped { .. } => "stepped",
        }
    }

    /// Structural well-formedness of a declared law. A hostile document is
    /// refused here, never repaired.
    fn validate(&self) -> Result<(), PerformanceError> {
        match self {
            Self::Unit { min, max } => {
                if !min.is_finite() || !max.is_finite() || min >= max {
                    return Err(PerformanceError::InvalidLaw);
                }
            }
            Self::Discrete { vocab } => {
                if vocab.is_empty() || vocab.len() > MAX_PERFORMANCE_VOCAB_TOKENS {
                    return Err(PerformanceError::VocabTokens(vocab.len()));
                }
                for token in vocab {
                    if token.is_empty() || token.len() > MAX_PERFORMANCE_PARAM_BYTES {
                        return Err(PerformanceError::TokenBytes(token.len()));
                    }
                }
                for (index, token) in vocab.iter().enumerate() {
                    if vocab[..index].iter().any(|other| other == token) {
                        return Err(PerformanceError::DuplicateToken(token.clone()));
                    }
                }
            }
            Self::Toggle => {}
            Self::Stepped { min, max } => {
                if min >= max || max.saturating_sub(*min) > i64::from(u16::MAX) {
                    return Err(PerformanceError::InvalidLaw);
                }
            }
        }
        Ok(())
    }

    /// Quantize one authored raw value onto this law's lattice. A raw value
    /// the law cannot represent — a non-finite number, an unknown token, a
    /// kind mismatch — is refused rather than guessed at: the recorder skips
    /// and counts it, because inventing a neutral would put an edit in the
    /// take the operator never made.
    pub fn encode(&self, raw: &PerformanceRawValue) -> Option<u16> {
        match (self, raw) {
            (Self::Unit { min, max }, PerformanceRawValue::Continuous(value)) => {
                if !value.is_finite() {
                    return None;
                }
                let span = max - min;
                let normalized = ((value.clamp(*min, *max) - min) / span).clamp(0.0, 1.0);
                Some((normalized * PERFORMANCE_Q16_SCALE).round() as u16)
            }
            (Self::Discrete { vocab }, PerformanceRawValue::Token(token)) => vocab
                .iter()
                .position(|candidate| candidate == token)
                .and_then(|index| u16::try_from(index).ok()),
            (Self::Toggle, PerformanceRawValue::Toggle(value)) => Some(u16::from(*value)),
            (Self::Stepped { min, max }, PerformanceRawValue::Integer(value)) => {
                let clamped = (*value).clamp(*min, *max);
                u16::try_from(clamped - min).ok()
            }
            _ => None,
        }
    }

    /// Recover the authored value from its code. `None` only for a code the
    /// law's validated lattice cannot hold, which `validate_value` refuses at
    /// the document boundary before any replay begins.
    pub fn decode(&self, code: u16) -> Option<PerformanceRawValue> {
        match self {
            Self::Unit { min, max } => {
                let normalized = f32::from(code) / PERFORMANCE_Q16_SCALE;
                Some(PerformanceRawValue::Continuous(
                    min + normalized * (max - min),
                ))
            }
            Self::Discrete { vocab } => vocab
                .get(usize::from(code))
                .map(|token| PerformanceRawValue::Token(token.clone())),
            Self::Toggle => match code {
                0 => Some(PerformanceRawValue::Toggle(false)),
                1 => Some(PerformanceRawValue::Toggle(true)),
                _ => None,
            },
            Self::Stepped { min, max } => {
                let value = min.saturating_add(i64::from(code));
                (value <= *max).then_some(PerformanceRawValue::Integer(value))
            }
        }
    }

    /// Whether a stored value code is representable under this law.
    fn validate_value(&self, code: u16) -> Result<(), PerformanceError> {
        let valid = match self {
            Self::Unit { .. } => true,
            Self::Discrete { vocab } => usize::from(code) < vocab.len(),
            Self::Toggle => code <= 1,
            Self::Stepped { min, max } => i64::from(code) <= max.saturating_sub(*min),
        };
        if valid {
            Ok(())
        } else {
            Err(PerformanceError::InvalidValue(code))
        }
    }

    fn encode_canonical(&self, bytes: &mut Vec<u8>) {
        bytes.push(self.code());
        match self {
            Self::Unit { min, max } => {
                bytes.extend_from_slice(&min.to_bits().to_le_bytes());
                bytes.extend_from_slice(&max.to_bits().to_le_bytes());
            }
            Self::Discrete { vocab } => {
                bytes.extend_from_slice(&(vocab.len() as u16).to_le_bytes());
                for token in vocab {
                    bytes.extend_from_slice(&(token.len() as u16).to_le_bytes());
                    bytes.extend_from_slice(token.as_bytes());
                }
            }
            Self::Toggle => {}
            Self::Stepped { min, max } => {
                bytes.extend_from_slice(&min.to_le_bytes());
                bytes.extend_from_slice(&max.to_le_bytes());
            }
        }
    }
}

/// One authored raw value as the record tap observed it, before quantization.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceRawValue {
    Continuous(f32),
    Token(String),
    Toggle(bool),
    Integer(i64),
}

impl PerformanceRawValue {
    /// The wire value this raw value dispatches as, shared by live replay and
    /// export replay so both build identical payloads. `None` only for a
    /// non-finite continuous value, which decoding a validated lattice cannot
    /// produce.
    pub fn to_json(&self) -> Option<serde_json::Value> {
        Some(match self {
            Self::Continuous(value) => {
                serde_json::Value::Number(serde_json::Number::from_f64(f64::from(*value))?)
            }
            Self::Token(token) => serde_json::Value::String(token.clone()),
            Self::Toggle(value) => serde_json::Value::Bool(*value),
            Self::Integer(value) => serde_json::Value::Number((*value).into()),
        })
    }
}

/// One interned control address: the control identity plus the value law its
/// events are encoded against.
///
/// The document encoding is a deliberately flat, hand-validated struct — serde
/// ignores `deny_unknown_fields` on internally tagged enums, so a derived
/// encoding could not reject a hostile extra field. `RawAddress` carries every
/// possible field as an `Option` and `from_parts`/`validate` refuse a shape
/// that mixes fields across kinds or laws.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceAddress {
    pub control: PerformanceControl,
    pub law: PerformanceValueLaw,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAddress {
    kind: String,
    #[serde(default)]
    layer: Option<u32>,
    #[serde(default)]
    param: Option<String>,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    group_id: Option<String>,
    #[serde(default)]
    node_kind: Option<String>,
    #[serde(default)]
    target_q16: Option<u16>,
    #[serde(default)]
    routing: Option<u32>,
    law: String,
    #[serde(default)]
    min: Option<f32>,
    #[serde(default)]
    max: Option<f32>,
    #[serde(default)]
    vocab: Option<Vec<String>>,
    #[serde(default)]
    step_min: Option<i64>,
    #[serde(default)]
    step_max: Option<i64>,
}

impl RawAddress {
    fn into_address(self) -> Result<PerformanceAddress, PerformanceError> {
        let control = PerformanceControl::from_parts(
            &self.kind,
            self.layer,
            self.param,
            self.node_id,
            self.group_id,
            self.node_kind,
            self.target_q16,
            self.routing,
        )?;
        let law = match self.law.as_str() {
            "unit" => match (
                self.min,
                self.max,
                &self.vocab,
                self.step_min,
                self.step_max,
            ) {
                (Some(min), Some(max), None, None, None) => PerformanceValueLaw::Unit { min, max },
                _ => return Err(PerformanceError::MalformedAddress(self.law)),
            },
            "discrete" => match (self.min, self.max, self.vocab, self.step_min, self.step_max) {
                (None, None, Some(vocab), None, None) => {
                    if vocab.len() > MAX_PERFORMANCE_VOCAB_TOKENS {
                        return Err(PerformanceError::VocabTokens(vocab.len()));
                    }
                    PerformanceValueLaw::Discrete { vocab }
                }
                _ => return Err(PerformanceError::MalformedAddress(self.law)),
            },
            "toggle" => match (
                self.min,
                self.max,
                &self.vocab,
                self.step_min,
                self.step_max,
            ) {
                (None, None, None, None, None) => PerformanceValueLaw::Toggle,
                _ => return Err(PerformanceError::MalformedAddress(self.law)),
            },
            "stepped" => match (
                self.min,
                self.max,
                &self.vocab,
                self.step_min,
                self.step_max,
            ) {
                (None, None, None, Some(min), Some(max)) => {
                    PerformanceValueLaw::Stepped { min, max }
                }
                _ => return Err(PerformanceError::MalformedAddress(self.law)),
            },
            other => return Err(PerformanceError::MalformedAddress(other.to_string())),
        };
        let address = PerformanceAddress { control, law };
        address.validate()?;
        Ok(address)
    }
}

impl Serialize for PerformanceAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = 2usize;
        fields += usize::from(self.control.layer().is_some());
        fields += usize::from(self.control.param().is_some());
        fields += usize::from(self.control.node().is_some());
        fields += usize::from(self.control.group().is_some());
        fields += usize::from(self.control.node_kind().is_some());
        fields += usize::from(self.control.target_q16().is_some());
        fields += usize::from(self.control.routing().is_some());
        fields += match &self.law {
            PerformanceValueLaw::Unit { .. } => 2,
            PerformanceValueLaw::Discrete { .. } => 1,
            PerformanceValueLaw::Toggle => 0,
            PerformanceValueLaw::Stepped { .. } => 2,
        };
        let mut state = serializer.serialize_struct("PerformanceAddress", fields)?;
        state.serialize_field("kind", self.control.kind_token())?;
        if let Some(layer) = self.control.layer() {
            state.serialize_field("layer", &layer)?;
        }
        if let Some(node) = self.control.node() {
            state.serialize_field("node_id", &node.to_string())?;
        }
        if let Some(group) = self.control.group() {
            state.serialize_field("group_id", &group.to_string())?;
        }
        if let Some(node_kind) = self.control.node_kind() {
            state.serialize_field("node_kind", node_kind)?;
        }
        if let Some(target_q16) = self.control.target_q16() {
            state.serialize_field("target_q16", &target_q16)?;
        }
        if let Some(routing) = self.control.routing() {
            state.serialize_field("routing", &routing)?;
        }
        if let Some(param) = self.control.param() {
            state.serialize_field("param", param)?;
        }
        state.serialize_field("law", self.law.law_token())?;
        match &self.law {
            PerformanceValueLaw::Unit { min, max } => {
                state.serialize_field("min", min)?;
                state.serialize_field("max", max)?;
            }
            PerformanceValueLaw::Discrete { vocab } => {
                state.serialize_field("vocab", vocab)?;
            }
            PerformanceValueLaw::Toggle => {}
            PerformanceValueLaw::Stepped { min, max } => {
                state.serialize_field("step_min", min)?;
                state.serialize_field("step_max", max)?;
            }
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for PerformanceAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        RawAddress::deserialize(deserializer)?
            .into_address()
            .map_err(de::Error::custom)
    }
}

impl PerformanceAddress {
    fn validate(&self) -> Result<(), PerformanceError> {
        self.control.validate()?;
        self.law.validate()
    }

    fn encode_canonical(&self, bytes: &mut Vec<u8>) {
        bytes.push(self.control.code());
        bytes.extend_from_slice(&self.control.layer().unwrap_or(0).to_le_bytes());
        let param = self.control.param().unwrap_or("");
        bytes.extend_from_slice(&(param.len() as u16).to_le_bytes());
        bytes.extend_from_slice(param.as_bytes());
        self.law.encode_canonical(bytes);
        match &self.control {
            PerformanceControl::RackNodeMaster {
                node, node_kind, ..
            }
            | PerformanceControl::RackNodeLayer {
                node, node_kind, ..
            } => {
                bytes.extend_from_slice(&node.to_le_bytes());
                bytes.extend_from_slice(&(node_kind.len() as u16).to_le_bytes());
                bytes.extend_from_slice(node_kind.as_bytes());
            }
            PerformanceControl::RackNodeGroup {
                group,
                node,
                node_kind,
                ..
            } => {
                bytes.extend_from_slice(&group.to_le_bytes());
                bytes.extend_from_slice(&node.to_le_bytes());
                bytes.extend_from_slice(&(node_kind.len() as u16).to_le_bytes());
                bytes.extend_from_slice(node_kind.as_bytes());
            }
            PerformanceControl::GroupParam { group, .. }
            | PerformanceControl::GroupMatteParam { group, .. } => {
                bytes.extend_from_slice(&group.to_le_bytes());
            }
            PerformanceControl::MorphGlide { target_q16 } => {
                bytes.extend_from_slice(&target_q16.to_le_bytes());
            }
            PerformanceControl::Routing { routing, .. } => {
                bytes.extend_from_slice(&routing.to_le_bytes());
            }
            PerformanceControl::Master { .. }
            | PerformanceControl::MasterTransform { .. }
            | PerformanceControl::LayerParam { .. }
            | PerformanceControl::LayerEffect { .. }
            | PerformanceControl::LayerTransform { .. }
            | PerformanceControl::LayerVisible { .. }
            | PerformanceControl::LayerPattern { .. }
            | PerformanceControl::Ntsc { .. }
            | PerformanceControl::Temporal { .. }
            | PerformanceControl::MotionMaster { .. }
            | PerformanceControl::MotionLayer { .. }
            | PerformanceControl::BusCrossfade
            | PerformanceControl::BusMix { .. }
            | PerformanceControl::MorphPosition
            | PerformanceControl::GestureCanvas { .. }
            | PerformanceControl::LayerText { .. }
            | PerformanceControl::MorphLaw
            | PerformanceControl::MorphCapture => {}
        }
    }
}

/// One portable, fixed-width performance event. The quantized code is the only
/// representation — there is no retained float path, so live recording and
/// offline replay observe identical bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceEvent {
    /// Reference tick relative to the take origin (arming is tick zero).
    pub tick: u32,
    /// Index into the take's interned address table.
    pub address: u16,
    /// The value code, on the address law's lattice.
    pub value: u16,
}

impl PerformanceEvent {
    fn encode_into(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.tick.to_le_bytes());
        bytes.extend_from_slice(&self.address.to_le_bytes());
        bytes.extend_from_slice(&self.value.to_le_bytes());
    }
}

/// Typed refusals. Nothing here repairs; a hostile or ill-formed input is
/// named and refused.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceError {
    TooManyAddresses(usize),
    TooManyEvents(usize),
    NonMonotonicTick { previous: u32, tick: u32 },
    AddressOutOfRange { address: u16, count: usize },
    InvalidValue(u16),
    InvalidLaw,
    UnrepresentableEdit,
    MalformedAddress(String),
    DuplicateAddress(String),
    ParamBytes(usize),
    NodeKindBytes(usize),
    VocabTokens(usize),
    TokenBytes(usize),
    DuplicateToken(String),
    EventCount { declared: u32, observed: usize },
    LengthBeforeLastEvent { length: u32, last: u32 },
    UnsupportedVersion(u16),
    ChecksumMismatch { declared: String, computed: String },
    DocumentBytes(usize),
    Serialization(String),
}

impl fmt::Display for PerformanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyAddresses(count) => {
                write!(
                    formatter,
                    "take addresses {count} distinct controls; the cap is {MAX_PERFORMANCE_ADDRESSES}"
                )
            }
            Self::TooManyEvents(count) => {
                write!(
                    formatter,
                    "take declares {count} events; the cap is {MAX_PERFORMANCE_EVENTS}"
                )
            }
            Self::NonMonotonicTick { previous, tick } => {
                write!(
                    formatter,
                    "event tick {tick} precedes the previous tick {previous}"
                )
            }
            Self::AddressOutOfRange { address, count } => {
                write!(
                    formatter,
                    "event names address {address} but the table holds {count}"
                )
            }
            Self::InvalidValue(code) => {
                write!(formatter, "value code {code} is outside its address law")
            }
            Self::InvalidLaw => write!(formatter, "address law is malformed"),
            Self::UnrepresentableEdit => {
                write!(formatter, "the edit's value cannot ride its address law")
            }
            Self::MalformedAddress(kind) => {
                write!(formatter, "address shape for '{kind}' is malformed")
            }
            Self::DuplicateAddress(name) => {
                write!(formatter, "control '{name}' appears twice in the table")
            }
            Self::ParamBytes(bytes) => {
                write!(
                    formatter,
                    "param name is {bytes} bytes; the bound is 1..={MAX_PERFORMANCE_PARAM_BYTES}"
                )
            }
            Self::NodeKindBytes(bytes) => {
                write!(
                    formatter,
                    "node kind is {bytes} bytes; the bound is 1..={MAX_PERFORMANCE_PARAM_BYTES}"
                )
            }
            Self::VocabTokens(count) => {
                write!(
                    formatter,
                    "discrete vocabulary holds {count} tokens; the bound is 1..={MAX_PERFORMANCE_VOCAB_TOKENS}"
                )
            }
            Self::TokenBytes(bytes) => {
                write!(
                    formatter,
                    "vocabulary token is {bytes} bytes; the bound is 1..={MAX_PERFORMANCE_PARAM_BYTES}"
                )
            }
            Self::DuplicateToken(token) => {
                write!(formatter, "vocabulary token '{token}' appears twice")
            }
            Self::EventCount { declared, observed } => {
                write!(
                    formatter,
                    "document declares {declared} events but carries {observed}"
                )
            }
            Self::LengthBeforeLastEvent { length, last } => {
                write!(
                    formatter,
                    "declared length {length} precedes the last event tick {last}"
                )
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported performance take version {version}")
            }
            Self::ChecksumMismatch { declared, computed } => {
                write!(
                    formatter,
                    "checksum mismatch: declared {declared}, computed {computed}"
                )
            }
            Self::DocumentBytes(bytes) => {
                write!(
                    formatter,
                    "document is {bytes} bytes; the bound is 1..={MAX_PERFORMANCE_SERIALIZED_BYTES}"
                )
            }
            Self::Serialization(error) => write!(formatter, "serialization failed: {error}"),
        }
    }
}

/// Portable, bounded performance take. It contains no wall time, GPU state,
/// source path, or runtime identity — layers are addressed by saved stack
/// position, controls by their closed typed identity.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerformanceTake {
    addresses: Vec<PerformanceAddress>,
    events: Vec<PerformanceEvent>,
    /// Declared take length in reference ticks, stamped at disarm so a loop
    /// period is honest even when the take ends in silence.
    length_ticks: u32,
    truncated: bool,
    /// Recording was never disarmed: the take was captured while still armed.
    incomplete: bool,
}

impl PerformanceTake {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn addresses(&self) -> &[PerformanceAddress] {
        &self.addresses
    }

    pub fn events(&self) -> &[PerformanceEvent] {
        &self.events
    }

    pub const fn length_ticks(&self) -> u32 {
        self.length_ticks
    }

    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    pub const fn incomplete(&self) -> bool {
        self.incomplete
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Record one accepted authored edit at a reference tick.
    ///
    /// The address is interned on first sight together with its law; a later
    /// edit to the same control quantizes against the *stored* law, so the
    /// take's lattice can never shift under its own events. `Ok(false)` means
    /// the event cap was reached and `truncated` is now set; `Err` means the
    /// edit could not be represented (unknown token, non-finite number, law
    /// mismatch) and the take is byte-identical to before.
    pub fn record_accepted(
        &mut self,
        tick: u32,
        control: PerformanceControl,
        law: PerformanceValueLaw,
        raw: &PerformanceRawValue,
    ) -> Result<bool, PerformanceError> {
        control.validate()?;
        if let Some(previous) = self.events.last() {
            if tick < previous.tick {
                return Err(PerformanceError::NonMonotonicTick {
                    previous: previous.tick,
                    tick,
                });
            }
        }
        // Resolve the law and quantize before mutating anything, so a refused
        // edit leaves both the table and the events byte-identical.
        let existing = self
            .addresses
            .iter()
            .position(|candidate| candidate.control == control);
        let code = match existing {
            Some(index) => self.addresses[index].law.encode(raw),
            None => {
                law.validate()?;
                if self.addresses.len() >= MAX_PERFORMANCE_ADDRESSES {
                    return Err(PerformanceError::TooManyAddresses(self.addresses.len() + 1));
                }
                law.encode(raw)
            }
        }
        .ok_or(PerformanceError::UnrepresentableEdit)?;
        if self.events.len() >= MAX_PERFORMANCE_EVENTS {
            self.truncated = true;
            return Ok(false);
        }
        let address = match existing {
            Some(index) => index,
            None => {
                self.addresses.push(PerformanceAddress { control, law });
                self.addresses.len() - 1
            }
        };
        self.events.push(PerformanceEvent {
            tick,
            address: address as u16,
            value: code,
        });
        self.length_ticks = self.length_ticks.max(tick);
        Ok(true)
    }

    /// Stamp the take's declared length at disarm. The length never precedes
    /// the last recorded event.
    pub fn finalize(&mut self, length_ticks: u32) {
        let last = self.events.last().map(|event| event.tick).unwrap_or(0);
        self.length_ticks = length_ticks.max(last);
        self.incomplete = false;
    }

    /// Mark the take as captured while recording was still armed.
    pub fn mark_incomplete(&mut self, length_ticks: u32) {
        let last = self.events.last().map(|event| event.tick).unwrap_or(0);
        self.length_ticks = length_ticks.max(last);
        self.incomplete = true;
    }

    /// Hashed flag word. Truncation and incompleteness are facts about the
    /// take, so they are inside the digest rather than beside it.
    pub fn flags(&self) -> u16 {
        let mut flags = 0;
        if self.truncated {
            flags |= PERFORMANCE_FLAG_TRUNCATED;
        }
        if self.incomplete {
            flags |= PERFORMANCE_FLAG_INCOMPLETE;
        }
        flags
    }

    pub fn replay(&self) -> PerformanceReplay<'_> {
        PerformanceReplay {
            events: &self.events,
            cursor: 0,
        }
    }

    /// The canonical, domain-separated, explicit little-endian field stream.
    /// The address table is inside the stream: an event is meaningless without
    /// the lattice it was quantized on, so the two can never be presented
    /// separately under one digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            PERFORMANCE_CHECKSUM_DOMAIN.len()
                + 16
                + self.addresses.len() * 32
                + self.events.len() * PERFORMANCE_EVENT_ENCODED_BYTES,
        );
        bytes.extend_from_slice(PERFORMANCE_CHECKSUM_DOMAIN);
        bytes.extend_from_slice(&PERFORMANCE_ALGORITHM_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.flags().to_le_bytes());
        bytes.extend_from_slice(&self.length_ticks.to_le_bytes());
        bytes.extend_from_slice(&(self.addresses.len() as u16).to_le_bytes());
        for address in &self.addresses {
            address.encode_canonical(&mut bytes);
        }
        bytes.extend_from_slice(&self.declared_event_count().to_le_bytes());
        for event in &self.events {
            event.encode_into(&mut bytes);
        }
        bytes
    }

    pub fn checksum(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }

    pub fn checksum_hex(&self) -> String {
        format!("{:x}", Sha256::digest(self.canonical_bytes()))
    }

    /// The event count is bounded by `MAX_PERFORMANCE_EVENTS`, so it always
    /// fits the hashed `u32` field.
    fn declared_event_count(&self) -> u32 {
        u32::try_from(self.events.len()).unwrap_or(u32::MAX)
    }

    /// Structural validation shared by the document decode path and the
    /// Rust-assembled-take path.
    fn validate(&self) -> Result<(), PerformanceError> {
        if self.addresses.len() > MAX_PERFORMANCE_ADDRESSES {
            return Err(PerformanceError::TooManyAddresses(self.addresses.len()));
        }
        if self.events.len() > MAX_PERFORMANCE_EVENTS {
            return Err(PerformanceError::TooManyEvents(self.events.len()));
        }
        for address in &self.addresses {
            address.validate()?;
        }
        for (index, address) in self.addresses.iter().enumerate() {
            if self.addresses[..index]
                .iter()
                .any(|other| other.control == address.control)
            {
                return Err(PerformanceError::DuplicateAddress(
                    address.control.describe(),
                ));
            }
        }
        let mut previous = 0u32;
        for event in &self.events {
            if event.tick < previous {
                return Err(PerformanceError::NonMonotonicTick {
                    previous,
                    tick: event.tick,
                });
            }
            previous = event.tick;
            let address = self.addresses.get(usize::from(event.address)).ok_or(
                PerformanceError::AddressOutOfRange {
                    address: event.address,
                    count: self.addresses.len(),
                },
            )?;
            address.law.validate_value(event.value)?;
        }
        if let Some(last) = self.events.last() {
            if self.length_ticks < last.tick {
                return Err(PerformanceError::LengthBeforeLastEvent {
                    length: self.length_ticks,
                    last: last.tick,
                });
            }
        }
        Ok(())
    }
}

/// Borrow-only monotonic drain cursor over a recorded take.
#[derive(Debug, Clone)]
pub struct PerformanceReplay<'a> {
    events: &'a [PerformanceEvent],
    cursor: usize,
}

impl<'a> PerformanceReplay<'a> {
    /// Consume every event whose reference tick is due. A low display rate may
    /// cross several ticks in one frame; every crossed event is returned in
    /// recorded order rather than collapsed, so replay applies exactly what
    /// recording stored. The cursor is monotonic and never rewinds.
    pub fn events_due(&mut self, reference_tick: u32) -> &'a [PerformanceEvent] {
        let start = self.cursor;
        while self
            .events
            .get(self.cursor)
            .is_some_and(|event| event.tick <= reference_tick)
        {
            self.cursor += 1;
        }
        &self.events[start..self.cursor]
    }

    pub const fn finished(&self) -> bool {
        self.cursor >= self.events.len()
    }
}

/// Owned monotonic cursor for a host replay session (the borrow-only cursor
/// cannot live across frames beside a mutable host).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerformanceCursor {
    cursor: usize,
}

impl PerformanceCursor {
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// The half-open index range of events due at `reference_tick`.
    pub fn range_due(
        &mut self,
        events: &[PerformanceEvent],
        reference_tick: u32,
    ) -> (usize, usize) {
        let start = self.cursor;
        while events
            .get(self.cursor)
            .is_some_and(|event| event.tick <= reference_tick)
        {
            self.cursor += 1;
        }
        (start, self.cursor)
    }

    pub fn finished(&self, events: &[PerformanceEvent]) -> bool {
        self.cursor >= events.len()
    }
}

/// Count-capped sequences: the allocation is bounded before the input is
/// trusted, so a declared-huge sequence is rejected by the cap rather than by
/// allocating first and checking afterwards.
#[derive(Debug, Clone, PartialEq)]
struct BoundedPerformanceEvents(Vec<PerformanceEvent>);

impl<'de> Deserialize<'de> for BoundedPerformanceEvents {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = BoundedPerformanceEvents;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {MAX_PERFORMANCE_EVENTS} performance events"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(
                    sequence
                        .size_hint()
                        .unwrap_or(0)
                        .min(MAX_PERFORMANCE_EVENTS),
                );
                while values.len() < MAX_PERFORMANCE_EVENTS {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedPerformanceEvents(values));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("too many performance events"));
                }
                Ok(BoundedPerformanceEvents(values))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BoundedPerformanceAddresses(Vec<PerformanceAddress>);

impl<'de> Deserialize<'de> for BoundedPerformanceAddresses {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = BoundedPerformanceAddresses;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {MAX_PERFORMANCE_ADDRESSES} performance addresses"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(
                    sequence
                        .size_hint()
                        .unwrap_or(0)
                        .min(MAX_PERFORMANCE_ADDRESSES),
                );
                while values.len() < MAX_PERFORMANCE_ADDRESSES {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedPerformanceAddresses(values));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("too many performance addresses"));
                }
                Ok(BoundedPerformanceAddresses(values))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

/// Portable document for a recorded performance take, carried whole in a patch
/// and beside an export. The field set is frozen: `version`, `length_ticks`,
/// `truncated`, `incomplete`, `event_count`, `checksum`, `addresses`,
/// `events`. Operational paths and filesystem metadata must never enter it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceTakeDocument {
    pub version: u16,
    pub length_ticks: u32,
    pub truncated: bool,
    pub incomplete: bool,
    pub event_count: u32,
    pub checksum: String,
    pub addresses: Vec<PerformanceAddress>,
    pub events: Vec<PerformanceEvent>,
}

impl PerformanceTakeDocument {
    pub fn capture(take: &PerformanceTake) -> Self {
        Self {
            version: PERFORMANCE_ALGORITHM_VERSION,
            length_ticks: take.length_ticks,
            truncated: take.truncated,
            incomplete: take.incomplete,
            event_count: take.declared_event_count(),
            checksum: take.checksum_hex(),
            addresses: take.addresses.clone(),
            events: take.events.clone(),
        }
    }

    /// The single acceptance path. It rebuilds the take through the same
    /// structural validation that governs live recording and then re-derives
    /// and compares the canonical checksum, so neither an ill-formed stream
    /// nor a mismatched digest can be accepted.
    pub fn decode(&self) -> Result<PerformanceTake, PerformanceError> {
        if self.version != PERFORMANCE_ALGORITHM_VERSION {
            return Err(PerformanceError::UnsupportedVersion(self.version));
        }
        if usize::try_from(self.event_count) != Ok(self.events.len()) {
            return Err(PerformanceError::EventCount {
                declared: self.event_count,
                observed: self.events.len(),
            });
        }
        let take = PerformanceTake {
            addresses: self.addresses.clone(),
            events: self.events.clone(),
            length_ticks: self.length_ticks,
            truncated: self.truncated,
            incomplete: self.incomplete,
        };
        take.validate()?;
        let computed = take.checksum_hex();
        if computed != self.checksum {
            return Err(PerformanceError::ChecksumMismatch {
                declared: self.checksum.clone(),
                computed,
            });
        }
        Ok(take)
    }

    pub fn validate(&self) -> Result<(), PerformanceError> {
        self.decode().map(|_| ())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, PerformanceError> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| PerformanceError::Serialization(error.to_string()))?;
        // Sidecars are line-oriented text artifacts. Keep one canonical LF so
        // a stored document can be compared and re-emitted byte-for-byte
        // without test-only trimming or platform-dependent CRLF rewriting.
        bytes.push(b'\n');
        validate_document_bytes(bytes.len())?;
        Ok(bytes)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, PerformanceError> {
        validate_document_bytes(bytes.len())?;
        serde_json::from_slice(bytes)
            .map_err(|error| PerformanceError::Serialization(error.to_string()))
    }
}

impl<'de> Deserialize<'de> for PerformanceTakeDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            version: u16,
            length_ticks: u32,
            truncated: bool,
            incomplete: bool,
            event_count: u32,
            checksum: String,
            addresses: BoundedPerformanceAddresses,
            events: BoundedPerformanceEvents,
        }

        let raw = Raw::deserialize(deserializer)?;
        let document = Self {
            version: raw.version,
            length_ticks: raw.length_ticks,
            truncated: raw.truncated,
            incomplete: raw.incomplete,
            event_count: raw.event_count,
            checksum: raw.checksum,
            addresses: raw.addresses.0,
            events: raw.events.0,
        };
        document.validate().map_err(de::Error::custom)?;
        Ok(document)
    }
}

fn validate_document_bytes(bytes: usize) -> Result<(), PerformanceError> {
    if bytes == 0 || bytes > MAX_PERFORMANCE_SERIALIZED_BYTES {
        Err(PerformanceError::DocumentBytes(bytes))
    } else {
        Ok(())
    }
}

/// Accepted-frame reference clock, `GestureEventRecorder`'s exact tick law: it
/// accumulates accepted, program-advancing seconds and derives the tick
/// *before* this frame's delta is added, so the first accepted frame lands at
/// tick 0. Program Freeze contributes zero delta and a rejected frame is never
/// fed to it, so neither can accumulate catch-up debt for time the audience
/// never saw.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerformanceClock {
    accepted_seconds: f64,
}

impl PerformanceClock {
    pub fn reference_tick(&self) -> u64 {
        (self.accepted_seconds * f64::from(PERFORMANCE_REFERENCE_FPS))
            .round()
            .clamp(0.0, u64::MAX as f64) as u64
    }

    pub fn reference_tick_u32(&self) -> u32 {
        u32::try_from(self.reference_tick()).unwrap_or(u32::MAX)
    }

    /// Advance by one accepted program-advancing frame.
    pub fn advance_accepted(&mut self, delta_seconds: f32) {
        self.accepted_seconds += f64::from(crate::temporal::sanitize_delta(delta_seconds));
    }

    pub fn reset(&mut self) {
        self.accepted_seconds = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_law() -> PerformanceValueLaw {
        PerformanceValueLaw::Unit { min: 0.0, max: 1.0 }
    }

    fn master(param: &str) -> PerformanceControl {
        PerformanceControl::Master {
            param: param.to_string(),
        }
    }

    fn recorded_take() -> PerformanceTake {
        let mut take = PerformanceTake::default();
        take.record_accepted(
            0,
            master("hue_shift"),
            unit_law(),
            &PerformanceRawValue::Continuous(0.25),
        )
        .unwrap();
        take.record_accepted(
            3,
            PerformanceControl::Temporal {
                param: "feedback".to_string(),
            },
            PerformanceValueLaw::Unit {
                min: 0.0,
                max: 0.95,
            },
            &PerformanceRawValue::Continuous(0.5),
        )
        .unwrap();
        take.record_accepted(
            3,
            master("hue_shift"),
            unit_law(),
            &PerformanceRawValue::Continuous(0.75),
        )
        .unwrap();
        take.finalize(10);
        take
    }

    #[test]
    fn the_reference_fps_is_the_temporal_constant_not_a_second_literal() {
        assert_eq!(PERFORMANCE_REFERENCE_FPS, TEMPORAL_REFERENCE_FPS);
        let source = include_str!("performance_track.rs");
        let after_constants = source
            .split_once("PERFORMANCE_REFERENCE_FPS: f32 = TEMPORAL_REFERENCE_FPS")
            .expect("the constant reuse must exist")
            .1;
        // Built at runtime so this needle cannot match itself in source text.
        let second_literal = format!("{}{}", "30", ".0");
        assert!(
            !after_constants.contains(&second_literal),
            "the reference rate must be the shared constant, never a second literal"
        );
    }

    #[test]
    fn unit_law_quantizes_on_the_declared_range_with_exact_endpoints() {
        let law = PerformanceValueLaw::Unit {
            min: -1.0,
            max: 3.0,
        };
        assert_eq!(law.encode(&PerformanceRawValue::Continuous(-1.0)), Some(0));
        assert_eq!(
            law.encode(&PerformanceRawValue::Continuous(3.0)),
            Some(65_535)
        );
        assert_eq!(
            law.encode(&PerformanceRawValue::Continuous(9.0)),
            Some(65_535),
            "out-of-range input clamps to the declared lattice"
        );
        let PerformanceRawValue::Continuous(low) = law.decode(0).unwrap() else {
            panic!("unit law decodes to a continuous value");
        };
        let PerformanceRawValue::Continuous(high) = law.decode(65_535).unwrap() else {
            panic!("unit law decodes to a continuous value");
        };
        assert_eq!(low, -1.0);
        assert_eq!(high, 3.0);
    }

    #[test]
    fn non_finite_and_mismatched_raw_values_are_refused_not_guessed() {
        let law = unit_law();
        assert_eq!(law.encode(&PerformanceRawValue::Continuous(f32::NAN)), None);
        assert_eq!(
            law.encode(&PerformanceRawValue::Token("hue".to_string())),
            None
        );
        let discrete = PerformanceValueLaw::Discrete {
            vocab: vec!["ramp".to_string(), "sweep".to_string()],
        };
        assert_eq!(
            discrete.encode(&PerformanceRawValue::Token("unknown".to_string())),
            None,
            "an unknown token is refused rather than defaulted"
        );
    }

    #[test]
    fn discrete_toggle_and_stepped_round_trip_exactly() {
        let discrete = PerformanceValueLaw::Discrete {
            vocab: vec!["ramp".to_string(), "sweep".to_string()],
        };
        let code = discrete
            .encode(&PerformanceRawValue::Token("sweep".to_string()))
            .unwrap();
        assert_eq!(
            discrete.decode(code),
            Some(PerformanceRawValue::Token("sweep".to_string()))
        );
        let toggle = PerformanceValueLaw::Toggle;
        assert_eq!(
            toggle
                .encode(&PerformanceRawValue::Toggle(true))
                .and_then(|code| toggle.decode(code)),
            Some(PerformanceRawValue::Toggle(true))
        );
        let stepped = PerformanceValueLaw::Stepped { min: -2, max: 7 };
        let code = stepped.encode(&PerformanceRawValue::Integer(5)).unwrap();
        assert_eq!(
            stepped.decode(code),
            Some(PerformanceRawValue::Integer(5)),
            "stepped values land on exact integers, never a nearby float"
        );
        assert_eq!(
            stepped.encode(&PerformanceRawValue::Integer(99)),
            stepped.encode(&PerformanceRawValue::Integer(7)),
            "out-of-range integers clamp to the declared bound"
        );
    }

    #[test]
    fn hostile_laws_are_refused_by_name() {
        assert_eq!(
            PerformanceValueLaw::Unit { min: 1.0, max: 1.0 }.validate(),
            Err(PerformanceError::InvalidLaw)
        );
        assert_eq!(
            PerformanceValueLaw::Unit {
                min: f32::NAN,
                max: 1.0
            }
            .validate(),
            Err(PerformanceError::InvalidLaw)
        );
        assert_eq!(
            PerformanceValueLaw::Discrete { vocab: Vec::new() }.validate(),
            Err(PerformanceError::VocabTokens(0))
        );
        assert_eq!(
            PerformanceValueLaw::Discrete {
                vocab: vec!["a".to_string(), "a".to_string()]
            }
            .validate(),
            Err(PerformanceError::DuplicateToken("a".to_string()))
        );
        assert_eq!(
            PerformanceValueLaw::Stepped {
                min: 0,
                max: 100_000
            }
            .validate(),
            Err(PerformanceError::InvalidLaw),
            "a stepped span must fit the u16 value lane"
        );
    }

    #[test]
    fn recording_interns_each_control_once_and_keeps_the_first_law() {
        let mut take = PerformanceTake::default();
        take.record_accepted(
            0,
            master("hue_shift"),
            unit_law(),
            &PerformanceRawValue::Continuous(0.5),
        )
        .unwrap();
        // A conflicting later law for the same control must not shift the
        // lattice the first event was quantized on.
        take.record_accepted(
            1,
            master("hue_shift"),
            PerformanceValueLaw::Unit {
                min: 0.0,
                max: 360.0,
            },
            &PerformanceRawValue::Continuous(0.5),
        )
        .unwrap();
        assert_eq!(take.addresses().len(), 1);
        assert_eq!(take.addresses()[0].law, unit_law());
        assert_eq!(take.events().len(), 2);
        assert_eq!(take.events()[0].value, take.events()[1].value);
    }

    #[test]
    fn a_refused_edit_leaves_the_address_table_untouched() {
        let mut take = PerformanceTake::default();
        let before = take.clone();
        let error = take
            .record_accepted(
                0,
                master("hue_shift"),
                unit_law(),
                &PerformanceRawValue::Continuous(f32::NAN),
            )
            .unwrap_err();
        assert_eq!(error, PerformanceError::UnrepresentableEdit);
        assert_eq!(take, before, "a refused edit interns nothing");
    }

    #[test]
    fn the_event_cap_sets_truncated_and_refuses_further_recording() {
        let mut take = PerformanceTake::default();
        for tick in 0..MAX_PERFORMANCE_EVENTS {
            assert!(take
                .record_accepted(
                    tick as u32,
                    master("hue_shift"),
                    unit_law(),
                    &PerformanceRawValue::Continuous(0.5),
                )
                .unwrap());
        }
        assert!(!take.truncated());
        assert!(!take
            .record_accepted(
                MAX_PERFORMANCE_EVENTS as u32,
                master("hue_shift"),
                unit_law(),
                &PerformanceRawValue::Continuous(0.5),
            )
            .unwrap());
        assert!(take.truncated());
        assert_eq!(take.events().len(), MAX_PERFORMANCE_EVENTS);
    }

    #[test]
    fn non_monotonic_ticks_are_rejected_leaving_the_take_unchanged() {
        let mut take = recorded_take();
        let before = take.clone();
        let error = take
            .record_accepted(
                1,
                master("brightness"),
                unit_law(),
                &PerformanceRawValue::Continuous(0.5),
            )
            .unwrap_err();
        assert_eq!(
            error,
            PerformanceError::NonMonotonicTick {
                previous: 3,
                tick: 1
            }
        );
        assert_eq!(take, before);
    }

    #[test]
    fn the_flags_are_inside_the_digest() {
        let complete = recorded_take();
        let mut incomplete = complete.clone();
        incomplete.mark_incomplete(10);
        assert_ne!(
            complete.checksum_hex(),
            incomplete.checksum_hex(),
            "an incomplete take can never present a complete take's digest"
        );
        let mut truncated = complete.clone();
        truncated.truncated = true;
        assert_ne!(complete.checksum_hex(), truncated.checksum_hex());
    }

    #[test]
    fn the_checksum_covers_the_address_table() {
        let take = recorded_take();
        let mut retabled = take.clone();
        retabled.addresses[0].law = PerformanceValueLaw::Unit { min: 0.0, max: 2.0 };
        assert_ne!(
            take.checksum_hex(),
            retabled.checksum_hex(),
            "events are meaningless without their lattice, so the table is hashed"
        );
    }

    #[test]
    fn the_document_round_trips_and_the_decode_path_is_the_single_acceptance() {
        let take = recorded_take();
        let document = PerformanceTakeDocument::capture(&take);
        let bytes = document.to_json_bytes().unwrap();
        let restored = PerformanceTakeDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored, document);
        let decoded = restored.decode().unwrap();
        assert_eq!(decoded, take);
        assert_eq!(decoded.checksum_hex(), take.checksum_hex());
    }

    #[test]
    fn every_control_kind_round_trips_through_the_document() {
        let controls = [
            master("hue_shift"),
            PerformanceControl::MasterTransform {
                param: "position_x".to_string(),
            },
            PerformanceControl::LayerParam {
                layer: 2,
                param: "opacity".to_string(),
            },
            PerformanceControl::LayerEffect {
                layer: 1,
                param: "pixelate".to_string(),
            },
            PerformanceControl::LayerTransform {
                layer: 3,
                param: "scale_x".to_string(),
            },
            PerformanceControl::LayerVisible { layer: 1 },
            PerformanceControl::LayerPattern {
                layer: 4,
                param: "freq_x".to_string(),
            },
            PerformanceControl::Ntsc {
                param: "snow".to_string(),
            },
            PerformanceControl::Temporal {
                param: "feedback".to_string(),
            },
            PerformanceControl::MotionMaster {
                param: "amount".to_string(),
            },
            PerformanceControl::MotionLayer {
                layer: 2,
                param: "amount".to_string(),
            },
            PerformanceControl::BusCrossfade,
            PerformanceControl::BusMix {
                param: "mix_softness".to_string(),
            },
            PerformanceControl::MorphPosition,
            PerformanceControl::GestureCanvas {
                param: "radius".to_string(),
            },
            PerformanceControl::RackNodeMaster {
                node: 11,
                node_kind: "color_correct".to_string(),
                param: "brightness".to_string(),
            },
            PerformanceControl::RackNodeLayer {
                layer: 2,
                node: 12,
                node_kind: "pixelate".to_string(),
                param: "amount".to_string(),
            },
            PerformanceControl::RackNodeGroup {
                group: 21,
                node: 13,
                node_kind: "feedback".to_string(),
                param: "amount".to_string(),
            },
            PerformanceControl::GroupParam {
                group: 21,
                param: "opacity".to_string(),
            },
            PerformanceControl::GroupMatteParam {
                group: 21,
                param: "threshold".to_string(),
            },
            PerformanceControl::LayerText {
                layer: 3,
                param: "size".to_string(),
            },
            PerformanceControl::MorphLaw,
            PerformanceControl::MorphGlide { target_q16: 49_151 },
            PerformanceControl::MorphCapture,
            PerformanceControl::Routing {
                routing: 4,
                param: "depth".to_string(),
            },
        ];
        let mut take = PerformanceTake::default();
        for (tick, control) in controls.iter().enumerate() {
            let (law, raw) = match control {
                PerformanceControl::LayerVisible { .. } => (
                    PerformanceValueLaw::Toggle,
                    PerformanceRawValue::Toggle(true),
                ),
                PerformanceControl::MorphLaw => (
                    PerformanceValueLaw::Discrete {
                        vocab: vec!["linear".to_string(), "equal_power".to_string()],
                    },
                    PerformanceRawValue::Token("equal_power".to_string()),
                ),
                PerformanceControl::MorphCapture => (
                    PerformanceValueLaw::Discrete {
                        vocab: vec!["a".to_string(), "b".to_string()],
                    },
                    PerformanceRawValue::Token("b".to_string()),
                ),
                _ => (unit_law(), PerformanceRawValue::Continuous(0.5)),
            };
            take.record_accepted(tick as u32, control.clone(), law, &raw)
                .unwrap();
        }
        take.finalize(controls.len() as u32);
        let document = PerformanceTakeDocument::capture(&take);
        let bytes = document.to_json_bytes().unwrap();
        let restored = PerformanceTakeDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored.decode().unwrap(), take);
    }

    #[test]
    fn append_only_control_codes_and_tokens_are_frozen() {
        let controls = [
            (0, "master", master("brightness")),
            (
                1,
                "master_transform",
                PerformanceControl::MasterTransform {
                    param: "position_x".to_string(),
                },
            ),
            (
                2,
                "layer_param",
                PerformanceControl::LayerParam {
                    layer: 1,
                    param: "opacity".to_string(),
                },
            ),
            (
                3,
                "layer_effect",
                PerformanceControl::LayerEffect {
                    layer: 1,
                    param: "pixelate".to_string(),
                },
            ),
            (
                4,
                "layer_transform",
                PerformanceControl::LayerTransform {
                    layer: 1,
                    param: "scale_x".to_string(),
                },
            ),
            (
                5,
                "layer_visible",
                PerformanceControl::LayerVisible { layer: 1 },
            ),
            (
                6,
                "layer_pattern",
                PerformanceControl::LayerPattern {
                    layer: 1,
                    param: "freq_x".to_string(),
                },
            ),
            (
                7,
                "ntsc",
                PerformanceControl::Ntsc {
                    param: "snow".to_string(),
                },
            ),
            (
                8,
                "temporal",
                PerformanceControl::Temporal {
                    param: "feedback".to_string(),
                },
            ),
            (
                9,
                "motion_master",
                PerformanceControl::MotionMaster {
                    param: "amount".to_string(),
                },
            ),
            (
                10,
                "motion_layer",
                PerformanceControl::MotionLayer {
                    layer: 1,
                    param: "amount".to_string(),
                },
            ),
            (11, "bus_crossfade", PerformanceControl::BusCrossfade),
            (
                12,
                "bus_mix",
                PerformanceControl::BusMix {
                    param: "mix_softness".to_string(),
                },
            ),
            (13, "morph_position", PerformanceControl::MorphPosition),
            (
                14,
                "gesture_canvas",
                PerformanceControl::GestureCanvas {
                    param: "radius".to_string(),
                },
            ),
            (
                15,
                "rack_node_master",
                PerformanceControl::RackNodeMaster {
                    node: 1,
                    node_kind: "color_correct".to_string(),
                    param: "brightness".to_string(),
                },
            ),
            (
                16,
                "rack_node_layer",
                PerformanceControl::RackNodeLayer {
                    layer: 1,
                    node: 2,
                    node_kind: "pixelate".to_string(),
                    param: "amount".to_string(),
                },
            ),
            (
                17,
                "rack_node_group",
                PerformanceControl::RackNodeGroup {
                    group: 3,
                    node: 4,
                    node_kind: "feedback".to_string(),
                    param: "amount".to_string(),
                },
            ),
            (
                18,
                "group_param",
                PerformanceControl::GroupParam {
                    group: 3,
                    param: "opacity".to_string(),
                },
            ),
            (
                19,
                "group_matte_param",
                PerformanceControl::GroupMatteParam {
                    group: 3,
                    param: "threshold".to_string(),
                },
            ),
            (
                20,
                "layer_text",
                PerformanceControl::LayerText {
                    layer: 1,
                    param: "size".to_string(),
                },
            ),
            (21, "morph_law", PerformanceControl::MorphLaw),
            (
                22,
                "morph_glide",
                PerformanceControl::MorphGlide { target_q16: 32_768 },
            ),
            (23, "morph_capture", PerformanceControl::MorphCapture),
            (
                24,
                "routing",
                PerformanceControl::Routing {
                    routing: 2,
                    param: "depth".to_string(),
                },
            ),
        ];
        for (code, token, control) in controls {
            assert_eq!(control.code(), code, "{token} code moved");
            assert_eq!(control.kind_token(), token, "code {code} token moved");
        }
    }

    #[test]
    fn v2_identity_fields_round_trip_with_exact_wire_types() {
        let controls = [
            PerformanceControl::RackNodeGroup {
                group: u64::MAX,
                node: 9_007_199_254_740_993,
                node_kind: "feedback".to_string(),
                param: "amount".to_string(),
            },
            PerformanceControl::MorphGlide { target_q16: 51_234 },
            PerformanceControl::Routing {
                routing: 17,
                param: "depth".to_string(),
            },
        ];
        for control in controls {
            let address = PerformanceAddress {
                control: control.clone(),
                law: unit_law(),
            };
            let value = serde_json::to_value(&address).unwrap();
            match &control {
                PerformanceControl::RackNodeGroup { group, node, .. } => {
                    assert_eq!(value["group_id"], group.to_string());
                    assert_eq!(value["node_id"], node.to_string());
                    assert!(value["group_id"].is_string());
                    assert!(value["node_id"].is_string());
                }
                PerformanceControl::MorphGlide { target_q16 } => {
                    assert_eq!(value["target_q16"], u64::from(*target_q16));
                    assert!(value["target_q16"].is_number());
                }
                PerformanceControl::Routing { routing, .. } => {
                    assert_eq!(value["routing"], u64::from(*routing));
                    assert!(value["routing"].is_number());
                }
                _ => unreachable!(),
            }
            let restored: PerformanceAddress = serde_json::from_value(value).unwrap();
            assert_eq!(restored, address);
        }
    }

    #[test]
    fn v2_address_shapes_refuse_hostile_identity_fields() {
        let hostile = [
            serde_json::json!({
                "kind": "rack_node_master", "node_id": 7,
                "node_kind": "pixelate", "param": "amount",
                "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "rack_node_master", "node_id": "0",
                "node_kind": "pixelate", "param": "amount",
                "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "rack_node_master", "node_id": "01",
                "node_kind": "pixelate", "param": "amount",
                "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "rack_node_master", "node_id": "+1",
                "node_kind": "pixelate", "param": "amount",
                "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "rack_node_master", "node_id": "1",
                "group_id": "2", "node_kind": "pixelate", "param": "amount",
                "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "rack_node_group", "node_id": "1",
                "group_id": "2", "node_kind": "", "param": "amount",
                "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "group_param", "group_id": "18446744073709551616",
                "param": "opacity", "law": "unit", "min": 0.0, "max": 1.0
            }),
            serde_json::json!({
                "kind": "morph_glide", "target_q16": "65535",
                "law": "unit", "min": 0.25, "max": 64.0
            }),
            serde_json::json!({
                "kind": "morph_glide", "target_q16": 65535,
                "param": "duration", "law": "unit", "min": 0.25, "max": 64.0
            }),
            serde_json::json!({
                "kind": "routing", "routing": 0, "group_id": "2",
                "param": "depth", "law": "unit", "min": -1.0, "max": 1.0
            }),
        ];
        for value in hostile {
            assert!(
                serde_json::from_value::<PerformanceAddress>(value).is_err(),
                "a hostile identity shape reached the typed address table"
            );
        }

        assert!(matches!(
            (PerformanceAddress {
                control: PerformanceControl::RackNodeMaster {
                    node: 0,
                    node_kind: "pixelate".to_string(),
                    param: "amount".to_string(),
                },
                law: unit_law(),
            })
            .validate(),
            Err(PerformanceError::MalformedAddress(_))
        ));
        assert_eq!(
            (PerformanceAddress {
                control: PerformanceControl::RackNodeMaster {
                    node: 1,
                    node_kind: String::new(),
                    param: "amount".to_string(),
                },
                law: unit_law(),
            })
            .validate(),
            Err(PerformanceError::NodeKindBytes(0))
        );
    }

    #[test]
    fn new_canonical_identity_tails_are_binary_and_unambiguous() {
        let address = PerformanceAddress {
            control: PerformanceControl::RackNodeGroup {
                group: 0x0102_0304_0506_0708,
                node: 0x1112_1314_1516_1718,
                node_kind: "fx".to_string(),
                param: "amount".to_string(),
            },
            law: PerformanceValueLaw::Toggle,
        };
        let mut encoded = Vec::new();
        address.encode_canonical(&mut encoded);
        let mut expected_tail = Vec::new();
        expected_tail.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        expected_tail.extend_from_slice(&0x1112_1314_1516_1718u64.to_le_bytes());
        expected_tail.extend_from_slice(&2u16.to_le_bytes());
        expected_tail.extend_from_slice(b"fx");
        assert!(encoded.ends_with(&expected_tail));

        let glide = PerformanceAddress {
            control: PerformanceControl::MorphGlide { target_q16: 0xabcd },
            law: unit_law(),
        };
        let mut encoded = Vec::new();
        glide.encode_canonical(&mut encoded);
        assert!(encoded.ends_with(&0xabcdu16.to_le_bytes()));

        let routing = PerformanceAddress {
            control: PerformanceControl::Routing {
                routing: 0x0102_0304,
                param: "depth".to_string(),
            },
            law: unit_law(),
        };
        let mut encoded = Vec::new();
        routing.encode_canonical(&mut encoded);
        assert!(encoded.ends_with(&0x0102_0304u32.to_le_bytes()));
    }

    #[test]
    fn stored_v1_take_decodes_replays_and_reencodes_byte_exactly() {
        let literal = include_bytes!("../tests/fixtures/performance-take-v1-brightness.json");
        let document = PerformanceTakeDocument::from_json_bytes(literal).unwrap();
        let take = document.decode().unwrap();
        assert_eq!(
            take.checksum_hex(),
            "be4bb410f3984214fc13667f4135208d089d14aefbbff2fb2f6e19ff5a0758d6"
        );
        assert_eq!(
            take.addresses(),
            &[PerformanceAddress {
                control: master("brightness"),
                law: PerformanceValueLaw::Unit {
                    min: -1.0,
                    max: 1.0,
                },
            }]
        );
        assert_eq!(
            take.events(),
            &[PerformanceEvent {
                tick: 7,
                address: 0,
                value: 40_959,
            }]
        );
        let mut replay = take.replay();
        assert!(replay.events_due(6).is_empty());
        assert_eq!(replay.events_due(7), take.events());
        assert!(replay.finished());
        let replayed = take.addresses()[0]
            .law
            .decode(take.events()[0].value)
            .expect("stored v1 value decodes through its stored law");
        let PerformanceRawValue::Continuous(brightness) = replayed else {
            panic!("stored v1 brightness must replay on the continuous lane")
        };
        assert!(
            (brightness - 0.25).abs() < 1.0e-4,
            "stored v1 brightness replayed as {brightness}"
        );

        let reencoded = PerformanceTakeDocument::capture(&take)
            .to_json_bytes()
            .unwrap();
        assert_eq!(reencoded, literal, "the stored v1 JSON is byte-exact");
    }

    #[test]
    fn hostile_documents_are_refused_by_the_typed_vocabulary() {
        let take = recorded_take();
        let mut document = PerformanceTakeDocument::capture(&take);
        document.checksum = "00".repeat(32);
        assert!(matches!(
            document.validate(),
            Err(PerformanceError::ChecksumMismatch { .. })
        ));

        let mut document = PerformanceTakeDocument::capture(&take);
        document.event_count += 1;
        assert!(matches!(
            document.validate(),
            Err(PerformanceError::EventCount { .. })
        ));

        let mut document = PerformanceTakeDocument::capture(&take);
        document.version = 2;
        assert_eq!(
            document.validate(),
            Err(PerformanceError::UnsupportedVersion(2))
        );

        let mut document = PerformanceTakeDocument::capture(&take);
        document.events[0].address = 99;
        document.event_count = document.events.len() as u32;
        assert!(matches!(
            document.validate(),
            Err(PerformanceError::AddressOutOfRange { .. })
        ));

        let mut document = PerformanceTakeDocument::capture(&take);
        document.length_ticks = 1;
        assert!(matches!(
            document.validate(),
            Err(PerformanceError::LengthBeforeLastEvent { .. })
        ));
    }

    #[test]
    fn unknown_fields_and_oversized_documents_are_rejected() {
        let take = recorded_take();
        let document = PerformanceTakeDocument::capture(&take);
        let mut value = serde_json::to_value(&document).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("hostile".to_string(), serde_json::Value::Bool(true));
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(PerformanceTakeDocument::from_json_bytes(&bytes).is_err());

        // A hostile extra field inside one address is likewise rejected: the
        // flat hand-validated encoding is what buys this, since serde ignores
        // `deny_unknown_fields` on tagged enums.
        let mut value = serde_json::to_value(&document).unwrap();
        value["addresses"][0]
            .as_object_mut()
            .unwrap()
            .insert("hostile".to_string(), serde_json::Value::Bool(true));
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(PerformanceTakeDocument::from_json_bytes(&bytes).is_err());

        // A law field mixed across kinds is a malformed shape, not a repair.
        let mut value = serde_json::to_value(&document).unwrap();
        value["addresses"][0]
            .as_object_mut()
            .unwrap()
            .insert("vocab".to_string(), serde_json::json!(["a"]));
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(PerformanceTakeDocument::from_json_bytes(&bytes).is_err());

        assert!(matches!(
            PerformanceTakeDocument::from_json_bytes(&[]),
            Err(PerformanceError::DocumentBytes(0))
        ));
        let oversized = vec![b' '; MAX_PERFORMANCE_SERIALIZED_BYTES + 1];
        assert!(matches!(
            PerformanceTakeDocument::from_json_bytes(&oversized),
            Err(PerformanceError::DocumentBytes(_))
        ));
    }

    #[test]
    fn an_out_of_vocab_value_code_is_refused_at_the_document_boundary() {
        let mut take = PerformanceTake::default();
        take.record_accepted(
            0,
            PerformanceControl::Temporal {
                param: "slit_map".to_string(),
            },
            PerformanceValueLaw::Discrete {
                vocab: vec!["ramp".to_string(), "sweep".to_string()],
            },
            &PerformanceRawValue::Token("sweep".to_string()),
        )
        .unwrap();
        take.finalize(0);
        let mut document = PerformanceTakeDocument::capture(&take);
        document.events[0].value = 7;
        assert_eq!(
            document.validate(),
            Err(PerformanceError::InvalidValue(7)),
            "a code outside the declared vocabulary can never reach replay"
        );
    }

    #[test]
    fn the_replay_cursor_is_monotonic_and_returns_crossed_events_in_order() {
        let take = recorded_take();
        let mut replay = take.replay();
        assert_eq!(replay.events_due(0).len(), 1);
        let crossed = replay.events_due(5);
        assert_eq!(crossed.len(), 2);
        assert_eq!(crossed[0].address, 1);
        assert_eq!(crossed[1].address, 0);
        assert!(replay.finished());
        assert!(replay.events_due(0).is_empty(), "the cursor never rewinds");

        let mut cursor = PerformanceCursor::default();
        assert_eq!(cursor.range_due(take.events(), 0), (0, 1));
        assert_eq!(cursor.range_due(take.events(), 5), (1, 3));
        assert!(cursor.finished(take.events()));
    }

    #[test]
    fn the_clock_derives_the_tick_before_adding_the_delta() {
        let mut clock = PerformanceClock::default();
        assert_eq!(
            clock.reference_tick(),
            0,
            "the first accepted frame is tick 0"
        );
        clock.advance_accepted(1.0 / PERFORMANCE_REFERENCE_FPS);
        assert_eq!(clock.reference_tick(), 1);
        clock.advance_accepted(f32::NAN);
        assert_eq!(
            clock.reference_tick(),
            2,
            "a hostile delta takes the shared sanitize law's single reference tick"
        );
        clock.reset();
        assert_eq!(clock.reference_tick(), 0);
    }

    #[test]
    fn finalize_never_stamps_a_length_before_the_last_event() {
        let mut take = recorded_take();
        take.finalize(1);
        assert_eq!(take.length_ticks(), 3);
        take.finalize(100);
        assert_eq!(take.length_ticks(), 100);
    }

    #[test]
    fn grouping_of_edits_into_frames_never_changes_the_digest() {
        // The same three edits recorded in one frame batch and across three
        // frame batches must hash identically: the checksum covers the
        // portable stream only.
        let mut grouped = PerformanceTake::default();
        for _ in 0..3 {
            grouped
                .record_accepted(
                    2,
                    master("hue_shift"),
                    unit_law(),
                    &PerformanceRawValue::Continuous(0.5),
                )
                .unwrap();
        }
        grouped.finalize(2);
        let mut split = PerformanceTake::default();
        split
            .record_accepted(
                2,
                master("hue_shift"),
                unit_law(),
                &PerformanceRawValue::Continuous(0.5),
            )
            .unwrap();
        for _ in 0..2 {
            split
                .record_accepted(
                    2,
                    master("hue_shift"),
                    unit_law(),
                    &PerformanceRawValue::Continuous(0.5),
                )
                .unwrap();
        }
        split.finalize(2);
        assert_eq!(grouped.checksum_hex(), split.checksum_hex());
    }
}
