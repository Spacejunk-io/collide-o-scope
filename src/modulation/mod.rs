//! Modulation matrix: internal LFO sources routed to effect parameters.
//!
//! This is the engine's autonomous heartbeat. Each frame the render loop
//! calls `update` (advancing the beat clock and sampling every LFO), then
//! `modulate` to produce *modulated copies* of the master effects and NTSC
//! params. Base values — what the UI sliders edit — are never mutated, so
//! manual control and modulation compose: the slider sets the center, the
//! LFO breathes around it.
//!
//! LFO rates are expressed in beats (quarter notes), synced to a BPM clock
//! driven by tap tempo. Every source added later (audio transients, MIDI
//! CCs, gyroscope axes) should enter through this same matrix: a source
//! produces a value in [-1, 1], a routing scales it into a parameter range.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::audio::{AudioBandConfig, AudioLevels, MAX_AUDIO_BANDS};
use crate::composition::RuntimeComposition;
use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::image_routing::StableLayerId;
use crate::motion::MotionParams;
use crate::ntsc::NtscParams;
use crate::performance::SavedLayerPosition;
use crate::spatial::{
    SpatialTransform, ANCHOR_MAX, ANCHOR_MIN, CROP_MAX, POSITION_MAX, POSITION_MIN, SCALE_MAX,
    SCALE_MIN, SKEW_LIMIT_DEGREES,
};
use crate::visual_rack::{
    GroupId, NodeId, NodeKindTag, NodeParamType, RuntimeMaskParams, RuntimeVisualNode,
    RuntimeVisualNodeKind, RuntimeVisualRack, NODE_PARAM_DESCRIPTORS,
};

pub const NUM_LFOS: usize = 4;
pub const MAX_ROUTINGS: usize = 64;
pub const AUDIO_SOURCE_LIVE: &str = "live";
pub const AUDIO_SOURCE_FILE: &str = "file";

pub fn normalize_audio_source_kind(value: &str) -> &'static str {
    if value.eq_ignore_ascii_case(AUDIO_SOURCE_FILE) {
        AUDIO_SOURCE_FILE
    } else {
        AUDIO_SOURCE_LIVE
    }
}

/// Modulation targets: (key, min, max). Depth ±1.0 spans half the range
/// in each direction from the base value, clamped to [min, max].
pub const TARGETS: &[(&str, f32, f32)] = &[
    ("pixelate", 1.0, 32.0),
    ("rgb_split", 0.0, 30.0),
    ("hue_shift", -180.0, 180.0),
    ("saturation", -1.0, 1.0),
    ("brightness", -1.0, 1.0),
    ("contrast", -1.0, 1.0),
    ("posterize", 0.0, 16.0),
    ("grain_intensity", 0.0, 0.3),
    ("grain_size", 1.0, 4.0),
    ("vignette", 0.0, 1.5),
    ("color_drift", 0.0, 0.02),
    ("downsample", 0.05, 1.0),
    ("breathe_scale", 0.0, 0.05),
    ("breathe_rotation", 0.0, 2.0),
    ("breathe_position", 0.0, 0.02),
    ("key_threshold", 0.0, 1.0),
    ("key_softness", 0.0, 0.5),
    ("key_color_r", 0.0, 1.0),
    ("key_color_g", 0.0, 1.0),
    ("key_color_b", 0.0, 1.0),
    ("key_tolerance", 0.0, 1.0),
    ("cellular_amount", 0.0, 1.0),
    ("cellular_scale", 2.0, 32.0),
    ("cellular_warp", 0.0, 1.0),
    ("cellular_speed", 0.0, 2.0),
    ("cellular_gap_amount", 0.0, 1.0),
    ("cellular_gap_threshold", 0.0, 1.0),
    ("cellular_gap_softness", 0.0, 0.5),
    ("shift_amount", 0.0, 1.0),
    ("shift_block_size", 2.0, 256.0),
    ("shift_density", 0.0, 1.0),
    ("shift_speed", 0.0, 20.0),
    ("ntsc_snow", 0.0, 1.0),
    ("ntsc_tracking_snow", 0.0, 1.0),
    ("ntsc_edge_wave", 0.0, 20.0),
    ("ntsc_edge_wave_speed", 0.0, 10.0),
    ("ntsc_head_shift", -100.0, 100.0),
    ("ntsc_tracking_wave", 0.0, 50.0),
    ("ntsc_chroma_loss", 0.0, 0.01),
    ("ntsc_composite_noise", 0.0, 0.5),
    ("ntsc_luma_noise", 0.0, 0.2),
    ("ntsc_chroma_noise", 0.0, 0.5),
    ("ntsc_luma_smear", 0.0, 1.0),
    ("ntsc_sharpening", -1.0, 2.0),
    ("temporal_feedback", 0.0, 0.95),
    ("temporal_slitscan", 0.0, 1.0),
    ("temporal_fb_zoom", 0.9, 1.1),
    ("temporal_fb_rotate", -5.0, 5.0),
    ("temporal_slit_angle", -180.0, 180.0),
    ("temporal_key_threshold", 0.0, 1.0),
    ("temporal_key_softness", 0.0, 0.5),
    ("temporal_key_history", 1.0, 23.0),
    // M3 originals expose only continuous bounded values. Topology/gate,
    // seeds, fold/count controls, and Collision Score configuration remain
    // discrete authored endpoint state.
    ("temporal_loom_amount", 0.0, 1.0),
    ("temporal_loom_depth", 0.0, 1.0),
    ("temporal_loom_phase", -1_000.0, 1_000.0),
    ("temporal_loom_scale", 0.01, 100.0),
    ("temporal_loom_angle", -180.0, 180.0),
    ("temporal_atlas_amount", 0.0, 1.0),
    ("temporal_atlas_collision", 0.0, 1.0),
    ("temporal_garden_amount", 0.0, 1.0),
    ("temporal_garden_threshold", 0.0, 1.0),
    ("temporal_garden_softness", 0.0, 0.5),
    ("temporal_garden_decay", 0.0, 1.0),
    // Program-wide spatial controls. Continuous geometry is modulatable;
    // discrete fit/edge/sampling choices remain explicitly authored.
    ("position_x", POSITION_MIN, POSITION_MAX),
    ("position_y", POSITION_MIN, POSITION_MAX),
    ("scale_x", SCALE_MIN, SCALE_MAX),
    ("scale_y", SCALE_MIN, SCALE_MAX),
    ("anchor_x", ANCHOR_MIN, ANCHOR_MAX),
    ("anchor_y", ANCHOR_MIN, ANCHOR_MAX),
    ("rotation_deg", -180.0, 180.0),
    ("skew_deg", -SKEW_LIMIT_DEGREES, SKEW_LIMIT_DEGREES),
    ("skew_axis_deg", -180.0, 180.0),
    ("crop_left", 0.0, CROP_MAX),
    ("crop_top", 0.0, CROP_MAX),
    ("crop_right", 0.0, CROP_MAX),
    ("crop_bottom", 0.0, CROP_MAX),
    // Master Motion exposes only Curved Shutter's continuous authored
    // controls. Faraday is a layer-recipient law; source, quality, carrier,
    // donor identity, and algorithm provenance remain discrete authored state.
    ("motion_shutter_angle", 0.0, 360.0),
    ("motion_shutter_phase", -1.0, 1.0),
    ("motion_shutter_curvature", -2.0, 2.0),
    ("motion_shutter_chromatic_lag", 0.0, 1.0),
    // S3b gesture-field etching exposes only the three derived continuous
    // canvas scalars. The recorded event track is authored topology and has no
    // modulation address at all: nothing here can add, remove, retime, or
    // rewrite a recorded gesture. The keys are deliberately prefixed rather
    // than reusing bare `radius`/`strength`/`retention`, which would alias
    // against another subsystem through key equality.
    ("gesture_radius", 0.0, 1.0),
    ("gesture_strength", 0.0, 1.0),
    ("gesture_retention", 0.0, 1.0),
    // The patch-morph crossfader; applied by the app, not apply_offset.
    ("morph", 0.0, 1.0),
];
const MORPH_TARGET_INDEX: usize = TARGETS.len() - 1;

/// Canonicalize the only retired target spelling. Runtime/UI output always
/// uses `layerN_key_threshold`; old patch files remain loadable.
pub fn canonical_target(target: &str) -> Cow<'_, str> {
    let Some(rest) = target.strip_prefix("layer") else {
        return Cow::Borrowed(target);
    };
    let Some((number, suffix)) = rest.split_once('_') else {
        return Cow::Borrowed(target);
    };
    if suffix != "key" {
        return Cow::Borrowed(target);
    }
    let Ok(layer) = number.parse::<usize>() else {
        return Cow::Borrowed(target);
    };
    if layer == 0 {
        return Cow::Borrowed(target);
    }
    Cow::Owned(format!("layer{layer}_key_threshold"))
}

/// Component selected from a vector/color descriptor. It is encoded in the
/// compact address rather than retained as an unbounded string at frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StableModComponent {
    Scalar,
    X,
    Y,
    Red,
    Green,
    Blue,
}

/// One validated modulatable parameter in the authoritative rack registry.
/// Descriptor indices are compact process-local addresses; patch and web
/// persistence always use the descriptor key rendered by [`fmt::Display`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StableNodeParameter {
    Wet,
    Descriptor {
        descriptor_index: u16,
        component: StableModComponent,
    },
}

impl StableNodeParameter {
    fn parse(value: &str) -> Option<Self> {
        if value == "wet" {
            return Some(Self::Wet);
        }
        let (key, suffix) = value
            .rsplit_once('.')
            .map_or((value, None), |(key, suffix)| (key, Some(suffix)));
        let (index, descriptor) = NODE_PARAM_DESCRIPTORS
            .iter()
            .enumerate()
            .find(|(_, descriptor)| descriptor.key == key && descriptor.modulatable)?;
        let component = match (descriptor.value_type, suffix) {
            (NodeParamType::Float, None) => StableModComponent::Scalar,
            (NodeParamType::Vec2, Some("x")) => StableModComponent::X,
            (NodeParamType::Vec2, Some("y")) => StableModComponent::Y,
            (NodeParamType::Color, Some("r")) => StableModComponent::Red,
            (NodeParamType::Color, Some("g")) => StableModComponent::Green,
            (NodeParamType::Color, Some("b")) => StableModComponent::Blue,
            _ => return None,
        };
        Some(Self::Descriptor {
            descriptor_index: u16::try_from(index).ok()?,
            component,
        })
    }

    fn descriptor(self) -> Option<&'static crate::visual_rack::NodeParamDescriptor> {
        match self {
            Self::Wet => None,
            Self::Descriptor {
                descriptor_index, ..
            } => NODE_PARAM_DESCRIPTORS.get(usize::from(descriptor_index)),
        }
    }

    pub fn range(self) -> [f32; 2] {
        match self {
            Self::Wet => [0.0, 1.0],
            Self::Descriptor { .. } => self
                .descriptor()
                .and_then(|descriptor| descriptor.range)
                .unwrap_or([0.0, 1.0]),
        }
    }

    fn is_valid_for_kind(self, kind: NodeKindTag) -> bool {
        match self {
            Self::Wet => !matches!(
                kind,
                NodeKindTag::LegacyCanonical | NodeKindTag::LegacyTemporal
            ),
            Self::Descriptor { .. } => self
                .descriptor()
                .is_some_and(|descriptor| descriptor.kind == kind && descriptor.modulatable),
        }
    }

    /// Parameters are serialized by descriptor key, while the compact typed
    /// representation retains the registry index needed for direct mutation.
    /// Several node kinds intentionally share keys such as `amount` and
    /// `speed`; compare their wire identity when resolving against a rack,
    /// then canonicalize to that rack's concrete descriptor index.
    fn same_wire_parameter(self, other: Self) -> bool {
        match (self, other) {
            (Self::Wet, Self::Wet) => true,
            (
                Self::Descriptor {
                    descriptor_index: left,
                    component: left_component,
                },
                Self::Descriptor {
                    descriptor_index: right,
                    component: right_component,
                },
            ) => {
                left_component == right_component
                    && NODE_PARAM_DESCRIPTORS
                        .get(usize::from(left))
                        .zip(NODE_PARAM_DESCRIPTORS.get(usize::from(right)))
                        .is_some_and(|(left, right)| left.key == right.key)
            }
            _ => false,
        }
    }
}

impl fmt::Display for StableNodeParameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Wet => formatter.write_str("wet"),
            Self::Descriptor {
                descriptor_index,
                component,
            } => {
                let descriptor = NODE_PARAM_DESCRIPTORS
                    .get(usize::from(descriptor_index))
                    .ok_or(fmt::Error)?;
                formatter.write_str(descriptor.key)?;
                match component {
                    StableModComponent::Scalar => Ok(()),
                    StableModComponent::X => formatter.write_str(".x"),
                    StableModComponent::Y => formatter.write_str(".y"),
                    StableModComponent::Red => formatter.write_str(".r"),
                    StableModComponent::Green => formatter.write_str(".g"),
                    StableModComponent::Blue => formatter.write_str(".b"),
                }
            }
        }
    }
}

impl Serialize for StableNodeParameter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for StableNodeParameter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).ok_or_else(|| de::Error::custom("invalid modulatable node parameter"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StableModScope {
    Master,
    Layer(StableLayerId),
    Group(GroupId),
}

/// Direct group values which are not owned by a node in the group's rack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroupModParameter {
    Opacity,
    PositionX,
    PositionY,
    ScaleX,
    ScaleY,
    AnchorX,
    AnchorY,
    RotationDeg,
    SkewDeg,
    SkewAxisDeg,
    CropLeft,
    CropTop,
    CropRight,
    CropBottom,
    MatteAmount,
    MatteThreshold,
    MatteSoftness,
}

impl GroupModParameter {
    fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "opacity" => Self::Opacity,
            "position.x" => Self::PositionX,
            "position.y" => Self::PositionY,
            "scale.x" => Self::ScaleX,
            "scale.y" => Self::ScaleY,
            "anchor.x" => Self::AnchorX,
            "anchor.y" => Self::AnchorY,
            "rotation_deg" => Self::RotationDeg,
            "skew_deg" => Self::SkewDeg,
            "skew_axis_deg" => Self::SkewAxisDeg,
            "crop_left" => Self::CropLeft,
            "crop_top" => Self::CropTop,
            "crop_right" => Self::CropRight,
            "crop_bottom" => Self::CropBottom,
            "matte.amount" => Self::MatteAmount,
            "matte.threshold" => Self::MatteThreshold,
            "matte.softness" => Self::MatteSoftness,
            _ => return None,
        })
    }

    pub const fn key(self) -> &'static str {
        match self {
            Self::Opacity => "opacity",
            Self::PositionX => "position.x",
            Self::PositionY => "position.y",
            Self::ScaleX => "scale.x",
            Self::ScaleY => "scale.y",
            Self::AnchorX => "anchor.x",
            Self::AnchorY => "anchor.y",
            Self::RotationDeg => "rotation_deg",
            Self::SkewDeg => "skew_deg",
            Self::SkewAxisDeg => "skew_axis_deg",
            Self::CropLeft => "crop_left",
            Self::CropTop => "crop_top",
            Self::CropRight => "crop_right",
            Self::CropBottom => "crop_bottom",
            Self::MatteAmount => "matte.amount",
            Self::MatteThreshold => "matte.threshold",
            Self::MatteSoftness => "matte.softness",
        }
    }

    pub const fn range(self) -> [f32; 2] {
        match self {
            Self::Opacity => [0.0, 1.0],
            Self::PositionX | Self::PositionY => [POSITION_MIN, POSITION_MAX],
            Self::ScaleX | Self::ScaleY => [SCALE_MIN, SCALE_MAX],
            Self::AnchorX | Self::AnchorY => [ANCHOR_MIN, ANCHOR_MAX],
            Self::RotationDeg | Self::SkewAxisDeg => [-180.0, 180.0],
            Self::SkewDeg => [-SKEW_LIMIT_DEGREES, SKEW_LIMIT_DEGREES],
            Self::CropLeft | Self::CropTop | Self::CropRight | Self::CropBottom => [0.0, CROP_MAX],
            Self::MatteAmount | Self::MatteThreshold => [0.0, 1.0],
            Self::MatteSoftness => [0.0, 0.5],
        }
    }
}

/// Composition-wide continuous values which are not owned by a group or rack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompositionModParameter {
    BusCrossfade,
}

impl CompositionModParameter {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "bus_crossfade" => Some(Self::BusCrossfade),
            _ => None,
        }
    }

    pub const fn key(self) -> &'static str {
        match self {
            Self::BusCrossfade => "bus_crossfade",
        }
    }

    pub const fn range(self) -> [f32; 2] {
        match self {
            Self::BusCrossfade => [0.0, 1.0],
        }
    }
}

impl Serialize for CompositionModParameter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.key())
    }
}

impl<'de> Deserialize<'de> for CompositionModParameter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value)
            .ok_or_else(|| de::Error::custom("invalid modulatable composition parameter"))
    }
}

impl Serialize for GroupModParameter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.key())
    }
}

impl<'de> Deserialize<'de> for GroupModParameter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).ok_or_else(|| de::Error::custom("invalid modulatable group parameter"))
    }
}

/// Saved scope uses a bounded layer position and therefore never persists a
/// process-local [`StableLayerId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum SavedStableModScope {
    Master,
    SavedLayer { position: SavedLayerPosition },
    Group { group_id: GroupId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target", rename_all = "snake_case")]
pub enum SavedMissingTarget {
    Node {
        node_id: NodeId,
        parameter: StableNodeParameter,
    },
    GroupValue {
        parameter: GroupModParameter,
    },
}

/// Stable modulation target persisted in a patch. Explicit missing variants
/// preserve authored intent after deletion/failed mapping and can never
/// retarget a newly inserted layer, group, or node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SavedStableModTarget {
    Node {
        scope: SavedStableModScope,
        node_id: NodeId,
        parameter: StableNodeParameter,
    },
    GroupValue {
        group_id: GroupId,
        parameter: GroupModParameter,
    },
    CompositionValue {
        parameter: CompositionModParameter,
    },
    MissingSavedLayer {
        saved_position: SavedLayerPosition,
        node_id: NodeId,
        parameter: StableNodeParameter,
    },
    MissingGroup {
        group_id: GroupId,
        missing_target: SavedMissingTarget,
    },
    MissingNode {
        scope: SavedStableModScope,
        node_id: NodeId,
        parameter: StableNodeParameter,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedStableModTarget {
    Live(StableModTarget),
    Missing(SavedStableModTarget),
}

/// Typed form of the stable web/persistence grammar. IDs are numeric domain
/// values here and decimal strings only at the browser boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StableModTarget {
    Node {
        scope: StableModScope,
        node_id: NodeId,
        parameter: StableNodeParameter,
    },
    GroupValue {
        group_id: GroupId,
        parameter: GroupModParameter,
    },
    CompositionValue {
        parameter: CompositionModParameter,
    },
}

impl StableModTarget {
    pub fn parse(value: &str) -> Option<Self> {
        let parts = value.split('/').collect::<Vec<_>>();
        match parts.as_slice() {
            ["node", "master", node, parameter] => Some(Self::Node {
                scope: StableModScope::Master,
                node_id: parse_node_id(node)?,
                parameter: StableNodeParameter::parse(parameter)?,
            }),
            ["node", "layer", layer_id, node, parameter] => Some(Self::Node {
                scope: StableModScope::Layer(parse_stable_layer_id(layer_id)?),
                node_id: parse_node_id(node)?,
                parameter: StableNodeParameter::parse(parameter)?,
            }),
            ["node", "group", group, node, parameter] => Some(Self::Node {
                scope: StableModScope::Group(parse_group_id(group)?),
                node_id: parse_node_id(node)?,
                parameter: StableNodeParameter::parse(parameter)?,
            }),
            ["group", group, parameter] => Some(Self::GroupValue {
                group_id: parse_group_id(group)?,
                parameter: GroupModParameter::parse(parameter)?,
            }),
            ["composition", parameter] => Some(Self::CompositionValue {
                parameter: CompositionModParameter::parse(parameter)?,
            }),
            _ => None,
        }
    }

    pub fn range(self) -> [f32; 2] {
        match self {
            Self::Node { parameter, .. } => parameter.range(),
            Self::GroupValue { parameter, .. } => parameter.range(),
            Self::CompositionValue { parameter } => parameter.range(),
        }
    }
}

impl fmt::Display for StableModTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Node {
                scope: StableModScope::Master,
                node_id,
                parameter,
            } => write!(
                formatter,
                "node/master/{node_id}/{parameter}",
                node_id = node_id.get()
            ),
            Self::Node {
                scope: StableModScope::Layer(layer_id),
                node_id,
                parameter,
            } => write!(
                formatter,
                "node/layer/{}/{}/{parameter}",
                layer_id.get(),
                node_id.get()
            ),
            Self::Node {
                scope: StableModScope::Group(group_id),
                node_id,
                parameter,
            } => write!(
                formatter,
                "node/group/{}/{}/{parameter}",
                group_id.get(),
                node_id.get()
            ),
            Self::GroupValue {
                group_id,
                parameter,
            } => write!(formatter, "group/{}/{}", group_id.get(), parameter.key()),
            Self::CompositionValue { parameter } => {
                write!(formatter, "composition/{}", parameter.key())
            }
        }
    }
}

fn decimal_u64(value: &str) -> Option<u64> {
    (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn parse_node_id(value: &str) -> Option<NodeId> {
    NodeId::new(decimal_u64(value)?)
}

fn parse_group_id(value: &str) -> Option<GroupId> {
    GroupId::new(decimal_u64(value)?)
}

fn parse_stable_layer_id(value: &str) -> Option<StableLayerId> {
    StableLayerId::new(decimal_u64(value)?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StableModAddress(u16);

impl StableModAddress {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Dense lookup rebuilt only when creative topology changes. Missing stable
/// targets never allocate sparse ID-sized arrays and simply resolve to None.
#[derive(Debug, Clone, Default)]
pub struct StableModAddressBook {
    addresses: BTreeMap<StableModTarget, StableModAddress>,
    targets: Vec<StableModTarget>,
}

impl StableModAddressBook {
    pub fn from_composition(
        master_rack: &RuntimeVisualRack,
        layer_racks: &[(StableLayerId, RuntimeVisualRack)],
        composition: &RuntimeComposition,
    ) -> Result<Self, String> {
        let mut book = Self::default();
        book.add_rack(StableModScope::Master, master_rack)?;
        for (layer_id, rack) in layer_racks {
            book.add_rack(StableModScope::Layer(*layer_id), rack)?;
        }
        book.insert(StableModTarget::CompositionValue {
            parameter: CompositionModParameter::BusCrossfade,
        })?;
        for group in composition.groups() {
            let direct_parameters = [
                GroupModParameter::Opacity,
                GroupModParameter::PositionX,
                GroupModParameter::PositionY,
                GroupModParameter::ScaleX,
                GroupModParameter::ScaleY,
                GroupModParameter::AnchorX,
                GroupModParameter::AnchorY,
                GroupModParameter::RotationDeg,
                GroupModParameter::SkewDeg,
                GroupModParameter::SkewAxisDeg,
                GroupModParameter::CropLeft,
                GroupModParameter::CropTop,
                GroupModParameter::CropRight,
                GroupModParameter::CropBottom,
            ];
            for parameter in direct_parameters {
                book.insert(StableModTarget::GroupValue {
                    group_id: group.id,
                    parameter,
                })?;
            }
            if group.matte.is_some() {
                for parameter in [
                    GroupModParameter::MatteAmount,
                    GroupModParameter::MatteThreshold,
                    GroupModParameter::MatteSoftness,
                ] {
                    book.insert(StableModTarget::GroupValue {
                        group_id: group.id,
                        parameter,
                    })?;
                }
            }
            book.add_rack(StableModScope::Group(group.id), &group.rack)?;
        }
        Ok(book)
    }

    fn add_rack(&mut self, scope: StableModScope, rack: &RuntimeVisualRack) -> Result<(), String> {
        for node in rack.iter() {
            let kind = node.kind.tag();
            let wet = StableNodeParameter::Wet;
            if wet.is_valid_for_kind(kind) {
                self.insert(StableModTarget::Node {
                    scope,
                    node_id: node.stable_id,
                    parameter: wet,
                })?;
            }
            for (index, descriptor) in NODE_PARAM_DESCRIPTORS.iter().enumerate() {
                if descriptor.kind != kind
                    || !descriptor.modulatable
                    || !runtime_node_supports_descriptor(node, descriptor.key)
                {
                    continue;
                }
                let components: &[StableModComponent] = match descriptor.value_type {
                    NodeParamType::Float => &[StableModComponent::Scalar],
                    NodeParamType::Vec2 => &[StableModComponent::X, StableModComponent::Y],
                    NodeParamType::Color => &[
                        StableModComponent::Red,
                        StableModComponent::Green,
                        StableModComponent::Blue,
                    ],
                    _ => continue,
                };
                let descriptor_index = u16::try_from(index)
                    .map_err(|_| "node parameter registry exceeds compact address space")?;
                for &component in components {
                    self.insert(StableModTarget::Node {
                        scope,
                        node_id: node.stable_id,
                        parameter: StableNodeParameter::Descriptor {
                            descriptor_index,
                            component,
                        },
                    })?;
                }
            }
        }
        Ok(())
    }

    fn insert(&mut self, target: StableModTarget) -> Result<(), String> {
        if self.addresses.contains_key(&target) {
            return Err(format!("duplicate stable modulation target {target}"));
        }
        let index = u16::try_from(self.targets.len())
            .map_err(|_| "stable modulation address space exceeds 65535 entries")?;
        let address = StableModAddress(index);
        self.targets.push(target);
        self.addresses.insert(target, address);
        Ok(())
    }

    pub fn address(&self, target: StableModTarget) -> Option<StableModAddress> {
        self.addresses.get(&target).copied().or_else(|| {
            self.targets
                .iter()
                .position(|candidate| candidate.same_wire_target(target))
                .and_then(|index| u16::try_from(index).ok())
                .map(StableModAddress)
        })
    }

    pub fn target(&self, address: StableModAddress) -> Option<StableModTarget> {
        self.targets.get(address.index()).copied()
    }

    fn canonical_target(&self, target: StableModTarget) -> Option<StableModTarget> {
        self.address(target)
            .and_then(|address| self.target(address))
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "stable address-book inspection is exercised by persistence tests"
        )
    )]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl StableModTarget {
    fn same_wire_target(self, other: Self) -> bool {
        match (self, other) {
            (
                Self::Node {
                    scope: left_scope,
                    node_id: left_node,
                    parameter: left_parameter,
                },
                Self::Node {
                    scope: right_scope,
                    node_id: right_node,
                    parameter: right_parameter,
                },
            ) => {
                left_scope == right_scope
                    && left_node == right_node
                    && left_parameter.same_wire_parameter(right_parameter)
            }
            (
                Self::GroupValue {
                    group_id: left_group,
                    parameter: left_parameter,
                },
                Self::GroupValue {
                    group_id: right_group,
                    parameter: right_parameter,
                },
            ) => left_group == right_group && left_parameter == right_parameter,
            (
                Self::CompositionValue {
                    parameter: left_parameter,
                },
                Self::CompositionValue {
                    parameter: right_parameter,
                },
            ) => left_parameter == right_parameter,
            _ => false,
        }
    }

    /// Capture a runtime target without ever persisting a process-local layer
    /// identity. The address book distinguishes a missing node from a valid
    /// node whose current modulation contribution merely happens to be zero.
    pub fn capture(
        self,
        book: &StableModAddressBook,
        mut position_of_layer: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
        group_exists: impl Fn(GroupId) -> bool,
    ) -> Result<SavedStableModTarget, String> {
        let canonical = book.canonical_target(self);
        Ok(match self {
            Self::Node {
                scope,
                node_id,
                parameter,
            } => {
                let parameter = match canonical {
                    Some(Self::Node {
                        parameter: canonical,
                        ..
                    }) => canonical,
                    _ => parameter,
                };
                let saved_scope = match scope {
                    StableModScope::Master => SavedStableModScope::Master,
                    StableModScope::Layer(layer_id) => {
                        let position = position_of_layer(layer_id).ok_or_else(|| {
                            format!(
                                "runtime modulation layer {} cannot be captured atomically",
                                layer_id.get()
                            )
                        })?;
                        SavedStableModScope::SavedLayer { position }
                    }
                    StableModScope::Group(group_id) => {
                        if !group_exists(group_id) {
                            return Ok(SavedStableModTarget::MissingGroup {
                                group_id,
                                missing_target: SavedMissingTarget::Node { node_id, parameter },
                            });
                        }
                        SavedStableModScope::Group { group_id }
                    }
                };
                if book.address(self).is_none() {
                    SavedStableModTarget::MissingNode {
                        scope: saved_scope,
                        node_id,
                        parameter,
                    }
                } else {
                    SavedStableModTarget::Node {
                        scope: saved_scope,
                        node_id,
                        parameter,
                    }
                }
            }
            Self::GroupValue {
                group_id,
                parameter,
            } => {
                if group_exists(group_id) && book.address(self).is_some() {
                    SavedStableModTarget::GroupValue {
                        group_id,
                        parameter,
                    }
                } else {
                    SavedStableModTarget::MissingGroup {
                        group_id,
                        missing_target: SavedMissingTarget::GroupValue { parameter },
                    }
                }
            }
            Self::CompositionValue { parameter } => {
                SavedStableModTarget::CompositionValue { parameter }
            }
        })
    }
}

impl SavedStableModTarget {
    /// Resolve a saved target atomically against current stable identities.
    /// Every explicit missing form remains missing even if its old numeric
    /// position/ID has since been reused.
    pub fn resolve(
        self,
        book: &StableModAddressBook,
        mut layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: impl Fn(GroupId) -> bool,
    ) -> ResolvedStableModTarget {
        match self {
            Self::MissingSavedLayer { .. }
            | Self::MissingGroup { .. }
            | Self::MissingNode { .. } => ResolvedStableModTarget::Missing(self),
            Self::CompositionValue { parameter } => {
                let live = StableModTarget::CompositionValue { parameter };
                if book.address(live).is_some() {
                    ResolvedStableModTarget::Live(book.canonical_target(live).unwrap_or(live))
                } else {
                    // The composition address is an invariant of every current
                    // address book. Retain the typed value rather than
                    // inventing an impossible group/node tombstone if a future
                    // bounded book deliberately omits it.
                    ResolvedStableModTarget::Missing(self)
                }
            }
            Self::GroupValue {
                group_id,
                parameter,
            } => {
                let live = StableModTarget::GroupValue {
                    group_id,
                    parameter,
                };
                if !group_exists(group_id) || book.address(live).is_none() {
                    ResolvedStableModTarget::Missing(Self::MissingGroup {
                        group_id,
                        missing_target: SavedMissingTarget::GroupValue { parameter },
                    })
                } else {
                    ResolvedStableModTarget::Live(book.canonical_target(live).unwrap_or(live))
                }
            }
            Self::Node {
                scope,
                node_id,
                parameter,
            } => {
                let live_scope = match scope {
                    SavedStableModScope::Master => StableModScope::Master,
                    SavedStableModScope::SavedLayer { position } => {
                        let Some(layer_id) = layer_at_position(position) else {
                            return ResolvedStableModTarget::Missing(Self::MissingSavedLayer {
                                saved_position: position,
                                node_id,
                                parameter,
                            });
                        };
                        StableModScope::Layer(layer_id)
                    }
                    SavedStableModScope::Group { group_id } => {
                        if !group_exists(group_id) {
                            return ResolvedStableModTarget::Missing(Self::MissingGroup {
                                group_id,
                                missing_target: SavedMissingTarget::Node { node_id, parameter },
                            });
                        }
                        StableModScope::Group(group_id)
                    }
                };
                let live = StableModTarget::Node {
                    scope: live_scope,
                    node_id,
                    parameter,
                };
                if book.address(live).is_none() {
                    ResolvedStableModTarget::Missing(Self::MissingNode {
                        scope,
                        node_id,
                        parameter,
                    })
                } else {
                    ResolvedStableModTarget::Live(book.canonical_target(live).unwrap_or(live))
                }
            }
        }
    }

    /// Non-routable diagnostic key retained in runtime Routing when a saved
    /// target cannot resolve. It is intentionally outside the accepted live
    /// web grammar and round-trips only through the typed patch field.
    pub fn persistence_key(self) -> String {
        match self {
            Self::Node {
                scope,
                node_id,
                parameter,
            } => match scope {
                SavedStableModScope::Master => {
                    format!("node/master/{}/{parameter}", node_id.get())
                }
                SavedStableModScope::SavedLayer { position } => format!(
                    "node/saved_layer/{}/{}/{parameter}",
                    position.get(),
                    node_id.get()
                ),
                SavedStableModScope::Group { group_id } => format!(
                    "node/group/{}/{}/{parameter}",
                    group_id.get(),
                    node_id.get()
                ),
            },
            Self::GroupValue {
                group_id,
                parameter,
            } => format!("group/{}/{}", group_id.get(), parameter.key()),
            Self::CompositionValue { parameter } => {
                format!("composition/{}", parameter.key())
            }
            Self::MissingSavedLayer {
                saved_position,
                node_id,
                parameter,
            } => format!(
                "missing/saved_layer/{}/{}/{parameter}",
                saved_position.get(),
                node_id.get()
            ),
            Self::MissingGroup {
                group_id,
                missing_target,
            } => match missing_target {
                SavedMissingTarget::Node { node_id, parameter } => format!(
                    "missing/group/{}/{}/{parameter}",
                    group_id.get(),
                    node_id.get()
                ),
                SavedMissingTarget::GroupValue { parameter } => {
                    format!("missing/group/{}/{}", group_id.get(), parameter.key())
                }
            },
            Self::MissingNode {
                scope,
                node_id,
                parameter,
            } => format!(
                "missing/node/{}/{}/{parameter}",
                saved_scope_key(scope),
                node_id.get()
            ),
        }
    }
}

fn saved_scope_key(scope: SavedStableModScope) -> String {
    match scope {
        SavedStableModScope::Master => "master".to_string(),
        SavedStableModScope::SavedLayer { position } => {
            format!("saved_layer/{}", position.get())
        }
        SavedStableModScope::Group { group_id } => format!("group/{}", group_id.get()),
    }
}

#[derive(Debug, Clone)]
pub struct StableModulationFrame {
    offsets: Vec<f32>,
}

impl StableModulationFrame {
    /// Full value offset after depth and half-range scaling. Missing addresses
    /// are inert by construction.
    pub fn offset(&self, address: StableModAddress) -> f32 {
        self.offsets.get(address.index()).copied().unwrap_or(0.0)
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "target-oriented lookup is retained for editor and test adapters"
        )
    )]
    pub fn target_offset(&self, book: &StableModAddressBook, target: StableModTarget) -> f32 {
        book.address(target)
            .map_or(0.0, |address| self.offset(address))
    }
}

fn runtime_node_supports_descriptor(node: &RuntimeVisualNode, key: &str) -> bool {
    match node.kind {
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(_)) => {
            key.starts_with("rectangle_")
        }
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(_)) => key.starts_with("ellipse_"),
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(_)) => key.starts_with("image_"),
        _ => true,
    }
}

fn stable_component_slot<'a>(
    component: StableModComponent,
    scalar: Option<&'a mut f32>,
    vector: Option<&'a mut [f32]>,
) -> Option<&'a mut f32> {
    match component {
        StableModComponent::Scalar => scalar,
        StableModComponent::X | StableModComponent::Red => vector?.get_mut(0),
        StableModComponent::Y | StableModComponent::Green => vector?.get_mut(1),
        StableModComponent::Blue => vector?.get_mut(2),
    }
}

fn add_stable_offset(slot: &mut f32, offset: f32, range: [f32; 2]) {
    *slot = (finite_or(*slot, range[0]) + finite_or(offset, 0.0)).clamp(range[0], range[1]);
}

fn wrap_stable_degrees(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let wrapped = (value + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 && value.is_sign_positive() {
        180.0
    } else {
        wrapped
    }
}

fn apply_stable_node_offset(
    node: &mut RuntimeVisualNode,
    parameter: StableNodeParameter,
    offset: f32,
) {
    if offset == 0.0 || !parameter.is_valid_for_kind(node.kind.tag()) {
        return;
    }
    if parameter == StableNodeParameter::Wet {
        add_stable_offset(&mut node.wet, offset, [0.0, 1.0]);
        return;
    }
    let StableNodeParameter::Descriptor {
        descriptor_index,
        component,
    } = parameter
    else {
        return;
    };
    let Some(descriptor) = NODE_PARAM_DESCRIPTORS.get(usize::from(descriptor_index)) else {
        return;
    };
    if !runtime_node_supports_descriptor(node, descriptor.key) {
        return;
    }
    let Some(range) = descriptor.range else {
        return;
    };

    let slot = match &mut node.kind {
        RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal => None,
        RuntimeVisualNodeKind::Transform(value) => match descriptor.key {
            "position" => stable_component_slot(component, None, Some(&mut value.position)),
            "scale" => stable_component_slot(component, None, Some(&mut value.scale)),
            "anchor" => stable_component_slot(component, None, Some(&mut value.anchor)),
            "rotation_deg" => Some(&mut value.rotation_deg),
            "skew_deg" => Some(&mut value.skew_deg),
            "skew_axis_deg" => Some(&mut value.skew_axis_deg),
            "crop_left" => Some(&mut value.crop[0]),
            "crop_top" => Some(&mut value.crop[1]),
            "crop_right" => Some(&mut value.crop[2]),
            "crop_bottom" => Some(&mut value.crop[3]),
            _ => None,
        },
        RuntimeVisualNodeKind::DigitalColor(value) => match descriptor.key {
            "pixelate_size" => Some(&mut value.pixelate_size),
            "rgb_split" => Some(&mut value.rgb_split),
            "downsample" => Some(&mut value.downsample),
            "hue_shift" => Some(&mut value.hue_shift),
            "saturation" => Some(&mut value.saturation),
            "brightness" => Some(&mut value.brightness),
            "contrast" => Some(&mut value.contrast),
            "posterize" => Some(&mut value.posterize),
            "invert" => Some(&mut value.invert),
            "vignette" => Some(&mut value.vignette),
            "color_drift" => Some(&mut value.color_drift),
            _ => None,
        },
        RuntimeVisualNodeKind::Key(value) => match descriptor.key {
            "threshold" => Some(&mut value.threshold),
            "softness" => Some(&mut value.softness),
            "color" => stable_component_slot(component, None, Some(&mut value.color)),
            "tolerance" => Some(&mut value.tolerance),
            _ => None,
        },
        RuntimeVisualNodeKind::Cellular(value) => match descriptor.key {
            "amount" => Some(&mut value.amount),
            "scale" => Some(&mut value.scale),
            "warp" => Some(&mut value.warp),
            "speed" => Some(&mut value.speed),
            "gap_amount" => Some(&mut value.gap_amount),
            "gap_threshold" => Some(&mut value.gap_threshold),
            "gap_softness" => Some(&mut value.gap_softness),
            _ => None,
        },
        RuntimeVisualNodeKind::Shift(value) => match descriptor.key {
            "amount" => Some(&mut value.amount),
            "block_size" => Some(&mut value.block_size),
            "density" => Some(&mut value.density),
            "speed" => Some(&mut value.speed),
            _ => None,
        },
        RuntimeVisualNodeKind::Grain(value) => match descriptor.key {
            "intensity" => Some(&mut value.intensity),
            "size" => Some(&mut value.size),
            _ => None,
        },
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(value)) => match descriptor.key {
            "rectangle_center" => stable_component_slot(component, None, Some(&mut value.center)),
            "rectangle_size" => stable_component_slot(component, None, Some(&mut value.size)),
            "rectangle_rotation_deg" => Some(&mut value.rotation_deg),
            "rectangle_feather" => Some(&mut value.feather),
            _ => None,
        },
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(value)) => match descriptor.key {
            "ellipse_center" => stable_component_slot(component, None, Some(&mut value.center)),
            "ellipse_radii" => stable_component_slot(component, None, Some(&mut value.radii)),
            "ellipse_rotation_deg" => Some(&mut value.rotation_deg),
            "ellipse_feather" => Some(&mut value.feather),
            _ => None,
        },
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(value)) => match descriptor.key {
            "image_amount" => Some(&mut value.amount),
            "image_threshold" => Some(&mut value.threshold),
            "image_softness" => Some(&mut value.softness),
            _ => None,
        },
        // Displace exposes its two gains and nothing else: the donor route and
        // the boundary law are topology and have no modulatable address.
        RuntimeVisualNodeKind::Displace(value) => match descriptor.key {
            "amount_x" => Some(&mut value.amount_x),
            "amount_y" => Some(&mut value.amount_y),
            _ => None,
        },
        // Symmetry exposes its declared continuous controls and nothing else.
        // The two image routes, the two motion routes, the mode, the boundary,
        // the authored seed, and the six mask bits are stable authored topology
        // with no modulatable address, so modulation can never rewrite the
        // sector table.
        RuntimeVisualNodeKind::Symmetry(value) => match descriptor.key {
            "symmetry_base_folds" => Some(&mut value.base_folds),
            "symmetry_fold_offset" => Some(&mut value.fold_offset),
            "symmetry_radial_phase_deg" => Some(&mut value.radial_phase_deg),
            "symmetry_orbit_phase" => Some(&mut value.orbit_phase),
            "symmetry_planar_axis_deg" => Some(&mut value.planar_axis_deg),
            "symmetry_planar_phase" => Some(&mut value.planar_phase),
            "symmetry_cell_skew" => Some(&mut value.cell_skew),
            "symmetry_spiral_scale" => Some(&mut value.spiral_scale),
            "symmetry_orbit_radius" => Some(&mut value.orbit_radius),
            "symmetry_orbit_spin_deg" => Some(&mut value.orbit_spin_deg),
            "symmetry_motion_gain" => Some(&mut value.motion_gain),
            "symmetry_hue_span" => Some(&mut value.hue_span),
            "symmetry_center" => stable_component_slot(component, None, Some(&mut value.center)),
            _ => None,
        },
        // Residual exposes its wet authority and its detail gain and nothing
        // else: both routes, both discrete laws, and the quantization seed are
        // topology and have no modulatable address.
        RuntimeVisualNodeKind::Residual(value) => match descriptor.key {
            "mix" => Some(&mut value.mix),
            "detail_gain" => Some(&mut value.detail_gain),
            _ => None,
        },
        // A Study exposes no modulatable address of any kind (the Field
        // Collider v1 precedent): its document digest is opaque authored
        // topology and its inputs arrive through the Study's own opcodes.
        RuntimeVisualNodeKind::Study(_) => None,
    };
    if let Some(slot) = slot {
        if matches!(
            descriptor.key,
            "rotation_deg"
                | "skew_axis_deg"
                | "rectangle_rotation_deg"
                | "ellipse_rotation_deg"
                | "symmetry_radial_phase_deg"
                | "symmetry_planar_axis_deg"
                | "symmetry_orbit_spin_deg"
        ) {
            *slot = wrap_stable_degrees(finite_or(*slot, 0.0) + finite_or(offset, 0.0));
        } else {
            add_stable_offset(slot, offset, range);
        }
    }
}

fn apply_stable_rack_modulation(
    book: &StableModAddressBook,
    frame: &StableModulationFrame,
    scope: StableModScope,
    rack: &mut RuntimeVisualRack,
) {
    for (index, target) in book.targets.iter().copied().enumerate() {
        let StableModTarget::Node {
            scope: target_scope,
            node_id,
            parameter,
        } = target
        else {
            continue;
        };
        if target_scope != scope {
            continue;
        }
        let Some(node) = rack.get_mut(node_id) else {
            continue;
        };
        apply_stable_node_offset(
            node,
            parameter,
            frame.offset(StableModAddress(index as u16)),
        );
    }
    for node_id in rack.iter().map(|node| node.stable_id).collect::<Vec<_>>() {
        if let Some(node) = rack.get_mut(node_id) {
            if let RuntimeVisualNodeKind::Transform(value) = &mut node.kind {
                *value = value.sanitized();
            }
        }
    }
}

/// Project one immutable stable modulation sample into caller-owned creative
/// value clones. Missing/deleted targets are inert and topology, routes,
/// enabled state, blend, group membership/bus/solo/bypass and authored bases
/// are never changed.
pub fn apply_stable_modulation(
    book: &StableModAddressBook,
    frame: &StableModulationFrame,
    master_rack: &mut RuntimeVisualRack,
    layer_racks: &mut [(StableLayerId, RuntimeVisualRack)],
    composition: &mut RuntimeComposition,
) {
    apply_stable_rack_modulation(book, frame, StableModScope::Master, master_rack);
    for (layer_id, rack) in layer_racks {
        apply_stable_rack_modulation(book, frame, StableModScope::Layer(*layer_id), rack);
    }
    for (index, target) in book.targets.iter().copied().enumerate() {
        let StableModTarget::CompositionValue { parameter } = target else {
            continue;
        };
        let offset = frame.offset(StableModAddress(index as u16));
        match parameter {
            CompositionModParameter::BusCrossfade => composition.set_bus_crossfade(
                finite_or(composition.bus_crossfade(), 0.5) + finite_or(offset, 0.0),
            ),
        }
    }
    let group_ids: Vec<_> = composition.groups().map(|group| group.id).collect();
    for group_id in group_ids {
        let Some(group) = composition.group_mut(group_id) else {
            continue;
        };
        for (index, target) in book.targets.iter().copied().enumerate() {
            let StableModTarget::GroupValue {
                group_id: target_group,
                parameter,
            } = target
            else {
                continue;
            };
            if target_group != group_id {
                continue;
            }
            let offset = frame.offset(StableModAddress(index as u16));
            let slot = match parameter {
                GroupModParameter::Opacity => Some(&mut group.opacity),
                GroupModParameter::PositionX => Some(&mut group.transform.position[0]),
                GroupModParameter::PositionY => Some(&mut group.transform.position[1]),
                GroupModParameter::ScaleX => Some(&mut group.transform.scale[0]),
                GroupModParameter::ScaleY => Some(&mut group.transform.scale[1]),
                GroupModParameter::AnchorX => Some(&mut group.transform.anchor[0]),
                GroupModParameter::AnchorY => Some(&mut group.transform.anchor[1]),
                GroupModParameter::RotationDeg => Some(&mut group.transform.rotation_deg),
                GroupModParameter::SkewDeg => Some(&mut group.transform.skew_deg),
                GroupModParameter::SkewAxisDeg => Some(&mut group.transform.skew_axis_deg),
                GroupModParameter::CropLeft => Some(&mut group.transform.crop[0]),
                GroupModParameter::CropTop => Some(&mut group.transform.crop[1]),
                GroupModParameter::CropRight => Some(&mut group.transform.crop[2]),
                GroupModParameter::CropBottom => Some(&mut group.transform.crop[3]),
                GroupModParameter::MatteAmount => {
                    group.matte.as_mut().map(|matte| &mut matte.amount)
                }
                GroupModParameter::MatteThreshold => {
                    group.matte.as_mut().map(|matte| &mut matte.threshold)
                }
                GroupModParameter::MatteSoftness => {
                    group.matte.as_mut().map(|matte| &mut matte.softness)
                }
            };
            let Some(slot) = slot else {
                continue;
            };
            if matches!(
                parameter,
                GroupModParameter::RotationDeg | GroupModParameter::SkewAxisDeg
            ) {
                *slot = wrap_stable_degrees(finite_or(*slot, 0.0) + finite_or(offset, 0.0));
            } else {
                add_stable_offset(slot, offset, parameter.range());
            }
        }
        group.transform = group.transform.sanitized();
        apply_stable_rack_modulation(
            book,
            frame,
            StableModScope::Group(group_id),
            &mut group.rack,
        );
    }
}

/// Resolve a target's legal value range, including any positive, dynamically
/// named one-based layer target. Actual-stack bounds are enforced when a
/// frame-sized routing accumulator is built, not while parsing persisted
/// target names.
pub fn target_range(target: &str) -> Option<(f32, f32)> {
    let target = canonical_target(target);
    let target = target.as_ref();
    if let Some(stable) = StableModTarget::parse(target) {
        let [min, max] = stable.range();
        return Some((min, max));
    }
    if let Some((_, min, max)) = TARGETS.iter().find(|(key, _, _)| *key == target) {
        return Some((*min, *max));
    }

    let rest = target.strip_prefix("layer")?;
    let (number, suffix) = rest.split_once('_')?;
    let layer = number.parse::<usize>().ok()?;
    if layer == 0 {
        return None;
    }
    match suffix {
        "opacity" | "key_threshold" => Some((0.0, 1.0)),
        "speed" => Some((0.25, 4.0)),
        "fps" => Some((1.0, 240.0)),
        "pixelate" => Some((1.0, 32.0)),
        "rgb_split" => Some((0.0, 30.0)),
        "hue_shift" => Some((-180.0, 180.0)),
        "saturation" | "brightness" | "contrast" => Some((-1.0, 1.0)),
        "posterize" => Some((0.0, 16.0)),
        "grain_intensity" => Some((0.0, 0.3)),
        "grain_size" => Some((1.0, 4.0)),
        "vignette" => Some((0.0, 1.5)),
        "color_drift" => Some((0.0, 0.02)),
        "breathe_scale" => Some((0.0, 0.05)),
        "breathe_rotation" => Some((0.0, 2.0)),
        "breathe_position" => Some((0.0, 0.02)),
        "key_softness" => Some((0.0, 0.5)),
        "key_color_r" | "key_color_g" | "key_color_b" | "key_tolerance" => Some((0.0, 1.0)),
        "downsample" => Some((0.05, 1.0)),
        "cellular_amount" => Some((0.0, 1.0)),
        "cellular_scale" => Some((2.0, 32.0)),
        "cellular_warp" => Some((0.0, 1.0)),
        "cellular_speed" => Some((0.0, 2.0)),
        "cellular_gap_amount" => Some((0.0, 1.0)),
        "cellular_gap_threshold" => Some((0.0, 1.0)),
        "cellular_gap_softness" => Some((0.0, 0.5)),
        "shift_amount" => Some((0.0, 1.0)),
        "shift_block_size" => Some((2.0, 256.0)),
        "shift_density" => Some((0.0, 1.0)),
        "shift_speed" => Some((0.0, 20.0)),
        "position_x" | "position_y" => Some((POSITION_MIN, POSITION_MAX)),
        "scale_x" | "scale_y" => Some((SCALE_MIN, SCALE_MAX)),
        "anchor_x" | "anchor_y" => Some((ANCHOR_MIN, ANCHOR_MAX)),
        "rotation_deg" | "skew_axis_deg" => Some((-180.0, 180.0)),
        "skew_deg" => Some((-SKEW_LIMIT_DEGREES, SKEW_LIMIT_DEGREES)),
        "crop_left" | "crop_top" | "crop_right" | "crop_bottom" => Some((0.0, CROP_MAX)),
        "motion_transplant_amount"
        | "motion_confidence_threshold"
        | "motion_refresh"
        | "motion_decay"
        | "motion_occlusion"
        | "motion_shutter_chromatic_lag" => Some((0.0, 1.0)),
        "motion_confidence_softness" => Some((0.0, 0.5)),
        "motion_shutter_angle" => Some((0.0, 360.0)),
        "motion_shutter_phase" => Some((-1.0, 1.0)),
        "motion_shutter_curvature" => Some((-2.0, 2.0)),
        _ => None,
    }
}

/// Apply only the bounded continuous gesture-canvas destinations.
///
/// This is the entire modulation surface S3b owns. It reads three derived
/// scalars off a *copy* of the authored canvas and never sees, borrows, or
/// mutates the recorded event track.
fn apply_gesture_canvas_offsets(
    canvas: &mut crate::gesture_canvas::GestureCanvasParams,
    mut offset: impl FnMut(&'static str, f32, f32) -> f32,
) {
    canvas.radius += offset("gesture_radius", 0.0, 1.0);
    canvas.strength += offset("gesture_strength", 0.0, 1.0);
    canvas.retention += offset("gesture_retention", 0.0, 1.0);
    *canvas = canvas.sanitized();
}

pub fn is_valid_target(target: &str) -> bool {
    target_range(target).is_some()
}

/// Per-layer values after modulation, aligned with the layers vec.
#[derive(Debug, Clone, Copy)]
pub struct LayerModulation {
    pub opacity: f32,
    pub speed: f32,
    pub fps: f32,
    pub effects: EffectUniforms,
    pub transform: SpatialTransform,
}

/// One-frame modulation cache. Every route is accumulated exactly once; all
/// master, morph, and layer consumers then read frame-sized indexed storage.
/// Keeping this frame-local makes route edits immediately authoritative while
/// avoiding repeated scans and target parsing in both live and export paths.
pub struct ModulationFrame {
    offsets: RoutingOffsets,
}

impl ModulationFrame {
    /// Offset for the program morph crossfader. Its compiled slot avoids a
    /// target-name scan on every live and offline frame.
    pub fn morph_offset(&self) -> f32 {
        let (_, min, max) = TARGETS[MORPH_TARGET_INDEX];
        self.offsets.master[MORPH_TARGET_INDEX] * (max - min) * 0.5
    }

    pub fn modulate(
        &self,
        effects: &EffectUniforms,
        transform: &SpatialTransform,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
    ) -> (EffectUniforms, SpatialTransform, NtscParams, TemporalParams) {
        ModMatrix::modulate_from_offsets(effects, transform, ntsc, temporal, &self.offsets)
    }

    /// Apply only bounded continuous Motion destinations to an authored
    /// master. Base state and every discrete/provenance field remain intact.
    pub fn modulate_motion(&self, motion: &MotionParams) -> MotionParams {
        ModMatrix::modulate_master_motion_from_offsets(motion, &self.offsets)
    }

    /// Apply only the bounded continuous gesture-canvas destinations to an
    /// authored base. The recorded track has no address here and is neither
    /// read nor written.
    pub fn modulate_gesture_canvas(
        &self,
        canvas: &crate::gesture_canvas::GestureCanvasParams,
    ) -> crate::gesture_canvas::GestureCanvasParams {
        ModMatrix::modulate_gesture_canvas_from_offsets(canvas, &self.offsets)
    }

    #[cfg(test)]
    pub fn modulate_layers<'a>(
        &self,
        layers: impl IntoIterator<Item = (&'a EffectUniforms, &'a SpatialTransform, f32, f32, f32)>,
    ) -> Vec<LayerModulation> {
        layers
            .into_iter()
            .enumerate()
            .map(|(index, (effects, transform, opacity, speed, fps))| {
                ModMatrix::modulate_layer_from_offsets(
                    index,
                    effects,
                    transform,
                    opacity,
                    speed,
                    fps,
                    &self.offsets,
                )
            })
            .collect()
    }

    /// Resolve one slot from this already-accumulated frame sample.
    ///
    /// Unlike [`ModMatrix::modulate_layer_full`], this does not rescan routes
    /// or allocate from an authored target index. Shared frame planners use it
    /// while walking richer layer descriptors so render and transport values
    /// can be emitted together without first building an intermediate vector.
    pub(crate) fn modulate_layer(
        &self,
        index: usize,
        effects: &EffectUniforms,
        transform: &SpatialTransform,
        opacity: f32,
        speed: f32,
        fps: f32,
    ) -> LayerModulation {
        ModMatrix::modulate_layer_from_offsets(
            index,
            effects,
            transform,
            opacity,
            speed,
            fps,
            &self.offsets,
        )
    }

    /// Resolve one layer's Motion values from the same frame-local routing
    /// accumulator used by effects and transport.
    pub(crate) fn modulate_layer_motion(
        &self,
        index: usize,
        motion: &MotionParams,
    ) -> MotionParams {
        ModMatrix::modulate_layer_motion_from_offsets(index, motion, &self.offsets)
    }
}

/// Parsed destination kept beside a route so the render path never reparses
/// `layerN_*` strings or allocates formatted target names at frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompiledTarget {
    Master(usize),
    Layer { index: usize, suffix: usize },
    Stable(StableModTarget),
    Invalid,
}

fn master_target_index(target: &str) -> Option<usize> {
    TARGETS.iter().position(|(key, _, _)| *key == target)
}

fn compile_target(target: &str) -> CompiledTarget {
    let target = canonical_target(target);
    if let Some(index) = master_target_index(target.as_ref()) {
        return CompiledTarget::Master(index);
    }
    if let Some(target) = StableModTarget::parse(target.as_ref()) {
        return CompiledTarget::Stable(target);
    }
    let Some((layer, suffix)) = parse_layer_target(target.as_ref()) else {
        return CompiledTarget::Invalid;
    };
    layer_suffix_index(suffix).map_or(CompiledTarget::Invalid, |suffix| CompiledTarget::Layer {
        index: layer,
        suffix,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LfoShape {
    Sine,
    Triangle,
    Saw,
    Square,
    SampleHold,
}

impl LfoShape {
    pub fn from_str(s: &str) -> Self {
        match s {
            "triangle" => Self::Triangle,
            "saw" => Self::Saw,
            "square" => Self::Square,
            "sample_hold" => Self::SampleHold,
            _ => Self::Sine,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sine => "sine",
            Self::Triangle => "triangle",
            Self::Saw => "saw",
            Self::Square => "square",
            Self::SampleHold => "sample_hold",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lfo {
    pub shape: LfoShape,
    /// Cycle length in beats (quarter notes): 4.0 = one cycle per 4/4 bar.
    pub beats: f32,
    /// Phase offset, 0..1 of a cycle.
    pub phase: f32,
    /// Independent deterministic seed for Sample & Hold. Zero reproduces the
    /// historical cycle/index sequence exactly.
    pub seed: u32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            shape: LfoShape::Sine,
            beats: 4.0,
            phase: 0.0,
            seed: 0,
        }
    }
}

impl Lfo {
    /// Set a finite phase offset and wrap it into one cycle.
    /// Non-finite external values safely reset to the cycle origin.
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = finite_or(phase, 0.0).rem_euclid(1.0);
    }

    /// Finite, wrapped phase for snapshots and other read-only consumers.
    pub fn normalized_phase(&self) -> f32 {
        finite_or(self.phase, 0.0).rem_euclid(1.0)
    }

    /// Bipolar output in [-1, 1] at the given global beat position.
    pub fn value(&self, beat: f64, lfo_index: usize) -> f32 {
        let beats = self.beats.max(0.0625) as f64;
        // Keep the sampler finite even if a caller bypasses `set_phase` and
        // writes the public compatibility field directly.
        let phase = self.normalized_phase() as f64;
        let cycles = beat / beats + phase;
        let p = cycles.rem_euclid(1.0) as f32;
        match self.shape {
            LfoShape::Sine => (p * std::f32::consts::TAU).sin(),
            LfoShape::Triangle => 1.0 - 4.0 * (p - 0.5).abs(),
            LfoShape::Saw => 2.0 * p - 1.0,
            LfoShape::Square => {
                if p < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoShape::SampleHold => {
                // Deterministic pseudo-random value held for each full cycle.
                let cycle = cycles.floor() as i64 as u64;
                let mut h = cycle
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(lfo_index as u64 + 1);
                if self.seed != 0 {
                    h ^= u64::from(self.seed).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                }
                h ^= h >> 33;
                h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                h ^= h >> 33;
                (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
            }
        }
    }
}

/// Number of assignable MIDI CC slots (A–D in the UI).
pub const NUM_MIDI_SLOTS: usize = 4;

/// A modulation source: an internal LFO, an audio analysis band, or a
/// MIDI CC slot. LFOs are bipolar [-1, 1]; audio and MIDI are unipolar [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModSource {
    Lfo(usize),
    AudioLevel,
    AudioBass,
    AudioMid,
    AudioHigh,
    /// One of the configurable analysis bands, zero-indexed internally and
    /// serialized as `audio_band1` through `audio_band8`.
    AudioBand(usize),
    AudioOnset,
    AudioBright,
    AudioNoise,
    Midi(usize),
    GyroYaw,
    GyroPitch,
    GyroRoll,
    PadX,
    PadY,
}

impl ModSource {
    pub fn try_from_str(s: &str) -> Option<Self> {
        Some(match s {
            "lfo0" => Self::Lfo(0),
            "lfo1" => Self::Lfo(1),
            "lfo2" => Self::Lfo(2),
            "lfo3" => Self::Lfo(3),
            "audio_level" => Self::AudioLevel,
            "audio_bass" => Self::AudioBass,
            "audio_mid" => Self::AudioMid,
            "audio_high" => Self::AudioHigh,
            "audio_band1" => Self::AudioBand(0),
            "audio_band2" => Self::AudioBand(1),
            "audio_band3" => Self::AudioBand(2),
            "audio_band4" => Self::AudioBand(3),
            "audio_band5" => Self::AudioBand(4),
            "audio_band6" => Self::AudioBand(5),
            "audio_band7" => Self::AudioBand(6),
            "audio_band8" => Self::AudioBand(7),
            "audio_onset" => Self::AudioOnset,
            "audio_bright" => Self::AudioBright,
            "audio_noise" => Self::AudioNoise,
            "midi_a" => Self::Midi(0),
            "midi_b" => Self::Midi(1),
            "midi_c" => Self::Midi(2),
            "midi_d" => Self::Midi(3),
            "gyro_yaw" => Self::GyroYaw,
            "gyro_pitch" => Self::GyroPitch,
            "gyro_roll" => Self::GyroRoll,
            "pad_x" => Self::PadX,
            "pad_y" => Self::PadY,
            _ => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lfo(0) => "lfo0",
            Self::Lfo(1) => "lfo1",
            Self::Lfo(2) => "lfo2",
            Self::Lfo(_) => "lfo3",
            Self::AudioLevel => "audio_level",
            Self::AudioBass => "audio_bass",
            Self::AudioMid => "audio_mid",
            Self::AudioHigh => "audio_high",
            Self::AudioBand(0) => "audio_band1",
            Self::AudioBand(1) => "audio_band2",
            Self::AudioBand(2) => "audio_band3",
            Self::AudioBand(3) => "audio_band4",
            Self::AudioBand(4) => "audio_band5",
            Self::AudioBand(5) => "audio_band6",
            Self::AudioBand(6) => "audio_band7",
            Self::AudioBand(_) => "audio_band8",
            Self::AudioOnset => "audio_onset",
            Self::AudioBright => "audio_bright",
            Self::AudioNoise => "audio_noise",
            Self::Midi(0) => "midi_a",
            Self::Midi(1) => "midi_b",
            Self::Midi(2) => "midi_c",
            Self::Midi(_) => "midi_d",
            Self::GyroYaw => "gyro_yaw",
            Self::GyroPitch => "gyro_pitch",
            Self::GyroRoll => "gyro_roll",
            Self::PadX => "pad_x",
            Self::PadY => "pad_y",
        }
    }
}

/// Response shape applied to a source before routing depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// `SCurve` is part of the persisted patch and web-control vocabulary.
#[allow(clippy::enum_variant_names)]
pub enum Curve {
    Linear,
    Exp,
    Log,
    SCurve,
    Steps,
}

impl Curve {
    pub fn from_str(value: &str) -> Self {
        match value {
            "exp" => Self::Exp,
            "log" => Self::Log,
            "s_curve" => Self::SCurve,
            "steps" => Self::Steps,
            _ => Self::Linear,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::SCurve => "s_curve",
            Self::Steps => "steps",
        }
    }
}

/// Runtime and persisted configuration for one gyroscope axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroAxisConfig {
    /// Orientation which maps to the centered value 0.5.
    pub center_degrees: f32,
    /// Absolute degrees from center which reach either end of the range.
    pub range_degrees: f32,
    /// Centered exponent control: k = 2^expo.
    pub expo: f32,
    pub invert: bool,
}

impl GyroAxisConfig {
    fn with_range(range_degrees: f32) -> Self {
        Self {
            center_degrees: 0.0,
            range_degrees,
            expo: 0.0,
            invert: false,
        }
    }
}

/// Runtime and persisted shaping for one XY-pad axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadAxisConfig {
    pub curve: Curve,
    pub curve_amount: f32,
    /// Number of evenly spaced positions, including both 0 and 1. Values are
    /// snapped to the nearest position; zero or one disables quantization.
    pub quantize: u32,
}

impl Default for PadAxisConfig {
    fn default() -> Self {
        Self {
            curve: Curve::Linear,
            curve_amount: 0.0,
            quantize: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadConfig {
    pub axes: [PadAxisConfig; 2],
    pub spring_enabled: bool,
    /// Exponential return rate in inverse seconds.
    pub spring_rate: f32,
}

impl Default for PadConfig {
    fn default() -> Self {
        Self {
            axes: [PadAxisConfig::default(); 2],
            spring_enabled: false,
            spring_rate: 4.0,
        }
    }
}

/// A single modulation routing: source → parameter, shaped and scaled by depth.
#[derive(Debug, Clone)]
pub struct Routing {
    id: u64,
    pub source: ModSource,
    target: String,
    pub depth: f32,
    pub curve: Curve,
    pub curve_amount: f32,
    /// Seconds to follow a rising source. Zero is instantaneous.
    pub attack: f32,
    /// Seconds to follow a falling source. Zero is instantaneous.
    pub release: f32,
    compiled_target: CompiledTarget,
    /// Typed patch diagnostic retained when saved identity cannot resolve.
    /// It never contributes and is cleared by an authored live target edit.
    saved_missing_target: Option<SavedStableModTarget>,
    state: f32,
    cached: f32,
}

static NEXT_ROUTING_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_routing_id() -> u64 {
    NEXT_ROUTING_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("modulation routing id space exhausted")
}

impl Routing {
    pub fn new(source: ModSource, target: impl Into<String>, depth: f32) -> Self {
        let target = target.into();
        let target = canonical_target(&target).into_owned();
        Self {
            id: allocate_routing_id(),
            source,
            compiled_target: compile_target(&target),
            target,
            saved_missing_target: None,
            depth,
            curve: Curve::Linear,
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
            state: 0.0,
            cached: 0.0,
        }
    }

    pub fn route_id(&self) -> u64 {
        self.id
    }

    pub fn cached_value(&self) -> f32 {
        finite_or(self.cached, 0.0).clamp(-1.0, 1.0)
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn stable_target(&self) -> Option<StableModTarget> {
        match self.compiled_target {
            CompiledTarget::Stable(target) => Some(target),
            _ => None,
        }
    }

    pub fn saved_missing_target(&self) -> Option<SavedStableModTarget> {
        self.saved_missing_target
    }

    pub fn new_missing(source: ModSource, target: SavedStableModTarget, depth: f32) -> Self {
        let mut routing = Self::new(source, target.persistence_key(), depth);
        routing.compiled_target = CompiledTarget::Invalid;
        routing.saved_missing_target = Some(target);
        routing
    }

    /// Clear transient state after replacing a route's identity/configuration.
    pub fn reset_runtime(&mut self) {
        self.state = 0.0;
        self.cached = 0.0;
    }

    /// Change a semantic destination atomically and clear transient response
    /// state. Source/target changes define a new signal; response-time edits do
    /// not and therefore deliberately preserve continuity.
    pub fn set_target(&mut self, target: impl Into<String>) -> bool {
        let target = target.into();
        let target = canonical_target(&target).into_owned();
        if self.target == target {
            return false;
        }
        self.compiled_target = compile_target(&target);
        self.target = target;
        self.saved_missing_target = None;
        self.reset_runtime();
        true
    }

    /// Retain an authored stable destination as an explicit, non-routable
    /// persistence tombstone. Route identity and response configuration stay
    /// intact; only the vanished destination and its transient signal state
    /// change.
    fn set_missing_target(&mut self, target: SavedStableModTarget) {
        self.target = target.persistence_key();
        self.compiled_target = CompiledTarget::Invalid;
        self.saved_missing_target = Some(target);
        self.reset_runtime();
    }

    fn advance(&mut self, desired: f32, dt: f32) {
        let tau = if desired >= self.state {
            self.attack
        } else {
            self.release
        };
        self.state = exponential_follow(self.state, desired, dt, tau);
        self.cached = self.state;
    }
}

/// Beat clock with tap tempo. Tapping re-anchors the downbeat, so the
/// performer's taps both set the tempo and align the LFO phase to it.
pub struct Clock {
    pub bpm: f32,
    anchor: Instant,
    taps: Vec<Instant>,
    /// When Some, an external MIDI clock owns the beat position; the
    /// internal anchor-based clock is bypassed until it goes away.
    external_beat: Option<f64>,
    /// Keeps the public beat continuous when a paused program resumes while
    /// an external MIDI clock has continued advancing in the background.
    external_offset: f64,
    /// A frozen logical beat while the master program transport is paused.
    /// Incoming MIDI clock telemetry may continue updating `external_beat`,
    /// but it cannot move the rendered modulation phase until resume.
    paused_beat: Option<f64>,
}

const TAP_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_TAPS: usize = 8;

impl Clock {
    pub fn new() -> Self {
        Self {
            bpm: 120.0,
            anchor: Instant::now(),
            taps: Vec::new(),
            external_beat: None,
            external_offset: 0.0,
            paused_beat: None,
        }
    }

    /// Global beat position (quarter notes since the last downbeat anchor),
    /// or the external MIDI clock's position when one is driving.
    pub fn beat(&self, now: Instant) -> f64 {
        if let Some(beat) = self.paused_beat {
            return beat;
        }
        match self.external_beat {
            Some(beat) => beat + self.external_offset,
            None => self.internal_beat(now),
        }
    }

    fn internal_beat(&self, now: Instant) -> f64 {
        now.saturating_duration_since(self.anchor).as_secs_f64() * (self.bpm as f64 / 60.0)
    }

    fn anchor_internal_at(&mut self, beat: f64, now: Instant) {
        let elapsed = finite_f64_or(beat, 0.0).max(0.0) / (self.bpm as f64 / 60.0);
        self.anchor = now
            .checked_sub(Duration::from_secs_f64(elapsed))
            .unwrap_or(now);
    }

    /// Freeze or resume the logical beat without accumulating wall-clock or
    /// external MIDI catch-up. This is idempotent so repeated absolute web
    /// transport commands cannot perturb phase.
    pub fn set_paused(&mut self, paused: bool, now: Instant) {
        match (paused, self.paused_beat) {
            (true, None) => self.paused_beat = Some(self.beat(now)),
            (false, Some(frozen)) => {
                self.paused_beat = None;
                if let Some(raw_external) = self.external_beat {
                    self.external_offset = frozen - raw_external;
                } else {
                    self.anchor_internal_at(frozen, now);
                }
            }
            _ => {}
        }
    }

    #[cfg(test)]
    pub fn is_paused(&self) -> bool {
        self.paused_beat.is_some()
    }

    /// Hand the beat position to (or take it back from) an external clock.
    /// When the external clock disappears, re-anchor so the internal clock
    /// continues from the same position instead of jumping.
    pub fn set_external_beat(&mut self, beat: Option<f64>, now: Instant) {
        if let Some(raw) = beat {
            let raw = finite_f64_or(raw, 0.0).max(0.0);
            if self.external_beat.is_none() {
                // Preserve the legacy handoff (external beat is authoritative)
                // while running. During pause, align the new source underneath
                // the frozen phase so its eventual resume remains continuous.
                self.external_offset = self.paused_beat.map_or(0.0, |frozen| frozen - raw);
            }
            self.external_beat = Some(raw);
            return;
        }

        if let Some(last_raw) = self.external_beat.take() {
            let logical = self.paused_beat.unwrap_or(last_raw + self.external_offset);
            self.anchor_internal_at(logical, now);
        }
        self.external_offset = 0.0;
    }

    pub fn is_external(&self) -> bool {
        self.external_beat.is_some()
    }

    /// Set the internal tempo without moving the current beat position.
    ///
    /// Callers that already sampled a timestamp should use [`Self::set_bpm_at`]
    /// so the tempo change and the surrounding clock work share one instant.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.set_bpm_at(bpm, Instant::now());
    }

    /// Timestamped form of [`Self::set_bpm`]. Changing the slope of an
    /// anchor-based clock must also move its anchor; otherwise elapsed time is
    /// retroactively multiplied by the new BPM and the beat jumps immediately.
    pub fn set_bpm_at(&mut self, bpm: f32, now: Instant) {
        let bpm = finite_or(bpm, self.bpm).clamp(30.0, 300.0);
        if bpm == self.bpm {
            return;
        }

        if self.external_beat.is_none() {
            let beat = self.beat(now);
            self.bpm = bpm;
            self.anchor_internal_at(beat, now);
            return;
        }
        self.bpm = bpm;
    }

    pub fn tap(&mut self, now: Instant) {
        if let Some(&last) = self.taps.last() {
            if now.duration_since(last) > TAP_TIMEOUT {
                self.taps.clear();
            }
        }
        self.taps.push(now);
        if self.taps.len() > MAX_TAPS {
            self.taps.remove(0);
        }
        if self.taps.len() >= 2 {
            let first = self.taps[0];
            let span = now.duration_since(first).as_secs_f64();
            let intervals = (self.taps.len() - 1) as f64;
            let avg = span / intervals;
            if avg > 0.0 {
                self.bpm = ((60.0 / avg) as f32).clamp(30.0, 300.0);
            }
        }
        // Each tap is a downbeat: re-anchor so beat 0 lands on it.
        self.anchor = now;
        if self.paused_beat.is_some() {
            self.paused_beat = Some(0.0);
        }
    }
}

pub struct ModMatrix {
    pub clock: Clock,
    pub lfos: [Lfo; NUM_LFOS],
    pub routings: Vec<Routing>,
    /// Latest sampled LFO values (refreshed by `update`), for UI meters.
    pub lfo_values: [f32; NUM_LFOS],
    /// Beat position at the last update, for the panel's beat light.
    pub current_beat: f64,
    last_update: Option<Instant>,
    /// Latest audio levels (pushed by the app each frame from the analyzer).
    pub audio: AudioLevels,
    /// Whether audio capture should be running (the app syncs the analyzer).
    pub audio_enabled: bool,
    /// Gain applied to normalized audio levels before routing.
    pub audio_gain: f32,
    /// Preferred input device name; empty = system default.
    pub audio_device: String,
    /// `live` captures a CPAL input/system-playback source; `file` analyzes
    /// `audio_clip_path` against the piece-local program clock.
    pub audio_source_kind: String,
    /// Persisted source identity for deterministic, circular file analysis.
    pub audio_clip_path: String,
    /// Verified content reference retained while `audio_clip_path` points at
    /// the resolved host file used by the live decoder. Runtime-only: patch
    /// conversion writes this identity back into `audio_clip_path`.
    pub audio_clip_source_reference: Option<String>,
    /// Validated 3–8-band layout mirrored into `AudioAnalyzer`.
    pub audio_band_config: AudioBandConfig,
    /// Latest MIDI slot values 0..1 (pushed by the app from the MIDI engine).
    pub midi: [f32; NUM_MIDI_SLOTS],
    /// Whether MIDI input should be connected (the app syncs the engine).
    pub midi_enabled: bool,
    /// CC number bound to each slot.
    pub midi_ccs: [u8; NUM_MIDI_SLOTS],
    /// When Some(slot), the next CC message seen binds that slot (MIDI learn).
    pub midi_learn: Option<usize>,
    /// Follow external MIDI timing clock (0xF8) for BPM and beat position.
    pub midi_clock_sync: bool,
    /// Phone orientation [yaw, pitch, roll], each 0..1 (0.5 = level).
    /// Streamed from the web remote; holds the last value received.
    pub gyro: [f32; 3],
    /// Most recent DeviceOrientation degrees [alpha, beta, gamma].
    pub gyro_raw: [f32; 3],
    pub gyro_config: [GyroAxisConfig; 3],
    /// XY performance pad [x, y], each 0..1. Touched from the web remote;
    /// optionally springs toward center after release.
    pub pad: [f32; 2],
    pub pad_active: bool,
    pub pad_config: PadConfig,
}

impl ModMatrix {
    pub fn new() -> Self {
        Self {
            clock: Clock::new(),
            lfos: std::array::from_fn(|_| Lfo::default()),
            routings: Vec::new(),
            lfo_values: [0.0; NUM_LFOS],
            current_beat: 0.0,
            last_update: None,
            audio: AudioLevels::default(),
            audio_enabled: false,
            audio_gain: 1.0,
            audio_device: String::new(),
            audio_source_kind: AUDIO_SOURCE_LIVE.to_string(),
            audio_clip_path: String::new(),
            audio_clip_source_reference: None,
            audio_band_config: AudioBandConfig::default(),
            midi: [0.0; NUM_MIDI_SLOTS],
            midi_enabled: false,
            // CC1 (mod wheel) and the common first knobs on most controllers.
            midi_ccs: [1, 2, 3, 4],
            midi_learn: None,
            midi_clock_sync: false,
            gyro: [0.5; 3],
            gyro_raw: [0.0; 3],
            gyro_config: [
                GyroAxisConfig::with_range(180.0),
                GyroAxisConfig::with_range(90.0),
                GyroAxisConfig::with_range(90.0),
            ],
            pad: [0.5; 2],
            pad_active: false,
            pad_config: PadConfig::default(),
        }
    }

    /// Advance all time-dependent modulation state exactly once per live frame.
    pub fn update(&mut self, now: Instant) {
        let dt = self
            .last_update
            .map(|last| now.saturating_duration_since(last).as_secs_f32())
            .unwrap_or(0.0);
        self.last_update = Some(now);
        self.update_at_beat(self.clock.beat(now), dt);
    }

    /// Forget only the live-frame timestamp used to derive modulation `dt`.
    ///
    /// The next [`Self::update`] then advances with `dt = 0`, preventing time
    /// spent rebuilding a patch from being applied as spring or slew motion.
    /// Beat phase and the most recently published beat remain untouched.
    pub fn reset_update_timing(&mut self) {
        self.last_update = None;
    }

    /// Advance at an explicit beat and time step. The offline exporter passes
    /// `frame_index / fps` and `1 / fps`, making slew and spring motion fully
    /// deterministic and independent of render performance.
    pub fn update_at_beat(&mut self, beat: f64, dt: f32) {
        let dt = finite_or(dt, 0.0).max(0.0);
        self.current_beat = beat;
        for (i, lfo) in self.lfos.iter().enumerate() {
            self.lfo_values[i] = lfo.value(beat, i);
        }

        if self.pad_config.spring_enabled && !self.pad_active {
            let rate = finite_or(self.pad_config.spring_rate, 0.0).max(0.0);
            if rate > 0.0 && dt > 0.0 {
                let alpha = 1.0 - (-rate * dt).exp();
                for value in &mut self.pad {
                    *value += (0.5 - *value) * alpha;
                    *value = (*value).clamp(0.0, 1.0);
                }
            }
        }

        // Source state is independent of routing response caches, so each
        // route can sample and advance in place without a frame-local heap
        // allocation. Consumers below only read the cache; every route still
        // advances exactly once per frame.
        for index in 0..self.routings.len() {
            let (source, curve, curve_amount) = {
                let routing = &self.routings[index];
                (routing.source, routing.curve, routing.curve_amount)
            };
            let desired = shape(self.source_value(source), curve, curve_amount);
            self.routings[index].advance(desired, dt);
        }
    }

    /// Store a DeviceOrientation sample and apply calibration/range/expo.
    pub fn set_gyro_degrees(&mut self, alpha: f32, beta: f32, gamma: f32) {
        self.gyro_raw = [alpha, beta, gamma].map(|v| finite_or(v, 0.0));
        self.recompute_gyro();
    }

    /// Make the current orientation the centered (0.5) position on all axes.
    pub fn calibrate_gyro(&mut self) {
        for (axis, raw) in self.gyro_config.iter_mut().zip(self.gyro_raw) {
            axis.center_degrees = raw;
        }
        self.recompute_gyro();
    }

    /// Release a vanished phone stream without leaving its last pose applied.
    ///
    /// Raw values move to the persisted calibration centers as well as the
    /// normalized outputs moving to 0.5. A later physical sample therefore
    /// resumes from the same calibration instead of changing that contract.
    pub fn recenter_gyro(&mut self) {
        for (raw, config) in self.gyro_raw.iter_mut().zip(self.gyro_config) {
            *raw = finite_or(config.center_degrees, 0.0);
        }
        self.recompute_gyro();
    }

    /// Re-apply gyroscope configuration after a config field changes.
    pub fn recompute_gyro(&mut self) {
        for i in 0..3 {
            let cfg = self.gyro_config[i];
            let mut delta = self.gyro_raw[i] - finite_or(cfg.center_degrees, 0.0);
            if i == 0 {
                // Yaw wraps at 360 degrees; choose the shortest calibrated arc.
                delta = (delta + 180.0).rem_euclid(360.0) - 180.0;
            }
            let range = finite_or(cfg.range_degrees, 90.0).abs().max(0.001);
            let mut centered = (delta / range).clamp(-1.0, 1.0);
            if cfg.invert {
                centered = -centered;
            }
            let exponent = 2.0_f32.powf(finite_or(cfg.expo, 0.0).clamp(-2.0, 2.0));
            centered = centered.signum() * centered.abs().powf(exponent);
            self.gyro[i] = (0.5 + centered * 0.5).clamp(0.0, 1.0);
        }
    }

    pub fn set_pad(&mut self, x: f32, y: f32, active: bool) {
        self.pad = [
            finite_or(x, 0.5).clamp(0.0, 1.0),
            finite_or(y, 0.5).clamp(0.0, 1.0),
        ];
        self.pad_active = active;
    }

    /// Current value of a modulation source.
    pub fn source_value(&self, source: ModSource) -> f32 {
        match source {
            ModSource::Lfo(i) => self.lfo_values[i.min(NUM_LFOS - 1)],
            ModSource::AudioLevel => self.audio.level,
            ModSource::AudioBass => self.audio.bass,
            ModSource::AudioMid => self.audio.mid,
            ModSource::AudioHigh => self.audio.high,
            ModSource::AudioBand(i) => self.audio.bands[i.min(MAX_AUDIO_BANDS - 1)],
            ModSource::AudioOnset => self.audio.onset,
            ModSource::AudioBright => self.audio.bright,
            ModSource::AudioNoise => self.audio.noise,
            ModSource::Midi(i) => self.midi[i.min(NUM_MIDI_SLOTS - 1)],
            ModSource::GyroYaw => finite_or(self.gyro[0], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0,
            ModSource::GyroPitch => finite_or(self.gyro[1], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0,
            ModSource::GyroRoll => finite_or(self.gyro[2], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0,
            ModSource::PadX => self.pad_source_value(0),
            ModSource::PadY => self.pad_source_value(1),
        }
    }

    fn pad_source_value(&self, axis: usize) -> f32 {
        let cfg = self.pad_config.axes[axis.min(1)];
        let centered = finite_or(self.pad[axis.min(1)], 0.5).clamp(0.0, 1.0) * 2.0 - 1.0;
        let value = shape(centered, cfg.curve, cfg.curve_amount).clamp(-1.0, 1.0);
        if cfg.quantize > 1 {
            let intervals = (cfg.quantize - 1) as f32;
            let unit = (value + 1.0) * 0.5;
            ((unit * intervals).round() / intervals) * 2.0 - 1.0
        } else {
            value
        }
    }

    /// Produce modulated copies of the effect, NTSC, and temporal params.
    /// Base values are untouched; each routing adds
    /// `source × depth × half-range`, clamped.
    #[cfg(test)]
    pub fn modulate(
        &self,
        effects: &EffectUniforms,
        transform: &SpatialTransform,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
    ) -> (EffectUniforms, SpatialTransform, NtscParams, TemporalParams) {
        Self::modulate_from_offsets(
            effects,
            transform,
            ntsc,
            temporal,
            &self.accumulate_offsets(0),
        )
    }

    fn modulate_from_offsets(
        effects: &EffectUniforms,
        transform: &SpatialTransform,
        ntsc: &NtscParams,
        temporal: &TemporalParams,
        offsets: &RoutingOffsets,
    ) -> (EffectUniforms, SpatialTransform, NtscParams, TemporalParams) {
        let mut fx = *effects;
        let mut spatial = transform.sanitized();
        let mut np = ntsc.clone();
        let mut tp = *temporal;

        for (index, &(target, min, max)) in TARGETS.iter().enumerate() {
            let offset = offsets.master[index] * (max - min) * 0.5;
            if offset != 0.0 {
                apply_offset(
                    &mut fx,
                    &mut spatial,
                    &mut np,
                    &mut tp,
                    target,
                    offset,
                    (min, max),
                );
            }
        }

        (fx, spatial.sanitized(), np, tp)
    }

    fn modulate_master_motion_from_offsets(
        base: &MotionParams,
        offsets: &RoutingOffsets,
    ) -> MotionParams {
        let mut motion = base.sanitized();
        let offset = |target: &'static str, min: f32, max: f32| {
            master_target_index(target)
                .and_then(|index| offsets.master.get(index))
                .copied()
                .unwrap_or(0.0)
                * (max - min)
                * 0.5
        };
        apply_motion_offsets(&mut motion, offset, false);
        motion.sanitized()
    }

    fn modulate_gesture_canvas_from_offsets(
        base: &crate::gesture_canvas::GestureCanvasParams,
        offsets: &RoutingOffsets,
    ) -> crate::gesture_canvas::GestureCanvasParams {
        let mut canvas = base.sanitized();
        let offset = |target: &'static str, min: f32, max: f32| {
            master_target_index(target)
                .and_then(|index| offsets.master.get(index))
                .copied()
                .unwrap_or(0.0)
                * (max - min)
                * 0.5
        };
        apply_gesture_canvas_offsets(&mut canvas, offset);
        canvas
    }

    fn modulate_layer_motion_from_offsets(
        index: usize,
        base: &MotionParams,
        offsets: &RoutingOffsets,
    ) -> MotionParams {
        let mut motion = base.sanitized();
        let offset = |target: &'static str, min: f32, max: f32| {
            offsets.layer_value(index, target) * (max - min) * 0.5
        };
        apply_motion_offsets(&mut motion, offset, true);
        motion.sanitized()
    }

    fn modulate_layer_from_offsets(
        index: usize,
        base_effects: &EffectUniforms,
        base_transform: &SpatialTransform,
        base_opacity: f32,
        base_speed: f32,
        base_fps: f32,
        offsets: &RoutingOffsets,
    ) -> LayerModulation {
        let mut effects = *base_effects;
        let mut transform = base_transform.sanitized();
        let offset = |suffix: &'static str, min: f32, max: f32| {
            offsets.layer_value(index, suffix) * (max - min) * 0.5
        };
        let opacity = (base_opacity + offset("opacity", 0.0, 1.0)).clamp(0.0, 1.0);
        let speed = (base_speed + offset("speed", 0.25, 4.0)).clamp(0.25, 4.0);
        let fps = (base_fps + offset("fps", 1.0, 240.0)).clamp(1.0, 240.0);

        macro_rules! apply {
            ($field:ident, $suffix:literal, $min:expr, $max:expr) => {
                effects.$field = (effects.$field + offset($suffix, $min, $max)).clamp($min, $max);
            };
        }
        apply!(key_threshold, "key_threshold", 0.0, 1.0);
        for (channel, suffix) in
            effects
                .key_color
                .iter_mut()
                .zip(["key_color_r", "key_color_g", "key_color_b"])
        {
            *channel = (*channel + offset(suffix, 0.0, 1.0)).clamp(0.0, 1.0);
        }
        apply!(key_tolerance, "key_tolerance", 0.0, 1.0);
        apply!(pixelate_size, "pixelate", 1.0, 32.0);
        apply!(rgb_split, "rgb_split", 0.0, 30.0);
        apply!(hue_shift, "hue_shift", -180.0, 180.0);
        apply!(saturation, "saturation", -1.0, 1.0);
        apply!(brightness, "brightness", -1.0, 1.0);
        apply!(contrast, "contrast", -1.0, 1.0);
        apply!(posterize, "posterize", 0.0, 16.0);
        apply!(grain_intensity, "grain_intensity", 0.0, 0.3);
        apply!(grain_size, "grain_size", 1.0, 4.0);
        apply!(vignette, "vignette", 0.0, 1.5);
        apply!(color_drift, "color_drift", 0.0, 0.02);
        apply!(breathe_scale, "breathe_scale", 0.0, 0.05);
        apply!(breathe_rotation, "breathe_rotation", 0.0, 2.0);
        apply!(breathe_position, "breathe_position", 0.0, 0.02);
        apply!(key_softness, "key_softness", 0.0, 0.5);
        apply!(downsample, "downsample", 0.05, 1.0);
        apply!(cellular_amount, "cellular_amount", 0.0, 1.0);
        apply!(cellular_scale, "cellular_scale", 2.0, 32.0);
        apply!(cellular_warp, "cellular_warp", 0.0, 1.0);
        apply!(cellular_speed, "cellular_speed", 0.0, 2.0);
        apply!(cellular_gap_amount, "cellular_gap_amount", 0.0, 1.0);
        apply!(cellular_gap_threshold, "cellular_gap_threshold", 0.0, 1.0);
        apply!(cellular_gap_softness, "cellular_gap_softness", 0.0, 0.5);
        apply!(shift_amount, "shift_amount", 0.0, 1.0);
        apply!(shift_block_size, "shift_block_size", 2.0, 256.0);
        apply!(shift_density, "shift_density", 0.0, 1.0);
        apply!(shift_speed, "shift_speed", 0.0, 20.0);

        apply_spatial_offsets(&mut transform, offset);

        LayerModulation {
            opacity,
            speed,
            fps,
            effects,
            transform: transform.sanitized(),
        }
    }

    /// Modulate a complete stack from one O(routes) accumulator pass. This is
    /// the live/export hot-path API; the single-layer method remains as a
    /// compatibility convenience for patch tooling and focused tests.
    /// Build one immutable modulation sample for an actual live/export stack.
    ///
    /// Layer storage is sized only from this trusted caller-supplied count.
    /// A persisted or network-authored target with an enormous layer number is
    /// therefore ignored by bounds checks instead of driving an allocation.
    pub fn frame(&self, layer_count: usize) -> ModulationFrame {
        ModulationFrame {
            offsets: self.accumulate_offsets(layer_count),
        }
    }

    /// Accumulate stable rack/group routes into the caller's compact topology
    /// address book. A persisted route whose scope, node, or parameter is
    /// currently missing remains stored in `routings` but contributes zero.
    pub fn stable_frame(&self, book: &StableModAddressBook) -> StableModulationFrame {
        let mut offsets = vec![0.0; book.len()];
        for routing in &self.routings {
            let CompiledTarget::Stable(target) = routing.compiled_target else {
                continue;
            };
            let Some(address) = book.address(target) else {
                continue;
            };
            let [min, max] = book.target(address).unwrap_or(target).range();
            let amount = routing.cached_value() * finite_or(routing.depth, 0.0) * (max - min) * 0.5;
            if let Some(slot) = offsets.get_mut(address.index()) {
                *slot += amount;
            }
        }
        StableModulationFrame { offsets }
    }

    /// Modulate one layer. Batch callers should prefer [`Self::modulate_layers`]
    /// so all layer destinations share one routing accumulator pass.
    #[cfg(test)]
    pub fn modulate_layer_full(
        &self,
        index: usize,
        base_effects: &EffectUniforms,
        base_transform: &SpatialTransform,
        base_opacity: f32,
        base_speed: f32,
        base_fps: f32,
    ) -> LayerModulation {
        // This test/tooling convenience must not size storage from the target
        // index: a valid parsed target may be as large as `usize::MAX - 1`.
        // Remap only the requested layer into one trusted local slot.
        let mut offsets = RoutingOffsets::new(1);
        for routing in &self.routings {
            let CompiledTarget::Layer {
                index: target_index,
                suffix,
            } = routing.compiled_target
            else {
                continue;
            };
            if target_index == index {
                let amount = routing.cached_value() * finite_or(routing.depth, 0.0);
                if amount != 0.0 {
                    offsets.add(CompiledTarget::Layer { index: 0, suffix }, amount);
                }
            }
        }
        Self::modulate_layer_from_offsets(
            0,
            base_effects,
            base_transform,
            base_opacity,
            base_speed,
            base_fps,
            &offsets,
        )
    }

    /// Summed modulation offset for one named target — for values the app
    /// applies itself (e.g. the morph crossfader) rather than via
    /// `modulate`. Uses the same depth × half-range scaling.
    #[cfg(test)]
    pub fn target_offset(&self, target: &str, layer_count: usize) -> f32 {
        let Some((min, max)) = target_range(target) else {
            return 0.0;
        };
        let compiled = compile_target(target);
        self.accumulate_offsets(layer_count).value(compiled) * (max - min) * 0.5
    }

    fn accumulate_offsets(&self, layer_count: usize) -> RoutingOffsets {
        let mut offsets = RoutingOffsets::new(layer_count);
        for routing in &self.routings {
            let amount = routing.cached_value() * finite_or(routing.depth, 0.0);
            if amount != 0.0 {
                offsets.add(routing.compiled_target, amount);
            }
        }
        offsets
    }

    pub fn add_routing(&mut self) {
        if self.routings.len() < MAX_ROUTINGS {
            self.routings
                .push(Routing::new(ModSource::Lfo(0), "rgb_split", 0.0));
        }
    }

    pub fn remove_routing(&mut self, index: usize) {
        if index < self.routings.len() {
            self.routings.remove(index);
        }
    }

    /// Keep positional layer targets attached to the same logical layer when
    /// one layer is removed. Routes for the removed layer are discarded.
    pub fn remap_layer_targets_after_remove(&mut self, removed: usize) {
        self.routings.retain_mut(|routing| {
            let Some((layer, suffix)) = parse_layer_target(&routing.target) else {
                return true;
            };
            if layer == removed {
                return false;
            }
            let remapped = if layer > removed { layer - 1 } else { layer };
            let _ = routing.set_target(format!("layer{}_{suffix}", remapped + 1));
            true
        });
    }

    /// Tombstone typed targets owned by the removed stable layer, then apply
    /// the established positional-target permutation. A newly inserted layer
    /// can therefore never inherit a deleted typed destination, while legacy
    /// `layerN_*` routes retain their historical remove/remap behavior.
    pub fn remap_layer_targets_after_remove_with_stable_id(
        &mut self,
        removed: usize,
        removed_layer_id: StableLayerId,
    ) {
        let saved_position = u32::try_from(removed)
            .ok()
            .and_then(SavedLayerPosition::new);
        self.routings.retain_mut(|routing| {
            let Some(StableModTarget::Node {
                scope: StableModScope::Layer(layer_id),
                node_id,
                parameter,
            }) = routing.stable_target()
            else {
                return true;
            };
            if layer_id != removed_layer_id {
                return true;
            }
            let Some(saved_position) = saved_position else {
                // The persisted layer-position domain is deliberately
                // bounded. Such a route cannot be represented truthfully and
                // follows the legacy removed-owner law rather than remaining
                // live with a dangling process identity.
                return false;
            };
            routing.set_missing_target(SavedStableModTarget::MissingSavedLayer {
                saved_position,
                node_id,
                parameter,
            });
            true
        });
        self.remap_layer_targets_after_remove(removed);
    }

    /// Tombstone every typed destination owned by an explicitly deleted
    /// group. Group IDs are monotonic and never reused, but retaining the
    /// missing identity is still essential for truthful patch diagnostics and
    /// prevents any future import/editor path from rebinding by root order.
    pub fn tombstone_group_targets_after_remove(&mut self, removed_group_id: GroupId) {
        for routing in &mut self.routings {
            let missing = match routing.stable_target() {
                Some(StableModTarget::Node {
                    scope: StableModScope::Group(group_id),
                    node_id,
                    parameter,
                }) if group_id == removed_group_id => Some(SavedStableModTarget::MissingGroup {
                    group_id,
                    missing_target: SavedMissingTarget::Node { node_id, parameter },
                }),
                Some(StableModTarget::GroupValue {
                    group_id,
                    parameter,
                }) if group_id == removed_group_id => Some(SavedStableModTarget::MissingGroup {
                    group_id,
                    missing_target: SavedMissingTarget::GroupValue { parameter },
                }),
                _ => None,
            };
            if let Some(missing) = missing {
                routing.set_missing_target(missing);
            }
        }
    }

    /// Apply the same stable permutation as moving an element in a Vec.
    pub fn remap_layer_targets_after_move(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        for routing in &mut self.routings {
            let Some((layer, suffix)) = parse_layer_target(&routing.target) else {
                continue;
            };
            let remapped = if layer == from {
                to
            } else if from < to && layer > from && layer <= to {
                layer - 1
            } else if to < from && layer >= to && layer < from {
                layer + 1
            } else {
                layer
            };
            let _ = routing.set_target(format!("layer{}_{suffix}", remapped + 1));
        }
    }

    /// Reset LFOs and routings; tempo is left alone (losing a dialed-in
    /// BPM mid-set would be crueler than any stale routing).
    pub fn reset(&mut self) {
        self.lfos = std::array::from_fn(|_| Lfo::default());
        self.routings.clear();
    }
}

fn parse_layer_target(target: &str) -> Option<(usize, &str)> {
    let rest = target.strip_prefix("layer")?;
    let (number, suffix) = rest.split_once('_')?;
    let one_based = number.parse::<usize>().ok()?;
    (one_based > 0).then_some((one_based - 1, suffix))
}

/// Source-depth sums, before multiplication by each destination's half-range.
/// Master storage is fixed; layer storage is sized from the actual consumer
/// stack and rebuilt once per batch so routing edits need no invalidation
/// protocol. Untrusted target indices are never used as allocation sizes.
#[derive(Clone)]
struct RoutingOffsets {
    master: [f32; TARGETS.len()],
    layer: Vec<[f32; LAYER_TARGET_SUFFIXES.len()]>,
}

const LAYER_TARGET_SUFFIXES: &[&str] = &[
    "opacity",
    "speed",
    "fps",
    "key_threshold",
    "key_color_r",
    "key_color_g",
    "key_color_b",
    "key_tolerance",
    "pixelate",
    "rgb_split",
    "hue_shift",
    "saturation",
    "brightness",
    "contrast",
    "posterize",
    "grain_intensity",
    "grain_size",
    "vignette",
    "color_drift",
    "breathe_scale",
    "breathe_rotation",
    "breathe_position",
    "cellular_amount",
    "cellular_scale",
    "cellular_warp",
    "cellular_speed",
    "cellular_gap_amount",
    "cellular_gap_threshold",
    "cellular_gap_softness",
    "key_softness",
    "downsample",
    // Appended so every established layer suffix retains its compiled index.
    "shift_amount",
    "shift_block_size",
    "shift_density",
    "shift_speed",
    // Appended so all established compiled suffix indices remain stable.
    "position_x",
    "position_y",
    "scale_x",
    "scale_y",
    "anchor_x",
    "anchor_y",
    "rotation_deg",
    "skew_deg",
    "skew_axis_deg",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
    // Appended so every established compiled suffix index remains stable.
    "motion_transplant_amount",
    "motion_confidence_threshold",
    "motion_confidence_softness",
    "motion_refresh",
    "motion_decay",
    "motion_occlusion",
    "motion_shutter_angle",
    "motion_shutter_phase",
    "motion_shutter_curvature",
    "motion_shutter_chromatic_lag",
];

impl RoutingOffsets {
    fn new(layer_count: usize) -> Self {
        Self {
            master: [0.0; TARGETS.len()],
            layer: vec![[0.0; LAYER_TARGET_SUFFIXES.len()]; layer_count],
        }
    }

    fn add(&mut self, target: CompiledTarget, amount: f32) {
        match target {
            CompiledTarget::Master(index) => {
                if let Some(slot) = self.master.get_mut(index) {
                    *slot += amount;
                }
            }
            CompiledTarget::Layer { index, suffix } => {
                if let Some(slot) = self
                    .layer
                    .get_mut(index)
                    .and_then(|values| values.get_mut(suffix))
                {
                    *slot += amount;
                }
            }
            CompiledTarget::Invalid => {}
            CompiledTarget::Stable(_) => {}
        }
    }

    #[cfg(test)]
    fn value(&self, target: CompiledTarget) -> f32 {
        match target {
            CompiledTarget::Master(index) => self.master.get(index).copied().unwrap_or(0.0),
            CompiledTarget::Layer { index, suffix } => self
                .layer
                .get(index)
                .and_then(|values| values.get(suffix))
                .copied()
                .unwrap_or(0.0),
            CompiledTarget::Invalid => 0.0,
            CompiledTarget::Stable(_) => 0.0,
        }
    }

    fn layer_value(&self, layer: usize, suffix: &str) -> f32 {
        let Some(suffix) = layer_suffix_index(suffix) else {
            return 0.0;
        };
        self.layer
            .get(layer)
            .and_then(|values| values.get(suffix))
            .copied()
            .unwrap_or(0.0)
    }
}

fn layer_suffix_index(suffix: &str) -> Option<usize> {
    Some(match suffix {
        "opacity" => 0,
        "speed" => 1,
        "fps" => 2,
        "key_threshold" => 3,
        "key_color_r" => 4,
        "key_color_g" => 5,
        "key_color_b" => 6,
        "key_tolerance" => 7,
        "pixelate" => 8,
        "rgb_split" => 9,
        "hue_shift" => 10,
        "saturation" => 11,
        "brightness" => 12,
        "contrast" => 13,
        "posterize" => 14,
        "grain_intensity" => 15,
        "grain_size" => 16,
        "vignette" => 17,
        "color_drift" => 18,
        "breathe_scale" => 19,
        "breathe_rotation" => 20,
        "breathe_position" => 21,
        "cellular_amount" => 22,
        "cellular_scale" => 23,
        "cellular_warp" => 24,
        "cellular_speed" => 25,
        "cellular_gap_amount" => 26,
        "cellular_gap_threshold" => 27,
        "cellular_gap_softness" => 28,
        "key_softness" => 29,
        "downsample" => 30,
        "shift_amount" => 31,
        "shift_block_size" => 32,
        "shift_density" => 33,
        "shift_speed" => 34,
        "position_x" => 35,
        "position_y" => 36,
        "scale_x" => 37,
        "scale_y" => 38,
        "anchor_x" => 39,
        "anchor_y" => 40,
        "rotation_deg" => 41,
        "skew_deg" => 42,
        "skew_axis_deg" => 43,
        "crop_left" => 44,
        "crop_top" => 45,
        "crop_right" => 46,
        "crop_bottom" => 47,
        "motion_transplant_amount" => 48,
        "motion_confidence_threshold" => 49,
        "motion_confidence_softness" => 50,
        "motion_refresh" => 51,
        "motion_decay" => 52,
        "motion_occlusion" => 53,
        "motion_shutter_angle" => 54,
        "motion_shutter_phase" => 55,
        "motion_shutter_curvature" => 56,
        "motion_shutter_chromatic_lag" => 57,
        _ => return None,
    })
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite_f64_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Apply every continuous spatial destination, then let the caller sanitize
/// the transform once so paired crop limits are resolved atomically.
fn apply_spatial_offsets(
    transform: &mut SpatialTransform,
    mut offset: impl FnMut(&'static str, f32, f32) -> f32,
) {
    transform.position[0] += offset("position_x", POSITION_MIN, POSITION_MAX);
    transform.position[1] += offset("position_y", POSITION_MIN, POSITION_MAX);
    transform.scale[0] += offset("scale_x", SCALE_MIN, SCALE_MAX);
    transform.scale[1] += offset("scale_y", SCALE_MIN, SCALE_MAX);
    transform.anchor[0] += offset("anchor_x", ANCHOR_MIN, ANCHOR_MAX);
    transform.anchor[1] += offset("anchor_y", ANCHOR_MIN, ANCHOR_MAX);
    transform.rotation_deg += offset("rotation_deg", -180.0, 180.0);
    transform.skew_deg += offset("skew_deg", -SKEW_LIMIT_DEGREES, SKEW_LIMIT_DEGREES);
    transform.skew_axis_deg += offset("skew_axis_deg", -180.0, 180.0);
    transform.crop[0] += offset("crop_left", 0.0, CROP_MAX);
    transform.crop[1] += offset("crop_top", 0.0, CROP_MAX);
    transform.crop[2] += offset("crop_right", 0.0, CROP_MAX);
    transform.crop[3] += offset("crop_bottom", 0.0, CROP_MAX);
    *transform = transform.sanitized();
}

/// Apply only the bounded numeric Motion law. The caller chooses whether the
/// scope is a layer recipient; master Faraday values are never routable.
fn apply_motion_offsets(
    motion: &mut MotionParams,
    mut offset: impl FnMut(&'static str, f32, f32) -> f32,
    include_faraday: bool,
) {
    if include_faraday {
        motion.transplant.amount += offset("motion_transplant_amount", 0.0, 1.0);
        motion.transplant.confidence_threshold += offset("motion_confidence_threshold", 0.0, 1.0);
        motion.transplant.confidence_softness += offset("motion_confidence_softness", 0.0, 0.5);
        motion.transplant.refresh += offset("motion_refresh", 0.0, 1.0);
        motion.transplant.decay += offset("motion_decay", 0.0, 1.0);
        motion.transplant.occlusion += offset("motion_occlusion", 0.0, 1.0);
    }
    motion.shutter.angle_degrees += offset("motion_shutter_angle", 0.0, 360.0);
    motion.shutter.phase += offset("motion_shutter_phase", -1.0, 1.0);
    motion.shutter.curvature += offset("motion_shutter_curvature", -2.0, 2.0);
    motion.shutter.chromatic_lag += offset("motion_shutter_chromatic_lag", 0.0, 1.0);
    *motion = motion.sanitized();
}

fn apply_spatial_offset(
    transform: &mut SpatialTransform,
    target: &str,
    offset: f32,
    min: f32,
    max: f32,
) -> bool {
    let slot = match target {
        "position_x" => &mut transform.position[0],
        "position_y" => &mut transform.position[1],
        "scale_x" => &mut transform.scale[0],
        "scale_y" => &mut transform.scale[1],
        "anchor_x" => &mut transform.anchor[0],
        "anchor_y" => &mut transform.anchor[1],
        "skew_deg" => &mut transform.skew_deg,
        "crop_left" => &mut transform.crop[0],
        "crop_top" => &mut transform.crop[1],
        "crop_right" => &mut transform.crop[2],
        "crop_bottom" => &mut transform.crop[3],
        "rotation_deg" => {
            transform.rotation_deg += offset;
            return true;
        }
        "skew_axis_deg" => {
            transform.skew_axis_deg += offset;
            return true;
        }
        _ => return false,
    };
    *slot = (*slot + offset).clamp(min, max);
    true
}

/// Shape a signed value without ever discarding its polarity.
pub fn shape(value: f32, curve: Curve, amount: f32) -> f32 {
    let value = finite_or(value, 0.0).clamp(-1.0, 1.0);
    let sign = value.signum();
    let magnitude = value.abs();
    let amount = finite_or(amount, 0.0).clamp(-2.0, 2.0);
    let shaped = match curve {
        Curve::Linear => magnitude,
        Curve::Exp => magnitude.powf(2.0_f32.powf(amount)),
        Curve::Log => magnitude.powf(2.0_f32.powf(-amount)),
        Curve::SCurve => magnitude * magnitude * (3.0 - 2.0 * magnitude),
        Curve::Steps => {
            // -2..+2 maps to 2, 4, 8, 16, 32 equal increments.
            let steps = 2.0_f32.powf(amount + 3.0).round().clamp(2.0, 32.0);
            (magnitude * steps).floor() / steps
        }
    };
    sign * shaped.clamp(0.0, 1.0)
}

fn exponential_follow(current: f32, desired: f32, dt: f32, tau: f32) -> f32 {
    let dt = finite_or(dt, 0.0).max(0.0);
    let tau = finite_or(tau, 0.0).max(0.0);
    if tau <= f32::EPSILON {
        return desired;
    }
    let alpha = 1.0 - (-dt / tau).exp();
    current + (desired - current) * alpha
}

fn apply_offset(
    fx: &mut EffectUniforms,
    spatial: &mut SpatialTransform,
    np: &mut NtscParams,
    tp: &mut TemporalParams,
    target: &str,
    offset: f32,
    range: (f32, f32),
) {
    let (min, max) = range;
    if apply_spatial_offset(spatial, target, offset, min, max) {
        return;
    }
    let slot: &mut f32 = match target {
        "pixelate" => &mut fx.pixelate_size,
        "rgb_split" => &mut fx.rgb_split,
        "hue_shift" => &mut fx.hue_shift,
        "saturation" => &mut fx.saturation,
        "brightness" => &mut fx.brightness,
        "contrast" => &mut fx.contrast,
        "posterize" => &mut fx.posterize,
        "grain_intensity" => &mut fx.grain_intensity,
        "grain_size" => &mut fx.grain_size,
        "vignette" => &mut fx.vignette,
        "color_drift" => &mut fx.color_drift,
        "downsample" => &mut fx.downsample,
        "breathe_scale" => &mut fx.breathe_scale,
        "breathe_rotation" => &mut fx.breathe_rotation,
        "breathe_position" => &mut fx.breathe_position,
        "key_threshold" => &mut fx.key_threshold,
        "key_softness" => &mut fx.key_softness,
        "key_color_r" => &mut fx.key_color[0],
        "key_color_g" => &mut fx.key_color[1],
        "key_color_b" => &mut fx.key_color[2],
        "key_tolerance" => &mut fx.key_tolerance,
        "cellular_amount" => &mut fx.cellular_amount,
        "cellular_scale" => &mut fx.cellular_scale,
        "cellular_warp" => &mut fx.cellular_warp,
        "cellular_speed" => &mut fx.cellular_speed,
        "cellular_gap_amount" => &mut fx.cellular_gap_amount,
        "cellular_gap_threshold" => &mut fx.cellular_gap_threshold,
        "cellular_gap_softness" => &mut fx.cellular_gap_softness,
        "shift_amount" => &mut fx.shift_amount,
        "shift_block_size" => &mut fx.shift_block_size,
        "shift_density" => &mut fx.shift_density,
        "shift_speed" => &mut fx.shift_speed,
        "ntsc_snow" => &mut np.snow_intensity,
        "ntsc_tracking_snow" => &mut np.tracking_noise_snow,
        "ntsc_edge_wave" => &mut np.edge_wave_intensity,
        "ntsc_edge_wave_speed" => &mut np.edge_wave_speed,
        "ntsc_head_shift" => &mut np.head_switching_shift,
        "ntsc_tracking_wave" => &mut np.tracking_noise_wave,
        "ntsc_chroma_loss" => &mut np.chroma_loss,
        "ntsc_composite_noise" => &mut np.composite_noise_intensity,
        "ntsc_luma_noise" => &mut np.luma_noise_intensity,
        "ntsc_chroma_noise" => &mut np.chroma_noise_intensity,
        "ntsc_luma_smear" => &mut np.luma_smear,
        "ntsc_sharpening" => &mut np.composite_sharpening,
        "temporal_feedback" => &mut tp.feedback,
        "temporal_slitscan" => &mut tp.slitscan,
        "temporal_fb_zoom" => &mut tp.fb_zoom,
        "temporal_fb_rotate" => &mut tp.fb_rotate,
        "temporal_slit_angle" => &mut tp.slit_angle,
        "temporal_key_threshold" => &mut tp.key_threshold,
        "temporal_key_softness" => &mut tp.key_softness,
        "temporal_key_history" => &mut tp.key_history,
        "temporal_loom_amount" => &mut tp.originals.loom.amount,
        "temporal_loom_depth" => &mut tp.originals.loom.depth,
        "temporal_loom_phase" => &mut tp.originals.loom.phase,
        "temporal_loom_scale" => &mut tp.originals.loom.scale,
        "temporal_loom_angle" => &mut tp.originals.loom.angle,
        "temporal_atlas_amount" => &mut tp.originals.atlas.amount,
        "temporal_atlas_collision" => &mut tp.originals.atlas.collision,
        "temporal_garden_amount" => &mut tp.originals.garden.amount,
        "temporal_garden_threshold" => &mut tp.originals.garden.threshold,
        "temporal_garden_softness" => &mut tp.originals.garden.softness,
        "temporal_garden_decay" => &mut tp.originals.garden.decay,
        _ => return,
    };
    *slot = (*slot + offset).clamp(min, max);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn sample_hold(seed: u32) -> Lfo {
        Lfo {
            shape: LfoShape::SampleHold,
            beats: 4.0,
            phase: 0.0,
            seed,
        }
    }

    #[test]
    fn sample_hold_seed_zero_matches_the_legacy_golden_sequence() {
        let lfo = sample_hold(0);
        let observed = [
            lfo.value(0.0, 0).to_bits(),
            lfo.value(4.0, 0).to_bits(),
            lfo.value(8.0, 0).to_bits(),
            lfo.value(0.0, 2).to_bits(),
        ];
        assert_eq!(
            observed,
            [0x3f7e_a360, 0x3e14_9bd0, 0xbf45_a34b, 0x3f7b_ea1f]
        );
    }

    #[test]
    fn sample_hold_is_constant_within_a_cycle_and_changes_at_the_boundary() {
        let lfo = sample_hold(0);
        let held = lfo.value(0.0, 0);
        for beat in [0.25, 1.0, 2.5, 3.999] {
            assert_eq!(lfo.value(beat, 0).to_bits(), held.to_bits());
        }
        assert_ne!(lfo.value(4.0, 0).to_bits(), held.to_bits());
    }

    #[test]
    fn sample_hold_seed_selects_an_independent_deterministic_sequence() {
        let legacy = sample_hold(0);
        let seeded = sample_hold(42);
        let first = seeded.value(0.0, 0);

        assert_eq!(first.to_bits(), 0x3f12_db40);
        assert_eq!(seeded.value(0.0, 0).to_bits(), first.to_bits());
        assert_ne!(first.to_bits(), legacy.value(0.0, 0).to_bits());
        assert_ne!(
            seeded.value(4.0, 0).to_bits(),
            legacy.value(4.0, 0).to_bits()
        );
    }

    #[test]
    fn manual_bpm_changes_preserve_beat_phase_and_use_the_new_rate() {
        for new_bpm in [60.0, 300.0] {
            let mut clock = Clock::new();
            let downbeat = Instant::now();
            clock.tap(downbeat);
            let change = downbeat + Duration::from_millis(1_950);
            let beat_before = clock.beat(change);
            assert!((beat_before - 3.9).abs() < 1e-9);

            clock.set_bpm_at(new_bpm, change);

            let beat_after = clock.beat(change);
            assert!(
                (beat_after - beat_before).abs() < 1e-8,
                "{new_bpm} BPM moved beat {beat_before} to {beat_after}"
            );
            let one_second_later = clock.beat(change + Duration::from_secs(1));
            let expected = beat_before + new_bpm as f64 / 60.0;
            assert!((one_second_later - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn internal_clock_pause_is_idempotent_and_resumes_without_catch_up() {
        let mut clock = Clock::new();
        let downbeat = Instant::now();
        clock.tap(downbeat);
        let pause = downbeat + Duration::from_secs(2);
        clock.set_paused(true, pause);
        let frozen = clock.beat(pause);
        assert!((frozen - 4.0).abs() < 1e-9);
        assert!(clock.is_paused());

        clock.set_paused(true, pause + Duration::from_secs(20));
        assert_eq!(clock.beat(pause + Duration::from_secs(20)), frozen);

        let resume = pause + Duration::from_secs(20);
        clock.set_paused(false, resume);
        assert!(!clock.is_paused());
        assert!((clock.beat(resume) - frozen).abs() < 1e-9);
        assert!((clock.beat(resume + Duration::from_millis(500)) - (frozen + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn external_clock_telemetry_advances_under_pause_without_moving_program_phase() {
        let mut clock = Clock::new();
        let start = Instant::now();
        clock.set_external_beat(Some(10.0), start);
        clock.set_paused(true, start);
        assert_eq!(clock.beat(start), 10.0);

        // Hardware continues to publish its absolute transport while the
        // visual program is frozen.
        clock.set_external_beat(Some(42.0), start + Duration::from_secs(8));
        assert_eq!(clock.beat(start + Duration::from_secs(8)), 10.0);

        let resume = start + Duration::from_secs(8);
        clock.set_paused(false, resume);
        assert_eq!(clock.beat(resume), 10.0);
        clock.set_external_beat(Some(42.5), resume + Duration::from_millis(250));
        assert_eq!(clock.beat(resume + Duration::from_millis(250)), 10.5);

        // Falling back to the internal clock also preserves that logical
        // position rather than exposing the raw external count.
        let handoff = resume + Duration::from_millis(250);
        clock.set_external_beat(None, handoff);
        assert!((clock.beat(handoff) - 10.5).abs() < 1e-9);
    }

    #[test]
    fn curves_preserve_sign_and_endpoints() {
        for curve in [
            Curve::Linear,
            Curve::Exp,
            Curve::Log,
            Curve::SCurve,
            Curve::Steps,
        ] {
            approx(shape(0.0, curve, 1.0), 0.0);
            approx(shape(1.0, curve, 1.0), 1.0);
            approx(shape(-1.0, curve, 1.0), -1.0);
            assert!(shape(0.4, curve, 1.0) >= 0.0);
            assert!(shape(-0.4, curve, 1.0) <= 0.0);
        }
        assert!(shape(0.5, Curve::Exp, 1.0) < 0.5);
        assert!(shape(0.5, Curve::Log, 1.0) > 0.5);
        approx(shape(0.5, Curve::SCurve, 0.0), 0.5);
    }

    #[test]
    fn configurable_audio_band_sources_roundtrip_and_legacy_aliases_hold() {
        let mut matrix = ModMatrix::new();
        matrix.audio.bands = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88];
        matrix.audio.bass = matrix.audio.bands[0];
        matrix.audio.mid = matrix.audio.bands[1];
        matrix.audio.high = matrix.audio.bands[2];

        for index in 0..MAX_AUDIO_BANDS {
            let source = ModSource::AudioBand(index);
            assert_eq!(ModSource::try_from_str(source.as_str()), Some(source));
            approx(matrix.source_value(source), matrix.audio.bands[index]);
        }
        approx(
            matrix.source_value(ModSource::AudioBass),
            matrix.source_value(ModSource::AudioBand(0)),
        );
        approx(
            matrix.source_value(ModSource::AudioMid),
            matrix.source_value(ModSource::AudioBand(1)),
        );
        approx(
            matrix.source_value(ModSource::AudioHigh),
            matrix.source_value(ModSource::AudioBand(2)),
        );
        assert_eq!(ModSource::try_from_str("audio_band9"), None);
    }

    #[test]
    fn exponential_slew_uses_distinct_attack_and_release() {
        let mut matrix = ModMatrix::new();
        let mut route = Routing::new(ModSource::Midi(0), "brightness", 1.0);
        route.attack = 1.0;
        route.release = 2.0;
        matrix.routings.push(route);

        matrix.midi[0] = 1.0;
        matrix.update_at_beat(0.0, 1.0);
        let attacked = 1.0 - (-1.0_f32).exp();
        approx(matrix.routings[0].cached, attacked);

        matrix.midi[0] = 0.0;
        matrix.update_at_beat(0.0, 2.0);
        approx(matrix.routings[0].cached, attacked * (-1.0_f32).exp());

        let one_step = exponential_follow(0.0, 1.0, 1.0, 0.7);
        let mut ten_steps = 0.0;
        for _ in 0..10 {
            ten_steps = exponential_follow(ten_steps, 1.0, 0.1, 0.7);
        }
        approx(one_step, ten_steps);
    }

    #[test]
    fn update_timing_reset_zeroes_one_delta_without_reanchoring_beat() {
        let mut matrix = ModMatrix::new();
        let downbeat = Instant::now();
        matrix.clock.tap(downbeat);
        matrix.pad_config.spring_enabled = true;
        matrix.pad_config.spring_rate = 4.0;
        matrix.set_pad(1.0, 0.0, false);
        matrix.update(downbeat + Duration::from_millis(250));

        let published_beat = matrix.current_beat;
        let future = downbeat + Duration::from_secs(10);
        let future_beat = matrix.clock.beat(future);
        matrix.set_pad(1.0, 0.0, false);

        matrix.reset_update_timing();

        assert_eq!(matrix.current_beat, published_beat);
        assert_eq!(matrix.clock.beat(future), future_beat);
        matrix.update(future);
        assert_eq!(matrix.pad, [1.0, 0.0]);

        matrix.update(future + Duration::from_millis(250));
        assert!(matrix.pad[0] < 1.0);
        assert!(matrix.pad[1] > 0.0);
    }

    #[test]
    fn offsets_sum_before_clamp_and_bases_are_immutable() {
        fn render(reversed: bool) -> (f32, f32) {
            let mut matrix = ModMatrix::new();
            matrix.midi[0] = 1.0;
            let positive = Routing::new(ModSource::Midi(0), "brightness", 1.0);
            let negative = Routing::new(ModSource::Midi(0), "brightness", -1.0);
            matrix.routings = if reversed {
                vec![negative, positive]
            } else {
                vec![positive, negative]
            };
            matrix.update_at_beat(0.0, 1.0 / 30.0);
            let base = EffectUniforms {
                brightness: 0.9,
                ..Default::default()
            };
            let (modulated, _, _, _) = matrix.modulate(
                &base,
                &SpatialTransform::default(),
                &NtscParams::default(),
                &TemporalParams::default(),
            );
            (base.brightness, modulated.brightness)
        }

        let forward = render(false);
        let reverse = render(true);
        approx(forward.0, 0.9);
        approx(reverse.0, 0.9);
        approx(forward.1, 0.9);
        approx(reverse.1, forward.1);
    }

    #[test]
    fn consumers_are_pure_and_routing_lifecycle_keeps_state_aligned() {
        let mut matrix = ModMatrix::new();
        matrix.midi = [0.25, 0.75, 0.0, 0.0];
        let mut first = Routing::new(ModSource::Midi(0), "brightness", 1.0);
        first.attack = 1.0;
        let mut second = Routing::new(ModSource::Midi(1), "contrast", 1.0);
        second.attack = 1.0;
        matrix.routings = vec![first, second];
        matrix.update_at_beat(0.0, 0.5);
        let cached = matrix.routings[1].cached;

        let _ = matrix.target_offset("contrast", 0);
        let _ = matrix.modulate(
            &EffectUniforms::default(),
            &SpatialTransform::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        let _ = matrix.modulate_layer_full(
            0,
            &EffectUniforms::default(),
            &SpatialTransform::default(),
            1.0,
            1.0,
            30.0,
        );
        approx(matrix.routings[1].cached, cached);

        matrix.remove_routing(0);
        approx(matrix.routings[0].cached, cached);
        matrix.add_routing();
        approx(matrix.routings[1].cached, 0.0);
        matrix.routings[0].reset_runtime();
        approx(matrix.routings[0].cached, 0.0);
    }

    #[test]
    fn compiled_targets_and_batched_layers_match_single_layer_semantics() {
        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 0.8;
        matrix.routings = vec![
            Routing::new(ModSource::Midi(0), "brightness", 0.5),
            Routing::new(ModSource::Midi(0), "layer1_opacity", -0.25),
            Routing::new(ModSource::Midi(0), "layer2_cellular_gap_softness", 0.75),
            Routing::new(ModSource::Midi(0), "layer2_fps", 0.2),
        ];
        matrix.update_at_beat(0.0, 0.0);
        let first = EffectUniforms::default();
        let second = EffectUniforms::default();
        let spatial = SpatialTransform::default();
        let expected_first = matrix.modulate_layer_full(0, &first, &spatial, 0.9, 1.0, 30.0);
        let expected_second = matrix.modulate_layer_full(1, &second, &spatial, 0.7, 1.5, 24.0);

        let batched = matrix.frame(2).modulate_layers([
            (&first, &spatial, 0.9, 1.0, 30.0),
            (&second, &spatial, 0.7, 1.5, 24.0),
        ]);

        approx(batched[0].opacity, expected_first.opacity);
        approx(
            batched[0].effects.brightness,
            expected_first.effects.brightness,
        );
        approx(batched[1].fps, expected_second.fps);
        approx(
            batched[1].effects.cellular_gap_softness,
            expected_second.effects.cellular_gap_softness,
        );
        let frame = matrix.frame(2);
        let cached = frame.modulate_layers([
            (&first, &spatial, 0.9, 1.0, 30.0),
            (&second, &spatial, 0.7, 1.5, 24.0),
        ]);
        approx(cached[0].opacity, batched[0].opacity);
        approx(cached[1].fps, batched[1].fps);
        let (master, _, _, _) = frame.modulate(
            &EffectUniforms::default(),
            &SpatialTransform::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(master.brightness, 0.4);
        assert_eq!(
            matrix.routings[2].compiled_target,
            compile_target("layer2_cellular_gap_softness")
        );
    }

    #[test]
    fn target_change_resets_signal_but_response_time_change_preserves_it() {
        let mut route = Routing::new(ModSource::Midi(0), "brightness", 1.0);
        route.attack = 1.0;
        route.advance(1.0, 1.0);
        let live = route.cached_value();
        assert!(live > 0.0);

        route.attack = 2.0;
        route.release = 3.0;
        approx(route.cached_value(), live);
        assert!(!route.set_target("brightness"));
        approx(route.cached_value(), live);
        assert!(route.set_target("layer1_brightness"));
        approx(route.cached_value(), 0.0);
        assert_eq!(route.compiled_target, compile_target("layer1_brightness"));

        // Target identity and its compiled destination change atomically.
        route.state = 1.0;
        route.cached = 1.0;
        assert!(route.set_target("contrast"));
        route.state = 1.0;
        route.cached = 1.0;
        let mut matrix = ModMatrix::new();
        matrix.routings.push(route);
        let frame = matrix.frame(0);
        let (effects, _, _, _) = frame.modulate(
            &EffectUniforms::default(),
            &SpatialTransform::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(effects.brightness, 0.0);
        approx(effects.contrast, 1.0);
    }

    #[test]
    fn gyro_calibration_wrap_and_invert_are_stable() {
        let mut matrix = ModMatrix::new();
        matrix.set_gyro_degrees(359.0, 10.0, -20.0);
        matrix.calibrate_gyro();
        for value in matrix.gyro {
            approx(value, 0.5);
        }

        matrix.set_gyro_degrees(1.0, 20.0, -20.0);
        assert!(matrix.gyro[0] > 0.5, "yaw must cross 360 on shortest arc");
        assert!(matrix.gyro[1] > 0.5);
        matrix.gyro_config[1].invert = true;
        matrix.recompute_gyro();
        assert!(matrix.gyro[1] < 0.5);
    }

    #[test]
    fn gyro_recenter_releases_last_pose_without_losing_calibration() {
        let mut matrix = ModMatrix::new();
        matrix.set_gyro_degrees(270.0, 25.0, -30.0);
        matrix.calibrate_gyro();
        let centers = matrix.gyro_raw;
        matrix.set_gyro_degrees(320.0, 60.0, 20.0);
        assert_ne!(matrix.gyro, [0.5; 3]);

        matrix.recenter_gyro();

        assert_eq!(matrix.gyro_raw, centers);
        assert_eq!(matrix.gyro, [0.5; 3]);
    }

    #[test]
    fn pad_quantize_and_spring_are_deterministic() {
        let mut matrix = ModMatrix::new();
        matrix.pad_config.axes[0].quantize = 4;
        matrix.set_pad(0.74, 0.9, true);
        approx(matrix.source_value(ModSource::PadX), 1.0 / 3.0);

        // N means exactly N evenly spaced positions, inclusive of endpoints,
        // with nearest-position snapping and symmetric midpoint behavior.
        let samples = [0.0, 0.16, 0.34, 0.5, 0.66, 0.84, 1.0];
        let quantized: Vec<f32> = samples
            .into_iter()
            .map(|value| {
                matrix.set_pad(value, 0.5, true);
                matrix.source_value(ModSource::PadX)
            })
            .collect();
        for (actual, expected) in
            quantized
                .into_iter()
                .zip([-1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0])
        {
            approx(actual, expected);
        }

        matrix.pad_config.spring_enabled = true;
        matrix.pad_config.spring_rate = 4.0;
        matrix.set_pad(0.74, 0.9, false);
        matrix.update_at_beat(0.0, 0.25);
        approx(matrix.pad[0], 0.5 + 0.24 * (-1.0_f32).exp());
        approx(matrix.pad[1], 0.5 + 0.4 * (-1.0_f32).exp());

        let mut one = ModMatrix::new();
        one.pad_config.spring_enabled = true;
        one.pad_config.spring_rate = 4.0;
        one.set_pad(1.0, 0.0, false);
        one.update_at_beat(0.0, 1.0);
        let mut ten = ModMatrix::new();
        ten.pad_config.spring_enabled = true;
        ten.pad_config.spring_rate = 4.0;
        ten.set_pad(1.0, 0.0, false);
        for frame in 0..10 {
            ten.update_at_beat(frame as f64, 0.1);
        }
        approx(one.pad[0], ten.pad[0]);
        approx(one.pad[1], ten.pad[1]);
    }

    #[test]
    fn lfo_phase_setter_and_sampler_reject_nonfinite_values() {
        let mut lfo = Lfo::default();
        lfo.set_phase(1.25);
        approx(lfo.phase, 0.25);
        lfo.set_phase(-0.25);
        approx(lfo.phase, 0.75);

        for phase in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            lfo.set_phase(phase);
            approx(lfo.phase, 0.0);
            assert!(lfo.value(3.0, 0).is_finite());

            // Direct field writes remain safe for compatibility with existing
            // internal callers while they migrate to `set_phase`.
            lfo.phase = phase;
            assert!(lfo.value(3.0, 0).is_finite());
        }
    }

    #[test]
    fn arbitrary_positive_layer_targets_are_bounded_by_the_actual_frame_size() {
        assert_eq!(target_range("layer16_brightness"), Some((-1.0, 1.0)));
        assert_eq!(target_range("layer16_downsample"), Some((0.05, 1.0)));
        assert_eq!(target_range("layer17_opacity"), Some((0.0, 1.0)));
        assert_eq!(target_range("layer257_downsample"), Some((0.05, 1.0)));
        assert_eq!(target_range("layer0_speed"), None);
        assert_eq!(target_range("layer16_unknown"), None);

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings = vec![
            Routing::new(ModSource::Midi(0), "layer17_brightness", 1.0),
            Routing::new(
                ModSource::Midi(0),
                format!("layer{}_brightness", usize::MAX),
                1.0,
            ),
        ];
        matrix.update_at_beat(0.0, 0.0);
        let base = EffectUniforms::default();

        let frame = matrix.frame(17);
        assert_eq!(frame.offsets.layer.len(), 17);
        let spatial = SpatialTransform::default();
        let modulated = ModMatrix::modulate_layer_from_offsets(
            16,
            &base,
            &spatial,
            1.0,
            1.0,
            30.0,
            &frame.offsets,
        );
        approx(modulated.effects.brightness, 1.0);

        // A forged, parseable destination does not influence allocation and
        // is simply ignored when it lies outside the caller's actual stack.
        let one_layer_frame = matrix.frame(1);
        assert_eq!(one_layer_frame.offsets.layer.len(), 1);
        let first = one_layer_frame
            .modulate_layers([(&base, &spatial, 1.0, 1.0, 30.0)])
            .remove(0);
        approx(first.effects.brightness, 0.0);

        // The single-layer test/tooling adapter also uses one local slot,
        // even when asked to inspect the largest representable parsed index.
        let huge = matrix.modulate_layer_full(usize::MAX - 1, &base, &spatial, 1.0, 1.0, 30.0);
        approx(huge.effects.brightness, 1.0);
        approx(base.brightness, 0.0);
    }

    #[test]
    fn cellular_targets_modulate_master_and_layers_without_mutating_bases() {
        for (target, range) in [
            ("cellular_amount", (0.0, 1.0)),
            ("cellular_scale", (2.0, 32.0)),
            ("cellular_warp", (0.0, 1.0)),
            ("cellular_speed", (0.0, 2.0)),
            ("cellular_gap_amount", (0.0, 1.0)),
            ("cellular_gap_threshold", (0.0, 1.0)),
            ("cellular_gap_softness", (0.0, 0.5)),
            ("layer16_cellular_amount", (0.0, 1.0)),
            ("layer16_cellular_scale", (2.0, 32.0)),
            ("layer16_cellular_warp", (0.0, 1.0)),
            ("layer16_cellular_speed", (0.0, 2.0)),
            ("layer16_cellular_gap_amount", (0.0, 1.0)),
            ("layer16_cellular_gap_threshold", (0.0, 1.0)),
            ("layer16_cellular_gap_softness", (0.0, 0.5)),
        ] {
            assert_eq!(target_range(target), Some(range));
        }

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        for target in [
            "cellular_amount",
            "cellular_scale",
            "cellular_warp",
            "cellular_speed",
            "cellular_gap_amount",
            "cellular_gap_threshold",
            "cellular_gap_softness",
            "layer1_cellular_amount",
            "layer1_cellular_scale",
            "layer1_cellular_warp",
            "layer1_cellular_speed",
            "layer1_cellular_gap_amount",
            "layer1_cellular_gap_threshold",
            "layer1_cellular_gap_softness",
        ] {
            matrix
                .routings
                .push(Routing::new(ModSource::Midi(0), target, 1.0));
        }
        matrix.update_at_beat(0.0, 0.0);

        let base = EffectUniforms::default();
        let (master, _, _, _) = matrix.modulate(
            &base,
            &SpatialTransform::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(master.cellular_amount, 0.5);
        approx(master.cellular_scale, 25.0);
        approx(master.cellular_warp, 0.85);
        approx(master.cellular_speed, 1.25);
        approx(master.cellular_gap_amount, 0.5);
        approx(master.cellular_gap_threshold, 1.0);
        approx(master.cellular_gap_softness, 0.33);

        let layer =
            matrix.modulate_layer_full(0, &base, &SpatialTransform::default(), 1.0, 1.0, 30.0);
        approx(layer.effects.cellular_amount, 0.5);
        approx(layer.effects.cellular_scale, 25.0);
        approx(layer.effects.cellular_warp, 0.85);
        approx(layer.effects.cellular_speed, 1.25);
        approx(layer.effects.cellular_gap_amount, 0.5);
        approx(layer.effects.cellular_gap_threshold, 1.0);
        approx(layer.effects.cellular_gap_softness, 0.33);

        approx(base.cellular_amount, 0.0);
        approx(base.cellular_scale, 10.0);
        approx(base.cellular_warp, 0.35);
        approx(base.cellular_speed, 0.25);
        approx(base.cellular_gap_amount, 0.0);
        approx(base.cellular_gap_threshold, 0.65);
        approx(base.cellular_gap_softness, 0.08);
    }

    #[test]
    fn shift_targets_modulate_master_and_arbitrary_layers_without_mutating_bases() {
        for (target, range) in [
            ("shift_amount", (0.0, 1.0)),
            ("shift_block_size", (2.0, 256.0)),
            ("shift_density", (0.0, 1.0)),
            ("shift_speed", (0.0, 20.0)),
            ("layer17_shift_amount", (0.0, 1.0)),
            ("layer17_shift_block_size", (2.0, 256.0)),
            ("layer17_shift_density", (0.0, 1.0)),
            ("layer17_shift_speed", (0.0, 20.0)),
        ] {
            assert_eq!(target_range(target), Some(range));
        }

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        for target in [
            "shift_amount",
            "shift_block_size",
            "shift_density",
            "shift_speed",
            "layer17_shift_amount",
            "layer17_shift_block_size",
            "layer17_shift_density",
            "layer17_shift_speed",
        ] {
            matrix
                .routings
                .push(Routing::new(ModSource::Midi(0), target, 1.0));
        }
        matrix.update_at_beat(0.0, 0.0);

        let base = EffectUniforms::default();
        let (master, _, _, _) = matrix.modulate(
            &base,
            &SpatialTransform::default(),
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(master.shift_amount, 0.5);
        approx(master.shift_block_size, 135.0);
        approx(master.shift_density, 1.0);
        approx(master.shift_speed, 13.0);

        let frame = matrix.frame(17);
        let layer = ModMatrix::modulate_layer_from_offsets(
            16,
            &base,
            &SpatialTransform::default(),
            1.0,
            1.0,
            30.0,
            &frame.offsets,
        );
        approx(layer.effects.shift_amount, 0.5);
        approx(layer.effects.shift_block_size, 135.0);
        approx(layer.effects.shift_density, 1.0);
        approx(layer.effects.shift_speed, 13.0);

        approx(base.shift_amount, 0.0);
        approx(base.shift_block_size, 8.0);
        approx(base.shift_density, 0.5);
        approx(base.shift_speed, 3.0);
    }

    #[test]
    fn centered_performance_sources_are_bipolar_without_changing_telemetry() {
        let mut matrix = ModMatrix::new();
        matrix.gyro = [0.5, 0.25, 0.75];
        matrix.set_pad(0.5, 0.25, true);
        assert_eq!(matrix.gyro, [0.5, 0.25, 0.75]);
        assert_eq!(matrix.pad, [0.5, 0.25]);
        approx(matrix.source_value(ModSource::GyroYaw), 0.0);
        approx(matrix.source_value(ModSource::GyroPitch), -0.5);
        approx(matrix.source_value(ModSource::GyroRoll), 0.5);
        approx(matrix.source_value(ModSource::PadX), 0.0);
        approx(matrix.source_value(ModSource::PadY), -0.5);
    }

    #[test]
    fn layer_route_targets_follow_move_and_remove_permutations_by_identity() {
        let mut matrix = ModMatrix::new();
        matrix.routings = vec![
            Routing::new(ModSource::Lfo(0), "layer1_brightness", 0.1),
            Routing::new(ModSource::Lfo(1), "layer2_key", 0.2),
            Routing::new(ModSource::Lfo(2), "layer3_opacity", 0.3),
            Routing::new(ModSource::Lfo(3), "morph", 0.4),
        ];
        let ids: Vec<u64> = matrix.routings.iter().map(Routing::route_id).collect();
        assert_eq!(matrix.routings[1].target, "layer2_key_threshold");

        matrix.remap_layer_targets_after_move(0, 2);
        assert_eq!(matrix.routings[0].target, "layer3_brightness");
        assert_eq!(matrix.routings[1].target, "layer1_key_threshold");
        assert_eq!(matrix.routings[2].target, "layer2_opacity");
        assert_eq!(matrix.routings[3].target, "morph");
        assert_eq!(
            matrix
                .routings
                .iter()
                .map(Routing::route_id)
                .collect::<Vec<_>>(),
            ids
        );

        matrix.remap_layer_targets_after_remove(1);
        assert_eq!(matrix.routings.len(), 3);
        assert_eq!(matrix.routings[0].target, "layer2_brightness");
        assert_eq!(matrix.routings[1].target, "layer1_key_threshold");
        assert_eq!(matrix.routings[2].target, "morph");
        assert_eq!(matrix.routings[0].route_id(), ids[0]);
        assert_eq!(matrix.routings[1].route_id(), ids[1]);
        assert_eq!(matrix.routings[2].route_id(), ids[3]);
    }

    #[test]
    fn stable_layer_removal_tombstones_typed_owner_without_changing_legacy_law() {
        let removed_layer_id = StableLayerId::new(41).unwrap();
        let surviving_layer_id = StableLayerId::new(42).unwrap();
        let removed_target = StableModTarget::parse("node/layer/41/3/wet").unwrap();
        let surviving_target = StableModTarget::parse("node/layer/42/4/wet").unwrap();

        let mut removed = Routing::new(ModSource::Lfo(1), removed_target.to_string(), 0.37);
        removed.curve = Curve::SCurve;
        removed.curve_amount = -0.65;
        removed.attack = 1.25;
        removed.release = 2.5;
        removed.state = 0.4;
        removed.cached = -0.2;
        let removed_route_id = removed.route_id();

        let surviving = Routing::new(ModSource::Lfo(2), surviving_target.to_string(), -0.2);
        let surviving_route_id = surviving.route_id();
        let legacy_removed = Routing::new(ModSource::Lfo(3), "layer2_opacity", 0.1);
        let legacy_after = Routing::new(ModSource::AudioBass, "layer3_brightness", 0.3);
        let legacy_after_id = legacy_after.route_id();

        let mut matrix = ModMatrix::new();
        matrix.routings = vec![removed, surviving, legacy_removed, legacy_after];
        matrix.remap_layer_targets_after_remove_with_stable_id(1, removed_layer_id);

        assert_eq!(
            matrix.routings.len(),
            3,
            "legacy removed-owner route is dropped"
        );
        let tombstone = &matrix.routings[0];
        let expected = SavedStableModTarget::MissingSavedLayer {
            saved_position: SavedLayerPosition::new(1).unwrap(),
            node_id: NodeId::new(3).unwrap(),
            parameter: StableNodeParameter::Wet,
        };
        assert_eq!(tombstone.route_id(), removed_route_id);
        assert_eq!(tombstone.source, ModSource::Lfo(1));
        assert_eq!(tombstone.depth, 0.37);
        assert_eq!(tombstone.curve, Curve::SCurve);
        assert_eq!(tombstone.curve_amount, -0.65);
        assert_eq!(tombstone.attack, 1.25);
        assert_eq!(tombstone.release, 2.5);
        assert_eq!(tombstone.state, 0.0);
        assert_eq!(tombstone.cached, 0.0);
        assert_eq!(tombstone.target(), expected.persistence_key());
        assert_eq!(tombstone.stable_target(), None);
        assert_eq!(tombstone.saved_missing_target(), Some(expected));
        assert!(matches!(
            expected.resolve(
                &StableModAddressBook::default(),
                |_| Some(surviving_layer_id),
                |_| true,
            ),
            ResolvedStableModTarget::Missing(value) if value == expected
        ));

        assert_eq!(matrix.routings[1].route_id(), surviving_route_id);
        assert_eq!(matrix.routings[1].stable_target(), Some(surviving_target));
        assert_eq!(matrix.routings[1].saved_missing_target(), None);
        assert_eq!(matrix.routings[2].route_id(), legacy_after_id);
        assert_eq!(matrix.routings[2].target(), "layer2_brightness");
    }

    #[test]
    fn expanded_continuous_targets_include_key_temporal_ntsc_and_layer_fps() {
        for (target, range) in [
            ("key_color_r", (0.0, 1.0)),
            ("key_tolerance", (0.0, 1.0)),
            ("ntsc_edge_wave_speed", (0.0, 10.0)),
            ("ntsc_tracking_wave", (0.0, 50.0)),
            ("ntsc_composite_noise", (0.0, 0.5)),
            ("ntsc_chroma_noise", (0.0, 0.5)),
            ("ntsc_luma_smear", (0.0, 1.0)),
            ("ntsc_sharpening", (-1.0, 2.0)),
            ("temporal_key_threshold", (0.0, 1.0)),
            ("temporal_key_softness", (0.0, 0.5)),
            ("temporal_key_history", (1.0, 23.0)),
            ("temporal_loom_amount", (0.0, 1.0)),
            ("temporal_loom_depth", (0.0, 1.0)),
            ("temporal_loom_phase", (-1_000.0, 1_000.0)),
            ("temporal_loom_scale", (0.01, 100.0)),
            ("temporal_loom_angle", (-180.0, 180.0)),
            ("temporal_atlas_amount", (0.0, 1.0)),
            ("temporal_atlas_collision", (0.0, 1.0)),
            ("temporal_garden_amount", (0.0, 1.0)),
            ("temporal_garden_threshold", (0.0, 1.0)),
            ("temporal_garden_softness", (0.0, 0.5)),
            ("temporal_garden_decay", (0.0, 1.0)),
            ("layer1_fps", (1.0, 240.0)),
            ("layer1_key", (0.0, 1.0)),
            ("layer1_key_threshold", (0.0, 1.0)),
        ] {
            assert_eq!(target_range(target), Some(range), "{target}");
        }

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix
            .routings
            .push(Routing::new(ModSource::Midi(0), "layer1_fps", 1.0));
        matrix.update_at_beat(0.0, 1.0 / 30.0);
        let base = EffectUniforms::default();
        let layer =
            matrix.modulate_layer_full(0, &base, &SpatialTransform::default(), 1.0, 1.0, 30.0);
        approx(layer.fps, 149.5);
        approx(base.key_threshold, EffectUniforms::default().key_threshold);
    }

    #[test]
    fn motion_modulation_is_bounded_continuous_and_preserves_discrete_laws() {
        use crate::motion::{
            CurvedShutterParams, CurvedShutterQuality, FaradayParams, MotionCarrier, MotionDonor,
            MotionFieldSource, MotionLatticeQuality, MOTION_ALGORITHM_VERSION,
        };

        for (target, range) in [
            ("motion_shutter_angle", (0.0, 360.0)),
            ("motion_shutter_phase", (-1.0, 1.0)),
            ("motion_shutter_curvature", (-2.0, 2.0)),
            ("motion_shutter_chromatic_lag", (0.0, 1.0)),
            ("layer1_motion_transplant_amount", (0.0, 1.0)),
            ("layer1_motion_confidence_softness", (0.0, 0.5)),
            ("layer1_motion_shutter_angle", (0.0, 360.0)),
        ] {
            assert_eq!(target_range(target), Some(range), "{target}");
        }
        for discrete in [
            "motion_algorithm_version",
            "motion_field_source",
            "motion_lattice_quality",
            "layer1_motion_donor",
            "layer1_motion_carrier",
            "layer1_motion_shutter_quality",
        ] {
            assert_eq!(target_range(discrete), None, "{discrete}");
        }

        let layer_id = StableLayerId::new(91).unwrap();
        let saved_position = SavedLayerPosition::new(2).unwrap();
        let base = MotionParams {
            field_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::High,
            transplant: FaradayParams {
                amount: 0.25,
                donor: MotionDonor::Selected {
                    layer_id,
                    saved_position,
                },
                carrier: MotionCarrier::FirstSourceFrame,
                confidence_softness: 0.1,
                ..FaradayParams::default()
            },
            shutter: CurvedShutterParams {
                angle_degrees: 20.0,
                quality: CurvedShutterQuality::High,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings = [
            "motion_shutter_angle",
            "motion_shutter_phase",
            "layer1_motion_transplant_amount",
            "layer1_motion_confidence_softness",
            "layer1_motion_shutter_angle",
        ]
        .into_iter()
        .map(|target| Routing::new(ModSource::Midi(0), target, 1.0))
        .collect();
        matrix.update_at_beat(0.0, 0.0);
        let frame = matrix.frame(1);
        let master = frame.modulate_motion(&base);
        let layer = frame.modulate_layer_motion(0, &base);

        approx(master.shutter.angle_degrees, 200.0);
        approx(master.shutter.phase, 1.0);
        approx(master.transplant.amount, 0.25);
        approx(layer.transplant.amount, 0.75);
        approx(layer.transplant.confidence_softness, 0.35);
        approx(layer.shutter.angle_degrees, 200.0);
        assert_eq!(layer.algorithm_version, MOTION_ALGORITHM_VERSION);
        assert_eq!(layer.field_source, MotionFieldSource::CodecVectors);
        assert_eq!(layer.lattice_quality, MotionLatticeQuality::High);
        assert_eq!(layer.transplant.donor, base.transplant.donor);
        assert_eq!(layer.transplant.carrier, MotionCarrier::FirstSourceFrame);
        assert_eq!(layer.shutter.quality, CurvedShutterQuality::High);
        assert_eq!(base.transplant.amount, 0.25, "base state is immutable");
        assert_eq!(base.shutter.angle_degrees, 20.0);
    }

    #[test]
    fn temporal_originals_modulation_changes_only_continuous_bounded_values() {
        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings = [
            "temporal_loom_amount",
            "temporal_loom_depth",
            "temporal_loom_phase",
            "temporal_loom_scale",
            "temporal_loom_angle",
            "temporal_atlas_amount",
            "temporal_atlas_collision",
            "temporal_garden_amount",
            "temporal_garden_threshold",
            "temporal_garden_softness",
            "temporal_garden_decay",
        ]
        .into_iter()
        .map(|target| Routing::new(ModSource::Midi(0), target, 1.0))
        .collect();
        matrix.update_at_beat(0.0, 0.0);

        let mut base = TemporalParams::default();
        base.originals.loom.depth = 0.25;
        base.originals.loom.topology = crate::temporal::TemporalTopology::Radial;
        base.originals.loom.interpolation = crate::temporal::TemporalInterpolation::Linear;
        base.originals.loom.folds = 7;
        base.originals.loom.quantization = 8;
        base.originals.atlas.seed = 0xdead_beef;
        base.originals.atlas.territories = 9;
        base.originals.garden.gate = crate::temporal::RefreshGardenGate::Matte;
        base.originals.garden.decay = 0.2;
        base.originals.garden.max_hold_ticks = 47;
        base.originals.garden.matte_route =
            crate::temporal::RefreshGardenMatteRoute::SelectedLayer {
                layer_id: StableLayerId::new(91).unwrap(),
                saved_position: SavedLayerPosition::new(3).unwrap(),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            };
        base.originals.garden.motion_route =
            crate::temporal::RefreshGardenMotionRoute::MissingSelectedLayer {
                saved_position: SavedLayerPosition::new(5).unwrap(),
            };
        base.originals.score.enabled = true;
        base.originals.score.seed = 0x1234_5678;
        base.originals.score.state_count = 11;
        base.originals.score.trigger = crate::temporal::CollisionScoreTrigger::Manual;
        base.originals.score.loop_driver =
            crate::temporal::CollisionScoreLoopDriver::SelectedLayer {
                layer_id: StableLayerId::new(91).unwrap(),
                saved_position: SavedLayerPosition::new(3).unwrap(),
            };
        base.originals.reset.loop_boundary = crate::temporal::TemporalEventResetMode::Memory;
        base.originals.reset.downbeat = crate::temporal::TemporalEventResetMode::Score;

        let (_, _, _, modulated) = matrix.frame(0).modulate(
            &EffectUniforms::default(),
            &SpatialTransform::default(),
            &NtscParams::default(),
            &base,
        );
        approx(modulated.originals.loom.amount, 0.5);
        approx(modulated.originals.loom.depth, 0.75);
        approx(modulated.originals.loom.phase, 1_000.0);
        approx(modulated.originals.loom.scale, 50.995);
        approx(modulated.originals.loom.angle, 180.0);
        approx(modulated.originals.atlas.amount, 0.5);
        approx(modulated.originals.atlas.collision, 0.5);
        approx(modulated.originals.garden.amount, 0.5);
        approx(modulated.originals.garden.threshold, 0.6);
        approx(modulated.originals.garden.softness, 0.28);
        approx(modulated.originals.garden.decay, 0.7);

        assert_eq!(
            modulated.originals.loom.topology,
            base.originals.loom.topology
        );
        assert_eq!(
            modulated.originals.loom.interpolation,
            base.originals.loom.interpolation
        );
        assert_eq!(modulated.originals.loom.folds, base.originals.loom.folds);
        assert_eq!(
            modulated.originals.loom.quantization,
            base.originals.loom.quantization
        );
        assert_eq!(modulated.originals.atlas.seed, base.originals.atlas.seed);
        assert_eq!(
            modulated.originals.atlas.territories,
            base.originals.atlas.territories
        );
        assert_eq!(modulated.originals.garden.gate, base.originals.garden.gate);
        assert_eq!(
            modulated.originals.garden.matte_route,
            base.originals.garden.matte_route
        );
        assert_eq!(
            modulated.originals.garden.motion_route,
            base.originals.garden.motion_route
        );
        assert_eq!(
            modulated.originals.garden.max_hold_ticks,
            base.originals.garden.max_hold_ticks
        );
        assert_eq!(modulated.originals.score, base.originals.score);
        assert_eq!(modulated.originals.reset, base.originals.reset);
        approx(base.originals.loom.amount, 0.0);
        approx(base.originals.garden.decay, 0.2);

        for discrete in [
            "temporal_loom_topology",
            "temporal_atlas_seed",
            "temporal_garden_gate",
            "temporal_score_seed",
            "temporal_score_state_count",
        ] {
            assert_eq!(
                target_range(discrete),
                None,
                "{discrete} must stay authored"
            );
        }
    }

    #[test]
    fn spatial_targets_modulate_master_and_arbitrary_layers_without_mutating_bases() {
        for (target, range) in [
            ("position_x", (POSITION_MIN, POSITION_MAX)),
            ("scale_y", (SCALE_MIN, SCALE_MAX)),
            ("anchor_x", (ANCHOR_MIN, ANCHOR_MAX)),
            ("rotation_deg", (-180.0, 180.0)),
            ("skew_deg", (-SKEW_LIMIT_DEGREES, SKEW_LIMIT_DEGREES)),
            ("crop_left", (0.0, CROP_MAX)),
            ("layer17_position_x", (POSITION_MIN, POSITION_MAX)),
            ("layer17_scale_x", (SCALE_MIN, SCALE_MAX)),
            ("layer17_crop_right", (0.0, CROP_MAX)),
        ] {
            assert_eq!(target_range(target), Some(range), "{target}");
        }

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings = vec![
            Routing::new(ModSource::Midi(0), "position_x", 0.25),
            Routing::new(ModSource::Midi(0), "rotation_deg", 0.2),
            Routing::new(ModSource::Midi(0), "crop_left", 0.4),
            Routing::new(ModSource::Midi(0), "layer17_scale_x", 0.1),
            Routing::new(ModSource::Midi(0), "layer17_rotation_deg", 0.2),
            Routing::new(ModSource::Midi(0), "layer17_crop_right", 0.4),
        ];
        matrix.update_at_beat(0.0, 0.0);
        let base = SpatialTransform {
            position: [0.1, 0.2],
            rotation_deg: 170.0,
            fit: crate::spatial::FitMode::Fill,
            edge: crate::spatial::EdgeMode::Repeat,
            sampling: crate::spatial::SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let frame = matrix.frame(17);
        let (_, master, _, _) = frame.modulate(
            &EffectUniforms::default(),
            &base,
            &NtscParams::default(),
            &TemporalParams::default(),
        );
        approx(master.position[0], 1.1);
        approx(master.position[1], 0.2);
        approx(master.rotation_deg, -154.0);
        approx(master.crop[0], CROP_MAX * 0.2);
        assert_eq!(master.fit, base.fit);
        assert_eq!(master.edge, base.edge);
        assert_eq!(master.sampling, base.sampling);

        let layer = ModMatrix::modulate_layer_from_offsets(
            16,
            &EffectUniforms::default(),
            &base,
            1.0,
            1.0,
            30.0,
            &frame.offsets,
        );
        approx(layer.transform.scale[0], 2.6);
        approx(layer.transform.rotation_deg, -154.0);
        approx(layer.transform.crop[2], CROP_MAX * 0.2);
        assert_eq!(layer.transform.fit, base.fit);
        assert_eq!(layer.transform.edge, base.edge);
        assert_eq!(layer.transform.sampling, base.sampling);
        assert_eq!(base.position, [0.1, 0.2]);
        assert_eq!(base.rotation_deg, 170.0);
        assert_eq!(base.crop, [0.0; 4]);
    }

    #[test]
    fn stable_node_application_covers_every_registered_component() {
        use crate::visual_rack::{
            ImageMatte, RuntimeImageMatte, RuntimeMaskParams, RuntimeVisualNode,
            RuntimeVisualNodeKind,
        };

        let kinds = [
            RuntimeVisualNodeKind::Transform(SpatialTransform::default()),
            RuntimeVisualNodeKind::DigitalColor(crate::visual_rack::DigitalColorParams::default()),
            RuntimeVisualNodeKind::Key(crate::visual_rack::KeyParams::default()),
            RuntimeVisualNodeKind::Cellular(crate::visual_rack::CellularParams::default()),
            RuntimeVisualNodeKind::Shift(crate::visual_rack::ShiftParams::default()),
            RuntimeVisualNodeKind::Grain(crate::visual_rack::GrainParams::default()),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(
                crate::visual_rack::RectangleMask::default(),
            )),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(
                crate::visual_rack::EllipseMask::default(),
            )),
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte::resolve_routes(ImageMatte::default(), &mut |_| None, &|_| false),
            )),
            RuntimeVisualNodeKind::Displace(crate::visual_rack::RuntimeDisplaceParams::default()),
            RuntimeVisualNodeKind::Residual(crate::visual_rack::RuntimeResidualParams::default()),
        ];
        let empty_composition =
            RuntimeComposition::try_from_parts(Vec::new(), Vec::new(), Some(1), 0.5).unwrap();
        for (index, kind) in kinds.into_iter().enumerate() {
            let node_id = NodeId::new(3 + index as u64).unwrap();
            let rack = RuntimeVisualRack::try_from_parts(
                vec![RuntimeVisualNode::authored(node_id, kind)],
                Some(node_id.get() + 1),
            )
            .unwrap();
            let book =
                StableModAddressBook::from_composition(&rack, &[], &empty_composition).unwrap();
            assert!(!book.is_empty());
            for target in book.targets.iter().copied() {
                let StableModTarget::Node {
                    node_id: target_node,
                    parameter,
                    ..
                } = target
                else {
                    continue;
                };
                assert_eq!(target_node, node_id);
                let original = *rack.get(node_id).unwrap();
                let mut changed = original;
                apply_stable_node_offset(&mut changed, parameter, 0.123);
                if changed == original {
                    apply_stable_node_offset(&mut changed, parameter, -0.123);
                }
                assert_ne!(
                    changed, original,
                    "registered stable target {target} did not reach its value"
                );
                assert_eq!(changed.stable_id, original.stable_id);
                assert_eq!(changed.enabled, original.enabled);
                assert_eq!(changed.blend, original.blend);
            }
        }
    }

    #[test]
    fn stable_address_and_application_support_the_257th_layer_and_reorder() {
        use crate::visual_rack::{RuntimeVisualNode, RuntimeVisualNodeKind, ShiftParams};

        let node_id = NodeId::new(3).unwrap();
        let target_layer = StableLayerId::new(257).unwrap();
        let target_rack = RuntimeVisualRack::try_from_parts(
            vec![RuntimeVisualNode::authored(
                node_id,
                RuntimeVisualNodeKind::Shift(ShiftParams::default()),
            )],
            Some(4),
        )
        .unwrap();
        let mut layers: Vec<_> = (1..=257)
            .map(|id| {
                let layer_id = StableLayerId::new(id).unwrap();
                let rack = if layer_id == target_layer {
                    target_rack.clone()
                } else {
                    RuntimeVisualRack::empty()
                };
                (layer_id, rack)
            })
            .collect();
        let master = RuntimeVisualRack::empty();
        let composition =
            RuntimeComposition::try_from_parts(Vec::new(), Vec::new(), Some(1), 0.5).unwrap();
        let target = StableModTarget::parse("node/layer/257/3/amount").unwrap();

        let book = StableModAddressBook::from_composition(&master, &layers, &composition).unwrap();
        let mut frame = StableModulationFrame {
            offsets: vec![0.0; book.len()],
        };
        frame.offsets[book.address(target).unwrap().index()] = 0.25;
        let authored = layers.clone();
        let mut modulated_master = master.clone();
        let mut modulated_composition = composition.clone();
        apply_stable_modulation(
            &book,
            &frame,
            &mut modulated_master,
            &mut layers,
            &mut modulated_composition,
        );
        let amount = |racks: &[(StableLayerId, RuntimeVisualRack)]| {
            let node = racks
                .iter()
                .find(|(layer_id, _)| *layer_id == target_layer)
                .unwrap()
                .1
                .get(node_id)
                .unwrap();
            let RuntimeVisualNodeKind::Shift(value) = node.kind else {
                panic!("shift topology changed");
            };
            value.amount
        };
        approx(amount(&layers), 0.25);
        approx(amount(&authored), 0.0);

        layers.reverse();
        let reordered_book =
            StableModAddressBook::from_composition(&master, &layers, &composition).unwrap();
        let mut reordered_frame = StableModulationFrame {
            offsets: vec![0.0; reordered_book.len()],
        };
        reordered_frame.offsets[reordered_book.address(target).unwrap().index()] = 0.2;
        apply_stable_modulation(
            &reordered_book,
            &reordered_frame,
            &mut modulated_master,
            &mut layers,
            &mut modulated_composition,
        );
        approx(amount(&layers), 0.45);

        let removed = layers
            .iter_mut()
            .find(|(layer_id, _)| *layer_id == target_layer)
            .unwrap()
            .1
            .remove(node_id);
        assert!(removed.is_some());
        let before_missing = layers.clone();
        apply_stable_modulation(
            &reordered_book,
            &reordered_frame,
            &mut modulated_master,
            &mut layers,
            &mut modulated_composition,
        );
        assert_eq!(layers, before_missing, "deleted stable node must be inert");
    }

    fn runtime_group_with_matte(
        id: u64,
        matte: Option<crate::visual_rack::RuntimeImageMatte>,
    ) -> crate::composition::RuntimeGroup {
        use crate::composition::{BusAssignment, GroupName, RuntimeGroup, RuntimeGroupMembers};

        RuntimeGroup {
            id: GroupId::new(id).unwrap(),
            name: GroupName::new(format!("Group {id}")).unwrap(),
            members: RuntimeGroupMembers::try_from_vec(Vec::new()).unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        }
    }

    fn default_runtime_matte() -> crate::visual_rack::RuntimeImageMatte {
        crate::visual_rack::RuntimeImageMatte::resolve_routes(
            crate::visual_rack::ImageMatte::default(),
            &mut |_| None,
            &|_| false,
        )
    }

    #[test]
    fn group_matte_and_bus_targets_are_bounded_stable_and_inert_when_missing() {
        use crate::composition::RuntimeRootItem;
        use crate::visual_rack::MatteChannel;

        let first_id = GroupId::new(7).unwrap();
        let target_id = GroupId::new(9).unwrap();
        let mut target_matte = default_runtime_matte();
        target_matte.channel = MatteChannel::Red;
        target_matte.invert = true;
        target_matte.amount = 0.2;
        target_matte.threshold = 0.6;
        target_matte.softness = 0.1;
        let first = runtime_group_with_matte(first_id.get(), Some(default_runtime_matte()));
        let target_group = runtime_group_with_matte(target_id.get(), Some(target_matte));
        let composition = RuntimeComposition::try_from_parts(
            vec![first.clone(), target_group.clone()],
            vec![
                RuntimeRootItem::Group { group_id: first_id },
                RuntimeRootItem::Group {
                    group_id: target_id,
                },
            ],
            Some(10),
            0.25,
        )
        .unwrap();
        let master = RuntimeVisualRack::empty();
        let book = StableModAddressBook::from_composition(&master, &[], &composition).unwrap();
        let amount_target = StableModTarget::parse("group/9/matte.amount").unwrap();
        let threshold_target = StableModTarget::parse("group/9/matte.threshold").unwrap();
        let softness_target = StableModTarget::parse("group/9/matte.softness").unwrap();
        let bus_target = StableModTarget::parse("composition/bus_crossfade").unwrap();
        assert_eq!(amount_target.range(), [0.0, 1.0]);
        assert_eq!(threshold_target.range(), [0.0, 1.0]);
        assert_eq!(softness_target.range(), [0.0, 0.5]);
        assert_eq!(bus_target.range(), [0.0, 1.0]);

        let mut frame = StableModulationFrame {
            offsets: vec![0.0; book.len()],
        };
        for (target, offset) in [
            (amount_target, 0.25),
            (threshold_target, -0.3),
            (softness_target, 0.2),
            (bus_target, 0.4),
        ] {
            frame.offsets[book.address(target).unwrap().index()] = offset;
            approx(frame.target_offset(&book, target), offset);
        }
        let mut evaluated = composition.clone();
        apply_stable_modulation(
            &book,
            &frame,
            &mut RuntimeVisualRack::empty(),
            &mut [],
            &mut evaluated,
        );
        let matte = evaluated.group(target_id).unwrap().matte.unwrap();
        approx(matte.amount, 0.45);
        approx(matte.threshold, 0.3);
        approx(matte.softness, 0.3);
        approx(evaluated.bus_crossfade(), 0.65);
        assert_eq!(matte.channel, MatteChannel::Red);
        assert!(matte.invert);
        assert_eq!(
            composition.group(target_id).unwrap().matte,
            Some(target_matte)
        );

        // Rebuilding in the opposite internal group order must still bind by
        // GroupId, never by address order or root position.
        let reordered = RuntimeComposition::try_from_parts(
            vec![target_group, first],
            vec![
                RuntimeRootItem::Group { group_id: first_id },
                RuntimeRootItem::Group {
                    group_id: target_id,
                },
            ],
            Some(10),
            0.25,
        )
        .unwrap();
        let reordered_book =
            StableModAddressBook::from_composition(&master, &[], &reordered).unwrap();
        let mut reordered_frame = StableModulationFrame {
            offsets: vec![0.0; reordered_book.len()],
        };
        reordered_frame.offsets[reordered_book.address(amount_target).unwrap().index()] = 0.1;
        let mut reordered_evaluated = reordered.clone();
        apply_stable_modulation(
            &reordered_book,
            &reordered_frame,
            &mut RuntimeVisualRack::empty(),
            &mut [],
            &mut reordered_evaluated,
        );
        approx(
            reordered_evaluated
                .group(target_id)
                .unwrap()
                .matte
                .unwrap()
                .amount,
            0.3,
        );
        approx(
            reordered_evaluated
                .group(first_id)
                .unwrap()
                .matte
                .unwrap()
                .amount,
            1.0,
        );

        let no_matte = RuntimeComposition::try_from_parts(
            vec![runtime_group_with_matte(target_id.get(), None)],
            vec![RuntimeRootItem::Group {
                group_id: target_id,
            }],
            Some(10),
            0.5,
        )
        .unwrap();
        let missing_book = StableModAddressBook::from_composition(&master, &[], &no_matte).unwrap();
        assert!(missing_book.address(amount_target).is_none());
        assert!(missing_book.address(bus_target).is_some());
    }

    #[test]
    fn deleted_group_targets_become_typed_tombstones_without_touching_composition_values() {
        let removed = GroupId::new(9).unwrap();
        let survivor = GroupId::new(7).unwrap();
        let mut matrix = ModMatrix::new();
        matrix.routings = vec![
            Routing::new(ModSource::Lfo(0), "group/9/matte.amount", 0.4),
            Routing::new(ModSource::Lfo(1), "group/9/opacity", -0.2),
            Routing::new(ModSource::Lfo(2), "group/7/matte.softness", 0.3),
            Routing::new(ModSource::Lfo(3), "composition/bus_crossfade", 0.5),
        ];

        matrix.tombstone_group_targets_after_remove(removed);

        for (index, parameter) in [GroupModParameter::MatteAmount, GroupModParameter::Opacity]
            .into_iter()
            .enumerate()
        {
            assert_eq!(matrix.routings[index].stable_target(), None);
            assert_eq!(
                matrix.routings[index].saved_missing_target(),
                Some(SavedStableModTarget::MissingGroup {
                    group_id: removed,
                    missing_target: SavedMissingTarget::GroupValue { parameter },
                })
            );
            assert_eq!(matrix.routings[index].cached_value(), 0.0);
        }
        assert_eq!(
            matrix.routings[2].stable_target(),
            Some(StableModTarget::GroupValue {
                group_id: survivor,
                parameter: GroupModParameter::MatteSoftness,
            })
        );
        assert_eq!(
            matrix.routings[3].stable_target(),
            Some(StableModTarget::CompositionValue {
                parameter: CompositionModParameter::BusCrossfade,
            })
        );
    }

    #[test]
    fn displace_exposes_stable_addresses_for_its_two_gains_only() {
        use crate::visual_rack::{
            DisplaceBoundary, EdgeTiming, ResolvedImageSource, ResolvedImageTap,
            RuntimeDisplaceParams, RuntimeVisualNodeKind, RuntimeVisualRack,
        };

        let authored = RuntimeDisplaceParams {
            tap: ResolvedImageTap {
                source: ResolvedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            amount_x: 0.25,
            amount_y: -0.25,
            boundary: DisplaceBoundary::Mirror,
        };
        let mut rack = RuntimeVisualRack::empty();
        let node_id = rack
            .push(RuntimeVisualNodeKind::Displace(authored))
            .unwrap();

        let mut book = StableModAddressBook::default();
        book.add_rack(StableModScope::Master, &rack).unwrap();

        // Exactly one address per gain, plus the shared structural wet.
        let keys: Vec<_> = book
            .targets
            .iter()
            .filter_map(|target| match target {
                StableModTarget::Node { parameter, .. } => match parameter {
                    StableNodeParameter::Wet => Some("wet"),
                    StableNodeParameter::Descriptor {
                        descriptor_index, ..
                    } => NODE_PARAM_DESCRIPTORS
                        .get(usize::from(*descriptor_index))
                        .map(|descriptor| descriptor.key),
                },
                _ => None,
            })
            .collect();
        assert_eq!(keys, vec!["wet", "amount_x", "amount_y"]);

        // Offsets land on the addressed axis and clamp to the ±1 UV domain.
        let address_of = |key: &str| {
            book.targets
                .iter()
                .position(|target| {
                    matches!(
                        target,
                        StableModTarget::Node {
                            parameter: StableNodeParameter::Descriptor { descriptor_index, .. },
                            ..
                        } if NODE_PARAM_DESCRIPTORS[usize::from(*descriptor_index)].key == key
                    )
                })
                .map(|index| StableModAddress(index as u16))
                .unwrap()
        };
        let mut offsets = vec![0.0_f32; book.targets.len()];
        offsets[address_of("amount_x").index()] = 0.5;
        offsets[address_of("amount_y").index()] = -5.0;
        let frame = StableModulationFrame { offsets };

        let mut modulated = rack.clone();
        apply_stable_rack_modulation(&book, &frame, StableModScope::Master, &mut modulated);
        let RuntimeVisualNodeKind::Displace(params) = modulated.get(node_id).unwrap().kind else {
            panic!("displace node")
        };
        assert!((params.amount_x - 0.75).abs() < 1e-5);
        assert_eq!(params.amount_y, -1.0, "offsets clamp into the UV domain");
        assert_eq!(
            params.tap, authored.tap,
            "modulation never touches the donor route"
        );
        assert_eq!(
            params.boundary, authored.boundary,
            "modulation never touches the boundary law"
        );

        // The route and boundary descriptors have no modulatable address.
        for key in ["donor_tap", "boundary"] {
            let index = NODE_PARAM_DESCRIPTORS
                .iter()
                .position(|descriptor| {
                    descriptor.kind == NodeKindTag::Displace && descriptor.key == key
                })
                .unwrap();
            let parameter = StableNodeParameter::Descriptor {
                descriptor_index: index as u16,
                component: StableModComponent::Scalar,
            };
            assert!(!parameter.is_valid_for_kind(NodeKindTag::Displace));
        }
    }

    /// Symmetry publishes one stable address per declared continuous control
    /// (plus the shared structural wet) and nothing else. Angular controls wrap
    /// instead of clamping, and neither the routes, the discrete laws, the
    /// authored seed, nor the six mask bits can be reached by modulation — so
    /// no route can rewrite the sector table.
    #[test]
    fn symmetry_exposes_stable_addresses_for_its_declared_continuous_controls_only() {
        use crate::symmetry::{
            RuntimeSymmetryParams, SymmetryBoundary, SymmetryMode, SymmetryMotionMask,
            SymmetrySourceMask,
        };
        use crate::visual_rack::{RuntimeVisualNodeKind, RuntimeVisualRack};

        let authored = RuntimeSymmetryParams {
            mode: SymmetryMode::PlanarPmm,
            base_folds: 6.0,
            radial_phase_deg: 170.0,
            boundary: SymmetryBoundary::CellularReentry,
            seed: 4_242,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: false,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: false,
            },
            ..RuntimeSymmetryParams::default()
        };
        let mut rack = RuntimeVisualRack::empty();
        let node_id = rack
            .push(RuntimeVisualNodeKind::Symmetry(authored))
            .unwrap();

        let mut book = StableModAddressBook::default();
        book.add_rack(StableModScope::Master, &rack).unwrap();

        let keys: Vec<_> = book
            .targets
            .iter()
            .filter_map(|target| match target {
                StableModTarget::Node { parameter, .. } => match parameter {
                    StableNodeParameter::Wet => Some("wet".to_string()),
                    StableNodeParameter::Descriptor {
                        descriptor_index,
                        component,
                    } => NODE_PARAM_DESCRIPTORS
                        .get(usize::from(*descriptor_index))
                        .map(|descriptor| {
                            let suffix = match component {
                                StableModComponent::Scalar => "",
                                StableModComponent::X => ".x",
                                StableModComponent::Y => ".y",
                                StableModComponent::Red => ".r",
                                StableModComponent::Green => ".g",
                                StableModComponent::Blue => ".b",
                            };
                            format!("{}{suffix}", descriptor.key)
                        }),
                },
                _ => None,
            })
            .collect();
        assert_eq!(
            keys,
            vec![
                "wet",
                "symmetry_base_folds",
                "symmetry_fold_offset",
                "symmetry_radial_phase_deg",
                "symmetry_orbit_phase",
                "symmetry_planar_axis_deg",
                "symmetry_planar_phase",
                "symmetry_cell_skew",
                "symmetry_spiral_scale",
                "symmetry_orbit_radius",
                "symmetry_orbit_spin_deg",
                "symmetry_motion_gain",
                "symmetry_hue_span",
                "symmetry_center.x",
                "symmetry_center.y",
            ]
        );

        let address_of = |key: &str, component: StableModComponent| {
            book.targets
                .iter()
                .position(|target| {
                    matches!(
                        target,
                        StableModTarget::Node {
                            parameter: StableNodeParameter::Descriptor { descriptor_index, component: found },
                            ..
                        } if NODE_PARAM_DESCRIPTORS[usize::from(*descriptor_index)].key == key
                            && *found == component
                    )
                })
                .map(|index| StableModAddress(index as u16))
                .unwrap()
        };
        let mut offsets = vec![0.0_f32; book.targets.len()];
        offsets[address_of("symmetry_fold_offset", StableModComponent::Scalar).index()] = 4.0;
        offsets[address_of("symmetry_hue_span", StableModComponent::Scalar).index()] = 9.0;
        offsets[address_of("symmetry_radial_phase_deg", StableModComponent::Scalar).index()] = 20.0;
        offsets[address_of("symmetry_center", StableModComponent::Y).index()] = 0.25;
        let frame = StableModulationFrame { offsets };

        let mut modulated = rack.clone();
        apply_stable_rack_modulation(&book, &frame, StableModScope::Master, &mut modulated);
        let RuntimeVisualNodeKind::Symmetry(params) = modulated.get(node_id).unwrap().kind else {
            panic!("symmetry node")
        };
        assert!((params.fold_offset - 4.0).abs() < 1e-5);
        assert_eq!(
            params.hue_span, 1.0,
            "offsets clamp into the declared range"
        );
        assert!(
            (params.radial_phase_deg - (-170.0)).abs() < 1e-3,
            "an angular control wraps rather than clamping: got {}",
            params.radial_phase_deg
        );
        assert!((params.center[1] - 0.75).abs() < 1e-5);
        assert_eq!(params.center[0], authored.center[0]);
        // The rounded fold count observes the modulated sum exactly once.
        assert_eq!(params.values().effective_folds(), 10);

        assert_eq!(params.donors, authored.donors, "routes are never modulated");
        assert_eq!(params.motion, authored.motion, "routes are never modulated");
        assert_eq!(params.mode, authored.mode);
        assert_eq!(params.boundary, authored.boundary);
        assert_eq!(params.seed, authored.seed);
        assert_eq!(params.source_mask, authored.source_mask);
        assert_eq!(params.motion_mask, authored.motion_mask);
        assert_eq!(
            params.sector_table(crate::symmetry::SymmetryNodeDomain::new(1, 3)),
            authored.sector_table(crate::symmetry::SymmetryNodeDomain::new(1, 3)),
            "modulation can never reroll the sector table"
        );

        for key in [
            "symmetry_mode",
            "symmetry_boundary",
            "symmetry_seed",
            "symmetry_donor0_tap",
            "symmetry_donor1_tap",
            "symmetry_motion0_donor",
            "symmetry_motion1_donor",
            "symmetry_source_carrier",
            "symmetry_source_donor0",
            "symmetry_source_donor1",
            "symmetry_source_history",
            "symmetry_motion_slot0",
            "symmetry_motion_slot1",
        ] {
            let index = NODE_PARAM_DESCRIPTORS
                .iter()
                .position(|descriptor| {
                    descriptor.kind == NodeKindTag::Symmetry && descriptor.key == key
                })
                .unwrap();
            let parameter = StableNodeParameter::Descriptor {
                descriptor_index: index as u16,
                component: StableModComponent::Scalar,
            };
            assert!(!parameter.is_valid_for_kind(NodeKindTag::Symmetry), "{key}");
        }
    }

    #[test]
    fn residual_exposes_stable_addresses_for_its_mix_and_detail_gain_under_unique_wire_keys() {
        use crate::visual_rack::{
            EdgeTiming, ResidualBlock, ResidualQuantization, ResolvedImageSource, ResolvedImageTap,
            RuntimeResidualParams, RuntimeVisualNodeKind, RuntimeVisualRack,
        };

        let authored = RuntimeResidualParams {
            structure: ResolvedImageTap {
                source: ResolvedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            detail: ResolvedImageTap {
                source: ResolvedImageSource::AllBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            block: ResidualBlock::Sixteen,
            quantization: ResidualQuantization::Medium,
            mix: 0.25,
            detail_gain: 2.0,
            seed: 0x00c0_ffee,
            ..RuntimeResidualParams::default()
        };
        let mut rack = RuntimeVisualRack::empty();
        let node_id = rack
            .push(RuntimeVisualNodeKind::Residual(authored))
            .unwrap();

        let mut book = StableModAddressBook::default();
        book.add_rack(StableModScope::Master, &rack).unwrap();

        // Exactly one address per continuous value, plus the shared wet. Both
        // routes, both discrete laws and the seed register nothing.
        let keys: Vec<_> = book
            .targets
            .iter()
            .filter_map(|target| match target {
                StableModTarget::Node { parameter, .. } => match parameter {
                    StableNodeParameter::Wet => Some("wet"),
                    StableNodeParameter::Descriptor {
                        descriptor_index, ..
                    } => NODE_PARAM_DESCRIPTORS
                        .get(usize::from(*descriptor_index))
                        .map(|descriptor| descriptor.key),
                },
                _ => None,
            })
            .collect();
        assert_eq!(keys, vec!["wet", "mix", "detail_gain"]);

        let address_of = |key: &str| {
            book.targets
                .iter()
                .position(|target| {
                    matches!(
                        target,
                        StableModTarget::Node {
                            parameter: StableNodeParameter::Descriptor { descriptor_index, .. },
                            ..
                        } if NODE_PARAM_DESCRIPTORS[usize::from(*descriptor_index)].key == key
                    )
                })
                .map(|index| StableModAddress(index as u16))
                .unwrap()
        };
        let mut offsets = vec![0.0_f32; book.targets.len()];
        offsets[address_of("mix").index()] = 0.5;
        offsets[address_of("detail_gain").index()] = 9.0;
        let frame = StableModulationFrame { offsets };

        let mut modulated = rack.clone();
        apply_stable_rack_modulation(&book, &frame, StableModScope::Master, &mut modulated);
        let RuntimeVisualNodeKind::Residual(params) = modulated.get(node_id).unwrap().kind else {
            panic!("residual node")
        };
        assert!((params.mix - 0.75).abs() < 1e-5);
        assert_eq!(
            params.detail_gain, 4.0,
            "offsets clamp into the declared gain domain"
        );
        assert_eq!(
            params.routes(),
            authored.routes(),
            "modulation never touches either donor route"
        );
        assert_eq!(
            (params.block, params.quantization),
            (authored.block, authored.quantization)
        );
        assert_eq!(
            params.seed, authored.seed,
            "modulation never touches the quantization seed"
        );

        // Both routes, both discrete laws and the seed have no modulatable
        // address at all.
        for key in [
            "structure_tap",
            "detail_tap",
            "block",
            "quantization",
            "seed",
        ] {
            let index = NODE_PARAM_DESCRIPTORS
                .iter()
                .position(|descriptor| {
                    descriptor.kind == NodeKindTag::Residual && descriptor.key == key
                })
                .unwrap();
            let parameter = StableNodeParameter::Descriptor {
                descriptor_index: index as u16,
                component: StableModComponent::Scalar,
            };
            assert!(!parameter.is_valid_for_kind(NodeKindTag::Residual));
        }

        // The real cross-resolution hazard: `StableNodeParameter::parse` binds
        // the FIRST modulatable row with a matching key, `same_wire_parameter`
        // compares keys rather than indices, and `runtime_node_supports_
        // descriptor` returns true for every non-Mask kind. Both Residual wire
        // keys are therefore globally unique among modulatable rows, so no
        // route authored for another kind can resolve onto this one.
        for key in ["mix", "detail_gain"] {
            let owners: Vec<_> = NODE_PARAM_DESCRIPTORS
                .iter()
                .filter(|descriptor| descriptor.key == key && descriptor.modulatable)
                .map(|descriptor| descriptor.kind)
                .collect();
            assert_eq!(
                owners,
                vec![NodeKindTag::Residual],
                "modulatable wire key {key} must be owned by Residual alone"
            );
            let parsed = StableNodeParameter::parse(key).expect("a modulatable key parses");
            assert!(parsed.is_valid_for_kind(NodeKindTag::Residual));
            for other in [
                NodeKindTag::Cellular,
                NodeKindTag::Shift,
                NodeKindTag::Grain,
                NodeKindTag::DigitalColor,
                NodeKindTag::Key,
                NodeKindTag::Transform,
                NodeKindTag::Mask,
                NodeKindTag::Displace,
            ] {
                assert!(
                    !parsed.is_valid_for_kind(other),
                    "{key} must not cross-resolve onto {other:?}"
                );
            }
        }

        // A shared key such as `amount` still aliases across its own kinds, and
        // that aliasing must never reach Residual.
        let shared = StableNodeParameter::parse("amount").expect("a shared key parses");
        assert!(!shared.is_valid_for_kind(NodeKindTag::Residual));
        let mix = StableNodeParameter::parse("mix").unwrap();
        assert!(!mix.same_wire_parameter(shared));
        assert!(mix.same_wire_parameter(StableNodeParameter::parse("mix").unwrap()));
    }

    /// S3b's whole modulation surface: three derived continuous scalars with
    /// distinct wire keys, applied to a copy, and no address anywhere that
    /// could reach a recorded gesture.
    #[test]
    fn gesture_canvas_exposes_three_uniquely_named_continuous_targets_and_no_track_address() {
        for (target, range) in [
            ("gesture_radius", (0.0_f32, 1.0_f32)),
            ("gesture_strength", (0.0, 1.0)),
            ("gesture_retention", (0.0, 1.0)),
        ] {
            assert_eq!(target_range(target), Some(range), "{target}");
            assert!(is_valid_target(target));
            assert_eq!(
                TARGETS.iter().filter(|(key, _, _)| *key == target).count(),
                1,
                "{target} must appear exactly once in the master table"
            );
            assert!(
                !LAYER_TARGET_SUFFIXES.contains(&target),
                "{target} is a master-scope subsystem and owns no layer suffix"
            );
            assert!(
                matches!(compile_target(target), CompiledTarget::Master(_)),
                "{target} must compile to a master slot, not a stable node address"
            );
        }

        // Distinct keys, not bare ones. A bare `radius`/`strength`/`retention`
        // would cross-resolve against another subsystem through key equality.
        for bare in ["radius", "strength", "retention", "gesture", "canvas"] {
            assert_eq!(target_range(bare), None, "{bare} must not be a target");
        }
        // The recording itself has no address at all.
        for track in [
            "gesture_track",
            "gesture_events",
            "gesture_recording",
            "gesture_checksum",
            "layer1_gesture_radius",
        ] {
            assert_eq!(target_range(track), None, "{track} must not be modulatable");
        }

        // Appending before the crossfader keeps the compiled morph slot valid.
        assert_eq!(TARGETS[MORPH_TARGET_INDEX].0, "morph");

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix
            .routings
            .push(Routing::new(ModSource::Midi(0), "gesture_strength", 1.0));
        matrix.update_at_beat(0.0, 1.0 / 30.0);
        let base = crate::gesture_canvas::GestureCanvasParams {
            radius: 0.25,
            strength: 0.5,
            retention: 0.75,
        };
        let frame = matrix.frame(0);
        let evaluated = frame.modulate_gesture_canvas(&base);
        assert!(
            evaluated.strength > base.strength,
            "a positive MIDI route must raise the evaluated strength"
        );
        assert_eq!(evaluated.radius, base.radius);
        assert_eq!(evaluated.retention, base.retention);
        assert_eq!(
            base,
            crate::gesture_canvas::GestureCanvasParams {
                radius: 0.25,
                strength: 0.5,
                retention: 0.75,
            },
            "modulation must contribute to a copy and never rewrite the authored base"
        );

        // Offsets clamp into the declared range instead of escaping it.
        let mut deep = ModMatrix::new();
        deep.midi[0] = 1.0;
        deep.routings
            .push(Routing::new(ModSource::Midi(0), "gesture_retention", 1.0));
        deep.routings
            .push(Routing::new(ModSource::Midi(0), "gesture_radius", 1.0));
        deep.update_at_beat(0.0, 1.0 / 30.0);
        let saturated = deep.frame(0).modulate_gesture_canvas(&base);
        assert!((0.0..=1.0).contains(&saturated.radius));
        assert!((0.0..=1.0).contains(&saturated.retention));

        // An inert matrix leaves the sanitized base exactly where it was.
        let inert = ModMatrix::new().frame(0).modulate_gesture_canvas(&base);
        assert_eq!(inert, base.sanitized());
    }
}
