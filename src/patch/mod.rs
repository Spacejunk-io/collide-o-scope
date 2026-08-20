pub mod editor;

use serde::de;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::composition::{
    CompositionTree, RootItem, RuntimeComposition, RuntimeGroup, RuntimeGroupMembers,
    RuntimeRootItem,
};
use crate::effects::params::{
    CollisionAtlasParams, CollisionScoreParams, CollisionScoreTrigger, FeedbackRigParams,
    FeedbackShape, RefreshGardenGate, RefreshGardenParams, TemporalInterpolation,
    TemporalLoomParams, TemporalOriginalsParams, TemporalParams, TemporalTopology, TimeDisplaceMap,
};
use crate::effects::EffectUniforms;
use crate::image_routing::{
    LayerImageStage, LayerMatte, LayerMatteConfig, SavedImageInput, StableLayerId,
};
use crate::layers::{BlendMode, Layer};
use crate::modulation::{
    Curve, GyroAxisConfig, Lfo, LfoShape, ModMatrix, ModSource, PadAxisConfig, PadConfig,
    ResolvedStableModTarget, Routing, SavedStableModScope, SavedStableModTarget,
    StableModAddressBook, MAX_ROUTINGS, NUM_LFOS,
};
use crate::motion::{
    CurvedShutterParams, CurvedShutterQuality, FaradayParams, FieldColliderMode,
    FieldColliderParams, FlowShapingParams, MotionBoundaryMode, MotionCarrier, MotionDonor,
    MotionFieldSource, MotionLatticeQuality, MotionParams, ProceduralFieldKind,
    ProceduralFieldParams, FIELD_COLLIDER_ALGORITHM_VERSION, MOTION_ALGORITHM_VERSION,
};
use crate::ntsc::NtscParams;
use crate::performance::{
    ClipSlotConfig, ClipSlotId, ClipSlots, SavedLayerPosition, SceneReferenceErrorKind,
    SceneReferenceIssue, Scenes,
};
use crate::spatial::{EdgeMode, FitMode, SamplingMode, SpatialTransform};
use crate::temporal::{
    CollisionScoreLoopDriver, RefreshGardenMatteRoute, RefreshGardenMotionRoute,
    TemporalEventResetMode, TemporalResetPolicy,
};
use crate::visual_rack::{
    EdgeTiming, GroupId, ImageDependency, ImageDependencyGraph, ImageGraphMode, ImageOrderingEdge,
    LegacyRackScope, MaskParams, NodeId, RuntimeImageMatte, RuntimeVisualRack, SavedImageSource,
    SavedImageTap, VisualNodeKind, VisualRack, VisualScopeId, RACK_PRIMARY_ROUTE_SLOT,
    RESIDUAL_DETAIL_SLOT, RESIDUAL_ROUTE_SLOTS, RESIDUAL_STRUCTURE_SLOT,
};

// --- Helpers for serde defaults ---

fn one() -> f32 {
    1.0
}
fn default_fps() -> f32 {
    30.0
}
fn default_cellular_scale() -> f32 {
    10.0
}
fn default_cellular_warp() -> f32 {
    0.35
}
fn default_cellular_speed() -> f32 {
    0.25
}
fn default_cellular_gap_threshold() -> f32 {
    0.65
}
fn default_cellular_gap_softness() -> f32 {
    0.08
}
fn default_shift_block_size() -> f32 {
    8.0
}
fn default_shift_density() -> f32 {
    0.5
}
fn default_shift_speed() -> f32 {
    3.0
}
fn default_key_color() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}
fn default_key_tolerance() -> f32 {
    0.15
}
fn default_temporal_key_threshold() -> f32 {
    0.1
}
fn default_temporal_key_softness() -> f32 {
    0.03
}
// B13 small-effects defaults. Every new DTO field is skip-serialized at its
// default so pre-B13 patches keep their bytes and canonical hashes.
fn default_contour_bands() -> f32 {
    10.0
}
fn default_contour_width() -> f32 {
    1.2
}
fn default_contour_fill() -> f32 {
    0.25
}
fn default_flatten_levels() -> f32 {
    5.0
}
fn default_colourpass_width() -> f32 {
    0.25
}
fn default_emboss_angle() -> f32 {
    45.0
}
fn default_halftone_pitch() -> f32 {
    0.4
}
fn default_moire_freq() -> f32 {
    0.4
}
fn default_bitcrush_levels() -> f32 {
    2.0
}
fn default_bitcrush_dither() -> f32 {
    1.0
}
fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0
}
fn is_one_f32(value: &f32) -> bool {
    *value == 1.0
}
fn is_default_contour_bands(value: &f32) -> bool {
    *value == default_contour_bands()
}
fn is_default_contour_width(value: &f32) -> bool {
    *value == default_contour_width()
}
fn is_default_contour_fill(value: &f32) -> bool {
    *value == default_contour_fill()
}
fn is_default_flatten_levels(value: &f32) -> bool {
    *value == default_flatten_levels()
}
fn is_default_colourpass_width(value: &f32) -> bool {
    *value == default_colourpass_width()
}
fn is_default_emboss_angle(value: &f32) -> bool {
    *value == default_emboss_angle()
}
fn is_default_halftone_pitch(value: &f32) -> bool {
    *value == default_halftone_pitch()
}
fn is_default_moire_freq(value: &f32) -> bool {
    *value == default_moire_freq()
}
fn is_default_bitcrush_levels(value: &f32) -> bool {
    *value == default_bitcrush_levels()
}
fn is_default_bitcrush_dither(value: &f32) -> bool {
    *value == default_bitcrush_dither()
}

fn is_legacy_spatial_identity(value: &SpatialTransform) -> bool {
    (*value).is_legacy_identity()
}

// --- Parameter metadata for stepping & comments ---

pub struct ParamMeta {
    pub step: f32,
    pub min: f32,
    pub max: f32,
    pub desc: &'static str,
}

pub fn param_meta(name: &str) -> Option<ParamMeta> {
    match name {
        "pixelate" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 32.0,
            desc: "pixel block size",
        }),
        "rgb_split" => Some(ParamMeta {
            step: 0.5,
            min: 0.0,
            max: 30.0,
            desc: "chromatic split px",
        }),
        "hue_shift" => Some(ParamMeta {
            step: 5.0,
            min: -180.0,
            max: 180.0,
            desc: "degrees",
        }),
        "saturation" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "color intensity",
        }),
        "brightness" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "exposure",
        }),
        "contrast" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "dynamic range",
        }),
        "posterize" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 16.0,
            desc: "color levels (0=off)",
        }),
        "downsample" => Some(ParamMeta {
            step: 0.05,
            min: 0.05,
            max: 1.0,
            desc: "render resolution fraction",
        }),
        "shift_amount" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "horizontal block displacement mix",
        }),
        "shift_block_size" => Some(ParamMeta {
            step: 1.0,
            min: 2.0,
            max: 256.0,
            desc: "horizontal band height in pixels",
        }),
        "shift_density" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "fraction of bands displaced per epoch",
        }),
        "shift_speed" => Some(ParamMeta {
            step: 0.25,
            min: 0.0,
            max: 20.0,
            desc: "deterministic pattern epochs per second",
        }),
        "cellular_amount" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "cellular effect mix",
        }),
        "cellular_scale" => Some(ParamMeta {
            step: 1.0,
            min: 2.0,
            max: 32.0,
            desc: "cells across frame height",
        }),
        "cellular_warp" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "bounded domain displacement",
        }),
        "cellular_speed" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 2.0,
            desc: "feature target epochs per second",
        }),
        "cellular_gap_amount" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "cell ridge transparency",
        }),
        "cellular_gap_threshold" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "ridge strength keyed out",
        }),
        "cellular_gap_softness" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.5,
            desc: "transparent gap edge feather",
        }),
        "grain_intensity" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.3,
            desc: "film grain amount",
        }),
        "grain_size" => Some(ParamMeta {
            step: 0.25,
            min: 1.0,
            max: 4.0,
            desc: "grain particle scale",
        }),
        "grain_algo" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 3.0,
            desc: "0=value 1=perlin 2=gaussian 3=salt&pepper",
        }),
        "breathe_scale" => Some(ParamMeta {
            step: 0.005,
            min: 0.0,
            max: 0.05,
            desc: "zoom oscillation",
        }),
        "breathe_rotation" => Some(ParamMeta {
            step: 0.1,
            min: 0.0,
            max: 2.0,
            desc: "rotation oscillation deg",
        }),
        "breathe_position" => Some(ParamMeta {
            step: 0.002,
            min: 0.0,
            max: 0.02,
            desc: "position drift",
        }),
        "vignette" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.5,
            desc: "edge darkening",
        }),
        "color_drift" => Some(ParamMeta {
            step: 0.002,
            min: 0.0,
            max: 0.02,
            desc: "chromatic aberration",
        }),
        "key_mode" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 4.0,
            desc: "0=off 1=bright 2=dark 3=remove chroma 4=keep chroma",
        }),
        "key_threshold" | "key_tolerance" | "key_color_r" | "key_color_g" | "key_color_b" => {
            Some(ParamMeta {
                step: 0.01,
                min: 0.0,
                max: 1.0,
                desc: "normalized key control",
            })
        }
        "key_softness" => Some(ParamMeta {
            step: 0.01,
            min: 0.0,
            max: 0.5,
            desc: "key edge feather",
        }),
        // B13 small effects.
        "contour" | "flatten" | "contour_dither" | "solarize" | "negative" | "colourpass"
        | "edge_amount" | "emboss" | "halftone" | "moire" | "row_smear" | "bitcrush"
        | "bitcrush_dither" | "contour_hue" | "contour_fill" | "colourpass_width"
        | "halftone_pitch" | "moire_freq" | "chroma_aberration" | "anamorphic_streak"
        | "key_border" | "key_shadow" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "normalized small-effect control",
        }),
        "contour_bands" => Some(ParamMeta {
            step: 1.0,
            min: 2.0,
            max: 40.0,
            desc: "luma bands",
        }),
        "contour_width" => Some(ParamMeta {
            step: 0.1,
            min: 0.2,
            max: 6.0,
            desc: "isoline width px",
        }),
        "flatten_levels" | "bitcrush_levels" => Some(ParamMeta {
            step: 1.0,
            min: 2.0,
            max: 16.0,
            desc: "quantize levels",
        }),
        "negative_mode" => Some(ParamMeta {
            step: 1.0,
            min: 0.0,
            max: 2.0,
            desc: "0=rgb 1=luma 2=hue-flip",
        }),
        "colourpass_hue" | "edge_hue" | "emboss_angle" | "halftone_angle" => Some(ParamMeta {
            step: 5.0,
            min: -180.0,
            max: 180.0,
            desc: "degrees",
        }),
        "multi_grid_x" | "multi_grid_y" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 8.0,
            desc: "tile count (1=off)",
        }),
        "barrel" => Some(ParamMeta {
            step: 0.05,
            min: -1.0,
            max: 1.0,
            desc: "radial lens distortion",
        }),
        "opacity" => Some(ParamMeta {
            step: 0.05,
            min: 0.0,
            max: 1.0,
            desc: "layer transparency",
        }),
        "speed" => Some(ParamMeta {
            step: 0.25,
            min: 0.25,
            max: 4.0,
            desc: "playback multiplier",
        }),
        "fps" => Some(ParamMeta {
            step: 1.0,
            min: 1.0,
            max: 240.0,
            desc: "decode frame rate",
        }),
        _ => None,
    }
}

// --- Serializable patch state ---

#[derive(Serialize, Clone)]
pub struct PatchState {
    pub master: EffectsConfig,
    /// Program-wide authored geometry. Its serde default takes the exact
    /// inactive historical sample path, so old patches remain pixel-compatible;
    /// once moved, newly exposed canvas starts transparent.
    #[serde(default, skip_serializing_if = "is_legacy_spatial_identity")]
    pub master_transform: SpatialTransform,
    /// Program-wide M4 motion authoring. Omitted is the exact historical
    /// no-op; hidden field/carrier pixels are never persisted here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_motion: Option<MotionConfig>,
    pub layers: Vec<LayerConfig>,
    /// Omitted means the pre-rack legacy order must be synthesized. `Some`
    /// with zero nodes is an explicitly authored empty rack and must never be
    /// collapsed back to the legacy shader marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master_rack: Option<VisualRack>,
    /// Saved-position form of the one-level runtime composition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition: Option<CompositionTree>,
    /// 0/omitted is the exact legacy schema; 1 declares explicit M2 visual
    /// topology. Future versions are rejected before any state is sanitized.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub visual_schema_version: u32,
    #[serde(default)]
    pub master_paused: bool,
    /// Hold file/Spout source images while program clocks and effects continue.
    #[serde(default)]
    pub media_frozen: bool,
    #[serde(default)]
    pub ntsc: Option<NtscConfig>,
    #[serde(default)]
    pub modulation: Option<ModConfig>,
    #[serde(default)]
    pub temporal: Option<TemporalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub morph: Option<crate::morph::MorphStateSnapshot>,
    /// Prepared multi-layer recalls. Empty is the exact legacy behavior.
    #[serde(default, skip_serializing_if = "Scenes::is_empty")]
    pub scenes: Scenes,
    /// Recorded gesture track. A track is authored topology rather than a
    /// value: it is carried whole, never interpolated, and an absent section is
    /// exactly the pre-gesture path. Its own bounded, checksum-verifying
    /// deserializer is the single acceptance gate, so a hostile patch cannot
    /// smuggle an ill-formed stream in through YAML.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gesture_track: Option<crate::gesture::GestureTrackDocument>,
    /// Authored gesture-canvas controls. These are ordinary continuous values,
    /// so unlike the track they interpolate; omitting the section is exactly
    /// the pre-gesture path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gesture_canvas: Option<GestureCanvasConfig>,
    /// Every Study document any rack node in this patch references, carried
    /// whole so the patch stays self-contained — the gesture-track precedent.
    /// Each document validates through its own strict deserializer and ABI
    /// gate on load; an absent section is exactly the pre-study path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub studies: Vec<crate::study::StudyDocument>,
    /// B9 recorded performance take, carried whole on the gesture-track law:
    /// a take is authored topology, never interpolated, and an absent section
    /// is exactly the pre-recorder path. Its own bounded, checksum-verifying
    /// deserializer is the single acceptance gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance_take: Option<crate::performance_track::PerformanceTakeDocument>,
}

impl PatchState {
    /// Every distinct Study digest any rack in this patch references, in
    /// digest order. The walk is saved-form only — master rack, layer racks,
    /// group racks — and never consults the host library.
    pub fn referenced_study_digests(&self) -> Vec<[u8; 32]> {
        let mut digests = std::collections::BTreeSet::new();
        let mut visit = |rack: &VisualRack| {
            for node in rack.iter() {
                if let crate::visual_rack::VisualNodeKind::Study(params) = node.kind {
                    if let Some(digest) = params.document_digest {
                        digests.insert(digest);
                    }
                }
            }
        };
        if let Some(rack) = &self.master_rack {
            visit(rack);
        }
        for layer in &self.layers {
            if let Some(rack) = &layer.rack {
                visit(rack);
            }
        }
        if let Some(composition) = &self.composition {
            for group in composition.groups() {
                visit(&group.rack);
            }
        }
        digests.into_iter().collect()
    }
}

impl<'de> Deserialize<'de> for PatchState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawPatchState {
            master: EffectsConfig,
            #[serde(default)]
            master_transform: SpatialTransform,
            #[serde(default)]
            master_motion: Option<MotionConfig>,
            layers: Vec<LayerConfig>,
            #[serde(default)]
            master_rack: Option<VisualRack>,
            #[serde(default)]
            composition: Option<CompositionTree>,
            #[serde(default)]
            visual_schema_version: u32,
            #[serde(default)]
            master_paused: bool,
            #[serde(default)]
            media_frozen: bool,
            #[serde(default)]
            ntsc: Option<NtscConfig>,
            #[serde(default)]
            modulation: Option<ModConfig>,
            #[serde(default)]
            temporal: Option<TemporalConfig>,
            #[serde(default)]
            morph: Option<crate::morph::MorphStateSnapshot>,
            #[serde(default)]
            scenes: Scenes,
            #[serde(default)]
            gesture_track: Option<crate::gesture::GestureTrackDocument>,
            #[serde(default)]
            gesture_canvas: Option<GestureCanvasConfig>,
            #[serde(default)]
            studies: Vec<crate::study::StudyDocument>,
            #[serde(default)]
            performance_take: Option<crate::performance_track::PerformanceTakeDocument>,
        }

        let raw = RawPatchState::deserialize(deserializer)?;
        let mut patch = Self {
            master: raw.master,
            master_transform: raw.master_transform,
            master_motion: raw.master_motion,
            layers: raw.layers,
            master_rack: raw.master_rack,
            composition: raw.composition,
            visual_schema_version: raw.visual_schema_version,
            master_paused: raw.master_paused,
            media_frozen: raw.media_frozen,
            ntsc: raw.ntsc,
            modulation: raw.modulation,
            temporal: raw.temporal,
            morph: raw.morph,
            scenes: raw.scenes,
            gesture_track: raw.gesture_track,
            gesture_canvas: raw.gesture_canvas.map(GestureCanvasConfig::sanitized),
            studies: raw.studies,
            performance_take: raw.performance_take,
        };
        patch
            .validate_creative_persistence()
            .map_err(de::Error::custom)?;
        patch.sanitize_performance_references();
        Ok(patch)
    }
}

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

/// Transport fields captured alongside a full patch snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchTransportState {
    pub master_paused: bool,
    pub media_frozen: bool,
}

/// The program-wide authored visual pair captured as one semantic unit.
/// Keeping effects and geometry together prevents save/export call sites from
/// accidentally omitting one half as the visual schema grows.
#[derive(Clone, Copy)]
pub struct PatchMasterVisual<'a> {
    pub effects: &'a EffectUniforms,
    pub transform: &'a SpatialTransform,
}

impl<'a> PatchMasterVisual<'a> {
    pub fn new(effects: &'a EffectUniforms, transform: &'a SpatialTransform) -> Self {
        Self { effects, transform }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookRackScope {
    Master,
    Layer(StableLayerId),
    Group(GroupId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookNodeRef {
    pub scope: LookRackScope,
    pub node_id: NodeId,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LookApplySummary {
    pub mapped_layers: usize,
    pub unused_patch_layers: usize,
    pub untouched_live_layers: usize,
    pub applied_racks: usize,
    pub skipped_racks: usize,
    pub applied_groups: usize,
    pub skipped_groups: usize,
    pub applied_group_ids: Vec<GroupId>,
    pub skipped_group_ids: Vec<GroupId>,
    pub applied_nodes: Vec<LookNodeRef>,
    pub skipped_nodes: Vec<LookNodeRef>,
    pub applied_bus_crossfade: bool,
    pub skipped_bus_crossfade: bool,
}

/// Serializable temporal (feedback/slit-scan) parameters for patch files.
#[derive(Serialize, Deserialize, Clone)]
pub struct TemporalConfig {
    #[serde(default)]
    pub feedback: f32,
    #[serde(default = "one")]
    pub fb_zoom: f32,
    #[serde(default)]
    pub fb_rotate: f32,
    #[serde(default)]
    pub slitscan: f32,
    #[serde(default)]
    pub slit_axis: f32,
    /// Arbitrary angle added after the original row/column-only format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slit_angle: Option<f32>,
    /// Additive B12 time-displace map. Skipped at the default `Ramp` so
    /// earlier patches keep their bytes and canonical hashes.
    #[serde(default, skip_serializing_if = "TimeDisplaceMapConfig::is_default")]
    pub slit_map: TimeDisplaceMapConfig,
    /// Additive B12 interpolation toggle. Skipped at the default `false`,
    /// which is the exact banded prior path.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub slit_interp: bool,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_temporal_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_temporal_key_softness")]
    pub key_softness: f32,
    #[serde(default = "one")]
    pub key_history: f32,
    /// Additive M3 authoring state. An absent block is the exact historical
    /// no-op, so old patches retain byte-compatible temporal behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub originals: Option<TemporalOriginalsConfig>,
    /// Additive B3 feedback rig. Skipped at identity so earlier patches keep
    /// their bytes and canonical hashes.
    #[serde(default, skip_serializing_if = "TemporalRigConfig::is_default")]
    pub rig: TemporalRigConfig,
    /// Additive B4 display physics. Skipped at the exact-off default so
    /// earlier patches keep their bytes and canonical hashes; hostile
    /// scalars sanitize on load and unknown fields are rejected.
    #[serde(default, skip_serializing_if = "display_config_is_default")]
    pub display: crate::display_physics::DisplayPhysicsParams,
    /// Additive B8 melting edge. Skipped at the exact-off default so
    /// earlier patches keep their bytes and canonical hashes; hostile
    /// scalars sanitize on load and unknown fields are rejected.
    #[serde(default, skip_serializing_if = "melt_config_is_default")]
    pub melt: crate::mixing_boundary::MeltParams,
    /// Additive B5 codec mosh. Skipped at the exact-bypass default so
    /// earlier patches keep their bytes and canonical hashes; hostile
    /// scalars sanitize on load and unknown fields are rejected.
    #[serde(default, skip_serializing_if = "mosh_config_is_default")]
    pub mosh: crate::codec_mosh::CodecMoshParams,
}

fn display_config_is_default(value: &crate::display_physics::DisplayPhysicsParams) -> bool {
    *value == crate::display_physics::DisplayPhysicsParams::default()
}

fn melt_config_is_default(value: &crate::mixing_boundary::MeltParams) -> bool {
    *value == crate::mixing_boundary::MeltParams::default()
}

fn mosh_config_is_default(value: &crate::codec_mosh::CodecMoshParams) -> bool {
    *value == crate::codec_mosh::CodecMoshParams::default()
}

/// Stable serialized vocabulary for the B12 time-displace map.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeDisplaceMapConfig {
    #[default]
    Ramp,
    Brightness,
    Radial,
    TbcRamp,
    Sweep,
}

impl TimeDisplaceMapConfig {
    pub fn from_runtime(value: TimeDisplaceMap) -> Self {
        match value {
            TimeDisplaceMap::Ramp => Self::Ramp,
            TimeDisplaceMap::Brightness => Self::Brightness,
            TimeDisplaceMap::Radial => Self::Radial,
            TimeDisplaceMap::TbcRamp => Self::TbcRamp,
            TimeDisplaceMap::Sweep => Self::Sweep,
        }
    }

    pub fn to_runtime(self) -> TimeDisplaceMap {
        match self {
            Self::Ramp => TimeDisplaceMap::Ramp,
            Self::Brightness => TimeDisplaceMap::Brightness,
            Self::Radial => TimeDisplaceMap::Radial,
            Self::TbcRamp => TimeDisplaceMap::TbcRamp,
            Self::Sweep => TimeDisplaceMap::Sweep,
        }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Stable serialized vocabulary for the B3 feedback-rig waveshaper.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackShapeConfig {
    #[default]
    Clamp,
    Soft,
    Wrap,
    Fold,
}

impl FeedbackShapeConfig {
    fn from_runtime(value: FeedbackShape) -> Self {
        match value {
            FeedbackShape::Clamp => Self::Clamp,
            FeedbackShape::Soft => Self::Soft,
            FeedbackShape::Wrap => Self::Wrap,
            FeedbackShape::Fold => Self::Fold,
        }
    }

    fn to_runtime(self) -> FeedbackShape {
        match self {
            Self::Clamp => FeedbackShape::Clamp,
            Self::Soft => FeedbackShape::Soft,
            Self::Wrap => FeedbackShape::Wrap,
            Self::Fold => FeedbackShape::Fold,
        }
    }
}

/// Serializable B3 feedback rig. An omitted section is exactly the historical
/// feedback path, so earlier patches keep their bytes and canonical hashes.
/// The edge law reuses the frozen program-wide boundary vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemporalRigConfig {
    pub offset_x: f32,
    pub offset_y: f32,
    pub reflect_x: bool,
    pub reflect_y: bool,
    pub hue_rotate: f32,
    pub saturation: f32,
    pub gain_r: f32,
    pub gain_g: f32,
    pub gain_b: f32,
    pub chroma_displace: f32,
    pub blur: f32,
    pub sharpen: f32,
    pub shape: FeedbackShapeConfig,
    pub drive: f32,
    pub pivot: f32,
    pub threshold: f32,
    pub noise: f32,
    pub edge: MotionBoundaryModeConfig,
    pub servo: bool,
    pub servo_defeated: bool,
}

impl Default for TemporalRigConfig {
    fn default() -> Self {
        Self::from_params(FeedbackRigParams::default())
    }
}

impl TemporalRigConfig {
    pub fn from_params(value: FeedbackRigParams) -> Self {
        let value = value.sanitized();
        Self {
            offset_x: value.offset_x,
            offset_y: value.offset_y,
            reflect_x: value.reflect_x,
            reflect_y: value.reflect_y,
            hue_rotate: value.hue_rotate,
            saturation: value.saturation,
            gain_r: value.gain_r,
            gain_g: value.gain_g,
            gain_b: value.gain_b,
            chroma_displace: value.chroma_displace,
            blur: value.blur,
            sharpen: value.sharpen,
            shape: FeedbackShapeConfig::from_runtime(value.shape),
            drive: value.drive,
            pivot: value.pivot,
            threshold: value.threshold,
            noise: value.noise,
            edge: MotionBoundaryModeConfig::from_runtime(value.edge),
            servo: value.servo,
            servo_defeated: value.servo_defeated,
        }
    }

    pub fn to_params(self) -> FeedbackRigParams {
        FeedbackRigParams {
            offset_x: self.offset_x,
            offset_y: self.offset_y,
            reflect_x: self.reflect_x,
            reflect_y: self.reflect_y,
            hue_rotate: self.hue_rotate,
            saturation: self.saturation,
            gain_r: self.gain_r,
            gain_g: self.gain_g,
            gain_b: self.gain_b,
            chroma_displace: self.chroma_displace,
            blur: self.blur,
            sharpen: self.sharpen,
            shape: self.shape.to_runtime(),
            drive: self.drive,
            pivot: self.pivot,
            threshold: self.threshold,
            noise: self.noise,
            edge: self.edge.to_runtime(),
            servo: self.servo,
            servo_defeated: self.servo_defeated,
        }
        .sanitized()
    }

    pub fn sanitized(self) -> Self {
        Self::from_params(self.to_params())
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Stable serialized vocabulary for the Temporal Topology Loom.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalTopologyConfig {
    #[default]
    Linear,
    Radial,
    Spiral,
    Contour,
    Folded,
    Kaleidoscopic,
}

impl TemporalTopologyConfig {
    fn from_runtime(value: TemporalTopology) -> Self {
        match value {
            TemporalTopology::Linear => Self::Linear,
            TemporalTopology::Radial => Self::Radial,
            TemporalTopology::Spiral => Self::Spiral,
            TemporalTopology::Contour => Self::Contour,
            TemporalTopology::Folded => Self::Folded,
            TemporalTopology::Kaleidoscopic => Self::Kaleidoscopic,
        }
    }

    fn to_runtime(self) -> TemporalTopology {
        match self {
            Self::Linear => TemporalTopology::Linear,
            Self::Radial => TemporalTopology::Radial,
            Self::Spiral => TemporalTopology::Spiral,
            Self::Contour => TemporalTopology::Contour,
            Self::Folded => TemporalTopology::Folded,
            Self::Kaleidoscopic => TemporalTopology::Kaleidoscopic,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalInterpolationConfig {
    #[default]
    Floor,
    Linear,
}

impl TemporalInterpolationConfig {
    fn from_runtime(value: TemporalInterpolation) -> Self {
        match value {
            TemporalInterpolation::Floor => Self::Floor,
            TemporalInterpolation::Linear => Self::Linear,
        }
    }

    fn to_runtime(self) -> TemporalInterpolation {
        match self {
            Self::Floor => TemporalInterpolation::Floor,
            Self::Linear => TemporalInterpolation::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshGardenGateConfig {
    #[default]
    TemporalDelta,
    Luma,
    Chroma,
    CellularRidge,
    AudioEnergy,
    AudioOnset,
    Matte,
    Motion,
}

impl RefreshGardenGateConfig {
    fn from_runtime(value: RefreshGardenGate) -> Self {
        match value {
            RefreshGardenGate::TemporalDelta => Self::TemporalDelta,
            RefreshGardenGate::Luma => Self::Luma,
            RefreshGardenGate::Chroma => Self::Chroma,
            RefreshGardenGate::CellularRidge => Self::CellularRidge,
            RefreshGardenGate::AudioEnergy => Self::AudioEnergy,
            RefreshGardenGate::AudioOnset => Self::AudioOnset,
            RefreshGardenGate::Matte => Self::Matte,
            RefreshGardenGate::Motion => Self::Motion,
        }
    }

    fn to_runtime(self) -> RefreshGardenGate {
        match self {
            Self::TemporalDelta => RefreshGardenGate::TemporalDelta,
            Self::Luma => RefreshGardenGate::Luma,
            Self::Chroma => RefreshGardenGate::Chroma,
            Self::CellularRidge => RefreshGardenGate::CellularRidge,
            Self::AudioEnergy => RefreshGardenGate::AudioEnergy,
            Self::AudioOnset => RefreshGardenGate::AudioOnset,
            Self::Matte => RefreshGardenGate::Matte,
            Self::Motion => RefreshGardenGate::Motion,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollisionScoreTriggerConfig {
    #[default]
    Boundary,
    Downbeat,
    AudioOnset,
    Manual,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalEventResetModeConfig {
    #[default]
    None,
    Score,
    Memory,
    All,
}

impl TemporalEventResetModeConfig {
    fn from_runtime(value: TemporalEventResetMode) -> Self {
        match value {
            TemporalEventResetMode::None => Self::None,
            TemporalEventResetMode::Score => Self::Score,
            TemporalEventResetMode::Memory => Self::Memory,
            TemporalEventResetMode::All => Self::All,
        }
    }

    fn to_runtime(self) -> TemporalEventResetMode {
        match self {
            Self::None => TemporalEventResetMode::None,
            Self::Score => TemporalEventResetMode::Score,
            Self::Memory => TemporalEventResetMode::Memory,
            Self::All => TemporalEventResetMode::All,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemporalResetPolicyConfig {
    pub loop_boundary: TemporalEventResetModeConfig,
    pub downbeat: TemporalEventResetModeConfig,
}

impl TemporalResetPolicyConfig {
    pub fn from_params(value: TemporalResetPolicy) -> Self {
        Self {
            loop_boundary: TemporalEventResetModeConfig::from_runtime(value.loop_boundary),
            downbeat: TemporalEventResetModeConfig::from_runtime(value.downbeat),
        }
    }

    pub fn to_params(self) -> TemporalResetPolicy {
        TemporalResetPolicy {
            loop_boundary: self.loop_boundary.to_runtime(),
            downbeat: self.downbeat.to_runtime(),
        }
    }
}

impl CollisionScoreTriggerConfig {
    fn from_runtime(value: CollisionScoreTrigger) -> Self {
        match value {
            CollisionScoreTrigger::Boundary => Self::Boundary,
            CollisionScoreTrigger::Downbeat => Self::Downbeat,
            CollisionScoreTrigger::AudioOnset => Self::AudioOnset,
            CollisionScoreTrigger::Manual => Self::Manual,
        }
    }

    fn to_runtime(self) -> CollisionScoreTrigger {
        match self {
            Self::Boundary => CollisionScoreTrigger::Boundary,
            Self::Downbeat => CollisionScoreTrigger::Downbeat,
            Self::AudioOnset => CollisionScoreTrigger::AudioOnset,
            Self::Manual => CollisionScoreTrigger::Manual,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemporalLoomConfig {
    pub amount: f32,
    pub topology: TemporalTopologyConfig,
    pub interpolation: TemporalInterpolationConfig,
    pub depth: f32,
    pub phase: f32,
    pub scale: f32,
    pub angle: f32,
    pub folds: u8,
    pub quantization: u8,
}

impl Default for TemporalLoomConfig {
    fn default() -> Self {
        Self::from_params(TemporalLoomParams::default())
    }
}

impl TemporalLoomConfig {
    pub fn from_params(value: TemporalLoomParams) -> Self {
        Self {
            amount: value.amount,
            topology: TemporalTopologyConfig::from_runtime(value.topology),
            interpolation: TemporalInterpolationConfig::from_runtime(value.interpolation),
            depth: value.depth,
            phase: value.phase,
            scale: value.scale,
            angle: value.angle,
            folds: value.folds,
            quantization: value.quantization,
        }
    }

    pub fn to_params(self) -> TemporalLoomParams {
        TemporalLoomParams {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            topology: self.topology.to_runtime(),
            interpolation: self.interpolation.to_runtime(),
            depth: finite_or(self.depth, 1.0).clamp(0.0, 1.0),
            phase: finite_or(self.phase, 0.0).clamp(-1_000.0, 1_000.0),
            scale: finite_or(self.scale, 1.0).clamp(0.01, 100.0),
            angle: finite_or(self.angle, 0.0).clamp(-180.0, 180.0),
            folds: self.folds.clamp(1, 16),
            quantization: self.quantization.min(24),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CollisionAtlasConfig {
    pub amount: f32,
    pub seed: u32,
    pub territories: u8,
    pub collision: f32,
}

impl Default for CollisionAtlasConfig {
    fn default() -> Self {
        Self::from_params(CollisionAtlasParams::default())
    }
}

impl CollisionAtlasConfig {
    pub fn from_params(value: CollisionAtlasParams) -> Self {
        Self {
            amount: value.amount,
            seed: value.seed,
            territories: value.territories,
            collision: value.collision,
        }
    }

    pub fn to_params(self) -> CollisionAtlasParams {
        CollisionAtlasParams {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            seed: self.seed,
            territories: self.territories.clamp(1, 64),
            collision: finite_or(self.collision, 0.0).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RefreshGardenConfig {
    pub amount: f32,
    pub gate: RefreshGardenGateConfig,
    pub threshold: f32,
    pub softness: f32,
    pub decay: f32,
    pub max_hold_ticks: u32,
    pub matte_route: RefreshGardenMatteRouteConfig,
    pub motion_route: RefreshGardenMotionRouteConfig,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefreshGardenMatteRouteConfig {
    #[default]
    None,
    SelectedLayer {
        saved_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
}

impl RefreshGardenMatteRouteConfig {
    fn from_runtime(value: RefreshGardenMatteRoute) -> Self {
        match value {
            RefreshGardenMatteRoute::None => Self::None,
            RefreshGardenMatteRoute::SelectedLayer {
                saved_position,
                stage,
                ..
            } => Self::SelectedLayer {
                saved_position,
                stage,
            },
            RefreshGardenMatteRoute::MissingSelectedLayer {
                saved_position,
                stage,
            } => Self::MissingSelectedLayer {
                saved_position,
                stage,
            },
        }
    }

    pub(crate) fn from_runtime_for_capture(
        value: RefreshGardenMatteRoute,
        layer_ids: &[StableLayerId],
    ) -> Self {
        match value {
            RefreshGardenMatteRoute::SelectedLayer {
                layer_id,
                saved_position,
                stage,
            } => layer_ids
                .iter()
                .position(|candidate| *candidate == layer_id)
                .and_then(|position| u32::try_from(position).ok())
                .and_then(SavedLayerPosition::new)
                .map_or(
                    Self::MissingSelectedLayer {
                        saved_position,
                        stage,
                    },
                    |saved_position| Self::SelectedLayer {
                        saved_position,
                        stage,
                    },
                ),
            other => Self::from_runtime(other),
        }
    }

    pub(crate) fn resolve_runtime(self, layer_ids: &[StableLayerId]) -> RefreshGardenMatteRoute {
        match self {
            Self::None => RefreshGardenMatteRoute::None,
            Self::SelectedLayer {
                saved_position,
                stage,
            } => saved_position.resolve(layer_ids).copied().map_or(
                RefreshGardenMatteRoute::MissingSelectedLayer {
                    saved_position,
                    stage,
                },
                |layer_id| RefreshGardenMatteRoute::SelectedLayer {
                    layer_id,
                    saved_position,
                    stage,
                },
            ),
            Self::MissingSelectedLayer {
                saved_position,
                stage,
            } => RefreshGardenMatteRoute::MissingSelectedLayer {
                saved_position,
                stage,
            },
        }
    }

    fn unresolved_runtime(self) -> RefreshGardenMatteRoute {
        match self {
            Self::None => RefreshGardenMatteRoute::None,
            Self::SelectedLayer {
                saved_position,
                stage,
            }
            | Self::MissingSelectedLayer {
                saved_position,
                stage,
            } => RefreshGardenMatteRoute::MissingSelectedLayer {
                saved_position,
                stage,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefreshGardenMotionRouteConfig {
    #[default]
    None,
    SelectedLayer {
        saved_position: SavedLayerPosition,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
    },
}

impl RefreshGardenMotionRouteConfig {
    fn from_runtime(value: RefreshGardenMotionRoute) -> Self {
        match value {
            RefreshGardenMotionRoute::None => Self::None,
            RefreshGardenMotionRoute::SelectedLayer { saved_position, .. } => {
                Self::SelectedLayer { saved_position }
            }
            RefreshGardenMotionRoute::MissingSelectedLayer { saved_position } => {
                Self::MissingSelectedLayer { saved_position }
            }
        }
    }

    pub(crate) fn from_runtime_for_capture(
        value: RefreshGardenMotionRoute,
        layer_ids: &[StableLayerId],
    ) -> Self {
        match value {
            RefreshGardenMotionRoute::SelectedLayer {
                layer_id,
                saved_position,
            } => layer_ids
                .iter()
                .position(|candidate| *candidate == layer_id)
                .and_then(|position| u32::try_from(position).ok())
                .and_then(SavedLayerPosition::new)
                .map_or(
                    Self::MissingSelectedLayer { saved_position },
                    |saved_position| Self::SelectedLayer { saved_position },
                ),
            other => Self::from_runtime(other),
        }
    }

    pub(crate) fn resolve_runtime(self, layer_ids: &[StableLayerId]) -> RefreshGardenMotionRoute {
        match self {
            Self::None => RefreshGardenMotionRoute::None,
            Self::SelectedLayer { saved_position } => {
                saved_position.resolve(layer_ids).copied().map_or(
                    RefreshGardenMotionRoute::MissingSelectedLayer { saved_position },
                    |layer_id| RefreshGardenMotionRoute::SelectedLayer {
                        layer_id,
                        saved_position,
                    },
                )
            }
            Self::MissingSelectedLayer { saved_position } => {
                RefreshGardenMotionRoute::MissingSelectedLayer { saved_position }
            }
        }
    }

    fn unresolved_runtime(self) -> RefreshGardenMotionRoute {
        match self {
            Self::None => RefreshGardenMotionRoute::None,
            Self::SelectedLayer { saved_position }
            | Self::MissingSelectedLayer { saved_position } => {
                RefreshGardenMotionRoute::MissingSelectedLayer { saved_position }
            }
        }
    }
}

impl Default for RefreshGardenConfig {
    fn default() -> Self {
        Self::from_params(RefreshGardenParams::default())
    }
}

impl RefreshGardenConfig {
    pub fn from_params(value: RefreshGardenParams) -> Self {
        Self {
            amount: value.amount,
            gate: RefreshGardenGateConfig::from_runtime(value.gate),
            threshold: value.threshold,
            softness: value.softness,
            decay: value.decay,
            max_hold_ticks: value.max_hold_ticks,
            matte_route: RefreshGardenMatteRouteConfig::from_runtime(value.matte_route),
            motion_route: RefreshGardenMotionRouteConfig::from_runtime(value.motion_route),
        }
    }

    pub fn to_params(self) -> RefreshGardenParams {
        RefreshGardenParams {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            gate: self.gate.to_runtime(),
            threshold: finite_or(self.threshold, 0.1).clamp(0.0, 1.0),
            softness: finite_or(self.softness, 0.03).clamp(0.0, 0.5),
            decay: finite_or(self.decay, 1.0).clamp(0.0, 1.0),
            max_hold_ticks: self.max_hold_ticks,
            matte_route: self.matte_route.unresolved_runtime(),
            motion_route: self.motion_route.unresolved_runtime(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CollisionScoreConfig {
    pub enabled: bool,
    pub seed: u32,
    pub state_count: u8,
    pub trigger: CollisionScoreTriggerConfig,
    /// Persisted position only. Live stable IDs are session-local and are
    /// re-resolved by the app after patch/Morph recall.
    #[serde(
        default,
        skip_serializing_if = "CollisionScoreLoopDriverConfig::is_none"
    )]
    pub loop_driver: CollisionScoreLoopDriverConfig,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollisionScoreLoopDriverConfig {
    #[default]
    None,
    SelectedLayer {
        saved_position: SavedLayerPosition,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
    },
}

impl CollisionScoreLoopDriverConfig {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn from_runtime_for_capture(
        value: CollisionScoreLoopDriver,
        layer_ids: &[StableLayerId],
    ) -> Self {
        match value {
            CollisionScoreLoopDriver::None => Self::None,
            CollisionScoreLoopDriver::SelectedLayer {
                layer_id,
                saved_position,
            } => layer_ids
                .iter()
                .position(|candidate| *candidate == layer_id)
                .and_then(|position| u32::try_from(position).ok())
                .and_then(SavedLayerPosition::new)
                .map_or(
                    Self::MissingSelectedLayer { saved_position },
                    |saved_position| Self::SelectedLayer { saved_position },
                ),
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position } => {
                Self::MissingSelectedLayer { saved_position }
            }
        }
    }

    fn resolve_runtime(self, layer_ids: &[StableLayerId]) -> CollisionScoreLoopDriver {
        match self {
            Self::None => CollisionScoreLoopDriver::None,
            Self::SelectedLayer { saved_position } => {
                saved_position.resolve(layer_ids).copied().map_or(
                    CollisionScoreLoopDriver::MissingSelectedLayer { saved_position },
                    |layer_id| CollisionScoreLoopDriver::SelectedLayer {
                        layer_id,
                        saved_position,
                    },
                )
            }
            Self::MissingSelectedLayer { saved_position } => {
                CollisionScoreLoopDriver::MissingSelectedLayer { saved_position }
            }
        }
    }
}

impl Default for CollisionScoreConfig {
    fn default() -> Self {
        Self::from_params(CollisionScoreParams::default())
    }
}

impl CollisionScoreConfig {
    pub fn from_params(value: CollisionScoreParams) -> Self {
        Self {
            enabled: value.enabled,
            seed: value.seed,
            state_count: value.state_count,
            trigger: CollisionScoreTriggerConfig::from_runtime(value.trigger),
            loop_driver: match value.loop_driver {
                CollisionScoreLoopDriver::None => CollisionScoreLoopDriverConfig::None,
                CollisionScoreLoopDriver::SelectedLayer { saved_position, .. } => {
                    CollisionScoreLoopDriverConfig::SelectedLayer { saved_position }
                }
                CollisionScoreLoopDriver::MissingSelectedLayer { saved_position } => {
                    CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position }
                }
            },
        }
    }

    pub fn to_params(self) -> CollisionScoreParams {
        CollisionScoreParams {
            enabled: self.enabled,
            seed: self.seed,
            state_count: self.state_count.clamp(2, 16),
            trigger: self.trigger.to_runtime(),
            loop_driver: match self.loop_driver {
                CollisionScoreLoopDriverConfig::None => CollisionScoreLoopDriver::None,
                CollisionScoreLoopDriverConfig::SelectedLayer { saved_position }
                | CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position } => {
                    // Loading never invents a session identity. The app may
                    // resolve only an authored SelectedLayer marker.
                    CollisionScoreLoopDriver::MissingSelectedLayer { saved_position }
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemporalOriginalsConfig {
    pub loom: TemporalLoomConfig,
    pub atlas: CollisionAtlasConfig,
    pub garden: RefreshGardenConfig,
    pub score: CollisionScoreConfig,
    pub reset: TemporalResetPolicyConfig,
}

impl TemporalOriginalsConfig {
    pub fn from_params(value: TemporalOriginalsParams) -> Self {
        Self {
            loom: TemporalLoomConfig::from_params(value.loom),
            atlas: CollisionAtlasConfig::from_params(value.atlas),
            garden: RefreshGardenConfig::from_params(value.garden),
            score: CollisionScoreConfig::from_params(value.score),
            reset: TemporalResetPolicyConfig::from_params(value.reset),
        }
    }

    pub fn to_params(self) -> TemporalOriginalsParams {
        TemporalOriginalsParams {
            loom: self.loom.to_params(),
            atlas: self.atlas.to_params(),
            garden: self.garden.to_params(),
            score: self.score.to_params(),
            reset: self.reset.to_params(),
        }
    }

    /// Clamp every bounded value without collapsing the persisted distinction
    /// between an authored selected conductor and an inert tombstone. Runtime
    /// conversion cannot carry that distinction until live layer IDs exist.
    pub fn sanitized(self) -> Self {
        let loop_driver = self.score.loop_driver;
        let garden_matte_route = self.garden.matte_route;
        let garden_motion_route = self.garden.motion_route;
        let mut sanitized = Self::from_params(self.to_params());
        sanitized.score.loop_driver = loop_driver;
        sanitized.garden.matte_route = garden_matte_route;
        sanitized.garden.motion_route = garden_motion_route;
        sanitized
    }

    pub fn is_default(&self) -> bool {
        self.to_params() == TemporalOriginalsParams::default()
    }
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self::from_params(&TemporalParams::default())
    }
}

impl TemporalConfig {
    pub fn from_params(p: &TemporalParams) -> Self {
        let originals = TemporalOriginalsConfig::from_params(p.originals);
        Self {
            feedback: p.feedback,
            fb_zoom: p.fb_zoom,
            fb_rotate: p.fb_rotate,
            slitscan: p.slitscan,
            slit_axis: p.slit_axis,
            slit_angle: Some(p.slit_angle),
            slit_map: TimeDisplaceMapConfig::from_runtime(p.slit_map),
            slit_interp: p.slit_interp,
            key_mode: p.key_mode as u32,
            key_threshold: p.key_threshold,
            key_softness: p.key_softness,
            key_history: p.key_history,
            originals: (!originals.is_default()).then_some(originals),
            rig: TemporalRigConfig::from_params(p.rig),
            display: p.display.sanitized(),
            melt: p.melt.sanitized(),
            mosh: p.mosh.sanitized(),
        }
    }

    fn from_params_for_capture(p: &TemporalParams, layer_ids: &[StableLayerId]) -> Self {
        let mut config = Self::from_params(p);
        let matte_route = RefreshGardenMatteRouteConfig::from_runtime_for_capture(
            p.originals.garden.matte_route,
            layer_ids,
        );
        let motion_route = RefreshGardenMotionRouteConfig::from_runtime_for_capture(
            p.originals.garden.motion_route,
            layer_ids,
        );
        if !matches!(matte_route, RefreshGardenMatteRouteConfig::None)
            || !matches!(motion_route, RefreshGardenMotionRouteConfig::None)
        {
            let garden = &mut config
                .originals
                .get_or_insert_with(TemporalOriginalsConfig::default)
                .garden;
            garden.matte_route = matte_route;
            garden.motion_route = motion_route;
        }
        let driver = CollisionScoreLoopDriverConfig::from_runtime_for_capture(
            p.originals.score.loop_driver,
            layer_ids,
        );
        if !driver.is_none() {
            config
                .originals
                .get_or_insert_with(TemporalOriginalsConfig::default)
                .score
                .loop_driver = driver;
        }
        config
    }

    pub fn to_params(&self) -> TemporalParams {
        TemporalParams {
            feedback: finite_or(self.feedback, 0.0).clamp(0.0, 0.95),
            fb_zoom: finite_or(self.fb_zoom, 1.0).clamp(0.9, 1.1),
            fb_rotate: finite_or(self.fb_rotate, 0.0).clamp(-5.0, 5.0),
            slitscan: finite_or(self.slitscan, 0.0).clamp(0.0, 1.0),
            slit_angle: self
                .slit_angle
                .map(|angle| finite_or(angle, 0.0).clamp(-180.0, 180.0))
                .unwrap_or_else(|| finite_or(self.slit_axis, 0.0).clamp(0.0, 1.0) * 90.0),
            slit_axis: finite_or(self.slit_axis, 0.0).clamp(0.0, 1.0),
            slit_map: self.slit_map.to_runtime(),
            slit_interp: self.slit_interp,
            key_mode: self.key_mode.min(4) as f32,
            key_threshold: finite_or(self.key_threshold, 0.1).clamp(0.0, 1.0),
            key_softness: finite_or(self.key_softness, 0.03).clamp(0.0, 0.5),
            key_history: finite_or(self.key_history, 1.0).round().clamp(1.0, 23.0),
            originals: self.originals.unwrap_or_default().to_params(),
            rig: self.rig.to_params(),
            display: self.display.sanitized(),
            melt: self.melt.sanitized(),
            mosh: self.mosh.sanitized(),
        }
    }
}

fn default_motion_algorithm_version() -> u16 {
    MOTION_ALGORITHM_VERSION
}

fn deserialize_motion_algorithm_version<'de, D>(deserializer: D) -> Result<u16, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = u16::deserialize(deserializer)?;
    if version == MOTION_ALGORITHM_VERSION {
        Ok(version)
    } else {
        Err(de::Error::custom(format!(
            "unsupported motion algorithm version {version}; expected {MOTION_ALGORITHM_VERSION}"
        )))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionFieldSourceConfig {
    #[default]
    Auto,
    CodecVectors,
    Lattice,
    ProceduralCurl,
    ProceduralRadial,
    ProceduralSpiral,
    ProceduralContour,
    ProceduralChroma,
    ProceduralWeave,
}

impl MotionFieldSourceConfig {
    fn from_runtime(value: MotionFieldSource) -> Self {
        match value {
            MotionFieldSource::Auto => Self::Auto,
            MotionFieldSource::CodecVectors => Self::CodecVectors,
            MotionFieldSource::Lattice => Self::Lattice,
            MotionFieldSource::Procedural(kind) => match kind {
                ProceduralFieldKind::Curl => Self::ProceduralCurl,
                ProceduralFieldKind::Radial => Self::ProceduralRadial,
                ProceduralFieldKind::Spiral => Self::ProceduralSpiral,
                ProceduralFieldKind::Contour => Self::ProceduralContour,
                ProceduralFieldKind::Chroma => Self::ProceduralChroma,
                ProceduralFieldKind::Weave => Self::ProceduralWeave,
            },
        }
    }

    fn to_runtime(self) -> MotionFieldSource {
        match self {
            Self::Auto => MotionFieldSource::Auto,
            Self::CodecVectors => MotionFieldSource::CodecVectors,
            Self::Lattice => MotionFieldSource::Lattice,
            Self::ProceduralCurl => MotionFieldSource::Procedural(ProceduralFieldKind::Curl),
            Self::ProceduralRadial => MotionFieldSource::Procedural(ProceduralFieldKind::Radial),
            Self::ProceduralSpiral => MotionFieldSource::Procedural(ProceduralFieldKind::Spiral),
            Self::ProceduralContour => MotionFieldSource::Procedural(ProceduralFieldKind::Contour),
            Self::ProceduralChroma => MotionFieldSource::Procedural(ProceduralFieldKind::Chroma),
            Self::ProceduralWeave => MotionFieldSource::Procedural(ProceduralFieldKind::Weave),
        }
    }
}

/// Serializable B2 procedural field scalars. An omitted section is exactly the
/// pre-B2 path, so every patch written before B2 keeps its original bytes and
/// its original canonical hash.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProceduralFieldConfig {
    pub scale: f32,
    pub rate: f32,
}

impl Default for ProceduralFieldConfig {
    fn default() -> Self {
        Self::from_params(ProceduralFieldParams::default())
    }
}

impl ProceduralFieldConfig {
    pub fn from_params(value: ProceduralFieldParams) -> Self {
        Self {
            scale: value.scale,
            rate: value.rate,
        }
    }

    pub fn to_params(self) -> ProceduralFieldParams {
        ProceduralFieldParams {
            scale: self.scale,
            rate: self.rate,
        }
        .sanitized()
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Serializable B2 flow-shaping controls. An omitted section is exactly the
/// pre-shaping advection path, so earlier patches keep their bytes and
/// canonical hashes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlowShapingConfig {
    pub stretch: f32,
    pub edge_repel: f32,
    pub vector_trash: f32,
    pub trash_block_size: f32,
}

impl Default for FlowShapingConfig {
    fn default() -> Self {
        Self::from_params(FlowShapingParams::default())
    }
}

impl FlowShapingConfig {
    pub fn from_params(value: FlowShapingParams) -> Self {
        Self {
            stretch: value.stretch,
            edge_repel: value.edge_repel,
            vector_trash: value.vector_trash,
            trash_block_size: value.trash_block_size,
        }
    }

    pub fn to_params(self) -> FlowShapingParams {
        FlowShapingParams {
            stretch: self.stretch,
            edge_repel: self.edge_repel,
            vector_trash: self.vector_trash,
            trash_block_size: self.trash_block_size,
        }
        .sanitized()
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionLatticeQualityConfig {
    Draft,
    #[default]
    Live,
    High,
}

impl MotionLatticeQualityConfig {
    fn from_runtime(value: MotionLatticeQuality) -> Self {
        match value {
            MotionLatticeQuality::Draft => Self::Draft,
            MotionLatticeQuality::Live => Self::Live,
            MotionLatticeQuality::High => Self::High,
        }
    }

    fn to_runtime(self) -> MotionLatticeQuality {
        match self {
            Self::Draft => MotionLatticeQuality::Draft,
            Self::Live => MotionLatticeQuality::Live,
            Self::High => MotionLatticeQuality::High,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionCarrierConfig {
    #[default]
    Transparent,
    Black,
    FirstSourceFrame,
}

impl MotionCarrierConfig {
    fn from_runtime(value: MotionCarrier) -> Self {
        match value {
            MotionCarrier::Transparent => Self::Transparent,
            MotionCarrier::Black => Self::Black,
            MotionCarrier::FirstSourceFrame => Self::FirstSourceFrame,
        }
    }

    fn to_runtime(self) -> MotionCarrier {
        match self {
            Self::Transparent => MotionCarrier::Transparent,
            Self::Black => MotionCarrier::Black,
            Self::FirstSourceFrame => MotionCarrier::FirstSourceFrame,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MotionDonorConfig {
    #[default]
    None,
    Selected {
        saved_position: SavedLayerPosition,
    },
    Missing {
        saved_position: SavedLayerPosition,
    },
}

impl MotionDonorConfig {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    fn from_runtime_cached(value: MotionDonor) -> Self {
        match value {
            MotionDonor::None => Self::None,
            MotionDonor::Selected { saved_position, .. } => Self::Selected { saved_position },
            MotionDonor::Missing { saved_position } => Self::Missing { saved_position },
        }
    }

    pub(crate) fn from_runtime_for_capture(
        value: MotionDonor,
        layer_ids: &[StableLayerId],
    ) -> Self {
        match value {
            MotionDonor::None => Self::None,
            MotionDonor::Selected {
                layer_id,
                saved_position,
            } => layer_ids
                .iter()
                .position(|candidate| *candidate == layer_id)
                .and_then(|position| u32::try_from(position).ok())
                .and_then(SavedLayerPosition::new)
                .map_or(Self::Missing { saved_position }, |saved_position| {
                    Self::Selected { saved_position }
                }),
            MotionDonor::Missing { saved_position } => Self::Missing { saved_position },
        }
    }

    pub(crate) fn resolve_runtime(self, layer_ids: &[StableLayerId]) -> MotionDonor {
        match self {
            Self::None => MotionDonor::None,
            Self::Selected { saved_position } => saved_position.resolve(layer_ids).copied().map_or(
                MotionDonor::Missing { saved_position },
                |layer_id| MotionDonor::Selected {
                    layer_id,
                    saved_position,
                },
            ),
            Self::Missing { saved_position } => MotionDonor::Missing { saved_position },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FaradayConfig {
    pub amount: f32,
    #[serde(default, skip_serializing_if = "MotionDonorConfig::is_none")]
    pub donor: MotionDonorConfig,
    pub carrier: MotionCarrierConfig,
    pub confidence_threshold: f32,
    pub confidence_softness: f32,
    pub refresh: f32,
    pub decay: f32,
    pub occlusion: f32,
}

impl Default for FaradayConfig {
    fn default() -> Self {
        Self::from_params(FaradayParams::default())
    }
}

impl FaradayConfig {
    pub fn from_params(value: FaradayParams) -> Self {
        Self {
            amount: value.amount,
            donor: MotionDonorConfig::from_runtime_cached(value.donor),
            carrier: MotionCarrierConfig::from_runtime(value.carrier),
            confidence_threshold: value.confidence_threshold,
            confidence_softness: value.confidence_softness,
            refresh: value.refresh,
            decay: value.decay,
            occlusion: value.occlusion,
        }
    }

    pub fn to_params(self) -> FaradayParams {
        FaradayParams {
            amount: finite_or(self.amount, 0.0).clamp(0.0, 1.0),
            donor: match self.donor {
                MotionDonorConfig::None => MotionDonor::None,
                MotionDonorConfig::Selected { saved_position }
                | MotionDonorConfig::Missing { saved_position } => {
                    MotionDonor::Missing { saved_position }
                }
            },
            carrier: self.carrier.to_runtime(),
            confidence_threshold: finite_or(self.confidence_threshold, 0.1).clamp(0.0, 1.0),
            confidence_softness: finite_or(self.confidence_softness, 0.05).clamp(0.0, 0.5),
            refresh: finite_or(self.refresh, 1.0).clamp(0.0, 1.0),
            decay: finite_or(self.decay, 1.0).clamp(0.0, 1.0),
            occlusion: finite_or(self.occlusion, 0.0).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurvedShutterQualityConfig {
    #[default]
    Sharp,
    Draft,
    Live,
    High,
}

impl CurvedShutterQualityConfig {
    fn from_runtime(value: CurvedShutterQuality) -> Self {
        match value {
            CurvedShutterQuality::Sharp => Self::Sharp,
            CurvedShutterQuality::Draft => Self::Draft,
            CurvedShutterQuality::Live => Self::Live,
            CurvedShutterQuality::High => Self::High,
        }
    }

    fn to_runtime(self) -> CurvedShutterQuality {
        match self {
            Self::Sharp => CurvedShutterQuality::Sharp,
            Self::Draft => CurvedShutterQuality::Draft,
            Self::Live => CurvedShutterQuality::Live,
            Self::High => CurvedShutterQuality::High,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CurvedShutterConfig {
    pub angle_degrees: f32,
    pub phase: f32,
    pub curvature: f32,
    pub chromatic_lag: f32,
    pub quality: CurvedShutterQualityConfig,
}

impl Default for CurvedShutterConfig {
    fn default() -> Self {
        Self::from_params(CurvedShutterParams::default())
    }
}

impl CurvedShutterConfig {
    pub fn from_params(value: CurvedShutterParams) -> Self {
        Self {
            angle_degrees: value.angle_degrees,
            phase: value.phase,
            curvature: value.curvature,
            chromatic_lag: value.chromatic_lag,
            quality: CurvedShutterQualityConfig::from_runtime(value.quality),
        }
    }

    pub fn to_params(self) -> CurvedShutterParams {
        CurvedShutterParams {
            angle_degrees: finite_or(self.angle_degrees, 0.0).clamp(0.0, 360.0),
            phase: finite_or(self.phase, 0.0).clamp(-1.0, 1.0),
            curvature: finite_or(self.curvature, 0.0).clamp(-2.0, 2.0),
            chromatic_lag: finite_or(self.chromatic_lag, 0.0).clamp(0.0, 1.0),
            quality: self.quality.to_runtime(),
        }
    }
}

/// Serializable Field Collider v1 recombination law.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldColliderModeConfig {
    #[default]
    Sum,
    Difference,
    Curl,
    Projection,
    CollisionBoundary,
}

impl FieldColliderModeConfig {
    fn from_runtime(value: FieldColliderMode) -> Self {
        match value {
            FieldColliderMode::Sum => Self::Sum,
            FieldColliderMode::Difference => Self::Difference,
            FieldColliderMode::Curl => Self::Curl,
            FieldColliderMode::Projection => Self::Projection,
            FieldColliderMode::CollisionBoundary => Self::CollisionBoundary,
        }
    }

    fn to_runtime(self) -> FieldColliderMode {
        match self {
            Self::Sum => FieldColliderMode::Sum,
            Self::Difference => FieldColliderMode::Difference,
            Self::Curl => FieldColliderMode::Curl,
            Self::Projection => FieldColliderMode::Projection,
            Self::CollisionBoundary => FieldColliderMode::CollisionBoundary,
        }
    }
}

/// Serializable motion-field boundary law.
///
/// The token order here is the frozen shader-code order shared with the two
/// image boundaries, not the prose order of the enrichment plan; see
/// [`MotionBoundaryMode`] for why motion deliberately does not mint its own
/// numbering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionBoundaryModeConfig {
    #[default]
    Transparent,
    Mirror,
    Wrap,
    Hold,
}

impl MotionBoundaryModeConfig {
    fn from_runtime(value: MotionBoundaryMode) -> Self {
        match value {
            MotionBoundaryMode::Transparent => Self::Transparent,
            MotionBoundaryMode::Mirror => Self::Mirror,
            MotionBoundaryMode::Wrap => Self::Wrap,
            MotionBoundaryMode::Hold => Self::Hold,
        }
    }

    fn to_runtime(self) -> MotionBoundaryMode {
        match self {
            Self::Transparent => MotionBoundaryMode::Transparent,
            Self::Mirror => MotionBoundaryMode::Mirror,
            Self::Wrap => MotionBoundaryMode::Wrap,
            Self::Hold => MotionBoundaryMode::Hold,
        }
    }
}

/// Serializable S5 Field Collider authoring.
///
/// This carries the strict version, the two discrete laws, and the two saved
/// donor identities — and nothing else. Derived vectors, the transient mapped
/// pair, gate parities, admitted field slots, runtime diagnostics, and
/// process-local stable IDs are all deliberately absent: they are frame-local
/// executor state, not authored topology. Version 1 adds no collider-only
/// continuous control, so there is no scalar here to interpolate or dice.
///
/// Both inputs are independent NAMED fields rather than a list, because slot
/// identity is route identity: a positional container would make input B's
/// meaning depend on whether input A happened to be authored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldColliderConfig {
    #[serde(
        default = "default_field_collider_algorithm_version",
        deserialize_with = "deserialize_field_collider_algorithm_version"
    )]
    pub algorithm_version: u16,
    pub enabled: bool,
    pub mode: FieldColliderModeConfig,
    pub boundary: MotionBoundaryModeConfig,
    #[serde(default, skip_serializing_if = "MotionDonorConfig::is_none")]
    pub input_a: MotionDonorConfig,
    #[serde(default, skip_serializing_if = "MotionDonorConfig::is_none")]
    pub input_b: MotionDonorConfig,
}

const fn default_field_collider_algorithm_version() -> u16 {
    FIELD_COLLIDER_ALGORITHM_VERSION
}

/// A patch declaring any other collider version is rejected at deserialize
/// time rather than migrated, exactly as the motion algorithm version is.
fn deserialize_field_collider_algorithm_version<'de, D>(deserializer: D) -> Result<u16, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u16::deserialize(deserializer)?;
    if value == FIELD_COLLIDER_ALGORITHM_VERSION {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(format!(
            "unsupported field collider algorithm version {value}; expected \
             {FIELD_COLLIDER_ALGORITHM_VERSION}"
        )))
    }
}

impl Default for FieldColliderConfig {
    fn default() -> Self {
        Self::from_params(FieldColliderParams::default())
    }
}

impl FieldColliderConfig {
    pub fn from_params(value: FieldColliderParams) -> Self {
        Self {
            algorithm_version: value.algorithm_version,
            enabled: value.enabled,
            mode: FieldColliderModeConfig::from_runtime(value.mode),
            boundary: MotionBoundaryModeConfig::from_runtime(value.boundary),
            input_a: MotionDonorConfig::from_runtime_cached(value.input_a),
            input_b: MotionDonorConfig::from_runtime_cached(value.input_b),
        }
    }

    /// Both donors collapse to `Missing`, exactly as the transplant's does:
    /// resolution is a separate, later step through `resolve_runtime` once a
    /// complete live layer stack exists.
    pub fn to_params(self) -> FieldColliderParams {
        let donor = |config: MotionDonorConfig| match config {
            MotionDonorConfig::None => MotionDonor::None,
            MotionDonorConfig::Selected { saved_position }
            | MotionDonorConfig::Missing { saved_position } => {
                MotionDonor::Missing { saved_position }
            }
        };
        FieldColliderParams {
            algorithm_version: FIELD_COLLIDER_ALGORITHM_VERSION,
            enabled: self.enabled,
            mode: self.mode.to_runtime(),
            boundary: self.boundary.to_runtime(),
            input_a: donor(self.input_a),
            input_b: donor(self.input_b),
        }
        .sanitized()
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Serializable M4 authoring. Runtime fields, carrier pixels, codec records,
/// telemetry, and process-local stable IDs are deliberately absent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MotionConfig {
    #[serde(
        default = "default_motion_algorithm_version",
        deserialize_with = "deserialize_motion_algorithm_version"
    )]
    pub algorithm_version: u16,
    pub field_source: MotionFieldSourceConfig,
    pub lattice_quality: MotionLatticeQualityConfig,
    /// B2 procedural field scalars. Skipped at default so pre-B2 patches keep
    /// their bytes and canonical hashes.
    #[serde(default, skip_serializing_if = "ProceduralFieldConfig::is_default")]
    pub procedural: ProceduralFieldConfig,
    /// B2 flow shaping. Skipped at default so earlier patches keep their
    /// bytes and canonical hashes.
    #[serde(default, skip_serializing_if = "FlowShapingConfig::is_default")]
    pub shaping: FlowShapingConfig,
    pub transplant: FaradayConfig,
    pub shutter: CurvedShutterConfig,
    /// Field Collider v1. An omitted section is exactly the pre-collider path,
    /// so every patch written before S5 keeps its original bytes and its
    /// original canonical hash.
    #[serde(default, skip_serializing_if = "FieldColliderConfig::is_default")]
    pub collider: FieldColliderConfig,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self::from_params(MotionParams::default())
    }
}

impl MotionConfig {
    pub fn from_params(value: MotionParams) -> Self {
        Self {
            algorithm_version: value.algorithm_version,
            field_source: MotionFieldSourceConfig::from_runtime(value.field_source),
            lattice_quality: MotionLatticeQualityConfig::from_runtime(value.lattice_quality),
            procedural: ProceduralFieldConfig::from_params(value.procedural),
            shaping: FlowShapingConfig::from_params(value.shaping),
            transplant: FaradayConfig::from_params(value.transplant),
            shutter: CurvedShutterConfig::from_params(value.shutter),
            collider: FieldColliderConfig::from_params(value.collider),
        }
    }

    pub fn from_params_for_capture(value: MotionParams, layer_ids: &[StableLayerId]) -> Self {
        let mut config = Self::from_params(value);
        config.transplant.donor =
            MotionDonorConfig::from_runtime_for_capture(value.transplant.donor, layer_ids);
        // Each collider slot recomputes its own saved position independently.
        // Slot identity is a named field, so clearing input A can never slide
        // input B's donor down into A's place.
        config.collider.input_a =
            MotionDonorConfig::from_runtime_for_capture(value.collider.input_a, layer_ids);
        config.collider.input_b =
            MotionDonorConfig::from_runtime_for_capture(value.collider.input_b, layer_ids);
        config
    }

    pub fn to_params(self) -> MotionParams {
        MotionParams {
            algorithm_version: if self.algorithm_version == MOTION_ALGORITHM_VERSION {
                self.algorithm_version
            } else {
                MOTION_ALGORITHM_VERSION
            },
            field_source: self.field_source.to_runtime(),
            lattice_quality: self.lattice_quality.to_runtime(),
            procedural: self.procedural.to_params(),
            shaping: self.shaping.to_params(),
            transplant: self.transplant.to_params(),
            shutter: self.shutter.to_params(),
            collider: self.collider.to_params(),
        }
    }

    pub(crate) fn resolve_runtime(self, layer_ids: &[StableLayerId]) -> MotionParams {
        let mut params = self.to_params().sanitized();
        params.transplant.donor = self.transplant.donor.resolve_runtime(layer_ids);
        params.collider.input_a = self.collider.input_a.resolve_runtime(layer_ids);
        params.collider.input_b = self.collider.input_b.resolve_runtime(layer_ids);
        params
    }

    /// Clamp continuous values while retaining Selected versus Missing donor
    /// intent until a complete live layer stack is available.
    pub fn sanitized(self) -> Self {
        let donor = self.transplant.donor;
        let (input_a, input_b) = (self.collider.input_a, self.collider.input_b);
        let mut sanitized = Self::from_params(self.to_params().sanitized());
        sanitized.transplant.donor = donor;
        // Both collider tombstones survive the sanitize for the same reason the
        // transplant's does: flattening Missing to None here would silently
        // rebind a dead slot the next time a layer occupied that position.
        sanitized.collider.input_a = input_a;
        sanitized.collider.input_b = input_b;
        sanitized
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

fn apply_motion_look(config: MotionConfig, live: &mut MotionParams) {
    let donor = live.transplant.donor;
    let (input_a, input_b) = (live.collider.input_a, live.collider.input_b);
    *live = config.to_params().sanitized();
    // A Look transfers bounded values and recipe laws, never donor-route
    // topology. The collider's enabled flag, mode, and boundary are recipe and
    // do travel; both of its inputs are topology and stay live.
    live.transplant.donor = donor;
    live.collider.input_a = input_a;
    live.collider.input_b = input_b;
}

/// Serializable S3b gesture-canvas authoring.
///
/// This carries the three *authored continuous* canvas controls and nothing
/// else. Etched pixels, the decay clock, the grid the host derived from the
/// current output, the canvas generation, and the recorded event track are all
/// deliberately absent: the first four are runtime state and the last is
/// topology carried whole by [`PatchState::gesture_track`]. An omitted section
/// is exactly the pre-gesture path, so every patch written before S3b keeps its
/// original bytes and its original canonical hash.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GestureCanvasConfig {
    pub radius: f32,
    pub strength: f32,
    pub retention: f32,
}

impl Default for GestureCanvasConfig {
    fn default() -> Self {
        Self::from_params(crate::gesture_canvas::GestureCanvasParams::default())
    }
}

impl GestureCanvasConfig {
    pub const fn from_params(value: crate::gesture_canvas::GestureCanvasParams) -> Self {
        Self {
            radius: value.radius,
            strength: value.strength,
            retention: value.retention,
        }
    }

    /// Every consumer sanitizes, so a hostile or legacy field lands on the
    /// documented default rather than a clamped extreme.
    pub fn to_params(self) -> crate::gesture_canvas::GestureCanvasParams {
        crate::gesture_canvas::GestureCanvasParams {
            radius: self.radius,
            strength: self.strength,
            retention: self.retention,
        }
        .sanitized()
    }

    pub fn sanitized(self) -> Self {
        Self::from_params(self.to_params())
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Serializable modulation matrix state for patch files.
#[derive(Serialize, Deserialize, Clone)]
pub struct ModConfig {
    #[serde(default = "default_bpm")]
    pub bpm: f32,
    #[serde(default)]
    pub lfos: Vec<LfoConfig>,
    #[serde(default)]
    pub routings: Vec<RoutingConfig>,
    #[serde(default)]
    pub audio_enabled: bool,
    #[serde(default = "one")]
    pub audio_gain: f32,
    #[serde(default)]
    pub audio_device: String,
    #[serde(default = "default_audio_source_kind")]
    pub audio_source_kind: String,
    #[serde(default)]
    pub audio_clip_path: String,
    /// Number of routable FFT bands. Older patches omit this and remain at
    /// the historical three-band layout.
    #[serde(default = "default_audio_band_count")]
    pub audio_band_count: usize,
    /// Ordered crossovers. Current patches store exactly count - 1 entries.
    /// Legacy patches stored `[bass, mid, analysis_ceiling]`; application
    /// migrates that third value into `audio_band_ceiling_hz`.
    #[serde(default = "default_audio_band_edges")]
    pub audio_band_edges: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_band_ceiling_hz: Option<f32>,
    #[serde(default)]
    pub midi_enabled: bool,
    #[serde(default = "default_midi_ccs")]
    pub midi_ccs: Vec<u8>,
    #[serde(default)]
    pub midi_clock_sync: bool,
    #[serde(default = "default_gyro_axes")]
    pub gyro: Vec<GyroAxisPatchConfig>,
    /// Latest DeviceOrientation sample in degrees. Older patches omit this;
    /// those load centered on their saved calibration instead of inventing
    /// an offset from a zero-valued sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gyro_raw: Option<Vec<f32>>,
    #[serde(default)]
    pub pad: PadPatchConfig,
    /// Saved XY gesture position. A loaded/exported patch owns no live
    /// pointer, so spring return (when enabled) resumes from this position.
    #[serde(default = "default_pad_position")]
    pub pad_position: Vec<f32>,
    /// B10 envelope configurations. Skip-serialized while all four sit at
    /// their defaults so pre-B10 patches keep their bytes and canonical
    /// hashes; an emitted section always carries all four slots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub envelopes: Vec<EnvelopeConfig>,
    /// B10 macro knob values, skip-serialized while all are zero.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub macros: Vec<f32>,
    /// B10 deterministic-generator seed; zero (the default) is omitted.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub generator_seed: u32,
}

/// One B10 envelope's authored configuration. Bend held state is
/// deliberately absent: a patch can never restore a held pad.
#[derive(Serialize, Deserialize, Clone, PartialEq)]
pub struct EnvelopeConfig {
    #[serde(default = "default_envelope_attack")]
    pub attack: f32,
    #[serde(default = "default_envelope_decay")]
    pub decay: f32,
    #[serde(default = "default_envelope_trigger")]
    pub trigger: String,
    #[serde(default = "default_envelope_mode")]
    pub mode: String,
}

fn default_envelope_attack() -> f32 {
    0.02
}
fn default_envelope_decay() -> f32 {
    0.5
}
fn default_envelope_trigger() -> String {
    "bend1".to_string()
}
fn default_envelope_mode() -> String {
    "once".to_string()
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        Self {
            attack: default_envelope_attack(),
            decay: default_envelope_decay(),
            trigger: default_envelope_trigger(),
            mode: default_envelope_mode(),
        }
    }
}

impl EnvelopeConfig {
    fn from_generator(envelope: &crate::modulation::EnvelopeGen) -> Self {
        Self {
            attack: envelope.attack,
            decay: envelope.decay,
            trigger: envelope.trigger.as_str().to_string(),
            mode: envelope.mode.as_str().to_string(),
        }
    }

    /// Apply onto one generator slot. Unknown trigger/mode tokens keep the
    /// slot's defaults rather than guessing; scalars sanitize to the
    /// documented ranges.
    fn apply_to(&self, envelope: &mut crate::modulation::EnvelopeGen) {
        envelope.attack = self.attack;
        envelope.decay = self.decay;
        if let Some(trigger) = crate::modulation::EnvelopeTrigger::try_from_str(&self.trigger) {
            envelope.trigger = trigger;
        }
        if let Some(mode) = crate::modulation::EnvelopeMode::try_from_str(&self.mode) {
            envelope.mode = mode;
        }
        envelope.sanitize();
    }
}

fn default_midi_ccs() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

fn default_bpm() -> f32 {
    120.0
}

fn default_audio_band_edges() -> Vec<f32> {
    vec![250.0, 2000.0, 8000.0]
}

fn default_audio_band_count() -> usize {
    3
}

fn default_audio_source_kind() -> String {
    crate::modulation::AUDIO_SOURCE_LIVE.to_string()
}

fn four() -> f32 {
    4.0
}

fn default_gyro_range() -> f32 {
    90.0
}

fn default_gyro_axes() -> Vec<GyroAxisPatchConfig> {
    vec![
        GyroAxisPatchConfig {
            range_degrees: 180.0,
            ..Default::default()
        },
        GyroAxisPatchConfig::default(),
        GyroAxisPatchConfig::default(),
    ]
}

fn default_pad_position() -> Vec<f32> {
    vec![0.5, 0.5]
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LfoConfig {
    #[serde(default = "default_shape")]
    pub shape: String,
    #[serde(default = "default_beats")]
    pub beats: f32,
    #[serde(default)]
    pub phase: f32,
    #[serde(default)]
    pub seed: u32,
}

fn default_shape() -> String {
    "sine".to_string()
}
fn default_beats() -> f32 {
    4.0
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RoutingConfig {
    pub source: String,
    pub target: String,
    /// Typed saved-position target for M2 racks/groups. The legacy string is
    /// retained for inspectability, but this field is authoritative and never
    /// contains a process-local stable layer ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_target: Option<SavedStableModTarget>,
    #[serde(default)]
    pub depth: f32,
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    #[serde(default)]
    pub attack: f32,
    #[serde(default)]
    pub release: f32,
}

fn default_curve() -> String {
    "linear".to_string()
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GyroAxisPatchConfig {
    #[serde(default)]
    pub center_degrees: f32,
    #[serde(default = "default_gyro_range")]
    pub range_degrees: f32,
    #[serde(default)]
    pub expo: f32,
    #[serde(default)]
    pub invert: bool,
}

impl Default for GyroAxisPatchConfig {
    fn default() -> Self {
        Self {
            center_degrees: 0.0,
            range_degrees: 90.0,
            expo: 0.0,
            invert: false,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PadAxisPatchConfig {
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    #[serde(default)]
    pub quantize: u32,
}

impl Default for PadAxisPatchConfig {
    fn default() -> Self {
        Self {
            curve: default_curve(),
            curve_amount: 0.0,
            quantize: 0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PadPatchConfig {
    #[serde(default)]
    pub x: PadAxisPatchConfig,
    #[serde(default)]
    pub y: PadAxisPatchConfig,
    #[serde(default)]
    pub spring_enabled: bool,
    #[serde(default = "four")]
    pub spring_rate: f32,
}

impl Default for PadPatchConfig {
    fn default() -> Self {
        Self {
            x: PadAxisPatchConfig::default(),
            y: PadAxisPatchConfig::default(),
            spring_enabled: false,
            spring_rate: 4.0,
        }
    }
}

impl ModConfig {
    pub fn from_matrix(m: &ModMatrix) -> Self {
        Self {
            bpm: m.clock.bpm,
            lfos: m
                .lfos
                .iter()
                .map(|l| LfoConfig {
                    shape: l.shape.as_str().to_string(),
                    beats: l.beats,
                    phase: l.normalized_phase(),
                    seed: l.seed,
                })
                .collect(),
            routings: m
                .routings
                .iter()
                .map(|r| RoutingConfig {
                    source: r.source.as_str().to_string(),
                    target: r.target().to_owned(),
                    stable_target: None,
                    depth: r.depth,
                    curve: r.curve.as_str().to_string(),
                    curve_amount: r.curve_amount,
                    attack: r.attack,
                    release: r.release,
                })
                .collect(),
            audio_enabled: m.audio_enabled,
            audio_gain: m.audio_gain,
            audio_device: m.audio_device.clone(),
            audio_source_kind: crate::modulation::normalize_audio_source_kind(&m.audio_source_kind)
                .to_string(),
            audio_clip_path: m
                .audio_clip_source_reference
                .clone()
                .unwrap_or_else(|| m.audio_clip_path.clone()),
            audio_band_count: m.audio_band_config.count(),
            audio_band_edges: m.audio_band_config.crossovers().to_vec(),
            audio_band_ceiling_hz: Some(m.audio_band_config.ceiling_hz()),
            midi_enabled: m.midi_enabled,
            midi_ccs: m.midi_ccs.to_vec(),
            midi_clock_sync: m.midi_clock_sync,
            gyro: m
                .gyro_config
                .iter()
                .map(|cfg| GyroAxisPatchConfig {
                    center_degrees: cfg.center_degrees,
                    range_degrees: cfg.range_degrees,
                    expo: cfg.expo,
                    invert: cfg.invert,
                })
                .collect(),
            gyro_raw: Some(m.gyro_raw.to_vec()),
            pad: PadPatchConfig {
                x: PadAxisPatchConfig::from_axis(m.pad_config.axes[0]),
                y: PadAxisPatchConfig::from_axis(m.pad_config.axes[1]),
                spring_enabled: m.pad_config.spring_enabled,
                spring_rate: m.pad_config.spring_rate,
            },
            pad_position: m.pad.to_vec(),
            envelopes: {
                // Emit the section only once any slot moved off its default,
                // so pre-B10 patches keep their bytes and canonical hashes.
                let captured: Vec<EnvelopeConfig> = m
                    .envelopes
                    .iter()
                    .map(EnvelopeConfig::from_generator)
                    .collect();
                if captured
                    .iter()
                    .all(|slot| *slot == EnvelopeConfig::default())
                {
                    Vec::new()
                } else {
                    captured
                }
            },
            macros: if m.macros.iter().all(|value| *value == 0.0) {
                Vec::new()
            } else {
                m.macros.to_vec()
            },
            generator_seed: m.generator_seed,
        }
    }

    /// Capture stable rack/group targets through the saved-position boundary.
    /// Legacy target strings keep their historical byte representation; only
    /// typed M2 targets receive the authoritative `stable_target` field.
    pub fn from_matrix_with_composition(
        m: &ModMatrix,
        book: &StableModAddressBook,
        layer_ids: &[StableLayerId],
        composition: &RuntimeComposition,
    ) -> Result<Self, String> {
        let mut captured = Self::from_matrix(m);
        for (config, routing) in captured.routings.iter_mut().zip(&m.routings) {
            let saved = if let Some(missing) = routing.saved_missing_target() {
                Some(missing)
            } else if let Some(target) = routing.stable_target() {
                Some(target.capture(
                    book,
                    |wanted| {
                        layer_ids
                            .iter()
                            .position(|candidate| *candidate == wanted)
                            .and_then(|position| u32::try_from(position).ok())
                            .and_then(SavedLayerPosition::new)
                    },
                    |group_id| composition.group(group_id).is_some(),
                )?)
            } else {
                None
            };
            if let Some(saved) = saved {
                config.target = saved.persistence_key();
                config.stable_target = Some(saved);
            }
        }
        Ok(captured)
    }

    pub fn apply_to_matrix(&self, m: &mut ModMatrix) {
        m.clock.set_bpm(finite_or(self.bpm, 120.0));
        m.lfos = std::array::from_fn(|_| Lfo::default());
        for (i, cfg) in self.lfos.iter().take(NUM_LFOS).enumerate() {
            let mut lfo = Lfo {
                shape: LfoShape::from_str(&cfg.shape),
                beats: finite_or(cfg.beats, 4.0).clamp(0.0625, 64.0),
                phase: 0.0,
                seed: cfg.seed,
            };
            lfo.set_phase(cfg.phase);
            m.lfos[i] = lfo;
        }
        // If a transitional patch happens to contain both spellings, the
        // canonical route wins and the legacy alias is ignored rather than
        // applying the same semantic destination twice.
        let canonical_key_targets: HashSet<String> = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .filter(|routing| {
                routing.target.starts_with("layer") && routing.target.ends_with("_key_threshold")
            })
            .map(|routing| routing.target.clone())
            .collect();
        m.routings = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .filter_map(|r| {
                let source = ModSource::try_from_str(&r.source)?;
                let target = crate::modulation::canonical_target(&r.target);
                if target.as_ref() != r.target && canonical_key_targets.contains(target.as_ref()) {
                    return None;
                }
                if !crate::modulation::is_valid_target(target.as_ref()) {
                    return None;
                }
                let mut routing = Routing::new(
                    source,
                    target.as_ref(),
                    finite_or(r.depth, 0.0).clamp(-1.0, 1.0),
                );
                routing.curve = Curve::from_str(&r.curve);
                routing.curve_amount = finite_or(r.curve_amount, 0.0).clamp(-2.0, 2.0);
                routing.attack = finite_or(r.attack, 0.0).clamp(0.0, 10.0);
                routing.release = finite_or(r.release, 0.0).clamp(0.0, 10.0);
                Some(routing)
            })
            .collect();
        m.audio_enabled = self.audio_enabled;
        m.audio_gain = finite_or(self.audio_gain, 1.0).clamp(0.0, 8.0);
        m.audio_device = self.audio_device.clone();
        m.audio_source_kind =
            crate::modulation::normalize_audio_source_kind(&self.audio_source_kind).to_string();
        m.audio_clip_source_reference =
            crate::media_source::parse_content_reference(&self.audio_clip_path)
                .ok()
                .flatten()
                .map(|_| self.audio_clip_path.clone());
        m.audio_clip_path = self.audio_clip_path.clone();
        let count = self
            .audio_band_count
            .clamp(crate::audio::MIN_AUDIO_BANDS, crate::audio::MAX_AUDIO_BANDS);
        let (crossovers, ceiling_hz) = match self.audio_band_ceiling_hz {
            Some(ceiling) => (self.audio_band_edges.as_slice(), ceiling),
            None if self.audio_band_edges.len() >= count => (
                &self.audio_band_edges[..count - 1],
                self.audio_band_edges[count - 1],
            ),
            None => (self.audio_band_edges.as_slice(), 8000.0),
        };
        m.audio_band_config = crate::audio::AudioBandConfig::new(count, crossovers, ceiling_hz);
        m.midi_enabled = self.midi_enabled;
        m.midi_ccs = [1, 2, 3, 4];
        for (i, &cc) in self.midi_ccs.iter().take(m.midi_ccs.len()).enumerate() {
            m.midi_ccs[i] = cc & 0x7F;
        }
        m.midi_clock_sync = self.midi_clock_sync;

        let defaults = default_gyro_axes();
        for i in 0..3 {
            let cfg = self.gyro.get(i).or_else(|| defaults.get(i)).unwrap();
            m.gyro_config[i] = GyroAxisConfig {
                center_degrees: finite_or(cfg.center_degrees, 0.0),
                range_degrees: finite_or(cfg.range_degrees, 90.0).abs().clamp(1.0, 360.0),
                expo: finite_or(cfg.expo, 0.0).clamp(-2.0, 2.0),
                invert: cfg.invert,
            };
        }
        for i in 0..3 {
            m.gyro_raw[i] = self
                .gyro_raw
                .as_ref()
                .and_then(|values| values.get(i))
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(m.gyro_config[i].center_degrees);
        }
        m.pad_config = PadConfig {
            axes: [self.pad.x.to_axis(), self.pad.y.to_axis()],
            spring_enabled: self.pad.spring_enabled,
            spring_rate: finite_or(self.pad.spring_rate, 4.0).clamp(0.1, 20.0),
        };
        for i in 0..2 {
            m.pad[i] = self
                .pad_position
                .get(i)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
        }
        // A saved patch has no owning browser pointer. Marking it released is
        // what lets deterministic spring return advance in live and export.
        m.pad_active = false;
        // B10 authored sources. An absent section is exactly the pre-B10
        // path: default envelopes, zero macros, seed zero. Runtime state
        // (bend holds, envelope levels, the generator clocks) resets below so
        // the restored program replays deterministically from its own zero.
        m.envelopes = std::array::from_fn(|_| crate::modulation::EnvelopeGen::default());
        for (slot, config) in m
            .envelopes
            .iter_mut()
            .zip(self.envelopes.iter().take(crate::modulation::NUM_ENVELOPES))
        {
            config.apply_to(slot);
        }
        m.macros = [0.0; crate::modulation::NUM_MACROS];
        for (slot, value) in m
            .macros
            .iter_mut()
            .zip(self.macros.iter().take(crate::modulation::NUM_MACROS))
        {
            *slot = finite_or(*value, 0.0).clamp(0.0, 1.0);
        }
        m.generator_seed = self.generator_seed;
        m.reset_performance_sources();
        m.recompute_gyro();
    }

    /// Resolve typed saved targets against the current runtime identities.
    /// Missing layers/groups/nodes remain explicit inert routes, so loading a
    /// patch can never silently retarget an identity that was deleted.
    pub fn apply_to_matrix_with_composition(
        &self,
        m: &mut ModMatrix,
        book: &StableModAddressBook,
        layer_ids: &[StableLayerId],
        composition: &RuntimeComposition,
    ) {
        self.apply_to_matrix(m);
        let canonical_key_targets: HashSet<String> = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .filter(|routing| {
                routing.stable_target.is_none()
                    && routing.target.starts_with("layer")
                    && routing.target.ends_with("_key_threshold")
            })
            .map(|routing| routing.target.clone())
            .collect();
        m.routings = self
            .routings
            .iter()
            .take(MAX_ROUTINGS)
            .filter_map(|config| {
                let source = ModSource::try_from_str(&config.source)?;
                let mut routing = if let Some(saved) = config.stable_target {
                    match saved.resolve(
                        book,
                        |position| position.resolve(layer_ids).copied(),
                        |group_id| composition.group(group_id).is_some(),
                    ) {
                        ResolvedStableModTarget::Live(target) => Routing::new(
                            source,
                            target.to_string(),
                            finite_or(config.depth, 0.0).clamp(-1.0, 1.0),
                        ),
                        ResolvedStableModTarget::Missing(missing) => Routing::new_missing(
                            source,
                            missing,
                            finite_or(config.depth, 0.0).clamp(-1.0, 1.0),
                        ),
                    }
                } else {
                    let target = crate::modulation::canonical_target(&config.target);
                    if target.as_ref() != config.target
                        && canonical_key_targets.contains(target.as_ref())
                    {
                        return None;
                    }
                    if !crate::modulation::is_valid_target(target.as_ref()) {
                        return None;
                    }
                    Routing::new(
                        source,
                        target.as_ref(),
                        finite_or(config.depth, 0.0).clamp(-1.0, 1.0),
                    )
                };
                routing.curve = Curve::from_str(&config.curve);
                routing.curve_amount = finite_or(config.curve_amount, 0.0).clamp(-2.0, 2.0);
                routing.attack = finite_or(config.attack, 0.0).clamp(0.0, 10.0);
                routing.release = finite_or(config.release, 0.0).clamp(0.0, 10.0);
                Some(routing)
            })
            .collect();
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

impl PadAxisPatchConfig {
    fn from_axis(axis: PadAxisConfig) -> Self {
        Self {
            curve: axis.curve.as_str().to_string(),
            curve_amount: axis.curve_amount,
            quantize: axis.quantize,
        }
    }

    fn to_axis(&self) -> PadAxisConfig {
        PadAxisConfig {
            curve: Curve::from_str(&self.curve),
            curve_amount: finite_or(self.curve_amount, 0.0).clamp(-2.0, 2.0),
            quantize: self.quantize.min(64),
        }
    }
}

/// Serializable NTSC/VHS effect parameters for patch files.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct NtscConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub tape_speed: u32,
    #[serde(default)]
    pub chroma_loss: f32,
    #[serde(default)]
    pub edge_wave_enabled: bool,
    #[serde(default)]
    pub edge_wave_intensity: f32,
    #[serde(default = "default_edge_wave_speed")]
    pub edge_wave_speed: f32,
    #[serde(default)]
    pub head_switching_enabled: bool,
    #[serde(default = "default_head_height")]
    pub head_switching_height: i32,
    #[serde(default)]
    pub head_switching_shift: f32,
    #[serde(default)]
    pub tracking_noise_enabled: bool,
    #[serde(default = "default_tracking_height")]
    pub tracking_noise_height: i32,
    #[serde(default)]
    pub tracking_noise_wave: f32,
    #[serde(default)]
    pub tracking_noise_snow: f32,
    #[serde(default)]
    pub snow_intensity: f32,
    #[serde(default)]
    pub composite_noise_intensity: f32,
    #[serde(default)]
    pub luma_noise_intensity: f32,
    #[serde(default)]
    pub chroma_noise_intensity: f32,
    #[serde(default)]
    pub luma_smear: f32,
    #[serde(default)]
    pub composite_sharpening: f32,
}

fn default_edge_wave_speed() -> f32 {
    0.5
}
fn default_head_height() -> i32 {
    8
}
fn default_tracking_height() -> i32 {
    24
}

impl NtscConfig {
    pub fn from_params(p: &NtscParams) -> Self {
        Self {
            enabled: p.enabled,
            tape_speed: p.tape_speed,
            chroma_loss: p.chroma_loss,
            edge_wave_enabled: p.edge_wave_enabled,
            edge_wave_intensity: p.edge_wave_intensity,
            edge_wave_speed: p.edge_wave_speed,
            head_switching_enabled: p.head_switching_enabled,
            head_switching_height: p.head_switching_height,
            head_switching_shift: p.head_switching_shift,
            tracking_noise_enabled: p.tracking_noise_enabled,
            tracking_noise_height: p.tracking_noise_height,
            tracking_noise_wave: p.tracking_noise_wave,
            tracking_noise_snow: p.tracking_noise_snow,
            snow_intensity: p.snow_intensity,
            composite_noise_intensity: p.composite_noise_intensity,
            luma_noise_intensity: p.luma_noise_intensity,
            chroma_noise_intensity: p.chroma_noise_intensity,
            luma_smear: p.luma_smear,
            composite_sharpening: p.composite_sharpening,
        }
    }

    pub fn to_params(&self) -> NtscParams {
        let finite = |value: f32, fallback: f32| {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        };
        NtscParams {
            enabled: self.enabled,
            tape_speed: self.tape_speed.min(2),
            chroma_loss: finite(self.chroma_loss, 0.0).clamp(0.0, 0.01),
            edge_wave_enabled: self.edge_wave_enabled,
            edge_wave_intensity: finite(self.edge_wave_intensity, 0.0).clamp(0.0, 20.0),
            edge_wave_speed: finite(self.edge_wave_speed, 0.5).clamp(0.0, 10.0),
            head_switching_enabled: self.head_switching_enabled,
            head_switching_height: self.head_switching_height.clamp(0, 24),
            head_switching_shift: finite(self.head_switching_shift, 0.0).clamp(-100.0, 100.0),
            tracking_noise_enabled: self.tracking_noise_enabled,
            tracking_noise_height: self.tracking_noise_height.clamp(0, 120),
            tracking_noise_wave: finite(self.tracking_noise_wave, 0.0).clamp(0.0, 50.0),
            tracking_noise_snow: finite(self.tracking_noise_snow, 0.0).clamp(0.0, 1.0),
            snow_intensity: finite(self.snow_intensity, 0.0).clamp(0.0, 1.0),
            composite_noise_intensity: finite(self.composite_noise_intensity, 0.0).clamp(0.0, 0.5),
            luma_noise_intensity: finite(self.luma_noise_intensity, 0.0).clamp(0.0, 0.2),
            chroma_noise_intensity: finite(self.chroma_noise_intensity, 0.0).clamp(0.0, 0.5),
            luma_smear: finite(self.luma_smear, 0.0).clamp(0.0, 1.0),
            composite_sharpening: finite(self.composite_sharpening, 0.0).clamp(-1.0, 2.0),
        }
    }
}

#[derive(Serialize, Clone)]
pub struct LayerConfig {
    pub filename: String,
    /// Stable identity for sources loaded outside the current library. File
    /// layers store their canonical path; live receivers use
    /// `spout://<sender-name>`. Old patches omit this and continue resolving
    /// video sources by filename.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_path: String,
    #[serde(default = "one")]
    pub opacity: f32,
    #[serde(default = "default_blend")]
    pub blend_mode: String,
    #[serde(default = "one")]
    pub speed: f32,
    #[serde(default = "default_fps")]
    pub fps: f32,
    #[serde(default)]
    pub paused: bool,
    #[serde(default = "default_true")]
    pub visible: bool,
    /// Skip the shared master shader for this layer. Missing in legacy
    /// patches means the historical behavior: master effects remain active.
    #[serde(default)]
    pub bypass_master_fx: bool,
    #[serde(default)]
    pub reroll_on_loop: bool,
    #[serde(default)]
    pub effects: EffectsConfig,
    /// Per-layer authored geometry. Missing data in old patches must resolve
    /// to the exact inactive historical identity. Once that state is moved,
    /// newly exposed canvas starts transparent.
    #[serde(default, skip_serializing_if = "is_legacy_spatial_identity")]
    pub transform: SpatialTransform,
    /// Omitted is the exact pre-M4 no-op. The nested donor stores a saved
    /// position only; live stable IDs never cross this boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion: Option<MotionConfig>,
    /// Omitted synthesizes the frozen layer legacy marker; Some(empty) is an
    /// explicitly authored pass-through rack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rack: Option<VisualRack>,
    /// Canonical prepared-source representation. Legacy mirrors above remain
    /// for older readers and are synthesized into slot 1 on old patch load.
    pub clip_slots: ClipSlots,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_clip_slot: Option<ClipSlotId>,
    /// Saved-position matte DTO; runtime IDs are resolved only after layers exist.
    #[serde(default, skip_serializing_if = "LayerMatteConfig::is_legacy_disabled")]
    pub matte: LayerMatteConfig,
    /// B7 pattern-synth authored state; `Some` iff `source_path` is the
    /// `synth://pattern` sentinel. The whole identity — every value the
    /// picture depends on — lives here, so offline reconstruction is
    /// perfect with no file, no content reference, no placeholder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<PatternSynthConfig>,
    /// B7 text-page authored state; `Some` iff `source_path` is the
    /// `text://page` sentinel. Same self-containment law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_page: Option<TextPageConfig>,
}

fn default_blend() -> String {
    "normal".to_string()
}
fn default_true() -> bool {
    true
}

impl<'de> Deserialize<'de> for LayerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawLayerConfig {
            filename: String,
            #[serde(default)]
            source_path: String,
            #[serde(default = "one")]
            opacity: f32,
            #[serde(default = "default_blend")]
            blend_mode: String,
            #[serde(default = "one")]
            speed: f32,
            #[serde(default = "default_fps")]
            fps: f32,
            #[serde(default)]
            paused: bool,
            #[serde(default = "default_true")]
            visible: bool,
            #[serde(default)]
            bypass_master_fx: bool,
            #[serde(default)]
            reroll_on_loop: bool,
            #[serde(default)]
            effects: EffectsConfig,
            #[serde(default)]
            transform: SpatialTransform,
            #[serde(default)]
            motion: Option<MotionConfig>,
            #[serde(default)]
            rack: Option<VisualRack>,
            /// `None` means the field was absent and must take the exact legacy
            /// one-source migration. An explicit empty array remains empty.
            #[serde(default)]
            clip_slots: Option<ClipSlots>,
            #[serde(default)]
            active_clip_slot: Option<ClipSlotId>,
            #[serde(default)]
            matte: LayerMatteConfig,
            #[serde(default)]
            pattern: Option<PatternSynthConfig>,
            #[serde(default)]
            text_page: Option<TextPageConfig>,
        }

        let raw = RawLayerConfig::deserialize(deserializer)?;
        if let Some(rack) = &raw.rack {
            rack.validate_for_scope(LegacyRackScope::Layer)
                .map_err(de::Error::custom)?;
        }
        let clip_slots = raw.clip_slots.unwrap_or_else(|| {
            ClipSlots::singleton(ClipSlotConfig::from_legacy(
                raw.filename.clone(),
                raw.source_path.clone(),
                raw.speed,
                raw.fps,
            ))
        });
        let active_clip_slot = clip_slots.active_or_first(raw.active_clip_slot);
        let mut config = Self {
            filename: raw.filename,
            source_path: raw.source_path,
            opacity: raw.opacity,
            blend_mode: raw.blend_mode,
            speed: raw.speed,
            fps: raw.fps,
            paused: raw.paused,
            visible: raw.visible,
            bypass_master_fx: raw.bypass_master_fx,
            reroll_on_loop: raw.reroll_on_loop,
            effects: raw.effects,
            transform: raw.transform,
            motion: raw.motion.map(MotionConfig::sanitized),
            rack: raw.rack,
            clip_slots,
            active_clip_slot,
            matte: raw.matte.sanitized(),
            // Hostile scalars sanitize to their neutral values on load; the
            // closed token vocabularies were already rejected by serde.
            pattern: raw.pattern.map(PatternSynthConfig::sanitized),
            text_page: raw.text_page.map(TextPageConfig::sanitized),
        };
        config.sync_legacy_mirrors_from_active_slot();
        Ok(config)
    }
}

/// Closed pattern-shape token vocabulary. An unknown token is a
/// deserialization rejection, never a silent default.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PatternShapeConfig {
    #[default]
    Scan,
    Radial,
    Spiral,
    Plasma,
    Lissajous,
    Rings,
    Starburst,
    Grid,
    Tunnel,
    Cells,
    Interference,
    Polygon,
}

impl PatternShapeConfig {
    pub fn to_runtime(self) -> crate::pattern_synth::PatternShape {
        use crate::pattern_synth::PatternShape as S;
        match self {
            Self::Scan => S::Scan,
            Self::Radial => S::Radial,
            Self::Spiral => S::Spiral,
            Self::Plasma => S::Plasma,
            Self::Lissajous => S::Lissajous,
            Self::Rings => S::Rings,
            Self::Starburst => S::Starburst,
            Self::Grid => S::Grid,
            Self::Tunnel => S::Tunnel,
            Self::Cells => S::Cells,
            Self::Interference => S::Interference,
            Self::Polygon => S::Polygon,
        }
    }

    pub fn from_runtime(value: crate::pattern_synth::PatternShape) -> Self {
        use crate::pattern_synth::PatternShape as S;
        match value {
            S::Scan => Self::Scan,
            S::Radial => Self::Radial,
            S::Spiral => Self::Spiral,
            S::Plasma => Self::Plasma,
            S::Lissajous => Self::Lissajous,
            S::Rings => Self::Rings,
            S::Starburst => Self::Starburst,
            S::Grid => Self::Grid,
            S::Tunnel => Self::Tunnel,
            S::Cells => Self::Cells,
            S::Interference => Self::Interference,
            S::Polygon => Self::Polygon,
        }
    }
}

/// Closed oscillator-waveform token vocabulary.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PatternWaveConfig {
    #[default]
    Sine,
    Triangle,
    Saw,
    Square,
    Pulse,
    SampleHold,
}

impl PatternWaveConfig {
    pub fn to_runtime(self) -> crate::pattern_synth::PatternWave {
        use crate::pattern_synth::PatternWave as W;
        match self {
            Self::Sine => W::Sine,
            Self::Triangle => W::Triangle,
            Self::Saw => W::Saw,
            Self::Square => W::Square,
            Self::Pulse => W::Pulse,
            Self::SampleHold => W::SampleHold,
        }
    }

    pub fn from_runtime(value: crate::pattern_synth::PatternWave) -> Self {
        use crate::pattern_synth::PatternWave as W;
        match value {
            W::Sine => Self::Sine,
            W::Triangle => Self::Triangle,
            W::Saw => Self::Saw,
            W::Square => Self::Square,
            W::Pulse => Self::Pulse,
            W::SampleHold => Self::SampleHold,
        }
    }
}

/// Closed colouriser token vocabulary.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PatternColorModeConfig {
    Mono,
    #[default]
    RgbPhase,
    HsvSweep,
    Duotone,
    Bands,
}

impl PatternColorModeConfig {
    pub fn to_runtime(self) -> crate::pattern_synth::PatternColorMode {
        use crate::pattern_synth::PatternColorMode as C;
        match self {
            Self::Mono => C::Mono,
            Self::RgbPhase => C::RgbPhase,
            Self::HsvSweep => C::HsvSweep,
            Self::Duotone => C::Duotone,
            Self::Bands => C::Bands,
        }
    }

    pub fn from_runtime(value: crate::pattern_synth::PatternColorMode) -> Self {
        use crate::pattern_synth::PatternColorMode as C;
        match value {
            C::Mono => Self::Mono,
            C::RgbPhase => Self::RgbPhase,
            C::HsvSweep => Self::HsvSweep,
            C::Duotone => Self::Duotone,
            C::Bands => Self::Bands,
        }
    }
}

/// The B7 pattern-synth patch DTO: three closed vocabularies plus the
/// twenty-two continuous values, byte-mirroring the runtime law's ranges.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct PatternSynthConfig {
    pub shape: PatternShapeConfig,
    pub wave: PatternWaveConfig,
    pub color_mode: PatternColorModeConfig,
    pub freq_x: f32,
    pub freq_y: f32,
    pub phase: f32,
    pub rate: f32,
    pub cross_mod: f32,
    pub wavefold: f32,
    pub pulse_width: f32,
    pub comparator: f32,
    pub comp_threshold: f32,
    pub comp_soft: f32,
    pub symmetry: f32,
    pub zoom: f32,
    pub rotate: f32,
    pub skew: f32,
    pub center_x: f32,
    pub center_y: f32,
    pub warp: f32,
    pub hue: f32,
    pub hue_spread: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub color_bands: f32,
}

impl Default for PatternSynthConfig {
    fn default() -> Self {
        Self::from_params(&crate::pattern_synth::PatternSynthParams::default())
    }
}

impl PatternSynthConfig {
    pub fn from_params(params: &crate::pattern_synth::PatternSynthParams) -> Self {
        let q = params.sanitized();
        Self {
            shape: PatternShapeConfig::from_runtime(q.shape),
            wave: PatternWaveConfig::from_runtime(q.wave),
            color_mode: PatternColorModeConfig::from_runtime(q.color_mode),
            freq_x: q.freq_x,
            freq_y: q.freq_y,
            phase: q.phase,
            rate: q.rate,
            cross_mod: q.cross_mod,
            wavefold: q.wavefold,
            pulse_width: q.pulse_width,
            comparator: q.comparator,
            comp_threshold: q.comp_threshold,
            comp_soft: q.comp_soft,
            symmetry: q.symmetry,
            zoom: q.zoom,
            rotate: q.rotate,
            skew: q.skew,
            center_x: q.center_x,
            center_y: q.center_y,
            warp: q.warp,
            hue: q.hue,
            hue_spread: q.hue_spread,
            saturation: q.saturation,
            brightness: q.brightness,
            color_bands: q.color_bands,
        }
    }

    pub fn to_params(self) -> crate::pattern_synth::PatternSynthParams {
        crate::pattern_synth::PatternSynthParams {
            shape: self.shape.to_runtime(),
            wave: self.wave.to_runtime(),
            color_mode: self.color_mode.to_runtime(),
            freq_x: self.freq_x,
            freq_y: self.freq_y,
            phase: self.phase,
            rate: self.rate,
            cross_mod: self.cross_mod,
            wavefold: self.wavefold,
            pulse_width: self.pulse_width,
            comparator: self.comparator,
            comp_threshold: self.comp_threshold,
            comp_soft: self.comp_soft,
            symmetry: self.symmetry,
            zoom: self.zoom,
            rotate: self.rotate,
            skew: self.skew,
            center_x: self.center_x,
            center_y: self.center_y,
            warp: self.warp,
            hue: self.hue,
            hue_spread: self.hue_spread,
            saturation: self.saturation,
            brightness: self.brightness,
            color_bands: self.color_bands,
        }
        .sanitized()
    }

    pub fn sanitized(self) -> Self {
        Self::from_params(&self.to_params())
    }
}

/// Closed text-page face token vocabulary — the two bundled licensed faces.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TextPageFontConfig {
    #[default]
    Mono,
    Sans,
}

impl TextPageFontConfig {
    pub fn to_runtime(self) -> crate::text_page::TextPageFont {
        match self {
            Self::Mono => crate::text_page::TextPageFont::Mono,
            Self::Sans => crate::text_page::TextPageFont::Sans,
        }
    }

    pub fn from_runtime(value: crate::text_page::TextPageFont) -> Self {
        match value {
            crate::text_page::TextPageFont::Mono => Self::Mono,
            crate::text_page::TextPageFont::Sans => Self::Sans,
        }
    }
}

/// Closed text-page shape-fan token vocabulary.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TextPageShapeConfig {
    #[default]
    None,
    Circle,
    Ring,
    Rect,
    Tri,
    Cross,
    Bars,
    Grid,
    Rings,
    Starburst,
}

impl TextPageShapeConfig {
    pub fn to_runtime(self) -> crate::text_page::TextPageShape {
        use crate::text_page::TextPageShape as S;
        match self {
            Self::None => S::None,
            Self::Circle => S::Circle,
            Self::Ring => S::Ring,
            Self::Rect => S::Rect,
            Self::Tri => S::Tri,
            Self::Cross => S::Cross,
            Self::Bars => S::Bars,
            Self::Grid => S::Grid,
            Self::Rings => S::Rings,
            Self::Starburst => S::Starburst,
        }
    }

    pub fn from_runtime(value: crate::text_page::TextPageShape) -> Self {
        use crate::text_page::TextPageShape as S;
        match value {
            S::None => Self::None,
            S::Circle => Self::Circle,
            S::Ring => Self::Ring,
            S::Rect => Self::Rect,
            S::Tri => Self::Tri,
            S::Cross => Self::Cross,
            S::Bars => Self::Bars,
            S::Grid => Self::Grid,
            S::Rings => Self::Rings,
            S::Starburst => Self::Starburst,
        }
    }
}

/// The B7 text-page patch DTO. Everything the raster depends on — the body,
/// the face, the layout, the shape fan — travels here, so a patch is
/// self-contained and the offline raster is byte-identical to the live one.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(deny_unknown_fields, default)]
pub struct TextPageConfig {
    pub body: String,
    pub font: TextPageFontConfig,
    pub size: f32,
    pub track: f32,
    pub x: f32,
    pub y: f32,
    pub rot_degrees: f32,
    pub repeat: u32,
    pub outline: f32,
    pub ink: [f32; 3],
    pub bg: [f32; 3],
    pub shape: TextPageShapeConfig,
    pub shape_count: u32,
    pub shape_size: f32,
    pub shape_x: f32,
    pub shape_y: f32,
    pub shape_fill: [f32; 3],
    pub shape_stroke: f32,
}

impl Default for TextPageConfig {
    fn default() -> Self {
        Self::from_params(&crate::text_page::TextPageParams::default())
    }
}

impl TextPageConfig {
    pub fn from_params(params: &crate::text_page::TextPageParams) -> Self {
        let q = params.sanitized();
        Self {
            body: q.body,
            font: TextPageFontConfig::from_runtime(q.font),
            size: q.size,
            track: q.track,
            x: q.x,
            y: q.y,
            rot_degrees: q.rot_degrees,
            repeat: q.repeat,
            outline: q.outline,
            ink: q.ink,
            bg: q.bg,
            shape: TextPageShapeConfig::from_runtime(q.shape),
            shape_count: q.shape_count,
            shape_size: q.shape_size,
            shape_x: q.shape_x,
            shape_y: q.shape_y,
            shape_fill: q.shape_fill,
            shape_stroke: q.shape_stroke,
        }
    }

    pub fn to_params(&self) -> crate::text_page::TextPageParams {
        crate::text_page::TextPageParams {
            body: self.body.clone(),
            font: self.font.to_runtime(),
            size: self.size,
            track: self.track,
            x: self.x,
            y: self.y,
            rot_degrees: self.rot_degrees,
            repeat: self.repeat,
            outline: self.outline,
            ink: self.ink,
            bg: self.bg,
            shape: self.shape.to_runtime(),
            shape_count: self.shape_count,
            shape_size: self.shape_size,
            shape_x: self.shape_x,
            shape_y: self.shape_y,
            shape_fill: self.shape_fill,
            shape_stroke: self.shape_stroke,
        }
        .sanitized()
    }

    pub fn sanitized(self) -> Self {
        Self::from_params(&self.to_params())
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EffectsConfig {
    /// Deterministic shader pattern seed. Zero is the exact legacy sequence.
    #[serde(default)]
    pub random_seed: u32,
    #[serde(default = "one")]
    pub pixelate: f32,
    #[serde(default)]
    pub rgb_split: f32,
    #[serde(default)]
    pub hue_shift: f32,
    #[serde(default)]
    pub saturation: f32,
    #[serde(default)]
    pub brightness: f32,
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub posterize: f32,
    #[serde(default)]
    pub invert: bool,
    /// Fraction of full render resolution (1.0 = full resolution).
    #[serde(default = "one")]
    pub downsample: f32,
    #[serde(default)]
    pub shift_amount: f32,
    #[serde(default = "default_shift_block_size")]
    pub shift_block_size: f32,
    #[serde(default = "default_shift_density")]
    pub shift_density: f32,
    #[serde(default = "default_shift_speed")]
    pub shift_speed: f32,
    #[serde(default)]
    pub grain_intensity: f32,
    #[serde(default = "one")]
    pub grain_size: f32,
    #[serde(default)]
    pub grain_algo: u32,
    #[serde(default)]
    pub color_grain: bool,
    #[serde(default)]
    pub breathe_scale: f32,
    #[serde(default)]
    pub breathe_rotation: f32,
    #[serde(default)]
    pub breathe_position: f32,
    #[serde(default)]
    pub vignette: f32,
    #[serde(default)]
    pub color_drift: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_key_softness")]
    pub key_softness: f32,
    #[serde(default = "default_key_color")]
    pub key_color: [f32; 3],
    #[serde(default = "default_key_tolerance")]
    pub key_tolerance: f32,
    #[serde(default)]
    pub cellular_amount: f32,
    #[serde(default = "default_cellular_scale")]
    pub cellular_scale: f32,
    #[serde(default = "default_cellular_warp")]
    pub cellular_warp: f32,
    #[serde(default = "default_cellular_speed")]
    pub cellular_speed: f32,
    #[serde(default)]
    pub cellular_gap_amount: f32,
    #[serde(default = "default_cellular_gap_threshold")]
    pub cellular_gap_threshold: f32,
    #[serde(default = "default_cellular_gap_softness")]
    pub cellular_gap_softness: f32,
    // B13 small effects. Skip-serialized at their defaults so pre-B13
    // patches keep their bytes and canonical hashes.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub contour: f32,
    #[serde(
        default = "default_contour_bands",
        skip_serializing_if = "is_default_contour_bands"
    )]
    pub contour_bands: f32,
    #[serde(
        default = "default_contour_width",
        skip_serializing_if = "is_default_contour_width"
    )]
    pub contour_width: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub contour_hue: f32,
    #[serde(
        default = "default_contour_fill",
        skip_serializing_if = "is_default_contour_fill"
    )]
    pub contour_fill: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub flatten: f32,
    #[serde(
        default = "default_flatten_levels",
        skip_serializing_if = "is_default_flatten_levels"
    )]
    pub flatten_levels: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub contour_dither: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub solarize: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub negative: f32,
    /// Permanent codes: 0 = rgb, 1 = luma-only, 2 = hue-flip.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub negative_mode: u32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub colourpass: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub colourpass_hue: f32,
    #[serde(
        default = "default_colourpass_width",
        skip_serializing_if = "is_default_colourpass_width"
    )]
    pub colourpass_width: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub edge_amount: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub edge_hue: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub emboss: f32,
    #[serde(
        default = "default_emboss_angle",
        skip_serializing_if = "is_default_emboss_angle"
    )]
    pub emboss_angle: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub halftone: f32,
    #[serde(
        default = "default_halftone_pitch",
        skip_serializing_if = "is_default_halftone_pitch"
    )]
    pub halftone_pitch: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub halftone_angle: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub moire: f32,
    #[serde(
        default = "default_moire_freq",
        skip_serializing_if = "is_default_moire_freq"
    )]
    pub moire_freq: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub row_smear: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub bitcrush: f32,
    #[serde(
        default = "default_bitcrush_levels",
        skip_serializing_if = "is_default_bitcrush_levels"
    )]
    pub bitcrush_levels: f32,
    #[serde(
        default = "default_bitcrush_dither",
        skip_serializing_if = "is_default_bitcrush_dither"
    )]
    pub bitcrush_dither: f32,
    #[serde(default = "one", skip_serializing_if = "is_one_f32")]
    pub multi_grid_x: f32,
    #[serde(default = "one", skip_serializing_if = "is_one_f32")]
    pub multi_grid_y: f32,
    /// Master-only optics; a layer's copies stay at their defaults.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub barrel: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub chroma_aberration: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub anamorphic_streak: f32,
    /// B8 key dressing: border and shadow join the key signal.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub key_border: f32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub key_border_color: u32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub key_shadow: f32,
}

fn default_key_threshold() -> f32 {
    0.5
}
fn default_key_softness() -> f32 {
    0.1
}

impl Default for EffectsConfig {
    fn default() -> Self {
        Self {
            random_seed: 0,
            pixelate: 1.0,
            rgb_split: 0.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: false,
            downsample: 1.0,
            shift_amount: 0.0,
            shift_block_size: 8.0,
            shift_density: 0.5,
            shift_speed: 3.0,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0,
            color_grain: false,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            vignette: 0.0,
            color_drift: 0.0,
            key_mode: 0,
            key_threshold: 0.5,
            key_softness: 0.1,
            key_color: default_key_color(),
            key_tolerance: default_key_tolerance(),
            cellular_amount: 0.0,
            cellular_scale: 10.0,
            cellular_warp: 0.35,
            cellular_speed: 0.25,
            cellular_gap_amount: 0.0,
            cellular_gap_threshold: 0.65,
            cellular_gap_softness: 0.08,
            contour: 0.0,
            contour_bands: default_contour_bands(),
            contour_width: default_contour_width(),
            contour_hue: 0.0,
            contour_fill: default_contour_fill(),
            flatten: 0.0,
            flatten_levels: default_flatten_levels(),
            contour_dither: 0.0,
            solarize: 0.0,
            negative: 0.0,
            negative_mode: 0,
            colourpass: 0.0,
            colourpass_hue: 0.0,
            colourpass_width: default_colourpass_width(),
            edge_amount: 0.0,
            edge_hue: 0.0,
            emboss: 0.0,
            emboss_angle: default_emboss_angle(),
            halftone: 0.0,
            halftone_pitch: default_halftone_pitch(),
            halftone_angle: 0.0,
            moire: 0.0,
            moire_freq: default_moire_freq(),
            row_smear: 0.0,
            bitcrush: 0.0,
            bitcrush_levels: default_bitcrush_levels(),
            bitcrush_dither: default_bitcrush_dither(),
            multi_grid_x: 1.0,
            multi_grid_y: 1.0,
            barrel: 0.0,
            chroma_aberration: 0.0,
            anamorphic_streak: 0.0,
            key_border: 0.0,
            key_border_color: 0,
            key_shadow: 0.0,
        }
    }
}

// --- Conversion: EffectUniforms <-> EffectsConfig ---

impl EffectsConfig {
    pub fn from_uniforms(u: &EffectUniforms) -> Self {
        Self {
            random_seed: u.random_seed,
            pixelate: u.pixelate_size,
            rgb_split: u.rgb_split,
            hue_shift: u.hue_shift,
            saturation: u.saturation,
            brightness: u.brightness,
            contrast: u.contrast,
            posterize: u.posterize,
            invert: u.invert > 0.5,
            downsample: u.downsample,
            shift_amount: u.shift_amount,
            shift_block_size: u.shift_block_size,
            shift_density: u.shift_density,
            shift_speed: u.shift_speed,
            grain_intensity: u.grain_intensity,
            grain_size: u.grain_size,
            grain_algo: u.grain_algo as u32,
            color_grain: u.color_grain > 0.5,
            breathe_scale: u.breathe_scale,
            breathe_rotation: u.breathe_rotation,
            breathe_position: u.breathe_position,
            vignette: u.vignette,
            color_drift: u.color_drift,
            key_mode: u.key_mode as u32,
            key_threshold: u.key_threshold,
            key_softness: u.key_softness,
            key_color: u.key_color,
            key_tolerance: u.key_tolerance,
            cellular_amount: u.cellular_amount,
            cellular_scale: u.cellular_scale,
            cellular_warp: u.cellular_warp,
            cellular_speed: u.cellular_speed,
            cellular_gap_amount: u.cellular_gap_amount,
            cellular_gap_threshold: u.cellular_gap_threshold,
            cellular_gap_softness: u.cellular_gap_softness,
            contour: u.contour,
            contour_bands: u.contour_bands,
            contour_width: u.contour_width,
            contour_hue: u.contour_hue,
            contour_fill: u.contour_fill,
            flatten: u.flatten,
            flatten_levels: u.flatten_levels,
            contour_dither: u.contour_dither,
            solarize: u.solarize,
            negative: u.negative,
            negative_mode: u.negative_mode as u32,
            colourpass: u.colourpass,
            colourpass_hue: u.colourpass_hue,
            colourpass_width: u.colourpass_width,
            edge_amount: u.edge_amount,
            edge_hue: u.edge_hue,
            emboss: u.emboss,
            emboss_angle: u.emboss_angle,
            halftone: u.halftone,
            halftone_pitch: u.halftone_pitch,
            halftone_angle: u.halftone_angle,
            moire: u.moire,
            moire_freq: u.moire_freq,
            row_smear: u.row_smear,
            bitcrush: u.bitcrush,
            bitcrush_levels: u.bitcrush_levels,
            bitcrush_dither: u.bitcrush_dither,
            multi_grid_x: u.multi_grid_x,
            multi_grid_y: u.multi_grid_y,
            barrel: u.barrel,
            chroma_aberration: u.chroma_aberration,
            anamorphic_streak: u.anamorphic_streak,
            key_border: u.key_border,
            key_border_color: (u.key_border_color.max(0.0) as u32).min(7),
            key_shadow: u.key_shadow,
        }
    }

    pub fn apply_to_uniforms(&self, u: &mut EffectUniforms) {
        u.random_seed = self.random_seed;
        u.pixelate_size = finite_or(self.pixelate, 1.0).clamp(1.0, 32.0);
        u.rgb_split = finite_or(self.rgb_split, 0.0).clamp(0.0, 30.0);
        u.hue_shift = finite_or(self.hue_shift, 0.0).clamp(-180.0, 180.0);
        u.saturation = finite_or(self.saturation, 0.0).clamp(-1.0, 1.0);
        u.brightness = finite_or(self.brightness, 0.0).clamp(-1.0, 1.0);
        u.contrast = finite_or(self.contrast, 0.0).clamp(-1.0, 1.0);
        u.posterize = finite_or(self.posterize, 0.0).clamp(0.0, 16.0);
        u.invert = if self.invert { 1.0 } else { 0.0 };
        u.downsample = finite_or(self.downsample, 1.0).clamp(0.05, 1.0);
        u.shift_amount = finite_or(self.shift_amount, 0.0).clamp(0.0, 1.0);
        u.shift_block_size = finite_or(self.shift_block_size, 8.0).clamp(2.0, 256.0);
        u.shift_density = finite_or(self.shift_density, 0.5).clamp(0.0, 1.0);
        u.shift_speed = finite_or(self.shift_speed, 3.0).clamp(0.0, 20.0);
        u.grain_intensity = finite_or(self.grain_intensity, 0.0).clamp(0.0, 0.3);
        u.grain_size = finite_or(self.grain_size, 1.0).clamp(1.0, 4.0);
        u.grain_algo = (self.grain_algo.min(3)) as f32;
        u.color_grain = if self.color_grain { 1.0 } else { 0.0 };
        u.breathe_scale = finite_or(self.breathe_scale, 0.0).clamp(0.0, 0.05);
        u.breathe_rotation = finite_or(self.breathe_rotation, 0.0).clamp(0.0, 2.0);
        u.breathe_position = finite_or(self.breathe_position, 0.0).clamp(0.0, 0.02);
        u.vignette = finite_or(self.vignette, 0.0).clamp(0.0, 1.5);
        u.color_drift = finite_or(self.color_drift, 0.0).clamp(0.0, 0.02);
        u.key_mode = self.key_mode.min(4) as f32;
        u.key_threshold = finite_or(self.key_threshold, 0.5).clamp(0.0, 1.0);
        u.key_softness = finite_or(self.key_softness, 0.1).clamp(0.0, 0.5);
        u.key_color = [
            finite_or(self.key_color[0], 0.0).clamp(0.0, 1.0),
            finite_or(self.key_color[1], 1.0).clamp(0.0, 1.0),
            finite_or(self.key_color[2], 0.0).clamp(0.0, 1.0),
        ];
        u.key_tolerance = finite_or(self.key_tolerance, 0.15).clamp(0.0, 1.0);
        u.cellular_amount = finite_or(self.cellular_amount, 0.0).clamp(0.0, 1.0);
        u.cellular_scale = finite_or(self.cellular_scale, 10.0).clamp(2.0, 32.0);
        u.cellular_warp = finite_or(self.cellular_warp, 0.35).clamp(0.0, 1.0);
        u.cellular_speed = finite_or(self.cellular_speed, 0.25).clamp(0.0, 2.0);
        u.cellular_gap_amount = finite_or(self.cellular_gap_amount, 0.0).clamp(0.0, 1.0);
        u.cellular_gap_threshold = finite_or(self.cellular_gap_threshold, 0.65).clamp(0.0, 1.0);
        u.cellular_gap_softness = finite_or(self.cellular_gap_softness, 0.08).clamp(0.0, 0.5);
        u.contour = finite_or(self.contour, 0.0).clamp(0.0, 1.0);
        u.contour_bands = finite_or(self.contour_bands, 10.0).clamp(2.0, 40.0);
        u.contour_width = finite_or(self.contour_width, 1.2).clamp(0.2, 6.0);
        u.contour_hue = finite_or(self.contour_hue, 0.0).clamp(0.0, 1.0);
        u.contour_fill = finite_or(self.contour_fill, 0.25).clamp(0.0, 1.0);
        u.flatten = finite_or(self.flatten, 0.0).clamp(0.0, 1.0);
        u.flatten_levels = finite_or(self.flatten_levels, 5.0).clamp(2.0, 16.0);
        u.contour_dither = finite_or(self.contour_dither, 0.0).clamp(0.0, 1.0);
        u.solarize = finite_or(self.solarize, 0.0).clamp(0.0, 1.0);
        u.negative = finite_or(self.negative, 0.0).clamp(0.0, 1.0);
        u.negative_mode = self.negative_mode.min(2) as f32;
        u.colourpass = finite_or(self.colourpass, 0.0).clamp(0.0, 1.0);
        u.colourpass_hue = finite_or(self.colourpass_hue, 0.0).clamp(-180.0, 180.0);
        u.colourpass_width = finite_or(self.colourpass_width, 0.25).clamp(0.0, 1.0);
        u.edge_amount = finite_or(self.edge_amount, 0.0).clamp(0.0, 1.0);
        u.edge_hue = finite_or(self.edge_hue, 0.0).clamp(-180.0, 180.0);
        u.emboss = finite_or(self.emboss, 0.0).clamp(0.0, 1.0);
        u.emboss_angle = finite_or(self.emboss_angle, 45.0).clamp(-180.0, 180.0);
        u.halftone = finite_or(self.halftone, 0.0).clamp(0.0, 1.0);
        u.halftone_pitch = finite_or(self.halftone_pitch, 0.4).clamp(0.0, 1.0);
        u.halftone_angle = finite_or(self.halftone_angle, 0.0).clamp(-180.0, 180.0);
        u.moire = finite_or(self.moire, 0.0).clamp(0.0, 1.0);
        u.moire_freq = finite_or(self.moire_freq, 0.4).clamp(0.0, 1.0);
        u.row_smear = finite_or(self.row_smear, 0.0).clamp(0.0, 1.0);
        u.bitcrush = finite_or(self.bitcrush, 0.0).clamp(0.0, 1.0);
        u.bitcrush_levels = finite_or(self.bitcrush_levels, 2.0).clamp(2.0, 16.0);
        u.bitcrush_dither = finite_or(self.bitcrush_dither, 1.0).clamp(0.0, 1.0);
        u.multi_grid_x = finite_or(self.multi_grid_x, 1.0).clamp(1.0, 8.0);
        u.multi_grid_y = finite_or(self.multi_grid_y, 1.0).clamp(1.0, 8.0);
        u.barrel = finite_or(self.barrel, 0.0).clamp(-1.0, 1.0);
        u.chroma_aberration = finite_or(self.chroma_aberration, 0.0).clamp(0.0, 1.0);
        u.anamorphic_streak = finite_or(self.anamorphic_streak, 0.0).clamp(0.0, 1.0);
        u.key_border = finite_or(self.key_border, 0.0).clamp(0.0, 1.0);
        u.key_border_color = self.key_border_color.min(7) as f32;
        u.key_shadow = finite_or(self.key_shadow, 0.0).clamp(0.0, 1.0);
    }

    /// Get fields organized into groups for display.
    pub fn grouped_fields(&self) -> Vec<(&'static str, Vec<(&'static str, String)>)> {
        vec![
            (
                "digital",
                vec![
                    ("pixelate", format!("{:.1}", self.pixelate)),
                    ("rgb_split", format!("{:.1}", self.rgb_split)),
                    ("hue_shift", format!("{:.1}", self.hue_shift)),
                    ("saturation", format!("{:.2}", self.saturation)),
                    ("brightness", format!("{:.2}", self.brightness)),
                    ("contrast", format!("{:.2}", self.contrast)),
                    ("posterize", format!("{:.1}", self.posterize)),
                    ("invert", format!("{}", self.invert)),
                    ("downsample", format!("{:.2}", self.downsample)),
                ],
            ),
            (
                "shift",
                vec![
                    ("shift_amount", format!("{:.2}", self.shift_amount)),
                    ("shift_block_size", format!("{:.1}", self.shift_block_size)),
                    ("shift_density", format!("{:.2}", self.shift_density)),
                    ("shift_speed", format!("{:.2}", self.shift_speed)),
                ],
            ),
            (
                "cellular",
                vec![
                    ("cellular_amount", format!("{:.2}", self.cellular_amount)),
                    ("cellular_scale", format!("{:.1}", self.cellular_scale)),
                    ("cellular_warp", format!("{:.2}", self.cellular_warp)),
                    ("cellular_speed", format!("{:.2}", self.cellular_speed)),
                    (
                        "cellular_gap_amount",
                        format!("{:.2}", self.cellular_gap_amount),
                    ),
                    (
                        "cellular_gap_threshold",
                        format!("{:.2}", self.cellular_gap_threshold),
                    ),
                    (
                        "cellular_gap_softness",
                        format!("{:.2}", self.cellular_gap_softness),
                    ),
                ],
            ),
            (
                "analog",
                vec![
                    ("grain_intensity", format!("{:.2}", self.grain_intensity)),
                    ("grain_size", format!("{:.2}", self.grain_size)),
                    ("grain_algo", format!("{}", self.grain_algo)),
                    ("color_grain", format!("{}", self.color_grain)),
                    ("vignette", format!("{:.2}", self.vignette)),
                    ("color_drift", format!("{:.3}", self.color_drift)),
                ],
            ),
            (
                "motion",
                vec![
                    ("breathe_scale", format!("{:.3}", self.breathe_scale)),
                    ("breathe_rotation", format!("{:.2}", self.breathe_rotation)),
                    ("breathe_position", format!("{:.3}", self.breathe_position)),
                ],
            ),
            (
                "key",
                vec![
                    ("key_mode", format!("{}", self.key_mode)),
                    ("key_threshold", format!("{:.2}", self.key_threshold)),
                    ("key_softness", format!("{:.2}", self.key_softness)),
                    ("key_color_r", format!("{:.3}", self.key_color[0])),
                    ("key_color_g", format!("{:.3}", self.key_color[1])),
                    ("key_color_b", format!("{:.3}", self.key_color[2])),
                    ("key_tolerance", format!("{:.2}", self.key_tolerance)),
                    ("key_border", format!("{:.2}", self.key_border)),
                    ("key_border_color", self.key_border_color.to_string()),
                    ("key_shadow", format!("{:.2}", self.key_shadow)),
                ],
            ),
            (
                "small fx",
                vec![
                    ("contour", format!("{:.2}", self.contour)),
                    ("contour_bands", format!("{:.1}", self.contour_bands)),
                    ("contour_width", format!("{:.2}", self.contour_width)),
                    ("contour_hue", format!("{:.2}", self.contour_hue)),
                    ("contour_fill", format!("{:.2}", self.contour_fill)),
                    ("flatten", format!("{:.2}", self.flatten)),
                    ("flatten_levels", format!("{:.1}", self.flatten_levels)),
                    ("contour_dither", format!("{:.2}", self.contour_dither)),
                    ("solarize", format!("{:.2}", self.solarize)),
                    ("negative", format!("{:.2}", self.negative)),
                    ("negative_mode", format!("{}", self.negative_mode)),
                    ("colourpass", format!("{:.2}", self.colourpass)),
                    ("colourpass_hue", format!("{:.1}", self.colourpass_hue)),
                    ("colourpass_width", format!("{:.2}", self.colourpass_width)),
                    ("edge_amount", format!("{:.2}", self.edge_amount)),
                    ("edge_hue", format!("{:.1}", self.edge_hue)),
                    ("emboss", format!("{:.2}", self.emboss)),
                    ("emboss_angle", format!("{:.1}", self.emboss_angle)),
                    ("halftone", format!("{:.2}", self.halftone)),
                    ("halftone_pitch", format!("{:.2}", self.halftone_pitch)),
                    ("halftone_angle", format!("{:.1}", self.halftone_angle)),
                    ("moire", format!("{:.2}", self.moire)),
                    ("moire_freq", format!("{:.2}", self.moire_freq)),
                    ("row_smear", format!("{:.2}", self.row_smear)),
                    ("bitcrush", format!("{:.2}", self.bitcrush)),
                    ("bitcrush_levels", format!("{:.1}", self.bitcrush_levels)),
                    ("bitcrush_dither", format!("{:.2}", self.bitcrush_dither)),
                    ("multi_grid_x", format!("{:.0}", self.multi_grid_x)),
                    ("multi_grid_y", format!("{:.0}", self.multi_grid_y)),
                ],
            ),
            (
                "optics",
                vec![
                    ("barrel", format!("{:.2}", self.barrel)),
                    (
                        "chroma_aberration",
                        format!("{:.2}", self.chroma_aberration),
                    ),
                    (
                        "anamorphic_streak",
                        format!("{:.2}", self.anamorphic_streak),
                    ),
                ],
            ),
        ]
    }

    /// Set a single field by key name. Returns true if the key was recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "pixelate" => {
                if let Ok(v) = value.parse() {
                    self.pixelate = v;
                    return true;
                }
            }
            "rgb_split" => {
                if let Ok(v) = value.parse() {
                    self.rgb_split = v;
                    return true;
                }
            }
            "hue_shift" => {
                if let Ok(v) = value.parse() {
                    self.hue_shift = v;
                    return true;
                }
            }
            "saturation" => {
                if let Ok(v) = value.parse() {
                    self.saturation = v;
                    return true;
                }
            }
            "brightness" => {
                if let Ok(v) = value.parse() {
                    self.brightness = v;
                    return true;
                }
            }
            "contrast" => {
                if let Ok(v) = value.parse() {
                    self.contrast = v;
                    return true;
                }
            }
            "posterize" => {
                if let Ok(v) = value.parse() {
                    self.posterize = v;
                    return true;
                }
            }
            "invert" => {
                if let Ok(v) = value.parse() {
                    self.invert = v;
                    return true;
                }
            }
            "downsample" => {
                if let Ok(v) = value.parse() {
                    self.downsample = v;
                    return true;
                }
            }
            "shift_amount" => {
                if let Ok(v) = value.parse() {
                    self.shift_amount = v;
                    return true;
                }
            }
            "shift_block_size" => {
                if let Ok(v) = value.parse() {
                    self.shift_block_size = v;
                    return true;
                }
            }
            "shift_density" => {
                if let Ok(v) = value.parse() {
                    self.shift_density = v;
                    return true;
                }
            }
            "shift_speed" => {
                if let Ok(v) = value.parse() {
                    self.shift_speed = v;
                    return true;
                }
            }
            "grain_intensity" => {
                if let Ok(v) = value.parse() {
                    self.grain_intensity = v;
                    return true;
                }
            }
            "grain_size" => {
                if let Ok(v) = value.parse() {
                    self.grain_size = v;
                    return true;
                }
            }
            "grain_algo" => {
                if let Ok(v) = value.parse() {
                    self.grain_algo = v;
                    return true;
                }
            }
            "color_grain" => {
                if let Ok(v) = value.parse() {
                    self.color_grain = v;
                    return true;
                }
            }
            "breathe_scale" => {
                if let Ok(v) = value.parse() {
                    self.breathe_scale = v;
                    return true;
                }
            }
            "breathe_rotation" => {
                if let Ok(v) = value.parse() {
                    self.breathe_rotation = v;
                    return true;
                }
            }
            "breathe_position" => {
                if let Ok(v) = value.parse() {
                    self.breathe_position = v;
                    return true;
                }
            }
            "vignette" => {
                if let Ok(v) = value.parse() {
                    self.vignette = v;
                    return true;
                }
            }
            "color_drift" => {
                if let Ok(v) = value.parse() {
                    self.color_drift = v;
                    return true;
                }
            }
            "key_mode" => {
                if let Ok(v) = value.parse() {
                    self.key_mode = v;
                    return true;
                }
            }
            "key_threshold" => {
                if let Ok(v) = value.parse() {
                    self.key_threshold = v;
                    return true;
                }
            }
            "key_softness" => {
                if let Ok(v) = value.parse() {
                    self.key_softness = v;
                    return true;
                }
            }
            "key_color_r" | "key_color_g" | "key_color_b" => {
                if let Ok(v) = value.parse() {
                    let index = match key {
                        "key_color_r" => 0,
                        "key_color_g" => 1,
                        _ => 2,
                    };
                    self.key_color[index] = v;
                    return true;
                }
            }
            "key_tolerance" => {
                if let Ok(v) = value.parse() {
                    self.key_tolerance = v;
                    return true;
                }
            }
            "cellular_amount" => {
                if let Ok(v) = value.parse() {
                    self.cellular_amount = v;
                    return true;
                }
            }
            "cellular_scale" => {
                if let Ok(v) = value.parse() {
                    self.cellular_scale = v;
                    return true;
                }
            }
            "cellular_warp" => {
                if let Ok(v) = value.parse() {
                    self.cellular_warp = v;
                    return true;
                }
            }
            "cellular_speed" => {
                if let Ok(v) = value.parse() {
                    self.cellular_speed = v;
                    return true;
                }
            }
            "cellular_gap_amount" => {
                if let Ok(v) = value.parse() {
                    self.cellular_gap_amount = v;
                    return true;
                }
            }
            "cellular_gap_threshold" => {
                if let Ok(v) = value.parse() {
                    self.cellular_gap_threshold = v;
                    return true;
                }
            }
            "cellular_gap_softness" => {
                if let Ok(v) = value.parse() {
                    self.cellular_gap_softness = v;
                    return true;
                }
            }
            "negative_mode" => {
                if let Ok(v) = value.parse() {
                    self.negative_mode = v;
                    return true;
                }
            }
            // B13 small effects: every remaining control is a plain float.
            "contour" | "contour_bands" | "contour_width" | "contour_hue" | "contour_fill"
            | "flatten" | "flatten_levels" | "contour_dither" | "solarize" | "negative"
            | "colourpass" | "colourpass_hue" | "colourpass_width" | "edge_amount" | "edge_hue"
            | "emboss" | "emboss_angle" | "halftone" | "halftone_pitch" | "halftone_angle"
            | "moire" | "moire_freq" | "row_smear" | "bitcrush" | "bitcrush_levels"
            | "bitcrush_dither" | "multi_grid_x" | "multi_grid_y" | "barrel"
            | "chroma_aberration" | "anamorphic_streak" | "key_border" | "key_shadow" => {
                if let Ok(v) = value.parse::<f32>() {
                    let slot = match key {
                        "contour" => &mut self.contour,
                        "contour_bands" => &mut self.contour_bands,
                        "contour_width" => &mut self.contour_width,
                        "contour_hue" => &mut self.contour_hue,
                        "contour_fill" => &mut self.contour_fill,
                        "flatten" => &mut self.flatten,
                        "flatten_levels" => &mut self.flatten_levels,
                        "contour_dither" => &mut self.contour_dither,
                        "solarize" => &mut self.solarize,
                        "negative" => &mut self.negative,
                        "colourpass" => &mut self.colourpass,
                        "colourpass_hue" => &mut self.colourpass_hue,
                        "colourpass_width" => &mut self.colourpass_width,
                        "edge_amount" => &mut self.edge_amount,
                        "edge_hue" => &mut self.edge_hue,
                        "emboss" => &mut self.emboss,
                        "emboss_angle" => &mut self.emboss_angle,
                        "halftone" => &mut self.halftone,
                        "halftone_pitch" => &mut self.halftone_pitch,
                        "halftone_angle" => &mut self.halftone_angle,
                        "moire" => &mut self.moire,
                        "moire_freq" => &mut self.moire_freq,
                        "row_smear" => &mut self.row_smear,
                        "bitcrush" => &mut self.bitcrush,
                        "bitcrush_levels" => &mut self.bitcrush_levels,
                        "bitcrush_dither" => &mut self.bitcrush_dither,
                        "multi_grid_x" => &mut self.multi_grid_x,
                        "multi_grid_y" => &mut self.multi_grid_y,
                        "barrel" => &mut self.barrel,
                        "chroma_aberration" => &mut self.chroma_aberration,
                        "key_border" => &mut self.key_border,
                        "key_shadow" => &mut self.key_shadow,
                        _ => &mut self.anamorphic_streak,
                    };
                    *slot = v;
                    return true;
                }
            }
            "key_border_color" => {
                if let Ok(v) = value.parse::<u32>() {
                    self.key_border_color = v.min(7);
                    return true;
                }
            }
            _ => {}
        }
        false
    }
}

// --- Conversion: Layer <-> LayerConfig ---

impl LayerConfig {
    pub fn from_layer(layer: &Layer) -> Self {
        let filename = layer.filename.clone();
        let source_path = layer.source_reference_for_persistence().to_owned();
        let clip_slots = layer.clip_slots_for_persistence();
        Self {
            filename,
            source_path,
            opacity: layer.opacity,
            blend_mode: layer.blend_mode.key().to_string(),
            speed: layer.speed,
            fps: layer.fps,
            paused: layer.paused,
            visible: layer.visible,
            bypass_master_fx: layer.bypass_master_fx,
            reroll_on_loop: layer.reroll_on_loop,
            effects: EffectsConfig::from_uniforms(&layer.effects),
            transform: layer.transform.sanitized(),
            motion: {
                let motion = MotionConfig::from_params(layer.motion);
                (!motion.is_default()).then_some(motion)
            },
            rack: None,
            clip_slots,
            active_clip_slot: Some(layer.active_clip_slot),
            matte: LayerMatteConfig::default(),
            pattern: layer.pattern_params().map(PatternSynthConfig::from_params),
            text_page: layer.text_page_params().map(TextPageConfig::from_params),
        }
    }

    /// Re-establish the compatibility mirrors from the active canonical slot.
    /// Slot IDs are resolved by search, never used as indices.
    pub fn sync_legacy_mirrors_from_active_slot(&mut self) {
        self.active_clip_slot = self.clip_slots.active_or_first(self.active_clip_slot);
        let Some(slot) = self.active_clip_slot.and_then(|id| self.clip_slots.get(id)) else {
            return;
        };
        self.filename.clone_from(&slot.filename);
        self.source_path.clone_from(&slot.source_path);
        self.speed = slot.transport.rate as f32;
        if let Some(sample_fps) = slot.transport.sample_fps {
            self.fps = sample_fps as f32;
        }
    }

    /// Compatibility bridge for legacy/native code that still edits the
    /// top-level source and cadence fields directly. Canonical callers should
    /// edit the active slot instead.
    pub fn sync_active_slot_from_legacy_mirrors(&mut self) {
        self.active_clip_slot = self.clip_slots.active_or_first(self.active_clip_slot);
        let Some(slot) = self
            .active_clip_slot
            .and_then(|id| self.clip_slots.get_mut(id))
        else {
            return;
        };
        slot.filename.clone_from(&self.filename);
        slot.source_path.clone_from(&self.source_path);
        slot.transport.rate = f64::from(finite_or(self.speed, 1.0));
        slot.transport.sample_fps = Some(f64::from(finite_or(self.fps, 30.0)));
        slot.transport = slot.transport.sanitized();
    }

    /// Procedural pieces intentionally contain one source slot and no routed
    /// matte topology. The selected source is retained, assigned canonical ID
    /// 1, and its transport mirrors are updated after mutation.
    pub fn collapse_to_generated_single_slot(&mut self) {
        let selected = self
            .active_clip_slot
            .and_then(|id| self.clip_slots.get(id))
            .or_else(|| self.clip_slots.iter().next())
            .cloned()
            .unwrap_or_else(|| {
                ClipSlotConfig::from_legacy(
                    self.filename.clone(),
                    self.source_path.clone(),
                    self.speed,
                    self.fps,
                )
            });
        let mut selected = selected;
        selected.id = ClipSlotId::LEGACY;
        selected.filename.clone_from(&self.filename);
        selected.source_path.clone_from(&self.source_path);
        selected.transport.rate = f64::from(finite_or(self.speed, 1.0));
        selected.transport.sample_fps = Some(f64::from(finite_or(self.fps, 30.0)));
        selected.transport = selected.transport.sanitized();
        self.clip_slots = ClipSlots::singleton(selected);
        self.active_clip_slot = Some(ClipSlotId::LEGACY);
        self.matte = LayerMatteConfig::default();
        self.sync_legacy_mirrors_from_active_slot();
    }

    pub fn apply_to_layer(&self, layer: &mut Layer) {
        layer.opacity = finite_or(self.opacity, 1.0).clamp(0.0, 1.0);
        layer.blend_mode =
            BlendMode::from_key(self.blend_mode.as_str()).unwrap_or(BlendMode::Normal);
        layer.speed = finite_or(self.speed, 1.0).clamp(0.25, 4.0);
        layer.fps = finite_or(self.fps, 30.0).clamp(1.0, 240.0);
        layer.paused = self.paused;
        layer.visible = self.visible;
        layer.bypass_master_fx = self.bypass_master_fx;
        layer.reroll_on_loop = self.reroll_on_loop;
        self.effects.apply_to_uniforms(&mut layer.effects);
        // The B13 optics author at master scope only; a hostile or hand-edited
        // layer section cannot install one on a layer copy.
        layer.effects.clear_master_only_effects();
        layer.transform = self.transform.sanitized();
        layer.motion = self.motion.unwrap_or_default().to_params().sanitized();
        // B7 pattern values land only on a layer that actually is a pattern
        // source; the constructor already installed the config's params, so
        // this is the same values-follow-kind law the effects take.
        if let (Some(config), Some(params)) = (self.pattern, layer.pattern_params_mut()) {
            *params = config.to_params();
        }
    }

    /// Apply only the visual contribution of this saved position. Source
    /// identity, playback cadence, pause state, and loop-reroll policy belong
    /// to the current live stack and are deliberately retained.
    pub fn apply_look_to_layer(&self, layer: &mut Layer) {
        self.apply_look_to_fields(
            &mut layer.opacity,
            &mut layer.blend_mode,
            &mut layer.visible,
            &mut layer.bypass_master_fx,
            &mut layer.effects,
            &mut layer.transform,
        );
        if let Some(motion) = self.motion {
            apply_motion_look(motion, &mut layer.motion);
        }
        // B7 pattern values transfer as a look only onto a live pattern
        // layer — the matte kind-match precedent: values move, the source
        // identity never does. A text page's body is content identity, not a
        // look, and is deliberately not transferred.
        if let (Some(config), Some(params)) = (self.pattern, layer.pattern_params_mut()) {
            *params = config.to_params();
        }
    }

    fn apply_look_to_fields(
        &self,
        opacity: &mut f32,
        blend_mode: &mut BlendMode,
        visible: &mut bool,
        bypass_master_fx: &mut bool,
        effects: &mut EffectUniforms,
        transform: &mut SpatialTransform,
    ) {
        *opacity = finite_or(self.opacity, 1.0).clamp(0.0, 1.0);
        *blend_mode = BlendMode::from_key(self.blend_mode.as_str()).unwrap_or(BlendMode::Normal);
        *visible = self.visible;
        *bypass_master_fx = self.bypass_master_fx;
        self.effects.apply_to_uniforms(effects);
        // Look application targets layer scope here; the B13 optics stay
        // master-only.
        effects.clear_master_only_effects();
        *transform = self.transform.sanitized();
    }

    /// Get top-level layer fields as (key, value_string) pairs.
    pub fn top_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("filename", self.filename.clone()),
            ("opacity", format!("{:.2}", self.opacity)),
            ("blend_mode", self.blend_mode.clone()),
            ("speed", format!("{:.2}", self.speed)),
            ("fps", format!("{:.1}", self.fps)),
            ("paused", format!("{}", self.paused)),
            ("visible", format!("{}", self.visible)),
            ("bypass_master_fx", format!("{}", self.bypass_master_fx)),
            ("reroll_on_loop", format!("{}", self.reroll_on_loop)),
            ("position_x", format!("{:.4}", self.transform.position[0])),
            ("position_y", format!("{:.4}", self.transform.position[1])),
            ("scale_x", format!("{:.4}", self.transform.scale[0])),
            ("scale_y", format!("{:.4}", self.transform.scale[1])),
            ("anchor_x", format!("{:.4}", self.transform.anchor[0])),
            ("anchor_y", format!("{:.4}", self.transform.anchor[1])),
            (
                "rotation_deg",
                format!("{:.3}", self.transform.rotation_deg),
            ),
            ("skew_deg", format!("{:.3}", self.transform.skew_deg)),
            (
                "skew_axis_deg",
                format!("{:.3}", self.transform.skew_axis_deg),
            ),
            (
                "fit",
                format!("{:?}", self.transform.fit).to_ascii_lowercase(),
            ),
            ("crop_left", format!("{:.4}", self.transform.crop[0])),
            ("crop_top", format!("{:.4}", self.transform.crop[1])),
            ("crop_right", format!("{:.4}", self.transform.crop[2])),
            ("crop_bottom", format!("{:.4}", self.transform.crop[3])),
            (
                "edge",
                format!("{:?}", self.transform.edge).to_ascii_lowercase(),
            ),
            (
                "sampling",
                format!("{:?}", self.transform.sampling).to_ascii_lowercase(),
            ),
        ]
    }

    /// Set a top-level field by key name. Returns true if recognized.
    pub fn set_field(&mut self, key: &str, value: &str) -> bool {
        match key {
            "opacity" => {
                if let Ok(v) = value.parse() {
                    self.opacity = v;
                    return true;
                }
            }
            "blend_mode" => {
                self.blend_mode = value.to_string();
                return true;
            }
            "speed" => {
                if let Ok(v) = value.parse() {
                    self.speed = v;
                    return true;
                }
            }
            "fps" => {
                if let Ok(v) = value.parse() {
                    self.fps = v;
                    return true;
                }
            }
            "paused" => {
                if let Ok(v) = value.parse() {
                    self.paused = v;
                    return true;
                }
            }
            "visible" => {
                if let Ok(v) = value.parse() {
                    self.visible = v;
                    return true;
                }
            }
            "bypass_master_fx" => {
                if let Ok(v) = value.parse() {
                    self.bypass_master_fx = v;
                    return true;
                }
            }
            "reroll_on_loop" => {
                if let Ok(v) = value.parse() {
                    self.reroll_on_loop = v;
                    return true;
                }
            }
            "position_x" | "position_y" | "scale_x" | "scale_y" | "anchor_x" | "anchor_y"
            | "rotation_deg" | "skew_deg" | "skew_axis_deg" | "crop_left" | "crop_top"
            | "crop_right" | "crop_bottom" => {
                let Ok(parsed) = value.parse::<f32>() else {
                    return false;
                };
                match key {
                    "position_x" => self.transform.position[0] = parsed,
                    "position_y" => self.transform.position[1] = parsed,
                    "scale_x" => self.transform.scale[0] = parsed,
                    "scale_y" => self.transform.scale[1] = parsed,
                    "anchor_x" => self.transform.anchor[0] = parsed,
                    "anchor_y" => self.transform.anchor[1] = parsed,
                    "rotation_deg" => self.transform.rotation_deg = parsed,
                    "skew_deg" => self.transform.skew_deg = parsed,
                    "skew_axis_deg" => self.transform.skew_axis_deg = parsed,
                    "crop_left" => self.transform.crop[0] = parsed,
                    "crop_top" => self.transform.crop[1] = parsed,
                    "crop_right" => self.transform.crop[2] = parsed,
                    "crop_bottom" => self.transform.crop[3] = parsed,
                    _ => unreachable!(),
                }
                self.transform = self.transform.sanitized();
                return true;
            }
            "fit" => {
                self.transform.fit = match value.to_ascii_lowercase().as_str() {
                    "stretch" => FitMode::Stretch,
                    "fit" => FitMode::Fit,
                    "fill" => FitMode::Fill,
                    "native" => FitMode::Native,
                    _ => return false,
                };
                return true;
            }
            "edge" => {
                self.transform.edge = match value.to_ascii_lowercase().as_str() {
                    "transparent" => EdgeMode::Transparent,
                    "clamp" => EdgeMode::Clamp,
                    "repeat" => EdgeMode::Repeat,
                    "mirror" => EdgeMode::Mirror,
                    _ => return false,
                };
                return true;
            }
            "sampling" => {
                self.transform.sampling = match value.to_ascii_lowercase().as_str() {
                    "linear" => SamplingMode::Linear,
                    "nearest" => SamplingMode::Nearest,
                    _ => return false,
                };
                return true;
            }
            _ => {}
        }
        false
    }
}

fn apply_positional_looks<T>(
    saved_layers: &[LayerConfig],
    live_layers: &mut [T],
    mut apply: impl FnMut(&LayerConfig, &mut T),
) -> LookApplySummary {
    let mapped_layers = saved_layers.len().min(live_layers.len());
    for (config, layer) in saved_layers.iter().zip(live_layers.iter_mut()) {
        apply(config, layer);
    }
    LookApplySummary {
        mapped_layers,
        unused_patch_layers: saved_layers.len().saturating_sub(mapped_layers),
        untouched_live_layers: live_layers.len().saturating_sub(mapped_layers),
        ..LookApplySummary::default()
    }
}

fn apply_layer_matte_look_values(
    sampled: LayerMatteConfig,
    live: &mut LayerMatte,
    layer_ids: &[StableLayerId],
) -> bool {
    let sampled = sampled.to_runtime(|position| position.resolve(layer_ids).copied());
    if sampled.enabled != live.enabled
        || sampled.input != live.input
        || sampled.channel != live.channel
        || sampled.invert != live.invert
    {
        return false;
    }
    live.amount = sampled.amount;
    live.threshold = sampled.threshold;
    live.softness = sampled.softness;
    true
}

fn record_look_rack_nodes(
    summary: &mut LookApplySummary,
    scope: LookRackScope,
    sampled: &RuntimeVisualRack,
    applied: bool,
) {
    let destination = if applied {
        &mut summary.applied_nodes
    } else {
        &mut summary.skipped_nodes
    };
    destination.extend(sampled.iter().map(|node| LookNodeRef {
        scope,
        node_id: node.stable_id,
    }));
}

fn runtime_group_visual_topology_matches(sampled: &RuntimeGroup, live: &RuntimeGroup) -> bool {
    sampled.id == live.id
        && sampled.members == live.members
        && match (sampled.matte, live.matte) {
            (Some(sampled), Some(live)) => {
                sampled.tap == live.tap
                    && sampled.channel == live.channel
                    && sampled.invert == live.invert
            }
            (None, None) => true,
            _ => false,
        }
        && sampled.rack.len() == live.rack.len()
        && sampled.rack.topology_signature() == live.rack.topology_signature()
        && sampled
            .rack
            .iter()
            .zip(live.rack.iter())
            .all(|(sampled, live)| {
                sampled.stable_id == live.stable_id
                    && sampled.kind.tag() == live.kind.tag()
                    && match (sampled.kind, live.kind) {
                        (
                            crate::visual_rack::RuntimeVisualNodeKind::Mask(
                                crate::visual_rack::RuntimeMaskParams::Image(sampled),
                            ),
                            crate::visual_rack::RuntimeVisualNodeKind::Mask(
                                crate::visual_rack::RuntimeMaskParams::Image(live),
                            ),
                        ) => {
                            sampled.tap == live.tap
                                && sampled.channel == live.channel
                                && sampled.invert == live.invert
                        }
                        (
                            crate::visual_rack::RuntimeVisualNodeKind::Mask(sampled),
                            crate::visual_rack::RuntimeVisualNodeKind::Mask(live),
                        ) => std::mem::discriminant(&sampled) == std::mem::discriminant(&live),
                        (
                            crate::visual_rack::RuntimeVisualNodeKind::Displace(sampled),
                            crate::visual_rack::RuntimeVisualNodeKind::Displace(live),
                        ) => sampled.tap == live.tap,
                        // All four Symmetry slots plus both masks. A group-scope
                        // Look must refuse a differently routed or differently
                        // armed field rather than retarget it.
                        (
                            crate::visual_rack::RuntimeVisualNodeKind::Symmetry(sampled),
                            crate::visual_rack::RuntimeVisualNodeKind::Symmetry(live),
                        ) => {
                            sampled.donors == live.donors
                                && sampled.motion == live.motion
                                && sampled.source_mask.sanitized() == live.source_mask.sanitized()
                                && sampled.motion_mask == live.motion_mask
                        }
                        (
                            crate::visual_rack::RuntimeVisualNodeKind::Residual(sampled),
                            crate::visual_rack::RuntimeVisualNodeKind::Residual(live),
                        ) => sampled.routes() == live.routes(),
                        _ => true,
                    }
            })
}

fn apply_runtime_group_look_values(sampled: &RuntimeGroup, live: &mut RuntimeGroup) -> bool {
    if !runtime_group_visual_topology_matches(sampled, live) {
        return false;
    }
    live.opacity = sampled.opacity;
    live.transform = sampled.transform;
    if !crate::morph::apply_runtime_rack_values_strict(&sampled.rack, &mut live.rack) {
        return false;
    }
    if let (Some(sampled), Some(live)) = (sampled.matte, &mut live.matte) {
        live.amount = sampled.amount;
        live.threshold = sampled.threshold;
        live.softness = sampled.softness;
    }
    true
}

fn look_root_identity_matches(
    saved: &CompositionTree,
    live: &RuntimeComposition,
    layer_ids: &[StableLayerId],
) -> bool {
    saved.root().len() == live.root().len()
        && saved
            .root()
            .iter()
            .zip(live.root())
            .all(|(saved, live)| match (*saved, *live) {
                (RootItem::Layer { layer, .. }, RuntimeRootItem::Layer { layer_id, .. }) => {
                    layer.resolve(layer_ids).copied() == Some(layer_id)
                }
                (
                    RootItem::Group { group_id: saved_id },
                    RuntimeRootItem::Group { group_id: live_id },
                ) => saved_id == live_id,
                _ => false,
            })
}

fn apply_saved_composition_look(
    saved: &CompositionTree,
    live: &mut RuntimeComposition,
    layer_ids: &[StableLayerId],
    summary: &mut LookApplySummary,
) {
    if look_root_identity_matches(saved, live, layer_ids) {
        live.set_bus_crossfade(saved.bus_crossfade());
        // The B8 bus mixer is a composition-level value bundle and travels
        // with the crossfade under the same identity gate.
        live.set_mixer(saved.mixer());
        summary.applied_bus_crossfade = true;
    } else {
        summary.skipped_bus_crossfade = true;
    }

    // Route resolution must not hold an immutable borrow across the eventual
    // per-group mutation. This stable set also ensures a deleted output stays
    // missing rather than being inferred from root position.
    let live_group_ids: HashSet<_> = live.groups().map(|group| group.id).collect();
    for saved_group in saved.groups() {
        let scope = LookRackScope::Group(saved_group.id);
        let mapped_members = saved_group
            .members
            .iter()
            .map(|position| position.resolve(layer_ids).copied())
            .collect::<Option<Vec<_>>>();
        let sampled = mapped_members
            .and_then(|members| RuntimeGroupMembers::try_from_vec(members).ok())
            .map(|members| {
                let group_exists = |group_id| live_group_ids.contains(&group_id);
                let rack = saved_group.rack.resolve_routes(
                    |position| position.resolve(layer_ids).copied(),
                    group_exists,
                );
                let matte = saved_group.matte.map(|matte| {
                    RuntimeImageMatte::resolve_routes(
                        matte,
                        &mut |position| position.resolve(layer_ids).copied(),
                        &group_exists,
                    )
                });
                RuntimeGroup {
                    id: saved_group.id,
                    name: saved_group.name.clone(),
                    members,
                    opacity: saved_group.opacity,
                    transform: saved_group.transform,
                    rack,
                    matte,
                    solo: saved_group.solo,
                    bypass: saved_group.bypass,
                    bus: saved_group.bus,
                }
            });

        let applied = sampled.as_ref().is_some_and(|sampled| {
            live.group_mut(sampled.id)
                .is_some_and(|live_group| apply_runtime_group_look_values(sampled, live_group))
        });
        if applied {
            summary.applied_groups += 1;
            summary.applied_group_ids.push(saved_group.id);
        } else {
            summary.skipped_groups += 1;
            summary.skipped_group_ids.push(saved_group.id);
        }
        if let Some(sampled) = &sampled {
            record_look_rack_nodes(summary, scope, &sampled.rack, applied);
        } else {
            let destination = if applied {
                &mut summary.applied_nodes
            } else {
                &mut summary.skipped_nodes
            };
            destination.extend(saved_group.rack.iter().map(|node| LookNodeRef {
                scope,
                node_id: node.stable_id,
            }));
        }
    }
}

fn saved_position_at(position: usize) -> Result<SavedLayerPosition, String> {
    u32::try_from(position)
        .ok()
        .and_then(SavedLayerPosition::new)
        .ok_or_else(|| format!("saved layer position {position} exceeds bounds"))
}

fn legacy_composition_for_positions(
    positions_front_to_back: &[SavedLayerPosition],
) -> Result<CompositionTree, String> {
    let back_to_front: Vec<_> = positions_front_to_back.iter().rev().copied().collect();
    CompositionTree::legacy_for_layers(&back_to_front)
        .map_err(|error| format!("synthesize legacy composition: {error}"))
}

fn validation_layer_scope(position: SavedLayerPosition) -> Result<VisualScopeId, String> {
    StableLayerId::new(u64::from(position.get()) + 1)
        .map(VisualScopeId::Layer)
        .ok_or_else(|| format!("invalid validation layer position {}", position.get()))
}

fn collect_rack_dependencies(
    rack: &VisualRack,
    consumer: VisualScopeId,
    below: &BTreeMap<VisualScopeId, Vec<VisualScopeId>>,
    dependencies: &mut Vec<ImageDependency>,
    ordering_edges: &mut Vec<ImageOrderingEdge>,
) -> Result<(), String> {
    for node in rack.iter().filter(|node| node.enabled && node.wet > 0.0) {
        // Mirror the live planner's admission predicate exactly, slot for
        // slot: a node that cannot collect a tap at frame time must not claim a
        // saved edge here, and a slot that cannot be sampled must not claim one
        // either. Slot-ordered, so a multi-route kind claims every slot it will
        // bind.
        let mut routes: [Option<SavedImageTap>; RESIDUAL_ROUTE_SLOTS] =
            [None; RESIDUAL_ROUTE_SLOTS];
        match node.kind {
            VisualNodeKind::Mask(MaskParams::Image(matte)) if matte.amount > 0.0 => {
                routes[usize::from(RACK_PRIMARY_ROUTE_SLOT)] = Some(matte.tap);
            }
            VisualNodeKind::Displace(displace) if !displace.is_exact_bypass() => {
                routes[usize::from(RACK_PRIMARY_ROUTE_SLOT)] = Some(displace.tap);
            }
            VisualNodeKind::Residual(residual) if !residual.is_exact_bypass() => {
                let [structure, detail] = residual.routes();
                routes[usize::from(RESIDUAL_STRUCTURE_SLOT)] = Some(structure);
                routes[usize::from(RESIDUAL_DETAIL_SLOT)] = Some(detail);
            }
            // A Symmetry Field answers admission per image slot: a donor no
            // sector record can name claims nothing here, exactly as it
            // collects no tap in the live planner. The destructure is the
            // compile-time proof that both frozen image slots are carried.
            VisualNodeKind::Symmetry(symmetry) if !symmetry.is_exact_bypass() => {
                let [donor0, donor1] = symmetry.admitted_donor_taps();
                routes[0] = donor0;
                routes[1] = donor1;
            }
            _ => continue,
        }
        for tap in routes.into_iter().flatten() {
            collect_saved_tap_dependency(tap, consumer, below, dependencies, ordering_edges)?;
        }
    }
    Ok(())
}

fn collect_saved_tap_dependency(
    tap: SavedImageTap,
    consumer: VisualScopeId,
    below: &BTreeMap<VisualScopeId, Vec<VisualScopeId>>,
    dependencies: &mut Vec<ImageDependency>,
    ordering_edges: &mut Vec<ImageOrderingEdge>,
) -> Result<(), String> {
    match tap.source {
        SavedImageSource::SelectedLayer {
            layer_position,
            stage: LayerImageStage::PostLocalEffects,
        } => dependencies.push(ImageDependency {
            consumer,
            producer: validation_layer_scope(layer_position)?,
            timing: tap.timing,
        }),
        SavedImageSource::SelectedLayer {
            layer_position,
            stage: LayerImageStage::PreLocalEffects,
            ..
        } if tap.timing == EdgeTiming::PreviousFrame => dependencies.push(ImageDependency {
            consumer,
            producer: validation_layer_scope(layer_position)?,
            timing: tap.timing,
        }),
        SavedImageSource::SelectedLayer { .. }
        | SavedImageSource::MissingSelectedLayer { .. }
        | SavedImageSource::MissingGroupOutput { .. } => {}
        SavedImageSource::OneBelow => {
            if let Some(producer) = below_for(consumer, below)?.last().copied() {
                dependencies.push(ImageDependency {
                    consumer,
                    producer,
                    timing: tap.timing,
                });
            }
        }
        SavedImageSource::AllBelow => {
            collect_all_below_dependency(
                consumer,
                below_for(consumer, below)?,
                tap.timing,
                dependencies,
                ordering_edges,
            );
        }
        SavedImageSource::GroupOutput { group_id } => dependencies.push(ImageDependency {
            consumer,
            producer: VisualScopeId::Group(group_id),
            timing: tap.timing,
        }),
        SavedImageSource::CleanProgram => dependencies.push(ImageDependency {
            consumer,
            producer: VisualScopeId::Program,
            timing: tap.timing,
        }),
        // The gesture canvas is not produced by any scope in this saved graph,
        // so a route to it claims no dependency edge and no ordering edge. A
        // dormant saved route and a woken one therefore agree by construction
        // rather than by a second rule. The programme tap is published outside
        // the graph — after the frame is accepted — so the identical law
        // holds: N-1 by construction, no edge to claim.
        SavedImageSource::GestureCanvas | SavedImageSource::ProgramTap => {}
    }
    Ok(())
}

fn collect_legacy_layer_matte_dependency(
    input: SavedImageInput,
    consumer: VisualScopeId,
    below: &BTreeMap<VisualScopeId, Vec<VisualScopeId>>,
    dependencies: &mut Vec<ImageDependency>,
    ordering_edges: &mut Vec<ImageOrderingEdge>,
) -> Result<(), String> {
    match input {
        SavedImageInput::SelectedLayer {
            layer_position,
            stage: LayerImageStage::PostLocalEffects,
        } => dependencies.push(ImageDependency {
            consumer,
            producer: validation_layer_scope(layer_position)?,
            timing: EdgeTiming::CurrentFrame,
        }),
        SavedImageInput::SelectedLayer {
            stage: LayerImageStage::PreLocalEffects,
            ..
        }
        | SavedImageInput::MissingSelectedLayer { .. }
        | SavedImageInput::MissingGroupOutput { .. } => {}
        SavedImageInput::OneBelow => {
            if let Some(producer) = below_for(consumer, below)?.last().copied() {
                dependencies.push(ImageDependency {
                    consumer,
                    producer,
                    timing: EdgeTiming::CurrentFrame,
                });
            }
        }
        SavedImageInput::AllBelow => {
            collect_all_below_dependency(
                consumer,
                below_for(consumer, below)?,
                EdgeTiming::CurrentFrame,
                dependencies,
                ordering_edges,
            );
        }
        SavedImageInput::ProgramHistory => dependencies.push(ImageDependency {
            consumer,
            producer: VisualScopeId::Program,
            timing: EdgeTiming::PreviousFrame,
        }),
        SavedImageInput::CleanProgram => dependencies.push(ImageDependency {
            consumer,
            producer: VisualScopeId::Program,
            timing: EdgeTiming::CurrentFrame,
        }),
        SavedImageInput::GroupOutput { group_id } => {
            dependencies.push(ImageDependency {
                consumer,
                producer: VisualScopeId::Group(group_id),
                timing: EdgeTiming::CurrentFrame,
            });
        }
    }
    Ok(())
}

/// AllBelow is one authored image tap backed by a compact composited prefix.
/// The final prefix output represents that tap for resource accounting, while
/// the remaining producer-before-consumer relations participate only in the
/// same-frame DAG. Previous-frame reads deliberately carry no ordering edges.
fn collect_all_below_dependency(
    consumer: VisualScopeId,
    producers: &[VisualScopeId],
    timing: EdgeTiming,
    dependencies: &mut Vec<ImageDependency>,
    ordering_edges: &mut Vec<ImageOrderingEdge>,
) {
    let Some(representative) = producers.last().copied() else {
        return;
    };
    dependencies.push(ImageDependency {
        consumer,
        producer: representative,
        timing,
    });
    if timing == EdgeTiming::CurrentFrame {
        ordering_edges.extend(
            producers
                .iter()
                .copied()
                .map(|producer| ImageOrderingEdge { producer, consumer }),
        );
    }
}

fn below_for(
    consumer: VisualScopeId,
    below: &BTreeMap<VisualScopeId, Vec<VisualScopeId>>,
) -> Result<&[VisualScopeId], String> {
    below
        .get(&consumer)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("visual dependency consumer {consumer:?} is not in composition"))
}

type ValidationTopology = (
    Vec<VisualScopeId>,
    BTreeMap<VisualScopeId, Vec<VisualScopeId>>,
    Vec<ImageOrderingEdge>,
);

/// Interpret one-level back-to-front composition order exactly as the live
/// planner does. A group's member prefix can see preceding root outputs plus
/// preceding siblings; the group post-scope itself can see only preceding root
/// outputs because its member composite has not yet become an external input.
fn validation_topology(composition: &CompositionTree) -> Result<ValidationTopology, String> {
    let mut scopes = Vec::new();
    let mut below: BTreeMap<VisualScopeId, Vec<VisualScopeId>> = BTreeMap::new();
    let mut ordering_edges = Vec::new();
    let mut preceding_root_outputs = Vec::new();
    for item in composition.root() {
        match *item {
            RootItem::Layer { layer, .. } => {
                let scope = validation_layer_scope(layer)?;
                scopes.push(scope);
                below.insert(scope, preceding_root_outputs.clone());
                preceding_root_outputs.push(scope);
            }
            RootItem::Group { group_id } => {
                let group = composition.group(group_id).ok_or_else(|| {
                    format!("composition root references group {}", group_id.get())
                })?;
                let group_scope = VisualScopeId::Group(group_id);
                let mut preceding_members = preceding_root_outputs.clone();
                for layer in group.members.iter() {
                    let member_scope = validation_layer_scope(layer)?;
                    scopes.push(member_scope);
                    below.insert(member_scope, preceding_members.clone());
                    preceding_members.push(member_scope);
                    ordering_edges.push(ImageOrderingEdge {
                        producer: member_scope,
                        consumer: group_scope,
                    });
                }
                scopes.push(group_scope);
                below.insert(group_scope, preceding_root_outputs.clone());
                preceding_root_outputs.push(group_scope);
            }
        }
    }
    scopes.push(VisualScopeId::Master);
    below.insert(VisualScopeId::Master, preceding_root_outputs);
    Ok((scopes, below, ordering_edges))
}

fn collect_saved_rack_group_ids(rack: &VisualRack, group_ids: &mut BTreeSet<GroupId>) {
    group_ids.extend(rack.referenced_group_ids());
}

fn collect_saved_composition_group_ids(
    composition: &CompositionTree,
    group_ids: &mut BTreeSet<GroupId>,
) {
    for item in composition.root() {
        if let RootItem::Group { group_id } = item {
            group_ids.insert(*group_id);
        }
    }
    for group in composition.groups() {
        group_ids.insert(group.id);
        collect_saved_rack_group_ids(&group.rack, group_ids);
        if let Some(group_id) = group.matte.and_then(|matte| matte.tap.referenced_group()) {
            group_ids.insert(group_id);
        }
    }
}

fn collect_saved_morph_slot_group_ids(
    slot: &crate::morph::MorphSlot,
    group_ids: &mut BTreeSet<GroupId>,
) {
    if let Some(rack) = &slot.master_rack {
        collect_saved_rack_group_ids(rack, group_ids);
    }
    if let Some(racks) = &slot.layer_racks {
        for rack in racks {
            collect_saved_rack_group_ids(rack, group_ids);
        }
    }
    if let Some(composition) = &slot.composition {
        collect_saved_composition_group_ids(composition, group_ids);
    }
}

fn saved_modulation_group_id(target: SavedStableModTarget) -> Option<GroupId> {
    match target {
        SavedStableModTarget::Node {
            scope: SavedStableModScope::Group { group_id },
            ..
        }
        | SavedStableModTarget::GroupValue { group_id, .. }
        | SavedStableModTarget::MissingGroup { group_id, .. }
        | SavedStableModTarget::MissingNode {
            scope: SavedStableModScope::Group { group_id },
            ..
        } => Some(group_id),
        SavedStableModTarget::Node { .. }
        | SavedStableModTarget::CompositionValue { .. }
        | SavedStableModTarget::MissingSavedLayer { .. }
        | SavedStableModTarget::MissingNode { .. } => None,
    }
}

/// The allocator state retained for one logical rack owner. A zero rack
/// cursor, or an observed `u64::MAX` node identity, exhausts the domain rather
/// than allowing it to wrap and reuse an authored identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PersistedNodeCursor {
    greatest_observed: Option<NodeId>,
    exhausted: bool,
}

impl PersistedNodeCursor {
    fn observe_node(&mut self, node_id: NodeId) {
        self.greatest_observed = Some(
            self.greatest_observed
                .map_or(node_id, |greatest| greatest.max(node_id)),
        );
        self.exhausted |= node_id.get() == u64::MAX;
    }

    fn observe_rack(&mut self, rack: &VisualRack) {
        let next = rack.next_node_id_raw();
        if next == 0 {
            self.exhausted = true;
            self.greatest_observed = Some(NodeId::new(u64::MAX).expect("u64::MAX is nonzero"));
            return;
        }
        // A validated saved rack cursor is at least FIRST_AUTHORED, so the
        // immediately preceding live-or-retired identity is always nonzero.
        let preceding = NodeId::new(next - 1).expect("validated rack cursors exceed zero");
        self.observe_node(preceding);
    }

    fn reserve_on(self, rack: &mut VisualRack) {
        if self.exhausted {
            rack.observe_node_reference(NodeId::new(u64::MAX).expect("u64::MAX is nonzero"));
        } else if let Some(node_id) = self.greatest_observed {
            rack.observe_node_reference(node_id);
        }
    }
}

/// Node IDs are rack-local, so retained identities are merged only inside the
/// same master, saved-layer-position, or GroupId ownership domain.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PersistedNodeCursors {
    master: PersistedNodeCursor,
    layers: BTreeMap<SavedLayerPosition, PersistedNodeCursor>,
    groups: BTreeMap<GroupId, PersistedNodeCursor>,
}

impl PersistedNodeCursors {
    fn for_scope_mut(&mut self, scope: SavedStableModScope) -> &mut PersistedNodeCursor {
        match scope {
            SavedStableModScope::Master => &mut self.master,
            SavedStableModScope::SavedLayer { position } => {
                self.layers.entry(position).or_default()
            }
            SavedStableModScope::Group { group_id } => self.groups.entry(group_id).or_default(),
        }
    }

    fn observe_rack(&mut self, scope: SavedStableModScope, rack: &VisualRack) {
        self.for_scope_mut(scope).observe_rack(rack);
    }

    fn observe_modulation_target(&mut self, target: SavedStableModTarget) {
        match target {
            SavedStableModTarget::Node { scope, node_id, .. }
            | SavedStableModTarget::MissingNode { scope, node_id, .. } => {
                self.for_scope_mut(scope).observe_node(node_id)
            }
            SavedStableModTarget::GroupValue { .. }
            | SavedStableModTarget::CompositionValue { .. }
            | SavedStableModTarget::MissingSavedLayer { .. }
            | SavedStableModTarget::MissingGroup { .. } => {}
        }
    }

    fn reserve_master_on(&self, rack: &mut VisualRack) {
        self.master.reserve_on(rack);
    }

    fn reserve_layer_on(&self, position: SavedLayerPosition, rack: &mut VisualRack) {
        if let Some(cursor) = self.layers.get(&position) {
            cursor.reserve_on(rack);
        }
    }

    fn reserve_groups_on(&self, composition: &mut CompositionTree) {
        for (group_id, cursor) in &self.groups {
            if let Some(group) = composition.group_mut(*group_id) {
                cursor.reserve_on(&mut group.rack);
            }
        }
    }
}

fn collect_saved_composition_node_cursors(
    composition: &CompositionTree,
    cursors: &mut PersistedNodeCursors,
) {
    for group in composition.groups() {
        cursors.observe_rack(
            SavedStableModScope::Group { group_id: group.id },
            &group.rack,
        );
    }
}

fn collect_saved_morph_slot_node_cursors(
    slot: &crate::morph::MorphSlot,
    cursors: &mut PersistedNodeCursors,
) {
    if let Some(rack) = &slot.master_rack {
        cursors.observe_rack(SavedStableModScope::Master, rack);
    }
    if let Some(racks) = &slot.layer_racks {
        for (position, rack) in racks.iter().enumerate() {
            let Ok(position) = u32::try_from(position) else {
                continue;
            };
            let Some(position) = SavedLayerPosition::new(position) else {
                continue;
            };
            cursors.observe_rack(SavedStableModScope::SavedLayer { position }, rack);
        }
    }
    if let Some(composition) = &slot.composition {
        collect_saved_composition_node_cursors(composition, cursors);
    }
}

fn reserve_saved_morph_slot_node_cursors(
    slot: &mut crate::morph::MorphSlot,
    cursors: &PersistedNodeCursors,
) {
    if let Some(rack) = &mut slot.master_rack {
        cursors.reserve_master_on(rack);
    }
    if let Some(racks) = &mut slot.layer_racks {
        for (position, rack) in racks.iter_mut().enumerate() {
            let Ok(position) = u32::try_from(position) else {
                continue;
            };
            let Some(position) = SavedLayerPosition::new(position) else {
                continue;
            };
            cursors.reserve_layer_on(position, rack);
        }
    }
    if let Some(composition) = &mut slot.composition {
        cursors.reserve_groups_on(composition);
    }
}

// --- Full patch snapshot ---

impl PatchState {
    /// Every persisted GroupId shares one monotonic allocation domain, even
    /// when the retaining field is dormant or explicitly missing. Keeping the
    /// scan sorted makes cursor repair independent from serialization order.
    fn persisted_group_ids(&self) -> BTreeSet<GroupId> {
        let mut group_ids = BTreeSet::new();
        if let Some(composition) = &self.composition {
            collect_saved_composition_group_ids(composition, &mut group_ids);
        }
        if let Some(rack) = &self.master_rack {
            collect_saved_rack_group_ids(rack, &mut group_ids);
        }
        for layer in &self.layers {
            if let Some(rack) = &layer.rack {
                collect_saved_rack_group_ids(rack, &mut group_ids);
            }
            if let SavedImageInput::GroupOutput { group_id }
            | SavedImageInput::MissingGroupOutput { group_id } = layer.matte.input
            {
                group_ids.insert(group_id);
            }
        }
        if let Some(morph) = &self.morph {
            if let Some(slot) = &morph.a {
                collect_saved_morph_slot_group_ids(slot, &mut group_ids);
            }
            if let Some(slot) = &morph.b {
                collect_saved_morph_slot_group_ids(slot, &mut group_ids);
            }
        }
        if let Some(modulation) = &self.modulation {
            group_ids.extend(
                modulation
                    .routings
                    .iter()
                    .filter_map(|routing| routing.stable_target)
                    .filter_map(saved_modulation_group_id),
            );
        }
        group_ids
    }

    /// Merge every live or retired NodeId inside its rack-local ownership
    /// domain. Morph A/B cursors carry deleted-node provenance, while typed
    /// stable modulation Node/MissingNode targets retain identities even when
    /// the addressed node itself is absent.
    fn persisted_node_cursors(&self) -> PersistedNodeCursors {
        let mut cursors = PersistedNodeCursors::default();
        if let Some(rack) = &self.master_rack {
            cursors.observe_rack(SavedStableModScope::Master, rack);
        }
        for (position, layer) in self.layers.iter().enumerate() {
            let Ok(position) = u32::try_from(position) else {
                continue;
            };
            let Some(position) = SavedLayerPosition::new(position) else {
                continue;
            };
            if let Some(rack) = &layer.rack {
                cursors.observe_rack(SavedStableModScope::SavedLayer { position }, rack);
            }
        }
        if let Some(composition) = &self.composition {
            collect_saved_composition_node_cursors(composition, &mut cursors);
        }
        if let Some(morph) = &self.morph {
            if let Some(slot) = &morph.a {
                collect_saved_morph_slot_node_cursors(slot, &mut cursors);
            }
            if let Some(slot) = &morph.b {
                collect_saved_morph_slot_node_cursors(slot, &mut cursors);
            }
        }
        if let Some(modulation) = &self.modulation {
            for target in modulation
                .routings
                .iter()
                .filter_map(|routing| routing.stable_target)
            {
                cursors.observe_modulation_target(target);
            }
        }
        cursors
    }

    fn reserve_persisted_node_cursors(&mut self, cursors: &PersistedNodeCursors) {
        if let Some(rack) = &mut self.master_rack {
            cursors.reserve_master_on(rack);
        }
        for (position, layer) in self.layers.iter_mut().enumerate() {
            let Ok(position) = u32::try_from(position) else {
                continue;
            };
            let Some(position) = SavedLayerPosition::new(position) else {
                continue;
            };
            if let Some(rack) = &mut layer.rack {
                cursors.reserve_layer_on(position, rack);
            }
        }
        if let Some(composition) = &mut self.composition {
            cursors.reserve_groups_on(composition);
        }
        if let Some(morph) = &mut self.morph {
            if let Some(slot) = &mut morph.a {
                reserve_saved_morph_slot_node_cursors(slot, cursors);
            }
            if let Some(slot) = &mut morph.b {
                reserve_saved_morph_slot_node_cursors(slot, cursors);
            }
        }
    }

    fn reserve_persisted_group_ids(&self, composition: &mut CompositionTree) {
        for group_id in self.persisted_group_ids() {
            composition.observe_group_reference(group_id);
        }
    }

    fn validate_creative_persistence(&mut self) -> Result<(), String> {
        if self.visual_schema_version > 1 {
            return Err(format!(
                "unsupported visual_schema_version {}; maximum is 1",
                self.visual_schema_version
            ));
        }
        // A patch may also be assembled in Rust, so the document's own
        // deserializer is not the only way in. Re-run the single acceptance
        // path here: it revalidates every event through the live-ingest
        // validator and re-derives the canonical checksum.
        if let Some(track) = &self.gesture_track {
            track
                .validate()
                .map_err(|error| format!("invalid saved gesture track: {error}"))?;
        }
        if let Some(take) = &self.performance_take {
            take.validate()
                .map_err(|error| format!("invalid saved performance take: {error}"))?;
        }
        if let Some(rack) = &self.master_rack {
            rack.validate_for_scope(LegacyRackScope::Master)
                .map_err(|error| format!("invalid saved master rack: {error}"))?;
        }
        for (position, layer) in self.layers.iter().enumerate() {
            if let Some(rack) = &layer.rack {
                rack.validate_for_scope(LegacyRackScope::Layer)
                    .map_err(|error| format!("invalid saved layer rack {position}: {error}"))?;
            }
        }
        if self.master_rack.is_some()
            || self.composition.is_some()
            || self.layers.iter().any(|layer| layer.rack.is_some())
        {
            self.visual_schema_version = 1;
        }
        let positions = (0..self.layers.len())
            .map(|position| {
                u32::try_from(position)
                    .ok()
                    .and_then(SavedLayerPosition::new)
                    .ok_or_else(|| format!("saved layer position {position} exceeds patch bounds"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let persisted_group_ids = self.persisted_group_ids();
        let persisted_node_cursors = self.persisted_node_cursors();
        if let Some(composition) = &mut self.composition {
            composition
                .validate_for_layers(&positions)
                .map_err(|error| format!("invalid saved composition: {error}"))?;
            for group in composition.groups() {
                group
                    .rack
                    .validate_for_scope(LegacyRackScope::Group)
                    .map_err(|error| {
                        format!("invalid saved group {} rack: {error}", group.id.get())
                    })?;
            }
            // Every persisted reference participates in the same monotonic
            // GroupId domain. Missing references deliberately advance the
            // cursor so a future group cannot inherit their identity.
            for group_id in persisted_group_ids {
                composition.observe_group_reference(group_id);
            }
        }
        self.reserve_persisted_node_cursors(&persisted_node_cursors);
        self.validate_visual_graph()?;
        Ok(())
    }

    /// Validate the whole authored current-frame image graph without pruning
    /// missing intent. Saved positions are mapped to deterministic validation
    /// identities only for cycle analysis; those identities are never stored.
    fn validate_visual_graph(&self) -> Result<(), String> {
        // The M2 advanced planner is deliberately bounded, but a globally
        // exact legacy graph must not retroactively inherit that cap. This
        // recognizes both omitted M2 fields and the explicit graph emitted by
        // capture: neither form has an authored image edge to validate and
        // both continue through the established legacy renderer path.
        if self.layers.len() > crate::composition::MAX_COMPOSITION_LAYERS
            && self.is_global_legacy_exact()
        {
            return Ok(());
        }
        let positions = (0..self.layers.len())
            .map(saved_position_at)
            .collect::<Result<Vec<_>, _>>()?;
        let synthesized;
        let composition = match &self.composition {
            Some(composition) => composition,
            None => {
                synthesized = legacy_composition_for_positions(&positions)?;
                &synthesized
            }
        };
        let (scopes, below, mut ordering_edges) = validation_topology(composition)?;
        let mut dependencies = Vec::new();

        collect_rack_dependencies(
            self.effective_master_rack().as_ref(),
            VisualScopeId::Master,
            &below,
            &mut dependencies,
            &mut ordering_edges,
        )?;
        for (position, layer) in self.layers.iter().enumerate() {
            let saved_position = saved_position_at(position)?;
            let consumer = validation_layer_scope(saved_position)?;
            let rack = layer.rack.as_ref().map_or_else(
                || std::borrow::Cow::Owned(VisualRack::synthetic_legacy(LegacyRackScope::Layer)),
                std::borrow::Cow::Borrowed,
            );
            collect_rack_dependencies(
                rack.as_ref(),
                consumer,
                &below,
                &mut dependencies,
                &mut ordering_edges,
            )?;
            if layer.matte.enabled && layer.matte.amount > 0.0 {
                collect_legacy_layer_matte_dependency(
                    layer.matte.input,
                    consumer,
                    &below,
                    &mut dependencies,
                    &mut ordering_edges,
                )?;
            }
        }
        for group in composition.groups() {
            if group.bypass {
                continue;
            }
            let consumer = VisualScopeId::Group(group.id);
            collect_rack_dependencies(
                &group.rack,
                consumer,
                &below,
                &mut dependencies,
                &mut ordering_edges,
            )?;
            if let Some(matte) = group.matte.filter(|matte| matte.amount > 0.0) {
                collect_saved_tap_dependency(
                    matte.tap,
                    consumer,
                    &below,
                    &mut dependencies,
                    &mut ordering_edges,
                )?;
            }
        }
        ImageDependencyGraph::validate_with_ordering_edges(
            &scopes,
            &dependencies,
            &ordering_edges,
            ImageGraphMode::Advanced,
        )
        .map(|_| ())
        .map_err(|error| format!("invalid visual image graph: {error}"))
    }

    /// Saved-form counterpart of the evaluated planner's exact-legacy law.
    /// Layer configs are stored front-to-back while a composition root is
    /// authored back-to-front, so an explicit capture must be the exact
    /// reversed positional sequence on Program. Cursor values are deliberately
    /// irrelevant: they do not affect pixels or execution topology.
    fn is_global_legacy_exact(&self) -> bool {
        if !self
            .effective_master_rack()
            .is_exact_legacy(LegacyRackScope::Master)
        {
            return false;
        }
        if !self.layers.iter().all(|layer| {
            let matte = layer.matte.sanitized();
            layer
                .rack
                .as_ref()
                .is_none_or(|rack| rack.is_exact_legacy(LegacyRackScope::Layer))
                && (!matte.enabled || matte.amount <= 0.0)
        }) {
            return false;
        }

        let Some(composition) = &self.composition else {
            return true;
        };
        composition.groups().len() == 0
            && composition.bus_crossfade() == 0.5
            && composition.root().len() == self.layers.len()
            && composition
                .root()
                .iter()
                .zip((0..self.layers.len()).rev())
                .all(|(item, expected_position)| {
                    matches!(
                        item,
                        RootItem::Layer {
                            layer,
                            bus: crate::composition::BusAssignment::Program,
                        } if usize::try_from(layer.get()).ok() == Some(expected_position)
                    )
                })
    }

    /// Legacy omission is resolved at the consumption boundary, never during
    /// serde, so explicit empty racks remain observably different on disk.
    pub fn effective_master_rack(&self) -> std::borrow::Cow<'_, VisualRack> {
        self.master_rack.as_ref().map_or_else(
            || std::borrow::Cow::Owned(VisualRack::synthetic_legacy(LegacyRackScope::Master)),
            std::borrow::Cow::Borrowed,
        )
    }

    pub fn effective_layer_rack(
        &self,
        position: usize,
    ) -> Option<std::borrow::Cow<'_, VisualRack>> {
        if position >= self.layers.len() {
            return None;
        }
        match &self.layers[position].rack {
            Some(rack) => Some(std::borrow::Cow::Borrowed(rack)),
            None => Some(std::borrow::Cow::Owned(VisualRack::synthetic_legacy(
                LegacyRackScope::Layer,
            ))),
        }
    }

    pub fn capture(
        master: PatchMasterVisual<'_>,
        layers: &[Layer],
        ntsc_params: &NtscParams,
        mod_matrix: &ModMatrix,
        temporal: &TemporalParams,
        transport: PatchTransportState,
        morph: &crate::morph::Morph,
    ) -> Self {
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let mut captured_layers: Vec<_> = layers.iter().map(LayerConfig::from_layer).collect();
        for (captured, layer) in captured_layers.iter_mut().zip(layers) {
            captured.matte = LayerMatteConfig::from_runtime(layer.matte, |wanted| {
                layers
                    .iter()
                    .position(|candidate| candidate.stable_layer_id() == wanted)
                    .and_then(|position| u32::try_from(position).ok())
                    .and_then(SavedLayerPosition::new)
            })
            .unwrap_or_default();
            let motion = MotionConfig::from_params_for_capture(layer.motion, &layer_ids);
            captured.motion = (!motion.is_default()).then_some(motion);
        }
        Self {
            master: EffectsConfig::from_uniforms(master.effects),
            master_transform: master.transform.sanitized(),
            master_motion: None,
            layers: captured_layers,
            master_rack: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: transport.master_paused,
            media_frozen: transport.media_frozen,
            ntsc: Some(NtscConfig::from_params(ntsc_params)),
            modulation: Some(ModConfig::from_matrix(mod_matrix)),
            temporal: Some(TemporalConfig::from_params_for_capture(
                temporal, &layer_ids,
            )),
            morph: Some(morph.snapshot_at_beat(mod_matrix.current_beat)),
            scenes: Scenes::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        }
    }

    /// M4 capture sibling. Existing callers remain exact-zero until Main owns
    /// a live `master_motion` value and switches to this entry point.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn capture_with_motion(
        master: PatchMasterVisual<'_>,
        master_motion: &MotionParams,
        layers: &[Layer],
        ntsc_params: &NtscParams,
        mod_matrix: &ModMatrix,
        temporal: &TemporalParams,
        transport: PatchTransportState,
        morph: &crate::morph::Morph,
    ) -> Self {
        let mut patch = Self::capture(
            master,
            layers,
            ntsc_params,
            mod_matrix,
            temporal,
            transport,
            morph,
        );
        let motion = MotionConfig::from_params(*master_motion);
        patch.master_motion = (!motion.is_default()).then_some(motion);
        patch
    }

    /// Capture the complete M2 creative graph through stable runtime IDs into
    /// saved positional identities. Every fallible mapping is completed before
    /// a patch value is published.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_with_composition(
        master: PatchMasterVisual<'_>,
        layers: &[Layer],
        master_rack: &RuntimeVisualRack,
        layer_racks: &[RuntimeVisualRack],
        composition: &RuntimeComposition,
        ntsc_params: &NtscParams,
        mod_matrix: &ModMatrix,
        temporal: &TemporalParams,
        transport: PatchTransportState,
        morph: &crate::morph::Morph,
    ) -> Result<Self, String> {
        if layer_racks.len() != layers.len() {
            return Err(format!(
                "patch capture has {} layer racks for {} layers",
                layer_racks.len(),
                layers.len()
            ));
        }
        master_rack
            .validate_for_scope(LegacyRackScope::Master)
            .map_err(|error| format!("invalid runtime master rack: {error}"))?;
        for (position, rack) in layer_racks.iter().enumerate() {
            rack.validate_for_scope(LegacyRackScope::Layer)
                .map_err(|error| format!("invalid runtime layer rack {position}: {error}"))?;
        }
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let position_of_layer = |wanted| {
            layer_ids
                .iter()
                .position(|candidate| *candidate == wanted)
                .and_then(|position| u32::try_from(position).ok())
                .and_then(SavedLayerPosition::new)
        };
        let saved_master_rack = master_rack
            .capture_routes(position_of_layer)
            .map_err(|error| format!("capture master rack routes: {error}"))?;
        let saved_layer_racks = layer_racks
            .iter()
            .enumerate()
            .map(|(position, rack)| {
                rack.capture_routes(position_of_layer)
                    .map_err(|error| format!("capture layer rack {position} routes: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let saved_composition = composition
            .capture(position_of_layer)
            .map_err(|error| format!("capture composition: {error}"))?;
        let address_racks: Vec<_> = layer_ids
            .iter()
            .copied()
            .zip(layer_racks.iter().cloned())
            .collect();
        let address_book =
            StableModAddressBook::from_composition(master_rack, &address_racks, composition)?;
        let saved_modulation = ModConfig::from_matrix_with_composition(
            mod_matrix,
            &address_book,
            &layer_ids,
            composition,
        )?;

        let mut patch = Self::capture(
            master,
            layers,
            ntsc_params,
            mod_matrix,
            temporal,
            transport,
            morph,
        );
        patch.master_rack = Some(saved_master_rack);
        for (layer, rack) in patch.layers.iter_mut().zip(saved_layer_racks) {
            layer.rack = Some(rack);
        }
        patch.composition = Some(saved_composition);
        patch.visual_schema_version = 1;
        patch.modulation = Some(saved_modulation);
        patch.validate_creative_persistence()?;
        Ok(patch)
    }

    /// M4-aware complete creative capture. Motion authoring is additive to the
    /// existing graph transaction and carries no hidden field/carrier pixels.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_with_composition_and_motion(
        master: PatchMasterVisual<'_>,
        master_motion: &MotionParams,
        layers: &[Layer],
        master_rack: &RuntimeVisualRack,
        layer_racks: &[RuntimeVisualRack],
        composition: &RuntimeComposition,
        ntsc_params: &NtscParams,
        mod_matrix: &ModMatrix,
        temporal: &TemporalParams,
        transport: PatchTransportState,
        morph: &crate::morph::Morph,
    ) -> Result<Self, String> {
        let mut patch = Self::capture_with_composition(
            master,
            layers,
            master_rack,
            layer_racks,
            composition,
            ntsc_params,
            mod_matrix,
            temporal,
            transport,
            morph,
        )?;
        let motion = MotionConfig::from_params(*master_motion);
        patch.master_motion = (!motion.is_default()).then_some(motion);
        Ok(patch)
    }

    /// Canonicalize active slot mirrors and finite matte fields. Scene intent
    /// is never pruned here; invalid references are reported by
    /// [`Self::validate_scene_references`] so staging can reject atomically.
    pub fn sanitize_performance_references(&mut self) {
        for layer in &mut self.layers {
            layer.sync_legacy_mirrors_from_active_slot();
            layer.matte = layer.matte.sanitized();
        }
    }

    /// Validate every saved Scene reference without mutating the patch. The
    /// returned order is deterministic: Scene order, then binding order.
    pub fn validate_scene_references(&self) -> Vec<SceneReferenceIssue> {
        let mut issues = Vec::new();
        for scene in self.scenes.iter() {
            for binding in scene.bindings.iter() {
                let Some(layer) = binding.layer_position.resolve(&self.layers) else {
                    issues.push(SceneReferenceIssue {
                        scene_id: scene.id,
                        layer_position: binding.layer_position,
                        slot_id: binding.slot_id,
                        cue_id: binding.cue_id,
                        kind: SceneReferenceErrorKind::Layer,
                    });
                    continue;
                };
                let Some(slot) = layer.clip_slots.get(binding.slot_id) else {
                    issues.push(SceneReferenceIssue {
                        scene_id: scene.id,
                        layer_position: binding.layer_position,
                        slot_id: binding.slot_id,
                        cue_id: binding.cue_id,
                        kind: SceneReferenceErrorKind::Slot,
                    });
                    continue;
                };
                if binding
                    .cue_id
                    .is_some_and(|cue_id| slot.transport.cue(cue_id).is_none())
                {
                    issues.push(SceneReferenceIssue {
                        scene_id: scene.id,
                        layer_position: binding.layer_position,
                        slot_id: binding.slot_id,
                        cue_id: binding.cue_id,
                        kind: SceneReferenceErrorKind::Cue,
                    });
                }
            }
        }
        issues
    }

    pub fn apply(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        layers: &mut [Layer],
        ntsc_params: &mut NtscParams,
        mod_matrix: &mut ModMatrix,
        temporal: &mut TemporalParams,
    ) {
        self.master.apply_to_uniforms(master);
        *master_transform = self.master_transform.sanitized();
        for (config, layer) in self.layers.iter().zip(layers.iter_mut()) {
            config.apply_to_layer(layer);
        }
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        for (config, layer) in self.layers.iter().zip(layers.iter_mut()) {
            if let Some(motion) = config.motion {
                layer.motion = motion.resolve_runtime(&layer_ids);
            }
        }
        if let Some(ref ntsc) = self.ntsc {
            *ntsc_params = ntsc.to_params();
        }
        if let Some(ref modulation) = self.modulation {
            modulation.apply_to_matrix(mod_matrix);
        }
        if let Some(ref temporal_cfg) = self.temporal {
            let mut restored = temporal_cfg.to_params();
            if let Some(originals) = temporal_cfg.originals {
                restored.originals.garden.matte_route =
                    originals.garden.matte_route.resolve_runtime(&layer_ids);
                restored.originals.garden.motion_route =
                    originals.garden.motion_route.resolve_runtime(&layer_ids);
                restored.originals.score.loop_driver =
                    originals.score.loop_driver.resolve_runtime(&layer_ids);
            }
            *temporal = restored;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub fn apply_with_motion(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        layers: &mut [Layer],
        ntsc_params: &mut NtscParams,
        mod_matrix: &mut ModMatrix,
        temporal: &mut TemporalParams,
    ) {
        self.apply(
            master,
            master_transform,
            layers,
            ntsc_params,
            mod_matrix,
            temporal,
        );
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        *master_motion = self
            .master_motion
            .unwrap_or_default()
            .resolve_runtime(&layer_ids);
    }

    /// Resolve and apply the complete persisted creative graph atomically.
    /// The compatibility [`Self::apply`] entry point remains unchanged for the
    /// legacy runtime until Main wires this sibling.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_with_composition(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
        ntsc_params: &mut NtscParams,
        mod_matrix: &mut ModMatrix,
        temporal: &mut TemporalParams,
    ) -> Result<(), String> {
        if layers.len() != self.layers.len() || layer_racks.len() != layers.len() {
            return Err(format!(
                "patch restore requires {} layers/racks; live has {} layers and {} racks",
                self.layers.len(),
                layers.len(),
                layer_racks.len()
            ));
        }
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let saved_positions = (0..self.layers.len())
            .map(|position| {
                u32::try_from(position)
                    .ok()
                    .and_then(SavedLayerPosition::new)
                    .ok_or_else(|| format!("saved layer position {position} exceeds bounds"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut saved_composition = match &self.composition {
            Some(composition) => composition.clone(),
            None => legacy_composition_for_positions(&saved_positions)?,
        };
        let persisted_node_cursors = self.persisted_node_cursors();
        self.reserve_persisted_group_ids(&mut saved_composition);
        persisted_node_cursors.reserve_groups_on(&mut saved_composition);
        let resolved_composition = saved_composition
            .resolve(|position| position.resolve(&layer_ids).copied())
            .map_err(|error| format!("resolve composition: {error}"))?;
        let group_exists = |group_id| resolved_composition.contains_group(group_id);
        let mut saved_master_rack = self.effective_master_rack().into_owned();
        persisted_node_cursors.reserve_master_on(&mut saved_master_rack);
        let resolved_master_rack = saved_master_rack.resolve_routes(
            |position| position.resolve(&layer_ids).copied(),
            group_exists,
        );
        let resolved_layer_racks = self
            .layers
            .iter()
            .enumerate()
            .map(|(position, layer)| {
                let mut rack = layer.rack.as_ref().map_or_else(
                    || VisualRack::synthetic_legacy(LegacyRackScope::Layer),
                    Clone::clone,
                );
                if let Some(saved_position) = u32::try_from(position)
                    .ok()
                    .and_then(SavedLayerPosition::new)
                {
                    persisted_node_cursors.reserve_layer_on(saved_position, &mut rack);
                }
                rack.resolve_routes(
                    |position| position.resolve(&layer_ids).copied(),
                    group_exists,
                )
            })
            .collect::<Vec<_>>();
        let address_racks: Vec<_> = layer_ids
            .iter()
            .copied()
            .zip(resolved_layer_racks.iter().cloned())
            .collect();
        let address_book = StableModAddressBook::from_composition(
            &resolved_master_rack,
            &address_racks,
            &resolved_composition,
        )?;

        self.apply(
            master,
            master_transform,
            layers,
            ntsc_params,
            mod_matrix,
            temporal,
        );
        for (config, layer) in self.layers.iter().zip(layers.iter_mut()) {
            layer.matte = config
                .matte
                .to_runtime(|position| position.resolve(&layer_ids).copied());
        }
        *master_rack = resolved_master_rack;
        layer_racks.clone_from_slice(&resolved_layer_racks);
        *composition = resolved_composition;
        if let Some(modulation) = &self.modulation {
            modulation.apply_to_matrix_with_composition(
                mod_matrix,
                &address_book,
                &layer_ids,
                composition,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_with_composition_and_motion(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
        ntsc_params: &mut NtscParams,
        mod_matrix: &mut ModMatrix,
        temporal: &mut TemporalParams,
    ) -> Result<(), String> {
        self.apply_with_composition(
            master,
            master_transform,
            layers,
            master_rack,
            layer_racks,
            composition,
            ntsc_params,
            mod_matrix,
            temporal,
        )?;
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        *master_motion = self
            .master_motion
            .unwrap_or_default()
            .resolve_runtime(&layer_ids);
        Ok(())
    }

    /// Transfer a saved visual look onto the current positional stack without
    /// replacing sources or transport/control state. Extra saved/current
    /// layers are intentionally ignored/retained respectively.
    pub fn apply_look(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        layers: &mut [Layer],
        ntsc_params: &mut NtscParams,
        temporal: &mut TemporalParams,
    ) -> LookApplySummary {
        self.master.apply_to_uniforms(master);
        *master_transform = self.master_transform.sanitized();
        let live_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let summary = apply_positional_looks(&self.layers, layers, |config, layer| {
            config.apply_look_to_layer(layer);
            let _ = apply_layer_matte_look_values(config.matte, &mut layer.matte, &live_ids);
        });
        if let Some(ref ntsc) = self.ntsc {
            *ntsc_params = ntsc.to_params();
        }
        if let Some(ref temporal_cfg) = self.temporal {
            let loop_driver = temporal.originals.score.loop_driver;
            let garden_matte_route = temporal.originals.garden.matte_route;
            let garden_motion_route = temporal.originals.garden.motion_route;
            *temporal = temporal_cfg.to_params();
            // A Look transfers values, not donor-route topology. Preserve an
            // exact live Selected identity and an authored tombstone alike.
            temporal.originals.score.loop_driver = loop_driver;
            temporal.originals.garden.matte_route = garden_matte_route;
            temporal.originals.garden.motion_route = garden_motion_route;
        }
        summary
    }

    #[allow(dead_code)]
    pub fn apply_look_with_motion(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        layers: &mut [Layer],
        ntsc_params: &mut NtscParams,
        temporal: &mut TemporalParams,
    ) -> LookApplySummary {
        let summary = self.apply_look(master, master_transform, layers, ntsc_params, temporal);
        if let Some(motion) = self.master_motion {
            apply_motion_look(motion, master_motion);
        }
        summary
    }

    /// Apply visual values from a Look while retaining the current creative
    /// topology, route donors, IDs, ordering and monotonic cursors. Racks are
    /// strict compatibility units; groups match independently by stable ID,
    /// membership, rack signature, matte presence and immutable image route.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_look_with_composition(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
        ntsc_params: &mut NtscParams,
        temporal: &mut TemporalParams,
    ) -> LookApplySummary {
        let mut summary = self.apply_look(master, master_transform, layers, ntsc_params, temporal);
        let layer_ids: Vec<_> = layers.iter().map(Layer::stable_layer_id).collect();
        let group_exists = |group_id| composition.contains_group(group_id);
        let sampled_master = self.effective_master_rack().resolve_routes(
            |position| position.resolve(&layer_ids).copied(),
            group_exists,
        );
        if crate::morph::apply_runtime_rack_values_strict(&sampled_master, master_rack) {
            summary.applied_racks += 1;
            record_look_rack_nodes(&mut summary, LookRackScope::Master, &sampled_master, true);
        } else {
            summary.skipped_racks += 1;
            record_look_rack_nodes(&mut summary, LookRackScope::Master, &sampled_master, false);
        }

        let mapped_racks = self.layers.len().min(layers.len()).min(layer_racks.len());
        for (position, live) in layer_racks.iter_mut().take(mapped_racks).enumerate() {
            let saved = self.layers[position]
                .rack
                .as_ref()
                .map_or_else(
                    || VisualRack::synthetic_legacy(LegacyRackScope::Layer),
                    Clone::clone,
                )
                .resolve_routes(
                    |position| position.resolve(&layer_ids).copied(),
                    group_exists,
                );
            if crate::morph::apply_runtime_rack_values_strict(&saved, live) {
                summary.applied_racks += 1;
                record_look_rack_nodes(
                    &mut summary,
                    LookRackScope::Layer(layer_ids[position]),
                    &saved,
                    true,
                );
            } else {
                summary.skipped_racks += 1;
                record_look_rack_nodes(
                    &mut summary,
                    LookRackScope::Layer(layer_ids[position]),
                    &saved,
                    false,
                );
            }
        }
        summary.skipped_racks += self.layers.len().saturating_sub(mapped_racks);

        if let Some(saved_composition) = &self.composition {
            apply_saved_composition_look(saved_composition, composition, &layer_ids, &mut summary);
        }
        summary
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_look_with_composition_and_motion(
        &self,
        master: &mut EffectUniforms,
        master_transform: &mut SpatialTransform,
        master_motion: &mut MotionParams,
        layers: &mut [Layer],
        master_rack: &mut RuntimeVisualRack,
        layer_racks: &mut [RuntimeVisualRack],
        composition: &mut RuntimeComposition,
        ntsc_params: &mut NtscParams,
        temporal: &mut TemporalParams,
    ) -> LookApplySummary {
        let summary = self.apply_look_with_composition(
            master,
            master_transform,
            layers,
            master_rack,
            layer_racks,
            composition,
            ntsc_params,
            temporal,
        );
        if let Some(motion) = self.master_motion {
            apply_motion_look(motion, master_motion);
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_studies_section_round_trips_and_the_digest_walk_finds_every_rack() {
        use crate::study::{
            StudyAbiVersion, StudyCapability, StudyInstruction, StudyLicenseNotice, StudyMetadata,
            StudyPublicationBoundary, StudyRegister, STUDY_SCHEMA_VERSION,
        };
        let register = |value: u8| StudyRegister::new(value).unwrap();
        let document = crate::study::StudyDocument {
            schema_version: STUDY_SCHEMA_VERSION,
            abi: StudyAbiVersion::default(),
            metadata: StudyMetadata {
                name: "Patch fixture".into(),
                author: "Patch tests".into(),
                description: String::new(),
                license: StudyLicenseNotice {
                    identifier: "CC0-1.0".into(),
                    notice: String::new(),
                    publication_boundary: StudyPublicationBoundary::StudyDataOnlyDoesNotLicenseHost,
                },
            },
            capabilities: vec![StudyCapability::CurrentColor],
            instructions: vec![
                StudyInstruction::LoadCurrentColor { dst: register(0) },
                StudyInstruction::OutputColor { color: register(0) },
            ],
        };
        let compiled = crate::study_eval::CompiledStudy::compile(&document).unwrap();
        let digest = *compiled.canonical_digest();

        let mut patch = minimal_patch(1);
        let mut rack = VisualRack::synthetic_legacy(crate::visual_rack::LegacyRackScope::Master);
        rack.push(crate::visual_rack::VisualNodeKind::Study(
            crate::visual_rack::StudyRackParams {
                document_digest: Some(digest),
            },
        ))
        .unwrap();
        patch.master_rack = Some(rack);
        patch.studies = vec![document.clone()];

        assert_eq!(patch.referenced_study_digests(), vec![digest]);

        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(yaml.contains("studies:"));
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.studies, vec![document]);
        assert_eq!(restored.referenced_study_digests(), vec![digest]);

        // An empty section stays off the page entirely — the pre-study path
        // is byte-identical.
        let bare = minimal_patch(1);
        assert!(!serde_yaml::to_string(&bare).unwrap().contains("studies:"));
    }

    fn saved_layer(
        filename: &str,
        opacity: f32,
        blend_mode: &str,
        visible: bool,
        bypass_master_fx: bool,
        brightness: f32,
        random_seed: u32,
    ) -> LayerConfig {
        let source_path = format!("patch://{filename}");
        LayerConfig {
            filename: filename.to_string(),
            source_path: source_path.clone(),
            opacity,
            blend_mode: blend_mode.to_string(),
            speed: 4.0,
            fps: 240.0,
            paused: true,
            visible,
            bypass_master_fx,
            reroll_on_loop: true,
            effects: EffectsConfig {
                brightness,
                random_seed,
                ..Default::default()
            },
            transform: SpatialTransform {
                position: [opacity, brightness],
                ..SpatialTransform::default()
            },
            motion: None,
            rack: None,
            clip_slots: ClipSlots::singleton(ClipSlotConfig::from_legacy(
                filename.to_string(),
                source_path,
                4.0,
                240.0,
            )),
            active_clip_slot: Some(ClipSlotId::LEGACY),
            matte: LayerMatteConfig::default(),
            pattern: None,
            text_page: None,
        }
    }

    struct FakeLiveLayer {
        topology_id: u64,
        source_path: String,
        filename: String,
        speed: f32,
        fps: f32,
        paused: bool,
        reroll_on_loop: bool,
        opacity: f32,
        blend_mode: BlendMode,
        visible: bool,
        bypass_master_fx: bool,
        effects: EffectUniforms,
        transform: SpatialTransform,
    }

    fn fake_live_layer(topology_id: u64, filename: &str, opacity: f32) -> FakeLiveLayer {
        FakeLiveLayer {
            topology_id,
            source_path: format!("live://{filename}"),
            filename: filename.to_string(),
            speed: 0.5 + topology_id as f32 / 10.0,
            fps: 20.0 + topology_id as f32,
            paused: topology_id.is_multiple_of(2),
            reroll_on_loop: topology_id.is_multiple_of(3),
            opacity,
            blend_mode: BlendMode::Multiply,
            visible: true,
            bypass_master_fx: false,
            effects: EffectUniforms {
                brightness: -0.9,
                random_seed: topology_id as u32 + 1_000,
                ..Default::default()
            },
            transform: SpatialTransform::new_layer_default(),
        }
    }

    #[test]
    fn curated_blend_keys_round_trip_exactly_through_yaml_json_and_look_application() {
        for expected in BlendMode::ALL {
            let config = saved_layer("blend-study.mov", 0.75, expected.key(), true, false, 0.0, 7);

            let yaml = serde_yaml::to_string(&config).unwrap();
            let from_yaml: LayerConfig = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(
                from_yaml.blend_mode,
                expected.key(),
                "YAML collapsed {expected:?}"
            );
            assert!(
                yaml.lines()
                    .any(|line| line == format!("blend_mode: {}", expected.key())),
                "YAML did not serialize the exact key for {expected:?}: {yaml}"
            );

            let json = serde_json::to_string(&config).unwrap();
            let from_json: LayerConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(
                from_json.blend_mode,
                expected.key(),
                "JSON collapsed {expected:?}"
            );
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&json).unwrap()["blend_mode"],
                expected.key()
            );

            let mut applied = BlendMode::Normal;
            let mut opacity = 0.0;
            let mut visible = false;
            let mut bypass_master_fx = true;
            let mut effects = EffectUniforms::default();
            let mut transform = SpatialTransform::default();
            from_json.apply_look_to_fields(
                &mut opacity,
                &mut applied,
                &mut visible,
                &mut bypass_master_fx,
                &mut effects,
                &mut transform,
            );
            assert_eq!(applied, expected, "look application collapsed {expected:?}");
        }

        let missing: LayerConfig =
            serde_yaml::from_str("filename: legacy.mov\neffects: {}\n").unwrap();
        assert_eq!(missing.blend_mode, BlendMode::Normal.key());

        let unknown = saved_layer("future.mov", 1.0, "future_blend", true, false, 0.0, 0);
        let mut applied = BlendMode::Difference;
        let mut opacity = 0.0;
        let mut visible = false;
        let mut bypass_master_fx = true;
        let mut effects = EffectUniforms::default();
        let mut transform = SpatialTransform::default();
        unknown.apply_look_to_fields(
            &mut opacity,
            &mut applied,
            &mut visible,
            &mut bypass_master_fx,
            &mut effects,
            &mut transform,
        );
        assert_eq!(
            applied,
            BlendMode::Normal,
            "unknown legacy strings retain the established Normal fallback"
        );
    }

    #[test]
    fn look_application_is_visual_only_positional_and_reports_mismatches() {
        let saved = vec![
            saved_layer("saved-a.mp4", 0.2, "difference", false, true, 0.25, 41),
            saved_layer("saved-b.mp4", 0.7, "screen", true, false, -0.3, 97),
        ];
        let mut live = vec![
            fake_live_layer(11, "live-a.mov", 0.91),
            fake_live_layer(22, "live-b.mov", 0.92),
            fake_live_layer(33, "live-c.mov", 0.93),
        ];
        let topology_and_sources_before: Vec<_> = live
            .iter()
            .map(|layer| {
                (
                    layer.topology_id,
                    layer.source_path.clone(),
                    layer.filename.clone(),
                )
            })
            .collect();
        let transport_before: Vec<_> = live
            .iter()
            .map(|layer| (layer.speed, layer.fps, layer.paused, layer.reroll_on_loop))
            .collect();
        let untouched_visual_before = (
            live[2].opacity,
            live[2].blend_mode,
            live[2].visible,
            live[2].bypass_master_fx,
            live[2].effects.brightness,
            live[2].effects.random_seed,
            live[2].transform,
        );

        let summary = apply_positional_looks(&saved, &mut live, |config, layer| {
            config.apply_look_to_fields(
                &mut layer.opacity,
                &mut layer.blend_mode,
                &mut layer.visible,
                &mut layer.bypass_master_fx,
                &mut layer.effects,
                &mut layer.transform,
            );
        });

        assert_eq!(
            summary,
            LookApplySummary {
                mapped_layers: 2,
                unused_patch_layers: 0,
                untouched_live_layers: 1,
                ..LookApplySummary::default()
            }
        );
        assert_eq!(live[0].opacity, 0.2);
        assert_eq!(live[0].blend_mode, BlendMode::Difference);
        assert!(!live[0].visible);
        assert!(live[0].bypass_master_fx);
        assert_eq!(live[0].effects.brightness, 0.25);
        assert_eq!(live[0].effects.random_seed, 41);
        assert_eq!(live[0].transform.position, [0.2, 0.25]);
        assert_eq!(live[1].opacity, 0.7);
        assert_eq!(live[1].blend_mode, BlendMode::Screen);
        assert!(live[1].visible);
        assert!(!live[1].bypass_master_fx);
        assert_eq!(live[1].effects.brightness, -0.3);
        assert_eq!(live[1].effects.random_seed, 97);
        assert_eq!(live[1].transform.position, [0.7, -0.3]);
        assert_eq!(
            (
                live[2].opacity,
                live[2].blend_mode,
                live[2].visible,
                live[2].bypass_master_fx,
                live[2].effects.brightness,
                live[2].effects.random_seed,
                live[2].transform,
            ),
            untouched_visual_before
        );

        let topology_and_sources_after: Vec<_> = live
            .iter()
            .map(|layer| {
                (
                    layer.topology_id,
                    layer.source_path.clone(),
                    layer.filename.clone(),
                )
            })
            .collect();
        let transport_after: Vec<_> = live
            .iter()
            .map(|layer| (layer.speed, layer.fps, layer.paused, layer.reroll_on_loop))
            .collect();
        assert_eq!(topology_and_sources_after, topology_and_sources_before);
        assert_eq!(transport_after, transport_before);

        let mut one_live_position = [()];
        let reverse_mismatch = apply_positional_looks(&saved, &mut one_live_position, |_, _| {});
        assert_eq!(
            reverse_mismatch,
            LookApplySummary {
                mapped_layers: 1,
                unused_patch_layers: 1,
                untouched_live_layers: 0,
                ..LookApplySummary::default()
            }
        );
    }

    #[test]
    fn look_application_preserves_absent_ntsc_and_temporal_but_applies_present_sections() {
        let mut patch = PatchState {
            master: EffectsConfig {
                brightness: 0.55,
                random_seed: 31_337,
                ..Default::default()
            },
            master_transform: SpatialTransform {
                rotation_deg: 23.0,
                ..SpatialTransform::default()
            },
            master_motion: None,
            layers: Vec::new(),
            master_rack: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: true,
            media_frozen: true,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
            scenes: Scenes::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        };
        let mut master = EffectUniforms {
            brightness: -0.4,
            ..Default::default()
        };
        let mut master_transform = SpatialTransform::default();
        let mut ntsc = NtscParams {
            enabled: true,
            snow_intensity: 0.33,
            ..Default::default()
        };
        let original_ntsc = ntsc.clone();
        let mut temporal = TemporalParams {
            feedback: 0.42,
            slit_angle: 73.0,
            ..Default::default()
        };
        let live_driver = CollisionScoreLoopDriver::SelectedLayer {
            layer_id: StableLayerId::new(81).unwrap(),
            saved_position: SavedLayerPosition::new(2).unwrap(),
        };
        temporal.originals.score.loop_driver = live_driver;

        let summary = patch.apply_look(
            &mut master,
            &mut master_transform,
            &mut [],
            &mut ntsc,
            &mut temporal,
        );
        assert_eq!(summary, LookApplySummary::default());
        assert_eq!(master.brightness, 0.55);
        assert_eq!(master.random_seed, 31_337);
        assert_eq!(master_transform.rotation_deg, 23.0);
        assert_eq!(ntsc, original_ntsc);
        assert_eq!(temporal.feedback, 0.42);
        assert_eq!(temporal.slit_angle, 73.0);

        let saved_ntsc = NtscParams {
            enabled: false,
            tape_speed: 2,
            chroma_loss: 0.007,
            snow_intensity: 0.6,
            ..Default::default()
        };
        let saved_temporal = TemporalParams {
            feedback: 0.7,
            fb_zoom: 1.04,
            slit_angle: -35.0,
            ..Default::default()
        };
        patch.ntsc = Some(NtscConfig::from_params(&saved_ntsc));
        patch.temporal = Some(TemporalConfig::from_params(&saved_temporal));

        patch.apply_look(
            &mut master,
            &mut master_transform,
            &mut [],
            &mut ntsc,
            &mut temporal,
        );
        assert_eq!(ntsc, saved_ntsc);
        assert_eq!(temporal.feedback, 0.7);
        assert_eq!(temporal.fb_zoom, 1.04);
        assert_eq!(temporal.slit_angle, -35.0);
        assert_eq!(
            temporal.originals.score.loop_driver, live_driver,
            "Apply Look must not retarget the live Score conductor"
        );
    }

    #[test]
    fn legacy_patch_defaults_media_frozen_without_conflating_master_pause() {
        let legacy: PatchState =
            serde_yaml::from_str("master: {}\nlayers: []\nmaster_paused: true\n").unwrap();

        assert!(legacy.master_paused);
        assert!(!legacy.media_frozen);
        assert_eq!(legacy.master_transform, SpatialTransform::default());
    }

    #[test]
    fn legacy_one_source_layer_migrates_to_exact_canonical_slot_one() {
        let legacy: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: archive.mov
    source_path: C:/media/archive.mov
    speed: 1.75
    fps: 24
    paused: true
    effects: {}
"#,
        )
        .unwrap();
        let layer = &legacy.layers[0];
        assert_eq!(layer.clip_slots.len(), 1);
        assert_eq!(layer.active_clip_slot, Some(ClipSlotId::LEGACY));
        let slot = layer.clip_slots.get(ClipSlotId::LEGACY).unwrap();
        assert_eq!(slot.filename, "archive.mov");
        assert_eq!(slot.source_path, "C:/media/archive.mov");
        assert_eq!(
            slot.transport.direction,
            crate::transport::PlaybackDirection::Forward
        );
        assert_eq!(
            slot.transport.end_behavior,
            crate::transport::EndBehavior::Loop
        );
        assert_eq!(
            slot.transport.in_point,
            crate::transport::NormalizedTime::ZERO
        );
        assert_eq!(
            slot.transport.out_point,
            crate::transport::NormalizedTime::ONE
        );
        assert_eq!(slot.transport.rate, 1.75);
        assert_eq!(slot.transport.sample_fps, Some(24.0));
        assert_eq!(slot.saved_playhead, crate::transport::NormalizedTime::ZERO);
        assert!(layer.paused, "legacy layer pause remains a layer control");
        assert!(layer.matte.is_legacy_disabled());
        assert!(legacy.scenes.is_empty());

        let canonical = serde_yaml::to_string(&legacy).unwrap();
        assert!(canonical.contains("clip_slots:"));
        assert!(canonical.contains("active_clip_slot: 1"));
        assert!(!canonical.contains("matte:"));
        assert!(!canonical.contains("scenes:"));
    }

    #[test]
    fn canonical_slots_win_over_legacy_mirrors_and_active_id_is_lookup_based() {
        let patch: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: stale.mov
    source_path: stale/path.mov
    speed: 4
    fps: 120
    effects: {}
    clip_slots:
      - id: 4000
        name: Archive B
        filename: canonical-b.mov
        source_path: cos-sha256://bbbb
        transport:
          rate: 0.5
          sample_fps: 18
      - id: 7
        name: Archive A
        filename: canonical-a.mov
        source_path: cos-sha256://aaaa
        transport:
          rate: 1.25
          sample_fps: 25
    active_clip_slot: 7
"#,
        )
        .unwrap();
        let layer = &patch.layers[0];
        assert_eq!(layer.active_clip_slot.unwrap().get(), 7);
        assert_eq!(layer.filename, "canonical-a.mov");
        assert_eq!(layer.source_path, "cos-sha256://aaaa");
        assert_eq!(layer.speed, 1.25);
        assert_eq!(layer.fps, 25.0);
        assert_eq!(layer.clip_slots.iter().next().unwrap().id.get(), 4000);

        let fallback: LayerConfig = serde_yaml::from_str(
            r#"
filename: stale.mov
effects: {}
clip_slots:
  - { id: 31, filename: first.mov, transport: {} }
  - { id: 2, filename: second.mov, transport: {} }
active_clip_slot: 30
"#,
        )
        .unwrap();
        assert_eq!(fallback.active_clip_slot.unwrap().get(), 31);
        assert_eq!(fallback.filename, "first.mov");
    }

    #[test]
    fn scene_load_preserves_invalid_intent_and_reports_deterministic_diagnostics() {
        let patch: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: donor.mov
    effects: {}
    clip_slots:
      - id: 7
        filename: donor.mov
        transport:
          cues:
            - { id: 3, at: 0.25 }
    active_clip_slot: 7
  - filename: carrier.mov
    effects: {}
scenes:
  - id: 9
    name: bounded
    bindings:
      - { layer_position: 0, slot_id: 7, cue_id: 3 }
      - { layer_position: 1, slot_id: 999 }
      - { layer_position: 2, slot_id: 1 }
"#,
        )
        .unwrap();
        let scene = patch
            .scenes
            .get(crate::performance::SceneId::new(9).unwrap())
            .unwrap();
        let bindings: Vec<_> = scene.bindings.iter().copied().collect();
        assert_eq!(bindings.len(), 3, "load must never prune authored intent");
        assert_eq!(bindings[0].layer_position.get(), 0);
        assert_eq!(bindings[0].slot_id.get(), 7);
        assert_eq!(bindings[0].cue_id.unwrap().get(), 3);
        let issues = patch.validate_scene_references();
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].kind, SceneReferenceErrorKind::Slot);
        assert_eq!(issues[0].layer_position.get(), 1);
        assert_eq!(issues[1].kind, SceneReferenceErrorKind::Layer);
        assert_eq!(issues[1].layer_position.get(), 2);

        let canonical = serde_yaml::to_string(&patch).unwrap();
        let restored: PatchState = serde_yaml::from_str(&canonical).unwrap();
        assert_eq!(restored.scenes, patch.scenes);
    }

    #[test]
    fn hostile_explicit_empty_slot_array_is_rejected() {
        let error = match serde_yaml::from_str::<LayerConfig>(
            "filename: blank.mov\neffects: {}\nclip_slots: []\nactive_clip_slot: 9\n",
        ) {
            Ok(_) => panic!("explicit empty clip slot array must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("may not be empty"));
    }

    #[test]
    fn spatial_patch_schema_preserves_legacy_identity_and_round_trips_all_modes() {
        let legacy: PatchState = serde_yaml::from_str(
            "master: {}\nlayers:\n  - filename: legacy.mov\n    effects: {}\n",
        )
        .unwrap();
        assert_eq!(legacy.master_transform, SpatialTransform::default());
        assert_eq!(legacy.layers[0].transform, SpatialTransform::default());
        let legacy_yaml = serde_yaml::to_string(&legacy).unwrap();
        assert!(
            !legacy_yaml.contains("transform:"),
            "legacy identity should not churn canonical patch bytes"
        );
        assert!(!legacy_yaml.contains("master_transform:"));

        let mut authored = legacy;
        authored.master_transform = SpatialTransform {
            position: [0.2, -0.3],
            scale: [-2.0, 3.0],
            anchor: [0.1, 0.9],
            rotation_deg: 45.0,
            skew_deg: -12.0,
            skew_axis_deg: 30.0,
            fit: FitMode::Fill,
            crop: [0.1, 0.2, 0.3, 0.1],
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
        };
        authored.layers[0].transform = SpatialTransform {
            fit: FitMode::Native,
            edge: EdgeMode::Transparent,
            ..authored.master_transform
        };
        let yaml = serde_yaml::to_string(&authored).unwrap();
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.master_transform, authored.master_transform);
        assert_eq!(restored.layers[0].transform, authored.layers[0].transform);
    }

    #[test]
    fn patch_schema_round_trips_stacks_beyond_the_retired_sixteen_layer_limit() {
        let layers = (0..24)
            .map(|index| {
                saved_layer(
                    &format!("clip-{index}.mp4"),
                    1.0 - index as f32 / 100.0,
                    "normal",
                    true,
                    false,
                    index as f32 / 100.0,
                    index as u32 + 1,
                )
            })
            .collect();
        let patch = PatchState {
            master: EffectsConfig::default(),
            master_transform: SpatialTransform::default(),
            master_motion: None,
            layers,
            master_rack: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: false,
            media_frozen: false,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
            scenes: Scenes::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        };

        let yaml = serde_yaml::to_string(&patch).unwrap();
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(restored.layers.len(), 24);
        assert_eq!(restored.layers[16].filename, "clip-16.mp4");
        assert_eq!(restored.layers[23].effects.random_seed, 24);
    }

    #[test]
    fn patch_capture_rebases_in_flight_morph_to_remaining_beats() {
        let mut matrix = ModMatrix::new();
        matrix.update_at_beat(103.0, 0.0);
        let mut morph = crate::morph::Morph::default();
        morph.start_glide(1.0, 8.0, 100.0);

        let patch = PatchState::capture(
            PatchMasterVisual::new(&EffectUniforms::default(), &SpatialTransform::default()),
            &[],
            &NtscParams::default(),
            &matrix,
            &TemporalParams::default(),
            PatchTransportState {
                master_paused: false,
                media_frozen: false,
            },
            &morph,
        );
        let snapshot = patch.morph.unwrap();
        assert!((snapshot.t - 0.375).abs() < 1e-6);
        let glide = snapshot.glide.unwrap();
        assert_eq!(glide.start_beat, 0.0);
        assert_eq!(glide.duration_beats, 5.0);
        let restored = crate::morph::Morph::from_snapshot(snapshot);
        assert!((restored.position_at_beat(0.0) - 0.375).abs() < 1e-6);
        assert!((restored.position_at_beat(5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn downsample_persists_for_master_and_layers_with_legacy_default() {
        let master = EffectUniforms {
            downsample: 0.35,
            ..Default::default()
        };
        let layer_effects = EffectsConfig {
            downsample: 0.6,
            ..Default::default()
        };
        let patch = PatchState {
            master: EffectsConfig::from_uniforms(&master),
            master_transform: SpatialTransform::default(),
            master_motion: None,
            layers: vec![LayerConfig {
                filename: "clip.mp4".to_string(),
                source_path: String::new(),
                opacity: 1.0,
                blend_mode: "normal".to_string(),
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                bypass_master_fx: true,
                reroll_on_loop: false,
                effects: layer_effects,
                transform: SpatialTransform::default(),
                motion: None,
                rack: None,
                clip_slots: ClipSlots::singleton(ClipSlotConfig::from_legacy(
                    "clip.mp4".to_string(),
                    String::new(),
                    1.0,
                    30.0,
                )),
                active_clip_slot: Some(ClipSlotId::LEGACY),
                matte: LayerMatteConfig::default(),
                pattern: None,
                text_page: None,
            }],
            master_rack: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: false,
            media_frozen: false,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
            scenes: Scenes::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        };

        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(yaml.contains("downsample: 0.35"));
        let parsed: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.master.downsample, 0.35);
        assert_eq!(parsed.layers[0].effects.downsample, 0.6);
        assert!(parsed.layers[0].bypass_master_fx);

        let mut restored = EffectUniforms::default();
        parsed.master.apply_to_uniforms(&mut restored);
        assert_eq!(restored.downsample, 0.35);

        let legacy: PatchState = serde_yaml::from_str(
            "master: {}\nlayers:\n  - filename: legacy.mp4\n    effects: {}\n",
        )
        .unwrap();
        assert_eq!(legacy.master.downsample, 1.0);
        assert_eq!(legacy.layers[0].effects.downsample, 1.0);
        assert!(!legacy.layers[0].bypass_master_fx);

        let mut invalid = EffectsConfig {
            downsample: f32::NAN,
            ..Default::default()
        };
        invalid.apply_to_uniforms(&mut restored);
        assert_eq!(restored.downsample, 1.0);
        invalid.downsample = -10.0;
        invalid.apply_to_uniforms(&mut restored);
        assert_eq!(restored.downsample, 0.05);

        assert!(invalid.set_field("downsample", "0.4"));
        assert_eq!(invalid.downsample, 0.4);
        assert!(invalid
            .grouped_fields()
            .iter()
            .flat_map(|(_, fields)| fields)
            .any(|(key, _)| *key == "downsample"));
        let metadata = param_meta("downsample").unwrap();
        assert_eq!((metadata.min, metadata.max), (0.05, 1.0));
    }

    #[test]
    fn layer_master_fx_bypass_round_trips_and_native_editor_exposes_it() {
        let legacy: LayerConfig = serde_yaml::from_str(
            "filename: legacy.mp4\nopacity: 1\nblend_mode: normal\neffects: {}\n",
        )
        .unwrap();
        assert!(!legacy.bypass_master_fx);

        let mut edited = legacy;
        assert!(edited.set_field("bypass_master_fx", "true"));
        assert!(edited.bypass_master_fx);
        assert!(edited
            .top_fields()
            .iter()
            .any(|(key, value)| *key == "bypass_master_fx" && value == "true"));

        let yaml = serde_yaml::to_string(&edited).unwrap();
        let restored: LayerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.bypass_master_fx);
    }

    #[test]
    fn cellular_controls_round_trip_sanitize_and_keep_legacy_defaults() {
        let configured = EffectsConfig {
            cellular_amount: 0.8,
            cellular_scale: 24.0,
            cellular_warp: 0.65,
            cellular_speed: 1.5,
            cellular_gap_amount: 0.9,
            cellular_gap_threshold: 0.4,
            cellular_gap_softness: 0.12,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&configured).unwrap();
        let decoded: EffectsConfig = serde_yaml::from_str(&yaml).unwrap();
        let mut uniforms = EffectUniforms::default();
        decoded.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.cellular_amount, 0.8);
        assert_eq!(uniforms.cellular_scale, 24.0);
        assert_eq!(uniforms.cellular_warp, 0.65);
        assert_eq!(uniforms.cellular_speed, 1.5);
        assert_eq!(uniforms.cellular_gap_amount, 0.9);
        assert_eq!(uniforms.cellular_gap_threshold, 0.4);
        assert_eq!(uniforms.cellular_gap_softness, 0.12);

        let legacy: EffectsConfig = serde_yaml::from_str("pixelate: 4.0\n").unwrap();
        assert_eq!(legacy.cellular_amount, 0.0);
        assert_eq!(legacy.cellular_scale, 10.0);
        assert_eq!(legacy.cellular_warp, 0.35);
        assert_eq!(legacy.cellular_speed, 0.25);
        assert_eq!(legacy.cellular_gap_amount, 0.0);
        assert_eq!(legacy.cellular_gap_threshold, 0.65);
        assert_eq!(legacy.cellular_gap_softness, 0.08);

        let invalid: EffectsConfig = serde_yaml::from_str(
            "cellular_amount: .nan\ncellular_scale: -9\ncellular_warp: 7\ncellular_speed: .inf\ncellular_gap_amount: 9\ncellular_gap_threshold: -4\ncellular_gap_softness: .nan\n",
        )
        .unwrap();
        invalid.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.cellular_amount, 0.0);
        assert_eq!(uniforms.cellular_scale, 2.0);
        assert_eq!(uniforms.cellular_warp, 1.0);
        assert_eq!(uniforms.cellular_speed, 0.25);
        assert_eq!(uniforms.cellular_gap_amount, 1.0);
        assert_eq!(uniforms.cellular_gap_threshold, 0.0);
        assert_eq!(uniforms.cellular_gap_softness, 0.08);

        let mut editable = EffectsConfig::default();
        for (key, value) in [
            ("cellular_amount", "0.7"),
            ("cellular_scale", "18"),
            ("cellular_warp", "0.4"),
            ("cellular_speed", "0.9"),
            ("cellular_gap_amount", "0.8"),
            ("cellular_gap_threshold", "0.55"),
            ("cellular_gap_softness", "0.1"),
        ] {
            assert!(editable.set_field(key, value));
            let metadata = param_meta(key).unwrap();
            assert!(metadata.min < metadata.max);
            assert!(metadata.step > 0.0);
        }
        let cellular_group = editable
            .grouped_fields()
            .into_iter()
            .find(|(name, _)| *name == "cellular")
            .unwrap();
        assert_eq!(cellular_group.1.len(), 7);
    }

    #[test]
    fn shift_controls_round_trip_sanitize_and_keep_legacy_defaults() {
        let configured = EffectsConfig {
            shift_amount: 0.8,
            shift_block_size: 48.0,
            shift_density: 0.7,
            shift_speed: 9.5,
            ..Default::default()
        };
        let decoded: EffectsConfig =
            serde_yaml::from_str(&serde_yaml::to_string(&configured).unwrap()).unwrap();
        let mut uniforms = EffectUniforms::default();
        decoded.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.shift_amount, 0.8);
        assert_eq!(uniforms.shift_block_size, 48.0);
        assert_eq!(uniforms.shift_density, 0.7);
        assert_eq!(uniforms.shift_speed, 9.5);
        assert_eq!(EffectsConfig::from_uniforms(&uniforms).shift_speed, 9.5);

        let legacy: EffectsConfig = serde_yaml::from_str("pixelate: 4.0\n").unwrap();
        assert_eq!(legacy.shift_amount, 0.0);
        assert_eq!(legacy.shift_block_size, 8.0);
        assert_eq!(legacy.shift_density, 0.5);
        assert_eq!(legacy.shift_speed, 3.0);

        let invalid: EffectsConfig = serde_yaml::from_str(
            "shift_amount: .nan\nshift_block_size: -9\nshift_density: 7\nshift_speed: .inf\n",
        )
        .unwrap();
        invalid.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.shift_amount, 0.0);
        assert_eq!(uniforms.shift_block_size, 2.0);
        assert_eq!(uniforms.shift_density, 1.0);
        assert_eq!(uniforms.shift_speed, 3.0);

        let mut editable = EffectsConfig::default();
        for (key, value) in [
            ("shift_amount", "0.6"),
            ("shift_block_size", "32"),
            ("shift_density", "0.4"),
            ("shift_speed", "7.25"),
        ] {
            assert!(editable.set_field(key, value));
            let metadata = param_meta(key).unwrap();
            assert!(metadata.min < metadata.max);
            assert!(metadata.step > 0.0);
        }
        let shift_group = editable
            .grouped_fields()
            .into_iter()
            .find(|(name, _)| *name == "shift")
            .unwrap();
        assert_eq!(shift_group.1.len(), 4);
    }

    #[test]
    fn chroma_and_temporal_keys_round_trip_with_safe_legacy_defaults() {
        let configured = EffectsConfig {
            key_mode: 3,
            key_color: [0.1, 0.8, 0.2],
            key_tolerance: 0.24,
            key_softness: 0.06,
            ..Default::default()
        };
        let restored: EffectsConfig =
            serde_yaml::from_str(&serde_yaml::to_string(&configured).unwrap()).unwrap();
        let mut uniforms = EffectUniforms::default();
        restored.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.key_mode, 3.0);
        assert_eq!(uniforms.key_color, [0.1, 0.8, 0.2]);
        assert_eq!(uniforms.key_tolerance, 0.24);

        let legacy: EffectsConfig = serde_yaml::from_str("pixelate: 2\n").unwrap();
        assert_eq!(legacy.key_mode, 0);
        assert_eq!(legacy.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(legacy.key_tolerance, 0.15);

        let invalid: EffectsConfig =
            serde_yaml::from_str("key_mode: 99\nkey_color: [.nan, .inf, -2]\nkey_tolerance: 9\n")
                .unwrap();
        invalid.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.key_mode, 4.0);
        assert_eq!(uniforms.key_color, [0.0, 1.0, 0.0]);
        assert_eq!(uniforms.key_tolerance, 1.0);

        let legacy_temporal: TemporalConfig = serde_yaml::from_str("feedback: 0.2\n").unwrap();
        let temporal = legacy_temporal.to_params();
        assert_eq!(temporal.key_mode, 0.0);
        assert_eq!(temporal.key_threshold, 0.1);
        assert_eq!(temporal.key_softness, 0.03);
        assert_eq!(temporal.key_history, 1.0);

        let invalid_temporal: TemporalConfig = serde_yaml::from_str(
            "key_mode: 99\nkey_threshold: .nan\nkey_softness: 8\nkey_history: 99\n",
        )
        .unwrap();
        let temporal = invalid_temporal.to_params();
        assert_eq!(temporal.key_mode, 4.0);
        assert_eq!(temporal.key_threshold, 0.1);
        assert_eq!(temporal.key_softness, 0.5);
        assert_eq!(temporal.key_history, 23.0);
        assert_eq!(temporal.originals, TemporalOriginalsParams::default());
    }

    #[test]
    fn temporal_originals_round_trip_only_authored_state_and_preserve_driver_tombstones() {
        let saved_position = SavedLayerPosition::new(7).unwrap();
        let mut originals = TemporalOriginalsConfig::default();
        originals.loom.amount = 0.7;
        originals.loom.topology = TemporalTopologyConfig::Kaleidoscopic;
        originals.loom.interpolation = TemporalInterpolationConfig::Linear;
        originals.loom.folds = 12;
        originals.atlas.amount = 0.5;
        originals.atlas.seed = 0xdead_beef;
        originals.garden.amount = 0.8;
        originals.garden.gate = RefreshGardenGateConfig::AudioOnset;
        originals.score.enabled = true;
        originals.score.seed = 0x1234_5678;
        originals.score.trigger = CollisionScoreTriggerConfig::Boundary;
        originals.score.loop_driver =
            CollisionScoreLoopDriverConfig::SelectedLayer { saved_position };
        originals.reset.loop_boundary = TemporalEventResetModeConfig::Memory;
        originals.reset.downbeat = TemporalEventResetModeConfig::Score;

        let configured = TemporalConfig {
            originals: Some(originals),
            ..TemporalConfig::default()
        };
        let yaml = serde_yaml::to_string(&configured).unwrap();
        assert!(yaml.contains("originals:"));
        assert!(yaml.contains("kind: selected_layer"));
        assert!(!yaml.contains("layer_id"));
        assert!(!yaml.contains("event_ordinal"));
        assert!(!yaml.contains("history_valid"));
        assert!(!yaml.contains("carrier"));
        assert!(serde_yaml::from_str::<TemporalConfig>(
            "originals:\n  score:\n    loop_driver:\n      kind: selected_layer\n      saved_position: 0\n      layer_id: 91\n"
        )
        .is_err());

        let restored: TemporalConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.originals, Some(originals));
        let params = restored.to_params();
        assert_eq!(params.originals.loom.amount, 0.7);
        assert_eq!(
            params.originals.loom.topology,
            TemporalTopology::Kaleidoscopic
        );
        assert_eq!(params.originals.atlas.seed, 0xdead_beef);
        assert_eq!(params.originals.garden.gate, RefreshGardenGate::AudioOnset);
        assert_eq!(params.originals.score.seed, 0x1234_5678);
        assert_eq!(
            params.originals.score.loop_driver,
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position }
        );
        assert_eq!(
            params.originals.reset.loop_boundary,
            TemporalEventResetMode::Memory
        );

        let tombstone = TemporalConfig {
            originals: Some(TemporalOriginalsConfig {
                score: CollisionScoreConfig {
                    loop_driver: CollisionScoreLoopDriverConfig::MissingSelectedLayer {
                        saved_position,
                    },
                    ..CollisionScoreConfig::default()
                },
                ..TemporalOriginalsConfig::default()
            }),
            ..TemporalConfig::default()
        };
        let tombstone_yaml = serde_yaml::to_string(&tombstone).unwrap();
        assert!(tombstone_yaml.contains("kind: missing_selected_layer"));
        let restored: TemporalConfig = serde_yaml::from_str(&tombstone_yaml).unwrap();
        assert!(matches!(
            restored.originals.unwrap().score.loop_driver,
            CollisionScoreLoopDriverConfig::MissingSelectedLayer { .. }
        ));
    }

    #[test]
    fn refresh_garden_routes_capture_positions_resolve_ids_and_preserve_tombstones() {
        let other = StableLayerId::new(44).unwrap();
        let wanted = StableLayerId::new(91).unwrap();
        let ids = [other, wanted];
        let stale = SavedLayerPosition::new(7).unwrap();
        let matte = RefreshGardenMatteRoute::SelectedLayer {
            layer_id: wanted,
            saved_position: stale,
            stage: LayerImageStage::PostLocalEffects,
        };
        let motion = RefreshGardenMotionRoute::SelectedLayer {
            layer_id: wanted,
            saved_position: stale,
        };
        let config = RefreshGardenConfig {
            matte_route: RefreshGardenMatteRouteConfig::from_runtime_for_capture(matte, &ids),
            motion_route: RefreshGardenMotionRouteConfig::from_runtime_for_capture(motion, &ids),
            ..RefreshGardenConfig::default()
        };
        assert!(matches!(
            config.matte_route,
            RefreshGardenMatteRouteConfig::SelectedLayer {
                saved_position,
                stage: LayerImageStage::PostLocalEffects,
            } if saved_position.get() == 1
        ));
        assert!(matches!(
            config.motion_route,
            RefreshGardenMotionRouteConfig::SelectedLayer { saved_position }
                if saved_position.get() == 1
        ));

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(!yaml.contains("layer_id"));
        let restored: RefreshGardenConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored, config);
        let sanitized = TemporalOriginalsConfig {
            garden: config,
            ..TemporalOriginalsConfig::default()
        }
        .sanitized();
        assert_eq!(sanitized.garden.matte_route, config.matte_route);
        assert_eq!(sanitized.garden.motion_route, config.motion_route);
        let captured_position = SavedLayerPosition::new(1).unwrap();
        assert_eq!(
            restored.matte_route.resolve_runtime(&ids),
            RefreshGardenMatteRoute::SelectedLayer {
                layer_id: wanted,
                saved_position: captured_position,
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            restored.motion_route.resolve_runtime(&ids),
            RefreshGardenMotionRoute::SelectedLayer {
                layer_id: wanted,
                saved_position: captured_position,
            }
        );

        let missing_matte = RefreshGardenMatteRouteConfig::MissingSelectedLayer {
            saved_position: stale,
            stage: LayerImageStage::PreLocalEffects,
        };
        let missing_motion = RefreshGardenMotionRouteConfig::MissingSelectedLayer {
            saved_position: stale,
        };
        assert!(matches!(
            missing_matte.resolve_runtime(&ids),
            RefreshGardenMatteRoute::MissingSelectedLayer { saved_position, .. }
                if saved_position == stale
        ));
        assert_eq!(
            missing_motion.resolve_runtime(&ids),
            RefreshGardenMotionRoute::MissingSelectedLayer {
                saved_position: stale,
            }
        );
        assert!(serde_yaml::from_str::<RefreshGardenConfig>(
            "matte_route:\n  kind: selected_layer\n  saved_position: 1\n  layer_id: 91\n"
        )
        .is_err());
    }

    #[test]
    fn motion_round_trip_is_authored_only_strict_and_resolves_selected_identity() {
        let wanted = StableLayerId::new(91).unwrap();
        let other = StableLayerId::new(44).unwrap();
        let stale_position = SavedLayerPosition::new(7).unwrap();
        let params = MotionParams {
            algorithm_version: MOTION_ALGORITHM_VERSION,
            field_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::High,
            procedural: ProceduralFieldParams {
                scale: 0.75,
                rate: -1.5,
            },
            shaping: FlowShapingParams {
                stretch: 0.3,
                edge_repel: 0.4,
                vector_trash: 0.2,
                trash_block_size: 32.0,
            },
            transplant: FaradayParams {
                amount: 0.75,
                donor: MotionDonor::Selected {
                    layer_id: wanted,
                    saved_position: stale_position,
                },
                carrier: MotionCarrier::FirstSourceFrame,
                confidence_threshold: 0.3,
                confidence_softness: 0.2,
                refresh: 0.6,
                decay: 0.8,
                occlusion: 0.4,
            },
            shutter: CurvedShutterParams {
                angle_degrees: 225.0,
                phase: -0.4,
                curvature: 1.5,
                chromatic_lag: 0.25,
                quality: CurvedShutterQuality::High,
            },
            collider: FieldColliderParams::default(),
        };
        let config = MotionConfig::from_params_for_capture(params, &[other, wanted]);
        assert_eq!(
            config.transplant.donor,
            MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(1).unwrap()
            }
        );

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("algorithm_version: 1"));
        assert!(yaml.contains("field_source: codec_vectors"));
        assert!(yaml.contains("kind: selected"));
        assert!(!yaml.contains("layer_id"));
        assert!(!yaml.contains("velocity_uv_per_second"));
        assert!(!yaml.contains("carrier_valid"));
        assert!(serde_yaml::from_str::<MotionConfig>(
            "transplant:\n  donor:\n    kind: selected\n    saved_position: 0\n    layer_id: 91\n"
        )
        .is_err());
        assert!(serde_yaml::from_str::<MotionConfig>("algorithm_version: 2\n").is_err());

        let restored: MotionConfig = serde_yaml::from_str(&yaml).unwrap();
        let mut runtime = restored.to_params();
        assert_eq!(
            runtime.transplant.donor,
            MotionDonor::Missing {
                saved_position: SavedLayerPosition::new(1).unwrap()
            }
        );
        runtime.transplant.donor = restored.transplant.donor.resolve_runtime(&[other, wanted]);
        assert_eq!(
            runtime.transplant.donor,
            MotionDonor::Selected {
                layer_id: wanted,
                saved_position: SavedLayerPosition::new(1).unwrap()
            }
        );
        assert_eq!(runtime.shutter.quality.sample_count(), 16);
        assert_eq!(runtime.procedural.scale, 0.75);
        assert_eq!(runtime.procedural.rate, -1.5);
        assert_eq!(runtime.shaping.stretch, 0.3);
        assert_eq!(runtime.shaping.edge_repel, 0.4);
        assert_eq!(runtime.shaping.vector_trash, 0.2);
        assert_eq!(runtime.shaping.trash_block_size, 32.0);
    }

    #[test]
    fn temporal_rig_round_trips_and_an_absent_section_is_the_prior_path() {
        use crate::motion::MotionBoundaryMode;

        // A default temporal block serializes without the rig section, so
        // every pre-B3 patch keeps its bytes and canonical hashes.
        let default_yaml = serde_yaml::to_string(&TemporalConfig::default()).unwrap();
        assert!(!default_yaml.contains("rig"));
        let absent: TemporalConfig = serde_yaml::from_str("feedback: 0.2\n").unwrap();
        assert!(absent.rig.is_default());
        assert!(absent.to_params().rig.is_identity());

        // A non-default rig round trips whole, including both closed
        // vocabularies and the two servo switches.
        let params = TemporalParams {
            rig: FeedbackRigParams {
                offset_x: 0.25,
                reflect_y: true,
                hue_rotate: -45.0,
                saturation: 1.5,
                gain_b: 1.8,
                chroma_displace: 0.03,
                blur: 0.6,
                sharpen: 1.2,
                shape: FeedbackShape::Fold,
                drive: 2.5,
                pivot: 0.4,
                threshold: 0.2,
                noise: 0.3,
                edge: MotionBoundaryMode::Mirror,
                servo: true,
                servo_defeated: true,
                ..FeedbackRigParams::default()
            },
            ..TemporalParams::default()
        };
        let config = TemporalConfig::from_params(&params);
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("rig:"));
        assert!(yaml.contains("shape: fold"));
        assert!(yaml.contains("edge: mirror"));
        let restored: TemporalConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.rig, config.rig);
        let runtime = restored.to_params();
        assert_eq!(runtime.rig, params.rig.sanitized());

        // Hostile scalars sanitize to neutral values and unknown fields are
        // rejected rather than ignored.
        let hostile: TemporalConfig =
            serde_yaml::from_str("rig:\n  drive: .nan\n  saturation: 99.0\n").unwrap();
        let runtime = hostile.to_params();
        assert_eq!(runtime.rig.drive, 1.0);
        assert_eq!(runtime.rig.saturation, 2.0);
        assert!(serde_yaml::from_str::<TemporalConfig>("rig:\n  seed: 4\n").is_err());
    }

    #[test]
    fn display_physics_round_trips_and_an_absent_section_is_the_prior_path() {
        use crate::display_physics::{DisplayModel, DisplayPhysicsParams, InterlaceMode};

        // A default temporal block serializes without the display section,
        // so every pre-B4 patch keeps its bytes and canonical hashes.
        let default_yaml = serde_yaml::to_string(&TemporalConfig::default()).unwrap();
        assert!(!default_yaml.contains("display"));
        let absent: TemporalConfig = serde_yaml::from_str("feedback: 0.2\n").unwrap();
        assert_eq!(absent.display, DisplayPhysicsParams::default());
        assert!(!absent.to_params().display.stage_active());

        // A non-default block round trips whole, including all three closed
        // vocabularies.
        let params = TemporalParams {
            display: DisplayPhysicsParams {
                il_amount: 0.7,
                il_mode: InterlaceMode::Bob,
                il_order: true,
                il_judder: 0.4,
                phosphor: 0.8,
                phos_b: 0.5,
                model: DisplayModel::SlotMask,
                scanlines: 0.6,
                mask_strength: 0.3,
                bloom: 0.2,
                sag: 0.1,
                ..DisplayPhysicsParams::default()
            },
            ..TemporalParams::default()
        };
        let config = TemporalConfig::from_params(&params);
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("display:"));
        assert!(yaml.contains("il_mode: bob"));
        assert!(yaml.contains("model: slot_mask"));
        let restored: TemporalConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.display, config.display);
        assert_eq!(restored.to_params().display, params.display.sanitized());

        // Hostile scalars sanitize to the neutral default rather than a
        // clamped extreme, and unknown fields are rejected.
        let hostile: TemporalConfig =
            serde_yaml::from_str("display:\n  phosphor: .nan\n  beam_width: 99.0\n").unwrap();
        let runtime = hostile.to_params();
        assert_eq!(runtime.display.phosphor, 0.0);
        assert_eq!(runtime.display.beam_width, 3.0);
        assert!(
            serde_yaml::from_str::<TemporalConfig>("display:\n  curvature: 1.0\n").is_err(),
            "an unknown display field is a deserialization rejection"
        );
        assert!(
            serde_yaml::from_str::<TemporalConfig>("display:\n  model: plasma\n").is_err(),
            "an unknown display model token is a deserialization rejection"
        );
    }

    #[test]
    fn codec_mosh_round_trips_and_an_absent_section_is_the_prior_path() {
        use crate::codec_mosh::CodecMoshParams;

        // A default temporal block serializes without the mosh section, so
        // every pre-B5 patch keeps its bytes and canonical hashes.
        let default_yaml = serde_yaml::to_string(&TemporalConfig::default()).unwrap();
        assert!(!default_yaml.contains("mosh"));
        let absent: TemporalConfig = serde_yaml::from_str("feedback: 0.2\n").unwrap();
        assert_eq!(absent.mosh, CodecMoshParams::default());
        assert!(!absent.to_params().mosh.is_active());

        // A non-default block round trips whole, including the discrete
        // recycle law.
        let params = TemporalParams {
            mosh: CodecMoshParams {
                amount: 0.85,
                key_removal: 1.0,
                hold: 0.5,
                drop: 0.2,
                shuffle: 0.3,
                rate: 0.7,
                bitrate_starve: 0.6,
                resync: 0.25,
                recycle: true,
            },
            ..TemporalParams::default()
        };
        let config = TemporalConfig::from_params(&params);
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("mosh:"));
        assert!(yaml.contains("recycle: true"));
        let restored: TemporalConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.mosh, config.mosh);
        assert_eq!(restored.to_params().mosh, params.mosh.sanitized());

        // Hostile scalars sanitize to the neutral default rather than a
        // clamped extreme, and unknown fields are rejected.
        let hostile: TemporalConfig =
            serde_yaml::from_str("mosh:\n  amount: .nan\n  key_removal: 99.0\n").unwrap();
        let runtime = hostile.to_params();
        assert_eq!(runtime.mosh.amount, 0.0);
        assert_eq!(runtime.mosh.key_removal, 1.0);
        assert!(
            serde_yaml::from_str::<TemporalConfig>("mosh:\n  codec: vp8\n").is_err(),
            "an unknown mosh field is a deserialization rejection"
        );
    }

    #[test]
    fn small_effects_round_trip_and_absent_fields_keep_prior_bytes() {
        // Every B13 field is skip-serialized at its default, so a default
        // effects block serializes without any of the new keys and pre-B13
        // patch bytes and canonical hashes keep.
        let default_yaml = serde_yaml::to_string(&EffectsConfig::default()).unwrap();
        for key in [
            "contour",
            "flatten",
            "solarize",
            "negative",
            "colourpass",
            "edge_amount",
            "emboss",
            "halftone",
            "moire",
            "row_smear",
            "bitcrush",
            "multi_grid_x",
            "multi_grid_y",
            "barrel",
            "chroma_aberration",
            "anamorphic_streak",
        ] {
            assert!(!default_yaml.contains(key), "default must omit {key}");
        }

        // A non-default set round trips whole, including the discrete mode.
        let mut config = EffectsConfig {
            contour: 0.8,
            contour_bands: 24.0,
            flatten: 0.5,
            flatten_levels: 7.0,
            contour_dither: 0.4,
            solarize: 0.6,
            negative: 0.9,
            negative_mode: 2,
            colourpass: 0.7,
            colourpass_hue: -120.0,
            edge_amount: 0.3,
            emboss_angle: -30.0,
            halftone: 0.5,
            halftone_angle: 15.0,
            moire: 0.2,
            row_smear: 0.4,
            bitcrush: 0.8,
            bitcrush_levels: 4.0,
            multi_grid_x: 3.0,
            multi_grid_y: 2.0,
            barrel: -0.4,
            chroma_aberration: 0.6,
            anamorphic_streak: 0.35,
            ..EffectsConfig::default()
        };
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: EffectsConfig = serde_yaml::from_str(&yaml).unwrap();
        let mut uniforms = EffectUniforms::default();
        restored.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.contour, 0.8);
        assert_eq!(uniforms.contour_bands, 24.0);
        assert_eq!(uniforms.negative_mode, 2.0);
        assert_eq!(uniforms.colourpass_hue, -120.0);
        assert_eq!(uniforms.bitcrush_levels, 4.0);
        assert_eq!(uniforms.multi_grid_x, 3.0);
        assert_eq!(uniforms.barrel, -0.4);
        assert_eq!(uniforms.anamorphic_streak, 0.35);

        // Hostile scalars sanitize to the neutral value or clamp, never NaN.
        config.contour_bands = f32::NAN;
        config.bitcrush_levels = 999.0;
        config.barrel = f32::NEG_INFINITY;
        let mut hostile = EffectUniforms::default();
        config.apply_to_uniforms(&mut hostile);
        assert_eq!(hostile.contour_bands, 10.0);
        assert_eq!(hostile.bitcrush_levels, 16.0);
        assert_eq!(hostile.barrel, 0.0);

        // Look application targets layer scope: the optics are cleared while
        // every shared control lands.
        let mut layer_config = saved_layer("optics.mp4", 1.0, "normal", true, false, 0.0, 0);
        layer_config.effects = EffectsConfig {
            halftone: 0.9,
            barrel: 0.8,
            chroma_aberration: 0.7,
            anamorphic_streak: 0.6,
            ..EffectsConfig::default()
        };
        let mut opacity = 1.0;
        let mut blend_mode = BlendMode::Normal;
        let mut visible = true;
        let mut bypass = false;
        let mut effects = EffectUniforms::default();
        let mut transform = SpatialTransform::default();
        layer_config.apply_look_to_fields(
            &mut opacity,
            &mut blend_mode,
            &mut visible,
            &mut bypass,
            &mut effects,
            &mut transform,
        );
        assert_eq!(effects.halftone, 0.9);
        assert_eq!(effects.barrel, 0.0);
        assert_eq!(effects.chroma_aberration, 0.0);
        assert_eq!(effects.anamorphic_streak, 0.0);
    }

    #[test]
    fn time_displace_map_round_trips_and_an_absent_section_is_the_prior_path() {
        // A default temporal block serializes without either B12 field, so
        // every pre-B12 patch keeps its bytes and canonical hashes.
        let default_yaml = serde_yaml::to_string(&TemporalConfig::default()).unwrap();
        assert!(!default_yaml.contains("slit_map"));
        assert!(!default_yaml.contains("slit_interp"));
        let absent: TemporalConfig = serde_yaml::from_str("slitscan: 0.5\n").unwrap();
        assert_eq!(absent.slit_map, TimeDisplaceMapConfig::Ramp);
        assert!(!absent.slit_interp);
        let runtime = absent.to_params();
        assert_eq!(runtime.slit_map, TimeDisplaceMap::Ramp);
        assert!(!runtime.slit_interp);
        assert!(!runtime.time_displace_active());

        // Every non-default map round trips whole, with the toggle.
        for (map, token) in [
            (TimeDisplaceMap::Brightness, "brightness"),
            (TimeDisplaceMap::Radial, "radial"),
            (TimeDisplaceMap::TbcRamp, "tbc_ramp"),
            (TimeDisplaceMap::Sweep, "sweep"),
        ] {
            let params = TemporalParams {
                slitscan: 0.5,
                slit_map: map,
                slit_interp: true,
                ..TemporalParams::default()
            };
            let yaml = serde_yaml::to_string(&TemporalConfig::from_params(&params)).unwrap();
            assert!(yaml.contains(&format!("slit_map: {token}")), "{yaml}");
            assert!(yaml.contains("slit_interp: true"));
            let restored: TemporalConfig = serde_yaml::from_str(&yaml).unwrap();
            let runtime = restored.to_params();
            assert_eq!(runtime.slit_map, map);
            assert!(runtime.slit_interp);
            assert!(runtime.time_displace_active());
        }

        // An unknown map token is rejected rather than silently defaulted.
        assert!(serde_yaml::from_str::<TemporalConfig>("slit_map: melt\n").is_err());
    }

    #[test]
    fn flow_shaping_config_round_trips_and_an_absent_section_is_the_prior_path() {
        // A default motion block serializes without the shaping section.
        let default_yaml = serde_yaml::to_string(&MotionConfig::default()).unwrap();
        assert!(!default_yaml.contains("shaping"));
        let absent: MotionConfig = serde_yaml::from_str(
            "field_source: lattice
",
        )
        .unwrap();
        assert_eq!(absent.shaping, FlowShapingConfig::default());
        assert!(absent.shaping.to_params().is_exact_zero());

        // Hostile scalars sanitize to neutral values, unknown fields are
        // rejected, and a non-default block round trips exactly.
        let hostile: MotionConfig = serde_yaml::from_str(
            "shaping:
  stretch: .nan
  trash_block_size: 9999.0
",
        )
        .unwrap();
        let runtime = hostile.to_params();
        assert_eq!(runtime.shaping.stretch, 0.0);
        assert_eq!(runtime.shaping.trash_block_size, 256.0);
        assert!(serde_yaml::from_str::<MotionConfig>(
            "shaping:
  stretch: 0.5
  seed: 4
"
        )
        .is_err());
        let config = MotionConfig::from_params(MotionParams {
            shaping: FlowShapingParams {
                stretch: 0.25,
                edge_repel: 0.5,
                vector_trash: 0.75,
                trash_block_size: 48.0,
            },
            ..MotionParams::default()
        });
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("shaping:"));
        let restored: MotionConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.shaping, config.shaping);
    }

    #[test]
    fn procedural_field_config_round_trips_and_an_absent_section_is_the_prior_path() {
        // A default motion block serializes without the B2 section, so every
        // pre-B2 patch keeps its original bytes and its canonical hash.
        let default_yaml = serde_yaml::to_string(&MotionConfig::default()).unwrap();
        assert!(!default_yaml.contains("procedural"));
        let absent: MotionConfig = serde_yaml::from_str("field_source: lattice\n").unwrap();
        assert_eq!(absent.procedural, ProceduralFieldConfig::default());

        // All six kinds round trip through their stable tokens.
        for (kind, token) in [
            (ProceduralFieldKind::Curl, "procedural_curl"),
            (ProceduralFieldKind::Radial, "procedural_radial"),
            (ProceduralFieldKind::Spiral, "procedural_spiral"),
            (ProceduralFieldKind::Contour, "procedural_contour"),
            (ProceduralFieldKind::Chroma, "procedural_chroma"),
            (ProceduralFieldKind::Weave, "procedural_weave"),
        ] {
            let config = MotionConfig::from_params(MotionParams {
                field_source: MotionFieldSource::Procedural(kind),
                procedural: ProceduralFieldParams {
                    scale: 0.9,
                    rate: 1.5,
                },
                ..MotionParams::default()
            });
            let yaml = serde_yaml::to_string(&config).unwrap();
            assert!(yaml.contains(&format!("field_source: {token}")), "{token}");
            assert!(yaml.contains("procedural:"));
            let restored: MotionConfig = serde_yaml::from_str(&yaml).unwrap();
            let runtime = restored.to_params();
            assert_eq!(runtime.field_source, MotionFieldSource::Procedural(kind));
            assert_eq!(runtime.procedural.scale, 0.9);
            assert_eq!(runtime.procedural.rate, 1.5);
        }

        // Hostile scalars sanitize on load to neutral values, and unknown
        // fields inside the section are rejected rather than ignored.
        let hostile: MotionConfig =
            serde_yaml::from_str("procedural:\n  scale: .nan\n  rate: 99.0\n").unwrap();
        let runtime = hostile.to_params();
        assert_eq!(runtime.procedural.scale, 0.5);
        assert_eq!(runtime.procedural.rate, 2.0);
        assert!(
            serde_yaml::from_str::<MotionConfig>("procedural:\n  scale: 0.5\n  seed: 4\n").is_err()
        );
    }

    #[test]
    fn field_collider_round_trips_as_authored_only_topology_with_two_independent_slots() {
        let first = StableLayerId::new(41).unwrap();
        let second = StableLayerId::new(52).unwrap();
        let stale = SavedLayerPosition::new(9).unwrap();
        let params = MotionParams {
            transplant: FaradayParams {
                amount: 0.5,
                ..FaradayParams::default()
            },
            collider: FieldColliderParams {
                enabled: true,
                mode: FieldColliderMode::CollisionBoundary,
                boundary: MotionBoundaryMode::Mirror,
                input_a: MotionDonor::Selected {
                    layer_id: first,
                    saved_position: stale,
                },
                input_b: MotionDonor::Selected {
                    layer_id: second,
                    saved_position: stale,
                },
                ..FieldColliderParams::default()
            },
            ..MotionParams::default()
        };

        // Capture recomputes EACH slot's saved position independently against
        // the live stack. Both started from the same stale 9 and must land on
        // their own real positions, not on one shared value.
        let config = MotionConfig::from_params_for_capture(params, &[second, first]);
        assert_eq!(
            config.collider.input_a,
            MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(1).unwrap()
            }
        );
        assert_eq!(
            config.collider.input_b,
            MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(0).unwrap()
            }
        );

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("mode: collision_boundary"));
        assert!(yaml.contains("boundary: mirror"));
        assert!(yaml.contains("enabled: true"));
        // Authored topology only. A process-lifetime stable ID, a derived
        // vector, the transient pair, and gate parities are all absent.
        assert!(!yaml.contains("layer_id"));
        assert!(!yaml.contains("velocity"));
        assert!(!yaml.contains("derived"));
        assert!(!yaml.contains("pair"));
        assert!(!yaml.contains("gate"));

        let restored: MotionConfig = serde_yaml::from_str(&yaml).unwrap();
        let runtime = restored.to_params();
        // `to_params` never resolves: both slots collapse to Missing until a
        // complete live layer stack exists.
        assert_eq!(
            runtime.collider.input_a,
            MotionDonor::Missing {
                saved_position: SavedLayerPosition::new(1).unwrap()
            }
        );
        assert_eq!(
            runtime.collider.input_b,
            MotionDonor::Missing {
                saved_position: SavedLayerPosition::new(0).unwrap()
            }
        );
        assert!(!runtime.collider.is_admitted());

        let resolved = restored.resolve_runtime(&[second, first]);
        assert_eq!(
            resolved.collider.admission(),
            crate::motion::FieldColliderAdmission::Admitted {
                input_a: first,
                input_b: second,
            }
        );
        assert_eq!(resolved.collider.mode, FieldColliderMode::CollisionBoundary);
        assert_eq!(resolved.collider.boundary, MotionBoundaryMode::Mirror);

        // A hostile version is a hard deserialization error, never migrated.
        let hostile = yaml.replace("algorithm_version: 1", "algorithm_version: 2");
        assert!(serde_yaml::from_str::<MotionConfig>(&hostile).is_err());
        // And an unknown key inside the block is rejected outright.
        assert!(serde_yaml::from_str::<FieldColliderConfig>(
            "enabled: true\ninput_c:\n  kind: none\n"
        )
        .is_err());
    }

    #[test]
    fn a_collider_tombstone_survives_sanitization_and_never_rebinds() {
        let occupant = StableLayerId::new(88).unwrap();
        let position = SavedLayerPosition::new(0).unwrap();
        let config = MotionConfig {
            collider: FieldColliderConfig {
                enabled: true,
                input_a: MotionDonorConfig::Missing {
                    saved_position: position,
                },
                input_b: MotionDonorConfig::Selected {
                    saved_position: position,
                },
                ..FieldColliderConfig::default()
            },
            ..MotionConfig::default()
        };
        // Sanitizing must not flatten Missing into None: doing so would let the
        // dead slot rebind the next time that position was occupied.
        let sanitized = config.sanitized();
        assert_eq!(
            sanitized.collider.input_a,
            MotionDonorConfig::Missing {
                saved_position: position
            }
        );
        assert_eq!(
            sanitized.collider.input_b,
            MotionDonorConfig::Selected {
                saved_position: position
            }
        );

        let resolved = sanitized.resolve_runtime(&[occupant]);
        assert_eq!(
            resolved.collider.input_a,
            MotionDonor::Missing {
                saved_position: position
            },
            "a tombstone rebound onto whatever now occupies its position"
        );
        assert_eq!(
            resolved.collider.input_b,
            MotionDonor::Selected {
                layer_id: occupant,
                saved_position: position
            }
        );
    }

    #[test]
    fn a_look_carries_the_collider_recipe_and_preserves_live_input_topology() {
        let live_a = MotionDonor::Selected {
            layer_id: StableLayerId::new(5).unwrap(),
            saved_position: SavedLayerPosition::new(0).unwrap(),
        };
        let live_b = MotionDonor::Selected {
            layer_id: StableLayerId::new(6).unwrap(),
            saved_position: SavedLayerPosition::new(1).unwrap(),
        };
        let mut live = MotionParams {
            collider: FieldColliderParams {
                input_a: live_a,
                input_b: live_b,
                ..FieldColliderParams::default()
            },
            ..MotionParams::default()
        };
        apply_motion_look(
            MotionConfig {
                collider: FieldColliderConfig {
                    enabled: true,
                    mode: FieldColliderModeConfig::Curl,
                    boundary: MotionBoundaryModeConfig::Wrap,
                    input_a: MotionDonorConfig::Selected {
                        saved_position: SavedLayerPosition::new(4).unwrap(),
                    },
                    input_b: MotionDonorConfig::None,
                    ..FieldColliderConfig::default()
                },
                ..MotionConfig::default()
            },
            &mut live,
        );
        // Recipe travels...
        assert!(live.collider.enabled);
        assert_eq!(live.collider.mode, FieldColliderMode::Curl);
        assert_eq!(live.collider.boundary, MotionBoundaryMode::Wrap);
        // ...but both inputs are topology and stay exactly as authored live.
        assert_eq!(live.collider.input_a, live_a);
        assert_eq!(live.collider.input_b, live_b);
    }

    #[test]
    fn a_pre_collider_patch_loads_unchanged_and_a_default_block_is_omitted() {
        // An omitted section is exactly the pre-collider path.
        let legacy: MotionConfig =
            serde_yaml::from_str("algorithm_version: 1\ntransplant:\n  amount: 0.25\n").unwrap();
        assert_eq!(legacy.collider, FieldColliderConfig::default());
        assert!(legacy.collider.is_default());
        assert!(legacy.to_params().collider.is_exact_m4());

        // A default block emits no key at all, so a pre-collider patch keeps
        // its original bytes and its original canonical hash.
        let yaml = serde_yaml::to_string(&MotionConfig::default()).unwrap();
        assert!(!yaml.contains("collider"));

        // An enabled block does emit one.
        let enabled = MotionConfig {
            collider: FieldColliderConfig {
                enabled: true,
                ..FieldColliderConfig::default()
            },
            ..MotionConfig::default()
        };
        assert!(!enabled.collider.is_default());
        assert!(serde_yaml::to_string(&enabled)
            .unwrap()
            .contains("collider"));
    }

    #[test]
    fn motion_missing_tombstone_never_rebinds_and_look_preserves_live_donor() {
        let occupied = StableLayerId::new(77).unwrap();
        let saved_position = SavedLayerPosition::new(0).unwrap();
        let missing = MotionDonorConfig::Missing { saved_position };
        assert_eq!(
            missing.resolve_runtime(&[occupied]),
            MotionDonor::Missing { saved_position }
        );

        let selected_live = MotionDonor::Selected {
            layer_id: occupied,
            saved_position,
        };
        let mut live = MotionParams {
            transplant: FaradayParams {
                donor: selected_live,
                ..FaradayParams::default()
            },
            ..MotionParams::default()
        };
        apply_motion_look(
            MotionConfig {
                transplant: FaradayConfig {
                    amount: 0.9,
                    donor: MotionDonorConfig::Missing { saved_position },
                    ..FaradayConfig::default()
                },
                shutter: CurvedShutterConfig {
                    angle_degrees: 180.0,
                    ..CurvedShutterConfig::default()
                },
                ..MotionConfig::default()
            },
            &mut live,
        );
        assert_eq!(live.transplant.amount, 0.9);
        assert_eq!(live.shutter.angle_degrees, 180.0);
        assert_eq!(live.transplant.donor, selected_live);
    }

    #[test]
    fn old_patch_motion_is_exact_zero_and_default_capture_stays_omitted() {
        let old: PatchState = serde_yaml::from_str(
            "master: {}\nmaster_transform: {}\nlayers:\n  - filename: old.mov\n",
        )
        .unwrap();
        assert_eq!(old.master_motion, None);
        assert_eq!(old.layers[0].motion, None);

        let yaml = serde_yaml::to_string(&minimal_patch(1)).unwrap();
        assert!(!yaml.contains("master_motion:"));
        assert!(!yaml.contains("\n  motion:"));

        let hostile: MotionConfig = serde_yaml::from_str(
            "transplant:\n  amount: 8\n  confidence_threshold: .nan\n  confidence_softness: 9\n  refresh: -1\n  decay: 7\n  occlusion: 4\nshutter:\n  angle_degrees: 999\n  phase: -9\n  curvature: 7\n  chromatic_lag: 3\n",
        )
        .unwrap();
        let bounded = hostile.to_params();
        assert_eq!(bounded.transplant.amount, 1.0);
        assert_eq!(bounded.transplant.confidence_threshold, 0.1);
        assert_eq!(bounded.transplant.confidence_softness, 0.5);
        assert_eq!(bounded.transplant.refresh, 0.0);
        assert_eq!(bounded.transplant.decay, 1.0);
        assert_eq!(bounded.transplant.occlusion, 1.0);
        assert_eq!(bounded.shutter.angle_degrees, 360.0);
        assert_eq!(bounded.shutter.phase, -1.0);
        assert_eq!(bounded.shutter.curvature, 2.0);
        assert_eq!(bounded.shutter.chromatic_lag, 1.0);
    }

    #[test]
    fn score_driver_capture_recomputes_position_and_missing_never_rebinds() {
        let wanted = StableLayerId::new(91).unwrap();
        let other = StableLayerId::new(44).unwrap();
        let stale_position = SavedLayerPosition::new(7).unwrap();
        let actual_position = SavedLayerPosition::new(1).unwrap();
        let selected = CollisionScoreLoopDriverConfig::from_runtime_for_capture(
            CollisionScoreLoopDriver::SelectedLayer {
                layer_id: wanted,
                saved_position: stale_position,
            },
            &[other, wanted],
        );
        assert_eq!(
            selected,
            CollisionScoreLoopDriverConfig::SelectedLayer {
                saved_position: actual_position
            }
        );
        assert_eq!(
            selected.resolve_runtime(&[other, wanted]),
            CollisionScoreLoopDriver::SelectedLayer {
                layer_id: wanted,
                saved_position: actual_position,
            }
        );

        let absent = CollisionScoreLoopDriverConfig::from_runtime_for_capture(
            CollisionScoreLoopDriver::SelectedLayer {
                layer_id: wanted,
                saved_position: stale_position,
            },
            &[other],
        );
        assert_eq!(
            absent,
            CollisionScoreLoopDriverConfig::MissingSelectedLayer {
                saved_position: stale_position
            }
        );
        assert_eq!(
            absent.resolve_runtime(&[wanted]),
            CollisionScoreLoopDriver::MissingSelectedLayer {
                saved_position: stale_position
            },
            "a tombstone must remain inert even if its old position is occupied"
        );
    }

    #[test]
    fn patch_apply_tombstones_an_unresolved_selected_score_driver() {
        let saved_position = SavedLayerPosition::new(3).unwrap();
        let mut patch = minimal_patch(0);
        patch.temporal = Some(TemporalConfig {
            originals: Some(TemporalOriginalsConfig {
                score: CollisionScoreConfig {
                    enabled: true,
                    loop_driver: CollisionScoreLoopDriverConfig::SelectedLayer { saved_position },
                    ..CollisionScoreConfig::default()
                },
                ..TemporalOriginalsConfig::default()
            }),
            ..TemporalConfig::default()
        });

        let mut temporal = TemporalParams::default();
        let mut composition =
            RuntimeComposition::try_from_parts(Vec::new(), Vec::new(), Some(1), 0.5).unwrap();
        patch
            .apply_with_composition(
                &mut EffectUniforms::default(),
                &mut SpatialTransform::default(),
                &mut [],
                &mut RuntimeVisualRack::empty(),
                &mut [],
                &mut composition,
                &mut NtscParams::default(),
                &mut ModMatrix::new(),
                &mut temporal,
            )
            .unwrap();
        assert_eq!(
            temporal.originals.score.loop_driver,
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position }
        );

        patch
            .temporal
            .as_mut()
            .unwrap()
            .originals
            .as_mut()
            .unwrap()
            .score
            .loop_driver = CollisionScoreLoopDriverConfig::MissingSelectedLayer { saved_position };
        temporal.originals.score.loop_driver = CollisionScoreLoopDriver::None;
        patch.apply(
            &mut EffectUniforms::default(),
            &mut SpatialTransform::default(),
            &mut [],
            &mut NtscParams::default(),
            &mut ModMatrix::new(),
            &mut temporal,
        );
        assert_eq!(
            temporal.originals.score.loop_driver,
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position },
            "an authored tombstone must remain inert"
        );
    }

    /// A patch with modulation state survives a YAML round-trip, and old
    /// patches without a `modulation:` section still parse.
    #[test]
    fn mod_config_yaml_round_trip() {
        let mut matrix = ModMatrix::new();
        matrix.clock.set_bpm(140.0);
        matrix.lfos[1].shape = LfoShape::Saw;
        matrix.lfos[1].beats = 2.0;
        matrix.lfos[2].phase = 0.25;
        matrix.audio_enabled = true;
        matrix.audio_gain = 1.5;
        matrix.audio_source_kind = crate::modulation::AUDIO_SOURCE_FILE.to_string();
        matrix.audio_clip_path = "pulse-loop.wav".to_string();
        matrix.audio_band_config = crate::audio::AudioBandConfig::new(
            6,
            &[120.0, 480.0, 1500.0, 4000.0, 9000.0],
            16_000.0,
        );
        matrix.midi_enabled = true;
        matrix.midi_ccs = [21, 22, 23, 24];
        matrix.midi_clock_sync = true;
        matrix.set_gyro_degrees(355.0, 12.0, -18.0);
        matrix.gyro_config[0] = GyroAxisConfig {
            center_degrees: 350.0,
            range_degrees: 30.0,
            expo: 0.75,
            invert: true,
        };
        matrix.gyro_config[1].center_degrees = 5.0;
        matrix.recompute_gyro();
        matrix.set_pad(0.8, 0.2, false);
        matrix.pad_config.axes[0] = PadAxisConfig {
            curve: Curve::Exp,
            curve_amount: 0.75,
            quantize: 8,
        };
        matrix.pad_config.axes[1] = PadAxisConfig {
            curve: Curve::Steps,
            curve_amount: -0.5,
            quantize: 16,
        };
        matrix.pad_config.spring_enabled = true;
        matrix.pad_config.spring_rate = 7.0;
        let mut expressive = Routing::new(ModSource::AudioBright, "layer2_opacity", 0.6);
        expressive.curve = Curve::SCurve;
        expressive.curve_amount = 0.5;
        expressive.attack = 0.08;
        expressive.release = 0.4;
        matrix.routings.push(expressive);
        matrix
            .routings
            .push(Routing::new(ModSource::AudioBass, "ntsc_snow", -0.8));
        matrix
            .routings
            .push(Routing::new(ModSource::Midi(2), "vignette", 1.0));
        matrix
            .routings
            .push(Routing::new(ModSource::Lfo(3), "rgb_split", 0.5));
        matrix
            .routings
            .push(Routing::new(ModSource::AudioBand(5), "contrast", 0.25));

        let temporal = TemporalParams {
            feedback: 0.7,
            fb_zoom: 1.02,
            fb_rotate: -1.5,
            slitscan: 0.4,
            slit_angle: 37.0,
            slit_axis: 1.0,
            key_mode: 3.0,
            key_threshold: 0.22,
            key_softness: 0.05,
            key_history: 4.0,
            originals: Default::default(),
            rig: Default::default(),
            ..TemporalParams::default()
        };

        let patch = PatchState::capture(
            PatchMasterVisual::new(&EffectUniforms::default(), &SpatialTransform::default()),
            &[],
            &NtscParams::default(),
            &matrix,
            &temporal,
            PatchTransportState {
                master_paused: true,
                media_frozen: true,
            },
            &crate::morph::Morph::default(),
        );
        let yaml = serde_yaml::to_string(&patch).unwrap();
        let parsed: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert!(parsed.master_paused);
        assert!(parsed.media_frozen);

        let mut restored = ModMatrix::new();
        let mut restored_temporal = TemporalParams::default();
        parsed.apply(
            &mut EffectUniforms::default(),
            &mut SpatialTransform::default(),
            &mut [],
            &mut NtscParams::default(),
            &mut restored,
            &mut restored_temporal,
        );

        assert_eq!(restored_temporal.feedback, 0.7);
        assert_eq!(restored_temporal.fb_zoom, 1.02);
        assert_eq!(restored_temporal.fb_rotate, -1.5);
        assert_eq!(restored_temporal.slitscan, 0.4);
        assert_eq!(restored_temporal.slit_angle, 37.0);
        assert_eq!(restored_temporal.slit_axis, 1.0);
        assert_eq!(restored_temporal.key_mode, 3.0);
        assert_eq!(restored_temporal.key_threshold, 0.22);
        assert_eq!(restored_temporal.key_softness, 0.05);
        assert_eq!(restored_temporal.key_history, 4.0);

        assert_eq!(restored.clock.bpm, 140.0);
        assert_eq!(restored.lfos[1].shape, LfoShape::Saw);
        assert_eq!(restored.lfos[1].beats, 2.0);
        assert_eq!(restored.lfos[2].phase, 0.25);
        assert!(restored.audio_enabled);
        assert_eq!(restored.audio_gain, 1.5);
        assert_eq!(
            restored.audio_source_kind,
            crate::modulation::AUDIO_SOURCE_FILE
        );
        assert_eq!(restored.audio_clip_path, "pulse-loop.wav");
        assert_eq!(restored.audio_band_config.count(), 6);
        assert_eq!(
            restored.audio_band_config.crossovers(),
            &[120.0, 480.0, 1500.0, 4000.0, 9000.0]
        );
        assert_eq!(restored.audio_band_config.ceiling_hz(), 16_000.0);
        assert!(restored.midi_enabled);
        assert_eq!(restored.midi_ccs, [21, 22, 23, 24]);
        assert!(restored.midi_clock_sync);
        assert_eq!(restored.gyro_raw, [355.0, 12.0, -18.0]);
        assert_eq!(restored.gyro_config[0].center_degrees, 350.0);
        assert_eq!(restored.gyro_config[0].range_degrees, 30.0);
        assert_eq!(restored.gyro_config[0].expo, 0.75);
        assert!(restored.gyro_config[0].invert);
        assert_eq!(restored.gyro_config[1].center_degrees, 5.0);
        assert_eq!(restored.pad, [0.8, 0.2]);
        assert!(!restored.pad_active);
        assert_eq!(restored.pad_config.axes[0].curve, Curve::Exp);
        assert_eq!(restored.pad_config.axes[0].curve_amount, 0.75);
        assert_eq!(restored.pad_config.axes[0].quantize, 8);
        assert_eq!(restored.pad_config.axes[1].curve, Curve::Steps);
        assert_eq!(restored.pad_config.axes[1].quantize, 16);
        assert!(restored.pad_config.spring_enabled);
        assert_eq!(restored.pad_config.spring_rate, 7.0);
        assert_eq!(restored.routings.len(), 5);
        assert_eq!(restored.routings[0].source, ModSource::AudioBright);
        assert_eq!(restored.routings[0].target(), "layer2_opacity");
        assert_eq!(restored.routings[0].curve, Curve::SCurve);
        assert_eq!(restored.routings[0].curve_amount, 0.5);
        assert_eq!(restored.routings[0].attack, 0.08);
        assert_eq!(restored.routings[0].release, 0.4);
        assert_eq!(restored.routings[1].source, ModSource::AudioBass);
        assert_eq!(restored.routings[1].target(), "ntsc_snow");
        assert_eq!(restored.routings[1].depth, -0.8);
        assert_eq!(restored.routings[2].source, ModSource::Midi(2));
        assert_eq!(restored.routings[3].source, ModSource::Lfo(3));
        assert_eq!(restored.routings[4].source, ModSource::AudioBand(5));

        // Layer modulation math: bright 0.5 at depth 0.6 on layer 2 opacity.
        restored.audio.bright = 0.5;
        restored.update_at_beat(0.0, 1.0);
        let lm = restored.modulate_layer_full(
            1,
            &crate::effects::EffectUniforms::default(),
            &SpatialTransform::default(),
            0.4,
            1.0,
            30.0,
        );
        let expected = 0.4 + crate::modulation::shape(0.5, Curve::SCurve, 0.5) * 0.6 * 0.5;
        assert!((lm.opacity - expected).abs() < 1e-4);
        assert_eq!(lm.speed, 1.0, "untargeted values pass through");

        // Legacy patch without modulation section parses and applies cleanly.
        let legacy = "master:\n  pixelate: 4.0\nlayers: []\n";
        let parsed: PatchState = serde_yaml::from_str(legacy).unwrap();
        assert!(!parsed.master_paused);
        assert!(parsed.modulation.is_none());
        assert!(parsed.temporal.is_none());
        let mut untouched = ModMatrix::new();
        untouched.clock.set_bpm(99.0);
        let mut untouched_temporal = TemporalParams {
            feedback: 0.3,
            ..Default::default()
        };
        parsed.apply(
            &mut EffectUniforms::default(),
            &mut SpatialTransform::default(),
            &mut [],
            &mut NtscParams::default(),
            &mut untouched,
            &mut untouched_temporal,
        );
        assert_eq!(
            untouched.clock.bpm, 99.0,
            "absent section must not reset matrix"
        );
        assert_eq!(
            untouched_temporal.feedback, 0.3,
            "absent section must not reset temporal"
        );

        // Legacy modulation sections had no current motion samples. They
        // restore gyro axes at their saved centers and the pad at center.
        let legacy_mod = r#"
master: {}
layers: []
modulation:
  gyro:
    - { center_degrees: 90.0, range_degrees: 20.0 }
    - { center_degrees: 10.0, range_degrees: 30.0 }
    - { center_degrees: -5.0, range_degrees: 40.0 }
"#;
        let parsed: PatchState = serde_yaml::from_str(legacy_mod).unwrap();
        let mut legacy_matrix = ModMatrix::new();
        parsed
            .modulation
            .unwrap()
            .apply_to_matrix(&mut legacy_matrix);
        assert_eq!(legacy_matrix.gyro, [0.5; 3]);
        assert_eq!(legacy_matrix.pad, [0.5; 2]);
        assert!(!legacy_matrix.pad_active);

        // The historical three-entry list encoded two crossovers plus the
        // analysis ceiling. It must load exactly as it did before the band
        // count became configurable.
        let legacy_audio = r#"
master: {}
layers: []
modulation:
  audio_band_edges: [250.0, 2000.0, 8000.0]
  routings:
    - { source: audio_bass, target: brightness, depth: 0.5 }
"#;
        let parsed: PatchState = serde_yaml::from_str(legacy_audio).unwrap();
        let mut legacy_audio_matrix = ModMatrix::new();
        parsed
            .modulation
            .unwrap()
            .apply_to_matrix(&mut legacy_audio_matrix);
        assert_eq!(legacy_audio_matrix.audio_band_config.count(), 3);
        assert_eq!(
            legacy_audio_matrix.audio_band_config.crossovers(),
            &[250.0, 2000.0]
        );
        assert_eq!(legacy_audio_matrix.audio_band_config.ceiling_hz(), 8000.0);
        assert_eq!(legacy_audio_matrix.routings[0].source, ModSource::AudioBass);

        let alias_and_canonical: ModConfig = serde_yaml::from_str(
            r#"
bpm: 120
routings:
  - { source: lfo0, target: layer1_key, depth: 0.9 }
  - { source: lfo1, target: layer1_key_threshold, depth: 0.4 }
"#,
        )
        .unwrap();
        let mut normalized = ModMatrix::new();
        alias_and_canonical.apply_to_matrix(&mut normalized);
        assert_eq!(normalized.routings.len(), 1);
        assert_eq!(normalized.routings[0].target(), "layer1_key_threshold");
        assert_eq!(normalized.routings[0].source, ModSource::Lfo(1));

        // Canonical spellings beyond the accepted route window must not
        // suppress a legacy alias that is itself inside that window.
        let mut capped: ModConfig = serde_yaml::from_str("bpm: 120\n").unwrap();
        capped.routings.push(RoutingConfig {
            source: "lfo0".to_string(),
            target: "layer1_key".to_string(),
            stable_target: None,
            depth: 0.5,
            curve: "linear".to_string(),
            curve_amount: 0.0,
            attack: 30.0,
            release: 30.0,
        });
        for _ in 1..MAX_ROUTINGS {
            capped.routings.push(RoutingConfig {
                source: "lfo1".to_string(),
                target: "brightness".to_string(),
                stable_target: None,
                depth: 0.25,
                curve: "linear".to_string(),
                curve_amount: 0.0,
                attack: 0.0,
                release: 0.0,
            });
        }
        capped.routings.push(RoutingConfig {
            source: "lfo2".to_string(),
            target: "layer1_key_threshold".to_string(),
            stable_target: None,
            depth: 0.75,
            curve: "linear".to_string(),
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
        });
        let mut capped_matrix = ModMatrix::new();
        capped.apply_to_matrix(&mut capped_matrix);
        assert_eq!(capped_matrix.routings.len(), MAX_ROUTINGS);
        assert_eq!(capped_matrix.routings[0].target(), "layer1_key_threshold");
        assert_eq!(capped_matrix.routings[0].source, ModSource::Lfo(0));
        assert_eq!(capped_matrix.routings[0].attack, 10.0);
        assert_eq!(capped_matrix.routings[0].release, 10.0);
        assert_eq!(
            legacy_audio_matrix.audio_source_kind,
            crate::modulation::AUDIO_SOURCE_LIVE
        );
        assert!(legacy_audio_matrix.audio_clip_path.is_empty());
    }

    #[test]
    fn modulation_capture_prefers_retained_content_identity_without_changing_legacy_paths() {
        let mut matrix = ModMatrix::new();
        matrix.audio_clip_path = r"D:\resolved\analysis.wav".to_string();
        let reference = format!("cos-sha256://{}/4096", "b".repeat(64));
        matrix.audio_clip_source_reference = Some(reference.clone());
        assert_eq!(ModConfig::from_matrix(&matrix).audio_clip_path, reference);

        matrix.audio_clip_source_reference = None;
        assert_eq!(
            ModConfig::from_matrix(&matrix).audio_clip_path,
            r"D:\resolved\analysis.wav"
        );
    }

    fn empty_runtime_group_with_matte(
        id: GroupId,
        matte: Option<RuntimeImageMatte>,
    ) -> RuntimeGroup {
        RuntimeGroup {
            id,
            name: crate::composition::GroupName::new(format!("group-{}", id.get())).unwrap(),
            members: RuntimeGroupMembers::default(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte,
            solo: false,
            bypass: false,
            bus: crate::composition::BusAssignment::Program,
        }
    }

    fn default_runtime_group_matte() -> RuntimeImageMatte {
        RuntimeImageMatte::resolve_routes(
            crate::visual_rack::ImageMatte::default(),
            &mut |_| None,
            &|_| false,
        )
    }

    #[test]
    fn matte_and_bus_modulation_targets_roundtrip_typed_and_never_retarget() {
        use crate::modulation::{CompositionModParameter, GroupModParameter, SavedMissingTarget};

        let target_id = GroupId::new(9).unwrap();
        let other_id = GroupId::new(7).unwrap();
        let target_group =
            empty_runtime_group_with_matte(target_id, Some(default_runtime_group_matte()));
        let other_group =
            empty_runtime_group_with_matte(other_id, Some(default_runtime_group_matte()));
        let composition = RuntimeComposition::try_from_parts(
            vec![target_group.clone(), other_group.clone()],
            vec![
                RuntimeRootItem::Group {
                    group_id: target_id,
                },
                RuntimeRootItem::Group { group_id: other_id },
            ],
            Some(10),
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::empty();
        let book = StableModAddressBook::from_composition(&master, &[], &composition).unwrap();
        let mut matrix = ModMatrix::new();
        matrix.routings = [
            "group/9/matte.amount",
            "group/9/matte.threshold",
            "group/9/matte.softness",
            "composition/bus_crossfade",
        ]
        .into_iter()
        .enumerate()
        .map(|(index, target)| Routing::new(ModSource::Lfo(index), target, 0.5))
        .collect();

        let captured =
            ModConfig::from_matrix_with_composition(&matrix, &book, &[], &composition).unwrap();
        assert_eq!(
            captured
                .routings
                .iter()
                .map(|routing| routing.stable_target)
                .collect::<Vec<_>>(),
            vec![
                Some(SavedStableModTarget::GroupValue {
                    group_id: target_id,
                    parameter: GroupModParameter::MatteAmount,
                }),
                Some(SavedStableModTarget::GroupValue {
                    group_id: target_id,
                    parameter: GroupModParameter::MatteThreshold,
                }),
                Some(SavedStableModTarget::GroupValue {
                    group_id: target_id,
                    parameter: GroupModParameter::MatteSoftness,
                }),
                Some(SavedStableModTarget::CompositionValue {
                    parameter: CompositionModParameter::BusCrossfade,
                }),
            ]
        );
        let yaml = serde_yaml::to_string(&captured).unwrap();
        assert!(yaml.contains("parameter: matte.amount"));
        assert!(yaml.contains("parameter: bus_crossfade"));
        let restored_config: ModConfig = serde_yaml::from_str(&yaml).unwrap();

        // A changed internal group order cannot change the live GroupId target.
        let reordered = RuntimeComposition::try_from_parts(
            vec![other_group, target_group],
            vec![
                RuntimeRootItem::Group {
                    group_id: target_id,
                },
                RuntimeRootItem::Group { group_id: other_id },
            ],
            Some(10),
            0.5,
        )
        .unwrap();
        let reordered_book =
            StableModAddressBook::from_composition(&master, &[], &reordered).unwrap();
        let mut restored_matrix = ModMatrix::new();
        restored_config.apply_to_matrix_with_composition(
            &mut restored_matrix,
            &reordered_book,
            &[],
            &reordered,
        );
        assert_eq!(
            restored_matrix.routings[0].stable_target(),
            crate::modulation::StableModTarget::parse("group/9/matte.amount")
        );
        assert_eq!(
            restored_matrix.routings[3].stable_target(),
            crate::modulation::StableModTarget::parse("composition/bus_crossfade")
        );

        // Removing the matte makes its value address explicitly missing, but
        // the independent composition target stays live. It cannot attach to
        // the other group merely because that group occupies another order.
        let missing_matte = RuntimeComposition::try_from_parts(
            vec![empty_runtime_group_with_matte(target_id, None)],
            vec![RuntimeRootItem::Group {
                group_id: target_id,
            }],
            Some(10),
            0.5,
        )
        .unwrap();
        let missing_book =
            StableModAddressBook::from_composition(&master, &[], &missing_matte).unwrap();
        let mut missing_matrix = ModMatrix::new();
        restored_config.apply_to_matrix_with_composition(
            &mut missing_matrix,
            &missing_book,
            &[],
            &missing_matte,
        );
        assert_eq!(
            missing_matrix.routings[0].saved_missing_target(),
            Some(SavedStableModTarget::MissingGroup {
                group_id: target_id,
                missing_target: SavedMissingTarget::GroupValue {
                    parameter: GroupModParameter::MatteAmount,
                },
            })
        );
        assert_eq!(
            missing_matrix.routings[3].stable_target(),
            crate::modulation::StableModTarget::parse("composition/bus_crossfade")
        );
    }

    fn minimal_patch(layer_count: usize) -> PatchState {
        PatchState {
            master: EffectsConfig::default(),
            master_transform: SpatialTransform::default(),
            master_motion: None,
            layers: (0..layer_count)
                .map(|index| {
                    saved_layer(
                        &format!("layer-{index}.mov"),
                        1.0,
                        "normal",
                        true,
                        false,
                        0.0,
                        index as u32,
                    )
                })
                .collect(),
            master_rack: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: false,
            media_frozen: false,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
            scenes: Scenes::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        }
    }

    fn image_rack(source: SavedImageSource, timing: EdgeTiming) -> VisualRack {
        let mut rack = VisualRack::empty();
        rack.push(VisualNodeKind::Mask(MaskParams::Image(
            crate::visual_rack::ImageMatte {
                tap: SavedImageTap { source, timing },
                ..crate::visual_rack::ImageMatte::default()
            },
        )))
        .unwrap();
        rack
    }

    fn saved_group(
        id: GroupId,
        members: Vec<SavedLayerPosition>,
        rack: VisualRack,
    ) -> crate::composition::Group {
        crate::composition::Group {
            id,
            name: crate::composition::GroupName::new(format!("group-{}", id.get())).unwrap(),
            members: crate::composition::GroupMembers::try_from_vec(members).unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack,
            matte: None,
            solo: false,
            bypass: false,
            bus: crate::composition::BusAssignment::Program,
        }
    }

    #[test]
    fn visual_schema_distinguishes_legacy_omission_explicit_empty_and_future_versions() {
        let legacy: PatchState = serde_yaml::from_str("master: {}\nlayers: []\n").unwrap();
        assert_eq!(legacy.visual_schema_version, 0);
        assert!(legacy.master_rack.is_none());
        assert!(legacy.composition.is_none());
        assert!(legacy
            .effective_master_rack()
            .is_exact_legacy(LegacyRackScope::Master));
        let legacy_yaml = serde_yaml::to_string(&legacy).unwrap();
        assert!(!legacy_yaml.contains("visual_schema_version"));
        assert!(!legacy_yaml.contains("master_rack"));

        let mut explicit = minimal_patch(0);
        explicit.master_rack = Some(VisualRack::empty());
        explicit.validate_creative_persistence().unwrap();
        assert_eq!(explicit.visual_schema_version, 1);
        let explicit_yaml = serde_yaml::to_string(&explicit).unwrap();
        assert!(explicit_yaml.contains("visual_schema_version: 1"));
        let restored: PatchState = serde_yaml::from_str(&explicit_yaml).unwrap();
        assert!(restored.master_rack.as_ref().unwrap().is_empty());
        assert!(!restored
            .effective_master_rack()
            .is_exact_legacy(LegacyRackScope::Master));

        assert!(serde_yaml::from_str::<PatchState>(
            "master: {}\nlayers: []\nvisual_schema_version: 2\n"
        )
        .err()
        .unwrap()
        .to_string()
        .contains("unsupported visual_schema_version"));
    }

    #[test]
    fn omitted_composition_synthesizes_legacy_back_to_front_order() {
        let positions = (0..3)
            .map(saved_position_at)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let flattened = legacy_composition_for_positions(&positions)
            .unwrap()
            .flatten()
            .unwrap();
        assert_eq!(
            flattened
                .layers
                .iter()
                .map(|layer| layer.layer.get())
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
    }

    #[test]
    fn exact_legacy_over_256_skips_only_the_advanced_graph_cap() {
        const LAYERS: usize = 300;

        let mut dormant = minimal_patch(LAYERS);
        dormant.layers[0].matte = LayerMatteConfig {
            enabled: true,
            amount: 0.0,
            ..LayerMatteConfig::default()
        };
        dormant.validate_creative_persistence().unwrap();

        // This is the explicit saved shape produced by M2 patch capture. It
        // must round-trip just like omission and remain recognizably exact.
        let mut explicit = minimal_patch(LAYERS);
        explicit.master_rack = Some(VisualRack::synthetic_legacy(LegacyRackScope::Master));
        for layer in &mut explicit.layers {
            layer.rack = Some(VisualRack::synthetic_legacy(LegacyRackScope::Layer));
        }
        let positions = (0..LAYERS)
            .map(saved_position_at)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        explicit.composition = Some(legacy_composition_for_positions(&positions).unwrap());
        explicit.validate_creative_persistence().unwrap();
        let yaml = serde_yaml::to_string(&explicit).unwrap();
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.is_global_legacy_exact());
        assert_eq!(restored.layers.len(), LAYERS);

        let graph_cap_error = |mut patch: PatchState| {
            patch
                .validate_creative_persistence()
                .expect_err("advanced graph must retain its bounded scope cap")
        };

        let mut active_matte = minimal_patch(LAYERS);
        active_matte.layers[0].matte = LayerMatteConfig {
            enabled: true,
            amount: 0.25,
            ..LayerMatteConfig::default()
        };
        assert!(graph_cap_error(active_matte).contains("image graph has"));

        let mut custom_rack = minimal_patch(LAYERS);
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Shift(Default::default()))
            .unwrap();
        custom_rack.layers[0].rack = Some(rack);
        assert!(graph_cap_error(custom_rack).contains("image graph has"));

        let mut grouped = minimal_patch(LAYERS);
        let group_id = GroupId::new(1).unwrap();
        let group = saved_group(group_id, vec![positions[0]], VisualRack::empty());
        let group_root = positions
            .iter()
            .rev()
            .copied()
            .map(|layer| {
                if layer == positions[0] {
                    RootItem::Group { group_id }
                } else {
                    RootItem::Layer {
                        layer,
                        bus: crate::composition::BusAssignment::Program,
                    }
                }
            })
            .collect();
        grouped.composition =
            Some(CompositionTree::try_from_parts(vec![group], group_root, Some(2), 0.5).unwrap());
        assert!(graph_cap_error(grouped).contains("image graph has"));

        let mut non_program = minimal_patch(LAYERS);
        let root = positions
            .iter()
            .rev()
            .copied()
            .enumerate()
            .map(|(index, layer)| RootItem::Layer {
                layer,
                bus: if index == 0 {
                    crate::composition::BusAssignment::A
                } else {
                    crate::composition::BusAssignment::Program
                },
            })
            .collect();
        non_program.composition =
            Some(CompositionTree::try_from_parts(Vec::new(), root, Some(1), 0.5).unwrap());
        assert!(graph_cap_error(non_program).contains("image graph has"));
    }

    #[test]
    fn saved_visual_graph_rejects_current_cycles_but_allows_prelocal_and_previous_frame() {
        let pos0 = SavedLayerPosition::new(0).unwrap();
        let pos1 = SavedLayerPosition::new(1).unwrap();
        let mut cyclic = minimal_patch(2);
        cyclic.layers[0].rack = Some(image_rack(
            SavedImageSource::SelectedLayer {
                layer_position: pos1,
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        ));
        cyclic.layers[1].rack = Some(image_rack(
            SavedImageSource::SelectedLayer {
                layer_position: pos0,
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        ));
        let live = minimal_patch(2);
        let live_before = serde_yaml::to_string(&live).unwrap();
        let hostile_yaml = serde_yaml::to_string(&cyclic).unwrap();
        assert!(serde_yaml::from_str::<PatchState>(&hostile_yaml)
            .err()
            .unwrap()
            .to_string()
            .contains("cycle"));
        assert_eq!(serde_yaml::to_string(&live).unwrap(), live_before);

        let mut prelocal = minimal_patch(1);
        prelocal.layers[0].rack = Some(image_rack(
            SavedImageSource::SelectedLayer {
                layer_position: pos0,
                stage: LayerImageStage::PreLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        ));
        prelocal.validate_creative_persistence().unwrap();

        let mut previous = minimal_patch(1);
        previous.layers[0].rack = Some(image_rack(
            SavedImageSource::SelectedLayer {
                layer_position: pos0,
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::PreviousFrame,
        ));
        previous.validate_creative_persistence().unwrap();
    }

    #[test]
    fn group_structural_edges_reject_own_output_and_allow_a_preceding_sibling() {
        let pos0 = SavedLayerPosition::new(0).unwrap();
        let pos1 = SavedLayerPosition::new(1).unwrap();
        let group1 = GroupId::new(1).unwrap();
        let group2 = GroupId::new(2).unwrap();

        let mut own_output = minimal_patch(1);
        own_output.layers[0].rack = Some(image_rack(
            SavedImageSource::GroupOutput { group_id: group1 },
            EdgeTiming::CurrentFrame,
        ));
        own_output.composition = Some(
            CompositionTree::try_from_parts(
                vec![saved_group(group1, vec![pos0], VisualRack::empty())],
                vec![RootItem::Group { group_id: group1 }],
                Some(2),
                0.5,
            )
            .unwrap(),
        );
        assert!(own_output
            .validate_creative_persistence()
            .unwrap_err()
            .contains("cycle"));

        let mut self_group = minimal_patch(1);
        self_group.composition = Some(
            CompositionTree::try_from_parts(
                vec![saved_group(
                    group1,
                    vec![pos0],
                    image_rack(
                        SavedImageSource::GroupOutput { group_id: group1 },
                        EdgeTiming::CurrentFrame,
                    ),
                )],
                vec![RootItem::Group { group_id: group1 }],
                Some(2),
                0.5,
            )
            .unwrap(),
        );
        assert!(self_group
            .validate_creative_persistence()
            .unwrap_err()
            .contains("cycle"));

        let mut sibling = minimal_patch(2);
        sibling.composition = Some(
            CompositionTree::try_from_parts(
                vec![
                    saved_group(group1, vec![pos0], VisualRack::empty()),
                    saved_group(
                        group2,
                        vec![pos1],
                        image_rack(
                            SavedImageSource::GroupOutput { group_id: group1 },
                            EdgeTiming::CurrentFrame,
                        ),
                    ),
                ],
                vec![
                    RootItem::Group { group_id: group1 },
                    RootItem::Group { group_id: group2 },
                ],
                Some(3),
                0.5,
            )
            .unwrap(),
        );
        sibling.validate_creative_persistence().unwrap();
    }

    #[test]
    fn dormant_image_edges_round_trip_under_the_live_planner_active_edge_law() {
        let pos0 = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let self_source = SavedImageSource::GroupOutput { group_id };
        let round_trip = |mut patch: PatchState| {
            patch.validate_creative_persistence().unwrap();
            let yaml = serde_yaml::to_string(&patch).unwrap();
            serde_yaml::from_str::<PatchState>(&yaml).unwrap()
        };

        let mut disabled = image_rack(self_source, EdgeTiming::CurrentFrame);
        let disabled_id = disabled.iter().next().unwrap().stable_id;
        disabled.get_mut(disabled_id).unwrap().enabled = false;
        let mut zero_wet = image_rack(self_source, EdgeTiming::CurrentFrame);
        let zero_wet_id = zero_wet.iter().next().unwrap().stable_id;
        zero_wet.get_mut(zero_wet_id).unwrap().wet = 0.0;
        let mut zero_mask_amount = image_rack(self_source, EdgeTiming::CurrentFrame);
        let zero_amount_node = zero_mask_amount.iter().next().unwrap().stable_id;
        let VisualNodeKind::Mask(MaskParams::Image(matte)) =
            &mut zero_mask_amount.get_mut(zero_amount_node).unwrap().kind
        else {
            panic!("fixture must remain an image mask");
        };
        matte.amount = 0.0;

        for dormant_rack in [disabled, zero_wet, zero_mask_amount] {
            let mut patch = minimal_patch(1);
            patch.composition = Some(
                CompositionTree::try_from_parts(
                    vec![saved_group(group_id, vec![pos0], dormant_rack)],
                    vec![RootItem::Group { group_id }],
                    Some(2),
                    0.5,
                )
                .unwrap(),
            );
            let _restored = round_trip(patch);
        }

        let self_matte = |amount| crate::visual_rack::ImageMatte {
            tap: SavedImageTap {
                source: self_source,
                timing: EdgeTiming::CurrentFrame,
            },
            amount,
            ..crate::visual_rack::ImageMatte::default()
        };

        let mut zero_group_matte = saved_group(group_id, vec![pos0], VisualRack::empty());
        zero_group_matte.matte = Some(self_matte(0.0));
        let mut patch = minimal_patch(1);
        patch.composition = Some(
            CompositionTree::try_from_parts(
                vec![zero_group_matte],
                vec![RootItem::Group { group_id }],
                Some(2),
                0.5,
            )
            .unwrap(),
        );
        let _restored = round_trip(patch);

        let mut bypassed = saved_group(
            group_id,
            vec![pos0],
            image_rack(self_source, EdgeTiming::CurrentFrame),
        );
        bypassed.matte = Some(self_matte(1.0));
        bypassed.bypass = true;
        let mut patch = minimal_patch(1);
        patch.composition = Some(
            CompositionTree::try_from_parts(
                vec![bypassed],
                vec![RootItem::Group { group_id }],
                Some(2),
                0.5,
            )
            .unwrap(),
        );
        let restored = round_trip(patch);
        assert!(
            restored
                .composition
                .unwrap()
                .group(group_id)
                .unwrap()
                .bypass
        );

        let mut zero_layer_matte = minimal_patch(1);
        zero_layer_matte.layers[0].matte = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::SelectedLayer {
                layer_position: pos0,
                stage: LayerImageStage::PostLocalEffects,
            },
            amount: 0.0,
            ..LayerMatteConfig::default()
        };
        let restored = round_trip(zero_layer_matte);
        assert!(restored.layers[0].matte.enabled);
        assert_eq!(restored.layers[0].matte.amount, 0.0);
    }

    #[test]
    fn saved_displace_edges_are_dormant_at_zero_gain_and_cycle_when_woken() {
        use crate::visual_rack::{DisplaceBoundary, DisplaceParams};

        let pos0 = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let self_source = SavedImageSource::GroupOutput { group_id };
        let displace_rack = |params: DisplaceParams| {
            let mut rack = VisualRack::empty();
            rack.push(VisualNodeKind::Displace(params)).unwrap();
            rack
        };
        let self_route = DisplaceParams {
            tap: SavedImageTap {
                source: self_source,
                timing: EdgeTiming::CurrentFrame,
            },
            boundary: DisplaceBoundary::Wrap,
            ..DisplaceParams::default()
        };
        let patch_with = |rack: VisualRack| {
            let mut patch = minimal_patch(1);
            patch.composition = Some(
                CompositionTree::try_from_parts(
                    vec![saved_group(group_id, vec![pos0], rack)],
                    vec![RootItem::Group { group_id }],
                    Some(2),
                    0.5,
                )
                .unwrap(),
            );
            patch
        };

        // Dormant forms claim no saved edge, so a self route round trips.
        let mut disabled = displace_rack(DisplaceParams {
            amount_x: 0.5,
            ..self_route
        });
        let disabled_id = disabled.iter().next().unwrap().stable_id;
        disabled.get_mut(disabled_id).unwrap().enabled = false;
        let mut zero_wet = displace_rack(DisplaceParams {
            amount_x: 0.5,
            ..self_route
        });
        let zero_wet_id = zero_wet.iter().next().unwrap().stable_id;
        zero_wet.get_mut(zero_wet_id).unwrap().wet = 0.0;
        let zero_gain = displace_rack(self_route);

        for dormant in [disabled, zero_wet, zero_gain] {
            let mut patch = patch_with(dormant);
            patch.validate_creative_persistence().unwrap();
            let yaml = serde_yaml::to_string(&patch).unwrap();
            let restored = serde_yaml::from_str::<PatchState>(&yaml).unwrap();
            let composition = restored.composition.unwrap();
            let rack = &composition.group(group_id).unwrap().rack;
            assert!(matches!(
                rack.iter().next().unwrap().kind,
                VisualNodeKind::Displace(_)
            ));
        }

        // Waking either gain wakes the saved edge, and the same-frame self
        // route is then rejected at load rather than reaching the planner.
        for params in [
            DisplaceParams {
                amount_x: 0.25,
                ..self_route
            },
            DisplaceParams {
                amount_y: -0.25,
                ..self_route
            },
        ] {
            let error = patch_with(displace_rack(params))
                .validate_creative_persistence()
                .expect_err("a woken same-frame Displace self route must be rejected");
            assert!(
                error.contains("cycle") || error.contains("graph"),
                "unexpected saved-edge rejection: {error}"
            );
        }

        // The identical route at N-1 is a legitimate saved feedback edge.
        let previous_frame = displace_rack(DisplaceParams {
            tap: SavedImageTap {
                source: self_source,
                timing: EdgeTiming::PreviousFrame,
            },
            amount_x: 0.25,
            ..self_route
        });
        patch_with(previous_frame)
            .validate_creative_persistence()
            .unwrap();
    }

    /// The saved graph has no scope that produces the gesture canvas, so a
    /// route to it claims no dependency edge whether it is dormant or woken.
    /// That is what lets a group tap the field at the current frame from inside
    /// itself — the case that would be an immediate cycle for every other
    /// producer in the vocabulary.
    #[test]
    fn a_saved_gesture_canvas_route_claims_no_edge_dormant_or_woken() {
        use crate::visual_rack::{DisplaceBoundary, DisplaceParams};

        let pos0 = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let canvas_route = |timing, amount_x| DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::GestureCanvas,
                timing,
            },
            amount_x,
            boundary: DisplaceBoundary::Wrap,
            ..DisplaceParams::default()
        };
        let patch_with = |params: DisplaceParams| {
            let mut rack = VisualRack::empty();
            rack.push(VisualNodeKind::Displace(params)).unwrap();
            let mut patch = minimal_patch(1);
            patch.composition = Some(
                CompositionTree::try_from_parts(
                    vec![saved_group(group_id, vec![pos0], rack)],
                    vec![RootItem::Group { group_id }],
                    Some(2),
                    0.5,
                )
                .unwrap(),
            );
            patch
        };

        for timing in [EdgeTiming::CurrentFrame, EdgeTiming::PreviousFrame] {
            for amount_x in [0.0, 0.25] {
                let mut patch = patch_with(canvas_route(timing, amount_x));
                patch
                    .validate_creative_persistence()
                    .expect("a canvas route never closes a saved cycle");

                // It survives the YAML round trip as itself: a positionless
                // singleton has nothing to lose and nothing to tombstone.
                let yaml = serde_yaml::to_string(&patch).unwrap();
                assert!(yaml.contains("gesture_canvas"), "{yaml}");
                let restored = serde_yaml::from_str::<PatchState>(&yaml).unwrap();
                let composition = restored.composition.unwrap();
                let rack = &composition.group(group_id).unwrap().rack;
                let VisualNodeKind::Displace(params) = rack.iter().next().unwrap().kind else {
                    panic!("displace node")
                };
                assert_eq!(params.tap.source, SavedImageSource::GestureCanvas);
                assert_eq!(params.tap.timing, timing);
                assert!(rack.referenced_group_ids().next().is_none());
                assert!(rack.selected_layer_positions().next().is_none());
            }
        }
    }

    #[test]
    fn a_saved_program_tap_route_claims_no_edge_dormant_or_woken() {
        use crate::visual_rack::{DisplaceBoundary, DisplaceParams};

        let pos0 = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let tap_route = |timing, amount_x| DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::ProgramTap,
                timing,
            },
            amount_x,
            boundary: DisplaceBoundary::Wrap,
            ..DisplaceParams::default()
        };
        let patch_with = |params: DisplaceParams| {
            let mut rack = VisualRack::empty();
            rack.push(VisualNodeKind::Displace(params)).unwrap();
            let mut patch = minimal_patch(1);
            patch.composition = Some(
                CompositionTree::try_from_parts(
                    vec![saved_group(group_id, vec![pos0], rack)],
                    vec![RootItem::Group { group_id }],
                    Some(2),
                    0.5,
                )
                .unwrap(),
            );
            patch
        };

        for timing in [EdgeTiming::CurrentFrame, EdgeTiming::PreviousFrame] {
            for amount_x in [0.0, 0.25] {
                let mut patch = patch_with(tap_route(timing, amount_x));
                patch
                    .validate_creative_persistence()
                    .expect("a programme-tap route never closes a saved cycle");

                // It survives the YAML round trip as itself: a positionless
                // singleton has nothing to lose and nothing to tombstone.
                let yaml = serde_yaml::to_string(&patch).unwrap();
                assert!(yaml.contains("program_tap"), "{yaml}");
                let restored = serde_yaml::from_str::<PatchState>(&yaml).unwrap();
                let composition = restored.composition.unwrap();
                let rack = &composition.group(group_id).unwrap().rack;
                let VisualNodeKind::Displace(params) = rack.iter().next().unwrap().kind else {
                    panic!("displace node")
                };
                assert_eq!(params.tap.source, SavedImageSource::ProgramTap);
                assert_eq!(params.tap.timing, timing);
                assert!(rack.referenced_group_ids().next().is_none());
                assert!(rack.selected_layer_positions().next().is_none());
            }
        }
    }

    #[test]
    fn saved_residual_edges_are_dormant_at_zero_mix_and_cycle_per_slot_when_woken() {
        use crate::visual_rack::{ResidualParams, RESIDUAL_DETAIL_SLOT, RESIDUAL_STRUCTURE_SLOT};

        let pos0 = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let self_source = SavedImageSource::GroupOutput { group_id };
        let residual_rack = |params: ResidualParams| {
            let mut rack = VisualRack::empty();
            rack.push(VisualNodeKind::Residual(params)).unwrap();
            rack
        };
        // Both slots are inert `OneBelow` until one of them is pointed at the
        // owning group's own output.
        let self_route = |slot: u8, timing: EdgeTiming, mix: f32| {
            let mut params = ResidualParams {
                mix,
                ..ResidualParams::default()
            };
            *params.route_mut(slot).expect("both slots name a route") = SavedImageTap {
                source: self_source,
                timing,
            };
            params
        };
        let patch_with = |rack: VisualRack| {
            let mut patch = minimal_patch(1);
            patch.composition = Some(
                CompositionTree::try_from_parts(
                    vec![saved_group(group_id, vec![pos0], rack)],
                    vec![RootItem::Group { group_id }],
                    Some(2),
                    0.5,
                )
                .unwrap(),
            );
            patch
        };

        for slot in [RESIDUAL_STRUCTURE_SLOT, RESIDUAL_DETAIL_SLOT] {
            // Dormant forms claim no saved edge, so a self route round trips.
            let mut disabled = residual_rack(self_route(slot, EdgeTiming::CurrentFrame, 0.5));
            let disabled_id = disabled.iter().next().unwrap().stable_id;
            disabled.get_mut(disabled_id).unwrap().enabled = false;
            let mut zero_wet = residual_rack(self_route(slot, EdgeTiming::CurrentFrame, 0.5));
            let zero_wet_id = zero_wet.iter().next().unwrap().stable_id;
            zero_wet.get_mut(zero_wet_id).unwrap().wet = 0.0;
            let zero_mix = residual_rack(self_route(slot, EdgeTiming::CurrentFrame, 0.0));

            for dormant in [disabled, zero_wet, zero_mix] {
                let mut patch = patch_with(dormant);
                patch.validate_creative_persistence().unwrap();
                let yaml = serde_yaml::to_string(&patch).unwrap();
                let restored = serde_yaml::from_str::<PatchState>(&yaml).unwrap();
                let composition = restored.composition.unwrap();
                let rack = &composition.group(group_id).unwrap().rack;
                assert!(matches!(
                    rack.iter().next().unwrap().kind,
                    VisualNodeKind::Residual(_)
                ));
            }

            // Waking the mix wakes the saved edge on whichever slot carries the
            // self route, and the same-frame cycle is rejected at load rather
            // than reaching the planner.
            let error = patch_with(residual_rack(self_route(
                slot,
                EdgeTiming::CurrentFrame,
                0.25,
            )))
            .validate_creative_persistence()
            .expect_err("a woken same-frame Residual self route must be rejected");
            assert!(
                error.contains("cycle") || error.contains("graph"),
                "unexpected saved-edge rejection for slot {slot}: {error}"
            );

            // The identical route at N-1 is a legitimate saved feedback edge.
            patch_with(residual_rack(self_route(
                slot,
                EdgeTiming::PreviousFrame,
                0.25,
            )))
            .validate_creative_persistence()
            .unwrap();
        }
    }

    /// A group-scope Look is refused whole when any node's routing topology
    /// differs. Without the Residual arm the `_ => true` default would accept a
    /// mismatched pair and then copy values across two different recombinations.
    #[test]
    fn group_look_refuses_a_residual_whose_structure_or_detail_route_differs() {
        use crate::visual_rack::{
            EdgeTiming as RackEdgeTiming, ResolvedImageSource, ResolvedImageTap,
            RuntimeResidualParams, RuntimeVisualNodeKind, RESIDUAL_DETAIL_SLOT,
            RESIDUAL_STRUCTURE_SLOT,
        };

        let group_id = GroupId::new(9).unwrap();
        let routed = |source: ResolvedImageSource| ResolvedImageTap {
            source,
            timing: RackEdgeTiming::CurrentFrame,
        };
        let base = RuntimeResidualParams {
            structure: routed(ResolvedImageSource::OneBelow),
            detail: routed(ResolvedImageSource::CleanProgram),
            mix: 0.5,
            ..RuntimeResidualParams::default()
        };
        let group_with = |params: RuntimeResidualParams| {
            let mut group = empty_runtime_group_with_matte(group_id, None);
            group
                .rack
                .push(RuntimeVisualNodeKind::Residual(params))
                .unwrap();
            group
        };

        let sampled = group_with(base);
        let mut live = group_with(base);
        assert!(
            runtime_group_visual_topology_matches(&sampled, &live),
            "an identical pair of routes is Look-compatible"
        );
        assert!(apply_runtime_group_look_values(&sampled, &mut live));

        for slot in [RESIDUAL_STRUCTURE_SLOT, RESIDUAL_DETAIL_SLOT] {
            let mut rerouted = base;
            *rerouted.route_mut(slot).expect("both slots name a route") =
                routed(ResolvedImageSource::AllBelow);
            let mut live = group_with(rerouted);
            let before = live.clone();
            assert!(
                !runtime_group_visual_topology_matches(&sampled, &live),
                "slot {slot} must join the group Look routing gate"
            );
            assert!(!apply_runtime_group_look_values(&sampled, &mut live));
            assert_eq!(live.rack, before.rack, "a refused Look changes nothing");
        }
    }

    /// A pre-Residual patch has no `residual` node kind anywhere, and adding
    /// the kind must leave that YAML byte-identical on the way back out. The
    /// node's own absent fields default through `#[serde(default)]` to the
    /// exact bypass, so a legacy visual path is unchanged.
    #[test]
    fn a_legacy_patch_without_any_residual_section_round_trips_unchanged() {
        use crate::visual_rack::{ResidualBlock, ResidualParams, ResidualQuantization};

        let mut patch = minimal_patch(1);
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Grain(
            crate::visual_rack::GrainParams::default(),
        ))
        .unwrap();
        patch.layers[0].rack = Some(rack);
        patch.visual_schema_version = 1;
        patch.validate_creative_persistence().unwrap();

        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(
            !yaml.contains("residual"),
            "a patch with no Residual node must not gain a residual section"
        );
        let restored = serde_yaml::from_str::<PatchState>(&yaml).unwrap();
        assert_eq!(
            serde_yaml::to_string(&restored).unwrap(),
            yaml,
            "a legacy patch must round trip byte for byte"
        );
        assert_eq!(
            restored.layers[0].rack.as_ref().unwrap().iter().count(),
            patch.layers[0].rack.as_ref().unwrap().iter().count()
        );

        // An authored Residual node whose params section is entirely absent
        // deserializes through the patch's own YAML reader to the exact-bypass
        // default, so it claims no edge and renders the pre-Residual path.
        let rack_yaml =
            "nodes:\n- stable_id: 3\n  kind:\n    kind: residual\n    params: {}\nnext_node_id: 4\n";
        let rack: VisualRack = serde_yaml::from_str(rack_yaml).unwrap();
        let VisualNodeKind::Residual(params) = rack.iter().next().unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(params, ResidualParams::default());
        assert_eq!(params.mix, 0.0);
        assert_eq!(params.detail_gain, 1.0);
        assert_eq!(params.block, ResidualBlock::Eight);
        assert_eq!(params.quantization, ResidualQuantization::Off);
        assert_eq!(params.seed, 0);
        assert!(
            params.is_exact_bypass(),
            "an absent residual section must be an exact bypass"
        );

        // That dormant node claims no saved image edge, so a patch carrying it
        // validates and round trips exactly like the legacy one.
        let mut with_dormant = minimal_patch(1);
        with_dormant.layers[0].rack = Some(rack);
        with_dormant.visual_schema_version = 1;
        with_dormant.validate_creative_persistence().unwrap();
        let dormant_yaml = serde_yaml::to_string(&with_dormant).unwrap();
        let restored = serde_yaml::from_str::<PatchState>(&dormant_yaml).unwrap();
        assert_eq!(serde_yaml::to_string(&restored).unwrap(), dormant_yaml);
    }

    #[test]
    fn saved_symmetry_edges_are_dormant_per_slot_and_cycle_only_when_that_slot_is_armed() {
        use crate::symmetry::{SymmetryMode, SymmetryParams};

        let pos0 = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let self_tap = SavedImageTap {
            source: SavedImageSource::GroupOutput { group_id },
            timing: EdgeTiming::CurrentFrame,
        };
        let symmetry_rack = |params: SymmetryParams| {
            let mut rack = VisualRack::empty();
            rack.push(VisualNodeKind::Symmetry(params)).unwrap();
            rack
        };
        // Slot 1 holds the self route; slot 0 stays on its neutral default.
        let woken = SymmetryParams {
            mode: SymmetryMode::Dihedral,
            base_folds: 6.0,
            donors: [SymmetryParams::default().donors[0], self_tap],
            ..SymmetryParams::default()
        };
        let patch_with = |rack: VisualRack| {
            let mut patch = minimal_patch(1);
            patch.composition = Some(
                CompositionTree::try_from_parts(
                    vec![saved_group(group_id, vec![pos0], rack)],
                    vec![RootItem::Group { group_id }],
                    Some(2),
                    0.5,
                )
                .unwrap(),
            );
            patch
        };

        // Dormant forms claim no saved edge at either slot: an exact-default
        // node, a disabled or zero-wet node, and — the per-slot case — a woken
        // node whose slot 1 source-mask bit is clear.
        let mut disabled = symmetry_rack(SymmetryParams {
            source_mask: crate::symmetry::SymmetrySourceMask {
                donor1: true,
                ..crate::symmetry::SymmetrySourceMask::CARRIER_ONLY
            },
            ..woken
        });
        let disabled_id = disabled.iter().next().unwrap().stable_id;
        disabled.get_mut(disabled_id).unwrap().enabled = false;
        let mut zero_wet = symmetry_rack(SymmetryParams {
            source_mask: crate::symmetry::SymmetrySourceMask {
                donor1: true,
                ..crate::symmetry::SymmetrySourceMask::CARRIER_ONLY
            },
            ..woken
        });
        let zero_wet_id = zero_wet.iter().next().unwrap().stable_id;
        zero_wet.get_mut(zero_wet_id).unwrap().wet = 0.0;
        let exact_default = symmetry_rack(SymmetryParams {
            donors: [SymmetryParams::default().donors[0], self_tap],
            ..SymmetryParams::default()
        });
        let unarmed_slot = symmetry_rack(woken);

        for dormant in [disabled, zero_wet, exact_default, unarmed_slot] {
            let mut patch = patch_with(dormant);
            patch.validate_creative_persistence().unwrap();
            let yaml = serde_yaml::to_string(&patch).unwrap();
            let restored = serde_yaml::from_str::<PatchState>(&yaml).unwrap();
            let composition = restored.composition.unwrap();
            let rack = &composition.group(group_id).unwrap().rack;
            assert!(matches!(
                rack.iter().next().unwrap().kind,
                VisualNodeKind::Symmetry(_)
            ));
        }

        // Arming slot 1 wakes that slot's saved edge, and the same-frame self
        // route is then rejected at load rather than reaching the planner.
        let error = patch_with(symmetry_rack(SymmetryParams {
            source_mask: crate::symmetry::SymmetrySourceMask {
                donor1: true,
                ..crate::symmetry::SymmetrySourceMask::CARRIER_ONLY
            },
            ..woken
        }))
        .validate_creative_persistence()
        .expect_err("a woken same-frame Symmetry self route must be rejected");
        assert!(
            error.contains("cycle") || error.contains("graph"),
            "unexpected saved-edge rejection: {error}"
        );

        // The identical route at N-1 is a legitimate saved feedback edge.
        patch_with(symmetry_rack(SymmetryParams {
            source_mask: crate::symmetry::SymmetrySourceMask {
                donor1: true,
                ..crate::symmetry::SymmetrySourceMask::CARRIER_ONLY
            },
            donors: [
                SymmetryParams::default().donors[0],
                SavedImageTap {
                    timing: EdgeTiming::PreviousFrame,
                    ..self_tap
                },
            ],
            ..woken
        }))
        .validate_creative_persistence()
        .unwrap();
    }

    #[test]
    fn every_persisted_group_identity_advances_one_shared_cursor_without_reuse() {
        use crate::modulation::{GroupModParameter, SavedMissingTarget, StableNodeParameter};

        let group_id = |raw| GroupId::new(raw).unwrap();
        let pos0 = SavedLayerPosition::new(0).unwrap();

        let mut live_group = saved_group(
            group_id(1),
            vec![pos0],
            image_rack(
                SavedImageSource::MissingGroupOutput {
                    group_id: group_id(2),
                },
                EdgeTiming::CurrentFrame,
            ),
        );
        live_group.matte = Some(crate::visual_rack::ImageMatte {
            tap: SavedImageTap {
                source: SavedImageSource::MissingGroupOutput {
                    group_id: group_id(3),
                },
                timing: EdgeTiming::CurrentFrame,
            },
            ..crate::visual_rack::ImageMatte::default()
        });

        let mut patch = minimal_patch(1);
        patch.composition = Some(
            CompositionTree::try_from_parts(
                vec![live_group],
                vec![RootItem::Group {
                    group_id: group_id(1),
                }],
                Some(4),
                0.5,
            )
            .unwrap(),
        );
        patch.master_rack = Some(image_rack(
            SavedImageSource::MissingGroupOutput {
                group_id: group_id(4),
            },
            EdgeTiming::CurrentFrame,
        ));
        patch.layers[0].rack = Some(image_rack(
            SavedImageSource::MissingGroupOutput {
                group_id: group_id(5),
            },
            EdgeTiming::CurrentFrame,
        ));
        patch.layers[0].matte = LayerMatteConfig {
            input: SavedImageInput::MissingGroupOutput {
                group_id: group_id(6),
            },
            ..LayerMatteConfig::default()
        };

        let mut morph_group = saved_group(
            group_id(9),
            vec![pos0],
            image_rack(
                SavedImageSource::MissingGroupOutput {
                    group_id: group_id(10),
                },
                EdgeTiming::CurrentFrame,
            ),
        );
        morph_group.matte = Some(crate::visual_rack::ImageMatte {
            tap: SavedImageTap {
                source: SavedImageSource::MissingGroupOutput {
                    group_id: group_id(11),
                },
                timing: EdgeTiming::CurrentFrame,
            },
            ..crate::visual_rack::ImageMatte::default()
        });
        let morph_composition = CompositionTree::try_from_parts(
            vec![morph_group],
            vec![RootItem::Group {
                group_id: group_id(9),
            }],
            Some(12),
            0.5,
        )
        .unwrap();
        patch.morph = Some(crate::morph::MorphStateSnapshot {
            a: Some(crate::morph::MorphSlot {
                master_rack: Some(image_rack(
                    SavedImageSource::MissingGroupOutput {
                        group_id: group_id(7),
                    },
                    EdgeTiming::CurrentFrame,
                )),
                layer_racks: Some(vec![image_rack(
                    SavedImageSource::MissingGroupOutput {
                        group_id: group_id(8),
                    },
                    EdgeTiming::CurrentFrame,
                )]),
                composition: Some(morph_composition),
                ..crate::morph::MorphSlot::default()
            }),
            ..crate::morph::MorphStateSnapshot::default()
        });

        let node_id = NodeId::new(1).unwrap();
        let stable_targets = [
            SavedStableModTarget::GroupValue {
                group_id: group_id(12),
                parameter: GroupModParameter::Opacity,
            },
            SavedStableModTarget::MissingGroup {
                group_id: group_id(13),
                missing_target: SavedMissingTarget::GroupValue {
                    parameter: GroupModParameter::Opacity,
                },
            },
            SavedStableModTarget::Node {
                scope: SavedStableModScope::Group {
                    group_id: group_id(14),
                },
                node_id,
                parameter: StableNodeParameter::Wet,
            },
            SavedStableModTarget::MissingNode {
                scope: SavedStableModScope::Group {
                    group_id: group_id(15),
                },
                node_id,
                parameter: StableNodeParameter::Wet,
            },
        ];
        let mut modulation = ModConfig::from_matrix(&ModMatrix::new());
        modulation.routings = stable_targets
            .into_iter()
            .enumerate()
            .map(|(index, stable_target)| RoutingConfig {
                source: format!("lfo{}", index % NUM_LFOS),
                target: stable_target.persistence_key(),
                stable_target: Some(stable_target),
                depth: 0.5,
                curve: "linear".to_string(),
                curve_amount: 0.0,
                attack: 0.0,
                release: 0.0,
            })
            .collect();
        patch.modulation = Some(modulation);

        assert_eq!(
            patch
                .persisted_group_ids()
                .into_iter()
                .map(GroupId::get)
                .collect::<Vec<_>>(),
            (1..=15).collect::<Vec<_>>()
        );
        patch.validate_creative_persistence().unwrap();
        assert_eq!(patch.composition.as_ref().unwrap().next_group_id_raw(), 16);

        let yaml = serde_yaml::to_string(&patch).unwrap();
        let mut restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        let composition = restored.composition.as_mut().unwrap();
        assert_eq!(composition.next_group_id_raw(), 16);
        let first = composition
            .insert_empty_group(
                crate::composition::GroupName::new("first-after-tombstones").unwrap(),
                1,
            )
            .unwrap();
        assert_eq!(first, group_id(16));
        composition.remove_group_ungroup(first).unwrap();
        let second = composition
            .insert_empty_group(
                crate::composition::GroupName::new("second-after-delete").unwrap(),
                1,
            )
            .unwrap();
        assert_eq!(second, group_id(17));
    }

    #[test]
    fn low_cursor_yaml_reserves_morph_and_modulation_node_ids_per_owner() {
        use crate::modulation::StableNodeParameter;

        fn retired_rack(next_node_id: u64) -> VisualRack {
            VisualRack::try_from_parts(Vec::new(), Some(next_node_id)).unwrap()
        }

        let position = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(1).unwrap();
        let morph_composition = |next_node_id| {
            CompositionTree::try_from_parts(
                vec![saved_group(
                    group_id,
                    vec![position],
                    retired_rack(next_node_id),
                )],
                vec![RootItem::Group { group_id }],
                Some(2),
                0.5,
            )
            .unwrap()
        };

        let mut patch = minimal_patch(1);
        // All three current owners expose the lowest valid authored cursor.
        patch.master_rack = Some(retired_rack(3));
        patch.layers[0].rack = Some(retired_rack(3));
        patch.composition = Some(
            CompositionTree::try_from_parts(
                vec![saved_group(group_id, vec![position], retired_rack(3))],
                vec![RootItem::Group { group_id }],
                Some(2),
                0.5,
            )
            .unwrap(),
        );
        // Empty Morph racks still retain every identity below their cursor.
        // Both slots participate in the same owner-specific allocation law.
        patch.morph = Some(crate::morph::MorphStateSnapshot {
            a: Some(crate::morph::MorphSlot {
                master_rack: Some(retired_rack(8)),
                layer_racks: Some(vec![retired_rack(12)]),
                composition: Some(morph_composition(16)),
                ..crate::morph::MorphSlot::default()
            }),
            b: Some(crate::morph::MorphSlot {
                master_rack: Some(retired_rack(9)),
                layer_racks: Some(vec![retired_rack(13)]),
                composition: Some(morph_composition(17)),
                ..crate::morph::MorphSlot::default()
            }),
            ..crate::morph::MorphStateSnapshot::default()
        });

        let node_id = |raw| NodeId::new(raw).unwrap();
        let stable_targets = [
            SavedStableModTarget::Node {
                scope: SavedStableModScope::Master,
                node_id: node_id(9),
                parameter: StableNodeParameter::Wet,
            },
            SavedStableModTarget::MissingNode {
                scope: SavedStableModScope::Master,
                node_id: node_id(10),
                parameter: StableNodeParameter::Wet,
            },
            SavedStableModTarget::Node {
                scope: SavedStableModScope::SavedLayer { position },
                node_id: node_id(13),
                parameter: StableNodeParameter::Wet,
            },
            SavedStableModTarget::MissingNode {
                scope: SavedStableModScope::SavedLayer { position },
                node_id: node_id(14),
                parameter: StableNodeParameter::Wet,
            },
            SavedStableModTarget::Node {
                scope: SavedStableModScope::Group { group_id },
                node_id: node_id(17),
                parameter: StableNodeParameter::Wet,
            },
            SavedStableModTarget::MissingNode {
                scope: SavedStableModScope::Group { group_id },
                node_id: node_id(18),
                parameter: StableNodeParameter::Wet,
            },
        ];
        let mut modulation = ModConfig::from_matrix(&ModMatrix::new());
        modulation.routings = stable_targets
            .into_iter()
            .enumerate()
            .map(|(index, stable_target)| RoutingConfig {
                source: format!("lfo{}", index % NUM_LFOS),
                target: stable_target.persistence_key(),
                stable_target: Some(stable_target),
                depth: 0.5,
                curve: "linear".to_string(),
                curve_amount: 0.0,
                attack: 0.0,
                release: 0.0,
            })
            .collect();
        patch.modulation = Some(modulation);

        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(yaml.contains("next_node_id: 3"));
        let mut restored: PatchState = serde_yaml::from_str(&yaml).unwrap();

        let master_id = restored
            .master_rack
            .as_mut()
            .unwrap()
            .push(VisualNodeKind::Shift(Default::default()))
            .unwrap();
        let layer_id = restored.layers[0]
            .rack
            .as_mut()
            .unwrap()
            .push(VisualNodeKind::Shift(Default::default()))
            .unwrap();
        let group_node_id = restored
            .composition
            .as_mut()
            .unwrap()
            .group_mut(group_id)
            .unwrap()
            .rack
            .push(VisualNodeKind::Shift(Default::default()))
            .unwrap();
        assert_eq!(master_id, node_id(11));
        assert_eq!(layer_id, node_id(15));
        assert_eq!(group_node_id, node_id(19));

        // Repair is persisted across both Morph slots as well, so no saved
        // representation of the same owner can later reintroduce reuse.
        let morph = restored.morph.as_ref().unwrap();
        for slot in [morph.a.as_ref().unwrap(), morph.b.as_ref().unwrap()] {
            assert_eq!(slot.master_rack.as_ref().unwrap().next_node_id_raw(), 11);
            assert_eq!(slot.layer_racks.as_ref().unwrap()[0].next_node_id_raw(), 15);
            assert_eq!(
                slot.composition
                    .as_ref()
                    .unwrap()
                    .group(group_id)
                    .unwrap()
                    .rack
                    .next_node_id_raw(),
                19
            );
        }
    }

    #[test]
    fn maximum_retained_node_id_exhausts_instead_of_wrapping() {
        use crate::modulation::StableNodeParameter;

        let mut patch = minimal_patch(0);
        patch.master_rack = Some(VisualRack::empty());
        let target = SavedStableModTarget::MissingNode {
            scope: SavedStableModScope::Master,
            node_id: NodeId::new(u64::MAX).unwrap(),
            parameter: StableNodeParameter::Wet,
        };
        let mut modulation = ModConfig::from_matrix(&ModMatrix::new());
        modulation.routings = vec![RoutingConfig {
            source: "lfo0".to_string(),
            target: target.persistence_key(),
            stable_target: Some(target),
            depth: 0.5,
            curve: "linear".to_string(),
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
        }];
        patch.modulation = Some(modulation);

        let yaml = serde_yaml::to_string(&patch).unwrap();
        let mut restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        let rack = restored.master_rack.as_mut().unwrap();
        assert_eq!(rack.next_node_id_raw(), 0);
        assert_eq!(
            rack.push(VisualNodeKind::Shift(Default::default())),
            Err(crate::visual_rack::RackError::NodeIdExhausted)
        );
    }

    #[test]
    fn synthesized_legacy_rack_reserves_retired_ids_before_runtime_publication() {
        use crate::modulation::StableNodeParameter;

        let mut patch = minimal_patch(0);
        patch.morph = Some(crate::morph::MorphStateSnapshot {
            a: Some(crate::morph::MorphSlot {
                master_rack: Some(VisualRack::try_from_parts(Vec::new(), Some(8)).unwrap()),
                ..crate::morph::MorphSlot::default()
            }),
            ..crate::morph::MorphStateSnapshot::default()
        });
        let target = SavedStableModTarget::MissingNode {
            scope: SavedStableModScope::Master,
            node_id: NodeId::new(10).unwrap(),
            parameter: StableNodeParameter::Wet,
        };
        let mut modulation = ModConfig::from_matrix(&ModMatrix::new());
        modulation.routings = vec![RoutingConfig {
            source: "lfo0".to_string(),
            target: target.persistence_key(),
            stable_target: Some(target),
            depth: 0.5,
            curve: "linear".to_string(),
            curve_amount: 0.0,
            attack: 0.0,
            release: 0.0,
        }];
        patch.modulation = Some(modulation);

        let yaml = serde_yaml::to_string(&patch).unwrap();
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.master_rack.is_none());

        let mut runtime_master = RuntimeVisualRack::empty();
        let mut runtime_composition =
            RuntimeComposition::try_from_parts(Vec::new(), Vec::new(), Some(1), 0.5).unwrap();
        restored
            .apply_with_composition(
                &mut EffectUniforms::default(),
                &mut SpatialTransform::default(),
                &mut [],
                &mut runtime_master,
                &mut [],
                &mut runtime_composition,
                &mut NtscParams::default(),
                &mut ModMatrix::new(),
                &mut TemporalParams::default(),
            )
            .unwrap();
        assert_eq!(runtime_master.next_node_id_raw(), 11);
        assert_eq!(
            runtime_master
                .push(crate::visual_rack::RuntimeVisualNodeKind::Shift(
                    Default::default(),
                ))
                .unwrap(),
            NodeId::new(11).unwrap()
        );
    }

    #[test]
    fn synthesized_legacy_composition_reserves_persisted_group_tombstones() {
        let pos0 = SavedLayerPosition::new(0).unwrap();
        let tombstone = GroupId::new(7).unwrap();
        let mut patch = minimal_patch(1);
        patch.layers[0].matte = LayerMatteConfig {
            input: SavedImageInput::MissingGroupOutput {
                group_id: tombstone,
            },
            ..LayerMatteConfig::default()
        };

        let mut synthesized = legacy_composition_for_positions(&[pos0]).unwrap();
        patch.reserve_persisted_group_ids(&mut synthesized);
        let allocated = synthesized
            .insert_empty_group(
                crate::composition::GroupName::new("after-legacy-tombstone").unwrap(),
                1,
            )
            .unwrap();
        assert_eq!(allocated, GroupId::new(8).unwrap());
        synthesized.remove_group_ungroup(allocated).unwrap();
        assert_eq!(synthesized.next_group_id_raw(), 9);
    }

    #[test]
    fn image_graph_counts_logical_taps_without_expanding_all_below_prefixes() {
        let mut previous_overflow = minimal_patch(9);
        for (position, layer) in previous_overflow.layers.iter_mut().enumerate() {
            layer.rack = Some(image_rack(
                SavedImageSource::SelectedLayer {
                    layer_position: saved_position_at(position).unwrap(),
                    stage: LayerImageStage::PostLocalEffects,
                },
                EdgeTiming::PreviousFrame,
            ));
        }
        assert!(previous_overflow
            .validate_creative_persistence()
            .unwrap_err()
            .contains("previous-frame taps"));

        let mut all_below = minimal_patch(66);
        all_below.master_rack = Some(image_rack(
            SavedImageSource::AllBelow,
            EdgeTiming::CurrentFrame,
        ));
        all_below.validate_creative_persistence().unwrap();
    }

    #[test]
    fn below_topology_never_double_counts_members_as_their_group_output() {
        let positions = (0..4)
            .map(saved_position_at)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let group_id = GroupId::new(1).unwrap();
        let composition = CompositionTree::try_from_parts(
            vec![saved_group(
                group_id,
                vec![positions[1], positions[2]],
                VisualRack::empty(),
            )],
            vec![
                RootItem::Layer {
                    layer: positions[0],
                    bus: crate::composition::BusAssignment::Program,
                },
                RootItem::Group { group_id },
                RootItem::Layer {
                    layer: positions[3],
                    bus: crate::composition::BusAssignment::Program,
                },
            ],
            Some(2),
            0.5,
        )
        .unwrap();
        let (_, below, _) = validation_topology(&composition).unwrap();
        let layer0 = validation_layer_scope(positions[0]).unwrap();
        let member1 = validation_layer_scope(positions[1]).unwrap();
        let member2 = validation_layer_scope(positions[2]).unwrap();
        let layer3 = validation_layer_scope(positions[3]).unwrap();
        let group = VisualScopeId::Group(group_id);
        assert_eq!(below[&member1], vec![layer0]);
        assert_eq!(below[&member2], vec![layer0, member1]);
        assert_eq!(below[&group], vec![layer0]);
        assert_eq!(below[&layer3], vec![layer0, group]);
        assert_eq!(below[&VisualScopeId::Master], vec![layer0, group, layer3]);
    }

    #[test]
    fn apply_look_matches_groups_independently_and_reports_exact_stable_ids() {
        let pos0 = SavedLayerPosition::new(0).unwrap();
        let pos1 = SavedLayerPosition::new(1).unwrap();
        let layer_ids = [
            StableLayerId::new(101).unwrap(),
            StableLayerId::new(202).unwrap(),
        ];
        let group1 = GroupId::new(1).unwrap();
        let group2 = GroupId::new(2).unwrap();

        let mut rack1 = VisualRack::empty();
        let node1 = rack1
            .push(VisualNodeKind::Transform(SpatialTransform {
                position: [0.2, -0.3],
                ..SpatialTransform::default()
            }))
            .unwrap();
        let mut rack2 = VisualRack::empty();
        let node2 = rack2
            .push(VisualNodeKind::Shift(crate::visual_rack::ShiftParams {
                amount: 0.7,
                ..crate::visual_rack::ShiftParams::default()
            }))
            .unwrap();
        let mut saved1 = saved_group(group1, vec![pos0], rack1);
        saved1.opacity = 0.25;
        saved1.transform.rotation_deg = 17.0;
        let mut saved2 = saved_group(group2, vec![pos1], rack2);
        saved2.opacity = 0.35;
        let saved = CompositionTree::try_from_parts(
            vec![saved1, saved2],
            vec![
                RootItem::Group { group_id: group1 },
                RootItem::Group { group_id: group2 },
            ],
            Some(3),
            0.2,
        )
        .unwrap();
        let mut live = saved
            .resolve(|position| position.resolve(&layer_ids).copied())
            .unwrap();
        {
            let live1 = live.group_mut(group1).unwrap();
            live1.opacity = 0.91;
            live1.transform.rotation_deg = -44.0;
            live1.rack.get_mut(node1).unwrap().wet = 0.1;
        }
        {
            let live2 = live.group_mut(group2).unwrap();
            live2.opacity = 0.82;
            live2.rack.get_mut(node2).unwrap().wet = 0.2;
            live2
                .rack
                .push(crate::visual_rack::RuntimeVisualNodeKind::Grain(
                    crate::visual_rack::GrainParams::default(),
                ))
                .unwrap();
        }
        live.set_bus_crossfade(0.8);

        let mut summary = LookApplySummary::default();
        apply_saved_composition_look(&saved, &mut live, &layer_ids, &mut summary);

        assert_eq!(summary.applied_group_ids, vec![group1]);
        assert_eq!(summary.skipped_group_ids, vec![group2]);
        assert_eq!(summary.applied_groups, 1);
        assert_eq!(summary.skipped_groups, 1);
        assert_eq!(
            summary.applied_nodes,
            vec![LookNodeRef {
                scope: LookRackScope::Group(group1),
                node_id: node1,
            }]
        );
        assert_eq!(
            summary.skipped_nodes,
            vec![LookNodeRef {
                scope: LookRackScope::Group(group2),
                node_id: node2,
            }]
        );
        assert!(summary.applied_bus_crossfade);
        assert_eq!(live.bus_crossfade(), 0.2);
        let applied = live.group(group1).unwrap();
        assert_eq!(applied.opacity, 0.25);
        assert_eq!(applied.transform.rotation_deg, 17.0);
        assert_eq!(applied.rack.get(node1).unwrap().wet, 1.0);
        let skipped = live.group(group2).unwrap();
        assert_eq!(skipped.opacity, 0.82);
        assert_eq!(skipped.rack.get(node2).unwrap().wet, 0.2);
    }

    #[test]
    fn apply_look_never_retargets_a_live_layer_matte() {
        let layer_ids = [
            StableLayerId::new(101).unwrap(),
            StableLayerId::new(202).unwrap(),
        ];
        let sampled = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::SelectedLayer {
                layer_position: SavedLayerPosition::new(0).unwrap(),
                stage: LayerImageStage::PostLocalEffects,
            },
            channel: crate::image_routing::MatteChannel::Luma,
            invert: true,
            amount: 0.8,
            threshold: 0.3,
            softness: 0.2,
        };
        let mut live = LayerMatte {
            enabled: true,
            input: crate::image_routing::ImageInput::SelectedLayer {
                layer_id: layer_ids[1],
                stage: LayerImageStage::PostLocalEffects,
            },
            channel: crate::image_routing::MatteChannel::Luma,
            invert: true,
            amount: 0.1,
            threshold: 0.6,
            softness: 0.4,
        };
        let rerouted = live;
        assert!(!apply_layer_matte_look_values(
            sampled, &mut live, &layer_ids
        ));
        assert_eq!(live, rerouted);

        live.input = crate::image_routing::ImageInput::SelectedLayer {
            layer_id: layer_ids[0],
            stage: LayerImageStage::PostLocalEffects,
        };
        assert!(apply_layer_matte_look_values(
            sampled, &mut live, &layer_ids
        ));
        assert_eq!(live.amount, 0.8);
        assert_eq!(live.threshold, 0.3);
        assert_eq!(live.softness, 0.2);
        assert_eq!(
            live.input,
            crate::image_routing::ImageInput::SelectedLayer {
                layer_id: layer_ids[0],
                stage: LayerImageStage::PostLocalEffects,
            }
        );
    }

    /// The gesture sections are additive in the strictest sense: a patch that
    /// never authored one must serialize to the exact bytes it did before S3b
    /// existed, and a patch written before S3b must load with both sections
    /// absent rather than defaulted-and-claimed.
    #[test]
    fn an_absent_gesture_section_is_exactly_the_pre_gesture_path_and_round_trips_unchanged() {
        let patch = minimal_patch(2);
        assert_eq!(patch.gesture_track, None);
        assert_eq!(patch.gesture_canvas, None);
        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(!yaml.contains("gesture_track:"));
        assert!(!yaml.contains("gesture_canvas:"));

        // A pre-S3b document names neither section and must not acquire one.
        let legacy_yaml = "master: {}\nmaster_transform: {}\nlayers:\n  - filename: old.mov\n";
        let legacy: PatchState = serde_yaml::from_str(legacy_yaml).unwrap();
        assert_eq!(legacy.gesture_track, None);
        assert_eq!(legacy.gesture_canvas, None);
        let reserialized = serde_yaml::to_string(&legacy).unwrap();
        assert!(!reserialized.contains("gesture"));
        let round_tripped: PatchState = serde_yaml::from_str(&reserialized).unwrap();
        assert_eq!(
            serde_yaml::to_string(&round_tripped).unwrap(),
            reserialized,
            "a legacy patch must round-trip unchanged"
        );

        // The default canvas is the pre-gesture path too, so an authored patch
        // only grows the section once a value actually moved.
        let mut authored = minimal_patch(1);
        authored.gesture_canvas = Some(GestureCanvasConfig::default());
        assert!(authored.gesture_canvas.unwrap().is_default());
        let mut moved = minimal_patch(1);
        moved.gesture_canvas = Some(GestureCanvasConfig {
            radius: 0.25,
            strength: 0.75,
            retention: 0.5,
        });
        let yaml = serde_yaml::to_string(&moved).unwrap();
        assert!(yaml.contains("gesture_canvas:"));
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.gesture_canvas, moved.gesture_canvas);
    }

    /// The B9 take section rides the gesture-track law: carried whole, absent
    /// by default, and gated by its own checksum-verifying deserializer.
    #[test]
    fn the_performance_take_section_is_additive_and_checksum_gated() {
        let patch = minimal_patch(1);
        assert_eq!(patch.performance_take, None);
        let yaml = serde_yaml::to_string(&patch).unwrap();
        assert!(!yaml.contains("performance_take:"));

        let mut take = crate::performance_track::PerformanceTake::default();
        take.record_accepted(
            0,
            crate::performance_track::PerformanceControl::Master {
                param: "brightness".to_string(),
            },
            crate::performance_track::PerformanceValueLaw::Unit {
                min: -1.0,
                max: 1.0,
            },
            &crate::performance_track::PerformanceRawValue::Continuous(0.5),
        )
        .unwrap();
        take.finalize(4);
        let mut carried = minimal_patch(1);
        carried.performance_take = Some(
            crate::performance_track::PerformanceTakeDocument::capture(&take),
        );
        let yaml = serde_yaml::to_string(&carried).unwrap();
        assert!(yaml.contains("performance_take:"));
        let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.performance_take, carried.performance_take);
        assert_eq!(
            restored.performance_take.unwrap().decode().unwrap(),
            take,
            "the carried take decodes to the exact recorded stream"
        );

        // A tampered digest is refused at the patch boundary, never repaired.
        let tampered = yaml.replace(&take.checksum_hex()[..8], "00000000");
        assert!(
            serde_yaml::from_str::<PatchState>(&tampered).is_err(),
            "a mismatched take digest must reject the patch section"
        );
    }

    /// The B10 sections are additive: a matrix that never authored an
    /// envelope, macro, or seed emits no section at all, and an authored one
    /// round-trips with hostile scalars sanitized and unknown tokens keeping
    /// the slot's defaults.
    #[test]
    fn b10_mod_sections_are_additive_and_sanitize_on_load() {
        let default_config = ModConfig::from_matrix(&crate::modulation::ModMatrix::new());
        assert!(default_config.envelopes.is_empty());
        assert!(default_config.macros.is_empty());
        assert_eq!(default_config.generator_seed, 0);
        let yaml = serde_yaml::to_string(&default_config).unwrap();
        assert!(!yaml.contains("envelopes:"));
        assert!(!yaml.contains("macros:"));
        assert!(!yaml.contains("generator_seed:"));

        let mut matrix = crate::modulation::ModMatrix::new();
        matrix.envelopes[1].attack = 0.25;
        matrix.envelopes[1].trigger = crate::modulation::EnvelopeTrigger::Beat(2);
        matrix.envelopes[1].mode = crate::modulation::EnvelopeMode::Gate;
        matrix.set_macro(2, 0.6);
        matrix.generator_seed = 41;
        let captured = ModConfig::from_matrix(&matrix);
        assert_eq!(captured.envelopes.len(), 4, "an emitted section is whole");
        assert_eq!(captured.macros.len(), 4);
        assert_eq!(captured.generator_seed, 41);
        let yaml = serde_yaml::to_string(&captured).unwrap();
        let restored: ModConfig = serde_yaml::from_str(&yaml).unwrap();
        let mut fresh = crate::modulation::ModMatrix::new();
        restored.apply_to_matrix(&mut fresh);
        assert_eq!(fresh.envelopes[1].attack, 0.25);
        assert_eq!(
            fresh.envelopes[1].trigger,
            crate::modulation::EnvelopeTrigger::Beat(2)
        );
        assert_eq!(
            fresh.envelopes[1].mode,
            crate::modulation::EnvelopeMode::Gate
        );
        assert!((fresh.macros[2] - 0.6).abs() < 1.0e-6);
        assert_eq!(fresh.generator_seed, 41);
        // Runtime state never persists: bends released, levels silent.
        assert_eq!(fresh.bend_held, [false; crate::modulation::NUM_BENDS]);
        assert_eq!(fresh.envelopes[1].level(), 0.0);

        // Hostile scalars sanitize; unknown tokens keep the defaults.
        let hostile: ModConfig = serde_yaml::from_str(
            "bpm: 120\nenvelopes:\n  - attack: .nan\n    decay: 900\n    trigger: bend9\n    mode: sideways\nmacros: [9.0, -3.0, .nan, 0.5]\ngenerator_seed: 7\n",
        )
        .unwrap();
        let mut sanitized = crate::modulation::ModMatrix::new();
        hostile.apply_to_matrix(&mut sanitized);
        assert!((sanitized.envelopes[0].attack - 0.02).abs() < 1.0e-6);
        assert!(
            (sanitized.envelopes[0].decay - 30.0).abs() < 1.0e-6,
            "an oversized decay clamps to the documented range"
        );
        assert_eq!(
            sanitized.envelopes[0].trigger,
            crate::modulation::EnvelopeTrigger::Bend(0),
            "an unknown trigger token keeps the default"
        );
        assert_eq!(sanitized.macros[0], 1.0);
        assert_eq!(sanitized.macros[1], 0.0);
        assert_eq!(sanitized.macros[2], 0.0);
        assert!((sanitized.macros[3] - 0.5).abs() < 1.0e-6);
        assert_eq!(sanitized.generator_seed, 7);
    }

    /// Hostile canvas values sanitize on load and an unknown key inside the
    /// section fails closed rather than being silently ignored.
    #[test]
    fn a_hostile_gesture_canvas_section_sanitizes_on_load_and_rejects_unknown_fields() {
        let hostile: PatchState = serde_yaml::from_str(
            "master: {}\nlayers: []\ngesture_canvas:\n  radius: .nan\n  strength: 9\n  retention: -4\n",
        )
        .unwrap();
        let canvas = hostile.gesture_canvas.expect("section present");
        // Non-finite takes the documented default; finite out-of-range clamps.
        assert!((canvas.radius - GestureCanvasConfig::default().radius).abs() < 1.0e-6);
        assert_eq!(canvas.strength, 1.0);
        assert_eq!(canvas.retention, 0.0);

        assert!(serde_yaml::from_str::<PatchState>(
            "master: {}\nlayers: []\ngesture_canvas:\n  radius: 0.2\n  decay: 3\n",
        )
        .is_err());

        // An omitted field inside a present section falls back to its default
        // rather than to zero.
        let partial: PatchState =
            serde_yaml::from_str("master: {}\nlayers: []\ngesture_canvas:\n  radius: 0.2\n")
                .unwrap();
        let canvas = partial.gesture_canvas.expect("section present");
        assert_eq!(canvas.radius, 0.2);
        assert_eq!(canvas.strength, GestureCanvasConfig::default().strength);
        assert_eq!(canvas.retention, GestureCanvasConfig::default().retention);
    }

    #[test]
    fn generator_sections_round_trip_and_absent_sections_keep_old_bytes() {
        // An ordinary layer serializes without either generator section, so
        // every pre-B7 patch keeps its bytes and canonical hashes.
        let plain = LayerConfig {
            filename: "clip.mp4".into(),
            source_path: String::new(),
            opacity: 1.0,
            blend_mode: "normal".into(),
            speed: 1.0,
            fps: 30.0,
            paused: false,
            visible: true,
            bypass_master_fx: false,
            reroll_on_loop: false,
            effects: EffectsConfig::default(),
            transform: SpatialTransform::default(),
            motion: None,
            rack: None,
            clip_slots: ClipSlots::singleton(ClipSlotConfig::from_legacy(
                "clip.mp4".into(),
                String::new(),
                1.0,
                30.0,
            )),
            active_clip_slot: None,
            matte: LayerMatteConfig::default(),
            pattern: None,
            text_page: None,
        };
        let yaml = serde_yaml::to_string(&plain).unwrap();
        assert!(!yaml.contains("pattern"));
        assert!(!yaml.contains("text_page"));
        let absent: LayerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(absent.pattern, None);
        assert_eq!(absent.text_page, None);

        // A pattern section round trips whole, including the three closed
        // vocabularies.
        let authored = crate::pattern_synth::PatternSynthParams {
            shape: crate::pattern_synth::PatternShape::Tunnel,
            wave: crate::pattern_synth::PatternWave::SampleHold,
            color_mode: crate::pattern_synth::PatternColorMode::Bands,
            wavefold: 0.7,
            hue: 0.31,
            ..crate::pattern_synth::PatternSynthParams::default()
        };
        let config = PatternSynthConfig::from_params(&authored);
        let pattern_yaml = serde_yaml::to_string(&config).unwrap();
        assert!(pattern_yaml.contains("shape: tunnel"));
        assert!(pattern_yaml.contains("wave: sample_hold"));
        assert!(pattern_yaml.contains("color_mode: bands"));
        let restored: PatternSynthConfig = serde_yaml::from_str(&pattern_yaml).unwrap();
        assert_eq!(restored, config);
        assert_eq!(restored.to_params(), authored.sanitized());

        // Hostile scalars sanitize to neutral values; unknown tokens and
        // unknown fields are deserialization rejections.
        let hostile: PatternSynthConfig =
            serde_yaml::from_str("freq_x: .nan\nbrightness: 99.0\n").unwrap();
        assert_eq!(hostile.to_params().freq_x, 0.18);
        assert_eq!(hostile.to_params().brightness, 1.5);
        assert!(
            serde_yaml::from_str::<PatternSynthConfig>("shape: hypercube\n").is_err(),
            "an unknown shape token is a deserialization rejection"
        );
        assert!(
            serde_yaml::from_str::<PatternSynthConfig>("voltage: 1.0\n").is_err(),
            "an unknown pattern field is a deserialization rejection"
        );

        // The text page round trips whole and its oversized body truncates on
        // load rather than refusing the patch.
        let page = crate::text_page::TextPageParams {
            body: "HELLO\nWORLD".into(),
            font: crate::text_page::TextPageFont::Sans,
            shape: crate::text_page::TextPageShape::Starburst,
            rot_degrees: 33.0,
            ..crate::text_page::TextPageParams::default()
        };
        let text_config = TextPageConfig::from_params(&page);
        let text_yaml = serde_yaml::to_string(&text_config).unwrap();
        assert!(text_yaml.contains("font: sans"));
        assert!(text_yaml.contains("shape: starburst"));
        let text_restored: TextPageConfig = serde_yaml::from_str(&text_yaml).unwrap();
        assert_eq!(text_restored, text_config);
        assert_eq!(text_restored.to_params(), page.sanitized());
        assert!(
            serde_yaml::from_str::<TextPageConfig>("font: papyrus\n").is_err(),
            "a face outside the bundled two is a deserialization rejection"
        );

        // Apply-to-layer look transfer stays kind-gated: a pattern config on
        // a saved position moves values only through a live pattern layer,
        // which is exercised by the runtime tests; here the DTO carries the
        // sentinel contract.
        assert_eq!(crate::layers::PATTERN_SOURCE_PATH, "synth://pattern");
        assert_eq!(crate::layers::TEXT_PAGE_SOURCE_PATH, "text://page");
    }
}
