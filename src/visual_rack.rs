//! Bounded authored model for the Milestone 2 Collision Rack.
//!
//! This module deliberately contains no GPU objects. It is the common,
//! deterministic contract consumed by patch migration, evaluation, live
//! rendering, export, modulation, and the browser. Untrusted collections are
//! bounded while they are deserialized, stable IDs are values rather than
//! indices, and all allocation arithmetic is checked before a renderer may
//! allocate replacement resources.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::image_routing::{LayerImageStage, StableLayerId};
use crate::performance::SavedLayerPosition;
use crate::spatial::SpatialTransform;

pub const MAX_NODES_PER_RACK: usize = 8;
pub const MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK: u32 = 32;
pub const MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME: u32 = 1_024;
/// One straight-alpha-aware bilinear lookup is implemented as four explicit
/// texture loads so interpolation can happen in premultiplied space. Resource
/// plans count those shader texture operations, not just the logical lookup.
pub const PREMULTIPLIED_BILINEAR_TEXTURE_OPS: u8 = 4;
/// Preserve the pre-M6 ceiling of 32 logical bilinear lookups while charging
/// the four shader texture operations now required by each Advanced lookup.
pub const MAX_TEXTURE_SAMPLES_PER_RACK: u32 =
    MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK * PREMULTIPLIED_BILINEAR_TEXTURE_OPS as u32;
/// Preserve the pre-M6 frame ceiling of 1,024 logical bilinear lookups under
/// the same explicit shader-operation accounting used by rack descriptors.
pub const MAX_TEXTURE_SAMPLES_PER_FRAME: u32 =
    MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME * PREMULTIPLIED_BILINEAR_TEXTURE_OPS as u32;
pub const MAX_CURRENT_IMAGE_TAPS: usize = 64;
pub const MAX_PREVIOUS_IMAGE_TAPS: usize = 8;
pub const MAX_IMAGE_DEPENDENCIES: usize = MAX_CURRENT_IMAGE_TAPS + MAX_PREVIOUS_IMAGE_TAPS;
pub const MAX_GRAPH_SCOPES: usize = 274;
pub const MAX_SAMPLED_TEXTURES_PER_PASS: u32 = 3;
pub const MAX_CREATIVE_GPU_BYTES: u64 = 512 * 1024 * 1024;
/// Rack ping/pong, A lane, B lane, Program lane, and one shared group-local
/// scratch surface. Image taps/history are charged in addition to this base.
pub const BASE_CREATIVE_SURFACE_LAYERS: u32 = 6;
/// CollisionRackExecutor currently owns a distinct RGBA16Float ping/pong pair
/// in addition to the six composition-host surfaces. Keeping this explicit in
/// the ledger prevents the host and rack executors from both claiming the same
/// memory. It may become zero when they genuinely share physical surfaces.
pub const ADVANCED_RACK_SURFACE_LAYERS: u32 = 2;
/// ProgramHistory is double-buffered so an encoder that was submitted but
/// later rejected by readback/output acceptance cannot overwrite committed
/// N-1 pixels. The dependency graph already charges the committed read image;
/// this constant charges its inactive write partner once per plan.
pub const ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS: u32 = 1;
/// Advanced LegacyTemporal retains the frozen 24-clean-frame ring plus one
/// post-temporal feedback image in Compat8. Working images and exact N-1
/// Program history remain RGBA16Float. Separating formats keeps 1080p usable
/// under the fixed creative cap without concealing the temporal allocation.
pub const ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS: u32 = 25;

/// Stable identity of a node within one authored rack. IDs are never vector
/// indices and zero is never accepted from a patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

impl NodeId {
    pub const LEGACY_CANONICAL: Self = Self(1);
    pub const LEGACY_TEMPORAL: Self = Self(2);
    pub const FIRST_AUTHORED: Self = Self(3);

    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Serialize for NodeId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for NodeId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| de::Error::custom("visual node id must be non-zero"))
    }
}

/// Stable identity of a one-level group. The same type is used by live and
/// missing group-output references, preventing deletion from retargeting a tap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(u64);

impl GroupId {
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Serialize for GroupId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for GroupId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| de::Error::custom("group id must be non-zero"))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeBlend {
    #[default]
    Normal,
    Screen,
    Multiply,
    Difference,
    Add,
    Subtract,
    Darken,
    Lighten,
    Overlay,
    SoftLight,
    HardLight,
    Exclusion,
    Dodge,
    Burn,
    AlphaCut,
}

impl NodeBlend {
    pub const fn code(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Screen => 1,
            Self::Multiply => 2,
            Self::Difference => 3,
            Self::Add => 4,
            Self::Subtract => 5,
            Self::Darken => 6,
            Self::Lighten => 7,
            Self::Overlay => 8,
            Self::SoftLight => 9,
            Self::HardLight => 10,
            Self::Exclusion => 11,
            Self::Dodge => 12,
            Self::Burn => 13,
            Self::AlphaCut => 14,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeTiming {
    #[default]
    CurrentFrame,
    /// Exactly N-1. Longer or arbitrary-delay edges are intentionally absent.
    PreviousFrame,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum SavedImageSource {
    SelectedLayer {
        layer_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    /// Retained after a failed restore or explicit layer deletion. It never
    /// resolves against a newly inserted layer at the vacated position.
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    #[default]
    OneBelow,
    AllBelow,
    GroupOutput {
        group_id: GroupId,
    },
    /// Retained after explicit group deletion; never retargeted if another
    /// group later occupies the same root position.
    MissingGroupOutput {
        group_id: GroupId,
    },
    CleanProgram,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SavedImageTap {
    pub source: SavedImageSource,
    pub timing: EdgeTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedImageSource {
    SelectedLayer {
        layer_id: StableLayerId,
        /// Saved provenance retained only as a missing-source fallback. Live
        /// consumers route by `layer_id`; this position is never dereferenced.
        saved_position: SavedLayerPosition,
        stage: LayerImageStage,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        stage: LayerImageStage,
    },
    OneBelow,
    AllBelow,
    GroupOutput(GroupId),
    MissingGroupOutput(GroupId),
    CleanProgram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedImageTap {
    pub source: ResolvedImageSource,
    pub timing: EdgeTiming,
}

impl SavedImageTap {
    pub fn to_runtime(
        self,
        mut layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: impl Fn(GroupId) -> bool,
    ) -> ResolvedImageTap {
        let source = match self.source {
            SavedImageSource::SelectedLayer {
                layer_position,
                stage,
            } => layer_at_position(layer_position).map_or(
                ResolvedImageSource::MissingSelectedLayer {
                    saved_position: layer_position,
                    stage,
                },
                |layer_id| ResolvedImageSource::SelectedLayer {
                    layer_id,
                    saved_position: layer_position,
                    stage,
                },
            ),
            SavedImageSource::MissingSelectedLayer {
                saved_position,
                stage,
            } => ResolvedImageSource::MissingSelectedLayer {
                saved_position,
                stage,
            },
            SavedImageSource::OneBelow => ResolvedImageSource::OneBelow,
            SavedImageSource::AllBelow => ResolvedImageSource::AllBelow,
            SavedImageSource::GroupOutput { group_id } if group_exists(group_id) => {
                ResolvedImageSource::GroupOutput(group_id)
            }
            SavedImageSource::GroupOutput { group_id }
            | SavedImageSource::MissingGroupOutput { group_id } => {
                ResolvedImageSource::MissingGroupOutput(group_id)
            }
            SavedImageSource::CleanProgram => ResolvedImageSource::CleanProgram,
        };
        ResolvedImageTap {
            source,
            timing: self.timing,
        }
    }

    pub const fn referenced_group(self) -> Option<GroupId> {
        match self.source {
            SavedImageSource::GroupOutput { group_id }
            | SavedImageSource::MissingGroupOutput { group_id } => Some(group_id),
            _ => None,
        }
    }

    pub const fn selected_layer_position(self) -> Option<SavedLayerPosition> {
        match self.source {
            SavedImageSource::SelectedLayer { layer_position, .. } => Some(layer_position),
            _ => None,
        }
    }
}

impl ResolvedImageTap {
    /// Capture a live route without ever persisting a process identity. A
    /// deleted live donor becomes explicitly missing at its original saved
    /// provenance; it cannot retarget a newly inserted layer there.
    pub fn to_saved(
        self,
        mut position_of_layer: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> SavedImageTap {
        let source = match self.source {
            ResolvedImageSource::SelectedLayer {
                layer_id,
                saved_position,
                stage,
            } => position_of_layer(layer_id).map_or(
                SavedImageSource::MissingSelectedLayer {
                    saved_position,
                    stage,
                },
                |layer_position| SavedImageSource::SelectedLayer {
                    layer_position,
                    stage,
                },
            ),
            ResolvedImageSource::MissingSelectedLayer {
                saved_position,
                stage,
            } => SavedImageSource::MissingSelectedLayer {
                saved_position,
                stage,
            },
            ResolvedImageSource::OneBelow => SavedImageSource::OneBelow,
            ResolvedImageSource::AllBelow => SavedImageSource::AllBelow,
            ResolvedImageSource::GroupOutput(group_id) => {
                SavedImageSource::GroupOutput { group_id }
            }
            ResolvedImageSource::MissingGroupOutput(group_id) => {
                SavedImageSource::MissingGroupOutput { group_id }
            }
            ResolvedImageSource::CleanProgram => SavedImageSource::CleanProgram,
        };
        SavedImageTap {
            source,
            timing: self.timing,
        }
    }

    pub fn mark_layer_missing(&mut self, removed: StableLayerId) {
        if let ResolvedImageSource::SelectedLayer {
            layer_id,
            saved_position,
            stage,
        } = self.source
        {
            if layer_id == removed {
                self.source = ResolvedImageSource::MissingSelectedLayer {
                    saved_position,
                    stage,
                };
            }
        }
    }

    pub fn mark_group_missing(&mut self, removed: GroupId) {
        if self.source == ResolvedImageSource::GroupOutput(removed) {
            self.source = ResolvedImageSource::MissingGroupOutput(removed);
        }
    }

    pub const fn referenced_group(self) -> Option<GroupId> {
        match self.source {
            ResolvedImageSource::GroupOutput(group_id)
            | ResolvedImageSource::MissingGroupOutput(group_id) => Some(group_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatteChannel {
    #[default]
    Alpha,
    Luma,
    Red,
    Green,
    Blue,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageMatte {
    pub tap: SavedImageTap,
    pub channel: MatteChannel,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
}

impl Default for ImageMatte {
    fn default() -> Self {
        Self {
            tap: SavedImageTap::default(),
            channel: MatteChannel::Alpha,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 0.1,
        }
    }
}

impl ImageMatte {
    pub fn sanitized(self) -> Self {
        Self {
            tap: self.tap,
            channel: self.channel,
            invert: self.invert,
            amount: finite_clamp(self.amount, 1.0, 0.0, 1.0),
            threshold: finite_clamp(self.threshold, 0.5, 0.0, 1.0),
            softness: finite_clamp(self.softness, 0.1, 0.0, 0.5),
        }
    }

    /// Preserve the deleted identity explicitly instead of letting a future
    /// group at another root position inherit this route.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved matte invalidation supports patch/editor migrations"
        )
    )]
    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        if self.tap.source == (SavedImageSource::GroupOutput { group_id: removed }) {
            self.tap.source = SavedImageSource::MissingGroupOutput { group_id: removed };
        }
    }

    pub const fn selected_layer_position(self) -> Option<SavedLayerPosition> {
        self.tap.selected_layer_position()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DigitalColorParams {
    pub pixelate_size: f32,
    pub rgb_split: f32,
    pub downsample: f32,
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub posterize: f32,
    pub invert: f32,
    pub vignette: f32,
    pub color_drift: f32,
}

impl Default for DigitalColorParams {
    fn default() -> Self {
        Self {
            pixelate_size: 1.0,
            rgb_split: 0.0,
            downsample: 1.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: 0.0,
            vignette: 0.0,
            color_drift: 0.0,
        }
    }
}

impl DigitalColorParams {
    fn sanitized(self) -> Self {
        Self {
            pixelate_size: finite_clamp(self.pixelate_size, 1.0, 1.0, 32.0),
            rgb_split: finite_clamp(self.rgb_split, 0.0, 0.0, 30.0),
            downsample: finite_clamp(self.downsample, 1.0, 0.05, 1.0),
            hue_shift: wrap_degrees(self.hue_shift),
            saturation: finite_clamp(self.saturation, 0.0, -1.0, 1.0),
            brightness: finite_clamp(self.brightness, 0.0, -1.0, 1.0),
            contrast: finite_clamp(self.contrast, 0.0, -1.0, 1.0),
            posterize: finite_clamp(self.posterize, 0.0, 0.0, 16.0),
            invert: finite_clamp(self.invert, 0.0, 0.0, 1.0),
            vignette: finite_clamp(self.vignette, 0.0, 0.0, 1.5),
            color_drift: finite_clamp(self.color_drift, 0.0, 0.0, 0.02),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyMode {
    #[default]
    KeepBright,
    KeepDark,
    RemoveColor,
    KeepColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyParams {
    pub mode: KeyMode,
    pub threshold: f32,
    pub softness: f32,
    pub color: [f32; 3],
    pub tolerance: f32,
    pub invert: bool,
}

impl Default for KeyParams {
    fn default() -> Self {
        Self {
            mode: KeyMode::KeepBright,
            threshold: 0.5,
            softness: 0.1,
            color: [0.0, 1.0, 0.0],
            tolerance: 0.15,
            invert: false,
        }
    }
}

impl KeyParams {
    fn sanitized(self) -> Self {
        Self {
            mode: self.mode,
            threshold: finite_clamp(self.threshold, 0.5, 0.0, 1.0),
            softness: finite_clamp(self.softness, 0.1, 0.0, 0.5),
            color: self.color.map(|value| finite_clamp(value, 0.0, 0.0, 1.0)),
            tolerance: finite_clamp(self.tolerance, 0.15, 0.0, 1.0),
            invert: self.invert,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CellularParams {
    pub amount: f32,
    pub scale: f32,
    pub warp: f32,
    pub speed: f32,
    pub gap_amount: f32,
    pub gap_threshold: f32,
    pub gap_softness: f32,
    pub seed: u32,
}

impl Default for CellularParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            scale: 10.0,
            warp: 0.35,
            speed: 0.25,
            gap_amount: 0.0,
            gap_threshold: 0.65,
            gap_softness: 0.08,
            seed: 0,
        }
    }
}

impl CellularParams {
    fn sanitized(self) -> Self {
        Self {
            amount: finite_clamp(self.amount, 0.0, 0.0, 1.0),
            scale: finite_clamp(self.scale, 10.0, 2.0, 32.0),
            warp: finite_clamp(self.warp, 0.35, 0.0, 1.0),
            speed: finite_clamp(self.speed, 0.25, 0.0, 2.0),
            gap_amount: finite_clamp(self.gap_amount, 0.0, 0.0, 1.0),
            gap_threshold: finite_clamp(self.gap_threshold, 0.65, 0.0, 1.0),
            gap_softness: finite_clamp(self.gap_softness, 0.08, 0.0, 0.5),
            seed: self.seed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShiftParams {
    pub amount: f32,
    pub block_size: f32,
    pub density: f32,
    pub speed: f32,
    pub seed: u32,
}

impl Default for ShiftParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            block_size: 8.0,
            density: 0.5,
            speed: 3.0,
            seed: 0,
        }
    }
}

impl ShiftParams {
    fn sanitized(self) -> Self {
        Self {
            amount: finite_clamp(self.amount, 0.0, 0.0, 1.0),
            block_size: finite_clamp(self.block_size, 8.0, 2.0, 256.0),
            density: finite_clamp(self.density, 0.5, 0.0, 1.0),
            speed: finite_clamp(self.speed, 3.0, 0.0, 20.0),
            seed: self.seed,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrainAlgorithm {
    #[default]
    Gaussian,
    Perlin,
    SaltPepper,
    Blue,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GrainParams {
    pub intensity: f32,
    pub size: f32,
    pub algorithm: GrainAlgorithm,
    pub color: bool,
    pub seed: u32,
}

impl Default for GrainParams {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            size: 1.0,
            algorithm: GrainAlgorithm::Gaussian,
            color: false,
            seed: 0,
        }
    }
}

impl GrainParams {
    fn sanitized(self) -> Self {
        Self {
            intensity: finite_clamp(self.intensity, 0.0, 0.0, 0.3),
            size: finite_clamp(self.size, 1.0, 1.0, 4.0),
            algorithm: self.algorithm,
            color: self.color,
            seed: self.seed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RectangleMask {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub rotation_deg: f32,
    pub feather: f32,
    pub invert: bool,
}

impl Default for RectangleMask {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            size: [1.0, 1.0],
            rotation_deg: 0.0,
            feather: 0.0,
            invert: false,
        }
    }
}

impl RectangleMask {
    fn sanitized(self) -> Self {
        Self {
            center: self.center.map(|value| finite_clamp(value, 0.5, -2.0, 3.0)),
            size: self.size.map(|value| finite_clamp(value, 1.0, 0.0, 4.0)),
            rotation_deg: wrap_degrees(self.rotation_deg),
            feather: finite_clamp(self.feather, 0.0, 0.0, 1.0),
            invert: self.invert,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EllipseMask {
    pub center: [f32; 2],
    pub radii: [f32; 2],
    pub rotation_deg: f32,
    pub feather: f32,
    pub invert: bool,
}

impl Default for EllipseMask {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            radii: [0.5, 0.5],
            rotation_deg: 0.0,
            feather: 0.0,
            invert: false,
        }
    }
}

impl EllipseMask {
    fn sanitized(self) -> Self {
        Self {
            center: self.center.map(|value| finite_clamp(value, 0.5, -2.0, 3.0)),
            radii: self.radii.map(|value| finite_clamp(value, 0.5, 0.0, 2.0)),
            rotation_deg: wrap_degrees(self.rotation_deg),
            feather: finite_clamp(self.feather, 0.0, 0.0, 1.0),
            invert: self.invert,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mask", content = "params", rename_all = "snake_case")]
pub enum MaskParams {
    Rectangle(RectangleMask),
    Ellipse(EllipseMask),
    Image(ImageMatte),
}

impl Default for MaskParams {
    fn default() -> Self {
        Self::Rectangle(RectangleMask::default())
    }
}

impl MaskParams {
    fn sanitized(self) -> Self {
        match self {
            Self::Rectangle(value) => Self::Rectangle(value.sanitized()),
            Self::Ellipse(value) => Self::Ellipse(value.sanitized()),
            Self::Image(value) => Self::Image(value.sanitized()),
        }
    }

    pub const fn image_tap(self) -> Option<SavedImageTap> {
        match self {
            Self::Image(matte) => Some(matte.tap),
            Self::Rectangle(_) | Self::Ellipse(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "params", rename_all = "snake_case")]
pub enum VisualNodeKind {
    /// Frozen marker for the existing monolithic effects shader. Its values
    /// continue to come from the established effect/transform state.
    LegacyCanonical,
    /// Frozen marker for the established program-history temporal stage.
    LegacyTemporal,
    Transform(SpatialTransform),
    DigitalColor(DigitalColorParams),
    Key(KeyParams),
    Cellular(CellularParams),
    Shift(ShiftParams),
    Grain(GrainParams),
    Mask(MaskParams),
}

impl VisualNodeKind {
    pub const fn tag(self) -> NodeKindTag {
        match self {
            Self::LegacyCanonical => NodeKindTag::LegacyCanonical,
            Self::LegacyTemporal => NodeKindTag::LegacyTemporal,
            Self::Transform(_) => NodeKindTag::Transform,
            Self::DigitalColor(_) => NodeKindTag::DigitalColor,
            Self::Key(_) => NodeKindTag::Key,
            Self::Cellular(_) => NodeKindTag::Cellular,
            Self::Shift(_) => NodeKindTag::Shift,
            Self::Grain(_) => NodeKindTag::Grain,
            Self::Mask(_) => NodeKindTag::Mask,
        }
    }

    fn sanitized(self) -> Self {
        match self {
            Self::Transform(value) => Self::Transform(value.sanitized()),
            Self::DigitalColor(value) => Self::DigitalColor(value.sanitized()),
            Self::Key(value) => Self::Key(value.sanitized()),
            Self::Cellular(value) => Self::Cellular(value.sanitized()),
            Self::Shift(value) => Self::Shift(value.sanitized()),
            Self::Grain(value) => Self::Grain(value.sanitized()),
            Self::Mask(value) => Self::Mask(value.sanitized()),
            marker => marker,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct VisualNode {
    pub stable_id: NodeId,
    pub enabled: bool,
    pub wet: f32,
    pub blend: NodeBlend,
    pub kind: VisualNodeKind,
}

impl VisualNode {
    pub fn authored(stable_id: NodeId, kind: VisualNodeKind) -> Self {
        Self {
            stable_id,
            enabled: true,
            wet: 1.0,
            blend: NodeBlend::Normal,
            kind: kind.sanitized(),
        }
    }

    fn sanitized(self) -> Self {
        Self {
            stable_id: self.stable_id,
            enabled: self.enabled,
            wet: finite_clamp(self.wet, 1.0, 0.0, 1.0),
            blend: self.blend,
            kind: self.kind.sanitized(),
        }
    }

    fn is_exact_legacy_marker(self, expected: NodeKindTag) -> bool {
        self.enabled
            && self.wet == 1.0
            && self.blend == NodeBlend::Normal
            && self.kind.tag() == expected
    }
}

impl<'de> Deserialize<'de> for VisualNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            stable_id: NodeId,
            #[serde(default = "default_true")]
            enabled: bool,
            #[serde(default = "default_one")]
            wet: f32,
            #[serde(default)]
            blend: NodeBlend,
            kind: VisualNodeKind,
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            stable_id: raw.stable_id,
            enabled: raw.enabled,
            wet: raw.wet,
            blend: raw.blend,
            kind: raw.kind,
        }
        .sanitized())
    }
}

const fn default_true() -> bool {
    true
}

const fn default_one() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKindTag {
    LegacyCanonical,
    LegacyTemporal,
    Transform,
    DigitalColor,
    Key,
    Cellular,
    Shift,
    Grain,
    Mask,
}

impl NodeKindTag {
    const fn signature_code(self) -> u8 {
        match self {
            Self::LegacyCanonical => 1,
            Self::LegacyTemporal => 2,
            Self::Transform => 3,
            Self::DigitalColor => 4,
            Self::Key => 5,
            Self::Cellular => 6,
            Self::Shift => 7,
            Self::Grain => 8,
            Self::Mask => 9,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeResourceBudget {
    pub full_frame_passes: u8,
    /// Pre-M6 logical lookup units, retained separately so the expanded
    /// shader-operation ceiling cannot admit more historical sample work.
    pub logical_texture_lookups_per_pixel: u8,
    /// Actual shader texture instructions after a premultiplied bilinear
    /// lookup expands into four explicit `textureLoad` operations.
    pub texture_samples_per_pixel: u8,
    pub sampled_textures_in_pass: u8,
    pub cross_input_taps: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeKindDescriptor {
    pub tag: NodeKindTag,
    pub key: &'static str,
    pub title: &'static str,
    pub budget: NodeResourceBudget,
}

pub const NODE_KIND_DESCRIPTORS: [NodeKindDescriptor; 9] = [
    NodeKindDescriptor {
        tag: NodeKindTag::LegacyCanonical,
        key: "legacy_canonical",
        title: "Legacy Canonical",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Frozen conservative LegacyCanonical accounting remains twelve
            // logical units; it is not multiplied merely because Advanced
            // filters elsewhere expand into explicit loads.
            logical_texture_lookups_per_pixel: 12,
            texture_samples_per_pixel: 12,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::LegacyTemporal,
        key: "legacy_temporal",
        title: "Legacy Temporal",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Advanced worst case: current + slit history + two Loom ages +
            // temporal key are one operation each, while transformed feedback
            // is four explicit premultiplied loads. LegacyExact retains its
            // lower-cost mode-0 shader branch under this conservative bound.
            logical_texture_lookups_per_pixel: 6,
            texture_samples_per_pixel: 9,
            sampled_textures_in_pass: 3,
            cross_input_taps: 1,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Transform,
        key: "transform",
        title: "Transform",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Dry plus transformed, each a four-load premultiplied bilinear.
            logical_texture_lookups_per_pixel: 2,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 2,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::DigitalColor,
        key: "digital_color",
        title: "Digital / Color",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Dry plus center/red/blue processed lookups at the worst-case
            // color-drift or RGB-split branch.
            logical_texture_lookups_per_pixel: 4,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 4,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Key,
        key: "key",
        title: "Key",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            logical_texture_lookups_per_pixel: 1,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Cellular,
        key: "cellular",
        title: "Cellular",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Dry plus the displaced processed lookup.
            logical_texture_lookups_per_pixel: 2,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 2,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Shift,
        key: "shift",
        title: "Shift",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Dry plus the shifted processed lookup.
            logical_texture_lookups_per_pixel: 2,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 2,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Grain,
        key: "grain",
        title: "Grain",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            logical_texture_lookups_per_pixel: 1,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS,
            sampled_textures_in_pass: 1,
            cross_input_taps: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Mask,
        key: "mask",
        title: "Mask",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Four source loads plus one donor sampling instruction. Shape
            // masks use only the source portion; the descriptor is worst-case.
            logical_texture_lookups_per_pixel: 2,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS + 1,
            sampled_textures_in_pass: 2,
            cross_input_taps: 1,
        },
    },
];

pub fn node_kind_descriptor(tag: NodeKindTag) -> &'static NodeKindDescriptor {
    NODE_KIND_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.tag == tag)
        .expect("descriptor registry covers every NodeKindTag")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeParamType {
    Bool,
    Float,
    Unsigned,
    Enum,
    Vec2,
    Color,
    ImageTap,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeParamDescriptor {
    pub kind: NodeKindTag,
    pub key: &'static str,
    pub value_type: NodeParamType,
    pub range: Option<[f32; 2]>,
    pub default: Option<f32>,
    pub dice_eligible: bool,
    pub modulatable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "control metadata is exposed for descriptor contract tests and UI adapters"
    )
)]
pub struct NodeControlDescriptor {
    pub key: &'static str,
    pub value_type: NodeParamType,
    pub range: Option<[f32; 2]>,
    pub default: Option<f32>,
    pub dice_eligible: bool,
    pub modulatable: bool,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "control metadata is exposed for descriptor contract tests and UI adapters"
    )
)]
pub const NODE_CONTROL_DESCRIPTORS: &[NodeControlDescriptor] = &[
    NodeControlDescriptor {
        key: "enabled",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeControlDescriptor {
        key: "wet",
        value_type: NodeParamType::Float,
        range: Some([0.0, 1.0]),
        default: Some(1.0),
        dice_eligible: true,
        modulatable: true,
    },
    NodeControlDescriptor {
        key: "blend",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
];

macro_rules! float_param {
    ($kind:ident, $key:literal, $min:expr, $max:expr, $default:expr) => {
        NodeParamDescriptor {
            kind: NodeKindTag::$kind,
            key: $key,
            value_type: NodeParamType::Float,
            range: Some([$min, $max]),
            default: Some($default),
            dice_eligible: true,
            modulatable: true,
        }
    };
}

/// Authoritative parameter paths used by patch, Morph, modulation, Dice,
/// procedural generation, and browser consumers. Structural node controls
/// (`enabled`, `wet`, and `blend`) are described by
/// [`NODE_CONTROL_DESCRIPTORS`]. Discrete and route fields are intentionally
/// present even when they are neither modulatable nor Dice-eligible.
pub const NODE_PARAM_DESCRIPTORS: &[NodeParamDescriptor] = &[
    NodeParamDescriptor {
        kind: NodeKindTag::Transform,
        key: "position",
        value_type: NodeParamType::Vec2,
        range: Some([-4.0, 4.0]),
        default: Some(0.0),
        dice_eligible: true,
        modulatable: true,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Transform,
        key: "scale",
        value_type: NodeParamType::Vec2,
        range: Some([-16.0, 16.0]),
        default: Some(1.0),
        dice_eligible: true,
        modulatable: true,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Transform,
        key: "anchor",
        value_type: NodeParamType::Vec2,
        range: Some([-2.0, 3.0]),
        default: Some(0.5),
        dice_eligible: false,
        modulatable: true,
    },
    float_param!(Transform, "rotation_deg", -180.0, 180.0, 0.0),
    float_param!(Transform, "skew_deg", -89.0, 89.0, 0.0),
    float_param!(Transform, "skew_axis_deg", -180.0, 180.0, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Transform,
        key: "fit_mode",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Transform, "crop_left", 0.0, 1.0, 0.0),
    float_param!(Transform, "crop_top", 0.0, 1.0, 0.0),
    float_param!(Transform, "crop_right", 0.0, 1.0, 0.0),
    float_param!(Transform, "crop_bottom", 0.0, 1.0, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Transform,
        key: "edge_mode",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Transform,
        key: "sampling",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(DigitalColor, "pixelate_size", 1.0, 32.0, 1.0),
    float_param!(DigitalColor, "rgb_split", 0.0, 30.0, 0.0),
    float_param!(DigitalColor, "downsample", 0.05, 1.0, 1.0),
    float_param!(DigitalColor, "hue_shift", -180.0, 180.0, 0.0),
    float_param!(DigitalColor, "saturation", -1.0, 1.0, 0.0),
    float_param!(DigitalColor, "brightness", -1.0, 1.0, 0.0),
    float_param!(DigitalColor, "contrast", -1.0, 1.0, 0.0),
    float_param!(DigitalColor, "posterize", 0.0, 16.0, 0.0),
    float_param!(DigitalColor, "invert", 0.0, 1.0, 0.0),
    float_param!(DigitalColor, "vignette", 0.0, 1.5, 0.0),
    float_param!(DigitalColor, "color_drift", 0.0, 0.02, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Key,
        key: "mode",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Key, "threshold", 0.0, 1.0, 0.5),
    float_param!(Key, "softness", 0.0, 0.5, 0.1),
    NodeParamDescriptor {
        kind: NodeKindTag::Key,
        key: "color",
        value_type: NodeParamType::Color,
        range: Some([0.0, 1.0]),
        default: None,
        dice_eligible: true,
        modulatable: true,
    },
    float_param!(Key, "tolerance", 0.0, 1.0, 0.15),
    NodeParamDescriptor {
        kind: NodeKindTag::Key,
        key: "invert",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Cellular, "amount", 0.0, 1.0, 0.0),
    float_param!(Cellular, "scale", 2.0, 32.0, 10.0),
    float_param!(Cellular, "warp", 0.0, 1.0, 0.35),
    float_param!(Cellular, "speed", 0.0, 2.0, 0.25),
    float_param!(Cellular, "gap_amount", 0.0, 1.0, 0.0),
    float_param!(Cellular, "gap_threshold", 0.0, 1.0, 0.65),
    float_param!(Cellular, "gap_softness", 0.0, 0.5, 0.08),
    NodeParamDescriptor {
        kind: NodeKindTag::Cellular,
        key: "seed",
        value_type: NodeParamType::Unsigned,
        range: None,
        default: Some(0.0),
        dice_eligible: true,
        modulatable: false,
    },
    float_param!(Shift, "amount", 0.0, 1.0, 0.0),
    float_param!(Shift, "block_size", 2.0, 256.0, 8.0),
    float_param!(Shift, "density", 0.0, 1.0, 0.5),
    float_param!(Shift, "speed", 0.0, 20.0, 3.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Shift,
        key: "seed",
        value_type: NodeParamType::Unsigned,
        range: None,
        default: Some(0.0),
        dice_eligible: true,
        modulatable: false,
    },
    float_param!(Grain, "intensity", 0.0, 0.3, 0.0),
    float_param!(Grain, "size", 1.0, 4.0, 1.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Grain,
        key: "algorithm",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Grain,
        key: "color",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Grain,
        key: "seed",
        value_type: NodeParamType::Unsigned,
        range: None,
        default: Some(0.0),
        dice_eligible: true,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "variant",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "rectangle_center",
        value_type: NodeParamType::Vec2,
        range: Some([-2.0, 3.0]),
        default: Some(0.5),
        dice_eligible: true,
        modulatable: true,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "rectangle_size",
        value_type: NodeParamType::Vec2,
        range: Some([0.0, 4.0]),
        default: Some(1.0),
        dice_eligible: true,
        modulatable: true,
    },
    float_param!(Mask, "rectangle_rotation_deg", -180.0, 180.0, 0.0),
    float_param!(Mask, "rectangle_feather", 0.0, 1.0, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "rectangle_invert",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "ellipse_center",
        value_type: NodeParamType::Vec2,
        range: Some([-2.0, 3.0]),
        default: Some(0.5),
        dice_eligible: true,
        modulatable: true,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "ellipse_radii",
        value_type: NodeParamType::Vec2,
        range: Some([0.0, 2.0]),
        default: Some(0.5),
        dice_eligible: true,
        modulatable: true,
    },
    float_param!(Mask, "ellipse_rotation_deg", -180.0, 180.0, 0.0),
    float_param!(Mask, "ellipse_feather", 0.0, 1.0, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "ellipse_invert",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "image_tap",
        value_type: NodeParamType::ImageTap,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "image_channel",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Mask,
        key: "image_invert",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Mask, "image_amount", 0.0, 1.0, 1.0),
    float_param!(Mask, "image_threshold", 0.0, 1.0, 0.5),
    float_param!(Mask, "image_softness", 0.0, 0.5, 0.1),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyRackScope {
    Layer,
    Master,
    Group,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

const fn signature_step(mut hash: u64, value: u64) -> u64 {
    let bytes = value.to_le_bytes();
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        index += 1;
    }
    hash
}

const fn legacy_signature(master: bool) -> u64 {
    let mut hash = signature_step(FNV_OFFSET, NodeId::LEGACY_CANONICAL.get());
    hash = signature_step(hash, NodeKindTag::LegacyCanonical.signature_code() as u64);
    if master {
        hash = signature_step(hash, NodeId::LEGACY_TEMPORAL.get());
        hash = signature_step(hash, NodeKindTag::LegacyTemporal.signature_code() as u64);
    }
    hash
}

pub const LEGACY_LAYER_RACK_SIGNATURE: u64 = legacy_signature(false);
pub const LEGACY_MASTER_RACK_SIGNATURE: u64 = legacy_signature(true);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RackError {
    TooManyNodes { limit: usize },
    DuplicateNodeId(NodeId),
    ReservedNodeId { id: NodeId, expected: NodeKindTag },
    LegacyMarkerId { tag: NodeKindTag, expected: NodeId },
    MutableLegacyMarker(NodeId),
    LegacyTemporalOnLayer,
    LegacyMarkerOnGroup(NodeKindTag),
    InvalidNextNodeId { next: u64, greatest_observed: u64 },
    NodeIdExhausted,
    UnknownNode(NodeId),
    InvalidMoveIndex { index: usize, len: usize },
    ResourceOverflow,
    RackLogicalLookupBudget { lookups: u32, limit: u32 },
    RackSampleBudget { samples: u32, limit: u32 },
    PassTextureBudget { textures: u32, limit: u32 },
}

impl fmt::Display for RackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyNodes { limit } => {
                write!(formatter, "a visual rack may contain at most {limit} nodes")
            }
            Self::DuplicateNodeId(id) => write!(formatter, "duplicate visual node id {}", id.get()),
            Self::ReservedNodeId { id, expected } => write!(
                formatter,
                "reserved visual node id {} must identify {expected:?}",
                id.get()
            ),
            Self::LegacyMarkerId { tag, expected } => write!(
                formatter,
                "{tag:?} must use reserved visual node id {}",
                expected.get()
            ),
            Self::MutableLegacyMarker(id) => write!(
                formatter,
                "legacy marker {} must remain enabled, fully wet, and Normal",
                id.get()
            ),
            Self::LegacyTemporalOnLayer => {
                formatter.write_str("the legacy temporal marker is valid only in the master rack")
            }
            Self::LegacyMarkerOnGroup(tag) => {
                write!(formatter, "{tag:?} is not valid in a group rack")
            }
            Self::InvalidNextNodeId {
                next,
                greatest_observed,
            } => write!(
                formatter,
                "next visual node id {next} must advance past observed id {greatest_observed}"
            ),
            Self::NodeIdExhausted => formatter.write_str("visual node identity space is exhausted"),
            Self::UnknownNode(id) => write!(formatter, "visual node {} does not exist", id.get()),
            Self::InvalidMoveIndex { index, len } => write!(
                formatter,
                "node move index {index} exceeds rack length {len}"
            ),
            Self::ResourceOverflow => {
                formatter.write_str("visual rack resource arithmetic overflowed")
            }
            Self::RackLogicalLookupBudget { lookups, limit } => write!(
                formatter,
                "rack requests {lookups} logical texture lookups per pixel; limit is {limit}"
            ),
            Self::RackSampleBudget { samples, limit } => write!(
                formatter,
                "rack requests {samples} texture samples per pixel; limit is {limit}"
            ),
            Self::PassTextureBudget { textures, limit } => write!(
                formatter,
                "rack pass requests {textures} sampled textures; limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for RackError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RackResourceBudget {
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_samples_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
    pub cross_input_taps: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualRack {
    nodes: Vec<VisualNode>,
    /// Zero means exhausted; otherwise this is strictly greater than every ID
    /// ever issued by this rack, including deleted nodes.
    next_node_id: u64,
}

impl VisualRack {
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            next_node_id: NodeId::FIRST_AUTHORED.get(),
        }
    }

    pub fn synthetic_legacy(scope: LegacyRackScope) -> Self {
        if scope == LegacyRackScope::Group {
            return Self::empty();
        }
        let mut nodes = vec![VisualNode::authored(
            NodeId::LEGACY_CANONICAL,
            VisualNodeKind::LegacyCanonical,
        )];
        if scope == LegacyRackScope::Master {
            nodes.push(VisualNode::authored(
                NodeId::LEGACY_TEMPORAL,
                VisualNodeKind::LegacyTemporal,
            ));
        }
        Self {
            nodes,
            next_node_id: NodeId::FIRST_AUTHORED.get(),
        }
    }

    pub fn try_from_parts(
        nodes: Vec<VisualNode>,
        next_node_id: Option<u64>,
    ) -> Result<Self, RackError> {
        if nodes.len() > MAX_NODES_PER_RACK {
            return Err(RackError::TooManyNodes {
                limit: MAX_NODES_PER_RACK,
            });
        }
        let mut ids = BTreeSet::new();
        let mut greatest = NodeId::LEGACY_TEMPORAL.get();
        for node in &nodes {
            if !ids.insert(node.stable_id) {
                return Err(RackError::DuplicateNodeId(node.stable_id));
            }
            greatest = greatest.max(node.stable_id.get());
            validate_legacy_marker(*node)?;
        }
        let inferred = greatest.checked_add(1).unwrap_or(0);
        let next = next_node_id.unwrap_or(inferred);
        if next != 0 && next <= greatest || next == 0 && greatest != u64::MAX {
            return Err(RackError::InvalidNextNodeId {
                next,
                greatest_observed: greatest,
            });
        }
        let rack = Self {
            nodes: nodes.into_iter().map(VisualNode::sanitized).collect(),
            next_node_id: next,
        };
        rack.resource_budget()?;
        Ok(rack)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack inspection is exercised by persistence tests"
        )
    )]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &VisualNode> {
        self.nodes.iter()
    }

    pub fn get(&self, id: NodeId) -> Option<&VisualNode> {
        self.nodes.iter().find(|node| node.stable_id == id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut VisualNode> {
        self.nodes.iter_mut().find(|node| node.stable_id == id)
    }

    pub const fn next_node_id_raw(&self) -> u64 {
        self.next_node_id
    }

    /// Advance this rack's allocator past an identity retained outside its
    /// live node vector (for example by Morph or a missing modulation target).
    /// `0` remains the single exhausted cursor: observing `u64::MAX` can never
    /// wrap around and make an old identity reusable.
    pub(crate) fn observe_node_reference(&mut self, id: NodeId) {
        if self.next_node_id != 0 && id.get() >= self.next_node_id {
            self.next_node_id = id.get().checked_add(1).unwrap_or(0);
        }
    }

    pub fn topology_signature(&self) -> u64 {
        self.nodes.iter().fold(FNV_OFFSET, |hash, node| {
            let hash = signature_step(hash, node.stable_id.get());
            signature_step(hash, u64::from(node.kind.tag().signature_code()))
        })
    }

    pub fn is_exact_legacy(&self, scope: LegacyRackScope) -> bool {
        let synthetic = Self::synthetic_legacy(scope);
        self.nodes == synthetic.nodes
            && self.topology_signature()
                == match scope {
                    LegacyRackScope::Layer => LEGACY_LAYER_RACK_SIGNATURE,
                    LegacyRackScope::Master => LEGACY_MASTER_RACK_SIGNATURE,
                    LegacyRackScope::Group => FNV_OFFSET,
                }
    }

    pub fn validate_for_scope(&self, scope: LegacyRackScope) -> Result<(), RackError> {
        let canonical_index = self
            .nodes
            .iter()
            .position(|node| node.kind.tag() == NodeKindTag::LegacyCanonical);
        let temporal_index = self
            .nodes
            .iter()
            .position(|node| node.kind.tag() == NodeKindTag::LegacyTemporal);
        if scope == LegacyRackScope::Group {
            if canonical_index.is_some() {
                return Err(RackError::LegacyMarkerOnGroup(NodeKindTag::LegacyCanonical));
            }
            if temporal_index.is_some() {
                return Err(RackError::LegacyMarkerOnGroup(NodeKindTag::LegacyTemporal));
            }
            return self.resource_budget().map(|_| ());
        }
        if scope == LegacyRackScope::Layer && temporal_index.is_some() {
            return Err(RackError::LegacyTemporalOnLayer);
        }
        // Explicit M2 racks may place immutable host-boundary markers at any
        // authored position. Exact omitted-patch migration remains frozen by
        // `synthetic_legacy`/`is_exact_legacy`; ordering is a planner concern.
        self.resource_budget().map(|_| ())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack mutation is retained for patch/editor tooling"
        )
    )]
    pub fn insert(&mut self, index: usize, kind: VisualNodeKind) -> Result<NodeId, RackError> {
        if self.nodes.len() == MAX_NODES_PER_RACK {
            return Err(RackError::TooManyNodes {
                limit: MAX_NODES_PER_RACK,
            });
        }
        if index > self.nodes.len() {
            return Err(RackError::InvalidMoveIndex {
                index,
                len: self.nodes.len(),
            });
        }
        if matches!(
            kind,
            VisualNodeKind::LegacyCanonical | VisualNodeKind::LegacyTemporal
        ) {
            return Err(RackError::LegacyMarkerId {
                tag: kind.tag(),
                expected: if matches!(kind, VisualNodeKind::LegacyCanonical) {
                    NodeId::LEGACY_CANONICAL
                } else {
                    NodeId::LEGACY_TEMPORAL
                },
            });
        }
        let id = NodeId::new(self.next_node_id).ok_or(RackError::NodeIdExhausted)?;
        self.next_node_id = self.next_node_id.checked_add(1).unwrap_or(0);
        self.nodes.insert(index, VisualNode::authored(id, kind));
        if let Err(error) = self.resource_budget() {
            self.nodes.remove(index);
            // The consumed identity intentionally remains retired.
            return Err(error);
        }
        Ok(id)
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack mutation is retained for patch/editor tooling"
        )
    )]
    pub fn push(&mut self, kind: VisualNodeKind) -> Result<NodeId, RackError> {
        self.insert(self.nodes.len(), kind)
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack mutation is retained for patch/editor tooling"
        )
    )]
    pub fn remove(&mut self, id: NodeId) -> Option<VisualNode> {
        if id == NodeId::LEGACY_CANONICAL || id == NodeId::LEGACY_TEMPORAL {
            return None;
        }
        let index = self.nodes.iter().position(|node| node.stable_id == id)?;
        Some(self.nodes.remove(index))
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack reorder is retained for patch/editor tooling"
        )
    )]
    pub fn move_node(
        &mut self,
        id: NodeId,
        new_index: usize,
        scope: LegacyRackScope,
    ) -> Result<(), RackError> {
        if new_index >= self.nodes.len() {
            return Err(RackError::InvalidMoveIndex {
                index: new_index,
                len: self.nodes.len(),
            });
        }
        let old_index = self
            .nodes
            .iter()
            .position(|node| node.stable_id == id)
            .ok_or(RackError::UnknownNode(id))?;
        let node = self.nodes.remove(old_index);
        self.nodes.insert(new_index, node);
        if let Err(error) = self.validate_for_scope(scope) {
            let node = self.nodes.remove(new_index);
            self.nodes.insert(old_index, node);
            return Err(error);
        }
        Ok(())
    }

    pub fn resource_budget(&self) -> Result<RackResourceBudget, RackError> {
        let mut result = RackResourceBudget::default();
        for node in self.nodes.iter().filter(|node| node.enabled) {
            let budget = node_kind_descriptor(node.kind.tag()).budget;
            result.full_frame_passes = result
                .full_frame_passes
                .checked_add(u32::from(budget.full_frame_passes))
                .ok_or(RackError::ResourceOverflow)?;
            result.logical_texture_lookups_per_pixel = result
                .logical_texture_lookups_per_pixel
                .checked_add(u32::from(budget.logical_texture_lookups_per_pixel))
                .ok_or(RackError::ResourceOverflow)?;
            result.texture_samples_per_pixel = result
                .texture_samples_per_pixel
                .checked_add(u32::from(budget.texture_samples_per_pixel))
                .ok_or(RackError::ResourceOverflow)?;
            result.max_sampled_textures_in_pass = result
                .max_sampled_textures_in_pass
                .max(u32::from(budget.sampled_textures_in_pass));
            result.cross_input_taps = result
                .cross_input_taps
                .checked_add(u32::from(budget.cross_input_taps))
                .ok_or(RackError::ResourceOverflow)?;
        }
        if result.logical_texture_lookups_per_pixel > MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK {
            return Err(RackError::RackLogicalLookupBudget {
                lookups: result.logical_texture_lookups_per_pixel,
                limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK,
            });
        }
        if result.texture_samples_per_pixel > MAX_TEXTURE_SAMPLES_PER_RACK {
            return Err(RackError::RackSampleBudget {
                samples: result.texture_samples_per_pixel,
                limit: MAX_TEXTURE_SAMPLES_PER_RACK,
            });
        }
        if result.max_sampled_textures_in_pass > MAX_SAMPLED_TEXTURES_PER_PASS {
            return Err(RackError::PassTextureBudget {
                textures: result.max_sampled_textures_in_pass,
                limit: MAX_SAMPLED_TEXTURES_PER_PASS,
            });
        }
        Ok(result)
    }

    pub fn referenced_group_ids(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.nodes.iter().filter_map(|node| match node.kind {
            VisualNodeKind::Mask(mask) => mask.image_tap()?.referenced_group(),
            _ => None,
        })
    }

    pub fn selected_layer_positions(&self) -> impl Iterator<Item = SavedLayerPosition> + '_ {
        self.nodes.iter().filter_map(|node| match node.kind {
            VisualNodeKind::Mask(MaskParams::Image(matte)) => matte.selected_layer_position(),
            _ => None,
        })
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack invalidation supports patch/editor migrations"
        )
    )]
    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        for node in &mut self.nodes {
            let VisualNodeKind::Mask(MaskParams::Image(matte)) = &mut node.kind else {
                continue;
            };
            matte.mark_group_output_missing(removed);
        }
    }
}

fn validate_legacy_marker(node: VisualNode) -> Result<(), RackError> {
    match node.stable_id {
        NodeId::LEGACY_CANONICAL if node.kind.tag() != NodeKindTag::LegacyCanonical => {
            return Err(RackError::ReservedNodeId {
                id: node.stable_id,
                expected: NodeKindTag::LegacyCanonical,
            });
        }
        NodeId::LEGACY_TEMPORAL if node.kind.tag() != NodeKindTag::LegacyTemporal => {
            return Err(RackError::ReservedNodeId {
                id: node.stable_id,
                expected: NodeKindTag::LegacyTemporal,
            });
        }
        _ => {}
    }
    let expected = match node.kind.tag() {
        NodeKindTag::LegacyCanonical => Some(NodeId::LEGACY_CANONICAL),
        NodeKindTag::LegacyTemporal => Some(NodeId::LEGACY_TEMPORAL),
        _ => None,
    };
    if let Some(expected) = expected {
        if node.stable_id != expected {
            return Err(RackError::LegacyMarkerId {
                tag: node.kind.tag(),
                expected,
            });
        }
        if !node.is_exact_legacy_marker(node.kind.tag()) {
            return Err(RackError::MutableLegacyMarker(node.stable_id));
        }
    }
    Ok(())
}

impl Default for VisualRack {
    fn default() -> Self {
        Self::empty()
    }
}

impl Serialize for VisualRack {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("VisualRack", 2)?;
        state.serialize_field("nodes", &NodeSequence(&self.nodes))?;
        state.serialize_field("next_node_id", &self.next_node_id)?;
        state.end()
    }
}

struct NodeSequence<'a>(&'a [VisualNode]);

impl Serialize for NodeSequence<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for node in self.0 {
            sequence.serialize_element(node)?;
        }
        sequence.end()
    }
}

struct BoundedNodes(Vec<VisualNode>);

impl<'de> Deserialize<'de> for BoundedNodes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NodesVisitor;

        impl<'de> Visitor<'de> for NodesVisitor {
            type Value = BoundedNodes;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {MAX_NODES_PER_RACK} visual nodes")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut nodes =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_NODES_PER_RACK));
                while let Some(node) = sequence.next_element::<VisualNode>()? {
                    if nodes.len() == MAX_NODES_PER_RACK {
                        return Err(de::Error::custom(format_args!(
                            "a visual rack may contain at most {MAX_NODES_PER_RACK} nodes"
                        )));
                    }
                    nodes.push(node);
                }
                Ok(BoundedNodes(nodes))
            }
        }

        deserializer.deserialize_seq(NodesVisitor)
    }
}

impl<'de> Deserialize<'de> for VisualRack {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default = "empty_bounded_nodes")]
            nodes: BoundedNodes,
            #[serde(default)]
            next_node_id: Option<u64>,
        }

        fn empty_bounded_nodes() -> BoundedNodes {
            BoundedNodes(Vec::new())
        }

        let raw = Raw::deserialize(deserializer)?;
        VisualRack::try_from_parts(raw.nodes.0, raw.next_node_id).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeImageMatte {
    pub tap: ResolvedImageTap,
    pub channel: MatteChannel,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
}

impl RuntimeImageMatte {
    pub fn resolve_routes(
        saved: ImageMatte,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: &impl Fn(GroupId) -> bool,
    ) -> Self {
        let saved = saved.sanitized();
        Self {
            tap: saved.tap.to_runtime(layer_at_position, group_exists),
            channel: saved.channel,
            invert: saved.invert,
            amount: saved.amount,
            threshold: saved.threshold,
            softness: saved.softness,
        }
    }

    pub fn capture_routes(
        self,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> ImageMatte {
        ImageMatte {
            tap: self.tap.to_saved(position_of_layer),
            channel: self.channel,
            invert: self.invert,
            amount: self.amount,
            threshold: self.threshold,
            softness: self.softness,
        }
        .sanitized()
    }

    pub(crate) fn sanitized(self) -> Self {
        Self {
            tap: self.tap,
            channel: self.channel,
            invert: self.invert,
            amount: finite_clamp(self.amount, 1.0, 0.0, 1.0),
            threshold: finite_clamp(self.threshold, 0.5, 0.0, 1.0),
            softness: finite_clamp(self.softness, 0.1, 0.0, 0.5),
        }
    }

    pub const fn selected_layer_id(self) -> Option<StableLayerId> {
        match self.tap.source {
            ResolvedImageSource::SelectedLayer { layer_id, .. } => Some(layer_id),
            _ => None,
        }
    }

    pub const fn referenced_group(self) -> Option<GroupId> {
        self.tap.referenced_group()
    }

    pub fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        self.tap.mark_layer_missing(removed);
    }

    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        self.tap.mark_group_missing(removed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RuntimeMaskParams {
    Rectangle(RectangleMask),
    Ellipse(EllipseMask),
    Image(RuntimeImageMatte),
}

impl RuntimeMaskParams {
    fn resolve(
        saved: MaskParams,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: &impl Fn(GroupId) -> bool,
    ) -> Self {
        match saved.sanitized() {
            MaskParams::Rectangle(value) => Self::Rectangle(value),
            MaskParams::Ellipse(value) => Self::Ellipse(value),
            MaskParams::Image(value) => Self::Image(RuntimeImageMatte::resolve_routes(
                value,
                layer_at_position,
                group_exists,
            )),
        }
    }

    fn capture(
        self,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> MaskParams {
        match self {
            Self::Rectangle(value) => MaskParams::Rectangle(value.sanitized()),
            Self::Ellipse(value) => MaskParams::Ellipse(value.sanitized()),
            Self::Image(value) => MaskParams::Image(value.capture_routes(position_of_layer)),
        }
    }

    fn sanitized(self) -> Self {
        match self {
            Self::Rectangle(value) => Self::Rectangle(value.sanitized()),
            Self::Ellipse(value) => Self::Ellipse(value.sanitized()),
            Self::Image(value) => Self::Image(value.sanitized()),
        }
    }

    pub const fn image_tap(self) -> Option<ResolvedImageTap> {
        match self {
            Self::Image(value) => Some(value.tap),
            Self::Rectangle(_) | Self::Ellipse(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RuntimeVisualNodeKind {
    LegacyCanonical,
    LegacyTemporal,
    Transform(SpatialTransform),
    DigitalColor(DigitalColorParams),
    Key(KeyParams),
    Cellular(CellularParams),
    Shift(ShiftParams),
    Grain(GrainParams),
    Mask(RuntimeMaskParams),
}

impl RuntimeVisualNodeKind {
    pub const fn tag(self) -> NodeKindTag {
        match self {
            Self::LegacyCanonical => NodeKindTag::LegacyCanonical,
            Self::LegacyTemporal => NodeKindTag::LegacyTemporal,
            Self::Transform(_) => NodeKindTag::Transform,
            Self::DigitalColor(_) => NodeKindTag::DigitalColor,
            Self::Key(_) => NodeKindTag::Key,
            Self::Cellular(_) => NodeKindTag::Cellular,
            Self::Shift(_) => NodeKindTag::Shift,
            Self::Grain(_) => NodeKindTag::Grain,
            Self::Mask(_) => NodeKindTag::Mask,
        }
    }

    fn resolve(
        saved: VisualNodeKind,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: &impl Fn(GroupId) -> bool,
    ) -> Self {
        match saved.sanitized() {
            VisualNodeKind::LegacyCanonical => Self::LegacyCanonical,
            VisualNodeKind::LegacyTemporal => Self::LegacyTemporal,
            VisualNodeKind::Transform(value) => Self::Transform(value),
            VisualNodeKind::DigitalColor(value) => Self::DigitalColor(value),
            VisualNodeKind::Key(value) => Self::Key(value),
            VisualNodeKind::Cellular(value) => Self::Cellular(value),
            VisualNodeKind::Shift(value) => Self::Shift(value),
            VisualNodeKind::Grain(value) => Self::Grain(value),
            VisualNodeKind::Mask(value) => Self::Mask(RuntimeMaskParams::resolve(
                value,
                layer_at_position,
                group_exists,
            )),
        }
    }

    fn capture(
        self,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> VisualNodeKind {
        match self {
            Self::LegacyCanonical => VisualNodeKind::LegacyCanonical,
            Self::LegacyTemporal => VisualNodeKind::LegacyTemporal,
            Self::Transform(value) => VisualNodeKind::Transform(value.sanitized()),
            Self::DigitalColor(value) => VisualNodeKind::DigitalColor(value.sanitized()),
            Self::Key(value) => VisualNodeKind::Key(value.sanitized()),
            Self::Cellular(value) => VisualNodeKind::Cellular(value.sanitized()),
            Self::Shift(value) => VisualNodeKind::Shift(value.sanitized()),
            Self::Grain(value) => VisualNodeKind::Grain(value.sanitized()),
            Self::Mask(value) => VisualNodeKind::Mask(value.capture(position_of_layer)),
        }
    }

    fn sanitized(self) -> Self {
        match self {
            Self::Transform(value) => Self::Transform(value.sanitized()),
            Self::DigitalColor(value) => Self::DigitalColor(value.sanitized()),
            Self::Key(value) => Self::Key(value.sanitized()),
            Self::Cellular(value) => Self::Cellular(value.sanitized()),
            Self::Shift(value) => Self::Shift(value.sanitized()),
            Self::Grain(value) => Self::Grain(value.sanitized()),
            Self::Mask(value) => Self::Mask(value.sanitized()),
            marker => marker,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeVisualNode {
    pub stable_id: NodeId,
    pub enabled: bool,
    pub wet: f32,
    pub blend: NodeBlend,
    pub kind: RuntimeVisualNodeKind,
}

impl RuntimeVisualNode {
    pub fn authored(stable_id: NodeId, kind: RuntimeVisualNodeKind) -> Self {
        Self {
            stable_id,
            enabled: true,
            wet: 1.0,
            blend: NodeBlend::Normal,
            kind: kind.sanitized(),
        }
    }

    fn sanitized(self) -> Self {
        Self {
            stable_id: self.stable_id,
            enabled: self.enabled,
            wet: finite_clamp(self.wet, 1.0, 0.0, 1.0),
            blend: self.blend,
            kind: self.kind.sanitized(),
        }
    }

    fn is_exact_legacy_marker(self, expected: NodeKindTag) -> bool {
        self.enabled
            && self.wet == 1.0
            && self.blend == NodeBlend::Normal
            && self.kind.tag() == expected
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeRackError {
    InvalidRack(RackError),
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "reported by retained runtime image-route editor APIs"
        )
    )]
    NotImageMask(NodeId),
}

impl fmt::Display for RuntimeRackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRack(error) => {
                write!(formatter, "runtime visual rack is invalid: {error}")
            }
            Self::NotImageMask(id) => {
                write!(
                    formatter,
                    "runtime visual node {} is not an image mask",
                    id.get()
                )
            }
        }
    }
}

impl std::error::Error for RuntimeRackError {}

impl From<RackError> for RuntimeRackError {
    fn from(value: RackError) -> Self {
        Self::InvalidRack(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteCaptureError {
    InvalidRack(RackError),
}

impl fmt::Display for RouteCaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRack(error) => {
                write!(formatter, "captured visual rack is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for RouteCaptureError {}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeVisualRack {
    nodes: Vec<RuntimeVisualNode>,
    next_node_id: u64,
}

impl VisualRack {
    /// Resolve all image routes as one immutable live rack. Saved positions are
    /// retained only inside selected-route fallback provenance and are not
    /// exposed through any runtime node parameter.
    pub fn resolve_routes(
        &self,
        mut layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: impl Fn(GroupId) -> bool,
    ) -> RuntimeVisualRack {
        let nodes = self
            .nodes
            .iter()
            .map(|node| RuntimeVisualNode {
                stable_id: node.stable_id,
                enabled: node.enabled,
                wet: node.wet,
                blend: node.blend,
                kind: RuntimeVisualNodeKind::resolve(
                    node.kind,
                    &mut layer_at_position,
                    &group_exists,
                ),
            })
            .collect();
        RuntimeVisualRack {
            nodes,
            next_node_id: self.next_node_id,
        }
    }
}

impl RuntimeVisualRack {
    pub fn empty() -> Self {
        VisualRack::empty().resolve_routes(|_| None, |_| false)
    }

    pub fn synthetic_legacy(scope: LegacyRackScope) -> Self {
        VisualRack::synthetic_legacy(scope).resolve_routes(|_| None, |_| false)
    }

    pub fn try_from_parts(
        nodes: Vec<RuntimeVisualNode>,
        next_node_id: Option<u64>,
    ) -> Result<Self, RuntimeRackError> {
        if nodes.len() > MAX_NODES_PER_RACK {
            return Err(RackError::TooManyNodes {
                limit: MAX_NODES_PER_RACK,
            }
            .into());
        }
        let mut ids = BTreeSet::new();
        let mut greatest = NodeId::LEGACY_TEMPORAL.get();
        for node in &nodes {
            if !ids.insert(node.stable_id) {
                return Err(RackError::DuplicateNodeId(node.stable_id).into());
            }
            greatest = greatest.max(node.stable_id.get());
            validate_runtime_legacy_marker(*node)?;
        }
        let inferred = greatest.checked_add(1).unwrap_or(0);
        let next = next_node_id.unwrap_or(inferred);
        if next != 0 && next <= greatest || next == 0 && greatest != u64::MAX {
            return Err(RackError::InvalidNextNodeId {
                next,
                greatest_observed: greatest,
            }
            .into());
        }
        let rack = Self {
            nodes: nodes
                .into_iter()
                .map(RuntimeVisualNode::sanitized)
                .collect(),
            next_node_id: next,
        };
        rack.resource_budget()?;
        Ok(rack)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[allow(
        dead_code,
        reason = "runtime-rack emptiness remains a compatibility inspection API"
    )]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &RuntimeVisualNode> {
        self.nodes.iter()
    }

    pub fn get(&self, id: NodeId) -> Option<&RuntimeVisualNode> {
        self.nodes.iter().find(|node| node.stable_id == id)
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut RuntimeVisualNode> {
        self.nodes.iter_mut().find(|node| node.stable_id == id)
    }

    pub const fn next_node_id_raw(&self) -> u64 {
        self.next_node_id
    }

    pub fn topology_signature(&self) -> u64 {
        self.nodes.iter().fold(FNV_OFFSET, |hash, node| {
            let hash = signature_step(hash, node.stable_id.get());
            signature_step(hash, u64::from(node.kind.tag().signature_code()))
        })
    }

    pub fn is_exact_legacy(&self, scope: LegacyRackScope) -> bool {
        let synthetic = Self::synthetic_legacy(scope);
        self.nodes == synthetic.nodes
            && self.topology_signature()
                == match scope {
                    LegacyRackScope::Layer => LEGACY_LAYER_RACK_SIGNATURE,
                    LegacyRackScope::Master => LEGACY_MASTER_RACK_SIGNATURE,
                    LegacyRackScope::Group => FNV_OFFSET,
                }
    }

    pub fn validate_for_scope(&self, scope: LegacyRackScope) -> Result<(), RuntimeRackError> {
        let canonical_index = self
            .nodes
            .iter()
            .position(|node| node.kind.tag() == NodeKindTag::LegacyCanonical);
        let temporal_index = self
            .nodes
            .iter()
            .position(|node| node.kind.tag() == NodeKindTag::LegacyTemporal);
        if scope == LegacyRackScope::Group {
            if canonical_index.is_some() {
                return Err(RackError::LegacyMarkerOnGroup(NodeKindTag::LegacyCanonical).into());
            }
            if temporal_index.is_some() {
                return Err(RackError::LegacyMarkerOnGroup(NodeKindTag::LegacyTemporal).into());
            }
            return self.resource_budget().map(|_| ());
        }
        if scope == LegacyRackScope::Layer && temporal_index.is_some() {
            return Err(RackError::LegacyTemporalOnLayer.into());
        }
        // Marker order is executable topology. The advanced frame planner
        // splits custom GPU segments at these immutable host boundaries.
        self.resource_budget().map(|_| ())
    }

    pub fn insert(
        &mut self,
        index: usize,
        kind: RuntimeVisualNodeKind,
    ) -> Result<NodeId, RuntimeRackError> {
        if self.nodes.len() == MAX_NODES_PER_RACK {
            return Err(RackError::TooManyNodes {
                limit: MAX_NODES_PER_RACK,
            }
            .into());
        }
        if index > self.nodes.len() {
            return Err(RackError::InvalidMoveIndex {
                index,
                len: self.nodes.len(),
            }
            .into());
        }
        if matches!(
            kind,
            RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal
        ) {
            return Err(RackError::LegacyMarkerId {
                tag: kind.tag(),
                expected: if matches!(kind, RuntimeVisualNodeKind::LegacyCanonical) {
                    NodeId::LEGACY_CANONICAL
                } else {
                    NodeId::LEGACY_TEMPORAL
                },
            }
            .into());
        }
        let id = NodeId::new(self.next_node_id).ok_or(RackError::NodeIdExhausted)?;
        self.next_node_id = self.next_node_id.checked_add(1).unwrap_or(0);
        self.nodes
            .insert(index, RuntimeVisualNode::authored(id, kind));
        if let Err(error) = self.resource_budget() {
            self.nodes.remove(index);
            return Err(error);
        }
        Ok(id)
    }

    pub fn push(&mut self, kind: RuntimeVisualNodeKind) -> Result<NodeId, RuntimeRackError> {
        self.insert(self.nodes.len(), kind)
    }

    pub fn remove(&mut self, id: NodeId) -> Option<RuntimeVisualNode> {
        if id == NodeId::LEGACY_CANONICAL || id == NodeId::LEGACY_TEMPORAL {
            return None;
        }
        let index = self.nodes.iter().position(|node| node.stable_id == id)?;
        Some(self.nodes.remove(index))
    }

    pub fn move_node(
        &mut self,
        id: NodeId,
        new_index: usize,
        scope: LegacyRackScope,
    ) -> Result<(), RuntimeRackError> {
        if new_index >= self.nodes.len() {
            return Err(RackError::InvalidMoveIndex {
                index: new_index,
                len: self.nodes.len(),
            }
            .into());
        }
        let old_index = self
            .nodes
            .iter()
            .position(|node| node.stable_id == id)
            .ok_or(RackError::UnknownNode(id))?;
        let node = self.nodes.remove(old_index);
        self.nodes.insert(new_index, node);
        if let Err(error) = self.validate_for_scope(scope) {
            let node = self.nodes.remove(new_index);
            self.nodes.insert(old_index, node);
            return Err(error);
        }
        Ok(())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "route inspection is retained for rack editor adapters"
        )
    )]
    pub fn image_mask_route(&self, id: NodeId) -> Option<ResolvedImageTap> {
        match self.get(id)?.kind {
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) => Some(matte.tap),
            _ => None,
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "route mutation is retained for rack editor adapters"
        )
    )]
    pub fn set_image_mask_route(
        &mut self,
        id: NodeId,
        tap: ResolvedImageTap,
    ) -> Result<(), RuntimeRackError> {
        let Some(node) = self.get_mut(id) else {
            return Err(RackError::UnknownNode(id).into());
        };
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) = &mut node.kind else {
            return Err(RuntimeRackError::NotImageMask(id));
        };
        matte.tap = tap;
        Ok(())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "descriptor lookup is retained for rack editor adapters"
        )
    )]
    pub fn node_descriptor(&self, id: NodeId) -> Option<&'static NodeKindDescriptor> {
        self.get(id)
            .map(|node| node_kind_descriptor(node.kind.tag()))
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "descriptor iteration is retained for rack editor adapters"
        )
    )]
    pub fn parameter_descriptors(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = &'static NodeParamDescriptor> {
        let tag = self.get(id).map(|node| node.kind.tag());
        NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(move |descriptor| Some(descriptor.kind) == tag)
    }

    pub fn resource_budget(&self) -> Result<RackResourceBudget, RuntimeRackError> {
        let mut result = RackResourceBudget::default();
        for node in self.nodes.iter().filter(|node| node.enabled) {
            let budget = node_kind_descriptor(node.kind.tag()).budget;
            result.full_frame_passes = result
                .full_frame_passes
                .checked_add(u32::from(budget.full_frame_passes))
                .ok_or(RackError::ResourceOverflow)?;
            result.logical_texture_lookups_per_pixel = result
                .logical_texture_lookups_per_pixel
                .checked_add(u32::from(budget.logical_texture_lookups_per_pixel))
                .ok_or(RackError::ResourceOverflow)?;
            result.texture_samples_per_pixel = result
                .texture_samples_per_pixel
                .checked_add(u32::from(budget.texture_samples_per_pixel))
                .ok_or(RackError::ResourceOverflow)?;
            result.max_sampled_textures_in_pass = result
                .max_sampled_textures_in_pass
                .max(u32::from(budget.sampled_textures_in_pass));
            result.cross_input_taps = result
                .cross_input_taps
                .checked_add(u32::from(budget.cross_input_taps))
                .ok_or(RackError::ResourceOverflow)?;
        }
        if result.logical_texture_lookups_per_pixel > MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK {
            return Err(RackError::RackLogicalLookupBudget {
                lookups: result.logical_texture_lookups_per_pixel,
                limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK,
            }
            .into());
        }
        if result.texture_samples_per_pixel > MAX_TEXTURE_SAMPLES_PER_RACK {
            return Err(RackError::RackSampleBudget {
                samples: result.texture_samples_per_pixel,
                limit: MAX_TEXTURE_SAMPLES_PER_RACK,
            }
            .into());
        }
        if result.max_sampled_textures_in_pass > MAX_SAMPLED_TEXTURES_PER_PASS {
            return Err(RackError::PassTextureBudget {
                textures: result.max_sampled_textures_in_pass,
                limit: MAX_SAMPLED_TEXTURES_PER_PASS,
            }
            .into());
        }
        Ok(result)
    }

    pub fn referenced_group_ids(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.nodes.iter().filter_map(|node| match node.kind {
            RuntimeVisualNodeKind::Mask(mask) => mask.image_tap()?.referenced_group(),
            _ => None,
        })
    }

    pub fn selected_layer_ids(&self) -> impl Iterator<Item = StableLayerId> + '_ {
        self.nodes.iter().filter_map(|node| match node.kind {
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) => {
                matte.selected_layer_id()
            }
            _ => None,
        })
    }

    pub fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        for node in &mut self.nodes {
            let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) = &mut node.kind
            else {
                continue;
            };
            matte.tap.mark_layer_missing(removed);
        }
    }

    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        for node in &mut self.nodes {
            let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) = &mut node.kind
            else {
                continue;
            };
            matte.tap.mark_group_missing(removed);
        }
    }

    pub fn capture_routes(
        &self,
        mut position_of_layer: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> Result<VisualRack, RouteCaptureError> {
        let nodes = self
            .nodes
            .iter()
            .map(|node| VisualNode {
                stable_id: node.stable_id,
                enabled: node.enabled,
                wet: node.wet,
                blend: node.blend,
                kind: node.kind.capture(&mut position_of_layer),
            })
            .collect();
        VisualRack::try_from_parts(nodes, Some(self.next_node_id))
            .map_err(RouteCaptureError::InvalidRack)
    }
}

fn validate_runtime_legacy_marker(node: RuntimeVisualNode) -> Result<(), RuntimeRackError> {
    match node.stable_id {
        NodeId::LEGACY_CANONICAL if node.kind.tag() != NodeKindTag::LegacyCanonical => {
            return Err(RackError::ReservedNodeId {
                id: node.stable_id,
                expected: NodeKindTag::LegacyCanonical,
            }
            .into());
        }
        NodeId::LEGACY_TEMPORAL if node.kind.tag() != NodeKindTag::LegacyTemporal => {
            return Err(RackError::ReservedNodeId {
                id: node.stable_id,
                expected: NodeKindTag::LegacyTemporal,
            }
            .into());
        }
        _ => {}
    }
    let expected = match node.kind.tag() {
        NodeKindTag::LegacyCanonical => Some(NodeId::LEGACY_CANONICAL),
        NodeKindTag::LegacyTemporal => Some(NodeId::LEGACY_TEMPORAL),
        _ => None,
    };
    if let Some(expected) = expected {
        if node.stable_id != expected {
            return Err(RackError::LegacyMarkerId {
                tag: node.kind.tag(),
                expected,
            }
            .into());
        }
        if !node.is_exact_legacy_marker(node.kind.tag()) {
            return Err(RackError::MutableLegacyMarker(node.stable_id).into());
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VisualScopeId {
    Master,
    Layer(StableLayerId),
    Group(GroupId),
    /// Final clean program output, used only as an image producer.
    Program,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDependency {
    pub consumer: VisualScopeId,
    pub producer: VisualScopeId,
    pub timing: EdgeTiming,
}

/// A same-frame producer-before-consumer relation used only for ordering and
/// cycle validation. Unlike [`ImageDependency`], it is not a texture tap and
/// therefore does not consume a tap/history budget. This keeps one AllBelow
/// input as one logical tap even when a caller expands its composite prefix
/// for a small bounded validation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageOrderingEdge {
    pub producer: VisualScopeId,
    pub consumer: VisualScopeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGraphMode {
    Advanced,
    /// Migration-only escape hatch for M1 CleanProgram mattes. The edge is
    /// omitted and evaluates as transparent with an explicit diagnostic.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "frozen M1 migration mode is exercised by compatibility goldens"
        )
    )]
    LegacyM1TransparentProgram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGraphDiagnostic {
    MissingProducer {
        consumer: VisualScopeId,
        producer: VisualScopeId,
    },
    LegacyCurrentProgramTransparent {
        consumer: VisualScopeId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageGraphError {
    TooManyScopes { count: usize, limit: usize },
    TooManyDependencies { count: usize, limit: usize },
    TooManyCurrentTaps { count: usize, limit: usize },
    TooManyPreviousTaps { count: usize, limit: usize },
    DuplicateScope(VisualScopeId),
    MissingConsumer(VisualScopeId),
    CurrentProgramInput { consumer: VisualScopeId },
    CurrentCycle { scopes: Vec<VisualScopeId> },
}

impl fmt::Display for ImageGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyScopes { count, limit } => write!(
                formatter,
                "image graph has {count} scopes; limit is {limit}"
            ),
            Self::TooManyDependencies { count, limit } => write!(
                formatter,
                "image graph has {count} dependencies; limit is {limit}"
            ),
            Self::TooManyCurrentTaps { count, limit } => write!(
                formatter,
                "image graph has {count} current-frame taps; limit is {limit}"
            ),
            Self::TooManyPreviousTaps { count, limit } => write!(
                formatter,
                "image graph has {count} previous-frame taps; limit is {limit}"
            ),
            Self::DuplicateScope(scope) => write!(formatter, "duplicate visual scope {scope:?}"),
            Self::MissingConsumer(scope) => {
                write!(formatter, "image dependency consumer {scope:?} is missing")
            }
            Self::CurrentProgramInput { consumer } => write!(
                formatter,
                "same-frame CleanProgram input is invalid for advanced consumer {consumer:?}"
            ),
            Self::CurrentCycle { scopes } => {
                write!(formatter, "same-frame image dependency cycle: {scopes:?}")
            }
        }
    }
}

impl std::error::Error for ImageGraphError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGraphPlan {
    /// Producer-before-consumer order for current-frame scopes. Missing and
    /// previous-frame producers do not participate.
    pub current_topological_order: Vec<VisualScopeId>,
    pub current_taps: usize,
    pub previous_taps: usize,
    pub diagnostics: Vec<ImageGraphDiagnostic>,
}

pub struct ImageDependencyGraph;

impl ImageDependencyGraph {
    pub fn validate(
        scopes: &[VisualScopeId],
        dependencies: &[ImageDependency],
        mode: ImageGraphMode,
    ) -> Result<ImageGraphPlan, ImageGraphError> {
        Self::validate_with_ordering_edges(scopes, dependencies, &[], mode)
    }

    pub fn validate_with_ordering_edges(
        scopes: &[VisualScopeId],
        dependencies: &[ImageDependency],
        ordering_edges: &[ImageOrderingEdge],
        mode: ImageGraphMode,
    ) -> Result<ImageGraphPlan, ImageGraphError> {
        if scopes.len() > MAX_GRAPH_SCOPES {
            return Err(ImageGraphError::TooManyScopes {
                count: scopes.len(),
                limit: MAX_GRAPH_SCOPES,
            });
        }
        if dependencies.len() > MAX_IMAGE_DEPENDENCIES {
            return Err(ImageGraphError::TooManyDependencies {
                count: dependencies.len(),
                limit: MAX_IMAGE_DEPENDENCIES,
            });
        }

        let mut known = BTreeSet::new();
        for scope in scopes.iter().copied().chain([VisualScopeId::Program]) {
            if !known.insert(scope) {
                return Err(ImageGraphError::DuplicateScope(scope));
            }
        }

        let mut current_taps = 0_usize;
        let mut previous_taps = 0_usize;
        let mut diagnostics = Vec::new();
        let mut adjacency: BTreeMap<VisualScopeId, BTreeSet<VisualScopeId>> = known
            .iter()
            .copied()
            .map(|scope| (scope, BTreeSet::new()))
            .collect();
        let mut indegree: BTreeMap<VisualScopeId, usize> =
            known.iter().copied().map(|scope| (scope, 0)).collect();

        for dependency in dependencies {
            if !known.contains(&dependency.consumer) {
                return Err(ImageGraphError::MissingConsumer(dependency.consumer));
            }
            if dependency.timing == EdgeTiming::PreviousFrame {
                previous_taps += 1;
                continue;
            }
            if dependency.producer == VisualScopeId::Program {
                if mode == ImageGraphMode::Advanced {
                    return Err(ImageGraphError::CurrentProgramInput {
                        consumer: dependency.consumer,
                    });
                }
                diagnostics.push(ImageGraphDiagnostic::LegacyCurrentProgramTransparent {
                    consumer: dependency.consumer,
                });
                continue;
            }
            if !known.contains(&dependency.producer) {
                diagnostics.push(ImageGraphDiagnostic::MissingProducer {
                    consumer: dependency.consumer,
                    producer: dependency.producer,
                });
                continue;
            }
            current_taps += 1;
            if adjacency
                .get_mut(&dependency.producer)
                .expect("known producer has adjacency")
                .insert(dependency.consumer)
            {
                *indegree
                    .get_mut(&dependency.consumer)
                    .expect("known consumer has indegree") += 1;
            }
        }

        if current_taps > MAX_CURRENT_IMAGE_TAPS {
            return Err(ImageGraphError::TooManyCurrentTaps {
                count: current_taps,
                limit: MAX_CURRENT_IMAGE_TAPS,
            });
        }
        if previous_taps > MAX_PREVIOUS_IMAGE_TAPS {
            return Err(ImageGraphError::TooManyPreviousTaps {
                count: previous_taps,
                limit: MAX_PREVIOUS_IMAGE_TAPS,
            });
        }

        for edge in ordering_edges {
            if !known.contains(&edge.consumer) {
                return Err(ImageGraphError::MissingConsumer(edge.consumer));
            }
            if edge.producer == VisualScopeId::Program {
                if mode == ImageGraphMode::Advanced {
                    return Err(ImageGraphError::CurrentProgramInput {
                        consumer: edge.consumer,
                    });
                }
                diagnostics.push(ImageGraphDiagnostic::LegacyCurrentProgramTransparent {
                    consumer: edge.consumer,
                });
                continue;
            }
            if !known.contains(&edge.producer) {
                diagnostics.push(ImageGraphDiagnostic::MissingProducer {
                    consumer: edge.consumer,
                    producer: edge.producer,
                });
                continue;
            }
            if adjacency
                .get_mut(&edge.producer)
                .expect("known ordering producer has adjacency")
                .insert(edge.consumer)
            {
                *indegree
                    .get_mut(&edge.consumer)
                    .expect("known ordering consumer has indegree") += 1;
            }
        }

        let mut ready: BTreeSet<_> = indegree
            .iter()
            .filter_map(|(scope, degree)| (*degree == 0).then_some(*scope))
            .collect();
        let mut order = Vec::with_capacity(known.len());
        while let Some(scope) = ready.pop_first() {
            order.push(scope);
            for consumer in adjacency
                .get(&scope)
                .expect("every known scope has adjacency")
            {
                let degree = indegree
                    .get_mut(consumer)
                    .expect("every known scope has indegree");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*consumer);
                }
            }
        }
        if order.len() != known.len() {
            let scopes = indegree
                .into_iter()
                .filter_map(|(scope, degree)| (degree != 0).then_some(scope))
                .collect();
            return Err(ImageGraphError::CurrentCycle { scopes });
        }

        Ok(ImageGraphPlan {
            current_topological_order: order,
            current_taps,
            previous_taps,
            diagnostics,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreativeResourceLimits {
    pub max_texture_dimension_2d: u32,
    pub max_texture_array_layers: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_creative_bytes: u64,
}

impl Default for CreativeResourceLimits {
    fn default() -> Self {
        Self {
            max_texture_dimension_2d: 8_192,
            max_texture_array_layers: 256,
            max_sampled_textures_per_shader_stage: MAX_SAMPLED_TEXTURES_PER_PASS,
            max_creative_bytes: MAX_CREATIVE_GPU_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourcePreflightError {
    ZeroDimension,
    DimensionLimit { requested: [u32; 2], limit: u32 },
    Rack { index: usize, error: RackError },
    FrameLogicalLookupBudget { lookups: u32, limit: u32 },
    FrameSampleBudget { samples: u32, limit: u32 },
    SampledTextureLimit { requested: u32, limit: u32 },
    TextureArrayLayerLimit { requested: u32, limit: u32 },
    ArithmeticOverflow,
    CreativeMemoryBudget { bytes: u64, limit: u64 },
}

impl fmt::Display for ResourcePreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("creative output dimensions must be non-zero"),
            Self::DimensionLimit { requested, limit } => write!(formatter, "creative output {}x{} exceeds adapter dimension limit {limit}", requested[0], requested[1]),
            Self::Rack { index, error } => write!(formatter, "rack {index} failed resource validation: {error}"),
            Self::FrameLogicalLookupBudget { lookups, limit } => write!(formatter, "frame requests {lookups} logical texture lookups per pixel; limit is {limit}"),
            Self::FrameSampleBudget { samples, limit } => write!(formatter, "frame requests {samples} texture samples per pixel; limit is {limit}"),
            Self::SampledTextureLimit { requested, limit } => write!(formatter, "pass requests {requested} sampled textures; adapter limit is {limit}"),
            Self::TextureArrayLayerLimit { requested, limit } => write!(formatter, "creative plan requests {requested} retained texture layers; adapter limit is {limit}"),
            Self::ArithmeticOverflow => formatter.write_str("creative GPU resource arithmetic overflowed"),
            Self::CreativeMemoryBudget { bytes, limit } => write!(formatter, "creative plan requests {bytes} bytes; limit is {limit}"),
        }
    }
}

impl std::error::Error for ResourcePreflightError {}

fn validate_frame_texture_budgets(
    logical_lookups: u32,
    shader_operations: u32,
) -> Result<(), ResourcePreflightError> {
    if logical_lookups > MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME {
        return Err(ResourcePreflightError::FrameLogicalLookupBudget {
            lookups: logical_lookups,
            limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME,
        });
    }
    if shader_operations > MAX_TEXTURE_SAMPLES_PER_FRAME {
        return Err(ResourcePreflightError::FrameSampleBudget {
            samples: shader_operations,
            limit: MAX_TEXTURE_SAMPLES_PER_FRAME,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreativeResourcePlan {
    pub output_size: [u32; 2],
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_samples_per_pixel: u32,
    /// Total full-frame allocations, independent of storage format.
    pub retained_surface_layers: u32,
    /// Full-frame RGBA16Float allocations (8 bytes per pixel).
    pub rgba16_surface_layers: u32,
    /// Full-frame Compat8 allocations (4 bytes per pixel).
    pub compat8_surface_layers: u32,
    pub creative_bytes: u64,
}

impl CreativeResourcePlan {
    /// Validate the complete immutable creative plan before any GPU allocation.
    /// Advanced processing uses RGBA16Float (8 bytes/pixel). The conservative
    /// base covers rack ping/pong, independent A/B/Program accumulators, and
    /// shared group-local scratch. Current retained taps, unique materialized
    /// PreLocal donors, and every exact N-1 history/stage image add retained
    /// layers in this same ledger.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "base preflight constructor is exposed for resource goldens"
        )
    )]
    pub fn preflight(
        output_size: [u32; 2],
        racks: &[VisualRack],
        graph: &ImageGraphPlan,
        limits: CreativeResourceLimits,
    ) -> Result<Self, ResourcePreflightError> {
        Self::preflight_with_surface_formats(output_size, racks, graph, 0, 0, limits)
    }

    /// Advanced host-boundary allocations that are not individual rack/tap
    /// surfaces (currently the fixed Temporal ring/feedback) join the same
    /// checked RGBA16 ledger before any GPU constructor runs.
    #[allow(
        dead_code,
        reason = "RGBA16-only compatibility constructor remains part of the preflight API"
    )]
    pub fn preflight_with_additional_surfaces(
        output_size: [u32; 2],
        racks: &[VisualRack],
        graph: &ImageGraphPlan,
        additional_surface_layers: u32,
        limits: CreativeResourceLimits,
    ) -> Result<Self, ResourcePreflightError> {
        Self::preflight_with_surface_formats(
            output_size,
            racks,
            graph,
            additional_surface_layers,
            0,
            limits,
        )
    }

    /// Join renderer-owned full-frame allocations to the same immutable
    /// ledger while retaining their actual storage formats. Every tap, N-1
    /// image, host surface, and extra RGBA16 surface is charged at 8 B/px;
    /// temporal Compat8 history/feedback is charged at 4 B/px. The total layer
    /// count remains checked independently against adapter limits.
    pub fn preflight_with_surface_formats(
        output_size: [u32; 2],
        racks: &[VisualRack],
        graph: &ImageGraphPlan,
        additional_rgba16_layers: u32,
        additional_compat8_layers: u32,
        limits: CreativeResourceLimits,
    ) -> Result<Self, ResourcePreflightError> {
        if output_size[0] == 0 || output_size[1] == 0 {
            return Err(ResourcePreflightError::ZeroDimension);
        }
        if output_size[0] > limits.max_texture_dimension_2d
            || output_size[1] > limits.max_texture_dimension_2d
        {
            return Err(ResourcePreflightError::DimensionLimit {
                requested: output_size,
                limit: limits.max_texture_dimension_2d,
            });
        }

        let mut passes = 0_u32;
        let mut logical_lookups = 0_u32;
        let mut samples = 0_u32;
        let mut max_textures = 0_u32;
        for (index, rack) in racks.iter().enumerate() {
            let budget = rack
                .resource_budget()
                .map_err(|error| ResourcePreflightError::Rack { index, error })?;
            passes = passes
                .checked_add(budget.full_frame_passes)
                .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
            logical_lookups = logical_lookups
                .checked_add(budget.logical_texture_lookups_per_pixel)
                .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
            samples = samples
                .checked_add(budget.texture_samples_per_pixel)
                .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
            max_textures = max_textures.max(budget.max_sampled_textures_in_pass);
        }
        validate_frame_texture_budgets(logical_lookups, samples)?;
        let texture_limit = limits
            .max_sampled_textures_per_shader_stage
            .min(MAX_SAMPLED_TEXTURES_PER_PASS);
        if max_textures > texture_limit {
            return Err(ResourcePreflightError::SampledTextureLimit {
                requested: max_textures,
                limit: texture_limit,
            });
        }

        let tap_layers = graph
            .current_taps
            .checked_add(graph.previous_taps)
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        let rgba16_surface_layers = u32::try_from(tap_layers)
            .ok()
            .and_then(|value| value.checked_add(BASE_CREATIVE_SURFACE_LAYERS))
            .and_then(|value| value.checked_add(additional_rgba16_layers))
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        let retained_surface_layers = rgba16_surface_layers
            .checked_add(additional_compat8_layers)
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        if retained_surface_layers > limits.max_texture_array_layers {
            return Err(ResourcePreflightError::TextureArrayLayerLimit {
                requested: retained_surface_layers,
                limit: limits.max_texture_array_layers,
            });
        }
        let pixels = u64::from(output_size[0])
            .checked_mul(u64::from(output_size[1]))
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        let rgba16_bytes = pixels
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_mul(u64::from(rgba16_surface_layers)))
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        let compat8_bytes = pixels
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_mul(u64::from(additional_compat8_layers)))
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        let creative_bytes = rgba16_bytes
            .checked_add(compat8_bytes)
            .ok_or(ResourcePreflightError::ArithmeticOverflow)?;
        let byte_limit = limits.max_creative_bytes.min(MAX_CREATIVE_GPU_BYTES);
        if creative_bytes > byte_limit {
            return Err(ResourcePreflightError::CreativeMemoryBudget {
                bytes: creative_bytes,
                limit: byte_limit,
            });
        }
        Ok(Self {
            output_size,
            full_frame_passes: passes,
            logical_texture_lookups_per_pixel: logical_lookups,
            texture_samples_per_pixel: samples,
            retained_surface_layers,
            rgba16_surface_layers,
            compat8_surface_layers: additional_compat8_layers,
            creative_bytes,
        })
    }
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

fn wrap_degrees(value: f32) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn live_id(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn saved_position(value: u32) -> SavedLayerPosition {
        SavedLayerPosition::new(value).unwrap()
    }

    fn scope(value: u64) -> VisualScopeId {
        VisualScopeId::Layer(live_id(value))
    }

    #[test]
    fn ninth_node_is_rejected_before_collection_growth() {
        let nodes = (3..=11)
            .map(|id| format!(r#"{{"stable_id":{id},"kind":{{"kind":"grain","params":{{}}}}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(r#"{{"nodes":[{nodes}],"next_node_id":12}}"#);
        let error = serde_json::from_str::<VisualRack>(&json).unwrap_err();
        assert!(error.to_string().contains("at most 8"));
    }

    #[test]
    fn hostile_ids_duplicates_and_cursor_rewinds_reject() {
        let zero = r#"{"nodes":[{"stable_id":0,"kind":{"kind":"grain","params":{}}}]}"#;
        assert!(serde_json::from_str::<VisualRack>(zero).is_err());

        let duplicate = r#"{"nodes":[{"stable_id":3,"kind":{"kind":"grain","params":{}}},{"stable_id":3,"kind":{"kind":"shift","params":{}}}],"next_node_id":4}"#;
        assert!(serde_json::from_str::<VisualRack>(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));

        let rewind =
            r#"{"nodes":[{"stable_id":9,"kind":{"kind":"grain","params":{}}}],"next_node_id":9}"#;
        assert!(serde_json::from_str::<VisualRack>(rewind)
            .unwrap_err()
            .to_string()
            .contains("advance past"));

        let reserved =
            r#"{"nodes":[{"stable_id":1,"kind":{"kind":"grain","params":{}}}],"next_node_id":3}"#;
        assert!(serde_json::from_str::<VisualRack>(reserved)
            .unwrap_err()
            .to_string()
            .contains("reserved"));
    }

    #[test]
    fn deserialize_sanitizes_hostile_floats() {
        let yaml = "nodes:\n  - stable_id: 3\n    wet: .nan\n    kind:\n      kind: shift\n      params:\n        amount: .inf\n        block_size: -99\nnext_node_id: 4\n";
        let rack = serde_yaml::from_str::<VisualRack>(yaml).unwrap();
        let node = rack.get(NodeId::new(3).unwrap()).unwrap();
        assert_eq!(node.wet, 1.0);
        let VisualNodeKind::Shift(params) = node.kind else {
            panic!("expected Shift");
        };
        assert_eq!(params.amount, 0.0);
        assert_eq!(params.block_size, 2.0);
    }

    #[test]
    fn synthetic_legacy_racks_have_frozen_exact_signatures() {
        let layer = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        let master = VisualRack::synthetic_legacy(LegacyRackScope::Master);
        assert!(layer.is_exact_legacy(LegacyRackScope::Layer));
        assert!(master.is_exact_legacy(LegacyRackScope::Master));
        assert_eq!(layer.topology_signature(), LEGACY_LAYER_RACK_SIGNATURE);
        assert_eq!(master.topology_signature(), LEGACY_MASTER_RACK_SIGNATURE);
        assert_ne!(LEGACY_LAYER_RACK_SIGNATURE, LEGACY_MASTER_RACK_SIGNATURE);

        let mut changed = layer.clone();
        changed.nodes[0].wet = 0.5;
        assert!(!changed.is_exact_legacy(LegacyRackScope::Layer));
        assert!(matches!(
            VisualRack::try_from_parts(changed.nodes, Some(3)),
            Err(RackError::MutableLegacyMarker(_))
        ));
    }

    #[test]
    fn retired_allocator_cursor_does_not_change_saved_or_runtime_legacy_exactness() {
        let mut saved = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        let retired = saved
            .push(VisualNodeKind::Shift(ShiftParams::default()))
            .unwrap();
        assert_eq!(retired, NodeId::new(3).unwrap());
        saved.remove(retired).unwrap();
        assert_eq!(saved.next_node_id_raw(), 4);
        assert!(saved.is_exact_legacy(LegacyRackScope::Layer));
        let next = saved
            .push(VisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        assert_eq!(next, NodeId::new(4).unwrap());

        let mut runtime = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let retired = runtime
            .push(RuntimeVisualNodeKind::Shift(ShiftParams::default()))
            .unwrap();
        assert_eq!(retired, NodeId::new(3).unwrap());
        runtime.remove(retired).unwrap();
        assert_eq!(runtime.next_node_id_raw(), 4);
        assert!(runtime.is_exact_legacy(LegacyRackScope::Master));
        let next = runtime
            .push(RuntimeVisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        assert_eq!(next, NodeId::new(4).unwrap());

        let mut altered = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        altered.get_mut(NodeId::LEGACY_CANONICAL).unwrap().wet = 0.5;
        assert!(!altered.is_exact_legacy(LegacyRackScope::Master));
    }

    #[test]
    fn explicit_racks_may_reorder_immutable_host_boundary_markers() {
        let exact_master = VisualRack::synthetic_legacy(LegacyRackScope::Master);
        let exact_signature = exact_master.topology_signature();
        let mut master = exact_master.clone();
        let shift = master
            .push(VisualNodeKind::Shift(ShiftParams::default()))
            .unwrap();
        let _grain = master
            .push(VisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        master.move_node(shift, 0, LegacyRackScope::Master).unwrap();
        master
            .move_node(NodeId::LEGACY_TEMPORAL, 1, LegacyRackScope::Master)
            .unwrap();
        assert_eq!(
            master
                .iter()
                .map(|node| node.kind.tag())
                .collect::<Vec<_>>(),
            vec![
                NodeKindTag::Shift,
                NodeKindTag::LegacyTemporal,
                NodeKindTag::LegacyCanonical,
                NodeKindTag::Grain,
            ]
        );
        master.validate_for_scope(LegacyRackScope::Master).unwrap();
        assert!(!master.is_exact_legacy(LegacyRackScope::Master));
        assert_ne!(master.topology_signature(), exact_signature);
        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Master).topology_signature(),
            exact_signature
        );

        let mut layer = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        let authored = layer
            .push(VisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        layer
            .move_node(NodeId::LEGACY_CANONICAL, 1, LegacyRackScope::Layer)
            .unwrap();
        assert_eq!(layer.iter().next().unwrap().stable_id, authored);
        layer.validate_for_scope(LegacyRackScope::Layer).unwrap();
    }

    #[test]
    fn groups_start_empty_and_reject_legacy_markers() {
        let empty = VisualRack::synthetic_legacy(LegacyRackScope::Group);
        assert!(empty.is_empty());
        assert!(empty.is_exact_legacy(LegacyRackScope::Group));
        assert!(matches!(
            VisualRack::synthetic_legacy(LegacyRackScope::Layer)
                .validate_for_scope(LegacyRackScope::Group),
            Err(RackError::LegacyMarkerOnGroup(NodeKindTag::LegacyCanonical))
        ));
        assert!(matches!(
            VisualRack::synthetic_legacy(LegacyRackScope::Master)
                .validate_for_scope(LegacyRackScope::Group),
            Err(RackError::LegacyMarkerOnGroup(NodeKindTag::LegacyCanonical))
        ));
    }

    #[test]
    fn deleted_node_identity_is_retired_and_reorder_uses_id() {
        let mut rack = VisualRack::empty();
        let first = rack
            .push(VisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        let second = rack
            .push(VisualNodeKind::Shift(ShiftParams::default()))
            .unwrap();
        rack.move_node(second, 0, LegacyRackScope::Group).unwrap();
        assert_eq!(rack.iter().next().unwrap().stable_id, second);
        assert_eq!(rack.remove(first).unwrap().stable_id, first);
        let third = rack
            .push(VisualNodeKind::Key(KeyParams::default()))
            .unwrap();
        assert!(third.get() > second.get());
        assert_ne!(third, first);
    }

    #[test]
    fn descriptor_registry_is_complete_unique_and_budgeted() {
        let tags: BTreeSet<_> = NODE_KIND_DESCRIPTORS
            .iter()
            .map(|value| value.tag)
            .collect();
        let keys: BTreeSet<_> = NODE_KIND_DESCRIPTORS
            .iter()
            .map(|value| value.key)
            .collect();
        assert_eq!(tags.len(), 9);
        assert_eq!(keys.len(), 9);
        for descriptor in NODE_KIND_DESCRIPTORS {
            assert!(descriptor.budget.full_frame_passes > 0);
            assert!(descriptor.budget.logical_texture_lookups_per_pixel > 0);
            assert!(descriptor.budget.texture_samples_per_pixel > 0);
            assert!(descriptor.budget.sampled_textures_in_pass > 0);
            assert_eq!(node_kind_descriptor(descriptor.tag), &descriptor);
        }
        assert!(NODE_PARAM_DESCRIPTORS
            .iter()
            .all(|param| tags.contains(&param.kind)));
        assert_eq!(
            NODE_CONTROL_DESCRIPTORS
                .iter()
                .map(|descriptor| descriptor.key)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["blend", "enabled", "wet"])
        );
        assert_eq!(NODE_CONTROL_DESCRIPTORS.len(), 3);
        assert_eq!(
            node_kind_descriptor(NodeKindTag::Transform)
                .budget
                .texture_samples_per_pixel,
            8,
            "Transform must budget both four-load premultiplied lookups"
        );
        let mut transform_rack = VisualRack::empty();
        transform_rack
            .push(VisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        assert_eq!(
            transform_rack
                .resource_budget()
                .unwrap()
                .texture_samples_per_pixel,
            8
        );
        let advanced_texture_ops = [
            (NodeKindTag::Transform, 8),
            (NodeKindTag::DigitalColor, 16),
            (NodeKindTag::Key, 4),
            (NodeKindTag::Cellular, 8),
            (NodeKindTag::Shift, 8),
            (NodeKindTag::Grain, 4),
            (NodeKindTag::Mask, 5),
        ];
        for (kind, expected) in advanced_texture_ops {
            assert_eq!(
                node_kind_descriptor(kind).budget.texture_samples_per_pixel,
                expected,
                "{kind:?} must declare worst-case shader texture operations"
            );
        }
        let temporal = node_kind_descriptor(NodeKindTag::LegacyTemporal).budget;
        assert_eq!(temporal.logical_texture_lookups_per_pixel, 6);
        assert_eq!(temporal.texture_samples_per_pixel, 9);
        assert_eq!(temporal.sampled_textures_in_pass, 3);

        assert_eq!(MAX_TEXTURE_SAMPLES_PER_RACK, 128);
        assert_eq!(MAX_TEXTURE_SAMPLES_PER_FRAME, 4_096);
        let mut maximal_advanced_rack = VisualRack::empty();
        for _ in 0..MAX_NODES_PER_RACK {
            maximal_advanced_rack
                .push(VisualNodeKind::DigitalColor(DigitalColorParams::default()))
                .unwrap();
        }
        assert_eq!(
            maximal_advanced_rack
                .resource_budget()
                .unwrap()
                .logical_texture_lookups_per_pixel,
            MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK
        );
        assert_eq!(
            maximal_advanced_rack
                .resource_budget()
                .unwrap()
                .texture_samples_per_pixel,
            MAX_TEXTURE_SAMPLES_PER_RACK,
            "the former 32-logical-lookup ceiling must remain admissible when each lookup is charged as four shader operations"
        );
    }

    #[test]
    fn parameter_registry_covers_every_authored_field_per_kind() {
        let expected: &[(NodeKindTag, &[&str])] = &[
            (NodeKindTag::LegacyCanonical, &[]),
            (NodeKindTag::LegacyTemporal, &[]),
            (
                NodeKindTag::Transform,
                &[
                    "position",
                    "scale",
                    "anchor",
                    "rotation_deg",
                    "skew_deg",
                    "skew_axis_deg",
                    "fit_mode",
                    "crop_left",
                    "crop_top",
                    "crop_right",
                    "crop_bottom",
                    "edge_mode",
                    "sampling",
                ],
            ),
            (
                NodeKindTag::DigitalColor,
                &[
                    "pixelate_size",
                    "rgb_split",
                    "downsample",
                    "hue_shift",
                    "saturation",
                    "brightness",
                    "contrast",
                    "posterize",
                    "invert",
                    "vignette",
                    "color_drift",
                ],
            ),
            (
                NodeKindTag::Key,
                &[
                    "mode",
                    "threshold",
                    "softness",
                    "color",
                    "tolerance",
                    "invert",
                ],
            ),
            (
                NodeKindTag::Cellular,
                &[
                    "amount",
                    "scale",
                    "warp",
                    "speed",
                    "gap_amount",
                    "gap_threshold",
                    "gap_softness",
                    "seed",
                ],
            ),
            (
                NodeKindTag::Shift,
                &["amount", "block_size", "density", "speed", "seed"],
            ),
            (
                NodeKindTag::Grain,
                &["intensity", "size", "algorithm", "color", "seed"],
            ),
            (
                NodeKindTag::Mask,
                &[
                    "variant",
                    "rectangle_center",
                    "rectangle_size",
                    "rectangle_rotation_deg",
                    "rectangle_feather",
                    "rectangle_invert",
                    "ellipse_center",
                    "ellipse_radii",
                    "ellipse_rotation_deg",
                    "ellipse_feather",
                    "ellipse_invert",
                    "image_tap",
                    "image_channel",
                    "image_invert",
                    "image_amount",
                    "image_threshold",
                    "image_softness",
                ],
            ),
        ];
        for (kind, expected_keys) in expected {
            let descriptors: Vec<_> = NODE_PARAM_DESCRIPTORS
                .iter()
                .filter(|descriptor| descriptor.kind == *kind)
                .collect();
            let actual: BTreeSet<_> = descriptors
                .iter()
                .map(|descriptor| descriptor.key)
                .collect();
            let expected: BTreeSet<_> = expected_keys.iter().copied().collect();
            assert_eq!(actual, expected, "descriptor coverage for {kind:?}");
            assert_eq!(
                descriptors.len(),
                actual.len(),
                "duplicate descriptor for {kind:?}"
            );
        }

        // SpatialTransform's arrays are represented as vector/corner controls;
        // no authored field may disappear from Patch/Morph/web coverage.
        let transform = NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == NodeKindTag::Transform)
            .count();
        assert_eq!(transform, 13);
    }

    #[test]
    fn ellipse_mask_serde_and_sanitization_are_bounded() {
        let yaml = "nodes:\n  - stable_id: 3\n    kind:\n      kind: mask\n      params:\n        mask: ellipse\n        params:\n          center: [.nan, 99]\n          radii: [-3, .inf]\n          rotation_deg: 540\n          feather: 9\n          invert: true\nnext_node_id: 4\n";
        let rack = serde_yaml::from_str::<VisualRack>(yaml).unwrap();
        let node = rack.get(NodeId::new(3).unwrap()).unwrap();
        let VisualNodeKind::Mask(MaskParams::Ellipse(ellipse)) = node.kind else {
            panic!("expected ellipse mask");
        };
        assert_eq!(ellipse.center, [0.5, 3.0]);
        assert_eq!(ellipse.radii, [0.0, 0.5]);
        assert_eq!(ellipse.rotation_deg, 180.0);
        assert_eq!(ellipse.feather, 1.0);
        assert!(ellipse.invert);
        let json = serde_json::to_string(&rack).unwrap();
        assert_eq!(serde_json::from_str::<VisualRack>(&json).unwrap(), rack);
    }

    #[test]
    fn current_cycle_rejects_but_the_same_delayed_cycle_is_legal() {
        let scopes = [scope(1), scope(2)];
        let current = [
            ImageDependency {
                consumer: scope(1),
                producer: scope(2),
                timing: EdgeTiming::CurrentFrame,
            },
            ImageDependency {
                consumer: scope(2),
                producer: scope(1),
                timing: EdgeTiming::CurrentFrame,
            },
        ];
        assert!(matches!(
            ImageDependencyGraph::validate(&scopes, &current, ImageGraphMode::Advanced),
            Err(ImageGraphError::CurrentCycle { .. })
        ));

        let delayed = [
            current[0],
            ImageDependency {
                timing: EdgeTiming::PreviousFrame,
                ..current[1]
            },
        ];
        let plan =
            ImageDependencyGraph::validate(&scopes, &delayed, ImageGraphMode::Advanced).unwrap();
        assert_eq!(plan.current_taps, 1);
        assert_eq!(plan.previous_taps, 1);
    }

    #[test]
    fn graph_is_general_not_restricted_to_positional_monotonicity() {
        let scopes = [scope(1), scope(2), scope(3)];
        let edges = [
            ImageDependency {
                consumer: scope(1),
                producer: scope(3),
                timing: EdgeTiming::CurrentFrame,
            },
            ImageDependency {
                consumer: scope(2),
                producer: scope(1),
                timing: EdgeTiming::CurrentFrame,
            },
        ];
        let plan =
            ImageDependencyGraph::validate(&scopes, &edges, ImageGraphMode::Advanced).unwrap();
        let index = |needle| {
            plan.current_topological_order
                .iter()
                .position(|scope| *scope == needle)
                .unwrap()
        };
        assert!(index(scope(3)) < index(scope(1)));
        assert!(index(scope(1)) < index(scope(2)));
    }

    #[test]
    fn same_frame_program_is_advanced_error_and_legacy_transparent_diagnostic() {
        let edge = [ImageDependency {
            consumer: scope(1),
            producer: VisualScopeId::Program,
            timing: EdgeTiming::CurrentFrame,
        }];
        assert!(matches!(
            ImageDependencyGraph::validate(&[scope(1)], &edge, ImageGraphMode::Advanced),
            Err(ImageGraphError::CurrentProgramInput { .. })
        ));
        let legacy = ImageDependencyGraph::validate(
            &[scope(1)],
            &edge,
            ImageGraphMode::LegacyM1TransparentProgram,
        )
        .unwrap();
        assert_eq!(legacy.current_taps, 0);
        assert_eq!(
            legacy.diagnostics,
            vec![ImageGraphDiagnostic::LegacyCurrentProgramTransparent { consumer: scope(1) }]
        );
    }

    #[test]
    fn previous_tap_cap_is_enforced() {
        let scopes = [scope(1), scope(2)];
        let edges = vec![
            ImageDependency {
                consumer: scope(1),
                producer: scope(2),
                timing: EdgeTiming::PreviousFrame,
            };
            MAX_PREVIOUS_IMAGE_TAPS + 1
        ];
        assert!(matches!(
            ImageDependencyGraph::validate(&scopes, &edges, ImageGraphMode::Advanced),
            Err(ImageGraphError::TooManyPreviousTaps { .. })
        ));
    }

    #[test]
    fn ordering_edges_detect_cycles_without_inflating_logical_tap_count() {
        let scopes = [scope(1), scope(2), scope(3)];
        let logical = [ImageDependency {
            consumer: scope(3),
            producer: scope(2),
            timing: EdgeTiming::CurrentFrame,
        }];
        let ordering = [
            ImageOrderingEdge {
                producer: scope(1),
                consumer: scope(3),
            },
            // Duplicate of the logical relation must remain one DAG edge.
            ImageOrderingEdge {
                producer: scope(2),
                consumer: scope(3),
            },
        ];
        let plan = ImageDependencyGraph::validate_with_ordering_edges(
            &scopes,
            &logical,
            &ordering,
            ImageGraphMode::Advanced,
        )
        .unwrap();
        assert_eq!(plan.current_taps, 1);

        let cycle = [ImageOrderingEdge {
            producer: scope(3),
            consumer: scope(2),
        }];
        assert!(matches!(
            ImageDependencyGraph::validate_with_ordering_edges(
                &scopes,
                &logical,
                &cycle,
                ImageGraphMode::Advanced,
            ),
            Err(ImageGraphError::CurrentCycle { .. })
        ));
    }

    #[test]
    fn resource_preflight_is_checked_and_bounded() {
        let graph =
            ImageDependencyGraph::validate(&[scope(1)], &[], ImageGraphMode::Advanced).unwrap();
        let plan = CreativeResourcePlan::preflight(
            [1920, 1080],
            &[VisualRack::synthetic_legacy(LegacyRackScope::Master)],
            &graph,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.retained_surface_layers, BASE_CREATIVE_SURFACE_LAYERS);
        assert!(plan.creative_bytes > 0);

        let hostile = CreativeResourcePlan::preflight(
            [u32::MAX, u32::MAX],
            &[],
            &graph,
            CreativeResourceLimits {
                max_texture_dimension_2d: u32::MAX,
                max_texture_array_layers: u32::MAX,
                max_sampled_textures_per_shader_stage: u32::MAX,
                max_creative_bytes: u64::MAX,
            },
        );
        assert_eq!(hostile, Err(ResourcePreflightError::ArithmeticOverflow));
    }

    #[test]
    fn legacy_logical_lookup_1025_rejects_while_advanced_128_shader_ops_fit() {
        assert_eq!(
            validate_frame_texture_budgets(MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME, 1_024),
            Ok(())
        );
        assert_eq!(
            validate_frame_texture_budgets(MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME + 1, 1_025),
            Err(ResourcePreflightError::FrameLogicalLookupBudget {
                lookups: 1_025,
                limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME,
            }),
            "the exact first Legacy-equivalent lookup over the frozen frame ceiling must reject"
        );
        assert_eq!(
            validate_frame_texture_budgets(
                MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK,
                MAX_TEXTURE_SAMPLES_PER_RACK,
            ),
            Ok(()),
            "32 logical Advanced lookups expanded to 128 shader operations must remain admissible"
        );
    }

    #[test]
    fn logical_lookup_ceiling_is_frozen_while_advanced_shader_ops_are_explicit() {
        let graph =
            ImageDependencyGraph::validate(&[scope(1)], &[], ImageGraphMode::Advanced).unwrap();
        let legacy_layer = VisualRack::synthetic_legacy(LegacyRackScope::Layer);

        let legacy_at_1_020 = vec![legacy_layer.clone(); 85];
        let admitted = CreativeResourcePlan::preflight(
            [1, 1],
            &legacy_at_1_020,
            &graph,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(admitted.logical_texture_lookups_per_pixel, 1_020);

        let legacy_at_1_032 = vec![legacy_layer; 86];
        assert_eq!(
            CreativeResourcePlan::preflight(
                [1, 1],
                &legacy_at_1_032,
                &graph,
                CreativeResourceLimits::default(),
            ),
            Err(ResourcePreflightError::FrameLogicalLookupBudget {
                lookups: 1_032,
                limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME,
            })
        );

        let mut digital = VisualRack::empty();
        digital
            .push(VisualNodeKind::DigitalColor(DigitalColorParams::default()))
            .unwrap();
        let mut grain = VisualRack::empty();
        grain
            .push(VisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        let mut exact_boundary = legacy_at_1_020;
        exact_boundary.push(digital);
        let admitted = CreativeResourcePlan::preflight(
            [1, 1],
            &exact_boundary,
            &graph,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(
            admitted.logical_texture_lookups_per_pixel,
            MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME
        );
        exact_boundary.push(grain);
        assert_eq!(
            CreativeResourcePlan::preflight(
                [1, 1],
                &exact_boundary,
                &graph,
                CreativeResourceLimits::default(),
            ),
            Err(ResourcePreflightError::FrameLogicalLookupBudget {
                lookups: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME + 1,
                limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME,
            })
        );
    }

    #[test]
    fn temporal_originals_preflight_needs_no_new_textures_but_requires_three_bindings() {
        let graph =
            ImageDependencyGraph::validate(&[scope(1)], &[], ImageGraphMode::Advanced).unwrap();
        let rack = VisualRack::synthetic_legacy(LegacyRackScope::Master);
        let admitted = CreativeResourcePlan::preflight(
            [320, 180],
            std::slice::from_ref(&rack),
            &graph,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(
            admitted.retained_surface_layers,
            BASE_CREATIVE_SURFACE_LAYERS
        );
        assert_eq!(admitted.texture_samples_per_pixel, 21);

        assert!(matches!(
            CreativeResourcePlan::preflight(
                [320, 180],
                &[rack],
                &graph,
                CreativeResourceLimits {
                    max_sampled_textures_per_shader_stage: 2,
                    ..CreativeResourceLimits::default()
                },
            ),
            Err(ResourcePreflightError::SampledTextureLimit {
                requested: 3,
                limit: 2,
            })
        ));
    }

    #[test]
    fn group_ab_program_and_history_are_all_charged_in_one_ledger() {
        let graph = ImageDependencyGraph::validate(
            &[scope(1)],
            &[ImageDependency {
                consumer: scope(1),
                producer: VisualScopeId::Program,
                timing: EdgeTiming::PreviousFrame,
            }],
            ImageGraphMode::Advanced,
        )
        .unwrap();
        let plan = CreativeResourcePlan::preflight(
            [320, 180],
            &[VisualRack::empty()],
            &graph,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.retained_surface_layers, 7);
        assert_eq!(plan.creative_bytes, 320_u64 * 180 * 8 * 7);
    }

    #[test]
    fn mixed_host_ledger_admits_1080p_and_rejects_4k_before_allocation() {
        let graph =
            ImageDependencyGraph::validate(&[scope(1)], &[], ImageGraphMode::Advanced).unwrap();
        let plan = CreativeResourcePlan::preflight_with_surface_formats(
            [1920, 1080],
            &[VisualRack::empty()],
            &graph,
            ADVANCED_RACK_SURFACE_LAYERS,
            ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(
            plan.rgba16_surface_layers,
            BASE_CREATIVE_SURFACE_LAYERS + ADVANCED_RACK_SURFACE_LAYERS
        );
        assert_eq!(
            plan.compat8_surface_layers,
            ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS
        );
        assert_eq!(
            plan.creative_bytes,
            1920_u64
                * 1080
                * ((BASE_CREATIVE_SURFACE_LAYERS + ADVANCED_RACK_SURFACE_LAYERS) as u64 * 8
                    + ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS as u64 * 4)
        );
        assert!(plan.creative_bytes < MAX_CREATIVE_GPU_BYTES);

        assert!(matches!(
            CreativeResourcePlan::preflight_with_surface_formats(
                [3840, 2160],
                &[VisualRack::empty()],
                &graph,
                ADVANCED_RACK_SURFACE_LAYERS,
                ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS,
                CreativeResourceLimits::default(),
            ),
            Err(ResourcePreflightError::CreativeMemoryBudget { .. })
        ));
    }

    fn saved_image_mask_node(id: u64, source: SavedImageSource) -> VisualNode {
        VisualNode::authored(
            NodeId::new(id).unwrap(),
            VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
                tap: SavedImageTap {
                    source,
                    timing: EdgeTiming::CurrentFrame,
                },
                ..ImageMatte::default()
            })),
        )
    }

    #[test]
    fn runtime_selected_route_follows_stable_donor_through_reorder_and_capture() {
        let saved = VisualRack::try_from_parts(
            vec![saved_image_mask_node(
                3,
                SavedImageSource::SelectedLayer {
                    layer_position: saved_position(1),
                    stage: LayerImageStage::PostLocalEffects,
                },
            )],
            Some(4),
        )
        .unwrap();
        let runtime = saved.resolve_routes(
            |position| (position == saved_position(1)).then(|| live_id(42)),
            |_| false,
        );
        assert_eq!(
            runtime.image_mask_route(NodeId::new(3).unwrap()),
            Some(ResolvedImageTap {
                source: ResolvedImageSource::SelectedLayer {
                    layer_id: live_id(42),
                    saved_position: saved_position(1),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::CurrentFrame,
            })
        );

        // The live layer moved from saved position 1 to 7. Capture asks by
        // stable ID and writes only the new saved position, never process ID 42.
        let captured = runtime
            .capture_routes(|layer_id| (layer_id == live_id(42)).then(|| saved_position(7)))
            .unwrap();
        let VisualNodeKind::Mask(MaskParams::Image(matte)) =
            captured.get(NodeId::new(3).unwrap()).unwrap().kind
        else {
            panic!("expected captured image mask");
        };
        assert_eq!(
            matte.tap.source,
            SavedImageSource::SelectedLayer {
                layer_position: saved_position(7),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(captured.topology_signature(), saved.topology_signature());
        let json = serde_json::to_string(&captured).unwrap();
        assert!(json.contains("\"layer_position\":7"));
        assert!(!json.contains("layer_id"));
    }

    #[test]
    fn deleted_runtime_donors_remain_explicitly_missing_through_roundtrip() {
        let group_id = GroupId::new(9).unwrap();
        let saved = VisualRack::try_from_parts(
            vec![
                saved_image_mask_node(
                    3,
                    SavedImageSource::SelectedLayer {
                        layer_position: saved_position(1),
                        stage: LayerImageStage::PreLocalEffects,
                    },
                ),
                saved_image_mask_node(4, SavedImageSource::GroupOutput { group_id }),
            ],
            Some(5),
        )
        .unwrap();
        let mut runtime = saved.resolve_routes(|_| Some(live_id(42)), |_| true);
        runtime.mark_layer_output_missing(live_id(42));
        runtime.mark_group_output_missing(group_id);
        let captured = runtime.capture_routes(|_| Some(saved_position(8))).unwrap();
        let sources: Vec<_> = captured
            .iter()
            .map(|node| {
                let VisualNodeKind::Mask(MaskParams::Image(matte)) = node.kind else {
                    panic!("expected image mask");
                };
                matte.tap.source
            })
            .collect();
        assert_eq!(
            sources,
            vec![
                SavedImageSource::MissingSelectedLayer {
                    saved_position: saved_position(1),
                    stage: LayerImageStage::PreLocalEffects,
                },
                SavedImageSource::MissingGroupOutput { group_id },
            ]
        );

        // Even if replacements now exist at the old position/ID lookup, the
        // explicit missing variants cannot silently reconnect.
        let re_resolved = captured.resolve_routes(|_| Some(live_id(99)), |_| true);
        assert!(matches!(
            re_resolved.image_mask_route(NodeId::new(3).unwrap()),
            Some(ResolvedImageTap {
                source: ResolvedImageSource::MissingSelectedLayer { .. },
                ..
            })
        ));
        assert!(matches!(
            re_resolved.image_mask_route(NodeId::new(4).unwrap()),
            Some(ResolvedImageTap {
                source: ResolvedImageSource::MissingGroupOutput(id),
                ..
            }) if id == group_id
        ));
    }

    #[test]
    fn runtime_routes_remain_keyed_by_node_id_across_move_remove_and_insert() {
        let saved = VisualRack::try_from_parts(
            vec![
                saved_image_mask_node(
                    3,
                    SavedImageSource::SelectedLayer {
                        layer_position: saved_position(1),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                ),
                saved_image_mask_node(
                    4,
                    SavedImageSource::SelectedLayer {
                        layer_position: saved_position(2),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                ),
            ],
            Some(5),
        )
        .unwrap();
        let mut runtime = saved.resolve_routes(
            |position| Some(live_id(u64::from(position.get()) + 100)),
            |_| false,
        );
        let replacement_route = ResolvedImageTap {
            source: ResolvedImageSource::SelectedLayer {
                layer_id: live_id(303),
                saved_position: saved_position(3),
                stage: LayerImageStage::PreLocalEffects,
            },
            timing: EdgeTiming::PreviousFrame,
        };
        runtime
            .set_image_mask_route(NodeId::new(4).unwrap(), replacement_route)
            .unwrap();
        runtime
            .move_node(NodeId::new(4).unwrap(), 0, LegacyRackScope::Group)
            .unwrap();
        assert_eq!(
            runtime.iter().next().unwrap().stable_id,
            NodeId::new(4).unwrap()
        );
        assert_eq!(
            runtime.image_mask_route(NodeId::new(4).unwrap()),
            Some(replacement_route)
        );
        assert_ne!(
            runtime.image_mask_route(NodeId::new(3).unwrap()),
            Some(replacement_route)
        );
        assert_eq!(
            runtime
                .node_descriptor(NodeId::new(4).unwrap())
                .unwrap()
                .tag,
            NodeKindTag::Mask
        );
        assert!(runtime
            .parameter_descriptors(NodeId::new(4).unwrap())
            .any(|descriptor| descriptor.key == "image_tap"));
        assert_eq!(runtime.resource_budget().unwrap().cross_input_taps, 2);

        assert_eq!(
            runtime.remove(NodeId::new(3).unwrap()).unwrap().stable_id,
            NodeId::new(3).unwrap()
        );
        let inserted = runtime
            .push(RuntimeVisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        assert_eq!(inserted, NodeId::new(5).unwrap());
        while runtime.len() < MAX_NODES_PER_RACK {
            runtime
                .push(RuntimeVisualNodeKind::Grain(GrainParams::default()))
                .unwrap();
        }
        assert!(matches!(
            runtime.push(RuntimeVisualNodeKind::Grain(GrainParams::default())),
            Err(RuntimeRackError::InvalidRack(
                RackError::TooManyNodes { .. }
            ))
        ));
    }

    #[test]
    fn runtime_legacy_topology_and_resource_contract_match_saved_rack() {
        for scope in [
            LegacyRackScope::Layer,
            LegacyRackScope::Master,
            LegacyRackScope::Group,
        ] {
            let saved = VisualRack::synthetic_legacy(scope);
            let runtime = saved.resolve_routes(|_| None, |_| false);
            assert!(runtime.is_exact_legacy(scope));
            assert_eq!(runtime.topology_signature(), saved.topology_signature());
            assert_eq!(
                runtime.resource_budget().unwrap(),
                saved.resource_budget().unwrap()
            );
            assert_eq!(runtime.capture_routes(|_| None).unwrap(), saved);
        }
    }

    #[test]
    fn missing_group_reference_remains_missing_and_never_retargets() {
        let group = GroupId::new(9).unwrap();
        let saved = SavedImageTap {
            source: SavedImageSource::GroupOutput { group_id: group },
            timing: EdgeTiming::PreviousFrame,
        };
        let runtime = saved.to_runtime(|_| None, |_| false);
        assert_eq!(
            runtime,
            ResolvedImageTap {
                source: ResolvedImageSource::MissingGroupOutput(group),
                timing: EdgeTiming::PreviousFrame,
            }
        );
    }
}
