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
use crate::scan_processor::ScanProcessorParams;
use crate::spatial::SpatialTransform;
use crate::symmetry::{RuntimeSymmetryParams, SymmetryParams};

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
/// The fixed Collision Rack bind layout remains capped at three sampled
/// textures. Dedicated creative passes are split by the composition planner
/// and preflight against their own declared ceiling.
pub const MAX_SAMPLED_TEXTURES_PER_PASS: u32 = 3;
/// A dedicated creative pass owns its own bind layout and therefore its own,
/// separate ceiling. Eight simultaneously sampled textures in one pass was
/// proven portable under the production device floor
/// (`Limits::default()`, `src/renderer/state.rs`) by the S2 probe, whose
/// receipt `s2-eight-texture-floor-receipt.json` records the enforced-cap
/// argument: the floor guarantees sixteen, and a seventeen-texture layout was
/// refused on the same device. This ceiling is independent of
/// [`MAX_SAMPLED_TEXTURES_PER_PASS`]; raising either one never raises the
/// other, and the fixed Collision Rack layout stays capped at three.
pub const MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS: u32 = 8;
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
    /// The etched gesture field, presented as a premultiplied donor image.
    ///
    /// The canvas is a master-scope singleton: there is exactly one live field
    /// and it is not a layer, a group, or a composite prefix. It therefore has
    /// **no saved position to preserve** — deliberately, not by omission. A
    /// route to it survives every reorder, deletion, and insertion unchanged,
    /// and when no canvas is admitted it resolves to a transparent field with
    /// a named diagnostic rather than degrading into some other producer.
    GestureCanvas,
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
    /// The etched gesture field. A master-scope singleton with no saved
    /// position: see `SavedImageSource::GestureCanvas`.
    GestureCanvas,
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
            // The singleton has no position and no identity to look up, so the
            // route survives verbatim in both directions.
            SavedImageSource::GestureCanvas => ResolvedImageSource::GestureCanvas,
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
            ResolvedImageSource::GestureCanvas => SavedImageSource::GestureCanvas,
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

/// Boundary law for carrier coordinates pushed outside the source domain by a
/// Displace vector. `Transparent` is the authored default and the only law that
/// removes coverage; the other three keep the sample opaque by remapping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplaceBoundary {
    #[default]
    Transparent,
    Mirror,
    Wrap,
    Hold,
}

impl DisplaceBoundary {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Transparent => 0,
            Self::Mirror => 1,
            Self::Wrap => 2,
            Self::Hold => 3,
        }
    }
}

/// Authored state of the named two-input Displace node. The donor is a stable
/// image tap under the generic tap laws; the amounts are independent finite UV
/// gains; the boundary is stable authored topology rather than a morphable
/// value. Neutral donor encoding is `RG = 0.5`, so a transparent or missing
/// donor is exact zero displacement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaceParams {
    pub tap: SavedImageTap,
    pub amount_x: f32,
    pub amount_y: f32,
    pub boundary: DisplaceBoundary,
}

impl DisplaceParams {
    fn sanitized(self) -> Self {
        Self {
            tap: self.tap,
            amount_x: finite_clamp(self.amount_x, 0.0, -1.0, 1.0),
            amount_y: finite_clamp(self.amount_y, 0.0, -1.0, 1.0),
            boundary: self.boundary,
        }
    }

    /// Zero gain on both axes is an authored no-op. The planner then collects
    /// no donor and the executor encodes no pass, so an exact-default Displace
    /// delegates before allocating or encoding anything. Hostile non-finite
    /// input sanitizes to zero and is therefore also an exact bypass.
    pub fn is_exact_bypass(self) -> bool {
        let sanitized = self.sanitized();
        sanitized.amount_x == 0.0 && sanitized.amount_y == 0.0
    }

    pub const fn selected_layer_position(self) -> Option<SavedLayerPosition> {
        self.tap.selected_layer_position()
    }

    pub const fn referenced_group(self) -> Option<GroupId> {
        self.tap.referenced_group()
    }

    /// Preserve a deleted group identity explicitly so a future group at the
    /// same root position cannot inherit this donor route.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved displace invalidation supports patch/editor migrations"
        )
    )]
    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        if self.tap.source == (SavedImageSource::GroupOutput { group_id: removed }) {
            self.tap.source = SavedImageSource::MissingGroupOutput { group_id: removed };
        }
    }
}

/// Persisted schema stamp for the Residual Counterpoint decomposition. It is a
/// version of the algorithm, not an authored value: sanitization always
/// normalizes it to the current constant so an older or hostile patch cannot
/// select a decomposition law this build does not implement.
pub const RESIDUAL_ALGORITHM_VERSION: u16 = 1;
/// Every single-route node kind occupies this one slot. Naming it keeps a
/// one-route consumer identity explicit rather than an unexplained literal.
pub const RACK_PRIMARY_ROUTE_SLOT: u8 = 0;
/// Residual carries exactly two authored image routes. The slot index is the
/// route identity everywhere a consumer must distinguish them.
pub const RESIDUAL_ROUTE_SLOTS: usize = 2;
/// Slot 0 supplies the large-scale structure whose block mean becomes DC.
pub const RESIDUAL_STRUCTURE_SLOT: u8 = 0;
/// Slot 1 supplies the large scale the carrier's detail is measured against.
pub const RESIDUAL_DETAIL_SLOT: u8 = 1;

/// Fixed block-mean grid vocabulary. The edge is in output pixels and the grid
/// is `ceil(output_dim / edge)`, so a smaller block is a larger, more expensive
/// grid. Codes are permanent and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualBlock {
    Four,
    #[default]
    Eight,
    Sixteen,
    ThirtyTwo,
    SixtyFour,
}

impl ResidualBlock {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Four => 0,
            Self::Eight => 1,
            Self::Sixteen => 2,
            Self::ThirtyTwo => 3,
            Self::SixtyFour => 4,
        }
    }

    /// Block edge in output pixels. The vocabulary is closed so a grid can be
    /// preflighted against its own byte and cell limits before any allocation.
    pub const fn edge(self) -> u32 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
            Self::ThirtyTwo => 32,
            Self::SixtyFour => 64,
        }
    }
}

/// Fixed quantization vocabulary applied to both the DC and the AC term.
/// Codes are permanent and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualQuantization {
    #[default]
    Off,
    Coarse,
    Medium,
    Fine,
}

impl ResidualQuantization {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Coarse => 1,
            Self::Medium => 2,
            Self::Fine => 3,
        }
    }

    /// Level count of the seeded lattice. Zero is the authored default and
    /// means exact identity, not a one-level collapse.
    pub const fn levels(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Coarse => 8,
            Self::Medium => 32,
            Self::Fine => 128,
        }
    }
}

/// Authored state of the named two-input Residual Counterpoint node. The node
/// recombines one route's large-scale structure with the carrier's detail
/// measured against a second route:
///
/// ```text
/// dc  = quantize(mean(structure))
/// ac  = quantize(carrier_premultiplied - mean(detail))
/// out = dc + detail_gain * ac
/// ```
///
/// Both routes are stable image taps under the generic tap laws and are read
/// only through their reduced block means, never at full resolution. `mix` is
/// the wet/dry authority: zero is the authored default and an exact bypass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResidualParams {
    pub algorithm_version: u16,
    pub structure: SavedImageTap,
    pub detail: SavedImageTap,
    pub block: ResidualBlock,
    pub quantization: ResidualQuantization,
    pub mix: f32,
    pub detail_gain: f32,
    pub seed: u32,
}

impl Default for ResidualParams {
    fn default() -> Self {
        Self {
            algorithm_version: RESIDUAL_ALGORITHM_VERSION,
            structure: SavedImageTap::default(),
            detail: SavedImageTap::default(),
            block: ResidualBlock::Eight,
            quantization: ResidualQuantization::Off,
            mix: 0.0,
            detail_gain: 1.0,
            seed: 0,
        }
    }
}

impl ResidualParams {
    fn sanitized(self) -> Self {
        Self {
            algorithm_version: RESIDUAL_ALGORITHM_VERSION,
            structure: self.structure,
            detail: self.detail,
            block: self.block,
            quantization: self.quantization,
            mix: finite_clamp(self.mix, 0.0, 0.0, 1.0),
            detail_gain: finite_clamp(self.detail_gain, 1.0, 0.0, 4.0),
            seed: self.seed,
        }
    }

    /// Zero mix is an authored no-op. The planner then collects neither route,
    /// the executor encodes no pass and allocates no block-mean surface, and
    /// the saved-patch dependency walk claims no edge. Hostile non-finite mix
    /// sanitizes to zero and is therefore also an exact bypass, never a full
    /// recombination.
    pub fn is_exact_bypass(self) -> bool {
        self.sanitized().mix == 0.0
    }

    /// Slot 0 is `structure` and slot 1 is `detail`. Any other index names no
    /// route at all: it yields `None` rather than silently aliasing a real
    /// slot, so an unknown remote slot is rejected instead of misapplied.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "slot-indexed saved route access is consumed by the ordered route action"
        )
    )]
    pub const fn route(self, slot: u8) -> Option<SavedImageTap> {
        match slot {
            RESIDUAL_STRUCTURE_SLOT => Some(self.structure),
            RESIDUAL_DETAIL_SLOT => Some(self.detail),
            _ => None,
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "slot-indexed saved route access is consumed by the ordered route action"
        )
    )]
    pub fn route_mut(&mut self, slot: u8) -> Option<&mut SavedImageTap> {
        match slot {
            RESIDUAL_STRUCTURE_SLOT => Some(&mut self.structure),
            RESIDUAL_DETAIL_SLOT => Some(&mut self.detail),
            _ => None,
        }
    }

    /// Both routes in slot order. Route walkers iterate this fixed array so a
    /// second slot can never be dropped by a one-value closure.
    pub const fn routes(self) -> [SavedImageTap; RESIDUAL_ROUTE_SLOTS] {
        [self.structure, self.detail]
    }

    pub const fn selected_layer_positions(
        self,
    ) -> [Option<SavedLayerPosition>; RESIDUAL_ROUTE_SLOTS] {
        [
            self.structure.selected_layer_position(),
            self.detail.selected_layer_position(),
        ]
    }

    pub const fn referenced_groups(self) -> [Option<GroupId>; RESIDUAL_ROUTE_SLOTS] {
        [
            self.structure.referenced_group(),
            self.detail.referenced_group(),
        ]
    }

    /// Preserve a deleted group identity explicitly in every slot so a future
    /// group at the same root position cannot inherit either route.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved residual invalidation supports patch/editor migrations"
        )
    )]
    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        for route in [&mut self.structure, &mut self.detail] {
            if route.source == (SavedImageSource::GroupOutput { group_id: removed }) {
                route.source = SavedImageSource::MissingGroupOutput { group_id: removed };
            }
        }
    }
}

/// Hard edge bound for either dimension of one block-mean grid.
pub const RESIDUAL_GRID_MAX_EDGE: u32 = 2_048;
/// Hard cell bound for one node's block-mean grid.
pub const RESIDUAL_GRID_MAX_CELLS: u64 = 2_100_000;
/// Exact storage cost of one block-mean cell: a single `Rgba16Float` texel.
/// This is an equality, not a ceiling — a wider cell is a different plan.
pub const RESIDUAL_MEAN_BYTES_PER_CELL: u64 = 8;
/// Exact number of block-mean surfaces one active node owns: one per authored
/// route slot, structure and detail. Also an equality, not a ceiling.
pub const RESIDUAL_MEAN_SURFACES_PER_NODE: u32 = 2;
/// Explicit texture operations per mean cell: the four quadrant centres of its
/// block. This is a bounded four-tap estimator, not a full box integral.
pub const RESIDUAL_MEAN_TAPS_PER_CELL: u64 = 4;
/// Hard byte bound for one node's complete block-mean working set. The cell
/// cap alone does not imply it: a full-cap grid is 33,600,000 bytes across two
/// surfaces, so this bound binds first.
pub const RESIDUAL_NODE_MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Composition-wide block-mean working-set ceiling.
pub const RESIDUAL_AGGREGATE_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// The recombination pass binds the carrier plus both block means, and never
/// either route at full resolution.
pub const RESIDUAL_RECOMBINATION_SAMPLED_TEXTURES: u32 = 3;
/// Frozen dynamic-offset stride of one recombination uniform record.
pub const RESIDUAL_UNIFORM_STRIDE_BYTES: u64 = 256;
/// Nominal ceiling on simultaneously active Residual nodes. The aggregate byte
/// cap can bind well before it and is checked independently.
pub const RESIDUAL_MAX_ACTIVE_NODES: u32 = 16;

/// Typed rejection from Residual Counterpoint resource admission. Every §2
/// bound is a separate variant so an over-budget grid names the bound it broke
/// instead of being silently clamped to a smaller one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualResourceError {
    InvalidDimensions([u32; 2]),
    DeviceTextureDimension {
        dimensions: [u32; 2],
        limit: u32,
    },
    GridEdge {
        dimensions: [u32; 2],
        limit: u32,
    },
    CellCount {
        count: u64,
        limit: u64,
    },
    NodeBytes {
        bytes: u64,
        limit: u64,
    },
    AggregateBytes {
        bytes: u64,
        limit: u64,
    },
    TooManyActiveNodes {
        count: u32,
        limit: u32,
    },
    SampledTextures {
        requested: u32,
        limit: u32,
    },
    UniformStride {
        stride: u64,
        alignment: u32,
    },
    /// Reconciliation: the executor allocated a cell wider or narrower than the
    /// exact `Rgba16Float` texel the plan charged for.
    AllocatedCellBytes {
        allocated: u64,
        expected: u64,
    },
    /// Reconciliation: the executor owns a different number of block-mean
    /// surfaces per node than the exact two the plan charged for.
    AllocatedSurfacesPerNode {
        allocated: u32,
        expected: u32,
    },
    /// Reconciliation: the executor's uniform arena stride is not the frozen
    /// dynamic-offset stride the plan declared.
    AllocatedUniformStride {
        allocated: u64,
        expected: u64,
    },
    /// Reconciliation: declared and physical byte totals disagree.
    AllocatedBytes {
        allocated: u64,
        planned: u64,
    },
    ArithmeticOverflow,
}

impl fmt::Display for ResidualResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions([width, height]) => write!(
                formatter,
                "residual block-mean grid needs positive dimensions, got {width}x{height}"
            ),
            Self::DeviceTextureDimension { dimensions, limit } => write!(
                formatter,
                "residual carrier {}x{} exceeds this device's {limit}px texture limit",
                dimensions[0], dimensions[1]
            ),
            Self::GridEdge { dimensions, limit } => write!(
                formatter,
                "residual block-mean grid {}x{} exceeds the {limit} cell edge bound",
                dimensions[0], dimensions[1]
            ),
            Self::CellCount { count, limit } => write!(
                formatter,
                "residual block-mean grid holds {count} cells, above the {limit} cell bound"
            ),
            Self::NodeBytes { bytes, limit } => write!(
                formatter,
                "one residual node's block means need {bytes} bytes, above the {limit} byte bound"
            ),
            Self::AggregateBytes { bytes, limit } => write!(
                formatter,
                "residual block means need {bytes} bytes, above the {limit} byte composition bound"
            ),
            Self::TooManyActiveNodes { count, limit } => write!(
                formatter,
                "{count} active residual nodes exceed the nominal {limit} node bound"
            ),
            Self::SampledTextures { requested, limit } => write!(
                formatter,
                "residual recombination samples {requested} textures, above this pass's {limit}"
            ),
            Self::UniformStride { stride, alignment } => write!(
                formatter,
                "residual uniform stride {stride} is not a multiple of this device's {alignment} byte dynamic-offset alignment"
            ),
            Self::AllocatedCellBytes {
                allocated,
                expected,
            } => write!(
                formatter,
                "residual block-mean cells allocate {allocated} bytes rather than exactly {expected}"
            ),
            Self::AllocatedSurfacesPerNode {
                allocated,
                expected,
            } => write!(
                formatter,
                "residual nodes allocate {allocated} block-mean surfaces rather than exactly {expected}"
            ),
            Self::AllocatedUniformStride {
                allocated,
                expected,
            } => write!(
                formatter,
                "residual uniform arena strides {allocated} bytes rather than the declared {expected}"
            ),
            Self::AllocatedBytes { allocated, planned } => write!(
                formatter,
                "residual block means allocate {allocated} bytes while the plan declared {planned}"
            ),
            Self::ArithmeticOverflow => {
                write!(formatter, "residual resource accounting overflowed")
            }
        }
    }
}

/// Canonical block-mean grid for one active node. `ceil(output / block edge)`
/// on each axis, admitted against its own edge and cell bounds before any
/// allocation. An over-budget grid is a typed rejection, never a clamp to a
/// smaller grid that would silently change the authored decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualGrid {
    pub width: u32,
    pub height: u32,
    pub block_pixels: u32,
    pub cell_count: u64,
}

impl ResidualGrid {
    pub fn for_output(
        output_dimensions: [u32; 2],
        block: ResidualBlock,
    ) -> Result<Self, ResidualResourceError> {
        let [output_width, output_height] = output_dimensions;
        if output_width == 0 || output_height == 0 {
            return Err(ResidualResourceError::InvalidDimensions(output_dimensions));
        }
        let block_pixels = block.edge();
        let width = output_width
            .checked_add(block_pixels - 1)
            .ok_or(ResidualResourceError::ArithmeticOverflow)?
            / block_pixels;
        let height = output_height
            .checked_add(block_pixels - 1)
            .ok_or(ResidualResourceError::ArithmeticOverflow)?
            / block_pixels;
        if width > RESIDUAL_GRID_MAX_EDGE || height > RESIDUAL_GRID_MAX_EDGE {
            return Err(ResidualResourceError::GridEdge {
                dimensions: [width, height],
                limit: RESIDUAL_GRID_MAX_EDGE,
            });
        }
        let cell_count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(ResidualResourceError::ArithmeticOverflow)?;
        if cell_count > RESIDUAL_GRID_MAX_CELLS {
            return Err(ResidualResourceError::CellCount {
                count: cell_count,
                limit: RESIDUAL_GRID_MAX_CELLS,
            });
        }
        Ok(Self {
            width,
            height,
            block_pixels,
            cell_count,
        })
    }
}

/// Device facts the block-mean budget needs. Hosts copy these from their
/// adapter; the residual domain never depends on `wgpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualResourceLimits {
    pub max_texture_dimension_2d: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_residual_bytes: u64,
}

impl Default for ResidualResourceLimits {
    fn default() -> Self {
        Self {
            max_texture_dimension_2d: 8_192,
            min_uniform_buffer_offset_alignment: 256,
            max_sampled_textures_per_shader_stage: MAX_SAMPLED_TEXTURES_PER_PASS,
            max_residual_bytes: RESIDUAL_AGGREGATE_MAX_BYTES,
        }
    }
}

/// One active Residual node offered to composition-wide block-mean preflight.
/// Bypassed, disabled, and zero-wet nodes are filtered by the caller under the
/// same admission predicate the planner and the saved-patch walk use, so a
/// delegating node charges nothing here either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualResourceRequest {
    pub output_dimensions: [u32; 2],
    pub block: ResidualBlock,
}

/// Byte-exact block-mean working set, budgeted entirely outside the full-frame
/// layer ledger. Reduced surfaces are sub-full-frame and cannot be honestly
/// expressed as `additional_rgba16_layers`; they meet the creative number only
/// at the shared `MAX_CREATIVE_GPU_BYTES` cap, exactly as motion bytes do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualResourcePlan {
    pub active_nodes: u32,
    pub mean_surfaces: u32,
    pub mean_cells: u64,
    pub mean_sample_operations: u64,
    pub mean_surface_bytes: u64,
    pub total_bytes: u64,
    pub max_grid_dimensions: [u32; 2],
    pub bytes_per_cell: u64,
    pub surfaces_per_node: u32,
    pub sampled_textures_in_recombination: u32,
    pub uniform_stride_bytes: u64,
}

impl Default for ResidualResourcePlan {
    fn default() -> Self {
        Self {
            active_nodes: 0,
            mean_surfaces: 0,
            mean_cells: 0,
            mean_sample_operations: 0,
            mean_surface_bytes: 0,
            total_bytes: 0,
            max_grid_dimensions: [0, 0],
            bytes_per_cell: RESIDUAL_MEAN_BYTES_PER_CELL,
            surfaces_per_node: RESIDUAL_MEAN_SURFACES_PER_NODE,
            sampled_textures_in_recombination: RESIDUAL_RECOMBINATION_SAMPLED_TEXTURES,
            uniform_stride_bytes: RESIDUAL_UNIFORM_STRIDE_BYTES,
        }
    }
}

/// Physical facts read back from a prepared executor. Reconciliation fails
/// closed when they disagree with the declared plan, the way motion resources
/// and the runtime precision ledger already do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualAllocationSnapshot {
    pub mean_surfaces: u32,
    pub bytes_per_cell: u64,
    pub surfaces_per_node: u32,
    pub uniform_stride_bytes: u64,
    pub total_bytes: u64,
}

impl ResidualResourcePlan {
    pub fn preflight(
        requests: &[ResidualResourceRequest],
        limits: ResidualResourceLimits,
    ) -> Result<Self, ResidualResourceError> {
        let mut plan = Self::default();

        // The recombination pass reads the carrier and both reduced means. It
        // never reads either route at full resolution, which is exactly what
        // keeps it inside the fixed three-texture bind layout.
        let texture_limit = limits
            .max_sampled_textures_per_shader_stage
            .min(MAX_SAMPLED_TEXTURES_PER_PASS);
        if RESIDUAL_RECOMBINATION_SAMPLED_TEXTURES > texture_limit {
            return Err(ResidualResourceError::SampledTextures {
                requested: RESIDUAL_RECOMBINATION_SAMPLED_TEXTURES,
                limit: texture_limit,
            });
        }

        // The stride is frozen, so a device whose dynamic-offset alignment does
        // not divide it cannot host the arena at all.
        let alignment = limits.min_uniform_buffer_offset_alignment;
        if alignment == 0 || !RESIDUAL_UNIFORM_STRIDE_BYTES.is_multiple_of(u64::from(alignment)) {
            return Err(ResidualResourceError::UniformStride {
                stride: RESIDUAL_UNIFORM_STRIDE_BYTES,
                alignment,
            });
        }

        for request in requests {
            let dimensions = request.output_dimensions;
            if dimensions[0] == 0 || dimensions[1] == 0 {
                return Err(ResidualResourceError::InvalidDimensions(dimensions));
            }
            if dimensions[0] > limits.max_texture_dimension_2d
                || dimensions[1] > limits.max_texture_dimension_2d
            {
                return Err(ResidualResourceError::DeviceTextureDimension {
                    dimensions,
                    limit: limits.max_texture_dimension_2d,
                });
            }
            let grid = ResidualGrid::for_output(dimensions, request.block)?;
            let node_cells = grid
                .cell_count
                .checked_mul(u64::from(RESIDUAL_MEAN_SURFACES_PER_NODE))
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            let node_bytes = node_cells
                .checked_mul(RESIDUAL_MEAN_BYTES_PER_CELL)
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            if node_bytes > RESIDUAL_NODE_MAX_BYTES {
                return Err(ResidualResourceError::NodeBytes {
                    bytes: node_bytes,
                    limit: RESIDUAL_NODE_MAX_BYTES,
                });
            }
            plan.active_nodes = plan
                .active_nodes
                .checked_add(1)
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            if plan.active_nodes > RESIDUAL_MAX_ACTIVE_NODES {
                return Err(ResidualResourceError::TooManyActiveNodes {
                    count: plan.active_nodes,
                    limit: RESIDUAL_MAX_ACTIVE_NODES,
                });
            }
            plan.mean_surfaces = plan
                .mean_surfaces
                .checked_add(RESIDUAL_MEAN_SURFACES_PER_NODE)
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            plan.mean_cells = plan
                .mean_cells
                .checked_add(node_cells)
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            plan.mean_sample_operations = node_cells
                .checked_mul(RESIDUAL_MEAN_TAPS_PER_CELL)
                .and_then(|operations| plan.mean_sample_operations.checked_add(operations))
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            plan.mean_surface_bytes = plan
                .mean_surface_bytes
                .checked_add(node_bytes)
                .ok_or(ResidualResourceError::ArithmeticOverflow)?;
            plan.max_grid_dimensions[0] = plan.max_grid_dimensions[0].max(grid.width);
            plan.max_grid_dimensions[1] = plan.max_grid_dimensions[1].max(grid.height);
        }

        plan.total_bytes = plan.mean_surface_bytes;
        let byte_limit = limits.max_residual_bytes.min(RESIDUAL_AGGREGATE_MAX_BYTES);
        if plan.total_bytes > byte_limit {
            return Err(ResidualResourceError::AggregateBytes {
                bytes: plan.total_bytes,
                limit: byte_limit,
            });
        }
        Ok(plan)
    }

    /// Fail closed when the physical working set disagrees with the declared
    /// plan. The exact rows of the resource table — eight bytes per cell, two
    /// surfaces per node, the frozen uniform stride — are equalities here.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "block-mean reconciliation is consumed by the prepared executor"
        )
    )]
    pub fn reconcile(
        self,
        actual: ResidualAllocationSnapshot,
    ) -> Result<(), ResidualResourceError> {
        if actual.bytes_per_cell != RESIDUAL_MEAN_BYTES_PER_CELL {
            return Err(ResidualResourceError::AllocatedCellBytes {
                allocated: actual.bytes_per_cell,
                expected: RESIDUAL_MEAN_BYTES_PER_CELL,
            });
        }
        if actual.surfaces_per_node != RESIDUAL_MEAN_SURFACES_PER_NODE {
            return Err(ResidualResourceError::AllocatedSurfacesPerNode {
                allocated: actual.surfaces_per_node,
                expected: RESIDUAL_MEAN_SURFACES_PER_NODE,
            });
        }
        if actual.uniform_stride_bytes != self.uniform_stride_bytes {
            return Err(ResidualResourceError::AllocatedUniformStride {
                allocated: actual.uniform_stride_bytes,
                expected: self.uniform_stride_bytes,
            });
        }
        if actual.mean_surfaces != self.mean_surfaces {
            return Err(ResidualResourceError::AllocatedSurfacesPerNode {
                allocated: actual.mean_surfaces,
                expected: self.mean_surfaces,
            });
        }
        if actual.total_bytes != self.total_bytes {
            return Err(ResidualResourceError::AllocatedBytes {
                allocated: actual.total_bytes,
                planned: self.total_bytes,
            });
        }
        Ok(())
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

/// The data-only Study rack node's authored state: a content-addressed
/// reference to a validated Study document. `VisualNodeKind` is `Copy` and a
/// document is heap content, so the node carries only the 32-byte canonical
/// digest — the same identity `CompiledStudy::canonical_digest` derives —
/// while documents themselves live in the bounded host Study library and
/// travel with patches in their own `studies` section. A digest whose
/// document is absent from the library is a named diagnostic and an inert
/// pass, never a fallback onto another document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StudyRackParams {
    pub document_digest: Option<[u8; 32]>,
}

impl StudyRackParams {
    pub const fn sanitized(self) -> Self {
        self
    }

    /// No document, no pass: the planner emits no dedicated step and the
    /// executor encodes nothing — a real delegation, not a cosmetic one.
    pub const fn is_exact_bypass(self) -> bool {
        self.document_digest.is_none()
    }

    pub fn digest_hex(&self) -> Option<String> {
        self.document_digest
            .map(|digest| digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    pub fn parse_digest_hex(hex: &str) -> Option<[u8; 32]> {
        if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        let mut digest = [0_u8; 32];
        for (index, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
            let text = std::str::from_utf8(chunk).ok()?;
            digest[index] = u8::from_str_radix(text, 16).ok()?;
        }
        Some(digest)
    }
}

impl Serialize for StudyRackParams {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("StudyRackParams", 1)?;
        state.serialize_field("document_digest", &self.digest_hex())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for StudyRackParams {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            #[serde(default)]
            document_digest: Option<String>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let document_digest = match raw.document_digest {
            None => None,
            Some(hex) => Some(StudyRackParams::parse_digest_hex(&hex).ok_or_else(|| {
                serde::de::Error::custom("study document digest must be 64 hex characters")
            })?),
        };
        Ok(Self { document_digest })
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
    Displace(DisplaceParams),
    /// The dedicated eight-texture Symmetry Field. Unlike every kind above it,
    /// this one is not encodable as an ordinary Collision Rack pass: the rack's
    /// fixed bind layout carries two sampled textures, so the composition
    /// planner splits it into its own dedicated step.
    Symmetry(SymmetryParams),
    Residual(ResidualParams),
    /// The data-only Study interpreter, a dedicated pass like the Symmetry
    /// Field: the fixed rack layout cannot bind the clean-history array, so
    /// the planner lifts it into its own step.
    Study(StudyRackParams),
    /// The B1 Scan Processor, a dedicated pass for a different reason than
    /// the two above: it is the tree's first non-fullscreen-triangle pass —
    /// an instanced ribbon per scanline accumulated additively into its own
    /// transient — so it cannot ride the rack's fixed fullscreen layout at
    /// all.
    ScanProcessor(ScanProcessorParams),
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
            Self::Displace(_) => NodeKindTag::Displace,
            Self::Symmetry(_) => NodeKindTag::Symmetry,
            Self::Residual(_) => NodeKindTag::Residual,
            Self::Study(_) => NodeKindTag::Study,
            Self::ScanProcessor(_) => NodeKindTag::ScanProcessor,
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
            Self::Displace(value) => Self::Displace(value.sanitized()),
            Self::Symmetry(value) => Self::Symmetry(value.sanitized()),
            Self::Residual(value) => Self::Residual(value.sanitized()),
            Self::ScanProcessor(value) => Self::ScanProcessor(value.sanitized()),
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
    Displace,
    Symmetry,
    Residual,
    Study,
    ScanProcessor,
}

impl NodeKindTag {
    /// Permanent append-only kind codes. A code identifies a node kind inside
    /// every persisted topology signature, so an existing entry must never be
    /// renumbered or reused; new kinds only append.
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
            Self::Displace => 10,
            Self::Residual => 11,
            Self::Symmetry => 12,
            Self::Study => 13,
            Self::ScanProcessor => 14,
        }
    }

    /// Kinds the composition planner lifts out of ordinary rack segmentation
    /// into their own dedicated pass. Their `sampled_textures_in_pass` is
    /// charged against [`MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS`] instead of
    /// the fixed rack layout's [`MAX_SAMPLED_TEXTURES_PER_PASS`]; the two
    /// ceilings are independent and neither may be raised to admit the other.
    pub const fn occupies_dedicated_pass(self) -> bool {
        match self {
            // The Study interpreter binds the clean-history D2 array beside
            // its carrier and owns its own uniform layout, so like Symmetry
            // it cannot ride an ordinary rack segment. The Scan Processor is
            // dedicated for a stronger reason still: it is instanced ribbon
            // geometry accumulating additively into its own transient, not a
            // fullscreen triangle at all.
            Self::Symmetry | Self::Study | Self::ScanProcessor => true,
            Self::LegacyCanonical
            | Self::LegacyTemporal
            | Self::Transform
            | Self::DigitalColor
            | Self::Key
            | Self::Cellular
            | Self::Shift
            | Self::Grain
            | Self::Mask
            // Residual reads two donors alongside its carrier, which still
            // fits the fixed rack layout's three sampled textures, so it stays
            // an ordinary segmented rack pass.
            | Self::Residual
            | Self::Displace => false,
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
    /// Passes that run over a reduced block grid instead of the full output.
    /// They are charged separately from `full_frame_passes` because their cost
    /// scales with the grid, not with the frame.
    pub reduced_resolution_passes: u8,
    /// Sub-full-frame surfaces the node owns for the lifetime of its plan.
    /// Full-output-resolution layers are charged as RGBA16/Compat8 layers by
    /// the creative resource plan; this field counts only reduced grids, whose
    /// bytes are accounted byte-exactly rather than per output pixel.
    pub reduced_resolution_surfaces: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeKindDescriptor {
    pub tag: NodeKindTag,
    pub key: &'static str,
    pub title: &'static str,
    pub budget: NodeResourceBudget,
}

pub const NODE_KIND_DESCRIPTORS: [NodeKindDescriptor; 14] = [
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Displace,
        key: "displace",
        title: "Displace",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Dry carrier, displaced carrier, and the donor vector field. The
            // donor is filtered manually in premultiplied space like the two
            // carrier lookups, so all three cost four explicit loads each.
            logical_texture_lookups_per_pixel: 3,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 3,
            sampled_textures_in_pass: 2,
            cross_input_taps: 1,
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Residual,
        key: "residual",
        title: "Residual Counterpoint",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // The recombination pass reads the dry carrier, the structure
            // route's block mean, and the detail route's block mean. Each is a
            // four-load premultiplied bilinear, so three logical lookups cost
            // twelve explicit shader texture operations.
            logical_texture_lookups_per_pixel: 3,
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 3,
            // Carrier plus both block-mean surfaces. Both authored routes are
            // read through their means, never at full resolution, so the pass
            // stays inside the fixed three-texture rack bind layout.
            sampled_textures_in_pass: 3,
            cross_input_taps: 2,
            // One reduction pass per route, each writing its own block-mean
            // grid. Neither is a full-output-resolution surface.
            reduced_resolution_passes: 2,
            reduced_resolution_surfaces: 2,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Symmetry,
        key: "symmetry",
        title: "Symmetry Field",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // Dry carrier, the folded source, and one motion vector/gate pair.
            logical_texture_lookups_per_pixel: 4,
            // Ten explicit operations: four loads for the dry carrier, four for
            // the folded source, and one each for the vector and gate lanes,
            // which are read at their own grid resolution without filtering.
            texture_samples_per_pixel: PREMULTIPLIED_BILINEAR_TEXTURE_OPS * 2 + 2,
            // Carrier, donor 0, donor 1, the clean-history D2 array, and a
            // vector/gate pair per motion slot: eight simultaneous bindings in
            // one dedicated pass.
            sampled_textures_in_pass: 8,
            // The two fixed image slots. Motion slots are admitted through the
            // motion planner's donor flags, not as cross-scope image taps.
            cross_input_taps: 2,
            // The Symmetry Field is one full-output dedicated pass. It owns no
            // reduced block grid and no persistent sub-frame surface, so both
            // reduced-resolution ledgers stay at zero.
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::Study,
        key: "study",
        title: "Study",
        budget: NodeResourceBudget {
            full_frame_passes: 1,
            // The declared admission budget, not the ABI worst case: one
            // carrier load plus up to seven history loads, each a single
            // textureLoad with no bilinear expansion. LoadCurrentColor reads
            // the already-loaded carrier register and costs nothing. A valid
            // document whose history loads exceed this budget stays valid
            // ABI but is refused at plan time by name — the over-budget
            // Residual-grid law, never a silent clamp.
            logical_texture_lookups_per_pixel: 8,
            texture_samples_per_pixel: 8,
            // Carrier plus the committed clean-history D2 array; two
            // simultaneous bindings in one dedicated pass, no sampler.
            sampled_textures_in_pass: 2,
            // A Study reads only its carrier and the master ring: no image
            // taps, no donors, no routes — the whole tombstone/route surface
            // is structurally absent in ABI 1.0.
            cross_input_taps: 0,
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
        },
    },
    NodeKindDescriptor {
        tag: NodeKindTag::ScanProcessor,
        key: "scan_processor",
        title: "Scan Processor",
        budget: NodeResourceBudget {
            // The instanced geometry pass into the transient accumulator plus
            // the fullscreen resolve that applies the node law. The geometry
            // pass's per-vertex carrier fetches are charged in the dedicated
            // `ScanProcessorResourcePlan` as the tree's one named vertex
            // budget; the per-pixel fields here describe the resolve.
            full_frame_passes: 2,
            // Resolve: the dry carrier and the accumulated raster, one
            // textureLoad each.
            logical_texture_lookups_per_pixel: 2,
            texture_samples_per_pixel: 2,
            // Geometry binds one texture (the carrier, vertex stage); the
            // resolve binds two (carrier plus accumulator). The frame's
            // requirement is the widest single pass.
            sampled_textures_in_pass: 2,
            // The scan reads only its carrier: no image taps, no donors, no
            // routes — the whole tombstone/route surface is structurally
            // absent.
            cross_input_taps: 0,
            reduced_resolution_passes: 0,
            reduced_resolution_surfaces: 0,
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
    /// A motion-field route. Like [`NodeParamType::ImageTap`] it is stable
    /// authored topology rather than a value, and it is deliberately a distinct
    /// type: a motion route resolves against the motion planner's donor flags,
    /// not against the image dependency graph.
    MotionDonor,
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
    // Displace exposes exactly two continuous values. The donor route and the
    // boundary law are stable authored topology: present in the registry so
    // every consumer can enumerate them, but neither modulatable nor diced.
    NodeParamDescriptor {
        kind: NodeKindTag::Displace,
        key: "donor_tap",
        value_type: NodeParamType::ImageTap,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Displace, "amount_x", -1.0, 1.0, 0.0),
    float_param!(Displace, "amount_y", -1.0, 1.0, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Displace,
        key: "boundary",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    // Symmetry Field. Every wire key is prefixed so none of them can
    // cross-resolve against another kind through the deliberate shared-key path
    // in `StableNodeParameter::same_wire_parameter`: this node owns several
    // values whose bare names ("scale", "phase", "seed", "center") are already
    // in use elsewhere in the registry.
    //
    // The four routes, the two discrete laws, the seed, and the six mask bits
    // are stable authored topology: enumerable so every consumer can see them,
    // but neither modulatable nor Dice-eligible.
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_mode",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_boundary",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_seed",
        value_type: NodeParamType::Unsigned,
        range: None,
        default: Some(0.0),
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_donor0_tap",
        value_type: NodeParamType::ImageTap,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_donor1_tap",
        value_type: NodeParamType::ImageTap,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_motion0_donor",
        value_type: NodeParamType::MotionDonor,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_motion1_donor",
        value_type: NodeParamType::MotionDonor,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_source_carrier",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_source_donor0",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_source_donor1",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_source_history",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_motion_slot0",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_motion_slot1",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Symmetry, "symmetry_base_folds", 1.0, 32.0, 1.0),
    float_param!(Symmetry, "symmetry_fold_offset", -32.0, 32.0, 0.0),
    float_param!(Symmetry, "symmetry_radial_phase_deg", -180.0, 180.0, 0.0),
    float_param!(Symmetry, "symmetry_orbit_phase", -1.0, 1.0, 0.0),
    float_param!(Symmetry, "symmetry_planar_axis_deg", -180.0, 180.0, 0.0),
    float_param!(Symmetry, "symmetry_planar_phase", -4.0, 4.0, 0.0),
    float_param!(Symmetry, "symmetry_cell_skew", -1.0, 1.0, 0.0),
    float_param!(Symmetry, "symmetry_spiral_scale", -1.0, 1.0, 0.0),
    float_param!(Symmetry, "symmetry_orbit_radius", 0.0, 1.0, 0.0),
    float_param!(Symmetry, "symmetry_orbit_spin_deg", -180.0, 180.0, 0.0),
    float_param!(Symmetry, "symmetry_motion_gain", -1.0, 1.0, 0.0),
    float_param!(Symmetry, "symmetry_hue_span", 0.0, 1.0, 0.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Symmetry,
        key: "symmetry_center",
        value_type: NodeParamType::Vec2,
        range: Some([-1.0, 2.0]),
        default: Some(0.5),
        dice_eligible: true,
        modulatable: true,
    },
    // Residual exposes exactly two continuous values, under wire keys that are
    // unique across every kind so a modulation route authored for another node
    // can never cross-resolve onto this one. Both routes, both discrete laws,
    // and the quantization seed are stable authored topology: enumerable, but
    // never modulatable and never diced. `algorithm_version` is deliberately
    // absent — it is a persisted schema stamp, not an authored field.
    NodeParamDescriptor {
        kind: NodeKindTag::Residual,
        key: "structure_tap",
        value_type: NodeParamType::ImageTap,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Residual,
        key: "detail_tap",
        value_type: NodeParamType::ImageTap,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(Residual, "mix", 0.0, 1.0, 0.0),
    float_param!(Residual, "detail_gain", 0.0, 4.0, 1.0),
    NodeParamDescriptor {
        kind: NodeKindTag::Residual,
        key: "block",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Residual,
        key: "quantization",
        value_type: NodeParamType::Enum,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::Residual,
        key: "seed",
        value_type: NodeParamType::Unsigned,
        range: None,
        default: Some(0.0),
        dice_eligible: false,
        modulatable: false,
    },
    // The Scan Processor's continuous set, prefixed like Symmetry's so no
    // wire key can collide with another kind's. `scan_lines` and
    // `scan_samples` are plan-time geometry (they size the instanced draw and
    // the vertex ledger, the Residual block-grid law) and the two reversals
    // are discrete laws; neither class is modulatable or Dice-eligible.
    NodeParamDescriptor {
        kind: NodeKindTag::ScanProcessor,
        key: "scan_lines",
        value_type: NodeParamType::Unsigned,
        range: Some([16.0, 1_080.0]),
        default: Some(320.0),
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::ScanProcessor,
        key: "scan_samples",
        value_type: NodeParamType::Unsigned,
        range: Some([64.0, 512.0]),
        default: Some(256.0),
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::ScanProcessor,
        key: "scan_reverse_h",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    NodeParamDescriptor {
        kind: NodeKindTag::ScanProcessor,
        key: "scan_reverse_v",
        value_type: NodeParamType::Bool,
        range: None,
        default: None,
        dice_eligible: false,
        modulatable: false,
    },
    float_param!(ScanProcessor, "scan_amount", 0.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_ribbon_width", 0.0, 1.0, 0.12),
    float_param!(ScanProcessor, "scan_velocity_mix", 0.0, 1.0, 0.8),
    float_param!(ScanProcessor, "scan_tilt_x", -1.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_tilt_y", -1.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_perspective", 0.0, 1.0, 0.3),
    float_param!(ScanProcessor, "scan_s_curve", -1.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_skew", -1.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_collapse", 0.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_osc_amount", 0.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_osc_freq", 0.0, 1.0, 0.25),
    float_param!(ScanProcessor, "scan_osc_lock", 0.0, 1.0, 1.0),
    float_param!(ScanProcessor, "scan_lissajous", 0.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_mono", 0.0, 1.0, 0.0),
    float_param!(ScanProcessor, "scan_hue", 0.0, 1.0, 0.0),
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
    DedicatedPassTextureBudget { textures: u32, limit: u32 },
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
            Self::DedicatedPassTextureBudget { textures, limit } => write!(
                formatter,
                "dedicated pass requests {textures} sampled textures; limit is {limit}"
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
    /// Simultaneous bindings of the widest *ordinary* rack pass. Dedicated
    /// kinds are deliberately excluded so the fixed rack bind layout keeps its
    /// own three-texture ceiling.
    pub max_sampled_textures_in_pass: u32,
    /// Simultaneous bindings of the widest *dedicated* pass, charged against
    /// [`MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS`].
    pub max_sampled_textures_in_dedicated_pass: u32,
    pub cross_input_taps: u32,
    /// Summed reduced-grid passes. These are additional to
    /// `full_frame_passes`; a node that declares one of each encodes both.
    pub reduced_resolution_passes: u32,
    /// Summed reduced-grid surfaces the rack's nodes own. Their bytes are
    /// charged byte-exactly by the owning node's resource plan, not per output
    /// pixel, so they are counted here rather than as full-frame layers.
    pub reduced_resolution_surfaces: u32,
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
            charge_sampled_textures(
                &mut result,
                node.kind.tag(),
                budget.sampled_textures_in_pass,
            );
            result.cross_input_taps = result
                .cross_input_taps
                .checked_add(u32::from(budget.cross_input_taps))
                .ok_or(RackError::ResourceOverflow)?;
            result.reduced_resolution_passes = result
                .reduced_resolution_passes
                .checked_add(u32::from(budget.reduced_resolution_passes))
                .ok_or(RackError::ResourceOverflow)?;
            result.reduced_resolution_surfaces = result
                .reduced_resolution_surfaces
                .checked_add(u32::from(budget.reduced_resolution_surfaces))
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
        validate_dedicated_pass_textures(result.max_sampled_textures_in_dedicated_pass)?;
        Ok(result)
    }

    /// Every group named by every route slot, in node then slot order. The
    /// walk yields a fixed slot-width array per node so a multi-route kind can
    /// never have its second route silently dropped by a one-value closure.
    pub fn referenced_group_ids(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.nodes
            .iter()
            .flat_map(|node| -> [Option<GroupId>; 2] {
                match node.kind {
                    VisualNodeKind::Mask(mask) => [
                        mask.image_tap().and_then(SavedImageTap::referenced_group),
                        None,
                    ],
                    VisualNodeKind::Displace(displace) => [displace.referenced_group(), None],
                    VisualNodeKind::Residual(residual) => residual.referenced_groups(),
                    VisualNodeKind::Symmetry(symmetry) => symmetry.referenced_groups(),
                    _ => [None; 2],
                }
            })
            .flatten()
    }

    pub fn selected_layer_positions(&self) -> impl Iterator<Item = SavedLayerPosition> + '_ {
        // Slot width is the widest authored route count in the tree: Symmetry
        // names two image donors plus two motion donors. A narrower kind pads
        // rather than shrinking the array, so no route is ever truncated away.
        self.nodes
            .iter()
            .flat_map(|node| -> [Option<SavedLayerPosition>; 4] {
                match node.kind {
                    VisualNodeKind::Mask(MaskParams::Image(matte)) => {
                        [matte.selected_layer_position(), None, None, None]
                    }
                    VisualNodeKind::Displace(displace) => {
                        [displace.selected_layer_position(), None, None, None]
                    }
                    VisualNodeKind::Residual(residual) => {
                        let [structure, detail] = residual.selected_layer_positions();
                        [structure, detail, None, None]
                    }
                    VisualNodeKind::Symmetry(symmetry) => symmetry.selected_layer_positions(),
                    _ => [None; 4],
                }
            })
            .flatten()
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
            match &mut node.kind {
                VisualNodeKind::Mask(MaskParams::Image(matte)) => {
                    matte.mark_group_output_missing(removed);
                }
                VisualNodeKind::Displace(displace) => {
                    displace.mark_group_output_missing(removed);
                }
                VisualNodeKind::Symmetry(symmetry) => {
                    symmetry.mark_group_output_missing(removed);
                }
                VisualNodeKind::Residual(residual) => {
                    residual.mark_group_output_missing(removed);
                }
                _ => {}
            }
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

/// Route-resolved Displace state. The saved position survives only inside the
/// resolved tap's missing-source provenance; live routing is by stable ID.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeDisplaceParams {
    pub tap: ResolvedImageTap,
    pub amount_x: f32,
    pub amount_y: f32,
    pub boundary: DisplaceBoundary,
}

impl Default for RuntimeDisplaceParams {
    fn default() -> Self {
        Self {
            tap: ResolvedImageTap {
                source: ResolvedImageSource::OneBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            amount_x: 0.0,
            amount_y: 0.0,
            boundary: DisplaceBoundary::Transparent,
        }
    }
}

impl RuntimeDisplaceParams {
    pub fn resolve_routes(
        saved: DisplaceParams,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: &impl Fn(GroupId) -> bool,
    ) -> Self {
        let saved = saved.sanitized();
        Self {
            tap: saved.tap.to_runtime(layer_at_position, group_exists),
            amount_x: saved.amount_x,
            amount_y: saved.amount_y,
            boundary: saved.boundary,
        }
    }

    pub fn capture_routes(
        self,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> DisplaceParams {
        DisplaceParams {
            tap: self.tap.to_saved(position_of_layer),
            amount_x: self.amount_x,
            amount_y: self.amount_y,
            boundary: self.boundary,
        }
        .sanitized()
    }

    pub(crate) fn sanitized(self) -> Self {
        Self {
            tap: self.tap,
            amount_x: finite_clamp(self.amount_x, 0.0, -1.0, 1.0),
            amount_y: finite_clamp(self.amount_y, 0.0, -1.0, 1.0),
            boundary: self.boundary,
        }
    }

    /// Mirror of [`DisplaceParams::is_exact_bypass`] for the live model.
    pub fn is_exact_bypass(self) -> bool {
        let sanitized = self.sanitized();
        sanitized.amount_x == 0.0 && sanitized.amount_y == 0.0
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

/// Route-resolved Residual Counterpoint state. Both saved positions survive
/// only inside their resolved tap's missing-source provenance; live routing is
/// by stable ID, per slot and independently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeResidualParams {
    pub algorithm_version: u16,
    pub structure: ResolvedImageTap,
    pub detail: ResolvedImageTap,
    pub block: ResidualBlock,
    pub quantization: ResidualQuantization,
    pub mix: f32,
    pub detail_gain: f32,
    pub seed: u32,
}

impl Default for RuntimeResidualParams {
    fn default() -> Self {
        Self {
            algorithm_version: RESIDUAL_ALGORITHM_VERSION,
            structure: ResolvedImageTap {
                source: ResolvedImageSource::OneBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            detail: ResolvedImageTap {
                source: ResolvedImageSource::OneBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            block: ResidualBlock::Eight,
            quantization: ResidualQuantization::Off,
            mix: 0.0,
            detail_gain: 1.0,
            seed: 0,
        }
    }
}

impl RuntimeResidualParams {
    pub fn resolve_routes(
        saved: ResidualParams,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: &impl Fn(GroupId) -> bool,
    ) -> Self {
        let saved = saved.sanitized();
        Self {
            algorithm_version: saved.algorithm_version,
            structure: saved
                .structure
                .to_runtime(&mut *layer_at_position, group_exists),
            detail: saved.detail.to_runtime(layer_at_position, group_exists),
            block: saved.block,
            quantization: saved.quantization,
            mix: saved.mix,
            detail_gain: saved.detail_gain,
            seed: saved.seed,
        }
    }

    pub fn capture_routes(
        self,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> ResidualParams {
        ResidualParams {
            algorithm_version: self.algorithm_version,
            structure: self.structure.to_saved(&mut *position_of_layer),
            detail: self.detail.to_saved(position_of_layer),
            block: self.block,
            quantization: self.quantization,
            mix: self.mix,
            detail_gain: self.detail_gain,
            seed: self.seed,
        }
        .sanitized()
    }

    pub(crate) fn sanitized(self) -> Self {
        Self {
            algorithm_version: RESIDUAL_ALGORITHM_VERSION,
            structure: self.structure,
            detail: self.detail,
            block: self.block,
            quantization: self.quantization,
            mix: finite_clamp(self.mix, 0.0, 0.0, 1.0),
            detail_gain: finite_clamp(self.detail_gain, 1.0, 0.0, 4.0),
            seed: self.seed,
        }
    }

    /// Mirror of [`ResidualParams::is_exact_bypass`] for the live model.
    pub fn is_exact_bypass(self) -> bool {
        self.sanitized().mix == 0.0
    }

    /// Slot 0 is `structure` and slot 1 is `detail`; any other index names no
    /// route and is rejected rather than aliased onto a real slot.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "slot-indexed runtime route access is consumed by the ordered route action"
        )
    )]
    pub const fn route(self, slot: u8) -> Option<ResolvedImageTap> {
        match slot {
            RESIDUAL_STRUCTURE_SLOT => Some(self.structure),
            RESIDUAL_DETAIL_SLOT => Some(self.detail),
            _ => None,
        }
    }

    /// The ordered `SetVisualNodeResidualRoute` action rewrites exactly the
    /// slot it names through this accessor; an unknown slot names no route.
    pub fn route_mut(&mut self, slot: u8) -> Option<&mut ResolvedImageTap> {
        match slot {
            RESIDUAL_STRUCTURE_SLOT => Some(&mut self.structure),
            RESIDUAL_DETAIL_SLOT => Some(&mut self.detail),
            _ => None,
        }
    }

    pub const fn routes(self) -> [ResolvedImageTap; RESIDUAL_ROUTE_SLOTS] {
        [self.structure, self.detail]
    }

    pub const fn selected_layer_ids(self) -> [Option<StableLayerId>; RESIDUAL_ROUTE_SLOTS] {
        [
            match self.structure.source {
                ResolvedImageSource::SelectedLayer { layer_id, .. } => Some(layer_id),
                _ => None,
            },
            match self.detail.source {
                ResolvedImageSource::SelectedLayer { layer_id, .. } => Some(layer_id),
                _ => None,
            },
        ]
    }

    pub const fn referenced_groups(self) -> [Option<GroupId>; RESIDUAL_ROUTE_SLOTS] {
        [
            self.structure.referenced_group(),
            self.detail.referenced_group(),
        ]
    }

    pub fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        self.structure.mark_layer_missing(removed);
        self.detail.mark_layer_missing(removed);
    }

    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        self.structure.mark_group_missing(removed);
        self.detail.mark_group_missing(removed);
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
    Displace(RuntimeDisplaceParams),
    Symmetry(RuntimeSymmetryParams),
    Residual(RuntimeResidualParams),
    Study(StudyRackParams),
    ScanProcessor(ScanProcessorParams),
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
            Self::Displace(_) => NodeKindTag::Displace,
            Self::Symmetry(_) => NodeKindTag::Symmetry,
            Self::Residual(_) => NodeKindTag::Residual,
            Self::Study(_) => NodeKindTag::Study,
            Self::ScanProcessor(_) => NodeKindTag::ScanProcessor,
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
            VisualNodeKind::Displace(value) => Self::Displace(
                RuntimeDisplaceParams::resolve_routes(value, layer_at_position, group_exists),
            ),
            VisualNodeKind::Symmetry(value) => Self::Symmetry(
                RuntimeSymmetryParams::resolve_routes(value, layer_at_position, group_exists),
            ),
            VisualNodeKind::Residual(value) => Self::Residual(
                RuntimeResidualParams::resolve_routes(value, layer_at_position, group_exists),
            ),
            // A Study owns no routes: the digest is opaque authored identity
            // and resolves against the host library at prepare, not here.
            VisualNodeKind::Study(value) => Self::Study(value.sanitized()),
            // The Scan Processor owns no routes either: it reads only its
            // carrier.
            VisualNodeKind::ScanProcessor(value) => Self::ScanProcessor(value.sanitized()),
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
            Self::Displace(value) => {
                VisualNodeKind::Displace(value.capture_routes(position_of_layer))
            }
            Self::Symmetry(value) => {
                VisualNodeKind::Symmetry(value.capture_routes(position_of_layer))
            }
            Self::Residual(value) => {
                VisualNodeKind::Residual(value.capture_routes(position_of_layer))
            }
            Self::Study(value) => VisualNodeKind::Study(value.sanitized()),
            Self::ScanProcessor(value) => VisualNodeKind::ScanProcessor(value.sanitized()),
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
            Self::Displace(value) => Self::Displace(value.sanitized()),
            Self::Symmetry(value) => Self::Symmetry(value.sanitized()),
            Self::Residual(value) => Self::Residual(value.sanitized()),
            Self::ScanProcessor(value) => Self::ScanProcessor(value.sanitized()),
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
            charge_sampled_textures(
                &mut result,
                node.kind.tag(),
                budget.sampled_textures_in_pass,
            );
            result.cross_input_taps = result
                .cross_input_taps
                .checked_add(u32::from(budget.cross_input_taps))
                .ok_or(RackError::ResourceOverflow)?;
            result.reduced_resolution_passes = result
                .reduced_resolution_passes
                .checked_add(u32::from(budget.reduced_resolution_passes))
                .ok_or(RackError::ResourceOverflow)?;
            result.reduced_resolution_surfaces = result
                .reduced_resolution_surfaces
                .checked_add(u32::from(budget.reduced_resolution_surfaces))
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
        validate_dedicated_pass_textures(result.max_sampled_textures_in_dedicated_pass)?;
        Ok(result)
    }

    /// Every group named by every route slot, in node then slot order, walked
    /// over a fixed slot-width array so a two-route kind reports both.
    pub fn referenced_group_ids(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.nodes
            .iter()
            .flat_map(|node| -> [Option<GroupId>; 2] {
                match node.kind {
                    RuntimeVisualNodeKind::Mask(mask) => [
                        mask.image_tap()
                            .and_then(ResolvedImageTap::referenced_group),
                        None,
                    ],
                    RuntimeVisualNodeKind::Displace(displace) => {
                        [displace.referenced_group(), None]
                    }
                    RuntimeVisualNodeKind::Residual(residual) => residual.referenced_groups(),
                    RuntimeVisualNodeKind::Symmetry(symmetry) => symmetry.referenced_groups(),
                    _ => [None; 2],
                }
            })
            .flatten()
    }

    pub fn selected_layer_ids(&self) -> impl Iterator<Item = StableLayerId> + '_ {
        // Same widest-slot law as the saved walk: Symmetry's four donors set
        // the width and every narrower kind pads into it.
        self.nodes
            .iter()
            .flat_map(|node| -> [Option<StableLayerId>; 4] {
                match node.kind {
                    RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) => {
                        [matte.selected_layer_id(), None, None, None]
                    }
                    RuntimeVisualNodeKind::Displace(displace) => {
                        [displace.selected_layer_id(), None, None, None]
                    }
                    RuntimeVisualNodeKind::Residual(residual) => {
                        let [structure, detail] = residual.selected_layer_ids();
                        [structure, detail, None, None]
                    }
                    RuntimeVisualNodeKind::Symmetry(symmetry) => symmetry.selected_layer_ids(),
                    _ => [None; 4],
                }
            })
            .flatten()
    }

    pub fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        for node in &mut self.nodes {
            match &mut node.kind {
                RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) => {
                    matte.tap.mark_layer_missing(removed);
                }
                RuntimeVisualNodeKind::Displace(displace) => {
                    displace.mark_layer_output_missing(removed);
                }
                RuntimeVisualNodeKind::Symmetry(symmetry) => {
                    symmetry.mark_layer_output_missing(removed);
                }
                RuntimeVisualNodeKind::Residual(residual) => {
                    residual.mark_layer_output_missing(removed);
                }
                _ => {}
            }
        }
    }

    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        for node in &mut self.nodes {
            match &mut node.kind {
                RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) => {
                    matte.tap.mark_group_missing(removed);
                }
                RuntimeVisualNodeKind::Displace(displace) => {
                    displace.mark_group_output_missing(removed);
                }
                RuntimeVisualNodeKind::Symmetry(symmetry) => {
                    symmetry.mark_group_output_missing(removed);
                }
                RuntimeVisualNodeKind::Residual(residual) => {
                    residual.mark_group_output_missing(removed);
                }
                _ => {}
            }
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
    /// Dynamic-offset granularity of the device's uniform arena. Reduced
    /// creative passes freeze their stride, so this is a real admission fact
    /// rather than a formatting detail.
    pub min_uniform_buffer_offset_alignment: u32,
    pub max_creative_bytes: u64,
}

impl Default for CreativeResourceLimits {
    fn default() -> Self {
        Self {
            max_texture_dimension_2d: 8_192,
            max_texture_array_layers: 256,
            max_sampled_textures_per_shader_stage: MAX_SAMPLED_TEXTURES_PER_PASS,
            min_uniform_buffer_offset_alignment: 256,
            max_creative_bytes: MAX_CREATIVE_GPU_BYTES,
        }
    }
}

impl From<CreativeResourceLimits> for ResidualResourceLimits {
    fn from(limits: CreativeResourceLimits) -> Self {
        Self {
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            max_sampled_textures_per_shader_stage: limits.max_sampled_textures_per_shader_stage,
            max_residual_bytes: RESIDUAL_AGGREGATE_MAX_BYTES,
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
        let mut max_dedicated_textures = 0_u32;
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
            max_dedicated_textures =
                max_dedicated_textures.max(budget.max_sampled_textures_in_dedicated_pass);
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
        // A dedicated creative pass owns its own bind layout, so it is checked
        // against the device's reported ceiling directly. The fixed rack
        // layout's `.min(MAX_SAMPLED_TEXTURES_PER_PASS)` clamp above must never
        // be applied here, and this check must never relax the clamp above.
        if max_dedicated_textures > limits.max_sampled_textures_per_shader_stage {
            return Err(ResourcePreflightError::SampledTextureLimit {
                requested: max_dedicated_textures,
                limit: limits.max_sampled_textures_per_shader_stage,
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

/// Charge one node's simultaneous-binding count to the ceiling that governs
/// the pass it is actually encoded in. Ordinary kinds share the fixed rack
/// bind layout; dedicated kinds own their own layout and their own ceiling.
fn charge_sampled_textures(budget: &mut RackResourceBudget, tag: NodeKindTag, textures: u8) {
    let textures = u32::from(textures);
    if tag.occupies_dedicated_pass() {
        budget.max_sampled_textures_in_dedicated_pass =
            budget.max_sampled_textures_in_dedicated_pass.max(textures);
    } else {
        budget.max_sampled_textures_in_pass = budget.max_sampled_textures_in_pass.max(textures);
    }
}

/// The dedicated-pass simultaneous-binding ceiling, shared by the saved and
/// runtime racks. It is a constant-only check: the device's own reported limit
/// is applied separately by [`CreativeResourcePlan`], without the ordinary
/// rack's `.min(MAX_SAMPLED_TEXTURES_PER_PASS)` clamp.
pub(crate) fn validate_dedicated_pass_textures(textures: u32) -> Result<(), RackError> {
    if textures > MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS {
        return Err(RackError::DedicatedPassTextureBudget {
            textures,
            limit: MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS,
        });
    }
    Ok(())
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
    fn study_params_serialize_as_hex_reject_hostile_digests_and_default_to_bypass() {
        // Default: no document, exact bypass — a real delegation.
        let default = StudyRackParams::default();
        assert!(default.is_exact_bypass());
        assert_eq!(default.digest_hex(), None);

        // Round trip through the hex form the patch carries.
        let digest = [0xab_u8; 32];
        let params = StudyRackParams {
            document_digest: Some(digest),
        };
        assert!(!params.is_exact_bypass());
        let yaml = serde_yaml::to_string(&params).unwrap();
        assert!(yaml.contains(&"ab".repeat(32)));
        let restored: StudyRackParams = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored, params);

        // Hostile digests are rejections, not truncations or fallbacks.
        for hostile in [
            "\"zz\"",
            &format!("\"{}\"", "ab".repeat(31)),
            &format!("\"{}\"", "ab".repeat(33)),
        ] {
            let doc = format!("document_digest: {hostile}");
            assert!(serde_yaml::from_str::<StudyRackParams>(&doc).is_err());
        }
        // Unknown fields are rejected like every bounded section.
        assert!(serde_yaml::from_str::<StudyRackParams>(
            "document_digest: null
extra: 1"
        )
        .is_err());
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
        assert_eq!(tags.len(), NODE_KIND_DESCRIPTORS.len());
        assert_eq!(keys.len(), NODE_KIND_DESCRIPTORS.len());
        // Ten historical kinds through Displace (code 10), plus Residual
        // Counterpoint (11), the Symmetry Field (12), the Study (13), and the
        // Scan Processor (14). Kind codes are append-only, so this count only
        // ever grows.
        assert_eq!(NODE_KIND_DESCRIPTORS.len(), 14);
        for descriptor in NODE_KIND_DESCRIPTORS {
            assert!(descriptor.budget.full_frame_passes > 0);
            assert!(descriptor.budget.logical_texture_lookups_per_pixel > 0);
            assert!(descriptor.budget.texture_samples_per_pixel > 0);
            assert!(descriptor.budget.sampled_textures_in_pass > 0);
            assert_eq!(node_kind_descriptor(descriptor.tag), &descriptor);
            // Reduced-resolution work is an opt-in charge: every historical
            // kind declares zero of both, so the new fields cannot inflate an
            // existing plan.
            if descriptor.tag != NodeKindTag::Residual {
                assert_eq!(
                    descriptor.budget.reduced_resolution_passes, 0,
                    "{:?} must not declare reduced-resolution passes",
                    descriptor.tag
                );
                assert_eq!(
                    descriptor.budget.reduced_resolution_surfaces, 0,
                    "{:?} must not declare reduced-resolution surfaces",
                    descriptor.tag
                );
            }
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
            (NodeKindTag::Symmetry, 10),
            (NodeKindTag::Residual, 12),
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
            (
                // `algorithm_version` is deliberately absent: it is a persisted
                // schema stamp that sanitization always normalizes, not an
                // authored field any consumer may edit.
                NodeKindTag::Residual,
                &[
                    "structure_tap",
                    "detail_tap",
                    "mix",
                    "detail_gain",
                    "block",
                    "quantization",
                    "seed",
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
                min_uniform_buffer_offset_alignment: u32::MAX,
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

    #[test]
    fn displace_kind_code_is_ten_and_append_only() {
        assert_eq!(NodeKindTag::Displace.signature_code(), 10);
        // Every historical code keeps its value; Displace only appends.
        for (tag, code) in [
            (NodeKindTag::LegacyCanonical, 1_u8),
            (NodeKindTag::LegacyTemporal, 2),
            (NodeKindTag::Transform, 3),
            (NodeKindTag::DigitalColor, 4),
            (NodeKindTag::Key, 5),
            (NodeKindTag::Cellular, 6),
            (NodeKindTag::Shift, 7),
            (NodeKindTag::Grain, 8),
            (NodeKindTag::Mask, 9),
            (NodeKindTag::Displace, 10),
            (NodeKindTag::Residual, 11),
        ] {
            assert_eq!(tag.signature_code(), code);
        }
        let codes: BTreeSet<_> = NODE_KIND_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.tag.signature_code())
            .collect();
        assert_eq!(codes.len(), NODE_KIND_DESCRIPTORS.len());
        assert_eq!(node_kind_descriptor(NodeKindTag::Displace).key, "displace");

        // Adding the kind must not disturb the frozen legacy rack signatures.
        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Layer).topology_signature(),
            LEGACY_LAYER_RACK_SIGNATURE
        );
        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Master).topology_signature(),
            LEGACY_MASTER_RACK_SIGNATURE
        );
    }

    #[test]
    fn displace_defaults_sanitize_and_declare_exact_bypass() {
        let default = DisplaceParams::default();
        assert_eq!(default.tap, SavedImageTap::default());
        assert_eq!(default.tap.source, SavedImageSource::OneBelow);
        assert_eq!(default.tap.timing, EdgeTiming::CurrentFrame);
        assert_eq!(default.amount_x, 0.0);
        assert_eq!(default.amount_y, 0.0);
        assert_eq!(default.boundary, DisplaceBoundary::Transparent);
        assert!(default.is_exact_bypass());

        // Non-finite input takes the neutral fallback rather than a clamped
        // extreme, so hostile values collapse to an exact bypass, never to a
        // full-scale displacement.
        let hostile = DisplaceParams {
            amount_x: f32::NAN,
            amount_y: f32::NEG_INFINITY,
            ..DisplaceParams::default()
        }
        .sanitized();
        assert_eq!(hostile.amount_x, 0.0);
        assert_eq!(hostile.amount_y, 0.0);
        assert!(hostile.is_exact_bypass());
        assert!(DisplaceParams {
            amount_x: f32::INFINITY,
            amount_y: f32::NAN,
            ..DisplaceParams::default()
        }
        .is_exact_bypass());

        // Finite overshoot clamps into the ±1 UV domain and stays live.
        let overshoot = DisplaceParams {
            amount_x: 12.0,
            amount_y: -12.0,
            ..DisplaceParams::default()
        }
        .sanitized();
        assert_eq!(overshoot.amount_x, 1.0);
        assert_eq!(overshoot.amount_y, -1.0);
        assert!(!overshoot.is_exact_bypass());

        // The runtime twin follows the identical law.
        let runtime = RuntimeDisplaceParams::default();
        assert!(runtime.is_exact_bypass());
        assert_eq!(runtime.tap.source, ResolvedImageSource::OneBelow);
        assert_eq!(runtime.boundary, DisplaceBoundary::Transparent);
        assert_eq!(
            RuntimeDisplaceParams {
                amount_x: 9.0,
                ..RuntimeDisplaceParams::default()
            }
            .sanitized()
            .amount_x,
            1.0
        );
    }

    #[test]
    fn displace_serde_round_trips_and_defaults_every_absent_field() {
        let authored = DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::SelectedLayer {
                    layer_position: saved_position(4),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::PreviousFrame,
            },
            amount_x: 0.5,
            amount_y: -0.25,
            boundary: DisplaceBoundary::Mirror,
        };
        let json = serde_json::to_string(&authored).unwrap();
        assert_eq!(
            serde_json::from_str::<DisplaceParams>(&json).unwrap(),
            authored
        );

        // An empty object is the exact default: absent fields never fabricate
        // a gain, a boundary, or a donor.
        assert_eq!(
            serde_json::from_str::<DisplaceParams>("{}").unwrap(),
            DisplaceParams::default()
        );

        // The node round trips through the bounded rack deserializer.
        let node = r#"{"nodes":[{"stable_id":3,"kind":{"kind":"displace","params":{"amount_x":0.5,"boundary":"wrap"}}}],"next_node_id":4}"#;
        let rack: VisualRack = serde_json::from_str(node).unwrap();
        let VisualNodeKind::Displace(params) = rack.iter().next().unwrap().kind else {
            panic!("displace node must survive deserialization");
        };
        assert_eq!(params.amount_x, 0.5);
        assert_eq!(params.amount_y, 0.0);
        assert_eq!(params.boundary, DisplaceBoundary::Wrap);
        assert_eq!(params.tap, SavedImageTap::default());

        // Hostile out-of-range values sanitize during deserialization.
        let hostile = r#"{"nodes":[{"stable_id":3,"kind":{"kind":"displace","params":{"amount_x":50,"amount_y":-50}}}],"next_node_id":4}"#;
        let rack: VisualRack = serde_json::from_str(hostile).unwrap();
        let VisualNodeKind::Displace(params) = rack.iter().next().unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!((params.amount_x, params.amount_y), (1.0, -1.0));

        // An unknown boundary is a closed-vocabulary rejection, not a default.
        assert!(serde_json::from_str::<DisplaceParams>(r#"{"boundary":"teleport"}"#).is_err());
    }

    #[test]
    fn displace_routes_are_visible_to_every_saved_and_runtime_accessor() {
        let group = GroupId::new(21).unwrap();
        let mut saved = VisualRack::empty();
        saved
            .push(VisualNodeKind::Displace(DisplaceParams {
                tap: SavedImageTap {
                    source: SavedImageSource::SelectedLayer {
                        layer_position: saved_position(6),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                amount_x: 0.5,
                ..DisplaceParams::default()
            }))
            .unwrap();
        let group_node = saved
            .push(VisualNodeKind::Displace(DisplaceParams {
                tap: SavedImageTap {
                    source: SavedImageSource::GroupOutput { group_id: group },
                    timing: EdgeTiming::CurrentFrame,
                },
                amount_y: 0.5,
                ..DisplaceParams::default()
            }))
            .unwrap();
        assert_eq!(
            saved.selected_layer_positions().collect::<Vec<_>>(),
            vec![saved_position(6)]
        );
        assert_eq!(
            saved.referenced_group_ids().collect::<Vec<_>>(),
            vec![group]
        );

        // Deleting the group leaves a tombstone that never rebinds.
        saved.mark_group_output_missing(group);
        let VisualNodeKind::Displace(params) = saved.get(group_node).unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!(
            params.tap.source,
            SavedImageSource::MissingGroupOutput { group_id: group }
        );
        assert_eq!(
            saved.referenced_group_ids().collect::<Vec<_>>(),
            vec![group],
            "a tombstone still names its saved identity"
        );

        // Resolution binds the live donor; the vacated position never retargets.
        let live = live_id(77);
        let mut runtime = saved.resolve_routes(
            |position| (position == saved_position(6)).then_some(live),
            |_| false,
        );
        assert_eq!(runtime.selected_layer_ids().collect::<Vec<_>>(), vec![live]);
        runtime.mark_layer_output_missing(live);
        assert!(runtime.selected_layer_ids().next().is_none());
        let RuntimeVisualNodeKind::Displace(params) = runtime.iter().next().unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!(
            params.tap.source,
            ResolvedImageSource::MissingSelectedLayer {
                saved_position: saved_position(6),
                stage: LayerImageStage::PostLocalEffects,
            }
        );

        // Capture preserves the missing identity rather than inventing a donor.
        let recaptured = runtime.capture_routes(|_| None).unwrap();
        let VisualNodeKind::Displace(params) = recaptured.iter().next().unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!(
            params.tap.source,
            SavedImageSource::MissingSelectedLayer {
                saved_position: saved_position(6),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(params.amount_x, 0.5, "route capture preserves the gains");
    }

    #[test]
    fn the_gesture_canvas_route_is_a_positionless_singleton_in_the_closed_vocabulary() {
        // Serde tag is the ordinary snake_case member of the closed vocabulary,
        // and it carries no payload because there is no identity to carry.
        let saved = SavedImageTap {
            source: SavedImageSource::GestureCanvas,
            timing: EdgeTiming::CurrentFrame,
        };
        let json = serde_json::to_string(&saved).unwrap();
        assert!(json.contains("\"source\":\"gesture_canvas\""), "{json}");
        assert_eq!(
            serde_json::from_str::<SavedImageTap>(&json).unwrap(),
            saved,
            "the saved route round-trips exactly"
        );
        // The vocabulary stays closed: a near-miss tag is rejected rather than
        // defaulted into some other producer.
        assert!(serde_json::from_str::<SavedImageSource>(r#"{"source":"gesture_field"}"#).is_err());
        assert_eq!(
            serde_json::from_str::<SavedImageSource>(r#"{"source":"gesture_canvas"}"#).unwrap(),
            SavedImageSource::GestureCanvas
        );

        // Both directions are a fixed point: nothing is looked up, so nothing
        // can be lost. `to_runtime` is handed resolvers that would answer for
        // *any* layer or group, and the route still refuses to become one.
        let runtime = saved.to_runtime(|_| Some(live_id(99)), |_| true);
        assert_eq!(runtime.source, ResolvedImageSource::GestureCanvas);
        assert_eq!(runtime.timing, EdgeTiming::CurrentFrame);
        assert_eq!(runtime.to_saved(|_| Some(saved_position(3))), saved);

        // No saved position and no group identity — deliberately, not by
        // omission. A canvas is a master-scope singleton.
        assert_eq!(saved.selected_layer_position(), None);
        assert_eq!(saved.referenced_group(), None);
        assert_eq!(runtime.referenced_group(), None);

        // Topology edits never touch it: there is no position to invalidate.
        let mut invalidated = runtime;
        invalidated.mark_layer_missing(live_id(99));
        invalidated.mark_group_missing(GroupId::new(21).unwrap());
        assert_eq!(invalidated, runtime);

        // A rack-level route survives resolution and capture unchanged, and it
        // is reported by neither positional accessor.
        let mut rack = VisualRack::empty();
        rack.push(VisualNodeKind::Displace(DisplaceParams {
            tap: saved,
            amount_x: 0.5,
            ..DisplaceParams::default()
        }))
        .unwrap();
        assert!(rack.selected_layer_positions().next().is_none());
        assert!(rack.referenced_group_ids().next().is_none());
        let resolved = rack.resolve_routes(|_| Some(live_id(99)), |_| true);
        let RuntimeVisualNodeKind::Displace(params) = resolved.iter().next().unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!(params.tap.source, ResolvedImageSource::GestureCanvas);
        let recaptured = resolved.capture_routes(|_| None).unwrap();
        let VisualNodeKind::Displace(params) = recaptured.iter().next().unwrap().kind else {
            panic!("displace node")
        };
        assert_eq!(
            params.tap.source,
            SavedImageSource::GestureCanvas,
            "a positionless route cannot decay into a missing donor"
        );

        // The same route is authorable through an image matte.
        let matte = ImageMatte {
            tap: saved,
            ..ImageMatte::default()
        };
        assert_eq!(
            matte.sanitized().tap.source,
            SavedImageSource::GestureCanvas
        );
        assert_eq!(matte.selected_layer_position(), None);
        let mut untouched = matte;
        untouched.mark_group_output_missing(GroupId::new(21).unwrap());
        assert_eq!(untouched.tap.source, SavedImageSource::GestureCanvas);
    }

    #[test]
    fn displace_exposes_only_two_modulatable_dice_eligible_parameters() {
        let descriptors: Vec<_> = NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == NodeKindTag::Displace)
            .collect();
        assert_eq!(descriptors.len(), 4);

        let continuous: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.modulatable)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(continuous, vec!["amount_x", "amount_y"]);
        let diceable: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.dice_eligible)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(diceable, vec!["amount_x", "amount_y"]);

        for descriptor in &descriptors {
            match descriptor.key {
                "amount_x" | "amount_y" => {
                    assert_eq!(descriptor.range, Some([-1.0, 1.0]));
                    assert_eq!(descriptor.default, Some(0.0));
                    assert_eq!(descriptor.value_type, NodeParamType::Float);
                }
                // Route and boundary are stable authored topology: enumerable
                // but never modulatable, never diced, and never ranged.
                "donor_tap" => {
                    assert_eq!(descriptor.value_type, NodeParamType::ImageTap);
                    assert!(descriptor.range.is_none());
                }
                "boundary" => {
                    assert_eq!(descriptor.value_type, NodeParamType::Enum);
                    assert!(descriptor.range.is_none());
                }
                other => panic!("unexpected displace parameter {other}"),
            }
        }
    }

    #[test]
    fn displace_nodes_stay_inside_the_rack_lookup_and_pass_ceilings() {
        let mut rack = VisualRack::empty();
        for _ in 0..MAX_NODES_PER_RACK {
            rack.push(VisualNodeKind::Displace(DisplaceParams {
                amount_x: 0.5,
                ..DisplaceParams::default()
            }))
            .unwrap();
        }
        let budget = rack.resource_budget().unwrap();
        assert_eq!(budget.full_frame_passes, MAX_NODES_PER_RACK as u32);
        assert_eq!(
            budget.logical_texture_lookups_per_pixel,
            3 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(
            budget.texture_samples_per_pixel,
            12 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(budget.max_sampled_textures_in_pass, 2);
        assert_eq!(budget.cross_input_taps, MAX_NODES_PER_RACK as u32);
        assert!(budget.logical_texture_lookups_per_pixel <= MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK);
        assert!(budget.texture_samples_per_pixel <= MAX_TEXTURE_SAMPLES_PER_RACK);
        assert!(budget.max_sampled_textures_in_pass <= MAX_SAMPLED_TEXTURES_PER_PASS);
    }

    /// The Symmetry Field takes append-only kind code 12. Code 11 belongs to
    /// Residual Counterpoint, which was authored in parallel from the same
    /// published SHA and landed first; a kind code enters every persisted
    /// topology signature, so the two can never share one. Every
    /// historical code keeps its value and both frozen legacy rack signatures
    /// stay bit-identical, because `topology_signature` hashes only
    /// `(stable_id, kind code)` pairs.
    #[test]
    fn symmetry_kind_code_is_twelve_and_append_only() {
        assert_eq!(NodeKindTag::Symmetry.signature_code(), 12);
        for (tag, code) in [
            (NodeKindTag::LegacyCanonical, 1_u8),
            (NodeKindTag::LegacyTemporal, 2),
            (NodeKindTag::Transform, 3),
            (NodeKindTag::DigitalColor, 4),
            (NodeKindTag::Key, 5),
            (NodeKindTag::Cellular, 6),
            (NodeKindTag::Shift, 7),
            (NodeKindTag::Grain, 8),
            (NodeKindTag::Mask, 9),
            (NodeKindTag::Displace, 10),
            (NodeKindTag::Symmetry, 12),
            (NodeKindTag::Study, 13),
            (NodeKindTag::ScanProcessor, 14),
        ] {
            assert_eq!(tag.signature_code(), code);
        }
        let codes: BTreeSet<_> = NODE_KIND_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.tag.signature_code())
            .collect();
        assert_eq!(codes.len(), NODE_KIND_DESCRIPTORS.len());
        assert_eq!(node_kind_descriptor(NodeKindTag::Symmetry).key, "symmetry");
        assert_eq!(
            node_kind_descriptor(NodeKindTag::Symmetry).title,
            "Symmetry Field"
        );

        // Three kinds are lifted into dedicated passes: the Symmetry Field's
        // eight-texture fold, the Study interpreter's history-array pass, and
        // the Scan Processor's instanced ribbon geometry.
        let dedicated: Vec<_> = NODE_KIND_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.tag.occupies_dedicated_pass())
            .map(|descriptor| descriptor.tag)
            .collect();
        assert_eq!(
            dedicated,
            vec![
                NodeKindTag::Symmetry,
                NodeKindTag::Study,
                NodeKindTag::ScanProcessor
            ]
        );

        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Layer).topology_signature(),
            LEGACY_LAYER_RACK_SIGNATURE
        );
        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Master).topology_signature(),
            LEGACY_MASTER_RACK_SIGNATURE
        );
    }

    /// The Scan Processor's declared surface: fifteen continuous modulatable
    /// Dice-eligible floats, two plan-time geometry integers, and two
    /// discrete reversal laws — nothing else, and none of the discrete class
    /// carries a modulatable address.
    #[test]
    fn scan_processor_exposes_fifteen_continuous_controls_and_four_discrete_laws() {
        let rows: Vec<_> = NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == NodeKindTag::ScanProcessor)
            .collect();
        assert_eq!(rows.len(), 19);
        let continuous: Vec<_> = rows
            .iter()
            .filter(|descriptor| descriptor.modulatable)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(
            continuous,
            vec![
                "scan_amount",
                "scan_ribbon_width",
                "scan_velocity_mix",
                "scan_tilt_x",
                "scan_tilt_y",
                "scan_perspective",
                "scan_s_curve",
                "scan_skew",
                "scan_collapse",
                "scan_osc_amount",
                "scan_osc_freq",
                "scan_osc_lock",
                "scan_lissajous",
                "scan_mono",
                "scan_hue",
            ]
        );
        for descriptor in &rows {
            assert_eq!(
                descriptor.modulatable, descriptor.dice_eligible,
                "{} must be Dice-eligible exactly when modulatable",
                descriptor.key
            );
            assert_eq!(
                descriptor.modulatable,
                descriptor.value_type == NodeParamType::Float,
                "{} continuous exactly when Float",
                descriptor.key
            );
        }
        for geometry in ["scan_lines", "scan_samples"] {
            let row = rows
                .iter()
                .find(|descriptor| descriptor.key == geometry)
                .expect("geometry row");
            assert_eq!(row.value_type, NodeParamType::Unsigned);
        }
        for reversal in ["scan_reverse_h", "scan_reverse_v"] {
            let row = rows
                .iter()
                .find(|descriptor| descriptor.key == reversal)
                .expect("reversal row");
            assert_eq!(row.value_type, NodeParamType::Bool);
        }
    }

    /// The default is an exact bypass, any deflection wakes it, hostile
    /// scalars land on the neutral default, and the vertex ledger is exactly
    /// two per sample per line with the structural cap at the maxima.
    #[test]
    fn scan_processor_defaults_sanitize_and_declare_exact_bypass() {
        use crate::scan_processor::{ScanProcessorParams, MAX_SCAN_PROCESSOR_VERTICES};
        let default = ScanProcessorParams::default();
        assert!(default.is_exact_bypass());
        assert_eq!(default.vertex_count(), 320 * 256 * 2);
        let woken = ScanProcessorParams {
            amount: 0.5,
            ..ScanProcessorParams::default()
        };
        assert!(!woken.is_exact_bypass());
        let hostile = ScanProcessorParams {
            amount: f32::NAN,
            lines: u32::MAX,
            samples_per_line: 0,
            ..ScanProcessorParams::default()
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.amount, 0.0);
        assert_eq!(clean.lines, 1_080);
        assert_eq!(clean.samples_per_line, 64);
        assert!(clean.is_exact_bypass());
        let maxed = ScanProcessorParams {
            lines: 1_080,
            samples_per_line: 512,
            ..ScanProcessorParams::default()
        };
        assert_eq!(maxed.vertex_count(), MAX_SCAN_PROCESSOR_VERTICES);
    }

    /// The node's serde: the tagged kind round trips, absent fields default,
    /// and an unknown field is a deserialization rejection.
    #[test]
    fn scan_processor_serde_round_trips_and_rejects_unknown_fields() {
        use crate::scan_processor::ScanProcessorParams;
        let node = VisualNode::authored(
            NodeId::new(7).unwrap(),
            VisualNodeKind::ScanProcessor(ScanProcessorParams {
                amount: 0.4,
                reverse_h: true,
                lines: 240,
                ..ScanProcessorParams::default()
            }),
        );
        let yaml = serde_yaml::to_string(&node).expect("serialize");
        assert!(yaml.contains("kind: scan_processor"));
        let restored: VisualNode = serde_yaml::from_str(&yaml).expect("round trip");
        assert_eq!(restored, node);
        let partial = "stable_id: 3\nkind:\n  kind: scan_processor\n  params:\n    amount: 0.25\n";
        let parsed: VisualNode = serde_yaml::from_str(partial).expect("partial params");
        let VisualNodeKind::ScanProcessor(params) = parsed.kind else {
            panic!("scan processor kind");
        };
        assert_eq!(params.amount, 0.25);
        assert_eq!(params.lines, 320);
        let hostile =
            "stable_id: 3\nkind:\n  kind: scan_processor\n  params:\n    unknown_field: 1\n";
        assert!(serde_yaml::from_str::<VisualNode>(hostile).is_err());
    }

    /// The exact default is cyclic at one fold, carrier only, with no motion,
    /// no history, no hue and a neutral phase/axis/center — and it is an exact
    /// bypass. Hostile non-finite input takes the neutral fallback, so it
    /// collapses to a bypass rather than to a full-scale fold.
    #[test]
    fn symmetry_defaults_sanitize_and_declare_exact_bypass() {
        let default = SymmetryParams::default();
        assert_eq!(default.mode, crate::symmetry::SymmetryMode::Cyclic);
        assert_eq!(default.effective_folds(), 1);
        assert_eq!(default.radial_phase_deg, 0.0);
        assert_eq!(default.orbit_phase, 0.0);
        assert_eq!(default.planar_axis_deg, 0.0);
        assert_eq!(default.planar_phase, 0.0);
        assert_eq!(default.center, [0.5, 0.5]);
        assert_eq!(default.motion_gain, 0.0);
        assert_eq!(default.hue_span, 0.0);
        assert_eq!(default.seed, 0);
        assert!(default.source_mask.is_carrier_only());
        assert!(default.motion_mask.is_empty());
        for slot in 0..2_u8 {
            assert_eq!(
                default.donor_tap(slot).unwrap().source,
                SavedImageSource::OneBelow
            );
            assert_eq!(
                default.donor_tap(slot).unwrap().timing,
                EdgeTiming::CurrentFrame
            );
            assert_eq!(
                default.motion_donor(slot).unwrap(),
                crate::symmetry::SavedMotionDonor::None
            );
        }
        assert!(default.table_is_neutral());
        assert!(default.is_exact_bypass());

        // Hostile values fall back to neutral, never to a clamped extreme.
        let hostile = SymmetryParams {
            base_folds: f32::NAN,
            fold_offset: f32::INFINITY,
            radial_phase_deg: f32::NEG_INFINITY,
            orbit_phase: f32::NAN,
            motion_gain: f32::INFINITY,
            hue_span: f32::NAN,
            ..SymmetryParams::default()
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.base_folds, 1.0);
        assert_eq!(clean.fold_offset, 0.0);
        assert_eq!(clean.radial_phase_deg, 0.0);
        assert_eq!(clean.orbit_phase, 0.0);
        assert_eq!(clean.motion_gain, 0.0);
        assert_eq!(clean.hue_span, 0.0);
        assert!(hostile.is_exact_bypass());

        // Finite overshoot clamps and stays live.
        let overshoot = SymmetryParams {
            motion_gain: 12.0,
            hue_span: 12.0,
            ..SymmetryParams::default()
        }
        .sanitized();
        assert_eq!(overshoot.motion_gain, 1.0);
        assert_eq!(overshoot.hue_span, 1.0);
        assert!(!overshoot.is_exact_bypass());

        // Both halves of the bypass predicate are load bearing: an identity
        // fold whose table can still reach a donor, the history ring, a motion
        // donor, or a hue rotation is not a bypass.
        for armed in [
            SymmetryParams {
                base_folds: 4.0,
                ..SymmetryParams::default()
            },
            SymmetryParams {
                source_mask: crate::symmetry::SymmetrySourceMask {
                    donor0: true,
                    ..crate::symmetry::SymmetrySourceMask::CARRIER_ONLY
                },
                ..SymmetryParams::default()
            },
            SymmetryParams {
                source_mask: crate::symmetry::SymmetrySourceMask {
                    clean_history: true,
                    ..crate::symmetry::SymmetrySourceMask::CARRIER_ONLY
                },
                ..SymmetryParams::default()
            },
            SymmetryParams {
                motion_mask: crate::symmetry::SymmetryMotionMask {
                    slot0: true,
                    slot1: false,
                },
                ..SymmetryParams::default()
            },
            SymmetryParams {
                hue_span: 0.25,
                ..SymmetryParams::default()
            },
        ] {
            assert!(!armed.is_exact_bypass());
        }

        // The runtime twin follows the identical law through the same values.
        let runtime = RuntimeSymmetryParams::default();
        assert!(runtime.is_exact_bypass());
        assert_eq!(runtime.donors[0].source, ResolvedImageSource::OneBelow);
        assert_eq!(runtime.donors[1].source, ResolvedImageSource::OneBelow);
        assert_eq!(runtime.motion, [crate::motion::MotionDonor::None; 2]);
        assert_eq!(
            RuntimeSymmetryParams {
                hue_span: 9.0,
                ..RuntimeSymmetryParams::default()
            }
            .sanitized()
            .hue_span,
            1.0
        );
        assert!(!RuntimeSymmetryParams {
            hue_span: 0.5,
            ..RuntimeSymmetryParams::default()
        }
        .is_exact_bypass());
    }

    /// The node serializes through the bounded rack deserializer with its kind
    /// key, defaults every absent field, sanitizes hostile values during
    /// deserialization, and rejects an unknown discrete token rather than
    /// defaulting it.
    #[test]
    fn symmetry_serde_round_trips_and_defaults_every_absent_field() {
        let params = SymmetryParams {
            mode: crate::symmetry::SymmetryMode::Dihedral,
            base_folds: 6.0,
            hue_span: 0.5,
            seed: 99,
            ..SymmetryParams::default()
        };
        let encoded = serde_json::to_string(&params).expect("params serialize");
        let decoded: SymmetryParams = serde_json::from_str(&encoded).expect("params deserialize");
        assert_eq!(decoded, params);

        let empty: SymmetryParams = serde_json::from_str("{}").expect("an empty map deserializes");
        assert_eq!(empty, SymmetryParams::default());

        let node: VisualNode = serde_json::from_str(
            r#"{"stable_id":3,"kind":{"kind":"symmetry","params":{"base_folds":900.0,"hue_span":-5.0,"motion_gain":50.0}}}"#,
        )
        .expect("a symmetry node deserializes through the bounded rack model");
        let VisualNodeKind::Symmetry(value) = node.kind else {
            panic!("symmetry node")
        };
        assert_eq!(value.base_folds, 32.0, "finite overshoot clamps");
        assert_eq!(
            value.hue_span, 0.0,
            "an out of range span clamps to its floor"
        );
        assert_eq!(value.motion_gain, 1.0);
        assert_eq!(value.seed, 0, "an absent seed is the historical default");
        assert!(value.source_mask.is_carrier_only());

        // Closed vocabularies stay closed.
        assert!(serde_json::from_str::<SymmetryParams>(r#"{"mode":"hyperbolic"}"#).is_err());
        assert!(serde_json::from_str::<SymmetryParams>(r#"{"boundary":"teleport"}"#).is_err());
    }

    /// Both image slots and both motion slots are visible to every saved and
    /// runtime rack accessor, addressed by slot index. A tombstoned slot keeps
    /// naming its saved identity and never rebinds against a replacement.
    #[test]
    fn symmetry_routes_are_visible_to_every_saved_and_runtime_accessor() {
        let group = GroupId::new(31).unwrap();
        let mut saved = VisualRack::empty();
        let node = saved
            .push(VisualNodeKind::Symmetry(SymmetryParams {
                base_folds: 6.0,
                donors: [
                    SavedImageTap {
                        source: SavedImageSource::SelectedLayer {
                            layer_position: saved_position(4),
                            stage: LayerImageStage::PostLocalEffects,
                        },
                        timing: EdgeTiming::CurrentFrame,
                    },
                    SavedImageTap {
                        source: SavedImageSource::GroupOutput { group_id: group },
                        timing: EdgeTiming::PreviousFrame,
                    },
                ],
                motion: [
                    crate::symmetry::SavedMotionDonor::Selected {
                        saved_position: saved_position(5),
                    },
                    crate::symmetry::SavedMotionDonor::Selected {
                        saved_position: saved_position(6),
                    },
                ],
                ..SymmetryParams::default()
            }))
            .unwrap();

        // A `filter_map` walker could only ever surface one route per node.
        assert_eq!(
            saved.selected_layer_positions().collect::<Vec<_>>(),
            vec![saved_position(4), saved_position(5), saved_position(6)]
        );
        assert_eq!(
            saved.referenced_group_ids().collect::<Vec<_>>(),
            vec![group]
        );

        saved.mark_group_output_missing(group);
        let VisualNodeKind::Symmetry(params) = saved.get(node).unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(
            params.donor_tap(1).unwrap().source,
            SavedImageSource::MissingGroupOutput { group_id: group }
        );
        assert_eq!(
            params.donor_tap(0).unwrap().source,
            SavedImageSource::SelectedLayer {
                layer_position: saved_position(4),
                stage: LayerImageStage::PostLocalEffects,
            },
            "slot zero is untouched by slot one losing its group"
        );
        assert!(params.donor_tap(2).is_none());
        assert!(params.motion_donor(2).is_none());

        // Resolution binds each slot independently; a position that cannot be
        // resolved becomes an explicit tombstone at its own slot.
        let image_donor = live_id(41);
        let motion_donor = live_id(42);
        let mut runtime = saved.resolve_routes(
            |position| match position.get() {
                4 => Some(image_donor),
                5 => Some(motion_donor),
                _ => None,
            },
            |_| false,
        );
        assert_eq!(
            runtime.selected_layer_ids().collect::<Vec<_>>(),
            vec![image_donor, motion_donor]
        );
        let RuntimeVisualNodeKind::Symmetry(params) = runtime.iter().next().unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(
            params.motion_donor(1).unwrap(),
            crate::motion::MotionDonor::Missing {
                saved_position: saved_position(6)
            }
        );

        runtime.mark_layer_output_missing(image_donor);
        let RuntimeVisualNodeKind::Symmetry(params) = runtime.iter().next().unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(
            params.donor_tap(0).unwrap().source,
            ResolvedImageSource::MissingSelectedLayer {
                saved_position: saved_position(4),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            params.motion_donor(0).unwrap(),
            crate::motion::MotionDonor::Selected {
                layer_id: motion_donor,
                saved_position: saved_position(5),
            },
            "an image slot losing its donor never disturbs a motion slot"
        );
        runtime.mark_layer_output_missing(motion_donor);
        assert!(runtime.selected_layer_ids().next().is_none());

        // Capture preserves every missing identity rather than inventing one,
        // and re-resolving against a replacement at the vacated position must
        // not rebind it.
        let recaptured = runtime.capture_routes(|_| None).unwrap();
        let VisualNodeKind::Symmetry(params) = recaptured.iter().next().unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(
            params.donor_tap(0).unwrap().source,
            SavedImageSource::MissingSelectedLayer {
                saved_position: saved_position(4),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            params.motion_donor(0).unwrap(),
            crate::symmetry::SavedMotionDonor::Missing {
                saved_position: saved_position(5)
            }
        );
        assert_eq!(params.base_folds, 6.0, "route capture preserves the values");

        let rebound = recaptured.resolve_routes(|_| Some(live_id(99)), |_| true);
        let RuntimeVisualNodeKind::Symmetry(params) = rebound.iter().next().unwrap().kind else {
            panic!("symmetry node")
        };
        assert!(
            params.selected_layer_ids().iter().all(Option::is_none),
            "a tombstoned slot never rebinds against a replacement"
        );
    }

    /// Every Symmetry wire key is prefixed and unique across the whole
    /// registry, so none of them can cross-resolve through the deliberate
    /// shared-key path. Only the declared continuous controls are modulatable
    /// and Dice-eligible; the four routes, two discrete laws, seed, and six
    /// mask bits are enumerable topology.
    #[test]
    fn symmetry_exposes_only_its_declared_continuous_controls_as_modulatable() {
        let descriptors: Vec<_> = NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == NodeKindTag::Symmetry)
            .collect();
        assert_eq!(descriptors.len(), 26);

        let continuous: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.modulatable)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(
            continuous,
            vec![
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
                "symmetry_center",
            ]
        );
        let diceable: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.dice_eligible)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(diceable, continuous);

        for descriptor in &descriptors {
            assert!(
                descriptor.key.starts_with("symmetry_"),
                "{} must be prefixed so it cannot alias another kind",
                descriptor.key
            );
            if descriptor.modulatable {
                assert!(descriptor.range.is_some(), "{}", descriptor.key);
            } else {
                assert!(descriptor.range.is_none(), "{}", descriptor.key);
                assert!(!descriptor.dice_eligible, "{}", descriptor.key);
                assert!(
                    matches!(
                        descriptor.value_type,
                        NodeParamType::Enum
                            | NodeParamType::Bool
                            | NodeParamType::Unsigned
                            | NodeParamType::ImageTap
                            | NodeParamType::MotionDonor
                    ),
                    "{}",
                    descriptor.key
                );
            }
            // No other kind may already own this wire key.
            assert!(
                NODE_PARAM_DESCRIPTORS
                    .iter()
                    .filter(|other| other.key == descriptor.key)
                    .all(|other| other.kind == NodeKindTag::Symmetry),
                "{} is not unique across the registry",
                descriptor.key
            );
        }

        let routes: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| {
                matches!(
                    descriptor.value_type,
                    NodeParamType::ImageTap | NodeParamType::MotionDonor
                )
            })
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(
            routes,
            vec![
                "symmetry_donor0_tap",
                "symmetry_donor1_tap",
                "symmetry_motion0_donor",
                "symmetry_motion1_donor"
            ],
            "exactly two image slots and two motion slots, addressed by slot"
        );
    }

    /// The dedicated pass declares eight simultaneously sampled textures and is
    /// charged against its own ceiling. The fixed Collision Rack layout keeps
    /// its separate three-texture cap, and neither ceiling admits the other.
    #[test]
    fn symmetry_declares_eight_textures_in_a_dedicated_pass_that_admits_eight_and_refuses_nine() {
        assert_eq!(MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS, 8);
        assert_eq!(MAX_SAMPLED_TEXTURES_PER_PASS, 3);

        let budget = node_kind_descriptor(NodeKindTag::Symmetry).budget;
        assert_eq!(budget.full_frame_passes, 1);
        assert_eq!(budget.logical_texture_lookups_per_pixel, 4);
        assert_eq!(budget.texture_samples_per_pixel, 10);
        assert_eq!(budget.sampled_textures_in_pass, 8);
        assert_eq!(budget.cross_input_taps, 2);

        assert!(validate_dedicated_pass_textures(8).is_ok());
        assert_eq!(
            validate_dedicated_pass_textures(9),
            Err(RackError::DedicatedPassTextureBudget {
                textures: 9,
                limit: 8,
            })
        );

        // A rack holding the dedicated node still reports zero ordinary-pass
        // textures for it, so the three-texture rack layout is untouched.
        let mut rack = VisualRack::empty();
        rack.push(VisualNodeKind::Symmetry(SymmetryParams::default()))
            .unwrap();
        let budget = rack.resource_budget().unwrap();
        assert_eq!(budget.max_sampled_textures_in_pass, 0);
        assert_eq!(budget.max_sampled_textures_in_dedicated_pass, 8);

        // An ordinary node beside it charges the ordinary accumulator only.
        rack.push(VisualNodeKind::Displace(DisplaceParams::default()))
            .unwrap();
        let budget = rack.resource_budget().unwrap();
        assert_eq!(budget.max_sampled_textures_in_pass, 2);
        assert_eq!(budget.max_sampled_textures_in_dedicated_pass, 8);

        // Preflight checks the dedicated ceiling against the device's own
        // reported limit, without the fixed rack layout's `.min(3)` clamp. The
        // stub default limit is that same 3, so a fixture must raise it
        // explicitly the way production reads the device floor of 16.
        let graph = ImageGraphPlan {
            current_topological_order: Vec::new(),
            current_taps: 0,
            previous_taps: 0,
            diagnostics: Vec::new(),
        };
        let stub = CreativeResourceLimits::default();
        assert_eq!(stub.max_sampled_textures_per_shader_stage, 3);
        assert_eq!(
            CreativeResourcePlan::preflight_with_surface_formats(
                [640, 360],
                std::slice::from_ref(&rack),
                &graph,
                0,
                0,
                stub,
            ),
            Err(ResourcePreflightError::SampledTextureLimit {
                requested: 8,
                limit: 3,
            })
        );
        let floor = CreativeResourceLimits {
            max_sampled_textures_per_shader_stage: 16,
            ..CreativeResourceLimits::default()
        };
        assert!(CreativeResourcePlan::preflight_with_surface_formats(
            [640, 360],
            std::slice::from_ref(&rack),
            &graph,
            0,
            0,
            floor,
        )
        .is_ok());
    }

    /// A node declaring four logical lookups admits a full rack of eight, and
    /// the dedicated accumulator never inflates the ordinary rack ceiling.
    #[test]
    fn symmetry_nodes_stay_inside_the_rack_lookup_and_pass_ceilings() {
        let mut rack = VisualRack::empty();
        for _ in 0..MAX_NODES_PER_RACK {
            rack.push(VisualNodeKind::Symmetry(SymmetryParams {
                base_folds: 6.0,
                ..SymmetryParams::default()
            }))
            .unwrap();
        }
        let budget = rack.resource_budget().unwrap();
        assert_eq!(budget.full_frame_passes, MAX_NODES_PER_RACK as u32);
        assert_eq!(
            budget.logical_texture_lookups_per_pixel,
            4 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(
            budget.texture_samples_per_pixel,
            10 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(budget.max_sampled_textures_in_pass, 0);
        assert_eq!(budget.max_sampled_textures_in_dedicated_pass, 8);
        assert_eq!(budget.cross_input_taps, 2 * MAX_NODES_PER_RACK as u32);
        assert_eq!(
            budget.logical_texture_lookups_per_pixel,
            MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK
        );
        assert!(budget.texture_samples_per_pixel <= MAX_TEXTURE_SAMPLES_PER_RACK);
        assert!(budget.max_sampled_textures_in_pass <= MAX_SAMPLED_TEXTURES_PER_PASS);
        assert!(
            budget.max_sampled_textures_in_dedicated_pass
                <= MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS
        );
    }

    #[test]
    fn residual_kind_code_is_eleven_and_append_only() {
        assert_eq!(NodeKindTag::Residual.signature_code(), 11);
        // Every historical code keeps its value; Residual only appends.
        for (tag, code) in [
            (NodeKindTag::LegacyCanonical, 1_u8),
            (NodeKindTag::LegacyTemporal, 2),
            (NodeKindTag::Transform, 3),
            (NodeKindTag::DigitalColor, 4),
            (NodeKindTag::Key, 5),
            (NodeKindTag::Cellular, 6),
            (NodeKindTag::Shift, 7),
            (NodeKindTag::Grain, 8),
            (NodeKindTag::Mask, 9),
            (NodeKindTag::Displace, 10),
            (NodeKindTag::Residual, 11),
        ] {
            assert_eq!(tag.signature_code(), code);
        }
        let codes: BTreeSet<_> = NODE_KIND_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.tag.signature_code())
            .collect();
        assert_eq!(codes.len(), NODE_KIND_DESCRIPTORS.len());
        assert_eq!(node_kind_descriptor(NodeKindTag::Residual).key, "residual");

        // Both discrete vocabularies are closed and their shader codes are
        // append-only from zero, exactly like DisplaceBoundary.
        for (block, code, edge) in [
            (ResidualBlock::Four, 0_u32, 4_u32),
            (ResidualBlock::Eight, 1, 8),
            (ResidualBlock::Sixteen, 2, 16),
            (ResidualBlock::ThirtyTwo, 3, 32),
            (ResidualBlock::SixtyFour, 4, 64),
        ] {
            assert_eq!(block.code(), code);
            assert_eq!(block.edge(), edge);
        }
        for (quantization, code, levels) in [
            (ResidualQuantization::Off, 0_u32, 0_u32),
            (ResidualQuantization::Coarse, 1, 8),
            (ResidualQuantization::Medium, 2, 32),
            (ResidualQuantization::Fine, 3, 128),
        ] {
            assert_eq!(quantization.code(), code);
            assert_eq!(quantization.levels(), levels);
        }
        assert_eq!(
            ResidualQuantization::default().levels(),
            0,
            "the default quantization is exact identity, not a one-level collapse"
        );
        assert_eq!(RESIDUAL_ALGORITHM_VERSION, 1);
        assert_eq!(RESIDUAL_ROUTE_SLOTS, 2);
        assert_eq!(RESIDUAL_STRUCTURE_SLOT, 0);
        assert_eq!(RESIDUAL_DETAIL_SLOT, 1);

        // Adding the kind must not disturb the frozen legacy rack signatures.
        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Layer).topology_signature(),
            LEGACY_LAYER_RACK_SIGNATURE
        );
        assert_eq!(
            VisualRack::synthetic_legacy(LegacyRackScope::Master).topology_signature(),
            LEGACY_MASTER_RACK_SIGNATURE
        );
    }

    #[test]
    fn residual_defaults_sanitize_and_declare_exact_bypass() {
        let default = ResidualParams::default();
        assert_eq!(default.algorithm_version, RESIDUAL_ALGORITHM_VERSION);
        assert_eq!(default.structure, SavedImageTap::default());
        assert_eq!(default.detail, SavedImageTap::default());
        assert_eq!(default.structure.source, SavedImageSource::OneBelow);
        assert_eq!(default.detail.source, SavedImageSource::OneBelow);
        assert_eq!(default.structure.timing, EdgeTiming::CurrentFrame);
        assert_eq!(default.detail.timing, EdgeTiming::CurrentFrame);
        assert_eq!(default.block, ResidualBlock::Eight);
        assert_eq!(default.quantization, ResidualQuantization::Off);
        assert_eq!(default.mix, 0.0);
        assert_eq!(default.detail_gain, 1.0);
        assert_eq!(default.seed, 0);
        assert!(default.is_exact_bypass());

        // Non-finite mix takes the neutral zero fallback rather than a clamped
        // extreme, so hostile input collapses to an exact bypass and never to a
        // full recombination. A non-finite detail gain takes its own neutral
        // 1.0, not either end of the range.
        for hostile_mix in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let hostile = ResidualParams {
                mix: hostile_mix,
                detail_gain: f32::NAN,
                ..ResidualParams::default()
            }
            .sanitized();
            assert_eq!(hostile.mix, 0.0);
            assert_eq!(hostile.detail_gain, 1.0);
            assert!(hostile.is_exact_bypass());
        }

        // Finite overshoot clamps into range and stays live.
        let overshoot = ResidualParams {
            mix: 12.0,
            detail_gain: 12.0,
            seed: 4_242,
            ..ResidualParams::default()
        }
        .sanitized();
        assert_eq!(overshoot.mix, 1.0);
        assert_eq!(overshoot.detail_gain, 4.0);
        assert_eq!(overshoot.seed, 4_242, "the seed is never rescaled");
        assert!(!overshoot.is_exact_bypass());
        assert_eq!(
            ResidualParams {
                detail_gain: -12.0,
                ..ResidualParams::default()
            }
            .sanitized()
            .detail_gain,
            0.0
        );

        // The version stamp is normalized rather than trusted, so a patch can
        // never select a decomposition law this build does not implement.
        assert_eq!(
            ResidualParams {
                algorithm_version: 9_999,
                ..ResidualParams::default()
            }
            .sanitized()
            .algorithm_version,
            RESIDUAL_ALGORITHM_VERSION
        );

        // The runtime twin follows the identical law.
        let runtime = RuntimeResidualParams::default();
        assert!(runtime.is_exact_bypass());
        assert_eq!(runtime.structure.source, ResolvedImageSource::OneBelow);
        assert_eq!(runtime.detail.source, ResolvedImageSource::OneBelow);
        assert_eq!(runtime.block, ResidualBlock::Eight);
        assert_eq!(runtime.quantization, ResidualQuantization::Off);
        assert_eq!(runtime.detail_gain, 1.0);
        let hostile_runtime = RuntimeResidualParams {
            mix: f32::NEG_INFINITY,
            detail_gain: 9.0,
            ..RuntimeResidualParams::default()
        }
        .sanitized();
        assert_eq!(hostile_runtime.mix, 0.0);
        assert_eq!(hostile_runtime.detail_gain, 4.0);
        assert!(hostile_runtime.is_exact_bypass());
    }

    #[test]
    fn residual_serde_round_trips_and_defaults_every_absent_field() {
        let authored = ResidualParams {
            algorithm_version: RESIDUAL_ALGORITHM_VERSION,
            structure: SavedImageTap {
                source: SavedImageSource::SelectedLayer {
                    layer_position: saved_position(4),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::PreviousFrame,
            },
            detail: SavedImageTap {
                source: SavedImageSource::CleanProgram,
                timing: EdgeTiming::CurrentFrame,
            },
            block: ResidualBlock::ThirtyTwo,
            quantization: ResidualQuantization::Medium,
            mix: 0.75,
            detail_gain: 2.5,
            seed: 9,
        };
        let json = serde_json::to_string(&authored).unwrap();
        assert_eq!(
            serde_json::from_str::<ResidualParams>(&json).unwrap(),
            authored
        );

        // An empty object is the exact default: absent fields never fabricate a
        // mix, a gain, a block, a quantization law, or a route.
        assert_eq!(
            serde_json::from_str::<ResidualParams>("{}").unwrap(),
            ResidualParams::default()
        );

        // The node round trips through the bounded rack deserializer.
        let node = r#"{"nodes":[{"stable_id":3,"kind":{"kind":"residual","params":{"mix":0.5,"block":"sixty_four"}}}],"next_node_id":4}"#;
        let rack: VisualRack = serde_json::from_str(node).unwrap();
        let VisualNodeKind::Residual(params) = rack.iter().next().unwrap().kind else {
            panic!("residual node must survive deserialization");
        };
        assert_eq!(params.mix, 0.5);
        assert_eq!(params.detail_gain, 1.0);
        assert_eq!(params.block, ResidualBlock::SixtyFour);
        assert_eq!(params.quantization, ResidualQuantization::Off);
        assert_eq!(params.structure, SavedImageTap::default());
        assert_eq!(params.detail, SavedImageTap::default());

        // Hostile out-of-range values sanitize during deserialization.
        let hostile = r#"{"nodes":[{"stable_id":3,"kind":{"kind":"residual","params":{"mix":50,"detail_gain":-50,"algorithm_version":600}}}],"next_node_id":4}"#;
        let rack: VisualRack = serde_json::from_str(hostile).unwrap();
        let VisualNodeKind::Residual(params) = rack.iter().next().unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!((params.mix, params.detail_gain), (1.0, 0.0));
        assert_eq!(params.algorithm_version, RESIDUAL_ALGORITHM_VERSION);

        // Both vocabularies are closed: an unknown token is a rejection, not a
        // silent default.
        assert!(serde_json::from_str::<ResidualParams>(r#"{"block":"three"}"#).is_err());
        assert!(serde_json::from_str::<ResidualParams>(r#"{"quantization":"lossy"}"#).is_err());
    }

    #[test]
    fn residual_routes_are_visible_to_every_saved_and_runtime_accessor() {
        let structure_group = GroupId::new(21).unwrap();
        let detail_group = GroupId::new(22).unwrap();
        let mut saved = VisualRack::empty();
        let layer_node = saved
            .push(VisualNodeKind::Residual(ResidualParams {
                structure: SavedImageTap {
                    source: SavedImageSource::SelectedLayer {
                        layer_position: saved_position(6),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                detail: SavedImageTap {
                    source: SavedImageSource::SelectedLayer {
                        layer_position: saved_position(9),
                        stage: LayerImageStage::PreLocalEffects,
                    },
                    timing: EdgeTiming::PreviousFrame,
                },
                mix: 0.5,
                ..ResidualParams::default()
            }))
            .unwrap();
        let group_node = saved
            .push(VisualNodeKind::Residual(ResidualParams {
                structure: SavedImageTap {
                    source: SavedImageSource::GroupOutput {
                        group_id: structure_group,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                detail: SavedImageTap {
                    source: SavedImageSource::GroupOutput {
                        group_id: detail_group,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                mix: 0.25,
                ..ResidualParams::default()
            }))
            .unwrap();

        // Both slots reach the tombstone machinery, in slot order. A one-value
        // walker would report only the structure route.
        assert_eq!(
            saved.selected_layer_positions().collect::<Vec<_>>(),
            vec![saved_position(6), saved_position(9)]
        );
        assert_eq!(
            saved.referenced_group_ids().collect::<Vec<_>>(),
            vec![structure_group, detail_group]
        );

        // Slot indices are the route identity; an unknown index names no route
        // at all rather than aliasing a real one.
        let VisualNodeKind::Residual(mut params) = saved.get(layer_node).unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(
            params.route(RESIDUAL_STRUCTURE_SLOT),
            Some(params.structure)
        );
        assert_eq!(params.route(RESIDUAL_DETAIL_SLOT), Some(params.detail));
        assert_eq!(params.route(2), None);
        assert_eq!(params.route(u8::MAX), None);
        assert_eq!(params.routes(), [params.structure, params.detail]);
        assert!(params.route_mut(2).is_none());
        params.route_mut(RESIDUAL_DETAIL_SLOT).unwrap().source = SavedImageSource::CleanProgram;
        assert_eq!(params.detail.source, SavedImageSource::CleanProgram);
        assert_eq!(
            params.structure.source,
            SavedImageSource::SelectedLayer {
                layer_position: saved_position(6),
                stage: LayerImageStage::PostLocalEffects,
            },
            "editing one slot never touches the other"
        );

        // Deleting one group tombstones only that slot, and the tombstone never
        // rebinds to a replacement at the same identity.
        saved.mark_group_output_missing(detail_group);
        let VisualNodeKind::Residual(params) = saved.get(group_node).unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(
            params.structure.source,
            SavedImageSource::GroupOutput {
                group_id: structure_group
            }
        );
        assert_eq!(
            params.detail.source,
            SavedImageSource::MissingGroupOutput {
                group_id: detail_group
            }
        );
        assert_eq!(
            saved.referenced_group_ids().collect::<Vec<_>>(),
            vec![structure_group, detail_group],
            "a tombstone still names its saved identity"
        );

        // Resolution binds each live donor independently; a vacated position
        // never retargets the other slot.
        let structure_live = live_id(77);
        let detail_live = live_id(78);
        let mut runtime = saved.resolve_routes(
            |position| {
                if position == saved_position(6) {
                    Some(structure_live)
                } else if position == saved_position(9) {
                    Some(detail_live)
                } else {
                    None
                }
            },
            |group| group == structure_group,
        );
        assert_eq!(
            runtime.selected_layer_ids().collect::<Vec<_>>(),
            vec![structure_live, detail_live]
        );
        assert_eq!(
            runtime.referenced_group_ids().collect::<Vec<_>>(),
            vec![structure_group, detail_group]
        );
        let RuntimeVisualNodeKind::Residual(mut live) = runtime.iter().next().unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(live.route(RESIDUAL_STRUCTURE_SLOT), Some(live.structure));
        assert_eq!(live.route(RESIDUAL_DETAIL_SLOT), Some(live.detail));
        assert_eq!(live.route(9), None);
        assert_eq!(live.routes(), [live.structure, live.detail]);
        assert!(live.route_mut(9).is_none());
        assert_eq!(
            live.detail.timing,
            EdgeTiming::PreviousFrame,
            "each slot keeps its own edge timing"
        );

        // Removing one live donor tombstones only its slot.
        runtime.mark_layer_output_missing(structure_live);
        assert_eq!(
            runtime.selected_layer_ids().collect::<Vec<_>>(),
            vec![detail_live]
        );
        let RuntimeVisualNodeKind::Residual(params) = runtime.iter().next().unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(
            params.structure.source,
            ResolvedImageSource::MissingSelectedLayer {
                saved_position: saved_position(6),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert!(matches!(
            params.detail.source,
            ResolvedImageSource::SelectedLayer {
                layer_id,
                ..
            } if layer_id == detail_live
        ));

        // Capture preserves the missing identity rather than inventing a donor,
        // and never writes a live id back into the patch.
        let recaptured = runtime
            .capture_routes(|id| (id == detail_live).then_some(saved_position(9)))
            .unwrap();
        let VisualNodeKind::Residual(params) = recaptured.iter().next().unwrap().kind else {
            panic!("residual node")
        };
        assert_eq!(
            params.structure.source,
            SavedImageSource::MissingSelectedLayer {
                saved_position: saved_position(6),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            params.detail.source,
            SavedImageSource::SelectedLayer {
                layer_position: saved_position(9),
                stage: LayerImageStage::PreLocalEffects,
            }
        );
        assert_eq!(params.mix, 0.5, "route capture preserves the values");
        assert_eq!(params.detail.timing, EdgeTiming::PreviousFrame);
    }

    #[test]
    fn residual_exposes_only_two_modulatable_dice_eligible_parameters() {
        let descriptors: Vec<_> = NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == NodeKindTag::Residual)
            .collect();
        assert_eq!(descriptors.len(), 7);

        let continuous: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.modulatable)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(continuous, vec!["mix", "detail_gain"]);
        let diceable: Vec<_> = descriptors
            .iter()
            .filter(|descriptor| descriptor.dice_eligible)
            .map(|descriptor| descriptor.key)
            .collect();
        assert_eq!(diceable, vec!["mix", "detail_gain"]);

        // Both modulatable keys are unique across the whole registry, so a
        // route authored for another kind can never cross-resolve onto this
        // one through first-modulatable-row key resolution.
        for key in ["mix", "detail_gain"] {
            assert_eq!(
                NODE_PARAM_DESCRIPTORS
                    .iter()
                    .filter(|descriptor| descriptor.key == key)
                    .count(),
                1,
                "{key} must remain a unique modulation wire key"
            );
        }

        for descriptor in &descriptors {
            match descriptor.key {
                "mix" => {
                    assert_eq!(descriptor.range, Some([0.0, 1.0]));
                    assert_eq!(descriptor.default, Some(0.0));
                    assert_eq!(descriptor.value_type, NodeParamType::Float);
                }
                "detail_gain" => {
                    assert_eq!(descriptor.range, Some([0.0, 4.0]));
                    assert_eq!(descriptor.default, Some(1.0));
                    assert_eq!(descriptor.value_type, NodeParamType::Float);
                }
                // Routes, discrete laws, and the quantization seed are stable
                // authored topology: enumerable but never modulatable, never
                // diced, and never ranged.
                "structure_tap" | "detail_tap" => {
                    assert_eq!(descriptor.value_type, NodeParamType::ImageTap);
                    assert!(descriptor.range.is_none());
                }
                "block" | "quantization" => {
                    assert_eq!(descriptor.value_type, NodeParamType::Enum);
                    assert!(descriptor.range.is_none());
                }
                "seed" => {
                    assert_eq!(descriptor.value_type, NodeParamType::Unsigned);
                    assert!(descriptor.range.is_none());
                }
                other => panic!("unexpected residual parameter {other}"),
            }
        }
    }

    #[test]
    fn residual_nodes_stay_inside_the_rack_lookup_and_pass_ceilings() {
        // The per-node ledger delta is exact, not a bound.
        let ledger = node_kind_descriptor(NodeKindTag::Residual).budget;
        assert_eq!(ledger.full_frame_passes, 1);
        assert_eq!(ledger.logical_texture_lookups_per_pixel, 3);
        assert_eq!(ledger.texture_samples_per_pixel, 12);
        assert_eq!(ledger.sampled_textures_in_pass, 3);
        assert_eq!(ledger.cross_input_taps, 2);
        assert_eq!(ledger.reduced_resolution_passes, 2);
        assert_eq!(ledger.reduced_resolution_surfaces, 2);

        let mut rack = VisualRack::empty();
        for _ in 0..MAX_NODES_PER_RACK {
            rack.push(VisualNodeKind::Residual(ResidualParams {
                mix: 0.5,
                ..ResidualParams::default()
            }))
            .unwrap();
        }
        let budget = rack.resource_budget().unwrap();
        assert_eq!(budget.full_frame_passes, MAX_NODES_PER_RACK as u32);
        assert_eq!(
            budget.logical_texture_lookups_per_pixel,
            3 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(
            budget.texture_samples_per_pixel,
            12 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(budget.max_sampled_textures_in_pass, 3);
        assert_eq!(budget.cross_input_taps, 2 * MAX_NODES_PER_RACK as u32);
        assert_eq!(
            budget.reduced_resolution_passes,
            2 * MAX_NODES_PER_RACK as u32
        );
        assert_eq!(
            budget.reduced_resolution_surfaces,
            2 * MAX_NODES_PER_RACK as u32
        );
        assert!(budget.logical_texture_lookups_per_pixel <= MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK);
        assert!(budget.texture_samples_per_pixel <= MAX_TEXTURE_SAMPLES_PER_RACK);
        assert!(budget.max_sampled_textures_in_pass <= MAX_SAMPLED_TEXTURES_PER_PASS);
        assert!(budget.cross_input_taps as usize <= MAX_CURRENT_IMAGE_TAPS);

        // A full rack of Residual nodes still fits the fixed three-texture rack
        // bind layout exactly; it must never be admitted by raising the cap.
        assert_eq!(MAX_SAMPLED_TEXTURES_PER_PASS, 3);

        // The runtime twin declares the identical ledger.
        let runtime = rack.resolve_routes(|_| None, |_| false);
        assert_eq!(runtime.resource_budget().unwrap(), budget);
    }

    #[test]
    fn residual_block_mean_grids_derive_from_the_block_vocabulary_and_reject_over_budget_edges() {
        // The grid is ceil(output / block edge) on each axis, so the closed
        // vocabulary alone fixes the reduction factor.
        for (block, expected) in [
            (ResidualBlock::Four, [480_u32, 270_u32]),
            (ResidualBlock::Eight, [240, 135]),
            (ResidualBlock::Sixteen, [120, 68]),
            (ResidualBlock::ThirtyTwo, [60, 34]),
            (ResidualBlock::SixtyFour, [30, 17]),
        ] {
            let grid = ResidualGrid::for_output([1920, 1080], block).unwrap();
            assert_eq!([grid.width, grid.height], expected, "{block:?}");
            assert_eq!(grid.block_pixels, block.edge());
            assert_eq!(
                grid.cell_count,
                u64::from(expected[0]) * u64::from(expected[1])
            );
        }

        // Zero is not a grid.
        assert_eq!(
            ResidualGrid::for_output([0, 8], ResidualBlock::Four),
            Err(ResidualResourceError::InvalidDimensions([0, 8]))
        );

        // The edge bound holds at exactly the limit and rejects one cell over
        // rather than clamping onto a coarser grid the author never chose.
        let widest =
            ResidualGrid::for_output([RESIDUAL_GRID_MAX_EDGE * 4, 4], ResidualBlock::Four).unwrap();
        assert_eq!(widest.width, RESIDUAL_GRID_MAX_EDGE);
        assert_eq!(
            ResidualGrid::for_output([RESIDUAL_GRID_MAX_EDGE * 4 + 1, 4], ResidualBlock::Four),
            Err(ResidualResourceError::GridEdge {
                dimensions: [RESIDUAL_GRID_MAX_EDGE + 1, 1],
                limit: RESIDUAL_GRID_MAX_EDGE,
            })
        );

        // The cell bound is independent of the edge bound: 2048 x 1025 is
        // inside every edge and inside the cell cap, and one further cell row
        // is over the cell cap while still inside every edge.
        let inside = ResidualGrid::for_output([8192, 4100], ResidualBlock::Four).unwrap();
        assert_eq!([inside.width, inside.height], [2048, 1025]);
        assert_eq!(inside.cell_count, 2_099_200);
        assert!(inside.cell_count <= RESIDUAL_GRID_MAX_CELLS);
        assert_eq!(
            ResidualGrid::for_output([8192, 4104], ResidualBlock::Four),
            Err(ResidualResourceError::CellCount {
                count: 2_101_248,
                limit: RESIDUAL_GRID_MAX_CELLS,
            })
        );
    }

    #[test]
    fn residual_resource_plan_charges_exact_reduced_bytes_and_binds_before_the_cell_cap() {
        // Two 240x135 means at eight bytes a cell is the entire charge. No
        // full-frame layer appears in this ledger at all.
        let plan = ResidualResourcePlan::preflight(
            &[ResidualResourceRequest {
                output_dimensions: [1920, 1080],
                block: ResidualBlock::Eight,
            }],
            ResidualResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(plan.active_nodes, 1);
        assert_eq!(plan.mean_surfaces, 2);
        assert_eq!(plan.mean_cells, 240 * 135 * 2);
        assert_eq!(plan.mean_sample_operations, 240 * 135 * 2 * 4);
        assert_eq!(plan.mean_surface_bytes, 518_400);
        assert_eq!(plan.total_bytes, 518_400);
        assert_eq!(plan.max_grid_dimensions, [240, 135]);
        assert_eq!(plan.bytes_per_cell, RESIDUAL_MEAN_BYTES_PER_CELL);
        assert_eq!(plan.surfaces_per_node, RESIDUAL_MEAN_SURFACES_PER_NODE);
        assert_eq!(
            plan.sampled_textures_in_recombination,
            RESIDUAL_RECOMBINATION_SAMPLED_TEXTURES
        );
        assert_eq!(plan.uniform_stride_bytes, RESIDUAL_UNIFORM_STRIDE_BYTES);

        // A composition with no live node charges nothing while still
        // declaring every exact row.
        let dormant =
            ResidualResourcePlan::preflight(&[], ResidualResourceLimits::default()).unwrap();
        assert_eq!(dormant, ResidualResourcePlan::default());
        assert_eq!(dormant.total_bytes, 0);

        // The per-node byte cap binds before the cell cap. A 2048 x 1024 grid
        // is 2,097,152 cells — inside the 2,100,000 cell bound — and is
        // exactly the 32 MiB node bound.
        let exact = ResidualResourcePlan::preflight(
            &[ResidualResourceRequest {
                output_dimensions: [8192, 4096],
                block: ResidualBlock::Four,
            }],
            ResidualResourceLimits::default(),
        )
        .unwrap();
        assert_eq!(exact.mean_cells, 2_097_152 * 2);
        assert_eq!(exact.total_bytes, RESIDUAL_NODE_MAX_BYTES);

        // One further cell row is still inside the cell bound and is rejected
        // by the byte bound, so neither bound is derived from the other.
        assert_eq!(
            ResidualResourcePlan::preflight(
                &[ResidualResourceRequest {
                    output_dimensions: [8192, 4100],
                    block: ResidualBlock::Four,
                }],
                ResidualResourceLimits::default(),
            ),
            Err(ResidualResourceError::NodeBytes {
                bytes: 33_587_200,
                limit: RESIDUAL_NODE_MAX_BYTES,
            })
        );
        const { assert!(2_099_200 <= RESIDUAL_GRID_MAX_CELLS) };
        const { assert!(33_587_200 > RESIDUAL_NODE_MAX_BYTES) };

        // A full-cap grid is 16,800,000 mean sample operations: 2,100,000
        // cells, four quadrant taps each, across both surfaces. It is a
        // nominal arithmetic bound, not an admissible node.
        assert_eq!(
            RESIDUAL_GRID_MAX_CELLS * RESIDUAL_MEAN_TAPS_PER_CELL * 2,
            16_800_000
        );
        const {
            assert!(
                RESIDUAL_GRID_MAX_CELLS * RESIDUAL_MEAN_BYTES_PER_CELL * 2
                    > RESIDUAL_NODE_MAX_BYTES
            )
        };
    }

    #[test]
    fn residual_resource_plan_rejects_every_independent_limit_one_unit_over() {
        let tiny = |count: usize| {
            vec![
                ResidualResourceRequest {
                    output_dimensions: [64, 64],
                    block: ResidualBlock::SixtyFour,
                };
                count
            ]
        };

        // Nominal active-node bound.
        assert!(ResidualResourcePlan::preflight(
            &tiny(RESIDUAL_MAX_ACTIVE_NODES as usize),
            ResidualResourceLimits::default(),
        )
        .is_ok());
        assert_eq!(
            ResidualResourcePlan::preflight(
                &tiny(RESIDUAL_MAX_ACTIVE_NODES as usize + 1),
                ResidualResourceLimits::default(),
            ),
            Err(ResidualResourceError::TooManyActiveNodes {
                count: RESIDUAL_MAX_ACTIVE_NODES + 1,
                limit: RESIDUAL_MAX_ACTIVE_NODES,
            })
        );

        // Aggregate byte bound, with every node comfortably inside its own.
        let large = ResidualResourceRequest {
            output_dimensions: [5600, 4000],
            block: ResidualBlock::Four,
        };
        let pair =
            ResidualResourcePlan::preflight(&[large, large], ResidualResourceLimits::default())
                .unwrap();
        assert_eq!(pair.total_bytes, 44_800_000);
        assert_eq!(
            ResidualResourcePlan::preflight(
                &[large, large, large],
                ResidualResourceLimits::default(),
            ),
            Err(ResidualResourceError::AggregateBytes {
                bytes: 67_200_000,
                limit: RESIDUAL_AGGREGATE_MAX_BYTES,
            })
        );
        const { assert!(22_400_000 < RESIDUAL_NODE_MAX_BYTES) };

        // Recombination texture bound, against a device narrower than the pass.
        assert_eq!(
            ResidualResourcePlan::preflight(
                &tiny(1),
                ResidualResourceLimits {
                    max_sampled_textures_per_shader_stage: 2,
                    ..ResidualResourceLimits::default()
                },
            ),
            Err(ResidualResourceError::SampledTextures {
                requested: RESIDUAL_RECOMBINATION_SAMPLED_TEXTURES,
                limit: 2,
            })
        );

        // Frozen uniform stride against the device's dynamic-offset alignment.
        for alignment in [0_u32, 96, 512, RESIDUAL_UNIFORM_STRIDE_BYTES as u32 + 1] {
            assert_eq!(
                ResidualResourcePlan::preflight(
                    &tiny(1),
                    ResidualResourceLimits {
                        min_uniform_buffer_offset_alignment: alignment,
                        ..ResidualResourceLimits::default()
                    },
                ),
                Err(ResidualResourceError::UniformStride {
                    stride: RESIDUAL_UNIFORM_STRIDE_BYTES,
                    alignment,
                })
            );
        }
        for alignment in [1_u32, 64, 128, 256] {
            assert!(ResidualResourcePlan::preflight(
                &tiny(1),
                ResidualResourceLimits {
                    min_uniform_buffer_offset_alignment: alignment,
                    ..ResidualResourceLimits::default()
                },
            )
            .is_ok());
        }

        // Device texture bound on the carrier the grid is reduced from.
        assert_eq!(
            ResidualResourcePlan::preflight(
                &[ResidualResourceRequest {
                    output_dimensions: [8_193, 8],
                    block: ResidualBlock::Four,
                }],
                ResidualResourceLimits::default(),
            ),
            Err(ResidualResourceError::DeviceTextureDimension {
                dimensions: [8_193, 8],
                limit: 8_192,
            })
        );
        assert_eq!(
            ResidualResourcePlan::preflight(
                &[ResidualResourceRequest {
                    output_dimensions: [0, 8],
                    block: ResidualBlock::Four,
                }],
                ResidualResourceLimits::default(),
            ),
            Err(ResidualResourceError::InvalidDimensions([0, 8]))
        );

        // The grid edge bound survives a device wide enough to reach it.
        assert_eq!(
            ResidualResourcePlan::preflight(
                &[ResidualResourceRequest {
                    output_dimensions: [RESIDUAL_GRID_MAX_EDGE * 4 + 4, 4],
                    block: ResidualBlock::Four,
                }],
                ResidualResourceLimits {
                    max_texture_dimension_2d: 16_384,
                    ..ResidualResourceLimits::default()
                },
            ),
            Err(ResidualResourceError::GridEdge {
                dimensions: [RESIDUAL_GRID_MAX_EDGE + 1, 1],
                limit: RESIDUAL_GRID_MAX_EDGE,
            })
        );
    }

    #[test]
    fn residual_resource_plan_reconciles_actual_allocations_and_fails_closed() {
        let plan = ResidualResourcePlan::preflight(
            &[ResidualResourceRequest {
                output_dimensions: [1920, 1080],
                block: ResidualBlock::Eight,
            }],
            ResidualResourceLimits::default(),
        )
        .unwrap();
        let honest = ResidualAllocationSnapshot {
            mean_surfaces: plan.mean_surfaces,
            bytes_per_cell: RESIDUAL_MEAN_BYTES_PER_CELL,
            surfaces_per_node: RESIDUAL_MEAN_SURFACES_PER_NODE,
            uniform_stride_bytes: RESIDUAL_UNIFORM_STRIDE_BYTES,
            total_bytes: plan.total_bytes,
        };
        assert_eq!(plan.reconcile(honest), Ok(()));

        // Every exact row is an equality, so one unit either way fails closed.
        assert_eq!(
            plan.reconcile(ResidualAllocationSnapshot {
                bytes_per_cell: RESIDUAL_MEAN_BYTES_PER_CELL + 1,
                ..honest
            }),
            Err(ResidualResourceError::AllocatedCellBytes {
                allocated: RESIDUAL_MEAN_BYTES_PER_CELL + 1,
                expected: RESIDUAL_MEAN_BYTES_PER_CELL,
            })
        );
        assert_eq!(
            plan.reconcile(ResidualAllocationSnapshot {
                surfaces_per_node: RESIDUAL_MEAN_SURFACES_PER_NODE + 1,
                ..honest
            }),
            Err(ResidualResourceError::AllocatedSurfacesPerNode {
                allocated: RESIDUAL_MEAN_SURFACES_PER_NODE + 1,
                expected: RESIDUAL_MEAN_SURFACES_PER_NODE,
            })
        );
        assert_eq!(
            plan.reconcile(ResidualAllocationSnapshot {
                uniform_stride_bytes: RESIDUAL_UNIFORM_STRIDE_BYTES * 2,
                ..honest
            }),
            Err(ResidualResourceError::AllocatedUniformStride {
                allocated: RESIDUAL_UNIFORM_STRIDE_BYTES * 2,
                expected: RESIDUAL_UNIFORM_STRIDE_BYTES,
            })
        );
        assert_eq!(
            plan.reconcile(ResidualAllocationSnapshot {
                mean_surfaces: plan.mean_surfaces + 1,
                ..honest
            }),
            Err(ResidualResourceError::AllocatedSurfacesPerNode {
                allocated: plan.mean_surfaces + 1,
                expected: plan.mean_surfaces,
            })
        );
        assert_eq!(
            plan.reconcile(ResidualAllocationSnapshot {
                total_bytes: plan.total_bytes - 1,
                ..honest
            }),
            Err(ResidualResourceError::AllocatedBytes {
                allocated: plan.total_bytes - 1,
                planned: plan.total_bytes,
            })
        );
    }
}
