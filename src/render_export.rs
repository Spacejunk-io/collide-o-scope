//! Offline high-quality patch renderer.
//!
//! Renders a patch (layer configs + master effects + NTSC) to an MP4 file
//! at configurable resolution and duration using a headless wgpu device
//! and piping raw RGBA frames to ffmpeg.

use std::io::{Read as IoRead, Write as IoWrite};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::composition::{CompositionTree, RuntimeComposition};
use crate::effects::EffectUniforms;
use crate::evaluated_frame::evaluated_composition::{
    AdvancedCompositionPlan, CompositionPlanInput, EvaluatedCompositionPlan, TemporalMasterPath,
};
use crate::evaluated_frame::{
    EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, ResolvedImageInput,
    SourceTap,
};
use crate::image_routing::{LayerMatte, StableLayerId};
use crate::layers::BlendMode;
use crate::media_safety::{MediaDeviceLimits, MediaSafetyPolicy};
#[cfg(test)]
use crate::ntsc::{
    plan_selective_ntsc, NtscFrameMetadata, SelectiveNtscGeneration, SelectiveNtscLayerDescriptor,
};
use crate::ntsc::{
    process_selective_ntsc_batch_with_state_and_resolution, reference_frame_for_output,
    NtscExportQuality, NtscState, SelectiveNtscBatch, SelectiveNtscPlan,
};
use crate::patch::PatchState;
use crate::performance::SavedLayerPosition;
use crate::renderer::blend::composite_shader_source;
use crate::renderer::composition::{
    CompositionEncodeKind, CompositionFrameTiming, CompositionGpuExecutor,
    CompositionMotionFrameInput, CompositionPreparedKind, CompositionSourceDescriptor,
    COMPOSITION_PRESENT_FORMAT,
};
use crate::renderer::compositor::{
    encode_matte_composite, encode_program_history_copy, validate_selective_matte_topology,
    ImageRoutingGpuResources, ImageTapTexture, MatteCompositePipeline, MatteCompositeUniforms,
    MatteResourceLimits, MatteResourcePlan,
};
use crate::renderer::state::{
    conditional_layer_slots, master_fx_composition_path, temporal_bypass_overlay_active,
    temporal_bypass_overlay_indices, visible_stack_indices, MasterFxCompositionPath,
};
use crate::spatial::{EffectPassUniforms, SpatialTransform};
use crate::transport::{
    ClipTransportConfig, ClipTransportState, FrameSelection, ProgramTransportTick,
    TransportTimeline,
};
use crate::video::decoder::{validate_media_dimensions, MAX_MEDIA_PIXELS};
#[cfg(test)]
use crate::video::CodecMotionFrame;
use crate::video::{CodecFrameIdentity, CodecMotionProduct, DecodedStillImage, VideoDecoder};
use crate::visual_rack::{CreativeResourceLimits, RuntimeVisualRack};

/// App shutdown must not wait forever on a wedged graphics backend. By the
/// time this expires ExportJob::cancel has already killed/reaped ffmpeg,
/// removed its partial output, and forbidden any later encoder registration.
const EXPORT_DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(1);
/// Public export presets top out at UHD. Keep arbitrary protocol input from
/// allocating a pathological set of full-frame GPU intermediates.
const MAX_EXPORT_EDGE: u32 = 16_384;
const OUTCOME_RUNNING: u8 = 0;
const OUTCOME_CANCEL_REQUESTED: u8 = 1;
const OUTCOME_SUCCEEDED: u8 = 2;
const OUTCOME_FAILED: u8 = 3;
const OUTCOME_CANCELLED: u8 = 4;
/// Keep protocol snapshots bounded even when a hand-authored legacy patch
/// contains an excessive number of unavailable layers.
const MAX_EXPORT_WARNINGS: usize = 128;
const MAX_EXPORT_WARNING_CHARS: usize = 1_024;
/// Bumped from 3 when the sidecar began recording per-slot rack route
/// identity. Every schema-3 key keeps its name and meaning; schema 4 is purely
/// additive and adds two independent sections:
///
/// - `authored_residual_nodes`: the per-slot resolved/missing route identity
///   and the discrete recombination law of every Residual Counterpoint node
///   admitted by this job.
/// - `symmetry_fields`: the per-slot image and motion route identity of every
///   Symmetry Field node admitted by this job.
///
/// Both landed against the same schema-3 base, so there is one schema 4 that
/// carries both sections rather than two competing definitions.
///
/// Bumped from 4 when the sidecar began recording the Field Collider. Every
/// schema-4 key keeps its name and meaning; schema 5 is purely additive and
/// adds one section, following the same 3 -> 4 precedent:
///
/// - `field_collider`: the authored identity of both collider inputs, the
///   admitted output slot, the typed diagnostic, the byte-exact resource
///   delta, and the discrete recombination law of the one collider this job
///   admitted. Absent when the job ran the exact M4 recipe.
///
/// The section carries authored identity, topology, diagnostics, and budgets
/// only. Derived vectors, the transient mapped pair, gate parities, and raw
/// codec records are deliberately never recorded, and neither is any host path
/// or filesystem metadata.
///
/// Bumped from 5 when the sidecar began recording the B5 codec mosh. Every
/// schema-5 key keeps its name and meaning; schema 6 is purely additive and
/// adds one section, following the same precedent:
///
/// - `codec_mosh`: the authored recipe (the codec controls, motion-wake
///   shaping, and discrete recycle law), the encode dimensions, and encoder
///   identity (`mpeg4/avcodec-<version>`). Present only when at least one
///   accepted frame ran the round trip. The encoder identity is the honesty
///   record: per-host repeatability is claimed, cross-machine bit-identity
///   is not. The mutated bitstream bytes are deliberately never recorded.
///
/// Schema 7 adds the motion-wipe, vector-smear, retained-trail recipe,
/// accepted-frame min/max recipe observation, and bounded per-layer authored /
/// observed Mosh Send provenance.
const EXPORT_MOTION_SIDECAR_SCHEMA_VERSION: u16 = 7;
const MAX_EXPORT_MOTION_SIDECAR_SOURCES: usize = 256;
const MAX_EXPORT_MOTION_SIDECAR_SCOPES: usize = 256;
const MAX_EXPORT_MOTION_DISTINCT_STATES: usize = 512;
/// Authored Symmetry Field nodes whose route identity is recorded. Racks are
/// bounded per scope, but the number of scopes is not, so the section carries
/// its own cap and truncation flag like every other sidecar list.
const MAX_EXPORT_SYMMETRY_SIDECAR_NODES: usize = 256;
const MAX_EXPORT_MOTION_SIDECAR_BYTES: usize = 4 * 1024 * 1024;
/// One rack holds at most `MAX_NODES_PER_RACK` nodes and a job admits at most
/// `MAX_EXPORT_MOTION_SIDECAR_SCOPES` scopes, so this cap is generous for any
/// admissible patch and still bounds a hand-authored one.
const MAX_EXPORT_RESIDUAL_SIDECAR_NODES: usize = 512;

const fn transparent_accumulation_clear() -> wgpu::Color {
    wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    }
}

/// Configuration for an offline render job.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration_secs: f32,
    pub output_path: String,
    /// Optional media file whose first audio stream is muxed into the MP4.
    ///
    /// Audio transport is deliberately independent from visual layer
    /// transport: it starts at source time zero at 1x, ignores layer
    /// pause/speed/modulation/loop state, is trimmed when long, and is padded
    /// with silence when short. This remains deterministic when visual speed
    /// changes over time and cannot be represented by one audio tempo.
    pub audio_path: Option<String>,
    /// Runtime candidate paired with a content-addressed `audio_path`.
    /// Never persisted or trusted without re-fingerprinting.
    pub audio_path_hint: Option<String>,
    /// Runtime candidates aligned with `PatchState::layers`. These allow a
    /// just-loaded patch-adjacent source to export even when it is outside the
    /// active library, while the saved content identity remains authoritative.
    pub layer_source_hints: Vec<String>,
    /// Runtime candidate for content-addressed imported analysis audio.
    pub analysis_audio_path_hint: Option<String>,
    /// Spatial quality for CPU NTSC processing. This affects both the global
    /// post-composite path and selective inherited-layer processing.
    pub ntsc_quality: NtscExportQuality,
    /// Exact bounded Curved Shutter sample policy for this export. `Authored`
    /// preserves each scope's live tier; explicit variants replace only the
    /// tier after Morph/modulation and before resource preflight.
    pub shutter_samples: ExportShutterSamples,
    /// Shared host-session policy used only when reconstructing saved sources.
    /// Export output dimensions remain governed by `validate_export_dimensions`.
    pub media_safety_policy: MediaSafetyPolicy,
    /// Bounded accepted manual Score/Garden events recorded by the live host.
    /// This is runtime performance data, never inferred from wall time or
    /// serialized into the authored patch.
    pub(crate) temporal_event_track: crate::temporal::TemporalEventTrack,
    /// The recorded gesture performance this job replays, carried as the
    /// portable sidecar document so the recording's canonical checksum travels
    /// with its own event stream.
    ///
    /// `None` is the pre-gesture path: nothing is replayed, no gesture sidecar
    /// is published, and the honesty law holds by construction because an
    /// unrecorded live gesture never reaches this field.
    pub(crate) gesture_track: Option<crate::gesture::GestureTrackDocument>,
    /// The B9 performance take this job replays by reference tick, carried as
    /// its portable checksummed document exactly as the gesture track is.
    /// `None` is the pre-recorder path: nothing is replayed and no
    /// performance sidecar is published.
    pub(crate) performance_take: Option<crate::performance_track::PerformanceTakeDocument>,
}

/// Closed offline Curved Shutter sample policy. The explicit variants name
/// their literal shader sample count so protocol/UI requests cannot imply an
/// approximate or silently substituted quality.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExportShutterSamples {
    #[default]
    #[serde(rename = "authored")]
    Authored,
    #[serde(rename = "samples_1")]
    Samples1,
    #[serde(rename = "samples_4")]
    Samples4,
    #[serde(rename = "samples_8")]
    Samples8,
    #[serde(rename = "samples_16")]
    Samples16,
}

impl ExportShutterSamples {
    pub const fn requested_count(self) -> Option<u8> {
        match self {
            Self::Authored => None,
            Self::Samples1 => Some(1),
            Self::Samples4 => Some(4),
            Self::Samples8 => Some(8),
            Self::Samples16 => Some(16),
        }
    }

    const fn quality_override(self) -> Option<crate::motion::CurvedShutterQuality> {
        match self {
            Self::Authored => None,
            Self::Samples1 => Some(crate::motion::CurvedShutterQuality::Sharp),
            Self::Samples4 => Some(crate::motion::CurvedShutterQuality::Draft),
            Self::Samples8 => Some(crate::motion::CurvedShutterQuality::Live),
            Self::Samples16 => Some(crate::motion::CurvedShutterQuality::High),
        }
    }

    pub const fn is_valid(self) -> bool {
        matches!(
            self,
            Self::Authored | Self::Samples1 | Self::Samples4 | Self::Samples8 | Self::Samples16
        )
    }

    const fn key(self) -> &'static str {
        match self {
            Self::Authored => "authored",
            Self::Samples1 => "samples_1",
            Self::Samples4 => "samples_4",
            Self::Samples8 => "samples_8",
            Self::Samples16 => "samples_16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportMotionScopeIdentity {
    Master,
    Layer {
        saved_position: u32,
        stable_id: u64,
        source_tap_id: u64,
    },
}

/// Bounded, frame-local motion provenance published only after the
/// corresponding encoded frame is accepted. Hidden field/carrier pixels and
/// raw codec records never enter this report.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportMotionScopeMetadata {
    pub scope: ExportMotionScopeIdentity,
    pub algorithm_version: u16,
    pub requested_source: crate::motion::MotionFieldSource,
    pub lattice_quality: crate::motion::MotionLatticeQuality,
    /// Pure planner decision after exact codec preparation/fallback.
    pub source_origin: crate::motion::MotionFieldOrigin,
    /// Field origin actually attached/generated for the accepted GPU frame.
    pub rendered_source_origin: crate::motion::MotionFieldOrigin,
    /// The evaluated scope required a field even when admission or first-frame
    /// priming prevented one from becoming resident.
    pub field_planned: bool,
    pub field_attached: bool,
    pub source_diagnostic: crate::motion::MotionSourceDiagnostic,
    pub codec_provenance: Option<crate::video::CodecMotionProvenance>,
    pub source_generation: Option<u64>,
    pub frame_ordinal: Option<u64>,
    /// Exact digest of the committed codec proof chain and vector payload.
    pub codec_product_sha256: Option<[u8; 32]>,
    /// Number of adjacent decoded transitions represented by the accepted
    /// codec product. `None` means codec metadata was unavailable.
    pub codec_transition_count: Option<u8>,
    pub codec_elapsed_seconds: Option<f32>,
    pub donor_saved_position: Option<u32>,
    pub donor_stable_id: Option<u64>,
    pub carrier: crate::motion::MotionCarrier,
    pub transplant_admitted: bool,
    pub shutter_active: bool,
    pub shutter_angle_degrees: f32,
    pub shutter_quality: crate::motion::CurvedShutterQuality,
    pub shutter_sample_count: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExportMotionMetadata {
    pub accepted_frame: Option<u64>,
    pub algorithm_version: u16,
    pub scopes: Vec<ExportMotionScopeMetadata>,
    pub scopes_truncated: bool,
    /// The one admitted Field Collider, recorded from the shared evaluated
    /// plan. `None` is the exact M4 job.
    pub(crate) field_collider: Option<FieldColliderSidecar>,
}

impl Default for ExportMotionMetadata {
    fn default() -> Self {
        Self {
            accepted_frame: None,
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: Vec::new(),
            scopes_truncated: false,
            field_collider: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct MotionSidecarArtifact {
    file_name: String,
    width: u32,
    height: u32,
    fps: u32,
    duration_seconds: f32,
    requested_shutter_sample_policy: String,
    requested_shutter_sample_count: Option<u8>,
}

impl MotionSidecarArtifact {
    fn from_config(config: &ExportConfig) -> Self {
        Self {
            file_name: std::path::Path::new(&config.output_path)
                .file_name()
                .map(|name| bounded_export_warning(&name.to_string_lossy()))
                .unwrap_or_default(),
            width: config.width,
            height: config.height,
            fps: config.fps,
            duration_seconds: config.duration_secs,
            requested_shutter_sample_policy: config.shutter_samples.key().to_owned(),
            requested_shutter_sample_count: config.shutter_samples.requested_count(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum MotionSidecarScopeIdentity {
    Master,
    Layer {
        saved_position: u32,
        stable_id: u64,
        source_tap_id: u64,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
struct MotionSidecarSource {
    saved_position: u32,
    stable_id: u64,
    source_tap_id: u64,
    kind: String,
    logical_name: String,
    persisted_reference: String,
    fingerprint_sha256: Option<String>,
    fingerprint_bytes: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct MotionSidecarAuthoredScope {
    scope: MotionSidecarScopeIdentity,
    algorithm_version: u16,
    requested_source: String,
    lattice_quality: String,
    carrier: String,
    donor_saved_position: Option<u32>,
    donor_stable_id: Option<u64>,
    shutter_angle_degrees: f32,
    shutter_quality: String,
    shutter_sample_count: u8,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct MotionSidecarScopeState {
    scope: MotionSidecarScopeIdentity,
    algorithm_version: u16,
    requested_source: String,
    lattice_quality: String,
    source_origin: String,
    rendered_source_origin: String,
    field_planned: bool,
    field_attached: bool,
    source_diagnostic: String,
    codec_provenance: Option<String>,
    source_generation: Option<u64>,
    frame_ordinal: Option<u64>,
    codec_product_sha256: Option<String>,
    codec_transition_count: Option<u8>,
    codec_elapsed_seconds: Option<f32>,
    donor_saved_position: Option<u32>,
    donor_stable_id: Option<u64>,
    carrier: String,
    transplant_admitted: bool,
    shutter_active: bool,
    shutter_angle_degrees: f32,
    shutter_quality: String,
    shutter_sample_count: u8,
}

/// One authored Residual Counterpoint route slot as this job resolved it.
/// Only stable identities travel: a saved position, a job-local stable ID, or
/// a group ID. Operational paths and filesystem metadata never enter here.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct ResidualSidecarRoute {
    slot: u8,
    slot_name: &'static str,
    source: &'static str,
    timing: crate::visual_rack::EdgeTiming,
    /// False only for a retained tombstone. A tombstone keeps its saved
    /// provenance and is never rebound onto a neighbouring source.
    resolved: bool,
    saved_position: Option<u32>,
    stable_id: Option<u64>,
    group_id: Option<u64>,
    stage: Option<crate::image_routing::LayerImageStage>,
}

/// One admitted Residual Counterpoint node. `authored_mix` and
/// `authored_detail_gain` are the values this job admitted, before Morph and
/// stable modulation are projected into the per-frame graph; every other
/// field is stable authored topology that neither can move.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct ResidualSidecarNode {
    scope: MotionSidecarScopeIdentity,
    node_id: u64,
    enabled: bool,
    wet: f32,
    algorithm_version: u16,
    block: crate::visual_rack::ResidualBlock,
    block_edge: u32,
    quantization: crate::visual_rack::ResidualQuantization,
    quantization_levels: u32,
    authored_mix: f32,
    authored_detail_gain: f32,
    seed: u32,
    /// True when the admitted values make the node an exact delegation: no
    /// tap, no pass, no mean surface, and no saved dependency edge.
    exact_bypass: bool,
    routes: Vec<ResidualSidecarRoute>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct MotionSidecarDistinctState {
    first_accepted_frame: u64,
    state: MotionSidecarScopeState,
}

#[derive(Debug, Clone, serde::Serialize)]
struct MotionSidecarLastFrame {
    accepted_frame: u64,
    scopes: Vec<MotionSidecarScopeState>,
    scopes_truncated: bool,
}

/// One collider input, recorded by named slot.
///
/// Both slots are emitted always and never compacted, so a tombstone is
/// recorded as a tombstone rather than re-resolved against whatever now
/// occupies the vacated position. `resolved` false with a `saved_position`
/// present is exactly that tombstone.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct FieldColliderSidecarInput {
    /// The closed named slot token, never an index.
    slot: &'static str,
    resolved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    saved_position: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stable_id: Option<u64>,
    /// The admitted primitive field slot this input was read from.
    #[serde(skip_serializing_if = "Option::is_none")]
    field_slot: Option<u8>,
}

/// The one Field Collider this job admitted, recorded after an accepted frame.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct FieldColliderSidecar {
    algorithm_version: u16,
    recipient: MotionSidecarScopeIdentity,
    admitted: bool,
    /// The discrete recombination law, as its closed snake_case token.
    mode: &'static str,
    boundary: &'static str,
    /// Derived slots append after every primitive field slot.
    output_slot: u8,
    output_grid: [u32; 2],
    inputs: Vec<FieldColliderSidecarInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostic: Option<String>,
    /// The byte-exact collider-specific delta, twenty bytes per grid cell.
    bytes_per_cell: u64,
    derived_vector_bytes: u64,
    derived_gate_bytes: u64,
    transient_pair_bytes: u64,
    total_bytes: u64,
    low_resolution_passes: u32,
    nearest_lookups: u32,
    max_sampled_textures_in_pass: u32,
}

/// The rack scope that owns one Symmetry Field. Unlike the motion scope
/// identity this carries a group arm, because a rack — and therefore a
/// dedicated pass — can be authored at master, layer, or group scope.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SymmetrySidecarScope {
    Master,
    Layer { saved_position: u32, stable_id: u64 },
    Group { group_id: u64 },
}

/// One image slot of one Symmetry Field, recorded by slot index.
///
/// Slot index is route identity, so the record is emitted for both slots
/// always — including an unarmed one — and never compacted. `resolved` is the
/// single fact an operator needs: false means the authored donor did not
/// resolve for this job and the slot bound the neutral view, with
/// `saved_position` retained so the tombstone is legible and can never rebind.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct SymmetrySidecarImageRoute {
    slot: u8,
    /// The source-mask bit. An unarmed slot claims no dependency edge and
    /// reserves no binding, so it is recorded but never counted as missing.
    armed: bool,
    source: &'static str,
    timing: &'static str,
    resolved: bool,
    stable_id: Option<u64>,
    saved_position: Option<u32>,
    group_id: Option<u64>,
    stage: Option<&'static str>,
}

/// One motion slot of one Symmetry Field, recorded by slot index.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct SymmetrySidecarMotionRoute {
    slot: u8,
    /// The motion-mask bit. An unarmed slot requests no primitive
    /// vector/gate field at all.
    armed: bool,
    donor: &'static str,
    resolved: bool,
    stable_id: Option<u64>,
    saved_position: Option<u32>,
}

/// Authored provenance for one Symmetry Field node, resolved once per job by
/// [`resolve_export_creative_graph`]. Morph may only carry values, never
/// routes, so this record is complete for the whole render and is not
/// re-observed per frame.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct SymmetrySidecarNode {
    scope: SymmetrySidecarScope,
    node_id: u64,
    enabled: bool,
    wet: f32,
    mode: crate::symmetry::SymmetryMode,
    boundary: crate::symmetry::SymmetryBoundary,
    seed: u32,
    exact_bypass: bool,
    image_routes: Vec<SymmetrySidecarImageRoute>,
    motion_routes: Vec<SymmetrySidecarMotionRoute>,
}

#[derive(Debug, serde::Serialize)]
struct ExportMotionSidecar {
    schema_version: u16,
    artifact: MotionSidecarArtifact,
    algorithm_version: u16,
    cross_gpu_pixel_identity_guaranteed: bool,
    sources: Vec<MotionSidecarSource>,
    sources_truncated: bool,
    authored_scopes: Vec<MotionSidecarAuthoredScope>,
    authored_scopes_truncated: bool,
    authored_residual_nodes: Vec<ResidualSidecarNode>,
    authored_residual_nodes_truncated: bool,
    distinct_dynamic_states: Vec<MotionSidecarDistinctState>,
    distinct_dynamic_states_truncated: bool,
    symmetry_fields: Vec<SymmetrySidecarNode>,
    symmetry_fields_truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_collider: Option<FieldColliderSidecar>,
    #[serde(skip_serializing_if = "Option::is_none")]
    codec_mosh: Option<CodecMoshSidecar>,
    last_accepted_frame: Option<MotionSidecarLastFrame>,
    warnings: Vec<String>,
}

/// B5 codec-mosh export provenance: authored bases, accepted-frame recipe/send
/// bounds, and encoder identity. Recorded once per job, only after an accepted
/// frame actually ran the round trip; the bitstream itself never enters the
/// sidecar.
#[derive(Debug, Clone, serde::Serialize)]
struct CodecMoshSidecar {
    /// `mpeg4/avcodec-<linked version>` — the per-host repeatability record.
    encoder: String,
    encode_width: u32,
    encode_height: u32,
    amount: f32,
    key_removal: f32,
    hold: f32,
    drop: f32,
    shuffle: f32,
    rate: f32,
    bitrate_starve: f32,
    resync: f32,
    wipe: f32,
    smear: f32,
    trail: f32,
    recycle: bool,
    layer_sends: Vec<CodecMoshLayerSendSidecar>,
    layer_sends_truncated: bool,
    /// Actual frame-evaluated recipe bounds. The scalar fields above retain
    /// the authored patch-base shape for schema compatibility; Morph,
    /// modulation, and replay may move the recipe during the render.
    observed: CodecMoshObservedSidecar,
}

#[derive(Debug, Clone, serde::Serialize)]
struct CodecMoshLayerSendSidecar {
    saved_position: u32,
    stable_id: u64,
    authored: f32,
    observed_min: f32,
    observed_max: f32,
    /// True iff this layer was visible, positive-opacity, and outside the
    /// exact Temporal dry prefix on at least one accepted Mosh frame.
    entered_codec_mosh: bool,
}

#[derive(Debug, Clone, Copy)]
struct CodecMoshLayerSendObservation {
    min: f32,
    max: f32,
    seen: bool,
    entered_codec_mosh: bool,
}

impl Default for CodecMoshLayerSendObservation {
    fn default() -> Self {
        Self {
            min: 1.0,
            max: 1.0,
            seen: false,
            entered_codec_mosh: false,
        }
    }
}

impl CodecMoshLayerSendObservation {
    fn observe(&mut self, send: f32, entered_codec_mosh: bool) {
        let send = crate::layers::clamp_layer_mosh_send(send);
        if self.seen {
            self.min = self.min.min(send);
            self.max = self.max.max(send);
        } else {
            self.min = send;
            self.max = send;
            self.seen = true;
        }
        self.entered_codec_mosh |= entered_codec_mosh;
    }
}

fn codec_mosh_authored_layer_send(
    patch: &crate::patch::PatchState,
    source_index: usize,
) -> Result<f32, String> {
    patch
        .layers
        .get(source_index)
        .map(|layer| crate::layers::clamp_layer_mosh_send(layer.mosh_send))
        .ok_or_else(|| {
            format!(
                "Codec-Mosh sidecar layer source index {source_index} is absent from the immutable saved patch"
            )
        })
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
struct CodecMoshContinuousSidecar {
    amount: f32,
    key_removal: f32,
    hold: f32,
    drop: f32,
    shuffle: f32,
    rate: f32,
    bitrate_starve: f32,
    resync: f32,
    wipe: f32,
    smear: f32,
    trail: f32,
}

impl From<crate::codec_mosh::CodecMoshParams> for CodecMoshContinuousSidecar {
    fn from(params: crate::codec_mosh::CodecMoshParams) -> Self {
        let params = params.sanitized();
        Self {
            amount: params.amount,
            key_removal: params.key_removal,
            hold: params.hold,
            drop: params.drop,
            shuffle: params.shuffle,
            rate: params.rate,
            bitrate_starve: params.bitrate_starve,
            resync: params.resync,
            wipe: params.wipe,
            smear: params.smear,
            trail: params.trail,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct CodecMoshObservedSidecar {
    accepted_frames: u64,
    min: CodecMoshContinuousSidecar,
    max: CodecMoshContinuousSidecar,
    recycle_false_seen: bool,
    recycle_true_seen: bool,
}

#[derive(Debug, Clone)]
struct CodecMoshObservationAccumulator {
    observed: CodecMoshObservedSidecar,
}

impl CodecMoshObservationAccumulator {
    fn new(params: crate::codec_mosh::CodecMoshParams) -> Self {
        let params = params.sanitized();
        let values = CodecMoshContinuousSidecar::from(params);
        Self {
            observed: CodecMoshObservedSidecar {
                accepted_frames: 1,
                min: values,
                max: values,
                recycle_false_seen: !params.recycle,
                recycle_true_seen: params.recycle,
            },
        }
    }

    fn observe(&mut self, params: crate::codec_mosh::CodecMoshParams) {
        let params = params.sanitized();
        let values = CodecMoshContinuousSidecar::from(params);
        macro_rules! widen {
            ($field:ident) => {
                self.observed.min.$field = self.observed.min.$field.min(values.$field);
                self.observed.max.$field = self.observed.max.$field.max(values.$field);
            };
        }
        widen!(amount);
        widen!(key_removal);
        widen!(hold);
        widen!(drop);
        widen!(shuffle);
        widen!(rate);
        widen!(bitrate_starve);
        widen!(resync);
        widen!(wipe);
        widen!(smear);
        widen!(trail);
        self.observed.accepted_frames = self.observed.accepted_frames.saturating_add(1);
        self.observed.recycle_false_seen |= !params.recycle;
        self.observed.recycle_true_seen |= params.recycle;
    }

    fn finish(self) -> CodecMoshObservedSidecar {
        self.observed
    }
}

struct ExportMotionSidecarAccumulator {
    artifact: MotionSidecarArtifact,
    sources: Vec<MotionSidecarSource>,
    sources_truncated: bool,
    authored_scopes: Vec<MotionSidecarAuthoredScope>,
    authored_scopes_truncated: bool,
    residual_nodes: Vec<ResidualSidecarNode>,
    residual_nodes_truncated: bool,
    distinct: Vec<MotionSidecarDistinctState>,
    distinct_truncated: bool,
    symmetry_fields: Vec<SymmetrySidecarNode>,
    symmetry_fields_truncated: bool,
    field_collider: Option<FieldColliderSidecar>,
    codec_mosh: Option<CodecMoshSidecar>,
    last: Option<MotionSidecarLastFrame>,
}

/// Shared state for progress/cancellation between the render thread and the UI.
pub struct ExportProgress {
    /// 0..10000 representing 0.0%..100.0%
    pub progress: AtomicU32,
    /// Set to true to request cancellation.
    pub cancel: Arc<AtomicBool>,
    /// Set to true once cancellation cleanup has completed.
    pub cancelled: AtomicBool,
    /// Set to true when the job is complete (success or failure).
    pub done: AtomicBool,
    /// Error message if the job failed (empty = success).
    pub error: std::sync::Mutex<String>,
    /// Recoverable source substitutions encountered by the worker. These do
    /// not change a successful export into a failure; callers surface the
    /// messages separately from `error`.
    warnings: Mutex<Vec<String>>,
    motion_metadata: Mutex<ExportMotionMetadata>,
    /// Atomic arbitration between a cancellation request and success/failure.
    /// Public cancellation changes RUNNING -> CANCEL_REQUESTED without locks;
    /// only an owner that has completed external cleanup may publish a
    /// terminal outcome.
    outcome: AtomicU8,
    /// Serializes publication of the user-visible terminal fields.
    terminal: Mutex<()>,
    /// True only after ffmpeg has started and may have created/truncated the
    /// destination. Early cancellation must not delete a pre-existing file.
    output_started: AtomicBool,
    /// The supervisor owns an encoder process or is in its spawn window.
    encoder_active: AtomicBool,
    /// True only when no encoder process can still be alive or appear late.
    encoder_cleanup_complete: AtomicBool,
    /// Set immediately before the worker deliberately closes encoder stdin.
    /// An encoder exit without this flag is treated as an unexpected worker
    /// failure rather than a successful completion.
    encoder_finish_requested: AtomicBool,
    /// Internal worker failure/panic requests encoder shutdown without being
    /// mislabeled as a user cancellation.
    abort: AtomicBool,
    pending_worker_error: Mutex<Option<String>>,
}

impl ExportProgress {
    pub fn new() -> Self {
        Self {
            progress: AtomicU32::new(0),
            cancel: Arc::new(AtomicBool::new(false)),
            cancelled: AtomicBool::new(false),
            done: AtomicBool::new(false),
            error: std::sync::Mutex::new(String::new()),
            warnings: Mutex::new(Vec::new()),
            motion_metadata: Mutex::new(ExportMotionMetadata::default()),
            outcome: AtomicU8::new(OUTCOME_RUNNING),
            terminal: Mutex::new(()),
            output_started: AtomicBool::new(false),
            encoder_active: AtomicBool::new(false),
            encoder_cleanup_complete: AtomicBool::new(true),
            encoder_finish_requested: AtomicBool::new(false),
            abort: AtomicBool::new(false),
            pending_worker_error: Mutex::new(None),
        }
    }

    pub fn progress_f32(&self) -> f32 {
        self.progress.load(Ordering::Relaxed) as f32 / 10000.0
    }

    /// Snapshot recoverable source-substitution diagnostics for the control
    /// panel. The clone is deliberately small and avoids exposing a lock to
    /// the render loop.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    #[allow(
        dead_code,
        reason = "bounded M4 export telemetry is a frozen UI seam pending control-panel publication"
    )]
    pub fn motion_metadata(&self) -> ExportMotionMetadata {
        self.motion_metadata
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn publish_motion_metadata(&self, metadata: ExportMotionMetadata) {
        *self
            .motion_metadata
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = metadata;
    }

    fn record_warning(&self, warning: impl AsRef<str>) {
        let warning = bounded_export_warning(warning.as_ref());
        log::warn!("Export: {warning}");
        let mut warnings = self
            .warnings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if warnings.len() < MAX_EXPORT_WARNINGS {
            warnings.push(warning);
        } else if warnings
            .last()
            .is_some_and(|last| last != "Additional export source warnings were omitted.")
        {
            if let Some(last) = warnings.last_mut() {
                *last = "Additional export source warnings were omitted.".to_string();
            }
        }
    }

    fn terminal_guard(&self) -> MutexGuard<'_, ()> {
        self.terminal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Request cancellation without taking a lock or waiting for the process.
    /// The encoder supervisor observes this flag, kills/reaps its child, and
    /// publishes terminal cancellation only after deleting the partial file.
    fn request_cancel(&self) {
        if self
            .outcome
            .compare_exchange(
                OUTCOME_RUNNING,
                OUTCOME_CANCEL_REQUESTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
            || self.outcome.load(Ordering::Acquire) == OUTCOME_CANCEL_REQUESTED
        {
            self.cancel.store(true, Ordering::Release);
            log::debug!("export cancellation requested");
        }
    }
}

fn bounded_export_warning(message: &str) -> String {
    if message.chars().count() <= MAX_EXPORT_WARNING_CHARS {
        return message.to_string();
    }
    let mut bounded = message
        .chars()
        .take(MAX_EXPORT_WARNING_CHARS.saturating_sub(1))
        .collect::<String>();
    bounded.push('…');
    bounded
}

/// Handle to a running export job.
pub struct ExportJob {
    pub progress: Arc<ExportProgress>,
    thread: Option<std::thread::JoinHandle<()>>,
    output_path: String,
}

impl ExportJob {
    /// Start an export job on a background thread.
    #[cfg(test)]
    pub fn start(patch: PatchState, config: ExportConfig, library_folder: &str) -> Self {
        Self::start_inner(patch, config, library_folder, None)
    }

    /// Live exports share the renderer device so completing or cancelling an
    /// export cannot tear down a second backend device underneath the live
    /// presentation loop.
    pub fn start_with_gpu(
        patch: PatchState,
        config: ExportConfig,
        library_folder: &str,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self::start_inner(patch, config, library_folder, Some((device, queue)))
    }

    fn start_inner(
        patch: PatchState,
        config: ExportConfig,
        library_folder: &str,
        shared_gpu: Option<(wgpu::Device, wgpu::Queue)>,
    ) -> Self {
        let progress = Arc::new(ExportProgress::new());
        let prog = progress.clone();
        let lib_folder = library_folder.to_string();
        let output_path = config.output_path.clone();

        let thread = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_export(&patch, &config, &prog, &lib_folder, shared_gpu)
            }));
            let worker_error = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("unknown panic");
                    Some(format!("export worker panicked: {detail}"))
                }
            };

            finalize_export_worker(&prog, &config.output_path, worker_error);
        });

        Self {
            progress,
            thread: Some(thread),
            output_path,
        }
    }

    /// Check if the job is done.
    pub fn is_done(&self) -> bool {
        self.progress.done.load(Ordering::Acquire)
    }

    /// Request cancellation. This method is deliberately lock-free and does
    /// not claim terminal completion; the supervisor owns process cleanup.
    pub fn cancel(&self) {
        self.progress.request_cancel();
    }

    /// A terminal job may be replaced once the supervisor has proven that no
    /// encoder process or partial artifact remains. Drop gives the render
    /// worker its bounded grace period; a wedged GPU destructor cannot
    /// permanently disable the Render button.
    pub fn can_replace(&self) -> bool {
        self.is_done()
            && self
                .progress
                .encoder_cleanup_complete
                .load(Ordering::Acquire)
    }
}

impl Drop for ExportJob {
    fn drop(&mut self) {
        // The deadline covers the whole Drop operation, including the cancel
        // request. cancel() is lock-free; any encoder remains owned by its
        // dedicated supervisor even if GPU cleanup outlives this handle.
        let deadline = std::time::Instant::now() + EXPORT_DROP_JOIN_TIMEOUT;
        self.cancel();

        let Some(thread) = self.thread.take() else {
            return;
        };
        if thread.thread().id() == std::thread::current().id() {
            // Safe construction never transfers an ExportJob into its own
            // worker, but avoid a self-join panic if that invariant is ever
            // changed. Cancellation is already visible to the worker.
            log::error!("ExportJob dropped from its own worker; skipping self-join");
            return;
        }

        while !thread.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if !thread.is_finished() {
            // Rust has no safe timed join or thread termination. The render
            // worker owns only its GPU state and a supervisor JoinHandle; the
            // supervisor thread itself remains the sole live Child owner and
            // will kill/reap it before publishing a terminal result.
            log::error!(
                "export worker did not exit within {:?}; detaching isolated GPU cleanup",
                EXPORT_DROP_JOIN_TIMEOUT
            );
            return;
        }

        let worker_panicked = thread.join().is_err();
        if !self.progress.done.load(Ordering::Acquire) {
            finalize_export_worker(
                &self.progress,
                &self.output_path,
                Some(if worker_panicked {
                    "export worker terminated unexpectedly".to_string()
                } else {
                    "export worker exited without publishing terminal state".to_string()
                }),
            );
        }
    }
}

fn finalize_export_worker(
    progress: &ExportProgress,
    output_path: &str,
    worker_error: Option<String>,
) {
    // If the render worker unwound while a supervisor still owns ffmpeg, that
    // supervisor is responsible for reap/removal and terminal publication.
    // Publishing here would make `done` lie about external cleanup.
    if progress.encoder_active.load(Ordering::Acquire)
        && !progress.encoder_cleanup_complete.load(Ordering::Acquire)
    {
        if let Some(error) = worker_error {
            *progress
                .pending_worker_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
            progress.abort.store(true, Ordering::Release);
        }
        return;
    }

    // Serialize the final commit with ExportJob::cancel(). Whichever obtains
    // this lock first defines whether cancellation was still accepted while
    // the job was running.
    let _terminal = progress.terminal_guard();
    if progress.done.load(Ordering::Acquire) {
        return;
    }
    let mut decision = progress.outcome.load(Ordering::Acquire);
    if worker_error.is_some() && decision == OUTCOME_RUNNING {
        decision = match progress.outcome.compare_exchange(
            OUTCOME_RUNNING,
            OUTCOME_FAILED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => OUTCOME_FAILED,
            Err(actual) => actual,
        };
    }
    let error = if decision == OUTCOME_CANCEL_REQUESTED {
        let cleanup_error = remove_started_output(progress, Some(output_path));
        progress.cancelled.store(true, Ordering::Relaxed);
        progress.outcome.store(OUTCOME_CANCELLED, Ordering::Release);
        Some(match cleanup_error {
            Some(error) => format!("export cancelled; {error}"),
            None => "export cancelled".to_string(),
        })
    } else if decision == OUTCOME_FAILED {
        let worker_error = worker_error
            .unwrap_or_else(|| "export worker failed without diagnostic detail".to_string());
        // ffmpeg opens/truncates the destination at startup. Any later error
        // or caught panic must therefore remove that incomplete artifact just
        // as cancellation does.
        let cleanup_error = remove_started_output(progress, Some(output_path));
        Some(match cleanup_error {
            Some(error) => format!("{worker_error}; {error}"),
            None => worker_error,
        })
    } else if decision == OUTCOME_RUNNING {
        if progress
            .outcome
            .compare_exchange(
                OUTCOME_RUNNING,
                OUTCOME_SUCCEEDED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // Cancellation won after the first decision load but before the
            // success commit. Cleanup is already complete on this path.
            let cleanup_error = remove_started_output(progress, Some(output_path));
            progress.cancelled.store(true, Ordering::Relaxed);
            progress.outcome.store(OUTCOME_CANCELLED, Ordering::Release);
            let error = match cleanup_error {
                Some(error) => format!("export cancelled; {error}"),
                None => "export cancelled".to_string(),
            };
            *progress
                .error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = error;
            progress.done.store(true, Ordering::Release);
            return;
        }
        None
    } else {
        return;
    };
    if let Some(error) = error {
        *progress
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = error;
    }
    // Release pairs with is_done's Acquire, publishing cancelled/error before
    // the UI observes the terminal state.
    progress.done.store(true, Ordering::Release);
}

fn remove_started_output(progress: &ExportProgress, output_path: Option<&str>) -> Option<String> {
    if !progress.output_started.load(Ordering::Acquire) {
        return None;
    }
    let path = output_path?;
    let mut errors = Vec::new();
    if let Err(error) = remove_partial_output(path) {
        errors.push(error);
    }
    let sidecar = motion_sidecar_path(path);
    if let Err(error) = remove_partial_path(&sidecar, "motion sidecar") {
        errors.push(error);
    }
    // The gesture sidecar is cleanup-coupled to the video exactly as the motion
    // report is: a cancelled or failed render must never leave a recording
    // receipt standing next to a deleted output.
    let gesture = gesture_sidecar_path(path);
    if let Err(error) = remove_partial_path(&gesture, "gesture sidecar") {
        errors.push(error);
    }
    let performance = performance_sidecar_path(path);
    if let Err(error) = remove_partial_path(&performance, "performance sidecar") {
        errors.push(error);
    }
    (!errors.is_empty()).then(|| errors.join("; "))
}

/// Publish cancellation after process/file cleanup but before the large wgpu
/// object graph local to `run_export` is dropped. This keeps the UI terminal
/// even if a graphics backend takes unusually long to destroy resources.
fn publish_cancelled_terminal(progress: &ExportProgress, output_path: Option<&str>) {
    let _terminal = progress.terminal_guard();
    if progress.done.load(Ordering::Acquire) {
        return;
    }
    if progress.outcome.load(Ordering::Acquire) != OUTCOME_CANCEL_REQUESTED {
        return;
    }
    let cleanup_error = remove_started_output(progress, output_path);
    progress.cancelled.store(true, Ordering::Relaxed);
    progress.outcome.store(OUTCOME_CANCELLED, Ordering::Release);
    *progress
        .error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = match cleanup_error {
        Some(error) => format!("export cancelled; {error}"),
        None => "export cancelled".to_string(),
    };
    progress.done.store(true, Ordering::Release);
}

fn remove_partial_output(path: &str) -> Result<(), String> {
    remove_partial_path(std::path::Path::new(path), "partial output")
}

fn remove_partial_path(path: &std::path::Path, label: &str) -> Result<(), String> {
    for attempt in 0..5 {
        match std::fs::remove_file(path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) if attempt == 4 => {
                return Err(format!(
                    "failed to remove {label} '{}': {error}",
                    path.display()
                ));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    unreachable!()
}

fn motion_sidecar_path(output_path: &str) -> std::path::PathBuf {
    let mut path = std::ffi::OsString::from(output_path);
    path.push(".motion.json");
    std::path::PathBuf::from(path)
}

/// The gesture sidecar sits beside the rendered video exactly as the motion
/// report does. Its name is derived from the output path; nothing about that
/// path enters the file.
fn gesture_sidecar_path(output_path: &str) -> std::path::PathBuf {
    let mut path = std::ffi::OsString::from(output_path);
    path.push(".gesture.json");
    std::path::PathBuf::from(path)
}

/// Publish the replayed gesture recording beside its render.
///
/// The commit is the staged no-replace transaction `src/procedural.rs` uses for
/// generated pieces: write a private temporary sibling with `create_new`, sync
/// it, then rename it into place *without* replacement, so two jobs racing for
/// the same destination fail closed instead of one silently overwriting the
/// other's receipt. The payload is the frozen six-field
/// `GestureTrackDocument`; `to_json_bytes` revalidates the whole stream and
/// re-derives its canonical checksum before a byte is staged. No operational
/// path, directory, timestamp, host name, or other filesystem metadata enters
/// the document.
fn write_gesture_sidecar_noreplace(
    output_path: &str,
    document: &crate::gesture::GestureTrackDocument,
) -> Result<(), String> {
    let bytes = document
        .to_json_bytes()
        .map_err(|error| format!("serialize export gesture sidecar: {error}"))?;
    let sidecar_path = gesture_sidecar_path(output_path);
    if sidecar_path.exists() {
        return Err(format!(
            "refusing to overwrite export gesture sidecar '{}'",
            sidecar_path.display()
        ));
    }
    let temp_path = sidecar_temp_path(&sidecar_path);
    if let Err(error) = crate::procedural::write_new_file(&temp_path, &bytes) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(format!("stage export gesture sidecar: {error}"));
    }
    if let Err(error) = crate::procedural::rename_noreplace(&temp_path, &sidecar_path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(format!(
            "commit export gesture sidecar '{}': {error}",
            sidecar_path.display()
        ));
    }
    crate::procedural::sync_parent(&sidecar_path).map_err(|error| {
        format!(
            "export gesture sidecar committed at '{}', but synchronizing its parent directory failed: {error}",
            sidecar_path.display()
        )
    })
}

/// The B9 performance sidecar sits beside the rendered video exactly as the
/// gesture recording does; its name is derived from the output path and
/// nothing about that path enters the file.
fn performance_sidecar_path(output_path: &str) -> std::path::PathBuf {
    let mut path = std::ffi::OsString::from(output_path);
    path.push(".performance.json");
    std::path::PathBuf::from(path)
}

/// Publish the replayed performance take beside its render, through the same
/// staged no-replace commit the gesture sidecar uses. `to_json_bytes`
/// revalidates the whole stream and re-derives its canonical checksum before
/// a byte is staged.
fn write_performance_sidecar_noreplace(
    output_path: &str,
    document: &crate::performance_track::PerformanceTakeDocument,
) -> Result<(), String> {
    let bytes = document
        .to_json_bytes()
        .map_err(|error| format!("serialize export performance sidecar: {error}"))?;
    let sidecar_path = performance_sidecar_path(output_path);
    if sidecar_path.exists() {
        return Err(format!(
            "refusing to overwrite export performance sidecar '{}'",
            sidecar_path.display()
        ));
    }
    let temp_path = sidecar_temp_path(&sidecar_path);
    if let Err(error) = crate::procedural::write_new_file(&temp_path, &bytes) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(format!("stage export performance sidecar: {error}"));
    }
    if let Err(error) = crate::procedural::rename_noreplace(&temp_path, &sidecar_path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(format!(
            "commit export performance sidecar '{}': {error}",
            sidecar_path.display()
        ));
    }
    crate::procedural::sync_parent(&sidecar_path).map_err(|error| {
        format!(
            "export performance sidecar committed at '{}', but synchronizing its parent directory failed: {error}",
            sidecar_path.display()
        )
    })
}

/// Private staging sibling for one sidecar publication. Shared by the motion
/// report and the gesture recording so both stage the same way.
fn sidecar_temp_path(sidecar: &std::path::Path) -> std::path::PathBuf {
    let parent = sidecar
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut file_name = sidecar
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("sidecar.json"))
        .to_os_string();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    file_name.push(format!(".tmp-{}-{nonce}", std::process::id()));
    parent.join(file_name)
}

#[cfg(windows)]
fn atomic_replace_motion_sidecar(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // Same-directory MoveFileEx with replacement is the Windows atomic
    // publication primitive. WRITE_THROUGH keeps the rename from being
    // reported before the filesystem receives it.
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn atomic_replace_motion_sidecar(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

fn write_motion_sidecar_atomic(
    output_path: &str,
    sidecar: &ExportMotionSidecar,
) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(sidecar)
        .map_err(|error| format!("serialize export motion sidecar: {error}"))?;
    if bytes.len() > MAX_EXPORT_MOTION_SIDECAR_BYTES {
        return Err(format!(
            "export motion sidecar is {} bytes; limit is {}",
            bytes.len(),
            MAX_EXPORT_MOTION_SIDECAR_BYTES
        ));
    }
    let sidecar_path = motion_sidecar_path(output_path);
    let temp_path = sidecar_temp_path(&sidecar_path);
    let write_result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|error| {
                format!(
                    "create temporary export motion sidecar '{}': {error}",
                    temp_path.display()
                )
            })?;
        file.write_all(&bytes).map_err(|error| {
            format!(
                "write temporary export motion sidecar '{}': {error}",
                temp_path.display()
            )
        })?;
        file.sync_all().map_err(|error| {
            format!(
                "sync temporary export motion sidecar '{}': {error}",
                temp_path.display()
            )
        })?;
        drop(file);
        atomic_replace_motion_sidecar(&temp_path, &sidecar_path).map_err(|error| {
            format!(
                "commit export motion sidecar '{}': {error}",
                sidecar_path.display()
            )
        })?;
        if let Some(parent) = sidecar_path.parent() {
            if let Ok(directory) = std::fs::File::open(parent) {
                directory.sync_all().map_err(|error| {
                    format!(
                        "sync export motion sidecar directory '{}': {error}",
                        parent.display()
                    )
                })?;
            }
        }
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    write_result
}

fn check_cancelled(progress: &ExportProgress) -> Result<(), String> {
    if progress.cancel.load(Ordering::Acquire) {
        // All current callers are before ffmpeg startup, so no partial output
        // exists and terminal state can be published immediately.
        publish_cancelled_terminal(progress, None);
        Err("export cancelled".to_string())
    } else {
        Ok(())
    }
}

/// Wait without surrendering cancellation control to a blocking Child::wait.
/// Once cancellation is observed, `kill` is issued and the process is still
/// reaped before this function returns.
fn wait_for_ffmpeg(
    child: &mut Child,
    cancel: &AtomicBool,
    abort: &AtomicBool,
) -> std::io::Result<ExitStatus> {
    let mut kill_sent = false;
    loop {
        if (cancel.load(Ordering::Acquire) || abort.load(Ordering::Acquire)) && !kill_sent {
            match child.kill() {
                Ok(()) => kill_sent = true,
                Err(error) => log::warn!("encoder kill request failed; retrying: {error}"),
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(error) => {
                // Keep ownership and keep polling. Returning would drop an
                // unproven Child and make terminal cleanup claims false.
                log::warn!("encoder status poll failed; retaining child: {error}");
                kill_sent = false;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn force_reap_ffmpeg(child: &mut Child) -> ExitStatus {
    static TRUE_FLAG: AtomicBool = AtomicBool::new(true);
    static FALSE_FLAG: AtomicBool = AtomicBool::new(false);
    // This helper deliberately does not return until try_wait proves exit.
    // It runs only on the supervisor thread, so a pathological OS failure can
    // neither block the UI nor detach the still-owned Child.
    wait_for_ffmpeg(child, &TRUE_FLAG, &FALSE_FLAG)
        .expect("wait_for_ffmpeg retains ownership instead of returning errors")
}

struct EncoderCompletion {
    status: Result<ExitStatus, String>,
    stderr: Result<Vec<u8>, String>,
}

struct EncoderSession {
    stdin: ChildStdin,
    completion: std::sync::mpsc::Receiver<EncoderCompletion>,
    supervisor: std::thread::JoinHandle<()>,
}

/// Start the encoder under a dedicated supervisor. The supervisor is the only
/// owner of `Child`; the render worker receives only stdin and a completion
/// channel. Consequently no cancellation, worker unwind, or bounded Drop can
/// detach an unreaped encoder process.
fn start_encoder_supervisor(
    program: String,
    args: Vec<String>,
    progress: Arc<ExportProgress>,
    output_path: String,
) -> Result<EncoderSession, String> {
    progress.encoder_active.store(true, Ordering::Release);
    progress
        .encoder_cleanup_complete
        .store(false, Ordering::Release);
    progress
        .encoder_finish_requested
        .store(false, Ordering::Release);

    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let (completion_tx, completion_rx) = std::sync::mpsc::sync_channel(1);
    let supervisor_progress = progress.clone();
    let supervisor = match std::thread::Builder::new()
        .name("export-encoder-supervisor".into())
        .spawn(move || {
            let finish_without_child = |error: String| {
                supervisor_progress
                    .encoder_active
                    .store(false, Ordering::Release);
                supervisor_progress
                    .encoder_cleanup_complete
                    .store(true, Ordering::Release);
                if supervisor_progress.cancel.load(Ordering::Acquire) {
                    publish_cancelled_terminal(&supervisor_progress, None);
                } else {
                    finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                }
                let _ = ready_tx.send(Err(error));
            };

            if supervisor_progress.cancel.load(Ordering::Acquire) {
                finish_without_child("export cancelled before encoder startup".to_string());
                return;
            }

            let mut command = Command::new(&program);
            command
                .args(&args)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            let mut child = match command.spawn() {
                Ok(child) => child,
                Err(error) => {
                    finish_without_child(format!("Failed to spawn {program}: {error}"));
                    return;
                }
            };
            log::debug!("export encoder supervisor spawned process {}", child.id());
            supervisor_progress
                .output_started
                .store(true, Ordering::Release);
            // ffmpeg runs with `-y`, so spawning it is the moment this job takes
            // ownership of the destination name. Retiring a previous run's
            // gesture receipt here — and only here — is what lets the strictly
            // no-replace commit below stay strictly no-replace: a re-export
            // replaces its own stale receipt, while a second job racing for the
            // same destination still fails closed at the rename.
            let _ = remove_partial_path(&gesture_sidecar_path(&output_path), "gesture sidecar");
            let _ = remove_partial_path(
                &performance_sidecar_path(&output_path),
                "performance sidecar",
            );

            let Some(stdin) = child.stdin.take() else {
                let _ = force_reap_ffmpeg(&mut child);
                supervisor_progress
                    .encoder_active
                    .store(false, Ordering::Release);
                supervisor_progress
                    .encoder_cleanup_complete
                    .store(true, Ordering::Release);
                let error = "ffmpeg stdin pipe unavailable".to_string();
                finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                let _ = ready_tx.send(Err(error));
                return;
            };
            let Some(mut stderr) = child.stderr.take() else {
                drop(stdin);
                let _ = force_reap_ffmpeg(&mut child);
                supervisor_progress
                    .encoder_active
                    .store(false, Ordering::Release);
                supervisor_progress
                    .encoder_cleanup_complete
                    .store(true, Ordering::Release);
                let error = "ffmpeg stderr pipe unavailable".to_string();
                finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                let _ = ready_tx.send(Err(error));
                return;
            };
            let stderr_thread = match std::thread::Builder::new()
                .name("export-ffmpeg-stderr".into())
                .spawn(move || {
                    let mut bytes = Vec::new();
                    let result = stderr.read_to_end(&mut bytes).map(|_| bytes);
                    result.map_err(|error| format!("ffmpeg stderr read: {error}"))
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    drop(stdin);
                    let _ = force_reap_ffmpeg(&mut child);
                    supervisor_progress
                        .encoder_active
                        .store(false, Ordering::Release);
                    supervisor_progress
                        .encoder_cleanup_complete
                        .store(true, Ordering::Release);
                    let error = format!("failed to start ffmpeg stderr reader: {error}");
                    finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };

            if ready_tx.send(Ok(stdin)).is_err() {
                supervisor_progress.request_cancel();
            }

            log::debug!("export encoder supervisor waiting for process cleanup");
            let status = wait_for_ffmpeg(
                &mut child,
                &supervisor_progress.cancel,
                &supervisor_progress.abort,
            )
            .map_err(|error| format!("ffmpeg wait: {error}"));
            // `wait_for_ffmpeg` returns only after the child is reaped. The
            // stderr pipe is then closed, so this join cannot await a writer.
            let stderr = stderr_thread
                .join()
                .map_err(|_| "ffmpeg stderr reader panicked".to_string())
                .and_then(|result| result);
            log::debug!("export encoder supervisor reaped process");

            supervisor_progress
                .encoder_active
                .store(false, Ordering::Release);
            supervisor_progress
                .encoder_cleanup_complete
                .store(true, Ordering::Release);

            if supervisor_progress.cancel.load(Ordering::Acquire) {
                publish_cancelled_terminal(&supervisor_progress, Some(&output_path));
                log::debug!("export encoder supervisor published cancelled terminal state");
            } else if supervisor_progress.abort.load(Ordering::Acquire) {
                let error = supervisor_progress
                    .pending_worker_error
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .unwrap_or_else(|| "export worker aborted unexpectedly".to_string());
                finalize_export_worker(&supervisor_progress, &output_path, Some(error));
            } else if !supervisor_progress
                .encoder_finish_requested
                .load(Ordering::Acquire)
            {
                finalize_export_worker(
                    &supervisor_progress,
                    &output_path,
                    Some("export worker ended before closing the encoder cleanly".to_string()),
                );
            }

            let _ = completion_tx.send(EncoderCompletion { status, stderr });
        }) {
        Ok(thread) => thread,
        Err(error) => {
            progress.encoder_active.store(false, Ordering::Release);
            progress
                .encoder_cleanup_complete
                .store(true, Ordering::Release);
            return Err(format!("failed to start encoder supervisor: {error}"));
        }
    };

    match ready_rx.recv() {
        Ok(Ok(stdin)) => Ok(EncoderSession {
            stdin,
            completion: completion_rx,
            supervisor,
        }),
        Ok(Err(error)) => {
            let _ = supervisor.join();
            Err(error)
        }
        Err(_) => {
            let _ = supervisor.join();
            progress.encoder_active.store(false, Ordering::Release);
            progress
                .encoder_cleanup_complete
                .store(true, Ordering::Release);
            Err("encoder supervisor exited before startup completed".to_string())
        }
    }
}

/// Uniforms for the composite shader (must match renderer/state.rs).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    _pad: [f32; 2],
}

/// Export has two exact dry-overlay seams. Without Codec Mosh the authored
/// prefix can be reapplied on the GPU before the established opaque resolve.
/// With Codec Mosh, the first resolve/readback must remain wet-only; the dry
/// prefix is rendered once for the pre-mosh analysis/program tap and again on
/// top of the returned moshed wet audience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportTemporalBypassMode {
    Inactive,
    BeforeOpaque,
    AroundCodecMosh,
}

fn export_temporal_bypass_mode(
    path: TemporalMasterPath,
    codec_mosh_active: bool,
) -> ExportTemporalBypassMode {
    match (path, codec_mosh_active) {
        (TemporalMasterPath::IsolatedDryOverlay, false) => ExportTemporalBypassMode::BeforeOpaque,
        (TemporalMasterPath::IsolatedDryOverlay, true) => ExportTemporalBypassMode::AroundCodecMosh,
        (TemporalMasterPath::Inherited | TemporalMasterPath::LinkedDry, _) => {
            ExportTemporalBypassMode::Inactive
        }
    }
}

/// Back-to-front indices admitted to the shared wet accumulation. The extra
/// predicate is dormant unless at least one dry layer contributes, preserving
/// the established layer list and command order for every old frame.
fn export_temporal_wet_stack_indices(evaluated: &EvaluatedFramePlan) -> Vec<usize> {
    let isolated_temporal_overlay = temporal_bypass_overlay_active(evaluated);
    visible_stack_indices(
        evaluated.layers().iter().map(|layer| {
            layer.visible && (!isolated_temporal_overlay || !layer.bypass_temporal_fx)
        }),
    )
}

/// Whether the isolated route has any visible, non-zero wet contribution for
/// the global Master pass to process. This mirrors live's all-dry fast path and
/// avoids a full-frame shader pass over a defined transparent base.
fn export_temporal_wet_layer_contributes(evaluated: &EvaluatedFramePlan) -> bool {
    evaluated.layers().iter().any(|layer| {
        let opacity_contributes = layer.opacity > 0.0 || !layer.opacity.is_finite();
        layer.visible && !layer.bypass_temporal_fx && opacity_contributes
    })
}

/// Bottom-to-top dry sources, paired with their immutable evaluated slots.
/// Keeping this plan CPU-visible lets export validate every deferred texture
/// before it opens a stateful Temporal transaction.
fn export_temporal_bypass_overlay_sources(
    evaluated: &EvaluatedFramePlan,
) -> Vec<(usize, SourceTap)> {
    temporal_bypass_overlay_indices(evaluated)
        .into_iter()
        .map(|layer_index| (layer_index, evaluated.layers()[layer_index].source))
        .collect()
}

/// Legacy ProgramHistory is captured before Temporal. During isolated bypass
/// that seam contains only the wet partition, so publishing it would expose a
/// composition that is never visible. Hold the history invalid until the
/// partition returns to the inherited path.
fn export_may_publish_legacy_program_history(evaluated: &EvaluatedFramePlan) -> bool {
    !temporal_bypass_overlay_active(evaluated)
}

/// Observe the authored dry/wet membership without allocating after the first
/// frame. A first observation seeds an empty Temporal machine; every later
/// difference is a hard partition edge, even when Program Freeze supplies a
/// zero delta for that frame.
fn observe_export_temporal_bypass_partition(
    previous: &mut Vec<bool>,
    initialized: &mut bool,
    evaluated: &EvaluatedFramePlan,
) -> bool {
    let changed = *initialized
        && (previous.len() != evaluated.layers().len()
            || previous
                .iter()
                .zip(evaluated.layers())
                .any(|(prior, layer)| *prior != layer.bypass_temporal_fx));
    previous.clear();
    previous.extend(
        evaluated
            .layers()
            .iter()
            .map(|layer| layer.bypass_temporal_fx),
    );
    *initialized = true;
    changed
}

/// Mirror live's interval-generation law for the synchronous export owner.
/// Codec/recycle history belongs to one continuously armed interval; a dry
/// edge and a later re-arm must never resume the old MPEG-4 stream.
fn update_export_mosh_interval<T>(active: bool, was_active: &mut bool, retained: &mut Option<T>) {
    if *was_active != active {
        *was_active = active;
        *retained = None;
    } else if !active {
        *retained = None;
    }
}

/// Allocate the one isolated-Mosh programme candidate at the frame preflight
/// boundary. This is deliberately synchronous and checked: once Temporal,
/// Melt, Sync, or Display has encoded, no cold full-frame allocation remains.
fn ensure_export_temporal_bypass_program_candidate(
    device: &wgpu::Device,
    retained: &mut Option<wgpu::Texture>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if retained.is_some() {
        return Ok(());
    }
    let candidate = scoped_export_gpu_operation(
        device,
        format!("export Temporal-bypass programme candidate at {width}x{height}"),
        || {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Export Temporal Bypass Program Candidate"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        },
    )?;
    *retained = Some(candidate);
    Ok(())
}

/// Upload one small export uniform without mapping the new buffer.
///
/// `DeviceExt::create_buffer_init` maps at creation and immediately accesses
/// the mapped range. A queue upload preserves the exact bytes while allowing
/// device-loss validation to remain recoverable instead of panicking in the
/// mapped-range accessor.
fn create_uploaded_uniform<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    value: &T,
) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(value);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// Error scopes are thread-local even when export reuses the live renderer's
/// device. Keep every export-frame operation on this worker inside a typed
/// scope and latch failures so an early `break` cannot silently discard them.
struct ExportGpuErrorCapture {
    device: wgpu::Device,
    validation: Option<wgpu::ErrorScopeGuard>,
    internal: Option<wgpu::ErrorScopeGuard>,
    out_of_memory: Option<wgpu::ErrorScopeGuard>,
    context: String,
    sink: Arc<Mutex<Vec<String>>>,
}

impl ExportGpuErrorCapture {
    fn new(
        device: &wgpu::Device,
        context: impl Into<String>,
        sink: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        Self {
            device: device.clone(),
            validation: Some(validation),
            internal: Some(internal),
            out_of_memory: Some(out_of_memory),
            context: context.into(),
            sink,
        }
    }

    fn collect(&mut self) {
        if self.validation.is_none() {
            return;
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
        let errors = [
            self.out_of_memory
                .take()
                .and_then(|scope| pollster::block_on(scope.pop())),
            self.internal
                .take()
                .and_then(|scope| pollster::block_on(scope.pop())),
            self.validation
                .take()
                .and_then(|scope| pollster::block_on(scope.pop())),
        ]
        .into_iter()
        .flatten()
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
        if !errors.is_empty() {
            self.sink
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(format!("{}: {}", self.context, errors.join("; ")));
        }
    }
}

impl Drop for ExportGpuErrorCapture {
    fn drop(&mut self) {
        self.collect();
    }
}

fn take_export_gpu_errors(sink: &Arc<Mutex<Vec<String>>>) -> Option<String> {
    let mut errors = sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    (!errors.is_empty()).then(|| std::mem::take(&mut *errors).join("; "))
}

fn scoped_export_gpu_operation<T>(
    device: &wgpu::Device,
    context: impl Into<String>,
    operation: impl FnOnce() -> T,
) -> Result<T, String> {
    let sink = Arc::new(Mutex::new(Vec::new()));
    let scope = ExportGpuErrorCapture::new(device, context, sink.clone());
    let value = operation();
    drop(scope);
    take_export_gpu_errors(&sink).map_or(Ok(value), Err)
}

/// Internal layer state for offline rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportMotionSourceKind {
    Video,
    Still,
    SpoutBlack,
    UnavailableBlack,
    /// B7 pattern synth: reconstructed exactly from the patch, no bytes.
    PatternSynth,
    /// B7 text page: rastered exactly from the patch, no bytes.
    TextPage,
}

#[derive(Debug, Clone)]
struct ExportMotionSourceRecord {
    kind: ExportMotionSourceKind,
    logical_name: String,
    persisted_reference: String,
    fingerprint: Option<crate::media_source::ContentIdentity>,
}

impl ExportMotionSourceRecord {
    fn unavailable(layer: &crate::patch::LayerConfig) -> Self {
        let (logical_name, persisted_reference) = active_export_clip_source(layer);
        Self {
            kind: if crate::layers::spout_sender_from_source_path(persisted_reference).is_some() {
                ExportMotionSourceKind::SpoutBlack
            } else {
                ExportMotionSourceKind::UnavailableBlack
            },
            logical_name: logical_name.to_owned(),
            persisted_reference: persisted_reference.to_owned(),
            fingerprint: None,
        }
    }
}

struct ExportLayer {
    /// Index in the saved patch. Failed-to-open layers must not shift
    /// layer-specific modulation routes onto a different video.
    source_index: usize,
    motion_source: ExportMotionSourceRecord,
    /// `None` is an explicit deterministic-black placeholder for a live,
    /// missing, or undecodable source that cannot be sampled offline.
    decoder: Option<VideoDecoder>,
    /// Sparse decoder metadata committed only after the matching RGBA upload
    /// succeeds. Stills, offline Spout substitutions, and reverse-cache hits
    /// remain explicitly absent rather than inventing a lattice here.
    codec_motion: Option<CodecMotionProduct>,
    /// Exact source image currently resident in `texture`. Skipped-frame
    /// composition starts strictly after this accepted predecessor; cuts
    /// clear it before decode so no transition crosses a generation.
    codec_motion_predecessor: Option<CodecFrameIdentity>,
    /// Retain the still's Expert reservation for the lifetime of its uploaded
    /// texture. The decoded bytes are also retained within the conservative
    /// six-RGBA-buffer planning estimate.
    _still_source: Option<DecodedStillImage>,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    effects: EffectUniforms,
    transform: SpatialTransform,
    opacity: f32,
    mosh_send: f32,
    blend_mode: BlendMode,
    bypass_master_fx: bool,
    bypass_temporal_fx: bool,
    matte: LayerMatte,
    reroll_on_loop: bool,
    consumed_loop_generation: u64,
    /// Pure source-time state shared with live playback. Offline rendering
    /// owns no wall-clock pacer: every output frame supplies one exact program
    /// tick and decodes the returned absolute source selection.
    transport: ExportClipTransport,
    speed: f32,
    visible: bool,
    paused: bool,
    fps: f32,
    width: u32,
    height: u32,
    /// B7 pattern-synth authored values reconstructed from the patch;
    /// `Some` iff this layer is a pattern layer. The per-frame GPU pass is
    /// its only pixel producer offline exactly as live.
    pattern: Option<crate::pattern_synth::PatternSynthParams>,
}

fn motion_field_source_key(value: crate::motion::MotionFieldSource) -> &'static str {
    match value {
        crate::motion::MotionFieldSource::Auto => "auto",
        crate::motion::MotionFieldSource::CodecVectors => "codec_vectors",
        crate::motion::MotionFieldSource::Lattice => "lattice",
        crate::motion::MotionFieldSource::Procedural(kind) => kind.source_key(),
    }
}

fn motion_lattice_quality_key(value: crate::motion::MotionLatticeQuality) -> &'static str {
    match value {
        crate::motion::MotionLatticeQuality::Draft => "draft",
        crate::motion::MotionLatticeQuality::Live => "live",
        crate::motion::MotionLatticeQuality::High => "high",
    }
}

fn motion_carrier_key(value: crate::motion::MotionCarrier) -> &'static str {
    match value {
        crate::motion::MotionCarrier::Transparent => "transparent",
        crate::motion::MotionCarrier::Black => "black",
        crate::motion::MotionCarrier::FirstSourceFrame => "first_source_frame",
    }
}

fn shutter_quality_key(value: crate::motion::CurvedShutterQuality) -> &'static str {
    match value {
        crate::motion::CurvedShutterQuality::Sharp => "sharp",
        crate::motion::CurvedShutterQuality::Draft => "draft",
        crate::motion::CurvedShutterQuality::Live => "live",
        crate::motion::CurvedShutterQuality::High => "high",
    }
}

fn motion_origin_key(value: crate::motion::MotionFieldOrigin) -> &'static str {
    match value {
        crate::motion::MotionFieldOrigin::None => "none",
        crate::motion::MotionFieldOrigin::CodecVectors => "codec_vectors",
        crate::motion::MotionFieldOrigin::Lattice => "lattice",
        crate::motion::MotionFieldOrigin::LatticeFallback => "lattice_fallback",
        crate::motion::MotionFieldOrigin::Procedural(kind) => kind.source_key(),
    }
}

pub(crate) fn sha256_bytes_hex(value: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in value {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn motion_diagnostic_key(value: crate::motion::MotionSourceDiagnostic) -> &'static str {
    match value {
        crate::motion::MotionSourceDiagnostic::None => "none",
        crate::motion::MotionSourceDiagnostic::CodecUnavailable => "codec_unavailable",
        crate::motion::MotionSourceDiagnostic::CodecUnavailableFallback => {
            "codec_unavailable_fallback"
        }
    }
}

fn codec_provenance_key(value: crate::video::CodecMotionProvenance) -> &'static str {
    match value {
        crate::video::CodecMotionProvenance::FfmpegExportMvs => "ffmpeg_export_mvs",
    }
}

fn export_motion_source_kind_key(value: ExportMotionSourceKind) -> &'static str {
    match value {
        ExportMotionSourceKind::Video => "video",
        ExportMotionSourceKind::Still => "still",
        ExportMotionSourceKind::SpoutBlack => "spout_deterministic_black",
        ExportMotionSourceKind::UnavailableBlack => "unavailable_deterministic_black",
        ExportMotionSourceKind::PatternSynth => "pattern_synth",
        ExportMotionSourceKind::TextPage => "text_page",
    }
}

fn sidecar_scope_identity(value: ExportMotionScopeIdentity) -> MotionSidecarScopeIdentity {
    match value {
        ExportMotionScopeIdentity::Master => MotionSidecarScopeIdentity::Master,
        ExportMotionScopeIdentity::Layer {
            saved_position,
            stable_id,
            source_tap_id,
        } => MotionSidecarScopeIdentity::Layer {
            saved_position,
            stable_id,
            source_tap_id,
        },
    }
}

fn donor_identity(params: crate::motion::MotionParams) -> (Option<u32>, Option<u64>) {
    match params.transplant.donor {
        crate::motion::MotionDonor::None => (None, None),
        crate::motion::MotionDonor::Selected {
            layer_id,
            saved_position,
        } => (Some(saved_position.get()), Some(layer_id.get())),
        crate::motion::MotionDonor::Missing { saved_position } => {
            (Some(saved_position.get()), None)
        }
    }
}

fn sidecar_authored_scope(
    scope: MotionSidecarScopeIdentity,
    params: crate::motion::MotionParams,
) -> MotionSidecarAuthoredScope {
    let params = params.sanitized();
    let (donor_saved_position, donor_stable_id) = donor_identity(params);
    MotionSidecarAuthoredScope {
        scope,
        algorithm_version: params.algorithm_version,
        requested_source: motion_field_source_key(params.field_source).to_owned(),
        lattice_quality: motion_lattice_quality_key(params.lattice_quality).to_owned(),
        carrier: motion_carrier_key(params.transplant.carrier).to_owned(),
        donor_saved_position,
        donor_stable_id,
        shutter_angle_degrees: params.shutter.angle_degrees,
        shutter_quality: shutter_quality_key(params.shutter.quality).to_owned(),
        shutter_sample_count: params.shutter.quality.sample_count(),
    }
}

/// Operator-facing name of one Residual route slot. The slot is named, never
/// implied by array position, so a provenance reader can never attribute one
/// input's tombstone to the other.
fn residual_slot_name(slot: u8) -> &'static str {
    match slot {
        crate::visual_rack::RESIDUAL_DETAIL_SLOT => "detail",
        _ => "structure",
    }
}

/// One resolved route slot, reduced to stable identity only. Both halves of
/// the missing pair keep their saved provenance so a tombstone is legible
/// without ever naming a host path.
fn residual_sidecar_route(
    slot: u8,
    tap: crate::visual_rack::ResolvedImageTap,
) -> ResidualSidecarRoute {
    use crate::visual_rack::ResolvedImageSource;

    let mut record = ResidualSidecarRoute {
        slot,
        slot_name: residual_slot_name(slot),
        source: "one_below",
        timing: tap.timing,
        resolved: true,
        saved_position: None,
        stable_id: None,
        group_id: None,
        stage: None,
    };
    match tap.source {
        ResolvedImageSource::SelectedLayer {
            layer_id,
            saved_position,
            stage,
        } => {
            record.source = "selected_layer";
            record.saved_position = Some(saved_position.get());
            record.stable_id = Some(layer_id.get());
            record.stage = Some(stage);
        }
        ResolvedImageSource::MissingSelectedLayer {
            saved_position,
            stage,
        } => {
            record.source = "missing_selected_layer";
            record.resolved = false;
            record.saved_position = Some(saved_position.get());
            record.stage = Some(stage);
        }
        ResolvedImageSource::OneBelow => record.source = "one_below",
        ResolvedImageSource::AllBelow => record.source = "all_below",
        ResolvedImageSource::GroupOutput(group_id) => {
            record.source = "group_output";
            record.group_id = Some(group_id.get());
        }
        ResolvedImageSource::MissingGroupOutput(group_id) => {
            record.source = "missing_group_output";
            record.resolved = false;
            record.group_id = Some(group_id.get());
        }
        ResolvedImageSource::CleanProgram => record.source = "clean_program",
        // S3b's etched canvas is a positionless master-scope singleton, so it
        // resolves with no saved position, stable id, group, or layer stage.
        ResolvedImageSource::GestureCanvas => record.source = "gesture_canvas",
        // B16's programme tap is the same positionless master-scope
        // singleton shape.
        ResolvedImageSource::ProgramTap => record.source = "program_tap",
    }
    record
}

/// Every Residual Counterpoint node in one scope's admitted rack, in authored
/// node then slot order.
fn residual_sidecar_scope_nodes(
    scope: MotionSidecarScopeIdentity,
    rack: &RuntimeVisualRack,
    records: &mut Vec<ResidualSidecarNode>,
    truncated: &mut bool,
) {
    use crate::visual_rack::RuntimeVisualNodeKind;

    for node in rack.iter() {
        let RuntimeVisualNodeKind::Residual(params) = node.kind else {
            continue;
        };
        if records.len() == MAX_EXPORT_RESIDUAL_SIDECAR_NODES {
            *truncated = true;
            return;
        }
        let params = params.sanitized();
        records.push(ResidualSidecarNode {
            scope: scope.clone(),
            node_id: node.stable_id.get(),
            enabled: node.enabled,
            wet: node.wet,
            algorithm_version: params.algorithm_version,
            block: params.block,
            block_edge: params.block.edge(),
            quantization: params.quantization,
            quantization_levels: params.quantization.levels(),
            authored_mix: params.mix,
            authored_detail_gain: params.detail_gain,
            seed: params.seed,
            exact_bypass: !node.enabled || node.wet <= 0.0 || params.is_exact_bypass(),
            routes: params
                .routes()
                .into_iter()
                .enumerate()
                .map(|(slot, tap)| {
                    residual_sidecar_route(u8::try_from(slot).unwrap_or(u8::MAX), tap)
                })
                .collect(),
        });
    }
}

/// Per-slot route provenance for every Residual node this job admitted, master
/// scope first and then each layer scope in saved order. The walk uses the
/// resolved creative graph rather than the opened-source vector, so a failed
/// media open cannot quietly drop a scope's authored recombination law.
fn residual_sidecar_nodes(graph: &ExportCreativeGraph) -> (Vec<ResidualSidecarNode>, bool) {
    let mut records = Vec::new();
    let mut truncated = false;
    residual_sidecar_scope_nodes(
        MotionSidecarScopeIdentity::Master,
        &graph.master_rack,
        &mut records,
        &mut truncated,
    );
    for (position, (stable_id, rack)) in graph.layer_racks.iter().enumerate() {
        residual_sidecar_scope_nodes(
            MotionSidecarScopeIdentity::Layer {
                saved_position: u32::try_from(position).unwrap_or(u32::MAX),
                stable_id: stable_id.get(),
                source_tap_id: export_selective_layer_id(position),
            },
            rack,
            &mut records,
            &mut truncated,
        );
    }
    (records, truncated)
}

fn sidecar_scope_state(value: &ExportMotionScopeMetadata) -> MotionSidecarScopeState {
    MotionSidecarScopeState {
        scope: sidecar_scope_identity(value.scope),
        algorithm_version: value.algorithm_version,
        requested_source: motion_field_source_key(value.requested_source).to_owned(),
        lattice_quality: motion_lattice_quality_key(value.lattice_quality).to_owned(),
        source_origin: motion_origin_key(value.source_origin).to_owned(),
        rendered_source_origin: motion_origin_key(value.rendered_source_origin).to_owned(),
        field_planned: value.field_planned,
        field_attached: value.field_attached,
        source_diagnostic: motion_diagnostic_key(value.source_diagnostic).to_owned(),
        codec_provenance: value
            .codec_provenance
            .map(codec_provenance_key)
            .map(str::to_owned),
        source_generation: value.source_generation,
        frame_ordinal: value.frame_ordinal,
        codec_product_sha256: value.codec_product_sha256.map(sha256_bytes_hex),
        codec_transition_count: value.codec_transition_count,
        codec_elapsed_seconds: value.codec_elapsed_seconds,
        donor_saved_position: value.donor_saved_position,
        donor_stable_id: value.donor_stable_id,
        carrier: motion_carrier_key(value.carrier).to_owned(),
        transplant_admitted: value.transplant_admitted,
        shutter_active: value.shutter_active,
        shutter_angle_degrees: value.shutter_angle_degrees,
        shutter_quality: shutter_quality_key(value.shutter_quality).to_owned(),
        shutter_sample_count: value.shutter_sample_count,
    }
}

/// Per-slot provenance for one resolved Symmetry Field image route.
///
/// The slot index is supplied by the caller rather than searched for, because
/// slot index *is* route identity: compacting an unarmed or tombstoned slot out
/// of the list would silently renumber the surviving one.
fn symmetry_sidecar_image_route(
    slot: u8,
    armed: bool,
    tap: crate::visual_rack::ResolvedImageTap,
) -> SymmetrySidecarImageRoute {
    use crate::image_routing::LayerImageStage;
    use crate::visual_rack::{EdgeTiming, ResolvedImageSource};

    let timing = match tap.timing {
        EdgeTiming::CurrentFrame => "current_frame",
        EdgeTiming::PreviousFrame => "previous_frame",
    };
    let stage_key = |stage: LayerImageStage| match stage {
        LayerImageStage::PreLocalEffects => "pre_local_effects",
        LayerImageStage::PostLocalEffects => "post_local_effects",
    };
    let mut record = SymmetrySidecarImageRoute {
        slot,
        armed,
        source: "one_below",
        timing,
        resolved: true,
        stable_id: None,
        saved_position: None,
        group_id: None,
        stage: None,
    };
    match tap.source {
        ResolvedImageSource::SelectedLayer {
            layer_id,
            saved_position,
            stage,
        } => {
            record.source = "selected_layer";
            record.stable_id = Some(layer_id.get());
            record.saved_position = Some(saved_position.get());
            record.stage = Some(stage_key(stage));
        }
        ResolvedImageSource::MissingSelectedLayer {
            saved_position,
            stage,
        } => {
            record.source = "missing_selected_layer";
            record.resolved = false;
            record.saved_position = Some(saved_position.get());
            record.stage = Some(stage_key(stage));
        }
        ResolvedImageSource::OneBelow => record.source = "one_below",
        ResolvedImageSource::AllBelow => record.source = "all_below",
        ResolvedImageSource::GroupOutput(group_id) => {
            record.source = "group_output";
            record.group_id = Some(group_id.get());
        }
        ResolvedImageSource::MissingGroupOutput(group_id) => {
            record.source = "missing_group_output";
            record.resolved = false;
            record.group_id = Some(group_id.get());
        }
        ResolvedImageSource::CleanProgram => record.source = "clean_program",
        // S3b's etched canvas is a positionless master-scope singleton, so it
        // resolves with no saved position, stable id, group, or layer stage —
        // exactly as `residual_sidecar_route` records it.
        ResolvedImageSource::GestureCanvas => record.source = "gesture_canvas",
        // B16's programme tap is the same positionless master-scope
        // singleton shape.
        ResolvedImageSource::ProgramTap => record.source = "program_tap",
    }
    record
}

/// Per-slot provenance for one resolved Symmetry Field motion route.
fn symmetry_sidecar_motion_route(
    slot: u8,
    armed: bool,
    donor: crate::motion::MotionDonor,
) -> SymmetrySidecarMotionRoute {
    use crate::motion::MotionDonor;

    match donor {
        MotionDonor::None => SymmetrySidecarMotionRoute {
            slot,
            armed,
            donor: "none",
            resolved: true,
            stable_id: None,
            saved_position: None,
        },
        MotionDonor::Selected {
            layer_id,
            saved_position,
        } => SymmetrySidecarMotionRoute {
            slot,
            armed,
            donor: "selected",
            resolved: true,
            stable_id: Some(layer_id.get()),
            saved_position: Some(saved_position.get()),
        },
        MotionDonor::Missing { saved_position } => SymmetrySidecarMotionRoute {
            slot,
            armed,
            donor: "missing",
            resolved: false,
            stable_id: None,
            saved_position: Some(saved_position.get()),
        },
    }
}

fn symmetry_sidecar_node(
    scope: SymmetrySidecarScope,
    node: &crate::visual_rack::RuntimeVisualNode,
    symmetry: crate::symmetry::RuntimeSymmetryParams,
) -> SymmetrySidecarNode {
    let clean = symmetry.sanitized();
    let source_mask = clean.source_mask.sanitized();
    let armed_image = [source_mask.donor0, source_mask.donor1];
    let armed_motion = [clean.motion_mask.slot0, clean.motion_mask.slot1];
    SymmetrySidecarNode {
        scope,
        node_id: node.stable_id.get(),
        enabled: node.enabled,
        wet: node.wet,
        mode: clean.mode,
        boundary: clean.boundary,
        seed: clean.seed,
        exact_bypass: clean.is_exact_bypass(),
        image_routes: clean
            .donors
            .iter()
            .enumerate()
            .map(|(slot, tap)| {
                let slot = u8::try_from(slot).unwrap_or(u8::MAX);
                symmetry_sidecar_image_route(
                    slot,
                    armed_image.get(usize::from(slot)).copied().unwrap_or(false),
                    *tap,
                )
            })
            .collect(),
        motion_routes: clean
            .motion
            .iter()
            .enumerate()
            .map(|(slot, donor)| {
                let slot = u8::try_from(slot).unwrap_or(u8::MAX);
                symmetry_sidecar_motion_route(
                    slot,
                    armed_motion
                        .get(usize::from(slot))
                        .copied()
                        .unwrap_or(false),
                    *donor,
                )
            })
            .collect(),
    }
}

/// Walk every rack the export job actually renders and record the resolved or
/// missing identity of each Symmetry Field route, by slot.
///
/// The walk order matches the planner's own — layer racks, then group racks,
/// then master — so the sidecar reads in the order the frame is planned. Routes
/// were resolved exactly once by [`resolve_export_creative_graph`]; nothing
/// here re-resolves, and a tombstone is recorded as a tombstone rather than
/// being retargeted onto whatever now occupies the vacated position.
fn export_symmetry_sidecar_nodes(graph: &ExportCreativeGraph) -> (Vec<SymmetrySidecarNode>, bool) {
    use crate::visual_rack::{RuntimeVisualNodeKind, RuntimeVisualRack};

    let mut nodes = Vec::new();
    let mut truncated = false;
    let mut collect = |scope: SymmetrySidecarScope, rack: &RuntimeVisualRack| {
        for node in rack.iter() {
            let RuntimeVisualNodeKind::Symmetry(symmetry) = node.kind else {
                continue;
            };
            if nodes.len() == MAX_EXPORT_SYMMETRY_SIDECAR_NODES {
                truncated = true;
                return;
            }
            nodes.push(symmetry_sidecar_node(scope.clone(), node, symmetry));
        }
    };
    for (position, (stable_id, rack)) in graph.layer_racks.iter().enumerate() {
        collect(
            SymmetrySidecarScope::Layer {
                saved_position: u32::try_from(position).unwrap_or(u32::MAX),
                stable_id: stable_id.get(),
            },
            rack,
        );
    }
    for group in graph.composition.groups() {
        collect(
            SymmetrySidecarScope::Group {
                group_id: group.id.get(),
            },
            &group.rack,
        );
    }
    collect(SymmetrySidecarScope::Master, &graph.master_rack);
    (nodes, truncated)
}

fn same_distinct_motion_state(
    left: &MotionSidecarScopeState,
    right: &MotionSidecarScopeState,
) -> bool {
    let mut left = left.clone();
    let mut right = right.clone();
    left.source_generation = None;
    left.frame_ordinal = None;
    left.codec_product_sha256 = None;
    right.source_generation = None;
    right.frame_ordinal = None;
    right.codec_product_sha256 = None;
    left == right
}

impl ExportMotionSidecarAccumulator {
    fn new(
        config: &ExportConfig,
        graph: &ExportCreativeGraph,
        layers: &[ExportLayer],
        master: crate::motion::MotionParams,
        layer_motion: &[crate::motion::MotionParams],
    ) -> Self {
        let mut sources = Vec::new();
        let mut sources_truncated = false;
        for layer in layers {
            if sources.len() == MAX_EXPORT_MOTION_SIDECAR_SOURCES {
                sources_truncated = true;
                break;
            }
            let Some(stable_id) = graph.layer_ids.get(layer.source_index).copied() else {
                sources_truncated = true;
                continue;
            };
            let fingerprint = layer.motion_source.fingerprint.as_ref();
            sources.push(MotionSidecarSource {
                saved_position: u32::try_from(layer.source_index).unwrap_or(u32::MAX),
                stable_id: stable_id.get(),
                source_tap_id: export_selective_layer_id(layer.source_index),
                kind: export_motion_source_kind_key(layer.motion_source.kind).to_owned(),
                logical_name: bounded_export_warning(&layer.motion_source.logical_name),
                persisted_reference: bounded_export_warning(
                    &layer.motion_source.persisted_reference,
                ),
                fingerprint_sha256: fingerprint.map(|identity| identity.sha256.clone()),
                fingerprint_bytes: fingerprint.map(|identity| identity.byte_len),
            });
        }

        let mut authored_scopes = vec![sidecar_authored_scope(
            MotionSidecarScopeIdentity::Master,
            master,
        )];
        let mut authored_scopes_truncated = false;
        for (layer, params) in layers.iter().zip(layer_motion) {
            if authored_scopes.len() == MAX_EXPORT_MOTION_SIDECAR_SCOPES {
                authored_scopes_truncated = true;
                break;
            }
            let Some(stable_id) = graph.layer_ids.get(layer.source_index).copied() else {
                authored_scopes_truncated = true;
                continue;
            };
            authored_scopes.push(sidecar_authored_scope(
                MotionSidecarScopeIdentity::Layer {
                    saved_position: u32::try_from(layer.source_index).unwrap_or(u32::MAX),
                    stable_id: stable_id.get(),
                    source_tap_id: export_selective_layer_id(layer.source_index),
                },
                *params,
            ));
        }
        // Symmetry Field routes are authored topology resolved exactly once
        // per job. Morph carries values only, so recording them here — beside
        // the authored motion scopes rather than per accepted frame — is
        // complete and cannot inflate the distinct-state list.
        let (symmetry_fields, symmetry_fields_truncated) = export_symmetry_sidecar_nodes(graph);
        let (residual_nodes, residual_nodes_truncated) = residual_sidecar_nodes(graph);
        Self {
            artifact: MotionSidecarArtifact::from_config(config),
            sources,
            sources_truncated,
            authored_scopes,
            authored_scopes_truncated,
            residual_nodes,
            residual_nodes_truncated,
            distinct: Vec::new(),
            distinct_truncated: false,
            symmetry_fields,
            symmetry_fields_truncated,
            // Recorded only after an accepted frame, never at construction.
            field_collider: None,
            codec_mosh: None,
            last: None,
        }
    }

    fn observe_accepted(&mut self, metadata: &ExportMotionMetadata) {
        let Some(frame) = metadata.accepted_frame else {
            return;
        };
        let scopes_truncated =
            metadata.scopes_truncated || metadata.scopes.len() > MAX_EXPORT_MOTION_SIDECAR_SCOPES;
        let scopes = metadata
            .scopes
            .iter()
            .take(MAX_EXPORT_MOTION_SIDECAR_SCOPES)
            .map(sidecar_scope_state)
            .collect::<Vec<_>>();
        for state in &scopes {
            if self
                .distinct
                .iter()
                .any(|existing| same_distinct_motion_state(&existing.state, state))
            {
                continue;
            }
            if self.distinct.len() == MAX_EXPORT_MOTION_DISTINCT_STATES {
                self.distinct_truncated = true;
                continue;
            }
            self.distinct.push(MotionSidecarDistinctState {
                first_accepted_frame: frame,
                state: state.clone(),
            });
        }
        // Authored topology, resolved once per job by the shared evaluated
        // plan, so the first accepted frame is complete and later frames cannot
        // contradict it. Recording it here rather than at construction is what
        // makes "only after an accepted frame" literal.
        if self.field_collider.is_none() {
            self.field_collider = metadata.field_collider.clone();
        }
        self.last = Some(MotionSidecarLastFrame {
            accepted_frame: frame,
            scopes,
            scopes_truncated,
        });
    }

    fn finish(self, warnings: Vec<String>) -> ExportMotionSidecar {
        ExportMotionSidecar {
            schema_version: EXPORT_MOTION_SIDECAR_SCHEMA_VERSION,
            artifact: self.artifact,
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            cross_gpu_pixel_identity_guaranteed: false,
            sources: self.sources,
            sources_truncated: self.sources_truncated,
            authored_scopes: self.authored_scopes,
            authored_scopes_truncated: self.authored_scopes_truncated,
            authored_residual_nodes: self.residual_nodes,
            authored_residual_nodes_truncated: self.residual_nodes_truncated,
            distinct_dynamic_states: self.distinct,
            distinct_dynamic_states_truncated: self.distinct_truncated,
            symmetry_fields: self.symmetry_fields,
            symmetry_fields_truncated: self.symmetry_fields_truncated,
            field_collider: self.field_collider,
            codec_mosh: self.codec_mosh,
            last_accepted_frame: self.last,
            warnings: warnings
                .into_iter()
                .take(MAX_EXPORT_WARNINGS)
                .map(|warning| bounded_export_warning(&warning))
                .collect(),
        }
    }
}

/// Export-side storage around the engine-wide pure transport contract.
/// Decoder/GPU ownership remains on [`ExportLayer`]; this value contains only
/// the active Slot's authored law, saved phase, and immutable source facts.
#[derive(Debug, Clone, Copy)]
struct ExportClipTransport {
    authored: ClipTransportConfig,
    state: ClipTransportState,
    source_duration_seconds: f64,
    source_frame_count: u64,
}

impl ExportClipTransport {
    fn new(
        authored: ClipTransportConfig,
        saved_playhead: crate::transport::NormalizedTime,
        source_duration_seconds: f64,
        source_frame_count: u64,
    ) -> Self {
        let authored = authored.sanitized();
        Self {
            authored,
            state: ClipTransportState::at(saved_playhead, authored.direction),
            source_duration_seconds: finite_nonnegative_export(source_duration_seconds),
            source_frame_count: source_frame_count.min(u64::from(u32::MAX)),
        }
    }

    fn from_layer_config(
        layer: &crate::patch::LayerConfig,
        source_duration_seconds: f64,
        source_frame_count: u64,
    ) -> Self {
        let selected = layer
            .active_clip_slot
            .and_then(|id| layer.clip_slots.get(id))
            .or_else(|| layer.clip_slots.iter().next());
        let (authored, saved_playhead) = selected.map_or_else(
            || {
                let fallback = crate::performance::ClipSlotConfig::from_legacy(
                    layer.filename.clone(),
                    layer.source_path.clone(),
                    layer.speed,
                    layer.fps,
                );
                (fallback.transport, fallback.saved_playhead)
            },
            |slot| (slot.transport.sanitized(), slot.saved_playhead),
        );
        Self::new(
            authored,
            saved_playhead,
            source_duration_seconds,
            source_frame_count,
        )
    }

    fn select(
        &mut self,
        config: ClipTransportConfig,
        mut tick: ProgramTransportTick,
    ) -> FrameSelection {
        tick.source_duration_seconds = self.source_duration_seconds;
        tick.source_frame_count = self.source_frame_count;
        let (state, selection) = TransportTimeline::select(&config, self.state, tick);
        self.state = state;
        selection
    }

    /// Select the exact first image before any GPU render can observe the
    /// source texture. This establishes in/out and beat-loop clamping while
    /// retaining the persisted playhead and without advancing program time.
    fn seed_selection(&mut self) -> FrameSelection {
        self.select(
            self.authored,
            ProgramTransportTick {
                program_running: false,
                media_running: false,
                ..ProgramTransportTick::default()
            },
        )
    }
}

fn finite_nonnegative_export(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn estimated_export_source_frames(duration_seconds: f64, fps: f32) -> u64 {
    let frames = finite_nonnegative_export(duration_seconds) * f64::from(fps.max(0.0));
    if frames.is_finite() && frames > 0.0 {
        frames.round().clamp(1.0, f64::from(u32::MAX)) as u64
    } else {
        0
    }
}

fn export_transport_fps(transport: &ExportClipTransport, fallback: f32) -> f32 {
    transport
        .authored
        .sample_fps
        .unwrap_or_else(|| {
            if fallback.is_finite() && fallback > 0.0 {
                f64::from(fallback)
            } else {
                30.0
            }
        })
        .clamp(0.25, 480.0) as f32
}

/// Frame-local, morphable render values detached from decoder/GPU ownership.
/// Keeping these values named prevents positional mistakes as the persisted
/// layer state grows.
#[derive(Clone, Copy)]
struct ExportFrameLayerBase {
    effects: EffectUniforms,
    transform: SpatialTransform,
    opacity: f32,
    mosh_send: f32,
    speed: f32,
    fps: f32,
    blend_mode: BlendMode,
    visible: bool,
    paused: bool,
    bypass_master_fx: bool,
    bypass_temporal_fx: bool,
    /// B7 pattern-synth base, present only for a pattern layer. Morph
    /// samples write into it exactly as they write effects, so the shared
    /// evaluator sees the same post-morph base live rendering sees.
    pattern: Option<crate::pattern_synth::PatternSynthParams>,
}

impl From<&ExportLayer> for ExportFrameLayerBase {
    fn from(layer: &ExportLayer) -> Self {
        Self {
            effects: layer.effects,
            transform: layer.transform,
            opacity: layer.opacity,
            mosh_send: layer.mosh_send,
            speed: layer.speed,
            fps: layer.fps,
            blend_mode: layer.blend_mode,
            visible: layer.visible,
            paused: layer.paused,
            bypass_master_fx: layer.bypass_master_fx,
            bypass_temporal_fx: layer.bypass_temporal_fx,
            pattern: layer.pattern,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ExportMorphOverrides {
    fps: bool,
    effects: bool,
}

/// Retain the historical additive `layerN_speed`/`layerN_fps` modulation
/// response without narrowing the canonical transport law. The modulation
/// engine still evaluates its legacy proxy domains (0.25..4 and 1..240); only
/// that signed delta is applied around the Slot's wider 0..16 / 0.25..480
/// authored value. A zero route is therefore bit-stable even at rate 8.
fn modulated_export_transport_config(
    modulation: &crate::modulation::ModulationFrame,
    layer_index: usize,
    authored: ClipTransportConfig,
    base: &ExportFrameLayerBase,
    morph: ExportMorphOverrides,
) -> ClipTransportConfig {
    let authored_rate = finite_nonnegative_export(f64::from(base.speed)).clamp(0.0, 16.0);
    let rate_proxy = authored_rate.clamp(0.25, 4.0) as f32;

    let authored_fps = if base.fps.is_finite() && base.fps > 0.0 {
        f64::from(base.fps).clamp(0.25, 480.0)
    } else {
        authored.sample_fps.unwrap_or(30.0)
    };
    let fps_proxy = authored_fps.clamp(1.0, 240.0) as f32;
    let proxy = modulation.modulate_layer(
        layer_index,
        &base.effects,
        &base.transform,
        base.opacity,
        base.mosh_send,
        rate_proxy,
        fps_proxy,
    );
    let rate_delta = f64::from(proxy.speed - rate_proxy);
    let fps_delta = f64::from(proxy.fps - fps_proxy);

    let mut config = authored;
    config.rate = (authored_rate + rate_delta).clamp(0.0, 16.0);
    config.sample_fps = if config.sample_fps.is_some() || morph.fps || fps_delta != 0.0 {
        Some((authored_fps + fps_delta).clamp(0.25, 480.0))
    } else {
        None
    };
    // A Morph speed sample is already reflected in `authored_rate`; the
    // additive modulation delta above is intentionally applied after it.
    config.sanitized()
}

/// Source admission proves dimensions and a conservative host-memory plan, but
/// `wgpu` has no portable free-VRAM query. Catch backend validation/internal/OOM
/// failures around the actual Expert-sized texture allocation instead of
/// letting an export worker panic.
fn create_export_source_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &'static str,
) -> Result<(wgpu::Texture, wgpu::TextureView), String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let errors = [
        pollster::block_on(out_of_memory.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
    ]
    .into_iter()
    .flatten()
    .map(|error| error.to_string())
    .collect::<Vec<_>>();
    if let Some(error) = export_source_texture_allocation_error(label, width, height, &errors) {
        return Err(error);
    }
    Ok((texture, texture_view))
}

fn export_source_texture_allocation_error(
    label: &str,
    width: u32,
    height: u32,
    errors: &[String],
) -> Option<String> {
    (!errors.is_empty()).then(|| {
        format!(
            "could not allocate {label} at {width}x{height}: {}",
            errors.join("; ")
        )
    })
}

fn export_gpu_setup_error(width: u32, height: u32, errors: &[String]) -> Option<String> {
    (!errors.is_empty()).then(|| {
        format!(
            "could not initialize export GPU resources at {width}x{height}: {}",
            errors.join("; ")
        )
    })
}

fn create_export_readback_buffer(
    device: &wgpu::Device,
    size: u64,
    label: &'static str,
) -> Result<wgpu::Buffer, String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let errors = [
        pollster::block_on(out_of_memory.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
    ]
    .into_iter()
    .flatten()
    .map(|error| error.to_string())
    .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(buffer)
    } else {
        Err(format!(
            "could not allocate {label} ({size} bytes): {}",
            errors.join("; ")
        ))
    }
}

fn black_substitution_warning(source_index: usize, filename: &str, reason: &str) -> String {
    let filename = if filename.trim().is_empty() {
        "unnamed source"
    } else {
        filename
    };
    format!(
        "Layer {} ('{filename}') {reason}; substituted deterministic black.",
        source_index + 1
    )
}

fn strict_content_addressed_export_source(source_path: &str) -> bool {
    source_path.starts_with(crate::media_source::CONTENT_SHA256_PREFIX)
}

/// A legal muxed export audio source is any media file that can carry an
/// audio stream: an audio file, or a video whose own track is selected.
/// Stills are excluded — `is_supported_visual_file` was the wrong predicate
/// here, admitting images that carry no audio while filtering out
/// audio-only files that are this field's whole point.
fn is_supported_export_audio_source(path: &std::path::Path) -> bool {
    crate::audio::is_supported_audio_file(path) || crate::layers::is_video_file(path)
}

fn resolve_export_file_source<F>(
    persisted_source: &str,
    logical_name: &str,
    runtime_hint: Option<&str>,
    context: &crate::media_source::ResolveContext,
    accepts: F,
    fingerprints: &mut crate::media_source::FingerprintSession,
) -> Result<crate::media_source::ResolvedFile, crate::media_source::SourceResolveError>
where
    F: Fn(&std::path::Path) -> bool,
{
    let expected = crate::media_source::parse_content_reference(persisted_source)?;
    match (
        expected.as_ref(),
        runtime_hint.filter(|hint| !hint.is_empty()),
    ) {
        (Some(expected), Some(runtime_hint)) => crate::media_source::resolve_file_source(
            runtime_hint,
            logical_name,
            context,
            Some(expected),
            accepts,
            fingerprints,
        ),
        _ => crate::media_source::resolve_file_source(
            persisted_source,
            logical_name,
            context,
            None,
            accepts,
            fingerprints,
        ),
    }
}

fn resolve_export_visual_source(
    persisted_source: &str,
    logical_name: &str,
    runtime_hint: Option<&str>,
    context: &crate::media_source::ResolveContext,
    fingerprints: &mut crate::media_source::FingerprintSession,
) -> Result<crate::media_source::ResolvedVisualSource, crate::media_source::SourceResolveError> {
    if crate::layers::spout_sender_from_source_path(persisted_source).is_some()
        || persisted_source == crate::layers::PATTERN_SOURCE_PATH
        || persisted_source == crate::layers::TEXT_PAGE_SOURCE_PATH
    {
        return crate::media_source::resolve_visual_source(
            persisted_source,
            logical_name,
            context,
            None,
            crate::layers::is_supported_visual_file,
            fingerprints,
        );
    }
    resolve_export_file_source(
        persisted_source,
        logical_name,
        runtime_hint,
        context,
        crate::layers::is_supported_visual_file,
        fingerprints,
    )
    .map(crate::media_source::ResolvedVisualSource::File)
}

fn configured_blend_mode(value: &str) -> BlendMode {
    BlendMode::from_key(value).unwrap_or(BlendMode::Normal)
}

fn active_export_clip_source(layer: &crate::patch::LayerConfig) -> (&str, &str) {
    layer
        .active_clip_slot
        .and_then(|id| layer.clip_slots.get(id))
        .or_else(|| layer.clip_slots.iter().next())
        .map_or(
            (layer.filename.as_str(), layer.source_path.as_str()),
            |slot| (slot.filename.as_str(), slot.source_path.as_str()),
        )
}

fn upload_export_texture_checked(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    pixels: &[u8],
    width: u32,
    height: u32,
    label: &'static str,
) -> Result<(), String> {
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("{label} dimensions overflow for {width}x{height}"))?;
    if pixels.len() != expected {
        return Err(format!(
            "{label} has {} bytes; expected {expected} for {width}x{height}",
            pixels.len()
        ));
    }
    scoped_export_gpu_operation(
        device,
        format!("could not upload {label} at {width}x{height}"),
        || {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        },
    )
}

/// Admit only metadata that belongs to the exact source image just uploaded.
/// Decoder rejection/status remains intact for the shared motion evaluator;
/// this seam checks pairing/provenance, not source selection.
fn matching_export_codec_motion(
    motion: Option<CodecMotionProduct>,
    source_generation: u64,
    source_dimensions: [u32; 2],
) -> Option<CodecMotionProduct> {
    motion.filter(|motion| {
        motion.source_generation == source_generation
            && motion.source_dimensions == source_dimensions
            && motion.algorithm_version == crate::motion::MOTION_ALGORITHM_VERSION
    })
}

/// Retain an unavailable source as a one-pixel opaque-black texture. Keeping
/// the layer visible preserves the saved compositing stack (including Normal
/// or Multiply darkening) instead of silently changing the patch by omission.
fn black_placeholder_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_index: usize,
    layer_cfg: &crate::patch::LayerConfig,
) -> Result<ExportLayer, String> {
    let width = 1;
    let height = 1;
    let (texture, texture_view) =
        create_export_source_texture(device, width, height, "Export Black Placeholder")?;
    upload_export_texture_checked(
        device,
        queue,
        &texture,
        &[0, 0, 0, 255],
        width,
        height,
        "Export Black Placeholder",
    )?;

    let mut effects = EffectUniforms {
        resolution: [width as f32, height as f32],
        ..Default::default()
    };
    layer_cfg.effects.apply_to_uniforms(&mut effects);
    effects.clear_master_only_effects();
    let transport = ExportClipTransport::from_layer_config(layer_cfg, 0.0, 0);
    let fps = export_transport_fps(&transport, layer_cfg.fps);
    Ok(ExportLayer {
        source_index,
        motion_source: ExportMotionSourceRecord::unavailable(layer_cfg),
        decoder: None,
        codec_motion: None,
        codec_motion_predecessor: None,
        _still_source: None,
        texture,
        texture_view,
        effects,
        transform: layer_cfg.transform.sanitized(),
        opacity: layer_cfg.opacity,
        mosh_send: crate::layers::clamp_layer_mosh_send(layer_cfg.mosh_send),
        blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
        bypass_master_fx: layer_cfg.bypass_master_fx,
        bypass_temporal_fx: layer_cfg.bypass_temporal_fx,
        matte: LayerMatte::default(),
        reroll_on_loop: false,
        consumed_loop_generation: 0,
        transport,
        speed: transport.authored.rate as f32,
        visible: layer_cfg.visible,
        paused: true,
        fps,
        width,
        height,
        pattern: None,
    })
}

/// B7 pattern-synth reconstruction: no bytes exist or could — the per-frame
/// GPU pass computes the picture from the patch's own authored values, so
/// the texture is created renderable and uploaded nothing.
fn pattern_export_layer(
    device: &wgpu::Device,
    source_index: usize,
    layer_cfg: &crate::patch::LayerConfig,
) -> Result<ExportLayer, String> {
    let width = crate::pattern_synth::PATTERN_SYNTH_WIDTH;
    let height = crate::pattern_synth::PATTERN_SYNTH_HEIGHT;
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export Pattern Layer"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let errors = [
        pollster::block_on(out_of_memory.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(validation.pop()),
    ];
    if let Some(error) = errors.into_iter().flatten().next() {
        return Err(format!(
            "could not allocate Export Pattern Layer at {width}x{height}: {error}"
        ));
    }

    let mut effects = EffectUniforms {
        resolution: [width as f32, height as f32],
        ..Default::default()
    };
    layer_cfg.effects.apply_to_uniforms(&mut effects);
    effects.clear_master_only_effects();
    let transport = ExportClipTransport::from_layer_config(layer_cfg, 0.0, 0);
    let fps = export_transport_fps(&transport, layer_cfg.fps);
    let params = layer_cfg
        .pattern
        .map(crate::patch::PatternSynthConfig::to_params)
        .unwrap_or_default();
    Ok(ExportLayer {
        source_index,
        motion_source: {
            let (logical_name, persisted_reference) = active_export_clip_source(layer_cfg);
            ExportMotionSourceRecord {
                kind: ExportMotionSourceKind::PatternSynth,
                logical_name: logical_name.to_owned(),
                persisted_reference: persisted_reference.to_owned(),
                fingerprint: None,
            }
        },
        decoder: None,
        codec_motion: None,
        codec_motion_predecessor: None,
        _still_source: None,
        texture,
        texture_view,
        effects,
        transform: layer_cfg.transform.sanitized(),
        opacity: layer_cfg.opacity,
        mosh_send: crate::layers::clamp_layer_mosh_send(layer_cfg.mosh_send),
        blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
        bypass_master_fx: layer_cfg.bypass_master_fx,
        bypass_temporal_fx: layer_cfg.bypass_temporal_fx,
        matte: LayerMatte::default(),
        reroll_on_loop: false,
        consumed_loop_generation: 0,
        transport,
        speed: transport.authored.rate as f32,
        visible: layer_cfg.visible,
        paused: layer_cfg.paused,
        fps,
        width,
        height,
        pattern: Some(params),
    })
}

/// B7 text-page reconstruction: the identical CPU raster the live layer ran,
/// from the patch's own authored values, uploaded once like a still.
fn text_page_export_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_index: usize,
    layer_cfg: &crate::patch::LayerConfig,
    media_policy: &crate::media_safety::MediaSafetyPolicy,
    device_limits: crate::media_safety::MediaDeviceLimits,
) -> Result<ExportLayer, String> {
    let width = crate::text_page::TEXT_PAGE_WIDTH;
    let height = crate::text_page::TEXT_PAGE_HEIGHT;
    media_policy
        .plan(
            crate::media_safety::MediaSourceKind::Still,
            width,
            height,
            device_limits,
        )
        .map_err(|error| format!("text page source rejected: {error}"))?;
    let params = layer_cfg
        .text_page
        .as_ref()
        .map(crate::patch::TextPageConfig::to_params)
        .unwrap_or_default();
    let page = crate::text_page::render_text_page(&params, crate::text_page::bundled_fonts());
    let (texture, texture_view) =
        create_export_source_texture(device, width, height, "Export Text Page Layer")?;
    upload_export_texture_checked(
        device,
        queue,
        &texture,
        &page,
        width,
        height,
        "Export Text Page Layer",
    )?;

    let mut effects = EffectUniforms {
        resolution: [width as f32, height as f32],
        ..Default::default()
    };
    layer_cfg.effects.apply_to_uniforms(&mut effects);
    effects.clear_master_only_effects();
    let transport = ExportClipTransport::from_layer_config(layer_cfg, 0.0, 1);
    let fps = export_transport_fps(&transport, layer_cfg.fps);
    Ok(ExportLayer {
        source_index,
        motion_source: {
            let (logical_name, persisted_reference) = active_export_clip_source(layer_cfg);
            ExportMotionSourceRecord {
                kind: ExportMotionSourceKind::TextPage,
                logical_name: logical_name.to_owned(),
                persisted_reference: persisted_reference.to_owned(),
                fingerprint: None,
            }
        },
        decoder: None,
        codec_motion: None,
        codec_motion_predecessor: None,
        _still_source: None,
        texture,
        texture_view,
        effects,
        transform: layer_cfg.transform.sanitized(),
        opacity: layer_cfg.opacity,
        mosh_send: crate::layers::clamp_layer_mosh_send(layer_cfg.mosh_send),
        blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
        bypass_master_fx: layer_cfg.bypass_master_fx,
        bypass_temporal_fx: layer_cfg.bypass_temporal_fx,
        matte: LayerMatte::default(),
        reroll_on_loop: false,
        consumed_loop_generation: 0,
        transport,
        speed: transport.authored.rate as f32,
        visible: layer_cfg.visible,
        paused: layer_cfg.paused,
        fps,
        width,
        height,
        pattern: None,
    })
}

/// Upload an immutable source exactly once. With no decoder handle, the frame
/// loop can never advance or reopen this source; time-varying effects still
/// evaluate normally against these identical input pixels on every frame.
fn still_export_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_index: usize,
    layer_cfg: &crate::patch::LayerConfig,
    decoded: DecodedStillImage,
    fingerprint: Option<crate::media_source::ContentIdentity>,
) -> Result<ExportLayer, String> {
    let width = decoded.width;
    let height = decoded.height;
    let (texture, texture_view) =
        create_export_source_texture(device, width, height, "Export Still Layer")?;
    upload_export_texture_checked(
        device,
        queue,
        &texture,
        &decoded.rgba,
        width,
        height,
        "Export Still Layer",
    )?;

    let mut effects = EffectUniforms {
        resolution: [width as f32, height as f32],
        ..Default::default()
    };
    layer_cfg.effects.apply_to_uniforms(&mut effects);
    effects.clear_master_only_effects();
    let transport = ExportClipTransport::from_layer_config(layer_cfg, 0.0, 1);
    let fps = export_transport_fps(&transport, layer_cfg.fps);
    Ok(ExportLayer {
        source_index,
        motion_source: {
            let (logical_name, persisted_reference) = active_export_clip_source(layer_cfg);
            ExportMotionSourceRecord {
                kind: ExportMotionSourceKind::Still,
                logical_name: logical_name.to_owned(),
                persisted_reference: persisted_reference.to_owned(),
                fingerprint,
            }
        },
        decoder: None,
        codec_motion: None,
        codec_motion_predecessor: None,
        _still_source: Some(decoded),
        texture,
        texture_view,
        effects,
        transform: layer_cfg.transform.sanitized(),
        opacity: layer_cfg.opacity,
        mosh_send: crate::layers::clamp_layer_mosh_send(layer_cfg.mosh_send),
        blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
        bypass_master_fx: layer_cfg.bypass_master_fx,
        bypass_temporal_fx: layer_cfg.bypass_temporal_fx,
        matte: LayerMatte::default(),
        reroll_on_loop: false,
        consumed_loop_generation: 0,
        transport,
        speed: transport.authored.rate as f32,
        visible: layer_cfg.visible,
        paused: layer_cfg.paused,
        fps,
        width,
        height,
        pattern: None,
    })
}

fn media_has_audio_stream(path: &str) -> Result<bool, String> {
    ffmpeg_next::init().map_err(|error| format!("failed to initialize media probing: {error}"))?;
    let input = ffmpeg_next::format::input(path)
        .map_err(|error| format!("failed to open selected media: {error}"))?;
    Ok(input
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .is_some())
}

fn validate_export_dimensions(width: u32, height: u32) -> Result<u64, String> {
    if width > MAX_EXPORT_EDGE || height > MAX_EXPORT_EDGE {
        return Err(format!(
            "export dimensions {width}x{height} exceed the {MAX_EXPORT_EDGE}px safety edge limit"
        ));
    }
    validate_media_dimensions(width, height, None)
        .map(|rgba_bytes| rgba_bytes / 4)
        .map_err(|error| {
            format!("invalid export dimensions: {error}; maximum pixels {MAX_MEDIA_PIXELS}")
        })
}

/// Construct the complete ffmpeg argv separately so duration and audio
/// policy can be verified without a GPU or subprocess. Raw video always maps
/// explicitly from input zero. Optional audio starts at zero at 1x, then
/// `apad` + `atrim` yields exactly the requested program duration.
fn build_ffmpeg_args(config: &ExportConfig, audio_path: Option<&str>) -> Vec<String> {
    let size = format!("{}x{}", config.width, config.height);
    let fps = config.fps.to_string();
    let duration = format!("{:.6}", config.duration_secs);
    let mut args = [
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    args.push(size);
    args.extend([
        "-framerate".to_owned(),
        fps,
        "-i".to_owned(),
        "pipe:0".to_owned(),
    ]);
    if let Some(path) = audio_path {
        args.extend(["-i".to_owned(), path.to_owned()]);
    }

    args.extend(
        [
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    if audio_path.is_some() {
        args.extend([
            "-map".to_owned(),
            "1:a:0".to_owned(),
            "-filter:a".to_owned(),
            format!("asetpts=PTS-STARTPTS,apad,atrim=end={duration}"),
            "-c:a".to_owned(),
            "aac".to_owned(),
            "-b:a".to_owned(),
            "192k".to_owned(),
        ]);
    } else {
        args.push("-an".to_owned());
    }
    args.extend([
        "-t".to_owned(),
        duration,
        "-movflags".to_owned(),
        "+faststart".to_owned(),
        config.output_path.clone(),
    ]);
    args
}

fn export_frame_count(fps: u32, duration_secs: f32) -> u64 {
    (fps as f64 * duration_secs as f64).ceil() as u64
}

/// Frame-indexed program transport. A patch saved under global Pause exports
/// its held program state for the complete requested duration; decoder frames,
/// shader time, modulation/audio, morph, temporal history, and NTSC therefore
/// share the same frozen timestamp.
fn export_program_transport(frame_num: u64, frame_interval: f32, paused: bool) -> (f32, f32) {
    if paused {
        (0.0, 0.0)
    } else {
        (
            frame_num as f32 * frame_interval,
            if frame_num == 0 { 0.0 } else { frame_interval },
        )
    }
}

fn apply_export_loop_generation(
    effects: &mut EffectUniforms,
    reroll_on_loop: bool,
    consumed_generation: &mut u64,
    decoded_generation: u64,
) -> u64 {
    let loops_advanced = decoded_generation.saturating_sub(*consumed_generation);
    *consumed_generation = (*consumed_generation).max(decoded_generation);
    if reroll_on_loop && loops_advanced > 0 {
        effects.random_seed =
            crate::randomization::advance_seed(effects.random_seed, loops_advanced);
    }
    loops_advanced
}

/// Mirror live pause semantics for transient modulation state. Routing caches
/// are runtime state rather than patch data, so a recalled paused patch holds
/// their deterministic reconstructed value (zero) instead of sampling an LFO,
/// pad, or imported-audio source only in the offline renderer.
fn update_export_modulation(
    matrix: &mut crate::modulation::ModMatrix,
    beat: f64,
    delta_seconds: f32,
    paused: bool,
) {
    if paused {
        matrix.reset_update_timing();
    } else {
        matrix.update_at_beat(beat, delta_seconds);
    }
}

/// Live Pause holds the materialized patch bases. Re-sampling an active morph
/// only in export would produce a different held frame, especially when the
/// morph target has transient routing state.
#[cfg(test)]
fn export_morph_sample(
    morph: Option<&crate::morph::Morph>,
    beat: f64,
    offset: f32,
    paused: bool,
) -> Option<crate::morph::MorphSample> {
    let position = export_morph_position(morph, beat, offset, paused)?;
    morph?.sample(position)
}

fn export_morph_position(
    morph: Option<&crate::morph::Morph>,
    beat: f64,
    offset: f32,
    paused: bool,
) -> Option<f32> {
    if paused {
        return None;
    }
    let morph = morph.filter(|morph| morph.active())?;
    Some((morph.position_at_beat(beat) + offset).clamp(0.0, 1.0))
}

/// The export stack has no live layer IDs, but patch source positions are
/// stable for the lifetime of one job. Offset by one so every planned ID is
/// nonzero and can be mapped back without a sentinel collision.
fn export_selective_layer_id(source_index: usize) -> u64 {
    source_index as u64 + 1
}

/// Detached stable creative state for one export job. Saved positions are
/// resolved exactly once at admission; every frame clones this small bounded
/// value graph before Morph and stable modulation are projected into it.
#[derive(Debug, Clone)]
struct ExportCreativeGraph {
    layer_ids: Box<[StableLayerId]>,
    master_rack: RuntimeVisualRack,
    layer_racks: Vec<(StableLayerId, RuntimeVisualRack)>,
    composition: RuntimeComposition,
    address_book: crate::modulation::StableModAddressBook,
    /// The job's Study library, built once from the patch's own `studies`
    /// section so live and export resolve every digest identically. A
    /// document that fails validation aborts the job before the first frame.
    studies: crate::study_eval::StudyProgramLibrary,
}

fn export_saved_position(position: usize) -> Result<SavedLayerPosition, String> {
    u32::try_from(position)
        .ok()
        .and_then(SavedLayerPosition::new)
        .ok_or_else(|| format!("export layer position {position} exceeds saved-position bounds"))
}

fn export_stable_layer_id(position: usize) -> Result<StableLayerId, String> {
    u64::try_from(position)
        .ok()
        .and_then(|position| position.checked_add(1))
        .and_then(StableLayerId::new)
        .ok_or_else(|| format!("export layer position {position} exceeds stable-ID bounds"))
}

/// Resolve the same persisted rack/composition model used by live loading,
/// but against deterministic job-local IDs. The patch layer vector remains
/// front-to-back; only omitted legacy composition is synthesized in the
/// documented back-to-front root order.
fn resolve_export_creative_graph(patch: &PatchState) -> Result<ExportCreativeGraph, String> {
    let layer_ids = (0..patch.layers.len())
        .map(export_stable_layer_id)
        .collect::<Result<Vec<_>, _>>()?;
    let saved_positions = (0..patch.layers.len())
        .map(export_saved_position)
        .collect::<Result<Vec<_>, _>>()?;
    let synthesized;
    let saved_composition = match patch.composition.as_ref() {
        Some(composition) => composition,
        None => {
            let back_to_front = saved_positions.iter().rev().copied().collect::<Vec<_>>();
            synthesized = CompositionTree::legacy_for_layers(&back_to_front)
                .map_err(|error| format!("synthesize export composition: {error}"))?;
            &synthesized
        }
    };
    let resolve_position =
        |position: SavedLayerPosition| layer_ids.get(position.get() as usize).copied();
    let composition = saved_composition
        .resolve(resolve_position)
        .map_err(|error| format!("resolve export composition: {error}"))?;
    let group_exists = |group_id| composition.contains_group(group_id);
    let master_rack = patch.effective_master_rack().resolve_routes(
        |position| layer_ids.get(position.get() as usize).copied(),
        group_exists,
    );
    let layer_racks = patch
        .layers
        .iter()
        .enumerate()
        .map(|(position, _)| {
            let id = layer_ids[position];
            let rack = patch.effective_layer_rack(position).ok_or_else(|| {
                format!("export layer rack {position} is outside the saved stack")
            })?;
            Ok((
                id,
                rack.resolve_routes(
                    |saved| layer_ids.get(saved.get() as usize).copied(),
                    group_exists,
                ),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let address_book = crate::modulation::StableModAddressBook::from_composition(
        &master_rack,
        &layer_racks,
        &composition,
    )
    .map_err(|error| format!("resolve export stable modulation: {error}"))?;
    let mut studies = crate::study_eval::StudyProgramLibrary::default();
    for document in &patch.studies {
        studies
            .insert(document.clone())
            .map_err(|error| format!("export study document rejected: {error}"))?;
    }
    Ok(ExportCreativeGraph {
        layer_ids: layer_ids.into_boxed_slice(),
        master_rack,
        layer_racks,
        composition,
        address_book,
        studies,
    })
}

/// Apply only topology-compatible creative Morph values. Stable identities,
/// membership, routes, enabled state, and monotonic cursors stay owned by the
/// job's resolved patch graph, exactly as in the live frame boundary.
fn apply_export_creative_morph(
    sample: &crate::morph::MorphSample,
    graph: &mut ExportCreativeGraph,
) {
    let group_exists = |group_id| graph.composition.contains_group(group_id);
    if let Some(saved) = &sample.master_rack {
        let sampled = saved.resolve_routes(
            |position| graph.layer_ids.get(position.get() as usize).copied(),
            group_exists,
        );
        let _ = crate::morph::apply_runtime_rack_values_strict(&sampled, &mut graph.master_rack);
    }
    if let Some(saved_racks) = &sample.layer_racks {
        if saved_racks.len() == graph.layer_racks.len() {
            for (saved, (_, live)) in saved_racks.iter().zip(&mut graph.layer_racks) {
                let sampled = saved.resolve_routes(
                    |position| graph.layer_ids.get(position.get() as usize).copied(),
                    group_exists,
                );
                let _ = crate::morph::apply_runtime_rack_values_strict(&sampled, live);
            }
        }
    }
    if let Some(saved) = &sample.composition {
        if let Ok(sampled) =
            saved.resolve(|position| graph.layer_ids.get(position.get() as usize).copied())
        {
            let _ = crate::morph::apply_runtime_composition_values_strict(
                &sampled,
                &mut graph.composition,
            );
        }
    }
}

fn resolve_export_score_loop_driver(
    driver: crate::patch::CollisionScoreLoopDriverConfig,
    layer_ids: &[crate::image_routing::StableLayerId],
) -> crate::temporal::CollisionScoreLoopDriver {
    use crate::patch::CollisionScoreLoopDriverConfig as Saved;
    use crate::temporal::CollisionScoreLoopDriver as Runtime;
    match driver {
        Saved::None => Runtime::None,
        Saved::MissingSelectedLayer { saved_position } => {
            Runtime::MissingSelectedLayer { saved_position }
        }
        Saved::SelectedLayer { saved_position } => layer_ids
            .get(saved_position.get() as usize)
            .copied()
            .map_or(
                Runtime::MissingSelectedLayer { saved_position },
                |layer_id| Runtime::SelectedLayer {
                    layer_id,
                    saved_position,
                },
            ),
    }
}

#[derive(Clone)]
struct ExportFrameMorphWorld {
    creative_graph: ExportCreativeGraph,
    master: EffectUniforms,
    master_transform: SpatialTransform,
    master_motion: crate::motion::MotionParams,
    ntsc: crate::ntsc::NtscParams,
    temporal: crate::effects::params::TemporalParams,
    /// Authored gesture-canvas controls only. A Morph slot names a recording;
    /// it never owns one, so no morph position can add, remove, retime, or
    /// rewrite a recorded gesture event offline any more than it can live.
    gesture_canvas: crate::gesture_canvas::GestureCanvasParams,
    layer_bases: Vec<ExportFrameLayerBase>,
    layer_motion: Vec<crate::motion::MotionParams>,
    morph_overrides: Vec<ExportMorphOverrides>,
}

fn resolved_export_motion(
    config: Option<crate::patch::MotionConfig>,
    layer_ids: &[StableLayerId],
) -> crate::motion::MotionParams {
    // Delegate to the single shared resolver rather than resolving one donor
    // by hand here. `MotionConfig::resolve_runtime` sanitizes and then binds
    // EVERY Motion-subsystem donor the block owns — the Faraday transplant's
    // and both Field Collider inputs — so an offline render can never resolve a
    // different set of donors than a live one.
    config
        .unwrap_or_default()
        .sanitized()
        .resolve_runtime(layer_ids)
}

fn resolved_export_patch_temporal(
    config: Option<&crate::patch::TemporalConfig>,
    layer_ids: &[StableLayerId],
) -> crate::effects::params::TemporalParams {
    let mut params = config.map_or_else(Default::default, crate::patch::TemporalConfig::to_params);
    if let Some(originals) = config.and_then(|temporal| temporal.originals.as_ref()) {
        params.originals.garden.matte_route =
            originals.garden.matte_route.resolve_runtime(layer_ids);
        params.originals.garden.motion_route =
            originals.garden.motion_route.resolve_runtime(layer_ids);
        params.originals.score.loop_driver =
            resolve_export_score_loop_driver(originals.score.loop_driver, layer_ids);
    }
    params
}

fn resolved_export_morph_temporal(
    snapshot: crate::morph::MorphTemporalSnapshot,
    layer_ids: &[StableLayerId],
) -> crate::effects::params::TemporalParams {
    let mut params = snapshot.to_params();
    params.originals.garden.matte_route = snapshot
        .originals
        .garden
        .matte_route
        .resolve_runtime(layer_ids);
    params.originals.garden.motion_route = snapshot
        .originals
        .garden
        .motion_route
        .resolve_runtime(layer_ids);
    params.originals.score.loop_driver =
        resolve_export_score_loop_driver(snapshot.originals.score.loop_driver, layer_ids);
    params
}

fn apply_export_morph_world(
    sample: &crate::morph::MorphSample,
    layers: &[ExportLayer],
    world: &mut ExportFrameMorphWorld,
) {
    apply_export_creative_morph(sample, &mut world.creative_graph);
    sample.master.apply_to(&mut world.master);
    if let Some(transform) = sample.master_transform {
        world.master_transform = transform.sanitized();
    }
    if let Some(motion) = sample.master_motion {
        world.master_motion = resolved_export_motion(Some(motion), &world.creative_graph.layer_ids);
    }
    world.ntsc = sample.ntsc.to_params();
    world.temporal =
        resolved_export_morph_temporal(sample.temporal, &world.creative_graph.layer_ids);
    // Values only, and only when both slots name the same recording. The
    // route-match law lives in `MorphSample::apply_gesture_to`, shared with
    // live, so offline cannot interpolate across two different performances.
    sample.apply_gesture_to(&mut world.gesture_canvas);
    for sampled in &sample.layers {
        if let Some((position, _)) = layers
            .iter()
            .enumerate()
            .find(|(_, layer)| layer.source_index == sampled.position)
        {
            world.layer_bases[position].opacity = sampled.opacity;
            if let Some(mosh_send) = sampled.mosh_send {
                world.layer_bases[position].mosh_send = mosh_send;
            }
            world.layer_bases[position].speed = sampled.speed;
            if let Some(fps) = sampled.fps {
                world.layer_bases[position].fps = fps;
                world.morph_overrides[position].fps = true;
            }
            if let Some(effects) = sampled.effects {
                effects.apply_to(&mut world.layer_bases[position].effects);
                world.morph_overrides[position].effects = true;
            }
            if let Some(transform) = sampled.transform {
                world.layer_bases[position].transform = transform.sanitized();
            }
            if let Some(motion) = sampled.motion {
                world.layer_motion[position] =
                    resolved_export_motion(Some(motion), &world.creative_graph.layer_ids);
            }
            if let Some(key_threshold) = sampled.key_threshold {
                world.layer_bases[position].effects.key_threshold = key_threshold;
            }
            if let Some(blend_mode) = sampled.blend_mode {
                world.layer_bases[position].blend_mode = blend_mode.to_blend_mode();
            }
            if let Some(visible) = sampled.visible {
                world.layer_bases[position].visible = visible;
            }
            if let Some(paused) = sampled.paused {
                world.layer_bases[position].paused = paused;
            }
            if let Some(bypass_master_fx) = sampled.bypass_master_fx {
                world.layer_bases[position].bypass_master_fx = bypass_master_fx;
            }
            if let Some(bypass_temporal_fx) = sampled.bypass_temporal_fx {
                world.layer_bases[position].bypass_temporal_fx = bypass_temporal_fx;
            }
        }
    }
}

#[derive(Debug)]
struct ValidatedMorphCandidate<T> {
    requested_position: f32,
    selected_position: f32,
    requested_error: Option<String>,
    value: T,
}

/// Shared deterministic fallback law used by offline Morph preflight. The
/// exact requested interpolation is always tried first; then the nearest
/// captured endpoint (ties choose B), then the other endpoint. It never
/// depends on a prior rendered frame.
fn select_valid_morph_candidate<T>(
    requested_position: f32,
    mut validate: impl FnMut(f32) -> Result<T, String>,
) -> Result<ValidatedMorphCandidate<T>, String> {
    let requested_position = requested_position.clamp(0.0, 1.0);

    let mut requested_error = None;
    for position in crate::morph::preflight_sample_positions(requested_position) {
        match validate(position) {
            Ok(value) => {
                return Ok(ValidatedMorphCandidate {
                    requested_position,
                    selected_position: position,
                    requested_error,
                    value,
                });
            }
            Err(error) if requested_error.is_none() => requested_error = Some(error),
            Err(_) => {}
        }
    }
    Err(format!(
        "Morph sample {requested_position:.4} and both captured endpoints were rejected: {}",
        requested_error
            .as_deref()
            .unwrap_or("creative graph rejected")
    ))
}

fn evaluate_export_morph_world(
    world: &ExportFrameMorphWorld,
    layers: &[ExportLayer],
    modulation: &crate::modulation::ModulationFrame,
    context: FramePlanContext,
) -> EvaluatedFramePlan {
    EvaluatedFramePlan::evaluate(
        modulation,
        context,
        MasterFrameInput {
            effects: &world.master,
            transform: &world.master_transform,
            ntsc: &world.ntsc,
            temporal: &world.temporal,
        },
        layers
            .iter()
            .zip(&world.layer_bases)
            .enumerate()
            .map(|(slot, (layer, base))| LayerFrameInput {
                source: SourceTap::new(
                    export_selective_layer_id(layer.source_index),
                    slot,
                    layer.width,
                    layer.height,
                ),
                effects: &base.effects,
                transform: &base.transform,
                opacity: base.opacity,
                mosh_send: base.mosh_send,
                speed: base.speed,
                fps: base.fps,
                blend_mode: base.blend_mode,
                visible: base.visible,
                paused: base.paused,
                bypass_master_fx: base.bypass_master_fx,
                bypass_temporal_fx: base.bypass_temporal_fx,
                pattern: base.pattern.as_ref(),
            }),
    )
}

#[derive(Clone, Copy)]
struct ExportMotionPlanAdapter<'a> {
    master: crate::motion::MotionParams,
    layers: &'a [crate::motion::MotionParams],
    sources: &'a [ExportLayer],
    limits: crate::motion::MotionDeviceLimits,
    modulation: &'a crate::modulation::ModulationFrame,
    /// Authored route topology, deliberately independent of this frame's
    /// deterministic source samples.
    authored_motion_modulation: bool,
    shutter_samples: ExportShutterSamples,
    /// `None` uses decoder-advertised status for the first pure plan. A retry
    /// supplies the exact rasterization/attachment outcome aligned with
    /// `sources`, so Auto can fall back before any GPU work is encoded.
    codec_availability: Option<&'a [bool]>,
}

fn apply_export_shutter_samples(
    mut params: crate::motion::MotionParams,
    policy: ExportShutterSamples,
) -> crate::motion::MotionParams {
    if let Some(quality) = policy.quality_override() {
        params.shutter.quality = quality;
    }
    params
}

fn effective_export_motion_params(
    master: crate::motion::MotionParams,
    layers: &[crate::motion::MotionParams],
    modulation: &crate::modulation::ModulationFrame,
    policy: ExportShutterSamples,
) -> (
    crate::motion::MotionParams,
    Vec<crate::motion::MotionParams>,
) {
    let master = modulation.modulate_motion(&master);
    (
        apply_export_shutter_samples(master, policy),
        layers
            .iter()
            .enumerate()
            .map(|(index, params)| modulation.modulate_layer_motion(index, params))
            .map(|params| apply_export_shutter_samples(params, policy))
            .collect(),
    )
}

struct ExportMotionFieldProduct {
    scope: crate::visual_rack::VisualScopeId,
    codec_identity: crate::video::CodecMotionProductIdentity,
    source_generation: u64,
    frame_ordinal: u64,
    field: crate::motion::MotionField,
}

impl ExportMotionFieldProduct {
    fn attachment(
        &self,
    ) -> crate::evaluated_frame::evaluated_composition::MotionFieldAttachment<'_> {
        crate::evaluated_frame::evaluated_composition::MotionFieldAttachment {
            scope: self.scope,
            source_generation: self.source_generation,
            frame_ordinal: self.frame_ordinal,
            product_content_sha256: self.codec_identity.content_sha256,
            algorithm_version: self.field.algorithm_version(),
            source_dimensions: self.field.source_dimensions(),
            grid: self.field.grid(),
            field: &self.field,
        }
    }
}

fn export_codec_motion_fields(
    plan: &crate::evaluated_frame::evaluated_composition::AdvancedCompositionPlan,
    graph: &ExportCreativeGraph,
    layers: &[ExportLayer],
) -> (Vec<ExportMotionFieldProduct>, Vec<String>) {
    export_codec_motion_fields_from(
        plan,
        graph,
        layers
            .iter()
            .map(|layer| (layer.source_index, layer.codec_motion.as_ref())),
    )
}

fn export_codec_motion_fields_from<'a>(
    plan: &crate::evaluated_frame::evaluated_composition::AdvancedCompositionPlan,
    graph: &ExportCreativeGraph,
    sources: impl IntoIterator<Item = (usize, Option<&'a CodecMotionProduct>)>,
) -> (Vec<ExportMotionFieldProduct>, Vec<String>) {
    let Some(motion_plan) = plan.motion().advanced() else {
        return (Vec::new(), Vec::new());
    };
    let sources = sources
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut products = Vec::new();
    let mut diagnostics = Vec::new();
    for field_plan in motion_plan.fields() {
        if field_plan.source.origin != crate::motion::MotionFieldOrigin::CodecVectors {
            continue;
        }
        let crate::visual_rack::VisualScopeId::Layer(layer_id) = field_plan.scope else {
            diagnostics.push(format!(
                "Motion field {:?} selected codec vectors without a layer source.",
                field_plan.scope
            ));
            continue;
        };
        let Some(saved_position) = graph
            .layer_ids
            .iter()
            .position(|candidate| *candidate == layer_id)
        else {
            diagnostics.push(format!(
                "Motion field {:?} has no export source identity.",
                field_plan.scope
            ));
            continue;
        };
        let codec = match sources.get(&saved_position).copied() {
            None => {
                diagnostics.push(format!(
                    "Motion field {:?} has no prepared export source.",
                    field_plan.scope
                ));
                continue;
            }
            Some(None) => {
                diagnostics.push(format!(
                    "Motion field {:?} lost its exact codec metadata before attachment.",
                    field_plan.scope
                ));
                continue;
            }
            Some(Some(codec)) => codec,
        };
        let Some(scope_plan) = motion_plan.scope(field_plan.scope) else {
            diagnostics.push(format!(
                "Motion field {:?} has no evaluated scope.",
                field_plan.scope
            ));
            continue;
        };
        let Some(codec_identity) = codec.exact_identity() else {
            diagnostics.push(format!(
                "Motion field {:?} codec product lacks exact adjacent-reference identity.",
                field_plan.scope
            ));
            continue;
        };
        let field = match codec.rasterize(scope_plan.params.lattice_quality) {
            Ok(Some(field)) => field,
            Ok(None) => {
                diagnostics.push(format!(
                    "Motion field {:?} codec records contained no past-reference field.",
                    field_plan.scope
                ));
                continue;
            }
            Err(error) => {
                diagnostics.push(format!(
                    "Motion field {:?} codec rasterization was rejected ({error}).",
                    field_plan.scope
                ));
                continue;
            }
        };
        let product = ExportMotionFieldProduct {
            scope: field_plan.scope,
            codec_identity,
            source_generation: codec.source_generation,
            frame_ordinal: codec.frame_ordinal,
            field,
        };
        if field_plan.accepts(product.attachment()) {
            products.push(product);
        } else {
            diagnostics.push(format!(
                "Motion field {:?} failed exact attachment provenance.",
                field_plan.scope
            ));
        }
    }
    (products, diagnostics)
}

fn retain_exact_export_codec_fields(
    plan: &EvaluatedCompositionPlan,
    products: &mut Vec<ExportMotionFieldProduct>,
) {
    let motion = match plan {
        EvaluatedCompositionPlan::Advanced(plan) => plan.motion().advanced(),
        EvaluatedCompositionPlan::LegacyExact(_) => None,
    };
    products.retain(|product| {
        motion.is_some_and(|motion| {
            motion.fields().iter().any(|field| {
                field.source.origin == crate::motion::MotionFieldOrigin::CodecVectors
                    && field.scope == product.scope
                    && field.accepts(product.attachment())
            })
        })
    });
}

fn export_codec_fields_complete(
    plan: &EvaluatedCompositionPlan,
    products: &[ExportMotionFieldProduct],
) -> bool {
    let planned = planned_export_codec_scopes(plan);
    let attached = products
        .iter()
        .map(|product| product.scope)
        .collect::<std::collections::BTreeSet<_>>();
    planned.is_subset(&attached)
}

fn planned_export_codec_scopes(
    plan: &EvaluatedCompositionPlan,
) -> std::collections::BTreeSet<crate::visual_rack::VisualScopeId> {
    let EvaluatedCompositionPlan::Advanced(plan) = plan else {
        return std::collections::BTreeSet::new();
    };
    plan.motion()
        .advanced()
        .map_or_else(std::collections::BTreeSet::new, |motion| {
            motion
                .fields()
                .iter()
                .filter(|field| {
                    field.source.origin == crate::motion::MotionFieldOrigin::CodecVectors
                })
                .map(|field| field.scope)
                .collect()
        })
}

fn exact_export_codec_availability(
    graph: &ExportCreativeGraph,
    layers: &[ExportLayer],
    products: &[ExportMotionFieldProduct],
) -> Vec<bool> {
    let attached = products
        .iter()
        .map(|product| product.scope)
        .collect::<std::collections::BTreeSet<_>>();
    layers
        .iter()
        .map(|layer| {
            graph
                .layer_ids
                .get(layer.source_index)
                .is_some_and(|layer_id| {
                    attached.contains(&crate::visual_rack::VisualScopeId::Layer(*layer_id))
                })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExportRenderedMotionField {
    scope: crate::visual_rack::VisualScopeId,
    source_scope: crate::visual_rack::VisualScopeId,
    origin: crate::motion::MotionFieldOrigin,
    source_generation: Option<u64>,
    frame_ordinal: Option<u64>,
    product_content_sha256: Option<[u8; 32]>,
}

/// Read accepted field truth only after `commit_frame_history`. This is the
/// publication boundary shared by field/gate parity, temporal history, and
/// the CPU identity stored beside the committed GPU slot.
fn export_rendered_motion_fields(
    plan: &EvaluatedCompositionPlan,
    executor: Option<&CompositionGpuExecutor>,
) -> Vec<ExportRenderedMotionField> {
    let (EvaluatedCompositionPlan::Advanced(plan), Some(executor)) = (plan, executor) else {
        return Vec::new();
    };
    let Some(motion) = plan.motion().advanced() else {
        return Vec::new();
    };
    motion
        .scopes()
        .iter()
        .filter_map(|scope| {
            let metrics = executor.motion_metrics(scope.scope)?;
            if metrics.valid_fields > 0
                && metrics.field_origin != crate::motion::MotionFieldOrigin::None
            {
                Some(ExportRenderedMotionField {
                    scope: scope.scope,
                    source_scope: metrics.field_source_scope?,
                    origin: metrics.field_origin,
                    source_generation: metrics.field_source_generation,
                    frame_ordinal: metrics.field_frame_ordinal,
                    product_content_sha256: metrics.field_product_content_sha256,
                })
            } else {
                None
            }
        })
        .collect()
}

/// The saved, path-free export identity of one motion scope.
///
/// Extracted so the Field Collider's recipient is named by exactly the same
/// rule as every authored motion scope, rather than by a second inline copy
/// that could drift.
fn export_motion_scope_identity(
    scope: crate::visual_rack::VisualScopeId,
    graph: &ExportCreativeGraph,
) -> Option<ExportMotionScopeIdentity> {
    match scope {
        crate::visual_rack::VisualScopeId::Master => Some(ExportMotionScopeIdentity::Master),
        crate::visual_rack::VisualScopeId::Layer(layer_id) => {
            let saved_position = graph
                .layer_ids
                .iter()
                .position(|candidate| *candidate == layer_id)?;
            Some(ExportMotionScopeIdentity::Layer {
                saved_position: u32::try_from(saved_position).unwrap_or(u32::MAX),
                stable_id: layer_id.get(),
                source_tap_id: export_selective_layer_id(saved_position),
            })
        }
        crate::visual_rack::VisualScopeId::Group(_)
        | crate::visual_rack::VisualScopeId::Program => None,
    }
}

/// Record the one admitted Field Collider from the shared evaluated plan.
///
/// Live and export consume this same plan, so the section describes exactly
/// what rendered. It carries authored identity, topology, diagnostics, and the
/// byte-exact budget only — never a vector, a pair texel, or a codec record.
fn field_collider_sidecar(
    motion: &crate::evaluated_frame::evaluated_composition::AdvancedMotionPlan,
    graph: &ExportCreativeGraph,
) -> Option<FieldColliderSidecar> {
    use crate::evaluated_frame::evaluated_composition::MotionPlanDiagnostic;
    use crate::motion::{
        FieldColliderDiagnostic, FieldColliderMode, MotionBoundaryMode, MotionDonor,
    };

    let plan = motion.collider()?;
    let recipient_scope = motion.scope(plan.recipient_scope)?;
    let recipient =
        sidecar_scope_identity(export_motion_scope_identity(plan.recipient_scope, graph)?);
    let authored = recipient_scope.params.collider;
    let resources = motion.collider_resources();
    let mut inputs = Vec::with_capacity(2);
    for (slot, donor, field_slot) in [
        ("a", authored.input_a, plan.input_a_slot),
        ("b", authored.input_b, plan.input_b_slot),
    ] {
        let (resolved, saved_position, stable_id) = match donor {
            MotionDonor::None => (false, None, None),
            MotionDonor::Selected {
                layer_id,
                saved_position,
            } => (true, Some(saved_position.get()), Some(layer_id.get())),
            MotionDonor::Missing { saved_position } => (false, Some(saved_position.get()), None),
        };
        inputs.push(FieldColliderSidecarInput {
            slot,
            resolved,
            saved_position,
            stable_id,
            field_slot: resolved.then_some(field_slot),
        });
    }
    let diagnostic = motion.diagnostics().iter().find_map(|entry| match entry {
        MotionPlanDiagnostic::FieldCollider { diagnostic, .. }
            if *diagnostic != FieldColliderDiagnostic::None =>
        {
            Some(diagnostic.to_string())
        }
        _ => None,
    });
    Some(FieldColliderSidecar {
        algorithm_version: plan.algorithm_version,
        recipient,
        admitted: recipient_scope.collider_admitted,
        mode: match plan.mode {
            FieldColliderMode::Sum => "sum",
            FieldColliderMode::Difference => "difference",
            FieldColliderMode::Curl => "curl",
            FieldColliderMode::Projection => "projection",
            FieldColliderMode::CollisionBoundary => "collision_boundary",
        },
        boundary: match plan.boundary {
            MotionBoundaryMode::Transparent => "transparent",
            MotionBoundaryMode::Mirror => "mirror",
            MotionBoundaryMode::Wrap => "wrap",
            MotionBoundaryMode::Hold => "hold",
        },
        output_slot: plan.output_slot,
        output_grid: [plan.output_grid.width, plan.output_grid.height],
        inputs,
        diagnostic,
        bytes_per_cell: resources.bytes_per_cell(),
        derived_vector_bytes: resources.derived_vector_bytes,
        derived_gate_bytes: resources.derived_gate_bytes,
        transient_pair_bytes: resources.transient_pair_bytes,
        total_bytes: resources.total_bytes,
        low_resolution_passes: resources.low_resolution_passes,
        nearest_lookups: resources.nearest_lookups,
        max_sampled_textures_in_pass: resources.max_sampled_textures_in_pass,
    })
}

fn export_motion_metadata_for_frame(
    plan: &EvaluatedCompositionPlan,
    graph: &ExportCreativeGraph,
    layers: &[ExportLayer],
    rendered_fields: &[ExportRenderedMotionField],
    frame_num: u64,
) -> ExportMotionMetadata {
    export_motion_metadata_for_frame_from(
        plan,
        graph,
        layers
            .iter()
            .map(|layer| (layer.source_index, layer.codec_motion.as_ref())),
        rendered_fields.iter().copied(),
        frame_num,
    )
}

fn export_motion_metadata_for_frame_from<'a>(
    plan: &EvaluatedCompositionPlan,
    graph: &ExportCreativeGraph,
    sources: impl IntoIterator<Item = (usize, Option<&'a CodecMotionProduct>)>,
    rendered_fields: impl IntoIterator<Item = ExportRenderedMotionField>,
    frame_num: u64,
) -> ExportMotionMetadata {
    let mut metadata = ExportMotionMetadata {
        accepted_frame: Some(frame_num),
        ..ExportMotionMetadata::default()
    };
    let EvaluatedCompositionPlan::Advanced(plan) = plan else {
        return metadata;
    };
    let Some(motion) = plan.motion().advanced() else {
        return metadata;
    };
    metadata.field_collider = field_collider_sidecar(motion, graph);
    metadata.scopes_truncated = motion.scopes().len() > MAX_EXPORT_MOTION_SIDECAR_SCOPES;
    metadata
        .scopes
        .reserve(motion.scopes().len().min(MAX_EXPORT_MOTION_SIDECAR_SCOPES));
    let sources = sources
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    let rendered_fields = rendered_fields
        .into_iter()
        .map(|field| (field.scope, field))
        .collect::<std::collections::BTreeMap<_, _>>();
    for scope in motion
        .scopes()
        .iter()
        .take(MAX_EXPORT_MOTION_SIDECAR_SCOPES)
    {
        let Some(identity) = export_motion_scope_identity(scope.scope, graph) else {
            continue;
        };
        let (donor_saved_position, donor_stable_id) = match scope.params.transplant.donor {
            crate::motion::MotionDonor::None => (None, None),
            crate::motion::MotionDonor::Selected {
                layer_id,
                saved_position,
            } => (Some(saved_position.get()), Some(layer_id.get())),
            crate::motion::MotionDonor::Missing { saved_position } => {
                (Some(saved_position.get()), None)
            }
        };
        let rendered = rendered_fields.get(&scope.scope).copied();
        let field_planned = scope.admitted_field_slot().is_some();
        let field_attached = rendered.is_some();
        let rendered_source_origin =
            rendered.map_or(crate::motion::MotionFieldOrigin::None, |field| field.origin);
        let codec = rendered
            .filter(|field| field.origin == crate::motion::MotionFieldOrigin::CodecVectors)
            .and_then(|field| {
                let source_position = match field.source_scope {
                    crate::visual_rack::VisualScopeId::Layer(layer_id) => graph
                        .layer_ids
                        .iter()
                        .position(|candidate| *candidate == layer_id),
                    crate::visual_rack::VisualScopeId::Master
                    | crate::visual_rack::VisualScopeId::Group(_)
                    | crate::visual_rack::VisualScopeId::Program => None,
                }?;
                let codec = sources.get(&source_position).copied().flatten()?;
                let identity = codec.exact_identity()?;
                (Some(codec.source_generation) == field.source_generation
                    && Some(codec.frame_ordinal) == field.frame_ordinal
                    && Some(identity.content_sha256) == field.product_content_sha256)
                    .then_some(codec)
            });
        metadata.scopes.push(ExportMotionScopeMetadata {
            scope: identity,
            algorithm_version: scope.params.algorithm_version,
            requested_source: scope.params.field_source,
            lattice_quality: scope.params.lattice_quality,
            source_origin: scope.source.origin,
            rendered_source_origin,
            field_planned,
            field_attached,
            source_diagnostic: scope.source.diagnostic,
            codec_provenance: codec.map(|codec| codec.provenance),
            source_generation: rendered.and_then(|field| field.source_generation),
            frame_ordinal: rendered.and_then(|field| field.frame_ordinal),
            codec_product_sha256: rendered.and_then(|field| field.product_content_sha256),
            codec_transition_count: codec
                .and_then(|codec| u8::try_from(codec.transition_count()).ok()),
            codec_elapsed_seconds: codec.and_then(CodecMotionProduct::elapsed_seconds),
            donor_saved_position,
            donor_stable_id,
            carrier: scope.params.transplant.carrier,
            transplant_admitted: scope.transplant_admitted,
            shutter_active: !scope.params.shutter.is_exact_zero(),
            shutter_angle_degrees: scope.params.shutter.angle_degrees,
            shutter_quality: scope.params.shutter.quality,
            shutter_sample_count: scope.params.shutter.quality.sample_count(),
        });
    }
    metadata
}

fn export_motion_plan_warnings(plan: &EvaluatedCompositionPlan) -> Vec<String> {
    use crate::evaluated_frame::evaluated_composition::{
        CompositionPlanDiagnostic, ImageTapConsumer, MotionPlanDiagnostic,
    };

    let EvaluatedCompositionPlan::Advanced(plan) = plan else {
        return Vec::new();
    };
    let mut warnings = plan
        .diagnostics()
        .iter()
        .filter_map(|diagnostic| match *diagnostic {
            CompositionPlanDiagnostic::RefreshGardenMatteNotSelected => Some(
                "Refresh Garden has no selected matte route; its Matte gate resolves to zero."
                    .to_owned(),
            ),
            CompositionPlanDiagnostic::MissingSelectedLayer {
                consumer: ImageTapConsumer::RefreshGardenMatte,
                saved_position,
            } => Some(format!(
                "Refresh Garden is missing saved matte route position {}; its Matte gate resolves to zero.",
                saved_position.get()
            )),
            CompositionPlanDiagnostic::MissingStableScope {
                consumer: ImageTapConsumer::RefreshGardenMatte,
                producer,
            } => Some(format!(
                "Refresh Garden matte producer {producer:?} is unavailable; its Matte gate resolves to zero."
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let Some(motion) = plan.motion().advanced() else {
        return warnings;
    };
    warnings.extend(motion.diagnostics().iter().map(|diagnostic| match *diagnostic {
        MotionPlanDiagnostic::Source {
            scope,
            diagnostic: crate::motion::MotionSourceDiagnostic::None,
        } => format!("Motion field {scope:?} resolved without a source diagnostic."),
        MotionPlanDiagnostic::Source {
            scope,
            diagnostic: crate::motion::MotionSourceDiagnostic::CodecUnavailable,
        } => format!(
            "Motion field {scope:?} requested explicit codec vectors, but exact adjacent-reference metadata was unavailable; no field was admitted."
        ),
        MotionPlanDiagnostic::Source {
            scope,
            diagnostic: crate::motion::MotionSourceDiagnostic::CodecUnavailableFallback,
        } => format!(
            "Motion field {scope:?} could not admit exact codec vectors and selected the deterministic lattice fallback."
        ),
            MotionPlanDiagnostic::MasterTransplantRejected => {
                "Master Faraday transplant has no layer recipient and was rendered inactive."
                    .to_owned()
            }
            MotionPlanDiagnostic::FieldCollider {
                recipient,
                diagnostic,
            } => format!(
                "Motion recipient layer {} Field Collider was not admitted: {diagnostic}. The exact M4 transplant recipe rendered instead.",
                recipient.get()
            ),
            MotionPlanDiagnostic::DonorNotSelected { recipient } => format!(
                "Motion recipient layer {} has no selected donor; Faraday transplant was rendered inactive.",
                recipient.get()
            ),
            MotionPlanDiagnostic::MissingDonor {
                recipient,
                saved_position,
            } => format!(
                "Motion recipient layer {} is missing saved donor position {}; Faraday transplant was rendered inactive.",
                recipient.get(),
                saved_position.get()
            ),
            MotionPlanDiagnostic::ExcessTransplantRejected {
                recipient,
                admitted_recipient,
            } => format!(
                "Motion recipient layer {} exceeded the single-carrier law; layer {} retained the admitted Faraday transplant.",
                recipient.get(),
                admitted_recipient.get()
            ),
            MotionPlanDiagnostic::RefreshGardenMotionNotSelected =>
                "Refresh Garden has no selected motion route; its motion gate resolves to zero."
                    .to_owned(),
            MotionPlanDiagnostic::MissingRefreshGardenMotion { saved_position } => format!(
                "Refresh Garden is missing saved motion route position {}; its motion gate resolves to zero.",
                saved_position.get()
            ),
            MotionPlanDiagnostic::RefreshGardenMotionUnavailable =>
                "Refresh Garden's selected motion route is unavailable; its motion gate resolves to zero."
                    .to_owned(),
            MotionPlanDiagnostic::SymmetryMotionNotSelected { scope, node_id, slot } => format!(
                "Symmetry Field node {} in {scope:?} arms motion slot {slot} with no selected donor; that slot binds neutral vector/gate views.",
                node_id.get()
            ),
            MotionPlanDiagnostic::MissingSymmetryMotion { scope, node_id, slot, saved_position } => format!(
                "Symmetry Field node {} in {scope:?} is missing saved motion position {} at slot {slot}; that slot binds neutral vector/gate views and never rebinds.",
                node_id.get(),
                saved_position.get()
            ),
            MotionPlanDiagnostic::SymmetryMotionUnavailable { scope, node_id, slot } => format!(
                "Symmetry Field node {} in {scope:?} selected a motion donor at slot {slot}, but no motion field was admitted; that slot binds neutral vector/gate views.",
                node_id.get()
            ),
        }));
    warnings
}

fn export_codec_frame_facts(
    layer: &ExportLayer,
    available_override: Option<bool>,
) -> crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts {
    layer.codec_motion.as_ref().map_or_else(
        crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts::default,
        |motion| crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts {
            available: available_override.unwrap_or_else(|| motion.codec_vectors_available()),
            source_generation: motion.source_generation,
            frame_ordinal: motion.frame_ordinal,
        },
    )
}

fn export_motion_held_scope(
    graph: &ExportCreativeGraph,
    source_index: usize,
    selection: FrameSelection,
) -> Option<crate::visual_rack::VisualScopeId> {
    if !selection.held {
        return None;
    }
    graph
        .layer_ids
        .get(source_index)
        .copied()
        .map(crate::visual_rack::VisualScopeId::Layer)
}

fn export_motion_layer_plan_inputs(
    graph: &ExportCreativeGraph,
    params: &[crate::motion::MotionParams],
    sources: impl IntoIterator<
        Item = (
            usize,
            crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts,
        ),
    >,
) -> Result<Vec<crate::evaluated_frame::evaluated_composition::LayerMotionPlanInput>, String> {
    let sources = sources.into_iter().collect::<Vec<_>>();
    if params.len() != sources.len() {
        return Err(format!(
            "export motion layer count {} does not match source count {}",
            params.len(),
            sources.len()
        ));
    }
    sources
        .into_iter()
        .zip(params.iter().copied())
        .map(|((source_index, codec), params)| {
            let stable_id = graph.layer_ids.get(source_index).copied().ok_or_else(|| {
                format!("export motion source {source_index} has no stable layer identity")
            })?;
            Ok(
                crate::evaluated_frame::evaluated_composition::LayerMotionPlanInput {
                    stable_id,
                    params,
                    codec,
                },
            )
        })
        .collect()
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "legacy motion-omitted adapter remains a supported exact-path test seam"
    )
)]
fn plan_export_composition(
    evaluated: &EvaluatedFramePlan,
    graph: &ExportCreativeGraph,
    layer_mattes: &[LayerMatte],
    program_history_initialized: bool,
    resource_limits: CreativeResourceLimits,
) -> Result<EvaluatedCompositionPlan, String> {
    plan_export_composition_inner(
        evaluated,
        graph,
        layer_mattes,
        program_history_initialized,
        resource_limits,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_export_composition_with_motion(
    evaluated: &EvaluatedFramePlan,
    graph: &ExportCreativeGraph,
    layer_mattes: &[LayerMatte],
    program_history_initialized: bool,
    resource_limits: CreativeResourceLimits,
    motion: ExportMotionPlanAdapter<'_>,
) -> Result<EvaluatedCompositionPlan, String> {
    plan_export_composition_inner(
        evaluated,
        graph,
        layer_mattes,
        program_history_initialized,
        resource_limits,
        Some(motion),
    )
}

fn plan_export_composition_inner(
    evaluated: &EvaluatedFramePlan,
    graph: &ExportCreativeGraph,
    layer_mattes: &[LayerMatte],
    program_history_initialized: bool,
    resource_limits: CreativeResourceLimits,
    motion: Option<ExportMotionPlanAdapter<'_>>,
) -> Result<EvaluatedCompositionPlan, String> {
    let planned_motion = motion
        .map(|motion| {
            let (master, effective_layers) = effective_export_motion_params(
                motion.master,
                motion.layers,
                motion.modulation,
                motion.shutter_samples,
            );
            export_motion_layer_plan_inputs(
                graph,
                &effective_layers,
                motion.sources.iter().enumerate().map(|(index, source)| {
                    (
                        source.source_index,
                        export_codec_frame_facts(
                            source,
                            motion
                                .codec_availability
                                .and_then(|availability| availability.get(index).copied()),
                        ),
                    )
                }),
            )
            .map(|layers| (master, layers))
        })
        .transpose()?;
    let mut input =
        CompositionPlanInput::new(&graph.composition, &graph.master_rack, &graph.layer_racks)
            .with_layer_mattes(layer_mattes, program_history_initialized)
            // Every offline job admits exactly one gesture canvas through
            // `export_gesture_canvas` before the first frame is planned, and a
            // refused admission aborts the job outright. The offline planner
            // therefore always sees an admitted canvas, which is what keeps a
            // routed donor planning identically live and offline.
            .with_gesture_canvas(true)
            // The offline tap surface exists for the whole job and frame
            // zero reads its defined-transparent contents, so admission is
            // unconditional — the same capability law as live. Content
            // readiness changes the executor binding, never route topology.
            .with_program_tap(true)
            .with_studies(&graph.studies)
            .with_authored_motion_modulation(
                motion.is_some_and(|motion| motion.authored_motion_modulation),
            );
    input.resource_limits = resource_limits;
    if let (Some(motion), Some((master, layers))) = (motion, planned_motion.as_ref()) {
        input = input.with_motion(*master, layers, motion.limits);
    }
    evaluated
        .plan_composition(input)
        .map_err(|error| format!("export creative frame plan rejected: {error}"))
}

/// Resolve a persisted, zero-based patch position into the deterministic ID
/// used by this export job. Keeping the mapping independent of the current
/// resource-vector order prevents failed media opens or future scheduling
/// reorder from silently retargeting an authored donor.
fn export_runtime_matte(
    saved: crate::image_routing::LayerMatteConfig,
    patch_layer_count: usize,
) -> LayerMatte {
    saved.to_runtime(|saved_position| {
        let source_index = saved_position.get() as usize;
        (source_index < patch_layer_count).then(|| {
            StableLayerId::new(export_selective_layer_id(source_index))
                .expect("export layer IDs are offset from zero")
        })
    })
}

/// Reproduce the live planner input from frame-local, post-morph/post-Mod
/// values. `layers` remains in UI order (top to bottom); the shared planner is
/// the sole authority that reverses contributing entries into compositor
/// order while preserving the curated blend law for every contribution.
#[allow(clippy::too_many_arguments)]
fn plan_export_selective_ntsc(
    _frame_num: u64,
    _fps: u32,
    _paused: bool,
    _evaluated: &EvaluatedFramePlan,
) -> Option<SelectiveNtscPlan> {
    // Retain the old helper seam while the selective implementation remains
    // available to historical tests. Runtime export deliberately never enters
    // it: one final-program VHS kernel is both topology-complete and cheaper.
    None
}

#[cfg(test)]
fn export_selective_transform_fingerprint(
    effects: &EffectUniforms,
    transform: &SpatialTransform,
    master: Option<&EffectUniforms>,
    master_transform: Option<&SpatialTransform>,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytemuck::bytes_of(effects)
        .iter()
        .chain(master.into_iter().flat_map(bytemuck::bytes_of))
    {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    for value in std::iter::once(transform.fingerprint())
        .chain(master_transform.map(|transform| transform.fingerprint()))
    {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn export_ntsc_reference_frame(frame_num: u64, fps: u32, paused: bool) -> usize {
    if paused {
        0
    } else {
        reference_frame_for_output(frame_num, fps)
    }
}

pub(crate) fn export_temporal_reference_tick(frame_num: u64, fps: u32) -> u64 {
    let fps = u64::from(fps.max(1));
    frame_num
        .saturating_mul(crate::effects::params::TEMPORAL_REFERENCE_FPS as u64)
        .saturating_add(fps / 2)
        / fps
}

/// Decode the recorded gesture performance this job will replay.
///
/// `GestureTrackDocument::decode` is the single acceptance path: it revalidates
/// the whole stream through the same validator that governs live ingest and
/// then re-derives the canonical checksum over the portable event stream and
/// compares it against the digest the recording declares. A mismatch is
/// therefore an actionable export error raised *before* the first frame is
/// rendered, never a silent re-render of a performance nobody authored.
fn export_recorded_gesture_track(
    document: Option<&crate::gesture::GestureTrackDocument>,
) -> Result<crate::gesture::GestureTrack, String> {
    let Some(document) = document else {
        // The pre-gesture path. Nothing is replayed and nothing is published.
        return Ok(crate::gesture::GestureTrack::default());
    };
    document
        .decode()
        .map_err(|error| format!("recorded gesture track rejected before rendering: {error}"))
}

/// Whether a validated performance take can ever author the explicit dry
/// partition during this job. Scan the bounded event/address tables once at
/// admission; waiting until the event's frame would create a partial export
/// even though the incompatibility was knowable before FFmpeg started.
fn performance_take_authors_temporal_bypass(
    take: &crate::performance_track::PerformanceTake,
) -> bool {
    use crate::performance_track::{PerformanceControl as Control, PerformanceRawValue as Raw};
    take.events().iter().any(|event| {
        let Some(address) = take.addresses().get(usize::from(event.address)) else {
            return false;
        };
        matches!(
            (&address.control, address.law.decode(event.value)),
            (
                Control::LayerParam { param, .. },
                Some(Raw::Toggle(true))
            ) if param == "bypass_temporal_fx"
        )
    })
}

fn export_replayed_layer_mosh_send(
    raw: &crate::performance_track::PerformanceRawValue,
) -> Option<f32> {
    match raw {
        crate::performance_track::PerformanceRawValue::Continuous(value) => {
            Some(crate::layers::clamp_layer_mosh_send(*value))
        }
        _ => None,
    }
}

/// Apply one replayed B9 performance event to the export job's authored
/// bases.
///
/// Every family routes through the same applier the live action arm uses —
/// `EffectsSnapshot`, `apply_spatial_transform_edit`, `apply_motion_param`,
/// `apply_temporal_wire_edit`, `BusMixerEdit`, `PatternSynthEdit` — on the
/// export's own copies of the same engine types, so an offline take mutates
/// state through identical code and identical clamps. Layer addresses resolve
/// by saved stack position, which *is* the export layer identity; a position
/// the job does not hold is a safe no-op exactly as a vanished stable ID is
/// live. The Collision Score loop driver never has an address here, so the
/// temporal applier's live-only context is always `None`.
#[allow(clippy::too_many_arguments)]
fn apply_export_performance_event(
    control: &crate::performance_track::PerformanceControl,
    raw: &crate::performance_track::PerformanceRawValue,
    master_effects: &mut EffectUniforms,
    master_transform: &mut SpatialTransform,
    master_motion: &mut crate::motion::MotionParams,
    layer_motion: &mut [crate::motion::MotionParams],
    ntsc: &mut crate::ntsc::NtscParams,
    temporal: &mut crate::effects::params::TemporalParams,
    gesture_canvas: &mut crate::gesture_canvas::GestureCanvasParams,
    creative_graph: &mut ExportCreativeGraph,
    layers: &mut [ExportLayer],
    morph: &mut Option<crate::morph::Morph>,
) {
    use crate::performance_track::{PerformanceControl as Control, PerformanceRawValue as Raw};
    let json = raw.to_json();
    let layer_index = |layers: &[ExportLayer], position: u32| -> Option<usize> {
        let position = usize::try_from(position).ok()?;
        layers
            .iter()
            .position(|layer| layer.source_index == position)
    };
    let apply_effects = |effects: &mut EffectUniforms, param: &str, value: &serde_json::Value| {
        let mut snapshot = crate::web::state::EffectsSnapshot::from_uniforms(effects);
        snapshot.apply_param(param, value);
        snapshot.apply_to_uniforms(effects);
    };
    match control {
        Control::Master { param } => {
            if let Some(value) = json {
                apply_effects(master_effects, param, &value);
            }
        }
        Control::MasterTransform { param } => {
            if let Some(value) = json {
                crate::App::apply_spatial_transform_edit(master_transform, param, &value);
            }
        }
        Control::LayerParam { layer, param } => {
            let Some(index) = layer_index(layers, *layer) else {
                return;
            };
            let target = &mut layers[index];
            match (param.as_str(), raw) {
                ("opacity", Raw::Continuous(value)) => {
                    target.opacity = crate::layers::clamp_layer_opacity(*value);
                }
                ("mosh_send", raw) => {
                    if let Some(value) = export_replayed_layer_mosh_send(raw) {
                        target.mosh_send = value;
                    }
                }
                ("speed", Raw::Continuous(value)) => {
                    target.speed = crate::layers::clamp_layer_speed(*value);
                }
                ("fps", Raw::Continuous(value)) => {
                    if value.is_finite() {
                        target.fps = crate::layers::clamp_layer_fps(*value);
                    }
                }
                ("blend_mode", Raw::Token(token)) => {
                    if let Some(mode) = BlendMode::from_key(token) {
                        target.blend_mode = mode;
                    }
                }
                ("bypass_master_fx", Raw::Toggle(value)) => target.bypass_master_fx = *value,
                ("bypass_temporal_fx", Raw::Toggle(value)) => {
                    target.bypass_temporal_fx = *value;
                }
                (key, _) if key.starts_with("key_") => {
                    if let Some(value) = json {
                        apply_effects(&mut target.effects, key, &value);
                    }
                }
                _ => {}
            }
        }
        Control::LayerEffect { layer, param } => {
            let Some(index) = layer_index(layers, *layer) else {
                return;
            };
            if let Some(value) = json {
                apply_effects(&mut layers[index].effects, param, &value);
                layers[index].effects.clear_master_only_effects();
            }
        }
        Control::LayerTransform { layer, param } => {
            let Some(index) = layer_index(layers, *layer) else {
                return;
            };
            if let Some(value) = json {
                crate::App::apply_spatial_transform_edit(
                    &mut layers[index].transform,
                    param,
                    &value,
                );
            }
        }
        Control::LayerVisible { layer } => {
            if let (Some(index), Raw::Toggle(visible)) = (layer_index(layers, *layer), raw) {
                layers[index].visible = *visible;
            }
        }
        Control::LayerPattern { layer, param } => {
            let Some(index) = layer_index(layers, *layer) else {
                return;
            };
            if let Some(value) = json {
                if let Some(edit) = crate::pattern_synth::PatternSynthEdit::parse(param, &value) {
                    if let Some(params) = layers[index].pattern.as_mut() {
                        edit.apply(params);
                    }
                }
            }
        }
        Control::Ntsc { param } => {
            if let Some(value) = json {
                ntsc.set_param(param, &value);
            }
        }
        Control::Temporal { param } => {
            if let Some(value) = json {
                crate::apply_temporal_wire_edit(temporal, param, &value, None);
            }
        }
        Control::MotionMaster { param } => {
            if let Some(value) = json {
                let _ = crate::apply_motion_param(master_motion, true, param, &value);
            }
        }
        Control::MotionLayer { layer, param } => {
            let Some(index) = layer_index(layers, *layer) else {
                return;
            };
            if let Some(value) = json {
                let _ = crate::apply_motion_param(&mut layer_motion[index], false, param, &value);
            }
        }
        Control::BusCrossfade => {
            if let Raw::Continuous(value) = raw {
                creative_graph.composition.set_bus_crossfade(*value);
            }
        }
        Control::BusMix { param } => {
            if let Some(value) = json {
                if let Some(edit) = crate::mixing_boundary::BusMixerEdit::parse(param, &value) {
                    let mut mixer = creative_graph.composition.mixer();
                    edit.apply(&mut mixer);
                    creative_graph.composition.set_mixer(mixer);
                }
            }
        }
        Control::MorphPosition => {
            if let (Some(morph), Raw::Continuous(value)) = (morph.as_mut(), raw) {
                morph.set_position(*value);
            }
        }
        Control::GestureCanvas { param } => {
            if let Raw::Continuous(value) = raw {
                match param.as_str() {
                    "radius" => gesture_canvas.radius = *value,
                    "strength" => gesture_canvas.strength = *value,
                    "retention" => gesture_canvas.retention = *value,
                    _ => {}
                }
            }
        }
    }
}

/// Admit and build the one gesture canvas an offline job owns.
///
/// The grid is derived from the export output through the same host law the
/// live preview uses, so the two sessions etch the same lattice, and the
/// authored controls come from the patch. Every frozen limit is checked by
/// `GestureCanvasPlan::preflight` before a cell is allocated.
fn export_gesture_canvas(
    patch: &PatchState,
    width: u32,
    height: u32,
    limits: crate::gesture_canvas::GestureCanvasLimits,
) -> Result<crate::gesture_canvas::GestureCanvasState, String> {
    let grid = crate::gesture_canvas_host_grid(width, height);
    let request = crate::gesture_canvas::GestureCanvasRequest::new(grid)
        .with_decay_ticks(crate::gesture::MAX_GESTURE_DECAY_TICKS);
    crate::gesture_canvas::GestureCanvasPlan::preflight(&[request], limits)
        .map_err(|error| format!("export gesture canvas rejected: {error}"))?;
    let params = patch.gesture_canvas.unwrap_or_default().to_params();
    crate::gesture_canvas::GestureCanvasState::new(grid, params)
        .map_err(|error| format!("export gesture canvas rejected: {error}"))
}

/// Stage one offline gesture-canvas frame.
///
/// The 30 Hz address is `export_temporal_reference_tick(frame_num, fps)` — the
/// same rounded rational map the recorded temporal event track replays on and
/// the same address the live accepted-frame accumulator produces. Wall time
/// never reaches the replay cursor, the decay clock, or the field. A frozen
/// program holds rather than accumulating catch-up debt, exactly as live.
fn stage_export_gesture_canvas_frame(
    canvas: &mut crate::gesture_canvas::GestureCanvasState,
    replay: &mut crate::gesture::GestureReplay<'_>,
    frame_num: u64,
    fps: u32,
    program_advances: bool,
    evaluated_params: crate::gesture_canvas::GestureCanvasParams,
) -> Result<crate::gesture_canvas::GestureCanvasFramePlan, String> {
    // An abandoned frame is a discarded frame. Offline every early exit breaks
    // out of the render loop, but discarding first keeps the transaction law
    // total rather than relying on that.
    canvas.discard_staged();
    let reference_tick = export_temporal_reference_tick(frame_num, fps);
    let events = replay.events_due(u32::try_from(reference_tick).unwrap_or(u32::MAX));
    canvas
        .stage_frame(crate::gesture_canvas::GestureCanvasFrameInput {
            reference_tick,
            program_advances,
            events,
            evaluated_params: Some(evaluated_params),
        })
        .map_err(|error| format!("export gesture canvas frame refused: {error}"))
}

fn export_selective_topology_signature(plan: &SelectiveNtscPlan) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    };
    mix(plan.layers.len() as u64);
    for layer in &plan.layers {
        mix(layer.layer_id);
        mix(layer.bypass_master_fx as u64);
    }
    hash
}

fn run_export(
    patch: &PatchState,
    config: &ExportConfig,
    progress: &Arc<ExportProgress>,
    library_folder: &str,
    shared_gpu: Option<(wgpu::Device, wgpu::Queue)>,
) -> Result<(), String> {
    check_cancelled(progress)?;
    // Validate protocol-controlled sizes before creating a device or any of
    // the many full-frame GPU textures used by the compositor.
    validate_export_dimensions(config.width, config.height)?;
    if !config.width.is_multiple_of(2) || !config.height.is_multiple_of(2) {
        return Err("export dimensions must be even for yuv420p output".to_string());
    }
    if config.fps == 0 || config.fps > 240 {
        return Err("export FPS must be between 1 and 240".to_string());
    }
    if !config.duration_secs.is_finite()
        || config.duration_secs <= 0.0
        || config.duration_secs > 3600.0
    {
        return Err("export duration must be greater than 0 and at most 3600 seconds".to_string());
    }
    // Render enough complete CFR frames to cover the requested duration, then
    // let ffmpeg trim the mux to that exact duration. Rounding could otherwise
    // leave the video up to half a frame short.
    let total_frames = export_frame_count(config.fps, config.duration_secs);
    if total_frames == 0 {
        return Err("export duration is shorter than one frame".to_string());
    }

    // Reuse the live renderer device when available. Standalone audit/tests
    // still create an isolated headless device.
    let (device, queue) = if let Some(gpu) = shared_gpu {
        gpu
    } else {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("No GPU adapter found: {e}"))?;
        check_cancelled(progress)?;

        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Export Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("Failed to create export device: {e}"))?
    };
    check_cancelled(progress)?;

    let w = config.width;
    let h = config.height;
    let raw_device_limits = device.limits();
    let max_dimension = raw_device_limits.max_texture_dimension_2d;
    let creative_resource_limits = CreativeResourceLimits {
        max_texture_dimension_2d: max_dimension,
        max_texture_array_layers: raw_device_limits.max_texture_array_layers,
        max_sampled_textures_per_shader_stage: raw_device_limits
            .max_sampled_textures_per_shader_stage,
        min_uniform_buffer_offset_alignment: raw_device_limits.min_uniform_buffer_offset_alignment,
        ..CreativeResourceLimits::default()
    };
    let media_device_limits =
        MediaDeviceLimits::new(max_dimension, raw_device_limits.max_buffer_size);
    let motion_device_limits =
        crate::motion::MotionDeviceLimits::new(max_dimension, raw_device_limits.max_buffer_size);
    if w > max_dimension || h > max_dimension {
        return Err(format!(
            "export dimensions {w}x{h} exceed this GPU's {max_dimension}px texture limit"
        ));
    }

    // --- Build pipelines and mandatory output-sized resources ---
    // Safe admission constrains dimensions but cannot know free VRAM. Scope
    // the complete setup so UHD composite/history/readback allocations and
    // pipeline validation failures return through the worker's normal error
    // path instead of reaching the process-level uncaptured-error callback.
    let setup_validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let setup_internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let setup_out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    // --- Build pipelines (same as renderer/state.rs) ---
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let effects_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Effects Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let effects_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Effects Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fullscreen.wgsl").into()),
    });

    let effects_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Effects Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/effects.wgsl").into()),
    });

    let effects_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Export Effects PL"),
        bind_group_layouts: &[
            Some(&effects_bind_group_layout),
            Some(&effects_uniform_layout),
        ],
        immediate_size: 0,
    });

    let effects_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export Effects Pipeline"),
        layout: Some(&effects_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &effects_fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Composite pipeline
    let composite_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Composite BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let composite_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Composite Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let composite_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Composite Fragment"),
        source: wgpu::ShaderSource::Wgsl(composite_shader_source()),
    });

    let composite_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Export Composite PL"),
            bind_group_layouts: &[
                Some(&composite_bind_group_layout),
                Some(&composite_uniform_layout),
            ],
            immediate_size: 0,
        });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export Composite Pipeline"),
        layout: Some(&composite_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let matte_composite = MatteCompositePipeline::build(&device, &vertex_shader);
    let mut image_routing_gpu: Option<ImageRoutingGpuResources> = None;
    // Exact legacy export never constructs this value. Its RGBA16F arena and
    // histories are admitted lazily only after the immutable planner selects
    // Advanced, preserving the established legacy allocation/render path.
    let mut composition_gpu: Option<CompositionGpuExecutor> = None;
    // B7 pattern-synth executor, lazily constructed on the first pattern
    // layer's first frame — modulation can wake nothing here (presence is
    // authored), so a job with no pattern layer charges nothing.
    let mut pattern_synth_gpu: Option<crate::renderer::pattern_synth::PatternSynthGpu> = None;

    // --- Composite textures (same 3-texture scheme as live renderer) ---
    let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST;

    let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Export Composite {i}")),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        })
    });

    let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|i| {
        composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default())
    });
    // The B16 programme re-entry tap: one retained copy of final slot 2,
    // published at the acceptance decision exactly as live publishes it. wgpu
    // zero initialization keeps frame zero's routed taps defined transparent,
    // pixel-identical to live's unbound pre-first-commit fallback.
    let program_tap_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export program re-entry tap"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let program_tap_view = program_tap_texture.create_view(&wgpu::TextureViewDescriptor::default());
    // Capability is job-lifetime, but readiness begins cold exactly as live.
    // This distinction matters for every routed rack kind, not only Displace:
    // an inverted mask or Symmetry fallback can distinguish a valid transparent
    // donor from an unavailable one even when their sampled texels are equal.
    let mut program_tap_valid = false;
    // Codec Mosh is a synchronous CPU replacement. When a top prefix bypasses
    // Temporal, retain the pre-mosh wet+dry programme transactionally until
    // ffmpeg accepts the corresponding moshed-wet+dry frame. The surface stays
    // lazy so every pre-bypass export keeps its established allocation set.
    let mut temporal_bypass_program_candidate: Option<wgpu::Texture> = None;
    let (opaque_output_pipeline, opaque_output_bind_group_layout) =
        crate::renderer::state::build_opaque_output_pipeline(&device);
    let opaque_output_bind_group = crate::renderer::state::build_opaque_output_bind_group(
        &device,
        &opaque_output_bind_group_layout,
        &composite_views[0],
        &sampler,
    );
    // The B4 display stage rides the same slot-0 seam offline. Pipelines are
    // cheap; its surfaces stay lazy, so a patch that never arms the stage
    // (and is never modulated awake) allocates nothing here either.
    let mut display_physics_gpu = crate::renderer::display_physics::DisplayPhysicsGpu::new(
        &device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        [w, h],
    );
    // The B8 melting edge rides the same seam immediately upstream of the
    // display stage; its history surface is equally lazy.
    let mut melting_edge_gpu = crate::renderer::melting_edge::MeltingEdgeGpu::new(
        &device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        [w, h],
    );
    // The B14 sync latch rides the same seam between the melting edge and
    // the display stage. It owns no texture: a bounded per-line table and one
    // uniform buffer are its entire state.
    let mut sync_latch_gpu = crate::renderer::sync_latch::SyncLatchGpu::new(
        &device,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        [w, h],
    );

    // Temporal history is the largest mandatory output-sized allocation. It
    // belongs to the setup scope above even when a saved patch currently has
    // temporal amounts at zero because morph/modulation can activate them.
    let (temporal_pipeline, temporal_bgl, temporal_ubl) =
        crate::renderer::state::build_temporal_pipeline(&device);
    let (history_texture, history_view) =
        crate::renderer::state::build_history_texture(&device, w, h);
    let (feedback_texture, feedback_view) =
        crate::renderer::state::build_feedback_texture(&device, w, h);
    let temporal_prepared = crate::renderer::state::build_prepared_temporal_gpu_resources(
        &device,
        &temporal_bgl,
        &temporal_ubl,
        &composite_views[0],
        &history_view,
        &sampler,
        &feedback_view,
    );

    // --- Readback staging buffer ---
    let bytes_per_row = (w * 4 + 255) & !255;
    let buffer_size = (bytes_per_row * h) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let setup_errors = [
        pollster::block_on(setup_out_of_memory.pop()),
        pollster::block_on(setup_internal.pop()),
        pollster::block_on(setup_validation.pop()),
    ]
    .into_iter()
    .flatten()
    .map(|error| error.to_string())
    .collect::<Vec<_>>();
    if let Some(error) = export_gpu_setup_error(w, h, &setup_errors) {
        return Err(error);
    }

    // Precompile the optional layer-send kernels outside both the mandatory
    // setup scopes and the per-frame loop. The Result is consulted only when
    // an active partial send needs the feature, so a default export retains
    // its historical compatibility surface.
    let mosh_influence_pipelines = crate::renderer::state::MoshInfluencePipelines::build(&device);

    // Retained only for the dormant compatibility branch below. Final-program
    // VHS never allocates this per-layer staging buffer.
    let mut selective_staging: Option<wgpu::Buffer> = None;

    // --- Open video decoders for each layer ---
    let source_context = crate::media_source::ResolveContext::new(
        None,
        (!library_folder.is_empty()).then(|| std::path::PathBuf::from(library_folder)),
    );
    let mut source_fingerprints = crate::media_source::FingerprintSession::with_cancel(
        crate::media_source::FingerprintLimits::default(),
        Some(progress.cancel.clone()),
    )
    .map_err(|error| error.to_string())?;
    let mut layers: Vec<ExportLayer> = Vec::new();
    for (source_index, layer_cfg) in patch.layers.iter().enumerate() {
        check_cancelled(progress)?;
        let (clip_filename, clip_source_path) = active_export_clip_source(layer_cfg);
        let resolved = resolve_export_visual_source(
            clip_source_path,
            clip_filename,
            config
                .layer_source_hints
                .get(source_index)
                .map(String::as_str),
            &source_context,
            &mut source_fingerprints,
        );
        let resolved_file = match resolved {
            Ok(crate::media_source::ResolvedVisualSource::Spout { .. }) => {
                progress.record_warning(black_substitution_warning(
                    source_index,
                    clip_filename,
                    "is a live Spout source unavailable to offline export",
                ));
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                )?);
                continue;
            }
            Ok(crate::media_source::ResolvedVisualSource::PatternSynth) => {
                layers.push(pattern_export_layer(&device, source_index, layer_cfg)?);
                continue;
            }
            Ok(crate::media_source::ResolvedVisualSource::TextPage) => {
                layers.push(text_page_export_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                    &config.media_safety_policy,
                    media_device_limits,
                )?);
                continue;
            }
            Ok(crate::media_source::ResolvedVisualSource::File(resolved)) => resolved,
            Err(error) if strict_content_addressed_export_source(clip_source_path) => {
                return Err(format!(
                    "content-addressed export layer '{}' could not be resolved: {error}",
                    clip_filename
                ));
            }
            Err(error) => {
                progress.record_warning(black_substitution_warning(
                    source_index,
                    clip_filename,
                    &format!("could not be resolved ({error})"),
                ));
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                )?);
                continue;
            }
        };
        let source_fingerprint = match resolved_file.identity {
            Some(identity) => Some(identity),
            None => match source_fingerprints.fingerprint(&resolved_file.path) {
                Ok(identity) => Some(identity),
                Err(error) => {
                    if progress.cancel.load(Ordering::Acquire) {
                        return Err(
                            "export cancelled while fingerprinting a visual source".to_string()
                        );
                    }
                    progress.record_warning(format!(
                        "Layer {} ('{}') source fingerprint is unavailable ({error}); the motion sidecar records that omission.",
                        source_index + 1,
                        clip_filename
                    ));
                    None
                }
            },
        };
        let path = resolved_file.path;
        let path_text = path.to_string_lossy();
        if crate::layers::is_still_image_file(&path) {
            match crate::video::decode_still_image_with_media_policy(
                &path,
                &config.media_safety_policy,
                media_device_limits,
            ) {
                Ok(decoded) => {
                    layers.push(still_export_layer(
                        &device,
                        &queue,
                        source_index,
                        layer_cfg,
                        decoded,
                        source_fingerprint.clone(),
                    )?);
                    continue;
                }
                Err(error) => {
                    progress.record_warning(black_substitution_warning(
                        source_index,
                        clip_filename,
                        &format!("could not open as a still image ({error})"),
                    ));
                    layers.push(black_placeholder_layer(
                        &device,
                        &queue,
                        source_index,
                        layer_cfg,
                    )?);
                    continue;
                }
            }
        }
        let mut decoder = match VideoDecoder::open_with_cancel_and_media_policy(
            &path_text,
            progress.cancel.clone(),
            &config.media_safety_policy,
            media_device_limits,
        ) {
            Ok(d) => d,
            Err(e) => {
                progress.record_warning(black_substitution_warning(
                    source_index,
                    clip_filename,
                    &format!("could not open as video ({e})"),
                ));
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                )?);
                continue;
            }
        };

        let lw = decoder.width;
        let lh = decoder.height;

        check_cancelled(progress)?;

        let (texture, texture_view) =
            create_export_source_texture(&device, lw, lh, "Export Layer Tex")?;

        // Build the bounded source-wide index once on the export worker. Every
        // later presentation is an absolute seek from a known preceding
        // keyframe (or the bounded reverse cache), never an EOF reopen or a
        // render-thread scan from frame zero.
        if let Err(error) = decoder.build_keyframe_index() {
            if progress.cancel.load(Ordering::Acquire) {
                return Err("export cancelled while indexing a video source".to_string());
            }
            progress.record_warning(format!(
                "Layer {} ('{}') could not build its bounded keyframe index ({error}); using the decoder's deterministic fallback index.",
                source_index + 1,
                clip_filename
            ));
        }
        check_cancelled(progress)?;

        let duration_seconds = decoder.duration_seconds();
        let source_frame_count = estimated_export_source_frames(duration_seconds, decoder.fps);
        let mut transport =
            ExportClipTransport::from_layer_config(layer_cfg, duration_seconds, source_frame_count);
        let fps = export_transport_fps(&transport, decoder.fps);
        let seed_selection = transport.seed_selection();

        // Seed every layer texture with the exact saved-playhead selection
        // before frame zero. Paused/Frozen exports therefore hold the authored
        // frame rather than silently falling back to source frame zero.
        let first_frame = match decoder.seek_decode_after_for_generation(
            seed_selection.source_seconds,
            seed_selection.generation,
            None,
        ) {
            Ok(frame) if frame.metadata.source_generation == seed_selection.generation => frame,
            Ok(frame) => {
                progress.record_warning(black_substitution_warning(
                    source_index,
                    clip_filename,
                    &format!(
                        "returned stale generation {} while seeding requested generation {}",
                        frame.metadata.source_generation, seed_selection.generation
                    ),
                ));
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                )?);
                continue;
            }
            Err(error) => {
                progress.record_warning(black_substitution_warning(
                    source_index,
                    clip_filename,
                    &format!("could not decode its saved playhead ({error})"),
                ));
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                )?);
                continue;
            }
        };
        check_cancelled(progress)?;
        upload_export_texture_checked(
            &device,
            &queue,
            &texture,
            &first_frame.rgba,
            lw,
            lh,
            "Export Video Layer",
        )?;
        let codec_motion = matching_export_codec_motion(
            first_frame.codec_motion,
            first_frame.metadata.source_generation,
            [lw, lh],
        );

        let mut effects = EffectUniforms {
            resolution: [lw as f32, lh as f32],
            ..Default::default()
        };
        layer_cfg.effects.apply_to_uniforms(&mut effects);
        effects.clear_master_only_effects();

        layers.push(ExportLayer {
            source_index,
            motion_source: ExportMotionSourceRecord {
                kind: ExportMotionSourceKind::Video,
                logical_name: clip_filename.to_owned(),
                persisted_reference: clip_source_path.to_owned(),
                fingerprint: source_fingerprint,
            },
            // Offline loop authority is the pure timeline's boundary count;
            // absolute seeks deliberately do not consult decoder EOF state.
            consumed_loop_generation: 0,
            decoder: Some(decoder),
            codec_motion,
            codec_motion_predecessor: first_frame.metadata.codec_identity.filter(|identity| {
                identity.source_generation == first_frame.metadata.source_generation
            }),
            _still_source: None,
            texture,
            texture_view,
            effects,
            transform: layer_cfg.transform.sanitized(),
            opacity: layer_cfg.opacity,
            mosh_send: crate::layers::clamp_layer_mosh_send(layer_cfg.mosh_send),
            blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
            bypass_master_fx: layer_cfg.bypass_master_fx,
            bypass_temporal_fx: layer_cfg.bypass_temporal_fx,
            matte: LayerMatte::default(),
            reroll_on_loop: layer_cfg.reroll_on_loop,
            transport,
            speed: transport.authored.rate as f32,
            visible: layer_cfg.visible,
            paused: layer_cfg.paused,
            fps,
            width: lw,
            height: lh,
            pattern: None,
        });
    }

    // Saved matte donors are positional by design. Export assigns each patch
    // position the same deterministic nonzero ID used by SourceTap, so failed
    // media opens (which become black placeholders) never retarget a route.
    for layer in &mut layers {
        let layer_cfg = &patch.layers[layer.source_index];
        layer.matte = export_runtime_matte(layer_cfg.matte, patch.layers.len());
    }

    // Resolve saved composition/rack positions once against deterministic
    // export IDs. Missing media still owns its black placeholder ID, so a
    // failed decoder can never retarget a selected-layer image route.
    let mut base_creative_graph = resolve_export_creative_graph(patch)?;

    // --- Master effects ---
    let mut master_effects = EffectUniforms {
        resolution: [w as f32, h as f32],
        ..Default::default()
    };
    patch.master.apply_to_uniforms(&mut master_effects);
    let mut master_transform = patch.master_transform.sanitized();
    let mut base_master_motion =
        resolved_export_motion(patch.master_motion, &base_creative_graph.layer_ids);
    let mut base_layer_motion = layers
        .iter()
        .map(|layer| {
            resolved_export_motion(
                patch.layers[layer.source_index].motion,
                &base_creative_graph.layer_ids,
            )
        })
        .collect::<Vec<_>>();
    let mut motion_sidecar = ExportMotionSidecarAccumulator::new(
        config,
        &base_creative_graph,
        &layers,
        base_master_motion,
        &base_layer_motion,
    );

    // --- NTSC state ---
    let mut ntsc_state = NtscState::new();
    if let Some(ref ntsc_cfg) = patch.ntsc {
        ntsc_state.params = ntsc_cfg.to_params();
    }
    let mut base_ntsc = ntsc_state.params.clone();
    // Live global and selective workers own independent ntsc-rs state. Keep
    // those processors distinct offline as well, especially when a morph
    // crosses the discrete bypass boundary during one export.
    let mut selective_ntsc_state = NtscState::new();

    // --- B5 codec mosh (synchronous round trip, one kernel, this owner) ---
    // The engine opens lazily on the first active frame, because modulation
    // or a morph can wake a dormant authored stage mid-job. `threads = 1`
    // makes two renders on this host byte-identical; a missing mpeg4 pair is
    // an actionable export error, never a silent bypass.
    let mut mosh_engine: Option<crate::codec_mosh::MoshEngine> = None;
    let mut mosh_influence_gpu: Option<crate::renderer::state::MoshInfluenceGpu> = None;
    let mut mosh_interval_active = false;
    let mut mosh_used = false;
    let mut mosh_observed: Option<CodecMoshObservationAccumulator> = None;
    let mut mosh_layer_send_observed = vec![CodecMoshLayerSendObservation::default(); layers.len()];

    // --- Modulation matrix (deterministic: beat derived from frame index) ---
    // Imported audio is sampled from frame-indexed program time. Live capture,
    // MIDI, and other hardware sources read 0 offline; LFO motion renders
    // identically for the same patch every time.
    let mut mod_matrix = crate::modulation::ModMatrix::new();
    if let Some(ref mod_cfg) = patch.modulation {
        mod_cfg.apply_to_matrix_with_composition(
            &mut mod_matrix,
            &base_creative_graph.address_book,
            &base_creative_graph.layer_ids,
            &base_creative_graph.composition,
        );
    }
    let authored_motion_modulation = mod_matrix.has_authored_motion_routing();
    let patch_or_morph_authors_temporal_bypass =
        patch.layers.iter().any(|layer| layer.bypass_temporal_fx)
            || patch.morph.as_ref().is_some_and(|snapshot| {
                [snapshot.a.as_ref(), snapshot.b.as_ref()]
                    .into_iter()
                    .flatten()
                    .flat_map(|slot| slot.layers.iter())
                    .any(|layer| layer.bypass_temporal_fx == Some(true))
            });
    let analysis_clip = if mod_matrix.audio_enabled
        && mod_matrix.audio_source_kind == crate::modulation::AUDIO_SOURCE_FILE
    {
        let requested = mod_matrix.audio_clip_path.clone();
        let logical_name = if crate::media_source::parse_content_reference(&requested)
            .map_err(|error| error.to_string())?
            .is_some()
        {
            String::new()
        } else {
            std::path::Path::new(&requested)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| requested.clone())
        };
        let resolved = resolve_export_file_source(
            &requested,
            &logical_name,
            config.analysis_audio_path_hint.as_deref(),
            &source_context,
            |path: &std::path::Path| crate::audio::is_supported_audio_file(path),
            &mut source_fingerprints,
        )
        .map_err(|error| format!("deterministic audio-analysis clip: {error}"))?;
        Some(
            crate::audio::AudioClip::open(&resolved.path).map_err(|error| {
                format!(
                    "cannot load deterministic audio-analysis clip {}: {error}",
                    resolved.path.display()
                )
            })?,
        )
    } else {
        None
    };
    let mut export_morph = patch.morph.clone().map(crate::morph::Morph::from_snapshot);

    // --- Temporal effects (feedback/slit-scan), same pass as live ---
    let mut base_temporal =
        resolved_export_patch_temporal(patch.temporal.as_ref(), &base_creative_graph.layer_ids);
    let mut temporal_state = crate::renderer::state::TemporalState::default();
    let mut previous_temporal_bypass_partition = Vec::with_capacity(layers.len());
    let mut temporal_bypass_partition_initialized = false;
    let mut temporal_boundaries = crate::performance::BeatBoundaryTracker::default();
    let mut temporal_audio_onsets = crate::temporal::TemporalAudioOnsetTracker::default();
    let recorded_temporal_track = config.temporal_event_track.clone();
    let mut recorded_temporal_replay = recorded_temporal_track.replay();
    if recorded_temporal_track.truncated() {
        progress.record_warning(
            "The live temporal event track reached its bounded cap; export replays the retained prefix only.",
        );
    }

    // --- Recorded gesture performance, same 30 Hz timeline as Temporal ---
    // The checksum is verified here, before the first frame is rendered: a
    // tampered or corrupted recording aborts the job with a named error rather
    // than silently re-rendering a performance nobody authored.
    let recorded_gesture_track = export_recorded_gesture_track(config.gesture_track.as_ref())?;
    let mut recorded_gesture_replay = recorded_gesture_track.replay();
    if recorded_gesture_track.truncated() {
        progress.record_warning(
            "The live gesture track reached its bounded cap; export replays the retained prefix only.",
        );
    }
    if !recorded_gesture_track.is_complete() {
        progress.record_warning(
            "The recorded gesture track has unclosed strokes; export replays it exactly as recorded and never closes them.",
        );
    }
    // --- B10 video content analysis, offline half ---
    // The same CPU law the live host runs on its readbacks, fed from the
    // export's own frame bytes at the same reference cadence.
    let mut video_analysis_state = crate::modulation::VideoAnalysisState::default();
    let mut video_analysis_accumulator = 0.0f64;
    let mut video_analysis_primed = false;

    // --- B9 recorded performance take, replayed by reference tick ---
    // The checksum is verified before the first frame renders, exactly as the
    // gesture track's is; export replays the take once, straight through — the
    // loop flag is live playback transport, not part of the take.
    let mut performance_replay = match config.performance_take.as_ref() {
        Some(document) => {
            let take = document
                .decode()
                .map_err(|error| format!("recorded performance take rejected: {error}"))?;
            if take.truncated() {
                progress.record_warning(
                    "The recorded performance take reached its bounded cap; export replays the retained prefix only.",
                );
            }
            if take.incomplete() {
                progress.record_warning(
                    "The performance take was captured while recording was still armed; export replays it exactly as recorded.",
                );
            }
            Some((take, crate::performance_track::PerformanceCursor::default()))
        }
        None => None,
    };
    let performance_authors_temporal_bypass = performance_replay
        .as_ref()
        .is_some_and(|(take, _)| performance_take_authors_temporal_bypass(take));
    mod_matrix
        .validate_temporal_bypass_motion_routing(
            patch_or_morph_authors_temporal_bypass || performance_authors_temporal_bypass,
        )
        .map_err(|error| format!("export Temporal-bypass admission failed: {error}"))?;
    let mut base_gesture_canvas = patch.gesture_canvas.unwrap_or_default().to_params();
    let gesture_canvas_limits = crate::gesture_canvas::GestureCanvasLimits::device(
        max_dimension,
        raw_device_limits.min_uniform_buffer_offset_alignment,
    );
    let mut gesture_canvas = export_gesture_canvas(patch, w, h, gesture_canvas_limits)?;
    // The device half of that same admitted canvas. It is built once, before
    // the first frame is planned, from the identical request the CPU reference
    // was admitted under, so the offline session binds exactly the surface the
    // live session binds and `encode_staged_frame` is the one shared seam.
    let mut gesture_canvas_gpu =
        match crate::renderer::gesture_canvas::GestureCanvasResources::prepare(
            &device,
            &queue,
            &[
                crate::gesture_canvas::GestureCanvasRequest::new(gesture_canvas.grid())
                    .with_decay_ticks(crate::gesture::MAX_GESTURE_DECAY_TICKS),
            ],
            gesture_canvas_limits,
        ) {
            Ok(resources) => Some(resources),
            Err(error) => {
                return Err(format!("export gesture canvas rejected: {error}"));
            }
        };
    if base_temporal.originals.score.enabled {
        match base_temporal.originals.score.trigger {
            crate::temporal::CollisionScoreTrigger::Manual
                if recorded_temporal_track.events().is_empty() =>
            {
                progress.record_warning(
                    "Collision Score manual mode has no accepted live event track; deterministic export supplies zero manual events.",
                );
            }
            crate::temporal::CollisionScoreTrigger::AudioOnset if analysis_clip.is_none() => {
                progress.record_warning(
                    "Collision Score audio-onset events require a deterministic imported analysis clip; export supplies zero live-input onset events.",
                );
            }
            crate::temporal::CollisionScoreTrigger::Boundary
                if !matches!(
                    base_temporal.originals.score.loop_driver,
                    crate::temporal::CollisionScoreLoopDriver::SelectedLayer { .. }
                ) =>
            {
                progress.record_warning(
                    "Collision Score loop driver is missing; export supplies zero boundary events.",
                );
            }
            _ => {}
        }
    }
    let mut previous_selective_frame: Option<bool> = None;
    let mut previous_selective_topology: Option<u64> = None;

    // --- Spawn ffmpeg ---
    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create export directory {}: {error}",
                parent.display()
            )
        })?;
    }

    let audio_path = match config.audio_path.as_deref() {
        None => None,
        Some(persisted_source) => {
            let path = if crate::media_source::parse_content_reference(persisted_source)
                .map_err(|error| format!("selected export audio source: {error}"))?
                .is_some()
            {
                let logical_name = config
                    .audio_path_hint
                    .as_deref()
                    .and_then(|hint| std::path::Path::new(hint).file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                resolve_export_file_source(
                    persisted_source,
                    &logical_name,
                    config.audio_path_hint.as_deref(),
                    &source_context,
                    is_supported_export_audio_source,
                    &mut source_fingerprints,
                )
                .map_err(|error| format!("selected export audio source: {error}"))?
                .path
                .to_string_lossy()
                .into_owned()
            } else {
                persisted_source.to_owned()
            };
            match media_has_audio_stream(&path) {
                Ok(true) => Some(path),
                Ok(false) => {
                    return Err(format!(
                        "selected export audio source '{path}' contains no audio stream"
                    ));
                }
                Err(error) => {
                    return Err(format!(
                        "failed to inspect selected export audio source '{path}': {error}"
                    ));
                }
            }
        }
    };
    check_cancelled(progress)?;
    let encoder = start_encoder_supervisor(
        "ffmpeg".to_string(),
        build_ffmpeg_args(config, audio_path.as_deref()),
        progress.clone(),
        config.output_path.clone(),
    )?;
    let EncoderSession {
        stdin: mut ffmpeg_stdin,
        completion: encoder_completion,
        supervisor: encoder_supervisor,
    } = encoder;

    // --- Frame loop ---
    let frame_interval = 1.0 / config.fps as f32;
    let mut write_error = None;
    let mut emitted_motion_warnings = std::collections::BTreeSet::new();

    let frame_gpu_errors = Arc::new(Mutex::new(Vec::new()));

    for frame_num in 0..total_frames {
        if progress.cancel.load(Ordering::Acquire) {
            break;
        }

        // Apply every due B9 take event to the authored bases before this
        // frame's world is built — the offline mirror of the live
        // drain-before-render order, addressed by the same rounded rational
        // tick map every other 30 Hz consumer uses.
        if let Some((take, cursor)) = performance_replay.as_mut() {
            let tick = u32::try_from(export_temporal_reference_tick(frame_num, config.fps))
                .unwrap_or(u32::MAX);
            let (start, end) = cursor.range_due(take.events(), tick);
            for event in &take.events()[start..end] {
                let address = &take.addresses()[usize::from(event.address)];
                if let Some(raw) = address.law.decode(event.value) {
                    apply_export_performance_event(
                        &address.control,
                        &raw,
                        &mut master_effects,
                        &mut master_transform,
                        &mut base_master_motion,
                        &mut base_layer_motion,
                        &mut base_ntsc,
                        &mut base_temporal,
                        &mut base_gesture_canvas,
                        &mut base_creative_graph,
                        &mut layers,
                        &mut export_morph,
                    );
                }
            }
        }

        // Update time uniform for effects (breathe, grain seed, etc.)
        let (time, program_dt) =
            export_program_transport(frame_num, frame_interval, patch.master_paused);
        master_effects.time = time;

        // Sample the modulation matrix at this frame's beat position and
        // derive the modulated params for this frame (bases untouched).
        let beat = time as f64 * (mod_matrix.clock.bpm as f64 / 60.0);
        if let Some(clip) = &analysis_clip {
            mod_matrix.audio = clip
                .analyze_at_time(
                    time as f64,
                    mod_matrix.audio_gain,
                    mod_matrix.audio_band_config,
                )
                .levels;
        } else {
            // Live capture and hardware sources are deliberately unavailable
            // offline; their zero value makes exports repeatable.
            mod_matrix.audio = crate::audio::AudioLevels::default();
        }
        update_export_modulation(&mut mod_matrix, beat, program_dt, patch.master_paused);
        let modulation_frame = mod_matrix.frame(layers.len());
        // Keep render parameters detached from decoder/runtime handles. A
        // full morph sample can then drive exactly the same layer world as
        // live rendering without mutating the saved patch bases.
        let baseline_morph_world = ExportFrameMorphWorld {
            creative_graph: base_creative_graph.clone(),
            master: master_effects,
            master_transform,
            master_motion: base_master_motion,
            ntsc: base_ntsc.clone(),
            temporal: base_temporal,
            gesture_canvas: base_gesture_canvas,
            layer_bases: layers.iter().map(ExportFrameLayerBase::from).collect(),
            layer_motion: base_layer_motion.clone(),
            morph_overrides: vec![ExportMorphOverrides::default(); layers.len()],
        };
        let frame_mattes = layers.iter().map(|layer| layer.matte).collect::<Vec<_>>();
        let mut composition_history_ready = composition_gpu
            .as_ref()
            .is_some_and(CompositionGpuExecutor::program_history_initialized);
        let mut frame_morph_world = baseline_morph_world.clone();

        // Live Pause holds the already-materialized patch bases and does not
        // re-apply a morph. Mirror that exact held-state contract offline.
        if let Some(requested_position) = export_morph_position(
            export_morph.as_ref(),
            beat,
            modulation_frame.morph_offset(),
            patch.master_paused,
        ) {
            let Some(morph) = export_morph.as_ref() else {
                write_error = Some("active export Morph disappeared during sampling".to_string());
                break;
            };
            let selected = select_valid_morph_candidate(requested_position, |sample_position| {
                let sample = morph.sample(sample_position).ok_or_else(|| {
                    format!("Morph has no complete sample at {sample_position:.4}")
                })?;
                let mut candidate = baseline_morph_world.clone();
                apply_export_morph_world(&sample, &layers, &mut candidate);
                let evaluated = evaluate_export_morph_world(
                    &candidate,
                    &layers,
                    &modulation_frame,
                    FramePlanContext::new(w, h, time)
                        .with_study_inputs(mod_matrix.audio.bands, (beat.fract().abs()) as f32),
                );
                plan_export_composition_with_motion(
                    &evaluated,
                    &candidate.creative_graph,
                    &frame_mattes,
                    composition_history_ready,
                    creative_resource_limits,
                    ExportMotionPlanAdapter {
                        master: candidate.master_motion,
                        layers: &candidate.layer_motion,
                        sources: &layers,
                        limits: motion_device_limits,
                        modulation: &modulation_frame,
                        authored_motion_modulation,
                        shutter_samples: config.shutter_samples,
                        codec_availability: None,
                    },
                )?;
                Ok(candidate)
            });
            match selected {
                Ok(selected) => {
                    if (selected.selected_position - selected.requested_position).abs()
                        > f32::EPSILON
                    {
                        progress.record_warning(format!(
                            "Morph sample {:.4} was invalid ({}); used captured endpoint {:.1}",
                            selected.requested_position,
                            selected
                                .requested_error
                                .as_deref()
                                .unwrap_or("creative graph rejected"),
                            selected.selected_position
                        ));
                    }
                    frame_morph_world = selected.value;
                }
                Err(error) => {
                    write_error = Some(error);
                    break;
                }
            }
        }

        let stable_modulation_frame = mod_matrix.stable_frame(&base_creative_graph.address_book);
        crate::modulation::apply_stable_modulation(
            &base_creative_graph.address_book,
            &stable_modulation_frame,
            &mut frame_morph_world.creative_graph.master_rack,
            &mut frame_morph_world.creative_graph.layer_racks,
            &mut frame_morph_world.creative_graph.composition,
        );
        let ExportFrameMorphWorld {
            creative_graph: frame_creative_graph,
            master: frame_master,
            master_transform: frame_master_transform,
            master_motion: frame_master_motion,
            ntsc: frame_ntsc,
            temporal: frame_temporal,
            gesture_canvas: frame_gesture_canvas_base,
            layer_bases: mut frame_layer_bases,
            layer_motion: frame_layer_motion,
            morph_overrides,
        } = frame_morph_world;

        // Resolve source time before the immutable visual plan is built so a
        // OneShot's transparent terminal state is part of the exact plan used
        // by direct, selective-NTSC, matte, and temporal paths. This is the
        // same pure TransportTimeline contract used live; export specializes
        // only by supplying a frame-indexed clock and synchronous indexed
        // absolute decode.
        let mut source_discontinuity = false;
        let mut motion_held_scopes = Vec::with_capacity(layers.len());
        let score_loop_driver = match frame_temporal.originals.score.loop_driver {
            crate::temporal::CollisionScoreLoopDriver::SelectedLayer { layer_id, .. } => {
                Some(layer_id)
            }
            crate::temporal::CollisionScoreLoopDriver::None
            | crate::temporal::CollisionScoreLoopDriver::MissingSelectedLayer { .. } => None,
        };
        let mut temporal_loop_events = 0_u32;
        for index in 0..layers.len() {
            let transport_config = modulated_export_transport_config(
                &modulation_frame,
                index,
                layers[index].transport.authored,
                &frame_layer_bases[index],
                morph_overrides[index],
            );
            let selection = layers[index].transport.select(
                transport_config,
                ProgramTransportTick {
                    delta_seconds: f64::from(program_dt),
                    program_beat: beat,
                    program_running: !patch.master_paused,
                    media_running: !patch.media_frozen && !frame_layer_bases[index].paused,
                    ..ProgramTransportTick::default()
                },
            );
            source_discontinuity |= selection.discontinuity;
            if let Some(scope) = export_motion_held_scope(
                &frame_creative_graph,
                layers[index].source_index,
                selection,
            ) {
                motion_held_scopes.push(scope);
            }

            // Absolute seeks do not touch VideoDecoder's EOF generation.
            // Count each pure timeline boundary once and only once, including
            // multiple wraps/reflections within a large deterministic tick.
            let layer = &mut layers[index];
            let loop_boundaries = if matches!(
                transport_config.end_behavior,
                crate::transport::EndBehavior::Loop | crate::transport::EndBehavior::PingPong
            ) {
                selection.boundary_events
            } else {
                0
            };
            if frame_creative_graph
                .layer_ids
                .get(layer.source_index)
                .is_some_and(|layer_id| Some(*layer_id) == score_loop_driver)
            {
                temporal_loop_events = temporal_loop_events.saturating_add(loop_boundaries);
            }
            let next_loop_generation = layer
                .consumed_loop_generation
                .saturating_add(u64::from(loop_boundaries));
            let reroll_on_loop = layer.reroll_on_loop;
            let rerolled = apply_export_loop_generation(
                &mut layer.effects,
                reroll_on_loop,
                &mut layer.consumed_loop_generation,
                next_loop_generation,
            );
            if rerolled > 0 && !morph_overrides[index].effects {
                frame_layer_bases[index].effects.random_seed = layer.effects.random_seed;
            }

            if selection.discontinuity || selection.transparent {
                // No field from the previous source generation may cross a
                // seek/cue/loop cut or a transparent OneShot terminal.
                layer.codec_motion = None;
                layer.codec_motion_predecessor = None;
            }

            if selection.sample_due && !selection.transparent {
                if progress.cancel.load(Ordering::Acquire) {
                    write_error = Some("export cancelled during indexed source decode".to_string());
                    break;
                }
                let previous_accepted_codec_identity = layer
                    .codec_motion_predecessor
                    .filter(|identity| identity.source_generation == selection.generation);
                let decoded = layer.decoder.as_mut().map(|decoder| {
                    decoder.seek_decode_after_for_generation(
                        selection.source_seconds,
                        selection.generation,
                        previous_accepted_codec_identity,
                    )
                });
                if let Some(decoded) = decoded {
                    match decoded {
                        Ok(frame) if frame.metadata.source_generation == selection.generation => {
                            let codec_motion = matching_export_codec_motion(
                                frame.codec_motion,
                                frame.metadata.source_generation,
                                [layer.width, layer.height],
                            );
                            if let Err(error) = upload_export_texture_checked(
                                &device,
                                &queue,
                                &layer.texture,
                                &frame.rgba,
                                layer.width,
                                layer.height,
                                "Export Indexed Video Frame",
                            ) {
                                write_error = Some(format!(
                                    "layer {} GPU upload failed during export: {error}",
                                    layer.source_index + 1
                                ));
                                break;
                            }
                            // Pixels and decoder metadata publish as one
                            // export-side transaction only after upload.
                            layer.codec_motion = codec_motion;
                            layer.codec_motion_predecessor =
                                frame.metadata.codec_identity.filter(|identity| {
                                    identity.source_generation == frame.metadata.source_generation
                                });
                        }
                        Ok(frame) => {
                            write_error = Some(format!(
                                "layer {} decoder returned stale generation {} for requested generation {}",
                                layer.source_index + 1,
                                frame.metadata.source_generation,
                                selection.generation
                            ));
                            break;
                        }
                        Err(error) => {
                            write_error = Some(format!(
                                "layer {} indexed decode failed during export: {error}",
                                layer.source_index + 1
                            ));
                            break;
                        }
                    }
                }
            }
            frame_layer_bases[index].visible &= !selection.transparent;
        }
        if write_error.is_some() {
            break;
        }
        if source_discontinuity {
            // Match live source-cut generation invalidation: no temporal,
            // ProgramHistory, or previous-scope donor may cross a seek/cue/
            // loop discontinuity into the next authored source generation.
            temporal_state.reset_for(crate::temporal::TemporalResetCause::SourceCut);
            if let Some(resources) = image_routing_gpu.as_mut() {
                resources.history_valid = false;
            }
            if let Some(executor) = composition_gpu.as_mut() {
                executor.reset_history_for(crate::temporal::TemporalResetCause::SourceCut);
            }
            composition_history_ready = false;
        }
        let temporal_crossings = temporal_boundaries.observe(beat, 4, patch.master_paused);
        let temporal_audio_events =
            temporal_audio_onsets.observe(mod_matrix.audio.onset, !patch.master_paused);
        let recorded_reference_tick = export_temporal_reference_tick(frame_num, config.fps);
        let recorded_events = recorded_temporal_replay.events_due(recorded_reference_tick);
        let temporal_input = crate::temporal::TemporalFrameInput::new(
            program_dt,
            match (patch.master_paused, patch.media_frozen) {
                (false, false) => crate::temporal::TemporalFreezeState::Running,
                (false, true) => crate::temporal::TemporalFreezeState::MediaFrozen,
                (true, false) => crate::temporal::TemporalFreezeState::ProgramFrozen,
                (true, true) => crate::temporal::TemporalFreezeState::ProgramAndMediaFrozen,
            },
            false,
            crate::temporal::TemporalFrameEvents {
                boundary_events: temporal_loop_events,
                downbeat_events: temporal_crossings.bars,
                audio_onset_events: temporal_audio_events,
                manual_events: recorded_events.manual_events,
                garden_refresh_events: recorded_events.garden_refresh_events,
            },
        )
        .with_audio_energy(mod_matrix.audio.level);

        // The gesture canvas opens its transaction here, on the same frame-
        // indexed 30 Hz address the recorded temporal track replays on. The
        // evaluated controls are the one architectural law's frame-local copy:
        // Morph materialized them above, modulation offsets a copy here, and
        // the authored patch values are never written as a side effect.
        let frame_gesture_canvas =
            modulation_frame.modulate_gesture_canvas(&frame_gesture_canvas_base);
        if let Err(error) = stage_export_gesture_canvas_frame(
            &mut gesture_canvas,
            &mut recorded_gesture_replay,
            frame_num,
            config.fps,
            temporal_input.freeze.program_advances(),
            frame_gesture_canvas,
        ) {
            write_error = Some(error);
            break;
        }

        // Live and export cross the same post-morph boundary here. From this
        // point onward one immutable evaluator owns render, transport, source,
        // transform, blend, and temporal values for the complete frame.
        let mut evaluated_frame =
            EvaluatedFramePlan::evaluate(
                &modulation_frame,
                FramePlanContext::new(w, h, time)
                    .with_study_inputs(mod_matrix.audio.bands, (beat.fract().abs()) as f32),
                MasterFrameInput {
                    effects: &frame_master,
                    transform: &frame_master_transform,
                    ntsc: &frame_ntsc,
                    temporal: &frame_temporal,
                },
                layers.iter().zip(frame_layer_bases.iter()).enumerate().map(
                    |(slot, (layer, base))| LayerFrameInput {
                        source: SourceTap::new(
                            export_selective_layer_id(layer.source_index),
                            slot,
                            layer.width,
                            layer.height,
                        ),
                        effects: &base.effects,
                        transform: &base.transform,
                        opacity: base.opacity,
                        mosh_send: base.mosh_send,
                        speed: base.speed,
                        fps: base.fps,
                        blend_mode: base.blend_mode,
                        visible: base.visible,
                        paused: base.paused,
                        bypass_master_fx: base.bypass_master_fx,
                        bypass_temporal_fx: base.bypass_temporal_fx,
                        pattern: base.pattern.as_ref(),
                    },
                ),
            );
        let mut evaluated_composition = match plan_export_composition_with_motion(
            &evaluated_frame,
            &frame_creative_graph,
            &frame_mattes,
            composition_history_ready,
            creative_resource_limits,
            ExportMotionPlanAdapter {
                master: frame_master_motion,
                layers: &frame_layer_motion,
                sources: &layers,
                limits: motion_device_limits,
                modulation: &modulation_frame,
                authored_motion_modulation,
                shutter_samples: config.shutter_samples,
                codec_availability: None,
            },
        ) {
            Ok(plan) => plan,
            Err(error) => {
                write_error = Some(error);
                break;
            }
        };
        debug_assert_eq!(evaluated_frame.context().output_size, [w, h]);
        let mod_ntsc = evaluated_frame.ntsc().clone();
        // The B14 sync latch draws its faults on the master random seed,
        // taken from the same immutable frame sample every other consumer
        // reads, so live and offline draw the identical fault stream.
        let mod_sync_seed = evaluated_frame.master_pass().effects.random_seed;
        let (mut motion_field_products, motion_field_diagnostics) = match &evaluated_composition {
            EvaluatedCompositionPlan::Advanced(plan) => {
                export_codec_motion_fields(plan, &frame_creative_graph, &layers)
            }
            EvaluatedCompositionPlan::LegacyExact(_) => (Vec::new(), Vec::new()),
        };
        let planned_codec_scopes = planned_export_codec_scopes(&evaluated_composition);
        let attached_codec_scopes = motion_field_products
            .iter()
            .map(|product| product.scope)
            .collect::<std::collections::BTreeSet<_>>();
        if !planned_codec_scopes.is_subset(&attached_codec_scopes) {
            let codec_availability = exact_export_codec_availability(
                &frame_creative_graph,
                &layers,
                &motion_field_products,
            );
            evaluated_composition = match plan_export_composition_with_motion(
                &evaluated_frame,
                &frame_creative_graph,
                &frame_mattes,
                composition_history_ready,
                creative_resource_limits,
                ExportMotionPlanAdapter {
                    master: frame_master_motion,
                    layers: &frame_layer_motion,
                    sources: &layers,
                    limits: motion_device_limits,
                    modulation: &modulation_frame,
                    authored_motion_modulation,
                    shutter_samples: config.shutter_samples,
                    codec_availability: Some(&codec_availability),
                },
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    write_error = Some(format!("export codec fallback plan rejected: {error}"));
                    break;
                }
            };
            // The immutable retry may only remove codec requirements or
            // select lattice fallback. Retain products already rasterized
            // against the exact first plan; never perform a second fallible
            // allocation/rasterization after fallback preflight.
            retain_exact_export_codec_fields(&evaluated_composition, &mut motion_field_products);
        }
        if !export_codec_fields_complete(&evaluated_composition, &motion_field_products) {
            write_error =
                Some("export codec field set changed after immutable fallback planning".to_owned());
            break;
        }
        // The same composition plan that routes the inherited master prefix
        // owns the linked Temporal-bypass law. This copy is neutral only for
        // the rendered frame; authored/modulated temporal state stays intact.
        let mod_temporal = evaluated_composition.effective_temporal();
        let mosh_active = mod_temporal.mosh.is_active();
        let temporal_bypass_mode =
            export_temporal_bypass_mode(evaluated_composition.temporal_master_path(), mosh_active);
        if temporal_bypass_overlay_active(&evaluated_frame)
            != (temporal_bypass_mode != ExportTemporalBypassMode::Inactive)
        {
            write_error =
                Some("isolated Temporal bypass planner/export classification mismatch".to_string());
            break;
        }
        if temporal_bypass_mode != ExportTemporalBypassMode::Inactive && mod_ntsc.enabled {
            write_error = Some(
                "isolated Temporal bypass reached export with VHS enabled; planner must reject this topology"
                    .to_string(),
            );
            break;
        }
        if temporal_bypass_mode != ExportTemporalBypassMode::Inactive {
            match preflight_temporal_bypass_overlay_resources(&layers, &evaluated_frame, w, h) {
                Ok(resources) if !resources.is_empty() => {}
                Ok(_) => {
                    gesture_canvas.discard_staged();
                    write_error = Some(
                        "temporal bypass export plan had no contributing dry overlay".to_string(),
                    );
                    break;
                }
                Err(error) => {
                    gesture_canvas.discard_staged();
                    write_error = Some(format!(
                        "temporal bypass export resource preflight failed: {error}"
                    ));
                    break;
                }
            }
        }
        if temporal_bypass_mode == ExportTemporalBypassMode::AroundCodecMosh {
            if let Err(error) = ensure_export_temporal_bypass_program_candidate(
                &device,
                &mut temporal_bypass_program_candidate,
                w,
                h,
            ) {
                gesture_canvas.discard_staged();
                write_error = Some(format!(
                    "temporal bypass export candidate preflight failed: {error}"
                ));
                break;
            }
        }
        update_export_mosh_interval(mosh_active, &mut mosh_interval_active, &mut mosh_engine);
        if observe_export_temporal_bypass_partition(
            &mut previous_temporal_bypass_partition,
            &mut temporal_bypass_partition_initialized,
            &evaluated_frame,
        ) {
            // Membership is part of the Temporal machine's input identity.
            // Rebase every memory that may still contain pixels from a newly
            // dry layer. `BypassPartition` is HARD, so this also invalidates a
            // paused freeze hold and rebuilds it from the new wet stack.
            temporal_state.reset_for(crate::temporal::TemporalResetCause::BypassPartition);
            melting_edge_gpu.invalidate_history();
            display_physics_gpu.invalidate_memory();
            sync_latch_gpu.reset_for(crate::temporal::TemporalResetCause::BypassPartition);
            if let Some(resources) = image_routing_gpu.as_mut() {
                // Legacy ProgramHistory is partitioned composition state too:
                // neither a prior full stack nor an isolated wet-only stack
                // may become the N-1 donor after membership changes.
                resources.history_valid = false;
            }
            if let Some(executor) = composition_gpu.as_mut() {
                executor.reset_temporal_bypass_partition();
            }
            // Codec Mosh owns decoded/recycled prior images outside the GPU
            // Temporal state. Dropping the synchronous engine is its exact
            // generation cut; it reopens lazily if this frame remains armed.
            mosh_engine = None;
        }
        for warning in export_motion_plan_warnings(&evaluated_composition) {
            if emitted_motion_warnings.insert(warning.clone()) {
                progress.record_warning(warning);
            }
        }
        for warning in motion_field_diagnostics {
            if emitted_motion_warnings.insert(warning.clone()) {
                progress.record_warning(warning);
            }
        }
        let motion_field_attachments = motion_field_products
            .iter()
            .map(ExportMotionFieldProduct::attachment)
            .collect::<Vec<_>>();

        // --- GPU render ---
        let frame_gpu_scope = ExportGpuErrorCapture::new(
            &device,
            format!("export frame {} GPU operation failed", frame_num + 1),
            frame_gpu_errors.clone(),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Frame Encoder"),
        });

        // B7 pattern-synth sources render first, into their own layer
        // textures inside this same encoder — the identical seat the live
        // frame uses, from the identical plan-owned uniforms, so there is no
        // export-only synth path.
        {
            let pattern_jobs: Vec<(
                crate::pattern_synth::PatternSynthGpuUniforms,
                &wgpu::TextureView,
            )> = evaluated_frame
                .layers()
                .iter()
                .zip(layers.iter())
                .filter_map(|(evaluated, layer)| {
                    evaluated.pattern.map(|params| {
                        (
                            crate::pattern_synth::PatternSynthGpuUniforms::from_params(
                                &params,
                                evaluated_frame.context().time_seconds,
                            ),
                            &layer.texture_view,
                        )
                    })
                })
                .collect();
            if !pattern_jobs.is_empty() {
                let stage = pattern_synth_gpu.get_or_insert_with(|| {
                    crate::renderer::pattern_synth::PatternSynthGpu::new(&device)
                });
                stage.encode(&device, &queue, &mut encoder, &pattern_jobs);
            }
        }

        if let Err(error) = encode_export_mosh_influence_frame(
            &device,
            &queue,
            &mut encoder,
            &evaluated_frame,
            &layers,
            &sampler,
            &nearest_sampler,
            &composite_views[1],
            mosh_active,
            &mosh_influence_pipelines,
            &mut mosh_influence_gpu,
            [w, h],
        ) {
            write_error = Some(format!("export layer Codec-Mosh influence failed: {error}"));
            break;
        }

        let (selective_frame, advanced_history_staged) = match &evaluated_composition {
            EvaluatedCompositionPlan::Advanced(advanced) => {
                if composition_gpu.is_none() {
                    composition_gpu = match CompositionGpuExecutor::new(&device, &queue, [w, h]) {
                        Ok(executor) => Some(executor),
                        Err(error) => {
                            write_error = Some(format!(
                                "export advanced composition initialization failed: {error}"
                            ));
                            break;
                        }
                    };
                }
                let source_descriptors = advanced
                    .layers()
                    .iter()
                    .map(|planned| {
                        let layer = &layers[planned.base_layer_index];
                        CompositionSourceDescriptor::new(
                            planned.stable_id,
                            &layer.texture_view,
                            [layer.width, layer.height],
                        )
                    })
                    .collect::<Vec<_>>();
                let executor = composition_gpu
                    .as_mut()
                    .expect("advanced export executor was initialized above");
                // The same binding law as live: the presented donor of the one
                // admitted offline canvas, published before prepare so the tap
                // bind groups are built against it exactly once. An absent
                // canvas is the exact pre-gesture path.
                executor.bind_gesture_canvas(match gesture_canvas_gpu.as_ref() {
                    Some(resources) => match resources.presented_view() {
                        Some(view) => crate::renderer::composition::GestureCanvasBinding::bound(
                            view.clone(),
                            1,
                        ),
                        None => crate::renderer::composition::GestureCanvasBinding::default(),
                    },
                    None => crate::renderer::composition::GestureCanvasBinding::default(),
                });
                // Match live readiness exactly: the route is planned for the
                // job's whole lifetime, but frame zero is unbound until one
                // frame reaches the acceptance seam below. The texture never
                // rebuilds inside a job, so every later frame uses epoch one.
                executor.bind_program_tap(if program_tap_valid {
                    crate::renderer::composition::ProgramTapBinding::bound(
                        program_tap_view.clone(),
                        1,
                    )
                } else {
                    crate::renderer::composition::ProgramTapBinding::default()
                });
                match executor.prepare(&device, &queue, &evaluated_composition, &source_descriptors)
                {
                    Ok(CompositionPreparedKind::Advanced { .. }) => {}
                    Ok(CompositionPreparedKind::LegacyExact) => {
                        executor.discard_frame_history();
                        write_error = Some(
                            "advanced export planner/executor delegation mismatch".to_string(),
                        );
                        break;
                    }
                    Err(error) => {
                        executor.discard_frame_history();
                        write_error = Some(format!(
                            "export advanced composition preparation failed: {error}"
                        ));
                        break;
                    }
                }
                match executor.encode_with_motion(
                    &queue,
                    &mut encoder,
                    &evaluated_composition,
                    CompositionFrameTiming::from_temporal_input(temporal_input),
                    CompositionMotionFrameInput {
                        attachments: &motion_field_attachments,
                        held_scopes: &motion_held_scopes,
                    },
                ) {
                    Ok(CompositionEncodeKind::Advanced) => {}
                    Ok(CompositionEncodeKind::LegacyExact) => {
                        executor.discard_frame_history();
                        write_error =
                            Some("advanced export planner/executor encode mismatch".to_string());
                        break;
                    }
                    Err(error) => {
                        executor.discard_frame_history();
                        write_error = Some(format!("export advanced composition failed: {error}"));
                        break;
                    }
                }
                let output = executor.output();
                debug_assert_eq!(output.dimensions, [w, h]);
                debug_assert_eq!(
                    output.format,
                    crate::renderer::composition::COMPOSITION_WORKING_FORMAT
                );
                let _ = (output.texture, output.view);
                if let Err(error) = executor.encode_present(
                    &mut encoder,
                    &composite_views[0],
                    COMPOSITION_PRESENT_FORMAT,
                ) {
                    executor.discard_frame_history();
                    write_error = Some(format!(
                        "export advanced composition presentation failed: {error}"
                    ));
                    break;
                }

                // The shared executor owns LegacyTemporal for Advanced and
                // presents one straight-alpha compatibility image into slot
                // zero. The B4 display stage rides the shared slot-0 seam,
                // then reuse the established sole opaque boundary before
                // readback/NTSC, exactly as the live advanced handoff does.
                melting_edge_gpu.encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &composite_textures,
                    &composite_views,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    &mod_temporal.melt,
                    temporal_input.program_advancing_delta(),
                );
                sync_latch_gpu.encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &composite_textures,
                    &composite_views,
                    &mod_temporal.sync,
                    mod_sync_seed,
                    temporal_input.program_advancing_delta(),
                );
                display_physics_gpu.encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &composite_textures,
                    &composite_views,
                    &mod_temporal.display,
                    temporal_input.program_advancing_delta(),
                );
                if temporal_bypass_mode == ExportTemporalBypassMode::BeforeOpaque {
                    match render_planned_temporal_bypass_overlay_export(
                        Some(&*executor),
                        &evaluated_composition,
                        &device,
                        &queue,
                        &mut encoder,
                        &layers,
                        &evaluated_frame,
                        &composite_textures,
                        &composite_views,
                        &effects_pipeline,
                        &effects_bind_group_layout,
                        &effects_uniform_layout,
                        &composite_pipeline,
                        &composite_bind_group_layout,
                        &composite_uniform_layout,
                        &sampler,
                        &nearest_sampler,
                        w,
                        h,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            executor.discard_frame_history();
                            gesture_canvas.discard_staged();
                            write_error = Some(
                                "Advanced temporal bypass export had no contributing dry overlay"
                                    .to_string(),
                            );
                            break;
                        }
                        Err(error) => {
                            executor.discard_frame_history();
                            gesture_canvas.discard_staged();
                            write_error =
                                Some(format!("Advanced temporal bypass export failed: {error}"));
                            break;
                        }
                    }
                }
                crate::renderer::state::encode_opaque_output(
                    &mut encoder,
                    &opaque_output_pipeline,
                    &opaque_output_bind_group,
                    &composite_views[2],
                );
                // VHS is independent of per-layer Master bypass and finishes
                // every Advanced programme with one global CPU kernel.
                (false, true)
            }
            EvaluatedCompositionPlan::LegacyExact(_) => {
                // Exact delegation retains the old routing attachment and GPU
                // sequence byte-for-byte. The unified planner can only select
                // this branch when every raw authored matte is a no-op.
                let legacy_history_ready = image_routing_gpu
                    .as_ref()
                    .is_some_and(|resources| resources.history_valid);
                if let Err(error) =
                    evaluated_frame.attach_image_routing(frame_mattes, legacy_history_ready)
                {
                    write_error = Some(format!("export image routing rejected frame: {error}"));
                    break;
                }
                let selective_plan = plan_export_selective_ntsc(
                    frame_num,
                    config.fps,
                    patch.master_paused,
                    &evaluated_frame,
                );
                if selective_plan.is_some() {
                    if let Err(error) = validate_selective_matte_topology(
                        evaluated_frame.image_routing().is_active(),
                    ) {
                        write_error = Some(error);
                        break;
                    }
                }
                let selective_frame = selective_plan.is_some();
                let selective_topology = selective_plan
                    .as_ref()
                    .map(export_selective_topology_signature);
                let selective_edge =
                    previous_selective_frame.is_some_and(|previous| previous != selective_frame);
                let selective_topology_changed = selective_frame
                    && previous_selective_topology
                        .is_some_and(|previous| Some(previous) != selective_topology);
                if selective_edge || selective_topology_changed {
                    // Match live `reset_visual_generation`: no pre-switch feedback or
                    // slit history may cross a selective-VHS topology/path edge.
                    temporal_state = crate::renderer::state::TemporalState::default();
                }
                previous_selective_frame = Some(selective_frame);
                previous_selective_topology = selective_topology;
                if let Some(plan) = selective_plan {
                    if selective_staging.is_none() {
                        match create_export_readback_buffer(
                            &device,
                            buffer_size,
                            "Export Selective NTSC Readback",
                        ) {
                            Ok(buffer) => selective_staging = Some(buffer),
                            Err(error) => {
                                write_error = Some(error);
                                break;
                            }
                        }
                    }
                    let Some(selective_staging) = selective_staging.as_ref() else {
                        write_error =
                            Some("selective NTSC staging initialization failed".to_string());
                        break;
                    };
                    // Each slice is rendered through local FX and conditional direct
                    // master FX in one coherent command stream. Composite/VHS stays
                    // on the CPU, so no unprocessed intermediate reaches Temporal.
                    let slices = match render_and_readback_selective_ntsc_slices_export(
                        &device,
                        &queue,
                        &mut encoder,
                        &layers,
                        &evaluated_frame,
                        &plan,
                        &composite_textures,
                        &composite_views,
                        &effects_pipeline,
                        &effects_bind_group_layout,
                        &effects_uniform_layout,
                        &sampler,
                        &nearest_sampler,
                        selective_staging,
                        w,
                        h,
                        bytes_per_row,
                        &progress.cancel,
                    ) {
                        Ok(slices) => slices,
                        Err(error) => {
                            write_error = Some(error);
                            break;
                        }
                    };
                    if progress.cancel.load(Ordering::Acquire) {
                        write_error =
                            Some("export cancelled before selective NTSC processing".into());
                        break;
                    }
                    let processed = match process_selective_ntsc_batch_with_state_and_resolution(
                        &mut selective_ntsc_state,
                        SelectiveNtscBatch { plan, slices },
                        config.ntsc_quality,
                    ) {
                        Ok(processed) => processed,
                        Err(error) => {
                            write_error = Some(format!("selective NTSC export failed: {error}"));
                            break;
                        }
                    };
                    if progress.cancel.load(Ordering::Acquire) {
                        write_error =
                            Some("export cancelled during selective NTSC processing".into());
                        break;
                    }
                    if let Err(error) = upload_engine_composite_export(
                        &device,
                        &queue,
                        &composite_textures[0],
                        &processed.pixels,
                        w,
                        h,
                    ) {
                        write_error = Some(error);
                        break;
                    }
                    // CPU processing and queue upload are complete before Temporal is
                    // encoded. A fresh command stream prevents the earlier layer
                    // passes from overwriting the returned straight-alpha composite.
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Export Selective Post-NTSC Encoder"),
                    });
                } else {
                    // Byte-for-byte legacy path: VHS-off and all-inherited stacks keep
                    // the established direct render -> Temporal -> opaque -> global
                    // NTSC order without any selective slice allocation or composite.
                    if let Err(error) = render_layers_and_master_export_routed(
                        &device,
                        &queue,
                        &mut encoder,
                        &layers,
                        &evaluated_frame,
                        &composite_textures,
                        &composite_views,
                        &effects_pipeline,
                        &effects_bind_group_layout,
                        &effects_uniform_layout,
                        &composite_pipeline,
                        &composite_bind_group_layout,
                        &composite_uniform_layout,
                        &sampler,
                        &nearest_sampler,
                        &matte_composite,
                        &mut image_routing_gpu,
                        w,
                        h,
                    ) {
                        write_error = Some(error);
                        break;
                    }
                }

                // Temporal effects + history recording (identical pass to live)
                crate::renderer::state::encode_temporal_prepared_frame(
                    &queue,
                    &mut encoder,
                    &mod_temporal,
                    &temporal_pipeline,
                    &temporal_prepared,
                    &composite_textures,
                    &composite_views,
                    &history_texture,
                    &feedback_texture,
                    &mut temporal_state,
                    temporal_input,
                    w,
                    h,
                );

                // The B4 display stage rides the shared slot-0 seam offline,
                // exactly where live encodes it.
                melting_edge_gpu.encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &composite_textures,
                    &composite_views,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    &mod_temporal.melt,
                    temporal_input.program_advancing_delta(),
                );
                sync_latch_gpu.encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &composite_textures,
                    &composite_views,
                    &mod_temporal.sync,
                    mod_sync_seed,
                    temporal_input.program_advancing_delta(),
                );
                display_physics_gpu.encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &composite_textures,
                    &composite_views,
                    &mod_temporal.display,
                    temporal_input.program_advancing_delta(),
                );
                if temporal_bypass_mode == ExportTemporalBypassMode::BeforeOpaque {
                    let overlay = render_planned_temporal_bypass_overlay_export(
                        composition_gpu.as_ref(),
                        &evaluated_composition,
                        &device,
                        &queue,
                        &mut encoder,
                        &layers,
                        &evaluated_frame,
                        &composite_textures,
                        &composite_views,
                        &effects_pipeline,
                        &effects_bind_group_layout,
                        &effects_uniform_layout,
                        &composite_pipeline,
                        &composite_bind_group_layout,
                        &composite_uniform_layout,
                        &sampler,
                        &nearest_sampler,
                        w,
                        h,
                    );
                    match overlay {
                        Ok(true) => {}
                        Ok(false) => {
                            temporal_state.discard_staged();
                            gesture_canvas.discard_staged();
                            write_error = Some(
                                "temporal bypass export plan had no contributing dry overlay"
                                    .to_string(),
                            );
                            break;
                        }
                        Err(error) => {
                            temporal_state.discard_staged();
                            gesture_canvas.discard_staged();
                            write_error = Some(format!("temporal bypass export failed: {error}"));
                            break;
                        }
                    }
                }
                // Export consumes the same opaque SDR program image as live preview,
                // projector, Spout, and NTSC. Keep key alpha inside the engine and
                // flatten it over black exactly once at this boundary.
                crate::renderer::state::encode_opaque_output(
                    &mut encoder,
                    &opaque_output_pipeline,
                    &opaque_output_bind_group,
                    &composite_views[2],
                );
                (selective_frame, false)
            }
        };

        // Pack the layer send into alpha only for the Codec-Mosh readback.
        // RGB is copied byte-for-byte from the opaque programme, so analysis,
        // preview, and every non-Mosh export retain the exact old texture.
        let use_mosh_influence_alpha = mosh_active
            && mosh_influence_gpu
                .as_ref()
                .is_some_and(crate::renderer::state::MoshInfluenceGpu::valid);
        let mosh_readback_texture = if use_mosh_influence_alpha {
            mosh_influence_gpu
                .as_ref()
                .expect("validated export Mosh influence source")
                .encode_pack(&mut encoder, &composite_textures[2])
        } else {
            &composite_textures[2]
        };

        // Submit GPU work
        queue.submit(std::iter::once(encoder.finish()));

        // --- NTSC using the same half-resolution path as live output ---
        let mut pixels = match readback_pixels(
            &device,
            &queue,
            mosh_readback_texture,
            &staging,
            w,
            h,
            bytes_per_row,
            &progress.cancel,
        ) {
            Ok(pixels) => pixels,
            Err(error) => {
                temporal_state.discard_staged();
                gesture_canvas.discard_staged();
                if advanced_history_staged {
                    if let Some(executor) = composition_gpu.as_mut() {
                        executor.discard_frame_history();
                    }
                }
                write_error = Some(error);
                break;
            }
        };
        // An isolated dry prefix must be visible to analysis and the N-1
        // programme tap, but Codec Mosh must receive only the wet audience.
        // Render the pre-mosh full programme into a transaction candidate now;
        // it is published only after the final frame reaches ffmpeg.
        let mut temporal_bypass_analysis_pixels: Option<Vec<u8>> = None;
        if temporal_bypass_mode == ExportTemporalBypassMode::AroundCodecMosh {
            let candidate = temporal_bypass_program_candidate
                .as_ref()
                .expect("temporal bypass program candidate passed frame preflight");
            let mut overlay_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Export Pre-Mosh Temporal Bypass Overlay"),
                });
            let overlay = render_planned_temporal_bypass_overlay_export(
                composition_gpu.as_ref(),
                &evaluated_composition,
                &device,
                &queue,
                &mut overlay_encoder,
                &layers,
                &evaluated_frame,
                &composite_textures,
                &composite_views,
                &effects_pipeline,
                &effects_bind_group_layout,
                &effects_uniform_layout,
                &composite_pipeline,
                &composite_bind_group_layout,
                &composite_uniform_layout,
                &sampler,
                &nearest_sampler,
                w,
                h,
            );
            match overlay {
                Ok(true) => {}
                Ok(false) => {
                    temporal_state.discard_staged();
                    gesture_canvas.discard_staged();
                    write_error = Some(
                        "pre-mosh temporal bypass plan had no contributing dry overlay".to_string(),
                    );
                    break;
                }
                Err(error) => {
                    temporal_state.discard_staged();
                    gesture_canvas.discard_staged();
                    write_error = Some(format!("pre-mosh temporal bypass export failed: {error}"));
                    break;
                }
            }
            crate::renderer::state::encode_opaque_output(
                &mut overlay_encoder,
                &opaque_output_pipeline,
                &opaque_output_bind_group,
                &composite_views[2],
            );
            overlay_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[2],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: candidate,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(std::iter::once(overlay_encoder.finish()));
            if mod_matrix.video_analysis_armed() {
                temporal_bypass_analysis_pixels = match readback_pixels(
                    &device,
                    &queue,
                    candidate,
                    &staging,
                    w,
                    h,
                    bytes_per_row,
                    &progress.cancel,
                ) {
                    Ok(pixels) => Some(pixels),
                    Err(error) => {
                        temporal_state.discard_staged();
                        gesture_canvas.discard_staged();
                        write_error = Some(error);
                        break;
                    }
                };
            }
        }
        // B10 video analysis, from the exact bytes live reduces: the
        // pre-VHS/pre-mosh program image (the pre-blackout slot-2 seam the
        // live reduction and the program tap both observe). The sample lands
        // in the matrix now and is consumed by the NEXT frame's modulation
        // update — the same N-1 law the live readback obeys — on the same
        // 10 Hz reference cadence, so a take of chaos, envelopes, and video
        // reactivity replays identically for the same job.
        if mod_matrix.video_analysis_armed() {
            video_analysis_accumulator += f64::from(program_dt);
            let interval = 3.0 / f64::from(crate::effects::params::TEMPORAL_REFERENCE_FPS);
            if video_analysis_accumulator >= interval {
                let analysis_pixels = temporal_bypass_analysis_pixels
                    .as_deref()
                    .unwrap_or(pixels.as_slice());
                let grid = crate::modulation::reduce_video_analysis_grid(
                    analysis_pixels,
                    w as usize,
                    h as usize,
                );
                let sample = video_analysis_state.analyze(&grid, video_analysis_accumulator as f32);
                mod_matrix.set_video_analysis(sample);
                video_analysis_accumulator = 0.0;
            }
        } else if video_analysis_primed {
            video_analysis_state.reset();
            mod_matrix.set_video_analysis(crate::modulation::VideoAnalysisSample::default());
            video_analysis_accumulator = 0.0;
        }
        video_analysis_primed = mod_matrix.video_analysis_armed();

        // B5: the codec round trip runs before the final-program VHS finish.
        // Live executes both serial kernels in the same bounded worker hop, so
        // this ordering adds no readback, queue, copy, or latency stage.
        if mosh_active {
            let mosh_result = (|| -> Result<(), String> {
                if mosh_engine.is_none() {
                    mosh_engine = Some(crate::codec_mosh::MoshEngine::open(w, h)?);
                }
                let engine = mosh_engine.as_mut().expect("mosh engine opened above");
                let ordinal =
                    export_ntsc_reference_frame(frame_num, config.fps, patch.master_paused) as u64;
                engine.apply(
                    &mut pixels,
                    w,
                    h,
                    mod_temporal.mosh,
                    use_mosh_influence_alpha,
                    ordinal,
                    master_effects.random_seed,
                )
            })();
            if let Err(error) = mosh_result {
                temporal_state.discard_staged();
                gesture_canvas.discard_staged();
                if advanced_history_staged {
                    if let Some(executor) = composition_gpu.as_mut() {
                        executor.discard_frame_history();
                    }
                }
                write_error = Some(format!("codec mosh failed: {error}"));
                break;
            }
            mosh_used = true;
        }

        if temporal_bypass_mode == ExportTemporalBypassMode::AroundCodecMosh {
            // `pixels` is the CPU-returned wet audience here. Put it back into
            // the engine seam, reapply the independently dry prefix, and only
            // then read the visible frame that ffmpeg receives. Codec Mosh has
            // therefore affected wet pixels exactly once and dry pixels zero
            // times, while the final file still contains the complete stack.
            if let Err(error) = upload_export_texture_checked(
                &device,
                &queue,
                &composite_textures[0],
                &pixels,
                w,
                h,
                "Export Moshed Temporal Wet Program",
            ) {
                temporal_state.discard_staged();
                gesture_canvas.discard_staged();
                write_error = Some(error);
                break;
            }
            let mut final_overlay_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Export Post-Mosh Temporal Bypass Overlay"),
                });
            let overlay = render_planned_temporal_bypass_overlay_export(
                composition_gpu.as_ref(),
                &evaluated_composition,
                &device,
                &queue,
                &mut final_overlay_encoder,
                &layers,
                &evaluated_frame,
                &composite_textures,
                &composite_views,
                &effects_pipeline,
                &effects_bind_group_layout,
                &effects_uniform_layout,
                &composite_pipeline,
                &composite_bind_group_layout,
                &composite_uniform_layout,
                &sampler,
                &nearest_sampler,
                w,
                h,
            );
            match overlay {
                Ok(true) => {}
                Ok(false) => {
                    temporal_state.discard_staged();
                    gesture_canvas.discard_staged();
                    write_error = Some(
                        "post-mosh temporal bypass plan had no contributing dry overlay"
                            .to_string(),
                    );
                    break;
                }
                Err(error) => {
                    temporal_state.discard_staged();
                    gesture_canvas.discard_staged();
                    write_error = Some(format!("post-mosh temporal bypass export failed: {error}"));
                    break;
                }
            }
            crate::renderer::state::encode_opaque_output(
                &mut final_overlay_encoder,
                &opaque_output_pipeline,
                &opaque_output_bind_group,
                &composite_views[2],
            );
            queue.submit(std::iter::once(final_overlay_encoder.finish()));
            pixels = match readback_pixels(
                &device,
                &queue,
                &composite_textures[2],
                &staging,
                w,
                h,
                bytes_per_row,
                &progress.cancel,
            ) {
                Ok(pixels) => pixels,
                Err(error) => {
                    temporal_state.discard_staged();
                    gesture_canvas.discard_staged();
                    write_error = Some(error);
                    break;
                }
            };
        }

        // VHS is the final stylized programme pass, after creative composition,
        // Temporal/display processing, Codec Mosh, and any independently dry
        // overlay. Blackout remains a live-only absolute cut downstream.
        if !selective_frame && mod_ntsc.enabled {
            ntsc_state.params = mod_ntsc;
            if !ntsc_state.apply_at_reference_frame_with_resolution(
                &mut pixels,
                w,
                h,
                export_ntsc_reference_frame(frame_num, config.fps, patch.master_paused),
                config.ntsc_quality,
            ) {
                temporal_state.discard_staged();
                gesture_canvas.discard_staged();
                if advanced_history_staged {
                    if let Some(executor) = composition_gpu.as_mut() {
                        executor.discard_frame_history();
                    }
                }
                write_error = Some(format!(
                    "final-program VHS rejected export frame {frame_num} at {w}x{h}"
                ));
                break;
            }
        }

        // Write to ffmpeg
        if let Err(error) = ffmpeg_stdin.write_all(&pixels) {
            temporal_state.discard_staged();
            gesture_canvas.discard_staged();
            if advanced_history_staged {
                if let Some(executor) = composition_gpu.as_mut() {
                    executor.discard_frame_history();
                }
            }
            write_error = Some(format!("failed to write video frame to ffmpeg: {error}"));
            break;
        }

        drop(frame_gpu_scope);
        if let Some(error) = take_export_gpu_errors(&frame_gpu_errors) {
            write_error = Some(error);
        }
        if write_error.is_some() {
            temporal_state.discard_staged();
            gesture_canvas.discard_staged();
            if advanced_history_staged {
                if let Some(executor) = composition_gpu.as_mut() {
                    executor.discard_frame_history();
                }
            }
            break;
        }
        temporal_state.commit_staged();
        // An accepted frame commits the etch; every other exit above discarded
        // it, so a frame that never reached ffmpeg leaves no visible change.
        //
        // The device half is published here, at the acceptance decision and
        // before the CPU commit closes the transaction — byte for byte the
        // live ordering, which is what makes a canvas route read the same N-1
        // field in both sessions.
        if let Some(resources) = gesture_canvas_gpu.as_mut() {
            let mut gesture_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Export gesture canvas frame"),
                });
            match resources.encode_staged_frame(&queue, &mut gesture_encoder, 0, &gesture_canvas) {
                Ok(_) => {
                    queue.submit(std::iter::once(gesture_encoder.finish()));
                }
                Err(error) => {
                    gesture_canvas.discard_staged();
                    write_error = Some(format!("export gesture canvas device update: {error}"));
                    break;
                }
            }
        }
        gesture_canvas.commit_staged();
        // The programme tap shares the same acceptance seam. Ordinarily slot 2
        // is the pre-VHS/pre-mosh opaque audience image. Isolated Codec Mosh is
        // the one exception: slot 2 now holds the visible moshed-wet+dry file
        // frame, so publish the retained pre-mosh wet+dry candidate instead.
        // Neither source is made visible until ffmpeg has accepted the frame.
        {
            let program_tap_source =
                if temporal_bypass_mode == ExportTemporalBypassMode::AroundCodecMosh {
                    temporal_bypass_program_candidate
                        .as_ref()
                        .expect("accepted isolated mosh frame retained its pre-mosh programme")
                } else {
                    &composite_textures[2]
                };
            let mut tap_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Export program tap publish"),
            });
            tap_encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: program_tap_source,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &program_tap_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(std::iter::once(tap_encoder.finish()));
            program_tap_valid = true;
        }
        if advanced_history_staged {
            if let Some(executor) = composition_gpu.as_mut() {
                executor.commit_frame_history();
            }
        }
        let rendered_motion_fields =
            export_rendered_motion_fields(&evaluated_composition, composition_gpu.as_ref());
        let frame_motion_metadata = export_motion_metadata_for_frame(
            &evaluated_composition,
            &frame_creative_graph,
            &layers,
            &rendered_motion_fields,
            frame_num,
        );
        motion_sidecar.observe_accepted(&frame_motion_metadata);
        if mosh_active {
            if let Some(observed) = mosh_observed.as_mut() {
                observed.observe(mod_temporal.mosh);
            } else {
                mosh_observed = Some(CodecMoshObservationAccumulator::new(mod_temporal.mosh));
            }
            for (observation, layer) in mosh_layer_send_observed
                .iter_mut()
                .zip(evaluated_frame.layers())
            {
                observation.observe(
                    layer.mosh_send,
                    layer.visible
                        && layer.opacity.is_finite()
                        && layer.opacity > 0.0
                        && !layer.bypass_temporal_fx,
                );
            }
        }
        progress.publish_motion_metadata(frame_motion_metadata);

        // Update progress
        progress.progress.store(
            ((frame_num + 1) * 10000 / total_frames) as u32,
            Ordering::Relaxed,
        );
    }

    if let Some(gpu_error) = take_export_gpu_errors(&frame_gpu_errors) {
        write_error = Some(match write_error {
            Some(existing) => format!("{existing}; {gpu_error}"),
            None => gpu_error,
        });
    }
    // Several earlier failures break out of the render loop before reaching the
    // acceptance decision. An abandoned frame is a discarded frame, so the
    // canvas ends the job on committed truth or on nothing at all.
    gesture_canvas.discard_staged();

    // Tell the supervisor this is an intentional end-of-input before closing
    // stdin. A panic/unwind that drops stdin without this flag is a failure.
    progress
        .encoder_finish_requested
        .store(true, Ordering::Release);
    drop(ffmpeg_stdin);
    let completion = encoder_completion
        .recv()
        .map_err(|_| "encoder supervisor ended without a completion result".to_string())?;
    log::debug!("export worker received encoder completion");
    encoder_supervisor
        .join()
        .map_err(|_| "encoder supervisor panicked".to_string())?;

    // Cancellation owns the terminal state even if ffmpeg happened to fail
    // while its stdin was being closed. Cleanup is complete before observers
    // see `cancelled = true`.
    if progress.cancel.load(Ordering::Acquire) {
        publish_cancelled_terminal(progress, Some(&config.output_path));
        drop(layers);
        drop(staging);
        drop(selective_staging);
        drop(temporal_prepared);
        drop(history_view);
        drop(history_texture);
        drop(feedback_view);
        drop(feedback_texture);
        drop(opaque_output_bind_group);
        drop(opaque_output_bind_group_layout);
        drop(opaque_output_pipeline);
        drop(composite_views);
        drop(composite_textures);
        drop(temporal_pipeline);
        drop(temporal_bgl);
        drop(temporal_ubl);
        drop(composite_pipeline);
        drop(composite_pipeline_layout);
        drop(composite_fragment);
        drop(composite_uniform_layout);
        drop(composite_bind_group_layout);
        drop(effects_pipeline);
        drop(effects_pipeline_layout);
        drop(effects_fragment);
        drop(vertex_shader);
        drop(effects_uniform_layout);
        drop(effects_bind_group_layout);
        drop(nearest_sampler);
        drop(sampler);
        drop(queue);
        drop(device);
        drop(ntsc_state);
        drop(selective_ntsc_state);
        drop(mod_matrix);
        drop(export_morph);
        return Err("export cancelled".to_string());
    }

    let status = completion.status.inspect_err(|_| {
        let _ = std::fs::remove_file(&config.output_path);
    })?;
    let stderr_bytes = completion.stderr.inspect_err(|_| {
        let _ = std::fs::remove_file(&config.output_path);
    })?;
    let stderr = String::from_utf8_lossy(&stderr_bytes);
    if let Some(error) = write_error {
        let _ = std::fs::remove_file(&config.output_path);
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            error
        } else {
            format!("{error}: {}", detail.chars().take(300).collect::<String>())
        });
    }
    if !status.success() {
        let _ = std::fs::remove_file(&config.output_path);
        return Err(format!(
            "ffmpeg failed: {}",
            stderr.chars().take(300).collect::<String>()
        ));
    }

    check_cancelled(progress)?;
    // Recorded once per job, only when an accepted frame actually ran the
    // round trip: the authored recipe plus the encoder's identity — the
    // per-host repeatability record.
    if mosh_used {
        let observed = mosh_observed
            .take()
            .ok_or_else(|| {
                "codec mosh ran without an accepted frame-evaluated sidecar recipe".to_string()
            })?
            .finish();
        let recipe = patch
            .temporal
            .as_ref()
            .map(|temporal| temporal.mosh.sanitized())
            .unwrap_or_default();
        let layer_sends_truncated = layers.len() > MAX_EXPORT_MOTION_SIDECAR_SOURCES;
        let layer_sends = layers
            .iter()
            .zip(&mosh_layer_send_observed)
            .take(MAX_EXPORT_MOTION_SIDECAR_SOURCES)
            .map(|(layer, observation)| {
                // Performance replay deliberately mutates `ExportLayer` bases
                // in place. Authored provenance must instead remain pinned to
                // the immutable saved patch; replay/Morph/modulation belong
                // only in the observed bounds below.
                let authored = codec_mosh_authored_layer_send(patch, layer.source_index)?;
                Ok(CodecMoshLayerSendSidecar {
                    saved_position: u32::try_from(layer.source_index).unwrap_or(u32::MAX),
                    stable_id: export_selective_layer_id(layer.source_index),
                    authored,
                    observed_min: if observation.seen {
                        observation.min
                    } else {
                        authored
                    },
                    observed_max: if observation.seen {
                        observation.max
                    } else {
                        authored
                    },
                    entered_codec_mosh: observation.entered_codec_mosh,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let (encode_width, encode_height) = crate::codec_mosh::mosh_dimensions(w, h);
        motion_sidecar.codec_mosh = Some(CodecMoshSidecar {
            encoder: crate::codec_mosh::mosh_encoder_identity(),
            encode_width,
            encode_height,
            amount: recipe.amount,
            key_removal: recipe.key_removal,
            hold: recipe.hold,
            drop: recipe.drop,
            shuffle: recipe.shuffle,
            rate: recipe.rate,
            bitrate_starve: recipe.bitrate_starve,
            resync: recipe.resync,
            wipe: recipe.wipe,
            smear: recipe.smear,
            trail: recipe.trail,
            recycle: recipe.recycle,
            layer_sends,
            layer_sends_truncated,
            observed,
        });
    }
    let sidecar = motion_sidecar.finish(progress.warnings());
    write_motion_sidecar_atomic(&config.output_path, &sidecar)?;
    // A job that replayed a recording publishes it beside the render. An
    // export with no recorded gesture writes no sidecar at all, so the
    // pre-gesture output pair stays byte-identical and an unrecorded live
    // gesture is never implied replayable by a file that does not exist.
    if let Some(document) = config.gesture_track.as_ref() {
        if !document.events.is_empty() {
            write_gesture_sidecar_noreplace(&config.output_path, document)?;
        }
    }
    // The replayed performance take publishes on the same law: no take, no
    // file, so the pre-recorder output pair stays byte-identical.
    if let Some(document) = config.performance_take.as_ref() {
        if !document.events.is_empty() {
            write_performance_sidecar_noreplace(&config.output_path, document)?;
        }
    }
    // A cancellation that wins immediately after atomic sidecar publication
    // still owns terminal cleanup; `finalize_export_worker` removes both the
    // video and its paired report.
    check_cancelled(progress)?;

    Ok(())
}

/// Readback composite_textures[0] to CPU as RGBA bytes (no row padding).
// These arguments are the complete inputs to one GPU readback operation.
#[allow(clippy::too_many_arguments)]
fn readback_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    staging: &wgpu::Buffer,
    w: u32,
    h: u32,
    bytes_per_row: u32,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, String> {
    if cancel.load(Ordering::Acquire) {
        staging.destroy();
        return Err("export cancelled before GPU readback".to_string());
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    loop {
        if cancel.load(Ordering::Acquire) {
            // `unmap` explicitly cancels a pending map_async operation (and
            // releases an already-completed mapping). Destroying afterward
            // prevents device/resource teardown from waiting on this staging
            // allocation after run_export has published cancellation.
            staging.unmap();
            staging.destroy();
            return Err("export cancelled during GPU readback".to_string());
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)),
        });
        match rx.try_recv() {
            Ok(result) => {
                if let Err(error) = result {
                    staging.unmap();
                    staging.destroy();
                    return Err(format!("GPU readback map failed: {error}"));
                }
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => continue,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                staging.unmap();
                staging.destroy();
                return Err("GPU readback callback disconnected".to_string());
            }
        }
    }

    let data = slice.get_mapped_range();
    let row_bytes = (w * 4) as usize;
    let padded_row = bytes_per_row as usize;
    let mut pixels = Vec::with_capacity(row_bytes * h as usize);
    for row in 0..h as usize {
        let start = row * padded_row;
        pixels.extend_from_slice(&data[start..start + row_bytes]);
    }
    drop(data);
    staging.unmap();

    Ok(pixels)
}

fn upload_engine_composite_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "selective NTSC export dimensions overflow".to_string())?;
    if pixels.len() != expected {
        return Err(format!(
            "selective NTSC export composite has {} bytes; expected {expected}",
            pixels.len()
        ));
    }
    upload_export_texture_checked(
        device,
        queue,
        texture,
        pixels,
        width,
        height,
        "Export Selective NTSC Composite",
    )
}

/// Resolve only the pinned export texture selected by the immutable plan.
/// Every render/control value lives in `EvaluatedFramePlan`; mutable decoder
/// fields on `ExportLayer` are never render authority after evaluation.
fn export_layer_resource(
    layers: &[ExportLayer],
    source: SourceTap,
) -> Result<&ExportLayer, String> {
    let layer = layers.get(source.slot).ok_or_else(|| {
        format!(
            "evaluated export layer {} refers to missing resource slot {}",
            source.stable_id, source.slot
        )
    })?;
    let captured = SourceTap::new(
        export_selective_layer_id(layer.source_index),
        source.slot,
        layer.width,
        layer.height,
    );
    if captured != source {
        return Err(format!(
            "evaluated export resource mismatch at slot {}: planned id {} at {}x{}, captured id {} at {}x{}",
            source.slot,
            source.stable_id,
            source.size[0],
            source.size[1],
            captured.stable_id,
            captured.size[0],
            captured.size[1]
        ));
    }
    Ok(layer)
}

/// Resolve the complete dry prefix before Temporal stages any history. A
/// deferred source mismatch is therefore a frame-preflight error rather than
/// a late overlay failure after Temporal/Melt/Display have encoded work.
fn preflight_temporal_bypass_overlay_resources<'a>(
    layers: &'a [ExportLayer],
    evaluated: &EvaluatedFramePlan,
    output_width: u32,
    output_height: u32,
) -> Result<Vec<(usize, &'a ExportLayer)>, String> {
    if evaluated.context().output_size != [output_width, output_height] {
        return Err("temporal bypass export dimensions do not match target".to_string());
    }
    if layers.len() != evaluated.layers().len()
        || evaluated.layers().len() != evaluated.layer_passes().len()
    {
        return Err("temporal bypass export resource/plan alignment mismatch".to_string());
    }
    if evaluated.image_routing().is_active() {
        return Err("temporal bypass export cannot execute with routed image mattes".to_string());
    }

    export_temporal_bypass_overlay_sources(evaluated)
        .into_iter()
        .map(|(layer_index, source)| {
            export_layer_resource(layers, source).map(|layer| (layer_index, layer))
        })
        .collect()
}

/// Render all planned straight-alpha slices and synchronously collect them in
/// the shared plan's bottom-to-top order. Export is intentionally synchronous:
/// it must write each exact requested frame to ffmpeg, unlike live preview's
/// bounded delayed worker.
#[allow(clippy::too_many_arguments)]
fn render_and_readback_selective_ntsc_slices_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    plan: &SelectiveNtscPlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    cancel: &AtomicBool,
) -> Result<Vec<Vec<u8>>, String> {
    if layers.len() != evaluated.layers().len()
        || evaluated.layers().len() != evaluated.layer_passes().len()
    {
        return Err("selective NTSC export resource/plan alignment mismatch".into());
    }
    if plan.generation.width != width || plan.generation.height != height {
        return Err("selective NTSC export plan dimensions changed before rendering".into());
    }

    let master_buffer = create_uploaded_uniform(
        device,
        queue,
        "Export Selective NTSC Master FX Uniforms",
        evaluated.master_pass(),
    );
    let master_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Selective NTSC Master FX Input"),
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(nearest_sampler),
            },
        ],
    });
    let master_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Selective NTSC Master FX Uniforms BG"),
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: master_buffer.as_entire_binding(),
        }],
    });

    let mut slices = Vec::with_capacity(plan.layers.len());
    for planned_layer in &plan.layers {
        if cancel.load(Ordering::Acquire) {
            return Err("export cancelled before selective NTSC slice".to_string());
        }
        let source_index = evaluated
            .layers()
            .iter()
            .position(|layer| layer.source.stable_id == planned_layer.layer_id)
            .ok_or_else(|| {
                format!(
                    "selective NTSC export layer {} disappeared before rendering",
                    planned_layer.layer_id
                )
            })?;
        let evaluated_layer = &evaluated.layers()[source_index];
        let layer = export_layer_resource(layers, evaluated_layer.source)?;
        if !evaluated_layer.visible
            || evaluated_layer.bypass_master_fx != planned_layer.bypass_master_fx
        {
            return Err(format!(
                "selective NTSC export layer {} changed before rendering",
                planned_layer.layer_id
            ));
        }

        let fx_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Selective NTSC Layer FX Uniforms",
            &evaluated.layer_passes()[source_index],
        );
        let layer_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Selective NTSC Layer FX Input"),
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
            ],
        });
        let layer_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Selective NTSC Layer FX Uniforms BG"),
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Selective NTSC Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &layer_tex_bg, &[]);
            pass.set_bind_group(1, &layer_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let output_slot = if planned_layer.bypass_master_fx {
            1
        } else {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Selective NTSC Direct Master FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &master_tex_bg, &[]);
            pass.set_bind_group(1, &master_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
            2
        };

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[output_slot],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // One coherent slice must complete before the shared staging buffer is
        // reused. This is bounded and deterministic for the offline path.
        let finished = std::mem::replace(
            encoder,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Export Selective NTSC Slice Encoder"),
            }),
        );
        queue.submit(std::iter::once(finished.finish()));
        slices.push(map_export_readback(
            device,
            staging,
            width,
            height,
            bytes_per_row,
            cancel,
            "selective NTSC slice",
        )?);
    }
    Ok(slices)
}

fn map_export_readback(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    cancel: &AtomicBool,
    label: &str,
) -> Result<Vec<u8>, String> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    loop {
        if cancel.load(Ordering::Acquire) {
            staging.unmap();
            return Err(format!("export cancelled during GPU {label} readback"));
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_millis(100)),
        });
        match rx.try_recv() {
            Ok(Ok(())) => break,
            Ok(Err(error)) => {
                staging.unmap();
                return Err(format!("GPU {label} readback map failed: {error}"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => continue,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                staging.unmap();
                return Err(format!("GPU {label} readback callback disconnected"));
            }
        }
    }
    let data = slice.get_mapped_range();
    let row_bytes = (width * 4) as usize;
    let mut pixels = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let start = row * bytes_per_row as usize;
        pixels.extend_from_slice(&data[start..start + row_bytes]);
    }
    drop(data);
    staging.unmap();
    Ok(pixels)
}

fn ensure_export_image_routing_resources<'a>(
    device: &wgpu::Device,
    evaluated: &EvaluatedFramePlan,
    resources: &'a mut Option<ImageRoutingGpuResources>,
) -> Result<&'a mut ImageRoutingGpuResources, String> {
    let request = evaluated
        .image_routing()
        .resource_plan()
        .ok_or_else(|| "export image-routing resources requested for inactive plan".to_string())?;
    let adapter_plan = MatteResourcePlan::validate(
        request.output_size,
        evaluated.image_routing().taps().len(),
        MatteResourceLimits::from_wgpu(&device.limits()),
    )?;
    if adapter_plan != request {
        return Err(format!(
            "evaluated export image-routing plan {request:?} differs from adapter plan {adapter_plan:?}"
        ));
    }

    if resources.is_none() {
        *resources = Some(scoped_export_gpu_operation(
            device,
            "export image-routing full-frame allocation",
            || ImageRoutingGpuResources::build(device, adapter_plan),
        )?);
    }
    let resources = resources
        .as_mut()
        .ok_or_else(|| "export image-routing resources were not retained".to_string())?;
    if resources.output_size != adapter_plan.output_size {
        return Err("export image-routing output dimensions changed during the job".into());
    }
    if resources.tap_layers != adapter_plan.tap_layers {
        resources.taps =
            scoped_export_gpu_operation(device, "export image tap array reallocation", || {
                ImageTapTexture::build(device, adapter_plan.output_size, adapter_plan.tap_layers)
            })?;
        resources.tap_layers = adapter_plan.tap_layers;
    }
    Ok(resources)
}

#[allow(clippy::too_many_arguments)]
fn encode_export_effect_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    input: &wgpu::TextureView,
    output: &wgpu::TextureView,
    uniforms: &EffectPassUniforms,
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    label: &'static str,
) {
    let buffer = create_uploaded_uniform(device, queue, label, uniforms);
    let textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(nearest_sampler),
            },
        ],
    });
    let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(effects_pipeline);
    pass.set_bind_group(0, &textures, &[]);
    pass.set_bind_group(1, &uniform_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Build the same programme-space Codec-Mosh send field used by the live
/// renderer from the immutable export frame plan. The default all-one field
/// deliberately keeps the historical frame path: no full-frame texture or
/// pass is created and the ordinary opaque programme remains the readback
/// source. Immutable optional kernels were precompiled during job setup.
#[allow(clippy::too_many_arguments)]
fn encode_export_mosh_influence_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    evaluated: &EvaluatedFramePlan,
    layers: &[ExportLayer],
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    scratch: &wgpu::TextureView,
    mosh_active: bool,
    prepared: &Result<crate::renderer::state::MoshInfluencePipelines, String>,
    influence: &mut Option<crate::renderer::state::MoshInfluenceGpu>,
    dimensions: [u32; 2],
) -> Result<bool, String> {
    if layers.len() != evaluated.layers().len()
        || evaluated.layers().len() != evaluated.layer_passes().len()
    {
        if let Some(influence) = influence.as_mut() {
            influence.invalidate();
        }
        return Err("Codec-Mosh influence export resource/plan alignment mismatch".into());
    }

    if !mosh_active {
        if let Some(influence) = influence.as_mut() {
            influence.invalidate();
        }
        return Ok(false);
    }
    let first_authored =
        crate::renderer::state::mosh_influence_stack_start(evaluated.layers(), true);
    let Some(first_authored) = first_authored else {
        if let Some(influence) = influence.as_mut() {
            influence.invalidate();
        }
        return Ok(false);
    };

    if influence.is_none() {
        let prepared = prepared.as_ref().map_err(Clone::clone)?;
        *influence = Some(crate::renderer::state::MoshInfluenceGpu::new(
            device,
            sampler,
            nearest_sampler,
            scratch,
            dimensions,
            evaluated.layers().len(),
            prepared,
        )?);
    }
    influence
        .as_mut()
        .expect("export Codec-Mosh influence was initialized above")
        .begin_frame(device, encoder, evaluated.layers().len())?;

    // Bottom-to-top composition matters: a later full-send layer restores
    // its covered pixels after a dry or partially sent layer beneath it.
    for layer_index in (0..=first_authored).rev().filter(|&index| {
        let layer = &evaluated.layers()[index];
        layer.visible && !layer.bypass_temporal_fx
    }) {
        let evaluated_layer = &evaluated.layers()[layer_index];
        if !evaluated_layer.opacity.is_finite() || evaluated_layer.opacity <= 0.0 {
            continue;
        }
        let layer = export_layer_resource(layers, evaluated_layer.source)?;
        influence
            .as_mut()
            .expect("export Codec-Mosh influence stays alive for the frame")
            .encode_source_layer(
                device,
                queue,
                encoder,
                evaluated_layer.source,
                &layer.texture_view,
                &evaluated.layer_passes()[layer_index],
                evaluated_layer.mosh_send,
                evaluated_layer.opacity,
                evaluated_layer.blend_mode,
            )?;
    }
    influence
        .as_mut()
        .expect("export Codec-Mosh influence stays alive through publication")
        .finish_frame();
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn materialize_export_image_taps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    resources: &ImageRoutingGpuResources,
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
) -> Result<(), String> {
    let Some(tap_texture) = resources.taps.as_ref() else {
        if evaluated.image_routing().taps().is_empty() {
            return Ok(());
        }
        return Err("export tap plan has no GPU tap array".into());
    };
    for tap in evaluated.image_routing().taps() {
        let evaluated_layer = evaluated
            .layers()
            .get(tap.donor_layer_index)
            .ok_or_else(|| "export tap donor is outside the evaluated stack".to_string())?;
        let layer = export_layer_resource(layers, evaluated_layer.source)?;
        let output = tap_texture
            .views
            .get(tap.array_layer as usize)
            .ok_or_else(|| format!("export tap layer {} is missing", tap.array_layer))?;
        let uniforms = match tap.stage {
            crate::image_routing::LayerImageStage::PreLocalEffects => {
                evaluated.layer_pre_passes().get(tap.donor_layer_index)
            }
            crate::image_routing::LayerImageStage::PostLocalEffects => {
                evaluated.layer_passes().get(tap.donor_layer_index)
            }
        }
        .ok_or_else(|| "export tap pass/layer alignment mismatch".to_string())?;
        encode_export_effect_pass(
            device,
            queue,
            encoder,
            &layer.texture_view,
            output,
            uniforms,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            sampler,
            nearest_sampler,
            "Export Materialize Image Tap",
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_routed_layers_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    matte_composite: &MatteCompositePipeline,
    routing: &ImageRoutingGpuResources,
    output_width: u32,
    output_height: u32,
) -> Result<(), String> {
    let path = master_fx_composition_path(
        evaluated
            .layers()
            .iter()
            .map(|layer| (layer.visible, layer.bypass_master_fx, layer.opacity)),
    );
    let visible_layers =
        visible_stack_indices(evaluated.layers().iter().map(|layer| layer.visible));
    if visible_layers.is_empty() {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Routed Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[0],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        if path == MasterFxCompositionPath::LegacyPostComposite {
            render_master_effects_export(
                device,
                queue,
                encoder,
                evaluated.master_pass(),
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                sampler,
                nearest_sampler,
                output_width,
                output_height,
            );
        }
        return Ok(());
    }

    for (stack_index, &layer_index) in visible_layers.iter().enumerate() {
        let evaluated_layer = &evaluated.layers()[layer_index];
        let layer = export_layer_resource(layers, evaluated_layer.source)?;
        encode_export_effect_pass(
            device,
            queue,
            encoder,
            &layer.texture_view,
            &composite_views[1],
            &evaluated.layer_passes()[layer_index],
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            sampler,
            nearest_sampler,
            "Export Routed Layer FX",
        );

        let mut overlay_slot = 1;
        if path == MasterFxCompositionPath::ConditionalPerLayer && !evaluated_layer.bypass_master_fx
        {
            encode_export_effect_pass(
                device,
                queue,
                encoder,
                &composite_views[1],
                &composite_views[2],
                evaluated.master_pass(),
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                sampler,
                nearest_sampler,
                "Export Routed Conditional Master FX",
            );
            overlay_slot = 2;
        }

        if stack_index == 0 {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Routed Clear Base"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }

        let matte = evaluated
            .image_routing()
            .mattes()
            .get(layer_index)
            .ok_or_else(|| "export matte/layer alignment mismatch".to_string())?;
        let mut params = matte.params;
        let donor = match matte.resolved_input {
            ResolvedImageInput::Disabled => {
                params.amount = 0.0;
                params.donor_valid = false;
                &routing.history.view
            }
            ResolvedImageInput::MaterializedTap { tap_index } => {
                let tap = evaluated
                    .image_routing()
                    .taps()
                    .get(tap_index)
                    .ok_or_else(|| format!("export matte refers to missing tap {tap_index}"))?;
                routing
                    .taps
                    .as_ref()
                    .and_then(|texture| texture.views.get(tap.array_layer as usize))
                    .ok_or_else(|| {
                        format!(
                            "export matte tap array layer {} is missing",
                            tap.array_layer
                        )
                    })?
            }
            ResolvedImageInput::AllBelow => &composite_views[0],
            ResolvedImageInput::ProgramHistory => {
                params.donor_valid &= routing.history_valid;
                &routing.history.view
            }
            ResolvedImageInput::Transparent => {
                params.donor_valid = false;
                &routing.history.view
            }
        };
        let output_slot = if overlay_slot == 1 { 2 } else { 1 };
        encode_matte_composite(
            device,
            queue,
            encoder,
            matte_composite,
            sampler,
            &composite_views[0],
            &composite_views[overlay_slot],
            donor,
            &composite_views[output_slot],
            MatteCompositeUniforms::new(
                evaluated_layer.opacity,
                evaluated_layer.blend_mode.as_u32(),
                params,
            ),
        );
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[output_slot],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
        );
    }

    if path == MasterFxCompositionPath::LegacyPostComposite {
        render_master_effects_export(
            device,
            queue,
            encoder,
            evaluated.master_pass(),
            composite_textures,
            composite_views,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            sampler,
            nearest_sampler,
            output_width,
            output_height,
        );
    }
    Ok(())
}

/// Routed extension of the established export compositor. The inactive arm
/// invokes the old function literally and performs no tap allocation.
#[allow(clippy::too_many_arguments)]
fn render_layers_and_master_export_routed(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    matte_composite: &MatteCompositePipeline,
    routing: &mut Option<ImageRoutingGpuResources>,
    output_width: u32,
    output_height: u32,
) -> Result<(), String> {
    if !evaluated.image_routing().is_active() {
        render_layers_and_master_export(
            device,
            queue,
            encoder,
            layers,
            evaluated,
            composite_textures,
            composite_views,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_uniform_layout,
            sampler,
            nearest_sampler,
            output_width,
            output_height,
        )?;
        if let Some(resources) = routing.as_mut() {
            if export_may_publish_legacy_program_history(evaluated) {
                encode_program_history_copy(encoder, &composite_textures[0], resources);
            } else {
                // Slot 0 is wet-only until the post-Temporal dry overlay. It
                // is not a visible programme and must never become N-1 state.
                resources.history_valid = false;
            }
        }
        return Ok(());
    }

    let resources = ensure_export_image_routing_resources(device, evaluated, routing)?;
    materialize_export_image_taps(
        device,
        queue,
        encoder,
        layers,
        evaluated,
        resources,
        effects_pipeline,
        effects_bind_group_layout,
        effects_uniform_layout,
        sampler,
        nearest_sampler,
    )?;
    render_routed_layers_export(
        device,
        queue,
        encoder,
        layers,
        evaluated,
        composite_textures,
        composite_views,
        effects_pipeline,
        effects_bind_group_layout,
        effects_uniform_layout,
        sampler,
        nearest_sampler,
        matte_composite,
        resources,
        output_width,
        output_height,
    )?;
    if export_may_publish_legacy_program_history(evaluated) {
        encode_program_history_copy(encoder, &composite_textures[0], resources);
    } else {
        // Planner validation rejects this topology, but keep the adapter
        // fail-closed if an isolated dry plan ever reaches the routed arm.
        resources.history_valid = false;
    }
    Ok(())
}

/// Mirror [`crate::renderer::state::Renderer::render_evaluated_frame`].
/// The legacy branch deliberately delegates to the two pre-existing export
/// passes unchanged; only a visible bypass enters the conditional path.
#[allow(clippy::too_many_arguments)]
fn render_layers_and_master_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) -> Result<(), String> {
    if evaluated.context().output_size != [output_width, output_height] {
        return Err(format!(
            "evaluated export frame is {}x{}, target is {output_width}x{output_height}",
            evaluated.context().output_size[0],
            evaluated.context().output_size[1]
        ));
    }
    if layers.len() != evaluated.layers().len()
        || evaluated.layers().len() != evaluated.layer_passes().len()
    {
        return Err("evaluated export resource/plan alignment mismatch".into());
    }
    let isolated_temporal_overlay = temporal_bypass_overlay_active(evaluated);
    let path = master_fx_composition_path(evaluated.layers().iter().filter_map(|layer| {
        (!isolated_temporal_overlay || !layer.bypass_temporal_fx).then_some((
            layer.visible,
            layer.bypass_master_fx,
            layer.opacity,
        ))
    }));
    match path {
        MasterFxCompositionPath::LegacyPostComposite => {
            render_layers_export(
                device,
                queue,
                encoder,
                layers,
                evaluated,
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                composite_pipeline,
                composite_bind_group_layout,
                composite_uniform_layout,
                sampler,
                nearest_sampler,
                output_width,
                output_height,
            )?;
            if !isolated_temporal_overlay || export_temporal_wet_layer_contributes(evaluated) {
                render_master_effects_export(
                    device,
                    queue,
                    encoder,
                    evaluated.master_pass(),
                    composite_textures,
                    composite_views,
                    effects_pipeline,
                    effects_bind_group_layout,
                    effects_uniform_layout,
                    sampler,
                    nearest_sampler,
                    output_width,
                    output_height,
                );
            }
        }
        MasterFxCompositionPath::ConditionalPerLayer => {
            render_layers_with_conditional_master_export(
                device,
                queue,
                encoder,
                layers,
                evaluated,
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                composite_pipeline,
                composite_bind_group_layout,
                composite_uniform_layout,
                sampler,
                nearest_sampler,
                output_width,
                output_height,
            )?;
        }
    }
    Ok(())
}

/// Reapply the exact top-prefix dry stack to the wet programme in slot 0.
///
/// The planner admits this only for LegacyExact without routing or VHS. Each
/// dry layer therefore follows the same bounded three-slot law as live:
/// source -> local FX -> optional Master FX -> opacity/blend over the current
/// wet/dry accumulation, in bottom-to-top order. The caller decides whether
/// slot 0 is the straight-alpha post-Temporal engine image or an uploaded
/// opaque Codec-Mosh wet replacement.
#[allow(clippy::too_many_arguments)]
fn render_temporal_bypass_overlay_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) -> Result<bool, String> {
    let overlay_resources = preflight_temporal_bypass_overlay_resources(
        layers,
        evaluated,
        output_width,
        output_height,
    )?;
    if overlay_resources.is_empty() {
        return Ok(false);
    }

    for (layer_index, layer) in overlay_resources {
        let evaluated_layer = &evaluated.layers()[layer_index];
        encode_export_effect_pass(
            device,
            queue,
            encoder,
            &layer.texture_view,
            &composite_views[1],
            &evaluated.layer_passes()[layer_index],
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            sampler,
            nearest_sampler,
            "Export Temporal Bypass Layer FX",
        );

        let overlay_slot = if evaluated_layer.bypass_master_fx {
            1
        } else {
            encode_export_effect_pass(
                device,
                queue,
                encoder,
                &composite_views[1],
                &composite_views[2],
                evaluated.master_pass(),
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                sampler,
                nearest_sampler,
                "Export Temporal Bypass Inherited Master FX",
            );
            2
        };
        let output_slot = if overlay_slot == 1 { 2 } else { 1 };
        let composite_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Temporal Bypass Composite Uniforms",
            &CompositeUniforms {
                opacity: evaluated_layer.opacity,
                blend_mode: evaluated_layer.blend_mode.as_u32(),
                _pad: [0.0; 2],
            },
        );
        let composite_texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Temporal Bypass Composite Inputs"),
            layout: composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&composite_views[overlay_slot]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let composite_uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Temporal Bypass Composite Uniforms BG"),
            layout: composite_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: composite_buffer.as_entire_binding(),
            }],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Temporal Bypass Final Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[output_slot],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &composite_texture_group, &[]);
            pass.set_bind_group(1, &composite_uniform_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[output_slot],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(true)
}

/// Offline counterpart of live's precomposed Advanced overlay route. Every
/// view comes from the shared executor after the layer's ordered rack and
/// layer Motion; export never substitutes a legacy raw-source pass.
#[allow(clippy::too_many_arguments)]
fn render_advanced_temporal_bypass_overlay_export(
    executor: &CompositionGpuExecutor,
    advanced: &AdvancedCompositionPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) -> Result<bool, String> {
    let retained = executor
        .temporal_bypass_overlays(advanced)
        .map_err(|error| error.to_string())?;
    let expected = advanced.temporal_dry_layers();
    if retained.len() != expected.len() {
        return Err(format!(
            "Advanced Temporal bypass retained {} export overlays for {} planned dry layers",
            retained.len(),
            expected.len()
        ));
    }
    let mut overlays = Vec::with_capacity(retained.len());
    for (index, retained) in retained.into_iter().enumerate() {
        if expected.get(index).copied() != Some(retained.stable_id) {
            return Err(format!(
                "Advanced Temporal bypass retained export layer {} out of planned order at slot {index}",
                retained.stable_id.get()
            ));
        }
        let layer_plan = advanced
            .layers()
            .iter()
            .find(|layer| layer.stable_id == retained.stable_id)
            .ok_or_else(|| {
                format!(
                    "Advanced Temporal bypass retained unknown export layer {}",
                    retained.stable_id.get()
                )
            })?;
        let layer = advanced
            .base()
            .layers()
            .get(layer_plan.base_layer_index)
            .ok_or_else(|| {
                format!(
                    "Advanced Temporal bypass export layer {} has invalid base index {}",
                    retained.stable_id.get(),
                    layer_plan.base_layer_index
                )
            })?;
        if layer.visible && (layer.opacity > 0.0 || !layer.opacity.is_finite()) {
            overlays.push((retained.view, layer));
        }
    }
    if overlays.is_empty() {
        return Ok(false);
    }
    for (view, layer) in overlays {
        let overlay_view = if layer.bypass_master_fx {
            view
        } else {
            encode_export_effect_pass(
                device,
                queue,
                encoder,
                view,
                &composite_views[1],
                advanced.base().master_pass(),
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                sampler,
                nearest_sampler,
                "Export Advanced Temporal Bypass Inherited Master FX",
            );
            &composite_views[1]
        };
        let composite_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Advanced Temporal Bypass Composite Uniforms",
            &CompositeUniforms {
                opacity: layer.opacity,
                blend_mode: layer.blend_mode.as_u32(),
                _pad: [0.0; 2],
            },
        );
        let composite_texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Advanced Temporal Bypass Composite Inputs"),
            layout: composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let composite_uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Advanced Temporal Bypass Composite Uniforms BG"),
            layout: composite_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: composite_buffer.as_entire_binding(),
            }],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Advanced Temporal Bypass Final Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &composite_texture_group, &[]);
            pass.set_bind_group(1, &composite_uniform_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[2],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn render_planned_temporal_bypass_overlay_export(
    executor: Option<&CompositionGpuExecutor>,
    composition: &EvaluatedCompositionPlan,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) -> Result<bool, String> {
    match composition {
        EvaluatedCompositionPlan::LegacyExact(_) => render_temporal_bypass_overlay_export(
            device,
            queue,
            encoder,
            layers,
            evaluated,
            composite_textures,
            composite_views,
            effects_pipeline,
            effects_bind_group_layout,
            effects_uniform_layout,
            composite_pipeline,
            composite_bind_group_layout,
            composite_uniform_layout,
            sampler,
            nearest_sampler,
            output_width,
            output_height,
        ),
        EvaluatedCompositionPlan::Advanced(advanced) => {
            render_advanced_temporal_bypass_overlay_export(
                executor.ok_or_else(|| {
                    "Advanced Temporal bypass has no prepared export executor".to_string()
                })?,
                advanced,
                device,
                queue,
                encoder,
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                composite_pipeline,
                composite_bind_group_layout,
                composite_uniform_layout,
                sampler,
                nearest_sampler,
                output_width,
                output_height,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_layers_with_conditional_master_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) -> Result<(), String> {
    let visible_layers = export_temporal_wet_stack_indices(evaluated);
    debug_assert!(visible_layers
        .iter()
        .any(|&index| evaluated.layers()[index].bypass_master_fx));

    let master_buffer = create_uploaded_uniform(
        device,
        queue,
        "Export Conditional Master FX Uniforms",
        evaluated.master_pass(),
    );
    let master_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Conditional Master FX Input"),
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(nearest_sampler),
            },
        ],
    });
    let master_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Conditional Master FX Uniforms BG"),
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: master_buffer.as_entire_binding(),
        }],
    });

    for (stack_index, &layer_index) in visible_layers.iter().enumerate() {
        let evaluated_layer = &evaluated.layers()[layer_index];
        let layer = export_layer_resource(layers, evaluated_layer.source)?;
        let layer_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Conditional Layer FX Uniforms",
            &evaluated.layer_passes()[layer_index],
        );
        let layer_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Layer FX Input"),
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
            ],
        });
        let layer_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Layer FX Uniforms BG"),
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: layer_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &layer_tex_bg, &[]);
            pass.set_bind_group(1, &layer_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let slots = conditional_layer_slots(evaluated_layer.bypass_master_fx);
        if let Some(master_output) = slots.master_output {
            debug_assert_eq!(master_output, 2);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Master FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[master_output],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &master_tex_bg, &[]);
            pass.set_bind_group(1, &master_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if stack_index == 0 {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Clear Base"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }

        let overlay_slot = slots.master_output.unwrap_or(1);
        let comp_uniforms = CompositeUniforms {
            opacity: evaluated_layer.opacity,
            blend_mode: evaluated_layer.blend_mode.as_u32(),
            _pad: [0.0; 2],
        };
        let comp_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Conditional Composite Uniforms",
            &comp_uniforms,
        );
        let composite_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Composite Textures BG"),
            layout: composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&composite_views[overlay_slot]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let composite_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Composite Uniform BG"),
            layout: composite_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: comp_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[slots.composite_output],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &composite_tex_bg, &[]);
            pass.set_bind_group(1, &composite_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[slots.composite_output],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
        );
    }
    Ok(())
}

/// Render all visible layers composited (mirrors Renderer::render_layers).
// Keep the GPU resources explicit so their render-pass lifetimes remain clear.
#[allow(clippy::too_many_arguments)]
fn render_layers_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    evaluated: &EvaluatedFramePlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) -> Result<(), String> {
    let visible_layers = export_temporal_wet_stack_indices(evaluated);

    if visible_layers.is_empty() {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[0],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        return Ok(());
    }

    for (i, &layer_index) in visible_layers.iter().enumerate() {
        let evaluated_layer = &evaluated.layers()[layer_index];
        let layer = export_layer_resource(layers, evaluated_layer.source)?;
        let fx_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Layer FX",
            &evaluated.layer_passes()[layer_index],
        );

        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
            ],
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Bottom layer: cleared base + Normal blend with real opacity
        // (mirrors Renderer::render_layers).
        if i == 0 {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Clear Base"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        {
            let comp_uniforms = CompositeUniforms {
                opacity: evaluated_layer.opacity,
                blend_mode: evaluated_layer.blend_mode.as_u32(),
                _pad: [0.0; 2],
            };
            let comp_buffer =
                create_uploaded_uniform(device, queue, "Export Composite Uniforms", &comp_uniforms);

            let composite_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&composite_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            let composite_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: composite_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: comp_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Export Composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &composite_views[2],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(composite_pipeline);
                pass.set_bind_group(0, &composite_tex_bg, &[]);
                pass.set_bind_group(1, &composite_uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[2],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    Ok(())
}

/// Apply master effects to composite_textures[0] (mirrors Renderer::render_master_effects).
// Keep the GPU resources explicit so their render-pass lifetimes remain clear.
#[allow(clippy::too_many_arguments)]
fn render_master_effects_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    pass_uniforms: &EffectPassUniforms,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let fx_buffer = create_uploaded_uniform(device, queue, "Export Master FX", pass_uniforms);

    let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(nearest_sampler),
            },
        ],
    });

    let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: fx_buffer.as_entire_binding(),
        }],
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Master FX"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[2],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(effects_pipeline);
        pass.set_bind_group(0, &tex_bg, &[]);
        pass.set_bind_group(1, &uniform_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[2],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: output_width,
            height: output_height,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{EdgeMode, FitMode, SamplingMode, SpatialGpuUniforms};

    #[test]
    fn temporal_bypass_export_mode_brackets_codec_mosh_only_for_isolated_dry() {
        assert_eq!(
            export_temporal_bypass_mode(TemporalMasterPath::IsolatedDryOverlay, false),
            ExportTemporalBypassMode::BeforeOpaque
        );
        assert_eq!(
            export_temporal_bypass_mode(TemporalMasterPath::IsolatedDryOverlay, true),
            ExportTemporalBypassMode::AroundCodecMosh
        );
        for path in [TemporalMasterPath::Inherited, TemporalMasterPath::LinkedDry] {
            assert_eq!(
                export_temporal_bypass_mode(path, false),
                ExportTemporalBypassMode::Inactive
            );
            assert_eq!(
                export_temporal_bypass_mode(path, true),
                ExportTemporalBypassMode::Inactive
            );
        }
    }

    #[test]
    fn export_codec_mosh_interval_never_resumes_a_retired_engine() {
        let mut was_active = false;
        let mut retained = Some(1_u8);

        update_export_mosh_interval(false, &mut was_active, &mut retained);
        assert!(retained.is_none(), "a dry interval retains no codec owner");

        retained = Some(2);
        update_export_mosh_interval(true, &mut was_active, &mut retained);
        assert!(retained.is_none(), "the first arm is a fresh generation");

        retained = Some(3);
        update_export_mosh_interval(true, &mut was_active, &mut retained);
        assert_eq!(retained, Some(3), "continuous arming keeps codec history");

        update_export_mosh_interval(false, &mut was_active, &mut retained);
        assert!(retained.is_none(), "the dry edge retires codec history");

        retained = Some(4);
        update_export_mosh_interval(true, &mut was_active, &mut retained);
        assert!(
            retained.is_none(),
            "re-arming cannot resume a retired stream"
        );
    }

    #[test]
    fn codec_mosh_sidecar_bounds_the_frame_evaluated_recipe() {
        let first = crate::codec_mosh::CodecMoshParams {
            amount: 0.25,
            wipe: 0.8,
            smear: 0.1,
            trail: 0.4,
            recycle: false,
            ..Default::default()
        };
        let second = crate::codec_mosh::CodecMoshParams {
            amount: 0.75,
            wipe: 0.2,
            smear: 0.9,
            trail: 1.0,
            recycle: true,
            ..Default::default()
        };
        let mut observed = CodecMoshObservationAccumulator::new(first);
        observed.observe(second);
        let observed = observed.finish();
        assert_eq!(observed.accepted_frames, 2);
        assert_eq!(observed.min.amount, 0.25);
        assert_eq!(observed.max.amount, 0.75);
        assert_eq!(observed.min.wipe, 0.2);
        assert_eq!(observed.max.wipe, 0.8);
        assert_eq!(observed.min.smear, 0.1);
        assert_eq!(observed.max.smear, 0.9);
        assert_eq!(observed.min.trail, 0.4);
        assert_eq!(observed.max.trail, 1.0);
        assert!(observed.recycle_false_seen);
        assert!(observed.recycle_true_seen);

        let mut layer = CodecMoshLayerSendObservation::default();
        layer.observe(0.8, false);
        layer.observe(0.2, true);
        layer.observe(f32::NAN, false);
        assert_eq!(layer.min, 0.2);
        assert_eq!(layer.max, 1.0);
        assert!(layer.entered_codec_mosh);
    }

    #[test]
    fn codec_mosh_authored_layer_send_is_pinned_to_the_saved_patch() {
        let mut patch = three_layer_legacy_patch();
        patch.layers[0].mosh_send = 0.25;
        // Export performance replay mutates its detached working layer to this
        // value; authored provenance must remain the saved patch base.
        let replayed_working_send = 0.9_f32;
        assert_ne!(patch.layers[0].mosh_send, replayed_working_send);
        assert_eq!(codec_mosh_authored_layer_send(&patch, 0).unwrap(), 0.25);
        assert!(codec_mosh_authored_layer_send(&patch, patch.layers.len()).is_err());
    }

    #[test]
    fn temporal_bypass_program_candidate_is_lazy_and_checked() {
        let source = include_str!("render_export.rs");
        let start = source
            .find("fn ensure_export_temporal_bypass_program_candidate(")
            .expect("isolated Mosh candidate helper exists");
        let end = source[start..]
            .find("/// Upload one small export uniform")
            .map(|offset| start + offset)
            .expect("candidate helper has a bounded source section");
        let helper = &source[start..end];

        assert!(
            helper.contains("if retained.is_some()"),
            "the full-frame candidate must remain a one-time cold allocation"
        );
        assert!(
            helper.contains("scoped_export_gpu_operation("),
            "candidate allocation must synchronously surface validation/OOM errors"
        );
    }

    fn temporal_bypass_stack_fixture(bypass: [bool; 3]) -> EvaluatedFramePlan {
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(3);
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 36, 0.0),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            bypass
                .into_iter()
                .enumerate()
                .map(|(slot, bypass_temporal_fx)| LayerFrameInput {
                    source: SourceTap::new((slot + 1) as u64, slot, 64, 36),
                    effects: &effects,
                    transform: &transform,
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: slot == 1,
                    bypass_temporal_fx,
                    pattern: None,
                }),
        )
    }

    #[test]
    fn temporal_bypass_export_partitions_wet_and_dry_in_stack_order() {
        let inherited = temporal_bypass_stack_fixture([false; 3]);
        assert_eq!(export_temporal_wet_stack_indices(&inherited), vec![2, 1, 0]);
        assert!(temporal_bypass_overlay_indices(&inherited).is_empty());
        assert!(export_may_publish_legacy_program_history(&inherited));

        // UI slots 0 and 1 form the admitted top prefix. Slot 2 remains wet;
        // the dry overlay is reapplied from its lower member to UI Layer 1.
        let isolated = temporal_bypass_stack_fixture([true, true, false]);
        assert_eq!(export_temporal_wet_stack_indices(&isolated), vec![2]);
        assert!(export_temporal_wet_layer_contributes(&isolated));
        assert_eq!(temporal_bypass_overlay_indices(&isolated), vec![1, 0]);
        assert_eq!(
            export_temporal_bypass_overlay_sources(&isolated)
                .into_iter()
                .map(|(layer_index, source)| (layer_index, source.slot, source.stable_id))
                .collect::<Vec<_>>(),
            vec![(1, 1, 2), (0, 0, 1)]
        );
        assert!(!export_may_publish_legacy_program_history(&isolated));

        let all_dry = temporal_bypass_stack_fixture([true; 3]);
        assert!(export_temporal_wet_stack_indices(&all_dry).is_empty());
        assert!(!export_temporal_wet_layer_contributes(&all_dry));

        let mut previous = Vec::new();
        let mut initialized = false;
        assert!(!observe_export_temporal_bypass_partition(
            &mut previous,
            &mut initialized,
            &inherited,
        ));
        assert!(!observe_export_temporal_bypass_partition(
            &mut previous,
            &mut initialized,
            &inherited,
        ));
        assert!(observe_export_temporal_bypass_partition(
            &mut previous,
            &mut initialized,
            &isolated,
        ));
        assert!(!observe_export_temporal_bypass_partition(
            &mut previous,
            &mut initialized,
            &isolated,
        ));
        assert!(observe_export_temporal_bypass_partition(
            &mut previous,
            &mut initialized,
            &inherited,
        ));
    }

    #[test]
    fn temporal_bypass_export_preflights_before_partition_and_temporal_encoding() {
        let source = include_str!("render_export.rs");
        let run_export = &source[source
            .find("fn run_export(")
            .expect("offline export entry point exists")..];
        let preflight = run_export
            .find("match preflight_temporal_bypass_overlay_resources(")
            .expect("isolated dry resources are preflighted");
        let candidate = run_export
            .find("ensure_export_temporal_bypass_program_candidate(")
            .expect("isolated Mosh candidate is allocated during preflight");
        let mosh_interval = run_export
            .find("update_export_mosh_interval(")
            .expect("Codec Mosh interval is observed before stateful rendering");
        let partition = run_export
            .find("if observe_export_temporal_bypass_partition(")
            .expect("bypass partition is observed");
        let temporal_encode = run_export
            .find("encode_temporal_prepared_frame(")
            .expect("legacy Temporal is encoded");
        assert!(
            preflight < candidate
                && candidate < mosh_interval
                && mosh_interval < partition
                && partition < temporal_encode,
            "all cold resources and codec edges must precede every stateful partition/Temporal operation"
        );

        let partition_reset = &run_export[partition..temporal_encode];
        assert!(
            partition_reset.contains("resources.history_valid = false;"),
            "every partition edge must invalidate retained legacy ProgramHistory"
        );
    }

    #[test]
    fn advanced_temporal_bypass_export_reuses_retained_surfaces_on_both_mosh_paths() {
        let source = include_str!("render_export.rs").replace("\r\n", "\n");
        let helper_start = source
            .find("fn render_advanced_temporal_bypass_overlay_export(")
            .expect("Advanced dry-overlay helper");
        let helper_end = source[helper_start..]
            .find("fn render_planned_temporal_bypass_overlay_export(")
            .map(|offset| helper_start + offset)
            .expect("Advanced dry-overlay helper boundary");
        let helper = &source[helper_start..helper_end];
        let compact = helper
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(compact.contains(".temporal_bypass_overlays(advanced)"));
        assert!(compact.contains("retained.len()!=expected.len()"));
        assert!(compact.contains("expected.get(index).copied()!=Some(retained.stable_id)"));
        assert!(compact.contains(".get(layer_plan.base_layer_index)"));
        assert!(
            !helper.contains("ExportLayer") && !helper.contains("texture_view"),
            "Advanced export must never substitute a raw legacy source for a retained layer"
        );

        let run_start = source.find("fn run_export(").expect("offline export entry");
        let run_end = source[run_start..]
            .find("/// Readback composite_textures[0]")
            .map(|offset| run_start + offset)
            .expect("offline export body boundary");
        let run = &source[run_start..run_end];
        assert_eq!(
            run.matches("render_planned_temporal_bypass_overlay_export(")
                .count(),
            4,
            "Advanced/Legacy no-Mosh and pre/post-Mosh seams must all use one dispatcher"
        );

        // Advanced no-Mosh: shared executor output -> external Temporal family
        // -> retained dry surfaces -> the sole opaque audience boundary.
        let advanced = run
            .find("// The shared executor owns LegacyTemporal for Advanced")
            .expect("Advanced presentation handoff");
        let before_opaque = run[advanced..]
            .find("if temporal_bypass_mode == ExportTemporalBypassMode::BeforeOpaque")
            .map(|offset| advanced + offset)
            .expect("Advanced before-opaque policy");
        let advanced_overlay = run[before_opaque..]
            .find("render_planned_temporal_bypass_overlay_export(")
            .map(|offset| before_opaque + offset)
            .expect("Advanced retained overlay call");
        let advanced_opaque = run[advanced_overlay..]
            .find("crate::renderer::state::encode_opaque_output(")
            .map(|offset| advanced_overlay + offset)
            .expect("Advanced opaque boundary");
        assert!(before_opaque < advanced_overlay && advanced_overlay < advanced_opaque);

        // Around Codec Mosh: reconstruct the complete pre-Mosh programme for
        // analysis/tap provenance, but feed only the wet readback to the CPU;
        // then upload that replacement and reuse the same retained surfaces.
        let pre_label = run
            .find("label: Some(\"Export Pre-Mosh Temporal Bypass Overlay\")")
            .expect("pre-Mosh overlay encoder");
        let pre_overlay = run[pre_label..]
            .find("render_planned_temporal_bypass_overlay_export(")
            .map(|offset| pre_label + offset)
            .expect("pre-Mosh retained overlay");
        let candidate_copy = run[pre_overlay..]
            .find("texture: candidate,")
            .map(|offset| pre_overlay + offset)
            .expect("pre-Mosh candidate publication");
        assert!(pre_label < pre_overlay && pre_overlay < candidate_copy);

        let wet_upload = run
            .find("\"Export Moshed Temporal Wet Program\"")
            .expect("CPU Mosh wet upload");
        let post_label = run[wet_upload..]
            .find("label: Some(\"Export Post-Mosh Temporal Bypass Overlay\")")
            .map(|offset| wet_upload + offset)
            .expect("post-Mosh overlay encoder");
        let post_overlay = run[post_label..]
            .find("render_planned_temporal_bypass_overlay_export(")
            .map(|offset| post_label + offset)
            .expect("post-Mosh retained overlay");
        let final_opaque = run[post_overlay..]
            .find("crate::renderer::state::encode_opaque_output(")
            .map(|offset| post_overlay + offset)
            .expect("post-Mosh opaque boundary");
        assert!(
            wet_upload < post_label && post_label < post_overlay && post_overlay < final_opaque
        );
    }

    #[test]
    fn temporal_bypass_all_dry_export_skips_the_empty_global_master_pass() {
        let source = include_str!("render_export.rs");
        let start = source
            .find("fn render_layers_and_master_export(")
            .expect("offline exact compositor exists");
        let end = source[start..]
            .find("fn render_temporal_bypass_overlay_export(")
            .map(|offset| start + offset)
            .expect("offline exact compositor has a bounded source section");
        let compositor = &source[start..end];

        assert!(
            compositor.contains(
                "if !isolated_temporal_overlay || export_temporal_wet_layer_contributes(evaluated)"
            ),
            "an all-dry isolated stack must preserve its transparent wet base without a global Master pass"
        );
    }

    fn config(duration_secs: f32) -> ExportConfig {
        ExportConfig {
            width: 1920,
            height: 1080,
            fps: 24,
            duration_secs,
            output_path: "render.mp4".to_owned(),
            audio_path: None,
            audio_path_hint: None,
            layer_source_hints: Vec::new(),
            analysis_audio_path_hint: None,
            ntsc_quality: NtscExportQuality::LiveParity,
            shutter_samples: ExportShutterSamples::Authored,
            media_safety_policy: MediaSafetyPolicy::default(),
            temporal_event_track: crate::temporal::TemporalEventTrack::default(),
            gesture_track: None,
            performance_take: None,
        }
    }

    fn temporal_event_acceptance_params() -> crate::effects::params::TemporalParams {
        use crate::temporal::{
            CollisionAtlasParams, CollisionScoreLoopDriver, CollisionScoreParams,
            CollisionScoreTrigger, LongExposureParams, RefreshGardenGate, RefreshGardenParams,
            TemporalEventResetMode, TemporalInterpolation, TemporalLoomParams,
            TemporalOriginalsParams, TemporalResetPolicy, TemporalTopology,
        };

        crate::effects::params::TemporalParams {
            feedback: 0.63,
            fb_zoom: 1.01,
            slitscan: 0.42,
            originals: TemporalOriginalsParams {
                loom: TemporalLoomParams {
                    amount: 0.75,
                    topology: TemporalTopology::Spiral,
                    interpolation: TemporalInterpolation::Linear,
                    depth: 0.8,
                    phase: 0.125,
                    scale: 1.4,
                    angle: 19.0,
                    folds: 5,
                    quantization: 9,
                },
                atlas: CollisionAtlasParams {
                    amount: 0.4,
                    seed: 0,
                    territories: 11,
                    collision: 0.7,
                },
                garden: RefreshGardenParams {
                    amount: 0.35,
                    gate: RefreshGardenGate::AudioOnset,
                    threshold: 0.45,
                    softness: 0.1,
                    decay: 0.91,
                    max_hold_ticks: 17,
                    ..RefreshGardenParams::default()
                },
                long_exposure: LongExposureParams {
                    amount: 0.62,
                    shutter_frames: 17,
                },
                score: CollisionScoreParams {
                    enabled: true,
                    seed: 0x51c0_4e11,
                    state_count: 7,
                    trigger: CollisionScoreTrigger::Boundary,
                    loop_driver: CollisionScoreLoopDriver::SelectedLayer {
                        layer_id: StableLayerId::new(22).unwrap(),
                        saved_position: SavedLayerPosition::new(1).unwrap(),
                    },
                },
                reset: TemporalResetPolicy {
                    loop_boundary: TemporalEventResetMode::Memory,
                    downbeat: TemporalEventResetMode::Score,
                },
            },
            ..crate::effects::params::TemporalParams::default()
        }
    }

    fn temporal_acceptance_freeze(frame: u32, fps: u32) -> (bool, bool) {
        if (fps / 2..fps * 3 / 4).contains(&frame) {
            (true, false)
        } else if (fps..fps * 5 / 4).contains(&frame) {
            (false, true)
        } else if (fps * 3 / 2..fps * 7 / 4).contains(&frame) {
            (true, true)
        } else {
            (false, false)
        }
    }

    /// Exercise independently-owned live/export trackers and states while
    /// comparing the complete production frame input and staged plan at every
    /// accepted boundary. The returned bytes make a second run a deterministic
    /// replay proof rather than merely a final-state comparison.
    fn temporal_live_export_acceptance_trace(fps: u32) -> Vec<u8> {
        use crate::performance::BeatBoundaryTracker;
        use crate::renderer::state::TemporalState;
        use crate::temporal::{
            TemporalAudioOnsetTracker, TemporalFrameEvents, TemporalFrameInput, TemporalFreezeState,
        };

        let params = temporal_event_acceptance_params();
        let mut live_state = TemporalState::default();
        let mut export_state = TemporalState::default();
        let mut live_beats = BeatBoundaryTracker::default();
        let mut export_beats = BeatBoundaryTracker::default();
        let mut live_onsets = TemporalAudioOnsetTracker::default();
        let mut export_onsets = TemporalAudioOnsetTracker::default();
        live_beats.reanchor(0.0);
        export_beats.reanchor(0.0);
        live_onsets.reanchor(0.0);
        export_onsets.reanchor(0.0);
        let mut live_beat = 0.0_f64;
        let mut export_beat = 0.0_f64;
        let mut trace = Vec::new();
        let mut saw = [false; 4];
        let mut consumed_boundaries = 0_u64;

        for frame in 0..fps * 4 {
            let (master_paused, media_frozen) = temporal_acceptance_freeze(frame, fps);
            let live_freeze = crate::temporal_freeze_state(!master_paused, media_frozen);
            let export_freeze = match (master_paused, media_frozen) {
                (false, false) => TemporalFreezeState::Running,
                (false, true) => TemporalFreezeState::MediaFrozen,
                (true, false) => TemporalFreezeState::ProgramFrozen,
                (true, true) => TemporalFreezeState::ProgramAndMediaFrozen,
            };
            assert_eq!(live_freeze, export_freeze);
            saw[match live_freeze {
                TemporalFreezeState::Running => 0,
                TemporalFreezeState::ProgramFrozen => 1,
                TemporalFreezeState::MediaFrozen => 2,
                TemporalFreezeState::ProgramAndMediaFrozen => 3,
            }] = true;

            let delta = 1.0 / fps as f32;
            if live_freeze.program_advances() {
                live_beat += 2.0 / f64::from(fps);
            }
            if export_freeze.program_advances() {
                export_beat += 2.0 / f64::from(fps);
            }
            let live_crossings = live_beats.observe(live_beat, 4, !live_freeze.program_advances());
            let export_crossings =
                export_beats.observe(export_beat, 4, !export_freeze.program_advances());
            assert_eq!(live_crossings, export_crossings);

            let envelope_phase = frame % (fps / 2);
            let audio_energy = if (fps / 8..fps / 4).contains(&envelope_phase) {
                0.8
            } else {
                0.1
            };
            let live_audio_events =
                live_onsets.observe(audio_energy, live_freeze.program_advances());
            let export_audio_events =
                export_onsets.observe(audio_energy, export_freeze.program_advances());
            assert_eq!(live_audio_events, export_audio_events);

            // This is the selected stable layer's transport event stream. A
            // double crossing proves counts are retained, not collapsed.
            let boundary_events = if frame > 0
                && frame % (fps / 2) == 0
                && live_freeze.program_advances()
                && live_freeze.media_advances()
            {
                if frame == fps * 2 {
                    2
                } else {
                    1
                }
            } else {
                0
            };
            let events = TemporalFrameEvents {
                boundary_events,
                downbeat_events: live_crossings.bars,
                audio_onset_events: live_audio_events,
                manual_events: u32::from(frame == fps * 3 + 1),
                garden_refresh_events: 0,
            };
            let blackout = (fps * 2..fps * 9 / 4).contains(&frame);
            let live_input = TemporalFrameInput::new(delta, live_freeze, blackout, events)
                .with_audio_energy(audio_energy);
            let export_input = TemporalFrameInput::new(
                delta,
                export_freeze,
                blackout,
                TemporalFrameEvents {
                    boundary_events,
                    downbeat_events: export_crossings.bars,
                    audio_onset_events: export_audio_events,
                    manual_events: u32::from(frame == fps * 3 + 1),
                    garden_refresh_events: 0,
                },
            )
            .with_audio_energy(audio_energy);
            assert_eq!(live_input, export_input);

            let live_plan = live_state.stage_frame(&params, live_input, [320, 180]);
            let export_plan = export_state.stage_frame(&params, export_input, [320, 180]);
            assert_eq!(live_plan, export_plan, "frame {frame} at {fps} fps");
            assert_eq!(
                live_plan.originals_uniforms.long_exposure_values,
                [0.62, 17.0, 0.0, 0.0],
                "live Long Exposure uniforms drifted at frame {frame}, {fps} fps"
            );
            assert_eq!(
                export_plan.originals_uniforms.long_exposure_values,
                live_plan.originals_uniforms.long_exposure_values,
                "export Long Exposure uniforms diverged at frame {frame}, {fps} fps"
            );
            consumed_boundaries =
                consumed_boundaries.saturating_add(u64::from(live_plan.score_events_consumed));
            live_state.commit_staged();
            export_state.commit_staged();
            assert_eq!(live_state.metrics(), export_state.metrics());
            trace.extend_from_slice(
                format!(
                    "{frame}:{live_input:?}:{live_plan:?}:{:?}\n",
                    live_state.metrics()
                )
                .as_bytes(),
            );
        }

        assert!(saw.into_iter().all(std::convert::identity));
        assert!(consumed_boundaries >= 4);
        trace
    }

    #[test]
    fn m3_event_acceptance_live_export_long_exposure_uniforms_score_resets_and_freezes_replay_at_24_30_60(
    ) {
        for fps in [24_u32, 30, 60] {
            let first = temporal_live_export_acceptance_trace(fps);
            let replay = temporal_live_export_acceptance_trace(fps);
            assert_eq!(first, replay, "non-deterministic {fps} fps trace");
        }
    }

    #[test]
    fn m3_event_acceptance_program_media_freeze_and_blackout_hidden_evolution() {
        use crate::renderer::state::TemporalState;
        use crate::temporal::{
            CollisionScoreParams, CollisionScoreTrigger, TemporalFrameAction, TemporalFrameEvents,
            TemporalFrameInput, TemporalFreezeState,
        };

        for (program_running, media_frozen, expected) in [
            (true, false, TemporalFreezeState::Running),
            (true, true, TemporalFreezeState::MediaFrozen),
            (false, false, TemporalFreezeState::ProgramFrozen),
            (false, true, TemporalFreezeState::ProgramAndMediaFrozen),
        ] {
            assert_eq!(
                crate::temporal_freeze_state(program_running, media_frozen),
                expected
            );
        }

        let mut params = crate::effects::params::TemporalParams::default();
        params.originals.score = CollisionScoreParams {
            enabled: true,
            state_count: 8,
            trigger: CollisionScoreTrigger::Manual,
            ..CollisionScoreParams::default()
        };
        let mut seeded = TemporalState::default();
        seeded.stage_frame(
            &params,
            TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::Running,
                false,
                TemporalFrameEvents::default(),
            ),
            [64, 36],
        );
        seeded.commit_staged();
        let seeded_metrics = seeded.metrics();

        for freeze in [
            TemporalFreezeState::ProgramFrozen,
            TemporalFreezeState::ProgramAndMediaFrozen,
        ] {
            let mut state = seeded.clone();
            let plan = state.stage_frame(
                &params,
                TemporalFrameInput::new(
                    1.0 / 30.0,
                    freeze,
                    false,
                    TemporalFrameEvents {
                        manual_events: 2,
                        ..TemporalFrameEvents::default()
                    },
                ),
                [64, 36],
            );
            assert!(matches!(plan.action, TemporalFrameAction::HoldFrozenOutput));
            assert_eq!(plan.score_events_consumed, 0);
            state.commit_staged();
            assert_eq!(state.metrics(), seeded_metrics);
        }

        let mut media_frozen = seeded.clone();
        let plan = media_frozen.stage_frame(
            &params,
            TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::MediaFrozen,
                false,
                TemporalFrameEvents {
                    manual_events: 2,
                    ..TemporalFrameEvents::default()
                },
            ),
            [64, 36],
        );
        assert!(matches!(plan.action, TemporalFrameAction::Advance { .. }));
        assert_eq!(plan.score_events_consumed, 2);
        assert!(!TemporalFreezeState::MediaFrozen.media_advances());
        assert!(TemporalFreezeState::MediaFrozen.program_advances());
        media_frozen.commit_staged();
        assert_eq!(media_frozen.metrics().score_event_ordinal, 2);

        let mut visible = seeded.clone();
        let mut hidden = seeded;
        for events in [
            TemporalFrameEvents::default(),
            TemporalFrameEvents {
                manual_events: 3,
                ..TemporalFrameEvents::default()
            },
        ] {
            let visible_plan = visible.stage_frame(
                &params,
                TemporalFrameInput::new(1.0 / 60.0, TemporalFreezeState::Running, false, events),
                [64, 36],
            );
            let hidden_plan = hidden.stage_frame(
                &params,
                TemporalFrameInput::new(1.0 / 60.0, TemporalFreezeState::Running, true, events),
                [64, 36],
            );
            assert_eq!(visible_plan, hidden_plan);
            visible.commit_staged();
            hidden.commit_staged();
            assert_eq!(visible.metrics(), hidden.metrics());
        }
    }

    #[test]
    fn m3_event_acceptance_boundary_driver_reorder_delete_and_tombstone() {
        use crate::patch::CollisionScoreLoopDriverConfig as Saved;
        use crate::temporal::CollisionScoreLoopDriver as Runtime;

        let one = StableLayerId::new(11).unwrap();
        let driver = StableLayerId::new(22).unwrap();
        let three = StableLayerId::new(33).unwrap();
        let stale_position = SavedLayerPosition::new(1).unwrap();
        let selected = Runtime::SelectedLayer {
            layer_id: driver,
            saved_position: stale_position,
        };

        let reordered = [three, one, driver];
        let captured = Saved::from_runtime_for_capture(selected, &reordered);
        assert_eq!(
            captured,
            Saved::SelectedLayer {
                saved_position: SavedLayerPosition::new(2).unwrap()
            }
        );
        assert_eq!(
            resolve_export_score_loop_driver(captured, &reordered),
            Runtime::SelectedLayer {
                layer_id: driver,
                saved_position: SavedLayerPosition::new(2).unwrap(),
            }
        );

        let after_unrelated_delete = [one, driver];
        let captured = Saved::from_runtime_for_capture(selected, &after_unrelated_delete);
        assert_eq!(
            resolve_export_score_loop_driver(captured, &after_unrelated_delete),
            Runtime::SelectedLayer {
                layer_id: driver,
                saved_position: SavedLayerPosition::new(1).unwrap(),
            }
        );

        let without_driver = [three, one];
        let tombstone = Saved::from_runtime_for_capture(selected, &without_driver);
        assert_eq!(
            tombstone,
            Saved::MissingSelectedLayer {
                saved_position: stale_position
            }
        );
        assert_eq!(
            resolve_export_score_loop_driver(
                tombstone,
                &[three, one, StableLayerId::new(99).unwrap()],
            ),
            Runtime::MissingSelectedLayer {
                saved_position: stale_position
            },
            "a replacement at the old saved position must never steal the conductor"
        );
    }

    #[test]
    fn m3_event_acceptance_export_prepares_exact_once_with_full_originals_input() {
        let source = include_str!("render_export.rs");
        let prepared = source
            .find("let temporal_prepared = crate::renderer::state::build_prepared_temporal_gpu_resources")
            .expect("prepared Exact resources");
        let frame_loop = source
            .find("for frame_num in 0..total_frames")
            .expect("export frame loop");
        assert!(
            prepared < frame_loop,
            "prepared bindings must precede warmed frames"
        );
        let encode = source
            .find("crate::renderer::state::encode_temporal_prepared_frame(")
            .expect("prepared full-input encoder");
        assert!(encode > frame_loop);
        let call = &source[encode
            ..source[encode..]
                .find(");")
                .map(|end| encode + end + 2)
                .expect("prepared encoder call terminator")];
        assert!(call.contains("&mod_temporal"));
        assert!(call.contains("temporal_input"));
        assert!(
            !call.contains("program_dt,\n"),
            "the compatibility dt/bool adapter must stay out of export"
        );
    }

    fn three_layer_legacy_patch() -> PatchState {
        serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: top.mp4
  - filename: middle.mp4
  - filename: bottom.mp4
"#,
        )
        .unwrap()
    }

    #[test]
    fn export_resolves_patch_and_morph_garden_routes_against_job_local_ids() {
        use crate::image_routing::LayerImageStage;
        use crate::patch::{
            RefreshGardenMatteRouteConfig, RefreshGardenMotionRouteConfig, TemporalOriginalsConfig,
        };
        use crate::temporal::{RefreshGardenMatteRoute, RefreshGardenMotionRoute};

        let ids = [
            StableLayerId::new(1).unwrap(),
            StableLayerId::new(2).unwrap(),
        ];
        let zero = SavedLayerPosition::new(0).unwrap();
        let one = SavedLayerPosition::new(1).unwrap();
        let mut config = crate::patch::TemporalConfig::default();
        let originals = config
            .originals
            .get_or_insert_with(TemporalOriginalsConfig::default);
        originals.garden.matte_route = RefreshGardenMatteRouteConfig::SelectedLayer {
            saved_position: one,
            stage: LayerImageStage::PreLocalEffects,
        };
        originals.garden.motion_route = RefreshGardenMotionRouteConfig::MissingSelectedLayer {
            saved_position: zero,
        };
        let patch = resolved_export_patch_temporal(Some(&config), &ids);
        assert_eq!(
            patch.originals.garden.matte_route,
            RefreshGardenMatteRoute::SelectedLayer {
                layer_id: ids[1],
                saved_position: one,
                stage: LayerImageStage::PreLocalEffects,
            }
        );
        assert_eq!(
            patch.originals.garden.motion_route,
            RefreshGardenMotionRoute::MissingSelectedLayer {
                saved_position: zero,
            }
        );

        let mut morph = crate::morph::MorphTemporalSnapshot::default();
        morph.originals.garden.matte_route = RefreshGardenMatteRouteConfig::MissingSelectedLayer {
            saved_position: one,
            stage: LayerImageStage::PostLocalEffects,
        };
        morph.originals.garden.motion_route = RefreshGardenMotionRouteConfig::SelectedLayer {
            saved_position: zero,
        };
        let sampled = resolved_export_morph_temporal(morph, &ids);
        assert_eq!(
            sampled.originals.garden.matte_route,
            RefreshGardenMatteRoute::MissingSelectedLayer {
                saved_position: one,
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            sampled.originals.garden.motion_route,
            RefreshGardenMotionRoute::SelectedLayer {
                layer_id: ids[0],
                saved_position: zero,
            }
        );
    }

    fn garden_export_plan(
        graph: &ExportCreativeGraph,
        temporal: &crate::effects::params::TemporalParams,
    ) -> EvaluatedCompositionPlan {
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(graph.layer_ids.len());
        let evaluated = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 36, 0.0),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &ntsc,
                temporal,
            },
            graph
                .layer_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(slot, id)| LayerFrameInput {
                    source: SourceTap::new(id.get(), slot, 64, 36),
                    effects: &effects,
                    transform: &transform,
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                    pattern: None,
                }),
        );
        plan_export_composition(
            &evaluated,
            graph,
            &vec![LayerMatte::default(); graph.layer_ids.len()],
            false,
            CreativeResourceLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn export_surfaces_closed_garden_matte_and_motion_route_warnings() {
        let graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        let mut temporal = crate::effects::params::TemporalParams::default();
        temporal.originals.garden.amount = 1.0;
        temporal.originals.garden.gate = crate::temporal::RefreshGardenGate::Matte;
        let matte = export_motion_plan_warnings(&garden_export_plan(&graph, &temporal));
        assert!(matte
            .iter()
            .any(|warning| warning.contains("no selected matte route")));

        temporal.originals.garden.gate = crate::temporal::RefreshGardenGate::Motion;
        temporal.originals.garden.motion_route =
            crate::temporal::RefreshGardenMotionRoute::MissingSelectedLayer {
                saved_position: SavedLayerPosition::new(2).unwrap(),
            };
        let motion = export_motion_plan_warnings(&garden_export_plan(&graph, &temporal));
        assert!(motion.iter().any(|warning| {
            warning.contains("missing saved motion route position 2")
                && warning.contains("resolves to zero")
        }));
    }

    fn dormant_cycle_saved_rack(
        amount: f32,
        donor_position: u32,
    ) -> crate::visual_rack::VisualRack {
        use crate::image_routing::LayerImageStage;
        use crate::visual_rack::{
            EdgeTiming, ImageMatte, MaskParams, SavedImageSource, SavedImageTap, VisualNodeKind,
            VisualRack,
        };

        let mut rack = VisualRack::synthetic_legacy(crate::visual_rack::LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
            tap: SavedImageTap {
                source: SavedImageSource::SelectedLayer {
                    layer_position: SavedLayerPosition::new(donor_position).unwrap(),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::CurrentFrame,
            },
            amount,
            ..ImageMatte::default()
        })))
        .unwrap();
        rack
    }

    fn dormant_cycle_morph_fixture() -> (crate::morph::Morph, ExportCreativeGraph) {
        use crate::composition::{BusAssignment, RuntimeComposition, RuntimeRootItem};
        use crate::modulation::StableModAddressBook;
        use crate::visual_rack::LegacyRackScope;

        let ids = [
            StableLayerId::new(1).unwrap(),
            StableLayerId::new(2).unwrap(),
        ];
        let a_racks = vec![
            dormant_cycle_saved_rack(1.0, 1),
            dormant_cycle_saved_rack(0.0, 0),
        ];
        let b_racks = vec![
            dormant_cycle_saved_rack(0.0, 1),
            dormant_cycle_saved_rack(1.0, 0),
        ];
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: ids[1],
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: ids[0],
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        let resolve = |position: SavedLayerPosition| ids.get(position.get() as usize).copied();
        let layer_racks = a_racks
            .iter()
            .enumerate()
            .map(|(index, rack)| (ids[index], rack.resolve_routes(resolve, |_| false)))
            .collect::<Vec<_>>();
        let master_rack =
            crate::visual_rack::RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let address_book =
            StableModAddressBook::from_composition(&master_rack, &layer_racks, &composition)
                .unwrap();
        let graph = ExportCreativeGraph {
            studies: crate::study_eval::StudyProgramLibrary::default(),
            layer_ids: ids.to_vec().into_boxed_slice(),
            master_rack,
            layer_racks,
            composition,
            address_book,
        };
        let mut a = crate::morph::MorphSlot {
            layer_racks: Some(a_racks),
            ..crate::morph::MorphSlot::default()
        };
        a.master.brightness = -1.0;
        let mut b = crate::morph::MorphSlot {
            layer_racks: Some(b_racks),
            ..crate::morph::MorphSlot::default()
        };
        b.master.brightness = 1.0;
        (
            crate::morph::Morph {
                a: Some(a),
                b: Some(b),
                ..crate::morph::Morph::default()
            },
            graph,
        )
    }

    fn validate_dormant_cycle_sample(
        morph: &crate::morph::Morph,
        baseline: &ExportCreativeGraph,
        position: f32,
    ) -> Result<(f32, u64, usize), String> {
        let sample = morph
            .sample(position)
            .ok_or_else(|| "missing Morph sample".to_string())?;
        let mut graph = baseline.clone();
        apply_export_creative_morph(&sample, &mut graph);
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(2);
        let evaluated = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 36, position),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            graph
                .layer_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(slot, id)| LayerFrameInput {
                    source: SourceTap::new(id.get(), slot, 64, 36),
                    effects: &effects,
                    transform: &transform,
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                    pattern: None,
                }),
        );
        let plan = plan_export_composition(
            &evaluated,
            &graph,
            &[LayerMatte::default(); 2],
            false,
            CreativeResourceLimits::default(),
        )?;
        let EvaluatedCompositionPlan::Advanced(advanced) = plan else {
            return Err("dormant cycle fixture unexpectedly delegated legacy".into());
        };
        Ok((
            sample.master.brightness,
            advanced.topology_signature(),
            advanced.image_taps().len(),
        ))
    }

    #[test]
    fn morph_dormant_cycle_fallback_is_frame_independent_at_24_30_60() {
        let (morph, baseline) = dormant_cycle_morph_fixture();
        assert!(validate_dormant_cycle_sample(&morph, &baseline, 0.5).is_err());
        assert!(validate_dormant_cycle_sample(&morph, &baseline, 0.0).is_ok());
        assert!(validate_dormant_cycle_sample(&morph, &baseline, 1.0).is_ok());

        for fps in [24_u32, 30, 60] {
            for (frame, expected_position, expected_brightness) in
                [(fps / 3, 0.0_f32, -1.0_f32), (fps / 2, 1.0, 1.0)]
            {
                let requested = frame as f32 / fps as f32;
                let export = select_valid_morph_candidate(requested, |position| {
                    validate_dormant_cycle_sample(&morph, &baseline, position)
                })
                .unwrap();

                // Independent reproduction of Main's detached live attempt
                // order: requested, nearest (tie B), then other.
                let nearest = if requested < 0.5 { 0.0 } else { 1.0 };
                let other = 1.0 - nearest;
                let live = [requested, nearest, other]
                    .into_iter()
                    .enumerate()
                    .filter(|(index, position)| {
                        *index == 0
                            || (*position - requested).abs() > f32::EPSILON
                                && (*index != 2 || (*position - nearest).abs() > f32::EPSILON)
                    })
                    .find_map(|(_, position)| {
                        validate_dormant_cycle_sample(&morph, &baseline, position)
                            .ok()
                            .map(|value| (position, value))
                    })
                    .unwrap();

                assert_eq!(export.selected_position, expected_position);
                assert_eq!(export.value.0, expected_brightness);
                assert_eq!(
                    (export.selected_position, export.value),
                    live,
                    "live/export Morph fallback diverged at {fps} fps"
                );
                assert_eq!(export.value.2, 1, "accepted endpoint has one active edge");
            }
        }
    }

    #[test]
    fn export_creative_planning_preserves_exact_legacy_at_24_30_and_60_fps() {
        let patch = three_layer_legacy_patch();
        let graph = resolve_export_creative_graph(&patch).unwrap();
        assert_eq!(
            graph.layer_ids.as_ref(),
            &[
                StableLayerId::new(1).unwrap(),
                StableLayerId::new(2).unwrap(),
                StableLayerId::new(3).unwrap(),
            ]
        );
        assert_eq!(
            graph
                .composition
                .flatten()
                .unwrap()
                .layers
                .iter()
                .map(|layer| layer.layer_id)
                .collect::<Vec<_>>(),
            vec![
                StableLayerId::new(3).unwrap(),
                StableLayerId::new(2).unwrap(),
                StableLayerId::new(1).unwrap(),
            ]
        );

        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let mut signature = None;
        let mut advanced_signature = None;
        for fps in [24_u32, 30, 60] {
            let modulation = crate::modulation::ModMatrix::new().frame(3);
            let evaluated = EvaluatedFramePlan::evaluate(
                &modulation,
                FramePlanContext::new(64, 36, fps as f32 / fps as f32),
                MasterFrameInput {
                    effects: &effects,
                    transform: &transform,
                    ntsc: &ntsc,
                    temporal: &temporal,
                },
                graph
                    .layer_ids
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(slot, id)| LayerFrameInput {
                        source: SourceTap::new(id.get(), slot, 64, 36),
                        effects: &effects,
                        transform: &transform,
                        opacity: 1.0,
                        mosh_send: 1.0,
                        speed: 1.0,
                        fps: fps as f32,
                        blend_mode: BlendMode::Normal,
                        visible: true,
                        paused: false,
                        bypass_master_fx: false,
                        bypass_temporal_fx: false,
                        pattern: None,
                    }),
            );
            let planned = plan_export_composition(
                &evaluated,
                &graph,
                &[LayerMatte::default(); 3],
                false,
                CreativeResourceLimits::default(),
            )
            .unwrap();
            let EvaluatedCompositionPlan::LegacyExact(exact) = planned else {
                panic!("omitted creative fields must keep the exact legacy export path");
            };
            assert_eq!(
                exact.flattened_layers(),
                &[
                    StableLayerId::new(3).unwrap(),
                    StableLayerId::new(2).unwrap(),
                    StableLayerId::new(1).unwrap(),
                ]
            );
            if let Some(expected) = signature {
                assert_eq!(exact.topology_signature(), expected);
            } else {
                signature = Some(exact.topology_signature());
            }
            assert_eq!(
                bytemuck::bytes_of(&exact.base().master_pass_uniforms()),
                bytemuck::bytes_of(&evaluated.master_pass_uniforms())
            );

            let planned = plan_export_composition(
                &evaluated,
                &graph,
                &[
                    LayerMatte {
                        enabled: true,
                        input: crate::image_routing::ImageInput::OneBelow,
                        ..LayerMatte::default()
                    },
                    LayerMatte::default(),
                    LayerMatte::default(),
                ],
                false,
                CreativeResourceLimits::default(),
            )
            .unwrap();
            let EvaluatedCompositionPlan::Advanced(advanced) = planned else {
                panic!("an authored matte must enter the unified advanced plan");
            };
            assert!(evaluated.image_routing().mattes().is_empty());
            assert_eq!(
                advanced.image_taps()[0].resolved,
                crate::evaluated_frame::evaluated_composition::PlannedImageSource::OneBelow(
                    crate::visual_rack::VisualScopeId::Layer(StableLayerId::new(2).unwrap())
                )
            );
            if let Some(expected) = advanced_signature {
                assert_eq!(advanced.topology_signature(), expected);
            } else {
                advanced_signature = Some(advanced.topology_signature());
            }
        }

        // Final-program VHS is independent of the Advanced topology and
        // per-layer Master bypass. Mixed stacks must therefore remain
        // admissible instead of being routed into the old flat slice path.
        let selective_ntsc = crate::ntsc::NtscParams {
            enabled: true,
            ..crate::ntsc::NtscParams::default()
        };
        let modulation = crate::modulation::ModMatrix::new().frame(0);
        let selective = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 36, 0.0),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &selective_ntsc,
                temporal: &temporal,
            },
            graph
                .layer_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(slot, id)| LayerFrameInput {
                    source: SourceTap::new(id.get(), slot, 64, 36),
                    effects: &effects,
                    transform: &transform,
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: slot == 0,
                    bypass_temporal_fx: false,
                    pattern: None,
                }),
        );
        let selective_advanced = plan_export_composition(
            &selective,
            &graph,
            &[
                LayerMatte {
                    enabled: true,
                    input: crate::image_routing::ImageInput::OneBelow,
                    ..LayerMatte::default()
                },
                LayerMatte::default(),
                LayerMatte::default(),
            ],
            false,
            CreativeResourceLimits::default(),
        )
        .unwrap();
        CompositionGpuExecutor::validate_plan(&selective_advanced)
            .expect("final-program VHS admits mixed Advanced Master bypass");
    }

    /// One Residual Counterpoint node whose two slots are named independently.
    /// Every discrete field is deliberately off its default so a value that
    /// fails to travel is visible rather than accidentally correct.
    fn residual_saved_rack(
        scope: crate::visual_rack::LegacyRackScope,
        structure: crate::visual_rack::SavedImageSource,
        detail: crate::visual_rack::SavedImageSource,
        mix: f32,
    ) -> crate::visual_rack::VisualRack {
        use crate::visual_rack::{
            EdgeTiming, ResidualBlock, ResidualParams, ResidualQuantization, SavedImageTap,
            VisualNodeKind, VisualRack,
        };

        let mut rack = VisualRack::synthetic_legacy(scope);
        rack.push(VisualNodeKind::Residual(ResidualParams {
            structure: SavedImageTap {
                source: structure,
                timing: EdgeTiming::CurrentFrame,
            },
            detail: SavedImageTap {
                source: detail,
                timing: EdgeTiming::PreviousFrame,
            },
            block: ResidualBlock::Sixteen,
            quantization: ResidualQuantization::Medium,
            mix,
            detail_gain: 2.5,
            seed: 0x00c0_ffee,
            ..ResidualParams::default()
        }))
        .unwrap();
        rack
    }

    /// Two stacked clips where the upper scope carries the Residual node.
    fn residual_export_patch(
        structure: crate::visual_rack::SavedImageSource,
        detail: crate::visual_rack::SavedImageSource,
        mix: f32,
    ) -> PatchState {
        let mut patch: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: carrier.mp4
  - filename: donor.mp4
"#,
        )
        .unwrap();
        patch.layers[0].rack = Some(residual_saved_rack(
            crate::visual_rack::LegacyRackScope::Layer,
            structure,
            detail,
            mix,
        ));
        patch
    }

    fn residual_saved_layer(
        position: u32,
        stage: crate::image_routing::LayerImageStage,
    ) -> crate::visual_rack::SavedImageSource {
        crate::visual_rack::SavedImageSource::SelectedLayer {
            layer_position: SavedLayerPosition::new(position).unwrap(),
            stage,
        }
    }

    /// The payload the shader actually consumes, taken through the real rack
    /// compiler. Legacy markers belong to the frozen legacy path and never
    /// enter that compiler, so this mirrors the executor's own segmentation
    /// instead of comparing the authored struct with itself.
    fn compiled_residual_payload(
        rack: &crate::visual_rack::RuntimeVisualRack,
    ) -> crate::visual_rack::RuntimeResidualParams {
        use crate::renderer::rack::{CollisionRackPlan, RackPassKind};
        use crate::visual_rack::{RuntimeVisualNodeKind, RuntimeVisualRack};

        let authored = rack
            .iter()
            .find_map(|node| match node.kind {
                RuntimeVisualNodeKind::Residual(params) => Some(params),
                _ => None,
            })
            .expect("authored residual node");
        let mut segment = RuntimeVisualRack::empty();
        segment
            .push(RuntimeVisualNodeKind::Residual(authored))
            .unwrap();
        let compiled = CollisionRackPlan::compile(&segment, [128, 64], [128, 64]).unwrap();
        match compiled.passes()[0].kind {
            RackPassKind::Residual(params) => params,
            other => panic!("compiled a {other:?} where a residual pass was authored"),
        }
    }

    /// A job records what it actually admitted: the discrete recombination law
    /// and, per slot, whether that route resolved onto a job-local identity or
    /// stayed a tombstone. The record is bounded and carries stable identities
    /// only — never a host path, a filename, or filesystem metadata.
    #[test]
    fn field_collider_export_provenance_records_both_slots_and_no_pixels() {
        use crate::evaluated_frame::evaluated_composition::EvaluatedFieldColliderPlan;
        use crate::motion::{
            FieldColliderMode, MotionBoundaryMode, MotionDonor, MotionGrid, MotionParams,
            FIELD_COLLIDER_ALGORITHM_VERSION,
        };
        use crate::performance::SavedLayerPosition;

        // A record built from the shared evaluated plan: slot A resolves onto a
        // live layer, slot B is a retained tombstone. Both are recorded, always
        // and never compacted, so a tombstone can never be read as belonging to
        // its partner.
        let grid = MotionGrid::for_source([640, 480], Default::default()).unwrap();
        let plan = EvaluatedFieldColliderPlan {
            output_slot: 2,
            recipient_scope: crate::visual_rack::VisualScopeId::Layer(
                crate::image_routing::StableLayerId::new(41).unwrap(),
            ),
            input_a_scope: crate::visual_rack::VisualScopeId::Layer(
                crate::image_routing::StableLayerId::new(42).unwrap(),
            ),
            input_a_slot: 0,
            input_b_scope: crate::visual_rack::VisualScopeId::Layer(
                crate::image_routing::StableLayerId::new(43).unwrap(),
            ),
            input_b_slot: 1,
            output_grid: grid,
            algorithm_version: FIELD_COLLIDER_ALGORITHM_VERSION,
            mode: FieldColliderMode::CollisionBoundary,
            boundary: MotionBoundaryMode::Wrap,
        };
        let _ = MotionParams::default();
        let _ = MotionDonor::Missing {
            saved_position: SavedLayerPosition::new(4).unwrap(),
        };

        let record = FieldColliderSidecar {
            algorithm_version: plan.algorithm_version,
            recipient: MotionSidecarScopeIdentity::Layer {
                saved_position: 0,
                stable_id: 41,
                source_tap_id: 1,
            },
            admitted: true,
            mode: "collision_boundary",
            boundary: "wrap",
            output_slot: plan.output_slot,
            output_grid: [plan.output_grid.width, plan.output_grid.height],
            inputs: vec![
                FieldColliderSidecarInput {
                    slot: "a",
                    resolved: true,
                    saved_position: Some(1),
                    stable_id: Some(42),
                    field_slot: Some(0),
                },
                FieldColliderSidecarInput {
                    slot: "b",
                    resolved: false,
                    saved_position: Some(9),
                    stable_id: None,
                    field_slot: None,
                },
            ],
            diagnostic: None,
            bytes_per_cell: 20,
            derived_vector_bytes: grid.vector_count * 8,
            derived_gate_bytes: grid.vector_count * 4,
            transient_pair_bytes: grid.vector_count * 8,
            total_bytes: grid.vector_count * 20,
            low_resolution_passes: 2,
            nearest_lookups: 5,
            max_sampled_textures_in_pass: 3,
        };

        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["algorithm_version"], 1);
        assert_eq!(json["mode"], "collision_boundary");
        assert_eq!(json["boundary"], "wrap");
        assert_eq!(json["output_slot"], 2);
        assert_eq!(json["bytes_per_cell"], 20);
        assert_eq!(json["low_resolution_passes"], 2);
        assert_eq!(json["max_sampled_textures_in_pass"], 3);

        // BOTH slots are present, named, and never compacted.
        let inputs = json["inputs"].as_array().unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0]["slot"], "a");
        assert_eq!(inputs[0]["resolved"], true);
        assert_eq!(inputs[0]["field_slot"], 0);
        assert_eq!(inputs[1]["slot"], "b");
        assert_eq!(inputs[1]["resolved"], false);
        assert_eq!(inputs[1]["saved_position"], 9);
        // A tombstone carries no live identity and no admitted field slot.
        assert!(inputs[1].get("stable_id").is_none());
        assert!(inputs[1].get("field_slot").is_none());

        // Authored identity, topology, diagnostics, and budgets ONLY. No
        // vector, no pair texel, no gate parity, no codec record, no host path.
        let text = serde_json::to_string(&record).unwrap();
        for forbidden in [
            "velocity",
            "pixels",
            "texel",
            "codec_record",
            "renders/",
            "\\\\",
        ] {
            assert!(
                !text.contains(forbidden),
                "the collider provenance record leaked {forbidden}"
            );
        }
        // `max_sampled_textures_in_pass` is a BUDGET key, not pixel content:
        // the only occurrence of "texture" is that documented count.
        assert_eq!(text.matches("texture").count(), 1);
        assert!(text.contains("max_sampled_textures_in_pass"));
    }

    #[test]
    fn residual_export_provenance_names_every_route_slot_and_the_recombination_law() {
        use crate::image_routing::LayerImageStage;
        use crate::visual_rack::{ResidualBlock, ResidualQuantization, SavedImageSource};

        // Slot 0 resolves onto the donor below. Slot 1 names a saved position
        // this export does not contain and must stay a retained tombstone
        // instead of sliding onto its live partner.
        let patch = residual_export_patch(
            residual_saved_layer(1, LayerImageStage::PostLocalEffects),
            residual_saved_layer(9, LayerImageStage::PreLocalEffects),
            0.8,
        );
        let graph = resolve_export_creative_graph(&patch).unwrap();
        let mut export_config = config(1.0);
        export_config.output_path = "renders/residual-provenance.mp4".to_owned();
        let sidecar = ExportMotionSidecarAccumulator::new(
            &export_config,
            &graph,
            &[],
            crate::motion::MotionParams::default(),
            &[],
        )
        .finish(Vec::new());

        assert_eq!(sidecar.schema_version, 7);
        assert!(sidecar.field_collider.is_none());
        assert!(sidecar.codec_mosh.is_none());
        assert!(!sidecar.authored_residual_nodes_truncated);
        assert_eq!(sidecar.authored_residual_nodes.len(), 1);
        let record = &sidecar.authored_residual_nodes[0];
        assert_eq!(
            record.scope,
            MotionSidecarScopeIdentity::Layer {
                saved_position: 0,
                stable_id: 1,
                source_tap_id: export_selective_layer_id(0),
            }
        );
        assert!(record.enabled);
        assert_eq!(record.wet, 1.0);
        assert_eq!(
            record.algorithm_version,
            crate::visual_rack::RESIDUAL_ALGORITHM_VERSION
        );
        assert_eq!(record.block, ResidualBlock::Sixteen);
        assert_eq!(record.block_edge, 16);
        assert_eq!(record.quantization, ResidualQuantization::Medium);
        assert_eq!(record.quantization_levels, 32);
        assert_eq!(record.authored_mix, 0.8);
        assert_eq!(record.authored_detail_gain, 2.5);
        assert_eq!(record.seed, 0x00c0_ffee);
        assert!(!record.exact_bypass);

        assert_eq!(
            record.routes.len(),
            crate::visual_rack::RESIDUAL_ROUTE_SLOTS
        );
        let structure = &record.routes[usize::from(crate::visual_rack::RESIDUAL_STRUCTURE_SLOT)];
        assert_eq!(structure.slot, crate::visual_rack::RESIDUAL_STRUCTURE_SLOT);
        assert_eq!(structure.slot_name, "structure");
        assert_eq!(structure.source, "selected_layer");
        assert_eq!(
            structure.timing,
            crate::visual_rack::EdgeTiming::CurrentFrame
        );
        assert!(structure.resolved);
        assert_eq!(structure.saved_position, Some(1));
        assert_eq!(structure.stable_id, Some(2));
        assert_eq!(structure.stage, Some(LayerImageStage::PostLocalEffects));
        assert_eq!(structure.group_id, None);

        let detail = &record.routes[usize::from(crate::visual_rack::RESIDUAL_DETAIL_SLOT)];
        assert_eq!(detail.slot, crate::visual_rack::RESIDUAL_DETAIL_SLOT);
        assert_eq!(detail.slot_name, "detail");
        assert_eq!(detail.source, "missing_selected_layer");
        assert_eq!(detail.timing, crate::visual_rack::EdgeTiming::PreviousFrame);
        assert!(
            !detail.resolved,
            "an out-of-range saved position is a tombstone, never a neighbour"
        );
        assert_eq!(detail.saved_position, Some(9));
        assert_eq!(
            detail.stable_id, None,
            "a tombstone must not publish a job-local identity it never had"
        );
        assert_eq!(detail.stage, Some(LayerImageStage::PreLocalEffects));

        // The published document carries no operational path or filename.
        let json = serde_json::to_string(&sidecar.authored_residual_nodes).unwrap();
        for forbidden in ["carrier.mp4", "donor.mp4", "renders", "/", "\\", ".mp4"] {
            assert!(
                !json.contains(forbidden),
                "residual provenance leaked {forbidden}: {json}"
            );
        }

        // A dormant node is still fully documented — the provenance says what
        // the job admitted, and delegation is one of its facts.
        let dormant = residual_export_patch(
            residual_saved_layer(1, LayerImageStage::PostLocalEffects),
            SavedImageSource::AllBelow,
            0.0,
        );
        let dormant_graph = resolve_export_creative_graph(&dormant).unwrap();
        let (dormant_records, dormant_truncated) = residual_sidecar_nodes(&dormant_graph);
        assert!(!dormant_truncated);
        assert_eq!(dormant_records.len(), 1);
        assert!(dormant_records[0].exact_bypass);
        assert_eq!(dormant_records[0].authored_mix, 0.0);
        assert_eq!(dormant_records[0].routes[1].source, "all_below");
        assert!(dormant_records[0].routes[1].resolved);

        // The cap bounds a hand-authored patch instead of growing the report.
        let mut saturated = vec![dormant_records[0].clone(); MAX_EXPORT_RESIDUAL_SIDECAR_NODES];
        let mut truncated = false;
        residual_sidecar_scope_nodes(
            MotionSidecarScopeIdentity::Master,
            &dormant_graph.layer_racks[0].1,
            &mut saturated,
            &mut truncated,
        );
        assert!(truncated);
        assert_eq!(saturated.len(), MAX_EXPORT_RESIDUAL_SIDECAR_NODES);
    }

    /// The recombination pass consumes no clock of its own: its shader payload
    /// and its reduced-surface budget are a pure function of the patch, the
    /// authored seed, and the resolved routes. Export must therefore produce
    /// exactly one payload for a given frame index at every rate, and the same
    /// payload at every frame index.
    #[test]
    fn residual_export_payload_is_frame_index_derived_and_carries_no_clock() {
        use crate::evaluated_frame::evaluated_composition::{ImageTapConsumer, PlannedImageTap};
        use crate::image_routing::LayerImageStage;
        use crate::visual_rack::{ResidualResourcePlan, SavedImageSource};

        let patch = residual_export_patch(
            residual_saved_layer(1, LayerImageStage::PostLocalEffects),
            SavedImageSource::OneBelow,
            0.65,
        );
        let graph = resolve_export_creative_graph(&patch).unwrap();
        let effects = EffectUniforms {
            // A neighbouring effect that genuinely does consume program time,
            // so an accidental clock dependency would have somewhere to leak
            // in from rather than being trivially absent.
            shift_amount: 0.6,
            shift_speed: 4.0,
            cellular_amount: 0.5,
            cellular_speed: 1.5,
            ..EffectUniforms::default()
        };
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();

        let mut canonical: Option<(
            crate::visual_rack::RuntimeResidualParams,
            ResidualResourcePlan,
            Vec<PlannedImageTap>,
        )> = None;
        for fps in [24_u32, 30, 60] {
            for frame_index in [0_u64, 45, 1234] {
                let (time, _delta) = export_program_transport(frame_index, 1.0 / fps as f32, false);
                let modulation = crate::modulation::ModMatrix::new().frame(graph.layer_ids.len());
                let evaluated = EvaluatedFramePlan::evaluate(
                    &modulation,
                    FramePlanContext::new(64, 32, time),
                    MasterFrameInput {
                        effects: &effects,
                        transform: &transform,
                        ntsc: &ntsc,
                        temporal: &temporal,
                    },
                    graph
                        .layer_ids
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(slot, id)| LayerFrameInput {
                            source: SourceTap::new(id.get(), slot, 64, 32),
                            effects: &effects,
                            transform: &transform,
                            opacity: 1.0,
                            mosh_send: 1.0,
                            speed: 1.0,
                            fps: fps as f32,
                            blend_mode: BlendMode::Normal,
                            visible: true,
                            paused: false,
                            bypass_master_fx: false,
                            bypass_temporal_fx: false,
                            pattern: None,
                        }),
                );
                let planned = plan_export_composition(
                    &evaluated,
                    &graph,
                    &[LayerMatte::default(); 2],
                    false,
                    CreativeResourceLimits::default(),
                )
                .unwrap();
                // The shared executor accepts the shared planner's result. A
                // planner/executor disagreement is a hard export error, so
                // this is the point where an export-only path would have to
                // exist — and does not.
                CompositionGpuExecutor::validate_plan(&planned).unwrap();
                let EvaluatedCompositionPlan::Advanced(advanced) = planned else {
                    panic!("a live Residual mix must enter the unified advanced plan");
                };

                // Both slots are collected as two distinct consumers.
                let taps = advanced
                    .image_taps()
                    .iter()
                    .filter(|tap| matches!(tap.consumer, ImageTapConsumer::RackNode { .. }))
                    .cloned()
                    .collect::<Vec<_>>();
                assert_eq!(taps.len(), crate::visual_rack::RESIDUAL_ROUTE_SLOTS);

                let residual = advanced.residual_resources();
                assert_eq!(residual.active_nodes, 1);
                assert_eq!(residual.mean_surfaces, 2);
                assert_eq!(residual.max_grid_dimensions, [4, 2]);

                // The executor payload the shader actually consumes.
                let compiled = compiled_residual_payload(&graph.layer_racks[0].1);
                assert_eq!(
                    compiled.seed, 0x00c0_ffee,
                    "the authored seed is recalled, never re-drawn offline"
                );

                let evidence = (compiled, residual, taps);
                if let Some(expected) = &canonical {
                    assert_eq!(
                        &evidence, expected,
                        "residual payload moved at {fps} fps, frame {frame_index}"
                    );
                } else {
                    canonical = Some(evidence);
                }
            }
        }

        // There is no export-only recombination path. Everything this module
        // contributes is provenance; the rack shader, the two reduced passes,
        // the mean surfaces and their preflight all live behind the shared
        // executor, reached identically by live and offline rendering.
        let source = include_str!("render_export.rs");
        let production = &source[..source.find("\nmod tests {").expect("test module")];
        for forbidden in [
            "RackPassKind",
            "CollisionRackPlan",
            "rack_node.wgsl",
            "ResidualResourcePlan",
            "prepare_residual_means",
        ] {
            assert!(
                !production.contains(forbidden),
                "export grew its own residual path through {forbidden}"
            );
        }
    }

    /// Live and offline build their stable creative graph independently — live
    /// from process-lifetime layer IDs, export from `position + 1` — and then
    /// run the same address book, the same stable modulation frame, and the
    /// same rack compiler. A master-scope address is identical on both sides,
    /// so the same patch, the same seed and the same frame ordinal must hand
    /// the shader the same Residual payload on both, derived from the frame
    /// ordinal rather than read off a wall clock.
    #[test]
    fn residual_live_and_export_share_one_stable_modulation_payload_at_24_30_and_60_fps() {
        use crate::modulation::{ModMatrix, ModSource, Routing, StableModAddressBook};
        use crate::visual_rack::{LegacyRackScope, RuntimeVisualRack, SavedImageSource};

        let saved_master = residual_saved_rack(
            LegacyRackScope::Master,
            SavedImageSource::AllBelow,
            SavedImageSource::CleanProgram,
            0.25,
        );
        let node_id = saved_master
            .iter()
            .find(|node| matches!(node.kind, crate::visual_rack::VisualNodeKind::Residual(_)))
            .expect("authored residual node")
            .stable_id;

        // Live process-lifetime identities and export's position-derived ones
        // are deliberately different numbers; only the master-scope address is
        // shared, which is exactly what makes this a parity proof rather than
        // a comparison of one value with itself.
        let live_ids = [
            StableLayerId::new(4242).unwrap(),
            StableLayerId::new(777).unwrap(),
        ];
        let patch = {
            let mut patch: PatchState = serde_yaml::from_str(
                r#"
master: {}
layers:
  - filename: carrier.mp4
  - filename: donor.mp4
"#,
            )
            .unwrap();
            patch.master_rack = Some(saved_master.clone());
            patch
        };
        let export_graph = resolve_export_creative_graph(&patch).unwrap();
        assert_eq!(
            export_graph.layer_ids.as_ref(),
            &[
                StableLayerId::new(1).unwrap(),
                StableLayerId::new(2).unwrap(),
            ],
            "export identity stays position-derived"
        );

        let live_composition = crate::composition::RuntimeComposition::try_from_parts(
            Vec::new(),
            live_ids
                .iter()
                .rev()
                .map(|layer_id| crate::composition::RuntimeRootItem::Layer {
                    layer_id: *layer_id,
                    bus: crate::composition::BusAssignment::Program,
                })
                .collect(),
            None,
            0.5,
        )
        .unwrap();
        let live_master_template = saved_master.resolve_routes(
            |position| live_ids.get(position.get() as usize).copied(),
            |_| false,
        );
        let live_layer_template = live_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
                )
            })
            .collect::<Vec<_>>();

        let routings = vec![
            Routing::new(
                ModSource::Lfo(0),
                format!("node/master/{}/mix", node_id.get()),
                0.6,
            ),
            Routing::new(
                ModSource::Lfo(0),
                format!("node/master/{}/detail_gain", node_id.get()),
                0.4,
            ),
        ];
        let matrix_at = |beat: f64, delta_seconds: f32| {
            let mut matrix = ModMatrix::new();
            matrix.lfos[0].beats = 3.0;
            matrix.lfos[0].set_phase(0.125);
            matrix.routings = routings.clone();
            matrix.update_at_beat(beat, delta_seconds);
            matrix
        };
        // One (beat, delta) instant projected into both worlds. Each side runs
        // its own address book over its own resolved racks; only the authored
        // master-scope address is shared.
        let sample = |beat: f64, delta_seconds: f32| {
            let matrix = matrix_at(beat, delta_seconds);
            let mut export_master = export_graph.master_rack.clone();
            let mut export_layers = export_graph.layer_racks.clone();
            let mut export_composition = export_graph.composition.clone();
            let export_book = StableModAddressBook::from_composition(
                &export_master,
                &export_layers,
                &export_composition,
            )
            .unwrap();
            crate::modulation::apply_stable_modulation(
                &export_book,
                &matrix.stable_frame(&export_book),
                &mut export_master,
                &mut export_layers,
                &mut export_composition,
            );

            let live_matrix = matrix_at(beat, delta_seconds);
            let mut live_master = live_master_template.clone();
            let mut live_layers = live_layer_template.clone();
            let mut live_composition = live_composition.clone();
            let live_book = StableModAddressBook::from_composition(
                &live_master,
                &live_layers,
                &live_composition,
            )
            .unwrap();
            crate::modulation::apply_stable_modulation(
                &live_book,
                &live_matrix.stable_frame(&live_book),
                &mut live_master,
                &mut live_layers,
                &mut live_composition,
            );
            (
                compiled_residual_payload(&export_master),
                compiled_residual_payload(&live_master),
            )
        };

        // Both worlds resolved the two authored slots to the same thing before
        // any modulation ran, so a later divergence can only come from the
        // per-frame projection under test.
        let authored_routes = compiled_residual_payload(&live_master_template).routes();
        assert_eq!(
            compiled_residual_payload(&export_graph.master_rack).routes(),
            authored_routes
        );

        let mut moved = false;
        for fps in [24_u32, 30, 60] {
            let mut previous: Option<crate::visual_rack::RuntimeResidualParams> = None;
            for frame_index in [0_u64, 45, 90] {
                // Offline time is the frame ordinal times the frame interval —
                // never a wall clock and never a live input.
                let (time, delta_seconds) =
                    export_program_transport(frame_index, 1.0 / fps as f32, false);
                assert_eq!(time, frame_index as f32 * (1.0 / fps as f32));
                let beat = f64::from(time) * 2.0;

                let (export_params, live_params) = sample(beat, delta_seconds);
                assert_eq!(
                    export_params, live_params,
                    "live and export diverged at {fps} fps, frame {frame_index}"
                );

                // Re-deriving the same ordinal reproduces it exactly. Nothing
                // on this path reads a clock the replay cannot also see.
                assert_eq!(
                    sample(beat, delta_seconds).0,
                    export_params,
                    "the payload for {fps} fps frame {frame_index} was not reproducible"
                );

                // The topology half is authored recall, never a per-frame draw.
                assert_eq!(export_params.routes(), authored_routes);
                assert_eq!(
                    export_params.block,
                    crate::visual_rack::ResidualBlock::Sixteen
                );
                assert_eq!(
                    export_params.quantization,
                    crate::visual_rack::ResidualQuantization::Medium
                );
                assert_eq!(
                    export_params.seed, 0x00c0_ffee,
                    "the authored seed is recalled, never re-drawn offline"
                );
                assert_eq!(
                    export_params.algorithm_version,
                    crate::visual_rack::RESIDUAL_ALGORITHM_VERSION
                );
                assert!((0.0..=1.0).contains(&export_params.mix));
                assert!((0.0..=4.0).contains(&export_params.detail_gain));

                if previous.is_some_and(|previous: crate::visual_rack::RuntimeResidualParams| {
                    previous.mix != export_params.mix
                        || previous.detail_gain != export_params.detail_gain
                }) {
                    moved = true;
                }
                previous = Some(export_params);
            }
        }

        // The routes really drive the two continuous values, so the equalities
        // above are not the trivial equality of one constant with itself.
        assert!(
            moved,
            "a beat-driven residual route must move as the frame ordinal advances"
        );
    }

    fn m4_motion_evaluated_frame(graph: &ExportCreativeGraph, fps: u32) -> EvaluatedFramePlan {
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(graph.layer_ids.len());
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 32, 1.0),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            graph
                .layer_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(slot, id)| LayerFrameInput {
                    source: SourceTap::new(id.get(), slot, 64, 32),
                    effects: &effects,
                    transform: &transform,
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: fps as f32,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                    pattern: None,
                }),
        )
    }

    fn m4_motion_params(
        graph: &ExportCreativeGraph,
    ) -> (
        crate::motion::MotionParams,
        Vec<crate::motion::MotionParams>,
    ) {
        use crate::motion::{
            CurvedShutterParams, CurvedShutterQuality, FaradayParams, MotionCarrier,
            MotionFieldSource, MotionLatticeQuality, MotionParams,
        };

        let master = MotionParams {
            field_source: MotionFieldSource::Lattice,
            lattice_quality: MotionLatticeQuality::Draft,
            shutter: CurvedShutterParams {
                angle_degrees: 180.0,
                quality: CurvedShutterQuality::Draft,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };
        let mut layers = vec![MotionParams::default(); graph.layer_ids.len()];
        layers[0] = MotionParams {
            field_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::Live,
            shutter: CurvedShutterParams {
                angle_degrees: 90.0,
                quality: CurvedShutterQuality::Live,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };
        layers[1] = MotionParams {
            field_source: MotionFieldSource::Auto,
            lattice_quality: MotionLatticeQuality::High,
            shutter: CurvedShutterParams {
                angle_degrees: 270.0,
                quality: CurvedShutterQuality::High,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };
        let recipient = MotionParams {
            transplant: FaradayParams {
                amount: 0.75,
                carrier: MotionCarrier::FirstSourceFrame,
                ..FaradayParams::default()
            },
            ..MotionParams::default()
        };
        let mut saved = crate::patch::MotionConfig::from_params(recipient);
        saved.transplant.donor = crate::patch::MotionDonorConfig::Selected {
            saved_position: SavedLayerPosition::new(0).unwrap(),
        };
        layers[2] = resolved_export_motion(Some(saved), &graph.layer_ids);
        (master, layers)
    }

    fn m4_motion_plan(
        evaluated: &EvaluatedFramePlan,
        graph: &ExportCreativeGraph,
        master: crate::motion::MotionParams,
        layers: &[crate::evaluated_frame::evaluated_composition::LayerMotionPlanInput],
    ) -> crate::evaluated_frame::evaluated_composition::AdvancedCompositionPlan {
        let mattes = vec![LayerMatte::default(); graph.layer_ids.len()];
        let input =
            CompositionPlanInput::new(&graph.composition, &graph.master_rack, &graph.layer_racks)
                .with_layer_mattes(&mattes, false)
                .with_motion(
                    master,
                    layers,
                    crate::motion::MotionDeviceLimits::new(16_384, u64::MAX),
                );
        match evaluated.plan_composition(input).unwrap() {
            EvaluatedCompositionPlan::Advanced(plan) => *plan,
            EvaluatedCompositionPlan::LegacyExact(_) => {
                panic!("active motion fixture unexpectedly delegated LegacyExact")
            }
        }
    }

    fn assert_m4_motion_plans_equal(
        left: &crate::evaluated_frame::evaluated_composition::AdvancedCompositionPlan,
        right: &crate::evaluated_frame::evaluated_composition::AdvancedCompositionPlan,
    ) {
        let left = left.motion().advanced().unwrap();
        let right = right.motion().advanced().unwrap();
        assert_eq!(left.scopes(), right.scopes());
        assert_eq!(left.fields(), right.fields());
        assert_eq!(left.resources(), right.resources());
        assert_eq!(left.diagnostics(), right.diagnostics());
        assert_eq!(left.budget(), right.budget());
        assert_eq!(left.topology_signature(), right.topology_signature());
    }

    #[test]
    fn m4_live_style_and_export_style_motion_plans_are_equal_at_24_30_60() {
        use crate::evaluated_frame::evaluated_composition::{
            LayerMotionPlanInput, MotionCodecFrameFacts,
        };
        use crate::motion::{MotionFieldOrigin, MotionSourceDiagnostic};

        let graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        let (master, params) = m4_motion_params(&graph);
        let facts = [
            MotionCodecFrameFacts {
                available: true,
                source_generation: 11,
                frame_ordinal: 17,
            },
            MotionCodecFrameFacts::default(),
            MotionCodecFrameFacts::default(),
        ];
        let mut canonical = None;
        for fps in [24_u32, 30, 60] {
            let evaluated = m4_motion_evaluated_frame(&graph, fps);
            let export_inputs =
                export_motion_layer_plan_inputs(&graph, &params, facts.into_iter().enumerate())
                    .unwrap();
            let live_inputs = graph
                .layer_ids
                .iter()
                .copied()
                .zip(params.iter().copied())
                .zip(facts)
                .map(|((stable_id, params), codec)| LayerMotionPlanInput {
                    stable_id,
                    params,
                    codec,
                })
                .collect::<Vec<_>>();
            assert_eq!(export_inputs, live_inputs);

            let export = m4_motion_plan(&evaluated, &graph, master, &export_inputs);
            let live = m4_motion_plan(&evaluated, &graph, master, &live_inputs);
            assert_m4_motion_plans_equal(&live, &export);
            if let Some(previous) = canonical.as_ref() {
                assert_m4_motion_plans_equal(previous, &export);
            } else {
                canonical = Some(export.clone());
            }

            let motion = export.motion().advanced().unwrap();
            assert_eq!(
                motion
                    .scope(crate::visual_rack::VisualScopeId::Master)
                    .unwrap()
                    .source
                    .origin,
                MotionFieldOrigin::Lattice
            );
            assert_eq!(
                motion
                    .scope(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
                    .unwrap()
                    .source
                    .origin,
                MotionFieldOrigin::CodecVectors
            );
            let fallback = motion
                .scope(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[1]))
                .unwrap();
            assert_eq!(fallback.source.origin, MotionFieldOrigin::LatticeFallback);
            assert_eq!(
                fallback.source.diagnostic,
                MotionSourceDiagnostic::CodecUnavailableFallback
            );
            let recipient = motion
                .scope(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[2]))
                .unwrap();
            assert!(recipient.transplant_admitted);
            assert_eq!(
                recipient.donor_scope,
                Some(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
            );
        }

        // A zero shutter angle is literal zero even when every other authored
        // tier/source selector is non-default, and must preserve delegation.
        let evaluated = m4_motion_evaluated_frame(&graph, 30);
        let mut exact_zero = crate::motion::MotionParams::default();
        exact_zero.shutter.quality = crate::motion::CurvedShutterQuality::High;
        exact_zero.field_source = crate::motion::MotionFieldSource::CodecVectors;
        let zero = vec![exact_zero; graph.layer_ids.len()];
        let zero_inputs = export_motion_layer_plan_inputs(
            &graph,
            &zero,
            (0..zero.len()).map(|index| (index, MotionCodecFrameFacts::default())),
        )
        .unwrap();
        let mattes = vec![LayerMatte::default(); graph.layer_ids.len()];
        let zero_plan = evaluated
            .plan_composition(
                CompositionPlanInput::new(
                    &graph.composition,
                    &graph.master_rack,
                    &graph.layer_racks,
                )
                .with_layer_mattes(&mattes, false)
                .with_motion(
                    exact_zero,
                    &zero_inputs,
                    crate::motion::MotionDeviceLimits::new(16_384, u64::MAX),
                ),
            )
            .unwrap();
        assert!(matches!(
            zero_plan,
            EvaluatedCompositionPlan::LegacyExact(_)
        ));
    }

    #[test]
    fn export_shutter_request_is_exact_and_candidate_final_resources_match() {
        use crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts;
        use crate::motion::{
            CurvedShutterParams, CurvedShutterQuality, MotionFieldSource, MotionParams,
        };

        let policies = [
            (ExportShutterSamples::Authored, None, "authored"),
            (ExportShutterSamples::Samples1, Some(1), "samples_1"),
            (ExportShutterSamples::Samples4, Some(4), "samples_4"),
            (ExportShutterSamples::Samples8, Some(8), "samples_8"),
            (ExportShutterSamples::Samples16, Some(16), "samples_16"),
        ];
        for (policy, count, wire_key) in policies {
            assert!(policy.is_valid());
            assert_eq!(policy.requested_count(), count);
            assert_eq!(serde_json::to_value(policy).unwrap(), wire_key);
        }

        let graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        let evaluated = m4_motion_evaluated_frame(&graph, 30);
        let master = MotionParams::default();
        let mut authored_layers = vec![MotionParams::default(); graph.layer_ids.len()];
        authored_layers[0] = MotionParams {
            field_source: MotionFieldSource::Lattice,
            shutter: CurvedShutterParams {
                angle_degrees: 180.0,
                quality: CurvedShutterQuality::Sharp,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };
        let modulation = crate::modulation::ModMatrix::new().frame(authored_layers.len());
        let plan = |policy| {
            let (effective_master, effective_layers) =
                effective_export_motion_params(master, &authored_layers, &modulation, policy);
            let inputs = export_motion_layer_plan_inputs(
                &graph,
                &effective_layers,
                (0..effective_layers.len()).map(|index| (index, MotionCodecFrameFacts::default())),
            )
            .unwrap();
            m4_motion_plan(&evaluated, &graph, effective_master, &inputs)
        };

        // Morph-candidate preflight and final immutable planning consume the
        // same post-Morph/post-modulation explicit-count policy.
        let candidate = plan(ExportShutterSamples::Samples16);
        let final_plan = plan(ExportShutterSamples::Samples16);
        assert_m4_motion_plans_equal(&candidate, &final_plan);
        let candidate_motion = candidate.motion().advanced().unwrap();
        assert_eq!(candidate_motion.resources().max_shutter_samples, 16);
        assert_eq!(candidate_motion.budget().full_frame_passes, 1);
        assert_eq!(
            candidate_motion.budget().logical_texture_lookups_per_pixel,
            18
        );
        assert_eq!(candidate_motion.budget().texture_samples_per_pixel, 66);
        assert_eq!(
            candidate_motion
                .scope(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
                .unwrap()
                .params
                .shutter
                .quality,
            CurvedShutterQuality::High
        );

        let authored = plan(ExportShutterSamples::Authored);
        let authored_motion = authored.motion().advanced().unwrap();
        assert_eq!(authored_motion.resources().max_shutter_samples, 1);
        assert_eq!(
            authored_motion.budget().logical_texture_lookups_per_pixel,
            3
        );
        assert_eq!(authored_motion.budget().texture_samples_per_pixel, 6);
        let authored_metadata = export_motion_metadata_for_frame_from(
            &EvaluatedCompositionPlan::Advanced(Box::new(authored.clone())),
            &graph,
            [(0, None), (1, None), (2, None)],
            [],
            0,
        );
        for saved_position in 0..3 {
            let scope = authored_metadata
                .scopes
                .iter()
                .find(|scope| {
                    matches!(
                        scope.scope,
                        ExportMotionScopeIdentity::Layer {
                            saved_position: candidate,
                            ..
                        } if candidate == saved_position
                    )
                })
                .unwrap();
            assert_eq!(scope.field_planned, saved_position == 0);
            assert!(!scope.field_attached);
        }

        // Frame-local modulation is resolved before the fixed export tier, so
        // candidate preflight accounts for an angle activated by modulation.
        let mut matrix = crate::modulation::ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings.push(crate::modulation::Routing::new(
            crate::modulation::ModSource::Midi(0),
            "layer1_motion_shutter_angle",
            1.0,
        ));
        matrix.update_at_beat(0.0, 0.0);
        let modulated = matrix.frame(authored_layers.len());
        let zero_layers = vec![MotionParams::default(); graph.layer_ids.len()];
        let (_, modulated_layers) = effective_export_motion_params(
            MotionParams::default(),
            &zero_layers,
            &modulated,
            ExportShutterSamples::Samples16,
        );
        assert_eq!(modulated_layers[0].shutter.angle_degrees, 180.0);
        assert_eq!(
            modulated_layers[0].shutter.quality,
            CurvedShutterQuality::High
        );

        // An explicit request changes only quality. It cannot activate a
        // zero-angle shutter or weaken literal LegacyExact delegation.
        let modulation = crate::modulation::ModMatrix::new().frame(graph.layer_ids.len());
        let (zero_master, zero_layers) = effective_export_motion_params(
            MotionParams::default(),
            &zero_layers,
            &modulation,
            ExportShutterSamples::Samples16,
        );
        assert!(zero_layers.iter().all(|params| {
            params.shutter.quality == CurvedShutterQuality::High && params.shutter.is_exact_zero()
        }));
        let zero_inputs = export_motion_layer_plan_inputs(
            &graph,
            &zero_layers,
            (0..zero_layers.len()).map(|index| (index, MotionCodecFrameFacts::default())),
        )
        .unwrap();
        let mattes = vec![LayerMatte::default(); graph.layer_ids.len()];
        let zero_plan = evaluated
            .plan_composition(
                CompositionPlanInput::new(
                    &graph.composition,
                    &graph.master_rack,
                    &graph.layer_racks,
                )
                .with_layer_mattes(&mattes, false)
                .with_motion(
                    zero_master,
                    &zero_inputs,
                    crate::motion::MotionDeviceLimits::new(16_384, u64::MAX),
                ),
            )
            .unwrap();
        assert!(matches!(
            zero_plan,
            EvaluatedCompositionPlan::LegacyExact(_)
        ));
    }

    #[test]
    fn m4_export_donor_resolution_holds_and_fixed_shutter_tiers_use_authored_laws() {
        use crate::motion::{CurvedShutterQuality, MotionDonor};
        use crate::transport::NormalizedTime;

        let graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        let mut saved = crate::patch::MotionConfig::default();
        saved.transplant.amount = 1.0;
        saved.transplant.donor = crate::patch::MotionDonorConfig::Selected {
            saved_position: SavedLayerPosition::new(1).unwrap(),
        };
        let resolved = resolved_export_motion(Some(saved), &graph.layer_ids);
        assert_eq!(
            resolved.transplant.donor,
            MotionDonor::Selected {
                layer_id: graph.layer_ids[1],
                saved_position: SavedLayerPosition::new(1).unwrap(),
            }
        );

        saved.transplant.donor = crate::patch::MotionDonorConfig::Selected {
            saved_position: SavedLayerPosition::new(9).unwrap(),
        };
        assert_eq!(
            resolved_export_motion(Some(saved), &graph.layer_ids)
                .transplant
                .donor,
            MotionDonor::Missing {
                saved_position: SavedLayerPosition::new(9).unwrap(),
            }
        );

        assert_eq!(CurvedShutterQuality::Sharp.sample_count(), 1);
        assert_eq!(CurvedShutterQuality::Draft.sample_count(), 4);
        assert_eq!(CurvedShutterQuality::Live.sample_count(), 8);
        assert_eq!(CurvedShutterQuality::High.sample_count(), 16);
        let mut zero = crate::motion::MotionParams::default();
        zero.shutter.quality = CurvedShutterQuality::High;
        assert!(zero.shutter.is_exact_zero());
        assert!(zero.is_exact_zero());

        let mut transport = ExportClipTransport::new(
            ClipTransportConfig {
                sample_fps: Some(30.0),
                ..ClipTransportConfig::default()
            },
            NormalizedTime::clamped(0.25),
            1.0,
            30,
        );
        let _ = transport.seed_selection();
        let advancing = transport.select(
            transport.authored,
            ProgramTransportTick {
                delta_seconds: 1.0 / 30.0,
                program_running: true,
                media_running: true,
                ..ProgramTransportTick::default()
            },
        );
        assert!(!advancing.held);
        assert_eq!(export_motion_held_scope(&graph, 0, advancing), None);
        let held = transport.select(
            transport.authored,
            ProgramTransportTick {
                delta_seconds: 1.0,
                program_running: true,
                media_running: false,
                ..ProgramTransportTick::default()
            },
        );
        assert!(held.held);
        assert_eq!(
            export_motion_held_scope(&graph, 0, held),
            Some(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
        );

        let before_seek: CodecMotionProduct = m4_codec_motion_frame(held.generation, 4).into();
        let seek = transport.select(
            transport.authored,
            ProgramTransportTick {
                program_running: true,
                media_running: true,
                seek_to: Some(NormalizedTime::clamped(0.75)),
                ..ProgramTransportTick::default()
            },
        );
        assert!(seek.discontinuity);
        assert_ne!(seek.generation, held.generation);
        assert!(
            matching_export_codec_motion(Some(before_seek), seek.generation, [64, 32]).is_none()
        );
    }

    fn m4_codec_past_reference_proof(
        source_generation: u64,
        destination_ordinal: u64,
        time_base_numerator: i32,
        time_base_denominator: i32,
    ) -> crate::video::CodecPastReferenceProof {
        let reference_ordinal = destination_ordinal
            .checked_sub(1)
            .expect("available codec fixture has an adjacent predecessor");
        let destination_pts =
            i64::try_from(destination_ordinal).expect("codec fixture ordinal fits the PTS domain");
        crate::video::CodecPastReferenceProof {
            policy: crate::video::AdjacentReferencePolicy::Mpeg4Part2SimpleProgressiveIp,
            reference: crate::video::CodecFrameIdentity {
                source_generation,
                pts: destination_pts - 1,
                presentation_ordinal: reference_ordinal,
            },
            destination: crate::video::CodecFrameIdentity {
                source_generation,
                pts: destination_pts,
                presentation_ordinal: destination_ordinal,
            },
            elapsed_ticks: 1,
            time_base: crate::video::CodecTimeBase::new(time_base_numerator, time_base_denominator)
                .expect("codec fixture time base is positive"),
        }
    }

    fn m4_codec_motion_frame(source_generation: u64, frame_ordinal: u64) -> CodecMotionFrame {
        CodecMotionFrame {
            source_dimensions: [64, 32],
            frame_delta_seconds: 1.0 / 30.0,
            source_generation,
            frame_ordinal,
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            provenance: crate::video::CodecMotionProvenance::FfmpegExportMvs,
            frame_type: crate::video::CodecMotionFrameType::Predictive,
            status: crate::video::CodecMotionStatus::Available,
            past_reference_proof: Some(m4_codec_past_reference_proof(
                source_generation,
                frame_ordinal,
                1,
                30,
            )),
            vectors: vec![crate::motion::CodecMotionVector {
                destination: [8, 8],
                block: [16, 16],
                motion: [-4, 0],
                motion_scale: 1,
                seconds_from_reference: 1.0 / 30.0,
                reference: crate::motion::CodecReferenceDirection::Past,
                visibility: 1.0,
            }],
        }
    }

    #[test]
    fn m4_export_codec_attachment_is_exact_and_missing_malformed_or_stale_stays_zero() {
        use crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts;

        let graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        let (master, params) = m4_motion_params(&graph);
        let evaluated = m4_motion_evaluated_frame(&graph, 30);
        let facts = [
            MotionCodecFrameFacts {
                available: true,
                source_generation: 11,
                frame_ordinal: 17,
            },
            MotionCodecFrameFacts::default(),
            MotionCodecFrameFacts::default(),
        ];
        let inputs =
            export_motion_layer_plan_inputs(&graph, &params, facts.into_iter().enumerate())
                .unwrap();
        let plan = m4_motion_plan(&evaluated, &graph, master, &inputs);
        let codec: CodecMotionProduct = m4_codec_motion_frame(11, 17).into();
        let sources = [(0, Some(&codec)), (1, None), (2, None)];
        let (products, diagnostics) = export_codec_motion_fields_from(&plan, &graph, sources);
        assert!(diagnostics.is_empty());
        assert_eq!(products.len(), 1);
        let attachment = products[0].attachment();
        let field_plan = plan
            .motion()
            .advanced()
            .unwrap()
            .fields()
            .iter()
            .find(|field| field.scope == attachment.scope)
            .unwrap();
        assert!(field_plan.accepts(attachment));
        let sample = products[0].field.sample(0, 0).unwrap();
        assert!(sample.velocity_uv_per_second[0] > 0.0);
        assert_eq!(sample.velocity_uv_per_second[1], 0.0);

        let mut rendered = plan
            .motion()
            .advanced()
            .unwrap()
            .fields()
            .iter()
            .map(|field| {
                if field.source.origin == crate::motion::MotionFieldOrigin::CodecVectors {
                    let product = products
                        .iter()
                        .find(|product| product.scope == field.scope)
                        .unwrap();
                    ExportRenderedMotionField {
                        scope: field.scope,
                        source_scope: field.scope,
                        origin: field.source.origin,
                        source_generation: Some(product.source_generation),
                        frame_ordinal: Some(product.frame_ordinal),
                        product_content_sha256: Some(product.codec_identity.content_sha256),
                    }
                } else {
                    ExportRenderedMotionField {
                        scope: field.scope,
                        source_scope: field.scope,
                        origin: field.source.origin,
                        source_generation: None,
                        frame_ordinal: None,
                        product_content_sha256: None,
                    }
                }
            })
            .collect::<Vec<_>>();
        let donor = *rendered
            .iter()
            .find(|field| {
                field.scope == crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0])
            })
            .unwrap();
        rendered.push(ExportRenderedMotionField {
            scope: crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[2]),
            ..donor
        });

        let metadata = export_motion_metadata_for_frame_from(
            &EvaluatedCompositionPlan::Advanced(Box::new(plan.clone())),
            &graph,
            [(0, Some(&codec)), (1, None), (2, None)],
            rendered.iter().copied(),
            23,
        );
        assert_eq!(metadata.accepted_frame, Some(23));
        assert_eq!(
            metadata.algorithm_version,
            crate::motion::MOTION_ALGORITHM_VERSION
        );
        assert!(!metadata.scopes_truncated);
        let codec_metadata = metadata
            .scopes
            .iter()
            .find(|scope| {
                matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 0,
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            codec_metadata.source_origin,
            crate::motion::MotionFieldOrigin::CodecVectors
        );
        assert_eq!(
            codec_metadata.rendered_source_origin,
            crate::motion::MotionFieldOrigin::CodecVectors
        );
        assert!(codec_metadata.field_attached);
        assert_eq!(
            codec_metadata.codec_provenance,
            Some(crate::video::CodecMotionProvenance::FfmpegExportMvs)
        );
        assert_eq!(codec_metadata.source_generation, Some(11));
        assert_eq!(codec_metadata.frame_ordinal, Some(17));
        assert_eq!(
            codec_metadata.codec_product_sha256,
            Some(products[0].codec_identity.content_sha256)
        );
        assert_eq!(codec_metadata.codec_transition_count, Some(1));
        assert_eq!(codec_metadata.codec_elapsed_seconds, Some(1.0 / 30.0));
        let unattached = export_motion_metadata_for_frame_from(
            &EvaluatedCompositionPlan::Advanced(Box::new(plan.clone())),
            &graph,
            [(0, Some(&codec)), (1, None), (2, None)],
            [],
            23,
        );
        let unattached_codec = unattached
            .scopes
            .iter()
            .find(|scope| {
                matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 0,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(!unattached_codec.field_attached);
        assert!(unattached_codec.field_planned);
        assert_eq!(
            unattached_codec.rendered_source_origin,
            crate::motion::MotionFieldOrigin::None
        );
        assert_eq!(unattached_codec.codec_provenance, None);
        assert_eq!(unattached_codec.codec_transition_count, None);
        assert_eq!(unattached_codec.codec_elapsed_seconds, None);
        let fallback_metadata = metadata
            .scopes
            .iter()
            .find(|scope| {
                matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 1,
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            fallback_metadata.source_origin,
            crate::motion::MotionFieldOrigin::LatticeFallback
        );
        assert_eq!(fallback_metadata.codec_provenance, None);
        let recipient_metadata = metadata
            .scopes
            .iter()
            .find(|scope| {
                matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 2,
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            recipient_metadata.source_origin,
            crate::motion::MotionFieldOrigin::LatticeFallback
        );
        assert_eq!(
            recipient_metadata.rendered_source_origin,
            crate::motion::MotionFieldOrigin::CodecVectors,
            "Faraday recipient metadata must describe its admitted donor field"
        );
        assert_eq!(
            recipient_metadata.codec_product_sha256,
            Some(products[0].codec_identity.content_sha256)
        );
        assert_eq!(
            recipient_metadata.codec_provenance,
            Some(crate::video::CodecMotionProvenance::FfmpegExportMvs)
        );

        let unavailable_inputs = export_motion_layer_plan_inputs(
            &graph,
            &params,
            (0..params.len()).map(|index| (index, MotionCodecFrameFacts::default())),
        )
        .unwrap();
        let unavailable_plan = m4_motion_plan(&evaluated, &graph, master, &unavailable_inputs);
        let (mut retry_products, retry_diagnostics) = export_codec_motion_fields_from(
            &plan,
            &graph,
            [(0, Some(&codec)), (1, None), (2, None)],
        );
        assert!(retry_diagnostics.is_empty());
        assert!(export_codec_fields_complete(
            &EvaluatedCompositionPlan::Advanced(Box::new(plan.clone())),
            &retry_products
        ));
        let fallback_plan = EvaluatedCompositionPlan::Advanced(Box::new(unavailable_plan.clone()));
        retain_exact_export_codec_fields(&fallback_plan, &mut retry_products);
        assert!(retry_products.is_empty());
        assert!(export_codec_fields_complete(
            &fallback_plan,
            &retry_products
        ));
        let unavailable_metadata = export_motion_metadata_for_frame_from(
            &EvaluatedCompositionPlan::Advanced(Box::new(unavailable_plan)),
            &graph,
            [(0, None), (1, None), (2, None)],
            [],
            25,
        );
        let explicit_unavailable = unavailable_metadata
            .scopes
            .iter()
            .find(|scope| {
                matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 0,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(explicit_unavailable.field_planned);
        assert!(!explicit_unavailable.field_attached);
        assert_eq!(
            explicit_unavailable.source_origin,
            crate::motion::MotionFieldOrigin::None
        );
        assert_eq!(
            explicit_unavailable.source_diagnostic,
            crate::motion::MotionSourceDiagnostic::CodecUnavailable
        );

        let (missing, missing_diagnostics) =
            export_codec_motion_fields_from(&plan, &graph, [(0, None), (1, None), (2, None)]);
        assert!(missing.is_empty());
        assert!(missing_diagnostics
            .iter()
            .any(|message| message.contains("lost its exact codec metadata")));

        let mut malformed = m4_codec_motion_frame(11, 17);
        malformed.vectors[0].motion_scale = 0;
        let malformed: CodecMotionProduct = malformed.into();
        let (malformed_products, malformed_diagnostics) = export_codec_motion_fields_from(
            &plan,
            &graph,
            [(0, Some(&malformed)), (1, None), (2, None)],
        );
        assert!(malformed_products.is_empty());
        assert!(malformed_diagnostics
            .iter()
            .any(|message| message.contains("lacks exact adjacent-reference identity")));

        let stale: CodecMotionProduct = m4_codec_motion_frame(12, 17).into();
        let (stale_products, stale_diagnostics) = export_codec_motion_fields_from(
            &plan,
            &graph,
            [(0, Some(&stale)), (1, None), (2, None)],
        );
        assert!(stale_products.is_empty());
        assert!(stale_diagnostics
            .iter()
            .any(|message| message.contains("failed exact attachment provenance")));

        let composed_facts = [
            MotionCodecFrameFacts {
                available: true,
                source_generation: 11,
                frame_ordinal: 18,
            },
            MotionCodecFrameFacts::default(),
            MotionCodecFrameFacts::default(),
        ];
        let composed_inputs = export_motion_layer_plan_inputs(
            &graph,
            &params,
            composed_facts.into_iter().enumerate(),
        )
        .unwrap();
        let composed_plan = m4_motion_plan(&evaluated, &graph, master, &composed_inputs);
        let mut sequence =
            crate::video::CodecMotionSequence::from_frame(m4_codec_motion_frame(11, 17)).unwrap();
        sequence
            .push_contiguous(m4_codec_motion_frame(11, 18))
            .unwrap();
        let composed: CodecMotionProduct = sequence.into();
        let (composed_products, composed_diagnostics) = export_codec_motion_fields_from(
            &composed_plan,
            &graph,
            [(0, Some(&composed)), (1, None), (2, None)],
        );
        assert!(composed_diagnostics.is_empty());
        assert_eq!(composed_products.len(), 1);
        assert_eq!(composed.transition_count(), 2);
        let composed_metadata = export_motion_metadata_for_frame_from(
            &EvaluatedCompositionPlan::Advanced(Box::new(composed_plan.clone())),
            &graph,
            [(0, Some(&composed)), (1, None), (2, None)],
            [ExportRenderedMotionField {
                scope: composed_products[0].scope,
                source_scope: composed_products[0].scope,
                origin: crate::motion::MotionFieldOrigin::CodecVectors,
                source_generation: Some(composed_products[0].source_generation),
                frame_ordinal: Some(composed_products[0].frame_ordinal),
                product_content_sha256: Some(composed_products[0].codec_identity.content_sha256),
            }],
            24,
        );
        assert_eq!(
            composed_metadata
                .scopes
                .iter()
                .find(|scope| matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 0,
                        ..
                    }
                ))
                .and_then(|scope| scope.codec_transition_count),
            Some(2)
        );
        assert_eq!(
            composed_metadata
                .scopes
                .iter()
                .find(|scope| matches!(
                    scope.scope,
                    ExportMotionScopeIdentity::Layer {
                        saved_position: 0,
                        ..
                    }
                ))
                .and_then(|scope| scope.codec_elapsed_seconds),
            Some(2.0 / 30.0)
        );
        assert!(composed_plan
            .motion()
            .advanced()
            .unwrap()
            .fields()
            .iter()
            .any(|field| field.accepts(composed_products[0].attachment())));
    }

    #[test]
    fn m4_export_two_pass_mixed_codec_failure_retains_good_product_before_gpu() {
        use crate::evaluated_frame::evaluated_composition::MotionCodecFrameFacts;
        use crate::motion::MotionFieldOrigin;

        let graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        let (master, params) = m4_motion_params(&graph);
        let evaluated = m4_motion_evaluated_frame(&graph, 30);
        let initial_facts = [
            MotionCodecFrameFacts {
                available: true,
                source_generation: 11,
                frame_ordinal: 17,
            },
            MotionCodecFrameFacts {
                available: true,
                source_generation: 11,
                frame_ordinal: 17,
            },
            MotionCodecFrameFacts::default(),
        ];
        let initial_inputs =
            export_motion_layer_plan_inputs(&graph, &params, initial_facts.into_iter().enumerate())
                .unwrap();
        let initial_plan = m4_motion_plan(&evaluated, &graph, master, &initial_inputs);
        let initial = EvaluatedCompositionPlan::Advanced(Box::new(initial_plan.clone()));
        assert_eq!(planned_export_codec_scopes(&initial).len(), 2);

        let good: CodecMotionProduct = m4_codec_motion_frame(11, 17).into();
        let mut malformed = m4_codec_motion_frame(11, 17);
        malformed.vectors[0].motion_scale = 0;
        let malformed: CodecMotionProduct = malformed.into();
        let (mut products, diagnostics) = export_codec_motion_fields_from(
            &initial_plan,
            &graph,
            [(0, Some(&good)), (1, Some(&malformed)), (2, None)],
        );
        assert_eq!(products.len(), 1);
        assert_eq!(
            products[0].scope,
            crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0])
        );
        assert!(diagnostics
            .iter()
            .any(|message| message.contains("lacks exact adjacent-reference identity")));
        assert!(
            !export_codec_fields_complete(&initial, &products),
            "the first pass must detect the missing second codec field before GPU encode"
        );
        let retained_digest = products[0].codec_identity.content_sha256;

        // This is the immutable production retry shape: exact availability is
        // recomputed from successfully rasterized products, the plan retries
        // once, and the successful first-pass product is retained rather than
        // rasterized a second time.
        let attached = products
            .iter()
            .map(|product| product.scope)
            .collect::<std::collections::BTreeSet<_>>();
        let retry_facts = graph
            .layer_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, layer_id)| {
                let available =
                    attached.contains(&crate::visual_rack::VisualScopeId::Layer(layer_id));
                MotionCodecFrameFacts {
                    available,
                    source_generation: if available {
                        initial_facts[index].source_generation
                    } else {
                        0
                    },
                    frame_ordinal: if available {
                        initial_facts[index].frame_ordinal
                    } else {
                        0
                    },
                }
            });
        let retry_inputs =
            export_motion_layer_plan_inputs(&graph, &params, retry_facts.enumerate()).unwrap();
        let retry_plan = m4_motion_plan(&evaluated, &graph, master, &retry_inputs);
        let retry = EvaluatedCompositionPlan::Advanced(Box::new(retry_plan.clone()));
        retain_exact_export_codec_fields(&retry, &mut products);

        assert_eq!(products.len(), 1);
        assert_eq!(products[0].codec_identity.content_sha256, retained_digest);
        assert!(export_codec_fields_complete(&retry, &products));
        assert_eq!(
            planned_export_codec_scopes(&retry),
            products
                .iter()
                .map(|product| product.scope)
                .collect::<std::collections::BTreeSet<_>>()
        );
        let motion = retry_plan.motion().advanced().unwrap();
        assert_eq!(
            motion
                .scope(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
                .unwrap()
                .source
                .origin,
            MotionFieldOrigin::CodecVectors
        );
        assert_eq!(
            motion
                .scope(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[1]))
                .unwrap()
                .source
                .origin,
            MotionFieldOrigin::LatticeFallback
        );
    }

    #[test]
    fn m4_motion_sidecar_is_atomic_bounded_versioned_and_cleanup_coupled() {
        use crate::motion::{
            CurvedShutterQuality, MotionCarrier, MotionFieldOrigin, MotionFieldSource,
            MotionLatticeQuality, MotionSourceDiagnostic,
        };

        let unique = format!(
            "collide-o-scope-motion-sidecar-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("artifact.mp4");
        std::fs::write(&output, b"accepted-video").unwrap();

        let mut export_config = config(1.0);
        export_config.width = 64;
        export_config.height = 32;
        export_config.fps = 30;
        export_config.output_path = output.to_string_lossy().into_owned();
        export_config.shutter_samples = ExportShutterSamples::Samples16;
        let mut accumulator = ExportMotionSidecarAccumulator {
            artifact: MotionSidecarArtifact::from_config(&export_config),
            sources: vec![MotionSidecarSource {
                saved_position: 0,
                stable_id: 1,
                source_tap_id: export_selective_layer_id(0),
                kind: "video".to_owned(),
                logical_name: "donor.mov".to_owned(),
                persisted_reference: "sha256:fixture".to_owned(),
                fingerprint_sha256: Some("a".repeat(64)),
                fingerprint_bytes: Some(1234),
            }],
            sources_truncated: false,
            authored_scopes: vec![sidecar_authored_scope(
                MotionSidecarScopeIdentity::Master,
                crate::motion::MotionParams::default(),
            )],
            authored_scopes_truncated: false,
            residual_nodes: Vec::new(),
            residual_nodes_truncated: false,
            distinct: Vec::new(),
            distinct_truncated: false,
            symmetry_fields: Vec::new(),
            symmetry_fields_truncated: false,
            field_collider: None,
            codec_mosh: None,
            last: None,
        };
        let scope = ExportMotionScopeMetadata {
            scope: ExportMotionScopeIdentity::Layer {
                saved_position: 0,
                stable_id: 1,
                source_tap_id: export_selective_layer_id(0),
            },
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            requested_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::Live,
            source_origin: MotionFieldOrigin::CodecVectors,
            rendered_source_origin: MotionFieldOrigin::CodecVectors,
            field_planned: true,
            field_attached: true,
            source_diagnostic: MotionSourceDiagnostic::None,
            codec_provenance: Some(crate::video::CodecMotionProvenance::FfmpegExportMvs),
            source_generation: Some(1),
            frame_ordinal: Some(7),
            codec_product_sha256: Some([7; 32]),
            codec_transition_count: Some(4),
            codec_elapsed_seconds: Some(4.0 / 30.0),
            donor_saved_position: Some(0),
            donor_stable_id: Some(1),
            carrier: MotionCarrier::FirstSourceFrame,
            transplant_admitted: true,
            shutter_active: true,
            shutter_angle_degrees: 180.0,
            shutter_quality: CurvedShutterQuality::High,
            shutter_sample_count: CurvedShutterQuality::High.sample_count(),
        };
        accumulator.observe_accepted(&ExportMotionMetadata {
            accepted_frame: Some(0),
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: vec![scope.clone()],
            scopes_truncated: false,
            field_collider: None,
        });
        let mut same_dynamic_state = scope.clone();
        same_dynamic_state.source_generation = Some(2);
        same_dynamic_state.frame_ordinal = Some(8);
        same_dynamic_state.codec_product_sha256 = Some([8; 32]);
        accumulator.observe_accepted(&ExportMotionMetadata {
            accepted_frame: Some(1),
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: vec![same_dynamic_state],
            scopes_truncated: false,
            field_collider: None,
        });
        let mut fallback = scope.clone();
        fallback.source_origin = MotionFieldOrigin::LatticeFallback;
        fallback.rendered_source_origin = MotionFieldOrigin::LatticeFallback;
        fallback.field_attached = true;
        fallback.source_diagnostic = MotionSourceDiagnostic::CodecUnavailableFallback;
        fallback.codec_provenance = None;
        fallback.source_generation = None;
        fallback.frame_ordinal = None;
        fallback.codec_product_sha256 = None;
        fallback.codec_transition_count = None;
        fallback.codec_elapsed_seconds = None;
        accumulator.observe_accepted(&ExportMotionMetadata {
            accepted_frame: Some(2),
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: vec![fallback],
            scopes_truncated: false,
            field_collider: None,
        });
        accumulator.observe_accepted(&ExportMotionMetadata {
            accepted_frame: Some(3),
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: vec![scope; MAX_EXPORT_MOTION_SIDECAR_SCOPES + 3],
            scopes_truncated: true,
            field_collider: None,
        });
        let warnings = (0..MAX_EXPORT_WARNINGS + 5)
            .map(|index| format!("warning-{index}"))
            .collect();
        let sidecar = accumulator.finish(warnings);
        assert_eq!(sidecar.schema_version, EXPORT_MOTION_SIDECAR_SCHEMA_VERSION);
        assert!(!sidecar.cross_gpu_pixel_identity_guaranteed);
        assert_eq!(sidecar.distinct_dynamic_states.len(), 2);
        assert_eq!(sidecar.warnings.len(), MAX_EXPORT_WARNINGS);
        let last = sidecar.last_accepted_frame.as_ref().unwrap();
        assert_eq!(last.accepted_frame, 3);
        assert_eq!(last.scopes.len(), MAX_EXPORT_MOTION_SIDECAR_SCOPES);
        assert!(last.scopes_truncated);

        let output_string = output.to_string_lossy().into_owned();
        let sidecar_path = motion_sidecar_path(&output_string);
        std::fs::write(&sidecar_path, b"stale-report").unwrap();
        write_motion_sidecar_atomic(&output_string, &sidecar).unwrap();
        let bytes = std::fs::read(&sidecar_path).unwrap();
        assert!(bytes.len() <= MAX_EXPORT_MOTION_SIDECAR_BYTES);
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Schema 4 appended the per-slot Symmetry Field route section and the
        // Residual Counterpoint section; schema 5 appended the Field Collider
        // section; schema 6 appended Codec Mosh and schema 7 appended its
        // motion shaping plus frame-evaluated recipe bounds. Every earlier
        // key below is re-asserted unchanged, and a job that never ran the
        // collider or the mosh omits both sections entirely.
        assert_eq!(json["schema_version"], 7);
        assert!(json.get("field_collider").is_none());
        assert!(json.get("codec_mosh").is_none());
        assert!(json["symmetry_fields"].as_array().unwrap().is_empty());
        assert_eq!(json["symmetry_fields_truncated"], false);
        assert_eq!(json["cross_gpu_pixel_identity_guaranteed"], false);
        // Schema 4's residual section is present and empty for a patch with no
        // Residual node, so an existing consumer sees no dynamic-state change.
        assert_eq!(
            json["authored_residual_nodes"],
            serde_json::Value::Array(Vec::new())
        );
        assert_eq!(json["authored_residual_nodes_truncated"], false);
        assert_eq!(
            json["artifact"]["requested_shutter_sample_policy"],
            "samples_16"
        );
        assert_eq!(json["artifact"]["requested_shutter_sample_count"], 16);
        assert_eq!(json["authored_scopes"][0]["shutter_sample_count"], 1);
        assert_eq!(json["sources"][0]["fingerprint_bytes"], 1234);
        assert_eq!(json["last_accepted_frame"]["accepted_frame"], 3);
        assert_eq!(
            json["last_accepted_frame"]["scopes"][0]["shutter_sample_count"],
            16
        );
        assert_eq!(
            json["last_accepted_frame"]["scopes"][0]["codec_transition_count"],
            4
        );
        assert_eq!(
            json["last_accepted_frame"]["scopes"][0]["rendered_source_origin"],
            "codec_vectors"
        );
        assert_eq!(
            json["last_accepted_frame"]["scopes"][0]["field_attached"],
            true
        );
        let codec_elapsed = json["last_accepted_frame"]["scopes"][0]["codec_elapsed_seconds"]
            .as_f64()
            .unwrap();
        assert!((codec_elapsed - f64::from(4.0_f32 / 30.0)).abs() < 1.0e-7);
        assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".motion.json.tmp-")
        }));

        let progress = ExportProgress::new();
        progress.output_started.store(true, Ordering::Release);
        assert_eq!(remove_started_output(&progress, Some(&output_string)), None);
        assert!(!output.exists());
        assert!(!sidecar_path.exists());
        std::fs::remove_dir(&directory).unwrap();
    }

    /// Export provenance records both image slots and both motion slots of
    /// every Symmetry Field by slot index, and keeps a resolved route
    /// distinguishable from a retained tombstone.
    ///
    /// Each pair below names the *same* saved position twice: slot 0 selects
    /// it and slot 1 is a retained tombstone at it. A walker that compacted
    /// slots, or that re-resolved a tombstone against whatever now occupies
    /// the vacated position, would make the two records identical.
    #[test]
    fn symmetry_export_provenance_records_both_image_and_motion_slots_by_index_with_missing_identity(
    ) {
        use crate::symmetry::{
            SavedMotionDonor, SymmetryMotionMask, SymmetryParams, SymmetrySourceMask,
        };
        use crate::visual_rack::{
            EdgeTiming, LegacyRackScope, SavedImageSource, SavedImageTap, VisualNodeKind,
            VisualRack,
        };

        let one = SavedLayerPosition::new(1).unwrap();
        let two = SavedLayerPosition::new(2).unwrap();
        let mut patch = three_layer_legacy_patch();

        let mut layer_rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        layer_rack
            .push(VisualNodeKind::Symmetry(SymmetryParams {
                base_folds: 6.0,
                source_mask: SymmetrySourceMask {
                    carrier: true,
                    donor0: true,
                    donor1: true,
                    clean_history: false,
                },
                motion_mask: SymmetryMotionMask {
                    slot0: true,
                    slot1: true,
                },
                donors: [
                    SavedImageTap {
                        source: SavedImageSource::SelectedLayer {
                            layer_position: two,
                            stage: crate::image_routing::LayerImageStage::PreLocalEffects,
                        },
                        timing: EdgeTiming::PreviousFrame,
                    },
                    SavedImageTap {
                        source: SavedImageSource::MissingSelectedLayer {
                            saved_position: two,
                            stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                        },
                        timing: EdgeTiming::CurrentFrame,
                    },
                ],
                motion: [
                    SavedMotionDonor::Selected {
                        saved_position: one,
                    },
                    SavedMotionDonor::Missing {
                        saved_position: one,
                    },
                ],
                ..SymmetryParams::default()
            }))
            .unwrap();
        patch.layers[0].rack = Some(layer_rack);

        // A second, entirely default node proves an unarmed slot is still
        // recorded at its own index and is never reported as missing.
        let mut master_rack = VisualRack::synthetic_legacy(LegacyRackScope::Master);
        master_rack
            .push(VisualNodeKind::Symmetry(SymmetryParams::default()))
            .unwrap();
        patch.master_rack = Some(master_rack);

        let graph = resolve_export_creative_graph(&patch).unwrap();
        let (nodes, truncated) = export_symmetry_sidecar_nodes(&graph);
        assert!(!truncated);
        assert_eq!(nodes.len(), 2, "one layer-scope node and one master node");

        let value = serde_json::to_value(&nodes).unwrap();
        let layer = &value[0];
        assert_eq!(layer["scope"]["kind"], "layer");
        assert_eq!(layer["scope"]["saved_position"], 0);
        assert_eq!(layer["scope"]["stable_id"], 1);
        assert_eq!(layer["exact_bypass"], false);

        let images = layer["image_routes"].as_array().unwrap();
        assert_eq!(images.len(), crate::symmetry::SYMMETRY_IMAGE_SLOTS);
        assert_eq!(images[0]["slot"], 0);
        assert_eq!(images[0]["armed"], true);
        assert_eq!(images[0]["source"], "selected_layer");
        assert_eq!(images[0]["resolved"], true);
        // Export identity is position + 1, so saved position 2 is stable ID 3.
        assert_eq!(images[0]["stable_id"], 3);
        assert_eq!(images[0]["saved_position"], 2);
        assert_eq!(images[0]["timing"], "previous_frame");
        assert_eq!(images[0]["stage"], "pre_local_effects");
        assert_eq!(images[1]["slot"], 1);
        assert_eq!(images[1]["armed"], true);
        assert_eq!(images[1]["source"], "missing_selected_layer");
        assert_eq!(images[1]["resolved"], false);
        assert_eq!(images[1]["stable_id"], serde_json::Value::Null);
        assert_eq!(images[1]["saved_position"], 2);
        assert_eq!(images[1]["stage"], "post_local_effects");

        let motions = layer["motion_routes"].as_array().unwrap();
        assert_eq!(motions.len(), crate::symmetry::SYMMETRY_MOTION_SLOTS);
        assert_eq!(motions[0]["slot"], 0);
        assert_eq!(motions[0]["armed"], true);
        assert_eq!(motions[0]["donor"], "selected");
        assert_eq!(motions[0]["resolved"], true);
        assert_eq!(motions[0]["stable_id"], 2);
        assert_eq!(motions[0]["saved_position"], 1);
        assert_eq!(motions[1]["slot"], 1);
        assert_eq!(motions[1]["armed"], true);
        assert_eq!(motions[1]["donor"], "missing");
        assert_eq!(motions[1]["resolved"], false);
        assert_eq!(motions[1]["stable_id"], serde_json::Value::Null);
        assert_eq!(motions[1]["saved_position"], 1);

        let master = &value[1];
        assert_eq!(master["scope"]["kind"], "master");
        assert_eq!(master["exact_bypass"], true);
        assert_eq!(master["mode"], "cyclic");
        assert_eq!(master["boundary"], "transparent");
        for slot in 0..crate::symmetry::SYMMETRY_IMAGE_SLOTS {
            assert_eq!(master["image_routes"][slot]["slot"], slot);
            assert_eq!(master["image_routes"][slot]["armed"], false);
            assert_eq!(master["image_routes"][slot]["source"], "one_below");
            assert_eq!(master["image_routes"][slot]["resolved"], true);
        }
        for slot in 0..crate::symmetry::SYMMETRY_MOTION_SLOTS {
            assert_eq!(master["motion_routes"][slot]["slot"], slot);
            assert_eq!(master["motion_routes"][slot]["armed"], false);
            assert_eq!(master["motion_routes"][slot]["donor"], "none");
            assert_eq!(master["motion_routes"][slot]["resolved"], true);
        }

        // Provenance is shareable: no operational path or filesystem metadata.
        let text = serde_json::to_string(&nodes).unwrap();
        assert!(!text.contains("top.mp4"));
        assert!(!text.contains("middle.mp4"));
        assert!(!text.contains(".mp4"));
    }

    /// Export reaches the dedicated eight-texture Symmetry Field step through
    /// the one shared planner entry, at every export rate, and is refused when
    /// the reported device ceiling cannot bind eight textures.
    ///
    /// This is the "no export-only symmetry path" proof at the planner seam:
    /// the offline job builds the same `EvaluatedCompositionPlan::Advanced`
    /// carrying the same `EvaluatedScopeStep::SymmetryField` that the live
    /// executor consumes, and its topology does not move with the frame rate.
    #[test]
    fn export_planning_emits_the_dedicated_symmetry_step_and_refuses_it_below_the_device_floor() {
        use crate::evaluated_frame::evaluated_composition::EvaluatedScopeStep;
        use crate::symmetry::{SymmetryMode, SymmetryParams, SymmetrySourceMask};
        use crate::visual_rack::{
            EdgeTiming, LegacyRackScope, SavedImageSource, SavedImageTap, VisualNodeKind,
            VisualRack,
        };

        let mut patch = three_layer_legacy_patch();
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        let node_id = rack
            .push(VisualNodeKind::Symmetry(SymmetryParams {
                mode: SymmetryMode::Dihedral,
                base_folds: 6.0,
                source_mask: SymmetrySourceMask {
                    carrier: true,
                    donor0: true,
                    donor1: false,
                    clean_history: true,
                },
                donors: [
                    SavedImageTap {
                        source: SavedImageSource::OneBelow,
                        timing: EdgeTiming::CurrentFrame,
                    },
                    SavedImageTap::default(),
                ],
                ..SymmetryParams::default()
            }))
            .unwrap();
        patch.layers[0].rack = Some(rack);

        let graph = resolve_export_creative_graph(&patch).unwrap();
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let evaluate = |fps: u32| {
            let modulation = crate::modulation::ModMatrix::new().frame(3);
            EvaluatedFramePlan::evaluate(
                &modulation,
                FramePlanContext::new(64, 36, fps as f32 / fps as f32),
                MasterFrameInput {
                    effects: &effects,
                    transform: &transform,
                    ntsc: &ntsc,
                    temporal: &temporal,
                },
                graph
                    .layer_ids
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(slot, id)| LayerFrameInput {
                        source: SourceTap::new(id.get(), slot, 64, 36),
                        effects: &effects,
                        transform: &transform,
                        opacity: 1.0,
                        mosh_send: 1.0,
                        speed: 1.0,
                        fps: fps as f32,
                        blend_mode: BlendMode::Normal,
                        visible: true,
                        paused: false,
                        bypass_master_fx: false,
                        bypass_temporal_fx: false,
                        pattern: None,
                    }),
            )
        };
        // The enforced device floor, exactly as `creative_resource_limits`
        // reads it from the real adapter in `run_export`.
        let floor_limits = CreativeResourceLimits {
            max_sampled_textures_per_shader_stage: wgpu::Limits::default()
                .max_sampled_textures_per_shader_stage,
            ..CreativeResourceLimits::default()
        };

        let mut signature = None;
        for fps in [24_u32, 30, 60] {
            let evaluated = evaluate(fps);
            let planned = plan_export_composition(
                &evaluated,
                &graph,
                &[LayerMatte::default(); 3],
                false,
                floor_limits,
            )
            .unwrap();
            let EvaluatedCompositionPlan::Advanced(advanced) = planned else {
                panic!("a dedicated Symmetry Field must enter the unified advanced plan");
            };
            let layer = advanced
                .layers()
                .iter()
                .find(|layer| layer.stable_id == StableLayerId::new(1).unwrap())
                .expect("layer 0 is planned");
            let field = layer
                .execution
                .steps()
                .iter()
                .find_map(|step| match step {
                    EvaluatedScopeStep::SymmetryField { plan } => Some(plan),
                    _ => None,
                })
                .expect("the offline plan carries the dedicated step");
            assert_eq!(field.node_id, node_id);
            assert!(field.enabled);

            let resources = advanced.symmetry_field_resources();
            assert_eq!(resources.full_frame_passes, 1);
            assert_eq!(resources.max_sampled_textures_in_pass, 8);
            assert_eq!(resources.uniform_bytes, 1_024);

            if let Some(expected) = signature {
                assert_eq!(
                    advanced.topology_signature(),
                    expected,
                    "the dedicated step's topology cannot depend on the export rate"
                );
            } else {
                signature = Some(advanced.topology_signature());
            }
        }

        // The same authored patch is refused where the reported ceiling is the
        // ordinary rack constant, so the dedicated admission is genuinely live
        // at the export entry and is not clamped down to three.
        let evaluated = evaluate(30);
        assert_eq!(
            CreativeResourceLimits::default().max_sampled_textures_per_shader_stage,
            crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_PASS
        );
        let refused = plan_export_composition(
            &evaluated,
            &graph,
            &[LayerMatte::default(); 3],
            false,
            CreativeResourceLimits::default(),
        );
        assert!(
            refused.is_err(),
            "eight simultaneous bindings cannot be admitted against a three-texture ceiling"
        );
    }

    /// The 32-record sector table of the same authored patch is identical in
    /// the live program and in its offline render.
    ///
    /// A live `StableLayerId` is process lifetime: it is deliberately not
    /// serialized, an export job numbers layers `position + 1`, and a fresh
    /// process or a replaced clip mints different values for the same authored
    /// layer. Anything that feeds one of those values into the table domain
    /// makes the exported file disagree with the program it was rendered from
    /// and rerolls a saved node's records on every reload. This fixture drives
    /// export's own resolution against the shared planner's live-style
    /// resolution and compares the tables the two dedicated steps carry.
    #[test]
    fn the_symmetry_sector_table_is_identical_live_and_offline_for_the_same_authored_patch() {
        use crate::evaluated_frame::evaluated_composition::{
            CompositionPlanInput, EvaluatedScopeStep,
        };
        use crate::symmetry::{SymmetryMode, SymmetryParams, SymmetrySourceMask};
        use crate::visual_rack::{
            EdgeTiming, LegacyRackScope, SavedImageSource, SavedImageTap, VisualNodeKind,
            VisualRack,
        };

        let mut patch = three_layer_legacy_patch();
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        let node_id = rack
            .push(VisualNodeKind::Symmetry(SymmetryParams {
                mode: SymmetryMode::Dihedral,
                base_folds: 5.0,
                seed: 0x0BAD_5EED,
                hue_span: 0.4,
                source_mask: SymmetrySourceMask {
                    carrier: true,
                    donor0: true,
                    donor1: false,
                    clean_history: true,
                },
                donors: [
                    SavedImageTap {
                        source: SavedImageSource::OneBelow,
                        timing: EdgeTiming::CurrentFrame,
                    },
                    SavedImageTap::default(),
                ],
                ..SymmetryParams::default()
            }))
            .unwrap();
        patch.layers[0].rack = Some(rack);

        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let evaluate = |ids: &[StableLayerId]| {
            let modulation = crate::modulation::ModMatrix::new().frame(3);
            EvaluatedFramePlan::evaluate(
                &modulation,
                FramePlanContext::new(64, 36, 1.0),
                MasterFrameInput {
                    effects: &effects,
                    transform: &transform,
                    ntsc: &ntsc,
                    temporal: &temporal,
                },
                ids.iter()
                    .copied()
                    .enumerate()
                    .map(|(slot, id)| LayerFrameInput {
                        source: SourceTap::new(id.get(), slot, 64, 36),
                        effects: &effects,
                        transform: &transform,
                        opacity: 1.0,
                        mosh_send: 1.0,
                        speed: 1.0,
                        fps: 30.0,
                        blend_mode: BlendMode::Normal,
                        visible: true,
                        paused: false,
                        bypass_master_fx: false,
                        bypass_temporal_fx: false,
                        pattern: None,
                    }),
            )
        };
        let floor_limits = CreativeResourceLimits {
            max_sampled_textures_per_shader_stage: wgpu::Limits::default()
                .max_sampled_textures_per_shader_stage,
            ..CreativeResourceLimits::default()
        };

        let table_of = |plan: &EvaluatedCompositionPlan, id: StableLayerId| {
            let EvaluatedCompositionPlan::Advanced(advanced) = plan else {
                panic!("a dedicated Symmetry Field must enter the unified advanced plan");
            };
            let layer = advanced
                .layers()
                .iter()
                .find(|layer| layer.stable_id == id)
                .expect("the authoring layer is planned");
            let field = layer
                .execution
                .steps()
                .iter()
                .find_map(|step| match step {
                    EvaluatedScopeStep::SymmetryField { plan } => Some(plan),
                    _ => None,
                })
                .expect("the dedicated step exists");
            assert_eq!(field.node_id, node_id);
            field.params.sector_table(field.domain)
        };

        // Export's own resolution: layer identity is position + 1.
        let export_graph = resolve_export_creative_graph(&patch).unwrap();
        assert_eq!(export_graph.layer_ids[0], StableLayerId::new(1).unwrap());
        let export_plan = plan_export_composition(
            &evaluate(&export_graph.layer_ids),
            &export_graph,
            &[LayerMatte::default(); 3],
            false,
            floor_limits,
        )
        .unwrap();
        let export_table = table_of(&export_plan, StableLayerId::new(1).unwrap());

        // A live host resolving the same patch after ordinary layer churn. The
        // process-global counter has already handed out 909 identities, so the
        // same authored layer is a completely different live ID.
        let live_ids = [910_u64, 911, 912].map(|id| StableLayerId::new(id).unwrap());
        let live_composition = patch
            .composition
            .clone()
            .unwrap_or_else(|| {
                let back_to_front = (0..patch.layers.len())
                    .rev()
                    .map(|position| SavedLayerPosition::new(position as u32).unwrap())
                    .collect::<Vec<_>>();
                crate::composition::CompositionTree::legacy_for_layers(&back_to_front).unwrap()
            })
            .resolve(|position| live_ids.get(position.get() as usize).copied())
            .unwrap();
        let group_exists = |group_id| live_composition.contains_group(group_id);
        let live_master = patch.effective_master_rack().resolve_routes(
            |position| live_ids.get(position.get() as usize).copied(),
            group_exists,
        );
        let live_racks = (0..patch.layers.len())
            .map(|position| {
                (
                    live_ids[position],
                    patch
                        .effective_layer_rack(position)
                        .unwrap()
                        .resolve_routes(
                            |saved| live_ids.get(saved.get() as usize).copied(),
                            group_exists,
                        ),
                )
            })
            .collect::<Vec<_>>();
        let mut live_input =
            CompositionPlanInput::new(&live_composition, &live_master, &live_racks);
        live_input.resource_limits = floor_limits;
        let live_plan =
            EvaluatedCompositionPlan::evaluate(&evaluate(&live_ids), live_input).unwrap();
        let live_table = table_of(&live_plan, live_ids[0]);

        assert_eq!(
            live_table, export_table,
            "the sector table is authored identity; it cannot depend on a \
             process-lifetime layer ID"
        );

        // The same node after a stack reorder. Its saved position moves from 0
        // to 2, so export now numbers it 3 rather than 1 — and the table it
        // carries is still the same table.
        let mut reordered = patch.clone();
        let moved = reordered.layers.remove(0);
        reordered.layers.push(moved);
        let reordered_graph = resolve_export_creative_graph(&reordered).unwrap();
        let reordered_plan = plan_export_composition(
            &evaluate(&reordered_graph.layer_ids),
            &reordered_graph,
            &[LayerMatte::default(); 3],
            false,
            floor_limits,
        )
        .unwrap();
        assert_eq!(
            table_of(&reordered_plan, StableLayerId::new(3).unwrap()),
            export_table,
            "reordering the stack must never reroll a sector table"
        );
    }

    /// The Symmetry Field's only time input is the shared frame-plan context,
    /// which export derives from `frame_num` and the export FPS. The same
    /// instant therefore packs a byte-identical 1,024-byte record at 24, 30 and
    /// 60 fps — the dedicated pass reads no wall clock and no per-rate counter.
    #[test]
    fn symmetry_field_uniforms_are_frame_index_derived_and_exact_at_equal_24_30_and_60_fps_times() {
        use crate::symmetry::{
            SymmetryFrameUniforms, SymmetryGpuBindings, SymmetryGpuUniforms, SymmetryMode,
            SymmetryNodeDomain, SymmetryParams, SymmetrySourceMask,
        };
        use crate::visual_rack::VisualScopeId;

        let params = SymmetryParams {
            mode: SymmetryMode::LogSpiral,
            base_folds: 7.0,
            radial_phase_deg: 41.0,
            orbit_phase: 0.375,
            spiral_scale: 0.6,
            hue_span: 0.5,
            motion_gain: 0.25,
            seed: 0x5359_4d4d,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: true,
                clean_history: true,
            },
            ..SymmetryParams::default()
        };
        let domain = SymmetryNodeDomain::for_scope(VisualScopeId::Master, 9_001);
        let table = params.sector_table(domain);
        let mut canonical: Option<Vec<u8>> = None;
        for fps in [24_u32, 30, 60] {
            // One second in, expressed the way `run_export` expresses it.
            let frame_interval = 1.0 / fps as f32;
            let (time, _) = export_program_transport(u64::from(fps), frame_interval, false);
            assert_eq!(
                time, 1.0,
                "the frame-index derivation must land on the same instant at {fps} fps"
            );
            let packed = SymmetryGpuUniforms::pack(
                params,
                (1920, 1080),
                &table,
                SymmetryGpuBindings::default(),
                SymmetryFrameUniforms {
                    wet: 0.75,
                    blend_code: 2,
                    time_seconds: time,
                },
            );
            let evidence = bytemuck::bytes_of(&packed).to_vec();
            assert_eq!(evidence.len(), 1_024);
            if let Some(expected) = &canonical {
                assert_eq!(&evidence, expected);
            } else {
                canonical = Some(evidence);
            }
        }

        // A different instant must actually move the record, so the equality
        // above is not vacuous.
        let moved = SymmetryGpuUniforms::pack(
            params,
            (1920, 1080),
            &table,
            SymmetryGpuBindings::default(),
            SymmetryFrameUniforms {
                wet: 0.75,
                blend_code: 2,
                time_seconds: 2.0,
            },
        );
        assert_ne!(
            bytemuck::bytes_of(&moved).to_vec(),
            canonical.expect("canonical bytes")
        );
    }

    /// One recorded two-stroke performance, addressed on the 30 Hz timeline the
    /// live host records on. It is built through `GestureTrack::record_accepted`
    /// rather than assembled by hand, so the fixture is exactly the shape a live
    /// session produces.
    pub(super) fn recorded_gesture_fixture() -> crate::gesture::GestureTrack {
        use crate::gesture::{GestureEvent, GestureMode, GesturePhase, GestureTrack};

        /// One authored point of the fixture. A named record rather than a
        /// six-wide tuple so the stroke, phase, and mode stay readable.
        struct Point {
            tick: u64,
            stroke: u8,
            phase: GesturePhase,
            mode: GestureMode,
            position: [f32; 2],
            direction: [f32; 2],
        }

        let mut track = GestureTrack::default();
        let points = [
            Point {
                tick: 11,
                stroke: 0,
                phase: GesturePhase::Begin,
                mode: GestureMode::Push,
                position: [0.30, 0.50],
                direction: [1.0, 0.0],
            },
            Point {
                tick: 12,
                stroke: 1,
                phase: GesturePhase::Begin,
                mode: GestureMode::Curl,
                position: [0.70, 0.30],
                direction: [0.0, 1.0],
            },
            Point {
                tick: 13,
                stroke: 0,
                phase: GesturePhase::Move,
                mode: GestureMode::Push,
                position: [0.42, 0.51],
                direction: [1.0, 0.1],
            },
            Point {
                tick: 16,
                stroke: 1,
                phase: GesturePhase::Move,
                mode: GestureMode::Curl,
                position: [0.68, 0.44],
                direction: [-0.2, 1.0],
            },
            Point {
                tick: 20,
                stroke: 0,
                phase: GesturePhase::Move,
                mode: GestureMode::Push,
                position: [0.55, 0.53],
                direction: [1.0, 0.2],
            },
            Point {
                tick: 27,
                stroke: 0,
                phase: GesturePhase::End,
                mode: GestureMode::Push,
                position: [0.66, 0.56],
                direction: [1.0, 0.25],
            },
            Point {
                tick: 31,
                stroke: 1,
                phase: GesturePhase::End,
                mode: GestureMode::Curl,
                position: [0.66, 0.58],
                direction: [-0.4, 1.0],
            },
        ];
        for point in points {
            let event = GestureEvent::quantized(
                point.stroke,
                point.phase,
                point.mode,
                point.position,
                0.8,
                point.direction,
            );
            assert!(
                track.record_accepted(point.tick, event).unwrap(),
                "the fixture must fit inside the bounded track"
            );
        }
        assert!(track.is_complete());
        track
    }

    fn gesture_fixture_canvas(
        params: crate::gesture_canvas::GestureCanvasParams,
    ) -> crate::gesture_canvas::GestureCanvasState {
        let grid = crate::gesture_canvas_host_grid(320, 180);
        crate::gesture_canvas::GestureCanvasState::new(grid, params).unwrap()
    }

    #[test]
    fn a_recorded_gesture_sidecar_round_trips_and_carries_no_path_or_filesystem_metadata() {
        let unique = format!(
            "collide-o-scope-gesture-sidecar-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(&unique);
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("artifact.mp4");
        std::fs::write(&output, b"accepted-video").unwrap();
        let output_string = output.to_string_lossy().into_owned();

        let track = recorded_gesture_fixture();
        let document = crate::gesture::GestureTrackDocument::capture(&track);
        write_gesture_sidecar_noreplace(&output_string, &document).unwrap();

        let sidecar_path = gesture_sidecar_path(&output_string);
        assert_eq!(
            sidecar_path.file_name().unwrap(),
            "artifact.mp4.gesture.json"
        );
        let bytes = std::fs::read(&sidecar_path).unwrap();
        assert!(bytes.len() <= crate::gesture::MAX_GESTURE_SERIALIZED_BYTES);

        // The frozen six-field sidecar, and nothing else.
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let mut keys = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "checksum",
                "event_count",
                "events",
                "origin_tick",
                "truncated",
                "version"
            ]
        );
        assert_eq!(
            json["version"],
            u64::from(crate::gesture::GESTURE_ALGORITHM_VERSION)
        );
        assert_eq!(json["origin_tick"], 11);
        assert_eq!(json["event_count"], 7);
        assert_eq!(json["truncated"], false);

        // Operational paths and filesystem metadata never enter the payload.
        let text = String::from_utf8(bytes.clone()).unwrap();
        for forbidden in [
            unique.as_str(),
            "artifact.mp4",
            ".mp4",
            "renders",
            "Temp",
            "temp",
            "path",
            "directory",
            "modified",
            "created",
            "accessed",
        ] {
            assert!(
                !text.contains(forbidden),
                "gesture sidecar must not carry {forbidden}: {text}"
            );
        }

        // Round trip: the bytes decode to the identical document, the identical
        // track, and the identical canonical digest.
        let restored = crate::gesture::GestureTrackDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored, document);
        let restored_track = restored.decode().unwrap();
        assert_eq!(restored_track, track);
        assert_eq!(restored_track.checksum_hex(), track.checksum_hex());
        assert_eq!(restored.checksum, track.checksum_hex());

        // The commit is no-replace: a second publication refuses rather than
        // overwriting a receipt another job already owns.
        let second = write_gesture_sidecar_noreplace(&output_string, &document).unwrap_err();
        assert!(
            second.contains("refusing to overwrite"),
            "unexpected error: {second}"
        );
        assert_eq!(std::fs::read(&sidecar_path).unwrap(), bytes);
        assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".gesture.json.tmp-")
        }));

        // Cleanup-coupled: the receipt is removed with the video it describes.
        let progress = ExportProgress::new();
        progress.output_started.store(true, Ordering::Release);
        assert_eq!(remove_started_output(&progress, Some(&output_string)), None);
        assert!(!output.exists());
        assert!(!sidecar_path.exists());
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn a_re_export_retires_its_own_stale_gesture_receipt_while_the_commit_stays_no_replace() {
        let unique = format!(
            "collide-o-scope-gesture-reexport-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(&unique);
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("artifact.mp4");
        let output_string = output.to_string_lossy().into_owned();
        let sidecar_path = gesture_sidecar_path(&output_string);

        let document = crate::gesture::GestureTrackDocument::capture(&recorded_gesture_fixture());
        write_gesture_sidecar_noreplace(&output_string, &document).unwrap();
        let first = std::fs::read(&sidecar_path).unwrap();

        // Without retirement the commit refuses, which is the whole point of the
        // no-replace transaction: nothing silently overwrites a standing receipt.
        assert!(write_gesture_sidecar_noreplace(&output_string, &document).is_err());
        assert_eq!(std::fs::read(&sidecar_path).unwrap(), first);

        // A re-export claims the destination name when it spawns ffmpeg with
        // `-y`, and retiring the previous run's receipt at that moment is what
        // lets the same job publish again.
        remove_partial_path(&sidecar_path, "gesture sidecar").unwrap();
        assert!(!sidecar_path.exists());
        write_gesture_sidecar_noreplace(&output_string, &document).unwrap();
        assert_eq!(std::fs::read(&sidecar_path).unwrap(), first);

        // Retiring a receipt that was never written is not an error, so an
        // ordinary first export is unaffected.
        std::fs::remove_file(&sidecar_path).unwrap();
        remove_partial_path(&sidecar_path, "gesture sidecar").unwrap();

        // The retirement lives at the output claim, not at the commit.
        let source = include_str!("render_export.rs");
        let claim = source
            .split_once(".output_started\n                .store(true, Ordering::Release);")
            .expect("the encoder supervisor claims the output name")
            .1;
        assert!(
            claim[..800].contains(
                "remove_partial_path(&gesture_sidecar_path(&output_path), \"gesture sidecar\")"
            ),
            "a re-export must retire its own stale gesture receipt when it claims the output"
        );

        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn a_gesture_checksum_mismatch_is_an_actionable_export_error_before_any_frame_renders() {
        let track = recorded_gesture_fixture();
        let document = crate::gesture::GestureTrackDocument::capture(&track);
        assert_eq!(
            export_recorded_gesture_track(Some(&document)).unwrap(),
            track
        );

        // A silently edited but still well-formed stream. The digest is the only
        // thing that can catch it, and it must, before any frame renders.
        let mut tampered = document.clone();
        tampered.events[2].x = tampered.events[2].x.wrapping_add(1);
        let error = export_recorded_gesture_track(Some(&tampered)).unwrap_err();
        assert!(
            error.starts_with("recorded gesture track rejected before rendering"),
            "unexpected error: {error}"
        );
        assert!(error.contains("checksum"), "unexpected error: {error}");

        // A declared digest that no longer describes its own events.
        let mut restamped = document.clone();
        restamped.checksum = "0".repeat(64);
        let error = export_recorded_gesture_track(Some(&restamped)).unwrap_err();
        assert!(error.contains("checksum"), "unexpected error: {error}");

        // An ill-formed stream is refused on its own terms rather than repaired.
        // `decode` runs the shared validator before it compares digests, so the
        // orphaned Move is named as a stream fault rather than as a checksum
        // mismatch.
        let mut orphaned = document.clone();
        orphaned.events.remove(0);
        orphaned.event_count -= 1;
        let error = export_recorded_gesture_track(Some(&orphaned)).unwrap_err();
        assert!(
            error.starts_with("recorded gesture track rejected before rendering"),
            "unexpected error: {error}"
        );
        assert!(
            error.to_lowercase().contains("stroke"),
            "unexpected error: {error}"
        );

        // A declared count that disagrees with the stream never reaches the
        // renderer either.
        let mut miscounted = document.clone();
        miscounted.event_count = miscounted.event_count.saturating_add(1);
        assert!(export_recorded_gesture_track(Some(&miscounted)).is_err());
    }

    #[test]
    fn an_absent_gesture_track_replays_nothing_and_publishes_no_sidecar() {
        // The pre-gesture path. Nothing decodes, nothing etches, nothing is
        // written, and no unrecorded live gesture is implied replayable by a
        // file that does not exist.
        let empty = export_recorded_gesture_track(None).unwrap();
        assert!(empty.events().is_empty());
        assert!(!empty.truncated());
        assert_eq!(empty, crate::gesture::GestureTrack::default());

        let params = crate::gesture_canvas::GestureCanvasParams::default();
        let mut canvas = gesture_fixture_canvas(params);
        let before = canvas.field().cells().to_vec();
        let mut replay = empty.replay();
        for frame in 0..90u64 {
            let plan = stage_export_gesture_canvas_frame(
                &mut canvas,
                &mut replay,
                frame,
                30,
                true,
                params,
            )
            .unwrap();
            assert_eq!(plan.applied_samples, 0);
            canvas.commit_staged();
        }
        assert_eq!(canvas.field().cells(), before.as_slice());

        assert!(config(1.0).gesture_track.is_none());
        let source = include_str!("render_export.rs");
        let publication = source
            .split_once("write_motion_sidecar_atomic(&config.output_path, &sidecar)?;")
            .expect("the motion sidecar publication anchors the gesture one")
            .1;
        let publication = &publication[..500];
        assert!(
            publication.contains("if let Some(document) = config.gesture_track.as_ref()"),
            "the gesture sidecar must be published only for a job that carries a recording"
        );
        assert!(
            publication.contains("if !document.events.is_empty()"),
            "an empty recording must publish no sidecar at all"
        );
    }

    /// The offline job must build the device half of its canvas from the same
    /// admitted request the CPU reference was admitted under, bind its
    /// presented donor *before* the executor prepares tap bind groups, and
    /// publish each accepted transaction at the same acceptance decision the
    /// live loop publishes at. Any one of those three in the wrong place is a
    /// silently different offline image rather than an error, so the ordering
    /// is asserted here rather than left to reading.
    #[test]
    fn the_offline_job_builds_binds_and_publishes_its_gesture_canvas_in_the_live_order() {
        let source = include_str!("render_export.rs");
        let start = source
            .find("let mut gesture_canvas = export_gesture_canvas(")
            .expect("the offline canvas is built before the render loop");
        let build = &source[start..start + 1_200];
        assert!(
            build.contains("GestureCanvasResources::prepare("),
            "the offline job must build the device half of its admitted canvas"
        );
        assert!(
            build.contains("gesture_canvas_limits"),
            "the device half must be admitted under the same limits as the CPU reference"
        );

        let bind = source
            .find("executor.bind_gesture_canvas(")
            .expect("the offline executor binds the presented canvas");
        let prepare = source[bind..]
            .find("executor.prepare(&device, &queue, &evaluated_composition")
            .expect("the offline executor prepares after binding");
        assert!(
            prepare > 0,
            "the canvas must be bound before prepare builds the tap bind groups"
        );

        let commit = source
            .find("gesture_canvas.commit_staged();")
            .expect("the offline acceptance decision commits the canvas");
        let publish = source[..commit]
            .rfind("encode_staged_frame(&queue, &mut gesture_encoder, 0, &gesture_canvas)")
            .expect("the offline job publishes the staged transaction");
        assert!(
            source[publish..commit].len() < 500,
            "the device publication must sit at the acceptance decision, immediately before the CPU commit"
        );
        assert!(
            source[..publish].contains("temporal_state.commit_staged();"),
            "publication belongs after the frame reached ffmpeg, not before it"
        );
    }

    /// B16's offline ordering, pinned exactly as the canvas's is: the
    /// job-lifetime tap surface is built before the loop, bound before
    /// prepare, and published only at the acceptance decision — after the
    /// frame reached ffmpeg — byte for byte the live ordering with the
    /// blackout branch simply never taken (export has no blackout).
    #[test]
    fn the_offline_job_publishes_the_program_tap_at_the_acceptance_decision() {
        let source = include_str!("render_export.rs");

        let build = source
            .find("let program_tap_texture = device.create_texture(")
            .expect("the offline tap surface is built before the render loop");
        assert!(
            source[build..build + 900].contains("wgpu::TextureUsages::TEXTURE_BINDING"),
            "the tap is a routable sampled surface, unlike the held-audience copy"
        );
        let frame_loop = source
            .find("for frame_num in 0..total_frames")
            .expect("the offline render loop exists");
        let cold_readiness = concat!("let mut program_tap_", "valid = false;");
        assert!(
            source[build..frame_loop].contains(cold_readiness),
            "the persistent route must begin content-cold just as live does"
        );

        let bind = source
            .find("executor.bind_program_tap(")
            .expect("the offline executor binds the job-lifetime tap");
        let prepare = source[bind..]
            .find("executor.prepare(&device, &queue, &evaluated_composition")
            .expect("the offline executor prepares after binding");
        assert!(
            prepare > 0,
            "the tap must be bound before prepare builds the tap bind groups"
        );
        assert!(
            source[bind..bind + prepare].contains("if program_tap_valid")
                && source[bind..bind + prepare].contains("ProgramTapBinding::default()"),
            "frame zero must retain the route while binding an unready donor"
        );

        let publish = source
            .find("label: Some(\"Export program tap publish\"),")
            .expect("the offline job publishes the tap copy");
        assert!(
            source[..publish].contains("temporal_state.commit_staged();"),
            "publication belongs after the frame reached ffmpeg, not before it"
        );
        let accepted_tail = &source[publish..publish + 1_500];
        let submitted = accepted_tail
            .find("queue.submit(std::iter::once(tap_encoder.finish()));")
            .expect("the accepted tap copy is submitted");
        let ready = accepted_tail
            .find("program_tap_valid = true;")
            .expect("accepted publication marks the tap ready");
        assert!(
            submitted < ready,
            "readiness must publish only after the accepted copy is submitted"
        );
        // The ordinary copy reads final slot 2. Isolated Codec Mosh instead
        // selects the retained pre-mosh wet+dry candidate, so the tap never
        // exposes the wet-only codec input or the moshed visible replacement.
        assert!(
            source[..publish].contains("let program_tap_source =")
                && source[..publish].contains("ExportTemporalBypassMode::AroundCodecMosh")
                && source[publish..publish + 900].contains("texture: program_tap_source,"),
            "the offline tap copy must select the deliberate pre-mosh programme source"
        );
        // Admission is unconditional offline because the surface exists for
        // the whole job; the readiness guard above keeps frame zero unbound.
        assert!(
            source.contains(".with_program_tap(true)"),
            "the offline planner admits the job-lifetime tap unconditionally"
        );
    }

    #[test]
    fn live_and_export_gesture_canvases_read_back_the_same_field_at_24_30_and_60_fps() {
        use crate::gesture_canvas::{GestureCanvasFrameInput, GestureCanvasParams};

        let track = recorded_gesture_fixture();
        let params = GestureCanvasParams {
            radius: 0.28,
            strength: 0.8,
            retention: 0.94,
        };

        for fps in [24u32, 30, 60] {
            let frames = u64::from(fps) * 2;

            // Offline: the production helper, addressed purely by frame index.
            let mut export_canvas = gesture_fixture_canvas(params);
            let mut export_replay = track.replay();
            for frame in 0..frames {
                stage_export_gesture_canvas_frame(
                    &mut export_canvas,
                    &mut export_replay,
                    frame,
                    fps,
                    true,
                    params,
                )
                .unwrap();
                export_canvas.commit_staged();
            }

            // Live: the accepted-frame accumulator the host records on. The two
            // derivations are independent and must produce the same addresses,
            // and therefore the same field, cell for cell.
            let mut live_canvas = gesture_fixture_canvas(params);
            let mut live_replay = track.replay();
            let mut recorder = crate::gesture::GestureEventRecorder::default();
            for _ in 0..frames {
                let reference_tick = recorder.reference_tick();
                let events =
                    live_replay.events_due(u32::try_from(reference_tick).unwrap_or(u32::MAX));
                live_canvas
                    .stage_frame(GestureCanvasFrameInput {
                        reference_tick,
                        program_advances: true,
                        events,
                        evaluated_params: Some(params),
                    })
                    .unwrap();
                live_canvas.commit_staged();
                recorder.record_accepted(1.0 / fps as f32, &[]);
            }

            assert_eq!(
                export_canvas.field().cells(),
                live_canvas.field().cells(),
                "live and export gesture fields must agree at {fps} fps"
            );
            assert_eq!(export_canvas.last_tick(), live_canvas.last_tick());
            assert!(
                export_canvas
                    .field()
                    .cells()
                    .iter()
                    .any(|cell| cell.coverage > 0.0),
                "the {fps} fps fixture must actually etch something"
            );

            // Frame index alone determines the field: a second identical run is
            // byte-identical, so nothing wall-clock reached the replay.
            let mut repeat_canvas = gesture_fixture_canvas(params);
            let mut repeat_replay = track.replay();
            for frame in 0..frames {
                stage_export_gesture_canvas_frame(
                    &mut repeat_canvas,
                    &mut repeat_replay,
                    frame,
                    fps,
                    true,
                    params,
                )
                .unwrap();
                repeat_canvas.commit_staged();
            }
            assert_eq!(repeat_canvas.field().cells(), export_canvas.field().cells());
        }
    }

    #[test]
    fn an_export_frame_that_never_reaches_ffmpeg_leaves_the_gesture_canvas_unchanged() {
        use crate::gesture_canvas::GestureCanvasParams;

        let track = recorded_gesture_fixture();
        let params = GestureCanvasParams {
            radius: 0.3,
            strength: 0.9,
            retention: 0.9,
        };
        let mut canvas = gesture_fixture_canvas(params);
        let mut replay = track.replay();
        for frame in 0..20u64 {
            stage_export_gesture_canvas_frame(&mut canvas, &mut replay, frame, 30, true, params)
                .unwrap();
            canvas.commit_staged();
        }
        let committed = canvas.field().cells().to_vec();
        let committed_generation = canvas.generation();
        let committed_tick = canvas.last_tick();

        // A staged-then-discarded frame is invisible.
        stage_export_gesture_canvas_frame(&mut canvas, &mut replay, 20, 30, true, params).unwrap();
        canvas.discard_staged();
        assert_eq!(canvas.field().cells(), committed.as_slice());
        assert_eq!(canvas.generation(), committed_generation);
        assert_eq!(canvas.last_tick(), committed_tick);

        // A program-frozen frame holds and never bills the frozen ticks as decay.
        for frame in 21..400u64 {
            let plan = stage_export_gesture_canvas_frame(
                &mut canvas,
                &mut replay,
                frame,
                30,
                false,
                params,
            )
            .unwrap();
            assert!(plan.held);
            assert!(plan.is_exact_bypass());
            canvas.commit_staged();
        }
        assert_eq!(canvas.field().cells(), committed.as_slice());
        assert_eq!(canvas.last_tick(), committed_tick);

        // Every early export exit breaks out of the render loop, and the loop is
        // followed by an unconditional discard, so no frame can leak a staged
        // transaction into the finished job.
        let source = include_str!("render_export.rs");
        let staged = source
            .split_once("if let Err(error) = stage_export_gesture_canvas_frame(")
            .expect("the export frame loop stages the gesture canvas")
            .1;
        let decision = staged
            .split_once("temporal_state.commit_staged();")
            .expect("the acceptance decision follows staging")
            .0;
        assert!(
            decision.matches("gesture_canvas.discard_staged();").count() >= 4,
            "every temporal abort path must retain the established gesture discard coverage \
             (including readback, codec-mosh, ffmpeg, and GPU-error failure)"
        );
        assert!(
            !decision.contains("Instant::now()"),
            "wall time must never reach the offline gesture canvas"
        );
        let after_loop = staged
            .split_once("if let Some(gpu_error) = take_export_gpu_errors(&frame_gpu_errors) {")
            .expect("the render loop ends before the encoder is finished")
            .1;
        assert!(
            after_loop[..600].contains("gesture_canvas.discard_staged();"),
            "an abandoned frame must be discarded after the render loop"
        );
    }

    fn map_spatial_uv(uniforms: SpatialGpuUniforms, uv: [f32; 2]) -> [f32; 2] {
        [
            uniforms.inverse_row_0[0] * uv[0]
                + uniforms.inverse_row_0[1] * uv[1]
                + uniforms.inverse_row_0[2],
            uniforms.inverse_row_1[0] * uv[0]
                + uniforms.inverse_row_1[1] * uv[1]
                + uniforms.inverse_row_1[2],
        ]
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_uv_close(actual: [f32; 2], expected: [f32; 2]) {
        assert_close(actual[0], expected[0]);
        assert_close(actual[1], expected[1]);
    }

    #[test]
    fn spatial_4_by_3_card_fit_fill_stretch_and_native_have_exact_landmarks() {
        let dimensions = (640, 480, 1920, 1080);
        let uniforms = |fit| {
            SpatialTransform {
                fit,
                edge: EdgeMode::Transparent,
                ..SpatialTransform::default()
            }
            .gpu_uniforms(dimensions.0, dimensions.1, dimensions.2, dimensions.3)
        };

        let stretch = uniforms(FitMode::Stretch);
        assert_uv_close(map_spatial_uv(stretch, [0.0, 0.0]), [0.0, 0.0]);
        assert_uv_close(map_spatial_uv(stretch, [1.0, 1.0]), [1.0, 1.0]);

        // 4:3 fitted into 16:9 occupies the middle 75% of the width.
        let fit = uniforms(FitMode::Fit);
        assert_uv_close(map_spatial_uv(fit, [0.125, 0.0]), [0.0, 0.0]);
        assert_uv_close(map_spatial_uv(fit, [0.875, 1.0]), [1.0, 1.0]);
        assert!(map_spatial_uv(fit, [0.0, 0.5])[0] < 0.0);
        assert!(map_spatial_uv(fit, [1.0, 0.5])[0] > 1.0);

        // Fill crops equal top and bottom source regions while covering every
        // output pixel: output Y 0..1 maps to source-local Y 1/8..7/8.
        let fill = uniforms(FitMode::Fill);
        assert_uv_close(map_spatial_uv(fill, [0.0, 0.0]), [0.0, 0.125]);
        assert_uv_close(map_spatial_uv(fill, [1.0, 1.0]), [1.0, 0.875]);

        // Native 640x480 pixels occupy exactly that many pixels in 1920x1080.
        let native = uniforms(FitMode::Native);
        assert_uv_close(map_spatial_uv(native, [1.0 / 3.0, 5.0 / 18.0]), [0.0, 0.0]);
        assert_uv_close(map_spatial_uv(native, [2.0 / 3.0, 13.0 / 18.0]), [1.0, 1.0]);

        assert_eq!(
            stretch.modes[3], 0,
            "exact Stretch identity must retain the historical shader bypass"
        );
        for pass in [fit, fill, native] {
            assert_eq!(
                pass.modes[0], 0,
                "the acceptance card uses transparent edges"
            );
            assert_eq!(pass.modes[2], 1, "the transform must remain invertible");
            assert_eq!(
                pass.modes[3], 1,
                "the explicit edge law activates spatial sampling"
            );
            assert_uv_close(map_spatial_uv(pass, [0.5, 0.5]), [0.5, 0.5]);
        }
    }

    #[test]
    fn spatial_anchor_skew_axis_and_affine_order_preserve_landmarks() {
        // Anchor selection alone must be visually inert.
        let anchor_only = SpatialTransform {
            anchor: [0.2, 0.8],
            edge: EdgeMode::Transparent,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(100, 100, 100, 100);
        assert_uv_close(map_spatial_uv(anchor_only, [0.0, 0.0]), [0.0, 0.0]);
        assert_uv_close(map_spatial_uv(anchor_only, [1.0, 1.0]), [1.0, 1.0]);

        // After every authored affine operation, the anchor itself lands at
        // anchor + position for Stretch. This catches accidental translation
        // before pivoting as well as crop-relative anchor regressions.
        let pivoted = SpatialTransform {
            position: [0.1, -0.2],
            scale: [1.7, 0.8],
            anchor: [0.25, 0.75],
            rotation_deg: 31.0,
            skew_deg: 23.0,
            skew_axis_deg: -17.0,
            edge: EdgeMode::Transparent,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(100, 100, 100, 100);
        assert_uv_close(map_spatial_uv(pivoted, [0.35, 0.55]), [0.25, 0.75]);

        // Canonical forward order is scale, skew, then rotation. Starting at
        // local (0.5, 0.75): 2x X scale leaves the Y delta .25; 45-degree X
        // skew makes the delta (.25,.25); clockwise 90-degree rotation makes
        // it (-.25,.25), hence output (0.25,0.75).
        let ordered = SpatialTransform {
            scale: [2.0, 1.0],
            rotation_deg: 90.0,
            skew_deg: 45.0,
            edge: EdgeMode::Transparent,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(100, 100, 100, 100);
        assert_uv_close(map_spatial_uv(ordered, [0.25, 0.75]), [0.5, 0.75]);

        // Rotating the skew axis by 90 degrees turns X shear into the
        // corresponding Y shear; this landmark fixes the axis convention.
        let axis = SpatialTransform {
            skew_deg: 45.0,
            skew_axis_deg: 90.0,
            edge: EdgeMode::Transparent,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(100, 100, 100, 100);
        assert_uv_close(map_spatial_uv(axis, [0.75, 0.25]), [0.75, 0.5]);
    }

    #[test]
    fn spatial_hostile_values_are_finite_bounded_or_explicitly_transparent() {
        let hostile = SpatialTransform {
            position: [f32::NAN, f32::INFINITY],
            scale: [f32::NEG_INFINITY, f32::NAN],
            anchor: [f32::INFINITY, f32::NEG_INFINITY],
            rotation_deg: f32::NAN,
            skew_deg: f32::INFINITY,
            skew_axis_deg: f32::NEG_INFINITY,
            crop: [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX],
            edge: EdgeMode::Transparent,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let clean = hostile.sanitized();
        for value in clean
            .position
            .into_iter()
            .chain(clean.scale)
            .chain(clean.anchor)
            .chain([clean.rotation_deg, clean.skew_deg, clean.skew_axis_deg])
            .chain(clean.crop)
        {
            assert!(value.is_finite());
        }
        assert!(clean.crop[0] + clean.crop[2] < 1.0);
        assert!(clean.crop[1] + clean.crop[3] < 1.0);

        let pass = hostile.gpu_uniforms(u32::MAX, u32::MAX, 1, 1);
        for value in pass
            .inverse_row_0
            .into_iter()
            .chain(pass.inverse_row_1)
            .chain(pass.crop)
        {
            assert!(value.is_finite());
        }
        assert_eq!(pass.modes, [0, 1, 1, 1]);

        for collapsed in [
            SpatialTransform {
                scale: [0.0, 1.0],
                ..SpatialTransform::default()
            },
            SpatialTransform {
                scale: [1.0e-8, 1.0],
                ..SpatialTransform::default()
            },
        ] {
            let pass = collapsed.gpu_uniforms(640, 480, 1920, 1080);
            assert_eq!(pass.modes[2], 0, "collapsed geometry must be invalid");
            assert_eq!(
                pass.modes[3], 1,
                "invalid geometry must select transparent output"
            );
        }
        assert_eq!(
            SpatialTransform::default()
                .gpu_uniforms(0, 480, 1920, 1080)
                .modes[2],
            0
        );
    }

    #[test]
    fn shared_evaluation_and_pass_uniforms_are_exact_at_equal_24_30_and_60_fps_times() {
        use crate::effects::params::TemporalParams;
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };
        use crate::modulation::ModMatrix;

        let transform = SpatialTransform {
            position: [0.125, -0.25],
            scale: [1.5, 0.75],
            anchor: [0.25, 0.8],
            rotation_deg: 33.0,
            skew_deg: -12.0,
            skew_axis_deg: 71.0,
            fit: FitMode::Fit,
            crop: [0.1, 0.05, 0.2, 0.15],
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
        };
        let master_effects = EffectUniforms {
            cellular_amount: 0.8,
            cellular_speed: 1.25,
            shift_amount: 0.7,
            shift_density: 1.0,
            shift_speed: 4.0,
            random_seed: 0x5041_5249,
            ..EffectUniforms::default()
        };
        let layer_effects = EffectUniforms {
            rgb_split: 7.0,
            breathe_position: 0.01,
            ..master_effects
        };
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = TemporalParams::default();
        let modulation = ModMatrix::new().frame(1);
        let mut canonical: Option<Vec<u8>> = None;
        for fps in [24_u32, 30, 60] {
            // The same one-second instant has different frame ordinals at
            // each rate. Both live and export consume this shared immutable
            // evaluator before specializing the common shader payload.
            let time_seconds = fps as f32 / fps as f32;
            let plan = EvaluatedFramePlan::evaluate(
                &modulation,
                FramePlanContext::new(1920, 1080, time_seconds),
                MasterFrameInput {
                    effects: &master_effects,
                    transform: &transform,
                    ntsc: &ntsc,
                    temporal: &temporal,
                },
                [LayerFrameInput {
                    source: SourceTap::new(7, 0, 640, 480),
                    effects: &layer_effects,
                    transform: &transform,
                    opacity: 0.625,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Difference,
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                    pattern: None,
                }],
            );
            let master_pass = *plan.master_pass();
            let layer_pass = plan.layer_passes()[0];
            let mut evidence = Vec::with_capacity(
                std::mem::size_of::<EffectPassUniforms>() * 2 + std::mem::size_of::<f32>(),
            );
            evidence.extend_from_slice(bytemuck::bytes_of(&master_pass));
            evidence.extend_from_slice(bytemuck::bytes_of(&layer_pass));
            evidence.extend_from_slice(&plan.layers()[0].opacity.to_ne_bytes());
            if let Some(expected) = &canonical {
                assert_eq!(&evidence, expected);
            } else {
                canonical = Some(evidence);
            }
        }
    }

    #[test]
    fn active_spatial_sampling_keeps_cellular_and_shift_under_the_edge_law() {
        let shader = include_str!("shaders/effects.wgsl");
        let cellular = shader
            .split("// --- Cellular / Worley domain warp ---")
            .nth(1)
            .unwrap()
            .split("// --- Shift")
            .next()
            .unwrap();
        assert!(cellular.contains("if uniforms.spatial_modes.w == 0u"));
        assert!(cellular.contains("sample_uv = clamp(warped_uv"));
        assert!(cellular.contains("sample_uv = warped_uv"));

        let shift = shader
            .split("// --- Shift (seeded horizontal block displacement) ---")
            .nth(1)
            .unwrap()
            .split("// --- Downsample")
            .next()
            .unwrap();
        assert!(shift.contains("if uniforms.spatial_modes.w == 0u"));
        assert!(shift.contains("sample_uv.x = fract"));
        assert!(shift.contains("sample_uv.x += offset"));
    }

    /// Minimal offscreen use of the production fullscreen/effects shaders.
    /// It deliberately owns no filesystem output: every acceptance image is
    /// read back from a small temporary GPU texture and dropped with the test.
    struct SpatialShaderHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::RenderPipeline,
        texture_layout: wgpu::BindGroupLayout,
        uniform_layout: wgpu::BindGroupLayout,
        composite_pipeline: wgpu::RenderPipeline,
        composite_texture_layout: wgpu::BindGroupLayout,
        composite_uniform_layout: wgpu::BindGroupLayout,
        matte_composite: MatteCompositePipeline,
        sampler: wgpu::Sampler,
        nearest_sampler: wgpu::Sampler,
        _source_texture: wgpu::Texture,
        source_view: wgpu::TextureView,
        output_texture: wgpu::Texture,
        output_view: wgpu::TextureView,
        staging: wgpu::Buffer,
        width: u32,
        height: u32,
        bytes_per_row: u32,
    }

    impl SpatialShaderHarness {
        fn new(width: u32, height: u32) -> Result<Self, String> {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .map_err(|error| format!("no GPU adapter for spatial acceptance test: {error}"))?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Spatial Acceptance Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .map_err(|error| format!("spatial acceptance device request failed: {error}"))?;

            let (
                pipeline,
                texture_layout,
                uniform_layout,
                sampler,
                nearest_sampler,
                source_texture,
                source_view,
                output_texture,
                output_view,
                staging,
                bytes_per_row,
            ) = scoped_export_gpu_operation(&device, "spatial acceptance GPU setup", || {
                let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("Spatial Acceptance Linear Sampler"),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
                let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("Spatial Acceptance Nearest Sampler"),
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                let texture_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Spatial Acceptance Texture Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                        ],
                    });
                let uniform_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Spatial Acceptance Uniform Layout"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });
                let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Spatial Acceptance Vertex"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/fullscreen.wgsl").into(),
                    ),
                });
                let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Spatial Acceptance Effects"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/effects.wgsl").into()),
                });
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Spatial Acceptance Pipeline Layout"),
                        bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
                        immediate_size: 0,
                    });
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Spatial Acceptance Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vertex,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fragment,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                });

                let source_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Spatial Acceptance 4:3 Card"),
                    size: wgpu::Extent3d {
                        width: 4,
                        height: 3,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let source_view =
                    source_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let output_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Spatial Acceptance Output"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let output_view =
                    output_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let bytes_per_row = (width * 4 + 255) & !255;
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Spatial Acceptance Readback"),
                    size: u64::from(bytes_per_row) * u64::from(height),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                (
                    pipeline,
                    texture_layout,
                    uniform_layout,
                    sampler,
                    nearest_sampler,
                    source_texture,
                    source_view,
                    output_texture,
                    output_view,
                    staging,
                    bytes_per_row,
                )
            })?;

            // An opaque, nonuniform card makes both coverage and accidental
            // edge smearing observable without depending on external media.
            let card = spatial_acceptance_card();
            scoped_export_gpu_operation(&device, "spatial acceptance card upload", || {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &source_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &card,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * 4),
                        rows_per_image: Some(3),
                    },
                    wgpu::Extent3d {
                        width: 4,
                        height: 3,
                        depth_or_array_layers: 1,
                    },
                );
            })?;
            let (composite_pipeline, composite_texture_layout, composite_uniform_layout) =
                scoped_export_gpu_operation(&device, "spatial acceptance composite setup", || {
                    build_test_composite_pipeline(&device)
                })?;
            let matte_composite = scoped_export_gpu_operation(
                &device,
                "spatial acceptance matte composite setup",
                || {
                    let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Spatial Acceptance Matte Vertex"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("shaders/fullscreen.wgsl").into(),
                        ),
                    });
                    MatteCompositePipeline::build(&device, &vertex)
                },
            )?;

            Ok(Self {
                device,
                queue,
                pipeline,
                texture_layout,
                uniform_layout,
                composite_pipeline,
                composite_texture_layout,
                composite_uniform_layout,
                matte_composite,
                sampler,
                nearest_sampler,
                _source_texture: source_texture,
                source_view,
                output_texture,
                output_view,
                staging,
                width,
                height,
                bytes_per_row,
            })
        }

        fn render(
            &self,
            transform: SpatialTransform,
            effects: EffectUniforms,
        ) -> Result<Vec<u8>, String> {
            let pass_uniforms = EffectPassUniforms::for_target(
                effects,
                transform,
                (4, 3),
                (self.width, self.height),
            );
            self.render_pass_uniforms(pass_uniforms)
        }

        fn render_pass_uniforms(
            &self,
            pass_uniforms: EffectPassUniforms,
        ) -> Result<Vec<u8>, String> {
            let uniform_buffer = create_uploaded_uniform(
                &self.device,
                &self.queue,
                "Spatial Acceptance Pass Uniforms",
                &pass_uniforms,
            );
            let texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Spatial Acceptance Texture Bind Group"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                    },
                ],
            });
            let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Spatial Acceptance Uniform Bind Group"),
                layout: &self.uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            scoped_export_gpu_operation(&self.device, "spatial acceptance render", || {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Spatial Acceptance Encoder"),
                        });
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Spatial Acceptance Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.output_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &texture_bind_group, &[]);
                    pass.set_bind_group(1, &uniform_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
                self.queue.submit(std::iter::once(encoder.finish()));
            })?;

            readback_pixels(
                &self.device,
                &self.queue,
                &self.output_texture,
                &self.staging,
                self.width,
                self.height,
                self.bytes_per_row,
                &AtomicBool::new(false),
            )
        }

        fn export_layer(
            &self,
            source_index: usize,
            pixels: &[u8],
            base: ExportFrameLayerBase,
        ) -> Result<ExportLayer, String> {
            let (texture, texture_view) = create_export_source_texture(
                &self.device,
                4,
                3,
                "Spatial Full-Stack Export Source",
            )?;
            upload_export_texture_checked(
                &self.device,
                &self.queue,
                &texture,
                pixels,
                4,
                3,
                "Spatial Full-Stack Export Source",
            )?;
            let transport = ExportClipTransport {
                authored: ClipTransportConfig {
                    rate: f64::from(base.speed),
                    sample_fps: Some(f64::from(base.fps)),
                    ..ClipTransportConfig::default()
                }
                .sanitized(),
                state: ClipTransportState::default(),
                source_duration_seconds: 0.0,
                source_frame_count: 1,
            };
            Ok(ExportLayer {
                source_index,
                motion_source: ExportMotionSourceRecord {
                    kind: ExportMotionSourceKind::Still,
                    logical_name: format!("fixture-{source_index}"),
                    persisted_reference: String::new(),
                    fingerprint: None,
                },
                decoder: None,
                codec_motion: None,
                codec_motion_predecessor: None,
                _still_source: None,
                texture,
                texture_view,
                effects: base.effects,
                transform: base.transform,
                opacity: base.opacity,
                mosh_send: base.mosh_send,
                blend_mode: base.blend_mode,
                bypass_master_fx: base.bypass_master_fx,
                bypass_temporal_fx: base.bypass_temporal_fx,
                matte: LayerMatte::default(),
                reroll_on_loop: false,
                consumed_loop_generation: 0,
                transport,
                speed: base.speed,
                visible: base.visible,
                paused: base.paused,
                fps: base.fps,
                width: 4,
                height: 3,
                pattern: None,
            })
        }

        #[cfg(target_os = "windows")]
        fn render_export_full_stack(
            &self,
            layers: &mut [ExportLayer],
            plan: &crate::evaluated_frame::EvaluatedFramePlan,
            delta_seconds: f32,
        ) -> Result<Vec<u8>, String> {
            self.render_export_full_stack_timed(layers, plan, delta_seconds)
                .map(|(pixels, _)| pixels)
        }

        fn render_export_full_stack_timed(
            &self,
            layers: &mut [ExportLayer],
            plan: &crate::evaluated_frame::EvaluatedFramePlan,
            delta_seconds: f32,
        ) -> Result<(Vec<u8>, std::time::Duration), String> {
            let mut routing = None;
            self.render_export_full_stack_timed_with_routing(
                layers,
                plan,
                delta_seconds,
                &mut routing,
            )
        }

        fn render_export_full_stack_with_routing(
            &self,
            layers: &mut [ExportLayer],
            plan: &crate::evaluated_frame::EvaluatedFramePlan,
            delta_seconds: f32,
            routing: &mut Option<ImageRoutingGpuResources>,
        ) -> Result<Vec<u8>, String> {
            self.render_export_full_stack_timed_with_routing(layers, plan, delta_seconds, routing)
                .map(|(pixels, _)| pixels)
        }

        fn render_export_full_stack_timed_with_routing(
            &self,
            layers: &mut [ExportLayer],
            plan: &crate::evaluated_frame::EvaluatedFramePlan,
            delta_seconds: f32,
            routing: &mut Option<ImageRoutingGpuResources>,
        ) -> Result<(Vec<u8>, std::time::Duration), String> {
            assert!(!plan.ntsc().enabled, "golden explicitly disables NTSC");
            let temporal = plan.temporal();
            assert_eq!(temporal.feedback, 0.0);
            assert_eq!(temporal.slitscan, 0.0);
            assert_eq!(temporal.key_mode, 0.0);
            assert_eq!(layers.len(), plan.layers().len());

            let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST;
            let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|index| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Spatial Full-Stack Composite {index}")),
                    size: wgpu::Extent3d {
                        width: self.width,
                        height: self.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage,
                    view_formats: &[],
                })
            });
            let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|index| {
                composite_textures[index].create_view(&wgpu::TextureViewDescriptor::default())
            });
            let (temporal_pipeline, temporal_layout, temporal_uniform_layout) =
                crate::renderer::state::build_temporal_pipeline(&self.device);
            let (history_texture, history_view) = crate::renderer::state::build_history_texture(
                &self.device,
                self.width,
                self.height,
            );
            let (feedback_texture, feedback_view) = crate::renderer::state::build_feedback_texture(
                &self.device,
                self.width,
                self.height,
            );
            let (opaque_pipeline, opaque_layout) =
                crate::renderer::state::build_opaque_output_pipeline(&self.device);
            let opaque_bind_group = crate::renderer::state::build_opaque_output_bind_group(
                &self.device,
                &opaque_layout,
                &composite_views[0],
                &self.sampler,
            );
            let mut temporal_state = crate::renderer::state::TemporalState::default();
            // Exclude one-time pipeline/texture construction. The sample still
            // includes CPU encoding, GPU execution, synchronization, and the
            // complete 8-bit RGBA readback used by the smoke assertion.
            let started = std::time::Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Spatial Full-Stack Export Encoder"),
                });
            render_layers_and_master_export_routed(
                &self.device,
                &self.queue,
                &mut encoder,
                layers,
                plan,
                &composite_textures,
                &composite_views,
                &self.pipeline,
                &self.texture_layout,
                &self.uniform_layout,
                &self.composite_pipeline,
                &self.composite_texture_layout,
                &self.composite_uniform_layout,
                &self.sampler,
                &self.nearest_sampler,
                &self.matte_composite,
                routing,
                self.width,
                self.height,
            )?;
            crate::renderer::state::encode_temporal_with_dt(
                &self.device,
                &self.queue,
                &mut encoder,
                temporal,
                &temporal_pipeline,
                &temporal_layout,
                &temporal_uniform_layout,
                &self.sampler,
                &composite_textures,
                &composite_views,
                &history_texture,
                &history_view,
                &feedback_texture,
                &feedback_view,
                &mut temporal_state,
                delta_seconds,
                true,
                self.width,
                self.height,
            );
            if temporal_bypass_overlay_active(plan) {
                let rendered = render_temporal_bypass_overlay_export(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    layers,
                    plan,
                    &composite_textures,
                    &composite_views,
                    &self.pipeline,
                    &self.texture_layout,
                    &self.uniform_layout,
                    &self.composite_pipeline,
                    &self.composite_texture_layout,
                    &self.composite_uniform_layout,
                    &self.sampler,
                    &self.nearest_sampler,
                    self.width,
                    self.height,
                )?;
                if !rendered {
                    return Err(
                        "spatial full-stack fixture planned no Temporal bypass overlay".to_string(),
                    );
                }
            }
            crate::renderer::state::encode_opaque_output(
                &mut encoder,
                &opaque_pipeline,
                &opaque_bind_group,
                &composite_views[2],
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            temporal_state.commit_staged();

            let bytes_per_row = (self.width * 4 + 255) & !255;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Spatial Full-Stack Export Readback"),
                size: u64::from(bytes_per_row) * u64::from(self.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let pixels = readback_pixels(
                &self.device,
                &self.queue,
                &composite_textures[2],
                &staging,
                self.width,
                self.height,
                bytes_per_row,
                &AtomicBool::new(false),
            )?;
            Ok((pixels, started.elapsed()))
        }
    }

    fn spatial_acceptance_card() -> Vec<u8> {
        let mut card = Vec::with_capacity(4 * 3 * 4);
        for y in 0..3_u8 {
            for x in 0..4_u8 {
                card.extend_from_slice(&[40 + x * 50, 30 + y * 70, 210 - x * 30, 255]);
            }
        }
        card
    }

    fn second_spatial_acceptance_card() -> Vec<u8> {
        spatial_acceptance_card()
            .chunks_exact(4)
            .flat_map(|pixel| {
                [
                    220 - pixel[1] / 2,
                    35 + pixel[2] / 3,
                    20 + pixel[0] / 2,
                    255,
                ]
            })
            .collect()
    }

    fn render_prepared_temporal_originals_pair(
        harness: &SpatialShaderHarness,
        params: &crate::effects::params::TemporalParams,
    ) -> Result<(Vec<u8>, crate::temporal::TemporalStateMetrics), String> {
        use crate::renderer::state::TemporalState;
        use crate::temporal::{TemporalFrameEvents, TemporalFrameInput, TemporalFreezeState};

        assert_eq!([harness.width, harness.height], [4, 3]);
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|index| {
            harness.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Prepared Originals Export Composite {index}")),
                size: wgpu::Extent3d {
                    width: 4,
                    height: 3,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage,
                view_formats: &[],
            })
        });
        let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|index| {
            composite_textures[index].create_view(&wgpu::TextureViewDescriptor::default())
        });
        let (legacy_pipeline, texture_layout, legacy_uniform_layout) =
            crate::renderer::state::build_temporal_pipeline(&harness.device);
        let (history_texture, history_view) =
            crate::renderer::state::build_history_texture(&harness.device, 4, 3);
        let (feedback_texture, feedback_view) =
            crate::renderer::state::build_feedback_texture(&harness.device, 4, 3);
        let prepared = crate::renderer::state::build_prepared_temporal_gpu_resources(
            &harness.device,
            &texture_layout,
            &legacy_uniform_layout,
            &composite_views[0],
            &history_view,
            &harness.sampler,
            &feedback_view,
        );
        let frames = [spatial_acceptance_card(), second_spatial_acceptance_card()];
        let mut state = TemporalState::default();
        for (index, pixels) in frames.iter().enumerate() {
            upload_export_texture_checked(
                &harness.device,
                &harness.queue,
                &composite_textures[0],
                pixels,
                4,
                3,
                "Prepared Originals Export Input",
            )?;
            let mut encoder =
                harness
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Prepared Originals Export Encoder"),
                    });
            crate::renderer::state::encode_temporal_prepared_frame(
                &harness.queue,
                &mut encoder,
                params,
                &legacy_pipeline,
                &prepared,
                &composite_textures,
                &composite_views,
                &history_texture,
                &feedback_texture,
                &mut state,
                TemporalFrameInput::new(
                    1.0 / 30.0,
                    TemporalFreezeState::Running,
                    false,
                    TemporalFrameEvents {
                        manual_events: if index == 1 { 3 } else { 0 },
                        ..TemporalFrameEvents::default()
                    },
                )
                .with_audio_energy(if index == 1 { 0.8 } else { 0.1 }),
                4,
                3,
            );
            harness.queue.submit(std::iter::once(encoder.finish()));
            state.commit_staged();
        }

        let bytes_per_row = 256;
        let staging = harness.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prepared Originals Export Readback"),
            size: u64::from(bytes_per_row) * 3,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pixels = readback_pixels(
            &harness.device,
            &harness.queue,
            &composite_textures[0],
            &staging,
            4,
            3,
            bytes_per_row,
            &AtomicBool::new(false),
        )?;
        Ok((pixels, state.metrics()))
    }

    #[test]
    #[ignore = "requires a GPU adapter; prepared Exact export resources stay offscreen"]
    fn m3_event_acceptance_gpu_prepared_exact_export_originals_and_counted_events() {
        use crate::temporal::{
            CollisionAtlasParams, CollisionScoreParams, CollisionScoreTrigger,
            TemporalInterpolation, TemporalLoomParams, TemporalTopology,
        };

        let harness = SpatialShaderHarness::new(4, 3).unwrap();
        let (legacy, _) = render_prepared_temporal_originals_pair(
            &harness,
            &crate::effects::params::TemporalParams::default(),
        )
        .unwrap();
        assert_eq!(legacy, second_spatial_acceptance_card());

        let originals = crate::effects::params::TemporalParams {
            originals: crate::temporal::TemporalOriginalsParams {
                loom: TemporalLoomParams {
                    amount: 1.0,
                    topology: TemporalTopology::Linear,
                    interpolation: TemporalInterpolation::Linear,
                    depth: 1.0,
                    ..TemporalLoomParams::default()
                },
                atlas: CollisionAtlasParams {
                    amount: 0.35,
                    seed: 0,
                    territories: 7,
                    collision: 0.8,
                },
                score: CollisionScoreParams {
                    enabled: true,
                    seed: 0x51c0_4e11,
                    state_count: 6,
                    trigger: CollisionScoreTrigger::Manual,
                    ..CollisionScoreParams::default()
                },
                ..crate::temporal::TemporalOriginalsParams::default()
            },
            ..crate::effects::params::TemporalParams::default()
        };
        let (pixels, metrics) =
            render_prepared_temporal_originals_pair(&harness, &originals).unwrap();
        assert_ne!(
            pixels, legacy,
            "nonzero Loom/Atlas stayed on the legacy path"
        );
        assert_eq!(metrics.score_event_ordinal, 3);
        assert_eq!(metrics.score_state, 3);
        assert_eq!(metrics.history_valid, 2);
    }

    fn m4_gpu_export_motion_velocity(
        harness: &SpatialShaderHarness,
        velocity_uv_per_second: i32,
        quality: crate::motion::CurvedShutterQuality,
    ) -> Result<(Vec<u8>, u8), String> {
        use crate::evaluated_frame::evaluated_composition::{
            LayerMotionPlanInput, MotionCodecFrameFacts, MotionFieldAttachment,
        };
        use crate::motion::{
            CurvedShutterParams, MotionFieldSource, MotionLatticeQuality, MotionParams,
        };
        use crate::temporal::{TemporalFrameEvents, TemporalFrameInput, TemporalFreezeState};

        let patch: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: motion-card.mp4
"#,
        )
        .unwrap();
        let graph = resolve_export_creative_graph(&patch)?;
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(1);
        let evaluated = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(harness.width, harness.height, 0.0),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(graph.layer_ids[0].get(), 0, 4, 3),
                effects: &effects,
                transform: &transform,
                opacity: 1.0,
                mosh_send: 1.0,
                speed: 1.0,
                fps: 30.0,
                blend_mode: BlendMode::Normal,
                visible: true,
                paused: false,
                bypass_master_fx: false,
                bypass_temporal_fx: false,
                pattern: None,
            }],
        );
        let params = MotionParams {
            field_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::Live,
            shutter: CurvedShutterParams {
                angle_degrees: 360.0,
                quality,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };
        let facts = MotionCodecFrameFacts {
            available: true,
            source_generation: 5,
            frame_ordinal: 9,
        };
        let layer_motion = [LayerMotionPlanInput {
            stable_id: graph.layer_ids[0],
            params,
            codec: facts,
        }];
        let mattes = [LayerMatte::default()];
        let evaluated_composition = evaluated
            .plan_composition(
                CompositionPlanInput::new(
                    &graph.composition,
                    &graph.master_rack,
                    &graph.layer_racks,
                )
                .with_layer_mattes(&mattes, false)
                .with_motion(
                    MotionParams::default(),
                    &layer_motion,
                    crate::motion::MotionDeviceLimits::new(
                        harness.device.limits().max_texture_dimension_2d,
                        harness.device.limits().max_buffer_size,
                    ),
                ),
            )
            .map_err(|error| error.to_string())?;
        let EvaluatedCompositionPlan::Advanced(advanced) = &evaluated_composition else {
            return Err("active export GPU motion fixture delegated LegacyExact".to_owned());
        };
        let codec: CodecMotionProduct = CodecMotionFrame {
            source_dimensions: [4, 3],
            frame_delta_seconds: 1.0 / 30.0,
            source_generation: facts.source_generation,
            frame_ordinal: facts.frame_ordinal,
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            provenance: crate::video::CodecMotionProvenance::FfmpegExportMvs,
            frame_type: crate::video::CodecMotionFrameType::Predictive,
            status: crate::video::CodecMotionStatus::Available,
            past_reference_proof: Some(m4_codec_past_reference_proof(
                facts.source_generation,
                facts.frame_ordinal,
                1,
                4,
            )),
            vectors: vec![crate::motion::CodecMotionVector {
                destination: [2, 1],
                block: [4, 3],
                motion: [-velocity_uv_per_second, 0],
                motion_scale: 1,
                seconds_from_reference: 0.25,
                reference: crate::motion::CodecReferenceDirection::Past,
                visibility: 1.0,
            }],
        }
        .into();
        let (products, diagnostics) =
            export_codec_motion_fields_from(advanced, &graph, [(0, Some(&codec))]);
        if !diagnostics.is_empty() || products.len() != 1 {
            return Err(format!(
                "export codec adapter rejected GPU fixture: {diagnostics:?}"
            ));
        }
        let attachments = products
            .iter()
            .map(ExportMotionFieldProduct::attachment)
            .collect::<Vec<MotionFieldAttachment<'_>>>();
        let mut executor = CompositionGpuExecutor::new(
            &harness.device,
            &harness.queue,
            [harness.width, harness.height],
        )
        .map_err(|error| error.to_string())?;
        let sources = [CompositionSourceDescriptor::new(
            graph.layer_ids[0],
            &harness.source_view,
            [4, 3],
        )];
        executor
            .prepare(
                &harness.device,
                &harness.queue,
                &evaluated_composition,
                &sources,
            )
            .map_err(|error| error.to_string())?;
        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("M4 export motion GPU acceptance encoder"),
            });
        executor
            .encode_with_motion(
                &harness.queue,
                &mut encoder,
                &evaluated_composition,
                CompositionFrameTiming::from_temporal_input(TemporalFrameInput::new(
                    1.0 / 30.0,
                    TemporalFreezeState::Running,
                    false,
                    TemporalFrameEvents::default(),
                )),
                CompositionMotionFrameInput {
                    attachments: &attachments,
                    held_scopes: &[],
                },
            )
            .map_err(|error| error.to_string())?;
        executor
            .encode_present(
                &mut encoder,
                &harness.output_view,
                COMPOSITION_PRESENT_FORMAT,
            )
            .map_err(|error| error.to_string())?;
        harness.queue.submit(std::iter::once(encoder.finish()));
        harness
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| error.to_string())?;
        executor.commit_frame_history();
        let metrics = executor
            .motion_metrics(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
            .ok_or_else(|| "export motion metrics were not published".to_owned())?;
        if metrics.valid_fields != 1 {
            return Err(format!("export motion field did not commit: {metrics:?}"));
        }
        if metrics.field_origin != crate::motion::MotionFieldOrigin::CodecVectors
            || metrics.field_source_scope
                != Some(crate::visual_rack::VisualScopeId::Layer(graph.layer_ids[0]))
            || metrics.field_source_generation != Some(facts.source_generation)
            || metrics.field_frame_ordinal != Some(facts.frame_ordinal)
            || metrics.field_product_content_sha256
                != Some(products[0].codec_identity.content_sha256)
        {
            return Err(format!(
                "export motion field committed mismatched provenance: {metrics:?}"
            ));
        }
        let pixels = readback_pixels(
            &harness.device,
            &harness.queue,
            &harness.output_texture,
            &harness.staging,
            harness.width,
            harness.height,
            harness.bytes_per_row,
            &AtomicBool::new(false),
        )?;
        Ok((pixels, metrics.shutter_samples))
    }

    #[test]
    #[ignore = "requires a GPU adapter; accepted export motion metadata stays offscreen"]
    fn m4_export_runtime_metadata_tracks_prime_freeze_hold_and_codec_donor_truth() {
        use crate::evaluated_frame::evaluated_composition::{
            MotionCodecFrameFacts, MotionFieldAttachment,
        };
        use crate::motion::{
            FaradayParams, MotionCarrier, MotionDonor, MotionFieldOrigin, MotionParams,
        };
        use crate::temporal::{TemporalFrameEvents, TemporalFrameInput, TemporalFreezeState};
        use crate::visual_rack::VisualScopeId;

        fn submit_motion_frame(
            harness: &SpatialShaderHarness,
            executor: &mut CompositionGpuExecutor,
            plan: &EvaluatedCompositionPlan,
            freeze: TemporalFreezeState,
            attachments: &[MotionFieldAttachment<'_>],
            held_scopes: &[VisualScopeId],
        ) -> Result<Vec<ExportRenderedMotionField>, String> {
            let mut encoder =
                harness
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Export Motion Metadata Transaction"),
                    });
            executor
                .encode_with_motion(
                    &harness.queue,
                    &mut encoder,
                    plan,
                    CompositionFrameTiming::from_temporal_input(TemporalFrameInput::new(
                        1.0 / 30.0,
                        freeze,
                        false,
                        TemporalFrameEvents::default(),
                    )),
                    CompositionMotionFrameInput {
                        attachments,
                        held_scopes,
                    },
                )
                .map_err(|error| error.to_string())?;
            harness.queue.submit(std::iter::once(encoder.finish()));
            harness
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| error.to_string())?;
            executor.commit_frame_history();
            Ok(export_rendered_motion_fields(plan, Some(executor)))
        }

        fn layer_scope(
            metadata: &ExportMotionMetadata,
            saved_position: u32,
        ) -> &ExportMotionScopeMetadata {
            metadata
                .scopes
                .iter()
                .find(|scope| {
                    matches!(
                        scope.scope,
                        ExportMotionScopeIdentity::Layer {
                            saved_position: candidate,
                            ..
                        } if candidate == saved_position
                    )
                })
                .unwrap()
        }

        fn assert_held_codec_truth(metadata: &ExportMotionMetadata, expected_digest: [u8; 32]) {
            let codec = layer_scope(metadata, 1);
            assert!(codec.field_attached);
            assert_eq!(
                codec.rendered_source_origin,
                MotionFieldOrigin::CodecVectors
            );
            assert_eq!(codec.source_generation, Some(11));
            assert_eq!(codec.frame_ordinal, Some(17));
            assert_eq!(codec.codec_product_sha256, Some(expected_digest));
            assert_eq!(
                codec.codec_provenance,
                Some(crate::video::CodecMotionProvenance::FfmpegExportMvs)
            );

            let recipient = layer_scope(metadata, 2);
            assert!(recipient.field_attached);
            assert_eq!(
                recipient.rendered_source_origin,
                MotionFieldOrigin::CodecVectors
            );
            assert_eq!(recipient.source_generation, Some(11));
            assert_eq!(recipient.frame_ordinal, Some(17));
            assert_eq!(recipient.codec_product_sha256, Some(expected_digest));
            assert_eq!(recipient.donor_saved_position, Some(1));
            assert_eq!(
                recipient.codec_provenance,
                Some(crate::video::CodecMotionProvenance::FfmpegExportMvs)
            );
        }

        let harness = SpatialShaderHarness::new(64, 32).unwrap();
        let mut graph = resolve_export_creative_graph(&three_layer_legacy_patch()).unwrap();
        graph.composition = crate::composition::RuntimeComposition::try_from_parts(
            Vec::new(),
            graph
                .layer_ids
                .iter()
                .copied()
                .map(|layer_id| crate::composition::RuntimeRootItem::Layer {
                    layer_id,
                    bus: crate::composition::BusAssignment::Program,
                })
                .collect(),
            None,
            0.5,
        )
        .unwrap();
        let (master, mut params) = m4_motion_params(&graph);
        let codec_params = params[0];
        let auto_params = params[1];
        params[0] = auto_params;
        params[1] = codec_params;
        params[2] = MotionParams {
            transplant: FaradayParams {
                amount: 0.75,
                donor: MotionDonor::Selected {
                    layer_id: graph.layer_ids[1],
                    saved_position: SavedLayerPosition::new(1).unwrap(),
                },
                carrier: MotionCarrier::FirstSourceFrame,
                ..FaradayParams::default()
            },
            ..MotionParams::default()
        };
        let evaluated = m4_motion_evaluated_frame(&graph, 30);
        let facts = [
            MotionCodecFrameFacts::default(),
            MotionCodecFrameFacts {
                available: true,
                source_generation: 11,
                frame_ordinal: 17,
            },
            MotionCodecFrameFacts::default(),
        ];
        let inputs =
            export_motion_layer_plan_inputs(&graph, &params, facts.into_iter().enumerate())
                .unwrap();
        let advanced = m4_motion_plan(&evaluated, &graph, master, &inputs);
        let plan = EvaluatedCompositionPlan::Advanced(Box::new(advanced.clone()));

        let codec: CodecMotionProduct = m4_codec_motion_frame(11, 17).into();
        let (products, diagnostics) = export_codec_motion_fields_from(
            &advanced,
            &graph,
            [(0, None), (1, Some(&codec)), (2, None)],
        );
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        assert_eq!(products.len(), 1);
        let accepted_digest = products[0].codec_identity.content_sha256;
        let attachments = products
            .iter()
            .map(ExportMotionFieldProduct::attachment)
            .collect::<Vec<_>>();

        let mut changed_frame = m4_codec_motion_frame(11, 17);
        changed_frame.vectors[0].motion[0] = -8;
        let changed_codec: CodecMotionProduct = changed_frame.into();
        let (changed_products, changed_diagnostics) = export_codec_motion_fields_from(
            &advanced,
            &graph,
            [(0, None), (1, Some(&changed_codec)), (2, None)],
        );
        assert!(changed_diagnostics.is_empty(), "{changed_diagnostics:?}");
        assert_eq!(changed_products.len(), 1);
        assert_ne!(
            changed_products[0].codec_identity.content_sha256,
            accepted_digest
        );
        let changed_attachments = changed_products
            .iter()
            .map(ExportMotionFieldProduct::attachment)
            .collect::<Vec<_>>();

        let sources = graph
            .layer_ids
            .iter()
            .copied()
            .map(|layer_id| {
                CompositionSourceDescriptor::new(layer_id, &harness.source_view, [64, 32])
            })
            .collect::<Vec<_>>();
        let mut executor =
            CompositionGpuExecutor::new(&harness.device, &harness.queue, [64, 32]).unwrap();
        executor
            .prepare(&harness.device, &harness.queue, &plan, &sources)
            .unwrap();

        let frozen_empty = submit_motion_frame(
            &harness,
            &mut executor,
            &plan,
            TemporalFreezeState::ProgramFrozen,
            &attachments,
            &[],
        )
        .unwrap();
        assert!(frozen_empty.is_empty());
        let frozen_empty_metadata = export_motion_metadata_for_frame_from(
            &plan,
            &graph,
            [(0, None), (1, Some(&codec)), (2, None)],
            frozen_empty,
            0,
        );
        let unprimed_codec = layer_scope(&frozen_empty_metadata, 1);
        assert!(unprimed_codec.field_planned);
        assert!(!unprimed_codec.field_attached);
        assert_eq!(unprimed_codec.codec_provenance, None);
        assert_eq!(unprimed_codec.codec_product_sha256, None);

        let first_running = submit_motion_frame(
            &harness,
            &mut executor,
            &plan,
            TemporalFreezeState::Running,
            &attachments,
            &[],
        )
        .unwrap();
        let first_metadata = export_motion_metadata_for_frame_from(
            &plan,
            &graph,
            [(0, None), (1, Some(&codec)), (2, None)],
            first_running,
            1,
        );
        assert_held_codec_truth(&first_metadata, accepted_digest);
        let priming_lattice = layer_scope(&first_metadata, 0);
        assert!(priming_lattice.field_planned);
        assert!(!priming_lattice.field_attached);
        assert_eq!(
            priming_lattice.rendered_source_origin,
            MotionFieldOrigin::None
        );

        let second_running = submit_motion_frame(
            &harness,
            &mut executor,
            &plan,
            TemporalFreezeState::Running,
            &attachments,
            &[],
        )
        .unwrap();
        let second_metadata = export_motion_metadata_for_frame_from(
            &plan,
            &graph,
            [(0, None), (1, Some(&codec)), (2, None)],
            second_running,
            2,
        );
        assert_held_codec_truth(&second_metadata, accepted_digest);
        let published_lattice = layer_scope(&second_metadata, 0);
        assert!(published_lattice.field_attached);
        assert_eq!(
            published_lattice.rendered_source_origin,
            MotionFieldOrigin::LatticeFallback
        );
        assert_eq!(published_lattice.codec_provenance, None);

        for (frame_num, freeze) in [
            (3, TemporalFreezeState::ProgramFrozen),
            (4, TemporalFreezeState::MediaFrozen),
        ] {
            let frozen = submit_motion_frame(
                &harness,
                &mut executor,
                &plan,
                freeze,
                &changed_attachments,
                &[],
            )
            .unwrap();
            let metadata = export_motion_metadata_for_frame_from(
                &plan,
                &graph,
                [(0, None), (1, Some(&codec)), (2, None)],
                frozen,
                frame_num,
            );
            assert_held_codec_truth(&metadata, accepted_digest);
        }

        let held_scope = [VisualScopeId::Layer(graph.layer_ids[1])];
        let held = submit_motion_frame(
            &harness,
            &mut executor,
            &plan,
            TemporalFreezeState::Running,
            &changed_attachments,
            &held_scope,
        )
        .unwrap();
        let held_metadata = export_motion_metadata_for_frame_from(
            &plan,
            &graph,
            [(0, None), (1, Some(&codec)), (2, None)],
            held,
            5,
        );
        assert_held_codec_truth(&held_metadata, accepted_digest);
    }

    #[test]
    fn m4_gpu_export_adapter_static_1_2_4_fields_and_fixed_tiers_when_opted_in() {
        if std::env::var_os("COLLIDE_GPU_GOLDENS").is_none() {
            return;
        }
        use crate::motion::CurvedShutterQuality;

        let harness = SpatialShaderHarness::new(32, 24).unwrap();
        let static_pixels = m4_gpu_export_motion_velocity(&harness, 0, CurvedShutterQuality::Sharp)
            .unwrap()
            .0;
        let one = m4_gpu_export_motion_velocity(&harness, 1, CurvedShutterQuality::Sharp)
            .unwrap()
            .0;
        let two = m4_gpu_export_motion_velocity(&harness, 2, CurvedShutterQuality::Sharp)
            .unwrap()
            .0;
        let four = m4_gpu_export_motion_velocity(&harness, 4, CurvedShutterQuality::Sharp)
            .unwrap()
            .0;
        assert_ne!(static_pixels, one);
        assert_ne!(one, two);
        assert_ne!(two, four);

        for (policy, expected) in [
            (ExportShutterSamples::Samples1, 1),
            (ExportShutterSamples::Samples4, 4),
            (ExportShutterSamples::Samples8, 8),
            (ExportShutterSamples::Samples16, 16),
        ] {
            let quality = policy
                .quality_override()
                .expect("explicit export sample policy has a fixed tier");
            let (first, samples) = m4_gpu_export_motion_velocity(&harness, 2, quality).unwrap();
            let replay = m4_gpu_export_motion_velocity(&harness, 2, quality)
                .unwrap()
                .0;
            assert_eq!(samples, expected);
            assert_eq!(policy.requested_count(), Some(samples));
            // Same-adapter deterministic replay only. The durable report is
            // explicit that cross-GPU pixel identity is not guaranteed.
            assert_eq!(first, replay);
        }
    }

    fn build_test_composite_pipeline(
        device: &wgpu::Device,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Full-Stack Composite Textures"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spatial Full-Stack Composite Uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spatial Full-Stack Composite Vertex"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fullscreen.wgsl").into()),
        });
        let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spatial Full-Stack Composite Fragment"),
            source: wgpu::ShaderSource::Wgsl(composite_shader_source()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spatial Full-Stack Composite Pipeline Layout"),
            bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Spatial Full-Stack Composite Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        (pipeline, texture_layout, uniform_layout)
    }

    fn rgba_at(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * width + x) * 4) as usize;
        pixels[offset..offset + 4].try_into().unwrap()
    }

    fn opaque_pixel_count(pixels: &[u8]) -> usize {
        pixels
            .chunks_exact(4)
            .filter(|pixel| pixel[3] == 255)
            .count()
    }

    fn assert_transparent_pixels_are_clear(pixels: &[u8]) {
        for pixel in pixels.chunks_exact(4).filter(|pixel| pixel[3] == 0) {
            assert_eq!(pixel, [0, 0, 0, 0], "transparent coverage smeared edge RGB");
        }
    }

    // Frozen from the pre-Spatial `src/shaders/effects.wgsl` at commit
    // b208cac: default EffectUniforms sampled this same opaque 4x3 card at its
    // native 4x3 target with no active branches. The historical fragment pass
    // therefore returned these bytes verbatim. Keep the literal independent
    // of `spatial_acceptance_card()` so the new shader cannot author its own
    // expected result at runtime.
    const PRE_SPATIAL_B208CAC_IDENTITY_GOLDEN_RGBA: [u8; 48] = [
        40, 30, 210, 255, 90, 30, 180, 255, 140, 30, 150, 255, 190, 30, 120, 255, 40, 100, 210,
        255, 90, 100, 180, 255, 140, 100, 150, 255, 190, 100, 120, 255, 40, 170, 210, 255, 90, 170,
        180, 255, 140, 170, 150, 255, 190, 170, 120, 255,
    ];

    #[test]
    #[ignore = "requires a GPU adapter; frozen b208cac identity pixels stay offscreen"]
    fn gpu_inactive_spatial_identity_matches_frozen_b208cac_pixels() {
        let harness = SpatialShaderHarness::new(4, 3).unwrap();
        let pixels = harness
            .render(SpatialTransform::default(), EffectUniforms::default())
            .unwrap();
        assert_eq!(pixels, PRE_SPATIAL_B208CAC_IDENTITY_GOLDEN_RGBA);
    }

    // Frozen from the mode-1 LegacyExact active-spatial branch. This fixture
    // is intentionally linear-filtered and translated, so Advanced mode 2's
    // four-load premultiplied filter cannot silently replace legacy sampling.
    const PRE_M6_ACTIVE_SPATIAL_GOLDEN_RGBA: [u8; 48] = [
        40, 30, 210, 255, 70, 30, 196, 255, 118, 30, 166, 255, 168, 30, 136, 255, 40, 100, 210,
        255, 70, 100, 196, 255, 118, 100, 166, 255, 168, 100, 136, 255, 40, 170, 210, 255, 70, 170,
        196, 255, 118, 170, 166, 255, 168, 170, 136, 255,
    ];

    #[test]
    #[ignore = "requires a GPU adapter; freezes LegacyExact active-spatial bytes"]
    fn gpu_legacy_active_spatial_matches_frozen_pre_m6_pixels() {
        let harness = SpatialShaderHarness::new(4, 3).unwrap();
        let pixels = harness
            .render(
                SpatialTransform {
                    position: [0.125, 0.0],
                    edge: EdgeMode::Clamp,
                    ..SpatialTransform::default()
                },
                EffectUniforms::default(),
            )
            .unwrap();
        assert_eq!(pixels, PRE_M6_ACTIVE_SPATIAL_GOLDEN_RGBA);
    }

    fn cpu_reference_rotate([x, y]: [f32; 2], degrees: f32) -> [f32; 2] {
        let (sin, cos) = degrees.to_radians().sin_cos();
        [cos * x - sin * y, sin * x + cos * y]
    }

    /// Independent forward implementation of the documented Stretch affine
    /// law. It intentionally does not call `gpu_uniforms`, invert its rows, or
    /// share any production matrix helper with the shader payload builder.
    fn cpu_reference_spatial_landmark(
        transform: SpatialTransform,
        local_uv: [f32; 2],
        output_aspect: f32,
    ) -> [f32; 2] {
        assert_eq!(transform.fit, FitMode::Stretch);
        assert_eq!(transform.crop, [0.0; 4]);
        let mut physical = [
            (local_uv[0] - transform.anchor[0]) * transform.scale[0] * output_aspect,
            (local_uv[1] - transform.anchor[1]) * transform.scale[1],
        ];
        physical = cpu_reference_rotate(physical, -transform.skew_axis_deg);
        physical[0] += transform.skew_deg.to_radians().tan() * physical[1];
        physical = cpu_reference_rotate(physical, transform.skew_axis_deg);
        physical = cpu_reference_rotate(physical, transform.rotation_deg);
        [
            transform.anchor[0] + transform.position[0] + physical[0] / output_aspect,
            transform.anchor[1] + transform.position[1] + physical[1],
        ]
    }

    fn cpu_reference_rotate_then_scale_landmark(
        transform: SpatialTransform,
        local_uv: [f32; 2],
        output_aspect: f32,
    ) -> [f32; 2] {
        assert_eq!(transform.skew_deg, 0.0);
        let physical = cpu_reference_rotate(
            [
                (local_uv[0] - transform.anchor[0]) * output_aspect,
                local_uv[1] - transform.anchor[1],
            ],
            transform.rotation_deg,
        );
        [
            transform.anchor[0]
                + transform.position[0]
                + physical[0] * transform.scale[0] / output_aspect,
            transform.anchor[1] + transform.position[1] + physical[1] * transform.scale[1],
        ]
    }

    fn exact_color_centroid(pixels: &[u8], width: u32, color: [u8; 4]) -> [f32; 2] {
        let mut count = 0_u32;
        let mut sum = [0.0_f32; 2];
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            if pixel == color {
                let x = index as u32 % width;
                let y = index as u32 / width;
                sum[0] += x as f32 + 0.5;
                sum[1] += y as f32 + 0.5;
                count += 1;
            }
        }
        assert!(count >= 8, "landmark color was not materially rasterized");
        [sum[0] / count as f32, sum[1] / count as f32]
    }

    #[test]
    #[ignore = "requires a GPU adapter; compares offscreen raster landmarks to independent CPU math"]
    fn gpu_affine_landmarks_match_independent_cpu_and_fix_scale_skew_rotation_order() {
        const WIDTH: u32 = 80;
        const HEIGHT: u32 = 60;
        let harness = SpatialShaderHarness::new(WIDTH, HEIGHT).unwrap();
        let card = spatial_acceptance_card();
        let source_x = 2_u32;
        let source_y = 0_u32;
        let source_uv = [(source_x as f32 + 0.5) / 4.0, (source_y as f32 + 0.5) / 3.0];
        let color = rgba_at(&card, 4, source_x, source_y);
        let aspect = WIDTH as f32 / HEIGHT as f32;

        let assert_landmark = |label: &str, transform: SpatialTransform| -> [f32; 2] {
            let expected = cpu_reference_spatial_landmark(transform, source_uv, aspect);
            let pixels = harness
                .render(transform, EffectUniforms::default())
                .unwrap();
            let observed = exact_color_centroid(&pixels, WIDTH, color);
            let expected_pixels = [expected[0] * WIDTH as f32, expected[1] * HEIGHT as f32];
            assert!(
                (observed[0] - expected_pixels[0]).abs() <= 0.5001,
                "{label} X landmark: observed {}, CPU {}",
                observed[0],
                expected_pixels[0]
            );
            assert!(
                (observed[1] - expected_pixels[1]).abs() <= 0.5001,
                "{label} Y landmark: observed {}, CPU {}",
                observed[1],
                expected_pixels[1]
            );
            observed
        };

        // Independent one-axis changes prove X and Y scale are not coupled.
        for (label, scale) in [("x-scale", [1.35, 1.0]), ("y-scale", [1.0, 0.65])] {
            assert_landmark(
                label,
                SpatialTransform {
                    scale,
                    edge: EdgeMode::Transparent,
                    sampling: SamplingMode::Nearest,
                    ..SpatialTransform::default()
                },
            );
        }

        // Ninety-degree pivot landmarks cover the upper-left source anchor
        // (translated/scaled so its rotated footprint remains visible) and
        // the ordinary center anchor independently.
        assert_landmark(
            "top-left-anchor-90",
            SpatialTransform {
                position: [0.4, 0.1],
                scale: [0.5, 0.5],
                anchor: [0.0, 0.0],
                rotation_deg: 90.0,
                edge: EdgeMode::Transparent,
                sampling: SamplingMode::Nearest,
                ..SpatialTransform::default()
            },
        );
        assert_landmark(
            "center-anchor-90",
            SpatialTransform {
                scale: [0.5, 0.5],
                rotation_deg: 90.0,
                edge: EdgeMode::Transparent,
                sampling: SamplingMode::Nearest,
                ..SpatialTransform::default()
            },
        );

        for (label, axis) in [("x-skew-axis", 0.0), ("rotated-skew-axis", 90.0)] {
            assert_landmark(
                label,
                SpatialTransform {
                    scale: [0.7, 0.7],
                    skew_deg: 22.0,
                    skew_axis_deg: axis,
                    edge: EdgeMode::Transparent,
                    sampling: SamplingMode::Nearest,
                    ..SpatialTransform::default()
                },
            );
        }

        let ordered = SpatialTransform {
            scale: [1.3, 0.6],
            rotation_deg: 30.0,
            edge: EdgeMode::Transparent,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let observed = assert_landmark("scale-then-rotate", ordered);
        let wrong = cpu_reference_rotate_then_scale_landmark(ordered, source_uv, aspect);
        let wrong_pixels = [wrong[0] * WIDTH as f32, wrong[1] * HEIGHT as f32];
        let wrong_distance = ((observed[0] - wrong_pixels[0]).powi(2)
            + (observed[1] - wrong_pixels[1]).powi(2))
        .sqrt();
        assert!(
            wrong_distance >= 4.0,
            "GPU landmark was not distinguishable from rotate-then-scale ({wrong_distance}px)"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; renders only to temporary offscreen textures"]
    fn gpu_spatial_card_modes_edges_and_hostile_values_match_the_cpu_contract() {
        let harness = SpatialShaderHarness::new(16, 9).unwrap();
        let transform = |fit, edge| SpatialTransform {
            fit,
            edge,
            ..SpatialTransform::default()
        };

        let fit = harness
            .render(
                transform(FitMode::Fit, EdgeMode::Transparent),
                EffectUniforms::default(),
            )
            .unwrap();
        assert_eq!(opaque_pixel_count(&fit), 12 * 9);
        for y in 0..9 {
            for x in [0, 1, 14, 15] {
                assert_eq!(rgba_at(&fit, 16, x, y), [0, 0, 0, 0]);
            }
        }
        assert_transparent_pixels_are_clear(&fit);

        let clamped_fit = harness
            .render(
                transform(FitMode::Fit, EdgeMode::Clamp),
                EffectUniforms::default(),
            )
            .unwrap();
        assert_eq!(opaque_pixel_count(&clamped_fit), 16 * 9);

        for mode in [FitMode::Fill, FitMode::Stretch] {
            let pixels = harness
                .render(
                    transform(mode, EdgeMode::Transparent),
                    EffectUniforms::default(),
                )
                .unwrap();
            assert_eq!(opaque_pixel_count(&pixels), 16 * 9, "{mode:?}");
        }

        let native = harness
            .render(
                transform(FitMode::Native, EdgeMode::Transparent),
                EffectUniforms::default(),
            )
            .unwrap();
        assert_eq!(opaque_pixel_count(&native), 4 * 3);
        for y in 0..9 {
            for x in 0..16 {
                let expected_opaque = (6..=9).contains(&x) && (3..=5).contains(&y);
                assert_eq!(rgba_at(&native, 16, x, y)[3] == 255, expected_opaque);
            }
        }
        assert_transparent_pixels_are_clear(&native);

        let collapsed = harness
            .render(
                SpatialTransform {
                    scale: [0.0, 1.0],
                    edge: EdgeMode::Transparent,
                    ..SpatialTransform::default()
                },
                EffectUniforms::default(),
            )
            .unwrap();
        assert!(collapsed.iter().all(|byte| *byte == 0));

        let sanitized = harness
            .render(
                SpatialTransform {
                    position: [f32::NAN, f32::INFINITY],
                    scale: [f32::NEG_INFINITY, f32::NAN],
                    anchor: [f32::NAN, f32::NEG_INFINITY],
                    rotation_deg: f32::INFINITY,
                    skew_deg: f32::NAN,
                    skew_axis_deg: f32::NEG_INFINITY,
                    edge: EdgeMode::Transparent,
                    ..SpatialTransform::default()
                },
                EffectUniforms::default(),
            )
            .unwrap();
        assert_eq!(opaque_pixel_count(&sanitized), 16 * 9);

        // Matched half-scale geometry compares the two explicit edge laws.
        // Cellular/Shift are allowed to move UVs outside and Transparent
        // must expose transparent coverage instead of inheriting legacy clamp
        // or fract behavior. Explicit Clamp must remain fully covered.
        let effects = EffectUniforms {
            time: 0.75,
            cellular_amount: 1.0,
            cellular_scale: 4.0,
            cellular_warp: 1.0,
            cellular_speed: 1.5,
            shift_amount: 1.0,
            shift_block_size: 2.0,
            shift_density: 1.0,
            shift_speed: 7.0,
            random_seed: 0x4544_4745,
            ..EffectUniforms::default()
        };
        let transparent_effects = harness
            .render(
                SpatialTransform {
                    scale: [0.5, 0.5],
                    edge: EdgeMode::Transparent,
                    ..SpatialTransform::default()
                },
                effects,
            )
            .unwrap();
        assert!(opaque_pixel_count(&transparent_effects) < 16 * 9);
        assert!(opaque_pixel_count(&transparent_effects) > 0);
        assert_transparent_pixels_are_clear(&transparent_effects);

        let clamped_effects = harness
            .render(
                SpatialTransform {
                    scale: [0.5, 0.5],
                    edge: EdgeMode::Clamp,
                    sampling: SamplingMode::Nearest,
                    ..SpatialTransform::default()
                },
                effects,
            )
            .unwrap();
        assert_eq!(opaque_pixel_count(&clamped_effects), 16 * 9);
    }

    struct TemporarySpatialCard {
        path: std::path::PathBuf,
    }

    impl TemporarySpatialCard {
        fn create() -> Self {
            Self::create_with_pixels("primary", &spatial_acceptance_card())
        }

        fn create_with_pixels(tag: &str, pixels: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "collideoscope-spatial-parity-{tag}-{}-{}.png",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            image::save_buffer(&path, pixels, 4, 3, image::ColorType::Rgba8).unwrap();
            Self { path }
        }
    }

    impl Drop for TemporarySpatialCard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn parity_master_base() -> EffectUniforms {
        EffectUniforms {
            brightness: 0.05,
            cellular_amount: 0.45,
            cellular_scale: 7.0,
            cellular_warp: 0.5,
            cellular_speed: 0.8,
            shift_amount: 0.35,
            shift_block_size: 3.0,
            shift_density: 1.0,
            shift_speed: 4.0,
            random_seed: 0x4c49_5645,
            ..EffectUniforms::default()
        }
    }

    fn parity_master_transform_base() -> SpatialTransform {
        SpatialTransform {
            position: [0.02, -0.03],
            scale: [1.05, 0.95],
            anchor: [0.45, 0.6],
            rotation_deg: 4.0,
            skew_deg: -3.0,
            skew_axis_deg: 12.0,
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        }
    }

    fn parity_layer_base() -> ExportFrameLayerBase {
        ExportFrameLayerBase {
            effects: EffectUniforms {
                hue_shift: 12.0,
                rgb_split: 2.0,
                random_seed: 0x4c41_5945,
                ..EffectUniforms::default()
            },
            transform: SpatialTransform {
                position: [-0.04, 0.06],
                scale: [0.9, 1.1],
                anchor: [0.35, 0.7],
                rotation_deg: -8.0,
                skew_deg: 5.0,
                skew_axis_deg: -20.0,
                fit: FitMode::Fit,
                edge: EdgeMode::Transparent,
                sampling: SamplingMode::Nearest,
                ..SpatialTransform::default()
            },
            opacity: 0.72,
            mosh_send: 0.63,
            speed: 1.0,
            fps: 30.0,
            blend_mode: BlendMode::Difference,
            visible: true,
            paused: false,
            bypass_master_fx: false,
            bypass_temporal_fx: false,
            pattern: None,
        }
    }

    #[cfg(target_os = "windows")]
    fn parity_second_layer_base() -> ExportFrameLayerBase {
        ExportFrameLayerBase {
            effects: EffectUniforms {
                hue_shift: -18.0,
                brightness: 0.08,
                random_seed: 0x4c41_5932,
                ..EffectUniforms::default()
            },
            transform: SpatialTransform {
                position: [0.09, -0.05],
                scale: [1.15, 0.82],
                anchor: [0.62, 0.38],
                rotation_deg: 11.0,
                skew_deg: -6.0,
                skew_axis_deg: 28.0,
                fit: FitMode::Fill,
                edge: EdgeMode::Mirror,
                sampling: SamplingMode::Nearest,
                ..SpatialTransform::default()
            },
            opacity: 0.58,
            mosh_send: 1.0,
            speed: 0.9,
            fps: 24.0,
            blend_mode: BlendMode::Screen,
            visible: true,
            paused: false,
            bypass_master_fx: false,
            bypass_temporal_fx: false,
            pattern: None,
        }
    }

    fn parity_morph() -> crate::morph::Morph {
        use crate::morph::{
            LayerMorphSnapshot, Morph, MorphBlendLaw, MorphGlide, MorphMasterSnapshot,
            MorphNtscSnapshot, MorphSlot, MorphTemporalSnapshot,
        };

        let master_a = EffectUniforms {
            brightness: -0.2,
            ..EffectUniforms::default()
        };
        let master_b = EffectUniforms {
            brightness: 0.35,
            ..EffectUniforms::default()
        };
        let layer_a = EffectUniforms {
            hue_shift: -70.0,
            rgb_split: 3.0,
            ..EffectUniforms::default()
        };
        let layer_b = EffectUniforms {
            hue_shift: 95.0,
            rgb_split: 12.0,
            ..EffectUniforms::default()
        };
        let layer2_a = EffectUniforms {
            hue_shift: 40.0,
            brightness: -0.08,
            ..EffectUniforms::default()
        };
        let layer2_b = EffectUniforms {
            hue_shift: -35.0,
            brightness: 0.16,
            ..EffectUniforms::default()
        };
        let master_transform_a = SpatialTransform {
            position: [-0.12, 0.08],
            scale: [0.8, 1.25],
            anchor: [0.2, 0.75],
            rotation_deg: -35.0,
            skew_deg: -14.0,
            skew_axis_deg: 25.0,
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let master_transform_b = SpatialTransform {
            position: [0.18, -0.1],
            scale: [1.7, 0.65],
            anchor: [0.8, 0.3],
            rotation_deg: 72.0,
            skew_deg: 21.0,
            skew_axis_deg: -48.0,
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let layer_transform_a = SpatialTransform {
            position: [-0.2, 0.12],
            scale: [0.65, 1.4],
            anchor: [0.15, 0.85],
            rotation_deg: -55.0,
            skew_deg: 18.0,
            skew_axis_deg: 33.0,
            fit: FitMode::Fit,
            edge: EdgeMode::Transparent,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let layer_transform_b = SpatialTransform {
            position: [0.22, -0.16],
            scale: [1.8, 0.72],
            anchor: [0.78, 0.28],
            rotation_deg: 88.0,
            skew_deg: -24.0,
            skew_axis_deg: -61.0,
            fit: FitMode::Fit,
            edge: EdgeMode::Transparent,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let layer2_transform_a = SpatialTransform {
            position: [0.14, -0.09],
            scale: [1.3, 0.78],
            anchor: [0.7, 0.25],
            rotation_deg: 28.0,
            skew_deg: -11.0,
            skew_axis_deg: 52.0,
            fit: FitMode::Fill,
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let layer2_transform_b = SpatialTransform {
            position: [-0.1, 0.13],
            scale: [0.76, 1.32],
            anchor: [0.3, 0.72],
            rotation_deg: -42.0,
            skew_deg: 16.0,
            skew_axis_deg: -37.0,
            fit: FitMode::Fill,
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let slot = |master,
                    master_transform,
                    effects,
                    transform,
                    opacity,
                    speed,
                    effects2,
                    transform2,
                    opacity2,
                    speed2| MorphSlot {
            master: MorphMasterSnapshot::capture(master),
            master_transform: Some(master_transform),
            master_motion: None,
            master_rack: None,
            layer_racks: None,
            composition: None,
            gesture: None,
            // Gate fixture: these downstream worlds are deliberately present
            // in both Morph slots but explicitly disabled/no-op.
            ntsc: MorphNtscSnapshot::capture(&crate::ntsc::NtscParams::default()),
            temporal: MorphTemporalSnapshot::capture(
                &crate::effects::params::TemporalParams::default(),
            ),
            layers: vec![
                LayerMorphSnapshot {
                    position: 0,
                    opacity,
                    mosh_send: Some((0.15_f32 + opacity * 0.65).clamp(0.0, 1.0)),
                    speed,
                    fps: Some(30.0),
                    effects: Some(MorphMasterSnapshot::capture(effects)),
                    transform: Some(transform),
                    bypass_temporal_fx: Some(opacity >= 0.5),
                    ..LayerMorphSnapshot::default()
                },
                LayerMorphSnapshot {
                    position: 1,
                    opacity: opacity2,
                    mosh_send: Some((0.85_f32 - opacity2 * 0.55).clamp(0.0, 1.0)),
                    speed: speed2,
                    fps: Some(24.0),
                    effects: Some(MorphMasterSnapshot::capture(effects2)),
                    transform: Some(transform2),
                    bypass_temporal_fx: Some(opacity2 < 0.5),
                    ..LayerMorphSnapshot::default()
                },
            ],
        };
        Morph {
            a: Some(slot(
                &master_a,
                master_transform_a,
                &layer_a,
                layer_transform_a,
                0.35,
                0.8,
                &layer2_a,
                layer2_transform_a,
                0.82,
                0.75,
            )),
            b: Some(slot(
                &master_b,
                master_transform_b,
                &layer_b,
                layer_transform_b,
                0.9,
                1.6,
                &layer2_b,
                layer2_transform_b,
                0.42,
                1.25,
            )),
            t: 0.15,
            blend_law: MorphBlendLaw::EqualPower,
            glide: Some(MorphGlide::new(0.15, 0.85, 0.0, 4.0)),
        }
    }

    fn parity_modulation_frame(
        beat: f64,
        delta_seconds: f32,
        layer_count: usize,
    ) -> crate::modulation::ModulationFrame {
        use crate::modulation::{ModMatrix, ModSource, Routing};

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 0.6;
        matrix.lfos[0].beats = 3.0;
        matrix.lfos[0].set_phase(0.125);
        matrix.routings = vec![
            Routing::new(ModSource::Midi(0), "brightness", 0.12),
            Routing::new(ModSource::Lfo(0), "rotation_deg", 0.08),
            Routing::new(ModSource::Midi(0), "position_x", 0.025),
            Routing::new(ModSource::Lfo(0), "layer1_rotation_deg", 0.06),
            Routing::new(ModSource::Midi(0), "layer1_position_y", 0.04),
            Routing::new(ModSource::Midi(0), "layer1_opacity", -0.15),
            Routing::new(ModSource::Lfo(0), "layer1_mosh_send", -0.2),
            Routing::new(ModSource::Lfo(0), "layer2_scale_x", 0.035),
            Routing::new(ModSource::Midi(0), "layer2_opacity", 0.1),
            Routing::new(ModSource::Midi(0), "layer2_mosh_send", 0.15),
            Routing::new(ModSource::Midi(0), "mosh_wipe", 0.4),
            Routing::new(ModSource::Lfo(0), "morph", 0.1),
        ];
        matrix.update_at_beat(beat, delta_seconds);
        matrix.frame(layer_count)
    }

    fn live_style_parity_plan(
        device: &wgpu::Device,
        card_path: &std::path::Path,
        frame_index: u64,
        fps: u32,
    ) -> crate::evaluated_frame::EvaluatedFramePlan {
        use crate::effects::params::TemporalParams;
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };

        let time = frame_index as f32 / fps as f32;
        let beat = time as f64 * 2.0;
        let modulation = parity_modulation_frame(beat, 1.0 / fps as f32, 1);
        let mut master = parity_master_base();
        let mut master_transform = parity_master_transform_base();
        let mut ntsc = crate::ntsc::NtscParams::default();
        let mut temporal = TemporalParams::default();
        let mut layer = crate::layers::Layer::new(card_path.to_str().unwrap(), device).unwrap();
        let base = parity_layer_base();
        layer.effects = base.effects;
        layer.transform = base.transform;
        layer.opacity = base.opacity;
        layer.mosh_send = base.mosh_send;
        layer.speed = base.speed;
        layer.fps = base.fps;
        layer.blend_mode = base.blend_mode;
        layer.visible = base.visible;
        layer.paused = base.paused;
        layer.bypass_master_fx = base.bypass_master_fx;
        layer.bypass_temporal_fx = base.bypass_temporal_fx;

        // Live materializes Morph into mutable runtime state first, then the
        // immutable evaluator applies the already-sampled modulation frame.
        let mut morph = parity_morph();
        morph.settle_glide_at(beat);
        let position = (morph.position_at_beat(beat) + modulation.morph_offset()).clamp(0.0, 1.0);
        morph.apply(
            position,
            &mut master,
            &mut master_transform,
            &mut ntsc,
            &mut temporal,
            std::slice::from_mut(&mut layer),
        );

        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(16, 9, time),
            MasterFrameInput {
                effects: &master,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(layer.layer_id(), 0, layer.width, layer.height),
                effects: &layer.effects,
                transform: &layer.transform,
                opacity: layer.opacity,
                mosh_send: layer.mosh_send,
                speed: layer.speed,
                fps: layer.fps,
                blend_mode: layer.blend_mode,
                visible: layer.visible,
                paused: layer.paused,
                bypass_master_fx: layer.bypass_master_fx,
                bypass_temporal_fx: layer.bypass_temporal_fx,
                pattern: None,
            }],
        )
    }

    fn export_style_parity_plan(
        frame_index: u64,
        fps: u32,
    ) -> crate::evaluated_frame::EvaluatedFramePlan {
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };

        let (time, delta_seconds) = export_program_transport(frame_index, 1.0 / fps as f32, false);
        let beat = time as f64 * 2.0;
        let modulation = parity_modulation_frame(beat, delta_seconds, 1);
        let mut master = parity_master_base();
        let mut master_transform = parity_master_transform_base();
        let mut layer = parity_layer_base();

        // Export independently samples its detached Morph world instead of
        // borrowing the live materialized payload.
        let morph = parity_morph();
        let sample =
            export_morph_sample(Some(&morph), beat, modulation.morph_offset(), false).unwrap();
        sample.master.apply_to(&mut master);
        if let Some(value) = sample.master_transform {
            master_transform = value.sanitized();
        }
        let ntsc = sample.ntsc.to_params();
        let temporal = sample.temporal.to_params();
        let sampled = sample
            .layers
            .into_iter()
            .find(|value| value.position == 0)
            .unwrap();
        layer.opacity = sampled.opacity;
        if let Some(value) = sampled.mosh_send {
            layer.mosh_send = value;
        }
        layer.speed = sampled.speed;
        if let Some(value) = sampled.fps {
            layer.fps = value;
        }
        if let Some(value) = sampled.effects {
            value.apply_to(&mut layer.effects);
        }
        if let Some(value) = sampled.transform {
            layer.transform = value.sanitized();
        }
        if let Some(value) = sampled.key_threshold {
            layer.effects.key_threshold = value;
        }
        if let Some(value) = sampled.blend_mode {
            layer.blend_mode = value.to_blend_mode();
        }
        if let Some(value) = sampled.visible {
            layer.visible = value;
        }
        if let Some(value) = sampled.paused {
            layer.paused = value;
        }
        if let Some(value) = sampled.bypass_master_fx {
            layer.bypass_master_fx = value;
        }
        if let Some(value) = sampled.bypass_temporal_fx {
            layer.bypass_temporal_fx = value;
        }

        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(16, 9, time),
            MasterFrameInput {
                effects: &master,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(export_selective_layer_id(0), 0, 4, 3),
                effects: &layer.effects,
                transform: &layer.transform,
                opacity: layer.opacity,
                mosh_send: layer.mosh_send,
                speed: layer.speed,
                fps: layer.fps,
                blend_mode: layer.blend_mode,
                visible: layer.visible,
                paused: layer.paused,
                bypass_master_fx: layer.bypass_master_fx,
                bypass_temporal_fx: layer.bypass_temporal_fx,
                pattern: None,
            }],
        )
    }

    #[cfg(target_os = "windows")]
    fn configure_live_layer(layer: &mut crate::layers::Layer, base: ExportFrameLayerBase) {
        layer.effects = base.effects;
        layer.transform = base.transform;
        layer.opacity = base.opacity;
        layer.mosh_send = base.mosh_send;
        layer.speed = base.speed;
        layer.fps = base.fps;
        layer.blend_mode = base.blend_mode;
        layer.visible = base.visible;
        layer.paused = base.paused;
        layer.bypass_master_fx = base.bypass_master_fx;
        layer.bypass_temporal_fx = base.bypass_temporal_fx;
    }

    #[cfg(target_os = "windows")]
    fn apply_export_morph_layer(
        layer: &mut ExportFrameLayerBase,
        sampled: crate::morph::LayerMorphSnapshot,
    ) {
        layer.opacity = sampled.opacity;
        if let Some(value) = sampled.mosh_send {
            layer.mosh_send = value;
        }
        layer.speed = sampled.speed;
        if let Some(value) = sampled.fps {
            layer.fps = value;
        }
        if let Some(value) = sampled.effects {
            value.apply_to(&mut layer.effects);
        }
        if let Some(value) = sampled.transform {
            layer.transform = value.sanitized();
        }
        if let Some(value) = sampled.key_threshold {
            layer.effects.key_threshold = value;
        }
        if let Some(value) = sampled.blend_mode {
            layer.blend_mode = value.to_blend_mode();
        }
        if let Some(value) = sampled.visible {
            layer.visible = value;
        }
        if let Some(value) = sampled.paused {
            layer.paused = value;
        }
        if let Some(value) = sampled.bypass_master_fx {
            layer.bypass_master_fx = value;
        }
        if let Some(value) = sampled.bypass_temporal_fx {
            layer.bypass_temporal_fx = value;
        }
    }

    /// Build the final two-layer live world through the runtime's mutable
    /// Morph materialization path, then freeze it through the shared planner.
    #[cfg(target_os = "windows")]
    fn live_full_stack_world(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cards: [&std::path::Path; 2],
        frame_index: u64,
        fps: u32,
    ) -> (
        crate::evaluated_frame::EvaluatedFramePlan,
        Vec<crate::layers::Layer>,
    ) {
        use crate::effects::params::TemporalParams;
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };

        let time = frame_index as f32 / fps as f32;
        let beat = time as f64 * 2.0;
        let modulation = parity_modulation_frame(beat, 1.0 / fps as f32, 2);
        let mut master = parity_master_base();
        let mut master_transform = parity_master_transform_base();
        let mut ntsc = crate::ntsc::NtscParams::default();
        let mut temporal = TemporalParams::default();
        let mut layers: Vec<crate::layers::Layer> = cards
            .into_iter()
            .zip([parity_layer_base(), parity_second_layer_base()])
            .map(|(path, base)| {
                let mut layer = crate::layers::Layer::new(path.to_str().unwrap(), device).unwrap();
                let frame = layer
                    .take_ready_media_frame()
                    .unwrap()
                    .expect("temporary still must publish its seed frame");
                layer.upload_frame(device, queue, &frame.rgba).unwrap();
                configure_live_layer(&mut layer, base);
                layer
            })
            .collect();

        let mut morph = parity_morph();
        morph.settle_glide_at(beat);
        let position = (morph.position_at_beat(beat) + modulation.morph_offset()).clamp(0.0, 1.0);
        morph.apply(
            position,
            &mut master,
            &mut master_transform,
            &mut ntsc,
            &mut temporal,
            &mut layers,
        );

        let plan = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(16, 9, time),
            MasterFrameInput {
                effects: &master,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            layers
                .iter()
                .enumerate()
                .map(|(slot, layer)| LayerFrameInput {
                    source: SourceTap::new(layer.layer_id(), slot, layer.width, layer.height),
                    effects: &layer.effects,
                    transform: &layer.transform,
                    opacity: layer.opacity,
                    mosh_send: layer.mosh_send,
                    speed: layer.speed,
                    fps: layer.fps,
                    blend_mode: layer.blend_mode,
                    visible: layer.visible,
                    paused: layer.paused,
                    bypass_master_fx: layer.bypass_master_fx,
                    bypass_temporal_fx: layer.bypass_temporal_fx,
                    pattern: None,
                }),
        );
        (plan, layers)
    }

    /// Independently reconstruct the same final world through export's
    /// detached Morph sampler. No live payload or mutable Layer is reused.
    #[cfg(target_os = "windows")]
    fn export_full_stack_world(
        frame_index: u64,
        fps: u32,
    ) -> (
        crate::evaluated_frame::EvaluatedFramePlan,
        Vec<ExportFrameLayerBase>,
    ) {
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };

        let (time, delta_seconds) = export_program_transport(frame_index, 1.0 / fps as f32, false);
        let beat = time as f64 * 2.0;
        let modulation = parity_modulation_frame(beat, delta_seconds, 2);
        let mut master = parity_master_base();
        let mut master_transform = parity_master_transform_base();
        let mut layers = vec![parity_layer_base(), parity_second_layer_base()];
        let morph = parity_morph();
        let sample =
            export_morph_sample(Some(&morph), beat, modulation.morph_offset(), false).unwrap();
        sample.master.apply_to(&mut master);
        if let Some(value) = sample.master_transform {
            master_transform = value.sanitized();
        }
        let ntsc = sample.ntsc.to_params();
        let temporal = sample.temporal.to_params();
        for sampled in sample.layers {
            let position = sampled.position;
            apply_export_morph_layer(&mut layers[position], sampled);
        }

        let plan = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(16, 9, time),
            MasterFrameInput {
                effects: &master,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            layers
                .iter()
                .enumerate()
                .map(|(slot, layer)| LayerFrameInput {
                    source: SourceTap::new(export_selective_layer_id(slot), slot, 4, 3),
                    effects: &layer.effects,
                    transform: &layer.transform,
                    opacity: layer.opacity,
                    mosh_send: layer.mosh_send,
                    speed: layer.speed,
                    fps: layer.fps,
                    blend_mode: layer.blend_mode,
                    visible: layer.visible,
                    paused: layer.paused,
                    bypass_master_fx: layer.bypass_master_fx,
                    bypass_temporal_fx: layer.bypass_temporal_fx,
                    pattern: None,
                }),
        );
        (plan, layers)
    }

    #[cfg(target_os = "windows")]
    fn render_live_full_stack(
        renderer: &mut crate::renderer::state::Renderer,
        layers: &[crate::layers::Layer],
        plan: &crate::evaluated_frame::EvaluatedFramePlan,
        delta_seconds: f32,
    ) -> Result<Vec<u8>, String> {
        assert!(!plan.ntsc().enabled, "golden explicitly disables NTSC");
        assert_eq!(plan.temporal().feedback, 0.0);
        assert_eq!(plan.temporal().slitscan, 0.0);
        assert_eq!(plan.temporal().key_mode, 0.0);
        assert_eq!(layers.len(), plan.layers().len());

        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Spatial Full-Stack Live Encoder"),
            });
        let resources = crate::renderer::state::LiveFrameResources::capture(layers);
        renderer.render_evaluated_frame(&mut encoder, &resources, plan)?;
        renderer.render_temporal_with_dt(&mut encoder, plan.temporal(), delta_seconds, true);
        renderer.render_opaque_output(&mut encoder);
        renderer.queue.submit(std::iter::once(encoder.finish()));
        renderer.commit_temporal_frame();

        let bytes_per_row = (renderer.output_width * 4 + 255) & !255;
        let staging = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Full-Stack Live Readback"),
            size: u64::from(bytes_per_row) * u64::from(renderer.output_height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        readback_pixels(
            &renderer.device,
            &renderer.queue,
            &renderer.composite_textures[2],
            &staging,
            renderer.output_width,
            renderer.output_height,
            bytes_per_row,
            &AtomicBool::new(false),
        )
    }

    #[cfg(target_os = "windows")]
    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(bytes);
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn matte_test_base(visible: bool) -> ExportFrameLayerBase {
        ExportFrameLayerBase {
            effects: EffectUniforms::default(),
            transform: SpatialTransform::default(),
            opacity: 1.0,
            mosh_send: 1.0,
            speed: 1.0,
            fps: 30.0,
            blend_mode: BlendMode::Normal,
            visible,
            paused: false,
            bypass_master_fx: false,
            bypass_temporal_fx: false,
            pattern: None,
        }
    }

    fn simple_matte_plan(
        width: u32,
        height: u32,
        sources: &[SourceTap],
        visible: &[bool],
        mattes: &[LayerMatte],
        history_ready: bool,
    ) -> Result<EvaluatedFramePlan, String> {
        assert_eq!(sources.len(), visible.len());
        assert_eq!(sources.len(), mattes.len());
        let effects = vec![EffectUniforms::default(); sources.len()];
        let transforms = vec![SpatialTransform::default(); sources.len()];
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(sources.len());
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(width, height, 0.0),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            sources
                .iter()
                .enumerate()
                .map(|(index, source)| LayerFrameInput {
                    source: *source,
                    effects: &effects[index],
                    transform: &transforms[index],
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: visible[index],
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                    pattern: None,
                }),
        )
        .with_image_routing(mattes.iter().copied(), history_ready)
    }

    fn selected_luma_matte(layer_id: StableLayerId) -> LayerMatte {
        LayerMatte {
            enabled: true,
            input: crate::image_routing::ImageInput::SelectedLayer {
                layer_id,
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            },
            channel: crate::image_routing::MatteChannel::Luma,
            invert: false,
            amount: 1.0,
            threshold: 0.0,
            softness: 1.0,
        }
    }

    #[test]
    fn export_saved_position_mapping_survives_resource_reorder_and_rejects_bad_routes_visibly() {
        use crate::image_routing::{
            ImageInput, ImageRouteCycle, ImageRouteDiagnostic, LayerImageStage, LayerMatteConfig,
            SavedImageInput,
        };
        use crate::performance::SavedLayerPosition;

        let saved = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::SelectedLayer {
                layer_position: SavedLayerPosition::new(1).unwrap(),
                stage: LayerImageStage::PreLocalEffects,
            },
            ..LayerMatteConfig::default()
        };
        let runtime = export_runtime_matte(saved, 2);
        let donor_id = StableLayerId::new(export_selective_layer_id(1)).unwrap();
        assert_eq!(
            runtime.input,
            ImageInput::SelectedLayer {
                layer_id: donor_id,
                stage: LayerImageStage::PreLocalEffects,
            }
        );

        // Current resource order is deliberately the reverse of patch order.
        // The saved position still resolves to source-index 1, now at slot 0.
        let sources = [
            SourceTap::new(export_selective_layer_id(1), 0, 4, 3),
            SourceTap::new(export_selective_layer_id(0), 1, 4, 3),
        ];
        let reordered = simple_matte_plan(
            16,
            9,
            &sources,
            &[false, true],
            &[LayerMatte::default(), runtime],
            false,
        )
        .unwrap();
        assert_eq!(reordered.image_routing().taps().len(), 1);
        assert_eq!(reordered.image_routing().taps()[0].donor_layer_index, 0);
        assert_eq!(
            reordered.image_routing().mattes()[1].diagnostic,
            ImageRouteDiagnostic::Ready
        );

        let cycle = LayerMatte {
            enabled: true,
            input: ImageInput::CleanProgram,
            ..LayerMatte::default()
        };
        let cyclic = simple_matte_plan(16, 9, &sources[..1], &[true], &[cycle], false).unwrap();
        assert_eq!(
            cyclic.image_routing().mattes()[0].diagnostic,
            ImageRouteDiagnostic::Cycle(ImageRouteCycle::CleanProgramSameFrame)
        );
        assert_eq!(
            cyclic.image_routing().mattes()[0].resolved_input,
            ResolvedImageInput::Transparent
        );

        // The CPU planner rejects the full-frame history allocation before a
        // GPU object exists and leaves the legacy-safe plan untouched.
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::default();
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = crate::effects::params::TemporalParams::default();
        let modulation = crate::modulation::ModMatrix::new().frame(1);
        let mut oversized = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(16_384, 16_384, 0.0),
            MasterFrameInput {
                effects: &effects,
                transform: &transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(1, 0, 4, 3),
                effects: &effects,
                transform: &transform,
                opacity: 1.0,
                mosh_send: 1.0,
                speed: 1.0,
                fps: 30.0,
                blend_mode: BlendMode::Normal,
                visible: true,
                paused: false,
                bypass_master_fx: false,
                bypass_temporal_fx: false,
                pattern: None,
            }],
        );
        assert!(oversized
            .attach_image_routing(
                [LayerMatte {
                    enabled: true,
                    input: ImageInput::OneBelow,
                    ..LayerMatte::default()
                }],
                false,
            )
            .unwrap_err()
            .contains("bounded limit"));
        assert!(!oversized.image_routing().is_active());
        assert!(oversized.image_routing().mattes().is_empty());
    }

    #[test]
    #[ignore = "requires a GPU adapter; routed export textures stay offscreen"]
    fn gpu_export_missing_donor_is_transparent_and_never_reuses_a_stale_tap() {
        use crate::image_routing::{ImageInput, LayerImageStage, MatteChannel};

        let harness = SpatialShaderHarness::new(16, 9).unwrap();
        let primary = spatial_acceptance_card();
        let donor = second_spatial_acceptance_card();
        let mut layers = vec![
            harness
                .export_layer(0, &primary, matte_test_base(true))
                .unwrap(),
            harness
                .export_layer(1, &donor, matte_test_base(false))
                .unwrap(),
        ];
        let sources = [
            SourceTap::new(export_selective_layer_id(0), 0, 4, 3),
            SourceTap::new(export_selective_layer_id(1), 1, 4, 3),
        ];
        let valid = simple_matte_plan(
            16,
            9,
            &sources,
            &[true, false],
            &[
                selected_luma_matte(StableLayerId::new(export_selective_layer_id(1)).unwrap()),
                LayerMatte::default(),
            ],
            false,
        )
        .unwrap();
        let mut routing = None;
        let valid_pixels = harness
            .render_export_full_stack_with_routing(&mut layers, &valid, 1.0 / 30.0, &mut routing)
            .unwrap();
        assert!(valid_pixels
            .chunks_exact(4)
            .any(|pixel| pixel[..3] != [0, 0, 0]));
        assert!(routing.as_ref().is_some_and(|state| state.history_valid));

        // Keep the same resources and history alive while the next authored ID
        // is absent. A stale array slice would visibly resurrect `primary`.
        let missing = simple_matte_plan(
            16,
            9,
            &sources,
            &[true, false],
            &[
                LayerMatte {
                    enabled: true,
                    input: ImageInput::SelectedLayer {
                        layer_id: StableLayerId::new(999).unwrap(),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    channel: MatteChannel::Luma,
                    amount: 1.0,
                    threshold: 0.0,
                    softness: 1.0,
                    ..LayerMatte::default()
                },
                LayerMatte::default(),
            ],
            true,
        )
        .unwrap();
        assert_eq!(
            missing.image_routing().mattes()[0].resolved_input,
            ResolvedImageInput::Transparent
        );
        assert!(missing.image_routing().taps().is_empty());
        let missing_pixels = harness
            .render_export_full_stack_with_routing(&mut layers, &missing, 1.0 / 30.0, &mut routing)
            .unwrap();
        assert!(missing_pixels
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 0, 0, 255]));
    }

    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires a Windows window surface and real GPU; matte parity stays offscreen/temp"]
    fn gpu_selected_layer_matte_live_and_export_match_a_fixed_golden() {
        use std::sync::Arc;
        use winit::platform::windows::EventLoopBuilderExtWindows;

        let mut event_loop_builder = winit::event_loop::EventLoop::<()>::builder();
        event_loop_builder.with_any_thread(true);
        let event_loop = event_loop_builder.build().unwrap();
        #[allow(deprecated)]
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_visible(false)
                        .with_inner_size(winit::dpi::PhysicalSize::new(16, 9)),
                )
                .unwrap(),
        );
        let mut live_renderer = crate::renderer::state::Renderer::new(window, 16, 9).unwrap();
        let export_harness = SpatialShaderHarness::new(16, 9).unwrap();
        let primary_pixels = spatial_acceptance_card();
        let donor_pixels = second_spatial_acceptance_card();
        let primary_card =
            TemporarySpatialCard::create_with_pixels("matte-primary", &primary_pixels);
        let donor_card = TemporarySpatialCard::create_with_pixels("matte-donor", &donor_pixels);
        let mut live_layers: Vec<crate::layers::Layer> = [
            (&primary_card.path, matte_test_base(true)),
            (&donor_card.path, matte_test_base(false)),
        ]
        .into_iter()
        .map(|(path, base)| {
            let mut layer =
                crate::layers::Layer::new(path.to_str().unwrap(), &live_renderer.device).unwrap();
            let frame = layer
                .take_ready_media_frame()
                .unwrap()
                .expect("temporary matte card must publish its seed frame");
            layer
                .upload_frame(&live_renderer.device, &live_renderer.queue, &frame.rgba)
                .unwrap();
            configure_live_layer(&mut layer, base);
            layer
        })
        .collect();
        let live_sources = [
            SourceTap::new(
                live_layers[0].layer_id(),
                0,
                live_layers[0].width,
                live_layers[0].height,
            ),
            SourceTap::new(
                live_layers[1].layer_id(),
                1,
                live_layers[1].width,
                live_layers[1].height,
            ),
        ];
        let live_plan = simple_matte_plan(
            16,
            9,
            &live_sources,
            &[true, false],
            &[
                selected_luma_matte(StableLayerId::new(live_layers[1].layer_id()).unwrap()),
                LayerMatte::default(),
            ],
            false,
        )
        .unwrap();

        let mut export_layers = vec![
            export_harness
                .export_layer(0, &primary_pixels, matte_test_base(true))
                .unwrap(),
            export_harness
                .export_layer(1, &donor_pixels, matte_test_base(false))
                .unwrap(),
        ];
        let export_sources = [
            SourceTap::new(export_selective_layer_id(0), 0, 4, 3),
            SourceTap::new(export_selective_layer_id(1), 1, 4, 3),
        ];
        let export_plan = simple_matte_plan(
            16,
            9,
            &export_sources,
            &[true, false],
            &[
                selected_luma_matte(StableLayerId::new(export_selective_layer_id(1)).unwrap()),
                LayerMatte::default(),
            ],
            false,
        )
        .unwrap();

        // Authored mutations after evaluation must not become an alternate
        // runtime authority for either resource consumer.
        live_layers[0].opacity = 0.0;
        export_layers[0].opacity = 0.0;
        let live_pixels =
            render_live_full_stack(&mut live_renderer, &live_layers, &live_plan, 1.0 / 30.0)
                .unwrap();
        let export_pixels = export_harness
            .render_export_full_stack(&mut export_layers, &export_plan, 1.0 / 30.0)
            .unwrap();
        assert_eq!(live_pixels, export_pixels);
        assert!(live_pixels
            .chunks_exact(4)
            .any(|pixel| pixel[..3] != [0, 0, 0]));

        // Filled after the first opt-in run; this literal makes subsequent
        // live/export agreement insufficient to self-author the expected image.
        const MATTE_GOLDEN_SHA256: &str =
            "869654f91f66fc164963f00a7724dbd01ebd9e34b47786d2d9a773aab4791552";
        assert_eq!(sha256_hex(&live_pixels), MATTE_GOLDEN_SHA256);
    }

    #[test]
    #[ignore = "requires a GPU adapter; renders only to temporary offscreen textures"]
    fn gpu_live_and_export_morph_modulation_parity_is_exact_at_24_30_and_60_fps() {
        let harness = SpatialShaderHarness::new(16, 9).unwrap();
        let card = TemporarySpatialCard::create();
        let mut cross_fps_evidence: Option<Vec<u8>> = None;
        for fps in [24_u32, 30, 60] {
            let frame_index = u64::from(fps);
            let live = live_style_parity_plan(&harness.device, &card.path, frame_index, fps);
            let export = export_style_parity_plan(frame_index, fps);
            assert_eq!(live.context().time_seconds.to_bits(), 1.0_f32.to_bits());
            assert_eq!(live.context(), export.context());

            let live_master = live.master_pass_uniforms();
            let export_master = export.master_pass_uniforms();
            let live_layer = live.layer_pass_uniforms(0).unwrap();
            let export_layer = export.layer_pass_uniforms(0).unwrap();
            assert_eq!(
                bytemuck::bytes_of(&live_master),
                bytemuck::bytes_of(&export_master),
                "master pass diverged at {fps} fps"
            );
            assert_eq!(
                bytemuck::bytes_of(&live_layer),
                bytemuck::bytes_of(&export_layer),
                "layer pass diverged at {fps} fps"
            );
            assert_eq!(
                live.layers()[0].opacity.to_bits(),
                export.layers()[0].opacity.to_bits(),
                "opacity diverged at {fps} fps"
            );
            assert_eq!(
                live.layers()[0].mosh_send.to_bits(),
                export.layers()[0].mosh_send.to_bits(),
                "Mosh Send diverged at {fps} fps"
            );
            assert_ne!(
                live.layers()[0].mosh_send.to_bits(),
                parity_layer_base().mosh_send.to_bits(),
                "fixture failed to exercise Mosh Send Morph + modulation"
            );
            assert_eq!(
                live.temporal().mosh.wipe.to_bits(),
                export.temporal().mosh.wipe.to_bits(),
                "Mosh Motion Wipe modulation diverged at {fps} fps"
            );
            assert!(
                live.temporal().mosh.wipe > 0.0,
                "fixture failed to exercise global Codec-Mosh modulation"
            );
            assert_eq!(
                live.layers()[0].bypass_temporal_fx,
                export.layers()[0].bypass_temporal_fx,
                "Temporal bypass diverged at {fps} fps"
            );
            assert_ne!(
                live.layers()[0].transform,
                parity_layer_base().transform,
                "fixture failed to exercise spatial Morph + modulation"
            );

            // Render each independently evaluated payload. Reusing one byte
            // block for both consumers would make the proof tautological.
            let live_layer_pixels = harness.render_pass_uniforms(live_layer).unwrap();
            let export_layer_pixels = harness.render_pass_uniforms(export_layer).unwrap();
            assert_eq!(
                live_layer_pixels, export_layer_pixels,
                "layer shader pixels diverged at {fps} fps"
            );
            let live_master_pixels = harness.render_pass_uniforms(live_master).unwrap();
            let export_master_pixels = harness.render_pass_uniforms(export_master).unwrap();
            assert_eq!(
                live_master_pixels, export_master_pixels,
                "master shader pixels diverged at {fps} fps"
            );

            let mut evidence = Vec::new();
            evidence.extend_from_slice(bytemuck::bytes_of(&live_master));
            evidence.extend_from_slice(bytemuck::bytes_of(&live_layer));
            evidence.extend_from_slice(&live.layers()[0].opacity.to_ne_bytes());
            evidence.extend_from_slice(&live.layers()[0].mosh_send.to_ne_bytes());
            evidence.extend_from_slice(&live.temporal().mosh.wipe.to_ne_bytes());
            evidence.extend_from_slice(&live_layer_pixels);
            evidence.extend_from_slice(&live_master_pixels);
            if let Some(expected) = &cross_fps_evidence {
                assert_eq!(
                    &evidence, expected,
                    "equal one-second live/export sample changed at {fps} fps"
                );
            } else {
                cross_fps_evidence = Some(evidence);
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires a Windows window surface and real GPU; all images stay offscreen/temp"]
    fn gpu_two_layer_live_and_export_full_stack_matches_fixed_golden_at_24_30_and_60_fps() {
        use std::sync::Arc;
        use winit::platform::windows::EventLoopBuilderExtWindows;

        let mut event_loop_builder = winit::event_loop::EventLoop::<()>::builder();
        event_loop_builder.with_any_thread(true);
        let event_loop = event_loop_builder.build().unwrap();
        #[allow(deprecated)]
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_visible(false)
                        .with_inner_size(winit::dpi::PhysicalSize::new(16, 9)),
                )
                .unwrap(),
        );
        let mut live_renderer = crate::renderer::state::Renderer::new(window, 16, 9).unwrap();
        let export_harness = SpatialShaderHarness::new(16, 9).unwrap();
        let first_card = TemporarySpatialCard::create();
        let first_pixels = spatial_acceptance_card();
        let second_pixels = second_spatial_acceptance_card();
        let second_card = TemporarySpatialCard::create_with_pixels("secondary", &second_pixels);

        // Generated once from the complete production live renderer at the
        // one-second fixture below, then cross-checked byte-for-byte against
        // the independently reconstructed export stack. This is intentionally
        // a stored digest, not a runtime self-reference.
        const FINAL_FRAME_GOLDEN_SHA256: &str =
            "3818c4e65a94102f3086049cb5dfdac2b74318cdaaaca0f940cdc0fe045ddefc";
        let mut cross_fps_pixels: Option<Vec<u8>> = None;
        for fps in [24_u32, 30, 60] {
            let frame_index = u64::from(fps);
            let (live_plan, mut live_layers) = live_full_stack_world(
                &live_renderer.device,
                &live_renderer.queue,
                [&first_card.path, &second_card.path],
                frame_index,
                fps,
            );
            let (export_plan, export_bases) = export_full_stack_world(frame_index, fps);

            assert_eq!(live_plan.context(), export_plan.context());
            assert_eq!(
                live_plan.context().time_seconds.to_bits(),
                1.0_f32.to_bits()
            );
            assert!(!live_plan.ntsc().enabled);
            assert!(!live_plan.temporal().is_active());
            assert!((live_plan.temporal().fb_zoom - 1.0).abs() <= f32::EPSILON);
            assert_eq!(live_plan.temporal().fb_rotate, 0.0);
            assert_eq!(live_plan.layers().len(), 2);
            assert_eq!(export_plan.layers().len(), 2);
            assert_eq!(
                bytemuck::bytes_of(live_plan.master_pass()),
                bytemuck::bytes_of(export_plan.master_pass()),
                "master payload diverged at {fps} fps"
            );
            for index in 0..2 {
                assert_eq!(
                    bytemuck::bytes_of(&live_plan.layer_passes()[index]),
                    bytemuck::bytes_of(&export_plan.layer_passes()[index]),
                    "layer {index} payload diverged at {fps} fps"
                );
                assert_eq!(
                    live_plan.layers()[index].opacity.to_bits(),
                    export_plan.layers()[index].opacity.to_bits(),
                    "layer {index} opacity diverged at {fps} fps"
                );
                assert_eq!(
                    live_plan.layers()[index].blend_mode,
                    export_plan.layers()[index].blend_mode,
                    "layer {index} blend diverged at {fps} fps"
                );
                assert_eq!(
                    live_plan.layers()[index].bypass_temporal_fx,
                    export_plan.layers()[index].bypass_temporal_fx,
                    "layer {index} Temporal bypass diverged at {fps} fps"
                );
            }
            assert_ne!(
                live_plan.layers()[0].transform,
                parity_layer_base().transform,
                "fixture failed to exercise layer-one spatial Morph/modulation"
            );
            assert_ne!(
                live_plan.layers()[1].transform,
                parity_second_layer_base().transform,
                "fixture failed to exercise layer-two spatial Morph/modulation"
            );

            // After evaluation, deliberately corrupt every authored field
            // that a stale live consumer might reread. Source identity and the
            // pinned texture stay intact; the final pixels must remain owned
            // entirely by the immutable plan.
            live_layers[0].effects = EffectUniforms::default();
            live_layers[0].transform = SpatialTransform::default();
            live_layers[0].opacity = 0.0;
            live_layers[0].blend_mode = BlendMode::Multiply;
            live_layers[0].visible = false;
            live_layers[0].bypass_master_fx = true;
            live_layers[0].bypass_temporal_fx = !live_layers[0].bypass_temporal_fx;

            // These are two real consumers: live owns actual `Layer` textures
            // and uses `Renderer::render_evaluated_frame`; export owns detached
            // `ExportLayer`s and uses `render_layers_and_master_export`.
            let live_pixels = render_live_full_stack(
                &mut live_renderer,
                &live_layers,
                &live_plan,
                1.0 / fps as f32,
            )
            .unwrap();
            let mut export_layers = export_bases
                .into_iter()
                .enumerate()
                .map(|(index, base)| {
                    export_harness.export_layer(
                        index,
                        if index == 0 {
                            &first_pixels
                        } else {
                            &second_pixels
                        },
                        base,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            for layer in &mut export_layers {
                layer.effects = EffectUniforms::default();
                layer.transform = SpatialTransform::default();
                layer.opacity = 0.0;
                layer.blend_mode = BlendMode::Multiply;
                layer.visible = false;
                layer.bypass_master_fx = true;
                layer.bypass_temporal_fx = !layer.bypass_temporal_fx;
            }
            let export_pixels = export_harness
                .render_export_full_stack(&mut export_layers, &export_plan, 1.0 / fps as f32)
                .unwrap();
            assert_eq!(
                live_pixels, export_pixels,
                "complete live/export program images diverged at {fps} fps"
            );
            assert!(live_pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
            let digest = sha256_hex(&live_pixels);
            assert_eq!(
                digest, FINAL_FRAME_GOLDEN_SHA256,
                "fixed final-frame golden diverged at {fps} fps"
            );
            if let Some(expected) = &cross_fps_pixels {
                assert_eq!(
                    &live_pixels, expected,
                    "the equal one-second final frame changed at {fps} fps"
                );
            } else {
                cross_fps_pixels = Some(live_pixels);
            }
        }
    }

    #[test]
    #[ignore = "real-GPU 1080p eight-layer smoke/performance gate; intentionally opt-in"]
    fn gpu_1080p60_eight_transformed_layers_complete_within_debug_smoke_ceiling() {
        use crate::effects::params::TemporalParams;
        use crate::evaluated_frame::{
            EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
        };
        use crate::modulation::ModMatrix;

        const WIDTH: u32 = 1920;
        const HEIGHT: u32 = 1080;
        const FPS: u32 = 60;
        // This is deliberately ~15x the 16.67 ms realtime target for
        // unoptimized test binaries and includes queue synchronization plus an
        // 8 MiB RGBA readback. It remains tight enough to catch a catastrophic
        // per-layer/per-pixel regression while the realtime target is reported.
        const DEBUG_SMOKE_CEILING: std::time::Duration = std::time::Duration::from_millis(250);

        let harness = SpatialShaderHarness::new(WIDTH, HEIGHT).unwrap();
        let blends = [
            BlendMode::Normal,
            BlendMode::Screen,
            BlendMode::Multiply,
            BlendMode::Difference,
        ];
        let bases: Vec<ExportFrameLayerBase> = (0..8)
            .map(|index| {
                let offset = index as f32 - 3.5;
                ExportFrameLayerBase {
                    effects: EffectUniforms {
                        hue_shift: offset * 7.0,
                        brightness: offset * 0.006,
                        random_seed: 0x5045_5246_u32.wrapping_add(index as u32),
                        ..EffectUniforms::default()
                    },
                    transform: SpatialTransform {
                        position: [offset * 0.015, -offset * 0.01],
                        scale: [0.86 + index as f32 * 0.025, 1.08 - index as f32 * 0.018],
                        anchor: [0.2 + index as f32 * 0.075, 0.75 - index as f32 * 0.06],
                        rotation_deg: offset * 4.0,
                        skew_deg: offset * 1.5,
                        skew_axis_deg: -30.0 + index as f32 * 9.0,
                        fit: if index % 2 == 0 {
                            FitMode::Fit
                        } else {
                            FitMode::Fill
                        },
                        edge: if index % 3 == 0 {
                            EdgeMode::Transparent
                        } else {
                            EdgeMode::Mirror
                        },
                        sampling: SamplingMode::Nearest,
                        ..SpatialTransform::default()
                    },
                    opacity: 0.38 + index as f32 * 0.055,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: FPS as f32,
                    blend_mode: blends[index % blends.len()],
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                    pattern: None,
                }
            })
            .collect();
        let master = EffectUniforms {
            brightness: 0.02,
            ..EffectUniforms::default()
        };
        let master_transform = SpatialTransform {
            rotation_deg: 1.25,
            skew_deg: 0.75,
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
            ..SpatialTransform::default()
        };
        let ntsc = crate::ntsc::NtscParams::default();
        let temporal = TemporalParams::default();
        let modulation = ModMatrix::new().frame(8);
        let plan = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(WIDTH, HEIGHT, 1.0),
            MasterFrameInput {
                effects: &master,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            bases
                .iter()
                .enumerate()
                .map(|(slot, layer)| LayerFrameInput {
                    source: SourceTap::new(export_selective_layer_id(slot), slot, 4, 3),
                    effects: &layer.effects,
                    transform: &layer.transform,
                    opacity: layer.opacity,
                    mosh_send: layer.mosh_send,
                    speed: layer.speed,
                    fps: layer.fps,
                    blend_mode: layer.blend_mode,
                    visible: layer.visible,
                    paused: layer.paused,
                    bypass_master_fx: layer.bypass_master_fx,
                    bypass_temporal_fx: layer.bypass_temporal_fx,
                    pattern: None,
                }),
        );
        assert_eq!(plan.layer_passes().len(), 8);
        assert!(plan
            .layers()
            .iter()
            .all(|layer| layer.transform != SpatialTransform::default()));

        let primary = spatial_acceptance_card();
        let secondary = second_spatial_acceptance_card();
        let mut layers = bases
            .into_iter()
            .enumerate()
            .map(|(index, base)| {
                harness.export_layer(
                    index,
                    if index % 2 == 0 { &primary } else { &secondary },
                    base,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let (pixels, elapsed) = harness
            .render_export_full_stack_timed(&mut layers, &plan, 1.0 / FPS as f32)
            .unwrap();
        eprintln!(
            "1080p60 / 8 transformed layers: {:.2} ms measured (16.67 ms realtime target; {:.0} ms debug smoke ceiling)",
            elapsed.as_secs_f64() * 1_000.0,
            DEBUG_SMOKE_CEILING.as_secs_f64() * 1_000.0,
        );
        assert_eq!(pixels.len(), WIDTH as usize * HEIGHT as usize * 4);
        assert!(pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[..3] != [0, 0, 0]));
        assert!(
            elapsed <= DEBUG_SMOKE_CEILING,
            "1080p60 eight-layer frame took {:.2} ms, above the documented {:.0} ms debug smoke ceiling",
            elapsed.as_secs_f64() * 1_000.0,
            DEBUG_SMOKE_CEILING.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    fn export_layer_accumulation_starts_transparent() {
        let clear = transparent_accumulation_clear();
        assert_eq!([clear.r, clear.g, clear.b, clear.a], [0.0; 4]);
    }

    #[test]
    fn export_loop_reroll_consumes_every_generation_once() {
        let mut effects = EffectUniforms {
            random_seed: 9,
            ..Default::default()
        };
        let expected = crate::randomization::advance_seed(9, 3);
        let mut consumed = 0;
        assert_eq!(
            apply_export_loop_generation(&mut effects, true, &mut consumed, 3),
            3
        );
        assert_eq!(effects.random_seed, expected);
        assert_eq!(consumed, 3);
        assert_eq!(
            apply_export_loop_generation(&mut effects, true, &mut consumed, 3),
            0
        );
        assert_eq!(effects.random_seed, expected);
        assert_eq!(
            apply_export_loop_generation(&mut effects, true, &mut consumed, 2),
            0,
            "a stale generation cannot reroll or rewind the consumer"
        );
        assert_eq!(consumed, 3);
    }

    #[test]
    fn offline_boundary_events_drive_reroll_without_decoder_eof_or_double_count() {
        let config = ClipTransportConfig {
            rate: 10.0,
            ..ClipTransportConfig::default()
        };
        let mut transport = ExportClipTransport::new(
            config,
            crate::transport::NormalizedTime::clamped(0.9),
            1.0,
            30,
        );
        let _ = transport.seed_selection();
        let selected = transport.select(
            config,
            ProgramTransportTick {
                delta_seconds: 0.35,
                source_duration_seconds: 99.0,
                source_frame_count: 99,
                ..ProgramTransportTick::default()
            },
        );
        assert_eq!(selected.boundary_events, 4);

        let mut effects = EffectUniforms {
            random_seed: 0x1020_3040,
            ..EffectUniforms::default()
        };
        let mut consumed = 0;
        let timeline_generation = consumed + u64::from(selected.boundary_events);
        assert_eq!(
            apply_export_loop_generation(&mut effects, true, &mut consumed, timeline_generation,),
            4
        );
        assert_eq!(
            effects.random_seed,
            crate::randomization::advance_seed(0x1020_3040, 4)
        );
        assert_eq!(
            apply_export_loop_generation(&mut effects, true, &mut consumed, timeline_generation,),
            0,
            "the same absolute selection cannot be consumed twice"
        );
    }

    #[test]
    fn canonical_rate_eight_survives_the_legacy_modulation_proxy() {
        let modulation = crate::modulation::ModMatrix::new().frame(1);
        let mut base = parity_layer_base();
        base.speed = 8.0;
        base.fps = 480.0;
        let config = modulated_export_transport_config(
            &modulation,
            0,
            ClipTransportConfig {
                rate: 8.0,
                sample_fps: Some(480.0),
                ..ClipTransportConfig::default()
            },
            &base,
            ExportMorphOverrides::default(),
        );
        assert_eq!(config.rate, 8.0);
        assert_eq!(config.sample_fps, Some(480.0));
    }

    fn mix_transport_signature(signature: &mut u64, value: u64) {
        // Stable FNV-1a over explicit scalar evidence; unlike DefaultHasher,
        // this is a durable cross-toolchain golden contract.
        for byte in value.to_le_bytes() {
            *signature ^= u64::from(byte);
            *signature = signature.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn deterministic_offline_transport_signature(fps: u32) -> u64 {
        use crate::transport::{
            BeatLoop, ClipBeatGrid, CueId, CuePoint, CuePoints, EndBehavior, NormalizedTime,
            PlaybackDirection,
        };

        let mut cues = CuePoints::default();
        let cue = CueId::new(9).unwrap();
        assert!(cues.insert(CuePoint {
            id: cue,
            at: NormalizedTime::clamped(0.65),
        }));
        let config = ClipTransportConfig {
            direction: PlaybackDirection::Reverse,
            end_behavior: EndBehavior::PingPong,
            in_point: NormalizedTime::clamped(0.1),
            out_point: NormalizedTime::clamped(0.9),
            rate: 1.375,
            sample_fps: Some(23.976),
            beat_grid: Some(ClipBeatGrid {
                bpm: 128.0,
                length_beats: Some(7.5),
                sync_to_program: true,
                beats_per_bar: 4,
            }),
            beat_loop: Some(BeatLoop {
                start_beat: 0.5,
                length_beats: 4.0,
            }),
            cues,
            ..ClipTransportConfig::default()
        };
        let saved = NormalizedTime::clamped(0.73);
        let mut offline = ExportClipTransport::new(config, saved, 3.75, 90);
        let offline_seed = offline.seed_selection();

        // The live side starts from the same persisted state and invokes the
        // same pure contract. Keeping it explicit catches drift in the export
        // adapter's tick/source-fact injection rather than comparing a helper
        // to itself.
        let mut live_state = ClipTransportState::at(saved, config.direction);
        let (next_live, live_seed) = TransportTimeline::select(
            &config,
            live_state,
            ProgramTransportTick {
                program_running: false,
                media_running: false,
                source_duration_seconds: 3.75,
                source_frame_count: 90,
                ..ProgramTransportTick::default()
            },
        );
        live_state = next_live;
        assert_eq!(offline_seed, live_seed);
        assert!((offline_seed.normalized_time.get() - 0.6).abs() < 1e-12);

        let mut signature = 0xcbf2_9ce4_8422_2325;
        for frame in 0..fps * 6 {
            let seconds = f64::from(frame) / f64::from(fps);
            let tick = ProgramTransportTick {
                delta_seconds: if frame == 0 {
                    0.0
                } else {
                    1.0 / f64::from(fps)
                },
                program_beat: seconds * 128.0 / 60.0,
                program_running: !(fps * 5..fps * 5 + fps / 4).contains(&frame),
                media_running: !(fps * 3..fps * 3 + fps / 2).contains(&frame),
                source_duration_seconds: 3.75,
                source_frame_count: 90,
                cue_id: (frame == fps * 2).then_some(cue),
                seek_to: (frame == fps * 4).then_some(NormalizedTime::clamped(0.2)),
                ..ProgramTransportTick::default()
            };
            let offline_selected = offline.select(config, tick);
            let (next_live, live_selected) = TransportTimeline::select(&config, live_state, tick);
            live_state = next_live;
            assert_eq!(
                offline_selected, live_selected,
                "parity frame {frame} at {fps} fps"
            );

            mix_transport_signature(
                &mut signature,
                offline_selected.normalized_time.get().to_bits(),
            );
            mix_transport_signature(
                &mut signature,
                offline_selected.logical_time.get().to_bits(),
            );
            mix_transport_signature(&mut signature, offline_selected.source_seconds.to_bits());
            mix_transport_signature(
                &mut signature,
                offline_selected.frame_index.unwrap_or(u64::MAX),
            );
            mix_transport_signature(&mut signature, offline_selected.generation);
            mix_transport_signature(&mut signature, u64::from(offline_selected.boundary_events));
            let flags = u64::from(offline_selected.sample_due)
                | (u64::from(offline_selected.held) << 1)
                | (u64::from(offline_selected.transparent) << 2)
                | (u64::from(offline_selected.discontinuity) << 3)
                | (u64::from(offline_selected.completed) << 4);
            mix_transport_signature(&mut signature, flags);
        }
        signature
    }

    #[test]
    fn offline_transport_has_deterministic_24_30_60_fps_goldens_and_live_parity() {
        let signatures = [24, 30, 60].map(deterministic_offline_transport_signature);
        assert_eq!(
            signatures,
            [
                0x55ca_058c_bbe6_db3a,
                0x2fe1_deb3_4664_77b1,
                0xfd17_ec73_0125_ffb2,
            ]
        );
    }

    #[test]
    fn offline_freeze_gates_and_one_shot_visibility_match_the_pure_law() {
        use crate::transport::{EndBehavior, NormalizedTime};

        let looped = ClipTransportConfig::default();
        let mut transport =
            ExportClipTransport::new(looped, NormalizedTime::clamped(0.4), 10.0, 100);
        let saved_playhead = transport.seed_selection();
        assert_eq!(saved_playhead.normalized_time, NormalizedTime::clamped(0.4));
        assert_eq!(saved_playhead.source_seconds, 4.0);
        let program_frozen = transport.select(
            looped,
            ProgramTransportTick {
                delta_seconds: 1.0,
                program_running: false,
                ..ProgramTransportTick::default()
            },
        );
        assert_eq!(program_frozen.logical_time, NormalizedTime::clamped(0.4));
        let media_frozen = transport.select(
            looped,
            ProgramTransportTick {
                delta_seconds: 1.0,
                media_running: false,
                ..ProgramTransportTick::default()
            },
        );
        assert_eq!(media_frozen.logical_time, NormalizedTime::clamped(0.4));

        let one_shot = ClipTransportConfig {
            end_behavior: EndBehavior::OneShot,
            ..ClipTransportConfig::default()
        };
        let mut transport =
            ExportClipTransport::new(one_shot, NormalizedTime::clamped(0.95), 1.0, 30);
        let _ = transport.seed_selection();
        let terminal = transport.select(
            one_shot,
            ProgramTransportTick {
                delta_seconds: 0.1,
                ..ProgramTransportTick::default()
            },
        );
        assert!(!terminal.transparent, "the terminal frame presents once");
        let transparent = transport.select(one_shot, ProgramTransportTick::default());
        assert!(transparent.transparent);
        let authored_visible = true;
        let evaluated_visible = authored_visible && !transparent.transparent;
        assert!(!evaluated_visible);
    }

    #[test]
    fn selective_export_uses_shared_order_opacity_and_reference_clock() {
        let params = crate::ntsc::NtscParams {
            enabled: true,
            ..Default::default()
        };
        let plan = plan_selective_ntsc(
            SelectiveNtscGeneration {
                visual_epoch: 1,
                topology_generation: 1,
                width: 2,
                height: 2,
                sample_sequence: 17,
            },
            NtscFrameMetadata {
                params,
                reference_frame: reference_frame_for_output(17, 60),
            },
            [
                SelectiveNtscLayerDescriptor {
                    layer_id: export_selective_layer_id(0),
                    visible: true,
                    bypass_master_fx: true,
                    opacity: 0.4,
                    blend_mode: BlendMode::Difference.as_u32(),
                    transform_fingerprint: 10,
                },
                SelectiveNtscLayerDescriptor {
                    layer_id: export_selective_layer_id(1),
                    visible: false,
                    bypass_master_fx: false,
                    opacity: 1.0,
                    blend_mode: BlendMode::Screen.as_u32(),
                    transform_fingerprint: 20,
                },
                SelectiveNtscLayerDescriptor {
                    layer_id: export_selective_layer_id(2),
                    visible: true,
                    bypass_master_fx: false,
                    opacity: 0.8,
                    blend_mode: BlendMode::Multiply.as_u32(),
                    transform_fingerprint: 30,
                },
            ],
        )
        .unwrap();

        assert_eq!(plan.generation.sample_sequence, 17);
        assert_eq!(plan.metadata.reference_frame, 8);
        assert_eq!(
            plan.layers
                .iter()
                .map(|layer| layer.layer_id)
                .collect::<Vec<_>>(),
            [3, 1]
        );
        assert_eq!(plan.layers[0].blend_mode, BlendMode::Multiply.as_u32());
        assert_eq!(plan.layers[1].blend_mode, BlendMode::Difference.as_u32());
        assert_eq!(plan.layers[0].opacity, 0.8);
        assert_eq!(plan.layers[1].opacity, 0.4);
        assert!(!plan.layers[0].bypass_master_fx);
        assert!(plan.layers[1].bypass_master_fx);

        let signature = export_selective_topology_signature(&plan);
        let mut continuous_change = plan.clone();
        continuous_change.layers[0].opacity = 0.2;
        continuous_change.layers[0].transform_fingerprint ^= 1;
        assert_eq!(
            export_selective_topology_signature(&continuous_change),
            signature,
            "ordinary modulation must not erase Temporal history"
        );
        let mut bypass_change = plan.clone();
        bypass_change.layers[0].bypass_master_fx = true;
        assert_ne!(
            export_selective_topology_signature(&bypass_change),
            signature
        );
    }

    #[test]
    fn selective_export_fingerprint_tracks_layer_and_inherited_master_geometry() {
        let effects = EffectUniforms::default();
        let identity = SpatialTransform::default();
        let mut moved = identity;
        moved.position = [0.125, -0.25];

        let base = export_selective_transform_fingerprint(&effects, &identity, None, None);
        assert_eq!(
            base,
            export_selective_transform_fingerprint(&effects, &identity, None, None)
        );
        assert_ne!(
            base,
            export_selective_transform_fingerprint(&effects, &moved, None, None)
        );

        let inherited = export_selective_transform_fingerprint(
            &effects,
            &identity,
            Some(&effects),
            Some(&identity),
        );
        assert_ne!(
            inherited,
            export_selective_transform_fingerprint(
                &effects,
                &identity,
                Some(&effects),
                Some(&moved),
            )
        );
    }

    fn value_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.windows(2)
            .find(|window| window[0] == flag)
            .map(|window| window[1].as_str())
    }

    #[test]
    fn mux_audio_is_trimmed_or_silence_padded_to_requested_duration() {
        let config = config(1.25);
        let args = build_ffmpeg_args(&config, Some("source with audio.mp4"));

        assert_eq!(value_after(&args, "-t"), Some("1.250000"));
        assert_eq!(value_after(&args, "-map"), Some("0:v:0"));
        assert!(args.windows(2).any(|pair| pair == ["-map", "1:a:0"]));
        assert_eq!(
            value_after(&args, "-filter:a"),
            Some("asetpts=PTS-STARTPTS,apad,atrim=end=1.250000")
        );
        assert!(!args.iter().any(|arg| arg == "-shortest"));
        assert_eq!(args.last().map(String::as_str), Some("render.mp4"));
    }

    #[test]
    fn silent_export_maps_only_video() {
        let config = config(2.0);
        let args = build_ffmpeg_args(&config, None);

        assert!(args.iter().any(|arg| arg == "-an"));
        assert!(!args.iter().any(|arg| arg == "-filter:a"));
        assert!(!args.iter().any(|arg| arg == "1:a:0"));
        assert!(!args.iter().any(|arg| arg == "-shortest"));
    }

    #[test]
    fn frame_count_covers_fractional_requested_duration() {
        assert_eq!(export_frame_count(30, 1.0), 30);
        assert_eq!(export_frame_count(30, 1.001), 31);
        assert_eq!(export_frame_count(24, 1.25), 30);
    }

    #[test]
    fn master_pause_freezes_every_export_program_frame_at_zero() {
        for frame in [0, 1, 29, 300] {
            assert_eq!(
                export_program_transport(frame, 1.0 / 30.0, true),
                (0.0, 0.0)
            );
        }
        assert_eq!(export_program_transport(0, 1.0 / 30.0, false), (0.0, 0.0));
        let (time, dt) = export_program_transport(30, 1.0 / 30.0, false);
        assert!((time - 1.0).abs() < 1e-6);
        assert!((dt - 1.0 / 30.0).abs() < 1e-6);
        assert_eq!(export_ntsc_reference_frame(300, 60, true), 0);
        assert_eq!(export_ntsc_reference_frame(300, 60, false), 150);
    }

    #[test]
    fn recorded_manual_temporal_events_replay_on_one_reference_timeline() {
        let mut track = crate::temporal::TemporalEventTrack::default();
        assert!(track.record_accepted(
            700,
            crate::temporal::TemporalFrameEvents {
                manual_events: 2,
                ..crate::temporal::TemporalFrameEvents::default()
            }
        ));
        assert!(track.record_accepted(
            730,
            crate::temporal::TemporalFrameEvents {
                garden_refresh_events: 3,
                ..crate::temporal::TemporalFrameEvents::default()
            }
        ));
        for fps in [24, 30, 60] {
            let mut replay = track.replay();
            let first = replay.events_due(export_temporal_reference_tick(0, fps));
            assert_eq!(first.manual_events, 2, "{fps} fps");
            let one_second = replay.events_due(export_temporal_reference_tick(fps.into(), fps));
            assert_eq!(one_second.garden_refresh_events, 3, "{fps} fps");
            assert_eq!(
                replay.events_due(export_temporal_reference_tick((fps * 2).into(), fps)),
                crate::temporal::TemporalFrameEvents::default(),
                "{fps} fps"
            );
        }
    }

    #[test]
    fn master_pause_does_not_initialize_transient_modulation_only_offline() {
        use crate::modulation::{ModMatrix, ModSource, Routing};

        fn matrix_with_positive_lfo() -> ModMatrix {
            let mut matrix = ModMatrix::new();
            matrix.lfos[7].set_phase(0.25);
            matrix
                .routings
                .push(Routing::new(ModSource::Lfo(7), "brightness", 1.0));
            matrix
        }

        let mut paused = matrix_with_positive_lfo();
        update_export_modulation(&mut paused, 0.0, 0.0, true);
        assert_eq!(paused.routings[0].cached_value(), 0.0);

        let mut running = matrix_with_positive_lfo();
        update_export_modulation(&mut running, 0.0, 0.0, false);
        assert!(running.routings[0].cached_value() > 0.99);
    }

    #[test]
    fn master_pause_holds_materialized_bases_instead_of_resampling_morph() {
        let mut a = crate::morph::MorphSlot::default();
        a.master.brightness = -1.0;
        let mut b = crate::morph::MorphSlot::default();
        b.master.brightness = 1.0;
        let morph = crate::morph::Morph {
            a: Some(a),
            b: Some(b),
            t: 1.0,
            ..Default::default()
        };

        assert!(export_morph_sample(Some(&morph), 0.0, 0.0, true).is_none());
        let running = export_morph_sample(Some(&morph), 0.0, 0.0, false).unwrap();
        assert_eq!(running.master.brightness, 1.0);
    }

    #[test]
    fn export_dimensions_are_bounded_before_gpu_allocation() {
        assert!(validate_export_dimensions(0, 1080).is_err());
        assert!(validate_export_dimensions(1920, 0).is_err());
        assert!(validate_export_dimensions(3840, 2160).is_ok());
        assert!(validate_export_dimensions(4096, 2160).is_err());
        assert!(validate_export_dimensions(MAX_EXPORT_EDGE + 1, 2).is_err());
    }

    #[test]
    fn source_texture_scope_errors_are_descriptive_and_cpu_testable() {
        assert_eq!(
            export_source_texture_allocation_error("Expert source", 8192, 4096, &[]),
            None
        );
        let error = export_source_texture_allocation_error(
            "Expert source",
            8192,
            4096,
            &["out of memory".into(), "backend rejected allocation".into()],
        )
        .unwrap();
        assert!(error.contains("Expert source"));
        assert!(error.contains("8192x4096"));
        assert!(error.contains("out of memory; backend rejected allocation"));

        let setup = export_gpu_setup_error(3840, 2160, &["Out of Memory".into()]).unwrap();
        assert_eq!(
            setup,
            "could not initialize export GPU resources at 3840x2160: Out of Memory"
        );

        let sink = Arc::new(Mutex::new(vec![
            "frame validation".to_string(),
            "frame out of memory".to_string(),
        ]));
        assert_eq!(
            take_export_gpu_errors(&sink).as_deref(),
            Some("frame validation; frame out of memory")
        );
        assert!(take_export_gpu_errors(&sink).is_none());
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_export_upload_scope_reports_invalid_extent() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Export Upload Scope Test"),
            ..Default::default()
        }))
        .expect("GPU device");
        let (texture, _view) =
            create_export_source_texture(&device, 1, 1, "Export Upload Scope Test").unwrap();

        let error = upload_export_texture_checked(
            &device,
            &queue,
            &texture,
            &[0; 8],
            2,
            1,
            "Export Upload Scope Test",
        )
        .expect_err("extent wider than the texture must be recoverable");
        assert!(error.contains("Export Upload Scope Test"));
    }

    #[test]
    fn recoverable_source_warnings_survive_a_success_without_becoming_errors() {
        let progress = ExportProgress::new();
        progress.record_warning(black_substitution_warning(
            1,
            "missing.mov",
            "could not be resolved",
        ));
        finalize_export_worker(&progress, "", None);

        assert!(progress.done.load(Ordering::Acquire));
        assert_eq!(progress.outcome.load(Ordering::Acquire), OUTCOME_SUCCEEDED);
        assert!(progress.error.lock().unwrap().is_empty());
        assert_eq!(
            progress.warnings(),
            vec!["Layer 2 ('missing.mov') could not be resolved; substituted deterministic black."]
        );
    }

    #[test]
    fn content_addressed_export_sources_never_enter_legacy_black_fallback() {
        let digest = "0".repeat(64);
        assert!(strict_content_addressed_export_source(&format!(
            "cos-sha256://{digest}/12"
        )));
        assert!(strict_content_addressed_export_source(
            "cos-sha256://malformed"
        ));
        assert!(!strict_content_addressed_export_source(
            "C:/legacy/missing.mov"
        ));
    }

    #[test]
    fn export_preflight_reverifies_patch_adjacent_runtime_hints_outside_library() {
        let unique = format!(
            "collideoscope-export-source-hint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let patch_media = root.join("patch-media");
        let active_library = root.join("active-library");
        std::fs::create_dir_all(&patch_media).unwrap();
        std::fs::create_dir_all(&active_library).unwrap();
        let visual = patch_media.join("outside.mp4");
        let analysis_audio = patch_media.join("outside.wav");
        std::fs::write(&visual, b"frame-one").unwrap();
        std::fs::write(&analysis_audio, b"audio-one").unwrap();

        let mut identity_session = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        let visual_reference = identity_session
            .fingerprint(&visual)
            .unwrap()
            .source_reference();
        let audio_reference = identity_session
            .fingerprint(&analysis_audio)
            .unwrap()
            .source_reference();
        let visual_hint = visual.to_string_lossy().into_owned();
        let audio_hint = analysis_audio.to_string_lossy().into_owned();
        let context = crate::media_source::ResolveContext::new(None, Some(active_library));

        let mut preflight = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        let resolved_visual = resolve_export_visual_source(
            &visual_reference,
            "outside.mp4",
            Some(&visual_hint),
            &context,
            &mut preflight,
        )
        .unwrap();
        assert!(matches!(
            resolved_visual,
            crate::media_source::ResolvedVisualSource::File(ref resolved)
                if resolved.path == visual.canonicalize().unwrap()
        ));
        let resolved_audio = resolve_export_file_source(
            &audio_reference,
            "outside.wav",
            Some(&audio_hint),
            &context,
            |path: &std::path::Path| crate::audio::is_supported_audio_file(path),
            &mut preflight,
        )
        .unwrap();
        assert_eq!(resolved_audio.path, analysis_audio.canonicalize().unwrap());

        // Same-length replacements prove that export trusts the digest, not
        // the retained host path or a stale metadata-only admission decision.
        std::fs::write(&visual, b"frame-two").unwrap();
        std::fs::write(&analysis_audio, b"audio-two").unwrap();
        let mut changed_preflight = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        assert!(resolve_export_visual_source(
            &visual_reference,
            "outside.mp4",
            Some(&visual_hint),
            &context,
            &mut changed_preflight,
        )
        .is_err());
        let mut changed_audio_preflight = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        assert!(resolve_export_file_source(
            &audio_reference,
            "outside.wav",
            Some(&audio_hint),
            &context,
            |path: &std::path::Path| crate::audio::is_supported_audio_file(path),
            &mut changed_audio_preflight,
        )
        .is_err());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn muxed_export_audio_accepts_audio_only_and_video_sources_but_not_stills() {
        // The predicate law itself: a muxed audio source is any media file
        // that can carry an audio stream — audio files and videos — and
        // never a still, which carries none.
        assert!(is_supported_export_audio_source(std::path::Path::new(
            "track.wav"
        )));
        assert!(is_supported_export_audio_source(std::path::Path::new(
            "track.flac"
        )));
        assert!(is_supported_export_audio_source(std::path::Path::new(
            "clip.mp4"
        )));
        assert!(!is_supported_export_audio_source(std::path::Path::new(
            "image.png"
        )));

        // A content-addressed audio-only export audio source resolves
        // through the muxed-audio predicate, exactly what ExportConfig's
        // "optional media file" doc promises.
        let unique = format!(
            "collide-o-scope-muxed-audio-accepts-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let library = root.join("library");
        std::fs::create_dir_all(&library).unwrap();
        let score = library.join("score.flac");
        std::fs::write(&score, b"audio-only export source bytes").unwrap();

        let mut identity_session = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        let reference = identity_session
            .fingerprint(&score)
            .unwrap()
            .source_reference();
        let context = crate::media_source::ResolveContext::new(None, Some(library));

        let mut session = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        let resolved = resolve_export_file_source(
            &reference,
            "score.flac",
            None,
            &context,
            is_supported_export_audio_source,
            &mut session,
        )
        .unwrap();
        assert_eq!(resolved.path, score.canonicalize().unwrap());

        // The defect this fix removes, pinned so it cannot return: the
        // visual-file predicate filters the same audio-only source out and
        // resolution fails.
        let mut wrong_predicate_session = crate::media_source::FingerprintSession::new(
            crate::media_source::FingerprintLimits::default(),
        )
        .unwrap();
        assert!(resolve_export_file_source(
            &reference,
            "score.flac",
            None,
            &context,
            crate::layers::is_supported_visual_file,
            &mut wrong_predicate_session,
        )
        .is_err());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn export_warning_snapshot_is_count_and_character_bounded() {
        let progress = ExportProgress::new();
        progress.record_warning("é".repeat(MAX_EXPORT_WARNING_CHARS + 20));
        for index in 1..=MAX_EXPORT_WARNINGS {
            progress.record_warning(format!("warning {index}"));
        }

        let warnings = progress.warnings();
        assert_eq!(warnings.len(), MAX_EXPORT_WARNINGS);
        assert_eq!(warnings[0].chars().count(), MAX_EXPORT_WARNING_CHARS);
        assert!(warnings[0].ends_with('…'));
        assert_eq!(
            warnings.last().map(String::as_str),
            Some("Additional export source warnings were omitted.")
        );
    }

    #[test]
    fn unreadable_explicit_audio_is_a_probe_error_not_no_stream() {
        let missing = std::env::temp_dir().join(format!(
            "collideoscope-missing-audio-probe-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let error = media_has_audio_stream(&missing.to_string_lossy()).unwrap_err();
        assert!(error.contains("failed to open selected media"));
    }

    #[test]
    fn every_saved_blend_mode_has_an_exact_export_equivalent() {
        for mode in BlendMode::ALL {
            assert_eq!(configured_blend_mode(mode.key()), mode);
        }
        assert_eq!(configured_blend_mode("unknown"), BlendMode::Normal);
    }

    #[test]
    fn cancellation_request_is_prompt_and_nonterminal_until_cleanup() {
        let progress = ExportProgress::new();
        progress.encoder_active.store(true, Ordering::Release);
        progress
            .encoder_cleanup_complete
            .store(false, Ordering::Release);
        let started = std::time::Instant::now();
        progress.request_cancel();
        assert!(started.elapsed() < Duration::from_millis(25));
        assert!(progress.cancel.load(Ordering::Acquire));
        assert!(!progress.done.load(Ordering::Acquire));
        assert_eq!(
            progress.outcome.load(Ordering::Acquire),
            OUTCOME_CANCEL_REQUESTED
        );
    }

    #[test]
    fn pre_spawn_cancel_has_no_late_child_or_partial_output() {
        let path = std::env::temp_dir().join(format!(
            "collideoscope-supervisor-prespawn-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let progress = Arc::new(ExportProgress::new());
        progress.request_cancel();
        let result = start_encoder_supervisor(
            "definitely-not-a-real-encoder".to_string(),
            Vec::new(),
            progress.clone(),
            path.to_string_lossy().into_owned(),
        );
        assert!(result.is_err());
        assert!(progress.done.load(Ordering::Acquire));
        assert!(progress.cancelled.load(Ordering::Acquire));
        assert!(progress.encoder_cleanup_complete.load(Ordering::Acquire));
        assert!(!path.exists());
    }

    #[test]
    fn repeated_cancel_is_idempotent() {
        let progress = ExportProgress::new();
        progress.request_cancel();
        progress.request_cancel();
        assert_eq!(
            progress.outcome.load(Ordering::Acquire),
            OUTCOME_CANCEL_REQUESTED
        );
    }

    #[test]
    fn concurrent_cancel_and_failure_never_overwrite_an_accepted_cancel() {
        for _ in 0..100 {
            let progress = Arc::new(ExportProgress::new());
            let barrier = Arc::new(std::sync::Barrier::new(3));
            let cancel_progress = progress.clone();
            let cancel_barrier = barrier.clone();
            let cancel = std::thread::spawn(move || {
                cancel_barrier.wait();
                cancel_progress.request_cancel();
            });
            let failure_progress = progress.clone();
            let failure_barrier = barrier.clone();
            let failure = std::thread::spawn(move || {
                failure_barrier.wait();
                finalize_export_worker(
                    &failure_progress,
                    "",
                    Some("synthetic failure".to_string()),
                );
            });
            barrier.wait();
            cancel.join().unwrap();
            failure.join().unwrap();

            assert!(progress.done.load(Ordering::Acquire));
            if progress.cancel.load(Ordering::Acquire) {
                assert!(progress.cancelled.load(Ordering::Acquire));
                assert_eq!(progress.outcome.load(Ordering::Acquire), OUTCOME_CANCELLED);
            } else {
                assert_eq!(progress.outcome.load(Ordering::Acquire), OUTCOME_FAILED);
            }
        }
    }

    #[test]
    fn cleaned_terminal_job_is_replaceable_while_gpu_worker_unwinds() {
        let progress = Arc::new(ExportProgress::new());
        progress.done.store(true, Ordering::Release);
        progress
            .encoder_cleanup_complete
            .store(true, Ordering::Release);
        let (release, blocked) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = blocked.recv();
        });
        let mut job = ExportJob {
            progress,
            thread: Some(thread),
            output_path: String::new(),
        };
        assert!(job.can_replace());
        release.send(()).unwrap();
        job.thread.take().unwrap().join().unwrap();
    }

    #[test]
    fn terminal_cancel_follows_partial_removal() {
        let path = std::env::temp_dir().join(format!(
            "collideoscope-cancel-order-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"partial").unwrap();
        let progress = ExportProgress::new();
        progress.output_started.store(true, Ordering::Release);
        progress.request_cancel();
        publish_cancelled_terminal(&progress, Some(&path.to_string_lossy()));
        assert!(!path.exists());
        assert!(progress.done.load(Ordering::Acquire));
        assert!(progress.cancelled.load(Ordering::Acquire));
    }

    #[cfg(windows)]
    #[test]
    fn supervisor_owns_child_until_reap_then_publishes_cancel() {
        let path = std::env::temp_dir().join(format!(
            "collideoscope-supervisor-cancel-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"partial").unwrap();
        let progress = Arc::new(ExportProgress::new());
        let session = start_encoder_supervisor(
            "powershell.exe".to_string(),
            vec![
                "-NoLogo".to_string(),
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-Command".to_string(),
                "Start-Sleep -Seconds 30".to_string(),
            ],
            progress.clone(),
            path.to_string_lossy().into_owned(),
        )
        .unwrap();

        let requested = std::time::Instant::now();
        progress.request_cancel();
        assert!(requested.elapsed() < Duration::from_millis(25));
        drop(session.stdin);
        let completion = session
            .completion
            .recv_timeout(Duration::from_secs(2))
            .expect("supervisor did not report completion");
        session.supervisor.join().unwrap();

        assert!(completion.status.is_ok());
        assert!(progress.encoder_cleanup_complete.load(Ordering::Acquire));
        assert!(!progress.encoder_active.load(Ordering::Acquire));
        assert!(!path.exists());
        assert!(progress.done.load(Ordering::Acquire));
        assert!(progress.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn drop_deadline_includes_the_cancel_request() {
        let progress = Arc::new(ExportProgress::new());
        let (release, blocked) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = blocked.recv();
        });
        let job = ExportJob {
            progress,
            thread: Some(thread),
            output_path: String::new(),
        };
        let started = std::time::Instant::now();
        drop(job);
        assert!(started.elapsed() < Duration::from_millis(1200));
        release.send(()).unwrap();
    }

    /// Exact regression for the former live hang: ffmpeg's slow encoder can
    /// back up the raw-video pipe at 720p60, so cancelling around 3% must
    /// break that write, reap the process, publish `cancelled`, and delete the
    /// partial MP4. Run explicitly on a GPU-equipped host with the audit clips.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit*.mp4"]
    fn live_720p60_cancellation_is_bounded_and_clean() {
        use crate::patch::{EffectsConfig, LayerConfig, PatchState};

        for filename in ["audit.mp4", "audit_audio.mp4"] {
            assert!(
                std::path::Path::new("videos").join(filename).is_file(),
                "missing videos/{filename}"
            );
        }
        let layers = ["audit.mp4", "audit_audio.mp4"]
            .into_iter()
            .map(|filename| LayerConfig {
                filename: filename.to_string(),
                source_path: String::new(),
                opacity: 1.0,
                mosh_send: 1.0,
                blend_mode: "normal".to_string(),
                bypass_master_fx: false,
                bypass_temporal_fx: false,
                reroll_on_loop: false,
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                effects: EffectsConfig::default(),
                transform: SpatialTransform::default(),
                rack: None,
                motion: None,
                clip_slots: crate::performance::ClipSlots::singleton(
                    crate::performance::ClipSlotConfig::from_legacy(
                        filename.to_string(),
                        String::new(),
                        1.0,
                        30.0,
                    ),
                ),
                active_clip_slot: Some(crate::performance::ClipSlotId::LEGACY),
                matte: crate::image_routing::LayerMatteConfig::default(),
                pattern: None,
                text_page: None,
            })
            .collect();
        let patch = PatchState {
            master: EffectsConfig::default(),
            master_transform: SpatialTransform::default(),
            master_rack: None,
            master_motion: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: false,
            media_frozen: false,
            layers,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
            snapshot_bank: None,
            scenes: crate::performance::Scenes::default(),
            autopilot: crate::performance::AutopilotPlan::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        };
        let output = std::env::temp_dir().join(format!(
            "collideoscope-live-cancel-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = ExportConfig {
            width: 1280,
            height: 720,
            fps: 60,
            duration_secs: 30.0,
            output_path: output.to_string_lossy().into_owned(),
            audio_path: None,
            audio_path_hint: None,
            layer_source_hints: Vec::new(),
            analysis_audio_path_hint: None,
            ntsc_quality: NtscExportQuality::LiveParity,
            shutter_samples: ExportShutterSamples::Authored,
            media_safety_policy: MediaSafetyPolicy::default(),
            temporal_event_track: crate::temporal::TemporalEventTrack::default(),
            gesture_track: None,
            performance_take: None,
        };
        let mut job = ExportJob::start(patch, config, "videos");

        let progress_deadline = std::time::Instant::now() + Duration::from_secs(60);
        while job.progress.progress.load(Ordering::Relaxed) < 300 && !job.is_done() {
            assert!(
                std::time::Instant::now() < progress_deadline,
                "export did not reach 3% within 60 seconds"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!job.is_done(), "export finished before cancellation point");

        let cancel_started = std::time::Instant::now();
        job.cancel();
        while !job.is_done() && cancel_started.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(job.is_done(), "cancellation exceeded three seconds");
        assert!(job.progress.cancelled.load(Ordering::Relaxed));
        assert!(!output.exists(), "cancelled export left a partial MP4");
        assert!(job
            .progress
            .encoder_cleanup_complete
            .load(Ordering::Acquire));
        assert!(!job.progress.encoder_active.load(Ordering::Acquire));

        let worker_deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !job.thread.as_ref().unwrap().is_finished()
            && std::time::Instant::now() < worker_deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            job.thread.as_ref().unwrap().is_finished(),
            "export worker remained stuck while destroying cancelled GPU resources"
        );
        job.thread.take().unwrap().join().unwrap();
        eprintln!(
            "720p60 cancellation committed and worker exited in {:?}",
            cancel_started.elapsed()
        );
    }
}

#[cfg(test)]
mod effects_audit {
    use super::*;
    use crate::patch::{EffectsConfig, LayerConfig, NtscConfig, PatchState, TemporalConfig};

    fn base_patch() -> PatchState {
        PatchState {
            master: EffectsConfig::default(),
            master_transform: SpatialTransform::default(),
            master_rack: None,
            master_motion: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: false,
            media_frozen: false,
            layers: vec![LayerConfig {
                filename: "audit.mp4".to_string(),
                source_path: String::new(),
                opacity: 1.0,
                mosh_send: 1.0,
                blend_mode: "normal".to_string(),
                bypass_master_fx: false,
                bypass_temporal_fx: false,
                reroll_on_loop: false,
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                effects: EffectsConfig::default(),
                transform: SpatialTransform::default(),
                rack: None,
                motion: None,
                clip_slots: crate::performance::ClipSlots::singleton(
                    crate::performance::ClipSlotConfig::from_legacy(
                        "audit.mp4".to_string(),
                        String::new(),
                        1.0,
                        30.0,
                    ),
                ),
                active_clip_slot: Some(crate::performance::ClipSlotId::LEGACY),
                matte: crate::image_routing::LayerMatteConfig::default(),
                pattern: None,
                text_page: None,
            }],
            ntsc: Some(NtscConfig::default()),
            modulation: None,
            temporal: Some(TemporalConfig::default()),
            morph: None,
            snapshot_bank: None,
            scenes: crate::performance::Scenes::default(),
            autopilot: crate::performance::AutopilotPlan::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
        }
    }

    fn render(label: &str, patch: PatchState) {
        render_with_gesture(label, patch, None);
    }

    /// The one labeled-render helper. A recorded gesture performance is carried
    /// by value exactly as the live host hands it over, and the committed output
    /// path is returned so a case can inspect the sidecar it published.
    fn render_with_gesture(
        label: &str,
        patch: PatchState,
        gesture_track: Option<crate::gesture::GestureTrackDocument>,
    ) -> String {
        let config = ExportConfig {
            width: 320,
            height: 180,
            fps: 24,
            duration_secs: 1.0,
            output_path: format!("renders/audit_{label}.mp4"),
            audio_path: None,
            audio_path_hint: None,
            layer_source_hints: Vec::new(),
            analysis_audio_path_hint: None,
            ntsc_quality: NtscExportQuality::LiveParity,
            shutter_samples: ExportShutterSamples::Authored,
            media_safety_policy: MediaSafetyPolicy::default(),
            temporal_event_track: crate::temporal::TemporalEventTrack::default(),
            gesture_track,
            performance_take: None,
        };
        let output_path = config.output_path.clone();
        let job = ExportJob::start(patch, config, "videos");
        while !job.is_done() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let err = job.progress.error.lock().unwrap().clone();
        assert!(err.is_empty(), "{label}: export failed: {err}");
        output_path
    }

    /// The S6 labeled export case: a transform authored by the preview gizmo,
    /// rendered offline beside the identical transform authored numerically.
    ///
    /// Two files come out of this, and the claim is that they are decoded-frame
    /// identical — `ffmpeg -i renders/audit_native_gizmo_transform.mp4 -f
    /// framemd5 -` against the `_numeric` twin. That is the whole S6 pixel
    /// claim: the gizmo introduces no geometry, so an offline render cannot
    /// tell which surface authored the transform.
    ///
    /// It is also the case that proves the gizmo reaches the pixels at all —
    /// the transform is a real rotation, scale, and translation well away from
    /// the legacy identity, so a gizmo that silently authored nothing would
    /// produce a frame identical to `base_patch` instead.
    ///
    /// There is deliberately no export-only gizmo path to exercise: the gizmo
    /// exists only in the editor preview, and what crosses into a patch is an
    /// ordinary `SpatialTransform`.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_native_gizmo_transform_pipeline() {
        use crate::transform_gizmo::{
            GizmoDrag, GizmoEdits, GizmoFrame, GizmoModifiers, GizmoScope,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        const OUTPUT: (u32, u32) = (320, 180);

        // Author through the real drag law, exactly as the host would: grab a
        // handle, move the pointer, take the absolute values it asks for.
        let mut authored = SpatialTransform::default();
        let apply = |edits: GizmoEdits, into: &mut SpatialTransform| {
            for edit in edits.iter() {
                assert!(
                    crate::App::apply_spatial_transform_edit(
                        into,
                        edit.param.as_str(),
                        &serde_json::json!(f64::from(edit.value)),
                    ),
                    "the gizmo vocabulary must be accepted by the one authoring function"
                );
            }
        };

        // A body drag, then a corner scale, then a rotation about the anchor.
        let frame = GizmoFrame::new(authored, OUTPUT, OUTPUT).expect("identity is renderable");
        let move_handle = frame
            .handle_position(crate::transform_gizmo::GizmoHandle::Translate)
            .expect("the move handle is placed");
        let (translate, _) =
            GizmoDrag::begin(GizmoScope::Master, frame, move_handle).expect("move handle hit");
        apply(
            translate.update(
                [move_handle[0] + 0.15, move_handle[1] - 0.15],
                GizmoModifiers::NONE,
            ),
            &mut authored,
        );

        let frame = GizmoFrame::new(authored, OUTPUT, OUTPUT).expect("still renderable");
        let corner = frame
            .handle_position(crate::transform_gizmo::GizmoHandle::Scale(
                crate::transform_gizmo::ScaleCorner::BottomRight,
            ))
            .expect("the corner handle is placed");
        let (scale, _) = GizmoDrag::begin(GizmoScope::Master, frame, corner).expect("corner hit");
        apply(
            scale.update([corner[0] - 0.18, corner[1] - 0.18], GizmoModifiers::NONE),
            &mut authored,
        );

        let frame = GizmoFrame::new(authored, OUTPUT, OUTPUT).expect("still renderable");
        let rotate_handle = frame
            .handle_position(crate::transform_gizmo::GizmoHandle::Rotate)
            .expect("the rotate handle is placed");
        let (rotate, _) =
            GizmoDrag::begin(GizmoScope::Master, frame, rotate_handle).expect("rotate hit");
        let pivot = frame.anchor_output();
        apply(
            rotate.update([pivot[0] + 0.3, pivot[1] - 0.1], GizmoModifiers::NONE),
            &mut authored,
        );

        assert_ne!(
            authored,
            SpatialTransform::default(),
            "the drag sequence must leave the legacy identity, or this case proves nothing"
        );

        // The identical values, authored the way the browser numeric editor
        // does: one absolute field at a time, through the same function.
        let mut numeric = SpatialTransform::default();
        for (param, value) in [
            ("position_x", f64::from(authored.position[0])),
            ("position_y", f64::from(authored.position[1])),
            ("scale_x", f64::from(authored.scale[0])),
            ("scale_y", f64::from(authored.scale[1])),
            ("rotation_deg", f64::from(authored.rotation_deg)),
        ] {
            assert!(crate::App::apply_spatial_transform_edit(
                &mut numeric,
                param,
                &serde_json::json!(value)
            ));
        }
        assert_eq!(
            numeric, authored,
            "the two authoring surfaces must agree before a single frame renders"
        );

        let mut gizmo_patch = base_patch();
        gizmo_patch.master_transform = authored;
        render("native_gizmo_transform", gizmo_patch);

        let mut numeric_patch = base_patch();
        numeric_patch.master_transform = numeric;
        render("native_gizmo_transform_numeric", numeric_patch);

        // The untouched twin. Comparing against it is what makes the case
        // discriminating rather than merely self-consistent: without it, a
        // gizmo that authored nothing at all would still produce two identical
        // files and look like a pass.
        render("native_gizmo_transform_identity", base_patch());
    }

    /// Two stacked clips, an ordinary single-donor Faraday transplant, and no
    /// image tap anywhere.
    ///
    /// Before the composite-rank tie-break this composition could not be
    /// prepared at all. `execution_order` ordered the two independent siblings
    /// by ascending `StableLayerId`, export numbers layers `position + 1`
    /// front-to-back, and `build_block_schedules` drains back-to-front — so the
    /// second drain landed on a layer that owned no retained tap and
    /// preparation failed with "executed before structural admission without a
    /// current retained tap". It is the smallest composition that reproduces
    /// the defect: one Motion effect to force an Advanced plan, and nothing
    /// else at all.
    ///
    /// Every other labeled export case carries a rack node whose tap happened
    /// to retain the sibling and hide this. This one deliberately carries none.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_tapless_advanced_motion_pipeline() {
        use crate::patch::{FaradayConfig, MotionCarrierConfig, MotionConfig, MotionDonorConfig};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        let donor = patch.layers[0].clone();
        patch.layers.push(donor);
        patch.layers[1].effects.hue_shift = 200.0;

        // No rack on either layer, so the composition owns no image tap and the
        // schedule has nothing to retain.
        assert!(patch.layers.iter().all(|layer| layer.rack.is_none()));

        patch.layers[0].motion = Some(MotionConfig {
            transplant: FaradayConfig {
                amount: 0.85,
                donor: MotionDonorConfig::Selected {
                    saved_position: SavedLayerPosition::new(1).expect("layer 1 exists"),
                },
                carrier: MotionCarrierConfig::FirstSourceFrame,
                refresh: 0.35,
                decay: 0.9,
                ..FaradayConfig::default()
            },
            ..MotionConfig::default()
        });
        render("tapless_advanced_motion", patch);
    }

    /// Renders every effect through the real shader chain into labeled
    /// files under renders/, for objective pixel-level verification
    /// (ffprobe signalstats/entropy). Needs a GPU, ffmpeg on PATH, and
    /// videos/audit.mp4 — run explicitly:
    ///   cargo test --release effects_audit -- --ignored --nocapture
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_final_program_vhs_over_mixed_master_bypass() {
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        let mut patch = base_patch();
        let mut bottom = patch.layers[0].clone();
        bottom.opacity = 0.75;
        bottom.blend_mode = "multiply".into();
        patch.layers.push(bottom);
        patch.layers[0].bypass_master_fx = true;
        patch.layers[0].opacity = 0.6;
        patch.layers[0].blend_mode = "difference".into();
        patch.master.hue_shift = 75.0;
        let ntsc = patch.ntsc.as_mut().unwrap();
        ntsc.enabled = true;
        ntsc.snow_intensity = 0.45;
        ntsc.edge_wave_enabled = true;
        ntsc.edge_wave_intensity = 4.0;
        render("final_program_vhs_mixed_master_bypass", patch);

        let mut all_bypass = base_patch();
        all_bypass.layers[0].bypass_master_fx = true;
        let ntsc = all_bypass.ntsc.as_mut().unwrap();
        ntsc.enabled = true;
        ntsc.snow_intensity = 0.45;
        render("final_program_vhs_all_master_bypass", all_bypass);
    }

    /// Two stacked clips where the upper layer's Displace reads the lower
    /// layer's post-local image as its vector field. This is the labeled
    /// export case for the shared node implementation: the offline renderer
    /// consumes the same evaluated plan and the same rack shader, with no
    /// export-only displacement path.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_displace_two_input_pipeline() {
        use crate::visual_rack::{
            DisplaceBoundary, DisplaceParams, EdgeTiming, LegacyRackScope, SavedImageSource,
            SavedImageTap, VisualNodeKind, VisualRack,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        let donor = patch.layers[0].clone();
        patch.layers.push(donor);

        // Layer 0 is the upper scope; it displaces itself by the layer below.
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Displace(DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::OneBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            amount_x: 0.35,
            amount_y: -0.2,
            boundary: DisplaceBoundary::Wrap,
        }))
        .unwrap();
        patch.layers[0].rack = Some(rack);
        patch.layers[0].effects.hue_shift = 90.0;
        render("displace_two_input", patch);
    }

    /// One layer whose rack carries a dedicated eight-texture Symmetry Field
    /// reading its own carrier, the layer below through image slot 0, the
    /// committed Compat8 clean-history ring, and the layer below's primitive
    /// motion vector/gate pair through motion slot 0 — whose own visible Motion
    /// effect is exactly zero, so the donor field exists only because the
    /// planner set `required_as_donor`. Image slot 1 and motion slot 1 stay
    /// unarmed, so one armed and one unarmed slot of each class travel through
    /// the same render.
    ///
    /// This is the labeled export case for the dedicated pass: the offline
    /// renderer builds the same evaluated plan, constructs the same
    /// `SymmetryFieldExecutor`, and runs the same `symmetry_field.wgsl`. There
    /// is no export-only symmetry path. Its `.motion.json` sidecar carries the
    /// schema-4 per-slot route records.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_symmetry_field_dedicated_pass() {
        use crate::symmetry::{
            SavedMotionDonor, SymmetryBoundary, SymmetryMode, SymmetryMotionMask, SymmetryParams,
            SymmetrySourceMask,
        };
        use crate::visual_rack::{
            EdgeTiming, LegacyRackScope, SavedImageSource, SavedImageTap, VisualNodeKind,
            VisualRack,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        let donor = patch.layers[0].clone();
        patch.layers.push(donor);

        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Symmetry(SymmetryParams {
            mode: SymmetryMode::Dihedral,
            base_folds: 6.0,
            radial_phase_deg: 17.0,
            center: [0.4137, 0.5279],
            boundary: SymmetryBoundary::Mirror,
            hue_span: 0.25,
            motion_gain: 0.4,
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
            donors: [
                SavedImageTap {
                    source: SavedImageSource::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
                SavedImageTap::default(),
            ],
            motion: [
                SavedMotionDonor::Selected {
                    saved_position: SavedLayerPosition::new(1).expect("layer 1 exists"),
                },
                SavedMotionDonor::None,
            ],
            ..SymmetryParams::default()
        }))
        .unwrap();
        patch.layers[0].rack = Some(rack);
        render("symmetry_field", patch);
    }

    /// One clip whose layer rack carries a Study reading the carrier, the
    /// clean-history ring, the beat phase, and its deterministic randomness,
    /// hue-rotating a history mix over the live image.
    ///
    /// This is the labeled export case for the Study authored surface: the
    /// offline renderer resolves the same digest against the patch's own
    /// `studies` section, builds the same evaluated plan, and runs the same
    /// fixed `study_interpreter.wgsl`. There is no export-only Study path.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_study_field_pipeline() {
        use crate::study::{
            StudyAbiVersion, StudyCapability, StudyInstruction, StudyLicenseNotice, StudyMetadata,
            StudyPublicationBoundary, StudyRegister, STUDY_SCHEMA_VERSION,
        };
        use crate::visual_rack::{LegacyRackScope, VisualNodeKind, VisualRack};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let register = |value: u8| StudyRegister::new(value).unwrap();
        let document = crate::study::StudyDocument {
            schema_version: STUDY_SCHEMA_VERSION,
            abi: StudyAbiVersion::default(),
            metadata: StudyMetadata {
                name: "Audit study".into(),
                author: "effects_audit".into(),
                description: "History mix hue-rotated by beat and randomness".into(),
                license: StudyLicenseNotice {
                    identifier: "CC0-1.0".into(),
                    notice: String::new(),
                    publication_boundary: StudyPublicationBoundary::StudyDataOnlyDoesNotLicenseHost,
                },
            },
            capabilities: vec![
                StudyCapability::CurrentColor,
                StudyCapability::HistoryRead,
                StudyCapability::BeatPhase,
                StudyCapability::DeterministicRandom,
            ],
            instructions: vec![
                StudyInstruction::LoadCurrentColor { dst: register(0) },
                StudyInstruction::LoadHistoryColor {
                    dst: register(1),
                    age: 6,
                },
                StudyInstruction::LoadBeatPhase { dst: register(2) },
                StudyInstruction::LoadDeterministicRandom {
                    dst: register(3),
                    domain: 5,
                },
                StudyInstruction::ConstantScalar {
                    dst: register(4),
                    value: 0.5,
                },
                StudyInstruction::Mix {
                    dst: register(5),
                    a: register(0),
                    b: register(1),
                    amount: register(4),
                },
                StudyInstruction::Multiply {
                    dst: register(6),
                    left: register(2),
                    right: register(3),
                },
                StudyInstruction::HueRotate {
                    dst: register(7),
                    color: register(5),
                    turns: register(6),
                },
                StudyInstruction::OutputColor { color: register(7) },
            ],
        };
        let compiled = crate::study_eval::CompiledStudy::compile(&document).unwrap();
        let digest = *compiled.canonical_digest();

        let mut patch = base_patch();
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Study(crate::visual_rack::StudyRackParams {
            document_digest: Some(digest),
        }))
        .unwrap();
        patch.layers[0].rack = Some(rack);
        patch.studies = vec![document];
        render("study_field", patch);
    }

    /// Two stacked clips whose upper layer's Faraday carrier is advected by a
    /// Field Collider. Input A names the recipient itself and input B names the
    /// layer beneath it, which section 5 explicitly permits: either input may
    /// equal the recipient, and only A aliasing B is forbidden.
    ///
    /// This is the labeled export case for the Motion-subsystem block: the
    /// offline renderer consumes the same evaluated plan, the same two
    /// low-resolution passes, and the same `motion_collide.wgsl`, with no
    /// export-only collider path.
    ///
    /// The case carries a small Displace node for a reason unrelated to the
    /// collider. An Advanced composition whose layers own NO image tap cannot
    /// be scheduled today: `execution_order` breaks ties between equally-ready
    /// sibling scopes with `BTreeSet::pop_first`, which orders by ascending
    /// stable id, while `build_root_schedule` requires the composition's
    /// back-to-front order. Export assigns ids front-to-back, so the two
    /// disagree and preparation fails with "executed before structural
    /// admission without a current retained tap". A plain single-donor Faraday
    /// transplant on a two-layer tapless stack reproduces it exactly, with no
    /// collider present, so it is a pre-existing M4 defect rather than a
    /// collider one — motion simply had no labeled export case before now to
    /// surface it. Every other labeled Advanced export case carries a rack node
    /// for the same structural reason, and one image tap is what retains the
    /// sibling and makes the schedule legal.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_field_collider_pipeline() {
        use crate::patch::{
            FaradayConfig, FieldColliderConfig, FieldColliderModeConfig, MotionBoundaryModeConfig,
            MotionCarrierConfig, MotionConfig, MotionDonorConfig,
        };
        use crate::visual_rack::{
            DisplaceBoundary, DisplaceParams, EdgeTiming, LegacyRackScope, SavedImageSource,
            SavedImageTap, VisualNodeKind, VisualRack,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        let donor = patch.layers[0].clone();
        patch.layers.push(donor);
        // Give the lower input visibly different local colour so the collided
        // advection is legible against either input taken alone.
        patch.layers[1].effects.hue_shift = 200.0;

        // A small Displace on the upper layer, tapping the layer below. It is
        // NOT part of what this case demonstrates: it exists because an
        // Advanced composition whose layers own no image tap at all cannot be
        // scheduled today (see the doc comment above), and every other labeled
        // Advanced export case carries a rack node for the same structural
        // reason. Its amounts are deliberately small so the collided advection
        // remains the legible subject of the render.
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Displace(DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::OneBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            amount_x: 0.02,
            amount_y: 0.02,
            boundary: DisplaceBoundary::Hold,
        }))
        .unwrap();
        patch.layers[0].rack = Some(rack);

        // Layer 0 is the recipient and owns the shared carrier and advection
        // controls. Input A is the recipient's own field, input B the layer
        // below it; neither carries an authored Motion effect of its own, so
        // only `required_as_donor` pulls their primitive fields into the plan.
        patch.layers[0].motion = Some(MotionConfig {
            transplant: FaradayConfig {
                amount: 0.85,
                carrier: MotionCarrierConfig::FirstSourceFrame,
                confidence_threshold: 0.05,
                confidence_softness: 0.1,
                refresh: 0.35,
                decay: 0.9,
                ..FaradayConfig::default()
            },
            collider: FieldColliderConfig {
                enabled: true,
                mode: FieldColliderModeConfig::CollisionBoundary,
                boundary: MotionBoundaryModeConfig::Mirror,
                input_a: MotionDonorConfig::Selected {
                    saved_position: SavedLayerPosition::new(0).expect("layer 0 exists"),
                },
                input_b: MotionDonorConfig::Selected {
                    saved_position: SavedLayerPosition::new(1).expect("layer 1 exists"),
                },
                ..FieldColliderConfig::default()
            },
            ..MotionConfig::default()
        });
        render("field_collider", patch);
    }

    /// One clip whose Faraday carrier is advected by a synthetic B2 Curl
    /// field. The layer is its own donor, so the only field in the plan is the
    /// procedural one, and the stack is deliberately tapless — the composite
    /// rank landed, so a bare Advanced motion case schedules without a
    /// chaperone rack node.
    ///
    /// This is the labeled export case for B2: the offline renderer consumes
    /// the same evaluated plan, the same `motion_procedural.wgsl` synthesis
    /// pass, and the same advection path, with the pass's only time input the
    /// shared frame-plan context derived from `frame_num` and the export FPS.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_procedural_motion_field_pipeline() {
        use crate::patch::{
            FaradayConfig, MotionCarrierConfig, MotionConfig, MotionDonorConfig,
            MotionFieldSourceConfig, ProceduralFieldConfig,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        patch.layers[0].motion = Some(MotionConfig {
            field_source: MotionFieldSourceConfig::ProceduralCurl,
            procedural: ProceduralFieldConfig {
                scale: 0.35,
                rate: 0.5,
            },
            transplant: FaradayConfig {
                amount: 0.85,
                carrier: MotionCarrierConfig::FirstSourceFrame,
                confidence_threshold: 0.05,
                confidence_softness: 0.1,
                refresh: 0.35,
                decay: 0.9,
                donor: MotionDonorConfig::Selected {
                    saved_position: SavedLayerPosition::new(0).expect("layer 0 exists"),
                },
                ..FaradayConfig::default()
            },
            ..MotionConfig::default()
        });
        render("procedural_motion_field", patch);
    }

    /// Decoded-frame digests for one rendered file, via the same ffmpeg CLI
    /// the export path already requires.
    fn decoded_framemd5(path: &str) -> String {
        let output = std::process::Command::new(crate::host_paths::ffmpeg())
            .args(["-v", "error", "-i", path, "-f", "framemd5", "-"])
            .output()
            .expect("ffmpeg framemd5");
        assert!(output.status.success(), "framemd5 failed for {path}");
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    /// The B2 flow-shaping labeled export case: a self-donor Faraday advection
    /// driven by a Radial field, with all three shaping controls authored —
    /// stretch growing the flow outward, edge repel pushing off luma edges,
    /// and vector trash shoving hashed cells on its fixed 8 Hz event clock.
    /// Offline consumes the same apply shader and the same frame-plan time, so
    /// the trash epochs are frame-indexed and the render is repeatable.
    ///
    /// The case renders an `_unshaped` twin — identical patch, shaping zero —
    /// and asserts the decoded frames differ: shaping demonstrably reaches the
    /// pixels through the real export path, and an authored zero demonstrably
    /// remains a different program.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_motion_flow_shaping_pipeline() {
        use crate::patch::{
            FaradayConfig, FlowShapingConfig, MotionCarrierConfig, MotionConfig, MotionDonorConfig,
            MotionFieldSourceConfig, ProceduralFieldConfig,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let motion = |shaping: FlowShapingConfig| MotionConfig {
            field_source: MotionFieldSourceConfig::ProceduralRadial,
            procedural: ProceduralFieldConfig {
                scale: 0.3,
                rate: 0.4,
            },
            shaping,
            transplant: FaradayConfig {
                amount: 0.8,
                carrier: MotionCarrierConfig::FirstSourceFrame,
                confidence_threshold: 0.05,
                confidence_softness: 0.1,
                refresh: 0.4,
                decay: 0.9,
                donor: MotionDonorConfig::Selected {
                    saved_position: SavedLayerPosition::new(0).expect("layer 0 exists"),
                },
                ..FaradayConfig::default()
            },
            ..MotionConfig::default()
        };
        let mut shaped = base_patch();
        shaped.layers[0].motion = Some(motion(FlowShapingConfig {
            stretch: 0.6,
            edge_repel: 0.5,
            vector_trash: 0.35,
            trash_block_size: 24.0,
        }));
        render("motion_flow_shaping", shaped);
        let mut unshaped = base_patch();
        unshaped.layers[0].motion = Some(motion(FlowShapingConfig::default()));
        render("motion_flow_shaping_unshaped", unshaped);
        assert_ne!(
            decoded_framemd5("renders/audit_motion_flow_shaping.mp4"),
            decoded_framemd5("renders/audit_motion_flow_shaping_unshaped.mp4"),
            "authored shaping must change the decoded frames"
        );
    }

    /// The B3 feedback-rig labeled export case: a hot feedback loop with the
    /// full rig authored — offset and reflection reshaping the trail's
    /// geometry, hue/gain/saturation working the colour in-loop, the fold
    /// waveshaper and threshold bounding it, deterministic loop noise, and the
    /// engaged servo holding the above-unity gains. Offline consumes the same
    /// temporal shaders and the same frame-plan cadence, so the loop replays
    /// exactly.
    ///
    /// The case renders an `_unrigged` twin — identical patch, identity rig —
    /// and asserts the decoded frames differ: the rig demonstrably reaches the
    /// pixels through the real export path, and an authored identity remains
    /// the exact prior program.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_feedback_rig_pipeline() {
        use crate::effects::params::{FeedbackRigParams, FeedbackShape};
        use crate::patch::TemporalConfig;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let temporal = |rig: FeedbackRigParams| {
            let mut params = crate::effects::params::TemporalParams {
                feedback: 0.9,
                fb_zoom: 1.02,
                fb_rotate: 2.0,
                ..Default::default()
            };
            params.rig = rig;
            TemporalConfig::from_params(&params)
        };
        let mut rigged = base_patch();
        rigged.temporal = Some(temporal(FeedbackRigParams {
            offset_x: 0.01,
            reflect_x: true,
            hue_rotate: 12.0,
            saturation: 1.2,
            gain_r: 1.3,
            gain_g: 1.1,
            gain_b: 1.4,
            chroma_displace: 0.004,
            blur: 0.2,
            sharpen: 0.6,
            shape: FeedbackShape::Fold,
            drive: 1.6,
            pivot: 0.45,
            threshold: 0.08,
            noise: 0.15,
            edge: crate::motion::MotionBoundaryMode::Mirror,
            servo: true,
            ..FeedbackRigParams::default()
        }));
        let rigged_temporal = rigged.temporal.clone();
        render("feedback_rig", rigged);
        let mut unrigged = base_patch();
        unrigged.temporal = Some(temporal(FeedbackRigParams::default()));
        render("feedback_rig_unrigged", unrigged);
        assert_ne!(
            decoded_framemd5("renders/audit_feedback_rig.mp4"),
            decoded_framemd5("renders/audit_feedback_rig_unrigged.mp4"),
            "an authored rig must change the decoded frames"
        );
        // Determinism: the same rigged patch renders byte-identically twice.
        render("feedback_rig_repeat", {
            let mut repeat = base_patch();
            repeat.temporal = rigged_temporal;
            repeat
        });
        assert_eq!(
            decoded_framemd5("renders/audit_feedback_rig.mp4"),
            decoded_framemd5("renders/audit_feedback_rig_repeat.mp4"),
            "the rigged loop must replay deterministically"
        );
    }

    /// B13's labeled export case for the whole small-effects tranche: a
    /// master program authoring contour, flatten, solarize, negative,
    /// colourpass, halftone, moiré, bitcrush, row smear, the multi grid, and
    /// all three master-only optics over a layer authoring its own find-edge
    /// and emboss, rendered through the real export path. The `_plain` twin
    /// is the identical patch with every B13 control at its default and must
    /// decode differently; the `_repeat` render must decode identically, so
    /// the moiré animation's frame-indexed clock is proven deterministic.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_small_effects_pipeline() {
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut authored = base_patch();
        authored.master.contour = 0.6;
        authored.master.contour_bands = 14.0;
        authored.master.flatten = 0.5;
        authored.master.flatten_levels = 6.0;
        authored.master.contour_dither = 0.8;
        authored.master.solarize = 0.35;
        authored.master.negative = 0.4;
        authored.master.negative_mode = 2;
        authored.master.colourpass = 0.5;
        authored.master.colourpass_hue = -40.0;
        authored.master.halftone = 0.45;
        authored.master.halftone_angle = 30.0;
        authored.master.moire = 0.3;
        authored.master.bitcrush = 0.35;
        authored.master.bitcrush_levels = 3.0;
        authored.master.row_smear = 0.4;
        authored.master.multi_grid_x = 2.0;
        authored.master.multi_grid_y = 2.0;
        authored.master.barrel = -0.6;
        authored.master.chroma_aberration = 0.8;
        authored.master.anamorphic_streak = 0.5;
        authored.layers[0].effects.edge_amount = 0.5;
        authored.layers[0].effects.edge_hue = 150.0;
        authored.layers[0].effects.emboss = 0.4;
        let authored_master = authored.master.clone();
        let authored_layer = authored.layers[0].effects.clone();
        render("small_effects", authored);
        render("small_effects_plain", base_patch());
        assert_ne!(
            decoded_framemd5("renders/audit_small_effects.mp4"),
            decoded_framemd5("renders/audit_small_effects_plain.mp4"),
            "the authored small effects must change the decoded frames"
        );
        render("small_effects_repeat", {
            let mut repeat = base_patch();
            repeat.master = authored_master;
            repeat.layers[0].effects = authored_layer;
            repeat
        });
        assert_eq!(
            decoded_framemd5("renders/audit_small_effects.mp4"),
            decoded_framemd5("renders/audit_small_effects_repeat.mp4"),
            "the small effects must replay deterministically"
        );
    }

    /// B12's labeled export case: slit-scan under a non-default time-displace
    /// map, rendered through the real export path. The case renders a `_ramp`
    /// twin — identical patch, default Ramp map with the floor law — and
    /// asserts the decoded frames differ: the map demonstrably reaches the
    /// pixels, and the authored default remains the exact prior program. The
    /// `_repeat` render proves the Sweep map's reference-tick clock replays
    /// deterministically offline.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_time_displace_pipeline() {
        use crate::effects::params::{TemporalParams, TimeDisplaceMap};
        use crate::patch::TemporalConfig;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let temporal = |map: TimeDisplaceMap, interp: bool| {
            TemporalConfig::from_params(&TemporalParams {
                slitscan: 0.85,
                slit_map: map,
                slit_interp: interp,
                ..TemporalParams::default()
            })
        };
        let mut displaced = base_patch();
        displaced.temporal = Some(temporal(TimeDisplaceMap::Sweep, true));
        let displaced_temporal = displaced.temporal.clone();
        render("time_displace", displaced);
        let mut ramp = base_patch();
        ramp.temporal = Some(temporal(TimeDisplaceMap::Ramp, false));
        render("time_displace_ramp", ramp);
        assert_ne!(
            decoded_framemd5("renders/audit_time_displace.mp4"),
            decoded_framemd5("renders/audit_time_displace_ramp.mp4"),
            "an authored time-displace map must change the decoded frames"
        );
        render("time_displace_repeat", {
            let mut repeat = base_patch();
            repeat.temporal = displaced_temporal;
            repeat
        });
        assert_eq!(
            decoded_framemd5("renders/audit_time_displace.mp4"),
            decoded_framemd5("renders/audit_time_displace_repeat.mp4"),
            "the swept displacement must replay deterministically"
        );
    }

    /// Three stacked clips where the upper layer's Residual Counterpoint node
    /// recombines the middle layer's large-scale structure with its own detail
    /// measured against the bottom layer. This is the labeled export case for
    /// the shared node implementation: the offline renderer consumes the same
    /// evaluated plan, the same two reduced block-mean passes, and the same
    /// rack shader, with no export-only recombination path.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_residual_counterpoint_pipeline() {
        use crate::image_routing::LayerImageStage;
        use crate::visual_rack::{
            EdgeTiming, LegacyRackScope, ResidualBlock, ResidualParams, ResidualQuantization,
            SavedImageSource, SavedImageTap, VisualNodeKind, VisualRack,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        let structure_donor = patch.layers[0].clone();
        patch.layers.push(structure_donor);
        let detail_donor = patch.layers[0].clone();
        patch.layers.push(detail_donor);
        // Give the donors visibly different local colour so the recombination
        // is legible: the DC term comes from one and the AC from the other.
        patch.layers[1].effects.hue_shift = 200.0;
        patch.layers[2].effects.brightness = -0.35;

        // Layer 0 is the upper scope. Slot 0 takes its large-scale structure
        // from the middle layer; slot 1 measures its own detail against the
        // bottom layer, so both authored slots are exercised independently.
        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Residual(ResidualParams {
            structure: SavedImageTap {
                source: SavedImageSource::SelectedLayer {
                    layer_position: SavedLayerPosition::new(1).unwrap(),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::CurrentFrame,
            },
            detail: SavedImageTap {
                source: SavedImageSource::SelectedLayer {
                    layer_position: SavedLayerPosition::new(2).unwrap(),
                    stage: LayerImageStage::PostLocalEffects,
                },
                timing: EdgeTiming::CurrentFrame,
            },
            block: ResidualBlock::Sixteen,
            quantization: ResidualQuantization::Medium,
            mix: 0.85,
            detail_gain: 2.0,
            seed: 0x00c0_ffee,
            ..ResidualParams::default()
        }))
        .unwrap();
        patch.layers[0].rack = Some(rack);
        patch.layers[0].effects.hue_shift = 90.0;
        render("residual_counterpoint", patch);
    }

    /// B1's labeled export case: a layer-scope Scan Processor authoring
    /// deflection, a locked oscillator, tilt, and colourise renders through
    /// the real export path — the same evaluated plan and the same
    /// `scan_processor.wgsl`, no export-only path. The `_bypass` twin
    /// carries the node at its exact default on the identical topology and
    /// must decode differently from the authored render (the deflection
    /// demonstrably reaches the pixels), and the `_repeat` render must
    /// decode identically, proving the frame-plan oscillator clock
    /// deterministic. The bypass-equals-no-node byte identity is
    /// deliberately claimed inside one plan variant by the production GPU
    /// fixture, not here: adding any node flips the plan from the frozen
    /// LegacyExact path to Advanced, and the two paths are equivalent, not
    /// byte-equal.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_scan_processor_pipeline() {
        use crate::scan_processor::ScanProcessorParams;
        use crate::visual_rack::{LegacyRackScope, VisualNodeKind, VisualRack};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_scan = |params: ScanProcessorParams| {
            let mut patch = base_patch();
            let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
            rack.push(VisualNodeKind::ScanProcessor(params)).unwrap();
            patch.layers[0].rack = Some(rack);
            patch
        };
        let authored = ScanProcessorParams {
            amount: 0.7,
            osc_amount: 0.3,
            osc_freq: 0.5,
            osc_lock: 1.0,
            tilt_y: 0.25,
            hue: 0.4,
            lines: 240,
            samples_per_line: 128,
            ..ScanProcessorParams::default()
        };
        render("scan_processor", with_scan(authored));
        render(
            "scan_processor_bypass",
            with_scan(ScanProcessorParams::default()),
        );
        assert_ne!(
            decoded_framemd5("renders/audit_scan_processor.mp4"),
            decoded_framemd5("renders/audit_scan_processor_bypass.mp4"),
            "an authored deflection must change the decoded frames"
        );
        render("scan_processor_repeat", with_scan(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_scan_processor.mp4"),
            decoded_framemd5("renders/audit_scan_processor_repeat.mp4"),
            "the drawn raster must replay deterministically"
        );
    }

    /// B6's first labeled export case: a real separable block DCT with a
    /// coarse quantiser renders through the real export path — the same
    /// four-pass `corruption.wgsl` pipeline live uses, no export-only path.
    /// The `_clean` twin at the exact-bypass default must decode
    /// differently; the `_repeat` render must decode identically.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_block_dct_pipeline() {
        use crate::block_dct::BlockDctParams;
        use crate::visual_rack::{LegacyRackScope, VisualNodeKind, VisualRack};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_dct = |params: BlockDctParams| {
            let mut patch = base_patch();
            let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
            rack.push(VisualNodeKind::BlockDct(params)).unwrap();
            patch.layers[0].rack = Some(rack);
            patch
        };
        let authored = BlockDctParams {
            amount: 1.0,
            quantize: 0.6,
            hf_penalty: 0.7,
            chroma_crush: 0.8,
            block: 0.35,
        };
        render("block_dct", with_dct(authored));
        render("block_dct_clean", with_dct(BlockDctParams::default()));
        assert_ne!(
            decoded_framemd5("renders/audit_block_dct.mp4"),
            decoded_framemd5("renders/audit_block_dct_clean.mp4"),
            "an authored quantised transform must change the decoded frames"
        );
        render("block_dct_repeat", with_dct(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_block_dct.mp4"),
            decoded_framemd5("renders/audit_block_dct_repeat.mp4"),
            "the transform must replay deterministically"
        );
    }

    /// B6's second labeled export case: the bounded bright-run stretch.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_pixel_sort_pipeline() {
        use crate::pixel_sort::PixelSortParams;
        use crate::visual_rack::{LegacyRackScope, VisualNodeKind, VisualRack};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_sort = |params: PixelSortParams| {
            let mut patch = base_patch();
            let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
            rack.push(VisualNodeKind::PixelSort(params)).unwrap();
            patch.layers[0].rack = Some(rack);
            patch
        };
        let authored = PixelSortParams {
            amount: 1.0,
            threshold: 0.3,
        };
        render("pixel_sort", with_sort(authored));
        render("pixel_sort_clean", with_sort(PixelSortParams::default()));
        assert_ne!(
            decoded_framemd5("renders/audit_pixel_sort.mp4"),
            decoded_framemd5("renders/audit_pixel_sort_clean.mp4"),
            "authored streaks must change the decoded frames"
        );
        render("pixel_sort_repeat", with_sort(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_pixel_sort.mp4"),
            decoded_framemd5("renders/audit_pixel_sort_repeat.mp4"),
            "the streaks must replay deterministically"
        );
    }

    /// B6's third labeled export case: the reconstruction-filter avalanche
    /// with its retained previous-output cascade. The determinism claim
    /// covers the whole chain — the deterministic lane epochs on frame-plan
    /// time, the node-id-seeded hash lanes, and the tick-clocked history —
    /// because a repeat render must reproduce every inherited error exactly.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_filter_avalanche_pipeline() {
        use crate::filter_avalanche::{AvalancheAxis, AvalancheParams};
        use crate::visual_rack::{LegacyRackScope, VisualNodeKind, VisualRack};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_avalanche = |params: AvalancheParams| {
            let mut patch = base_patch();
            let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
            rack.push(VisualNodeKind::Avalanche(params)).unwrap();
            patch.layers[0].rack = Some(rack);
            patch
        };
        let authored = AvalancheParams {
            amount: 0.9,
            run: 0.7,
            axis: AvalancheAxis::Up,
        };
        render("filter_avalanche", with_avalanche(authored));
        render(
            "filter_avalanche_clean",
            with_avalanche(AvalancheParams::default()),
        );
        assert_ne!(
            decoded_framemd5("renders/audit_filter_avalanche.mp4"),
            decoded_framemd5("renders/audit_filter_avalanche_clean.mp4"),
            "authored corruption must change the decoded frames"
        );
        render("filter_avalanche_repeat", with_avalanche(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_filter_avalanche.mp4"),
            decoded_framemd5("renders/audit_filter_avalanche_repeat.mp4"),
            "the cascade must replay deterministically"
        );
    }

    /// B4's labeled export case: a display stage authoring real fields (Bob
    /// with 3:2 judder), phosphor persistence, and an aperture-grille model
    /// with scanlines, bloom, and sag renders through the real export path —
    /// the same slot-0 seam and the same `display_physics.wgsl` live uses,
    /// no export-only path. The `_flat` twin carries the stage at its
    /// exact-off default and must decode differently; the `_repeat` render
    /// must decode identically, proving the stage's own reference clock and
    /// the phosphor store deterministic frame-indexed offline.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_display_physics_pipeline() {
        use crate::display_physics::{DisplayModel, DisplayPhysicsParams, InterlaceMode};

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_display = |display: DisplayPhysicsParams| {
            let mut patch = base_patch();
            patch.temporal = Some(crate::patch::TemporalConfig {
                display,
                ..crate::patch::TemporalConfig::default()
            });
            patch
        };
        let authored = DisplayPhysicsParams {
            il_amount: 0.8,
            il_mode: InterlaceMode::Bob,
            il_judder: 0.5,
            phosphor: 0.85,
            model: DisplayModel::ApertureGrille,
            scanlines: 0.7,
            mask_strength: 0.5,
            bloom: 0.4,
            sag: 0.3,
            ..DisplayPhysicsParams::default()
        };
        render("display_physics", with_display(authored));
        render(
            "display_physics_flat",
            with_display(DisplayPhysicsParams::default()),
        );
        assert_ne!(
            decoded_framemd5("renders/audit_display_physics.mp4"),
            decoded_framemd5("renders/audit_display_physics_flat.mp4"),
            "an authored display stage must change the decoded frames"
        );
        render("display_physics_repeat", with_display(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_display_physics.mp4"),
            decoded_framemd5("renders/audit_display_physics_repeat.mp4"),
            "the display stage must replay deterministically"
        );
    }

    /// B14's labeled export case: the tape/NTSC horizontal shear latched, so
    /// every slip stays where it happened and accumulates into the bounded
    /// per-line table, through the real export path — the same stage and the
    /// same shader the three live seams encode, no export-only shear. The
    /// `_healed` twin carries the identical four controls with the switch
    /// **off**, so both renders draw the identical fault stream from the
    /// identical hash lanes and differ only in whether the faults heal: the
    /// decoded frames must differ, which is the whole tranche's claim made
    /// visible. The `_repeat` render must decode identically, proving that a
    /// table which is deliberately absent from the patch still regrows
    /// deterministically from the seed and the frame-indexed clock.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_sync_latch_pipeline() {
        use crate::sync_latch::SyncLatchParams;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_sync = |sync: SyncLatchParams| {
            let mut patch = base_patch();
            patch.temporal = Some(crate::patch::TemporalConfig {
                sync,
                ..crate::patch::TemporalConfig::default()
            });
            patch
        };
        let authored = SyncLatchParams {
            amount: 0.9,
            rate: 0.8,
            spread: 0.2,
            bias: 0.6,
            latched: true,
        };
        render("sync_latch", with_sync(authored));
        render(
            "sync_latch_healed",
            with_sync(SyncLatchParams {
                latched: false,
                ..authored
            }),
        );
        assert_ne!(
            decoded_framemd5("renders/audit_sync_latch.mp4"),
            decoded_framemd5("renders/audit_sync_latch_healed.mp4"),
            "latching must change the decoded frames — a fault that heals and              a fault that stays are different programs"
        );
        render("sync_latch_repeat", with_sync(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_sync_latch.mp4"),
            decoded_framemd5("renders/audit_sync_latch_repeat.mp4"),
            "the latched table must regrow deterministically offline"
        );

        // The exact-off default is the prior path: no shear, no pass encoded.
        render("sync_latch_off", with_sync(SyncLatchParams::default()));
        assert_ne!(
            decoded_framemd5("renders/audit_sync_latch.mp4"),
            decoded_framemd5("renders/audit_sync_latch_off.mp4"),
            "an authored shear must differ from the dormant default"
        );
    }

    /// B5's labeled export case: a real mpeg4 encode→break→decode round trip
    /// over the finished programme, through the real export path — the same
    /// engine the live worker owns, synchronously per frame, no export-only
    /// mosh. The `_clean` twin carries the stage at its exact-bypass default
    /// (amount zero: no encoder alive, no pixels touched) and must decode
    /// differently; the `_repeat` render must decode identically, proving
    /// the fault clock and the codec pair deterministic per host with
    /// `threads = 1`. Cross-machine bit-identity is deliberately not
    /// claimed — the sidecar records the encoder identity instead.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_codec_mosh_pipeline() {
        use crate::codec_mosh::CodecMoshParams;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_mosh = |mosh: CodecMoshParams| {
            let mut patch = base_patch();
            patch.layers[0].mosh_send = 0.35;
            patch.temporal = Some(crate::patch::TemporalConfig {
                mosh,
                ..crate::patch::TemporalConfig::default()
            });
            patch
        };
        let authored = CodecMoshParams {
            amount: 0.9,
            key_removal: 0.95,
            hold: 0.6,
            drop: 0.15,
            shuffle: 0.4,
            rate: 1.0,
            bitrate_starve: 0.6,
            resync: 0.2,
            wipe: 0.8,
            smear: 0.65,
            trail: 0.9,
            recycle: false,
        };
        render("codec_mosh", with_mosh(authored));
        render("codec_mosh_clean", with_mosh(CodecMoshParams::default()));
        assert_ne!(
            decoded_framemd5("renders/audit_codec_mosh.mp4"),
            decoded_framemd5("renders/audit_codec_mosh_clean.mp4"),
            "an authored mosh must change the decoded frames"
        );
        render("codec_mosh_repeat", with_mosh(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_codec_mosh.mp4"),
            decoded_framemd5("renders/audit_codec_mosh_repeat.mp4"),
            "the round trip must replay deterministically on one host"
        );

        // The sidecar records the recipe and the encoder identity — the
        // per-host honesty record — only for the moshed render.
        let sidecar: serde_json::Value = serde_json::from_slice(
            &std::fs::read("renders/audit_codec_mosh.mp4.motion.json").unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar["schema_version"], 7);
        let mosh = &sidecar["codec_mosh"];
        assert!(
            mosh["encoder"]
                .as_str()
                .is_some_and(|encoder| encoder.starts_with("mpeg4/avcodec-")),
            "{mosh}"
        );
        let close = |value: &serde_json::Value, expected: f32| {
            let observed = value.as_f64().expect("sidecar scalar") as f32;
            assert!(
                (observed - expected).abs() < 1.0e-6,
                "expected {expected}, observed {observed} in {mosh}"
            );
        };
        close(&mosh["wipe"], authored.wipe);
        close(&mosh["smear"], authored.smear);
        close(&mosh["trail"], authored.trail);
        assert!(mosh["observed"]["accepted_frames"]
            .as_u64()
            .is_some_and(|frames| frames > 0));
        for (field, expected) in [
            ("wipe", authored.wipe),
            ("smear", authored.smear),
            ("trail", authored.trail),
        ] {
            close(&mosh["observed"]["min"][field], expected);
            close(&mosh["observed"]["max"][field], expected);
        }
        assert_eq!(mosh["observed"]["recycle_false_seen"], true);
        let sends = mosh["layer_sends"].as_array().expect("layer-send evidence");
        assert_eq!(sends.len(), 1);
        close(&sends[0]["authored"], 0.35);
        close(&sends[0]["observed_min"], 0.35);
        close(&sends[0]["observed_max"], 0.35);
        assert_eq!(sends[0]["entered_codec_mosh"], true);
        assert_eq!(mosh["layer_sends_truncated"], false);
        let clean: serde_json::Value = serde_json::from_slice(
            &std::fs::read("renders/audit_codec_mosh_clean.mp4.motion.json").unwrap(),
        )
        .unwrap();
        assert!(clean.get("codec_mosh").is_none());
    }

    /// B16's labeled export case: the finished programme routed back into the
    /// rig as an ordinary Displace donor — every frame warped by the previous
    /// frame's pre-blackout opaque audience image, through the real export
    /// path, the same acceptance-decision copy live performs, no export-only
    /// tap. The `_untapped` twin carries the identical node at its exact
    /// bypass (zero gains), so both renders are the same Advanced plan family
    /// and must decode differently — the re-entry loop demonstrably reaches
    /// the pixels. The `_repeat` render must decode identically, proving the
    /// whole two-frame feedback chain (decode → composite → opaque resolve →
    /// tap publish → next frame's donor) deterministic frame-indexed offline.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_program_reentry_pipeline() {
        use crate::visual_rack::{
            DisplaceBoundary, DisplaceParams, EdgeTiming, LegacyRackScope, SavedImageSource,
            SavedImageTap, VisualNodeKind, VisualRack,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_tap = |amount: f32| {
            let mut patch = base_patch();
            let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
            rack.push(VisualNodeKind::Displace(DisplaceParams {
                tap: SavedImageTap {
                    source: SavedImageSource::ProgramTap,
                    timing: EdgeTiming::CurrentFrame,
                },
                amount_x: amount,
                amount_y: amount,
                boundary: DisplaceBoundary::Mirror,
            }))
            .unwrap();
            patch.layers[0].rack = Some(rack);
            patch
        };
        render("program_reentry", with_tap(0.4));
        render("program_reentry_untapped", with_tap(0.0));
        assert_ne!(
            decoded_framemd5("renders/audit_program_reentry.mp4"),
            decoded_framemd5("renders/audit_program_reentry_untapped.mp4"),
            "a routed programme tap must change the decoded frames"
        );
        render("program_reentry_repeat", with_tap(0.4));
        assert_eq!(
            decoded_framemd5("renders/audit_program_reentry.mp4"),
            decoded_framemd5("renders/audit_program_reentry_repeat.mp4"),
            "the re-entry loop must replay deterministically"
        );
    }

    /// B7's pattern labeled export case, and the self-containment proof: the
    /// patch's only layer is a pattern synth — no file, no content
    /// reference, no library — and the export must render it through the
    /// same GPU pass live uses. The `_default` twin holds the identical
    /// topology with default pattern values and must decode differently; the
    /// `_repeat` render must decode identically, proving the frame-plan
    /// clock deterministic offline.
    #[test]
    #[ignore = "requires a GPU and ffmpeg on PATH"]
    fn render_pattern_synth_pipeline() {
        std::fs::create_dir_all("renders").ok();
        let with_pattern = |authored: bool| {
            let mut patch = base_patch();
            let mut config = crate::patch::PatternSynthConfig::default();
            if authored {
                config.shape = crate::patch::PatternShapeConfig::Tunnel;
                config.wave = crate::patch::PatternWaveConfig::Saw;
                config.color_mode = crate::patch::PatternColorModeConfig::HsvSweep;
                config.cross_mod = 0.6;
                config.wavefold = 0.5;
                config.comparator = 0.4;
                config.rate = 0.4;
                config.warp = 0.3;
            }
            patch.layers[0].filename = "Pattern Synth".to_string();
            patch.layers[0].source_path = crate::layers::PATTERN_SOURCE_PATH.to_string();
            patch.layers[0].clip_slots = crate::performance::ClipSlots::singleton(
                crate::performance::ClipSlotConfig::from_legacy(
                    "Pattern Synth".to_string(),
                    crate::layers::PATTERN_SOURCE_PATH.to_string(),
                    1.0,
                    30.0,
                ),
            );
            patch.layers[0].pattern = Some(config);
            patch
        };
        render("pattern_synth", with_pattern(true));
        render("pattern_synth_default", with_pattern(false));
        assert_ne!(
            decoded_framemd5("renders/audit_pattern_synth.mp4"),
            decoded_framemd5("renders/audit_pattern_synth_default.mp4"),
            "authored pattern values must change the decoded frames"
        );
        render("pattern_synth_repeat", with_pattern(true));
        assert_eq!(
            decoded_framemd5("renders/audit_pattern_synth.mp4"),
            decoded_framemd5("renders/audit_pattern_synth_repeat.mp4"),
            "the pattern clock must replay deterministically"
        );
    }

    /// B7's text-page labeled export case: the patch's only layer is a
    /// typeset page rastered from its own authored values through the same
    /// CPU raster live uses — no file, no font on the host, no library. The
    /// `_alt` twin authors a different body and shape fan and must decode
    /// differently; the `_repeat` render must decode identically.
    #[test]
    #[ignore = "requires a GPU and ffmpeg on PATH"]
    fn render_text_page_pipeline() {
        std::fs::create_dir_all("renders").ok();
        let with_page = |alt: bool| {
            let mut patch = base_patch();
            let mut config = crate::patch::TextPageConfig::default();
            if alt {
                config.body = "OTHER\nPAGE".to_string();
                config.font = crate::patch::TextPageFontConfig::Sans;
                config.shape = crate::patch::TextPageShapeConfig::Rings;
                config.rot_degrees = 20.0;
                config.outline = 4.0;
            } else {
                config.body = "COLLIDE\nO SCOPE".to_string();
                config.shape = crate::patch::TextPageShapeConfig::Circle;
                config.shape_size = 0.4;
            }
            patch.layers[0].filename = "Text Page".to_string();
            patch.layers[0].source_path = crate::layers::TEXT_PAGE_SOURCE_PATH.to_string();
            patch.layers[0].clip_slots = crate::performance::ClipSlots::singleton(
                crate::performance::ClipSlotConfig::from_legacy(
                    "Text Page".to_string(),
                    crate::layers::TEXT_PAGE_SOURCE_PATH.to_string(),
                    1.0,
                    30.0,
                ),
            );
            patch.layers[0].text_page = Some(config);
            patch
        };
        render("text_page", with_page(false));
        render("text_page_alt", with_page(true));
        assert_ne!(
            decoded_framemd5("renders/audit_text_page.mp4"),
            decoded_framemd5("renders/audit_text_page_alt.mp4"),
            "a different authored page must decode differently"
        );
        render("text_page_repeat", with_page(false));
        assert_eq!(
            decoded_framemd5("renders/audit_text_page.mp4"),
            decoded_framemd5("renders/audit_text_page_repeat.mp4"),
            "the page raster must replay deterministically"
        );
    }

    /// B8's bus labeled export case: two layers on the A and B lanes meet
    /// under an authored wipe, blend family, border rule, dirty-mixer fault
    /// stage, and bus melt — through the real export path, the same `fs_bus`
    /// live uses, no export-only mixer. The `_plain` twin carries the same
    /// A/B topology with the mixer at its exact-legacy default and must
    /// decode differently; the `_repeat` render must decode identically,
    /// proving the dirt event clock and the melt store deterministic
    /// frame-indexed offline.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_bus_mixing_boundary_pipeline() {
        use crate::composition::{BusAssignment, CompositionTree, RootItem};
        use crate::mixing_boundary::{
            BackColor, BusMixParams, BusMixerState, DirtParams, MeltParams, WipePattern,
        };
        use crate::performance::SavedLayerPosition;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_mixer = |mixer: BusMixerState| {
            let mut patch = base_patch();
            let mut top = patch.layers[0].clone();
            top.effects.hue_shift = 140.0;
            patch.layers.push(top);
            let tree = CompositionTree::try_from_parts(
                Vec::new(),
                vec![
                    RootItem::Layer {
                        layer: SavedLayerPosition::new(0).expect("layer 0 exists"),
                        bus: BusAssignment::A,
                    },
                    RootItem::Layer {
                        layer: SavedLayerPosition::new(1).expect("layer 1 exists"),
                        bus: BusAssignment::B,
                    },
                ],
                None,
                0.5,
            )
            .expect("A/B composition")
            .with_mixer(mixer);
            patch.composition = Some(tree);
            patch
        };
        let authored = BusMixerState {
            mix: BusMixParams {
                pattern: WipePattern::Circle,
                soft: 0.2,
                border: 0.5,
                border_color: BackColor::Cyan,
                blend: crate::layers::BlendMode::PinLight,
                ..BusMixParams::default()
            },
            dirt: DirtParams {
                dirt: 0.8,
                rate: 0.6,
                ..DirtParams::default()
            },
            melt: MeltParams {
                melt: 1.2,
                hold: 1.2,
                ..MeltParams::default()
            },
        };
        render("bus_mixing_boundary", with_mixer(authored));
        render(
            "bus_mixing_boundary_plain",
            with_mixer(BusMixerState::default()),
        );
        assert_ne!(
            decoded_framemd5("renders/audit_bus_mixing_boundary.mp4"),
            decoded_framemd5("renders/audit_bus_mixing_boundary_plain.mp4"),
            "an authored bus mixer must change the decoded frames"
        );
        render("bus_mixing_boundary_repeat", with_mixer(authored));
        assert_eq!(
            decoded_framemd5("renders/audit_bus_mixing_boundary.mp4"),
            decoded_framemd5("renders/audit_bus_mixing_boundary_repeat.mp4"),
            "the bus mixer must replay deterministically"
        );
    }

    /// B8's master labeled export case: the program's own coverage boundary
    /// (a master luminance key) melts through the slot-0 melting-edge stage
    /// while the key signal carries its border and shadow dressing — the
    /// same seam and shaders live uses, no export-only path. The `_dry`
    /// twin keeps the key but zeroes the melt and the dressing and must
    /// decode differently; the `_repeat` render must decode identically,
    /// proving the stage's reference-tick store deterministic offline.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_melting_edge_and_key_dressing_pipeline() {
        use crate::mixing_boundary::MeltParams;

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let with_melt = |melt: MeltParams, border: f32, shadow: f32| {
            let mut patch = base_patch();
            patch.master.key_mode = 1;
            patch.master.key_threshold = 0.45;
            patch.master.key_softness = 0.05;
            patch.master.key_border = border;
            patch.master.key_border_color = 2;
            patch.master.key_shadow = shadow;
            patch.temporal = Some(crate::patch::TemporalConfig {
                melt,
                ..crate::patch::TemporalConfig::default()
            });
            patch
        };
        let authored = MeltParams {
            melt: 1.5,
            width: 1.0,
            hold: 1.2,
            chroma: 0.8,
            creep: 0.35,
            ..MeltParams::default()
        };
        render("melting_edge_key_dressing", with_melt(authored, 0.8, 0.9));
        render(
            "melting_edge_key_dressing_dry",
            with_melt(MeltParams::default(), 0.0, 0.0),
        );
        assert_ne!(
            decoded_framemd5("renders/audit_melting_edge_key_dressing.mp4"),
            decoded_framemd5("renders/audit_melting_edge_key_dressing_dry.mp4"),
            "an authored melt and dressing must change the decoded frames"
        );
        render(
            "melting_edge_key_dressing_repeat",
            with_melt(authored, 0.8, 0.9),
        );
        assert_eq!(
            decoded_framemd5("renders/audit_melting_edge_key_dressing.mp4"),
            decoded_framemd5("renders/audit_melting_edge_key_dressing_repeat.mp4"),
            "the melting edge must replay deterministically"
        );
    }

    /// S3b's labeled export case. It is the one that renders a job carrying a
    /// recorded gesture performance end to end: the track's canonical checksum
    /// is verified before the first frame, the events replay on the offline 30 Hz
    /// frame-index timeline, and the recording is published beside the video as
    /// the frozen six-field sidecar. The authored canvas controls travel in the
    /// patch, so the case also proves the pre-render admission of the offline
    /// canvas against the frozen resource table.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_gesture_field_etching_pipeline() {
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        patch.gesture_canvas = Some(crate::patch::GestureCanvasConfig {
            radius: 0.3,
            strength: 0.85,
            retention: 0.92,
        });
        patch.master.hue_shift = 120.0;

        let track = super::tests::recorded_gesture_fixture();
        let document = crate::gesture::GestureTrackDocument::capture(&track);
        let output = render_with_gesture("gesture_field_etching", patch, Some(document.clone()));

        let sidecar_path = gesture_sidecar_path(&output);
        let bytes = std::fs::read(&sidecar_path).expect("the labeled case publishes its sidecar");
        let restored = crate::gesture::GestureTrackDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored, document);
        assert_eq!(restored.decode().unwrap(), track);
    }

    /// The labeled case this stage exists for: a recorded gesture performance
    /// routed as the donor of a Displace node over a live portrait, rendered
    /// end to end through the real export path.
    ///
    /// It follows `render_displace_two_input_pipeline` exactly — the same node,
    /// the same authored gains, the same offline executor — and changes only
    /// the one thing this stage added: the donor route is
    /// `SavedImageSource::GestureCanvas` rather than another layer. The
    /// previous stage could not write this case because nothing sampled the
    /// etched field; the canvas is now built, etched on the offline 30 Hz
    /// timeline, and presented as an ordinary image tap, so it can.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_gesture_canvas_displace_donor_pipeline() {
        use crate::visual_rack::{
            DisplaceBoundary, DisplaceParams, EdgeTiming, LegacyRackScope, SavedImageSource,
            SavedImageTap, VisualNodeKind, VisualRack,
        };

        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let mut patch = base_patch();
        // A wide, strong, long-retaining etch, so the recorded stroke is
        // legible in the rendered frames rather than a sub-pixel nudge.
        patch.gesture_canvas = Some(crate::patch::GestureCanvasConfig {
            radius: 0.45,
            strength: 1.0,
            retention: 0.995,
        });

        let mut rack = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        rack.push(VisualNodeKind::Displace(DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::GestureCanvas,
                timing: EdgeTiming::CurrentFrame,
            },
            amount_x: 0.4,
            amount_y: 0.4,
            boundary: DisplaceBoundary::Mirror,
        }))
        .unwrap();
        patch.layers[0].rack = Some(rack);

        let track = super::tests::recorded_gesture_fixture();
        let document = crate::gesture::GestureTrackDocument::capture(&track);
        let output = render_with_gesture(
            "gesture_canvas_displace_donor",
            patch,
            Some(document.clone()),
        );

        // The recording is still published beside its render, so the labeled
        // artifact carries the exact performance that warped it.
        let sidecar_path = gesture_sidecar_path(&output);
        let bytes = std::fs::read(&sidecar_path).expect("the labeled case publishes its sidecar");
        let restored = crate::gesture::GestureTrackDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored.decode().unwrap(), track);
    }

    #[test]
    #[ignore = "renders the full effects audit matrix; run explicitly"]
    fn render_effects_matrix() {
        std::fs::create_dir_all("renders").ok();
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first (any short clip)"
        );

        render("baseline", base_patch());

        let mut p = base_patch();
        p.master.brightness = 0.4;
        render("brightness", p);

        let mut p = base_patch();
        p.master.invert = true;
        render("invert", p);

        let mut p = base_patch();
        p.master.posterize = 2.0;
        render("posterize", p);

        let mut p = base_patch();
        p.master.pixelate = 32.0;
        render("pixelate", p);

        let mut p = base_patch();
        p.master.vignette = 1.4;
        render("vignette", p);

        let mut p = base_patch();
        p.master.hue_shift = 120.0;
        render("hue", p);

        let mut p = base_patch();
        p.master.contrast = 0.8;
        render("contrast", p);

        let mut p = base_patch();
        p.master.saturation = -1.0;
        render("saturation", p);

        let mut p = base_patch();
        p.master.grain_intensity = 0.28;
        p.master.color_grain = true;
        render("grain", p);

        let mut p = base_patch();
        p.master.rgb_split = 25.0;
        render("rgbsplit", p);

        let mut p = base_patch();
        p.master.color_drift = 0.02;
        p.master.breathe_scale = 0.05;
        render("drift_breathe", p);

        let mut p = base_patch();
        p.master.cellular_amount = 0.85;
        p.master.cellular_scale = 12.0;
        p.master.cellular_warp = 0.6;
        p.master.cellular_speed = 0.75;
        render("cellular", p);

        let mut p = base_patch();
        p.master.random_seed = 0x5348_4946;
        p.master.shift_amount = 0.8;
        p.master.shift_block_size = 24.0;
        p.master.shift_density = 0.75;
        p.master.shift_speed = 6.0;
        render("shift", p);

        let mut p = base_patch();
        p.layers[0].effects.key_mode = 1;
        p.layers[0].effects.key_threshold = 0.55;
        render("key", p);

        let mut p = base_patch();
        p.temporal.as_mut().unwrap().feedback = 0.92;
        p.temporal.as_mut().unwrap().fb_zoom = 1.05;
        render("temporal_fb", p);

        let mut p = base_patch();
        p.temporal.as_mut().unwrap().slitscan = 0.85;
        render("slitscan", p);

        let mut p = base_patch();
        {
            let n = p.ntsc.as_mut().unwrap();
            n.enabled = true;
            n.snow_intensity = 0.9;
            n.tracking_noise_enabled = true;
            n.tracking_noise_snow = 0.7;
        }
        render("ntsc", p);

        let mut p = base_patch();
        p.layers[0].opacity = 0.3;
        render("opacity", p);
    }

    // ===== B9 performance recorder =====

    fn render_with_take(
        label: &str,
        patch: PatchState,
        performance_take: Option<crate::performance_track::PerformanceTakeDocument>,
    ) -> String {
        let config = ExportConfig {
            width: 320,
            height: 180,
            fps: 24,
            duration_secs: 1.0,
            output_path: format!("renders/audit_{label}.mp4"),
            audio_path: None,
            audio_path_hint: None,
            layer_source_hints: Vec::new(),
            analysis_audio_path_hint: None,
            ntsc_quality: NtscExportQuality::LiveParity,
            shutter_samples: ExportShutterSamples::Authored,
            media_safety_policy: MediaSafetyPolicy::default(),
            temporal_event_track: crate::temporal::TemporalEventTrack::default(),
            gesture_track: None,
            performance_take,
        };
        let output_path = config.output_path.clone();
        let job = ExportJob::start(patch, config, "videos");
        while !job.is_done() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let err = job.progress.error.lock().unwrap().clone();
        assert!(err.is_empty(), "{label}: export failed: {err}");
        output_path
    }

    fn performance_take_fixture() -> crate::performance_track::PerformanceTakeDocument {
        use crate::performance_track::{
            PerformanceControl as Control, PerformanceRawValue as Raw, PerformanceTake,
            PerformanceTakeDocument, PerformanceValueLaw as Law,
        };
        let mut take = PerformanceTake::default();
        take.record_accepted(
            0,
            Control::Master {
                param: "brightness".to_string(),
            },
            Law::Unit {
                min: -1.0,
                max: 1.0,
            },
            &Raw::Continuous(0.6),
        )
        .unwrap();
        take.record_accepted(
            6,
            Control::Master {
                param: "hue_shift".to_string(),
            },
            Law::Unit {
                min: -180.0,
                max: 180.0,
            },
            &Raw::Continuous(120.0),
        )
        .unwrap();
        take.record_accepted(
            12,
            Control::Temporal {
                param: "feedback".to_string(),
            },
            Law::Unit {
                min: 0.0,
                max: 0.95,
            },
            &Raw::Continuous(0.8),
        )
        .unwrap();
        take.record_accepted(
            18,
            Control::LayerEffect {
                layer: 0,
                param: "pixelate".to_string(),
            },
            Law::Unit {
                min: 1.0,
                max: 32.0,
            },
            &Raw::Continuous(12.0),
        )
        .unwrap();
        take.finalize(20);
        PerformanceTakeDocument::capture(&take)
    }

    #[test]
    fn export_admission_finds_temporal_bypass_authored_only_by_a_later_take_event() {
        use crate::performance_track::{
            PerformanceControl as Control, PerformanceRawValue as Raw, PerformanceTake,
            PerformanceValueLaw as Law,
        };
        let mut take = PerformanceTake::default();
        take.record_accepted(
            0,
            Control::LayerParam {
                layer: 0,
                param: "bypass_temporal_fx".to_string(),
            },
            Law::Toggle,
            &Raw::Toggle(false),
        )
        .unwrap();
        take.record_accepted(
            240,
            Control::LayerParam {
                layer: 0,
                param: "bypass_temporal_fx".to_string(),
            },
            Law::Toggle,
            &Raw::Toggle(true),
        )
        .unwrap();
        take.finalize(300);

        assert!(performance_take_authors_temporal_bypass(&take));

        let mut unrelated = PerformanceTake::default();
        unrelated
            .record_accepted(
                240,
                Control::LayerParam {
                    layer: 0,
                    param: "bypass_master_fx".to_string(),
                },
                Law::Toggle,
                &Raw::Toggle(true),
            )
            .unwrap();
        assert!(!performance_take_authors_temporal_bypass(&unrelated));
    }

    /// The performance sidecar publishes on the gesture sidecar's exact
    /// no-replace law, is cleanup-coupled to the video, and is retired at the
    /// output claim so a re-export can publish again.
    #[test]
    fn the_performance_sidecar_publishes_no_replace_and_retires_at_the_output_claim() {
        let unique = format!(
            "cos-performance-sidecar-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(&unique);
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("artifact.mp4");
        std::fs::write(&output, b"accepted-video").unwrap();
        let output_string = output.to_string_lossy().into_owned();

        let document = performance_take_fixture();
        write_performance_sidecar_noreplace(&output_string, &document).unwrap();
        let sidecar_path = performance_sidecar_path(&output_string);
        assert_eq!(
            sidecar_path.file_name().unwrap(),
            "artifact.mp4.performance.json"
        );
        let bytes = std::fs::read(&sidecar_path).unwrap();
        assert!(bytes.len() <= crate::performance_track::MAX_PERFORMANCE_SERIALIZED_BYTES);
        let restored =
            crate::performance_track::PerformanceTakeDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored, document);
        restored.decode().unwrap();

        // No-replace: a second publication refuses rather than overwriting.
        let second = write_performance_sidecar_noreplace(&output_string, &document).unwrap_err();
        assert!(second.contains("refusing to overwrite"), "{second}");
        assert_eq!(std::fs::read(&sidecar_path).unwrap(), bytes);
        // No staging residue survives a refused commit.
        assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));

        // Cleanup coupling: the terminal cleanup names the performance
        // sidecar beside the motion and gesture ones, so a cancelled or
        // failed render never leaves a take receipt beside a deleted video.
        let source = include_str!("render_export.rs");
        let cleanup = source
            .find("fn remove_started_output(")
            .expect("terminal cleanup");
        assert!(
            source[cleanup..cleanup + 2_500].contains("performance_sidecar_path(path)"),
            "terminal cleanup must retire the performance sidecar with the video"
        );
        remove_partial_path(&sidecar_path, "performance sidecar").unwrap();
        assert!(!sidecar_path.exists());
        // Retiring a receipt that was never written is not an error.
        remove_partial_path(&sidecar_path, "performance sidecar").unwrap();

        // The retirement lives at the output claim, beside the gesture one.
        let claim = source
            .split_once(".output_started\n                .store(true, Ordering::Release);")
            .expect("the encoder supervisor claims the output name")
            .1;
        assert!(
            claim[..800].contains("performance_sidecar_path(&output_path)"),
            "a re-export must retire its own stale performance receipt at the claim"
        );
    }

    /// The offline applier mutates the export bases through the shared
    /// appliers, and an address the job cannot resolve is a safe no-op.
    #[test]
    fn export_replay_applies_take_events_through_the_shared_appliers() {
        use crate::performance_track::{PerformanceControl as Control, PerformanceRawValue as Raw};
        let mut master_effects = EffectUniforms::default();
        let mut master_transform = SpatialTransform::default();
        let mut master_motion = crate::motion::MotionParams::default();
        let mut layer_motion: Vec<crate::motion::MotionParams> = Vec::new();
        let mut ntsc = crate::ntsc::NtscParams::default();
        let mut temporal = crate::effects::params::TemporalParams::default();
        let mut gesture_canvas = crate::gesture_canvas::GestureCanvasParams::default();
        let mut graph = resolve_export_creative_graph(&base_patch()).expect("graph");
        let mut layers: Vec<ExportLayer> = Vec::new();
        let mut morph: Option<crate::morph::Morph> = None;

        let mut apply = |control: Control, raw: Raw| {
            apply_export_performance_event(
                &control,
                &raw,
                &mut master_effects,
                &mut master_transform,
                &mut master_motion,
                &mut layer_motion,
                &mut ntsc,
                &mut temporal,
                &mut gesture_canvas,
                &mut graph,
                &mut layers,
                &mut morph,
            );
        };

        apply(
            Control::Master {
                param: "brightness".to_string(),
            },
            Raw::Continuous(0.6),
        );
        apply(
            Control::MasterTransform {
                param: "position_x".to_string(),
            },
            Raw::Continuous(0.25),
        );
        apply(
            Control::Temporal {
                param: "feedback".to_string(),
            },
            Raw::Continuous(0.8),
        );
        apply(
            Control::Ntsc {
                param: "snow_intensity".to_string(),
            },
            Raw::Continuous(0.5),
        );
        apply(
            Control::MotionMaster {
                param: "shutter_angle".to_string(),
            },
            Raw::Continuous(270.0),
        );
        apply(Control::BusCrossfade, Raw::Continuous(0.8));
        apply(
            Control::BusMix {
                param: "wipe_pattern".to_string(),
            },
            Raw::Token("circle".to_string()),
        );
        apply(
            Control::GestureCanvas {
                param: "radius".to_string(),
            },
            Raw::Continuous(0.9),
        );
        // A layer address with no layer at the position, and a morph edit with
        // no morph state, are safe no-ops rather than panics or retargets.
        apply(
            Control::LayerEffect {
                layer: 5,
                param: "pixelate".to_string(),
            },
            Raw::Continuous(9.0),
        );
        apply(Control::MorphPosition, Raw::Continuous(0.5));

        assert!((master_effects.brightness - 0.6).abs() < 1.0e-6);
        assert!((master_transform.position[0] - 0.25).abs() < 1.0e-6);
        assert!((temporal.feedback - 0.8).abs() < 1.0e-6);
        assert!((ntsc.snow_intensity - 0.5).abs() < 1.0e-6);
        assert!((master_motion.shutter.angle_degrees - 270.0).abs() < 1.0e-6);
        assert!((graph.composition.bus_crossfade() - 0.8).abs() < 1.0e-6);
        assert_eq!(
            graph.composition.mixer().mix.pattern,
            crate::mixing_boundary::WipePattern::Circle
        );
        assert!((gesture_canvas.radius - 0.9).abs() < 1.0e-6);
        assert_eq!(
            export_replayed_layer_mosh_send(&Raw::Continuous(0.375)),
            Some(0.375)
        );
        assert_eq!(
            export_replayed_layer_mosh_send(&Raw::Continuous(-4.0)),
            Some(0.0)
        );
        assert_eq!(
            export_replayed_layer_mosh_send(&Raw::Continuous(f32::NAN)),
            Some(1.0)
        );
        assert_eq!(
            export_replayed_layer_mosh_send(&Raw::Token("0.5".to_string())),
            None
        );
    }

    /// B10's labeled export case: the new sources drive the picture through
    /// the one matrix law — an envelope fired every beat into brightness,
    /// deterministic chaos into hue, drift into position, and the
    /// video-reactive brightness (computed offline from the job's own frame
    /// bytes) into contrast. The `_unrouted` twin is the identical patch with
    /// no routes and must decode differently — the sources demonstrably reach
    /// the pixels — and the `_repeat` render must decode identically, which
    /// is the whole deterministic-replay claim BENDR never made: seeded,
    /// frame-indexed chaos and video analysis replay the same trajectory.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_mod_sources_pipeline() {
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let modulation = {
            let mut matrix = crate::modulation::ModMatrix::new();
            matrix.envelopes[0].trigger = crate::modulation::EnvelopeTrigger::Beat(1);
            matrix.envelopes[0].attack = 0.02;
            matrix.envelopes[0].decay = 0.3;
            matrix.generator_seed = 11;
            matrix.routings = vec![
                crate::modulation::Routing::new(
                    crate::modulation::ModSource::Envelope(0),
                    "brightness",
                    0.8,
                ),
                crate::modulation::Routing::new(
                    crate::modulation::ModSource::Chaos,
                    "hue_shift",
                    1.0,
                ),
                crate::modulation::Routing::new(
                    crate::modulation::ModSource::Drift,
                    "position_x",
                    0.6,
                ),
                crate::modulation::Routing::new(
                    crate::modulation::ModSource::VideoBrightness,
                    "contrast",
                    0.9,
                ),
            ];
            crate::patch::ModConfig::from_matrix(&matrix)
        };
        let mut routed = base_patch();
        routed.modulation = Some(modulation.clone());
        render("mod_sources", routed);

        let mut unrouted = base_patch();
        let mut silent = modulation.clone();
        silent.routings = Vec::new();
        unrouted.modulation = Some(silent);
        render("mod_sources_unrouted", unrouted);
        assert_ne!(
            decoded_framemd5("renders/audit_mod_sources.mp4"),
            decoded_framemd5("renders/audit_mod_sources_unrouted.mp4"),
            "the routed sources must change the decoded frames"
        );

        let mut repeat = base_patch();
        repeat.modulation = Some(modulation);
        render("mod_sources_repeat", repeat);
        assert_eq!(
            decoded_framemd5("renders/audit_mod_sources.mp4"),
            decoded_framemd5("renders/audit_mod_sources_repeat.mp4"),
            "seeded generators and offline video analysis must replay identically"
        );
    }

    /// B9's labeled export case: a recorded take rides beside the patch and
    /// replays by reference tick through the shared appliers. The `_untaken`
    /// twin renders the identical patch with no take and must decode
    /// differently — the take demonstrably reaches the pixels — and the
    /// `_repeat` render must decode identically, which is the record/replay
    /// determinism claim: same take, same patch, same frames.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_performance_recorder_pipeline() {
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        std::fs::create_dir_all("renders").ok();

        let document = performance_take_fixture();
        let output = render_with_take("performance_recorder", base_patch(), Some(document.clone()));
        // The job published the take beside the render, byte-verifiable.
        let sidecar = performance_sidecar_path(&output);
        let published = crate::performance_track::PerformanceTakeDocument::from_json_bytes(
            &std::fs::read(&sidecar).unwrap(),
        )
        .unwrap();
        assert_eq!(published, document);

        render_with_take("performance_recorder_untaken", base_patch(), None);
        assert_ne!(
            decoded_framemd5("renders/audit_performance_recorder.mp4"),
            decoded_framemd5("renders/audit_performance_recorder_untaken.mp4"),
            "a replayed take must change the decoded frames"
        );
        assert!(
            !performance_sidecar_path("renders/audit_performance_recorder_untaken.mp4").exists(),
            "no take, no sidecar"
        );

        render_with_take("performance_recorder_repeat", base_patch(), Some(document));
        assert_eq!(
            decoded_framemd5("renders/audit_performance_recorder.mp4"),
            decoded_framemd5("renders/audit_performance_recorder_repeat.mp4"),
            "the same take against the same patch must replay identically"
        );
    }
}
