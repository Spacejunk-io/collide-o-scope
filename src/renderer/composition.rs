//! Shared live/export GPU boundary for evaluated creative compositions.
//!
//! The immutable [`EvaluatedCompositionPlan`] is the only authored contract
//! accepted here. Exact legacy plans are explicitly delegated to the frozen
//! renderer/export path; advanced plans are prepared transactionally and keep
//! every GPU handle inside this executor so a warmed encode creates nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::composition::{BusAssignment, RuntimeRootItem};
use crate::evaluated_frame::evaluated_composition::{
    AdvancedCompositionPlan, CompositePrefix, EvaluatedCompositionPlan, EvaluatedCorruptionKind,
    EvaluatedCorruptionPlan, EvaluatedRefreshGardenSignalPlan, EvaluatedScanProcessorPlan,
    EvaluatedScopeExecution, EvaluatedScopeStep, EvaluatedStudyPlan, EvaluatedSymmetryFieldPlan,
    ImageTapConsumer, LegacyCanonicalApplication, MotionFieldAttachment, PlannedImageSource,
    PlannedImageTap,
};
use crate::image_routing::{LayerImageStage, StableLayerId};
use crate::layers::BlendMode;
use crate::motion::MotionFieldOrigin;
use crate::program_recorder::CaptureTarget;
use crate::renderer::composition_host::{
    BusFrameParams, CompositionHost, CompositionHostError, HostCapacities, HostCompositeInputs,
    HostCompositeUniforms, HostEffectSource, HostFrameTiming, HostMatteInputs,
    HostRoutedGardenInput, HostSurface, HostTemporalInput, HostTextureInputs, HostUniformSlot,
};
use crate::renderer::compositor::{MatteChannelCode, MatteCompositeUniforms, ResolvedMatteParams};
use crate::renderer::corruption::{
    CorruptionGpuExecutor, CorruptionGpuUniforms, CorruptionPassPipeline,
};
use crate::renderer::motion::{
    MotionFieldReadParity, MotionFrameInput, MotionGpuError, MotionGpuFieldSource,
    MotionGpuFieldSpec, MotionGpuResources, MotionGpuScopeSpec, MotionRuntimeDiagnostic,
    MotionRuntimeMetrics,
};
use crate::renderer::rack::{
    CollisionRackExecutor, RackGpuError, RackImageBindings, RackImageInput, RackResidualMeans,
    RackSourceBinding,
};
use crate::renderer::readback::{
    PreparedRgbaReadback, RecorderReadbackAdmission, RecorderReadbackAllocationSnapshot,
    RecorderReadbackCaptureStatus, RecorderReadbackError, RecorderReadbackPoll,
    RecorderReadbackReadiness, RecorderReadbackRequest, RecorderReadbackReservation,
    RecorderReadbackTag,
};
use crate::renderer::scan_processor::{ScanProcessorGpuExecutor, ScanProcessorGpuUniforms};
use crate::renderer::study::{StudyGpuExecutor, StudyGpuFrameUniforms};
use crate::renderer::symmetry_field::{
    SymmetryFieldBindings, SymmetryFieldExecutor, SymmetryFieldGpuError, SymmetryFieldInput,
    SymmetryFieldMotionBindings, SymmetryFieldMotionInput, SymmetryHistoryCursor,
    SymmetryMotionViews,
};
#[cfg(test)]
use crate::renderer::symmetry_field::{
    SYMMETRY_FIELD_CARRIER_PARITIES, SYMMETRY_FIELD_MOTION_GROUPS,
};
use crate::symmetry::{SYMMETRY_IMAGE_SLOTS, SYMMETRY_MOTION_SLOTS};
use crate::temporal::{TemporalFrameInput, TemporalResetCause, TemporalStateMetrics};
use crate::visual_rack::{
    EdgeTiming, GroupId, MatteChannel, NodeId, VisualScopeId, RACK_PRIMARY_ROUTE_SLOT,
};

pub(crate) const COMPOSITION_WORKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const COMPOSITION_PRESENT_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba8UnormSrgb;

/// Stable source view selected before command encoding. Preparing bind groups
/// from this descriptor is permitted; retaining the descriptor itself is not
/// required because wgpu bind groups retain their texture resources.
pub(crate) struct CompositionSourceDescriptor<'a> {
    pub stable_id: StableLayerId,
    pub view: &'a wgpu::TextureView,
    pub dimensions: [u32; 2],
    /// Caller-owned resource identity. A source replacement with unchanged
    /// stable ID and dimensions must advance this token so preparation can
    /// replace only bindings which retain the old view.
    pub resource_epoch: u64,
}

impl<'a> CompositionSourceDescriptor<'a> {
    pub const fn new(
        stable_id: StableLayerId,
        view: &'a wgpu::TextureView,
        dimensions: [u32; 2],
    ) -> Self {
        Self {
            stable_id,
            view,
            dimensions,
            resource_epoch: 0,
        }
    }

    pub const fn with_resource_epoch(mut self, resource_epoch: u64) -> Self {
        self.resource_epoch = resource_epoch;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CompositionFrameTiming {
    temporal: TemporalFrameInput,
}

impl CompositionFrameTiming {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "legacy dt/pause adapter remains a supported composition API"
        )
    )]
    pub fn new(delta_seconds: f32, advance_program: bool) -> Self {
        let delta_seconds = if delta_seconds.is_finite() {
            delta_seconds.max(0.0)
        } else {
            0.0
        };
        Self {
            temporal: TemporalFrameInput::legacy(delta_seconds, advance_program),
        }
    }

    /// Full T0 temporal-domain input. Main and export use this constructor to
    /// feed the identical freeze/blackout/event batch into the shared staged
    /// state; [`Self::new`] remains the legacy dt/pause compatibility seam.
    pub const fn from_temporal_input(temporal: TemporalFrameInput) -> Self {
        Self { temporal }
    }

    pub const fn temporal_input(self) -> TemporalFrameInput {
        self.temporal
    }
}

/// Frame-borrowed products/facts for the evaluated motion plan. The shared
/// executor remains the sole authority for source decisions, exact attachment
/// matching, and transactional publication.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CompositionMotionFrameInput<'a> {
    pub attachments: &'a [MotionFieldAttachment<'a>],
    pub held_scopes: &'a [VisualScopeId],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompositionPreparedKind {
    /// Caller must invoke the pre-existing exact renderer. No advanced GPU
    /// resource is allocated, cleared, or encoded for this plan.
    LegacyExact,
    /// Executor-held resources are ready for the reported immutable topology.
    Advanced { topology_signature: u64 },
}

pub(crate) struct CompositionGpuOutput<'a> {
    pub texture: &'a wgpu::Texture,
    pub view: &'a wgpu::TextureView,
    pub dimensions: [u32; 2],
    pub format: wgpu::TextureFormat,
}

/// One executor-retained post-local dry layer. The view is RGBA16Float and
/// remains valid until the next topology prepare; callers composite these in
/// the returned order after the shared Temporal family. No legacy source
/// rerender may substitute for this image because it already contains the
/// advanced rack and layer-Motion result.
pub(crate) struct CompositionTemporalBypassOverlay<'a> {
    pub stable_id: StableLayerId,
    pub view: &'a wgpu::TextureView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompositionEncodeKind {
    LegacyExact,
    Advanced,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompositionGpuError {
    ZeroDimensions([u32; 2]),
    DimensionsExceedDevice {
        requested: [u32; 2],
        limit: u32,
    },
    DuplicateSource(StableLayerId),
    MissingSource(StableLayerId),
    UnknownSource(StableLayerId),
    SourceDimensionMismatch {
        layer_id: StableLayerId,
        planned: [u32; 2],
        supplied: [u32; 2],
    },
    TopologyNotPrepared {
        prepared: Option<u64>,
        requested: u64,
    },
    #[allow(
        dead_code,
        reason = "explicit delegate error remains part of the shared adapter contract"
    )]
    LegacyExactMustDelegate,
    ResourceCreation(String),
    Host(String),
    Rack(String),
    Motion(String),
    Readback(String),
    InvalidSchedule(String),
    PresentFormatUnsupported(wgpu::TextureFormat),
}

impl fmt::Display for CompositionGpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions(dimensions) => write!(
                formatter,
                "advanced composition dimensions must be nonzero, got {}x{}",
                dimensions[0], dimensions[1]
            ),
            Self::DimensionsExceedDevice { requested, limit } => write!(
                formatter,
                "advanced composition dimensions {}x{} exceed device limit {limit}",
                requested[0], requested[1]
            ),
            Self::DuplicateSource(layer_id) => write!(
                formatter,
                "advanced composition received duplicate source for stable layer {}",
                layer_id.get()
            ),
            Self::MissingSource(layer_id) => write!(
                formatter,
                "advanced composition is missing source for stable layer {}",
                layer_id.get()
            ),
            Self::UnknownSource(layer_id) => write!(
                formatter,
                "advanced composition received unknown stable layer source {}",
                layer_id.get()
            ),
            Self::SourceDimensionMismatch {
                layer_id,
                planned,
                supplied,
            } => write!(
                formatter,
                "advanced composition source {} is {}x{}, planned {}x{}",
                layer_id.get(), supplied[0], supplied[1], planned[0], planned[1]
            ),
            Self::TopologyNotPrepared {
                prepared,
                requested,
            } => write!(
                formatter,
                "advanced composition topology {requested} is not prepared (prepared {prepared:?})"
            ),
            Self::LegacyExactMustDelegate => formatter.write_str(
                "exact legacy composition must be delegated to the frozen renderer path",
            ),
            Self::ResourceCreation(message) => {
                write!(formatter, "advanced composition GPU resource creation failed: {message}")
            }
            Self::Host(message) => write!(formatter, "advanced composition host failed: {message}"),
            Self::Rack(message) => write!(formatter, "advanced Collision Rack failed: {message}"),
            Self::Motion(message) => write!(formatter, "advanced motion failed: {message}"),
            Self::Readback(message) => {
                write!(formatter, "advanced scope readback failed: {message}")
            }
            Self::InvalidSchedule(message) => {
                write!(formatter, "advanced composition schedule is invalid: {message}")
            }
            Self::PresentFormatUnsupported(format) => write!(
                formatter,
                "advanced composition cannot present into {format:?}; expected {COMPOSITION_PRESENT_FORMAT:?}"
            ),
        }
    }
}

impl std::error::Error for CompositionGpuError {}

impl From<CompositionHostError> for CompositionGpuError {
    fn from(value: CompositionHostError) -> Self {
        Self::Host(value.to_string())
    }
}

impl From<RackGpuError> for CompositionGpuError {
    fn from(value: RackGpuError) -> Self {
        Self::Rack(value.to_string())
    }
}

/// The dedicated Symmetry Field executor reports through the same channel the
/// fixed rack executor does: both are creative-pass executors, and an operator
/// reading the message needs the failure, not the module that raised it.
impl From<SymmetryFieldGpuError> for CompositionGpuError {
    fn from(value: SymmetryFieldGpuError) -> Self {
        Self::Rack(value.to_string())
    }
}

impl From<MotionGpuError> for CompositionGpuError {
    fn from(value: MotionGpuError) -> Self {
        Self::Motion(value.to_string())
    }
}

impl From<RecorderReadbackError> for CompositionGpuError {
    fn from(value: RecorderReadbackError) -> Self {
        Self::Readback(value.to_string())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CompositionAllocationSnapshot {
    pub host_objects: u64,
    pub rack_objects: u64,
    pub retained_textures: u64,
    pub retained_views: u64,
    pub executor_bindings: u64,
    pub readback_objects: u64,
    pub readback_bytes: u64,
    pub readback_staging_bytes: u64,
    pub readback_texture_bytes: u64,
    /// Exact payload bytes from the admitted format/resource plans which
    /// created the live objects represented above.
    pub creative_bytes: u64,
    pub motion_bytes: u64,
}

impl CompositionAllocationSnapshot {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "aggregate allocation accounting is exercised by GPU goldens"
        )
    )]
    pub const fn total(self) -> u64 {
        self.host_objects
            + self.rack_objects
            + self.retained_textures
            + self.retained_views
            + self.executor_bindings
            + self.readback_objects
    }
}

struct RetainedSurface {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

enum TapBacking {
    Transparent,
    ProgramHistory,
    /// The etched gesture field. It owns **no** tap surface: the canvas
    /// textures are charged once by `GestureCanvasPlan`, and allocating a
    /// retained parity here as well would double-charge the same image against
    /// two independent ledgers.
    GestureCanvas,
    /// The programme tap. It likewise owns **no** tap surface: the one
    /// retained copy is charged by the renderer-owned full-frame texture
    /// floor, and a retained parity here would double-charge the same image
    /// against two independent ledgers.
    ProgramTap,
    CurrentPreLocal {
        layer_id: StableLayerId,
    },
    Current(RetainedSurface),
    Previous {
        surfaces: [RetainedSurface; 2],
        initialized: bool,
        staged: bool,
    },
}

struct PreparedTap {
    planned: PlannedImageTap,
    backing: TapBacking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RootTask {
    Layer(StableLayerId),
    Group(GroupId),
}

impl RootTask {
    const fn output_scope(self) -> VisualScopeId {
        match self {
            Self::Layer(id) => VisualScopeId::Layer(id),
            Self::Group(id) => VisualScopeId::Group(id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScheduledSource {
    Ping,
    RetainedTap(usize),
}

#[derive(Debug, Clone, Copy)]
struct ScheduledAdmission<T> {
    item: T,
    source: ScheduledSource,
}

struct RootScheduleEntry {
    task: RootTask,
    drains: Box<[ScheduledAdmission<RootTask>]>,
}

struct MemberScheduleEntry {
    layer_id: StableLayerId,
    drains: Box<[ScheduledAdmission<StableLayerId>]>,
}

struct PreparedRackSegment {
    scope: VisualScopeId,
    segment_index: u8,
    uniform_base_slot: usize,
    /// Bind groups retain their texture views, so both committed N-1 read
    /// parities are prepared once and selected without allocating in encode.
    bindings: [RackImageBindings; 2],
    /// Reduced block-mean surfaces for every active Residual node in this
    /// segment, allocated once from the admitted plan and never in encode.
    residual_means: RackResidualMeans,
    tap_indices: Box<[(NodeId, u8, crate::visual_rack::ResolvedImageTap, usize)]>,
}

/// One prepared dedicated Symmetry Field step.
///
/// The node owns its own eight-texture bind groups rather than sharing a
/// segment's, because slot index is route identity here: `tap_indices` records
/// which image slot each admitted route occupies, and a readiness flip on one
/// slot can never move the other slot's donor.
/// One prepared dedicated Study pass: the arena slot its program occupies
/// and its bind group over (Ping carrier, committed history array). No taps,
/// no motion, no donors — the whole binding set is topology-fixed.
struct PreparedStudyField {
    scope: VisualScopeId,
    node_id: NodeId,
    uniform_slot: u32,
    bind_group: wgpu::BindGroup,
}

/// One prepared dedicated Scan Processor pass: the arena slot its uniforms
/// occupy and its two bind groups (geometry over the Ping carrier; resolve
/// over Ping plus the shared accumulator). No taps, no motion, no donors —
/// the whole binding set is topology-fixed, and the vertex count is a
/// draw-call argument read from the evaluated plan at encode.
struct PreparedScanProcessorField {
    scope: VisualScopeId,
    node_id: NodeId,
    uniform_slot: u32,
    geometry_bind_group: wgpu::BindGroup,
    resolve_bind_group: wgpu::BindGroup,
}

/// The lazily allocated retained history of one Filter Avalanche node: the
/// node's own previous output, program memory on the melt-history precedent.
struct AvalancheHistory {
    surface: RetainedSurface,
    /// (Ping carrier, history) — the warm read pair.
    bind_group: wgpu::BindGroup,
}

/// One prepared dedicated B6 corruption step. The pass groups are
/// topology-fixed at prepare; only the avalanche's history pair arrives
/// lazily, on its first armed frame, and is counted when it does.
struct PreparedCorruptionField {
    scope: VisualScopeId,
    node_id: NodeId,
    /// First arena slot; a Block DCT step owns four consecutive slots.
    uniform_base_slot: u32,
    /// DCT: [(Ping,Ping), (auxA,Ping), (auxB,Ping), (auxA,Ping)];
    /// Sort: one (Ping,Ping); Avalanche: the cold fallback (Ping,Ping),
    /// which degrades exactly to BENDR's shipped single-frame law.
    pass_groups: Vec<wgpu::BindGroup>,
    history: Option<AvalancheHistory>,
    /// Published only by the frame-history transaction, so a discarded
    /// frame leaves no stale validity and no double-advanced store clock.
    history_valid: bool,
    history_staged: bool,
    last_store_tick: u64,
    staged_store_tick: u64,
}

struct PreparedSymmetryField {
    scope: VisualScopeId,
    node_id: NodeId,
    uniform_slot: usize,
    /// Both committed N-1 read parities, exactly like a rack segment. Each
    /// holds one image bind group per carrier parity, so four per node.
    bindings: [SymmetryFieldBindings; 2],
    /// Every combination of the two motion slots' committed ping/pong parities,
    /// prepared once per node. The motion group holds no image view, so it is
    /// independent of the N-1 read parity and is deliberately NOT rebuilt per
    /// read index; four per node, not eight.
    motion_bindings: SymmetryFieldMotionBindings,
    /// The admitted motion field slot each motion route resolved to, carried so
    /// encode can ask the motion resources for that field's committed parity
    /// without re-walking the evaluated plan.
    motion_field_slots: [Option<u8>; SYMMETRY_MOTION_SLOTS],
    /// `(image slot, tap index)` for each admitted image route.
    tap_indices: Box<[(usize, usize)]>,
}

struct PreparedAdvanced {
    topology_signature: u64,
    source_keys: Box<[(StableLayerId, [u32; 2], u64)]>,
    /// The gesture-canvas identity these bind groups were built against. A
    /// changed identity is a rebuild, not a repair: nothing here rebinds.
    gesture_canvas_identity: (bool, u64),
    /// Whether a routed canvas tap actually reads a presented field. This is
    /// what `tap_ready` answers with, so an unbound canvas gives its consumer
    /// `donor_valid = false` instead of a confident empty field.
    gesture_canvas_bound: bool,
    /// The programme-tap identity these bind groups were built against, on
    /// exactly the canvas law: a changed identity is a rebuild, not a repair.
    program_tap_identity: (bool, u64),
    /// Whether a routed programme tap actually reads a published copy, so an
    /// unbound tap gives its consumer `donor_valid = false` instead of a
    /// confident empty image.
    program_tap_bound: bool,
    host: CompositionHost,
    rack: CollisionRackExecutor,
    /// Present only when the plan emits at least one dedicated step. It owns no
    /// output-sized surface, so an absent executor costs nothing.
    symmetry: Option<SymmetryFieldExecutor>,
    symmetry_fields: Vec<PreparedSymmetryField>,
    study: Option<StudyGpuExecutor>,
    study_fields: Vec<PreparedStudyField>,
    scan: Option<ScanProcessorGpuExecutor>,
    scan_fields: Vec<PreparedScanProcessorField>,
    corruption: Option<CorruptionGpuExecutor>,
    corruption_fields: Vec<PreparedCorruptionField>,
    motion: Option<MotionGpuResources>,
    taps: Box<[PreparedTap]>,
    /// Current-frame PreLocal donors are independently materialized and
    /// deduplicated by stable layer ID. They cannot alias a single transient
    /// Pong surface when more than one donor is used in an authored rack.
    prelocal_surfaces: BTreeMap<StableLayerId, RetainedSurface>,
    /// Post-local/post-layer-Motion images for the authored dry prefix. One
    /// surface per authored member is preflighted in the shared resource
    /// ledger and overwritten exactly once per encoded frame.
    temporal_dry_surfaces: BTreeMap<StableLayerId, RetainedSurface>,
    root_schedule: Box<[RootScheduleEntry]>,
    member_schedules: BTreeMap<GroupId, Box<[MemberScheduleEntry]>>,
    external_effects: BTreeMap<StableLayerId, HostEffectSource>,
    ping_effect: HostEffectSource,
    group_effect: HostEffectSource,
    ping_rack_source: RackSourceBinding,
    rack_segments: Vec<PreparedRackSegment>,
    composite_a: HostCompositeInputs,
    composite_b: HostCompositeInputs,
    composite_program: HostCompositeInputs,
    composite_group: HostCompositeInputs,
    prefix_group_composite: HostCompositeInputs,
    /// Like rack image bindings, matte bindings are immutable and prepared for
    /// both possible committed N-1 read parities.
    matte_bindings: BTreeMap<ImageTapConsumer, [HostMatteInputs; 2]>,
    temporal: HostTemporalInput,
    routed_garden: Option<HostRoutedGardenInput>,
    bus_inputs: HostTextureInputs,
    /// Retained for the bus melt's lazy history allocation: the melt is a
    /// value, never topology, so its surface cannot be sized at prepare.
    device: wgpu::Device,
    /// The bus melt's own previous output — one retained working-format
    /// surface on the temporal-feedback single-surface precedent, lazily
    /// allocated on the first armed frame, retained thereafter, and
    /// invalidated (never freed) on disarm so a re-arm cannot resurrect a
    /// stale trail.
    bus_melt: Option<(RetainedSurface, wgpu::BindGroup)>,
    bus_melt_valid: bool,
    bus_melt_staged: bool,
    /// The melt store's own 30 Hz reference accumulator: committed debt plus
    /// the staged value the frame-history transaction publishes.
    bus_melt_tick_debt: f32,
    bus_melt_staged_debt: f32,
    present: HostTextureInputs,
    effect_slots: BTreeMap<(VisualScopeId, usize), HostUniformSlot>,
    exact_layer_slots: BTreeMap<StableLayerId, HostUniformSlot>,
    prelocal_slots: BTreeMap<StableLayerId, HostUniformSlot>,
    composite_slots: BTreeMap<VisualScopeId, HostUniformSlot>,
    prefix_composite_slot: HostUniformSlot,
    matte_slots: BTreeMap<ImageTapConsumer, HostUniformSlot>,
    tap_history_read_index: usize,
    history_staged: bool,
    retained_textures: u64,
    retained_views: u64,
    executor_bindings: u64,
    creative_bytes: u64,
    motion_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct ArmedScopeReadback {
    reservation: RecorderReadbackReservation,
    captured: bool,
}

/// One reusable RGBA8 conversion surface plus the common fixed staging pool.
/// It lives outside `PreparedAdvanced`, so a topology replacement cannot
/// silently retarget a recorder or allocate during encode.
struct PreparedScopeRecorderReadback {
    target: CaptureTarget,
    conversion_texture: wgpu::Texture,
    conversion_view: wgpu::TextureView,
    staging: PreparedRgbaReadback,
    armed: Option<ArmedScopeReadback>,
}

#[allow(
    dead_code,
    reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
)]
impl PreparedScopeRecorderReadback {
    fn prepare(
        device: &wgpu::Device,
        dimensions: [u32; 2],
        target: CaptureTarget,
    ) -> Result<Self, RecorderReadbackError> {
        if matches!(target, CaptureTarget::Program) {
            return Err(RecorderReadbackError::UnsupportedTarget(target));
        }
        let staging = PreparedRgbaReadback::prepare(device, dimensions, 1)?;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let conversion_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prepared post-effects scope recorder conversion"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COMPOSITION_PRESENT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let conversion_view =
            conversion_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let scope_error = [
            ("out of memory", pollster::block_on(out_of_memory.pop())),
            ("internal/backend", pollster::block_on(internal.pop())),
            ("validation", pollster::block_on(validation.pop())),
        ]
        .into_iter()
        .find_map(|(kind, error)| error.map(|error| format!("{kind}: {error}")));
        if let Some(message) = scope_error {
            return Err(RecorderReadbackError::ResourceCreation(message));
        }
        Ok(Self {
            target,
            conversion_texture,
            conversion_view,
            staging,
            armed: None,
        })
    }

    fn allocation_snapshot(&self) -> RecorderReadbackAllocationSnapshot {
        self.staging.allocation_snapshot(1)
    }

    fn begin(&mut self, tag: RecorderReadbackTag) -> RecorderReadbackAdmission {
        if self.armed.is_some() {
            return RecorderReadbackAdmission::Busy;
        }
        let admission = self
            .staging
            .reserve(RecorderReadbackRequest::new(self.target, tag));
        if let RecorderReadbackAdmission::Scheduled(reservation) = admission {
            self.armed = Some(ArmedScopeReadback {
                reservation,
                captured: false,
            });
        }
        admission
    }

    fn capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        host: &CompositionHost,
        present: &HostTextureInputs,
        scope: VisualScopeId,
    ) -> Result<(), CompositionGpuError> {
        if capture_target_scope(self.target) != Some(scope) {
            return Ok(());
        }
        let Some(armed) = self.armed else {
            return Ok(());
        };
        if armed.captured {
            return Ok(());
        }
        host.encode_present(
            encoder,
            present,
            &self.conversion_view,
            COMPOSITION_PRESENT_FORMAT,
        )?;
        self.staging
            .encode_reserved(encoder, armed.reservation, &self.conversion_texture)?;
        self.armed = Some(ArmedScopeReadback {
            captured: true,
            ..armed
        });
        Ok(())
    }

    fn finish(
        &mut self,
        reservation: RecorderReadbackReservation,
    ) -> Result<RecorderReadbackCaptureStatus, RecorderReadbackError> {
        let Some(armed) = self.armed else {
            return Err(RecorderReadbackError::InvalidReservation);
        };
        if armed.reservation != reservation {
            return Err(RecorderReadbackError::InvalidReservation);
        }
        self.armed = None;
        if armed.captured {
            Ok(RecorderReadbackCaptureStatus::Captured)
        } else {
            self.staging.discard_unsubmitted(reservation)?;
            Ok(RecorderReadbackCaptureStatus::SourceUnavailable)
        }
    }

    fn discard_armed_unsubmitted(&mut self) {
        let Some(armed) = self.armed.take() else {
            return;
        };
        let _ = self.staging.discard_unsubmitted(armed.reservation);
    }
}

const fn capture_target_scope(target: CaptureTarget) -> Option<VisualScopeId> {
    match target {
        CaptureTarget::Program => None,
        CaptureTarget::Layer(layer_id) => Some(VisualScopeId::Layer(layer_id)),
        CaptureTarget::Group(group_id) => Some(VisualScopeId::Group(group_id)),
    }
}

/// Executor-owned preparation. The public boundary intentionally returns only
/// a value report; callers never own a half-built advanced resource graph.
/// The etched gesture field this executor binds for canvas image routes.
///
/// `None` is the exact pre-gesture path: a canvas route reads the rack-owned
/// zero texture and its consumer sees `donor_valid = false`. The `epoch` is a
/// caller-owned monotonic identity for the *resources*, not for their contents:
/// a canvas rebuild (resize, admission change, host restart) advances it, which
/// is what forces the executor to re-prepare bind groups that would otherwise
/// be reused across a topology that did not itself change.
#[derive(Debug, Clone, Default)]
pub(crate) struct GestureCanvasBinding {
    view: Option<wgpu::TextureView>,
    epoch: u64,
}

impl GestureCanvasBinding {
    pub(crate) fn bound(view: wgpu::TextureView, epoch: u64) -> Self {
        Self {
            view: Some(view),
            epoch,
        }
    }

    pub(crate) fn is_bound(&self) -> bool {
        self.view.is_some()
    }

    /// The identity two bindings are compared by. An unbound binding is always
    /// epoch zero so it cannot be confused with a bound one that happens to
    /// share a caller's counter.
    fn identity(&self) -> (bool, u64) {
        (self.view.is_some(), self.epoch)
    }
}

/// The programme tap binding shares the canvas binding's exact shape: an
/// optional view plus a caller-owned monotonic resource epoch. One type serves
/// both master-scope singletons; the two bind seams are separate functions on
/// the executor, so a caller cannot hand one singleton to the other's routes.
pub(crate) type ProgramTapBinding = GestureCanvasBinding;

pub(crate) struct CompositionGpuExecutor {
    dimensions: [u32; 2],
    topology_signature: Option<u64>,
    prepared: Option<PreparedAdvanced>,
    scope_recorder_readback: Option<PreparedScopeRecorderReadback>,
    /// The gesture canvas binding future prepares will use, plus the identity
    /// the last prepare actually used. They are separate facts: setting a
    /// binding never rebuilds anything on its own, and a stale prepared
    /// identity is what makes the next prepare rebuild rather than reuse.
    gesture_canvas: GestureCanvasBinding,
    /// The programme tap binding future prepares will use, on exactly the
    /// canvas law: setting it never rebuilds anything on its own.
    program_tap: ProgramTapBinding,
}

impl CompositionGpuExecutor {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        dimensions: [u32; 2],
    ) -> Result<Self, CompositionGpuError> {
        if dimensions.contains(&0) {
            return Err(CompositionGpuError::ZeroDimensions(dimensions));
        }
        let limit = device.limits().max_texture_dimension_2d;
        if dimensions[0] > limit || dimensions[1] > limit {
            return Err(CompositionGpuError::DimensionsExceedDevice {
                requested: dimensions,
                limit,
            });
        }
        Ok(Self {
            dimensions,
            topology_signature: None,
            prepared: None,
            scope_recorder_readback: None,
            gesture_canvas: GestureCanvasBinding::default(),
            program_tap: ProgramTapBinding::default(),
        })
    }

    pub const fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    /// Publish the etched gesture field a canvas image route will bind.
    ///
    /// This is deliberately separate from `prepare`: every other executor input
    /// is per-topology, while the canvas is a long-lived singleton the host
    /// owns. Calling this never allocates and never rebuilds — the next
    /// `prepare` observes the changed identity and rebuilds then, in the one
    /// place that already owns bind-group construction.
    pub(crate) fn bind_gesture_canvas(&mut self, binding: GestureCanvasBinding) {
        self.gesture_canvas = binding;
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the physical-GPU payoff fixture asserts the binding it prepared against"
        )
    )]
    pub(crate) fn gesture_canvas_bound(&self) -> bool {
        self.gesture_canvas.is_bound()
    }

    /// Publish the programme-tap image a tap route will bind, on exactly the
    /// canvas seam: deliberately separate from `prepare`, never allocating,
    /// never rebuilding — the next `prepare` observes the changed identity.
    pub(crate) fn bind_program_tap(&mut self, binding: ProgramTapBinding) {
        self.program_tap = binding;
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the physical-GPU payoff fixture asserts the binding it prepared against"
        )
    )]
    pub(crate) fn program_tap_bound(&self) -> bool {
        self.program_tap.is_bound()
    }

    pub fn program_history_initialized(&self) -> bool {
        self.prepared
            .as_ref()
            .is_some_and(|prepared| prepared.host.program_history_initialized())
    }

    #[allow(dead_code, reason = "compatibility wrapper for embedders")]
    pub fn reset_history(&mut self) {
        self.reset_history_for(TemporalResetCause::PatchGeneration);
    }

    pub(crate) fn reset_history_for(&mut self, cause: TemporalResetCause) {
        if let Some(prepared) = &mut self.prepared {
            prepared.host.reset_program_history();
            prepared.host.reset_temporal_for(cause);
            prepared.tap_history_read_index = 0;
            prepared.history_staged = false;
            for tap in &mut prepared.taps {
                if let TapBacking::Previous {
                    initialized,
                    staged,
                    ..
                } = &mut tap.backing
                {
                    *initialized = false;
                    *staged = false;
                }
            }
            if matches!(
                cause,
                TemporalResetCause::PatchGeneration
                    | TemporalResetCause::ApplyLook
                    | TemporalResetCause::SourceCut
                    | TemporalResetCause::Seek
                    | TemporalResetCause::Resize
                    | TemporalResetCause::BroadRevert
                    | TemporalResetCause::ManualClear
            ) {
                if let Some(motion) = &mut prepared.motion {
                    motion.reset();
                }
            }
        }
    }

    /// Operator-facing M4 memory clear. Authored parameters, Temporal,
    /// Program/N-1 history, image taps, and the held audience image remain
    /// untouched.
    pub(crate) fn clear_motion_memory(&mut self) {
        if let Some(motion) = self
            .prepared
            .as_mut()
            .and_then(|prepared| prepared.motion.as_mut())
        {
            motion.reset();
        }
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "Main telemetry consumes this frozen runtime seam")
    )]
    pub(crate) fn motion_metrics(&self, scope: VisualScopeId) -> Option<MotionRuntimeMetrics> {
        self.prepared
            .as_ref()
            .and_then(|prepared| prepared.motion.as_ref())
            .and_then(|motion| motion.metrics(scope))
    }

    #[allow(
        dead_code,
        reason = "Main/export telemetry consumes this frozen runtime seam"
    )]
    pub(crate) fn motion_diagnostics(&self) -> &[MotionRuntimeDiagnostic] {
        self.prepared
            .as_ref()
            .and_then(|prepared| prepared.motion.as_ref())
            .map_or(&[], MotionGpuResources::diagnostics)
    }

    /// Operator-facing M3 memory clear. This is intentionally narrower than
    /// [`Self::reset_history`]: stable image taps and N-1 Program history stay
    /// published, while the temporal ring/carrier/Score reset and a frozen
    /// audience image remains available until Program Freeze is released.
    pub fn clear_temporal_memory(&mut self) {
        if let Some(prepared) = &mut self.prepared {
            prepared.host.clear_temporal_memory();
        }
    }

    /// Rebase only the shared Temporal-family memory when the exact renderer's
    /// wet/dry membership changes. Advanced composition cannot execute the
    /// mixed partition, but a retained executor may still exist from an older
    /// accepted topology and must not later expose stale Temporal pixels.
    pub(crate) fn reset_temporal_bypass_partition(&mut self) {
        if let Some(prepared) = &mut self.prepared {
            prepared
                .host
                .reset_temporal_for(TemporalResetCause::BypassPartition);
        }
    }

    #[allow(
        dead_code,
        reason = "native telemetry selects this adapter only while Advanced composition is active"
    )]
    pub(crate) fn temporal_state_metrics(&self) -> Option<TemporalStateMetrics> {
        self.prepared
            .as_ref()
            .map(|prepared| prepared.host.temporal_state_metrics())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "allocation snapshots are exposed for warmed-frame GPU goldens"
        )
    )]
    pub fn allocation_snapshot(&self) -> CompositionAllocationSnapshot {
        let Some(prepared) = &self.prepared else {
            return self.scope_recorder_readback.as_ref().map_or_else(
                CompositionAllocationSnapshot::default,
                |readback| {
                    let readback = readback.allocation_snapshot();
                    CompositionAllocationSnapshot {
                        readback_objects: readback.total_objects(),
                        readback_bytes: readback.total_bytes(),
                        ..CompositionAllocationSnapshot::default()
                    }
                },
            );
        };
        let readback = self
            .scope_recorder_readback
            .as_ref()
            .map(PreparedScopeRecorderReadback::allocation_snapshot)
            .unwrap_or_default();
        CompositionAllocationSnapshot {
            host_objects: prepared.host.allocation_snapshot().total(),
            rack_objects: prepared.rack.allocation_snapshot().total() + prepared.symmetry_objects(),
            retained_textures: prepared.retained_textures,
            retained_views: prepared.retained_views,
            executor_bindings: prepared.executor_bindings,
            readback_objects: readback.total_objects(),
            readback_bytes: readback.total_bytes(),
            readback_staging_bytes: readback.buffer_bytes,
            readback_texture_bytes: readback.conversion_texture_bytes,
            creative_bytes: prepared.creative_bytes,
            motion_bytes: prepared.motion_bytes,
        }
    }

    /// Cold-prepare the one stable layer/group capture target. Re-selecting a
    /// target reuses the same fixed RGBA8 surface and two staging buffers.
    /// Program capture belongs to `Renderer` because only it observes the
    /// post-NTSC/post-blackout audience slot.
    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn prepare_scope_recorder_readback(
        &mut self,
        device: &wgpu::Device,
        target: CaptureTarget,
    ) -> Result<RecorderReadbackAllocationSnapshot, RecorderReadbackError> {
        if matches!(target, CaptureTarget::Program) {
            return Err(RecorderReadbackError::UnsupportedTarget(target));
        }
        if let Some(readback) = &mut self.scope_recorder_readback {
            if readback.armed.is_some() {
                return Err(RecorderReadbackError::InvalidReservation);
            }
            readback.target = target;
            return Ok(readback.allocation_snapshot());
        }
        let readback = PreparedScopeRecorderReadback::prepare(device, self.dimensions, target)?;
        let snapshot = readback.allocation_snapshot();
        self.scope_recorder_readback = Some(readback);
        Ok(snapshot)
    }

    /// Arm one exact post-effects boundary before `encode_with_motion`.
    /// Missing/deleted scopes fail visibly and never retarget to Program.
    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn begin_scope_recorder_readback(
        &mut self,
        tag: RecorderReadbackTag,
    ) -> RecorderReadbackAdmission {
        let Some(target) = self
            .scope_recorder_readback
            .as_ref()
            .map(|readback| readback.target)
        else {
            return RecorderReadbackAdmission::Unprepared;
        };
        if !self
            .prepared
            .as_ref()
            .is_some_and(|prepared| prepared.has_capture_target(target))
        {
            return RecorderReadbackAdmission::SourceUnavailable;
        }
        self.scope_recorder_readback
            .as_mut()
            .expect("checked prepared scope readback")
            .begin(tag)
    }

    /// Confirm that the armed boundary was actually encountered. Invoke after
    /// a successful encode and before submission; unavailable is a capture
    /// drop, not a reason to reject the creative frame.
    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn finish_scope_recorder_readback(
        &mut self,
        reservation: RecorderReadbackReservation,
    ) -> Result<RecorderReadbackCaptureStatus, RecorderReadbackError> {
        self.scope_recorder_readback
            .as_mut()
            .ok_or(RecorderReadbackError::InvalidReservation)?
            .finish(reservation)
    }

    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn map_scope_recorder_readback(
        &self,
        reservation: RecorderReadbackReservation,
    ) -> Result<(), RecorderReadbackError> {
        self.scope_recorder_readback
            .as_ref()
            .ok_or(RecorderReadbackError::InvalidReservation)?
            .staging
            .map(reservation)
    }

    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn poll_scope_recorder_readback_into(
        &mut self,
        device: &wgpu::Device,
        destination: &mut [u8],
    ) -> Result<RecorderReadbackPoll, RecorderReadbackError> {
        let _ = device.poll(wgpu::PollType::Poll);
        match self.scope_recorder_readback.as_mut() {
            Some(readback) => readback.staging.poll_into(destination),
            None => Ok(RecorderReadbackPoll::Idle),
        }
    }

    /// Non-consuming oldest-slot observation for lease-on-ready harvesting.
    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn scope_recorder_readback_readiness(
        &self,
        device: &wgpu::Device,
    ) -> RecorderReadbackReadiness {
        let _ = device.poll(wgpu::PollType::Poll);
        self.scope_recorder_readback
            .as_ref()
            .map_or(RecorderReadbackReadiness::Idle, |readback| {
                readback.staging.oldest_readiness()
            })
    }

    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn discard_unsubmitted_scope_recorder_readback(
        &mut self,
        reservation: RecorderReadbackReservation,
    ) -> Result<(), RecorderReadbackError> {
        let readback = self
            .scope_recorder_readback
            .as_mut()
            .ok_or(RecorderReadbackError::InvalidReservation)?;
        if readback
            .armed
            .is_some_and(|armed| armed.reservation == reservation)
        {
            readback.armed = None;
        }
        readback.staging.discard_unsubmitted(reservation)
    }

    /// True only when the warmed executor exactly matches both the immutable
    /// topology and every source view-generation key. Callers use this before
    /// planning donor readiness so a topology/source replacement is diagnosed
    /// cold on its first frame instead of borrowing the prior graph's history.
    pub fn is_prepared_for(
        &self,
        plan: &EvaluatedCompositionPlan,
        source_keys: &[(StableLayerId, [u32; 2], u64)],
    ) -> bool {
        let EvaluatedCompositionPlan::Advanced(plan) = plan else {
            return false;
        };
        let gesture_identity = self.gesture_canvas.identity();
        let program_tap_identity = self.program_tap.identity();
        self.prepared.as_ref().is_some_and(|prepared| {
            prepared.topology_signature == plan.topology_signature()
                && prepared.source_keys.as_ref() == source_keys
                && prepared.gesture_canvas_identity == gesture_identity
                && prepared.program_tap_identity == program_tap_identity
        })
    }

    pub fn validate_plan(plan: &EvaluatedCompositionPlan) -> Result<(), CompositionGpuError> {
        let _ = plan;
        Ok(())
    }

    /// Validate a complete stable source set and atomically select the plan's
    /// topology. Concrete per-pass resources are installed here as the
    /// implementation grows; encode never repairs or allocates missing state.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &EvaluatedCompositionPlan,
        sources: &[CompositionSourceDescriptor<'_>],
    ) -> Result<CompositionPreparedKind, CompositionGpuError> {
        let EvaluatedCompositionPlan::Advanced(plan) = plan else {
            self.topology_signature = None;
            self.prepared = None;
            return Ok(CompositionPreparedKind::LegacyExact);
        };
        Self::validate_plan(&EvaluatedCompositionPlan::Advanced(plan.clone()))?;

        let mut supplied = BTreeMap::new();
        for source in sources {
            if supplied.insert(source.stable_id, source).is_some() {
                return Err(CompositionGpuError::DuplicateSource(source.stable_id));
            }
        }
        for layer in plan.layers() {
            let source = supplied
                .remove(&layer.stable_id)
                .ok_or(CompositionGpuError::MissingSource(layer.stable_id))?;
            let planned = plan.base().layers()[layer.base_layer_index].source.size;
            if source.dimensions != planned {
                return Err(CompositionGpuError::SourceDimensionMismatch {
                    layer_id: layer.stable_id,
                    planned,
                    supplied: source.dimensions,
                });
            }
        }
        if let Some(unknown) = supplied.keys().next().copied() {
            return Err(CompositionGpuError::UnknownSource(unknown));
        }
        let source_keys = plan
            .layers()
            .iter()
            .map(|layer| {
                let source = sources
                    .iter()
                    .find(|source| source.stable_id == layer.stable_id)
                    .expect("validated source exists");
                (source.stable_id, source.dimensions, source.resource_epoch)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let signature = plan.topology_signature();
        let gesture_identity = self.gesture_canvas.identity();
        let program_tap_identity = self.program_tap.identity();
        if self.prepared.as_ref().is_some_and(|prepared| {
            prepared.topology_signature == signature
                && prepared.source_keys == source_keys
                // A rebuilt gesture canvas produces a new presented view while
                // the topology and every source key stay identical, so the
                // canvas identity has to be part of the reuse test or the tap
                // bind groups would keep a destroyed surface's view. The
                // programme tap obeys the identical law.
                && prepared.gesture_canvas_identity == gesture_identity
                && prepared.program_tap_identity == program_tap_identity
        }) {
            self.topology_signature = Some(signature);
            return Ok(CompositionPreparedKind::Advanced {
                topology_signature: signature,
            });
        }

        let prepared = PreparedAdvanced::new(
            device,
            queue,
            self.dimensions,
            plan,
            sources,
            source_keys,
            &self.gesture_canvas,
            &self.program_tap,
        )?;
        self.prepared = Some(prepared);
        self.topology_signature = Some(signature);
        Ok(CompositionPreparedKind::Advanced {
            topology_signature: signature,
        })
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "compatibility wrapper for embedders without motion products"
        )
    )]
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &EvaluatedCompositionPlan,
        timing: CompositionFrameTiming,
    ) -> Result<CompositionEncodeKind, CompositionGpuError> {
        self.encode_with_motion(
            queue,
            encoder,
            plan,
            timing,
            CompositionMotionFrameInput::default(),
        )
    }

    /// Recycle a caller-verified stale scope completion without allocating a
    /// full-frame scratch or acquiring a recorder lease.
    #[allow(
        dead_code,
        reason = "native Main consumes scope capture; alternate targets retain the prepared adapter without a caller"
    )]
    pub(crate) fn recycle_ready_scope_recorder_readback_without_copy(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<RecorderReadbackPoll, RecorderReadbackError> {
        let _ = device.poll(wgpu::PollType::Poll);
        match self.scope_recorder_readback.as_mut() {
            Some(readback) => readback.staging.recycle_oldest_ready_without_copy(),
            None => Ok(RecorderReadbackPoll::Idle),
        }
    }

    pub fn encode_with_motion(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &EvaluatedCompositionPlan,
        timing: CompositionFrameTiming,
        motion_input: CompositionMotionFrameInput<'_>,
    ) -> Result<CompositionEncodeKind, CompositionGpuError> {
        let EvaluatedCompositionPlan::Advanced(plan) = plan else {
            return Ok(CompositionEncodeKind::LegacyExact);
        };
        Self::validate_plan(&EvaluatedCompositionPlan::Advanced(plan.clone()))?;
        if self.topology_signature != Some(plan.topology_signature()) {
            return Err(CompositionGpuError::TopologyNotPrepared {
                prepared: self.topology_signature,
                requested: plan.topology_signature(),
            });
        }
        let (prepared, scope_readback) = (&mut self.prepared, &mut self.scope_recorder_readback);
        let prepared = prepared
            .as_mut()
            .ok_or(CompositionGpuError::TopologyNotPrepared {
                prepared: None,
                requested: plan.topology_signature(),
            })?;
        if let Err(error) =
            prepared.encode(queue, encoder, plan, timing, motion_input, scope_readback)
        {
            prepared.discard_frame_history();
            if let Some(readback) = scope_readback {
                readback.discard_armed_unsubmitted();
            }
            return Err(error);
        }
        Ok(CompositionEncodeKind::Advanced)
    }

    pub fn output(&self) -> CompositionGpuOutput<'_> {
        let prepared = self
            .prepared
            .as_ref()
            .expect("advanced composition output requested before prepare");
        let output = prepared.host.surface(HostSurface::Ping);
        CompositionGpuOutput {
            texture: output.texture,
            view: output.view,
            dimensions: self.dimensions,
            format: COMPOSITION_WORKING_FORMAT,
        }
    }

    /// Borrow the exact post-local dry surfaces in compositor order. A plan
    /// mismatch or absent retained member is a hard schedule error: callers
    /// must never fall back to re-rendering the raw legacy layer source.
    pub(crate) fn temporal_bypass_overlays<'a>(
        &'a self,
        plan: &AdvancedCompositionPlan,
    ) -> Result<Vec<CompositionTemporalBypassOverlay<'a>>, CompositionGpuError> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or(CompositionGpuError::TopologyNotPrepared {
                prepared: None,
                requested: plan.topology_signature(),
            })?;
        if prepared.topology_signature != plan.topology_signature() {
            return Err(CompositionGpuError::TopologyNotPrepared {
                prepared: Some(prepared.topology_signature),
                requested: plan.topology_signature(),
            });
        }
        plan.temporal_dry_layers()
            .iter()
            .map(|stable_id| {
                let surface = prepared
                    .temporal_dry_surfaces
                    .get(stable_id)
                    .ok_or_else(|| {
                        CompositionGpuError::InvalidSchedule(format!(
                            "Temporal dry layer {} has no retained post-local surface",
                            stable_id.get()
                        ))
                    })?;
                Ok(CompositionTemporalBypassOverlay {
                    stable_id: *stable_id,
                    view: &surface.view,
                })
            })
            .collect()
    }

    /// Final RGBA16Float -> target conversion is intentionally a render/blit,
    /// never a texture copy into the existing Rgba8UnormSrgb composites.
    pub fn encode_present(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) -> Result<(), CompositionGpuError> {
        if target_format != COMPOSITION_PRESENT_FORMAT {
            return Err(CompositionGpuError::PresentFormatUnsupported(target_format));
        }
        let prepared = self
            .prepared
            .as_ref()
            .ok_or(CompositionGpuError::TopologyNotPrepared {
                prepared: None,
                requested: self.topology_signature.unwrap_or(0),
            })?;
        prepared
            .host
            .encode_present(encoder, &prepared.present, target, target_format)
            .map_err(Into::into)
    }

    /// Publish every CPU-visible N-1 readiness/cadence bit only after the
    /// caller has successfully submitted the command buffer containing this
    /// frame. No GPU object or command encoder is created here.
    pub fn commit_frame_history(&mut self) {
        let Some(prepared) = &mut self.prepared else {
            return;
        };
        prepared.commit_frame_history();
    }

    /// Roll back all staged CPU history/cadence state when encode, present, or
    /// outer submission is abandoned.
    pub fn discard_frame_history(&mut self) {
        if let Some(prepared) = &mut self.prepared {
            prepared.discard_frame_history();
        }
        if let Some(readback) = &mut self.scope_recorder_readback {
            readback.discard_armed_unsubmitted();
        }
    }
}

impl PreparedAdvanced {
    #[allow(
        dead_code,
        reason = "the prepared scope-capture adapter uses this exact target-presence check"
    )]
    fn has_capture_target(&self, target: CaptureTarget) -> bool {
        match target {
            CaptureTarget::Program => false,
            CaptureTarget::Layer(layer_id) => self
                .source_keys
                .iter()
                .any(|(candidate, _, _)| *candidate == layer_id),
            CaptureTarget::Group(group_id) => {
                self.member_schedules.contains_key(&group_id)
                    || self.root_schedule.iter().any(
                        |entry| matches!(entry.task, RootTask::Group(candidate) if candidate == group_id),
                    )
            }
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the two singleton bindings are independent inputs, not a bundle"
    )]
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: [u32; 2],
        plan: &AdvancedCompositionPlan,
        sources: &[CompositionSourceDescriptor<'_>],
        source_keys: Box<[(StableLayerId, [u32; 2], u64)]>,
        gesture_canvas: &GestureCanvasBinding,
        program_tap: &ProgramTapBinding,
    ) -> Result<Self, CompositionGpuError> {
        let (
            effect_slot_count,
            effect_slots,
            exact_layer_slots,
            prelocal_slots,
            composite_slots,
            prefix_composite_slot,
            matte_slots,
        ) = assign_uniform_slots(plan)?;
        let retain_program_history = plan
            .image_taps()
            .iter()
            .any(|tap| matches!(tap.resolved, PlannedImageSource::ProgramHistory));

        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let built = (|| {
            let host = CompositionHost::new(
                device,
                dimensions,
                HostCapacities {
                    effect_slots: effect_slot_count.max(1),
                    composite_slots: (composite_slots.len() + 1).max(1),
                    matte_slots: matte_slots.len().max(1),
                    retain_program_history,
                    resources: plan.resources(),
                },
            )?;
            let rack = CollisionRackExecutor::new_with_uniform_slots(
                device,
                queue,
                dimensions,
                rack_uniform_slot_count(plan)?,
            )?;
            let taps = prepare_tap_surfaces(device, dimensions, plan.image_taps())?;
            let prelocal_surfaces = prepare_current_prelocal_surfaces(device, dimensions, &taps);
            let temporal_dry_surfaces = plan
                .temporal_dry_layers()
                .iter()
                .copied()
                .map(|layer_id| {
                    (
                        layer_id,
                        create_retained_surface(device, dimensions, "Advanced Temporal dry layer"),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            validate_actual_surface_ledger(
                plan,
                retain_program_history,
                &taps,
                prelocal_surfaces.len(),
                temporal_dry_surfaces.len(),
            )?;

            let source_lookup: BTreeMap<_, _> = sources
                .iter()
                .map(|source| (source.stable_id, source))
                .collect();
            let external_effects = plan
                .layers()
                .iter()
                .map(|layer| {
                    let source = source_lookup[&layer.stable_id];
                    (
                        layer.stable_id,
                        host.prepare_effect_source(device, source.view),
                    )
                })
                .collect();

            let ping = host.surface(HostSurface::Ping);
            let pong = host.surface(HostSurface::Pong);
            let group_scratch = host.surface(HostSurface::GroupScratch);
            let ping_effect = host.prepare_effect_source(device, ping.view);
            let group_effect = host.prepare_effect_source(device, group_scratch.view);
            let ping_rack_source = rack.prepare_source(device, ping.view, dimensions)?;
            let rack_zero = rack.surface(1).ok_or_else(|| {
                CompositionGpuError::InvalidSchedule("rack scratch 1 missing".into())
            })?;

            let motion = if let Some(motion_plan) = plan.motion().advanced() {
                let field_specs = motion_plan
                    .fields()
                    .iter()
                    .map(|field| MotionGpuFieldSpec {
                        grid: field.grid,
                        requires_luma: matches!(
                            field.source.origin,
                            MotionFieldOrigin::Lattice | MotionFieldOrigin::LatticeFallback
                        ) || motion_plan.scope(field.scope).is_some_and(|scope| {
                            scope.params.field_source == crate::motion::MotionFieldSource::Auto
                        }),
                        procedural: field.source.origin.procedural_kind(),
                        required_as_garden_signal: field.required_as_garden_signal,
                    })
                    .collect::<Vec<_>>();
                let prepared_resources = MotionGpuResources::prepare(
                    device,
                    motion_plan.resources(),
                    &field_specs,
                    dimensions,
                    motion_plan.collider(),
                )?;
                if let Some(mut resources) = prepared_resources {
                    let field_sources = motion_plan
                        .fields()
                        .iter()
                        .filter(|field| {
                            matches!(
                                field.source.origin,
                                MotionFieldOrigin::Lattice | MotionFieldOrigin::LatticeFallback
                            ) || motion_plan.scope(field.scope).is_some_and(|scope| {
                                scope.params.field_source == crate::motion::MotionFieldSource::Auto
                            }) || field
                                .source
                                .origin
                                .procedural_kind()
                                .is_some_and(|kind| kind.reads_image())
                        })
                        .map(|field| {
                            let view = match field.scope {
                                VisualScopeId::Master => ping.view,
                                VisualScopeId::Layer(layer_id) => source_lookup
                                    .get(&layer_id)
                                    .map(|source| source.view)
                                    .ok_or(CompositionGpuError::MissingSource(layer_id))?,
                                VisualScopeId::Group(_) | VisualScopeId::Program => {
                                    return Err(CompositionGpuError::Motion(
                                        "group motion fields are not admitted in algorithm v1"
                                            .into(),
                                    ));
                                }
                            };
                            Ok(MotionGpuFieldSource {
                                slot: field.slot,
                                view,
                            })
                        })
                        .collect::<Result<Vec<_>, CompositionGpuError>>()?;
                    let scope_specs = motion_plan
                        .scopes()
                        .iter()
                        .filter_map(|scope| {
                            let effect_active =
                                scope.transplant_admitted || !scope.params.shutter.is_exact_zero();
                            if !effect_active {
                                return None;
                            }
                            // A collider recipient advects from the derived
                            // field and therefore needs no primitive slot of
                            // its own; every other scope still requires one.
                            let render_field_slot = if scope.collider_admitted {
                                scope.admitted_field_slot().unwrap_or_default()
                            } else {
                                scope.admitted_field_slot()?
                            };
                            Some(MotionGpuScopeSpec {
                                scope: scope.scope,
                                render_field_slot,
                                uses_carrier: scope.transplant_admitted,
                                derived_collider: scope.collider_admitted,
                            })
                        })
                        .collect::<Vec<_>>();
                    resources.prepare_composition_bindings(
                        device,
                        queue,
                        motion_plan,
                        &field_sources,
                        &scope_specs,
                        ping.view,
                        pong.view,
                    )?;
                    Some(resources)
                } else {
                    None
                }
            } else {
                None
            };

            let composite_a =
                host.prepare_composite_inputs(device, host.surface(HostSurface::A).view, ping.view);
            let composite_b =
                host.prepare_composite_inputs(device, host.surface(HostSurface::B).view, ping.view);
            let composite_program = host.prepare_composite_inputs(
                device,
                host.surface(HostSurface::Program).view,
                ping.view,
            );
            let composite_group =
                host.prepare_composite_inputs(device, group_scratch.view, ping.view);
            let prefix_group_composite =
                host.prepare_composite_inputs(device, ping.view, group_scratch.view);

            let mut matte_bindings = BTreeMap::new();
            for tap in plan.image_taps() {
                if !matches!(
                    tap.consumer,
                    ImageTapConsumer::LayerMatte { .. } | ImageTapConsumer::GroupMatte { .. }
                ) {
                    continue;
                }
                let tap_index = plan
                    .image_taps()
                    .iter()
                    .position(|candidate| candidate.consumer == tap.consumer)
                    .expect("tap is in its source slice");
                let base = match tap.consumer {
                    ImageTapConsumer::LayerMatte { layer_id } => {
                        let layer = layer_plan(plan, layer_id)?;
                        if layer.group_id.is_some() {
                            group_scratch.view
                        } else {
                            host.surface(bus_surface(layer.bus)).view
                        }
                    }
                    ImageTapConsumer::GroupMatte { .. } => rack_zero.view,
                    ImageTapConsumer::RackNode { .. } | ImageTapConsumer::RefreshGardenMatte => {
                        unreachable!()
                    }
                };
                matte_bindings.insert(
                    tap.consumer,
                    std::array::from_fn(|read_index| {
                        let donor = prepared_tap_view(
                            &host,
                            &prelocal_surfaces,
                            rack_zero.view,
                            gesture_canvas.view.as_ref(),
                            program_tap.view.as_ref(),
                            &taps[tap_index],
                            read_index,
                        );
                        host.prepare_matte_inputs(device, base, ping.view, donor)
                    }),
                );
            }
            // Amount-zero group mattes deliberately have no active graph tap,
            // yet their dry path remains an authored ordered step.
            for group in plan.groups() {
                if group.matte.is_some() {
                    let consumer = ImageTapConsumer::GroupMatte { group_id: group.id };
                    matte_bindings.entry(consumer).or_insert_with(|| {
                        std::array::from_fn(|_| {
                            host.prepare_matte_inputs(
                                device,
                                rack_zero.view,
                                ping.view,
                                rack_zero.view,
                            )
                        })
                    });
                }
            }

            let rack_segments = prepare_rack_segments(
                device,
                plan,
                &host,
                &rack,
                &taps,
                &prelocal_surfaces,
                gesture_canvas.view.as_ref(),
                program_tap.view.as_ref(),
            )?;
            let symmetry_slot_count = symmetry_field_step_count(plan);
            let symmetry = if symmetry_slot_count == 0 {
                None
            } else {
                Some(SymmetryFieldExecutor::new(
                    device,
                    queue,
                    dimensions,
                    symmetry_slot_count,
                )?)
            };
            let symmetry_fields = match &symmetry {
                Some(executor) => prepare_symmetry_fields(
                    device,
                    plan,
                    &host,
                    executor,
                    motion.as_ref(),
                    &taps,
                    &prelocal_surfaces,
                    rack_zero.view,
                    gesture_canvas.view.as_ref(),
                    program_tap.view.as_ref(),
                    [ping.view, pong.view],
                )?,
                None => Vec::new(),
            };
            let study_slot_count = study_field_step_count(plan);
            let study = (study_slot_count > 0).then(|| {
                StudyGpuExecutor::new(
                    device,
                    crate::renderer::composition_host::HOST_WORKING_FORMAT,
                    study_slot_count as u32,
                )
            });
            let study_fields = match &study {
                Some(executor) => prepare_study_fields(
                    device,
                    queue,
                    plan,
                    executor,
                    ping.view,
                    host.temporal_history_view(),
                ),
                None => Vec::new(),
            };
            let scan_slot_count = scan_processor_field_step_count(plan);
            let scan = (scan_slot_count > 0).then(|| {
                ScanProcessorGpuExecutor::new(
                    device,
                    crate::renderer::composition_host::HOST_WORKING_FORMAT,
                    scan_slot_count as u32,
                    dimensions,
                )
            });
            let scan_fields = match &scan {
                Some(executor) => prepare_scan_processor_fields(device, plan, executor, ping.view),
                None => Vec::new(),
            };
            let (corruption_slot_count, corruption_has_dct) = corruption_field_totals(plan);
            let corruption = (corruption_slot_count > 0).then(|| {
                CorruptionGpuExecutor::new(
                    device,
                    crate::renderer::composition_host::HOST_WORKING_FORMAT,
                    corruption_slot_count as u32,
                    dimensions,
                    corruption_has_dct,
                )
            });
            let corruption_fields = match &corruption {
                Some(executor) => prepare_corruption_fields(device, plan, executor, ping.view),
                None => Vec::new(),
            };
            let retained_scope_sources = retained_scope_sources(plan, &taps);
            let (root_schedule, member_schedules) =
                build_block_schedules(plan, &retained_scope_sources)?;
            let temporal = host.prepare_temporal_input(device, ping.view, pong.view);
            let garden_signal = plan.refresh_garden_signal();
            let garden_resources = plan.refresh_garden_resources();
            if garden_signal.is_routed()
                != (garden_resources.full_frame_passes == 1
                    && garden_resources.max_sampled_textures_in_pass == 3)
            {
                return Err(CompositionGpuError::InvalidSchedule(
                    "routed Garden signal and immutable resource plan disagree".into(),
                ));
            }
            let routed_garden = match garden_signal {
                EvaluatedRefreshGardenSignalPlan::Inline => None,
                EvaluatedRefreshGardenSignalPlan::Matte { valid } => {
                    let signal = valid
                        .then(|| {
                            tap_index_for_consumer(plan, ImageTapConsumer::RefreshGardenMatte).map(
                                |tap_index| {
                                    prepared_tap_view(
                                        &host,
                                        &prelocal_surfaces,
                                        rack_zero.view,
                                        gesture_canvas.view.as_ref(),
                                        program_tap.view.as_ref(),
                                        &taps[tap_index],
                                        0,
                                    )
                                },
                            )
                        })
                        .flatten()
                        .unwrap_or(rack_zero.view);
                    Some(host.prepare_routed_garden_input(device, pong.view, signal))
                }
                EvaluatedRefreshGardenSignalPlan::Motion { valid, .. } => {
                    let signal = valid
                        .then(|| {
                            motion
                                .as_ref()
                                .and_then(MotionGpuResources::garden_signal_view)
                        })
                        .flatten()
                        .unwrap_or(rack_zero.view);
                    Some(host.prepare_routed_garden_input(device, pong.view, signal))
                }
            };
            let bus_inputs = host.prepare_bus_inputs(
                device,
                host.surface(HostSurface::A).view,
                host.surface(HostSurface::B).view,
                host.surface(HostSurface::Program).view,
            );
            let present = host.prepare_copy_source(device, ping.view);

            // Defined initialization is recorded once during preparation. It
            // does not publish any history-valid bit.
            let mut initialize = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Advanced composition retained-surface initialization"),
            });
            host.encode_clear(&mut initialize, rack_zero.view, wgpu::Color::TRANSPARENT);
            for tap in &taps {
                match &tap.backing {
                    TapBacking::Current(surface) => {
                        host.encode_clear(&mut initialize, &surface.view, wgpu::Color::TRANSPARENT);
                    }
                    TapBacking::Previous { surfaces, .. } => {
                        for surface in surfaces {
                            host.encode_clear(
                                &mut initialize,
                                &surface.view,
                                wgpu::Color::TRANSPARENT,
                            );
                        }
                    }
                    TapBacking::Transparent
                    | TapBacking::ProgramHistory
                    | TapBacking::GestureCanvas
                    | TapBacking::ProgramTap
                    | TapBacking::CurrentPreLocal { .. } => {}
                }
            }
            for surface in prelocal_surfaces.values() {
                host.encode_clear(&mut initialize, &surface.view, wgpu::Color::TRANSPARENT);
            }
            for surface in temporal_dry_surfaces.values() {
                host.encode_clear(&mut initialize, &surface.view, wgpu::Color::TRANSPARENT);
            }
            queue.submit(Some(initialize.finish()));

            let retained_textures = taps
                .iter()
                .map(|tap| match tap.backing {
                    TapBacking::Current(_) => 1,
                    TapBacking::Previous { .. } => 2,
                    _ => 0,
                })
                .sum::<u64>()
                + prelocal_surfaces.len() as u64
                + temporal_dry_surfaces.len() as u64;
            Ok(Self {
                topology_signature: plan.topology_signature(),
                source_keys,
                gesture_canvas_identity: gesture_canvas.identity(),
                gesture_canvas_bound: gesture_canvas.is_bound(),
                program_tap_identity: program_tap.identity(),
                program_tap_bound: program_tap.is_bound(),
                host,
                rack,
                symmetry,
                symmetry_fields,
                study,
                study_fields,
                scan,
                scan_fields,
                corruption,
                corruption_fields,
                motion,
                taps: taps.into_boxed_slice(),
                prelocal_surfaces,
                temporal_dry_surfaces,
                root_schedule,
                member_schedules,
                external_effects,
                ping_effect,
                group_effect,
                ping_rack_source,
                rack_segments,
                composite_a,
                composite_b,
                composite_program,
                composite_group,
                prefix_group_composite,
                matte_bindings,
                temporal,
                routed_garden,
                bus_inputs,
                device: device.clone(),
                bus_melt: None,
                bus_melt_valid: false,
                bus_melt_staged: false,
                bus_melt_tick_debt: 0.0,
                bus_melt_staged_debt: 0.0,
                present,
                effect_slots,
                exact_layer_slots,
                prelocal_slots,
                composite_slots,
                prefix_composite_slot,
                matte_slots,
                tap_history_read_index: 0,
                history_staged: false,
                retained_textures,
                retained_views: retained_textures,
                executor_bindings: 0,
                creative_bytes: plan.resources().creative_bytes,
                motion_bytes: plan
                    .motion()
                    .advanced()
                    .map_or(0, |motion| motion.resources().total_bytes),
            })
        })();
        let scope_error = [
            ("out of memory", pollster::block_on(out_of_memory.pop())),
            ("internal/backend", pollster::block_on(internal.pop())),
            ("validation", pollster::block_on(validation.pop())),
        ]
        .into_iter()
        .find_map(|(kind, error)| error.map(|error| format!("{kind}: {error}")));
        match (built, scope_error) {
            (Err(error), _) => Err(error),
            (Ok(_), Some(message)) => Err(CompositionGpuError::ResourceCreation(message)),
            (Ok(prepared), None) => Ok(prepared),
        }
    }

    fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        timing: CompositionFrameTiming,
        motion_input: CompositionMotionFrameInput<'_>,
        scope_readback: &mut Option<PreparedScopeRecorderReadback>,
    ) -> Result<(), CompositionGpuError> {
        if self.topology_signature != plan.topology_signature() {
            return Err(CompositionGpuError::TopologyNotPrepared {
                prepared: Some(self.topology_signature),
                requested: plan.topology_signature(),
            });
        }
        self.discard_frame_history();
        // The bus melt's lazy surface: the first armed frame allocates once
        // (deliberately before the warm-allocation snapshot), a warmed armed
        // frame allocates nothing, and disarming invalidates the trail
        // without freeing the surface.
        let bus_melt_armed = plan.mixer().melt.is_armed();
        if bus_melt_armed && self.bus_melt.is_none() {
            let surface = create_retained_surface(
                &self.device,
                self.host.dimensions(),
                "Advanced composition bus melt history",
            );
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Advanced composition bus melt history group"),
                layout: self.host.bus_history_layout(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&surface.view),
                }],
            });
            self.retained_textures += 1;
            self.retained_views += 1;
            self.executor_bindings += 1;
            self.bus_melt = Some((surface, group));
        }
        if !bus_melt_armed {
            self.bus_melt_valid = false;
        }
        // B6 avalanche histories: the same lazy law per node — the first
        // armed frame allocates once (before the warm-allocation snapshot),
        // a warmed armed frame allocates nothing, and disarming invalidates
        // the trail without freeing the surface. Blackout never clears them:
        // the cascade is program memory on the melt precedent.
        let mut armed_avalanche: Vec<(VisualScopeId, NodeId)> = Vec::new();
        for_each_corruption_step(plan, |scope, field| {
            if matches!(field.kind, EvaluatedCorruptionKind::Avalanche(_)) && field.is_active() {
                armed_avalanche.push((scope, field.node_id));
            }
        });
        for field in &mut self.corruption_fields {
            let armed = armed_avalanche
                .iter()
                .any(|(scope, node)| *scope == field.scope && *node == field.node_id);
            if armed && field.history.is_none() {
                if let Some(executor) = &self.corruption {
                    let surface = create_retained_surface(
                        &self.device,
                        self.host.dimensions(),
                        "Advanced composition avalanche history",
                    );
                    let bind_group = executor.create_bind_group(
                        &self.device,
                        self.host.surface(HostSurface::Ping).view,
                        &surface.view,
                    );
                    self.retained_textures += 1;
                    self.retained_views += 1;
                    self.executor_bindings += 1;
                    field.history = Some(AvalancheHistory {
                        surface,
                        bind_group,
                    });
                }
            }
            if !armed {
                field.history_valid = false;
            }
        }
        let allocations_before = self.allocation_snapshot();
        self.write_uniforms(queue, plan)?;
        if let (Some(resources), Some(motion_plan)) = (&mut self.motion, plan.motion().advanced()) {
            resources.begin_frame(
                queue,
                motion_plan,
                timing.temporal_input(),
                MotionFrameInput {
                    attachments: motion_input.attachments,
                    held_scopes: motion_input.held_scopes,
                },
                plan.base().context().time_seconds,
            )?;
            for field in motion_plan
                .fields()
                .iter()
                .filter(|field| field.scope != VisualScopeId::Master)
            {
                resources.encode_field_scope(encoder, motion_plan, field.scope)?;
            }
            // Primitive fields are staged and encoded first, the derived
            // collided field second, and the carrier is advected third. The
            // collider therefore reads this frame's primitive vectors, and
            // every advection below reads this frame's derived field.
            resources.encode_collider(encoder)?;
            resources.encode_garden_signal(encoder)?;
        }

        for surface in [
            HostSurface::A,
            HostSurface::B,
            HostSurface::Program,
            HostSurface::GroupScratch,
        ] {
            self.host.encode_clear(
                encoder,
                self.host.surface(surface).view,
                wgpu::Color::TRANSPARENT,
            );
        }
        let rack_zero = self
            .rack
            .surface(1)
            .ok_or_else(|| CompositionGpuError::InvalidSchedule("rack scratch 1 missing".into()))?;
        self.host
            .encode_clear(encoder, rack_zero.view, wgpu::Color::TRANSPARENT);

        self.stage_previous_prelocal(queue, encoder, plan)?;
        if let Some(tap_index) = self.tap_index(ImageTapConsumer::RefreshGardenMatte) {
            // Routed Garden is a master host-boundary consumer rather than a
            // rack node, so it must explicitly materialize a selected
            // PreLocal source. PostLocal sources are staged by the ordinary
            // scope-output capture below.
            self.ensure_current_prelocal(encoder, tap_index)?;
        }
        self.capture_root_prefix(queue, encoder, plan, 0)?;

        let mut completed_root_items = 0_usize;
        for schedule_index in 0..self.root_schedule.len() {
            let task = self.root_schedule[schedule_index].task;
            match task {
                RootTask::Layer(layer_id) => {
                    self.execute_layer(queue, encoder, plan, layer_id)?;
                    self.capture_scope_output(
                        encoder,
                        VisualScopeId::Layer(layer_id),
                        HostSurface::Ping,
                        scope_readback,
                    )?;
                    if plan.is_temporal_dry_layer(layer_id) {
                        self.retain_temporal_dry_layer(encoder, layer_id)?;
                    }
                }
                RootTask::Group(group_id) => {
                    self.execute_group(
                        queue,
                        encoder,
                        plan,
                        group_id,
                        completed_root_items,
                        scope_readback,
                    )?;
                }
            }

            for drain_index in 0..self.root_schedule[schedule_index].drains.len() {
                let admission = self.root_schedule[schedule_index].drains[drain_index];
                match admission.item {
                    RootTask::Layer(layer_id) => {
                        if !plan.is_temporal_dry_layer(layer_id) {
                            self.load_scheduled_source(encoder, admission.source)?;
                            self.admit_layer(queue, encoder, plan, layer_id, None)?;
                        }
                    }
                    RootTask::Group(group_id) => {
                        self.load_scheduled_source(encoder, admission.source)?;
                        self.admit_group(queue, encoder, plan, group_id)?;
                    }
                }
                completed_root_items += 1;
                self.capture_root_prefix(queue, encoder, plan, completed_root_items)?;
            }
        }

        self.host.encode_bus(
            queue,
            encoder,
            &self.bus_inputs,
            self.host.surface(HostSurface::Ping).view,
            &BusFrameParams {
                history_valid: self.bus_melt_valid,
                ..bus_frame_params(plan)
            },
            self.bus_melt
                .as_ref()
                .filter(|_| bus_melt_armed)
                .map(|(_, group)| group),
        );
        // Store the bus output as the melt's N-1 memory — at most one melt
        // step per frame, advanced on the stage's own 30 Hz reference
        // accumulator so live and export creep at the same rate and Pause
        // holds the trail still. Validity publishes only on the
        // frame-history commit, so a discarded frame leaves no stale bit.
        self.bus_melt_staged_debt = self.bus_melt_tick_debt;
        self.bus_melt_staged = false;
        if bus_melt_armed {
            let debt = (self.bus_melt_tick_debt
                + timing.temporal_input().program_advancing_delta()
                    * crate::effects::params::TEMPORAL_REFERENCE_FPS)
                .min(2.0);
            if debt >= 1.0 {
                if let Some((surface, _)) = &self.bus_melt {
                    copy_texture(
                        encoder,
                        self.host.surface(HostSurface::Ping).texture,
                        &surface.texture,
                        self.host.dimensions(),
                    );
                    self.bus_melt_staged = true;
                }
                self.bus_melt_staged_debt = (debt - 1.0).min(1.0);
            } else {
                self.bus_melt_staged_debt = debt;
            }
        }
        self.execute_master(queue, encoder, plan, timing)?;
        if let (Some(resources), Some(motion_plan)) = (&self.motion, plan.motion().advanced()) {
            resources.encode_field_scope(encoder, motion_plan, VisualScopeId::Master)?;
        }
        self.execute_motion_scope(encoder, plan, VisualScopeId::Master)?;
        self.capture_scope_output(
            encoder,
            VisualScopeId::Master,
            HostSurface::Ping,
            scope_readback,
        )?;

        // Every non-Program Previous tap writes the inactive parity directly.
        // Publishing is a CPU parity swap in `commit_frame_history`; an
        // abandoned/submitted-but-rejected frame therefore cannot overwrite
        // the committed image read by the next accepted frame.
        if self
            .taps
            .iter()
            .any(|tap| matches!(tap.backing, TapBacking::Previous { staged: false, .. }))
        {
            return Err(CompositionGpuError::InvalidSchedule(
                "a previous-frame tap producer did not stage this frame".into(),
            ));
        }
        self.host
            .encode_stage_program_history(encoder, self.host.surface(HostSurface::Program).texture);
        self.history_staged = true;

        if allocations_before != self.allocation_snapshot() {
            return Err(CompositionGpuError::InvalidSchedule(
                "warmed encode changed the GPU allocation snapshot".into(),
            ));
        }
        Ok(())
    }

    fn retain_temporal_dry_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer_id: StableLayerId,
    ) -> Result<(), CompositionGpuError> {
        let surface = self.temporal_dry_surfaces.get(&layer_id).ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(format!(
                "Temporal dry layer {} reached encode without a retained surface",
                layer_id.get()
            ))
        })?;
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Ping).texture,
            &surface.texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    /// Objects owned by the dedicated Symmetry Field executor. They are folded
    /// into the rack object count because both are creative-pass executors; the
    /// dedicated one contributes no bytes to the surface ledger, so
    /// `creative_bytes` is unchanged by its presence.
    fn symmetry_objects(&self) -> u64 {
        self.symmetry.as_ref().map_or(0, |executor| {
            let snapshot = executor.allocation_snapshot();
            snapshot.textures + snapshot.buffers + snapshot.bind_groups + snapshot.pipelines
        })
    }

    fn allocation_snapshot(&self) -> CompositionAllocationSnapshot {
        CompositionAllocationSnapshot {
            host_objects: self.host.allocation_snapshot().total(),
            rack_objects: self.rack.allocation_snapshot().total() + self.symmetry_objects(),
            retained_textures: self.retained_textures,
            retained_views: self.retained_views,
            executor_bindings: self.executor_bindings,
            readback_objects: 0,
            readback_bytes: 0,
            readback_staging_bytes: 0,
            readback_texture_bytes: 0,
            creative_bytes: self.creative_bytes,
            motion_bytes: self.motion_bytes,
        }
    }

    fn commit_frame_history(&mut self) {
        if !self.history_staged {
            return;
        }
        self.host.commit_frame_history();
        self.tap_history_read_index = 1 - self.tap_history_read_index;
        if self.host.program_history_views().is_some() {
            debug_assert_eq!(
                self.host.program_history_read_index(),
                self.tap_history_read_index
            );
        }
        for tap in &mut self.taps {
            if let TapBacking::Previous {
                initialized,
                staged,
                ..
            } = &mut tap.backing
            {
                if *staged {
                    *initialized = true;
                    *staged = false;
                }
            }
        }
        if let Some(motion) = &mut self.motion {
            motion.commit_frame();
        }
        self.bus_melt_tick_debt = self.bus_melt_staged_debt;
        if self.bus_melt_staged {
            self.bus_melt_valid = true;
            self.bus_melt_staged = false;
        }
        for field in &mut self.corruption_fields {
            field.last_store_tick = field.staged_store_tick;
            if field.history_staged {
                field.history_valid = true;
                field.history_staged = false;
            }
        }
        self.history_staged = false;
    }

    fn discard_frame_history(&mut self) {
        self.host.discard_frame_history();
        if let Some(motion) = &mut self.motion {
            motion.discard_frame();
        }
        for tap in &mut self.taps {
            if let TapBacking::Previous { staged, .. } = &mut tap.backing {
                *staged = false;
            }
        }
        self.bus_melt_staged = false;
        self.bus_melt_staged_debt = self.bus_melt_tick_debt;
        for field in &mut self.corruption_fields {
            field.history_staged = false;
            field.staged_store_tick = field.last_store_tick;
        }
        self.history_staged = false;
    }

    fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        plan: &AdvancedCompositionPlan,
    ) -> Result<(), CompositionGpuError> {
        for layer in plan.layers() {
            let base_index = layer.base_layer_index;
            self.host.write_effect_uniform(
                queue,
                self.prelocal_slots[&layer.stable_id],
                &plan.base().layer_pre_passes()[base_index],
            )?;
            if let Some(slot) = self.exact_layer_slots.get(&layer.stable_id) {
                self.host.write_effect_uniform(
                    queue,
                    *slot,
                    &plan.base().layer_passes()[base_index],
                )?;
            }
            self.write_execution_effect_uniforms(
                queue,
                VisualScopeId::Layer(layer.stable_id),
                &layer.execution,
            )?;
            let evaluated = &plan.base().layers()[base_index];
            self.host.write_composite_uniform(
                queue,
                self.composite_slots[&VisualScopeId::Layer(layer.stable_id)],
                &HostCompositeUniforms::new(evaluated.opacity, evaluated.blend_mode),
            )?;
            if let Some(matte) = layer.legacy_matte {
                let consumer = ImageTapConsumer::LayerMatte {
                    layer_id: layer.stable_id,
                };
                let valid = self
                    .tap_index(consumer)
                    .is_some_and(|index| self.tap_ready(index));
                let mut params = matte.params;
                params.donor_valid = valid;
                self.host.write_matte_uniform(
                    queue,
                    self.matte_slots[&consumer],
                    &MatteCompositeUniforms::new(
                        evaluated.opacity,
                        evaluated.blend_mode.as_u32(),
                        params,
                    ),
                )?;
            }
        }
        for group in plan.groups() {
            self.write_execution_effect_uniforms(
                queue,
                VisualScopeId::Group(group.id),
                &group.execution,
            )?;
            self.host.write_composite_uniform(
                queue,
                self.composite_slots[&VisualScopeId::Group(group.id)],
                &HostCompositeUniforms::new(group.opacity, BlendMode::Normal),
            )?;
            if let Some(matte) = group.matte {
                let consumer = ImageTapConsumer::GroupMatte { group_id: group.id };
                let valid = self
                    .tap_index(consumer)
                    .is_some_and(|index| self.tap_ready(index));
                self.host.write_matte_uniform(
                    queue,
                    self.matte_slots[&consumer],
                    &MatteCompositeUniforms::new(
                        1.0,
                        BlendMode::Normal.as_u32(),
                        runtime_matte_params(matte, valid),
                    ),
                )?;
            }
        }
        self.write_execution_effect_uniforms(
            queue,
            VisualScopeId::Master,
            &plan.master().execution,
        )?;
        self.host.write_composite_uniform(
            queue,
            self.prefix_composite_slot,
            &HostCompositeUniforms::new(1.0, BlendMode::Normal),
        )?;
        Ok(())
    }

    fn write_execution_effect_uniforms(
        &self,
        queue: &wgpu::Queue,
        scope: VisualScopeId,
        execution: &EvaluatedScopeExecution,
    ) -> Result<(), CompositionGpuError> {
        for (index, step) in execution.steps().iter().enumerate() {
            let pass = match step {
                EvaluatedScopeStep::MaterializeSpatial { pass, .. }
                | EvaluatedScopeStep::LegacyCanonical { pass, .. } => pass,
                EvaluatedScopeStep::CollisionRack { .. }
                | EvaluatedScopeStep::LegacyTemporal { .. }
                | EvaluatedScopeStep::SymmetryField { .. }
                | EvaluatedScopeStep::StudyField { .. }
                | EvaluatedScopeStep::ScanProcessorField { .. }
                | EvaluatedScopeStep::CorruptionField { .. }
                | EvaluatedScopeStep::GroupMatte { .. } => continue,
            };
            self.host
                .write_effect_uniform(queue, self.effect_slots[&(scope, index)], pass)?;
        }
        Ok(())
    }

    fn tap_index(&self, consumer: ImageTapConsumer) -> Option<usize> {
        self.taps
            .iter()
            .position(|tap| tap.planned.consumer == consumer)
    }

    fn tap_ready(&self, index: usize) -> bool {
        match self.taps[index].backing {
            TapBacking::Transparent => false,
            TapBacking::ProgramHistory => self.host.program_history_initialized(),
            // Ready exactly when the host published a presented field. This is
            // the same binding `prepared_tap_view` reads, recorded at prepare
            // time, so readiness and the bound view cannot disagree.
            TapBacking::GestureCanvas => self.gesture_canvas_bound,
            // The identical law for the programme tap: ready exactly when the
            // host published a committed copy, recorded at prepare time.
            TapBacking::ProgramTap => self.program_tap_bound,
            TapBacking::CurrentPreLocal { .. } | TapBacking::Current(_) => true,
            TapBacking::Previous { initialized, .. } => initialized,
        }
    }

    fn stage_previous_prelocal(
        &mut self,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
    ) -> Result<(), CompositionGpuError> {
        for index in 0..self.taps.len() {
            let layer_id = match self.taps[index].planned.resolved {
                PlannedImageSource::SelectedLayer {
                    layer_id,
                    stage: LayerImageStage::PreLocalEffects,
                } if self.taps[index].planned.origin.timing() == EdgeTiming::PreviousFrame => {
                    layer_id
                }
                _ => continue,
            };
            let write_index = 1 - self.tap_history_read_index;
            let TapBacking::Previous {
                surfaces, staged, ..
            } = &mut self.taps[index].backing
            else {
                return Err(CompositionGpuError::InvalidSchedule(
                    "previous PreLocal tap lacks history staging".into(),
                ));
            };
            self.host.encode_effect(
                encoder,
                &self.external_effects[&layer_id],
                &surfaces[write_index].view,
                self.prelocal_slots[&layer_id],
            )?;
            *staged = true;
            let _ = plan;
        }
        Ok(())
    }

    fn ensure_current_prelocal(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tap_index: usize,
    ) -> Result<(), CompositionGpuError> {
        let PlannedImageSource::SelectedLayer {
            layer_id,
            stage: LayerImageStage::PreLocalEffects,
        } = self.taps[tap_index].planned.resolved
        else {
            return Ok(());
        };
        if self.taps[tap_index].planned.origin.timing() != EdgeTiming::CurrentFrame {
            return Ok(());
        }
        let surface = self.prelocal_surfaces.get(&layer_id).ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(format!(
                "current PreLocal donor {} has no retained surface",
                layer_id.get()
            ))
        })?;
        self.host.encode_effect(
            encoder,
            &self.external_effects[&layer_id],
            &surface.view,
            self.prelocal_slots[&layer_id],
        )?;
        Ok(())
    }

    fn capture_scope_output(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scope: VisualScopeId,
        source: HostSurface,
        scope_readback: &mut Option<PreparedScopeRecorderReadback>,
    ) -> Result<(), CompositionGpuError> {
        let source = self.host.surface(source);
        let dimensions = self.host.dimensions();
        let write_index = 1 - self.tap_history_read_index;
        for tap in &mut self.taps {
            if tap_boundary_scope(&tap.planned) == Some(scope) {
                stage_tap_from_texture(encoder, source.texture, tap, dimensions, write_index);
            }
        }
        if let Some(readback) = scope_readback {
            readback.capture(encoder, &self.host, &self.present, scope)?;
        }
        Ok(())
    }

    fn capture_root_prefix(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        preceding_root_outputs: usize,
    ) -> Result<(), CompositionGpuError> {
        self.capture_prefix_with_crossfade(
            queue,
            encoder,
            CompositePrefix::Root {
                preceding_root_outputs,
            },
            bus_frame_params(plan),
        )
    }

    fn capture_prefix_with_crossfade(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        prefix: CompositePrefix,
        bus_params: BusFrameParams,
    ) -> Result<(), CompositionGpuError> {
        if !self.taps.iter().any(|tap| {
            matches!(tap.planned.resolved, PlannedImageSource::AllBelow(candidate) if candidate == prefix)
        }) {
            return Ok(());
        }
        // A prefix capture runs the same bus law with the same uniforms so
        // an "all below" tap sees a consistent partial state; only the melt
        // hold stays closed, because the history belongs to the main bus
        // output alone.
        self.host.encode_bus(
            queue,
            encoder,
            &self.bus_inputs,
            self.host.surface(HostSurface::Ping).view,
            &BusFrameParams {
                history_valid: false,
                ..bus_params
            },
            None,
        );
        let source_surface = match prefix {
            CompositePrefix::Root { .. } => HostSurface::Ping,
            CompositePrefix::GroupMember { .. } => {
                self.host.encode_composite(
                    encoder,
                    &self.prefix_group_composite,
                    self.host.surface(HostSurface::Pong).view,
                    self.prefix_composite_slot,
                )?;
                HostSurface::Pong
            }
        };
        let source = self.host.surface(source_surface);
        let dimensions = self.host.dimensions();
        let write_index = 1 - self.tap_history_read_index;
        for tap in &mut self.taps {
            if matches!(tap.planned.resolved, PlannedImageSource::AllBelow(candidate) if candidate == prefix)
            {
                stage_tap_from_texture(encoder, source.texture, tap, dimensions, write_index);
            }
        }
        Ok(())
    }

    fn load_scheduled_source(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: ScheduledSource,
    ) -> Result<(), CompositionGpuError> {
        let ScheduledSource::RetainedTap(index) = source else {
            return Ok(());
        };
        let TapBacking::Current(surface) = &self.taps[index].backing else {
            return Err(CompositionGpuError::InvalidSchedule(
                "late structural admission did not reference a current retained tap".into(),
            ));
        };
        copy_texture(
            encoder,
            &surface.texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    fn encode_effect_from_ping(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: HostUniformSlot,
    ) -> Result<(), CompositionGpuError> {
        self.host.encode_effect(
            encoder,
            &self.ping_effect,
            self.host.surface(HostSurface::Pong).view,
            slot,
        )?;
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Pong).texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    fn execute_motion_scope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        scope: VisualScopeId,
    ) -> Result<(), CompositionGpuError> {
        let (Some(resources), Some(motion_plan)) = (&self.motion, plan.motion().advanced()) else {
            return Ok(());
        };
        let scratch = self.rack.surface(0).ok_or_else(|| {
            CompositionGpuError::InvalidSchedule("rack scratch 0 missing for motion".into())
        })?;
        let ping = self.host.surface(HostSurface::Ping);
        let pong = self.host.surface(HostSurface::Pong);
        resources.encode_scope(
            encoder,
            motion_plan,
            scope,
            ping.texture,
            pong.texture,
            pong.view,
            scratch.texture,
            scratch.view,
            self.host.dimensions(),
        )?;
        Ok(())
    }

    fn execute_layer(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        layer_id: StableLayerId,
    ) -> Result<(), CompositionGpuError> {
        let layer = layer_plan(plan, layer_id)?;
        if layer.execution.is_exact_legacy() {
            self.host.encode_effect(
                encoder,
                &self.external_effects[&layer_id],
                self.host.surface(HostSurface::Ping).view,
                self.exact_layer_slots[&layer_id],
            )?;
            self.execute_motion_scope(encoder, plan, VisualScopeId::Layer(layer_id))?;
            return Ok(());
        }
        let scope = VisualScopeId::Layer(layer_id);
        for (index, step) in layer.execution.steps().iter().enumerate() {
            match step {
                EvaluatedScopeStep::MaterializeSpatial { .. } => {
                    if index == 0 {
                        self.host.encode_effect(
                            encoder,
                            &self.external_effects[&layer_id],
                            self.host.surface(HostSurface::Ping).view,
                            self.effect_slots[&(scope, index)],
                        )?;
                    } else {
                        self.encode_effect_from_ping(encoder, self.effect_slots[&(scope, index)])?;
                    }
                }
                EvaluatedScopeStep::LegacyCanonical { .. } => {
                    self.encode_effect_from_ping(encoder, self.effect_slots[&(scope, index)])?;
                }
                EvaluatedScopeStep::CollisionRack {
                    segment_index,
                    plan: rack_plan,
                } => self.execute_rack_segment(
                    queue,
                    encoder,
                    plan,
                    scope,
                    *segment_index,
                    rack_plan,
                )?,
                EvaluatedScopeStep::SymmetryField { plan: field } => {
                    self.execute_symmetry_field(queue, encoder, plan, scope, field)?;
                }
                EvaluatedScopeStep::StudyField { plan: field } => {
                    self.execute_study_field(queue, encoder, plan, scope, field)?;
                }
                EvaluatedScopeStep::ScanProcessorField { plan: field } => {
                    self.execute_scan_processor_field(queue, encoder, plan, scope, field)?;
                }
                EvaluatedScopeStep::CorruptionField { plan: field } => {
                    self.execute_corruption_field(queue, encoder, plan, scope, field)?;
                }
                EvaluatedScopeStep::LegacyTemporal { .. }
                | EvaluatedScopeStep::GroupMatte { .. } => {
                    return Err(CompositionGpuError::InvalidSchedule(format!(
                        "layer {} contains a non-layer host boundary",
                        layer_id.get()
                    )));
                }
            }
        }
        self.execute_motion_scope(encoder, plan, scope)?;
        Ok(())
    }

    /// Encode one dedicated Symmetry Field step.
    ///
    /// The step's contract is the same as every other ordered step's: read the
    /// scope's carrier out of Ping and leave the result in Ping. The dedicated
    /// executor owns no surface of its own, so the pass renders into the
    /// existing Pong scratch and is copied back, exactly as
    /// `encode_effect_from_ping` does.
    fn execute_symmetry_field(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        scope: VisualScopeId,
        field: &EvaluatedSymmetryFieldPlan,
    ) -> Result<(), CompositionGpuError> {
        // A dormant or exactly-bypassed node is a real delegation: Ping already
        // holds the carrier, so nothing is encoded and nothing is copied.
        if SymmetryFieldExecutor::is_inert(field) {
            return Ok(());
        }
        let prepared_index = self
            .symmetry_fields
            .iter()
            .position(|prepared| prepared.scope == scope && prepared.node_id == field.node_id)
            .ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "Symmetry Field node {} in {scope:?} was not prepared",
                    field.node_id.get()
                ))
            })?;
        let prelocal = self.symmetry_fields[prepared_index]
            .tap_indices
            .iter()
            .find_map(|(_, tap_index)| {
                matches!(
                    self.taps[*tap_index].backing,
                    TapBacking::CurrentPreLocal { .. }
                )
                .then_some(*tap_index)
            });
        if let Some(tap_index) = prelocal {
            self.ensure_current_prelocal(encoder, tap_index)?;
        }
        let read_index = self.tap_history_read_index;
        for binding_index in 0..self.symmetry_fields[prepared_index].tap_indices.len() {
            let (slot, tap_index) = self.symmetry_fields[prepared_index].tap_indices[binding_index];
            let ready = self.tap_ready(tap_index);
            let updated = self.symmetry_fields[prepared_index].bindings[read_index]
                .set_donor_ready(field.node_id, slot, ready);
            debug_assert!(updated);
        }
        // Each motion slot reads the parity its own field committed THIS frame,
        // the same `render_field_index` motion rendering wrote through, so a
        // routed consumer can never observe the stale half of a ping/pong pair.
        // A slot whose field has no staged frame, or whose committed parity is
        // not materialized yet, reports not-ready: the packed record's validity
        // lane closes and the shader decodes exactly zero displacement while the
        // prebuilt group stays untouched.
        let committed: [Option<MotionFieldReadParity>; SYMMETRY_MOTION_SLOTS] =
            std::array::from_fn(|slot| {
                let field_slot = self.symmetry_fields[prepared_index].motion_field_slots[slot]?;
                self.motion.as_ref()?.field_read_parity(field_slot)
            });
        let motion_parity: [usize; SYMMETRY_MOTION_SLOTS] =
            std::array::from_fn(|slot| committed[slot].map_or(0, |parity| parity.index));
        for (slot, parity) in committed.into_iter().enumerate() {
            let updated = self.symmetry_fields[prepared_index]
                .motion_bindings
                .set_motion_ready(
                    field.node_id,
                    slot,
                    parity.is_some_and(|parity| parity.valid),
                );
            debug_assert!(updated);
        }
        let executor = self.symmetry.as_ref().ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(
                "a Symmetry Field step was planned without its dedicated executor".into(),
            )
        })?;
        let (history_write, history_valid) = self.host.temporal_history_read_cursor();
        let _report = executor.encode_at(
            queue,
            encoder,
            field,
            &self.symmetry_fields[prepared_index].bindings[read_index],
            &self.symmetry_fields[prepared_index].motion_bindings,
            // Parity 0 is Ping, which is where every ordered step leaves its
            // result and therefore where this step finds its carrier.
            0,
            motion_parity,
            self.host.surface(HostSurface::Pong).view,
            self.symmetry_fields[prepared_index].uniform_slot,
            SymmetryHistoryCursor {
                write_index: history_write,
                valid: history_valid,
            },
            plan.base().context().time_seconds,
        )?;
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Pong).texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    /// Encode one dedicated Study interpreter step. Same contract as every
    /// ordered step: carrier out of Ping, result back in Ping via the Pong
    /// scratch. An inert plan — disabled, dry, or unresolved digest — is a
    /// real delegation: nothing is encoded and nothing is copied.
    fn execute_study_field(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        scope: VisualScopeId,
        field: &EvaluatedStudyPlan,
    ) -> Result<(), CompositionGpuError> {
        if StudyGpuExecutor::is_inert(field) {
            return Ok(());
        }
        let prepared = self
            .study_fields
            .iter()
            .find(|prepared| prepared.scope == scope && prepared.node_id == field.node_id)
            .ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "Study node {} in {scope:?} was not prepared",
                    field.node_id.get()
                ))
            })?;
        let executor = self.study.as_ref().ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(
                "a Study step was planned without its dedicated executor".into(),
            )
        })?;
        // Frame inputs: audio and beat from the shared frame-plan context
        // (the same immutable sample live, frame-derived offline), ring
        // validity from the committed cursor the temporal pass consumes.
        let (history_write, history_valid) = self.host.temporal_history_read_cursor();
        let context = plan.base().context();
        let frame = StudyGpuFrameUniforms::from_parts(
            &crate::study_eval::StudyFrameContext {
                audio_bands: context.study_audio_bands,
                beat_phase: context.study_beat_phase,
                valid_history: history_valid,
            },
            field.instruction_count,
            history_write,
            crate::temporal::TEMPORAL_HISTORY_LEN,
            field.wet,
            field.blend,
        );
        executor.write_frame(queue, prepared.uniform_slot, &frame);
        executor.encode_at(
            encoder,
            &prepared.bind_group,
            self.host.surface(HostSurface::Pong).view,
            prepared.uniform_slot,
        );
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Pong).texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    /// Encode one dedicated Scan Processor step: the instanced ribbon
    /// geometry pass into the shared accumulator, the fullscreen resolve
    /// into Pong, then the established Pong→Ping copy. An inert plan —
    /// disabled, dry, or no deflection authored — is a real delegation:
    /// nothing is encoded and nothing is copied.
    fn execute_scan_processor_field(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        scope: VisualScopeId,
        field: &EvaluatedScanProcessorPlan,
    ) -> Result<(), CompositionGpuError> {
        if ScanProcessorGpuExecutor::is_inert(field) {
            return Ok(());
        }
        let prepared = self
            .scan_fields
            .iter()
            .find(|prepared| prepared.scope == scope && prepared.node_id == field.node_id)
            .ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "Scan Processor node {} in {scope:?} was not prepared",
                    field.node_id.get()
                ))
            })?;
        let executor = self.scan.as_ref().ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(
                "a Scan Processor step was planned without its dedicated executor".into(),
            )
        })?;
        // The only time input is the shared frame-plan seconds — the same
        // immutable sample live, frame-derived offline — so Pause holds the
        // detuned oscillator still and export replays it structurally.
        let frame = ScanProcessorGpuUniforms::from_parts(
            &field.params,
            self.host.dimensions(),
            plan.base().context().time_seconds,
            field.wet,
            field.blend,
        );
        executor.write_frame(queue, prepared.uniform_slot, &frame);
        executor.encode_at(
            encoder,
            &prepared.geometry_bind_group,
            &prepared.resolve_bind_group,
            self.host.surface(HostSurface::Pong).view,
            prepared.uniform_slot,
            &field.params,
        );
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Pong).texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    /// Encode one dedicated B6 corruption step: the Block DCT's four-pass
    /// separable pipeline through the shared intermediates, the Pixel Sort
    /// pass, or the Filter Avalanche pass over its retained history. Same
    /// step contract as every dedicated pass: read Ping, land in Ping.
    fn execute_corruption_field(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        scope: VisualScopeId,
        field: &EvaluatedCorruptionPlan,
    ) -> Result<(), CompositionGpuError> {
        if CorruptionGpuExecutor::is_inert(field) {
            return Ok(());
        }
        let index = self
            .corruption_fields
            .iter()
            .position(|prepared| prepared.scope == scope && prepared.node_id == field.node_id)
            .ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "corruption node {} in {scope:?} was not prepared",
                    field.node_id.get()
                ))
            })?;
        let executor = self.corruption.as_ref().ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(
                "a corruption step was planned without its dedicated executor".into(),
            )
        })?;
        // The only time input is the shared frame-plan seconds, so Pause
        // holds the avalanche's fault stream and export replays it
        // structurally. The deterministic seed is the node's stable authored
        // id — persisted topology, identical live and offline, unlike a
        // process-lifetime layer id.
        let time = plan.base().context().time_seconds;
        let epoch = crate::filter_avalanche::avalanche_epoch(time);
        let seed = field.node_id.get() as u32;
        let prepared = &self.corruption_fields[index];
        let history_ready = prepared.history_valid && prepared.history.is_some();
        let uniforms = CorruptionGpuUniforms::for_plan(
            field,
            self.host.dimensions(),
            seed,
            epoch,
            history_ready,
        );
        let base = prepared.uniform_base_slot;
        for (offset, record) in uniforms.iter().enumerate() {
            executor.write_pass(queue, base + offset as u32, record);
        }
        let pong = self.host.surface(HostSurface::Pong).view;
        match field.kind {
            EvaluatedCorruptionKind::BlockDct(_) => {
                let aux = executor.aux_views().ok_or_else(|| {
                    CompositionGpuError::InvalidSchedule(
                        "a Block DCT step was planned without its intermediates".into(),
                    )
                })?;
                executor.encode_pass(
                    encoder,
                    CorruptionPassPipeline::DctStage,
                    &prepared.pass_groups[0],
                    aux[0],
                    base,
                );
                executor.encode_pass(
                    encoder,
                    CorruptionPassPipeline::DctStage,
                    &prepared.pass_groups[1],
                    aux[1],
                    base + 1,
                );
                executor.encode_pass(
                    encoder,
                    CorruptionPassPipeline::DctStage,
                    &prepared.pass_groups[2],
                    aux[0],
                    base + 2,
                );
                executor.encode_pass(
                    encoder,
                    CorruptionPassPipeline::DctFinal,
                    &prepared.pass_groups[3],
                    pong,
                    base + 3,
                );
            }
            EvaluatedCorruptionKind::PixelSort(_) => {
                executor.encode_pass(
                    encoder,
                    CorruptionPassPipeline::PixelSort,
                    &prepared.pass_groups[0],
                    pong,
                    base,
                );
            }
            EvaluatedCorruptionKind::Avalanche(_) => {
                // A valid committed history binds; anything else takes the
                // cold fallback (the carrier as its own history), which is
                // exactly BENDR's shipped single-frame law.
                let group = match (&prepared.history, history_ready) {
                    (Some(history), true) => &history.bind_group,
                    _ => &prepared.pass_groups[0],
                };
                executor.encode_pass(
                    encoder,
                    CorruptionPassPipeline::Avalanche,
                    group,
                    pong,
                    base,
                );
            }
        }
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Pong).texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        // The avalanche's history store: the node's own output (now in
        // Ping), advanced at most once per 30 Hz reference tick on the
        // frame-plan clock (the melt rate law: live and export cascade at
        // the same speed), staged here and published only by the
        // frame-history transaction. The first armed frame always stores so
        // validity can begin.
        if matches!(field.kind, EvaluatedCorruptionKind::Avalanche(_)) {
            let store_tick = (f64::from(time.max(0.0)) * 30.0).floor() as u64;
            let ping = self.host.surface(HostSurface::Ping).texture;
            let dimensions = self.host.dimensions();
            let prepared = &mut self.corruption_fields[index];
            if let Some(history) = &prepared.history {
                if store_tick != prepared.last_store_tick || !prepared.history_valid {
                    copy_texture(encoder, ping, &history.surface.texture, dimensions);
                    prepared.history_staged = true;
                    prepared.staged_store_tick = store_tick;
                }
            }
        }
        Ok(())
    }

    fn execute_rack_segment(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        scope: VisualScopeId,
        segment_index: u8,
        rack_plan: &crate::renderer::rack::CollisionRackPlan,
    ) -> Result<(), CompositionGpuError> {
        let prepared_index = self
            .rack_segments
            .iter()
            .position(|segment| segment.scope == scope && segment.segment_index == segment_index)
            .ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "rack segment {segment_index} for {scope:?} was not prepared"
                ))
            })?;
        let prelocal = self.rack_segments[prepared_index]
            .tap_indices
            .iter()
            .find_map(|(_, _, _, tap_index)| {
                matches!(
                    self.taps[*tap_index].backing,
                    TapBacking::CurrentPreLocal { .. }
                )
                .then_some(*tap_index)
            });
        if let Some(tap_index) = prelocal {
            self.ensure_current_prelocal(encoder, tap_index)?;
        }
        for binding_index in 0..self.rack_segments[prepared_index].tap_indices.len() {
            let (node_id, slot, tap, tap_index) =
                self.rack_segments[prepared_index].tap_indices[binding_index];
            let valid = self.tap_ready(tap_index);
            let updated = self.rack_segments[prepared_index].bindings[self.tap_history_read_index]
                .set_valid(node_id, slot, tap, valid);
            debug_assert!(updated);
        }
        let report = self.rack.encode_at(
            queue,
            encoder,
            rack_plan,
            &self.ping_rack_source,
            &self.rack_segments[prepared_index].bindings[self.tap_history_read_index],
            &self.rack_segments[prepared_index].residual_means,
            self.rack_segments[prepared_index].uniform_base_slot,
            plan.base().context().time_seconds,
        )?;
        let output = self.rack.output(report);
        copy_texture(
            encoder,
            output.texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    fn execute_group(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        group_id: GroupId,
        preceding_root_outputs: usize,
        scope_readback: &mut Option<PreparedScopeRecorderReadback>,
    ) -> Result<(), CompositionGpuError> {
        let group = group_plan(plan, group_id)?;
        self.host.encode_clear(
            encoder,
            self.host.surface(HostSurface::GroupScratch).view,
            wgpu::Color::TRANSPARENT,
        );
        self.capture_prefix_with_crossfade(
            queue,
            encoder,
            CompositePrefix::GroupMember {
                group_id,
                preceding_root_outputs,
                preceding_members: 0,
            },
            bus_frame_params(plan),
        )?;

        let schedule_len = self
            .member_schedules
            .get(&group_id)
            .map_or(0, |schedule| schedule.len());
        let mut completed_members = 0_usize;
        for schedule_index in 0..schedule_len {
            let layer_id = self.member_schedules[&group_id][schedule_index].layer_id;
            self.execute_layer(queue, encoder, plan, layer_id)?;
            self.capture_scope_output(
                encoder,
                VisualScopeId::Layer(layer_id),
                HostSurface::Ping,
                scope_readback,
            )?;
            let drain_len = self.member_schedules[&group_id][schedule_index]
                .drains
                .len();
            for drain_index in 0..drain_len {
                let admission =
                    self.member_schedules[&group_id][schedule_index].drains[drain_index];
                self.load_scheduled_source(encoder, admission.source)?;
                self.admit_layer(queue, encoder, plan, admission.item, Some(group_id))?;
                completed_members += 1;
                self.capture_prefix_with_crossfade(
                    queue,
                    encoder,
                    CompositePrefix::GroupMember {
                        group_id,
                        preceding_root_outputs,
                        preceding_members: completed_members,
                    },
                    bus_frame_params(plan),
                )?;
            }
        }

        // GroupScratch is the authoritative member composite. A group whose
        // first authored operation is a rack node has no MaterializeSpatial
        // host step to seed Ping, so establish that boundary explicitly.
        // A leading spatial step may overwrite Ping from the same scratch
        // source; the copy is intentionally harmless and allocation-free.
        copy_texture(
            encoder,
            self.host.surface(HostSurface::GroupScratch).texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        if !group.bypass {
            let scope = VisualScopeId::Group(group_id);
            for (index, step) in group.execution.steps().iter().enumerate() {
                match step {
                    EvaluatedScopeStep::MaterializeSpatial { .. } => {
                        if index == 0 {
                            self.host.encode_effect(
                                encoder,
                                &self.group_effect,
                                self.host.surface(HostSurface::Ping).view,
                                self.effect_slots[&(scope, index)],
                            )?;
                        } else {
                            self.encode_effect_from_ping(
                                encoder,
                                self.effect_slots[&(scope, index)],
                            )?;
                        }
                    }
                    EvaluatedScopeStep::LegacyCanonical { .. } => {
                        self.encode_effect_from_ping(encoder, self.effect_slots[&(scope, index)])?;
                    }
                    EvaluatedScopeStep::CollisionRack {
                        segment_index,
                        plan: rack_plan,
                    } => self.execute_rack_segment(
                        queue,
                        encoder,
                        plan,
                        scope,
                        *segment_index,
                        rack_plan,
                    )?,
                    EvaluatedScopeStep::SymmetryField { plan: field } => {
                        self.execute_symmetry_field(queue, encoder, plan, scope, field)?;
                    }
                    EvaluatedScopeStep::StudyField { plan: field } => {
                        self.execute_study_field(queue, encoder, plan, scope, field)?;
                    }
                    EvaluatedScopeStep::ScanProcessorField { plan: field } => {
                        self.execute_scan_processor_field(queue, encoder, plan, scope, field)?;
                    }
                    EvaluatedScopeStep::CorruptionField { plan: field } => {
                        self.execute_corruption_field(queue, encoder, plan, scope, field)?;
                    }
                    EvaluatedScopeStep::GroupMatte { .. } => {
                        self.execute_group_matte(encoder, plan, group_id)?;
                    }
                    EvaluatedScopeStep::LegacyTemporal { .. } => {
                        return Err(CompositionGpuError::InvalidSchedule(format!(
                            "group {} contains LegacyTemporal",
                            group_id.get()
                        )));
                    }
                }
            }
        }
        self.capture_scope_output(
            encoder,
            VisualScopeId::Group(group_id),
            HostSurface::Ping,
            scope_readback,
        )?;
        Ok(())
    }

    fn execute_group_matte(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        group_id: GroupId,
    ) -> Result<(), CompositionGpuError> {
        let consumer = ImageTapConsumer::GroupMatte { group_id };
        if let Some(tap_index) = self.tap_index(consumer) {
            self.ensure_current_prelocal(encoder, tap_index)?;
        }
        let zero = self
            .rack
            .surface(1)
            .ok_or_else(|| CompositionGpuError::InvalidSchedule("rack scratch 1 missing".into()))?;
        let target = self
            .rack
            .surface(0)
            .ok_or_else(|| CompositionGpuError::InvalidSchedule("rack scratch 0 missing".into()))?;
        self.host
            .encode_clear(encoder, zero.view, wgpu::Color::TRANSPARENT);
        self.host.encode_matte(
            encoder,
            &self.matte_bindings.get(&consumer).ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "group {} matte binding is absent",
                    group_id.get()
                ))
            })?[self.tap_history_read_index],
            target.view,
            self.matte_slots[&consumer],
        )?;
        copy_texture(
            encoder,
            target.texture,
            self.host.surface(HostSurface::Ping).texture,
            self.host.dimensions(),
        );
        let _ = plan;
        Ok(())
    }

    fn admit_layer(
        &mut self,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        layer_id: StableLayerId,
        group_id: Option<GroupId>,
    ) -> Result<(), CompositionGpuError> {
        let layer = layer_plan(plan, layer_id)?;
        if layer.group_id != group_id {
            return Err(CompositionGpuError::InvalidSchedule(format!(
                "layer {} admission group changed after prepare",
                layer_id.get()
            )));
        }
        let evaluated = &plan.base().layers()[layer.base_layer_index];
        let visible = evaluated.visible
            && evaluated.opacity.is_finite()
            && evaluated.opacity > 0.0
            && (group_id.is_some() || layer.admitted_to_program);
        if !visible {
            return Ok(());
        }
        if layer.admitted_to_program {
            self.apply_master_admission(encoder, plan, layer_id)?;
        }
        let destination = if group_id.is_some() {
            HostSurface::GroupScratch
        } else {
            bus_surface(layer.bus)
        };
        if layer.legacy_matte.is_some() {
            let consumer = ImageTapConsumer::LayerMatte { layer_id };
            let tap_index = self.tap_index(consumer).ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(format!(
                    "layer {} matte has no planned donor",
                    layer_id.get()
                ))
            })?;
            self.ensure_current_prelocal(encoder, tap_index)?;
            if !self.tap_ready(tap_index) {
                let zero = self.rack.surface(1).ok_or_else(|| {
                    CompositionGpuError::InvalidSchedule("rack scratch 1 missing".into())
                })?;
                self.host
                    .encode_clear(encoder, zero.view, wgpu::Color::TRANSPARENT);
            }
            let target = self.rack.surface(0).ok_or_else(|| {
                CompositionGpuError::InvalidSchedule("rack scratch 0 missing".into())
            })?;
            self.host.encode_matte(
                encoder,
                &self.matte_bindings[&consumer][self.tap_history_read_index],
                target.view,
                self.matte_slots[&consumer],
            )?;
            copy_texture(
                encoder,
                target.texture,
                self.host.surface(destination).texture,
                self.host.dimensions(),
            );
        } else {
            let inputs = match destination {
                HostSurface::A => &self.composite_a,
                HostSurface::B => &self.composite_b,
                HostSurface::Program => &self.composite_program,
                HostSurface::GroupScratch => &self.composite_group,
                HostSurface::Ping | HostSurface::Pong => {
                    return Err(CompositionGpuError::InvalidSchedule(
                        "layer admission selected a transient surface".into(),
                    ));
                }
            };
            self.host.encode_composite(
                encoder,
                inputs,
                self.host.surface(HostSurface::Pong).view,
                self.composite_slots[&VisualScopeId::Layer(layer_id)],
            )?;
            copy_texture(
                encoder,
                self.host.surface(HostSurface::Pong).texture,
                self.host.surface(destination).texture,
                self.host.dimensions(),
            );
        }
        Ok(())
    }

    fn apply_master_admission(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        layer_id: StableLayerId,
    ) -> Result<(), CompositionGpuError> {
        if plan.master().canonical_bypass_layers.contains(&layer_id) {
            return Ok(());
        }
        for (index, step) in plan.master().execution.steps().iter().enumerate() {
            let application = match step {
                EvaluatedScopeStep::MaterializeSpatial { application, .. }
                | EvaluatedScopeStep::LegacyCanonical { application, .. } => application,
                _ => continue,
            };
            if *application == LegacyCanonicalApplication::PreCompositeLayerAdmission {
                self.encode_effect_from_ping(
                    encoder,
                    self.effect_slots[&(VisualScopeId::Master, index)],
                )?;
            }
        }
        Ok(())
    }

    fn admit_group(
        &self,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        group_id: GroupId,
    ) -> Result<(), CompositionGpuError> {
        let group = group_plan(plan, group_id)?;
        if !group.span.admitted_to_program || !group.opacity.is_finite() || group.opacity <= 0.0 {
            return Ok(());
        }
        let destination = bus_surface(group.bus);
        let inputs = match destination {
            HostSurface::A => &self.composite_a,
            HostSurface::B => &self.composite_b,
            HostSurface::Program => &self.composite_program,
            _ => unreachable!(),
        };
        self.host.encode_composite(
            encoder,
            inputs,
            self.host.surface(HostSurface::Pong).view,
            self.composite_slots[&VisualScopeId::Group(group_id)],
        )?;
        copy_texture(
            encoder,
            self.host.surface(HostSurface::Pong).texture,
            self.host.surface(destination).texture,
            self.host.dimensions(),
        );
        Ok(())
    }

    fn execute_master(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdvancedCompositionPlan,
        timing: CompositionFrameTiming,
    ) -> Result<(), CompositionGpuError> {
        let mut captured_clean_program = false;
        for (index, step) in plan.master().execution.steps().iter().enumerate() {
            match step {
                EvaluatedScopeStep::MaterializeSpatial { application, .. }
                | EvaluatedScopeStep::LegacyCanonical { application, .. } => {
                    if *application == LegacyCanonicalApplication::ScopeLocal {
                        self.encode_effect_from_ping(
                            encoder,
                            self.effect_slots[&(VisualScopeId::Master, index)],
                        )?;
                    }
                }
                EvaluatedScopeStep::CollisionRack {
                    segment_index,
                    plan: rack_plan,
                } => self.execute_rack_segment(
                    queue,
                    encoder,
                    plan,
                    VisualScopeId::Master,
                    *segment_index,
                    rack_plan,
                )?,
                EvaluatedScopeStep::LegacyTemporal { params } => {
                    if !captured_clean_program {
                        copy_texture(
                            encoder,
                            self.host.surface(HostSurface::Ping).texture,
                            self.host.surface(HostSurface::Program).texture,
                            self.host.dimensions(),
                        );
                        captured_clean_program = true;
                    }
                    let ping = self.host.surface(HostSurface::Ping);
                    let pong = self.host.surface(HostSurface::Pong);
                    let wrote_ping = if let Some(routed) = &self.routed_garden {
                        self.host.encode_temporal_routed(
                            queue,
                            encoder,
                            &self.temporal,
                            ping.texture,
                            ping.view,
                            pong.texture,
                            pong.view,
                            routed,
                            params,
                            HostFrameTiming::from_temporal_input(timing.temporal_input()),
                        )
                    } else {
                        self.host.encode_temporal(
                            queue,
                            encoder,
                            &self.temporal,
                            ping.texture,
                            pong.texture,
                            pong.view,
                            params,
                            HostFrameTiming::from_temporal_input(timing.temporal_input()),
                        );
                        false
                    };
                    if !wrote_ping {
                        copy_texture(encoder, pong.texture, ping.texture, self.host.dimensions());
                    }
                }
                EvaluatedScopeStep::SymmetryField { plan: field } => {
                    self.execute_symmetry_field(
                        queue,
                        encoder,
                        plan,
                        VisualScopeId::Master,
                        field,
                    )?;
                }
                EvaluatedScopeStep::StudyField { plan: field } => {
                    self.execute_study_field(queue, encoder, plan, VisualScopeId::Master, field)?;
                }
                EvaluatedScopeStep::ScanProcessorField { plan: field } => {
                    self.execute_scan_processor_field(
                        queue,
                        encoder,
                        plan,
                        VisualScopeId::Master,
                        field,
                    )?;
                }
                EvaluatedScopeStep::CorruptionField { plan: field } => {
                    self.execute_corruption_field(
                        queue,
                        encoder,
                        plan,
                        VisualScopeId::Master,
                        field,
                    )?;
                }
                EvaluatedScopeStep::GroupMatte { .. } => {
                    return Err(CompositionGpuError::InvalidSchedule(
                        "master execution contains a group matte".into(),
                    ));
                }
            }
        }
        if !captured_clean_program {
            copy_texture(
                encoder,
                self.host.surface(HostSurface::Ping).texture,
                self.host.surface(HostSurface::Program).texture,
                self.host.dimensions(),
            );
        }
        Ok(())
    }
}

fn create_retained_surface(
    device: &wgpu::Device,
    dimensions: [u32; 2],
    label: &'static str,
) -> RetainedSurface {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COMPOSITION_WORKING_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    RetainedSurface { texture, view }
}

fn prepare_tap_surfaces(
    device: &wgpu::Device,
    dimensions: [u32; 2],
    planned: &[PlannedImageTap],
) -> Result<Vec<PreparedTap>, CompositionGpuError> {
    Ok(planned
        .iter()
        .cloned()
        .map(|planned| {
            let backing = if matches!(planned.resolved, PlannedImageSource::Transparent) {
                TapBacking::Transparent
            } else if matches!(planned.resolved, PlannedImageSource::ProgramHistory) {
                TapBacking::ProgramHistory
            } else if matches!(planned.resolved, PlannedImageSource::GestureCanvas) {
                // Deliberately ahead of the previous-frame branch. A canvas
                // route at N-1 timing must not allocate a parity pair: the
                // canvas is a frame-committed singleton with one image, and the
                // planner records no previous-frame dependency for it either,
                // so the declared and actual ledgers agree at zero.
                TapBacking::GestureCanvas
            } else if matches!(planned.resolved, PlannedImageSource::ProgramTap) {
                // Deliberately ahead of the previous-frame branch for the same
                // reason as the canvas: the tap is a frame-committed singleton
                // with one image — it *is* the N-1 image — so a route at N-1
                // timing must not allocate a parity pair, and the planner
                // records no previous-frame dependency for it either.
                TapBacking::ProgramTap
            } else if planned.origin.timing() == EdgeTiming::PreviousFrame {
                TapBacking::Previous {
                    surfaces: std::array::from_fn(|_| {
                        create_retained_surface(
                            device,
                            dimensions,
                            "Advanced composition previous tap parity",
                        )
                    }),
                    initialized: false,
                    staged: false,
                }
            } else if matches!(
                planned.resolved,
                PlannedImageSource::SelectedLayer {
                    stage: LayerImageStage::PreLocalEffects,
                    ..
                }
            ) {
                let PlannedImageSource::SelectedLayer { layer_id, .. } = planned.resolved else {
                    unreachable!("matched selected PreLocal source")
                };
                TapBacking::CurrentPreLocal { layer_id }
            } else {
                TapBacking::Current(create_retained_surface(
                    device,
                    dimensions,
                    "Advanced composition current tap",
                ))
            };
            PreparedTap { planned, backing }
        })
        .collect())
}

fn prepare_current_prelocal_surfaces(
    device: &wgpu::Device,
    dimensions: [u32; 2],
    taps: &[PreparedTap],
) -> BTreeMap<StableLayerId, RetainedSurface> {
    let mut surfaces = BTreeMap::new();
    for tap in taps {
        let TapBacking::CurrentPreLocal { layer_id } = tap.backing else {
            continue;
        };
        surfaces.entry(layer_id).or_insert_with(|| {
            create_retained_surface(
                device,
                dimensions,
                "Advanced composition current PreLocal donor",
            )
        });
    }
    surfaces
}

fn validate_actual_surface_ledger(
    plan: &AdvancedCompositionPlan,
    retain_program_history: bool,
    taps: &[PreparedTap],
    current_prelocal_surfaces: usize,
    temporal_dry_surfaces: usize,
) -> Result<(), CompositionGpuError> {
    // Spelled exhaustively rather than with a wildcard: this is the fail-closed
    // reconcile seam, so a new backing must be charged deliberately instead of
    // inheriting zero by accident.
    let tap_surfaces: u32 = taps
        .iter()
        .map(|tap| match tap.backing {
            TapBacking::Current(_) => 1,
            TapBacking::Previous { .. } => 2,
            TapBacking::Transparent
            | TapBacking::ProgramHistory
            | TapBacking::GestureCanvas
            | TapBacking::ProgramTap
            | TapBacking::CurrentPreLocal { .. } => 0,
        })
        .sum();
    let actual = 6_u32
        .checked_add(2)
        .and_then(|value| value.checked_add(2 * u32::from(retain_program_history)))
        .and_then(|value| value.checked_add(tap_surfaces))
        .and_then(|value| value.checked_add(current_prelocal_surfaces as u32))
        .and_then(|value| value.checked_add(temporal_dry_surfaces as u32))
        .ok_or_else(|| CompositionGpuError::InvalidSchedule("surface ledger overflowed".into()))?;
    if actual != plan.resources().rgba16_surface_layers {
        return Err(CompositionGpuError::InvalidSchedule(format!(
            "resource ledger declares {} RGBA16 layers but executor allocates exactly {actual}",
            plan.resources().rgba16_surface_layers
        )));
    }
    Ok(())
}

fn prepared_tap_view<'a>(
    host: &'a CompositionHost,
    prelocal_surfaces: &'a BTreeMap<StableLayerId, RetainedSurface>,
    zero: &'a wgpu::TextureView,
    gesture_canvas: Option<&'a wgpu::TextureView>,
    program_tap: Option<&'a wgpu::TextureView>,
    tap: &'a PreparedTap,
    read_index: usize,
) -> &'a wgpu::TextureView {
    match &tap.backing {
        TapBacking::Transparent => zero,
        TapBacking::ProgramHistory => host
            .program_history_views()
            .map_or(zero, |views| views[read_index]),
        // The presented gesture field when the host published one, and the
        // rack-owned zero texture when it did not. An unbound canvas is inert
        // rather than wrong: under the frozen donor decode a fully transparent
        // donor yields exactly zero displacement, which is the same answer an
        // un-etched canvas gives. `tap_ready` reports the same fact, so the
        // node also sees `donor_valid = false` instead of a confident empty
        // field. Both sites read the one binding, so they cannot disagree.
        TapBacking::GestureCanvas => gesture_canvas.unwrap_or(zero),
        // The published programme copy when the host published one, and the
        // rack-owned zero texture when it did not — the identical law: an
        // unbound tap is inert rather than wrong, `tap_ready` reports the same
        // fact, and both sites read the one binding so they cannot disagree.
        TapBacking::ProgramTap => program_tap.unwrap_or(zero),
        TapBacking::CurrentPreLocal { layer_id } => prelocal_surfaces
            .get(layer_id)
            .map_or(zero, |surface| &surface.view),
        TapBacking::Current(surface) => &surface.view,
        TapBacking::Previous { surfaces, .. } => &surfaces[read_index].view,
    }
}

fn tap_boundary_scope(tap: &PlannedImageTap) -> Option<VisualScopeId> {
    match tap.resolved {
        PlannedImageSource::Scope(scope) | PlannedImageSource::OneBelow(scope) => Some(scope),
        PlannedImageSource::SelectedLayer {
            layer_id,
            stage: LayerImageStage::PostLocalEffects,
        } => Some(VisualScopeId::Layer(layer_id)),
        PlannedImageSource::SelectedLayer {
            stage: LayerImageStage::PreLocalEffects,
            ..
        }
        | PlannedImageSource::AllBelow(_)
        | PlannedImageSource::ProgramHistory
        // The gesture canvas is etched outside the composition graph, and the
        // programme tap is published outside it, so neither names a producing
        // scope and neither takes part in any scope ordering.
        | PlannedImageSource::GestureCanvas
        | PlannedImageSource::ProgramTap
        | PlannedImageSource::Transparent => None,
    }
}

fn stage_tap_from_texture(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    tap: &mut PreparedTap,
    dimensions: [u32; 2],
    write_index: usize,
) {
    match &mut tap.backing {
        TapBacking::Current(surface) => {
            copy_texture(encoder, source, &surface.texture, dimensions);
        }
        TapBacking::Previous {
            surfaces, staged, ..
        } => {
            copy_texture(encoder, source, &surfaces[write_index].texture, dimensions);
            *staged = true;
        }
        TapBacking::Transparent
        | TapBacking::ProgramHistory
        | TapBacking::GestureCanvas
        | TapBacking::ProgramTap
        | TapBacking::CurrentPreLocal { .. } => {}
    }
}

fn copy_texture(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    target: &wgpu::Texture,
    dimensions: [u32; 2],
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        },
    );
}

fn runtime_matte_params(
    matte: crate::visual_rack::RuntimeImageMatte,
    donor_valid: bool,
) -> ResolvedMatteParams {
    ResolvedMatteParams {
        channel: match matte.channel {
            MatteChannel::Alpha => MatteChannelCode::Alpha,
            MatteChannel::Luma => MatteChannelCode::Luma,
            MatteChannel::Red => MatteChannelCode::Red,
            MatteChannel::Green => MatteChannelCode::Green,
            MatteChannel::Blue => MatteChannelCode::Blue,
        },
        invert: matte.invert,
        amount: matte.amount,
        threshold: matte.threshold,
        softness: matte.softness,
        donor_valid,
    }
    .sanitized()
}

type UniformSlotMaps = (
    usize,
    BTreeMap<(VisualScopeId, usize), HostUniformSlot>,
    BTreeMap<StableLayerId, HostUniformSlot>,
    BTreeMap<StableLayerId, HostUniformSlot>,
    BTreeMap<VisualScopeId, HostUniformSlot>,
    HostUniformSlot,
    BTreeMap<ImageTapConsumer, HostUniformSlot>,
);

fn assign_uniform_slots(
    plan: &AdvancedCompositionPlan,
) -> Result<UniformSlotMaps, CompositionGpuError> {
    fn next_slot(next: &mut u32) -> Result<HostUniformSlot, CompositionGpuError> {
        let slot = HostUniformSlot(*next);
        *next = next.checked_add(1).ok_or_else(|| {
            CompositionGpuError::InvalidSchedule("uniform slot count overflowed".into())
        })?;
        Ok(slot)
    }

    let mut next_effect = 0_u32;
    let mut effect_slots = BTreeMap::new();
    let mut exact_layer_slots = BTreeMap::new();
    let mut prelocal_slots = BTreeMap::new();
    for layer in plan.layers() {
        prelocal_slots.insert(layer.stable_id, next_slot(&mut next_effect)?);
        if layer.execution.is_exact_legacy() {
            exact_layer_slots.insert(layer.stable_id, next_slot(&mut next_effect)?);
        }
        for (index, step) in layer.execution.steps().iter().enumerate() {
            if matches!(
                step,
                EvaluatedScopeStep::MaterializeSpatial { .. }
                    | EvaluatedScopeStep::LegacyCanonical { .. }
            ) {
                effect_slots.insert(
                    (VisualScopeId::Layer(layer.stable_id), index),
                    next_slot(&mut next_effect)?,
                );
            }
        }
    }
    for group in plan.groups() {
        for (index, step) in group.execution.steps().iter().enumerate() {
            if matches!(
                step,
                EvaluatedScopeStep::MaterializeSpatial { .. }
                    | EvaluatedScopeStep::LegacyCanonical { .. }
            ) {
                effect_slots.insert(
                    (VisualScopeId::Group(group.id), index),
                    next_slot(&mut next_effect)?,
                );
            }
        }
    }
    for (index, step) in plan.master().execution.steps().iter().enumerate() {
        if matches!(
            step,
            EvaluatedScopeStep::MaterializeSpatial { .. }
                | EvaluatedScopeStep::LegacyCanonical { .. }
        ) {
            effect_slots.insert((VisualScopeId::Master, index), next_slot(&mut next_effect)?);
        }
    }

    let mut composite_slots = BTreeMap::new();
    let mut next_composite = 0_u32;
    for layer in plan.layers() {
        composite_slots.insert(
            VisualScopeId::Layer(layer.stable_id),
            next_slot(&mut next_composite)?,
        );
    }
    for group in plan.groups() {
        composite_slots.insert(
            VisualScopeId::Group(group.id),
            next_slot(&mut next_composite)?,
        );
    }
    let prefix_composite_slot = next_slot(&mut next_composite)?;

    let mut matte_slots = BTreeMap::new();
    let mut next_matte = 0_u32;
    for layer in plan
        .layers()
        .iter()
        .filter(|layer| layer.legacy_matte.is_some())
    {
        matte_slots.insert(
            ImageTapConsumer::LayerMatte {
                layer_id: layer.stable_id,
            },
            next_slot(&mut next_matte)?,
        );
    }
    for group in plan.groups().iter().filter(|group| group.matte.is_some()) {
        matte_slots.insert(
            ImageTapConsumer::GroupMatte { group_id: group.id },
            next_slot(&mut next_matte)?,
        );
    }
    Ok((
        next_effect as usize,
        effect_slots,
        exact_layer_slots,
        prelocal_slots,
        composite_slots,
        prefix_composite_slot,
        matte_slots,
    ))
}

fn layer_plan(
    plan: &AdvancedCompositionPlan,
    layer_id: StableLayerId,
) -> Result<
    &crate::evaluated_frame::evaluated_composition::EvaluatedLayerScopePlan,
    CompositionGpuError,
> {
    plan.layers()
        .iter()
        .find(|layer| layer.stable_id == layer_id)
        .ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(format!(
                "stable layer {} is absent from the evaluated plan",
                layer_id.get()
            ))
        })
}

fn group_plan(
    plan: &AdvancedCompositionPlan,
    group_id: GroupId,
) -> Result<
    &crate::evaluated_frame::evaluated_composition::EvaluatedGroupScopePlan,
    CompositionGpuError,
> {
    plan.groups()
        .iter()
        .find(|group| group.id == group_id)
        .ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(format!(
                "group {} is absent from the evaluated plan",
                group_id.get()
            ))
        })
}

/// Everything one bus pass needs from the evaluated frame plan: the plan is
/// the sole source, so live and export encode identical bus uniforms for
/// the same frame facts.
fn bus_frame_params(plan: &AdvancedCompositionPlan) -> BusFrameParams {
    let context = plan.base().context();
    BusFrameParams {
        crossfade: plan.bus_crossfade(),
        mixer: plan.mixer(),
        time_seconds: context.time_seconds,
        random_seed: plan.base().master_pass().effects.random_seed,
        output_size: context.output_size,
        history_valid: false,
    }
}

const fn bus_surface(bus: BusAssignment) -> HostSurface {
    match bus {
        BusAssignment::A => HostSurface::A,
        BusAssignment::B => HostSurface::B,
        BusAssignment::Program => HostSurface::Program,
    }
}

/// One dynamic-offset uniform slot per dedicated step the frame can encode
/// before a single queue submit. Counted from the EMITTED steps, exactly as the
/// planner's own dedicated ledger is, so the arena can never be short.
fn study_field_step_count(plan: &AdvancedCompositionPlan) -> usize {
    let mut count = 0;
    let mut tally = |execution: &EvaluatedScopeExecution| {
        count += execution
            .steps()
            .iter()
            .filter(|step| matches!(step, EvaluatedScopeStep::StudyField { .. }))
            .count();
    };
    for layer in plan.layers() {
        tally(&layer.execution);
    }
    for group in plan.groups() {
        tally(&group.execution);
    }
    tally(&plan.master().execution);
    count
}

/// Assign every planned Study step an arena slot in deterministic
/// layers-then-groups-then-master step order, upload each resolved program
/// once, and build the topology-fixed bind group. Inert steps still own
/// their slot so slot numbering never depends on frame-local values.
fn prepare_study_fields(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: &AdvancedCompositionPlan,
    executor: &StudyGpuExecutor,
    carrier: &wgpu::TextureView,
    history: &wgpu::TextureView,
) -> Vec<PreparedStudyField> {
    let mut prepared = Vec::new();
    let mut visit = |scope: VisualScopeId, execution: &EvaluatedScopeExecution| {
        for step in execution.steps() {
            let EvaluatedScopeStep::StudyField { plan: field } = step else {
                continue;
            };
            let uniform_slot = prepared.len() as u32;
            if let Some(program) = &field.program {
                executor.write_program(queue, uniform_slot, program);
            }
            prepared.push(PreparedStudyField {
                scope,
                node_id: field.node_id,
                uniform_slot,
                bind_group: executor.create_bind_group(device, carrier, history),
            });
        }
    };
    for layer in plan.layers() {
        visit(VisualScopeId::Layer(layer.stable_id), &layer.execution);
    }
    for group in plan.groups() {
        visit(VisualScopeId::Group(group.id), &group.execution);
    }
    visit(VisualScopeId::Master, &plan.master().execution);
    prepared
}

fn scan_processor_field_step_count(plan: &AdvancedCompositionPlan) -> usize {
    let mut count = 0;
    let mut tally = |execution: &EvaluatedScopeExecution| {
        count += execution
            .steps()
            .iter()
            .filter(|step| matches!(step, EvaluatedScopeStep::ScanProcessorField { .. }))
            .count();
    };
    for layer in plan.layers() {
        tally(&layer.execution);
    }
    for group in plan.groups() {
        tally(&group.execution);
    }
    tally(&plan.master().execution);
    count
}

/// Assign every planned Scan Processor step an arena slot in deterministic
/// layers-then-groups-then-master step order and build its two topology-fixed
/// bind groups. Inert steps still own their slot so slot numbering never
/// depends on frame-local values.
fn prepare_scan_processor_fields(
    device: &wgpu::Device,
    plan: &AdvancedCompositionPlan,
    executor: &ScanProcessorGpuExecutor,
    carrier: &wgpu::TextureView,
) -> Vec<PreparedScanProcessorField> {
    let mut prepared = Vec::new();
    let mut visit = |scope: VisualScopeId, execution: &EvaluatedScopeExecution| {
        for step in execution.steps() {
            let EvaluatedScopeStep::ScanProcessorField { plan: field } = step else {
                continue;
            };
            let uniform_slot = prepared.len() as u32;
            prepared.push(PreparedScanProcessorField {
                scope,
                node_id: field.node_id,
                uniform_slot,
                geometry_bind_group: executor.create_geometry_bind_group(device, carrier),
                resolve_bind_group: executor.create_resolve_bind_group(device, carrier),
            });
        }
    };
    for layer in plan.layers() {
        visit(VisualScopeId::Layer(layer.stable_id), &layer.execution);
    }
    for group in plan.groups() {
        visit(VisualScopeId::Group(group.id), &group.execution);
    }
    visit(VisualScopeId::Master, &plan.master().execution);
    prepared
}

/// Walk every planned corruption step in the deterministic
/// layers-then-groups-then-master step order.
fn for_each_corruption_step(
    plan: &AdvancedCompositionPlan,
    mut visit: impl FnMut(VisualScopeId, &EvaluatedCorruptionPlan),
) {
    let mut walk = |scope: VisualScopeId, execution: &EvaluatedScopeExecution| {
        for step in execution.steps() {
            if let EvaluatedScopeStep::CorruptionField { plan: field } = step {
                visit(scope, field);
            }
        }
    };
    for layer in plan.layers() {
        walk(VisualScopeId::Layer(layer.stable_id), &layer.execution);
    }
    for group in plan.groups() {
        walk(VisualScopeId::Group(group.id), &group.execution);
    }
    walk(VisualScopeId::Master, &plan.master().execution);
}

/// Total uniform-slot count (a Block DCT step owns four) and whether any
/// DCT step exists, so the shared intermediates allocate only when needed.
fn corruption_field_totals(plan: &AdvancedCompositionPlan) -> (usize, bool) {
    let mut slots = 0usize;
    let mut has_dct = false;
    for_each_corruption_step(plan, |_, field| {
        slots += field.kind.pass_count() as usize;
        has_dct |= matches!(field.kind, EvaluatedCorruptionKind::BlockDct(_));
    });
    (slots, has_dct)
}

/// Assign every planned corruption step its arena slots in deterministic
/// step order and build its topology-fixed pass groups. Inert steps still
/// own their slots so numbering never depends on frame-local values.
fn prepare_corruption_fields(
    device: &wgpu::Device,
    plan: &AdvancedCompositionPlan,
    executor: &CorruptionGpuExecutor,
    carrier: &wgpu::TextureView,
) -> Vec<PreparedCorruptionField> {
    let mut prepared = Vec::new();
    let mut next_slot = 0u32;
    for_each_corruption_step(plan, |scope, field| {
        let pass_groups = match field.kind {
            EvaluatedCorruptionKind::BlockDct(_) => {
                let aux = executor
                    .aux_views()
                    .expect("a plan with a DCT step allocates the intermediates");
                vec![
                    executor.create_bind_group(device, carrier, carrier),
                    executor.create_bind_group(device, aux[0], carrier),
                    executor.create_bind_group(device, aux[1], carrier),
                    executor.create_bind_group(device, aux[0], carrier),
                ]
            }
            EvaluatedCorruptionKind::PixelSort(_) | EvaluatedCorruptionKind::Avalanche(_) => {
                vec![executor.create_bind_group(device, carrier, carrier)]
            }
        };
        prepared.push(PreparedCorruptionField {
            scope,
            node_id: field.node_id,
            uniform_base_slot: next_slot,
            pass_groups,
            history: None,
            history_valid: false,
            history_staged: false,
            last_store_tick: 0,
            staged_store_tick: 0,
        });
        next_slot += field.kind.pass_count();
    });
    prepared
}

fn symmetry_field_step_count(plan: &AdvancedCompositionPlan) -> usize {
    let mut count = 0;
    let mut tally = |execution: &EvaluatedScopeExecution| {
        count += execution
            .steps()
            .iter()
            .filter(|step| matches!(step, EvaluatedScopeStep::SymmetryField { .. }))
            .count();
    };
    for layer in plan.layers() {
        tally(&layer.execution);
    }
    for group in plan.groups() {
        tally(&group.execution);
    }
    tally(&plan.master().execution);
    count
}

/// Prepare every dedicated step's input bind groups, for both committed N-1
/// read parities and both carrier parities, once.
#[allow(
    clippy::too_many_arguments,
    reason = "preparation binds every borrowed GPU input explicitly"
)]
fn prepare_symmetry_fields(
    device: &wgpu::Device,
    plan: &AdvancedCompositionPlan,
    host: &CompositionHost,
    executor: &SymmetryFieldExecutor,
    motion: Option<&MotionGpuResources>,
    taps: &[PreparedTap],
    prelocal_surfaces: &BTreeMap<StableLayerId, RetainedSurface>,
    zero: &wgpu::TextureView,
    gesture_canvas: Option<&wgpu::TextureView>,
    program_tap: Option<&wgpu::TextureView>,
    carriers: [&wgpu::TextureView; 2],
) -> Result<Vec<PreparedSymmetryField>, CompositionGpuError> {
    let mut prepared = Vec::new();
    let mut next_uniform_slot = 0_usize;
    let mut visit = |scope: VisualScopeId,
                     execution: &EvaluatedScopeExecution|
     -> Result<(), CompositionGpuError> {
        for step in execution.steps() {
            let EvaluatedScopeStep::SymmetryField { plan: field } = step else {
                continue;
            };
            // Slot index is route identity: each slot is looked up by its own
            // consumer key, so an unarmed or missing slot 0 never slides slot
            // 1's donor down into its place.
            let mut indices = Vec::new();
            for slot in 0..SYMMETRY_IMAGE_SLOTS {
                let consumer = ImageTapConsumer::RackNode {
                    scope,
                    node_id: field.node_id,
                    slot: slot as u8,
                };
                let Some(tap_index) = tap_index_for_consumer(plan, consumer) else {
                    continue;
                };
                indices.push((slot, tap_index));
            }
            let prelocal_donors = indices
                .iter()
                .filter_map(|(_, tap_index)| match taps[*tap_index].backing {
                    TapBacking::CurrentPreLocal { layer_id } => Some(layer_id),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            if prelocal_donors.len() > 1 {
                return Err(CompositionGpuError::InvalidSchedule(
                    "a Symmetry Field contains more than one current PreLocal donor".into(),
                ));
            }
            let bindings = std::array::from_fn(|read_index| {
                let mut donors: [Option<&wgpu::TextureView>; SYMMETRY_IMAGE_SLOTS] = [None, None];
                for (slot, tap_index) in &indices {
                    donors[*slot] = match taps[*tap_index].backing {
                        TapBacking::Transparent => None,
                        _ => Some(prepared_tap_view(
                            host,
                            prelocal_surfaces,
                            zero,
                            gesture_canvas,
                            program_tap,
                            &taps[*tap_index],
                            read_index,
                        )),
                    };
                }
                executor.prepare_bindings(
                    device,
                    carriers,
                    &[SymmetryFieldInput {
                        node_id: field.node_id,
                        donors,
                        // The clean-history ring is borrowed from the host,
                        // never duplicated.
                        history: Some(host.temporal_history_view()),
                    }],
                )
            });
            let [first, second] = bindings;
            // The motion group is prepared ONCE per node, outside the N-1
            // parity loop: it holds no image view, so rebuilding it per read
            // index would double its cost for nothing. Each admitted slot hands
            // over BOTH committed parities of its field's vector/gate pair, and
            // the executor prebuilds every `[slot 0][slot 1]` combination — the
            // two slots are independent fields and are never required to agree.
            //
            // An UNADMITTED slot stays `None` and binds the executor's
            // defined-zero neutral pair; the planner has already published the
            // diagnostic that names it. An ADMITTED slot whose field the
            // prepared resources do not own is a renderer/planner disagreement
            // rather than a lost donor, so it is refused by name instead of
            // being silently degraded to neutral — a neutral fallback there
            // would hide exactly the wiring failure this hand-off exists to fix.
            let mut motion_views: [Option<SymmetryMotionViews<'_>>; SYMMETRY_MOTION_SLOTS] =
                [None, None];
            for (slot, admitted) in field.motion_field_slots.iter().enumerate() {
                let Some(field_slot) = *admitted else {
                    continue;
                };
                let views = motion
                    .and_then(|motion| motion.field_primitive_views(field_slot))
                    .ok_or_else(|| {
                        CompositionGpuError::InvalidSchedule(format!(
                            "Symmetry Field node {} motion slot {slot} was admitted against \
                             motion field {field_slot}, which the prepared motion resources do \
                             not own",
                            field.node_id.get()
                        ))
                    })?;
                motion_views[slot] = Some(SymmetryMotionViews {
                    vectors: views.vectors,
                    gates: views.gates,
                    grid: views.grid,
                });
            }
            let motion_bindings = executor.prepare_motion_bindings(
                device,
                &[SymmetryFieldMotionInput {
                    node_id: field.node_id,
                    motion: motion_views,
                }],
            )?;
            let uniform_slot = next_uniform_slot;
            next_uniform_slot = next_uniform_slot.checked_add(1).ok_or_else(|| {
                CompositionGpuError::InvalidSchedule(
                    "Symmetry Field uniform slot count overflowed during preparation".into(),
                )
            })?;
            prepared.push(PreparedSymmetryField {
                scope,
                node_id: field.node_id,
                uniform_slot,
                bindings: [first?, second?],
                motion_bindings,
                motion_field_slots: field.motion_field_slots,
                tap_indices: indices.into_boxed_slice(),
            });
        }
        Ok(())
    };
    for layer in plan.layers() {
        visit(VisualScopeId::Layer(layer.stable_id), &layer.execution)?;
    }
    for group in plan.groups() {
        visit(VisualScopeId::Group(group.id), &group.execution)?;
    }
    visit(VisualScopeId::Master, &plan.master().execution)?;
    Ok(prepared)
}

/// Resolve one planned tap by its complete consumer identity. `RackNode`
/// carries its route slot, so the first positional match is the exact route
/// rather than whichever slot of a multi-route node happened to be collected
/// first.
fn tap_index_for_consumer(
    plan: &AdvancedCompositionPlan,
    consumer: ImageTapConsumer,
) -> Option<usize> {
    plan.image_taps()
        .iter()
        .position(|tap| tap.consumer == consumer)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the two singleton donor views are independent inputs, not a bundle"
)]
fn prepare_rack_segments(
    device: &wgpu::Device,
    plan: &AdvancedCompositionPlan,
    host: &CompositionHost,
    rack: &CollisionRackExecutor,
    taps: &[PreparedTap],
    prelocal_surfaces: &BTreeMap<StableLayerId, RetainedSurface>,
    gesture_canvas: Option<&wgpu::TextureView>,
    program_tap: Option<&wgpu::TextureView>,
) -> Result<Vec<PreparedRackSegment>, CompositionGpuError> {
    let zero = rack
        .surface(1)
        .ok_or_else(|| CompositionGpuError::InvalidSchedule("rack scratch 1 missing".into()))?;
    let mut prepared = Vec::new();
    let mut next_uniform_slot = 0_usize;
    let mut visit = |scope: VisualScopeId,
                     execution: &EvaluatedScopeExecution|
     -> Result<(), CompositionGpuError> {
        for step in execution.steps() {
            let EvaluatedScopeStep::CollisionRack {
                segment_index,
                plan: rack_plan,
            } = step
            else {
                continue;
            };
            let mut indices = Vec::new();
            for pass in rack_plan.passes() {
                // Every authored slot of an ordinary rack pass is bound
                // independently. A single-route kind fills only the primary
                // slot; a two-route kind reaches this loop twice and its
                // donors can never alias. The Symmetry Field is lifted into
                // its own dedicated step and never reaches this segment
                // builder at all.
                for (index, tap) in pass.kind.image_taps().into_iter().enumerate() {
                    let Some(tap) = tap else {
                        continue;
                    };
                    let slot = u8::try_from(index).unwrap_or(RACK_PRIMARY_ROUTE_SLOT);
                    let consumer = ImageTapConsumer::RackNode {
                        scope,
                        node_id: pass.node_id,
                        slot,
                    };
                    let Some(tap_index) = tap_index_for_consumer(plan, consumer) else {
                        continue;
                    };
                    indices.push((pass.node_id, slot, tap, tap_index));
                }
            }
            let prelocal_donors = indices
                .iter()
                .filter_map(|(_, _, _, tap_index)| match taps[*tap_index].backing {
                    TapBacking::CurrentPreLocal { layer_id } => Some(layer_id),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            if prelocal_donors.len() > 1 {
                return Err(CompositionGpuError::InvalidSchedule(
                    "a rack segment contains more than one current PreLocal donor".into(),
                ));
            }
            let bindings = std::array::from_fn(|read_index| {
                let inputs = indices
                    .iter()
                    .map(|(node_id, slot, tap, tap_index)| {
                        let view = match taps[*tap_index].backing {
                            TapBacking::Transparent => None,
                            _ => Some(prepared_tap_view(
                                host,
                                prelocal_surfaces,
                                zero.view,
                                gesture_canvas,
                                program_tap,
                                &taps[*tap_index],
                                read_index,
                            )),
                        };
                        RackImageInput {
                            node_id: *node_id,
                            slot: *slot,
                            tap: *tap,
                            view,
                        }
                    })
                    .collect::<Vec<_>>();
                rack.prepare_image_bindings(device, &inputs)
            });
            let [first, second] = bindings;
            let residual_means = rack.prepare_residual_means(device, rack_plan)?;
            let uniform_base_slot = next_uniform_slot;
            next_uniform_slot = next_uniform_slot
                .checked_add(rack_plan.uniform_slots())
                .ok_or_else(|| {
                    CompositionGpuError::InvalidSchedule(
                        "rack uniform slot count overflowed during preparation".into(),
                    )
                })?;
            prepared.push(PreparedRackSegment {
                scope,
                segment_index: *segment_index,
                uniform_base_slot,
                bindings: [first?, second?],
                residual_means,
                tap_indices: indices.into_boxed_slice(),
            });
        }
        Ok(())
    };
    for layer in plan.layers() {
        visit(VisualScopeId::Layer(layer.stable_id), &layer.execution)?;
    }
    for group in plan.groups() {
        visit(VisualScopeId::Group(group.id), &group.execution)?;
    }
    visit(VisualScopeId::Master, &plan.master().execution)?;
    Ok(prepared)
}

fn rack_uniform_slot_count(plan: &AdvancedCompositionPlan) -> Result<usize, CompositionGpuError> {
    let mut count = 0_usize;
    let executions = plan
        .layers()
        .iter()
        .map(|layer| &layer.execution)
        .chain(plan.groups().iter().map(|group| &group.execution))
        .chain(std::iter::once(&plan.master().execution));
    for execution in executions {
        for step in execution.steps() {
            let EvaluatedScopeStep::CollisionRack {
                plan: rack_plan, ..
            } = step
            else {
                continue;
            };
            count = count
                .checked_add(rack_plan.uniform_slots())
                .ok_or_else(|| {
                    CompositionGpuError::InvalidSchedule(
                        "rack uniform slot count overflowed during planning".into(),
                    )
                })?;
        }
    }
    Ok(count.max(1))
}

fn tap_scope_producer(tap: &PreparedTap) -> Option<VisualScopeId> {
    if tap.planned.origin.timing() != EdgeTiming::CurrentFrame {
        return None;
    }
    match tap.planned.resolved {
        PlannedImageSource::Scope(scope) | PlannedImageSource::OneBelow(scope) => Some(scope),
        PlannedImageSource::SelectedLayer {
            layer_id,
            stage: LayerImageStage::PostLocalEffects,
        } => Some(VisualScopeId::Layer(layer_id)),
        PlannedImageSource::SelectedLayer {
            stage: LayerImageStage::PreLocalEffects,
            ..
        }
        | PlannedImageSource::AllBelow(_)
        | PlannedImageSource::ProgramHistory
        // The gesture canvas is etched outside the composition graph, and the
        // programme tap is published outside it, so neither names a producing
        // scope and neither takes part in any scope ordering.
        | PlannedImageSource::GestureCanvas
        | PlannedImageSource::ProgramTap
        | PlannedImageSource::Transparent => None,
    }
}

fn retained_scope_sources(
    _plan: &AdvancedCompositionPlan,
    taps: &[PreparedTap],
) -> BTreeMap<VisualScopeId, usize> {
    let mut retained = BTreeMap::new();
    for (index, tap) in taps.iter().enumerate() {
        if matches!(tap.backing, TapBacking::Current(_)) {
            if let Some(scope) = tap_scope_producer(tap) {
                retained.entry(scope).or_insert(index);
            }
        }
    }
    retained
}

fn scheduled_source(
    scope: VisualScopeId,
    current: VisualScopeId,
    first: bool,
    retained: &BTreeMap<VisualScopeId, usize>,
) -> Result<ScheduledSource, CompositionGpuError> {
    if scope == current && first {
        return Ok(ScheduledSource::Ping);
    }
    retained
        .get(&scope)
        .copied()
        .map(ScheduledSource::RetainedTap)
        .ok_or_else(|| {
            CompositionGpuError::InvalidSchedule(format!(
                "scope {scope:?} executed before structural admission without a current retained tap"
            ))
        })
}

type BlockSchedules = (
    Box<[RootScheduleEntry]>,
    BTreeMap<GroupId, Box<[MemberScheduleEntry]>>,
);

fn build_block_schedules(
    plan: &AdvancedCompositionPlan,
    retained: &BTreeMap<VisualScopeId, usize>,
) -> Result<BlockSchedules, CompositionGpuError> {
    let root_desired = plan
        .root()
        .iter()
        .map(|item| match *item {
            RuntimeRootItem::Layer { layer_id, .. } => RootTask::Layer(layer_id),
            RuntimeRootItem::Group { group_id } => RootTask::Group(group_id),
        })
        .collect::<Vec<_>>();
    let grouped_layers: BTreeMap<_, _> = plan
        .groups()
        .iter()
        .flat_map(|group| group.members.iter().map(move |id| (*id, group.id)))
        .collect();
    let mut root_execution = Vec::new();
    for scope in plan.execution_order() {
        let task = match *scope {
            VisualScopeId::Layer(id) if !grouped_layers.contains_key(&id) => {
                Some(RootTask::Layer(id))
            }
            VisualScopeId::Group(id) => Some(RootTask::Group(id)),
            _ => None,
        };
        if let Some(task) = task {
            root_execution.push(task);
        }
    }
    if root_execution.len() != root_desired.len()
        || root_execution
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != root_desired.len()
    {
        return Err(CompositionGpuError::InvalidSchedule(
            "planner execution order did not contain every root block exactly once".into(),
        ));
    }
    let mut rendered = BTreeSet::new();
    let mut cursor = 0_usize;
    let mut root_schedule = Vec::with_capacity(root_execution.len());
    for task in root_execution {
        rendered.insert(task);
        let mut drains = Vec::new();
        while root_desired
            .get(cursor)
            .is_some_and(|candidate| rendered.contains(candidate))
        {
            let item = root_desired[cursor];
            let source = scheduled_source(
                item.output_scope(),
                task.output_scope(),
                drains.is_empty(),
                retained,
            )?;
            drains.push(ScheduledAdmission { item, source });
            cursor += 1;
        }
        root_schedule.push(RootScheduleEntry {
            task,
            drains: drains.into_boxed_slice(),
        });
    }
    if cursor != root_desired.len() {
        return Err(CompositionGpuError::InvalidSchedule(
            "root block schedule did not drain the complete back-to-front stack".into(),
        ));
    }

    let mut member_schedules = BTreeMap::new();
    for group in plan.groups() {
        let desired = group.members.as_ref();
        let execution = plan
            .execution_order()
            .iter()
            .filter_map(|scope| match *scope {
                VisualScopeId::Layer(id) if desired.contains(&id) => Some(id),
                _ => None,
            })
            .collect::<Vec<_>>();
        if execution.len() != desired.len() {
            return Err(CompositionGpuError::InvalidSchedule(format!(
                "group {} execution block lost a member",
                group.id.get()
            )));
        }
        let mut rendered = BTreeSet::new();
        let mut cursor = 0_usize;
        let mut entries = Vec::with_capacity(execution.len());
        for layer_id in execution {
            rendered.insert(layer_id);
            let mut drains = Vec::new();
            while desired
                .get(cursor)
                .is_some_and(|candidate| rendered.contains(candidate))
            {
                let item = desired[cursor];
                let source = scheduled_source(
                    VisualScopeId::Layer(item),
                    VisualScopeId::Layer(layer_id),
                    drains.is_empty(),
                    retained,
                )?;
                drains.push(ScheduledAdmission { item, source });
                cursor += 1;
            }
            entries.push(MemberScheduleEntry {
                layer_id,
                drains: drains.into_boxed_slice(),
            });
        }
        if cursor != desired.len() {
            return Err(CompositionGpuError::InvalidSchedule(format!(
                "group {} member schedule did not drain completely",
                group.id.get()
            )));
        }
        member_schedules.insert(group.id, entries.into_boxed_slice());
    }
    Ok((root_schedule.into_boxed_slice(), member_schedules))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        GroupName, RuntimeComposition, RuntimeGroup, RuntimeGroupMembers, RuntimeRootItem,
    };
    use crate::effects::params::{
        CollisionAtlasParams, TemporalInterpolation, TemporalLoomParams, TemporalOriginalsParams,
        TemporalParams, TemporalTopology,
    };
    use crate::effects::EffectUniforms;
    use crate::evaluated_frame::evaluated_composition::{
        CompositionPlanInput, LayerMotionPlanInput, MotionCodecFrameFacts,
    };
    use crate::evaluated_frame::{
        EvaluatedFramePlan, FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap,
    };
    use crate::modulation::ModMatrix;
    use crate::motion::{
        CurvedShutterParams, CurvedShutterQuality, FaradayParams, FieldColliderMode,
        FieldColliderParams, MotionBoundaryMode, MotionCarrier, MotionDeviceLimits, MotionDonor,
        MotionField, MotionFieldOrigin, MotionFieldSource, MotionGrid, MotionParams,
        MotionVectorSample,
    };
    use crate::ntsc::NtscParams;
    use crate::performance::SavedLayerPosition;
    use crate::spatial::SpatialTransform;
    use crate::temporal::{TemporalFrameEvents, TemporalFrameInput, TemporalFreezeState};
    use crate::visual_rack::{
        DigitalColorParams, LegacyRackScope, ResolvedImageTap, RuntimeImageMatte,
        RuntimeMaskParams, RuntimeVisualNodeKind, RuntimeVisualRack,
    };

    #[test]
    fn composition_timing_preserves_the_full_temporal_event_batch() {
        let input = TemporalFrameInput::new(
            1.0 / 59.94,
            TemporalFreezeState::MediaFrozen,
            true,
            TemporalFrameEvents {
                boundary_events: 2,
                downbeat_events: 3,
                audio_onset_events: 4,
                manual_events: 5,
                garden_refresh_events: 6,
            },
        );
        assert_eq!(
            CompositionFrameTiming::from_temporal_input(input).temporal_input(),
            input
        );
        assert_eq!(
            CompositionFrameTiming::new(1.0 / 30.0, false)
                .temporal_input()
                .freeze,
            TemporalFreezeState::ProgramFrozen
        );
    }

    /// A tapless Advanced composition must schedule.
    ///
    /// `execution_order` is a topological sort whose ready set is drained with
    /// `BTreeSet::pop_first`, i.e. by ascending `StableLayerId`.
    /// `build_block_schedules` then drains the composition's *back-to-front*
    /// stack as each scope renders, and only the first drain after a task may
    /// read that task's own output; every later one needs a retained tap.
    ///
    /// When no layer taps another there are no edges to order the siblings, so
    /// the sort falls back to id order. A stack whose ids ascend front-to-back
    /// — exactly what an export job produces, numbering layers `position + 1` —
    /// therefore executes in the reverse of the order the schedule wants, and
    /// preparation fails on a layer that owns no tap to retain.
    ///
    /// This needs no GPU and no rack node: an ordinary Faraday transplant is
    /// enough to force an Advanced plan.
    #[test]
    fn a_tapless_advanced_stack_schedules_in_composition_order() {
        use crate::evaluated_frame::evaluated_composition::CompositionPlanInput;

        // Front-to-back [1, 2] means layer 1 is the front, so the back-to-front
        // composite order is [2, 1] while ascending id order is [1, 2]. The two
        // disagree, which is the whole defect.
        let dimensions = [16, 16];
        let base = evaluated_base(&[1, 2], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(2),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let motion = [
            LayerMotionPlanInput {
                stable_id: stable_layer(1),
                params: MotionParams {
                    transplant: FaradayParams {
                        amount: 0.75,
                        donor: MotionDonor::Selected {
                            layer_id: stable_layer(2),
                            saved_position: SavedLayerPosition::new(1).unwrap(),
                        },
                        ..Default::default()
                    },
                    ..MotionParams::default()
                },
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(2),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        let evaluated = EvaluatedCompositionPlan::evaluate(
            &base,
            CompositionPlanInput::new(&composition, &master, &racks).with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = evaluated else {
            panic!("an admitted transplant forces an Advanced plan");
        };

        // No layer taps any other, so nothing is retained anywhere.
        assert!(
            advanced.image_taps().is_empty(),
            "this fixture must be genuinely tapless, or it proves nothing"
        );

        // Among scopes with no ordering edge between them, execution follows
        // the composition's back-to-front order rather than ascending id.
        let layers_in_execution: Vec<_> = advanced
            .execution_order()
            .iter()
            .filter_map(|scope| match scope {
                VisualScopeId::Layer(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(
            layers_in_execution,
            vec![stable_layer(2), stable_layer(1)],
            "independent siblings must execute back-to-front, not by ascending id"
        );

        // The renderer's schedule therefore builds against an EMPTY retained
        // map: every drain reads the output of the task that just rendered.
        let (root, members) = build_block_schedules(&advanced, &BTreeMap::new())
            .expect("a tapless Advanced stack must schedule");
        assert!(members.is_empty());
        assert_eq!(root.len(), 2);
        for entry in root.iter() {
            assert_eq!(entry.drains.len(), 1, "each task drains exactly itself");
            assert!(
                matches!(entry.drains[0].source, ScheduledSource::Ping),
                "a self-drain reads Ping, never a retained tap"
            );
        }
    }

    /// The composite-rank tie-break must survive collapsed groups.
    ///
    /// `atomic_group_execution_order` runs a *second* topological sort over
    /// group-collapsed tasks, and its output — not the scope sort's — becomes
    /// the plan's execution order. Both sorts therefore carry the same rank.
    /// A group is ranked at its own root slot, which `below_topology` records,
    /// so a group task sorts into exactly the slot its members occupy and the
    /// two sorts cannot disagree about where the block belongs.
    ///
    /// Here the group sits at the BOTTOM of the stack while holding the LOWEST
    /// member ids, so id order and composite order disagree for the group as a
    /// whole — the case a rank that only understood loose layers would miss.
    #[test]
    fn the_composite_rank_tie_break_survives_collapsed_groups() {
        use crate::evaluated_frame::evaluated_composition::CompositionPlanInput;

        let dimensions = [16, 16];
        // Front-to-back [9, 1, 2]: the loose layer 9 is the front, and the
        // group holding 1 and 2 is beneath it.
        let base = evaluated_base(&[9, 1, 2], dimensions);
        let group_id = GroupId::new(5).unwrap();
        let group = RuntimeGroup {
            id: group_id,
            name: GroupName::new("collapsed").unwrap(),
            // Member order is back-to-front inside the group, so layer 2 sits
            // beneath layer 1 and the two disagree with ascending id order.
            members: RuntimeGroupMembers::try_from_vec(vec![stable_layer(2), stable_layer(1)])
                .unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Group),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        };
        let composition = RuntimeComposition::try_from_parts(
            vec![group],
            vec![
                RuntimeRootItem::Group { group_id },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(9),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![
            (
                stable_layer(9),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(2),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let motion = [
            LayerMotionPlanInput {
                stable_id: stable_layer(9),
                params: MotionParams {
                    transplant: FaradayParams {
                        amount: 0.75,
                        donor: MotionDonor::Selected {
                            layer_id: stable_layer(1),
                            saved_position: SavedLayerPosition::new(1).unwrap(),
                        },
                        ..Default::default()
                    },
                    ..MotionParams::default()
                },
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(1),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(2),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        let evaluated = EvaluatedCompositionPlan::evaluate(
            &base,
            CompositionPlanInput::new(&composition, &master, &racks).with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = evaluated else {
            panic!("an admitted transplant forces an Advanced plan");
        };
        assert!(advanced.image_taps().is_empty(), "genuinely tapless");

        // The group's members stay contiguous and in member order, the whole
        // group block precedes the loose layer above it, and the block sits
        // where the composition put it rather than where the ids would.
        let layers_in_execution: Vec<_> = advanced
            .execution_order()
            .iter()
            .filter_map(|scope| match scope {
                VisualScopeId::Layer(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(
            layers_in_execution,
            vec![stable_layer(2), stable_layer(1), stable_layer(9)],
            "a collapsed group must execute as one contiguous back-to-front block"
        );

        // And the renderer schedules it with nothing retained.
        let (root, members) = build_block_schedules(&advanced, &BTreeMap::new())
            .expect("a tapless grouped stack must schedule");
        assert_eq!(root.len(), 2);
        assert_eq!(members.len(), 1);
        for entry in root.iter() {
            assert_eq!(entry.drains.len(), 1);
            assert!(matches!(entry.drains[0].source, ScheduledSource::Ping));
        }
    }

    fn stable_layer(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn recorder_metadata(capture_index: u64) -> crate::program_recorder::RecorderFrameMetadata {
        crate::program_recorder::RecorderFrameMetadata {
            capture_index,
            capture_time_ns: capture_index * 1_000,
            program_time_ns: capture_index * 2_000,
            visual_epoch: 9,
            program_frozen: false,
            media_frozen: false,
            blackout: false,
            audio_clock: None,
        }
    }

    fn evaluated_base(ids_front_to_back: &[u64], dimensions: [u32; 2]) -> EvaluatedFramePlan {
        evaluated_base_with_temporal(ids_front_to_back, dimensions, &TemporalParams::default())
    }

    fn evaluated_base_with_temporal(
        ids_front_to_back: &[u64],
        dimensions: [u32; 2],
        temporal: &TemporalParams,
    ) -> EvaluatedFramePlan {
        evaluated_base_with_temporal_bypass(ids_front_to_back, dimensions, temporal, &[])
    }

    fn evaluated_base_with_temporal_bypass(
        ids_front_to_back: &[u64],
        dimensions: [u32; 2],
        temporal: &TemporalParams,
        bypass_temporal: &[u64],
    ) -> EvaluatedFramePlan {
        let effects = vec![EffectUniforms::default(); ids_front_to_back.len()];
        let transforms = vec![SpatialTransform::new_layer_default(); ids_front_to_back.len()];
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let matrix = ModMatrix::new();
        let modulation = matrix.frame(ids_front_to_back.len());
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(dimensions[0], dimensions[1], 2.0),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal,
            },
            ids_front_to_back
                .iter()
                .enumerate()
                .map(|(slot, id)| LayerFrameInput {
                    source: SourceTap::new(*id, slot, dimensions[0], dimensions[1]),
                    effects: &effects[slot],
                    transform: &transforms[slot],
                    opacity: 1.0,
                    mosh_send: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    bypass_temporal_fx: bypass_temporal.contains(id),
                    pattern: None,
                }),
        )
    }

    #[test]
    fn temporal_dry_surfaces_reconcile_one_for_one_with_the_advanced_ledger() {
        let base = evaluated_base_with_temporal_bypass(
            &[1, 2, 3],
            [64, 48],
            &TemporalParams::default(),
            &[1, 2],
        );
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(3),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = [1, 2, 3]
            .map(|id| {
                (
                    stable_layer(id),
                    RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
                )
            })
            .to_vec();
        racks[0]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
                invert: 1.0,
                ..DigitalColorParams::default()
            }))
            .unwrap();
        let planned = EvaluatedCompositionPlan::evaluate(
            &base,
            CompositionPlanInput::new(&composition, &master, &racks),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = planned else {
            panic!("the custom layer rack must select Advanced");
        };
        assert_eq!(
            advanced.temporal_dry_layers(),
            &[stable_layer(2), stable_layer(1)]
        );
        validate_actual_surface_ledger(&advanced, false, &[], 0, 2)
            .expect("two planned dry layers must reconcile with two retained surfaces");
        assert!(validate_actual_surface_ledger(&advanced, false, &[], 0, 1).is_err());
        assert!(validate_actual_surface_ledger(&advanced, false, &[], 0, 3).is_err());
    }

    struct GpuHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    struct TestSource {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        dimensions: [u32; 2],
    }

    impl GpuHarness {
        fn new() -> Self {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .expect("GPU adapter for advanced composition test");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Advanced composition test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("GPU device for advanced composition test");
            Self { device, queue }
        }

        fn source(&self, dimensions: [u32; 2], rgba: [u8; 4], label: &'static str) -> TestSource {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let source = TestSource {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                texture,
                dimensions,
            };
            self.write_source(&source, rgba);
            source
        }

        /// A deliberately non-uniform carrier. A displacement that moves a
        /// sample by a fraction of the frame is invisible against a flat
        /// colour, so the payoff fixture needs a source whose pixels actually
        /// differ from their neighbours.
        fn patterned_source(&self, dimensions: [u32; 2], label: &'static str) -> TestSource {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut bytes = Vec::with_capacity(dimensions[0] as usize * dimensions[1] as usize * 4);
            for y in 0..dimensions[1] {
                for x in 0..dimensions[0] {
                    let checker = u8::from((x / 3 + y / 3) % 2 == 0) * 200;
                    bytes.extend_from_slice(&[
                        checker.saturating_add((x * 5) as u8),
                        (y * 7) as u8,
                        255 - checker,
                        255,
                    ]);
                }
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimensions[0] * 4),
                    rows_per_image: Some(dimensions[1]),
                },
                wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
            );
            TestSource {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                texture,
                dimensions,
            }
        }

        fn write_source(&self, source: &TestSource, rgba: [u8; 4]) {
            let bytes = rgba
                .into_iter()
                .cycle()
                .take(source.dimensions[0] as usize * source.dimensions[1] as usize * 4)
                .collect::<Vec<_>>();
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &source.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(source.dimensions[0] * 4),
                    rows_per_image: Some(source.dimensions[1]),
                },
                wgpu::Extent3d {
                    width: source.dimensions[0],
                    height: source.dimensions[1],
                    depth_or_array_layers: 1,
                },
            );
        }

        fn render(
            &self,
            executor: &mut CompositionGpuExecutor,
            plan: &EvaluatedCompositionPlan,
            delta_seconds: f32,
            commit: bool,
        ) -> Vec<[f32; 4]> {
            self.render_with_motion(
                executor,
                plan,
                delta_seconds,
                commit,
                CompositionMotionFrameInput::default(),
            )
        }

        fn render_with_motion(
            &self,
            executor: &mut CompositionGpuExecutor,
            plan: &EvaluatedCompositionPlan,
            delta_seconds: f32,
            commit: bool,
            motion_input: CompositionMotionFrameInput<'_>,
        ) -> Vec<[f32; 4]> {
            let dimensions = executor.dimensions();
            let unpadded_row = dimensions[0] * 8;
            let padded_row = (unpadded_row + 255) & !255;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Advanced composition test readback"),
                size: u64::from(padded_row) * u64::from(dimensions[1]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Advanced composition test encoder"),
                });
            executor
                .encode_with_motion(
                    &self.queue,
                    &mut encoder,
                    plan,
                    CompositionFrameTiming::new(delta_seconds, true),
                    motion_input,
                )
                .unwrap();
            let output = executor.output();
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(dimensions[1]),
                    },
                },
                wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            let (send, receive) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("advanced composition GPU wait");
            receive.recv().expect("map callback").expect("map result");
            if commit {
                executor.commit_frame_history();
            } else {
                executor.discard_frame_history();
            }
            let mapped = slice.get_mapped_range();
            let mut pixels = Vec::with_capacity(dimensions[0] as usize * dimensions[1] as usize);
            for row in 0..dimensions[1] as usize {
                let start = row * padded_row as usize;
                for pixel in mapped[start..start + unpadded_row as usize].chunks_exact(8) {
                    pixels.push(std::array::from_fn(|channel| {
                        let offset = channel * 2;
                        half_to_f32(u16::from_le_bytes([pixel[offset], pixel[offset + 1]]))
                    }));
                }
            }
            drop(mapped);
            staging.unmap();
            pixels
        }

        fn submit(
            &self,
            executor: &mut CompositionGpuExecutor,
            plan: &EvaluatedCompositionPlan,
            delta_seconds: f32,
            commit: bool,
        ) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Advanced composition temporal sequence encoder"),
                });
            executor
                .encode(
                    &self.queue,
                    &mut encoder,
                    plan,
                    CompositionFrameTiming::new(delta_seconds, true),
                )
                .unwrap();
            self.queue.submit(std::iter::once(encoder.finish()));
            if commit {
                executor.commit_frame_history();
            } else {
                executor.discard_frame_history();
            }
        }

        fn submit_with_motion_temporal(
            &self,
            executor: &mut CompositionGpuExecutor,
            plan: &EvaluatedCompositionPlan,
            temporal: TemporalFrameInput,
            motion_input: CompositionMotionFrameInput<'_>,
            commit: bool,
        ) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Advanced composition motion truth encoder"),
                });
            executor
                .encode_with_motion(
                    &self.queue,
                    &mut encoder,
                    plan,
                    CompositionFrameTiming::from_temporal_input(temporal),
                    motion_input,
                )
                .unwrap();
            self.queue.submit(std::iter::once(encoder.finish()));
            if commit {
                executor.commit_frame_history();
            } else {
                executor.discard_frame_history();
            }
        }
    }

    fn half_to_f32(value: u16) -> f32 {
        let sign = u32::from(value & 0x8000) << 16;
        let exponent = (value >> 10) & 0x1f;
        let fraction = value & 0x03ff;
        let bits = if exponent == 0 {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = u32::from(fraction);
                let mut exponent = -14_i32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                fraction &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
            }
        } else if exponent == 0x1f {
            sign | 0x7f80_0000 | (u32::from(fraction) << 13)
        } else {
            sign | (u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13)
        };
        f32::from_bits(bits)
    }

    fn group_rack_matte_bus_fixture(
        dimensions: [u32; 2],
    ) -> (EvaluatedCompositionPlan, RuntimeComposition) {
        let base = evaluated_base(&[2, 1], dimensions);
        let mut group_rack = RuntimeVisualRack::empty();
        group_rack
            .push(RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
                invert: 1.0,
                ..Default::default()
            }))
            .unwrap();
        let group_id = GroupId::new(10).unwrap();
        let group = RuntimeGroup {
            id: group_id,
            name: GroupName::new("Golden group").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![stable_layer(2)]).unwrap(),
            opacity: 0.5,
            transform: SpatialTransform::default(),
            rack: group_rack,
            matte: Some(RuntimeImageMatte {
                tap: ResolvedImageTap {
                    source: crate::visual_rack::ResolvedImageSource::SelectedLayer {
                        layer_id: stable_layer(1),
                        saved_position: SavedLayerPosition::new(2).unwrap(),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                channel: MatteChannel::Alpha,
                invert: false,
                amount: 1.0,
                threshold: 0.5,
                softness: 1.0,
            }),
            solo: false,
            bypass: false,
            bus: BusAssignment::B,
        };
        let composition = RuntimeComposition::try_from_parts(
            vec![group],
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::A,
                },
                RuntimeRootItem::Group { group_id },
            ],
            None,
            0.25,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![
            (
                stable_layer(2),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let plan = EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            ),
        )
        .unwrap();
        assert!(matches!(plan, EvaluatedCompositionPlan::Advanced(_)));
        (plan, composition)
    }

    /// A canvas route is charged exactly once — by `GestureCanvasPlan` — and
    /// must never also claim a retained tap surface here. The declared and
    /// actual ledgers are reconciled by `validate_actual_surface_ledger`, which
    /// fails closed on any disagreement, so this is the fixture that proves the
    /// two sides agree at zero rather than merely asserting it in prose.
    #[test]
    fn a_gesture_canvas_tap_charges_no_retained_surface_on_either_side_of_the_ledger() {
        let base = evaluated_base(&[2, 1], [64, 48]);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        let canvas_displace = |timing| {
            RuntimeVisualNodeKind::Displace(crate::visual_rack::RuntimeDisplaceParams {
                tap: ResolvedImageTap {
                    source: crate::visual_rack::ResolvedImageSource::GestureCanvas,
                    timing,
                },
                amount_x: 0.5,
                amount_y: -0.25,
                boundary: crate::visual_rack::DisplaceBoundary::Wrap,
            })
        };
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        master
            .push(canvas_displace(EdgeTiming::CurrentFrame))
            .unwrap();
        let mut layer_rack = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        // Both timings, because an N-1 route is where a parity pair would
        // otherwise be allocated behind the ledger's back.
        layer_rack
            .push(canvas_displace(EdgeTiming::PreviousFrame))
            .unwrap();
        let racks = vec![
            (stable_layer(2), layer_rack),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let plan = EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_gesture_canvas(true),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(plan) = plan else {
            panic!("a canvas-routed Displace is an advanced composition");
        };
        assert_eq!(plan.image_taps().len(), 2);
        assert!(plan.image_taps().iter().all(|tap| tap.resolved
            == crate::evaluated_frame::evaluated_composition::PlannedImageSource::GestureCanvas));

        // The backings the executor would build for these taps, by the same law
        // `prepare_tap_surfaces` applies — and none of them owns a surface.
        let taps: Vec<_> = plan
            .image_taps()
            .iter()
            .cloned()
            .map(|planned| PreparedTap {
                planned,
                backing: TapBacking::GestureCanvas,
            })
            .collect();
        validate_actual_surface_ledger(&plan, false, &taps, 0, 0)
            .expect("a canvas route charges zero retained surfaces on both sides");

        // The validator really is live: one extra actual surface fails closed.
        assert!(validate_actual_surface_ledger(&plan, false, &taps, 1, 0).is_err());
        assert!(validate_actual_surface_ledger(&plan, true, &taps, 0, 0).is_err());
    }

    /// A programme-tap route is charged exactly once — by the renderer-owned
    /// full-frame texture floor — and must never also claim a retained tap
    /// surface here. Same fail-closed reconcile as the canvas fixture above.
    #[test]
    fn a_program_tap_charges_no_retained_surface_on_either_side_of_the_ledger() {
        let base = evaluated_base(&[2, 1], [64, 48]);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        let tap_displace = |timing| {
            RuntimeVisualNodeKind::Displace(crate::visual_rack::RuntimeDisplaceParams {
                tap: ResolvedImageTap {
                    source: crate::visual_rack::ResolvedImageSource::ProgramTap,
                    timing,
                },
                amount_x: 0.5,
                amount_y: -0.25,
                boundary: crate::visual_rack::DisplaceBoundary::Wrap,
            })
        };
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        master.push(tap_displace(EdgeTiming::CurrentFrame)).unwrap();
        let mut layer_rack = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        // Both timings, because an N-1 route is where a parity pair would
        // otherwise be allocated behind the ledger's back — and the tap *is*
        // the N-1 image, so nothing may stage for it.
        layer_rack
            .push(tap_displace(EdgeTiming::PreviousFrame))
            .unwrap();
        let racks = vec![
            (stable_layer(2), layer_rack),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let plan = EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_program_tap(true),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(plan) = plan else {
            panic!("a tap-routed Displace is an advanced composition");
        };
        assert_eq!(plan.image_taps().len(), 2);
        assert!(plan.image_taps().iter().all(|tap| tap.resolved
            == crate::evaluated_frame::evaluated_composition::PlannedImageSource::ProgramTap));

        // The backings the executor would build for these taps, by the same law
        // `prepare_tap_surfaces` applies — and none of them owns a surface.
        let taps: Vec<_> = plan
            .image_taps()
            .iter()
            .cloned()
            .map(|planned| PreparedTap {
                planned,
                backing: TapBacking::ProgramTap,
            })
            .collect();
        validate_actual_surface_ledger(&plan, false, &taps, 0, 0)
            .expect("a programme-tap route charges zero retained surfaces on both sides");

        // The validator really is live: one extra actual surface fails closed.
        assert!(validate_actual_surface_ledger(&plan, false, &taps, 1, 0).is_err());
        assert!(validate_actual_surface_ledger(&plan, true, &taps, 0, 0).is_err());
    }

    /// One layer carrying one Displace node whose donor is the etched gesture
    /// canvas. This is the shape the payoff fixture renders three ways.
    fn gesture_donor_plan(dimensions: [u32; 2], amount: f32) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        layer_rack
            .push(RuntimeVisualNodeKind::Displace(
                crate::visual_rack::RuntimeDisplaceParams {
                    tap: ResolvedImageTap {
                        source: crate::visual_rack::ResolvedImageSource::GestureCanvas,
                        timing: EdgeTiming::CurrentFrame,
                    },
                    amount_x: amount,
                    amount_y: amount,
                    boundary: crate::visual_rack::DisplaceBoundary::Hold,
                },
            ))
            .unwrap();
        let racks = vec![(stable_layer(1), layer_rack)];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_gesture_canvas(true),
        )
        .unwrap()
    }

    /// THE PAYOFF. A recorded stroke is etched into a real GPU canvas, that
    /// canvas is routed as the donor of a real Displace node over a real
    /// carrier, and the result is read back off the device.
    ///
    /// Three claims, and each of them is the whole point of wiring the canvas:
    ///
    /// (a) the live 30 Hz derivation and the offline one produce *byte
    ///     identical* presented donor images — the same recorded performance
    ///     lands on the same addresses in both sessions;
    /// (b) the rendered pixels differ from the same plan rendered without a
    ///     canvas — the recording actually reaches the image, rather than
    ///     merely existing beside it;
    /// (c) an admitted but never-etched canvas renders byte-identically to no
    ///     canvas at all — an unrecorded field is inert, which is the honesty
    ///     law spelled in pixels.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_a_recorded_gesture_reaches_the_image_through_a_routed_displace_donor() {
        use crate::gesture::{GestureEvent, GestureEventRecorder, GestureMode, GesturePhase};
        use crate::gesture_canvas::{
            GestureCanvasFrameInput, GestureCanvasGrid, GestureCanvasLimits, GestureCanvasParams,
            GestureCanvasRequest, GestureCanvasState, GESTURE_CANVAS_MAX_EDGE,
        };
        use crate::render_export::export_temporal_reference_tick;
        use crate::renderer::gesture_canvas::{
            GestureCanvasResources, GESTURE_CANVAS_PRESENTED_TEXEL_BYTES,
        };

        let gpu = GpuHarness::new();
        let dimensions = [48_u32, 32];
        let canvas_grid = GestureCanvasGrid::new(24, 16).unwrap();
        let canvas_limits = GestureCanvasLimits::device(GESTURE_CANVAS_MAX_EDGE, 256);
        let params = GestureCanvasParams {
            radius: 0.6,
            strength: 1.0,
            retention: 1.0,
        };
        let fps = 30_u32;
        let frames = 8_u64;

        // A short deterministic stroke straight down the middle. Recorded
        // once, replayed twice.
        let mut track = crate::gesture::GestureTrack::default();
        for (tick, phase, x) in [
            (0_u64, GesturePhase::Begin, 0.35_f32),
            (2, GesturePhase::Move, 0.5),
            (4, GesturePhase::Move, 0.65),
            (6, GesturePhase::End, 0.8),
        ] {
            assert!(
                track
                    .record_accepted(
                        tick,
                        GestureEvent::quantized(
                            0,
                            phase,
                            GestureMode::Push,
                            [x, 0.5],
                            1.0,
                            [1.0, 0.0],
                        ),
                    )
                    .unwrap()
            );
        }

        // The two reference-tick derivations, proven equal before a pixel is
        // rendered so a later pixel difference cannot be blamed on the clock.
        let mut live_ticks = Vec::with_capacity(frames as usize);
        let mut recorder = GestureEventRecorder::default();
        for _ in 0..frames {
            live_ticks.push(recorder.reference_tick());
            recorder.record_accepted(1.0 / fps as f32, &[]);
        }
        let export_ticks: Vec<u64> = (0..frames)
            .map(|frame| export_temporal_reference_tick(frame, fps))
            .collect();
        assert_eq!(live_ticks, export_ticks);

        // Etch one session. The ordering is the wired ordering exactly:
        // stage, encode the staged transaction, then commit.
        let etch = |ticks: &[u64]| {
            let mut resources = GestureCanvasResources::prepare(
                &gpu.device,
                &gpu.queue,
                &[GestureCanvasRequest::new(canvas_grid)],
                canvas_limits,
            )
            .unwrap();
            let mut state = GestureCanvasState::new(canvas_grid, params).unwrap();
            let mut replay = track.replay();
            for tick in ticks {
                let due = replay.events_due(u32::try_from(*tick).unwrap());
                state
                    .stage_frame(GestureCanvasFrameInput {
                        reference_tick: *tick,
                        program_advances: true,
                        events: due,
                        evaluated_params: Some(params),
                    })
                    .unwrap();
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Gesture payoff etch"),
                        });
                resources
                    .encode_staged_frame(&gpu.queue, &mut encoder, 0, &state)
                    .unwrap();
                gpu.queue.submit(std::iter::once(encoder.finish()));
                state.commit_staged();
            }
            resources
        };

        let live_resources = etch(&live_ticks);
        let export_resources = etch(&export_ticks);

        // (a) The two presented donor images are byte identical.
        let read_presented = |resources: &GestureCanvasResources| {
            let canvas = resources.canvas(0).unwrap();
            let unpadded = canvas_grid.width * GESTURE_CANVAS_PRESENTED_TEXEL_BYTES;
            let padded = (unpadded + 255) & !255;
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gesture payoff presented readback"),
                size: u64::from(padded) * u64::from(canvas_grid.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gesture payoff presented encoder"),
                });
            encoder.copy_texture_to_buffer(
                canvas.presented_texture().as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(canvas_grid.height),
                    },
                },
                wgpu::Extent3d {
                    width: canvas_grid.width,
                    height: canvas_grid.height,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            let (send, receive) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            receive.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let mut compact = Vec::new();
            for row in 0..canvas_grid.height as usize {
                let start = row * padded as usize;
                compact.extend_from_slice(&mapped[start..start + unpadded as usize]);
            }
            drop(mapped);
            staging.unmap();
            compact
        };
        let live_presented = read_presented(&live_resources);
        assert_eq!(
            live_presented,
            read_presented(&export_resources),
            "the live and offline presented gesture fields differ"
        );
        assert!(
            live_presented.iter().any(|byte| *byte != 0),
            "the recorded stroke etched nothing at all"
        );

        // Render the same plan three ways.
        let source = gpu.patterned_source(dimensions, "Gesture payoff carrier");
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let plan = gesture_donor_plan(dimensions, 0.4);
        let render = |binding: GestureCanvasBinding| {
            let mut executor =
                CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
            let bound = binding.is_bound();
            executor.bind_gesture_canvas(binding);
            assert_eq!(executor.gesture_canvas_bound(), bound);
            executor
                .prepare(&gpu.device, &gpu.queue, &plan, &sources)
                .unwrap();
            gpu.render(&mut executor, &plan, 1.0 / 30.0, true)
        };

        let unbound = render(GestureCanvasBinding::default());
        let empty_canvas = GestureCanvasResources::prepare(
            &gpu.device,
            &gpu.queue,
            &[GestureCanvasRequest::new(canvas_grid)],
            canvas_limits,
        )
        .unwrap();
        let empty = render(GestureCanvasBinding::bound(
            empty_canvas.presented_view().unwrap().clone(),
            1,
        ));
        let etched = render(GestureCanvasBinding::bound(
            live_resources.presented_view().unwrap().clone(),
            2,
        ));

        // (c) An admitted but never-etched canvas is exactly no canvas at all.
        // This is arithmetic, not tolerance: a fully transparent donor and a
        // missing binding both decode to exactly zero displacement.
        assert_eq!(
            empty, unbound,
            "an unrecorded gesture canvas changed the rendered image"
        );

        // (b) The recording reaches the image.
        let differing = |left: &[[f32; 4]], right: &[[f32; 4]]| {
            left.iter()
                .zip(right)
                .filter(|(a, b)| {
                    a.iter()
                        .zip(b.iter())
                        .any(|(left, right)| (left - right).abs() > 1.0e-3)
                })
                .count()
        };
        assert!(
            differing(&etched, &unbound) > 0,
            "the etched gesture field displaced nothing; the donor never reached the node"
        );

        // One executor, one topology, two different canvases. The tap bind
        // groups are built at prepare time and reused across frames, so without
        // the canvas identity in the reuse test this second prepare would keep
        // the empty canvas's view and the image would not move at all.
        let mut reused = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        reused.bind_gesture_canvas(GestureCanvasBinding::bound(
            empty_canvas.presented_view().unwrap().clone(),
            1,
        ));
        reused
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let reused_empty = gpu.render(&mut reused, &plan, 1.0 / 30.0, true);
        assert_eq!(reused_empty, empty);
        reused.bind_gesture_canvas(GestureCanvasBinding::bound(
            live_resources.presented_view().unwrap().clone(),
            2,
        ));
        reused
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let reused_etched = gpu.render(&mut reused, &plan, 1.0 / 30.0, true);
        assert!(
            differing(&reused_etched, &reused_empty) > 0,
            "a rebuilt canvas left a stale presented view bound to an unchanged topology"
        );
    }

    /// One layer carrying one Displace node whose donor is the programme tap.
    /// This is the shape the B16 payoff fixture renders three ways.
    fn program_tap_donor_plan(dimensions: [u32; 2], amount: f32) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        layer_rack
            .push(RuntimeVisualNodeKind::Displace(
                crate::visual_rack::RuntimeDisplaceParams {
                    tap: ResolvedImageTap {
                        source: crate::visual_rack::ResolvedImageSource::ProgramTap,
                        timing: EdgeTiming::CurrentFrame,
                    },
                    amount_x: amount,
                    amount_y: amount,
                    boundary: crate::visual_rack::DisplaceBoundary::Hold,
                },
            ))
            .unwrap();
        let racks = vec![(stable_layer(1), layer_rack)];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_program_tap(true),
        )
        .unwrap()
    }

    /// The B16 payoff in executor terms. A routed programme tap is rendered
    /// three ways over a real carrier, and each claim is a leg of the tranche:
    ///
    /// (a) an unbound tap and a bound-but-never-published (all-zero) tap are
    ///     byte identical — live's unbound pre-first-commit transparency
    ///     and export's zero-initialized job-lifetime surface are the same
    ///     pixels, by arithmetic (the frozen donor decode yields exactly zero
    ///     for a fully transparent donor);
    /// (b) a published programme copy reaches the pixels through the routed
    ///     donor — the re-entry loop's read half demonstrably works;
    /// (c) invalidating a previously bound tap and then publishing a different
    ///     tap on an unchanged topology both re-prepare, rather than retaining
    ///     the stale view — the patch-generation and renderer-rebuild laws the
    ///     binding identity exists for.
    ///
    /// The N-1 publication half (the copy at the acceptance decision, the
    /// blackout hold, and the byte-identical offline ordering) is pinned by
    /// the source-order tests in `main.rs`/`render_export.rs` and rendered
    /// end to end by `render_program_reentry_pipeline`.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_a_program_tap_donor_feeds_the_previous_frame_back_through_a_routed_displace() {
        let gpu = GpuHarness::new();
        let dimensions = [48_u32, 32];
        let source = gpu.patterned_source(dimensions, "Program tap payoff carrier");
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let plan = program_tap_donor_plan(dimensions, 0.4);
        let render = |binding: ProgramTapBinding| {
            let mut executor =
                CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
            let bound = binding.is_bound();
            executor.bind_program_tap(binding);
            assert_eq!(executor.program_tap_bound(), bound);
            executor
                .prepare(&gpu.device, &gpu.queue, &plan, &sources)
                .unwrap();
            gpu.render(&mut executor, &plan, 1.0 / 30.0, true)
        };

        let unbound = render(ProgramTapBinding::default());

        // A tap surface nothing ever published: wgpu's guaranteed zero
        // initialization is the exact state export's job-lifetime surface has
        // at frame zero.
        let never_published = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Program tap payoff never-published tap"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let zeroed = render(ProgramTapBinding::bound(
            never_published.create_view(&wgpu::TextureViewDescriptor::default()),
            1,
        ));

        // (a) Unpublished and unbound are the same program, in pixels.
        assert_eq!(
            zeroed, unbound,
            "a never-published programme tap changed the rendered image"
        );

        // (b) A published programme copy reaches the image. The donor here is
        // an opaque non-neutral image in the tap's exact slot format, standing
        // for the previous accepted frame's audience copy.
        let previous = gpu.patterned_source(dimensions, "Program tap payoff previous frame");
        let fed = render(ProgramTapBinding::bound(previous.view.clone(), 2));
        let differing = |left: &[[f32; 4]], right: &[[f32; 4]]| {
            left.iter()
                .zip(right)
                .filter(|(a, b)| {
                    a.iter()
                        .zip(b.iter())
                        .any(|(left, right)| (left - right).abs() > 1.0e-3)
                })
                .count()
        };
        assert!(
            differing(&fed, &unbound) > 0,
            "the published programme copy displaced nothing; the tap never reached the node"
        );

        // (c) One executor, one topology, the exact patch lifecycle. It begins
        // bound, invalidates to the default binding without sampling the old
        // patch, then binds the newly published image. Each identity transition
        // must rebuild even though the immutable plan never changes.
        let mut reused = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        reused.bind_program_tap(ProgramTapBinding::bound(
            never_published.create_view(&wgpu::TextureViewDescriptor::default()),
            1,
        ));
        reused
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let reused_zeroed = gpu.render(&mut reused, &plan, 1.0 / 30.0, true);
        assert_eq!(reused_zeroed, zeroed);

        reused.bind_program_tap(ProgramTapBinding::default());
        reused
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        assert!(!reused.program_tap_bound());
        let reused_invalidated = gpu.render(&mut reused, &plan, 1.0 / 30.0, true);
        assert_eq!(
            reused_invalidated, unbound,
            "tap invalidation retained pixels from the previous patch"
        );

        reused.bind_program_tap(ProgramTapBinding::bound(previous.view.clone(), 1));
        reused
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let reused_fed = gpu.render(&mut reused, &plan, 1.0 / 30.0, true);
        assert!(
            differing(&reused_fed, &reused_zeroed) > 0,
            "a rebuilt renderer left a stale programme-tap view bound to an unchanged topology"
        );
    }

    fn motion_shutter_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![(
            stable_layer(1),
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
        )];
        let motion = [LayerMotionPlanInput {
            stable_id: stable_layer(1),
            params: MotionParams {
                shutter: CurvedShutterParams {
                    angle_degrees: 180.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            codec: MotionCodecFrameFacts::default(),
        }];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap()
    }

    fn motion_transform_shutter_fixture(
        dimensions: [u32; 2],
        transform: SpatialTransform,
    ) -> EvaluatedCompositionPlan {
        let effects = EffectUniforms::default();
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let modulation = ModMatrix::new().frame(1);
        let base = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(dimensions[0], dimensions[1], 2.0),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(1, 0, dimensions[0], dimensions[1]),
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
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![(
            stable_layer(1),
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
        )];
        let motion = [LayerMotionPlanInput {
            stable_id: stable_layer(1),
            params: MotionParams {
                field_source: MotionFieldSource::CodecVectors,
                shutter: CurvedShutterParams {
                    angle_degrees: 360.0,
                    quality: CurvedShutterQuality::Live,
                    ..Default::default()
                },
                ..Default::default()
            },
            codec: MotionCodecFrameFacts {
                available: true,
                source_generation: 7,
                frame_ordinal: 9,
            },
        }];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap()
    }

    fn motion_faraday_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[2, 1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![
            (
                stable_layer(2),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let motion = [
            LayerMotionPlanInput {
                stable_id: stable_layer(2),
                params: MotionParams {
                    transplant: FaradayParams {
                        amount: 1.0,
                        donor: MotionDonor::Selected {
                            layer_id: stable_layer(1),
                            saved_position: SavedLayerPosition::new(2).unwrap(),
                        },
                        carrier: MotionCarrier::FirstSourceFrame,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(1),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap()
    }

    fn motion_codec_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![(
            stable_layer(1),
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
        )];
        let motion = [LayerMotionPlanInput {
            stable_id: stable_layer(1),
            params: MotionParams {
                field_source: MotionFieldSource::CodecVectors,
                shutter: CurvedShutterParams {
                    angle_degrees: 180.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            codec: MotionCodecFrameFacts {
                available: true,
                source_generation: 7,
                frame_ordinal: 9,
            },
        }];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            )
            .with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn motion_lattice_shutter_is_warm_transactional_and_static_safe() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let plan = motion_shutter_fixture(dimensions);
        let source = gpu.source(dimensions, [255, 0, 0, 255], "Motion static red source");
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let warmed = executor.allocation_snapshot();
        let advanced = match &plan {
            EvaluatedCompositionPlan::Advanced(plan) => plan,
            EvaluatedCompositionPlan::LegacyExact(_) => panic!("fixture must be Advanced"),
        };
        assert_eq!(warmed.creative_bytes, advanced.resources().creative_bytes);
        assert_eq!(
            warmed.motion_bytes,
            advanced
                .motion()
                .advanced()
                .expect("motion fixture must allocate motion")
                .resources()
                .total_bytes
        );
        assert!(warmed.motion_bytes > 0);
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        assert_eq!(executor.allocation_snapshot(), warmed);

        // Frame one primes luma; frame two publishes a deterministic static
        // field; frame three consumes it through the fixed Sharp shutter.
        gpu.submit(&mut executor, &plan, 1.0 / 30.0, true);
        let primed = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .expect("primed motion metrics");
        assert_eq!(primed.valid_fields, 0);
        assert_eq!(primed.field_origin, MotionFieldOrigin::None);
        gpu.submit(&mut executor, &plan, 1.0 / 30.0, true);
        let published = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .expect("published lattice metrics");
        assert_eq!(published.valid_fields, 1);
        assert_eq!(published.field_origin, MotionFieldOrigin::LatticeFallback);
        assert_eq!(
            published.field_source_scope,
            Some(VisualScopeId::Layer(stable_layer(1)))
        );
        let pixels = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        let metrics = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .expect("motion metrics");
        assert_eq!(metrics.memory_generation, 3);
        assert_eq!(metrics.valid_fields, 1);
        assert_eq!(metrics.valid_luma_fields, 1);
        assert_eq!(metrics.shutter_samples, 1);
        assert!(!metrics.frame_staged);
        assert_eq!(executor.allocation_snapshot(), warmed);
        for pixel in pixels {
            assert!((pixel[0] - 1.0).abs() <= 0.01, "{pixel:?}");
            assert!(pixel[1].abs() <= 0.01, "{pixel:?}");
            assert!(pixel[2].abs() <= 0.01, "{pixel:?}");
            assert!((pixel[3] - 1.0).abs() <= 0.01, "{pixel:?}");
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn authored_transform_shutter_uses_the_shared_warm_transactional_plan() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let baseline = motion_transform_shutter_fixture(dimensions, SpatialTransform::default());
        let moved = motion_transform_shutter_fixture(
            dimensions,
            SpatialTransform {
                position: [0.25, 0.0],
                rotation_deg: 12.0,
                ..SpatialTransform::default()
            },
        );
        assert_eq!(baseline.topology_signature(), moved.topology_signature());

        let source = gpu.source(
            dimensions,
            [0, 0, 0, 0],
            "Motion transform patterned source",
        );
        let mut bytes = vec![0_u8; dimensions[0] as usize * dimensions[1] as usize * 4];
        for row in 0..dimensions[1] as usize {
            for column in 0..dimensions[0] as usize / 2 {
                let offset = (row * dimensions[0] as usize + column) * 4;
                bytes[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        gpu.queue.write_texture(
            source.texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * 4),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let grid = MotionGrid::for_source(dimensions, Default::default()).unwrap();
        let zero_field = MotionField::zeroed(dimensions, grid, MotionFieldOrigin::CodecVectors)
            .expect("canonical zero transform fixture field");
        let attachment = MotionFieldAttachment {
            scope: VisualScopeId::Layer(stable_layer(1)),
            source_generation: 7,
            frame_ordinal: 9,
            product_content_sha256: [7; 32],
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            source_dimensions: dimensions,
            grid,
            field: &zero_field,
        };
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &baseline, &sources)
            .unwrap();
        let warmed = executor.allocation_snapshot();
        let _ = gpu.render_with_motion(
            &mut executor,
            &baseline,
            1.0 / 30.0,
            true,
            CompositionMotionFrameInput {
                attachments: &[attachment],
                held_scopes: &[],
            },
        );
        executor
            .prepare(&gpu.device, &gpu.queue, &moved, &sources)
            .unwrap();
        assert_eq!(executor.allocation_snapshot(), warmed);
        let pixels = gpu.render_with_motion(
            &mut executor,
            &moved,
            1.0 / 30.0,
            true,
            CompositionMotionFrameInput {
                attachments: &[attachment],
                held_scopes: &[],
            },
        );

        // Live and export own independent executors but consume this same
        // immutable frame-plan sequence and transaction boundary. Their
        // physical-GPU output must therefore agree without re-resolving
        // transforms or field-source policy in either caller.
        let mut export_executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        export_executor
            .prepare(&gpu.device, &gpu.queue, &baseline, &sources)
            .unwrap();
        let _ = gpu.render_with_motion(
            &mut export_executor,
            &baseline,
            1.0 / 30.0,
            true,
            CompositionMotionFrameInput {
                attachments: &[attachment],
                held_scopes: &[],
            },
        );
        export_executor
            .prepare(&gpu.device, &gpu.queue, &moved, &sources)
            .unwrap();
        let export_pixels = gpu.render_with_motion(
            &mut export_executor,
            &moved,
            1.0 / 30.0,
            true,
            CompositionMotionFrameInput {
                attachments: &[attachment],
                held_scopes: &[],
            },
        );
        assert_eq!(pixels, export_pixels);

        let middle_row = &pixels[8 * 16..9 * 16];
        let partially_covered = middle_row
            .iter()
            .filter(|pixel| pixel[3] > 0.05 && pixel[3] < 0.95)
            .count();
        assert!(
            partially_covered >= 2,
            "subframe transform resolve did not span an exposure: {middle_row:?}"
        );
        assert_eq!(executor.allocation_snapshot(), warmed);
        let metrics = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .unwrap();
        assert_eq!(metrics.memory_generation, 2);
        assert_eq!(metrics.valid_fields, 1);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn faraday_uses_an_exact_zero_donor_field_and_one_transactional_carrier() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let plan = motion_faraday_fixture(dimensions);
        let recipient = gpu.source(dimensions, [255, 0, 0, 255], "Faraday red recipient");
        let donor = gpu.source(dimensions, [0, 255, 0, 255], "Faraday green exact donor");
        let sources = [
            CompositionSourceDescriptor::new(stable_layer(2), &recipient.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(1), &donor.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let warmed = executor.allocation_snapshot();
        gpu.submit(&mut executor, &plan, 1.0 / 30.0, true);
        gpu.submit(&mut executor, &plan, 1.0 / 30.0, true);
        let pixels = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        let recipient_metrics = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(2)))
            .expect("recipient motion metrics");
        let donor_metrics = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .expect("exact donor motion metrics");
        assert_eq!(recipient_metrics.active_field_slots, 1);
        assert_eq!(recipient_metrics.persistent_carriers, 1);
        assert_eq!(
            recipient_metrics.field_origin,
            MotionFieldOrigin::LatticeFallback
        );
        assert_eq!(
            recipient_metrics.field_source_scope,
            Some(VisualScopeId::Layer(stable_layer(1)))
        );
        assert!(recipient_metrics.carrier_valid);
        assert_eq!(donor_metrics.active_field_slots, 1);
        assert_eq!(donor_metrics.persistent_carriers, 0);
        assert_eq!(donor_metrics.valid_fields, 1);
        assert_eq!(executor.allocation_snapshot(), warmed);
        for pixel in pixels {
            assert!((pixel[0] - 1.0).abs() <= 0.01, "{pixel:?}");
            assert!(pixel[1].abs() <= 0.01, "{pixel:?}");
            assert!(pixel[2].abs() <= 0.01, "{pixel:?}");
            assert!((pixel[3] - 1.0).abs() <= 0.01, "{pixel:?}");
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn codec_attachment_is_exactly_admitted_uploaded_and_rejects_stale_generation() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let plan = motion_codec_fixture(dimensions);
        let source = gpu.source(dimensions, [255, 0, 0, 255], "Codec motion red source");
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let grid = MotionGrid::for_source(dimensions, Default::default()).unwrap();
        let field = MotionField::from_samples(
            dimensions,
            grid,
            MotionFieldOrigin::CodecVectors,
            std::iter::repeat_n(
                MotionVectorSample {
                    velocity_uv_per_second: [1.0, 0.0],
                    confidence: 1.0,
                    visibility: 1.0,
                },
                usize::try_from(grid.vector_count).unwrap(),
            ),
        )
        .unwrap();
        let valid = MotionFieldAttachment {
            scope: VisualScopeId::Layer(stable_layer(1)),
            source_generation: 7,
            frame_ordinal: 9,
            product_content_sha256: [7; 32],
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            source_dimensions: dimensions,
            grid,
            field: &field,
        };
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        for freeze in [
            TemporalFreezeState::ProgramFrozen,
            TemporalFreezeState::MediaFrozen,
        ] {
            gpu.submit_with_motion_temporal(
                &mut executor,
                &plan,
                TemporalFrameInput::new(1.0 / 30.0, freeze, false, TemporalFrameEvents::default()),
                CompositionMotionFrameInput {
                    attachments: &[valid],
                    held_scopes: &[],
                },
                true,
            );
            let unprimed = executor
                .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
                .unwrap();
            assert_eq!(unprimed.valid_fields, 0);
            assert_eq!(unprimed.field_origin, MotionFieldOrigin::None);
        }
        let pixels = gpu.render_with_motion(
            &mut executor,
            &plan,
            1.0 / 30.0,
            true,
            CompositionMotionFrameInput {
                attachments: &[valid],
                held_scopes: &[],
            },
        );
        assert!(executor.motion_diagnostics().is_empty());
        let accepted = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .unwrap();
        assert_eq!(accepted.valid_fields, 1);
        assert_eq!(accepted.field_origin, MotionFieldOrigin::CodecVectors);
        assert_eq!(
            accepted.field_source_scope,
            Some(VisualScopeId::Layer(stable_layer(1)))
        );
        assert_eq!(accepted.field_source_generation, Some(7));
        assert_eq!(accepted.field_frame_ordinal, Some(9));
        assert_eq!(accepted.field_product_content_sha256, Some([7; 32]));
        assert!(pixels.iter().all(|pixel| pixel[0] >= 0.99));

        let changed_payload = MotionFieldAttachment {
            product_content_sha256: [8; 32],
            ..valid
        };
        gpu.submit_with_motion_temporal(
            &mut executor,
            &plan,
            TemporalFrameInput::new(
                1.0 / 30.0,
                TemporalFreezeState::MediaFrozen,
                false,
                TemporalFrameEvents::default(),
            ),
            CompositionMotionFrameInput {
                attachments: &[changed_payload],
                held_scopes: &[],
            },
            true,
        );
        assert_eq!(
            executor
                .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
                .unwrap()
                .field_product_content_sha256,
            Some([7; 32]),
            "Media Freeze must retain the committed codec identity"
        );
        let _ = gpu.render_with_motion(
            &mut executor,
            &plan,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput {
                attachments: &[changed_payload],
                held_scopes: &[],
            },
        );
        assert_eq!(
            executor
                .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
                .unwrap()
                .field_product_content_sha256,
            Some([7; 32]),
            "discard must not publish a staged codec product identity"
        );

        let stale = MotionFieldAttachment {
            source_generation: 8,
            ..valid
        };
        let _ = gpu.render_with_motion(
            &mut executor,
            &plan,
            1.0 / 30.0,
            true,
            CompositionMotionFrameInput {
                attachments: &[stale],
                held_scopes: &[],
            },
        );
        assert!(executor.motion_diagnostics().iter().any(|diagnostic| {
            matches!(
                diagnostic,
                MotionRuntimeDiagnostic::RejectedCodecAttachment {
                    scope: VisualScopeId::Layer(id)
                } if *id == stable_layer(1)
            )
        }));
        let retained = executor
            .motion_metrics(VisualScopeId::Layer(stable_layer(1)))
            .unwrap();
        assert_eq!(retained.field_source_generation, Some(7));
        assert_eq!(retained.field_product_content_sha256, Some([7; 32]));
    }

    fn temporal_originals_fixture(
        dimensions: [u32; 2],
        topology: TemporalTopology,
        interpolation: TemporalInterpolation,
        seed: u32,
    ) -> EvaluatedCompositionPlan {
        let temporal = TemporalParams {
            originals: TemporalOriginalsParams {
                loom: TemporalLoomParams {
                    amount: 1.0,
                    topology,
                    interpolation,
                    depth: 0.9,
                    phase: 0.17,
                    scale: 1.4,
                    angle: 31.0,
                    folds: 7,
                    quantization: 0,
                },
                atlas: CollisionAtlasParams {
                    amount: 0.72,
                    seed,
                    territories: 12,
                    collision: 0.55,
                },
                ..TemporalOriginalsParams::default()
            },
            ..TemporalParams::default()
        };
        let base = evaluated_base_with_temporal(&[1], dimensions, &temporal);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::try_from_parts(
            vec![
                crate::visual_rack::RuntimeVisualNode::authored(
                    crate::visual_rack::NodeId::new(3).unwrap(),
                    RuntimeVisualNodeKind::Transform(SpatialTransform::default()),
                ),
                crate::visual_rack::RuntimeVisualNode::authored(
                    crate::visual_rack::NodeId::LEGACY_TEMPORAL,
                    RuntimeVisualNodeKind::LegacyTemporal,
                ),
                crate::visual_rack::RuntimeVisualNode::authored(
                    crate::visual_rack::NodeId::new(4).unwrap(),
                    RuntimeVisualNodeKind::DigitalColor(DigitalColorParams::default()),
                ),
            ],
            Some(5),
        )
        .unwrap();
        let racks = vec![(
            stable_layer(1),
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
        )];
        let plan = EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            ),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("authored temporal fixture must use Advanced")
        };
        assert!(matches!(
            advanced.master().execution.steps(),
            [
                EvaluatedScopeStep::MaterializeSpatial { .. },
                EvaluatedScopeStep::CollisionRack { .. },
                EvaluatedScopeStep::LegacyTemporal { .. },
                EvaluatedScopeStep::CollisionRack { .. },
            ]
        ));
        plan
    }

    fn group_output_reference_fixture(
        dimensions: [u32; 2],
        missing_group: bool,
    ) -> EvaluatedCompositionPlan {
        let base = if missing_group {
            evaluated_base(&[2], dimensions)
        } else {
            evaluated_base(&[2, 1, 3], dimensions)
        };
        let group_id = GroupId::new(12).unwrap();
        let mut group_rack = RuntimeVisualRack::empty();
        group_rack
            .push(RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
                invert: 1.0,
                ..Default::default()
            }))
            .unwrap();
        let group = RuntimeGroup {
            id: group_id,
            name: GroupName::new("Referenced group").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![stable_layer(1)]).unwrap(),
            // The reference must be captured before this admission opacity.
            // A red-channel hard threshold below distinguishes 1.0 from 0.25.
            opacity: 0.25,
            transform: SpatialTransform::default(),
            rack: group_rack,
            matte: Some(RuntimeImageMatte {
                tap: ResolvedImageTap {
                    source: crate::visual_rack::ResolvedImageSource::SelectedLayer {
                        layer_id: stable_layer(3),
                        saved_position: SavedLayerPosition::new(3).unwrap(),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                channel: MatteChannel::Alpha,
                invert: false,
                amount: 1.0,
                threshold: 0.5,
                softness: 0.0,
            }),
            solo: false,
            bypass: false,
            // A is fully absent at crossfade 1.0, so only the later Program
            // consumer can make the referenced group visible in the result.
            bus: BusAssignment::A,
        };
        let composition = if missing_group {
            RuntimeComposition::try_from_parts(
                Vec::new(),
                vec![RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                }],
                None,
                1.0,
            )
            .unwrap()
        } else {
            RuntimeComposition::try_from_parts(
                vec![group],
                vec![
                    RuntimeRootItem::Layer {
                        layer_id: stable_layer(3),
                        bus: BusAssignment::A,
                    },
                    RuntimeRootItem::Group { group_id },
                    RuntimeRootItem::Layer {
                        layer_id: stable_layer(2),
                        bus: BusAssignment::Program,
                    },
                ],
                None,
                1.0,
            )
            .unwrap()
        };
        let mut consumer = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        consumer
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte {
                    tap: ResolvedImageTap {
                        source: if missing_group {
                            crate::visual_rack::ResolvedImageSource::MissingGroupOutput(group_id)
                        } else {
                            crate::visual_rack::ResolvedImageSource::GroupOutput(group_id)
                        },
                        timing: EdgeTiming::CurrentFrame,
                    },
                    channel: MatteChannel::Red,
                    invert: false,
                    amount: 1.0,
                    threshold: 0.5,
                    softness: 0.0,
                },
            )))
            .unwrap();
        let racks = if missing_group {
            vec![(stable_layer(2), consumer)]
        } else {
            vec![
                (stable_layer(2), consumer),
                (
                    stable_layer(1),
                    RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
                ),
                (
                    stable_layer(3),
                    RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
                ),
            ]
        };
        let plan = EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master),
                &racks,
            ),
        )
        .unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("group-output reference fixture must be advanced")
        };
        if !missing_group {
            let producer = advanced
                .execution_order()
                .iter()
                .position(|scope| *scope == VisualScopeId::Group(group_id))
                .unwrap();
            let consumer = advanced
                .execution_order()
                .iter()
                .position(|scope| *scope == VisualScopeId::Layer(stable_layer(2)))
                .unwrap();
            assert!(
                producer < consumer,
                "group producer must precede its consumer"
            );
        }
        let consumer_tap = advanced
            .image_taps()
            .iter()
            .find(|tap| {
                matches!(
                    tap.consumer,
                    ImageTapConsumer::RackNode {
                        scope: VisualScopeId::Layer(layer_id),
                        ..
                    } if layer_id == stable_layer(2)
                )
            })
            .expect("consumer image-mask tap");
        if missing_group {
            assert_eq!(consumer_tap.resolved, PlannedImageSource::Transparent);
        } else {
            assert_eq!(
                advanced
                    .groups()
                    .iter()
                    .find(|group| group.id == group_id)
                    .unwrap()
                    .output_stage,
                crate::composition::GroupOutputStage::PostProcessingPreAdmission
            );
            assert_eq!(
                consumer_tap.resolved,
                PlannedImageSource::Scope(VisualScopeId::Group(group_id))
            );
        }
        plan
    }

    /// One layer rack whose only authored node is a Symmetry Field, so the
    /// plan must lift it out of segmentation into a dedicated step.
    ///
    /// The device ceiling is raised explicitly: every planner fixture runs on
    /// `CreativeResourceLimits::default()`, whose
    /// `max_sampled_textures_per_shader_stage` is the fixed rack layout's three,
    /// while production reads the real device. Without this the dedicated pass
    /// is correctly refused before a GPU is ever reached.
    #[derive(Clone, Copy, PartialEq)]
    enum StudyFixtureMode {
        Resolved,
        Unresolved,
        NoNode,
    }

    fn study_layer_fixture(
        dimensions: [u32; 2],
        mode: StudyFixtureMode,
    ) -> EvaluatedCompositionPlan {
        use crate::study::{StudyCapability, StudyInstruction};
        use crate::study_eval::tests::{document, register};

        let mut library = crate::study_eval::StudyProgramLibrary::default();
        let digest = library
            .insert(document(
                vec![StudyCapability::CurrentColor],
                vec![
                    StudyInstruction::LoadCurrentColor { dst: register(0) },
                    StudyInstruction::ConstantScalar {
                        dst: register(1),
                        value: 1.0 / 3.0,
                    },
                    StudyInstruction::HueRotate {
                        dst: register(2),
                        color: register(0),
                        turns: register(1),
                    },
                    StudyInstruction::OutputColor { color: register(2) },
                ],
            ))
            .unwrap();

        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::A,
            }],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::empty();
        if mode != StudyFixtureMode::NoNode {
            layer_rack
                .push(RuntimeVisualNodeKind::Study(
                    crate::visual_rack::StudyRackParams {
                        document_digest: Some(digest),
                    },
                ))
                .unwrap();
        }
        let racks = vec![(stable_layer(1), layer_rack)];
        let mut input = crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
            &composition,
            &master,
            &racks,
        );
        if mode == StudyFixtureMode::Resolved {
            input = input.with_studies(&library);
        }
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        let plan = EvaluatedCompositionPlan::evaluate(&base, input).unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("a Study node forces an Advanced plan");
        };
        assert_eq!(
            study_field_step_count(advanced),
            usize::from(mode != StudyFixtureMode::NoNode),
        );
        plan
    }

    /// The pixels claim for the authored surface: a resolved Study observably
    /// transforms the audience image through the full composition path, an
    /// unresolved digest is byte-identical to no node at all (inert, never a
    /// fallback), and warm frames allocate nothing.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_study_field_reaches_the_pixels_and_unresolved_digests_are_inert() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let source = gpu.source(dimensions, [0, 200, 255, 255], "Study layer source");
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let render_mode = |mode: StudyFixtureMode| {
            let plan = study_layer_fixture(dimensions, mode);
            let mut executor =
                CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
            executor
                .prepare(&gpu.device, &gpu.queue, &plan, &sources)
                .unwrap();
            let warmed = executor.allocation_snapshot();
            let first = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
            assert_eq!(
                executor.allocation_snapshot(),
                warmed,
                "a warmed Study frame must allocate nothing"
            );
            executor.reset_history();
            let second = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
            assert_eq!(first, second, "the Study pass is deterministic");
            first
        };
        let active = render_mode(StudyFixtureMode::Resolved);
        let unresolved = render_mode(StudyFixtureMode::Unresolved);
        let control = render_mode(StudyFixtureMode::NoNode);
        assert_ne!(
            active, control,
            "a resolved document must reach the audience pixels"
        );
        assert_eq!(
            unresolved, control,
            "an unresolved digest is inert — never a fallback onto another document"
        );
        assert!(active.iter().flatten().all(|value| value.is_finite()));
    }

    #[derive(Clone, Copy, PartialEq)]
    enum ScanFixtureMode {
        /// Raster collapse at zero velocity mix: the density payoff, with the
        /// beam-energy law deliberately disengaged so nothing but additive
        /// line overlap can raise a pixel above the source maximum.
        Collapsed,
        /// The authored default: an exact bypass.
        Bypass,
        NoNode,
    }

    fn scan_layer_fixture(dimensions: [u32; 2], mode: ScanFixtureMode) -> EvaluatedCompositionPlan {
        use crate::scan_processor::ScanProcessorParams;

        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::A,
            }],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::empty();
        match mode {
            ScanFixtureMode::Collapsed => {
                layer_rack
                    .push(RuntimeVisualNodeKind::ScanProcessor(ScanProcessorParams {
                        collapse: 0.85,
                        velocity_mix: 0.0,
                        lines: 32,
                        samples_per_line: 64,
                        ..ScanProcessorParams::default()
                    }))
                    .unwrap();
            }
            ScanFixtureMode::Bypass => {
                layer_rack
                    .push(RuntimeVisualNodeKind::ScanProcessor(
                        ScanProcessorParams::default(),
                    ))
                    .unwrap();
            }
            ScanFixtureMode::NoNode => {}
        }
        let racks = vec![(stable_layer(1), layer_rack)];
        let mut input = crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
            &composition,
            &master,
            &racks,
        );
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        let plan = EvaluatedCompositionPlan::evaluate(&base, input).unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("a Scan Processor node forces an Advanced plan");
        };
        assert_eq!(
            scan_processor_field_step_count(advanced),
            usize::from(mode != ScanFixtureMode::NoNode),
        );
        plan
    }

    /// The pixels claim for the drawn raster, and the fixture that
    /// distinguishes the mechanism from the imitation: with the beam-energy
    /// law disengaged (`velocity_mix = 0`), a collapsed raster's additive
    /// line overlap drives pixels far above the flat source's maximum
    /// luminance — which no single-sample displacement of the same image can
    /// ever do. The authored default is byte-identical to no node at all,
    /// and warm frames allocate nothing.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_scan_processor_density_exceeds_any_displacement_and_default_is_bypass() {
        let gpu = GpuHarness::new();
        let dimensions = [64, 64];
        // Mid-grey: sRGB 120 decodes to ~0.188 linear, so a doubled pixel is
        // unmistakable and still far below the clamp.
        let source = gpu.source(dimensions, [120, 120, 120, 255], "Scan flat grey source");
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let render_mode = |mode: ScanFixtureMode| {
            let plan = scan_layer_fixture(dimensions, mode);
            let mut executor =
                CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
            executor
                .prepare(&gpu.device, &gpu.queue, &plan, &sources)
                .unwrap();
            let warmed = executor.allocation_snapshot();
            let first = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
            assert_eq!(
                executor.allocation_snapshot(),
                warmed,
                "a warmed Scan Processor frame must allocate nothing"
            );
            executor.reset_history();
            let second = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
            assert_eq!(first, second, "the drawn raster is deterministic");
            first
        };
        let collapsed = render_mode(ScanFixtureMode::Collapsed);
        let bypassed = render_mode(ScanFixtureMode::Bypass);
        let control = render_mode(ScanFixtureMode::NoNode);

        assert_eq!(
            bypassed, control,
            "the authored default is an exact bypass, byte-identical to no node"
        );
        assert_ne!(collapsed, control, "a collapsed raster reaches the pixels");

        let max_channel = |image: &[[f32; 4]]| {
            image
                .iter()
                .flat_map(|pixel| pixel[..3].iter().copied())
                .fold(0.0_f32, f32::max)
        };
        let source_max = max_channel(&control);
        let ridge = max_channel(&collapsed);
        assert!(
            source_max < 0.25,
            "the flat grey control must stay near 0.19 linear, got {source_max}"
        );
        assert!(
            ridge > source_max * 2.0,
            "additive line density must exceed any single-sample bound: \
             ridge {ridge} vs source max {source_max}"
        );
        assert!(collapsed.iter().flatten().all(|value| value.is_finite()));
    }

    fn symmetry_field_layer_fixture(
        dimensions: [u32; 2],
        dedicated: bool,
    ) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::A,
            }],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::empty();
        if dedicated {
            layer_rack
                .push(RuntimeVisualNodeKind::Symmetry(
                    crate::symmetry::RuntimeSymmetryParams {
                        mode: crate::symmetry::SymmetryMode::Dihedral,
                        base_folds: 4.0,
                        radial_phase_deg: 13.0,
                        center: [0.4137, 0.5279],
                        boundary: crate::symmetry::SymmetryBoundary::Mirror,
                        ..Default::default()
                    },
                ))
                .unwrap();
        } else {
            // The control: one ordinary rack node in the same authored place.
            layer_rack
                .push(RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
                    invert: 1.0,
                    ..Default::default()
                }))
                .unwrap();
        }
        let racks = vec![(stable_layer(1), layer_rack)];
        let mut input = crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
            &composition,
            &master,
            &racks,
        );
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        let plan = EvaluatedCompositionPlan::evaluate(&base, input).unwrap();
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("a dedicated step forces an Advanced plan");
        };
        assert_eq!(
            symmetry_field_step_count(advanced),
            usize::from(dedicated),
            "the planner must emit exactly one dedicated step, and none for the control"
        );
        plan
    }

    /// Three layers; the topmost carries one Symmetry Field whose two image
    /// slots each select a DIFFERENT layer at the current frame's PreLocal
    /// stage.
    ///
    /// Both routes are armed in the source mask, so both are really admitted as
    /// taps rather than silently dropped. This is the schedule the renderer
    /// cannot honour: a current-frame PreLocal donor is materialized into a
    /// dedicated surface just before the node runs, and two of them on one node
    /// would need two such materializations interleaved with one pass.
    fn two_prelocal_symmetry_donors_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[3, 2, 1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::A,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::A,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(3),
                    bus: BusAssignment::A,
                },
            ],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let prelocal = |id: u64, position: u32| ResolvedImageTap {
            source: crate::visual_rack::ResolvedImageSource::SelectedLayer {
                layer_id: stable_layer(id),
                saved_position: SavedLayerPosition::new(position).unwrap(),
                stage: LayerImageStage::PreLocalEffects,
            },
            timing: EdgeTiming::CurrentFrame,
        };
        let mut layer_rack = RuntimeVisualRack::empty();
        layer_rack
            .push(RuntimeVisualNodeKind::Symmetry(
                crate::symmetry::RuntimeSymmetryParams {
                    mode: crate::symmetry::SymmetryMode::Dihedral,
                    base_folds: 5.0,
                    radial_phase_deg: 21.0,
                    center: [0.4137, 0.5279],
                    boundary: crate::symmetry::SymmetryBoundary::Mirror,
                    source_mask: crate::symmetry::SymmetrySourceMask {
                        carrier: true,
                        donor0: true,
                        donor1: true,
                        clean_history: false,
                    },
                    donors: [prelocal(1, 3), prelocal(2, 2)],
                    ..Default::default()
                },
            ))
            .unwrap();
        // Front to back, exactly as the evaluated base lists them.
        let racks = vec![
            (stable_layer(3), layer_rack),
            (
                stable_layer(2),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let mut input = crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
            &composition,
            &master,
            &racks,
        );
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        EvaluatedCompositionPlan::evaluate(&base, input).unwrap()
    }

    /// Two layers; the upper carries a Symmetry Field whose motion slot 0
    /// selects the lower layer, whose OWN Motion is exactly zero.
    ///
    /// The donor's primitive vector/gate field therefore exists only because
    /// the planner set `required_as_donor`. That is the exact configuration the
    /// live hand-off has to serve: an authored route, an admitted field, and a
    /// donor with nothing visible of its own.
    fn symmetry_motion_donor_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[2, 1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::empty();
        layer_rack
            .push(RuntimeVisualNodeKind::Symmetry(
                crate::symmetry::RuntimeSymmetryParams {
                    mode: crate::symmetry::SymmetryMode::Dihedral,
                    base_folds: 6.0,
                    radial_phase_deg: 17.0,
                    center: [0.4137, 0.5279],
                    boundary: crate::symmetry::SymmetryBoundary::Mirror,
                    motion_gain: 0.75,
                    motion_mask: crate::symmetry::SymmetryMotionMask {
                        slot0: true,
                        slot1: false,
                    },
                    motion: [
                        MotionDonor::Selected {
                            layer_id: stable_layer(1),
                            saved_position: SavedLayerPosition::new(2).unwrap(),
                        },
                        MotionDonor::None,
                    ],
                    ..Default::default()
                },
            ))
            .unwrap();
        let racks = vec![
            (stable_layer(2), layer_rack),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let motion = [
            LayerMotionPlanInput {
                stable_id: stable_layer(2),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(1),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        let mut input = crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
            &composition,
            &master,
            &racks,
        )
        .with_motion(
            MotionParams::default(),
            &motion,
            MotionDeviceLimits::new(8_192, u64::MAX),
        );
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        EvaluatedCompositionPlan::evaluate(&base, input).unwrap()
    }

    /// The planner really admits the donor field, without an adapter.
    #[test]
    fn a_symmetry_motion_slot_resolves_to_an_admitted_donor_field_slot() {
        let plan = symmetry_motion_donor_fixture([8, 8]);
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("a dedicated step forces an Advanced plan");
        };
        let field = advanced
            .layers()
            .iter()
            .flat_map(|layer| layer.execution.steps())
            .find_map(|step| match step {
                EvaluatedScopeStep::SymmetryField { plan } => Some(plan),
                _ => None,
            })
            .expect("the dedicated step");
        assert!(
            field.motion_field_slots[0].is_some(),
            "an armed motion slot naming a live donor must resolve to an admitted field"
        );
        assert_eq!(field.motion_field_slots[1], None);
        let motion = advanced
            .motion()
            .advanced()
            .expect("an advanced motion plan");
        let donor = motion
            .scope(VisualScopeId::Layer(stable_layer(1)))
            .expect("the donor scope");
        assert_eq!(donor.admitted_field_slot(), field.motion_field_slots[0]);
        assert!(
            donor.params.is_exact_zero(),
            "the donor's own Motion must be exactly zero, so only required_as_donor \
             can have pulled its primitive field into the plan"
        );
    }

    /// The live hand-off, end to end on a real device: an authored motion route
    /// reaches a prepared motion bind group, the encode selects the donor's
    /// committed parity, and a warmed frame still allocates nothing.
    ///
    /// Offline export drives this same `CompositionGpuExecutor` over this same
    /// evaluated plan and the same `symmetry_field.wgsl`, so there is no
    /// export-only motion path to prove separately.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_symmetry_field_binds_its_authored_motion_donor_field() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let plan = symmetry_motion_donor_fixture(dimensions);
        let donor = gpu.source(dimensions, [255, 0, 0, 255], "Symmetry motion donor");
        let carrier = gpu.source(dimensions, [0, 200, 255, 255], "Symmetry motion carrier");
        let sources = [
            CompositionSourceDescriptor::new(stable_layer(1), &donor.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(2), &carrier.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .expect("an authored motion route must prepare");
        let warmed = executor.allocation_snapshot();
        let first = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert_eq!(
            executor.allocation_snapshot(),
            warmed,
            "selecting a committed motion parity must not allocate"
        );
        let second = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert_eq!(executor.allocation_snapshot(), warmed);
        assert!(first.iter().flatten().all(|value| value.is_finite()));
        assert!(second.iter().flatten().all(|value| value.is_finite()));
    }

    /// One authored composition, expressed under whichever layer-identity
    /// scheme the caller names and with the motion route either armed or not.
    ///
    /// The donor's OWN Motion is exactly zero — `MotionParams::is_exact_zero()`
    /// reads only `transplant.amount` and `shutter.angle_degrees`, and both stay
    /// at their defaults — while it still publishes a codec field. So the
    /// primitive vector/gate pair exists for exactly one reason: an armed motion
    /// slot on the layer above set `required_as_donor`. Unarm the slot and the
    /// field is never admitted at all, which is the honest control.
    ///
    /// Everything else is held identical between the two arms: carrier-only
    /// source mask, zero hue span, same geometry, same seed. The armed arm
    /// therefore differs from the unarmed one only in each sector record's
    /// motion lane, and with a stationary field even that difference is
    /// invisible — `raw + vec2f(0.0)` is bit-identical to `raw`.
    /// A two-layer composition whose top layer's Faraday carrier is advected
    /// from a Field Collider.
    ///
    /// Input A names the recipient itself and input B names the layer beneath
    /// it. That is deliberate and legal: section 5 allows either input to equal
    /// the recipient and forbids only A aliasing B, so this fixture exercises
    /// the permissive half of that law while keeping the composition topology
    /// identical to the proven Symmetry payoff fixture beside it.
    fn field_collider_payoff_fixture(
        dimensions: [u32; 2],
        recipient_id: u64,
        donor_id: u64,
        collider: FieldColliderParams,
    ) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[recipient_id, donor_id], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(donor_id),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(recipient_id),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = vec![
            (
                stable_layer(recipient_id),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
            (
                stable_layer(donor_id),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        // The recipient owns the shared carrier/advection controls. Both
        // collider inputs take codec fields so a known uniform observation can
        // be attached to each; neither input has any authored Motion *effect*
        // of its own, so only `required_as_donor` can pull its field in.
        let codec = MotionCodecFrameFacts {
            available: true,
            source_generation: 7,
            frame_ordinal: 9,
        };
        let motion = [
            LayerMotionPlanInput {
                stable_id: stable_layer(recipient_id),
                params: MotionParams {
                    field_source: MotionFieldSource::CodecVectors,
                    transplant: FaradayParams {
                        amount: 1.0,
                        carrier: MotionCarrier::FirstSourceFrame,
                        confidence_threshold: 0.0,
                        confidence_softness: 0.0,
                        refresh: 0.0,
                        decay: 1.0,
                        occlusion: 0.0,
                        ..Default::default()
                    },
                    collider,
                    ..MotionParams::default()
                },
                codec,
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(donor_id),
                params: MotionParams {
                    field_source: MotionFieldSource::CodecVectors,
                    ..MotionParams::default()
                },
                codec,
            },
        ];
        EvaluatedCompositionPlan::evaluate(
            &base,
            CompositionPlanInput::new(&composition, &master, &racks).with_motion(
                MotionParams::default(),
                &motion,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
        .unwrap()
    }

    /// THE delivery gate.
    ///
    /// A pure type, an isolated shader, or a hidden test is not delivery. This
    /// fixture drives a known A/B pair through both low-resolution collider
    /// passes and proves, in one place, that:
    ///
    /// - the derived collided field actually advects the carrier and reaches
    ///   the audience image — the frame MOVES, and moves differently from what
    ///   either input alone would have produced;
    /// - a disabled collider is BYTE-IDENTICAL to the exact M4 path, which is
    ///   the compatibility claim;
    /// - the live process-lifetime layer identities and export's
    ///   `position + 1` identities render the same authored patch identically,
    ///   so there is no export-only collider path;
    /// - a warm frame allocates nothing across the eight prebuilt parity bind
    ///   groups.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_field_collider_derived_field_reaches_the_pixels() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let grid = MotionGrid::for_source(dimensions, Default::default()).unwrap();
        // 6 UV/s through one reference tick is 0.2 UV — several pixels here,
        // far outside binary16 noise and well inside the clamp.
        let a_field = uniform_motion_field(dimensions, grid, [6.0, -6.0]);
        let b_field = uniform_motion_field(dimensions, grid, [-6.0, 6.0]);
        let stationary = uniform_motion_field(dimensions, grid, [0.0, 0.0]);

        // The recipient is the upper layer, so it carries the higher live
        // identity exactly as the Symmetry payoff fixture beside it does.
        const LIVE_RECIPIENT: u64 = 812;
        const LIVE_DONOR: u64 = 811;

        let armed = |recipient: u64, donor: u64| FieldColliderParams {
            enabled: true,
            mode: FieldColliderMode::Difference,
            boundary: MotionBoundaryMode::Hold,
            input_a: MotionDonor::Selected {
                layer_id: stable_layer(recipient),
                saved_position: SavedLayerPosition::new(1).unwrap(),
            },
            input_b: MotionDonor::Selected {
                layer_id: stable_layer(donor),
                saved_position: SavedLayerPosition::new(2).unwrap(),
            },
            ..Default::default()
        };

        let live = field_collider_payoff_fixture(
            dimensions,
            LIVE_RECIPIENT,
            LIVE_DONOR,
            armed(LIVE_RECIPIENT, LIVE_DONOR),
        );
        // Export numbers the same authored layers `position + 1`.
        let offline = field_collider_payoff_fixture(dimensions, 2, 1, armed(2, 1));
        // The identical patch with the block disabled: the parked single-donor
        // recipe, retained verbatim and simply not running.
        let mut parked = armed(LIVE_RECIPIENT, LIVE_DONOR);
        parked.enabled = false;
        let disabled =
            field_collider_payoff_fixture(dimensions, LIVE_RECIPIENT, LIVE_DONOR, parked);

        // The collider really is admitted, and both inputs really were pulled
        // in as honest primitive fields by `required_as_donor` alone.
        let EvaluatedCompositionPlan::Advanced(advanced) = &live else {
            panic!("an admitted collider forces an Advanced plan");
        };
        let motion = advanced
            .motion()
            .advanced()
            .expect("an advanced motion plan");
        let plan = motion.collider().expect("an admitted collider plan");
        assert_ne!(plan.input_a_slot, plan.input_b_slot);
        for scope in [plan.input_a_scope, plan.input_b_scope] {
            let input = motion.scope(scope).expect("an admitted collider input");
            assert!(input.required_as_donor);
        }
        let recipient_scope = motion.scope(plan.recipient_scope).expect("the recipient");
        assert!(recipient_scope.collider_admitted);
        assert!(recipient_scope.transplant_admitted);
        // Twenty bytes a cell, two low-resolution passes, three sampled
        // textures — the published ledger, on the plan that actually renders.
        let resources = motion.collider_resources();
        assert_eq!(resources.total_bytes, plan.output_grid.vector_count * 20);
        assert_eq!(resources.low_resolution_passes, 2);
        assert_eq!(resources.max_sampled_textures_in_pass, 3);

        let carrier_source = checkered_source(&gpu, dimensions, "Collider payoff carrier");
        let donor_source = gpu.source(dimensions, [255, 0, 0, 255], "Collider payoff donor");
        let live_sources = [
            CompositionSourceDescriptor::new(
                stable_layer(LIVE_DONOR),
                &donor_source.view,
                dimensions,
            ),
            CompositionSourceDescriptor::new(
                stable_layer(LIVE_RECIPIENT),
                &carrier_source.view,
                dimensions,
            ),
        ];
        let offline_sources = [
            CompositionSourceDescriptor::new(stable_layer(1), &donor_source.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(2), &carrier_source.view, dimensions),
        ];
        let attachment = |scope, field| payoff_attachment(scope, dimensions, grid, field);
        let a_scope = plan.input_a_scope;
        let b_scope = plan.input_b_scope;

        let render = |plan: &EvaluatedCompositionPlan,
                      sources: &[CompositionSourceDescriptor],
                      attachments: &[_]| {
            let mut executor =
                CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
            executor
                .prepare(&gpu.device, &gpu.queue, plan, sources)
                .expect("an admitted collider must prepare");
            let warmed = executor.allocation_snapshot();
            let pixels = gpu.render_with_motion(
                &mut executor,
                plan,
                1.0 / 30.0,
                false,
                CompositionMotionFrameInput {
                    attachments,
                    held_scopes: &[],
                },
            );
            // A warm frame binds one of the eight prebuilt parity groups per
            // pass and creates nothing.
            assert_eq!(
                executor.allocation_snapshot(),
                warmed,
                "a warm collider frame allocated"
            );
            pixels
        };

        // (1) The collided field MOVES the frame.
        let collided = render(
            &live,
            &live_sources,
            &[attachment(a_scope, &a_field), attachment(b_scope, &b_field)],
        );
        let both_still = render(
            &live,
            &live_sources,
            &[
                attachment(a_scope, &stationary),
                attachment(b_scope, &stationary),
            ],
        );
        assert_ne!(
            collided, both_still,
            "the derived collided field must reach the audience image"
        );

        // (2) It is genuinely the COLLISION, not either input alone. The
        // difference of (6,-6) and (-6,6) is (12,-12); of (6,-6) and (0,0) it
        // is (6,-6). Both inputs demonstrably contribute.
        let a_only = render(
            &live,
            &live_sources,
            &[
                attachment(a_scope, &a_field),
                attachment(b_scope, &stationary),
            ],
        );
        assert_ne!(
            collided, a_only,
            "input B must contribute to the derived field"
        );
        let b_only = render(
            &live,
            &live_sources,
            &[
                attachment(a_scope, &stationary),
                attachment(b_scope, &b_field),
            ],
        );
        assert_ne!(
            collided, b_only,
            "input A must contribute to the derived field"
        );

        // (3) An unmaterialized input yields the exact invalid/zero sample and
        // never reuses its surviving partner.
        let one_missing = render(&live, &live_sources, &[attachment(a_scope, &a_field)]);
        assert_eq!(
            one_missing, both_still,
            "a missing input must yield the exact zero sample, not its partner's field"
        );

        // (4) Live and export identities render the same authored patch
        // identically: there is no export-only collider path.
        let exported = render(
            &offline,
            &offline_sources,
            &[
                attachment(VisualScopeId::Layer(stable_layer(2)), &a_field),
                attachment(VisualScopeId::Layer(stable_layer(1)), &b_field),
            ],
        );
        assert_eq!(
            collided, exported,
            "live and export layer identities must render the same authored collider"
        );

        // (5) Disabled is BYTE-IDENTICAL to exact M4: the parked single-donor
        // recipe names no transplant donor, so offering the collider's fields
        // changes nothing at all.
        let parked_pixels = render(
            &disabled,
            &live_sources,
            &[attachment(a_scope, &a_field), attachment(b_scope, &b_field)],
        );
        let parked_control = render(&disabled, &live_sources, &[]);
        assert_eq!(
            parked_pixels, parked_control,
            "a disabled collider must be byte-identical to exact M4"
        );
        assert_ne!(
            collided, parked_pixels,
            "enabling the collider must change the image it produces"
        );
    }

    fn symmetry_motion_payoff_fixture(
        dimensions: [u32; 2],
        carrier_id: u64,
        donor_id: u64,
        armed: bool,
    ) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[carrier_id, donor_id], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(donor_id),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(carrier_id),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.0,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut layer_rack = RuntimeVisualRack::empty();
        layer_rack
            .push(RuntimeVisualNodeKind::Symmetry(
                crate::symmetry::RuntimeSymmetryParams {
                    mode: crate::symmetry::SymmetryMode::Dihedral,
                    // Twelve reachable sector records. With one motion slot
                    // armed the motion lane is a two-way draw, so the odds that
                    // no reachable record names the slot are 1 in 4,096 — and
                    // the fixture asserts it outright rather than trusting them.
                    base_folds: 12.0,
                    radial_phase_deg: 17.0,
                    center: [0.4137, 0.5279],
                    boundary: crate::symmetry::SymmetryBoundary::Mirror,
                    motion_gain: 1.0,
                    motion_mask: crate::symmetry::SymmetryMotionMask {
                        slot0: armed,
                        slot1: false,
                    },
                    motion: [
                        if armed {
                            MotionDonor::Selected {
                                layer_id: stable_layer(donor_id),
                                saved_position: SavedLayerPosition::new(2).unwrap(),
                            }
                        } else {
                            MotionDonor::None
                        },
                        MotionDonor::None,
                    ],
                    ..Default::default()
                },
            ))
            .unwrap();
        let racks = vec![
            (stable_layer(carrier_id), layer_rack),
            (
                stable_layer(donor_id),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        let motion = [
            LayerMotionPlanInput {
                stable_id: stable_layer(carrier_id),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: stable_layer(donor_id),
                params: MotionParams {
                    field_source: MotionFieldSource::CodecVectors,
                    ..MotionParams::default()
                },
                codec: MotionCodecFrameFacts {
                    available: true,
                    source_generation: 7,
                    frame_ordinal: 9,
                },
            },
        ];
        let mut input = crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
            &composition,
            &master,
            &racks,
        )
        .with_motion(
            MotionParams::default(),
            &motion,
            MotionDeviceLimits::new(8_192, u64::MAX),
        );
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        EvaluatedCompositionPlan::evaluate(&base, input).unwrap()
    }

    /// A spatially uniform codec field, so a moved pixel can only be the
    /// authored route and never an accident of one grid cell.
    fn uniform_motion_field(
        dimensions: [u32; 2],
        grid: MotionGrid,
        velocity_uv_per_second: [f32; 2],
    ) -> MotionField {
        MotionField::from_samples(
            dimensions,
            grid,
            MotionFieldOrigin::CodecVectors,
            std::iter::repeat_n(
                MotionVectorSample {
                    velocity_uv_per_second,
                    confidence: 1.0,
                    visibility: 1.0,
                },
                grid.vector_count as usize,
            ),
        )
        .expect("a uniform codec field")
    }

    /// The fixture's codec attachment. The generation and ordinal match the
    /// donor's `MotionCodecFrameFacts` exactly, because
    /// `EvaluatedMotionFieldPlan::accepts` compares every one of them.
    fn payoff_attachment<'a>(
        scope: VisualScopeId,
        dimensions: [u32; 2],
        grid: MotionGrid,
        field: &'a MotionField,
    ) -> MotionFieldAttachment<'a> {
        MotionFieldAttachment {
            scope,
            source_generation: 7,
            frame_ordinal: 9,
            product_content_sha256: [11; 32],
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            source_dimensions: dimensions,
            grid,
            field,
        }
    }

    /// A carrier a displacement cannot land on unnoticed: a two-pixel checker
    /// under a per-column ramp, so no translation of a few pixels reproduces
    /// the original frame.
    fn checkered_source(gpu: &GpuHarness, dimensions: [u32; 2], label: &'static str) -> TestSource {
        let source = gpu.source(dimensions, [0, 0, 0, 255], label);
        let mut bytes = vec![0_u8; dimensions[0] as usize * dimensions[1] as usize * 4];
        for row in 0..dimensions[1] as usize {
            for column in 0..dimensions[0] as usize {
                let offset = (row * dimensions[0] as usize + column) * 4;
                let ramp = (column * 255 / dimensions[0].max(1) as usize) as u8;
                let texel = if (column / 2 + row / 2) % 2 == 0 {
                    [255, ramp, 16, 255]
                } else {
                    [16, ramp, 255, 255]
                };
                bytes[offset..offset + 4].copy_from_slice(&texel);
            }
        }
        gpu.queue.write_texture(
            source.texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dimensions[0] * 4),
                rows_per_image: Some(dimensions[1]),
            },
            wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
        );
        source
    }

    /// **The payoff.** An authored Symmetry Field motion route reaches the
    /// pixels — which is precisely what binding `motion: [None, None]` broke,
    /// and what the three-bind-group split exists to fix.
    ///
    /// The donor's own visible Motion is exactly zero, so its primitive
    /// vector/gate pair is in the plan only because `required_as_donor` put it
    /// there. A known uniform field is driven through it and four things are
    /// proven at once:
    ///
    /// - **(a)** the frame differs both from the same node fed a stationary
    ///   field and from the same node with the slot unarmed;
    /// - **(b)** the live layer-identity scheme (process-lifetime ids) and
    ///   export's `position + 1` scheme render byte-identical frames from the
    ///   same authored patch at the same frame index;
    /// - **(c)** an unarmed slot, and an armed slot whose field never
    ///   materialized, are byte-identical to the stationary neutral pair;
    /// - **(d)** the warm-allocation snapshot still holds with eight prebuilt
    ///   bind groups per node.
    ///
    /// Every render is one frame from a pristine executor with `commit: false`,
    /// so no temporal ring or motion memory advance can leak between the
    /// comparisons.
    ///
    /// The fixture was confirmed discriminating rather than incidentally true:
    /// restoring `motion: [None, None]` in `prepare_symmetry_fields` — the exact
    /// gap this stage closed — fails it at (a) on this adapter.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_symmetry_field_authored_motion_route_reaches_the_pixels() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let grid = MotionGrid::for_source(dimensions, Default::default()).unwrap();
        // 6 UV/s through one reference tick is 0.2 UV — better than three pixels
        // here, far outside binary16 noise, and well inside the ±64 clamp.
        let moving = uniform_motion_field(dimensions, grid, [6.0, -6.0]);
        let stationary = uniform_motion_field(dimensions, grid, [0.0, 0.0]);

        // Live process-lifetime identities and export's position + 1 identities,
        // for the same authored composition.
        const LIVE_CARRIER: u64 = 908;
        const LIVE_DONOR: u64 = 907;
        let live = symmetry_motion_payoff_fixture(dimensions, LIVE_CARRIER, LIVE_DONOR, true);
        let offline = symmetry_motion_payoff_fixture(dimensions, 2, 1, true);
        let unarmed = symmetry_motion_payoff_fixture(dimensions, LIVE_CARRIER, LIVE_DONOR, false);

        // The route really is armed on a donor with nothing visible of its own,
        // and the reachable part of the sector table really names it.
        let EvaluatedCompositionPlan::Advanced(advanced) = &live else {
            panic!("a dedicated step forces an Advanced plan");
        };
        let field_plan = advanced
            .layers()
            .iter()
            .flat_map(|layer| layer.execution.steps())
            .find_map(|step| match step {
                EvaluatedScopeStep::SymmetryField { plan } => Some(plan),
                _ => None,
            })
            .expect("the dedicated step");
        assert!(
            field_plan.motion_field_slots[0].is_some(),
            "the armed slot must resolve to an admitted primitive field"
        );
        let donor_scope = VisualScopeId::Layer(stable_layer(LIVE_DONOR));
        let donor = advanced
            .motion()
            .advanced()
            .expect("an advanced motion plan")
            .scope(donor_scope)
            .expect("the donor scope");
        assert!(
            donor.params.is_exact_zero(),
            "only required_as_donor can have pulled this field into the plan"
        );
        let folds = usize::from(field_plan.params.values().effective_folds());
        let table = field_plan.params.sector_table(field_plan.domain);
        assert!(
            table.records()[..folds]
                .iter()
                .any(|record| record.motion == Some(0)),
            "no reachable sector names the armed motion slot"
        );

        let carrier_source = checkered_source(&gpu, dimensions, "Symmetry payoff carrier");
        let donor_source = gpu.source(dimensions, [255, 0, 0, 255], "Symmetry payoff donor");
        let live_sources = [
            CompositionSourceDescriptor::new(
                stable_layer(LIVE_DONOR),
                &donor_source.view,
                dimensions,
            ),
            CompositionSourceDescriptor::new(
                stable_layer(LIVE_CARRIER),
                &carrier_source.view,
                dimensions,
            ),
        ];
        let offline_sources = [
            CompositionSourceDescriptor::new(stable_layer(1), &donor_source.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(2), &carrier_source.view, dimensions),
        ];
        let attachment = |scope, field| payoff_attachment(scope, dimensions, grid, field);

        // (a) and (d): the armed route with a real field, twice, from one warm
        // executor.
        let mut armed = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        armed
            .prepare(&gpu.device, &gpu.queue, &live, &live_sources)
            .expect("an authored motion route must prepare");
        let warmed = armed.allocation_snapshot();
        let moved = gpu.render_with_motion(
            &mut armed,
            &live,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput {
                attachments: &[attachment(donor_scope, &moving)],
                held_scopes: &[],
            },
        );
        assert_eq!(
            armed.allocation_snapshot(),
            warmed,
            "selecting a committed motion parity must allocate nothing"
        );
        let repeated = gpu.render_with_motion(
            &mut armed,
            &live,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput {
                attachments: &[attachment(donor_scope, &moving)],
                held_scopes: &[],
            },
        );
        assert_eq!(armed.allocation_snapshot(), warmed);
        assert_eq!(moved, repeated, "the routed pass is deterministic");

        // The same authored node, the same admitted field, a stationary vector.
        let mut held = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        held.prepare(&gpu.device, &gpu.queue, &live, &live_sources)
            .unwrap();
        let still = gpu.render_with_motion(
            &mut held,
            &live,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput {
                attachments: &[attachment(donor_scope, &stationary)],
                held_scopes: &[],
            },
        );
        assert_ne!(
            moved, still,
            "an authored motion route must reach the image"
        );

        // (c) The armed slot whose field never materialized: no attachment at
        // all, so the committed parity holds nothing and the readiness bit
        // closes the record's validity lane.
        let mut unmaterialized =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        unmaterialized
            .prepare(&gpu.device, &gpu.queue, &live, &live_sources)
            .unwrap();
        let missing = gpu.render_with_motion(
            &mut unmaterialized,
            &live,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput::default(),
        );
        assert_eq!(
            missing, still,
            "a missing motion field must be byte-identical to the neutral pair"
        );

        // (c) The unarmed slot, offered the moving field it never asked for.
        let mut control = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        control
            .prepare(&gpu.device, &gpu.queue, &unarmed, &live_sources)
            .unwrap();
        let unarmed_pixels = gpu.render_with_motion(
            &mut control,
            &unarmed,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput {
                attachments: &[attachment(donor_scope, &moving)],
                held_scopes: &[],
            },
        );
        assert_ne!(
            moved, unarmed_pixels,
            "arming the slot must be visible against the unarmed control"
        );
        assert_eq!(
            unarmed_pixels, still,
            "an unarmed slot must be byte-identical to the neutral pair"
        );

        // (b) Export's layer-identity scheme, same authored patch, same frame.
        let mut export = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        export
            .prepare(&gpu.device, &gpu.queue, &offline, &offline_sources)
            .unwrap();
        let offline_pixels = gpu.render_with_motion(
            &mut export,
            &offline,
            1.0 / 30.0,
            false,
            CompositionMotionFrameInput {
                attachments: &[attachment(VisualScopeId::Layer(stable_layer(1)), &moving)],
                held_scopes: &[],
            },
        );
        assert_eq!(
            moved, offline_pixels,
            "live and offline must render the same authored patch identically"
        );
        assert!(moved.iter().flatten().all(|value| value.is_finite()));
    }

    /// The precondition of the refusal, proven without an adapter: the planner
    /// really admits BOTH image slots as separate current-frame PreLocal taps
    /// naming two different layers. Without this the GPU fixture below could
    /// pass for the wrong reason.
    #[test]
    fn two_symmetry_image_slots_can_plan_two_distinct_current_prelocal_donors() {
        let plan = two_prelocal_symmetry_donors_fixture([8, 8]);
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("a dedicated step forces an Advanced plan");
        };
        assert_eq!(symmetry_field_step_count(advanced), 1);
        let mut donors = Vec::new();
        for slot in 0..SYMMETRY_IMAGE_SLOTS {
            let tap = advanced
                .image_taps()
                .iter()
                .find(|tap| {
                    matches!(
                        tap.consumer,
                        ImageTapConsumer::RackNode {
                            scope: VisualScopeId::Layer(layer_id),
                            slot: tap_slot,
                            ..
                        } if layer_id == stable_layer(3) && usize::from(tap_slot) == slot
                    )
                })
                .unwrap_or_else(|| panic!("image slot {slot} must be admitted as its own tap"));
            let PlannedImageSource::SelectedLayer {
                layer_id,
                stage: LayerImageStage::PreLocalEffects,
            } = tap.resolved
            else {
                panic!("slot {slot} must resolve to a current-frame PreLocal donor");
            };
            assert_eq!(tap.origin.timing(), EdgeTiming::CurrentFrame);
            donors.push(layer_id);
        }
        assert_eq!(donors, vec![stable_layer(1), stable_layer(2)]);
    }

    /// The renderer refuses that schedule by name rather than silently binding
    /// one donor twice or dropping the second.
    ///
    /// This is the fixture the renderer stage deferred. It drives the real
    /// evaluated plan through the real `prepare`, so the refusal is proven at
    /// the seam that owns it, not by calling a predicate directly.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_symmetry_field_refuses_two_current_prelocal_donors() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let plan = two_prelocal_symmetry_donors_fixture(dimensions);
        let first = gpu.source(dimensions, [255, 0, 0, 255], "PreLocal donor one");
        let second = gpu.source(dimensions, [0, 255, 0, 255], "PreLocal donor two");
        let carrier = gpu.source(dimensions, [0, 0, 255, 255], "Symmetry Field carrier");
        let sources = [
            CompositionSourceDescriptor::new(stable_layer(1), &first.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(2), &second.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(3), &carrier.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let error = executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .expect_err("two current PreLocal donors on one node must be refused");
        let CompositionGpuError::InvalidSchedule(message) = &error else {
            panic!("the refusal must be the typed InvalidSchedule, not {error:?}");
        };
        assert!(
            message.contains("a Symmetry Field contains more than one current PreLocal donor"),
            "the refusal must name the dedicated step: {message}"
        );
    }

    /// The dedicated executor is real end to end: the composition builds it,
    /// executes the authored step in place, and a warmed frame allocates
    /// nothing. Two frames of the same plan are byte identical.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_symmetry_field_executes_in_place_and_warm_frames_allocate_nothing() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let plan = symmetry_field_layer_fixture(dimensions, true);
        let control_plan = symmetry_field_layer_fixture(dimensions, false);
        let source = gpu.source(
            dimensions,
            [0, 200, 255, 255],
            "Symmetry Field layer source",
        );
        let sources = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let mut control = CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        control
            .prepare(&gpu.device, &gpu.queue, &control_plan, &sources)
            .unwrap();
        let control_warm = control.allocation_snapshot();
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let warmed = executor.allocation_snapshot();
        assert!(warmed.total() > 0);
        // The exact frozen resource delta of one dedicated step, measured on a
        // real device against the identical composition carrying an ordinary
        // rack node instead: three tiny neutral textures, one uniform arena,
        // one uniform bind group, one pipeline, TWO image bind groups per
        // prepared N-1 parity — the composition prepares both committed read
        // parities, exactly as it does for a rack segment, and each holds one
        // group per carrier parity — plus FOUR motion bind groups, one per
        // combination of the two motion slots' committed field parities. The
        // motion groups are prepared once per node rather than per read
        // parity, which is the whole point of splitting them out: the three
        // parity dimensions add to eight groups instead of multiplying to
        // sixteen. Zero full-frame surfaces either way.
        const N_MINUS_ONE_PARITIES: u64 = 2;
        assert_eq!(
            warmed.rack_objects - control_warm.rack_objects,
            3 + 1
                + 1
                + 1
                + SYMMETRY_FIELD_CARRIER_PARITIES as u64 * N_MINUS_ONE_PARITIES
                + SYMMETRY_FIELD_MOTION_GROUPS as u64,
            "one dedicated step costs exactly its declared objects"
        );
        assert_eq!(warmed.retained_textures, control_warm.retained_textures);
        assert_eq!(warmed.retained_views, control_warm.retained_views);
        let EvaluatedCompositionPlan::Advanced(advanced) = &plan else {
            panic!("fixture must be Advanced");
        };
        // The dedicated pass owns no full-frame surface, so the admitted
        // creative byte ledger is untouched by its presence.
        assert_eq!(warmed.creative_bytes, advanced.resources().creative_bytes);

        let first = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert_eq!(
            executor.allocation_snapshot(),
            warmed,
            "a warmed frame carrying a dedicated pass must allocate nothing"
        );
        executor.reset_history();
        let second = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert_eq!(executor.allocation_snapshot(), warmed);
        assert_eq!(first, second, "the dedicated pass is deterministic");
        assert!(first.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_group_rack_matte_bus_is_fixed_at_24_30_and_60_fps() {
        let gpu = GpuHarness::new();
        let dimensions = [4, 4];
        let (plan, _composition) = group_rack_matte_bus_fixture(dimensions);
        let front = gpu.source(dimensions, [0, 255, 0, 255], "Advanced green group member");
        let back = gpu.source(dimensions, [255, 0, 0, 128], "Advanced red A donor");
        let sources = [
            CompositionSourceDescriptor::new(stable_layer(2), &front.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(1), &back.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let warmed = executor.allocation_snapshot();
        assert!(warmed.total() > 0);
        let advanced = match &plan {
            EvaluatedCompositionPlan::Advanced(plan) => plan,
            EvaluatedCompositionPlan::LegacyExact(_) => panic!("fixture must be Advanced"),
        };
        let resources = advanced.resources();
        assert_eq!(warmed.creative_bytes, resources.creative_bytes);
        assert_eq!(
            warmed.motion_bytes,
            advanced
                .motion()
                .advanced()
                .map_or(0, |motion| motion.resources().total_bytes)
        );
        let reconciled = crate::precision::RuntimeResourceLedger::reconcile(
            crate::precision::RuntimeAllocationSnapshot {
                output_size: dimensions,
                path: crate::precision::SETTLED_ADVANCED_PRECISION_PATH,
                working_layers: resources.rgba16_surface_layers,
                history_layers: resources.compat8_surface_layers,
                creative_bytes: warmed.creative_bytes,
                motion_bytes: warmed.motion_bytes,
                ntsc_bytes: 0,
                staging_bytes: warmed.readback_staging_bytes,
                readback_bytes: warmed.readback_texture_bytes,
            },
            crate::precision::PrecisionResourceLimits::default(),
        )
        .expect("physical allocation snapshot must reconcile with its admitted plan");
        assert_eq!(reconciled.creative_bytes, resources.creative_bytes);
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        assert_eq!(executor.allocation_snapshot(), warmed);

        let mut outputs = Vec::new();
        for fps in [24.0_f32, 30.0, 60.0] {
            executor.reset_history();
            let pixels = gpu.render(&mut executor, &plan, 1.0 / fps, false);
            assert_eq!(executor.allocation_snapshot(), warmed);
            outputs.push(pixels);
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[1], outputs[2]);

        let donor_alpha = 128.0 / 255.0;
        let shaped = donor_alpha * donor_alpha * (3.0 - 2.0 * donor_alpha);
        let expected = crate::renderer::composition_host::host_bus_reference(
            [1.0, 0.0, 0.0, donor_alpha],
            [1.0, 0.0, 1.0, shaped * 0.5],
            [0.0; 4],
            0.25,
        );
        for pixel in &outputs[0] {
            for channel in 0..4 {
                assert!(
                    (pixel[channel] - expected[channel]).abs() <= 0.015,
                    "channel {channel}: actual {}, expected {}",
                    pixel[channel],
                    expected[channel]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn recorder_scope_capture_is_post_effects_fifo_warm_and_never_falls_back() {
        let gpu = GpuHarness::new();
        let dimensions = [4, 4];
        let (plan, _composition) = group_rack_matte_bus_fixture(dimensions);
        let member = gpu.source(dimensions, [0, 255, 0, 255], "Scope capture green member");
        let donor = gpu.source(dimensions, [255, 0, 0, 128], "Scope capture alpha donor");
        let sources = [
            CompositionSourceDescriptor::new(stable_layer(2), &member.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(1), &donor.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &sources)
            .unwrap();
        let prepared = executor
            .prepare_scope_recorder_readback(&gpu.device, CaptureTarget::Layer(stable_layer(2)))
            .unwrap();
        assert_eq!(prepared.buffers, 2);
        assert_eq!(prepared.conversion_textures, 1);
        assert!(
            prepared.total_bytes() <= crate::renderer::readback::RECORDER_GPU_READBACK_MAX_BYTES
        );
        let warmed = executor.allocation_snapshot();

        // An outer rejected frame can revoke an armed reservation without
        // submitting it, after which the same fixed slot is reusable.
        let RecorderReadbackAdmission::Scheduled(abandoned) = executor
            .begin_scope_recorder_readback(RecorderReadbackTag::new(17, recorder_metadata(0)))
        else {
            panic!("layer scope must arm")
        };
        executor
            .discard_unsubmitted_scope_recorder_readback(abandoned)
            .unwrap();
        assert!(matches!(
            executor.map_scope_recorder_readback(abandoned),
            Err(RecorderReadbackError::InvalidReservation)
        ));

        let RecorderReadbackAdmission::Scheduled(abandoned_with_frame) = executor
            .begin_scope_recorder_readback(RecorderReadbackTag::new(17, recorder_metadata(0)))
        else {
            panic!("layer scope must re-arm")
        };
        executor.discard_frame_history();
        assert!(matches!(
            executor.map_scope_recorder_readback(abandoned_with_frame),
            Err(RecorderReadbackError::InvalidReservation)
        ));

        let capture = |executor: &mut CompositionGpuExecutor,
                       capture_index: u64|
         -> (Vec<u8>, RecorderReadbackRequest) {
            let RecorderReadbackAdmission::Scheduled(reservation) = executor
                .begin_scope_recorder_readback(RecorderReadbackTag::new(
                    17,
                    recorder_metadata(capture_index),
                ))
            else {
                panic!("prepared scope must arm")
            };
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Post-effects scope recorder test encoder"),
                });
            executor
                .encode(
                    &gpu.queue,
                    &mut encoder,
                    &plan,
                    CompositionFrameTiming::new(1.0 / 30.0, true),
                )
                .unwrap();
            assert_eq!(
                executor
                    .finish_scope_recorder_readback(reservation)
                    .unwrap(),
                RecorderReadbackCaptureStatus::Captured
            );
            gpu.queue.submit(std::iter::once(encoder.finish()));
            executor.map_scope_recorder_readback(reservation).unwrap();
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("scope recorder GPU wait");
            let mut pixels = vec![0_u8; dimensions[0] as usize * dimensions[1] as usize * 4];
            let RecorderReadbackPoll::Ready(completed) = executor
                .poll_scope_recorder_readback_into(&gpu.device, &mut pixels)
                .unwrap()
            else {
                panic!("scope recorder capture must be ready")
            };
            executor.commit_frame_history();
            (pixels, completed.request)
        };

        let (layer_pixels, layer_request) = capture(&mut executor, 1);
        assert_eq!(layer_request.target, CaptureTarget::Layer(stable_layer(2)));
        assert_eq!(layer_request.tag.metadata().capture_index, 1);
        assert_eq!(layer_request.tag.capture_generation(), 17);
        assert!(layer_pixels
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 255, 0, 255]));
        assert_eq!(executor.allocation_snapshot(), warmed);

        // Selecting the group reuses every object. Its rack inverts the green
        // member before the group matte boundary, yielding magenta.
        let group_id = GroupId::new(10).unwrap();
        let group_prepared = executor
            .prepare_scope_recorder_readback(&gpu.device, CaptureTarget::Group(group_id))
            .unwrap();
        assert_eq!(group_prepared, prepared);
        assert_eq!(executor.allocation_snapshot(), warmed);
        let (group_pixels, group_request) = capture(&mut executor, 2);
        assert_eq!(group_request.target, CaptureTarget::Group(group_id));
        assert!(group_pixels
            .chunks_exact(4)
            .all(|pixel| { pixel[0] >= 250 && pixel[1] <= 5 && pixel[2] >= 250 && pixel[3] > 0 }));

        // A topology replacement which deletes the stable group is explicit
        // unavailable; it can never retarget the final Program image.
        let no_group = motion_shutter_fixture(dimensions);
        executor
            .prepare(&gpu.device, &gpu.queue, &no_group, &sources[1..])
            .unwrap();
        assert_eq!(
            executor
                .begin_scope_recorder_readback(RecorderReadbackTag::new(17, recorder_metadata(3))),
            RecorderReadbackAdmission::SourceUnavailable
        );
        assert_eq!(executor.allocation_snapshot().readback_objects, 3);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn advanced_temporal_originals_are_deterministic_and_allocation_free_at_authored_position() {
        let gpu = GpuHarness::new();
        let dimensions = [9, 7];
        let source = gpu.source(
            dimensions,
            [255, 0, 0, 255],
            "Advanced temporal-originals live source",
        );
        let descriptors = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let first_plan = temporal_originals_fixture(
            dimensions,
            TemporalTopology::Linear,
            TemporalInterpolation::Floor,
            0,
        );
        executor
            .prepare(&gpu.device, &gpu.queue, &first_plan, &descriptors)
            .unwrap();
        let warmed = executor.allocation_snapshot();
        let palette = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
            [255, 0, 255, 255],
            [0, 255, 255, 255],
            [255, 255, 255, 255],
            [0, 0, 0, 255],
        ];

        for fps in [24.0_f32, 30.0, 60.0] {
            for topology in [
                TemporalTopology::Linear,
                TemporalTopology::Radial,
                TemporalTopology::Spiral,
                TemporalTopology::Contour,
                TemporalTopology::Folded,
                TemporalTopology::Kaleidoscopic,
            ] {
                for interpolation in [TemporalInterpolation::Floor, TemporalInterpolation::Linear] {
                    for seed in [0, 1, 0x6a09_e667, u32::MAX] {
                        let plan =
                            temporal_originals_fixture(dimensions, topology, interpolation, seed);
                        executor
                            .prepare(&gpu.device, &gpu.queue, &plan, &descriptors)
                            .unwrap();
                        assert_eq!(executor.allocation_snapshot(), warmed);

                        let mut replay = || {
                            executor.reset_history();
                            for frame in 0..14_usize {
                                let color = palette[(frame + seed as usize) % palette.len()];
                                gpu.write_source(&source, color);
                                gpu.submit(&mut executor, &plan, 1.0 / fps, true);
                            }
                            let color = palette[(14 + seed as usize) % palette.len()];
                            gpu.write_source(&source, color);
                            gpu.render(&mut executor, &plan, 1.0 / fps, true)
                        };
                        let first = replay();
                        let second = replay();
                        assert_eq!(
                            first, second,
                            "fps={fps}, topology={topology:?}, interpolation={interpolation:?}, seed={seed}"
                        );
                        assert_eq!(executor.allocation_snapshot(), warmed);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn production_group_output_reference_is_pre_admission_and_missing_is_transparent() {
        let gpu = GpuHarness::new();
        let dimensions = [3, 3];
        let valid = group_output_reference_fixture(dimensions, false);
        let missing = group_output_reference_fixture(dimensions, true);
        let consumer = gpu.source(dimensions, [255, 255, 255, 255], "Group-output consumer");
        let member = gpu.source(dimensions, [0, 0, 0, 255], "Group-output processed member");
        let matte_donor = gpu.source(dimensions, [255, 255, 255, 255], "Group-output matte donor");
        let sources = [
            CompositionSourceDescriptor::new(stable_layer(2), &consumer.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(1), &member.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(3), &matte_donor.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &valid, &sources)
            .unwrap();
        let referenced = gpu.render(&mut executor, &valid, 1.0 / 30.0, false);
        assert!(
            referenced.iter().all(|pixel| pixel[3] >= 0.99),
            "the post-rack/post-matte, pre-opacity group reference was not captured: {:?}; order {:?}",
            referenced[0],
            match &valid {
                EvaluatedCompositionPlan::Advanced(plan) => plan.execution_order(),
                EvaluatedCompositionPlan::LegacyExact(_) => &[],
            }
        );

        executor
            .prepare(&gpu.device, &gpu.queue, &missing, &sources[..1])
            .unwrap();
        let tombstone = gpu.render(&mut executor, &missing, 1.0 / 30.0, false);
        assert!(
            tombstone.iter().all(|pixel| pixel[3] <= 0.001),
            "a deleted/missing group reference must bind transparent: {:?}",
            tombstone[0]
        );
    }

    fn program_history_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        master
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte {
                    tap: ResolvedImageTap {
                        source: crate::visual_rack::ResolvedImageSource::CleanProgram,
                        timing: EdgeTiming::PreviousFrame,
                    },
                    channel: MatteChannel::Red,
                    invert: false,
                    amount: 1.0,
                    threshold: 0.5,
                    softness: 0.0,
                },
            )))
            .unwrap();
        let racks = vec![(
            stable_layer(1),
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
        )];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &master,
                &racks,
            ),
        )
        .unwrap()
    }

    fn selected_previous_fixture(dimensions: [u32; 2]) -> EvaluatedCompositionPlan {
        let base = evaluated_base(&[2, 1], dimensions);
        let composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(1),
                    bus: BusAssignment::A,
                },
                RuntimeRootItem::Layer {
                    layer_id: stable_layer(2),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            1.0,
        )
        .unwrap();
        let mut consumer = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        consumer
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte {
                    tap: ResolvedImageTap {
                        source: crate::visual_rack::ResolvedImageSource::SelectedLayer {
                            layer_id: stable_layer(1),
                            saved_position: SavedLayerPosition::new(2).unwrap(),
                            stage: LayerImageStage::PostLocalEffects,
                        },
                        timing: EdgeTiming::PreviousFrame,
                    },
                    channel: MatteChannel::Red,
                    invert: false,
                    amount: 1.0,
                    threshold: 0.5,
                    softness: 0.0,
                },
            )))
            .unwrap();
        let racks = vec![
            (stable_layer(2), consumer),
            (
                stable_layer(1),
                RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
            ),
        ];
        EvaluatedCompositionPlan::evaluate(
            &base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &composition,
                &RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master),
                &racks,
            ),
        )
        .unwrap()
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn nonprogram_previous_tap_discards_submitted_rejected_parity() {
        let gpu = GpuHarness::new();
        let dimensions = [2, 2];
        let plan = selected_previous_fixture(dimensions);
        let consumer = gpu.source(dimensions, [255, 255, 255, 255], "Previous consumer");
        let donor = gpu.source(dimensions, [255, 255, 255, 255], "Previous donor A");
        let descriptors = [
            CompositionSourceDescriptor::new(stable_layer(2), &consumer.view, dimensions),
            CompositionSourceDescriptor::new(stable_layer(1), &donor.view, dimensions),
        ];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &descriptors)
            .unwrap();

        let first = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        assert!(first.iter().all(|pixel| pixel[3] <= 0.001));
        executor.clear_temporal_memory();
        gpu.write_source(&donor, [0, 0, 0, 255]);
        let rejected = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert!(rejected.iter().all(|pixel| pixel[3] >= 0.999));
        gpu.write_source(&donor, [255, 255, 255, 255]);
        let after_reject = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        assert!(
            after_reject.iter().all(|pixel| pixel[3] >= 0.999),
            "discard published rejected non-Program history: {:?}",
            after_reject[0]
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn program_history_is_first_frame_transparent_n_minus_one_and_commit_discard_safe() {
        let gpu = GpuHarness::new();
        let dimensions = [2, 2];
        let plan = program_history_fixture(dimensions);
        let source = gpu.source(dimensions, [255, 255, 255, 255], "Program history source");
        let descriptors = [CompositionSourceDescriptor::new(
            stable_layer(1),
            &source.view,
            dimensions,
        )];
        let mut executor =
            CompositionGpuExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &descriptors)
            .unwrap();
        assert!(!executor.program_history_initialized());

        let cold = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        assert!(cold.iter().all(|pixel| pixel[3] <= 0.001));
        assert!(executor.program_history_initialized());
        executor.clear_temporal_memory();
        assert!(
            executor.program_history_initialized(),
            "manual temporal clear must not revoke N-1 Program history"
        );

        // The rejected frame writes transparent B only into the inactive
        // parity. Discard must keep committed opaque A as the next N-1 donor.
        gpu.write_source(&source, [0, 0, 0, 255]);
        let rejected = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert!(
            rejected.iter().all(|pixel| pixel[3] >= 0.999),
            "rejected frame sampled committed A: {:?}",
            rejected[0]
        );
        gpu.write_source(&source, [255, 255, 255, 255]);
        let after_reject = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        assert!(
            after_reject.iter().all(|pixel| pixel[3] >= 0.999),
            "post-reject frame sampled committed A: {:?}",
            after_reject[0]
        );

        executor.reset_history();
        assert!(!executor.program_history_initialized());
        let reset = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert!(reset.iter().all(|pixel| pixel[3] <= 0.001));

        // Replacing a view under the same stable ID and dimensions is keyed
        // by the caller's resource epoch. Preparation must bind the new black
        // texture and start history cold; a stale white bind would make the
        // second frame opaque.
        let replacement = gpu.source(
            dimensions,
            [0, 0, 0, 255],
            "Program history replacement source",
        );
        let replacement_descriptors =
            [
                CompositionSourceDescriptor::new(stable_layer(1), &replacement.view, dimensions)
                    .with_resource_epoch(2),
            ];
        let replacement_keys = [(stable_layer(1), dimensions, 2)];
        assert!(!executor.is_prepared_for(&plan, &replacement_keys));
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &replacement_descriptors)
            .unwrap();
        assert!(executor.is_prepared_for(&plan, &replacement_keys));
        assert!(!executor.program_history_initialized());
        let replacement_cold = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        let replacement_warm = gpu.render(&mut executor, &plan, 1.0 / 30.0, true);
        assert!(replacement_cold.iter().all(|pixel| pixel[3] <= 0.001));
        assert!(replacement_warm.iter().all(|pixel| pixel[3] <= 0.001));

        // Advanced -> exact destroys prepared history. Re-entering the same
        // topology/source identity is a clean rebuild, never a stale revival.
        let exact_base = evaluated_base(&[1], dimensions);
        let exact_composition = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![RuntimeRootItem::Layer {
                layer_id: stable_layer(1),
                bus: BusAssignment::Program,
            }],
            None,
            0.5,
        )
        .unwrap();
        let exact_master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let exact_racks = vec![(
            stable_layer(1),
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
        )];
        let exact = EvaluatedCompositionPlan::evaluate(
            &exact_base,
            crate::evaluated_frame::evaluated_composition::CompositionPlanInput::new(
                &exact_composition,
                &exact_master,
                &exact_racks,
            ),
        )
        .unwrap();
        assert!(matches!(exact, EvaluatedCompositionPlan::LegacyExact(_)));
        executor
            .prepare(&gpu.device, &gpu.queue, &exact, &[])
            .unwrap();
        assert!(!executor.program_history_initialized());
        executor
            .prepare(&gpu.device, &gpu.queue, &plan, &replacement_descriptors)
            .unwrap();
        assert!(!executor.program_history_initialized());
        let after_exact = gpu.render(&mut executor, &plan, 1.0 / 30.0, false);
        assert!(after_exact.iter().all(|pixel| pixel[3] <= 0.001));
    }
}
