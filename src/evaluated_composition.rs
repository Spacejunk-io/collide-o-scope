//! Pure Milestone 2 composition and Collision Rack planning.
//!
//! This module owns no GPU state and never mutates authored/runtime values.
//! It wraps an already evaluated M0/M1 frame in a stable-ID composition,
//! resolves every image route, validates current-frame topology, compiles
//! marker-free GPU rack segments, and preflights one conservative creative
//! resource ledger. Live and export can therefore consume the same plan.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{EvaluatedFramePlan, EvaluatedLayerMatte};
use crate::composition::{
    BusAssignment, FlattenedGroupSpan, GroupOutputStage, RuntimeComposition,
    RuntimeCompositionError, RuntimeRootItem,
};
use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::image_routing::{
    ImageInput, ImageRouteCycle, ImageRouteDiagnostic, LayerImageStage, LayerMatte,
    MissingImageInput, StableLayerId,
};
use crate::motion::{
    resolve_motion_source, MotionDeviceLimits, MotionDonor, MotionField, MotionFieldOrigin,
    MotionGrid, MotionParams, MotionPlanError, MotionResourcePlan, MotionScopeResourceRequest,
    MotionSourceDecision, MotionSourceDiagnostic, MOTION_ALGORITHM_VERSION,
};
use crate::performance::SavedLayerPosition;
use crate::renderer::compositor::{MatteChannelCode, ResolvedMatteParams};
use crate::renderer::rack::{CollisionRackPlan, RackCompileError};
use crate::spatial::{EffectPassUniforms, SpatialGpuUniforms, SpatialTransform};
use crate::visual_rack::{
    CreativeResourceLimits, CreativeResourcePlan, EdgeTiming, GroupId, ImageDependency,
    ImageDependencyGraph, ImageGraphError, ImageGraphMode, ImageGraphPlan, LegacyRackScope, NodeId,
    NodeKindTag, ResolvedImageSource, ResolvedImageTap, ResourcePreflightError, RouteCaptureError,
    RuntimeImageMatte, RuntimeMaskParams, RuntimeRackError, RuntimeVisualNode,
    RuntimeVisualNodeKind, RuntimeVisualRack, VisualRack, VisualScopeId,
    ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS, ADVANCED_RACK_SURFACE_LAYERS,
    ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS, MAX_IMAGE_DEPENDENCIES,
    MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME, MAX_TEXTURE_SAMPLES_PER_FRAME,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

/// Inputs are frame-local values after Morph and stable modulation projection.
/// The planner intentionally has no callback into mutable performance state.
#[derive(Clone, Copy)]
pub struct CompositionPlanInput<'a> {
    pub composition: &'a RuntimeComposition,
    pub master_rack: &'a RuntimeVisualRack,
    pub layer_racks: &'a [(StableLayerId, RuntimeVisualRack)],
    /// Raw M1 layer mattes for unified M2 routing. Supplying this slice avoids
    /// resolving/budgeting them through the positional M1 compositor first.
    /// `None` preserves compatibility for callers that already attached M1
    /// routing to the base frame.
    pub layer_mattes: Option<&'a [LayerMatte]>,
    pub program_history_initialized: bool,
    pub resource_limits: CreativeResourceLimits,
    /// Omitted motion retains the literal pre-M4 exact path. Live and export
    /// provide the same immutable values through `with_motion`.
    pub motion: Option<MotionPlanInput<'a>>,
}

impl<'a> CompositionPlanInput<'a> {
    pub fn new(
        composition: &'a RuntimeComposition,
        master_rack: &'a RuntimeVisualRack,
        layer_racks: &'a [(StableLayerId, RuntimeVisualRack)],
    ) -> Self {
        Self {
            composition,
            master_rack,
            layer_racks,
            layer_mattes: None,
            program_history_initialized: false,
            resource_limits: CreativeResourceLimits::default(),
            motion: None,
        }
    }

    /// Route authored mattes through the unified stable-ID composition graph.
    /// Count and topology are validated atomically by `evaluate`.
    pub fn with_layer_mattes(
        mut self,
        mattes: &'a [LayerMatte],
        program_history_initialized: bool,
    ) -> Self {
        self.layer_mattes = Some(mattes);
        self.program_history_initialized = program_history_initialized;
        self
    }

    pub fn with_motion(
        mut self,
        master: MotionParams,
        layers: &'a [LayerMotionPlanInput],
        limits: MotionDeviceLimits,
    ) -> Self {
        self.motion = Some(MotionPlanInput {
            master,
            layers,
            limits,
        });
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionCodecFrameFacts {
    pub available: bool,
    pub source_generation: u64,
    pub frame_ordinal: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerMotionPlanInput {
    pub stable_id: StableLayerId,
    pub params: MotionParams,
    pub codec: MotionCodecFrameFacts,
}

#[derive(Debug, Clone, Copy)]
pub struct MotionPlanInput<'a> {
    pub master: MotionParams,
    pub layers: &'a [LayerMotionPlanInput],
    pub limits: MotionDeviceLimits,
}

/// Exact omitted-patch topology. No advanced rack is compiled or allocated.
#[derive(Debug, Clone)]
pub struct LegacyExactCompositionPlan {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "exact payload is consumed by the delegated legacy path and parity tests"
        )
    )]
    base: EvaluatedFramePlan,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "exact order is retained for delegate/parity verification"
        )
    )]
    flattened_layers: Box<[StableLayerId]>,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "exact topology identity is retained for delegate/parity verification"
        )
    )]
    topology_signature: u64,
}

impl LegacyExactCompositionPlan {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "exact-plan access supports legacy delegates and parity tests"
        )
    )]
    pub fn base(&self) -> &EvaluatedFramePlan {
        &self.base
    }

    /// Back-to-front composition order. The legacy/UI base remains
    /// front-to-back, so a three-layer exact migration returns its reverse.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "exact-order access supports legacy delegates and parity tests"
        )
    )]
    pub fn flattened_layers(&self) -> &[StableLayerId] {
        &self.flattened_layers
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "exact topology identity supports legacy delegates and parity tests"
        )
    )]
    pub const fn topology_signature(&self) -> u64 {
        self.topology_signature
    }
}

#[derive(Debug, Clone)]
pub enum EvaluatedCompositionPlan {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "legacy payload remains available to exact-path delegates"
        )
    )]
    LegacyExact(Box<LegacyExactCompositionPlan>),
    Advanced(Box<AdvancedCompositionPlan>),
}

impl EvaluatedCompositionPlan {
    pub fn evaluate(
        base: &EvaluatedFramePlan,
        input: CompositionPlanInput<'_>,
    ) -> Result<Self, CompositionPlanError> {
        Planner::new(base, input)?.finish()
    }

    #[allow(
        dead_code,
        reason = "uniform plan inspection API retained for renderer/export adapters"
    )]
    pub fn base(&self) -> &EvaluatedFramePlan {
        match self {
            Self::LegacyExact(plan) => plan.base(),
            Self::Advanced(plan) => plan.base(),
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "uniform topology inspection is exercised by adapter tests"
        )
    )]
    pub const fn topology_signature(&self) -> u64 {
        match self {
            Self::LegacyExact(plan) => plan.topology_signature(),
            Self::Advanced(plan) => plan.topology_signature(),
        }
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "uniform exact-path classification is exercised by adapter tests"
        )
    )]
    pub const fn is_legacy_exact(&self) -> bool {
        matches!(self, Self::LegacyExact(_))
    }
}

/// An exact host/GPU boundary in authored rack order.
#[derive(Debug, Clone)]
pub enum EvaluatedScopeStep {
    /// Neutral effects plus the M0 SpatialTransform, exactly once. All later
    /// steps consume and produce output-sized images.
    MaterializeSpatial {
        pass: EffectPassUniforms,
        application: LegacyCanonicalApplication,
    },
    /// Existing effect block with identity output-to-output spatial sampling.
    LegacyCanonical {
        pass: EffectPassUniforms,
        application: LegacyCanonicalApplication,
    },
    /// One marker-free ordered segment compiled by the fixed rack executor.
    CollisionRack {
        segment_index: u8,
        plan: CollisionRackPlan,
    },
    /// Stateful master-only boundary in its exact authored position.
    LegacyTemporal { params: TemporalParams },
    /// Group matte is post-transform/rack and pre-opacity/admission.
    GroupMatte {
        #[allow(
            dead_code,
            reason = "matte payload is retained for planned-boundary diagnostics"
        )]
        matte: RuntimeImageMatte,
    },
}

impl EvaluatedScopeStep {
    #[allow(
        dead_code,
        reason = "node-tag inspection is retained for plan diagnostics"
    )]
    pub const fn node_kind_tag(&self) -> Option<NodeKindTag> {
        match self {
            Self::LegacyCanonical { .. } => Some(NodeKindTag::LegacyCanonical),
            Self::LegacyTemporal { .. } => Some(NodeKindTag::LegacyTemporal),
            Self::MaterializeSpatial { .. }
            | Self::CollisionRack { .. }
            | Self::GroupMatte { .. } => None,
        }
    }
}

/// The inherited canonical block historically observes each layer's
/// `bypass_master_fx` flag. Layer markers operate on their local scope; a
/// master marker is therefore a pre-composite admission boundary, while
/// custom master segments and Temporal remain composition-global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyCanonicalApplication {
    ScopeLocal,
    PreCompositeLayerAdmission,
}

#[derive(Debug, Clone)]
pub enum EvaluatedScopeExecution {
    /// Scope-local layer use of the frozen combined M0/M1 path. The whole
    /// composition exact path is tagged separately; an advanced master always
    /// expands its host boundaries so bypass/NTSC scheduling remains explicit.
    ExactLegacy {
        #[allow(
            dead_code,
            reason = "scope identity is retained for exact-path diagnostics"
        )]
        scope: LegacyRackScope,
    },
    Ordered {
        steps: Box<[EvaluatedScopeStep]>,
    },
}

impl EvaluatedScopeExecution {
    pub fn steps(&self) -> &[EvaluatedScopeStep] {
        match self {
            Self::ExactLegacy { .. } => &[],
            Self::Ordered { steps } => steps,
        }
    }

    pub const fn is_exact_legacy(&self) -> bool {
        matches!(self, Self::ExactLegacy { .. })
    }
}

#[derive(Debug, Clone)]
pub struct EvaluatedMasterScopePlan {
    pub execution: EvaluatedScopeExecution,
    /// Contributing layers to which a master LegacyCanonical boundary applies.
    /// Empty when the explicit master rack contains no canonical marker.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "canonical admission membership is exposed for plan diagnostics"
        )
    )]
    pub canonical_layers: Box<[StableLayerId]>,
    /// Contributing layers retaining inherited `bypass_master_fx` behavior.
    pub canonical_bypass_layers: Box<[StableLayerId]>,
    /// Existing selective NTSC is also a pre-composite per-layer operation;
    /// Temporal remains the downstream composition-global boundary.
    pub selective_ntsc_layers: Box<[StableLayerId]>,
    pub selective_ntsc_bypass_layers: Box<[StableLayerId]>,
}

#[derive(Debug, Clone)]
pub struct EvaluatedLayerScopePlan {
    pub stable_id: StableLayerId,
    /// Index into the owned [`EvaluatedFramePlan`] payload.
    pub base_layer_index: usize,
    pub group_id: Option<GroupId>,
    pub bus: BusAssignment,
    pub admitted_to_program: bool,
    /// M1 matte payload retained exactly, but routed by this plan's stable
    /// composition domains and unified dependency/resource ledger.
    pub legacy_matte: Option<EvaluatedLayerMatte>,
    pub execution: EvaluatedScopeExecution,
}

#[derive(Debug, Clone)]
pub struct EvaluatedGroupScopePlan {
    pub id: GroupId,
    #[allow(
        dead_code,
        reason = "authored group label is retained in immutable plans for diagnostics/UI"
    )]
    pub name: String,
    pub members: Box<[StableLayerId]>,
    pub span: FlattenedGroupSpan,
    pub opacity: f32,
    #[allow(
        dead_code,
        reason = "authored group transform is retained alongside its compiled spatial step"
    )]
    pub transform: SpatialTransform,
    pub matte: Option<RuntimeImageMatte>,
    pub solo: bool,
    pub bypass: bool,
    pub bus: BusAssignment,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "group output boundary metadata is exposed for routing diagnostics"
        )
    )]
    pub output_stage: GroupOutputStage,
    pub execution: EvaluatedScopeExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageTapConsumer {
    RackNode {
        scope: VisualScopeId,
        node_id: NodeId,
    },
    GroupMatte {
        group_id: GroupId,
    },
    LayerMatte {
        layer_id: StableLayerId,
    },
}

impl ImageTapConsumer {
    pub const fn scope(self) -> VisualScopeId {
        match self {
            Self::RackNode { scope, .. } => scope,
            Self::GroupMatte { group_id } => VisualScopeId::Group(group_id),
            Self::LayerMatte { layer_id } => VisualScopeId::Layer(layer_id),
        }
    }
}

/// Compact identity of an AllBelow composite. It never expands one logical
/// tap into O(layer-count) retained textures or dependency records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompositePrefix {
    Root {
        preceding_root_outputs: usize,
    },
    GroupMember {
        group_id: GroupId,
        preceding_root_outputs: usize,
        preceding_members: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannedImageSource {
    Scope(VisualScopeId),
    /// Stable layer source at the declared stage. Current-frame PreLocal is an
    /// external source-input binding, not a dependency on evaluated scope
    /// output; PostLocal and all previous-frame variants still retain their
    /// declared temporal dependency.
    SelectedLayer {
        layer_id: StableLayerId,
        stage: LayerImageStage,
    },
    OneBelow(VisualScopeId),
    /// Exact root/group-local prefix, ordered back-to-front.
    AllBelow(CompositePrefix),
    ProgramHistory,
    Transparent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannedImageTapOrigin {
    Rack(ResolvedImageTap),
    LegacyLayerMatte {
        input: ImageInput,
        diagnostic: ImageRouteDiagnostic,
    },
}

impl PlannedImageTapOrigin {
    pub const fn timing(self) -> EdgeTiming {
        match self {
            Self::Rack(tap) => tap.timing,
            Self::LegacyLayerMatte {
                input: ImageInput::ProgramHistory,
                ..
            } => EdgeTiming::PreviousFrame,
            Self::LegacyLayerMatte { .. } => EdgeTiming::CurrentFrame,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedImageTap {
    pub consumer: ImageTapConsumer,
    pub origin: PlannedImageTapOrigin,
    pub resolved: PlannedImageSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionPlanDiagnostic {
    MissingSelectedLayer {
        consumer: ImageTapConsumer,
        saved_position: SavedLayerPosition,
    },
    MissingStableScope {
        consumer: ImageTapConsumer,
        producer: VisualScopeId,
    },
    MissingGroupOutput {
        consumer: ImageTapConsumer,
        group_id: GroupId,
    },
    NoOneBelow {
        consumer: ImageTapConsumer,
    },
    NoAllBelow {
        consumer: ImageTapConsumer,
    },
    LegacyCleanProgramTransparent {
        consumer: ImageTapConsumer,
    },
    ProgramHistoryUninitialized {
        consumer: ImageTapConsumer,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionPlanDiagnostic {
    Source {
        scope: VisualScopeId,
        diagnostic: MotionSourceDiagnostic,
    },
    MasterTransplantRejected,
    DonorNotSelected {
        recipient: StableLayerId,
    },
    MissingDonor {
        recipient: StableLayerId,
        saved_position: SavedLayerPosition,
    },
    ExcessTransplantRejected {
        recipient: StableLayerId,
        admitted_recipient: StableLayerId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluatedMotionFieldPlan {
    pub slot: u8,
    pub scope: VisualScopeId,
    pub source_dimensions: [u32; 2],
    pub grid: MotionGrid,
    pub algorithm_version: u16,
    pub source: MotionSourceDecision,
    pub codec: MotionCodecFrameFacts,
    pub required_as_donor: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedMotionScopePlan {
    pub scope: VisualScopeId,
    pub params: MotionParams,
    pub source_dimensions: [u32; 2],
    /// Sanitized post-Morph/post-modulation authored transform for this exact
    /// frame. The executor retains the last accepted value transactionally
    /// and evaluates shutter samples between the two immutable frame facts.
    pub transform: SpatialTransform,
    pub spatial: SpatialGpuUniforms,
    pub source: MotionSourceDecision,
    pub field_slot: Option<u8>,
    pub required_as_donor: bool,
    pub donor_scope: Option<VisualScopeId>,
    pub donor_field_slot: Option<u8>,
    pub transplant_admitted: bool,
    pub codec: MotionCodecFrameFacts,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionFrameBudget {
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_samples_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
}

#[derive(Debug, Clone)]
pub struct AdvancedMotionPlan {
    scopes: Box<[EvaluatedMotionScopePlan]>,
    fields: Box<[EvaluatedMotionFieldPlan]>,
    resources: MotionResourcePlan,
    diagnostics: Box<[MotionPlanDiagnostic]>,
    budget: MotionFrameBudget,
    topology_signature: u64,
}

#[derive(Debug, Clone, Default)]
pub enum EvaluatedMotionPlan {
    #[default]
    LegacyExact,
    Advanced(Box<AdvancedMotionPlan>),
}

impl EvaluatedMotionPlan {
    pub const fn is_legacy_exact(&self) -> bool {
        matches!(self, Self::LegacyExact)
    }

    pub fn advanced(&self) -> Option<&AdvancedMotionPlan> {
        match self {
            Self::LegacyExact => None,
            Self::Advanced(plan) => Some(plan),
        }
    }

    pub const fn topology_signature(&self) -> u64 {
        match self {
            Self::LegacyExact => 0,
            Self::Advanced(plan) => plan.topology_signature,
        }
    }
}

impl AdvancedMotionPlan {
    pub fn scopes(&self) -> &[EvaluatedMotionScopePlan] {
        &self.scopes
    }

    pub fn fields(&self) -> &[EvaluatedMotionFieldPlan] {
        &self.fields
    }

    pub const fn resources(&self) -> MotionResourcePlan {
        self.resources
    }

    pub fn diagnostics(&self) -> &[MotionPlanDiagnostic] {
        &self.diagnostics
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "export/runtime metadata consumes this frozen accessor"
        )
    )]
    pub const fn budget(&self) -> MotionFrameBudget {
        self.budget
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "export/runtime metadata consumes this frozen accessor"
        )
    )]
    pub const fn topology_signature(&self) -> u64 {
        self.topology_signature
    }

    pub fn scope(&self, scope: VisualScopeId) -> Option<&EvaluatedMotionScopePlan> {
        self.scopes
            .iter()
            .find(|candidate| candidate.scope == scope)
    }

    #[allow(
        dead_code,
        reason = "live/export attachment adapters consume this frozen slot accessor"
    )]
    pub fn field(&self, slot: u8) -> Option<&EvaluatedMotionFieldPlan> {
        self.fields.iter().find(|field| field.slot == slot)
    }
}

/// Frame-owned field products are borrowed across prepare/encode only. Exact
/// provenance matching prevents stale codec vectors binding after resize,
/// quality changes, seek, or decoder generation replacement.
#[derive(Debug, Clone, Copy)]
pub struct MotionFieldAttachment<'a> {
    pub scope: VisualScopeId,
    pub source_generation: u64,
    pub frame_ordinal: u64,
    pub algorithm_version: u16,
    pub source_dimensions: [u32; 2],
    pub grid: MotionGrid,
    pub field: &'a MotionField,
}

impl EvaluatedMotionFieldPlan {
    pub fn accepts(self, attachment: MotionFieldAttachment<'_>) -> bool {
        self.scope == attachment.scope
            && self.source.origin == MotionFieldOrigin::CodecVectors
            && self.codec.available
            && self.codec.source_generation == attachment.source_generation
            && self.codec.frame_ordinal == attachment.frame_ordinal
            && self.algorithm_version == MOTION_ALGORITHM_VERSION
            && self.algorithm_version == attachment.algorithm_version
            && self.algorithm_version == attachment.field.algorithm_version()
            && self.source_dimensions == attachment.source_dimensions
            && self.source_dimensions == attachment.field.source_dimensions()
            && self.grid == attachment.grid
            && self.grid == attachment.field.grid()
            && attachment.field.origin() == MotionFieldOrigin::CodecVectors
    }
}

#[derive(Debug, Clone)]
pub struct AdvancedCompositionPlan {
    base: EvaluatedFramePlan,
    master: EvaluatedMasterScopePlan,
    layers: Box<[EvaluatedLayerScopePlan]>,
    groups: Box<[EvaluatedGroupScopePlan]>,
    root: Box<[RuntimeRootItem]>,
    bus_crossfade: f32,
    image_taps: Box<[PlannedImageTap]>,
    diagnostics: Box<[CompositionPlanDiagnostic]>,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "below topology is retained for planner diagnostics and goldens"
        )
    )]
    below: BelowTopology,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "image graph is retained for planner diagnostics and goldens"
        )
    )]
    graph: ImageGraphPlan,
    execution_order: Box<[VisualScopeId]>,
    resources: CreativeResourcePlan,
    motion: EvaluatedMotionPlan,
    topology_signature: u64,
}

/// Frame-local NTSC routing classification shared by live and offline
/// executors. Only pixels with a finite positive effective admission weight
/// participate, so dormant zero-opacity/group/bus branches cannot create a
/// false selective-VHS split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedNtscPath {
    Disabled,
    AllApplying,
    AllBypass,
    Mixed,
}

impl AdvancedCompositionPlan {
    pub fn base(&self) -> &EvaluatedFramePlan {
        &self.base
    }

    pub fn master(&self) -> &EvaluatedMasterScopePlan {
        &self.master
    }

    pub fn layers(&self) -> &[EvaluatedLayerScopePlan] {
        &self.layers
    }

    pub fn groups(&self) -> &[EvaluatedGroupScopePlan] {
        &self.groups
    }

    pub fn root(&self) -> &[RuntimeRootItem] {
        &self.root
    }

    pub const fn bus_crossfade(&self) -> f32 {
        self.bus_crossfade
    }

    pub fn image_taps(&self) -> &[PlannedImageTap] {
        &self.image_taps
    }

    pub fn diagnostics(&self) -> &[CompositionPlanDiagnostic] {
        &self.diagnostics
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "planner topology inspection is exercised by goldens"
        )
    )]
    pub const fn below_topology(&self) -> &BelowTopology {
        &self.below
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "planner graph inspection is exercised by goldens")
    )]
    pub const fn graph(&self) -> &ImageGraphPlan {
        &self.graph
    }

    /// Stable producer-before-consumer order including structural group and
    /// final master/program boundaries.
    pub fn execution_order(&self) -> &[VisualScopeId] {
        &self.execution_order
    }

    pub const fn resources(&self) -> CreativeResourcePlan {
        self.resources
    }

    pub const fn motion(&self) -> &EvaluatedMotionPlan {
        &self.motion
    }

    pub const fn topology_signature(&self) -> u64 {
        self.topology_signature
    }

    pub fn ntsc_path(&self) -> AdvancedNtscPath {
        if !self.base.ntsc().enabled {
            return AdvancedNtscPath::Disabled;
        }
        match (
            self.master.selective_ntsc_layers.is_empty(),
            self.master.selective_ntsc_bypass_layers.is_empty(),
        ) {
            (true, true) => AdvancedNtscPath::Disabled,
            (false, true) => AdvancedNtscPath::AllApplying,
            (true, false) => AdvancedNtscPath::AllBypass,
            (false, false) => AdvancedNtscPath::Mixed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BelowLocation {
    Root {
        root_index: usize,
    },
    GroupMember {
        group_id: GroupId,
        root_index: usize,
        member_index: usize,
    },
    Master,
}

/// Compact, immutable interpretation of OneBelow/AllBelow. Storage is linear
/// in actual composition size; stable ID magnitude never affects allocation.
#[derive(Debug, Clone)]
pub struct BelowTopology {
    root_outputs: Box<[VisualScopeId]>,
    locations: BTreeMap<VisualScopeId, BelowLocation>,
    group_members: BTreeMap<GroupId, Box<[StableLayerId]>>,
}

impl BelowTopology {
    pub fn root_outputs(&self) -> &[VisualScopeId] {
        &self.root_outputs
    }

    pub fn prefix_for(&self, consumer: VisualScopeId) -> Option<CompositePrefix> {
        match self.locations.get(&consumer).copied()? {
            BelowLocation::Root { root_index } => Some(CompositePrefix::Root {
                preceding_root_outputs: root_index,
            }),
            BelowLocation::GroupMember {
                group_id,
                root_index,
                member_index,
            } => Some(CompositePrefix::GroupMember {
                group_id,
                preceding_root_outputs: root_index,
                preceding_members: member_index,
            }),
            BelowLocation::Master => Some(CompositePrefix::Root {
                preceding_root_outputs: self.root_outputs.len(),
            }),
        }
    }

    pub fn one_below(&self, consumer: VisualScopeId) -> Option<VisualScopeId> {
        self.prefix_for(consumer)
            .and_then(|prefix| self.last_in_prefix(prefix))
    }

    pub fn prefix_len(&self, prefix: CompositePrefix) -> usize {
        match prefix {
            CompositePrefix::Root {
                preceding_root_outputs,
            } => preceding_root_outputs.min(self.root_outputs.len()),
            CompositePrefix::GroupMember {
                group_id,
                preceding_root_outputs,
                preceding_members,
            } => preceding_root_outputs
                .min(self.root_outputs.len())
                .saturating_add(
                    preceding_members.min(
                        self.group_members
                            .get(&group_id)
                            .map_or(0, |members| members.len()),
                    ),
                ),
        }
    }

    pub fn prefix_contains(&self, prefix: CompositePrefix, producer: VisualScopeId) -> bool {
        match prefix {
            CompositePrefix::Root {
                preceding_root_outputs,
            } => self
                .root_outputs
                .get(..preceding_root_outputs.min(self.root_outputs.len()))
                .is_some_and(|outputs| outputs.contains(&producer)),
            CompositePrefix::GroupMember {
                group_id,
                preceding_root_outputs,
                preceding_members,
            } => {
                self.root_outputs
                    .get(..preceding_root_outputs.min(self.root_outputs.len()))
                    .is_some_and(|outputs| outputs.contains(&producer))
                    || matches!(producer, VisualScopeId::Layer(id) if self
                        .group_members
                        .get(&group_id)
                        .and_then(|members| members.get(..preceding_members.min(members.len())))
                        .is_some_and(|members| members.contains(&id)))
            }
        }
    }

    fn last_in_prefix(&self, prefix: CompositePrefix) -> Option<VisualScopeId> {
        match prefix {
            CompositePrefix::Root {
                preceding_root_outputs,
            } => preceding_root_outputs
                .checked_sub(1)
                .and_then(|index| self.root_outputs.get(index).copied()),
            CompositePrefix::GroupMember {
                group_id,
                preceding_root_outputs,
                preceding_members,
            } => preceding_members
                .checked_sub(1)
                .and_then(|index| {
                    self.group_members
                        .get(&group_id)
                        .and_then(|members| members.get(index))
                        .copied()
                        .map(VisualScopeId::Layer)
                })
                .or_else(|| {
                    preceding_root_outputs
                        .checked_sub(1)
                        .and_then(|index| self.root_outputs.get(index).copied())
                }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionPlanError {
    InvalidBaseLayerId {
        base_index: usize,
    },
    DuplicateBaseLayerId(StableLayerId),
    DuplicateLayerRack(StableLayerId),
    MissingLayerRack(StableLayerId),
    UnknownLayerRack(StableLayerId),
    DuplicateLayerMotion(StableLayerId),
    MissingLayerMotion(StableLayerId),
    UnknownLayerMotion(StableLayerId),
    Composition(RuntimeCompositionError),
    Rack {
        scope: VisualScopeId,
        error: RuntimeRackError,
    },
    RackCompile {
        scope: VisualScopeId,
        segment_index: u8,
        error: RackCompileError,
    },
    RouteCapture {
        scope: VisualScopeId,
        error: RouteCaptureError,
    },
    TooManyImageRoutes {
        count: usize,
        limit: usize,
    },
    LayerMatteCount {
        count: usize,
        layers: usize,
    },
    ImageGraph(ImageGraphError),
    CurrentCycle {
        scopes: Vec<VisualScopeId>,
    },
    /// Scope-level dependencies can be acyclic while requiring two groups to
    /// be evaluated in interleaved pieces. Groups are deliberately atomic
    /// compositor tasks, so such a collapsed group-task cycle is rejected
    /// before the GPU executor allocates or records work.
    AtomicGroupCycle {
        tasks: Vec<VisualScopeId>,
    },
    AmbiguousMasterBypass {
        layers: Vec<StableLayerId>,
    },
    Resource(ResourcePreflightError),
    Motion(MotionPlanError),
    MotionCombinedMemoryBudget {
        bytes: u64,
        limit: u64,
    },
    MotionCombinedSampleBudget {
        samples: u32,
        limit: u32,
    },
    Internal(&'static str),
}

impl fmt::Display for CompositionPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBaseLayerId { base_index } => {
                write!(
                    formatter,
                    "base layer {base_index} has zero stable identity"
                )
            }
            Self::DuplicateBaseLayerId(id) => {
                write!(formatter, "base frame repeats stable layer {}", id.get())
            }
            Self::DuplicateLayerRack(id) => {
                write!(
                    formatter,
                    "layer rack input repeats stable layer {}",
                    id.get()
                )
            }
            Self::MissingLayerRack(id) => {
                write!(formatter, "stable layer {} has no runtime rack", id.get())
            }
            Self::UnknownLayerRack(id) => {
                write!(
                    formatter,
                    "runtime rack names unknown stable layer {}",
                    id.get()
                )
            }
            Self::DuplicateLayerMotion(id) => write!(
                formatter,
                "motion input repeats stable layer {}",
                id.get()
            ),
            Self::MissingLayerMotion(id) => write!(
                formatter,
                "stable layer {} has no evaluated motion input",
                id.get()
            ),
            Self::UnknownLayerMotion(id) => write!(
                formatter,
                "motion input names unknown stable layer {}",
                id.get()
            ),
            Self::Composition(error) => {
                write!(formatter, "runtime composition is invalid: {error}")
            }
            Self::Rack { scope, error } => write!(formatter, "{scope:?} rack is invalid: {error}"),
            Self::RackCompile {
                scope,
                segment_index,
                error,
            } => write!(
                formatter,
                "{scope:?} rack segment {segment_index} cannot compile: {error}"
            ),
            Self::RouteCapture { scope, error } => {
                write!(
                    formatter,
                    "{scope:?} rack cannot enter resource preflight: {error}"
                )
            }
            Self::TooManyImageRoutes { count, limit } => {
                write!(
                    formatter,
                    "frame has {count} active image routes; limit is {limit}"
                )
            }
            Self::LayerMatteCount { count, layers } => write!(
                formatter,
                "frame has {count} authored layer mattes for {layers} evaluated layers"
            ),
            Self::ImageGraph(error) => {
                write!(formatter, "image dependency graph is invalid: {error}")
            }
            Self::CurrentCycle { scopes } => {
                write!(formatter, "current-frame composition cycle: {scopes:?}")
            }
            Self::AtomicGroupCycle { tasks } => write!(
                formatter,
                "current-frame routes require interleaved atomic group tasks: {tasks:?}"
            ),
            Self::AmbiguousMasterBypass { layers } => write!(
                formatter,
                "advanced master ordering cannot preserve inherited bypass for layers {layers:?}"
            ),
            Self::Resource(error) => {
                write!(formatter, "creative resource preflight failed: {error}")
            }
            Self::Motion(error) => write!(formatter, "motion resource preflight failed: {error}"),
            Self::MotionCombinedMemoryBudget { bytes, limit } => write!(
                formatter,
                "creative plus motion resources request {bytes} bytes; limit is {limit}"
            ),
            Self::MotionCombinedSampleBudget { samples, limit } => write!(
                formatter,
                "creative plus motion processing requests {samples} texture samples per pixel; limit is {limit}"
            ),
            Self::Internal(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CompositionPlanError {}

struct Planner<'a> {
    base: &'a EvaluatedFramePlan,
    input: CompositionPlanInput<'a>,
    base_ids: Vec<StableLayerId>,
    base_index: BTreeMap<StableLayerId, usize>,
    racks: BTreeMap<StableLayerId, &'a RuntimeVisualRack>,
}

impl<'a> Planner<'a> {
    fn new(
        base: &'a EvaluatedFramePlan,
        input: CompositionPlanInput<'a>,
    ) -> Result<Self, CompositionPlanError> {
        // Exact flat legacy stacks remain dynamically sized. These vectors
        // grow only with actual yielded layers; stable ID values are map keys,
        // never capacities. Advanced admission is checked after exact-path
        // recognition in `finish`.
        let mut base_ids = Vec::with_capacity(base.layers().len());
        let mut base_index = BTreeMap::new();
        for (index, layer) in base.layers().iter().enumerate() {
            let id = StableLayerId::new(layer.source.stable_id)
                .ok_or(CompositionPlanError::InvalidBaseLayerId { base_index: index })?;
            if base_index.insert(id, index).is_some() {
                return Err(CompositionPlanError::DuplicateBaseLayerId(id));
            }
            base_ids.push(id);
        }
        input
            .composition
            .validate_for_layers(&base_ids)
            .map_err(CompositionPlanError::Composition)?;
        input
            .master_rack
            .validate_for_scope(LegacyRackScope::Master)
            .map_err(|error| CompositionPlanError::Rack {
                scope: VisualScopeId::Master,
                error,
            })?;

        let mut racks = BTreeMap::new();
        for (id, rack) in input.layer_racks {
            if racks.insert(*id, rack).is_some() {
                return Err(CompositionPlanError::DuplicateLayerRack(*id));
            }
            rack.validate_for_scope(LegacyRackScope::Layer)
                .map_err(|error| CompositionPlanError::Rack {
                    scope: VisualScopeId::Layer(*id),
                    error,
                })?;
        }
        for id in &base_ids {
            if !racks.contains_key(id) {
                return Err(CompositionPlanError::MissingLayerRack(*id));
            }
        }
        if let Some(id) = racks.keys().find(|id| !base_index.contains_key(id)) {
            return Err(CompositionPlanError::UnknownLayerRack(*id));
        }
        Ok(Self {
            base,
            input,
            base_ids,
            base_index,
            racks,
        })
    }

    fn finish(self) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        let flattened = self
            .input
            .composition
            .flatten()
            .map_err(CompositionPlanError::Composition)?;
        let flat_ids: Vec<_> = flattened
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect();
        let motion = self.evaluate_motion(&flat_ids)?;
        if self.is_global_legacy_exact(&flat_ids) && motion.is_legacy_exact() {
            let topology_signature =
                legacy_topology_signature(&flat_ids, self.input.master_rack, &self.racks);
            return Ok(EvaluatedCompositionPlan::LegacyExact(Box::new(
                LegacyExactCompositionPlan {
                    base: self.base.clone(),
                    flattened_layers: flat_ids.into_boxed_slice(),
                    topology_signature,
                },
            )));
        }

        let output = self.base.context().output_size;
        let identity_spatial =
            SpatialTransform::default().gpu_uniforms(output[0], output[1], output[0], output[1]);
        let mut layer_plans = Vec::with_capacity(flattened.layers.len());
        let owned_mattes;
        let legacy_mattes = if let Some(mattes) = self.input.layer_mattes {
            if mattes.len() != self.base.layers().len() {
                return Err(CompositionPlanError::LayerMatteCount {
                    count: mattes.len(),
                    layers: self.base.layers().len(),
                });
            }
            owned_mattes = mattes
                .iter()
                .copied()
                .map(|matte| composition_layer_matte(matte, self.input.program_history_initialized))
                .collect::<Vec<_>>();
            owned_mattes.as_slice()
        } else {
            self.base.image_routing().mattes()
        };
        if !legacy_mattes.is_empty() && legacy_mattes.len() != self.base.layers().len() {
            return Err(CompositionPlanError::Internal(
                "base image-routing mattes lost layer alignment",
            ));
        }
        for layer in &flattened.layers {
            let base_layer_index = self.base_index[&layer.layer_id];
            let rack = self.racks[&layer.layer_id];
            let canonical = EffectPassUniforms::new(
                self.base.layer_passes()[base_layer_index].effects,
                identity_spatial,
            );
            let execution = compile_scope_execution(
                rack,
                LegacyRackScope::Layer,
                VisualScopeId::Layer(layer.layer_id),
                ScopeHostPayload {
                    materialize: self.base.layer_pre_passes()[base_layer_index],
                    canonical: Some(canonical),
                    temporal: None,
                    canonical_application: LegacyCanonicalApplication::ScopeLocal,
                    allow_exact_scope: true,
                },
                output,
            )?;
            let legacy_matte = legacy_mattes
                .get(base_layer_index)
                .copied()
                .filter(|matte| matte.authored.enabled && matte.authored.amount > 0.0);
            layer_plans.push(EvaluatedLayerScopePlan {
                stable_id: layer.layer_id,
                base_layer_index,
                group_id: layer.group_id,
                bus: layer.bus,
                admitted_to_program: layer.admitted_to_program,
                legacy_matte,
                execution,
            });
        }

        let neutral = EffectUniforms {
            time: self.base.context().time_seconds,
            ..EffectUniforms::default()
        };
        let mut group_plans = Vec::with_capacity(flattened.groups.len());
        for span in &flattened.groups {
            let group = self.input.composition.group(span.group_id).ok_or(
                CompositionPlanError::Internal("validated group span lost its runtime group"),
            )?;
            let materialize = EffectPassUniforms::for_target(
                neutral,
                group.transform,
                (output[0], output[1]),
                (output[0], output[1]),
            );
            let mut execution = compile_scope_execution(
                &group.rack,
                LegacyRackScope::Group,
                VisualScopeId::Group(group.id),
                ScopeHostPayload {
                    materialize,
                    canonical: None,
                    temporal: None,
                    canonical_application: LegacyCanonicalApplication::ScopeLocal,
                    allow_exact_scope: false,
                },
                output,
            )?;
            if let Some(matte) = group.matte {
                let EvaluatedScopeExecution::Ordered { steps } = &mut execution else {
                    return Err(CompositionPlanError::Internal(
                        "group rack cannot use a legacy execution path",
                    ));
                };
                let mut owned = steps.to_vec();
                owned.push(EvaluatedScopeStep::GroupMatte { matte });
                *steps = owned.into_boxed_slice();
            }
            group_plans.push(EvaluatedGroupScopePlan {
                id: group.id,
                name: group.name.as_str().to_string(),
                members: group.members.iter().collect::<Vec<_>>().into_boxed_slice(),
                span: *span,
                opacity: group.opacity,
                transform: group.transform,
                matte: group.matte,
                solo: group.solo,
                bypass: group.bypass,
                bus: group.bus,
                output_stage: span.output_stage,
                execution,
            });
        }

        let master_materialize = EffectPassUniforms::for_target(
            neutral,
            *self.base.master_transform(),
            (output[0], output[1]),
            (output[0], output[1]),
        );
        let master_canonical =
            EffectPassUniforms::new(self.base.master_pass().effects, identity_spatial);
        let master_has_canonical = self
            .input
            .master_rack
            .iter()
            .any(|node| matches!(node.kind, RuntimeVisualNodeKind::LegacyCanonical));
        let mut canonical_layers = Vec::new();
        let mut canonical_bypass_layers = Vec::new();
        let mut selective_ntsc_layers = Vec::new();
        let mut selective_ntsc_bypass_layers = Vec::new();
        let mut contributing_bypass = Vec::new();
        for layer in &layer_plans {
            let evaluated = &self.base.layers()[layer.base_layer_index];
            if !layer_effectively_contributes(
                layer,
                evaluated,
                &group_plans,
                self.input.composition.bus_crossfade(),
            ) {
                continue;
            }
            if evaluated.bypass_master_fx {
                contributing_bypass.push(layer.stable_id);
            }
            if self.base.ntsc().enabled {
                if evaluated.bypass_master_fx {
                    selective_ntsc_bypass_layers.push(layer.stable_id);
                } else {
                    selective_ntsc_layers.push(layer.stable_id);
                }
            }
            if master_has_canonical {
                if evaluated.bypass_master_fx {
                    canonical_bypass_layers.push(layer.stable_id);
                } else {
                    canonical_layers.push(layer.stable_id);
                }
            }
        }
        let master_has_custom = self.input.master_rack.iter().any(|node| {
            !matches!(
                node.kind,
                RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal
            )
        });
        let mut saw_global_master_step = false;
        let mut canonical_after_global = false;
        for node in self.input.master_rack.iter() {
            if matches!(node.kind, RuntimeVisualNodeKind::LegacyCanonical) {
                canonical_after_global |= saw_global_master_step;
            } else {
                saw_global_master_step = true;
            }
        }
        if (master_has_custom || canonical_after_global) && !contributing_bypass.is_empty() {
            return Err(CompositionPlanError::AmbiguousMasterBypass {
                layers: contributing_bypass,
            });
        }
        // Frozen master semantics are composition-global when every
        // contributing layer inherits master FX. Only a real contributing
        // bypass partition distributes the materialize/canonical prefix over
        // eligible layer admissions. This keeps nonlinear canonical effects
        // downstream of group opacity/rack/matte and A/B composition in the
        // ordinary no-bypass case.
        let canonical_application = if contributing_bypass.is_empty() {
            LegacyCanonicalApplication::ScopeLocal
        } else {
            LegacyCanonicalApplication::PreCompositeLayerAdmission
        };
        let master = EvaluatedMasterScopePlan {
            execution: compile_scope_execution(
                self.input.master_rack,
                LegacyRackScope::Master,
                VisualScopeId::Master,
                ScopeHostPayload {
                    materialize: master_materialize,
                    canonical: Some(master_canonical),
                    temporal: Some(*self.base.temporal()),
                    canonical_application,
                    allow_exact_scope: false,
                },
                output,
            )?,
            canonical_layers: canonical_layers.into_boxed_slice(),
            canonical_bypass_layers: canonical_bypass_layers.into_boxed_slice(),
            selective_ntsc_layers: selective_ntsc_layers.into_boxed_slice(),
            selective_ntsc_bypass_layers: selective_ntsc_bypass_layers.into_boxed_slice(),
        };

        let below = below_topology(self.input.composition)?;
        let mut taps = Vec::new();
        let mut diagnostics = Vec::new();
        let mut dependencies = Vec::new();
        let mut route_edges = BTreeSet::new();
        let mut prefix_constraints = BTreeMap::new();
        let known_scopes: BTreeSet<_> = std::iter::once(VisualScopeId::Master)
            .chain(flat_ids.iter().copied().map(VisualScopeId::Layer))
            .chain(
                group_plans
                    .iter()
                    .map(|group| VisualScopeId::Group(group.id)),
            )
            .chain(std::iter::once(VisualScopeId::Program))
            .collect();

        for layer in &layer_plans {
            collect_rack_taps(
                VisualScopeId::Layer(layer.stable_id),
                self.racks[&layer.stable_id],
                &below,
                &known_scopes,
                &mut taps,
                &mut diagnostics,
                &mut dependencies,
                &mut route_edges,
                &mut prefix_constraints,
            )?;
            if let Some(matte) = layer.legacy_matte {
                collect_tap(
                    ImageTapConsumer::LayerMatte {
                        layer_id: layer.stable_id,
                    },
                    PlannedImageTapOrigin::LegacyLayerMatte {
                        input: matte.authored.input,
                        diagnostic: matte.diagnostic,
                    },
                    &below,
                    &known_scopes,
                    &mut taps,
                    &mut diagnostics,
                    &mut dependencies,
                    &mut route_edges,
                    &mut prefix_constraints,
                )?;
            }
        }
        for group in &group_plans {
            let runtime =
                self.input
                    .composition
                    .group(group.id)
                    .ok_or(CompositionPlanError::Internal(
                        "planned group lost its runtime value",
                    ))?;
            if !group.bypass {
                collect_rack_taps(
                    VisualScopeId::Group(group.id),
                    &runtime.rack,
                    &below,
                    &known_scopes,
                    &mut taps,
                    &mut diagnostics,
                    &mut dependencies,
                    &mut route_edges,
                    &mut prefix_constraints,
                )?;
                if let Some(matte) = runtime.matte.filter(|matte| matte.amount > 0.0) {
                    collect_tap(
                        ImageTapConsumer::GroupMatte { group_id: group.id },
                        PlannedImageTapOrigin::Rack(matte.tap),
                        &below,
                        &known_scopes,
                        &mut taps,
                        &mut diagnostics,
                        &mut dependencies,
                        &mut route_edges,
                        &mut prefix_constraints,
                    )?;
                }
            }
        }
        collect_rack_taps(
            VisualScopeId::Master,
            self.input.master_rack,
            &below,
            &known_scopes,
            &mut taps,
            &mut diagnostics,
            &mut dependencies,
            &mut route_edges,
            &mut prefix_constraints,
        )?;

        // The logical image graph is sparse: unused direct-root layers remain
        // dynamically supported and never consume the bounded graph address
        // space. Actual execution order below still includes every real scope.
        let mut sparse_scopes = BTreeSet::new();
        for dependency in &dependencies {
            sparse_scopes.insert(dependency.consumer);
            if dependency.producer != VisualScopeId::Program {
                sparse_scopes.insert(dependency.producer);
            }
        }
        let scopes: Vec<_> = sparse_scopes.into_iter().collect();
        let graph =
            ImageDependencyGraph::validate(&scopes, &dependencies, ImageGraphMode::Advanced)
                .map_err(CompositionPlanError::ImageGraph)?;

        let execution_order = execution_order(
            &known_scopes,
            self.input.composition,
            &below,
            &route_edges,
            &prefix_constraints,
        )?;
        let resources = resource_preflight(
            output,
            self.input,
            &layer_plans,
            &group_plans,
            &master,
            &graph,
            &taps,
            &motion,
        )?;
        let topology_signature = advanced_topology_signature(
            self.input.composition,
            self.input.master_rack,
            &self.racks,
            &layer_plans,
            &group_plans,
            &taps,
            &motion,
        );

        Ok(EvaluatedCompositionPlan::Advanced(Box::new(
            AdvancedCompositionPlan {
                base: self.base.clone_without_image_routing(),
                master,
                layers: layer_plans.into_boxed_slice(),
                groups: group_plans.into_boxed_slice(),
                root: self.input.composition.root().to_vec().into_boxed_slice(),
                bus_crossfade: self.input.composition.bus_crossfade(),
                image_taps: taps.into_boxed_slice(),
                diagnostics: diagnostics.into_boxed_slice(),
                below,
                graph,
                execution_order: execution_order.into_boxed_slice(),
                resources,
                motion,
                topology_signature,
            },
        )))
    }

    fn evaluate_motion(
        &self,
        flat_ids: &[StableLayerId],
    ) -> Result<EvaluatedMotionPlan, CompositionPlanError> {
        let Some(input) = self.input.motion else {
            return Ok(EvaluatedMotionPlan::LegacyExact);
        };
        let mut supplied = BTreeMap::new();
        for layer in input.layers {
            if supplied.insert(layer.stable_id, *layer).is_some() {
                return Err(CompositionPlanError::DuplicateLayerMotion(layer.stable_id));
            }
            if !self.base_index.contains_key(&layer.stable_id) {
                return Err(CompositionPlanError::UnknownLayerMotion(layer.stable_id));
            }
        }
        if supplied.len() != self.base_ids.len() {
            let missing = self
                .base_ids
                .iter()
                .find(|id| !supplied.contains_key(id))
                .copied()
                .ok_or(CompositionPlanError::Internal(
                    "motion layer count diverged without a missing stable ID",
                ))?;
            return Err(CompositionPlanError::MissingLayerMotion(missing));
        }

        let output = self.base.context().output_size;
        let master_params = input.master.sanitized();
        let mut scopes = Vec::with_capacity(flat_ids.len().saturating_add(1));
        scopes.push(EvaluatedMotionScopePlan {
            scope: VisualScopeId::Master,
            params: master_params,
            source_dimensions: output,
            transform: *self.base.master_transform(),
            spatial: self.base.master_pass().spatial,
            source: resolve_motion_source(master_params.field_source, true, false),
            field_slot: None,
            required_as_donor: false,
            donor_scope: None,
            donor_field_slot: None,
            transplant_admitted: false,
            codec: MotionCodecFrameFacts::default(),
        });
        for id in flat_ids {
            let authored = supplied[id];
            let params = authored.params.sanitized();
            let base_layer = &self.base.layers()[self.base_index[id]];
            scopes.push(EvaluatedMotionScopePlan {
                scope: VisualScopeId::Layer(*id),
                params,
                source_dimensions: base_layer.source.size,
                transform: base_layer.transform,
                spatial: base_layer.spatial,
                source: resolve_motion_source(params.field_source, false, authored.codec.available),
                field_slot: None,
                required_as_donor: false,
                donor_scope: None,
                donor_field_slot: None,
                transplant_admitted: false,
                codec: authored.codec,
            });
        }
        if scopes.iter().all(|scope| scope.params.is_exact_zero()) {
            return Ok(EvaluatedMotionPlan::LegacyExact);
        }

        let mut diagnostics = Vec::new();
        if master_params.transplant.amount > 0.0 {
            diagnostics.push(MotionPlanDiagnostic::MasterTransplantRejected);
        }
        let mut admitted_recipient = None;
        for id in flat_ids {
            let recipient_index = scopes
                .iter()
                .position(|scope| scope.scope == VisualScopeId::Layer(*id))
                .ok_or(CompositionPlanError::Internal(
                    "motion recipient disappeared from compact scope plan",
                ))?;
            if scopes[recipient_index].params.transplant.amount <= 0.0 {
                continue;
            }
            let donor = scopes[recipient_index].params.transplant.donor;
            let donor_id = match donor {
                MotionDonor::None => {
                    diagnostics.push(MotionPlanDiagnostic::DonorNotSelected { recipient: *id });
                    continue;
                }
                MotionDonor::Missing { saved_position } => {
                    diagnostics.push(MotionPlanDiagnostic::MissingDonor {
                        recipient: *id,
                        saved_position,
                    });
                    continue;
                }
                MotionDonor::Selected { layer_id, .. }
                    if !self.base_index.contains_key(&layer_id) =>
                {
                    let saved_position = match donor {
                        MotionDonor::Selected { saved_position, .. } => saved_position,
                        _ => unreachable!(),
                    };
                    diagnostics.push(MotionPlanDiagnostic::MissingDonor {
                        recipient: *id,
                        saved_position,
                    });
                    continue;
                }
                MotionDonor::Selected { layer_id, .. } => layer_id,
            };
            if let Some(admitted) = admitted_recipient {
                diagnostics.push(MotionPlanDiagnostic::ExcessTransplantRejected {
                    recipient: *id,
                    admitted_recipient: admitted,
                });
                continue;
            }
            let donor_index = scopes
                .iter()
                .position(|scope| scope.scope == VisualScopeId::Layer(donor_id))
                .ok_or(CompositionPlanError::Internal(
                    "resolved Faraday donor disappeared from compact scope plan",
                ))?;
            admitted_recipient = Some(*id);
            scopes[recipient_index].transplant_admitted = true;
            scopes[recipient_index].donor_scope = Some(VisualScopeId::Layer(donor_id));
            scopes[donor_index].required_as_donor = true;
        }

        let mut fields = Vec::new();
        for scope in &mut scopes {
            let field_required = scope.required_as_donor || !scope.params.shutter.is_exact_zero();
            if !field_required {
                continue;
            }
            let grid =
                MotionGrid::for_source(scope.source_dimensions, scope.params.lattice_quality)
                    .map_err(CompositionPlanError::Motion)?;
            let slot = u8::try_from(fields.len())
                .map_err(|_| CompositionPlanError::Internal("motion field slot exceeds u8"))?;
            scope.field_slot = Some(slot);
            if scope.source.diagnostic != MotionSourceDiagnostic::None {
                diagnostics.push(MotionPlanDiagnostic::Source {
                    scope: scope.scope,
                    diagnostic: scope.source.diagnostic,
                });
            }
            fields.push(EvaluatedMotionFieldPlan {
                slot,
                scope: scope.scope,
                source_dimensions: scope.source_dimensions,
                grid,
                algorithm_version: scope.params.algorithm_version,
                source: scope.source,
                codec: scope.codec,
                required_as_donor: scope.required_as_donor,
            });
        }
        for scope in &mut scopes {
            scope.donor_field_slot = scope
                .donor_scope
                .and_then(|donor| fields.iter().find(|field| field.scope == donor))
                .map(|field| field.slot);
        }

        let requests = scopes
            .iter()
            .map(|scope| {
                let mut params = scope.params;
                if !scope.transplant_admitted {
                    params.transplant.amount = 0.0;
                }
                MotionScopeResourceRequest {
                    source_dimensions: scope.source_dimensions,
                    output_dimensions: output,
                    params,
                    is_master: scope.scope == VisualScopeId::Master,
                    codec_vectors_available: scope.codec.available,
                    required_as_donor: scope.required_as_donor,
                }
            })
            .collect::<Vec<_>>();
        let resources = MotionResourcePlan::preflight(&requests, input.limits)
            .map_err(CompositionPlanError::Motion)?;
        let budget = motion_frame_budget(&scopes)?;
        let topology_signature = motion_topology_signature(&scopes, &fields, resources);
        Ok(EvaluatedMotionPlan::Advanced(Box::new(
            AdvancedMotionPlan {
                scopes: scopes.into_boxed_slice(),
                fields: fields.into_boxed_slice(),
                resources,
                diagnostics: diagnostics.into_boxed_slice(),
                budget,
                topology_signature,
            },
        )))
    }

    fn is_global_legacy_exact(&self, flat_ids: &[StableLayerId]) -> bool {
        !self.base.image_routing().is_active()
            && !self.input.layer_mattes.is_some_and(|mattes| {
                mattes.iter().any(|matte| {
                    let matte = matte.sanitized();
                    matte.enabled && matte.amount > 0.0
                })
            })
            && self.input.composition.groups().len() == 0
            && self.input.composition.bus_crossfade() == 0.5
            && flat_ids
                .iter()
                .copied()
                .eq(self.base_ids.iter().rev().copied())
            && self.input.composition.root().iter().all(|item| {
                matches!(
                    item,
                    RuntimeRootItem::Layer {
                        bus: BusAssignment::Program,
                        ..
                    }
                )
            })
            && self
                .input
                .master_rack
                .is_exact_legacy(LegacyRackScope::Master)
            && self
                .base_ids
                .iter()
                .all(|id| self.racks[id].is_exact_legacy(LegacyRackScope::Layer))
    }
}

#[derive(Clone, Copy)]
struct ScopeHostPayload {
    materialize: EffectPassUniforms,
    canonical: Option<EffectPassUniforms>,
    temporal: Option<TemporalParams>,
    canonical_application: LegacyCanonicalApplication,
    allow_exact_scope: bool,
}

fn compile_scope_execution(
    rack: &RuntimeVisualRack,
    legacy_scope: LegacyRackScope,
    scope: VisualScopeId,
    host: ScopeHostPayload,
    output: [u32; 2],
) -> Result<EvaluatedScopeExecution, CompositionPlanError> {
    rack.validate_for_scope(legacy_scope)
        .map_err(|error| CompositionPlanError::Rack { scope, error })?;
    if host.allow_exact_scope
        && legacy_scope != LegacyRackScope::Group
        && rack.is_exact_legacy(legacy_scope)
    {
        return Ok(EvaluatedScopeExecution::ExactLegacy {
            scope: legacy_scope,
        });
    }

    let mut steps = vec![EvaluatedScopeStep::MaterializeSpatial {
        pass: host.materialize,
        application: host.canonical_application,
    }];
    let mut pending = Vec::new();
    let mut segment_index = 0_u8;
    let mut first_global_node = None;
    for node in rack.iter().copied() {
        match node.kind {
            RuntimeVisualNodeKind::LegacyCanonical => {
                flush_segment(&mut pending, &mut steps, &mut segment_index, scope, output)?;
                let application =
                    if legacy_scope == LegacyRackScope::Master && first_global_node.is_some() {
                        LegacyCanonicalApplication::ScopeLocal
                    } else {
                        host.canonical_application
                    };
                steps.push(EvaluatedScopeStep::LegacyCanonical {
                    pass: host.canonical.ok_or(CompositionPlanError::Internal(
                        "legacy canonical marker has no host payload",
                    ))?,
                    application,
                });
            }
            RuntimeVisualNodeKind::LegacyTemporal => {
                first_global_node.get_or_insert(node.stable_id);
                flush_segment(&mut pending, &mut steps, &mut segment_index, scope, output)?;
                steps.push(EvaluatedScopeStep::LegacyTemporal {
                    params: host.temporal.ok_or(CompositionPlanError::Internal(
                        "legacy temporal marker has no master payload",
                    ))?,
                });
            }
            _ => {
                first_global_node.get_or_insert(node.stable_id);
                pending.push(node);
            }
        }
    }
    flush_segment(&mut pending, &mut steps, &mut segment_index, scope, output)?;
    Ok(EvaluatedScopeExecution::Ordered {
        steps: steps.into_boxed_slice(),
    })
}

fn flush_segment(
    pending: &mut Vec<RuntimeVisualNode>,
    steps: &mut Vec<EvaluatedScopeStep>,
    segment_index: &mut u8,
    scope: VisualScopeId,
    output: [u32; 2],
) -> Result<(), CompositionPlanError> {
    if pending.is_empty() {
        return Ok(());
    }
    let mut ordinary = Vec::new();
    for node in std::mem::take(pending) {
        let is_image_consumer = matches!(
            node.kind,
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(_))
        );
        if is_image_consumer {
            compile_segment_nodes(
                std::mem::take(&mut ordinary),
                steps,
                segment_index,
                scope,
                output,
            )?;
            compile_segment_nodes(vec![node], steps, segment_index, scope, output)?;
        } else {
            ordinary.push(node);
        }
    }
    compile_segment_nodes(ordinary, steps, segment_index, scope, output)
}

fn compile_segment_nodes(
    nodes: Vec<RuntimeVisualNode>,
    steps: &mut Vec<EvaluatedScopeStep>,
    segment_index: &mut u8,
    scope: VisualScopeId,
    output: [u32; 2],
) -> Result<(), CompositionPlanError> {
    if nodes.is_empty() {
        return Ok(());
    }
    let rack = RuntimeVisualRack::try_from_parts(nodes, None)
        .map_err(|error| CompositionPlanError::Rack { scope, error })?;
    let index = *segment_index;
    let plan = CollisionRackPlan::compile(&rack, output, output).map_err(|error| {
        CompositionPlanError::RackCompile {
            scope,
            segment_index: index,
            error,
        }
    })?;
    steps.push(EvaluatedScopeStep::CollisionRack {
        segment_index: index,
        plan,
    });
    *segment_index = segment_index
        .checked_add(1)
        .ok_or(CompositionPlanError::Internal(
            "rack segment counter overflowed its bounded domain",
        ))?;
    Ok(())
}

fn below_topology(composition: &RuntimeComposition) -> Result<BelowTopology, CompositionPlanError> {
    let mut root_outputs = Vec::with_capacity(composition.root().len().min(256));
    let mut locations = BTreeMap::new();
    let mut group_members = BTreeMap::new();
    for (root_index, item) in composition.root().iter().copied().enumerate() {
        match item {
            RuntimeRootItem::Layer { layer_id, .. } => {
                locations.insert(
                    VisualScopeId::Layer(layer_id),
                    BelowLocation::Root { root_index },
                );
                root_outputs.push(VisualScopeId::Layer(layer_id));
            }
            RuntimeRootItem::Group { group_id } => {
                let group = composition
                    .group(group_id)
                    .ok_or(CompositionPlanError::Internal(
                        "validated root lost its runtime group",
                    ))?;
                let members: Box<[_]> = group.members.iter().collect::<Vec<_>>().into_boxed_slice();
                for (member_index, member) in members.iter().copied().enumerate() {
                    locations.insert(
                        VisualScopeId::Layer(member),
                        BelowLocation::GroupMember {
                            group_id,
                            root_index,
                            member_index,
                        },
                    );
                }
                locations.insert(
                    VisualScopeId::Group(group_id),
                    BelowLocation::Root { root_index },
                );
                group_members.insert(group_id, members);
                root_outputs.push(VisualScopeId::Group(group_id));
            }
        }
    }
    locations.insert(VisualScopeId::Master, BelowLocation::Master);
    Ok(BelowTopology {
        root_outputs: root_outputs.into_boxed_slice(),
        locations,
        group_members,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_rack_taps(
    scope: VisualScopeId,
    rack: &RuntimeVisualRack,
    below: &BelowTopology,
    known_scopes: &BTreeSet<VisualScopeId>,
    taps: &mut Vec<PlannedImageTap>,
    diagnostics: &mut Vec<CompositionPlanDiagnostic>,
    dependencies: &mut Vec<ImageDependency>,
    route_edges: &mut BTreeSet<(VisualScopeId, VisualScopeId)>,
    prefix_constraints: &mut BTreeMap<VisualScopeId, CompositePrefix>,
) -> Result<(), CompositionPlanError> {
    for node in rack.iter() {
        if !node.enabled || node.wet <= 0.0 {
            continue;
        }
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) = node.kind else {
            continue;
        };
        if matte.amount <= 0.0 {
            continue;
        }
        collect_tap(
            ImageTapConsumer::RackNode {
                scope,
                node_id: node.stable_id,
            },
            PlannedImageTapOrigin::Rack(matte.tap),
            below,
            known_scopes,
            taps,
            diagnostics,
            dependencies,
            route_edges,
            prefix_constraints,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_tap(
    consumer: ImageTapConsumer,
    origin: PlannedImageTapOrigin,
    below: &BelowTopology,
    known_scopes: &BTreeSet<VisualScopeId>,
    taps: &mut Vec<PlannedImageTap>,
    diagnostics: &mut Vec<CompositionPlanDiagnostic>,
    dependencies: &mut Vec<ImageDependency>,
    route_edges: &mut BTreeSet<(VisualScopeId, VisualScopeId)>,
    prefix_constraints: &mut BTreeMap<VisualScopeId, CompositePrefix>,
) -> Result<(), CompositionPlanError> {
    let planned_binding_limit = MAX_TEXTURE_SAMPLES_PER_FRAME as usize;
    if taps.len() == planned_binding_limit {
        return Err(CompositionPlanError::TooManyImageRoutes {
            count: taps.len() + 1,
            limit: planned_binding_limit,
        });
    }
    let consumer_scope = consumer.scope();
    let mut dependency_producer = None;
    let mut current_edge = None;
    let timing = origin.timing();
    let source = match origin {
        PlannedImageTapOrigin::Rack(tap) => TapRouteSource::from(tap.source),
        PlannedImageTapOrigin::LegacyLayerMatte { input, diagnostic } => {
            if input == ImageInput::ProgramHistory
                && diagnostic
                    == ImageRouteDiagnostic::Missing(MissingImageInput::ProgramHistoryUninitialized)
            {
                TapRouteSource::ProgramHistoryUninitialized
            } else {
                TapRouteSource::from(input)
            }
        }
    };
    let resolved = match source {
        TapRouteSource::SelectedLayer { layer_id, stage } => {
            let producer = VisualScopeId::Layer(layer_id);
            if known_scopes.contains(&producer) {
                if timing == EdgeTiming::PreviousFrame || stage == LayerImageStage::PostLocalEffects
                {
                    dependency_producer = Some(producer);
                    current_edge = Some(producer);
                }
                PlannedImageSource::SelectedLayer { layer_id, stage }
            } else {
                diagnostics
                    .push(CompositionPlanDiagnostic::MissingStableScope { consumer, producer });
                PlannedImageSource::Transparent
            }
        }
        TapRouteSource::MissingSelectedLayer { saved_position } => {
            diagnostics.push(CompositionPlanDiagnostic::MissingSelectedLayer {
                consumer,
                saved_position,
            });
            PlannedImageSource::Transparent
        }
        TapRouteSource::GroupOutput(group_id) => {
            let producer = VisualScopeId::Group(group_id);
            if known_scopes.contains(&producer) {
                dependency_producer = Some(producer);
                current_edge = Some(producer);
                PlannedImageSource::Scope(producer)
            } else {
                diagnostics
                    .push(CompositionPlanDiagnostic::MissingStableScope { consumer, producer });
                PlannedImageSource::Transparent
            }
        }
        TapRouteSource::MissingGroupOutput(group_id) => {
            diagnostics.push(CompositionPlanDiagnostic::MissingGroupOutput { consumer, group_id });
            PlannedImageSource::Transparent
        }
        TapRouteSource::OneBelow => {
            if let Some(producer) = below.one_below(consumer_scope) {
                dependency_producer = Some(producer);
                current_edge = Some(producer);
                PlannedImageSource::OneBelow(producer)
            } else {
                diagnostics.push(CompositionPlanDiagnostic::NoOneBelow { consumer });
                PlannedImageSource::Transparent
            }
        }
        TapRouteSource::AllBelow => {
            let prefix = below
                .prefix_for(consumer_scope)
                .ok_or(CompositionPlanError::Internal(
                    "image consumer has no below domain",
                ))?;
            if let Some(producer) = below.last_in_prefix(prefix) {
                dependency_producer = Some(producer);
                if timing == EdgeTiming::CurrentFrame
                    && prefix_constraints
                        .insert(consumer_scope, prefix)
                        .is_some_and(|prior| prior != prefix)
                {
                    return Err(CompositionPlanError::Internal(
                        "one scope resolved conflicting AllBelow domains",
                    ));
                }
                PlannedImageSource::AllBelow(prefix)
            } else {
                diagnostics.push(CompositionPlanDiagnostic::NoAllBelow { consumer });
                PlannedImageSource::Transparent
            }
        }
        TapRouteSource::ProgramHistory => {
            dependency_producer = Some(VisualScopeId::Program);
            PlannedImageSource::ProgramHistory
        }
        TapRouteSource::ProgramHistoryUninitialized => {
            dependency_producer = Some(VisualScopeId::Program);
            diagnostics.push(CompositionPlanDiagnostic::ProgramHistoryUninitialized { consumer });
            // Readiness is a frame diagnostic/uniform bit, not topology. Keep
            // the stable ProgramHistory binding so warming it never forces a
            // prepare/reallocation loop; the matte's donor_valid=false makes
            // the cold sample defined transparent.
            PlannedImageSource::ProgramHistory
        }
        TapRouteSource::CleanProgram
            if matches!(origin, PlannedImageTapOrigin::LegacyLayerMatte { .. }) =>
        {
            diagnostics.push(CompositionPlanDiagnostic::LegacyCleanProgramTransparent { consumer });
            PlannedImageSource::Transparent
        }
        TapRouteSource::CleanProgram if timing == EdgeTiming::PreviousFrame => {
            dependency_producer = Some(VisualScopeId::Program);
            PlannedImageSource::ProgramHistory
        }
        TapRouteSource::CleanProgram => {
            dependency_producer = Some(VisualScopeId::Program);
            current_edge = Some(VisualScopeId::Program);
            PlannedImageSource::Scope(VisualScopeId::Program)
        }
    };
    if let Some(producer) = dependency_producer {
        dependencies.push(ImageDependency {
            consumer: consumer_scope,
            producer,
            timing,
        });
        if timing == EdgeTiming::CurrentFrame {
            if let Some(producer) = current_edge {
                route_edges.insert((producer, consumer_scope));
            }
        }
    }
    taps.push(PlannedImageTap {
        consumer,
        origin,
        resolved,
    });
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum TapRouteSource {
    SelectedLayer {
        layer_id: StableLayerId,
        stage: LayerImageStage,
    },
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
    },
    OneBelow,
    AllBelow,
    GroupOutput(GroupId),
    MissingGroupOutput(GroupId),
    CleanProgram,
    ProgramHistory,
    ProgramHistoryUninitialized,
}

impl From<ResolvedImageSource> for TapRouteSource {
    fn from(source: ResolvedImageSource) -> Self {
        match source {
            ResolvedImageSource::SelectedLayer {
                layer_id, stage, ..
            } => Self::SelectedLayer { layer_id, stage },
            ResolvedImageSource::MissingSelectedLayer { saved_position, .. } => {
                Self::MissingSelectedLayer { saved_position }
            }
            ResolvedImageSource::OneBelow => Self::OneBelow,
            ResolvedImageSource::AllBelow => Self::AllBelow,
            ResolvedImageSource::GroupOutput(group_id) => Self::GroupOutput(group_id),
            ResolvedImageSource::MissingGroupOutput(group_id) => Self::MissingGroupOutput(group_id),
            ResolvedImageSource::CleanProgram => Self::CleanProgram,
        }
    }
}

impl From<ImageInput> for TapRouteSource {
    fn from(input: ImageInput) -> Self {
        match input {
            ImageInput::SelectedLayer { layer_id, stage } => {
                Self::SelectedLayer { layer_id, stage }
            }
            ImageInput::MissingSelectedLayer { saved_position, .. } => {
                Self::MissingSelectedLayer { saved_position }
            }
            ImageInput::OneBelow => Self::OneBelow,
            ImageInput::AllBelow => Self::AllBelow,
            ImageInput::CleanProgram => Self::CleanProgram,
            ImageInput::ProgramHistory => Self::ProgramHistory,
            ImageInput::GroupOutput { group_id } => Self::GroupOutput(group_id),
            ImageInput::MissingGroupOutput { group_id } => Self::MissingGroupOutput(group_id),
        }
    }
}

fn execution_order(
    known_scopes: &BTreeSet<VisualScopeId>,
    composition: &RuntimeComposition,
    below: &BelowTopology,
    route_edges: &BTreeSet<(VisualScopeId, VisualScopeId)>,
    prefix_constraints: &BTreeMap<VisualScopeId, CompositePrefix>,
) -> Result<Vec<VisualScopeId>, CompositionPlanError> {
    let mut edges = route_edges.clone();
    for group in composition.groups() {
        for member in group.members.iter() {
            // A group output consumes every member output. A member which
            // current-taps its own group therefore forms a real cycle.
            edges.insert((VisualScopeId::Layer(member), VisualScopeId::Group(group.id)));
        }
    }
    for output in below.root_outputs() {
        edges.insert((*output, VisualScopeId::Master));
    }
    edges.insert((VisualScopeId::Master, VisualScopeId::Program));

    let mut adjacency: BTreeMap<VisualScopeId, BTreeSet<VisualScopeId>> = BTreeMap::new();
    let mut indegree: BTreeMap<_, usize> = known_scopes
        .iter()
        .copied()
        .map(|scope| (scope, 0))
        .collect();
    for &(producer, consumer) in &edges {
        if !known_scopes.contains(&producer) || !known_scopes.contains(&consumer) {
            continue;
        }
        if adjacency.entry(producer).or_default().insert(consumer)
            && !prefix_constraints
                .get(&consumer)
                .is_some_and(|prefix| below.prefix_contains(*prefix, producer))
        {
            *indegree
                .get_mut(&consumer)
                .expect("known consumer has indegree") += 1;
        }
    }
    for (consumer, prefix) in prefix_constraints {
        if !known_scopes.contains(consumer) {
            continue;
        }
        *indegree
            .get_mut(consumer)
            .expect("known prefix consumer has indegree") += below.prefix_len(*prefix);
    }
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .filter_map(|(scope, degree)| (*degree == 0).then_some(*scope))
        .collect();
    let mut order = Vec::with_capacity(known_scopes.len());
    while let Some(scope) = ready.pop_first() {
        order.push(scope);
        if let Some(consumers) = adjacency.get(&scope) {
            for consumer in consumers {
                if prefix_constraints
                    .get(consumer)
                    .is_some_and(|prefix| below.prefix_contains(*prefix, scope))
                {
                    continue;
                }
                decrement_indegree(*consumer, &mut indegree, &mut ready);
            }
        }
        for (consumer, prefix) in prefix_constraints {
            if below.prefix_contains(*prefix, scope) {
                decrement_indegree(*consumer, &mut indegree, &mut ready);
            }
        }
    }
    if order.len() != known_scopes.len() {
        return Err(CompositionPlanError::CurrentCycle {
            scopes: indegree
                .into_iter()
                .filter_map(|(scope, degree)| (degree != 0).then_some(scope))
                .collect(),
        });
    }
    atomic_group_execution_order(
        &order,
        known_scopes,
        composition,
        below,
        &edges,
        prefix_constraints,
    )
}

fn layer_effectively_contributes(
    layer: &EvaluatedLayerScopePlan,
    evaluated: &crate::evaluated_frame::EvaluatedLayer,
    groups: &[EvaluatedGroupScopePlan],
    bus_crossfade: f32,
) -> bool {
    if !evaluated.visible
        || !layer.admitted_to_program
        || !evaluated.opacity.is_finite()
        || evaluated.opacity <= 0.0
    {
        return false;
    }
    if let Some(group_id) = layer.group_id {
        let Some(group) = groups.iter().find(|group| group.id == group_id) else {
            return false;
        };
        if !group.opacity.is_finite() || group.opacity <= 0.0 {
            return false;
        }
    }
    let crossfade = if bus_crossfade.is_finite() {
        bus_crossfade.clamp(0.0, 1.0)
    } else {
        0.5
    };
    match layer.bus {
        BusAssignment::A => crossfade < 1.0,
        BusAssignment::B => crossfade > 0.0,
        BusAssignment::Program => true,
    }
}

/// Collapse each group and all of its members into one scheduling task. The
/// compositor owns a single group accumulator, so allowing the global scope
/// order to weave between two groups would require unbudgeted retained group
/// accumulators. Scope-level DAG validation above still governs the order
/// *inside* one group; this second DAG proves that those scopes can be emitted
/// as one contiguous block.
fn atomic_group_execution_order(
    scope_order: &[VisualScopeId],
    known_scopes: &BTreeSet<VisualScopeId>,
    composition: &RuntimeComposition,
    below: &BelowTopology,
    edges: &BTreeSet<(VisualScopeId, VisualScopeId)>,
    prefix_constraints: &BTreeMap<VisualScopeId, CompositePrefix>,
) -> Result<Vec<VisualScopeId>, CompositionPlanError> {
    let member_groups: BTreeMap<_, _> = composition
        .groups()
        .flat_map(|group| group.members.iter().map(move |member| (member, group.id)))
        .collect();
    let task_for = |scope: VisualScopeId| match scope {
        VisualScopeId::Layer(layer_id) => member_groups
            .get(&layer_id)
            .copied()
            .map(VisualScopeId::Group)
            .unwrap_or(scope),
        _ => scope,
    };

    let tasks: BTreeSet<_> = known_scopes.iter().copied().map(task_for).collect();
    let mut task_edges = BTreeSet::new();
    for &(producer, consumer) in edges {
        let producer = task_for(producer);
        let consumer = task_for(consumer);
        if producer != consumer {
            task_edges.insert((producer, consumer));
        }
    }
    for (&consumer, &prefix) in prefix_constraints {
        let consumer = task_for(consumer);
        for producer in known_scopes.iter().copied() {
            if below.prefix_contains(prefix, producer) {
                let producer = task_for(producer);
                if producer != consumer {
                    task_edges.insert((producer, consumer));
                }
            }
        }
    }

    let mut adjacency: BTreeMap<VisualScopeId, BTreeSet<VisualScopeId>> = BTreeMap::new();
    let mut indegree: BTreeMap<_, usize> = tasks.iter().copied().map(|task| (task, 0)).collect();
    for (producer, consumer) in task_edges {
        if adjacency.entry(producer).or_default().insert(consumer) {
            *indegree
                .get_mut(&consumer)
                .expect("collapsed group consumer is a known task") += 1;
        }
    }
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .filter_map(|(task, degree)| (*degree == 0).then_some(*task))
        .collect();
    let mut task_order = Vec::with_capacity(tasks.len());
    while let Some(task) = ready.pop_first() {
        task_order.push(task);
        if let Some(consumers) = adjacency.get(&task) {
            for consumer in consumers {
                decrement_indegree(*consumer, &mut indegree, &mut ready);
            }
        }
    }
    if task_order.len() != tasks.len() {
        return Err(CompositionPlanError::AtomicGroupCycle {
            tasks: indegree
                .into_iter()
                .filter_map(|(task, degree)| (degree != 0).then_some(task))
                .collect(),
        });
    }

    let mut order = Vec::with_capacity(scope_order.len());
    for task in task_order {
        if matches!(task, VisualScopeId::Group(_)) {
            order.extend(
                scope_order
                    .iter()
                    .copied()
                    .filter(|scope| task_for(*scope) == task),
            );
        } else {
            order.push(task);
        }
    }
    debug_assert_eq!(order.len(), scope_order.len());
    Ok(order)
}

fn decrement_indegree(
    consumer: VisualScopeId,
    indegree: &mut BTreeMap<VisualScopeId, usize>,
    ready: &mut BTreeSet<VisualScopeId>,
) {
    let degree = indegree
        .get_mut(&consumer)
        .expect("known scope has indegree");
    debug_assert!(*degree > 0);
    *degree = degree.saturating_sub(1);
    if *degree == 0 {
        ready.insert(consumer);
    }
}

/// Preserve authored matte intent for the unified graph without invoking the
/// positional M1 resolver or allocating its independent GPU ledger. Producer
/// availability and below domains are resolved later against RuntimeComposition.
fn composition_layer_matte(
    matte: LayerMatte,
    program_history_initialized: bool,
) -> EvaluatedLayerMatte {
    let authored = matte.sanitized();
    let diagnostic = if !authored.enabled {
        ImageRouteDiagnostic::Disabled
    } else {
        match authored.input {
            ImageInput::MissingSelectedLayer { saved_position, .. } => {
                ImageRouteDiagnostic::Missing(MissingImageInput::SelectedLayer(saved_position))
            }
            ImageInput::MissingGroupOutput { group_id } => {
                ImageRouteDiagnostic::Missing(MissingImageInput::GroupOutputUnavailable(group_id))
            }
            ImageInput::ProgramHistory if !program_history_initialized => {
                ImageRouteDiagnostic::Missing(MissingImageInput::ProgramHistoryUninitialized)
            }
            ImageInput::CleanProgram => {
                ImageRouteDiagnostic::Cycle(ImageRouteCycle::CleanProgramSameFrame)
            }
            ImageInput::SelectedLayer { .. }
            | ImageInput::OneBelow
            | ImageInput::AllBelow
            | ImageInput::ProgramHistory
            | ImageInput::GroupOutput { .. } => ImageRouteDiagnostic::Ready,
        }
    };
    EvaluatedLayerMatte {
        authored,
        resolved_input: if authored.enabled {
            super::ResolvedImageInput::Transparent
        } else {
            super::ResolvedImageInput::Disabled
        },
        params: ResolvedMatteParams {
            channel: match authored.channel {
                crate::image_routing::MatteChannel::Alpha => MatteChannelCode::Alpha,
                crate::image_routing::MatteChannel::Luma => MatteChannelCode::Luma,
                crate::image_routing::MatteChannel::Red => MatteChannelCode::Red,
                crate::image_routing::MatteChannel::Green => MatteChannelCode::Green,
                crate::image_routing::MatteChannel::Blue => MatteChannelCode::Blue,
            },
            invert: authored.invert,
            amount: authored.amount,
            threshold: authored.threshold,
            softness: authored.softness,
            donor_valid: diagnostic == ImageRouteDiagnostic::Ready,
        },
        diagnostic,
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "one immutable preflight boundary validates all independent creative ledgers"
)]
fn resource_preflight(
    output: [u32; 2],
    input: CompositionPlanInput<'_>,
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    master: &EvaluatedMasterScopePlan,
    graph: &ImageGraphPlan,
    taps: &[PlannedImageTap],
    motion: &EvaluatedMotionPlan,
) -> Result<CreativeResourcePlan, CompositionPlanError> {
    let mut racks = Vec::with_capacity(
        input
            .layer_racks
            .len()
            .min(MAX_IMAGE_DEPENDENCIES)
            .saturating_add(groups.len().saturating_mul(3))
            .saturating_add(2),
    );
    for layer in layers {
        let rack = input
            .layer_racks
            .iter()
            .find_map(|(id, rack)| (*id == layer.stable_id).then_some(rack))
            .ok_or(CompositionPlanError::MissingLayerRack(layer.stable_id))?;
        if !layer.execution.is_exact_legacy() {
            racks.push(capture_budget_rack(
                rack,
                VisualScopeId::Layer(layer.stable_id),
            )?);
            racks.push(synthetic_transform_budget_rack()?);
        }
        if let Some(matte) = layer.legacy_matte {
            racks.push(synthetic_legacy_matte_budget_rack(matte)?);
        }
    }
    for group in groups {
        let runtime = input
            .composition
            .group(group.id)
            .ok_or(CompositionPlanError::Internal(
                "resource preflight lost a validated group",
            ))?;
        if !group.bypass {
            racks.push(capture_budget_rack(
                &runtime.rack,
                VisualScopeId::Group(group.id),
            )?);
            racks.push(synthetic_transform_budget_rack()?);
            if let Some(matte) = runtime.matte {
                racks.push(synthetic_matte_budget_rack(matte)?);
            }
        }
    }
    if !master.execution.is_exact_legacy() {
        racks.push(capture_budget_rack(
            input.master_rack,
            VisualScopeId::Master,
        )?);
        racks.push(synthetic_transform_budget_rack()?);
    }
    let previous_scope_staging = taps.iter().try_fold(0_u32, |count, tap| {
        let needs_staging = tap.origin.timing() == EdgeTiming::PreviousFrame
            && !matches!(
                tap.resolved,
                PlannedImageSource::ProgramHistory | PlannedImageSource::Transparent
            );
        count
            .checked_add(u32::from(needs_staging))
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))
    })?;
    let program_history_staging = u32::from(
        taps.iter()
            .any(|tap| matches!(tap.resolved, PlannedImageSource::ProgramHistory)),
    )
    .saturating_mul(ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS);
    let current_prelocal_donors = taps
        .iter()
        .filter_map(|tap| {
            (tap.origin.timing() == EdgeTiming::CurrentFrame)
                .then_some(&tap.resolved)
                .and_then(|resolved| match resolved {
                    PlannedImageSource::SelectedLayer {
                        layer_id,
                        stage: LayerImageStage::PreLocalEffects,
                    } => Some(*layer_id),
                    _ => None,
                })
        })
        .collect::<BTreeSet<_>>();
    let current_prelocal_surfaces = u32::try_from(current_prelocal_donors.len())
        .map_err(|_| CompositionPlanError::Resource(ResourcePreflightError::ArithmeticOverflow))?;
    let additional_rgba16 = ADVANCED_RACK_SURFACE_LAYERS
        .checked_add(previous_scope_staging)
        .and_then(|value| value.checked_add(program_history_staging))
        .and_then(|value| value.checked_add(current_prelocal_surfaces))
        .ok_or(CompositionPlanError::Resource(
            ResourcePreflightError::ArithmeticOverflow,
        ))?;
    let creative = CreativeResourcePlan::preflight_with_surface_formats(
        output,
        &racks,
        graph,
        additional_rgba16,
        ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS,
        input.resource_limits,
    )
    .map_err(CompositionPlanError::Resource)?;
    if let Some(motion) = motion.advanced() {
        let combined_bytes = creative
            .creative_bytes
            .checked_add(motion.resources.total_bytes)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        let byte_limit = input
            .resource_limits
            .max_creative_bytes
            .min(crate::visual_rack::MAX_CREATIVE_GPU_BYTES);
        if combined_bytes > byte_limit {
            return Err(CompositionPlanError::MotionCombinedMemoryBudget {
                bytes: combined_bytes,
                limit: byte_limit,
            });
        }
        let combined_logical_lookups = creative
            .logical_texture_lookups_per_pixel
            .checked_add(motion.budget.logical_texture_lookups_per_pixel)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        if combined_logical_lookups > MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME {
            return Err(CompositionPlanError::Resource(
                ResourcePreflightError::FrameLogicalLookupBudget {
                    lookups: combined_logical_lookups,
                    limit: MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME,
                },
            ));
        }
        let combined_samples = creative
            .texture_samples_per_pixel
            .checked_add(motion.budget.texture_samples_per_pixel)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        if combined_samples > MAX_TEXTURE_SAMPLES_PER_FRAME {
            return Err(CompositionPlanError::MotionCombinedSampleBudget {
                samples: combined_samples,
                limit: MAX_TEXTURE_SAMPLES_PER_FRAME,
            });
        }
        let texture_limit = input
            .resource_limits
            .max_sampled_textures_per_shader_stage
            .min(crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_PASS);
        if motion.budget.max_sampled_textures_in_pass > texture_limit {
            return Err(CompositionPlanError::Resource(
                ResourcePreflightError::SampledTextureLimit {
                    requested: motion.budget.max_sampled_textures_in_pass,
                    limit: texture_limit,
                },
            ));
        }
    }
    Ok(creative)
}

fn synthetic_legacy_matte_budget_rack(
    matte: EvaluatedLayerMatte,
) -> Result<VisualRack, CompositionPlanError> {
    synthetic_matte_budget_rack(RuntimeImageMatte {
        tap: ResolvedImageTap {
            source: ResolvedImageSource::OneBelow,
            timing: EdgeTiming::CurrentFrame,
        },
        channel: match matte.authored.channel {
            crate::image_routing::MatteChannel::Alpha => crate::visual_rack::MatteChannel::Alpha,
            crate::image_routing::MatteChannel::Luma => crate::visual_rack::MatteChannel::Luma,
            crate::image_routing::MatteChannel::Red => crate::visual_rack::MatteChannel::Red,
            crate::image_routing::MatteChannel::Green => crate::visual_rack::MatteChannel::Green,
            crate::image_routing::MatteChannel::Blue => crate::visual_rack::MatteChannel::Blue,
        },
        invert: matte.authored.invert,
        amount: matte.authored.amount,
        threshold: matte.authored.threshold,
        softness: matte.authored.softness,
    })
}

fn capture_budget_rack(
    rack: &RuntimeVisualRack,
    scope: VisualScopeId,
) -> Result<VisualRack, CompositionPlanError> {
    rack.capture_routes(|_| None)
        .map_err(|error| CompositionPlanError::RouteCapture { scope, error })
}

fn synthetic_transform_budget_rack() -> Result<VisualRack, CompositionPlanError> {
    let mut rack = RuntimeVisualRack::empty();
    rack.push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
        .map_err(|error| CompositionPlanError::Rack {
            scope: VisualScopeId::Master,
            error,
        })?;
    capture_budget_rack(&rack, VisualScopeId::Master)
}

fn synthetic_matte_budget_rack(
    matte: RuntimeImageMatte,
) -> Result<VisualRack, CompositionPlanError> {
    let mut rack = RuntimeVisualRack::empty();
    rack.push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)))
        .map_err(|error| CompositionPlanError::Rack {
            scope: VisualScopeId::Master,
            error,
        })?;
    capture_budget_rack(&rack, VisualScopeId::Master)
}

fn legacy_topology_signature(
    flat_ids: &[StableLayerId],
    master: &RuntimeVisualRack,
    racks: &BTreeMap<StableLayerId, &RuntimeVisualRack>,
) -> u64 {
    let mut hash = hash_value(FNV_OFFSET, 0x4c45_4741_4359);
    hash = hash_value(hash, master.topology_signature());
    for id in flat_ids {
        hash = hash_value(hash, id.get());
        hash = hash_value(hash, racks[id].topology_signature());
    }
    hash
}

fn advanced_topology_signature(
    composition: &RuntimeComposition,
    master: &RuntimeVisualRack,
    racks: &BTreeMap<StableLayerId, &RuntimeVisualRack>,
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    taps: &[PlannedImageTap],
    motion: &EvaluatedMotionPlan,
) -> u64 {
    let mut hash = hash_value(FNV_OFFSET, 0x4144_5641_4e43_4544);
    hash = hash_value(hash, master.topology_signature());
    for item in composition.root() {
        match *item {
            RuntimeRootItem::Layer { layer_id, bus } => {
                hash = hash_value(hash, 1);
                hash = hash_value(hash, layer_id.get());
                hash = hash_value(hash, bus_code(bus));
            }
            RuntimeRootItem::Group { group_id } => {
                hash = hash_value(hash, 2);
                hash = hash_value(hash, group_id.get());
            }
        }
    }
    for layer in layers {
        hash = hash_value(hash, layer.stable_id.get());
        hash = hash_value(hash, racks[&layer.stable_id].topology_signature());
        hash = hash_value(hash, layer.group_id.map_or(0, GroupId::get));
        hash = hash_value(hash, bus_code(layer.bus));
        hash = hash_value(hash, u64::from(layer.admitted_to_program));
    }
    for group in groups {
        hash = hash_value(hash, group.id.get());
        hash = hash_value(hash, u64::from(group.solo));
        hash = hash_value(hash, u64::from(group.bypass));
        hash = hash_value(hash, bus_code(group.bus));
        for member in &group.members {
            hash = hash_value(hash, member.get());
        }
        if let Some(runtime) = composition.group(group.id) {
            hash = hash_value(hash, runtime.rack.topology_signature());
            hash = hash_value(hash, u64::from(runtime.matte.is_some()));
        }
    }
    for tap in taps {
        hash = hash_consumer(hash, tap.consumer);
        hash = hash_value(hash, timing_code(tap.origin.timing()));
        hash = hash_source(hash, &tap.resolved);
        hash = hash_value(
            hash,
            match tap.origin {
                PlannedImageTapOrigin::Rack(_) => 1,
                PlannedImageTapOrigin::LegacyLayerMatte { .. } => 2,
            },
        );
        let stage = match tap.origin {
            PlannedImageTapOrigin::Rack(ResolvedImageTap {
                source:
                    ResolvedImageSource::SelectedLayer { stage, .. }
                    | ResolvedImageSource::MissingSelectedLayer { stage, .. },
                ..
            })
            | PlannedImageTapOrigin::LegacyLayerMatte {
                input:
                    ImageInput::SelectedLayer { stage, .. }
                    | ImageInput::MissingSelectedLayer { stage, .. },
                ..
            } => Some(stage),
            _ => None,
        };
        if let Some(stage) = stage {
            hash = hash_value(hash, stage_code(stage));
        }
    }
    if !motion.is_legacy_exact() {
        hash = hash_value(hash, 0x4d4f_5449_4f4e);
        hash = hash_value(hash, motion.topology_signature());
    }
    hash
}

fn motion_frame_budget(
    scopes: &[EvaluatedMotionScopePlan],
) -> Result<MotionFrameBudget, CompositionPlanError> {
    let mut budget = MotionFrameBudget::default();
    for scope in scopes {
        let shutter_active = !scope.params.shutter.is_exact_zero();
        let faraday_active = scope.transplant_admitted;
        if !shutter_active && !faraday_active {
            continue;
        }
        let shutter_samples = if shutter_active {
            u32::from(scope.params.shutter.quality.sample_count())
        } else {
            1
        };
        let carrier_lookups = if shutter_active && scope.params.shutter.chromatic_lag > 0.0 {
            shutter_samples
                .checked_mul(3)
                .ok_or(CompositionPlanError::Resource(
                    ResourcePreflightError::ArithmeticOverflow,
                ))?
        } else {
            shutter_samples
        };
        let carrier_texture_ops = carrier_lookups
            .checked_mul(u32::from(
                crate::visual_rack::PREMULTIPLIED_BILINEAR_TEXTURE_OPS,
            ))
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        budget.full_frame_passes = budget
            .full_frame_passes
            .checked_add(1 + u32::from(faraday_active))
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        budget.logical_texture_lookups_per_pixel = budget
            .logical_texture_lookups_per_pixel
            .checked_add(carrier_lookups + 2 + u32::from(faraday_active) * 4)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        budget.texture_samples_per_pixel = budget
            .texture_samples_per_pixel
            .checked_add(carrier_texture_ops + 2 + u32::from(faraday_active) * 7)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        budget.max_sampled_textures_in_pass = 3;
    }
    Ok(budget)
}

fn motion_topology_signature(
    scopes: &[EvaluatedMotionScopePlan],
    fields: &[EvaluatedMotionFieldPlan],
    resources: MotionResourcePlan,
) -> u64 {
    let mut hash = hash_value(FNV_OFFSET, 0x4d4f_5449_4f4e_0001);
    hash = hash_value(hash, u64::from(resources.active_field_slots));
    hash = hash_value(hash, u64::from(resources.persistent_carriers));
    hash = hash_value(hash, u64::from(resources.max_shutter_samples));
    for scope in scopes {
        hash = hash_scope(hash, scope.scope);
        hash = hash_value(hash, u64::from(scope.params.algorithm_version));
        hash = hash_value(
            hash,
            u64::from(scope.field_slot.map_or(u8::MAX, |slot| slot)),
        );
        hash = hash_value(
            hash,
            u64::from(scope.donor_field_slot.map_or(u8::MAX, |slot| slot)),
        );
        hash = hash_value(hash, u64::from(scope.transplant_admitted));
        hash = hash_value(hash, u64::from(!scope.params.shutter.is_exact_zero()));
        hash = hash_value(hash, u64::from(scope.params.shutter.quality.sample_count()));
    }
    for field in fields {
        hash = hash_value(hash, u64::from(field.slot));
        hash = hash_scope(hash, field.scope);
        hash = hash_value(hash, u64::from(field.grid.width));
        hash = hash_value(hash, u64::from(field.grid.height));
        hash = hash_value(hash, u64::from(field.grid.block_pixels));
        hash = hash_value(hash, u64::from(field.required_as_donor));
        hash = hash_value(
            hash,
            match field.source.origin {
                MotionFieldOrigin::None => 0,
                MotionFieldOrigin::CodecVectors => 1,
                MotionFieldOrigin::Lattice => 2,
                MotionFieldOrigin::LatticeFallback => 3,
            },
        );
    }
    hash
}

fn hash_consumer(mut hash: u64, consumer: ImageTapConsumer) -> u64 {
    match consumer {
        ImageTapConsumer::RackNode { scope, node_id } => {
            hash = hash_value(hash, 1);
            hash = hash_scope(hash, scope);
            hash_value(hash, node_id.get())
        }
        ImageTapConsumer::GroupMatte { group_id } => {
            hash = hash_value(hash, 2);
            hash_value(hash, group_id.get())
        }
        ImageTapConsumer::LayerMatte { layer_id } => {
            hash = hash_value(hash, 3);
            hash_value(hash, layer_id.get())
        }
    }
}

fn hash_source(mut hash: u64, source: &PlannedImageSource) -> u64 {
    match source {
        PlannedImageSource::Scope(scope) => {
            hash = hash_value(hash, 1);
            hash_scope(hash, *scope)
        }
        PlannedImageSource::SelectedLayer { layer_id, stage } => {
            hash = hash_value(hash, 6);
            hash = hash_value(hash, layer_id.get());
            hash_value(hash, stage_code(*stage))
        }
        PlannedImageSource::OneBelow(scope) => {
            hash = hash_value(hash, 2);
            hash_scope(hash, *scope)
        }
        PlannedImageSource::AllBelow(prefix) => {
            hash = hash_value(hash, 3);
            match prefix {
                CompositePrefix::Root {
                    preceding_root_outputs,
                } => {
                    hash = hash_value(hash, 1);
                    hash_value(hash, *preceding_root_outputs as u64)
                }
                CompositePrefix::GroupMember {
                    group_id,
                    preceding_root_outputs,
                    preceding_members,
                } => {
                    hash = hash_value(hash, 2);
                    hash = hash_value(hash, group_id.get());
                    hash = hash_value(hash, *preceding_root_outputs as u64);
                    hash_value(hash, *preceding_members as u64)
                }
            }
        }
        PlannedImageSource::ProgramHistory => hash_value(hash, 4),
        PlannedImageSource::Transparent => hash_value(hash, 5),
    }
}

fn hash_scope(mut hash: u64, scope: VisualScopeId) -> u64 {
    match scope {
        VisualScopeId::Master => hash_value(hash, 1),
        VisualScopeId::Layer(id) => {
            hash = hash_value(hash, 2);
            hash_value(hash, id.get())
        }
        VisualScopeId::Group(id) => {
            hash = hash_value(hash, 3);
            hash_value(hash, id.get())
        }
        VisualScopeId::Program => hash_value(hash, 4),
    }
}

const fn bus_code(bus: BusAssignment) -> u64 {
    match bus {
        BusAssignment::A => 1,
        BusAssignment::B => 2,
        BusAssignment::Program => 3,
    }
}

const fn timing_code(timing: EdgeTiming) -> u64 {
    match timing {
        EdgeTiming::CurrentFrame => 1,
        EdgeTiming::PreviousFrame => 2,
    }
}

const fn stage_code(stage: LayerImageStage) -> u64 {
    match stage {
        LayerImageStage::PreLocalEffects => 1,
        LayerImageStage::PostLocalEffects => 2,
    }
}

fn hash_value(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{GroupName, RuntimeGroup, RuntimeGroupMembers};
    use crate::effects::params::{
        CollisionAtlasParams, TemporalInterpolation, TemporalLoomParams, TemporalOriginalsParams,
        TemporalParams, TemporalTopology,
    };
    use crate::evaluated_frame::{FramePlanContext, LayerFrameInput, MasterFrameInput, SourceTap};
    use crate::image_routing::{LayerMatte, MatteChannel as LegacyMatteChannel};
    use crate::layers::BlendMode;
    use crate::modulation::{ModMatrix, ModSource, Routing};
    use crate::ntsc::NtscParams;
    use crate::spatial::SpatialTransform;
    use crate::visual_rack::{
        DigitalColorParams, MatteChannel as RackMatteChannel, RuntimeVisualNode,
    };

    fn layer_id(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn group_id(value: u64) -> GroupId {
        GroupId::new(value).unwrap()
    }

    fn saved_position(value: u32) -> SavedLayerPosition {
        SavedLayerPosition::new(value).unwrap()
    }

    fn base(ids_front_to_back: &[u64], bypass: &[u64]) -> EvaluatedFramePlan {
        base_with_ntsc(ids_front_to_back, bypass, false)
    }

    fn base_with_ntsc(
        ids_front_to_back: &[u64],
        bypass: &[u64],
        ntsc_enabled: bool,
    ) -> EvaluatedFramePlan {
        base_with_ntsc_and_opacities(
            ids_front_to_back,
            bypass,
            ntsc_enabled,
            &vec![1.0; ids_front_to_back.len()],
        )
    }

    fn base_with_ntsc_and_opacities(
        ids_front_to_back: &[u64],
        bypass: &[u64],
        ntsc_enabled: bool,
        opacities: &[f32],
    ) -> EvaluatedFramePlan {
        assert_eq!(ids_front_to_back.len(), opacities.len());
        let effects = vec![EffectUniforms::default(); ids_front_to_back.len()];
        let transforms = vec![SpatialTransform::new_layer_default(); ids_front_to_back.len()];
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams {
            enabled: ntsc_enabled,
            ..NtscParams::default()
        };
        let temporal = TemporalParams::default();
        let matrix = ModMatrix::new();
        let modulation = matrix.frame(ids_front_to_back.len());
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 64, 1.25),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            ids_front_to_back
                .iter()
                .enumerate()
                .map(|(index, id)| LayerFrameInput {
                    source: SourceTap::new(*id, index, 64, 64),
                    effects: &effects[index],
                    transform: &transforms[index],
                    opacity: opacities[index],
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: bypass.contains(id),
                }),
        )
    }

    fn legacy_composition(ids_front_to_back: &[u64]) -> RuntimeComposition {
        RuntimeComposition::try_from_parts(
            Vec::new(),
            ids_front_to_back
                .iter()
                .rev()
                .map(|id| RuntimeRootItem::Layer {
                    layer_id: layer_id(*id),
                    bus: BusAssignment::Program,
                })
                .collect(),
            None,
            0.5,
        )
        .unwrap()
    }

    fn legacy_racks(ids: &[u64]) -> Vec<(StableLayerId, RuntimeVisualRack)> {
        ids.iter()
            .map(|id| {
                (
                    layer_id(*id),
                    RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer),
                )
            })
            .collect()
    }

    fn base_with_temporal(temporal: &TemporalParams) -> EvaluatedFramePlan {
        let effects = EffectUniforms::default();
        let transform = SpatialTransform::new_layer_default();
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let matrix = ModMatrix::new();
        let modulation = matrix.frame(1);
        EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 64, 1.25),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(1, 0, 64, 64),
                effects: &effects,
                transform: &transform,
                opacity: 1.0,
                speed: 1.0,
                fps: 30.0,
                blend_mode: BlendMode::Normal,
                visible: true,
                paused: false,
                bypass_master_fx: false,
            }],
        )
    }

    fn image_matte(source: ResolvedImageSource, timing: EdgeTiming) -> RuntimeImageMatte {
        RuntimeImageMatte {
            tap: ResolvedImageTap { source, timing },
            channel: RackMatteChannel::Alpha,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 0.1,
        }
    }

    fn push_image_mask(
        rack: &mut RuntimeVisualRack,
        source: ResolvedImageSource,
        timing: EdgeTiming,
    ) -> NodeId {
        rack.push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
            image_matte(source, timing),
        )))
        .unwrap()
    }

    fn plan(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        EvaluatedCompositionPlan::evaluate(
            base,
            CompositionPlanInput::new(composition, master, racks),
        )
    }

    fn plan_with_motion(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
        master_motion: MotionParams,
        layers: &[LayerMotionPlanInput],
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        EvaluatedCompositionPlan::evaluate(
            base,
            CompositionPlanInput::new(composition, master, racks).with_motion(
                master_motion,
                layers,
                MotionDeviceLimits::new(8_192, u64::MAX),
            ),
        )
    }

    fn advanced(result: EvaluatedCompositionPlan) -> AdvancedCompositionPlan {
        match result {
            EvaluatedCompositionPlan::Advanced(plan) => *plan,
            EvaluatedCompositionPlan::LegacyExact(_) => panic!("expected advanced plan"),
        }
    }

    #[test]
    fn exact_legacy_is_tagged_without_collision_rack_work_and_reverses_ui_order() {
        let base = base(&[10, 20, 30], &[]);
        let composition = legacy_composition(&[10, 20, 30]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[10, 20, 30]);
        let result = plan(&base, &composition, &master, &racks).unwrap();
        let EvaluatedCompositionPlan::LegacyExact(exact) = result else {
            panic!("exact legacy stack compiled advanced work");
        };
        assert_eq!(
            exact.flattened_layers(),
            &[layer_id(30), layer_id(20), layer_id(10)]
        );

        let zero_amount = [
            LayerMatte {
                enabled: true,
                amount: 0.0,
                ..LayerMatte::default()
            },
            LayerMatte::default(),
            LayerMatte::default(),
        ];
        let input = CompositionPlanInput::new(&composition, &master, &racks)
            .with_layer_mattes(&zero_amount, false);
        assert!(matches!(
            EvaluatedCompositionPlan::evaluate(&base, input).unwrap(),
            EvaluatedCompositionPlan::LegacyExact(_)
        ));
    }

    #[test]
    fn exact_zero_motion_preserves_literal_legacy_classification_and_signature() {
        let base = base(&[10, 20], &[]);
        let composition = legacy_composition(&[10, 20]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[10, 20]);
        let without = plan(&base, &composition, &master, &racks).unwrap();
        let layers = [10, 20].map(|id| LayerMotionPlanInput {
            stable_id: layer_id(id),
            params: MotionParams::default(),
            codec: MotionCodecFrameFacts::default(),
        });
        let with = plan_with_motion(
            &base,
            &composition,
            &master,
            &racks,
            MotionParams::default(),
            &layers,
        )
        .unwrap();
        assert!(without.is_legacy_exact());
        assert!(with.is_legacy_exact());
        assert_eq!(without.topology_signature(), with.topology_signature());
    }

    #[test]
    fn motion_plan_carries_the_exact_modulated_spatial_fact() {
        let effects = EffectUniforms::default();
        let authored = SpatialTransform::default();
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings = vec![Routing::new(ModSource::Midi(0), "layer1_position_x", 0.25)];
        matrix.update_at_beat(0.0, 0.0);
        let modulation = matrix.frame(1);
        let base = EvaluatedFramePlan::evaluate(
            &modulation,
            FramePlanContext::new(64, 64, 1.25),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal: &temporal,
            },
            [LayerFrameInput {
                source: SourceTap::new(1, 0, 64, 64),
                effects: &effects,
                transform: &authored,
                opacity: 1.0,
                speed: 1.0,
                fps: 30.0,
                blend_mode: BlendMode::Normal,
                visible: true,
                paused: false,
                bypass_master_fx: false,
            }],
        );
        assert_eq!(base.layers()[0].transform.position[0], 1.0);
        let composition = legacy_composition(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[1]);
        let layers = [LayerMotionPlanInput {
            stable_id: layer_id(1),
            params: MotionParams {
                shutter: crate::motion::CurvedShutterParams {
                    angle_degrees: 180.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            codec: MotionCodecFrameFacts::default(),
        }];
        let advanced = advanced(
            plan_with_motion(
                &base,
                &composition,
                &master,
                &racks,
                MotionParams::default(),
                &layers,
            )
            .unwrap(),
        );
        let motion = advanced.motion().advanced().unwrap();
        let scope = motion.scope(VisualScopeId::Layer(layer_id(1))).unwrap();
        assert_eq!(scope.transform, base.layers()[0].transform);
        assert_eq!(scope.spatial, base.layers()[0].spatial);
    }

    #[test]
    fn shutter_and_exact_faraday_donor_compile_one_canonical_field_plan() {
        let base = base(&[10, 20], &[]);
        let composition = legacy_composition(&[10, 20]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[10, 20]);
        let donor = MotionParams::default();
        let recipient = MotionParams {
            transplant: crate::motion::FaradayParams {
                amount: 0.8,
                donor: MotionDonor::Selected {
                    layer_id: layer_id(20),
                    saved_position: saved_position(1),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let layers = [
            LayerMotionPlanInput {
                stable_id: layer_id(10),
                params: recipient,
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: layer_id(20),
                params: donor,
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        let advanced = advanced(
            plan_with_motion(
                &base,
                &composition,
                &master,
                &racks,
                MotionParams::default(),
                &layers,
            )
            .unwrap(),
        );
        let motion = advanced.motion().advanced().unwrap();
        assert_eq!(motion.fields().len(), 1);
        assert_eq!(motion.fields()[0].scope, VisualScopeId::Layer(layer_id(20)));
        assert!(motion.fields()[0].required_as_donor);
        assert_eq!(motion.resources().active_field_slots, 1);
        assert_eq!(motion.resources().persistent_carriers, 1);
        assert_eq!(motion.resources().carrier_bytes, 64 * 64 * 16);
        let recipient = motion.scope(VisualScopeId::Layer(layer_id(10))).unwrap();
        assert!(recipient.transplant_admitted);
        assert_eq!(recipient.donor_field_slot, Some(0));
    }

    #[test]
    fn missing_and_excess_faraday_are_visible_and_never_retarget() {
        let base = base(&[10, 20, 30], &[]);
        let composition = legacy_composition(&[10, 20, 30]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[10, 20, 30]);
        let transplant = |donor| MotionParams {
            transplant: crate::motion::FaradayParams {
                amount: 1.0,
                donor,
                ..Default::default()
            },
            ..Default::default()
        };
        let layers = [
            LayerMotionPlanInput {
                stable_id: layer_id(10),
                params: transplant(MotionDonor::Missing {
                    saved_position: saved_position(8),
                }),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: layer_id(20),
                params: transplant(MotionDonor::Selected {
                    layer_id: layer_id(30),
                    saved_position: saved_position(2),
                }),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: layer_id(30),
                params: transplant(MotionDonor::Selected {
                    layer_id: layer_id(20),
                    saved_position: saved_position(1),
                }),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        let advanced = advanced(
            plan_with_motion(
                &base,
                &composition,
                &master,
                &racks,
                MotionParams::default(),
                &layers,
            )
            .unwrap(),
        );
        let motion = advanced.motion().advanced().unwrap();
        assert_eq!(motion.resources().active_transplants, 1);
        assert_eq!(motion.resources().active_field_slots, 1);
        assert!(motion.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            MotionPlanDiagnostic::MissingDonor { recipient, .. } if *recipient == layer_id(10)
        )));
        assert!(motion.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            MotionPlanDiagnostic::ExcessTransplantRejected { recipient, .. }
                if *recipient == layer_id(20) || *recipient == layer_id(30)
        )));
    }

    #[test]
    fn codec_attachment_requires_exact_stable_generation_algorithm_dimensions_and_grid() {
        let grid =
            MotionGrid::for_source([64, 64], crate::motion::MotionLatticeQuality::Live).unwrap();
        let field = MotionField::zeroed([64, 64], grid, MotionFieldOrigin::CodecVectors).unwrap();
        let plan = EvaluatedMotionFieldPlan {
            slot: 0,
            scope: VisualScopeId::Layer(layer_id(10)),
            source_dimensions: [64, 64],
            grid,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            source: MotionSourceDecision {
                origin: MotionFieldOrigin::CodecVectors,
                diagnostic: MotionSourceDiagnostic::None,
            },
            codec: MotionCodecFrameFacts {
                available: true,
                source_generation: 7,
                frame_ordinal: 9,
            },
            required_as_donor: false,
        };
        let attachment = MotionFieldAttachment {
            scope: plan.scope,
            source_generation: 7,
            frame_ordinal: 9,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            source_dimensions: [64, 64],
            grid,
            field: &field,
        };
        assert!(plan.accepts(attachment));
        assert!(!plan.accepts(MotionFieldAttachment {
            source_generation: 8,
            ..attachment
        }));
        assert!(!plan.accepts(MotionFieldAttachment {
            source_dimensions: [32, 64],
            ..attachment
        }));
    }

    #[test]
    fn exact_legacy_and_sparse_advanced_paths_admit_more_than_256_direct_layers() {
        let ids: Vec<_> = (1..=300).map(|value| value as u64).collect();
        let base_plan = base(&ids, &[]);
        let composition = legacy_composition(&ids);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&ids);
        assert!(plan(&base_plan, &composition, &master, &racks)
            .unwrap()
            .is_legacy_exact());

        racks[149]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let advanced = advanced(plan(&base_plan, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.layers().len(), 300);
        assert!(advanced.graph().current_topological_order.len() <= 1);
        assert!(advanced.resources().full_frame_passes > 0);
    }

    #[test]
    fn hybrid_markers_split_exactly_and_materialize_spatial_once() {
        let base = base(&[1], &[]);
        let composition = legacy_composition(&[1]);
        let mut racks = legacy_racks(&[1]);
        racks[0]
            .1
            .push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        let master = RuntimeVisualRack::try_from_parts(
            vec![
                RuntimeVisualNode::authored(
                    NodeId::new(3).unwrap(),
                    RuntimeVisualNodeKind::Transform(SpatialTransform::default()),
                ),
                RuntimeVisualNode::authored(
                    NodeId::LEGACY_TEMPORAL,
                    RuntimeVisualNodeKind::LegacyTemporal,
                ),
                RuntimeVisualNode::authored(
                    NodeId::new(4).unwrap(),
                    RuntimeVisualNodeKind::DigitalColor(DigitalColorParams::default()),
                ),
            ],
            Some(5),
        )
        .unwrap();
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        let layer_steps = advanced.layers()[0].execution.steps();
        assert!(matches!(
            layer_steps[0],
            EvaluatedScopeStep::MaterializeSpatial { .. }
        ));
        assert!(matches!(
            layer_steps[1],
            EvaluatedScopeStep::LegacyCanonical { .. }
        ));
        assert!(matches!(
            layer_steps[2],
            EvaluatedScopeStep::CollisionRack { .. }
        ));
        if let EvaluatedScopeStep::MaterializeSpatial { pass, application } = layer_steps[0] {
            assert_eq!(application, LegacyCanonicalApplication::ScopeLocal);
            assert_eq!(
                bytemuck::bytes_of(&pass.spatial),
                bytemuck::bytes_of(&base.layer_pre_passes()[0].spatial)
            );
        }
        if let EvaluatedScopeStep::LegacyCanonical { pass, application } = layer_steps[1] {
            let identity = SpatialTransform::default().gpu_uniforms(64, 64, 64, 64);
            assert_eq!(application, LegacyCanonicalApplication::ScopeLocal);
            assert_eq!(
                bytemuck::bytes_of(&pass.spatial),
                bytemuck::bytes_of(&identity)
            );
        }
        let master_steps = advanced.master().execution.steps();
        assert!(matches!(
            master_steps[0],
            EvaluatedScopeStep::MaterializeSpatial { .. }
        ));
        assert!(matches!(
            master_steps[1],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 0,
                ..
            }
        ));
        assert!(matches!(
            master_steps[2],
            EvaluatedScopeStep::LegacyTemporal { .. }
        ));
        assert!(matches!(
            master_steps[3],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn advanced_plan_preserves_temporal_originals_at_the_authored_marker() {
        let originals = TemporalOriginalsParams {
            loom: TemporalLoomParams {
                amount: 0.75,
                topology: TemporalTopology::Kaleidoscopic,
                interpolation: TemporalInterpolation::Linear,
                depth: 0.8,
                phase: 0.125,
                scale: 2.0,
                angle: 33.0,
                folds: 7,
                quantization: 5,
            },
            atlas: CollisionAtlasParams {
                amount: 0.6,
                seed: 0x6a09_e667,
                territories: 11,
                collision: 0.4,
            },
            ..TemporalOriginalsParams::default()
        };
        let temporal = TemporalParams {
            originals,
            ..TemporalParams::default()
        };
        let base = base_with_temporal(&temporal);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::try_from_parts(
            vec![
                RuntimeVisualNode::authored(
                    NodeId::new(3).unwrap(),
                    RuntimeVisualNodeKind::DigitalColor(DigitalColorParams::default()),
                ),
                RuntimeVisualNode::authored(
                    NodeId::LEGACY_TEMPORAL,
                    RuntimeVisualNodeKind::LegacyTemporal,
                ),
                RuntimeVisualNode::authored(
                    NodeId::new(4).unwrap(),
                    RuntimeVisualNodeKind::Transform(SpatialTransform::default()),
                ),
            ],
            Some(5),
        )
        .unwrap();
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        let steps = advanced.master().execution.steps();
        assert!(matches!(steps[1], EvaluatedScopeStep::CollisionRack { .. }));
        let EvaluatedScopeStep::LegacyTemporal { params } = steps[2] else {
            panic!("authored temporal marker moved in the advanced plan");
        };
        assert_eq!(params.originals, originals);
        assert!(matches!(steps[3], EvaluatedScopeStep::CollisionRack { .. }));
    }

    #[test]
    fn reordered_master_canonical_becomes_scope_local_without_bypass() {
        let base = base(&[1], &[]);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::try_from_parts(
            vec![
                RuntimeVisualNode::authored(
                    NodeId::new(3).unwrap(),
                    RuntimeVisualNodeKind::DigitalColor(DigitalColorParams::default()),
                ),
                RuntimeVisualNode::authored(
                    NodeId::LEGACY_CANONICAL,
                    RuntimeVisualNodeKind::LegacyCanonical,
                ),
                RuntimeVisualNode::authored(
                    NodeId::LEGACY_TEMPORAL,
                    RuntimeVisualNodeKind::LegacyTemporal,
                ),
            ],
            Some(4),
        )
        .unwrap();
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(matches!(
            advanced.master().execution.steps(),
            [
                EvaluatedScopeStep::MaterializeSpatial { .. },
                EvaluatedScopeStep::CollisionRack { .. },
                EvaluatedScopeStep::LegacyCanonical {
                    application: LegacyCanonicalApplication::ScopeLocal,
                    ..
                },
                EvaluatedScopeStep::LegacyTemporal { .. },
            ]
        ));
    }

    #[test]
    fn temporal_before_canonical_with_contributing_bypass_is_rejected() {
        let base = base(&[1], &[1]);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::try_from_parts(
            vec![
                RuntimeVisualNode::authored(
                    NodeId::LEGACY_TEMPORAL,
                    RuntimeVisualNodeKind::LegacyTemporal,
                ),
                RuntimeVisualNode::authored(
                    NodeId::LEGACY_CANONICAL,
                    RuntimeVisualNodeKind::LegacyCanonical,
                ),
            ],
            Some(3),
        )
        .unwrap();
        assert!(matches!(
            plan(&base, &composition, &master, &racks),
            Err(CompositionPlanError::AmbiguousMasterBypass { layers })
                if layers == vec![layer_id(1)]
        ));
    }

    #[test]
    fn custom_master_with_contributing_bypass_is_rejected_visibly() {
        let base = base(&[1, 2], &[1]);
        let composition = legacy_composition(&[1, 2]);
        let racks = legacy_racks(&[1, 2]);
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        master
            .push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        assert!(matches!(
            plan(&base, &composition, &master, &racks),
            Err(CompositionPlanError::AmbiguousMasterBypass { layers })
                if layers == vec![layer_id(1)]
        ));
    }

    #[test]
    fn dormant_zero_weight_bypass_does_not_create_master_or_ntsc_split() {
        let base = base_with_ntsc_and_opacities(&[1, 2], &[1], true, &[0.0, 1.0]);
        let composition = legacy_composition(&[1, 2]);
        let racks = legacy_racks(&[1, 2]);
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        master
            .push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.ntsc_path(), AdvancedNtscPath::AllApplying);
        assert_eq!(
            advanced.master().selective_ntsc_layers.as_ref(),
            &[layer_id(2)]
        );
        assert!(advanced.master().selective_ntsc_bypass_layers.is_empty());
        assert!(advanced.master().canonical_bypass_layers.is_empty());
    }

    #[test]
    fn zero_bus_weight_and_group_opacity_are_not_contributors() {
        let base = base_with_ntsc(&[3, 2, 1], &[1, 2], true);
        let group = RuntimeGroup {
            id: group_id(10),
            name: GroupName::new("Dormant").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![layer_id(1)]).unwrap(),
            opacity: 0.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        };
        let composition = RuntimeComposition::try_from_parts(
            vec![group],
            vec![
                RuntimeRootItem::Group {
                    group_id: group_id(10),
                },
                RuntimeRootItem::Layer {
                    layer_id: layer_id(2),
                    bus: BusAssignment::A,
                },
                RuntimeRootItem::Layer {
                    layer_id: layer_id(3),
                    bus: BusAssignment::B,
                },
            ],
            None,
            1.0,
        )
        .unwrap();
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        master
            .push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        let racks = legacy_racks(&[1, 2, 3]);
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.ntsc_path(), AdvancedNtscPath::AllApplying);
        assert_eq!(
            advanced.master().selective_ntsc_layers.as_ref(),
            &[layer_id(3)]
        );
        assert!(advanced.master().selective_ntsc_bypass_layers.is_empty());
    }

    #[test]
    fn homogeneous_all_bypass_is_classified_without_false_global_ntsc() {
        let base = base_with_ntsc(&[1], &[1], true);
        let racks = legacy_racks(&[1]);
        // A custom master with a *contributing* bypass is intentionally
        // ambiguous. Use a grouped advanced topology with the exact master to
        // exercise homogeneous NTSC classification independently.
        let group = RuntimeGroup {
            id: group_id(9),
            name: GroupName::new("G").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![layer_id(1)]).unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        };
        let composition = RuntimeComposition::try_from_parts(
            vec![group],
            vec![RuntimeRootItem::Group {
                group_id: group_id(9),
            }],
            None,
            0.5,
        )
        .unwrap();
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.ntsc_path(), AdvancedNtscPath::AllBypass);
    }

    #[test]
    fn exact_master_in_advanced_group_preserves_per_layer_bypass_admission() {
        let (_, composition, master, racks) = grouped_fixture();
        // Re-evaluate the same stable stack with one contributing bypass.
        let base = base_with_ntsc(&[4, 3, 2, 1], &[2], true);
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(!advanced.master().execution.is_exact_legacy());
        let steps = advanced.master().execution.steps();
        assert!(matches!(
            steps,
            [
                EvaluatedScopeStep::MaterializeSpatial {
                    application: LegacyCanonicalApplication::PreCompositeLayerAdmission,
                    ..
                },
                EvaluatedScopeStep::LegacyCanonical {
                    application: LegacyCanonicalApplication::PreCompositeLayerAdmission,
                    ..
                },
                EvaluatedScopeStep::LegacyTemporal { .. }
            ]
        ));
        assert_eq!(
            advanced.master().canonical_bypass_layers.as_ref(),
            &[layer_id(2)]
        );
        assert!(!advanced.master().canonical_layers.contains(&layer_id(2)));
        assert!(advanced.master().canonical_layers.contains(&layer_id(3)));
        assert_eq!(
            advanced.master().selective_ntsc_bypass_layers.as_ref(),
            &[layer_id(2)]
        );
        assert!(advanced
            .master()
            .selective_ntsc_layers
            .contains(&layer_id(3)));
    }

    #[test]
    fn no_bypass_master_canonical_remains_global_after_group_composition() {
        let (base, composition, master, racks) = grouped_fixture();
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(matches!(
            advanced.master().execution.steps(),
            [
                EvaluatedScopeStep::MaterializeSpatial {
                    application: LegacyCanonicalApplication::ScopeLocal,
                    ..
                },
                EvaluatedScopeStep::LegacyCanonical {
                    application: LegacyCanonicalApplication::ScopeLocal,
                    ..
                },
                EvaluatedScopeStep::LegacyTemporal { .. }
            ]
        ));
        assert!(advanced.master().canonical_bypass_layers.is_empty());
        assert_eq!(advanced.master().canonical_layers.len(), 4);
    }

    fn grouped_fixture() -> (
        EvaluatedFramePlan,
        RuntimeComposition,
        RuntimeVisualRack,
        Vec<(StableLayerId, RuntimeVisualRack)>,
    ) {
        let base = base(&[4, 3, 2, 1], &[]);
        let group = RuntimeGroup {
            id: group_id(10),
            name: GroupName::new("G").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![layer_id(2), layer_id(3)]).unwrap(),
            opacity: 0.75,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::B,
        };
        let composition = RuntimeComposition::try_from_parts(
            vec![group],
            vec![
                RuntimeRootItem::Layer {
                    layer_id: layer_id(1),
                    bus: BusAssignment::A,
                },
                RuntimeRootItem::Group {
                    group_id: group_id(10),
                },
                RuntimeRootItem::Layer {
                    layer_id: layer_id(4),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.25,
        )
        .unwrap();
        (
            base,
            composition,
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master),
            legacy_racks(&[1, 2, 3, 4]),
        )
    }

    fn two_group_fixture() -> (
        EvaluatedFramePlan,
        RuntimeComposition,
        RuntimeVisualRack,
        Vec<(StableLayerId, RuntimeVisualRack)>,
    ) {
        let group_a = RuntimeGroup {
            id: group_id(10),
            name: GroupName::new("A").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![layer_id(1), layer_id(2)]).unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        };
        let group_b = RuntimeGroup {
            id: group_id(20),
            name: GroupName::new("B").unwrap(),
            members: RuntimeGroupMembers::try_from_vec(vec![layer_id(3), layer_id(4)]).unwrap(),
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: RuntimeVisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        };
        let composition = RuntimeComposition::try_from_parts(
            vec![group_a, group_b],
            vec![
                RuntimeRootItem::Group {
                    group_id: group_id(10),
                },
                RuntimeRootItem::Group {
                    group_id: group_id(20),
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        (
            base(&[4, 3, 2, 1], &[]),
            composition,
            RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master),
            legacy_racks(&[1, 2, 3, 4]),
        )
    }

    #[test]
    fn groups_are_contiguous_atomic_tasks_and_cross_group_task_cycles_reject() {
        let (base, composition, master, mut racks) = two_group_fixture();
        // One direction is schedulable and the emitted scope order keeps each
        // group's members plus output in a single contiguous block.
        push_image_mask(
            &mut racks
                .iter_mut()
                .find(|(id, _)| *id == layer_id(3))
                .unwrap()
                .1,
            ResolvedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                saved_position: saved_position(0),
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        );
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        let order = advanced.execution_order();
        for group in [group_id(10), group_id(20)] {
            let positions = order
                .iter()
                .enumerate()
                .filter_map(|(index, scope)| {
                    let belongs = match scope {
                        VisualScopeId::Group(id) => *id == group,
                        VisualScopeId::Layer(id) => composition
                            .group(group)
                            .is_some_and(|group| group.members.iter().any(|member| member == *id)),
                        _ => false,
                    };
                    belongs.then_some(index)
                })
                .collect::<Vec<_>>();
            assert_eq!(positions.len(), 3);
            assert_eq!(positions[2] - positions[0], 2);
        }

        // The scope graph remains acyclic, but the reverse cross-group edge
        // would require retaining/interleaving two group accumulators.
        push_image_mask(
            &mut racks
                .iter_mut()
                .find(|(id, _)| *id == layer_id(2))
                .unwrap()
                .1,
            ResolvedImageSource::SelectedLayer {
                layer_id: layer_id(4),
                saved_position: saved_position(3),
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        );
        assert!(matches!(
            plan(&base, &composition, &master, &racks),
            Err(CompositionPlanError::AtomicGroupCycle { tasks })
                if tasks.contains(&VisualScopeId::Group(group_id(10)))
                    && tasks.contains(&VisualScopeId::Group(group_id(20)))
        ));
    }

    #[test]
    fn group_flatten_bus_and_below_domains_use_root_outputs() {
        let (base, composition, mut master, mut racks) = grouped_fixture();
        let node_one = push_image_mask(
            &mut racks[2].1,
            ResolvedImageSource::OneBelow,
            EdgeTiming::CurrentFrame,
        );
        let node_all = push_image_mask(
            &mut racks[2].1,
            ResolvedImageSource::AllBelow,
            EdgeTiming::CurrentFrame,
        );
        let root_node = push_image_mask(
            &mut racks[3].1,
            ResolvedImageSource::OneBelow,
            EdgeTiming::CurrentFrame,
        );
        let master_node = push_image_mask(
            &mut master,
            ResolvedImageSource::AllBelow,
            EdgeTiming::CurrentFrame,
        );
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.bus_crossfade(), 0.25);
        assert_eq!(
            advanced.groups()[0].members.as_ref(),
            &[layer_id(2), layer_id(3)]
        );
        assert_eq!(advanced.groups()[0].span.start, 1);
        assert_eq!(advanced.groups()[0].span.len, 2);

        let find = |scope, node_id| {
            advanced
                .image_taps()
                .iter()
                .find(|tap| {
                    matches!(
                        tap.consumer,
                        ImageTapConsumer::RackNode { scope: owner, node_id: id }
                            if owner == scope && id == node_id
                    )
                })
                .unwrap()
        };
        assert_eq!(
            find(VisualScopeId::Layer(layer_id(3)), node_one).resolved,
            PlannedImageSource::OneBelow(VisualScopeId::Layer(layer_id(2)))
        );
        let PlannedImageSource::AllBelow(prefix) =
            find(VisualScopeId::Layer(layer_id(3)), node_all).resolved
        else {
            panic!()
        };
        assert!(advanced
            .below_topology()
            .prefix_contains(prefix, VisualScopeId::Layer(layer_id(1))));
        assert!(advanced
            .below_topology()
            .prefix_contains(prefix, VisualScopeId::Layer(layer_id(2))));
        assert!(!advanced
            .below_topology()
            .prefix_contains(prefix, VisualScopeId::Group(group_id(10))));
        assert_eq!(
            find(VisualScopeId::Layer(layer_id(4)), root_node).resolved,
            PlannedImageSource::OneBelow(VisualScopeId::Group(group_id(10)))
        );
        let PlannedImageSource::AllBelow(master_prefix) =
            find(VisualScopeId::Master, master_node).resolved
        else {
            panic!()
        };
        for producer in [
            VisualScopeId::Layer(layer_id(1)),
            VisualScopeId::Group(group_id(10)),
            VisualScopeId::Layer(layer_id(4)),
        ] {
            assert!(advanced
                .below_topology()
                .prefix_contains(master_prefix, producer));
        }
    }

    #[test]
    fn member_current_own_group_output_cycles_but_previous_frame_is_admitted() {
        let (base, composition, master, mut racks) = grouped_fixture();
        push_image_mask(
            &mut racks[1].1,
            ResolvedImageSource::GroupOutput(group_id(10)),
            EdgeTiming::CurrentFrame,
        );
        assert!(matches!(
            plan(&base, &composition, &master, &racks),
            Err(CompositionPlanError::CurrentCycle { .. })
        ));

        let mut racks = legacy_racks(&[1, 2, 3, 4]);
        push_image_mask(
            &mut racks[1].1,
            ResolvedImageSource::GroupOutput(group_id(10)),
            EdgeTiming::PreviousFrame,
        );
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.graph().previous_taps, 1);
    }

    #[test]
    fn previous_self_edges_are_legal_but_history_tap_cap_is_enforced() {
        let ids: Vec<_> = (1..=9).collect();
        let base_plan = base(&ids, &[]);
        let composition = legacy_composition(&ids);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&ids);
        for (id, rack) in &mut racks {
            push_image_mask(
                rack,
                ResolvedImageSource::SelectedLayer {
                    layer_id: *id,
                    saved_position: saved_position(0),
                    stage: LayerImageStage::PostLocalEffects,
                },
                EdgeTiming::PreviousFrame,
            );
        }
        assert!(matches!(
            plan(&base_plan, &composition, &master, &racks),
            Err(CompositionPlanError::ImageGraph(
                ImageGraphError::TooManyPreviousTaps { count: 9, .. }
            ))
        ));
        racks.pop();
        let ids = &ids[..8];
        let base = base(ids, &[]);
        let composition = legacy_composition(ids);
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.graph().previous_taps, 8);
    }

    #[test]
    fn current_clean_program_rejects_and_missing_route_is_transparent_with_diagnostic() {
        let base_plan = base(&[1], &[]);
        let composition = legacy_composition(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1]);
        push_image_mask(
            &mut racks[0].1,
            ResolvedImageSource::CleanProgram,
            EdgeTiming::CurrentFrame,
        );
        assert!(matches!(
            plan(&base_plan, &composition, &master, &racks),
            Err(CompositionPlanError::ImageGraph(
                ImageGraphError::CurrentProgramInput { .. }
            ))
        ));

        let mut legacy_base = base(&[1], &[]);
        legacy_base
            .attach_image_routing(
                [LayerMatte {
                    enabled: true,
                    input: ImageInput::CleanProgram,
                    ..LayerMatte::default()
                }],
                false,
            )
            .unwrap();
        let legacy =
            advanced(plan(&legacy_base, &composition, &master, &legacy_racks(&[1])).unwrap());
        assert_eq!(
            legacy.image_taps()[0].resolved,
            PlannedImageSource::Transparent
        );
        assert!(matches!(
            legacy.diagnostics()[0],
            CompositionPlanDiagnostic::LegacyCleanProgramTransparent { .. }
        ));

        let mut racks = legacy_racks(&[1]);
        push_image_mask(
            &mut racks[0].1,
            ResolvedImageSource::MissingSelectedLayer {
                saved_position: saved_position(99),
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        );
        let advanced = advanced(plan(&base_plan, &composition, &master, &racks).unwrap());
        assert_eq!(
            advanced.image_taps()[0].resolved,
            PlannedImageSource::Transparent
        );
        assert!(matches!(
            advanced.diagnostics()[0],
            CompositionPlanDiagnostic::MissingSelectedLayer { .. }
        ));
    }

    #[test]
    fn selected_stable_donor_survives_root_reorder() {
        let base = base(&[1, 2], &[]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        push_image_mask(
            &mut racks[0].1,
            ResolvedImageSource::SelectedLayer {
                layer_id: layer_id(2),
                saved_position: saved_position(0),
                stage: LayerImageStage::PreLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        );
        let first = legacy_composition(&[1, 2]);
        let second = RuntimeComposition::try_from_parts(
            Vec::new(),
            vec![
                RuntimeRootItem::Layer {
                    layer_id: layer_id(1),
                    bus: BusAssignment::Program,
                },
                RuntimeRootItem::Layer {
                    layer_id: layer_id(2),
                    bus: BusAssignment::Program,
                },
            ],
            None,
            0.5,
        )
        .unwrap();
        for composition in [&first, &second] {
            let advanced = advanced(plan(&base, composition, &master, &racks).unwrap());
            assert_eq!(
                advanced.image_taps()[0].resolved,
                PlannedImageSource::SelectedLayer {
                    layer_id: layer_id(2),
                    stage: LayerImageStage::PreLocalEffects,
                }
            );
        }
    }

    #[test]
    fn base_layer_mattes_join_unified_routes_resources_and_cycles() {
        let mut base = base(&[1, 2], &[]);
        base.attach_image_routing(
            vec![
                LayerMatte {
                    enabled: true,
                    input: ImageInput::SelectedLayer {
                        layer_id: layer_id(2),
                        stage: LayerImageStage::PreLocalEffects,
                    },
                    channel: LegacyMatteChannel::Luma,
                    ..LayerMatte::default()
                },
                LayerMatte::default(),
            ],
            true,
        )
        .unwrap();
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[1, 2]);
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(!advanced.base().image_routing().is_active());
        assert!(advanced.layers()[1].legacy_matte.is_some());
        assert!(
            matches!(advanced.image_taps()[0].consumer, ImageTapConsumer::LayerMatte { layer_id: id } if id == layer_id(1))
        );
        assert_eq!(
            advanced.image_taps()[0].resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(2),
                stage: LayerImageStage::PreLocalEffects,
            }
        );
        assert!(advanced.resources().full_frame_passes >= 1);

        let mut cycle_racks = legacy_racks(&[1, 2]);
        push_image_mask(
            &mut cycle_racks[1].1,
            ResolvedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                saved_position: saved_position(0),
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
        );
        let postlocal_cycle_mattes = [
            LayerMatte {
                enabled: true,
                input: ImageInput::SelectedLayer {
                    layer_id: layer_id(2),
                    stage: LayerImageStage::PostLocalEffects,
                },
                ..LayerMatte::default()
            },
            LayerMatte::default(),
        ];
        let cycle_input = CompositionPlanInput::new(&composition, &master, &cycle_racks)
            .with_layer_mattes(&postlocal_cycle_mattes, true);
        assert!(matches!(
            EvaluatedCompositionPlan::evaluate(&base, cycle_input),
            Err(CompositionPlanError::ImageGraph(
                ImageGraphError::CurrentCycle { .. }
            )) | Err(CompositionPlanError::CurrentCycle { .. })
        ));
    }

    #[test]
    fn base_group_and_one_below_mattes_use_m2_domains() {
        let (base, composition, master, racks) = grouped_fixture();
        let mattes = vec![
            LayerMatte {
                enabled: true,
                input: ImageInput::OneBelow,
                ..LayerMatte::default()
            },
            LayerMatte::default(),
            LayerMatte::default(),
            LayerMatte {
                enabled: true,
                input: ImageInput::GroupOutput {
                    group_id: group_id(10),
                },
                ..LayerMatte::default()
            },
        ];
        let input = CompositionPlanInput::new(&composition, &master, &racks)
            .with_layer_mattes(&mattes, false);
        let advanced = advanced(EvaluatedCompositionPlan::evaluate(&base, input).unwrap());
        assert!(base.image_routing().mattes().is_empty());
        let layer4 = advanced
            .image_taps()
            .iter()
            .find(|tap| {
                tap.consumer
                    == ImageTapConsumer::LayerMatte {
                        layer_id: layer_id(4),
                    }
            })
            .unwrap();
        assert_eq!(
            layer4.resolved,
            PlannedImageSource::OneBelow(VisualScopeId::Group(group_id(10)))
        );
        let group_tap = advanced
            .image_taps()
            .iter()
            .find(|tap| {
                tap.consumer
                    == ImageTapConsumer::LayerMatte {
                        layer_id: layer_id(1),
                    }
            })
            .unwrap();
        assert_eq!(
            group_tap.resolved,
            PlannedImageSource::Scope(VisualScopeId::Group(group_id(10)))
        );

        let wrong_count = CompositionPlanInput::new(&composition, &master, &racks)
            .with_layer_mattes(&mattes[..3], false);
        assert!(matches!(
            EvaluatedCompositionPlan::evaluate(&base, wrong_count),
            Err(CompositionPlanError::LayerMatteCount {
                count: 3,
                layers: 4
            })
        ));
    }

    #[test]
    fn base_program_history_is_one_previous_edge_in_the_creative_ledger() {
        let mut warm_base = base(&[1], &[]);
        warm_base
            .attach_image_routing(
                [LayerMatte {
                    enabled: true,
                    input: ImageInput::ProgramHistory,
                    ..LayerMatte::default()
                }],
                true,
            )
            .unwrap();
        let composition = legacy_composition(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let racks = legacy_racks(&[1]);
        let warm = advanced(plan(&warm_base, &composition, &master, &racks).unwrap());
        assert_eq!(warm.graph().previous_taps, 1);
        assert_eq!(
            warm.image_taps()[0].resolved,
            PlannedImageSource::ProgramHistory
        );
        assert_eq!(
            warm.resources().retained_surface_layers,
            crate::visual_rack::BASE_CREATIVE_SURFACE_LAYERS
                + ADVANCED_RACK_SURFACE_LAYERS
                + ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS
                + 2
        );
        assert_eq!(
            warm.resources().rgba16_surface_layers,
            crate::visual_rack::BASE_CREATIVE_SURFACE_LAYERS + ADVANCED_RACK_SURFACE_LAYERS + 2
        );
        assert_eq!(
            warm.resources().compat8_surface_layers,
            ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS
        );

        let mut cold = base(&[1], &[]);
        cold.attach_image_routing(
            [LayerMatte {
                enabled: true,
                input: ImageInput::ProgramHistory,
                ..LayerMatte::default()
            }],
            false,
        )
        .unwrap();
        let cold = advanced(plan(&cold, &composition, &master, &racks).unwrap());
        assert_eq!(cold.graph().previous_taps, 1);
        assert_eq!(
            cold.image_taps()[0].resolved,
            PlannedImageSource::ProgramHistory
        );
        assert!(matches!(
            cold.diagnostics()[0],
            CompositionPlanDiagnostic::ProgramHistoryUninitialized { .. }
        ));
        assert_eq!(warm.topology_signature(), cold.topology_signature());
    }

    #[test]
    fn resource_rejection_is_atomic_and_plan_owns_immutable_payload() {
        let base = base(&[1], &[]);
        let composition = legacy_composition(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1]);
        let node = racks[0]
            .1
            .push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        let composition_before = composition.clone();
        let racks_before = racks.clone();
        let mut input = CompositionPlanInput::new(&composition, &master, &racks);
        input.resource_limits.max_creative_bytes = 1;
        assert!(matches!(
            EvaluatedCompositionPlan::evaluate(&base, input),
            Err(CompositionPlanError::Resource(
                ResourcePreflightError::CreativeMemoryBudget { .. }
            ))
        ));
        assert_eq!(composition, composition_before);
        assert_eq!(racks, racks_before);

        let owned = advanced(plan(&base, &composition, &master, &racks).unwrap());
        let signature = owned.topology_signature();
        racks[0].1.get_mut(node).unwrap().wet = 0.0;
        assert_eq!(owned.topology_signature(), signature);
        assert!(matches!(
            owned.layers()[0].execution.steps()[2],
            EvaluatedScopeStep::CollisionRack { .. }
        ));
    }

    #[test]
    fn topology_signature_is_deterministic_and_uses_sparse_hostile_ids() {
        let ids = [u64::MAX - 2, u64::MAX - 1];
        let base = base(&ids, &[]);
        let composition = legacy_composition(&ids);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&ids);
        racks[0]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let first = plan(&base, &composition, &master, &racks).unwrap();
        let second = plan(&base, &composition, &master, &racks).unwrap();
        assert_eq!(first.topology_signature(), second.topology_signature());
    }

    #[test]
    fn current_prelocal_self_and_reciprocal_routes_are_source_inputs_not_cycles() {
        let base_plan = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        for (consumer, donor) in [(0, 2), (1, 1)] {
            push_image_mask(
                &mut racks[consumer].1,
                ResolvedImageSource::SelectedLayer {
                    layer_id: layer_id(donor),
                    saved_position: saved_position(0),
                    stage: LayerImageStage::PreLocalEffects,
                },
                EdgeTiming::CurrentFrame,
            );
        }
        let planned = advanced(plan(&base_plan, &composition, &master, &racks).unwrap());
        assert_eq!(planned.graph().current_taps, 0);
        assert_eq!(planned.graph().previous_taps, 0);
        assert_eq!(planned.image_taps().len(), 2);
        assert_eq!(
            planned.resources().rgba16_surface_layers,
            crate::visual_rack::BASE_CREATIVE_SURFACE_LAYERS + ADVANCED_RACK_SURFACE_LAYERS + 2
        );
        assert!(planned.image_taps().iter().all(|tap| matches!(
            tap.resolved,
            PlannedImageSource::SelectedLayer {
                stage: LayerImageStage::PreLocalEffects,
                ..
            }
        )));

        let prelocal_mattes = [
            LayerMatte {
                enabled: true,
                input: ImageInput::SelectedLayer {
                    layer_id: layer_id(2),
                    stage: LayerImageStage::PreLocalEffects,
                },
                ..LayerMatte::default()
            },
            LayerMatte {
                enabled: true,
                input: ImageInput::SelectedLayer {
                    layer_id: layer_id(1),
                    stage: LayerImageStage::PreLocalEffects,
                },
                ..LayerMatte::default()
            },
        ];
        let clean_racks = legacy_racks(&[1, 2]);
        let input = CompositionPlanInput::new(&composition, &master, &clean_racks)
            .with_layer_mattes(&prelocal_mattes, false);
        let mattes = advanced(EvaluatedCompositionPlan::evaluate(&base_plan, input).unwrap());
        assert_eq!(mattes.graph().current_taps, 0);
        assert_eq!(mattes.image_taps().len(), 2);

        let mut duplicate_racks = legacy_racks(&[1, 2]);
        for _ in 0..2 {
            push_image_mask(
                &mut duplicate_racks[0].1,
                ResolvedImageSource::SelectedLayer {
                    layer_id: layer_id(2),
                    saved_position: saved_position(1),
                    stage: LayerImageStage::PreLocalEffects,
                },
                EdgeTiming::CurrentFrame,
            );
        }
        let duplicate =
            advanced(plan(&base_plan, &composition, &master, &duplicate_racks).unwrap());
        assert_eq!(duplicate.image_taps().len(), 2);
        assert_eq!(duplicate.graph().current_taps, 0);
        assert_eq!(
            duplicate.resources().rgba16_surface_layers,
            crate::visual_rack::BASE_CREATIVE_SURFACE_LAYERS + ADVANCED_RACK_SURFACE_LAYERS + 1,
            "two consumers of one PreLocal donor share its prepared surface"
        );

        let mut constrained = CompositionPlanInput::new(&composition, &master, &duplicate_racks);
        constrained.resource_limits.max_creative_bytes =
            duplicate.resources().creative_bytes.saturating_sub(1);
        assert!(matches!(
            EvaluatedCompositionPlan::evaluate(&base_plan, constrained),
            Err(CompositionPlanError::Resource(
                ResourcePreflightError::CreativeMemoryBudget { .. }
            ))
        ));
    }

    #[test]
    fn reciprocal_postlocal_routes_remain_a_current_cycle() {
        let base_plan = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        for (consumer, donor) in [(0, 2), (1, 1)] {
            push_image_mask(
                &mut racks[consumer].1,
                ResolvedImageSource::SelectedLayer {
                    layer_id: layer_id(donor),
                    saved_position: saved_position(0),
                    stage: LayerImageStage::PostLocalEffects,
                },
                EdgeTiming::CurrentFrame,
            );
        }
        assert!(matches!(
            plan(&base_plan, &composition, &master, &racks),
            Err(CompositionPlanError::ImageGraph(
                ImageGraphError::CurrentCycle { .. }
            )) | Err(CompositionPlanError::CurrentCycle { .. })
        ));
    }

    #[test]
    fn logical_missing_bindings_do_not_consume_the_dependency_cap() {
        let ids: Vec<_> = (1..=10).collect();
        let base_plan = base(&ids, &[]);
        let composition = legacy_composition(&ids);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = Vec::new();
        for id in &ids {
            let mut rack = RuntimeVisualRack::empty();
            for route in 0..8 {
                push_image_mask(
                    &mut rack,
                    ResolvedImageSource::MissingSelectedLayer {
                        saved_position: saved_position((*id as u32 - 1) * 8 + route),
                        stage: LayerImageStage::PreLocalEffects,
                    },
                    EdgeTiming::CurrentFrame,
                );
            }
            racks.push((layer_id(*id), rack));
        }
        let advanced = advanced(plan(&base_plan, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.image_taps().len(), 80);
        assert_eq!(advanced.diagnostics().len(), 80);
        assert_eq!(advanced.graph().current_taps, 0);
        assert_eq!(advanced.graph().previous_taps, 0);
    }

    #[test]
    fn previous_scope_tap_charges_history_and_distinct_staging() {
        let base_plan = base(&[1], &[]);
        let composition = legacy_composition(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1]);
        push_image_mask(
            &mut racks[0].1,
            ResolvedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                saved_position: saved_position(0),
                stage: LayerImageStage::PreLocalEffects,
            },
            EdgeTiming::PreviousFrame,
        );
        let advanced = advanced(plan(&base_plan, &composition, &master, &racks).unwrap());
        assert_eq!(advanced.graph().previous_taps, 1);
        assert_eq!(
            advanced.resources().rgba16_surface_layers,
            crate::visual_rack::BASE_CREATIVE_SURFACE_LAYERS + ADVANCED_RACK_SURFACE_LAYERS + 2
        );
        assert_eq!(
            advanced.resources().compat8_surface_layers,
            ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS
        );
    }
}
