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
use crate::effects::params::{RefreshGardenGate, TemporalParams};
use crate::effects::EffectUniforms;
use crate::image_routing::{
    ImageInput, ImageRouteCycle, ImageRouteDiagnostic, LayerImageStage, LayerMatte,
    MissingImageInput, StableLayerId,
};
use crate::motion::{
    resolve_motion_source, FieldColliderAdmission, FieldColliderDiagnostic, FieldColliderInput,
    FieldColliderMode, FieldColliderResourcePlan, MotionBoundaryMode, MotionDeviceLimits,
    MotionDonor, MotionField, MotionFieldOrigin, MotionGrid, MotionParams, MotionPlanError,
    MotionResourcePlan, MotionScopeResourceRequest, MotionSourceDecision, MotionSourceDiagnostic,
    MOTION_ALGORITHM_VERSION,
};
use crate::performance::SavedLayerPosition;
use crate::renderer::compositor::{MatteChannelCode, ResolvedMatteParams};
use crate::renderer::rack::{CollisionRackPlan, RackCompileError};
use crate::spatial::{EffectPassUniforms, SpatialGpuUniforms, SpatialTransform};
use crate::symmetry::{RuntimeSymmetryParams, SymmetryNodeDomain, SYMMETRY_MOTION_SLOTS};
use crate::temporal::{RefreshGardenMatteRoute, RefreshGardenMotionRoute};
use crate::visual_rack::{
    CreativeResourceLimits, CreativeResourcePlan, EdgeTiming, GroupId, ImageDependency,
    ImageDependencyGraph, ImageGraphError, ImageGraphMode, ImageGraphPlan, LegacyRackScope,
    NodeBlend, NodeId, NodeKindTag, ResidualResourceError, ResidualResourceLimits,
    ResidualResourcePlan, ResidualResourceRequest, ResolvedImageSource, ResolvedImageTap,
    ResourcePreflightError, RouteCaptureError, RuntimeImageMatte, RuntimeMaskParams,
    RuntimeRackError, RuntimeVisualNode, RuntimeVisualNodeKind, RuntimeVisualRack, VisualNodeKind,
    VisualRack, VisualScopeId, ADVANCED_PROGRAM_HISTORY_STAGING_LAYERS,
    ADVANCED_RACK_SURFACE_LAYERS, ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS, MAX_IMAGE_DEPENDENCIES,
    MAX_LOGICAL_TEXTURE_LOOKUPS_PER_FRAME, MAX_TEXTURE_SAMPLES_PER_FRAME, RACK_PRIMARY_ROUTE_SLOT,
    RESIDUAL_DETAIL_SLOT, RESIDUAL_ROUTE_SLOTS, RESIDUAL_STRUCTURE_SLOT,
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
    /// Whether this session has an admitted gesture canvas.
    ///
    /// `false` is the pre-gesture default: a route to the canvas resolves to a
    /// transparent field with a named diagnostic, exactly as a missing donor
    /// does, and never silently rebinds to a layer or a group. The canvas is a
    /// master-scope singleton, so this is a bare admission fact rather than an
    /// identity — there is nothing to name and no position to preserve.
    pub gesture_canvas_admitted: bool,
    /// Whether the programme tap holds a committed frame.
    ///
    /// `false` is the pre-first-frame default: a route to the tap resolves to
    /// a transparent field with a named diagnostic, exactly as an unadmitted
    /// gesture canvas does, and never silently rebinds to a layer or a group.
    /// The tap is a master-scope singleton, so this is a bare admission fact
    /// rather than an identity — there is nothing to name and no position to
    /// preserve. Live answers from the renderer's published-tap validity;
    /// export admits unconditionally because its tap surface exists for the
    /// whole job and frame zero reads its defined-transparent contents (the
    /// `with_gesture_canvas(true)` precedent).
    pub program_tap_admitted: bool,
    /// The host Study library. A Study node's digest resolves here at plan
    /// time; `None` (the default) or an absent digest plans an inert pass
    /// with the diagnostic surfaced host-side, never a fallback onto another
    /// document. Live and export supply the same library so both sides
    /// resolve identically.
    pub studies: Option<&'a crate::study_eval::StudyProgramLibrary>,
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
            gesture_canvas_admitted: false,
            program_tap_admitted: false,
            studies: None,
        }
    }

    /// Attach the host Study library for digest resolution.
    pub const fn with_studies(
        mut self,
        studies: &'a crate::study_eval::StudyProgramLibrary,
    ) -> Self {
        self.studies = Some(studies);
        self
    }

    /// Declare whether an admitted gesture canvas backs this frame. Live and
    /// export must supply the same fact so a routed donor plans identically on
    /// both sides.
    pub const fn with_gesture_canvas(mut self, admitted: bool) -> Self {
        self.gesture_canvas_admitted = admitted;
        self
    }

    /// Declare whether the programme tap holds a committed frame. Live and
    /// export must supply the same admission law so a routed tap plans
    /// identically on both sides once the first frame commits.
    pub const fn with_program_tap(mut self, admitted: bool) -> Self {
        self.program_tap_admitted = admitted;
        self
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
    /// One dedicated eight-texture Symmetry Field pass, at its exact authored
    /// rack position. It owns its own bind layout, so it can never share an
    /// ordinary rack segment: the segmenter flushes what came before, emits
    /// this step, and resumes segmentation behind it.
    SymmetryField { plan: EvaluatedSymmetryFieldPlan },
    /// One dedicated Study interpreter pass, at its exact authored rack
    /// position — the same lift as the Symmetry Field: it binds the
    /// committed clean-history array and owns its own uniform layout, so it
    /// can never share an ordinary rack segment.
    StudyField { plan: EvaluatedStudyPlan },
    /// One dedicated Scan Processor pass, at its exact authored rack
    /// position. It is lifted for a stronger reason than the two above: it
    /// is instanced ribbon geometry accumulated additively into its own
    /// transient, not a fullscreen triangle, so no ordinary segment could
    /// ever encode it.
    ScanProcessorField { plan: EvaluatedScanProcessorPlan },
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
            // A dedicated step is still one authored node, so it reports its
            // own kind rather than hiding behind a segment.
            Self::SymmetryField { .. } => Some(NodeKindTag::Symmetry),
            Self::StudyField { .. } => Some(NodeKindTag::Study),
            Self::ScanProcessorField { .. } => Some(NodeKindTag::ScanProcessor),
            Self::MaterializeSpatial { .. }
            | Self::CollisionRack { .. }
            | Self::GroupMatte { .. } => None,
        }
    }
}

/// The evaluated payload of one dedicated Symmetry Field pass.
///
/// It carries the authored node controls an ordinary rack pass would observe,
/// the route-resolved parameters, and the motion field each armed motion slot
/// actually resolved to. Nothing here is frame-local readiness: a donor that
/// fails to bind changes the executor's neutral-view choice, never this plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedSymmetryFieldPlan {
    pub node_id: NodeId,
    /// The authored node's stable sector-table domain, resolved once here from
    /// its owning scope and its stable node id. The dedicated executor reads
    /// the table through this rather than re-deriving an owner it cannot see.
    pub domain: SymmetryNodeDomain,
    pub enabled: bool,
    pub wet: f32,
    pub blend: NodeBlend,
    pub params: RuntimeSymmetryParams,
    /// The admitted motion field per motion slot. `None` is an incomplete
    /// pair — an unarmed slot, an unselected or tombstoned donor, or a donor
    /// whose field was never admitted — and the executor binds the neutral
    /// vector/gate views for it. The planner has already published a
    /// `MotionPlanDiagnostic` naming the slot in every case but "unarmed".
    pub motion_field_slots: [Option<u8>; SYMMETRY_MOTION_SLOTS],
    /// This one pass's declared constant resource table.
    pub resources: SymmetryFieldResourcePlan,
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
    /// One authored route slot of one rack node. Single-route kinds always
    /// occupy slot 0; a multi-route kind names each slot explicitly, because
    /// the consumer identity is what orders, hashes, and binds a tap. Without
    /// the slot, two routes on one node would collapse to one identity and the
    /// second donor would silently resolve onto the first one's surface.
    RackNode {
        scope: VisualScopeId,
        node_id: NodeId,
        /// Which of the node's fixed image routes this tap is. Slot index is
        /// route identity: a single-route node always uses slot 0, while a
        /// multi-route node addresses each route by its own permanent slot.
        /// Without it two taps from one node compare equal, `tap_index_for_-
        /// consumer` binds the first route twice, and a slot-1 route change
        /// leaves the topology signature unmoved.
        slot: u8,
    },
    GroupMatte {
        group_id: GroupId,
    },
    LayerMatte {
        layer_id: StableLayerId,
    },
    RefreshGardenMatte,
}

impl ImageTapConsumer {
    pub const fn scope(self) -> VisualScopeId {
        match self {
            Self::RackNode { scope, .. } => scope,
            Self::GroupMatte { group_id } => VisualScopeId::Group(group_id),
            Self::LayerMatte { layer_id } => VisualScopeId::Layer(layer_id),
            Self::RefreshGardenMatte => VisualScopeId::Master,
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
    /// The etched gesture field, presented as a premultiplied donor image.
    ///
    /// It is a *producer with no scope*: nothing in the composition graph makes
    /// it, so it contributes no dependency, no ordering edge, and no retained
    /// tap surface. That is what keeps a same-frame route to it from ever
    /// closing a cycle, including from the master scope that owns it.
    GestureCanvas,
    /// The finished programme at N-1: the pre-blackout opaque audience image
    /// published at the frame-acceptance decision.
    ///
    /// Like the gesture canvas it is a *producer with no scope*: nothing in
    /// the composition graph makes it — the copy is published after the frame
    /// encoder is accepted — so it contributes no dependency, no ordering
    /// edge, and no retained tap surface, and no same-frame route to it can
    /// close a cycle. It also has no N-1 parity of its own: the tap *is* the
    /// N-1 image, so both timings read the same committed copy rather than
    /// quietly claiming a history pair nothing writes.
    ProgramTap,
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
    /// A route named the gesture canvas but no canvas is admitted. The tap
    /// resolves transparent and stays visible as this diagnostic; it is never
    /// repointed at some other producer.
    GestureCanvasUnavailable {
        consumer: ImageTapConsumer,
    },
    /// A route named the programme tap before the first committed frame. The
    /// tap resolves transparent and stays visible as this diagnostic; it is
    /// never repointed at some other producer.
    ProgramTapUnavailable {
        consumer: ImageTapConsumer,
    },
    RefreshGardenMatteNotSelected,
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
    RefreshGardenMotionNotSelected,
    MissingRefreshGardenMotion {
        saved_position: SavedLayerPosition,
    },
    RefreshGardenMotionUnavailable,
    /// An armed Symmetry Field motion slot names no donor at all.
    SymmetryMotionNotSelected {
        scope: VisualScopeId,
        node_id: NodeId,
        slot: u8,
    },
    /// An armed Symmetry Field motion slot holds a tombstoned donor. The saved
    /// position is retained for the operator, never to rebind: whatever layer
    /// later occupies that position does not inherit the route.
    MissingSymmetryMotion {
        scope: VisualScopeId,
        node_id: NodeId,
        slot: u8,
        saved_position: SavedLayerPosition,
    },
    /// An armed Symmetry Field motion slot selected a live donor, but this
    /// frame carries no motion input at all, so no primitive field exists to
    /// request. The slot binds neutral vector/gate views.
    SymmetryMotionUnavailable {
        scope: VisualScopeId,
        node_id: NodeId,
        slot: u8,
    },
    /// An enabled Field Collider did not survive admission. The block delegates
    /// to the exact M4 recipe; the typed cause travels so the operator can see
    /// which slot, or which law, refused it.
    FieldCollider {
        recipient: StableLayerId,
        diagnostic: FieldColliderDiagnostic,
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
    pub required_as_garden_signal: bool,
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
    pub required_as_garden_signal: bool,
    pub donor_scope: Option<VisualScopeId>,
    pub donor_field_slot: Option<u8>,
    pub transplant_admitted: bool,
    /// An admitted Field Collider owns this scope's advection field. The
    /// single-donor recipe is *parked*, not erased: `params.transplant` still
    /// carries its authored donor, amount, carrier, confidence, refresh, decay,
    /// and occlusion verbatim, and disabling the collider resumes it exactly.
    pub collider_admitted: bool,
    pub codec: MotionCodecFrameFacts,
}

impl EvaluatedMotionScopePlan {
    /// The canonical field actually admitted by this scope. An admitted
    /// Faraday transplant replaces the recipient's own field with its donor;
    /// routed consumers must observe the same choice as motion rendering.
    pub const fn admitted_field_slot(self) -> Option<u8> {
        if self.transplant_admitted && !self.collider_admitted {
            self.donor_field_slot
        } else {
            // A collider recipient has no single donor to substitute, so a
            // routed *primitive* consumer observes this scope's own field. The
            // derived collided field is not a primitive field and deliberately
            // has no primitive slot: it is reached through
            // `AdvancedMotionPlan::collider`, never through this accessor.
            self.field_slot
        }
    }
}

/// The immutable evaluated plan of the one admitted Field Collider.
///
/// Live and offline rendering consume this exact value; there is no
/// export-only collider path. `output_slot` appends *after* every primitive
/// field slot, and the derived attachment it names is an internal executor
/// value: it is never added to `CodecMotionProduct`, the live codec field
/// cache, or export codec acquisition, all three of which describe primitive
/// acquisition only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluatedFieldColliderPlan {
    pub output_slot: u8,
    pub recipient_scope: VisualScopeId,
    pub input_a_scope: VisualScopeId,
    pub input_a_slot: u8,
    pub input_b_scope: VisualScopeId,
    pub input_b_slot: u8,
    /// The recipient's own grid. The derived field is indexed in composition
    /// output UV and carries recipient-local vectors, so the Faraday apply pass
    /// consumes it under identity transforms.
    pub output_grid: MotionGrid,
    pub algorithm_version: u16,
    pub mode: FieldColliderMode,
    pub boundary: MotionBoundaryMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EvaluatedRefreshGardenSignalPlan {
    #[default]
    Inline,
    Matte {
        valid: bool,
    },
    Motion {
        layer_id: Option<StableLayerId>,
        field_slot: Option<u8>,
        valid: bool,
    },
}

impl EvaluatedRefreshGardenSignalPlan {
    pub const fn is_routed(self) -> bool {
        matches!(self, Self::Matte { .. } | Self::Motion { .. })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionFrameBudget {
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_samples_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefreshGardenResourcePlan {
    pub full_frame_passes: u32,
    pub low_resolution_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_operations_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
}

fn refresh_garden_resource_plan(
    signal: EvaluatedRefreshGardenSignalPlan,
) -> RefreshGardenResourcePlan {
    if !signal.is_routed() {
        return RefreshGardenResourcePlan::default();
    }
    RefreshGardenResourcePlan {
        full_frame_passes: 1,
        low_resolution_passes: u32::from(matches!(
            signal,
            EvaluatedRefreshGardenSignalPlan::Motion { valid: true, .. }
        )),
        logical_texture_lookups_per_pixel: 3,
        // Current and signal each use one operation. Advanced feedback uses
        // four explicit covered-color textureLoad operations.
        texture_operations_per_pixel: 6,
        max_sampled_textures_in_pass: 3,
    }
}

/// The constant resource table of the dedicated Symmetry Field pass.
///
/// Modelled on [`RefreshGardenResourcePlan`]: a step that is not a rack node
/// pass declares its own table rather than widening `NodeResourceBudget`. The
/// per-pixel terms are informational here — the authored node is still charged
/// once through the ordinary rack ledger by `capture_budget_rack`, so adding
/// them again in `resource_preflight` would double count. The simultaneous
/// binding count is the term this plan actually gates on, and it is checked
/// against the device's own reported ceiling without the fixed rack layout's
/// `.min(MAX_SAMPLED_TEXTURES_PER_PASS)` clamp.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SymmetryFieldResourcePlan {
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_operations_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
    /// One dynamic-offset uniform record per encoded pass.
    pub uniform_bytes: u32,
}

/// The table for `passes` dedicated Symmetry Field steps. Dormant with none.
///
/// Every term is read from the frozen node descriptor rather than restated, so
/// the step's table and the rack ledger can never disagree about one node's
/// cost. Passes and per-pixel work accumulate; simultaneous bindings do not —
/// each pass owns its own bind group, so the frame's requirement is the widest
/// single pass, exactly as `MotionFrameBudget` and `RefreshGardenResourcePlan`
/// treat theirs.
fn symmetry_field_resource_plan(
    passes: u32,
) -> Result<SymmetryFieldResourcePlan, CompositionPlanError> {
    if passes == 0 {
        return Ok(SymmetryFieldResourcePlan::default());
    }
    let descriptor = crate::visual_rack::node_kind_descriptor(NodeKindTag::Symmetry).budget;
    let scale = |per_pass: u8| {
        u32::from(per_pass)
            .checked_mul(passes)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))
    };
    Ok(SymmetryFieldResourcePlan {
        full_frame_passes: scale(descriptor.full_frame_passes)?,
        logical_texture_lookups_per_pixel: scale(descriptor.logical_texture_lookups_per_pixel)?,
        texture_operations_per_pixel: scale(descriptor.texture_samples_per_pixel)?,
        max_sampled_textures_in_pass: u32::from(descriptor.sampled_textures_in_pass),
        uniform_bytes: SYMMETRY_UNIFORM_BYTES.checked_mul(passes).ok_or(
            CompositionPlanError::Resource(ResourcePreflightError::ArithmeticOverflow),
        )?,
    })
}

/// The frozen dynamic-offset uniform record size, restated from
/// `SymmetryGpuUniforms`' compile-time size assertion.
const SYMMETRY_UNIFORM_BYTES: u32 = 1_024;

/// The evaluated payload of one dedicated Study interpreter pass. The digest
/// is opaque authored identity: the renderer resolves it against the host
/// Study library at prepare, and an unresolved digest is an inert pass with
/// a named diagnostic, never a fallback onto another document.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedStudyPlan {
    pub node_id: NodeId,
    pub enabled: bool,
    pub wet: f32,
    pub blend: NodeBlend,
    pub params: crate::visual_rack::StudyRackParams,
    /// The digest-resolved, GPU-encoded program, fixed at plan time so live
    /// and export execute the identical instruction stream. `None` — no
    /// digest authored, or the digest is absent from the library — is an
    /// inert pass.
    pub program: Option<Box<[crate::study_eval::StudyGpuOp]>>,
    /// The resolved document's live instruction count; zero when unresolved.
    pub instruction_count: u32,
    pub resources: StudyFieldResourcePlan,
}

/// The dedicated Study pass's ledger, the `SymmetryFieldResourcePlan` shape:
/// per-pixel terms are declarative (the rack ledger already charges the
/// authored node once), and the simultaneous-binding count is what this plan
/// gates on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StudyFieldResourcePlan {
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_operations_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
    pub uniform_bytes: u32,
}

/// The table for `passes` dedicated Study steps, every term read from the
/// frozen node descriptor so the step's table and the rack ledger can never
/// disagree about one node's cost.
fn study_field_resource_plan(passes: u32) -> Result<StudyFieldResourcePlan, CompositionPlanError> {
    if passes == 0 {
        return Ok(StudyFieldResourcePlan::default());
    }
    let descriptor = crate::visual_rack::node_kind_descriptor(NodeKindTag::Study).budget;
    let scale = |per_pass: u8| {
        u32::from(per_pass)
            .checked_mul(passes)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))
    };
    Ok(StudyFieldResourcePlan {
        full_frame_passes: scale(descriptor.full_frame_passes)?,
        logical_texture_lookups_per_pixel: scale(descriptor.logical_texture_lookups_per_pixel)?,
        texture_operations_per_pixel: scale(descriptor.texture_samples_per_pixel)?,
        max_sampled_textures_in_pass: u32::from(descriptor.sampled_textures_in_pass),
        uniform_bytes: STUDY_FIELD_UNIFORM_BYTES.checked_mul(passes).ok_or(
            CompositionPlanError::Resource(ResourcePreflightError::ArithmeticOverflow),
        )?,
    })
}

/// One pass's uniform charge: the 64-byte frame block plus the 8,192-byte
/// bounded instruction buffer, both compile-time asserted in
/// `renderer/study.rs`.
const STUDY_FIELD_UNIFORM_BYTES: u32 = 64 + 8_192;

/// The evaluated payload of one dedicated Scan Processor pass — the tree's
/// first non-fullscreen-triangle step. The node reads only its carrier, so
/// there is no route, no donor, and no resolution state here: the sanitized
/// authored params size the instanced draw and everything else is fixed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedScanProcessorPlan {
    pub node_id: NodeId,
    pub enabled: bool,
    pub wet: f32,
    pub blend: NodeBlend,
    pub params: crate::scan_processor::ScanProcessorParams,
    pub resources: ScanProcessorResourcePlan,
}

/// The dedicated Scan Processor ledger, the `SymmetryFieldResourcePlan`
/// shape plus two terms no prior pass owns: a named vertex budget (this is
/// the one pass in the tree with geometry) and the shared full-frame
/// `Rgba16Float` transient the ribbons accumulate into, charged byte-exactly
/// at 8 bytes per output pixel while any step exists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanProcessorResourcePlan {
    pub full_frame_passes: u32,
    pub logical_texture_lookups_per_pixel: u32,
    pub texture_operations_per_pixel: u32,
    pub max_sampled_textures_in_pass: u32,
    pub uniform_bytes: u32,
    /// Instanced ribbon vertices summed across every emitted step. Each step
    /// is individually refused one vertex over
    /// [`crate::scan_processor::MAX_SCAN_PROCESSOR_VERTICES`] at lift time;
    /// this field carries the frame total for the ledger and its tests.
    pub vertices: u32,
    /// One shared accumulator, not one per node: every scan pass in the frame
    /// clears and reuses it, so the charge does not scale with step count.
    pub transient_bytes: u64,
}

/// The table for the emitted Scan Processor steps. Dormant with none.
fn scan_processor_resource_plan(
    passes: u32,
    vertices: u32,
    output: [u32; 2],
) -> Result<ScanProcessorResourcePlan, CompositionPlanError> {
    if passes == 0 {
        return Ok(ScanProcessorResourcePlan::default());
    }
    let descriptor = crate::visual_rack::node_kind_descriptor(NodeKindTag::ScanProcessor).budget;
    let scale = |per_pass: u8| {
        u32::from(per_pass)
            .checked_mul(passes)
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))
    };
    let transient_bytes = u64::from(output[0])
        .checked_mul(u64::from(output[1]))
        .and_then(|pixels| pixels.checked_mul(8))
        .ok_or(CompositionPlanError::Resource(
            ResourcePreflightError::ArithmeticOverflow,
        ))?;
    Ok(ScanProcessorResourcePlan {
        full_frame_passes: scale(descriptor.full_frame_passes)?,
        logical_texture_lookups_per_pixel: scale(descriptor.logical_texture_lookups_per_pixel)?,
        texture_operations_per_pixel: scale(descriptor.texture_samples_per_pixel)?,
        max_sampled_textures_in_pass: u32::from(descriptor.sampled_textures_in_pass),
        uniform_bytes: SCAN_PROCESSOR_UNIFORM_BYTES.checked_mul(passes).ok_or(
            CompositionPlanError::Resource(ResourcePreflightError::ArithmeticOverflow),
        )?,
        vertices,
        transient_bytes,
    })
}

/// One pass's dynamic-offset uniform record, restated from the compile-time
/// size assertion in `scan_processor.rs`.
const SCAN_PROCESSOR_UNIFORM_BYTES: u32 = 128;

/// Count the dedicated Scan Processor steps the segmenter actually emitted,
/// and sum their authored vertex requests.
fn count_scan_processor_steps(
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    master: &EvaluatedMasterScopePlan,
) -> Result<(u32, u32), CompositionPlanError> {
    let mut passes = 0_u32;
    let mut vertices = 0_u32;
    let mut tally = |execution: &EvaluatedScopeExecution| -> Result<(), CompositionPlanError> {
        for step in execution.steps() {
            if let EvaluatedScopeStep::ScanProcessorField { plan } = step {
                passes = passes.checked_add(1).ok_or(CompositionPlanError::Resource(
                    ResourcePreflightError::ArithmeticOverflow,
                ))?;
                vertices = vertices.checked_add(plan.params.vertex_count()).ok_or(
                    CompositionPlanError::Resource(ResourcePreflightError::ArithmeticOverflow),
                )?;
            }
        }
        Ok(())
    };
    for layer in layers {
        tally(&layer.execution)?;
    }
    for group in groups {
        tally(&group.execution)?;
    }
    tally(&master.execution)?;
    Ok((passes, vertices))
}

fn validate_scan_processor_textures(
    scan_processor: ScanProcessorResourcePlan,
    limits: CreativeResourceLimits,
) -> Result<(), CompositionPlanError> {
    validate_dedicated_sampled_textures(scan_processor.max_sampled_textures_in_pass, limits)
}

/// Plan-visible Study identity: every emitted step's node, digest, and
/// resolution state. This feeds the advanced topology signature, so a
/// document assignment — or a library insert that resolves a previously
/// missing digest — re-prepares the renderer and re-uploads the program
/// arena; without it the executor would keep serving a stale program.
fn study_identity_hash(
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    master: &EvaluatedMasterScopePlan,
) -> u64 {
    let mut hash = FNV_OFFSET;
    let mut tally = |execution: &EvaluatedScopeExecution| {
        for step in execution.steps() {
            let EvaluatedScopeStep::StudyField { plan } = step else {
                continue;
            };
            hash = hash_value(hash, 0x5354_5544_5949_4400); // "STUDYID"
            hash = hash_value(hash, plan.node_id.get());
            match plan.params.document_digest {
                None => hash = hash_value(hash, 0),
                Some(digest) => {
                    hash = hash_value(hash, 1);
                    for chunk in digest.chunks_exact(8) {
                        let word = u64::from_le_bytes(chunk.try_into().expect("eight-byte chunks"));
                        hash = hash_value(hash, word);
                    }
                }
            }
            hash = hash_value(hash, u64::from(plan.program.is_some()));
            hash = hash_value(hash, u64::from(plan.instruction_count));
        }
    };
    for layer in layers {
        tally(&layer.execution);
    }
    for group in groups {
        tally(&group.execution);
    }
    tally(&master.execution);
    hash
}

/// Count the dedicated Study steps the segmenter actually emitted.
fn count_study_field_steps(
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    master: &EvaluatedMasterScopePlan,
) -> Result<u32, CompositionPlanError> {
    let mut count = 0_u32;
    let mut tally = |execution: &EvaluatedScopeExecution| -> Result<(), CompositionPlanError> {
        for step in execution.steps() {
            if matches!(step, EvaluatedScopeStep::StudyField { .. }) {
                count = count.checked_add(1).ok_or(CompositionPlanError::Resource(
                    ResourcePreflightError::ArithmeticOverflow,
                ))?;
            }
        }
        Ok(())
    };
    for layer in layers {
        tally(&layer.execution)?;
    }
    for group in groups {
        tally(&group.execution)?;
    }
    tally(&master.execution)?;
    Ok(count)
}

/// Admit the dedicated pass's simultaneous-binding count.
///
/// Two independent ceilings, both required. The first is this project's own
/// dedicated-pass policy, `MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS`. The second
/// is the device's reported capability, read **raw**: a dedicated pass owns its
/// own bind-group layout, so the fixed Collision Rack layout's
/// `.min(MAX_SAMPLED_TEXTURES_PER_PASS)` clamp — which governs three-texture
/// rack passes and appears twice in `resource_preflight` — must never be
/// applied here. Equally, nothing here may relax that clamp: the two ceilings
/// are independent and neither was raised to admit the other.
fn validate_symmetry_field_textures(
    symmetry_field: SymmetryFieldResourcePlan,
    limits: CreativeResourceLimits,
) -> Result<(), CompositionPlanError> {
    validate_dedicated_sampled_textures(symmetry_field.max_sampled_textures_in_pass, limits)
}

fn validate_study_field_textures(
    study_field: StudyFieldResourcePlan,
    limits: CreativeResourceLimits,
) -> Result<(), CompositionPlanError> {
    validate_dedicated_sampled_textures(study_field.max_sampled_textures_in_pass, limits)
}

/// One predicate for every dedicated pass's simultaneous-binding admission,
/// so the Symmetry and Study validators cannot drift apart.
fn validate_dedicated_sampled_textures(
    requested: u32,
    limits: CreativeResourceLimits,
) -> Result<(), CompositionPlanError> {
    if requested > crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS {
        return Err(CompositionPlanError::Resource(
            ResourcePreflightError::SampledTextureLimit {
                requested,
                limit: crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS,
            },
        ));
    }
    if requested > limits.max_sampled_textures_per_shader_stage {
        return Err(CompositionPlanError::Resource(
            ResourcePreflightError::SampledTextureLimit {
                requested,
                limit: limits.max_sampled_textures_per_shader_stage,
            },
        ));
    }
    Ok(())
}

/// Count the dedicated Symmetry Field steps the segmenter actually emitted.
fn count_symmetry_field_steps(
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    master: &EvaluatedMasterScopePlan,
) -> Result<u32, CompositionPlanError> {
    let mut count = 0_u32;
    let mut tally = |execution: &EvaluatedScopeExecution| -> Result<(), CompositionPlanError> {
        for step in execution.steps() {
            if matches!(step, EvaluatedScopeStep::SymmetryField { .. }) {
                count = count.checked_add(1).ok_or(CompositionPlanError::Resource(
                    ResourcePreflightError::ArithmeticOverflow,
                ))?;
            }
        }
        Ok(())
    };
    for layer in layers {
        tally(&layer.execution)?;
    }
    for group in groups {
        tally(&group.execution)?;
    }
    tally(&master.execution)?;
    Ok(count)
}

#[derive(Debug, Clone)]
pub struct AdvancedMotionPlan {
    scopes: Box<[EvaluatedMotionScopePlan]>,
    fields: Box<[EvaluatedMotionFieldPlan]>,
    resources: MotionResourcePlan,
    collider: Option<EvaluatedFieldColliderPlan>,
    collider_resources: FieldColliderResourcePlan,
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

    /// The one admitted Field Collider, or `None` for the exact M4 plan.
    pub const fn collider(&self) -> Option<EvaluatedFieldColliderPlan> {
        self.collider
    }

    /// The byte-exact collider-specific resource delta. All zeros with no
    /// admitted collider, so the exact M4 plan charges nothing.
    pub const fn collider_resources(&self) -> FieldColliderResourcePlan {
        self.collider_resources
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
    /// Exact digest of the validated adjacent-reference proof chain and its
    /// sparse vector payload. Production codec adapters always provide the
    /// product identity digest; fixed values are used only by GPU fixtures.
    pub product_content_sha256: [u8; 32],
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
    /// The sanitized frame copy of the B8 mixing-boundary state.
    mixer: crate::mixing_boundary::BusMixerState,
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
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the reduced block-mean ledger is consumed by the prepared executor"
        )
    )]
    residual_resources: ResidualResourcePlan,
    motion: EvaluatedMotionPlan,
    refresh_garden_signal: EvaluatedRefreshGardenSignalPlan,
    refresh_garden_resources: RefreshGardenResourcePlan,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the dedicated Symmetry Field ledger precedes its executor"
        )
    )]
    symmetry_field_resources: SymmetryFieldResourcePlan,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the dedicated Study ledger is consumed by tests and future preflight surfaces"
        )
    )]
    study_field_resources: StudyFieldResourcePlan,
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the dedicated Scan Processor ledger is consumed by the prepared executor and tests"
        )
    )]
    scan_processor_resources: ScanProcessorResourcePlan,
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

    pub const fn mixer(&self) -> crate::mixing_boundary::BusMixerState {
        self.mixer
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

    /// Byte-exact reduced block-mean working set, deliberately outside the
    /// full-frame layer ledger and reconciled against the prepared executor.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the reduced block-mean ledger is consumed by the prepared executor"
        )
    )]
    pub const fn residual_resources(&self) -> ResidualResourcePlan {
        self.residual_resources
    }

    pub const fn motion(&self) -> &EvaluatedMotionPlan {
        &self.motion
    }

    pub const fn refresh_garden_signal(&self) -> EvaluatedRefreshGardenSignalPlan {
        self.refresh_garden_signal
    }

    pub const fn refresh_garden_resources(&self) -> RefreshGardenResourcePlan {
        self.refresh_garden_resources
    }

    /// The combined constant table of every dedicated Symmetry Field pass this
    /// frame encodes. Default when the frame encodes none.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the dedicated Symmetry Field ledger precedes its executor"
        )
    )]
    pub const fn symmetry_field_resources(&self) -> SymmetryFieldResourcePlan {
        self.symmetry_field_resources
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the dedicated Study ledger is consumed by tests and future preflight surfaces"
        )
    )]
    pub const fn study_field_resources(&self) -> StudyFieldResourcePlan {
        self.study_field_resources
    }

    /// The combined constant table of every dedicated Scan Processor pass
    /// this frame encodes, including the named vertex budget and the shared
    /// transient accumulator's byte charge. Default when the frame encodes
    /// none.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the dedicated Scan Processor ledger is consumed by the prepared executor and tests"
        )
    )]
    pub const fn scan_processor_resources(&self) -> ScanProcessorResourcePlan {
        self.scan_processor_resources
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

    /// Where one scope sits in the composition's back-to-front composite order.
    ///
    /// This is the order `build_block_schedules` drains, so it is the order
    /// scopes should execute in whenever nothing forces otherwise. A grouped
    /// layer ranks inside its own group's root slot, so a group's members stay
    /// contiguous and in member order; the group itself ranks at that same root
    /// slot, immediately after its members. Master ranks last among scopes, and
    /// anything absent from the topology — Program — ranks after that.
    ///
    /// The tuple is deliberately ordered `(root, member)`: comparing it
    /// lexicographically reproduces the composite walk exactly, without a
    /// second flattening pass that could disagree with `root_outputs`.
    fn composite_rank(&self, scope: VisualScopeId) -> (usize, usize) {
        match self.locations.get(&scope).copied() {
            Some(BelowLocation::Root { root_index }) => (root_index, usize::MAX),
            Some(BelowLocation::GroupMember {
                root_index,
                member_index,
                ..
            }) => (root_index, member_index),
            Some(BelowLocation::Master) | None => (usize::MAX, usize::MAX),
        }
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
    /// A resolved Study document performs more per-pixel texture loads than
    /// the node's declared admission budget. The document stays valid ABI;
    /// it simply cannot be admitted under the current resource ceiling —
    /// the over-budget Residual-grid law, never a silent clamp.
    StudyLoadBudget {
        node: NodeId,
        loads: u32,
        limit: u32,
    },
    /// A Scan Processor's authored geometry requests more instanced ribbon
    /// vertices than the named vertex budget. The sanitized ranges cannot
    /// exceed it today, so this is defense in depth with a typed error —
    /// the over-budget Residual-grid law, never a silent clamp.
    ScanProcessorVertexBudget {
        node: NodeId,
        vertices: u32,
        limit: u32,
    },
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
    /// A Residual Counterpoint block-mean working set broke one of its own
    /// independently enforced bounds. Reduced surfaces are never silently
    /// clamped to a smaller grid; the offending bound is named instead.
    Residual(ResidualResourceError),
    /// The reduced block-mean bytes are admissible on their own but push the
    /// composition past the shared creative cap once full-frame and motion
    /// bytes are counted with them.
    ResidualCombinedMemoryBudget {
        bytes: u64,
        limit: u64,
    },
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
            Self::StudyLoadBudget { node, loads, limit } => {
                write!(
                    formatter,
                    "study node {} performs {loads} per-pixel texture loads; the admission budget is {limit}",
                    node.get()
                )
            }
            Self::ScanProcessorVertexBudget {
                node,
                vertices,
                limit,
            } => {
                write!(
                    formatter,
                    "scan processor node {} requests {vertices} ribbon vertices; the vertex budget is {limit}",
                    node.get()
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
            Self::Residual(error) => {
                write!(formatter, "residual resource preflight failed: {error}")
            }
            Self::ResidualCombinedMemoryBudget { bytes, limit } => write!(
                formatter,
                "creative plus residual block-mean resources request {bytes} bytes; limit is {limit}"
            ),
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

/// One armed Symmetry Field motion slot, addressed by its permanent slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SymmetryMotionRequest {
    scope: VisualScopeId,
    node_id: NodeId,
    slot: u8,
    donor: MotionDonor,
}

impl SymmetryMotionRequest {
    fn unselected_diagnostic(self) -> MotionPlanDiagnostic {
        MotionPlanDiagnostic::SymmetryMotionNotSelected {
            scope: self.scope,
            node_id: self.node_id,
            slot: self.slot,
        }
    }

    fn missing_diagnostic(self, saved_position: SavedLayerPosition) -> MotionPlanDiagnostic {
        MotionPlanDiagnostic::MissingSymmetryMotion {
            scope: self.scope,
            node_id: self.node_id,
            slot: self.slot,
            saved_position,
        }
    }

    fn unserved_diagnostic(self) -> MotionPlanDiagnostic {
        match self.donor {
            MotionDonor::None => self.unselected_diagnostic(),
            MotionDonor::Missing { saved_position } => self.missing_diagnostic(saved_position),
            MotionDonor::Selected { .. } => MotionPlanDiagnostic::SymmetryMotionUnavailable {
                scope: self.scope,
                node_id: self.node_id,
                slot: self.slot,
            },
        }
    }
}

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
        let garden = self.base.temporal().originals.garden;
        let temporal_marker_present = self
            .input
            .master_rack
            .iter()
            .any(|node| matches!(node.kind, RuntimeVisualNodeKind::LegacyTemporal));
        let routed_garden_active = temporal_marker_present
            && garden.amount.is_finite()
            && garden.amount > 0.0
            && matches!(
                garden.gate,
                RefreshGardenGate::Matte | RefreshGardenGate::Motion
            );
        if self.is_global_legacy_exact(&flat_ids)
            && motion.is_legacy_exact()
            && !routed_garden_active
        {
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
                &motion,
                self.input.studies,
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
                &motion,
                self.input.studies,
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
                self.input.composition.mixer(),
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
                &motion,
                self.input.studies,
            )?,
            canonical_layers: canonical_layers.into_boxed_slice(),
            canonical_bypass_layers: canonical_bypass_layers.into_boxed_slice(),
            selective_ntsc_layers: selective_ntsc_layers.into_boxed_slice(),
            selective_ntsc_bypass_layers: selective_ntsc_bypass_layers.into_boxed_slice(),
        };

        let below = below_topology(self.input.composition)?;
        let gesture_canvas_admitted = self.input.gesture_canvas_admitted;
        let program_tap_admitted = self.input.program_tap_admitted;
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
                gesture_canvas_admitted,
                program_tap_admitted,
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
                    gesture_canvas_admitted,
                    program_tap_admitted,
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
                    gesture_canvas_admitted,
                    program_tap_admitted,
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
                        gesture_canvas_admitted,
                        program_tap_admitted,
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
            gesture_canvas_admitted,
            program_tap_admitted,
            &mut taps,
            &mut diagnostics,
            &mut dependencies,
            &mut route_edges,
            &mut prefix_constraints,
        )?;

        if routed_garden_active && garden.gate == RefreshGardenGate::Matte {
            match garden.matte_route {
                RefreshGardenMatteRoute::None => {
                    diagnostics.push(CompositionPlanDiagnostic::RefreshGardenMatteNotSelected);
                }
                RefreshGardenMatteRoute::SelectedLayer {
                    layer_id,
                    saved_position,
                    stage,
                } => collect_tap(
                    ImageTapConsumer::RefreshGardenMatte,
                    PlannedImageTapOrigin::Rack(ResolvedImageTap {
                        source: ResolvedImageSource::SelectedLayer {
                            layer_id,
                            saved_position,
                            stage,
                        },
                        timing: EdgeTiming::CurrentFrame,
                    }),
                    &below,
                    &known_scopes,
                    gesture_canvas_admitted,
                    program_tap_admitted,
                    &mut taps,
                    &mut diagnostics,
                    &mut dependencies,
                    &mut route_edges,
                    &mut prefix_constraints,
                )?,
                RefreshGardenMatteRoute::MissingSelectedLayer {
                    saved_position,
                    stage,
                } => collect_tap(
                    ImageTapConsumer::RefreshGardenMatte,
                    PlannedImageTapOrigin::Rack(ResolvedImageTap {
                        source: ResolvedImageSource::MissingSelectedLayer {
                            saved_position,
                            stage,
                        },
                        timing: EdgeTiming::CurrentFrame,
                    }),
                    &below,
                    &known_scopes,
                    gesture_canvas_admitted,
                    program_tap_admitted,
                    &mut taps,
                    &mut diagnostics,
                    &mut dependencies,
                    &mut route_edges,
                    &mut prefix_constraints,
                )?,
            }
        }

        let refresh_garden_signal = match garden.gate {
            RefreshGardenGate::Matte if routed_garden_active => {
                EvaluatedRefreshGardenSignalPlan::Matte {
                    valid: taps.iter().any(|tap| {
                        tap.consumer == ImageTapConsumer::RefreshGardenMatte
                            && !matches!(tap.resolved, PlannedImageSource::Transparent)
                    }),
                }
            }
            RefreshGardenGate::Motion if routed_garden_active => {
                let layer_id = match garden.motion_route {
                    RefreshGardenMotionRoute::SelectedLayer { layer_id, .. } => Some(layer_id),
                    RefreshGardenMotionRoute::None
                    | RefreshGardenMotionRoute::MissingSelectedLayer { .. } => None,
                };
                let field_slot = layer_id.and_then(|layer_id| {
                    motion.advanced().and_then(|motion| {
                        motion
                            .scope(VisualScopeId::Layer(layer_id))
                            .and_then(|scope| scope.admitted_field_slot())
                    })
                });
                EvaluatedRefreshGardenSignalPlan::Motion {
                    layer_id,
                    field_slot,
                    valid: field_slot.is_some(),
                }
            }
            _ => EvaluatedRefreshGardenSignalPlan::Inline,
        };
        let refresh_garden_resources = refresh_garden_resource_plan(refresh_garden_signal);

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
        // Re-derive the dedicated-pass table from the emitted steps rather than
        // from the authored racks, so the segmenter and the ledger cannot
        // disagree about how many dedicated passes the frame actually encodes.
        let symmetry_field_resources = symmetry_field_resource_plan(count_symmetry_field_steps(
            &layer_plans,
            &group_plans,
            &master,
        )?)?;
        let study_field_resources = study_field_resource_plan(count_study_field_steps(
            &layer_plans,
            &group_plans,
            &master,
        )?)?;
        validate_study_field_textures(study_field_resources, self.input.resource_limits)?;
        let study_identity = study_identity_hash(&layer_plans, &group_plans, &master);
        let (scan_passes, scan_vertices) =
            count_scan_processor_steps(&layer_plans, &group_plans, &master)?;
        let scan_processor_resources =
            scan_processor_resource_plan(scan_passes, scan_vertices, output)?;
        validate_scan_processor_textures(scan_processor_resources, self.input.resource_limits)?;
        let (resources, residual_resources) = resource_preflight(
            output,
            self.input,
            &layer_plans,
            &group_plans,
            &master,
            &graph,
            &taps,
            &motion,
            refresh_garden_signal,
            symmetry_field_resources,
        )?;
        let topology_signature = advanced_topology_signature(
            self.input.composition,
            self.input.master_rack,
            &self.racks,
            &layer_plans,
            &group_plans,
            &taps,
            &motion,
            refresh_garden_signal,
            symmetry_field_resources,
            study_field_resources,
            study_identity,
            scan_processor_resources,
            residual_resources,
        );

        Ok(EvaluatedCompositionPlan::Advanced(Box::new(
            AdvancedCompositionPlan {
                base: self.base.clone_without_image_routing(),
                master,
                layers: layer_plans.into_boxed_slice(),
                groups: group_plans.into_boxed_slice(),
                root: self.input.composition.root().to_vec().into_boxed_slice(),
                bus_crossfade: self.input.composition.bus_crossfade(),
                mixer: self.input.composition.mixer().sanitized(),
                image_taps: taps.into_boxed_slice(),
                diagnostics: diagnostics.into_boxed_slice(),
                below,
                graph,
                execution_order: execution_order.into_boxed_slice(),
                resources,
                residual_resources,
                motion,
                refresh_garden_signal,
                refresh_garden_resources,
                symmetry_field_resources,
                study_field_resources,
                scan_processor_resources,
                topology_signature,
            },
        )))
    }

    /// Every armed Symmetry Field motion slot in the composition, in the same
    /// layers-then-groups-then-master order tap collection uses.
    ///
    /// This is authored topology only. It never inspects a donor layer's own
    /// `MotionParams`, and it never inspects whether a field happens to exist,
    /// so the requests it produces are stable across frames.
    fn symmetry_motion_requests(&self) -> Vec<SymmetryMotionRequest> {
        let mut requests = Vec::new();
        let mut collect = |scope: VisualScopeId, rack: &RuntimeVisualRack| {
            for node in rack.iter() {
                if !node.enabled || node.wet <= 0.0 {
                    continue;
                }
                let RuntimeVisualNodeKind::Symmetry(symmetry) = node.kind else {
                    continue;
                };
                if symmetry.is_exact_bypass() {
                    continue;
                }
                for (slot, donor) in symmetry.admitted_motion_donors().into_iter().enumerate() {
                    let Some(donor) = donor else {
                        continue;
                    };
                    let Ok(slot) = u8::try_from(slot) else {
                        continue;
                    };
                    requests.push(SymmetryMotionRequest {
                        scope,
                        node_id: node.stable_id,
                        slot,
                        donor,
                    });
                }
            }
        };
        for (id, rack) in &self.racks {
            collect(VisualScopeId::Layer(*id), rack);
        }
        for group in self.input.composition.groups() {
            collect(VisualScopeId::Group(group.id), &group.rack);
        }
        collect(VisualScopeId::Master, self.input.master_rack);
        requests
    }

    fn evaluate_motion(
        &self,
        flat_ids: &[StableLayerId],
    ) -> Result<EvaluatedMotionPlan, CompositionPlanError> {
        let temporal_marker_present = self
            .input
            .master_rack
            .iter()
            .any(|node| matches!(node.kind, RuntimeVisualNodeKind::LegacyTemporal));
        let garden = self.base.temporal().originals.garden;
        let garden_motion_active = temporal_marker_present
            && garden.amount.is_finite()
            && garden.amount > 0.0
            && garden.gate == RefreshGardenGate::Motion;
        let symmetry_motion = self.symmetry_motion_requests();
        let Some(input) = self.input.motion else {
            let mut diagnostics = Vec::new();
            if garden_motion_active {
                diagnostics.push(match garden.motion_route {
                    RefreshGardenMotionRoute::None => {
                        MotionPlanDiagnostic::RefreshGardenMotionNotSelected
                    }
                    RefreshGardenMotionRoute::SelectedLayer { .. } => {
                        MotionPlanDiagnostic::RefreshGardenMotionUnavailable
                    }
                    RefreshGardenMotionRoute::MissingSelectedLayer { saved_position } => {
                        MotionPlanDiagnostic::MissingRefreshGardenMotion { saved_position }
                    }
                });
            }
            // A frame with no motion input at all can serve no primitive
            // vector/gate pair, so every armed Symmetry slot is an incomplete
            // pair that binds neutral views and says so by name.
            for request in &symmetry_motion {
                diagnostics.push(request.unserved_diagnostic());
            }
            if diagnostics.is_empty() {
                return Ok(EvaluatedMotionPlan::LegacyExact);
            }
            return Ok(EvaluatedMotionPlan::Advanced(Box::new(
                AdvancedMotionPlan {
                    scopes: Box::new([]),
                    fields: Box::new([]),
                    resources: MotionResourcePlan::default(),
                    collider: None,
                    collider_resources: FieldColliderResourcePlan::default(),
                    diagnostics: diagnostics.into_boxed_slice(),
                    budget: MotionFrameBudget::default(),
                    topology_signature: motion_topology_signature(
                        &[],
                        &[],
                        MotionResourcePlan::default(),
                        None,
                    ),
                },
            )));
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
            required_as_garden_signal: false,
            donor_scope: None,
            donor_field_slot: None,
            transplant_admitted: false,
            collider_admitted: false,
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
                required_as_garden_signal: false,
                donor_scope: None,
                donor_field_slot: None,
                transplant_admitted: false,
                collider_admitted: false,
                codec: authored.codec,
            });
        }
        let mut diagnostics = Vec::new();
        let mut garden_recipient_index = None;
        if garden_motion_active {
            match garden.motion_route {
                RefreshGardenMotionRoute::None => {
                    diagnostics.push(MotionPlanDiagnostic::RefreshGardenMotionNotSelected);
                }
                RefreshGardenMotionRoute::MissingSelectedLayer { saved_position } => {
                    diagnostics
                        .push(MotionPlanDiagnostic::MissingRefreshGardenMotion { saved_position });
                }
                RefreshGardenMotionRoute::SelectedLayer {
                    layer_id,
                    saved_position,
                } => {
                    if let Some(index) = scopes
                        .iter()
                        .position(|scope| scope.scope == VisualScopeId::Layer(layer_id))
                    {
                        garden_recipient_index = Some(index);
                    } else {
                        diagnostics.push(MotionPlanDiagnostic::MissingRefreshGardenMotion {
                            saved_position,
                        });
                    }
                }
            }
        }
        if master_params.transplant.amount > 0.0 {
            diagnostics.push(MotionPlanDiagnostic::MasterTransplantRejected);
        }
        let mut admitted_recipient = None;
        let mut admitted_collider: Option<(usize, usize, usize)> = None;
        for id in flat_ids {
            let recipient_index = scopes
                .iter()
                .position(|scope| scope.scope == VisualScopeId::Layer(*id))
                .ok_or(CompositionPlanError::Internal(
                    "motion recipient disappeared from compact scope plan",
                ))?;
            if scopes[recipient_index].params.transplant.amount <= 0.0 {
                // `collider_admission` answers this identically, so a collider
                // authored on a zero-amount scope is reported there rather than
                // silently skipped here.
                if scopes[recipient_index].params.collider.enabled {
                    diagnostics.push(MotionPlanDiagnostic::FieldCollider {
                        recipient: *id,
                        diagnostic: FieldColliderDiagnostic::NoActiveTransplant,
                    });
                }
                continue;
            }
            // The Field Collider is resolved BEFORE the single-donor recipe,
            // because an admitted collider parks that recipe rather than
            // ambiguously running both. `collider_admission` is the one shared
            // predicate — the same call the executor and the dependency walk
            // make — and everything below it is ordinary environmental
            // resolution against the live stack, exactly as the transplant
            // donor's own `base_index` check is.
            match scopes[recipient_index].params.collider_admission(false) {
                FieldColliderAdmission::Admitted { input_a, input_b } => {
                    let mut resolved = [None, None];
                    let mut refused = false;
                    for (slot, candidate) in [
                        (FieldColliderInput::A, input_a),
                        (FieldColliderInput::B, input_b),
                    ] {
                        // A donor that has left the stack is a tombstone, not a
                        // rebinding opportunity.
                        let index = self
                            .base_index
                            .contains_key(&candidate)
                            .then(|| {
                                scopes.iter().position(|scope| {
                                    scope.scope == VisualScopeId::Layer(candidate)
                                })
                            })
                            .flatten();
                        match index {
                            Some(index) => resolved[usize::from(slot.index())] = Some(index),
                            None => {
                                diagnostics.push(MotionPlanDiagnostic::FieldCollider {
                                    recipient: *id,
                                    diagnostic: FieldColliderDiagnostic::InputMissing {
                                        input: slot,
                                    },
                                });
                                refused = true;
                            }
                        }
                    }
                    if let ([Some(a_index), Some(b_index)], false) = (resolved, refused) {
                        if admitted_recipient.is_some() {
                            diagnostics.push(MotionPlanDiagnostic::ExcessTransplantRejected {
                                recipient: *id,
                                admitted_recipient: admitted_recipient.unwrap_or(*id),
                            });
                            continue;
                        }
                        admitted_recipient = Some(*id);
                        scopes[recipient_index].transplant_admitted = true;
                        scopes[recipient_index].collider_admitted = true;
                        // Both inputs demand an honest primitive field even
                        // when their own Motion is exactly zero. This is the
                        // established `required_as_donor` flag and the only
                        // path: `field_required` below reads it before the
                        // shutter, so a stationary donor still yields a field.
                        scopes[a_index].required_as_donor = true;
                        scopes[b_index].required_as_donor = true;
                        admitted_collider = Some((recipient_index, a_index, b_index));
                        continue;
                    }
                    // A refused collider delegates to exact M4 and falls
                    // through to the single-donor recipe it parked.
                }
                FieldColliderAdmission::Delegated { diagnostic } => {
                    if diagnostic.is_fault() {
                        diagnostics.push(MotionPlanDiagnostic::FieldCollider {
                            recipient: *id,
                            diagnostic,
                        });
                    }
                }
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

        // A Symmetry Field motion slot demands its donor's primitive
        // vector/gate field through the established flag, exactly as an
        // admitted Faraday transplant does. The donor's own Motion parameters
        // are never consulted: a donor whose visible Motion is exactly zero
        // still yields a field once `required_as_donor` is set, because
        // `field_required` below reads the flag before the shutter.
        for request in &symmetry_motion {
            let donor_id = match request.donor {
                MotionDonor::None => {
                    diagnostics.push(request.unselected_diagnostic());
                    continue;
                }
                MotionDonor::Missing { saved_position } => {
                    diagnostics.push(request.missing_diagnostic(saved_position));
                    continue;
                }
                MotionDonor::Selected {
                    layer_id,
                    saved_position,
                } => {
                    if !self.base_index.contains_key(&layer_id) {
                        // A donor that has left the stack is a tombstone, not a
                        // rebinding opportunity.
                        diagnostics.push(request.missing_diagnostic(saved_position));
                        continue;
                    }
                    layer_id
                }
            };
            let Some(donor_index) = scopes
                .iter()
                .position(|scope| scope.scope == VisualScopeId::Layer(donor_id))
            else {
                diagnostics.push(request.unserved_diagnostic());
                continue;
            };
            scopes[donor_index].required_as_donor = true;
        }

        if let Some(recipient_index) = garden_recipient_index {
            let signal_scope = if scopes[recipient_index].transplant_admitted
                && !scopes[recipient_index].collider_admitted
            {
                scopes[recipient_index]
                    .donor_scope
                    .and_then(|donor| scopes.iter().position(|scope| scope.scope == donor))
                    .ok_or(CompositionPlanError::Internal(
                        "admitted Garden motion recipient lost its donor scope",
                    ))?
            } else {
                // A collider recipient substitutes no single donor, so the
                // routed Garden signal honestly observes the recipient's own
                // primitive motion rather than one of the two collided inputs.
                recipient_index
            };
            scopes[signal_scope].required_as_garden_signal = true;
        }

        if !garden_motion_active
            && symmetry_motion.is_empty()
            && scopes.iter().all(|scope| {
                scope.params.is_exact_zero()
                    && !scope.required_as_donor
                    && !scope.required_as_garden_signal
                    // A disabled collider is exact M4 and must not hold the
                    // plan out of LegacyExact. An enabled one always produces
                    // at least a diagnostic, so it always has something to say.
                    && scope.params.collider.is_exact_m4()
            })
        {
            return Ok(EvaluatedMotionPlan::LegacyExact);
        }

        let mut fields = Vec::new();
        for scope in &mut scopes {
            let field_required = scope.required_as_donor
                || scope.required_as_garden_signal
                || !scope.params.shutter.is_exact_zero();
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
                required_as_garden_signal: scope.required_as_garden_signal,
            });
        }
        for scope in &mut scopes {
            scope.donor_field_slot = scope
                .donor_scope
                .and_then(|donor| fields.iter().find(|field| field.scope == donor))
                .map(|field| field.slot);
        }

        // Derived slots append AFTER every primitive field, own no decoder
        // attachment and no luma source, and are never confused with a
        // codec/lattice acquisition slot.
        let collider = match admitted_collider {
            Some((recipient_index, a_index, b_index)) => {
                let output_grid = MotionGrid::for_source(
                    scopes[recipient_index].source_dimensions,
                    scopes[recipient_index].params.lattice_quality,
                )
                .map_err(CompositionPlanError::Motion)?;
                let slot_for = |index: usize| {
                    scopes[index]
                        .field_slot
                        .ok_or(CompositionPlanError::Internal(
                            "an admitted Field Collider input was denied its primitive field",
                        ))
                };
                let output_slot = u8::try_from(fields.len()).map_err(|_| {
                    CompositionPlanError::Internal("derived motion field slot exceeds u8")
                })?;
                Some(EvaluatedFieldColliderPlan {
                    output_slot,
                    recipient_scope: scopes[recipient_index].scope,
                    input_a_scope: scopes[a_index].scope,
                    input_a_slot: slot_for(a_index)?,
                    input_b_scope: scopes[b_index].scope,
                    input_b_slot: slot_for(b_index)?,
                    output_grid,
                    algorithm_version: scopes[recipient_index].params.collider.algorithm_version,
                    mode: scopes[recipient_index].params.collider.mode,
                    boundary: scopes[recipient_index].params.collider.boundary,
                })
            }
            None => None,
        };
        let collider_resources = FieldColliderResourcePlan::preflight(
            collider
                .map(|plan| [plan.output_grid])
                .as_ref()
                .map_or(&[][..], |grids| &grids[..]),
            input.limits,
        )
        .map_err(CompositionPlanError::Motion)?;

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
                    required_as_garden_signal: scope.required_as_garden_signal,
                }
            })
            .collect::<Vec<_>>();
        let resources = MotionResourcePlan::preflight(&requests, input.limits)
            .map_err(CompositionPlanError::Motion)?;
        let budget = motion_frame_budget(&scopes)?;
        let topology_signature = motion_topology_signature(&scopes, &fields, resources, collider);
        Ok(EvaluatedMotionPlan::Advanced(Box::new(
            AdvancedMotionPlan {
                scopes: scopes.into_boxed_slice(),
                fields: fields.into_boxed_slice(),
                resources,
                collider,
                collider_resources,
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
    motion: &EvaluatedMotionPlan,
    studies: Option<&crate::study_eval::StudyProgramLibrary>,
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
                flush_segment(
                    &mut pending,
                    &mut steps,
                    &mut segment_index,
                    scope,
                    output,
                    motion,
                    studies,
                )?;
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
                flush_segment(
                    &mut pending,
                    &mut steps,
                    &mut segment_index,
                    scope,
                    output,
                    motion,
                    studies,
                )?;
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
    flush_segment(
        &mut pending,
        &mut steps,
        &mut segment_index,
        scope,
        output,
        motion,
        studies,
    )?;
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
    motion: &EvaluatedMotionPlan,
    studies: Option<&crate::study_eval::StudyProgramLibrary>,
) -> Result<(), CompositionPlanError> {
    if pending.is_empty() {
        return Ok(());
    }
    let mut ordinary = Vec::new();
    for node in std::mem::take(pending) {
        // Kind-only, deliberately not value-gated: segment indices and the
        // executor's uniform-slot reservation must not move when a gain
        // crosses zero. The value-gated predicate lives in `collect_rack_taps`
        // and its saved-patch twin, never here.
        if let RuntimeVisualNodeKind::Study(params) = node.kind {
            // Same dedicated lift as the Symmetry arm below, same kind-only
            // reasoning: segment indices and uniform-slot reservations must
            // not move when a value crosses zero.
            compile_segment_nodes(
                std::mem::take(&mut ordinary),
                steps,
                segment_index,
                scope,
                output,
            )?;
            let resolved = params
                .document_digest
                .and_then(|digest| studies.and_then(|library| library.get(&digest)));
            if let Some(entry) = resolved {
                let loads = 1 + entry.compiled.history_load_count();
                let limit = u32::from(
                    crate::visual_rack::node_kind_descriptor(NodeKindTag::Study)
                        .budget
                        .logical_texture_lookups_per_pixel,
                );
                if loads > limit {
                    return Err(CompositionPlanError::StudyLoadBudget {
                        node: node.stable_id,
                        loads,
                        limit,
                    });
                }
            }
            steps.push(EvaluatedScopeStep::StudyField {
                plan: EvaluatedStudyPlan {
                    node_id: node.stable_id,
                    enabled: node.enabled,
                    wet: node.wet,
                    blend: node.blend,
                    params: params.sanitized(),
                    program: resolved.map(|entry| entry.program.clone()),
                    instruction_count: resolved
                        .map_or(0, |entry| entry.compiled.instruction_count()),
                    resources: study_field_resource_plan(1)?,
                },
            });
            continue;
        }
        if let RuntimeVisualNodeKind::Symmetry(params) = node.kind {
            // A dedicated eight-texture pass owns its own bind layout, so it is
            // lifted out of segmentation entirely rather than isolated into a
            // one-node rack segment: flush the ordinary work that precedes it,
            // emit exactly one step at the authored position, and let the loop
            // resume accumulating behind it.
            compile_segment_nodes(
                std::mem::take(&mut ordinary),
                steps,
                segment_index,
                scope,
                output,
            )?;
            steps.push(EvaluatedScopeStep::SymmetryField {
                plan: symmetry_field_plan(scope, node, params, motion)?,
            });
            continue;
        }
        if let RuntimeVisualNodeKind::ScanProcessor(params) = node.kind {
            // The same dedicated lift, same kind-only reasoning — and the one
            // place the named vertex budget is enforced. The sanitized ranges
            // cannot exceed the cap today, so this refusal is the Residual
            // grid-edge law: defense in depth with a typed error, never a
            // silent clamp.
            compile_segment_nodes(
                std::mem::take(&mut ordinary),
                steps,
                segment_index,
                scope,
                output,
            )?;
            let params = params.sanitized();
            let vertices = params.vertex_count();
            if vertices > crate::scan_processor::MAX_SCAN_PROCESSOR_VERTICES {
                return Err(CompositionPlanError::ScanProcessorVertexBudget {
                    node: node.stable_id,
                    vertices,
                    limit: crate::scan_processor::MAX_SCAN_PROCESSOR_VERTICES,
                });
            }
            steps.push(EvaluatedScopeStep::ScanProcessorField {
                plan: EvaluatedScanProcessorPlan {
                    node_id: node.stable_id,
                    enabled: node.enabled,
                    wet: node.wet,
                    blend: node.blend,
                    params,
                    resources: scan_processor_resource_plan(1, vertices, output)?,
                },
            });
            continue;
        }
        // Deliberately kind-only, unlike the value-gated admission predicate in
        // `collect_rack_taps`: segment indices must never depend on frame-local
        // gains, or the uniform-slot reservation renumbers whenever one crosses
        // zero.
        let is_image_consumer = matches!(
            node.kind,
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(_))
                | RuntimeVisualNodeKind::Displace(_)
                | RuntimeVisualNodeKind::Residual(_)
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

/// Build one dedicated step's evaluated payload.
///
/// Motion slots are resolved through `admitted_field_slot`, never through
/// `field_slot`: an admitted Faraday transplant replaces a scope's own field
/// with its donor's, and a routed consumer that read the raw slot would sample
/// a different field than motion rendering wrote.
fn symmetry_field_plan(
    scope: VisualScopeId,
    node: RuntimeVisualNode,
    params: RuntimeSymmetryParams,
    motion: &EvaluatedMotionPlan,
) -> Result<EvaluatedSymmetryFieldPlan, CompositionPlanError> {
    let params = params.sanitized();
    let admitted = params.admitted_motion_donors();
    // The same value-gated predicate `symmetry_motion_requests` used to ask for
    // these fields. A dormant node asked for nothing, so it must observe
    // nothing rather than inheriting a field some other consumer admitted.
    let admitted_node = node.enabled && node.wet > 0.0 && !params.is_exact_bypass();
    let motion_field_slots = std::array::from_fn(|slot| {
        if !admitted_node {
            return None;
        }
        let MotionDonor::Selected { layer_id, .. } = admitted[slot]? else {
            return None;
        };
        motion
            .advanced()?
            .scope(VisualScopeId::Layer(layer_id))
            .and_then(|scope| scope.admitted_field_slot())
    });
    Ok(EvaluatedSymmetryFieldPlan {
        node_id: node.stable_id,
        domain: SymmetryNodeDomain::for_scope(scope, node.stable_id.get()),
        enabled: node.enabled,
        wet: node.wet,
        blend: node.blend,
        params,
        motion_field_slots,
        resources: symmetry_field_resource_plan(1)?,
    })
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
    gesture_canvas_admitted: bool,
    program_tap_admitted: bool,
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
        // Routes are collected by slot, so a node carrying several fixed
        // routes yields one tap per admitted slot and slot index stays route
        // identity all the way into the consumer key. A single-route kind fills
        // slot 0 only; Residual names both of its slots and Symmetry both of
        // its image slots, so those donors are separate consumer identities
        // that can never alias.
        let mut routes: [Option<ResolvedImageTap>; RESIDUAL_ROUTE_SLOTS] =
            [None; RESIDUAL_ROUTE_SLOTS];
        match node.kind {
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) if matte.amount > 0.0 => {
                routes[usize::from(RACK_PRIMARY_ROUTE_SLOT)] = Some(matte.tap);
            }
            // A Displace whose two amounts are both zero is an exact bypass:
            // it encodes no pass, so it must not claim a cross-scope donor,
            // a dependency edge, or a binding slot either.
            RuntimeVisualNodeKind::Displace(displace) if !displace.is_exact_bypass() => {
                routes[usize::from(RACK_PRIMARY_ROUTE_SLOT)] = Some(displace.tap);
            }
            // A Residual at zero mix is the same real delegation, and it
            // delegates both slots at once rather than half a decomposition.
            RuntimeVisualNodeKind::Residual(residual) if !residual.is_exact_bypass() => {
                let [structure, detail] = residual.routes();
                routes[usize::from(RESIDUAL_STRUCTURE_SLOT)] = Some(structure);
                routes[usize::from(RESIDUAL_DETAIL_SLOT)] = Some(detail);
            }
            // A Symmetry Field answers admission per image slot: a donor no
            // sector record can name claims nothing, exactly as a whole
            // bypassed node claims nothing. The destructure is the
            // compile-time proof that both frozen image slots are carried.
            RuntimeVisualNodeKind::Symmetry(symmetry) if !symmetry.is_exact_bypass() => {
                let [donor0, donor1] = symmetry.admitted_donor_taps();
                routes[0] = donor0;
                routes[1] = donor1;
            }
            _ => continue,
        }
        for (slot, tap) in routes.iter().enumerate() {
            let Some(tap) = *tap else {
                continue;
            };
            let slot = u8::try_from(slot).map_err(|_| {
                CompositionPlanError::Internal("rack route slot left its bounded domain")
            })?;
            collect_tap(
                ImageTapConsumer::RackNode {
                    scope,
                    node_id: node.stable_id,
                    slot,
                },
                PlannedImageTapOrigin::Rack(tap),
                below,
                known_scopes,
                gesture_canvas_admitted,
                program_tap_admitted,
                taps,
                diagnostics,
                dependencies,
                route_edges,
                prefix_constraints,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_tap(
    consumer: ImageTapConsumer,
    origin: PlannedImageTapOrigin,
    below: &BelowTopology,
    known_scopes: &BTreeSet<VisualScopeId>,
    gesture_canvas_admitted: bool,
    program_tap_admitted: bool,
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
        // The canvas is etched outside the composition graph, so it is a
        // producer with no scope: no `dependency_producer`, no `current_edge`,
        // and therefore no ordering constraint and no cycle. It also has no
        // N-1 parity — a frame-committed singleton has one image — so both
        // timings read the same committed field rather than quietly claiming a
        // history pair nothing writes.
        TapRouteSource::GestureCanvas if gesture_canvas_admitted => {
            PlannedImageSource::GestureCanvas
        }
        TapRouteSource::GestureCanvas => {
            diagnostics.push(CompositionPlanDiagnostic::GestureCanvasUnavailable { consumer });
            PlannedImageSource::Transparent
        }
        // The programme tap is published outside the composition graph — the
        // copy runs only after the frame encoder is accepted — so it is
        // likewise a producer with no scope: no `dependency_producer`, no
        // `current_edge`, no ordering constraint, and no cycle, even from the
        // master scope whose resolve it copies. The tap *is* the N-1 image,
        // so both timings read the same committed copy rather than claiming a
        // history pair nothing writes.
        TapRouteSource::ProgramTap if program_tap_admitted => PlannedImageSource::ProgramTap,
        TapRouteSource::ProgramTap => {
            diagnostics.push(CompositionPlanDiagnostic::ProgramTapUnavailable { consumer });
            PlannedImageSource::Transparent
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
    GestureCanvas,
    ProgramTap,
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
            ResolvedImageSource::GestureCanvas => Self::GestureCanvas,
            ResolvedImageSource::ProgramTap => Self::ProgramTap,
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
    // Among scopes whose dependencies are all satisfied, execute in the
    // composition's back-to-front composite order rather than in whatever order
    // `VisualScopeId` happens to sort. Both are valid topological orders — the
    // tie is between genuinely independent scopes — but only the composite one
    // agrees with the order `build_block_schedules` drains, and a scope drained
    // out of turn needs a retained tap it may not own. A tapless stack has no
    // edges at all between siblings, so before this the fallback decided the
    // whole order, and a stack whose ids ascend front-to-back could not be
    // scheduled. The scope id remains the final tiebreak, so the sort stays
    // total and deterministic.
    let rank = |scope: VisualScopeId| (below.composite_rank(scope), scope);
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .filter_map(|(scope, degree)| (*degree == 0).then_some(rank(*scope)))
        .collect();
    let mut order = Vec::with_capacity(known_scopes.len());
    while let Some((_, scope)) = ready.pop_first() {
        order.push(scope);
        if let Some(consumers) = adjacency.get(&scope) {
            for consumer in consumers {
                if prefix_constraints
                    .get(consumer)
                    .is_some_and(|prefix| below.prefix_contains(*prefix, scope))
                {
                    continue;
                }
                decrement_indegree(*consumer, &mut indegree, &mut ready, rank);
            }
        }
        for (consumer, prefix) in prefix_constraints {
            if below.prefix_contains(*prefix, scope) {
                decrement_indegree(*consumer, &mut indegree, &mut ready, rank);
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
    mixer: crate::mixing_boundary::BusMixerState,
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
    // An armed dirty mixer can throw the crossbar to either input or drop
    // line bands through to it mid-firing, so neither lane can be culled on
    // the fader position alone while dirt is authored.
    if !mixer.dirt.is_exact_off() && layer.bus != BusAssignment::Program {
        return true;
    }
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
    // This is the sort whose output becomes the plan's execution order, so it
    // carries the same composite-rank tie-break as the scope sort above. A
    // collapsed group is ranked too: `below_topology` records the group itself
    // at its root index, so a group task sorts into the same root slot its
    // members occupy and the two sorts cannot disagree.
    let rank = |task: VisualScopeId| (below.composite_rank(task), task);
    let mut ready: BTreeSet<_> = indegree
        .iter()
        .filter_map(|(task, degree)| (*degree == 0).then_some(rank(*task)))
        .collect();
    let mut task_order = Vec::with_capacity(tasks.len());
    while let Some((_, task)) = ready.pop_first() {
        task_order.push(task);
        if let Some(consumers) = adjacency.get(&task) {
            for consumer in consumers {
                decrement_indegree(*consumer, &mut indegree, &mut ready, rank);
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

/// Retire one edge into `consumer`, admitting it to `ready` when it has none
/// left.
///
/// The ready set is keyed by whatever order its caller wants to drain in:
/// `execution_order` keys by composite rank so independent scopes execute in
/// the order the renderer's schedule drains, while the atomic-group sort keys
/// by scope identity, where no composite order exists to prefer.
fn decrement_indegree<K: Ord>(
    consumer: VisualScopeId,
    indegree: &mut BTreeMap<VisualScopeId, usize>,
    ready: &mut BTreeSet<K>,
    key: impl Fn(VisualScopeId) -> K,
) {
    let degree = indegree
        .get_mut(&consumer)
        .expect("known scope has indegree");
    debug_assert!(*degree > 0);
    *degree = degree.saturating_sub(1);
    if *degree == 0 {
        ready.insert(key(consumer));
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
    refresh_garden_signal: EvaluatedRefreshGardenSignalPlan,
    symmetry_field: SymmetryFieldResourcePlan,
) -> Result<(CreativeResourcePlan, ResidualResourcePlan), CompositionPlanError> {
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
                PlannedImageSource::ProgramHistory
                    // The gesture canvas is a frame-committed singleton with
                    // one image and no N-1 parity, so an N-1 route to it stages
                    // nothing. The executor allocates nothing for it either;
                    // charging a parity pair here would put the declared and
                    // actual ledgers permanently one apart.
                    | PlannedImageSource::GestureCanvas
                    // The programme tap is likewise a frame-committed
                    // singleton with one image and no N-1 parity — the tap
                    // *is* the N-1 image — so an N-1 route to it stages
                    // nothing and the executor allocates nothing for it.
                    | PlannedImageSource::ProgramTap
                    | PlannedImageSource::Transparent
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
    // The dedicated pass's per-pixel terms are deliberately absent from the
    // combined arithmetic below: the authored node is already charged exactly
    // once through `capture_budget_rack` above, and charging it again here
    // would double count the same pass.
    validate_symmetry_field_textures(symmetry_field, input.resource_limits)?;
    // Reduced block-mean surfaces are sub-full-frame and byte-exact, so they
    // are budgeted entirely outside the full-frame layer ledger. Charging them
    // as `additional_rgba16_layers` would over-count by orders of magnitude.
    let residual = ResidualResourcePlan::preflight(
        &residual_resource_requests(output, &racks),
        ResidualResourceLimits::from(input.resource_limits),
    )
    .map_err(CompositionPlanError::Residual)?;
    let garden_budget = refresh_garden_resource_plan(refresh_garden_signal);
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
            .and_then(|value| value.checked_add(garden_budget.logical_texture_lookups_per_pixel))
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
            .and_then(|value| value.checked_add(garden_budget.texture_operations_per_pixel))
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
        let maximum_textures = motion
            .budget
            .max_sampled_textures_in_pass
            .max(garden_budget.max_sampled_textures_in_pass);
        if maximum_textures > texture_limit {
            return Err(CompositionPlanError::Resource(
                ResourcePreflightError::SampledTextureLimit {
                    requested: maximum_textures,
                    limit: texture_limit,
                },
            ));
        }
    } else if garden_budget.max_sampled_textures_in_pass > 0 {
        let combined_logical_lookups = creative
            .logical_texture_lookups_per_pixel
            .checked_add(garden_budget.logical_texture_lookups_per_pixel)
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
            .checked_add(garden_budget.texture_operations_per_pixel)
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
        if garden_budget.max_sampled_textures_in_pass > texture_limit {
            return Err(CompositionPlanError::Resource(
                ResourcePreflightError::SampledTextureLimit {
                    requested: garden_budget.max_sampled_textures_in_pass,
                    limit: texture_limit,
                },
            ));
        }
    }
    // The block-mean bytes meet the creative number only here, at the shared
    // cap, exactly the way motion bytes do. A composition with no live
    // Residual node charges nothing and this check is byte-for-byte inert.
    if residual.total_bytes > 0 {
        let motion_bytes = motion
            .advanced()
            .map_or(0, |motion| motion.resources.total_bytes);
        let combined_bytes = creative
            .creative_bytes
            .checked_add(motion_bytes)
            .and_then(|value| value.checked_add(residual.total_bytes))
            .ok_or(CompositionPlanError::Resource(
                ResourcePreflightError::ArithmeticOverflow,
            ))?;
        let byte_limit = input
            .resource_limits
            .max_creative_bytes
            .min(crate::visual_rack::MAX_CREATIVE_GPU_BYTES);
        if combined_bytes > byte_limit {
            return Err(CompositionPlanError::ResidualCombinedMemoryBudget {
                bytes: combined_bytes,
                limit: byte_limit,
            });
        }
    }
    Ok((creative, residual))
}

/// Every admitted Residual node's block-mean working set, gathered under the
/// same admission predicate the live planner and the saved-patch dependency
/// walk use. A disabled, zero-wet, or exact-bypass node delegates completely
/// and therefore charges no reduced surface here either.
fn residual_resource_requests(
    output: [u32; 2],
    racks: &[VisualRack],
) -> Vec<ResidualResourceRequest> {
    let mut requests = Vec::new();
    for rack in racks {
        for node in rack.iter() {
            if !node.enabled || node.wet <= 0.0 {
                continue;
            }
            let VisualNodeKind::Residual(residual) = node.kind else {
                continue;
            };
            if residual.is_exact_bypass() {
                continue;
            }
            requests.push(ResidualResourceRequest {
                output_dimensions: output,
                block: residual.block,
            });
        }
    }
    requests
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

#[allow(
    clippy::too_many_arguments,
    reason = "signature hashes each immutable composition domain explicitly, including routed Garden"
)]
fn advanced_topology_signature(
    composition: &RuntimeComposition,
    master: &RuntimeVisualRack,
    racks: &BTreeMap<StableLayerId, &RuntimeVisualRack>,
    layers: &[EvaluatedLayerScopePlan],
    groups: &[EvaluatedGroupScopePlan],
    taps: &[PlannedImageTap],
    motion: &EvaluatedMotionPlan,
    refresh_garden_signal: EvaluatedRefreshGardenSignalPlan,
    symmetry_field: SymmetryFieldResourcePlan,
    study_field: StudyFieldResourcePlan,
    study_identity: u64,
    scan_processor: ScanProcessorResourcePlan,
    residual: ResidualResourcePlan,
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
    hash = hash_value(
        hash,
        match refresh_garden_signal {
            EvaluatedRefreshGardenSignalPlan::Inline => 0,
            EvaluatedRefreshGardenSignalPlan::Matte { valid } => 1 + u64::from(valid),
            EvaluatedRefreshGardenSignalPlan::Motion {
                layer_id,
                field_slot,
                valid,
            } => {
                let mut value = 3 + u64::from(valid);
                value = hash_value(value, layer_id.map_or(0, StableLayerId::get));
                hash_value(value, u64::from(field_slot.unwrap_or(u8::MAX)))
            }
        },
    );
    if symmetry_field.full_frame_passes > 0 {
        // Append-only domain tag, in the style of the motion tag above.
        hash = hash_value(hash, 0x5359_4d4d_4554_5259);
        hash = hash_value(hash, u64::from(symmetry_field.full_frame_passes));
        hash = hash_value(hash, u64::from(symmetry_field.max_sampled_textures_in_pass));
        hash = hash_value(hash, u64::from(symmetry_field.uniform_bytes));
    }
    if study_field.full_frame_passes > 0 {
        // Append-only domain tag: "STUDYFLD".
        hash = hash_value(hash, 0x5354_5544_5946_4c44);
        hash = hash_value(hash, u64::from(study_field.full_frame_passes));
        hash = hash_value(hash, u64::from(study_field.max_sampled_textures_in_pass));
        hash = hash_value(hash, u64::from(study_field.uniform_bytes));
        hash = hash_value(hash, study_identity);
    }
    // Deliberately passes and layout only, never the vertex total: the vertex
    // count is a draw-call argument sized by authored values, so a lines or
    // samples edit re-encodes the next frame without re-preparing pipelines,
    // arenas, or the shared accumulator.
    if scan_processor.full_frame_passes > 0 {
        // Append-only domain tag: "SCANPROC".
        hash = hash_value(hash, 0x5343_414e_5052_4f43);
        hash = hash_value(hash, u64::from(scan_processor.full_frame_passes));
        hash = hash_value(hash, u64::from(scan_processor.max_sampled_textures_in_pass));
        hash = hash_value(hash, u64::from(scan_processor.uniform_bytes));
    }
    // The reduced block-mean grid is plan-visible resource topology: its
    // dimensions come from the authored block vocabulary, which the rack
    // signature deliberately does not hash. Without this a block change would
    // reuse mean surfaces sized for the previous grid.
    if residual.active_nodes > 0 {
        hash = hash_value(hash, 0x5245_5349_4455_414c);
        hash = hash_value(hash, u64::from(residual.active_nodes));
        hash = hash_value(hash, u64::from(residual.max_grid_dimensions[0]));
        hash = hash_value(hash, u64::from(residual.max_grid_dimensions[1]));
        hash = hash_value(hash, residual.total_bytes);
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
    collider: Option<EvaluatedFieldColliderPlan>,
) -> u64 {
    let mut hash = hash_value(FNV_OFFSET, 0x4d4f_5449_4f4e_0001);
    // The derived collider plan is prepared-binding topology in its own right.
    // Swapping the two inputs leaves every scope and field identical while
    // changing what each of the eight prebuilt parity bind groups means, so the
    // slot pair must enter the signature in order.
    if let Some(plan) = collider {
        hash = hash_value(hash, u64::from(plan.output_slot));
        hash = hash_scope(hash, plan.recipient_scope);
        hash = hash_scope(hash, plan.input_a_scope);
        hash = hash_value(hash, u64::from(plan.input_a_slot));
        hash = hash_scope(hash, plan.input_b_scope);
        hash = hash_value(hash, u64::from(plan.input_b_slot));
        hash = hash_value(hash, u64::from(plan.output_grid.width));
        hash = hash_value(hash, u64::from(plan.output_grid.height));
        hash = hash_value(hash, u64::from(plan.mode.code()));
        hash = hash_value(hash, u64::from(plan.boundary.code()));
        hash = hash_value(hash, u64::from(plan.algorithm_version));
    }
    hash = hash_value(hash, u64::from(resources.active_field_slots));
    hash = hash_value(hash, u64::from(resources.persistent_carriers));
    hash = hash_value(hash, u64::from(resources.active_garden_signals));
    hash = hash_value(hash, resources.garden_signal_bytes);
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
        hash = hash_value(hash, u64::from(scope.collider_admitted));
        hash = hash_value(hash, u64::from(scope.params.collider.enabled));
        hash = hash_value(hash, u64::from(scope.params.collider.mode.code()));
        hash = hash_value(hash, u64::from(scope.params.collider.boundary.code()));
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
        hash = hash_value(hash, u64::from(field.required_as_garden_signal));
        // `signature_code` is append-only: the original four keep 0-3 and the
        // six procedural kinds occupy 4-9. Contour and Chroma bind the field
        // scope's image while the pure kinds bind nothing, so a kind change
        // must re-prepare rather than silently reuse stale bind groups.
        hash = hash_value(hash, field.source.origin.signature_code());
    }
    hash
}

fn hash_consumer(mut hash: u64, consumer: ImageTapConsumer) -> u64 {
    match consumer {
        ImageTapConsumer::RackNode {
            scope,
            node_id,
            slot,
        } => {
            hash = hash_value(hash, 1);
            hash = hash_scope(hash, scope);
            hash = hash_value(hash, node_id.get());
            // The slot is part of the identity, not decoration, and it must
            // enter the hash. Without it a change confined to a node's second
            // route leaves the topology signature unchanged and
            // `CompositionGpuExecutor::is_prepared_for` silently reuses the
            // stale bindings built for the previous route.
            hash_value(hash, u64::from(slot))
        }
        ImageTapConsumer::GroupMatte { group_id } => {
            hash = hash_value(hash, 2);
            hash_value(hash, group_id.get())
        }
        ImageTapConsumer::LayerMatte { layer_id } => {
            hash = hash_value(hash, 3);
            hash_value(hash, layer_id.get())
        }
        ImageTapConsumer::RefreshGardenMatte => hash_value(hash, 4),
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
        // Append-only, exactly like the node kind codes: 6 is SelectedLayer.
        PlannedImageSource::GestureCanvas => hash_value(hash, 7),
        PlannedImageSource::ProgramTap => hash_value(hash, 8),
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
    use crate::motion::{FaradayParams, FieldColliderParams, FIELD_COLLIDER_ALGORITHM_VERSION};
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
                    pattern: None,
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
        base_with_temporal_for(&[1], temporal)
    }

    fn base_with_temporal_for(
        ids_front_to_back: &[u64],
        temporal: &TemporalParams,
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
            FramePlanContext::new(64, 64, 1.25),
            MasterFrameInput {
                effects: &master_effects,
                transform: &master_transform,
                ntsc: &ntsc,
                temporal,
            },
            ids_front_to_back
                .iter()
                .enumerate()
                .map(|(index, id)| LayerFrameInput {
                    source: SourceTap::new(*id, index, 64, 64),
                    effects: &effects[index],
                    transform: &transforms[index],
                    opacity: 1.0,
                    speed: 1.0,
                    fps: 30.0,
                    blend_mode: BlendMode::Normal,
                    visible: true,
                    paused: false,
                    bypass_master_fx: false,
                    pattern: None,
                }),
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

    fn push_displace(
        rack: &mut RuntimeVisualRack,
        source: ResolvedImageSource,
        timing: EdgeTiming,
        amount_x: f32,
    ) -> NodeId {
        rack.push(RuntimeVisualNodeKind::Displace(
            crate::visual_rack::RuntimeDisplaceParams {
                tap: ResolvedImageTap { source, timing },
                amount_x,
                amount_y: 0.0,
                boundary: crate::visual_rack::DisplaceBoundary::Wrap,
            },
        ))
        .unwrap()
    }

    fn residual_params(
        structure: (ResolvedImageSource, EdgeTiming),
        detail: (ResolvedImageSource, EdgeTiming),
        mix: f32,
    ) -> crate::visual_rack::RuntimeResidualParams {
        crate::visual_rack::RuntimeResidualParams {
            structure: ResolvedImageTap {
                source: structure.0,
                timing: structure.1,
            },
            detail: ResolvedImageTap {
                source: detail.0,
                timing: detail.1,
            },
            mix,
            ..crate::visual_rack::RuntimeResidualParams::default()
        }
    }

    fn push_residual(
        rack: &mut RuntimeVisualRack,
        params: crate::visual_rack::RuntimeResidualParams,
    ) -> NodeId {
        rack.push(RuntimeVisualNodeKind::Residual(params)).unwrap()
    }

    /// Slot-addressed tap lookup. A node-only lookup would silently return the
    /// first collected slot for a two-route node.
    fn residual_tap_for(
        plan: &AdvancedCompositionPlan,
        node: NodeId,
        slot: u8,
    ) -> Option<&PlannedImageTap> {
        plan.image_taps().iter().find(|tap| {
            matches!(
                tap.consumer,
                ImageTapConsumer::RackNode { node_id, slot: owner, .. }
                    if node_id == node && owner == slot
            )
        })
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

    fn collider_recipient(
        mode: FieldColliderMode,
        boundary: MotionBoundaryMode,
        a: u64,
        b: u64,
    ) -> MotionParams {
        MotionParams {
            transplant: FaradayParams {
                amount: 0.75,
                ..Default::default()
            },
            collider: FieldColliderParams {
                enabled: true,
                mode,
                boundary,
                input_a: MotionDonor::Selected {
                    layer_id: layer_id(a),
                    saved_position: saved_position(1),
                },
                input_b: MotionDonor::Selected {
                    layer_id: layer_id(b),
                    saved_position: saved_position(2),
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Plan a three-layer stack where layer 10 collides layers 20 and 30.
    fn plan_collider(recipient: MotionParams) -> AdvancedCompositionPlan {
        let base = base(&[10, 20, 30], &[]);
        let composition = legacy_composition(&[10, 20, 30]);
        let racks = legacy_racks(&[10, 20, 30]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = [
            LayerMotionPlanInput {
                stable_id: layer_id(10),
                params: recipient,
                codec: MotionCodecFrameFacts::default(),
            },
            // Both donors are authored EXACTLY zero: their fields must still be
            // admitted, which is the whole point of `required_as_donor`.
            LayerMotionPlanInput {
                stable_id: layer_id(20),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: layer_id(30),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        advanced(
            plan_with_motion(
                &base,
                &composition,
                &master,
                &racks,
                MotionParams::default(),
                &layers,
            )
            .unwrap(),
        )
    }

    #[test]
    fn a_collider_admits_both_inputs_as_honest_primitive_fields_at_exactly_zero_motion() {
        let advanced = plan_collider(collider_recipient(
            FieldColliderMode::Curl,
            MotionBoundaryMode::Wrap,
            20,
            30,
        ));
        let motion = advanced.motion().advanced().expect("collider motion plan");

        // Two primitive fields, one per input, both admitted purely because a
        // collider named them — their own Motion is exactly zero.
        assert_eq!(motion.fields().len(), 2);
        for field in motion.fields() {
            assert!(field.required_as_donor, "{:?}", field.scope);
            assert!(!field.required_as_garden_signal);
        }
        let scopes: Vec<_> = motion.fields().iter().map(|field| field.scope).collect();
        assert!(scopes.contains(&VisualScopeId::Layer(layer_id(20))));
        assert!(scopes.contains(&VisualScopeId::Layer(layer_id(30))));
        assert!(
            !scopes.contains(&VisualScopeId::Layer(layer_id(10))),
            "a collider recipient needs no primitive field of its own"
        );

        let plan = motion.collider().expect("an admitted collider plan");
        assert_eq!(plan.recipient_scope, VisualScopeId::Layer(layer_id(10)));
        assert_eq!(plan.input_a_scope, VisualScopeId::Layer(layer_id(20)));
        assert_eq!(plan.input_b_scope, VisualScopeId::Layer(layer_id(30)));
        assert_ne!(plan.input_a_slot, plan.input_b_slot);
        assert_eq!(plan.mode, FieldColliderMode::Curl);
        assert_eq!(plan.boundary, MotionBoundaryMode::Wrap);
        assert_eq!(plan.algorithm_version, FIELD_COLLIDER_ALGORITHM_VERSION);
        // Each input's slot is resolved through `admitted_field_slot`, so the
        // collider observes exactly the field motion rendering publishes.
        for (scope, slot) in [
            (plan.input_a_scope, plan.input_a_slot),
            (plan.input_b_scope, plan.input_b_slot),
        ] {
            assert_eq!(
                motion.scope(scope).unwrap().admitted_field_slot(),
                Some(slot)
            );
        }
        // Derived slots append AFTER every primitive field slot.
        assert_eq!(plan.output_slot, 2);
        assert!(plan.output_slot as usize >= motion.fields().len());
        assert!(!motion
            .diagnostics()
            .iter()
            .any(|entry| matches!(entry, MotionPlanDiagnostic::FieldCollider { .. })));

        // The recipient's transplant is admitted and owned by the collider.
        let recipient = motion.scope(plan.recipient_scope).unwrap();
        assert!(recipient.transplant_admitted);
        assert!(recipient.collider_admitted);
        assert_eq!(recipient.donor_scope, None);
        assert_eq!(recipient.donor_field_slot, None);

        // The byte-exact ledger: 20 bytes per output-grid cell, two low
        // resolution passes, three sampled textures.
        let resources = motion.collider_resources();
        let cells = plan.output_grid.vector_count;
        assert_eq!(resources.active_colliders, 1);
        assert_eq!(resources.total_bytes, cells * 20);
        assert_eq!(resources.bytes_per_cell(), 20);
        assert_eq!(resources.low_resolution_passes, 2);
        assert_eq!(resources.nearest_lookups, 5);
        assert_eq!(resources.max_sampled_textures_in_pass, 3);
    }

    #[test]
    fn a_collider_recombines_a_procedural_field_against_a_codec_field() {
        use crate::motion::{MotionFieldSource, ProceduralFieldKind};

        let base = base(&[10, 20, 30], &[]);
        let composition = legacy_composition(&[10, 20, 30]);
        let racks = legacy_racks(&[10, 20, 30]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = [
            LayerMotionPlanInput {
                stable_id: layer_id(10),
                params: collider_recipient(
                    FieldColliderMode::Sum,
                    MotionBoundaryMode::Wrap,
                    20,
                    30,
                ),
                codec: MotionCodecFrameFacts::default(),
            },
            // Input A synthesizes; its authored Motion is otherwise zero, so
            // only `required_as_donor` pulls the procedural field in.
            LayerMotionPlanInput {
                stable_id: layer_id(20),
                params: MotionParams {
                    field_source: MotionFieldSource::Procedural(ProceduralFieldKind::Curl),
                    ..MotionParams::default()
                },
                codec: MotionCodecFrameFacts::default(),
            },
            // Input B observes real codec truth.
            LayerMotionPlanInput {
                stable_id: layer_id(30),
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
        let motion = advanced.motion().advanced().expect("collider motion plan");
        let plan = motion.collider().expect("an admitted collider plan");
        let origin = |scope| {
            let slot = motion.scope(scope).unwrap().admitted_field_slot().unwrap();
            motion.fields()[usize::from(slot)].source.origin
        };
        assert_eq!(
            origin(plan.input_a_scope),
            MotionFieldOrigin::Procedural(ProceduralFieldKind::Curl)
        );
        assert_eq!(origin(plan.input_b_scope), MotionFieldOrigin::CodecVectors);
        // A procedural input needs no luma observation plane, so the two
        // fields differ by exactly the codec/lattice machinery and the mixed
        // pair still satisfies the ordinary collider ledger.
        assert_eq!(motion.collider_resources().active_colliders, 1);
    }

    #[test]
    fn a_disabled_collider_is_exact_m4_and_costs_nothing() {
        let mut recipient = collider_recipient(
            FieldColliderMode::Sum,
            MotionBoundaryMode::Transparent,
            20,
            30,
        );
        recipient.collider.enabled = false;
        // With the collider off, the parked single-donor recipe runs. Its
        // authored inputs are retained verbatim and simply do nothing.
        recipient.transplant.donor = MotionDonor::Selected {
            layer_id: layer_id(20),
            saved_position: saved_position(1),
        };
        let advanced = plan_collider(recipient);
        let motion = advanced.motion().advanced().unwrap();

        assert_eq!(motion.collider(), None);
        assert_eq!(
            motion.collider_resources(),
            FieldColliderResourcePlan::default()
        );
        assert_eq!(motion.collider_resources().total_bytes, 0);
        assert_eq!(motion.collider_resources().low_resolution_passes, 0);
        // Exactly the M4 shape: ONE donor field, the recipient substituting it.
        assert_eq!(motion.fields().len(), 1);
        assert_eq!(motion.fields()[0].scope, VisualScopeId::Layer(layer_id(20)));
        let recipient_scope = motion.scope(VisualScopeId::Layer(layer_id(10))).unwrap();
        assert!(recipient_scope.transplant_admitted);
        assert!(!recipient_scope.collider_admitted);
        assert_eq!(
            recipient_scope.donor_scope,
            Some(VisualScopeId::Layer(layer_id(20)))
        );
        assert_eq!(recipient_scope.admitted_field_slot(), Some(0));
        // A disabled block is not a fault and publishes no diagnostic.
        assert!(!motion
            .diagnostics()
            .iter()
            .any(|entry| matches!(entry, MotionPlanDiagnostic::FieldCollider { .. })));

        // And the authored collider inputs survived, unerased, ready to resume.
        assert!(matches!(
            recipient_scope.params.collider.input_a,
            MotionDonor::Selected { layer_id: id, .. } if id == layer_id(20)
        ));
        assert!(matches!(
            recipient_scope.params.collider.input_b,
            MotionDonor::Selected { layer_id: id, .. } if id == layer_id(30)
        ));
    }

    #[test]
    fn an_aliased_missing_or_removed_collider_input_delegates_with_a_typed_diagnostic() {
        let diagnostic_of = |recipient: MotionParams| {
            let advanced = plan_collider(recipient);
            let motion = advanced.motion().advanced().unwrap();
            let found = motion.diagnostics().iter().find_map(|entry| match entry {
                MotionPlanDiagnostic::FieldCollider { diagnostic, .. } => Some(*diagnostic),
                _ => None,
            });
            (motion.collider().is_some(), found)
        };

        let base = collider_recipient(
            FieldColliderMode::Sum,
            MotionBoundaryMode::Transparent,
            20,
            30,
        );

        // A and B may never alias each other.
        let mut aliased = base;
        aliased.collider.input_b = aliased.collider.input_a;
        assert_eq!(
            diagnostic_of(aliased),
            (false, Some(FieldColliderDiagnostic::AliasedInputs))
        );

        // A tombstone refuses by name and by slot, and never rebinds.
        let mut tombstoned = base;
        tombstoned.collider.input_b = MotionDonor::Missing {
            saved_position: saved_position(2),
        };
        assert_eq!(
            diagnostic_of(tombstoned),
            (
                false,
                Some(FieldColliderDiagnostic::InputMissing {
                    input: FieldColliderInput::B
                })
            )
        );

        // An unselected slot.
        let mut unselected = base;
        unselected.collider.input_a = MotionDonor::None;
        assert_eq!(
            diagnostic_of(unselected),
            (
                false,
                Some(FieldColliderDiagnostic::InputUnselected {
                    input: FieldColliderInput::A
                })
            )
        );

        // A donor that has left the stack is a tombstone, not a rebinding
        // opportunity: it resolves to no scope and refuses by slot.
        let mut removed = base;
        removed.collider.input_b = MotionDonor::Selected {
            layer_id: layer_id(9_999),
            saved_position: saved_position(2),
        };
        assert_eq!(
            diagnostic_of(removed),
            (
                false,
                Some(FieldColliderDiagnostic::InputMissing {
                    input: FieldColliderInput::B
                })
            )
        );

        // No active transplant: there is no carrier to advect.
        let mut inert = base;
        inert.transplant.amount = 0.0;
        assert_eq!(
            diagnostic_of(inert),
            (false, Some(FieldColliderDiagnostic::NoActiveTransplant))
        );
    }

    #[test]
    fn a_collider_input_may_name_its_own_recipient_and_still_be_admitted() {
        // A layer colliding its own field against another layer's is authored
        // topology, not a cycle. Only A-aliases-B is refused.
        let recipient = collider_recipient(
            FieldColliderMode::Difference,
            MotionBoundaryMode::Hold,
            10,
            30,
        );
        let advanced = plan_collider(recipient);
        let motion = advanced.motion().advanced().unwrap();
        let plan = motion.collider().expect("a self-naming input is admitted");
        assert_eq!(plan.recipient_scope, VisualScopeId::Layer(layer_id(10)));
        assert_eq!(plan.input_a_scope, VisualScopeId::Layer(layer_id(10)));
        assert_eq!(plan.input_b_scope, VisualScopeId::Layer(layer_id(30)));
        // The recipient now carries its OWN primitive field, because it is
        // also a donor, and that field is what a routed primitive consumer
        // observes.
        let recipient_scope = motion.scope(plan.recipient_scope).unwrap();
        assert!(recipient_scope.required_as_donor);
        assert_eq!(
            recipient_scope.admitted_field_slot(),
            Some(plan.input_a_slot)
        );
        // Only a Source diagnostic (the deterministic lattice fallback these
        // fixtures always take) may be present; no collider fault.
        assert!(!motion
            .diagnostics()
            .iter()
            .any(|entry| matches!(entry, MotionPlanDiagnostic::FieldCollider { .. })));
    }

    #[test]
    fn the_collider_topology_signature_tracks_every_discrete_authored_choice() {
        let signature = |mode, boundary, a, b| {
            plan_collider(collider_recipient(mode, boundary, a, b))
                .motion()
                .topology_signature()
        };
        let reference = signature(
            FieldColliderMode::Sum,
            MotionBoundaryMode::Transparent,
            20,
            30,
        );
        assert_ne!(
            reference,
            signature(
                FieldColliderMode::Curl,
                MotionBoundaryMode::Transparent,
                20,
                30
            ),
            "a mode change must invalidate prepared bindings"
        );
        assert_ne!(
            reference,
            signature(FieldColliderMode::Sum, MotionBoundaryMode::Mirror, 20, 30),
            "a boundary change must invalidate prepared bindings"
        );
        assert_ne!(
            reference,
            signature(
                FieldColliderMode::Sum,
                MotionBoundaryMode::Transparent,
                30,
                20
            ),
            "swapping the two inputs is a different dependency graph"
        );

        let mut disabled = collider_recipient(
            FieldColliderMode::Sum,
            MotionBoundaryMode::Transparent,
            20,
            30,
        );
        disabled.collider.enabled = false;
        disabled.transplant.donor = MotionDonor::Selected {
            layer_id: layer_id(20),
            saved_position: saved_position(1),
        };
        assert_ne!(
            reference,
            plan_collider(disabled).motion().topology_signature(),
            "enabling the collider is a topology change"
        );
    }

    #[test]
    fn a_default_collider_never_holds_the_plan_out_of_legacy_exact() {
        // The exact pre-collider path: no motion effect anywhere, so the plan
        // must still collapse to LegacyExact and allocate nothing.
        let base = base(&[10, 20], &[]);
        let composition = legacy_composition(&[10, 20]);
        let racks = legacy_racks(&[10, 20]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = [
            LayerMotionPlanInput {
                stable_id: layer_id(10),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: layer_id(20),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            },
        ];
        let evaluated = plan_with_motion(
            &base,
            &composition,
            &master,
            &racks,
            MotionParams::default(),
            &layers,
        )
        .unwrap();
        assert!(
            matches!(evaluated, EvaluatedCompositionPlan::LegacyExact(_)),
            "a default collider must not hold the composition out of LegacyExact"
        );
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
                pattern: None,
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
            required_as_garden_signal: false,
        };
        let attachment = MotionFieldAttachment {
            scope: plan.scope,
            source_generation: 7,
            frame_ordinal: 9,
            product_content_sha256: [7; 32],
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
    fn routed_garden_matte_is_a_stable_current_frame_tap_with_a_three_texture_pass() {
        let mut temporal = TemporalParams::default();
        temporal.originals.garden.amount = 0.8;
        temporal.originals.garden.gate = RefreshGardenGate::Matte;
        temporal.originals.garden.matte_route = RefreshGardenMatteRoute::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
            stage: LayerImageStage::PostLocalEffects,
        };
        let base = base_with_temporal(&temporal);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let advanced = advanced(plan(&base, &composition, &master, &racks).unwrap());

        assert_eq!(
            advanced.refresh_garden_signal(),
            EvaluatedRefreshGardenSignalPlan::Matte { valid: true }
        );
        assert_eq!(
            advanced.refresh_garden_resources(),
            RefreshGardenResourcePlan {
                full_frame_passes: 1,
                low_resolution_passes: 0,
                logical_texture_lookups_per_pixel: 3,
                texture_operations_per_pixel: 6,
                max_sampled_textures_in_pass: 3,
            }
        );
        let tap = advanced
            .image_taps()
            .iter()
            .find(|tap| tap.consumer == ImageTapConsumer::RefreshGardenMatte)
            .expect("selected Garden matte tap");
        assert_eq!(tap.origin.timing(), EdgeTiming::CurrentFrame);
        assert_eq!(
            tap.resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert!(advanced.diagnostics().is_empty());
    }

    #[test]
    fn routed_garden_motion_admits_the_selected_zero_effect_field_and_signal_ledger() {
        let mut temporal = TemporalParams::default();
        temporal.originals.garden.amount = 1.0;
        temporal.originals.garden.gate = RefreshGardenGate::Motion;
        temporal.originals.garden.motion_route = RefreshGardenMotionRoute::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
        };
        let base = base_with_temporal(&temporal);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = [LayerMotionPlanInput {
            stable_id: layer_id(1),
            params: MotionParams::default(),
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
        let motion = advanced.motion().advanced().expect("Garden motion plan");
        assert_eq!(motion.fields().len(), 1);
        assert_eq!(motion.fields()[0].scope, VisualScopeId::Layer(layer_id(1)));
        assert!(motion.fields()[0].required_as_garden_signal);
        assert!(!motion.fields()[0].required_as_donor);
        assert_eq!(motion.resources().active_garden_signals, 1);
        assert_eq!(
            motion.resources().garden_signal_bytes,
            motion.fields()[0].grid.vector_count
        );
        assert_eq!(motion.resources().persistent_carriers, 0);
        assert!(matches!(
            advanced.refresh_garden_signal(),
            EvaluatedRefreshGardenSignalPlan::Motion {
                layer_id: Some(id),
                field_slot: Some(0),
                valid: true,
            } if id == layer_id(1)
        ));
        assert_eq!(advanced.refresh_garden_resources().low_resolution_passes, 1);
        assert_eq!(
            advanced
                .refresh_garden_resources()
                .max_sampled_textures_in_pass,
            3
        );
    }

    #[test]
    fn routed_garden_motion_observes_the_selected_layers_admitted_donor_field() {
        let mut temporal = TemporalParams::default();
        temporal.originals.garden.amount = 1.0;
        temporal.originals.garden.gate = RefreshGardenGate::Motion;
        temporal.originals.garden.motion_route = RefreshGardenMotionRoute::SelectedLayer {
            layer_id: layer_id(10),
            saved_position: saved_position(0),
        };
        let base = base_with_temporal_for(&[10, 20], &temporal);
        let composition = legacy_composition(&[10, 20]);
        let racks = legacy_racks(&[10, 20]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = [
            LayerMotionPlanInput {
                stable_id: layer_id(10),
                params: MotionParams {
                    transplant: crate::motion::FaradayParams {
                        amount: 0.8,
                        donor: MotionDonor::Selected {
                            layer_id: layer_id(20),
                            saved_position: saved_position(1),
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                },
                codec: MotionCodecFrameFacts::default(),
            },
            LayerMotionPlanInput {
                stable_id: layer_id(20),
                params: MotionParams::default(),
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
        let motion = advanced.motion().advanced().expect("Garden motion plan");
        assert_eq!(motion.fields().len(), 1);
        assert_eq!(motion.fields()[0].scope, VisualScopeId::Layer(layer_id(20)));
        assert!(motion.fields()[0].required_as_donor);
        assert!(motion.fields()[0].required_as_garden_signal);
        let selected = motion
            .scope(VisualScopeId::Layer(layer_id(10)))
            .expect("selected recipient scope");
        assert!(selected.transplant_admitted);
        assert_eq!(selected.admitted_field_slot(), Some(0));
        assert!(matches!(
            advanced.refresh_garden_signal(),
            EvaluatedRefreshGardenSignalPlan::Motion {
                layer_id: Some(id),
                field_slot: Some(0),
                valid: true,
            } if id == layer_id(10)
        ));
        assert_eq!(motion.resources().active_garden_signals, 1);
    }

    #[test]
    fn routed_garden_missing_and_unavailable_routes_are_explicit_closed_signals() {
        let mut temporal = TemporalParams::default();
        temporal.originals.garden.amount = 1.0;
        temporal.originals.garden.gate = RefreshGardenGate::Motion;
        temporal.originals.garden.motion_route = RefreshGardenMotionRoute::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
        };
        let base = base_with_temporal(&temporal);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let unavailable = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(matches!(
            unavailable.refresh_garden_signal(),
            EvaluatedRefreshGardenSignalPlan::Motion {
                layer_id: Some(id),
                field_slot: None,
                valid: false,
            } if id == layer_id(1)
        ));
        assert!(unavailable
            .motion()
            .advanced()
            .unwrap()
            .diagnostics()
            .contains(&MotionPlanDiagnostic::RefreshGardenMotionUnavailable));
        assert_eq!(
            unavailable.refresh_garden_resources().low_resolution_passes,
            0
        );

        temporal.originals.garden.motion_route = RefreshGardenMotionRoute::MissingSelectedLayer {
            saved_position: saved_position(0),
        };
        let missing_base = base_with_temporal(&temporal);
        let missing = advanced(plan(&missing_base, &composition, &master, &racks).unwrap());
        assert_eq!(
            missing.refresh_garden_signal(),
            EvaluatedRefreshGardenSignalPlan::Motion {
                layer_id: None,
                field_slot: None,
                valid: false,
            }
        );
        assert!(missing.motion().advanced().unwrap().diagnostics().contains(
            &MotionPlanDiagnostic::MissingRefreshGardenMotion {
                saved_position: saved_position(0),
            }
        ));
        assert_eq!(missing.refresh_garden_resources().low_resolution_passes, 0);
    }

    #[test]
    fn routed_garden_is_dormant_when_the_master_temporal_marker_is_absent() {
        let mut temporal = TemporalParams::default();
        temporal.originals.garden.amount = 1.0;
        temporal.originals.garden.gate = RefreshGardenGate::Motion;
        temporal.originals.garden.motion_route = RefreshGardenMotionRoute::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
        };
        let base = base_with_temporal(&temporal);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::empty();
        let layers = [LayerMotionPlanInput {
            stable_id: layer_id(1),
            params: MotionParams::default(),
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
        assert_eq!(
            advanced.refresh_garden_signal(),
            EvaluatedRefreshGardenSignalPlan::Inline
        );
        assert_eq!(
            advanced.refresh_garden_resources(),
            RefreshGardenResourcePlan::default()
        );
        assert!(advanced.motion().is_legacy_exact());
    }

    #[test]
    fn authored_routes_at_zero_garden_amount_preserve_literal_legacy_exact() {
        let mut temporal = TemporalParams::default();
        temporal.originals.garden.gate = RefreshGardenGate::Matte;
        temporal.originals.garden.matte_route = RefreshGardenMatteRoute::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
            stage: LayerImageStage::PostLocalEffects,
        };
        let base = base_with_temporal(&temporal);
        let composition = legacy_composition(&[1]);
        let racks = legacy_racks(&[1]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        assert!(matches!(
            plan(&base, &composition, &master, &racks).unwrap(),
            EvaluatedCompositionPlan::LegacyExact(_)
        ));
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
                        ImageTapConsumer::RackNode { scope: owner, node_id: id, .. }
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

    fn displace_tap_for(plan: &AdvancedCompositionPlan, node: NodeId) -> Option<&PlannedImageTap> {
        plan.image_taps().iter().find(|tap| {
            matches!(
                tap.consumer,
                ImageTapConsumer::RackNode { node_id, .. } if node_id == node
            )
        })
    }

    #[test]
    fn displace_collects_its_donor_only_while_enabled_wet_and_nonzero() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let donor = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };

        // Exact-default Displace: no tap, no dependency, no binding slot.
        let mut racks = legacy_racks(&[1, 2]);
        let node = push_displace(&mut racks[1].1, donor, EdgeTiming::CurrentFrame, 0.0);
        let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(
            displace_tap_for(&compiled, node).is_none(),
            "a zero-gain Displace must not claim a cross-scope donor"
        );

        // Either axis alone wakes the same authored route.
        for (amount_x, amount_y) in [(0.5_f32, 0.0_f32), (0.0, -0.5)] {
            let mut racks = legacy_racks(&[1, 2]);
            let node = racks[1]
                .1
                .push(RuntimeVisualNodeKind::Displace(
                    crate::visual_rack::RuntimeDisplaceParams {
                        tap: ResolvedImageTap {
                            source: donor,
                            timing: EdgeTiming::CurrentFrame,
                        },
                        amount_x,
                        amount_y,
                        boundary: crate::visual_rack::DisplaceBoundary::Wrap,
                    },
                ))
                .unwrap();
            let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
            let tap = displace_tap_for(&compiled, node)
                .expect("a live Displace collects exactly one donor tap");
            assert_eq!(
                tap.resolved,
                PlannedImageSource::SelectedLayer {
                    layer_id: layer_id(1),
                    stage: LayerImageStage::PostLocalEffects,
                }
            );
        }

        // Disabled and zero-wet nodes stay dormant even with live gains.
        let mutations: [fn(&mut RuntimeVisualNode); 2] =
            [|node| node.enabled = false, |node| node.wet = 0.0];
        for mutate in mutations {
            let mut racks = legacy_racks(&[1, 2]);
            let node = push_displace(&mut racks[1].1, donor, EdgeTiming::CurrentFrame, 0.75);
            mutate(racks[1].1.get_mut(node).unwrap());
            let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
            assert!(displace_tap_for(&compiled, node).is_none());
        }
    }

    #[test]
    fn displace_self_route_cycles_on_the_current_frame_but_n_minus_one_is_admitted() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let own_output = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(2),
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };

        // A layer whose Displace reads its own post-local output this frame.
        let mut racks = legacy_racks(&[1, 2]);
        push_displace(&mut racks[1].1, own_output, EdgeTiming::CurrentFrame, 0.5);
        let Err(error) = plan(&base, &composition, &master, &racks) else {
            panic!("a current-frame self route must be rejected before allocation");
        };
        // The image dependency graph rejects it before scope ordering runs, and
        // names the offending scope rather than failing anonymously.
        assert!(
            matches!(
                error,
                CompositionPlanError::ImageGraph(ImageGraphError::CurrentCycle { ref scopes })
                    if scopes.as_slice() == [VisualScopeId::Layer(layer_id(2))]
            ),
            "unexpected rejection for a Displace self route: {error:?}"
        );

        // The identical route at N-1 is a legitimate feedback edge.
        let mut racks = legacy_racks(&[1, 2]);
        push_displace(&mut racks[1].1, own_output, EdgeTiming::PreviousFrame, 0.5);
        let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(compiled.graph().previous_taps, 1);

        // A zero-gain self route collects nothing, so it cannot cycle at all.
        let mut racks = legacy_racks(&[1, 2]);
        push_displace(&mut racks[1].1, own_output, EdgeTiming::CurrentFrame, 0.0);
        assert!(plan(&base, &composition, &master, &racks).is_ok());
    }

    fn plan_with_gesture_canvas(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
        admitted: bool,
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        EvaluatedCompositionPlan::evaluate(
            base,
            CompositionPlanInput::new(composition, master, racks).with_gesture_canvas(admitted),
        )
    }

    #[test]
    fn a_gesture_canvas_donor_plans_outside_scope_ordering_and_charges_no_tap_surface() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);

        // Baseline: the same topology, advanced for the same reason, but with
        // no image route at all.
        let mut bare_master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        bare_master
            .push(RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
                invert: 1.0,
                ..DigitalColorParams::default()
            }))
            .unwrap();
        let bare = advanced(
            plan_with_gesture_canvas(
                &base,
                &composition,
                &bare_master,
                &legacy_racks(&[1, 2]),
                true,
            )
            .unwrap(),
        );

        // Every scope routes a same-frame donor at the canvas at once —
        // including the master, which is the scope that owns it and would be
        // the self-cycle if the canvas were an ordinary producer.
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let master_node = push_displace(
            &mut master,
            ResolvedImageSource::GestureCanvas,
            EdgeTiming::CurrentFrame,
            0.5,
        );
        let mut racks = legacy_racks(&[1, 2]);
        let lower_node = push_displace(
            &mut racks[0].1,
            ResolvedImageSource::GestureCanvas,
            EdgeTiming::CurrentFrame,
            0.5,
        );
        let upper_node = push_displace(
            &mut racks[1].1,
            ResolvedImageSource::GestureCanvas,
            EdgeTiming::PreviousFrame,
            -0.25,
        );
        let compiled =
            advanced(plan_with_gesture_canvas(&base, &composition, &master, &racks, true).unwrap());

        for node in [master_node, lower_node, upper_node] {
            let tap = displace_tap_for(&compiled, node).expect("the canvas donor is admitted");
            assert_eq!(tap.resolved, PlannedImageSource::GestureCanvas);
        }
        assert!(!compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::GestureCanvasUnavailable { .. }
        )));

        // A producer with no scope claims no dependency at either timing, so it
        // takes part in no ordering and cannot close a cycle.
        assert_eq!(compiled.graph().current_taps, 0);
        assert_eq!(compiled.graph().previous_taps, 0);
        assert_eq!(
            compiled.graph().current_topological_order,
            bare.graph().current_topological_order
        );

        // The canvas surface is charged once, by `GestureCanvasPlan`. Routing
        // to it must not also claim a retained tap surface in the composition
        // ledger, which `validate_actual_surface_ledger` reconciles exactly.
        assert_eq!(
            compiled.resources().rgba16_surface_layers,
            bare.resources().rgba16_surface_layers,
            "a canvas route must not add a retained tap surface"
        );
        assert_eq!(
            compiled.resources().creative_bytes,
            bare.resources().creative_bytes
        );
    }

    #[test]
    fn an_unadmitted_gesture_canvas_route_is_transparent_and_named_rather_than_rebound() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        let node = push_displace(
            &mut racks[1].1,
            ResolvedImageSource::GestureCanvas,
            EdgeTiming::CurrentFrame,
            0.5,
        );

        let compiled = advanced(
            plan_with_gesture_canvas(&base, &composition, &master, &racks, false).unwrap(),
        );
        let tap = displace_tap_for(&compiled, node)
            .expect("an unavailable canvas still reports a diagnostic tap");
        assert_eq!(
            tap.resolved,
            PlannedImageSource::Transparent,
            "an unadmitted canvas resolves transparent, exactly as a missing donor does"
        );
        assert!(compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::GestureCanvasUnavailable { consumer }
                if matches!(consumer, ImageTapConsumer::RackNode { node_id, .. } if *node_id == node)
        )));
        // It is never quietly repointed at the layer below, a group, or the
        // program: that is the whole reason the diagnostic exists.
        assert_eq!(compiled.graph().current_taps, 0);
        assert_eq!(compiled.graph().previous_taps, 0);

        // Admission is the only difference. The identical rack, planned with a
        // canvas, resolves to the field.
        let admitted =
            advanced(plan_with_gesture_canvas(&base, &composition, &master, &racks, true).unwrap());
        assert_eq!(
            displace_tap_for(&admitted, node).unwrap().resolved,
            PlannedImageSource::GestureCanvas
        );
    }

    fn plan_with_program_tap(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
        admitted: bool,
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        EvaluatedCompositionPlan::evaluate(
            base,
            CompositionPlanInput::new(composition, master, racks).with_program_tap(admitted),
        )
    }

    #[test]
    fn a_program_tap_donor_plans_outside_scope_ordering_and_charges_no_tap_surface() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);

        // Baseline: the same topology, advanced for the same reason, but with
        // no image route at all.
        let mut bare_master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        bare_master
            .push(RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
                invert: 1.0,
                ..DigitalColorParams::default()
            }))
            .unwrap();
        let bare = advanced(
            plan_with_program_tap(
                &base,
                &composition,
                &bare_master,
                &legacy_racks(&[1, 2]),
                true,
            )
            .unwrap(),
        );

        // Every scope routes a same-frame donor at the tap at once — including
        // the master, whose own resolve is the very image the tap copies and
        // which would be the self-cycle if the tap were an ordinary producer.
        // The tap is N-1 by construction, so no same-frame cycle is
        // expressible.
        let mut master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let master_node = push_displace(
            &mut master,
            ResolvedImageSource::ProgramTap,
            EdgeTiming::CurrentFrame,
            0.5,
        );
        let mut racks = legacy_racks(&[1, 2]);
        let lower_node = push_displace(
            &mut racks[0].1,
            ResolvedImageSource::ProgramTap,
            EdgeTiming::CurrentFrame,
            0.5,
        );
        let upper_node = push_displace(
            &mut racks[1].1,
            ResolvedImageSource::ProgramTap,
            EdgeTiming::PreviousFrame,
            -0.25,
        );
        let compiled =
            advanced(plan_with_program_tap(&base, &composition, &master, &racks, true).unwrap());

        for node in [master_node, lower_node, upper_node] {
            let tap = displace_tap_for(&compiled, node).expect("the tap donor is admitted");
            assert_eq!(tap.resolved, PlannedImageSource::ProgramTap);
        }
        assert!(!compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::ProgramTapUnavailable { .. }
        )));

        // A producer with no scope claims no dependency at either timing, so it
        // takes part in no ordering and cannot close a cycle. An N-1 route to
        // it also stages nothing: the tap *is* the N-1 image.
        assert_eq!(compiled.graph().current_taps, 0);
        assert_eq!(compiled.graph().previous_taps, 0);
        assert_eq!(
            compiled.graph().current_topological_order,
            bare.graph().current_topological_order
        );

        // The tap surface is charged once, by the renderer-owned full-frame
        // texture floor. Routing to it must not also claim a retained tap
        // surface in the composition ledger, which
        // `validate_actual_surface_ledger` reconciles exactly.
        assert_eq!(
            compiled.resources().rgba16_surface_layers,
            bare.resources().rgba16_surface_layers,
            "a programme-tap route must not add a retained tap surface"
        );
        assert_eq!(
            compiled.resources().creative_bytes,
            bare.resources().creative_bytes
        );
    }

    #[test]
    fn an_unavailable_program_tap_route_is_transparent_and_named_rather_than_rebound() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        let node = push_displace(
            &mut racks[1].1,
            ResolvedImageSource::ProgramTap,
            EdgeTiming::CurrentFrame,
            0.5,
        );

        // Before the first committed frame the tap holds nothing, and the
        // route must say so by name rather than borrowing another producer.
        let compiled =
            advanced(plan_with_program_tap(&base, &composition, &master, &racks, false).unwrap());
        let tap = displace_tap_for(&compiled, node)
            .expect("an unavailable tap still reports a diagnostic tap");
        assert_eq!(
            tap.resolved,
            PlannedImageSource::Transparent,
            "an uncommitted tap resolves transparent, exactly as a missing donor does"
        );
        assert!(compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::ProgramTapUnavailable { consumer }
                if matches!(consumer, ImageTapConsumer::RackNode { node_id, .. } if *node_id == node)
        )));
        // It is never quietly repointed at the layer below, a group, or the
        // program scope: that is the whole reason the diagnostic exists.
        assert_eq!(compiled.graph().current_taps, 0);
        assert_eq!(compiled.graph().previous_taps, 0);

        // Admission is the only difference. The identical rack, planned after
        // the first committed frame, resolves to the tap.
        let admitted =
            advanced(plan_with_program_tap(&base, &composition, &master, &racks, true).unwrap());
        assert_eq!(
            displace_tap_for(&admitted, node).unwrap().resolved,
            PlannedImageSource::ProgramTap
        );
    }

    #[test]
    fn displace_removal_leaves_a_tombstone_that_never_rebinds_after_replacement() {
        let frame = base(&[1, 2], &[]);
        let replaced_frame = base(&[3, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let saved = SavedLayerPosition::new(0).unwrap();

        let mut racks = legacy_racks(&[1, 2]);
        let node = push_displace(
            &mut racks[1].1,
            ResolvedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                saved_position: saved,
                stage: LayerImageStage::PostLocalEffects,
            },
            EdgeTiming::CurrentFrame,
            0.5,
        );
        let compiled = advanced(plan(&frame, &composition, &master, &racks).unwrap());
        assert!(matches!(
            displace_tap_for(&compiled, node).unwrap().resolved,
            PlannedImageSource::SelectedLayer { .. }
        ));

        // Deleting the donor tombstones the route. A different layer later
        // occupying that saved position must not inherit it.
        racks[1].1.mark_layer_output_missing(layer_id(1));
        let replaced_composition = legacy_composition(&[3, 2]);
        let mut replaced_racks = legacy_racks(&[3, 2]);
        let slot = replaced_racks
            .iter_mut()
            .find(|(id, _)| *id == layer_id(2))
            .expect("layer 2 survives the replacement");
        slot.1 = racks[1].1.clone();
        let compiled = advanced(
            plan(
                &replaced_frame,
                &replaced_composition,
                &master,
                &replaced_racks,
            )
            .unwrap(),
        );
        let tap = displace_tap_for(&compiled, node)
            .expect("a missing donor still reports a diagnostic tap");
        assert_eq!(tap.resolved, PlannedImageSource::Transparent);
        assert!(compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::MissingSelectedLayer {
                saved_position, ..
            } if *saved_position == saved
        )));
    }

    #[test]
    fn reordering_the_stack_never_slides_a_displace_donor_onto_another_layer() {
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let donor = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let consumer_rack = |amount_x| {
            let mut rack = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
            let node = push_displace(&mut rack, donor, EdgeTiming::CurrentFrame, amount_x);
            (rack, node)
        };

        // The same authored route, evaluated under two different stack orders.
        let (rack, node) = consumer_rack(0.5);
        let mut forward = legacy_racks(&[1, 2]);
        forward[1].1 = rack.clone();
        let mut reversed = legacy_racks(&[1, 2]);
        reversed[1].1 = rack;

        for (frame, composition, racks) in [
            (base(&[1, 2], &[]), legacy_composition(&[1, 2]), &forward),
            (base(&[2, 1], &[]), legacy_composition(&[2, 1]), &reversed),
        ] {
            let compiled = advanced(plan(&frame, &composition, &master, racks).unwrap());
            let tap = displace_tap_for(&compiled, node).expect("the donor survives reorder");
            assert_eq!(
                tap.resolved,
                PlannedImageSource::SelectedLayer {
                    layer_id: layer_id(1),
                    stage: LayerImageStage::PostLocalEffects,
                },
                "a stable-ID donor must follow its layer, never a stack position"
            );
        }

        // Moving the node inside its own rack is likewise route-inert.
        let (mut rack, node) = consumer_rack(0.5);
        let before = rack.get(node).unwrap().kind;
        rack.move_node(node, 0, LegacyRackScope::Layer).unwrap();
        assert_eq!(rack.iter().next().unwrap().stable_id, node);
        assert_eq!(rack.get(node).unwrap().kind, before);
    }

    #[test]
    fn residual_collects_both_slots_only_while_enabled_wet_and_live() {
        let base = base(&[1, 2, 3], &[]);
        let composition = legacy_composition(&[1, 2, 3]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let structure = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let detail = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(2),
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let authored = |mix| {
            residual_params(
                (structure, EdgeTiming::CurrentFrame),
                (detail, EdgeTiming::CurrentFrame),
                mix,
            )
        };

        // Exact-default Residual: zero mix collects neither slot and charges
        // no reduced surface.
        let mut racks = legacy_racks(&[1, 2, 3]);
        let node = push_residual(&mut racks[2].1, authored(0.0));
        let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert!(residual_tap_for(&compiled, node, RESIDUAL_STRUCTURE_SLOT).is_none());
        assert!(residual_tap_for(&compiled, node, RESIDUAL_DETAIL_SLOT).is_none());
        assert_eq!(
            compiled.residual_resources(),
            ResidualResourcePlan::default()
        );

        // A live mix wakes both slots at once, each onto its own donor.
        let mut racks = legacy_racks(&[1, 2, 3]);
        let node = push_residual(&mut racks[2].1, authored(0.4));
        let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
        assert_eq!(
            residual_tap_for(&compiled, node, RESIDUAL_STRUCTURE_SLOT)
                .unwrap()
                .resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            residual_tap_for(&compiled, node, RESIDUAL_DETAIL_SLOT)
                .unwrap()
                .resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(2),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            compiled
                .image_taps()
                .iter()
                .filter(|tap| matches!(
                    tap.consumer,
                    ImageTapConsumer::RackNode { node_id, .. } if node_id == node
                ))
                .count(),
            2,
            "two authored slots are two consumer identities, never one aliased tap"
        );
        assert_eq!(compiled.residual_resources().active_nodes, 1);
        assert_eq!(compiled.residual_resources().mean_surfaces, 2);
        assert_eq!(compiled.residual_resources().total_bytes, 8 * 8 * 2 * 8);

        // Disabled and zero-wet nodes stay dormant on both slots.
        let mutations: [fn(&mut RuntimeVisualNode); 2] =
            [|node| node.enabled = false, |node| node.wet = 0.0];
        for mutate in mutations {
            let mut racks = legacy_racks(&[1, 2, 3]);
            let node = push_residual(&mut racks[2].1, authored(0.4));
            mutate(racks[2].1.get_mut(node).unwrap());
            let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
            assert!(residual_tap_for(&compiled, node, RESIDUAL_STRUCTURE_SLOT).is_none());
            assert!(residual_tap_for(&compiled, node, RESIDUAL_DETAIL_SLOT).is_none());
            assert_eq!(compiled.residual_resources().active_nodes, 0);
        }

        // Hostile non-finite mix collapses to bypass, never to full mix.
        for hostile in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut racks = legacy_racks(&[1, 2, 3]);
            let node = push_residual(&mut racks[2].1, authored(hostile));
            let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
            assert!(residual_tap_for(&compiled, node, RESIDUAL_STRUCTURE_SLOT).is_none());
            assert!(residual_tap_for(&compiled, node, RESIDUAL_DETAIL_SLOT).is_none());
        }
    }

    #[test]
    fn residual_self_route_cycles_per_slot_on_the_current_frame_but_n_minus_one_is_admitted() {
        let base = base(&[1, 2, 3], &[]);
        let composition = legacy_composition(&[1, 2, 3]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let neighbour = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let own_output = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(3),
            saved_position: SavedLayerPosition::new(2).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };

        for slot in [RESIDUAL_STRUCTURE_SLOT, RESIDUAL_DETAIL_SLOT] {
            let route_for = |timing, mix| {
                let mut params = residual_params(
                    (neighbour, EdgeTiming::CurrentFrame),
                    (neighbour, EdgeTiming::CurrentFrame),
                    mix,
                );
                *params.route_mut(slot).expect("both slots name a route") = ResolvedImageTap {
                    source: own_output,
                    timing,
                };
                params
            };

            let mut racks = legacy_racks(&[1, 2, 3]);
            push_residual(&mut racks[2].1, route_for(EdgeTiming::CurrentFrame, 0.6));
            let Err(error) = plan(&base, &composition, &master, &racks) else {
                panic!("a current-frame self route on slot {slot} must be rejected");
            };
            // The graph rejects it before scope ordering runs and names the
            // offending scope rather than failing anonymously.
            assert!(
                matches!(
                    error,
                    CompositionPlanError::ImageGraph(ImageGraphError::CurrentCycle { ref scopes })
                        if scopes.as_slice() == [VisualScopeId::Layer(layer_id(3))]
                ),
                "unexpected rejection for a Residual self route on slot {slot}: {error:?}"
            );

            // The identical route at N-1 is a legitimate feedback edge, and
            // the other slot keeps its ordinary current-frame donor.
            let mut racks = legacy_racks(&[1, 2, 3]);
            let node = push_residual(&mut racks[2].1, route_for(EdgeTiming::PreviousFrame, 0.6));
            let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
            assert_eq!(compiled.graph().previous_taps, 1);
            assert_eq!(
                residual_tap_for(&compiled, node, slot).unwrap().resolved,
                PlannedImageSource::SelectedLayer {
                    layer_id: layer_id(3),
                    stage: LayerImageStage::PostLocalEffects,
                }
            );

            // A zero-mix self route collects nothing, so it cannot cycle.
            let mut racks = legacy_racks(&[1, 2, 3]);
            push_residual(&mut racks[2].1, route_for(EdgeTiming::CurrentFrame, 0.0));
            assert!(plan(&base, &composition, &master, &racks).is_ok());
        }
    }

    #[test]
    fn residual_slot_routes_never_alias_and_a_slot_one_change_moves_the_topology_signature() {
        let base = base(&[1, 2, 3], &[]);
        let composition = legacy_composition(&[1, 2, 3]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let first = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let second = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(2),
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let signature_of = |params: crate::visual_rack::RuntimeResidualParams| {
            let mut racks = legacy_racks(&[1, 2, 3]);
            push_residual(&mut racks[2].1, params);
            advanced(plan(&base, &composition, &master, &racks).unwrap()).topology_signature()
        };
        let authored = |detail: ResolvedImageSource, timing| {
            residual_params((first, EdgeTiming::CurrentFrame), (detail, timing), 0.5)
        };

        // Identical authored state is an identical signature.
        assert_eq!(
            signature_of(authored(first, EdgeTiming::CurrentFrame)),
            signature_of(authored(first, EdgeTiming::CurrentFrame))
        );

        // Changing only slot 1's source or timing must move it, or a prepared
        // executor reuses the previous frame bindings for a rewritten graph.
        assert_ne!(
            signature_of(authored(first, EdgeTiming::CurrentFrame)),
            signature_of(authored(second, EdgeTiming::CurrentFrame))
        );
        assert_ne!(
            signature_of(authored(second, EdgeTiming::CurrentFrame)),
            signature_of(authored(second, EdgeTiming::PreviousFrame))
        );

        // The block vocabulary drives the reduced grid dimensions, which the
        // rack signature deliberately does not carry.
        let with_block = |block| {
            let mut params = authored(second, EdgeTiming::CurrentFrame);
            params.block = block;
            params
        };
        assert_ne!(
            signature_of(with_block(crate::visual_rack::ResidualBlock::Eight)),
            signature_of(with_block(crate::visual_rack::ResidualBlock::Sixteen))
        );
    }

    #[test]
    fn residual_removal_leaves_a_per_slot_tombstone_that_never_rebinds_after_replacement() {
        let frame = base(&[1, 2, 3], &[]);
        let replaced_frame = base(&[4, 2, 3], &[]);
        let composition = legacy_composition(&[1, 2, 3]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let saved = SavedLayerPosition::new(0).unwrap();
        let structure = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: saved,
            stage: LayerImageStage::PostLocalEffects,
        };
        let detail = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(2),
            saved_position: SavedLayerPosition::new(1).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };

        let mut racks = legacy_racks(&[1, 2, 3]);
        let node = push_residual(
            &mut racks[2].1,
            residual_params(
                (structure, EdgeTiming::CurrentFrame),
                (detail, EdgeTiming::CurrentFrame),
                0.5,
            ),
        );
        let compiled = advanced(plan(&frame, &composition, &master, &racks).unwrap());
        assert!(matches!(
            residual_tap_for(&compiled, node, RESIDUAL_STRUCTURE_SLOT)
                .unwrap()
                .resolved,
            PlannedImageSource::SelectedLayer { .. }
        ));

        // Deleting the structure donor tombstones that slot alone. A different
        // layer later occupying its saved position must not inherit it.
        racks[2].1.mark_layer_output_missing(layer_id(1));
        let replaced_composition = legacy_composition(&[4, 2, 3]);
        let mut replaced_racks = legacy_racks(&[4, 2, 3]);
        let owner = replaced_racks
            .iter_mut()
            .find(|(id, _)| *id == layer_id(3))
            .expect("layer 3 survives the replacement");
        owner.1 = racks[2].1.clone();
        let compiled = advanced(
            plan(
                &replaced_frame,
                &replaced_composition,
                &master,
                &replaced_racks,
            )
            .unwrap(),
        );
        assert_eq!(
            residual_tap_for(&compiled, node, RESIDUAL_STRUCTURE_SLOT)
                .expect("a missing donor still reports a diagnostic tap")
                .resolved,
            PlannedImageSource::Transparent
        );
        assert_eq!(
            residual_tap_for(&compiled, node, RESIDUAL_DETAIL_SLOT)
                .unwrap()
                .resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(2),
                stage: LayerImageStage::PostLocalEffects,
            },
            "the surviving slot keeps its own donor"
        );
        assert!(compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::MissingSelectedLayer {
                consumer: ImageTapConsumer::RackNode { slot, .. },
                saved_position,
            } if *saved_position == saved && *slot == RESIDUAL_STRUCTURE_SLOT
        )));
    }

    #[test]
    fn residual_block_means_join_the_shared_creative_cap_outside_the_full_frame_ledger() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let donor = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };
        let mut racks = legacy_racks(&[1, 2]);
        push_residual(
            &mut racks[1].1,
            residual_params(
                (donor, EdgeTiming::CurrentFrame),
                (donor, EdgeTiming::CurrentFrame),
                0.5,
            ),
        );
        let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());

        // The full-frame ledger is still exactly its own formula: reduced
        // block means are never folded into it as whole output layers.
        let resources = compiled.resources();
        let pixels = 64_u64 * 64;
        assert_eq!(
            resources.creative_bytes,
            pixels * 8 * u64::from(resources.rgba16_surface_layers)
                + pixels * 4 * u64::from(resources.compat8_surface_layers)
        );
        let residual = compiled.residual_resources();
        assert_eq!(residual.active_nodes, 1);
        assert_eq!(residual.max_grid_dimensions, [8, 8]);
        assert_eq!(residual.total_bytes, 1_024);

        // They meet the creative number only at the shared cap, so a limit
        // that admits the full-frame ledger alone still rejects the pair.
        let mut input = CompositionPlanInput::new(&composition, &master, &racks);
        input.resource_limits.max_creative_bytes = resources.creative_bytes;
        let rejected = EvaluatedCompositionPlan::evaluate(&base, input);
        assert!(
            matches!(
                rejected,
                Err(CompositionPlanError::ResidualCombinedMemoryBudget { bytes, limit })
                    if bytes == resources.creative_bytes + residual.total_bytes
                        && limit == resources.creative_bytes
            ),
            "unexpected combined-cap outcome"
        );
    }

    #[test]
    fn residual_admission_agrees_with_the_saved_predicate_while_segmentation_stays_kind_only() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let donor = ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(1),
            saved_position: SavedLayerPosition::new(0).unwrap(),
            stage: LayerImageStage::PostLocalEffects,
        };

        for enabled in [true, false] {
            for wet in [1.0_f32, 0.0] {
                for mix in [0.5_f32, 0.0, f32::NAN] {
                    let mut racks = legacy_racks(&[1, 2]);
                    let node = push_residual(
                        &mut racks[1].1,
                        residual_params(
                            (donor, EdgeTiming::CurrentFrame),
                            (donor, EdgeTiming::CurrentFrame),
                            mix,
                        ),
                    );
                    {
                        let authored = racks[1].1.get_mut(node).unwrap();
                        authored.enabled = enabled;
                        authored.wet = wet;
                    }
                    let RuntimeVisualNodeKind::Residual(live) = racks[1].1.get(node).unwrap().kind
                    else {
                        panic!("the pushed node is a Residual");
                    };
                    // The saved twin's predicate is the one the saved-patch
                    // dependency walk consults; it must agree value for value.
                    let captured = racks[1].1.capture_routes(|_| None).unwrap();
                    let VisualNodeKind::Residual(saved) = captured.iter().last().unwrap().kind
                    else {
                        panic!("the captured node is a Residual");
                    };
                    assert_eq!(saved.is_exact_bypass(), live.is_exact_bypass());
                    let admitted = enabled && wet > 0.0 && !live.is_exact_bypass();

                    let compiled = advanced(plan(&base, &composition, &master, &racks).unwrap());
                    for slot in [RESIDUAL_STRUCTURE_SLOT, RESIDUAL_DETAIL_SLOT] {
                        assert_eq!(
                            residual_tap_for(&compiled, node, slot).is_some(),
                            admitted,
                            "slot {slot} disagreed for enabled={enabled} wet={wet} mix={mix}"
                        );
                    }
                    assert_eq!(
                        compiled.residual_resources().active_nodes,
                        u32::from(admitted)
                    );

                    // Segmentation is deliberately kind-only: the node is
                    // always alone in its own rack segment, so segment indices
                    // never renumber when a frame-local value crosses zero.
                    let owner = compiled
                        .layers()
                        .iter()
                        .find(|layer| layer.stable_id == racks[1].0)
                        .expect("the consumer layer keeps a scope plan");
                    let segments: Vec<_> = owner
                        .execution
                        .steps()
                        .iter()
                        .filter_map(|step| match step {
                            EvaluatedScopeStep::CollisionRack { plan, .. } => Some(plan),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(segments.len(), 1);
                    assert_eq!(segments[0].passes().len(), 1);
                    assert_eq!(segments[0].passes()[0].node_id, node);
                }
            }
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

    // --- S4 Symmetry Field: the evaluator and admission ---

    /// A woken Symmetry Field. Six-fold dihedral geometry keeps it out of the
    /// exact-bypass domain, and each image slot is armed only when a route is
    /// supplied, because the source mask is what admits a slot.
    fn symmetry_params(
        donor0: Option<(ResolvedImageSource, EdgeTiming)>,
        donor1: Option<(ResolvedImageSource, EdgeTiming)>,
    ) -> RuntimeSymmetryParams {
        let mut params = RuntimeSymmetryParams {
            mode: crate::symmetry::SymmetryMode::Dihedral,
            base_folds: 6.0,
            ..RuntimeSymmetryParams::default()
        };
        if let Some((source, timing)) = donor0 {
            params.donors[0] = ResolvedImageTap { source, timing };
            params.source_mask.donor0 = true;
        }
        if let Some((source, timing)) = donor1 {
            params.donors[1] = ResolvedImageTap { source, timing };
            params.source_mask.donor1 = true;
        }
        params
    }

    fn push_symmetry(rack: &mut RuntimeVisualRack, params: RuntimeSymmetryParams) -> NodeId {
        rack.push(RuntimeVisualNodeKind::Symmetry(params)).unwrap()
    }

    fn layer_source(id: u64, position: u32) -> ResolvedImageSource {
        ResolvedImageSource::SelectedLayer {
            layer_id: layer_id(id),
            saved_position: saved_position(position),
            stage: LayerImageStage::PostLocalEffects,
        }
    }

    /// Every tap one node collected, ordered by slot. A first-match lookup
    /// would silently hide the second route, so this deliberately collects all
    /// of them and keys them by slot.
    fn symmetry_taps(plan: &AdvancedCompositionPlan, node: NodeId) -> Vec<(u8, &PlannedImageTap)> {
        let mut found: Vec<_> = plan
            .image_taps()
            .iter()
            .filter_map(|tap| match tap.consumer {
                ImageTapConsumer::RackNode { node_id, slot, .. } if node_id == node => {
                    Some((slot, tap))
                }
                _ => None,
            })
            .collect();
        found.sort_by_key(|(slot, _)| *slot);
        found
    }

    /// Planned layers are in flattened root order, which is the reverse of the
    /// front-to-back fixture order, so a fixture must address a layer by its
    /// stable ID rather than by position.
    fn layer_plan(plan: &AdvancedCompositionPlan, id: u64) -> &EvaluatedLayerScopePlan {
        plan.layers()
            .iter()
            .find(|layer| layer.stable_id == layer_id(id))
            .expect("the planned layer exists")
    }

    fn symmetry_steps(execution: &EvaluatedScopeExecution) -> Vec<&EvaluatedSymmetryFieldPlan> {
        execution
            .steps()
            .iter()
            .filter_map(|step| match step {
                EvaluatedScopeStep::SymmetryField { plan } => Some(plan),
                _ => None,
            })
            .collect()
    }

    /// Every planner fixture otherwise runs at `CreativeResourceLimits::-
    /// default()`, whose sampled-texture stub is the *ordinary rack* ceiling of
    /// three. Production reads the device, whose enforced floor is sixteen, so
    /// an eight-texture dedicated pass has to raise this explicitly.
    fn plan_at_device_floor(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        let mut input = CompositionPlanInput::new(composition, master, racks);
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        EvaluatedCompositionPlan::evaluate(base, input)
    }

    fn plan_with_studies_at_device_floor(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
        studies: &crate::study_eval::StudyProgramLibrary,
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        let mut input = CompositionPlanInput::new(composition, master, racks).with_studies(studies);
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        EvaluatedCompositionPlan::evaluate(base, input)
    }

    /// The Study lift follows the Symmetry shape exactly: flush before, one
    /// dedicated step at the authored position, segmentation resumes behind
    /// it — and the digest resolves against the supplied library at plan
    /// time, never against the ambient host.
    #[test]
    fn a_study_flushes_its_segment_resolves_its_digest_and_charges_the_ledger() {
        use crate::study::{StudyCapability, StudyInstruction};
        use crate::study_eval::tests::{document, register};

        let mut library = crate::study_eval::StudyProgramLibrary::default();
        let digest = library
            .insert(document(
                vec![StudyCapability::CurrentColor],
                vec![
                    StudyInstruction::LoadCurrentColor { dst: register(0) },
                    StudyInstruction::OutputColor { color: register(0) },
                ],
            ))
            .unwrap();

        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let node = racks[1]
            .1
            .push(RuntimeVisualNodeKind::Study(
                crate::visual_rack::StudyRackParams {
                    document_digest: Some(digest),
                },
            ))
            .unwrap();
        racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();

        let compiled = advanced(
            plan_with_studies_at_device_floor(&base, &composition, &master, &racks, &library)
                .unwrap(),
        );
        let steps = layer_plan(&compiled, 2).execution.steps();
        assert!(matches!(
            steps[2],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 0,
                ..
            }
        ));
        let EvaluatedScopeStep::StudyField { plan: field } = &steps[3] else {
            panic!("a Study must own a dedicated step, not a rack segment");
        };
        assert_eq!(field.node_id, node);
        assert_eq!(field.params.document_digest, Some(digest));
        assert!(field.program.is_some(), "the digest must resolve");
        assert_eq!(field.instruction_count, 2);
        assert!(matches!(
            steps[4],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 1,
                ..
            }
        ));
        assert_eq!(steps[3].node_kind_tag(), Some(NodeKindTag::Study));

        // The ledger re-derives from the emitted step: one pass, two
        // simultaneous bindings, the 64 + 8,192 uniform bytes.
        let resources = compiled.study_field_resources();
        assert_eq!(resources.full_frame_passes, 1);
        assert_eq!(resources.max_sampled_textures_in_pass, 2);
        assert_eq!(resources.uniform_bytes, 8_256);

        // Without the library the same authored graph plans an inert pass —
        // program None, never a fallback onto another document — and the
        // topology signature differs, so the renderer re-prepares (and
        // re-uploads the program arena) the moment the digest resolves.
        let unresolved =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        let unresolved_steps = layer_plan(&unresolved, 2).execution.steps();
        let EvaluatedScopeStep::StudyField { plan: inert } = &unresolved_steps[3] else {
            panic!("the dedicated position is kind-only and survives resolution failure");
        };
        assert!(inert.program.is_none());
        assert_eq!(inert.instruction_count, 0);
        assert_ne!(
            compiled.topology_signature(),
            unresolved.topology_signature(),
            "digest resolution is plan-visible identity"
        );

        // The admission budget is a plan-time refusal by name: a valid
        // document with eight history loads (nine loads with the carrier)
        // exceeds the declared eight-lookup budget, while seven admit.
        let heavy_document = |history_loads: u8| {
            let mut instructions: Vec<StudyInstruction> = (0..history_loads)
                .map(|slot| StudyInstruction::LoadHistoryColor {
                    dst: register(slot),
                    age: slot + 1,
                })
                .collect();
            let mut accumulated = register(0);
            for slot in 1..history_loads {
                let dst = register(32 + slot);
                instructions.push(StudyInstruction::Add {
                    dst,
                    left: accumulated,
                    right: register(slot),
                });
                accumulated = dst;
            }
            instructions.push(StudyInstruction::OutputColor { color: accumulated });
            document(vec![StudyCapability::HistoryRead], instructions)
        };
        let mut heavy_library = crate::study_eval::StudyProgramLibrary::default();
        let heavy = heavy_library.insert(heavy_document(8)).unwrap();
        let mut heavy_racks = legacy_racks(&[1, 2]);
        heavy_racks[1]
            .1
            .push(RuntimeVisualNodeKind::Study(
                crate::visual_rack::StudyRackParams {
                    document_digest: Some(heavy),
                },
            ))
            .unwrap();
        let refused = plan_with_studies_at_device_floor(
            &base,
            &composition,
            &master,
            &heavy_racks,
            &heavy_library,
        );
        assert!(matches!(
            refused,
            Err(CompositionPlanError::StudyLoadBudget {
                loads: 9,
                limit: 8,
                ..
            })
        ));
        let seven = heavy_library.insert(heavy_document(7)).unwrap();
        let mut admitted_racks = legacy_racks(&[1, 2]);
        admitted_racks[1]
            .1
            .push(RuntimeVisualNodeKind::Study(
                crate::visual_rack::StudyRackParams {
                    document_digest: Some(seven),
                },
            ))
            .unwrap();
        plan_with_studies_at_device_floor(
            &base,
            &composition,
            &master,
            &admitted_racks,
            &heavy_library,
        )
        .expect("seven history loads plus the carrier fill the budget exactly");
    }

    /// The Scan Processor lift follows the Study shape exactly: flush before,
    /// one kind-only dedicated step at the authored position, segmentation
    /// resumes behind it. The dedicated ledger re-derives from the emitted
    /// steps — two passes, two simultaneous bindings, 128 uniform bytes, the
    /// named vertex budget, and the shared 8-byte-per-pixel transient — and
    /// the topology signature is pass-layout identity only, so a geometry
    /// edit re-encodes without re-preparing.
    #[test]
    fn a_scan_processor_flushes_its_segment_and_charges_the_named_vertex_ledger() {
        use crate::scan_processor::ScanProcessorParams;

        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let node = racks[1]
            .1
            .push(RuntimeVisualNodeKind::ScanProcessor(ScanProcessorParams {
                amount: 0.5,
                lines: 240,
                samples_per_line: 96,
                ..ScanProcessorParams::default()
            }))
            .unwrap();
        racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();

        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        let steps = layer_plan(&compiled, 2).execution.steps();
        assert!(matches!(
            steps[2],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 0,
                ..
            }
        ));
        let EvaluatedScopeStep::ScanProcessorField { plan: field } = &steps[3] else {
            panic!("a Scan Processor must own a dedicated step, not a rack segment");
        };
        assert_eq!(field.node_id, node);
        assert_eq!(field.params.amount, 0.5);
        assert_eq!(field.params.vertex_count(), 240 * 96 * 2);
        assert!(!field.params.is_exact_bypass());
        assert!(matches!(
            steps[4],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 1,
                ..
            }
        ));
        assert_eq!(steps[3].node_kind_tag(), Some(NodeKindTag::ScanProcessor));

        // The dedicated ledger: geometry + resolve, the widest pass binds
        // two textures, one 128-byte record, the summed vertex request, and
        // the 64x64 output's 8-byte-per-pixel shared transient.
        let resources = compiled.scan_processor_resources();
        assert_eq!(resources.full_frame_passes, 2);
        assert_eq!(resources.logical_texture_lookups_per_pixel, 2);
        assert_eq!(resources.max_sampled_textures_in_pass, 2);
        assert_eq!(resources.uniform_bytes, 128);
        assert_eq!(resources.vertices, 240 * 96 * 2);
        assert_eq!(resources.transient_bytes, 64 * 64 * 8);

        // A plan with no scan node charges nothing and carries a different
        // signature; a geometry edit keeps the signature (the vertex count
        // is a draw-call argument, not topology). The control rack carries an
        // ordinary node so it still plans Advanced.
        let mut plain_racks = legacy_racks(&[1, 2]);
        plain_racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let plain =
            advanced(plan_at_device_floor(&base, &composition, &master, &plain_racks).unwrap());
        assert_eq!(
            plain.scan_processor_resources(),
            ScanProcessorResourcePlan::default()
        );
        assert_ne!(compiled.topology_signature(), plain.topology_signature());
        let mut resized_racks = legacy_racks(&[1, 2]);
        resized_racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        resized_racks[1]
            .1
            .push(RuntimeVisualNodeKind::ScanProcessor(ScanProcessorParams {
                amount: 0.5,
                lines: 1_080,
                samples_per_line: 512,
                ..ScanProcessorParams::default()
            }))
            .unwrap();
        resized_racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let resized =
            advanced(plan_at_device_floor(&base, &composition, &master, &resized_racks).unwrap());
        assert_eq!(
            resized.scan_processor_resources().vertices,
            crate::scan_processor::MAX_SCAN_PROCESSOR_VERTICES
        );
        assert_eq!(
            resized.topology_signature(),
            compiled.topology_signature(),
            "geometry is a draw-call argument, never plan-visible topology"
        );

        // A default node is an exact bypass but the lift is kind-only: the
        // dedicated position survives so slot numbering never depends on
        // frame-local values, and the executor skips it at encode.
        let mut bypass_racks = legacy_racks(&[1, 2]);
        bypass_racks[1]
            .1
            .push(RuntimeVisualNodeKind::ScanProcessor(
                ScanProcessorParams::default(),
            ))
            .unwrap();
        let bypassed =
            advanced(plan_at_device_floor(&base, &composition, &master, &bypass_racks).unwrap());
        let bypass_steps = layer_plan(&bypassed, 2).execution.steps();
        let EvaluatedScopeStep::ScanProcessorField { plan: inert } = &bypass_steps[2] else {
            panic!("the dedicated position is kind-only and survives an exact bypass");
        };
        assert!(inert.params.is_exact_bypass());
    }

    fn plan_with_motion_at_device_floor(
        base: &EvaluatedFramePlan,
        composition: &RuntimeComposition,
        master: &RuntimeVisualRack,
        racks: &[(StableLayerId, RuntimeVisualRack)],
        layers: &[LayerMotionPlanInput],
    ) -> Result<EvaluatedCompositionPlan, CompositionPlanError> {
        let mut input = CompositionPlanInput::new(composition, master, racks).with_motion(
            MotionParams::default(),
            layers,
            MotionDeviceLimits::new(8_192, u64::MAX),
        );
        input.resource_limits.max_sampled_textures_per_shader_stage = 16;
        EvaluatedCompositionPlan::evaluate(base, input)
    }

    fn zero_motion_layers(ids: &[u64]) -> Vec<LayerMotionPlanInput> {
        ids.iter()
            .map(|id| LayerMotionPlanInput {
                stable_id: layer_id(*id),
                params: MotionParams::default(),
                codec: MotionCodecFrameFacts::default(),
            })
            .collect()
    }

    #[test]
    fn a_symmetry_field_flushes_its_ordinary_segment_emits_one_dedicated_step_and_resumes() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let node = push_symmetry(
            &mut racks[1].1,
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None),
        );
        racks[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();

        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        let steps = layer_plan(&compiled, 2).execution.steps();
        assert!(matches!(
            steps[0],
            EvaluatedScopeStep::MaterializeSpatial { .. }
        ));
        assert!(matches!(
            steps[1],
            EvaluatedScopeStep::LegacyCanonical { .. }
        ));
        // The ordinary work before the node is flushed into segment 0 ...
        assert!(matches!(
            steps[2],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 0,
                ..
            }
        ));
        // ... the dedicated pass sits at its exact authored position ...
        let EvaluatedScopeStep::SymmetryField { plan: field } = &steps[3] else {
            panic!("a Symmetry Field must own a dedicated step, not a rack segment");
        };
        assert_eq!(field.node_id, node);
        assert!(field.enabled);
        // ... and segmentation resumes behind it with the next segment index.
        assert!(matches!(
            steps[4],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 1,
                ..
            }
        ));
        assert_eq!(steps.len(), 5);
        assert_eq!(
            steps[3].node_kind_tag(),
            Some(NodeKindTag::Symmetry),
            "a dedicated step still reports the authored node kind it came from"
        );

        // Segmentation is kind-only: a disabled, zero-wet, exact-default node
        // still owns its dedicated position, so segment indices and the
        // executor's uniform-slot reservation never move with a value.
        let mut dormant = legacy_racks(&[1, 2]);
        dormant[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let dormant_node = push_symmetry(&mut dormant[1].1, RuntimeSymmetryParams::default());
        dormant[1].1.get_mut(dormant_node).unwrap().enabled = false;
        dormant[1].1.get_mut(dormant_node).unwrap().wet = 0.0;
        dormant[1]
            .1
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &dormant).unwrap());
        let steps = layer_plan(&compiled, 2).execution.steps();
        assert_eq!(steps.len(), 5);
        assert!(matches!(steps[3], EvaluatedScopeStep::SymmetryField { .. }));
        assert!(matches!(
            steps[4],
            EvaluatedScopeStep::CollisionRack {
                segment_index: 1,
                ..
            }
        ));
        assert!(
            symmetry_taps(&compiled, dormant_node).is_empty(),
            "a dormant node owns its position but collects no route"
        );
    }

    #[test]
    fn the_dedicated_pass_is_admitted_at_eight_and_refused_at_nine_against_the_raw_device_ceiling()
    {
        let plan_for = |textures| SymmetryFieldResourcePlan {
            max_sampled_textures_in_pass: textures,
            ..symmetry_field_resource_plan(1).unwrap()
        };
        let limits_for = |textures| CreativeResourceLimits {
            max_sampled_textures_per_shader_stage: textures,
            ..CreativeResourceLimits::default()
        };

        // The frozen declaration is eight, and eight is admitted at the device
        // floor of sixteen. This case cannot pass if the ordinary rack's
        // `.min(MAX_SAMPLED_TEXTURES_PER_PASS)` clamp is ever applied here,
        // because that clamp would reduce sixteen to three.
        assert_eq!(
            symmetry_field_resource_plan(1)
                .unwrap()
                .max_sampled_textures_in_pass,
            8
        );
        assert!(validate_symmetry_field_textures(plan_for(8), limits_for(16)).is_ok());
        assert!(validate_symmetry_field_textures(plan_for(8), limits_for(8)).is_ok());

        // Nine exceeds this project's dedicated-pass policy even on a device
        // that could serve it.
        assert!(matches!(
            validate_symmetry_field_textures(plan_for(9), limits_for(16)),
            Err(CompositionPlanError::Resource(
                ResourcePreflightError::SampledTextureLimit {
                    requested: 9,
                    limit: 8
                }
            ))
        ));

        // A device that reports fewer than eight refuses the pass by its own
        // number, not by the policy constant.
        assert!(matches!(
            validate_symmetry_field_textures(plan_for(8), limits_for(7)),
            Err(CompositionPlanError::Resource(
                ResourcePreflightError::SampledTextureLimit {
                    requested: 8,
                    limit: 7
                }
            ))
        ));

        // End to end: the default limits stub is the ordinary rack ceiling of
        // three, so an authored Symmetry Field is refused there and admitted
        // once the fixture reports the real device floor.
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        push_symmetry(
            &mut racks[1].1,
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None),
        );
        assert!(matches!(
            plan(&base, &composition, &master, &racks),
            Err(CompositionPlanError::Resource(
                ResourcePreflightError::SampledTextureLimit {
                    requested: 8,
                    limit: 3
                }
            ))
        ));
        assert!(plan_at_device_floor(&base, &composition, &master, &racks).is_ok());
    }

    #[test]
    fn ordinary_rack_segments_stay_capped_at_three_while_the_dedicated_pass_owns_eight() {
        // The two ceilings are separate constants and neither was raised to
        // admit the other.
        assert_eq!(crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_PASS, 3);
        assert_eq!(
            crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS,
            8
        );

        // The ordinary clamp survives verbatim at both of its sites in this
        // module, and the dedicated check contains no clamp at all.
        let source = include_str!("evaluated_composition.rs");
        let implementation = source
            .split_once("\nmod tests {")
            .expect("the test module follows the implementation")
            .0;
        assert_eq!(
            implementation
                .matches(".min(crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_PASS)")
                .count(),
            2,
            "the ordinary rack clamp must stay at exactly its two established sites"
        );
        // Both dedicated kinds (Symmetry, Study) delegate to the one shared
        // predicate, so the clamp law is asserted where it actually lives.
        let dedicated = implementation
            .split_once("fn validate_dedicated_sampled_textures(")
            .expect("the dedicated admission check exists")
            .1;
        let dedicated = &dedicated[..dedicated.find("\n}\n").expect("its body is bounded")];
        assert!(
            !dedicated.contains(".min("),
            "the dedicated ceiling must never be clamped by the ordinary rack ceiling"
        );
        assert!(dedicated.contains("MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS"));
        assert!(dedicated.contains("limits.max_sampled_textures_per_shader_stage"));

        // The two hardcoded LegacyExact matte threes are independent of both
        // ceilings and did not move with either.
        assert!(include_str!("renderer/compositor.rs")
            .contains("if limits.max_sampled_textures_per_shader_stage < 3 {"));
        assert!(include_str!("evaluated_frame.rs")
            .contains("max_sampled_textures_per_shader_stage: 3,"));

        // An ordinary rack alongside a dedicated pass still charges its own
        // widest pass to the three-texture accumulator.
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let mut racks = legacy_racks(&[1, 2]);
        push_displace(
            &mut racks[1].1,
            layer_source(1, 0),
            EdgeTiming::CurrentFrame,
            0.5,
        );
        push_symmetry(
            &mut racks[1].1,
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None),
        );
        let budget = racks[1].1.resource_budget().unwrap();
        assert_eq!(
            budget.max_sampled_textures_in_pass, 2,
            "Displace's two bindings, not the dedicated pass's eight"
        );
        assert_eq!(budget.max_sampled_textures_in_dedicated_pass, 8);
        assert!(plan_at_device_floor(&base, &composition, &master, &racks).is_ok());
    }

    #[test]
    fn symmetry_collects_one_tap_per_armed_image_slot_and_slot_index_is_route_identity() {
        let base = base(&[1, 2, 3], &[]);
        let composition = legacy_composition(&[1, 2, 3]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);

        // Both slots armed: two taps, one per slot, each resolving its own
        // donor. A first-match consumer lookup could not tell them apart.
        let mut racks = legacy_racks(&[1, 2, 3]);
        let node = push_symmetry(
            &mut racks[2].1,
            symmetry_params(
                Some((layer_source(1, 0), EdgeTiming::CurrentFrame)),
                Some((layer_source(2, 1), EdgeTiming::CurrentFrame)),
            ),
        );
        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        let taps = symmetry_taps(&compiled, node);
        assert_eq!(taps.len(), 2);
        assert_eq!(taps[0].0, 0);
        assert_eq!(
            taps[0].1.resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(1),
                stage: LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(taps[1].0, 1);
        assert_eq!(
            taps[1].1.resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(2),
                stage: LayerImageStage::PostLocalEffects,
            }
        );

        // Disarming slot 0 leaves slot 1 addressed as slot 1. The surviving
        // route must not slide down into the vacated slot.
        let mut racks = legacy_racks(&[1, 2, 3]);
        let mut params = symmetry_params(
            Some((layer_source(1, 0), EdgeTiming::CurrentFrame)),
            Some((layer_source(2, 1), EdgeTiming::CurrentFrame)),
        );
        params.source_mask.donor0 = false;
        let node = push_symmetry(&mut racks[2].1, params);
        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        let taps = symmetry_taps(&compiled, node);
        assert_eq!(taps.len(), 1);
        assert_eq!(taps[0].0, 1);
        assert_eq!(
            taps[0].1.resolved,
            PlannedImageSource::SelectedLayer {
                layer_id: layer_id(2),
                stage: LayerImageStage::PostLocalEffects,
            }
        );

        // An exact-default node is a real delegation: no tap at either slot.
        let mut racks = legacy_racks(&[1, 2, 3]);
        let node = push_symmetry(&mut racks[2].1, RuntimeSymmetryParams::default());
        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        assert!(symmetry_taps(&compiled, node).is_empty());
    }

    #[test]
    fn the_topology_signature_moves_when_the_same_route_moves_to_the_other_image_slot() {
        // The consumer key must carry the slot: two identical routes differing
        // only in which slot holds them are otherwise indistinguishable.
        let scope = VisualScopeId::Layer(layer_id(2));
        let node_id = NodeId::new(7).unwrap();
        assert_ne!(
            hash_consumer(
                FNV_OFFSET,
                ImageTapConsumer::RackNode {
                    scope,
                    node_id,
                    slot: 0
                }
            ),
            hash_consumer(
                FNV_OFFSET,
                ImageTapConsumer::RackNode {
                    scope,
                    node_id,
                    slot: 1
                }
            )
        );

        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let route = (layer_source(1, 0), EdgeTiming::CurrentFrame);
        let compile = |params| {
            let mut racks = legacy_racks(&[1, 2]);
            let node = push_symmetry(&mut racks[1].1, params);
            let compiled =
                advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
            (compiled, node)
        };

        let (at_slot_zero, first) = compile(symmetry_params(Some(route), None));
        let (at_slot_one, second) = compile(symmetry_params(None, Some(route)));
        let first = symmetry_taps(&at_slot_zero, first);
        let second = symmetry_taps(&at_slot_one, second);

        // Everything except the slot is identical: one tap, same origin, same
        // resolved source, same timing.
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(first[0].1.origin, second[0].1.origin);
        assert_eq!(first[0].1.resolved, second[0].1.resolved);
        assert_eq!(first[0].0, 0);
        assert_eq!(second[0].0, 1);
        assert_ne!(
            at_slot_zero.topology_signature(),
            at_slot_one.topology_signature(),
            "moving a route between slots must invalidate the prepared bindings"
        );
    }

    #[test]
    fn the_three_symmetry_admission_sites_agree_route_for_route() {
        // All three sites share one predicate and one per-slot helper. The
        // segmentation site is deliberately kind-only.
        let planner = include_str!("evaluated_composition.rs");
        let saved = include_str!("patch/mod.rs");
        for source in [planner, saved] {
            assert!(source.contains("Symmetry(symmetry) if !symmetry.is_exact_bypass()"));
            assert!(source.contains("symmetry.admitted_donor_taps()"));
        }

        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let route = (layer_source(1, 0), EdgeTiming::CurrentFrame);
        let cases = [
            (true, 1.0_f32, symmetry_params(Some(route), Some(route))),
            (true, 1.0, symmetry_params(Some(route), None)),
            (true, 1.0, symmetry_params(None, Some(route))),
            (true, 1.0, RuntimeSymmetryParams::default()),
            (false, 1.0, symmetry_params(Some(route), Some(route))),
            (true, 0.0, symmetry_params(Some(route), Some(route))),
        ];
        for (enabled, wet, params) in cases {
            let mut racks = legacy_racks(&[1, 2]);
            let node = push_symmetry(&mut racks[1].1, params);
            let live = racks[1].1.get_mut(node).unwrap();
            live.enabled = enabled;
            live.wet = wet;
            let compiled =
                advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());

            // The saved-patch walk's own predicate, evaluated on the captured
            // twin of the same node.
            let captured = params.capture_routes(&mut |_| Some(saved_position(0)));
            let expected: Vec<u8> = if enabled && wet > 0.0 && !captured.is_exact_bypass() {
                captured
                    .admitted_donor_taps()
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, tap)| tap.map(|_| slot as u8))
                    .collect()
            } else {
                Vec::new()
            };
            let collected: Vec<u8> = symmetry_taps(&compiled, node)
                .into_iter()
                .map(|(slot, _)| slot)
                .collect();
            assert_eq!(
                collected, expected,
                "the live planner and the saved walk must admit the same slots"
            );

            // The segmentation site never consults a value.
            assert_eq!(symmetry_steps(&layer_plan(&compiled, 2).execution).len(), 1);
        }
    }

    #[test]
    fn a_symmetry_motion_donor_yields_a_field_even_when_its_own_motion_is_exactly_zero() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = zero_motion_layers(&[1, 2]);
        assert!(
            layers.iter().all(|layer| layer.params.is_exact_zero()),
            "the donor's own Motion must be exactly zero for this proof to mean anything"
        );

        // Without the route the frame has no motion work at all, so the whole
        // composition stays on the literal pre-M4 exact path.
        let racks = legacy_racks(&[1, 2]);
        let quiet = plan_with_motion_at_device_floor(&base, &composition, &master, &racks, &layers)
            .unwrap();
        assert!(matches!(quiet, EvaluatedCompositionPlan::LegacyExact(_)));

        // Arming a Symmetry motion slot pulls the donor's primitive
        // vector/gate field into the plan through `required_as_donor`.
        let mut racks = legacy_racks(&[1, 2]);
        let mut params =
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None);
        params.motion_mask.slot0 = true;
        params.motion[0] = MotionDonor::Selected {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
        };
        let node = push_symmetry(&mut racks[1].1, params);
        let compiled = advanced(
            plan_with_motion_at_device_floor(&base, &composition, &master, &racks, &layers)
                .unwrap(),
        );
        let motion = compiled
            .motion()
            .advanced()
            .expect("an admitted field plan");
        assert_eq!(motion.fields().len(), 1);
        assert_eq!(motion.fields()[0].scope, VisualScopeId::Layer(layer_id(1)));
        assert!(motion.fields()[0].required_as_donor);
        assert_eq!(motion.resources().active_field_slots, 1);
        let donor = motion.scope(VisualScopeId::Layer(layer_id(1))).unwrap();
        assert!(donor.params.is_exact_zero());
        assert!(donor.required_as_donor);

        // The dedicated step observes the same admitted slot motion rendering
        // wrote, and the unarmed slot stays empty.
        let field = symmetry_steps(&layer_plan(&compiled, 2).execution);
        assert_eq!(field.len(), 1);
        assert_eq!(field[0].node_id, node);
        assert_eq!(
            field[0].motion_field_slots,
            [donor.admitted_field_slot(), None]
        );
        assert_eq!(field[0].motion_field_slots[0], Some(0));
    }

    #[test]
    fn an_incomplete_symmetry_motion_pair_binds_neutral_and_names_its_own_slot() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let layers = zero_motion_layers(&[1, 2]);

        let mut racks = legacy_racks(&[1, 2]);
        let tombstoned = saved_position(3);
        let mut params =
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None);
        params.motion_mask.slot0 = true;
        params.motion_mask.slot1 = true;
        // Slot 0 is a tombstone, slot 1 names a live donor.
        params.motion[0] = MotionDonor::Missing {
            saved_position: tombstoned,
        };
        params.motion[1] = MotionDonor::Selected {
            layer_id: layer_id(1),
            saved_position: saved_position(0),
        };
        let node = push_symmetry(&mut racks[1].1, params);
        let compiled = advanced(
            plan_with_motion_at_device_floor(&base, &composition, &master, &racks, &layers)
                .unwrap(),
        );
        let motion = compiled.motion().advanced().unwrap();
        assert!(
            motion.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                MotionPlanDiagnostic::MissingSymmetryMotion {
                    scope: VisualScopeId::Layer(owner),
                    node_id,
                    slot: 0,
                    saved_position,
                } if *owner == layer_id(2)
                    && *node_id == node
                    && *saved_position == tombstoned
            )),
            "a tombstoned motion slot must name itself: {:?}",
            motion.diagnostics()
        );
        let field = symmetry_steps(&layer_plan(&compiled, 2).execution);
        assert_eq!(field[0].motion_field_slots[0], None);
        assert_eq!(field[0].motion_field_slots[1], Some(0));

        // An armed slot with no donor at all is equally visible, and the
        // tombstone never rebinds onto the layer that now occupies its saved
        // position.
        let mut racks = legacy_racks(&[1, 2]);
        let mut params =
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None);
        params.motion_mask.slot1 = true;
        let node = push_symmetry(&mut racks[1].1, params);
        let compiled = advanced(
            plan_with_motion_at_device_floor(&base, &composition, &master, &racks, &layers)
                .unwrap(),
        );
        let motion = compiled.motion().advanced().unwrap();
        assert!(motion.fields().is_empty());
        assert!(motion.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            MotionPlanDiagnostic::SymmetryMotionNotSelected {
                node_id, slot: 1, ..
            } if *node_id == node
        )));
        let field = symmetry_steps(&layer_plan(&compiled, 2).execution);
        assert_eq!(field[0].motion_field_slots, [None, None]);
    }

    #[test]
    fn a_symmetry_slot_self_route_cycles_on_the_current_frame_but_n_minus_one_is_admitted() {
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let own_output = layer_source(2, 1);

        // Slot 1 alone reads its own scope this frame.
        let mut racks = legacy_racks(&[1, 2]);
        push_symmetry(
            &mut racks[1].1,
            symmetry_params(
                Some((layer_source(1, 0), EdgeTiming::CurrentFrame)),
                Some((own_output, EdgeTiming::CurrentFrame)),
            ),
        );
        let Err(error) = plan_at_device_floor(&base, &composition, &master, &racks) else {
            panic!("a current-frame self route must be rejected before allocation");
        };
        assert!(
            matches!(
                error,
                CompositionPlanError::ImageGraph(ImageGraphError::CurrentCycle { ref scopes })
                    if scopes.as_slice() == [VisualScopeId::Layer(layer_id(2))]
            ),
            "the offending scope must be named: {error:?}"
        );

        // The identical route at N-1 is a legitimate feedback edge, and slot 0
        // keeps its own current-frame edge alongside it.
        let mut racks = legacy_racks(&[1, 2]);
        push_symmetry(
            &mut racks[1].1,
            symmetry_params(
                Some((layer_source(1, 0), EdgeTiming::CurrentFrame)),
                Some((own_output, EdgeTiming::PreviousFrame)),
            ),
        );
        let compiled =
            advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        assert_eq!(compiled.graph().previous_taps, 1);
        assert_eq!(compiled.graph().current_taps, 1);

        // Disarming the offending slot removes the edge entirely: an
        // unreadable route cannot cycle.
        let mut racks = legacy_racks(&[1, 2]);
        let mut params = symmetry_params(
            Some((layer_source(1, 0), EdgeTiming::CurrentFrame)),
            Some((own_output, EdgeTiming::CurrentFrame)),
        );
        params.source_mask.donor1 = false;
        push_symmetry(&mut racks[1].1, params);
        assert!(plan_at_device_floor(&base, &composition, &master, &racks).is_ok());
    }

    #[test]
    fn a_missing_symmetry_donor_tombstones_only_its_own_slot_and_never_rebinds() {
        let frame = base(&[1, 2], &[]);
        let replaced_frame = base(&[3, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let saved = saved_position(0);

        let mut racks = legacy_racks(&[1, 2]);
        let node = push_symmetry(
            &mut racks[1].1,
            symmetry_params(
                Some((layer_source(2, 1), EdgeTiming::PreviousFrame)),
                Some((layer_source(1, 0), EdgeTiming::CurrentFrame)),
            ),
        );
        let compiled =
            advanced(plan_at_device_floor(&frame, &composition, &master, &racks).unwrap());
        assert!(matches!(
            symmetry_taps(&compiled, node)[1].1.resolved,
            PlannedImageSource::SelectedLayer { .. }
        ));

        // Deleting slot 1's donor tombstones that slot only.
        racks[1].1.mark_layer_output_missing(layer_id(1));
        let replaced_composition = legacy_composition(&[3, 2]);
        let mut replaced_racks = legacy_racks(&[3, 2]);
        let slot = replaced_racks
            .iter_mut()
            .find(|(id, _)| *id == layer_id(2))
            .expect("layer 2 survives the replacement");
        slot.1 = racks[1].1.clone();
        let compiled = advanced(
            plan_at_device_floor(
                &replaced_frame,
                &replaced_composition,
                &master,
                &replaced_racks,
            )
            .unwrap(),
        );
        let taps = symmetry_taps(&compiled, node);
        assert_eq!(taps.len(), 2);
        assert!(
            matches!(
                taps[0].1.resolved,
                PlannedImageSource::SelectedLayer {
                    layer_id: id,
                    ..
                } if id == layer_id(2)
            ),
            "slot 0 is untouched by slot 1's loss"
        );
        assert_eq!(taps[1].1.resolved, PlannedImageSource::Transparent);
        assert!(compiled.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            CompositionPlanDiagnostic::MissingSelectedLayer {
                consumer: ImageTapConsumer::RackNode {
                    node_id,
                    slot: 1,
                    ..
                },
                saved_position,
            } if *node_id == node && *saved_position == saved
        )));
    }

    #[test]
    fn the_dedicated_symmetry_ledger_is_charged_exactly_once_beside_the_rack_ledger() {
        let descriptor = crate::visual_rack::node_kind_descriptor(NodeKindTag::Symmetry).budget;
        assert_eq!(
            symmetry_field_resource_plan(0).unwrap(),
            SymmetryFieldResourcePlan::default()
        );
        assert_eq!(
            symmetry_field_resource_plan(1).unwrap(),
            SymmetryFieldResourcePlan {
                full_frame_passes: 1,
                logical_texture_lookups_per_pixel: 4,
                texture_operations_per_pixel: 10,
                max_sampled_textures_in_pass: 8,
                uniform_bytes: 1_024,
            }
        );
        // Two dedicated passes accumulate work but not simultaneous bindings:
        // each pass owns its own bind group.
        assert_eq!(
            symmetry_field_resource_plan(2).unwrap(),
            SymmetryFieldResourcePlan {
                full_frame_passes: 2,
                logical_texture_lookups_per_pixel: 8,
                texture_operations_per_pixel: 20,
                max_sampled_textures_in_pass: 8,
                uniform_bytes: 2_048,
            }
        );
        assert_eq!(
            u32::from(descriptor.logical_texture_lookups_per_pixel),
            symmetry_field_resource_plan(1)
                .unwrap()
                .logical_texture_lookups_per_pixel,
            "the step table is read from the frozen descriptor, never restated"
        );

        // The authored node is charged once, through the ordinary rack ledger.
        // The step table beside it must not add a second charge. The baseline
        // carries an ordinary custom node so both plans are Advanced and differ
        // only by the dedicated pass.
        let base = base(&[1, 2], &[]);
        let composition = legacy_composition(&[1, 2]);
        let master = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Master);
        let baseline_rack = |racks: &mut Vec<(StableLayerId, RuntimeVisualRack)>| {
            racks[1]
                .1
                .push(RuntimeVisualNodeKind::DigitalColor(
                    DigitalColorParams::default(),
                ))
                .unwrap();
        };
        let mut racks = legacy_racks(&[1, 2]);
        baseline_rack(&mut racks);
        let without = advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        assert_eq!(
            without.symmetry_field_resources(),
            SymmetryFieldResourcePlan::default()
        );

        let mut racks = legacy_racks(&[1, 2]);
        baseline_rack(&mut racks);
        push_symmetry(
            &mut racks[1].1,
            symmetry_params(Some((layer_source(1, 0), EdgeTiming::CurrentFrame)), None),
        );
        let with = advanced(plan_at_device_floor(&base, &composition, &master, &racks).unwrap());
        assert_eq!(with.symmetry_field_resources(), {
            symmetry_field_resource_plan(1).unwrap()
        });
        assert_eq!(
            with.resources().logical_texture_lookups_per_pixel
                - without.resources().logical_texture_lookups_per_pixel,
            u32::from(descriptor.logical_texture_lookups_per_pixel),
        );
        assert_eq!(
            with.resources().texture_samples_per_pixel
                - without.resources().texture_samples_per_pixel,
            u32::from(descriptor.texture_samples_per_pixel),
        );
        assert_eq!(
            with.resources().full_frame_passes - without.resources().full_frame_passes,
            u32::from(descriptor.full_frame_passes),
        );
        // The Compat8 clean-history ring is reused, never rebuilt: a dedicated
        // pass adds no history layer.
        assert_eq!(
            with.resources().compat8_surface_layers,
            without.resources().compat8_surface_layers
        );
        assert_ne!(with.topology_signature(), without.topology_signature());
    }
}
