//! Executable capability truth.
//!
//! The registry is deliberately data-only and path-free. Runtime code supplies
//! evidence to the same evaluator used by the registry; documentation renders
//! deterministic Windows/macOS/Linux snapshots from the frozen current-tree
//! facts rather than inventing a second status vocabulary.

use serde::{Deserialize, Serialize};

pub const CAPABILITY_REGISTRY_SCHEMA: &str = "collide-o-scope-capability-registry/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    Windows,
    Macos,
    Linux,
}

impl Platform {
    pub const ALL: [Self; 3] = [Self::Windows, Self::Macos, Self::Linux];

    pub const fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::Macos
        } else {
            Self::Linux
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Windows => "Windows",
            Self::Macos => "macOS",
            Self::Linux => "Linux",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKey {
    AcceptedCreativeMutationV1,
    AdvancedPrecision,
    BoundedMeshWarp,
    CaptureInput,
    CorrelatedEngineGpuTiming,
    D3d11vaHardwareDecode,
    ExactVfrLiveTransport,
    FinalProgramVhs,
    Full16TemporalHistory,
    LiveRecorderAudioMux,
    NdiInput,
    NdiOutput,
    ProxyBrowserSurface,
    SourceDescriptorColorTruth,
    SpoutInput,
    SpoutOutput,
    #[serde(rename = "study_motion_abi_1_1")]
    StudyMotionAbi11,
    SupervisedGpuRecoveryPhaseA,
    SyphonInput,
    SyphonOutput,
    TransactionalControlListeners,
    ZeroCopyDecode,
}

impl CapabilityKey {
    /// Stable key order is also the generated-document order.
    pub const ALL: [Self; 22] = [
        Self::AcceptedCreativeMutationV1,
        Self::AdvancedPrecision,
        Self::BoundedMeshWarp,
        Self::CaptureInput,
        Self::CorrelatedEngineGpuTiming,
        Self::D3d11vaHardwareDecode,
        Self::ExactVfrLiveTransport,
        Self::FinalProgramVhs,
        Self::Full16TemporalHistory,
        Self::LiveRecorderAudioMux,
        Self::NdiInput,
        Self::NdiOutput,
        Self::ProxyBrowserSurface,
        Self::SourceDescriptorColorTruth,
        Self::SpoutInput,
        Self::SpoutOutput,
        Self::StudyMotionAbi11,
        Self::SupervisedGpuRecoveryPhaseA,
        Self::SyphonInput,
        Self::SyphonOutput,
        Self::TransactionalControlListeners,
        Self::ZeroCopyDecode,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AcceptedCreativeMutationV1 => "accepted_creative_mutation_v1",
            Self::AdvancedPrecision => "advanced_precision",
            Self::BoundedMeshWarp => "bounded_mesh_warp",
            Self::CaptureInput => "capture_input",
            Self::CorrelatedEngineGpuTiming => "correlated_engine_gpu_timing",
            Self::D3d11vaHardwareDecode => "d3d11va_hardware_decode",
            Self::ExactVfrLiveTransport => "exact_vfr_live_transport",
            Self::FinalProgramVhs => "final_program_vhs",
            Self::Full16TemporalHistory => "full16_temporal_history",
            Self::LiveRecorderAudioMux => "live_recorder_audio_mux",
            Self::NdiInput => "ndi_input",
            Self::NdiOutput => "ndi_output",
            Self::ProxyBrowserSurface => "proxy_browser_surface",
            Self::SourceDescriptorColorTruth => "source_descriptor_color_truth",
            Self::SpoutInput => "spout_input",
            Self::SpoutOutput => "spout_output",
            Self::StudyMotionAbi11 => "study_motion_abi_1_1",
            Self::SupervisedGpuRecoveryPhaseA => "supervised_gpu_recovery_phase_a",
            Self::SyphonInput => "syphon_input",
            Self::SyphonOutput => "syphon_output",
            Self::TransactionalControlListeners => "transactional_control_listeners",
            Self::ZeroCopyDecode => "zero_copy_decode",
        }
    }

    pub const fn title(self) -> &'static str {
        match self {
            Self::AcceptedCreativeMutationV1 => "Accepted creative-mutation recording v1",
            Self::AdvancedPrecision => "Advanced precision path",
            Self::BoundedMeshWarp => "Bounded mesh warp",
            Self::CaptureInput => "External capture input",
            Self::CorrelatedEngineGpuTiming => "Correlated engine and GPU-stage timing",
            Self::D3d11vaHardwareDecode => "D3D11VA hardware decode",
            Self::ExactVfrLiveTransport => "Exact PTS-driven VFR live transport",
            Self::FinalProgramVhs => "Final-program VHS",
            Self::Full16TemporalHistory => "Full-16 temporal history",
            Self::LiveRecorderAudioMux => "Live recorder audio mux",
            Self::NdiInput => "NDI input",
            Self::NdiOutput => "NDI output",
            Self::ProxyBrowserSurface => "Browser proxy control",
            Self::SourceDescriptorColorTruth => "Source descriptor and color-truth diagnostics",
            Self::SpoutInput => "Spout input",
            Self::SpoutOutput => "Spout output",
            Self::StudyMotionAbi11 => "Study motion ABI 1.1",
            Self::SupervisedGpuRecoveryPhaseA => "Supervised GPU recovery (Phase A)",
            Self::SyphonInput => "Syphon input",
            Self::SyphonOutput => "Syphon output",
            Self::TransactionalControlListeners => {
                "Transactional control listeners and TLS identity"
            }
            Self::ZeroCopyDecode => "Zero-copy decode",
        }
    }
}

const EXTERNAL_DEFERRED_EVIDENCE_BOUNDARY_RECEIPT: &str =
    "p10-external-deferred-capability-evidence-boundary";

#[cfg(test)]
const EXTERNAL_DEFERRED_EVIDENCE_KEYS: [CapabilityKey; 9] = [
    CapabilityKey::BoundedMeshWarp,
    CapabilityKey::CaptureInput,
    CapabilityKey::NdiInput,
    CapabilityKey::NdiOutput,
    CapabilityKey::SpoutInput,
    CapabilityKey::SpoutOutput,
    CapabilityKey::SyphonInput,
    CapabilityKey::SyphonOutput,
    CapabilityKey::ZeroCopyDecode,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatus {
    Implemented,
    EvaluationRequired,
    Deferred,
    RejectedByMeasurement,
    UnavailableOnPlatform,
}

impl CapabilityStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Implemented => "Implemented",
            Self::EvaluationRequired => "Evaluation required",
            Self::Deferred => "Deferred",
            Self::RejectedByMeasurement => "Rejected by measurement",
            Self::UnavailableOnPlatform => "Unavailable on platform",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityDeferredReason {
    BackendNotIntegrated,
    SdkOrLicenseRequired,
    NetworkPolicyRequired,
    VenueRequirementNotEstablished,
    OwnedProgramPcmUnavailable,
    ExactPtsTimelineNotIntegrated,
    LiveSourceUnavailableOffline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvaluationRequirement {
    InteroperabilityProof,
    PhysicalDeviceValidation,
    RendererIntegrationProof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityRejectionReason {
    MeasuredResourceTradeoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityPlatformRequirement {
    Windows,
    Macos,
    BackendSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum CapabilityDecision {
    Implemented,
    EvaluationRequired(CapabilityEvaluationRequirement),
    Deferred(CapabilityDeferredReason),
    RejectedByMeasurement(CapabilityRejectionReason),
    UnavailableOnPlatform(CapabilityPlatformRequirement),
}

impl CapabilityDecision {
    pub const fn status(self) -> CapabilityStatus {
        match self {
            Self::Implemented => CapabilityStatus::Implemented,
            Self::EvaluationRequired(_) => CapabilityStatus::EvaluationRequired,
            Self::Deferred(_) => CapabilityStatus::Deferred,
            Self::RejectedByMeasurement(_) => CapabilityStatus::RejectedByMeasurement,
            Self::UnavailableOnPlatform(_) => CapabilityStatus::UnavailableOnPlatform,
        }
    }

    pub const fn reason(self) -> CapabilityReason {
        match self {
            Self::Implemented => CapabilityReason::ProductionPathIntegrated,
            Self::EvaluationRequired(requirement) => CapabilityReason::Evaluation(requirement),
            Self::Deferred(reason) => CapabilityReason::Deferred(reason),
            Self::RejectedByMeasurement(reason) => CapabilityReason::Rejected(reason),
            Self::UnavailableOnPlatform(requirement) => CapabilityReason::Unavailable(requirement),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum CapabilityReason {
    ProductionPathIntegrated,
    Evaluation(CapabilityEvaluationRequirement),
    Deferred(CapabilityDeferredReason),
    Rejected(CapabilityRejectionReason),
    Unavailable(CapabilityPlatformRequirement),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleCapability {
    HardwareDecode,
    ZeroCopyDecode,
    SyphonInput,
    SyphonOutput,
    NdiInput,
    NdiOutput,
    CaptureInput,
    BoundedMeshWarp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CapabilityEvidence {
    pub platform_supported: bool,
    pub backend_integrated: bool,
    pub interoperability_proven: bool,
    pub sdk_license_authorized: bool,
    pub network_policy_authorized: bool,
    pub venue_requirement_proven: bool,
}

/// Shared evidence evaluator. Production probes and the generated registry use
/// this function; documentation is never consulted as a predicate.
pub const fn evaluate_scale_capability(
    capability: ScaleCapability,
    evidence: CapabilityEvidence,
) -> CapabilityDecision {
    if matches!(
        capability,
        ScaleCapability::NdiInput | ScaleCapability::NdiOutput
    ) {
        if !evidence.sdk_license_authorized {
            return CapabilityDecision::Deferred(CapabilityDeferredReason::SdkOrLicenseRequired);
        }
        if !evidence.network_policy_authorized {
            return CapabilityDecision::Deferred(CapabilityDeferredReason::NetworkPolicyRequired);
        }
    }
    if matches!(capability, ScaleCapability::BoundedMeshWarp) && !evidence.venue_requirement_proven
    {
        return CapabilityDecision::Deferred(
            CapabilityDeferredReason::VenueRequirementNotEstablished,
        );
    }
    if !evidence.platform_supported {
        let platform = match capability {
            ScaleCapability::SyphonInput | ScaleCapability::SyphonOutput => {
                CapabilityPlatformRequirement::Macos
            }
            _ => CapabilityPlatformRequirement::BackendSpecific,
        };
        return CapabilityDecision::UnavailableOnPlatform(platform);
    }
    if !evidence.backend_integrated {
        return CapabilityDecision::Deferred(CapabilityDeferredReason::BackendNotIntegrated);
    }
    if !evidence.interoperability_proven {
        return CapabilityDecision::EvaluationRequired(
            CapabilityEvaluationRequirement::InteroperabilityProof,
        );
    }
    CapabilityDecision::Implemented
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySurface {
    BrowserControl,
    NativeControl,
    LiveProgram,
    LiveRecording,
    OfflineExport,
    Backend,
    HardwareInteroperability,
    PhysicalVenue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySurfaceRecord {
    pub surface: CapabilitySurface,
    pub status: CapabilityStatus,
    pub reason: CapabilityReason,
}

impl CapabilitySurfaceRecord {
    fn new(surface: CapabilitySurface, decision: CapabilityDecision) -> Self {
        Self {
            surface,
            status: decision.status(),
            reason: decision.reason(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EvidenceReceiptId(pub String);

impl EvidenceReceiptId {
    fn new(value: &'static str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLimitation {
    AcceptedMutationVocabularyV1Only,
    AdditiveAbiVersionPinned,
    VerifiedContentIdentityRequired,
    OriginalMediaUsedForExport,
    AsynchronousFinalProgramLatency,
    VideoOnlyRecorder,
    LiveAudioCaptureRequired,
    AverageFrameRateCadence,
    WindowsOnly,
    LiveSourceIsBlackOffline,
    EvaluationFixtureOnly,
    ExternalProofRequired,
    LicenseAndNetworkAuthorizationRequired,
    VenueRequirementRequired,
    EngineSubmissionIsNotPhotonTime,
    PhysicalTimingAndPerformanceProofRequired,
    OptionalGpuTimestamps,
    DisplayGeometryIntegrationStopped,
    TransparentGpuContinuityUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityLimitationRecord {
    pub code: CapabilityLimitation,
    pub text: String,
}

impl CapabilityLimitationRecord {
    fn new(code: CapabilityLimitation, text: &'static str) -> Self {
        Self {
            code,
            text: text.to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRecord {
    pub key: CapabilityKey,
    pub title: String,
    pub platform: Platform,
    pub status: CapabilityStatus,
    pub reason: CapabilityReason,
    pub surfaces: Vec<CapabilitySurfaceRecord>,
    pub evidence_receipt_ids: Vec<EvidenceReceiptId>,
    pub known_limitations: Vec<CapabilityLimitationRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRuntimeFacts {
    pub platform: Platform,
    /// The production server owns independently published loopback-v4,
    /// loopback-v6 and LAN-TLS listener lifecycles plus one transactional TLS
    /// identity generation.
    pub transactional_control_listeners_integrated: bool,
    /// Engine action correlation and optional asynchronous GPU timestamp
    /// harvesting are connected to the bounded Stage Health publication.
    pub correlated_engine_gpu_timing_integrated: bool,
    /// Frozen source descriptors/policy reach decoder telemetry, native Stage
    /// Health and export provenance. This fact does not claim display-geometry
    /// application, which remains a separate evaluation surface.
    pub source_descriptor_color_truth_integrated: bool,
    /// Device loss reaches the supervised exit/relaunch contract. This fact
    /// intentionally says nothing about unavailable in-process Phase B.
    pub supervised_gpu_recovery_phase_a_integrated: bool,
    pub hardware_decode: CapabilityEvidence,
    pub zero_copy_decode: CapabilityEvidence,
    pub syphon_input: CapabilityEvidence,
    pub syphon_output: CapabilityEvidence,
    pub ndi_input: CapabilityEvidence,
    pub ndi_output: CapabilityEvidence,
    pub capture_input: CapabilityEvidence,
    pub bounded_mesh_warp: CapabilityEvidence,
}

impl CapabilityRuntimeFacts {
    /// Frozen current-tree facts used for cross-platform documentation. A live
    /// caller may replace any evidence record with its production probe.
    pub const fn current_tree(platform: Platform) -> Self {
        let windows = matches!(platform, Platform::Windows);
        let macos = matches!(platform, Platform::Macos);
        Self {
            platform,
            transactional_control_listeners_integrated: true,
            correlated_engine_gpu_timing_integrated: true,
            source_descriptor_color_truth_integrated: true,
            supervised_gpu_recovery_phase_a_integrated: true,
            hardware_decode: CapabilityEvidence {
                platform_supported: true,
                backend_integrated: windows,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            zero_copy_decode: CapabilityEvidence {
                platform_supported: true,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            syphon_input: CapabilityEvidence {
                platform_supported: macos,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            syphon_output: CapabilityEvidence {
                platform_supported: macos,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            ndi_input: CapabilityEvidence {
                platform_supported: true,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            ndi_output: CapabilityEvidence {
                platform_supported: true,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            capture_input: CapabilityEvidence {
                platform_supported: true,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
            bounded_mesh_warp: CapabilityEvidence {
                platform_supported: true,
                backend_integrated: false,
                interoperability_proven: false,
                sdk_license_authorized: false,
                network_policy_authorized: false,
                venue_requirement_proven: false,
            },
        }
    }
}

fn scale_decision(key: CapabilityKey, facts: CapabilityRuntimeFacts) -> CapabilityDecision {
    let (capability, evidence) = match key {
        CapabilityKey::BoundedMeshWarp => {
            (ScaleCapability::BoundedMeshWarp, facts.bounded_mesh_warp)
        }
        CapabilityKey::CaptureInput => (ScaleCapability::CaptureInput, facts.capture_input),
        CapabilityKey::D3d11vaHardwareDecode => {
            if !matches!(facts.platform, Platform::Windows) {
                return CapabilityDecision::UnavailableOnPlatform(
                    CapabilityPlatformRequirement::Windows,
                );
            }
            (ScaleCapability::HardwareDecode, facts.hardware_decode)
        }
        CapabilityKey::NdiInput => (ScaleCapability::NdiInput, facts.ndi_input),
        CapabilityKey::NdiOutput => (ScaleCapability::NdiOutput, facts.ndi_output),
        CapabilityKey::SyphonInput => (ScaleCapability::SyphonInput, facts.syphon_input),
        CapabilityKey::SyphonOutput => (ScaleCapability::SyphonOutput, facts.syphon_output),
        CapabilityKey::ZeroCopyDecode => (ScaleCapability::ZeroCopyDecode, facts.zero_copy_decode),
        _ => unreachable!("non-scale capability passed to scale_decision"),
    };
    evaluate_scale_capability(capability, evidence)
}

fn primary_decision(key: CapabilityKey, facts: CapabilityRuntimeFacts) -> CapabilityDecision {
    let integrated = |available| {
        if available {
            CapabilityDecision::Implemented
        } else {
            CapabilityDecision::Deferred(CapabilityDeferredReason::BackendNotIntegrated)
        }
    };
    match key {
        CapabilityKey::AcceptedCreativeMutationV1
        | CapabilityKey::AdvancedPrecision
        | CapabilityKey::FinalProgramVhs
        | CapabilityKey::ProxyBrowserSurface
        | CapabilityKey::StudyMotionAbi11 => CapabilityDecision::Implemented,
        CapabilityKey::TransactionalControlListeners => {
            integrated(facts.transactional_control_listeners_integrated)
        }
        CapabilityKey::CorrelatedEngineGpuTiming => {
            integrated(facts.correlated_engine_gpu_timing_integrated)
        }
        CapabilityKey::SourceDescriptorColorTruth => {
            integrated(facts.source_descriptor_color_truth_integrated)
        }
        CapabilityKey::SupervisedGpuRecoveryPhaseA => {
            integrated(facts.supervised_gpu_recovery_phase_a_integrated)
        }
        CapabilityKey::ExactVfrLiveTransport => {
            CapabilityDecision::Deferred(CapabilityDeferredReason::ExactPtsTimelineNotIntegrated)
        }
        CapabilityKey::Full16TemporalHistory => CapabilityDecision::RejectedByMeasurement(
            CapabilityRejectionReason::MeasuredResourceTradeoff,
        ),
        CapabilityKey::LiveRecorderAudioMux => CapabilityDecision::Implemented,
        CapabilityKey::SpoutInput | CapabilityKey::SpoutOutput => {
            if matches!(facts.platform, Platform::Windows) {
                CapabilityDecision::Implemented
            } else {
                CapabilityDecision::UnavailableOnPlatform(CapabilityPlatformRequirement::Windows)
            }
        }
        CapabilityKey::BoundedMeshWarp
        | CapabilityKey::CaptureInput
        | CapabilityKey::D3d11vaHardwareDecode
        | CapabilityKey::NdiInput
        | CapabilityKey::NdiOutput
        | CapabilityKey::SyphonInput
        | CapabilityKey::SyphonOutput
        | CapabilityKey::ZeroCopyDecode => scale_decision(key, facts),
    }
}

fn surfaces(key: CapabilityKey, primary: CapabilityDecision) -> Vec<CapabilitySurfaceRecord> {
    use CapabilitySurface as Surface;
    let implemented = CapabilityDecision::Implemented;
    let physical = CapabilityDecision::EvaluationRequired(
        CapabilityEvaluationRequirement::PhysicalDeviceValidation,
    );
    let physical_after_primary = if matches!(primary, CapabilityDecision::Implemented) {
        physical
    } else {
        primary
    };
    let rejected = CapabilityDecision::RejectedByMeasurement(
        CapabilityRejectionReason::MeasuredResourceTradeoff,
    );
    match key {
        CapabilityKey::AcceptedCreativeMutationV1 => vec![
            CapabilitySurfaceRecord::new(Surface::BrowserControl, implemented),
            CapabilitySurfaceRecord::new(Surface::NativeControl, implemented),
            CapabilitySurfaceRecord::new(Surface::LiveRecording, implemented),
        ],
        CapabilityKey::AdvancedPrecision => vec![
            CapabilitySurfaceRecord::new(Surface::LiveProgram, implemented),
            CapabilitySurfaceRecord::new(Surface::OfflineExport, implemented),
            CapabilitySurfaceRecord::new(Surface::HardwareInteroperability, physical),
        ],
        CapabilityKey::TransactionalControlListeners => vec![
            CapabilitySurfaceRecord::new(Surface::BrowserControl, primary),
            CapabilitySurfaceRecord::new(Surface::NativeControl, primary),
            CapabilitySurfaceRecord::new(Surface::Backend, primary),
            CapabilitySurfaceRecord::new(Surface::PhysicalVenue, physical_after_primary),
        ],
        CapabilityKey::CorrelatedEngineGpuTiming => vec![
            CapabilitySurfaceRecord::new(Surface::BrowserControl, primary),
            CapabilitySurfaceRecord::new(Surface::NativeControl, primary),
            CapabilitySurfaceRecord::new(Surface::Backend, primary),
            CapabilitySurfaceRecord::new(
                Surface::HardwareInteroperability,
                if matches!(primary, CapabilityDecision::Implemented) {
                    CapabilityDecision::EvaluationRequired(
                        CapabilityEvaluationRequirement::InteroperabilityProof,
                    )
                } else {
                    primary
                },
            ),
            CapabilitySurfaceRecord::new(Surface::PhysicalVenue, physical_after_primary),
        ],
        CapabilityKey::SourceDescriptorColorTruth => {
            let renderer_evaluation = if matches!(primary, CapabilityDecision::Implemented) {
                CapabilityDecision::EvaluationRequired(
                    CapabilityEvaluationRequirement::RendererIntegrationProof,
                )
            } else {
                primary
            };
            vec![
                CapabilitySurfaceRecord::new(Surface::NativeControl, primary),
                CapabilitySurfaceRecord::new(Surface::Backend, primary),
                CapabilitySurfaceRecord::new(Surface::LiveProgram, renderer_evaluation),
                CapabilitySurfaceRecord::new(Surface::OfflineExport, renderer_evaluation),
            ]
        }
        CapabilityKey::SupervisedGpuRecoveryPhaseA => vec![
            CapabilitySurfaceRecord::new(Surface::NativeControl, primary),
            CapabilitySurfaceRecord::new(Surface::Backend, primary),
            CapabilitySurfaceRecord::new(Surface::PhysicalVenue, physical_after_primary),
        ],
        CapabilityKey::ProxyBrowserSurface => vec![
            CapabilitySurfaceRecord::new(Surface::BrowserControl, implemented),
            CapabilitySurfaceRecord::new(Surface::NativeControl, implemented),
            CapabilitySurfaceRecord::new(Surface::LiveProgram, implemented),
        ],
        CapabilityKey::FinalProgramVhs => vec![
            CapabilitySurfaceRecord::new(Surface::BrowserControl, implemented),
            CapabilitySurfaceRecord::new(Surface::NativeControl, implemented),
            CapabilitySurfaceRecord::new(Surface::LiveProgram, implemented),
            CapabilitySurfaceRecord::new(Surface::OfflineExport, implemented),
        ],
        CapabilityKey::StudyMotionAbi11 => vec![
            CapabilitySurfaceRecord::new(Surface::LiveProgram, implemented),
            CapabilitySurfaceRecord::new(Surface::OfflineExport, implemented),
        ],
        CapabilityKey::LiveRecorderAudioMux => vec![CapabilitySurfaceRecord::new(
            Surface::LiveRecording,
            primary,
        )],
        CapabilityKey::ExactVfrLiveTransport => {
            vec![CapabilitySurfaceRecord::new(Surface::LiveProgram, primary)]
        }
        CapabilityKey::Full16TemporalHistory => vec![
            CapabilitySurfaceRecord::new(Surface::Backend, rejected),
            CapabilitySurfaceRecord::new(Surface::LiveProgram, rejected),
            CapabilitySurfaceRecord::new(Surface::OfflineExport, rejected),
        ],
        CapabilityKey::SpoutInput => {
            let offline = if matches!(primary, CapabilityDecision::Implemented) {
                CapabilityDecision::Deferred(CapabilityDeferredReason::LiveSourceUnavailableOffline)
            } else {
                primary
            };
            vec![
                CapabilitySurfaceRecord::new(Surface::BrowserControl, primary),
                CapabilitySurfaceRecord::new(Surface::NativeControl, primary),
                CapabilitySurfaceRecord::new(Surface::LiveProgram, primary),
                CapabilitySurfaceRecord::new(Surface::OfflineExport, offline),
                CapabilitySurfaceRecord::new(Surface::PhysicalVenue, physical_after_primary),
            ]
        }
        CapabilityKey::SpoutOutput => vec![
            CapabilitySurfaceRecord::new(Surface::BrowserControl, primary),
            CapabilitySurfaceRecord::new(Surface::NativeControl, primary),
            CapabilitySurfaceRecord::new(Surface::LiveProgram, primary),
            CapabilitySurfaceRecord::new(Surface::PhysicalVenue, physical_after_primary),
        ],
        CapabilityKey::D3d11vaHardwareDecode => vec![
            CapabilitySurfaceRecord::new(Surface::Backend, primary),
            CapabilitySurfaceRecord::new(Surface::HardwareInteroperability, primary),
            CapabilitySurfaceRecord::new(Surface::LiveProgram, {
                if matches!(primary, CapabilityDecision::UnavailableOnPlatform(_)) {
                    primary
                } else {
                    CapabilityDecision::Deferred(CapabilityDeferredReason::BackendNotIntegrated)
                }
            }),
        ],
        CapabilityKey::BoundedMeshWarp
        | CapabilityKey::CaptureInput
        | CapabilityKey::NdiInput
        | CapabilityKey::NdiOutput
        | CapabilityKey::SyphonInput
        | CapabilityKey::SyphonOutput
        | CapabilityKey::ZeroCopyDecode => vec![
            CapabilitySurfaceRecord::new(Surface::Backend, primary),
            CapabilitySurfaceRecord::new(Surface::HardwareInteroperability, primary),
            CapabilitySurfaceRecord::new(Surface::PhysicalVenue, physical_after_primary),
        ],
    }
}

fn evidence(key: CapabilityKey) -> Vec<EvidenceReceiptId> {
    let ids: &[&str] = match key {
        CapabilityKey::AcceptedCreativeMutationV1 => &["d4-accepted-creative-mutation"],
        CapabilityKey::AdvancedPrecision => &["m6-precision-gpu-receipt"],
        CapabilityKey::D3d11vaHardwareDecode => {
            &["s12d-hw-decode-backend-note", "hw-decode-interop-receipt"]
        }
        CapabilityKey::ExactVfrLiveTransport => &["v1.6.0-decoder-delivery-receipt"],
        CapabilityKey::FinalProgramVhs => &["b5-codec-mosh-note"],
        CapabilityKey::Full16TemporalHistory => &[
            "s12c-full16-history-candidate-note",
            "full16-history-candidate-receipt",
        ],
        CapabilityKey::LiveRecorderAudioMux => &["live-recorder-audio-mux-note"],
        CapabilityKey::ProxyBrowserSurface => &["s8c-browser-proxy-surface-note"],
        CapabilityKey::TransactionalControlListeners => &["p10-capability-campaign-truth-closure"],
        CapabilityKey::CorrelatedEngineGpuTiming => &[
            "p10-capability-campaign-truth-closure",
            "p1-action-to-photon-fixture-protocol",
            "p1-action-to-photon-fixture-unexecuted",
        ],
        CapabilityKey::SourceDescriptorColorTruth => &["p4b-source-descriptor-stop-receipt"],
        CapabilityKey::SupervisedGpuRecoveryPhaseA => &["p7-gpu-loss-phase-a"],
        CapabilityKey::StudyMotionAbi11 => &["d1-study-motion-abi-1.1"],
        CapabilityKey::BoundedMeshWarp
        | CapabilityKey::CaptureInput
        | CapabilityKey::NdiInput
        | CapabilityKey::NdiOutput
        | CapabilityKey::SpoutInput
        | CapabilityKey::SpoutOutput
        | CapabilityKey::SyphonInput
        | CapabilityKey::SyphonOutput
        | CapabilityKey::ZeroCopyDecode => &[EXTERNAL_DEFERRED_EVIDENCE_BOUNDARY_RECEIPT],
    };
    ids.iter().copied().map(EvidenceReceiptId::new).collect()
}

fn limitations(key: CapabilityKey) -> Vec<CapabilityLimitationRecord> {
    use CapabilityLimitation as Limitation;
    match key {
        CapabilityKey::AcceptedCreativeMutationV1 => vec![CapabilityLimitationRecord::new(
            Limitation::AcceptedMutationVocabularyV1Only,
            "Only the frozen v1 scalar/color/enum creative vocabulary is recordable; topology, routes, safety/transport, and replay remain excluded.",
        )],
        CapabilityKey::TransactionalControlListeners => vec![
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "The in-process listener, identity, and fault fixtures pass; second-host reachability and packet-capture proof remain environment-dependent.",
            ),
        ],
        CapabilityKey::CorrelatedEngineGpuTiming => vec![
            CapabilityLimitationRecord::new(
                Limitation::EngineSubmissionIsNotPhotonTime,
                "Ingress-to-apply, apply-to-submit, queue submission, swapchain presentation, and optical emission are distinct domains; engine submission is never photon time.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::PhysicalTimingAndPerformanceProofRequired,
                "The physical action-to-photon fixture and the fixed-fixture instrumentation-overhead gate have not been executed on the target display/adapter.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::OptionalGpuTimestamps,
                "GPU stages report unsupported when the adapter lacks reliable timestamp-query support; CPU action correlation remains available.",
            ),
        ],
        CapabilityKey::SourceDescriptorColorTruth => vec![
            CapabilityLimitationRecord::new(
                Limitation::DisplayGeometryIntegrationStopped,
                "Descriptor and conversion-policy truth is visible and exported, but clean aperture, SAR, rotation, and mirror are not yet applied to live or exported pixels; those surfaces remain evaluation-only.",
            ),
        ],
        CapabilityKey::SupervisedGpuRecoveryPhaseA => vec![
            CapabilityLimitationRecord::new(
                Limitation::TransparentGpuContinuityUnavailable,
                "Only Phase-A supervised restart is available; in-process Phase-B resource rebuild, source rebinding, and automatic audience continuity are unavailable.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "A packaged launcher deadline and real relaunch-to-recovery-surface measurement remain unexecuted.",
            ),
        ],
        CapabilityKey::StudyMotionAbi11 => vec![
            CapabilityLimitationRecord::new(
                Limitation::AdditiveAbiVersionPinned,
                "ABI 1.1 is additive and explicitly selected; ABI 1.0 remains frozen, and unresolved or unsupported group motion resolves neutral.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "The retained implementation is cross-platform code, but its physical GPU receipt covers only the available audit adapter.",
            ),
        ],
        CapabilityKey::ProxyBrowserSurface => vec![
            CapabilityLimitationRecord::new(
                Limitation::VerifiedContentIdentityRequired,
                "Only sources with a verified cos-sha256 identity can be proxied.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::OriginalMediaUsedForExport,
                "Offline export resolves the original media; a proxy never becomes patch or export identity.",
            ),
        ],
        CapabilityKey::FinalProgramVhs => vec![CapabilityLimitationRecord::new(
            Limitation::AsynchronousFinalProgramLatency,
            "Live VHS is an asynchronous final-program replacement; export runs the corresponding synchronous stage.",
        )],
        CapabilityKey::LiveRecorderAudioMux => vec![
            CapabilityLimitationRecord::new(
                Limitation::LiveAudioCaptureRequired,
                "Program audio is muxed exactly when the live audio capture stream is running at recording start; otherwise the artifact stays video-only and reports audio_not_muxed=true.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "Ring, drift, and device-loss laws are proven in software with an opt-in ffprobe fixture; a physical audio interface driving a live recording is hardware-matrix proof.",
            ),
        ],
        CapabilityKey::ExactVfrLiveTransport => vec![CapabilityLimitationRecord::new(
            Limitation::AverageFrameRateCadence,
            "Live public cadence is derived from the stream average frame rate, not an exact PTS timeline.",
        )],
        CapabilityKey::SpoutInput => vec![
            CapabilityLimitationRecord::new(
                Limitation::WindowsOnly,
                "The integrated Spout backend is Windows-only.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::LiveSourceIsBlackOffline,
                "Offline export represents live Spout input with the explicit deterministic-black policy.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "An external sender and target adapter are required for physical interoperability proof.",
            ),
        ],
        CapabilityKey::SpoutOutput => vec![
            CapabilityLimitationRecord::new(
                Limitation::WindowsOnly,
                "The integrated Spout backend is Windows-only.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "An external receiver and target adapter are required for physical interoperability proof.",
            ),
        ],
        CapabilityKey::AdvancedPrecision => vec![CapabilityLimitationRecord::new(
            Limitation::ExternalProofRequired,
            "One local GPU receipt does not prove all adapters or all three target platforms.",
        )],
        CapabilityKey::Full16TemporalHistory => vec![CapabilityLimitationRecord::new(
            Limitation::EvaluationFixtureOnly,
            "Full-16 temporal storage remains a measurement fixture and is not a production precision mode.",
        )],
        CapabilityKey::D3d11vaHardwareDecode => vec![
            CapabilityLimitationRecord::new(
                Limitation::WindowsOnly,
                "The current evaluation backend is specifically D3D11VA on Windows.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::EvaluationFixtureOnly,
                "The backend is constructed by the opt-in probe only; live decode remains software.",
            ),
        ],
        CapabilityKey::NdiInput | CapabilityKey::NdiOutput => vec![
            CapabilityLimitationRecord::new(
                Limitation::LicenseAndNetworkAuthorizationRequired,
                "SDK/license and venue network-policy authorization must precede integration proof.",
            ),
            CapabilityLimitationRecord::new(
                Limitation::ExternalProofRequired,
                "A licensed external endpoint is required for interoperability proof.",
            ),
        ],
        CapabilityKey::BoundedMeshWarp => vec![CapabilityLimitationRecord::new(
            Limitation::VenueRequirementRequired,
            "A demonstrated venue need must precede backend work and physical proof.",
        )],
        CapabilityKey::CaptureInput
        | CapabilityKey::SyphonInput
        | CapabilityKey::SyphonOutput
        | CapabilityKey::ZeroCopyDecode => vec![CapabilityLimitationRecord::new(
            Limitation::ExternalProofRequired,
            "Target hardware or an external endpoint is required for interoperability proof.",
        )],
    }
}

pub fn capability_registry(facts: CapabilityRuntimeFacts) -> Vec<CapabilityRecord> {
    CapabilityKey::ALL
        .into_iter()
        .map(|key| {
            let decision = primary_decision(key, facts);
            CapabilityRecord {
                key,
                title: key.title().to_owned(),
                platform: facts.platform,
                status: decision.status(),
                reason: decision.reason(),
                surfaces: surfaces(key, decision),
                evidence_receipt_ids: evidence(key),
                known_limitations: limitations(key),
            }
        })
        .collect()
}

pub fn capability_record(key: CapabilityKey, facts: CapabilityRuntimeFacts) -> CapabilityRecord {
    capability_registry(facts)
        .into_iter()
        .find(|record| record.key == key)
        .expect("every closed capability key has one registry record")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformCapabilityRegistry {
    pub platform: Platform,
    pub capabilities: Vec<CapabilityRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRegistryDocument {
    pub schema: String,
    pub platforms: Vec<PlatformCapabilityRegistry>,
}

pub fn canonical_registry_document() -> CapabilityRegistryDocument {
    CapabilityRegistryDocument {
        schema: CAPABILITY_REGISTRY_SCHEMA.to_owned(),
        platforms: Platform::ALL
            .into_iter()
            .map(|platform| PlatformCapabilityRegistry {
                platform,
                capabilities: capability_registry(CapabilityRuntimeFacts::current_tree(platform)),
            })
            .collect(),
    }
}

pub fn canonical_registry_json() -> String {
    let mut json = serde_json::to_string_pretty(&canonical_registry_document())
        .expect("the closed capability registry is serializable");
    json.push('\n');
    json
}

fn matrix_status(key: CapabilityKey, platform: Platform) -> CapabilityStatus {
    capability_record(key, CapabilityRuntimeFacts::current_tree(platform)).status
}

pub fn generated_capability_markdown() -> String {
    let mut output = String::from(
        "# Capability registry\n\n\
This file is generated by `cargo run --locked --bin generate_capabilities`. \
Do not edit it by hand. The stable JSON form is `docs/capability-registry.json`.\n\n\
| Registry key | Capability | Windows | macOS | Linux | Evidence receipt IDs |\n\
| --- | --- | --- | --- | --- | --- |\n",
    );
    for key in CapabilityKey::ALL {
        let windows = matrix_status(key, Platform::Windows).label();
        let macos = matrix_status(key, Platform::Macos).label();
        let linux = matrix_status(key, Platform::Linux).label();
        let receipts = evidence(key)
            .into_iter()
            .map(|id| format!("`{}`", id.0))
            .collect::<Vec<_>>()
            .join(", ");
        let receipts = if receipts.is_empty() {
            "—".to_owned()
        } else {
            receipts
        };
        output.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} |\n",
            key.as_str(),
            key.title(),
            windows,
            macos,
            linux,
            receipts
        ));
    }
    output.push_str(
        "\nStatuses are runtime facts, not UI labels: **Implemented**, **Evaluation required**, \
**Deferred**, **Rejected by measurement**, and **Unavailable on platform** are the complete vocabulary. \
Surface-specific status, typed reasons, limitations, and evidence identifiers are in the JSON artifact.\n",
    );
    output
}

pub fn generated_readme_summary() -> String {
    let keys = [
        CapabilityKey::AcceptedCreativeMutationV1,
        CapabilityKey::StudyMotionAbi11,
        CapabilityKey::ProxyBrowserSurface,
        CapabilityKey::TransactionalControlListeners,
        CapabilityKey::CorrelatedEngineGpuTiming,
        CapabilityKey::SourceDescriptorColorTruth,
        CapabilityKey::SupervisedGpuRecoveryPhaseA,
        CapabilityKey::D3d11vaHardwareDecode,
        CapabilityKey::FinalProgramVhs,
        CapabilityKey::LiveRecorderAudioMux,
        CapabilityKey::ExactVfrLiveTransport,
        CapabilityKey::SpoutInput,
        CapabilityKey::AdvancedPrecision,
        CapabilityKey::Full16TemporalHistory,
    ];
    let mut output = String::from(
        "<!-- BEGIN GENERATED CAPABILITY SUMMARY -->\n\
The authoritative, executable capability matrix is [generated here](docs/capability-registry.md).\n\n\
| Registry key | Windows | macOS | Linux |\n\
| --- | --- | --- | --- |\n",
    );
    for key in keys {
        output.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            key.as_str(),
            matrix_status(key, Platform::Windows).label(),
            matrix_status(key, Platform::Macos).label(),
            matrix_status(key, Platform::Linux).label(),
        ));
    }
    output.push_str("<!-- END GENERATED CAPABILITY SUMMARY -->");
    output
}

pub fn generated_proxy_surface_snippet() -> String {
    let record = capability_record(
        CapabilityKey::ProxyBrowserSurface,
        CapabilityRuntimeFacts::current_tree(Platform::Windows),
    );
    debug_assert_eq!(record.status, CapabilityStatus::Implemented);
    String::from(
        "<!-- BEGIN GENERATED PROXY CAPABILITY -->\n\
Registry key `proxy_browser_surface`: **Implemented**. The browser and native surfaces both request proxy encoding and report lifecycle status. Evidence receipt: `s8c-browser-proxy-surface-note`.\n\
<!-- END GENERATED PROXY CAPABILITY -->"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize)]
    struct AuditCampaignDocument {
        schema: String,
        campaigns: Vec<AuditCampaignRecord>,
    }

    #[derive(Debug, Deserialize)]
    struct AuditCampaignRecord {
        id: String,
        status: String,
        gate: String,
    }

    fn record(key: CapabilityKey, platform: Platform) -> CapabilityRecord {
        capability_record(key, CapabilityRuntimeFacts::current_tree(platform))
    }

    #[test]
    fn registry_is_complete_unique_sorted_serializable_and_path_free() {
        let document = canonical_registry_document();
        for platform in &document.platforms {
            assert_eq!(platform.capabilities.len(), CapabilityKey::ALL.len());
            for (expected, actual) in CapabilityKey::ALL.iter().zip(&platform.capabilities) {
                assert_eq!(*expected, actual.key);
                assert!(
                    !actual.evidence_receipt_ids.is_empty(),
                    "{:?}/{} lacks an evidence boundary",
                    platform.platform,
                    actual.key.as_str()
                );
                assert!(actual
                    .evidence_receipt_ids
                    .iter()
                    .all(|receipt| !receipt.0.trim().is_empty()));
            }
        }
        let first = canonical_registry_json();
        let second = canonical_registry_json();
        assert_eq!(first, second);
        let value: serde_json::Value = serde_json::from_str(&first).unwrap();
        for platform in value["platforms"].as_array().unwrap() {
            let keys = platform["capabilities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|record| record["key"].as_str().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                keys,
                CapabilityKey::ALL
                    .iter()
                    .map(|key| key.as_str())
                    .collect::<Vec<_>>()
            );
        }
        assert!(!first.contains("C:\\"));
        assert!(!first.contains("/Users/"));
        assert!(!first.contains("/home/"));
    }

    #[test]
    fn the_external_and_deferred_boundary_receipt_is_attached_to_exactly_nine_keys() {
        for platform in Platform::ALL {
            let records = capability_registry(CapabilityRuntimeFacts::current_tree(platform));
            let attached = records
                .iter()
                .filter(|record| {
                    record
                        .evidence_receipt_ids
                        .iter()
                        .any(|receipt| receipt.0 == EXTERNAL_DEFERRED_EVIDENCE_BOUNDARY_RECEIPT)
                })
                .map(|record| record.key)
                .collect::<Vec<_>>();
            assert_eq!(attached, EXTERNAL_DEFERRED_EVIDENCE_KEYS);
        }
    }

    #[test]
    fn proxy_browser_surface_contradiction_is_closed() {
        let proxy = record(CapabilityKey::ProxyBrowserSurface, Platform::Windows);
        assert_eq!(proxy.status, CapabilityStatus::Implemented);
        assert!(proxy.surfaces.iter().any(|surface| {
            surface.surface == CapabilitySurface::BrowserControl
                && surface.status == CapabilityStatus::Implemented
        }));
        let panel = include_str!("../static/app.js");
        assert!(panel.contains("action: 'request_layer_proxy'"));
        let precision_doc = include_str!("../docs/precision-and-scale.md");
        assert!(!precision_doc.contains("browser panel has no proxy surface"));
        assert!(precision_doc.contains("BEGIN GENERATED PROXY CAPABILITY"));
    }

    #[test]
    fn d3d11va_is_evaluation_only_on_windows_and_unavailable_elsewhere() {
        assert_eq!(
            record(CapabilityKey::D3d11vaHardwareDecode, Platform::Windows).status,
            CapabilityStatus::EvaluationRequired
        );
        assert_eq!(
            record(CapabilityKey::D3d11vaHardwareDecode, Platform::Macos).status,
            CapabilityStatus::UnavailableOnPlatform
        );
        assert_eq!(
            record(CapabilityKey::D3d11vaHardwareDecode, Platform::Linux).status,
            CapabilityStatus::UnavailableOnPlatform
        );
        let backend = include_str!("video/hw_decode.rs");
        assert!(backend.contains("evaluation-only"));
        assert!(backend.contains("hardware_decode_backend_exists_on_this_platform"));
        assert!(!include_str!("../docs/precision-and-scale.md")
            .contains("does not add a hardware decoder"));
    }

    #[test]
    fn final_program_vhs_is_implemented_live_and_offline_in_one_order() {
        let vhs = record(CapabilityKey::FinalProgramVhs, Platform::Windows);
        assert_eq!(vhs.status, CapabilityStatus::Implemented);
        for surface in [
            CapabilitySurface::LiveProgram,
            CapabilitySurface::OfflineExport,
        ] {
            assert!(vhs.surfaces.iter().any(|entry| {
                entry.surface == surface && entry.status == CapabilityStatus::Implemented
            }));
        }
        let readme = include_str!("../README.md");
        assert!(readme.contains("Temporal → Codec Mosh → final-program VHS → blackout"));
    }

    #[test]
    fn recorder_audio_mux_is_implemented_and_exact_vfr_stays_deferred() {
        // The recorder half of the former pair landed with its own owned
        // Program PCM clock; the source-text pins hold the implementation to
        // the truthful report law rather than a label.
        assert_eq!(
            record(CapabilityKey::LiveRecorderAudioMux, Platform::Windows).status,
            CapabilityStatus::Implemented
        );
        assert_eq!(
            record(CapabilityKey::ExactVfrLiveTransport, Platform::Windows).status,
            CapabilityStatus::Deferred
        );
        let recorder = include_str!("program_recorder.rs");
        assert!(recorder.contains("audio_not_muxed: audio.is_none()"));
        assert!(recorder.contains("pub struct ProgramAudioTap"));
        assert!(recorder.contains("fn correct_drift"));
        assert!(include_str!("video/decoder.rs").contains("stream.avg_frame_rate()"));
    }

    #[test]
    fn spout_scope_and_advanced_full16_decisions_are_exact() {
        assert_eq!(
            record(CapabilityKey::SpoutInput, Platform::Windows).status,
            CapabilityStatus::Implemented
        );
        assert_eq!(
            record(CapabilityKey::SpoutOutput, Platform::Linux).status,
            CapabilityStatus::UnavailableOnPlatform
        );
        assert_eq!(
            record(CapabilityKey::AdvancedPrecision, Platform::Windows).status,
            CapabilityStatus::Implemented
        );
        assert_eq!(
            record(CapabilityKey::Full16TemporalHistory, Platform::Windows).status,
            CapabilityStatus::RejectedByMeasurement
        );
        let precision = include_str!("precision.rs");
        assert!(precision.contains("AdvancedWorking16HistoryCompat8"));
        assert!(precision.contains("ExperimentalFull16History"));
        assert!(precision.contains("EvaluationOnly"));
    }

    #[test]
    fn p10_d1_and_d4_operator_capabilities_are_implemented_without_upgrading_stops() {
        for platform in Platform::ALL {
            let d1 = record(CapabilityKey::StudyMotionAbi11, platform);
            assert_eq!(d1.status, CapabilityStatus::Implemented);
            assert!(d1.surfaces.iter().all(|surface| {
                matches!(
                    surface.surface,
                    CapabilitySurface::LiveProgram | CapabilitySurface::OfflineExport
                ) && surface.status == CapabilityStatus::Implemented
            }));
            assert_eq!(d1.evidence_receipt_ids[0].0, "d1-study-motion-abi-1.1");

            let d4 = record(CapabilityKey::AcceptedCreativeMutationV1, platform);
            assert_eq!(d4.status, CapabilityStatus::Implemented);
            assert!(d4.surfaces.iter().any(|surface| {
                surface.surface == CapabilitySurface::LiveRecording
                    && surface.status == CapabilityStatus::Implemented
            }));
            assert_eq!(
                d4.evidence_receipt_ids[0].0,
                "d4-accepted-creative-mutation"
            );
        }

        let registry = canonical_registry_json();
        for unavailable_operator_capability in [
            "photosensitivity_advisor",
            "portable_show_bundle",
            "straight_alpha_key_fill",
        ] {
            assert!(
                !registry.contains(unavailable_operator_capability),
                "registry upgraded unavailable campaign {unavailable_operator_capability}"
            );
        }
        assert!(
            include_str!("../docs/rfcs/d2-photosensitivity-risk-advisor.md")
                .contains("does not declare an available live")
        );
        assert!(include_str!("../docs/rfcs/d3-portable-show-bundle.md").contains("operator UI"));
        assert!(include_str!("../docs/rfcs/d5-straight-alpha-export.md")
            .contains("cannot be reached by the current MP4 action"));
    }

    #[test]
    fn p10_new_operator_capabilities_have_runtime_facts_and_truthful_surface_boundaries() {
        let expected_identity = [
            (
                CapabilityKey::TransactionalControlListeners,
                "transactional_control_listeners",
                "Transactional control listeners and TLS identity",
            ),
            (
                CapabilityKey::CorrelatedEngineGpuTiming,
                "correlated_engine_gpu_timing",
                "Correlated engine and GPU-stage timing",
            ),
            (
                CapabilityKey::SourceDescriptorColorTruth,
                "source_descriptor_color_truth",
                "Source descriptor and color-truth diagnostics",
            ),
            (
                CapabilityKey::SupervisedGpuRecoveryPhaseA,
                "supervised_gpu_recovery_phase_a",
                "Supervised GPU recovery (Phase A)",
            ),
        ];
        for (key, stable_key, title) in expected_identity {
            assert!(CapabilityKey::ALL.contains(&key));
            assert_eq!(key.as_str(), stable_key);
            assert_eq!(key.title(), title);
            for platform in Platform::ALL {
                let capability = record(key, platform);
                assert_eq!(capability.status, CapabilityStatus::Implemented);
                assert!(!capability.surfaces.is_empty());
                assert!(!capability.evidence_receipt_ids.is_empty());
                assert!(!capability.known_limitations.is_empty());
            }
        }

        let listeners = record(
            CapabilityKey::TransactionalControlListeners,
            Platform::Windows,
        );
        for surface in [
            CapabilitySurface::BrowserControl,
            CapabilitySurface::NativeControl,
            CapabilitySurface::Backend,
        ] {
            assert!(listeners.surfaces.iter().any(|entry| {
                entry.surface == surface && entry.status == CapabilityStatus::Implemented
            }));
        }
        assert!(listeners.surfaces.iter().any(|entry| {
            entry.surface == CapabilitySurface::PhysicalVenue
                && entry.status == CapabilityStatus::EvaluationRequired
        }));

        let timing = record(CapabilityKey::CorrelatedEngineGpuTiming, Platform::Windows);
        assert!(timing.known_limitations.iter().any(|limitation| {
            limitation.code == CapabilityLimitation::EngineSubmissionIsNotPhotonTime
        }));
        assert!(timing.known_limitations.iter().any(|limitation| {
            limitation.code == CapabilityLimitation::PhysicalTimingAndPerformanceProofRequired
        }));
        assert!(timing.surfaces.iter().any(|entry| {
            entry.surface == CapabilitySurface::HardwareInteroperability
                && entry.status == CapabilityStatus::EvaluationRequired
        }));

        let descriptors = record(CapabilityKey::SourceDescriptorColorTruth, Platform::Windows);
        for surface in [CapabilitySurface::NativeControl, CapabilitySurface::Backend] {
            assert!(descriptors.surfaces.iter().any(|entry| {
                entry.surface == surface && entry.status == CapabilityStatus::Implemented
            }));
        }
        for surface in [
            CapabilitySurface::LiveProgram,
            CapabilitySurface::OfflineExport,
        ] {
            assert!(descriptors.surfaces.iter().any(|entry| {
                entry.surface == surface
                    && entry.status == CapabilityStatus::EvaluationRequired
                    && entry.reason
                        == CapabilityReason::Evaluation(
                            CapabilityEvaluationRequirement::RendererIntegrationProof,
                        )
            }));
        }

        let recovery = record(
            CapabilityKey::SupervisedGpuRecoveryPhaseA,
            Platform::Windows,
        );
        assert!(!recovery
            .surfaces
            .iter()
            .any(|entry| entry.surface == CapabilitySurface::LiveProgram));
        assert!(recovery.known_limitations.iter().any(|limitation| {
            limitation.code == CapabilityLimitation::TransparentGpuContinuityUnavailable
        }));

        let mut facts = CapabilityRuntimeFacts::current_tree(Platform::Windows);
        facts.transactional_control_listeners_integrated = false;
        facts.correlated_engine_gpu_timing_integrated = false;
        facts.source_descriptor_color_truth_integrated = false;
        facts.supervised_gpu_recovery_phase_a_integrated = false;
        for key in expected_identity.map(|entry| entry.0) {
            assert_eq!(
                capability_record(key, facts).status,
                CapabilityStatus::Deferred,
                "{key:?} must derive its status from its production integration fact"
            );
        }

        assert!(include_str!("web/server.rs").contains("Ipv4Addr::LOCALHOST"));
        assert!(include_str!("web/tls_identity.rs").contains("atomic_replace"));
        assert!(include_str!("action_correlation.rs").contains("ActionCorrelationMonitor"));
        assert!(include_str!("renderer/gpu_timing.rs").contains("map_buffer_on_submit"));
        assert!(include_str!("video/source_descriptor.rs").contains("SourceUvReference"));
        assert!(include_str!("gpu_recovery.rs").contains("SupervisedRestartRequired"));

        let registry = canonical_registry_json();
        assert!(!registry.contains("transparent_gpu_recovery"));
        assert!(!registry.contains("source_display_geometry_application"));
    }

    #[test]
    fn p10_campaign_status_is_closed_unique_and_matches_d1_through_d5_receipts() {
        let document: AuditCampaignDocument =
            serde_json::from_str(include_str!("../docs/campaigns/audit-campaign-status.json"))
                .unwrap();
        assert_eq!(document.schema, "collide-o-scope-audit-campaign-status/1");
        let allowed = [
            "retained",
            "implemented",
            "evaluation",
            "deferred",
            "rejected",
        ];
        let mut ids = std::collections::BTreeSet::new();
        for campaign in &document.campaigns {
            assert!(ids.insert(campaign.id.as_str()), "duplicate campaign id");
            assert!(
                allowed.contains(&campaign.status.as_str()),
                "unknown campaign status {}",
                campaign.status
            );
        }
        for (id, status, gate) in [
            ("d1_study_motion_abi_1_1", "implemented", "complete"),
            (
                "d2_photosensitivity_advisor",
                "evaluation",
                "accessibility_legal_and_p1_gpu_timing",
            ),
            (
                "d3_portable_show_bundle",
                "retained",
                "machine_a_clean_machine_b_live_export",
            ),
            ("d4_accepted_creative_mutation", "implemented", "complete"),
            (
                "d5_straight_alpha_key_fill",
                "retained",
                "application_action_live_acquisition_and_p1_readback",
            ),
        ] {
            let campaign = document
                .campaigns
                .iter()
                .find(|campaign| campaign.id == id)
                .expect("required D1-D5 campaign");
            assert_eq!(campaign.status, status);
            assert_eq!(campaign.gate, gate);
        }

        assert!(include_str!("../docs/rfcs/d1-study-motion-abi-1.1.md")
            .contains("ABI 1.1 implemented additively"));
        assert!(
            include_str!("../docs/rfcs/d4-accepted-creative-mutation.md")
                .contains("Status: **implemented")
        );
        let old_d5 = include_str!("../docs/rfcs/d5-straight-alpha-and-key-fill-export.md");
        assert!(old_d5.contains("superseded; not an availability statement"));
        assert!(old_d5.contains("docs/rfcs/d5-straight-alpha-export.md"));
    }
}
