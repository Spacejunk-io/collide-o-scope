//! Pure precision measurement, exact surface accounting, and capability truth.
//!
//! The settled Advanced path works in RGBA16Float while retaining temporal
//! history/feedback in Compat8. Full-16 history is represented only as an
//! evaluation candidate. No value in this module claims portable free VRAM or
//! the existence of an external SDK/backend.

#![allow(
    dead_code,
    reason = "pure measurement, capability-evidence, and experimental full-16 planning remain acceptance/inspection contracts; production separately enforces the settled renderer ledger"
)]

use std::fmt;

use serde::{Deserialize, Serialize};

pub const PRECISION_MAX_EDGE: u32 = 16_384;
pub const PRECISION_MAX_PIXELS: u64 = 8_192 * 4_320;
pub const PRECISION_MAX_SURFACE_LAYERS: u32 = 256;
pub const PRECISION_MAX_GPU_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const PRECISION_MAX_HOST_TRANSFER_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const PRECISION_MAX_TOTAL_BYTES: u64 = 20 * 1024 * 1024 * 1024;
pub const PRECISION_MAX_FIXTURE_SAMPLES: usize = 1_048_576;
pub const PRECISION_MAX_FINITE_SAMPLE: f32 = 65_504.0;
const GRADIENT_EPSILON: f32 = 1.0 / 65_535.0;
const METRIC_EPSILON: f64 = 1.0e-15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceStorage {
    Compat8,
    Rgba16Float,
}

impl SurfaceStorage {
    pub const fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::Compat8 => 4,
            Self::Rgba16Float => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionPath {
    LegacyCompat8,
    AdvancedWorking16HistoryCompat8,
    ExperimentalFull16History,
}

pub const SETTLED_ADVANCED_PRECISION_PATH: PrecisionPath =
    PrecisionPath::AdvancedWorking16HistoryCompat8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionPathStatus {
    Compatibility,
    Settled,
    EvaluationOnly,
}

impl PrecisionPath {
    pub const fn working_storage(self) -> SurfaceStorage {
        match self {
            Self::LegacyCompat8 => SurfaceStorage::Compat8,
            Self::AdvancedWorking16HistoryCompat8 | Self::ExperimentalFull16History => {
                SurfaceStorage::Rgba16Float
            }
        }
    }

    pub const fn history_storage(self) -> SurfaceStorage {
        match self {
            Self::LegacyCompat8 | Self::AdvancedWorking16HistoryCompat8 => SurfaceStorage::Compat8,
            Self::ExperimentalFull16History => SurfaceStorage::Rgba16Float,
        }
    }

    pub const fn status(self) -> PrecisionPathStatus {
        match self {
            Self::LegacyCompat8 => PrecisionPathStatus::Compatibility,
            Self::AdvancedWorking16HistoryCompat8 => PrecisionPathStatus::Settled,
            Self::ExperimentalFull16History => PrecisionPathStatus::EvaluationOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionResourceLimits {
    pub max_edge: u32,
    pub max_pixels: u64,
    pub max_surface_layers: u32,
    pub max_gpu_bytes: u64,
    pub max_host_transfer_bytes: u64,
    pub max_total_bytes: u64,
}

impl Default for PrecisionResourceLimits {
    fn default() -> Self {
        Self {
            max_edge: 8_192,
            max_pixels: PRECISION_MAX_PIXELS,
            max_surface_layers: 128,
            max_gpu_bytes: 8 * 1024 * 1024 * 1024,
            max_host_transfer_bytes: 2 * 1024 * 1024 * 1024,
            max_total_bytes: 10 * 1024 * 1024 * 1024,
        }
    }
}

impl PrecisionResourceLimits {
    pub fn validate(self) -> Result<Self, PrecisionError> {
        if self.max_edge == 0
            || self.max_edge > PRECISION_MAX_EDGE
            || self.max_pixels == 0
            || self.max_pixels > PRECISION_MAX_PIXELS
            || self.max_surface_layers == 0
            || self.max_surface_layers > PRECISION_MAX_SURFACE_LAYERS
            || self.max_gpu_bytes == 0
            || self.max_gpu_bytes > PRECISION_MAX_GPU_BYTES
            || self.max_host_transfer_bytes > PRECISION_MAX_HOST_TRANSFER_BYTES
            || self.max_total_bytes == 0
            || self.max_total_bytes > PRECISION_MAX_TOTAL_BYTES
            || self.max_gpu_bytes > self.max_total_bytes
            || self.max_host_transfer_bytes > self.max_total_bytes
        {
            return Err(PrecisionError::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionResourceRequest {
    pub output_size: [u32; 2],
    pub path: PrecisionPath,
    pub working_layers: u32,
    pub history_layers: u32,
    pub staging_bytes: u64,
    pub readback_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionResourcePlan {
    pub output_size: [u32; 2],
    pub output_pixels: u64,
    pub path: PrecisionPath,
    pub working_storage: SurfaceStorage,
    pub history_storage: SurfaceStorage,
    pub working_layers: u32,
    pub history_layers: u32,
    pub working_bytes: u64,
    pub history_bytes: u64,
    pub gpu_bytes: u64,
    pub staging_bytes: u64,
    pub readback_bytes: u64,
    pub host_transfer_bytes: u64,
    pub total_bytes: u64,
    /// Portable wgpu does not expose one truthful cross-backend free-VRAM
    /// value. This remains false even after a request passes arithmetic caps.
    pub portable_vram_budget_measured: bool,
}

impl PrecisionResourcePlan {
    pub fn preflight(
        request: PrecisionResourceRequest,
        limits: PrecisionResourceLimits,
    ) -> Result<Self, PrecisionError> {
        let limits = limits.validate()?;
        let [width, height] = request.output_size;
        if width == 0 || height == 0 {
            return Err(PrecisionError::ZeroDimension);
        }
        if width > limits.max_edge || height > limits.max_edge {
            return Err(PrecisionError::EdgeLimit {
                requested: request.output_size,
                limit: limits.max_edge,
            });
        }
        let output_pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(PrecisionError::ArithmeticOverflow)?;
        if output_pixels > limits.max_pixels {
            return Err(PrecisionError::PixelLimit {
                pixels: output_pixels,
                limit: limits.max_pixels,
            });
        }
        let surface_layers = request
            .working_layers
            .checked_add(request.history_layers)
            .ok_or(PrecisionError::ArithmeticOverflow)?;
        if surface_layers > limits.max_surface_layers {
            return Err(PrecisionError::SurfaceLayerLimit {
                layers: surface_layers,
                limit: limits.max_surface_layers,
            });
        }

        let working_storage = request.path.working_storage();
        let history_storage = request.path.history_storage();
        let working_bytes = surface_bytes(output_pixels, request.working_layers, working_storage)?;
        let history_bytes = surface_bytes(output_pixels, request.history_layers, history_storage)?;
        let gpu_bytes = working_bytes
            .checked_add(history_bytes)
            .ok_or(PrecisionError::ArithmeticOverflow)?;
        if gpu_bytes > limits.max_gpu_bytes {
            return Err(PrecisionError::GpuByteLimit {
                bytes: gpu_bytes,
                limit: limits.max_gpu_bytes,
            });
        }
        let host_transfer_bytes = request
            .staging_bytes
            .checked_add(request.readback_bytes)
            .ok_or(PrecisionError::ArithmeticOverflow)?;
        if host_transfer_bytes > limits.max_host_transfer_bytes {
            return Err(PrecisionError::HostTransferByteLimit {
                bytes: host_transfer_bytes,
                limit: limits.max_host_transfer_bytes,
            });
        }
        let total_bytes = gpu_bytes
            .checked_add(host_transfer_bytes)
            .ok_or(PrecisionError::ArithmeticOverflow)?;
        if total_bytes > limits.max_total_bytes {
            return Err(PrecisionError::TotalByteLimit {
                bytes: total_bytes,
                limit: limits.max_total_bytes,
            });
        }
        Ok(Self {
            output_size: request.output_size,
            output_pixels,
            path: request.path,
            working_storage,
            history_storage,
            working_layers: request.working_layers,
            history_layers: request.history_layers,
            working_bytes,
            history_bytes,
            gpu_bytes,
            staging_bytes: request.staging_bytes,
            readback_bytes: request.readback_bytes,
            host_transfer_bytes,
            total_bytes,
            portable_vram_budget_measured: false,
        })
    }
}

/// Byte facts read from the accepted runtime plans and their allocation
/// snapshots. Categories are disjoint: creative owns format-ledger surfaces,
/// motion owns field/carrier resources, NTSC owns its processing textures,
/// staging owns mapped transfer buffers, and readback owns conversion images
/// or other retained capture payloads outside those buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeAllocationSnapshot {
    pub output_size: [u32; 2],
    pub path: PrecisionPath,
    pub working_layers: u32,
    pub history_layers: u32,
    pub creative_bytes: u64,
    pub motion_bytes: u64,
    pub ntsc_bytes: u64,
    pub staging_bytes: u64,
    pub readback_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeResourceLedger {
    pub precision: PrecisionResourcePlan,
    pub creative_bytes: u64,
    pub motion_bytes: u64,
    pub ntsc_bytes: u64,
    pub staging_bytes: u64,
    pub readback_bytes: u64,
    /// Sum of every disjoint retained GPU texture/buffer category above.
    pub gpu_resident_bytes: u64,
    /// Staging/readback are already included in `gpu_resident_bytes`; this is
    /// a subset reported separately for the host-transfer cap.
    pub host_transfer_bytes: u64,
}

impl RuntimeResourceLedger {
    /// Reconcile the actual accepted allocation snapshot with the precision
    /// format ledger. A creative byte mismatch fails closed rather than
    /// allowing object counters or planner padding to masquerade as proof.
    pub fn reconcile(
        snapshot: RuntimeAllocationSnapshot,
        limits: PrecisionResourceLimits,
    ) -> Result<Self, PrecisionError> {
        let limits = limits.validate()?;
        let precision = PrecisionResourcePlan::preflight(
            PrecisionResourceRequest {
                output_size: snapshot.output_size,
                path: snapshot.path,
                working_layers: snapshot.working_layers,
                history_layers: snapshot.history_layers,
                staging_bytes: snapshot.staging_bytes,
                readback_bytes: snapshot.readback_bytes,
            },
            limits,
        )?;
        if precision.gpu_bytes != snapshot.creative_bytes {
            return Err(PrecisionError::CreativeByteMismatch {
                calculated: precision.gpu_bytes,
                allocated: snapshot.creative_bytes,
            });
        }
        let gpu_resident_bytes = snapshot
            .creative_bytes
            .checked_add(snapshot.motion_bytes)
            .and_then(|value| value.checked_add(snapshot.ntsc_bytes))
            .and_then(|value| value.checked_add(snapshot.staging_bytes))
            .and_then(|value| value.checked_add(snapshot.readback_bytes))
            .ok_or(PrecisionError::ArithmeticOverflow)?;
        if gpu_resident_bytes > limits.max_gpu_bytes {
            return Err(PrecisionError::GpuByteLimit {
                bytes: gpu_resident_bytes,
                limit: limits.max_gpu_bytes,
            });
        }
        if gpu_resident_bytes > limits.max_total_bytes {
            return Err(PrecisionError::TotalByteLimit {
                bytes: gpu_resident_bytes,
                limit: limits.max_total_bytes,
            });
        }
        Ok(Self {
            precision,
            creative_bytes: snapshot.creative_bytes,
            motion_bytes: snapshot.motion_bytes,
            ntsc_bytes: snapshot.ntsc_bytes,
            staging_bytes: snapshot.staging_bytes,
            readback_bytes: snapshot.readback_bytes,
            gpu_resident_bytes,
            host_transfer_bytes: precision.host_transfer_bytes,
        })
    }
}

fn surface_bytes(pixels: u64, layers: u32, storage: SurfaceStorage) -> Result<u64, PrecisionError> {
    pixels
        .checked_mul(u64::from(layers))
        .and_then(|value| value.checked_mul(storage.bytes_per_pixel()))
        .ok_or(PrecisionError::ArithmeticOverflow)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionResourceDelta {
    pub additional_bytes: u64,
    pub released_bytes: u64,
}

impl PrecisionResourceDelta {
    pub fn between(baseline: PrecisionResourcePlan, candidate: PrecisionResourcePlan) -> Self {
        if candidate.total_bytes >= baseline.total_bytes {
            Self {
                additional_bytes: candidate.total_bytes - baseline.total_bytes,
                released_bytes: 0,
            }
        } else {
            Self {
                additional_bytes: 0,
                released_bytes: baseline.total_bytes - candidate.total_bytes,
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearRgbaFixture {
    samples: Box<[[f32; 4]]>,
}

impl LinearRgbaFixture {
    pub fn try_from_samples(samples: Vec<[f32; 4]>) -> Result<Self, PrecisionError> {
        if samples.is_empty() || samples.len() > PRECISION_MAX_FIXTURE_SAMPLES {
            return Err(PrecisionError::FixtureSampleCount(samples.len()));
        }
        if samples
            .iter()
            .flatten()
            .any(|value| !value.is_finite() || value.abs() > PRECISION_MAX_FINITE_SAMPLE)
        {
            return Err(PrecisionError::NonFiniteOrUnboundedSample);
        }
        Ok(Self {
            samples: samples.into_boxed_slice(),
        })
    }

    pub fn samples(&self) -> &[[f32; 4]] {
        &self.samples
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionMeasurement {
    pub sample_count: usize,
    pub channel_count: u64,
    pub rmse: f64,
    pub max_absolute_error: f32,
    pub clamped_channel_events: u64,
    pub reference_gradient_events: u64,
    pub retained_gradient_events: u64,
}

pub fn measure_precision(
    reference: &LinearRgbaFixture,
    observed: &LinearRgbaFixture,
) -> Result<PrecisionMeasurement, PrecisionError> {
    if reference.samples.len() != observed.samples.len() {
        return Err(PrecisionError::FixtureLengthMismatch {
            reference: reference.samples.len(),
            observed: observed.samples.len(),
        });
    }
    let mut squared_error = 0.0_f64;
    let mut max_absolute_error = 0.0_f32;
    let mut clamped_channel_events = 0_u64;
    for (reference_sample, observed_sample) in reference.samples.iter().zip(observed.samples.iter())
    {
        for (reference_channel, observed_channel) in
            reference_sample.iter().zip(observed_sample.iter())
        {
            let error = (*observed_channel - *reference_channel).abs();
            squared_error += f64::from(error) * f64::from(error);
            max_absolute_error = max_absolute_error.max(error);
            if (*observed_channel <= 0.0 && *reference_channel > 0.0)
                || (*observed_channel >= 1.0 && *reference_channel < 1.0)
            {
                clamped_channel_events += 1;
            }
        }
    }

    let mut reference_gradient_events = 0_u64;
    let mut retained_gradient_events = 0_u64;
    for (reference_pair, observed_pair) in reference
        .samples
        .windows(2)
        .zip(observed.samples.windows(2))
    {
        let reference_delta = luminance(reference_pair[1]) - luminance(reference_pair[0]);
        if reference_delta.abs() < GRADIENT_EPSILON {
            continue;
        }
        reference_gradient_events += 1;
        let observed_delta = luminance(observed_pair[1]) - luminance(observed_pair[0]);
        if observed_delta.abs() >= GRADIENT_EPSILON
            && observed_delta.is_sign_positive() == reference_delta.is_sign_positive()
        {
            retained_gradient_events += 1;
        }
    }
    let channel_count = u64::try_from(reference.samples.len())
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or(PrecisionError::ArithmeticOverflow)?;
    Ok(PrecisionMeasurement {
        sample_count: reference.samples.len(),
        channel_count,
        rmse: (squared_error / channel_count as f64).sqrt(),
        max_absolute_error,
        clamped_channel_events,
        reference_gradient_events,
        retained_gradient_events,
    })
}

fn luminance(sample: [f32; 4]) -> f32 {
    sample[0].mul_add(0.2126, sample[1].mul_add(0.7152, sample[2] * 0.0722))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveGainVerdict {
    NoMeasuredGain,
    MeasuredObjectiveGain,
    ResourceOrMetricTradeoff,
    ObjectiveRegression,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArtisticGainAssessment {
    pub verdict: ObjectiveGainVerdict,
    pub rmse_reduction: f64,
    pub max_error_reduction: f64,
    pub clamped_events_avoided: i64,
    pub gradients_recovered: i64,
    pub resources: PrecisionResourceDelta,
}

impl ArtisticGainAssessment {
    pub fn compare(
        baseline: PrecisionMeasurement,
        candidate: PrecisionMeasurement,
        resources: PrecisionResourceDelta,
    ) -> Result<Self, PrecisionError> {
        if baseline.sample_count != candidate.sample_count
            || baseline.channel_count != candidate.channel_count
            || baseline.reference_gradient_events != candidate.reference_gradient_events
        {
            return Err(PrecisionError::IncomparableMeasurements);
        }
        let rmse_reduction = baseline.rmse - candidate.rmse;
        let max_error_reduction =
            f64::from(baseline.max_absolute_error) - f64::from(candidate.max_absolute_error);
        let clamped_events_avoided = signed_difference(
            baseline.clamped_channel_events,
            candidate.clamped_channel_events,
        )?;
        let gradients_recovered = signed_difference(
            candidate.retained_gradient_events,
            baseline.retained_gradient_events,
        )?;
        let improved = rmse_reduction > METRIC_EPSILON
            || max_error_reduction > METRIC_EPSILON
            || clamped_events_avoided > 0
            || gradients_recovered > 0;
        let regressed = rmse_reduction < -METRIC_EPSILON
            || max_error_reduction < -METRIC_EPSILON
            || clamped_events_avoided < 0
            || gradients_recovered < 0;
        let verdict = match (improved, regressed, resources.additional_bytes != 0) {
            (false, false, _) => ObjectiveGainVerdict::NoMeasuredGain,
            (true, false, false) => ObjectiveGainVerdict::MeasuredObjectiveGain,
            (true, _, true) | (true, true, false) => ObjectiveGainVerdict::ResourceOrMetricTradeoff,
            (false, true, _) => ObjectiveGainVerdict::ObjectiveRegression,
        };
        Ok(Self {
            verdict,
            rmse_reduction,
            max_error_reduction,
            clamped_events_avoided,
            gradients_recovered,
            resources,
        })
    }
}

fn signed_difference(left: u64, right: u64) -> Result<i64, PrecisionError> {
    let difference = i128::from(left) - i128::from(right);
    i64::try_from(difference).map_err(|_| PrecisionError::ArithmeticOverflow)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CapabilityEvidence {
    pub platform_supported: bool,
    pub backend_integrated: bool,
    pub interoperability_proven: bool,
    pub sdk_license_authorized: bool,
    pub network_policy_authorized: bool,
    pub venue_requirement_proven: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityDeferredReason {
    PlatformUnsupported,
    BackendNotIntegrated,
    SdkOrLicenseRequired,
    NetworkPolicyRequired,
    VenueRequirementNotEstablished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityEvaluationRequirement {
    InteroperabilityProof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityDecision {
    Available,
    EvaluationRequired(CapabilityEvaluationRequirement),
    Deferred(CapabilityDeferredReason),
}

/// The one production source of capability evidence. Until this landed, the
/// only constructors of [`CapabilityEvidence`] were test fixtures typing
/// literals; every production question about a scale capability must now be
/// asked through here (or the convenience [`scale_capability_decision`]) so
/// there is exactly one place evidence can come from and exactly one place a
/// future backend, authorization, or proof changes it.
///
/// What each field honestly reports today:
///
/// - `platform_supported` is a real compile-target probe: the platform API a
///   backend would integrate against exists (D3D11VA/Media Foundation,
///   VideoToolbox/AVFoundation, VAAPI/V4L2 for decode and capture; Syphon is
///   macOS-only by definition; NDI ships SDKs for all three; the mesh warp is
///   ordinary portable `wgpu` geometry).
/// - `backend_integrated` is `false` for every capability, because no
///   backend exists in this codebase. This is a fact about the source tree,
///   not probeable at runtime; the constant lives here so the tree change
///   that integrates a backend is the same change that flips it.
/// - `sdk_license_authorized` and `network_policy_authorized` are `false`:
///   no authorization store exists, and neither is a thing a coding session
///   may grant itself — an NDI SDK license is a purchase and a network
///   policy is an operator/venue decision.
/// - `interoperability_proven` is `false`: no interop receipt exists. The
///   S2-receipt pattern (a tracked artifact regenerated by an opt-in probe)
///   is the shape such proof would take.
/// - `venue_requirement_proven` is `false`: a demonstrated venue requirement
///   is an operator fact nobody has recorded.
///
/// This function moves nothing to `Available` — with `backend_integrated`
/// false, every capability is `Deferred` on every platform, and the test
/// suite pins that so an accidental flip cannot ship silently.
pub fn probe_capability_evidence(capability: ScaleCapability) -> CapabilityEvidence {
    let platform_supported = match capability {
        ScaleCapability::HardwareDecode
        | ScaleCapability::ZeroCopyDecode
        | ScaleCapability::CaptureInput
        | ScaleCapability::NdiInput
        | ScaleCapability::NdiOutput
        | ScaleCapability::BoundedMeshWarp => true,
        ScaleCapability::SyphonInput | ScaleCapability::SyphonOutput => {
            cfg!(target_os = "macos")
        }
    };
    CapabilityEvidence {
        platform_supported,
        backend_integrated: false,
        interoperability_proven: false,
        sdk_license_authorized: false,
        network_policy_authorized: false,
        venue_requirement_proven: false,
    }
}

/// The single predicate for "what is this capability's status on this host
/// right now": the production probe fed through the evaluator.
pub fn scale_capability_decision(capability: ScaleCapability) -> CapabilityDecision {
    evaluate_scale_capability(capability, probe_capability_evidence(capability))
}

pub fn evaluate_scale_capability(
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
    if capability == ScaleCapability::BoundedMeshWarp && !evidence.venue_requirement_proven {
        return CapabilityDecision::Deferred(
            CapabilityDeferredReason::VenueRequirementNotEstablished,
        );
    }
    if !evidence.platform_supported {
        return CapabilityDecision::Deferred(CapabilityDeferredReason::PlatformUnsupported);
    }
    if !evidence.backend_integrated {
        return CapabilityDecision::Deferred(CapabilityDeferredReason::BackendNotIntegrated);
    }
    if !evidence.interoperability_proven {
        return CapabilityDecision::EvaluationRequired(
            CapabilityEvaluationRequirement::InteroperabilityProof,
        );
    }
    CapabilityDecision::Available
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionError {
    InvalidLimits,
    ZeroDimension,
    EdgeLimit { requested: [u32; 2], limit: u32 },
    PixelLimit { pixels: u64, limit: u64 },
    SurfaceLayerLimit { layers: u32, limit: u32 },
    GpuByteLimit { bytes: u64, limit: u64 },
    HostTransferByteLimit { bytes: u64, limit: u64 },
    TotalByteLimit { bytes: u64, limit: u64 },
    CreativeByteMismatch { calculated: u64, allocated: u64 },
    FixtureSampleCount(usize),
    NonFiniteOrUnboundedSample,
    FixtureLengthMismatch { reference: usize, observed: usize },
    IncomparableMeasurements,
    ArithmeticOverflow,
}

impl fmt::Display for PrecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("invalid precision resource limits"),
            Self::ZeroDimension => formatter.write_str("precision surface dimensions must be non-zero"),
            Self::EdgeLimit { requested, limit } => write!(
                formatter,
                "precision surface {}x{} exceeds edge limit {limit}",
                requested[0], requested[1]
            ),
            Self::PixelLimit { pixels, limit } => {
                write!(formatter, "precision surface has {pixels} pixels; limit is {limit}")
            }
            Self::SurfaceLayerLimit { layers, limit } => write!(
                formatter,
                "precision plan has {layers} surfaces; limit is {limit}"
            ),
            Self::GpuByteLimit { bytes, limit } => {
                write!(formatter, "precision GPU ledger is {bytes} bytes; limit is {limit}")
            }
            Self::HostTransferByteLimit { bytes, limit } => write!(
                formatter,
                "precision host-transfer ledger is {bytes} bytes; limit is {limit}"
            ),
            Self::TotalByteLimit { bytes, limit } => {
                write!(formatter, "precision total ledger is {bytes} bytes; limit is {limit}")
            }
            Self::CreativeByteMismatch {
                calculated,
                allocated,
            } => write!(
                formatter,
                "precision format ledger calculates {calculated} creative bytes; allocation snapshot reports {allocated}"
            ),
            Self::FixtureSampleCount(count) => write!(
                formatter,
                "precision fixture has {count} samples; valid range is 1..={PRECISION_MAX_FIXTURE_SAMPLES}"
            ),
            Self::NonFiniteOrUnboundedSample => formatter.write_str(
                "precision fixtures require finite samples inside the RGBA16Float numeric envelope",
            ),
            Self::FixtureLengthMismatch { reference, observed } => write!(
                formatter,
                "precision fixture lengths differ: reference {reference}, observed {observed}"
            ),
            Self::IncomparableMeasurements => formatter.write_str(
                "precision measurements must share sample and reference-gradient counts",
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("precision resource arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for PrecisionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> PrecisionResourceLimits {
        PrecisionResourceLimits {
            max_edge: PRECISION_MAX_EDGE,
            max_pixels: PRECISION_MAX_PIXELS,
            max_surface_layers: PRECISION_MAX_SURFACE_LAYERS,
            max_gpu_bytes: PRECISION_MAX_GPU_BYTES,
            max_host_transfer_bytes: PRECISION_MAX_HOST_TRANSFER_BYTES,
            max_total_bytes: PRECISION_MAX_TOTAL_BYTES,
        }
    }

    #[test]
    fn settled_advanced_path_and_full16_candidate_have_exact_byte_goldens() {
        let output_size = [1_920, 1_080];
        let pixels = 1_920_u64 * 1_080;
        let baseline = PrecisionResourcePlan::preflight(
            PrecisionResourceRequest {
                output_size,
                path: SETTLED_ADVANCED_PRECISION_PATH,
                working_layers: 8,
                history_layers: 25,
                staging_bytes: 4_096,
                readback_bytes: 8_192,
            },
            limits(),
        )
        .unwrap();
        assert_eq!(baseline.working_storage, SurfaceStorage::Rgba16Float);
        assert_eq!(baseline.history_storage, SurfaceStorage::Compat8);
        assert_eq!(baseline.working_bytes, pixels * 8 * 8);
        assert_eq!(baseline.history_bytes, pixels * 25 * 4);
        assert!(!baseline.portable_vram_budget_measured);

        let full16 = PrecisionResourcePlan::preflight(
            PrecisionResourceRequest {
                path: PrecisionPath::ExperimentalFull16History,
                ..PrecisionResourceRequest {
                    output_size,
                    path: SETTLED_ADVANCED_PRECISION_PATH,
                    working_layers: 8,
                    history_layers: 25,
                    staging_bytes: 4_096,
                    readback_bytes: 8_192,
                }
            },
            limits(),
        )
        .unwrap();
        assert_eq!(full16.path.status(), PrecisionPathStatus::EvaluationOnly);
        assert_eq!(full16.history_bytes, pixels * 25 * 8);
        assert_eq!(
            PrecisionResourceDelta::between(baseline, full16),
            PrecisionResourceDelta {
                additional_bytes: pixels * 25 * 4,
                released_bytes: 0,
            }
        );
    }

    #[test]
    fn resource_preflight_rejects_zero_hostile_and_underbudget_requests() {
        let request = PrecisionResourceRequest {
            output_size: [0, 1],
            path: SETTLED_ADVANCED_PRECISION_PATH,
            working_layers: 1,
            history_layers: 0,
            staging_bytes: 0,
            readback_bytes: 0,
        };
        assert_eq!(
            PrecisionResourcePlan::preflight(request, limits()),
            Err(PrecisionError::ZeroDimension)
        );
        assert!(matches!(
            PrecisionResourcePlan::preflight(
                PrecisionResourceRequest {
                    output_size: [8_192, 4_320],
                    working_layers: PRECISION_MAX_SURFACE_LAYERS,
                    history_layers: 1,
                    ..request
                },
                limits()
            ),
            Err(PrecisionError::SurfaceLayerLimit { .. })
        ));
        assert!(PrecisionResourceLimits {
            max_total_bytes: u64::MAX,
            ..limits()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn runtime_ledger_reconciles_actual_categories_and_rejects_one_byte_over_cap() {
        let output_size = [16, 16];
        let pixels = 16_u64 * 16;
        let creative_bytes = pixels * (8 * 8 + 25 * 4);
        let snapshot = RuntimeAllocationSnapshot {
            output_size,
            path: SETTLED_ADVANCED_PRECISION_PATH,
            working_layers: 8,
            history_layers: 25,
            creative_bytes,
            motion_bytes: 1_024,
            ntsc_bytes: 2_048,
            staging_bytes: 4_096,
            readback_bytes: 1_024,
        };
        let exact_total = creative_bytes + 1_024 + 2_048 + 4_096 + 1_024;
        let exact_limits = PrecisionResourceLimits {
            max_gpu_bytes: exact_total,
            max_host_transfer_bytes: 5_120,
            max_total_bytes: exact_total,
            ..limits()
        };
        let ledger = RuntimeResourceLedger::reconcile(snapshot, exact_limits).unwrap();
        assert_eq!(ledger.precision.working_bytes, pixels * 8 * 8);
        assert_eq!(ledger.precision.history_bytes, pixels * 25 * 4);
        assert_eq!(ledger.creative_bytes, creative_bytes);
        assert_eq!(ledger.gpu_resident_bytes, exact_total);
        assert_eq!(ledger.host_transfer_bytes, 5_120);

        assert_eq!(
            RuntimeResourceLedger::reconcile(
                snapshot,
                PrecisionResourceLimits {
                    max_gpu_bytes: exact_total - 1,
                    ..exact_limits
                }
            ),
            Err(PrecisionError::GpuByteLimit {
                bytes: exact_total,
                limit: exact_total - 1,
            })
        );
        assert_eq!(
            RuntimeResourceLedger::reconcile(
                RuntimeAllocationSnapshot {
                    creative_bytes: creative_bytes + 1,
                    ..snapshot
                },
                exact_limits,
            ),
            Err(PrecisionError::CreativeByteMismatch {
                calculated: creative_bytes,
                allocated: creative_bytes + 1,
            })
        );
    }

    #[test]
    fn objective_measurement_detects_error_clipping_and_gradient_recovery() {
        let reference = LinearRgbaFixture::try_from_samples(vec![
            [0.1, 0.1, 0.1, 1.0],
            [0.100_02, 0.100_02, 0.100_02, 1.0],
            [0.4, 0.4, 0.4, 1.0],
        ])
        .unwrap();
        let baseline = LinearRgbaFixture::try_from_samples(vec![
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 1.0],
        ])
        .unwrap();
        let candidate = LinearRgbaFixture::try_from_samples(reference.samples().to_vec()).unwrap();
        let baseline_measurement = measure_precision(&reference, &baseline).unwrap();
        let candidate_measurement = measure_precision(&reference, &candidate).unwrap();
        assert!(baseline_measurement.rmse > candidate_measurement.rmse);
        assert!(baseline_measurement.clamped_channel_events > 0);
        assert!(
            baseline_measurement.retained_gradient_events
                < candidate_measurement.retained_gradient_events
        );
        let gain = ArtisticGainAssessment::compare(
            baseline_measurement,
            candidate_measurement,
            PrecisionResourceDelta {
                additional_bytes: 1024,
                released_bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(gain.verdict, ObjectiveGainVerdict::ResourceOrMetricTradeoff);
        assert!(gain.rmse_reduction > 0.0);
        assert!(gain.gradients_recovered > 0);
    }

    #[test]
    fn hostile_fixtures_and_incomparable_measurements_fail_closed() {
        assert_eq!(
            LinearRgbaFixture::try_from_samples(vec![[f32::NAN; 4]]),
            Err(PrecisionError::NonFiniteOrUnboundedSample)
        );
        let one = LinearRgbaFixture::try_from_samples(vec![[0.0; 4]]).unwrap();
        let two = LinearRgbaFixture::try_from_samples(vec![[0.0; 4], [1.0; 4]]).unwrap();
        assert!(matches!(
            measure_precision(&one, &two),
            Err(PrecisionError::FixtureLengthMismatch { .. })
        ));
    }

    #[test]
    fn external_and_mesh_capabilities_are_deferred_without_real_evidence() {
        for capability in [
            ScaleCapability::HardwareDecode,
            ScaleCapability::ZeroCopyDecode,
            ScaleCapability::SyphonInput,
            ScaleCapability::SyphonOutput,
            ScaleCapability::NdiInput,
            ScaleCapability::NdiOutput,
            ScaleCapability::CaptureInput,
            ScaleCapability::BoundedMeshWarp,
        ] {
            assert!(matches!(
                evaluate_scale_capability(capability, CapabilityEvidence::default()),
                CapabilityDecision::Deferred(_)
            ));
        }
        assert_eq!(
            evaluate_scale_capability(
                ScaleCapability::NdiInput,
                CapabilityEvidence {
                    platform_supported: true,
                    backend_integrated: true,
                    interoperability_proven: true,
                    network_policy_authorized: true,
                    ..CapabilityEvidence::default()
                }
            ),
            CapabilityDecision::Deferred(CapabilityDeferredReason::SdkOrLicenseRequired)
        );
        let integrated = CapabilityEvidence {
            platform_supported: true,
            backend_integrated: true,
            ..CapabilityEvidence::default()
        };
        assert_eq!(
            evaluate_scale_capability(ScaleCapability::SyphonInput, integrated),
            CapabilityDecision::EvaluationRequired(
                CapabilityEvaluationRequirement::InteroperabilityProof
            )
        );
    }

    /// Pins the production probe's decision for every capability on the
    /// platform running the test. Nothing may be `Available` or even
    /// `EvaluationRequired` today — `backend_integrated` is false for the
    /// whole tree — and each capability's exact deferred reason is the
    /// actionable one: NDI names its purchase before anything else, the mesh
    /// warp names its venue fact, Syphon names the platform where that is
    /// the truth, and everything else lands on the honest
    /// `BackendNotIntegrated` — engineering, not an external gate. If a
    /// backend ever lands, this test is the loud reminder that its
    /// capability must move through `EvaluationRequired` with a real
    /// interoperability receipt, never straight to `Available`.
    #[test]
    fn the_production_probe_defers_every_capability_with_its_actionable_reason() {
        let expected_syphon = if cfg!(target_os = "macos") {
            CapabilityDeferredReason::BackendNotIntegrated
        } else {
            CapabilityDeferredReason::PlatformUnsupported
        };
        let table = [
            (
                ScaleCapability::HardwareDecode,
                CapabilityDeferredReason::BackendNotIntegrated,
            ),
            (
                ScaleCapability::ZeroCopyDecode,
                CapabilityDeferredReason::BackendNotIntegrated,
            ),
            (ScaleCapability::SyphonInput, expected_syphon),
            (ScaleCapability::SyphonOutput, expected_syphon),
            (
                ScaleCapability::NdiInput,
                CapabilityDeferredReason::SdkOrLicenseRequired,
            ),
            (
                ScaleCapability::NdiOutput,
                CapabilityDeferredReason::SdkOrLicenseRequired,
            ),
            (
                ScaleCapability::CaptureInput,
                CapabilityDeferredReason::BackendNotIntegrated,
            ),
            (
                ScaleCapability::BoundedMeshWarp,
                CapabilityDeferredReason::VenueRequirementNotEstablished,
            ),
        ];
        for (capability, reason) in table {
            assert_eq!(
                scale_capability_decision(capability),
                CapabilityDecision::Deferred(reason),
                "probe decision changed for {capability:?}"
            );
        }
    }
}
