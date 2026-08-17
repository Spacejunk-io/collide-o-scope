//! Pure motion-domain authoring, field construction, and resource planning.
//!
//! This module deliberately owns no decoder, GPU, clock, filesystem, or UI
//! state. Codec adapters provide validated side-data records, lattice callers
//! provide bounded luma planes, and render hosts preflight every low-resolution
//! field and persistent carrier before allocating a resource.

#![allow(
    dead_code,
    reason = "M4 exposes the frozen motion contract before all host/GPU consumers land"
)]

use std::fmt;

use crate::image_routing::StableLayerId;
use crate::performance::SavedLayerPosition;

/// Persisted provenance for the first motion-field algorithm family.
pub const MOTION_ALGORITHM_VERSION: u16 = 1;
/// Hard edge bound for any low-resolution motion surface.
pub const MOTION_FIELD_MAX_EDGE: u32 = 2_048;
/// Hard element bound for one low-resolution motion surface.
pub const MOTION_FIELD_MAX_VECTORS: u64 = 2_100_000;
/// Hard byte bound for one field's complete low-resolution working set.
pub const MOTION_FIELD_MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Composition-wide prepared motion working-set ceiling.
pub const MOTION_RESOURCE_MAX_BYTES: u64 = 128 * 1024 * 1024;
/// At most this many scopes may own a live field in one composition plan.
pub const MOTION_FIELD_MAX_ACTIVE_SLOTS: u32 = 16;
/// M4 intentionally admits one expensive motion transplant at a time.
pub const MOTION_MAX_ACTIVE_TRANSPLANTS: u32 = 1;
/// Compact field quantization covers this signed donor-local UV velocity.
/// Sixty-four is not an aesthetic range: it is the smallest round bound that
/// retains the complete High lattice search (8 pixels at 60 Hz) on every
/// dimension large enough to admit a displaced candidate. Larger codec-side
/// values clamp here instead of escaping the field contract.
pub const MOTION_MAX_UV_PER_SECOND: f32 = 64.0;
/// Hostile codec side data cannot contain an unbounded record array.
pub const MOTION_CODEC_VECTOR_MAX_RECORDS: usize = 262_144;

// Vector, gate, and carrier targets are double-buffered so staging can be
// committed or discarded with the outer accepted-frame transaction. The
// logical carrier count remains one even though it owns two parity surfaces.
const VECTOR_TEXTURE_BYTES_PER_CELL: u64 = 8; // two RG16Float surfaces
const GATE_TEXTURE_BYTES_PER_CELL: u64 = 4; // two RG8Unorm surfaces
const GARDEN_SIGNAL_BYTES_PER_CELL: u64 = 1; // one transient R8Unorm surface
const LUMA_PING_PONG_BYTES_PER_CELL: u64 = 2; // two R8Unorm surfaces
const CARRIER_BYTES_PER_PIXEL: u64 = 16; // two RGBA16Float surfaces
const PACKED_VECTOR_BYTES: u64 = 8;
const MAX_CODEC_BLOCK_PIXELS: u16 = 256;
const MAX_CODEC_REFERENCE_SECONDS: f32 = 10.0;

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn unit(value: f32, fallback: f32) -> f32 {
    finite_or(value, fallback).clamp(0.0, 1.0)
}

/// Authored source-selection law for a motion field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MotionFieldSource {
    /// Layers prefer valid codec vectors and visibly fall back to the lattice.
    /// The master has no codec source and therefore resolves to the lattice.
    #[default]
    Auto,
    /// Codec-only. Unavailable or invalid side data yields a zero field and a
    /// visible diagnostic; it never silently invokes the lattice.
    CodecVectors,
    /// Always run deterministic block matching.
    Lattice,
}

/// Fixed, artist-visible Motion Lattice tiers. Timing pressure never changes
/// these values behind the performer's back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MotionLatticeQuality {
    Draft,
    #[default]
    Live,
    High,
}

impl MotionLatticeQuality {
    pub const fn block_pixels(self) -> u32 {
        match self {
            Self::Draft => 16,
            Self::Live => 8,
            Self::High => 4,
        }
    }

    pub const fn search_radius(self) -> i32 {
        match self {
            Self::Draft => 2,
            Self::Live => 4,
            Self::High => 8,
        }
    }

    pub const fn update_hz(self) -> u32 {
        match self {
            Self::Draft => 15,
            Self::Live => 30,
            Self::High => 60,
        }
    }
}

/// Runtime donor identity. Saved patches retain only `saved_position` and
/// resolve a fresh process-stable ID after constructing the layer stack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MotionDonor {
    #[default]
    None,
    Selected {
        layer_id: StableLayerId,
        saved_position: SavedLayerPosition,
    },
    Missing {
        saved_position: SavedLayerPosition,
    },
}

/// Explicit deterministic initialization for persistent Faraday memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MotionCarrier {
    /// Transparent premultiplied RGBA.
    #[default]
    Transparent,
    /// Opaque black.
    Black,
    /// The first accepted recipient source frame.
    FirstSourceFrame,
}

/// Bounded authored Faraday Motion Transplant controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaradayParams {
    /// Zero is an exact bypass and owns no carrier.
    pub amount: f32,
    pub donor: MotionDonor,
    pub carrier: MotionCarrier,
    pub confidence_threshold: f32,
    pub confidence_softness: f32,
    pub refresh: f32,
    pub decay: f32,
    pub occlusion: f32,
}

impl Default for FaradayParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            donor: MotionDonor::None,
            carrier: MotionCarrier::Transparent,
            confidence_threshold: 0.1,
            confidence_softness: 0.05,
            refresh: 1.0,
            decay: 1.0,
            occlusion: 0.0,
        }
    }
}

impl FaradayParams {
    pub fn sanitized(self) -> Self {
        Self {
            amount: unit(self.amount, 0.0),
            donor: self.donor,
            carrier: self.carrier,
            confidence_threshold: unit(self.confidence_threshold, 0.1),
            confidence_softness: finite_or(self.confidence_softness, 0.05).clamp(0.0, 0.5),
            refresh: unit(self.refresh, 1.0),
            decay: unit(self.decay, 1.0),
            occlusion: unit(self.occlusion, 0.0),
        }
    }
}

/// Fixed Curved Shutter tiers. `Sharp` retains one source observation while a
/// zero shutter angle bypasses the shutter pass entirely.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CurvedShutterQuality {
    #[default]
    Sharp,
    Draft,
    Live,
    High,
}

impl CurvedShutterQuality {
    pub const fn sample_count(self) -> u8 {
        match self {
            Self::Sharp => 1,
            Self::Draft => 4,
            Self::Live => 8,
            Self::High => 16,
        }
    }
}

/// Motion-trajectory exposure authored against one frame period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvedShutterParams {
    /// 0 degrees is an exact sharp fast path; 360 spans one frame period.
    pub angle_degrees: f32,
    pub phase: f32,
    pub curvature: f32,
    pub chromatic_lag: f32,
    pub quality: CurvedShutterQuality,
}

impl Default for CurvedShutterParams {
    fn default() -> Self {
        Self {
            angle_degrees: 0.0,
            phase: 0.0,
            curvature: 0.0,
            chromatic_lag: 0.0,
            quality: CurvedShutterQuality::Sharp,
        }
    }
}

impl CurvedShutterParams {
    pub fn sanitized(self) -> Self {
        Self {
            angle_degrees: finite_or(self.angle_degrees, 0.0).clamp(0.0, 360.0),
            phase: finite_or(self.phase, 0.0).clamp(-1.0, 1.0),
            curvature: finite_or(self.curvature, 0.0).clamp(-2.0, 2.0),
            chromatic_lag: unit(self.chromatic_lag, 0.0),
            quality: self.quality,
        }
    }

    pub fn is_exact_zero(self) -> bool {
        self.angle_degrees == 0.0
    }
}

/// Complete M4 authored motion contract for one master or layer scope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionParams {
    pub algorithm_version: u16,
    pub field_source: MotionFieldSource,
    pub lattice_quality: MotionLatticeQuality,
    pub transplant: FaradayParams,
    pub shutter: CurvedShutterParams,
}

impl Default for MotionParams {
    fn default() -> Self {
        Self {
            algorithm_version: MOTION_ALGORITHM_VERSION,
            field_source: MotionFieldSource::Auto,
            lattice_quality: MotionLatticeQuality::Live,
            transplant: FaradayParams::default(),
            shutter: CurvedShutterParams::default(),
        }
    }
}

impl MotionParams {
    pub fn sanitized(self) -> Self {
        Self {
            algorithm_version: if self.algorithm_version == MOTION_ALGORITHM_VERSION {
                self.algorithm_version
            } else {
                MOTION_ALGORITHM_VERSION
            },
            field_source: self.field_source,
            lattice_quality: self.lattice_quality,
            transplant: self.transplant.sanitized(),
            shutter: self.shutter.sanitized(),
        }
    }

    pub fn is_exact_zero(self) -> bool {
        self.transplant.amount == 0.0 && self.shutter.angle_degrees == 0.0
    }
}

/// Effective field source and a stable diagnostic suitable for telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MotionFieldOrigin {
    #[default]
    None,
    CodecVectors,
    Lattice,
    LatticeFallback,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MotionSourceDiagnostic {
    #[default]
    None,
    CodecUnavailable,
    CodecUnavailableFallback,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionSourceDecision {
    pub origin: MotionFieldOrigin,
    pub diagnostic: MotionSourceDiagnostic,
}

/// Resolve authored source truth without consulting timing or load pressure.
pub const fn resolve_motion_source(
    requested: MotionFieldSource,
    is_master: bool,
    codec_vectors_available: bool,
) -> MotionSourceDecision {
    match requested {
        MotionFieldSource::Auto if is_master => MotionSourceDecision {
            origin: MotionFieldOrigin::Lattice,
            diagnostic: MotionSourceDiagnostic::None,
        },
        MotionFieldSource::Auto if codec_vectors_available => MotionSourceDecision {
            origin: MotionFieldOrigin::CodecVectors,
            diagnostic: MotionSourceDiagnostic::None,
        },
        MotionFieldSource::Auto => MotionSourceDecision {
            origin: MotionFieldOrigin::LatticeFallback,
            diagnostic: MotionSourceDiagnostic::CodecUnavailableFallback,
        },
        MotionFieldSource::CodecVectors if codec_vectors_available => MotionSourceDecision {
            origin: MotionFieldOrigin::CodecVectors,
            diagnostic: MotionSourceDiagnostic::None,
        },
        MotionFieldSource::CodecVectors => MotionSourceDecision {
            origin: MotionFieldOrigin::None,
            diagnostic: MotionSourceDiagnostic::CodecUnavailable,
        },
        MotionFieldSource::Lattice => MotionSourceDecision {
            origin: MotionFieldOrigin::Lattice,
            diagnostic: MotionSourceDiagnostic::None,
        },
    }
}

/// Canonical low-resolution grid shared by codec and computed motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionGrid {
    pub width: u32,
    pub height: u32,
    pub block_pixels: u32,
    pub vector_count: u64,
}

impl MotionGrid {
    pub fn for_source(
        source_dimensions: [u32; 2],
        quality: MotionLatticeQuality,
    ) -> Result<Self, MotionPlanError> {
        let [source_width, source_height] = source_dimensions;
        if source_width == 0 || source_height == 0 {
            return Err(MotionPlanError::InvalidDimensions(source_dimensions));
        }
        let block_pixels = quality.block_pixels();
        let width = source_width
            .checked_add(block_pixels - 1)
            .ok_or(MotionPlanError::ArithmeticOverflow)?
            / block_pixels;
        let height = source_height
            .checked_add(block_pixels - 1)
            .ok_or(MotionPlanError::ArithmeticOverflow)?
            / block_pixels;
        if width > MOTION_FIELD_MAX_EDGE || height > MOTION_FIELD_MAX_EDGE {
            return Err(MotionPlanError::FieldEdge {
                dimensions: [width, height],
                limit: MOTION_FIELD_MAX_EDGE,
            });
        }
        let vector_count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(MotionPlanError::ArithmeticOverflow)?;
        if vector_count > MOTION_FIELD_MAX_VECTORS {
            return Err(MotionPlanError::VectorCount {
                count: vector_count,
                limit: MOTION_FIELD_MAX_VECTORS,
            });
        }
        Ok(Self {
            width,
            height,
            block_pixels,
            vector_count,
        })
    }
}

/// Device facts needed by pure preflight. Hosts copy these from their adapter;
/// the motion domain does not depend on `wgpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionDeviceLimits {
    pub max_texture_dimension_2d: u32,
    pub max_buffer_size: u64,
    pub max_motion_bytes: u64,
}

impl MotionDeviceLimits {
    pub const fn new(max_texture_dimension_2d: u32, max_buffer_size: u64) -> Self {
        Self {
            max_texture_dimension_2d,
            max_buffer_size,
            max_motion_bytes: MOTION_RESOURCE_MAX_BYTES,
        }
    }
}

/// One master/layer request supplied to composition-wide motion preflight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionScopeResourceRequest {
    pub source_dimensions: [u32; 2],
    pub output_dimensions: [u32; 2],
    pub params: MotionParams,
    pub is_master: bool,
    pub codec_vectors_available: bool,
    /// A transplant recipient selected this scope as its donor. Donor fields
    /// are budgeted even when the donor's own authored effects are exact-zero.
    pub required_as_donor: bool,
    /// Refresh Garden selected this scope as its routed motion signal. This
    /// admits the same canonical field even when the layer's own Motion
    /// effects are exact-zero.
    pub required_as_garden_signal: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionResourcePlan {
    pub active_field_slots: u32,
    pub active_transplants: u32,
    pub persistent_carriers: u32,
    pub vector_bytes: u64,
    pub gate_bytes: u64,
    pub luma_bytes: u64,
    pub carrier_bytes: u64,
    pub garden_signal_bytes: u64,
    pub active_garden_signals: u32,
    pub total_bytes: u64,
    pub max_field_dimensions: [u32; 2],
    pub max_shutter_samples: u8,
}

impl MotionResourcePlan {
    pub fn preflight(
        requests: &[MotionScopeResourceRequest],
        limits: MotionDeviceLimits,
    ) -> Result<Self, MotionPlanError> {
        let mut plan = Self::default();
        for request in requests {
            let params = request.params.sanitized();
            let transplant_active = params.transplant.amount > 0.0;
            let shutter_active = !params.shutter.is_exact_zero();
            let field_required =
                request.required_as_donor || request.required_as_garden_signal || shutter_active;
            if !transplant_active && !field_required {
                continue;
            }
            if request.is_master && transplant_active {
                return Err(MotionPlanError::MasterTransplant);
            }
            for dimensions in [request.source_dimensions, request.output_dimensions] {
                if dimensions[0] == 0 || dimensions[1] == 0 {
                    return Err(MotionPlanError::InvalidDimensions(dimensions));
                }
                if dimensions[0] > limits.max_texture_dimension_2d
                    || dimensions[1] > limits.max_texture_dimension_2d
                {
                    return Err(MotionPlanError::DeviceTextureDimension {
                        dimensions,
                        limit: limits.max_texture_dimension_2d,
                    });
                }
            }

            if field_required {
                let grid =
                    MotionGrid::for_source(request.source_dimensions, params.lattice_quality)?;
                let packed_bytes = checked_bytes(grid.vector_count, PACKED_VECTOR_BYTES)?;
                if packed_bytes > limits.max_buffer_size {
                    return Err(MotionPlanError::DeviceBuffer {
                        bytes: packed_bytes,
                        limit: limits.max_buffer_size,
                    });
                }
                let vector_bytes = checked_bytes(grid.vector_count, VECTOR_TEXTURE_BYTES_PER_CELL)?;
                let gate_bytes = checked_bytes(grid.vector_count, GATE_TEXTURE_BYTES_PER_CELL)?;
                let decision = resolve_motion_source(
                    params.field_source,
                    request.is_master,
                    request.codec_vectors_available,
                );
                let luma_bytes = if matches!(
                    decision.origin,
                    MotionFieldOrigin::Lattice | MotionFieldOrigin::LatticeFallback
                ) || params.field_source == MotionFieldSource::Auto
                {
                    checked_bytes(grid.vector_count, LUMA_PING_PONG_BYTES_PER_CELL)?
                } else {
                    0
                };
                let garden_signal_bytes = if request.required_as_garden_signal {
                    checked_bytes(grid.vector_count, GARDEN_SIGNAL_BYTES_PER_CELL)?
                } else {
                    0
                };
                let field_bytes = vector_bytes
                    .checked_add(gate_bytes)
                    .and_then(|value| value.checked_add(luma_bytes))
                    .and_then(|value| value.checked_add(garden_signal_bytes))
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                if field_bytes > MOTION_FIELD_MAX_BYTES {
                    return Err(MotionPlanError::FieldBytes {
                        bytes: field_bytes,
                        limit: MOTION_FIELD_MAX_BYTES,
                    });
                }

                plan.active_field_slots = plan
                    .active_field_slots
                    .checked_add(1)
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                if plan.active_field_slots > MOTION_FIELD_MAX_ACTIVE_SLOTS {
                    return Err(MotionPlanError::TooManyActiveFields {
                        count: plan.active_field_slots,
                        limit: MOTION_FIELD_MAX_ACTIVE_SLOTS,
                    });
                }
                plan.vector_bytes = plan
                    .vector_bytes
                    .checked_add(vector_bytes)
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                plan.gate_bytes = plan
                    .gate_bytes
                    .checked_add(gate_bytes)
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                plan.luma_bytes = plan
                    .luma_bytes
                    .checked_add(luma_bytes)
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                plan.max_field_dimensions[0] = plan.max_field_dimensions[0].max(grid.width);
                plan.max_field_dimensions[1] = plan.max_field_dimensions[1].max(grid.height);
                if request.required_as_garden_signal {
                    plan.active_garden_signals = plan
                        .active_garden_signals
                        .checked_add(1)
                        .ok_or(MotionPlanError::ArithmeticOverflow)?;
                    if plan.active_garden_signals > 1 {
                        return Err(MotionPlanError::TooManyGardenSignals {
                            count: plan.active_garden_signals,
                            limit: 1,
                        });
                    }
                    plan.garden_signal_bytes = garden_signal_bytes;
                }
            }
            if shutter_active {
                plan.max_shutter_samples = plan
                    .max_shutter_samples
                    .max(params.shutter.quality.sample_count());
            }

            if transplant_active {
                plan.active_transplants = plan
                    .active_transplants
                    .checked_add(1)
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                if plan.active_transplants > MOTION_MAX_ACTIVE_TRANSPLANTS {
                    return Err(MotionPlanError::TooManyTransplants {
                        count: plan.active_transplants,
                        limit: MOTION_MAX_ACTIVE_TRANSPLANTS,
                    });
                }
                plan.persistent_carriers = 1;
                let output_pixels = u64::from(request.output_dimensions[0])
                    .checked_mul(u64::from(request.output_dimensions[1]))
                    .ok_or(MotionPlanError::ArithmeticOverflow)?;
                plan.carrier_bytes = checked_bytes(output_pixels, CARRIER_BYTES_PER_PIXEL)?;
            }
        }

        plan.total_bytes = plan
            .vector_bytes
            .checked_add(plan.gate_bytes)
            .and_then(|value| value.checked_add(plan.luma_bytes))
            .and_then(|value| value.checked_add(plan.carrier_bytes))
            .and_then(|value| value.checked_add(plan.garden_signal_bytes))
            .ok_or(MotionPlanError::ArithmeticOverflow)?;
        if plan.total_bytes > limits.max_motion_bytes {
            return Err(MotionPlanError::AggregateBytes {
                bytes: plan.total_bytes,
                limit: limits.max_motion_bytes,
            });
        }
        Ok(plan)
    }
}

fn checked_bytes(count: u64, bytes_per_element: u64) -> Result<u64, MotionPlanError> {
    count
        .checked_mul(bytes_per_element)
        .ok_or(MotionPlanError::ArithmeticOverflow)
}

/// Canonical scalar consumed by Refresh Garden's routed Motion gate. Motion
/// vectors are normalized image-space velocity per second; confidence and
/// visibility are the two admitted field gates stored alongside that vector.
pub fn refresh_garden_motion_signal(
    velocity_uv_per_second: [f32; 2],
    confidence: f32,
    visibility: f32,
) -> f32 {
    let finite = |value: f32| if value.is_finite() { value } else { 0.0 };
    let x = finite(velocity_uv_per_second[0]);
    let y = finite(velocity_uv_per_second[1]);
    let confidence = finite(confidence).clamp(0.0, 1.0);
    let visibility = finite(visibility).clamp(0.0, 1.0);
    (x.hypot(y) * confidence * visibility).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionPlanError {
    InvalidDimensions([u32; 2]),
    FieldEdge { dimensions: [u32; 2], limit: u32 },
    VectorCount { count: u64, limit: u64 },
    FieldBytes { bytes: u64, limit: u64 },
    AggregateBytes { bytes: u64, limit: u64 },
    DeviceTextureDimension { dimensions: [u32; 2], limit: u32 },
    DeviceBuffer { bytes: u64, limit: u64 },
    TooManyActiveFields { count: u32, limit: u32 },
    TooManyTransplants { count: u32, limit: u32 },
    TooManyGardenSignals { count: u32, limit: u32 },
    MasterTransplant,
    ArithmeticOverflow,
}

impl fmt::Display for MotionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(dimensions) => write!(
                f,
                "motion dimensions must be non-zero, got {}x{}",
                dimensions[0], dimensions[1]
            ),
            Self::FieldEdge { dimensions, limit } => write!(
                f,
                "motion grid {}x{} exceeds the {limit}-cell edge limit",
                dimensions[0], dimensions[1]
            ),
            Self::VectorCount { count, limit } => {
                write!(f, "motion grid has {count} vectors, exceeding {limit}")
            }
            Self::FieldBytes { bytes, limit } => {
                write!(f, "motion field needs {bytes} bytes, exceeding {limit}")
            }
            Self::AggregateBytes { bytes, limit } => write!(
                f,
                "prepared motion resources need {bytes} bytes, exceeding {limit}"
            ),
            Self::DeviceTextureDimension { dimensions, limit } => write!(
                f,
                "motion resource {}x{} exceeds device texture edge {limit}",
                dimensions[0], dimensions[1]
            ),
            Self::DeviceBuffer { bytes, limit } => write!(
                f,
                "packed motion upload needs {bytes} bytes, exceeding device buffer limit {limit}"
            ),
            Self::TooManyActiveFields { count, limit } => {
                write!(f, "motion plan requests {count} fields; limit is {limit}")
            }
            Self::TooManyTransplants { count, limit } => write!(
                f,
                "motion plan requests {count} transplants; limit is {limit}"
            ),
            Self::TooManyGardenSignals { count, limit } => write!(
                f,
                "motion plan requests {count} routed Garden signals; limit is {limit}"
            ),
            Self::MasterTransplant => write!(f, "Faraday transplant requires a layer recipient"),
            Self::ArithmeticOverflow => write!(f, "motion resource arithmetic overflow"),
        }
    }
}

impl std::error::Error for MotionPlanError {}

/// Unpacked semantic sample. Velocities are donor-local UV per second.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MotionVectorSample {
    pub velocity_uv_per_second: [f32; 2],
    pub confidence: f32,
    /// One minus occlusion; zero suppresses admission from this cell.
    pub visibility: f32,
}

/// Eight-byte host representation. It is compact enough for the absolute
/// field cap and deterministic across CPU architectures.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedMotionVector {
    velocity_x: i16,
    velocity_y: i16,
    confidence: u8,
    visibility: u8,
    reserved: u16,
}

impl PackedMotionVector {
    pub fn from_sample(sample: MotionVectorSample) -> Self {
        Self {
            velocity_x: pack_velocity(sample.velocity_uv_per_second[0]),
            velocity_y: pack_velocity(sample.velocity_uv_per_second[1]),
            confidence: pack_unit(sample.confidence),
            visibility: pack_unit(sample.visibility),
            reserved: 0,
        }
    }

    pub fn sample(self) -> MotionVectorSample {
        MotionVectorSample {
            velocity_uv_per_second: [
                unpack_velocity(self.velocity_x),
                unpack_velocity(self.velocity_y),
            ],
            confidence: f32::from(self.confidence) / 255.0,
            visibility: f32::from(self.visibility) / 255.0,
        }
    }
}

fn pack_velocity(value: f32) -> i16 {
    let normalized = finite_or(value, 0.0)
        .clamp(-MOTION_MAX_UV_PER_SECOND, MOTION_MAX_UV_PER_SECOND)
        / MOTION_MAX_UV_PER_SECOND;
    (normalized * f32::from(i16::MAX)).round() as i16
}

fn unpack_velocity(value: i16) -> f32 {
    f32::from(value) / f32::from(i16::MAX) * MOTION_MAX_UV_PER_SECOND
}

fn pack_unit(value: f32) -> u8 {
    (unit(value, 0.0) * 255.0).round() as u8
}

/// Bounded low-resolution field. Hidden image pixels are never carried here
/// or serialized by patch DTOs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionField {
    source_dimensions: [u32; 2],
    grid: MotionGrid,
    algorithm_version: u16,
    origin: MotionFieldOrigin,
    vectors: Vec<PackedMotionVector>,
}

impl MotionField {
    pub fn from_samples(
        source_dimensions: [u32; 2],
        grid: MotionGrid,
        origin: MotionFieldOrigin,
        samples: impl IntoIterator<Item = MotionVectorSample>,
    ) -> Result<Self, MotionFieldError> {
        validate_grid(source_dimensions, grid)?;
        let expected =
            usize::try_from(grid.vector_count).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
        let mut vectors = Vec::new();
        vectors
            .try_reserve_exact(expected)
            .map_err(|_| MotionFieldError::Allocation)?;
        let mut samples = samples.into_iter();
        for _ in 0..expected {
            let Some(sample) = samples.next() else {
                return Err(MotionFieldError::SampleCount {
                    expected,
                    actual: vectors.len(),
                });
            };
            vectors.push(PackedMotionVector::from_sample(sample));
        }
        if samples.next().is_some() {
            return Err(MotionFieldError::SampleCount {
                expected,
                // The ingress contract is bounded: observe only the first
                // excess sample instead of exhausting a hostile iterator.
                actual: expected.saturating_add(1),
            });
        }
        Ok(Self {
            source_dimensions,
            grid,
            algorithm_version: MOTION_ALGORITHM_VERSION,
            origin,
            vectors,
        })
    }

    pub fn zeroed(
        source_dimensions: [u32; 2],
        grid: MotionGrid,
        origin: MotionFieldOrigin,
    ) -> Result<Self, MotionFieldError> {
        let count =
            usize::try_from(grid.vector_count).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
        Self::from_samples(
            source_dimensions,
            grid,
            origin,
            std::iter::repeat_n(MotionVectorSample::default(), count),
        )
    }

    pub const fn source_dimensions(&self) -> [u32; 2] {
        self.source_dimensions
    }

    pub const fn grid(&self) -> MotionGrid {
        self.grid
    }

    pub const fn algorithm_version(&self) -> u16 {
        self.algorithm_version
    }

    pub const fn origin(&self) -> MotionFieldOrigin {
        self.origin
    }

    pub fn packed_vectors(&self) -> &[PackedMotionVector] {
        &self.vectors
    }

    pub fn sample(&self, x: u32, y: u32) -> Option<MotionVectorSample> {
        if x >= self.grid.width || y >= self.grid.height {
            return None;
        }
        let index = u64::from(y)
            .checked_mul(u64::from(self.grid.width))?
            .checked_add(u64::from(x))?;
        usize::try_from(index)
            .ok()
            .and_then(|index| self.vectors.get(index))
            .copied()
            .map(PackedMotionVector::sample)
    }
}

fn validate_grid(source_dimensions: [u32; 2], grid: MotionGrid) -> Result<(), MotionFieldError> {
    if source_dimensions[0] == 0
        || source_dimensions[1] == 0
        || grid.width == 0
        || grid.height == 0
        || grid.block_pixels == 0
    {
        return Err(MotionFieldError::InvalidDimensions);
    }
    if grid.width > MOTION_FIELD_MAX_EDGE || grid.height > MOTION_FIELD_MAX_EDGE {
        return Err(MotionFieldError::GridLimit);
    }
    let expected = u64::from(grid.width)
        .checked_mul(u64::from(grid.height))
        .ok_or(MotionFieldError::ArithmeticOverflow)?;
    if expected != grid.vector_count || expected > MOTION_FIELD_MAX_VECTORS {
        return Err(MotionFieldError::GridLimit);
    }
    let packed_bytes = expected
        .checked_mul(PACKED_VECTOR_BYTES)
        .ok_or(MotionFieldError::ArithmeticOverflow)?;
    if packed_bytes > MOTION_FIELD_MAX_BYTES {
        return Err(MotionFieldError::GridLimit);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MotionFieldError {
    InvalidDimensions,
    PlaneStride,
    PlaneLength,
    MismatchedPlanes,
    GridLimit,
    SampleCount { expected: usize, actual: usize },
    TooManyCodecVectors { count: usize, limit: usize },
    InvalidCodecVector { index: usize, reason: &'static str },
    Allocation,
    ArithmeticOverflow,
}

impl fmt::Display for MotionFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "motion field dimensions must be non-zero"),
            Self::PlaneStride => write!(f, "luma stride is smaller than its width"),
            Self::PlaneLength => write!(f, "luma plane is shorter than dimensions and stride"),
            Self::MismatchedPlanes => write!(f, "motion lattice luma planes do not match"),
            Self::GridLimit => write!(f, "motion field exceeds its canonical grid limits"),
            Self::SampleCount { expected, actual } => {
                write!(f, "motion field expected {expected} samples, got {actual}")
            }
            Self::TooManyCodecVectors { count, limit } => {
                write!(f, "codec side data has {count} vectors; limit is {limit}")
            }
            Self::InvalidCodecVector { index, reason } => {
                write!(f, "invalid codec motion vector {index}: {reason}")
            }
            Self::Allocation => write!(f, "motion field allocation failed"),
            Self::ArithmeticOverflow => write!(f, "motion field arithmetic overflow"),
        }
    }
}

impl std::error::Error for MotionFieldError {}

/// Borrowed 8-bit luma view used by the deterministic CPU reference matcher.
#[derive(Debug, Clone, Copy)]
pub struct LumaPlane<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub pixels: &'a [u8],
}

impl LumaPlane<'_> {
    fn validate(self) -> Result<(), MotionFieldError> {
        if self.width == 0 || self.height == 0 {
            return Err(MotionFieldError::InvalidDimensions);
        }
        let width =
            usize::try_from(self.width).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
        if self.stride < width {
            return Err(MotionFieldError::PlaneStride);
        }
        let height =
            usize::try_from(self.height).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
        let required = height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(self.stride))
            .and_then(|offset| offset.checked_add(width))
            .ok_or(MotionFieldError::ArithmeticOverflow)?;
        if self.pixels.len() < required {
            return Err(MotionFieldError::PlaneLength);
        }
        Ok(())
    }

    fn pixel(self, x: u32, y: u32) -> u8 {
        let index = usize::try_from(y).unwrap() * self.stride + usize::try_from(x).unwrap();
        self.pixels[index]
    }
}

/// Deterministic source-pixel matching over the same low-resolution R8
/// observation grid used by the production GPU path. Candidate zero is
/// evaluated first, followed by increasing Chebyshev rings in row-major order.
/// Five fixed cross taps and strict first-wins ties are part of the algorithm,
/// making static and ambiguous observations prefer exact zero motion.
pub fn deterministic_motion_lattice(
    previous: LumaPlane<'_>,
    current: LumaPlane<'_>,
    quality: MotionLatticeQuality,
) -> Result<MotionField, MotionFieldError> {
    previous.validate()?;
    current.validate()?;
    if previous.width != current.width || previous.height != current.height {
        return Err(MotionFieldError::MismatchedPlanes);
    }
    let source_dimensions = [current.width, current.height];
    let grid = MotionGrid::for_source(source_dimensions, quality)
        .map_err(|_| MotionFieldError::GridLimit)?;
    let previous_observations = lattice_observations(previous, grid)?;
    let current_observations = lattice_observations(current, grid)?;
    let candidates = lattice_candidates(quality.search_radius());
    let count =
        usize::try_from(grid.vector_count).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| MotionFieldError::Allocation)?;
    for block_y in 0..grid.height {
        for block_x in 0..grid.width {
            let mut best_cost = f32::INFINITY;
            let mut second_cost = f32::INFINITY;
            let mut best = (0_i32, 0_i32);
            for &(dx, dy) in &candidates {
                let Some(cost) = lattice_candidate_cost(
                    &previous_observations,
                    &current_observations,
                    grid,
                    source_dimensions,
                    block_x,
                    block_y,
                    dx,
                    dy,
                ) else {
                    continue;
                };
                if cost < best_cost {
                    second_cost = best_cost;
                    best_cost = cost;
                    best = (dx, dy);
                } else if cost < second_cost {
                    second_cost = cost;
                }
            }
            let confidence = if !second_cost.is_finite() || second_cost <= f32::EPSILON {
                0.0
            } else {
                ((second_cost - best_cost).max(0.0) / second_cost).clamp(0.0, 1.0)
            };
            let hz = quality.update_hz() as f32;
            samples.push(MotionVectorSample {
                velocity_uv_per_second: [
                    best.0 as f32 / current.width as f32 * hz,
                    best.1 as f32 / current.height as f32 * hz,
                ],
                confidence,
                visibility: 1.0,
            });
        }
    }
    MotionField::from_samples(source_dimensions, grid, MotionFieldOrigin::Lattice, samples)
}

const LATTICE_CROSS_TAPS: [(i32, i32); 5] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)];

fn lattice_observations(
    plane: LumaPlane<'_>,
    grid: MotionGrid,
) -> Result<Vec<f32>, MotionFieldError> {
    let count =
        usize::try_from(grid.vector_count).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
    let mut observations = Vec::new();
    observations
        .try_reserve_exact(count)
        .map_err(|_| MotionFieldError::Allocation)?;
    for y in 0..grid.height {
        for x in 0..grid.width {
            let uv = [
                (x as f32 + 0.5) / grid.width as f32,
                (y as f32 + 0.5) / grid.height as f32,
            ];
            let sampled = sample_luma_plane_linear(plane, uv);
            // The production observation target is R8Unorm. Quantizing here
            // makes the CPU oracle consume the same bounded evidence.
            observations.push((sampled * 255.0).round().clamp(0.0, 255.0) / 255.0);
        }
    }
    Ok(observations)
}

fn sample_luma_plane_linear(plane: LumaPlane<'_>, uv: [f32; 2]) -> f32 {
    let coordinate = [
        uv[0] * plane.width as f32 - 0.5,
        uv[1] * plane.height as f32 - 0.5,
    ];
    let base = [coordinate[0].floor() as i64, coordinate[1].floor() as i64];
    let fraction = [
        coordinate[0] - base[0] as f32,
        coordinate[1] - base[1] as f32,
    ];
    let maximum = [i64::from(plane.width) - 1, i64::from(plane.height) - 1];
    let sample = |x: i64, y: i64| {
        f32::from(plane.pixel(
            u32::try_from(x.clamp(0, maximum[0])).unwrap(),
            u32::try_from(y.clamp(0, maximum[1])).unwrap(),
        )) / 255.0
    };
    let upper = [base[0] + 1, base[1] + 1];
    let row_0 = sample(base[0], base[1])
        + (sample(upper[0], base[1]) - sample(base[0], base[1])) * fraction[0];
    let row_1 = sample(base[0], upper[1])
        + (sample(upper[0], upper[1]) - sample(base[0], upper[1])) * fraction[0];
    row_0 + (row_1 - row_0) * fraction[1]
}

fn sample_observation_linear(observations: &[f32], grid: MotionGrid, uv: [f32; 2]) -> f32 {
    let coordinate = [
        uv[0] * grid.width as f32 - 0.5,
        uv[1] * grid.height as f32 - 0.5,
    ];
    let base = [coordinate[0].floor() as i64, coordinate[1].floor() as i64];
    let fraction = [
        coordinate[0] - base[0] as f32,
        coordinate[1] - base[1] as f32,
    ];
    let maximum = [i64::from(grid.width) - 1, i64::from(grid.height) - 1];
    let sample = |x: i64, y: i64| {
        let x = u64::try_from(x.clamp(0, maximum[0])).unwrap();
        let y = u64::try_from(y.clamp(0, maximum[1])).unwrap();
        observations[usize::try_from(y * u64::from(grid.width) + x).unwrap()]
    };
    let upper = [base[0] + 1, base[1] + 1];
    let row_0 = sample(base[0], base[1])
        + (sample(upper[0], base[1]) - sample(base[0], base[1])) * fraction[0];
    let row_1 = sample(base[0], upper[1])
        + (sample(upper[0], upper[1]) - sample(base[0], upper[1])) * fraction[0];
    row_0 + (row_1 - row_0) * fraction[1]
}

#[allow(clippy::too_many_arguments)]
fn lattice_candidate_cost(
    previous: &[f32],
    current: &[f32],
    grid: MotionGrid,
    source_dimensions: [u32; 2],
    cell_x: u32,
    cell_y: u32,
    dx: i32,
    dy: i32,
) -> Option<f32> {
    let center = [
        (cell_x as f32 + 0.5) / grid.width as f32,
        (cell_y as f32 + 0.5) / grid.height as f32,
    ];
    let displacement = [
        dx as f32 / source_dimensions[0] as f32,
        dy as f32 / source_dimensions[1] as f32,
    ];
    let displaced = [center[0] - displacement[0], center[1] - displacement[1]];
    if displaced
        .iter()
        .any(|coordinate| !(0.0..=1.0).contains(coordinate))
    {
        return None;
    }
    let step = [1.0 / grid.width as f32, 1.0 / grid.height as f32];
    Some(
        LATTICE_CROSS_TAPS
            .into_iter()
            .fold(0.0, |cost, (tap_x, tap_y)| {
                let tap = [tap_x as f32 * step[0], tap_y as f32 * step[1]];
                let current_sample = sample_observation_linear(
                    current,
                    grid,
                    [center[0] + tap[0], center[1] + tap[1]],
                );
                let previous_sample = sample_observation_linear(
                    previous,
                    grid,
                    [displaced[0] + tap[0], displaced[1] + tap[1]],
                );
                cost + (current_sample - previous_sample).abs()
            }),
    )
}

fn lattice_candidates(radius: i32) -> Vec<(i32, i32)> {
    let radius = radius.clamp(0, 8);
    let side = usize::try_from(radius * 2 + 1).unwrap();
    let mut candidates = Vec::with_capacity(side * side);
    candidates.push((0, 0));
    for ring in 1..=radius {
        for dy in -ring..=ring {
            for dx in -ring..=ring {
                if dx.abs().max(dy.abs()) == ring {
                    candidates.push((dx, dy));
                }
            }
        }
    }
    candidates
}

/// Temporal direction of one decoded prediction record. M4 uses past
/// references only; future-only B-frame data is explicitly unavailable rather
/// than being interpreted with the wrong time direction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CodecReferenceDirection {
    #[default]
    Past,
    Future,
}

/// Decoder-neutral counterpart of FFmpeg motion-vector side data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodecMotionVector {
    /// Destination block center in source pixels.
    pub destination: [i32; 2],
    pub block: [u16; 2],
    /// Reference-minus-destination displacement in `motion_scale` units,
    /// matching FFmpeg's exported convention. Rasterization negates it to get
    /// forward motion from the past reference into the current frame.
    pub motion: [i32; 2],
    pub motion_scale: u16,
    /// Positive elapsed program seconds between reference and destination.
    pub seconds_from_reference: f32,
    pub reference: CodecReferenceDirection,
    /// Decoder-provided visibility/occlusion confidence, clamped to 0..1.
    pub visibility: f32,
}

/// Rasterize valid past-reference codec vectors into the same grid used by
/// Motion Lattice. An intra frame, or future-only B-frame data, returns `None`.
pub fn rasterize_codec_motion_vectors(
    source_dimensions: [u32; 2],
    quality: MotionLatticeQuality,
    vectors: &[CodecMotionVector],
) -> Result<Option<MotionField>, MotionFieldError> {
    if vectors.len() > MOTION_CODEC_VECTOR_MAX_RECORDS {
        return Err(MotionFieldError::TooManyCodecVectors {
            count: vectors.len(),
            limit: MOTION_CODEC_VECTOR_MAX_RECORDS,
        });
    }
    let grid = MotionGrid::for_source(source_dimensions, quality)
        .map_err(|_| MotionFieldError::GridLimit)?;
    if vectors.is_empty() {
        return Ok(None);
    }
    let count =
        usize::try_from(grid.vector_count).map_err(|_| MotionFieldError::ArithmeticOverflow)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| MotionFieldError::Allocation)?;
    samples.resize(count, MotionVectorSample::default());
    let mut best_area = Vec::new();
    best_area
        .try_reserve_exact(count)
        .map_err(|_| MotionFieldError::Allocation)?;
    best_area.resize(count, u32::MAX);
    let mut observed_past = false;
    for (index, vector) in vectors.iter().copied().enumerate() {
        validate_codec_vector(index, vector, source_dimensions)?;
        if vector.reference == CodecReferenceDirection::Future {
            continue;
        }
        observed_past = true;
        let half_width = i64::from(vector.block[0]) / 2;
        let half_height = i64::from(vector.block[1]) / 2;
        let left = i64::from(vector.destination[0]) - half_width;
        let top = i64::from(vector.destination[1]) - half_height;
        let right = left + i64::from(vector.block[0]);
        let bottom = top + i64::from(vector.block[1]);
        let clipped_left = left.clamp(0, i64::from(source_dimensions[0]));
        let clipped_top = top.clamp(0, i64::from(source_dimensions[1]));
        let clipped_right = right.clamp(0, i64::from(source_dimensions[0]));
        let clipped_bottom = bottom.clamp(0, i64::from(source_dimensions[1]));
        if clipped_left >= clipped_right || clipped_top >= clipped_bottom {
            continue;
        }
        let block = i64::from(grid.block_pixels);
        let grid_left = u32::try_from(clipped_left / block).unwrap();
        let grid_top = u32::try_from(clipped_top / block).unwrap();
        let grid_right = u32::try_from((clipped_right + block - 1) / block)
            .unwrap()
            .min(grid.width);
        let grid_bottom = u32::try_from((clipped_bottom + block - 1) / block)
            .unwrap()
            .min(grid.height);
        let area = u32::from(vector.block[0]) * u32::from(vector.block[1]);
        let seconds = vector.seconds_from_reference;
        let forward_x = -(vector.motion[0] as f32 / f32::from(vector.motion_scale));
        let forward_y = -(vector.motion[1] as f32 / f32::from(vector.motion_scale));
        let sample = MotionVectorSample {
            velocity_uv_per_second: [
                forward_x / source_dimensions[0] as f32 / seconds,
                forward_y / source_dimensions[1] as f32 / seconds,
            ],
            confidence: 1.0,
            visibility: unit(vector.visibility, 0.0),
        };
        for y in grid_top..grid_bottom {
            for x in grid_left..grid_right {
                let cell = usize::try_from(u64::from(y) * u64::from(grid.width) + u64::from(x))
                    .map_err(|_| MotionFieldError::ArithmeticOverflow)?;
                if area < best_area[cell] {
                    best_area[cell] = area;
                    samples[cell] = sample;
                }
            }
        }
    }
    if !observed_past {
        return Ok(None);
    }
    MotionField::from_samples(
        source_dimensions,
        grid,
        MotionFieldOrigin::CodecVectors,
        samples,
    )
    .map(Some)
}

fn validate_codec_vector(
    index: usize,
    vector: CodecMotionVector,
    source_dimensions: [u32; 2],
) -> Result<(), MotionFieldError> {
    if vector.block[0] == 0 || vector.block[1] == 0 {
        return Err(MotionFieldError::InvalidCodecVector {
            index,
            reason: "zero block extent",
        });
    }
    if vector.block[0] > MAX_CODEC_BLOCK_PIXELS || vector.block[1] > MAX_CODEC_BLOCK_PIXELS {
        return Err(MotionFieldError::InvalidCodecVector {
            index,
            reason: "block extent exceeds limit",
        });
    }
    if vector.motion_scale == 0 {
        return Err(MotionFieldError::InvalidCodecVector {
            index,
            reason: "zero motion scale",
        });
    }
    if !vector.seconds_from_reference.is_finite()
        || vector.seconds_from_reference <= 0.0
        || vector.seconds_from_reference > MAX_CODEC_REFERENCE_SECONDS
    {
        return Err(MotionFieldError::InvalidCodecVector {
            index,
            reason: "invalid reference time",
        });
    }
    let coordinate_limit = i64::from(source_dimensions[0].max(source_dimensions[1]))
        + i64::from(MAX_CODEC_BLOCK_PIXELS);
    if vector
        .destination
        .iter()
        .any(|coordinate| i64::from(*coordinate).abs() > coordinate_limit)
    {
        return Err(MotionFieldError::InvalidCodecVector {
            index,
            reason: "destination outside bounded frame margin",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generous_limits() -> MotionDeviceLimits {
        MotionDeviceLimits::new(8_192, 256 * 1024 * 1024)
    }

    #[test]
    fn authored_defaults_are_exact_and_hostile_numbers_sanitize() {
        let defaults = MotionParams::default();
        assert!(defaults.is_exact_zero());
        assert_eq!(defaults.algorithm_version, MOTION_ALGORITHM_VERSION);
        assert_eq!(defaults.shutter.quality.sample_count(), 1);

        let hostile = MotionParams {
            algorithm_version: u16::MAX,
            transplant: FaradayParams {
                amount: f32::NAN,
                confidence_threshold: f32::INFINITY,
                confidence_softness: -5.0,
                refresh: 7.0,
                decay: -2.0,
                occlusion: f32::NAN,
                ..FaradayParams::default()
            },
            shutter: CurvedShutterParams {
                angle_degrees: f32::INFINITY,
                phase: -5.0,
                curvature: 9.0,
                chromatic_lag: f32::NAN,
                quality: CurvedShutterQuality::High,
            },
            ..MotionParams::default()
        }
        .sanitized();
        assert_eq!(hostile.algorithm_version, MOTION_ALGORITHM_VERSION);
        assert_eq!(hostile.transplant.amount, 0.0);
        assert_eq!(hostile.transplant.confidence_threshold, 0.1);
        assert_eq!(hostile.transplant.confidence_softness, 0.0);
        assert_eq!(hostile.transplant.refresh, 1.0);
        assert_eq!(hostile.transplant.decay, 0.0);
        assert_eq!(hostile.transplant.occlusion, 0.0);
        assert_eq!(hostile.shutter.angle_degrees, 0.0);
        assert_eq!(hostile.shutter.phase, -1.0);
        assert_eq!(hostile.shutter.curvature, 2.0);
        assert_eq!(hostile.shutter.chromatic_lag, 0.0);
    }

    #[test]
    fn quality_tiers_are_fixed_and_source_fallback_is_truthful() {
        assert_eq!(MotionLatticeQuality::Draft.block_pixels(), 16);
        assert_eq!(MotionLatticeQuality::Draft.search_radius(), 2);
        assert_eq!(MotionLatticeQuality::Draft.update_hz(), 15);
        assert_eq!(MotionLatticeQuality::Live.block_pixels(), 8);
        assert_eq!(MotionLatticeQuality::Live.search_radius(), 4);
        assert_eq!(MotionLatticeQuality::Live.update_hz(), 30);
        assert_eq!(MotionLatticeQuality::High.block_pixels(), 4);
        assert_eq!(MotionLatticeQuality::High.search_radius(), 8);
        assert_eq!(MotionLatticeQuality::High.update_hz(), 60);
        assert_eq!(CurvedShutterQuality::Sharp.sample_count(), 1);
        assert_eq!(CurvedShutterQuality::Draft.sample_count(), 4);
        assert_eq!(CurvedShutterQuality::Live.sample_count(), 8);
        assert_eq!(CurvedShutterQuality::High.sample_count(), 16);

        assert_eq!(
            resolve_motion_source(MotionFieldSource::Auto, false, false),
            MotionSourceDecision {
                origin: MotionFieldOrigin::LatticeFallback,
                diagnostic: MotionSourceDiagnostic::CodecUnavailableFallback,
            }
        );
        assert_eq!(
            resolve_motion_source(MotionFieldSource::CodecVectors, false, false),
            MotionSourceDecision {
                origin: MotionFieldOrigin::None,
                diagnostic: MotionSourceDiagnostic::CodecUnavailable,
            }
        );
        assert_eq!(
            resolve_motion_source(MotionFieldSource::Auto, true, true).origin,
            MotionFieldOrigin::Lattice
        );
    }

    #[test]
    fn packed_field_sanitizes_hostile_vectors_and_is_exactly_eight_bytes() {
        assert_eq!(std::mem::size_of::<PackedMotionVector>(), 8);
        let grid = MotionGrid::for_source([4, 4], MotionLatticeQuality::High).unwrap();
        let field = MotionField::from_samples(
            [4, 4],
            grid,
            MotionFieldOrigin::Lattice,
            [MotionVectorSample {
                velocity_uv_per_second: [f32::NAN, f32::INFINITY],
                confidence: 7.0,
                visibility: -1.0,
            }],
        )
        .unwrap();
        let sample = field.sample(0, 0).unwrap();
        assert_eq!(sample.velocity_uv_per_second, [0.0, 0.0]);
        assert_eq!(sample.confidence, 1.0);
        assert_eq!(sample.visibility, 0.0);
        assert_eq!(field.sample(1, 0), None);
    }

    #[test]
    fn packed_field_collection_is_fallible_and_stops_at_the_first_excess_sample() {
        let grid = MotionGrid::for_source([4, 4], MotionLatticeQuality::High).unwrap();
        assert_eq!(grid.vector_count, 1);
        assert_eq!(
            MotionField::from_samples(
                [4, 4],
                grid,
                MotionFieldOrigin::CodecVectors,
                std::iter::empty(),
            ),
            Err(MotionFieldError::SampleCount {
                expected: 1,
                actual: 0,
            })
        );
        assert_eq!(
            MotionField::from_samples(
                [4, 4],
                grid,
                MotionFieldOrigin::CodecVectors,
                std::iter::repeat(MotionVectorSample::default()),
            ),
            Err(MotionFieldError::SampleCount {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn routed_garden_scalar_and_low_resolution_resource_are_bounded_and_truthful() {
        assert_eq!(refresh_garden_motion_signal([0.0, 0.0], 1.0, 1.0), 0.0);
        assert_eq!(refresh_garden_motion_signal([3.0, 4.0], 0.5, 0.5), 1.0);
        assert!(
            (refresh_garden_motion_signal([0.6, 0.8], 0.5, 0.25) - 0.125).abs() <= f32::EPSILON
        );
        assert_eq!(
            refresh_garden_motion_signal([f32::NAN, f32::INFINITY], 2.0, -1.0),
            0.0
        );

        let request = MotionScopeResourceRequest {
            source_dimensions: [1920, 1080],
            output_dimensions: [1920, 1080],
            params: MotionParams::default(),
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: false,
            required_as_garden_signal: true,
        };
        let grid = MotionGrid::for_source([1920, 1080], MotionLatticeQuality::Live).unwrap();
        let plan = MotionResourcePlan::preflight(&[request], generous_limits()).unwrap();
        assert_eq!(plan.active_field_slots, 1);
        assert_eq!(plan.active_garden_signals, 1);
        assert_eq!(plan.garden_signal_bytes, grid.vector_count);
        assert_eq!(
            plan.total_bytes,
            plan.vector_bytes
                + plan.gate_bytes
                + plan.luma_bytes
                + plan.carrier_bytes
                + plan.garden_signal_bytes
        );
        assert!(matches!(
            MotionResourcePlan::preflight(&[request, request], generous_limits()),
            Err(MotionPlanError::TooManyGardenSignals { count: 2, limit: 1 })
        ));
    }

    #[test]
    fn resource_preflight_is_bounded_and_rejects_a_second_transplant() {
        let active = MotionParams {
            transplant: FaradayParams {
                amount: 1.0,
                donor: MotionDonor::Missing {
                    saved_position: SavedLayerPosition::new(0).unwrap(),
                },
                ..FaradayParams::default()
            },
            shutter: CurvedShutterParams {
                angle_degrees: 180.0,
                quality: CurvedShutterQuality::High,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };
        let request = MotionScopeResourceRequest {
            source_dimensions: [1920, 1080],
            output_dimensions: [1920, 1080],
            params: active,
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: false,
            required_as_garden_signal: false,
        };
        let plan = MotionResourcePlan::preflight(&[request], generous_limits()).unwrap();
        assert_eq!(plan.active_field_slots, 1);
        assert_eq!(plan.active_transplants, 1);
        assert_eq!(plan.persistent_carriers, 1);
        assert_eq!(plan.carrier_bytes, 1920 * 1080 * 16);
        assert_eq!(plan.max_shutter_samples, 16);
        assert!(plan.total_bytes <= MOTION_RESOURCE_MAX_BYTES);
        assert!(matches!(
            MotionResourcePlan::preflight(&[request, request], generous_limits()),
            Err(MotionPlanError::TooManyTransplants { count: 2, limit: 1 })
        ));
        assert!(matches!(
            MotionResourcePlan::preflight(
                &[MotionScopeResourceRequest {
                    is_master: true,
                    ..request
                }],
                generous_limits()
            ),
            Err(MotionPlanError::MasterTransplant)
        ));

        let carrier_only = MotionScopeResourceRequest {
            params: MotionParams {
                transplant: FaradayParams {
                    amount: 1.0,
                    ..FaradayParams::default()
                },
                ..MotionParams::default()
            },
            ..request
        };
        let carrier_bytes = 1920 * 1080 * 16;
        assert!(matches!(
            MotionResourcePlan::preflight(
                &[carrier_only],
                MotionDeviceLimits {
                    max_motion_bytes: carrier_bytes - 1,
                    ..generous_limits()
                }
            ),
            Err(MotionPlanError::AggregateBytes {
                bytes,
                limit
            }) if bytes == carrier_bytes && limit == carrier_bytes - 1
        ));
    }

    #[test]
    fn exact_donor_is_budgeted_but_a_missing_donor_never_allocates_or_retargets() {
        let recipient = MotionScopeResourceRequest {
            source_dimensions: [1920, 1080],
            output_dimensions: [1920, 1080],
            params: MotionParams {
                transplant: FaradayParams {
                    amount: 1.0,
                    donor: MotionDonor::Selected {
                        layer_id: StableLayerId::new(77).unwrap(),
                        saved_position: SavedLayerPosition::new(0).unwrap(),
                    },
                    ..FaradayParams::default()
                },
                ..MotionParams::default()
            },
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: false,
            required_as_garden_signal: false,
        };
        let donor = MotionScopeResourceRequest {
            source_dimensions: [1280, 720],
            output_dimensions: [1920, 1080],
            params: MotionParams::default(),
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: true,
            required_as_garden_signal: false,
        };
        let plan = MotionResourcePlan::preflight(&[recipient, donor], generous_limits()).unwrap();
        assert_eq!(plan.active_transplants, 1);
        assert_eq!(plan.active_field_slots, 1);
        assert_eq!(plan.max_field_dimensions, [160, 90]);

        let missing_recipient = MotionScopeResourceRequest {
            params: MotionParams {
                transplant: FaradayParams {
                    donor: MotionDonor::Missing {
                        saved_position: SavedLayerPosition::new(0).unwrap(),
                    },
                    ..recipient.params.transplant
                },
                ..recipient.params
            },
            ..recipient
        };
        let missing_plan =
            MotionResourcePlan::preflight(&[missing_recipient], generous_limits()).unwrap();
        assert_eq!(missing_plan.active_transplants, 1);
        assert_eq!(missing_plan.active_field_slots, 0);
        assert_eq!(missing_plan.vector_bytes, 0);
        assert_eq!(missing_plan.gate_bytes, 0);
        assert_eq!(missing_plan.luma_bytes, 0);
    }

    #[test]
    fn eight_k_high_double_buffered_field_stays_under_per_field_cap() {
        let request = MotionScopeResourceRequest {
            source_dimensions: [7680, 4320],
            output_dimensions: [7680, 4320],
            params: MotionParams {
                lattice_quality: MotionLatticeQuality::High,
                shutter: CurvedShutterParams {
                    angle_degrees: 1.0,
                    ..CurvedShutterParams::default()
                },
                ..MotionParams::default()
            },
            is_master: false,
            codec_vectors_available: false,
            required_as_donor: false,
            required_as_garden_signal: false,
        };
        let plan = MotionResourcePlan::preflight(&[request], generous_limits()).unwrap();
        let field_bytes = plan.vector_bytes + plan.gate_bytes + plan.luma_bytes;
        assert_eq!(plan.max_field_dimensions, [1920, 1080]);
        assert_eq!(field_bytes, 2_073_600 * 14);
        assert!(field_bytes <= MOTION_FIELD_MAX_BYTES);
    }

    fn textured(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .map(|index| {
                let x = (index % width) as f32;
                let y = (index / width) as f32;
                let signal = 0.5
                    + 0.18 * (x * 0.13).sin()
                    + 0.17 * (y * 0.17).cos()
                    + 0.12 * ((x + y) * 0.09).sin();
                (signal.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    fn shifted(previous: &[u8], width: u32, height: u32, dx: i32, dy: i32) -> Vec<u8> {
        let mut current = vec![0; previous.len()];
        for y in 0..height {
            for x in 0..width {
                let source_x = i64::from(x) - i64::from(dx);
                let source_y = i64::from(y) - i64::from(dy);
                if source_x >= 0
                    && source_y >= 0
                    && source_x < i64::from(width)
                    && source_y < i64::from(height)
                {
                    let destination = usize::try_from(y * width + x).unwrap();
                    let source = usize::try_from(
                        u32::try_from(source_y).unwrap() * width + u32::try_from(source_x).unwrap(),
                    )
                    .unwrap();
                    current[destination] = previous[source];
                }
            }
        }
        current
    }

    fn assert_lattice_shift(dx: i32, dy: i32) {
        let width = 64;
        let height = 48;
        let previous_pixels = textured(width, height);
        let current_pixels = shifted(&previous_pixels, width, height, dx, dy);
        let previous = LumaPlane {
            width,
            height,
            stride: width as usize,
            pixels: &previous_pixels,
        };
        let current = LumaPlane {
            width,
            height,
            stride: width as usize,
            pixels: &current_pixels,
        };
        let field =
            deterministic_motion_lattice(previous, current, MotionLatticeQuality::High).unwrap();
        let sample = field.sample(8, 6).unwrap();
        let frame_seconds = 1.0 / MotionLatticeQuality::High.update_hz() as f32;
        let recovered_x = sample.velocity_uv_per_second[0] * width as f32 * frame_seconds;
        let recovered_y = sample.velocity_uv_per_second[1] * height as f32 * frame_seconds;
        assert!(
            (recovered_x - dx as f32).abs() < 0.01,
            "{recovered_x} != {dx}"
        );
        assert!(
            (recovered_y - dy as f32).abs() < 0.01,
            "{recovered_y} != {dy}"
        );
        assert!(sample.confidence > 0.0);
    }

    #[test]
    fn deterministic_lattice_matches_one_two_and_four_pixel_fixtures() {
        assert_lattice_shift(1, 0);
        assert_lattice_shift(0, 2);
        assert_lattice_shift(4, -4);
    }

    #[test]
    fn deterministic_lattice_candidate_and_kernel_law_is_frozen() {
        assert_eq!(
            &lattice_candidates(2)[..9],
            &[
                (0, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
        );
        let high = lattice_candidates(MotionLatticeQuality::High.search_radius());
        assert_eq!(high.len(), 17 * 17);
        let mut unique = high.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), high.len());
        assert_eq!(
            LATTICE_CROSS_TAPS,
            [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        );
    }

    #[test]
    fn deterministic_lattice_static_fixture_is_zero_and_repeatable() {
        let width = 32;
        let height = 32;
        let pixels = textured(width, height);
        let plane = LumaPlane {
            width,
            height,
            stride: width as usize,
            pixels: &pixels,
        };
        let first = deterministic_motion_lattice(plane, plane, MotionLatticeQuality::High).unwrap();
        let second =
            deterministic_motion_lattice(plane, plane, MotionLatticeQuality::High).unwrap();
        assert_eq!(first, second);
        assert!(first
            .packed_vectors()
            .iter()
            .all(|sample| { sample.sample().velocity_uv_per_second == [0.0, 0.0] }));
    }

    #[test]
    fn codec_vectors_are_absent_on_intra_future_only_and_reject_malformed_data() {
        assert_eq!(
            rasterize_codec_motion_vectors([64, 64], MotionLatticeQuality::Live, &[]).unwrap(),
            None
        );
        let future = CodecMotionVector {
            destination: [32, 32],
            block: [16, 16],
            motion: [-2, 0],
            motion_scale: 1,
            seconds_from_reference: 1.0 / 30.0,
            reference: CodecReferenceDirection::Future,
            visibility: 1.0,
        };
        assert_eq!(
            rasterize_codec_motion_vectors([64, 64], MotionLatticeQuality::Live, &[future])
                .unwrap(),
            None
        );
        assert!(matches!(
            rasterize_codec_motion_vectors(
                [64, 64],
                MotionLatticeQuality::Live,
                &[CodecMotionVector {
                    motion_scale: 0,
                    reference: CodecReferenceDirection::Past,
                    ..future
                }]
            ),
            Err(MotionFieldError::InvalidCodecVector { .. })
        ));
    }

    #[test]
    fn codec_vector_direction_and_scale_are_canonicalized() {
        let field = rasterize_codec_motion_vectors(
            [64, 64],
            MotionLatticeQuality::Live,
            &[CodecMotionVector {
                destination: [32, 32],
                block: [16, 16],
                motion: [-4, 0],
                motion_scale: 2,
                seconds_from_reference: 1.0 / 30.0,
                reference: CodecReferenceDirection::Past,
                visibility: 0.5,
            }],
        )
        .unwrap()
        .unwrap();
        let sample = field.sample(4, 4).unwrap();
        let pixels_per_frame = sample.velocity_uv_per_second[0] * 64.0 / 30.0;
        assert!((pixels_per_frame - 2.0).abs() < 0.01);
        assert_eq!(sample.confidence, 1.0);
        assert!((sample.visibility - 0.5).abs() < 0.01);
    }
}
